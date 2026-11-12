"""infer_sw.py
===========
Full-volume sliding-window inference for CoarseLocNet."""

from __future__ import annotations

import math
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ceil_div(a: int, b: int) -> int:
    return max(1, (a + b - 1) // b)


def _align_down(x: int, k: int) -> int:
    return (x // k) * k


def _sliding_window_starts_aligned(
    size: int,
    roi: int,
    stride: int,
    align: int,
) -> List[int]:
    """Window starts aligned to ``align`` (token_stride).

    Last window reaches ``size - roi`` (after padding to token multiple).
    """
    if size <= roi:
        return [0]
    # Rounding a stride smaller than one token to zero would never advance.
    stride = max(align, _align_down(stride, align))
    starts: List[int] = [0]
    while True:
        nxt = starts[-1] + stride
        if nxt + roi >= size:
            break
        starts.append(_align_down(nxt, align))
    last_target = max(0, _ceil_div(size, align) * align - roi)
    if last_target > starts[-1]:
        starts.append(last_target)
    elif last_target < starts[-1]:
        starts.append(starts[-1])
    return starts


# Blend-weight template cache.
_WEIGHT_CACHE: Dict[Tuple, torch.Tensor] = {}


def _get_weight_template(
    roi: Tuple[int, int, int],
    tok_shape: Tuple[int, int, int],
    dtype: torch.dtype,
    device: torch.device,
    mode: str,
) -> Optional[torch.Tensor]:
    """Cached Gaussian importance map downsampled to the token grid."""
    if mode != "gaussian":
        return None
    key = (int(roi[0]), int(roi[1]), int(roi[2]),
           int(tok_shape[0]), int(tok_shape[1]), int(tok_shape[2]),
           dtype, str(device), mode)
    cached = _WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached
    rz, ry, rx = roi
    z = torch.arange(rz, dtype=torch.float32) - (rz - 1) / 2.0
    y = torch.arange(ry, dtype=torch.float32) - (ry - 1) / 2.0
    x = torch.arange(rx, dtype=torch.float32) - (rx - 1) / 2.0
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    sigma_scale = 1.0 / 8.0
    sz = max(1.0, rz * sigma_scale)
    sy = max(1.0, ry * sigma_scale)
    sx = max(1.0, rx * sigma_scale)
    w = torch.exp(-(zz ** 2) / (2 * sz ** 2)
                  - (yy ** 2) / (2 * sy ** 2)
                  - (xx ** 2) / (2 * sx ** 2))
    w = (w / w.max())[None, None]
    tw = F.avg_pool3d(w, kernel_size=(8, 8, 8) if tok_shape[0] else (1, 1, 1))
    # Note: actual token_stride is passed by caller via tok_shape; if the
    # simple 8x8x8 average is not the correct stride, we interpolate below.
    td, th, tw_ = tw.shape[-3:]
    if (td, th, tw_) != tok_shape:
        tw = F.interpolate(tw, size=tok_shape, mode="trilinear", align_corners=False)
    tw = tw.to(dtype=dtype, device=device).contiguous()
    _WEIGHT_CACHE[key] = tw
    return tw


def _pad_volume_to_token(
    image: torch.Tensor, token_stride: Tuple[int, int, int],
    extra_voxels: Tuple[int, int, int] = (0, 0, 0),
) -> torch.Tensor:
    """Pad ``image`` so ``(D, H, W)`` is a multiple of token_stride and
    at least ``extra_voxels`` voxels longer on each spatial axis.

    Padding mode = replicate (matches v2 semantics).
    """
    M, D, H, W = image.shape
    kd, kh, kw = token_stride
    pz = ((kd - (D % kd)) % kd) + extra_voxels[0]
    py = ((kh - (H % kh)) % kh) + extra_voxels[1]
    px = ((kw - (W % kw)) % kw) + extra_voxels[2]
    if pz or py or px:
        image = F.pad(image, (0, px, 0, py, 0, pz), mode="replicate")
    return image


def _pick_amp_dtype(name: str) -> Tuple[bool, Optional[torch.dtype]]:
    name = (name or "fp16").lower()
    if name in ("fp16", "float16", "half"):
        return True, torch.float16
    if name in ("bf16", "bfloat16"):
        return True, torch.bfloat16
    if name in ("fp32", "float32", "off", "none"):
        return False, None
    raise ValueError(f"Unknown amp_dtype={name!r}")


@torch.inference_mode()
def sliding_window_predict(
    model: nn.Module,
    image: torch.Tensor,
    mod_mask: torch.Tensor,
    cfg,
    device: torch.device,
    *,
    amp: bool = True,
    amp_dtype: str = "fp16",
    roi_size: Optional[Tuple[int, int, int]] = None,
    overlap: Optional[float] = None,
    sw_batch_size: int = 4,
    blend_mode: str = "avg",
    return_token_logits: bool = False,
    volume_on_gpu: bool = False,
    max_gpu_volume_gb: float = 4.0,
    pin_memory: bool = True,
    cuda_sync_timing: bool = False,
) -> torch.Tensor:
    """Run sliding-window inference over a full volume.

    See module docstring. The output is a CPU ``[d_tok, h_tok, w_tok]``
    tensor with the *same* values (up to FP rounding) as v2.
    """
    M, D, H, W = image.shape
    kd, kh, kw = cfg.target.token_stride
    if roi_size is None:
        roi_size = cfg.data.sw_roi_size
    if overlap is None:
        overlap = cfg.data.sw_overlap
    rz, ry, rx = int(roi_size[0]), int(roi_size[1]), int(roi_size[2])
    if not 0 <= overlap < 1:
        raise ValueError('Sliding-window overlap must be in [0, 1)')
    if any(k <= 0 or r < k or r % k for r, k in zip((rz, ry, rx), (kd, kh, kw))):
        raise ValueError('Inference ROI must be positive token-stride multiples')

    # CEIL token grid.
    d_tok = _ceil_div(D, kd)
    h_tok = _ceil_div(H, kh)
    w_tok = _ceil_div(W, kw)

    # Window stride in voxels.
    stride_z = max(1, int(rz * (1.0 - overlap)))
    stride_y = max(1, int(ry * (1.0 - overlap)))
    stride_x = max(1, int(rx * (1.0 - overlap)))

    starts_z = _sliding_window_starts_aligned(D, rz, stride_z, kd)
    starts_y = _sliding_window_starts_aligned(H, ry, stride_y, kh)
    starts_x = _sliding_window_starts_aligned(W, rx, stride_x, kw)

    # Pre-compute all (z0, y0, x0) triples (z-major).
    coords: List[Tuple[int, int, int]] = []
    for z0 in starts_z:
        for y0 in starts_y:
            for x0 in starts_x:
                coords.append((z0, y0, x0))
    n_windows = len(coords)
    if n_windows == 0:
        return torch.zeros((d_tok, h_tok, w_tok), dtype=torch.float32, device="cpu")

    # Determine how much extra padding we need so the LAST window in each
    # axis reaches exactly ``D - rz + rz`` (i.e. starts + rz fits in D').
    extra = (
        max(0, starts_z[-1] + rz - (D + ((kd - (D % kd)) % kd))),
        max(0, starts_y[-1] + ry - (H + ((kh - (H % kh)) % kh))),
        max(0, starts_x[-1] + rx - (W + ((kw - (W % kw)) % kw))),
    )
    image = _pad_volume_to_token(image, (kd, kh, kw), extra)
    M2, D2, H2, W2 = image.shape
    del M2  # not used

    # Decide CPU vs GPU staging.
    vol_bytes = M * D2 * H2 * W2 * 4  # float32
    gpu_volume_threshold = int(max_gpu_volume_gb * (1024 ** 3))
    can_keep_on_gpu = bool(volume_on_gpu and device.type == "cuda"
                            and vol_bytes <= gpu_volume_threshold)
    if volume_on_gpu and not can_keep_on_gpu and device.type == "cuda":
        # Caller asked for volume_on_gpu but volume is too large; fall back
        # to streaming CPU path (silent because GPU volume can be a
        # correctness-preserving optimisation, not a semantic change).
        pass

    if can_keep_on_gpu:
        image_dev = image.to(device, non_blocking=True).contiguous()
        cpu_image = None
    else:
        cpu_image = image
        if pin_memory and device.type == "cuda":
            try:
                cpu_image = image.pin_memory()
            except Exception:
                cpu_image = image
        image_dev = None

    # Pre-allocate output accumulators and mod_mask broadcast.
    prob_dtype = torch.float32
    weight_dtype = torch.float32
    prob_sum = torch.zeros((1, 1, d_tok, h_tok, w_tok),
                           device=device, dtype=prob_dtype)
    weight_sum = torch.zeros((1, 1, d_tok, h_tok, w_tok),
                             device=device, dtype=weight_dtype)

    bs = max(1, int(sw_batch_size))
    # Pre-build mod_mask broadcast at max batch size. .expand is a view
    # so this is O(M) memory; clone only if we actually need to write.
    mod_mask_dev = mod_mask.to(device, non_blocking=True)
    mod_mask_template = mod_mask_dev.unsqueeze(0).expand(bs, -1)

    # Cache the weight template (per-call) if needed.
    weight_template = _get_weight_template(
        (rz, ry, rx), (rz // kd, ry // kh, rx // kw),
        prob_dtype, device, blend_mode,
    )

    # AMP setup.
    # CUDA mixed precision is optional; CPU float16 Conv3d can fall back to
    # extremely slow software kernels. Keep portable CPU inference in fp32.
    use_amp, amp_torch_dtype = _pick_amp_dtype(amp_dtype) if amp and device.type == "cuda" else (False, None)
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    # Pre-compute the list of (tz0, ty0, tx0) token starts (cheap).
    token_starts = [(z0 // kd, y0 // kh, x0 // kw) for (z0, y0, x0) in coords]

    # Streaming batches.
    n_done = 0
    while n_done < n_windows:
        chunk = coords[n_done: n_done + bs]
        token_chunk = token_starts[n_done: n_done + bs]
        actual_bs = len(chunk)
        n_done += actual_bs

        # Materialise windows on device.
        if can_keep_on_gpu:
            # Stack views (no copy on GPU).
            wins = torch.stack(
                [image_dev[:, z0:z0 + rz, y0:y0 + ry, x0:x0 + rx]
                 for (z0, y0, x0) in chunk],
                dim=0,
            )
        else:
            cpu_wins = torch.stack(
                [cpu_image[:, z0:z0 + rz, y0:y0 + ry, x0:x0 + rx]
                 for (z0, y0, x0) in chunk],
                dim=0,
            )
            wins = cpu_wins.to(device, non_blocking=True)
            del cpu_wins

        # Forward
        if use_amp:
            with torch.amp.autocast(device_type=autocast_device, dtype=amp_torch_dtype):
                out = model(wins, mod_mask_template[:actual_bs])
        else:
            out = model(wins, mod_mask_template[:actual_bs])

        # Fusion of heads.
        if isinstance(out, dict):
            logits = out.get("crop_hit_logits", out.get("logits"))
            cross_logits = out.get("cross_slice_logits")
            if cross_logits is not None and logits is not None:
                p_hit = torch.sigmoid(logits.float())
                p_cross = torch.sigmoid(cross_logits.float())
                mode = getattr(getattr(cfg, "crop_eval", None), "fusion_mode", "avg")
                if mode == "min":
                    fused = torch.minimum(p_hit, p_cross)
                elif mode == "max":
                    fused = torch.maximum(p_hit, p_cross)
                elif mode == "weighted":
                    alpha = float(getattr(getattr(cfg, "crop_eval", None),
                                          "fusion_alpha", 0.5))
                    fused = alpha * p_hit + (1.0 - alpha) * p_cross
                else:  # avg
                    fused = 0.5 * (p_hit + p_cross)
                logits_f = fused
            else:
                logits_f = (torch.sigmoid(logits.float())
                            if not return_token_logits else logits.float())
        else:
            logits_f = (torch.sigmoid(out.float())
                        if not return_token_logits else out.float())

        # logits_f: (B, 1, td, th, tw)
        B, _, td, th, tw = logits_f.shape
        if (td, th, tw) != (rz // kd, ry // kh, rx // kw):
            raise ValueError('Model output token grid does not match ROI/token stride')

        # ---- vectorised scatter-add accumulation ----
        # Flat index per (b, tz, ty, tx) into the (d_tok*h_tok*w_tok) buffer.
        if B == 1:
            z0, y0, x0 = token_chunk[0]
            tz = torch.arange(td, device=device) + z0
            ty = torch.arange(th, device=device) + y0
            tx = torch.arange(tw, device=device) + x0
            # Discard predictions outside the real token grid. Clamping
            # them would merge padded-tail predictions into edge voxels.
            valid = ((tz[:,None,None] < d_tok) & (ty[None,:,None] < h_tok)
                     & (tx[None,None,:] < w_tok)).reshape(-1)
            flat_idx = (tz[:, None, None] * h_tok * w_tok
                        + ty[None, :, None] * w_tok
                        + tx[None, None, :]).reshape(-1)[valid]
            flat_lz = logits_f[0, 0, :td, :th, :tw].reshape(-1)[valid]
            if weight_template is not None:
                flat_wt = weight_template[0, 0, :td, :th, :tw].reshape(-1)[valid]
            else:
                flat_wt = torch.ones_like(flat_lz)
            prob_sum.view(-1).index_add_(0, flat_idx, flat_lz * flat_wt)
            weight_sum.view(-1).index_add_(0, flat_idx, flat_wt)
        else:
            # Build per-batch token offsets.
            # We compute one big flat index by concat across the batch.
            zs = torch.tensor([c[0] for c in token_chunk], device=device)
            ys = torch.tensor([c[1] for c in token_chunk], device=device)
            xs = torch.tensor([c[2] for c in token_chunk], device=device)
            # offset positions
            oz = zs[:, None, None, None] + torch.arange(td, device=device)[None, :, None, None]
            oy = ys[:, None, None, None] + torch.arange(th, device=device)[None, None, :, None]
            ox = xs[:, None, None, None] + torch.arange(tw, device=device)[None, None, None, :]
            valid = ((oz < d_tok) & (oy < h_tok) & (ox < w_tok)).reshape(-1)
            flat_idx = (
                oz * (h_tok * w_tok)
                + oy * w_tok
                + ox
            ).reshape(-1).to(torch.long)[valid]
            flat_lz = logits_f[:, 0, :td, :th, :tw].reshape(-1)[valid]
            if weight_template is not None:
                # Gaussian weights are relative to each window, independent
                # of its location in the full-volume grid.
                flat_wt = weight_template[0, 0, :td, :th, :tw].reshape(-1).repeat(B)[valid]
            else:
                flat_wt = torch.ones_like(flat_lz)
            prob_sum.view(-1).index_add_(0, flat_idx, flat_lz * flat_wt)
            weight_sum.view(-1).index_add_(0, flat_idx, flat_wt)

        del wins, logits_f, flat_idx, flat_lz
        if flat_wt is not None:
            del flat_wt

    prob = prob_sum / weight_sum.clamp_min(torch.finfo(weight_dtype).tiny)
    if cuda_sync_timing:
        torch.cuda.synchronize(device)
    return prob[0, 0].to("cpu", non_blocking=True)


def estimate_num_windows(
    volume_shape: Tuple[int, int, int],
    roi_size: Tuple[int, int, int],
    overlap: float,
) -> int:
    """Deterministic window-count estimate for cost-aware DDP sharding."""
    D, H, W = volume_shape
    rz, ry, rx = int(roi_size[0]), int(roi_size[1]), int(roi_size[2])
    sz = max(1, int(rz * (1.0 - overlap)))
    sy = max(1, int(ry * (1.0 - overlap)))
    sx = max(1, int(rx * (1.0 - overlap)))
    nz = max(1, (D - rz + sz) // sz) if D > rz else 1
    ny = max(1, (H - ry + sy) // sy) if H > ry else 1
    nx = max(1, (W - rx + sx) // sx) if W > rx else 1
    return int(nz * ny * nx)
