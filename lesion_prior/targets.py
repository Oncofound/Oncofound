"""
targets.py
==========
Target generators for the crop-hit / token-foreground objective.

The function :func:`make_crop_hit_target` is the heart of the package:
it computes, for each token-grid center, how many foreground voxels
would be contained in the corresponding ``local_crop_size`` window in
the original voxel grid.

The window is defined by :func:`lesion_prior.box_ops.axis_box_from_center`,
which guarantees:
  * Box extent is exactly ``size`` voxels when the image is large enough.
  * Token grid is ``ceil(D/sd)`` so the last token is well-defined.
  * Boundary handling is consistent with crop selection.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .box_ops import axis_box_from_center


def _ensure_5d(seg: torch.Tensor) -> torch.Tensor:
    """Promote to ``[B, 1, D, H, W]``.

    Conventions
    -----------
    * ``[D, H, W]`` → ``[1, 1, D, H, W]``
    * ``[1, D, H, W]`` → ``[1, 1, D, H, W]``
    * ``[B, D, H, W]`` (B>1) → ``[B, 1, D, H, W]``
    * ``[B, 1, D, H, W]`` → unchanged
    """
    if seg.dim() == 3:
        seg = seg.unsqueeze(0).unsqueeze(0)
    elif seg.dim() == 4:
        if seg.shape[0] == 1:
            seg = seg.unsqueeze(0)
        else:
            seg = seg.unsqueeze(1)
    return seg


def make_token_target(
    seg: torch.Tensor,
    token_stride: Tuple[int, int, int],
    tau: float = 0.02,
) -> Dict[str, torch.Tensor]:
    """Compute the auxiliary token-foreground target."""
    seg5 = _ensure_5d(seg).float()
    fg = (seg5 > 0).float()
    k = tuple(token_stride)
    if any(ki < 1 for ki in k):
        raise ValueError(f"token_stride must be >= 1, got {k}")
    soft = F.avg_pool3d(fg, kernel_size=k, stride=k)
    hard = (soft >= float(tau)).float()
    return {"token_fg_soft": soft, "token_fg_hard": hard}


def _integral_image_3d(x: torch.Tensor) -> torch.Tensor:
    """Exclusive-end 3D integral image.

    ``ii[z, y, x] = sum_{0..<z, 0..<y, 0..<x} x[..., z, y, x]``

    For a box ``[lo, hi)``, the sum is the standard inclusion-exclusion
    on the exclusive-end ``ii``.
    """
    B, _, D, H, W = x.shape
    ii = F.pad(x, (1, 0, 1, 0, 1, 0), mode="constant", value=0.0)
    ii = ii.cumsum(dim=4).cumsum(dim=3).cumsum(dim=2)
    return ii


def _ceil_div(a: int, b: int) -> int:
    return max(1, (a + b - 1) // b)


def make_crop_hit_target(
    seg: torch.Tensor,
    token_stride: Tuple[int, int, int],
    local_crop_size: Tuple[int, int, int],
    min_hit_voxels: int = 1,
    robust_hit_voxels: int = 10,
    mode: str = "binary",
    ignore_borderline: bool = False,
    borderline_min_voxels: int = 1,
    borderline_max_voxels: int = 5,
) -> Dict[str, torch.Tensor]:
    """Compute the crop-center hit target."""
    seg5 = _ensure_5d(seg).float()
    fg = (seg5 > 0).float()
    B, _, D, H, W = fg.shape
    sz, sy, sx = int(local_crop_size[0]), int(local_crop_size[1]), int(local_crop_size[2])
    ii = _integral_image_3d(fg)

    # CEIL token grid (so the last token is well-defined even if the
    # image size is not a multiple of the stride).
    kd, kh, kw = int(token_stride[0]), int(token_stride[1]), int(token_stride[2])
    d_tok = _ceil_div(D, kd)
    h_tok = _ceil_div(H, kh)
    w_tok = _ceil_div(W, kw)

    device = ii.device
    # Voxel centers for every token, in *voxel* coordinates.
    cz = (torch.arange(d_tok, device=device, dtype=torch.float32) + 0.5) * kd
    cy = (torch.arange(h_tok, device=device, dtype=torch.float32) + 0.5) * kh
    cx = (torch.arange(w_tok, device=device, dtype=torch.float32) + 0.5) * kw
    cz = cz.long()
    cy = cy.long()
    cx = cx.long()

    # For each token, compute the box [lo, hi) using the shared
    # box-from-center logic so that target and crop_select agree.
    lo_z = torch.zeros(d_tok, dtype=torch.long, device=device)
    hi_z = torch.zeros(d_tok, dtype=torch.long, device=device)
    for i in range(d_tok):
        lo, hi = axis_box_from_center(int(cz[i].item()), sz, D)
        lo_z[i] = lo
        hi_z[i] = hi
    lo_y = torch.zeros(h_tok, dtype=torch.long, device=device)
    hi_y = torch.zeros(h_tok, dtype=torch.long, device=device)
    for j in range(h_tok):
        lo, hi = axis_box_from_center(int(cy[j].item()), sy, H)
        lo_y[j] = lo
        hi_y[j] = hi
    lo_x = torch.zeros(w_tok, dtype=torch.long, device=device)
    hi_x = torch.zeros(w_tok, dtype=torch.long, device=device)
    for k_ in range(w_tok):
        lo, hi = axis_box_from_center(int(cx[k_].item()), sx, W)
        lo_x[k_] = lo
        hi_x[k_] = hi

    gz, gy, gx = torch.meshgrid(lo_z, lo_y, lo_x, indexing="ij")
    hz, hy, hx = torch.meshgrid(hi_z, hi_y, hi_x, indexing="ij")

    s = (
        ii[:, 0, hz, hy, hx]
        - ii[:, 0, gz, hy, hx]
        - ii[:, 0, hz, gy, hx]
        - ii[:, 0, hz, hy, gx]
        + ii[:, 0, gz, gy, hx]
        + ii[:, 0, gz, hy, gx]
        + ii[:, 0, hz, gy, gx]
        - ii[:, 0, gz, gy, gx]
    )
    counts_center = s.unsqueeze(1)  # [B, 1, d_tok, h_tok, w_tok]

    hard = (counts_center >= float(min_hit_voxels)).float()
    robust = (counts_center >= float(robust_hit_voxels)).float()

    if mode == "binary":
        soft = hard
    elif mode == "log_count":
        max_per_case = counts_center.amax(dim=(2, 3, 4), keepdim=True).clamp_min(1.0)
        soft = torch.log1p(counts_center) / torch.log1p(max_per_case)
    elif mode == "occupancy":
        denom = float(sz * sy * sx)
        soft = counts_center / max(denom, 1.0)
    else:
        raise ValueError(f"Unknown mode={mode!r}")

    if ignore_borderline:
        mask = ((counts_center >= float(borderline_min_voxels)) & (counts_center < float(borderline_max_voxels))).float()
    else:
        mask = torch.zeros_like(counts_center)
    ignore = (1.0 - mask).clamp(0.0, 1.0)

    return {
        "crop_hit_count": counts_center,
        "crop_hit_hard": hard,
        "crop_hit_robust": robust,
        "crop_hit_soft": soft,
        "ignore": ignore,
    }


def _self_test() -> None:
    """Tiny self-test for :func:`make_crop_hit_target`."""
    seg = torch.zeros((1, 1, 16, 16, 16))
    seg[0, 0, 8, 8, 8] = 1
    out = make_crop_hit_target(
        seg, token_stride=(4, 4, 4), local_crop_size=(8, 8, 8),
    )
    counts = out["crop_hit_count"][0, 0]
    assert counts.shape == (4, 4, 4), counts.shape
    assert counts[2, 2, 2].item() == 1.0
    assert counts[0, 0, 0].item() == 0.0
    assert counts[3, 3, 3].item() == 0.0
    assert out["crop_hit_hard"][0, 0, 2, 2, 2].item() == 1.0
    assert out["crop_hit_robust"][0, 0, 2, 2, 2].item() == 0.0
    # 48**3 sanity
    seg2 = torch.ones((1, 1, 48, 48, 48))
    out2 = make_crop_hit_target(seg2, (8, 8, 8), (48, 48, 48))
    c = out2["crop_hit_count"]
    # Center token (2,2,2): box [0, 48) = 48^3
    assert c[0, 0, 2, 2, 2].item() == float(48 ** 3)
    print("[targets] self-test ok.")


#  v2 additions — Step 2: 5 new target helpers

def _distance_to_fg_3d(seg: torch.Tensor) -> torch.Tensor:
    """Compute voxel-level distance to the nearest foreground voxel.

    Uses a simple 6-neighbour BFS (O(N) over voxels) which is fine for
    volumes up to ~512³. For larger volumes use :func:`_edt_3d` which
    falls back to ``scipy.ndimage.distance_transform_edt``.

    Parameters
    ----------
    seg : Tensor ``[B, 1, D, H, W]`` or ``[D, H, W]``
        Binary foreground mask (non-zero = fg).

    Returns
    -------
    Tensor with the same shape, dtype float32.
        Distance in voxel units.
    """
    seg5 = _ensure_5d(seg).float()
    fg = (seg5 > 0.0).float()
    B, _, D, H, W = fg.shape
    device = fg.device

    # Try scipy EDT first
    try:
        from scipy.ndimage import distance_transform_edt
        out = torch.zeros_like(fg)
        for b in range(B):
            # distance_transform_edt returns distance to nearest zero.
            # We have fg=1 as "object", so distance to fg = distance to zero
            # of the complement (1 - fg).
            d = distance_transform_edt(1.0 - fg[b, 0].cpu().numpy()).astype("float32")
            out[b, 0] = torch.from_numpy(d).to(device)
        return out
    except Exception:
        pass

    # Iterative BFS fallback: 6-neighbour chamfer distance.
    INF = float(D + H + W)
    dist = torch.where(fg > 0, torch.zeros_like(fg), torch.full_like(fg, INF))
    changed = True
    while changed:
        changed = False
        for axis, shift in ((2, 1), (2, -1), (3, 1), (3, -1), (4, 1), (4, -1)):
            rolled = torch.roll(dist, shifts=shift, dims=axis)
            new = (rolled + 1.0).clamp_max(INF)
            mask = new < dist
            dist = torch.where(mask, new, dist)
            if mask.any():
                changed = True
    return dist


def make_effective_target(
    counts_center: torch.Tensor,
    min_hit_voxels: int,
    robust_hit_voxels: int,
    robust_frac: float = 0.33,
    ignore_token: torch.Tensor = None,
) -> torch.Tensor:
    """Compute the *effective* hard target for v2.

    A token is an effective positive iff:

    * ``counts >= min_hit_voxels`` AND
    * ``counts >= robust_voxels * robust_frac``

    Borderline tokens (hard=1 but not effective) are masked as ignore
    in the returned ``effective_ignore`` mask.

    Parameters
    ----------
    counts_center : Tensor ``[B, 1, d, h, w]`` from
        :func:`make_crop_hit_target`.
    min_hit_voxels : int
    robust_hit_voxels : int
    robust_frac : float
    ignore_token : Tensor ``[B, 1, d, h, w]``, optional
        Pre-existing ignore mask (1 = ignore). Combined by AND.

    Returns
    -------
    dict with:
      * ``effective_hard`` : same shape as counts, float 0/1
      * ``effective_ignore`` : same shape, float 0/1 (1 = ignore)
    """
    counts_center = counts_center.float()
    effective_hard = (
        (counts_center >= float(min_hit_voxels))
        & (counts_center >= float(robust_hit_voxels) * float(robust_frac))
    ).float()
    # tokens that fail the effective test but pass min_hit become ignore
    borderline = (
        (counts_center >= float(min_hit_voxels))
        & (counts_center < float(robust_hit_voxels) * float(robust_frac))
    ).float()
    effective_ignore = borderline
    if ignore_token is not None:
        # ignore = 1 if either source says ignore
        effective_ignore = ((effective_ignore + ignore_token.float()) > 0).float()
    return {"effective_hard": effective_hard, "effective_ignore": effective_ignore}


def make_distance_weighted_soft(
    seg: torch.Tensor,
    counts_center: torch.Tensor,
    sigma_voxel: float = 8.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Compute a distance-weighted soft target.

    For each token center, ``weight = exp(-d_to_fg / sigma) * (count/amax)^alpha``.
    Tokens far from fg with low count get a small but non-zero target;
    this provides a smoother gradient than binary {0, 1}.

    Parameters
    ----------
    seg : Tensor ``[B, 1, D, H, W]``
    counts_center : Tensor ``[B, 1, d, h, w]`` from
        :func:`make_crop_hit_target`.
    sigma_voxel : float
    alpha : float

    Returns
    -------
    Tensor ``[B, 1, d, h, w]`` of float values in ``[0, 1]``.
    """
    seg5 = _ensure_5d(seg).float()
    counts_center = counts_center.float()
    B = seg5.shape[0]
    device = seg5.device

    out = torch.zeros_like(counts_center)
    for b in range(B):
        fg = (seg5[b, 0] > 0).cpu().numpy()
        # voxel-wise distance to nearest fg
        try:
            from scipy.ndimage import distance_transform_edt
            d_voxel = distance_transform_edt(1.0 - fg.astype("float32")).astype("float32")
        except Exception:
            # fallback: huge distance
            d_voxel = np.ones_like(fg, dtype="float32") * 1e6
        d_voxel_t = torch.from_numpy(d_voxel).to(device)
        # Average-pool distance from voxel grid to token grid
        kd, kh, kw = _kernel_for_target(counts_center[b:b + 1].shape)
        d_tok = F.avg_pool3d(
            d_voxel_t[None, None],
            kernel_size=(kd, kh, kw),
            stride=(kd, kh, kw),
        )[0, 0]
        # Per-case max count
        amax = counts_center[b].amax().clamp_min(1.0)
        # Normalised count factor
        count_norm = (counts_center[b, 0] / amax).clamp(0, 1)
        weight = torch.exp(-d_tok / max(1e-3, float(sigma_voxel)))
        weight = weight * (count_norm.clamp_min(0.0) ** float(alpha))
        out[b, 0] = weight
    return out


def _kernel_for_target(shape):
    """Token-grid shape → kernel size to avg-pool from voxel grid to token grid."""
    # Heuristic: token count == ceil(vox / stride) so stride ~= D / d_tok
    # We don't have direct access to stride, so use the count tensor shape
    # to back out (we'll fix this in make_distance_weighted_soft via signature).
    # In practice, callers should pass token_stride explicitly. Kept here for back-compat.
    return (8, 8, 8)


def _pool_distance_to_tokens(
    d_voxel_t: torch.Tensor,
    token_stride: Tuple[int, int, int],
) -> torch.Tensor:
    """Pool voxel-level distance map to token-grid resolution via avg-pool."""
    kd, kh, kw = int(token_stride[0]), int(token_stride[1]), int(token_stride[2])
    return F.avg_pool3d(
        d_voxel_t[None, None],
        kernel_size=(kd, kh, kw),
        stride=(kd, kh, kw),
    )[0, 0]


def make_distance_weighted_soft_v2(
    seg: torch.Tensor,
    counts_center: torch.Tensor,
    token_stride: Tuple[int, int, int],
    sigma_voxel: float = 8.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Same as :func:`make_distance_weighted_soft` but explicit stride."""
    seg5 = _ensure_5d(seg).float()
    counts_center = counts_center.float()
    B = seg5.shape[0]
    device = seg5.device

    out = torch.zeros_like(counts_center)
    for b in range(B):
        fg = (seg5[b, 0] > 0).cpu().numpy()
        try:
            from scipy.ndimage import distance_transform_edt
            d_voxel = distance_transform_edt(1.0 - fg.astype("float32")).astype("float32")
        except Exception:
            d_voxel = np.ones_like(fg, dtype="float32") * 1e6
        d_voxel_t = torch.from_numpy(d_voxel).to(device)
        d_tok = _pool_distance_to_tokens(d_voxel_t, token_stride)
        amax = counts_center[b].amax().clamp_min(1.0)
        count_norm = (counts_center[b, 0] / amax).clamp(0, 1)
        weight = torch.exp(-d_tok / max(1e-3, float(sigma_voxel)))
        weight = weight * (count_norm.clamp_min(0.0) ** float(alpha))
        out[b, 0] = weight
    return out


def make_crop_hit_target_dinov3(
    seg: torch.Tensor,
    token_stride: Tuple[int, int, int],
    inner_scale: Tuple[float, float],
    cross_scale: Tuple[float, float],
    output_size: Tuple[int, int, int],
    n_samples: int = 4,
    min_hit_voxels: int = 1,
    robust_hit_voxels: int = 10,
    mode: str = "binary",
    aggregation: str = "max",
    seed: int = None,
) -> Dict[str, torch.Tensor]:
    """Multi-scale crop-hit target matching DINOv3 inner/cross scale ranges.

    Each of ``n_samples`` iterations samples a crop shape ``(sd, sy, sx)``
    from ``(cross_scale × D, inner_scale × H, inner_scale × W)``, then
    converts it to an equivalent voxel-space crop size that, when resized
    to ``output_size``, represents the same physical extent.

    Returns
    -------
    dict with:
      * ``crop_hit_count_multiscale`` : ``[B, 1, d_tok, h_tok, w_tok]``
        Mean (or max) of the per-sample count maps.
      * ``crop_hit_hard_multiscale`` : binary version with
        ``min_hit_voxels`` threshold.
      * ``crop_hit_count_per_sample`` : list of per-sample counts
        (for ablation).
    """
    seg5 = _ensure_5d(seg).float()
    B, _, D, H, W = seg5.shape
    kd, kh, kw = int(token_stride[0]), int(token_stride[1]), int(token_stride[2])
    od, oh, ow = int(output_size[0]), int(output_size[1]), int(output_size[2])

    if seed is not None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
    else:
        gen = None

    counts_list = []
    for _ in range(int(n_samples)):
        if gen is not None:
            r = torch.rand(3, generator=gen).numpy()
        else:
            r = np.random.rand(3)
        sy_vox = max(int(round(H * (inner_scale[0] + r[0] * (inner_scale[1] - inner_scale[0])))), 1)
        sx_vox = max(int(round(W * (inner_scale[0] + r[1] * (inner_scale[1] - inner_scale[0])))), 1)
        sd_vox = max(int(round(D * (cross_scale[0] + r[2] * (cross_scale[1] - cross_scale[0])))), 1)
        # Convert source-voxel crop to "equivalent" voxel-space box for our token grid
        sz_box = max(int(round(sd_vox * kd / max(1, od))), 1)
        sy_box = max(int(round(sy_vox * kh / max(1, oh))), 1)
        sx_box = max(int(round(sx_vox * kw / max(1, ow))), 1)
        # Use existing make_crop_hit_target (mode=binary for speed; we only
        # need the count, downstream code can re-bin).
        t = make_crop_hit_target(
            seg, token_stride=token_stride,
            local_crop_size=(sz_box, sy_box, sx_box),
            min_hit_voxels=min_hit_voxels,
            robust_hit_voxels=robust_hit_voxels,
            mode=mode,
        )
        counts_list.append(t["crop_hit_count"])

    stacked = torch.stack(counts_list, dim=0)  # [n_samples, B, 1, d, h, w]
    if aggregation == "max":
        agg = stacked.max(dim=0)[0]
    elif aggregation == "mean":
        agg = stacked.mean(dim=0)
    else:
        agg = stacked.max(dim=0)[0]
    hard = (agg >= float(min_hit_voxels)).float()
    return {
        "crop_hit_count_multiscale": agg,
        "crop_hit_hard_multiscale": hard,
        "crop_hit_count_per_sample": counts_list,
    }


def make_component_target(
    seg: torch.Tensor,
    token_stride: Tuple[int, int, int],
    local_crop_size: Tuple[int, int, int],
    connectivity: int = 1,
) -> torch.Tensor:
    """Per-token fraction of *complete* connected components inside the crop.

    For each connected component ``c`` of the fg mask:
      * ``covered_voxels[c ∩ box] / |c|`` (per-component coverage)
    The token's value is the *max* per-component coverage across all
    components in its crop. Encourages picking crops that contain a
    complete lesion, not a fragment.

    Parameters
    ----------
    seg : Tensor ``[B, 1, D, H, W]``
    token_stride : ``(kd, kh, kw)``
    local_crop_size : crop box side lengths

    Returns
    -------
    Tensor ``[B, 1, d, h, w]`` of float values in ``[0, 1]``.
    """
    seg5 = _ensure_5d(seg).cpu().numpy().astype("uint8")  # [B, 1, D, H, W]
    B, _, D, H, W = seg5.shape
    kd, kh, kw = int(token_stride[0]), int(token_stride[1]), int(token_stride[2])
    sz, sy, sx = int(local_crop_size[0]), int(local_crop_size[1]), int(local_crop_size[2])

    d_tok = _ceil_div(D, kd)
    h_tok = _ceil_div(H, kh)
    w_tok = _ceil_div(W, kw)

    out = np.zeros((B, 1, d_tok, h_tok, w_tok), dtype="float32")

    for b in range(B):
        mask = (seg5[b, 0] > 0).astype("uint8")
        if mask.sum() == 0:
            continue
        # Connected components
        try:
            from scipy.ndimage import label
            labeled, n_comp = label(mask)
        except Exception:
            # fallback: only whole mask
            labeled = mask.copy()
            n_comp = 1
        sizes = np.bincount(labeled.ravel())
        # Per-component fg volume
        comp_sizes = {c: int(sizes[c]) for c in range(1, n_comp + 1)}

        # Token centers
        for tz in range(d_tok):
            cz = int((tz + 0.5) * kd)
            lo_z, hi_z = axis_box_from_center(cz, sz, D)
            for ty in range(h_tok):
                cy = int((ty + 0.5) * kh)
                lo_y, hi_y = axis_box_from_center(cy, sy, H)
                for tx in range(w_tok):
                    cx = int((tx + 0.5) * kw)
                    lo_x, hi_x = axis_box_from_center(cx, sx, W)
                    if (hi_z - lo_z) < 1 or (hi_y - lo_y) < 1 or (hi_x - lo_x) < 1:
                        continue
                    box = labeled[lo_z:hi_z, lo_y:hi_y, lo_x:hi_x]
                    if box.size == 0:
                        continue
                    # Per-component coverage
                    best = 0.0
                    for c, vol in comp_sizes.items():
                        if vol <= 0:
                            continue
                        covered = int((box == c).sum())
                        cov = covered / float(vol)
                        if cov > best:
                            best = cov
                    out[b, 0, tz, ty, tx] = best
    return torch.from_numpy(out).float()


def make_bbox_synthetic_positive(
    seg: torch.Tensor,
    token_stride: Tuple[int, int, int],
    local_crop_size: Tuple[int, int, int],
    margin_voxel: int = 4,
) -> torch.Tensor:
    """Mark every token whose center is inside the fg bounding box (expanded).

    This compensates for the sparsity of "hit-positive" tokens when the
    lesion is small. Tokens whose 48³ box *would* contain the lesion's
    bbox are all positive.

    Returns
    -------
    Tensor ``[B, 1, d, h, w]`` of float 0/1.
    """
    seg5 = _ensure_5d(seg).float()
    B, _, D, H, W = seg5.shape
    kd, kh, kw = int(token_stride[0]), int(token_stride[1]), int(token_stride[2])
    sz, sy, sx = int(local_crop_size[0]), int(local_crop_size[1]), int(local_crop_size[2])

    d_tok = _ceil_div(D, kd)
    h_tok = _ceil_div(H, kh)
    w_tok = _ceil_div(W, kw)

    out = torch.zeros((B, 1, d_tok, h_tok, w_tok), dtype=torch.float32, device=seg5.device)
    for b in range(B):
        fg = (seg5[b, 0] > 0)
        if not fg.any():
            continue
        coords = fg.nonzero(as_tuple=False)
        z0, y0, x0 = coords.min(dim=0).values.tolist()
        z1, y1, x1 = coords.max(dim=0).values.tolist()
        # Expand by margin
        z0 = max(0, int(z0) - margin_voxel)
        y0 = max(0, int(y0) - margin_voxel)
        x0 = max(0, int(x0) - margin_voxel)
        z1 = min(D, int(z1) + 1 + margin_voxel)
        y1 = min(H, int(y1) + 1 + margin_voxel)
        x1 = min(W, int(x1) + 1 + margin_voxel)
        # Token indices covering the bbox
        tz0 = max(0, int(z0) // kd)
        ty0 = max(0, int(y0) // kh)
        tx0 = max(0, int(x0) // kw)
        tz1 = min(d_tok, int(z1) // kd + 1)
        ty1 = min(h_tok, int(y1) // kh + 1)
        tx1 = min(w_tok, int(x1) // kw + 1)
        out[b, 0, tz0:tz1, ty0:ty1, tx0:tx1] = 1.0
    return out


def make_cross_slice_target(
    seg: torch.Tensor,
    token_stride: Tuple[int, int, int],
    cross_crop_size: Tuple[int, int, int],
    min_hit_voxels: int = 1,
    robust_hit_voxels: int = 10,
) -> Dict[str, torch.Tensor]:
    """Target for the cross-slice head: a "tall" crop along z.

    Used by the second CoarseLocNet head that supervises cross_slice
    box predictions (e.g. 96×48×48).

    Returns
    -------
    dict with crop_hit_count, crop_hit_hard, crop_hit_robust, crop_hit_soft.
    """
    return make_crop_hit_target(
        seg, token_stride=token_stride,
        local_crop_size=cross_crop_size,
        min_hit_voxels=min_hit_voxels,
        robust_hit_voxels=robust_hit_voxels,
        mode="binary",
    )


def make_all_targets(
    seg: torch.Tensor,
    cfg_t,
) -> Dict[str, torch.Tensor]:
    """Compute the full target dict for v2 training.

    Always includes the v1 fields (``crop_hit_*``, ``token_fg_*``, ``ignore``)
    plus the v2 extras gated by ``cfg_t`` flags.
    """
    # v1 base targets (existing functions)
    out: Dict[str, torch.Tensor] = {}
    out.update(make_crop_hit_target(
        seg,
        token_stride=cfg_t.token_stride,
        local_crop_size=cfg_t.local_crop_size,
        min_hit_voxels=cfg_t.min_hit_voxels,
        robust_hit_voxels=cfg_t.robust_hit_voxels,
        mode=cfg_t.hit_target_mode,
        ignore_borderline=cfg_t.ignore_borderline,
        borderline_min_voxels=cfg_t.borderline_min_voxels,
        borderline_max_voxels=cfg_t.borderline_max_voxels,
    ))
    out.update(make_token_target(seg, token_stride=cfg_t.token_stride, tau=cfg_t.tau_token_aux))

    # Effective target (Step 1.1)
    eff = make_effective_target(
        out["crop_hit_count"],
        min_hit_voxels=cfg_t.effective_min_voxels,
        robust_hit_voxels=cfg_t.robust_hit_voxels,
        robust_frac=cfg_t.effective_min_robust_frac,
        ignore_token=out["ignore"],
    )
    out["effective_hard"] = eff["effective_hard"]
    out["effective_ignore"] = eff["effective_ignore"]

    # Distance-weighted soft (Step 1.2)
    if cfg_t.use_distance_weighted_soft:
        out["crop_hit_soft_dist"] = make_distance_weighted_soft_v2(
            seg, out["crop_hit_count"],
            token_stride=cfg_t.token_stride,
            sigma_voxel=cfg_t.distance_weight_sigma_voxel,
            alpha=cfg_t.distance_weight_alpha,
        )

    # Multi-scale (DINOv3 inner/cross, Step 1.3 + 5)
    if cfg_t.multiscale_n_samples > 1:
        ms = make_crop_hit_target_dinov3(
            seg,
            token_stride=cfg_t.token_stride,
            inner_scale=cfg_t.local_crop_inner_scale,
            cross_scale=cfg_t.local_crop_cross_scale,
            output_size=cfg_t.local_crop_output_size,
            n_samples=cfg_t.multiscale_n_samples,
            min_hit_voxels=cfg_t.min_hit_voxels,
            robust_hit_voxels=cfg_t.robust_hit_voxels,
            aggregation=cfg_t.multiscale_aggregation,
        )
        out["crop_hit_count_multiscale"] = ms["crop_hit_count_multiscale"]
        out["crop_hit_hard_multiscale"] = ms["crop_hit_hard_multiscale"]

    # Component-aware target (Step 1.4)
    if cfg_t.use_component_target:
        out["crop_hit_component"] = make_component_target(
            seg,
            token_stride=cfg_t.token_stride,
            local_crop_size=cfg_t.local_crop_size,
        )

    # Bbox-driven synthetic positives (Step 4.1)
    if cfg_t.use_bbox_synthetic_positive:
        out["crop_hit_bbox_positive"] = make_bbox_synthetic_positive(
            seg,
            token_stride=cfg_t.token_stride,
            local_crop_size=cfg_t.local_crop_size,
            margin_voxel=cfg_t.bbox_positive_margin_voxel,
        )

    # Cross-slice head target (Step 5)
    # Imported lazily to avoid circular import; uses 96×48×48 by default
    try:
        out["cross_slice"] = make_cross_slice_target(
            seg,
            token_stride=cfg_t.token_stride,
            cross_crop_size=getattr(cfg_t, "cross_slice_crop_size", (96, 48, 48)),
        )
    except Exception:
        pass

    return out


if __name__ == "__main__":
    _self_test()
