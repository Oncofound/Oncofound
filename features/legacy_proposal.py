#!/usr/bin/env python3
"""
extract_feat_proposal.py
========================
Independent feature-extraction script (does NOT modify
get_feat_json_new.py). Reads the JSONL we built in task 3, picks
2 GLOBAL crops (random) + 8 LOCAL crops (centres from JSONL
proposal_centers), runs each crop through the 3D DINOv3 backbone
(teacher weights), saves cls-token features to disk.

Output layout (matches 0807 reference):
  {output_dir}/{model_name}_ep{ep}/{dataset}/{case_id}.npy
  with shape [10, embed_dim] (2 global + 8 local).

Designed to be invoked on 8 GPUs in parallel: each GPU processes
its own shard of cases, the script takes --gpu-id and --n-gpus.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# JSONL loading
def load_jsonl(jsonl_path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# Model loading (mirrors get_feat_json_new.py _load_model without
# the grid-crop machinery).
def load_model(repo_dir: str, model_config_file: str,
               weights_file: str, device: torch.device):
    from .extractor import load_backbone
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(model_config_file)
    return load_backbone(weights_file, model_config_file, device=device), cfg


# Crop helpers
def _pad_to_patch_multiple(vol: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Pad (C, H, W, D) so each spatial dim is a multiple of patch_size."""
    ps = int(patch_size)
    if ps <= 0:
        return vol
    C, H, W, D = vol.shape
    ph = (ps - H % ps) % ps
    pw = (ps - W % ps) % ps
    pd = (ps - D % ps) % ps
    if ph == 0 and pw == 0 and pd == 0:
        return vol
    return F.pad(vol, (0, pd, 0, pw, 0, ph), value=0.0)


def _resize_to_cube(vol: torch.Tensor, target_size: int) -> torch.Tensor:
    s = int(target_size)
    return F.interpolate(
        vol.unsqueeze(0),
        size=(s, s, s),
        mode="trilinear",
        align_corners=False,
    ).squeeze(0)


def _centered_crop(vol: torch.Tensor, hwd_size: Tuple[int, int, int],
                   center_hwd: Tuple[int, int, int]) -> torch.Tensor:
    """Extract a crop of (h_size, w_size, d_size) centred at (ch, cw, cd).

    If the crop exceeds the volume, it is zero-padded on the right/
    bottom to the requested size.
    """
    C, H, W, D = vol.shape
    ch, cw, cd = center_hwd
    h_size, w_size, d_size = hwd_size

    h0 = max(0, ch - h_size // 2)
    w0 = max(0, cw - w_size // 2)
    d0 = max(0, cd - d_size // 2)
    h1 = min(H, h0 + h_size)
    w1 = min(W, w0 + w_size)
    d1 = min(D, d0 + d_size)
    block = vol[:, h0:h1, w0:w1, d0:d1]
    pad_h = h_size - block.shape[1]
    pad_w = w_size - block.shape[2]
    pad_d = d_size - block.shape[3]
    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        block = F.pad(block, (0, pad_d, 0, pad_w, 0, pad_h), value=0.0)
    return block


def _random_global_center(rng: np.random.Generator, H: int, W: int, D: int,
                          target_size: int) -> Tuple[int, int, int]:
    """Random centre for a global crop of `target_size`. Stays inside the
    volume so the entire crop lands in real voxels."""
    margin_h = target_size // 2
    margin_w = target_size // 2
    margin_d = target_size // 2
    ch = int(rng.integers(margin_h, max(margin_h + 1, H - margin_h))) if H > 2 * margin_h else H // 2
    cw = int(rng.integers(margin_w, max(margin_w + 1, W - margin_w))) if W > 2 * margin_w else W // 2
    cd = int(rng.integers(margin_d, max(margin_d + 1, D - margin_d))) if D > 2 * margin_d else D // 2
    return ch, cw, cd


def _random_local_centers(rng: np.random.Generator, H: int, W: int, D: int,
                          crop_size: int, n: int = 8) -> List[List[int]]:
    """8 random local crop centres, each kept inside the volume so the
    whole crop_size^3 cube lands in real voxels (with zero-padding at the
    boundary if needed)."""
    m = crop_size // 2
    centers: List[List[int]] = []
    for _ in range(n):
        if H > 2 * m:
            ch = int(rng.integers(m, max(m + 1, H - m)))
        else:
            ch = H // 2
        if W > 2 * m:
            cw = int(rng.integers(m, max(m + 1, W - m)))
        else:
            cw = W // 2
        if D > 2 * m:
            cd = int(rng.integers(m, max(m + 1, D - m)))
        else:
            cd = D // 2
        centers.append([ch, cw, cd])
    return centers


# Forward pass
@torch.no_grad()
def forward_crops(model, crops: torch.Tensor, batch_size: int,
                  device: torch.device) -> np.ndarray:
    """Forward a stack of crops through the backbone, return cls tokens."""
    feats: List[np.ndarray] = []
    for i0 in range(0, crops.shape[0], batch_size):
        x = crops[i0:i0 + batch_size].to(device, non_blocking=True)
        out = model(x, is_training=False)
        if isinstance(out, dict):
            cls_feat = out["x_norm_clstoken"]
        else:
            cls_feat = out
        feats.append(cls_feat.detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


# Per-case processing
def load_and_normalize_image(image_path: str) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """Load a single .npy image and normalise it to (C, H, W, D) with C=1.

    Mirrors get_feat_json_new.py's preprocessing on the volume tensor.
    Returns the volume and the canonical (H, W, D) shape.
    """
    vol = np.load(image_path)
    t = torch.as_tensor(vol).float()
    if t.dim() == 6:
        t = t[0]
    if t.dim() == 5:
        t = t[..., 0] if t.shape[-1] == 3 else t[:, 0, ...]
    if t.dim() == 4:
        # Move channel-last to channel-first.
        t = t.permute(0, 2, 3, 1).contiguous()
    if t.dim() == 3:
        t = t.unsqueeze(0)  # add channel
    C, H, W, D = t.shape
    # If 4D with trailing channel, drop it.
    if C not in (1, 3) and C > 4:
        # Unusual — fall back to first slice.
        t = t[:1]
    # If multi-channel (3) we keep one channel only — backbone has in_chans=1.
    if t.shape[0] > 1:
        t = t[:1]
    return t, (int(H), int(W), int(D))


def process_case(rec: Dict, model, cfg, device: torch.device,
                 batch_size: int, rng_seed_base: int,
                 random_local_no_seg: bool = False) -> np.ndarray:
    """Run 2 global + 8 local crops through the backbone. Return [10, D].

    If random_local_no_seg is True AND rec['has_seg'] is False, ignore the
    JSONL local centres and draw 8 random local centres instead.
    """
    image_path = rec["image_path"]
    canonical_hwd = tuple(rec["canonical_hwd"])  # (H, W, D)
    vol, hwd_actual = load_and_normalize_image(image_path)
    C, H, W, D = vol.shape
    # If the image's actual shape differs from canonical, prefer actual.
    if (H, W, D) != canonical_hwd:
        H, W, D = hwd_actual

    patch_size = int(cfg.student.patch_size)
    global_size = int(cfg.crops.global_crops_size)        # 96
    local_size = int(cfg.crops.local_crops_size)          # 48
    # cls.sh forces crop_size=128, so we override local crop size.
    local_size_override = 128

    # Pad to patch multiple first so all crop arithmetic is consistent.
    vol = _pad_to_patch_multiple(vol, patch_size=patch_size)
    C, H, W, D = vol.shape

    rng = np.random.default_rng(rng_seed_base + int.from_bytes(__import__("hashlib").sha256(str(rec["case_id"]).encode()).digest()[:4], "little"))

    # ---- 2 GLOBAL crops ----
    global_crops: List[torch.Tensor] = []
    for gi in range(2):
        ch, cw, cd = _random_global_center(rng, H, W, D, global_size)
        crop = _centered_crop(vol, (global_size, global_size, global_size), (ch, cw, cd))
        # resize to 128 (cls.sh) so the backbone sees a consistent shape
        crop = _resize_to_cube(crop, target_size=local_size_override)
        # ensure padded to patch multiple
        crop = _pad_to_patch_multiple(crop, patch_size=patch_size)
        global_crops.append(crop)

    # ---- 8 LOCAL crops ----
    use_random_local = bool(random_local_no_seg and not rec.get("has_seg", False))
    if use_random_local:
        local_centers = _random_local_centers(rng, H, W, D,
                                              crop_size=local_size_override, n=8)
    else:
        local_centers = rec.get("proposal_centers", [])[:8]
        while len(local_centers) < 8:
            # fallback: image centre
            local_centers.append([H // 2, W // 2, D // 2])
        local_centers = local_centers[:8]
    local_crops: List[torch.Tensor] = []
    for c in local_centers:
        ch, cw, cd = int(c[0]), int(c[1]), int(c[2])
        # clamp into padded image
        ch = max(0, min(H - 1, ch))
        cw = max(0, min(W - 1, cw))
        cd = max(0, min(D - 1, cd))
        crop = _centered_crop(vol, (local_size_override, local_size_override, local_size_override),
                              (ch, cw, cd))
        crop = _pad_to_patch_multiple(crop, patch_size=patch_size)
        local_crops.append(crop)

    all_crops = torch.stack(global_crops + local_crops, dim=0)  # [10, C, S, S, S]
    feats = forward_crops(model, all_crops, batch_size=batch_size, device=device)
    return feats


# Sharded runner
def shard_for(rank: int, world: int, n: int):
    if world <= 0:
        return list(range(n))
    return [i for i in range(n) if (i % world) == rank]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--weights-file", required=True)
    ap.add_argument("--output-dir", required=True,
                    help="e.g. .../feat_baseline_0809_cls_ada/{model_name}_ep{ep}/{dataset}")
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--n-gpus", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=10,
                    help="forward batch size (10 = all crops at once)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--random-local-no-seg", action="store_true",
                    help="For records with has_seg=False, ignore JSONL centres "
                         "and draw 8 random local centres instead.")
    ap.add_argument("--only-no-seg", action="store_true",
                    help="Skip records that have seg (only process no-seg cases).")
    ap.add_argument("--limit", type=int, default=0,
                    help="if >0, only process first N cases (debug)")
    args = ap.parse_args()

    # When CUDA_VISIBLE_DEVICES is set, the only visible device is cuda:0.
    # Use cuda:0 regardless of args.gpu_id; args.gpu_id is only used for sharding.
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"[INFO] shard_rank={args.gpu_id}/{args.n_gpus} device={device}")

    print(f"[INFO] loading model from {args.weights_file}")
    model, cfg = load_model(args.repo_dir, args.model_config, args.weights_file, device)
    embed_dim = int(getattr(model, "embed_dim", 0)) or int(cfg.student.get("embed_dim", 1024))
    patch_size = int(cfg.student.patch_size)
    print(f"[INFO] model loaded; embed_dim={embed_dim} patch_size={patch_size}")

    rows = load_jsonl(args.jsonl)
    print(f"[INFO] loaded {len(rows)} rows from {args.jsonl}")

    my_indices = shard_for(args.gpu_id, args.n_gpus, len(rows))
    if args.only_no_seg:
        my_indices = [i for i in my_indices if not rows[i].get("has_seg", False)]
    if args.limit > 0:
        my_indices = my_indices[:args.limit]
    print(f"[INFO] rank {args.gpu_id}/{args.n_gpus}: {len(my_indices)} cases"
          f" (random_local_no_seg={args.random_local_no_seg}, only_no_seg={args.only_no_seg})")

    t0 = time.time()
    n_done = 0
    n_err = 0
    for idx in my_indices:
        rec = rows[idx]
        ds = rec["dataset"]
        cid = rec["orig_case_id"]
        save_dir = os.path.join(args.output_dir, ds)
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{cid}.npy")
        if os.path.exists(out_path):
            n_done += 1
            continue
        try:
            feats = process_case(rec, model, cfg, device, args.batch_size, args.seed,
                                 random_local_no_seg=args.random_local_no_seg)
            np.save(out_path, feats, allow_pickle=False)
            n_done += 1
        except Exception as exc:
            n_err += 1
            print(f"[ERR] {ds}/{cid}: {type(exc).__name__}: {exc}")
            continue
        if n_done % 50 == 0:
            print(f"[INFO] done {n_done}/{len(my_indices)} (elapsed {time.time()-t0:.1f}s)")

    print(f"[DONE] rank {args.gpu_id}: {n_done} ok, {n_err} err, elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()