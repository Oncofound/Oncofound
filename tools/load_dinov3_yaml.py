"""
load_dinov3_yaml.py
===================
Read crop scales from a DINOv3 training yaml and apply them to a
CoarseLocNet Config. This keeps CoarseLocNet's targets in sync with
DINOv3's actual local crop distribution.

Usage (CLI):
    python -m tools.load_dinov3_yaml \
        --dinov3_yaml ./configs/pretrain.yaml \
        --out work_dir/dinov3_overrides.yaml

Usage (programmatic):
    from tools.load_dinov3_yaml import apply_dinov3_scales_to_config
    cfg = apply_dinov3_scales_to_config(cfg, dinov3_yaml_path)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Optional

# Allow running as a script from any cwd
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _load_yaml(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        raise RuntimeError("PyYAML is required to parse DINOv3 configs.")
    with open(path) as f:
        return yaml.safe_load(f)


def load_dinov3_crop_scales(yaml_path: str) -> Dict[str, Any]:
    """Extract crop-scale info from a DINOv3 training yaml.

    Returns a dict with keys:
      inner_scale  : (lo, hi) tuple
      cross_scale  : (lo, hi) tuple
      output_size  : (d, h, w) tuple
      n_local_crops: int
    """
    cfg = _load_yaml(yaml_path)
    crops = cfg.get("crops", {}) or {}
    inner = crops.get("local_crops_inner_slice_scale") or [0.16, 0.48]
    cross = crops.get("local_crops_cross_slice_scale") or [0.2, 0.5]
    out_sz = int(crops.get("local_crops_size", 48))
    n_local = int(crops.get("local_crops_number", 8))
    return {
        "inner_scale": (float(inner[0]), float(inner[1])),
        "cross_scale": (float(cross[0]), float(cross[1])),
        "output_size": (int(out_sz), int(out_sz), int(out_sz)),
        "n_local_crops": int(n_local),
    }


def apply_dinov3_scales_to_config(cfg, dinov3_yaml_path: str):
    """Mutate ``cfg`` (a CoarseLocNet ``Config``) in-place with DINOv3 scales."""
    scales = load_dinov3_crop_scales(dinov3_yaml_path)
    cfg.target.local_crop_inner_scale = scales["inner_scale"]
    cfg.target.local_crop_cross_scale = scales["cross_scale"]
    cfg.target.local_crop_output_size = scales["output_size"]
    cfg.target.multiscale_n_samples = max(1, cfg.target.multiscale_n_samples)
    cfg.loss.topk_k = scales["n_local_crops"]
    # The cross-slice head box size in CoarseLocNet's voxel space: choose
    # the maximum extent in z, which corresponds to cross_scale[1] * D.
    # We approximate by 2x the in-plane size (a reasonable default).
    cz = int(scales["output_size"][0] * 2)
    cy = int(scales["output_size"][1])
    cx = int(scales["output_size"][2])
    cfg.model.cross_slice_crop_size = (cz, cy, cx)
    # Multi-scale target aggregation: take the max so any single scale
    # catching the lesion counts.
    cfg.target.multiscale_aggregation = "max"
    # If crop_setting implies thin slabs, enable cross-slice head by default
    if scales["cross_scale"][1] - scales["cross_scale"][0] > 0.1:
        cfg.model.use_cross_slice_head = True
        cfg.loss.lambda_cross_slice = max(cfg.loss.lambda_cross_slice, 0.5)
    return cfg


def write_dinov3_target_yaml(dinov3_yaml: str, out_yaml: str) -> str:
    """Write a CoarseLocNet-style overrides yaml that mirrors DINOv3 scales."""
    scales = load_dinov3_crop_scales(dinov3_yaml)
    payload = {
        "target": {
            "local_crop_inner_scale": list(scales["inner_scale"]),
            "local_crop_cross_scale": list(scales["cross_scale"]),
            "local_crop_output_size": list(scales["output_size"]),
            "multiscale_n_samples": 4,
            "multiscale_aggregation": "max",
        },
        "model": {
            "cross_slice_crop_size": [
                int(scales["output_size"][0] * 2),
                int(scales["output_size"][1]),
                int(scales["output_size"][2]),
            ],
            "use_cross_slice_head": True,
        },
        "loss": {
            "topk_k": int(scales["n_local_crops"]),
            "lambda_cross_slice": 0.5,
        },
    }
    import yaml
    os.makedirs(os.path.dirname(os.path.abspath(out_yaml)) or ".", exist_ok=True)
    with open(out_yaml, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return out_yaml


def main() -> None:
    p = argparse.ArgumentParser(description="Apply DINOv3 crop scales to CoarseLocNet config.")
    p.add_argument("--dinov3_yaml", required=True, help="Path to DINOv3 train yaml")
    p.add_argument("--out", required=True, help="Path to write overrides yaml")
    args = p.parse_args()
    out = write_dinov3_target_yaml(args.dinov3_yaml, args.out)
    print(f"[load_dinov3_yaml] wrote overrides to {out}")
    with open(out) as f:
        print(f.read())


if __name__ == "__main__":
    main()