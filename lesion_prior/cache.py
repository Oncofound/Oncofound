"""
cache.py
========
Preprocessed-volume cache.

Saves a serialized tensor pair ``(image, seg, meta)`` per case to
``<cache_dir>/<key>.pt``. The cache key is a hash of the source paths,
spacing, modality order and the orientation/normalization parameters
so that any change to the preprocessing invalidates the cache.

The cache uses a ``fcntl`` flock when available so that concurrent
writers (e.g. multiple DDP ranks or ``tools/build_cache.py`` workers)
do not corrupt the file.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class CacheKey:
    """Bundle of inputs that uniquely identify a preprocessed volume."""

    image_path: str
    seg_path: str
    spacing: Tuple[float, float, float]
    modality_order: Tuple[str, ...]
    orientation_ras: bool
    enforce_spacing: bool
    ct_clip: Tuple[float, float]
    nz_percentile: Tuple[float, float]
    preprocessing_version: str = "canonical_spacing_v1"

    def hash(self) -> str:
        h = hashlib.sha1()
        h.update(self.preprocessing_version.encode("utf-8"))
        h.update(b"|")
        h.update(self.image_path.encode("utf-8"))
        h.update(b"|")
        h.update(self.seg_path.encode("utf-8"))
        h.update(b"|")
        h.update(",".join(f"{s:.4f}" for s in self.spacing).encode("utf-8"))
        h.update(b"|")
        h.update(",".join(self.modality_order).encode("utf-8"))
        h.update(b"|")
        h.update(f"ras={int(self.orientation_ras)},sp={int(self.enforce_spacing)}".encode("utf-8"))
        h.update(b"|")
        h.update(f"clip={self.ct_clip[0]:.2f},{self.ct_clip[1]:.2f}".encode("utf-8"))
        h.update(b"|")
        h.update(f"nz={self.nz_percentile[0]:.3f},{self.nz_percentile[1]:.3f}".encode("utf-8"))
        return h.hexdigest()[:24]


def cache_path(cache_dir: str, key: str) -> str:
    # Two-level subdir to avoid huge flat directories
    sub = key[:2]
    return os.path.join(cache_dir, sub, f"{key}.pt")


def _atomic_save(obj, path: str) -> None:
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_cached(cache_dir: str, key: str, lock: bool = True) -> Optional[dict]:
    """Load a cached preprocessed bundle, or ``None`` if not present."""
    if not cache_dir:
        return None
    p = cache_path(cache_dir, key)
    if not os.path.exists(p):
        return None
    try:
        # Read-only is best for load; flock SH for serialization safety.
        with open(p, "rb") as f:
            if lock:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                except Exception:
                    pass
            obj = torch.load(f, map_location="cpu", weights_only=True)
            if lock:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
        if not isinstance(obj, dict) or any(not torch.is_tensor(obj.get(k)) for k in ('image','seg','mod_mask')):
            return None
        image,seg,mask=obj['image'],obj['seg'],obj['mod_mask']
        if image.ndim != 4 or seg.shape != image.shape[1:] or mask.shape != (image.shape[0],):
            return None
        if not image.numel() or not all(torch.isfinite(t).all() for t in (image,seg,mask)):
            return None
        return obj
    except Exception:
        return None


def save_cached(
    cache_dir: str,
    key: str,
    image: torch.Tensor,
    seg: torch.Tensor,
    mod_mask: torch.Tensor,
    meta: Optional[dict] = None,
) -> str:
    """Save a preprocessed bundle. Uses an exclusive flock."""
    if not cache_dir:
        return ""
    p = cache_path(cache_dir, key)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    payload = {
        "image": image.cpu().contiguous(),
        "seg": seg.cpu().contiguous(),
        "mod_mask": mod_mask.cpu().contiguous(),
        "meta": meta or {},
    }
    # We use a side-car lock file because fcntl.flock does not work
    # over plain torch.save atomicity on all filesystems.
    lock_path = p + ".lock"
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "w") as lf:
        try:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        except Exception:
            pass
        try:
            _atomic_save(payload, p)
        finally:
            try:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
    return p


def maybe_load_or_build_cache(
    cache_dir: str,
    key: str,
    build_fn,
):
    """Load cached bundle, or call ``build_fn`` to (re)build it.

    ``build_fn`` must return a dict ``{"image", "seg", "mod_mask", "meta"}``.
    """
    cached = load_cached(cache_dir, key)
    if cached is not None:
        return cached
    payload = build_fn()
    save_cached(
        cache_dir, key,
        image=payload["image"],
        seg=payload["seg"],
        mod_mask=payload["mod_mask"],
        meta=payload.get("meta"),
    )
    return load_cached(cache_dir, key)
