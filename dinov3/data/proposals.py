"""dinov3.data.proposals
=======================

Sidecar proposal lookup used by the unlabeled data loader.

The proposals are produced offline by
``processed_data/build_unlabeled_proposal_points.py`` and live in a
JSONL file on disk. This module loads them once into a read-only
mapping keyed by both the absolute image path and a portable
``relative_image_key`` (``V3/...``), so the same JSONL works across
different data-root relocations.

The lookups happen **outside** the ``PersistentDataset`` cache boundary,
which means changing the JSONL does **not** invalidate any cached
image volume on disk.

Only *centres* (not crop boxes or final local crops) are stored.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("dinov3")


def normalize_path(path: str) -> str:
    """Return an absolute, normalised version of ``path``."""
    if path is None:
        return ""
    return os.path.normpath(os.path.abspath(path))


def make_relative_image_key(path: str) -> str:
    """Return the portable ``V3/...`` key for ``path``.

    Falls back to the normalised absolute path if no ``V3/`` marker
    exists.
    """
    norm = normalize_path(path)
    if not norm:
        return ""
    for marker in ("/V3/", "\\V3\\"):
        idx = norm.rfind(marker)
        if idx >= 0:
            return norm[idx + len(marker):].replace("\\", "/")
    low = norm.lower()
    for marker in ("/v3/", "\\v3\\"):
        idx = low.rfind(marker)
        if idx >= 0:
            return norm[idx + len(marker):].replace("\\", "/")
    return norm


def _validate_entry(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate one JSONL entry. Returns ``None`` on any error.

    Successful returns always expose ``centers`` as ``np.ndarray`` of
    shape ``[N, 3]`` and ``scores`` as ``np.ndarray`` of shape ``[N]``.
    """
    if not isinstance(entry, dict):
        return None
    raw_centers = entry.get("proposal_centers")
    raw_scores = entry.get("proposal_scores")
    if raw_centers is None or raw_scores is None:
        return None
    try:
        centers = np.asarray(raw_centers, dtype=np.float32)
        scores = np.asarray(raw_scores, dtype=np.float32)
    except Exception:
        return None
    if centers.ndim != 2 or centers.shape[1] != 3:
        return None
    if scores.ndim != 1 or scores.shape[0] != centers.shape[0]:
        return None
    if not np.isfinite(centers).all() or not np.isfinite(scores).all():
        return None
    raw_shape = entry.get("raw_shape")
    if raw_shape is None or len(raw_shape) != 3:
        return None
    try:
        raw_shape_t = tuple(int(v) for v in raw_shape)
    except Exception:
        return None
    return {
        "case_id": str(entry.get("case_id", "")),
        "image_path": str(entry.get("image_path", "")),
        "relative_image_key": str(entry.get("relative_image_key", "")),
        "raw_shape": raw_shape_t,
        "centers": centers,
        "scores": scores,
        "ranks": entry.get("proposal_ranks"),
        "sources": entry.get("proposal_sources"),
        "metadata": entry.get("proposal_metadata") or {},
    }


def load_proposal_map(proposal_jsonl: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """Load a JSONL sidecar into a ``key -> entry`` mapping.

    Each entry is registered under both the absolute path and the
    relative key. If ``proposal_jsonl`` is ``None`` (or the file does
    not exist), the function returns an empty dict and prints a clear
    warning so callers can fall back to random crops.
    """
    if not proposal_jsonl:
        logger.warning(
            "[proposals] no proposal_jsonl provided; falling back to random local crops."
        )
        return {}
    if not os.path.exists(proposal_jsonl):
        logger.warning(
            "[proposals] proposal_jsonl=%s does not exist; falling back to random local crops.",
            proposal_jsonl,
        )
        return {}

    mapping: Dict[str, Dict[str, Any]] = {}
    n_lines = 0
    n_kept = 0
    with open(proposal_jsonl, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                import json
                obj = json.loads(line)
            except Exception as exc:
                logger.warning(
                    "[proposals] skipping malformed JSONL line %d: %s", line_no, exc
                )
                continue
            entry = _validate_entry(obj)
            if entry is None:
                logger.warning(
                    "[proposals] skipping invalid proposal entry at line %d "
                    "(missing fields, NaN/Inf, bad shape).", line_no,
                )
                continue
            n_kept += 1
            for key in (entry["image_path"], entry["relative_image_key"]):
                if key:
                    mapping[key] = entry
    logger.info(
        "[proposals] loaded %d valid entries (out of %d lines) from %s; "
        "registered %d keys",
        n_kept, n_lines, proposal_jsonl, len(mapping),
    )
    return mapping


def resolve_image_path_from_sample(sample: Dict[str, Any]) -> Optional[str]:
    """Return the absolute NIfTI path backing a MONAI sample.

    Tries (in order):

    * ``sample["image"].meta["filename_or_obj"]`` (current MONAI)
    * ``sample["image_meta_dict"]["filename_or_obj"]``
    * ``sample["image"].meta["filename"]``
    * ``sample["image"]`` if it is a plain ``str``
    """
    img = sample.get("image")
    candidates = []
    if img is not None and hasattr(img, "meta"):
        try:
            candidates.append(img.meta["filename_or_obj"])
        except Exception:
            pass
        try:
            candidates.append(img.meta["filename"])
        except Exception:
            pass
    meta_dict = sample.get("image_meta_dict")
    if isinstance(meta_dict, dict):
        for k in ("filename_or_obj", "filename"):
            if k in meta_dict:
                candidates.append(meta_dict[k])
    if isinstance(img, str):
        candidates.append(img)
    for cand in candidates:
        if isinstance(cand, str) and cand:
            return cand
    return None


def lookup_proposal_centers(
    sample: Dict[str, Any],
    proposal_map: Dict[str, Dict[str, Any]],
) -> Optional[np.ndarray]:
    """Look up proposal centres for ``sample``.

    Returns an ``[N, 3]`` ``np.float32`` array of ``(H, W, D)`` centres
    in the *raw* NIfTI voxel grid, or ``None`` if no proposal was
    registered for the current image or the proposal_map is empty.

    The centres are *not* clamped here — clamping happens later inside
    ``augmentations.proposal_centers_to_point_bboxes`` so that the
    same logic governs online shape-mismatch detection.
    """
    if not proposal_map:
        return None
    path = resolve_image_path_from_sample(sample)
    if not path:
        return None
    abs_key = normalize_path(path)
    rel_key = make_relative_image_key(path)
    entry = proposal_map.get(abs_key) or proposal_map.get(rel_key)
    if entry is None:
        return None
    return entry["centers"]


__all__ = [
    "normalize_path",
    "make_relative_image_key",
    "load_proposal_map",
    "resolve_image_path_from_sample",
    "lookup_proposal_centers",
]