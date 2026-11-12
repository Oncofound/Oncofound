"""build_unlabeled_proposal_points.py
====================================
Offline conversion from ``selected_crops.jsonl`` (produced by the
coarse-localization model) to a *proposal-points* JSONL that the
second 3D DINO stage can consult at training time.

Important constraints
---------------------
* Only **candidate centers** are persisted; no crop boxes or final
  local crops are cached.
* Coordinates are converted from the crop_light model's
  *processed voxel grid* back to the original NIfTI voxel grid that
  3D DINO reads with ``LoadImaged``.
* Both an absolute image path and a ``V3/...``-style relative key
  are stored so that the JSONL stays usable across data-root
  relocations.
* Per-case failures are recorded in ``errors.jsonl`` and skipped,
  not raised.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Spacing helper extracted from the original preprocessing implementation.
def _read_nifti_affine(path: str) -> Tuple[float, float, float]:
    import nibabel as nib
    try:
        img = nib.as_closest_canonical(nib.load(path))
        z = img.header.get_zooms()
        return (float(z[0]), float(z[1]), float(z[2]))
    except Exception:
        return (1.0, 1.0, 1.0)


logger = logging.getLogger("build_unlabeled_proposal_points")


# Path normalisation helpers
def normalize_absolute_path(path: str) -> str:
    """Normalise an absolute path the same way on every host."""
    if path is None:
        return ""
    return os.path.normpath(os.path.abspath(path))


def make_relative_image_key(path: str) -> str:
    """Return a portable key relative to the dataset root.

    Preference order:
      * substring after ``V3/`` if present (case sensitive, then
        lower-case fallback);
      * otherwise the normalised absolute path.
    """
    norm = normalize_absolute_path(path)
    if not norm:
        return ""
    markers = ("/V3/", "\\V3\\")
    for marker in markers:
        idx = norm.rfind(marker)
        if idx >= 0:
            return norm[idx + len(marker):].replace("\\", "/")
    # Try lowercase match for case-insensitive filesystems.
    low = norm.lower()
    for marker in ("/v3/", "\\v3\\"):
        idx = low.rfind(marker)
        if idx >= 0:
            return norm[idx + len(marker):].replace("\\", "/")
    return norm


# Manifest helpers
def _load_manifest(path: str) -> List[Dict[str, str]]:
    """Load a manifest CSV into a list of dicts."""
    with open(path, newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


# Coordinate-conversion (crop_light processed voxel -> raw NIfTI voxel).
def _apply_token_stride_padding(
    shape: Sequence[int],
    token_stride: Sequence[int],
) -> Tuple[List[int], int]:
    """Return the padded shape expected after the crop_light pipeline.

    The first project's ``LesionPriorDataset.__getitem__`` pads each
    spatial axis to a multiple of ``token_stride`` along the *end* of
    the axis (replicate-padding, last axis only ever). The padding is
    applied AFTER ``_resize_volume_to_spacing``, AFTER normalisation
    and AFTER geometric augmentation. For our purposes the
    augmentation is disabled (we run on a fresh NIfTI read), so the
    shape that ``processed_voxel`` lives in is

        processed = pad_end( resample_to_target_spacing(canonical) ,
                             token_stride )

    where ``pad_end`` only pads along the end of each axis.
    """
    padded = list(shape)
    for i, ts in enumerate(token_stride):
        if int(ts) <= 0:
            continue
        if padded[i] % int(ts) != 0:
            padded[i] = padded[i] + (int(ts) - padded[i] % int(ts))
    return padded, len(shape)


def _processed_to_canonical_voxel(
    processed_voxel: np.ndarray,
    resampled_shape: Sequence[int],
    padded_shape: Sequence[int],
    canonical_shape: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Map a coord in the padded, resampled grid back to canonical
    space using the F.interpolate(..., align_corners=False) convention.
    """
    processed_voxel = np.asarray(processed_voxel, dtype=np.float64)
    resampled_shape = np.asarray(resampled_shape, dtype=np.float64)
    padded_shape = np.asarray(padded_shape, dtype=np.float64)
    if canonical_shape is not None:
        # Padding appends voxels; it does not resize the volume. Clamp
        # replicated tail voxels to the resampled image before inverting
        # the actual trilinear resize (align_corners=False).
        unpadded = np.clip(processed_voxel, 0.0, resampled_shape - 1.0)
        return (unpadded + 0.5) * np.asarray(canonical_shape) / resampled_shape - 0.5
    # Historical conversion retained only for explicit legacy comparison.
    # First map padded->resampled, then resampled->canonical. With
    # align_corners=False, F.interpolate maps index i in size-N to
    # (i + 0.5) / N in [0, 1]. The inverse is therefore
    # canonical = (processed + 0.5) * canonical_size / processed_size - 0.5
    canonical_voxel = (processed_voxel + 0.5) * resampled_shape / padded_shape - 0.5
    return canonical_voxel


def _canonical_voxel_to_raw(
    canonical_voxel: np.ndarray,
    canonical_img: Any,
    raw_img: Any,
) -> np.ndarray:
    """Map canonical voxel coordinates to raw NIfTI voxel coordinates.

    ``canonical_img`` is the result of ``nib.as_closest_canonical``,
    ``raw_img`` is the original NIfTI on disk. We go through world
    coordinates so that arbitrary affine orientations are handled.
    """
    import nibabel as nib  # type: ignore

    canonical_voxel = np.asarray(canonical_voxel, dtype=np.float64)
    world = nib.affines.apply_affine(canonical_img.affine, canonical_voxel)
    raw_voxel = nib.affines.apply_affine(np.linalg.inv(raw_img.affine), world)
    return raw_voxel


# Crop-box -> center extraction (the JSONL stores z1,y1,x1,z2,y2,x2).
def _crop_box_to_processed_center(
    box: Dict[str, Any],
) -> np.ndarray:
    """Return the processed-voxel center of a selected crop box.

    The first project stores ``z1, y1, x1, z2, y2, x2`` as int in
    *processed* voxel coords (post resize + token-stride padding).
    We treat these axes as the array axes 0/1/2 of the processed
    volume (no anatomical direction assignment here).
    """
    z1 = float(box.get("z1", 0))
    y1 = float(box.get("y1", 0))
    x1 = float(box.get("x1", 0))
    z2 = float(box.get("z2", 0))
    y2 = float(box.get("y2", 0))
    x2 = float(box.get("x2", 0))
    center = np.array(
        [(z1 + z2 - 1) / 2.0, (y1 + y2 - 1) / 2.0, (x1 + x2 - 1) / 2.0],
        dtype=np.float64,
    )
    return center


# Image-side re-implementation of the crop_light preprocessing.
def _processed_shape_for_image(
    image_path: str,
    modalities: Sequence[str],
    target_spacing: Sequence[float],
    token_stride: Sequence[int],
    enforce_spacing: bool = True,
) -> Tuple[List[int], List[int]]:
    """Compute the resampled + token-stride-padded shape for one image.

    This mirrors the relevant pieces of ``LesionPriorDataset`` without
    ever storing the full volume — we only need shapes and affines to
    invert the coordinate transform.
    """
    paths = image_path.split("|")
    first_path = paths[0]

    # Resampled shape: replicate the dataset's pipeline. We only need
    # the canonical spatial shape, so a direct nibabel read is enough
    # (avoid calling ``_load_image_stack`` which needs a DataConfig).
    import nibabel as _nib  # type: ignore
    _raw_img = _nib.load(first_path)
    _canonical = _nib.as_closest_canonical(_raw_img)
    proc_shape = list(_canonical.shape)
    # If 4D, take the first modality's shape (single-channel spatial).
    if len(proc_shape) == 4 and len(paths) == 1:
        proc_shape = proc_shape[1:] if proc_shape[0] <= 16 else proc_shape[:3]
    if enforce_spacing:
        src_spacing = _read_nifti_affine(first_path)
        # Match dataset.py: ratio = src / target.
        src = np.asarray(src_spacing, dtype=np.float32)
        tgt = np.asarray(target_spacing, dtype=np.float32)
        ratio = src / np.clip(tgt, 1e-3, None)
        resampled_shape = [
            max(1, int(round(s * r)))
            for s, r in zip(proc_shape, ratio)
        ]
    else:
        resampled_shape = list(proc_shape)
    padded_shape, _ = _apply_token_stride_padding(resampled_shape, token_stride)
    return resampled_shape, padded_shape


# Main conversion: one entry per *image*.
def convert_one_image(
    image_path: str,
    modalities: Sequence[str],
    boxes: List[Dict[str, Any]],
    *,
    target_spacing: Sequence[float],
    token_stride: Sequence[int],
    topk: int,
    case_id: str,
    selector: str,
    source_crop_size: Sequence[int],
    checkpoint_name: str,
    enforce_spacing: bool = True,
    coordinate_mode: str = "corrected",
) -> Tuple[Optional[Dict[str, Any]], Dict[str, int]]:
    """Convert boxes for one image to a proposal entry.

    Returns ``(entry_or_None, stats)``. On any failure ``entry_or_None``
    is ``None`` and stats contains the relevant counters.
    """
    stats = {
        "raw_boxes": len(boxes),
        "kept_after_rank": 0,
        "kept_after_finite": 0,
        "kept_after_unique": 0,
        "kept_after_padding_clamp": 0,
        "kept_after_shape_clamp": 0,
        "illegal_coords": 0,
        "padding_clamped": 0,
        "shape_clamped": 0,
        "duplicates_dropped": 0,
    }
    if not image_path or not os.path.exists(image_path):
        return None, stats
    if not boxes:
        return None, stats
    if coordinate_mode not in ("corrected", "legacy"):
        raise ValueError("coordinate_mode must be corrected or legacy")

    try:
        import nibabel as nib  # type: ignore
        raw_img = nib.load(image_path)
        if len(raw_img.shape) != 3:
            raise ValueError("Proposal conversion requires a single 3D NIfTI image")
        canonical_img = nib.as_closest_canonical(raw_img)
        raw_shape = list(raw_img.shape if coordinate_mode == "corrected" else canonical_img.shape)
        # Resampled + padded shape that the processed voxel grid has.
        resampled_shape, padded_shape = _processed_shape_for_image(
            image_path,
            modalities,
            target_spacing=target_spacing,
            token_stride=token_stride,
            enforce_spacing=enforce_spacing,
        )
    except Exception as exc:
        logger.warning("[skip] failed to probe image %s: %s", image_path, exc)
        return None, stats

    # Sort boxes by rank then score (defensive).
    boxes_sorted = sorted(
        boxes,
        key=lambda b: (
            int(b.get("rank", 1_000_000)),
            -float(b.get("score", 0.0)),
        ),
    )
    if topk > 0:
        boxes_sorted = boxes_sorted[:topk]
    stats["kept_after_rank"] = len(boxes_sorted)

    centers: List[List[float]] = []
    scores: List[float] = []
    ranks: List[int] = []
    sources: List[str] = []
    seen = set()

    for b in boxes_sorted:
        center_processed = _crop_box_to_processed_center(b)
        # Step 1: clamp into padded shape first.
        proc_clamped = np.minimum(
            np.maximum(center_processed, [0.0, 0.0, 0.0]),
            (np.asarray(padded_shape, dtype=np.float64) - 1.0),
        )
        if not np.allclose(proc_clamped, center_processed, atol=1e-6):
            stats["padding_clamped"] += 1
        canonical_voxel = _processed_to_canonical_voxel(
            proc_clamped, resampled_shape, padded_shape,
            canonical_shape=canonical_img.shape if coordinate_mode == "corrected" else None,
        )
        raw_voxel = _canonical_voxel_to_raw(canonical_voxel, canonical_img, raw_img)
        # Validate finite.
        if not np.isfinite(raw_voxel).all():
            stats["illegal_coords"] += 1
            continue
        stats["kept_after_finite"] += 1
        # Clamp into raw shape.
        clamped = np.minimum(
            np.maximum(raw_voxel, [0.0, 0.0, 0.0]),
            (np.asarray(raw_shape, dtype=np.float64) - 1.0),
        )
        if not np.allclose(clamped, raw_voxel, atol=1e-6):
            stats["shape_clamped"] += 1
        # Round to int (we treat voxel coords as discrete grid indices).
        h = int(round(float(clamped[0])))
        w = int(round(float(clamped[1])))
        d = int(round(float(clamped[2])))
        # Final clamp after rounding.
        h = max(0, min(h, raw_shape[0] - 1))
        w = max(0, min(w, raw_shape[1] - 1))
        d = max(0, min(d, raw_shape[2] - 1))
        key = (h, w, d)
        if key in seen:
            stats["duplicates_dropped"] += 1
            continue
        seen.add(key)
        centers.append([h, w, d])
        scores.append(float(b.get("score", 0.0)))
        ranks.append(int(b.get("rank", len(centers))))
        sources.append(str(b.get("source", "gauss")))
        stats["kept_after_unique"] += 1
        if stats["padding_clamped"]:
            stats["kept_after_padding_clamp"] += 1
        if stats["shape_clamped"]:
            stats["kept_after_shape_clamp"] += 1

    if not centers:
        return None, stats

    entry = {
        "case_id": case_id,
        "image_path": normalize_absolute_path(image_path),
        "relative_image_key": make_relative_image_key(image_path),
        "coord_space": "raw_nifti_voxel",
        "raw_shape": raw_shape,
        "proposal_centers": centers,
        "proposal_scores": scores,
        "proposal_ranks": ranks,
        "proposal_sources": sources,
        "proposal_metadata": {
            "selector": selector,
            "topk": int(topk),
            "source_crop_size": list(source_crop_size),
            "target_spacing": [float(s) for s in target_spacing],
            "token_stride": [int(t) for t in token_stride],
            "checkpoint": checkpoint_name,
            "coordinate_mode": coordinate_mode,
        },
    }
    return entry, stats


# CLI
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert coarse selected_crops.jsonl into DINOv3 proposal points.",
    )
    p.add_argument(
        "--selected-crops",
        action="append",
        required=True,
        help="Path to selected_crops.jsonl produced by crop_light test.py. "
             "Repeatable: pass multiple times to merge boxes from several "
             "selector runs (e.g. one NMS run + one Gaussian run).",
    )
    p.add_argument(
        "--selector",
        action="append",
        choices=["gaussian", "nms"],
        help="Selector tag for the matching --selected-crops. Must appear "
             "the same number of times as --selected-crops and in the same "
             "order. Defaults to 'gaussian' for the first input if omitted.",
    )
    p.add_argument("--manifest", required=True,
                   help="Manifest CSV mapping case_id -> image_path. Used to look "
                        "up which NIfTI file corresponds to each box row.")
    p.add_argument("--output", required=True,
                   help="Path to write unlabeled_proposal_points.jsonl")
    p.add_argument("--errors", default=None,
                   help="Optional path to errors.jsonl (defaults to "
                        "<output>.errors.jsonl)")
    p.add_argument("--target-spacing", type=float, nargs=3, default=[1.5, 1.5, 1.5],
                   help="Target voxel spacing for crop_light's resample step.")
    p.add_argument("--token-stride", type=int, nargs=3, default=[8, 8, 8],
                   help="Token stride for crop_light's tail padding.")
    p.add_argument("--topk", type=int, default=8,
                   help="Maximum number of proposal centres to keep PER "
                        "SELECTOR per case. With 2 selectors and topk=8 the "
                        "final JSONL can carry up to 16 centres per case.")
    p.add_argument("--checkpoint-name", default="best_checkpoint.pt")
    p.add_argument("--coordinate-mode", choices=["corrected", "legacy"], default="corrected",
                   help="Correct inverse resize/padding/orientation, or historical conversion for comparison.")
    p.add_argument("--source-crop-size", type=int, nargs=3, default=[48, 48, 48],
                   help="Crop size used by crop_light's box selector.")
    p.add_argument("--modality", default="CT",
                   help="Default modality string for single-modality NIfTI files. "
                        "Used only to probe the resampled shape.")
    p.add_argument("--modality-key", default="modalities",
                   help="Manifest column that holds modality strings.")
    p.add_argument("--case-id-key", default="case_id",
                   help="Manifest column that holds the case id.")
    p.add_argument("--image-key", default="image",
                   help="Manifest column that holds the NIfTI path.")
    p.add_argument("--image-root", default=None,
                   help="Optional override for the image path prefix when "
                        "manifest entries are relative.")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, only process the first N cases (debug).")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _coalesce_selectors(args) -> List[Tuple[str, str]]:
    """Pair up ``--selected-crops`` and ``--selector`` CLI inputs.

    ``--selector`` is optional; an unspecified selector for input ``i``
    defaults to ``"gaussian"`` if it's the only input, otherwise raises.
    """
    inputs: List[str] = list(args.selected_crops)
    selectors: List[str] = list(args.selector or [])
    if len(selectors) == 0 and len(inputs) == 1:
        return [(inputs[0], "gaussian")]
    if len(selectors) != len(inputs):
        raise SystemExit(
            f"--selector must be specified the same number of times as "
            f"--selected-crops ({len(inputs)} inputs, {len(selectors)} selectors)."
        )
    return list(zip(inputs, selectors))


def _load_boxes_by_case(path: str) -> Tuple[Dict[str, List[Dict[str, Any]]], int]:
    boxes_by_case: Dict[str, List[Dict[str, Any]]] = {}
    n_rows = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            cid = obj.get("case_id", "")
            if not cid:
                continue
            boxes_by_case.setdefault(cid, []).append(obj)
            n_rows += 1
    return boxes_by_case, n_rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    input_pairs = _coalesce_selectors(args)
    logger.info(
        "merging %d selected_crops inputs: %s",
        len(input_pairs),
        [(p, s) for p, s in input_pairs],
    )

    manifest_rows = _load_manifest(args.manifest)
    logger.info("loaded manifest with %d rows from %s", len(manifest_rows), args.manifest)

    # case_id -> first matching manifest row
    case_to_rows: Dict[str, List[Dict[str, str]]] = {}
    for row in manifest_rows:
        cid = row.get(args.case_id_key, "") or row.get("case_id", "")
        if not cid:
            continue
        case_to_rows.setdefault(cid, []).append(row)
    logger.info("manifest has %d unique case_ids", len(case_to_rows))

    # case_id -> [(selector, boxes)] (one entry per input)
    per_input: List[Tuple[str, Dict[str, List[Dict[str, Any]]], int]] = []
    union_cases: set = set()
    for path, selector in input_pairs:
        boxes_by_case, n_rows = _load_boxes_by_case(path)
        per_input.append((selector, boxes_by_case, n_rows))
        union_cases.update(boxes_by_case.keys())
        logger.info(
            "loaded %d boxes for %d cases from %s (selector=%s)",
            n_rows, len(boxes_by_case), path, selector,
        )

    output_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    errors_path = args.errors or (os.path.splitext(output_path)[0] + ".errors.jsonl")

    n_emitted = 0
    n_matched_cases = 0
    n_unmatched_case_ids = 0
    n_short = 0
    total_centers = 0
    n_illegal_coords = 0
    n_padding_clamped = 0
    n_shape_clamped = 0
    n_duplicates = 0
    # Per-selector totals.
    selector_totals: Dict[str, int] = {}

    emitted_keys: set = set()
    processed_cases = 0
    t0 = time.time()
    with open(output_path, "w") as fout, open(errors_path, "w") as ferr:
        for cid in sorted(union_cases):
            if args.limit > 0 and processed_cases >= args.limit:
                break
            processed_cases += 1
            rows = case_to_rows.get(cid, [])
            if not rows:
                n_unmatched_case_ids += 1
                ferr.write(json.dumps({
                    "case_id": cid,
                    "error_type": "NoManifestRow",
                    "error_msg": "no manifest row matched this case_id",
                }) + "\n")
                continue
            row = rows[0]
            image_path = row.get(args.image_key, "") or row.get("image", "")
            if args.image_root:
                if not os.path.isabs(image_path):
                    image_path = os.path.join(args.image_root, image_path)
            modalities_str = row.get(args.modality_key, "") or row.get("modalities", "")
            modalities = [m for m in modalities_str.split("|") if m] or [args.modality]

            # Probe shape once per case (used by all selectors).
            try:
                _resampled_shape, _padded_shape = _processed_shape_for_image(
                    image_path,
                    modalities,
                    target_spacing=args.target_spacing,
                    token_stride=args.token_stride,
                )
            except Exception as exc:
                logger.warning("[skip] case_id=%s (shape probe): %s", cid, exc)
                ferr.write(json.dumps({
                    "case_id": cid,
                    "image_path": image_path,
                    "error_type": "ShapeProbeFailed",
                    "error_msg": str(exc),
                }) + "\n")
                continue

            # Merge per-selector entries for this case.
            all_centers: List[List[float]] = []
            all_scores: List[float] = []
            all_ranks: List[int] = []
            all_sources: List[str] = []
            case_seen_keys: set = set()
            per_selector_meta: List[Dict[str, Any]] = []
            case_n_illegal = 0
            case_n_padding = 0
            case_n_shape = 0
            case_n_dups = 0
            case_short = False
            case_failed = False
            primary_selector: Optional[str] = None
            first_entry: Optional[Dict[str, Any]] = None
            for selector, boxes_by_case, _nrows in per_input:
                boxes = boxes_by_case.get(cid, [])
                if not boxes:
                    # Selector had no boxes for this case; record meta.
                    per_selector_meta.append({
                        "selector": selector,
                        "n_boxes_in": 0,
                        "n_centers_out": 0,
                    })
                    continue
                try:
                    sub_entry, sub_stats = convert_one_image(
                        image_path=image_path,
                        modalities=modalities,
                        boxes=boxes,
                        target_spacing=args.target_spacing,
                        token_stride=args.token_stride,
                        topk=args.topk,
                        case_id=cid,
                        selector=selector,
                        source_crop_size=args.source_crop_size,
                        checkpoint_name=args.checkpoint_name,
                        coordinate_mode=args.coordinate_mode,
                    )
                except Exception as exc:
                    logger.warning(
                        "[skip] case_id=%s selector=%s: %s", cid, selector, exc
                    )
                    case_failed = True
                    per_selector_meta.append({
                        "selector": selector,
                        "error_type": type(exc).__name__,
                        "error_msg": str(exc),
                    })
                    continue
                if sub_entry is None:
                    per_selector_meta.append({
                        "selector": selector,
                        "n_boxes_in": len(boxes),
                        "n_centers_out": 0,
                        "stats": sub_stats,
                    })
                    if len(boxes) > 0 and len(boxes) >= args.topk:
                        # Selector was supposed to produce output but couldn't
                        case_short = True
                    continue
                if first_entry is None:
                    first_entry = sub_entry
                    primary_selector = selector
                # Pull the centres out, prefixed with the selector tag.
                for c, s, r, src in zip(
                    sub_entry["proposal_centers"],
                    sub_entry["proposal_scores"],
                    sub_entry["proposal_ranks"],
                    sub_entry["proposal_sources"],
                ):
                    key = (int(c[0]), int(c[1]), int(c[2]))
                    if key in case_seen_keys:
                        case_n_dups += 1
                        continue
                    case_seen_keys.add(key)
                    all_centers.append(c)
                    all_scores.append(s)
                    all_ranks.append(r)
                    # Tag the source with the selector that produced it so
                    # downstream code can distinguish e.g. "gauss:gauss"
                    # vs "nms:pred".
                    all_sources.append(f"{selector}:{src}")
                case_n_illegal += sub_stats["illegal_coords"]
                case_n_padding += sub_stats["padding_clamped"]
                case_n_shape += sub_stats["shape_clamped"]
                case_n_dups += sub_stats["duplicates_dropped"]
                if len(sub_entry["proposal_centers"]) < args.topk:
                    case_short = True
                per_selector_meta.append({
                    "selector": selector,
                    "n_boxes_in": len(boxes),
                    "n_centers_out": len(sub_entry["proposal_centers"]),
                    "stats": sub_stats,
                })

            if case_failed and not all_centers:
                ferr.write(json.dumps({
                    "case_id": cid,
                    "image_path": image_path,
                    "error_type": "AllSelectorsFailed",
                    "error_msg": "every selector raised an exception",
                    "selectors": per_selector_meta,
                }) + "\n")
                continue
            if not all_centers:
                ferr.write(json.dumps({
                    "case_id": cid,
                    "image_path": image_path,
                    "error_type": "NoValidProposals",
                    "error_msg": "no valid centres produced by any selector",
                    "selectors": per_selector_meta,
                }) + "\n")
                continue

            # Build the merged entry.
            entry = dict(first_entry)  # image_path / raw_shape / metadata
            entry["proposal_centers"] = all_centers
            entry["proposal_scores"] = all_scores
            entry["proposal_ranks"] = all_ranks
            entry["proposal_sources"] = all_sources
            entry["proposal_metadata"] = dict(entry.get("proposal_metadata") or {})
            entry["proposal_metadata"]["selectors"] = [
                s for s in input_pairs
            ]
            entry["proposal_metadata"]["primary_selector"] = primary_selector
            entry["proposal_metadata"]["per_selector"] = per_selector_meta

            key_abs = entry["image_path"]
            key_rel = entry["relative_image_key"]
            if key_abs in emitted_keys or key_rel in emitted_keys:
                continue
            emitted_keys.add(key_abs)
            emitted_keys.add(key_rel)
            fout.write(json.dumps(entry) + "\n")
            n_emitted += 1
            n_matched_cases += 1
            total_centers += len(entry["proposal_centers"])
            n_illegal_coords += case_n_illegal
            n_padding_clamped += case_n_padding
            n_shape_clamped += case_n_shape
            n_duplicates += case_n_dups
            # Per-selector counts (from per_selector_meta).
            for m in per_selector_meta:
                sel = m.get("selector")
                nc = int(m.get("n_centers_out", 0))
                if sel is not None:
                    selector_totals[sel] = selector_totals.get(sel, 0) + nc
            if case_short:
                n_short += 1

    elapsed = time.time() - t0
    logger.info(
        "done in %.2fs | manifest cases=%d | matched cases=%d | emitted=%d | "
        "centers total=%d | unmatched case_ids=%d | illegal coords=%d | "
        "padding clamped=%d | shape clamped=%d | duplicates dropped=%d | "
        "cases with < topk_per_selector=%d | avg centres / case=%.2f | "
        "per-selector centers=%s",
        elapsed,
        len(manifest_rows),
        n_matched_cases,
        n_emitted,
        total_centers,
        n_unmatched_case_ids,
        n_illegal_coords,
        n_padding_clamped,
        n_shape_clamped,
        n_duplicates,
        n_short,
        (float(total_centers) / n_emitted) if n_emitted > 0 else 0.0,
        selector_totals,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
