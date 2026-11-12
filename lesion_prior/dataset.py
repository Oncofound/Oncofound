"""
dataset.py
==========
``LesionPriorDataset``: multi-modality 3D lesion dataset returning
crop-hit targets. Designed to be small, defensive, and MONAI-free.

Each sample is a dict containing:
    * ``image``      : ``[M, D, H, W]`` image tensor.
    * ``seg``        : ``[D, H, W]`` integer label map.
    * ``mod_mask``   : ``[M]`` float (1 present, 0 missing).
    * ``targets``    : dict with crop-hit / token-fg tensors.
    * ``case_id``    : str
    * ``patient_id`` : str
    * ``cancer_type``: str
    * ``metadata``   : dict

The dataset supports *full-volume* validation/test (no center crop) by
default. Set ``data.full_volume_eval = false`` to fall back to the old
center-crop behaviour.
"""

from __future__ import annotations

import csv
import math
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
import torch

try:
    import nibabel as nib  # type: ignore
except Exception:
    nib = None

from .config import DataConfig, TargetConfig
from .dino_aug import DinoV3GeometricAug, DinoV3IntensityAug
from .targets import make_all_targets
from .cache import CacheKey, load_cached, save_cached


def _read_nifti_canonical(
    path: str, is_seg: bool = False
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Read a NIfTI and reorient it to the closest canonical form.

    Returns ``(arr, spacing)`` where ``spacing`` is ``(sx, sy, sz)``.
    """
    if nib is None:
        raise RuntimeError("nibabel is required to read NIfTI volumes.")
    img = nib.load(path)
    canonical = nib.as_closest_canonical(img)
    if is_seg:
        arr = np.asarray(canonical.get_fdata(), dtype=np.int32)
    else:
        arr = np.asarray(canonical.get_fdata(dtype=np.float32))
    try:
        z = canonical.header.get_zooms()
        spacing = (float(z[0]), float(z[1]), float(z[2]))
    except Exception:
        spacing = (1.0, 1.0, 1.0)
    return arr, spacing


def _read_nifti_4d_canonical(
    path: str, layout: str = "auto"
) -> Tuple[np.ndarray, str, Tuple[float, float, float]]:
    """Read a 4D NIfTI and decide channel layout.

    Returns ``(arr, resolved_layout, spacing)`` where ``arr`` is
    ``[C, D, H, W]`` (channel-major). ``resolved_layout`` is one of
    ``"MDHW"`` or ``"DHWM"``.

    Detection rules (when ``layout == "auto"``):
      * If ``last <= 16`` and ``first > 16``  → ``DHWM``.
      * If ``first <= 16`` and ``last > 16``  → ``MDHW``.
      * Otherwise raise ``ValueError`` so the user must pick a layout.
    """
    if nib is None:
        raise RuntimeError("nibabel is required to read NIfTI volumes.")
    img = nib.load(path)
    canonical = nib.as_closest_canonical(img)
    arr = np.asarray(canonical.get_fdata(dtype=np.float32))
    try:
        z = canonical.header.get_zooms()
        spacing = (float(z[0]), float(z[1]), float(z[2]))
    except Exception:
        spacing = (1.0, 1.0, 1.0)
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D NIfTI for {path}, got shape {arr.shape}")
    if layout == "MDHW":
        return arr, "MDHW", spacing
    if layout == "DHWM":
        return np.moveaxis(arr, -1, 0), "DHWM", spacing
    # auto
    last = arr.shape[-1]
    first = arr.shape[0]
    if first <= 16 and last > 16:
        # Channels along first dim
        return arr, "MDHW", spacing
    if last <= 16 and first > 16:
        return np.moveaxis(arr, -1, 0), "DHWM", spacing
    raise ValueError(
        f"Ambiguous 4D NIfTI layout for {path} with shape {arr.shape}; "
        f"please set cfg.data.image_layout to 'MDHW' or 'DHWM'."
    )


def _read_nifti_affine(path: str) -> Tuple[float, float, float]:
    if nib is None:
        return (1.0, 1.0, 1.0)
    try:
        # Images and segmentations have already been reoriented to RAS;
        # their voxel sizes must follow the same canonical axis order.
        img = nib.as_closest_canonical(nib.load(path))
        z = img.header.get_zooms()
        return (float(z[0]), float(z[1]), float(z[2]))
    except Exception:
        return (1.0, 1.0, 1.0)


def _normalize_modality(vol: np.ndarray, modality: str, cfg: DataConfig) -> np.ndarray:
    if modality.upper() == "CT":
        lo, hi = cfg.ct_clip_min, cfg.ct_clip_max
        vol = np.clip(vol, lo, hi)
        vol = (vol - lo) / max(1e-6, hi - lo)
    else:
        nz = vol[vol > 0]
        if nz.size > 0:
            lo = float(np.percentile(nz, cfg.nonzero_percentile_low))
            hi = float(np.percentile(nz, cfg.nonzero_percentile_high))
            if hi - lo < 1e-6:
                hi = lo + 1.0
            vol = np.clip(vol, lo, hi)
            mean = float(nz.mean())
            std = float(nz.std()) or 1.0
            vol = (vol - mean) / std
        else:
            mean = float(vol.mean())
            std = float(vol.std()) or 1.0
            vol = (vol - mean) / std
    return vol.astype(np.float32)


def _map_modalities_to_slots(
    arr: np.ndarray,
    modalities: Sequence[str],
    cfg: DataConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map ``arr`` (channel-major [M, D, H, W] or [1, D, H, W]) to the
    configured ``cfg.modality_order`` slots.

    Returns ``(slot_arr, mod_mask)`` where ``slot_arr`` is
    ``[len(slots), D, H, W]`` and ``mod_mask`` is ``[len(slots)]`` (1/0).
    Missing modalities are zero-filled.
    """
    slots = cfg.modality_order
    slot_to_idx = {m: i for i, m in enumerate(slots)}
    if arr.ndim == 3:
        # Single modality volume; treat as the first listed modality.
        first_mod = modalities[0] if modalities else "CT"
        out = np.zeros((len(slots),) + arr.shape, dtype=np.float32)
        if first_mod in slot_to_idx:
            out[slot_to_idx[first_mod]] = arr
        mod_mask = np.zeros(len(slots), dtype=np.float32)
        for slot, idx in slot_to_idx.items():
            mod_mask[idx] = 1.0 if slot == first_mod else 0.0
        return out, mod_mask
    # arr has explicit channels
    M = arr.shape[0]
    out = np.zeros((len(slots),) + arr.shape[1:], dtype=np.float32)
    mod_mask = np.zeros(len(slots), dtype=np.float32)
    for i, mod in enumerate(modalities):
        if i >= M:
            break
        if mod in slot_to_idx:
            out[slot_to_idx[mod]] = arr[i]
            mod_mask[slot_to_idx[mod]] = 1.0
    return out, mod_mask


def _load_image_stack(
    image_path: str,
    modalities: Sequence[str],
    cfg: DataConfig,
) -> np.ndarray:
    """Load a multi-modality image stack as ``[M, D, H, W]``.

    Supports:
      * ``|``-separated list of NIfTI paths (one per modality).
      * Single 4D NIfTI of shape ``[C, D, H, W]`` or ``[D, H, W, C]``
        (the layout is auto-detected or set via ``cfg.image_layout``).
    """
    paths = image_path.split("|")
    if len(paths) > 1:
        vol_list = []
        for p, mod in zip(paths, modalities):
            arr, _ = _read_nifti_canonical(p, is_seg=False)
            arr = _normalize_modality(arr, mod, cfg)
            vol_list.append(arr)
        while len(vol_list) < len(modalities):
            shape = vol_list[0].shape if vol_list else (1, 1, 1)
            vol_list.append(np.zeros(shape, dtype=np.float32))
        return np.stack(vol_list, axis=0).astype(np.float32)

    # Single file
    if nib is None:
        raise RuntimeError("nibabel is required to read NIfTI volumes.")
    img = nib.load(paths[0])
    canonical = nib.as_closest_canonical(img)
    arr = np.asarray(canonical.get_fdata(dtype=np.float32))
    if arr.ndim == 3:
        v = _normalize_modality(arr, modalities[0], cfg)
        out = np.zeros((len(modalities),) + arr.shape, dtype=np.float32)
        out[0] = v
        return out
    if arr.ndim == 4:
        if cfg.image_layout == "MDHW":
            arr4 = arr
        elif cfg.image_layout == "DHWM":
            arr4 = np.moveaxis(arr, -1, 0)
        else:  # auto
            last = arr.shape[-1]
            first = arr.shape[0]
            if first <= 16 and last > 16:
                arr4 = arr
            elif last <= 16 and first > 16:
                arr4 = np.moveaxis(arr, -1, 0)
            else:
                raise ValueError(
                    f"Ambiguous 4D NIfTI layout for {paths[0]} with shape {arr.shape}; "
                    f"please set cfg.data.image_layout to 'MDHW' or 'DHWM'."
                )
        # Per-channel normalization
        M = arr4.shape[0]
        for m in range(M):
            mod = modalities[m] if m < len(modalities) else "CT"
            arr4[m] = _normalize_modality(arr4[m], mod, cfg)
        return arr4.astype(np.float32)
    raise ValueError(f"Unexpected ndim={arr.ndim} for {paths[0]}")


def _resize_volume_to_spacing(
    vol: np.ndarray,
    src_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
    is_seg: bool = False,
    case_id: str = "",
    path: str = "",
) -> np.ndarray:
    """Resample ``vol`` to ``target_spacing``.

    Image is trilinear; seg is nearest. Raises on error.
    """
    import torch
    import torch.nn.functional as F
    if vol.ndim == 3:
        vol5 = vol[None, None]
    elif vol.ndim == 4:
        vol5 = vol[None]
    else:
        return vol
    src = np.asarray(src_spacing, dtype=np.float32)
    tgt = np.asarray(target_spacing, dtype=np.float32)
    ratio = src / np.clip(tgt, 1e-3, None)
    new_shape = [int(round(s * r)) for s, r in zip(vol5.shape[2:], ratio)]
    new_shape = [max(1, s) for s in new_shape]
    t = torch.from_numpy(vol5).float()
    if is_seg:
        out = F.interpolate(t, size=new_shape, mode="nearest")
    else:
        out = F.interpolate(t, size=new_shape, mode="trilinear", align_corners=False)
    out_np = out[0].numpy()
    if vol.ndim == 3:
        return out_np[0]
    return out_np


class LesionPriorDataset(torch.utils.data.Dataset):
    """Multi-modality 3D lesion dataset."""

    def __init__(
        self,
        cfg_data: DataConfig,
        cfg_target: TargetConfig,
        split: str = "train",
        manifest_rows: Optional[List[dict]] = None,
        augment: bool = True,
        max_samples: Optional[int] = None,
        compute_targets: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg_data
        self.cfg_t = cfg_target
        self.split = split
        self.augment = augment
        # If False, skip target generation (saves ~30% per-item CPU
        # time at test/val). Used by ``test.py`` for v3.
        self.compute_targets = bool(compute_targets)
        if manifest_rows is None:
            manifest_rows = self._read_manifest(cfg_data.manifest_csv)
        self.rows: List[dict] = [r for r in manifest_rows if r.get(cfg_data.split_key, "") == split]
        if not self.rows:
            raise RuntimeError(f"No rows for split={split!r} in manifest.")
        if max_samples is not None and max_samples > 0:
            self.rows = self.rows[: max_samples]
        self.geo_aug = DinoV3GeometricAug(roi_size=cfg_data.train_roi_size)
        self.int_aug = DinoV3IntensityAug()
        # Class-level set of cases that have already been logged as
        # failed. We share this across all dataset instances so that
        # the same bad case is reported only once globally.
        self._logged_skips = set()
        self._lock = __import__("threading").Lock()

    def _log_skip(self, case_id: str, exc: Exception) -> None:
        """Log a failed case only once globally, to avoid spam."""
        with self._lock:
            if case_id in self._logged_skips:
                return
            self._logged_skips.add(case_id)
        # Print on rank 0 / main process (no dist guard needed; if
        # multiple workers print, that's fine — the global set
        # dedupes).
        print(
            f"[lesion_prior.dataset] SKIP case_id={case_id!r}: {exc}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _read_manifest(path: str) -> List[dict]:
        with open(path) as f:
            return [dict(row) for row in csv.DictReader(f)]

    def _build_cache_key(self, row: dict) -> CacheKey:
        return CacheKey(
            image_path=row.get(self.cfg.image_key, ""),
            seg_path=row.get(self.cfg.seg_key, ""),
            spacing=tuple(self.cfg.spacing),
            modality_order=tuple(self.cfg.modality_order),
            orientation_ras=bool(self.cfg.orientation_ras),
            enforce_spacing=bool(self.cfg.enforce_spacing),
            ct_clip=(float(self.cfg.ct_clip_min), float(self.cfg.ct_clip_max)),
            nz_percentile=(float(self.cfg.nonzero_percentile_low), float(self.cfg.nonzero_percentile_high)),
        )

    def _load_case_from_disk(self, row: dict) -> Dict[str, torch.Tensor]:
        """Load one case, applying canonical orientation, spacing and
        modality-slot mapping. Raises (with case_id) on any error.
        """
        case_id = row.get(self.cfg.case_id_key, "")
        modalities_str = row.get(self.cfg.modalities_key, "")
        modalities = [m for m in modalities_str.split("|") if m]
        if not modalities:
            modalities = self.cfg.modality_order[:1]

        try:
            # 1. Load image
            img_arr = _load_image_stack(row[self.cfg.image_key], modalities, self.cfg)
            # 2. Resolve seg path. We *defer* allocating a zero seg until
            # after the image resample so that it matches the post-resize
            # shape (otherwise the shape-mismatch check below would raise
            # for unlabeled cases whose seg_path is empty / missing).
            seg_path = row.get(self.cfg.seg_key, "")
            seg_present = bool(seg_path) and os.path.exists(seg_path)
            if seg_present:
                seg_arr, _ = _read_nifti_canonical(seg_path, is_seg=True)
            else:
                # placeholder; will be replaced after image resize.
                seg_arr = np.zeros(img_arr.shape[1:], dtype=np.int32)
            # 3. Map modalities to slots
            out, mod_mask = _map_modalities_to_slots(img_arr, modalities, self.cfg)
            # 4. Enforce spacing on image
            if self.cfg.enforce_spacing:
                src_spacing = _read_nifti_affine(row[self.cfg.image_key].split("|")[0])
                out = _resize_volume_to_spacing(
                    out, src_spacing, self.cfg.spacing, is_seg=False,
                    case_id=case_id, path=row[self.cfg.image_key],
                )
            # If no seg is available, allocate zero seg at the *resized*
            # image shape so subsequent shape checks pass.
            if not seg_present:
                seg_arr = np.zeros(out.shape[1:], dtype=np.int32)
            # 5. Enforce spacing on seg (must match image)
            if self.cfg.enforce_spacing and seg_path and os.path.exists(seg_path):
                src_spacing = _read_nifti_affine(seg_path)
                seg_arr = _resize_volume_to_spacing(
                    seg_arr, src_spacing, self.cfg.spacing, is_seg=True,
                    case_id=case_id, path=seg_path,
                )
            # 6. Validate shapes. If they disagree by more than 10% in
            #    any axis, we treat the case as unfixable and raise
            #    (so the dataset's __getitem__ loop will skip to the
            #    next row). For small differences we optionally
            #    re-resample + crop/pad to align.
            if seg_arr.shape != out.shape[1:]:
                # Check if the mismatch is "small" (within 10% per axis)
                ratio = []
                for a, b in zip(seg_arr.shape, out.shape[1:]):
                    m = max(a, b)
                    if m == 0:
                        ratio.append(float("inf"))
                    else:
                        ratio.append(min(a, b) / m)
                small_mismatch = all(r >= 0.9 for r in ratio)
                if not self.cfg.auto_align_seg_shape or not small_mismatch:
                    raise RuntimeError(
                        f"seg shape {tuple(seg_arr.shape)} != image shape "
                        f"{tuple(out.shape[1:])} after preprocessing for "
                        f"case_id={case_id!r} "
                        f"(set data.auto_align_seg_shape=true with small mismatch to fix)"
                    )
                # Re-resample seg to the image's voxel grid using the
                # caller's target spacing.
                img_spacing = self.cfg.spacing
                if seg_path and os.path.exists(seg_path):
                    src_spacing = _read_nifti_affine(seg_path)
                else:
                    src_spacing = img_spacing
                seg_arr = _resize_volume_to_spacing(
                    seg_arr, src_spacing, img_spacing, is_seg=True,
                    case_id=case_id, path=seg_path or "",
                )
                # Crop or pad to the image's shape (centred)
                if seg_arr.shape != out.shape[1:]:
                    target_shape = out.shape[1:]
                    pad_z = max(0, target_shape[0] - seg_arr.shape[0])
                    pad_y = max(0, target_shape[1] - seg_arr.shape[1])
                    pad_x = max(0, target_shape[2] - seg_arr.shape[2])
                    if pad_z or pad_y or pad_x:
                        seg_arr = np.pad(
                            seg_arr,
                            ((0, pad_z), (0, pad_y), (0, pad_x)),
                            mode="constant", constant_values=0,
                        )
                    crop = (
                        max(0, (seg_arr.shape[0] - target_shape[0]) // 2),
                        max(0, (seg_arr.shape[1] - target_shape[1]) // 2),
                        max(0, (seg_arr.shape[2] - target_shape[2]) // 2),
                    )
                    seg_arr = seg_arr[
                        crop[0]: crop[0] + target_shape[0],
                        crop[1]: crop[1] + target_shape[1],
                        crop[2]: crop[2] + target_shape[2],
                    ]
                if seg_arr.shape != out.shape[1:]:
                    raise RuntimeError(
                        f"seg shape {tuple(seg_arr.shape)} != image shape "
                        f"{tuple(out.shape[1:])} after auto-align for case_id={case_id!r}"
                    )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load case_id={case_id!r} (path={row.get(self.cfg.image_key)!r}): {exc}"
            ) from exc

        image = torch.from_numpy(out).float()
        seg = torch.from_numpy(seg_arr).long()
        return {"image": image, "seg": seg, "mod_mask": torch.from_numpy(mod_mask).float()}

    def _load_case(self, row: dict) -> Dict[str, torch.Tensor]:
        # Try cache first
        if self.cfg.cache_preprocessed and self.cfg.cache_dir:
            key = self._build_cache_key(row).hash()
            cached = load_cached(self.cfg.cache_dir, key)
            if cached is not None:
                return {
                    "image": cached["image"].float(),
                    "seg": cached["seg"].long(),
                    "mod_mask": cached["mod_mask"].float(),
                }
        # Build and (optionally) cache
        payload = self._load_case_from_disk(row)
        if self.cfg.cache_preprocessed and self.cfg.cache_dir:
            try:
                save_cached(
                    self.cfg.cache_dir, self._build_cache_key(row).hash(),
                    image=payload["image"], seg=payload["seg"], mod_mask=payload["mod_mask"],
                )
            except Exception:
                pass
        return payload

    def _ensure_foreground_crop(
        self,
        image: torch.Tensor,
        seg: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fg_total = int((seg > 0).sum().item())
        if fg_total == 0:
            return image, seg
        roi = self.cfg.train_roi_size
        fg = (seg > 0).nonzero(as_tuple=False)
        if fg.numel() == 0:
            return image, seg
        z0, y0, x0 = fg.min(dim=0).values.tolist()
        z1, y1, x1 = fg.max(dim=0).values.tolist()
        half = (self.cfg_t.local_crop_size[0] // 2, self.cfg_t.local_crop_size[1] // 2, self.cfg_t.local_crop_size[2] // 2)
        z0 = max(int(z0) - half[0], 0)
        y0 = max(int(y0) - half[1], 0)
        x0 = max(int(x0) - half[2], 0)
        z1 = min(int(z1) + half[0] + 1, image.shape[1])
        y1 = min(int(y1) + half[1] + 1, image.shape[2])
        x1 = min(int(x1) + half[2] + 1, image.shape[3])
        cz = (z0 + z1) // 2
        cy = (y0 + y1) // 2
        cx = (x0 + x1) // 2
        rz, ry, rx = roi[0] // 2, roi[1] // 2, roi[2] // 2
        nz0 = max(0, cz - rz)
        ny0 = max(0, cy - ry)
        nx0 = max(0, cx - rx)
        nz1 = min(image.shape[1], nz0 + roi[0])
        ny1 = min(image.shape[2], ny0 + roi[1])
        nx1 = min(image.shape[3], nx0 + roi[2])
        if nz1 - nz0 < roi[0]:
            nz0 = max(0, nz1 - roi[0])
        if ny1 - ny0 < roi[1]:
            ny0 = max(0, ny1 - roi[1])
        if nx1 - nx0 < roi[2]:
            nx0 = max(0, nx1 - roi[2])
        nz1 = min(image.shape[1], nz0 + roi[0])
        ny1 = min(image.shape[2], ny0 + roi[1])
        nx1 = min(image.shape[3], nx0 + roi[2])
        image_c = image[:, nz0:nz1, ny0:ny1, nx0:nx1].contiguous()
        seg_c = seg[nz0:nz1, ny0:ny1, nx0:nx1].contiguous()
        return image_c, seg_c

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        case_id = row.get(self.cfg.case_id_key, str(index))
        patient_id = row.get(self.cfg.patient_id_key, "")
        cancer_type = row.get(self.cfg.cancer_type_key, "")
        # Try to load; if a single case fails (shape mismatch, missing
        # file, NIfTI parse error, …) we log the error once and skip to
        # the *next* row instead of crashing the whole epoch.
        last_exc: Optional[Exception] = None
        for offset in range(len(self.rows)):
            try_row = self.rows[(index + offset) % len(self.rows)]
            try_id = try_row.get(self.cfg.case_id_key, str((index + offset) % len(self.rows)))
            try:
                payload = self._load_case(try_row)
                image = payload["image"]
                seg = payload["seg"]
                mod_mask = payload["mod_mask"]
                case_id = try_id
                patient_id = try_row.get(self.cfg.patient_id_key, "")
                cancer_type = try_row.get(self.cfg.cancer_type_key, "")
                row = try_row
                break
            except Exception as exc:
                last_exc = exc
                self._log_skip(try_id, exc)
                continue
        else:
            # All rows failed; raise so the worker dies (this should
            # basically never happen).
            raise RuntimeError(
                f"All rows failed to load; last error: {last_exc}"
            )

        if self.augment and self.split == "train":
            if (self.cfg.foreground_guaranteed_crop_prob > 0
                    and int((seg > 0).sum().item()) > 0
                    and random.random() < self.cfg.foreground_guaranteed_crop_prob):
                image, seg = self._ensure_foreground_crop(image, seg)
            try:
                image, seg = self.geo_aug(image, seg)
            except Exception as exc:
                raise RuntimeError(f"Geo aug failed for case_id={case_id!r}: {exc}") from exc
            try:
                image, _ = self.int_aug(image)
            except Exception as exc:
                raise RuntimeError(f"Int aug failed for case_id={case_id!r}: {exc}") from exc
        elif self.augment and self.split != "train":
            try:
                image, seg = self.geo_aug(image, seg)
            except Exception as exc:
                raise RuntimeError(f"Geo aug failed for case_id={case_id!r}: {exc}") from exc
        else:
            if not self.cfg.full_volume_eval:
                if self.cfg.eval_center_crop or self.cfg.eval_resize_to_roi:
                    import torch.nn.functional as F
                    roi = self.cfg.val_roi_size
                    sub = image
                    cz, cy, cx = image.shape[1], image.shape[2], image.shape[3]
                    if self.cfg.eval_center_crop:
                        cz, cy, cx = min(image.shape[1], roi[0]), min(image.shape[2], roi[1]), min(image.shape[3], roi[2])
                        z0 = (image.shape[1] - cz) // 2
                        y0 = (image.shape[2] - cy) // 2
                        x0 = (image.shape[3] - cx) // 2
                        sub = image[:, z0:z0+cz, y0:y0+cy, x0:x0+cx]
                        seg = seg[z0:z0+cz, y0:y0+cy, x0:x0+cx]
                    if self.cfg.eval_resize_to_roi:
                        sub = F.interpolate(sub.unsqueeze(0), size=roi, mode="trilinear", align_corners=False).squeeze(0)
                        seg = F.interpolate(seg.unsqueeze(0).unsqueeze(0).float(), size=roi, mode="nearest").squeeze(0).squeeze(0).long()
                    image = sub

        # Pad to multiples of token_stride. ``F.pad`` requires a 4-tuple for
        # 4D tensors (last 2 dims), a 6-tuple for 5D tensors (last 3 dims).
        # ``image`` is 4D ``[M, D, H, W]``; ``seg5`` is 5D ``[1, 1, D, H, W]``.
        ts = self.cfg_t.token_stride
        for axis, tstride in zip((-3, -2, -1), ts):
            s = image.shape[axis]
            if s % tstride != 0:
                pad = tstride - (s % tstride)
                if axis == -1:
                    image = torch.nn.functional.pad(image, (0, pad, 0, 0), mode="replicate")
                    seg5 = seg.unsqueeze(0).unsqueeze(0).float()
                    seg5 = torch.nn.functional.pad(seg5, (0, pad, 0, 0, 0, 0), mode="replicate")
                    seg = seg5.squeeze(0).squeeze(0).long()
                elif axis == -2:
                    image = torch.nn.functional.pad(image, (0, 0, 0, pad), mode="replicate")
                    seg5 = seg.unsqueeze(0).unsqueeze(0).float()
                    seg5 = torch.nn.functional.pad(seg5, (0, 0, 0, pad, 0, 0), mode="replicate")
                    seg = seg5.squeeze(0).squeeze(0).long()
                else:
                    image = torch.nn.functional.pad(image, (0, 0, 0, 0, 0, pad), mode="replicate")
                    seg5 = seg.unsqueeze(0).unsqueeze(0).float()
                    seg5 = torch.nn.functional.pad(seg5, (0, 0, 0, 0, 0, pad), mode="replicate")
                    seg = seg5.squeeze(0).squeeze(0).long()

        # Build targets AFTER augmentation (v2: use make_all_targets which
        # combines v1 fields + new effective/distance/multiscale/component/
        # bbox/cross-slice targets based on cfg_t flags).
        if self.compute_targets:
            targets_all = make_all_targets(seg, self.cfg_t)
            targets = {}
            for k, v in targets_all.items():
                if isinstance(v, torch.Tensor):
                    if v.dim() == 5 and v.shape[0] == 1:
                        v = v[0]
                    targets[k] = v
                elif isinstance(v, dict):
                    # Nested dict (e.g. cross_slice target). Recursively
                    # squeeze leading singleton batch dim on each tensor.
                    inner = {}
                    for ik, iv in v.items():
                        if isinstance(iv, torch.Tensor):
                            if iv.dim() == 5 and iv.shape[0] == 1:
                                iv = iv[0]
                            inner[ik] = iv
                        else:
                            continue
                    if inner:
                        targets[k] = inner
                else:
                    # skip non-tensor entries (e.g. list of per-sample counts)
                    continue
        else:
            targets = {}

        # Record original volume shape (pre-pad) for cost estimation.
        meta = {
            "image_path": row.get(self.cfg.image_key, ""),
            "seg_path": row.get(self.cfg.seg_key, ""),
        }

        return {
            "case_id": case_id,
            "patient_id": patient_id,
            "cancer_type": cancer_type,
            "image": image,
            "seg": seg,
            "mod_mask": mod_mask,
            "targets": targets,
            "metadata": meta,
        }
