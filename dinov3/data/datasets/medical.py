# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import ast
import csv
import json
import logging
import os
from copy import deepcopy
from pickle import UnpicklingError
from typing import Any, Callable, Optional

import numpy as np
import torch
from monai.data import Dataset, PersistentDataset
from monai.data.dataset import convert_to_tensor
from monai.data.utils import SUPPORTED_PICKLE_MOD, look_up_option

logger = logging.getLogger("dinov3")


def _load_gene_onehot_table(csv_path: str) -> dict[int, np.ndarray]:
    table: dict[int, np.ndarray] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        gene_cols = [c for c in reader.fieldnames or [] if c not in ("class_name", "class_id")]
        for row in reader:
            class_id = int(row["class_id"])
            gene_vec = np.array([int(row[g]) for g in gene_cols], dtype=np.float32)
            if class_id in table:
                table[class_id] = np.maximum(table[class_id], gene_vec)
            else:
                table[class_id] = gene_vec
    return table


def _load_tumor_map_from_onehot(csv_path: str) -> dict[str, int]:
    mapping: dict[str, int] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["class_name"]] = int(row["class_id"])
    return mapping

def _infer_class_name_from_image_path(root: str, image_path: str, tumor_map: dict[str, int] | None = None) -> str:
    rel = os.path.relpath(image_path, root)
    part = rel.split(os.sep)[1]
    if tumor_map and part in tumor_map:
        return part
    return part


def _attach_tumor_gene_metadata(
    data: list[dict],
    *,
    root: str,
    gene_onehot_csv: str | None,
):
    tumor_map = _load_tumor_map_from_onehot(gene_onehot_csv)
    gene_table = _load_gene_onehot_table(gene_onehot_csv)
    class_to_idx = tumor_map or {
        class_name: i
        for i, class_name in enumerate(
            sorted({_infer_class_name_from_image_path(root, sample["image"], tumor_map) for sample in data})
        )
    }

    for sample in data:
        class_name = _infer_class_name_from_image_path(root, sample["image"], class_to_idx)
        sample["class_name"] = class_name
        tumor_type_id = int(class_to_idx[class_name])
        sample["tumor_type_id"] = tumor_type_id
        gene_vec = gene_table.get(tumor_type_id)
        sample["gene_onehot"] = gene_vec.copy() if gene_vec is not None else np.empty(0, dtype=np.float32)


class PanCancerDataset(Dataset):
    def __init__(
        self,
        root: str,
        meta: Optional[str] = None,
        min_slice: Optional[int] = 24,
        gene_onehot_csv: str | None = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        self.target_transform = target_transform

        with open(meta, "r") as f:
            meta = {k: (ast.literal_eval(v) if isinstance(v, str) else v) for k, v in json.load(f).items()}

        data = []
        missing_label = 0
        for filename, shape in meta.items():
            if shape[-1] < min_slice:
                continue
            image_path = os.path.join(root, filename)
            label_path = image_path.replace("/image/", "/label/")
            if not os.path.exists(label_path):
                missing_label += 1
                continue
            data.append({"image": image_path, "label": label_path})

        if missing_label > 0:
            logger.warning(f"PanCancerDataset skipped {missing_label} samples with missing label files")
        if gene_onehot_csv is not None:
            _attach_tumor_gene_metadata(
                data,
                root=root,
                gene_onehot_csv=gene_onehot_csv,
            )

        super().__init__(data, transform)

    def __getitem__(self, index: int) -> Any:
        return super().__getitem__(index), self.target_transform(index)


class PanCancerPersistentDataset(PersistentDataset):
    def __init__(
        self,
        root: str,
        cache_dir: str,
        meta: Optional[str] = None,
        min_slice: Optional[int] = 24,
        gene_onehot_csv: str | None = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        self.target_transform = target_transform

        with open(meta, "r") as f:
            meta = {k: (ast.literal_eval(v) if isinstance(v, str) else v) for k, v in json.load(f).items()}

        data = []
        missing_label = 0
        for filename, shape in meta.items():
            if shape[-1] < min_slice:
                continue
            image_path = os.path.join(root, filename)
            label_path = image_path.replace("/image/", "/label/")
            if not os.path.exists(label_path):
                missing_label += 1
                continue
            data.append({"image": image_path, "label": label_path})

        if missing_label > 0:
            logger.warning(f"PanCancerPersistentDataset skipped {missing_label} samples with missing label files")
        if gene_onehot_csv is not None:
            _attach_tumor_gene_metadata(
                data,
                root=root,
                gene_onehot_csv=gene_onehot_csv,
            )

        super().__init__(data, transform, cache_dir)

    def __getitem__(self, index: int) -> Any:
        return super().__getitem__(index), self.target_transform(index)


class PanCancerUnlabeledDataset(Dataset):
    """Only includes samples without a label file (pre-filtered via metaV3_noSEG.json)."""

    def __init__(
        self,
        root: str,
        meta: Optional[str] = None,
        min_slice: Optional[int] = 24,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        self.target_transform = target_transform

        with open(meta, "r") as f:
            meta = {k: (ast.literal_eval(v) if isinstance(v, str) else v) for k, v in json.load(f).items()}

        data = []
        for filename, shape in meta.items():
            if shape[-1] < min_slice:
                continue
            data.append({"image": os.path.join(root, filename)})

        logger.info(f"PanCancerUnlabeledDataset: {len(data)} unlabeled samples")
        super().__init__(data, transform)

    def __getitem__(self, index: int) -> Any:
        return super().__getitem__(index), self.target_transform(index)


class PanCancerUnlabeledPersistentDataset(PersistentDataset):
    """PersistentDataset version — only includes samples without a label file.

    Current mode retains unmatched samples with random-crop fallback.
    ``legacy_basename`` reconstructs the historical basename filter recorded
    in the target checkpoint logs; it drops unmatched samples. Basename
    matching can conflate scans in different folders and is retained only
    for historical data-selection fidelity.
    """

    def __init__(
        self,
        root: str,
        cache_dir: str,
        meta: Optional[str] = None,
        min_slice: Optional[int] = 24,
        proposal_jsonl: Optional[str] = None,
        proposal_match_mode: str = "current",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        self.target_transform = target_transform
        if proposal_match_mode not in ("current", "legacy_basename"):
            raise ValueError("proposal_match_mode must be current or legacy_basename")

        with open(meta, "r") as f:
            meta = {k: (ast.literal_eval(v) if isinstance(v, str) else v) for k, v in json.load(f).items()}

        # Optional filter: keep only samples that have a registered
        # proposal centre in ``proposal_jsonl``. Keyed by the full
        # absolute image path so we never confuse two scans that happen
        # to share a filename across different datasets (e.g. a
        # ProstateX scan present under both ProstateTumor/ and
        # ColorectalCancer/). The JSONL records full absolute paths
        # in ``image_path`` so they match the keys in ``meta`` exactly.
        proposal_paths: Optional[set] = None
        if proposal_jsonl and os.path.exists(proposal_jsonl):
            proposal_paths = set()
            with open(proposal_jsonl, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    img = obj.get("image_path", "")
                    if not img:
                        continue
                    proposal_paths.add(os.path.basename(img) if proposal_match_mode == "legacy_basename" else img)
            logger.info(
                f"PanCancerUnlabeledPersistentDataset: proposal_jsonl loaded "
                f"with {len(proposal_paths)} unique keys ({proposal_match_mode})"
            )

        data = []
        n_no_proposal = 0
        for filename, shape in meta.items():
            if shape[-1] < min_slice:
                continue
            if proposal_paths is not None:
                # ``filename`` in meta is already the absolute path
                # produced by the meta-build pipeline, so a direct
                # equality check matches the JSONL exactly. Samples
                # *without* a registered proposal are kept (so every
                # unlabeled case participates in training); at
                # augmentation time they fall through to the
                # gt_bbox=None branch of RandomResizedCrop3DSetting2
                # which samples random local-crop centres.
                key = os.path.basename(filename) if proposal_match_mode == "legacy_basename" else filename
                if key not in proposal_paths:
                    n_no_proposal += 1
                    if proposal_match_mode == "legacy_basename":
                        continue
            data.append({"image": os.path.join(root, filename)})

        logger.info(f"PanCancerUnlabeledPersistentDataset: {len(data)} unlabeled samples")
        if proposal_paths is not None:
            if n_no_proposal > 0:
                logger.info(
                    f"PanCancerUnlabeledPersistentDataset: {n_no_proposal} samples have no "
                    f"registered proposal (mode={proposal_match_mode}; legacy mode excludes these samples)"
                )
            else:
                logger.info(
                    f"PanCancerUnlabeledPersistentDataset: every sample has a registered proposal"
                )
        super().__init__(data, transform, cache_dir)

    def __getitem__(self, index: int) -> Any:
        return super().__getitem__(index), self.target_transform(index)


# Cache hardening for PersistentDataset on GPFS
#
# MONAI's PersistentDataset._cachecheck uses tempfile.TemporaryDirectory +
# shutil.move to publish the cache file. On GPFS this is not strictly atomic
# and can leave a half-written zip if a worker is SIGKILLed mid-write
# (PytorchStreamReader: "failed finding central directory" on the next read).
# Replace _cachecheck with a version that:
#   - reads via the standard torch.load and, on any zip error, deletes the
#     bad file so the next call rebuilds it;
#   - writes via os.replace from a pid-suffixed temp file, which is atomic
#     on POSIX (and on GPFS for files in the same directory).
# Idempotent: safe to call multiple times.

def _patch_monai_persistent_dataset():
    from monai.data import dataset as _monai_ds
    if getattr(_monai_ds.PersistentDataset, "_gpfs_patched", False):
        return

    def _cachecheck_gpfs(self, item_transformed):
        cache_dir = self.cache_dir
        hashfile = None
        if cache_dir is not None:
            data_item_md5 = self.hash_func(item_transformed).decode("utf-8")
            data_item_md5 += self.transform_hash
            hashfile = cache_dir / f"{data_item_md5}.pt"

        if hashfile is not None and hashfile.is_file():
            try:
                return torch.load(hashfile, weights_only=True)
            except (UnpicklingError, RuntimeError, OSError) as e:
                # Any read error means the file is unusable. Delete it and
                # fall through to recompute. This is the crucial recovery
                # path on GPFS — handles half-written zips from a previous
                # killed worker.
                try:
                    hashfile.unlink()
                    logger.warning(
                        f"monai-gpfs: removed corrupt cache file {hashfile} "
                        f"({type(e).__name__}: {str(e)[:80]})"
                    )
                except OSError:
                    pass

        _item_transformed = self._pre_transform(deepcopy(item_transformed))
        if hashfile is None:
            return _item_transformed

        try:
            pid = os.getpid()
            tmp = hashfile.with_suffix(f".pt.tmp.{pid}")
            torch.save(
                obj=convert_to_tensor(_item_transformed, convert_numeric=False),
                f=tmp,
                pickle_module=look_up_option(self.pickle_module, SUPPORTED_PICKLE_MOD),
                pickle_protocol=self.pickle_protocol,
            )
            # os.replace is atomic on POSIX when src and dst are on the same
            # filesystem. Both tmp and hashfile live in cache_dir, so this
            # is safe even on GPFS.
            os.replace(tmp, hashfile)
        except PermissionError:
            pass
        except OSError as e:
            logger.warning(f"monai-gpfs: cache write failed for {hashfile}: {e}")

        return _item_transformed

    _monai_ds.PersistentDataset._cachecheck = _cachecheck_gpfs
    _monai_ds.PersistentDataset._gpfs_patched = True
    logger.info("monai-gpfs: PersistentDataset._cachecheck patched for atomic writes")


# Apply the patch at import time so any later ``from .medical import ...``
# call sees a hardened PersistentDataset base class.
_patch_monai_persistent_dataset()
