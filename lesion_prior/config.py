"""
config.py
=========
Centralized configuration for the lesion_prior package.
All hyperparameters, paths and runtime switches are exposed as dataclasses.
"""

from __future__ import annotations

import os
import json
import copy
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Any, Optional


# Data
@dataclass
class DataConfig:
    """Dataset / IO configuration."""

    manifest_csv: str = ""
    image_key: str = "image_path"
    seg_key: str = "seg_path"
    case_id_key: str = "case_id"
    patient_id_key: str = "patient_id"
    cancer_type_key: str = "cancer_type"
    split_key: str = "split"
    modalities_key: str = "modalities"
    # Modality name list, ordered; missing channels are zero-filled.
    modality_order: List[str] = field(
        default_factory=lambda: ["CT", "PET", "T1", "T2", "FLAIR", "ADC"]
    )
    num_modalities: int = 6
    allow_missing_modalities: bool = True
    image_layout: str = "auto"  # "auto" / "MDHW" / "DHWM"
    spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5)
    train_roi_size: Tuple[int, int, int] = (96, 96, 96)
    val_roi_size: Tuple[int, int, int] = (128, 128, 128)
    num_workers: int = 4
    cache_rate: float = 0.0
    # Whether to use persistent workers in DataLoader
    persistent_workers: bool = False
    # Pin memory
    pin_memory: bool = False
    # Force a foreground crop with at least 1 fg voxel (only for training)
    foreground_guaranteed_crop_prob: float = 0.5
    # Whether to pre-load image into memory (only for small cohorts)
    preload: bool = False
    # Whether to apply orientation RAS
    orientation_ras: bool = True
    # Whether to enforce spacing
    enforce_spacing: bool = True
    # CT intensity normalization parameters
    ct_clip_min: float = -1000.0
    ct_clip_max: float = 400.0
    # MRI/PET percentile clip
    nonzero_percentile_low: float = 0.5
    nonzero_percentile_high: float = 99.5
    # Full-volume evaluation (no center crop) for val/test
    full_volume_eval: bool = True
    # Whether to also center crop in val/test (debug only)
    eval_center_crop: bool = False
    # Whether to resize val/test to val_roi_size
    eval_resize_to_roi: bool = False
    # If true, when image and seg shapes disagree (e.g. they were
    # stored with different in-plane resolutions), the dataset will
    # re-resample the seg to the image's grid before complaining.
    auto_align_seg_shape: bool = True
    # Cache directory for preprocessed volumes
    cache_dir: str = ""
    cache_preprocessed: bool = True
    cache_format: str = "pt"
    # prefetch factor for DataLoader workers
    prefetch_factor: int = 4
    # sw inference
    sw_roi_size: Tuple[int, int, int] = (128, 128, 128)
    sw_overlap: float = 0.25
    sw_batch_size: int = 4
    sw_blend_mode: str = "avg"  # "avg" or "gaussian"
    # ---- v3 test-only knobs ----
    # Whether to keep the full volume resident on GPU between windows.
    # Off by default to preserve existing behaviour. Recommended for
    # volumes < ``max_gpu_volume_gb``.
    volume_on_gpu: bool = False
    max_gpu_volume_gb: float = 4.0
    # Whether to compute supervision targets in __getitem__. Disable
    # for test/val to skip ~30% of per-item CPU work.
    compute_targets_in_dataset: bool = True


# Target
@dataclass
class TargetConfig:
    """Crop-center hit target configuration."""

    token_stride: Tuple[int, int, int] = (8, 8, 8)
    local_crop_size: Tuple[int, int, int] = (48, 48, 48)
    min_hit_voxels: int = 1
    robust_hit_voxels: int = 10
    hit_target_mode: str = "binary"  # binary / log_count / occupancy
    ignore_borderline: bool = False
    borderline_min_voxels: int = 1
    borderline_max_voxels: int = 5
    tau_token_aux: float = 0.02
    # When crop exceeds image boundary, we either pad (default) or
    # shift back into the image. Padding uses reflect mode in tensor utilities.
    boundary_behavior: str = "pad"  # "pad" / "shift"
    # If True, only foreground cases contribute to hit-recall loss.
    only_fg_hit_recall: bool = True

    # ---- v2 additions ----
    # Effective-target filter (Step 2: effective_target). Borderline tokens
    # become "ignore" so they don't pollute the BCE/Tversky gradient.
    effective_min_voxels: int = 3
    effective_min_robust_frac: float = 0.33
    # Distance-weighted soft target (Step 2: distance_weighted_soft).
    use_distance_weighted_soft: bool = False
    distance_weight_sigma_voxel: float = 8.0
    distance_weight_alpha: float = 0.5  # weight = exp(-d/sigma) * (count/amax)^alpha
    # DINOv3-style anisotropic local crop (Step 2: dinov3_crop_hit_target).
    # When multiscale_n_samples > 1 we sample crop sizes from these ranges.
    local_crop_inner_scale: Tuple[float, float] = (0.16, 0.48)
    local_crop_cross_scale: Tuple[float, float] = (0.2, 0.5)
    local_crop_output_size: Tuple[int, int, int] = (48, 48, 48)
    multiscale_n_samples: int = 1
    multiscale_aggregation: str = "max"  # "max" / "mean"
    # Connected-component-aware target (Step 2: component_target).
    use_component_target: bool = False
    # Bbox-driven synthetic positives (Step 4: bbox_synthetic_positive).
    use_bbox_synthetic_positive: bool = False
    bbox_positive_margin_voxel: int = 4


# Model
@dataclass
class ModelConfig:
    """CoarseLocNet model configuration."""

    base_channels: int = 96
    stem_channels: int = 64
    num_res_blocks: int = 3
    modality_dropout: float = 0.15
    use_modality_attention: bool = True
    max_params_million: float = 5.0
    # When base/stem cannot produce ~5M params, the user can ask for a
    # deeper model. We expose this only as a warning, not a hard switch.
    use_residual_trunk: bool = True
    # Final head intermediate channel ratio
    head_ratio: int = 2
    # Number of groups in GroupNorm
    norm_groups: int = 8
    # Optional dropout in trunk
    trunk_dropout: float = 0.0
    # Whether to use deep supervision at lower resolutions (not used for v1)
    deep_supervision: bool = False

    # ---- v2 additions ----
    # Cross-slice head: predicts whether a token is a positive crop
    # center when the crop is "tall" along the z-axis (DINOv3 cross-slice
    # scale range). When disabled (default), the model behaves identically
    # to v1.
    use_cross_slice_head: bool = False
    cross_slice_crop_size: Tuple[int, int, int] = (96, 48, 48)
    # When True, the cross_slice_head shares the first conv block of
    # crop_hit_head (saves ~50% of head params, slight perf drop).
    share_head_trunk: bool = False


# Loss
@dataclass
class LossConfig:
    """Per-case crop-hit loss configuration."""

    focal_gamma: float = 2.0
    ft_gamma: float = 1.3333333
    tversky_alpha: float = 0.3
    tversky_beta_phase1: float = 0.7
    tversky_beta_phase2: float = 0.85
    lambda_focal: float = 1.0
    lambda_tversky: float = 1.0
    lambda_hit_recall: float = 0.7
    lambda_token_aux: float = 0.2
    eps: float = 1e-6
    drw_start_frac: float = 0.4
    cb_alpha_phase2: float = 0.9
    # When True, ignore tokens with very small fg occupancy (< tau)
    ignore_low_token_aux: bool = True

    # ---- v2 additions ----
    # Top-k recall loss (Step 7).
    lambda_topk_recall: float = 0.0
    topk_k: int = 8
    topk_temperature: float = 1.0
    # Component-level recall loss (Step 7).
    lambda_component_recall: float = 0.0
    # Pairwise ranking loss (Step 7).
    lambda_ranking: float = 0.0
    ranking_margin: float = 0.2
    ranking_num_pairs: int = 64
    # Diversity loss (Step 7): penalises top-k picking the same cluster.
    lambda_diversity: float = 0.0
    diversity_sigma_voxels: float = 12.0
    # Cross-slice head loss (Step 7).
    lambda_cross_slice: float = 0.0
    # Count-weighted positive focal weight (Step 7).
    pos_weight_alpha: float = 0.0
    # Use effective_target mask in focal BCE (Step 1.1).
    use_effective_target: bool = False

    # ---- localization-sharpening (post-v2, safe additive term) ----
    # Center-sharpening loss: regress the crop-hit heatmap toward the
    # center-peaked distance-weighted soft target *inside* crop-hit
    # positive tokens, so the model learns a sharp peak at the tumor
    # center instead of a flat plateau. Requires
    # ``target.use_distance_weighted_soft = True`` to have any effect.
    # Default 0.0 => fully backward compatible (no behaviour change).
    lambda_center_soft: float = 0.0


# Crop evaluation
@dataclass
class CropEvalConfig:
    """Crop selection configuration at test time."""

    num_local_crops: int = 8
    local_crop_size: Tuple[int, int, int] = (48, 48, 48)
    nms_iou_threshold: float = 0.75
    min_center_distance: Tuple[int, int, int] = (8, 8, 8)
    threshold: float = 0.5
    min_hit_voxels: int = 1
    robust_hit_voxels: int = 10
    deterministic: bool = True
    # If fewer than k candidates pass threshold, fill from highest
    # non-redundant tokens. If still fewer, fill with uniform grid.
    fallback_to_grid: bool = True

    # ---- v2 additions ----
    # How to fuse crop_hit and cross_slice heads in inference.
    # "avg" → arithmetic mean; "min" → conservative min; "max" → optimistic
    # max; "weighted" → alpha * crop_hit + (1-alpha) * cross_slice.
    fusion_mode: str = "avg"
    fusion_alpha: float = 0.5
    # ---- v3 additions ----
    # When True, deterministic Gaussian-decay selector is preferred
    # over the legacy strict-NMS selector if both are available.
    prefer_gaussian_selector: bool = True


# Train
@dataclass
class TrainConfig:
    """Training loop configuration."""

    output_dir: str = "outputs/coarseloc"
    seed: int = 42
    epochs: int = 200
    batch_size: int = 2
    lr: float = 2e-4
    weight_decay: float = 1e-4
    amp: bool = True
    amp_dtype: str = "fp16"  # "fp16" / "bf16" / "fp32"
    grad_accum_steps: int = 1
    grad_clip_norm: float = 1.0
    val_every: int = 1
    save_every: int = 20
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    min_lr_ratio: float = 0.01
    # Sliding-window inference
    sw_overlap: float = 0.25
    sw_batch_size: int = 2
    # Resume from checkpoint
    resume: str = ""
    # How many batches to print each step
    log_every_n_steps: int = 10
    # If true, evaluate on val after every val_every epoch
    eval_on_train: bool = False
    # Limit train batches per epoch (debug)
    max_train_steps_per_epoch: int = 0
    max_val_steps_per_epoch: int = 0

    # ---- v2 additions ----
    # Curriculum: ramp effective_min_voxels from phase1 to phase2 at
    # transition_epoch_frac. See make_all_targets.
    curriculum_enabled: bool = False
    curriculum_phase1_min_count: int = 30
    curriculum_phase2_min_count: int = 5
    curriculum_transition_epoch_frac: float = 0.5
    # Hard-case mining: every hard_case_mining_every epochs, run val on
    # training set; cases with hit_rate < threshold get sampling_weight
    # multiplied by hard_case_mining_weight. (Implemented in train.py.)
    hard_case_mining_enabled: bool = False
    hard_case_mining_every: int = 5
    hard_case_mining_threshold: float = 0.3
    hard_case_mining_weight: float = 3.0
    # Whether to load DINOv3 crop scales from an external yaml
    dinov3_yaml_path: str = ""
    # Path to a previously-built prob cache (used by 3DINOv3 offline
    # inference; not used in training but recorded for traceability).
    prob_cache_dir: str = ""


# Visualization
@dataclass
class VisConfig:
    """Visualization configuration."""

    save_visualizations: bool = True
    max_vis_cases: int = 50
    vis_dpi: int = 110
    overlay_alpha: float = 0.35
    num_crop_panels: int = 8
    save_good_cases: bool = True
    save_bad_cases: bool = True
    save_random_cases: bool = True
    num_good: int = 8
    num_bad: int = 8
    num_random: int = 4
    crop_panels_per_row: int = 4
    overview_panels_per_row: int = 3


# Root config
@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    crop_eval: CropEvalConfig = field(default_factory=CropEvalConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    vis: VisConfig = field(default_factory=VisConfig)


# Helpers
def default_config() -> Config:
    """Return a default :class:`Config` instance.

    Returns
    -------
    Config
        Default configuration.
    """
    return Config()


def _to_dict(cfg: Any) -> Dict[str, Any]:
    """Recursively convert dataclass to dict (handles nested dataclasses and tuples)."""
    if hasattr(cfg, "__dataclass_fields__"):
        return {k: _to_dict(v) for k, v in asdict(cfg).items()}
    if isinstance(cfg, (list, tuple)):
        return [_to_dict(v) for v in cfg]
    if isinstance(cfg, dict):
        return {k: _to_dict(v) for k, v in cfg.items()}
    return cfg


def _from_dict(cls: Any, data: Dict[str, Any]) -> Any:
    """Reconstruct dataclass (or nested dataclass) from dict."""
    if not hasattr(cls, "__dataclass_fields__"):
        return data
    # Build a namespace of all the dataclass types reachable from ``cls``,
    # so we can resolve string annotations like ``"DataConfig"``.
    import inspect
    ns: Dict[str, Any] = {}
    for k, v in globals().items():
        if isinstance(v, type) and hasattr(v, "__dataclass_fields__"):
            ns[k] = v
    # Also pull in the same nested dataclass types via the class itself.
    for f in cls.__dataclass_fields__.values():  # type: ignore[attr-defined]
        ft = f.type
        if isinstance(ft, str):
            try:
                ft = eval(ft, ns)  # noqa: S307
            except Exception:
                ft = None
        if ft is not None and isinstance(ft, type) and hasattr(ft, "__dataclass_fields__"):
            ns.setdefault(ft.__name__, ft)
    out: Dict[str, Any] = {}
    for k, v in data.items():
        if k in cls.__dataclass_fields__:
            f = cls.__dataclass_fields__[k]
            ftype = f.type
            if isinstance(ftype, str):
                try:
                    ftype = eval(ftype, ns)  # noqa: S307
                except Exception:
                    ftype = None
            if ftype is not None and isinstance(ftype, type) and hasattr(ftype, "__dataclass_fields__"):
                out[k] = _from_dict(ftype, v)
            else:
                out[k] = v
    return cls(**out)


def save_config(cfg: Config, path: str) -> None:
    """Save a :class:`Config` to disk as YAML or JSON.

    Parameters
    ----------
    cfg : Config
        Configuration to save.
    path : str
        Output path. ``.yaml`` -> YAML (if PyYAML is installed) else JSON.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = _to_dict(cfg)
    if path.lower().endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore

            with open(path, "w") as f:
                yaml.safe_dump(payload, f, sort_keys=False)
            return
        except ImportError:
            pass
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)


def load_config(path: str) -> Config:
    """Load a :class:`Config` from disk.

    Parameters
    ----------
    path : str
        Path to YAML or JSON file previously written by :func:`save_config`.

    Returns
    -------
    Config
    """
    with open(path) as f:
        text = f.read()
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except ImportError:
        data = json.loads(text)
    return _from_dict(Config, data)


def merge_overrides(cfg: Config, overrides: Dict[str, str]) -> Config:
    """Apply dotted-key overrides, e.g. ``train.lr=1e-3`` to a config.

    Parameters
    ----------
    cfg : Config
        Base config.
    overrides : Dict[str, str]
        Mapping from dotted-key to stringified value.

    Returns
    -------
    Config
        A *copy* of the config with overrides applied.
    """
    new = copy.deepcopy(cfg)
    for k, v in overrides.items():
        cur: Any = new
        parts = k.split(".")
        for p in parts[:-1]:
            cur = getattr(cur, p)
        leaf = parts[-1]
        # Type cast using original field type
        old = getattr(cur, leaf)
        if isinstance(old, bool):
            val = v.lower() in ("1", "true", "yes", "y", "t")
        elif isinstance(old, int):
            val = int(v)
        elif isinstance(old, float):
            val = float(v)
        elif isinstance(old, tuple):
            val = tuple(_coerce(x) for x in v.split(","))
        elif isinstance(old, list):
            val = [_coerce(x) for x in v.split(",")]
        else:
            val = v
        setattr(cur, leaf, val)
    return new


def _coerce(x: str) -> Any:
    """Best-effort string -> int/float coercion."""
    try:
        return int(x)
    except ValueError:
        pass
    try:
        return float(x)
    except ValueError:
        pass
    return x
