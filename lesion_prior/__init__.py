"""
lesion_prior
============
Lightweight 3D multi-modality multi-cancer ``CoarseLocNet`` package.

The package predicts crop-center hit probability maps at the token grid
resolution. It is used to drive the selection of 8 ``48^3`` local crops
for downstream DINOv3 self-supervised pretraining.
"""

from .config import (
    Config,
    DataConfig,
    TargetConfig,
    ModelConfig,
    LossConfig,
    CropEvalConfig,
    TrainConfig,
    VisConfig,
    default_config,
    load_config,
    save_config,
    merge_overrides,
)
from .dino_aug import DinoV3GeometricAug, DinoV3IntensityAug
from .dataset import LesionPriorDataset
from .targets import make_crop_hit_target, make_token_target
from .model import CoarseLocNet, build_coarselocnet, count_parameters
from .crop_select import (
    CropBox,
    select_topk_local_crops,
    box_iou_3d,
    boxes_to_mask,
    count_tumor_voxels_in_box,
    crop_tensor_with_box,
)

__all__ = [
    "Config",
    "DataConfig",
    "TargetConfig",
    "ModelConfig",
    "LossConfig",
    "CropEvalConfig",
    "TrainConfig",
    "VisConfig",
    "default_config",
    "load_config",
    "save_config",
    "merge_overrides",
    "DinoV3GeometricAug",
    "DinoV3IntensityAug",
    "LesionPriorDataset",
    "make_crop_hit_target",
    "make_token_target",
    "CoarseLocNet",
    "build_coarselocnet",
    "count_parameters",
    "CropBox",
    "select_topk_local_crops",
    "box_iou_3d",
    "boxes_to_mask",
    "count_tumor_voxels_in_box",
    "crop_tensor_with_box",
]
