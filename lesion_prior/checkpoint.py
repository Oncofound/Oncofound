"""Strict CoarseLocNet checkpoint loading, including the recorded Config format."""
from pathlib import Path
from collections.abc import Mapping
import torch
from . import config as cm
from .model import build_coarselocnet


def load_checkpoint(path):
    # The historical checkpoint stores these configuration dataclasses.
    # Do not fall back to unrestricted pickle loading for unknown globals.
    classes = [cm.Config, cm.DataConfig, cm.TargetConfig, cm.ModelConfig,
               cm.LossConfig, cm.CropEvalConfig, cm.TrainConfig, cm.VisConfig]
    with torch.serialization.safe_globals(classes):
        state = torch.load(Path(path), map_location='cpu', weights_only=True)
    if not isinstance(state, Mapping):
        raise ValueError('Expected a checkpoint mapping or a model state_dict')
    return state


def build_model(cfg):
    return build_coarselocnet(
        num_modalities=cfg.data.num_modalities,
        base_channels=cfg.model.base_channels,
        stem_channels=cfg.model.stem_channels,
        num_res_blocks=cfg.model.num_res_blocks,
        modality_dropout=cfg.model.modality_dropout,
        use_modality_attention=cfg.model.use_modality_attention,
        norm_groups=cfg.model.norm_groups,
        trunk_dropout=cfg.model.trunk_dropout,
        use_residual_trunk=cfg.model.use_residual_trunk,
        use_cross_slice_head=cfg.model.use_cross_slice_head,
        share_head_trunk=cfg.model.share_head_trunk,
    )


def build_model_from_checkpoint(path, cfg):
    state = load_checkpoint(path)
    sd = state.get('model', state)
    if not isinstance(sd, Mapping) or not sd:
        raise ValueError('Checkpoint has no model tensors')
    if all(k.startswith('module.') for k in sd):
        sd = {k[7:]: v for k, v in sd.items()}
    # The supplied configuration is the explicit public architecture contract.
    # strict=True catches mismatched channels, heads, missing and extra keys.
    model = build_model(cfg)
    model.load_state_dict(sd, strict=True)
    if any(v.is_floating_point() and not torch.isfinite(v).all()
           for v in model.state_dict().values()):
        raise ValueError('Checkpoint contains non-finite model values')
    return model
