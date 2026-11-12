"""
model.py
========
``CoarseLocNet``: a lightweight 3D multi-modality network predicting
crop-center hit probability at the token-grid resolution.

The model has *two* heads sharing the same trunk:

* ``crop_hit_logits`` — supervised by the crop-hit / hit-recall loss.
* ``token_fg_logits`` — supervised by the token-fg auxiliary loss.

If only one head is requested (backwards compatibility), the model can
also return a single tensor instead of a dict.
"""

from __future__ import annotations

from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels: int, num_groups: int = 8) -> nn.GroupNorm:
    g = min(num_groups, num_channels)
    while num_channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=num_channels)


class ConvGNSiLU(nn.Module):
    """3x3x3 Conv3d + GroupNorm + SiLU."""

    def __init__(self, in_c: int, out_c: int, num_groups: int = 8) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.norm = _gn(out_c, num_groups)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResBlock3D(nn.Module):
    def __init__(self, channels: int, num_groups: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = _gn(channels, num_groups)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = _gn(channels, num_groups)
        self.act = nn.SiLU(inplace=True)
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.drop(out)
        return self.act(out + residual)


class PerModalityStem(nn.Module):
    """Shared per-modality stem producing token-grid features.

    Input  : ``[B, 1, D, H, W]``
    Output : ``[B, base_channels, d, h, w]`` with ``d = D // 8`` etc.
    """

    def __init__(self, stem_channels: int, base_channels: int, num_groups: int = 8) -> None:
        super().__init__()
        self.stage1 = ConvGNSiLU(1, stem_channels, num_groups)
        self.down1 = nn.Conv3d(stem_channels, stem_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_d1 = _gn(stem_channels, num_groups)
        self.act_d1 = nn.SiLU(inplace=True)
        self.stage2 = ConvGNSiLU(stem_channels, stem_channels, num_groups)
        self.down2 = nn.Conv3d(stem_channels, stem_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_d2 = _gn(stem_channels, num_groups)
        self.act_d2 = nn.SiLU(inplace=True)
        self.stage3 = ConvGNSiLU(stem_channels, base_channels, num_groups)
        self.down3 = nn.Conv3d(base_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_d3 = _gn(base_channels, num_groups)
        self.act_d3 = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.act_d1(self.norm_d1(self.down1(x)))
        x = self.stage2(x)
        x = self.act_d2(self.norm_d2(self.down2(x)))
        x = self.stage3(x)
        x = self.act_d3(self.norm_d3(self.down3(x)))
        return x


class CoarseLocNet(nn.Module):
    """Multi-modality crop-hit predictor with dual heads.

    Returns a dict ``{"crop_hit_logits": [B,1,d,h,w], "token_fg_logits":
    [B,1,d,h,w]}`` by default. For backwards compatibility, set
    ``return_dict=False`` in the forward call to receive just the
    ``crop_hit_logits`` tensor.
    """

    def __init__(
        self,
        num_modalities: int = 6,
        base_channels: int = 96,
        stem_channels: int = 64,
        num_res_blocks: int = 3,
        modality_dropout: float = 0.15,
        use_modality_attention: bool = True,
        norm_groups: int = 8,
        trunk_dropout: float = 0.0,
        use_residual_trunk: bool = True,
        use_cross_slice_head: bool = False,
        share_head_trunk: bool = False,
    ) -> None:
        super().__init__()
        self.num_modalities = int(num_modalities)
        self.base_channels = int(base_channels)
        self.stem_channels = int(stem_channels)
        self.num_res_blocks = int(num_res_blocks)
        self.modality_dropout = float(modality_dropout)
        self.use_modality_attention = bool(use_modality_attention)
        self.norm_groups = int(norm_groups)
        self.trunk_dropout = float(trunk_dropout)
        self.use_residual_trunk = bool(use_residual_trunk)
        self.use_cross_slice_head = bool(use_cross_slice_head)
        self.share_head_trunk = bool(share_head_trunk)

        self.stem = PerModalityStem(self.stem_channels, self.base_channels, num_groups=self.norm_groups)

        if self.use_residual_trunk:
            self.trunk = nn.Sequential(
                *[
                    ResBlock3D(self.base_channels, num_groups=self.norm_groups, dropout=self.trunk_dropout)
                    for _ in range(self.num_res_blocks)
                ]
            )
        else:
            layers = []
            for _ in range(self.num_res_blocks):
                layers += [
                    nn.Conv3d(self.base_channels, self.base_channels, kernel_size=3, padding=1, bias=False),
                    _gn(self.base_channels, self.norm_groups),
                    nn.SiLU(inplace=True),
                ]
            self.trunk = nn.Sequential(*layers)

        # Two heads, each independent
        head_mid = max(self.base_channels // 2, 16)
        # crop_hit_head: optional shared trunk (Step 6)
        if self.share_head_trunk:
            self._head_trunk = nn.Sequential(
                nn.Conv3d(self.base_channels, head_mid, kernel_size=3, padding=1, bias=False),
                _gn(head_mid, self.norm_groups),
                nn.SiLU(inplace=True),
            )
            self.crop_hit_head = nn.Sequential(
                self._head_trunk,
                nn.Conv3d(head_mid, 1, kernel_size=1),
            )
        else:
            self._head_trunk = None
            self.crop_hit_head = nn.Sequential(
                nn.Conv3d(self.base_channels, head_mid, kernel_size=3, padding=1, bias=False),
                _gn(head_mid, self.norm_groups),
                nn.SiLU(inplace=True),
                nn.Conv3d(head_mid, 1, kernel_size=1),
            )
        self.token_fg_head = nn.Sequential(
            nn.Conv3d(self.base_channels, head_mid, kernel_size=3, padding=1, bias=False),
            _gn(head_mid, self.norm_groups),
            nn.SiLU(inplace=True),
            nn.Conv3d(head_mid, 1, kernel_size=1),
        )
        # Cross-slice head (Step 5/6)
        if self.use_cross_slice_head:
            if self.share_head_trunk:
                self.cross_slice_head = nn.Sequential(
                    self._head_trunk,
                    nn.Conv3d(head_mid, 1, kernel_size=1),
                )
            else:
                self.cross_slice_head = nn.Sequential(
                    nn.Conv3d(self.base_channels, head_mid, kernel_size=3, padding=1, bias=False),
                    _gn(head_mid, self.norm_groups),
                    nn.SiLU(inplace=True),
                    nn.Conv3d(head_mid, 1, kernel_size=1),
                )

    def _modality_dropout_mask(
        self, mod_mask: torch.Tensor
    ) -> torch.Tensor:
        if not self.training or self.modality_dropout <= 0.0:
            return mod_mask > 0
        B, M = mod_mask.shape
        rand = torch.rand(B, M, device=mod_mask.device)
        drop = (rand < self.modality_dropout) & (mod_mask > 0)
        n_kept = ((mod_mask > 0) & ~drop).sum(dim=1)
        for i in range(B):
            if n_kept[i].item() == 0 and (mod_mask[i] > 0).any():
                present = (mod_mask[i] > 0).nonzero(as_tuple=True)[0]
                keep = present[torch.randint(0, present.numel(), (1,)).item()]
                drop[i, keep] = False
        return ((mod_mask > 0) & ~drop)

    def forward(
        self,
        image: torch.Tensor,
        mod_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if image.dim() != 5:
            raise ValueError(f"image must be [B, M, D, H, W], got {tuple(image.shape)}")
        B, M, D, H, W = image.shape
        if M != self.num_modalities:
            raise ValueError(f"image has M={M}, model expects {self.num_modalities}")
        if mod_mask is None:
            mod_mask = torch.ones(B, M, device=image.device, dtype=image.dtype)
        mod_mask = mod_mask.to(image.device).float()
        eff_mask = self._modality_dropout_mask(mod_mask).float()

        x = image.unsqueeze(2)
        x = x.reshape(B * M, 1, D, H, W)
        feats = self.stem(x)
        _, C, d, h, w = feats.shape
        feats = feats.reshape(B, M, C, d, h, w)
        feats = feats * eff_mask.view(B, M, 1, 1, 1, 1)

        if self.use_modality_attention:
            gap = feats.mean(dim=(3, 4, 5))
            score = gap.sum(dim=2, keepdim=True)
            score = score.masked_fill((eff_mask < 0.5).unsqueeze(-1), float("-inf"))
            row_all_neginf = torch.isinf(score).all(dim=1, keepdim=True)
            score = torch.where(row_all_neginf, torch.zeros_like(score), score)
            weights = torch.softmax(score, dim=1)
            feats = (feats * weights.view(B, M, 1, 1, 1, 1)).sum(dim=1)
        else:
            denom = eff_mask.sum(dim=1).clamp_min(1.0).view(B, 1, 1, 1, 1)
            feats = feats.sum(dim=1) / denom

        feats = self.trunk(feats)
        crop_hit_logits = self.crop_hit_head(feats)
        token_fg_logits = self.token_fg_head(feats)
        if return_dict:
            out = {"crop_hit_logits": crop_hit_logits, "token_fg_logits": token_fg_logits}
            if self.use_cross_slice_head:
                out["cross_slice_logits"] = self.cross_slice_head(feats)
            return out
        return crop_hit_logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_coarselocnet(
    num_modalities: int = 6,
    base_channels: int = 96,
    stem_channels: int = 64,
    num_res_blocks: int = 3,
    modality_dropout: float = 0.15,
    use_modality_attention: bool = True,
    norm_groups: int = 8,
    trunk_dropout: float = 0.0,
    use_residual_trunk: bool = True,
    use_cross_slice_head: bool = False,
    share_head_trunk: bool = False,
) -> CoarseLocNet:
    return CoarseLocNet(
        num_modalities=num_modalities,
        base_channels=base_channels,
        stem_channels=stem_channels,
        num_res_blocks=num_res_blocks,
        modality_dropout=modality_dropout,
        use_modality_attention=use_modality_attention,
        norm_groups=norm_groups,
        trunk_dropout=trunk_dropout,
        use_residual_trunk=use_residual_trunk,
        use_cross_slice_head=use_cross_slice_head,
        share_head_trunk=share_head_trunk,
    )
