"""integral_image.py
================
3D summed-area table for fast O(1) box tumor-voxel counting."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

from .crop_select import CropBox


def _build_prefix_3d(mask_bool: torch.Tensor) -> torch.Tensor:
    """Return ``(D+1, H+1, W+1)`` int64 prefix-sum with a leading zero slab.

    Accepts ``(D, H, W)`` bool tensor.
    """
    if mask_bool.dtype != torch.bool:
        mask_bool = mask_bool.bool()
    # Convert to int32 (small enough per voxel) before cumsum to avoid
    # surprising int8 promotions.
    m = mask_bool.to(torch.int32)
    p = m.cumsum(dim=0).cumsum(dim=1).cumsum(dim=2)
    # Prepend a zero plane on each axis.
    D, H, W = p.shape
    P = torch.zeros((D + 1, H + 1, W + 1), dtype=torch.int64, device=p.device)
    P[1:, 1:, 1:] = p.to(torch.int64)
    return P


def _query_one(P: torch.Tensor,
               z1: int, y1: int, x1: int,
               z2: int, y2: int, x2: int) -> int:
    return int(
        P[z2, y2, x2].item()
        - P[z1, y2, x2].item()
        - P[z2, y1, x2].item()
        - P[z2, y2, x1].item()
        + P[z1, y1, x2].item()
        + P[z1, y2, x1].item()
        + P[z2, y1, x1].item()
        - P[z1, y1, x1].item()
    )


def _query_batched(P: torch.Tensor,
                   z1: torch.Tensor, y1: torch.Tensor, x1: torch.Tensor,
                   z2: torch.Tensor, y2: torch.Tensor, x2: torch.Tensor
                   ) -> torch.Tensor:
    """Batched version of the 8-corner query.

    Inputs are 1-D long tensors of equal length; output is a 1-D int64
    tensor of the same length (still on the same device as ``P``).
    """
    return (
        P[z2, y2, x2]
        - P[z1, y2, x2]
        - P[z2, y1, x2]
        - P[z2, y2, x1]
        + P[z1, y1, x2]
        + P[z1, y2, x1]
        + P[z2, y1, x1]
        - P[z1, y1, x1]
    )


class IntegralVolume3D:
    """Wrap a (D, H, W) bool/int tensor as a queryable integral volume.

    Construction cost is ``O(D*H*W)`` (three cumsums).
    Query cost for one box is ``O(1)``; for ``K`` boxes is
    ``O(K)`` (vectorised).

    For tiny boxes ``K <= 4`` we use the per-box path which avoids the
    overhead of building 6-element index tensors; for larger ``K`` we
    batch.
    """

    __slots__ = ("P", "shape", "device")

    def __init__(self, seg: torch.Tensor) -> None:
        if seg.dim() == 4:
            seg = seg[0]
        if seg.dim() != 3:
            raise ValueError(f"seg must be (D,H,W) or (1,D,H,W), got {tuple(seg.shape)}")
        bool_mask = seg > 0
        self.P = _build_prefix_3d(bool_mask)
        self.shape = tuple(seg.shape)
        self.device = self.P.device

    def query(self,
              z1: int, y1: int, x1: int,
              z2: int, y2: int, x2: int) -> int:
        # Clamp to volume.
        z2 = max(z1, min(z2, self.shape[0]))
        y2 = max(y1, min(y2, self.shape[1]))
        x2 = max(x1, min(x2, self.shape[2]))
        return _query_one(self.P, z1, y1, x1, z2, y2, x2)

    def query_boxes(self, boxes: Sequence[CropBox]) -> torch.Tensor:
        """Vectorised query over ``boxes``."""
        n = len(boxes)
        if n == 0:
            return torch.zeros(0, dtype=torch.int64, device=self.device)
        z1 = torch.tensor([b.z1 for b in boxes], dtype=torch.long, device=self.device)
        y1 = torch.tensor([b.y1 for b in boxes], dtype=torch.long, device=self.device)
        x1 = torch.tensor([b.x1 for b in boxes], dtype=torch.long, device=self.device)
        z2 = torch.tensor([b.z2 for b in boxes], dtype=torch.long, device=self.device)
        y2 = torch.tensor([b.y2 for b in boxes], dtype=torch.long, device=self.device)
        x2 = torch.tensor([b.x2 for b in boxes], dtype=torch.long, device=self.device)
        # Clamp upper bounds to the volume (1 past the last index).
        z2 = torch.clamp(z2, min=0, max=self.shape[0])
        y2 = torch.clamp(y2, min=0, max=self.shape[1])
        x2 = torch.clamp(x2, min=0, max=self.shape[2])
        return _query_batched(self.P, z1, y1, x1, z2, y2, x2)

    def query_mask_per_box(self,
                           boxes: Sequence[CropBox],
                           volume_dim: Tuple[int, int, int],
                           ) -> torch.Tensor:
        """For each box, return a bool mask ``(n_boxes, *volume_dim)``
        indicating which voxels fall inside the box. Cheaper than a
        per-component check when ``n_components >> n_boxes``.

        Used by the component-aware metrics.
        """
        D, H, W = volume_dim
        out = torch.zeros((len(boxes), D, H, W), dtype=torch.bool, device=self.device)
        for i, b in enumerate(boxes):
            z2 = max(0, min(b.z2, D))
            y2 = max(0, min(b.y2, H))
            x2 = max(0, min(b.x2, W))
            z1 = max(0, min(b.z1, D))
            y1 = max(0, min(b.y1, H))
            x1 = max(0, min(b.x1, W))
            out[i, z1:z2, y1:y2, x1:x2] = True
        return out


def count_tumor_voxels_in_box_integral(integral: IntegralVolume3D,
                                       box: CropBox) -> int:
    """Drop-in replacement for the legacy ``count_tumor_voxels_in_box``
    that uses the precomputed integral volume.
    """
    return integral.query(box.z1, box.y1, box.x1, box.z2, box.y2, box.x2)