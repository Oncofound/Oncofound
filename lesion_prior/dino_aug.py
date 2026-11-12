"""
dino_aug.py
===========
Geometric and intensity augmentations used by the lesion_prior pipeline.

The intended real DINOv3 augmentations live in DINOv3 code. For now we
provide a MONAI-compatible *fallback* implementation that is fully
usable on CPU and GPU. Both augmenters are MONAI-style
``__call__(image, seg=None)`` so they can be swapped in place when the
real DINOv3 augmenters are imported.

Conventions
-----------
* ``image`` is a torch tensor of shape ``[C, D, H, W]`` (channel-first).
* ``seg`` is a torch tensor of shape ``[1, D, H, W]`` or ``[D, H, W]``.
* For each random op, the same parameters are used for image and seg.
* ``image`` is interpolated trilinearly; ``seg`` is interpolated with nearest.
* Intensity augmentation touches image only.
"""

from __future__ import annotations

from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn.functional as F


# TODO markers for the real DINOv3 augmentations.


def _to_5d(t: torch.Tensor, is_seg: bool) -> torch.Tensor:
    """Promote ``[C, D, H, W]`` or ``[D, H, W]`` to ``[1, C, D, H, W]``."""
    if t.dim() == 3:
        t = t.unsqueeze(0)
    if t.dim() == 4:
        t = t.unsqueeze(0)
    return t


def _from_5d(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_to_5d`. Returns shape that matches ``ref``."""
    out = t[0]
    if ref.dim() == 3:
        out = out[0]
    return out


def _rand_uniform(a: float, b: float) -> float:
    return float(torch.empty(1).uniform_(a, b).item())


def _rand_bool(p: float) -> bool:
    return bool(torch.rand(1).item() < p)


class DinoV3GeometricAug:
    """DINOv3-style geometric augmentation (MONAI fallback).

    Parameters
    ----------
    roi_size : Tuple[int, int, int]
        Final spatial size after random crop / resize.
    rand_flip : bool
    rand_rotate90 : bool
    rand_affine : bool
    rand_affine_prob : float
    rand_affine_rot_range : Tuple[float, float]
    rand_affine_scale_range : Tuple[float, float]
    p : float
    """

    def __init__(
        self,
        roi_size: Tuple[int, int, int] = (128, 128, 128),
        rand_flip: bool = True,
        rand_rotate90: bool = True,
        rand_affine: bool = True,
        rand_affine_prob: float = 0.8,
        rand_affine_rot_range: Tuple[float, float] = (-0.35, 0.35),
        rand_affine_scale_range: Tuple[float, float] = (0.85, 1.15),
        p: float = 1.0,
    ) -> None:
        self.roi_size = tuple(int(x) for x in roi_size)
        self.rand_flip = rand_flip
        self.rand_rotate90 = rand_rotate90
        self.rand_affine = rand_affine
        self.rand_affine_prob = float(rand_affine_prob)
        self.rot_range = tuple(rand_affine_rot_range)
        self.scale_range = tuple(rand_affine_scale_range)
        self.p = float(p)

    def _random_crop_resize(
        self,
        img5: torch.Tensor,
        seg5: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Random crop and resize to ``self.roi_size``.

        Correctly slices the *target* size along each axis, regardless of
        the input dimensions. The previous implementation accidentally
        sliced the *input* size, which produced off-by-``roi`` shapes.
        """
        _, _, D, H, W = img5.shape
        cd, ch, cw = self.roi_size
        # Choose top-left corner in [0, max(0, D - cd)]
        if D >= cd:
            z0 = int(torch.randint(0, D - cd + 1, (1,)).item())
        else:
            z0 = 0
        if H >= ch:
            y0 = int(torch.randint(0, H - ch + 1, (1,)).item())
        else:
            y0 = 0
        if W >= cw:
            x0 = int(torch.randint(0, W - cw + 1, (1,)).item())
        else:
            x0 = 0
        # Slice exactly ``cd, ch, cw`` voxels (NOT D, H, W as before).
        img5 = img5[:, :, z0:z0 + cd, y0:y0 + ch, x0:x0 + cw]
        if seg5 is not None:
            seg5 = seg5[:, :, z0:z0 + cd, y0:y0 + ch, x0:x0 + cw]
        # Resize to roi_size with trilinear / nearest
        if img5.shape[2:] != self.roi_size:
            img5 = F.interpolate(img5, size=self.roi_size, mode="trilinear", align_corners=False)
            if seg5 is not None:
                seg5 = F.interpolate(seg5, size=self.roi_size, mode="nearest")
        return img5, seg5

    def _flip(
        self,
        img5: torch.Tensor,
        seg5: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        for axis in (2, 3, 4):
            if _rand_bool(self.p) and self.rand_flip:
                img5 = torch.flip(img5, dims=(axis,))
                if seg5 is not None:
                    seg5 = torch.flip(seg5, dims=(axis,))
        return img5, seg5

    def _rotate90(
        self,
        img5: torch.Tensor,
        seg5: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.rand_rotate90:
            return img5, seg5
        k = int(torch.randint(0, 4, (1,)).item())
        if k > 0:
            img5 = torch.rot90(img5, k=k, dims=(3, 4))
            if seg5 is not None:
                seg5 = torch.rot90(seg5, k=k, dims=(3, 4))
        return img5, seg5

    def _affine(
        self,
        img5: torch.Tensor,
        seg5: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.rand_affine or not _rand_bool(self.rand_affine_prob):
            return img5, seg5
        device = img5.device
        dtype = img5.dtype
        rx = _rand_uniform(self.rot_range[0], self.rot_range[1])
        ry = _rand_uniform(self.rot_range[0], self.rot_range[1])
        rz = _rand_uniform(self.rot_range[0], self.rot_range[1])
        s = _rand_uniform(self.scale_range[0], self.scale_range[1])
        cosx = float(torch.cos(torch.tensor(rx)).item())
        sinx = float(torch.sin(torch.tensor(rx)).item())
        cosy = float(torch.cos(torch.tensor(ry)).item())
        siny = float(torch.sin(torch.tensor(ry)).item())
        cosz = float(torch.cos(torch.tensor(rz)).item())
        sinz = float(torch.sin(torch.tensor(rz)).item())
        Rx = torch.tensor([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]], dtype=dtype, device=device)
        Ry = torch.tensor([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]], dtype=dtype, device=device)
        Rz = torch.tensor([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]], dtype=dtype, device=device)
        R = Rz @ Ry @ Rx
        S = torch.eye(3, dtype=dtype, device=device) * s
        M = R @ S
        theta = torch.zeros(3, 4, dtype=dtype, device=device)
        theta[:, :3] = M
        theta = theta.unsqueeze(0)
        grid = F.affine_grid(theta, img5.shape, align_corners=False)
        img5 = F.grid_sample(img5, grid, mode="bilinear", padding_mode="border", align_corners=False)
        if seg5 is not None:
            seg5 = F.grid_sample(seg5, grid, mode="nearest", padding_mode="border", align_corners=False)
        return img5, seg5

    def __call__(
        self,
        image: torch.Tensor,
        seg: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        img5 = _to_5d(image, is_seg=False).float()
        seg5 = _to_5d(seg, is_seg=True).float() if seg is not None else None
        img5, seg5 = self._random_crop_resize(img5, seg5)
        img5, seg5 = self._flip(img5, seg5)
        img5, seg5 = self._rotate90(img5, seg5)
        img5, seg5 = self._affine(img5, seg5)
        return _from_5d(img5, image), (_from_5d(seg5, seg) if seg is not None else None)


class DinoV3IntensityAug:
    """DINOv3-style intensity augmentation (MONAI fallback)."""

    def __init__(
        self,
        prob_gaussian_noise: float = 0.3,
        prob_gaussian_smooth: float = 0.2,
        prob_intensity: float = 0.5,
        prob_contrast: float = 0.3,
        gaussian_noise_std: Tuple[float, float] = (0.0, 0.05),
        gaussian_smooth_sigma: Tuple[float, float] = (0.5, 1.5),
        intensity_scale_range: Tuple[float, float] = (0.9, 1.1),
        intensity_shift_range: Tuple[float, float] = (-0.05, 0.05),
        contrast_gamma_range: Tuple[float, float] = (0.8, 1.25),
    ) -> None:
        self.p_gn = float(prob_gaussian_noise)
        self.p_gs = float(prob_gaussian_smooth)
        self.p_int = float(prob_intensity)
        self.p_ct = float(prob_contrast)
        self.noise_std = tuple(gaussian_noise_std)
        self.smooth_sigma = tuple(gaussian_smooth_sigma)
        self.int_scale = tuple(intensity_scale_range)
        self.int_shift = tuple(intensity_shift_range)
        self.gamma_range = tuple(contrast_gamma_range)

    def _gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        if not _rand_bool(self.p_gn):
            return x
        a, b = self.noise_std
        if b <= 0:
            return x
        std = _rand_uniform(a, b)
        if std <= 0:
            return x
        sigma = x.std(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
        noise = torch.randn_like(x) * (sigma * std)
        return x + noise

    def _gaussian_smooth(self, x: torch.Tensor) -> torch.Tensor:
        if not _rand_bool(self.p_gs):
            return x
        a, b = self.smooth_sigma
        sigma = _rand_uniform(a, b)
        if sigma <= 0:
            return x
        k = 2 * int(round(3.0 * sigma)) + 1
        coords = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()  # shape (k,)
        channels = x.shape[1]
        # depth axis = -3
        kernel_d = g.view(1, 1, k, 1, 1).expand(channels, 1, k, 1, 1).contiguous()
        x = F.conv3d(x, kernel_d, padding=(k // 2, 0, 0), groups=channels)
        # height axis = -2
        kernel_h = g.view(1, 1, 1, k, 1).expand(channels, 1, 1, k, 1).contiguous()
        x = F.conv3d(x, kernel_h, padding=(0, k // 2, 0), groups=channels)
        # width axis = -1
        kernel_w = g.view(1, 1, 1, 1, k).expand(channels, 1, 1, 1, k).contiguous()
        x = F.conv3d(x, kernel_w, padding=(0, 0, k // 2), groups=channels)
        return x

    def _intensity(self, x: torch.Tensor) -> torch.Tensor:
        if not _rand_bool(self.p_int):
            return x
        s = _rand_uniform(self.int_scale[0], self.int_scale[1])
        t = _rand_uniform(self.int_shift[0], self.int_shift[1])
        return x * s + t

    def _contrast(self, x: torch.Tensor) -> torch.Tensor:
        if not _rand_bool(self.p_ct):
            return x
        gamma = _rand_uniform(self.gamma_range[0], self.gamma_range[1])
        if gamma <= 0:
            return x
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        out = (x - mean) * 1.0
        out = out.sign() * out.abs().clamp_min(0) ** gamma
        out = out + mean
        return out

    def __call__(
        self,
        image: torch.Tensor,
        seg: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = _to_5d(image, is_seg=False).float()
        x = self._gaussian_noise(x)
        x = self._gaussian_smooth(x)
        x = self._intensity(x)
        x = self._contrast(x)
        return _from_5d(x, image), seg
