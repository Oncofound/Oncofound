# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
import random

import math
import numpy as np
import torch
from torch import nn
from torch.nn.functional import interpolate
from torchvision.transforms import v2
import monai.transforms as v3
from monai.data.utils import get_valid_patch_size, get_random_patch
from monai.utils import InterpolateMode

from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, GaussianBlur, make_normalize_transform

logger = logging.getLogger("dinov3")


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        gram_teacher_crops_size=None,
        gram_teacher_no_distortions=False,
        teacher_no_color_jitter=False,
        local_crops_subset_of_global_crops=False,
        patch_size=16,
        share_color_jitter=False,
        horizontal_flips=True,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.gram_teacher_crops_size = gram_teacher_crops_size
        self.gram_teacher_no_distortions = gram_teacher_no_distortions
        self.teacher_no_color_jitter = teacher_no_color_jitter
        self.local_crops_subset_of_global_crops = local_crops_subset_of_global_crops
        self.patch_size = patch_size
        self.share_color_jitter = share_color_jitter
        self.mean = mean
        self.std = std

        # logger.info("###################################")
        # logger.info("Using data augmentation parameters:")
        # logger.info(f"global_crops_scale: {global_crops_scale}")
        # logger.info(f"local_crops_scale: {local_crops_scale}")
        # logger.info(f"local_crops_number: {local_crops_number}")
        # logger.info(f"global_crops_size: {global_crops_size}")
        # logger.info(f"local_crops_size: {local_crops_size}")
        # logger.info(f"gram_crops_size: {gram_teacher_crops_size}")
        # logger.info(f"gram_teacher_no_distortions: {gram_teacher_no_distortions}")
        # logger.info(f"teacher_no_color_jitter: {teacher_no_color_jitter}")
        # logger.info(f"local_crops_subset_of_global_crops: {local_crops_subset_of_global_crops}")
        # logger.info(f"patch_size if local_crops_subset_of_global_crops: {patch_size}")
        # logger.info(f"share_color_jitter: {share_color_jitter}")
        # logger.info(f"horizontal flips: {horizontal_flips}")
        # logger.info("###################################")

        # Global crops and gram teacher crops can have different sizes. We first take a crop of the maximum size
        # and then resize it to the desired size for global and gram teacher crops.
        global_crop_max_size = max(global_crops_size, gram_teacher_crops_size if gram_teacher_crops_size else 0)

        # random resized crop and flip
        self.geometric_augmentation_global = v2.Compose(
            [
                v2.RandomResizedCrop(
                    global_crop_max_size,
                    scale=global_crops_scale,
                    interpolation=v2.InterpolationMode.BICUBIC,
                ),
                v2.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
        )

        resize_global = nn.Identity()  # Resize transform applied to global crops after random crop
        self.resize_global_post_transf = (
            nn.Identity()
        )  # Resize transform applied to global crops after all other transforms
        self.resize_gram_teacher = None  # Resize transform applied to crops for gram teacher
        if gram_teacher_crops_size is not None:
            # All resize transforms will do nothing if the crop size is already the desired size.
            if gram_teacher_no_distortions:
                # When there a no distortions for the gram teacher crop, we can resize before the distortions.
                # This is the preferred order, because it keeps the image size for the augmentations consistent,
                # which matters e.g. for GaussianBlur.
                resize_global = v2.Resize(
                    global_crops_size,
                    interpolation=v2.InterpolationMode.BICUBIC,
                )
            else:
                # When there a no distortions for the gram teacher crop, we need to resize after the distortions,
                # because the distortions are shared between global and gram teacher crops.
                self.resize_global_post_transf = v2.Resize(
                    global_crops_size,
                    interpolation=v2.InterpolationMode.BICUBIC,
                )

            self.resize_gram_teacher = v2.Resize(
                gram_teacher_crops_size,
                interpolation=v2.InterpolationMode.BICUBIC,
            )

        self.geometric_augmentation_local = v2.Compose(
            [
                v2.RandomResizedCrop(
                    local_crops_size,
                    scale=local_crops_scale,
                    interpolation=v2.InterpolationMode.BICUBIC,
                ),
                v2.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
        )

        # color distortions / blurring
        color_jittering = v2.Compose(
            [
                v2.RandomApply(
                    [v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                v2.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = v2.Compose(
            [
                GaussianBlur(p=0.1),
                v2.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                make_normalize_transform(mean=mean, std=std),
            ]
        )

        if self.share_color_jitter:
            self.color_jittering = color_jittering
            self.global_transfo1 = v2.Compose([resize_global, global_transfo1_extra, self.normalize])
            self.global_transfo2 = v2.Compose([resize_global, global_transfo2_extra, self.normalize])
            self.local_transfo = v2.Compose([local_transfo_extra, self.normalize])
        else:
            self.global_transfo1 = v2.Compose(
                [resize_global, color_jittering, global_transfo1_extra, self.normalize]
            )
            self.global_transfo2 = v2.Compose(
                [resize_global, color_jittering, global_transfo2_extra, self.normalize]
            )
            self.local_transfo = v2.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image):
        output = {}
        output["weak_flag"] = True  # some residual from mugs

        if self.share_color_jitter:
            image = self.color_jittering(image)

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1_transf = self.global_transfo1(im1_base)
        global_crop_1 = self.resize_global_post_transf(global_crop_1_transf)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2_transf = self.global_transfo2(im2_base)
        global_crop_2 = self.resize_global_post_transf(global_crop_2_transf)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        if self.teacher_no_color_jitter:
            output["global_crops_teacher"] = [
                self.normalize(im1_base),
                self.normalize(im2_base),
            ]
        else:
            output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        if self.gram_teacher_crops_size is not None:
            # crops for gram teacher:
            if self.gram_teacher_no_distortions:
                gram_crop_1 = self.normalize(self.resize_gram_teacher(im1_base))
                gram_crop_2 = self.normalize(self.resize_gram_teacher(im2_base))
            else:
                gram_crop_1 = self.resize_gram_teacher(global_crop_1_transf)
                gram_crop_2 = self.resize_gram_teacher(global_crop_2_transf)
            output["gram_teacher_crops"] = [gram_crop_1, gram_crop_2]

        # local crops:
        if self.local_crops_subset_of_global_crops:
            _local_crops = [self.local_transfo(im1_base) for _ in range(self.local_crops_number // 2)] + [
                self.local_transfo(im2_base) for _ in range(self.local_crops_number // 2)
            ]

            local_crops = []
            offsets = []
            gs = self.global_crops_size
            ls = self.local_crops_size
            for img in _local_crops:
                rx, ry = np.random.randint(0, (gs - ls) // self.patch_size, 2) * self.patch_size
                local_crops.append(img[:, rx : rx + ls, ry : ry + ls])
                offsets.append((rx, ry))

            output["local_crops"] = local_crops
            output["offsets"] = offsets
        else:
            local_crops = [
                self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
            ]
            output["local_crops"] = local_crops
            output["offsets"] = ()

        return output


def _compute_gt_bbox_coverage(crop_slices, gt_bbox):
    """Return fraction of gt_bbox volume covered by crop_slices. Returns 1.0 if gt_bbox is None."""
    if gt_bbox is None:
        return 1.0
    bh0, bh1, bw0, bw1, bd0, bd1 = [int(v) for v in gt_bbox]
    sh = crop_slices[0]
    sw = crop_slices[1]
    sd = crop_slices[2]
    oh = max(0, min(sh.stop, bh1) - max(sh.start, bh0))
    ow = max(0, min(sw.stop, bw1) - max(sw.start, bw0))
    od = max(0, min(sd.stop, bd1) - max(sd.start, bd0))
    gt_vol = max(1, (bh1 - bh0) * (bw1 - bw0) * (bd1 - bd0))
    return (oh * ow * od) / gt_vol


def _compute_connected_components_bboxes(mask_hwd):
    """Return list of (h0,h1,w0,w1,d0,d1) bboxes for each connected component in mask."""
    try:
        from scipy import ndimage as ndi
    except ImportError:
        bbox = _compute_gt_bbox_from_mask(mask_hwd)
        return [bbox] if bbox is not None else []
    m = _as_bool_mask_hwd(mask_hwd)
    if m is None:
        return []
    m_np = m.numpy() if torch.is_tensor(m) else np.array(m)
    labeled, n = ndi.label(m_np)
    bboxes = []
    for i in range(1, n + 1):
        coords = np.argwhere(labeled == i)
        if coords.size == 0:
            continue
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        bboxes.append((int(mins[0]), int(maxs[0]) + 1,
                       int(mins[1]), int(maxs[1]) + 1,
                       int(mins[2]), int(maxs[2]) + 1))
    return bboxes


def _as_bool_mask_hwd(x):
    if x is None:
        return None
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 3:
        return None
    return (x > 0)


def _compute_gt_bbox_from_mask(mask_hwd):
    m = _as_bool_mask_hwd(mask_hwd)
    if m is None:
        return None
    coords = torch.nonzero(m, as_tuple=False)
    if coords.numel() == 0:
        return None
    mins = torch.min(coords.to(torch.int64), dim=0).values.tolist()
    maxs = torch.max(coords.to(torch.int64), dim=0).values.tolist()
    h0, w0, d0 = [int(v) for v in mins]
    h1, w1, d1 = [int(v) + 1 for v in maxs]
    return h0, h1, w0, w1, d0, d1


def _enforce_crop_size_contains_bbox(crop_size, gt_bbox, img_size):
    h, w, d = [int(v) for v in crop_size]
    H, W, D = [int(v) for v in img_size]
    if gt_bbox is not None:
        bh0, bh1, bw0, bw1, bd0, bd1 = [int(v) for v in gt_bbox]
        h = max(h, bh1 - bh0)
        w = max(w, bw1 - bw0)
        d = max(d, bd1 - bd0)
    return min(h, H), min(w, W), min(d, D)


def _sample_start_containing_interval(bbox_start, bbox_end, crop_size, total, rng):
    crop_size = max(1, min(int(crop_size), int(total)))
    if total <= crop_size:
        return 0
    lo = max(0, int(bbox_end) - crop_size)
    hi = min(int(bbox_start), int(total) - crop_size)
    if lo > hi:
        center = (int(bbox_start) + int(bbox_end) - 1) // 2
        return max(0, min(center - crop_size // 2, total - crop_size))
    return int(rng.randint(lo, hi))


class RandomResizedCrop3DSetting2(v3.Randomizable, v3.Crop):
    """Setting2: sample with original scale/position, retry up to max_tries times checking
    GT coverage thresholds. Local crops must cover >=50% of GT; global crops must cover 100%.
    Forces acceptance on the last attempt.
    """
    def __init__(
        self,
        size,
        inner_slice_scale,
        cross_slice_scale,
        ratio=(0.9, 1 / 0.9),
        interpolation='trilinear',
        lazy=False,
        is_global=False,
        max_tries=3,
    ):
        super().__init__(lazy)
        self.size = size
        self.inner_slice_scale = inner_slice_scale
        self.cross_slice_scale = cross_slice_scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.is_global = is_global
        self.max_tries = max_tries
        self._slices: tuple = ()

    def get_inner_slice_crop(self, height, width):
        area = height * width
        log_ratio = math.log(self.ratio[0]), math.log(self.ratio[1])
        for _ in range(10):
            target_area = area * self.R.uniform(*self.inner_slice_scale)
            ratio = math.exp(self.R.uniform(*log_ratio))
            w = int(round(math.sqrt(target_area * ratio)))
            h = int(round(math.sqrt(target_area / ratio)))
            if 0 < w <= width and 0 < h <= height:
                return h, w
        in_ratio = float(width) / float(height)
        if in_ratio < min(self.ratio):
            w = width
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = height
            w = int(round(h * max(self.ratio)))
        else:
            w = width
            h = height
        return h, w

    def randomize(self, img_size, gt_bbox=None, seed=None):
        height, width, depth = img_size
        threshold = 1.0 if self.is_global else 0.5

        best_slices = None
        for attempt in range(self.max_tries):
            crop_h, crop_w = self.get_inner_slice_crop(height, width)
            crop_d = int(round(depth * self.R.uniform(*self.cross_slice_scale)))
            crop_size = (crop_h, crop_w, crop_d)
            valid_size = get_valid_patch_size(img_size, crop_size)
            slices = get_random_patch(img_size, valid_size, self.R)

            if gt_bbox is None:
                self._slices = slices
                return

            coverage = _compute_gt_bbox_coverage(slices, gt_bbox)
            if coverage >= threshold:
                self._slices = slices
                return
            if best_slices is None:
                best_slices = slices

        # Force: use bbox-anchored crop on final attempt
        crop_h, crop_w = self.get_inner_slice_crop(height, width)
        crop_d = int(round(depth * self.R.uniform(*self.cross_slice_scale)))
        crop_size = _enforce_crop_size_contains_bbox((crop_h, crop_w, crop_d), gt_bbox, img_size)
        valid_size = get_valid_patch_size(img_size, crop_size)
        bh0, bh1, bw0, bw1, bd0, bd1 = [int(v) for v in gt_bbox]
        ch, cw, cd = [int(v) for v in valid_size]
        rng = random.Random(int(self.R.randint(0, 2**31 - 1)))
        h0 = _sample_start_containing_interval(bh0, bh1, ch, height, rng)
        w0 = _sample_start_containing_interval(bw0, bw1, cw, width, rng)
        d0 = _sample_start_containing_interval(bd0, bd1, cd, depth, rng)
        self._slices = (slice(h0, h0 + ch), slice(w0, w0 + cw), slice(d0, d0 + cd))

    def __call__(self, img, lazy=False, gt_bbox=None, seed=None):
        self.randomize(img.shape[1:], gt_bbox=gt_bbox, seed=seed)
        lazy_ = self.lazy if lazy is None else lazy
        cropped = super().__call__(img=img, slices=self._slices, lazy=lazy_)
        resized = interpolate(cropped.unsqueeze(0), size=self.size, mode=self.interpolation).squeeze(0)
        return resized


class GTCoverageCrop3D(v3.Randomizable, v3.Crop):
    """Setting3: CropForeground-style crop anchored to GT bbox, then verify 50%-100% GT coverage.
    Equivalent to RandCropByPosNegLabel logic: random scale [0.5,1.0] of GT bbox extent,
    positioned to include GT, retried up to max_tries times.
    """
    def __init__(
        self,
        size,
        inner_slice_scale,
        cross_slice_scale,
        ratio=(0.9, 1 / 0.9),
        interpolation='trilinear',
        lazy=False,
        coverage_range=(0.5, 1.0),
        max_tries=10,
    ):
        super().__init__(lazy)
        self.size = size
        self.inner_slice_scale = inner_slice_scale
        self.cross_slice_scale = cross_slice_scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.coverage_range = coverage_range
        self.max_tries = max_tries
        self._slices: tuple = ()

    def get_inner_slice_crop(self, height, width):
        area = height * width
        log_ratio = math.log(self.ratio[0]), math.log(self.ratio[1])
        for _ in range(10):
            target_area = area * self.R.uniform(*self.inner_slice_scale)
            ratio = math.exp(self.R.uniform(*log_ratio))
            w = int(round(math.sqrt(target_area * ratio)))
            h = int(round(math.sqrt(target_area / ratio)))
            if 0 < w <= width and 0 < h <= height:
                return h, w
        in_ratio = float(width) / float(height)
        if in_ratio < min(self.ratio):
            w = width
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = height
            w = int(round(h * max(self.ratio)))
        else:
            w = width
            h = height
        return h, w

    def randomize(self, img_size, gt_bbox=None, seed=None):
        height, width, depth = img_size

        if gt_bbox is None:
            crop_h, crop_w = self.get_inner_slice_crop(height, width)
            crop_d = int(round(depth * self.R.uniform(*self.cross_slice_scale)))
            valid_size = get_valid_patch_size(img_size, (crop_h, crop_w, crop_d))
            self._slices = get_random_patch(img_size, valid_size, self.R)
            return

        bh0, bh1, bw0, bw1, bd0, bd1 = [int(v) for v in gt_bbox]
        gt_h = bh1 - bh0
        gt_w = bw1 - bw0
        gt_d = bd1 - bd0

        for _ in range(self.max_tries):
            # Scale crop to cover a random fraction of GT extent
            scale = self.R.uniform(*self.coverage_range)
            crop_h = max(int(round(gt_h / scale)), 1)
            crop_w = max(int(round(gt_w / scale)), 1)
            crop_d = max(int(round(gt_d / scale)), 1)
            crop_size = _enforce_crop_size_contains_bbox((crop_h, crop_w, crop_d), gt_bbox, img_size)
            valid_size = get_valid_patch_size(img_size, crop_size)
            ch, cw, cd = [int(v) for v in valid_size]
            rng = random.Random(int(self.R.randint(0, 2**31 - 1)))
            h0 = _sample_start_containing_interval(bh0, bh1, ch, height, rng)
            w0 = _sample_start_containing_interval(bw0, bw1, cw, width, rng)
            d0 = _sample_start_containing_interval(bd0, bd1, cd, depth, rng)
            slices = (slice(h0, h0 + ch), slice(w0, w0 + cw), slice(d0, d0 + cd))
            coverage = _compute_gt_bbox_coverage(slices, gt_bbox)
            if self.coverage_range[0] <= coverage <= self.coverage_range[1] + 1e-6:
                self._slices = slices
                return

        # Fallback: use last computed slices
        self._slices = slices

    def __call__(self, img, lazy=False, gt_bbox=None, seed=None):
        self.randomize(img.shape[1:], gt_bbox=gt_bbox, seed=seed)
        lazy_ = self.lazy if lazy is None else lazy
        cropped = super().__call__(img=img, slices=self._slices, lazy=lazy_)
        resized = interpolate(cropped.unsqueeze(0), size=self.size, mode=self.interpolation).squeeze(0)
        return resized


class RandomResizedCrop3D(v3.Randomizable, v3.Crop):
    """Adapt torch RandomResizedCrop to 3D data by separating inner-slice and cross-slice dimensions.

    Args:
        size: expected output size of the crop.
        inner_slice_scale: lower and upper bounds for the inner-slice dimensions of the crop.
        cross_slice_scale: lower and upper bounds for the cross-slice dimensions of the crop.
        ratio: lower and upper bounds for the random aspect ratio of the inner-slice dimensions.
        interpolation: desired 3D interpolation method, default is 'trilinear'.
        lazy: a flag to indicate whether this transform should execute lazily or not.
    """
    def __init__(
        self,
        size,
        inner_slice_scale,
        cross_slice_scale,
        ratio=(0.9, 1/0.9),
        interpolation='trilinear',
        lazy=False,
    ):
        super().__init__(lazy)
        self.size = size
        self.inner_slice_scale = inner_slice_scale
        self.cross_slice_scale = cross_slice_scale
        self.ratio = ratio
        self.interpolation = interpolation
        self._slices: tuple[slice, ...] = ()

    def get_inner_slice_crop(self, height, width):
        """
        Adapted from torchvision RandomResizedCrop, applied to the inner-slice dimensions
        """
        area = height * width

        log_ratio = math.log(self.ratio[0]), math.log(self.ratio[1])
        for _ in range(10):
            target_area = area * self.R.uniform(*self.inner_slice_scale)
            ratio = math.exp(self.R.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * ratio)))
            h = int(round(math.sqrt(target_area / ratio)))

            if 0 < w <= width and 0 < h <= height:
                return h, w

        in_ratio = float(width) / float(height)
        if in_ratio < min(self.ratio):
            w = width
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = height
            w = int(round(h * max(self.ratio)))
        else:  # whole image
            w = width
            h = height
        return h, w

    def randomize(self, img_size, gt_bbox=None, seed=None):
        height, width, depth = img_size

        crop_h, crop_w = self.get_inner_slice_crop(height, width)
        crop_d = int(round(depth * self.R.uniform(*self.cross_slice_scale)))

        crop_size = (crop_h, crop_w, crop_d)
        crop_size = _enforce_crop_size_contains_bbox(crop_size, gt_bbox, img_size)
        valid_size = get_valid_patch_size(img_size, crop_size)

        if gt_bbox is None:
            self._slices = get_random_patch(img_size, valid_size, self.R)
            return

        bh0, bh1, bw0, bw1, bd0, bd1 = [int(v) for v in gt_bbox]
        ch, cw, cd = [int(v) for v in valid_size]
        rng = random.Random(int(seed) if seed is not None else int(self.R.randint(0, 2**31 - 1)))
        h0 = _sample_start_containing_interval(bh0, bh1, ch, height, rng)
        w0 = _sample_start_containing_interval(bw0, bw1, cw, width, rng)
        d0 = _sample_start_containing_interval(bd0, bd1, cd, depth, rng)
        self._slices = (slice(h0, h0 + ch), slice(w0, w0 + cw), slice(d0, d0 + cd))

    def __call__(self, img, lazy=False, gt_bbox=None, seed=None):
        self.randomize(img.shape[1:], gt_bbox=gt_bbox, seed=seed)
        lazy_ = self.lazy if lazy is None else lazy
        cropped = super().__call__(img=img, slices=self._slices, lazy=lazy_)
        resized = interpolate(cropped.unsqueeze(0), size=self.size, mode=self.interpolation).squeeze(0)
        return resized


class Resize3D():
    def __init__(self, size, interpolation='trilinear'):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        return interpolate(img.unsqueeze(0), size=self.size, mode=self.interpolation).squeeze(0)


def proposal_centers_to_point_bboxes(proposal_centers, img_size):
    """Pack a list of (h, w, d) centres into single-voxel bboxes.

    The bboxes are then fed to :class:`RandomResizedCrop3D`, which
    *only* uses them to ensure the crop contains the centre; the
    XY scale, XY ratio, Z scale and start position are still sampled
    randomly per call.

    The function silently drops NaN/Inf, malformed entries, and
    duplicates. Out-of-range coordinates are clamped to the image
    shape. Returns an empty list if no valid centre is provided.
    """
    if proposal_centers is None:
        return []
    centers = torch.as_tensor(proposal_centers, dtype=torch.float32)
    if centers.numel() == 0:
        return []
    if centers.ndim == 1:
        centers = centers.unsqueeze(0)
    if centers.ndim != 2 or centers.shape[1] != 3:
        return []
    height, width, depth = [int(v) for v in img_size]
    if height <= 0 or width <= 0 or depth <= 0:
        return []
    output = []
    seen = set()
    for center in centers:
        if not bool(torch.isfinite(center).all()):
            continue
        h = int(round(float(center[0])))
        w = int(round(float(center[1])))
        d = int(round(float(center[2])))
        h = max(0, min(h, height - 1))
        w = max(0, min(w, width - 1))
        d = max(0, min(d, depth - 1))
        key = (h, w, d)
        if key in seen:
            continue
        seen.add(key)
        output.append((h, h + 1, w, w + 1, d, d + 1))
    return output


class DataAugmentation3DINO(object):
    def __init__(
        self,
        global_crops_inner_slice_scale,
        global_crops_cross_slice_scale,
        local_crops_inner_slice_scale,
        local_crops_cross_slice_scale,
        local_crops_number,
        global_crops_size=96,
        local_crops_size=48,
        gram_teacher_crops_size=None,
        gram_teacher_no_distortions=False,
        teacher_no_color_jitter=False,
        local_crops_subset_of_global_crops=False,
        patch_size=16,
        share_color_jitter=False,
        flips_3d=True,
        crop_setting=1,
    ):
        self.global_crops_inner_slice_scale = global_crops_inner_slice_scale
        self.global_crops_cross_slice_scale = global_crops_cross_slice_scale
        self.local_crops_inner_slice_scale = local_crops_inner_slice_scale
        self.local_crops_cross_slice_scale = local_crops_cross_slice_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.gram_teacher_crops_size = gram_teacher_crops_size
        self.gram_teacher_no_distortions = gram_teacher_no_distortions
        self.teacher_no_color_jitter = teacher_no_color_jitter
        self.local_crops_subset_of_global_crops = local_crops_subset_of_global_crops
        self.patch_size = patch_size
        self.share_color_jitter = share_color_jitter
        self.crop_setting = crop_setting

        # logger.info("###################################")
        # logger.info("Using data augmentation parameters:")
        # logger.info(f"global_crops_inner_slice_scale: {global_crops_inner_slice_scale}")
        # logger.info(f"global_crops_cross_slice_scale: {global_crops_cross_slice_scale}")
        # logger.info(f"local_crops_inner_slice_scale: {local_crops_inner_slice_scale}")
        # logger.info(f"local_crops_cross_slice_scale: {local_crops_cross_slice_scale}")
        # logger.info(f"local_crops_number: {local_crops_number}")
        # logger.info(f"global_crops_size: {global_crops_size}")
        # logger.info(f"local_crops_size: {local_crops_size}")
        # logger.info(f"gram_crops_size: {gram_teacher_crops_size}")
        # logger.info(f"gram_teacher_no_distortions: {gram_teacher_no_distortions}")
        # logger.info(f"teacher_no_color_jitter: {teacher_no_color_jitter}")
        # logger.info(f"local_crops_subset_of_global_crops: {local_crops_subset_of_global_crops}")
        # logger.info(f"patch_size if local_crops_subset_of_global_crops: {patch_size}")
        # logger.info(f"share_color_jitter: {share_color_jitter}")
        # logger.info(f"flips_3d: {flips_3d}")
        # logger.info(f"crop_setting: {crop_setting}")
        # logger.info("###################################")

        # Global crops and gram teacher crops can have different sizes. We first take a crop of the maximum size
        # and then resize it to the desired size for global and gram teacher crops.
        global_crops_max_size = max(global_crops_size, gram_teacher_crops_size if gram_teacher_crops_size else 0)

        flips = [
            v3.RandFlip(prob=0.3 if flips_3d else 0.0, spatial_axis=0),
            v3.RandFlip(prob=0.3 if flips_3d else 0.0, spatial_axis=1),
            v3.RandFlip(prob=0.3 if flips_3d else 0.0, spatial_axis=2),
        ]

        if crop_setting == 1:
            self.geometric_augmentation_global = [
                RandomResizedCrop3D(
                    global_crops_max_size,
                    inner_slice_scale=global_crops_inner_slice_scale,
                    cross_slice_scale=global_crops_cross_slice_scale,
                    interpolation=InterpolateMode.TRILINEAR,
                ),
            ] + flips
            self.geometric_augmentation_local = [
                RandomResizedCrop3D(
                    local_crops_size,
                    inner_slice_scale=local_crops_inner_slice_scale,
                    cross_slice_scale=local_crops_cross_slice_scale,
                    interpolation=InterpolateMode.TRILINEAR,
                ),
            ] + flips

        elif crop_setting == 2:
            self.geometric_augmentation_global = [
                RandomResizedCrop3DSetting2(
                    global_crops_max_size,
                    inner_slice_scale=global_crops_inner_slice_scale,
                    cross_slice_scale=global_crops_cross_slice_scale,
                    interpolation=InterpolateMode.TRILINEAR,
                    is_global=True,
                    max_tries=3,
                ),
            ] + flips
            self.geometric_augmentation_local = [
                RandomResizedCrop3DSetting2(
                    local_crops_size,
                    inner_slice_scale=local_crops_inner_slice_scale,
                    cross_slice_scale=local_crops_cross_slice_scale,
                    interpolation=InterpolateMode.TRILINEAR,
                    is_global=False,
                    max_tries=3,
                ),
            ] + flips

        elif crop_setting == 3:
            self.geometric_augmentation_global = [
                RandomResizedCrop3D(
                    global_crops_max_size,
                    inner_slice_scale=global_crops_inner_slice_scale,
                    cross_slice_scale=global_crops_cross_slice_scale,
                    interpolation=InterpolateMode.TRILINEAR,
                ),
            ] + flips
            self.geometric_augmentation_local = [
                GTCoverageCrop3D(
                    local_crops_size,
                    inner_slice_scale=local_crops_inner_slice_scale,
                    cross_slice_scale=local_crops_cross_slice_scale,
                    interpolation=InterpolateMode.TRILINEAR,
                    coverage_range=(0.5, 1.0),
                    max_tries=10,
                ),
            ] + flips

        elif crop_setting == 4:
            # setting4 uses standard crop primitives; component logic is in __call__
            self.geometric_augmentation_global = [
                RandomResizedCrop3D(
                    global_crops_max_size,
                    inner_slice_scale=global_crops_inner_slice_scale,
                    cross_slice_scale=global_crops_cross_slice_scale,
                    interpolation=InterpolateMode.TRILINEAR,
                ),
            ] + flips
            self.geometric_augmentation_local = [
                RandomResizedCrop3D(
                    local_crops_size,
                    inner_slice_scale=local_crops_inner_slice_scale,
                    cross_slice_scale=local_crops_cross_slice_scale,
                    interpolation=InterpolateMode.TRILINEAR,
                ),
            ] + flips

        else:
            raise ValueError(f"Unknown crop_setting={crop_setting}. Must be 1, 2, 3, or 4.")

        resize_global = nn.Identity()  # Resize transform applied to global crops after random crop
        self.resize_global_post_transf = (
            nn.Identity()
        )  # Resize transform applied to global crops after all other transforms
        self.resize_gram_teacher = None  # Resize transform applied to crops for gram teacher
        if gram_teacher_crops_size is not None:
            if gram_teacher_no_distortions:
                resize_global = Resize3D(
                    global_crops_size,
                    interpolation=InterpolateMode.TRILINEAR,
                )
            else:
                self.resize_global_post_transf = Resize3D(
                    global_crops_size,
                    interpolation=InterpolateMode.TRILINEAR,
                )

            self.resize_gram_teacher = Resize3D(
                gram_teacher_crops_size,
                interpolation=InterpolateMode.TRILINEAR,
            )

        # contrast, noise, blurring
        color_jittering = v3.OneOf(
            [
                v3.RandAdjustContrast(prob=0.8, gamma=(0.5, 2)),
                v3.RandGaussianNoise(prob=0.8, std=0.002),
                v3.RandHistogramShift(prob=0.8, num_control_points=10),
            ]
        )

        global_transfo1_extra = v3.OneOf(
            [
                v3.RandGaussianSmooth(prob=1.0),
                v3.RandGaussianSharpen(prob=1.0),
            ]
        )

        global_transfo2_extra = v3.Compose(
            [
                v3.OneOf(
                    [
                        v3.RandGaussianSmooth(prob=0.1),
                        v3.RandGaussianSharpen(prob=0.1),
                    ]
                ),
                v3.RandGibbsNoise(prob=0.2)
            ]
        )

        local_transfo_extra = v3.RandGaussianSmooth(prob=0.5)

        # normalization
        self.normalize = v3.ToTensor(dtype=torch.float32, track_meta=False)

        if self.share_color_jitter:
            self.color_jittering = color_jittering
            self.global_transfo1 = v3.Compose([resize_global, global_transfo1_extra, self.normalize])
            self.global_transfo2 = v3.Compose([resize_global, global_transfo2_extra, self.normalize])
            self.local_transfo = v3.Compose([local_transfo_extra, self.normalize])
        else:
            self.global_transfo1 = v3.Compose(
                [resize_global, color_jittering, global_transfo1_extra, self.normalize]
            )
            self.global_transfo2 = v3.Compose(
                [resize_global, color_jittering, global_transfo2_extra, self.normalize]
            )
            self.local_transfo = v3.Compose([color_jittering, local_transfo_extra, self.normalize])

    def _apply_geometric_augmentation(self, transforms, image, gt_bbox, seed):
        x = transforms[0](image, gt_bbox=gt_bbox, seed=seed)
        for t in transforms[1:]:
            x = t(x)
        return x

    def __call__(self, image, label=None, proposal_centers=None):
        output = {}
        output["weak_flag"] = True  # some residual from mugs

        gt_bbox = _compute_gt_bbox_from_mask(label)
        proposal_bboxes = proposal_centers_to_point_bboxes(
            proposal_centers,
            image.shape[1:],
        )

        if self.share_color_jitter:
            image = self.color_jittering(image)

        # global crops:
        im1_base = self._apply_geometric_augmentation(
            self.geometric_augmentation_global, image, gt_bbox, np.random.randint(0, 2**31 - 1)
        )
        global_crop_1_transf = self.global_transfo1(im1_base)
        global_crop_1 = self.resize_global_post_transf(global_crop_1_transf)

        im2_base = self._apply_geometric_augmentation(
            self.geometric_augmentation_global, image, gt_bbox, np.random.randint(0, 2**31 - 1)
        )
        global_crop_2_transf = self.global_transfo2(im2_base)
        global_crop_2 = self.resize_global_post_transf(global_crop_2_transf)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        if self.teacher_no_color_jitter:
            output["global_crops_teacher"] = [
                self.normalize(im1_base),
                self.normalize(im2_base),
            ]
        else:
            output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        if self.gram_teacher_crops_size is not None:
            # crops for gram teacher:
            if self.gram_teacher_no_distortions:
                gram_crop_1 = self.normalize(self.resize_gram_teacher(im1_base))
                gram_crop_2 = self.normalize(self.resize_gram_teacher(im2_base))
            else:
                gram_crop_1 = self.resize_gram_teacher(global_crop_1_transf)
                gram_crop_2 = self.resize_gram_teacher(global_crop_2_transf)
            output["gram_teacher_crops"] = [gram_crop_1, gram_crop_2]

        # local crops:
        if self.local_crops_subset_of_global_crops:
            _local_crops = [self.local_transfo(im1_base) for _ in range(self.local_crops_number // 2)] + [
                self.local_transfo(im2_base) for _ in range(self.local_crops_number // 2)
            ]

            local_crops = []
            offsets = []
            gs = self.global_crops_size
            ls = self.local_crops_size
            for img in _local_crops:
                rx, ry, rz = np.random.randint(0, (gs - ls) // self.patch_size, 3) * self.patch_size
                local_crops.append(img[:, rx : rx + ls, ry : ry + ls, rz : rz + ls])
                offsets.append((rx, ry, rz))

            output["local_crops"] = local_crops
            output["offsets"] = offsets

        elif (
            label is None
            and len(proposal_bboxes) > 0
        ):
            # Proposal-anchored local crops for unlabeled data: each
            # local crop is *anchored* to a randomly chosen proposal
            # centre — i.e. the proposal point sits at the centre of the
            # crop box. The crop's XY scale, XY aspect ratio and Z
            # scale are still sampled per call (within
            # ``local_crops_inner_slice_scale`` /
            # ``local_crops_cross_slice_scale``); only the *starting*
            # position is fixed so that the chosen proposal point ends
            # up inside the box. Candidates are shuffled every call so
            # the 16 centres see roughly equal coverage across an epoch.
            local_crops = []
            n_bboxes = len(proposal_bboxes)
            proposal_order = torch.randperm(n_bboxes).tolist()
            H, W, D = [int(v) for v in image.shape[1:]]
            for local_index in range(self.local_crops_number):
                proposal_index = proposal_order[
                    local_index % n_bboxes
                ]
                # point_bbox is a 1-voxel bbox (h0=h1, w0=w1, d0=d1)
                # produced by proposal_centers_to_point_bboxes.
                point_bbox = proposal_bboxes[proposal_index]
                ph0, ph1, pw0, pw1, pd0, pd1 = [int(v) for v in point_bbox]
                # Centre voxel of the 1-voxel bbox — i.e. the proposal
                # point itself.
                ph = (ph0 + ph1) // 2
                pw = (pw0 + pw1) // 2
                pd = (pd0 + pd1) // 2

                # Sample crop size using the same distribution as
                # RandomResizedCrop3D for inner-slice (H, W) and
                # cross-slice (D) dimensions.
                local_aug = self.geometric_augmentation_local[0]
                crop_h, crop_w = local_aug.get_inner_slice_crop(H, W)
                crop_d = int(round(D * local_aug.R.uniform(*local_aug.cross_slice_scale)))
                # Don't exceed image bounds.
                crop_h = max(1, min(crop_h, H))
                crop_w = max(1, min(crop_w, W))
                crop_d = max(1, min(crop_d, D))

                # Anchor the proposal point at the *centre* of the crop
                # box, then clip to image bounds.
                h0 = int(round(ph - crop_h / 2.0))
                w0 = int(round(pw - crop_w / 2.0))
                d0 = int(round(pd - crop_d / 2.0))
                h0 = max(0, min(h0, H - crop_h))
                w0 = max(0, min(w0, W - crop_w))
                d0 = max(0, min(d0, D - crop_d))

                # Direct tensor slice (image has shape (C, H, W, D)).
                cropped = image[:, h0:h0 + crop_h, w0:w0 + crop_w, d0:d0 + crop_d]
                # Resize to ``local_crops_size``.
                local_crop = interpolate(
                    cropped.unsqueeze(0),
                    size=local_aug.size,
                    mode=local_aug.interpolation,
                ).squeeze(0)
                # Apply remaining geometric flips from
                # ``geometric_augmentation_local`` (e.g. RandFlip).
                for t in self.geometric_augmentation_local[1:]:
                    local_crop = t(local_crop)
                # Apply colour augmentation / normalisation.
                local_crop = self.local_transfo(local_crop)
                local_crops.append(local_crop)
            output["local_crops"] = local_crops
            output["offsets"] = ()

        elif self.crop_setting == 4:
            # setting4: whole-tumor global bbox crop + per-component local crops
            component_bboxes = _compute_connected_components_bboxes(label)
            local_crops = []

            if len(component_bboxes) == 0:
                # no label: fall back to standard random crops
                for _ in range(self.local_crops_number):
                    local_crops.append(
                        self.local_transfo(
                            self._apply_geometric_augmentation(
                                self.geometric_augmentation_local, image, None, np.random.randint(0, 2**31 - 1)
                            )
                        )
                    )
            else:
                # cycle through components to fill local_crops_number slots
                for i in range(self.local_crops_number):
                    comp_bbox = component_bboxes[i % len(component_bboxes)]
                    local_crops.append(
                        self.local_transfo(
                            self._apply_geometric_augmentation(
                                self.geometric_augmentation_local, image, comp_bbox, np.random.randint(0, 2**31 - 1)
                            )
                        )
                    )

            output["local_crops"] = local_crops
            output["offsets"] = ()

        else:
            local_crops = [
                self.local_transfo(
                    self._apply_geometric_augmentation(
                        self.geometric_augmentation_local, image, gt_bbox, np.random.randint(0, 2**31 - 1)
                    )
                )
                for _ in range(self.local_crops_number)
            ]
            output["local_crops"] = local_crops
            output["offsets"] = ()

        return output
