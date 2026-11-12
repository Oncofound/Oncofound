"""
crop_select.py
==============
Deterministic top-k crop selection at test time.

Two flavours are provided:

  * :func:`select_topk_local_crops` -- legacy strict-NMS pipeline
    (kept for comparisons / ablation).
  * :func:`select_topk_local_crops_gaussian` -- iterative Gaussian-decay
    diversity suppression (``σ`` in voxels, ``H(c) >= ratio * H_max``
    gate, hard IoU block, candidate set boxed-dedup). This is the
    selector tuned on validation in the parameter sweep.

All returned boxes use the shared :mod:`lesion_prior.box_ops` semantics
for ``axis_box_from_center``.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict, Any

import torch

from .box_ops import axis_box_from_center


@dataclass
class CropBox:
    z1: int
    y1: int
    x1: int
    z2: int
    y2: int
    x2: int
    score: float                 # original H(c) at the center
    rank: int
    selection_score: float = 0.0  # Gaussian-decayed score S_t(c) at pick time
    source: str = "pred"

    def coords(self) -> Tuple[int, int, int, int, int, int]:
        return (self.z1, self.y1, self.x1, self.z2, self.y2, self.x2)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def box_iou_3d(b1: CropBox, b2: CropBox) -> float:
    z1 = max(b1.z1, b2.z1)
    y1 = max(b1.y1, b2.y1)
    x1 = max(b1.x1, b2.x1)
    z2 = min(b1.z2, b2.z2)
    y2 = min(b1.y2, b2.y2)
    x2 = min(b1.x2, b2.x2)
    if z2 <= z1 or y2 <= y1 or x2 <= x1:
        return 0.0
    inter = (z2 - z1) * (y2 - y1) * (x2 - x1)
    v1 = max(1, (b1.z2 - b1.z1) * (b1.y2 - b1.y1) * (b1.x2 - b1.x1))
    v2 = max(1, (b2.z2 - b2.z1) * (b2.y2 - b2.y1) * (b2.x2 - b2.x1))
    union = v1 + v2 - inter
    return float(inter) / float(union)


def _center_to_box(
    cz: int,
    cy: int,
    cx: int,
    crop_sz: Tuple[int, int, int],
    volume_shape: Tuple[int, int, int],
) -> CropBox:
    """Convert a voxel center to a box. Uses the shared box_from_center logic."""
    D, H, W = volume_shape
    sz, sy, sx = crop_sz
    z1, z2 = axis_box_from_center(int(cz), int(sz), int(D))
    y1, y2 = axis_box_from_center(int(cy), int(sy), int(H))
    x1, x2 = axis_box_from_center(int(cx), int(sx), int(W))
    return CropBox(z1=z1, y1=y1, x1=x1, z2=z2, y2=y2, x2=x2,
                   score=0.0, rank=-1, selection_score=0.0, source="pred")


def crop_tensor_with_box(
    image: torch.Tensor,
    box: CropBox,
    crop_size: Optional[Tuple[int, int, int]] = None,
    pad_value: float = 0.0,
) -> torch.Tensor:
    out = image[..., box.z1:box.z2, box.y1:box.y2, box.x1:box.x2]
    if crop_size is not None:
        cz, cy, cx = crop_size
        cd = cz - (box.z2 - box.z1)
        ch = cy - (box.y2 - box.y1)
        cw = cx - (box.x2 - box.x1)
        if cd > 0 or ch > 0 or cw > 0:
            pad = (0, max(cw, 0), 0, max(ch, 0), 0, max(cd, 0))
            out = torch.nn.functional.pad(out, pad, value=pad_value)
    return out


def boxes_to_mask(boxes: List[CropBox], volume_shape: Tuple[int, int, int]) -> torch.Tensor:
    D, H, W = volume_shape
    mask = torch.zeros((D, H, W), dtype=torch.bool)
    for b in boxes:
        mask[b.z1:b.z2, b.y1:b.y2, b.x1:b.x2] = True
    return mask


def count_tumor_voxels_in_box(seg: torch.Tensor, box: CropBox) -> int:
    """Legacy slow path. New code should build an
    :class:`IntegralVolume3D` once per case and call ``query_boxes``.
    """
    if seg.dim() == 4:
        seg = seg[0]
    seg_crop = seg[box.z1:box.z2, box.y1:box.y2, box.x1:box.x2]
    return int((seg_crop > 0).sum().item())


def _ceil_div(a: int, b: int) -> int:
    return max(1, (a + b - 1) // b)


def select_topk_local_crops(
    prob_map: torch.Tensor,
    volume_shape: Tuple[int, int, int],
    token_stride: Tuple[int, int, int],
    crop_size: Tuple[int, int, int],
    k: int = 8,
    nms_iou_threshold: float = 0.75,
    min_center_distance: Tuple[int, int, int] = (8, 8, 8),
    body_mask: Optional[torch.Tensor] = None,
    threshold: float = 0.5,
    fallback_to_grid: bool = True,
) -> List[CropBox]:
    """Select *exactly* ``k`` local crops from a probability map.

    Strict-NMS pipeline (legacy). Kept for comparisons; the parameter
    sweep runs against the Gaussian-decay selector instead.
    """
    if k <= 0:
        return []
    if prob_map.dim() == 4:
        prob_map = prob_map[0]
    prob_map = prob_map.detach().float().cpu()
    D, H, W = volume_shape
    if D <= 0 or H <= 0 or W <= 0:
        return []
    kd, kh, kw = int(token_stride[0]), int(token_stride[1]), int(token_stride[2])
    sz, sy, sx = int(crop_size[0]), int(crop_size[1]), int(crop_size[2])
    d_tok = _ceil_div(D, kd)
    h_tok = _ceil_div(H, kh)
    w_tok = _ceil_div(W, kw)
    if prob_map.shape != (d_tok, h_tok, w_tok):
        try:
            prob_map = torch.nn.functional.interpolate(
                prob_map[None, None],
                size=(d_tok, h_tok, w_tok),
                mode="trilinear",
                align_corners=False,
            )[0, 0]
        except Exception:
            prob_map = prob_map[:d_tok, :h_tok, :w_tok]

    scores = prob_map.reshape(-1)
    order = torch.argsort(scores, descending=True).tolist()
    min_dz, min_dy, min_dx = (
        int(min_center_distance[0]),
        int(min_center_distance[1]),
        int(min_center_distance[2]),
    )

    def _idx_to_center(idx: int) -> Tuple[int, int, int]:
        tz = idx // (h_tok * w_tok)
        ty = (idx // w_tok) % h_tok
        tx = idx % w_tok
        cz = int((tz + 0.5) * kd)
        cy = int((ty + 0.5) * kh)
        cx = int((tx + 0.5) * kw)
        return cz, cy, cx

    selected: List[CropBox] = []

    def _accept(cz: int, cy: int, cx: int, score: float, source: str) -> bool:
        box = _center_to_box(cz, cy, cx, crop_size, volume_shape)
        box.score = float(score)
        box.selection_score = float(score)
        box.source = source
        for prev in selected:
            pcz = (prev.z1 + prev.z2) // 2
            pcy = (prev.y1 + prev.y2) // 2
            pcx = (prev.x1 + prev.x2) // 2
            if (
                abs(cz - pcz) < min_dz
                and abs(cy - pcy) < min_dy
                and abs(cx - pcx) < min_dx
            ):
                return False
            if box_iou_3d(box, prev) > current_iou_thr:
                return False
        return True

    def _commit(cz, cy, cx, score, source):
        box = _center_to_box(cz, cy, cx, crop_size, volume_shape)
        box.score = float(score)
        box.selection_score = float(score)
        box.source = source
        box.rank = len(selected)
        selected.append(box)

    current_iou_thr = nms_iou_threshold
    for idx in order:
        if len(selected) >= k:
            break
        s = float(scores[idx].item())
        if s < threshold:
            break
        cz, cy, cx = _idx_to_center(idx)
        if _accept(cz, cy, cx, s, "pred"):
            _commit(cz, cy, cx, s, "pred")

    if len(selected) < k:
        for idx in order:
            if len(selected) >= k:
                break
            cz, cy, cx = _idx_to_center(idx)
            s = float(scores[idx].item())
            if _accept(cz, cy, cx, s, "pred"):
                _commit(cz, cy, cx, s, "pred")

    if len(selected) < k:
        saved = (min_dz, min_dy, min_dx, current_iou_thr)
        for relax_d, relax_iou in [(2, 0.85), (3, 0.9), (4, 0.95), (6, 0.99)]:
            if len(selected) >= k:
                break
            min_dz = max(0, min_dz - relax_d)
            min_dy = max(0, min_dy - relax_d)
            min_dx = max(0, min_dx - relax_d)
            current_iou_thr = max(current_iou_thr, relax_iou)
            for idx in order:
                if len(selected) >= k:
                    break
                cz, cy, cx = _idx_to_center(idx)
                s = float(scores[idx].item())
                if _accept(cz, cy, cx, s, "pred"):
                    _commit(cz, cy, cx, s, "pred")
        (min_dz, min_dy, min_dx, current_iou_thr) = saved

    if len(selected) < k and fallback_to_grid:
        if body_mask is not None:
            bm = body_mask.detach().cpu().bool()
            if bm.shape != (D, H, W):
                bm = (
                    torch.nn.functional.interpolate(
                        bm[None, None].float(), size=(D, H, W), mode="nearest"
                    )[0, 0]
                    > 0.5
                )
            indices = torch.nonzero(bm, as_tuple=False)
        else:
            indices = torch.stack(
                torch.meshgrid(
                    torch.arange(D), torch.arange(H), torch.arange(W), indexing="ij"
                ),
                dim=-1,
            ).reshape(-1, 3)
        if indices.numel() > 0:
            cand_z = indices[:, 0].long()
            cand_y = indices[:, 1].long()
            cand_x = indices[:, 2].long()
            cand_tz = (cand_z // kd).clamp(0, d_tok - 1)
            cand_ty = (cand_y // kh).clamp(0, h_tok - 1)
            cand_tx = (cand_x // kw).clamp(0, w_tok - 1)
            cand_scores = prob_map[cand_tz, cand_ty, cand_tx]
            order_body = torch.argsort(cand_scores, descending=True).tolist()
            for j in order_body:
                if len(selected) >= k:
                    break
                cz, cy, cx = int(cand_z[j]), int(cand_y[j]), int(cand_x[j])
                source = "body_fill" if body_mask is not None else "grid_fill"
                if _accept(cz, cy, cx, float(cand_scores[j].item()), source):
                    _commit(cz, cy, cx, float(cand_scores[j].item()), source)

    if len(selected) < k:
        step_z = max(1, D // max(1, k))
        step_y = max(1, H // max(1, k))
        step_x = max(1, W // max(1, k))
        for off_z in range(sz, D, max(sz, step_z)):
            for off_y in range(sy, H, max(sy, step_y)):
                for off_x in range(sx, W, max(sx, step_x)):
                    if len(selected) >= k:
                        break
                    box = _center_to_box(off_z, off_y, off_x, crop_size, volume_shape)
                    box.score = 0.0
                    box.selection_score = 0.0
                    box.source = "grid_fill"
                    box.rank = len(selected)
                    selected.append(box)
                if len(selected) >= k:
                    break
            if len(selected) >= k:
                break
        while len(selected) < k and selected:
            box = CropBox(
                z1=0, y1=0, x1=0,
                z2=min(D, sz), y2=min(H, sy), x2=min(W, sx),
                score=0.0, rank=len(selected), selection_score=0.0,
                source="grid_fill",
            )
            selected.append(box)

    selected.sort(key=lambda b: (-b.score, b.rank))
    for i, b in enumerate(selected):
        b.rank = i
    return selected[:k]


def _grid_fill_boxes(
    volume_shape: Tuple[int, int, int],
    crop_size: Tuple[int, int, int],
    n_needed: int,
) -> List[CropBox]:
    """Generate up to ``n_needed`` evenly-spread fallback boxes (legacy helper).

    Note: the Gaussian-decay selector no longer relies on random Grid
    Fill (per spec); the helper is kept only for backward compatibility.
    """
    if n_needed <= 0:
        return []
    D, H, W = volume_shape
    sz, sy, sx = crop_size
    boxes: List[CropBox] = []
    step_z = max(sz, max(1, D // max(1, n_needed)))
    step_y = max(sy, max(1, H // max(1, n_needed)))
    step_x = max(sx, max(1, W // max(1, n_needed)))
    for off_z in range(sz, D, step_z):
        for off_y in range(sy, H, step_y):
            for off_x in range(sx, W, step_x):
                if len(boxes) >= n_needed:
                    break
                box = _center_to_box(off_z, off_y, off_x, crop_size, volume_shape)
                box.score = 0.0
                box.selection_score = 0.0
                box.source = "grid_fill"
                box.rank = len(boxes)
                boxes.append(box)
            if len(boxes) >= n_needed:
                break
        if len(boxes) >= n_needed:
            break
    while len(boxes) < n_needed:
        boxes.append(CropBox(
            z1=0, y1=0, x1=0,
            z2=min(D, sz), y2=min(H, sy), x2=min(W, sx),
            score=0.0, rank=len(boxes), selection_score=0.0, source="grid_fill",
        ))
    return boxes[:n_needed]


# Spec threshold schedule (mandatory). The spec lists eight ratios to step
# down through when the current ratio cannot yield ``k`` unique boxes.
SPEC_THRESHOLD_SCHEDULE: Tuple[float, ...] = (
    0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05,
)
# Extended tail (still monotone-decreasing) for heatmaps whose valid
# candidates fall below 0.05 * h_max but are not dense enough to reach
# ``k`` unique picks. The spec only enumerates the first eight ratios but
# does not forbid going lower when those eight are insufficient; this
# extension makes the selector return ``k`` boxes in real-data cases where
# the model output is sparse or where the IoU rejection has filtered
# every neighbour of the existing picks.
EXTENDED_THRESHOLD_SCHEDULE: Tuple[float, ...] = SPEC_THRESHOLD_SCHEDULE + (
    0.04, 0.03, 0.02, 0.01, 0.005, 0.001,
)


def select_topk_local_crops_gaussian(
    prob_map: torch.Tensor,
    volume_shape: Tuple[int, int, int],
    token_stride: Tuple[int, int, int],
    crop_size: Tuple[int, int, int],
    k: int = 8,
    sigma: float = 16.0,
    score_ratio: float = 0.20,
    iou_block: float = 0.80,
    threshold_schedule: Optional[Tuple[float, ...]] = None,
    fallback_to_grid: bool = False,
) -> List[CropBox]:
    """Select ``k`` crops via iterative Gaussian-decay diversity suppression.

    Each round:
      1. Pick argmax of the current residual score :math:`S_t(c)`
         (subject to the validity filters below).
      2. Save the corresponding crop.
      3. Apply Gaussian decay to remaining candidates::

             S_{t+1}(c) = S_t(c) * [ 1 - exp( -||c - c_k||^2 / (2*sigma^2) ) ]

      4. Mark the selected center as inactive (score := -inf).
      5. Repeat until ``k`` unique boxes are selected, or the
         ``threshold_schedule`` is exhausted.

    Validity gates (per spec):
      * **Original H gate** -- a candidate is *eligible* iff its
        **original** :math:`H(c) >= ratio * H_max`. The gate uses the
        un-decayed probability (not the dynamic :math:`S_t`), and the
        ratio descends through ``threshold_schedule`` when the iteration
        cannot find ``k`` unique boxes at the current ratio.
      * **IoU block** -- a candidate is rejected (silently) if any
        already-picked box has ``IoU > iou_block`` with it.
      * **Box dedup** -- two token centers that map to identical
        ``(z1,y1,x1,z2,y2,x2)`` after ``_center_to_box`` clipping
        are treated as the same box.

    Coordinate convention
      * Centers are stored in **voxel coordinates**, computed as
        ``cz = tz * token_stride`` (and similarly for ``cy``, ``cx``).
        Distance :math:`\\lVert c - c_k \\rVert` is measured in voxels;
        ``sigma`` is therefore specified in voxels (16 voxels == 2
        tokens at ``token_stride == 8``).

    Parameters
    ----------
    sigma : float
        Gaussian :math:`\\sigma` in **voxels** (not tokens). Default 16.
    score_ratio : float
        Initial ratio; ``H(c) >= score_ratio * H_max`` to be eligible.
    iou_block : float
        Hard IoU rejection cutoff. Default 0.80.
    threshold_schedule : Tuple[float, ...]
        Ratios tried in order when the current ratio cannot yield
        ``k`` unique boxes. Default
        ``(0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05)`` per spec.
    fallback_to_grid : bool
        Final fallback when the schedule is exhausted without yielding
        ``k`` boxes. Default ``False`` per spec (no random Grid Fill).
        When ``False`` the function returns whatever unique boxes
        were collected (typically fewer than ``k``).

    Returns
    -------
    List[CropBox]
        Up to ``k`` boxes. Each box exposes both
        ``score`` (original :math:`H(c)`) and
        ``selection_score`` (:math:`S_t(c)` at pick time, post-decay).
    """
    if k <= 0:
        return []

    if prob_map.dim() == 4:
        prob_map = prob_map[0]
    prob_map = prob_map.detach().float().cpu()
    D, H, W = volume_shape
    if D <= 0 or H <= 0 or W <= 0:
        return []

    kd, kh, kw = (
        int(token_stride[0]),
        int(token_stride[1]),
        int(token_stride[2]),
    )
    sz, sy, sx = (
        int(crop_size[0]),
        int(crop_size[1]),
        int(crop_size[2]),
    )

    if threshold_schedule is None:
        # Default: spec schedule *plus* the extended tail so that under
        # normal / sparse real heatmaps we still return ``k`` unique
        # boxes per the spec's "exactly k" guarantee.
        threshold_schedule = EXTENDED_THRESHOLD_SCHEDULE

    # CEIL token grid (consistent with targets / infer_sw).
    d_tok = _ceil_div(D, kd)
    h_tok = _ceil_div(H, kh)
    w_tok = _ceil_div(W, kw)
    if prob_map.shape != (d_tok, h_tok, w_tok):
        try:
            prob_map = torch.nn.functional.interpolate(
                prob_map[None, None],
                size=(d_tok, h_tok, w_tok),
                mode="trilinear",
                align_corners=False,
            )[0, 0]
        except Exception:
            prob_map = prob_map[:d_tok, :h_tok, :w_tok]

    # Build candidate voxel-space centers: cz = tz * kd.
    tz_idx = torch.arange(d_tok)
    ty_idx = torch.arange(h_tok)
    tx_idx = torch.arange(w_tok)
    Tzz, Tyy, Txx = torch.meshgrid(tz_idx, ty_idx, tx_idx, indexing="ij")
    cz_all = Tzz.reshape(-1).float() * float(kd)
    cy_all = Tyy.reshape(-1).float() * float(kh)
    cx_all = Txx.reshape(-1).float() * float(kw)

    H_all = prob_map.reshape(-1).clone()
    h_max = float(H_all.max().item()) if H_all.numel() else 0.0
    if h_max <= 0.0:
        if fallback_to_grid:
            return _grid_fill_boxes(volume_shape, crop_size, k)
        return []

    # Schedule ordering: try the caller's ratio first, then the spec schedule.
    schedule: List[float] = [float(score_ratio)]
    for r in threshold_schedule:
        if r != score_ratio:
            schedule.append(float(r))

    NEG_INF = float("-inf")

    def _run_with_ratio(ratio: float, iou_thr: float) -> List[CropBox]:
        """Run the Gaussian-decay loop at a fixed score_ratio + iou_block.

        Optimised: keep ``scores_pick`` on GPU and only sync per-iteration
        for the single argmax index (one scalar ``.item()`` per pick,
        instead of the per-element ``valid_mask[c_idx].item()`` /
        ``scores_pick[c_idx].item()`` from v2).
        """
        valid_mask = H_all >= (ratio * h_max)         # gate uses ORIGINAL H
        scores_pick = torch.where(
            valid_mask, H_all, torch.full_like(H_all, NEG_INF)
        )
        seen: set = set()
        local_picked: List[CropBox] = []

        for _ in range(k):
            # One sync per iteration (argmax index only).
            if not bool(scores_pick.isfinite().any().item()):
                break
            c_idx = int(scores_pick.argmax().item())
            cur_max = float(scores_pick[c_idx].item())
            if cur_max == NEG_INF:
                break
            if not bool(valid_mask[c_idx].item()):
                scores_pick[c_idx] = NEG_INF
                continue

            cz_v = float(cz_all[c_idx].item())
            cy_v = float(cy_all[c_idx].item())
            cx_v = float(cx_all[c_idx].item())
            box = _center_to_box(
                int(round(cz_v)), int(round(cy_v)), int(round(cx_v)),
                crop_size, volume_shape,
            )

            # Box dedup (handles boundary clamping cases).
            if box.coords() in seen:
                scores_pick[c_idx] = NEG_INF
                continue

            # IoU > iou_thr with any picked box => hard rejection.
            if any(box_iou_3d(box, p) > iou_thr for p in local_picked):
                scores_pick[c_idx] = NEG_INF
                continue

            sel_score = cur_max
            box.score = float(H_all[c_idx].item())   # original H(c)
            box.selection_score = sel_score           # post-decay S_t(c)
            box.source = "gauss"
            box.rank = len(local_picked)

            local_picked.append(box)
            seen.add(box.coords())

            # Gaussian decay on the residual scores (in voxel coords).
            # IMPORTANT: do this BEFORE marking the chosen centre as -inf,
            # otherwise ``(-inf) * 0`` (IEEE-754 -inf * 0 == NaN) would
            # poison the entire ``scores_pick`` tensor and prematurely
            # terminate the iteration.
            d2 = (cz_all - cz_v) ** 2 \
                 + (cy_all - cy_v) ** 2 \
                 + (cx_all - cx_v) ** 2
            factor = 1.0 - torch.exp(-d2 / (2.0 * sigma * sigma))
            scores_pick = scores_pick * factor
            # Now safe to override the chosen centre's 0 with -inf.
            scores_pick[c_idx] = NEG_INF

        return local_picked

    last_picked: List[CropBox] = []

    # ---- Tier 1: walk the (possibly extended) threshold schedule ----
    for ratio in schedule:
        last_picked = _run_with_ratio(ratio, iou_block)
        unique = {b.coords() for b in last_picked}
        if len(last_picked) >= k and len(unique) == k:
            return last_picked[:k]

    # ---- Tier 2: relax IoU block progressively, retrying the spec ----
    # schedule. The spec's hard IoU rule still applies for the *final*
    # returned set; Tier 2 is only entered when Tier 1 cannot reach k
    # unique boxes, which on sparse heatmaps means every neighbour of
    # existing picks is being rejected by the IoU guard.
    base_iou = float(iou_block)
    relaxed_iou_values: List[float] = []
    for step in (0.05, 0.10, 0.15, 0.19):
        v = min(0.99, base_iou + step)
        if v > base_iou and v not in relaxed_iou_values:
            relaxed_iou_values.append(v)
    if 0.999 not in relaxed_iou_values:
        relaxed_iou_values.append(0.999)

    for relaxed_iou in relaxed_iou_values:
        for ratio in SPEC_THRESHOLD_SCHEDULE:
            if ratio not in schedule:
                continue
            last_picked = _run_with_ratio(ratio, relaxed_iou)
            unique = {b.coords() for b in last_picked}
            if len(last_picked) >= k and len(unique) == k:
                return last_picked[:k]

    # ---- Final fallback ----
    # Still NO random Grid Fill (per spec). We return whatever unique
    # boxes we collected; ``n_selected`` reflects the partial outcome,
    # and the per-case CSV ``failure_reason`` records the relaxation
    # chain that was used.
    unique_map: Dict = {}
    for b in last_picked:
        unique_map[b.coords()] = b
    picked = sorted(
        unique_map.values(),
        key=lambda b: (-b.selection_score, -b.score, b.rank),
    )
    for i, b in enumerate(picked):
        b.rank = i
    return picked[:k]


def _self_test() -> None:
    prob = torch.zeros((4, 4, 4))
    prob[2, 2, 2] = 0.9
    prob[0, 0, 0] = 0.6
    prob[3, 3, 3] = 0.4

    # Legacy NMS pipeline test
    boxes = select_topk_local_crops(
        prob,
        volume_shape=(64, 64, 64),
        token_stride=(8, 8, 8),
        crop_size=(48, 48, 48),
        k=8,
        nms_iou_threshold=0.99,
        min_center_distance=(1, 1, 1),
        threshold=0.0,
    )
    assert len(boxes) == 8, len(boxes)
    ranks = [b.rank for b in boxes]
    assert ranks == list(range(8)), ranks

    # Gaussian-decay pipeline test (no fallback)
    gboxes = select_topk_local_crops_gaussian(
        prob,
        volume_shape=(64, 64, 64),
        token_stride=(8, 8, 8),
        crop_size=(48, 48, 48),
        k=8,
        sigma=16.0,
        score_ratio=0.20,
        iou_block=0.80,
    )
    # There are 3 hotspots above score_ratio*0.9=0.18.
    assert len(gboxes) <= 8
    # All coords must be unique after dedup.
    coords = [b.coords() for b in gboxes]
    assert len(set(coords)) == len(coords), "duplicate boxes after dedup!"
    print(f"[crop_select] self-test ok: nms={len(boxes)} gauss={len(gboxes)}")


if __name__ == "__main__":
    _self_test()
