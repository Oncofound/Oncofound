"""
box_ops.py
==========
Shared box-from-center arithmetic.

The crop-hit target computation (``targets.py``) and the crop selection
(``crop_select.py``) MUST agree on what a 48^3 crop around a given
voxel center means — otherwise the training signal and the test-time
selection target different things.

The rule used here:
  * The box is ``[lo, lo + size)`` (exclusive end), so the volume is
    *exactly* ``size^3`` voxels.
  * If the image is too small on some axis (size > length), the box
    spans the whole axis; downstream code is expected to *right-pad*
    the cropped tensor to ``size`` before feeding it to the model.
  * Otherwise, if the box would cross an image boundary, we *shift* it
    back into the image so that the volume stays exactly ``size``
    voxels.
"""

from __future__ import annotations

from typing import Tuple


def axis_box_from_center(
    center: int,
    size: int,
    length: int,
) -> Tuple[int, int]:
    """Return ``[lo, hi)`` for a crop of exact ``size`` centered on
    ``center`` along a single axis of length ``length``.

    Parameters
    ----------
    center : int
        Voxel center.
    size : int
        Edge length of the crop.
    length : int
        Voxel extent along this axis.

    Returns
    -------
    (lo, hi) : Tuple[int, int]
        ``hi - lo == size`` whenever ``length >= size``;
        otherwise ``(0, length)`` and the caller is expected to pad.
    """
    if size < 0:
        raise ValueError(f"size must be >= 0, got {size}")
    if length <= 0:
        return 0, 0
    if size == 0:
        center = max(0, min(int(center), length - 1))
        return center, center
    if length <= size:
        return 0, length
    c = int(center)
    lo = c - size // 2
    hi = lo + size
    # Shift back into bounds
    if lo < 0:
        hi -= lo
        lo = 0
    if hi > length:
        lo -= (hi - length)
        hi = length
    lo = max(0, lo)
    hi = min(length, hi)
    if hi - lo != size:
        # Should never happen; fall back to clamp (so we never crash).
        hi = min(length, lo + size)
        lo = max(0, hi - size)
    return int(lo), int(hi)
