"""Numba-accelerated gray-level co-occurrence feature helpers.

This module provides two entry points used by the notebook:

* ``glcm_score_from_crop`` for the scalar score comparison against
    ``skimage.feature.graycomatrix`` / ``graycoprops``.
* ``glcm_map_numba`` for sliding-window maps of contrast, energy,
    homogeneity, and score.

The sliding-window implementation uses incremental frontier updates so the
window state can be reused as the crop moves horizontally.

Reference:
* https://stackoverflow.com/questions/38955970/reduce-running-time-in-texture-analysis-using-glcm-python
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from skimage.color import rgb2hsv
from skimage.util import img_as_ubyte

DEFAULT_GLCM_DISTANCES = np.array([1, 3, 5], dtype=np.int64)
DEFAULT_GLCM_ANGLES = np.array(
    [0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0], dtype=np.float64
)
DEFAULT_GLCM_LEVELS = 256


def build_glcm_offsets(
    distances: np.ndarray | None = None,
    angles: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build integer row/column offsets for GLCM directions.

    Parameters
    ----------
    distances:
        1-D array of pixel distances to evaluate. If omitted, the module
        defaults to ``[1, 3, 5]``.
    angles:
        1-D array of angles in radians. If omitted, the module defaults to
        the four canonical directions ``[0, pi/4, pi/2, 3pi/4]``.

    Returns
    -------
    tuple of numpy.ndarray
        Two parallel ``int64`` arrays ``(row_offsets, col_offsets)``. Each
        pair describes a directional offset that can be applied to the image
        grid when accumulating co-occurrence counts.
    """
    if distances is None:
        distances = DEFAULT_GLCM_DISTANCES
    if angles is None:
        angles = DEFAULT_GLCM_ANGLES

    distances = np.asarray(distances, dtype=np.int64)
    angles = np.asarray(angles, dtype=np.float64)

    row_offsets = np.empty(distances.size * angles.size, dtype=np.int64)
    col_offsets = np.empty(distances.size * angles.size, dtype=np.int64)

    offset_index = 0
    for distance in distances:
        for angle in angles:
            row_offsets[offset_index] = int(np.round(np.sin(angle) * distance))
            col_offsets[offset_index] = int(np.round(np.cos(angle) * distance))
            offset_index += 1

    return row_offsets, col_offsets


@njit(fastmath=True, inline="always")
def _apply_pair_update(
    counts,
    totals,
    contrast_num,
    energy_asm,
    hom_num,
    offset_index,
    first_level,
    second_level,
    delta,
    symmetric,
    contrast_weights,
    hom_weights,
):
    """Apply a +/-1 update to a single offset-level pair.

    The function keeps the raw count matrix and the derived accumulators in
    sync. It is used by both the initial window build and the frontier update
    path while sliding the window horizontally.
    """
    old_count = counts[offset_index, first_level, second_level]
    new_count = old_count + delta
    counts[offset_index, first_level, second_level] = new_count
    totals[offset_index] += delta
    contrast_num[offset_index] += delta * contrast_weights[first_level, second_level]
    hom_num[offset_index] += delta * hom_weights[first_level, second_level]
    energy_asm[offset_index] += new_count * new_count - old_count * old_count

    if symmetric and first_level != second_level:
        old_count = counts[offset_index, second_level, first_level]
        new_count = old_count + delta
        counts[offset_index, second_level, first_level] = new_count
        totals[offset_index] += delta
        contrast_num[offset_index] += (
            delta * contrast_weights[first_level, second_level]
        )
        hom_num[offset_index] += delta * hom_weights[first_level, second_level]
        energy_asm[offset_index] += new_count * new_count - old_count * old_count


@njit(fastmath=True, inline="always")
def _build_window_state(
    image,
    top_row,
    left_col,
    window_height,
    window_width,
    row_offsets,
    col_offsets,
    counts,
    totals,
    contrast_num,
    energy_asm,
    hom_num,
    symmetric,
    contrast_weights,
    hom_weights,
):
    """Populate the full GLCM state for one window from scratch.

    This is the expensive baseline path used for the first window in each row.
    The function enumerates every valid pair inside the window, updates the raw
    count tensor, and accumulates the derived contrast, energy, and homogeneity
    numerators needed for fast scoring.
    """
    n_offsets = row_offsets.shape[0]
    for offset_index in range(n_offsets):
        row_offset = row_offsets[offset_index]
        col_offset = col_offsets[offset_index]
        for source_row in range(top_row, top_row + window_height):
            target_row = source_row + row_offset
            if target_row < top_row or target_row >= top_row + window_height:
                continue
            for source_col in range(left_col, left_col + window_width):
                target_col = source_col + col_offset
                if target_col < left_col or target_col >= left_col + window_width:
                    continue
                first_level = int(image[source_row, source_col])
                second_level = int(image[target_row, target_col])
                _apply_pair_update(
                    counts,
                    totals,
                    contrast_num,
                    energy_asm,
                    hom_num,
                    offset_index,
                    first_level,
                    second_level,
                    1,
                    symmetric,
                    contrast_weights,
                    hom_weights,
                )


@njit(fastmath=True, inline="always")
def _update_frontier_column(
    image,
    top_row,
    left_col,
    window_height,
    window_width,
    row_offsets,
    col_offsets,
    counts,
    totals,
    contrast_num,
    energy_asm,
    hom_num,
    frontier_col,
    delta,
    symmetric,
    contrast_weights,
    hom_weights,
):
    """Update the GLCM state for a vertical frontier column.

    A horizontal window shift removes one column from the left side and adds
    one column on the right side. This helper applies the change for a single
    frontier column and is called twice per shift: once with ``delta=-1`` for
    the outgoing column and once with ``delta=+1`` for the incoming column.
    """
    n_offsets = row_offsets.shape[0]
    right_col = left_col + window_width
    bottom_row = top_row + window_height
    for offset_index in range(n_offsets):
        row_offset = row_offsets[offset_index]
        col_offset = col_offsets[offset_index]
        for source_row in range(top_row, bottom_row):
            target_row = source_row + row_offset
            if target_row < top_row or target_row >= bottom_row:
                continue

            source_col = frontier_col
            target_col = source_col + col_offset
            if (
                left_col <= source_col < right_col
                and left_col <= target_col < right_col
            ):
                first_level = int(image[source_row, source_col])
                second_level = int(image[target_row, target_col])
                _apply_pair_update(
                    counts,
                    totals,
                    contrast_num,
                    energy_asm,
                    hom_num,
                    offset_index,
                    first_level,
                    second_level,
                    delta,
                    symmetric,
                    contrast_weights,
                    hom_weights,
                )

            if col_offset != 0:
                source_col = frontier_col - col_offset
                target_col = source_col + col_offset
                if (
                    left_col <= source_col < right_col
                    and left_col <= target_col < right_col
                ):
                    first_level = int(image[source_row, source_col])
                    second_level = int(image[target_row, target_col])
                    _apply_pair_update(
                        counts,
                        totals,
                        contrast_num,
                        energy_asm,
                        hom_num,
                        offset_index,
                        first_level,
                        second_level,
                        delta,
                        symmetric,
                        contrast_weights,
                        hom_weights,
                    )


@njit(fastmath=True, inline="always")
def _window_metrics_from_state(totals, contrast_num, energy_asm, hom_num):
    """Collapse accumulated per-offset state into scalar window metrics.

    Returns the mean contrast, mean energy, mean homogeneity, and the derived
    score used by the notebook. Offsets with zero valid pairs are ignored.
    """
    contrast_sum = 0.0
    energy_sum = 0.0
    homogeneity_sum = 0.0
    valid_offsets = 0

    for offset_index in range(totals.shape[0]):
        if totals[offset_index] > 0:
            total_count = float(totals[offset_index])
            contrast_sum += contrast_num[offset_index] / total_count
            energy_sum += np.sqrt(energy_asm[offset_index]) / total_count
            homogeneity_sum += hom_num[offset_index] / total_count
            valid_offsets += 1

    if valid_offsets == 0:
        return 0.0, 0.0, 0.0, 0.0

    contrast = contrast_sum / valid_offsets
    energy = energy_sum / valid_offsets
    homogeneity = homogeneity_sum / valid_offsets
    score = (energy / (contrast + 1e-8)) * 100.0
    return contrast, energy, homogeneity, score


@njit(fastmath=True)
def _glcm_score_numba_impl(
    v_channel_u8, levels, row_offsets, col_offsets, symmetric=True
):
    """Compute the scalar GLCM score for a full 2-D V channel.

    This is the low-level Numba kernel behind :func:`glcm_score_numba`. It
    mirrors the notebook's skimage comparison path by computing contrast and
    energy for each offset, averaging those per-offset statistics, and then
    combining them into the final score.
    """
    height, width = v_channel_u8.shape
    contrast_sum = 0.0
    energy_sum = 0.0
    pair_count = 0

    for offset_index in range(row_offsets.shape[0]):
        row_offset = row_offsets[offset_index]
        col_offset = col_offsets[offset_index]
        glcm = np.zeros((levels, levels), dtype=np.float64)

        for row_index in range(height):
            target_row = row_index + row_offset
            if target_row < 0 or target_row >= height:
                continue
            for col_index in range(width):
                target_col = col_index + col_offset
                if target_col < 0 or target_col >= width:
                    continue
                first_level = int(v_channel_u8[row_index, col_index])
                second_level = int(v_channel_u8[target_row, target_col])
                glcm[first_level, second_level] += 1.0
                if symmetric:
                    glcm[second_level, first_level] += 1.0

        total = glcm.sum()
        if total <= 0.0:
            continue
        glcm /= total

        contrast = 0.0
        asm = 0.0
        for first_level in range(levels):
            for second_level in range(levels):
                probability = glcm[first_level, second_level]
                diff = first_level - second_level
                contrast += diff * diff * probability
                asm += probability * probability

        contrast_sum += contrast
        energy_sum += np.sqrt(asm)
        pair_count += 1

    if pair_count == 0:
        return 0.0

    contrast_mean = contrast_sum / pair_count
    energy_mean = energy_sum / pair_count
    return (energy_mean / (contrast_mean + 1e-8)) * 100.0


def glcm_score_numba(
    v_channel_u8: np.ndarray,
    levels: int = DEFAULT_GLCM_LEVELS,
    distances: np.ndarray | None = None,
    angles: np.ndarray | None = None,
    symmetric: bool = True,
) -> float:
    """Return the Numba GLCM score for a quantized V-channel image.

    Parameters
    ----------
    v_channel_u8:
        A 2-D ``uint8`` array containing the V channel of an HSV image. Values
        should already be quantized to the range ``[0, levels - 1]``.
    levels:
        Number of gray levels in the input image. The default is 256, which
        matches the notebook validation against ``skimage``.
    distances:
        Optional 1-D array of pixel distances. If omitted, the default is
        ``[1, 3, 5]``.
    angles:
        Optional 1-D array of angles in radians. If omitted, the default is
        ``[0, pi/4, pi/2, 3pi/4]``.
    symmetric:
        If ``True``, each pair is counted in both directions, matching the
        notebook's ``graycomatrix(..., symmetric=True, normed=True)`` setup.

    Returns
    -------
    float
        The scalar score defined as ``energy_mean / (contrast_mean + 1e-8) *
        100``.

    Notes
    -----
    This is the function to call when you want a direct, reusable Numba
    implementation of the scalar score comparison used in the notebook.
    """
    row_offsets, col_offsets = build_glcm_offsets(distances=distances, angles=angles)
    return float(
        _glcm_score_numba_impl(
            np.asarray(v_channel_u8, dtype=np.uint8),
            int(levels),
            row_offsets,
            col_offsets,
            symmetric,
        )
    )


def glcm_score_from_crop(
    image_data,
    x_range,
    y_range,
    levels: int = DEFAULT_GLCM_LEVELS,
    distances: np.ndarray | None = None,
    angles: np.ndarray | None = None,
    symmetric: bool = True,
) -> float:
    """Compute the scalar score for a rectangular crop of an RGB image.

    Parameters
    ----------
    image_data:
        An RGB image as either a NumPy array or an xarray object with a
        ``values`` attribute. The array is expected to be indexed as
        ``(row, column, channel)``.
    x_range, y_range:
        Inclusive-exclusive pixel bounds for the crop. Each should be a
        two-element tuple ``(start, stop)``.
    levels:
        Gray-level count used after HSV conversion. The default is 256.
    distances, angles:
        Optional GLCM geometry settings forwarded to
        :func:`glcm_score_numba`.
    symmetric:
        Whether to use symmetric co-occurrence counting.

    Returns
    -------
    float
        The scalar score for the crop.

    Notes
    -----
    This wrapper performs the same conversion steps as the notebook: it crops
    the image, converts RGB to HSV, extracts the V channel, and then evaluates
    the Numba scalar score.
    """
    image_data = getattr(image_data, "values", image_data).squeeze()
    crop = image_data[y_range[0] : y_range[1], x_range[0] : x_range[1], :]
    hsv_crop = img_as_ubyte(rgb2hsv(crop))
    v_channel = hsv_crop[..., 2].astype(np.uint8)
    return glcm_score_numba(
        v_channel,
        levels=levels,
        distances=distances,
        angles=angles,
        symmetric=symmetric,
    )


@njit(parallel=True, fastmath=True)
def _glcm_map_numba_impl(
    image_data,
    window_height,
    window_width,
    stride,
    row_offsets,
    col_offsets,
    levels,
):
    """Compute contrast, energy, homogeneity, and score maps.

    This is the internal sliding-window kernel behind
    :func:`glcm_map_numba`. It computes the first window in each row from
    scratch and then reuses the existing state while shifting the window
    horizontally using frontier updates.
    """
    height, width = image_data.shape
    out_height = (height - window_height) // stride + 1
    out_width = (width - window_width) // stride + 1

    contrast_map = np.zeros((out_height, out_width), dtype=np.float64)
    energy_map = np.zeros((out_height, out_width), dtype=np.float64)
    homogeneity_map = np.zeros((out_height, out_width), dtype=np.float64)
    score_map = np.zeros((out_height, out_width), dtype=np.float64)

    n_offsets = row_offsets.shape[0]
    contrast_weights = np.empty((levels, levels), dtype=np.float64)
    hom_weights = np.empty((levels, levels), dtype=np.float64)
    for first_level in range(levels):
        for second_level in range(levels):
            diff = first_level - second_level
            diff_sq = diff * diff
            contrast_weights[first_level, second_level] = diff_sq
            hom_weights[first_level, second_level] = 1.0 / (1.0 + diff_sq)

    for output_row in prange(out_height):
        top_row = output_row * stride
        counts = np.zeros((n_offsets, levels, levels), dtype=np.int64)
        totals = np.zeros(n_offsets, dtype=np.int64)
        contrast_num = np.zeros(n_offsets, dtype=np.float64)
        energy_asm = np.zeros(n_offsets, dtype=np.float64)
        hom_num = np.zeros(n_offsets, dtype=np.float64)

        left_col = 0
        _build_window_state(
            image_data,
            top_row,
            left_col,
            window_height,
            window_width,
            row_offsets,
            col_offsets,
            counts,
            totals,
            contrast_num,
            energy_asm,
            hom_num,
            True,
            contrast_weights,
            hom_weights,
        )

        contrast, energy, homogeneity, score = _window_metrics_from_state(
            totals, contrast_num, energy_asm, hom_num
        )
        contrast_map[output_row, 0] = contrast
        energy_map[output_row, 0] = energy
        homogeneity_map[output_row, 0] = homogeneity
        score_map[output_row, 0] = score

        for output_col in range(1, out_width):
            for _ in range(stride):
                _update_frontier_column(
                    image_data,
                    top_row,
                    left_col,
                    window_height,
                    window_width,
                    row_offsets,
                    col_offsets,
                    counts,
                    totals,
                    contrast_num,
                    energy_asm,
                    hom_num,
                    left_col,
                    -1,
                    True,
                    contrast_weights,
                    hom_weights,
                )
                _update_frontier_column(
                    image_data,
                    top_row,
                    left_col + 1,
                    window_height,
                    window_width,
                    row_offsets,
                    col_offsets,
                    counts,
                    totals,
                    contrast_num,
                    energy_asm,
                    hom_num,
                    left_col + window_width,
                    1,
                    True,
                    contrast_weights,
                    hom_weights,
                )
                left_col += 1

            contrast, energy, homogeneity, score = _window_metrics_from_state(
                totals, contrast_num, energy_asm, hom_num
            )
            contrast_map[output_row, output_col] = contrast
            energy_map[output_row, output_col] = energy
            homogeneity_map[output_row, output_col] = homogeneity
            score_map[output_row, output_col] = score

    return contrast_map, energy_map, homogeneity_map, score_map


def glcm_map_numba(
    image_in_hsv,
    window_size,
    stride,
    distances: np.ndarray | None = None,
    angles: np.ndarray | None = None,
    levels: int = DEFAULT_GLCM_LEVELS,
):
    """Calculate sliding-window GLCM feature maps with Numba.

    Parameters
    ----------
    image_in_hsv:
        HSV image array. The function uses the V channel at ``[..., 2]``.
    window_size:
        Two-element tuple ``(window_height, window_width)`` describing the
        sliding analysis window.
    stride:
        Horizontal and vertical step size for the output map grid.
    distances:
        Optional GLCM distances. Defaults to ``[1, 3, 5]``.
    angles:
        Optional GLCM angles in radians. Defaults to the four canonical
        directions ``[0, pi/4, pi/2, 3pi/4]``.
    levels:
        Number of gray levels to use for the co-occurrence matrix.

    Returns
    -------
    dict[str, numpy.ndarray]
        A dictionary with four 2-D float64 arrays keyed by ``"contrast"``,
        ``"energy"``, ``"homogeneity"``, and ``"score"``.

    Notes
    -----
    This is the function to call when you want the fast map-producing version
    of the GLCM analysis. It is designed for visualization and dense sampling
    over an image crop.
    """
    if distances is None:
        distances = DEFAULT_GLCM_DISTANCES
    if angles is None:
        angles = DEFAULT_GLCM_ANGLES

    row_offsets, col_offsets = build_glcm_offsets(distances=distances, angles=angles)
    image_data = np.asarray(image_in_hsv[..., 2], dtype=np.uint8)
    window_height, window_width = window_size
    contrast_map, energy_map, homogeneity_map, score_map = _glcm_map_numba_impl(
        image_data,
        int(window_height),
        int(window_width),
        int(stride),
        row_offsets,
        col_offsets,
        int(levels),
    )
    return {
        "contrast": contrast_map,
        "energy": energy_map,
        "homogeneity": homogeneity_map,
        "score": score_map,
    }


__all__ = [
    "DEFAULT_GLCM_ANGLES",
    "DEFAULT_GLCM_DISTANCES",
    "DEFAULT_GLCM_LEVELS",
    "build_glcm_offsets",
    "glcm_map_numba",
    "glcm_score_from_crop",
    "glcm_score_numba",
]

if __name__ == "__main__":
    from skimage.feature import graycomatrix, graycoprops
    from skimage.color import rgb2hsv
    import skimage as ski

    test_img = ski.data.coffee()

    def _compare_glcm_score(image_data, x_range, y_range):
        """Compare the Numba score against the skimage reference for one crop."""
        crop = getattr(image_data, "values", image_data).squeeze()[
            y_range[0] : y_range[1], x_range[0] : x_range[1], :
        ]
        hsv_crop = img_as_ubyte(rgb2hsv(crop))
        v_channel = hsv_crop[..., 2]

        numba_score = glcm_score_from_crop(image_data, x_range=x_range, y_range=y_range)

        glcm = graycomatrix(
            v_channel,
            distances=[1, 3, 5],
            angles=[i * np.pi / 4 for i in range(4)],
            levels=256,
            symmetric=True,
            normed=True,
        )
        contrast = graycoprops(glcm, "contrast").mean()
        energy = graycoprops(glcm, "energy").mean()
        reference_score = 1.0 / (contrast + 1e-8) * energy * 100

        return numba_score, reference_score, abs(numba_score - reference_score)

    width, height = test_img.shape[1], test_img.shape[0]

    crop_specs = {
        "main crop": ((0, width // 2), (0, height // 2)),
        "alternate crop": ((0, width), (0, height)),
    }

    results = []
    for crop_name, (x_range, y_range) in crop_specs.items():
        numba_score, reference_score, abs_diff = _compare_glcm_score(
            test_img, x_range=x_range, y_range=y_range
        )
        results.append(
            {
                "crop": crop_name,
                "numba_score": numba_score,
                "skimage_score": reference_score,
                "abs_diff": abs_diff,
            }
        )

    print("GLCM Score Comparison:")
    for result in results:
        print(
            f"{result['crop']}: Numba={result['numba_score']:.6f}, "
            f"skimage={result['skimage_score']:.6f}, "
            f"abs_diff={result['abs_diff']:.6f}"
        )
