"""Vertical interpolation shared by coefficient generation and SEG-Y application."""

import numpy as np
from numba import njit, prange


def interpolate_trace_coefficients(z_samples, top_z, bottom_z, coefficients,
                                   valid_masks=None):
    """Interpolate map coefficients at one trace's vertical samples.

    Returns 1 outside the outer window bounds and for missing/holed picks.
    Within the bounds, values vary linearly through window centers and taper
    continuously to 1 at the first top and last bottom.
    """
    z_samples = np.asarray(z_samples, dtype=float)
    tops = np.asarray(top_z, dtype=float)
    bottoms = np.asarray(bottom_z, dtype=float)
    values = np.asarray(coefficients, dtype=float)
    valid = np.isfinite(tops) & np.isfinite(bottoms) & np.isfinite(values)
    out = np.ones_like(z_samples, dtype=np.float32)
    if not np.any(valid):
        return out
    centers = (tops[valid] + bottoms[valid]) / 2
    order = np.argsort(centers)
    centers = centers[order]
    values = values[valid][order]
    if len(centers) > 1 and np.any(np.diff(centers) <= 0):
        raise ValueError("Window centers must increase at each trace")
    start = np.min(tops[valid])
    end = np.max(bottoms[valid])
    knots = np.r_[start, centers, end]
    knot_values = np.r_[1.0, values, 1.0]
    inside = np.isfinite(z_samples) & (z_samples >= start) & (z_samples <= end)
    out[inside] = np.interp(z_samples[inside], knots, knot_values)
    if valid_masks is not None:
        mask = np.asarray(valid_masks, dtype=np.bool_)
        if mask.shape != tops.shape:
            raise ValueError("Validity masks must match the coefficient windows")
        for top, bottom, allowed in zip(tops, bottoms, mask):
            if not allowed and np.isfinite(top) and np.isfinite(bottom):
                out[(z_samples >= top) & (z_samples <= bottom)] = 1.0
    return out


@njit(parallel=True, cache=True)
def apply_coefficients_chunk(traces, inline_indices, crossline_indices,
                             z_samples, top_z, bottom_z, coefficients,
                             valid_masks=None):
    """Interpolate and multiply a trace batch without per-trace Python work."""
    n_traces, n_samples = traces.shape
    n_windows = coefficients.shape[0]
    output = traces.copy()
    changed = np.zeros(n_traces, dtype=np.uint8)
    overflow = np.zeros(n_traces, dtype=np.uint8)
    for trace_index in prange(n_traces):
        il = inline_indices[trace_index]
        xl = crossline_indices[trace_index]
        centers = np.empty(n_windows, dtype=np.float64)
        values = np.empty(n_windows, dtype=np.float64)
        count = 0
        first_top = np.inf
        last_bottom = -np.inf
        for window in range(n_windows):
            top = top_z[window, il, xl]
            bottom = bottom_z[window, il, xl]
            value = coefficients[window, il, xl]
            if not (np.isfinite(top) and np.isfinite(bottom) and np.isfinite(value)):
                continue
            centers[count] = (float(top) + float(bottom)) / 2.0
            values[count] = float(value)
            count += 1
            first_top = min(first_top, top)
            last_bottom = max(last_bottom, bottom)
        if count == 0:
            continue
        has_exclusion = False
        if valid_masks is not None:
            for window in range(n_windows):
                if not valid_masks[window, il, xl]:
                    has_exclusion = True
                    break
        segment = 0
        for sample in range(n_samples):
            z = z_samples[sample]
            if not np.isfinite(z) or z < first_top or z > last_bottom:
                continue
            if has_exclusion:
                excluded = False
                for window in range(n_windows):
                    if not valid_masks[window, il, xl]:
                        top = top_z[window, il, xl]
                        bottom = bottom_z[window, il, xl]
                        if np.isfinite(top) and np.isfinite(bottom) and top <= z <= bottom:
                            excluded = True
                            break
                if excluded:
                    continue
            while segment < count and z > centers[segment]:
                segment += 1
            if segment == 0:
                left_z, right_z = first_top, centers[0]
                left_value, right_value = 1.0, values[0]
            elif segment == count:
                left_z, right_z = centers[count - 1], last_bottom
                left_value, right_value = values[count - 1], 1.0
            else:
                left_z, right_z = centers[segment - 1], centers[segment]
                left_value, right_value = values[segment - 1], values[segment]
            factor = np.float32(left_value + (right_value - left_value) *
                                (z - left_z) / (right_z - left_z))
            if factor != 1.0:
                changed[trace_index] = 1
                corrected = traces[trace_index, sample] * factor
                if np.isfinite(traces[trace_index, sample]) and not np.isfinite(corrected):
                    overflow[trace_index] = 1
                output[trace_index, sample] = corrected
    return output, changed, overflow
