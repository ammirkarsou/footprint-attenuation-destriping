"""Apply per-window RMS coefficients with short vertical cosine transitions."""

from math import cos, pi
import numpy as np
from numba import njit, prange

DEFAULT_VERTICAL_TAPER_FRACTION = 0.10
INTERPOLATION_METHOD = "cosine_plateau_window_fraction"

@njit(cache=True)
def _profile_knots(tops, bottoms, coefficients, valid_masks, taper_fraction):
    """Create ordered knots: unity outside windows, plateaus inside them."""
    n_windows = len(coefficients)
    selected_top = np.empty(n_windows, dtype=np.float64)
    selected_bottom = np.empty(n_windows, dtype=np.float64)
    selected_value = np.empty(n_windows, dtype=np.float64)
    count = 0
    for index in range(n_windows):
        top = tops[index]
        bottom = bottoms[index]
        value = coefficients[index]
        if not (np.isfinite(top) and np.isfinite(bottom) and np.isfinite(value)):
            continue
        if valid_masks is not None and not valid_masks[index]:
            continue
        selected_top[count] = top
        selected_bottom[count] = bottom
        selected_value[count] = value
        count += 1

    knots_z = np.empty(4 * n_windows, dtype=np.float64)
    knots_value = np.empty(4 * n_windows, dtype=np.float64)
    if count == 0:
        return knots_z, knots_value, 0

    connected = np.zeros(max(0, count - 1), dtype=np.bool_)
    boundaries = np.empty(max(0, count - 1), dtype=np.float64)
    for index in range(count - 1):
        thickness = min(selected_bottom[index] - selected_top[index],
                        selected_bottom[index + 1] - selected_top[index + 1])
        tolerance = max(1e-5, 1e-6 * thickness)
        if selected_top[index + 1] <= selected_bottom[index] + tolerance:
            connected[index] = True
            boundary = (selected_bottom[index] + selected_top[index + 1]) / 2.0
            left_center = (selected_top[index] + selected_bottom[index]) / 2.0
            right_center = (selected_top[index + 1] + selected_bottom[index + 1]) / 2.0
            if boundary <= left_center or boundary >= right_center:
                boundary = (left_center + right_center) / 2.0
            boundaries[index] = boundary

    widths = np.empty(count, dtype=np.float64)
    for index in range(count):
        left_edge = (boundaries[index - 1] if index > 0 and connected[index - 1]
                     else selected_top[index])
        right_edge = (boundaries[index] if index < count - 1 and connected[index]
                      else selected_bottom[index])
        widths[index] = min(
            taper_fraction * (selected_bottom[index] - selected_top[index]),
            0.49 * (right_edge - left_edge),
        )

    knot_count = 0
    for index in range(count):
        if index == 0 or not connected[index - 1]:
            knots_z[knot_count] = selected_top[index]
            knots_value[knot_count] = 1.0
            knot_count += 1
            knots_z[knot_count] = selected_top[index] + widths[index]
            knots_value[knot_count] = selected_value[index]
            knot_count += 1
        if index < count - 1 and connected[index]:
            knots_z[knot_count] = boundaries[index] - widths[index]
            knots_value[knot_count] = selected_value[index]
            knot_count += 1
            knots_z[knot_count] = boundaries[index] + widths[index + 1]
            knots_value[knot_count] = selected_value[index + 1]
            knot_count += 1
        else:
            knots_z[knot_count] = selected_bottom[index] - widths[index]
            knots_value[knot_count] = selected_value[index]
            knot_count += 1
            knots_z[knot_count] = selected_bottom[index]
            knots_value[knot_count] = 1.0
            knot_count += 1
    return knots_z, knots_value, knot_count


def interpolate_trace_coefficients(z_samples, top_z, bottom_z, coefficients,
                                   valid_masks=None,
                                   taper_fraction=DEFAULT_VERTICAL_TAPER_FRACTION):
    """Return factors with a coefficient plateau and tapered window edges."""
    if not 0 < taper_fraction < 0.5:
        raise ValueError("Vertical taper fraction must be between 0 and 0.5")
    samples = np.asarray(z_samples, dtype=np.float64)
    tops = np.asarray(top_z, dtype=np.float64)
    bottoms = np.asarray(bottom_z, dtype=np.float64)
    values = np.asarray(coefficients, dtype=np.float64)
    if tops.shape != bottoms.shape or tops.shape != values.shape or tops.ndim != 1:
        raise ValueError("Window tops, bottoms, and coefficients must be 1D and match")
    masks = None
    if valid_masks is not None:
        masks = np.asarray(valid_masks, dtype=np.bool_)
        if masks.shape != tops.shape:
            raise ValueError("Validity masks must match the coefficient windows")
    valid = np.isfinite(tops) & np.isfinite(bottoms) & np.isfinite(values)
    centers = (tops + bottoms) / 2.0
    order = np.argsort(np.where(valid, centers, np.inf))
    tops, bottoms, values = tops[order], bottoms[order], values[order]
    if masks is not None:
        masks = masks[order]
    ordered_centers = centers[order][valid[order]]
    if len(ordered_centers) > 1 and np.any(np.diff(ordered_centers) <= 0):
        raise ValueError("Window centers must increase at each trace")
    knots_z, knots_value, count = _profile_knots(
        tops, bottoms, values, masks, taper_fraction
    )
    out = np.ones_like(samples, dtype=np.float32)
    if count:
        inside = (np.isfinite(samples) & (samples >= knots_z[0])
                  & (samples <= knots_z[count - 1]))
        positions = samples[inside]
        segment = np.searchsorted(knots_z[:count], positions, side="right") - 1
        segment = np.clip(segment, 0, count - 2)
        left_z, right_z = knots_z[segment], knots_z[segment + 1]
        fraction = (positions - left_z) / (right_z - left_z)
        smooth = 0.5 - 0.5 * np.cos(np.pi * fraction)
        out[inside] = (knots_value[segment]
                       + (knots_value[segment + 1] - knots_value[segment]) * smooth)
    if masks is not None:
        for top, bottom, allowed in zip(tops, bottoms, masks):
            if not allowed and np.isfinite(top) and np.isfinite(bottom):
                out[(samples >= top) & (samples <= bottom)] = 1.0
    return out


@njit(parallel=True, cache=True)
def apply_coefficients_chunk(traces, inline_indices, crossline_indices,
                             z_samples, top_z, bottom_z, coefficients,
                             valid_masks=None,
                             taper_fraction=DEFAULT_VERTICAL_TAPER_FRACTION):
    """Apply the same per-window plateau profile to a trace batch."""
    n_traces, n_samples = traces.shape
    n_windows = coefficients.shape[0]
    output = traces.copy()
    changed = np.zeros(n_traces, dtype=np.uint8)
    overflow = np.zeros(n_traces, dtype=np.uint8)
    for trace_index in prange(n_traces):
        il = inline_indices[trace_index]
        xl = crossline_indices[trace_index]
        masks = (valid_masks[:, il, xl] if valid_masks is not None else None)
        knots_z, knots_value, count = _profile_knots(
            top_z[:, il, xl], bottom_z[:, il, xl],
            coefficients[:, il, xl], masks, taper_fraction,
        )
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
            if not np.isfinite(z) or z < knots_z[0] or z > knots_z[count - 1]:
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
            while segment < count - 2 and z > knots_z[segment + 1]:
                segment += 1
            left_value = knots_value[segment]
            right_value = knots_value[segment + 1]
            if left_value == right_value:
                factor = np.float32(left_value)
            else:
                fraction = ((z - knots_z[segment]) /
                            (knots_z[segment + 1] - knots_z[segment]))
                smooth = 0.5 - 0.5 * cos(pi * fraction)
                factor = np.float32(left_value + (right_value - left_value) * smooth)
            if factor != 1.0:
                changed[trace_index] = 1
                corrected = traces[trace_index, sample] * factor
                if np.isfinite(traces[trace_index, sample]) and not np.isfinite(corrected):
                    overflow[trace_index] = 1
                output[trace_index, sample] = corrected
    return output, changed, overflow
