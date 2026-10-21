"""Vertical interpolation shared by coefficient generation and SEG-Y application."""

import numpy as np


def interpolate_trace_coefficients(z_samples, top_z, bottom_z, coefficients):
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
    return out
