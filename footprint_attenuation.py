import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
from pathlib import Path as FilePath
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy.ndimage import uniform_filter
from numpy.lib.stride_tricks import sliding_window_view
import re
import ast
from coefficient_interpolation import interpolate_trace_coefficients

def load_seismic_colormap(cmap_path, name="custom_seismic", smooth=True):
    valid_rows = []
    with open(cmap_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("#", "//", "!", "COLORMAP")):
                continue
            parts = line.replace(",", " ").split()
            try:
                values = [float(p) for p in parts]
                if len(values) in (3, 4, 5):
                    valid_rows.append(values)
            except ValueError:
                continue

    if not valid_rows:
        raise ValueError(f"Could not parse valid color data from {cmap_path}")

    colors = np.array(valid_rows, dtype=np.float32)
    if colors.shape[1] == 4:
        diffs = np.diff(colors[:, 0])
        if np.all(diffs >= 0) and np.allclose(diffs, np.round(diffs)):
            colors = colors[:, 1:4]
        else:
            colors = colors[:, :3]
    elif colors.shape[1] > 4:
        colors = colors[:, 1:4]

    if np.nanmax(colors) > 1.0:
        colors = colors / 255.0

    colors = np.clip(colors, 0.0, 1.0)
    return (
        LinearSegmentedColormap.from_list(name, colors, N=256)
        if smooth
        else ListedColormap(colors, name=name)
    )


# -------------------------------------------------------------
# 1. Modular Filtering Helpers
# -------------------------------------------------------------
def alpha_trimmed_mean_1d(arr, axis, win_size, alpha=0.2, chunk_size=500,
                          valid_mask=None):
    """
    Applies a 1D alpha-trimmed mean along axis 0 (inline) or axis 1 (crossline).
    """
    if win_size % 2 == 0:
        win_size += 1

    pad_w = win_size // 2
    n_trim = int(np.floor(alpha * win_size / 2.0))
    n_keep = win_size - 2 * n_trim

    if n_keep <= 0:
        raise ValueError(f"Alpha {alpha} is too high for window size {win_size}. No elements left.")

    data = np.ascontiguousarray(arr.T) if axis == 0 else np.ascontiguousarray(arr)
    nrows, ncols = data.shape
    # Sorting every filter window dominates temporary RAM on large maps.
    # Keep each batch near 4 million window values, regardless of map width.
    chunk_size = min(chunk_size, max(1, 4_000_000 // (ncols * win_size)))
    out = np.empty_like(data) if valid_mask is None else np.full_like(data, np.nan)

    padded = np.pad(data, ((0, 0), (pad_w, pad_w)), mode="edge")
    valid = None
    if valid_mask is not None:
        if valid_mask.shape != arr.shape:
            raise ValueError("valid_mask must have the same shape as the input array")
        valid = np.ascontiguousarray(valid_mask.T) if axis == 0 else np.asarray(valid_mask)
        full_rows = np.all(valid, axis=1)

    for r_start in range(0, nrows, chunk_size):
        r_end = min(r_start + chunk_size, nrows)
        rows = (slice(r_start, r_end) if valid is None else
                np.flatnonzero(full_rows[r_start:r_end]) + r_start)
        if valid is not None and len(rows) == 0:
            continue
        windows = sliding_window_view(padded[rows, :], win_size, axis=1)

        if n_trim > 0:
            sorted_win = np.sort(windows, axis=-1)
            out[rows, :] = np.mean(sorted_win[..., n_trim : win_size - n_trim], axis=-1)
        else:
            out[rows, :] = np.mean(windows, axis=-1)

    if valid is not None:
        # Filter each contiguous valid section separately. No filter window can
        # use samples from a hole or bridge the two sides of a hole.
        for row in np.flatnonzero(~full_rows):
            edges = np.diff(np.r_[False, valid[row], False].astype(np.int8))
            starts = np.flatnonzero(edges == 1)
            ends = np.flatnonzero(edges == -1)
            for start, end in zip(starts, ends):
                section = np.pad(data[row, start:end], (pad_w, pad_w), mode="edge")
                windows = sliding_window_view(section, win_size)
                if n_trim > 0:
                    sorted_win = np.sort(windows, axis=-1)
                    out[row, start:end] = np.mean(
                        sorted_win[:, n_trim : win_size - n_trim], axis=-1
                    )
                else:
                    out[row, start:end] = np.mean(windows, axis=-1)

    return np.ascontiguousarray(out.T) if axis == 0 else out


def run_alpha_trim_pipeline(data, pipeline_steps, inc_il, inc_xl,
                            valid_mask=None):
    """
    Executes a sequential chain of 1D alpha-trim filters.
    pipeline_steps: list of tuples -> [('inline'|'crossline', window_meters, alpha), ...]
    """
    result = data.copy()
    for direction, win_m, alpha in pipeline_steps:
        axis = 0 if direction == "inline" else 1
        inc = inc_il if direction == "inline" else inc_xl
        win_samples = int(np.round(win_m / inc))
        if win_samples % 2 == 0:
            win_samples += 1

        print(f" -> Running {direction.upper()} filter: {win_samples} samples ({win_m:.0f}m), alpha={alpha}")
        result = alpha_trimmed_mean_1d(
            result, axis=axis, win_size=win_samples, alpha=alpha,
            valid_mask=valid_mask
        )
    return result


# -------------------------------------------------------------
def validate_pass(parameters):
    """Normalize a pass to ((length, direction, repeats), ..., a1).

    The four-number shorthand retains the original directions and two
    repetitions for each filter.
    """
    try:
        if len(parameters) != 4:
            raise ValueError
        if all(not isinstance(value, (list, tuple)) for value in parameters[:3]):
            f1, f2, f3, multiplier = (float(value) for value in parameters)
            filters = ((f1, "inline", 2), (f2, "crossline", 2),
                       (f3, "crossline", 2))
        else:
            *filters, multiplier = parameters
            multiplier = float(multiplier)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Pass must be (f1, f2, f3, a1) or "
            "((f1, direction, repeats), (f2, direction, repeats), "
            "(f3, direction, repeats), a1)"
        ) from exc

    normalized = []
    for index, specification in enumerate(filters, start=1):
        try:
            length, direction, repeats = specification
            length = float(length)
            direction = direction.lower().strip()
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(
                f"Filter f{index} must be (length_metres, direction, repeats)"
            ) from exc
        if direction not in ("inline", "crossline"):
            raise ValueError(f"Filter f{index} direction must be 'inline' or 'crossline'")
        if isinstance(repeats, bool) or not isinstance(repeats, (int, np.integer)) or repeats < 1:
            raise ValueError(f"Filter f{index} repeats must be a positive integer")
        if not np.isfinite(length) or length <= 0:
            raise ValueError(f"Filter f{index} length must be finite and positive")
        normalized.append((length, direction, int(repeats)))

    if not np.isfinite(multiplier) or multiplier < 0:
        raise ValueError("a1 must be finite and nonnegative")
    if normalized[2][0] <= normalized[1][0]:
        raise ValueError("f3 must be longer than f2")
    return (*normalized, multiplier)


def parse_rms_map_entry(entry):
    """Accept (path, (lower, upper)) and older (path, lower, upper) entries."""
    if len(entry) == 2:
        filename, window = entry
        if window is None:
            return filename, None, None
        if len(window) != 2:
            raise ValueError("Each RMS window must be a (lower, upper) tuple")
        return filename, *window
    if len(entry) == 3:
        return entry
    raise ValueError("RMS maps must be (path, (lower, upper)) or (path, None)")


def load_horizon_grid(ilines, crosslines):
    """Interpolate the configured ASCII horizon onto the RMS map grid."""
    from create_rms_map import read_horizon, interpolate_horizon, z_scale

    points, z = read_horizon(
        horizon_file, delimiter=horizon_delimiter, columns=horizon_columns,
        fixed_widths=horizon_fixed_widths, skip_rows=horizon_skip_rows,
        missing_values=horizon_missing_values, decimal=horizon_decimal,
        encoding=horizon_encoding,
    )
    grid_il, grid_xl = np.meshgrid(ilines, crosslines, indexing="ij")
    surface = interpolate_horizon(points, z, grid_il.ravel(), grid_xl.ravel())
    surface = surface.reshape(grid_il.shape)
    if not np.any(np.isfinite(surface)):
        raise ValueError("The horizon does not cover any RMS map positions")
    return surface, z_scale(horizon_z_unit, vertical_domain)


def read_window_bounds(map_file, top, bottom, ilines, crosslines,
                       horizon_grid=None, horizon_scale=1.0):
    shape = (len(ilines), len(crosslines))
    if top is not None or bottom is not None:
        if top is None or bottom is None or not np.isfinite(top) or not np.isfinite(bottom) or top >= bottom:
            raise ValueError(f"{map_file}: provide both valid top and bottom Z values")
        if rms_window_mode == "absolute":
            return np.full(shape, top, dtype=np.float32), np.full(shape, bottom, dtype=np.float32)
        if horizon_grid is None:
            raise ValueError("Horizon-relative windows require horizon_file")
        top_grid = ((horizon_grid + top) * horizon_scale).astype(np.float32)
        bottom_grid = ((horizon_grid + bottom) * horizon_scale).astype(np.float32)
        sidecar = map_file.with_name(map_file.stem + "_window.npz")
        if sidecar.exists():
            stored_top, stored_bottom = read_window_bounds(
                map_file, None, None, ilines, crosslines
            )
            known = np.isfinite(stored_top) & np.isfinite(stored_bottom)
            if (not np.allclose(top_grid[known], stored_top[known], atol=0.01) or
                    not np.allclose(bottom_grid[known], stored_bottom[known], atol=0.01)):
                raise ValueError(f"{map_file}: horizon offsets disagree with {sidecar}")
        return top_grid, bottom_grid

    sidecar = map_file.with_name(map_file.stem + "_window.npz")
    if not sidecar.exists():
        raise FileNotFoundError(f"{sidecar}: window bounds required for vertical interpolation")
    with np.load(sidecar, allow_pickle=False) as data:
        if not np.array_equal(data["inlines"], ilines) or not np.array_equal(data["crosslines"], crosslines):
            raise ValueError(f"{sidecar}: coordinate axes differ from the RMS maps")
        if str(data["vertical_domain"]) != vertical_domain:
            raise ValueError(f"{sidecar}: vertical domain differs from configuration")
        expected_unit = "ms" if vertical_domain == "time" else "m"
        if str(data["z_unit"]) != expected_unit:
            raise ValueError(f"{sidecar}: expected vertical unit {expected_unit}")
        top_grid = np.asarray(data["top_z"], dtype=np.float32)
        bottom_grid = np.asarray(data["bottom_z"], dtype=np.float32)
    if top_grid.shape != shape or bottom_grid.shape != shape:
        raise ValueError(f"{sidecar}: window arrays do not match the RMS grid")
    valid = np.isfinite(top_grid) & np.isfinite(bottom_grid)
    if not np.any(valid):
        raise ValueError(f"{sidecar}: no finite window bounds found")
    if np.any(top_grid[valid] >= bottom_grid[valid]):
        raise ValueError(f"{sidecar}: some window tops are below their bottoms")
    return top_grid, bottom_grid


def load_holes(polygon_path):
    holes_file = polygon_path.with_name(polygon_path.stem + "_holes.npz")
    if not holes_file.exists():
        return []  # Existing single-polygon masks remain usable.
    with np.load(holes_file, allow_pickle=False) as saved:
        return [np.asarray(saved[key], dtype=float) for key in sorted(saved.files)]


def prepare_input(rms_map, ilines, crosslines, polygon, holes):
    if rms_map.shape != (len(ilines), len(crosslines)):
        raise ValueError("RMS map shape does not match the inline/crossline axes")
    if not np.any(np.isfinite(rms_map)):
        raise ValueError("RMS map contains no finite samples")
    if np.any(np.isinf(rms_map)):
        raise ValueError("RMS map contains infinite samples; check the SEG-Y amplitudes")
    n_pad_il = int(np.round(pad_inline_meters / inline_increment))
    n_pad_xl = int(np.round(pad_crossline_meters / crossline_increment))
    core = (slice(n_pad_il, n_pad_il + len(ilines)),
            slice(n_pad_xl, n_pad_xl + len(crosslines)))
    p1, p99 = np.nanpercentile(rms_map, [1.0, 99.0])
    clipped = np.nan_to_num(np.clip(rms_map, p1, p99), nan=0.0)
    padded = np.pad(clipped, ((n_pad_il, n_pad_il), (n_pad_xl, n_pad_xl)),
                    mode="constant", constant_values=0.0)

    il_step = np.median(np.diff(ilines)) if len(ilines) > 1 else 1.0
    xl_step = np.median(np.diff(crosslines)) if len(crosslines) > 1 else 1.0
    padded_il = np.r_[ilines[0] - np.arange(n_pad_il, 0, -1) * il_step,
                      ilines,
                      ilines[-1] + np.arange(1, n_pad_il + 1) * il_step]
    padded_xl = np.r_[crosslines[0] - np.arange(n_pad_xl, 0, -1) * xl_step,
                      crosslines,
                      crosslines[-1] + np.arange(1, n_pad_xl + 1) * xl_step]
    XL, IL = np.meshgrid(padded_xl, padded_il)
    points = np.column_stack((XL.ravel(), IL.ravel()))
    outer_grid = Path(polygon).contains_points(points).reshape(padded.shape)
    hole_grid = np.zeros(padded.shape, dtype=bool)
    for hole in holes:
        hole_grid |= Path(hole).contains_points(points, radius=1e-9).reshape(padded.shape)
    mask_grid = outer_grid & ~hole_grid

    win_il = max(1, int(np.round(filter_il_meters / inline_increment)) | 1)
    win_xl = max(1, int(np.round(filter_xl_meters / crossline_increment)) | 1)
    smooth = uniform_filter(mask_grid.astype(np.float32), size=(win_il, win_xl), mode="nearest")
    smooth = np.clip(smooth, 0.0, 1.0)
    smooth[smooth < 1.0] = 0.0
    smooth = uniform_filter(smooth, size=(win_il, win_xl), mode="nearest")

    inside = padded[mask_grid & (padded > 0)]
    if inside.size == 0:
        raise ValueError("No positive RMS samples remain inside the survey after excluding holes")
    median = float(np.median(inside))
    # Missing trace bins must not become zero-valued stripes inside the survey.
    padded[core][~np.isfinite(rms_map)] = median
    blended = padded * smooth + (1.0 - smooth) * median
    blended[hole_grid] = np.nan
    return {
        "input": blended, "outer": outer_grid, "holes": hole_grid,
        "valid_filter": ~hole_grid if holes else None, "core": core,
        "extent": [padded_xl.min() - xl_step / 2, padded_xl.max() + xl_step / 2,
                   padded_il.min() - il_step / 2, padded_il.max() + il_step / 2],
        "median": median,
    }


def run_pass(input_map, parameters, valid_filter):
    primary_filter, short_filter, long_filter, a1 = validate_pass(parameters)
    primary_steps = [(primary_filter[1], primary_filter[0], alpha)] * primary_filter[2]
    short_steps = [(short_filter[1], short_filter[0], alpha)] * short_filter[2]
    long_steps = [(long_filter[1], long_filter[0], alpha)] * long_filter[2]

    primary_filtered = run_alpha_trim_pipeline(
        input_map, primary_steps, inline_increment, crossline_increment, valid_filter
    )
    difference = primary_filtered - input_map
    short_difference = run_alpha_trim_pipeline(
        difference, short_steps, inline_increment, crossline_increment, valid_filter
    )
    threshold = a1 * np.nanstd(short_difference)
    large_difference = np.abs(difference) > threshold
    long_difference = run_alpha_trim_pipeline(
        difference, long_steps, inline_increment, crossline_increment, valid_filter
    )
    selected_difference = np.where(large_difference, long_difference, difference)
    filtered_difference = run_alpha_trim_pipeline(
        selected_difference, short_steps, inline_increment, crossline_increment,
        valid_filter
    )
    result = input_map + filtered_difference
    coefficient = np.ones_like(result, dtype=np.float32)
    np.divide(result, input_map, out=coefficient,
              where=np.isfinite(input_map) & (np.abs(input_map) > 1e-12))
    if valid_filter is not None:
        coefficient[~valid_filter] = np.nan
    filtered_coefficient = run_alpha_trim_pipeline(
        coefficient, short_steps, inline_increment, crossline_increment, valid_filter
    )
    output = input_map * filtered_coefficient
    return {
        "difference": difference, "short_difference": short_difference,
        "long_difference": long_difference, "filtered_difference": filtered_difference,
        "coefficient": coefficient, "filtered_coefficient": filtered_coefficient,
        "output": output, "threshold": threshold,
    }


def show_pass(map_file, number, parameters, input_map, result, extent, cmap):
    arrays = [input_map, result["difference"], result["short_difference"],
              result["long_difference"], result["filtered_difference"],
              result["coefficient"], result["filtered_coefficient"], result["output"]]
    primary_filter, short_filter, long_filter, _ = validate_pass(parameters)
    titles = ["Input blended RMS", f"{primary_filter[1].title()} difference",
              f"Short {short_filter[1]} difference",
              f"Long {long_filter[1]} difference", "Selected/filtered difference",
              "Raw coefficient", "Filtered coefficient", "Output blended RMS"]
    diff_scale = np.nanpercentile(np.abs(result["difference"]), 97)
    amplitude_scale = np.nanpercentile(np.r_[input_map.ravel(), result["output"].ravel()], 99.9)
    fig, axes = plt.subplots(2, 4, figsize=(22, 12), sharex=True, sharey=True)
    for ax, array, title in zip(axes.ravel(), arrays, titles):
        if "difference" in title.lower():
            kwargs = {"cmap": "bwr", "vmin": -diff_scale, "vmax": diff_scale}
        elif "coefficient" in title.lower():
            kwargs = {"cmap": "bwr", "vmin": 0.5, "vmax": 1.5}
        else:
            kwargs = {"cmap": cmap, "vmin": 0, "vmax": amplitude_scale}
        image = ax.imshow(array, origin="lower", aspect="equal", extent=extent, **kwargs)
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{map_file.name}: pass {number}, {parameters}, threshold={result['threshold']:.3g}")
    fig.tight_layout()
    plt.show(block=True)
    plt.close(fig)


def parse_pass_input(raw, default):
    if not raw.strip():
        if default is None:
            raise ValueError("Enter a pass tuple")
        return validate_pass(default)
    try:
        values = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        # Keep whitespace/semicolon-separated four-number input convenient.
        fields = re.split(r"[,;\s]+", raw.strip().strip("()"))
        if len(fields) != 4:
            raise ValueError(
                "Enter (f1, f2, f3, a1) or "
                "((f1, direction, repeats), (f2, direction, repeats), "
                "(f3, direction, repeats), a1)"
            ) from exc
        values = fields
    return validate_pass(values)


def select_passes(map_file, prepared, cmap):
    current = prepared["input"]
    cumulative = np.ones_like(current, dtype=np.float32)
    if prepared["valid_filter"] is not None:
        cumulative[~prepared["valid_filter"]] = np.nan
    accepted = []
    suggestion = 0
    while True:
        if parameter_input_mode == "interactive":
            default = pass_parameters[suggestion] if suggestion < len(pass_parameters) else None
            label = f"{default}" if default is not None else "pass tuple"
            raw = input(f"{map_file.name} pass {len(accepted)+1} [{label}; 'done' to finish]: ").strip()
            if raw.lower() in ("done", "stop"):
                if not accepted:
                    print("Accept at least one pass before finishing this RMS map.")
                    continue
                break
            try:
                parameters = parse_pass_input(raw, default)
            except ValueError as exc:
                print(exc)
                continue
        else:
            if suggestion >= len(pass_parameters):
                break
            parameters = validate_pass(pass_parameters[suggestion])

        result = run_pass(current, parameters, prepared["valid_filter"])
        if parameter_input_mode == "interactive":
            show_pass(map_file, len(accepted)+1, parameters, current, result,
                      prepared["extent"], cmap)
            while True:
                choice = input("Accept and [n]ext pass, accept and [f]inish map, [r]etry, or [s]top without this pass? [f]: ").strip().lower() or "f"
                if choice in ("n", "f", "r", "s"):
                    break
                print("Choose n, f, r, or s.")
            if choice == "r":
                continue
            if choice == "s":
                if not accepted:
                    print("Accept at least one pass before finishing this RMS map.")
                    continue
                break
        else:
            if preview_hardcoded_passes:
                show_pass(map_file, len(accepted)+1, parameters, current, result,
                          prepared["extent"], cmap)
            choice = "n"

        current = result["output"]
        cumulative *= result["filtered_coefficient"]
        accepted.append(parameters)
        if save_pass_differences:
            stem = map_file.with_suffix("")
            np.save(f"{stem}_pass_{len(accepted)}_diff_short.npy", result["short_difference"])
            np.save(f"{stem}_pass_{len(accepted)}_diff_long.npy", result["long_difference"])
        if choice == "f":
            break
        suggestion += 1

    if not accepted:
        raise ValueError("pass_parameters must contain at least one tuple")
    return cumulative, current, accepted


def main():
    if not rms_maps:
        raise ValueError("Configure at least one RMS map in rms_maps")
    if vertical_domain not in ("time", "depth"):
        raise ValueError("vertical_domain must be 'time' or 'depth'")
    if rms_window_mode not in ("absolute", "horizon"):
        raise ValueError("rms_window_mode must be 'absolute' or 'horizon'")
    if inline_increment <= 0 or crossline_increment <= 0:
        raise ValueError("Inline and crossline increments must be positive")
    if pad_inline_meters < 0 or pad_crossline_meters < 0:
        raise ValueError("Padding distances must be non-negative")
    if filter_il_meters <= 0 or filter_xl_meters <= 0:
        raise ValueError("Polygon blending filter lengths must be positive")
    if parameter_input_mode not in ("interactive", "hardcoded"):
        raise ValueError("parameter_input_mode must be 'interactive' or 'hardcoded'")
    if parameter_input_mode == "hardcoded" and not pass_parameters:
        raise ValueError("Configure at least one tuple in pass_parameters for hardcoded mode")
    if FilePath(output_bundle_file).suffix != ".npz":
        raise ValueError("output_bundle_file must end in .npz")
    if not FilePath(output_bundle_file).parent.is_dir():
        raise FileNotFoundError(
            f"Bundle output directory does not exist: {FilePath(output_bundle_file).parent}"
        )
    ilines = np.load(il_file)
    crosslines = np.load(xl_file)
    polygon_path = FilePath(polygon_file)
    polygon = np.load(polygon_path)
    holes = load_holes(polygon_path)
    needs_plot = parameter_input_mode == "interactive" or preview_hardcoded_passes
    if needs_plot and matplotlib.get_backend().lower() in {
        "agg", "pdf", "ps", "svg", "cairo", "template"
    }:
        raise RuntimeError(
            "Pass previews need an interactive Matplotlib backend and a "
            "graphical desktop. Install PyQt5 and run with MPLBACKEND=QtAgg "
            "if needed, or use hardcoded mode without previews."
        )
    if needs_plot and FilePath(colormap_file).is_file():
        cmap = load_seismic_colormap(colormap_file, smooth=True)
    elif needs_plot:
        print(f"Colormap file not found: {colormap_file}; using built-in seismic")
        cmap = "seismic"
    else:
        cmap = None
    print(f"Using {len(holes)} excluded hole(s)")

    # Check all vertical knots and map dimensions before starting any passes.
    map_entries = [parse_rms_map_entry(entry) for entry in rms_maps]
    horizon_grid = None
    horizon_scale = 1.0
    if rms_window_mode == "horizon" and any(top is not None for _, top, _ in map_entries):
        horizon_grid, horizon_scale = load_horizon_grid(ilines, crosslines)
    specifications = []
    sampling_settings = []
    for filename, top, bottom in map_entries:
        map_file = FilePath(filename)
        top_grid, bottom_grid = read_window_bounds(
            map_file, top, bottom, ilines, crosslines, horizon_grid, horizon_scale
        )
        rms_shape = np.load(map_file, mmap_mode="r").shape
        if rms_shape != (len(ilines), len(crosslines)):
            raise ValueError(f"{map_file}: map shape does not match the coordinate axes")
        specifications.append((map_file, top_grid, bottom_grid))
        sidecar = map_file.with_name(map_file.stem + "_window.npz")
        settings = None
        if sidecar.exists():
            with np.load(sidecar, allow_pickle=False) as saved:
                has_interval = "sample_interval" in saved.files
                has_start = "sample_start" in saved.files
                if has_interval != has_start:
                    raise ValueError(f"{sidecar}: incomplete sample-axis metadata")
                if has_interval:
                    settings = (float(saved["sample_interval"]),
                                float(saved["sample_start"]))
        sampling_settings.append(settings)
    known_settings = [settings for settings in sampling_settings if settings is not None]
    sample_metadata = {}
    if known_settings:
        reference = known_settings[0]
        if any(not np.allclose(settings, reference, rtol=0, atol=1e-6)
               for settings in known_settings[1:]):
            raise ValueError("RMS map sidecars have different sample interval/start settings")
        if len(known_settings) == len(specifications):
            sample_metadata = dict(sample_interval=reference[0], sample_start=reference[1])
        else:
            print("Some RMS maps lack sample-axis metadata; the bundle cannot verify it")
    specifications.sort(key=lambda spec: np.nanmedian((spec[1] + spec[2]) / 2))
    for previous, following in zip(specifications, specifications[1:]):
        before = (previous[1] + previous[2]) / 2
        after = (following[1] + following[2]) / 2
        valid = np.isfinite(before) & np.isfinite(after)
        if np.any(after[valid] <= before[valid]):
            raise ValueError("RMS window centers overlap or change order on the survey grid")

    completed = []
    for map_file, top_grid, bottom_grid in specifications:
        rms = np.load(map_file)
        finite_window = np.isfinite(top_grid) & np.isfinite(bottom_grid)
        if np.any(np.isfinite(rms) & ~finite_window):
            raise ValueError(f"{map_file}: finite RMS samples lack vertical window bounds")
        prepared = prepare_input(rms, ilines, crosslines, polygon, holes)
        print(f"\nProcessing {map_file.name}")
        coefficients, _, accepted = select_passes(map_file, prepared, cmap)
        core = prepared["core"]
        coeff_grid = coefficients[core].copy()
        coeff_grid[~prepared["outer"][core]] = 1.0
        coeff_grid[prepared["holes"][core]] = np.nan
        coeff_grid[~np.isfinite(rms) & ~prepared["holes"][core]] = 1.0
        corrected_rms = rms * coeff_grid
        coefficient_file = map_file.with_name(map_file.stem + "_coeff.npy")
        corrected_file = map_file.with_name(map_file.stem + "_corrected.npy")
        np.save(coefficient_file, coeff_grid)
        np.save(corrected_file, corrected_rms)
        print(f"Saved {coefficient_file} and {corrected_file}")
        completed.append((map_file, coeff_grid, top_grid, bottom_grid, accepted))

    coefficient_stack = np.stack([entry[1] for entry in completed])
    top_stack = np.stack([entry[2] for entry in completed])
    bottom_stack = np.stack([entry[3] for entry in completed])
    import json
    np.savez_compressed(
        output_bundle_file, coefficients=coefficient_stack, top_z=top_stack,
        bottom_z=bottom_stack, inlines=ilines, crosslines=crosslines,
        vertical_domain=vertical_domain,
        z_unit="ms" if vertical_domain == "time" else "m",
        rms_files=np.asarray([str(entry[0]) for entry in completed]),
        passes_json=json.dumps([entry[4] for entry in completed]),
        interpolation="linear_at_window_centers_taper_to_one_at_outer_bounds",
        **sample_metadata,
    )
    print(f"Saved vertical interpolation bundle: {output_bundle_file}")
    print("Apply coefficients at SEG-Y samples by interpolating through each trace's window centers.")


# User configuration: edit these values before running
# -------------------------------------------------------------
# "absolute": each (lower, upper) tuple is a fixed interval in ms or m.
# "horizon": each tuple is an offset from the horizon in horizon_z_unit.
# The maps are processed in vertical order, even if listed out of order.
rms_window_mode = "absolute"  # change to "horizon" for horizon-relative windows
# (RMS map, (lower, upper)). Use (RMS map, None) to read the map's
# *_window.npz sidecar instead. Older (map, lower, upper) entries also work.
rms_maps = [
    ("/data/h8/dug/brazil/teamipanema/total/brazilMPFW_100/seiTimeProc/test/QC/9999_SEGY_OUT_2/100sei/9999_SEGY_OUT_2/KDM_NorthPolygon_RMS.npy", (0.0, 1000.0)),
    # For rms_window_mode="horizon", replace the entries above with:
    # ("/path/to/RMS_horizon_1_0_1400ms.npy", (0.0, 1400.0)),
    # ("/path/to/RMS_horizon_2_1400_2800ms.npy", (1400.0, 2800.0)),
]
horizon_file = "/path/to/horizon.txt"  # required with horizon-relative tuples
horizon_z_unit = "ms"  # "ms" or "s" for time; "m" or "ft" for depth
# Match the ASCII layout settings used for this file in create_rms_map.py.
horizon_delimiter = "auto"
horizon_columns = "auto"  # or (inline_column, crossline_column, z_column), zero-based
horizon_fixed_widths = None
horizon_skip_rows = 0
horizon_missing_values = ("NaN")
horizon_decimal = "."
horizon_encoding = "utf-8-sig"
il_file = "/data/h8/dug/brazil/teamipanema/total/brazilMPFW_100/seiTimeProc/test/QC/9999_SEGY_OUT_2/100sei/9999_SEGY_OUT_2/KDM_NorthPolygon_Inlines.npy"
xl_file = "/data/h8/dug/brazil/teamipanema/total/brazilMPFW_100/seiTimeProc/test/QC/9999_SEGY_OUT_2/100sei/9999_SEGY_OUT_2/KDM_NorthPolygon_Crosslines.npy"
polygon_file = "/data/h8/dug/brazil/teamipanema/total/brazilMPFW_100/seiTimeProc/test/QC/9999_SEGY_OUT_2/100sei/9999_SEGY_OUT_2/selected_polygon.npy"
output_bundle_file = "/data/h8/dug/brazil/teamipanema/total/brazilMPFW_100/seiTimeProc/test/QC/9999_SEGY_OUT_2/100sei/9999_SEGY_OUT_2/destriping_coefficients.npz"
colormap_file = "/data/h8/dug/brazil/teamipanema/total/brazilMPFW_100/ScientificColourMaps8/oslo/oslo.alut"

vertical_domain = "time"  # time: ms; depth: m. Must match the RMS map windows.
inline_increment = 25.0  # metres between adjacent inline samples in each RMS map
crossline_increment = 25.0  # metres between adjacent crossline samples
pad_inline_meters = 3000.0
pad_crossline_meters = 200.0
filter_il_meters = 2000.0  # polygon blending aperture
filter_xl_meters = 2000.0
alpha = 0.3  # same alpha-trim value used by the previous script

# Choose how pass tuples are supplied:
#   "interactive": type/review one tuple at a time; values below are suggestions.
#   "hardcoded": run every tuple below in order for every RMS map, without prompts.
parameter_input_mode = "interactive"  # change to "hardcoded" for batch processing
# Each pass is (f1, f2, f3, a1). Each filter is
# (length_metres, "inline" or "crossline", number_of_repetitions).
# f2 is the short filter and f3 is the long filter; f3 must be longer than f2.
# The repetition count applies each time that filter is used in the pass.
# The four-number shorthand (f1, f2, f3, a1) still means 2 inline f1 filters,
# 2 crossline f2 filters, and 2 crossline f3 filters.
pass_parameters = [
    ((1000.0, "inline", 2),
     (4000.0, "crossline", 2),
     (16000.0, "crossline", 2),
     2.0),
    # ((800.0, "inline", 1), (3000.0, "crossline", 3),
    #  (12000.0, "crossline", 1), 1.5),
]

preview_hardcoded_passes = False  # display each QC plot in hardcoded mode
save_pass_differences = False  # accepted passes only, one file per aperture

if __name__ == "__main__":
    main()
