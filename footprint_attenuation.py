import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path as FilePath
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.widgets import Button
from numpy.lib.stride_tricks import sliding_window_view
import re
import ast
import json
from create_rms_map import END_OF_TRACE, TARGET_WORKING_BYTES


DEFAULT_FILTER_DIRECTIONS = ("inline", "crossline", "crossline")
DEFAULT_FILTER_REPEATS = 2


def mask_sidecar(map_file):
    path = FilePath(map_file)
    return path.with_name(path.stem + "_mask.npy")


def save_valid_mask(map_file, valid):
    np.save(mask_sidecar(map_file), np.asarray(valid, dtype=np.bool_))


def load_valid_mask(map_file, shape):
    path = mask_sidecar(map_file)
    if not path.is_file():
        return None
    valid = np.load(path, allow_pickle=False)
    if valid.shape != shape or valid.dtype != np.bool_:
        raise ValueError(f"{path}: expected a boolean mask with shape {shape}")
    return valid


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


def preview_colormap(path):
    """Use the optional custom palette when readable, otherwise viridis."""
    if not path:
        return "viridis"
    try:
        return load_seismic_colormap(path, smooth=True)
    except (OSError, ValueError) as exc:
        print(f"Colormap unavailable ({path}: {exc}); using viridis")
        return "viridis"


# -------------------------------------------------------------
# 1. 1D Directional Filtering Engine
# -------------------------------------------------------------
def alpha_trimmed_mean_1d(arr, axis, win_size, alpha=0.2):
    if win_size % 2 == 0:
        win_size += 1

    pad_w = win_size // 2
    n_trim = int(np.floor(alpha * win_size / 2.0))
    n_keep = win_size - 2 * n_trim

    if n_keep <= 0:
        raise ValueError(f"Alpha {alpha} is too high for window size {win_size}. No elements left.")

    data = np.ascontiguousarray(arr.T) if axis == 0 else np.ascontiguousarray(arr)
    nrows, ncols = data.shape
    # Allow room for the window view, its sorted copy, and the output.
    chunk_size = min(nrows, max(1, TARGET_WORKING_BYTES //
                                (4 * ncols * win_size * data.dtype.itemsize)))
    out = np.empty_like(data)

    padded = np.pad(data, ((0, 0), (pad_w, pad_w)), mode="edge")

    for r_start in range(0, nrows, chunk_size):
        r_end = min(r_start + chunk_size, nrows)
        windows = sliding_window_view(padded[r_start:r_end, :], win_size, axis=1)
        if n_trim > 0:
            sorted_win = np.sort(windows, axis=-1)
            out[r_start:r_end, :] = np.mean(sorted_win[..., n_trim : win_size - n_trim], axis=-1)
        else:
            out[r_start:r_end, :] = np.mean(windows, axis=-1)

    return np.ascontiguousarray(out.T) if axis == 0 else out


def run_alpha_trim_pipeline(data, pipeline_steps, inc_il, inc_xl):
    result = data.copy()
    for direction, win_m, alpha_val in pipeline_steps:
        axis = 0 if direction == "inline" else 1
        inc = inc_il if direction == "inline" else inc_xl
        win_samples = int(np.round(win_m / inc))
        if win_samples % 2 == 0:
            win_samples += 1

        print(f" -> Running {direction.upper()} filter: {win_samples} samples ({win_m:.0f}m), alpha={alpha_val}")
        result = alpha_trimmed_mean_1d(
            result, axis=axis, win_size=win_samples, alpha=alpha_val
        )
    return result


def validate_pass(parameters):
    """Normalize a pass; retain explicit directions/repeats for internal use."""
    try:
        if len(parameters) != 4:
            raise ValueError
        if all(not isinstance(value, (list, tuple)) for value in parameters[:3]):
            f1, f2, f3, multiplier = (float(value) for value in parameters)
            filters = tuple(
                (length, direction, DEFAULT_FILTER_REPEATS)
                for length, direction in zip(
                    (f1, f2, f3), DEFAULT_FILTER_DIRECTIONS
                )
            )
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
            raise ValueError(f"Filter f{index} must be (length_metres, direction, repeats)") from exc
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


def validate_user_pass(parameters):
    """Public pass format: only (f1, f2, f3, a1)."""
    try:
        if len(parameters) != 4 or any(
            isinstance(value, (list, tuple)) for value in parameters
        ):
            raise ValueError
        compact = tuple(float(value) for value in parameters)
    except (TypeError, ValueError) as exc:
        raise ValueError("Pass must be (f1, f2, f3, a1), all numeric") from exc
    validate_pass(compact)
    return compact


def parse_rms_map_entry(entry):
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
        if (top is None or bottom is None or not np.isfinite(top) or
                not np.isfinite(bottom) or
                (bottom != END_OF_TRACE and top >= bottom)):
            raise ValueError(f"{map_file}: provide both valid top and bottom Z values")
        if bottom == END_OF_TRACE:
            sidecar = map_file.with_name(map_file.stem + "_window.npz")
            if not sidecar.is_file():
                raise FileNotFoundError(
                    f"{sidecar}: a window sidecar is required to resolve bottom=-1"
                )
            with np.load(sidecar, allow_pickle=False) as saved:
                if "bottom_to_end" not in saved.files or not bool(saved["bottom_to_end"]):
                    raise ValueError(f"{sidecar}: window was not created with bottom=-1")
            stored_top, stored_bottom = read_window_bounds(
                map_file, None, None, ilines, crosslines
            )
            if rms_window_mode == "absolute":
                expected_top = np.full(shape, top, dtype=np.float32)
            else:
                if horizon_grid is None:
                    raise ValueError("Horizon-relative windows require horizon_file")
                expected_top = ((horizon_grid + top) * horizon_scale).astype(np.float32)
            known = np.isfinite(stored_top) & np.isfinite(expected_top)
            if not np.allclose(stored_top[known], expected_top[known], atol=0.01):
                raise ValueError(f"{map_file}: top offset disagrees with {sidecar}")
            return stored_top, stored_bottom
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
    holes_file = FilePath(polygon_path).with_name(FilePath(polygon_path).stem + "_holes.npz")
    if not holes_file.exists():
        return []
    with np.load(holes_file, allow_pickle=False) as saved:
        return [np.asarray(saved[key], dtype=float) for key in sorted(saved.files) if key.startswith("hole_")]


# -------------------------------------------------------------
# 2. Input Preparation (Pre-Blended Map Ingestion)
# -------------------------------------------------------------
def prepare_input(blended_map, ilines, crosslines):
    shape = (len(ilines), len(crosslines))
    if blended_map.shape != shape:
        raise ValueError(f"Blended RMS shape {blended_map.shape} does not match ({len(ilines)}, {len(crosslines)})")
    if not np.any(np.isfinite(blended_map)):
        raise ValueError("Blended RMS map contains no finite samples")

    n_pad_il = int(np.round(pad_inline_meters / inline_increment)) if pad_inline_meters else 0
    n_pad_xl = int(np.round(pad_crossline_meters / crossline_increment)) if pad_crossline_meters else 0
    core = (slice(n_pad_il, n_pad_il + len(ilines)),
            slice(n_pad_xl, n_pad_xl + len(crosslines)))

    il_step = np.median(np.diff(ilines)) if len(ilines) > 1 else 1.0
    xl_step = np.median(np.diff(crosslines)) if len(crosslines) > 1 else 1.0
    padded_il = np.r_[ilines[0] - np.arange(n_pad_il, 0, -1) * il_step,
                      ilines,
                      ilines[-1] + np.arange(1, n_pad_il + 1) * il_step]
    padded_xl = np.r_[crosslines[0] - np.arange(n_pad_xl, 0, -1) * xl_step,
                      crosslines,
                      crosslines[-1] + np.arange(1, n_pad_xl + 1) * xl_step]

    clean_map = np.nan_to_num(blended_map, nan=0.0)
    padded = np.pad(clean_map, ((n_pad_il, n_pad_il), (n_pad_xl, n_pad_xl)), mode="edge")

    return {
        "input": padded,
        "core": core,
        "extent": [padded_xl.min() - xl_step / 2, padded_xl.max() + xl_step / 2,
                   padded_il.min() - il_step / 2, padded_il.max() + il_step / 2],
    }


def run_pass(input_map, parameters):
    primary_filter, short_filter, long_filter, a1 = validate_pass(parameters)
    primary_steps = [(primary_filter[1], primary_filter[0], alpha)] * primary_filter[2]
    short_steps = [(short_filter[1], short_filter[0], alpha)] * short_filter[2]
    long_steps = [(long_filter[1], long_filter[0], alpha)] * long_filter[2]

    primary_filtered = run_alpha_trim_pipeline(
        input_map, primary_steps, inline_increment, crossline_increment
    )
    difference = primary_filtered - input_map
    short_difference = run_alpha_trim_pipeline(
        difference, short_steps, inline_increment, crossline_increment
    )
    threshold = a1 * np.nanstd(short_difference)
    large_difference = np.abs(difference) > threshold
    long_difference = run_alpha_trim_pipeline(
        difference, long_steps, inline_increment, crossline_increment
    )
    selected_difference = np.where(large_difference, long_difference, difference)
    filtered_difference = run_alpha_trim_pipeline(
        selected_difference, short_steps, inline_increment, crossline_increment
    )
    result = input_map + filtered_difference
    coefficient = np.ones_like(result, dtype=np.float32)
    np.divide(result, input_map, out=coefficient,
              where=np.isfinite(input_map) & (np.abs(input_map) > 1e-12))
    filtered_coefficient = run_alpha_trim_pipeline(
        coefficient, short_steps, inline_increment, crossline_increment
    )
    output = input_map * filtered_coefficient
    return {
        "difference": difference, "short_difference": short_difference,
        "long_difference": long_difference, "filtered_difference": filtered_difference,
        "coefficient": coefficient, "filtered_coefficient": filtered_coefficient,
        "output": output, "threshold": threshold,
    }


# -------------------------------------------------------------
# 3. Interactive Pass Previewer
# -------------------------------------------------------------
def show_pass(map_file, number, parameters, input_map, result, extent, cmap):
    diff_scale = float(np.nanpercentile(np.abs(result["difference"]), 97))
    amplitude_scale = float(np.nanpercentile(np.r_[input_map.ravel(), result["output"].ravel()], 99.9))

    views = {
        "input": {
            "data": input_map,
            "label": "INPUT BLENDED RMS",
            "cmap": cmap,
            "clim": (0, amplitude_scale),
            "cbar_label": "RMS Amplitude"
        },
        "output": {
            "data": result["output"],
            "label": "OUTPUT BLENDED RMS",
            "cmap": cmap,
            "clim": (0, amplitude_scale),
            "cbar_label": "RMS Amplitude"
        },
        "diff": {
            "data": result["difference"],
            "label": "PRIMARY DIFFERENCE",
            "cmap": "bwr",
            "clim": (-diff_scale, diff_scale),
            "cbar_label": "Difference Amplitude"
        },
        "coeff": {
            "data": result["filtered_coefficient"],
            "label": "FILTERED COEFFICIENT",
            "cmap": "bwr",
            "clim": (0.5, 1.5),
            "cbar_label": "Scale Factor"
        }
    }

    current_state = ["output"]

    fig, ax = plt.subplots(figsize=(15, 9))
    plt.subplots_adjust(bottom=0.10, top=0.92, left=0.07, right=0.93)

    try:
        mng = plt.get_current_fig_manager()
        if hasattr(mng, "window") and hasattr(mng.window, "showMaximized"):
            mng.window.showMaximized()
    except Exception:
        pass

    im = ax.imshow(
        views["output"]["data"],
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=views["output"]["cmap"],
        vmin=views["output"]["clim"][0],
        vmax=views["output"]["clim"][1],
        interpolation="nearest"
    )
    ax.set_xlabel("Crossline", fontsize=11)
    ax.set_ylabel("Inline", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("RMS Amplitude", fontsize=11)

    def update_view(state_name):
        current_state[0] = state_name
        cfg = views[state_name]
        im.set_data(cfg["data"])
        im.set_cmap(cfg["cmap"])
        im.set_clim(cfg["clim"])
        cbar.set_label(cfg["cbar_label"])

        status_color = "#0055ff" if state_name == "output" else "#b35900" if state_name == "input" else "#444444"
        ax.set_title(
            f"Pass {number} | {parameters} | Threshold: {result['threshold']:.3g}\n"
            f"Active View: [{cfg['label']}]",
            fontsize=12, fontweight="bold", color=status_color, pad=10
        )
        fig.suptitle(
            "[SPACE / T]: Toggle Input/Output  |  [1]: Input  |  [2]: Output  |  [D]: Diff  |  [C]: Coeff  |  [A]: Aspect  |  [ENTER]: Done",
            fontsize=10, color="gray", y=0.99
        )
        fig.canvas.draw_idle()

    def toggle_input_output(event=None):
        if current_state[0] == "output":
            update_view("input")
        else:
            update_view("output")

    def toggle_aspect(event=None):
        new_aspect = "equal" if ax.get_aspect() == "auto" else "auto"
        ax.set_aspect(new_aspect)
        btn_aspect.label.set_text(f"Aspect: {new_aspect.upper()}")
        fig.canvas.draw_idle()

    def close_viewer(event=None):
        plt.close(fig)

    def on_key(event):
        if event.key in (" ", "t", "T"):
            toggle_input_output()
        elif event.key == "1":
            update_view("input")
        elif event.key == "2":
            update_view("output")
        elif event.key in ("3", "d", "D"):
            update_view("diff")
        elif event.key in ("4", "c", "C"):
            update_view("coeff")
        elif event.key in ("a", "A"):
            toggle_aspect()
        elif event.key in ("enter", "return", "q", "escape"):
            close_viewer()

    fig.canvas.mpl_connect("key_press_event", on_key)

    ax_btn_toggle = fig.add_axes([0.15, 0.02, 0.22, 0.045])
    ax_btn_diff = fig.add_axes([0.39, 0.02, 0.14, 0.045])
    ax_btn_aspect = fig.add_axes([0.55, 0.02, 0.14, 0.045])
    ax_btn_done = fig.add_axes([0.71, 0.02, 0.16, 0.045])

    btn_toggle = Button(ax_btn_toggle, "Toggle Input / Output (Space)", color="#d9edf7", hovercolor="#bce8f1")
    btn_diff = Button(ax_btn_diff, "Difference (D)", color="#f5f5f5", hovercolor="#e0e0e0")
    btn_aspect = Button(ax_btn_aspect, "Aspect: AUTO", color="#f5f5f5", hovercolor="#e0e0e0")
    btn_done = Button(ax_btn_done, "Done / Accept (Enter)", color="#dff0d8", hovercolor="#d0e9c6")

    btn_toggle.on_clicked(toggle_input_output)
    btn_diff.on_clicked(lambda e: update_view("diff"))
    btn_aspect.on_clicked(toggle_aspect)
    btn_done.on_clicked(close_viewer)

    update_view("output")
    plt.show(block=True)
    plt.close(fig)


def parse_pass_input(raw, default):
    if not raw.strip():
        if default is None:
            raise ValueError("Enter (f1, f2, f3, a1)")
        return validate_user_pass(default)
    try:
        values = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        fields = re.split(r"[,;\s]+", raw.strip().strip("()"))
        if len(fields) != 4:
            raise ValueError("Enter (f1, f2, f3, a1)") from exc
        values = fields
    return validate_user_pass(values)


def select_passes(map_file, prepared, cmap):
    current = prepared["input"]
    cumulative = np.ones_like(current, dtype=np.float32)
    accepted = []
    suggestion = 0

    while True:
        if parameter_input_mode == "interactive":
            default = pass_parameters[suggestion] if suggestion < len(pass_parameters) else None
            label = f"{default}" if default is not None else "(f1, f2, f3, a1)"
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
            parameters = validate_user_pass(pass_parameters[suggestion])

        result = run_pass(current, parameters)
        if parameter_input_mode == "interactive":
            show_pass(map_file, len(accepted)+1, parameters, current, result,
                      prepared["extent"], cmap)
            while True:
                default_choice = "n" if suggestion + 1 < len(pass_parameters) else "f"
                choice = input(
                    "Accept and [n]ext pass, accept and [f]inish map, "
                    f"[r]etry, or [s]top without this pass? [{default_choice}]: "
                ).strip().lower() or default_choice
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


# -------------------------------------------------------------
# 4. Main Processing Routine
# -------------------------------------------------------------
def main():
    missing = [name for name in (
        "rms_maps", "il_file", "xl_file", "output_bundle_file",
        "vertical_domain", "inline_increment", "crossline_increment",
        "pad_inline_meters", "pad_crossline_meters", "alpha",
        "parameter_input_mode", "pass_parameters",
    ) if globals()[name] is None]
    if missing:
        raise ValueError("Missing attenuation settings from pipeline YAML: " +
                         ", ".join(missing))
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
    if parameter_input_mode not in ("interactive", "hardcoded"):
        raise ValueError("parameter_input_mode must be 'interactive' or 'hardcoded'")
    if parameter_input_mode == "hardcoded" and not pass_parameters:
        raise ValueError("Configure at least one tuple in pass_parameters for hardcoded mode")
    for index, parameters in enumerate(pass_parameters, start=1):
        try:
            validate_user_pass(parameters)
        except ValueError as exc:
            raise ValueError(f"Invalid pass_parameters entry {index}: {exc}") from exc
    if FilePath(output_bundle_file).suffix != ".npz":
        raise ValueError("output_bundle_file must end in .npz")
    if not FilePath(output_bundle_file).parent.is_dir():
        raise FileNotFoundError(
            f"Bundle output directory does not exist: {FilePath(output_bundle_file).parent}"
        )

    ilines = np.load(il_file)
    crosslines = np.load(xl_file)
    survey_shape = (len(ilines), len(crosslines))

    needs_plot = parameter_input_mode == "interactive" or preview_hardcoded_passes
    if needs_plot and matplotlib.get_backend().lower() in {
        "agg", "pdf", "ps", "svg", "cairo", "template"
    }:
        raise RuntimeError(
            "Pass previews need an interactive Matplotlib backend and a "
            "graphical desktop. Install PyQt5 and run with MPLBACKEND=QtAgg "
            "if needed, or use hardcoded mode without previews."
        )
    cmap = preview_colormap(colormap_file) if needs_plot else None

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
        if rms_shape != survey_shape:
            raise ValueError(f"{map_file}: map shape does not match coordinate axes")
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
        blended_rms = np.load(map_file)
        finite_window = np.isfinite(top_grid) & np.isfinite(bottom_grid)
        valid_mask = load_valid_mask(map_file, survey_shape)

        # Check raw unblended RMS map if present (to verify original finite bounds)
        raw_map_file = map_file.with_name(map_file.name.replace("_blended.npy", ".npy"))
        if raw_map_file.is_file() and raw_map_file != map_file:
            raw_rms = np.load(raw_map_file)
            if np.any(np.isfinite(raw_rms) & ~finite_window):
                raise ValueError(f"{raw_map_file.name}: finite raw RMS samples lack vertical window bounds")
        else:
            raw_rms = None

        print(f"\nProcessing Destriping on Blended Map: {map_file.name}")
        prepared = prepare_input(blended_rms, ilines, crosslines)

        coefficients, _, accepted = select_passes(map_file, prepared, cmap)

        core = prepared["core"]
        coeff_grid = coefficients[core].copy()

        # Outside horizon bounds and the selected survey boundary, including
        # holes, lock coefficients to 1.0 so the SEG-Y samples are unchanged.
        coeff_grid[~finite_window] = 1.0
        if valid_mask is not None:
            coeff_grid[~valid_mask] = 1.0
        if raw_rms is not None:
            coeff_grid[~np.isfinite(raw_rms)] = 1.0
            coeff_grid[raw_rms <= 0] = 1.0
            corrected_rms = raw_rms * coeff_grid
        else:
            coeff_grid[~np.isfinite(blended_rms)] = 1.0
            coeff_grid[blended_rms <= 0] = 1.0
            corrected_rms = blended_rms * coeff_grid
            corrected_rms[~finite_window] = np.nan

        coefficient_file = map_file.with_name(map_file.stem + "_coeff.npy")
        corrected_file = map_file.with_name(map_file.stem + "_corrected.npy")
        np.save(coefficient_file, coeff_grid)
        np.save(corrected_file, corrected_rms)
        print(f"Saved {coefficient_file.name} and {corrected_file.name}")
        completed.append((
            map_file, coeff_grid, top_grid, bottom_grid,
            valid_mask if valid_mask is not None else np.ones(survey_shape, dtype=np.bool_),
            accepted,
        ))

    coefficient_stack = np.stack([entry[1] for entry in completed])
    top_stack = np.stack([entry[2] for entry in completed])
    bottom_stack = np.stack([entry[3] for entry in completed])
    mask_stack = np.stack([entry[4] for entry in completed])

    np.savez_compressed(
        output_bundle_file, coefficients=coefficient_stack, top_z=top_stack,
        bottom_z=bottom_stack, valid_masks=mask_stack,
        inlines=ilines, crosslines=crosslines,
        vertical_domain=vertical_domain,
        z_unit="ms" if vertical_domain == "time" else "m",
        rms_files=np.asarray([str(entry[0]) for entry in completed]),
        passes_json=json.dumps([entry[5] for entry in completed]),
        interpolation="linear_at_window_centers_taper_to_one_at_outer_bounds",
        **sample_metadata,
    )
    print(f"\nSaved vertical interpolation bundle: {output_bundle_file}")
    print("Apply coefficients at SEG-Y samples by interpolating through each trace's window centers.")


# Runtime paths and processing settings are supplied by run_pipeline.py.
rms_window_mode = None
rms_maps = None
horizon_file = None
horizon_z_unit = None
horizon_delimiter = None
horizon_columns = None
horizon_fixed_widths = None
horizon_skip_rows = None
horizon_missing_values = None
horizon_decimal = None
horizon_encoding = None
il_file = None
xl_file = None
output_bundle_file = None
colormap_file = None
vertical_domain = None
inline_increment = None
crossline_increment = None
pad_inline_meters = None
pad_crossline_meters = None
alpha = None
parameter_input_mode = None
pass_parameters = None
preview_hardcoded_passes = False
save_pass_differences = False

if __name__ == "__main__":
    raise SystemExit("Run python run_pipeline.py <parameter-file.yml> to configure attenuation")
