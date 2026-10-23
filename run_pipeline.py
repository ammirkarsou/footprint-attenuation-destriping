"""Run RMS creation, masking, attenuation, and SEG-Y compensation from YAML."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import segyio
import yaml
from matplotlib.path import Path as PolygonPath

import apply_destriping
import create_rms_map
import footprint_attenuation
from footprint_attenuation import mask_sidecar, save_valid_mask, load_valid_mask


RMS_KEYS = {
    "window_mode", "vertical_domain", "constant_top", "constant_bottom",
    "horizon_file", "horizon_z_unit", "horizon_windows",
    "horizon_delimiter", "horizon_columns", "horizon_fixed_widths",
    "horizon_skip_rows", "horizon_missing_values", "horizon_decimal",
    "horizon_encoding", "sample_interval_override", "sample_start_override",
    "existing_maps", "inlines_file", "crosslines_file", "trace_chunk_size",
}
MASK_KEYS = {
    "mode", "scope", "outer", "holes", "preview_map_index",
    "pad_inline_meters", "pad_crossline_meters", "existing_blended_maps",
}
ATTENUATION_KEYS = {
    "pad_inline_meters", "pad_crossline_meters", "filter_il_meters",
    "filter_xl_meters", "hole_filter_il_meters", "hole_filter_xl_meters",
    "alpha", "parameter_input_mode", "pass_parameters",
    "preview_hardcoded_passes", "colormap_file", "save_pass_differences",
}


def section(config, name, allowed, *, required=True):
    value = config.get(name)
    if value is None and not required:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} setting(s): {', '.join(sorted(unknown))}")
    return value


def path_from_yaml(value, base, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty path")
    path = Path(os.path.expandvars(value)).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def polygon_points(value, name):
    try:
        points = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain [crossline, inline] coordinate pairs") from exc
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        raise ValueError(f"{name} must have at least three [crossline, inline] vertices")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} contains nonfinite coordinates")
    return points


def window_sidecar(map_file):
    return map_file.with_name(map_file.stem + "_window.npz")


def choose_saved_files(stage, required_paths, action):
    present = [path for path in required_paths if path.is_file()]
    if not present:
        if action == "reuse":
            raise FileNotFoundError(
                f"Cannot reuse {stage}; required files are missing: " +
                ", ".join(str(path) for path in required_paths)
            )
        return False
    complete = len(present) == len(required_paths)
    if action == "reuse":
        if not complete:
            missing = [str(path) for path in required_paths if not path.is_file()]
            raise FileNotFoundError(
                f"Cannot reuse incomplete {stage} files; missing: {', '.join(missing)}"
            )
        print(f"Reusing saved {stage}", flush=True)
        return True
    if action == "recalculate":
        print(f"Recalculating {stage} and overwriting saved files", flush=True)
        return False
    print(f"Saved {stage} files found:", flush=True)
    for path in present:
        print(f"  {path}", flush=True)
    if not complete:
        print("Some required files are missing; reuse is unavailable.", flush=True)
    while True:
        choices = ("[u]se for destriping, [r]ecalculate and overwrite, [q]uit"
                   if complete else "[r]ecalculate and overwrite, [q]uit")
        try:
            answer = input(f"{stage}: {choices}? ").strip().lower()
        except EOFError as exc:
            raise RuntimeError(
                "A saved-file decision needs keyboard input; set on_existing "
                "to 'reuse' or 'recalculate' for unattended runs"
            ) from exc
        if answer in ("u", "use", "reuse") and complete:
            print(f"Reusing saved {stage}", flush=True)
            return True
        if answer in ("r", "recalculate", "overwrite"):
            print(f"Recalculating {stage} and overwriting saved files", flush=True)
            return False
        if answer in ("q", "quit", "cancel"):
            raise RuntimeError(f"Stopped before changing {stage} files")
        print("Please choose one of the displayed options.", flush=True)


def publish_file(source, target):
    """Replace one persistent cache file after its new version is complete."""
    if source == target:
        return
    if not target.parent.is_dir():
        raise FileNotFoundError(f"Saved-file directory does not exist: {target.parent}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=f".{target.name}.", suffix=".partial",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_saved_rms(maps, il_file, xl_file, source, domain):
    inlines = np.load(il_file, allow_pickle=False)
    crosslines = np.load(xl_file, allow_pickle=False)
    if (inlines.ndim != 1 or crosslines.ndim != 1 or
            not np.all(np.diff(inlines) > 0) or
            not np.all(np.diff(crosslines) > 0)):
        raise ValueError("Saved RMS inline/crossline axes are invalid")
    with segyio.open(str(source), mode="r", strict=False, ignore_geometry=True) as segy:
        source_il = np.unique(segy.attributes(segyio.TraceField.INLINE_3D)[:])
        source_xl = np.unique(segy.attributes(segyio.TraceField.CROSSLINE_3D)[:])
    if not np.array_equal(inlines, source_il) or not np.array_equal(crosslines, source_xl):
        raise ValueError("Saved RMS axes differ from the input SEG-Y")
    expected_unit = "ms" if domain == "time" else "m"
    shape = (len(inlines), len(crosslines))
    for map_file in maps:
        map_data = np.load(map_file, mmap_mode="r", allow_pickle=False)
        if map_data.shape != shape or not np.any(np.isfinite(map_data)):
            raise ValueError(f"Saved RMS map has invalid shape or no data: {map_file}")
        with np.load(window_sidecar(map_file), allow_pickle=False) as saved:
            if (not np.array_equal(saved["inlines"], inlines) or
                    not np.array_equal(saved["crosslines"], crosslines) or
                    str(saved["vertical_domain"]) != domain or
                    str(saved["z_unit"]) != expected_unit or
                    saved["top_z"].shape != shape or
                    saved["bottom_z"].shape != shape):
                raise ValueError(f"Saved RMS window metadata does not match: {map_file}")


def get_mask_scope(mask_config, map_paths, working_dir, cache_dir):
    """Resolve single vs multiple mask scope from config, create_rms_map sidecar, or prompt."""
    if len(map_paths) <= 1:
        return "single"
    if mask_config.get("scope") in ("single", "multiple"):
        return mask_config["scope"]

    for parent in [working_dir, cache_dir, map_paths[0].parent]:
        if parent is not None:
            cand = parent / (map_paths[0].stem + "_mask_scope.txt")
            if cand.is_file():
                txt = cand.read_text(encoding="utf-8").strip().lower()
                if txt in ("single", "multiple"):
                    return txt

    while True:
        try:
            choice = input(
                "\nDo you want to use a [s]ingle mask for all RMS maps, or [m]ultiple masks? [single/multiple] (default: single): "
            ).strip().lower()
        except EOFError:
            return "single"
        if choice in ("", "s", "single", "1"):
            return "single"
        if choice in ("m", "multiple", "2", "each"):
            return "multiple"
        print("Please enter 'single' or 'multiple'.")


def blend_configured_mask(mask, rms_maps, il_unique, xl_unique, blended_outputs, spacing, attenuation):
    """Headless blending fallback when polygon vertices are specified directly in YAML."""
    from scipy.ndimage import uniform_filter

    outer = polygon_points(mask.get("outer"), "mask.outer")
    raw_holes = mask.get("holes", [])
    if not isinstance(raw_holes, list):
        raise ValueError("mask.holes must be a list of polygons")
    holes = [polygon_points(v, f"mask.holes[{i}]") for i, v in enumerate(raw_holes)]

    pad_il = mask["pad_inline_meters"]
    pad_xl = mask["pad_crossline_meters"]
    n_pad_il = int(np.round(pad_il / spacing[0]))
    n_pad_xl = int(np.round(pad_xl / spacing[1]))

    il_step = np.median(np.diff(il_unique)) if len(il_unique) > 1 else 1.0
    xl_step = np.median(np.diff(xl_unique)) if len(xl_unique) > 1 else 1.0

    padded_il = np.r_[il_unique[0] - np.arange(n_pad_il, 0, -1) * il_step,
                      il_unique,
                      il_unique[-1] + np.arange(1, n_pad_il + 1) * il_step]
    padded_xl = np.r_[xl_unique[0] - np.arange(n_pad_xl, 0, -1) * xl_step,
                      xl_unique,
                      xl_unique[-1] + np.arange(1, n_pad_xl + 1) * xl_step]
    XL, IL = np.meshgrid(padded_xl, padded_il)
    points = np.column_stack((XL.ravel(), IL.ravel()))

    outer_grid = PolygonPath(outer).contains_points(points).reshape(len(padded_il), len(padded_xl))
    hole_grid = np.zeros(outer_grid.shape, dtype=bool)
    for hole in holes:
        hole_grid |= PolygonPath(hole).contains_points(points, radius=1e-9).reshape(outer_grid.shape)

    b_il = attenuation["filter_il_meters"]
    b_xl = attenuation["filter_xl_meters"]
    h_il = attenuation.get("hole_filter_il_meters")
    h_xl = attenuation.get("hole_filter_xl_meters")
    h_il = b_il if h_il is None else float(h_il)
    h_xl = b_xl if h_xl is None else float(h_xl)

    def meters_to_window(m, inc):
        if m <= 0.0:
            return 1
        w = int(np.round(m / inc))
        if w <= 1:
            return 3
        if w % 2 == 0:
            w += 1
        return w

    def smooth_binary(grid, win_il, win_xl):
        if win_il <= 1 and win_xl <= 1:
            return grid.astype(np.float32)
        s = uniform_filter(grid.astype(np.float32), size=(max(1, win_il), max(1, win_xl)), mode="nearest")
        s = np.clip(s, 0.0, 1.0)
        s[s < 1.0] = 0.0
        s = uniform_filter(s, size=(max(1, win_il), max(1, win_xl)), mode="nearest")
        return np.clip(s, 0.0, 1.0)

    s_outer = smooth_binary(outer_grid, meters_to_window(b_il, spacing[0]), meters_to_window(b_xl, spacing[1]))
    s_holes = smooth_binary(~hole_grid, meters_to_window(h_il, spacing[0]), meters_to_window(h_xl, spacing[1])) if holes else 1.0
    smooth = s_outer * s_holes

    mask_grid = outer_grid & ~hole_grid
    il_slice = slice(n_pad_il, n_pad_il + len(il_unique)) if n_pad_il > 0 else slice(None)
    xl_slice = slice(n_pad_xl, n_pad_xl + len(xl_unique)) if n_pad_xl > 0 else slice(None)
    valid_core = mask_grid[il_slice, xl_slice]

    for r_path, out_path in zip(rms_maps, blended_outputs):
        rms = np.load(r_path)
        p1, p99 = np.nanpercentile(rms, [1.0, 99.0])
        clipped = np.nan_to_num(np.clip(rms, p1, p99), nan=0.0)
        padded = np.pad(clipped, ((n_pad_il, n_pad_il), (n_pad_xl, n_pad_xl)), mode="constant", constant_values=0.0)
        inside = padded[mask_grid & (padded > 0)]
        med = float(np.median(inside)) if inside.size > 0 else 0.0
        blended = padded * smooth + (1.0 - smooth) * med
        np.save(out_path, blended[il_slice, xl_slice])
        save_valid_mask(out_path, valid_core)


def draw_mask_batch(mask, maps, il_file, xl_file, spacing, scope, attenuation):
    command = [
        sys.executable, str(Path(__file__).with_name("create_mask.py")),
        "--rms-files", *[str(m) for m in maps],
        "--il-file", str(il_file),
        "--xl-file", str(xl_file),
        "--scope", str(scope),
        "--preview-index", str(mask.get("preview_map_index", 0)),
        "--inline-increment", str(spacing[0]),
        "--crossline-increment", str(spacing[1]),
        "--pad-inline-meters", str(mask["pad_inline_meters"]),
        "--pad-crossline-meters", str(mask["pad_crossline_meters"]),
        "--filter-il-meters", str(attenuation["filter_il_meters"]),
        "--filter-xl-meters", str(attenuation["filter_xl_meters"]),
    ]
    for key in ("hole_filter_il_meters", "hole_filter_xl_meters"):
        if key in attenuation and attenuation[key] is not None:
            cli_flag = f"--{key.replace('_', '-')}"
            command.extend([cli_flag, str(attenuation[key])])

    subprocess.run(command, check=True)


def run_pipeline(config, config_file):
    if not isinstance(config, dict):
        raise ValueError("YAML root must be a mapping")
    unknown = set(config) - {
        "input_segy", "output_segy", "temporary_directory", "geometry",
        "cache_directory", "on_existing", "rms", "mask", "attenuation",
        "application",
    }
    if unknown:
        raise ValueError(f"Unknown top-level setting(s): {', '.join(sorted(unknown))}")
    base = Path(config_file).resolve().parent
    source = path_from_yaml(config.get("input_segy"), base, "input_segy")
    target = path_from_yaml(config.get("output_segy"), base, "output_segy")
    if not source.is_file():
        raise FileNotFoundError(f"Input SEG-Y not found: {source}")
    if source == target:
        raise ValueError("output_segy must differ from input_segy")
    if not target.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {target.parent}")
    temporary_parent = path_from_yaml(
        config.get("temporary_directory", str(target.parent)), base,
        "temporary_directory",
    )
    if not temporary_parent.is_dir():
        raise FileNotFoundError(f"Temporary directory does not exist: {temporary_parent}")
    cache_directory = None
    if config.get("cache_directory") is not None:
        cache_directory = path_from_yaml(
            config["cache_directory"], base, "cache_directory"
        )
        if cache_directory.exists() and not cache_directory.is_dir():
            raise NotADirectoryError(f"Cache path is not a directory: {cache_directory}")
    on_existing = config.get("on_existing", "prompt")
    if on_existing not in ("prompt", "reuse", "recalculate"):
        raise ValueError("on_existing must be 'prompt', 'reuse', or 'recalculate'")

    geometry = section(config, "geometry", {"inline_spacing_m", "crossline_spacing_m"})
    try:
        spacing = (float(geometry["inline_spacing_m"]),
                   float(geometry["crossline_spacing_m"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("geometry requires inline_spacing_m and crossline_spacing_m") from exc
    if not all(np.isfinite(value) and value > 0 for value in spacing):
        raise ValueError("Geometry spacing must be finite and positive")

    rms = section(config, "rms", RMS_KEYS)
    mask = section(config, "mask", MASK_KEYS)
    attenuation = section(config, "attenuation", ATTENUATION_KEYS)
    application = section(
        config, "application", {"overwrite_output", "trace_chunk_size"}, required=False
    )
    if not isinstance(application.get("overwrite_output", False), bool):
        raise ValueError("application.overwrite_output must be true or false")
    if "trace_chunk_size" in application:
        print("Ignoring application.trace_chunk_size; chunk size is automatic")
    if target.exists() and not application.get("overwrite_output", False):
        raise FileExistsError(f"Output already exists: {target}")

    mode = rms.get("window_mode")
    if mode not in ("constant", "horizon"):
        raise ValueError("rms.window_mode must be 'constant' or 'horizon'")
    domain = rms.get("vertical_domain")
    if domain not in ("time", "depth"):
        raise ValueError("rms.vertical_domain must be 'time' or 'depth'")
    sample_settings = {}
    for key in ("sample_interval_override", "sample_start_override"):
        value = rms.get(key)
        if value is not None:
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"rms.{key} must be numeric or null") from exc
            if not np.isfinite(value) or (key == "sample_interval_override" and value <= 0):
                raise ValueError(f"rms.{key} has an invalid value")
        sample_settings[key] = value
    if "trace_chunk_size" in rms:
        print("Ignoring rms.trace_chunk_size; chunk size is automatic")
    if mode == "constant":
        if "constant_top" not in rms or "constant_bottom" not in rms:
            raise ValueError("Constant RMS mode needs constant_top and constant_bottom")
        try:
            top = float(rms["constant_top"])
            bottom = float(rms["constant_bottom"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Constant RMS top/bottom must be numeric") from exc
        if (not np.isfinite(top) or not np.isfinite(bottom) or
                (bottom != create_rms_map.END_OF_TRACE and top >= bottom)):
            raise ValueError("Constant RMS window requires finite top < bottom, or bottom: -1")
    else:
        if ("horizon_file" not in rms or "horizon_z_unit" not in rms or
                not rms.get("horizon_windows")):
            raise ValueError(
                "Horizon RMS mode needs horizon_file, horizon_z_unit, and horizon_windows"
            )
        horizon_file = path_from_yaml(rms["horizon_file"], base, "rms.horizon_file")
        if not isinstance(rms["horizon_windows"], list):
            raise ValueError("rms.horizon_windows must be a list of [lower, upper] pairs")
        windows = []
        for index, pair in enumerate(rms["horizon_windows"]):
            try:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ValueError
                lower, upper = (float(value) for value in pair)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"rms.horizon_windows[{index}] must be [lower, upper]") from exc
            if (not np.isfinite(lower) or not np.isfinite(upper) or
                    (upper != create_rms_map.END_OF_TRACE and lower >= upper)):
                raise ValueError(
                    f"rms.horizon_windows[{index}] needs finite lower < upper, or upper: -1"
                )
            windows.append((lower, upper))

    mask_mode = mask.get("mode")
    if mask_mode not in ("interactive", "vertices"):
        raise ValueError("mask.mode must be 'interactive' or 'vertices'")
    for key in ("pad_inline_meters", "pad_crossline_meters"):
        if key not in mask:
            raise ValueError(f"mask.{key} is required in the parameter file")
        try:
            value = float(mask[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"mask.{key} must be numeric") from exc
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"mask.{key} must be finite and nonnegative")
        mask[key] = value
    if mask_mode == "interactive":
        preview_index = mask.get("preview_map_index", 0)
        map_count = 1 if mode == "constant" else len(windows)
        if not isinstance(preview_index, int) or not 0 <= preview_index < map_count:
            raise ValueError("mask.preview_map_index is outside the RMS map list")
    if mask_mode == "vertices":
        outer = polygon_points(mask.get("outer"), "mask.outer")
        if not isinstance(mask.get("holes", []), list):
            raise ValueError("mask.holes must be a list of polygons")
        outer_path = PolygonPath(np.vstack((outer, outer[0])))
        for index, raw_hole in enumerate(mask.get("holes", [])):
            hole = polygon_points(raw_hole, f"mask.holes[{index}]")
            hole_path = PolygonPath(np.vstack((hole, hole[0])))
            if (not outer_path.contains_points(hole).all() or
                    outer_path.intersects_path(hole_path, filled=False)):
                raise ValueError(f"mask.holes[{index}] must lie inside mask.outer")

    existing_maps = rms.get("existing_maps")
    if existing_maps is not None:
        if not isinstance(existing_maps, list) or not existing_maps:
            raise ValueError("rms.existing_maps must be a nonempty list of paths")
        expected = 1 if mode == "constant" else len(windows)
        if len(existing_maps) != expected:
            raise ValueError(f"rms.existing_maps needs {expected} map path(s)")
        if "inlines_file" not in rms or "crosslines_file" not in rms:
            raise ValueError("rms.existing_maps also needs inlines_file and crosslines_file")
        existing_maps = [path_from_yaml(path, base, "rms.existing_maps")
                         for path in existing_maps]
        existing_il = path_from_yaml(rms["inlines_file"], base, "rms.inlines_file")
        existing_xl = path_from_yaml(rms["crosslines_file"], base, "rms.crosslines_file")
    elif "inlines_file" in rms or "crosslines_file" in rms:
        raise ValueError("rms.inlines_file and crosslines_file require existing_maps")

    existing_blended = None
    if "existing_blended_maps" in mask:
        raw_b = mask["existing_blended_maps"]
        if not isinstance(raw_b, list) or not raw_b:
            raise ValueError("mask.existing_blended_maps must be a nonempty list of paths")
        expected = 1 if mode == "constant" else len(windows)
        if len(raw_b) != expected:
            raise ValueError(f"mask.existing_blended_maps needs {expected} map path(s)")
        existing_blended = [path_from_yaml(p, base, "mask.existing_blended_maps") for p in raw_b]

    input_mode = attenuation.get("parameter_input_mode")
    if input_mode not in ("hardcoded", "interactive"):
        raise ValueError("attenuation.parameter_input_mode must be hardcoded or interactive")
    passes = attenuation.get("pass_parameters", [])
    if not isinstance(passes, list) or (input_mode == "hardcoded" and not passes):
        raise ValueError("attenuation.pass_parameters must be a nonempty list in hardcoded mode")
    normalized_passes = []
    for index, parameters in enumerate(passes):
        try:
            normalized_passes.append(footprint_attenuation.validate_user_pass(parameters))
        except ValueError as exc:
            raise ValueError(f"Invalid attenuation pass {index + 1}: {exc}") from exc
    normalized_attenuation = dict(attenuation)
    required_attenuation = (
        "pad_inline_meters", "pad_crossline_meters", "filter_il_meters",
        "filter_xl_meters", "alpha",
    )
    for key in required_attenuation:
        if attenuation.get(key) is None:
            raise ValueError(f"attenuation.{key} is required in the parameter file")
    for key in ("pad_inline_meters", "pad_crossline_meters", "filter_il_meters",
                "filter_xl_meters", "hole_filter_il_meters", "hole_filter_xl_meters",
                "alpha"):
        if key in attenuation:
            if attenuation[key] is None:
                normalized_attenuation[key] = None
                continue
            try:
                value = float(attenuation[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"attenuation.{key} must be numeric or null") from exc
            minimum = 0 if key.startswith("pad_") or key == "alpha" else 1e-12
            if not np.isfinite(value) or value < minimum or (key == "alpha" and value >= 1):
                raise ValueError(f"attenuation.{key} has an invalid value")
            normalized_attenuation[key] = value
    for key in ("preview_hardcoded_passes", "save_pass_differences"):
        if key in attenuation and not isinstance(attenuation[key], bool):
            raise ValueError(f"attenuation.{key} must be true or false")

    rms_overrides = {
        "input_file": str(source), "window_mode": mode, "vertical_domain": domain,
    }
    for key in RMS_KEYS - {
        "window_mode", "vertical_domain", "horizon_file", "existing_maps",
        "inlines_file", "crosslines_file", "trace_chunk_size",
    }:
        if key in rms:
            rms_overrides[key] = rms[key]
    rms_overrides.update(sample_settings)
    if mode == "horizon":
        rms_overrides["horizon_file"] = str(horizon_file)
        rms_overrides["horizon_windows"] = windows
    else:
        rms_overrides["constant_top"] = top
        rms_overrides["constant_bottom"] = bottom
    fp_overrides = {
        "vertical_domain": domain, "inline_increment": spacing[0],
        "crossline_increment": spacing[1], "rms_window_mode": "absolute",
        "parameter_input_mode": input_mode, "pass_parameters": normalized_passes,
    }
    # Feathering lengths are consumed by the mask stage, not attenuation.
    mask_feathering_keys = {
        "filter_il_meters", "filter_xl_meters",
        "hole_filter_il_meters", "hole_filter_xl_meters",
    }
    for key in ATTENUATION_KEYS - {
        "parameter_input_mode", "pass_parameters", "colormap_file",
    } - mask_feathering_keys:
        if key in normalized_attenuation:
            fp_overrides[key] = normalized_attenuation[key]
    if attenuation.get("colormap_file"):
        fp_overrides["colormap_file"] = str(path_from_yaml(
            attenuation["colormap_file"], base, "attenuation.colormap_file"
        ))

    saved = []
    def override(module, settings):
        for name, value in settings.items():
            saved.append((module, name, getattr(module, name)))
            setattr(module, name, value)

    try:
        if cache_directory is not None:
            cache_directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="destriping-", dir=temporary_parent
        ) as working_name:
            working = Path(working_name)
            print(f"Temporary work directory: {working}", flush=True)
            new_rms_file = working / "rms.npy"
            new_il_file = working / "inlines.npy"
            new_xl_file = working / "crosslines.npy"
            bundle_file = working / "coefficients.npz"
            if mode == "constant":
                new_maps = [new_rms_file]
            else:
                new_maps = [create_rms_map.output_for_window(
                    new_rms_file, index, lower, upper,
                    rms["horizon_z_unit"],
                ) for index, (lower, upper) in enumerate(windows, 1)]

            if existing_maps is not None:
                maps, il_file, xl_file = existing_maps, existing_il, existing_xl
            elif cache_directory is not None:
                cached_base = cache_directory / "rms.npy"
                if mode == "constant":
                    maps = [cached_base]
                else:
                    maps = [create_rms_map.output_for_window(
                        cached_base, index, lower, upper,
                        rms["horizon_z_unit"],
                    ) for index, (lower, upper) in enumerate(windows, 1)]
                il_file = cache_directory / "inlines.npy"
                xl_file = cache_directory / "crosslines.npy"
            else:
                maps, il_file, xl_file = new_maps, new_il_file, new_xl_file
            required_rms = [il_file, xl_file] + [
                path for map_file in maps
                for path in (map_file, window_sidecar(map_file))
            ]
            print("Stage 1/4: RMS maps", flush=True)
            reuse_rms = choose_saved_files("RMS maps", required_rms, on_existing)
            if reuse_rms:
                validate_saved_rms(maps, il_file, xl_file, source, domain)
                for saved_map, working_map in zip(maps, new_maps):
                    shutil.copyfile(saved_map, working_map)
                    shutil.copyfile(window_sidecar(saved_map),
                                    window_sidecar(working_map))
                new_il_file.write_bytes(il_file.read_bytes())
                new_xl_file.write_bytes(xl_file.read_bytes())
            else:
                if mode == "horizon" and not horizon_file.is_file():
                    raise FileNotFoundError(
                        f"Horizon file needed to recalculate RMS maps: {horizon_file}"
                    )
                override(create_rms_map, {
                    **rms_overrides, "output_rms_npy": str(new_rms_file),
                    "output_il_npy": str(new_il_file),
                    "output_xl_npy": str(new_xl_file),
                })
                create_rms_map.main()
                for fresh, saved_path in zip(
                    [new_il_file, new_xl_file] + [
                        path for map_file in new_maps
                        for path in (map_file, window_sidecar(map_file))
                    ],
                    required_rms,
                ):
                    publish_file(fresh, saved_path)

                # Persist scope choice if saved by create_rms_map
                work_scope_file = new_rms_file.with_name(new_rms_file.stem + "_mask_scope.txt")
                if work_scope_file.is_file() and cache_directory is not None:
                    publish_file(work_scope_file, cache_directory / (maps[0].stem + "_mask_scope.txt"))

            # -------------------------------------------------------------
            # Stage 2: Survey Mask & Blended RMS Generation
            # -------------------------------------------------------------
            print("Stage 2/4: survey mask & blended RMS generation", flush=True)
            survey_shape = (len(np.load(new_il_file, allow_pickle=False)),
                            len(np.load(new_xl_file, allow_pickle=False)))
            working_blended_maps = [m.with_name(m.stem + "_blended.npy") for m in new_maps]

            if existing_blended is not None:
                saved_blended_maps = existing_blended
            elif cache_directory is not None:
                saved_blended_maps = [cache_directory / (m.stem + "_blended.npy") for m in maps]
            else:
                saved_blended_maps = working_blended_maps

            required_blended = [
                path for b_map in saved_blended_maps
                for path in (b_map, window_sidecar(b_map), mask_sidecar(b_map))
            ]

            reuse_blended = choose_saved_files("blended RMS maps", required_blended, on_existing)
            if reuse_blended:
                for saved_b, work_b in zip(saved_blended_maps, working_blended_maps):
                    shutil.copyfile(saved_b, work_b)
                    shutil.copyfile(window_sidecar(saved_b), window_sidecar(work_b))
                    load_valid_mask(saved_b, survey_shape)
                    shutil.copyfile(mask_sidecar(saved_b), mask_sidecar(work_b))
            else:
                mask_scope = get_mask_scope(mask, maps, working, cache_directory)
                print(f"Masking strategy: {mask_scope.upper()} ({len(new_maps)} map{'s' if len(new_maps) > 1 else ''})", flush=True)

                if mask_mode == "interactive":
                    draw_mask_batch(mask, new_maps, new_il_file, new_xl_file, spacing, mask_scope, normalized_attenuation)
                else:
                    il_arr = np.load(new_il_file, allow_pickle=False)
                    xl_arr = np.load(new_xl_file, allow_pickle=False)
                    blend_configured_mask(mask, new_maps, il_arr, xl_arr, working_blended_maps, spacing, normalized_attenuation)

                # Validate and publish blended maps and duplicate window metadata
                for work_m, work_b, saved_b in zip(new_maps, working_blended_maps, saved_blended_maps):
                    if not work_b.is_file():
                        raise RuntimeError(f"Expected blended map was not generated: {work_b}")
                    if not mask_sidecar(work_b).is_file():
                        raise RuntimeError(f"Expected survey mask was not generated: {mask_sidecar(work_b)}")
                    load_valid_mask(work_b, survey_shape)
                    shutil.copyfile(window_sidecar(work_m), window_sidecar(work_b))
                    publish_file(work_b, saved_b)
                    publish_file(window_sidecar(work_b), window_sidecar(saved_b))
                    publish_file(mask_sidecar(work_b), mask_sidecar(saved_b))

            # -------------------------------------------------------------
            # Stage 3: Destriping Attenuation on Pre-Blended Maps
            # -------------------------------------------------------------
            override(footprint_attenuation, {
                **fp_overrides,
                "rms_maps": [(str(path), None) for path in working_blended_maps],
                "il_file": str(new_il_file),
                "xl_file": str(new_xl_file),
                "output_bundle_file": str(bundle_file),
            })
            print("Stage 3/4: calculating attenuation coefficients", flush=True)
            footprint_attenuation.main()

            # -------------------------------------------------------------
            # Stage 4: SEG-Y Trace Compensation
            # -------------------------------------------------------------
            print("Stage 4/4: compensating SEG-Y", flush=True)
            apply_destriping.apply_coefficients(
                source, bundle_file, target,
                sample_interval_override=sample_settings["sample_interval_override"],
                sample_start_override=sample_settings["sample_start_override"],
                overwrite_output=application.get("overwrite_output", False),
            )
        print(f"Complete: {target}; temporary files removed", flush=True)
    finally:
        for module, name, value in reversed(saved):
            setattr(module, name, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="YAML parameter file")
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    run_pipeline(config, args.config)


if __name__ == "__main__":
    main()
