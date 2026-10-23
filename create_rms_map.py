"""Create RMS maps between fixed levels or windows relative to a picked horizon."""

from pathlib import Path
import csv
import re

import numpy as np
import segyio
from numba import njit, prange
import matplotlib
import matplotlib.pyplot as plt

END_OF_TRACE = -1.0  # A lower window bound of -1 reaches the final SEG-Y sample.
TARGET_WORKING_BYTES = 64 * 1024 * 1024
MAX_TRACES_PER_CHUNK = 8192


def optimal_trace_chunk(n_samples, n_traces, *, trace_arrays, sample_bytes=4):
    """Size a chunk for trace data and its in-memory working copies."""
    if n_samples < 1 or n_traces < 1 or trace_arrays < 1 or sample_bytes < 1:
        raise ValueError("Sample, trace, and working-array counts must be positive")
    bytes_per_trace = n_samples * sample_bytes * trace_arrays + 64
    return min(n_traces, MAX_TRACES_PER_CHUNK,
               max(1, TARGET_WORKING_BYTES // bytes_per_trace))


# Runtime paths and processing settings are provided by run_pipeline.py.
input_file = None
output_rms_npy = None
output_il_npy = None
output_xl_npy = None
window_mode = None
vertical_domain = None
constant_top = None
constant_bottom = None
horizon_file = None
horizon_z_unit = None
horizon_windows = None
horizon_delimiter = "auto"
horizon_columns = None
horizon_fixed_widths = None
horizon_skip_rows = 0
horizon_missing_values = ("NaN",)
horizon_decimal = "."
horizon_encoding = "utf-8-sig"
sample_interval_override = None
sample_start_override = None
@njit(parallel=True, fastmath=True, cache=True)
def calculate_horizon_rms_numba(traces, top_idx, bot_idx):
    n_traces = traces.shape[0]
    n_samples = traces.shape[1]
    rms_out = np.empty(n_traces, dtype=np.float32)

    for i in prange(n_traces):
        t = top_idx[i]
        b = bot_idx[i]
        if t == -2147483648 or b == -2147483648:
            rms_out[i] = np.nan
            continue
        if t < 0:
            t = 0
        if b >= n_samples:
            b = n_samples - 1

        if b >= t:
            count = b - t + 1
            sq_sum = 0.0
            for k in range(t, b + 1):
                val = traces[i, k]
                sq_sum += val * val
            rms_out[i] = np.sqrt(sq_sum / count)
        else:
            rms_out[i] = np.nan

    return rms_out


@njit(parallel=True, fastmath=True, cache=True)
def calculate_multi_window_rms_numba(traces, top_indices, bottom_indices):
    """Compute every requested window while each trace is already in memory."""
    n_windows, n_traces = top_indices.shape
    n_samples = traces.shape[1]
    result = np.empty((n_windows, n_traces), dtype=np.float32)
    for i in prange(n_traces):
        for window in range(n_windows):
            top = top_indices[window, i]
            bottom = bottom_indices[window, i]
            if top == -2147483648 or bottom == -2147483648:
                result[window, i] = np.nan
                continue
            if top < 0:
                top = 0
            if bottom >= n_samples:
                bottom = n_samples - 1
            if bottom < top:
                result[window, i] = np.nan
                continue
            square_sum = 0.0
            for sample in range(top, bottom + 1):
                value = traces[i, sample]
                square_sum += value * value
            result[window, i] = np.sqrt(square_sum / (bottom - top + 1))
    return result


def split_horizon_fields(line, delimiter, widths, decimal):
    """Split one ASCII row without losing empty delimited fields."""
    if delimiter == "fixed":
        if not widths or any(not isinstance(w, int) or w <= 0 for w in widths):
            raise ValueError("horizon_fixed_widths must contain positive integers")
        boundaries = np.cumsum((0, *widths))
        if line[boundaries[-1]:].strip():
            raise ValueError("Fixed-width line has data beyond horizon_fixed_widths")
        return [line[a:b].strip() for a, b in zip(boundaries[:-1], boundaries[1:])]
    if delimiter == "whitespace":
        return line.split()
    if delimiter == "auto":
        pattern = r"\s*[,;|]\s*|\s+" if decimal == "." else r"\s*[;|]\s*|\s+"
        return [field.strip().strip('"') for field in re.split(pattern, line.strip())]
    characters = {"tab": "\t", "comma": ",", "semicolon": ";", "pipe": "|"}
    if delimiter not in characters:
        raise ValueError(f"Unsupported horizon_delimiter: {delimiter!r}")
    return [field.strip() for field in next(csv.reader([line], delimiter=characters[delimiter], skipinitialspace=True))]


def header_columns(fields):
    """Find inline, crossline, and vertical columns in a named header."""
    names = [re.sub(r"[^a-z0-9]", "", field.lower()) for field in fields]
    aliases = (
        {"il", "inl", "inline", "iline", "inlineno", "inlinenumber"},
        {"cl", "xl", "crl", "xline", "crossline", "crosslineno", "crosslinenumber"},
        {"z", "time", "twt", "depth", "tvd", "tvdss", "twtms", "twts", "depthm", "depthft"},
    )
    matches = [[i for i, name in enumerate(names) if name in options]
               for options in aliases]
    if all(len(group) == 1 for group in matches):
        return tuple(group[0] for group in matches)
    return None


def read_horizon(path, *, delimiter=None, columns=None, fixed_widths=None,
                 skip_rows=None, missing_values=None, decimal=None, encoding=None):
    """Read an ASCII horizon with configurable columns and missing picks."""
    delimiter = horizon_delimiter if delimiter is None else delimiter
    columns = horizon_columns if columns is None else columns
    if columns == "auto":
        columns = None
    fixed_widths = horizon_fixed_widths if fixed_widths is None else fixed_widths
    skip_rows = horizon_skip_rows if skip_rows is None else skip_rows
    missing_values = horizon_missing_values if missing_values is None else missing_values
    decimal = horizon_decimal if decimal is None else decimal
    encoding = horizon_encoding if encoding is None else encoding
    if decimal not in (".", ","):
        raise ValueError("horizon_decimal must be '.' or ','")
    if decimal == "," and delimiter == "comma":
        raise ValueError("Comma cannot be both the decimal and column delimiter")
    if not isinstance(skip_rows, int) or skip_rows < 0:
        raise ValueError("horizon_skip_rows must be a non-negative integer")
    if columns is not None and (len(columns) != 3 or len(set(columns)) != 3 or
                                any(not isinstance(i, int) or i < 0 for i in columns)):
        raise ValueError("horizon_columns must contain three distinct zero-based indices")

    rows = []
    mapped_columns = columns
    skipped_missing = 0
    missing = {str(value).casefold() for value in missing_values}
    with open(path, encoding=encoding) as source:
        for line_number, line in enumerate(source, 1):
            if line_number <= skip_rows:
                continue
            line = line.rstrip("\r\n")
            if not line.strip() or line.lstrip().startswith(("#", "//", "!")):
                continue
            line = line.split("#", 1)[0].split("//", 1)[0]
            if not line.strip():
                continue
            fields = split_horizon_fields(line, delimiter, fixed_widths, decimal)
            if not rows and columns is None:
                inferred = header_columns(fields)
                if inferred is not None:
                    mapped_columns = inferred
                    continue
                names = {re.sub(r"[^a-z0-9]", "", field.lower()) for field in fields}
                if {"x", "y"} <= names and not ({"il", "inl", "inline"} & names):
                    raise ValueError(
                        f"{path}:{line_number}: X/Y columns are coordinates, not "
                        "inline/crossline; export IL/CL or set horizon_columns explicitly"
                    )
            if not rows and all(not field.strip() or not any(char.isdigit() for char in field)
                                for field in fields):
                continue  # header or separator line before the first pick
            if mapped_columns is None:
                if len(fields) != 3:
                    raise ValueError(
                        f"{path}:{line_number}: found {len(fields)} columns; set "
                        "horizon_columns=(inline, crossline, z) using zero-based indices"
                    )
                mapped_columns = (0, 1, 2)
            if max(mapped_columns) >= len(fields):
                raise ValueError(f"{path}:{line_number}: fewer columns than horizon_columns requires")
            selected = [fields[i].strip() for i in mapped_columns]
            if any(not field or field.casefold() in missing for field in selected):
                skipped_missing += 1
                continue
            try:
                row = [float(field.replace(",", ".") if decimal == "," else field)
                       for field in selected]
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid numeric pick {selected}") from exc
            if not np.all(np.isfinite(row)):
                raise ValueError(f"{path}:{line_number}: horizon picks must be finite")
            rows.append(row)

    if not rows:
        raise ValueError(f"{path}: no horizon picks found")
    if skipped_missing:
        print(f"Skipped {skipped_missing} horizon rows with missing values")
    picks = np.asarray(rows, dtype=np.float64)
    points, first, inverse = np.unique(
        picks[:, :2], axis=0, return_index=True, return_inverse=True
    )
    if not np.allclose(picks[:, 2], picks[first[inverse], 2], rtol=0, atol=1e-6):
        raise ValueError(f"{path}: conflicting z values at the same inline/crossline")
    return points, picks[first, 2]


def interpolate_horizon(points, z, inlines, crosslines):
    """Interpolate to trace positions; retain NaN outside the picked area."""
    from scipy.interpolate import LinearNDInterpolator

    targets = np.column_stack((inlines, crosslines))
    if len(points) == 1:
        values = np.full(len(targets), np.nan)
        values[np.all(targets == points[0], axis=1)] = z[0]
        return values

    centered = points - points[0]
    if np.linalg.matrix_rank(centered) == 1:
        direction = centered[np.flatnonzero(np.any(centered != 0, axis=1))[0]]
        direction = direction / np.linalg.norm(direction)
        pick_distance = centered @ direction
        target_offset = targets - points[0]
        target_distance = target_offset @ direction
        perpendicular = target_offset - np.outer(target_distance, direction)
        order = np.argsort(pick_distance)
        values = np.interp(
            target_distance, pick_distance[order], z[order],
            left=np.nan, right=np.nan,
        )
        values[np.linalg.norm(perpendicular, axis=1) > 1e-6] = np.nan
        return values

    interpolator = LinearNDInterpolator(points, z, fill_value=np.nan)
    return np.asarray(interpolator(targets), dtype=np.float64)


def z_scale(unit, domain):
    scales = {"time": {"ms": 1.0, "s": 1000.0},
              "depth": {"m": 1.0, "ft": 0.3048}}
    if domain not in scales or unit not in scales[domain]:
        raise ValueError(f"Invalid vertical_domain={domain!r} and horizon_z_unit={unit!r}")
    return scales[domain][unit]


def sample_indices(top, bottom, start, interval):
    """Convert vertical coordinates to indices without casting missing picks."""
    if interval <= 0 or not np.isfinite(interval):
        raise ValueError("Sample interval must be finite and positive")
    valid = np.isfinite(top) & np.isfinite(bottom)
    if np.any(top[valid] >= bottom[valid]):
        raise ValueError("Every window must have its top above its bottom")
    missing = np.iinfo(np.int32).min
    top_idx = np.full(len(top), missing, dtype=np.int32)
    bot_idx = np.full(len(bottom), missing, dtype=np.int32)
    top_idx[valid] = np.ceil((top[valid] - start) / interval - 1e-9).astype(np.int32)
    bot_idx[valid] = np.floor((bottom[valid] - start) / interval + 1e-9).astype(np.int32)
    return top_idx, bot_idx


def output_for_window(path, index, lower, upper, unit):
    target = Path(path)
    if upper == END_OF_TRACE:
        return target.with_name(
            f"{target.stem}_horizon_{index}_{lower:g}{unit}_to_end{target.suffix}"
        )
    return target.with_name(
        f"{target.stem}_horizon_{index}_{lower:g}_{upper:g}{unit}{target.suffix}"
    )


def prompt_mask_choice(saved_maps, inlines, crosslines, base_output_path):
    """Plot all RMS maps in a single figure and prompt user for mask strategy."""
    n_maps = len(saved_maps)
    if n_maps == 0:
        return "single"

    is_interactive = matplotlib.get_backend().lower() not in {
        "agg", "pdf", "ps", "svg", "cairo", "template"
    }

    fig = None
    if is_interactive:
        ncols = min(n_maps, 3)
        nrows = int(np.ceil(n_maps / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(6.5 * ncols, 5.0 * nrows), squeeze=False
        )
        axes_flat = axes.ravel()

        il_step = np.median(np.diff(inlines)) if len(inlines) > 1 else 1.0
        xl_step = np.median(np.diff(crosslines)) if len(crosslines) > 1 else 1.0
        extent = [
            crosslines.min() - xl_step / 2,
            crosslines.max() + xl_step / 2,
            inlines.min() - il_step / 2,
            inlines.max() + il_step / 2,
        ]

        for idx, (path, grid, label) in enumerate(saved_maps):
            ax = axes_flat[idx]
            valid_vals = grid[np.isfinite(grid) & (grid > 0)]
            if valid_vals.size > 0:
                p1, p99 = np.nanpercentile(valid_vals, [1.0, 99.0])
            else:
                p1, p99 = 0.0, 1.0

            im = ax.imshow(
                grid,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="viridis",
                vmin=p1,
                vmax=p99,
                interpolation="nearest",
            )
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.set_xlabel("Crossline")
            ax.set_ylabel("Inline")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="RMS Amplitude")

        # Hide any unused subplots
        for idx in range(n_maps, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.suptitle(
            f"Generated RMS Maps ({n_maps} map{'s' if n_maps > 1 else ''})",
            fontsize=13,
            fontweight="bold",
            y=0.99,
        )
        plt.tight_layout()
        plt.ion()
        plt.show()
        plt.pause(0.2)
    else:
        print("\nNote: Non-interactive backend detected; skipping visual display.")

    print("\n" + "=" * 65)
    print(f"SURVEY MASK CONFIGURATION: {n_maps} RMS map{'s' if n_maps > 1 else ''} available")
    print("=" * 65)

    if n_maps > 1:
        prompt_msg = (
            "Do you want to make a [s]ingle mask for all RMS maps "
            "or [m]ultiple masks (one for each map)? [single/multiple] (default: single): "
        )
    else:
        prompt_msg = (
            "Do you want to proceed with a [s]ingle mask for this RMS map? [single] (default: single): "
        )

    choice = "single"
    while True:
        try:
            raw = input(prompt_msg).strip().lower()
        except EOFError:
            raw = "single"
            break
        if raw in ("", "s", "single", "all", "one", "1"):
            choice = "single"
            break
        if raw in ("m", "multiple", "each", "separate", "2"):
            choice = "multiple"
            break
        print("Please enter 'single' (or 's') or 'multiple' (or 'm').")

    print(
        f"\nSelected masking strategy: "
        f"{'SINGLE MASK (one mask for all maps)' if choice == 'single' else 'MULTIPLE MASKS (separate mask per map)'}"
    )

    # Save choice to a sidecar file so run_pipeline.py can read it directly
    choice_file = Path(base_output_path).with_name(
        Path(base_output_path).stem + "_mask_scope.txt"
    )
    choice_file.write_text(choice, encoding="utf-8")
    print(f"Mask preference saved to: {choice_file}\n")

    if fig is not None:
        plt.close(fig)
        plt.ioff()

    return choice


def main():
    if any(value is None for value in (input_file, output_rms_npy,
                                       output_il_npy, output_xl_npy)):
        raise ValueError("RMS paths are required; configure them in the pipeline YAML")
    if window_mode not in ("constant", "horizon"):
        raise ValueError("window_mode must be 'constant' or 'horizon'")
    if vertical_domain not in ("time", "depth"):
        raise ValueError("vertical_domain must be 'time' or 'depth'")
    if not Path(input_file).is_file():
        raise FileNotFoundError(f"SEG-Y file not found: {input_file}")
    outputs = [Path(path) for path in (output_rms_npy, output_il_npy, output_xl_npy)]
    if any(output.suffix != ".npy" for output in outputs):
        raise ValueError("RMS map and coordinate output filenames must end in .npy")
    if len(set(outputs)) != len(outputs):
        raise ValueError("RMS map and coordinate output paths must be different")
    for output in outputs:
        if not output.parent.is_dir():
            raise FileNotFoundError(f"Output directory does not exist: {output.parent}")
    if window_mode == "horizon":
        if not horizon_file or not Path(horizon_file).is_file():
            raise FileNotFoundError(f"Horizon file not found: {horizon_file}")
        scale = z_scale(horizon_z_unit, vertical_domain)
        if not horizon_windows:
            raise ValueError("horizon_windows must contain at least one interval")
        for lower, upper in horizon_windows:
            if (not np.isfinite(lower) or not np.isfinite(upper) or
                    (upper != END_OF_TRACE and lower >= upper)):
                raise ValueError(f"Invalid horizon window: {(lower, upper)}")
        points, horizon_z = read_horizon(horizon_file)

    with segyio.open(input_file, mode="r", strict=False, ignore_geometry=True) as f:
        f.mmap()
        n_traces = f.tracecount
        if n_traces == 0:
            raise ValueError("SEG-Y contains no traces")
        interval = (segyio.tools.dt(f) / 1000.0 if sample_interval_override is None
                    else sample_interval_override)
        if not np.isfinite(interval) or interval <= 0:
            raise ValueError("SEG-Y sample interval must be finite and positive")
        start = (f.header[0][segyio.TraceField.DelayRecordingTime]
                 if sample_start_override is None else sample_start_override)
        if not np.isfinite(start):
            raise ValueError("SEG-Y sample start must be finite")
        if sample_start_override is None:
            delays = f.attributes(segyio.TraceField.DelayRecordingTime)[:]
            if np.any(delays != start):
                raise ValueError(
                    "SEG-Y traces have different recording delays; set "
                    "sample_start_override only if a common sample start is correct"
                )
        inlines = f.attributes(segyio.TraceField.INLINE_3D)[:]
        crosslines = f.attributes(segyio.TraceField.CROSSLINE_3D)[:]
        last_sample = start + (f.samples.size - 1) * interval
        chunk_size = optimal_trace_chunk(
            f.samples.size, n_traces,
            trace_arrays=3 if int(f.format) == 6 else 2,
        )
        print(f"Found {n_traces} traces ({f.samples.size} samples each); "
              f"automatically processing {chunk_size} traces per chunk")

    unit_label = "ms" if vertical_domain == "time" else "m"

    if window_mode == "horizon":
        horizon_at_traces = interpolate_horizon(points, horizon_z, inlines, crosslines)
        print(f"Horizon coverage: {np.isfinite(horizon_at_traces).sum()}/{n_traces} traces")
        if not np.any(np.isfinite(horizon_at_traces)):
            raise ValueError("Horizon does not cover any SEG-Y trace positions")
        windows = []
        for i, (lower, upper) in enumerate(horizon_windows, 1):
            top = horizon_at_traces * scale + lower * scale
            if upper == END_OF_TRACE:
                valid = np.isfinite(top) & (top < last_sample)
                top = np.where(valid, top, np.nan)
                bottom = np.where(valid, last_sample, np.nan)
                label = f"Horizon Window {i} ({lower:g} {horizon_z_unit} - end)"
            else:
                bottom = horizon_at_traces * scale + upper * scale
                label = f"Horizon Window {i} ({lower:g} - {upper:g} {horizon_z_unit})"
            windows.append((top, bottom,
                            output_for_window(output_rms_npy, i, lower, upper, horizon_z_unit),
                            label))
    else:
        bottom = last_sample if constant_bottom == END_OF_TRACE else constant_bottom
        if constant_top >= bottom:
            raise ValueError("Constant RMS top must precede the last SEG-Y sample")
        windows = [
            (np.full(n_traces, constant_top), np.full(n_traces, bottom),
             Path(output_rms_npy),
             f"Constant Window ({constant_top:g} - "
             f"{'end' if constant_bottom == END_OF_TRACE else f'{constant_bottom:g} {unit_label}'})")
        ]

    il_unique = np.unique(inlines)
    xl_unique = np.unique(crosslines)
    grid_il = np.searchsorted(il_unique, inlines)
    grid_xl = np.searchsorted(xl_unique, crosslines)
    linear_cell = grid_il.astype(np.int64) * len(xl_unique) + grid_xl
    if len(np.unique(linear_cell)) != n_traces:
        raise ValueError(
            "SEG-Y has multiple traces at the same inline/crossline; "
            "RMS maps require one trace per grid cell"
        )

    indices = [sample_indices(top, bottom, start, interval)
               for top, bottom, _, _ in windows]
    top_indices = np.stack([pair[0] for pair in indices])
    bottom_indices = np.stack([pair[1] for pair in indices])
    rms_grids = [np.full((len(il_unique), len(xl_unique)), np.nan, dtype=np.float32)
                 for _ in windows]
    print(f"Calculating {len(windows)} RMS window(s) in one SEG-Y trace pass")
    with segyio.open(input_file, mode="r", strict=False, ignore_geometry=True) as f:
        f.mmap()
        trace_reader = f.trace.raw
        for first in range(0, n_traces, chunk_size):
            last = min(first + chunk_size, n_traces)
            traces = np.ascontiguousarray(trace_reader[first:last], dtype=np.float32)
            values = calculate_multi_window_rms_numba(
                traces, top_indices[:, first:last], bottom_indices[:, first:last]
            )
            for window_index, grid in enumerate(rms_grids):
                grid[grid_il[first:last], grid_xl[first:last]] = values[window_index]

    saved_maps = []
    for window_index, ((top, bottom, output_path, label), rms_grid) in enumerate(
        zip(windows, rms_grids)
    ):
        valid_count = np.count_nonzero(np.isfinite(rms_grid))
        if valid_count == 0:
            raise ValueError(
                f"{output_path}: no traces have samples inside this window; "
                "check horizon units, sample interval/start, and window offsets"
            )
        print(f"Valid RMS bins: {valid_count}/{n_traces} traces")
        np.save(output_path, rms_grid)
        top_grid = np.full(rms_grid.shape, np.nan, dtype=np.float32)
        bottom_grid = np.full(rms_grid.shape, np.nan, dtype=np.float32)
        top_grid[grid_il, grid_xl] = top
        bottom_grid[grid_il, grid_xl] = bottom
        window_file = output_path.with_name(output_path.stem + "_window.npz")
        np.savez_compressed(
            window_file, top_z=top_grid, bottom_z=bottom_grid,
            inlines=il_unique, crosslines=xl_unique,
            vertical_domain=vertical_domain,
            z_unit="ms" if vertical_domain == "time" else "m",
            bottom_to_end=(constant_bottom == END_OF_TRACE if window_mode == "constant"
                           else horizon_windows[window_index][1] == END_OF_TRACE),
            sample_interval=interval, sample_start=start,
        )
        print(f"RMS map saved: {output_path} (Shape: {rms_grid.shape})")
        print(f"Window bounds saved: {window_file}")
        saved_maps.append((output_path, rms_grid, label))

    np.save(output_il_npy, il_unique)
    np.save(output_xl_npy, xl_unique)
    print(f"Coordinate axes saved to {output_il_npy} and {output_xl_npy}")

    # Display all calculated RMS maps in a single figure and prompt user for mask strategy
    choice = prompt_mask_choice(saved_maps, il_unique, xl_unique, output_rms_npy)
    return choice


if __name__ == "__main__":
    raise SystemExit("Run python run_pipeline.py <parameter-file.yml> to configure RMS maps")
