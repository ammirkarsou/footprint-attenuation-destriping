"""Create RMS maps between fixed levels or windows relative to a picked horizon."""

from pathlib import Path
import csv
import re

import numpy as np
import segyio
from numba import njit, prange


input_file = "/data/h8/dug/brazil/teamipanema/total/brazilMPFW_100/seiTimeProc/test/QC/9999_SEGY_OUT_2/100sei/9999_SEGY_OUT_2/KDM_NorthPolygon_Test.sgy"
output_rms_npy = "/data/h8/dug/brazil/teamipanema/total/brazilMPFW_100/seiTimeProc/test/QC/9999_SEGY_OUT_2/100sei/9999_SEGY_OUT_2/KDM_NorthPolygon_RMS.npy"
output_il_npy = "/data/h8/dug/brazil/teamipanema/total/brazilMPFW_100/seiTimeProc/test/QC/9999_SEGY_OUT_2/100sei/9999_SEGY_OUT_2/KDM_NorthPolygon_Inlines.npy"
output_xl_npy = "/data/h8/dug/brazil/teamipanema/total/brazilMPFW_100/seiTimeProc/test/QC/9999_SEGY_OUT_2/100sei/9999_SEGY_OUT_2/KDM_NorthPolygon_Crosslines.npy"

# Choose "constant" for the original 0-1000 ms window, or "horizon".
window_mode = "constant"
vertical_domain = "time"  # "time" (SEG-Y axis in ms) or "depth" (SEG-Y axis in m)
constant_top = 0.0
constant_bottom = 1000.0

# Plain-text horizon. Picks may be on any inline/crossline spacing.
horizon_file = "/path/to/horizon.txt"
horizon_z_unit = "ms"  # "ms" or "s" for time; "m" or "ft" for depth
horizon_windows = [(0.0, 1400.0), (1400.0, 2800.0)]
# Window offsets use horizon_z_unit. Each window gets a separate RMS .npy file.
# No values are inferred beyond the picked horizon's coverage.

# ASCII layout. "auto" accepts mixed spaces, tabs, commas, semicolons, and |.
# Other choices: "whitespace", "tab", "comma", "semicolon", "pipe", "fixed".
horizon_delimiter = "auto"
# None maps a named header (e.g. IL XL X Y TWT), or uses the first three
# columns when there are exactly three. Otherwise use zero-based column
# positions, e.g. (0, 1, 4) for IL XL X Y Z.
horizon_columns = None
horizon_fixed_widths = None  # e.g. (8, 8, 12) with horizon_delimiter="fixed"
horizon_skip_rows = 0  # physical lines to skip before reading any data
horizon_missing_values = ("NaN",)  # add exporter-specific values, e.g. "-999.25"
horizon_decimal = "."  # set to "," with semicolon/tab/whitespace delimiters
horizon_encoding = "utf-8-sig"  # use "cp1252" for some older Windows exports

# Use SEG-Y sample interval and delay by default. Override in ms (time) or m
# (depth) when the file headers do not describe the vertical axis correctly.
sample_interval_override = None
sample_start_override = None


@njit(parallel=True, fastmath=True)
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

    # A horizon picked along a single line (including a diagonal) is 1-D.
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
    top_idx[valid] = np.rint((top[valid] - start) / interval).astype(np.int32)
    bot_idx[valid] = np.rint((bottom[valid] - start) / interval).astype(np.int32)
    return top_idx, bot_idx


def output_for_window(path, index, lower, upper, unit):
    target = Path(path)
    return target.with_name(
        f"{target.stem}_horizon_{index}_{lower:g}_{upper:g}{unit}{target.suffix}"
    )


def main():
    if window_mode not in ("constant", "horizon"):
        raise ValueError("window_mode must be 'constant' or 'horizon'")
    if vertical_domain not in ("time", "depth"):
        raise ValueError("vertical_domain must be 'time' or 'depth'")
    if window_mode == "horizon":
        scale = z_scale(horizon_z_unit, vertical_domain)
        if not horizon_windows:
            raise ValueError("horizon_windows must contain at least one interval")
        for lower, upper in horizon_windows:
            if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
                raise ValueError(f"Invalid horizon window: {(lower, upper)}")
        points, horizon_z = read_horizon(horizon_file)

    with segyio.open(input_file, mode="r", strict=False, ignore_geometry=True) as f:
        f.mmap()
        # SEG-Y stores time intervals in microseconds and depth intervals in
        # millimetres, giving ms or m after division by 1000.
        interval = (segyio.tools.dt(f) / 1000.0 if sample_interval_override is None
                    else sample_interval_override)
        start = (f.header[0][segyio.TraceField.DelayRecordingTime]
                 if sample_start_override is None else sample_start_override)
        n_traces = f.tracecount
        print(f"Loading {n_traces} traces ({f.samples.size} samples each)...")
        traces = np.ascontiguousarray(f.trace.raw[:], dtype=np.float32)
        inlines = f.attributes(segyio.TraceField.INLINE_3D)[:]
        crosslines = f.attributes(segyio.TraceField.CROSSLINE_3D)[:]

    if window_mode == "horizon":
        horizon_at_traces = interpolate_horizon(points, horizon_z, inlines, crosslines)
        print(f"Horizon coverage: {np.isfinite(horizon_at_traces).sum()}/{n_traces} traces")
        if not np.any(np.isfinite(horizon_at_traces)):
            raise ValueError("Horizon does not cover any SEG-Y trace positions")
        windows = (
            (horizon_at_traces * scale + lower * scale,
             horizon_at_traces * scale + upper * scale,
             output_for_window(output_rms_npy, i, lower, upper, horizon_z_unit))
            for i, (lower, upper) in enumerate(horizon_windows, 1)
        )
    else:
        windows = [
            (np.full(n_traces, constant_top), np.full(n_traces, constant_bottom),
             Path(output_rms_npy))
        ]

    il_unique = np.unique(inlines)
    xl_unique = np.unique(crosslines)
    grid_il = np.searchsorted(il_unique, inlines)
    grid_xl = np.searchsorted(xl_unique, crosslines)

    for top, bottom, output_path in windows:
        top_idx, bot_idx = sample_indices(top, bottom, start, interval)
        print(f"Calculating RMS amplitude: {output_path}")
        rms_values = calculate_horizon_rms_numba(traces, top_idx, bot_idx)
        rms_grid = np.full((len(il_unique), len(xl_unique)), np.nan, dtype=np.float32)
        rms_grid[grid_il, grid_xl] = rms_values
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
        )
        print(f"RMS map saved: {output_path} (Shape: {rms_grid.shape})")
        print(f"Window bounds saved: {window_file}")

    np.save(output_il_npy, il_unique)
    np.save(output_xl_npy, xl_unique)
    print(f"Coordinate axes saved to {output_il_npy} and {output_xl_npy}")


if __name__ == "__main__":
    main()
