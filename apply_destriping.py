"""Multiply the source SEG-Y by the vertically interpolated RMS-map coefficients."""

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import segyio

from coefficient_interpolation import apply_coefficients_chunk
from create_rms_map import optimal_trace_chunk


def load_coefficient_bundle(path):
    required = {
        "coefficients", "top_z", "bottom_z", "inlines", "crosslines",
        "vertical_domain", "z_unit", "interpolation",
    }
    with np.load(path, allow_pickle=False) as saved:
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(f"Coefficient bundle is missing: {', '.join(sorted(missing))}")
        bundle = {key: saved[key] for key in required}
        if "valid_masks" in saved.files:
            bundle["valid_masks"] = saved["valid_masks"]
        has_interval = "sample_interval" in saved.files
        has_start = "sample_start" in saved.files
        if has_interval != has_start:
            raise ValueError("Coefficient bundle has incomplete sample-axis metadata")
        if has_interval:
            bundle["sample_interval"] = float(saved["sample_interval"])
            bundle["sample_start"] = float(saved["sample_start"])

    coefficients = bundle["coefficients"]
    top = bundle["top_z"]
    bottom = bundle["bottom_z"]
    inlines = bundle["inlines"]
    crosslines = bundle["crosslines"]
    if coefficients.ndim != 3 or coefficients.shape[0] == 0:
        raise ValueError("Coefficient bundle must contain at least one 2D map")
    if top.shape != coefficients.shape or bottom.shape != coefficients.shape:
        raise ValueError("Coefficient and vertical-window arrays have different shapes")
    if "valid_masks" in bundle and (
        bundle["valid_masks"].shape != coefficients.shape or
        bundle["valid_masks"].dtype != np.bool_
    ):
        raise ValueError("Coefficient validity masks must be boolean and match the maps")
    if inlines.ndim != 1 or crosslines.ndim != 1 or (
        coefficients.shape[1:] != (len(inlines), len(crosslines))
    ):
        raise ValueError("Coefficient grid does not match the inline/crossline axes")
    if (not np.all(np.diff(inlines) > 0) or
            not np.all(np.diff(crosslines) > 0)):
        raise ValueError("Inline and crossline axes must be strictly increasing")
    if np.any(np.isinf(coefficients)) or np.any(np.isinf(top)) or np.any(np.isinf(bottom)):
        raise ValueError("Coefficient bundle contains infinite values")
    domain = str(bundle["vertical_domain"].item())
    unit = str(bundle["z_unit"].item())
    if (domain, unit) not in (("time", "ms"), ("depth", "m")):
        raise ValueError("Bundle vertical domain/unit must be time/ms or depth/m")
    if str(bundle["interpolation"].item()) != (
        "linear_at_window_centers_taper_to_one_at_outer_bounds"
    ):
        raise ValueError("Unsupported coefficient interpolation method")
    if "sample_interval" in bundle and (
        not np.isfinite(bundle["sample_interval"]) or bundle["sample_interval"] <= 0 or
        not np.isfinite(bundle["sample_start"])
    ):
        raise ValueError("Bundle sample interval/start are invalid")

    valid = np.isfinite(coefficients) & np.isfinite(top) & np.isfinite(bottom)
    if np.any(bottom[valid] <= top[valid]):
        raise ValueError("A coefficient window has its bottom at or above its top")
    previous_center = np.full(coefficients.shape[1:], -np.inf)
    for index in range(coefficients.shape[0]):
        center = (top[index] + bottom[index]) / 2
        if np.any(valid[index] & (center <= previous_center)):
            raise ValueError("Coefficient window centers must increase at every bin")
        previous_center = np.where(valid[index], center, previous_center)
    return bundle


def inspect_input(segy_path, bundle, interval_override, start_override):
    with segyio.open(str(segy_path), mode="r", strict=False, ignore_geometry=True) as segy:
        count = segy.tracecount
        if count == 0:
            raise ValueError("Input SEG-Y contains no traces")
        if int(segy.format) not in (1, 5, 6):
            raise ValueError(
                f"SEG-Y sample format {segy.format} is not floating point; "
                "compensation would quantize or overflow integer samples"
            )
        interval = (segyio.tools.dt(segy) / 1000.0 if interval_override is None
                    else interval_override)
        start = (segy.header[0][segyio.TraceField.DelayRecordingTime]
                 if start_override is None else start_override)
        if not np.isfinite(interval) or interval <= 0:
            raise ValueError("SEG-Y sample interval must be finite and positive")
        if not np.isfinite(start):
            raise ValueError("SEG-Y sample start must be finite")
        if "sample_interval" in bundle and (
            not np.isclose(interval, bundle["sample_interval"], rtol=0, atol=1e-6) or
            not np.isclose(start, bundle["sample_start"], rtol=0, atol=1e-6)
        ):
            raise ValueError(
                "SEG-Y sample interval/start differ from the RMS maps; "
                "check the input file and sample-axis overrides"
            )
        if start_override is None:
            delays = segy.attributes(segyio.TraceField.DelayRecordingTime)[:]
            if np.any(delays != start):
                raise ValueError(
                    "SEG-Y traces have different recording delays; use the same "
                    "sample_start_override as when creating the RMS maps"
                )
        il_headers = segy.attributes(segyio.TraceField.INLINE_3D)[:]
        xl_headers = segy.attributes(segyio.TraceField.CROSSLINE_3D)[:]
        n_samples = segy.samples.size
        sample_bytes = 8 if int(segy.format) == 6 else 4

    inlines = bundle["inlines"]
    crosslines = bundle["crosslines"]
    if (not np.array_equal(np.unique(il_headers), inlines) or
            not np.array_equal(np.unique(xl_headers), crosslines)):
        raise ValueError(
            "SEG-Y inline/crossline axes differ from the coefficient bundle; "
            "select the SEG-Y used to make the RMS maps"
        )
    il_index = np.searchsorted(inlines, il_headers)
    xl_index = np.searchsorted(crosslines, xl_headers)
    cells = il_index.astype(np.int64) * len(crosslines) + xl_index
    if len(np.unique(cells)) != count:
        raise ValueError("SEG-Y contains duplicate inline/crossline trace positions")
    samples = start + np.arange(n_samples, dtype=np.float64) * interval
    return il_index, xl_index, samples, sample_bytes


def apply_coefficients(input_path, bundle_path, output_path, *,
                       sample_interval_override=None, sample_start_override=None,
                       trace_chunk_size=None, overwrite_output=False):
    """Copy and compensate SEG-Y traces, preserving headers and trace order."""
    source = Path(input_path)
    bundle_file = Path(bundle_path)
    target = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"Input SEG-Y not found: {source}")
    if not bundle_file.is_file():
        raise FileNotFoundError(f"Coefficient bundle not found: {bundle_file}")
    if not target.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {target.parent}")
    if source.resolve() == target.resolve():
        raise ValueError("Output SEG-Y must be different from the input SEG-Y")
    if target.exists() and not overwrite_output:
        raise FileExistsError(
            f"Output already exists: {target}; change the name or set overwrite_output=True"
        )
    if trace_chunk_size is not None:
        print("Ignoring trace_chunk_size; chunk size is automatic")
    data = load_coefficient_bundle(bundle_file)
    il_index, xl_index, z_samples, sample_bytes = inspect_input(
        source, data, sample_interval_override, sample_start_override
    )
    chunk_size = optimal_trace_chunk(
        len(z_samples), len(il_index), trace_arrays=3, sample_bytes=sample_bytes
    )
    print(f"Input: {len(il_index)} traces, {len(z_samples)} samples per trace")
    print(f"Automatically processing {chunk_size} traces per chunk")
    print(f"Bundle: {data['coefficients'].shape[0]} RMS windows, "
          f"{data['vertical_domain'].item()} in {data['z_unit'].item()}")
    print(f"Writing compensated SEG-Y to {target}")

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=f".{target.name}.", suffix=".partial",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        shutil.copyfile(source, temporary)

        changed = 0
        with segyio.open(str(temporary), mode="r+", strict=False,
                         ignore_geometry=True) as segy:
            trace_io = segy.trace.raw
            for first in range(0, len(il_index), chunk_size):
                last = min(first + chunk_size, len(il_index))
                traces = np.ascontiguousarray(trace_io[first:last])
                corrected, changed_flags, overflow = apply_coefficients_chunk(
                    traces, il_index[first:last], xl_index[first:last], z_samples,
                    data["top_z"], data["bottom_z"], data["coefficients"],
                    data.get("valid_masks"),
                )
                bad = np.flatnonzero(overflow)
                if bad.size:
                    raise ValueError(f"Compensated trace {first + bad[0]} contains overflow")
                for offset in np.flatnonzero(changed_flags):
                    trace_io[first + int(offset)] = corrected[offset]
                changed += int(np.count_nonzero(changed_flags))
                if last == len(il_index) or (last // chunk_size) % 20 == 0:
                    print(f"Processed {last}/{len(il_index)} traces")
            segy.flush()

        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Done: {changed} traces compensated; {len(il_index) - changed} unchanged")
    return target


if __name__ == "__main__":
    raise SystemExit("Run python run_pipeline.py <parameter-file.yml> to configure compensation")
