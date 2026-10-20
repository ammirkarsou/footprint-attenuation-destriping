import segyio
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

input_file = "/data/h8/dug/brazil/teamipanema/total/brazilMPFW_100/seiTimeProc/test/QC/9999_SEGY_OUT_2/100sei/9999_SEGY_OUT_2/KDM_NorthPolygon_Test.sgy"
output_rms_npy = "/data/h8/dug/brazil/teamipanema/total/brazilMPFW_100/seiTimeProc/test/QC/9999_SEGY_OUT_2/100sei/9999_SEGY_OUT_2/KDM_NorthPolygon_RMS.npy"
output_il_npy = "/data/h8/dug/brazil/teamipanema/total/brazilMPFW_100/seiTimeProc/test/QC/9999_SEGY_OUT_2/100sei/9999_SEGY_OUT_2/KDM_NorthPolygon_Inlines.npy"
output_xl_npy = "/data/h8/dug/brazil/teamipanema/total/brazilMPFW_100/seiTimeProc/test/QC/9999_SEGY_OUT_2/100sei/9999_SEGY_OUT_2/KDM_NorthPolygon_Crosslines.npy"

# -------------------------------------------------------------
# 1. High-Performance Multi-threaded RMS Kernel
# -------------------------------------------------------------
@njit(parallel=True, fastmath=True)
def calculate_horizon_rms_numba(traces, top_idx, bot_idx):
    n_traces = traces.shape[0]
    rms_out = np.empty(n_traces, dtype=np.float32)

    for i in prange(n_traces):
        t = top_idx[i]
        b = bot_idx[i]

        if t < 0:
            t = 0
        if b >= traces.shape[1]:
            b = traces.shape[1] - 1

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


# -------------------------------------------------------------
# 2. Read Data & Headers
# -------------------------------------------------------------
with segyio.open(input_file, mode="r", strict=False, ignore_geometry=True) as f:
    f.mmap()
    dt_us = segyio.tools.dt(f)
    dt_ms = dt_us / 1000.0
    t0_ms = f.header[0][segyio.TraceField.DelayRecordingTime]
    n_samples = f.samples.size
    n_traces = f.tracecount

    print(f"Loading {n_traces} traces ({n_samples} samples each)...")
    traces = np.ascontiguousarray(f.trace.raw[:], dtype=np.float32)

    inlines = f.attributes(segyio.TraceField.INLINE_3D)[:]
    crosslines = f.attributes(segyio.TraceField.CROSSLINE_3D)[:]


# -------------------------------------------------------------
# 3. Define Surfaces & Compute Sample Indices
# -------------------------------------------------------------
top_time_ms = np.full(n_traces, 0.0)
bot_time_ms = np.full(n_traces, 1000.0)

top_idx = np.round((top_time_ms - t0_ms) / dt_ms).astype(np.int32)
bot_idx = np.round((bot_time_ms - t0_ms) / dt_ms).astype(np.int32)


# -------------------------------------------------------------
# 4. Compute RMS Map
# -------------------------------------------------------------
print("Calculating RMS amplitude...")
rms_values = calculate_horizon_rms_numba(traces, top_idx, bot_idx)

# Free the raw 7 GB buffer immediately
del traces


# -------------------------------------------------------------
# 5. Grid into 2D Array & Save to .npy
# -------------------------------------------------------------
il_unique = np.unique(inlines)
xl_unique = np.unique(crosslines)

# Vectorized indexing using searchsorted (significantly faster than dict lookups)
grid_il = np.searchsorted(il_unique, inlines)
grid_xl = np.searchsorted(xl_unique, crosslines)

rms_grid = np.full((len(il_unique), len(xl_unique)), np.nan, dtype=np.float32)
rms_grid[grid_il, grid_xl] = rms_values

# Save the grid and its coordinate axes to disk
np.save(output_rms_npy, rms_grid)
np.save(output_il_npy, il_unique)
np.save(output_xl_npy, xl_unique)

print(f"RMS map saved: {output_rms_npy} (Shape: {rms_grid.shape})")
print(f"Coordinate axes saved to {output_il_npy} and {output_xl_npy}")