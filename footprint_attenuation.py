import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy.ndimage import uniform_filter
from numpy.lib.stride_tricks import sliding_window_view
import sys

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
def alpha_trimmed_mean_1d(arr, axis, win_size, alpha=0.2, chunk_size=500):
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
    out = np.empty_like(data)

    padded = np.pad(data, ((0, 0), (pad_w, pad_w)), mode="edge")

    for r_start in range(0, nrows, chunk_size):
        r_end = min(r_start + chunk_size, nrows)
        chunk = padded[r_start:r_end, :]
        windows = sliding_window_view(chunk, win_size, axis=1)

        if n_trim > 0:
            sorted_win = np.sort(windows, axis=-1)
            out[r_start:r_end, :] = np.mean(sorted_win[..., n_trim : win_size - n_trim], axis=-1)
        else:
            out[r_start:r_end, :] = np.mean(windows, axis=-1)

    return np.ascontiguousarray(out.T) if axis == 0 else out


def run_alpha_trim_pipeline(data, pipeline_steps, inc_il, inc_xl):
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
        result = alpha_trimmed_mean_1d(result, axis=axis, win_size=win_samples, alpha=alpha)
    return result


# -------------------------------------------------------------
# 2. File Paths & Pipeline Parameters
# -------------------------------------------------------------
rms_file = "/data/h8/dug/brazil/teamipanema/total/brazilMPFW_100/seiTimeProc/test/QC/9999_SEGY_OUT_2/100sei/9999_SEGY_OUT_2/KDM_NorthPolygon_RMS.npy"
il_file = "/data/h8/dug/brazil/teamipanema/total/brazilMPFW_100/seiTimeProc/test/QC/9999_SEGY_OUT_2/100sei/9999_SEGY_OUT_2/KDM_NorthPolygon_Inlines.npy"
xl_file = "/data/h8/dug/brazil/teamipanema/total/brazilMPFW_100/seiTimeProc/test/QC/9999_SEGY_OUT_2/100sei/9999_SEGY_OUT_2/KDM_NorthPolygon_Crosslines.npy"
polygon_file = "/data/h8/dug/brazil/teamipanema/total/brazilMPFW_100/seiTimeProc/test/QC/9999_SEGY_OUT_2/100sei/9999_SEGY_OUT_2/selected_polygon.npy"

output_diff_4k_npy = "/data/h8/dug/brazil/teamipanema/total/brazilMPFW_100/seiTimeProc/test/QC/9999_SEGY_OUT_2/100sei/9999_SEGY_OUT_2/KDM_NorthPolygon_RMS_Diff_4k.npy"
output_diff_16k_npy = "/data/h8/dug/brazil/teamipanema/total/brazilMPFW_100/seiTimeProc/test/QC/9999_SEGY_OUT_2/100sei/9999_SEGY_OUT_2/KDM_NorthPolygon_RMS_Diff_16k.npy"

colormap_file = "/data/h8/dug/brazil/teamipanema/total/brazilMPFW_100/ScientificColourMaps8/oslo/oslo.alut"

inline_increment = 25.0
crossline_increment = 25.0

pad_inline_meters = 3000.0
pad_crossline_meters = 200.0

# Mask smoothing
filter_il_meters = 1000.0
filter_xl_meters = 1000.0

# -------------------------------------------------------------
# Easily Change Pipeline Steps Here: ('direction', window_meters, alpha)
# -------------------------------------------------------------
# Pass 1: Extract Inline structural trend
pipeline_inline = [
    ("inline", 1000.0, 0.3),
    ("inline", 1000.0, 0.3),
]

# Pass 2A: Crossline footprint filter (4,000 m aperture)
pipeline_xl_4k = [
    ("crossline", 4000.0, 0.3),
    ("crossline", 4000.0, 0.3),
]

# Pass 2B: Crossline footprint filter (16,000 m aperture)
pipeline_xl_16k = [
    ("crossline", 16000.0, 0.3),
    ("crossline", 16000.0, 0.3),
]

# -------------------------------------------------------------
# 3. Load Arrays & Build Coordinates
# -------------------------------------------------------------
n_pad_il = int(np.round(pad_inline_meters / inline_increment))
n_pad_xl = int(np.round(pad_crossline_meters / crossline_increment))

rms_map = np.load(rms_file)
il_unique = np.load(il_file)
xl_unique = np.load(xl_file)
polygon = np.load(polygon_file)

p1, p99 = np.nanpercentile(rms_map, [1.0, 99.0])
clipped_map = np.clip(rms_map, p1, p99)
clipped_map = np.nan_to_num(clipped_map, nan=0.0)

padded_rms = np.pad(clipped_map, pad_width=((n_pad_il, n_pad_il), (n_pad_xl, n_pad_xl)),mode="constant",constant_values=0.0)

il_step = np.median(np.diff(il_unique)) if len(il_unique) > 1 else 1.0
xl_step = np.median(np.diff(xl_unique)) if len(xl_unique) > 1 else 1.0

pad_il_before = il_unique[0] - np.arange(n_pad_il, 0, -1) * il_step
pad_il_after = il_unique[-1] + np.arange(1, n_pad_il + 1) * il_step
padded_il = np.concatenate([pad_il_before, il_unique, pad_il_after])

pad_xl_before = xl_unique[0] - np.arange(n_pad_xl, 0, -1) * xl_step
pad_xl_after = xl_unique[-1] + np.arange(1, n_pad_xl + 1) * xl_step
padded_xl = np.concatenate([pad_xl_before, xl_unique, pad_xl_after])

extent = [padded_xl.min(), padded_xl.max(), padded_il.min(), padded_il.max()]

# -------------------------------------------------------------
# 4. Mask Generation & Blending
# -------------------------------------------------------------
XL, IL = np.meshgrid(padded_xl, padded_il)
points = np.column_stack((XL.ravel(), IL.ravel()))

poly_path = Path(polygon)
mask_grid = poly_path.contains_points(points).reshape(padded_rms.shape).astype(np.float32)

win_il = int(np.round(filter_il_meters / inline_increment)) | 1
win_xl = int(np.round(filter_xl_meters / crossline_increment)) | 1

smoothed_mask = uniform_filter(mask_grid, size=(win_il, win_xl), mode="nearest")
smoothed_mask = np.clip(smoothed_mask, 0.0, 1.0)
smoothed_mask[smoothed_mask < 1.0] = 0.0
smoothed_mask = uniform_filter(smoothed_mask, size=(win_il, win_xl), mode="nearest")

inside_values = padded_rms[(mask_grid == 1.0) & (padded_rms > 0.0)]
if inside_values.size == 0:
    inside_values = padded_rms[padded_rms > 0.0]
median_val = float(np.median(inside_values))

blended_rms = padded_rms * smoothed_mask + (1.0 - smoothed_mask) * median_val

# -------------------------------------------------------------
# 5. Filtering Execution
# -------------------------------------------------------------
# Pass 1: Inline filtering
print("\n--- Running Inline Filter Pipeline ---")
inline_filtered = run_alpha_trim_pipeline(blended_rms, pipeline_inline, inline_increment, crossline_increment)
difference = inline_filtered - blended_rms

# Pass 2A: 4,000m Crossline filtering on difference
print("\n--- Running Crossline Filter Pipeline (4,000 m) ---")
diff_xl_4k = run_alpha_trim_pipeline(difference, pipeline_xl_4k, inline_increment, crossline_increment)
std_4k = np.std(diff_xl_4k)
mask_diff_4k = np.where(np.abs(difference) > 2.0 * std_4k, 1.0, 0.0)
restored_4k = blended_rms + diff_xl_4k
np.save(output_diff_4k_npy, diff_xl_4k)

# Pass 2B: 16,000m Crossline filtering on difference
print("\n--- Running Crossline Filter Pipeline (16,000 m) ---")
diff_xl_16k = run_alpha_trim_pipeline(difference, pipeline_xl_16k, inline_increment, crossline_increment)
std_16k = np.std(diff_xl_16k)
mask_diff_16k = np.where(np.abs(difference) > 2.0 * std_16k, 1.0, 0.0)
restored_16k = blended_rms + diff_xl_16k
np.save(output_diff_16k_npy, diff_xl_16k)

print("\n--- Running Crossline Filter Pipeline (4,000 m) ---")
diff_prec = mask_diff_4k*diff_xl_16k + (1-mask_diff_4k) * difference
diff_prec = run_alpha_trim_pipeline(diff_prec, pipeline_xl_4k, inline_increment, crossline_increment)

eps=sys.float_info.min
result = blended_rms+diff_prec

coeff = (result+eps)/(blended_rms+eps)

coeff_filt = run_alpha_trim_pipeline(coeff, pipeline_xl_4k, inline_increment, crossline_increment)

# -------------------------------------------------------------
# 6. Reusable QC Plotting Function
# -------------------------------------------------------------
my_cmap = load_seismic_colormap(colormap_file, smooth=True)
blend_vmax = np.percentile(blended_rms, 99.9)
diff_vmax = np.percentile(np.abs(difference), 97.0)


fig, axes = plt.subplots(2, 4, figsize=(22, 12), sharex=True, sharey=True)
axes = axes.ravel()

# 0. Blended RMS
im0 = axes[0].imshow(blended_rms, origin="lower", aspect="equal", extent=extent, cmap=my_cmap, vmin=0, vmax=blend_vmax)
axes[0].set_title(f"Blended RMS (Median: {median_val:.2f})", fontsize=10)
axes[0].set_ylabel("Inline")
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

# 1. Raw Inline Difference
im1 = axes[1].imshow(difference, origin="lower", aspect="equal", extent=extent, cmap="bwr", vmin=-diff_vmax, vmax=diff_vmax)
axes[1].set_title("Difference: Inline Filtered - Blended", fontsize=10)
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

# 2. Crossline Filtered Difference
im2 = axes[2].imshow(diff_xl_4k, origin="lower", aspect="equal", extent=extent, cmap="bwr", vmin=-diff_vmax, vmax=diff_vmax)
axes[2].set_title(f"Filtered Difference ", fontsize=10)
fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

# 5. Residual Noise / Footprint Removed
im5 = axes[3].imshow(diff_prec, origin="lower", aspect="equal", extent=extent, cmap="bwr", vmin=-diff_vmax, vmax=diff_vmax)
axes[3].set_title("Residual Footprint Removed (Diff - Filtered)", fontsize=10)
axes[3].set_xlabel("Crossline")
fig.colorbar(im5, ax=axes[3], fraction=0.046, pad=0.04)

im6 = axes[4].imshow(result, origin="lower", aspect="equal", extent=extent, cmap=my_cmap, vmin=0, vmax=blend_vmax)
axes[4].set_title(f"Blended RMS (Median: {median_val:.2f})", fontsize=10)
axes[4].set_ylabel("Inline")
fig.colorbar(im6, ax=axes[4], fraction=0.046, pad=0.04)

im7 = axes[5].imshow(coeff, origin="lower", aspect="equal", extent=extent, cmap='bwr',vmin=0.5,vmax=1.5)
axes[5].set_title(f"Blended RMS (Median: {median_val:.2f})", fontsize=10)
fig.colorbar(im7, ax=axes[5], fraction=0.046, pad=0.04)

im8 = axes[6].imshow(coeff_filt, origin="lower", aspect="equal", extent=extent, cmap='bwr',vmin=0.5,vmax=1.5)
axes[6].set_title(f"Blended RMS (Median: {median_val:.2f})", fontsize=10)
fig.colorbar(im8, ax=axes[6], fraction=0.046, pad=0.04)

im9 = axes[7].imshow(blended_rms*coeff_filt, origin="lower", aspect="equal", extent=extent, cmap=my_cmap, vmin=0, vmax=blend_vmax)
axes[7].set_title(f"", fontsize=10)
axes[7].set_ylabel("Inline")
fig.colorbar(im0, ax=axes[7], fraction=0.046, pad=0.04)

fig.suptitle(f"QC Dashboard", fontsize=14, weight="bold")
plt.tight_layout()

plt.show()