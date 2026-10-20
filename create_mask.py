import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector, Button

# -------------------------------------------------------------
# 1. File Paths
# -------------------------------------------------------------
rms_file = "/data/h8/dug/brazil/teamipanema/total/brazilMPFW_100/seiTimeProc/test/QC/9999_SEGY_OUT_2/100sei/9999_SEGY_OUT_2/KDM_NorthPolygon_RMS.npy"
il_file = "/data/h8/dug/brazil/teamipanema/total/brazilMPFW_100/seiTimeProc/test/QC/9999_SEGY_OUT_2/100sei/9999_SEGY_OUT_2/KDM_NorthPolygon_Inlines.npy"
xl_file = "/data/h8/dug/brazil/teamipanema/total/brazilMPFW_100/seiTimeProc/test/QC/9999_SEGY_OUT_2/100sei/9999_SEGY_OUT_2/KDM_NorthPolygon_Crosslines.npy"

output_polygon_file = "/data/h8/dug/brazil/teamipanema/total/brazilMPFW_100/seiTimeProc/test/QC/9999_SEGY_OUT_2/100sei/9999_SEGY_OUT_2/selected_polygon.npy"

# -------------------------------------------------------------
# 2. Geometry Spacing & Axis Padding Parameters
# -------------------------------------------------------------
# Bin spacing in meters per unit line
inline_increment = 25.0       # meters per 1 inline
crossline_increment = 25.0    # meters per 1 crossline

# Option A: Set padding in meters (set to None if defining by line count)
pad_inline_meters = 3000.0     # extra margin in meters along inline direction
pad_crossline_meters = 1000.0 # extra margin in meters along crossline direction

# Option B: Set padding directly in number of lines (used if meters are set to None)
pad_inline_lines = None       # e.g., 20
pad_crossline_lines = None    # e.g., 40

# -------------------------------------------------------------
# 3. Calculate Padding (Integer Line Counts)
# -------------------------------------------------------------
if pad_inline_meters is not None:
    n_pad_il = int(np.round(pad_inline_meters / inline_increment))
else:
    n_pad_il = int(pad_inline_lines) if pad_inline_lines is not None else 0

if pad_crossline_meters is not None:
    n_pad_xl = int(np.round(pad_crossline_meters / crossline_increment))
else:
    n_pad_xl = int(pad_crossline_lines) if pad_crossline_lines is not None else 0

print(f"Inline padding   : {n_pad_il} lines ({n_pad_il * inline_increment:.1f} m)")
print(f"Crossline padding: {n_pad_xl} lines ({n_pad_xl * crossline_increment:.1f} m)")

# -------------------------------------------------------------
# 4. Load Data & Pad Array with 0
# -------------------------------------------------------------
rms_map = np.load(rms_file)
il_unique = np.load(il_file)
xl_unique = np.load(xl_file)

# Dynamic range scaling (P1 to P99)
p1, p99 = np.nanpercentile(rms_map, [1.0, 99.0])
clipped_map = np.clip(rms_map, p1, p99)

# Replace existing NaNs inside the survey area with 0.0
clipped_map = np.nan_to_num(clipped_map, nan=0.0)

# Pad the 2D array directly with 0.0 along both axes
padded_map = np.pad(
    clipped_map,
    pad_width=((n_pad_il, n_pad_il), (n_pad_xl, n_pad_xl)),
    mode="constant",
    constant_values=0.0
)

# Step size between line coordinate indices (usually 1.0)
il_step = np.median(np.diff(il_unique)) if len(il_unique) > 1 else 1.0
xl_step = np.median(np.diff(xl_unique)) if len(xl_unique) > 1 else 1.0

# Calculate new extent covering the zero-padded area
new_xl_min = xl_unique.min() - n_pad_xl * xl_step
new_xl_max = xl_unique.max() + n_pad_xl * xl_step
new_il_min = il_unique.min() - n_pad_il * il_step
new_il_max = il_unique.max() + n_pad_il * il_step

# -------------------------------------------------------------
# 5. Confirmation Dialog Helper
# -------------------------------------------------------------
def ask_confirmation(n_vertices):
    message = (
        f"Polygon created with {n_vertices} vertices.\n\n"
        "Are you okay with the polygon picked and sure you want to continue?"
    )
    for lib in ['PyQt5.QtWidgets', 'PyQt6.QtWidgets', 'PySide2.QtWidgets', 'PySide6.QtWidgets']:
        try:
            mod = __import__(lib, fromlist=['QMessageBox'])
            QMessageBox = getattr(mod, 'QMessageBox')
            yes_btn = getattr(QMessageBox, 'Yes', None) or getattr(QMessageBox.StandardButton, 'Yes')
            no_btn = getattr(QMessageBox, 'No', None) or getattr(QMessageBox.StandardButton, 'No')
            reply = QMessageBox.question(None, "Confirm Polygon", message, yes_btn | no_btn, yes_btn)
            return reply == yes_btn
        except Exception:
            continue

    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        confirmed = messagebox.askyesno("Confirm Polygon", message)
        root.destroy()
        return confirmed
    except Exception:
        pass

    ans = input(f"\n[CONFIRMATION] Polygon has {n_vertices} vertices. Continue? (y/n): ")
    return ans.strip().lower() in ['y', 'yes']


# -------------------------------------------------------------
# 6. Interactive Plot & Polygon Selector
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 8))
plt.subplots_adjust(bottom=0.18)

im0 = ax.imshow(
    padded_map,
    origin="lower",
    aspect="auto",
    extent=[new_xl_min, new_xl_max, new_il_min, new_il_max],
    cmap="viridis"
)

ax.set_title(
    "Interactive Polygon Picker\n"
    "Left-click: Add points | Click 1st point: Close polygon | Drag vertices: Adjust | Enter: Confirm",
    fontsize=11
)
ax.set_xlabel("Crossline")
ax.set_ylabel("Inline")
fig.colorbar(im0, ax=ax, label="RMS Amplitude")

selected_polygon = None

def onselect(verts):
    pass

selector = PolygonSelector(
    ax,
    onselect,
    useblit=True,
    props=dict(color='red', linewidth=2),
    handle_props=dict(marker='o', markersize=6, markerfacecolor='yellow')
)

def confirm(event=None):
    global selected_polygon
    verts = selector.verts

    if len(verts) < 3:
        print("\nWarning: Please draw a valid polygon first (minimum 3 points).")
        return

    is_ok = ask_confirmation(len(verts))

    if is_ok:
        selected_polygon = np.array(verts)
        np.save(output_polygon_file, selected_polygon)
        print(f"\nPolygon confirmed and saved to: {output_polygon_file}")
        print(f"Total vertices: {len(selected_polygon)}")
        plt.close(fig)
    else:
        print("\nSelection rejected. You can continue editing.")

def reset(event=None):
    selector.clear()
    fig.canvas.draw_idle()
    print("\nPolygon cleared. Click on the plot to draw a new one.")

def on_key(event):
    if event.key in ['enter', 'return']:
        confirm()
    elif event.key in ['escape', 'r']:
        reset()

fig.canvas.mpl_connect('key_press_event', on_key)

# Bottom UI buttons
ax_reset = fig.add_axes([0.62, 0.05, 0.15, 0.06])
ax_confirm = fig.add_axes([0.79, 0.05, 0.18, 0.06])

btn_reset = Button(ax_reset, 'Reset / Redraw', color='#f0f0f0', hovercolor='#ffcccc')
btn_confirm = Button(ax_confirm, 'Confirm Polygon', color='#d4edda', hovercolor='#a3d9a5')

btn_reset.on_clicked(reset)
btn_confirm.on_clicked(confirm)

plt.show()

# -------------------------------------------------------------
# 7. Next Steps
# -------------------------------------------------------------
if selected_polygon is not None:
    print(f"\nSaved Polygon coordinates shape: {selected_polygon.shape}")