import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector, Button
from pathlib import Path as FilePath

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
# Physical distance between adjacent samples in the RMS grid.
inline_increment = 25.0       # metres between adjacent inline samples
crossline_increment = 25.0    # metres between adjacent crossline samples

# Option A: Set padding in meters (set to None if defining by line count)
pad_inline_meters = 3000.0     # extra margin in meters along inline direction
pad_crossline_meters = 1000.0 # extra margin in meters along crossline direction

# Option B: Set padding directly in number of lines (used if meters are set to None)
pad_inline_lines = None       # e.g., 20
pad_crossline_lines = None    # e.g., 40

# Optional command-line overrides let the YAML runner use temporary paths
# without changing the standalone script's configuration above.
parser = argparse.ArgumentParser(description="Draw a survey polygon and optional holes")
parser.add_argument("--rms-file")
parser.add_argument("--il-file")
parser.add_argument("--xl-file")
parser.add_argument("--output-polygon-file")
parser.add_argument("--inline-increment", type=float)
parser.add_argument("--crossline-increment", type=float)
parser.add_argument("--pad-inline-meters", type=float)
parser.add_argument("--pad-crossline-meters", type=float)
args = parser.parse_args()
for name in ("rms_file", "il_file", "xl_file", "output_polygon_file",
             "inline_increment", "crossline_increment", "pad_inline_meters",
             "pad_crossline_meters"):
    value = getattr(args, name)
    if value is not None:
        globals()[name] = value
output_holes_file = str(FilePath(output_polygon_file).with_name(
    FilePath(output_polygon_file).stem + "_holes.npz"
))

# -------------------------------------------------------------
# 3. Calculate Padding (Integer Line Counts)
# -------------------------------------------------------------
if matplotlib.get_backend().lower() in {"agg", "pdf", "ps", "svg", "cairo", "template"}:
    raise RuntimeError(
        "create_mask.py needs an interactive Matplotlib backend and a graphical "
        "desktop. Install PyQt5 and run with MPLBACKEND=QtAgg if needed."
    )
if inline_increment <= 0 or crossline_increment <= 0:
    raise ValueError("Inline and crossline increments must be positive")
if pad_inline_meters is not None:
    n_pad_il = int(np.round(pad_inline_meters / inline_increment))
else:
    n_pad_il = int(pad_inline_lines) if pad_inline_lines is not None else 0

if pad_crossline_meters is not None:
    n_pad_xl = int(np.round(pad_crossline_meters / crossline_increment))
else:
    n_pad_xl = int(pad_crossline_lines) if pad_crossline_lines is not None else 0
if n_pad_il < 0 or n_pad_xl < 0:
    raise ValueError("Padding must be non-negative")
if FilePath(output_polygon_file).suffix != ".npy":
    raise ValueError("output_polygon_file must end in .npy")
if not FilePath(output_polygon_file).parent.is_dir():
    raise FileNotFoundError(
        f"Mask output directory does not exist: {FilePath(output_polygon_file).parent}"
    )

print(f"Inline padding   : {n_pad_il} lines ({n_pad_il * inline_increment:.1f} m)")
print(f"Crossline padding: {n_pad_xl} lines ({n_pad_xl * crossline_increment:.1f} m)")

# -------------------------------------------------------------
# 4. Load Data & Pad Array with 0
# -------------------------------------------------------------
rms_map = np.load(rms_file)
il_unique = np.load(il_file)
xl_unique = np.load(xl_file)
if rms_map.shape != (len(il_unique), len(xl_unique)):
    raise ValueError("RMS map shape does not match the inline/crossline axes")
if not np.any(np.isfinite(rms_map)):
    raise ValueError("RMS map contains no finite samples")
if np.any(np.isinf(rms_map)):
    raise ValueError("RMS map contains infinite samples; check the SEG-Y amplitudes")

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
    extent=[new_xl_min - xl_step / 2, new_xl_max + xl_step / 2,
            new_il_min - il_step / 2, new_il_max + il_step / 2],
    cmap="viridis"
)

ax.set_title(
    "Draw the survey boundary; click the first point to close it\n"
    "Enter: confirm | Esc: redraw | Backspace: undo | Finish Masks: save",
    fontsize=11
)
ax.set_xlabel("Crossline")
ax.set_ylabel("Inline")
fig.colorbar(im0, ax=ax, label="RMS Amplitude")

selected_polygon = None
hole_polygons = []
confirmed_artists = []
masks_saved = False

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
    verts = np.asarray(selector.verts, dtype=float)

    if len(verts) < 3:
        print("\nWarning: Please draw a valid polygon first (minimum 3 points).")
        return

    if selected_polygon is not None:
        outer_path = Path(np.vstack((selected_polygon, selected_polygon[0])))
        hole_path = Path(np.vstack((verts, verts[0])))
        if not outer_path.contains_points(verts).all() or outer_path.intersects_path(
            hole_path, filled=False
        ):
            print("\nHole must lie completely inside the survey boundary. Redraw it.")
            return

    if not ask_confirmation(len(verts)):
        print("\nSelection rejected. You can continue editing.")
        return

    if selected_polygon is None:
        selected_polygon = verts.copy()
        color = "red"
        ax.set_title(
            "Draw a hole inside the survey, or click Finish Masks\n"
            "Confirm each hole with Enter; Backspace removes the last confirmed polygon",
            fontsize=11
        )
        print("\nSurvey boundary confirmed. Draw any holes, or click Finish Masks.")
    else:
        hole_polygons.append(verts.copy())
        color = "orange"
        print(f"\nHole {len(hole_polygons)} confirmed. Draw another, or click Finish Masks.")

    closed = np.vstack((verts, verts[0]))
    line, = ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=2)
    fill = ax.fill(verts[:, 0], verts[:, 1], color=color, alpha=0.15)[0]
    confirmed_artists.append((line, fill))
    selector.clear()
    fig.canvas.draw_idle()


def undo_last(event=None):
    global selected_polygon
    if not confirmed_artists:
        print("\nNo confirmed polygon to undo.")
        return
    for artist in confirmed_artists.pop():
        artist.remove()
    if hole_polygons:
        hole_polygons.pop()
        print("\nLast hole removed.")
    else:
        selected_polygon = None
        ax.set_title(
            "Draw the survey boundary; click the first point to close it\n"
            "Enter: confirm | Esc: redraw | Backspace: undo | Finish Masks: save",
            fontsize=11
        )
        print("\nSurvey boundary removed. Draw it again.")
    fig.canvas.draw_idle()


def finish(event=None):
    global masks_saved
    if selected_polygon is None:
        print("\nConfirm the survey boundary before finishing.")
        return
    if selector.verts:
        print("\nConfirm or reset the current hole before finishing.")
        return
    np.save(output_polygon_file, selected_polygon)
    np.savez(output_holes_file, **{
        f"hole_{i}": polygon for i, polygon in enumerate(hole_polygons, 1)
    })
    print(f"\nSurvey boundary saved to: {output_polygon_file}")
    print(f"{len(hole_polygons)} hole(s) saved to: {output_holes_file}")
    masks_saved = True
    plt.close(fig)

def reset(event=None):
    selector.clear()
    fig.canvas.draw_idle()
    print("\nPolygon cleared. Click on the plot to draw a new one.")

def on_key(event):
    if event.key in ['enter', 'return']:
        confirm()
    elif event.key in ['escape', 'r']:
        reset()
    elif event.key == 'backspace':
        undo_last()

fig.canvas.mpl_connect('key_press_event', on_key)

# Bottom UI buttons
ax_reset = fig.add_axes([0.27, 0.05, 0.16, 0.06])
ax_undo = fig.add_axes([0.45, 0.05, 0.15, 0.06])
ax_confirm = fig.add_axes([0.62, 0.05, 0.16, 0.06])
ax_finish = fig.add_axes([0.80, 0.05, 0.18, 0.06])

btn_reset = Button(ax_reset, 'Reset / Redraw', color='#f0f0f0', hovercolor='#ffcccc')
btn_undo = Button(ax_undo, 'Undo Last', color='#f0f0f0', hovercolor='#ffcccc')
btn_confirm = Button(ax_confirm, 'Confirm Polygon', color='#d4edda', hovercolor='#a3d9a5')
btn_finish = Button(ax_finish, 'Finish Masks', color='#d4edda', hovercolor='#a3d9a5')

btn_reset.on_clicked(reset)
btn_undo.on_clicked(undo_last)
btn_confirm.on_clicked(confirm)
btn_finish.on_clicked(finish)

plt.show()

# -------------------------------------------------------------
# 7. Next Steps
# -------------------------------------------------------------
if masks_saved:
    print(f"\nSaved Polygon coordinates shape: {selected_polygon.shape}")
    print(f"Confirmed holes: {len(hole_polygons)}")
else:
    print("\nNo mask files were saved. Confirm the boundary and click Finish Masks.")
