import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector, Button
from pathlib import Path as FilePath
from scipy.ndimage import uniform_filter

from footprint_attenuation import save_valid_mask

# Import native Qt from whichever binding Matplotlib is using
try:
    from PyQt5 import QtWidgets, QtCore
except ImportError:
    try:
        from PyQt6 import QtWidgets, QtCore
    except ImportError:
        try:
            from PySide2 import QtWidgets, QtCore
        except ImportError:
            from PySide6 import QtWidgets, QtCore

def resolve_qt_enum(scope, name):
    cls = getattr(QtCore.Qt, scope, None)
    if cls is not None and hasattr(cls, name):
        return getattr(cls, name)
    return getattr(QtCore.Qt, name, 0)

STAY_ON_TOP = resolve_qt_enum("WindowType", "WindowStaysOnTopHint")
ALIGN_CENTER = resolve_qt_enum("AlignmentFlag", "AlignCenter")

# -------------------------------------------------------------
# 1. CLI Arguments & Configuration
# -------------------------------------------------------------
parser = argparse.ArgumentParser(description="Multi-map aware boundary picker and blended RMS generator")
parser.add_argument("--rms-files", nargs="+", help="One or more RMS map .npy files")
parser.add_argument("--rms-file", help="Single RMS file (backward compatibility)")
parser.add_argument("--il-file", required=True)
parser.add_argument("--xl-file", required=True)
parser.add_argument("--scope", choices=["single", "multiple"], default=None,
                    help="Use a single mask for all maps, or multiple custom masks")
parser.add_argument("--preview-index", type=int, default=0,
                    help="Map index to display when drawing a single shared mask")
parser.add_argument("--output-blended-file", default=None,
                    help="Explicit output path (only used if processing a single map)")
parser.add_argument("--inline-increment", type=float, required=True)
parser.add_argument("--crossline-increment", type=float, required=True)
parser.add_argument("--pad-inline-meters", type=float, required=True)
parser.add_argument("--pad-crossline-meters", type=float, required=True)
parser.add_argument("--filter-il-meters", type=float, required=True)
parser.add_argument("--filter-xl-meters", type=float, required=True)
parser.add_argument("--hole-filter-il-meters", type=float)
parser.add_argument("--hole-filter-xl-meters", type=float)
args = parser.parse_args()

# Collect input maps list
raw_map_list = args.rms_files if args.rms_files else [args.rms_file] if args.rms_file else []
if not raw_map_list:
    raise ValueError("Provide at least one RMS map via --rms-files or --rms-file")
rms_map_paths = [FilePath(p).resolve() for p in raw_map_list]
if not 0 <= args.preview_index < len(rms_map_paths):
    raise ValueError("--preview-index must select one of the supplied RMS maps")

il_file = FilePath(args.il_file).resolve()
xl_file = FilePath(args.xl_file).resolve()
inline_increment = args.inline_increment
crossline_increment = args.crossline_increment
pad_inline_meters = args.pad_inline_meters
pad_crossline_meters = args.pad_crossline_meters

# Detect masking scope if not explicitly given via CLI
scope = args.scope
if scope is None and len(rms_map_paths) > 1:
    scope_file = rms_map_paths[0].with_name(rms_map_paths[0].stem + "_mask_scope.txt")
    if scope_file.is_file():
        scope = scope_file.read_text(encoding="utf-8").strip().lower()
    else:
        scope = "single"
elif scope is None:
    scope = "single"

# -------------------------------------------------------------
# 2. Geometry & Padding Setup
# -------------------------------------------------------------
il_unique = np.load(il_file)
xl_unique = np.load(xl_file)
survey_shape = (len(il_unique), len(xl_unique))

n_pad_il = int(np.round(pad_inline_meters / inline_increment))
n_pad_xl = int(np.round(pad_crossline_meters / crossline_increment))

il_step = np.median(np.diff(il_unique)) if len(il_unique) > 1 else 1.0
xl_step = np.median(np.diff(xl_unique)) if len(xl_unique) > 1 else 1.0

padded_il = np.r_[il_unique[0] - np.arange(n_pad_il, 0, -1) * il_step,
                  il_unique,
                  il_unique[-1] + np.arange(1, n_pad_il + 1) * il_step]
padded_xl = np.r_[xl_unique[0] - np.arange(n_pad_xl, 0, -1) * xl_step,
                  xl_unique,
                  xl_unique[-1] + np.arange(1, n_pad_xl + 1) * xl_step]
extent = [padded_xl.min() - xl_step / 2, padded_xl.max() + xl_step / 2,
          padded_il.min() - il_step / 2, padded_il.max() + il_step / 2]

XL, IL = np.meshgrid(padded_xl, padded_il)
grid_points = np.column_stack((XL.ravel(), IL.ravel()))

def pad_map(raw_map):
    p1, p99 = np.nanpercentile(raw_map, [1.0, 99.0])
    clipped = np.nan_to_num(np.clip(raw_map, p1, p99), nan=0.0)
    return np.pad(clipped, ((n_pad_il, n_pad_il), (n_pad_xl, n_pad_xl)),
                  mode="constant", constant_values=0.0), p99

def meters_to_window(meters, inc):
    if meters is None or meters <= 0.0:
        return 1
    w = int(np.round(meters / inc))
    if w <= 1:
        return 3
    if w % 2 == 0:
        w += 1
    return w

def smooth_binary_mask(grid, win_il, win_xl):
    if win_il <= 1 and win_xl <= 1:
        return grid.astype(np.float32)
    s = uniform_filter(grid.astype(np.float32), size=(max(1, win_il), max(1, win_xl)), mode="nearest")
    s = np.clip(s, 0.0, 1.0)
    s[s < 1.0] = 0.0
    s = uniform_filter(s, size=(max(1, win_il), max(1, win_xl)), mode="nearest")
    return np.clip(s, 0.0, 1.0)

def apply_mask_to_map(raw_map, outer_grid, hole_grid, b_il, b_xl, h_il, h_xl):
    padded, _ = pad_map(raw_map)
    mask_grid = outer_grid & ~hole_grid
    inside_vals = padded[mask_grid & (padded > 0)]
    median_val = float(np.median(inside_vals)) if inside_vals.size > 0 else 0.0

    win_b_il = meters_to_window(b_il, inline_increment)
    win_b_xl = meters_to_window(b_xl, crossline_increment)
    s_outer = smooth_binary_mask(outer_grid, win_b_il, win_b_xl)

    if np.any(hole_grid):
        win_h_il = meters_to_window(h_il, inline_increment)
        win_h_xl = meters_to_window(h_xl, crossline_increment)
        s_holes = smooth_binary_mask(~hole_grid, win_h_il, win_h_xl)
    else:
        s_holes = 1.0

    smooth = s_outer * s_holes
    full_blended = padded * smooth + (1.0 - smooth) * median_val

    il_slice = slice(n_pad_il, n_pad_il + len(il_unique)) if n_pad_il > 0 else slice(None)
    xl_slice = slice(n_pad_xl, n_pad_xl + len(xl_unique)) if n_pad_xl > 0 else slice(None)
    return full_blended[il_slice, xl_slice], full_blended, median_val


def core_valid_mask(outer_grid, hole_grid):
    return (outer_grid & ~hole_grid)[
        n_pad_il:n_pad_il + len(il_unique),
        n_pad_xl:n_pad_xl + len(xl_unique),
    ]

# -------------------------------------------------------------
# 3. Interactive Workflow Engine
# -------------------------------------------------------------
def run_masking_session(active_map_path, window_title_prefix=""):
    active_rms = np.load(active_map_path)
    if active_rms.shape != survey_shape:
        raise ValueError(f"{active_map_path.name} shape does not match coordinate axes")

    padded_display, p99 = pad_map(active_rms)

    # PHASE 1: Fast Picking
    selected_polygon = None
    hole_polygons = []
    confirmed_artists = []
    proceed_to_qc = False

    fig1, ax1 = plt.subplots(figsize=(14, 8))
    plt.subplots_adjust(bottom=0.12)

    ds = max(1, min(padded_display.shape) // 800)
    ax1.imshow(padded_display[::ds, ::ds], origin="lower", aspect="auto", extent=extent,
               cmap="viridis", vmin=0, vmax=p99, interpolation="nearest")
    ax1.set_title(f"{window_title_prefix}PHASE 1: Draw survey boundary; click first point to close\n"
                  f"Enter: confirm | Esc: reset | Backspace: undo | Proceed to Ramp QC: continue", fontsize=11)
    ax1.set_xlabel("Crossline")
    ax1.set_ylabel("Inline")

    def onselect(verts):
        pass

    selector = PolygonSelector(ax1, onselect, useblit=True, props=dict(color='red', linewidth=2),
                               handle_props=dict(marker='o', markersize=5, markerfacecolor='yellow'))

    def confirm_poly(event=None):
        nonlocal selected_polygon
        verts = np.asarray(selector.verts, dtype=float)
        if len(verts) < 3:
            return
        if selected_polygon is not None:
            outer_p = Path(np.vstack((selected_polygon, selected_polygon[0])))
            hole_p = Path(np.vstack((verts, verts[0])))
            if not outer_p.contains_points(verts).all() or outer_p.intersects_path(hole_p, filled=False):
                print("\nHole must lie completely inside the survey boundary.")
                return

        if selected_polygon is None:
            selected_polygon = verts.copy()
            color = "red"
            ax1.set_title(f"{window_title_prefix}Boundary confirmed. Draw holes + Enter | Proceed to Ramp QC to continue", fontsize=11)
        else:
            hole_polygons.append(verts.copy())
            color = "orange"
            ax1.set_title(f"{window_title_prefix}Boundary + {len(hole_polygons)} hole(s). Draw next hole or click Proceed", fontsize=11)

        closed = np.vstack((verts, verts[0]))
        line, = ax1.plot(closed[:, 0], closed[:, 1], color=color, linewidth=2)
        fill = ax1.fill(verts[:, 0], verts[:, 1], color=color, alpha=0.18)[0]
        confirmed_artists.append((line, fill))
        selector.clear()
        fig1.canvas.draw_idle()

    def undo_poly(event=None):
        nonlocal selected_polygon
        if not confirmed_artists:
            return
        for artist in confirmed_artists.pop():
            artist.remove()
        if hole_polygons:
            hole_polygons.pop()
        else:
            selected_polygon = None
            ax1.set_title(f"{window_title_prefix}Draw survey boundary; click first point to close", fontsize=11)
        fig1.canvas.draw_idle()

    def finish_picking(event=None):
        nonlocal proceed_to_qc
        if selected_polygon is None:
            print("\nConfirm the survey boundary before proceeding.")
            return
        proceed_to_qc = True
        plt.close(fig1)

    fig1.canvas.mpl_connect('key_press_event', lambda ev: confirm_poly() if ev.key in ('enter', 'return')
                            else undo_poly() if ev.key == 'backspace' else None)

    ax_undo = fig1.add_axes([0.38, 0.03, 0.16, 0.05])
    ax_conf = fig1.add_axes([0.56, 0.03, 0.20, 0.05])
    ax_next = fig1.add_axes([0.78, 0.03, 0.18, 0.05])

    btn_undo = Button(ax_undo, 'Undo Last', color='#f0f0f0')
    btn_conf = Button(ax_conf, 'Confirm Polygon (Enter)', color='#d4edda')
    btn_next = Button(ax_next, 'Proceed to Ramp QC', color='#cce5ff')

    btn_undo.on_clicked(undo_poly)
    btn_conf.on_clicked(confirm_poly)
    btn_next.on_clicked(finish_picking)

    plt.show()

    if not proceed_to_qc or selected_polygon is None:
        return None, None, None

    # Pre-rasterize geometry for rapid blending
    outer_grid = Path(selected_polygon).contains_points(grid_points).reshape(padded_display.shape)
    hole_grid = np.zeros(padded_display.shape, dtype=bool)
    for h in hole_polygons:
        hole_grid |= Path(h).contains_points(grid_points, radius=1e-9).reshape(padded_display.shape)

    # PHASE 2: Ramp Tuning & QC Viewer
    fig2, ax2 = plt.subplots(figsize=(14, 8.5))
    plt.subplots_adjust(top=0.92, bottom=0.08)

    im_qc = ax2.imshow(padded_display, origin="lower", aspect="auto", extent=extent,
                       cmap="viridis", vmin=0, vmax=p99, interpolation="nearest")
    ax2.set_xlabel("Crossline", fontsize=11)
    ax2.set_ylabel("Inline", fontsize=11)
    fig2.colorbar(im_qc, ax=ax2, label="RMS Amplitude")

    poly_closed = np.vstack([selected_polygon, selected_polygon[0]])
    line_outer, = ax2.plot(poly_closed[:, 0], poly_closed[:, 1], "r-", linewidth=1.8)
    hole_lines = []
    for h in hole_polygons:
        h_c = np.vstack([h, h[0]])
        hl, = ax2.plot(h_c[:, 0], h_c[:, 1], color="orange", linewidth=1.5)
        hole_lines.append(hl)

    b_il_init, b_xl_init = args.filter_il_meters, args.filter_xl_meters
    h_il_init, h_xl_init = args.hole_filter_il_meters, args.hole_filter_xl_meters
    h_il_init = b_il_init if h_il_init is None else h_il_init
    h_xl_init = b_xl_init if h_xl_init is None else h_xl_init

    _, init_full, init_med = apply_mask_to_map(active_rms, outer_grid, hole_grid,
                                              b_il_init, b_xl_init, h_il_init, h_xl_init)

    view_state = {
        "mode": "blended",
        "blended": init_full,
        "median": init_med,
        "params": (b_il_init, b_xl_init, h_il_init, h_xl_init)
    }

    def refresh_qc_figure():
        b_il, b_xl, h_il, h_xl = view_state["params"]
        if view_state["mode"] == "blended" and view_state["blended"] is not None:
            im_qc.set_data(view_state["blended"])
            line_outer.set_visible(False)
            for hl in hole_lines:
                hl.set_visible(False)
            ax2.set_title(
                f"{window_title_prefix}[QC VIEW: Blended RMS | Median: {view_state['median']:.2f}]\n"
                f"Boundary: {b_il:.0f}m IL x {b_xl:.0f}m XL  |  Holes: {h_il:.0f}m IL x {h_xl:.0f}m XL  |  [SPACE/T]: Toggle",
                fontsize=11.5, fontweight="bold", color="#0055ff", pad=10
            )
        else:
            im_qc.set_data(padded_display)
            line_outer.set_visible(True)
            for hl in hole_lines:
                hl.set_visible(True)
            ax2.set_title(
                f"{window_title_prefix}[RAW RMS VIEW]\nShowing picked boundary & {len(hole_polygons)} hole(s)  |  [SPACE/T]: Toggle",
                fontsize=11.5, fontweight="bold", color="black", pad=10
            )
        fig2.canvas.draw_idle()

    def toggle_qc_view():
        view_state["mode"] = "raw" if view_state["mode"] == "blended" else "blended"
        refresh_qc_figure()

    fig2.canvas.mpl_connect("key_press_event", lambda ev: toggle_qc_view() if ev.key in (" ", "t", "T") else None)

    final_params = None

    class RampControlWindow(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Ramp Control Box")
            self.setFixedSize(360, 310)
            if STAY_ON_TOP:
                self.setWindowFlags(self.windowFlags() | STAY_ON_TOP)

            layout = QtWidgets.QVBoxLayout(self)
            layout.setSpacing(10)

            title = QtWidgets.QLabel("<b>Feathering Ramps (Meters)</b>")
            if ALIGN_CENTER:
                title.setAlignment(ALIGN_CENTER)
            layout.addWidget(title)

            form = QtWidgets.QFormLayout()
            self.in_b_il = QtWidgets.QLineEdit(f"{b_il_init:.0f}")
            self.in_b_xl = QtWidgets.QLineEdit(f"{b_xl_init:.0f}")
            self.in_h_il = QtWidgets.QLineEdit(f"{h_il_init:.0f}")
            self.in_h_xl = QtWidgets.QLineEdit(f"{h_xl_init:.0f}")

            form.addRow("Boundary Inline (m):", self.in_b_il)
            form.addRow("Boundary Crossline (m):", self.in_b_xl)
            form.addRow("Hole Inline (m):", self.in_h_il)
            form.addRow("Hole Crossline (m):", self.in_h_xl)
            layout.addLayout(form)

            self.btn_apply = QtWidgets.QPushButton("Apply Mask")
            self.btn_apply.setStyleSheet("background-color: #cce5ff; font-weight: bold; height: 32px;")
            self.btn_apply.clicked.connect(self.apply_values)
            layout.addWidget(self.btn_apply)

            self.btn_toggle = QtWidgets.QPushButton("Toggle Raw / Blended (Space)")
            self.btn_toggle.setStyleSheet("height: 28px;")
            self.btn_toggle.clicked.connect(toggle_qc_view)
            layout.addWidget(self.btn_toggle)

            finish_label = "Save & Apply to All Maps" if scope == "single" and len(rms_map_paths) > 1 else "Save & Finish"
            self.btn_save = QtWidgets.QPushButton(finish_label)
            self.btn_save.setStyleSheet("background-color: #d4edda; font-weight: bold; height: 32px;")
            self.btn_save.clicked.connect(self.save_and_exit)
            layout.addWidget(self.btn_save)

        def parse_fields(self):
            try:
                values = tuple(float(field.text()) for field in (
                    self.in_b_il, self.in_b_xl, self.in_h_il, self.in_h_xl
                ))
            except ValueError:
                values = None
            if values is None or not all(np.isfinite(v) and v >= 0 for v in values):
                QtWidgets.QMessageBox.warning(
                    self, "Invalid ramp", "Enter four finite, nonnegative distances in meters."
                )
                return None
            return values

        def apply_values(self):
            values = self.parse_fields()
            if values is None:
                return
            b_il, b_xl, h_il, h_xl = values
            _, full_blend, med = apply_mask_to_map(active_rms, outer_grid, hole_grid, b_il, b_xl, h_il, h_xl)
            view_state["params"] = (b_il, b_xl, h_il, h_xl)
            view_state["blended"] = full_blend
            view_state["median"] = med
            view_state["mode"] = "blended"
            refresh_qc_figure()

        def save_and_exit(self):
            nonlocal final_params
            final_params = self.parse_fields()
            if final_params is None:
                return
            self.close()
            plt.close(fig2)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    control_box = RampControlWindow()
    control_box.show()
    refresh_qc_figure()
    plt.show()

    return outer_grid, hole_grid, final_params

# -------------------------------------------------------------
# 4. Main Batch Orchestration
# -------------------------------------------------------------
def main():
    if matplotlib.get_backend().lower() in {
        "agg", "pdf", "ps", "svg", "cairo", "template"
    }:
        raise RuntimeError(
            "Interactive masking needs a graphical Matplotlib backend. "
            "Run with MPLBACKEND=QtAgg on a graphical desktop."
        )
    print(f"\nMasking Scope: {scope.upper()} ({len(rms_map_paths)} map{'s' if len(rms_map_paths) > 1 else ''})")

    if scope == "single":
        # Process ONCE using the preview map, then batch-blend all maps
        preview_map_path = rms_map_paths[args.preview_index]
        print(f"Opening interactive picking for shared mask using: {preview_map_path.name}")

        outer_grid, hole_grid, params = run_masking_session(
            preview_map_path, window_title_prefix="[SHARED MASK] "
        )
        if params is None:
            raise RuntimeError("Masking cancelled before a blended RMS map was saved")

        b_il, b_xl, h_il, h_xl = params
        valid_mask = core_valid_mask(outer_grid, hole_grid)
        print("\n" + "=" * 65)
        print(f"APPLYING SHARED MASK TO ALL {len(rms_map_paths)} RMS MAPS")
        print(f"Ramps: Boundary=({b_il:.0f}m, {b_xl:.0f}m), Holes=({h_il:.0f}m, {h_xl:.0f}m)")
        print("=" * 65)

        for p in rms_map_paths:
            cur_raw = np.load(p)
            blended, _, _ = apply_mask_to_map(cur_raw, outer_grid, hole_grid, b_il, b_xl, h_il, h_xl)

            if len(rms_map_paths) == 1 and args.output_blended_file:
                out_path = FilePath(args.output_blended_file)
            else:
                out_path = p.with_name(p.stem + "_blended.npy")

            np.save(out_path, blended)
            save_valid_mask(out_path, valid_mask)
            print(f" -> Saved blended RMS: {out_path.name}")

    else:
        # Step through each map sequentially
        print("\n" + "=" * 65)
        print("MULTIPLE MASKS: Custom boundaries & ramps per map")
        print("=" * 65)

        for idx, p in enumerate(rms_map_paths, 1):
            print(f"\n--- [Map {idx}/{len(rms_map_paths)}] {p.name} ---")
            outer_grid, hole_grid, params = run_masking_session(
                p, window_title_prefix=f"[{idx}/{len(rms_map_paths)}: {p.name}] "
            )
            if params is None:
                raise RuntimeError(f"Masking cancelled before saving {p.name}")

            b_il, b_xl, h_il, h_xl = params
            valid_mask = core_valid_mask(outer_grid, hole_grid)
            cur_raw = np.load(p)
            blended, _, _ = apply_mask_to_map(cur_raw, outer_grid, hole_grid, b_il, b_xl, h_il, h_xl)

            out_path = p.with_name(p.stem + "_blended.npy")
            np.save(out_path, blended)
            save_valid_mask(out_path, valid_mask)
            print(f" -> Saved blended RMS: {out_path.name}")

    print("\nAll blended RMS maps generated successfully.")

if __name__ == "__main__":
    main()
