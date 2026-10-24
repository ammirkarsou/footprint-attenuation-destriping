import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector, Button
from pathlib import Path as FilePath
from scipy.ndimage import uniform_filter

from footprint_attenuation import save_valid_mask, save_mask_taper

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

scope = args.scope

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

def mask_blend_weights(outer_grid, hole_grid, b_il, b_xl, h_il, h_xl):
    win_b_il = meters_to_window(b_il, inline_increment)
    win_b_xl = meters_to_window(b_xl, crossline_increment)
    s_outer = smooth_binary_mask(outer_grid, win_b_il, win_b_xl)

    if np.any(hole_grid):
        win_h_il = meters_to_window(h_il, inline_increment)
        win_h_xl = meters_to_window(h_xl, crossline_increment)
        s_holes = smooth_binary_mask(~hole_grid, win_h_il, win_h_xl)
    else:
        s_holes = 1.0

    return (s_outer * s_holes).astype(np.float32)


def coefficient_taper(outer_grid, hole_grid, b_il, b_xl, h_il, h_xl):
    b_size = (meters_to_window(b_il, inline_increment),
              meters_to_window(b_xl, crossline_increment))
    # A single smoothing pass extends the outer ramp beyond the picked border.
    outer = uniform_filter(outer_grid.astype(np.float32), size=b_size, mode="nearest")
    if np.any(hole_grid):
        hole = smooth_binary_mask(
            ~hole_grid, meters_to_window(h_il, inline_increment),
            meters_to_window(h_xl, crossline_increment),
        )
        outer *= hole  # Keep the existing exclusion of hole interiors.
    return np.clip(outer, 0.0, 1.0).astype(np.float32)


def apply_mask_to_map(raw_map, outer_grid, hole_grid, b_il, b_xl, h_il, h_xl):
    padded, _ = pad_map(raw_map)
    mask_grid = outer_grid & ~hole_grid
    inside_vals = padded[mask_grid & (padded > 0)]
    median_val = float(np.median(inside_vals)) if inside_vals.size > 0 else 0.0

    smooth = mask_blend_weights(outer_grid, hole_grid, b_il, b_xl, h_il, h_xl)
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
class MaskPicker:
    """Pick, reuse, and revise one outer polygon and optional holes."""

    def __init__(self, fig, ax, title_prefix="", initial_outer=None, existing_holes=None):
        self.fig = fig
        self.ax = ax
        self.title_prefix = title_prefix
        self.outer = None
        self.holes = []
        self.outer_artists = None
        self.hole_artists = []
        self.edit_target = None
        self.edit_hole_index = None
        self.proceed = False
        self.reused_outer = initial_outer is not None and existing_holes is None

        self.selector = PolygonSelector(
            ax, lambda verts: None, useblit=True,
            props=dict(color="red", linewidth=2),
            handle_props=dict(marker="o", markersize=5, markerfacecolor="yellow"),
        )
        # PolygonSelector otherwise treats a right-click away from a handle
        # as a new vertex. Reserve right-click for removing the last point.
        # Motion events have button=None. Keep those so the live segment
        # follows the cursor, while rejecting right-clicks as new vertices.
        self.selector.validButtons = [1, None]
        if initial_outer is not None:
            if existing_holes is None:
                self.selector.verts = np.asarray(initial_outer, dtype=float).tolist()
            else:
                self.outer = np.asarray(initial_outer, dtype=float).copy()
                self.outer_artists = self._draw(self.outer, "red")
                for hole in existing_holes:
                    polygon = np.asarray(hole, dtype=float).copy()
                    self.holes.append(polygon)
                    self.hole_artists.append(self._draw(polygon, "orange"))
        self._set_title()

        fig.canvas.mpl_connect("button_press_event", self._on_mouse)
        fig.canvas.mpl_connect("key_press_event", self._on_key)
        buttons = (
            ("Undo Last", self.undo_poly, 0.38, 0.16, "#f0f0f0"),
            ("Confirm Polygon (Enter)", self.confirm_poly, 0.56, 0.20, "#d4edda"),
            ("Proceed to Ramp QC", self.finish_picking, 0.78, 0.18, "#cce5ff"),
        )
        self.buttons = []
        for label, callback, left, width, color in buttons:
            button = Button(fig.add_axes([left, 0.03, width, 0.05]), label, color=color)
            button.on_clicked(callback)
            self.buttons.append(button)

    def _set_title(self, message=None):
        if message is None:
            if self.outer is None and self.reused_outer:
                message = "Previous boundary loaded: Enter to reuse, drag vertices to adjust, or R to redraw"
            elif self.outer is None:
                message = "Draw boundary; click first point to close, then Enter to confirm"
            else:
                message = f"Boundary confirmed with {len(self.holes)} hole(s); draw holes or edit a polygon"
        self.ax.set_title(
            f"{self.title_prefix}{message}\n"
            "Right-click: remove point | B: edit boundary | E: edit last hole | R: redraw | Esc: cancel edit",
            fontsize=11,
        )
        self.fig.canvas.draw_idle()

    def _draw(self, vertices, color):
        closed = np.vstack((vertices, vertices[0]))
        line, = self.ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=2)
        fill = self.ax.fill(vertices[:, 0], vertices[:, 1], color=color, alpha=0.18)[0]
        return line, fill

    @staticmethod
    def _remove_artists(artists):
        if artists is not None:
            for artist in artists:
                artist.remove()

    @staticmethod
    def _hole_inside(outer, hole):
        outer_path = Path(np.vstack((outer, outer[0])))
        hole_path = Path(np.vstack((hole, hole[0])))
        return (outer_path.contains_points(hole).all()
                and not outer_path.intersects_path(hole_path, filled=False))

    def confirm_poly(self, event=None):
        vertices = np.asarray(self.selector.verts, dtype=float)
        if len(vertices) < 3 or not np.all(np.isfinite(vertices)):
            self._set_title("A polygon needs at least three finite points")
            return
        area = np.dot(vertices[:, 0], np.roll(vertices[:, 1], -1)) - np.dot(
            vertices[:, 1], np.roll(vertices[:, 0], -1)
        )
        if np.isclose(area, 0):
            self._set_title("Polygon has zero area; adjust its points")
            return

        if self.outer is None or self.edit_target == "outer":
            if any(not self._hole_inside(vertices, hole) for hole in self.holes):
                self._set_title("Edited boundary must still contain every hole")
                return
            self._remove_artists(self.outer_artists)
            self.outer = vertices.copy()
            self.outer_artists = self._draw(self.outer, "red")
            self.reused_outer = False
        else:
            if not self._hole_inside(self.outer, vertices):
                self._set_title("Hole must lie completely inside the boundary")
                return
            if self.edit_target == "hole":
                index = self.edit_hole_index
                self._remove_artists(self.hole_artists[index])
                self.holes[index] = vertices.copy()
                self.hole_artists[index] = self._draw(vertices, "orange")
            else:
                self.holes.append(vertices.copy())
                self.hole_artists.append(self._draw(vertices, "orange"))

        self.edit_target = None
        self.edit_hole_index = None
        self.selector.clear()
        self._set_title()

    def undo_vertex(self):
        vertices = list(self.selector.verts)
        if not vertices:
            return False
        remaining = [tuple(point) for point in vertices[:-1]]
        self.selector.clear()
        if remaining:
            # Matplotlib exposes no public API for an incomplete polygon with
            # existing vertices. This keeps the selector editable after undo.
            self.selector._xys = [*remaining, remaining[-1]]
            self.selector._selection_completed = False
            self.selector._draw_polygon()
        self.reused_outer = False
        self._set_title()
        return True

    def undo_poly(self, event=None):
        if self.undo_vertex():
            return
        if self.edit_target is not None:
            self.edit_target = None
            self.edit_hole_index = None
        elif self.holes:
            self.holes.pop()
            self._remove_artists(self.hole_artists.pop())
        elif self.outer is not None:
            self.outer = None
            self._remove_artists(self.outer_artists)
            self.outer_artists = None
        self._set_title()

    def edit_boundary(self, event=None):
        if self.outer is None:
            self._set_title("Confirm a boundary before editing it")
            return
        self.selector.clear()
        self.selector.verts = self.outer.tolist()
        self.edit_target = "outer"
        self.edit_hole_index = None
        self._set_title("Drag boundary vertices, then Enter to confirm; Esc cancels edit")

    def edit_last_hole(self, event=None):
        if not self.holes:
            self._set_title("There is no confirmed hole to edit")
            return
        self.selector.clear()
        self.edit_hole_index = len(self.holes) - 1
        self.selector.verts = self.holes[self.edit_hole_index].tolist()
        self.edit_target = "hole"
        self._set_title("Drag hole vertices, then Enter to confirm; Esc cancels edit")

    def redraw(self, event=None):
        self.selector.clear()
        self._remove_artists(self.outer_artists)
        for artists in self.hole_artists:
            self._remove_artists(artists)
        self.outer = None
        self.holes = []
        self.outer_artists = None
        self.hole_artists = []
        self.edit_target = None
        self.edit_hole_index = None
        self.reused_outer = False
        self._set_title()

    def finish_picking(self, event=None):
        if self.outer is None or self.edit_target is not None or self.selector.verts:
            self._set_title("Confirm the boundary and any current edits before Ramp QC")
            return
        self.proceed = True
        plt.close(self.fig)

    def _on_mouse(self, event):
        if event.inaxes is self.ax and event.button == 3:
            self.undo_vertex()

    def _on_key(self, event):
        if event.key in ("enter", "return"):
            self.confirm_poly()
        elif event.key in ("backspace", "delete"):
            self.undo_poly()
        elif event.key == "b":
            self.edit_boundary()
        elif event.key == "e":
            self.edit_last_hole()
        elif event.key == "r":
            self.redraw()
        elif event.key == "escape":
            if self.edit_target is not None:
                self.edit_target = None
                self.edit_hole_index = None
                self.selector.clear()
                self._set_title("Edit cancelled; confirmed polygon retained")
            else:
                self.reused_outer = False
                self._set_title()


def pick_mask_polygons(padded_display, p99, title_prefix, initial_outer=None,
                       existing_holes=None):
    fig, ax = plt.subplots(figsize=(14, 8))
    plt.subplots_adjust(bottom=0.12)
    ds = max(1, min(padded_display.shape) // 800)
    ax.imshow(padded_display[::ds, ::ds], origin="lower", aspect="auto", extent=extent,
              cmap="viridis", vmin=0, vmax=p99, interpolation="nearest")
    ax.set_xlabel("Crossline")
    ax.set_ylabel("Inline")
    picker = MaskPicker(fig, ax, title_prefix, initial_outer, existing_holes)
    plt.show()
    if not picker.proceed:
        return None, None
    return picker.outer.copy(), [hole.copy() for hole in picker.holes]


def review_mask_ramps(active_rms, padded_display, p99, outer_grid, hole_grid, window_title_prefix,
                      initial_params=None):
    fig2, ax2 = plt.subplots(figsize=(14, 8.5))
    plt.subplots_adjust(top=0.92, bottom=0.08)
    im_qc = ax2.imshow(padded_display, origin="lower", aspect="auto", extent=extent,
                       cmap="viridis", vmin=0, vmax=p99, interpolation="nearest")
    ax2.set_xlabel("Crossline", fontsize=11)
    ax2.set_ylabel("Inline", fontsize=11)
    fig2.colorbar(im_qc, ax=ax2, label="RMS Amplitude")
    if initial_params is None:
        b_il, b_xl = args.filter_il_meters, args.filter_xl_meters
        h_il = b_il if args.hole_filter_il_meters is None else args.hole_filter_il_meters
        h_xl = b_xl if args.hole_filter_xl_meters is None else args.hole_filter_xl_meters
        initial_params = (b_il, b_xl, h_il, h_xl)
    _, init_full, init_med = apply_mask_to_map(
        active_rms, outer_grid, hole_grid, *initial_params
    )
    view_state = {
        "mode": "blended", "blended": init_full, "median": init_med,
        "params": initial_params,
    }
    outcome = {"params": None, "edit": False}

    def refresh_qc_figure():
        b_il, b_xl, h_il, h_xl = view_state["params"]
        if view_state["mode"] == "blended":
            im_qc.set_data(view_state["blended"])
            ax2.set_title(
                f"{window_title_prefix}[BLENDED RMS OUTPUT | Median: {view_state['median']:.2f}]\n"
                f"Boundary: {b_il:.0f}m IL x {b_xl:.0f}m XL  |  Holes: {h_il:.0f}m IL x {h_xl:.0f}m XL  |  [SPACE/T]: Toggle  |  [E]: Edit polygon",
                fontsize=11.5, fontweight="bold", color="#0055ff", pad=10,
            )
        else:
            im_qc.set_data(padded_display)
            ax2.set_title(
                f"{window_title_prefix}[RAW RMS INPUT]\n"
                "[SPACE/T]: Toggle input/output  |  [E]: Edit polygon",
                fontsize=11.5, fontweight="bold", color="black", pad=10,
            )
        fig2.canvas.draw_idle()

    def toggle_qc_view():
        view_state["mode"] = "raw" if view_state["mode"] == "blended" else "blended"
        refresh_qc_figure()

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
            self.in_b_il = QtWidgets.QLineEdit(f"{initial_params[0]:.0f}")
            self.in_b_xl = QtWidgets.QLineEdit(f"{initial_params[1]:.0f}")
            self.in_h_il = QtWidgets.QLineEdit(f"{initial_params[2]:.0f}")
            self.in_h_xl = QtWidgets.QLineEdit(f"{initial_params[3]:.0f}")
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
            finish_label = ("Save & Apply to All Maps" if scope == "single" and len(rms_map_paths) > 1
                            else "Save & Finish")
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
            _, full_blend, median = apply_mask_to_map(
                active_rms, outer_grid, hole_grid, *values
            )
            view_state.update(params=values, blended=full_blend, median=median, mode="blended")
            refresh_qc_figure()

        def edit_mask(self):
            outcome["edit"] = True
            values = self.parse_fields()
            if values is not None:
                view_state["params"] = values
            self.close()
            plt.close(fig2)

        def save_and_exit(self):
            values = self.parse_fields()
            if values is None:
                return
            outcome["params"] = values
            self.close()
            plt.close(fig2)

    def on_qc_key(event):
        if event.key in (" ", "t", "T"):
            toggle_qc_view()
        elif event.key in ("e", "E"):
            control_box.edit_mask()

    fig2.canvas.mpl_connect("key_press_event", on_qc_key)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    control_box = RampControlWindow()
    control_box.show()
    refresh_qc_figure()
    plt.show()
    control_box.close()
    return outcome["params"], outcome["edit"], view_state["params"]


def run_masking_session(active_map_path, window_title_prefix="", initial_polygon=None):
    active_rms = np.load(active_map_path)
    if active_rms.shape != survey_shape:
        raise ValueError(f"{active_map_path.name} shape does not match coordinate axes")
    padded_display, p99 = pad_map(active_rms)
    outer_guess = initial_polygon
    existing_holes = None  # A new RMS window never inherits the previous holes.
    ramp_guess = None
    while True:
        selected_polygon, hole_polygons = pick_mask_polygons(
            padded_display, p99, window_title_prefix, outer_guess, existing_holes,
        )
        if selected_polygon is None:
            return None, None, None, None
        outer_grid = Path(selected_polygon).contains_points(grid_points).reshape(padded_display.shape)
        hole_grid = np.zeros(padded_display.shape, dtype=bool)
        for hole in hole_polygons:
            hole_grid |= Path(hole).contains_points(grid_points, radius=1e-9).reshape(padded_display.shape)
        params, edit_requested, ramp_guess = review_mask_ramps(
            active_rms, padded_display, p99, outer_grid, hole_grid,
            window_title_prefix, ramp_guess,
        )
        if edit_requested:
            outer_guess = selected_polygon
            existing_holes = hole_polygons
            continue
        return outer_grid, hole_grid, params, selected_polygon

# -------------------------------------------------------------
# 4. Main Batch Orchestration
# -------------------------------------------------------------
def main():
    global scope
    if matplotlib.get_backend().lower() in {
        "agg", "pdf", "ps", "svg", "cairo", "template"
    }:
        raise RuntimeError(
            "Interactive masking needs a graphical Matplotlib backend. "
            "Run with MPLBACKEND=QtAgg on a graphical desktop."
        )
    if scope is None:
        if len(rms_map_paths) == 1:
            scope = "single"
        else:
            while True:
                try:
                    answer = input(
                        "Use a [s]ingle mask for all RMS maps or [m]ultiple masks "
                        "(review each map)? [single/multiple] (default: single): "
                    ).strip().lower()
                except EOFError:
                    raise RuntimeError(
                        "Mask recalculation needs a scope choice; pass --scope "
                        "single or multiple for an unattended run"
                    ) from None
                if answer in ("", "s", "single", "1"):
                    scope = "single"
                    break
                if answer in ("m", "multiple", "2"):
                    scope = "multiple"
                    break
                print("Please enter 'single' or 'multiple'.")
    print(f"\nMasking Scope: {scope.upper()} ({len(rms_map_paths)} map{'s' if len(rms_map_paths) > 1 else ''})")

    if scope == "single":
        # Process ONCE using the preview map, then batch-blend all maps
        preview_map_path = rms_map_paths[args.preview_index]
        print(f"Opening interactive picking for shared mask using: {preview_map_path.name}")

        outer_grid, hole_grid, params, _ = run_masking_session(
            preview_map_path, window_title_prefix="[SHARED MASK] "
        )
        if params is None:
            raise RuntimeError("Masking cancelled before a blended RMS map was saved")

        b_il, b_xl, h_il, h_xl = params
        valid_mask = core_valid_mask(outer_grid, hole_grid)
        taper = coefficient_taper(outer_grid, hole_grid, b_il, b_xl, h_il, h_xl)[
            n_pad_il:n_pad_il + len(il_unique),
            n_pad_xl:n_pad_xl + len(xl_unique),
        ]
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
            save_mask_taper(out_path, taper)
            print(f" -> Saved blended RMS: {out_path.name}")

    else:
        # Step through each map sequentially
        print("\n" + "=" * 65)
        print("MULTIPLE MASKS: Custom boundaries & ramps per map")
        print("=" * 65)

        previous_polygon = None
        for idx, p in enumerate(rms_map_paths, 1):
            print(f"\n--- [Map {idx}/{len(rms_map_paths)}] {p.name} ---")
            outer_grid, hole_grid, params, previous_polygon = run_masking_session(
                p, window_title_prefix=f"[{idx}/{len(rms_map_paths)}: {p.name}] ",
                initial_polygon=previous_polygon,
            )
            if params is None:
                raise RuntimeError(f"Masking cancelled before saving {p.name}")

            b_il, b_xl, h_il, h_xl = params
            valid_mask = core_valid_mask(outer_grid, hole_grid)
            taper = coefficient_taper(outer_grid, hole_grid, b_il, b_xl, h_il, h_xl)[
                n_pad_il:n_pad_il + len(il_unique),
                n_pad_xl:n_pad_xl + len(xl_unique),
            ]
            cur_raw = np.load(p)
            blended, _, _ = apply_mask_to_map(cur_raw, outer_grid, hole_grid, b_il, b_xl, h_il, h_xl)

            out_path = p.with_name(p.stem + "_blended.npy")
            np.save(out_path, blended)
            save_valid_mask(out_path, valid_mask)
            save_mask_taper(out_path, taper)
            print(f" -> Saved blended RMS: {out_path.name}")

    print("\nAll blended RMS maps generated successfully.")

if __name__ == "__main__":
    main()
