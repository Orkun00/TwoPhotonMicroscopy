import sys
import time
import threading
import numpy as np
import csv
from shapely.geometry import Point, Polygon
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets


# ---- Compatibility helpers (Qt5 vs Qt6) ----
try:
    DashLine = QtCore.Qt.DashLine
    LeftButton = QtCore.Qt.LeftButton
except AttributeError:  # Qt6+
    DashLine = QtCore.Qt.PenStyle.DashLine
    LeftButton = QtCore.Qt.MouseButton.LeftButton


class ShapeGridGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Define Shape and Grid (PyQtGraph)")

        # --- Parameters ---
        self.roi_size = 200
        self.galvo_step_um = 0.2
        self.vertices = []
        self.polygon_drawn = False
        self.rect_mode = False
        self.rect_start = None
        self.grid_points = []
        self.big_jumps = []

        # Heatmap
        self.heatmap_data = np.full((self.roi_size, self.roi_size), np.nan)

        # --- Layout ---
        layout = QtWidgets.QHBoxLayout(self)

        # Left: graphics for ROI
        self.view = pg.GraphicsLayoutWidget()
        layout.addWidget(self.view, 2)

        self.plot = self.view.addPlot()
        self.plot.setAspectLocked(True)
        self.plot.setXRange(0, self.roi_size)
        self.plot.setYRange(0, self.roi_size)
        self.plot.invertY(True)

        # ROI box
        roi_box = QtWidgets.QGraphicsRectItem(0, 0, self.roi_size, self.roi_size)
        roi_box.setPen(pg.mkPen('b', style=DashLine, width=2))
        self.plot.addItem(roi_box)

        # Polygon + grid
        self.polygon_item = pg.PlotDataItem([], [], pen=pg.mkPen('k', width=2),
                                            symbol='o', symbolBrush='k')
        self.plot.addItem(self.polygon_item)

        self.grid_scatter = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None),
                                               brush=pg.mkBrush(255, 0, 0, 120))
        self.plot.addItem(self.grid_scatter)

        self.jump_lines = []

        # Right: heatmap
        self.img_view = pg.ImageView()
        layout.addWidget(self.img_view, 3)

        # start with all NaN (so empty), but don’t show yet
        self.heatmap_data = np.full((self.roi_size, self.roi_size), np.nan)

        # Controls
        ctrl_layout = QtWidgets.QVBoxLayout()
        layout.addLayout(ctrl_layout, 1)

        self.close_btn = QtWidgets.QPushButton("Close Polygon")
        self.close_btn.clicked.connect(self.close_polygon)
        ctrl_layout.addWidget(self.close_btn)

        self.grid_btn = QtWidgets.QPushButton("Generate Grid")
        self.grid_btn.clicked.connect(self.generate_grid)
        ctrl_layout.addWidget(self.grid_btn)

        self.export_btn = QtWidgets.QPushButton("Export CSV")
        self.export_btn.clicked.connect(self.export_csv)
        ctrl_layout.addWidget(self.export_btn)

        self.rect_btn = QtWidgets.QPushButton("Rectangle Mode (OFF)")
        self.rect_btn.setCheckable(True)
        self.rect_btn.clicked.connect(self.toggle_rectangle_mode)
        ctrl_layout.addWidget(self.rect_btn)

        self.refresh_btn = QtWidgets.QPushButton("Refresh Canvas")
        self.refresh_btn.clicked.connect(self.refresh_canvas)
        ctrl_layout.addWidget(self.refresh_btn)

        self.jump_thresh_label = QtWidgets.QLabel("Jump Thresh (µm):")
        ctrl_layout.addWidget(self.jump_thresh_label)
        self.jump_thresh_entry = QtWidgets.QLineEdit("0.4")
        ctrl_layout.addWidget(self.jump_thresh_entry)

        self.sim_btn = QtWidgets.QPushButton("Simulate Scan")
        self.sim_btn.clicked.connect(self.simulate_scan)
        ctrl_layout.addWidget(self.sim_btn)

        ctrl_layout.addStretch()

        # Mouse clicks
        self.plot.scene().sigMouseClicked.connect(self.add_vertex)

    # --- Polygon ---
    def toggle_rectangle_mode(self):
        self.rect_mode = not self.rect_mode
        self.rect_start = None
        if self.rect_mode:
            self.rect_btn.setText("Rectangle Mode (ON)")
            self.vertices.clear()
            self.polygon_item.setData([], [])
            self.polygon_drawn = False
        else:
            self.rect_btn.setText("Rectangle Mode (OFF)")

    def add_vertex(self, event):
        if event.button() != LeftButton:
            return
        pos = event.scenePos()
        if self.plot.sceneBoundingRect().contains(pos):
            mouse_point = self.plot.vb.mapSceneToView(pos)
            x, y = int(round(mouse_point.x())), int(round(mouse_point.y()))
            if not (0 <= x < self.roi_size and 0 <= y < self.roi_size):
                return
            if self.polygon_drawn:
                return

            if self.rect_mode:
                if self.rect_start is None:
                    self.rect_start = (x, y)
                else:
                    x1, y1 = self.rect_start
                    x2, y2 = x, y
                    self.vertices = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                    xs, ys = zip(*self.vertices + [self.vertices[0]])
                    self.polygon_item.setData(xs, ys, symbol='o',
                                              pen=pg.mkPen('k', width=2))
                    self.polygon_drawn = True
                    self.rect_start = None
            else:
                self.vertices.append((x, y))
                xs, ys = zip(*self.vertices)
                self.polygon_item.setData(xs, ys, symbol='o',
                                          pen=pg.mkPen('k', width=2))

    def close_polygon(self):
        if len(self.vertices) > 2 and not self.rect_mode:
            xs, ys = zip(*self.vertices + [self.vertices[0]])
            self.polygon_item.setData(xs, ys, symbol='o',
                                      pen=pg.mkPen('k', width=2))
            self.polygon_drawn = True

    # --- Grid generation ---
    def generate_grid(self):
        if not self.polygon_drawn:
            return
        self.grid_points.clear()
        self.big_jumps.clear()
        for line in self.jump_lines:
            self.plot.removeItem(line)
        self.jump_lines.clear()

        poly = Polygon(self.vertices)
        minx, miny, maxx, maxy = poly.bounds
        candidates = []
        for iy in range(int(miny), int(maxy) + 1):
            for ix in range(int(minx), int(maxx) + 1):
                if poly.contains(Point(ix, iy)) or poly.buffer(1).contains(Point(ix, iy)):
                    candidates.append((ix, iy))

        if not candidates:
            print("No valid scan points found.")
            return

        # Axis-priority nearest neighbor
        path = []
        current = candidates.pop(0)
        path.append(current)

        def axis_neighbors(curr, pool):
            cx, cy = curr
            return [(px, py) for px, py in pool
                    if (abs(px - cx) == 1 and py == cy) or (abs(py - cy) == 1 and px == cx)]

        while candidates:
            axn = axis_neighbors(current, candidates)
            if axn:
                next_pt = axn[0]
            else:
                next_pt = min(candidates,
                              key=lambda q: (q[0] - current[0])**2 + (q[1] - current[1])**2)
            path.append(next_pt)
            candidates.remove(next_pt)
            current = next_pt

        self.grid_points = path
        xs, ys = zip(*self.grid_points)
        self.grid_scatter.setData(xs, ys)

        # Big jumps
        try:
            jump_thresh_um = float(self.jump_thresh_entry.text())
        except ValueError:
            jump_thresh_um = 0.4
        jump_thresh_idx = jump_thresh_um / self.galvo_step_um

        for (a, b) in zip(self.grid_points, self.grid_points[1:]):
            dx, dy = b[0] - a[0], b[1] - a[1]
            dist_idx = (dx*dx + dy*dy)**0.5
            if dist_idx > jump_thresh_idx:
                dist_um = dist_idx * self.galvo_step_um
                line = pg.PlotDataItem([a[0], b[0]], [a[1], b[1]],
                                       pen=pg.mkPen('orange', width=2, style=DashLine))
                self.plot.addItem(line)
                self.jump_lines.append(line)
                self.big_jumps.append((a, b, dist_idx, dist_um))

        print(f"Generated {len(self.grid_points)} scan points.")
        print(f"Big jumps over {jump_thresh_um} µm: {len(self.big_jumps)}")
        for i, (a, b, di, du) in enumerate(self.big_jumps[:20], 1):
            print(f"  {i}. {a} -> {b}  jump = {du:.3f} µm ({di:.2f} idx)")

    # --- CSV export ---
    def export_csv(self):
        if not self.grid_points:
            print("No grid to export.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save CSV", filter="CSV Files (*.csv)")
        if path:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["X_index", "Y_index", "X_um", "Y_um"])
                for ix, iy in self.grid_points:
                    x_um, y_um = ix * self.galvo_step_um, iy * self.galvo_step_um
                    writer.writerow([ix, iy, round(x_um, 3), round(y_um, 3)])
            print(f"Exported {len(self.grid_points)} points to {path}")

    # --- Heatmap ---
    def update_heatmap(self, ix, iy, value):
        self.heatmap_data[iy, ix] = value

        safe_data = np.nan_to_num(self.heatmap_data, nan=0.0)

        corrected = np.flipud(np.rot90(safe_data, 1))

        # Transpose, then flip left-right and top-bottom

        self.img_view.setImage(corrected, autoLevels=False, autoRange=False)

    def simulate_scan(self):
        if not self.grid_points:
            print("No scan path.")
            return

        try:
            jump_thresh_um = float(self.jump_thresh_entry.text())
        except ValueError:
            jump_thresh_um = 0.4
        jump_thresh_idx = jump_thresh_um / self.galvo_step_um

        def run():
            prev = None
            for pt in self.grid_points:
                if prev is not None:
                    dx, dy = pt[0] - prev[0], pt[1] - prev[1]
                    dist_idx = (dx*dx + dy*dy)**0.5
                    time.sleep(0.05 if dist_idx > jump_thresh_idx else 0.01)
                val = np.random.randint(10, 100)
                self.update_heatmap(pt[0], pt[1], val)
                prev = pt

        threading.Thread(target=run, daemon=True).start()

    # --- Reset ---
    def refresh_canvas(self):
        self.vertices.clear()
        self.polygon_item.setData([], [])
        self.polygon_drawn = False
        self.rect_start = None
        self.grid_points.clear()
        self.grid_scatter.setData([], [])
        for line in self.jump_lines:
            self.plot.removeItem(line)
        self.jump_lines.clear()
        self.big_jumps.clear()
        print("Canvas reset. Ready for new shape.")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = ShapeGridGUI()
    gui.resize(1400, 700)
    gui.show()
    sys.exit(app.exec())
