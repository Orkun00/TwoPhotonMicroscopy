import tkinter as tk
from tkinter import filedialog, messagebox
from shapely.geometry import Point, Polygon
import numpy as np
import csv
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import threading


class ShapeGridGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Define Shape and Grid")

        # Fixed ROI in galvo index units (0..199)
        self.roi_size = 200
        self.galvo_step_um = 0.2  # physical step per index

        # Initial zoom/scale + pan offsets
        self.scale = 3
        self.pan_x = 0
        self.pan_y = 0
        self._drag_start = None

        canvas_size = self.roi_size * self.scale

        # Canvas
        self.canvas = tk.Canvas(master, width=canvas_size, height=canvas_size, bg="white")
        self.canvas.pack(side=tk.LEFT)

        self.vertices = []
        self.polygon_drawn = False
        self.grid_points = []
        self.big_jumps = []  # list of tuples: ((x1,y1),(x2,y2), dist_idx, dist_um)

        # Rectangle selection state
        self.rect_mode = False
        self.rect_start = None

        # ROI is full 200x200 grid
        self.roi = (0, 0, canvas_size, canvas_size)
        self.canvas.create_rectangle(*self.roi, outline="blue", dash=(4, 2), width=2)

        # Bind mouse events
        self.canvas.bind("<Button-1>", self.add_vertex)
        self.canvas.bind("<MouseWheel>", self.zoom)   # Windows/macOS
        self.canvas.bind("<Button-4>", self.zoom)     # Linux scroll up
        self.canvas.bind("<Button-5>", self.zoom)     # Linux scroll down

        # Pan with middle mouse or Shift+Left
        self.canvas.bind("<ButtonPress-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.do_pan)
        self.canvas.bind("<ButtonRelease-2>", self.end_pan)

        self.canvas.bind("<Shift-ButtonPress-1>", self.start_pan)
        self.canvas.bind("<Shift-B1-Motion>", self.do_pan)
        self.canvas.bind("<Shift-ButtonRelease-1>", self.end_pan)

        # Controls
        self.control_frame = tk.Frame(master)
        self.control_frame.pack(pady=5)

        self.close_button = tk.Button(self.control_frame, text="Close Polygon", command=self.close_polygon)
        self.close_button.grid(row=0, column=0, padx=5)

        self.grid_button = tk.Button(self.control_frame, text="Generate Grid", command=self.generate_grid)
        self.grid_button.grid(row=0, column=1, padx=5)

        self.export_button = tk.Button(self.control_frame, text="Export CSV", command=self.export_csv)
        self.export_button.grid(row=0, column=2, padx=5)

        self.rect_button = tk.Button(self.control_frame, text="Rectangle Mode", command=self.toggle_rectangle_mode)
        self.rect_button.grid(row=0, column=3, padx=5)

        self.refresh_button = tk.Button(self.control_frame, text="Refresh", command=self.refresh_canvas)
        self.refresh_button.grid(row=0, column=4, padx=5)

        self.reset_view_button = tk.Button(self.control_frame, text="Reset View", command=self.reset_view)
        self.reset_view_button.grid(row=0, column=5, padx=5)

        # Jump threshold UI (micrometers)
        tk.Label(self.control_frame, text="Jump Thresh (µm):").grid(row=0, column=6, padx=(20,4))
        self.jump_thresh_entry = tk.Entry(self.control_frame, width=6)
        self.jump_thresh_entry.insert(0, "0.4")  # default: 0.4 µm (2 index steps)
        self.jump_thresh_entry.grid(row=0, column=7, padx=4)

        # Simulate button
        self.sim_button = tk.Button(self.control_frame, text="Simulate Scan", command=self.simulate_scan)
        self.sim_button.grid(row=0, column=8, padx=5)

        # Heatmap storage
        self.heatmap_data = np.full((self.roi_size, self.roi_size), np.nan)

        # Matplotlib heatmap figure inside Tkinter
        fig = Figure(figsize=(4, 4))
        self.ax = fig.add_subplot(111)
        # Keep top-left as origin to match Tk canvas look
        self.im = self.ax.imshow(self.heatmap_data,
                                 origin="upper",
                                 cmap="inferno",
                                 interpolation="nearest")
        self.cbar = fig.colorbar(self.im, ax=self.ax, label="Signal Intensity")

        self.canvas_plot = FigureCanvasTkAgg(fig, master=master)
        self.canvas_plot.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # --------- Heatmap update / simulation ---------
    def update_heatmap(self, ix, iy, value):
        """Update heatmap with new scan value"""
        self.heatmap_data[iy, ix] = value  # matrix indexing: row=y, col=x
        if not hasattr(self, "clim_set"):
            self.im.set_clim(0, 100)  # adjust to expected PMT range
            self.clim_set = True
        self.im.set_data(self.heatmap_data)
        self.canvas_plot.draw()

    def simulate_scan(self):
        """Simulate scanning and insert extra delay on big jumps"""
        if not self.grid_points:
            print("No scan path. Generate grid first.")
            return

        try:
            jump_thresh_um = float(self.jump_thresh_entry.get())
        except ValueError:
            jump_thresh_um = 0.4

        jump_thresh_idx = jump_thresh_um / self.galvo_step_um

        def run():
            prev = None
            for pt in self.grid_points:
                if prev is not None:
                    dx = abs(pt[0] - prev[0])
                    dy = abs(pt[1] - prev[1])
                    dist_idx = (dx*dx + dy*dy) ** 0.5
                    # Delay policy: longer wait for big jumps
                    if dist_idx > jump_thresh_idx:
                        time.sleep(0.05)  # big jump delay
                    else:
                        time.sleep(0.01)  # normal dwell
                # fake signal
                val = np.random.randint(10, 100)
                self.update_heatmap(pt[0], pt[1], val)
                prev = pt

        threading.Thread(target=run, daemon=True).start()

    # --------- Coord transforms, zoom & pan ---------
    def px_to_idx(self, x, y):
        return int(round((x - self.pan_x) / self.scale)), int(round((y - self.pan_y) / self.scale))

    def idx_to_px(self, ix, iy):
        return ix * self.scale + self.pan_x, iy * self.scale + self.pan_y

    def inside_roi(self, x, y):
        ix, iy = self.px_to_idx(x, y)
        return 0 <= ix < self.roi_size and 0 <= iy < self.roi_size

    def zoom(self, event):
        old_scale = self.scale
        if event.num == 5 or event.delta < 0:
            self.scale /= 1.2
        elif event.num == 4 or event.delta > 0:
            self.scale *= 1.2
        self.scale = max(1, min(self.scale, 100))
        mouse_x, mouse_y = event.x, event.y
        factor = self.scale / old_scale
        self.pan_x = mouse_x - factor * (mouse_x - self.pan_x)
        self.pan_y = mouse_y - factor * (mouse_y - self.pan_y)
        self.redraw()

    def start_pan(self, event):
        self._drag_start = (event.x, event.y)

    def do_pan(self, event):
        if self._drag_start:
            dx = event.x - self._drag_start[0]
            dy = event.y - self._drag_start[1]
            self.pan_x += dx
            self.pan_y += dy
            self._drag_start = (event.x, event.y)
            self.redraw()

    def end_pan(self, event):
        self._drag_start = None

    def reset_view(self):
        self.scale = 3
        self.pan_x = 0
        self.pan_y = 0
        self.redraw()

    # --------- Drawing ---------
    def redraw(self):
        self.canvas.delete("all")
        # ROI
        px1, py1 = self.idx_to_px(0, 0)
        px2, py2 = self.idx_to_px(self.roi_size, self.roi_size)
        self.canvas.create_rectangle(px1, py1, px2, py2, outline="blue", dash=(4, 2), width=2)

        # Polygon
        if self.vertices:
            coords = [self.idx_to_px(ix, iy) for ix, iy in self.vertices]
            for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
                self.canvas.create_line(x1, y1, x2, y2, tags="shape")
            if self.polygon_drawn:
                self.canvas.create_line(coords[-1][0], coords[-1][1], coords[0][0], coords[0][1], tags="shape")
            for (px, py) in coords:
                self.canvas.create_oval(px-3, py-3, px+3, py+3, fill="black", tags="shape")

        # Spots
        if self.grid_points:
            spot_radius_idx = (0.5 / 2) / self.galvo_step_um
            spot_radius_px = spot_radius_idx * self.scale
            for ix, iy in self.grid_points:
                px, py = self.idx_to_px(ix, iy)
                self.canvas.create_oval(px - spot_radius_px, py - spot_radius_px,
                                        px + spot_radius_px, py + spot_radius_px,
                                        outline="red", tags="gridpoint")

        # Draw jump segments (orange)
        if self.big_jumps:
            for (a, b, _, _) in self.big_jumps:
                ax, ay = self.idx_to_px(*a)
                bx, by = self.idx_to_px(*b)
                self.canvas.create_line(ax, ay, bx, by, fill="orange", width=2, dash=(4, 3), tags="jumpline")

    # --------- Polygon editing ---------
    def toggle_rectangle_mode(self):
        self.rect_mode = not self.rect_mode
        self.rect_start = None
        if self.rect_mode:
            self.rect_button.config(relief=tk.SUNKEN, text="Rectangle Mode (ON)")
            self.vertices.clear()
            self.canvas.delete("shape")
            self.polygon_drawn = False
        else:
            self.rect_button.config(relief=tk.RAISED, text="Rectangle Mode (OFF)")

    def add_vertex(self, event):
        if self.polygon_drawn:
            return
        if not self.inside_roi(event.x, event.y):
            messagebox.showwarning("Invalid Point", "Click inside the blue box!")
            return

        ix, iy = self.px_to_idx(event.x, event.y)
        if self.rect_mode:
            if self.rect_start is None:
                self.rect_start = (ix, iy)
                px, py = self.idx_to_px(ix, iy)
                self.canvas.create_oval(px-3, py-3, px+3, py+3, fill="black", tags="shape")
            else:
                x1, y1 = self.rect_start
                x2, y2 = ix, iy
                self.vertices = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                px1, py1 = self.idx_to_px(x1, y1)
                px2, py2 = self.idx_to_px(x2, y2)
                self.canvas.create_rectangle(px1, py1, px2, py2, outline="black", width=2, tags="shape")
                self.polygon_drawn = True
                self.rect_start = None
        else:
            self.vertices.append((ix, iy))
            px, py = self.idx_to_px(ix, iy)
            self.canvas.create_oval(px-3, py-3, px+3, py+3, fill="black", tags="shape")
            if len(self.vertices) > 1:
                px2, py2 = self.idx_to_px(*self.vertices[-2])
                self.canvas.create_line(px2, py2, px, py, tags="shape")

    def close_polygon(self):
        if len(self.vertices) > 2 and not self.rect_mode:
            px1, py1 = self.idx_to_px(*self.vertices[-1])
            px2, py2 = self.idx_to_px(*self.vertices[0])
            self.canvas.create_line(px1, py1, px2, py2, tags="shape")
            self.polygon_drawn = True

    # --------- Grid generation (axis-priority nearest-neighbor) + jump logging ---------
    def generate_grid(self):
        if not self.polygon_drawn:
            return

        # clear old
        self.grid_points.clear()
        self.big_jumps.clear()
        self.canvas.delete("gridpoint")
        self.canvas.delete("jumpline")

        # Build polygon + bounds
        poly = Polygon(self.vertices)
        minx, miny, maxx, maxy = poly.bounds
        minx = max(minx, 0)
        miny = max(miny, 0)
        maxx = min(maxx, self.roi_size - 1)
        maxy = min(maxy, self.roi_size - 1)

        # Gather candidate points
        candidates = []
        for iy in range(int(miny), int(maxy) + 1):
            for ix in range(int(minx), int(maxx) + 1):
                p = Point(ix, iy)
                if poly.contains(p) or poly.buffer(1).contains(p):
                    candidates.append((ix, iy))

        if not candidates:
            print("No valid scan points found.")
            return

        # Order: axis-priority greedy nearest neighbor
        path = []
        current = candidates.pop(0)
        path.append(current)

        def axis_neighbors(curr, pool):
            cx, cy = curr
            out = []
            for px, py in pool:
                if (abs(px - cx) == 1 and py == cy) or (abs(py - cy) == 1 and px == cx):
                    out.append((px, py))
            return out

        while candidates:
            # prefer axis-aligned neighbor
            axn = axis_neighbors(current, candidates)
            if axn:
                next_pt = axn[0]
            else:
                # fallback: Euclidean nearest
                next_pt = min(candidates, key=lambda q: (q[0]-current[0])**2 + (q[1]-current[1])**2)
            path.append(next_pt)
            candidates.remove(next_pt)
            current = next_pt

        self.grid_points = path

        # Draw spots
        spot_radius_idx = (0.5 / 2) / self.galvo_step_um
        spot_radius_px = spot_radius_idx * self.scale
        for ix, iy in self.grid_points:
            px, py = self.idx_to_px(ix, iy)
            self.canvas.create_oval(px - spot_radius_px, py - spot_radius_px,
                                    px + spot_radius_px, py + spot_radius_px,
                                    outline="red", tags="gridpoint")

        # Compute & draw big jumps
        try:
            jump_thresh_um = float(self.jump_thresh_entry.get())
        except ValueError:
            jump_thresh_um = 0.4
        jump_thresh_idx = jump_thresh_um / self.galvo_step_um

        for (a, b) in zip(self.grid_points, self.grid_points[1:]):
            dx = abs(b[0] - a[0])
            dy = abs(b[1] - a[1])
            dist_idx = (dx*dx + dy*dy) ** 0.5
            if dist_idx > jump_thresh_idx:
                dist_um = dist_idx * self.galvo_step_um
                self.big_jumps.append((a, b, dist_idx, dist_um))
                ax, ay = self.idx_to_px(*a)
                bx, by = self.idx_to_px(*b)
                self.canvas.create_line(ax, ay, bx, by, fill="orange", width=2, dash=(4, 3), tags="jumpline")

        print(f"Generated {len(self.grid_points)} scan points (axis-priority NN).")
        print(f"Big jumps over {jump_thresh_um} µm: {len(self.big_jumps)}")
        for i, (a, b, di, du) in enumerate(self.big_jumps[:50], 1):
            print(f"  {i}. {a} -> {b}  jump = {du:.3f} µm ({di:.2f} idx)")
        if len(self.big_jumps) > 50:
            print(f"  ... and {len(self.big_jumps) - 50} more")

    # --------- CSV export ---------
    def export_csv(self):
        if not self.grid_points:
            print("No grid to export!")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                 filetypes=[("CSV files", "*.csv")])
        if file_path:
            with open(file_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["X_index", "Y_index", "X_um", "Y_um"])
                for ix, iy in self.grid_points:
                    x_um = ix * self.galvo_step_um
                    y_um = iy * self.galvo_step_um
                    writer.writerow([ix, iy, round(x_um, 3), round(y_um, 3)])
            print(f"Exported {len(self.grid_points)} points to {file_path}")

    def refresh_canvas(self):
        self.canvas.delete("shape")
        self.canvas.delete("gridpoint")
        self.canvas.delete("jumpline")
        self.vertices.clear()
        self.grid_points.clear()
        self.big_jumps.clear()
        self.polygon_drawn = False
        self.rect_start = None
        print("Canvas reset. Ready for new shape.")


if __name__ == "__main__":
    root = tk.Tk()
    app = ShapeGridGUI(root)
    root.mainloop()
