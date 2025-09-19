import tkinter as tk
from tkinter import filedialog, messagebox
from shapely.geometry import Point, Polygon
import numpy as np
import csv
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


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
        self.canvas.pack()

        self.vertices = []
        self.polygon_drawn = False
        self.grid_points = []

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
        # Heatmap storage
        self.heatmap_data = np.full((self.roi_size, self.roi_size), np.nan)

        # Matplotlib heatmap figure inside Tkinter
        fig = Figure(figsize=(4, 4))
        self.ax = fig.add_subplot(111)
        self.im = self.ax.imshow(self.heatmap_data,
                                 origin="upper",
                                 cmap="inferno",
                                 interpolation="nearest")
        self.cbar = fig.colorbar(self.im, ax=self.ax, label="Signal Intensity")

        self.canvas_plot = FigureCanvasTkAgg(fig, master=master)
        self.canvas_plot.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.sim_button = tk.Button(self.control_frame, text="Simulate Scan", command=self.simulate_scan)
        self.sim_button.grid(row=0, column=6, padx=5)

    def update_heatmap(self, ix, iy, value):
        """Update heatmap with new scan value"""
        self.heatmap_data[iy, ix] = value  # note: [row, col] = [y, x]
        self.im.set_data(self.heatmap_data)
        self.im.autoscale()
        self.canvas_plot.draw()

    def simulate_scan(self):
        import time
        import threading
        def run():
            for ix, iy in self.grid_points:  # snake path order
                value = np.random.randint(10, 100)  # fake intensity
                self.update_heatmap(ix, iy, value)
                #time.sleep(0.01)  # simulate measurement time
        threading.Thread(target=run, daemon=True).start()

    def px_to_idx(self, x, y):
        """Convert canvas pixels → galvo indices (apply pan/zoom inverse)"""
        return int(round((x - self.pan_x) / self.scale)), int(round((y - self.pan_y) / self.scale))

    def idx_to_px(self, ix, iy):
        """Convert galvo indices → canvas pixels (apply pan/zoom)"""
        return ix * self.scale + self.pan_x, iy * self.scale + self.pan_y

    def inside_roi(self, x, y):
        """Check if a pixel click is inside the ROI box (with pan/zoom considered)"""
        ix, iy = self.px_to_idx(x, y)
        return 0 <= ix < self.roi_size and 0 <= iy < self.roi_size

    def zoom(self, event):
        """Zoom in/out with mouse wheel"""
        old_scale = self.scale
        if event.num == 5 or event.delta < 0:  # scroll down
            self.scale /= 1.2
        elif event.num == 4 or event.delta > 0:  # scroll up
            self.scale *= 1.2

        # Clamp zoom
        self.scale = max(1, min(self.scale, 100))

        # Adjust pan so zoom centers around mouse position
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
        """Reset zoom and pan"""
        self.scale = 3
        self.pan_x = 0
        self.pan_y = 0
        self.redraw()

    def redraw(self):
        self.canvas.delete("all")

        # Draw ROI box
        px1, py1 = self.idx_to_px(0, 0)
        px2, py2 = self.idx_to_px(self.roi_size, self.roi_size)
        self.canvas.create_rectangle(px1, py1, px2, py2, outline="blue", dash=(4, 2), width=2)

        # Draw polygon
        if self.vertices:
            coords = [self.idx_to_px(ix, iy) for ix, iy in self.vertices]
            for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
                self.canvas.create_line(x1, y1, x2, y2, tags="shape")
            if self.polygon_drawn:
                self.canvas.create_line(coords[-1][0], coords[-1][1], coords[0][0], coords[0][1], tags="shape")

            for (px, py) in coords:
                self.canvas.create_oval(px-3, py-3, px+3, py+3, fill="black", tags="shape")

        # Draw grid spots
        if self.grid_points:
            spot_radius_idx = (0.5 / 2) / self.galvo_step_um
            spot_radius_px = spot_radius_idx * self.scale
            for ix, iy in self.grid_points:
                px, py = self.idx_to_px(ix, iy)
                self.canvas.create_oval(px - spot_radius_px, py - spot_radius_px,
                                        px + spot_radius_px, py + spot_radius_px,
                                        outline="red", tags="gridpoint")

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
                # First corner
                self.rect_start = (ix, iy)
                px, py = self.idx_to_px(ix, iy)
                self.canvas.create_oval(px-3, py-3, px+3, py+3, fill="black", tags="shape")
            else:
                # Second corner → define rectangle
                x1, y1 = self.rect_start
                x2, y2 = ix, iy
                self.vertices = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                px1, py1 = self.idx_to_px(x1, y1)
                px2, py2 = self.idx_to_px(x2, y2)
                self.canvas.create_rectangle(px1, py1, px2, py2, outline="black", width=2, tags="shape")
                self.polygon_drawn = True
                self.rect_start = None
        else:
            # Normal polygon mode
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

    def generate_grid(self):
        if not self.polygon_drawn:
            return

        poly = Polygon(self.vertices)
        minx, miny, maxx, maxy = poly.bounds

        # Clamp to ROI
        minx = max(minx, 0)
        miny = max(miny, 0)
        maxx = min(maxx, self.roi_size - 1)
        maxy = min(maxy, self.roi_size - 1)

        self.grid_points.clear()
        self.canvas.delete("gridpoint")

        # Laser spot radius: 0.25 µm / 0.2 µm per step = 1.25 index units
        spot_radius_idx = (0.5 / 2) / self.galvo_step_um
        spot_radius_px = spot_radius_idx * self.scale

        # Generate grid points (indices)
        for iy in range(int(miny), int(maxy) + 1):
            row_points = []
            for ix in range(int(minx), int(maxx) + 1):
                p = Point(ix, iy)
                if poly.contains(p) or poly.buffer(1).contains(p):  # buffer=1 idx margin
                    row_points.append((ix, iy))

            # Alternate direction every row (snake scan)
            if iy % 2 == 1:
                row_points.reverse()

            for ix, iy in row_points:
                self.grid_points.append((ix, iy))
                px, py = self.idx_to_px(ix, iy)
                self.canvas.create_oval(px - spot_radius_px, py - spot_radius_px,
                                        px + spot_radius_px, py + spot_radius_px,
                                        outline="red", tags="gridpoint")

        print(f"Generated {len(self.grid_points)} scan points.")

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
        self.vertices.clear()
        self.grid_points.clear()
        self.polygon_drawn = False
        self.rect_start = None
        print("Canvas reset. Ready for new shape.")


if __name__ == "__main__":
    root = tk.Tk()
    app = ShapeGridGUI(root)
    root.mainloop()
