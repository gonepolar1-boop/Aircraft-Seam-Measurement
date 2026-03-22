import os
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from visible_app.measurement_pipeline import MeasurementConfig, export_measurement_result, run_measurement_once


class FigurePanel(ttk.Frame):
    def __init__(self, master, title: str):
        super().__init__(master)
        self.figure = Figure(figsize=(6, 5), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_title(title)
        self.axes.axis("off")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def draw_image(self, image, cmap=None, title=None):
        self.axes.clear()
        self.axes.set_title(title or "")
        if image is None:
            self.axes.text(0.5, 0.5, "No Data", ha="center", va="center")
        else:
            self.axes.imshow(image, cmap=cmap)
        self.axes.axis("off")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def draw_scatter3d(self, points_a, points_b, title):
        self.figure.clf()
        axes = self.figure.add_subplot(111, projection="3d")
        axes.set_title(title)
        if points_a is not None and len(points_a):
            axes.scatter(points_a[:, 0], points_a[:, 1], points_a[:, 2], s=4, c="#0EA5E9", label="centerline")
        if points_b is not None and len(points_b):
            axes.scatter(points_b[:, 0], points_b[:, 1], points_b[:, 2], s=4, c="#F97316", label="edges")
        axes.set_xlabel("X")
        axes.set_ylabel("Y")
        axes.set_zlabel("Z")
        if (points_a is not None and len(points_a)) or (points_b is not None and len(points_b)):
            axes.legend(loc="upper right")
        self.figure.tight_layout()
        self.canvas.draw_idle()


class SeamMeasurementApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Aircraft Seam Measurement GUI")
        self.root.geometry("1440x920")

        self.result = None
        self.status_var = tk.StringVar(value="Ready")
        self.image_path_var = tk.StringVar()
        self.point_map_path_var = tk.StringVar()
        self.checkpoint_path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar(value=os.path.join(os.getcwd(), "visible_app", "outputs"))
        self.mask_path_var = tk.StringVar()
        self.image_view_mode_var = tk.StringVar(value="overlay")

        self.seg_threshold_var = tk.DoubleVar(value=0.8)
        self.mask_threshold_var = tk.IntVar(value=127)
        self.min_width_var = tk.IntVar(value=1)
        self.kernel_size_var = tk.IntVar(value=3)
        self.keep_largest_var = tk.BooleanVar(value=True)
        self.side_sample_count_var = tk.IntVar(value=5)
        self.sample_step_var = tk.DoubleVar(value=1.0)
        self.gap_limit_var = tk.DoubleVar(value=2.0)
        self.flush_limit_var = tk.DoubleVar(value=1.0)

        self.image_name_var = tk.StringVar(value="Image: -")
        self.model_name_var = tk.StringVar(value="Model: -")
        self.gap_mean_var = tk.StringVar(value="Gap Mean: -")
        self.flush_mean_var = tk.StringVar(value="Flush Mean: -")
        self.judge_var = tk.StringVar(value="Status: -")

        self._build_layout()

    def _build_layout(self):
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=1)

        self._build_left_panel(container)
        self._build_right_panel(container)
        self._build_bottom_bar(container)

    def _build_left_panel(self, parent):
        left = ttk.Frame(parent, padding=(0, 0, 12, 0))
        left.grid(row=0, column=0, sticky="nsw")

        for title, builder in (
            ("Data", self._build_data_section),
            ("Run", self._build_run_section),
            ("Parameters", self._build_param_section),
            ("Export", self._build_export_section),
        ):
            section = ttk.LabelFrame(left, text=title, padding=10)
            section.pack(fill=tk.X, pady=(0, 10))
            builder(section)

    def _add_path_row(self, parent, label, variable, command):
        ttk.Label(parent, text=label).pack(anchor="w")
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=(2, 8))
        ttk.Entry(row, textvariable=variable, width=36).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="Browse", command=command, width=8).pack(side=tk.LEFT, padx=(6, 0))

    def _build_data_section(self, parent):
        self._add_path_row(parent, "2D Image", self.image_path_var, self._choose_image)
        self._add_path_row(parent, "Point Map", self.point_map_path_var, self._choose_point_map)
        self._add_path_row(parent, "Model Checkpoint", self.checkpoint_path_var, self._choose_checkpoint)
        self._add_path_row(parent, "Existing Mask (Optional)", self.mask_path_var, self._choose_mask)
        self._add_path_row(parent, "Output Directory", self.output_dir_var, self._choose_output_dir)

    def _build_run_section(self, parent):
        ttk.Label(parent, text="Display Mode").pack(anchor="w")
        view_box = ttk.Combobox(
            parent,
            textvariable=self.image_view_mode_var,
            values=("original", "mask", "overlay"),
            state="readonly",
        )
        view_box.pack(fill=tk.X, pady=(2, 8))
        view_box.bind("<<ComboboxSelected>>", lambda _event: self._draw_segmentation_view())

        for text, command in (
            ("Load Preview", self.load_preview),
            ("Run Segmentation", self.run_segmentation_only),
            ("Extract 2D Geometry", self.run_geometry_only),
            ("Run 3D Measurement", self.run_measurement_only),
            ("Run Full Pipeline", self.run_full_pipeline),
        ):
            ttk.Button(parent, text=text, command=command).pack(fill=tk.X, pady=3)

    def _build_param_section(self, parent):
        fields = [
            ("Segmentation Threshold", self.seg_threshold_var),
            ("Mask Threshold", self.mask_threshold_var),
            ("Min Width", self.min_width_var),
            ("Kernel Size", self.kernel_size_var),
            ("3D Side Sample Count", self.side_sample_count_var),
            ("3D Sample Step (px)", self.sample_step_var),
            ("Gap Limit", self.gap_limit_var),
            ("Flush Limit", self.flush_limit_var),
        ]
        for label, var in fields:
            ttk.Label(parent, text=label).pack(anchor="w")
            ttk.Entry(parent, textvariable=var).pack(fill=tk.X, pady=(2, 8))
        ttk.Checkbutton(parent, text="Keep Largest Component", variable=self.keep_largest_var).pack(anchor="w")

    def _build_export_section(self, parent):
        ttk.Button(parent, text="Export CSV / JSON / Report", command=self.export_results).pack(fill=tk.X)

    def _build_right_panel(self, parent):
        right = ttk.Frame(parent)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(right)
        notebook.grid(row=0, column=0, sticky="nsew")

        self.tab_image = FigurePanel(notebook, "Original / Mask / Overlay")
        self.tab_geometry = FigurePanel(notebook, "2D Geometry")
        self.tab_measurement = FigurePanel(notebook, "3D Measurement")
        self.tab_curve = FigurePanel(notebook, "Profiles and Summary")

        notebook.add(self.tab_image, text="Image")
        notebook.add(self.tab_geometry, text="2D Geometry")
        notebook.add(self.tab_measurement, text="3D Measurement")
        notebook.add(self.tab_curve, text="Curves")

        self.summary_table = ttk.Treeview(right, columns=("key", "value"), show="headings", height=8)
        self.summary_table.heading("key", text="Metric")
        self.summary_table.heading("value", text="Value")
        self.summary_table.column("key", width=220, anchor="w")
        self.summary_table.column("value", width=180, anchor="w")
        self.summary_table.grid(row=1, column=0, sticky="ew", pady=(10, 0))

    def _build_bottom_bar(self, parent):
        bar = ttk.Frame(parent, padding=(0, 8, 0, 0))
        bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        for index in range(6):
            bar.columnconfigure(index, weight=1)

        ttk.Label(bar, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        ttk.Label(bar, textvariable=self.image_name_var).grid(row=0, column=1, sticky="w")
        ttk.Label(bar, textvariable=self.model_name_var).grid(row=0, column=2, sticky="w")
        ttk.Label(bar, textvariable=self.gap_mean_var).grid(row=0, column=3, sticky="w")
        ttk.Label(bar, textvariable=self.flush_mean_var).grid(row=0, column=4, sticky="w")
        ttk.Label(bar, textvariable=self.judge_var).grid(row=0, column=5, sticky="w")

    def _choose_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
        if path:
            self.image_path_var.set(path)

    def _choose_point_map(self):
        path = filedialog.askopenfilename(filetypes=[("Point Map", "*.npy *.npz")])
        if path:
            self.point_map_path_var.set(path)

    def _choose_checkpoint(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch Checkpoint", "*.pth *.pt")])
        if path:
            self.checkpoint_path_var.set(path)

    def _choose_mask(self):
        path = filedialog.askopenfilename(filetypes=[("Mask", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
        if path:
            self.mask_path_var.set(path)

    def _choose_output_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.output_dir_var.set(path)

    def _build_config(self):
        return MeasurementConfig(
            segmentation_threshold=float(self.seg_threshold_var.get()),
            mask_threshold=int(self.mask_threshold_var.get()),
            min_width=int(self.min_width_var.get()),
            kernel_size=int(self.kernel_size_var.get()),
            keep_largest_component=bool(self.keep_largest_var.get()),
            side_sample_count=int(self.side_sample_count_var.get()),
            sample_step_px=float(self.sample_step_var.get()),
            gap_limit=float(self.gap_limit_var.get()),
            flush_limit=float(self.flush_limit_var.get()),
            checkpoint_path=self.checkpoint_path_var.get().strip(),
        )

    def _set_status(self, text):
        self.status_var.set(text)
        self.root.update_idletasks()

    def _validate_common_inputs(self):
        if not self.image_path_var.get().strip():
            raise ValueError("Please choose a 2D image first.")
        if not self.point_map_path_var.get().strip():
            raise ValueError("Please choose a point_map first.")
        if not self.mask_path_var.get().strip() and not self.checkpoint_path_var.get().strip():
            raise ValueError("Please provide a checkpoint or an existing mask.")

    def load_preview(self):
        try:
            if not self.image_path_var.get().strip():
                raise ValueError("Please choose a 2D image first.")
            image = cv2.imread(self.image_path_var.get(), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError("Unable to read the 2D image.")
            self.tab_image.draw_image(image, cmap="gray", title="Original")
            self.image_name_var.set(f"Image: {os.path.basename(self.image_path_var.get())}")
            self.model_name_var.set(
                f"Model: {os.path.basename(self.checkpoint_path_var.get()) if self.checkpoint_path_var.get() else 'Existing mask'}"
            )
            self._set_status("Preview loaded")
        except Exception as exc:
            self._handle_error(exc)

    def _run_pipeline(self):
        self._validate_common_inputs()
        self.result = run_measurement_once(
            image_path=self.image_path_var.get(),
            point_map_path=self.point_map_path_var.get(),
            checkpoint_path=self.checkpoint_path_var.get() or None,
            config=self._build_config(),
            mask_path=self.mask_path_var.get() or None,
        )

    def run_segmentation_only(self):
        try:
            self._set_status("Running segmentation...")
            self._run_pipeline()
            self._draw_segmentation_view()
            self._set_status("Segmentation completed")
        except Exception as exc:
            self._handle_error(exc)

    def run_geometry_only(self):
        try:
            self._set_status("Extracting 2D geometry...")
            self._run_pipeline()
            self._draw_segmentation_view()
            self._draw_geometry_view()
            self._set_status("2D geometry completed")
        except Exception as exc:
            self._handle_error(exc)

    def run_measurement_only(self):
        try:
            self._set_status("Running 3D measurement...")
            self._run_pipeline()
            self._draw_measurement_view()
            self._draw_curve_view()
            self._update_summary()
            self._set_status("3D measurement completed")
        except Exception as exc:
            self._handle_error(exc)

    def run_full_pipeline(self):
        try:
            self._set_status("Running full pipeline...")
            self._run_pipeline()
            self._draw_all_views()
            self._update_summary()
            self._set_status("Full pipeline completed")
        except Exception as exc:
            self._handle_error(exc)

    def export_results(self):
        try:
            if self.result is None:
                self.run_full_pipeline()
            output_dir = self.output_dir_var.get().strip()
            if not output_dir:
                raise ValueError("Please choose an output directory first.")
            exports = export_measurement_result(self.result, output_dir)
            self.result["exports"] = exports
            self._set_status(f"Exported results to {output_dir}")
            messagebox.showinfo("Export Completed", f"Saved output files to:\n{output_dir}")
        except Exception as exc:
            self._handle_error(exc)

    def _draw_all_views(self):
        self._draw_segmentation_view()
        self._draw_geometry_view()
        self._draw_measurement_view()
        self._draw_curve_view()

    def _draw_segmentation_view(self):
        if self.result is None:
            return
        mode = self.image_view_mode_var.get()
        if mode == "original":
            self.tab_image.draw_image(self.result["image"], cmap="gray", title="Original")
        elif mode == "mask":
            self.tab_image.draw_image(self.result["mask"], cmap="gray", title="Mask")
        else:
            overlay_rgb = cv2.cvtColor(self.result["overlay"], cv2.COLOR_BGR2RGB)
            self.tab_image.draw_image(overlay_rgb, title="Overlay")

    def _draw_geometry_view(self):
        if self.result is None:
            return
        geometry = self.result["geometry_2d"]
        overlay = cv2.cvtColor(self.result["overlay"], cv2.COLOR_BGR2RGB)
        self.tab_geometry.figure.clf()
        ax1 = self.tab_geometry.figure.add_subplot(121)
        ax2 = self.tab_geometry.figure.add_subplot(122)
        ax1.imshow(overlay)
        ax1.set_title("Edges / Centerline")
        ax1.axis("off")
        width_profile = geometry["width_profile"]
        if len(width_profile):
            ax2.plot(width_profile[:, 0], width_profile[:, 1], color="#0F766E")
        ax2.set_title("Width Profile")
        ax2.set_xlabel("row")
        ax2.set_ylabel("px")
        ax2.grid(True, alpha=0.3)
        self.tab_geometry.figure.tight_layout()
        self.tab_geometry.canvas.draw_idle()

    def _draw_measurement_view(self):
        if self.result is None:
            return
        geometry_3d = self.result["geometry_3d"]
        centerline = geometry_3d["centerline_3d"][geometry_3d["centerline_3d_valid_mask"]]
        left_edge = geometry_3d["left_edge_3d"][geometry_3d["left_edge_3d_valid_mask"]]
        right_edge = geometry_3d["right_edge_3d"][geometry_3d["right_edge_3d_valid_mask"]]
        edges = np.concatenate([left_edge, right_edge], axis=0) if len(left_edge) or len(right_edge) else None
        self.tab_measurement.draw_scatter3d(centerline, edges, "Seam 3D Points")

    def _draw_curve_view(self):
        if self.result is None:
            return
        gap_profile = self.result["measurement_3d"]["gap_profile"]
        flush_profile = self.result["measurement_3d"]["flush_profile"]
        self.tab_curve.figure.clf()
        ax1 = self.tab_curve.figure.add_subplot(211)
        ax2 = self.tab_curve.figure.add_subplot(212)
        if len(gap_profile):
            ax1.plot(gap_profile[:, 0], gap_profile[:, 1], color="#2563EB")
        ax1.set_title("Gap Profile")
        ax1.set_xlabel("row")
        ax1.set_ylabel("gap")
        ax1.grid(True, alpha=0.3)
        if len(flush_profile):
            ax2.plot(flush_profile[:, 0], flush_profile[:, 1], color="#DC2626")
        ax2.set_title("Flush Profile")
        ax2.set_xlabel("row")
        ax2.set_ylabel("flush")
        ax2.grid(True, alpha=0.3)
        self.tab_curve.figure.tight_layout()
        self.tab_curve.canvas.draw_idle()

    def _update_summary(self):
        if self.result is None:
            return
        for item in self.summary_table.get_children():
            self.summary_table.delete(item)

        geom_summary = self.result["geometry_2d"]["summary"]
        measure_summary = self.result["measurement_3d"]["summary"]
        quality = self.result["quality"]
        rows = [
            ("mean_width", geom_summary.get("mean_width")),
            ("std_width", geom_summary.get("std_width")),
            ("valid_rows", geom_summary.get("valid_rows")),
            ("mean_gap", measure_summary.get("mean_gap")),
            ("max_gap", measure_summary.get("max_gap")),
            ("mean_flush", measure_summary.get("mean_flush")),
            ("max_abs_flush", measure_summary.get("max_abs_flush")),
            ("quality", quality.get("status_text")),
        ]
        for key, value in rows:
            text = value
            if isinstance(value, float):
                text = "N/A" if not np.isfinite(value) else f"{value:.6f}"
            self.summary_table.insert("", tk.END, values=(key, text))

        self.image_name_var.set(f"Image: {os.path.basename(self.result['inputs']['image_path'])}")
        checkpoint_path = self.result["inputs"].get("checkpoint_path")
        self.model_name_var.set(f"Model: {os.path.basename(checkpoint_path) if checkpoint_path else 'Existing mask'}")

        mean_gap = measure_summary.get("mean_gap", np.nan)
        mean_flush = measure_summary.get("mean_flush", np.nan)
        self.gap_mean_var.set(f"Gap Mean: {'N/A' if not np.isfinite(mean_gap) else f'{mean_gap:.4f}'}")
        self.flush_mean_var.set(f"Flush Mean: {'N/A' if not np.isfinite(mean_flush) else f'{mean_flush:.4f}'}")
        self.judge_var.set(f"Status: {quality.get('status_text')}")

    def _handle_error(self, exc: Exception):
        self._set_status("Run failed")
        traceback_text = "".join(traceback.format_exception(exc))
        messagebox.showerror("Error", f"{exc}\n\n{traceback_text}")


def main():
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    SeamMeasurementApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
