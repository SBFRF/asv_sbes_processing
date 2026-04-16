import os
import io
import h5py
import time
import threading
import numpy as np
import tkinter as tk
from PIL import Image
import tkinter.font as tkfont
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.colors import LogNorm
from tkinter import filedialog, messagebox
from matplotlib.widgets import LassoSelector
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

"""
bottomTracer is a GUI application for manually and semi-automatically annotating sonar profile
data stored in HDF5 files. It provides a visual interface to trace, edit, and save depth lines
(the seabed) based on sonar scans, supporting both automated factory lines and manual corrections.

See the associated README.md file for more.
"""


class bottomTracer:
    def __init__(self, root):
        """Handles functionality related to sonar data tracing and annotation."""
        self.root = root
        self.root.title("HDF5 Annotator")
        self.root.protocol("WM_DELETE_WINDOW", self.quit_gui)
        self.depth_option = tk.StringVar(value="Ping Depth")
        self.manual_line_saved = False
        self.edit_mode = False
        self.edited_line = None
        self.menu_frame = tk.Frame(root, borderwidth=2, relief="groove")
        self.menu_frame.pack(side="top", fill="x", padx=10, pady=10)
        for col in range(4):
            self.menu_frame.grid_columnconfigure(col, weight=1)
        tk.Label(self.menu_frame, text="Chunk Size:").grid(
            row=0, column=1, padx=0, pady=5
        )
        self.chunk_size_entry = tk.Entry(self.menu_frame, width=10, justify="center")
        self.chunk_size = 250
        self.chunk_size_entry.insert(0, str(self.chunk_size))
        self.chunk_size_entry.grid(row=0, column=2, padx=0, pady=5)
        tk.Label(self.menu_frame, text="Input File:").grid(
            row=1, column=1, padx=10, pady=5
        )
        self.input_file_path = tk.StringVar(value=os.getcwd())
        self.input_dir_entry = tk.Entry(
            self.menu_frame,
            textvariable=self.input_file_path,
            width=40,
            justify="center",
        )
        self.input_dir_entry.grid(row=1, column=2, padx=10, pady=5, sticky="w")
        self.browseInput_Button = tk.Button(
            self.menu_frame, text="Browse", command=self.choose_input_file
        )
        self.browseInput_Button.grid(row=1, column=3, padx=10, pady=5, sticky="w")
        self.load_button = tk.Button(
            self.menu_frame, text="Start Labeling", command=self.load_file, width=30
        )
        self.load_button.grid(row=3, column=0, padx=0, pady=5, columnspan=4)
        self.annotation_frame = tk.Frame(root)

        screen_width = root.winfo_screenwidth()  # Get screen dimensions
        screen_height = root.winfo_screenheight()
        max_width = int(
            screen_width * 0.75
        )  # Define margins (e.g., use 90% of available screen size)
        max_height = int(screen_height * 0.75)
        target_ratio = 5 / 3  # Desired aspect ratio
        if (
            max_width / target_ratio <= max_height
        ):  # Compute figure size while maintaining aspect ratio
            fig_width_px = max_width
            fig_height_px = int(max_width / target_ratio)
        else:
            fig_height_px = max_height
            fig_width_px = int(max_height * target_ratio)
        fig_width_px = fig_width_px - (fig_width_px % 2)  # Round to nearest even number
        fig_height_px = fig_height_px - (
            fig_height_px % 2
        )  # Round to nearest even number
        dpi = 100  # Convert pixels to inches (assuming 100 DPI)
        fig_width_in = fig_width_px / dpi
        fig_height_in = fig_height_px / dpi
        self.fig, self.ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=dpi)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.annotation_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)
        self.canvas_widget.bind("<Destroy>", self.on_canvas_destroy)
        self.jump_frame = tk.Frame(self.annotation_frame)
        self.jump_frame.pack(pady=5)
        tk.Label(self.jump_frame, text="Slice #").grid(row=0, column=0, padx=5)
        self.prev_button_jump = tk.Button(
            self.jump_frame, text="<<", command=self.prev_slice
        )
        self.prev_button_jump.grid(row=0, column=1, padx=5)
        self.jump_slice_var = tk.StringVar(value="1")
        self.jump_entry = tk.Entry(
            self.jump_frame, textvariable=self.jump_slice_var, width=5, justify="center"
        )
        self.jump_entry.grid(row=0, column=2, padx=5)
        self.next_button_jump = tk.Button(
            self.jump_frame, text=">>", command=self.next_slice
        )
        self.next_button_jump.grid(row=0, column=3, padx=5)
        self.jump_button = tk.Button(
            self.jump_frame, text="Jump To Slice", command=self.jump_to_slice
        )
        self.jump_button.grid(row=0, column=4, padx=5)
        self.y_axis_frame = tk.Frame(self.annotation_frame)
        self.y_axis_frame.pack(pady=5)
        tk.Label(self.y_axis_frame, text="Y Axis Limits:").pack(side="left", padx=5)
        self.ymin_entry = tk.Entry(self.y_axis_frame, width=10, justify="center")
        self.ymin_entry.pack(side="left", padx=5)
        tk.Label(self.y_axis_frame, text="-").pack(side="left", padx=5)
        self.ymax_entry = tk.Entry(self.y_axis_frame, width=10, justify="center")
        self.ymax_entry.pack(side="left", padx=5)
        self.y_update_button = tk.Button(
            self.y_axis_frame, text="Update", command=self.update_y_axis_limits
        )
        self.y_update_button.pack(side="left", padx=5)
        self.depth_frame = tk.Frame(self.annotation_frame)
        self.depth_frame.pack(pady=5)
        tk.Label(self.depth_frame, text="Toggle Factory Line:").grid(row=0, column=0)
        tk.Radiobutton(
            self.depth_frame,
            text="Ping Depth",
            variable=self.depth_option,
            value="Ping Depth",
            command=self.update_display,
        ).grid(row=0, column=1)
        tk.Radiobutton(
            self.depth_frame,
            text="Smooth Depth",
            variable=self.depth_option,
            value="Smooth Depth",
            command=self.update_display,
        ).grid(row=0, column=2)
        tk.Radiobutton(
            self.depth_frame,
            text="Off",
            variable=self.depth_option,
            value="Off",
            command=self.update_display,
        ).grid(row=0, column=3)
        self.nan_button = tk.Button(
            self.depth_frame, text="Omit Whole Slice", command=self.apply_traced_line
        )
        self.nan_button.grid(row=0, column=4, padx=(20, 0))
        self.button_frame = tk.Frame(self.annotation_frame)
        self.button_frame.pack(fill="both", expand=True)
        self.clear_button = tk.Button(
            self.button_frame, text="Clear Annotations", command=self.clear_annotations
        )
        self.clear_button.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        self.save_button = tk.Button(self.button_frame, text="", state="normal")
        self.save_button.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        self.save_depth_button = tk.Button(self.button_frame, text="", state="normal")
        self.save_depth_button.grid(row=0, column=2, padx=10, pady=5, sticky="ew")
        self.lasso_button = tk.Button(
            self.button_frame, text="", command=self.activate_lasso, state="disabled"
        )
        self.lasso_button.grid(row=0, column=3, padx=10, pady=5, sticky="ew")
        self.logscale_button = tk.Button(
            self.depth_frame, text="Toggle Log Scale", command=self.toggle_logscale
        )
        self.logscale_button.grid(row=0, column=5, padx=(20, 0))  # adjust as needed
        self.quit_button = tk.Button(
            self.button_frame, text="Quit", command=self.quit_gui
        )
        self.quit_button.grid(
            row=1, column=0, columnspan=4, padx=10, pady=5, sticky="ew"
        )
        for col in range(4):
            self.button_frame.grid_columnconfigure(col, weight=1)
        self.start_time = None
        self.base_name = None
        self.whole_record_file = None
        self.idx_start = 0
        self.data_blanking_distance_cm = 5
        self.image = None
        self.smooth_depth = None
        self.length_mm = None
        self.this_ping_depth_m = None
        self.bin_size = None
        self.smooth_depth_img = None
        self.this_ping_depth_img = None
        self.tracing = False
        self.last_x, self.last_y = (None, None)
        self.coordinates = []
        self.image_for_saving = None
        self.total_slices = None
        self.total_time = None
        self.applied_line = None
        self.slice_number = None
        self.slice_length = None
        self.ymin = None
        self.ymax = None
        self.lasso = None
        self.lasso_selected = np.zeros(0, dtype=bool)
        self.all_green_edits = None
        self.secondary_y = None
        self.use_log_scale = False

    def on_canvas_destroy(self, event):
        """Cleans up when the canvas is destroyed."""
        if hasattr(self, "_closing") and self._closing:
            return
        self._closing = True
        if self.root and self.root.winfo_exists():
            self.quit_gui()

    def choose_input_file(self):
        """Opens a file or directory selection dialog."""
        file_path = filedialog.askopenfilename(
            initialdir=os.getcwd(), filetypes=[("HDF5 files", "*.h5 *.hdf5")]
        )
        if file_path:
            self.input_file_path.set(file_path)

    def load_file(self):
        """Loads and initializes data or files for processing."""
        self.start_time = time.time()
        try:
            self.chunk_size = int(self.chunk_size_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Chunk size must be an integer.")
            return
        self.input_file_path = self.input_file_path.get()
        input_file = os.path.basename(self.input_file_path)
        self.base_name = os.path.splitext(input_file)[0]
        raw_dir = os.path.dirname(self.input_file_path)
        self.output_folder = raw_dir
        self.whole_record_file = os.path.join(
            raw_dir, f"{self.base_name}_bottomTraced_wholeRecord.h5"
        )
        with h5py.File(self.input_file_path, "a") as raw_h5:
            self.total_time = raw_h5["time"].shape[0]
            if "qaqc_depth_line" not in raw_h5:
                full_data = np.column_stack(
                    (np.arange(self.total_time), np.full(self.total_time, float(-999)))
                )
                raw_h5.create_dataset(
                    "qaqc_depth_line", data=full_data, maxshape=(self.total_time, 2)
                )
            self.menu_frame.pack_forget()
            self.annotation_frame.pack(fill="both", expand=True)
            self.idx_start = 0
            self.depth_option.set("Ping Depth")
            for child in self.depth_frame.winfo_children():
                child.config(state="normal")
            for child in self.y_axis_frame.winfo_children():
                child.config(state="normal")
            self.process_next_chunk()

        # Resize the main window to fit the figure canvas exactly
        canvas_width = int(self.fig.get_figwidth() * self.fig.dpi)
        canvas_height = int(self.fig.get_figheight() * self.fig.dpi)
        # Include extra margin for widgets if needed
        margin_w = 50  # depends on your layout
        margin_h = 250
        total_width = canvas_width + margin_w
        total_height = canvas_height + margin_h
        self.root.geometry(f"{total_width}x{total_height}")

    def process_next_chunk(self):
        """Handles functionality related to sonar data tracing and annotation."""
        if not self.input_file_path:
            return
        with h5py.File(self.input_file_path, "r") as f:
            self.total_slices = int(np.ceil(self.total_time / self.chunk_size))
            self.slice_number = self.idx_start // self.chunk_size + 1
            end_idx = min(self.idx_start + self.chunk_size, self.total_time)
            idx = slice(self.idx_start, end_idx)
            self.image = f["profile_data"][:, idx]
            if "smooth_depth_m" in f:
                self.smooth_depth = f["smooth_depth_m"][idx]
                if "length_mm" in f:
                    self.length_mm = f["length_mm"][idx]
                    self.bin_size = self.length_mm[0] / 1000.0 / self.image.shape[0]
                else:
                    self.length_mm = None
                    self.bin_size = None
                self.this_ping_depth_m = (
                    f["this_ping_depth_m"][idx] if "this_ping_depth_m" in f else None
                )
            else:
                self.smooth_depth = None
                self.length_mm = None
                self.this_ping_depth_m = None
                self.bin_size = None
        if self.bin_size is not None:
            self.smooth_depth_img = (
                self.smooth_depth / self.bin_size
                if self.smooth_depth is not None
                else None
            )
            self.this_ping_depth_img = (
                self.this_ping_depth_m / self.bin_size
                if self.this_ping_depth_m is not None
                else None
            )
            if self.smooth_depth_img is not None:
                self.smooth_depth_img[
                    self.smooth_depth < self.data_blanking_distance_cm / 100
                ] = np.nan
            if self.this_ping_depth_img is not None:
                self.this_ping_depth_img[
                    self.this_ping_depth_m < self.data_blanking_distance_cm / 100
                ] = np.nan
        else:
            self.smooth_depth_img = None
            self.this_ping_depth_img = None
        self.edit_mode = False
        self.update_display()
        self.edited_line = None
        self.all_green_edits = None

    def update_display(self):
        """Updates internal state or display elements."""
        self.ax.clear()
        if self.use_log_scale:
            norm = LogNorm(vmin=max(self.image.min(), 1e-3), vmax=self.image.max())
            self.ax.pcolormesh(self.image, cmap="plasma", norm=norm)
        else:
            self.ax.pcolormesh(self.image, cmap="plasma")
        self.slice_length = self.image.shape[1]
        self.ax.tick_params(axis="y", which="both", labelleft=True, labelright=True)
        self.ax.set_xlim(0, self.slice_length)
        self.ax.set_ylim(0, self.image.shape[0])
        self.sync_y_axis_entries()
        self.ax.set_xlabel("Ping Count", fontsize=15)
        self.ax.set_ylabel("Bin #", fontsize=15)
        self.ax.tick_params(axis="x", labelsize=14)
        self.ax.tick_params(axis="y", labelsize=14)
        if self.depth_option.get() != "Off" and (
            self.smooth_depth_img is not None or self.this_ping_depth_img is not None
        ):
            self.plot_depth()
        for txt in self.fig.texts[:]:
            txt.remove()
        idxS = self.idx_start
        idxE = min(self.idx_start + self.chunk_size - 1, self.total_time - 1)
        self.fig.text(
            0.01,
            0.98,
            f"Slice #{self.slice_number} of {self.total_slices}\nTime Indices: {idxS} - {idxE}",
            horizontalalignment="left",
            verticalalignment="top",
            fontsize=12,
            color="black",
        )
        self.add_secondary_y_axis()
        self.canvas.draw()
        self.coordinates = []
        self.tracing = False
        self.last_x, self.last_y = (None, None)
        self.enable_annotation()
        image_array = np.array(self.fig.canvas.renderer.buffer_rgba())
        self.image_for_saving = Image.fromarray(image_array)
        if self.depth_option.get() != "Off" and (not self.edit_mode):
            self.clear_button.grid_remove()
        self.update_button_states()
        self.jump_slice_var.set(str(self.slice_number))

    def add_secondary_y_axis(self):
        """Add a secondary Y axis showing depth in meters."""
        # Remove existing secondary Y axis if it exists
        if hasattr(self, "secondary_y") and self.secondary_y in self.fig.axes:
            self.fig.delaxes(self.secondary_y)
            self.secondary_y = None

        # Add new secondary Y axis if bin_size is valid
        if self.bin_size is not None:
            self.secondary_y = self.ax.twinx()
            self.secondary_y.set_ylim(
                self.ax.get_ylim()[0] * self.bin_size,
                self.ax.get_ylim()[1] * self.bin_size,
            )
            self.secondary_y.set_ylabel("Depth Range (m)", fontsize=15)
            self.secondary_y.tick_params(axis="y", labelsize=14)

    def update_button_states(self):
        """Updates internal state or display elements."""
        if self.depth_option.get() == "Off":
            self.clear_button.grid()
            self.clear_button.config(
                text="Clear Annotations", command=self.clear_annotations, state="normal"
            )
            self.save_button.config(
                text="Apply Traced Line", command=self.apply_traced_line, state="normal"
            )
            self.save_depth_button.config(
                text="Save Manual Depth Line", command=self.save_data, state="disabled"
            )
            self.lasso_button.config(text="", command=None, state="disabled")
            self.ax.set_title(
                "Manual Tracing Mode Enabled:\nLeft click to trace the depth line in green.",
                fontsize=16,
            )
        elif not self.edit_mode:
            self.clear_button.grid_remove()
            self.save_button.config(
                text=f"Save {self.depth_option.get()} Line",
                command=self.save_depth_line,
                state="normal",
            )
            self.save_depth_button.config(
                text=f"Edit {self.depth_option.get()} Line",
                command=self.enter_editing_mode,
                state="normal",
            )
            self.lasso_button.config(
                text=f"Clean {self.depth_option.get()} Line",
                command=self.activate_lasso,
                state="normal",
            )
        else:
            self.clear_button.grid()
            self.clear_button.config(
                text="Clear Annotations", command=self.clear_edit_mode, state="normal"
            )
            self.save_button.config(
                text="Apply Edits", command=self.apply_edits, state="normal"
            )
            self.save_depth_button.config(
                text="Save Edited Depth Line",
                command=self.save_edited_depth_line,
                state="disabled",
            )
            self.lasso_button.config(text="", command=None, state="disabled")
        self.next_button_jump.config(
            state="normal" if self.slice_number < self.total_slices else "disabled"
        )
        self.prev_button_jump.config(
            state="normal" if self.slice_number > 1 else "disabled"
        )
        self.canvas.draw()

    def update_y_axis_limits(self):
        """Updates internal state or display elements."""
        try:
            self.ymin = float(self.ymin_entry.get())
            self.ymax = float(self.ymax_entry.get())
        except ValueError:
            messagebox.showerror(
                "Invalid Input", "Please enter numeric values for Y Axis Limits."
            )
            return
        num_results = self.image.shape[0] if self.image is not None else 0
        if (
            self.ymin < 0
            or self.ymax < 0
            or self.ymin > num_results
            or (self.ymax > num_results)
        ):
            messagebox.showerror(
                "Invalid Input", f"Values must be between 0 and {num_results}."
            )
            return
        if self.ymin > self.ymax:
            messagebox.showerror(
                "Invalid Input", "Lower limit cannot be greater than upper limit."
            )
            return
        if self.ymax - self.ymin < 100:
            messagebox.showerror(
                "Invalid Input",
                "The difference between the limits must be at least 100.",
            )
            return
        self.ax.set_ylim(self.ymin, self.ymax)
        self.add_secondary_y_axis()
        self.canvas.draw()
        self.sync_y_axis_entries()

    def sync_y_axis_entries(self):
        """Synchronizes the Y-axis entry boxes with the current plot limits."""
        ymin, ymax = self.ax.get_ylim()
        if ymin > ymax:
            ymin, ymax = (ymax, ymin)
        self.ymin_entry.config(state="normal")
        self.ymax_entry.config(state="normal")
        self.ymin_entry.delete(0, tk.END)
        self.ymin_entry.insert(0, str(int(ymin)))
        self.ymax_entry.delete(0, tk.END)
        self.ymax_entry.insert(0, str(int(ymax)))
        self.add_secondary_y_axis()

    def toggle_logscale(self):
        """Toggles the color scale between linear and logarithmic."""
        self.use_log_scale = not self.use_log_scale
        self.update_display()

    def plot_depth(self):
        """Plots the depth line (ping or smooth) on the current plot."""
        if self.depth_option.get() == "Smooth Depth":
            data = self.smooth_depth_img
        elif self.depth_option.get() == "Ping Depth":
            data = self.this_ping_depth_img
        else:
            return
        x = np.arange(0, self.slice_length)
        if data is not None:
            alpha_val = 1.0 if not self.edit_mode else 0.35
            self.ax.plot(
                x,
                data[: self.slice_length],
                color="blue",
                linewidth=2,
                alpha=alpha_val,
                label=self.depth_option.get(),
            )
            self.ax.legend(loc="upper right")
            self.canvas.draw()

    def prev_slice(self):
        """Navigates between slices of sonar data."""
        self.clear_annotations()
        self.manual_line_saved = False
        self.edit_mode = False
        self.depth_option.set("Ping Depth")
        for child in self.depth_frame.winfo_children():
            child.config(state="normal")
        for child in self.y_axis_frame.winfo_children():
            child.config(state="normal")
        self.idx_start -= self.chunk_size
        self.process_next_chunk()

    def next_slice(self):
        """Navigates between slices of sonar data."""
        self.clear_annotations()
        self.manual_line_saved = False
        self.edit_mode = False
        self.depth_option.set("Ping Depth")
        self.y_update_button.config(state="normal")
        for child in self.depth_frame.winfo_children():
            child.config(state="normal")
        for child in self.y_axis_frame.winfo_children():
            child.config(state="normal")
        self.idx_start += self.chunk_size
        self.process_next_chunk()

    def jump_to_slice(self):
        """Jumps to a specific data slice based on user input."""
        try:
            target_slice = int(self.jump_slice_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Slice number must be an integer.")
            return
        if self.total_slices is None:
            messagebox.showerror("Error", "No file loaded.")
            return
        if target_slice < 1 or target_slice > self.total_slices:
            messagebox.showerror(
                "Invalid Slice",
                f"Slice number must be between 1 and {self.total_slices}.",
            )
            return
        self.idx_start = (target_slice - 1) * self.chunk_size
        self.depth_option.set("Ping Depth")
        for child in self.depth_frame.winfo_children():
            child.config(state="normal")
        for child in self.y_axis_frame.winfo_children():
            child.config(state="normal")
        self.process_next_chunk()

    def quit_gui(self):
        """Gracefully terminates the GUI application."""
        if hasattr(self, "_closing") and self._closing:
            return
        self._closing = True
        self.unbind_all_events()
        try:
            if self.canvas and self.canvas.get_tk_widget().winfo_exists():
                self.canvas.get_tk_widget().unbind("<Destroy>")  # prevent callback loop
                self.canvas.get_tk_widget().destroy()
        except Exception:
            pass
        try:
            if self.root and self.root.winfo_exists():
                self.root.quit()
                self.root.destroy()
        except Exception:
            pass

    def unbind_all_events(self):
        """Handles functionality related to sonar data tracing and annotation."""
        events = [
            "<Button-1>",
            "<B1-Motion>",
            "<Button-3>",
            "<B3-Motion>",
            "<ButtonRelease-1>",
            "<ButtonRelease-3>",
        ]
        for event in events:
            self.canvas_widget.unbind(event)

    def enable_annotation(self):
        """Handles functionality related to sonar data tracing and annotation."""
        if not self.manual_line_saved:
            if self.depth_option.get() == "Off":
                self.canvas_widget.bind("<Button-1>", self.start_tracing)
                self.canvas_widget.bind("<B1-Motion>", self.trace_line)
                self.canvas_widget.bind("<Button-3>", self.stop_tracing)
            elif self.edit_mode:
                self.canvas_widget.bind("<Button-1>", self.start_tracing_editing_green)
                self.canvas_widget.bind("<B1-Motion>", self.trace_line_editing_green)
                self.canvas_widget.bind("<Button-3>", self.start_tracing_editing_red)
                self.canvas_widget.bind("<B3-Motion>", self.trace_line_editing_red)
                self.canvas_widget.bind("<ButtonRelease-1>", self.stop_tracing)
                self.canvas_widget.bind("<ButtonRelease-3>", self.stop_tracing)
            else:
                self.unbind_all_events()
        else:
            self.unbind_all_events()

    def canvas_to_data(self, x, y):
        """Converts canvas (pixel) coordinates to data coordinates."""
        x_offset = self.canvas_widget.winfo_rootx() - self.root.winfo_rootx()
        y_offset = self.canvas_widget.winfo_rooty() - self.root.winfo_rooty()
        fig_x = x - x_offset
        fig_y = y - y_offset
        fig_y = self.canvas_widget.winfo_height() - fig_y
        data_x, data_y = self.ax.transData.inverted().transform((fig_x, fig_y))
        return (data_x, data_y)

    def start_tracing(self, event):
        """Begins the tracing operation for user input."""
        self.tracing = True
        for child in self.y_axis_frame.winfo_children():
            child.config(state="disabled")
        self.last_x, self.last_y = (event.x, event.y)
        self.ymin_entry.config(state="disabled")
        self.ymax_entry.config(state="disabled")
        self.y_update_button.config(state="disabled")
        self.coordinates.append(
            (event.x, event.y, *self.canvas_to_data(event.x, event.y), "green")
        )
        if self.depth_option.get() == "Off":
            self.save_button.config(
                text="Apply Traced Line", command=self.apply_traced_line, state="normal"
            )

    def trace_line(self, event):
        """Records line drawing as the user moves the mouse."""
        if self.tracing:
            data_x, data_y = self.canvas_to_data(event.x, event.y)
            x_min, x_max = self.ax.get_xlim()
            y_min, y_max = self.ax.get_ylim()
            if not (x_min <= data_x <= x_max and y_min <= data_y <= y_max):
                self.stop_tracing(event)
                return
            self.canvas_widget.create_line(
                self.last_x,
                self.last_y,
                event.x,
                event.y,
                fill="lime",
                width=2,
                tags="annotation",
            )
            if not self.coordinates or (event.x, event.y) != (
                self.coordinates[-1][0],
                self.coordinates[-1][1],
            ):
                self.coordinates.append((event.x, event.y, data_x, data_y, "green"))
            self.last_x, self.last_y = (event.x, event.y)

    def stop_tracing(self, event):
        """Stops the tracing operation."""
        self.tracing = False

    def start_tracing_editing_green(self, event):
        """Begins the tracing operation for user input."""
        self.tracing = True
        for child in self.y_axis_frame.winfo_children():
            child.config(state="disabled")
        self.last_x, self.last_y = (event.x, event.y)
        self.ymin_entry.config(state="disabled")
        self.ymax_entry.config(state="disabled")
        self.y_update_button.config(state="disabled")
        x_data, y_data = self.canvas_to_data(event.x, event.y)
        self.coordinates.append((event.x, event.y, x_data, y_data, "green"))

    def trace_line_editing_green(self, event):
        """Records line drawing as the user moves the mouse."""
        if self.tracing:
            data_x, data_y = self.canvas_to_data(event.x, event.y)
            if not (
                self.ax.get_xlim()[0] <= data_x <= self.ax.get_xlim()[1]
                and self.ax.get_ylim()[0] <= data_y <= self.ax.get_ylim()[1]
            ):
                self.stop_tracing(event)
                return
            self.canvas_widget.create_line(
                self.last_x,
                self.last_y,
                event.x,
                event.y,
                fill="lime",
                width=2,
                tags="annotation",
            )
            if not self.coordinates or (event.x, event.y) != (
                self.coordinates[-1][0],
                self.coordinates[-1][1],
            ):
                self.coordinates.append((event.x, event.y, data_x, data_y, "green"))
            self.last_x, self.last_y = (event.x, event.y)

    def start_tracing_editing_red(self, event):
        """Begins the tracing operation for user input."""
        self.tracing = True
        for child in self.y_axis_frame.winfo_children():
            child.config(state="disabled")
        self.last_x, self.last_y = (event.x, event.y)
        self.ymin_entry.config(state="disabled")
        self.ymax_entry.config(state="disabled")
        self.y_update_button.config(state="disabled")
        x_data, y_data = self.canvas_to_data(event.x, event.y)
        self.coordinates.append((event.x, event.y, x_data, y_data, "red"))

    def trace_line_editing_red(self, event):
        """Records line drawing as the user moves the mouse."""
        if self.tracing:
            data_x, data_y = self.canvas_to_data(event.x, event.y)
            if not (
                self.ax.get_xlim()[0] <= data_x <= self.ax.get_xlim()[1]
                and self.ax.get_ylim()[0] <= data_y <= self.ax.get_ylim()[1]
            ):
                self.stop_tracing(event)
                return
            self.canvas_widget.create_line(
                self.last_x,
                self.last_y,
                event.x,
                event.y,
                fill="red",
                width=2,
                tags="annotation",
            )
            if not self.coordinates or (event.x, event.y) != (
                self.coordinates[-1][0],
                self.coordinates[-1][1],
            ):
                self.coordinates.append((event.x, event.y, data_x, data_y, "red"))
            self.last_x, self.last_y = (event.x, event.y)

    def interpolate_coordinates_by_color(self, color):
        """Interpolates coordinates to create a continuous depth line."""
        x_coords = np.arange(self.slice_length)
        filtered = [pt for pt in self.coordinates if pt[4] == color]
        if not filtered:
            return np.column_stack((x_coords, np.full(self.slice_length, np.nan)))
        pts = sorted([(pt[2], pt[3]) for pt in filtered], key=lambda p: p[0])
        y_interp = np.full(self.slice_length, np.nan)
        for x in x_coords:
            for i in range(len(pts) - 1):
                x0, y0 = pts[i]
                x1, y1 = pts[i + 1]
                if x1 - x0 <= 2.0 and x0 <= x <= x1:
                    y_interp[x] = np.interp(x, [x0, x1], [y0, y1])
                    break
        return np.column_stack((x_coords, y_interp))

    def interpolate_coordinates(self):
        """Interpolates coordinates to create a continuous depth line."""
        x_coords = np.arange(self.slice_length)
        if not self.coordinates:
            return np.column_stack((x_coords, np.full(self.slice_length, float(-999))))
        pts = sorted([(pt[2], pt[3]) for pt in self.coordinates], key=lambda p: p[0])
        y_interp = np.full(self.slice_length, np.nan)
        for x in x_coords:
            for i in range(len(pts) - 1):
                x0, y0 = pts[i]
                x1, y1 = pts[i + 1]
                if x1 - x0 <= 2.0 and x0 <= x <= x1:
                    y_interp[x] = np.interp(x, [x0, x1], [y0, y1])
                    break
        return np.column_stack((x_coords, y_interp))

    def clear_annotations(self):
        """Clears annotations or resets state."""
        self.coordinates = []
        self.tracing = False
        self.last_x, self.last_y = (None, None)
        self.canvas_widget.delete("annotation")
        if self.applied_line:
            self.update_display()

    def clear_edit_mode(self):
        """Clears annotations or resets state."""
        self.clear_annotations()
        self.ax.clear()
        self.ax.pcolormesh(self.image, cmap="plasma")
        self.ax.set_ylim(self.ymin, self.ymax)
        if self.depth_option.get() == "Smooth Depth":
            data = self.smooth_depth_img
        elif self.depth_option.get() == "Ping Depth":
            data = self.this_ping_depth_img
        else:
            data = None
        if data is not None:
            x = np.arange(0, self.chunk_size)
            self.ax.plot(
                x,
                data,
                color="blue",
                linewidth=2,
                alpha=0.35,
                label=self.depth_option.get(),
            )
        self.ax.set_title(
            "Editing Mode Enabled:\nLeft click to draw edits (green), Right click to omit data (red).",
            fontsize=16,
        )
        self.ax.legend(loc="upper right")
        self.ax.set_xlabel("Ping Count", fontsize=15)
        self.ax.set_ylabel("Bin #", fontsize=15)
        self.ax.tick_params(axis="y", which="both", labelleft=True, labelright=True)
        self.ax.tick_params(axis="x", labelsize=14)
        self.ax.tick_params(axis="y", labelsize=14)
        self.canvas.draw()
        self.unbind_all_events()
        self.canvas_widget.bind("<Button-1>", self.start_tracing_editing_green)
        self.canvas_widget.bind("<B1-Motion>", self.trace_line_editing_green)
        self.canvas_widget.bind("<Button-3>", self.start_tracing_editing_red)
        self.canvas_widget.bind("<B3-Motion>", self.trace_line_editing_red)
        self.canvas_widget.bind("<ButtonRelease-1>", self.stop_tracing)
        self.canvas_widget.bind("<ButtonRelease-3>", self.stop_tracing)
        self.update_button_states()

    def activate_lasso(self):
        """Activates the lasso tool for cleaning depth data."""
        self.save_depth_button.config(state="disabled")
        self.save_button.config(state="disabled")
        if self.edit_mode or self.depth_option.get() == "Off":
            messagebox.showwarning(
                "Unavailable",
                "Lasso cleaning is only available before editing Ping/Smooth Depth lines.",
            )
            return
        if self.depth_option.get() == "Ping Depth":
            self.depth_data = self.this_ping_depth_img[: self.slice_length]
        else:
            self.depth_data = self.smooth_depth_img[: self.slice_length]
        self.points = np.column_stack((np.arange(self.slice_length), self.depth_data))

        def onselect(verts):
            path = Path(verts)
            selected = path.contains_points(self.points)
            for i, sel in enumerate(selected):
                if sel and (not np.isnan(self.points[i, 1])):
                    self.depth_data[i] = np.nan
            self.points[:, 1] = self.depth_data
            for line in self.ax.lines[:]:
                line.remove()
            valid_mask = ~np.isnan(self.points[:, 1])
            self.ax.plot(
                self.points[valid_mask, 0],
                self.points[valid_mask, 1],
                "bo",
                markersize=4,
                label=self.depth_option.get(),
            )
            self.canvas.draw()
            self.start_lasso()

        def finish_lasso():
            if self.lasso:
                self.lasso.disconnect_events()
                self.lasso = None
            for pt in self.coordinates:
                if pt[4] == "red":
                    x = int(pt[2])
                    self.depth_data[x] = np.nan
            self.apply_edits()
            self.lasso_button.config(
                text=f"Clean {self.depth_option.get()} Line",
                command=self.activate_lasso,
            )
            self.save_depth_button.config(state="normal")
            self.save_button.config(state="normal")

        def start_lasso():
            if self.lasso:
                self.lasso.disconnect_events()
            self.ax.set_title(
                "Lasso Tool Active: Use the middle mouse button to draw around points to remove. Press 'Finish Cleaning' when done.",
                fontsize=16,
            )
            self.lasso = LassoSelector(
                self.ax, onselect, props=dict(color="red"), useblit=True
            )
            self.lasso.set_active(True)
            self.canvas.draw()

        self.start_lasso = start_lasso
        self.finish_lasso = finish_lasso
        for line in self.ax.lines[:]:
            if line.get_label() == self.depth_option.get():
                line.remove()
        self.ax.plot(
            self.points[:, 0],
            self.points[:, 1],
            "bo",
            markersize=4,
            label=self.depth_option.get(),
        )
        self.lasso_button.config(text="Finish Cleaning", command=self.finish_lasso)
        self.start_lasso()

    def enter_editing_mode(self):
        """Enables editing mode for adjusting the depth line."""
        try:
            self.ymin = float(self.ymin_entry.get())
            self.ymax = float(self.ymax_entry.get())
        except ValueError:
            messagebox.showerror(
                "Invalid Input", "Please enter numeric Y-axis limits before editing."
            )
            return
        self.edit_mode = True
        self.ax.clear()
        self.ax.pcolormesh(self.image, cmap="plasma")
        if self.ymin is not None and self.ymax is not None:
            self.ax.set_ylim(self.ymin, self.ymax)
        self.ax.tick_params(axis="y", which="both", labelleft=True, labelright=True)
        if self.depth_option.get() == "Smooth Depth":
            data = self.smooth_depth_img
        elif self.depth_option.get() == "Ping Depth":
            data = self.this_ping_depth_img
        else:
            data = None
        if data is not None:
            x = np.arange(0, self.slice_length)
            self.ax.plot(
                x,
                data,
                color="blue",
                linewidth=2,
                alpha=0.35,
                label=self.depth_option.get(),
            )
        self.ax.set_title(
            "Editing Mode Enabled:\nLeft click to draw edits (green), Right click to omit data (red).",
            fontsize=16,
        )
        self.ax.legend(loc="upper right")
        self.ax.set_xlabel("Ping Count", fontsize=15)
        self.ax.set_ylabel("Bin #", fontsize=15)
        self.ax.tick_params(axis="x", labelsize=14)
        self.ax.tick_params(axis="y", labelsize=14)
        self.sync_y_axis_entries()
        self.add_secondary_y_axis()
        self.canvas.draw()
        for child in self.depth_frame.winfo_children():
            child.config(state="disabled")
        self.unbind_all_events()
        self.canvas_widget.bind("<Button-1>", self.start_tracing_editing_green)
        self.canvas_widget.bind("<B1-Motion>", self.trace_line_editing_green)
        self.canvas_widget.bind("<Button-3>", self.start_tracing_editing_red)
        self.canvas_widget.bind("<B3-Motion>", self.trace_line_editing_red)
        self.canvas_widget.bind("<ButtonRelease-1>", self.stop_tracing)
        self.canvas_widget.bind("<ButtonRelease-3>", self.stop_tracing)
        self.clear_button.grid()
        self.clear_button.config(
            text="Clear Annotations", command=self.clear_edit_mode, state="normal"
        )
        self.save_button.config(
            text="Apply Edits", command=self.apply_edits, state="normal"
        )
        self.save_depth_button.config(
            text="Save Edited Depth Line",
            command=self.save_edited_depth_line,
            state="disabled",
        )
        self.lasso_button.config(state="disabled")

    def apply_edits(self):
        """Applies user edits or traced data to the current slice."""
        num_results = self.image.shape[0]

        # Use previously edited line if it exists, otherwise use factory
        if self.edited_line is not None:
            base_line = self.edited_line.copy()
        elif self.depth_option.get() == "Smooth Depth":
            base_line = self.smooth_depth_img[: self.slice_length].copy()
        elif self.depth_option.get() == "Ping Depth":
            base_line = self.this_ping_depth_img[: self.slice_length].copy()
        else:
            messagebox.showerror("Error", "No depth line available!")
            return

        # Get green edits from this round
        new_green_edit = self.interpolate_coordinates_by_color("green")[
            : self.slice_length, 1
        ]
        # Initialize accumulator if needed
        if (
            self.all_green_edits is None
            or len(self.all_green_edits) != self.slice_length
        ):
            self.all_green_edits = np.full(self.slice_length, np.nan)
        # Combine current round with all prior green edits
        combined_green = np.copy(self.all_green_edits)
        self.all_green_edits = combined_green
        for i in range(self.slice_length):
            if not np.isnan(new_green_edit[i]):
                combined_green[i] = new_green_edit[i]

        red_edit = self.interpolate_coordinates_by_color("red")[: self.slice_length, 1]

        # Apply red removals and green overrides
        merged = base_line
        for i in range(self.slice_length):
            if not np.isnan(red_edit[i]):
                merged[i] = np.nan
            elif not np.isnan(combined_green[i]):
                merged[i] = combined_green[i]

        self.edited_line = merged
        x_vals = np.arange(self.slice_length)

        self.ax.clear()
        self.ax.pcolormesh(self.image, cmap="plasma")

        # Plot the full merged result as a solid baseline
        self.ax.plot(
            x_vals,
            merged,
            linestyle="-",
            color="blue",
            linewidth=2,
            label="Edited Depth Line",
        )

        # Always overlay new green edits on top
        green_mask = ~np.isnan(combined_green)
        manual_x = x_vals[green_mask]
        manual_y = combined_green[green_mask]
        segments = np.split(
            np.column_stack((manual_x, manual_y)),
            np.where(np.diff(manual_x) > 1)[0] + 1,
        )
        for i, seg in enumerate(segments):
            if len(seg) > 0:
                self.ax.plot(
                    seg[:, 0],
                    seg[:, 1],
                    linestyle="-",
                    color="lime",
                    linewidth=2,
                    label="New Manual Edits" if i == 0 else "_nolegend_",
                )

        self.ax.tick_params(axis="y", which="both", labelleft=True, labelright=True)
        self.ax.legend(loc="upper right")
        self.ax.set_title("")
        self.ax.set_xlabel("Ping Count", fontsize=15)
        self.ax.set_ylabel("Bin #", fontsize=15)
        self.ax.tick_params(axis="x", labelsize=14)
        self.ax.tick_params(axis="y", labelsize=14)
        self.ax.set_ylim(0, num_results)
        self.sync_y_axis_entries()
        self.add_secondary_y_axis()
        self.canvas.draw()
        self.clear_annotations()
        self.unbind_all_events()
        self.save_button.config(
            text="Continue Editing", command=self.continue_editing, state="normal"
        )
        self.save_depth_button.config(state="normal")

    def continue_editing(self):
        """Allows continued editing from the last applied edits."""
        self.edit_mode = True
        self.ax.clear()
        self.ax.pcolormesh(self.image, cmap="plasma")
        self.ax.set_ylim(0, self.image.shape[0])

        x = np.arange(0, self.slice_length)
        merged = self.edited_line
        if merged is not None:
            self.ax.plot(
                x,
                merged,
                color="blue",
                linewidth=2,
                alpha=0.35,
                label="Edited Depth Line",
            )

        self.ax.set_title(
            "Editing Mode Enabled:\nLeft click to draw edits (green), Right click to omit data (red).",
            fontsize=16,
        )
        self.ax.legend(loc="upper right")
        self.ax.set_xlabel("Ping Count", fontsize=15)
        self.ax.set_ylabel("Bin #", fontsize=15)
        self.ax.tick_params(axis="x", labelsize=14)
        self.ax.tick_params(axis="y", labelsize=14)

        self.sync_y_axis_entries()
        self.add_secondary_y_axis()
        self.canvas.draw()

        for child in self.depth_frame.winfo_children():
            child.config(state="disabled")

        self.unbind_all_events()
        self.canvas_widget.bind("<Button-1>", self.start_tracing_editing_green)
        self.canvas_widget.bind("<B1-Motion>", self.trace_line_editing_green)
        self.canvas_widget.bind("<Button-3>", self.start_tracing_editing_red)
        self.canvas_widget.bind("<B3-Motion>", self.trace_line_editing_red)
        self.canvas_widget.bind("<ButtonRelease-1>", self.stop_tracing)
        self.canvas_widget.bind("<ButtonRelease-3>", self.stop_tracing)

        self.clear_button.grid()
        self.clear_button.config(
            text="Clear Annotations", command=self.clear_edit_mode, state="normal"
        )
        self.save_button.config(
            text="Apply Edits", command=self.apply_edits, state="normal"
        )
        self.save_depth_button.config(
            text="Save Edited Depth Line",
            command=self.save_edited_depth_line,
            state="disabled",
        )
        self.y_update_button.config(state="normal")
        self.lasso_button.config(state="disabled")

    def apply_traced_line(self):
        """Applies user edits or traced data to the current slice."""
        if not self.coordinates:
            self.canvas_widget.delete("annotation")
            self.ax.clear()
            self.ax.pcolormesh(self.image, cmap="plasma")
            self.ax.set_xlim(0, self.slice_length)
            self.ax.set_ylim(0, self.image.shape[0])
            self.ax.set_xlabel("Ping Count", fontsize=15)
            self.ax.set_ylabel("Bin #", fontsize=15)
            self.ax.tick_params(axis="x", labelsize=14)
            self.ax.tick_params(axis="y", labelsize=14)
            self.canvas.draw()
            self.save_data()
            return
        traced_values = self.interpolate_coordinates_by_color("green")[
            : self.slice_length, 1
        ]
        self.canvas_widget.delete("annotation")
        x_vals = np.arange(self.slice_length)
        self.ax.clear()
        self.ax.pcolormesh(self.image, cmap="plasma")
        self.ax.plot(
            x_vals, traced_values, color="blue", linewidth=2, label="Manual Depth Line"
        )
        self.ax.legend(loc="upper right")
        self.ax.set_title("")
        self.ax.set_xlabel("Time (s)", fontsize=15)
        self.ax.set_ylabel("Bin #", fontsize=15)
        self.ax.tick_params(axis="x", labelsize=14)
        self.ax.tick_params(axis="y", labelsize=14)
        self.add_secondary_y_axis()
        self.canvas.draw()
        for child in self.depth_frame.winfo_children():
            child.config(state="disabled")
        self.unbind_all_events()
        self.applied_line = True
        self.save_button.config(state="disabled")
        self.save_depth_button.config(
            text="Save Manual Depth Line", command=self.save_data, state="normal"
        )

    def save_edited_depth_line(self):
        """Saves data or images to file."""
        if self.edited_line is None:
            messagebox.showerror("Error", "No edited depth line available!")
            return
        num_results = self.image.shape[0]
        self.ax.set_ylim(0, num_results)
        self.canvas.draw()
        merged = self.edited_line
        idxS = self.idx_start
        x_vals = np.arange(self.slice_length)
        self.ax.clear()
        self.ax.pcolormesh(self.image, cmap="plasma")
        self.ax.plot(
            x_vals, merged, color="cyan", linewidth=2, label="Saved Depth Line"
        )
        self.ax.tick_params(axis="y", which="both", labelleft=True, labelright=True)
        self.ax.set_xlim(0, self.slice_length)
        self.ax.set_ylim(0, self.image.shape[0])
        self.ax.legend(loc="upper right")
        self.ax.set_xlabel("Ping Count", fontsize=15)
        self.ax.set_ylabel("Bin #", fontsize=15)
        self.ax.tick_params(axis="x", labelsize=14)
        self.ax.tick_params(axis="y", labelsize=14)
        self.add_secondary_y_axis()
        default_filename = (
            f"{self.base_name}_bottomTraced_{idxS}-{idxS + self.slice_length - 1}.png"
        )
        self.ax.set_title(default_filename, fontsize=16)
        self.canvas.draw()
        self.root.update()
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", dpi=300)
        buf.seek(0)
        self.image_for_saving = Image.open(buf)
        raw_dir = os.path.dirname(self.input_file_path)
        qcplots = os.path.join(raw_dir, "qcPlots")
        os.makedirs(qcplots, exist_ok=True)
        png_path = os.path.join(qcplots, default_filename)
        self.image_for_saving.save(png_path)
        qcddata = os.path.join(raw_dir, "qcdData")
        os.makedirs(qcddata, exist_ok=True)
        h5_name = default_filename.replace(".png", ".h5")
        h5_path = os.path.join(qcddata, h5_name)
        with h5py.File(h5_path, "w") as hf:
            merged_data = np.column_stack((x_vals, merged))
            hf.create_dataset("depth_line_by_slice_idx", data=merged_data)
            time_indices = np.arange(idxS, idxS + self.slice_length)
            hf.create_dataset(
                "depth_line_by_time_idx", data=np.column_stack((time_indices, merged))
            )
            hf.create_dataset("profile_data_slice", data=self.image)
        self.update_whole_record(np.arange(idxS, idxS + self.slice_length), merged)
        print(
            f"\n============= Slice #: {str(self.slice_number).zfill(2)} =============="
        )
        print(f"Image saved: {os.path.normpath(png_path)}")
        print(f"Depth line saved: {os.path.normpath(h5_path)}")
        print(f"Whole record updated: {os.path.normpath(self.whole_record_file)}")
        print(
            f"Raw file qaqc_depth_line updated: {os.path.normpath(self.input_file_path)}"
        )
        print("========================================\n")
        self.unbind_all_events()
        self.slice_saved()

    def save_data(self):
        """Saves data or images to file."""
        self.clear_button.config(state="disabled")
        self.save_button.config(state="disabled")
        self.save_depth_button.config(state="disabled")
        num_results = self.image.shape[0]
        self.ax.set_ylim(0, num_results)
        idxS = self.idx_start
        default_filename = (
            f"{self.base_name}_bottomTraced_{idxS}-{idxS + self.slice_length - 1}.png"
        )
        self.ax.set_title(default_filename, fontsize=16)
        self.canvas.draw()
        self.sync_y_axis_entries()
        if self.image is None:
            messagebox.showerror("Error", "No image loaded to save.")
            return
        interp_values = self.interpolate_coordinates()[: self.slice_length, :]
        valid = ~np.isnan(interp_values[:, 1])
        self.clear_annotations()
        if np.any(valid):
            indices = np.where(valid)[0]
            splits = np.where(np.diff(indices) != 1)[0] + 1
            segments = np.split(indices, splits)
            first = True
            for seg in segments:
                if len(seg) >= 2:
                    if first:
                        self.ax.plot(
                            interp_values[seg, 0],
                            interp_values[seg, 1],
                            color="cyan",
                            linewidth=2,
                            label="Saved Depth Line",
                        )
                        first = False
                    else:
                        self.ax.plot(
                            interp_values[seg, 0],
                            interp_values[seg, 1],
                            color="cyan",
                            linewidth=2,
                        )
        else:
            self.ax.plot(
                np.arange(self.slice_length),
                np.full(self.slice_length, float(-999)),
                color="cyan",
                linewidth=2,
                label="Saved Depth Line",
            )
        self.ax.legend(loc="upper right")
        for line in self.ax.get_lines():
            if line.get_label() == "Manual Depth Line":
                line.remove()
        self.add_secondary_y_axis()
        self.canvas.draw()
        self.root.update()
        self.image_for_saving = Image.fromarray(
            np.array(self.fig.canvas.renderer.buffer_rgba())
        )
        raw_dir = os.path.dirname(self.input_file_path)
        qcplots = os.path.join(raw_dir, "qcPlots")
        os.makedirs(qcplots, exist_ok=True)
        png_path = os.path.join(qcplots, default_filename)
        self.image_for_saving.save(png_path)
        qcddata = os.path.join(raw_dir, "qcdData")
        os.makedirs(qcddata, exist_ok=True)
        h5_name = default_filename.replace(".png", ".h5")
        h5_path = os.path.join(qcddata, h5_name)
        with h5py.File(h5_path, "w") as hf:
            hf.create_dataset("depth_line_by_slice_idx", data=interp_values)
            time_indices = np.arange(idxS, idxS + self.slice_length)
            hf.create_dataset(
                "depth_line_by_time_idx",
                data=np.column_stack((time_indices, interp_values[:, 1])),
            )
            hf.create_dataset("profile_data_slice", data=self.image)
        self.update_whole_record(
            np.arange(idxS, idxS + self.slice_length), interp_values[:, 1]
        )
        print(
            f"\n============= Slice #: {str(self.slice_number).zfill(2)} =============="
        )
        print(f"Image saved: {os.path.normpath(png_path)}")
        print(f"Manual depth line saved: {os.path.normpath(h5_path)}")
        print(f"Whole record updated: {os.path.normpath(self.whole_record_file)}")
        print(
            f"Raw file qaqc_depth_line updated: {os.path.normpath(self.input_file_path)}"
        )
        print("========================================\n")
        self.manual_line_saved = True
        self.unbind_all_events()
        self.slice_saved()

    def save_depth_line(self):
        """Saves data or images to file."""
        self.clear_button.config(state="disabled")
        self.save_button.config(state="disabled")
        self.save_depth_button.config(state="disabled")
        num_results = self.image.shape[0]
        self.ax.set_ylim(0, num_results)
        self.canvas.draw()
        self.sync_y_axis_entries()
        if self.depth_option.get() not in ["Smooth Depth", "Ping Depth"]:
            messagebox.showerror("Error", "No depth line selected!")
            return
        data = (
            self.smooth_depth_img
            if self.depth_option.get() == "Smooth Depth"
            else self.this_ping_depth_img
        )
        if data is None:
            messagebox.showerror("Error", "Depth data not available!")
            return
        idxS = self.idx_start
        x_coords = np.arange(0, self.slice_length)
        depth_coords = data[: self.slice_length]
        self.ax.clear()
        self.ax.pcolormesh(self.image, cmap="plasma")
        self.ax.tick_params(axis="y", which="both", labelleft=True, labelright=True)
        self.ax.plot(
            x_coords, depth_coords, color="cyan", linewidth=2, label="Saved Depth Line"
        )
        self.ax.legend(loc="upper right")
        self.ax.set_xlabel("Ping Count", fontsize=15)
        self.ax.set_ylabel("Bin #", fontsize=15)
        self.ax.tick_params(axis="x", labelsize=14)
        self.ax.tick_params(axis="y", labelsize=14)
        default_filename = (
            f"{self.base_name}_bottomTraced_{idxS}-{idxS + self.slice_length - 1}.png"
        )
        self.ax.set_title(default_filename, fontsize=16)
        self.add_secondary_y_axis()
        self.canvas.draw()
        self.root.update()
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", dpi=300)
        buf.seek(0)
        self.image_for_saving = Image.open(buf)
        raw_dir = os.path.dirname(self.input_file_path)
        qcplots = os.path.join(raw_dir, "qcPlots")
        os.makedirs(qcplots, exist_ok=True)
        png_path = os.path.join(qcplots, default_filename)
        self.image_for_saving.save(png_path)
        qcddata = os.path.join(raw_dir, "qcdData")
        os.makedirs(qcddata, exist_ok=True)
        h5_name = default_filename.replace(".png", ".h5")
        h5_path = os.path.join(qcddata, h5_name)
        with h5py.File(h5_path, "w") as hf:
            merged_data = np.column_stack((x_coords, depth_coords))
            hf.create_dataset("depth_line_by_slice_idx", data=merged_data)
            time_indices = np.arange(idxS, idxS + self.slice_length)
            hf.create_dataset(
                "depth_line_by_time_idx",
                data=np.column_stack((time_indices, depth_coords)),
            )
            hf.create_dataset("profile_data_slice", data=self.image)
        self.update_whole_record(
            np.arange(idxS, idxS + self.slice_length), depth_coords
        )
        print(
            f"\n============= Slice #: {str(self.slice_number).zfill(2)} =============="
        )
        print(f"Image saved: {os.path.normpath(png_path)}")
        print(f"Depth line saved: {os.path.normpath(h5_path)}")
        print(f"Whole record updated: {os.path.normpath(self.whole_record_file)}")
        print(
            f"Raw file qaqc_depth_line updated: {os.path.normpath(self.input_file_path)}"
        )
        print("========================================\n")
        self.manual_line_saved = True
        self.unbind_all_events()
        self.slice_saved()

    def slice_saved(self):
        """Handles actions after a slice has been saved."""
        if self.slice_number != self.total_slices:
            self.next_slice()
        else:
            response = messagebox.askyesno(
                "All slices annotated",
                "All slices have been annotated. Are you done editing?",
            )
            if response:
                self.show_final_image_progress()

    def show_final_image_progress(self):
        """Displays a popup while generating the final whole-record image in a background thread."""
        # Create a top-level popup
        popup = tk.Toplevel(self.root)
        popup.title("Processing")
        tk.Label(
            popup,
            text="Generating final image...\nThis may take a moment.",
            font=("Helvetica", 14),
        ).pack(padx=20, pady=20)
        popup.geometry("300x100")
        popup.grab_set()
        popup.transient(self.root)
        popup.update()

        def background_task():
            self.generate_final_whole_record_image()

            def on_complete():
                messagebox.showinfo("Complete", "Final image has been saved.")
                self.quit_gui()  # Only called after user closes the message box

            self.root.after(0, popup.destroy)
            self.root.after(0, on_complete)

        threading.Thread(target=background_task, daemon=True).start()

    def generate_final_whole_record_image(self):
        """Creates a PNG of the full sonar record with qaqc_depth_line overlaid and secondary depth axis."""
        import matplotlib

        matplotlib.use("Agg")  # Set backend just for this scope
        import matplotlib.pyplot as plt

        with h5py.File(self.input_file_path, "r") as h5:
            profile_data = h5["profile_data"][:]
            try:
                depth_line = h5["qaqc_depth_line"][:, 1]
                depth_line = np.where(depth_line == -999, np.nan, depth_line)
            except KeyError:
                print("qaqc_depth_line not found — skipping final image.")
                return

            # Estimate bin size from length_mm if available
            if "length_mm" in h5:
                length_mm = h5["length_mm"][:]
                avg_length_m = np.mean(length_mm) / 1000.0  # convert to meters
                bin_size = avg_length_m / profile_data.shape[0]
            else:
                bin_size = None

        fig, ax = plt.subplots(figsize=(20, 12))
        ax.pcolormesh(profile_data, cmap="plasma")
        ax.set_title(f"{self.base_name}_bottomTraced_wholeRecord.png", fontsize=16)
        ax.set_xlabel("Ping Index", fontsize=15)
        ax.set_ylabel("Bin #", fontsize=15)

        # Plot the full depth line
        x_vals = np.arange(len(depth_line))
        valid = ~np.isnan(depth_line)
        ax.plot(
            x_vals[valid],
            depth_line[valid],
            color="cyan",
            linewidth=2,
            label="Saved Depth Line",
        )
        ax.set_ylim(0, self.image.shape[0])

        # Add secondary depth-axis
        if bin_size is not None:
            ax2 = ax.twinx()
            ax2.set_ylim(ax.get_ylim()[0] * bin_size, ax.get_ylim()[1] * bin_size)
            ax2.set_ylabel("Depth Range (m)", fontsize=15)
            ax2.tick_params(axis="y", labelsize=14)
            ax2.set_label("secondary_y")

        ax.legend(loc="upper right")
        ax.tick_params(axis="both", labelsize=14)
        fig.text(
            0.01,
            0.98,
            f"Whole Record\nTime Indices: 0 - {self.total_time}",
            horizontalalignment="left",
            verticalalignment="top",
            fontsize=12,
            color="black",
        )

        raw_dir = os.path.dirname(self.input_file_path)
        final_path = os.path.join(raw_dir, f"{self.base_name}_wholeRecord.png")
        fig.savefig(final_path, dpi=300)
        plt.close(fig)

        end_time = time.time()
        elapsed_minutes = (
            round((end_time - self.start_time) / 60, 2) if self.start_time else "N/A"
        )

        print("\n============= Final Summary ==============")
        print(f"Final whole-record image saved: {os.path.normpath(final_path)}")
        print(f"Total time elapsed: {elapsed_minutes} minutes")
        print("==========================================\n")

    def update_whole_record(self, time_indices, depth_values):
        """Updates internal state or display elements."""
        with h5py.File(self.whole_record_file, "a", locking=False) as hf:
            if "qaqc_depth_line" not in hf:
                full_data = np.column_stack(
                    (np.arange(self.total_time), np.full(self.total_time, float(-999)))
                )
                hf.create_dataset(
                    "qaqc_depth_line", data=full_data, maxshape=(self.total_time, 2)
                )
            dset = hf["qaqc_depth_line"]
            for i, idx in enumerate(time_indices):
                dset[idx, 0] = idx
                dset[idx, 1] = depth_values[i]
        with h5py.File(self.input_file_path, "a") as raw_h5:
            if "qaqc_depth_line" not in raw_h5:
                full_data = np.column_stack(
                    (np.arange(self.total_time), np.full(self.total_time, float(-999)))
                )
                raw_h5.create_dataset(
                    "qaqc_depth_line", data=full_data, maxshape=(self.total_time, 2)
                )
            raw_dset = raw_h5["qaqc_depth_line"]
            for i, idx in enumerate(time_indices):
                raw_dset[idx, 0] = idx
                raw_dset[idx, 1] = depth_values[i]

    @property
    def whole_record_data(self):
        """Returns all datasets from the input HDF5 file, handling scalar and array data."""
        if not hasattr(self, "input_file_path") or not os.path.exists(
            self.input_file_path
        ):
            return None
        with h5py.File(self.input_file_path, "r") as f:
            result = {}
            for key in f.keys():
                dataset = f[key]
                if dataset.shape == ():  # scalar
                    result[key] = dataset[()]
                else:
                    result[key] = dataset[:]
            return result


def run_sonar_tracer_gui(input_file=None, chunk_size=None):
    root = tk.Tk()

    screen_height = root.winfo_screenheight()
    root.geometry("500x155")
    # Dynamically compute font size based on screen height
    base_font_size = max(10, int(screen_height / 100))  # e.g., 1080px → size 10
    default_font = tkfont.nametofont("TkDefaultFont")
    default_font.configure(size=base_font_size)

    app = bottomTracer(root)
    # If arguments were passed, set them and auto-load the file
    if input_file and chunk_size:
        app.input_file_path.set(input_file)
        app.chunk_size_entry.delete(0, tk.END)
        app.chunk_size_entry.insert(0, str(chunk_size))
        root.after(100, app.load_file)  # Delay to let GUI init first
    root.mainloop()
    return app.whole_record_data


if __name__ == "__main__":
    run_sonar_tracer_gui()
