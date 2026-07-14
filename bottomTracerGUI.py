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
        self.root.title('HDF5 Annotator')
        self.root.protocol('WM_DELETE_WINDOW', self.quit_gui)
        self.depth_option = tk.StringVar(value='Ping Depth')
        self.manual_line_saved = False
        self.edit_mode = False
        self.edited_line = None
        self.menu_frame = tk.Frame(root, borderwidth=2, relief='groove')
        self.menu_frame.pack(side='top', fill='x', padx=10, pady=10)
        for col in range(4):
            self.menu_frame.grid_columnconfigure(col, weight=1)
        tk.Label(self.menu_frame, text='Chunk Size:').grid(row=0, column=1, padx=0, pady=5)
        self.chunk_size_entry = tk.Entry(self.menu_frame, width=10, justify='center')
        self.chunk_size = 250
        self.chunk_size_entry.insert(0, str(self.chunk_size))
        self.chunk_size_entry.grid(row=0, column=2, padx=0, pady=5)
        tk.Label(self.menu_frame, text='Input File:').grid(row=1, column=1, padx=10, pady=5)
        self.input_file_path = tk.StringVar(value=os.getcwd())
        self.input_dir_entry = tk.Entry(self.menu_frame, textvariable=self.input_file_path, width=40, justify='center')
        self.input_dir_entry.grid(row=1, column=2, padx=10, pady=5, sticky='w')
        self.browseInput_Button = tk.Button(self.menu_frame, text='Browse', command=self.choose_input_file)
        self.browseInput_Button.grid(row=1, column=3, padx=10, pady=5, sticky='w')
        self.load_button = tk.Button(self.menu_frame, text='Start Labeling', command=self.load_file, width=30)
        self.load_button.grid(row=3, column=0, padx=0, pady=5, columnspan=4)
        self.annotation_frame = tk.Frame(root)
        
        screen_width = root.winfo_screenwidth()# Get screen dimensions
        screen_height = root.winfo_screenheight()
        max_width = int(screen_width * 0.75) # Define margins (e.g., use 90% of available screen size)
        max_height = int(screen_height * 0.75)
        target_ratio = 5 / 3 # Desired aspect ratio
        if max_width / target_ratio <= max_height: # Compute figure size while maintaining aspect ratio
            fig_width_px = max_width
            fig_height_px = int(max_width / target_ratio)
        else:
            fig_height_px = max_height
            fig_width_px = int(max_height * target_ratio)
        fig_width_px = fig_width_px - (fig_width_px % 2)  # Round to nearest even number
        fig_height_px = fig_height_px - (fig_height_px % 2)  # Round to nearest even number
        dpi = 100  # Convert pixels to inches (assuming 100 DPI)
        fig_width_in = fig_width_px / dpi
        fig_height_in = fig_height_px / dpi
        self.fig, self.ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=dpi)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.annotation_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill='both', expand=True)
        self.canvas_widget.bind('<Destroy>', self.on_canvas_destroy)
        self.jump_frame = tk.Frame(self.annotation_frame)
        self.jump_frame.pack(pady=5)
        tk.Label(self.jump_frame, text='Slice #').grid(row=0, column=0, padx=5)
        self.prev_button_jump = tk.Button(self.jump_frame, text='<<', command=self.prev_slice)
        self.prev_button_jump.grid(row=0, column=1, padx=5)
        self.jump_slice_var = tk.StringVar(value='1')
        self.jump_entry = tk.Entry(self.jump_frame, textvariable=self.jump_slice_var, width=5, justify='center')
        self.jump_entry.grid(row=0, column=2, padx=5)
        self.next_button_jump = tk.Button(self.jump_frame, text='>>', command=self.next_slice)
        self.next_button_jump.grid(row=0, column=3, padx=5)
        self.jump_button = tk.Button(self.jump_frame, text='Jump To Slice', command=self.jump_to_slice)
        self.jump_button.grid(row=0, column=4, padx=5)
        self.y_axis_frame = tk.Frame(self.annotation_frame)
        self.y_axis_frame.pack(pady=5)
        tk.Label(self.y_axis_frame, text='Y Axis Limits:').pack(side='left', padx=5)
        self.ymin_entry = tk.Entry(self.y_axis_frame, width=10, justify='center')
        self.ymin_entry.pack(side='left', padx=5)
        tk.Label(self.y_axis_frame, text='-').pack(side='left', padx=5)
        self.ymax_entry = tk.Entry(self.y_axis_frame, width=10, justify='center')
        self.ymax_entry.pack(side='left', padx=5)
        self.y_update_button = tk.Button(self.y_axis_frame, text='Update', command=self.update_y_axis_limits)
        self.y_update_button.pack(side='left', padx=5)
        self.view_full_extent_button = tk.Button(self.y_axis_frame,text="View Full Extent",command=self.view_full_extent)
        self.view_full_extent_button.pack(side='left', padx=5)
        self.depth_frame = tk.Frame(self.annotation_frame)
        self.depth_frame.pack(pady=5)
        tk.Label(self.depth_frame, text='Toggle Depth Line:').grid(row=0, column=0)
        self.qaqc_radio = tk.Radiobutton(self.depth_frame, text='QAQC Depth', variable=self.depth_option, value='QAQC Depth', command=self.update_display)
        self.qaqc_radio.grid(row=0, column=1)
        tk.Radiobutton(self.depth_frame, text='Ping Depth', variable=self.depth_option, value='Ping Depth', command=self.update_display).grid(row=0, column=2)
        tk.Radiobutton(self.depth_frame, text='Smooth Depth', variable=self.depth_option, value='Smooth Depth', command=self.update_display).grid(row=0, column=3)
        tk.Radiobutton(self.depth_frame, text='Off', variable=self.depth_option, value='Off', command=self.update_display).grid(row=0, column=4)
        self.nan_button = tk.Button(self.depth_frame, text='Omit Whole Slice', command=self.apply_traced_line)
        self.nan_button.grid(row=0, column=5, padx=(20, 0))
        self.button_frame = tk.Frame(self.annotation_frame)
        self.button_frame.pack(fill='both', expand=True)
        self.clear_button = tk.Button(self.button_frame, text='Clear Annotations', command=self.clear_annotations)
        self.clear_button.grid(row=0, column=0, padx=10, pady=5, sticky='ew')
        self.save_button = tk.Button(self.button_frame, text='', state='normal')
        self.save_button.grid(row=0, column=1, padx=10, pady=5, sticky='ew')
        self.save_depth_button = tk.Button(self.button_frame, text='', state='normal')
        self.save_depth_button.grid(row=0, column=2, padx=10, pady=5, sticky='ew')
        self.lasso_button = tk.Button(self.button_frame, text='', command=self.activate_lasso, state='disabled')
        self.lasso_button.grid(row=0, column=3, padx=10, pady=5, sticky='ew')
        self.logscale_button = tk.Button(self.depth_frame, text='Toggle Log Scale', command=self.toggle_logscale)
        self.logscale_button.grid(row=0, column=6, padx=(20, 0))  # adjust as needed
        self.quit_button = tk.Button(self.button_frame, text='Quit', command=self.quit_gui)
        self.quit_button.grid(row=1, column=0, columnspan=4, padx=10, pady=5, sticky='ew')
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
        self.qaqc_depth_img = None
        self.has_existing_qaqc_in_slice = False
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
        self.user_ylim = None
        self.first_image_loaded = False     # becomes True once we load the first slice
        self.user_zoom_set = False          # becomes True when user edits Y limits
        self.qaqc_modified = False
        self.slice_already_qcd = False
        
    def on_canvas_destroy(self, event):
        """Cleans up when the canvas is destroyed."""
        if hasattr(self, '_closing') and self._closing:
            return
        self._closing = True
        if self.root and self.root.winfo_exists():
            self.quit_gui()

    def choose_input_file(self):
        """Opens a file or directory selection dialog."""
        file_path = filedialog.askopenfilename(initialdir=os.getcwd(), filetypes=[('HDF5 files', '*.h5 *.hdf5')])
        if file_path:
            self.input_file_path.set(file_path)

    def load_file(self):
        """Loads and initializes data or files for processing."""
        self.start_time = time.time()
        try:
            self.chunk_size = int(self.chunk_size_entry.get())
        except ValueError:
            messagebox.showerror('Invalid Input', 'Chunk size must be an integer.')
            return
        self.input_file_path = self.input_file_path.get()
        input_file = os.path.basename(self.input_file_path)
        self.base_name = os.path.splitext(input_file)[0]
        raw_dir = os.path.dirname(self.input_file_path)
        self.output_folder = raw_dir
        self.whole_record_file = os.path.join(raw_dir, f'{self.base_name}_bottomTraced_wholeRecord.h5')
        with h5py.File(self.input_file_path, 'a') as raw_h5:
            self.total_time = raw_h5['time'].shape[0]
            if 'qaqc_depth_line' not in raw_h5:
                full_data = np.column_stack((np.arange(self.total_time), np.full(self.total_time, np.nan, dtype = float)))
                raw_h5.create_dataset('qaqc_depth_line', data=full_data, maxshape=(self.total_time, 2))  
            self._ensure_whole_record_initialized()
            self.menu_frame.pack_forget()
            self.annotation_frame.pack(fill='both', expand=True)
            self.idx_start = 0
            self.depth_option.set('Off')
            for child in self.depth_frame.winfo_children():
                child.config(state='normal')
            for child in self.y_axis_frame.winfo_children():
                child.config(state='normal')
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
        """Loads the current slice, updates per-slice state, and redraws the display."""
        if not self.input_file_path:
            return
        with h5py.File(self.input_file_path, 'r') as f:
            self.total_slices = int(np.ceil(self.total_time / self.chunk_size))
            self.slice_number = self.idx_start // self.chunk_size + 1
            end_idx = min(self.idx_start + self.chunk_size, self.total_time)
            # Guard against stepping past the end
            if end_idx <= self.idx_start:
                return
            idx = slice(self.idx_start, end_idx)
            # --- Load image for this slice ---
            self.image = f['profile_data'][:, idx]
            # --- Check whether this slice has prior QC in whole_record.h5 ---
            # NaN  = missing / never QC'd; -999 = omitted / intentionally cleared; >= 0 = valid QAQC depth
            self.slice_already_qcd = False
            try:
                with h5py.File(self.whole_record_file, "r", locking=False) as hf:
                    if "qaqc_depth_line" in hf:
                        whole_vals = hf["qaqc_depth_line"][idx, 1].astype(float)
                        # Prior QC means anything non-missing counts: valid values and omitted values both mean the slice was QC'd before
                        self.slice_already_qcd = np.any(~np.isnan(whole_vals))
                    else:
                        self.slice_already_qcd = False
            except Exception as e:
                print(f"[WARN] Could not determine QC status from whole_record: {e}")
                self.slice_already_qcd = False
            # --- Load existing QAQC depth line for this slice from raw file ---
            if 'qaqc_depth_line' in f:
                raw_qaqc_vals = f['qaqc_depth_line'][idx, 1].astype(float)
                # QAQC exists for this slice if anything is non-missing (valid depth OR omitted both count as existing QAQC)
                self.has_existing_qaqc_in_slice = np.any(~np.isnan(raw_qaqc_vals))
                # Plotting/editing copy: keep omitted values hidden on the plot by converting -999 -> NaN
                qaqc_vals = raw_qaqc_vals.copy()
                qaqc_vals[qaqc_vals == -999] = np.nan
            else:
                raw_qaqc_vals = None
                qaqc_vals = None
                self.has_existing_qaqc_in_slice = False
            # --- Persist backscatter slice to whole_record.h5 ---
            try:
                with h5py.File(self.whole_record_file, "a", locking=False) as hf:
                    if "profile_data" in hf:
                        hf["profile_data"][:, self.idx_start:end_idx] = self.image
            except Exception as e:
                print(f"[WARN] Could not write profile_data slice to whole_record: {e}")
            # --- Initialize first-slice Y zoom exactly once (full depth) ---
            if not getattr(self, "first_image_loaded", False):
                self.user_ylim = (0.0, float(self.image.shape[0]))
                self.first_image_loaded = True
                self.user_zoom_set = False
            # --- Load optional depth products ---
            if 'smooth_depth_m' in f:
                self.smooth_depth = f['smooth_depth_m'][idx]
                if 'length_mm' in f:
                    self.length_mm = f['length_mm'][idx]
                    if len(self.length_mm) > 0 and self.image.shape[0] > 0:
                        self.bin_size = self.length_mm[0] / 1000.0 / self.image.shape[0]
                    else:
                        self.bin_size = None
                else:
                    self.length_mm = None
                    self.bin_size = None
                self.this_ping_depth_m = f['this_ping_depth_m'][idx] if 'this_ping_depth_m' in f else None
            else:
                self.smooth_depth = None
                self.length_mm = None
                self.this_ping_depth_m = None
                self.bin_size = None
        # --- Convert depth products to image-bin units if possible ---
        if self.bin_size is not None:
            self.smooth_depth_img = self.smooth_depth / self.bin_size if self.smooth_depth is not None else None
            self.this_ping_depth_img = self.this_ping_depth_m / self.bin_size if self.this_ping_depth_m is not None else None
            if self.smooth_depth_img is not None:
                self.smooth_depth_img[self.smooth_depth < self.data_blanking_distance_cm / 100] = np.nan
            if self.this_ping_depth_img is not None:
                self.this_ping_depth_img[self.this_ping_depth_m < self.data_blanking_distance_cm / 100] = np.nan
        else:
            self.smooth_depth_img = None
            self.this_ping_depth_img = None
        # --- Store QAQC image line for plotting/editing ---
        if qaqc_vals is not None:
            self.qaqc_depth_img = qaqc_vals
        else:
            self.qaqc_depth_img = None
        # --- Enable/disable QAQC radio based on whether this slice has QAQC data/history ---
        if self.has_existing_qaqc_in_slice:
            self.qaqc_radio.config(state='normal')
        else:
            self.qaqc_radio.config(state='disabled', disabledforeground='gray')
        # --- Default display choice for this slice ---
        if self.has_existing_qaqc_in_slice:
            self.depth_option.set('QAQC Depth')
        elif self.this_ping_depth_img is not None:
            self.depth_option.set('Ping Depth')
        elif self.smooth_depth_img is not None:
            self.depth_option.set('Smooth Depth')
        else:
            self.depth_option.set('Off')
        # --- Reset per-slice state ---
        self.edit_mode = False
        self.edited_line = None
        self.qaqc_modified = False
        self.all_green_edits = None
        self.update_display()

    def update_display(self):
        """Redraw the current slice, preserving user zoom."""
        # If called before first slice data is available, don't draw / don't sync limits
        if self.image is None:
            return
        # Initialize first-slice default zoom exactly once (full depth)
        if not getattr(self, "first_image_loaded", False):
            self.user_ylim = (0.0, float(self.image.shape[0]))
            self.first_image_loaded = True
        # Only "store current ylim" after the user has explicitly set zoom.
        # This avoids capturing matplotlib's startup default (0–1).
        if getattr(self, "user_zoom_set", False):
            self._store_current_ylim()
        self.ax.clear()
        if self.use_log_scale:
            plot_image = self.image.astype(float).copy()
            plot_image[plot_image <= 0] = np.nan
            if np.all(np.isnan(plot_image)):
                messagebox.showerror("Log Scale Error", "This slice has no positive values to display on a log scale.")
                self.use_log_scale = False
                self.ax.pcolormesh(self.image, cmap='plasma')
            else:
                norm = LogNorm(vmin=np.nanmin(plot_image), vmax=np.nanmax(plot_image))
                self.ax.pcolormesh(plot_image, cmap='plasma', norm=norm)
        else:
            self.ax.pcolormesh(self.image, cmap='plasma')
        self.slice_length = self.image.shape[1]
        self.ax.tick_params(axis='y', which='both', labelleft=True, labelright=False)
        self.ax.set_xlim(-0.5, self.slice_length + 0.5)
        # Apply target ylim (first slice full depth, later user zoom)
        self._apply_target_ylim()
        self.ax.set_xlabel('Ping Count', fontsize=15)
        self.ax.set_ylabel('Bin #', fontsize=15)
        self.ax.tick_params(axis='x', labelsize=14)
        self.ax.tick_params(axis='y', labelsize=14)
        if self.depth_option.get() != 'Off':
            self.plot_depth()
        for txt in self.fig.texts[:]:
            txt.remove()
        idxS = self.idx_start
        idxE = min(self.idx_start + self.chunk_size - 1, self.total_time - 1)
        qc_status = "Yes" if self.slice_already_qcd else "No"
        qc_color = "green" if self.slice_already_qcd else "red"
        # Clear previous text (important to avoid stacking)
        self.fig.texts.clear()
        # Base slice info
        self.fig.text(
            0.01, 0.98,
            f'Slice #{self.slice_number} of {self.total_slices}\n'
            f'Time Indices: {idxS} - {idxE}',
            horizontalalignment='left',
            verticalalignment='top',
            fontsize=12,
            color='black'
        )
        # Draw the label first
        label_text = self.fig.text(
            0.01, 0.938,
            'Prior QC: ',
            horizontalalignment='left',
            verticalalignment='top',
            fontsize=12,
            color='black'
        )
        # --- Compute exact position for Yes/No ---
        self.canvas.draw()  # needed to compute text size
        renderer = self.canvas.get_renderer()
        bbox = label_text.get_window_extent(renderer=renderer)
        # Convert pixel width → figure coordinates
        fig_width_pixels = self.fig.get_size_inches()[0] * self.fig.dpi
        x_offset = bbox.width / fig_width_pixels
        # Draw Yes/No immediately after label
        self.fig.text(
            0.01 + x_offset,
            0.938,
            qc_status,
            horizontalalignment='left',
            verticalalignment='top',
            fontsize=12,
            color=qc_color
        )
        self.add_secondary_y_axis()
        self.canvas.draw()
        self.coordinates = []
        self.tracing = False
        self.last_x, self.last_y = (None, None)
        self.enable_annotation()
        image_array = np.array(self.fig.canvas.renderer.buffer_rgba())
        self.image_for_saving = Image.fromarray(image_array)
        if self.depth_option.get() != 'Off' and (not self.edit_mode):
            self.clear_button.grid_remove()
        self.update_button_states()
        self.jump_slice_var.set(str(self.slice_number))

    def add_secondary_y_axis(self):
        """Add a secondary Y axis showing depth in meters."""
        # Remove existing secondary Y axis if it exists
        if hasattr(self, 'secondary_y') and self.secondary_y in self.fig.axes:
            self.fig.delaxes(self.secondary_y)
            self.secondary_y = None

        # Add new secondary Y axis if bin_size is valid
        if self.bin_size is not None:
            # Primary axis should not label the right side when using a secondary axis
            self.ax.tick_params(axis='y', which='both', labelright=False)
            self.secondary_y = self.ax.twinx()
            self.secondary_y.set_ylim(
                self.ax.get_ylim()[0] * self.bin_size,
                self.ax.get_ylim()[1] * self.bin_size
            )
            self.secondary_y.set_ylabel('Depth Range (m)', fontsize=15)
            self.secondary_y.tick_params(axis='y', labelsize=14)

    def update_button_states(self):
        """Updates internal state or display elements."""
        if self.depth_option.get() == 'Off':
            self.clear_button.grid()
            self.clear_button.config(text='Clear Annotations', command=self.clear_annotations, state='normal')
            self.save_button.config(text='Apply Traced Line', command=self.apply_traced_line, state='normal')
            self.logscale_button.config(state='normal')
            self.save_depth_button.config(text='Save Manual Depth Line', command=self.save_data, state='disabled')
            self.lasso_button.config(text='', command=None, state='disabled')
            self.ax.set_title('Manual Tracing Mode Enabled:\nLeft click to trace the depth line in green.', fontsize=16)
        elif not self.edit_mode:
            self.clear_button.grid_remove()
            self.logscale_button.config(state='normal')
            self.save_button.config(text=f'Save {self.depth_option.get()} Line', command=self.save_depth_line, state='normal')
            self.save_depth_button.config(text=f'Edit {self.depth_option.get()} Line', command=self.enter_editing_mode, state='normal')
            self.lasso_button.config(text=f'Clean {self.depth_option.get()} Line', command=self.activate_lasso, state='normal')
        else:
            self.clear_button.grid()
            self.logscale_button.config(state='normal')
            self.clear_button.config(text='Clear Annotations', command=self.clear_edit_mode, state='normal')
            self.save_button.config(text='Apply Edits', command=self.apply_edits, state='normal')
            self.save_depth_button.config(text='Save Edited Depth Line', command=self.save_edited_depth_line, state='disabled')
            self.lasso_button.config(text='', command=None, state='disabled')
        self.next_button_jump.config(state='normal' if self.slice_number < self.total_slices else 'disabled')
        self.prev_button_jump.config(state='normal' if self.slice_number > 1 else 'disabled')
        self.canvas.draw()

    def update_y_axis_limits(self):
        """Update y-axis limits from entry boxes, and persist them."""
        if self.image is None:
            messagebox.showerror("No data loaded", "Load a file/slice before setting Y axis limits.")
            return
        try:
            ymin = float(self.ymin_entry.get())
            ymax = float(self.ymax_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter numeric values for Y axis limits.")
            return
        # Normalize ordering
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        num_bins = self.image.shape[0]
        # Clamp to image bounds
        ymin = max(0.0, ymin)
        ymax = min(float(num_bins), ymax)
        if ymax <= ymin:
            messagebox.showerror("Invalid Range", "Y max must be greater than Y min.")
            return
        # Persist user choice and mark as intentional zoom
        self.user_ylim = (float(ymin), float(ymax))
        self.user_zoom_set = True
        self._apply_target_ylim()
        self.canvas.draw()
        
    def view_full_extent(self):
        """Set Y axis limits to full image extent (0 → image height)."""
        if self.image is None:
            messagebox.showerror("No data loaded", "Load a slice before adjusting the view.")
            return
        full_min = 0.0
        full_max = float(self.image.shape[0])
        # Persist this as the new user zoom
        self.user_ylim = (full_min, full_max)
        self.user_zoom_set = True
        self.ax.set_ylim(full_min, full_max)
        self.sync_y_axis_entries()
        self.canvas.draw()

    def sync_y_axis_entries(self):
        """Sync the y-axis entry boxes to the current axis limits."""
        if self.image is None:
            return
        y0, y1 = self.ax.get_ylim()
        if y0 > y1:
            y0, y1 = y1, y0
        # Keep entries clean / numeric
        self.ymin_entry.delete(0, tk.END)
        self.ymin_entry.insert(0, f"{y0:.0f}" if abs(y0 - round(y0)) < 1e-6 else f"{y0:.2f}")
        self.ymax_entry.delete(0, tk.END)
        self.ymax_entry.insert(0, f"{y1:.0f}" if abs(y1 - round(y1)) < 1e-6 else f"{y1:.2f}")
        
    def _store_current_ylim(self):
        """Remember the current Y limits so we can restore them after redraws."""
        if self.image is None:
            return
        y0, y1 = self.ax.get_ylim()
        if y0 > y1:
            y0, y1 = y1, y0
        # Reject matplotlib's typical startup default (0–1) unless user explicitly chose it.
        # This prevents accidental capture before the first slice is drawn.
        if (not getattr(self, "user_zoom_set", False)) and abs(y0 - 0.0) < 1e-9 and abs(y1 - 1.0) < 1e-9:
            return
        self.user_ylim = (float(y0), float(y1))

    def _get_target_ylim(self):
        """Compute the Y limits we should apply right now."""
        if self.image is None:
            # Don't force anything; caller should typically return early in this case
            return (0.0, 1.0)
        num_bins = float(self.image.shape[0])
        # If we have a stored zoom, use it; otherwise default to full depth
        if getattr(self, "user_ylim", None) is None:
            ymin, ymax = (0.0, num_bins)
        else:
            ymin, ymax = self.user_ylim
            ymin, ymax = float(ymin), float(ymax)
        # Clamp to data bounds
        ymin = max(0.0, ymin)
        ymax = min(num_bins, ymax)
        if ymax <= ymin:
            ymin, ymax = (0.0, num_bins)
        return (ymin, ymax)
    
    def _apply_target_ylim(self):
        """Apply chosen Y limits and keep UI in sync."""
        if self.image is None:
            return
        ymin, ymax = self._get_target_ylim()
        self.ax.set_ylim(ymin, ymax)
        # Keep the entry boxes matching what we actually applied
        self.sync_y_axis_entries()
        
    def toggle_logscale(self):
        """Toggles the color scale between linear and logarithmic."""
        self.use_log_scale = not self.use_log_scale
        self.update_display()

    def plot_depth(self):
        """Plots the selected depth line on the current plot."""
        if self.depth_option.get() == 'QAQC Depth':
            data = self.qaqc_depth_img
        elif self.depth_option.get() == 'Smooth Depth':
            data = self.smooth_depth_img
        elif self.depth_option.get() == 'Ping Depth':
            data = self.this_ping_depth_img
        else:
            return
        x = np.arange(0, self.slice_length)
        if data is not None:
            alpha_val = 1.0 if not self.edit_mode else 0.35
            self.ax.plot(x, data[:self.slice_length], color='blue', linewidth=2,
                        alpha=alpha_val, label=self.depth_option.get())
            self.ax.legend(loc='upper right')
            self.canvas.draw()

    def prev_slice(self):
        """Navigates between slices of sonar data."""
        self.clear_annotations()
        self.manual_line_saved = False
        self.edit_mode = False
        for child in self.depth_frame.winfo_children():
            child.config(state='normal')
        for child in self.y_axis_frame.winfo_children():
            child.config(state='normal')
        self.idx_start -= self.chunk_size
        self.process_next_chunk()

    def next_slice(self):
        """Navigates between slices of sonar data."""
        if self.slice_number is not None and self.slice_number >= self.total_slices:
            response = messagebox.askyesno('All slices annotated', 'All slices have been annotated. Are you done editing?')
            if response:
                self.show_final_image_progress()
            return
        self.clear_annotations()
        self.manual_line_saved = False
        self.edit_mode = False
        self.y_update_button.config(state='normal')
        for child in self.depth_frame.winfo_children():
            child.config(state='normal')
        for child in self.y_axis_frame.winfo_children():
            child.config(state='normal')
        self.idx_start += self.chunk_size
        self.process_next_chunk()

    def jump_to_slice(self):
        """Jumps to a specific data slice based on user input."""
        try:
            target_slice = int(self.jump_slice_var.get())
        except ValueError:
            messagebox.showerror('Invalid Input', 'Slice number must be an integer.')
            return
        if self.total_slices is None:
            messagebox.showerror('Error', 'No file loaded.')
            return
        if target_slice < 1 or target_slice > self.total_slices:
            messagebox.showerror('Invalid Slice', f'Slice number must be between 1 and {self.total_slices}.')
            return
        self.idx_start = (target_slice - 1) * self.chunk_size
        for child in self.depth_frame.winfo_children():
            child.config(state='normal')
        for child in self.y_axis_frame.winfo_children():
            child.config(state='normal')
        self.process_next_chunk()

    def quit_gui(self):
        """Gracefully terminates the GUI application."""
        if hasattr(self, '_closing') and self._closing:
            return
        self._closing = True
        self.unbind_all_events()
        try:
            if self.canvas and self.canvas.get_tk_widget().winfo_exists():
                self.canvas.get_tk_widget().unbind('<Destroy>')  # prevent callback loop
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
        events = ['<Button-1>', '<B1-Motion>', '<Button-3>', '<B3-Motion>', '<ButtonRelease-1>', '<ButtonRelease-3>']
        for event in events:
            self.canvas_widget.unbind(event)

    def enable_annotation(self):
        """Handles functionality related to sonar data tracing and annotation."""
        if not self.manual_line_saved:
            if self.depth_option.get() == 'Off':
                self.canvas_widget.bind('<Button-1>', self.start_tracing)
                self.canvas_widget.bind('<B1-Motion>', self.trace_line)
                self.canvas_widget.bind('<Button-3>', self.stop_tracing)
            elif self.edit_mode:
                self.canvas_widget.bind('<Button-1>', self.start_tracing_editing_green)
                self.canvas_widget.bind('<B1-Motion>', self.trace_line_editing_green)
                self.canvas_widget.bind('<Button-3>', self.start_tracing_editing_red)
                self.canvas_widget.bind('<B3-Motion>', self.trace_line_editing_red)
                self.canvas_widget.bind('<ButtonRelease-1>', self.stop_tracing)
                self.canvas_widget.bind('<ButtonRelease-3>', self.stop_tracing)
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
            child.config(state='disabled')
        self.last_x, self.last_y = (event.x, event.y)
        self.ymin_entry.config(state='disabled')
        self.ymax_entry.config(state='disabled')
        self.y_update_button.config(state='disabled')
        self.logscale_button.config(state='normal')
        self.coordinates.append((event.x, event.y, *self.canvas_to_data(event.x, event.y), 'green'))
        if self.depth_option.get() == 'Off':
            self.save_button.config(text='Apply Traced Line', command=self.apply_traced_line, state='normal')

    def trace_line(self, event):
        """Records line drawing as the user moves the mouse."""
        if self.tracing:
            data_x, data_y = self.canvas_to_data(event.x, event.y)
            x_min, x_max = self.ax.get_xlim()
            y_min, y_max = self.ax.get_ylim()
            if not (x_min <= data_x <= x_max and y_min <= data_y <= y_max):
                self.stop_tracing(event)
                return
            self.canvas_widget.create_line(self.last_x, self.last_y, event.x, event.y, fill='lime', width=2, tags='annotation')
            if not self.coordinates or (event.x, event.y) != (self.coordinates[-1][0], self.coordinates[-1][1]):
                self.coordinates.append((event.x, event.y, data_x, data_y, 'green'))
            self.last_x, self.last_y = (event.x, event.y)

    def stop_tracing(self, event):
        """Stops the tracing operation."""
        self.tracing = False

    def start_tracing_editing_green(self, event):
        """Begins the tracing operation for user input."""
        self.tracing = True
        for child in self.y_axis_frame.winfo_children():
            child.config(state='disabled')
        self.last_x, self.last_y = (event.x, event.y)
        self.ymin_entry.config(state='disabled')
        self.ymax_entry.config(state='disabled')
        self.y_update_button.config(state='disabled')
        self.logscale_button.config(state='normal')
        x_data, y_data = self.canvas_to_data(event.x, event.y)
        self.coordinates.append((event.x, event.y, x_data, y_data, 'green'))

    def trace_line_editing_green(self, event):
        """Records line drawing as the user moves the mouse."""
        if self.tracing:
            data_x, data_y = self.canvas_to_data(event.x, event.y)
            if not (self.ax.get_xlim()[0] <= data_x <= self.ax.get_xlim()[1] and self.ax.get_ylim()[0] <= data_y <= self.ax.get_ylim()[1]):
                self.stop_tracing(event)
                return
            self.canvas_widget.create_line(self.last_x, self.last_y, event.x, event.y, fill='lime', width=2, tags='annotation')
            if not self.coordinates or (event.x, event.y) != (self.coordinates[-1][0], self.coordinates[-1][1]):
                self.coordinates.append((event.x, event.y, data_x, data_y, 'green'))
            self.last_x, self.last_y = (event.x, event.y)

    def start_tracing_editing_red(self, event):
        """Begins the tracing operation for user input."""
        self.tracing = True
        for child in self.y_axis_frame.winfo_children():
            child.config(state='disabled')
        self.last_x, self.last_y = (event.x, event.y)
        self.ymin_entry.config(state='disabled')
        self.ymax_entry.config(state='disabled')
        self.y_update_button.config(state='disabled')
        self.logscale_button.config(state='normal')
        x_data, y_data = self.canvas_to_data(event.x, event.y)
        self.coordinates.append((event.x, event.y, x_data, y_data, 'red'))

    def trace_line_editing_red(self, event):
        """Records line drawing as the user moves the mouse."""
        if self.tracing:
            data_x, data_y = self.canvas_to_data(event.x, event.y)
            if not (self.ax.get_xlim()[0] <= data_x <= self.ax.get_xlim()[1] and self.ax.get_ylim()[0] <= data_y <= self.ax.get_ylim()[1]):
                self.stop_tracing(event)
                return
            self.canvas_widget.create_line(self.last_x, self.last_y, event.x, event.y, fill='red', width=2, tags='annotation')
            if not self.coordinates or (event.x, event.y) != (self.coordinates[-1][0], self.coordinates[-1][1]):
                self.coordinates.append((event.x, event.y, data_x, data_y, 'red'))
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
        self.canvas_widget.delete('annotation')
        if self.applied_line:
            self.update_display()

    def clear_edit_mode(self):
        """Clears annotations or resets state."""
        self.clear_annotations()
        self.ax.clear()
        if self.use_log_scale:
            plot_image = self.image.astype(float).copy()
            plot_image[plot_image <= 0] = np.nan
            norm = LogNorm(vmin=np.nanmin(plot_image), vmax=np.nanmax(plot_image))
            self.ax.pcolormesh(plot_image, cmap='plasma', norm=norm)
        else:
            self.ax.pcolormesh(self.image, cmap='plasma')
        self.ax.set_ylim(self.ymin, self.ymax)
        if self.depth_option.get() == 'QAQC Depth':
            data = self.qaqc_depth_img
        elif self.depth_option.get() == 'Smooth Depth':
            data = self.smooth_depth_img
        elif self.depth_option.get() == 'Ping Depth':
            data = self.this_ping_depth_img
        else:
            data = None
        if data is not None:
            x = np.arange(0, self.slice_length)
            self.ax.plot(x, data, color='blue', linewidth=2, alpha=0.35, label=self.depth_option.get())
        self.ax.set_title('Editing Mode Enabled:\nLeft click to draw edits (green), Right click to omit data (red).', fontsize=16)
        self.ax.legend(loc='upper right')
        self.ax.set_xlabel('Ping Count', fontsize=15)
        self.ax.set_ylabel('Bin #', fontsize=15)
        self.ax.tick_params(axis='y', which='both', labelleft=True, labelright=False)
        self.ax.tick_params(axis='x', labelsize=14)
        self.ax.tick_params(axis='y', labelsize=14)
        self.canvas.draw()
        self.unbind_all_events()
        self.canvas_widget.bind('<Button-1>', self.start_tracing_editing_green)
        self.canvas_widget.bind('<B1-Motion>', self.trace_line_editing_green)
        self.canvas_widget.bind('<Button-3>', self.start_tracing_editing_red)
        self.canvas_widget.bind('<B3-Motion>', self.trace_line_editing_red)
        self.canvas_widget.bind('<ButtonRelease-1>', self.stop_tracing)
        self.canvas_widget.bind('<ButtonRelease-3>', self.stop_tracing)
        self.update_button_states()

    def activate_lasso(self):
        """Activates the lasso tool for cleaning the currently selected depth line."""
        self.save_depth_button.config(state="disabled")
        self.save_button.config(state="disabled")
        self.logscale_button.config(state='normal')
        if self.edit_mode or self.depth_option.get() == 'Off':
            messagebox.showwarning(
                'Unavailable',
                'Lasso cleaning is only available before editing a depth line.'
            )
            return
        option = self.depth_option.get()
        # Use the line that is actually selected
        if option == 'QAQC Depth':
            source_data = self.qaqc_depth_img
        elif option == 'Ping Depth':
            source_data = self.this_ping_depth_img
        elif option == 'Smooth Depth':
            source_data = self.smooth_depth_img
        else:
            messagebox.showerror('Error', 'No depth line selected!')
            return

        if source_data is None:
            messagebox.showerror('Error', f'{option} data not available!')
            return
        # Work on a copy so changes only become permanent when cleaning is finished
        self.depth_data = source_data[:self.slice_length].copy()
        self.points = np.column_stack((np.arange(self.slice_length), self.depth_data))
        self._lasso_changed = False
        def onselect(verts):
            path = Path(verts)
            valid_mask = ~np.isnan(self.points[:, 1])
            selected = path.contains_points(self.points)
            to_remove = selected & valid_mask
            if np.any(to_remove):
                self._lasso_changed = True
                self.depth_data[to_remove] = np.nan
                self.points[:, 1] = self.depth_data
            # Redraw cleaned points view
            self.ax.clear()
            if self.use_log_scale:
                plot_image = self.image.astype(float).copy()
                plot_image[plot_image <= 0] = np.nan
                if np.all(np.isnan(plot_image)):
                    self.ax.pcolormesh(self.image, cmap='plasma')
                else:
                    norm = LogNorm(vmin=np.nanmin(plot_image), vmax=np.nanmax(plot_image))
                    self.ax.pcolormesh(plot_image, cmap='plasma', norm=norm)
            else:
                self.ax.pcolormesh(self.image, cmap='plasma')
            cleaned_mask = ~np.isnan(self.points[:, 1])
            self.ax.plot(
                self.points[cleaned_mask, 0],
                self.points[cleaned_mask, 1],
                'o',
                markersize=4,
                color='blue',
                label=option
            )
            self.ax.set_xlim(-0.5, self.slice_length + 0.5)
            self._apply_target_ylim()
            self.ax.set_xlabel('Ping Count', fontsize=15)
            self.ax.set_ylabel('Bin #', fontsize=15)
            self.ax.tick_params(axis='x', labelsize=14)
            self.ax.tick_params(axis='y', labelsize=14)
            self.ax.legend(loc='upper right')
            self.add_secondary_y_axis()
            self.canvas.draw()
            self.start_lasso()
            
        def finish_lasso():
            # Stop lasso tool
            if self.lasso:
                self.lasso.disconnect_events()
                self.lasso = None
            # Commit cleaned data back to the currently selected line
            if option == 'QAQC Depth':
                self.qaqc_depth_img[:self.slice_length] = self.depth_data
                if self._lasso_changed:
                    self.qaqc_modified = True
            elif option == 'Ping Depth':
                self.this_ping_depth_img[:self.slice_length] = self.depth_data
            elif option == 'Smooth Depth':
                self.smooth_depth_img[:self.slice_length] = self.depth_data
            if option == 'QAQC Depth' and self._lasso_changed:
                self.qaqc_modified = True
            # Restore normal line display
            self.ax.clear()
            if self.use_log_scale:
                plot_image = self.image.astype(float).copy()
                plot_image[plot_image <= 0] = np.nan
                if np.all(np.isnan(plot_image)):
                    self.ax.pcolormesh(self.image, cmap='plasma')
                else:
                    norm = LogNorm(vmin=np.nanmin(plot_image), vmax=np.nanmax(plot_image))
                    self.ax.pcolormesh(plot_image, cmap='plasma', norm=norm)
            else:
                self.ax.pcolormesh(self.image, cmap='plasma')
            x = np.arange(self.slice_length)
            self.ax.plot(
                x,
                self.depth_data,
                linestyle='-',
                color='blue',
                linewidth=2,
                label=option
            )
            self.ax.set_title('')
            self.ax.set_xlabel('Ping Count', fontsize=15)
            self.ax.set_ylabel('Bin #', fontsize=15)
            self.ax.tick_params(axis='x', labelsize=14)
            self.ax.tick_params(axis='y', labelsize=14)
            self.ax.set_xlim(-0.5, self.slice_length + 0.5)
            self._apply_target_ylim()
            self.ax.legend(loc='upper right')
            self.add_secondary_y_axis()
            self.lasso_button.config(text=f'Clean {option} Line', command=self.activate_lasso)
            self.save_depth_button.config(state="normal")
            self.save_button.config(state="normal")
            self.canvas.draw()

        def start_lasso():
            if self.lasso:
                self.lasso.disconnect_events()
            self.ax.set_title(
                "Lasso Tool Active: Use the middle mouse button to draw around points to remove. Press 'Finish Cleaning' when done.",
                fontsize=16
            )
            self.lasso = LassoSelector(self.ax, onselect, props=dict(color='red'), useblit=True)
            self.lasso.set_active(True)
            self.canvas.draw()
        self.start_lasso = start_lasso
        self.finish_lasso = finish_lasso
        # Remove the currently plotted line and replace with points for cleaning
        for line in self.ax.lines[:]:
            if line.get_label() == option:
                line.remove()
        valid_mask = ~np.isnan(self.points[:, 1])
        self.ax.plot(
            self.points[valid_mask, 0],
            self.points[valid_mask, 1],
            'o',
            markersize=4,
            color='blue',
            label=option
        )
        self.lasso_button.config(text='Finish Cleaning', command=self.finish_lasso)
        self.start_lasso()

    def enter_editing_mode(self):
        """Enables editing mode for adjusting the depth line."""
        try:
            self.ymin = float(self.ymin_entry.get())
            self.ymax = float(self.ymax_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter numeric Y-axis limits before editing.")
            return
        self.edit_mode = True
        self.ax.clear()
        if self.use_log_scale:
            plot_image = self.image.astype(float).copy()
            plot_image[plot_image <= 0] = np.nan
            norm = LogNorm(vmin=np.nanmin(plot_image), vmax=np.nanmax(plot_image))
            self.ax.pcolormesh(plot_image, cmap='plasma', norm=norm)
        else:
            self.ax.pcolormesh(self.image, cmap='plasma')
        if self.ymin is not None and self.ymax is not None:
            self.ax.set_ylim(self.ymin, self.ymax)
        self.ax.tick_params(axis='y', which='both', labelleft=True, labelright=False)
        if self.depth_option.get() == 'QAQC Depth':
            data = self.qaqc_depth_img
        elif self.depth_option.get() == 'Smooth Depth':
            data = self.smooth_depth_img
        elif self.depth_option.get() == 'Ping Depth':
            data = self.this_ping_depth_img
        else:
            data = None
        if data is not None:
            x = np.arange(0, self.slice_length)
            self.ax.plot(x, data, color='blue', linewidth=2, alpha=0.35, label=self.depth_option.get())
        self.ax.set_title('Editing Mode Enabled:\nLeft click to draw edits (green), Right click to omit data (red).', fontsize=16)
        self.ax.legend(loc='upper right')
        self.ax.set_xlabel('Ping Count', fontsize=15)
        self.ax.set_ylabel('Bin #', fontsize=15)
        self.ax.tick_params(axis='x', labelsize=14)
        self.ax.tick_params(axis='y', labelsize=14)
        self.sync_y_axis_entries()
        self.add_secondary_y_axis()
        self.canvas.draw()
        for child in self.depth_frame.winfo_children():
            child.config(state='disabled')
        self.unbind_all_events()
        self.canvas_widget.bind('<Button-1>', self.start_tracing_editing_green)
        self.canvas_widget.bind('<B1-Motion>', self.trace_line_editing_green)
        self.canvas_widget.bind('<Button-3>', self.start_tracing_editing_red)
        self.canvas_widget.bind('<B3-Motion>', self.trace_line_editing_red)
        self.canvas_widget.bind('<ButtonRelease-1>', self.stop_tracing)
        self.canvas_widget.bind('<ButtonRelease-3>', self.stop_tracing)
        self.clear_button.grid()
        self.clear_button.config(text='Clear Annotations', command=self.clear_edit_mode, state='normal')
        self.save_button.config(text='Apply Edits', command=self.apply_edits, state='normal')
        self.save_depth_button.config(text='Save Edited Depth Line', command=self.save_edited_depth_line, state='disabled')
        self.lasso_button.config(state='disabled')
        self.logscale_button.config(state='normal')

    def apply_edits(self):
        """Applies user edits or traced data to the current slice."""
        # Use previously edited line if it exists, otherwise use factory
        if self.edited_line is not None:
            base_line = self.edited_line.copy()
        elif self.depth_option.get() == 'QAQC Depth' and self.qaqc_depth_img is not None:
            base_line = self.qaqc_depth_img[:self.slice_length].copy()
        elif self.depth_option.get() == 'Smooth Depth' and self.smooth_depth_img is not None:
            base_line = self.smooth_depth_img[:self.slice_length].copy()
        elif self.depth_option.get() == 'Ping Depth' and self.this_ping_depth_img is not None:
            base_line = self.this_ping_depth_img[:self.slice_length].copy()
        else:
            messagebox.showerror('Error', 'No depth line available!')
            return
        # Get green edits from this round
        new_green_edit = self.interpolate_coordinates_by_color('green')[:self.slice_length, 1]
        # Initialize accumulator if needed
        if self.all_green_edits is None or len(self.all_green_edits) != self.slice_length:
            self.all_green_edits = np.full(self.slice_length, np.nan)
        # Combine current round with all prior green edits
        combined_green = np.copy(self.all_green_edits)
        self.all_green_edits = combined_green
        for i in range(self.slice_length):
            if not np.isnan(new_green_edit[i]):
                combined_green[i] = new_green_edit[i]
                
        red_edit = self.interpolate_coordinates_by_color('red')[:self.slice_length, 1]

        # Apply red removals and green overrides
        merged = base_line
        for i in range(self.slice_length):
            if not np.isnan(red_edit[i]):
                merged[i] = np.nan
            elif not np.isnan(combined_green[i]):
                merged[i] = combined_green[i]

        self.edited_line = merged
        if self.depth_option.get() == 'QAQC Depth':
            self.qaqc_modified = True
        x_vals = np.arange(self.slice_length)

        self.ax.clear()
        if self.use_log_scale:
            plot_image = self.image.astype(float).copy()
            plot_image[plot_image <= 0] = np.nan
            norm = LogNorm(vmin=np.nanmin(plot_image), vmax=np.nanmax(plot_image))
            self.ax.pcolormesh(plot_image, cmap='plasma', norm=norm)
        else:
            self.ax.pcolormesh(self.image, cmap='plasma')

        # Plot the full merged result as a solid baseline
        self.ax.plot(x_vals, merged, linestyle='-', color='blue', linewidth=2, label='Edited Depth Line')

        # Always overlay new green edits on top
        green_mask = ~np.isnan(combined_green)
        manual_x = x_vals[green_mask]
        manual_y = combined_green[green_mask]
        segments = np.split(np.column_stack((manual_x, manual_y)), np.where(np.diff(manual_x) > 1)[0] + 1)
        for i, seg in enumerate(segments):
            if len(seg) > 0:
                self.ax.plot(seg[:, 0], seg[:, 1], linestyle='-', color='lime', linewidth=2,
                            label='New Manual Edits' if i == 0 else '_nolegend_')

        self.ax.tick_params(axis='y', which='both', labelleft=True, labelright=False)
        self.ax.legend(loc='upper right')
        self.ax.set_title('')
        self.ax.set_xlabel('Ping Count', fontsize=15)
        self.ax.set_ylabel('Bin #', fontsize=15)
        self.ax.tick_params(axis='x', labelsize=14)
        self.ax.tick_params(axis='y', labelsize=14)
        self._apply_target_ylim()
        self.canvas.draw()
        self.clear_annotations()
        self.unbind_all_events()
        self.save_button.config(text='Continue Editing', command=self.continue_editing, state='normal')
        self.save_depth_button.config(state='normal')

    def continue_editing(self):
        """Allows continued editing from the last applied edits."""
        self.edit_mode = True
        self.ax.clear()
        if self.use_log_scale:
            plot_image = self.image.astype(float).copy()
            plot_image[plot_image <= 0] = np.nan
            norm = LogNorm(vmin=np.nanmin(plot_image), vmax=np.nanmax(plot_image))
            self.ax.pcolormesh(plot_image, cmap='plasma', norm=norm)
        else:
            self.ax.pcolormesh(self.image, cmap='plasma')
        self._apply_target_ylim()

        x = np.arange(0, self.slice_length)
        merged = self.edited_line
        if merged is not None:
            self.ax.plot(x, merged, color='blue', linewidth=2, alpha=0.35, label='Edited Depth Line')

        self.ax.set_title("Editing Mode Enabled:\nLeft click to draw edits (green), Right click to omit data (red).", fontsize=16)
        self.ax.legend(loc="upper right")
        self.ax.set_xlabel("Ping Count", fontsize=15)
        self.ax.set_ylabel("Bin #", fontsize=15)
        self.ax.tick_params(axis='x', labelsize=14)
        self.ax.tick_params(axis='y', labelsize=14)

        self.sync_y_axis_entries()
        self.add_secondary_y_axis()
        self.canvas.draw()

        for child in self.depth_frame.winfo_children():
            child.config(state='disabled')

        self.unbind_all_events()
        self.canvas_widget.bind("<Button-1>", self.start_tracing_editing_green)
        self.canvas_widget.bind("<B1-Motion>", self.trace_line_editing_green)
        self.canvas_widget.bind("<Button-3>", self.start_tracing_editing_red)
        self.canvas_widget.bind("<B3-Motion>", self.trace_line_editing_red)
        self.canvas_widget.bind("<ButtonRelease-1>", self.stop_tracing)
        self.canvas_widget.bind("<ButtonRelease-3>", self.stop_tracing)

        self.clear_button.grid()
        self.clear_button.config(text="Clear Annotations", command=self.clear_edit_mode, state="normal")
        self.save_button.config(text="Apply Edits", command=self.apply_edits, state="normal")
        self.save_depth_button.config(text="Save Edited Depth Line", command=self.save_edited_depth_line, state="disabled")
        self.y_update_button.config(state='normal')
        self.lasso_button.config(state="disabled")
        self.logscale_button.config(state='normal')
    
    def apply_traced_line(self):
        """Applies user edits or traced data to the current slice."""
        if self.image is None:
            messagebox.showerror("Error", "No image loaded.")
            return
        if not self.coordinates:
            snap = self._snapshot_view()
            try:
                # Clear any temporary annotations
                self.canvas_widget.delete('annotation')
                # Redraw base image for the saved export view
                self.ax.clear()
                if self.use_log_scale:
                    plot_image = self.image.astype(float).copy()
                    plot_image[plot_image <= 0] = np.nan
                    norm = LogNorm(vmin=np.nanmin(plot_image), vmax=np.nanmax(plot_image))
                    self.ax.pcolormesh(plot_image, cmap='plasma', norm=norm)
                else:
                    self.ax.pcolormesh(self.image, cmap='plasma')
                # X axis should reflect current slice
                self.slice_length = self.image.shape[1]
                self.ax.set_xlim(-0.5, self.slice_length + 0.5)
                # Full extent ONLY for the exported plot
                self.ax.set_ylim(0, float(self.image.shape[0]))
                self.ax.set_xlabel('Ping Count', fontsize=15)
                self.ax.set_ylabel('Bin #', fontsize=15)
                self.ax.tick_params(axis='x', labelsize=14)
                self.ax.tick_params(axis='y', labelsize=14)
                # IMPORTANT: do NOT sync y entries here (that would overwrite user inputs)
                self.canvas.draw()
                self.save_data(omit_slice=True)
            finally:
                # Restore user's zoom so the UI snaps back immediately and persists to next slice
                self._restore_view(snap)
            return
        traced_coords = np.array(self.coordinates, dtype=float)
        traced_coords = traced_coords[np.argsort(traced_coords[:, 0])]
        # Store the traced line to your state the same way you already do
        self.traced_line = traced_coords
        # Redraw current slice + traced line without changing zoom
        self.ax.clear()
        if self.use_log_scale:
            plot_image = self.image.astype(float).copy()
            plot_image[plot_image <= 0] = np.nan
            norm = LogNorm(vmin=np.nanmin(plot_image), vmax=np.nanmax(plot_image))
            self.ax.pcolormesh(plot_image, cmap='plasma', norm=norm)
        else:
            self.ax.pcolormesh(self.image, cmap='plasma')
        self.slice_length = self.image.shape[1]
        self.ax.set_xlim(-0.5, self.slice_length + 0.5)
        # Re-apply zoom (user view)
        self._apply_target_ylim()
        self.ax.plot(traced_coords[:, 0], traced_coords[:, 1], linewidth=2)
        self.ax.set_xlabel('Ping Count', fontsize=15)
        self.ax.set_ylabel('Bin #', fontsize=15)
        self.ax.tick_params(axis='x', labelsize=14)
        self.ax.tick_params(axis='y', labelsize=14)
        self.add_secondary_y_axis()
        self.canvas.draw()
        self.coordinates = []
        self.tracing = False
        self.last_x, self.last_y = (None, None)
        self.update_button_states()

    def save_edited_depth_line(self):
        """Saves data or images to file."""
        snap = self._snapshot_view()
        try:
            if self.edited_line is None:
                messagebox.showerror('Error', 'No edited depth line available!')
                return
            num_results = self.image.shape[0]
            self.ax.set_ylim(0, num_results)
            self.canvas.draw()
            merged = self.edited_line
            idxS = self.idx_start
            x_vals = np.arange(self.slice_length)
            self.ax.clear()
            if self.use_log_scale:
                plot_image = self.image.astype(float).copy()
                plot_image[plot_image <= 0] = np.nan
                norm = LogNorm(vmin=np.nanmin(plot_image), vmax=np.nanmax(plot_image))
                self.ax.pcolormesh(plot_image, cmap='plasma', norm=norm)
            else:
                self.ax.pcolormesh(self.image, cmap='plasma')
            self.ax.plot(x_vals, merged, color='cyan', linewidth=2, label='Saved Depth Line')
            self.ax.tick_params(axis='y', which='both', labelleft=True, labelright=False)
            self.ax.set_xlim(-0.5, self.slice_length + 0.5)
            self.ax.set_ylim(0, self.image.shape[0])
            self.ax.legend(loc='upper right')
            self.ax.set_xlabel('Ping Count', fontsize=15)
            self.ax.set_ylabel('Bin #', fontsize=15)
            self.ax.tick_params(axis='x', labelsize=14)
            self.ax.tick_params(axis='y', labelsize=14)
            self.add_secondary_y_axis()
            default_filename = f'{self.base_name}_bottomTraced_{idxS}-{idxS + self.slice_length - 1}.png'
            self.ax.set_title(default_filename, fontsize=16)
            self.canvas.draw()
            self.root.update()
            buf = io.BytesIO()
            self.fig.savefig(buf, format='png', dpi=300)
            buf.seek(0)
            self.image_for_saving = Image.open(buf)
            raw_dir = os.path.dirname(self.input_file_path)
            qcplots = os.path.join(raw_dir, 'qcPlots')
            os.makedirs(qcplots, exist_ok=True)
            png_path = os.path.join(qcplots, default_filename)
            self.image_for_saving.save(png_path)
            qcddata = os.path.join(raw_dir, 'qcdData')
            os.makedirs(qcddata, exist_ok=True)
            h5_name = default_filename.replace('.png', '.h5')
            h5_path = os.path.join(qcddata, h5_name)
            with h5py.File(h5_path, 'w') as hf:
                merged_data = np.column_stack((x_vals, merged))
                hf.create_dataset('depth_line_by_slice_idx', data=merged_data)
                time_indices = np.arange(idxS, idxS + self.slice_length)
                hf.create_dataset('depth_line_by_time_idx', data=np.column_stack((time_indices, merged)))
                hf.create_dataset('profile_data_slice', data=self.image)
            time_idx = np.arange(idxS, idxS + self.slice_length)
            existing = self._get_existing_qaqc_values(time_idx)
            new_vals = merged.astype(float).copy()
            new_vals[np.isnan(new_vals)] = np.nan  # keep NaN
            # Write ONLY the elements that are actually different (NaN-safe)
            changed_mask = ~self._nan_safe_equal(existing, new_vals, tol=0.0)
            idx_to_write = time_idx[changed_mask]
            vals_to_write = new_vals[changed_mask]
            if len(idx_to_write) > 0:
                self.update_whole_record(idx_to_write, vals_to_write)
            print(f"\n============= Slice #: {str(self.slice_number).zfill(2)} ==============")
            print(f"Image saved: {os.path.normpath(png_path)}")
            print(f"Depth line saved: {os.path.normpath(h5_path)}")
            print(f"Whole record updated: {os.path.normpath(self.whole_record_file)}")
            print(f"Raw file qaqc_depth_line updated: {os.path.normpath(self.input_file_path)}")
            print("========================================\n")
            self.unbind_all_events()
        finally:
            self._restore_view(snap)
        self.slice_saved()

    def save_data(self,omit_slice = False):
        """Saves data or images to file."""
        snap = self._snapshot_view()
        try:
            self.clear_button.config(state='disabled')
            self.save_button.config(state='disabled')
            self.save_depth_button.config(state='disabled')
            self.logscale_button.config(state='disabled')
            num_results = self.image.shape[0]
            self.ax.set_ylim(0, num_results)
            idxS = self.idx_start
            time_idx = np.arange(idxS, idxS + self.slice_length)
            if omit_slice:
                print("Omitting entire slice")
                blank_vals = np.full(self.slice_length, -999.0, dtype=float)
                self.update_whole_record(time_idx, blank_vals)

                self.manual_line_saved = True
                self.edited_line = None
                self.coordinates = []
                self.unbind_all_events()

                self.slice_saved()
                return
            default_filename = f'{self.base_name}_bottomTraced_{idxS}-{idxS + self.slice_length - 1}.png'
            self.ax.set_title(default_filename, fontsize=16)
            self.canvas.draw()
            if self.image is None:
                messagebox.showerror('Error', 'No image loaded to save.')
                return
            interp_values = self.interpolate_coordinates()[:self.slice_length, :]
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
                            self.ax.plot(interp_values[seg, 0], interp_values[seg, 1], color='cyan', linewidth=2, label='Saved Depth Line')
                            first = False
                        else:
                            self.ax.plot(interp_values[seg, 0], interp_values[seg, 1], color='cyan', linewidth=2)
            else:
                self.ax.plot(np.arange(self.slice_length), np.full(self.slice_length, float(-999)), color='cyan', linewidth=2, label='Saved Depth Line')
            self.ax.legend(loc='upper right')
            for line in self.ax.get_lines():
                if line.get_label() == 'Manual Depth Line':
                    line.remove()
            self.add_secondary_y_axis()
            self.canvas.draw()
            self.root.update()
            self.image_for_saving = Image.fromarray(np.array(self.fig.canvas.renderer.buffer_rgba()))
            raw_dir = os.path.dirname(self.input_file_path)
            qcplots = os.path.join(raw_dir, 'qcPlots')
            os.makedirs(qcplots, exist_ok=True)
            png_path = os.path.join(qcplots, default_filename)
            self.image_for_saving.save(png_path)
            qcddata = os.path.join(raw_dir, 'qcdData')
            os.makedirs(qcddata, exist_ok=True)
            h5_name = default_filename.replace('.png', '.h5')
            h5_path = os.path.join(qcddata, h5_name)
            with h5py.File(h5_path, 'w') as hf:
                hf.create_dataset('depth_line_by_slice_idx', data=interp_values)
                time_indices = np.arange(idxS, idxS + self.slice_length)
                hf.create_dataset('depth_line_by_time_idx', data=np.column_stack((time_indices, interp_values[:, 1])))
                hf.create_dataset('profile_data_slice', data=self.image)
            new_vals = interp_values[:, 1].astype(float).copy()
            # Treat -999 as missing (no edit)
            new_vals[new_vals == -999] = np.nan
            # Only write where the user actually provided a value
            valid_mask = ~np.isnan(new_vals)
            idx_to_write = time_idx[valid_mask]
            vals_to_write = new_vals[valid_mask]
            print(f"Writing {len(idx_to_write)} points out of {self.slice_length}")  # optional debug
            if len(idx_to_write) > 0:
                self.update_whole_record(idx_to_write, vals_to_write)
            else:
                print("No valid points to write")
            print(f"\n============= Slice #: {str(self.slice_number).zfill(2)} ==============")
            print(f"Image saved: {os.path.normpath(png_path)}")
            print(f"Manual depth line saved: {os.path.normpath(h5_path)}")
            print(f"Whole record updated: {os.path.normpath(self.whole_record_file)}")
            print(f"Raw file qaqc_depth_line updated: {os.path.normpath(self.input_file_path)}")
            print("========================================\n")
            self.manual_line_saved = True
            self.unbind_all_events()
        finally:
            self._restore_view(snap)
        self.slice_saved()

    def save_depth_line(self):
        """Saves the currently selected depth line to PNG/H5 and updates QAQC."""
        option = self.depth_option.get()
        if option == 'QAQC Depth':
            if not self.qaqc_modified:
                proceed = messagebox.askyesno(
                    'QAQC Depth Not Modified',
                    'The QAQC Depth Line has not been modified and does not need to be saved.\n\n'
                    'Would you like to proceed to the next slice without making any changes?'
                )
                if proceed:
                    self.next_slice()
                return
            data = self.qaqc_depth_img
        elif option == 'Smooth Depth':
            data = self.smooth_depth_img
        elif option == 'Ping Depth':
            data = self.this_ping_depth_img
        else:
            messagebox.showerror('Error', 'No depth line selected!')
            return
        if data is None:
            messagebox.showerror('Error', 'Depth data not available!')
            return
        snap = self._snapshot_view()
        try:
            self.clear_button.config(state='disabled')
            self.save_button.config(state='disabled')
            self.save_depth_button.config(state='disabled')
            self.logscale_button.config(state='disabled')
            num_results = self.image.shape[0]
            self.ax.set_ylim(0, num_results)
            self.canvas.draw()
            self.sync_y_axis_entries()
            idxS = self.idx_start
            x_coords = np.arange(0, self.slice_length)
            depth_coords = data[:self.slice_length]
            self.ax.clear()
            if self.use_log_scale:
                plot_image = self.image.astype(float).copy()
                plot_image[plot_image <= 0] = np.nan
                if np.all(np.isnan(plot_image)):
                    self.ax.pcolormesh(self.image, cmap='plasma')
                else:
                    norm = LogNorm(vmin=np.nanmin(plot_image), vmax=np.nanmax(plot_image))
                    self.ax.pcolormesh(plot_image, cmap='plasma', norm=norm)
            else:
                self.ax.pcolormesh(self.image, cmap='plasma')
            self.ax.tick_params(axis='y', which='both', labelleft=True, labelright=False)
            self.ax.plot(x_coords, depth_coords, color='cyan', linewidth=2, label='Saved Depth Line')
            self.ax.legend(loc='upper right')
            self.ax.set_xlabel('Ping Count', fontsize=15)
            self.ax.set_ylabel('Bin #', fontsize=15)
            self.ax.tick_params(axis='x', labelsize=14)
            self.ax.tick_params(axis='y', labelsize=14)
            default_filename = f'{self.base_name}_bottomTraced_{idxS}-{idxS + self.slice_length - 1}.png'
            self.ax.set_title(default_filename, fontsize=16)
            self.add_secondary_y_axis()
            self.canvas.draw()
            self.root.update()
            buf = io.BytesIO()
            self.fig.savefig(buf, format='png', dpi=300)
            buf.seek(0)
            self.image_for_saving = Image.open(buf)
            raw_dir = os.path.dirname(self.input_file_path)
            qcplots = os.path.join(raw_dir, 'qcPlots')
            os.makedirs(qcplots, exist_ok=True)
            png_path = os.path.join(qcplots, default_filename)
            self.image_for_saving.save(png_path)
            qcddata = os.path.join(raw_dir, 'qcdData')
            os.makedirs(qcddata, exist_ok=True)
            h5_name = default_filename.replace('.png', '.h5')
            h5_path = os.path.join(qcddata, h5_name)
            with h5py.File(h5_path, 'w') as hf:
                merged_data = np.column_stack((x_coords, depth_coords))
                hf.create_dataset('depth_line_by_slice_idx', data=merged_data)

                time_indices = np.arange(idxS, idxS + self.slice_length)
                hf.create_dataset('depth_line_by_time_idx', data=np.column_stack((time_indices, depth_coords)))
                hf.create_dataset('profile_data_slice', data=self.image)
            time_idx = np.arange(idxS, idxS + self.slice_length)
            existing = self._get_existing_qaqc_values(time_idx)
            new_vals = depth_coords.astype(float).copy()
            new_vals[np.isnan(new_vals)] = np.nan
            if option == 'QAQC Depth':
                changed_mask = ~self._nan_safe_equal(existing, new_vals, tol=0.0)
                idx_to_write = time_idx[changed_mask]
                vals_to_write = new_vals[changed_mask]
            else:
                missing_mask = np.isnan(existing) & ~np.isnan(new_vals)
                idx_to_write = time_idx[missing_mask]
                vals_to_write = new_vals[missing_mask]
            if len(idx_to_write) > 0:
                self.update_whole_record(idx_to_write, vals_to_write)
            print(f"\n============= Slice #: {str(self.slice_number).zfill(2)} ==============")
            print(f"Image saved: {os.path.normpath(png_path)}")
            print(f"Depth line saved: {os.path.normpath(h5_path)}")
            print(f"Whole record updated: {os.path.normpath(self.whole_record_file)}")
            print(f"Raw file qaqc_depth_line updated: {os.path.normpath(self.input_file_path)}")
            print("========================================\n")
            if option == 'QAQC Depth':
                self.qaqc_modified = False
            self.manual_line_saved = True
            self.unbind_all_events()
        finally:
            self._restore_view(snap)
        self.slice_saved()

    def _snapshot_view(self):
        """Capture current view + persisted zoom settings."""
        try:
            ax_ylim = self.ax.get_ylim()
        except Exception:
            ax_ylim = None
        return {
            "user_ylim": getattr(self, "user_ylim", None),
            "user_zoom_set": getattr(self, "user_zoom_set", False),
            "ax_ylim": ax_ylim,
        }

    def _restore_view(self, snap):
        """Restore view + persisted zoom settings."""
        if snap is None:
            return
        self.user_ylim = snap.get("user_ylim", None)
        self.user_zoom_set = snap.get("user_zoom_set", False)
        # Do NOT restore x-limits.
        # X range should always be controlled by the current slice length.
        if snap.get("ax_ylim") is not None:
            self.ax.set_ylim(*snap["ax_ylim"])
        self.sync_y_axis_entries()
        self.canvas.draw()

    def slice_saved(self):
        """Handles actions after a slice has been saved."""
        if self.slice_number != self.total_slices:
            self.next_slice()
        else:
            response = messagebox.askyesno('All slices annotated', 'All slices have been annotated. Are you done editing?')
            if response:
                self.show_final_image_progress()
            
    def show_final_image_progress(self):
        """Displays a popup while generating the final whole-record image in a background thread."""
        # Create a top-level popup
        popup = tk.Toplevel(self.root)
        popup.title("Processing")
        tk.Label(popup, text="Generating final image...\nThis may take a moment.", font=("Helvetica", 14)).pack(padx=20, pady=20)
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
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        # --- Load data from whole_record ---
        with h5py.File(self.whole_record_file, "r") as h5:
            profile_data = h5["profile_data"][:]
            depth_line = h5["qaqc_depth_line"][:, 1].astype(float)
            depth_line[depth_line == -999] = np.nan

        # --- Get bin size from raw file ---
        with h5py.File(self.input_file_path, "r") as raw:
            if "length_mm" in raw:
                length_mm = raw["length_mm"][:]
                avg_length_m = np.nanmean(length_mm) / 1000.0
                bin_size = avg_length_m / profile_data.shape[0]
            else:
                bin_size = None

        fig, ax = plt.subplots(figsize=(20, 12))

        # --- Handle sentinel values for plotting ---
        plot_data = profile_data.astype(float).copy()
        plot_data[plot_data <= 0] = np.nan  # removes -999 and invalid values

        ny, nx = plot_data.shape
        x = np.arange(nx)
        y = np.arange(ny)

        # --- Plot backscatter ---
        if self.use_log_scale:
            if np.all(np.isnan(plot_data)):
                print("[WARN] No valid data for log scale — falling back to linear.")
                ax.pcolormesh(x, y, plot_data, cmap='plasma', shading='auto')
            else:
                norm = LogNorm(vmin=np.nanmin(plot_data), vmax=np.nanmax(plot_data))
                ax.pcolormesh(x, y, plot_data, cmap='plasma', norm=norm, shading='auto')
        else:
            ax.pcolormesh(x, y, plot_data, cmap='plasma', shading='auto')

        # --- Labels ---
        ax.set_title(f'{self.base_name}_bottomTraced_wholeRecord.png', fontsize=16)
        ax.set_xlabel("Ping Index", fontsize=15)
        ax.set_ylabel("Bin #", fontsize=15)

        # --- Plot QAQC depth line ---
        x_vals = np.arange(len(depth_line))
        valid = ~np.isnan(depth_line)
        ax.plot(x_vals[valid], depth_line[valid], color="cyan", linewidth=2, label='Saved Depth Line')

        # Use full profile_data height, not last slice
        ax.set_ylim(0, profile_data.shape[0])

        # --- Secondary Y-axis (depth in meters) ---
        if bin_size is not None:
            ax2 = ax.twinx()
            ax2.set_ylim(ax.get_ylim()[0] * bin_size, ax.get_ylim()[1] * bin_size)
            ax2.set_ylabel("Depth Range (m)", fontsize=15)
            ax2.tick_params(axis='y', labelsize=14)

        # --- Final styling ---
        ax.legend(loc="upper right")
        ax.tick_params(axis='both', labelsize=14)

        fig.text(
            0.01, 0.98,
            f'Whole Record\nTime Indices: 0 - {self.total_time - 1}',
            horizontalalignment='left',
            verticalalignment='top',
            fontsize=12,
            color='black'
        )

        # --- Save ---
        raw_dir = os.path.dirname(self.input_file_path)
        final_path = os.path.join(raw_dir, f"{self.base_name}_wholeRecord.png")
        fig.savefig(final_path, dpi=300)
        plt.close(fig)

        # --- Timing output ---
        end_time = time.time()
        elapsed_minutes = round((end_time - self.start_time) / 60, 2) if self.start_time else "N/A"

        print("\n============= Final Summary ==============")
        print(f"Final whole-record image saved: {os.path.normpath(final_path)}")
        print(f"Total time elapsed: {elapsed_minutes} minutes")
        print("==========================================\n")
    
    def update_whole_record(self, time_indices, depth_values):
        """Write selected QAQC values into whole_record and raw input."""
        time_indices = np.asarray(time_indices, dtype=int)
        depth_values = np.asarray(depth_values, dtype=float)
        # Normalize NaN/negative to -999 in line units
        depth_line_vals = depth_values.copy()
        # Omitted = negative values (your convention)
        omit_mask = depth_line_vals < 0
        # Keep NaN as NaN (missing) and only convert explicit omissions to -999
        depth_line_vals[omit_mask] = -999.0
        # Convert to meters, using -999 for invalid/missing
        depth_m_vals = np.full(depth_line_vals.shape, np.nan, dtype=float)
        valid_mask = depth_line_vals >= 0
        omit_mask = depth_line_vals == -999
        depth_m_vals[valid_mask] = depth_line_vals[valid_mask] * self.bin_size
        depth_m_vals[omit_mask] = -999.0
        if self.bin_size is not None:
            depth_m_vals[valid_mask] = depth_line_vals[valid_mask] * self.bin_size
        # --- whole_record.h5 ---
        with h5py.File(self.whole_record_file, 'a', locking=False) as hf:
            if 'qaqc_depth_line' not in hf:
                full_data = np.column_stack((
                    np.arange(self.total_time),
                    np.full(self.total_time, -999.0)
                ))
                hf.create_dataset('qaqc_depth_line', data=full_data, maxshape=(self.total_time, 2))
            if 'qaqc_depth_m' not in hf:
                hf.create_dataset(
                    'qaqc_depth_m',
                    data=np.full(self.total_time, -999.0),
                    maxshape=(self.total_time,)
                )
            hf['qaqc_depth_line'][time_indices, 0] = time_indices
            hf['qaqc_depth_line'][time_indices, 1] = depth_line_vals
            hf['qaqc_depth_m'][time_indices] = depth_m_vals
        # --- raw input file ---
        with h5py.File(self.input_file_path, 'a') as raw_h5:
            if 'qaqc_depth_line' not in raw_h5:
                full_data = np.column_stack((
                    np.arange(self.total_time),
                    np.full(self.total_time, -999.0)
                ))
                raw_h5.create_dataset('qaqc_depth_line', data=full_data, maxshape=(self.total_time, 2))
            if 'qaqc_depth_m' not in raw_h5:
                raw_h5.create_dataset(
                    'qaqc_depth_m',
                    data=np.full(self.total_time, -999.0),
                    maxshape=(self.total_time,)
                )
            raw_h5['qaqc_depth_line'][time_indices, 0] = time_indices
            raw_h5['qaqc_depth_line'][time_indices, 1] = depth_line_vals
            raw_h5['qaqc_depth_m'][time_indices] = depth_m_vals
                
    @property
    def input_file_data(self):
        """Returns all datasets from the input HDF5 file, handling scalar and array data."""
        if not hasattr(self, 'input_file_path') or not os.path.exists(self.input_file_path):
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

    def _nan_safe_equal(self, a, b, tol=0.0):
        """Elementwise equality where NaN == NaN and optional tolerance for floats."""
        a = np.asarray(a)
        b = np.asarray(b)
        both_nan = np.isnan(a) & np.isnan(b)
        if tol and tol > 0:
            close = np.isclose(a, b, atol=tol, rtol=0.0, equal_nan=True)
            return close
        eq = (a == b)
        eq[both_nan] = True
        # comparisons with NaN yield False, so explicitly handle:
        eq[np.isnan(a) ^ np.isnan(b)] = False
        return eq

    def _get_existing_qaqc_values(self, time_indices):
        """Fetch existing QAQC values exactly as stored: NaN = missing, -999 = omitted, >=0 = valid depth."""
        with h5py.File(self.input_file_path, "r") as h5:
            if "qaqc_depth_line" not in h5:
                return np.full(len(time_indices), np.nan, dtype=float)
            vals = h5["qaqc_depth_line"][time_indices, 1].astype(float)
        return vals

    def _ensure_whole_record_initialized(self):
        """
        Ensure whole_record.h5 exists AND (critically) is seeded from the raw file
        so re-runs don't create a blank whole-record that looks like a reset.
        """
        # Read from raw as the source of truth
        with h5py.File(self.input_file_path, "r") as raw:
            raw_qaqc_line = raw["qaqc_depth_line"][:] if "qaqc_depth_line" in raw else None
            raw_qaqc_m = raw["qaqc_depth_m"][:] if "qaqc_depth_m" in raw else None
            if "profile_data" in raw:
                pd_shape = raw["profile_data"].shape          # (n_bins, total_time)
                pd_dtype = raw["profile_data"].dtype
            else:
                pd_shape = None
                pd_dtype = None
        with h5py.File(self.whole_record_file, "a", locking=False) as hf:
            if "qaqc_depth_line" not in hf:
                if raw_qaqc_line is not None and raw_qaqc_line.shape[0] == self.total_time:
                    hf.create_dataset("qaqc_depth_line", data=raw_qaqc_line, maxshape=(self.total_time, 2), dtype = float)
                else:
                    full_data = np.column_stack((np.arange(self.total_time), np.full(self.total_time, np.nan, dtype = float)))
                    hf.create_dataset("qaqc_depth_line", data=full_data, maxshape=(self.total_time, 2), dtype = float)

            if "qaqc_depth_m" not in hf:
                if raw_qaqc_m is not None and raw_qaqc_m.shape[0] == self.total_time:
                    hf.create_dataset("qaqc_depth_m", data=raw_qaqc_m, maxshape=(self.total_time,))
                else:
                    hf.create_dataset("qaqc_depth_m", data=np.full(self.total_time, np.nan, dtype = float), maxshape=(self.total_time))
                    
            if pd_shape is not None and "profile_data" not in hf:
                n_bins, n_time = pd_shape
                # Store full record, but fill progressively as slices are processed.
                # Choose chunking that matches your typical access pattern: all bins, chunk of time.
                chunk_t = min(getattr(self, "chunk_size", 1000), n_time)
                hf.create_dataset(
                    "profile_data",
                    shape=(n_bins, n_time),
                    dtype=pd_dtype,
                    chunks=(n_bins, chunk_t),
                    compression="gzip",
                    compression_opts=4,
                    fillvalue = -999
                )

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
    return app.input_file_data    
                
if __name__ == '__main__':
    run_sonar_tracer_gui()