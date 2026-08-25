import tkinter as tk
from tkinter import simpledialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy import constants, signal
import numpy as np
import requests
from PIL import Image, ImageTk
import io
import os
import itertools
from scipy import interpolate
import platform
import subprocess
import pyperclip
import spectrometry_analyzer
import json
import pickle
import h5py

class PlotConfigDialog(tk.Toplevel):
    def __init__(self, parent, active_plots, on_apply):
        super().__init__(parent)
        self.title("Select Plots to Display")
        
        self.geometry("400x500") 
        self.transient(parent)
        
        self.resizable(True, True) 

        self.active_plots = active_plots
        self.on_apply = on_apply
        self.vars = {}
        
        tk.Label(self, text="Select Diagnostics to Display:", font=("Arial", 11, "bold")).pack(anchor="w", padx=15, pady=10)
        
        self.names_map = {
            'Bt': 'Toroidal Magnetic Field',
            'Ip': 'Plasma Current',
            'U_loop': 'Loop Voltage',
            'ne': 'Electron Density',
            'Position': 'Plasma Position Displacements',
            'Te': 'Electron Temperature',
            'Confinement': 'Confinement Time',
            'Spectrometry': 'Impurity Spectrometry Evolution'
        }
        
        for key, label_text in self.names_map.items():
            var = tk.BooleanVar(value=self.active_plots.get(key, True))
            self.vars[key] = var
            cb = tk.Checkbutton(self, text=label_text, variable=var, font=("Arial", 10))
            cb.pack(anchor="w", padx=25, pady=3)
            
        tk.Button(self, text="Apply & Replot", command=self.apply, font=("Arial", 10, "bold")).pack(pady=20)
        
    def apply(self):
        for key in self.vars:
            self.active_plots[key] = self.vars[key].get()
        self.on_apply()
        self.destroy()

class IonSidebarPanel(tk.Toplevel):
    def __init__(self, parent, shot_ions_dict, on_update, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.title("Ion Control by Shot")
        self.transient(parent)
        self.resizable(True, True)
        self.shot_ions_dict = shot_ions_dict
        self.on_update = on_update
        self.ion_vars = {}
        self.scale_vars = {}
        self.cursor_lines = []
        self.last_cursor_x = None

        canvas = tk.Canvas(self, borderwidth=0)
        vscrollbar = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscrollbar.set)
        vscrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=frame, anchor="nw")

        def _on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        frame.bind("<Configure>", _on_frame_configure)

        def _on_mouse_wheel(event):
            if not canvas.winfo_exists(): 
                return
            if event.delta:
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            elif event.num == 5:  # Linux
                canvas.yview_scroll(1, "units")
            elif event.num == 4:
                canvas.yview_scroll(-1, "units")
        canvas.bind_all("<MouseWheel>", _on_mouse_wheel)
        canvas.bind_all("<Button-4>", _on_mouse_wheel)
        canvas.bind_all("<Button-5>", _on_mouse_wheel)

        title = tk.Label(frame, text="Ion Panel by Shot", font=("Arial", 12, "bold"))
        title.grid(row=0, column=0, columnspan=len(shot_ions_dict), sticky="w", pady=3)

        for col, shot in enumerate(shot_ions_dict.keys()):
            lbl = tk.Label(frame, text=f"Shot {shot}", font=("Arial", 10, "bold"))
            lbl.grid(row=1, column=col, sticky="n")

        all_ions = set()
        for ions in shot_ions_dict.values():
            all_ions.update(ion for ion,_,_ in ions)
        all_ions = sorted(list(all_ions))

        for row, ion in enumerate(all_ions, start=2):
            for col, shot in enumerate(shot_ions_dict.keys()):
                f = tk.Frame(frame)
                f.grid(row=row, column=col, sticky="w")
                shot_ions = [ion_tuple for ion_tuple in shot_ions_dict[shot] if ion_tuple[0] == ion]
                if shot_ions:
                    top_5_ions_for_shot = [item[0] for item in self.shot_ions_dict[shot][:5]]
                    is_in_top_5 = (ion in top_5_ions_for_shot)
                    var = tk.BooleanVar(value=is_in_top_5)
                    scale_var = tk.DoubleVar(value=1.0)
                    key = (shot, ion)
                    self.ion_vars[key] = var
                    self.scale_vars[key] = scale_var
                    cb = tk.Checkbutton(f, text=ion, variable=var, command=self.on_update)
                    cb.pack(side=tk.LEFT)
                    tk.Label(f, text=" x ").pack(side=tk.LEFT)
                    entry = tk.Entry(f, width=5, textvariable=scale_var)
                    entry.pack(side=tk.LEFT)
                    entry.bind("<Return>", lambda e: self.on_update())
                    entry.bind("<FocusOut>", lambda e: self.on_update())
                else:
                    tk.Label(f, text="—").pack(side=tk.LEFT)

        tk.Button(frame, text="Update Plot", command=self.on_update).grid(row=len(all_ions)+2, column=0, columnspan=len(shot_ions_dict), pady=6)

    def get_active_ions_and_scales(self):
        shots = set(shot for shot, _ in self.ion_vars.keys())
        res = {shot: [] for shot in shots}
        for (shot, ion), var in self.ion_vars.items():
            if var.get():
                scale = self.scale_vars[(shot, ion)].get()
                try:
                    scale = float(scale)
                except Exception:
                    scale = 1.0
                res[shot].append((ion, scale))
        return res

class FrameViewerPanel(tk.Toplevel):
    def __init__(self, parent, shot_number, h5_path, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.title(f"Raw Frame Viewer - Shot {shot_number}")
        self.geometry("700x550")
        self.transient(parent)

        self.int_time_ms = spectrometry_analyzer.get_spectrometer_integration_time(shot_number)

        with h5py.File(h5_path, 'r') as f:
            self.wavelengths = f['Wavelengths'][:]
            self.spectra = f['Spectra'][:]

        self.total_frames = self.spectra.shape[0]
        self.max_intensity = np.max(self.spectra)

        self.fig, self.ax = plt.subplots(figsize=(6, 4), facecolor='white')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.line, = self.ax.plot(self.wavelengths, self.spectra[0], lw=1.5, color='#003f5c')
        self.ax.set_ylim(0, self.max_intensity * 1.05)
        self.ax.set_xlim(np.min(self.wavelengths), np.max(self.wavelengths))
        self.ax.set_xlabel('Wavelength (nm)', fontsize=9)
        self.ax.set_ylabel('Intensity (A.U.)', fontsize=9)
        self.ax.set_title(f'Frame 0 | Time: 0.0 - {self.int_time_ms:.1f} ms', fontsize=10)
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.fig.tight_layout()

        self.slider = tk.Scale(self, from_=0, to=self.total_frames - 1, 
                               orient=tk.HORIZONTAL, label="Frame", 
                               command=self.actualizar_frame, bg='white', length=400)
        self.slider.pack(side=tk.BOTTOM, pady=10)

    def actualizar_frame(self, val):
        frame_actual = int(val)
        self.line.set_ydata(self.spectra[frame_actual])
        t_inicio = frame_actual * self.int_time_ms
        t_fin = (frame_actual + 1) * self.int_time_ms
        self.ax.set_title(f'Frame {frame_actual} | Time: {t_inicio:.1f} - {t_fin:.1f} ms', fontsize=10)
        self.canvas.draw_idle()

class TokamakDataViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("GOLEM Tokamak Data Viewer")
        self.shots = {}
        self.current_shot = None
        self.shot_ions_for_panel = {}
        self.ion_sidebar_panel = None
        self.color_palette = ['#003f5c', '#7a5195', '#ef5675', '#ffa600']
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.spec_peak_height = 50 
        self.cursor_dynamics_enabled = False
        self.image_refs = []
        
        # Plot optimization state configuration
        self.active_plots = {
            'Bt': True, 'Ip': True, 'U_loop': True, 'ne': True,
            'Position': True, 'Te': True, 'Confinement': True, 'Spectrometry': True
        }
        self.axs_list = []

        plt.rcParams.update({
            'font.size': 14, 'axes.labelsize': 14, 'xtick.labelsize': 12,
            'ytick.labelsize': 12, 'legend.fontsize': 'small',
            'lines.linewidth': 1.4, 'axes.titlesize': 10,
            'figure.facecolor': 'white',  
            'axes.facecolor': 'white',    
            'axes.edgecolor': 'black',    
            'axes.linewidth': 0.8,        
            'grid.color': 'lightgray',    
            'grid.linestyle': '--',       
            'grid.linewidth': 0.5         
        })

        try:
            self.nist_df = spectrometry_analyzer.load_nist("NIST.xlsx")
            if self.nist_df is None:
                messagebox.showwarning("Warning", "Could not load the NIST Excel file. Spectrometry analysis will not work.")
        except Exception as e:
            self.nist_df = None
            messagebox.showerror("Error", f"Error loading NIST file: {e}")

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.top_button_frame = tk.Frame(self.main_frame)
        self.top_button_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Button(self.top_button_frame, text="Load Shot", command=self.load_shot).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(self.top_button_frame, text="Load Local Shot", command=self.load_local_shot).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(self.top_button_frame, text="Configure Plots", command=self.open_plot_config).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(self.top_button_frame, text="Clear Shots", command=self.clear_shots).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(self.top_button_frame, text="View Raw Spectra", command=self.show_frame_viewer).pack(side=tk.LEFT, padx=5, pady=5)
        self.cursor_toggle_button = tk.Button(self.top_button_frame, text="Enable Cursor Dynamics", command=self.toggle_cursor_dynamics)
        self.cursor_toggle_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.sidebar_button = tk.Button(self.top_button_frame, text="Show Ion Panel", command=self.show_ion_sidebar)
        self.sidebar_button.pack(side=tk.RIGHT, padx=5, pady=5)

        self.plot_frame = tk.Frame(self.main_frame)
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas_container = tk.Frame(self.plot_frame)
        self.canvas_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig = plt.figure(figsize=(14, 12), facecolor='white')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_container)
        canvas_widget = self.canvas.get_tk_widget()

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.canvas_container)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.data_box_label = tk.Label(self.toolbar, text="", anchor="w", justify="left", font=("Courier New", 8))
        self.data_box_label.pack(side=tk.LEFT, padx=10)

        self.canvas.draw()

        self.right_panel = tk.Frame(self.main_frame, bg='white')
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.png_frame = tk.Frame(self.right_panel, bg='white')
        self.png_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.png_label = tk.Label(self.png_frame, bg='white')
        self.png_label.pack(expand=True)

        self.cursor_lines = []
        self.spectrometry_ax = None
        self.annot = None
        self.tooltip_cid = self.canvas.mpl_connect('motion_notify_event', self.on_hover)
        self.root.after(200, self.open_plot_config)

    def on_closing(self):
        plt.close('all')
        
        self.root.quit()
        self.root.destroy()
        
        import sys
        sys.exit(0)

    def open_plot_config(self):
        PlotConfigDialog(self.root, self.active_plots, self.plot_data)

    def load_shot(self):
        shot_number = simpledialog.askinteger("Input", "Enter the shot number:", parent=self.root)
        if not shot_number: return
        try:
            local_folder = f"shot_{shot_number}"
            os.makedirs(local_folder, exist_ok=True)
            
            # Basic diagnostics are always loaded (extremely light and critical for dependencies)
            bt_data = self.load_data(f"http://golem.fjfi.cvut.cz/shots/{shot_number}/Diagnostics/BasicDiagnostics/Results/Bt.csv", f"{local_folder}/Bt.csv", ['time_ms', 'Bt'])
            ip_data = self.load_data(f"http://golem.fjfi.cvut.cz/shots/{shot_number}/Diagnostics/BasicDiagnostics/Results/Ip.csv", f"{local_folder}/Ip.csv", ['time_ms', 'Ip'])
            u_loop_data = self.load_data(f"http://golem.fjfi.cvut.cz/shots/{shot_number}/Diagnostics/BasicDiagnostics/Results/U_loop.csv", f"{local_folder}/U_loop.csv", ['time_ms', 'U_loop'])
            
            t_spec_0 = self.find_plasma_formation_time(ip_data, threshold=0.03)
            end_time = self.find_plasma_end_time(ip_data, threshold=0.50)

            # OPTIMIZATION: Conditionally request/load other datasets only if active
            ne_data = pd.DataFrame(columns=['time_ms', 'ne'])
            if self.active_plots['ne'] or self.active_plots['Confinement']:
                try:
                    ne_data = self.load_data(f"http://golem.fjfi.cvut.cz/shots/{shot_number}/Diagnostics/Interferometry/ne_lav.csv", f"{local_folder}/ne.csv", ['time_ms', 'ne'])
                except Exception:
                    print(f"Warning: Interferometry data (ne) not found for shot {shot_number}.")
                
            fast_camera_vertical_data = pd.DataFrame(columns=['time_ms', 'vertical_displacement'])
            fast_camera_radial_data = pd.DataFrame(columns=['time_ms', 'radial_displacement'])
            if self.active_plots['Position']:
                try:
                    fast_camera_vertical_data = self.load_fast_camera_data(f"http://golem.fjfi.cvut.cz/shots/{shot_number}/Diagnostics/FastCameras/Camera_Vertical/CameraVerticalPosition", 'vertical_displacement')
                except Exception:
                    print(f"Warning: Fast camera vertical not found for shot {shot_number}.")
                try:
                    fast_camera_radial_data = self.load_fast_camera_data(f"http://golem.fjfi.cvut.cz/shots/{shot_number}/Diagnostics/FastCameras/Camera_Radial/CameraRadialPosition", 'radial_displacement')
                except Exception:
                    print(f"Warning: Fast camera radial not found for shot {shot_number}.")
                
                fast_camera_vertical_data.to_csv(f"{local_folder}/fast_camera_vertical.csv", index=False)
                fast_camera_radial_data.to_csv(f"{local_folder}/fast_camera_radial.csv", index=False)

            # Derived Data (Te)
            te_data = pd.DataFrame(columns=['time_ms', 'Te_0', 'Te_avg_a'])
            if self.active_plots['Te']:
                combined_data = pd.merge(ip_data, u_loop_data, on='time_ms', how='outer')
                combined_data = pd.merge(combined_data, bt_data, on='time_ms', how='outer').interpolate().fillna(0)
                R0, a0, nu = 0.4, 0.085, 2
                combined_data['R'], combined_data['a'] = R0, a0
                combined_data['j_avg_a'] = combined_data['Ip'] * 1e3 / (np.pi * combined_data['a']**2)
                combined_data['j_0'] = combined_data['j_avg_a'] * (nu + 1)
                l_i = np.log(1.65 + 0.89 * nu)
                combined_data['L_p'] = constants.mu_0 * combined_data['R'] * (np.log(8 * combined_data['R'] / combined_data['a']) - 7/4 + l_i / 2)
                dt = np.diff(combined_data['time_ms'].values[:2]).item() if len(combined_data['time_ms'].values) > 1 else 1.0
                n_win = max(5, int(0.5 / dt) + (1 - int(0.5 / dt) % 2)) if dt > 0 else 5
                combined_data['dIp_dt'] = signal.savgol_filter(combined_data['Ip'] * 1e3, n_win, 3, 1, delta=dt * 1e-3)
                combined_data['E_phi'] = (combined_data['U_loop'] - combined_data['L_p'] * combined_data['dIp_dt']) / (2 * np.pi * combined_data['R'])
                combined_data['eta_0'] = combined_data['E_phi'] / combined_data['j_0'].replace(0, np.nan)
                combined_data['eta_avg_a'] = combined_data['E_phi'] / combined_data['j_avg_a'].replace(0, np.nan)
                combined_data['Te_0'] = self.electron_temperature_Spitzer_eV(combined_data['eta_0'], eps=combined_data['a']/combined_data['R'])
                combined_data['Te_avg_a'] = self.electron_temperature_Spitzer_eV(combined_data['eta_avg_a'], eps=combined_data['a']/combined_data['R'])
                te_data = combined_data[['time_ms', 'Te_0', 'Te_avg_a']]
                te_data.to_csv(f"{local_folder}/Te.csv", index=False)
            
            # Confinement calculations
            confinement_time_data = pd.DataFrame(columns=['time_ms', 'tau'])
            if self.active_plots['Confinement'] and not ne_data.empty:
                Ip_interp = interpolate.interp1d(ip_data['time_ms'], ip_data['Ip'], bounds_error=False, fill_value=np.nan)(ne_data['time_ms'])
                U_l_interp = interpolate.interp1d(u_loop_data['time_ms'], u_loop_data['U_loop'], bounds_error=False, fill_value=np.nan)(ne_data['time_ms'])
                valid_idx = (ne_data['ne'] > 0) & (Ip_interp > 0) & (U_l_interp > 0)
                tau = (1.0345 * ne_data['ne'][valid_idx]) / (16e19 * Ip_interp[valid_idx]**(1/3) * U_l_interp[valid_idx]**(5/3))
                confinement_time_data = pd.DataFrame({'time_ms': ne_data['time_ms'][valid_idx], 'tau': tau.values})
                confinement_time_data.to_csv(f"{local_folder}/confinement_time.csv", index=False)
            
            # Heavy Spectrometry File download/calculation bypass optimization
            h5_file_path = None
            shot_ions = []
            if self.active_plots['Spectrometry']:
                h5_file_path = spectrometry_analyzer.download_h5(shot_number)
                if h5_file_path and self.nist_df is not None:
                    ion_labels, wls, intens = spectrometry_analyzer._detect_main_ions_for_panel(
                        h5_file_path, self.nist_df, peak_height=self.spec_peak_height)
                    shot_ions = list(zip(ion_labels, wls, intens))
                    self.shot_ions_for_panel[shot_number] = shot_ions
                    with open(f"{local_folder}/spectrometry_metadata.json", "w") as f:
                        json.dump(shot_ions, f)

            shot_data = {
                'Bt': bt_data, 'Ip': ip_data, 'U_loop': u_loop_data, 'ne': ne_data,
                'fast_camera_vertical': fast_camera_vertical_data, 'fast_camera_radial': fast_camera_radial_data,
                'Te': te_data, 'confinement_time': confinement_time_data, 'h5_path': h5_file_path,
                'formation_time': t_spec_0, 'end_time': end_time, 'shot_ions': shot_ions
            }
            
            with open(f"{local_folder}/shot_data.pkl", "wb") as f:
                pickle.dump(shot_data, f)
            
            self.shots[shot_number] = shot_data
            self.current_shot = shot_number
            self.plot_data()
            self.root.after(100, lambda: self.load_png_image(shot_number, local_folder))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load shot {shot_number}: {e}")

    def on_xlim_changed(self, event):
        if getattr(self, "ignore_xlim_callback", False):
            return
        if hasattr(self, 'axs_list'):
            for ax in self.axs_list:
                try:
                    ax.relim()
                    ax.autoscale_view(scalex=False, scaley=True)
                except Exception:
                    pass
        self.canvas.draw_idle()

    def plot_data(self):
        self.fig.clear()
        
        active_keys = [k for k, active in self.active_plots.items() if active]
        N = len(active_keys)
        
        if N == 0:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "No diagnostics selected.Click 'Configure Plots' to select.", 
                    ha='center', va='center', fontsize=12, color='gray')
            ax.axis('off')
            self.canvas.draw()
            return
            
        cols = 2 if N > 1 else 1
        rows = (N + cols - 1) // cols
        
        self.axs_list = []
        share_ax = None
        
        for i, key in enumerate(active_keys):
            if i == 0:
                ax = self.fig.add_subplot(rows, cols, i + 1)
                share_ax = ax
            else:
                ax = self.fig.add_subplot(rows, cols, i + 1, sharex=share_ax)
            ax.set_facecolor('white')
            ax.callbacks.connect('xlim_changed', self.on_xlim_changed)
            self.axs_list.append(ax)
            
        color_cycle = itertools.cycle(self.color_palette)
        labels_map = {
            'Bt': 'Bt [T]', 'Ip': 'Ip [kA]', 'U_loop': 'U_loop [V]', 'ne': 'ne [m^-3]',
            'Position': 'displacement [mm]', 'Te': 'Te [eV]', 'Confinement': 'τ_e [μs]', 'Spectrometry': 'Intensity [a.u.]'
        }
        
        for shot, data in self.shots.items():
            color = next(color_cycle)
            lighter_color = self.lighter_color(color, 1.5)
            
            for idx, key in enumerate(active_keys):
                ax = self.axs_list[idx]
                
                if key == 'Bt' and not data['Bt'].empty:
                    ax.plot(data['Bt']['time_ms'], data['Bt']['Bt'], label=str(shot), color=color)
                elif key == 'Ip' and not data['Ip'].empty:
                    ax.plot(data['Ip']['time_ms'], data['Ip']['Ip'], label=str(shot), color=color)
                elif key == 'U_loop' and not data['U_loop'].empty:
                    ax.plot(data['U_loop']['time_ms'], data['U_loop']['U_loop'], label=str(shot), color=color)
                elif key == 'ne' and not data['ne'].empty:
                    ax.plot(data['ne']['time_ms'], data['ne']['ne'], label=str(shot), color=color)
                elif key == 'Position':
                    if 'fast_camera_radial' in data and not data['fast_camera_radial'].empty:
                        ax.plot(data['fast_camera_radial']['time_ms'], data['fast_camera_radial']['radial_displacement'], label=f'Δr ({shot})', color=color)
                    if 'fast_camera_vertical' in data and not data['fast_camera_vertical'].empty:
                        ax.plot(data['fast_camera_vertical']['time_ms'], data['fast_camera_vertical']['vertical_displacement'], label=f'Δv ({shot})', color=lighter_color)
                elif key == 'Te' and not data['Te'].empty:
                    ax.plot(data['Te']['time_ms'], data['Te']['Te_0'], label=f'Te_0 ({shot})', color=color)
                    ax.plot(data['Te']['time_ms'], data['Te']['Te_avg_a'], label=f'Te_avg_a ({shot})', color=lighter_color, linestyle='--')
                elif key == 'Confinement' and not data['confinement_time'].empty:
                    ax.plot(data['confinement_time']['time_ms'], data['confinement_time']['tau'] * 1e6, label=str(shot), color=color)
                elif key == 'Spectrometry' and data.get('h5_path'):
                    ions_scales_dict = {}
                    if self.ion_sidebar_panel and tk.Toplevel.winfo_exists(self.ion_sidebar_panel):
                        user_config = self.ion_sidebar_panel.get_active_ions_and_scales()
                        ions_scales_dict = user_config.get(shot, {})
                    if ions_scales_dict:
                        spectrometry_analyzer.plot_ion_evolution_on_ax(
                            ax=ax, shot_number=shot, shot_color=color, h5_path=data.get('h5_path'),
                            nist_df=self.nist_df, peak_height=self.spec_peak_height,
                            ions_to_plot=[ion for ion, _ in ions_scales_dict],
                            scaling_dict={ion: scale for ion, scale in ions_scales_dict},
                            formation_time=data.get('formation_time', 0.0),
                            end_time=data.get('end_time', float('inf'))
                        )
                    else:
                        ions_this_shot = [ion for ion,_,_ in data.get('shot_ions', [])][:5]
                        spectrometry_analyzer.plot_ion_evolution_on_ax(
                            ax=ax, shot_number=shot, shot_color=color, h5_path=data.get('h5_path'),
                            nist_df=self.nist_df, peak_height=self.spec_peak_height,
                            ions_to_plot=ions_this_shot, scaling_dict={ion:1.0 for ion in ions_this_shot},
                            formation_time=data.get('formation_time', 0.0),
                            end_time=data.get('end_time', float('inf'))
                        )
                    ax.relim()
                    ax.autoscale(axis="y")
                    _, top = ax.get_ylim()
                    ax.set_ylim(0, top)
        self.spectrometry_ax = None
        for idx, key in enumerate(active_keys):
            ax = self.axs_list[idx]
            if key == 'Spectrometry':
                self.spectrometry_ax = ax
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            
            is_last_row = (idx >= (rows - 1) * cols) or (N <= cols)
            if is_last_row:
                ax.set_xlabel('time [ms]')
            else:
                ax.tick_params(axis='x', labelbottom=False)
                
            if ax.has_data():
                if key == 'Spectrometry':
                    ax.legend(fontsize='x-small', ncol=2)
                else:
                    ax.legend(loc='best')
            ax.set_ylabel(labels_map[key], labelpad=2)

        if hasattr(self, 'annot') and self.annot is not None:
            try: self.annot.remove()
            except Exception: pass
            
        if hasattr(self, 'spectrometry_ax') and self.spectrometry_ax is not None:
            self.annot = self.spectrometry_ax.annotate(
                "", xy=(0,0), xytext=(15, 15), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=1.5, alpha=0.9),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                fontsize=10, zorder=100
            )
            self.annot.set_visible(False)
        
        self.fig.tight_layout(pad=1.0)
        self.canvas.draw()

    def clear_shots(self):
        h5_paths_to_remove = {data['h5_path'] for data in self.shots.values() if data.get('h5_path') and os.path.exists(data['h5_path'])}
        for path in h5_paths_to_remove:
            try: os.remove(path)
            except OSError as e: print(f"Error deleting file {path}: {e}")
        self.shots = {}
        self.current_shot = None
        self.shot_ions_for_panel = {}
        if self.ion_sidebar_panel and tk.Toplevel.winfo_exists(self.ion_sidebar_panel):
            self.ion_sidebar_panel.destroy()
            self.ion_sidebar_panel = None
        for widget in self.png_frame.winfo_children(): widget.destroy()
        self.image_refs.clear()
        self.plot_data()

    def load_data(self, url, local_path, column_names, sep=','):
        if os.path.exists(local_path):
            return pd.read_csv(local_path, sep=sep)
        response = requests.get(url)
        response.raise_for_status()
        data = pd.read_csv(io.StringIO(response.text), header=None, names=column_names, sep=sep)
        data.to_csv(local_path, index=False)
        return data

    def load_fast_camera_data(self, url, column_name):
        response = requests.get(url)
        response.raise_for_status()
        lines = response.text.strip().split('')
        time_ms, values = [], []
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) == 2:
                try:
                    time_val, disp_val = float(parts[0]), float(parts[1])
                    time_ms.append(time_val)
                    values.append(disp_val)
                except (ValueError, IndexError):
                    continue
        return pd.DataFrame({'time_ms': time_ms, column_name: values})

    def load_png_image(self, shot_number, local_folder):
        local_full_path = f"{local_folder}/ScreenShotAll_full.png"
        if not os.path.exists(local_full_path):
            png_url = f"http://golem.fjfi.cvut.cz/shots/{shot_number}/Diagnostics/FastCameras/ScreenShotAll.png"
            try:
                response = requests.get(png_url)
                response.raise_for_status()
                with open(local_full_path, 'wb') as f:
                    f.write(response.content)
            except requests.exceptions.RequestException as e:
                print(f"Could not load image for shot {shot_number}: {e}")
                return

        for widget in self.png_frame.winfo_children():
            if hasattr(widget, "shot_number") and widget.shot_number == shot_number:
                return

        wrapper_frame = tk.Frame(self.png_frame, bg='white', padx=10)
        wrapper_frame.pack(side=tk.TOP, pady=5)
        wrapper_frame.shot_number = shot_number

        tk.Label(wrapper_frame, text=f"Shot #{shot_number}", bg='white', fg='black', font=("Arial", 10, "bold")).pack(side=tk.TOP)

        image = Image.open(local_full_path)
        image.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(image)
        img_label = tk.Label(wrapper_frame, image=photo, bg='white')
        img_label.image = photo
        img_label.pack(side=tk.TOP, pady=5)
        self.image_refs.append(photo)
        img_label.bind("<Button-1>", lambda event, path=local_full_path: self.open_in_system_viewer(path))

    def open_in_system_viewer(self, image_path):
        try:
            if not os.path.exists(image_path): raise FileNotFoundError(f"File not found: {image_path}")
            if platform.system() == "Windows":
                os.startfile(image_path)
            elif platform.system() == "Darwin":
                subprocess.run(["open", image_path], check=True)
            else:
                subprocess.run(["xdg-open", image_path], check=True)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open image: {e}")

    def electron_temperature_Spitzer_eV(self, eta_measured, Z_eff=3, eps=0, coulomb_logarithm=14):
        if not isinstance(eta_measured, pd.Series) or eta_measured.empty:
            return pd.Series(dtype=float)
        eta_s = eta_measured / Z_eff * (1 - np.sqrt(eps))**2
        term = 1.96 * constants.epsilon_0**2 / (np.sqrt(constants.m_e) * constants.elementary_charge**2 * coulomb_logarithm)
        Te_eV = (term * eta_s)**(-2 / 3) / (constants.elementary_charge * 2 * np.pi)
        return Te_eV.replace([np.inf, -np.inf], np.nan)

    def connect_cursor_events(self):
        self.motion_cid = self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.right_click_cid = self.canvas.mpl_connect('button_press_event', self.on_right_click)

    def disconnect_cursor_events(self):
        if hasattr(self, 'motion_cid'):
            self.canvas.mpl_disconnect(self.motion_cid)
            del self.motion_cid
        if hasattr(self, 'right_click_cid'):
            self.canvas.mpl_disconnect(self.right_click_cid)
            del self.right_click_cid

    def toggle_cursor_dynamics(self):
        self.cursor_dynamics_enabled = not self.cursor_dynamics_enabled
        self.cursor_toggle_button.config(
            text="Disable Cursor Dynamics" if self.cursor_dynamics_enabled else "Enable Cursor Dynamics"
        )
        if self.cursor_dynamics_enabled:
            self.connect_cursor_events()
        else:
            for line in getattr(self, 'cursor_lines', []):
                try: line.remove()
                except Exception: pass
            self.cursor_lines.clear()
            if hasattr(self, "data_box_label"):
                self.data_box_label.config(text="")
            self.disconnect_cursor_events()
        self.canvas.draw()

    def on_hover(self, event):
        if getattr(self, 'spectrometry_ax', None) is None:
            return
            
        if event.inaxes != self.spectrometry_ax:
            if getattr(self, 'annot', None) and self.annot.get_visible():
                self.annot.set_visible(False)
                self.canvas.draw_idle()
            return
            
        is_hovering = False
        
        for line in self.spectrometry_ax.get_lines():
            label = line.get_label()
            
            if not label or label.startswith('_'):
                continue
                
            cont, _ = line.contains(event)
            if cont:
                self.annot.set_text(label)
                self.annot.xy = (event.xdata, event.ydata)
                
                line_color = line.get_color()
                self.annot.get_bbox_patch().set_edgecolor(line_color)
                
                self.annot.set_visible(True)
                self.canvas.draw_idle()
                is_hovering = True
                break
                
        if not is_hovering and getattr(self, 'annot', None) and self.annot.get_visible():
            self.annot.set_visible(False)
            self.canvas.draw_idle()
    
    def on_mouse_move(self, event):
        if not event.inaxes or not self.cursor_dynamics_enabled or not hasattr(self, 'axs_list') or not self.axs_list:
            return

        x = event.xdata
        self.last_cursor_x = x 

        for line in list(getattr(self, 'cursor_lines', [])):
            try: line.remove()
            except Exception: pass
        self.cursor_lines.clear()

        for ax in self.axs_list:
            self.cursor_lines.append(ax.axvline(x=x, color='gray', linestyle='--', linewidth=0.8))

        header = "Shot	Time(ms)	Bt(T)	Ip(kA)	ne(m-3)	Te_0(eV)	tau_e(us)"
        data_table = [header]
        for shot, data in self.shots.items():
            vals = {}
            for key, df in data.items():
                if isinstance(df, pd.DataFrame) and 'time_ms' in df.columns and not df.empty:
                    idx = (df['time_ms'] - x).abs().idxmin()
                    for col in df.columns:
                        if col != 'time_ms':
                            vals[col] = df.loc[idx, col]
            row = (f"{shot}	{x:.2f}	"
                   f"{vals.get('Bt', np.nan):.2f}	{vals.get('Ip', np.nan):.2f}	"
                   f"{vals.get('ne', np.nan):.2e}	{vals.get('Te_0', np.nan):.1f}	"
                   f"{vals.get('tau', np.nan) * 1e6:.1f}")
            data_table.append(row.replace("nan", "---"))

        if hasattr(self, "data_box_label") and self.data_box_label:
            self.data_box_label.config(text="".join(data_table))

        self.canvas.draw_idle()

    def draw_cursor_at(self, x):
        if x is None or not getattr(self, 'cursor_dynamics_enabled', False) or not hasattr(self, 'axs_list') or not self.axs_list:
            return

        for line in list(getattr(self, 'cursor_lines', [])):
            try: line.remove()
            except Exception: pass
        self.cursor_lines.clear()

        for ax in self.axs_list:
            self.cursor_lines.append(ax.axvline(x=x, color='gray', linestyle='--', linewidth=0.8))

        header = "Shot	Time(ms)	Bt(T)	Ip(kA)	ne(m-3)	Te_0(eV)	tau_e(us)"
        data_table = [header]
        for shot, data in self.shots.items():
            vals = {}
            for key, df in data.items():
                if isinstance(df, pd.DataFrame) and 'time_ms' in df.columns and not df.empty:
                    idx = (df['time_ms'] - x).abs().idxmin()
                    for col in df.columns:
                        if col != 'time_ms':
                            vals[col] = df.loc[idx, col]
            row = (f"{shot}	{x:.2f}	"
                   f"{vals.get('Bt', np.nan):.2f}	{vals.get('Ip', np.nan):.2f}	"
                   f"{vals.get('ne', np.nan):.2e}	{vals.get('Te_0', np.nan):.1f}	"
                   f"{vals.get('tau', np.nan) * 1e6:.1f}")
            data_table.append(row.replace("nan", "---"))

        if hasattr(self, "data_box_label") and self.data_box_label:
            self.data_box_label.config(text="".join(data_table))

        self.canvas.draw_idle()

    def on_right_click(self, event):
        if not event.inaxes or not self.cursor_dynamics_enabled:
           return
        if event.button != 3:
           return

        x = event.xdata
        clipboard_text = "Shot	Time(ms)	Bt(T)	Ip(kA)	ne(m-3)	Te_0(eV)	tau_e(us)"
        for shot, data in self.shots.items():
            vals = {}
            for key, df in data.items():
                if isinstance(df, pd.DataFrame) and 'time_ms' in df.columns and not df.empty:
                    idx = (df['time_ms'] - x).abs().idxmin()
                    for col in df.columns:
                        if col != 'time_ms':
                            vals[col] = df.loc[idx, col]
            row = (f"{shot}	{x:.2f}	"
                   f"{vals.get('Bt', np.nan):.2f}	{vals.get('Ip', np.nan):.2f}	"
                   f"{vals.get('ne', np.nan):.2e}	{vals.get('Te_0', np.nan):.1f}	"
                   f"{vals.get('tau', np.nan) * 1e6:.1f}")
            clipboard_text += row.replace("nan", "")
        pyperclip.copy(clipboard_text)
        messagebox.showinfo("Copied", "Cursor data copied to clipboard.")

    @staticmethod
    def lighter_color(color, factor=1.5):
        r, g, b = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
        r = min(255, int(r * factor))
        g = min(255, int(g * factor))
        b = min(255, int(b * factor))
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def find_plasma_formation_time(self, ip_data, threshold=0.03):
        if ip_data.empty or 'Ip' not in ip_data.columns:
            return 0.0
        ip_values = ip_data['Ip'].values
        time_values = ip_data['time_ms'].values
        max_ip = np.max(ip_values)
        if max_ip <= 0.2: 
            return 0.0
        idx_max_ip = np.argmax(ip_values)
        umbral_start = max_ip * threshold
        y_ip_subida = ip_values[:idx_max_ip]
        indices_apagado = np.where(y_ip_subida < umbral_start)[0]
        if len(indices_apagado) == 0:
            return 0.0
        idx_start = indices_apagado[-1] + 1
        if idx_start < len(time_values):
            return time_values[idx_start]
        return 0.0 

    def find_plasma_end_time(self, ip_data, threshold=0.50):
        if ip_data.empty or 'Ip' not in ip_data.columns:
            return float('inf')
        ip_values = ip_data['Ip'].values
        time_values = ip_data['time_ms'].values
        max_ip = np.max(ip_values)
        if max_ip <= 0.2:
            return float('inf')
        idx_max_ip = np.argmax(ip_values)
        umbral_end = max_ip * threshold
        ip_post_pico = ip_values[idx_max_ip:]
        t_ip_post_pico = time_values[idx_max_ip:]
        indices_end = np.where(ip_post_pico < umbral_end)[0]
        if len(indices_end) > 0:
            return t_ip_post_pico[indices_end[0]] + 1.0
        else:
            return t_ip_post_pico[-1] if len(t_ip_post_pico) > 0 else float('inf')

    def load_local_shot(self):
        shot_number = simpledialog.askinteger("Input", "Enter the shot number to load from local storage:", parent=self.root)
        if not shot_number: return
        local_folder = f"shot_{shot_number}"
        if not os.path.exists(local_folder):
            messagebox.showerror("Error", f"No local data found for shot {shot_number}")
            return
        try:
            pickle_path = f"{local_folder}/shot_data.pkl"
            if os.path.exists(pickle_path):
                with open(pickle_path, "rb") as f:
                    shot_data = pickle.load(f)
                self.shots[shot_number] = shot_data
                self.current_shot = shot_number
                metadata_path = f"{local_folder}/spectrometry_metadata.json"
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        self.shot_ions_for_panel[shot_number] = json.load(f)
                self.plot_data()
                self.root.after(100, lambda: self.load_png_image(shot_number, local_folder))
            else:
                bt_data = pd.read_csv(f"{local_folder}/Bt.csv")
                ip_data = pd.read_csv(f"{local_folder}/Ip.csv")
                u_loop_data = pd.read_csv(f"{local_folder}/U_loop.csv")
                ne_data = pd.read_csv(f"{local_folder}/ne.csv") if os.path.exists(f"{local_folder}/ne.csv") else pd.DataFrame(columns=['time_ms', 'ne'])
                fast_camera_vertical_data = pd.read_csv(f"{local_folder}/fast_camera_vertical.csv") if os.path.exists(f"{local_folder}/fast_camera_vertical.csv") else pd.DataFrame(columns=['time_ms', 'vertical_displacement'])
                fast_camera_radial_data = pd.read_csv(f"{local_folder}/fast_camera_radial.csv") if os.path.exists(f"{local_folder}/fast_camera_radial.csv") else pd.DataFrame(columns=['time_ms', 'radial_displacement'])
                te_data = pd.read_csv(f"{local_folder}/Te.csv") if os.path.exists(f"{local_folder}/Te.csv") else pd.DataFrame(columns=['time_ms', 'Te_0', 'Te_avg_a'])
                confinement_time_data = pd.read_csv(f"{local_folder}/confinement_time.csv") if os.path.exists(f"{local_folder}/confinement_time.csv") else pd.DataFrame(columns=['time_ms', 'tau'])
                metadata_path = f"{local_folder}/spectrometry_metadata.json"
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        self.shot_ions_for_panel[shot_number] = json.load(f)
                shot_data = {
                    'Bt': bt_data, 'Ip': ip_data, 'U_loop': u_loop_data, 'ne': ne_data,
                    'fast_camera_vertical': fast_camera_vertical_data, 'fast_camera_radial': fast_camera_radial_data,
                    'Te': te_data, 'confinement_time': confinement_time_data, 'h5_path': None,
                    'formation_time': self.find_plasma_formation_time(ip_data, threshold=0.03),
                    'end_time': self.find_plasma_end_time(ip_data, threshold=0.50),
                    'shot_ions': self.shot_ions_for_panel.get(shot_number, [])
                }
                self.shots[shot_number] = shot_data
                self.current_shot = shot_number
                self.plot_data()
                self.root.after(100, lambda: self.load_png_image(shot_number, local_folder))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load local shot {shot_number}: {e}")
    
    def show_frame_viewer(self):
        if not self.current_shot or not self.shots.get(self.current_shot):
            messagebox.showwarning("Attention", "You must load a shot first.")
            return
        h5_path = self.shots[self.current_shot].get('h5_path')
        if not h5_path or not os.path.exists(h5_path):
            messagebox.showwarning("No Spectrometry", "This shot does not have a spectrometry file (.h5).")
            return
        FrameViewerPanel(self.root, self.current_shot, h5_path)

    def show_ion_sidebar(self):
        if not self.shot_ions_for_panel:
            messagebox.showwarning("No data", "Please load a shot with Spectrometry active first.")
            return
        if self.ion_sidebar_panel and tk.Toplevel.winfo_exists(self.ion_sidebar_panel):
            self.ion_sidebar_panel.lift()
            return
        self.ion_sidebar_panel = IonSidebarPanel(self.root, self.shot_ions_for_panel, self.on_ion_panel_update)

    def on_ion_panel_update(self):
        self.plot_data()

if __name__ == "__main__":
    import ctypes
    try: ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except:
        try: ctypes.windll.user32.SetProcessDPIAware()
        except: pass
    root = tk.Tk()
    root.tk.call('tk', 'scaling', 2.0)
    app = TokamakDataViewer(root)
    root.mainloop()
