import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from scipy.signal import medfilt, find_peaks
import requests
import io
import h5py
from scipy.signal import medfilt, find_peaks, savgol_filter

def multi_voigt(x, *params):
    y = np.zeros_like(x, dtype=float)
    for i in range(0, len(params), 4):
        amp = params[i]
        cen = params[i+1]
        sigma = params[i+2]
        gamma = params[i+3]
        y += amp * voigt_profile(x - cen, sigma, gamma)
    return y

class FullSpectrometerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Visualizador de Espectros y Ajuste Voigt (GOLEM)")
        self.root.geometry("1200x850")
        
        self.wl_data = None
        self.spectra_data = None
        self.detected_peaks = []
        self.current_frame = 0
        self.all_records = []
        
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def _build_ui(self):
        # Panel superior: Descarga
        top_frame = ttk.Frame(self.root, padding=5)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        
        ttk.Label(top_frame, text="Shot GOLEM:").pack(side=tk.LEFT, padx=5)
        self.shot_var = tk.StringVar()
        ttk.Entry(top_frame, textvariable=self.shot_var, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Descargar Datos", command=self.download_and_process).pack(side=tk.LEFT, padx=5)
        
        self.status_var = tk.StringVar(value="Esperando entrada...")
        ttk.Label(top_frame, textvariable=self.status_var, foreground="blue").pack(side=tk.LEFT, padx=15)
        
        # Panel central: Controles de visualización y análisis
        ctrl_frame = ttk.Frame(self.root, padding=5)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X)
        
        ttk.Label(ctrl_frame, text="Frame:").pack(side=tk.LEFT, padx=5)
        self.frame_var = tk.IntVar(value=0)
        self.slider = ttk.Scale(ctrl_frame, from_=0, to=100, orient=tk.HORIZONTAL, length=300, 
                                command=self.on_slider_change)
        self.slider.pack(side=tk.LEFT, padx=5)
        self.slider.state(['disabled'])
        
        self.frame_label = ttk.Label(ctrl_frame, text="0 / 0")
        self.frame_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(ctrl_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=15)
        
        ttk.Label(ctrl_frame, text="Peak:").pack(side=tk.LEFT, padx=5)
        self.peak_var = tk.StringVar()
        self.peak_combo = ttk.Combobox(ctrl_frame, textvariable=self.peak_var, state="readonly", width=25)
        self.peak_combo.pack(side=tk.LEFT, padx=5)
        
        self.btn_plot = ttk.Button(ctrl_frame, text="Ajustar Voigt en este frame", command=self.plot_fit)
        self.btn_plot.pack(side=tk.LEFT, padx=5)
        self.btn_plot.state(['disabled'])
        
        # Panel inferior: Gráficos
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        self.fig, (self.ax_raw, self.ax_fit) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1.5, 1]})
        self.fig.tight_layout(pad=3.0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Barra de herramientas nativa para Zoom y Paneo
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        
    def download_and_process(self):
        shot = self.shot_var.get().strip()
        if not shot: return
            
        self.status_var.set(f"Descargando H5 para el shot {shot}...")
        self.root.update()
            
        urls_to_try = [
            f"http://golem.fjfi.cvut.cz/shots/{shot}/Devices/Radiation/MiniSpectrometer/IRVISUV_0.h5",
            f"http://golem.fjfi.cvut.cz/shots/{shot}/Devices/Radiation/MiniSpectrometer/HR2000+ES-a/Spectrometer_vis_0.h5",
            f"http://golem.fjfi.cvut.cz/shots/{shot}/Diagnostics/Spectroscopy/Irvis/Results/data.h5",
            f"http://golem.fjfi.cvut.cz/shots/{shot}/Diagnostics/Spectroscopy/Spectrometer/data.h5",
            f"http://golem.fjfi.cvut.cz/shots/{shot}/Diagnostics/Spectroscopy/IRVIS/data.h5"
        ]
        
        success = False
        for url in urls_to_try:
            try:
                r = requests.get(url, timeout=10)
                is_html_type = 'text/html' in r.headers.get('Content-Type', '').lower()
                is_html_content = r.text.strip().lower().startswith(('<html', '<!doctype'))
                
                if r.status_code == 200 and not is_html_type and not is_html_content:
                    file_obj = io.BytesIO(r.content)
                    with h5py.File(file_obj, 'r') as f:
                        self.wl_data = f['Wavelengths'][:]
                        self.spectra_data = f['Spectra'][:].astype(float)
                    success = True
                    break
            except Exception:
                continue
                
        if not success:
            messagebox.showerror("Error", "No se encontró un archivo .h5 válido.")
            return
        self.all_records = []
            
        max_hold = np.max(self.spectra_data, axis=0)
        clean_max_hold = np.maximum(max_hold - medfilt(max_hold, 51), 0)
        idxs, _ = find_peaks(clean_max_hold, height=100, distance=5)
        self.detected_peaks = self.wl_data[idxs]
        
        combo_values = [f"{w:.2f} nm (Max Int: {clean_max_hold[i]:.0f})" for w, i in zip(self.detected_peaks, idxs)]
        self.peak_combo['values'] = combo_values
        if combo_values: 
            self.peak_combo.current(0)
            self.btn_plot.state(['!disabled'])
            
        total_frames = self.spectra_data.shape[0] - 1
        self.slider.config(to=total_frames)
        self.slider.state(['!disabled'])
        self.slider.set(0)

        self.ax_raw.clear()
        
        self.status_var.set("Datos cargados correctamente.")
        self.update_raw_plot(0)

    def on_slider_change(self, val):
        frame = int(float(val))
        self.current_frame = frame
        self.frame_label.config(text=f"{frame} / {self.spectra_data.shape[0]-1}")
        self.update_raw_plot(frame)

    def update_raw_plot(self, frame):
        xlim = self.ax_raw.get_xlim()
        ylim = self.ax_raw.get_ylim()
        
        if xlim == (0.0, 1.0) or ylim == (0.0, 1.0):
            xlim = (self.wl_data.min(), self.wl_data.max())
            ylim = (0, np.max(self.spectra_data) * 1.05)
            
        self.ax_raw.clear()
        self.ax_raw.plot(self.wl_data, self.spectra_data[frame], color='#004466', linewidth=1)
        self.ax_raw.set_title(f"Espectro Crudo - Frame {frame}", fontsize=11)
        self.ax_raw.set_ylabel("Intensidad (A.U.)")
        self.ax_raw.grid(True, linestyle='--', alpha=0.5)
        
        self.ax_raw.set_xlim(xlim)
        self.ax_raw.set_ylim(ylim)
            
        self.canvas.draw_idle()

    def plot_fit(self):
        selection_idx = self.peak_combo.current()
        if selection_idx == -1: return
        
        target_wl = self.detected_peaks[selection_idx]
        integration_width = 11.0
        
        raw_spectrum = self.spectra_data[self.current_frame]
        dynamic_bg = medfilt(raw_spectrum, kernel_size=51)
        clean_spectrum = np.maximum(raw_spectrum - dynamic_bg, 0)
        
        mask = (self.wl_data >= target_wl - integration_width / 2) & (self.wl_data <= target_wl + integration_width / 2)
        x_roi = self.wl_data[mask]
        y_roi = clean_spectrum[mask]
        raw_roi = raw_spectrum[mask]
        
        if len(x_roi) < 5 or np.max(y_roi) < 2.0:
            messagebox.showerror("Error", "Puntos insuficientes o señal muy baja en la ventana para ajustar.")
            return
            
        
        bg_guess = np.percentile(y_roi, 10)
        SATURATION_THRESH = 16382.0

        local_max = np.max(y_roi)

        y_detect = savgol_filter(y_roi, window_length=3, polyorder=2)
        
        peaks_idx_0, _ = find_peaks(y_detect, height=bg_guess + 25.0, prominence=0.5, distance=2)
        
        y_deriv2 = savgol_filter(y_roi, window_length=5, polyorder=3, deriv=2)
        y_deriv2[raw_roi > SATURATION_THRESH * 0.5] = 0.0        
        
        y_deriv2[y_roi > local_max * 0.75] = 0.0
            
        prom_2 = max(np.std(y_deriv2) * 0.05, 0.02)
        peaks_idx_2, _ = find_peaks(-y_deriv2, prominence=prom_2, distance=2)
        
        combined_peaks = sorted(list(set(peaks_idx_0).union(set(peaks_idx_2))))
        final_peaks = []
        for p in combined_peaks:
            if y_roi[p] <= bg_guess + 25.0:
                continue
            if not final_peaks:
                final_peaks.append(p)
            else:
                if p - final_peaks[-1] < 2:
                    if y_roi[p] > y_roi[final_peaks[-1]]:
                        final_peaks[-1] = p
                else:
                    final_peaks.append(p)
                    
        peaks_idx = final_peaks
        
        if len(peaks_idx) == 0:
            self.ax_fit.clear()
            self.ax_fit.plot(x_roi, y_roi, 'ko', label="Ruido de fondo", markersize=4)
            fit_info = "Sin señal espectral\nÁrea de Emisión: 0.00"
            self.ax_fit.text(0.02, 0.85, fit_info, transform=self.ax_fit.transAxes, 
                             bbox=dict(facecolor='#f5f5f5', alpha=0.8, edgecolor='gray'))
            self.ax_fit.set_title(f"Ajuste Desglosado: {target_wl:.2f} nm", fontsize=11)
            self.ax_fit.set_xlabel("Longitud de Onda (nm)")
            self.ax_fit.set_ylabel("Intensidad Neta")
            self.ax_fit.legend(loc="upper right", fontsize='small')
            self.ax_fit.grid(True, linestyle='--', alpha=0.5)
            self.canvas.draw_idle()
            return
            
        p0 = []
        bounds_lower = []
        bounds_upper = [] 
        max_posible_area = max(local_max * 15.0, 100.0)
        
        for p in peaks_idx:
            mu = x_roi[p]
            height = max(y_roi[p], 0.1)
            
            area_guess_factor = 10.0 if raw_roi[p] >= SATURATION_THRESH * 0.90 else 2.0
            
            p0.extend([height * area_guess_factor, mu, 0.4, 0.4])
            
            bounds_lower.extend([0, mu - 0.8, 0.15, 0.15])
            bounds_upper.extend([max_posible_area, mu + 0.8, 3.0, 3.0])
            
        valid_mask = raw_roi < (SATURATION_THRESH * 0.90)
        x_fit = x_roi[valid_mask]
        y_fit = y_roi[valid_mask]

        self.ax_fit.clear()
        
        try:
            if len(x_fit) < len(p0): 
                raise RuntimeError("Puntos insuficientes para optimizar.")
                
            popt, _ = curve_fit(multi_voigt, x_fit, y_fit, p0=p0, bounds=(bounds_lower, bounds_upper), maxfev=10000)

            self.registrar_picos_frame(popt)

            x_smooth = np.linspace(x_roi.min(), x_roi.max(), 500)
            y_fit_total = multi_voigt(x_smooth, *popt)
            bg_opt = 0.0  
            
            self.ax_fit.plot(x_roi, y_roi, 'ko', label="Datos Limpios", markersize=4)
            self.ax_fit.plot(x_smooth, y_fit_total, 'r-', label="Ajuste Total", linewidth=2)
            
            min_dist = float('inf')
            best_idx = -1
            for i in range(0, len(popt), 4):
                if abs(popt[i+1] - target_wl) < min_dist:
                    min_dist = abs(popt[i+1] - target_wl)
                    best_idx = i
                    
            fit_info_lines = []
            
            for i in range(0, len(popt), 4):
                A = popt[i]
                mu_fit = popt[i+1]
                sigma_fit = popt[i+2]
                gamma_fit = popt[i+3]
                
                y_indiv = bg_opt + A * voigt_profile(x_smooth - mu_fit, sigma_fit, gamma_fit)
                
                if i == best_idx and min_dist <= 1.0:
                    self.ax_fit.plot(x_smooth, y_indiv, 'g--', linewidth=1.5, label=f"Principal ({mu_fit:.2f} nm)")
                    self.ax_fit.fill_between(x_smooth, bg_opt, y_indiv, color='green', alpha=0.3)
                    fit_info_lines.append(f"★ Principal: {mu_fit:.2f} nm | Área: {A:.2f}")
                else:
                    self.ax_fit.plot(x_smooth, y_indiv, '--', linewidth=1.2, label=f"Solapado ({mu_fit:.2f} nm)")
                    self.ax_fit.fill_between(x_smooth, bg_opt, y_indiv, alpha=0.15)
                    fit_info_lines.append(f"  Secundario: {mu_fit:.2f} nm | Área: {A:.2f}")
                    
            self.ax_fit.text(0.02, 0.95, "\n".join(fit_info_lines), transform=self.ax_fit.transAxes, 
                             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray'), fontsize=9)
                             
        except RuntimeError:
            self.ax_fit.plot(x_roi, y_roi, 'ko-', label="Datos (Fallo optimizador)")
            fit_info = "Ajuste Voigt Fallido\nÁrea de Emisión: 0.00"
            self.ax_fit.text(0.02, 0.85, fit_info, transform=self.ax_fit.transAxes, 
                             bbox=dict(facecolor='#ffe6e6', alpha=0.8, edgecolor='red'))
            
        self.ax_fit.set_title(f"Ajuste Desglosado: {target_wl:.2f} nm", fontsize=11)
        self.ax_fit.set_xlabel("Longitud de Onda (nm)")
        self.ax_fit.set_ylabel("Intensidad Neta")
        
        handles, labels = self.ax_fit.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax_fit.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize='small')
        
        self.ax_fit.grid(True, linestyle='--', alpha=0.5)
        self.canvas.draw_idle()

    def on_closing(self):
        plt.close('all')
        self.root.quit()
        self.root.destroy()

    def registrar_picos_frame(self, popt, tol=0.6):
        for i in range(0, len(popt), 4):
            A = popt[i]
            mu = popt[i+1]
            sigma = popt[i+2]
            gamma = popt[i+3]
            
            if A <= 0:
                continue
                
            duplicado = False
            for reg in self.all_records:
                if reg['frame'] == self.current_frame and abs(reg['wavelength'] - mu) < tol:
                    duplicado = True
                    break
                    
            if not duplicado:
                fwhm = 0.5346 * (2 * gamma) + np.sqrt(0.2166 * (2 * gamma)**2 + (2.35482 * sigma)**2)
                self.all_records.append({
                    'frame': self.current_frame,
                    'wavelength': round(mu, 2),
                    'area': A,
                    'fwhm': round(fwhm, 3)
                })
if __name__ == "__main__":
    root = tk.Tk()
    app = FullSpectrometerGUI(root)
    root.mainloop()