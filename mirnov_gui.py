
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import requests
import io
import csv
import csv
import threading
import os

CACHE_DIR = "cache"

# Use TkAgg backend for embedding
matplotlib.use("TkAgg")

class MirnovApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Golem Mirnov Analysis Tool")
        self.root.geometry("1400x900")

        # --- Style ---
        style = ttk.Style()
        style.theme_use('clam')
        
        # --- Variables ---
        self.shot_var = tk.IntVar(value=33087)
        self.t_spec_start = tk.DoubleVar(value=4.0)
        self.t_spec_end = tk.DoubleVar(value=16.0)
        self.spec_vmin = tk.DoubleVar(value=0.2)
        self.spec_vmax = tk.DoubleVar(value=1.0)
        self.t_psd_start = tk.DoubleVar(value=8.0)
        self.t_psd_end = tk.DoubleVar(value=13.0)
        self.t_coh_start = tk.DoubleVar(value=10.0)
        self.t_coh_end = tk.DoubleVar(value=12.5)
        self.f_mhd_low = tk.DoubleVar(value=20.0)
        self.f_mhd_high = tk.DoubleVar(value=40.0)
        self.coh_threshold = tk.DoubleVar(value=0.7)
        self.status_var = tk.StringVar(value="Ready")

        # --- Layout ---
        main_pane = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        # Control Panel (Left)
        control_frame = ttk.Frame(main_pane, padding="10")
        main_pane.add(control_frame, weight=1)

        self._create_controls(control_frame)

        # Result Panel (Right)
        result_frame = ttk.Frame(main_pane, padding="5")
        main_pane.add(result_frame, weight=5)

        self.notebook = ttk.Notebook(result_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tabs
        self.tab_spec = ttk.Frame(self.notebook)
        self.tab_coh = ttk.Frame(self.notebook)
        self.tab_mode = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_spec, text="Spectrograms & PSD")
        self.notebook.add(self.tab_coh, text="Coherence & Phase")
        self.notebook.add(self.tab_mode, text="Mode Analysis (m)")

        # Placeholders for canvases
        self.canvas_spec = None
        self.canvas_coh = None
        self.canvas_mode = None
        
        self.toolbar_spec = None
        self.toolbar_coh = None
        self.toolbar_mode = None

        # Status Bar
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _create_controls(self, parent):
        # Parameters
        row = 0
        ttk.Label(parent, text="Shot Number:", font=('u', 10, 'bold')).grid(row=row, column=0, sticky="w", pady=5); row+=1
        ttk.Entry(parent, textvariable=self.shot_var).grid(row=row, column=0, sticky="ew", pady=5); row+=1

        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(row=row, column=0, sticky="ew", pady=10); row+=1

        ttk.Label(parent, text="Spectrogram Window (ms):", font=('u', 10, 'bold')).grid(row=row, column=0, sticky="w"); row+=1
        f_spec = ttk.Frame(parent)
        f_spec.grid(row=row, column=0, sticky="ew")
        ttk.Entry(f_spec, textvariable=self.t_spec_start, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(f_spec, text="to").pack(side=tk.LEFT)
        ttk.Entry(f_spec, textvariable=self.t_spec_end, width=8).pack(side=tk.LEFT, padx=2)
        row+=1

        ttk.Label(parent, text="Spec Color Range (vmin/vmax):", font=('u', 10, 'bold')).grid(row=row, column=0, sticky="w", pady=5); row+=1
        f_clim = ttk.Frame(parent)
        f_clim.grid(row=row, column=0, sticky="ew")
        ttk.Entry(f_clim, textvariable=self.spec_vmin, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(f_clim, text="to").pack(side=tk.LEFT)
        ttk.Entry(f_clim, textvariable=self.spec_vmax, width=8).pack(side=tk.LEFT, padx=2)
        row+=1

        ttk.Label(parent, text="PSD Window (ms):", font=('u', 10, 'bold')).grid(row=row, column=0, sticky="w", pady=5); row+=1
        f_psd = ttk.Frame(parent)
        f_psd.grid(row=row, column=0, sticky="ew")
        ttk.Entry(f_psd, textvariable=self.t_psd_start, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(f_psd, text="to").pack(side=tk.LEFT)
        ttk.Entry(f_psd, textvariable=self.t_psd_end, width=8).pack(side=tk.LEFT, padx=2)
        row+=1

        ttk.Label(parent, text="Coherence Window (ms):", font=('u', 10, 'bold')).grid(row=row, column=0, sticky="w", pady=5); row+=1
        f_coh = ttk.Frame(parent)
        f_coh.grid(row=row, column=0, sticky="ew")
        ttk.Entry(f_coh, textvariable=self.t_coh_start, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(f_coh, text="to").pack(side=tk.LEFT)
        ttk.Entry(f_coh, textvariable=self.t_coh_end, width=8).pack(side=tk.LEFT, padx=2)
        row+=1

        ttk.Label(parent, text="MHD Freq Band (kHz):", font=('u', 10, 'bold')).grid(row=row, column=0, sticky="w", pady=5); row+=1
        f_band = ttk.Frame(parent)
        f_band.grid(row=row, column=0, sticky="ew")
        ttk.Entry(f_band, textvariable=self.f_mhd_low, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(f_band, text="to").pack(side=tk.LEFT)
        ttk.Entry(f_band, textvariable=self.f_mhd_high, width=8).pack(side=tk.LEFT, padx=2)
        row+=1

        ttk.Label(parent, text="Coh Threshold (>):", font=('u', 10, 'bold')).grid(row=row, column=0, sticky="w", pady=5); row+=1
        ttk.Entry(parent, textvariable=self.coh_threshold).grid(row=row, column=0, sticky="ew", pady=5); row+=1

        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(row=row, column=0, sticky="ew", pady=20); row+=1

        self.btn_run = ttk.Button(parent, text="Run Analysis", command=self.on_run)
        self.btn_run.grid(row=row, column=0, sticky="ew", ipady=5); row+=1

    def on_run(self):
        # Disable button
        self.btn_run.config(state=tk.DISABLED)
        self.status_var.set("Running analysis... Please wait.")
        
        # Get values
        try:
            params = {
                'shot': self.shot_var.get(),
                't_window_spec': (self.t_spec_start.get()*1e-3, self.t_spec_end.get()*1e-3),
                't_window_psd': (self.t_psd_start.get()*1e-3, self.t_psd_end.get()*1e-3),
                't_window_coh': (self.t_coh_start.get()*1e-3, self.t_coh_end.get()*1e-3),
                'f_band_mhd': (self.f_mhd_low.get()*1e3, self.f_mhd_high.get()*1e3),
                'spec_vmin': self.spec_vmin.get(),
                'spec_vmax': self.spec_vmax.get(),
                'coh_threshold': self.coh_threshold.get()
            }
        except Exception as e:
            messagebox.showerror("Input Error", f"Invalid input values: {e}")
            self.btn_run.config(state=tk.NORMAL)
            return

        # Run in thread
        thread = threading.Thread(target=self.run_logic, args=(params,))
        thread.start()

    def run_logic(self, params):
        try:
            results = analyze_mirnov_discharge_headless(**params)
            self.root.after(0, self.update_gui, results)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Analysis Error", str(e)))
            self.root.after(0, lambda: self.status_var.set("Error occurred."))
        finally:
            self.root.after(0, lambda: self.btn_run.config(state=tk.NORMAL))

    def update_gui(self, results):
        self.status_var.set("Analysis complete.")
        
        # Helper to clear and plot
        def embed_figure(fig, parent, canvas_attr, toolbar_attr):
            old_canvas = getattr(self, canvas_attr)
            old_toolbar = getattr(self, toolbar_attr)
            
            if old_canvas:
                old_canvas.get_tk_widget().destroy()
            if old_toolbar:
                old_toolbar.destroy()
            
            if fig is None:
                return

            canvas = FigureCanvasTkAgg(fig, master=parent)
            canvas.draw()
            
            toolbar = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
            toolbar.update()
            toolbar.pack(side=tk.BOTTOM, fill=tk.X)
            
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            setattr(self, canvas_attr, canvas)
            setattr(self, toolbar_attr, toolbar)

        embed_figure(results.get('fig1'), self.tab_spec, 'canvas_spec', 'toolbar_spec')
        embed_figure(results.get('fig2'), self.tab_coh, 'canvas_coh', 'toolbar_coh')
        embed_figure(results.get('fig3'), self.tab_mode, 'canvas_mode', 'toolbar_mode')

# ============================================================
# ANALYSIS LOGIC (Headless)
# ============================================================

def load_csv_generic(url):
    try:
        r = requests.get(url, timeout=5) # Shorter timeout for GUI responsiveness
        r.raise_for_status()
        return parse_csv_content(r.text)
    except Exception:
        return None, None

def parse_csv_content(text_content):
    try:
        data = list(csv.reader(io.StringIO(text_content)))
        arr = np.array(data, dtype=float)
        if arr.ndim == 1: arr = arr[:, None]
        if arr.shape[1] == 2:
            t, u = arr[:, 0], arr[:, 1]
        elif arr.shape[1] == 1:
            u = arr[:, 0]
            dt = 1.0/1e6
            t = np.arange(len(u))*dt
        else:
            return None, None
        u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        return t, u
    except Exception:
        return None, None

def get_mirnov_urls_candidates(s, c):
    mc = int(c[-2:])
    url1 = f"https://golem.fjfi.cvut.cz/shots/{s}/DASs/LimiterMirnovCoils/mc{mc}.csv"
    url2 = f"https://golem.fjfi.cvut.cz/shots/{s}/Diagnostics/LimiterMirnovCoils/U_mc{mc}.csv"
    return [url2, url1]

def get_toroidal_urls_candidates(s):
    url1 = f"https://golem.fjfi.cvut.cz/shots/{s}/DASs/StandardDAS/BtCoil_raw.csv"
    url2 = f"https://golem.fjfi.cvut.cz/shots/{s}/Diagnostics/PlasmaDetection/U_BtCoil.csv"
    return [url2, url1]

def load_signal(shot, identifier, urls):
    # Ensure cache dir exists
    shot_dir = os.path.join(CACHE_DIR, str(shot))
    if not os.path.exists(shot_dir):
        os.makedirs(shot_dir, exist_ok=True)
    
    file_path = os.path.join(shot_dir, f"{identifier}.csv")
    
    # 1. Try local
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            t, u = parse_csv_content(content)
            if t is not None:
                print(f"Loaded {identifier} from cache.")
                return t, u
        except Exception as e:
            print(f"Cache read error for {identifier}: {e}")

    # 2. Try download
    for url in urls:
        try:
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            content = r.text
            t, u = parse_csv_content(content)
            if t is not None:
                # Save to cache
                with open(file_path, 'w', newline='') as f:
                    f.write(content)
                print(f"Downloaded and cached {identifier}.")
                return t, u
        except Exception:
            continue
            
    return None, None

def select_window(t, x, tmin, tmax):
    m = (t >= tmin) & (t <= tmax)
    return t[m], x[m]

def preprocess_signal(x, fs, fmin=1e3, fmax=150e3):
    # Ensure finite
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - np.mean(x)
    x = signal.detrend(x)
    sos = signal.butter(4, [fmin/(fs/2), fmax/(fs/2)], btype='band', output='sos')
    return signal.sosfiltfilt(sos, x)

def preprocess_for_coherence(x, fs):
    # Ensure finite
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - np.mean(x)
    x = signal.detrend(x)
    sos = signal.butter(4, [1e3/(fs/2), 150e3/(fs/2)], btype='band', output='sos')
    return signal.sosfiltfilt(sos, x)

def periodogram_simple(x, fs):
    N = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(N, d=1/fs)
    PSD = (np.abs(X)**2) / N
    return f, PSD

def analyze_mirnov_discharge_headless(
    shot,
    t_window_spec=(4e-3, 16e-3),
    t_window_psd=(8e-3, 13e-3),
    t_window_coh=(10e-3, 12.5e-3),
    f_band_mhd=(20e3, 40e3),
    coils_active=None,
    spec_vmin=0.2,
    spec_vmax=1.0,
    coh_threshold=0.7
):
    # Setup
    if coils_active is None:
        coils = {
            "MC01": {"label": "MC-out",  "polarity": -1.0},
            "MC09": {"label": "MC-in",   "polarity": -1.0},
            "MC05": {"label": "MC-up",   "polarity": -1.0},
            "MC13": {"label": "MC-down", "polarity": -1.0},
        }
    else:
        coils = coils_active # Assume valid dict passed if not None

    K_C = {"MC01": 166.736, "MC05": 76.68, "MC09": 105.598, "MC13": 0.0}

    # Data Loading
    t_tf, u_tf_raw = load_signal(shot, "BtCoil", get_toroidal_urls_candidates(shot))
    if t_tf is None:
        raise ValueError(f"Could not load Bt data for shot {shot}")

    t_mir = None
    U_corr_full = {}
    valid_coils = {}

    for cname, props in coils.items():
        candidates = get_mirnov_urls_candidates(shot, cname)
        t_c, u_raw = load_signal(shot, cname, candidates)
        
        if t_c is None:
            continue
            
        if t_mir is None: t_mir = t_c
        
        # Correction
        u_tf_interp = np.interp(t_c, t_tf, u_tf_raw)
        kc = K_C.get(cname, 0.0)
        pol = props["polarity"]
        if kc != 0:
            u_corr = (u_raw - (1.0/kc)*u_tf_interp) * pol
        else:
            u_corr = u_raw * pol
        U_corr_full[cname] = u_corr
        valid_coils[cname] = props
        
    if not valid_coils:
        raise ValueError("No valid Mirnov coils found.")

    dt_global = np.mean(np.diff(t_mir))
    fs_global = 1.0/dt_global

    results = {}

    # 1. Spectrograms & PSD
    # Use explicit Figure object
    fig1 = Figure(figsize=(10, 3*len(valid_coils)), dpi=100)
    gs = fig1.add_gridspec(len(valid_coils), 2, width_ratios=[3, 1])
    
    X_proc_spec = {}
    for c in valid_coils:
        X_proc_spec[c] = preprocess_signal(U_corr_full[c], fs_global)

    for i, (cname, props) in enumerate(valid_coils.items()):
        label = props["label"]
        x_full = X_proc_spec[cname]
        
        # Spec
        t_seg, x_seg = select_window(t_mir, x_full, *t_window_spec)
        fs_local = 1.0/np.mean(np.diff(t_seg))
        f, tt, Sxx = signal.spectrogram(x_seg, fs=fs_local, window="hann", nperseg=1024, 
                                        noverlap=int(0.9*1024), nfft=4096, scaling="spectrum", mode="magnitude")
        
        ax_spec = fig1.add_subplot(gs[i, 0])
        mask = f <= 150e3
        Sxx_plot = Sxx[mask, :]
        Sxx_norm = Sxx_plot / np.max(Sxx_plot)
        
        im = ax_spec.pcolormesh(tt*1e3 + t_seg[0]*1e3, f[mask]*1e-3, Sxx_norm, 
                                shading="auto", vmin=spec_vmin, vmax=spec_vmax, cmap="jet")
        fig1.colorbar(im, ax=ax_spec)
        ax_spec.set_ylabel("f [kHz]")
        ax_spec.text(0.02, 0.9, label, transform=ax_spec.transAxes, color='white', fontweight='bold')
        ax_spec.axhspan(f_band_mhd[0]*1e-3, f_band_mhd[1]*1e-3, alpha=0.1, color='white')
        if i == 0: ax_spec.set_title(f"Spectrograms - Shot {shot}")
        if i == len(valid_coils)-1: ax_spec.set_xlabel("t [ms]")

        # PSD
        t_psd, x_psd = select_window(t_mir, x_full, *t_window_psd)
        f_psd, Pxx = periodogram_simple(x_psd, 1.0/np.mean(np.diff(t_psd)))
        Pxx = gaussian_filter1d(Pxx, sigma=6)
        
        ax_psd = fig1.add_subplot(gs[i, 1])
        mask_psd = f_psd <= 150e3
        ax_psd.plot(f_psd[mask_psd]*1e-3, Pxx[mask_psd], 'k', lw=1)
        ax_psd.set_xlabel("f [kHz]")
        if i == 0: ax_psd.set_title("PSD")

    fig1.tight_layout()
    results['fig1'] = fig1

    # 2. Coherence & Phase
    ref_coil = "MC01" if "MC01" in valid_coils else list(valid_coils.keys())[0]
    others = [c for c in valid_coils if c != ref_coil]
    
    if others:
        X_proc_coh = {c: preprocess_for_coherence(U_corr_full[c], fs_global) for c in valid_coils}
        t_coh_s, x_ref = select_window(t_mir, X_proc_coh[ref_coil], *t_window_coh)
        fs_coh = 1.0/np.mean(np.diff(t_coh_s))
        
        fig2 = Figure(figsize=(4*len(others), 6), dpi=100)
        # axes will be created dynamically
        
        phase_data_calc = {}

        for j, c2_name in enumerate(others):
            lbl_ref = valid_coils[ref_coil]["label"]
            lbl_trg = valid_coils[c2_name]["label"]
            
            _, x1 = select_window(t_mir, X_proc_coh[ref_coil], *t_window_coh)
            _, x2 = select_window(t_mir, X_proc_coh[c2_name], *t_window_coh)
            
            nper = min(512, len(x1))
            f_c, Cxy = signal.coherence(x1, x2, fs=fs_coh, window="hann", nperseg=nper, noverlap=nper//2)
            f_c, P12 = signal.csd(x1, x2, fs=fs_coh, window="hann", nperseg=nper, noverlap=nper//2)
            phase = np.angle(P12)
            
            mask_f = f_c <= 150e3
            
            # Coherence Plot
            ax_c = fig2.add_subplot(2, len(others), j+1)
            ax_c.plot(f_c[mask_f]*1e-3, Cxy[mask_f])
            ax_c.axvspan(f_band_mhd[0]*1e-3, f_band_mhd[1]*1e-3, alpha=0.15, ec=None)
            ax_c.set_ylim(0, 1.05)
            ax_c.set_title(f"{lbl_ref} | {lbl_trg}")
            if j==0: ax_c.set_ylabel("Coherence")
            
            # Phase Plot
            ax_p = fig2.add_subplot(2, len(others), j+1+len(others), sharex=ax_c)
            ax_p.plot(f_c[mask_f]*1e-3, phase[mask_f])
            ax_p.axvspan(f_band_mhd[0]*1e-3, f_band_mhd[1]*1e-3, alpha=0.15, ec=None)
            ax_p.set_ylim(-np.pi, np.pi)
            ax_p.set_xlabel("f [kHz]")
            if j==0: ax_p.set_ylabel("Phase [rad]")
            
            # Avg Phase Calc
            mask_band = (f_c[mask_f] >= f_band_mhd[0]) & (f_c[mask_f] <= f_band_mhd[1])
            Cxy_masked = Cxy[mask_f][mask_band]
            phase_masked = phase[mask_f][mask_band]
            
            # valid_pts = Cxy_masked > coh_threshold
            # Use mean coherence check
            if len(Cxy_masked) > 0:
                is_valid = np.mean(Cxy_masked) > coh_threshold
            else:
                is_valid = False

            if is_valid:
                # Re-calculate valid_pts for phase averaging if needed, or use all points weighted.
                # For consistency with previous logic, we use points above threshold for phase average
                # knowing that the COIL itself is valid because the MEAN is high.
                valid_pts = Cxy_masked > coh_threshold
                
                # In rare case valid_pts is empty (should not verify if mean > threshold), fallback
                if not np.any(valid_pts):
                     # If validation passed via mean but no single point > threshold (unlikely), take all
                     valid_pts = np.ones(len(Cxy_masked), dtype=bool)

                ph_avg = np.angle(np.exp(1j*phase_masked[valid_pts]).mean())
                if lbl_ref == "MC-out":
                    if "MC-out" not in phase_data_calc: phase_data_calc["MC-out"] = 0.0
                    phase_data_calc[lbl_trg] = ph_avg

        fig2.tight_layout()
        results['fig2'] = fig2
        results['phase_data'] = phase_data_calc
    else:
        results['fig2'] = None
        results['phase_data'] = {}

    # 3. Mode Analysis
    phase_data_calc = results.get('phase_data', {})
    if len(phase_data_calc) >= 3:
        theta_geom = {"MC-out": 0.0, "MC-up": np.pi/2, "MC-in": np.pi, "MC-down": -np.pi/2}
        
        x_fit, y_fit, names_found = [], [], []
        for name in ["MC-out", "MC-up", "MC-in", "MC-down"]:
            if name in phase_data_calc:
                x_fit.append(theta_geom[name])
                y_fit.append(phase_data_calc[name])
                names_found.append(name)
        
        x_arr = np.array(x_fit)
        y_arr = np.array(y_fit)
        
        m_candidates = np.arange(-6, 7)
        best_m = 0
        min_error = float('inf')
        
        for m_test in m_candidates:
            diff = np.exp(1j * y_arr) - np.exp(1j * m_test * x_arr)
            err = np.sum(np.abs(diff)**2)
            if err < min_error:
                min_error = err
                best_m = m_test
        
        y_unwrapped = []
        for th, ph in zip(x_arr, y_arr):
            target = best_m * th
            k = np.round((target - ph)/(2*np.pi))
            y_unwrapped.append(ph + 2*np.pi*k)
        y_unwrapped = np.array(y_unwrapped)
        
        m_fine = np.linalg.lstsq(x_arr[:, None], y_unwrapped, rcond=None)[0][0]
        
        # Calculate RMSE (Root Mean Squared Error) instead of R^2
        # This gives a physical sense of deviation in radians
        y_pred = m_fine * x_arr
        mse = np.mean((y_unwrapped - y_pred)**2)
        rmse = np.sqrt(mse)
        
        # Normalize RMSE to pi units for display if desired, 
        # but displaying in raw radians or pi-fractions is fine. 
        # Let's display in pi units significantly:
        rmse_pi = rmse / np.pi

        fig3 = Figure(figsize=(6, 4), dpi=100)
        ax3 = fig3.add_subplot(111)
        ax3.plot(x_arr/np.pi, y_unwrapped/np.pi, 'o', label="Data", zorder=3)
        
        # Add labels to points using UNWRAPPED y values
        for i, name in enumerate(names_found):
             ax3.text(x_arr[i]/np.pi, y_unwrapped[i]/np.pi + 0.1, 
                      name, ha='center', fontsize=9, color='blue')

        th_range = np.linspace(-1, 1, 100)*np.pi
        ax3.plot(th_range/np.pi, (m_fine*th_range)/np.pi, 'r--', label=f"Fit m={m_fine:.2f}\nRMSE={rmse_pi:.4f}$\pi$")
        ax3.set_xlabel(r"$\theta$ [$\pi$ rad]")
        ax3.set_ylabel(r"$\Phi$ [$\pi$ rad]")
        ax3.set_title(f"Mode Fit (Shot {shot}) - Integer: {best_m}")
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        results['fig3'] = fig3
        results['m_fine'] = m_fine
    else:
        results['fig3'] = None

    return results

if __name__ == "__main__":
    root = tk.Tk()
    app = MirnovApp(root)
    root.mainloop()
