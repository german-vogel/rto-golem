import os
import tempfile
import requests
import h5py
import pandas as pd
import numpy as np
import itertools
from scipy.signal import savgol_filter, find_peaks, medfilt
from numpy import trapz
from scipy.integrate import simpson
import re
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.special import voigt_profile
import warnings

# Silenciar advertencias matemáticas esperadas durante los ajustes
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=OptimizeWarning)

SPECTROMETER_IDENTIFIER = "IRVISUV_0.h5"
SPECTROMETER_URL_FMT    = "http://golem.fjfi.cvut.cz/shots/{shot_no}/Devices/Radiation/MiniSpectrometer/{identifier}"
WL_MIN, WL_MAX          = 400, 900
TOLERANCE               = 0.7
N_BASELINE_FRAMES       = 3
MAX_IONS_TO_PLOT        = 5
BASELINE_WIN            = 101
BASELINE_POLY           = 3
SMOOTH_WIN              = 5
SMOOTH_POLY             = 2

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb_color):
    rgb_color = tuple(max(0, min(255, int(c))) for c in rgb_color)
    return '#{:02x}{:02x}{:02x}'.format(rgb_color[0], rgb_color[1], rgb_color[2])

def lighten_color(hex_color, amount=0.3):
    try:
        r, g, b = hex_to_rgb(hex_color)
        r = min(255, int(r * (1 + amount)))
        g = min(255, int(g * (1 + amount)))
        b = min(255, int(b * (1 + amount)))
        return rgb_to_hex((r, g, b))
    except Exception as e:
        return hex_color

def download_h5(shot_no):
    urls_to_try = [
        f"http://golem.fjfi.cvut.cz/shots/{shot_no}/Devices/Radiation/MiniSpectrometer/IRVISUV_0.h5",
        f"http://golem.fjfi.cvut.cz/shots/{shot_no}/Devices/Radiation/MiniSpectrometer/HR2000+ES-a/Spectrometer_vis_0.h5",
        f"http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/Spectroscopy/Irvis/Results/data.h5",
        f"http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/Spectroscopy/Spectrometer/data.h5",
        f"http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/Spectroscopy/IRVIS/data.h5"
    ]

    for url in urls_to_try:
        try:
            r = requests.get(url, timeout=10) 
            if r.status_code == 200:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{shot_no}_spectrometry.h5")
                tmp.write(r.content)
                tmp.close()
                return tmp.name
        except requests.exceptions.RequestException:
            continue

    print(f"No se encontraron datos de espectrometría (.h5) para el disparo {shot_no} en ninguna ruta.")
    return None

def load_nist(file_path=None):
    if file_path is None:
        base = os.path.dirname(__file__)
        file_path = os.path.join(base, "NIST.xlsx")
    else:
        if not os.path.isabs(file_path):
            base = os.path.dirname(__file__)
            file_path = os.path.join(base, file_path)
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        df.columns = df.columns.str.strip() 
        
        df['Wavelength'] = pd.to_numeric(df['Wavelength'], errors='coerce')
        if 'RI' in df.columns:
            df['RI'] = pd.to_numeric(df['RI'], errors='coerce').fillna(0)
            
        return df.dropna(subset=['Wavelength']).reset_index(drop=True)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{file_path}'.")
        return None
    except ImportError:
        print("Error: Falta la librería 'openpyxl'. Instálala con 'pip install openpyxl'.")
        return None
    except Exception as e:
        print(f"Error inesperado al cargar el archivo NIST: {e}")
        return None

def multi_voigt(x, *params):
    y = np.full_like(x, params[0], dtype=float)
    for i in range(1, len(params), 4):
        A = params[i]
        mu = params[i+1]
        sigma = params[i+2]
        gamma = params[i+3]
        y += A * voigt_profile(x - mu, sigma, gamma)
    return y

def _map_peaks(wl_arr, signal, nist_df, peak_height, peak_distance):
    idxs, _ = find_peaks(signal, height=peak_height, distance=peak_distance)
    if not idxs.any(): return [], [], []
    wls, intensities = wl_arr[idxs], signal[idxs]
    ions, mapped_wls = [], []
    
    for wl_peak in wls:
        sel = nist_df[(nist_df['Wavelength'] >= wl_peak - TOLERANCE) & (nist_df['Wavelength'] <= wl_peak + TOLERANCE)].copy()
        
        if not sel.empty:
            sel['delta'] = np.abs(sel['Wavelength'] - wl_peak)
            sel = sel.sort_values(by='delta')
            min_delta = sel['delta'].iloc[0]
            tied_candidates = sel[sel['delta'] <= min_delta + 0.01]
            
            if len(tied_candidates) > 1 and 'RI' in tied_candidates.columns:
                winning_ion = tied_candidates.iloc[0]['Ion']
                same_ion_candidates = tied_candidates[tied_candidates['Ion'] == winning_ion]
                best = same_ion_candidates.sort_values(by='RI', ascending=False).iloc[0]
            else:
                best = sel.iloc[0]
                
            ions.append(f"{best['Ion']} ({best['Wavelength']:.1f} nm)")
            mapped_wls.append(best['Wavelength'])
        else:
            ions.append("Unknown")
            mapped_wls.append(wl_peak)
            
    return ions, mapped_wls, intensities

def _integrate_peak_robust(spectrum, wavelengths, center_wl, integration_width=6.0, prominence_thresh=0.5):
    SATURATION_THRESH = 16382.0  
    
    roi_mask = (wavelengths >= center_wl - integration_width / 2) & (wavelengths <= center_wl + integration_width / 2)
    x_roi = wavelengths[roi_mask]
    y_roi = spectrum[roi_mask]
    
    if len(x_roi) < 5 or np.max(y_roi) < 2.0: 
        return 0.0

    peaks_idx, _ = find_peaks(y_roi, prominence=prominence_thresh, distance=3)
    
    if len(peaks_idx) == 0:
        peaks_idx = [np.argmax(y_roi)]

    bg_guess = np.percentile(y_roi, 10)
    p0 = [max(0.0, bg_guess)]
    bounds_lower = [0.0]
    bounds_upper = [max(np.max(y_roi), 0.001)] 
    
    for p in peaks_idx:
        mu = x_roi[p]
        height = y_roi[p] - bg_guess
        if height < 0: 
            height = 0.1
        
        is_peak_saturated = (y_roi[p] >= SATURATION_THRESH * 0.95)
        area_guess_factor = 1.0 if is_peak_saturated else 0.3
        
        A_guess = height * area_guess_factor
        sigma_guess = 0.1
        gamma_guess = 0.1
        
        p0.extend([A_guess, mu, sigma_guess, gamma_guess])
        

        bounds_lower.extend([0, mu - 0.3, 0.001, 0.001])
        bounds_upper.extend([np.inf, mu + 0.3, 0.3, 0.4])

    valid_mask = y_roi < (SATURATION_THRESH * 0.98)
    x_fit = x_roi[valid_mask]
    y_fit = y_roi[valid_mask]

    if len(x_fit) < len(p0):
        from scipy.integrate import simpson
        y_clean = np.maximum(y_roi - bg_guess, 0)
        return max(simpson(y=y_clean, x=x_roi), 0.0)

    try:
        popt, _ = curve_fit(multi_voigt, x_fit, y_fit, p0=p0, bounds=(bounds_lower, bounds_upper), maxfev=5000)
    except RuntimeError:
        from scipy.integrate import simpson
        y_clean = np.maximum(y_roi - bg_guess, 0)
        return max(simpson(y=y_clean, x=x_roi), 0.0)

    best_area = 0.0
    min_dist = float('inf')

    # Extraer el área correspondiente estrictamente a la línea de interés
    for i in range(1, len(popt), 4):
        A = popt[i]
        mu_fit = popt[i+1]
        dist = abs(mu_fit - center_wl) 

        if dist < min_dist:
            min_dist = dist
            best_area = A

    if min_dist > 1.0:
        return 0.0

    return best_area

def get_spectrometer_integration_time(shot_no):
    urls_to_try = [
        f"http://golem.fjfi.cvut.cz/shots/{shot_no}/Devices/Radiation/MiniSpectrometer/DumpedCommunication.txt",
        f"http://golem.fjfi.cvut.cz/shots/{shot_no}/Devices/Radiation/MiniSpectrometer/HR2000+ES-a/DumpedCommunication.txt",
        f"http://golem.fjfi.cvut.cz/shots/{shot_no}/Diagnostics/Spectroscopy/DumpedCommunication.txt"
    ]
    
    texto_log = None
    for url in urls_to_try:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                texto_log = r.text
                break
        except requests.exceptions.RequestException:
            continue
            
    if not texto_log:
        print(f"Aviso: No se encontró DumpedCommunication.txt para shot {shot_no}. Usando 2.0 ms.")
        return 2.0

    aliases_prioritarios = ["IRVISUV", "VIS"]
    
    for alias in aliases_prioritarios:
        patron = rf"Setting spectrometer[^\n]+\({alias}\)[\s\S]*?Integration time is (\d+) us"
        match = re.search(patron, texto_log)
        
        if match:
            tiempo_us = float(match.group(1))
            tiempo_ms = tiempo_us / 1000.0
            return tiempo_ms

    match_fallback = re.search(r"Integration time is (\d+) us", texto_log)
    if match_fallback:
        return float(match_fallback.group(1)) / 1000.0

    return 2.0 

def plot_ion_evolution_on_ax(ax, shot_number, shot_color, h5_path, nist_df, peak_height, 
                           ions_to_plot=None, scaling_dict=None, formation_time=0.0, end_time=float('inf')):
    ax.set_xlabel("Tiempo [ms]")
    ax.set_ylabel("Intensidad (A.U.)")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    if h5_path is None or nist_df is None:
        return
    try:
        import h5py
        from scipy.signal import savgol_filter, medfilt
        
        int_time_ms = get_spectrometer_integration_time(shot_number)

        with h5py.File(h5_path, 'r') as f:
            all_wl = f['Wavelengths'][:]
            all_spectra = f['Spectra'][:].astype(float)

        time_points = all_spectra.shape[0]
        time_axis_ms = np.arange(time_points) * int_time_ms + int_time_ms

        ref_spectrum_raw = np.max(all_spectra, axis=0)
        bg_ref = savgol_filter(ref_spectrum_raw, window_length=BASELINE_WIN, polyorder=BASELINE_POLY)
        residual_ref = np.maximum(ref_spectrum_raw - bg_ref, 0)
        smooth_ref = savgol_filter(residual_ref, window_length=SMOOTH_WIN, polyorder=SMOOTH_POLY)
        mask_ref = (all_wl >= WL_MIN) & (all_wl <= WL_MAX)
        
        ions, wls, intensities_ref = _map_peaks(all_wl[mask_ref], smooth_ref[mask_ref], nist_df, peak_height, peak_distance=5)
        sorted_ions_data = sorted(zip(ions, wls, intensities_ref), key=lambda x: x[2], reverse=True)
        
        if ions_to_plot is not None and scaling_dict is not None:
            ions_to_plot_data = [item for item in sorted_ions_data if item[0] in ions_to_plot]
        else:
            ions_to_plot_data = sorted_ions_data[:MAX_IONS_TO_PLOT]
            
        if not ions_to_plot_data:
            return
            
        color_shades = [lighten_color(shot_color, amount=i * 0.2) for i in range(len(ions_to_plot_data))]
        background_static = medfilt(all_spectra[0], kernel_size=51)
        
        for i, (ion_label, center_wl, _) in enumerate(ions_to_plot_data):
            if ions_to_plot is not None and scaling_dict is not None:
                if ion_label not in ions_to_plot: continue
                scale_factor = scaling_dict.get(ion_label, 1.0)
            else:
                scale_factor = 1.0
                
            raw_integrated_intensities = []
            for frame_idx in range(time_points):
                clean_spectrum = np.maximum(all_spectra[frame_idx] - background_static, 0)
                integral = _integrate_peak_robust(clean_spectrum, all_wl, center_wl, integration_width=6.0)
                raw_integrated_intensities.append(integral * scale_factor)
                
            valid_evolution = np.array(raw_integrated_intensities)
            
            if np.max(valid_evolution) > 0:
                ion_color_shade = color_shades[i % len(color_shades)]
                label_text = ion_label
                
                ax.plot(time_axis_ms, valid_evolution, color=ion_color_shade, linestyle='-', 
                        marker='.', markersize=8, label=label_text, linewidth=1.5)
                
        ax.legend(fontsize='x-small', ncol=2)
        ax.set_ylim(bottom=0)
        
    except Exception as e:
        print(f"Error procesando el archivo H5 {h5_path} para shot {shot_number}: {e}")
        import traceback
        traceback.print_exc()

def _detect_main_ions_for_panel(h5_path, nist_df, peak_height=50):
    import h5py
    from scipy.signal import savgol_filter
    ions, wls, intens = [], [], []
    with h5py.File(h5_path, 'r') as f:
        all_wl = f['Wavelengths'][:]
        all_spectra = f['Spectra'][:].astype(float)
        
        spectrum = np.max(all_spectra, axis=0)
        
        bg = savgol_filter(spectrum, BASELINE_WIN, BASELINE_POLY)
        residual = np.maximum(spectrum - bg, 0)
        smooth = savgol_filter(residual, SMOOTH_WIN, SMOOTH_POLY)
        mask = (all_wl >= WL_MIN) & (all_wl <= WL_MAX)
        
        ions, wls, intens = _map_peaks(all_wl[mask], smooth[mask], nist_df, peak_height, peak_distance=5)
        
        sorted_items = sorted(zip(ions, wls, intens), key=lambda x: x[2], reverse=True)
        ions, wls, intens = zip(*sorted_items) if sorted_items else ([],[],[])
        
    return list(ions), list(wls), list(intens)