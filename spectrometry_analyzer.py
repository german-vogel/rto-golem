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
MAX_IONS_TO_PLOT        = 5

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
    y = np.zeros_like(x, dtype=float)
    for i in range(0, len(params), 4):
        amp = params[i]
        cen = params[i+1]
        sigma = params[i+2]
        gamma = params[i+3]
        y += amp * voigt_profile(x - cen, sigma, gamma)
    return y

def _map_peaks(raw_spectrum, clean_spectrum, wl_arr, nist_df, peak_height):
    from scipy.signal import find_peaks, savgol_filter
    from scipy.optimize import curve_fit
    import numpy as np

    KNOWN_PEAKS = [
        ("H I (Hδ)", 410.1734), ("H I (Hg)", 434.0472),
        ("H I (Hb)", 486.135), ("H I (Hα)", 656.279),
        ("He I", 388.8648), ("He I", 402.61914), ("He I", 447.14802),
        ("He I", 471.31457), ("He I", 492.19313), ("He I", 501.56783),
        ("He I", 504.7738), ("He I", 587.5621), ("He I", 667.8151),
        ("He I", 706.519), ("He I", 728.1349),
        ("He II", 468.58), ("He II", 656.02),
        ("C I", 505.214919), ("C I", 711.318),
        ("C II", 392.0693), ("C II", 426.7), ("C II", 513.3282),
        ("C II", 514.5168), ("C II", 657.80481), ("C II", 658.28764),
        ("C II", 711.9925), ("C II", 723.642),
        ("C III", 464.742), ("C III", 465.025), ("C III", 465.147),
        ("N I", 672.261), ("N I", 742.3641), ("N I", 744.2298), ("N I", 746.8312),
        ("N II", 399.5), ("N II", 460.1478), ("N II", 463.054),
        ("N II", 500.515), ("N II", 567.956), ("N II", 571.077),
        ("O I", 777.1944),
        ("O II", 397.3256), ("O II", 407.58617), ("O II", 431.963),
        ("O II", 434.9426), ("O II", 441.4899), ("O II", 459.0974),
        ("O II", 463.88558), ("O II", 464.18103), ("O II", 464.91347),
        ("O II", 465.08384), ("O II", 466.16324),
        ("Cl II", 479.4556), ("Cl II", 481.007), ("Cl II", 481.948),
        ("Cl II", 521.7945), ("Cl II", 539.2125), ("Cl II", 542.3257),
        ("Cl II", 544.3375), ("Cl II", 545.7037),
        ("Mo I", 379.8252), ("Mo I", 386.4104), ("Mo I", 390.2953),
        ("Mo I", 406.9882), ("Mo I", 418.8324), ("Mo I", 441.1695),
        ("Mo I", 550.6494), ("Mo I", 553.3031), ("Mo I", 557.0444)
    ]

    TOLERANCE = 0.7
    SATURATION_THRESH = 16382.0
    
    bg_guess = np.percentile(clean_spectrum, 10)
    y_detect = savgol_filter(clean_spectrum, window_length=3, polyorder=2)
    roi_centers, _ = find_peaks(y_detect, height=bg_guess + peak_height, distance=15)
    
    all_found_wls = []
    all_found_amps = []
    
    for idx in roi_centers:
        center_wl = wl_arr[idx]
        integration_width = 11.0
        
        mask = (wl_arr >= center_wl - integration_width / 2) & (wl_arr <= center_wl + integration_width / 2)
        x_roi = wl_arr[mask]
        y_roi = clean_spectrum[mask]
        raw_roi = raw_spectrum[mask]
        
        if len(x_roi) < 5 or np.max(y_roi) < 2.0:
            continue
            
        local_max = np.max(y_roi)
        
        y_detect_roi = savgol_filter(y_roi, window_length=3, polyorder=2)
        peaks_idx_0, _ = find_peaks(y_detect_roi, height=bg_guess + 15.0, prominence=0.5, distance=2)
        
        y_deriv2 = savgol_filter(y_roi, window_length=5, polyorder=3, deriv=2)
        y_deriv2[raw_roi > SATURATION_THRESH * 0.5] = 0.0        
        y_deriv2[y_roi > local_max * 0.75] = 0.0
            
        prom_2 = max(np.std(y_deriv2) * 0.05, 0.02)
        peaks_idx_2, _ = find_peaks(-y_deriv2, prominence=prom_2, distance=2)
        
        combined_peaks = sorted(list(set(peaks_idx_0).union(set(peaks_idx_2))))
        final_peaks = []
        for p in combined_peaks:
            if y_roi[p] <= bg_guess + 15.0:
                continue
            if not final_peaks:
                final_peaks.append(p)
            else:
                if p - final_peaks[-1] < 2:
                    if y_roi[p] > y_roi[final_peaks[-1]]:
                        final_peaks[-1] = p
                else:
                    final_peaks.append(p)
                    
        if not final_peaks:
            continue
            
        p0 = []
        bounds_lower = []
        bounds_upper = [] 
        max_posible_area = max(local_max * 15.0, 100.0)
        
        for p in final_peaks:
            mu = x_roi[p]
            height = max(y_roi[p], 0.1)
            area_guess_factor = 10.0 if raw_roi[p] >= SATURATION_THRESH * 0.90 else 2.0
            p0.extend([height * area_guess_factor, mu, 0.4, 0.4])
            bounds_lower.extend([0, mu - 0.8, 0.15, 0.15])
            bounds_upper.extend([max_posible_area, mu + 0.8, 3.0, 3.0])
            
        valid_mask = raw_roi < (SATURATION_THRESH * 0.90)
        x_fit = x_roi[valid_mask]
        y_fit = y_roi[valid_mask]

        if len(x_fit) < len(p0): 
            continue
            
        try:
            popt, _ = curve_fit(multi_voigt, x_fit, y_fit, p0=p0, bounds=(bounds_lower, bounds_upper), maxfev=10000)
            
            for i in range(0, len(popt), 4):
                A = popt[i]
                mu_fit = popt[i+1]
                
                if A > 0:
                    is_duplicate = False
                    for prev_wl in all_found_wls:
                        if abs(prev_wl - mu_fit) < 0.8:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        all_found_wls.append(mu_fit)
                        all_found_amps.append(A)
        except RuntimeError:
            pass

    ions, mapped_wls, intensities = [], [], []
    
    for wl_peak, intensity in zip(all_found_wls, all_found_amps):
        best_known_ion = None
        best_known_wl = None
        min_dist = TOLERANCE
        
        for ion_label, ion_wl in KNOWN_PEAKS:
            dist = abs(wl_peak - ion_wl)
            if dist <= min_dist:
                min_dist = dist
                best_known_ion = ion_label
                best_known_wl = ion_wl
                
        if best_known_ion is not None:
            ions.append(f"{best_known_ion} ({best_known_wl:.2f} nm)")
            mapped_wls.append(best_known_wl)
            intensities.append(intensity)
            continue
            
        if nist_df is not None and not nist_df.empty:
            sel = nist_df[(nist_df['Wavelength'] >= wl_peak - TOLERANCE) & (nist_df['Wavelength'] <= wl_peak + TOLERANCE)].copy()
            if not sel.empty:
                sel['delta'] = np.abs(sel['Wavelength'] - wl_peak)
                sel['delta_round'] = np.round(sel['delta'], 1)
                min_delta = sel['delta_round'].min()
                tied = sel[sel['delta_round'] == min_delta]
                unique_ions = tied['Ion'].dropna().unique()
                unique_ions = list(dict.fromkeys([str(ion).strip() for ion in unique_ions]))
                combined_label = "/".join(unique_ions)
                avg_wl = tied['Wavelength'].mean()
                ions.append(f"{combined_label} ({avg_wl:.2f} nm)")
                mapped_wls.append(avg_wl)
                intensities.append(intensity)
                continue
                
        ions.append("Unknown")
        mapped_wls.append(wl_peak)
        intensities.append(intensity)
        
    return ions, mapped_wls, intensities

def _integrate_peak_robust(raw_spectrum, clean_spectrum, wavelengths, center_wl, integration_width=11.0):
    SATURATION_THRESH = 16382.0
    
    mask = (wavelengths >= center_wl - integration_width / 2) & (wavelengths <= center_wl + integration_width / 2)
    x_roi = wavelengths[mask]
    y_roi = clean_spectrum[mask]
    raw_roi = raw_spectrum[mask]
    
    if len(x_roi) < 5 or np.max(y_roi) < 2.0:
        return 0.0
        
    bg_guess = np.percentile(y_roi, 10)
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
        y_clean = np.maximum(y_roi - bg_guess, 0)
        from scipy.integrate import simpson
        return max(simpson(y=y_clean, x=x_roi), 0.0)
        
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

    if len(x_fit) < len(p0): 
        y_clean = np.maximum(y_roi - bg_guess, 0)
        from scipy.integrate import simpson
        return max(simpson(y=y_clean, x=x_roi), 0.0)
        
    try:
        popt, _ = curve_fit(multi_voigt, x_fit, y_fit, p0=p0, bounds=(bounds_lower, bounds_upper), maxfev=10000)
    except RuntimeError:
        y_clean = np.maximum(y_roi - bg_guess, 0)
        from scipy.integrate import simpson
        return max(simpson(y=y_clean, x=x_roi), 0.0)

    best_area = 0.0
    min_dist = float('inf')

    for i in range(0, len(popt), 4):
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
        patron_avg = rf"Acquisition statistic for spectrometer[^\n]+\({alias}\)[\s\S]*?Average time:\s*([\d\.]+)\s*ms"
        match_avg = re.search(patron_avg, texto_log)
        if match_avg:
            return float(match_avg.group(1))

    for alias in aliases_prioritarios:
        patron_int = rf"Setting spectrometer[^\n]+\({alias}\)[\s\S]*?Integration time is (\d+) us"
        match_int = re.search(patron_int, texto_log)
        if match_int:
            tiempo_us = float(match_int.group(1))
            return tiempo_us / 1000.0


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
        ref_bg = medfilt(ref_spectrum_raw, kernel_size=51)
        clean_ref = np.maximum(ref_spectrum_raw - ref_bg, 0)
        mask_ref = (all_wl >= WL_MIN) & (all_wl <= WL_MAX)
        
        ions, wls, intensities_ref = _map_peaks(
            ref_spectrum_raw[mask_ref], 
            clean_ref[mask_ref], 
            all_wl[mask_ref], 
            nist_df, 
            peak_height
        )
        sorted_ions_data = sorted(zip(ions, wls, intensities_ref), key=lambda x: x[2], reverse=True)
        
        if ions_to_plot is not None and scaling_dict is not None:
            ions_to_plot_data = [item for item in sorted_ions_data if item[0] in ions_to_plot]
        else:
            ions_to_plot_data = sorted_ions_data[:MAX_IONS_TO_PLOT]
            
        if not ions_to_plot_data:
            return
            
        color_shades = [lighten_color(shot_color, amount=i * 0.2) for i in range(len(ions_to_plot_data))]
        
        for i, (ion_label, center_wl, _) in enumerate(ions_to_plot_data):
            if ions_to_plot is not None and scaling_dict is not None:
                if ion_label not in ions_to_plot: continue
                scale_factor = scaling_dict.get(ion_label, 1.0)
            else:
                scale_factor = 1.0
                
            raw_integrated_intensities = []
            for frame_idx in range(time_points):
                raw_spectrum = all_spectra[frame_idx]
                
                dynamic_bg = medfilt(raw_spectrum, kernel_size=51)
                clean_spectrum = np.maximum(raw_spectrum - dynamic_bg, 0)                
                integral = _integrate_peak_robust(raw_spectrum, clean_spectrum, all_wl, center_wl, integration_width=11.0)
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
    ions, wls, intens = [], [], []
    with h5py.File(h5_path, 'r') as f:
        all_wl = f['Wavelengths'][:]
        all_spectra = f['Spectra'][:].astype(float)
        
        spectrum_raw = np.max(all_spectra, axis=0)
        ref_bg = medfilt(spectrum_raw, kernel_size=51)
        clean_ref = np.maximum(spectrum_raw - ref_bg, 0)
        mask = (all_wl >= WL_MIN) & (all_wl <= WL_MAX)
        
        ions, wls, intens = _map_peaks(
            spectrum_raw[mask], 
            clean_ref[mask], 
            all_wl[mask], 
            nist_df, 
            peak_height
        )
        
        sorted_items = sorted(zip(ions, wls, intens), key=lambda x: x[2], reverse=True)
        ions, wls, intens = zip(*sorted_items) if sorted_items else ([],[],[])
        
    return list(ions), list(wls), list(intens)