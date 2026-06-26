#!/usr/bin/env python3

"""
Fragment Ion Extractor — Enhanced Edition (FIIE V2)
------------------------------------------------------------
High-throughput Target m/z Extraction Workflow
• Supports mzML and MGF
• Includes MS1 / MS2 / Both filtering
• Metadata table + available columns viewer
• Single matrix download with specialized flags
"""

__author__ = ("Saurabh Baheti: baheti.saurabh@mayo.edu")

import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import re
import json
import logging
import io
from typing import List, Tuple
from collections import Counter

from pyteomics import mzml, mgf
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import plotly.express as px
from streamlit.errors import StreamlitAPIException

from faq.faq_FIIE import *
# -------------------------------------------------------------------------
# 1. GLOBAL FONT & MATPLOTLIB SETTINGS
# -------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Nimbus Sans", "DejaVu Sans"],
    "font.weight": "bold",
    "font.size": 12,          
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "svg.fonttype": "none",   
    "pdf.fonttype": 42,       
    "ps.fonttype": 42
})

# -----------------------------------------------------------------------------
# Streamlit page config & CSS
# -----------------------------------------------------------------------------
def try_set_page_config(*args, **kwargs):
    try:
        st.set_page_config(*args, **kwargs)
    except StreamlitAPIException:
        pass

try_set_page_config(
    page_title="Fragment Ion Extractor | Mayo Clinic",
    layout="wide",
)

logging.basicConfig(filename="FragmentIonExtractor.log", level=logging.INFO)

def inject_custom_css():
    st.markdown("""
        <style>
        /* Hide Streamlit Cruft */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {padding-top: 2rem !important; max-width: 1600px;}

        /* Material Design Card Styling */
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 12px !important;
            border: 1px solid #e0e6ed !important;
            background-color: #ffffff;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02), 0 4px 12px rgba(0, 0, 0, 0.04) !important;
            transition: all 0.25s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
            padding: 10px !important;
        }

        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05), 0 12px 24px rgba(0, 90, 156, 0.1) !important;
            border-color: #005a9c !important;
        }

        h1, h2, h3, h4 {
            color: #005a9c !important;
            font-weight: 600 !important;
            letter-spacing: -0.5px !important;
        }
        
        .subtitle-text {
            color: #5f6368 !important;
            font-size: 1.1em;
            font-weight: 400;
        }

        /* Modernized Buttons */
        .stButton>button {
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        .stButton>button:hover {
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            transform: translateY(-1px);
        }
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

def _reset_module_flags():
    """Ensure only one module is open at a time."""
    for key in ["show_overview", "show_spectra", "show_extraction"]:
        st.session_state[key] = False


# ------------------------------------------------------------------
# Helper Functions (V2 Analytics)
# ------------------------------------------------------------------
def arr_or_empty(x):
    return np.asarray(x) if x is not None else np.array([], dtype=float)

def get_rt_min(spectrum, ftype="mzml"):
    if ftype.lower() == "mzml":
        rt = None
        try:
            rt = spectrum.get('scanList', {}).get('scan', [{}])[0].get('scan start time')
        except Exception:
            pass
        if rt is None:
            rt = spectrum.get('scan start time')
        try:
            return float(rt) if rt is not None else None
        except Exception:
            return None
    elif ftype.lower() == "mgf":
        rtsec = spectrum.get('params', {}).get('rtinseconds')
        try:
            return (float(rtsec) / 60.0) if rtsec is not None else None
        except Exception:
            return None
    return None

def parse_scan_number(scan_id: str):
    """Parses scan numbers, reliably handling sample IDs with underscores."""
    if not isinstance(scan_id, str):
        return None
    m = re.search(r"(?:scan|index)[=\s_]+(\d+)", scan_id, re.IGNORECASE)
    if m:
        try: return int(m.group(1))
        except Exception: return None
    return None

def get_bpi(spec: dict, intens: np.ndarray) -> float:
    bpi = spec.get('base peak intensity')
    if bpi is not None:
        try: return float(bpi)
        except Exception: pass
    return float(np.max(intens)) if intens.size else 0.0

def get_tic(spec: dict, intens: np.ndarray) -> float:
    tic = spec.get('total ion current')
    if tic is not None:
        try: return float(tic)
        except Exception: pass
    return float(np.sum(intens)) if intens.size else 0.0

def ppm_window(mz: float, ppm: float) -> Tuple[float, float]:
    d = mz * ppm * 1e-6
    return (mz - d, mz + d)

def nearest_or_max_peak(mz_arr: np.ndarray, int_arr: np.ndarray, target: float, tol_ppm: float, prefer_closest=True) -> float:
    if mz_arr.size == 0:
        return 0.0
    lo, hi = ppm_window(target, tol_ppm)
    lo_idx = np.searchsorted(mz_arr, lo, side='left')
    hi_idx = np.searchsorted(mz_arr, hi, side='right')
    if hi_idx <= lo_idx:
        return 0.0
    win_mz = mz_arr[lo_idx:hi_idx]
    win_int = int_arr[lo_idx:hi_idx]
    if win_mz.size == 0:
        return 0.0
    if prefer_closest:
        j = int(np.argmin(np.abs(win_mz - target)))
        return float(win_int[j])
    else:
        return float(win_int.max())

def extract_ions_from_arrays(mz_array, intensity_array, query_mzs, tol_ppm, prefer_closest, bpi, noise_pct):
    q_raw, q_nf = {}, {}
    thr = (noise_pct / 100.0) * bpi if bpi > 0 else float('inf')
    for target in query_mzs:
        lab = f"q_{target:g}"
        inten = nearest_or_max_peak(mz_array, intensity_array, target, tol_ppm, prefer_closest)
        q_raw[lab] = float(inten)
        q_nf[lab + "_nf"] = float(inten) if inten >= thr else np.nan
    return q_raw, q_nf

@st.cache_data(show_spinner=False)
def inspect_mzml(file_bytes, max_probe=1000):
    info = {
        "fileDescription": None, "softwareList": None, "instrumentConfigurationList": None, "dataProcessingList": None,
        "run_attrs": {}, "spectrum_count": 0, "chromatogram_count": 0,
        "ms_levels": [], "rt_min": None, "rt_max": None,
        "spectrum_keys_counter": Counter(), "scan_param_keys": Counter(),
        "precursor_param_keys": Counter(), "product_param_keys": Counter(),
        "first_spectrum": None,
    }
    def _first(reader, path):
        try: return next(reader.iterfind(path), None)
        except Exception: return None

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file_bytes); tmp.flush()
        path = tmp.name

    with mzml.MzML(path) as rdr:
        info["fileDescription"] = _first(rdr, "fileDescription")
        info["softwareList"] = _first(rdr, "softwareList")
        info["instrumentConfigurationList"] = _first(rdr, "instrumentConfigurationList")
        info["dataProcessingList"] = _first(rdr, "dataProcessingList")
        run = _first(rdr, "run") or {}
        if isinstance(run, dict):
            info["run_attrs"] = {k: run.get(k) for k in ("id", "defaultInstrumentConfigurationRef", "startTimeStamp")}

    with mzml.MzML(path) as rdr:
        info["spectrum_count"] = sum(1 for _ in rdr.iterfind("spectrumList/spectrum"))
    with mzml.MzML(path) as rdr:
        info["chromatogram_count"] = sum(1 for _ in rdr.iterfind("chromatogramList/chromatogram"))

    with mzml.read(path) as reader:
        for i, spec in enumerate(reader):
            if i == 0: info["first_spectrum"] = str(spec)
            mslev = spec.get("ms level")
            if mslev is not None: info["ms_levels"].append(mslev)
            rt = get_rt_min(spec, "mzml")
            if rt is not None:
                info["rt_min"] = rt if info["rt_min"] is None else min(info["rt_min"], rt)
                info["rt_max"] = rt if info["rt_max"] is None else max(info["rt_max"], rt)
            if i + 1 >= max_probe: break
    info["ms_levels"] = sorted(set(info["ms_levels"]))
    return info

def spectrum_row_basic(spec):
    row = {}
    sid = spec.get('id')
    row['id'] = sid
    row['ScanNumber'] = parse_scan_number(sid)
    row['ms level'] = spec.get('ms level')
    row['RT(min)'] = get_rt_min(spec, 'mzml')
    row['total ion current'] = spec.get('total ion current')
    row['base peak m/z'] = spec.get('base peak m/z')
    row['base peak intensity'] = spec.get('base peak intensity')
    mzs = arr_or_empty(spec.get('m/z array'))
    row['num points'] = int(mzs.size)
    try:
        prec = spec.get('precursorList', {}).get('precursor', [{}])[0]
        sel  = prec.get('selectedIonList', {}).get('selectedIon', [{}])[0]
        row['precursor selected ion m/z'] = sel.get('selected ion m/z')
        row['precursor charge state'] = sel.get('charge state')
        row['isolation target m/z'] = prec.get('isolationWindow', {}).get('isolation window target m/z')
    except Exception:
        pass
    return row

def flatten_dict_shallow(prefix, d, out):
    if not isinstance(d, dict): return
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, (np.ndarray, list)):
            if isinstance(v, np.ndarray):
                out[f"{key}.__array_len__"] = len(v)
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                for kk, vv in v[0].items():
                    kk2 = f"{key}[0].{kk}"
                    if isinstance(vv, (np.ndarray, list, dict)):
                        out[f"{kk2}.__type__"] = type(vv).__name__
                    else:
                        out[kk2] = vv
            else:
                out[f"{key}.__len__"] = len(v)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                kk2 = f"{key}.{kk}"
                if isinstance(vv, (np.ndarray, list, dict)):
                    if isinstance(vv, np.ndarray): out[f"{kk2}.__array_len__"] = len(vv)
                    elif isinstance(vv, list): out[f"{kk2}.__list_len__"] = len(vv)
                    else: out[f"{kk2}.__type__"] = type(vv).__name__
                else:
                    out[kk2] = vv
        else:
            out[key] = v

def spectrum_row_extended(spec):
    row = spectrum_row_basic(spec)
    flatten_dict_shallow("", spec, row)
    return row

@st.cache_data(show_spinner=False)
def build_scan_table(file_bytes, ftype='mzml', n_scans=50, ms_level=None, extended=False):
    rows = []
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file_bytes); tmp.flush()
        path = tmp.name
    if ftype.lower() == 'mzml':
        with mzml.read(path) as reader:
            for spec in reader:
                if ms_level and spec.get('ms level') != ms_level:
                    continue
                row = spectrum_row_extended(spec) if extended else spectrum_row_basic(spec)
                rows.append(row)
                if len(rows) >= n_scans: break
    elif ftype.lower() == 'mgf':
        with mgf.read(path) as reader:
            for spec in reader:
                p = spec.get('params', {})
                r = {
                    'title': p.get('title'),
                    'RT(min)': get_rt_min(spec, 'mgf'),
                    'charge': p.get('charge'),
                    'num points': int(arr_or_empty(spec.get('m/z array')).size)
                }
                rows.append(r)
                if len(rows) >= n_scans: break
    df = pd.DataFrame(rows)
    key_order = [c for c in ['id', 'ScanNumber', 'ms level', 'RT(min)', 'total ion current',
                             'base peak m/z', 'base peak intensity', 'num points',
                             'precursor selected ion m/z', 'precursor charge state', 'isolation target m/z'] if c in df.columns]
    other_cols = [c for c in df.columns if c not in key_order]
    return df[key_order + other_cols] if not df.empty else df

@st.cache_data(show_spinner=False)
def extract_long_table(file_bytes, ftype, ms_level, target_mzs, ppm_tolerance, prefer_closest, noise_pct):
    rows = []
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file_bytes); tmp.flush()
        path = tmp.name

    if ftype.lower() == 'mzml':
        with mzml.read(path) as reader:
            for spectrum in reader:
                if ms_level and spectrum.get('ms level') != ms_level:
                    continue
                mzs = arr_or_empty(spectrum.get("m/z array"))
                ints = arr_or_empty(spectrum.get("intensity array"))
                bpi = get_bpi(spectrum, ints)
                tic = get_tic(spectrum, ints)
                rt_min = get_rt_min(spectrum, 'mzml')
                scan_id = spectrum.get('id', None)
                scan_no = parse_scan_number(scan_id)
                precursor_mz = None; charge = None; iso_target = None
                try:
                    prec = spectrum.get('precursorList', {}).get('precursor', [{}])[0]
                    sel  = prec.get('selectedIonList', {}).get('selectedIon', [{}])[0]
                    precursor_mz = sel.get('selected ion m/z')
                    charge       = sel.get('charge state')
                    iso_target   = prec.get('isolationWindow', {}).get('isolation window target m/z')
                    if precursor_mz is not None: precursor_mz = float(precursor_mz)
                    if charge is not None:
                        try: charge = int(charge)
                        except Exception: charge = None
                    if iso_target is not None: iso_target = float(iso_target)
                except Exception:
                    pass
                q_raw, q_nf = extract_ions_from_arrays(mzs, ints, target_mzs, ppm_tolerance, prefer_closest, bpi, noise_pct)
                row = {
                    'Scan': scan_id, 'ScanNumber': scan_no, 'ms level': spectrum.get('ms level'),
                    'RT(min)': rt_min, 'BPI': bpi, 'TIC': tic,
                    'precursor m/z': precursor_mz, 'precursor charge': charge, 'isolation target m/z': iso_target
                }
                row.update(q_raw); row.update(q_nf)
                rows.append(row)
    else:
        with mgf.read(path) as reader:
            for spectrum in reader:
                mzs = arr_or_empty(spectrum.get("m/z array"))
                ints = arr_or_empty(spectrum.get("intensity array"))
                bpi = float(np.max(ints)) if ints.size else 0.0
                tic = float(np.sum(ints)) if ints.size else 0.0
                rt_min = get_rt_min(spectrum, 'mgf')
                scan_id = spectrum.get("params", {}).get('title', None)
                scan_no = parse_scan_number(scan_id)
                q_raw, q_nf = extract_ions_from_arrays(mzs, ints, target_mzs, ppm_tolerance, prefer_closest, bpi, noise_pct)
                row = {
                    'Scan': scan_id, 'ScanNumber': scan_no, 'ms level': 2,
                    'RT(min)': rt_min, 'BPI': bpi, 'TIC': tic,
                    'precursor m/z': None, 'precursor charge': None, 'isolation target m/z': None
                }
                row.update(q_raw); row.update(q_nf)
                rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        if 'ScanNumber' in df.columns:
            df['ScanNumber'] = pd.to_numeric(df['ScanNumber'], errors='coerce')
        if 'RT(min)' in df.columns:
            df['RT(min)'] = pd.to_numeric(df['RT(min)'], errors='coerce')
    return df

META_RENAME = {
    'ScanNumber': 'Scan number',
    'ms level': 'MS level',
    'RT(min)': 'Retention time (min)',
    'BPI': 'Base peak intensity',
    'precursor m/z': 'Precursor selected ion m/z',
    'precursor charge': 'Precursor charge state',
    'isolation target m/z': 'Isolation target',
}

def build_export_table(long_df: pd.DataFrame, include_nf: bool = True) -> pd.DataFrame:
    if long_df.empty: return long_df
    df = long_df.copy()
    df.rename(columns=META_RENAME, inplace=True)
    meta_keys = ['Scan number','MS level','Retention time (min)',
                 'Base peak intensity','Precursor selected ion m/z',
                 'Precursor charge state','Isolation target']
    meta_cols = [c for c in meta_keys if c in df.columns]
    q_cols = [c for c in df.columns if c.startswith('q_')] if include_nf else \
             [c for c in df.columns if c.startswith('q_') and not c.endswith('_nf')]
    front = (['Scan'] if 'Scan' in df.columns else []) + meta_cols + q_cols
    rest = [c for c in df.columns if c not in front]
    return df[front + rest]


def strip_nf_suffix(col: str) -> str:
    return col[:-3] if str(col).endswith("_nf") else str(col)


def parse_ion_annotation_table(uploaded_file):
    if uploaded_file is None:
        return None

    try:
        if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
            ann = pd.read_excel(uploaded_file)
        else:
            ann = pd.read_csv(uploaded_file)
    except Exception as e:
        st.warning(f"Could not read annotation table: {e}")
        return None

    required = {"Ion", "Annotation"}
    if not required.issubset(set(ann.columns)):
        st.warning("Annotation file must contain columns: Ion, Annotation")
        return None

    ann = ann[["Ion", "Annotation"]].copy()
    ann["Ion"] = ann["Ion"].astype(str).map(strip_nf_suffix)
    ann["Annotation"] = ann["Annotation"].astype(str)
    return ann.drop_duplicates()


def build_ion_summary(long_df: pd.DataFrame, selected_cols: list, raw_cols: list | None = None, nf_cols: list | None = None) -> pd.DataFrame:
    rows = []
    n_scans = len(long_df)

    raw_cols = raw_cols or []
    nf_cols = nf_cols or []

    for col in selected_cols:
        if col not in long_df.columns:
            continue

        vals = pd.to_numeric(long_df[col], errors="coerce")
        detected_mask = vals.fillna(0) > 0
        detected_vals = vals[detected_mask]

        detect_pct = (detected_mask.sum() / n_scans * 100.0) if n_scans else 0.0
        mean_int = float(detected_vals.mean()) if not detected_vals.empty else 0.0
        median_int = float(detected_vals.median()) if not detected_vals.empty else 0.0
        max_int = float(detected_vals.max()) if not detected_vals.empty else 0.0
        cv = float(detected_vals.std() / detected_vals.mean()) if len(detected_vals) > 1 and detected_vals.mean() > 0 else np.nan

        base_col = strip_nf_suffix(col)
        noise_filtered_pct = np.nan
        if nf_cols and raw_cols and f"{base_col}_nf" in long_df.columns and base_col in long_df.columns:
            raw_detect = pd.to_numeric(long_df[base_col], errors="coerce").fillna(0) > 0
            nf_detect = pd.to_numeric(long_df[f"{base_col}_nf"], errors="coerce").fillna(0) > 0
            suppressed = raw_detect & (~nf_detect)
            noise_filtered_pct = (suppressed.sum() / n_scans * 100.0) if n_scans else 0.0

        confidence_score = detect_pct / 100.0
        if pd.notna(noise_filtered_pct):
            confidence_score *= (1 - noise_filtered_pct / 100.0)

        if confidence_score >= 0.80:
            confidence_label = "High"
        elif confidence_score >= 0.50:
            confidence_label = "Moderate"
        else:
            confidence_label = "Low"

        rows.append({
            "Ion": base_col,
            "Detection_%": round(detect_pct, 2),
            "Mean_Intensity": round(mean_int, 2),
            "Median_Intensity": round(median_int, 2),
            "Max_Intensity": round(max_int, 2),
            "CV": round(cv, 3) if pd.notna(cv) else np.nan,
            "Noise_Filtered_%": round(noise_filtered_pct, 2) if pd.notna(noise_filtered_pct) else np.nan,
            "Confidence_Score": round(confidence_score, 3),
            "Confidence_Label": confidence_label,
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        by=["Confidence_Score", "Detection_%", "Mean_Intensity"],
        ascending=[False, False, False]
    ).reset_index(drop=True)


def build_pi_scan_summary(out: pd.DataFrame, q_cols_exported: list) -> pd.DataFrame:
    if out.empty or not q_cols_exported:
        return pd.DataFrame()

    detect_count = (out[q_cols_exported].fillna(0) > 0).sum(axis=1)

    return pd.DataFrame({
        "Total_Scans": [len(out)],
        "Scans_with_Any_Target": [(detect_count > 0).sum()],
        "Scans_with_All_Targets": [int(out["flag_all_ions_present"].sum()) if "flag_all_ions_present" in out.columns else np.nan],
        "Mean_Targets_Detected_per_Scan": [round(float(detect_count.mean()), 2)],
        "Median_Targets_Detected_per_Scan": [round(float(detect_count.median()), 2)],
        "Noise_Filtered_Scans": [int(out["flag_noise_filtered"].sum()) if "flag_noise_filtered" in out.columns else np.nan],
    })


def plot_detection_frequency(ion_summary: pd.DataFrame):
    if ion_summary.empty:
        st.info("No ion summary available.")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(ion_summary["Ion"], ion_summary["Detection_%"])
    ax.set_title("Detection Frequency by Target Ion", fontweight="bold")
    ax.set_ylabel("Detection (%)", fontweight="bold")
    ax.set_xlabel("Target Ion", fontweight="bold")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_confidence_scores(ion_summary: pd.DataFrame):
    if ion_summary.empty:
        st.info("No ion summary available.")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(ion_summary["Ion"], ion_summary["Confidence_Score"])
    ax.set_title("Ion Confidence Scores", fontweight="bold")
    ax.set_ylabel("Confidence Score", fontweight="bold")
    ax.set_xlabel("Target Ion", fontweight="bold")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_codetection_heatmap(long_df: pd.DataFrame, selected_cols: list):
    if long_df.empty or len(selected_cols) < 2:
        st.info("Need at least two ions for co-detection heatmap.")
        return

    binary_df = pd.DataFrame({
        strip_nf_suffix(col): (pd.to_numeric(long_df[col], errors="coerce").fillna(0) > 0).astype(int)
        for col in selected_cols if col in long_df.columns
    })

    if binary_df.shape[1] < 2:
        st.info("Need at least two ions with data for co-detection heatmap.")
        return

    co_mat = binary_df.corr().fillna(0)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(co_mat.values, aspect="auto")
    ax.set_xticks(range(len(co_mat.columns)))
    ax.set_yticks(range(len(co_mat.index)))
    ax.set_xticklabels(co_mat.columns, rotation=45, ha="right")
    ax.set_yticklabels(co_mat.index)
    ax.set_title("Co-detection Correlation Heatmap", fontweight="bold")

    for i in range(co_mat.shape[0]):
        for j in range(co_mat.shape[1]):
            ax.text(j, i, f"{co_mat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_rt_profiles(long_df: pd.DataFrame, selected_cols: list, max_ions: int = 6):
    if long_df.empty or not selected_cols:
        st.info("No data available for RT profiling.")
        return

    ion_strength = {}
    for col in selected_cols:
        if col in long_df.columns:
            vals = pd.to_numeric(long_df[col], errors="coerce").fillna(0)
            ion_strength[col] = float(vals.sum())

    top_cols = sorted(selected_cols, key=lambda c: ion_strength.get(c, 0), reverse=True)[:max_ions]

    if not top_cols:
        st.info("No ion signals available for RT profiling.")
        return

    for col in top_cols:
        dfp = long_df[["RT(min)", col]].copy()
        dfp["RT(min)"] = pd.to_numeric(dfp["RT(min)"], errors="coerce")
        dfp[col] = pd.to_numeric(dfp[col], errors="coerce").fillna(0)
        dfp = dfp.dropna(subset=["RT(min)"]).sort_values("RT(min)")

        if dfp.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.plot(dfp["RT(min)"], dfp[col])
        ax.set_title(f"Retention Time Profile: {strip_nf_suffix(col)}", fontweight="bold")
        ax.set_xlabel("Retention Time (min)", fontweight="bold")
        ax.set_ylabel("Intensity", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)



def render_professional_fiie_start_panel(upload_disabled: bool = True):
    """
    Professional first-run landing panel for Fragment Ion Extractor.
    Mirrors the GlycoExplorer start experience without informal icons.
    """
    button_text = "Upload data to enable extraction" if upload_disabled else "Review File and Extract Ions"

    st.markdown(
        """
        <style>
        .fiie-start-wrap { margin-top: 1.4rem; margin-bottom: 1.5rem; }
        .fiie-start-title {
            font-size: 1.85rem; line-height: 1.15; font-weight: 700;
            color: #005a9c; margin-bottom: 0.35rem; letter-spacing: -0.01em;
        }
        .fiie-start-subtitle {
            font-size: 1.02rem; color: #2f3a45; margin-bottom: 1.6rem;
            max-width: 980px; line-height: 1.55;
        }
        .fiie-step-card {
            border: 1px solid #d9e2ef; border-radius: 10px; padding: 22px 24px;
            background: #f8fbff; min-height: 150px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }
        .fiie-step-number {
            width: 42px; height: 42px; border-radius: 50%; background: #005a9c;
            color: white; font-weight: 700; display: inline-flex; align-items: center;
            justify-content: center; margin-right: 14px; font-size: 1rem; vertical-align: middle;
        }
        .fiie-step-heading { font-size: 1.15rem; font-weight: 700; color: #005a9c; vertical-align: middle; }
        .fiie-step-body { color: #2f3a45; margin-left: 58px; margin-top: 8px; font-size: 0.96rem; line-height: 1.5; }
        .fiie-step-arrow { text-align: center; font-size: 2.2rem; color: #8a94a6; padding-top: 50px; font-weight: 300; }
        .fiie-info-box {
            background: #eef6ff; border-radius: 8px; padding: 16px 20px; margin-top: 24px;
            color: #004b8d; font-weight: 700; border: 1px solid #d8eaff;
        }
        .fiie-info-box-sub { font-weight: 400; margin-top: 6px; color: #263746; }
        .fiie-warning-box {
            background: #fff8e5; border-radius: 8px; padding: 16px 20px; margin-top: 14px;
            color: #8a5a00; font-weight: 700; border: 1px solid #f5e4b6;
        }
        .fiie-warning-box-sub { font-weight: 400; margin-top: 6px; color: #5f4300; }
        .fiie-muted-button {
            border: 1px solid #d9dfe8; border-radius: 7px; padding: 10px 18px;
            text-align: center; color: #8a94a6; font-weight: 600; margin-top: 28px; background: #ffffff;
        }
        .fiie-active-button-note {
            border: 1px solid #b8d6f2; border-radius: 7px; padding: 10px 18px;
            text-align: center; color: #005a9c; font-weight: 700; margin-top: 28px; background: #f8fbff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="fiie-start-wrap">
            <div class="fiie-start-title">Start Here</div>
            <div class="fiie-start-subtitle">
                Begin by uploading an mzML or MGF file from the left panel. Then review file integrity,
                inspect representative spectra, and extract target fragment ion intensities with reproducible settings.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, arrow_col, col2 = st.columns([0.46, 0.08, 0.46])

    with col1:
        st.markdown(
            """
            <div class="fiie-step-card">
                <div><span class="fiie-step-number">1</span><span class="fiie-step-heading">Upload Raw Data</span></div>
                <div class="fiie-step-body">Upload an mzML or MGF file using the left sidebar. Select the parser and confirm target m/z values.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with arrow_col:
        st.markdown("<div class='fiie-step-arrow'>›</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            """
            <div class="fiie-step-card">
                <div><span class="fiie-step-number">2</span><span class="fiie-step-heading">Extract and Review</span></div>
                <div class="fiie-step-body">Use the workflow sections to inspect metadata, profile spectra, extract target ions, and export summary tables.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="fiie-info-box">
            Upload data using the left sidebar
            <div class="fiie-info-box-sub">Required input: one mzML or MGF file. Optional input: ion annotation table with Ion and Annotation columns.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if upload_disabled:
        st.markdown(
            """
            <div class="fiie-warning-box">
                No raw data uploaded yet
                <div class="fiie-warning-box-sub">Please upload an mzML or MGF file to unlock file overview, spectral profiling, and targeted extraction.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    btn_class = "fiie-muted-button" if upload_disabled else "fiie-active-button-note"
    st.markdown(f"<div class='{btn_class}'>{button_text}</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# MAIN APPLICATION
# ------------------------------------------------------------------
def main():
    # UI State Initialization
    st.session_state.setdefault("show_overview", False)
    st.session_state.setdefault("show_spectra", False)
    st.session_state.setdefault("show_extraction", False)
    st.session_state.setdefault("selected_level", None)
    st.session_state.setdefault("faq_search_box", "")
    
    ss = st.session_state

    # ============================================================
    # Sidebar – Configuration & File Upload
    # ============================================================
    with st.sidebar:
        st.markdown("### FIIE Workflow (MS1/MS2)")
        st.caption("v2.0 | Bioinformatics Workflow")
        st.divider()

        st.markdown("**1. Raw Data Input**")
        file = st.file_uploader(
            "Upload mzML or MGF file", 
            type=["mzML", "mgf"],
            help="Upload raw or centroided mass spectrometry data."
        )
        ftype = st.radio("File Type Parser", ["mzml", "mgf"], horizontal=True, label_visibility="collapsed")
        
        st.divider()
        st.markdown("**2. Target Mass-to-Charge (m/z) Values**")
        default_mzs = [111.046, 154.089, 187.099, 196.100, 246.137, 260.153,
                       312.149, 344.176, 362.186, 376.202, 432.230, 464.256]
        default_mz_str = ", ".join(map(str, default_mzs))
        mz_input = st.text_area(
            "Target Ions", 
            value=default_mz_str, 
            height=90,
            help="Comma, space, or semicolon separated list of exact m/z targets to extract."
        )

        def parse_targets(s: str) -> List[float]:
            if not s: return []
            vals = []
            for chunk in s.replace(";", ",").split(","):
                for p in chunk.split():
                    try: vals.append(float(p.strip()))
                    except Exception: pass
            return vals

        targets = parse_targets(mz_input)

        st.divider()
        st.markdown("**3. Optional Ion Annotation Table**")
        annotation_file = st.file_uploader(
            "Upload ion annotation table (CSV/Excel)",
            type=["csv", "xlsx", "xls"],
            help="Optional file with columns: Ion, Annotation"
        )
        ion_annotation_df = parse_ion_annotation_table(annotation_file) if annotation_file else None

        st.divider()
        st.markdown("**4. Extraction Parameters**")
        ppm_tol = st.number_input(
            "Mass Tolerance (ppm)", 
            1, 100, 10, 1,
            help="Defines the ±ppm search window around each target m/z for extraction."
        )
        noise_pct = st.number_input(
            "Intensity Noise Threshold (% BPI)", 
            0.0, 100.0, 2.0, 0.5,
            help="Intensities below this percentage of the Base Peak Intensity (BPI) will be excluded."
        )
        prefer = st.checkbox(
            "Peak Selection: Prefer Closest m/z", 
            value=True,
            help="If checked, selects the nearest m/z. If unchecked, selects the maximum intensity peak within the tolerance window."
        )
        use_nf = st.checkbox(
            "Export Noise-Filtered Matrix", 
            value=True,
            help="Construct the final exported pivot matrix using the noise-filtered (_nf) intensity values."
        )

        if st.button("Reset Session Data", help="Clear uploaded files and cached extractions."):
            st.cache_data.clear()
            for key in ["show_overview", "show_spectra", "show_extraction", "selected_level", "faq_search_box"]:
                ss.pop(key, None)
            st.rerun()

    if not file:
        render_professional_fiie_start_panel(upload_disabled=True)
        st.markdown("<br>", unsafe_allow_html=True)
        faq_section()
        return

    file_bytes = file.read()

    # ============================================================
    # Modular Navigation
    # ============================================================
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    if c1.button(("▸ " if not ss["show_overview"] else "▾ ") + "Data Integrity & Overview", use_container_width=True):
        state = not ss["show_overview"]
        _reset_module_flags()
        ss["show_overview"] = state

    if c2.button(("▸ " if not ss["show_spectra"] else "▾ ") + "Raw Spectral Profiling", use_container_width=True):
        state = not ss["show_spectra"]
        _reset_module_flags()
        ss["show_spectra"] = state

    if c3.button(("▸ " if not ss["show_extraction"] else "▾ ") + "Targeted Extraction & Export", use_container_width=True):
        state = not ss["show_extraction"]
        _reset_module_flags()
        ss["show_extraction"] = state

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------------
    # Module 1: Data Integrity & Overview
    # ------------------------------------------------------------
    if ss["show_overview"]:
        with st.container(border=True):
            st.markdown("#### Global File Metrics")
            if ftype == "mzml":
                with st.spinner("Inspecting mzML file topology..."):
                    info = inspect_mzml(file_bytes)
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Spectra", f"{info['spectrum_count']:,}")
                m2.metric("Chromatograms", f"{info['chromatogram_count']:,}")
                m3.metric("Detected MS Levels", ", ".join(map(str, info["ms_levels"])) or "-")
                if info["rt_min"] and info["rt_max"]:
                    m4.metric("RT Range (min)", f"{info['rt_min']:.2f} – {info['rt_max']:.2f}")

                st.divider()
                st.markdown("**Execution Filter**")
                ms_opts = ["MS1", "MS2", "Both"]
                index_map = {1: 0, 2: 1, None: 2}
                sel = st.radio(
                    "Target MS Level for Extraction (mzML):", 
                    ms_opts,
                    index=index_map.get(ss.get("selected_level"), 2),
                    horizontal=True
                )
                ms_map = {"MS1": 1, "MS2": 2, "Both": None}
                ss["selected_level"] = ms_map[sel]
            else:
                ss["selected_level"] = None
                st.info("MGF files typically contain only MS2 spectra. Global metrics are limited.")

            st.divider()
            st.markdown("#### Spectrum Metadata Explorer")
            c_meta1, c_meta2 = st.columns([0.7, 0.3])
            n_meta_rows = c_meta1.slider("Rows to parse", 5, 500, 50, 5)
            extended_cols = c_meta2.checkbox("Flatten Nested Fields", value=True)

            meta_df = build_scan_table(file_bytes, ftype, n_scans=n_meta_rows, ms_level=ss.get("selected_level"), extended=extended_cols)
            if meta_df.empty:
                st.info("No spectra found for the current filter.")
            else:
                st.dataframe(meta_df, use_container_width=True, height=420)
                
                with st.expander("Available Metadata Columns"):
                    col_df = pd.DataFrame({"Column": list(meta_df.columns)})
                    st.dataframe(col_df, use_container_width=True)
                    st.download_button("Download column list (CSV)", col_df.to_csv(index=False), "metadata_columns.csv", mime="text/csv")


    # ------------------------------------------------------------
    # Module 2: Raw Spectral Profiling
    # ------------------------------------------------------------
    if ss["show_spectra"]:
        with st.container(border=True):
            st.markdown("#### Interactive Spectra Preview")
            
            p_c1, p_c2 = st.columns(2)
            n_scans_preview = p_c1.slider("Example spectra to plot", 1, 10, 3, 1)
            n_peaks_preview = p_c2.slider("Top-N peaks per spectrum (table)", 5, 50, 20, 5)
            show_raw = st.checkbox("Generate Plots", value=False)
            
            if show_raw:
                with st.spinner("Parsing spectra..."):
                    with tempfile.NamedTemporaryFile(delete=False) as tmp:
                        tmp.write(file_bytes)
                        tmp.flush()
                        path = tmp.name
                    
                    reader = mzml.read(path) if ftype == "mzml" else mgf.read(path)
                    plotted = 0
                    
                    for i, spectrum in enumerate(reader):
                        if plotted >= n_scans_preview:
                            break
                        if ss.get("selected_level") and spectrum.get("ms level") != ss.get("selected_level"):
                            continue
                        
                        mzs = arr_or_empty(spectrum.get("m/z array"))
                        ints = arr_or_empty(spectrum.get("intensity array"))
                        if not mzs.size:
                            continue
                        
                        rt = get_rt_min(spectrum, ftype)
                        mslev_str = f"MS{int(spectrum.get('ms level', 0))}, " if ftype == "mzml" else ""
                        
                        if ftype == "mzml":
                            scan_id = spectrum.get("id", f"scan{i}")
                        else:
                            scan_id = spectrum.get("params", {}).get("title", f"scan{i}")

                        title_str = f"Spectrum {i + 1} ({scan_id}, {mslev_str}RT={rt:.2f} min)" if rt else f"Spectrum {i + 1} ({scan_id})"
                        
                        st.markdown(f"**{title_str}**")
                        
                        # Plotly Chart
                        fig = px.line(x=mzs, y=ints, labels={"x": "m/z", "y": "Intensity"})
                        fig.update_layout(font=dict(family="Arial", size=12), margin=dict(t=10, b=20, l=20, r=20), height=350)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Top-N Table
                        df_spec = pd.DataFrame({"m/z": mzs, "Intensity": ints})
                        st.dataframe(df_spec.sort_values("Intensity", ascending=False).head(n_peaks_preview), use_container_width=True)
                        st.markdown("---")
                        plotted += 1

    # ------------------------------------------------------------
    # Module 3: Targeted Extraction & Export
    # ------------------------------------------------------------
    if ss["show_extraction"]:
        if not targets:
            st.warning("Enter at least one target m/z value in the sidebar.")
        else:
            with st.spinner("Extracting targeted ion intensities across all spectra..."):
                long_df = extract_long_table(
                    file_bytes, ftype, ss.get("selected_level"),
                    targets, ppm_tol, prefer, noise_pct
                )

            if long_df.empty:
                st.error("No data extracted. Verify your m/z targets and MS level filters.")
            else:
                with st.container(border=True):
                    st.markdown("#### Extracted Per-Spectrum Data (Metadata + Query Ions)")
                    export_df = build_export_table(long_df, include_nf=True)
                    st.dataframe(export_df.head(200), use_container_width=True)

                st.markdown("<br>", unsafe_allow_html=True)
                
                # Pivot Matrix & V2 Flags
                value_cols_raw = [c for c in long_df.columns if c.startswith("q_") and not c.endswith("_nf")]
                value_cols_nf  = [c for c in long_df.columns if c.startswith("q_") and c.endswith("_nf")]

                selected_cols = value_cols_nf if (use_nf and value_cols_nf) else value_cols_raw
                
                if not selected_cols:
                    st.warning("No query-ion columns found to build matrix.")
                else:
                    rt_round = long_df['RT(min)'].round(4).rename("RT(min)_round")
                    narrow_sel = pd.concat([rt_round, long_df['ScanNumber'], long_df[selected_cols]], axis=1)
                    mat_sel = narrow_sel.pivot_table(index=['RT(min)_round','ScanNumber'], values=selected_cols, aggfunc='max').sort_index()

                    if value_cols_raw:
                        narrow_raw = pd.concat([rt_round, long_df['ScanNumber'], long_df[value_cols_raw]], axis=1)
                        mat_raw = narrow_raw.pivot_table(index=['RT(min)_round','ScanNumber'], values=value_cols_raw, aggfunc='max').reindex(mat_sel.index)
                    else:
                        mat_raw = pd.DataFrame(index=mat_sel.index)

                    if value_cols_nf:
                        narrow_nf = pd.concat([rt_round, long_df['ScanNumber'], long_df[value_cols_nf]], axis=1)
                        mat_nf = narrow_nf.pivot_table(index=['RT(min)_round','ScanNumber'], values=value_cols_nf, aggfunc='max').reindex(mat_sel.index)
                    else:
                        mat_nf = pd.DataFrame(index=mat_sel.index)

                    raw_as_nf = pd.DataFrame(index=mat_sel.index)
                    if value_cols_raw and value_cols_nf:
                        col_map = {raw: f"{raw}_nf" for raw in value_cols_raw if f"{raw}_nf" in value_cols_nf}
                        if col_map:
                            raw_as_nf = mat_raw.rename(columns=col_map)[list(col_map.values())]

                    meta_index = (
                        long_df.assign(**{"RT(min)_round": long_df['RT(min)'].round(4)})
                               [['RT(min)_round','ScanNumber','ms level','RT(min)','BPI',
                                 'precursor m/z','precursor charge','isolation target m/z']]
                               .drop_duplicates(subset=['RT(min)_round','ScanNumber'])
                               .set_index(['RT(min)_round','ScanNumber'])
                               .rename(columns={
                                   'ms level':'MS level', 'RT(min)':'Retention time (min)',
                                   'BPI':'Base peak intensity', 'precursor m/z':'Precursor selected ion m/z',
                                   'precursor charge':'Precursor charge state', 'isolation target m/z':'Isolation target',
                               })
                    )
                    out = meta_index.join(mat_sel, how='right').reset_index()

                    # Resolve Flags
                    q_cols_exported = [c for c in out.columns if c.startswith('q_') and c in selected_cols]
                    out['flag_all_ions_present'] = (out[q_cols_exported].fillna(0) > 0).all(axis=1)

                    out['flag_noise_filtered'] = False
                    if use_nf and not mat_nf.empty:
                        if not raw_as_nf.empty:
                            common_nf = [c for c in raw_as_nf.columns if c in mat_nf.columns]
                            if common_nf:
                                suppressed_any = ((raw_as_nf[common_nf].fillna(0) > 0) & (mat_nf[common_nf].fillna(0) == 0)).any(axis=1)
                                sup_any_aligned = suppressed_any.reindex(mat_sel.index, fill_value=False)
                                out['flag_noise_filtered'] = [
                                    bool(sup_any_aligned.get((row['RT(min)_round'], row['ScanNumber']), False))
                                    for _, row in out[['RT(min)_round','ScanNumber']].iterrows()
                                ]
                        else:
                            out['flag_noise_filtered'] = (out[q_cols_exported].fillna(0) == 0).any(axis=1)

                    raw_all_but_nf_filters_some = pd.Series(False, index=mat_sel.index)
                    if not raw_as_nf.empty and not mat_nf.empty:
                        common_nf = [c for c in raw_as_nf.columns if c in mat_nf.columns]
                        all_raw_present_series = (raw_as_nf[common_nf].fillna(0) > 0).all(axis=1)
                        suppressed_any_series = ((raw_as_nf[common_nf].fillna(0) > 0) & (mat_nf[common_nf].fillna(0) == 0)).any(axis=1)
                        raw_all_but_nf_filters_some = all_raw_present_series & suppressed_any_series

                    out['raw_all_but_nf_filters_some'] = [
                        bool(raw_all_but_nf_filters_some.get((row['RT(min)_round'], row['ScanNumber']), False))
                        for _, row in out[['RT(min)_round','ScanNumber']].iterrows()
                    ]

                    # Column Ordering
                    front = [
                        'flag_noise_filtered', 'flag_all_ions_present', 'raw_all_but_nf_filters_some',
                        'Retention time (min)', 'RT(min)_round', 'ScanNumber', 'MS level',
                        'Base peak intensity', 'Precursor selected ion m/z',
                        'Precursor charge state', 'Isolation target'
                    ]
                    ordered = [c for c in front if c in out.columns] + q_cols_exported + \
                              [c for c in out.columns if c.startswith('q_') and c not in q_cols_exported]
                    out = out[ordered]

                    col_mat, col_plot = st.columns([0.55, 0.45], gap="large")
                    
                    with col_mat:
                        with st.container(border=True):
                            st.markdown(f"#### Quantitative Pivot Matrix ({'Filtered' if use_nf else 'Raw'})")
                            st.dataframe(out.head(150), use_container_width=True)
                    
                    with col_plot:
                        with st.container(border=True):
                            st.markdown("#### Detection Distribution")
                            q_cols_active = [c for c in out.columns if c.startswith('q_') and (c in selected_cols)]
                            if q_cols_active:
                                nz = (out[q_cols_active].fillna(0) > 0).sum(axis=1)
                                fig3, ax3 = plt.subplots(figsize=(5, 4))
                                bins = np.arange(-0.5, len(q_cols_active) + 1.5, 1)
                                ax3.hist(nz, bins=bins, edgecolor="black", color="#377eb8", alpha=0.8)
                                ax3.set_xlim(-0.5, len(q_cols_active) + 0.5)
                                ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
                                ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
                                ax3.set_title("Detected Targets per (RT, Scan)", fontweight="bold", pad=15)
                                ax3.set_xlabel(f"# targets detected ({'NF' if use_nf else 'RAW'})", fontweight="bold")
                                ax3.set_ylabel("Count", fontweight="bold")
                                plt.tight_layout()
                                st.pyplot(fig3)
                                plt.close(fig3)
                                st.caption(f"**Total Scans:** {len(nz):,} | **Min:** {int(nz.min())} | **Max:** {int(nz.max())} | **Mean:** {nz.mean():.2f}")

                    st.markdown("<br>", unsafe_allow_html=True)
                    with st.container(border=True):
                        st.markdown("#### PI Summary & Biological Interpretation")

                        ion_summary = build_ion_summary(
                            long_df=long_df,
                            selected_cols=selected_cols,
                            raw_cols=value_cols_raw,
                            nf_cols=value_cols_nf,
                        )

                        if ion_annotation_df is not None and not ion_summary.empty:
                            ion_summary = ion_summary.merge(ion_annotation_df, on="Ion", how="left")

                        scan_summary = build_pi_scan_summary(out, q_cols_exported)

                        s1, s2 = st.columns([0.35, 0.65], gap="large")
                        with s1:
                            st.markdown("**Run-level Summary**")
                            if scan_summary.empty:
                                st.info("No run-level summary available.")
                            else:
                                st.dataframe(scan_summary, use_container_width=True, hide_index=True)

                        with s2:
                            st.markdown("**Ion Summary Table**")
                            if ion_summary.empty:
                                st.info("No ion summary available.")
                            else:
                                st.dataframe(ion_summary, use_container_width=True, height=320, hide_index=True)

                        st.markdown("<br>", unsafe_allow_html=True)

                        p1, p2 = st.columns(2, gap="large")
                        with p1:
                            plot_detection_frequency(ion_summary)
                        with p2:
                            plot_confidence_scores(ion_summary)

                        st.markdown("<br>", unsafe_allow_html=True)

                        p3, p4 = st.columns(2, gap="large")
                        with p3:
                            plot_codetection_heatmap(long_df, selected_cols)
                        with p4:
                            st.markdown("**Interpretation notes**")
                            if ion_summary.empty:
                                st.info("No interpretation summary available.")
                            else:
                                high_conf = int((ion_summary["Confidence_Label"] == "High").sum())
                                moderate_conf = int((ion_summary["Confidence_Label"] == "Moderate").sum())
                                low_conf = int((ion_summary["Confidence_Label"] == "Low").sum())

                                st.write(f"- High-confidence ions: {high_conf}")
                                st.write(f"- Moderate-confidence ions: {moderate_conf}")
                                st.write(f"- Low-confidence ions: {low_conf}")

                                if "Annotation" in ion_summary.columns:
                                    annotated = int(ion_summary["Annotation"].notna().sum())
                                    st.write(f"- Annotated ions: {annotated}")

                                st.markdown("**Top ions**")
                                st.dataframe(
                                    ion_summary.head(5)[[c for c in ["Ion", "Detection_%", "Confidence_Label", "Annotation"] if c in ion_summary.columns]],
                                    use_container_width=True,
                                    hide_index=True,
                                )

                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("**Retention Time Profiles of Top Ions**")
                        plot_rt_profiles(long_df, selected_cols, max_ions=6)

                    # Export Artifacts
                    st.markdown("<br>", unsafe_allow_html=True)
                    with st.container(border=True):
                        st.markdown("#### Export Analysis Artifacts")
                        dl_c1, dl_c2, dl_c3 = st.columns(3)

                        dl_c1.download_button(
                            "Download Matrix with Flags (CSV)", 
                            out.to_csv(index=False).encode('utf-8'), 
                            "fiie_matrix_with_flags.csv",
                            use_container_width=True,
                            help="Download the consolidated pivot table of all scans and detection flags."
                        )

                        dl_c2.download_button(
                            "Download Per-Spectrum (CSV)", 
                            export_df.to_csv(index=False).encode('utf-8'), 
                            "fiie_per_spectrum_results.csv",
                            use_container_width=True,
                            help="Download the comprehensive list of extracted values and metadata before pivoting."
                        )

                        params = {
                            "ppm_tolerance": ppm_tol, 
                            "noise_filter": noise_pct, 
                            "prefer_closest": prefer,
                            "use_nf": use_nf, 
                            "targets": targets, 
                            "MS_level": ss.get("selected_level"),
                            "file": file.name
                        }
                        dl_c3.download_button(
                            "Download Run Parameters (JSON)",
                            json.dumps(params, indent=2), 
                            "fiie_run_parameters.json", 
                            "application/json",
                            use_container_width=True,
                            help="Download the exact parameters used for this extraction to maintain reproducibility."
                        )

                        dl_c4, dl_c5 = st.columns(2)
                        dl_c4.download_button(
                            "Download Ion Summary (CSV)",
                            ion_summary.to_csv(index=False).encode('utf-8') if not ion_summary.empty else b"",
                            "fiie_ion_summary.csv",
                            "text/csv",
                            use_container_width=True,
                            help="Download the PI-facing ion-level summary table."
                        )
                        dl_c5.download_button(
                            "Download Run Summary (CSV)",
                            scan_summary.to_csv(index=False).encode('utf-8') if not scan_summary.empty else b"",
                            "fiie_run_summary.csv",
                            "text/csv",
                            use_container_width=True,
                            help="Download the run-level summary table."
                        )

    # ------------------------------------------------------------
    # Default State & FAQ
    # ------------------------------------------------------------
    if not any([ss["show_overview"], ss["show_spectra"], ss["show_extraction"]]):
        faq_section()



# =============================================================================
# Centralized browser-safe downloads
# =============================================================================
# RConnect cannot save directly to a user's laptop. These overrides keep all
# downloads browser-based so the browser can ask where to save when configured.
try:
    from download_utility import (
        add_download_button,
        add_file_download_buttons,
        add_hosted_safe_file_download_button,
        add_bytes_download_button,
        add_bytes_download_button_tp,
        add_dataframe_download_buttons,
        add_matplotlib_download_buttons,
        add_plot_download_buttons,
        add_plotly_download_buttons,
        add_figure_download_buttons_tp,
        add_figure_download_buttons_tp_lf,
        add_plotly_html_download_button_tp,
        add_plotly_html_download_button_tp_lf,
        add_figure_download_buttons_common,
        add_plotly_html_download_button_common,
        save_profile_pdf_to_downloads_tp,
        save_profile_pdf_to_downloads_tp_lf,
        explain_download_location,
        _save_bytes_to_downloads,
        _save_file_to_downloads,
    )
except Exception:
    pass

if __name__ == "__main__":
    main()
