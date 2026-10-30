#!/usr/bin/env python3
"""
################################################################################
GlycoExplorer — Label-Free Quantification (LFQ) Workflow 
################################################################################
"""
__author__ = ("Saurabh Baheti: baheti.saurabh@mayo.edu")

import os, re, math, time, warnings
import base64
import textwrap
import zipfile
import datetime
from io import BytesIO
from pathlib import Path

import streamlit as st
from download_utility import add_download_button, add_matplotlib_download_buttons, add_plotly_download_buttons
import pandas as pd
import numpy as np

from PIL import Image
from scipy.stats import ttest_ind, f_oneway, ttest_rel, mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import plotly.express as px

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from streamlit.errors import StreamlitAPIException

try:
    from st_aggrid import AgGrid, GridOptionsBuilder
except Exception:
    AgGrid = None
    GridOptionsBuilder = None

# --- Externalized Module Imports ---
from ge.ui_utility import *
from ge.data_prep_utility import *
from ge.diff_results_utility import *
from ge.qc_utils import *
from ge.misc_utility import *
from ge.volcanoplot import *
from ge.boxplot import *
from ge.heatmap import *
from ge.lollipop import *
from ge.report_utility import *
from ge.batch_correcttion import *
from faq.faq_GE_LF import *   # label-free FAQ (previously imported the TMT FAQ by mistake)
from ge.waterfall_lf import *

# Optional import for ComBat Batch Correction
try:
    from neuroCombat import neuroCombat
    HAS_COMBAT = True
except ImportError:
    HAS_COMBAT = False

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = 500_000_000  # safety

# Enforce no background grids globally
plt.rcParams['axes.grid'] = False
sns.set_style("ticks", {"axes.grid": False})

# =============================================================================
# DESKTOP-SAFE FILE SAVE HELPERS
# =============================================================================
def _get_downloads_dir():
    """
    Return a safe Downloads folder path for macOS/Windows/Linux.
    """
    downloads = Path.home() / "Downloads"
    if downloads.exists():
        return downloads
    return Path.home()


def _safe_download_filename(filename):
    """
    Clean filename and avoid characters that can cause issues on macOS/Windows/Linux.
    """
    filename = str(filename).strip()
    filename = re.sub(r"[^A-Za-z0-9._-]+", "_", filename)
    filename = filename.strip("_")
    return filename or "downloaded_file"


def _coerce_to_bytes(data):
    """
    Convert common report/export objects to bytes.
    Also handles the case where a report generator returns a filepath string.
    """
    if data is None:
        return None

    if isinstance(data, BytesIO):
        return data.getvalue()

    if isinstance(data, memoryview):
        return data.tobytes()

    if isinstance(data, bytearray):
        return bytes(data)

    if isinstance(data, bytes):
        return data

    if isinstance(data, Path):
        return data.read_bytes()

    if isinstance(data, str):
        possible_path = Path(data).expanduser()
        if possible_path.exists() and possible_path.is_file():
            return possible_path.read_bytes()
        return data.encode("utf-8")

    try:
        return bytes(data)
    except Exception as e:
        raise TypeError(f"Cannot convert object of type {type(data)} to bytes: {e}")


def _save_bytes_to_downloads(data, filename):
    """
    Save bytes directly to the user's Downloads folder.
    This is more reliable than browser download inside packaged desktop apps.
    Existing files are not overwritten; a numeric suffix is added.
    """
    data = _coerce_to_bytes(data)
    if data is None:
        raise ValueError("No file data was available to save.")

    outdir = _get_downloads_dir()
    filename = _safe_download_filename(filename)
    outpath = outdir / filename

    base_outpath = outpath
    counter = 1
    while outpath.exists():
        outpath = base_outpath.with_name(f"{base_outpath.stem}_{counter}{base_outpath.suffix}")
        counter += 1

    outpath.write_bytes(data)
    return outpath


def _save_file_to_downloads(source_path, filename=None):
    """
    Save an existing file to the user's Downloads folder.
    Used for bundled example datasets in the standalone desktop app.
    """
    source_path = Path(source_path)
    if filename is None:
        filename = source_path.name
    return _save_bytes_to_downloads(source_path.read_bytes(), filename)


def _save_dataframe_csv_to_downloads(df, filename):
    """Save a DataFrame as CSV to Downloads."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return _save_bytes_to_downloads(csv_bytes, filename)


def _save_dataframe_excel_to_downloads(df, filename, sheet_name="Sheet1"):
    """Save a DataFrame as Excel to Downloads."""
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return _save_bytes_to_downloads(buffer.getvalue(), filename)



def _save_matplotlib_figure_to_downloads(fig, filename, fmt=None):
    """
    Save a Matplotlib figure directly to Downloads.
    Works reliably in packaged desktop builds where browser downloads may fail.
    """
    if fig is None:
        raise ValueError("No Matplotlib figure was available to save.")
    buffer = BytesIO()
    save_kwargs = {"bbox_inches": "tight", "dpi": 300}
    if fmt:
        save_kwargs["format"] = fmt
    fig.savefig(buffer, **save_kwargs)
    return _save_bytes_to_downloads(buffer.getvalue(), filename)


def _save_plotly_figure_to_downloads(fig, filename, fmt="html"):
    """
    Save a Plotly figure directly to Downloads.
    HTML does not require extra dependencies. Static image export requires kaleido.
    """
    if fig is None:
        raise ValueError("No Plotly figure was available to save.")
    fmt = (fmt or "html").lower()
    if fmt == "html":
        return _save_bytes_to_downloads(fig.to_html(include_plotlyjs="cdn"), filename)
    img_bytes = fig.to_image(format=fmt, scale=2)
    return _save_bytes_to_downloads(img_bytes, filename)


def _add_desktop_plot_save_buttons(fig, base_filename, key_prefix, include_svg=True, include_pdf=True):
    """Backward-compatible wrapper now using browser/RConnect-safe downloads."""
    if fig is None:
        try:
            current_fig = plt.gcf()
            if current_fig is not None and getattr(current_fig, "axes", None):
                fig = current_fig
        except Exception:
            fig = None
    if fig is None:
        return
    try:
        is_plotly = hasattr(fig, "to_html") and hasattr(fig, "to_image")
    except Exception:
        is_plotly = False
    if is_plotly:
        add_plotly_download_buttons(fig, base_filename=base_filename, key_prefix=key_prefix, include_html=True, include_png=False, include_svg=False)
    else:
        add_matplotlib_download_buttons(fig, base_filename=base_filename, key_prefix=key_prefix, include_png=True, include_svg=include_svg, include_pdf=include_pdf)


# -----------------------------------------------------------------------------
# Streamlit page config
# -----------------------------------------------------------------------------
try_set_page_config(page_title="GlycoExplorer LFQ | Mayo Clinic", layout="wide")
inject_custom_css() 

ss = st.session_state

# ---------------- GLOBAL sample/group selection (shared across modules) ----------------
ss.setdefault("global_sample_cols", [])          
ss.setdefault("global_group_samples", [])        
ss.setdefault("global_group_labels", [])         
ss.setdefault("global_n_groups", 2)              
ss.setdefault("global_control_group_idx", 0)     
ss.setdefault("global_all_groups", [])           

ss.setdefault("batch_dict", {})
ss.setdefault("batch_corrected", False)

# Permanent batch configuration state. These are separate from Streamlit
# widget keys so the setup survives when this panel is hidden.
ss.setdefault("enable_batch_correction", False)
ss.setdefault("batch_method_selected", "ComBat (Empirical Bayes)")
ss.setdefault("batch_num_selected", 2)
ss.setdefault("batch_config", {})
ss.setdefault("batch_correction_run_signature", None)
ss.setdefault("batch_correction_applied_method", "None")
ss.setdefault("batch_combat_info", None)

# One-time session defaults
ss.setdefault("merged_ready", False)
ss.setdefault("raw_merged_df", None)
ss.setdefault("merged_df", None)
ss.setdefault("diffexp_df", None)
ss.setdefault("filtered_df", None)
ss.setdefault("diff_done", False)
ss.setdefault("show_overview", False)
ss.setdefault("show_suite", False)
ss.setdefault("show_global_select", False)
ss.setdefault("upload_sig", None)  
ss.setdefault("show_report", False)

# Explore tab flags
for k in [
    "run_pca", "run_corr", "run_overview_heatmap", 
    "run_peptide_freq", "run_abundance", "run_glycan_class"
]:
    ss.setdefault(k, False)

# Gene plot flags
ss.setdefault("box_gene_list", [])
ss.setdefault("box_gene_plot", False)
ss.setdefault("box_gene_cleared", False)
ss.setdefault("lollipop_gene_ids", "")
ss.setdefault("lollipop_gene_list", [])
ss.setdefault("lollipop_gene_plot", False)

# Differential flags
ss.setdefault("control_group_idx", 0)
ss.setdefault("compare_open", False)
ss.setdefault("filter_open", False)
ss.setdefault("filtered_ready", False)

# Misc tab
ss.setdefault("explore_mode_user", None)
ss.setdefault("misc_user_df", None)
ss.setdefault("misc_user_df_valid", False)

ss.setdefault("tool_usage", {})

# Standardized report keys shared with ge.report_utility
ss.setdefault("waterfall_sig_col", "q-value")
ss.setdefault("waterfall_alpha", 0.05)
ss.setdefault("waterfall_logfc", 1.0)
ss.setdefault("waterfall_topn", 5)
ss.setdefault("pca_hide_outliers", False)
ss.setdefault("pca_label_outliers", True)
ss.setdefault("pca_zoom_q", 1.0)
ss.setdefault("cluster_palette", {})

# =============================================================================
# BATCH CORRECTION LOGIC
# =============================================================================
def apply_batch_correction(
    df: pd.DataFrame,
    batch_list: list[list[str]],
    method: str,
    group_map: dict[str, str] | None = None,
    batch_names: list[str] | None = None,
    strict: bool = False,
) -> pd.DataFrame:
    out = df.copy()
    
    if method == "Median Centering":
        global_med = out.median(axis=1)
        for b_cols in batch_list:
            valid = [c for c in b_cols if c in out.columns]
            if valid:
                b_med = out[valid].median(axis=1)
                out[valid] = out[valid].sub(b_med, axis=0).add(global_med, axis=0)
                
    elif method == "Mean Centering":
        global_mean = out.mean(axis=1)
        for b_cols in batch_list:
            valid = [c for c in b_cols if c in out.columns]
            if valid:
                b_mean = out[valid].mean(axis=1)
                out[valid] = out[valid].sub(b_mean, axis=0).add(global_mean, axis=0)
                
    elif method == "Standardize (Mean + Variance)":
        global_mean = out.mean(axis=1)
        global_std = out.std(axis=1).replace(0, 1e-9) 
        
        for b_cols in batch_list:
            valid = [c for c in b_cols if c in out.columns]
            if len(valid) > 1:
                b_mean = out[valid].mean(axis=1)
                b_std = out[valid].std(axis=1).replace(0, 1e-9)
                out[valid] = out[valid].sub(b_mean, axis=0).div(b_std, axis=0).mul(global_std, axis=0).add(global_mean, axis=0)
            elif len(valid) == 1:
                b_mean = out[valid].mean(axis=1)
                out[valid] = out[valid].sub(b_mean, axis=0).add(global_mean, axis=0)
                
    elif method == "ComBat (Empirical Bayes)":
        if not HAS_COMBAT:
            raise ImportError("The 'neuroCombat' package is not installed. Please run `pip install neuroCombat` to use this feature.")
            
        # Rebuild named batches, keeping the caller's batch order.
        named_batches = {}
        for idx, b_cols in enumerate(batch_list):
            b_name = (
                batch_names[idx]
                if batch_names is not None and idx < len(batch_names)
                else f"Batch {idx + 1}"
            )
            valid = [c for c in b_cols if c in out.columns]
            if valid:
                named_batches[b_name] = valid

        valid_cols = [c for cols in named_batches.values() for c in cols]
        if not valid_cols:
            return out

        if len(named_batches) < 2:
            raise RuntimeError("ComBat requires at least 2 non-empty batches.")

        # The biological group must be modelled, otherwise ComBat removes the
        # real group differences along with the batch offsets.
        if not group_map:
            raise RuntimeError(
                "ComBat needs the biological group of every batched sample. "
                "Assign samples to Group 1/2/3 before running batch correction."
            )

        missing_group = [c for c in valid_cols if c not in group_map]
        if missing_group:
            raise RuntimeError(
                "These batched samples have no biological group assigned: "
                + ", ".join(missing_group)
            )

        # This helper works on log2 values, while the shared ComBat routine
        # takes linear abundances and logs them itself. Hand it linear data
        # and ask for the log2 result back, so the scale is unchanged here.
        lin = np.exp2(out[valid_cols])

        try:
            corrected_log, info = apply_combat_batch_correction(
                lin,
                named_batches,
                valid_cols,
                group_map={c: group_map[c] for c in valid_cols},
                return_linear=False,
                strict=strict,
            )
        except ImportError:
            raise
        except ValueError as v_err:
            # Confounded or unusable design; the message explains the layout.
            raise RuntimeError(str(v_err))
        except Exception as e:
            raise RuntimeError(
                f"ComBat correction failed: {e}. "
                "Ensure batches have enough samples and non-zero variances."
            )

        out[valid_cols] = corrected_log[valid_cols]
        out.attrs["combat_info"] = info

    return out


# =============================================================================
# CORE LFQ STATISTICAL & FILTERING LOGIC
# =============================================================================

def replace_zeros_rowwise_min_fraction(
    df: pd.DataFrame, 
    sample_cols: list[str], 
    frac: float = 0.2,          
    replace_nan: bool = True, 
    add_debug: bool = True
) -> pd.DataFrame:
    out = df.copy()
    cols = [c for c in sample_cols if c in out.columns]
    if not cols:
        return out

    X = out[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    pos_mask = np.isfinite(X) & (X > 0)
    has_pos = pos_mask.any(axis=1)

    pos_vals = np.where(pos_mask, X, np.nan)
    row_min_pos = np.nanmin(pos_vals, axis=1)   
    fill = row_min_pos * float(frac)

    if replace_nan:
        repl_mask = (X == 0) | (~np.isfinite(X))
    else:
        repl_mask = (X == 0)

    X_new = X.copy()
    good = has_pos & np.isfinite(fill)
    if np.any(good):
        X_new[good] = np.where(repl_mask[good], fill[good, None], X[good])
        out[cols] = X_new

    if add_debug:
        tmp = np.where(pos_mask, X, np.inf)
        min_idx = np.argmin(tmp, axis=1)
        min_col = np.array(cols, dtype=object)[min_idx]
        min_col = np.where(has_pos, min_col, None)
        
        out["__min_pos_col__"] = min_col
        out["__min_pos_val__"] = np.where(has_pos, row_min_pos, np.nan)
        out["__impute_fill__"] = np.where(has_pos, fill, np.nan)
        
    return out


def filter_by_presence_rule(
    df: pd.DataFrame, 
    all_samples: list[str], 
    group_samples: list[list[str]],
    min_frac: float = 0.40, 
    presence_rule: str = "or",   
) -> pd.DataFrame:
    out = df.copy()
    all_cols = [c for c in all_samples if c in out.columns]
    if not all_cols:
        return out

    X = out[all_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    pres_all = X.gt(0)
    frac_all = pres_all.mean(axis=1)
    keep_all = frac_all >= float(min_frac)

    keep_group = pd.Series(False, index=out.index)
    for g in group_samples or []:
        gcols = [c for c in (g or []) if c in out.columns]
        if not gcols:
            continue
        pres_g = out[gcols].apply(pd.to_numeric, errors="coerce").fillna(0).gt(0)
        keep_group = keep_group | (pres_g.mean(axis=1) >= float(min_frac))

    rule = (presence_rule or "or").strip().lower()
    if rule == "all_only":
        keep = keep_all
    elif rule == "group_only":
        keep = keep_group
    elif rule == "and":
        keep = keep_all & keep_group
    else:
        keep = keep_all | keep_group
    
    return out.loc[keep].copy()

# =============================================================================
# PGLYCO LABEL-FREE PARSING LOGIC
# =============================================================================

def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = pd.Index(df.columns).map(lambda x: str(x).replace("\xa0", " ").strip())
    return df

def _clean_colname(x) -> str:
    x = str(x).replace("\ufeff", "").replace("\xa0", " ").strip()
    return x

def normalize_pglyco_df(pglyco_df: pd.DataFrame) -> pd.DataFrame:
    df = pglyco_df.copy()
    df.columns = [_clean_colname(c) for c in df.columns]
    if df.columns[0] in {"0", "Unnamed: 0", "index", "Index"}:
        df = df.drop(columns=[df.columns[0]])
    return df

def read_pglyco_file(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(uploaded_file)
    elif suffix in [".txt", ".tsv"]:
        df = pd.read_csv(uploaded_file, sep="\t", engine="python")
    elif suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(uploaded_file, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file type: {suffix}.")
    
    df.columns = [str(c).strip() for c in df.columns]
    return df

def classify_glycan(glycan_str: str) -> str:
    if pd.isna(glycan_str):
        return "Unknown"
    
    match = re.match(r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", str(glycan_str))
    if not match:
        return "Unknown"
    
    H, N, A, F = map(int, match.groups())
    
    if 4 <= H <= 12 and N == 2 and A == 0 and F in [0, 1]:
        return "High-mannose"
    elif 0 <= H <= 3 and N in [1, 2] and A == 0 and F in [0, 1]:
        return "Paucimannose"
    else:
        return "Complex/Hybrid"

def build_label_free_matrix_from_pglyco(
    pglyco_df: pd.DataFrame, 
    species: str = "Human", 
    version_label: str = "v3"
) -> pd.DataFrame:
    df = pglyco_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    def _safe_str_series(s: pd.Series) -> pd.Series:
        return s.astype("string").fillna("").str.strip()

    if "IsotopeArea" not in df.columns or not df["IsotopeArea"].notna().any():
        raise ValueError("Missing RAW intensity column: IsotopeArea")

    iso = df["IsotopeArea"].astype("string").fillna("").str.replace(",", "", regex=False).str.strip()
    df["IsotopeArea"] = pd.to_numeric(iso, errors="coerce")

    if pd.notna(df["IsotopeArea"]).any() and df["IsotopeArea"].max(skipna=True) < 1_000:
        raise ValueError("IsotopeArea appears log-transformed (max < 1000). Export RAW intensities.")

    df["IsotopeArea"] = df["IsotopeArea"].fillna(0).round(0).astype("int64")

    required_cols = ["Peptide", "Genes", "Proteins", "ProSites", "RawName"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    for c in required_cols:
        df[c] = _safe_str_series(df[c])

    def _has_real(series: pd.Series) -> bool:
        if series is None:
            return False
        s = _safe_str_series(series).replace({
            "": pd.NA, 
            "nan": pd.NA, 
            "NaN": pd.NA, 
            "None": pd.NA, 
            "NA": pd.NA, 
            "N/A": pd.NA
        }).dropna()
        if s.empty:
            return False
        return s.astype("string").str.contains(r"\d", regex=True).any()

    glycan_src, glycan_style = None, None
    for col in df.columns:
        clean_col = re.sub(r'\s+', '', str(col)).lower()
        if clean_col in ["correctedglycan(h,n,a,g,f)", "glycan(h,n,a,g,f)"] and _has_real(df[col]):
            glycan_src, glycan_style = col, "mouse"
            break
        elif clean_col in ["correctedglycan(h,n,a,f)", "glycan(h,n,a,f)"] and _has_real(df[col]):
            glycan_src, glycan_style = col, "human"
            break

    if not glycan_src:
        raise ValueError("No usable glycan column found")
        
    df["Glycan"] = _safe_str_series(df[glycan_src])

    def canonical_glycan_key(x: str) -> str:
        s = re.sub(r"\s+", " ", str(x).strip())
        nums = list(map(int, re.findall(r"\d+", s)))
        if glycan_style == "mouse":
            nums = (nums + [0] * 5)[:5]
            return f"H{nums[0]}_N{nums[1]}_A{nums[2]}_G{nums[3]}_F{nums[4]}"
        else:
            nums = (nums + [0] * 4)[:4]
            return f"H{nums[0]}_N{nums[1]}_A{nums[2]}_F{nums[3]}"

    df["Glycan_key"] = df["Glycan"].apply(canonical_glycan_key)

    def canonical_prosites(ps: str):
        if ps is None:
            return ("", "__EMPTY__")
        s = str(ps).strip()
        if s == "" or s.lower() in {"nan", "none", "na", "n/a"}:
            return ("", "__EMPTY__")
        parts = [re.sub(r"[^\w\(\)\-]+$", "", p.strip()) for p in re.split(r"[;,|\s]+", s) if p.strip()]
        norm = ";".join(sorted(parts))
        return (norm, "__" + norm)

    pros_norm_key = df["ProSites"].apply(canonical_prosites)
    df["ProSites_norm"] = pros_norm_key.apply(lambda x: x[0]).astype("string").fillna("")
    df["ProSites_key"]  = pros_norm_key.apply(lambda x: x[1]).astype("string").fillna("__EMPTY__")

    df["GlycanClass"] = df["Glycan"].apply(lambda g: classify_glycan(str(g)))

    df["Glycopeptide"] = (
        df["Peptide"] + "||" + 
        df["Glycan"] + "||" + 
        df["Genes"] + "||" + 
        df["ProSites_norm"]
    )

    group_keys = ["RawName", "Peptide", "Glycan_key", "ProSites_key"]
    df = df.sort_values(group_keys, kind="mergesort")

    uniq_sum = df.groupby(group_keys)["IsotopeArea"].apply(
        lambda s: int(np.unique(s.astype("int64")).sum())
    ).reset_index(name="IsotopeArea")
    
    df_rep = df.drop_duplicates(subset=group_keys, keep="first").copy()
    df_rep = df_rep.drop(columns=["IsotopeArea"]).merge(uniq_sum, on=group_keys, how="left")

    wide = df_rep.groupby(["Glycopeptide", "RawName"])["IsotopeArea"].sum().unstack(fill_value=0)

    meta = df_rep.drop_duplicates("Glycopeptide", keep="first").set_index("Glycopeptide")[
        ["Peptide", "Glycan", "Genes", "Proteins", "ProSites_norm", "GlycanClass"]
    ].rename(columns={"ProSites_norm": "ProSites"})
    
    merged_df = meta.join(wide, how="left").reset_index().rename(columns={"index": "Glycopeptide"})

    return merged_df

# =============================================================================
# MSFRAGGER LABEL-FREE PARSING LOGIC
# =============================================================================

def _msfragger_to_pglyco(glycan_str):
    """Converts MSFragger string 'HexNAc(4)Hex(5)NeuAc(2)' into pGlyco '5 4 2 0' (H, N, A, F)"""
    if pd.isna(glycan_str):
        return None
    
    clean_str = str(glycan_str).split('%')[0].strip()
    h = re.search(r'Hex\((\d+)\)', clean_str)
    n = re.search(r'HexNAc\((\d+)\)', clean_str)
    a = re.search(r'NeuAc\((\d+)\)', clean_str)
    f = re.search(r'Fuc\((\d+)\)', clean_str)
    
    return f"{h.group(1) if h else '0'} {n.group(1) if n else '0'} {a.group(1) if a else '0'} {f.group(1) if f else '0'}"

def _calculate_prosite(row):
    """Calculates absolute protein position of modified Asparagine (N)."""
    start_col = 'Protein Start' if 'Protein Start' in row else 'Start'
    
    if pd.isna(row.get('Assigned Modifications')) or pd.isna(row.get(start_col)):
        return "Unknown"
    
    try:
        start_pos = int(row[start_col])
    except ValueError:
        return "Unknown"
        
    matches = re.findall(r'(\d+)N\(([\d\.]+)\)', str(row['Assigned Modifications']))
    valid_sites = [start_pos + int(pos) - 1 for pos, mass in matches if float(mass) > 300]
    
    if valid_sites:
        return "; ".join(map(str, sorted(list(set(valid_sites)))))
    else:
        return "Unknown"

def process_msfragger_lfq_files(uploaded_files):
    """Parses multiple MSFragger LFQ TSV files, standardizes, and outer joins them into a unified matrix."""
    all_data = []
    
    for file in uploaded_files:
        sample_name = Path(file.name).stem 
        df = pd.read_csv(file, sep='\t', low_memory=False)
        
        if 'Total Glycan Composition' not in df.columns:
            continue
            
        df = df[df['Total Glycan Composition'].notna()].copy()
        if df.empty:
            continue
            
        df['Glycan(H,N,A,F)'] = df['Total Glycan Composition'].apply(_msfragger_to_pglyco)
        df['Glycan'] = df['Glycan(H,N,A,F)']
        df['GlycanClass'] = df['Glycan(H,N,A,F)'].apply(classify_glycan)
        df['ProSites'] = df.apply(_calculate_prosite, axis=1)
        
        if 'Gene' in df.columns:
            df.rename(columns={'Gene': 'Genes'}, inplace=True)
        if 'Protein ID' in df.columns:
            df.rename(columns={'Protein ID': 'Proteins'}, inplace=True)
            
        intensity_col = 'Intensity'
        if intensity_col not in df.columns:
            # Skip the 'Intensity NA' column as requested by the user
            int_cols = [c for c in df.columns if 'intensity' in c.lower() and 'intensity na' not in c.lower()]
            if int_cols:
                intensity_col = int_cols[0]
            else:
                continue 
        
        df['PSM_Count'] = 1
        agg_dict = {
            'Proteins': 'first', 
            'Genes': 'first', 
            'ProSites': 'first', 
            'GlycanClass': 'first', 
            'PSM_Count': 'sum', 
            intensity_col: 'max'
        }
        
        if 'Protein Description' in df.columns:
            agg_dict['Protein Description'] = 'first'
            
        grouped = df.groupby(['Peptide', 'Glycan(H,N,A,F)', 'Glycan']).agg(agg_dict).reset_index()
        grouped.rename(columns={intensity_col: f"Abundance: {sample_name}"}, inplace=True)
        all_data.append(grouped)
        
    if not all_data:
        return pd.DataFrame()
        
    merge_keys = ['Peptide', 'Glycan(H,N,A,F)', 'Glycan', 'Genes', 'ProSites', 'GlycanClass', 'Proteins']
    if 'Protein Description' in all_data[0].columns:
        merge_keys.append('Protein Description')
        
    master_df = all_data[0]
    for i in range(1, len(all_data)):
        master_df = pd.merge(master_df, all_data[i].drop(columns=['PSM_Count']), on=merge_keys, how='outer')
        if 'PSM_Count' in all_data[i].columns:
            temp_counts = master_df.pop('PSM_Count') if 'PSM_Count' in master_df.columns else pd.Series(0, index=master_df.index)
            master_df['PSM_Count'] = temp_counts.add(all_data[i]['PSM_Count'], fill_value=0)

    abundance_cols = [c for c in master_df.columns if c.startswith('Abundance:')]
    master_df[abundance_cols] = master_df[abundance_cols].fillna(0)
    
    master_df["Glycopeptide"] = (
        master_df["Peptide"].astype(str) + "||" + 
        master_df["Glycan"].astype(str) + "||" + 
        master_df["Genes"].astype(str) + "||" + 
        master_df["ProSites"].astype(str)
    )
    return master_df

# =============================================================================
# DIFFERENTIAL TESTING 
# =============================================================================

def compute_diffexp(
    df: pd.DataFrame, 
    control_cols: list[str], 
    test_cols: list[str], 
    control_label: str = "Control", 
    test_label: str = "Test",
    test_type: str = "Student's t-test (Independent, Equal Variance)", 
    do_zero_impute: bool = True, 
    zero_frac: float = 0.2,              
    do_presence_filter: bool = True, 
    presence_min_frac: float = 0.40, 
    presence_rule: str = "or", 
    group_samples_for_presence: list[list[str]] | None = None,
    min_n_per_group: int = 2, 
    add_debug_cols: bool = True
) -> pd.DataFrame:
    
    df = df.copy()
    ctrl = [c for c in control_cols if c in df.columns]
    test = [c for c in test_cols if c in df.columns]
    all_cols = list(dict.fromkeys(ctrl + test))
    
    if not ctrl or not test:
        raise ValueError("Need at least 1 valid sample column in both control and test groups.")
        
    for c in all_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if do_presence_filter:
        df = filter_by_presence_rule(
            df=df, 
            all_samples=all_cols, 
            group_samples=group_samples_for_presence or [ctrl, test], 
            min_frac=presence_min_frac, 
            presence_rule=presence_rule
        )

    if add_debug_cols:
        raw_vals = df[all_cols].to_numpy(dtype=float)
        raw_zero_count = np.sum((raw_vals == 0) | (~np.isfinite(raw_vals)), axis=1)
        pos_raw = np.where((raw_vals > 0) & np.isfinite(raw_vals), raw_vals, np.nan)
        true_row_min = np.nanmin(pos_raw, axis=1)

    if do_zero_impute:
        df = replace_zeros_rowwise_min_fraction(
            df=df, 
            sample_cols=all_cols, 
            frac=zero_frac, 
            replace_nan=True, 
            add_debug=False
        )

    for c in all_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if add_debug_cols:
        df["__row_min_pos__"] = true_row_min
        df["__impute_fill__"] = true_row_min * float(zero_frac)
        df["__zeros_or_na_before__"] = raw_zero_count
        post_vals = df[all_cols].to_numpy(dtype=float)
        df["__zeros_or_na_after__"] = np.sum((post_vals == 0) | (~np.isfinite(post_vals)), axis=1)

    df[control_label] = df[ctrl].mean(axis=1)
    df[test_label] = df[test].mean(axis=1)
    pseudo = 0.0
    df["log2FC"] = np.log2(df[test_label] + pseudo) - np.log2(df[control_label] + pseudo)

    pvals = []
    is_paired = "Paired" in test_type
    
    for i in range(len(df)):
        v1_raw = df.iloc[i][ctrl].astype(float).values
        v2_raw = df.iloc[i][test].astype(float).values

        if is_paired:
            if len(v1_raw) == len(v2_raw):
                mask = ~np.isnan(v1_raw) & ~np.isnan(v2_raw)
                v1, v2 = v1_raw[mask], v2_raw[mask]
            else:
                v1, v2 = [], [] 
        else:
            v1, v2 = v1_raw[~np.isnan(v1_raw)], v2_raw[~np.isnan(v2_raw)]

        if len(v1) >= min_n_per_group and len(v2) >= min_n_per_group:
            try:
                if test_type == "Student's t-test (Independent, Equal Variance)":
                    pval = float(ttest_ind(v1, v2, equal_var=True, nan_policy="omit")[1])
                elif test_type == "Welch's t-test (Independent, Unequal Variance)":
                    pval = float(ttest_ind(v1, v2, equal_var=False, nan_policy="omit")[1])
                elif test_type == "Paired t-test (Matched Samples)":
                    pval = float(ttest_rel(v1, v2, nan_policy="omit")[1])
                elif test_type == "Mann-Whitney U (Non-parametric Independent)":
                    pval = float(mannwhitneyu(v1, v2, nan_policy="omit")[1])
                elif test_type == "Wilcoxon Signed-Rank (Non-parametric Paired)":
                    pval = float(wilcoxon(v1, v2, nan_policy="omit")[1])
                else:
                    pval = np.nan
            except Exception:
                pval = np.nan
        else:
            pval = np.nan
            
        pvals.append(pval)

    df["p-value"] = pvals
    df["adj.p-value"] = multipletests(np.nan_to_num(df["p-value"], nan=1.0), method="fdr_bh")[1]
    df["q-value"] = df["adj.p-value"]

    df["Regulation"] = "Not Significant"
    df.loc[(df["q-value"] < 0.05) & (df["log2FC"] > 1), "Regulation"] = "Upregulated"
    df.loc[(df["q-value"] < 0.05) & (df["log2FC"] < -1), "Regulation"] = "Downregulated"
    df["__reg_sort__"] = df["Regulation"].map({
        "Upregulated": 0, 
        "Downregulated": 1, 
        "Not Significant": 2
    }).fillna(3)

    base_cols = ["Peptide", "Glycan", "Genes", "Proteins", "ProSites", "GlycanClass"]
    stats_cols = [control_label, test_label, "log2FC", "p-value", "q-value", "Regulation"]
    debug_cols = ["__row_min_pos__", "__impute_fill__", "__zeros_or_na_before__", "__zeros_or_na_after__"] if add_debug_cols else []
    
    keep_cols = (
        [c for c in base_cols if c in df.columns] + 
        stats_cols + 
        debug_cols + 
        all_cols
    )
    
    return df.sort_values(
        by=["__reg_sort__", "q-value"], 
        ascending=[True, True], 
        na_position="last"
    ).drop(columns=["__reg_sort__"], errors="ignore")[keep_cols]


def compute_diffexp_anova(
    df: pd.DataFrame, 
    group1_cols: list[str], 
    group2_cols: list[str], 
    group3_cols: list[str], 
    group1_label: str = "Group1", 
    group2_label: str = "Group2", 
    group3_label: str = "Group3",
    run_tukey: bool = False, 
    tukey_limit: int = 100, 
    do_zero_impute: bool = True, 
    zero_frac: float = 0.2, 
    do_presence_filter: bool = True, 
    presence_min_frac: float = 0.40, 
    presence_rule: str = "or"
) -> pd.DataFrame:
    
    df = df.copy()
    g1 = [c for c in group1_cols if c in df.columns]
    g2 = [c for c in group2_cols if c in df.columns]
    g3 = [c for c in group3_cols if c in df.columns]
    
    all_cols = list(dict.fromkeys(g1 + g2 + g3))

    if not (g1 and g2 and g3):
        raise ValueError("ANOVA requires 3 non-empty groups.")
        
    if do_presence_filter:
        df = filter_by_presence_rule(
            df=df, 
            all_samples=all_cols, 
            group_samples=[g1, g2, g3], 
            min_frac=presence_min_frac, 
            presence_rule=presence_rule
        )
        
    if do_zero_impute:
        df = replace_zeros_rowwise_min_fraction(
            df=df, 
            sample_cols=all_cols, 
            frac=zero_frac, 
            replace_nan=True
        )

    for c in all_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        
    df[group1_label] = df[g1].mean(axis=1)
    df[group2_label] = df[g2].mean(axis=1)
    df[group3_label] = df[g3].mean(axis=1)

    pvals = []
    for _, row in df.iterrows():
        v1 = row[g1].dropna().astype(float).values
        v2 = row[g2].dropna().astype(float).values
        v3 = row[g3].dropna().astype(float).values
        
        if len(v1) < 2 or len(v2) < 2 or len(v3) < 2:
            pvals.append(np.nan)
        else:
            try:
                pvals.append(float(f_oneway(v1, v2, v3).pvalue))
            except Exception:
                pvals.append(np.nan)

    df["p-value"] = pvals
    df["adj.p-value"] = multipletests(df["p-value"].fillna(1.0), method="fdr_bh")[1]
    df["q-value"] = df["adj.p-value"]
    df["Regulation"] = np.where(df["q-value"] < 0.05, "Significant", "Not Significant")
    
    base_cols = ["Peptide", "Glycan", "Genes", "Proteins", "ProSites", "GlycanClass"]
    keep = (
        [c for c in base_cols if c in df.columns] + 
        [group1_label, group2_label, group3_label, "p-value", "q-value", "Regulation"] + 
        all_cols
    )
    return df.sort_values("adj.p-value", na_position="last")[keep]


# -----------------------------------------------------------------------------
# limma for label-free data
#
# Same presence filter (and, if chosen, the same 0/NA replacement) as the
# classical tests, then the shared limma implementation in
# ge/diff_results_utility.py, which log2-transforms the raw intensities and
# treats remaining zeros/NAs as missing.
# -----------------------------------------------------------------------------
def _prepare_lf_matrix(df, model_cols, presence_cols, presence_groups,
                       do_zero_impute, zero_frac, do_presence_filter,
                       presence_min_frac, presence_rule):
    out = df.copy()
    for c in model_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if do_presence_filter:
        out = filter_by_presence_rule(
            df=out,
            all_samples=presence_cols,
            group_samples=presence_groups,
            min_frac=presence_min_frac,
            presence_rule=presence_rule,
        )
    if do_zero_impute:
        out = replace_zeros_rowwise_min_fraction(
            df=out, sample_cols=model_cols, frac=zero_frac,
            replace_nan=True, add_debug=False,
        )
    if out.empty:
        raise ValueError("No glycoforms are left after the presence filter. Lower the presence threshold.")
    return out


def compute_diffexp_limma_lf(df, group_cols, group_labels, control_idx, test_idx,
                             use_all_groups=True, do_zero_impute=False, zero_frac=0.2,
                             do_presence_filter=True, presence_min_frac=0.40,
                             presence_rule="or"):
    ctrl, test = list(group_cols[control_idx]), list(group_cols[test_idx])
    model_groups = [list(g) for g in group_cols if g] if use_all_groups else [ctrl, test]
    model_cols = list(dict.fromkeys(c for g in model_groups for c in g))
    prepared = _prepare_lf_matrix(
        df, model_cols, ctrl + test, [ctrl, test],
        do_zero_impute, zero_frac, do_presence_filter, presence_min_frac, presence_rule,
    )
    return compute_diffexp_limma(
        prepared, [list(g) for g in group_cols], list(group_labels),
        control_idx, test_idx, use_all_groups,
    )


def compute_diffexp_limma_f_lf(df, group_cols, group_labels, control_idx=0,
                               do_zero_impute=False, zero_frac=0.2,
                               do_presence_filter=True, presence_min_frac=0.40,
                               presence_rule="or"):
    groups = [list(g) for g in group_cols if g]
    all_cols = list(dict.fromkeys(c for g in groups for c in g))
    prepared = _prepare_lf_matrix(
        df, all_cols, all_cols, groups,
        do_zero_impute, zero_frac, do_presence_filter, presence_min_frac, presence_rule,
    )
    return compute_diffexp_limma_f(
        prepared, [list(g) for g in group_cols], list(group_labels), control_idx,
    )



# -----------------------------------------------------------------------------
# APP LOGIC
# -----------------------------------------------------------------------------

def _reset_all_data(clear_cache: bool = False):
    ss.pop("stat_method_label", None)
    ss["show_class_breakdown"] = False
    ss["global_sample_cols"] = []
    ss["global_group_samples"] = []
    ss["global_group_labels"] = []
    ss["global_n_groups"] = 2
    ss["global_control_group_idx"] = 0
    ss["global_all_groups"] = []
    
    ss["batch_dict"] = {}
    ss["batch_config"] = {}
    ss["batch_corrected"] = False
    ss["enable_batch_correction"] = False
    ss["batch_method_selected"] = "ComBat (Empirical Bayes)"
    ss["batch_num_selected"] = 2
    ss["batch_correction_run_signature"] = None
    ss["batch_correction_applied_method"] = "None"
    ss["batch_combat_info"] = None

    # Clear temporary Streamlit widget keys used by the batch panel.
    for _k in list(ss.keys()):
        if (
            str(_k).startswith("_ge_lfq_batch_")
            or str(_k).startswith("batch_sel_")
        ):
            ss.pop(_k, None)
    ss["show_report"] = False
    ss["merged_ready"] = False
    ss["raw_merged_df"] = None
    ss["merged_df"] = None
    ss["diffexp_df"] = None
    ss["filtered_df"] = None
    ss["diff_done"] = False
    ss["filtered_ready"] = False

    for k in [
        "run_pca", "run_corr", "run_overview_heatmap", 
        "run_peptide_freq", "run_abundance", "run_glycan_class"
    ]:
        ss[k] = False
        
    for key in [
        "show_volcano", "show_box", "show_heatmap", 
        "show_box_gene", "show_lollipop", "show_waterfall"
    ]:
        ss[key] = False

    ss["box_gene_list"] = []
    ss["box_gene_plot"] = False
    ss["box_gene_cleared"] = False
    ss["lollipop_gene_ids"] = ""
    ss["lollipop_gene_list"] = []
    ss["lollipop_gene_plot"] = False
    ss["compare_open"] = False
    ss["filter_open"] = False
    ss["misc_user_df"] = None
    ss["misc_user_df_valid"] = False

    ss.pop("final_pdf_bytes", None)
    ss.pop("final_report_zip_bytes", None)
    ss.pop("report_waterfall_fig", None)
    
    if clear_cache:
        try:
            st.cache_data.clear()
        except Exception:
            pass

def _make_upload_sig(*files):
    if any(f is None for f in files):
        return None
    return tuple((f.name, f.size) for f in files)

def init_session_state_defaults():
    defaults = {
        "merged_ready": False, 
        "diff_done": False,
        "raw_merged_df": None,
        "merged_df": None, 
        "diffexp_df": None, 
        "design_df": None, 
        "group_col": None, 
        "batch_col": None, 
        "tool_usage": {}
    }
    for k, v in defaults.items():
        ss.setdefault(k, v)

def _plot_genes_cb():
    ss["box_gene_plot"] = True

def _clear_genes_cb(): 
    ss["box_gene_list"] = []
    ss["box_gene_plot"] = False
    ss["box_gene_cleared"] = True

def _plot_lollipop_cb():
    ss["lollipop_gene_plot"] = True

def _clear_lollipop_cb():
    ss["lollipop_gene_ids"] = ""
    ss["lollipop_gene_list"] = []
    ss["lollipop_gene_plot"] = False


def render_professional_lfq_start_panel(pipeline_choice: str, merge_disabled: bool):
    """Professional first-run landing panel for the GlycoExplorer LFQ app."""
    if pipeline_choice == "pGlyco (LFQ)":
        upload_text = "Upload your pGlyco LFQ data matrix using the left sidebar."
        process_text = "Click Compile LFQ Data Matrix to create the glycopeptide abundance table."
    else:
        upload_text = "Upload one or more MSFragger LFQ sample-run files using the left sidebar."
        process_text = "Click Compile LFQ Data Matrix to join sample runs into one glycopeptide abundance table."

    button_text = "Upload data to enable processing" if merge_disabled else "Compile LFQ Data Matrix"

    st.markdown(
        """
        <style>
        .ge-start-title {
            font-size: 1.8rem;
            font-weight: 700;
            color: #005a9c;
            margin-bottom: 0.2rem;
        }
        .ge-start-subtitle {
            font-size: 1rem;
            color: #333333;
            margin-bottom: 1.5rem;
            line-height: 1.45;
        }
        .ge-step-card {
            border: 1px solid #d9e2ef;
            border-radius: 10px;
            padding: 22px 24px;
            background: #f8fbff;
            min-height: 135px;
        }
        .ge-step-number {
            width: 42px;
            height: 42px;
            border-radius: 50%;
            background: #005a9c;
            color: white;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin-right: 14px;
            font-size: 1rem;
        }
        .ge-step-heading {
            font-size: 1.15rem;
            font-weight: 700;
            color: #005a9c;
        }
        .ge-step-body {
            color: #333333;
            margin-left: 58px;
            margin-top: 8px;
            font-size: 0.95rem;
            line-height: 1.45;
        }
        .ge-info-box {
            background: #eef6ff;
            border-radius: 8px;
            padding: 16px 20px;
            margin-top: 24px;
            color: #004b8d;
            font-weight: 600;
        }
        .ge-warning-box {
            background: #fff8e5;
            border-radius: 8px;
            padding: 16px 20px;
            margin-top: 14px;
            color: #8a5a00;
            font-weight: 600;
        }
        .ge-muted-button {
            border: 1px solid #d9dfe8;
            border-radius: 7px;
            padding: 10px 18px;
            text-align: center;
            color: #8a94a6;
            font-weight: 600;
            margin-top: 28px;
            background: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="ge-start-title">Start Here</div>
        <div class="ge-start-subtitle">
            Begin your label-free glycopeptide analysis by uploading the required data from the left panel.
            Once uploaded, compile the LFQ matrix before moving to group setup, QC, statistics, and visualization.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, arrow_col, col2 = st.columns([0.46, 0.08, 0.46])

    with col1:
        st.markdown(
            f"""
            <div class="ge-step-card">
                <div>
                    <span class="ge-step-number">1</span>
                    <span class="ge-step-heading">Upload Data</span>
                </div>
                <div class="ge-step-body">{upload_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with arrow_col:
        st.markdown(
            "<div style='text-align:center; font-size:2rem; color:#8a94a6; padding-top:45px;'>›</div>",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="ge-step-card">
                <div>
                    <span class="ge-step-number">2</span>
                    <span class="ge-step-heading">Compile Matrix</span>
                </div>
                <div class="ge-step-body">{process_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="ge-info-box">
            Upload data using the left sidebar
            <div style="font-weight:400; margin-top:6px;">
                Use the file upload section in the left panel to add your LFQ input file(s).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if merge_disabled:
        st.markdown(
            """
            <div class="ge-warning-box">
                No data uploaded yet
                <div style="font-weight:400; margin-top:6px;">
                    Please upload the required file(s) to enable LFQ data processing.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(f"<div class='ge-muted-button'>{button_text}</div>", unsafe_allow_html=True)


def _create_lf_waterfall_report_fig(
    diff_df: pd.DataFrame,
    sig_col: str = "q-value",
    alpha_thr: float = 0.05,
    fc_thr: float = 1.0,
    top_n: int = 5,
):
    if diff_df is None or diff_df.empty or "log2FC" not in diff_df.columns:
        return None

    metric_col = sig_col if sig_col in diff_df.columns else ("q-value" if "q-value" in diff_df.columns else "p-value")
    if metric_col not in diff_df.columns:
        return None

    wfl = diff_df.copy()
    wfl["log2FC"] = pd.to_numeric(wfl["log2FC"], errors="coerce")
    wfl[metric_col] = pd.to_numeric(wfl[metric_col], errors="coerce")
    wfl = wfl.dropna(subset=["log2FC", metric_col]).sort_values("log2FC", ascending=False).reset_index(drop=True)
    if wfl.empty:
        return None

    wfl["Rank"] = wfl.index + 1
    wfl["Status"] = "Not Significant"
    wfl.loc[(wfl["log2FC"] >= fc_thr) & (wfl[metric_col] <= alpha_thr), "Status"] = "Up regulated"
    wfl.loc[(wfl["log2FC"] <= -fc_thr) & (wfl[metric_col] <= alpha_thr), "Status"] = "Down regulated"

    top_up = wfl[wfl["Status"] == "Up regulated"].head(top_n)
    top_down = wfl[wfl["Status"] == "Down regulated"].sort_values("log2FC", ascending=True).head(top_n)
    label_df = pd.concat([top_up, top_down], ignore_index=True)

    if not label_df.empty:
        label_df = label_df.copy()
        label_df["Point_ID"] = range(1, len(label_df) + 1)
        wfl["Point_ID"] = wfl["Rank"].map(label_df.set_index("Rank")["Point_ID"].to_dict())
    else:
        wfl["Point_ID"] = np.nan

    colors = {
        "Up regulated": "#d62728",
        "Down regulated": "#1f77b4",
        "Not Significant": "#bdbdbd",
    }

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    for status, sub in wfl.groupby("Status"):
        ax.scatter(
            sub["Rank"],
            sub["log2FC"],
            s=20,
            alpha=0.85,
            color=colors.get(status, "#999999"),
            label=status,
        )

    label_plot = wfl.dropna(subset=["Point_ID"]).copy()
    if not label_plot.empty:
        x_offset = max(float(wfl["Rank"].max()), 1.0) * 0.012
        y_offset = max(float(wfl["log2FC"].max() - wfl["log2FC"].min()), 1.0) * 0.018
        for _, row in label_plot.iterrows():
            x0 = float(row["Rank"])
            y0 = float(row["log2FC"])
            x1 = x0 + x_offset
            y1 = y0 + (y_offset if y0 >= 0 else -y_offset)
            ax.plot([x0, x1], [y0, y1], color="gray", linewidth=0.6, alpha=0.8)
            ax.text(
                x1,
                y1,
                str(int(row["Point_ID"])),
                fontsize=8,
                ha="left",
                va="center",
                fontweight="bold",
            )

    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(fc_thr, color="gray", linestyle="--", linewidth=0.9)
    ax.axhline(-fc_thr, color="gray", linestyle="--", linewidth=0.9)

    metric_label = "q-value" if "q" in str(metric_col).lower() else "p-value"
    ax.set_title(
        f"LFQ Glycopeptide Waterfall ({metric_label} < {alpha_thr}, |log2FC| > {fc_thr})",
        fontweight="bold",
    )
    ax.set_xlabel("Glycopeptide Rank", fontweight="bold")
    ax.set_ylabel("log2 Fold-Change", fontweight="bold")
    ax.grid(False)
    ax.legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    return fig


def _prepare_lfq_report_figures_for_export():
    diff_df = ss.get("diffexp_df")
    if isinstance(diff_df, pd.DataFrame) and not diff_df.empty and "log2FC" in diff_df.columns:
        if ss.get("report_waterfall_fig") is None:
            ss["report_waterfall_fig"] = _create_lf_waterfall_report_fig(
                diff_df=diff_df,
                sig_col=ss.get("lfq_wfl_sig_col", "q-value" if "q-value" in diff_df.columns else "p-value"),
                alpha_thr=ss.get("lfq_wfl_alpha", 0.05),
                fc_thr=ss.get("lfq_wfl_fc", 1.0),
                top_n=ss.get("lfq_wfl_topn", 5),
            )



# =============================================================================
# Hosted-safe download override
# =============================================================================
# IMPORTANT:
# On Posit/RStudio Connect, Python runs on the Connect server/container.
# Directly writing to Path.home()/Downloads will save on the server, not on the
# user's laptop. These wrappers keep direct Save-to-Downloads behavior for local
# standalone Windows/Mac, but switch to browser download buttons when hosted.

def _env_to_bool_for_downloads(value):
    if value is None:
        return None
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on", "hosted", "connect"}:
        return True
    if value in {"0", "false", "no", "n", "off", "local", "standalone"}:
        return False
    return None


def _is_hosted_mode() -> bool:
    import os

    explicit = _env_to_bool_for_downloads(os.environ.get("GLYCOEXPLORER_HOSTED"))
    if explicit is not None:
        return explicit

    if os.environ.get("POSIT_PRODUCT", "").upper() == "CONNECT":
        return True
    if os.environ.get("RSTUDIO_PRODUCT", "").upper() == "CONNECT":
        return True

    hosted_env_vars = [
        "CONNECT_CONTENT_GUID",
        "RSCONNECT_CONTENT_GUID",
        "RSCONNECT_APP_NAME",
        "CONNECT_SERVER",
        "POSIT_CONNECT",
        "RSTUDIO_CONNECT",
    ]
    return any(os.environ.get(v) for v in hosted_env_vars)


def _get_downloads_dir():
    import os
    from pathlib import Path

    custom_dir = os.environ.get("GLYCOEXPLORER_DOWNLOAD_DIR", "").strip()
    if custom_dir:
        custom_path = Path(custom_dir).expanduser()
        custom_path.mkdir(parents=True, exist_ok=True)
        return custom_path

    downloads = Path.home() / "Downloads"
    return downloads if downloads.exists() else Path.home()


def _safe_download_filename(filename):
    import re

    filename = str(filename).strip()
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1F]+', "_", filename)
    filename = re.sub(r"[^A-Za-z0-9._-]+", "_", filename)
    filename = filename.strip("._")
    return filename or "downloaded_file"


def _guess_mime_from_filename(filename):
    import mimetypes

    mime, _ = mimetypes.guess_type(str(filename))
    if mime:
        return mime

    lower = str(filename).lower()
    if lower.endswith(".csv"):
        return "text/csv"
    if lower.endswith(".tsv") or lower.endswith(".txt"):
        return "text/plain"
    if lower.endswith(".xlsx"):
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if lower.endswith(".pdf"):
        return "application/pdf"
    if lower.endswith(".zip"):
        return "application/zip"
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".svg"):
        return "image/svg+xml"
    if lower.endswith(".html"):
        return "text/html"
    return "application/octet-stream"


def _coerce_to_bytes_for_downloads(data):
    from io import BytesIO
    from pathlib import Path

    if data is None:
        raise ValueError("No file data was available.")
    if isinstance(data, BytesIO):
        return data.getvalue()
    if isinstance(data, memoryview):
        return data.tobytes()
    if isinstance(data, bytearray):
        return bytes(data)
    if isinstance(data, bytes):
        return data
    if isinstance(data, Path):
        return data.read_bytes()
    if isinstance(data, str):
        possible_path = Path(data).expanduser()
        if possible_path.exists() and possible_path.is_file():
            return possible_path.read_bytes()
        return data.encode("utf-8")
    return bytes(data)


def _hosted_download_button(data, filename, label=None, key_prefix="hosted_download"):
    import uuid
    import streamlit as st

    safe_name = _safe_download_filename(filename)
    file_bytes = _coerce_to_bytes_for_downloads(data)
    st.download_button(
        label=label or f"Download {safe_name}",
        data=file_bytes,
        file_name=safe_name,
        mime=_guess_mime_from_filename(safe_name),
        key=f"{key_prefix}_{safe_name}_{uuid.uuid4().hex[:8]}",
        use_container_width=True,
    )
    st.info("Hosted mode detected: use the browser download button above. Direct server-side saving is disabled.")
    return f"{safe_name} ready for browser download"


def _save_bytes_to_downloads(data, filename):
    """
    Local standalone: save directly to Downloads.
    Hosted/RConnect: render a browser download button instead of writing to the server.
    """
    from pathlib import Path

    file_bytes = _coerce_to_bytes_for_downloads(data)
    safe_name = _safe_download_filename(filename)

    if _is_hosted_mode():
        return _hosted_download_button(file_bytes, safe_name)

    outdir = _get_downloads_dir()
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / safe_name

    base_outpath = outpath
    counter = 1
    while outpath.exists():
        outpath = base_outpath.with_name(f"{base_outpath.stem}_{counter}{base_outpath.suffix}")
        counter += 1

    outpath.write_bytes(file_bytes)
    return outpath


def _save_file_to_downloads(source_path, filename=None):
    from pathlib import Path

    source_path = Path(source_path)
    return _save_bytes_to_downloads(source_path.read_bytes(), filename or source_path.name)


def add_hosted_safe_file_download_button(label, data, file_name, key, use_container_width=True):
    """
    One-call helper for new export buttons.
    Hosted/RConnect: st.download_button.
    Local standalone: direct Save-to-Downloads st.button.
    """
    import streamlit as st

    safe_name = _safe_download_filename(file_name)
    file_bytes = _coerce_to_bytes_for_downloads(data)

    if _is_hosted_mode():
        return st.download_button(
            label=label,
            data=file_bytes,
            file_name=safe_name,
            mime=_guess_mime_from_filename(safe_name),
            key=key,
            use_container_width=use_container_width,
        )

    if st.button(label, key=key, use_container_width=use_container_width):
        try:
            outpath = _save_bytes_to_downloads(file_bytes, safe_name)
            st.success(f"Saved to: {outpath}")
            return True
        except Exception as e:
            st.error(f"Could not save {safe_name}: {e}")
    return False

def main():
    init_session_state_defaults()

    st.sidebar.markdown("### GlycoExplorer (Label-Free)")
    st.sidebar.divider()

    species = "Human"
    version_label = "v3"

    pipeline_choice = st.sidebar.radio("Select Pipeline Engine", ["pGlyco (LFQ)", "MSFragger (LFQ)"])

    pglyco_upload = None
    msfragger_uploads = None

    if pipeline_choice == "pGlyco (LFQ)":
        pglyco_help = """**Required Columns:** `Peptide`, `Genes`, `Proteins`, `ProSites`, `RawName`, `IsotopeArea`, `Glycan(H,N,A,F)`"""
        pglyco_upload = st.sidebar.file_uploader(
            "Upload pGlyco LFQ Data Matrix", 
            type=["csv", "txt", "tsv", "xlsx", "xls"], 
            key="pglyco_lfq", 
            help=pglyco_help
        )
        merge_disabled = not pglyco_upload
        new_sig = _make_upload_sig(pglyco_upload)
    else:
        msf_help = """**Upload MULTIPLE TSV files** (one per LFQ sample run). E.g., `663_NL.tsv`. Required: `Total Glycan Composition`, `Intensity`, `Peptide`, `Protein ID`"""
        msfragger_uploads = st.sidebar.file_uploader(
            "Upload MSFragger Sample Runs", 
            type=["tsv", "txt"], 
            accept_multiple_files=True, 
            key="msf_lfq", 
            help=msf_help
        )
        merge_disabled = len(msfragger_uploads) == 0 if msfragger_uploads else True
        new_sig = tuple((f.name, f.size) for f in msfragger_uploads) if msfragger_uploads else None

    with st.sidebar.expander("Download Example Datasets", expanded=False):
        st.caption("Use these example files to see the required formatting and test the LFQ workflow.")
        script_dir = Path(__file__).resolve().parent
        example_pglyco_lfq = script_dir / "assets" / "example" / "GELF" / "pGlycoDB.csv"
        example_msf1 = script_dir / "assets" / "example" / "GELF" / "MSFragger_sample1.tsv"
        example_msf2 = script_dir / "assets" / "example" / "GELF" / "MSFragger_sample2.tsv"

        st.markdown("**pGlyco (LFQ)**")
        if example_pglyco_lfq.exists():
            add_download_button(example_pglyco_lfq.read_bytes(), "pGlycoDB_LFQ_example.csv", label="Download pGlycoDB.csv", key="save_ex_pglyco_lfq", mime="text/csv")

        st.markdown("<div style='padding-top: 10px;'></div>", unsafe_allow_html=True)

        st.markdown("**MSFragger (LFQ)**")
        col1, col2 = st.columns(2)

        if example_msf1.exists():
            with col1:
                add_download_button(example_msf1.read_bytes(), "MSFragger_sample1.tsv", label="Download Sample 1", key="save_ex_msf1", mime="text/tab-separated-values")

        if example_msf2.exists():
            with col2:
                add_download_button(example_msf2.read_bytes(), "MSFragger_sample2.tsv", label="Download Sample 2", key="save_ex_msf2", mime="text/tab-separated-values")

    if st.sidebar.button(
        "Reset session",
        key="main_reset_btn", 
        help="Clear all uploaded files, selections, and generated plots to start a fresh analysis."
    ):
        _reset_all_data(clear_cache=True)
        st.sidebar.success("Session data cleared.")
        st.rerun()

    if new_sig != ss.get("upload_sig"):
        ss["upload_sig"] = new_sig
        if ss.get("merged_ready") or ss.get("diff_done") or ss.get("merged_df") is not None or ss.get("diffexp_df") is not None:
            _reset_all_data()

    script_dir = Path(__file__).resolve().parent
    default_dir = script_dir / "assets" / "glycan_images"
    # Most users never change this, so it lives in a collapsed "Advanced" section.
    with st.sidebar.expander("Advanced: glycan image folder", expanded=False):
        user_input = st.text_input(
            "Enter folder path (optional):",
            value=str(default_dir) if default_dir.exists() else "",
            help="Folder with the glycan cartoons used in volcano and waterfall plots. "
                 "Leave as is to use the built-in images."
        )
    glycan_image_dir = Path(user_input).expanduser().resolve() if user_input else default_dir

    if not glycan_image_dir.is_dir():
        st.sidebar.error(f"Glycan image folder not found:\n{glycan_image_dir}")
        st.stop()

    # ---------------------- Merge & Process Data ----------------------
    # Before the files are uploaded the start panel below explains what to do, so the
    # (disabled) button is not drawn twice.
    with st.container(border=not merge_disabled):
        main_merge_label = "Compile LFQ Data Matrix" if not merge_disabled else "Upload data to enable processing"

        if (not merge_disabled) and st.button(
            main_merge_label,
            type="primary",
            disabled=merge_disabled,
            key="btn_merge",
            help="Compiles label-free glycopeptide identifications and intensities into an abundance matrix."
        ):
            if pipeline_choice == "pGlyco (LFQ)":
                try:
                    pglyco_df = read_pglyco_file(pglyco_upload)
                    pglyco_df = clean_cols(pglyco_df)
                    pglyco_df = normalize_pglyco_df(pglyco_df)

                    required_common = ["Peptide", "Genes", "Proteins", "ProSites", "RawName"]
                    glycan_col = None
                    for col in pglyco_df.columns:
                        clean_col = re.sub(r'\s+', '', str(col)).lower()
                        if clean_col in [
                            "glycan(h,n,a,g,f)", 
                            "glycan(h,n,a,f)", 
                            "correctedglycan(h,n,a,g,f)", 
                            "correctedglycan(h,n,a,f)"
                        ]:
                            glycan_col = col
                            break

                    missing = [c for c in required_common if c not in pglyco_df.columns]
                    if glycan_col is None:
                        missing.append("Glycan(H,N,A,G,F) or Glycan(H,N,A,F)")
                    
                    if missing:
                        st.error(f"pGlyco file is missing required columns: {', '.join(missing)}")
                        st.stop()

                    if pglyco_df.empty:
                        st.error("Uploaded pGlyco file is empty or unreadable.")
                        st.stop()

                    merged = build_label_free_matrix_from_pglyco(
                        pglyco_df, 
                        species=species, 
                        version_label=version_label
                    )
                    
                    ss["merged_df"] = merged
                    ss["raw_merged_df"] = merged.copy()
                    ss["merged_ready"] = True
                    ss["diff_done"] = False
                    ss["diffexp_df"] = None
                    ss["filtered_df"] = None
                    ss["filtered_ready"] = False
                    ss["batch_corrected"] = False
                    ss["enable_batch_correction"] = False
                    ss["batch_dict"] = {}
                    ss["batch_config"] = {}
                    ss["batch_correction_run_signature"] = None
                    ss["batch_correction_applied_method"] = "None"
                    ss["batch_combat_info"] = None
                    st.success("Successfully built pGlyco label-free matrix!")
                except Exception as e:
                    st.error(f"Processing failed: {e}")

            elif pipeline_choice == "MSFragger (LFQ)":
                try:
                    with st.spinner(f"Processing and joining {len(msfragger_uploads)} LFQ files..."):
                        merged = process_msfragger_lfq_files(msfragger_uploads)
                        if merged.empty:
                            st.error("No valid glycopeptides found after parsing the uploaded files.")
                        else:
                            ss["merged_df"] = merged
                            ss["raw_merged_df"] = merged.copy()
                            ss["merged_ready"] = True
                            ss["diff_done"] = False
                            ss["diffexp_df"] = None
                            ss["filtered_df"] = None
                            ss["filtered_ready"] = False
                            ss["batch_corrected"] = False
                            ss["enable_batch_correction"] = False
                            ss["batch_dict"] = {}
                            ss["batch_config"] = {}
                            ss["batch_correction_run_signature"] = None
                            ss["batch_correction_applied_method"] = "None"
                            ss["batch_combat_info"] = None
                            st.success(f"Successfully joined {len(msfragger_uploads)} files into {len(merged)} unique MSFragger glycoforms!")
                except Exception as e:
                    st.error(f"MSFragger LFQ Processing failed: {e}")

    # --------------------------------------

    # ---------------------- Show merged LFQ output after processing ----------------------
    downstream_active = (
        ss.get("show_global_select") or 
        ss.get("show_overview") or 
        ss.get("show_suite") or 
        ss.get("show_report")
    )

    if not downstream_active and ss.get("merged_ready") and not df_is_empty(ss.get("merged_df")):
        merged_display_df = ss["merged_df"]

        with st.expander("Integrated LFQ Glycopeptide Abundance Matrix", expanded=True):
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Glycoforms", f"{len(merged_display_df):,}")

            numeric_cols = [
                c for c in merged_display_df.columns
                if pd.api.types.is_numeric_dtype(merged_display_df[c]) and c != "PSM_Count"
            ]
            m2.metric("Samples Detected", len(numeric_cols))
            m3.metric(
                "Glycan Classes",
                merged_display_df["GlycanClass"].nunique() if "GlycanClass" in merged_display_df.columns else "N/A"
            )

            st.markdown("<br>", unsafe_allow_html=True)
            dl_col1, dl_col2, spacer = st.columns([1.5, 1.5, 7])

            csv_bytes = merged_display_df.to_csv(index=False).encode("utf-8")
            with dl_col1:
                add_download_button(csv_bytes, "LFQ_master_matrix.csv", label="Download CSV", key="save_lfq_master_csv", mime="text/csv")

            buffer_excel = BytesIO()
            with pd.ExcelWriter(buffer_excel, engine="openpyxl") as writer:
                merged_display_df.to_excel(writer, index=False, sheet_name="LFQ_Data")
            excel_bytes = buffer_excel.getvalue()
            with dl_col2:
                add_download_button(excel_bytes, "LFQ_master_matrix.xlsx", label="Download Excel", key="save_lfq_master_excel", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            st.markdown("<br>", unsafe_allow_html=True)
            tab_matrix, tab_classes = st.tabs(["Abundance matrix", "Glycan class composition"])
            with tab_matrix:
                show_grid(merged_display_df, page_size=15, height=450)
            with tab_classes:
                if "GlycanClass" in merged_display_df.columns:
                    st.caption(
                        "How the identified glycoforms are spread across glycan classes. This describes what was "
                        "identified, not data quality. To see which classes change between groups, run the "
                        "statistics and open **Glycan class breakdown** in Results Visualization."
                    )
                    create_glycan_class_summary(merged_display_df, "GlycanClass")
                else:
                    st.info("This dataset has no GlycanClass column.")


    if not ss.get("merged_ready") or df_is_empty(ss.get("merged_df")):
        ss["show_overview"] = False
        ss["show_suite"] = False
        ss["show_global_select"] = False

        render_professional_lfq_start_panel(
            pipeline_choice=pipeline_choice,
            merge_disabled=merge_disabled
        )

        st.markdown("<br>", unsafe_allow_html=True)
        faq_section()
        return

    st.markdown("<br>", unsafe_allow_html=True)

    def _reset_overview_flags():
        ss.update({
            "run_pca": False, 
            "run_corr": False, 
            "run_overview_heatmap": False, 
            "run_peptide_freq": False, 
            "run_abundance": False, 
            "run_glycan_class": False
        })

    _SECTIONS = ["show_global_select", "show_overview", "show_suite", "show_report"]

    def _toggle_section(name):
        # Runs as a button callback, i.e. before the page is redrawn, so the
        # ▸/▾ labels below always match the open section. (Toggling inside
        # "if st.button(...)" updated the label one run late; Streamlit then
        # treated the relabelled button as a new one and dropped the next click.)
        opening = not ss.get(name, False)
        for _s in _SECTIONS:
            ss[_s] = False
        ss[name] = opening
        _reset_overview_flags()

    c1, c2, c3, c4 = st.columns(4)
    for _col, _name, _label, _help, _key in [
        (c1, "show_global_select", "Data Filtering & Group Specification",
         'Choose the sample runs, put them into groups, pick the reference group and (optionally) correct batches.', "btn_global_select"),
        (c2, "show_overview", "Data Integrity Check & Sanity Profiling",
         'Quality control before testing: PCA, sample correlation, sample clustering and per-sample intensity distributions.', "btn_overview"),
        (c3, "show_suite", "Statistical Analysis Suite",
         'Pairwise tests, ANOVA or limma, then volcano, waterfall, box, heatmap, lollipop and glycan class plots.', "btn_suite"),
        (c4, "show_report", "Results Download",
         'Generate a PDF report and a ZIP bundle of all tables and figures.', "btn_report"),
    ]:
        _col.button(
            ("▾ " if ss.get(_name) else "▸ ") + _label,
            width="stretch",
            key=_key,
            help=_help,
            on_click=_toggle_section,
            args=(_name,),
        )

    # Step guide under the four buttons: what each step needs and whether it is done.
    _n_groups_ready = sum(1 for g in (ss.get("global_group_samples") or []) if g)
    _method_short = str(ss.get("stat_method_label", "Statistics done")).split(";")[0]
    c1.caption(
        f"Step 1 · :green[✓ {_n_groups_ready} groups defined]" if _n_groups_ready >= 2
        else "Step 1 · Start here: assign samples to at least 2 groups"
    )
    c2.caption("Step 2 · Check samples look as expected before testing")
    c3.caption(
        f"Step 3 · :green[✓ {_method_short}]" if ss.get("diff_done")
        else ("Step 3 · Ready to run statistics" if _n_groups_ready >= 2 else "Step 3 · Needs groups from step 1")
    )
    c4.caption("Step 4 · PDF report and ZIP of tables and figures" if ss.get("diff_done")
               else "Step 4 · Available after statistics")

    st.markdown("<br>", unsafe_allow_html=True)

    # ====================== DATA FILTERING & GROUP SPECIFICATION ======================
    if ss["show_global_select"]:
        df = ss["merged_df"]
        numeric_cols = sorted(
            [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != 'PSM_Count'], 
            key=str
        )

        def clean_label(name: str) -> str:
            s = re.sub(r"^\s*Abundance\s*:\s*", "", str(name))
            if ":" in s:
                s = s.split(":", 1)[-1]
            return s.strip()

        pretty_labels = {c: clean_label(c) for c in numeric_cols}
        fmt = lambda x: pretty_labels.get(x, x)

        def _sanitize(selected, allowed):
            allowed_set = set(allowed)
            return list(dict.fromkeys([x for x in (selected or []) if x in allowed_set]))

        with st.expander("What do I set here?", icon=":material/help:"):
            st.markdown(
                "1. **Sample columns**: the sample runs to analyse. Leave out blanks and failed runs.\n"
                "2. **Groups**: put every run into a biological group (2 or 3 groups). The statistics "
                "compare these groups.\n"
                "3. **Reference group**: the baseline, usually the control. Fold changes are reported as "
                "*other group vs reference*, so a positive log2FC means higher than the reference.\n"
                "4. **Batch correction (optional)**: only if runs were acquired in separate batches (days, "
                "columns, instruments). Every grouped run must be in a batch, and each batch should contain "
                "more than one group. If a batch holds only one group, batch and biology cannot be separated."
            )

        left, right = st.columns([0.42, 0.58], gap="large")

        with left:
            with st.container(border=True):
                st.markdown("#### Define Experimental Groups")
                # Seed the widget only when it has no value yet (first open, or after the
                # panel was closed). Overwriting it on every redraw undid the user's edits.
                if "global_sample_cols_ui" not in ss:
                    ss["global_sample_cols_ui"] = _sanitize(ss.get("global_sample_cols") or numeric_cols, numeric_cols)
                ss["global_sample_cols_ui"] = _sanitize(ss["global_sample_cols_ui"], numeric_cols)

                sample_cols = st.multiselect(
                    "Sample/abundance columns (used everywhere)", 
                    options=numeric_cols, 
                    help="Sample runs used in QC, statistics and the report. Remove blanks or failed runs.",
                    format_func=fmt, 
                    key="global_sample_cols_ui"
                )
                sample_cols = _sanitize(sample_cols, numeric_cols)

                if not sample_cols:
                    st.warning("Select at least 1 sample column.")
                    ss.update({
                        "global_sample_cols": [], 
                        "global_group_samples": [], 
                        "global_group_labels": [], 
                        "global_all_groups": [],
                        "batch_dict": {}
                    })
                else:
                    ss["global_sample_cols"] = sample_cols

                st.divider()
                n_groups = st.radio(
                    "Number of groups", 
                    [2, 3], 
                    help="Choose 3 to unlock the three-group tests (limma F-test or ANOVA).",
                    index=0 if ss.setdefault("global_n_groups_ui", 2) == 2 else 1, 
                    horizontal=True, 
                    key="global_n_groups_ui"
                )
                ss["global_n_groups"] = n_groups

                n = len(sample_cols)
                d1 = sample_cols[: max(1, n//3)]
                d2 = sample_cols[max(1, n//3): max(2, 2*n//3)]
                d3 = sample_cols[max(2, 2*n//3):]
                
                g_groups = ss.get("global_group_samples") or []
                g_labels = ss.get("global_group_labels") or []

                ss["global_grp1"] = _sanitize(
                    ss.get("global_grp1", g_groups[0] if len(g_groups) > 0 else d1), 
                    sample_cols
                )
                grp1 = _sanitize(
                    st.multiselect(
                        "Group 1 columns", 
                        options=sample_cols, 
                        format_func=fmt, 
                        key="global_grp1"
                    ), 
                    sample_cols
                )
                g1_label = st.text_input(
                    "Group 1 label", 
                    value=ss.setdefault("global_g1_label", g_labels[0] if len(g_labels) > 0 else "Group1"), 
                    key="global_g1_label"
                )

                options_g2 = [c for c in sample_cols if c not in set(grp1)]
                ss["global_grp2"] = _sanitize(
                    ss.get("global_grp2", g_groups[1] if len(g_groups) > 1 else d2), 
                    options_g2
                )
                grp2 = _sanitize(
                    st.multiselect(
                        "Group 2 columns", 
                        options=options_g2, 
                        format_func=fmt, 
                        key="global_grp2"
                    ), 
                    options_g2
                )
                g2_label = st.text_input(
                    "Group 2 label", 
                    value=ss.setdefault("global_g2_label", g_labels[1] if len(g_labels) > 1 else "Group2"), 
                    key="global_g2_label"
                )

                grp3, g3_label = [], None
                if n_groups == 3:
                    options_g3 = [c for c in sample_cols if c not in set(grp1) and c not in set(grp2)]
                    ss["global_grp3"] = _sanitize(
                        ss.get("global_grp3", g_groups[2] if len(g_groups) > 2 else d3), 
                        options_g3
                    )
                    grp3 = _sanitize(
                        st.multiselect(
                            "Group 3 columns", 
                            options=options_g3, 
                            format_func=fmt, 
                            key="global_grp3"
                        ), 
                        options_g3
                    )
                    g3_label = st.text_input(
                        "Group 3 label", 
                        value=ss.setdefault("global_g3_label", g_labels[2] if len(g_labels) > 2 else "Group3"), 
                        key="global_g3_label"
                    )
                else:
                    ss.pop("global_grp3", None)
                    ss.pop("global_g3_label", None)

                grp2 = _sanitize([c for c in grp2 if c not in set(grp1)], sample_cols)
                if n_groups == 3:
                    grp3 = _sanitize([c for c in grp3 if c not in set(grp1) and c not in set(grp2)], sample_cols)

                global_group_samples = [grp1, grp2] + ([grp3] if n_groups == 3 else [])
                global_group_labels  = [g1_label, g2_label] + ([g3_label] if n_groups == 3 else [])

                ss.update({
                    "global_group_samples": global_group_samples, 
                    "global_group_labels": global_group_labels, 
                    "global_all_groups": list(dict.fromkeys(grp1 + grp2 + (grp3 if n_groups == 3 else [])))
                })

                nonempty_groups = [
                    (i, lbl, cols) 
                    for i, (lbl, cols) in enumerate(zip(global_group_labels, global_group_samples)) 
                    if cols
                ]
                
                if nonempty_groups:
                    valid_idxs = [i for i, _, _ in nonempty_groups]
                    current = ss.get("global_control_group_idx", valid_idxs[0])
                    if current not in valid_idxs:
                        current = valid_idxs[0]

                    control_choice = st.selectbox(
                        "Global Reference (Baseline) Group", 
                        help="The baseline group, usually the control. log2FC = other group vs this group.",
                        options=valid_idxs, 
                        format_func=lambda i: global_group_labels[i], 
                        index=valid_idxs.index(current), 
                        key="global_control_choice"
                    )
                    ss["global_control_group_idx"] = control_choice
                    st.caption(f"Global reference set to: **{global_group_labels[control_choice]}**")
                else:
                    ss["global_control_group_idx"] = 0
                    st.info("Select at least one column in any group to set a global reference.")

            # ---- BATCH CORRECTION UI ----
            with st.container(border=True):
                st.markdown("#### Batch Effect Correction")

                # Only samples assigned to biological groups are eligible.
                batch_allowed_samples = [
                    c for c in (ss.get("global_all_groups", []) or [])
                    if c in sample_cols
                ]

                if not batch_allowed_samples:
                    st.info(
                        "Assign samples to Group 1/2/3 first before enabling batch correction."
                    )
                else:
                    # ---------------------------------------------------------
                    # Persistent ON/OFF state
                    # ---------------------------------------------------------
                    if "_ge_lfq_batch_toggle" not in ss:
                        ss["_ge_lfq_batch_toggle"] = bool(
                            ss.get("enable_batch_correction", False)
                        )

                    apply_batch = st.toggle(
                        "Apply Batch Effect Correction",
                        key="_ge_lfq_batch_toggle"
                    )
                    ss["enable_batch_correction"] = bool(apply_batch)

                    if apply_batch:
                        with st.container(border=True):
                            st.markdown("##### Batch Setup")

                            # -------------------------------------------------
                            # Persistent correction method
                            # -------------------------------------------------
                            batch_methods = [
                                "ComBat (Empirical Bayes)",
                                "Median Centering",
                                "Mean Centering",
                                "Standardize (Mean + Variance)"
                            ]

                            saved_method = ss.get(
                                "batch_method_selected",
                                batch_methods[0]
                            )
                            if saved_method not in batch_methods:
                                saved_method = batch_methods[0]

                            if "_ge_lfq_batch_method" not in ss:
                                ss["_ge_lfq_batch_method"] = saved_method

                            batch_method = st.selectbox(
                                "Batch Correction Method",
                                options=batch_methods,
                                help=(
                                    "ComBat is recommended for LFQ and requires "
                                    "the 'neuroCombat' package."
                                ),
                                key="_ge_lfq_batch_method"
                            )
                            ss["batch_method_selected"] = batch_method

                            # -------------------------------------------------
                            # Persistent number of batches
                            # -------------------------------------------------
                            saved_n = int(ss.get("batch_num_selected", 2))
                            saved_n = max(2, min(10, saved_n))

                            if "_ge_lfq_num_batches" not in ss:
                                ss["_ge_lfq_num_batches"] = saved_n

                            num_batches = int(
                                st.number_input(
                                    "Number of Batches",
                                    min_value=2,
                                    max_value=10,
                                    step=1,
                                    key="_ge_lfq_num_batches"
                                )
                            )
                            ss["batch_num_selected"] = num_batches

                            # -------------------------------------------------
                            # Persistent sample assignments
                            # -------------------------------------------------
                            stored_batch_config = (
                                ss.get("batch_config", {})
                                or ss.get("batch_dict", {})
                                or {}
                            )

                            batch_dict = {}
                            selected_so_far = []

                            for b in range(num_batches):
                                batch_name = f"Batch {b+1}"
                                widget_key = f"_ge_lfq_batch_sel_{b}"

                                if widget_key in ss:
                                    prev_source = ss.get(widget_key, [])
                                else:
                                    prev_source = stored_batch_config.get(
                                        batch_name, []
                                    )

                                prev_selected = [
                                    c for c in (prev_source or [])
                                    if c in batch_allowed_samples
                                ]

                                available_for_batch = [
                                    c for c in batch_allowed_samples
                                    if c not in selected_so_far
                                    or c in prev_selected
                                ]

                                prev_selected = [
                                    c for c in prev_selected
                                    if c in available_for_batch
                                ]

                                # Re-seed widget state when the panel is rebuilt.
                                ss[widget_key] = prev_selected

                                b_cols = st.multiselect(
                                    f"{batch_name} Samples",
                                    options=available_for_batch,
                                    format_func=fmt,
                                    key=widget_key
                                )

                                b_cols = [
                                    c for c in b_cols
                                    if c in batch_allowed_samples
                                ]

                                batch_dict[batch_name] = b_cols
                                selected_so_far.extend(
                                    [c for c in b_cols if c not in selected_so_far]
                                )

                            # Save immediately so navigation cannot erase the setup.
                            ss["batch_config"] = {
                                k: list(v) for k, v in batch_dict.items()
                            }

                            assigned_samples = [
                                c for cols in batch_dict.values() for c in cols
                            ]

                            duplicated = sorted({
                                c for c in assigned_samples
                                if sum(
                                    c in cols
                                    for cols in batch_dict.values()
                                ) > 1
                            })

                            unassigned_samples = [
                                c for c in batch_allowed_samples
                                if c not in assigned_samples
                            ]

                            nonempty_batches = {
                                k: v for k, v in batch_dict.items() if v
                            }

                            if duplicated:
                                st.error(
                                    "A sample cannot be assigned to more than one batch: "
                                    + ", ".join([fmt(c) for c in duplicated])
                                )

                            if unassigned_samples:
                                st.warning(
                                    "These grouped samples are not assigned to any batch: "
                                    + ", ".join(
                                        [fmt(c) for c in unassigned_samples]
                                    )
                                )

                            if len(nonempty_batches) < 2:
                                st.warning(
                                    "Assign samples to at least two non-empty batches."
                                )

                            current_signature = (
                                batch_method,
                                num_batches,
                                tuple(
                                    (name, tuple(cols))
                                    for name, cols in batch_dict.items()
                                )
                            )

                            # Changing the setup invalidates an earlier correction.
                            previous_signature = ss.get(
                                "batch_correction_run_signature"
                            )
                            if (
                                ss.get("batch_corrected", False)
                                and previous_signature is not None
                                and previous_signature != current_signature
                            ):
                                ss["merged_df"] = ss["raw_merged_df"].copy()
                                ss["batch_corrected"] = False
                                ss["batch_dict"] = {}
                                ss["batch_correction_applied_method"] = "None"
                                ss["batch_combat_info"] = None
                                ss["diffexp_df"] = None
                                ss["filtered_df"] = None
                                ss["diff_done"] = False
                                ss["filtered_ready"] = False

                                st.info(
                                    "Batch configuration changed. "
                                    "Click **Run Batch Correction** again."
                                )

                            can_run_batch = (
                                not duplicated
                                and not unassigned_samples
                                and len(nonempty_batches) >= 2
                            )

                            if st.button(
                                "Run Batch Correction",
                                type="primary",
                                key="run_batch_correction_btn_lfq",
                                disabled=not can_run_batch
                            ):
                                try:
                                    # LFQ batch correction is performed on log2 values.
                                    # Always start from the original uncorrected matrix.
                                    tmp_vals = (
                                        ss["raw_merged_df"][batch_allowed_samples]
                                        .apply(pd.to_numeric, errors="coerce")
                                        .astype(float)
                                    )
                                    tmp_vals[tmp_vals <= 0] = np.nan
                                    tmp_log = np.log2(tmp_vals)

                                    batch_list = list(
                                        nonempty_batches.values()
                                    )

                                    # ComBat must know the biological group of
                                    # every sample, or it strips the real group
                                    # differences together with the batch effect.
                                    group_map = {
                                        col: lbl
                                        for lbl, cols in zip(
                                            ss.get("global_group_labels", []) or [],
                                            ss.get("global_group_samples", []) or [],
                                        )
                                        for col in (cols or [])
                                    }

                                    corrected_log = apply_batch_correction(
                                        tmp_log,
                                        batch_list,
                                        batch_method,
                                        group_map=group_map,
                                        batch_names=list(
                                            nonempty_batches.keys()
                                        ),
                                        strict=False
                                    )

                                    ss["batch_combat_info"] = (
                                        corrected_log.attrs.get("combat_info")
                                    )

                                    corrected_raw = (
                                        2 ** corrected_log
                                    ).fillna(0)

                                    corrected_df = ss["raw_merged_df"].copy()
                                    corrected_df[
                                        batch_allowed_samples
                                    ] = corrected_raw[
                                        batch_allowed_samples
                                    ]

                                    ss["merged_df"] = corrected_df
                                    ss["batch_dict"] = {
                                        k: list(v)
                                        for k, v in nonempty_batches.items()
                                    }
                                    ss["batch_config"] = {
                                        k: list(v)
                                        for k, v in batch_dict.items()
                                    }
                                    ss["batch_corrected"] = True
                                    ss["batch_correction_run_signature"] = (
                                        current_signature
                                    )
                                    ss["batch_correction_applied_method"] = (
                                        batch_method
                                    )

                                    # Invalidate downstream results based on old data.
                                    ss["diffexp_df"] = None
                                    ss["filtered_df"] = None
                                    ss["diff_done"] = False
                                    ss["filtered_ready"] = False

                                    st.success(
                                        f"{batch_method} applied successfully. "
                                        "QC PCA can show before or after batch correction, "
                                        "and downstream analysis will use corrected abundances."
                                    )

                                    _cb_info = ss.get("batch_combat_info") or {}
                                    for _w in _cb_info.get("warnings", []):
                                        st.warning(_w)
                                    if _cb_info.get("confounding_report"):
                                        with st.expander(
                                            "Batch / group design check"
                                        ):
                                            st.text(
                                                _cb_info["confounding_report"]
                                            )

                                except ImportError as i_err:
                                    st.error(str(i_err))
                                except RuntimeError as r_err:
                                    st.error(str(r_err))
                                except Exception as e:
                                    st.error(
                                        f"Batch correction failed: {e}"
                                    )

                            elif ss.get("batch_corrected", False):
                                st.success(
                                    f"{ss.get('batch_correction_applied_method', batch_method)} "
                                    "applied successfully. QC PCA can show before or after "
                                    "batch correction, and downstream analysis will use "
                                    "corrected abundances."
                                )

                                _cb_info = ss.get("batch_combat_info") or {}
                                for _w in _cb_info.get("warnings", []):
                                    st.warning(_w)
                                if _cb_info.get("confounding_report"):
                                    with st.expander(
                                        "Batch / group design check"
                                    ):
                                        st.text(
                                            _cb_info["confounding_report"]
                                        )

                    else:
                        # Turning correction OFF restores original abundances.
                        # Keep batch_config so the setup can be restored if re-enabled.
                        if (
                            ss.get("batch_corrected", False)
                            and isinstance(
                                ss.get("raw_merged_df"),
                                pd.DataFrame
                            )
                            and not ss["raw_merged_df"].empty
                        ):
                            ss["merged_df"] = ss["raw_merged_df"].copy()
                            ss["diffexp_df"] = None
                            ss["filtered_df"] = None
                            ss["diff_done"] = False
                            ss["filtered_ready"] = False

                        ss["batch_dict"] = {}
                        ss["batch_corrected"] = False
                        ss["batch_correction_run_signature"] = None
                        ss["batch_correction_applied_method"] = "None"
                        ss["batch_combat_info"] = None

        with right:
            with st.container(border=True):
                st.markdown("#### Sample Mapping Summary")
                sample_cols = ss.get("global_sample_cols", [])
                group_samples = ss.get("global_group_samples", [])
                group_labels = ss.get("global_group_labels", [])
                control_idx = ss.get("global_control_group_idx", 0)
                batch_dict = (
                    ss.get("batch_config", {})
                    if ss.get("enable_batch_correction", False)
                    else ss.get("batch_dict", {})
                )

                rows = []
                for col in sample_cols:
                    grp_name, grp_num = "Unassigned", None
                    for i, cols in enumerate(group_samples):
                        if col in (cols or []):
                            grp_name = group_labels[i] if i < len(group_labels) else f"Group{i+1}"
                            grp_num = i + 1
                            break
                            
                    batch_name = "N/A"
                    for b_name, b_cols in batch_dict.items():
                        if col in b_cols:
                            batch_name = (
                                b_name
                                if ss.get("enable_batch_correction", False)
                                else "Unassigned"
                            )
                            break

                    rows.append({
                        "Column": col, 
                        "Sample": fmt(col), 
                        "Group": grp_name, 
                        "Group_Number": grp_num, 
                        "Batch": batch_name,
                        "Is_Reference": (grp_num is not None and (grp_num - 1) == control_idx)
                    })

                meta_df = pd.DataFrame(rows)
                st.dataframe(meta_df, use_container_width=True, hide_index=True)
                add_download_button(
                    meta_df.to_csv(index=False).encode("utf-8"),
                    "sample_metadata_mapping.csv",
                    label="Download sample metadata CSV",
                    key="save_sample_metadata_csv",
                    mime="text/csv",
                )

                if not meta_df.empty and (meta_df["Group"] == "Unassigned").sum() > 0:
                    unassigned_count = (meta_df['Group'] == 'Unassigned').sum()
                    st.warning(f"{unassigned_count} selected sample(s) are not assigned to any group.")

    # ====================== DATA INTEGRITY CHECK & SANITY PROFILING ======================
    elif ss.show_overview:
        df = ss.merged_df
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != 'PSM_Count']

        def clean_label(name: str) -> str:
            s = re.sub(r"^\s*Abundance\s*:\s*", "", str(name))
            if ":" in s:
                s = s.split(":", 1)[-1]
            return s.strip()

        pretty_labels = {c: clean_label(c) for c in numeric_cols}
        fmt = lambda x: pretty_labels.get(x, x)

        sample_cols = ss.get("global_sample_cols") or numeric_cols
        group_samples = ss.get("global_group_samples") or []
        group_labels = ss.get("global_group_labels") or []
        n_groups = ss.get("global_n_groups", 2)

        if not sample_cols:
            st.warning("No numeric abundance columns detected.")
        else:
            if not group_samples or not any(group_samples):
                n = len(sample_cols)
                if n_groups == 3:
                    group_samples = [
                        sample_cols[: max(1, n // 3)], 
                        sample_cols[max(1, n // 3): max(2, 2 * n // 3)], 
                        sample_cols[max(2, 2 * n // 3):]
                    ]
                    group_labels = ["Group1", "Group2", "Group3"]
                else:
                    group_samples = [sample_cols[:max(1, n // 2)], sample_cols[max(1, n // 2):]]
                    group_labels = ["Group1", "Group2"]

            all_groups = ss.get("global_all_groups") or list(dict.fromkeys([c for g in group_samples for c in g]))
            if not all_groups:
                all_groups = sample_cols

            left, right = st.columns([0.42, 0.58], gap="large")

            with left:
                with st.container(border=True):
                    with st.expander("Assigned Sample Mapping", expanded=False):
                        if not sample_cols:
                            st.info("No samples selected in Data Filtering & Group Specification.")
                        else:
                            rows = []
                            for col in sample_cols:
                                grp_name, grp_num, is_ctrl = "Unassigned", None, False
                                for i, enumerate_cols in enumerate(group_samples or []):
                                    if col in (enumerate_cols or []):
                                        grp_name = group_labels[i] if i < len(group_labels) else f"Group{i+1}"
                                        grp_num = i + 1
                                        is_ctrl = (ss.get("global_control_group_idx", None) is not None and i == ss.get("global_control_group_idx", None))
                                        break
                                rows.append({
                                    "Sample": fmt(col), 
                                    "Group": grp_name, 
                                    "Group #": grp_num, 
                                    "Is Reference": is_ctrl
                                })
                                
                            assigned_df = pd.DataFrame(rows)[pd.DataFrame(rows)["Group"] != "Unassigned"].copy()
                            
                            if assigned_df.empty:
                                st.info("No selected samples are assigned to a group yet.")
                            else:
                                st.dataframe(assigned_df, use_container_width=True, hide_index=True)

                    # Updated to mirror the TMT module options (Removed Heatmap, added Sample Clustering)
                    mode = st.radio(
                        "Analysis Module",
                        [
                            "Principal Component Analysis (PCA)",
                            "Correlation Matrix",
                            "Sample Clustering",
                            "Abundance Density",
                        ],
                        index=None,
                        key="overview_mode",
                        label_visibility="collapsed",
                        format_func=lambda m: {
                            "Abundance Density": "Intensity Distribution (per sample)"
                        }.get(m, m),
                    )

                    _qc_help = {
                        "Principal Component Analysis (PCA)":
                            "Each dot is a sample run. Runs of the same group should sit together. A run far from "
                            "its group may be an outlier (poor injection, failed run). If runs separate by batch or "
                            "acquisition date instead of by group, consider batch correction.",
                        "Correlation Matrix":
                            "Correlation of intensities between runs. Replicates of one group should correlate "
                            "highly. A run that correlates poorly with all others needs a closer look before testing.",
                        "Sample Clustering":
                            "Clusters runs on the most variable glycoforms without using the group labels. If "
                            "runs cluster by group, biology dominates. If they cluster by batch or run order, "
                            "check for batch effects.",
                        "Abundance Density":
                            "log2 intensity distribution of each run. The curves should roughly overlap. A curve "
                            "shifted left or right means that run had less or more material injected. The statistics "
                            "in this app do not correct for that, so check this before testing. A bump at 0 comes "
                            "from glycoforms not detected in that run.",
                    }
                    if mode in _qc_help:
                        st.info(_qc_help[mode], icon=":material/lightbulb:")
                    else:
                        st.caption("Pick a check above. Look at PCA first: it shows most problems at a glance.")
                    top_n_peptides = st.session_state.get("topn_pf", 20)
                    
                    if (mode or "").strip().lower() == "feature frequency profiling": 
                        top_n_peptides = st.slider("Top N peptides", 5, 50, top_n_peptides, 5, key="topn_pf")

                    if mode == "Principal Component Analysis (PCA)":
                        st.markdown("**PCA Display Options**")

                        if (
                            ss.get("batch_corrected", False)
                            and isinstance(
                                ss.get("raw_merged_df"),
                                pd.DataFrame
                            )
                            and not ss["raw_merged_df"].empty
                        ):
                            pca_data_source = st.radio(
                                "Data Source",
                                [
                                    "After Batch Correction",
                                    "Before Batch Correction"
                                ],
                                horizontal=True,
                                key="lfq_pca_data_source"
                            )

                            if pca_data_source == "Before Batch Correction":
                                pca_source_df = ss["raw_merged_df"]
                            else:
                                pca_source_df = ss["merged_df"]

                            st.caption(
                                "Switch between the same samples before and after "
                                "batch correction."
                            )
                        else:
                            pca_data_source = "Current Data"
                            pca_source_df = df

                        data_for_pca = pca_source_df[all_groups].dropna()
                        if not data_for_pca.empty and len(data_for_pca.columns) > 0:
                            n_comps = min(10, data_for_pca.shape[0], data_for_pca.shape[1])
                            pc_cols = [f"PC{i+1}" for i in range(n_comps)]
                        else:
                            pc_cols = ["PC1", "PC2"]
                        
                        col_x, col_y = st.columns(2)
                        with col_x:
                            ss["pca_x"] = st.selectbox("X-axis Component", options=pc_cols, index=0, key="pca_x_ui")
                        with col_y:
                            ss["pca_y"] = st.selectbox(
                                "Y-axis Component", 
                                options=pc_cols, 
                                index=1 if len(pc_cols) > 1 else 0, 
                                key="pca_y_ui"
                            )
                            
                        ss["pca_draw_ellipses"] = st.checkbox("Draw confidence ellipses", value=True, key="pca_ellipses_ui")
                        pca_opt1, pca_opt2 = st.columns(2)
                        with pca_opt1:
                            ss["pca_hide_outliers"] = st.checkbox(
                                "Hide outliers", value=False, key="pca_hide_outliers_ui",
                                help="Remove IQR-based outliers from the displayed PCA plot."
                            )
                        with pca_opt2:
                            ss["pca_label_outliers"] = st.checkbox(
                                "Label outliers", value=True, key="pca_label_outliers_ui",
                                help="Add sample names next to detected outliers."
                            )
                        ss["pca_zoom_q"] = st.slider(
                            "Zoom view", min_value=0.80, max_value=1.00, value=1.00, step=0.01,
                            key="pca_zoom_q_ui",
                            help="1.00 shows the full PCA. Lower values zoom into the central samples."
                        )
                        
                        st.markdown("**Customize Group Colors**")
                        default_palette = ["#b3e2cd", "#fdcdac", "#cbd5e8", "#decbe4", "#ccebc5", "#fed9a6"]
                        ss["pca_palette"] = {}
                        cols = st.columns(max(1, len(group_labels)))
                        for i, label in enumerate(group_labels):
                            with cols[i % len(cols)]:
                                ss["pca_palette"][label] = st.color_picker(
                                    label, 
                                    value=default_palette[i % len(default_palette)], 
                                    key=f"pca_color_{i}_ui"
                                )

                    if mode == "Sample Clustering":
                        st.markdown("**Sample clustering options**")
                        ss["overview_hm_topn"] = st.slider(
                            "Top N most variable features used for clustering", 10, 300, 50,
                            key="overview_hm_topn_ui"
                        )
                        ss["overview_hm_norm"] = st.radio(
                            "Data Scaling", ["Z-score", "Raw Abundance"],
                            key="overview_hm_norm_ui", horizontal=True
                        )
                        st.caption("Samples are clustered without using group labels. Group colors are used only to color sample labels.")

                        st.markdown("**Customize Group Colors**")
                        default_cluster_palette = ["#b3e2cd", "#fdcdac", "#cbd5e8", "#decbe4", "#ccebc5", "#fed9a6"]
                        ss["cluster_palette"] = {}
                        cluster_cols = st.columns(max(1, len(group_labels)))
                        for i, label in enumerate(group_labels):
                            with cluster_cols[i % len(cluster_cols)]:
                                ss["cluster_palette"][label] = st.color_picker(
                                    label,
                                    value=default_cluster_palette[i % len(default_cluster_palette)],
                                    key=f"cluster_color_{i}_ui"
                                )

                    flags = {
                        "Principal Component Analysis (PCA)": dict(
                            run_pca=True, run_corr=False, run_overview_heatmap=False, 
                            run_peptide_freq=False, run_abundance=False, run_glycan_class=False
                        ), 
                        "Correlation Matrix": dict(
                            run_pca=False, run_corr=True, run_overview_heatmap=False, 
                            run_peptide_freq=False, run_abundance=False, run_glycan_class=False
                        ), 
                        "Sample Clustering": dict(
                            run_pca=False, run_corr=False, run_overview_heatmap=True, 
                            run_peptide_freq=False, run_abundance=False, run_glycan_class=False
                        ),
                        "Abundance Density": dict(
                            run_pca=False, run_corr=False, run_overview_heatmap=False,
                            run_peptide_freq=False, run_abundance=True, run_glycan_class=False
                        ),
                    }
                    
                    for k in [
                        "run_pca", "run_corr", "run_overview_heatmap", 
                        "run_peptide_freq", "run_abundance", "run_glycan_class"
                    ]:
                        ss[k] = False
                        
                    if mode is not None:
                        ss.update(flags[mode])

            with right:
                if ss.get("run_pca", False):
                    if len(all_groups) >= 1:
                        batch_was_applied = (
                            ss.get("batch_corrected", False)
                            and isinstance(
                                ss.get("raw_merged_df"),
                                pd.DataFrame
                            )
                            and not ss["raw_merged_df"].empty
                        )

                        if batch_was_applied:
                            selected_pca_source = ss.get(
                                "lfq_pca_data_source",
                                "After Batch Correction"
                            )

                            if selected_pca_source == "Before Batch Correction":
                                pca_df = ss["raw_merged_df"]
                                pca_title = "PCA Before Batch Correction"
                            else:
                                pca_df = ss["merged_df"]
                                pca_title = "PCA After Batch Correction"

                            st.success(
                                f"{ss.get('batch_correction_applied_method', 'Batch correction')} "
                                "is active. Downstream analysis uses corrected abundances."
                            )
                            st.caption(
                                "Use **Data Source** in PCA Display Options to switch "
                                "between before and after batch correction."
                            )
                        else:
                            pca_df = df
                            pca_title = "PCA"

                        # One PCA at a time prevents duplicate download-button keys
                        # created internally by create_pca_plot().
                        create_pca_plot(
                            df=pca_df,
                            group_samples=group_samples,
                            group_labels=group_labels,
                            title=pca_title,
                            pc_x=ss.get("pca_x", "PC1"),
                            pc_y=ss.get("pca_y", "PC2"),
                            draw_ellipses=ss.get(
                                "pca_draw_ellipses", False
                            ),
                            hide_outliers=ss.get(
                                "pca_hide_outliers", False
                            ),
                            label_outliers=ss.get(
                                "pca_label_outliers", True
                            ),
                            zoom_q=ss.get("pca_zoom_q", 1.0),
                            palette=ss.get("pca_palette", None),
                            quant_mode="LFQ",
                            key_suffix="single_pca"
                        )
                    else:
                        st.info(
                            "No samples available for PCA "
                            "(check global selection)."
                        )

                elif ss.get("run_corr", False):
                    if len(all_groups) >= 2:
                        fig_obj = create_sample_corr_heatmap(df, all_groups)
                        if fig_obj is not None:
                            ss["report_corr_fig"] = fig_obj
                        _add_desktop_plot_save_buttons(
                            fig_obj,
                            "LFQ_sample_correlation_heatmap",
                            "lfq_corr_heatmap"
                        )
                    else:
                        st.warning("Need at least two sample columns for correlation heatmap.")

                elif ss.get("run_overview_heatmap", False):
                    grouped_samples_for_cluster = [
                        [c for c in (cols or []) if c in all_groups]
                        for cols in (group_samples or [])
                    ]
                    grouped_labels_for_cluster = list(group_labels or [])

                    if len(all_groups) >= 2 and grouped_samples_for_cluster and any(grouped_samples_for_cluster):
                        fig_obj = create_sample_clustering(
                            df=df,
                            group_samples=grouped_samples_for_cluster,
                            group_labels=grouped_labels_for_cluster,
                            title="Sample clustering",
                            top_n=ss.get("overview_hm_topn", 50),
                            norm_type=ss.get("overview_hm_norm", "Z-score"),
                            quant_mode="LFQ",
                            palette=ss.get("cluster_palette", None),
                            key_suffix="overview_cluster"
                        )
                    elif len(all_groups) >= 2:
                        fig_obj = create_sample_clustering(
                            df=df,
                            group_samples=[all_groups],
                            group_labels=["All selected samples"],
                            title="Sample clustering",
                            top_n=ss.get("overview_hm_topn", 50),
                            norm_type=ss.get("overview_hm_norm", "Z-score"),
                            quant_mode="LFQ",
                            palette={"All selected samples": "#b3e2cd"},
                            key_suffix="overview_cluster"
                        )
                    else:
                        fig_obj = None
                        st.warning("Need at least two sample columns for sample clustering.")

                    if fig_obj is not None:
                        ss["report_overview_heatmap_fig"] = fig_obj
                        _add_desktop_plot_save_buttons(
                            fig_obj,
                            "LFQ_sample_clustering",
                            "lfq_sample_clustering"
                        )

                elif ss.get("run_peptide_freq", False):
                    fig_obj = create_peptide_frequency_plot(df, "Peptide", top_n_peptides)
                    _add_desktop_plot_save_buttons(
                        fig_obj,
                        "LFQ_peptide_frequency",
                        "lfq_peptide_frequency"
                    )
                elif ss.get("run_abundance", False):
                    if len(all_groups) >= 1:
                        # create_abundance_density adds its own download buttons.
                        create_abundance_density(df, all_groups)
                        # The report looks for "report_density_fig".
                        ss["report_density_fig"] = ss.get("report_abundance_density_fig")
                    else:
                        st.info("No numeric sample columns detected for abundance density.")
                else:
                    st.info("Choose an Analysis Module on the left to begin.")

    # ====================== STATISTICAL ANALYSIS SUITE ======================
    if ss["show_suite"]:
        tabs = st.tabs(["Differential Analysis", "Results Visualization"])

        with tabs[0]:
            merged_df = ss.get("merged_df")
            if df_is_empty(merged_df):
                st.info("Upload and process data first.")
            else:
                group_samples = ss.get("global_group_samples", [])
                group_labels = ss.get("global_group_labels", [])
                n_groups = ss.get("global_n_groups", 2)
                ctrl_idx = ss.get("global_control_group_idx", 0)
                
                nonempty_groups = [
                    (i, lbl, cols) 
                    for i, (lbl, cols) in enumerate(zip(group_labels, group_samples)) 
                    if cols
                ]
                
                if len(nonempty_groups) < 2: 
                    st.warning("You must assign samples to at least **two different groups** in the 'Data Filtering & Group Specification' module before running statistical analysis.")
                else:
                    valid_idxs = [i for i, _, _ in nonempty_groups]
                    if ctrl_idx not in valid_idxs:
                        ctrl_idx = valid_idxs[0]
                        ss["global_control_group_idx"] = valid_idxs[0]
                        
                    ss.update({
                        "group_samples": group_samples, 
                        "group_labels": group_labels, 
                        "control_group_idx": ctrl_idx, 
                        "all_groups": list(dict.fromkeys([c for g in group_samples for c in g]))
                    })

                    with st.expander("Current Group Specification (read-only)", expanded=True):
                        st.write(f"**Number of groups:** {n_groups}\n**Reference group:** {group_labels[ctrl_idx]}")
                        for i, (lbl, cols) in enumerate(zip(group_labels, group_samples), start=1):
                            st.write(f"**Group {i} — {lbl}:** {len(cols)} samples")
                            st.caption(", ".join([str(c) for c in cols]) if cols else "(empty)")

                    ss.setdefault("compare_open", False)
                    ss.setdefault("filter_open", False)
                    
                    def _toggle_compare():
                        ss["compare_open"] = not ss.get("compare_open", False)

                    st.button(
                        ("▸ " if not ss["compare_open"] else "▾ ") + "Select Statistical Test",
                        key="btn_toggle_compare",
                        width="stretch",
                        help="Open the panel to choose the design and test, then run the analysis.",
                        on_click=_toggle_compare,
                    )

                    if not ss["compare_open"]:
                        st.info("Click Select Statistical Test to choose pairwise or ANOVA and run the analysis.")
                    else:
                        with st.container(border=True):
                            st.markdown("#### Test Parameters and Execution")

                            with st.expander("Which test should I use?", icon=":material/help:"):
                                st.markdown(
                                    "| Your design | Suggested test |\n"
                                    "|---|---|\n"
                                    "| 2 groups, independent samples, 3-6 replicates | **limma moderated t-test** |\n"
                                    "| 2 groups, independent samples, 8+ replicates | Welch's t-test or limma |\n"
                                    "| Same subjects measured twice (matched runs) | Paired t-test |\n"
                                    "| 3 groups | **limma moderated F-test** (default), then read the pairwise q-values |\n\n"
                                    "**Why limma?** With few replicates, each glycoform's variance is estimated from only "
                                    "a handful of values, so ordinary t-tests produce many false hits (tiny variance by "
                                    "chance) and miss real ones. limma shrinks each variance towards the typical variance "
                                    "of all glycoforms, which makes the test both more reliable and more powerful.\n\n"
                                    "**Missing values (label-free):** the classical tests need a value in every sample, so "
                                    "by default they replace 0/missing with 20% of the glycoform's smallest measured "
                                    "intensity. limma does not need that: it fits each glycoform on the values that were "
                                    "measured. Glycoforms missing from a whole compared group therefore get no limma "
                                    "p-value (listed at the bottom). Tick *Replace 0/NA* to impute them anyway; their "
                                    "fold changes are then driven by the imputed value, so read them with care.\n\n"
                                    "**Scale:** limma works on log2 intensities. The other tests run on raw intensities, "
                                    "so their p-values and log2FC can differ from limma's.\n\n"
                                    "**Presence filter:** drops glycoforms seen in too few samples before any test, so "
                                    "sparse identifications do not use up the multiple-testing budget.\n\n"
                                    "**q-value** is the Benjamini-Hochberg adjusted p-value (false discovery rate)."
                                )

                            nonempty = [(lbl, cols) for lbl, cols in zip(group_labels, group_samples) if cols]
                            nonempty_idx = [i for i, cols in enumerate(group_samples) if cols]
                            can_pairwise = len(nonempty) >= 2
                            can_anova = (len(nonempty) == 3) and (n_groups == 3)
                            labels_unique = len({lbl for lbl, _ in nonempty}) == len(nonempty)

                            # No st.form here on purpose: inside a form the options below
                            # would not update when the design or test is changed.
                            DESIGN_TWO = "Two groups (pairwise)"
                            DESIGN_THREE = "All three groups (limma F-test or ANOVA)"
                            comparison_choices = [
                                DESIGN_TWO if can_pairwise else DESIGN_TWO + " — need ≥2 non-empty groups",
                                DESIGN_THREE if can_anova else DESIGN_THREE + " — need exactly 3 non-empty groups",
                            ]
                            if ss.get("stat_design") not in comparison_choices:
                                ss.pop("stat_design", None)
                            comparison_type = st.radio(
                                "Select experimental design:",
                                comparison_choices,
                                index=1 if can_anova else 0,
                                horizontal=True,
                                key="stat_design",
                                help="Two groups: compares one pair and gives a fold change (volcano, waterfall). "
                                     "All three groups: tests whether any group differs, and limma also reports "
                                     "every pairwise comparison in the same table.",
                            )
                            is_pairwise_design = comparison_type.startswith(DESIGN_TWO)

                            selected_pair = None
                            pairs = {}
                            specific_test = PAIRWISE_TESTS[0]
                            multigroup_test = MULTIGROUP_TESTS[0]
                            limma_all_groups = True

                            if is_pairwise_design:
                                if not can_pairwise:
                                    st.warning("Please define at least two non-empty groups.")
                                else:
                                    pairs = {
                                        f"{group_labels[i]} vs {group_labels[j]}": (i, j)
                                        for a, i in enumerate(nonempty_idx)
                                        for j in nonempty_idx[a + 1:]
                                    }
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        selected_pair = st.selectbox(
                                            "Select pairwise comparison:",
                                            list(pairs.keys()),
                                            key="stat_pair",
                                            help="log2FC is always reported as the other group versus the reference group.",
                                        )
                                    with col2:
                                        specific_test = st.selectbox(
                                            "Select statistical test:",
                                            PAIRWISE_TESTS,
                                            key="stat_test",
                                            help="Not sure? Open 'Which test should I use?' above.",
                                        )
                                    st.caption(TEST_HELP.get(specific_test, ""))

                                    if specific_test == LIMMA_T and len(nonempty) > 2:
                                        limma_all_groups = st.checkbox(
                                            "Use all groups to estimate variance (recommended)",
                                            value=True,
                                            key="stat_limma_all_groups",
                                            help="Standard limma practice: the third group adds information about "
                                                 "measurement noise and gives more power. Turn off to use only the "
                                                 "two compared groups.",
                                        )
                            else:
                                selected_pair = "Three-group ANOVA"
                                if can_anova:
                                    multigroup_test = st.radio(
                                        "Select statistical test:",
                                        MULTIGROUP_TESTS,
                                        key="stat_multigroup_test",
                                        captions=[
                                            "Recommended. log2 scale, stable with few replicates. Also gives log2FC "
                                            "and q-value for every pair of groups.",
                                            "Classical test on raw intensities. Only says whether any group differs.",
                                        ],
                                        help=TEST_HELP[LIMMA_F],
                                    )
                                else:
                                    st.warning("Define exactly three non-empty groups in Data Filtering & Group Specification.")

                            uses_limma = (specific_test == LIMMA_T) if is_pairwise_design else (multigroup_test == LIMMA_F)

                            st.divider()
                            st.markdown("##### Preprocessing (before statistics)")
                            # Separate widget keys so limma and the classical tests keep
                            # their own sensible default for imputation.
                            if uses_limma:
                                do_zero_impute = st.checkbox(
                                    "Replace 0/NA intensities before differential",
                                    value=False,
                                    key="lf_impute_limma",
                                    help="Off is recommended for limma: it uses only measured values. Turn on to "
                                         "also test glycoforms missing from a whole group (fold changes then depend "
                                         "on the imputed value).",
                                )
                            else:
                                do_zero_impute = st.checkbox(
                                    "Replace 0/NA intensities before differential",
                                    value=True,
                                    key="lf_impute_classic",
                                    help="Replaces 0/missing with 20% of the glycoform's smallest measured intensity. "
                                         "The classical tests need a value in every sample.",
                                )
                            zero_frac = 0.2
                            do_presence_filter = st.checkbox(
                                "Filter by presence (drop sparse rows)",
                                value=True,
                                key="lf_presence_on",
                                help="Drops glycoforms measured in too few samples before testing.",
                            )
                            presence_min_frac = st.slider(
                                "Presence threshold (fraction)", 0.05, 1.00, 0.40, 0.05,
                                key="lf_presence_frac",
                                help="Fraction of samples that must have a value > 0.",
                            )

                            rule_options = {
                                "Total samples only": "all_only",
                                "At least one group only": "group_only",
                                "OR: Total samples OR one group": "or",
                                "AND: Total samples AND one group": "and"
                            }
                            rule_choice = st.radio(
                                "Presence rule",
                                list(rule_options.keys()),
                                index=2,
                                key="lf_presence_rule",
                                help="Where the threshold is applied: across all compared samples, within at least "
                                     "one group, or either/both. 'At least one group' keeps glycoforms that are "
                                     "present in one condition and absent in another.",
                            )
                            presence_rule_key = rule_options[rule_choice]

                            run_button = st.button(
                                "Run Analysis",
                                type="primary",
                                key="btn_run_diff",
                                help="Execute the selected statistical test to generate p-values, q-values and fold changes.",
                            )

                            if run_button:
                                ss["filter_open"] = False
                                # A new test makes any earlier filtered subset stale.
                                ss["filtered_df"] = None
                                ss["filtered_ready"] = False
                                prep = dict(
                                    do_zero_impute=do_zero_impute,
                                    zero_frac=zero_frac,
                                    do_presence_filter=do_presence_filter,
                                    presence_min_frac=presence_min_frac,
                                    presence_rule=presence_rule_key,
                                )
                                prep_note = (
                                    ("presence >= " + f"{presence_min_frac:.0%} ({rule_choice})" if do_presence_filter else "no presence filter")
                                    + ("; 0/NA imputed" if do_zero_impute else "; no imputation")
                                )
                                try:
                                    if not labels_unique:
                                        st.error("Group labels must be different from each other. Rename the groups in Data Filtering & Group Specification.")

                                    elif is_pairwise_design:
                                        if not can_pairwise or not selected_pair:
                                            st.error("Define at least two non-empty groups and pick a comparison.")
                                        else:
                                            i, j = pairs[selected_pair]
                                            c_i, t_i = (j, i) if ctrl_idx == j else (i, j)
                                            control_cols, test_cols = group_samples[c_i], group_samples[t_i]
                                            control_lab, test_lab = group_labels[c_i], group_labels[t_i]

                                            with st.spinner(f"Running {specific_test}..."):
                                                if specific_test == LIMMA_T:
                                                    diffexp_df = compute_diffexp_limma_lf(
                                                        df=merged_df,
                                                        group_cols=[list(c) for c in group_samples],
                                                        group_labels=list(group_labels),
                                                        control_idx=c_i,
                                                        test_idx=t_i,
                                                        use_all_groups=limma_all_groups,
                                                        **prep,
                                                    )
                                                else:
                                                    diffexp_df = compute_diffexp(
                                                        df=merged_df,
                                                        control_cols=control_cols,
                                                        test_cols=test_cols,
                                                        control_label=control_lab,
                                                        test_label=test_lab,
                                                        test_type=specific_test,
                                                        group_samples_for_presence=[control_cols, test_cols],
                                                        **prep,
                                                    )

                                            method_label = f"{specific_test}: {test_lab} vs {control_lab}"
                                            if specific_test == LIMMA_T and limma_all_groups and len(nonempty) > 2:
                                                method_label += " (variance from all groups)"
                                            ss.update({
                                                "pairwise_control_label": control_lab,
                                                "pairwise_test_label": test_lab,
                                                "pairwise_control_cols": control_cols,
                                                "pairwise_test_cols": test_cols,
                                                "stat_method_label": f"{method_label}; {prep_note}",
                                                "diffexp_df": diffexp_df,
                                                "diff_done": True
                                            })
                                    else:
                                        if not can_anova:
                                            st.error("Provide exactly three non-empty groups for ANOVA.")
                                        else:
                                            with st.spinner(f"Running {multigroup_test}..."):
                                                if multigroup_test == LIMMA_F:
                                                    diffexp_df = compute_diffexp_limma_f_lf(
                                                        df=merged_df,
                                                        group_cols=[list(c) for c in group_samples],
                                                        group_labels=list(group_labels),
                                                        control_idx=ctrl_idx,
                                                        **prep,
                                                    )
                                                else:
                                                    diffexp_df = compute_diffexp_anova(
                                                        df=merged_df,
                                                        group1_cols=group_samples[0],
                                                        group2_cols=group_samples[1],
                                                        group3_cols=group_samples[2],
                                                        group1_label=group_labels[0],
                                                        group2_label=group_labels[1],
                                                        group3_label=group_labels[2],
                                                        run_tukey=False,
                                                        **prep,
                                                    )
                                            ss.update({
                                                "stat_method_label": f"{multigroup_test}: {', '.join(group_labels[:3])}; {prep_note}",
                                                "diffexp_df": diffexp_df,
                                                "diff_done": True
                                            })
                                except Exception as e:
                                    st.error(f"Differential analysis failed: {e}")

                    if ss.get("diff_done") and df_is_nonempty(ss.get("diffexp_df")):
                        _res = ss["diffexp_df"]
                        _q = pd.to_numeric(_res["q-value"], errors="coerce") if "q-value" in _res.columns else pd.Series(dtype=float)
                        _n_sig = int((_q < 0.05).sum())
                        _n_untested = int(_q.isna().sum())
                        st.success(
                            f"**{ss.get('stat_method_label', 'Differential analysis complete')}**: "
                            f"{len(_res):,} glycoforms in the results, {_n_sig:,} with q < 0.05."
                        )
                        if "AveExpr" in _res.columns:
                            if "F" in _res.columns:
                                st.caption(
                                    "limma columns: **AveExpr** = mean log2 intensity; **F** = moderated F statistic; "
                                    "**p-value / q-value** = does any group differ; **log2FC (B vs A)** and "
                                    "**q-value (B vs A)** = each pair of groups (moderated t-test). "
                                    "Group-name columns show the mean raw intensity."
                                )
                            else:
                                st.caption(
                                    "limma columns: **log2FC** = mean log2(test) − mean log2(reference); "
                                    "**AveExpr** = mean log2 intensity; **t** = moderated t statistic. "
                                    "Group-name columns show the mean raw intensity."
                                )
                            if _n_untested:
                                st.caption(
                                    f"{_n_untested:,} glycoforms have no limma p-value because they were not "
                                    "measured in a whole compared group. They are listed at the bottom. Tick "
                                    "*Replace 0/NA* and re-run to test them with imputed values."
                                )
                        show_grid(ss["diffexp_df"], page_size=15, height=500)
                        
                        st.divider()
                        st.markdown("#### Filter & Download Results")
                        with st.expander("Download Full Dataset (Unfiltered)", expanded=False):
                            dl_col1, dl_col2, spacer = st.columns([1.5, 1.5, 7])
                            diff_csv_bytes = ss["diffexp_df"].to_csv(index=False).encode("utf-8")
                            with dl_col1:
                                add_download_button(diff_csv_bytes, "diffexp_results_glycoexplorer_LF.csv", label="Download CSV", key="save_diffexp_lf_csv", mime="text/csv")

                            buffer_diff = BytesIO()
                            with pd.ExcelWriter(buffer_diff, engine="openpyxl") as writer:
                                ss["diffexp_df"].to_excel(writer, index=False, sheet_name="Differential_Results")
                            diff_excel_bytes = buffer_diff.getvalue()
                            with dl_col2:
                                add_download_button(diff_excel_bytes, "diffexp_results_glycoexplorer_LF.xlsx", label="Download Excel", key="save_diffexp_lf_excel", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                        with st.container(border=True):
                            st.markdown("**Filter Significant Features**")
                            diff_df = ss["diffexp_df"]
                            is_anova = "log2FC" not in ss["diffexp_df"].columns
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                sig_metric = st.selectbox(
                                    "Significance Metric", 
                                    ["q-value", "p-value"] if "q-value" in diff_df.columns else ["p-value"], 
                                    key="filter_metric",
                                    help="q-value (false discovery rate) is recommended. Raw p-values give many false hits when thousands of glycoforms are tested."
                                )
                                alpha_thr = st.number_input(
                                    f"{sig_metric} cutoff (<)", 
                                    0.05, 
                                    step=0.01, 
                                    format="%.4f", 
                                    key="filter_alpha"
                                )
                            with col2:
                                if not is_anova:
                                    fc_thr = st.number_input("|log2FC| cutoff (>)", 1.0, step=0.5, format="%.2f", key="filter_fc")
                                else:
                                    fc_thr = 0.0
                                    st.info("Fold-change filtering is not applicable for ANOVA.")

                            if st.button("Apply Filters & View Subset", type="primary", key="btn_apply_filter"):
                                mask = diff_df[sig_metric] < alpha_thr
                                if not is_anova:
                                    mask = mask & (diff_df["log2FC"].abs() > fc_thr)
                                ss.update({"filtered_df": diff_df[mask].copy(), "filtered_ready": True})
                                
                            if ss.get("filtered_ready") and ss.get("filtered_df") is not None:
                                f_df = ss["filtered_df"]
                                st.success(f"Retained **{len(f_df)}** significant features matching the criteria.")
                                if not f_df.empty:
                                    show_grid(f_df, page_size=10, height=300)
                                    f_col1, f_col2, _ = st.columns([1.5, 1.5, 7])
                                    
                                    buf_f_csv, buf_f_xlsx = BytesIO(), BytesIO()
                                    f_df.to_csv(buf_f_csv, index=False)
                                    with pd.ExcelWriter(buf_f_xlsx, engine='openpyxl') as writer:
                                        f_df.to_excel(writer, index=False, sheet_name='Filtered_Results')
                                    with f_col1:
                                        add_download_button(buf_f_csv.getvalue(), "filtered_differential_results_LF.csv", label="Download Filtered CSV", key="save_filtered_lf_csv", mime="text/csv")
                                    with f_col2:
                                        add_download_button(buf_f_xlsx.getvalue(), "filtered_differential_results_LF.xlsx", label="Download Filtered Excel", key="save_filtered_lf_excel", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # ------------------- TAB: Results Visualizations -------------------
        with tabs[1]:
            if df_is_empty(ss.get("diffexp_df")):
                st.warning("Please complete the Differential Analysis first.")
            else:
                diffexp_df = ss.get("diffexp_df", pd.DataFrame())
                is_anova = "log2FC" not in ss.get("diffexp_df", pd.DataFrame()).columns
                
                group_samples = ss.get("group_samples", [])
                group_labels = ss.get("group_labels", [])
                ctrl_idx = ss.get("control_group_idx", 0)
                ctrl_idx = min(max(ctrl_idx, 0), len(group_samples) - 1) if group_samples else 0

                control_group = group_samples[ctrl_idx] if len(group_samples) > ctrl_idx else []
                control_label = group_labels[ctrl_idx] if len(group_labels) > ctrl_idx else "Reference"
                
                other_indices = [i for i in range(len(group_samples)) if i != ctrl_idx]
                test_a_group = group_samples[other_indices[0]] if len(other_indices) > 0 else []
                test_a_label = group_labels[other_indices[0]] if len(other_indices) > 0 else "Test"

                if not is_anova and ss.get("pairwise_control_cols") and ss.get("pairwise_test_cols"):
                    control_group, test_a_group = ss["pairwise_control_cols"], ss["pairwise_test_cols"]
                    control_label = ss.get("pairwise_control_label", control_label)
                    test_a_label = ss.get("pairwise_test_label", test_a_label)

                left, right = st.columns([0.42, 0.58], gap="large")
                for key in ["show_volcano", "show_box", "show_heatmap", "show_box_gene", "show_lollipop", "show_waterfall", "show_class_breakdown"]:
                    ss.setdefault(key, False)

                with left:
                    with st.container(border=True):
                        modes = (
                            ["Box plot", "Heatmap", "Box plot (Gene)", "Lollipop plot", "Glycan class breakdown"]
                            if is_anova
                            else ["Volcano", "Waterfall", "Box plot", "Heatmap", "Box plot (Gene)", "Lollipop plot", "Glycan class breakdown"]
                        )
                        mode = st.radio(
                            "Select Analysis Module", 
                            modes, 
                            index=None, 
                            key="explore_mode", 
                            horizontal=False,
                            label_visibility="collapsed"
                        )
                        
                        flags = {
                            "Volcano": dict(
                                show_volcano=True, show_waterfall=False, show_box=False, show_heatmap=False, 
                                show_box_gene=False, show_lollipop=False
                            ),
                            "Waterfall": dict(
                                show_volcano=False, show_waterfall=True, show_box=False, show_heatmap=False,
                                show_box_gene=False, show_lollipop=False
                            ), 
                            "Box plot": dict(
                                show_volcano=False, show_waterfall=False, show_box=True, show_heatmap=False, 
                                show_box_gene=False, show_lollipop=False
                            ), 
                            "Heatmap": dict(
                                show_volcano=False, show_waterfall=False, show_box=False, show_heatmap=True, 
                                show_box_gene=False, show_lollipop=False
                            ), 
                            "Box plot (Gene)": dict(
                                show_volcano=False, show_waterfall=False, show_box=False, show_heatmap=False, 
                                show_box_gene=True, show_lollipop=False
                            ), 
                            "Lollipop plot": dict(
                                show_volcano=False, show_waterfall=False, show_box=False, show_heatmap=False, 
                                show_box_gene=False, show_lollipop=True
                            ),
                            "Glycan class breakdown": dict(show_class_breakdown=True),
                        }
                        
                        for k in ["show_volcano","show_waterfall","show_box","show_heatmap","show_box_gene","show_lollipop","show_class_breakdown"]:
                            ss[k] = False
                            
                        if mode is not None:
                            ss.update(flags[mode])

                        if ss.get("show_volcano", False):
                            st.markdown("**Volcano plot options**")
                            volcano_plot_type = st.radio(
                                "Volcano plot type",
                                ["Static", "Interactive"],
                                index=0,
                                key="volcano_plot_type",
                                horizontal=True
                            )
                            sig_col = st.radio("Significance metric", ["p-value", "q-value"], key="volcano_sig_col", horizontal=True)
                            sig_threshold = st.slider(f"{sig_col} threshold", 1e-5, 0.1, 0.05, key="volcano_sig_threshold", format="%.5f")
                            log2fc_threshold = st.slider("log2FC threshold", 0.5, 3.0, 1.0, key="volcano_logfc")
                            top_n = st.slider("Top N markers to annotate", 1, 50, 10, key="volcano_topn")
                            top_mode = st.radio("Highlight Top Hits:", ["all", "up", "down", "up-down"], index=3, key="volcano_topmode", horizontal=True)

                        elif ss.get("show_waterfall", False):
                            st.markdown("**Waterfall options**")
                            metric_opts = [c for c in ["q-value", "p-value"] if c in diffexp_df.columns]
                            if not metric_opts:
                                metric_opts = ["p-value"]
                            wfl_sig_col = st.radio("Significance metric", metric_opts, key="lfq_wfl_sig_col", horizontal=True)
                            wfl_alpha_thr = st.slider(f"{wfl_sig_col} threshold", 1e-5, 0.1, 0.05, key="lfq_wfl_alpha", format="%.5f")
                            wfl_fc_thr = st.slider("log2FC threshold", 0.5, 3.0, 1.0, key="lfq_wfl_fc")
                            wfl_top_n = st.slider("Top N up/down features", 3, 20, 5, key="lfq_wfl_topn")
                            
                            st.caption(
                                "Waterfall is rendered full-width below so the structural table has enough room."
                            )

                        elif ss.get("show_box", False):
                            st.markdown("**Box plot options**")
                            sig_col_box = st.radio("Significance metric", ["p-value", "q-value"], key="box_sig_col", horizontal=True)
                            top_n_box = st.slider("Top N markers to display", 1, 50, 10, key="box_topn")
                            
                            # ADDED: Directional highlight options (only for pairwise, not ANOVA)
                            if not is_anova:
                                st.radio(
                                    "Highlight Top Hits:", 
                                    ["all", "up", "down", "up-down"], 
                                    index=3, 
                                    key="box_topmode", 
                                    horizontal=True
                                )
                            
                            st.markdown("**Customize Group Colors**")
                            if is_anova:
                                ss["box_anova_colors"] = {}
                                default_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
                                cols = st.columns(len(group_labels))
                                for i, label in enumerate(group_labels):
                                    with cols[i]:
                                        ss["box_anova_colors"][label] = st.color_picker(
                                            label, 
                                            value=default_palette[i % len(default_palette)], 
                                            key=f"box_anova_color_{i}"
                                        )
                            else:
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.color_picker(f"Color for {control_label}", value="#1f77b4", key="box_color_ctrl")
                                with c2:
                                    st.color_picker(f"Color for {test_a_label}", value="#d62728", key="box_color_test")

                        elif ss.get("show_heatmap", False):
                            st.markdown("**Heatmap options**")
                            sig_col_heatmap = st.radio("Significance metric", ["p-value", "q-value"], key="heatmap_sig_col", horizontal=True)
                            top_n_heatmap = st.slider("Top N markers to display", 1, max(10, len(diffexp_df)), 25, key="heatmap_topn")
                            normalize_heatmap = st.radio("Normalize data", ["Z-score", "Raw Abundance"], key="heatmap_norm", horizontal=True)
                            cluster_option = st.selectbox("Clustering", ["Hierarchical", "K-Means", "None"], key="heatmap_cluster")
                            
                            st.markdown("Customize Group Colors")
                            if is_anova:
                                default_palette = ["#87CEEB", "#FA8072", "#90EE90", "#DDA0DD", "#F0E68C"]
                                ss["heatmap_anova_colors"] = {}
                                cols = st.columns(len(group_labels))
                                for i, label in enumerate(group_labels):
                                    with cols[i]:
                                        ss["heatmap_anova_colors"][label] = st.color_picker(
                                            label, 
                                            value=default_palette[i % len(default_palette)], 
                                            key=f"hm_anova_color_{i}"
                                        )
                            else:
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.color_picker(control_label, value="#87CEEB", key="hm_color_ctrl")
                                with c2:
                                    st.color_picker(test_a_label, value="#FA8072", key="hm_color_test")
                            
                        elif ss.get("show_box_gene", False):
                            st.markdown("**Box plots for selected Gene IDs**")
                            user_input_genes = st.text_area(
                                "Enter Gene IDs to plot (one per line or comma-separated)", 
                                key="box_gene_ids", 
                                placeholder="e.g.\nACTB\nGAPDH, MUC1\nTP53"
                            )
                            user_genes = [g.strip() for g in re.split(r"[,\n;\t ]+", user_input_genes) if g.strip()]
                            
                            ss.update({
                                "box_gene_list": user_genes, 
                                "box_gene_plot": ss.setdefault("box_gene_plot", False)
                            })
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.button(
                                    "Plot selected genes", 
                                    key="btn_plot_genes", 
                                    type="primary", 
                                    on_click=_plot_genes_cb, 
                                    disabled=(len(user_genes) == 0)
                                )
                            with col_b:
                                st.button("Clear selection", key="btn_clear_genes", on_click=_clear_genes_cb)

                        elif ss.get("show_class_breakdown", False):
                            st.markdown("**Glycan class breakdown options**")
                            st.caption(
                                "Which glycan classes or traits change more often than expected? "
                                "Uses the current results table."
                            )
                            st.radio(
                                "Group glycoforms by",
                                ["Glycan class", "Sialylation / fucosylation"],
                                key="gcb_group_by",
                                horizontal=True,
                                help="Glycan class: high-mannose, paucimannose, complex/hybrid. "
                                     "Sialylation / fucosylation: read from the H N A F composition.",
                            )
                            gcb_metric_opts = ["q-value", "p-value"] if "q-value" in diffexp_df.columns else ["p-value"]
                            st.radio("Significance metric", gcb_metric_opts, key="gcb_sig_col", horizontal=True)
                            st.number_input("Cutoff (<)", value=0.05, min_value=0.0001, max_value=1.0,
                                            step=0.01, format="%.4f", key="gcb_alpha")
                            if not is_anova:
                                st.number_input("|log2FC| cutoff (>)", value=1.0, min_value=0.0, step=0.25,
                                                format="%.2f", key="gcb_fc")

                        elif ss.get("show_lollipop", False):
                            st.markdown("**Lollipop plot options**")                            
                            metric_opts = ["q-value", "p-value"] if "q-value" in diffexp_df.columns else ["p-value"]
                            ss["lollipop_sig_col"] = st.radio("Significance metric", metric_opts, key="lollipop_sig_col_ui", horizontal=True)
                            ss["lollipop_show_legend"] = st.checkbox("Distinguish glycans with shapes and legend", value=False, key="lollipop_legend_ui") 
                            
                            if "Genes" in diffexp_df.columns:
                                all_genes_raw = diffexp_df["Genes"].dropna().astype(str).tolist()
                                unique_genes = set()
                                for g_str in all_genes_raw:
                                    for g in re.split(r"[,\s;]+", g_str):
                                        if g.strip():
                                            unique_genes.add(g.strip())
                                unique_genes = sorted(list(unique_genes))
                            else:
                                unique_genes = []

                            selected_gene = st.selectbox(
                                "Select a Gene ID", 
                                options=["-- Select a gene --"] + unique_genes, 
                                key="lollipop_gene_select"
                            )

                            can_plot = False
                            if selected_gene != "-- Select a gene --":
                                ss["lollipop_gene_list"] = [selected_gene]
                                can_plot = True
                            else:
                                ss["lollipop_gene_list"] = []

                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.button(
                                    "Plot gene", 
                                    key="btn_plot_lollipop_gene", 
                                    type="primary",
                                    on_click=_plot_lollipop_cb, 
                                    disabled=not can_plot,
                                    help="Generate a lollipop plot summarizing structural significance for the chosen gene."
                                )
                            with col_b:
                                st.button(
                                    "Clear", 
                                    key="btn_clear_lollipop_gene", 
                                    on_click=_clear_lollipop_cb,
                                    help="Clear the current selection and hide the lollipop plot."
                                )

                with right:
                    if ss.get("show_volcano", False):
                        if is_anova:
                            st.info("Volcano plot is only available for pairwise (t-test) results.")
                        else:
                            if ss.get("volcano_plot_type", "Static") == "Interactive":
                                create_interactive_volcano_plot(
                                    df=diffexp_df,
                                    sig_threshold=ss["volcano_sig_threshold"],
                                    log2fc_threshold=ss["volcano_logfc"],
                                    sig_col=ss["volcano_sig_col"],
                                    log2fc_col="log2FC",
                                    control_label=control_label,
                                    test_label=test_a_label,
                                )
                            else:
                                create_volcano_plot(
                                    df=diffexp_df, 
                                    top_n=ss["volcano_topn"], 
                                    sig_threshold=ss["volcano_sig_threshold"], 
                                    log2fc_threshold=ss["volcano_logfc"], 
                                    label_col="Peptide", 
                                    sig_col=ss["volcano_sig_col"], 
                                    log2fc_col="log2FC", 
                                    control_label=control_label, 
                                    test_label=test_a_label, 
                                    glycan_image_dir=glycan_image_dir, 
                                    top_mode=ss["volcano_topmode"]
                                )

                    elif ss.get("show_waterfall", False):
                        st.info("Waterfall output is shown full-width below this section.")
                            
                    elif ss.get("show_box", False):
                        pval_col = "p-value" if ss["box_sig_col"] == "p-value" else "q-value"
                        if is_anova:
                            create_box_plots_anova(
                                df=diffexp_df, 
                                pval_col=pval_col, 
                                top_n=ss["box_topn"], 
                                group_samples=group_samples, 
                                group_labels=group_labels, 
                                glycan_image_dir=glycan_image_dir,
                                quant_mode="LFQ",
                                group_colors=ss.get("box_anova_colors") 
                            )
                        else:
                            create_box_plots(
                                df=diffexp_df, 
                                pval_col=pval_col, 
                                top_n=ss["box_topn"], 
                                control_cols=control_group, 
                                test_cols=test_a_group, 
                                control_label=control_label, 
                                test_label=test_a_label, 
                                glycan_image_dir=glycan_image_dir, 
                                control_color=ss.get("box_color_ctrl", "#1f77b4"), 
                                test_color=ss.get("box_color_test", "#d62728"),
                                quant_mode="LFQ",
                                top_mode=ss.get("box_topmode", "up-down") # ADDED: pass the mode to the plotting function
                            )
                            
                    elif ss.get("show_heatmap", False):
                        pval_col_heatmap = ss["heatmap_sig_col"]
                        if is_anova:
                            create_heatmap_anova(
                                df=diffexp_df,
                                pval_col=pval_col_heatmap,
                                top_n=ss["heatmap_topn"],
                                group_samples=group_samples,
                                group_labels=group_labels,
                                normalize=ss["heatmap_norm"],
                                cluster_option=ss["heatmap_cluster"],
                                group_colors=ss.get("heatmap_anova_colors"),
                                quant_mode="LFQ"
                            )
                        else:
                            create_heatmap(
                                df=diffexp_df,
                                pval_col=pval_col_heatmap,
                                top_n=ss["heatmap_topn"],
                                control_cols=control_group,
                                test_cols=test_a_group,
                                control_label=control_label,
                                test_label=test_a_label,
                                normalize=ss["heatmap_norm"],
                                cluster_option=ss["heatmap_cluster"],
                                control_color=ss.get("hm_color_ctrl", "#87CEEB"),
                                test_color=ss.get("hm_color_test", "#FA8072"),
                                quant_mode="LFQ"
                            )
                        
                    elif ss.get("show_box_gene", False):
                        if not ss.get("box_gene_plot", False):
                            st.info("Enter Gene IDs on the left and click Plot selected genes.")
                        else:
                            if not ss.get("box_gene_list", []):
                                st.info("Please enter at least one Gene ID.")
                            elif "Genes" not in diffexp_df.columns:
                                st.error("The dataset is missing a 'Genes' column.")
                            else:
                                sub_df = diffexp_df.loc[
                                    diffexp_df["Genes"].astype(str).str.lower().isin(
                                        {g.lower() for g in ss["box_gene_list"]}
                                    )
                                ].copy()
                                
                                if sub_df.empty:
                                    st.warning("No matching Gene IDs in the current dataset.")
                                else:
                                    pval_col = "p-value" if "p-value" in sub_df.columns else ("q-value" if "q-value" in sub_df.columns else None)
                                    if not pval_col:
                                        st.error("Neither 'p-value' nor 'q-value' column found.")
                                    else:
                                        if is_anova:
                                            grp3_cols_extracted = group_samples[2] if len(group_samples) > 2 else []
                                            grp3_label_extracted = group_labels[2] if len(group_labels) > 2 else ""

                                            create_box_plots_top_n_anova(
                                                df=sub_df, 
                                                pval_col=pval_col, 
                                                top_n=len(sub_df), 
                                                group1_cols=control_group, 
                                                group2_cols=test_a_group, 
                                                group3_cols=grp3_cols_extracted, 
                                                group1_label=control_label, 
                                                group2_label=test_a_label, 
                                                group3_label=grp3_label_extracted, 
                                                glycan_image_dir=glycan_image_dir,
                                                quant_mode="LFQ"
                                            )
                                        else:
                                            create_box_plots_top_n(
                                                df=sub_df, 
                                                pval_col=pval_col, 
                                                top_n=len(sub_df), 
                                                control_cols=control_group, 
                                                test_cols=test_a_group, 
                                                control_label=control_label, 
                                                test_label=test_a_label, 
                                                glycan_image_dir=glycan_image_dir,
                                                quant_mode="LFQ"
                                            )
                                            
                    elif ss.get("show_class_breakdown", False):
                        render_glycan_class_breakdown(
                            plot_df=diffexp_df,
                            is_anova=is_anova,
                            sig_col=ss.get("gcb_sig_col", "q-value" if "q-value" in diffexp_df.columns else "p-value"),
                            alpha=float(ss.get("gcb_alpha", 0.05)),
                            fc_thr=float(ss.get("gcb_fc", 1.0)),
                            group_by=ss.get("gcb_group_by", "Glycan class"),
                        )

                    elif ss.get("show_lollipop", False):
                        genes = ss.get("lollipop_gene_list", [])

                        if not ss.get("lollipop_gene_plot", False):
                            st.info("Select a Gene ID from the dropdown on the left and click **Plot gene**.")
                        elif len(genes) != 1 or genes[0] == "-- Select a gene --":
                            st.error("Please select a valid Gene ID.")
                        else:
                            sig_metric = ss.get("lollipop_sig_col", "q-value")
                            required = {"Genes", "Peptide", "ProSites", "Glycan", sig_metric}
                            missing = [c for c in required if c not in diffexp_df.columns]
                            
                            if missing:
                                st.error("Missing required columns for lollipop plot: " + ", ".join(missing))
                            else:
                                df_for_plot = diffexp_df.copy()
                                gene = genes[0]
                                
                                show_legend = ss.get("lollipop_show_legend", False)
                                
                                if show_legend:
                                    create_lollipop_legend(df_for_plot, gene, sig_col=sig_metric)
                                else:
                                    create_lollipop(df_for_plot, gene, sig_col=sig_metric)
                        
                if ss.get("show_waterfall", False):
                    st.markdown("---")
                    if is_anova:
                        st.info("Waterfall plot is only available for pairwise results with log2FC.")
                    else:
                        wfl_sig_col = ss.get("lfq_wfl_sig_col", "q-value" if "q-value" in diffexp_df.columns else "p-value")
                        wfl_alpha_thr = ss.get("lfq_wfl_alpha", 0.05)
                        wfl_fc_thr = ss.get("lfq_wfl_fc", 1.0)
                        wfl_top_n = ss.get("lfq_wfl_topn", 5)

                        ss["waterfall_sig_col"] = wfl_sig_col
                        ss["waterfall_alpha"] = wfl_alpha_thr
                        ss["waterfall_logfc"] = wfl_fc_thr
                        ss["waterfall_topn"] = wfl_top_n
                        ss["report_waterfall_fig"] = _create_lf_waterfall_report_fig(
                            diff_df=diffexp_df,
                            sig_col=wfl_sig_col,
                            alpha_thr=wfl_alpha_thr,
                            fc_thr=wfl_fc_thr,
                            top_n=wfl_top_n,
                        )

                        render_ge_lf_structural_waterfall(
                            diff_df=diffexp_df,
                            glycan_image_dir=glycan_image_dir,
                            sig_col=wfl_sig_col,
                            alpha_thr=wfl_alpha_thr,
                            fc_thr=wfl_fc_thr,
                            top_n=wfl_top_n,
                        )

    # ====================== RESULTS DOWNLOAD (REPORT) ======================
    if ss.get("show_report", False):
        with st.container(border=True):
            col_info, col_action = st.columns([0.65, 0.35], gap="large")
            with col_info:
                st.markdown(
                    "<p class='subtitle-text'>Generate a document summarizing your entire analysis workflow.</p>", 
                    unsafe_allow_html=True
                )
                st.markdown(
                    """
                    - **Experimental Design:** Cohort definitions, sample mapping, reference group, and batch status.
                    - **Data Integrity:** PCA, correlation matrices, sample clustering heatmaps, and glycan summaries.
                    - **Differential Analysis:** Top up/down glycopeptides, gene-level summaries, waterfall top features, and statistical tables.
                    - **Export Bundle:** A ZIP archive containing the PDF, report images, and tabular outputs.
                    """
                )
            with col_action:
                st.write("")
                st.write("")

                def _build_report_zip():
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                        if ss.get("final_pdf_bytes"):
                            zf.writestr("report/final_report.pdf", ss["final_pdf_bytes"])
                        if df_is_nonempty(ss.get("merged_df")):
                            zf.writestr("tables/merged_data.csv", ss["merged_df"].to_csv(index=False))
                        if df_is_nonempty(ss.get("diffexp_df")):
                            zf.writestr("tables/differential_results.csv", ss["diffexp_df"].to_csv(index=False))
                        if df_is_nonempty(ss.get("filtered_df")):
                            zf.writestr("tables/filtered_results.csv", ss["filtered_df"].to_csv(index=False))

                        if ss.get("global_sample_cols") and ss.get("global_group_samples") and ss.get("global_group_labels"):
                            rows = []
                            batch_dict = ss.get("batch_dict", {})
                            
                            for col in ss["global_sample_cols"]:
                                grp_name, grp_num = "Unassigned", None
                                for i, cols in enumerate(ss["global_group_samples"]):
                                    if col in (cols or []):
                                        grp_name = ss["global_group_labels"][i] if i < len(ss["global_group_labels"]) else f"Group{i+1}"
                                        grp_num = i + 1
                                        break
                                        
                                batch_name = "N/A"
                                for b_name, b_cols in batch_dict.items():
                                    if col in b_cols:
                                        batch_name = b_name if ss.get("batch_corrected") else "Unassigned"
                                        break
                                        
                                clean_sample_name = re.sub(r"^\s*Abundance\s*:\s*", "", str(col))
                                if ":" in clean_sample_name:
                                    clean_sample_name = clean_sample_name.split(":", 1)[-1].strip()
                                else:
                                    clean_sample_name = clean_sample_name.strip()
                                        
                                rows.append({
                                    "Column": col, 
                                    "Sample": clean_sample_name, 
                                    "Group": grp_name, 
                                    "Group_Number": grp_num, 
                                    "Batch": batch_name,
                                    "Is_Reference": (grp_num is not None and (grp_num - 1) == ss.get("global_control_group_idx", 0))
                                })
                                
                            if rows:
                                zf.writestr("tables/sample_metadata_mapping.csv", pd.DataFrame(rows).to_csv(index=False))

                        plot_keys = [
                            "report_pca_fig", "report_corr_fig", "report_density_fig", 
                            "report_overview_heatmap_fig", "report_qc_heatmap_fig", 
                            "report_volcano_fig", "report_waterfall_fig", "report_ma_fig", "report_heatmap_fig", 
                            "report_boxplot_fig", "report_lollipop_fig"
                        ]
                        
                        for plot_key in plot_keys:
                            if plot_key in ss and ss[plot_key]:
                                buf = BytesIO()
                                ss[plot_key].savefig(buf, format="png", dpi=300, bbox_inches="tight")
                                clean_name = plot_key.replace('report_', '').replace('_fig', '')
                                zf.writestr(f"plots/{clean_name}.png", buf.getvalue())
                                try:
                                    plt.close(ss[plot_key])
                                except Exception:
                                    pass

                        readme_text = (
                            f"GlycoExplorer Report Bundle (Label-Free)\n"
                            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                            f"Contents:\n"
                            f"- report/final_report.pdf (if generated)\n"
                            f"- tables/*.csv\n"
                            f"- plots/*.png"
                        )
                        zf.writestr("README.txt", readme_text)
                        
                    return zip_buffer.getvalue()

                if "final_pdf_bytes" not in ss:
                    if st.button("Compile PDF Report", use_container_width=True, type="primary"):
                        with st.spinner("Compiling plots and generating PDF..."):
                            try:
                                from ge.report_utility import generate_final_pdf
                                _prepare_lfq_report_figures_for_export()
                                ss["final_pdf_bytes"] = generate_final_pdf()
                                ss["final_report_zip_bytes"] = _build_report_zip()
                                ss["final_report_timestamp"] = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                                st.rerun() 
                            except Exception as e:
                                st.error(f"Failed to generate report: {e}")
                else:
                    st.success("Report generated successfully.")

                    if "final_report_timestamp" not in ss:
                        ss["final_report_timestamp"] = datetime.datetime.now().strftime("%Y%m%d_%H%M")

                    timestamp = ss["final_report_timestamp"]
                    pdf_filename = f"GlycoExplorer_LF_Report_{timestamp}.pdf"
                    zip_filename = f"GlycoExplorer_LF_Report_Bundle_{timestamp}.zip"

                    pdf_bytes = _coerce_to_bytes(ss.get("final_pdf_bytes"))

                    if pdf_bytes is None:
                        st.error("PDF report bytes are missing. Please regenerate the report.")
                    else:
                        st.caption(f"PDF size: {len(pdf_bytes) / 1024 / 1024:.2f} MB")

                    if "final_report_zip_bytes" not in ss or ss.get("final_report_zip_bytes") is None:
                        try:
                            ss["final_report_zip_bytes"] = _build_report_zip()
                        except Exception as e:
                            ss["final_report_zip_bytes"] = None
                            st.warning(f"PDF is ready, but ZIP bundle could not be prepared: {e}")

                    zip_bytes = _coerce_to_bytes(ss.get("final_report_zip_bytes"))

                    if zip_bytes is not None:
                        st.caption(f"ZIP size: {len(zip_bytes) / 1024 / 1024:.2f} MB")

                    dl_col1, dl_col2 = st.columns(2)

                    with dl_col1:
                        if pdf_bytes is not None:
                            add_download_button(pdf_bytes, pdf_filename, label="Download PDF", key="save_lf_pdf_to_downloads", mime="application/pdf", type="primary")
                        else:
                            st.error("PDF report bytes are missing. Please regenerate the report.")

                    with dl_col2:
                        if zip_bytes is not None:
                            add_download_button(zip_bytes, zip_filename, label="Download ZIP", key="save_lf_zip_to_downloads", mime="application/zip")
                        else:
                            st.warning("ZIP bundle is not available. Try regenerating the report.")

                    if st.button("Regenerate Report", width='stretch', key="regenerate_lf_final_report") :
                        ss.pop("final_pdf_bytes", None)
                        ss.pop("final_report_zip_bytes", None)
                        ss.pop("final_report_timestamp", None)
                        st.rerun()

    # ====================== DEFAULT STATE & FAQ ======================
    any_tab_active = (
        ss.get("show_global_select", False) or 
        ss.get("show_overview", False) or 
        ss.get("show_suite", False) or 
        ss.get("show_report", False)
    )

    if not any_tab_active:
        st.info("Choose a section above to begin.")
        
    try:
        faq_section()
    except Exception as e:
        st.error(f"Failed to load FAQ section: {e}")

# Run

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
    if "session_initialized" not in st.session_state:
        _reset_all_data(clear_cache=True)
        st.session_state["session_initialized"] = True
    main()