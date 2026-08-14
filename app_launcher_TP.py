#!/usr/bin/env python3
"""
################################################################################
Total Proteomics Analysis Workflow
################################################################################
"""

__author__ = ("Saurabh Baheti: baheti.saurabh@mayo.edu")

import os
import io
import re
import uuid
import numpy as np
import pandas as pd
import streamlit as st
from download_utility import add_download_button, add_matplotlib_download_buttons, add_plotly_download_buttons
import zipfile       
import datetime      
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib.patches as mpatches
from pathlib import Path

# Optional import for better text label spacing on the static volcano plot
try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False

# Optional import for ComBat Batch Correction
try:
    from neuroCombat import neuroCombat
    HAS_COMBAT = True
except ImportError:
    HAS_COMBAT = False

from io import BytesIO
from PIL import Image
from typing import List, Optional, Tuple
from scipy.stats import ttest_ind, ttest_rel, f_oneway, mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram

# Assuming you have local plotting utilities 
from tp.qc_utility import * # noqa
from tp.report_utility import * # noqa
from tp.ui_utility import * # noqa
from streamlit.errors import StreamlitAPIException

# Shared Total Protein helpers used by both TMT and LFQ launchers.
from tp.tp_common_utility import *

# Safely import FAQ
try:
    from faq.faq_TP import faq_section
except ImportError:
    def faq_section():
        pass

# -----------------------------------------------------------------------------
# Streamlit Setup & CSS Injection
# -----------------------------------------------------------------------------

try_set_page_config(page_title="Total Proteomics | Mayo Clinic", layout="wide")

inject_custom_css()

ss = st.session_state



def _get_downloads_dir() -> Path:
    """Return a safe Downloads folder path for macOS/Windows/Linux."""
    downloads = Path.home() / "Downloads"
    if downloads.exists():
        return downloads
    return Path.home()


def _safe_download_filename(filename: str) -> str:
    """Clean filename and avoid characters that can cause issues on macOS/Windows/Linux."""
    filename = str(filename).strip()
    filename = re.sub(r"[^A-Za-z0-9._-]+", "_", filename)
    filename = filename.strip("_")
    return filename or "downloaded_file"


def _save_bytes_to_downloads(data, filename: str) -> Path:
    """Save bytes directly to the user's Downloads folder without overwriting existing files."""
    outdir = _get_downloads_dir()
    filename = _safe_download_filename(filename)
    outpath = outdir / filename

    if isinstance(data, BytesIO):
        data = data.getvalue()
    if isinstance(data, str):
        data = data.encode("utf-8")
    if not isinstance(data, (bytes, bytearray)):
        data = bytes(data)

    base_outpath = outpath
    counter = 1
    while outpath.exists():
        outpath = base_outpath.with_name(f"{base_outpath.stem}_{counter}{base_outpath.suffix}")
        counter += 1

    outpath.write_bytes(data)
    return outpath


def _save_file_to_downloads(src_path, filename: str = None) -> Path:
    """Copy a local file into Downloads without overwriting existing files."""
    src_path = Path(src_path)
    return _save_bytes_to_downloads(src_path.read_bytes(), filename or src_path.name)



def _reset_all_data(clear_cache: bool = False):
    """Clear all derived data so the suite cannot run on stale state."""
    ss["global_sample_cols"] = []
    ss["global_group_samples"] = []
    ss["global_group_labels"] = []
    ss["global_n_groups"] = 2
    ss["global_control_group_idx"] = 0
    ss["global_all_groups"] = []
    
    ss["batch_dict"] = {}
    ss["batch_config"] = {}
    ss["irs_ref_dict"] = {}
    ss["enable_batch_correction"] = False
    ss["batch_method_ui"] = "None"
    ss["batch_num_selected"] = 2
    ss["batch_correction_requested"] = False
    ss["batch_correction_applied"] = False
    ss["batch_correction_applied_method"] = "None"
    ss["batch_correction_run_signature"] = None
    ss["batch_correction_last_error"] = None

    # Clear temporary Streamlit batch widget state.
    for _k in list(ss.keys()):
        if (
            str(_k).startswith("_tp_tmt_batch_")
            or str(_k).startswith("_tp_tmt_irs_")
            or str(_k).startswith("batch_sel_")
            or str(_k).startswith("irs_ref_")
        ):
            ss.pop(_k, None)

    ss["raw_df"] = None
    ss["expr_raw"] = None
    ss["expr_pre_batch"] = None
    ss["expr_processed"] = None
    ss["diff_results"] = None
    ss["filtered_df"] = None
    ss["filtered_ready"] = False
    
    ss["show_overview"] = False
    ss["show_suite"] = False
    ss["show_global_select"] = False
    ss["show_report"] = False
    ss["upload_sig"] = None 

    ss.pop("final_pdf_bytes", None)
    ss.pop("final_report_zip_bytes", None)
    ss.pop("final_report_timestamp", None)

    if clear_cache:
        try:
            st.cache_data.clear()
        except Exception:
            pass

def init_session_state_defaults():
    defaults = {
        "global_sample_cols": [],
        "global_group_samples": [],
        "global_group_labels": [],
        "global_n_groups": 2,
        "global_control_group_idx": 0,
        "global_all_groups": [],
        "batch_dict": {},
        "batch_config": {},
        "irs_ref_dict": {},
        "enable_batch_correction": False,
        "batch_method_ui": "None",
        "batch_num_selected": 2,
        "batch_correction_requested": False,
        "batch_correction_applied": False,
        "batch_correction_applied_method": "None",
        "batch_correction_run_signature": None,
        "batch_correction_last_error": None,
        "expr_pre_batch": None,
        "show_overview": False,
        "show_suite": False,
        "show_global_select": False,
        "show_report": False,
        "filtered_ready": False
    }
    for k, v in defaults.items():
        ss.setdefault(k, v)


# -----------------------------------------------------------------------------
# Core Utilities & Preprocessing Algorithms
# -----------------------------------------------------------------------------
def df_is_nonempty(df) -> bool:
    return isinstance(df, pd.DataFrame) and not df.empty

def clean_tmt_sample(col: str) -> str:
    if col is None: return col
    s = str(col)
    s = re.sub(r"(?i)abundances?\s*\(grouped\):?", "", s)
    s = re.sub(r"(?i)abundances?:?", "", s)
    s = re.sub(r"(?i)reporter\s*", "", s)
    s = re.sub(r"(?i)intensity\s*", "", s)
    s = s.replace(":", "").strip()
    match = re.search(r"(12[6-9]|13[0-4])[NC]?", s)
    if match: return match.group(0)
    return s.strip()

def quantile_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quantile-normalize a feature x sample matrix without changing its shape.

    Missing values and rows containing only missing values are preserved. The
    implementation uses positional assignment, so duplicate row labels cannot
    cause an index-length mismatch. This avoids the stack()/unstack() failure:
        ValueError: Length mismatch: Expected axis has X elements, new values have Y elements
    """
    if df is None:
        return None
    if df.empty:
        return df.copy()

    x = df.apply(pd.to_numeric, errors="coerce")
    arr = x.to_numpy(dtype=float, copy=True)

    # Normalize only rows having at least one finite value. All-missing rows are
    # restored as NaN in the final matrix.
    valid_rows = np.isfinite(arr).any(axis=1)
    if not valid_rows.any():
        return pd.DataFrame(np.nan, index=df.index, columns=df.columns, dtype=float)

    valid_arr = arr[valid_rows, :]
    sorted_arr = np.sort(valid_arr, axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        rank_means = np.nanmean(sorted_arr, axis=1)

    # Ranks that are missing in every sample can occur near the upper end when
    # columns contain different numbers of observed values. Fill them from the
    # nearest finite rank mean so every observed value receives a value.
    rank_means = (
        pd.Series(rank_means, dtype=float)
        .ffill()
        .bfill()
        .to_numpy(dtype=float)
    )

    normalized_valid = np.full(valid_arr.shape, np.nan, dtype=float)

    for col_idx in range(valid_arr.shape[1]):
        col = valid_arr[:, col_idx]
        observed_positions = np.flatnonzero(np.isfinite(col))
        if observed_positions.size == 0:
            continue

        order = np.argsort(col[observed_positions], kind="mergesort")
        sorted_positions = observed_positions[order]
        normalized_valid[sorted_positions, col_idx] = rank_means[: observed_positions.size]

    normalized_full = np.full(arr.shape, np.nan, dtype=float)
    normalized_full[valid_rows, :] = normalized_valid

    out = pd.DataFrame(normalized_full, index=df.index, columns=df.columns)
    if out.shape != df.shape:
        raise RuntimeError(
            f"Quantile normalization changed matrix shape from {df.shape} to {out.shape}."
        )
    return out

def apply_batch_correction(df: pd.DataFrame, batch_dict: dict, method: str, is_log2: bool, irs_refs: dict = None) -> pd.DataFrame:
    """
    Applies the selected statistical batch correction algorithm.
    Handles mathematical differences depending on whether df is log2 transformed.
    """
    out = df.copy()
    ss["batch_correction_last_error"] = None
    
    if method == "Internal Reference Scaling (IRS)":
        if not irs_refs:
            return out
        ref_cols = [irs_refs[b] for b in batch_dict.keys() if b in irs_refs and irs_refs[b] in out.columns]
        if not ref_cols:
            return out
            
        global_ref_mean = out[ref_cols].mean(axis=1)

        for b_name, cols in batch_dict.items():
            valid = [c for c in cols if c in out.columns]
            if not valid: continue
                
            b_ref_col = irs_refs.get(b_name)
            if not b_ref_col or b_ref_col not in out.columns: continue
                
            b_ref_vals = out[b_ref_col]
            
            if is_log2:
                # Additive shift for log2 data
                shift = global_ref_mean - b_ref_vals
                for col in valid:
                    out[col] = out[col] + shift
            else:
                # Multiplicative ratio for raw data
                scaling_factor = global_ref_mean / (b_ref_vals + 1e-6)
                for col in valid:
                    out[col] = out[col] * scaling_factor
                    
    elif method == "Median Centering":
        global_med = out.median(axis=1)
        for b_name, cols in batch_dict.items():
            valid = [c for c in cols if c in out.columns]
            if valid:
                b_med = out[valid].median(axis=1)
                if is_log2:
                    out[valid] = out[valid].sub(b_med, axis=0).add(global_med, axis=0)
                else:
                    out[valid] = out[valid].div(b_med + 1e-6, axis=0).mul(global_med, axis=0)
                
    elif method == "Mean Centering":
        global_mean = out.mean(axis=1)
        for b_name, cols in batch_dict.items():
            valid = [c for c in cols if c in out.columns]
            if valid:
                b_mean = out[valid].mean(axis=1)
                if is_log2:
                    out[valid] = out[valid].sub(b_mean, axis=0).add(global_mean, axis=0)
                else:
                    out[valid] = out[valid].div(b_mean + 1e-6, axis=0).mul(global_mean, axis=0)
                
    elif method == "Standardize (Mean + Variance)":
        global_mean = out.mean(axis=1)
        global_std = out.std(axis=1).replace(0, 1e-9) 
        
        for b_name, cols in batch_dict.items():
            valid = [c for c in cols if c in out.columns]
            if len(valid) > 1:
                b_mean = out[valid].mean(axis=1)
                b_std = out[valid].std(axis=1).replace(0, 1e-9)
                out[valid] = out[valid].sub(b_mean, axis=0).div(b_std, axis=0).mul(global_std, axis=0).add(global_mean, axis=0)
            elif len(valid) == 1:
                b_mean = out[valid].mean(axis=1)
                out[valid] = out[valid].sub(b_mean, axis=0).add(global_mean, axis=0)
                
    elif method == "ComBat (Empirical Bayes)":
        if not HAS_COMBAT:
            msg = (
                "The 'neuroCombat' package is not installed. "
                "Please run `pip install neuroCombat` to use ComBat."
            )
            ss["batch_correction_last_error"] = msg
            st.error(msg)
            return out
            
        sample_to_batch = {}
        for b_name, cols in batch_dict.items():
            for c in cols:
                if c in out.columns:
                    sample_to_batch[c] = b_name
                    
        valid_cols = list(sample_to_batch.keys())
        if len(valid_cols) < 2 or len(batch_dict) < 2:
            return out
            
        dat = out[valid_cols].copy()
        
        if not is_log2:
            # ComBat assumes normal-ish distributions, usually log2. 
            # We temporarily log2 transform it, apply combat, and exponentiate back.
            dat = np.log2(dat.replace(0, np.nan))
        
        na_mask = dat.isna()
        row_medians = dat.median(axis=1)
        dat_imputed = dat.apply(lambda col: col.fillna(row_medians)).fillna(0)
        
        covars = pd.DataFrame({"batch": [sample_to_batch[c] for c in valid_cols]}, index=valid_cols)
        
        try:
            combat_res = neuroCombat(dat=dat_imputed, covars=covars, batch_col="batch")
            corrected_df = pd.DataFrame(combat_res["data"], index=dat.index, columns=valid_cols)
            
            corrected_df[na_mask] = np.nan
            
            if not is_log2:
                corrected_df = np.exp2(corrected_df)
                
            out[valid_cols] = corrected_df
            
        except Exception as e:
            msg = (
                f"ComBat correction failed: {e}. "
                "Check if your batches have too few samples or zero-variance rows."
            )
            ss["batch_correction_last_error"] = msg
            st.error(msg)

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

def replace_zeros_rowwise_min_fraction(
    df: pd.DataFrame,
    sample_cols: list[str],
    frac: float = 0.2,          
    replace_nan: bool = True
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

    return out


# -----------------------------------------------------------------------------
# Differential Engine 
# -----------------------------------------------------------------------------
def compute_diffexp_tp(
    df: pd.DataFrame,
    control_cols: list[str],
    test_cols: list[str],
    control_label: str = "Control",
    test_label: str = "Test",
    test_type: str = "Student's t-test (Independent, Equal Variance)",
    min_n_per_group: int = 2
) -> pd.DataFrame:

    df = df.copy()
    ctrl = [c for c in control_cols if c in df.columns]
    test = [c for c in test_cols if c in df.columns]
    all_cols = list(dict.fromkeys(ctrl + test))
    
    if not ctrl or not test:
        raise ValueError("Need at least 1 valid sample column in both control and test groups.")

    df[control_label] = df[ctrl].mean(axis=1)
    df[test_label] = df[test].mean(axis=1)
    
    df["log2FC"] = df[test_label] - df[control_label]

    pvals = []
    is_paired = "Paired" in test_type

    for i in range(len(df)):
        row = df.iloc[i]
        v1_raw = row[ctrl].astype(float).values
        v2_raw = row[test].astype(float).values

        if is_paired:
            if len(v1_raw) == len(v2_raw):
                mask = ~np.isnan(v1_raw) & ~np.isnan(v2_raw)
                v1 = v1_raw[mask]
                v2 = v2_raw[mask]
            else:
                v1, v2 = [], [] 
        else:
            v1 = v1_raw[~np.isnan(v1_raw)]
            v2 = v2_raw[~np.isnan(v2_raw)]

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
    df["q-value"] = multipletests(np.nan_to_num(df["p-value"], nan=1.0), method="fdr_bh")[1]

    df["Regulation"] = "Not Significant"
    df.loc[(df["q-value"] < 0.05) & (df["log2FC"] > 1), "Regulation"] = "Upregulated"
    df.loc[(df["q-value"] < 0.05) & (df["log2FC"] < -1), "Regulation"] = "Downregulated"

    regulation_order = {"Upregulated": 0, "Downregulated": 1, "Not Significant": 2}
    df["__reg_sort__"] = df["Regulation"].map(regulation_order).fillna(3)

    metadata_cols = ["RowID", "GeneID", "Genes", "Description", "Accession", "Master Protein Accessions"]
    computed_cols = [control_label, test_label, "log2FC", "p-value", "q-value", "Regulation"]

    keep_cols = [c for c in metadata_cols if c in df.columns] + computed_cols + all_cols

    out = (
        df.sort_values(by=["__reg_sort__", "q-value"], ascending=[True, True], na_position="last")
          .drop(columns=["__reg_sort__"], errors="ignore")
    )
    return out[keep_cols]

def compute_diffexp_anova_tp(
    df: pd.DataFrame,
    group1_cols: list[str], group2_cols: list[str], group3_cols: list[str],
    group1_label: str = "Group1", group2_label: str = "Group2", group3_label: str = "Group3",
) -> pd.DataFrame:

    df = df.copy()
    g1 = [c for c in group1_cols if c in df.columns]
    g2 = [c for c in group2_cols if c in df.columns]
    g3 = [c for c in group3_cols if c in df.columns]
    all_cols = list(dict.fromkeys(g1 + g2 + g3))

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
            continue
        try:
            pvals.append(float(f_oneway(v1, v2, v3).pvalue))
        except Exception:
            pvals.append(np.nan)

    df["p-value"] = pvals
    df["q-value"] = multipletests(df["p-value"].fillna(1.0), method="fdr_bh")[1]
    df["Regulation"] = np.where(df["q-value"] < 0.05, "Significant", "Not Significant")

    metadata_cols = ["RowID", "GeneID", "Genes", "Description", "Accession", "Master Protein Accessions"]
    computed_cols = [group1_label, group2_label, group3_label, "p-value", "q-value", "Regulation"]
    
    keep_cols = [c for c in metadata_cols if c in df.columns] + computed_cols + all_cols

    return df.sort_values("q-value", na_position="last")[keep_cols]



def create_static_volcano_plot_tp(
    res: pd.DataFrame,
    gene_col_name: Optional[str],
    metric: str,
    alpha_thr: float,
    fc_thresh: float,
    top_n_annot: int = 0,
    control_label: str = "Control",
    test_label: str = "Test",
):
    """Create a static, downloadable volcano plot for total proteomics with optional top-protein annotations."""
    plot_df = res.copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df["log2FC"] = pd.to_numeric(plot_df["log2FC"], errors="coerce")
    plot_df = plot_df.dropna(subset=["log2FC", metric]).copy()

    plot_df["-log10(metric)"] = -np.log10(plot_df[metric].clip(lower=1e-300))

    mask_up = (plot_df[metric] <= alpha_thr) & (plot_df["log2FC"] >= fc_thresh)
    mask_dn = (plot_df[metric] <= alpha_thr) & (plot_df["log2FC"] <= -fc_thresh)
    mask_sig = mask_up | mask_dn
    mask_ns = ~mask_sig

    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    ax.scatter(
        plot_df.loc[mask_ns, "log2FC"],
        plot_df.loc[mask_ns, "-log10(metric)"],
        color="#bdbdbd",
        alpha=0.55,
        s=18,
        label="Not Significant",
    )

    ax.scatter(
        plot_df.loc[mask_up, "log2FC"],
        plot_df.loc[mask_up, "-log10(metric)"],
        color="#d62728",
        alpha=0.85,
        s=24,
        label="Up regulated",
    )

    ax.scatter(
        plot_df.loc[mask_dn, "log2FC"],
        plot_df.loc[mask_dn, "-log10(metric)"],
        color="#1f77b4",
        alpha=0.85,
        s=24,
        label="Down regulated",
    )

    ax.axhline(-np.log10(alpha_thr), color="black", linestyle="--", linewidth=1)
    #ax.axvline(fc_thresh, color="black", linestyle="--", linewidth=1)
    #ax.axvline(-fc_thresh, color="black", linestyle="--", linewidth=1)

    # Annotate top significant proteins in the downloadable static figure.
    if top_n_annot and top_n_annot > 0 and gene_col_name and gene_col_name in plot_df.columns:
        annot_df = plot_df.loc[mask_sig].copy()

        if not annot_df.empty:
            annot_df["abs_log2FC"] = annot_df["log2FC"].abs()
            annot_df = (
                annot_df.sort_values([metric, "abs_log2FC"], ascending=[True, False])
                .head(int(top_n_annot))
            )

            texts = []
            x_span = max(
                float(plot_df["log2FC"].max() - plot_df["log2FC"].min()),
                1.0
            )
            y_span = max(
                float(plot_df["-log10(metric)"].max() - plot_df["-log10(metric)"].min()),
                1.0
            )

            for _, row in annot_df.iterrows():
                label = str(row.get(gene_col_name, "")).strip()

                if not label or label.lower() == "nan":
                    continue

                x = float(row["log2FC"])
                y = float(row["-log10(metric)"])

                dx = 0.015 * x_span if x >= 0 else -0.015 * x_span
                dy = 0.015 * y_span

                txt = ax.text(
                    x + dx,
                    y + dy,
                    label,
                    fontsize=8,
                    fontweight="bold",
                    color="black",
                    ha="left" if x >= 0 else "right",
                    va="bottom",
                )

                texts.append(txt)

                ax.plot(
                    [x, x + dx],
                    [y, y + dy],
                    color="gray",
                    linewidth=0.5,
                    alpha=0.7,
                )

            if HAS_ADJUST_TEXT and texts:
                adjust_text(
                    texts,
                    ax=ax,
                    arrowprops=dict(
                        arrowstyle="-",
                        color="gray",
                        lw=0.5,
                        alpha=0.7
                    ),
                )

   # ax.set_title(
   #     f"Volcano Plot ({metric} < {alpha_thr}, |log2FC| > {fc_thresh})",
   #     fontweight="bold",
   # )

    ax.set_xlabel(
        fr"log$_2$ fold-change ({test_label}/{control_label})",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_ylabel(
        fr"$-$log$_{{10}}$ ({metric})",
        fontsize=12,
        fontweight="bold"
    )

    ax.grid(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()

    return fig



def _figure_to_bytes_tp(fig, fmt: str, dpi: int = 300) -> bytes:
    """
    Convert a matplotlib figure to bytes for direct save/export.
    """
    buf = BytesIO()
    if fmt.lower() == "pdf":
        fig.savefig(buf, format="pdf", bbox_inches="tight")
    elif fmt.lower() == "svg":
        fig.savefig(buf, format="svg", bbox_inches="tight")
    else:
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return buf.getvalue()


def add_figure_download_buttons_tp(fig, base_filename: str, key_suffix: str = "", include_pdf: bool = True):
    """Native Streamlit download buttons for Matplotlib figures with success messages."""
    if fig is None:
        return

    if not key_suffix:
        key_suffix = str(uuid.uuid4())[:8]

    base_filename = safe_filename(base_filename) 
    
    png_bytes = _figure_to_bytes_tp(fig, "png")
    svg_bytes = _figure_to_bytes_tp(fig, "svg")
    pdf_bytes = _figure_to_bytes_tp(fig, "pdf") if include_pdf else None

    cols = st.columns(3 if include_pdf else 2)

    with cols[0]:
        png_clicked = st.download_button(
            label="Download PNG",
            data=png_bytes,
            file_name=f"{base_filename}.png",
            mime="image/png",
            key=f"save_png_{base_filename}_{key_suffix}",
            use_container_width=True
        )
        if png_clicked:
            st.success(f"Downloaded {base_filename}.png")

    with cols[1]:
        svg_clicked = st.download_button(
            label="Download SVG",
            data=svg_bytes,
            file_name=f"{base_filename}.svg",
            mime="image/svg+xml",
            key=f"save_svg_{base_filename}_{key_suffix}",
            use_container_width=True
        )
        if svg_clicked:
            st.success(f"Downloaded {base_filename}.svg")

    if include_pdf:
        with cols[2]:
            pdf_clicked = st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name=f"{base_filename}.pdf",
                mime="application/pdf",
                key=f"save_pdf_{base_filename}_{key_suffix}",
                use_container_width=True
            )
            if pdf_clicked:
                st.success(f"Downloaded {base_filename}.pdf")

def add_plotly_html_download_button_tp(fig, base_filename: str, key_suffix: str = ""):
    """Save an interactive Plotly figure via Streamlit native download with success message."""
    if fig is None:
        return
    if not key_suffix:
        key_suffix = str(uuid.uuid4())[:8]

    base_filename = safe_filename(base_filename)

    try:
        html_bytes = fig.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8")
    except Exception as e:
        st.warning(f"Could not prepare interactive HTML export: {e}")
        return

    html_clicked = st.download_button(
        label="Download Interactive HTML",
        data=html_bytes,
        file_name=f"{base_filename}.html",
        mime="text/html",
        key=f"save_html_{base_filename}_{key_suffix}",
        use_container_width=True
    )
    if html_clicked:
        st.success(f"Downloaded {base_filename}.html")

def add_bytes_download_button_tp(
    label: str,
    data,
    filename: str,
    key: str,
    help_text: str = "Download file.",
    type: str = "secondary",
):
    """Render a browser-native download button for CSV/XLSX/PDF/ZIP/TXT bytes."""
    try:
        safe_name = _safe_download_filename(filename)
        file_bytes = _coerce_to_bytes_for_downloads(data)
    except Exception as e:
        st.error(f"Could not prepare {filename} for download: {e}")
        return False

    clicked = st.download_button(
        label=label,
        data=file_bytes,
        file_name=safe_name,
        mime=_guess_mime_from_filename(safe_name),
        key=key,
        help=help_text,
        type=type,
        use_container_width=True,
    )
    if clicked:
        st.success(f"Downloaded {safe_name}")
    return clicked

def create_static_ma_plot_tp(
    res: pd.DataFrame,
    gene_col_name: Optional[str],
    metric: str,
    alpha_thr: float,
    fc_thresh: float,
    mean_cols: list,
    top_n_annot: int = 0,
    control_label: str = "Group2",
    test_label: str = "Group1",
):
    """Create a static MA plot with optional top significant labels."""
    plot_df = res.copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df["log2FC"] = pd.to_numeric(plot_df["log2FC"], errors="coerce")

    for c in mean_cols:
        plot_df[c] = pd.to_numeric(plot_df[c], errors="coerce")

    plot_df["Average_Intensity"] = plot_df[mean_cols].mean(axis=1)
    plot_df = plot_df.dropna(subset=["Average_Intensity", "log2FC", metric]).copy()

    sig_mask = (plot_df[metric] <= alpha_thr) & (plot_df["log2FC"].abs() >= fc_thresh)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        plot_df.loc[~sig_mask, "Average_Intensity"],
        plot_df.loc[~sig_mask, "log2FC"],
        alpha=0.35,
        color="#a9a9a9",
        s=20,
        label="Not Significant",
    )

    ax.scatter(
        plot_df.loc[sig_mask, "Average_Intensity"],
        plot_df.loc[sig_mask, "log2FC"],
        alpha=0.85,
        color="#d62728",
        s=24,
        label="Significant",
    )

    ax.axhline(0, color="black", lw=1)
    ax.axhline(fc_thresh, color="gray", linestyle="--", linewidth=0.9)
    ax.axhline(-fc_thresh, color="gray", linestyle="--", linewidth=0.9)

    if top_n_annot > 0 and gene_col_name and gene_col_name in plot_df.columns:
        annot_df = plot_df.loc[sig_mask].sort_values(metric).head(int(top_n_annot))
        texts = []

        for _, row in annot_df.iterrows():
            label = str(row.get(gene_col_name, "")).strip()
            if not label or label.lower() == "nan":
                continue

            txt = ax.text(
                row["Average_Intensity"],
                row["log2FC"],
                label,
                fontsize=8,
                fontweight="bold",
                color="black",
            )
            texts.append(txt)

        if HAS_ADJUST_TEXT and texts:
            adjust_text(
                texts,
                ax=ax,
                arrowprops=dict(arrowstyle="-", color="black", lw=0.5, alpha=0.6),
            )

    # ax.set_title("MA Plot", fontweight="bold")

    ax.set_xlabel(
        r"Average Intensity (log$_2$)",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_ylabel(
        fr"log$_2$ fold-change ({test_label}/{control_label})",
        fontsize=12,
        fontweight="bold"
    )

    ax.grid(False)
    ax.legend(frameon=False, fontsize=8, loc="best")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    return fig, plot_df



def _p_to_star_label(pval: float) -> str:
    """Convert p/q-value to star label."""
    try:
        pval = float(pval)
    except Exception:
        return "ns"

    if not np.isfinite(pval):
        return "ns"
    if pval <= 0.001:
        return "***"
    if pval <= 0.01:
        return "**"
    if pval <= 0.05:
        return "*"
    return "ns"


def _add_manual_sig_bar(ax, x1: int, x2: int, y: float, h: float, label: str):
    """Add a manual significance bracket and label."""
    ax.plot(
        [x1, x1, x2, x2],
        [y, y + h, y + h, y],
        lw=1.1,
        color="black",
        clip_on=False,
    )

    ax.text(
        (x1 + x2) / 2,
        y + h,
        label,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="black",
        clip_on=False,
    )


def create_protein_profile_figure_tp(
    res: pd.DataFrame,
    expr: pd.DataFrame,
    gene_col_name: str,
    gene_name: str,
    group_labels: list,
    group_samples: list,
    pretty_labels: dict,
    palette: dict = None,
    annotate_stats: bool = True,
    sig_metric: Optional[str] = None,
):
    """Create one protein profile box/strip plot for a selected protein."""
    gene_rows = res.loc[res[gene_col_name].astype(str) == str(gene_name), "RowID"].dropna().tolist()
    if not gene_rows:
        return None

    valid_rows = [r for r in gene_rows if r in expr.index]
    if not valid_rows:
        return None

    plot_data = expr.loc[valid_rows].copy()
    plot_data = plot_data.reset_index().melt(
        id_vars="RowID",
        var_name="Sample",
        value_name="Abundance"
    )

    mapping_dict = {}
    for grp_label, grp_cols in zip(group_labels, group_samples):
        for c in grp_cols:
            mapping_dict[pretty_labels.get(c, c)] = grp_label

    plot_data["Group"] = plot_data["Sample"].map(mapping_dict)
    plot_data = plot_data.dropna(subset=["Group"])
    if plot_data.empty:
        return None

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.grid(False)

    unique_groups = [g for g in group_labels if g in plot_data["Group"].values]
    fallback_palette = dict(
        zip(
            unique_groups,
            sns.color_palette("Set2", len(unique_groups)).as_hex()
        )
    )
    palette = palette or fallback_palette

    sns.boxplot(
        data=plot_data,
        x="Group",
        y="Abundance",
        palette=palette,
        ax=ax,
        width=0.5,
        fliersize=0,
        order=unique_groups,
    )

    sns.stripplot(
        data=plot_data,
        x="Group",
        y="Abundance",
        color="black",
        dodge=False,
        size=4,
        alpha=0.5,
        ax=ax,
        jitter=0.08,
        order=unique_groups,
    )

    # Manual star annotation using the existing differential result.
    # This keeps top-protein profiles consistent with the selected p/q-value filter
    # and avoids missing/clipped statannotations in multi-panel grids and PDFs.
    if annotate_stats and len(unique_groups) >= 2:
        try:
            metric_for_star = sig_metric
            if metric_for_star is None:
                if "q-value" in res.columns:
                    metric_for_star = "q-value"
                elif "p-value" in res.columns:
                    metric_for_star = "p-value"

            if metric_for_star and metric_for_star in res.columns:
                gene_metric_val = (
                    res.loc[res[gene_col_name].astype(str) == str(gene_name), metric_for_star]
                    .apply(pd.to_numeric, errors="coerce")
                    .min()
                )

                star_label = _p_to_star_label(gene_metric_val)

                ymin, ymax = ax.get_ylim()
                y_range = ymax - ymin if ymax > ymin else 1.0
                y = ymax + 0.08 * y_range
                h = 0.04 * y_range

                x1 = 0
                x2 = len(unique_groups) - 1

                _add_manual_sig_bar(
                    ax=ax,
                    x1=x1,
                    x2=x2,
                    y=y,
                    h=h,
                    label=star_label
                )

                ax.set_ylim(ymin, ymax + 0.25 * y_range)

        except Exception as e:
            st.warning(f"Could not add significance stars for {gene_name}: {e}")

    ax.set_title(
        f"{gene_name}",
        pad=14,
        fontweight="bold",
        fontsize=13
    )
    ax.set_ylabel("Normalized Abundance", fontweight="bold", fontsize=11)
    ax.set_xlabel("")
    ax.set_xticklabels(
        unique_groups,
        fontsize=9,
        fontweight="bold",
        rotation=35,
        ha="right"
    )

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", labelsize=9)

    if len(unique_groups) >= 2:
        fig.subplots_adjust(bottom=0.26, top=0.82)
        legend_text = "Significance: ns p/q > 0.05, * ≤ 0.05, ** ≤ 0.01, *** ≤ 0.001"
        fig.text(
            0.5,
            0.02,
            legend_text,
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
            color="black"
        )
    else:
        fig.tight_layout()

    return fig


def create_profile_grid_figure_tp(
    figures: list,
    gene_names: list,
    n_cols: int = 4,
    panel_width: float = 4.2,
    panel_height: float = 3.8,
    dpi: int = 220,
):
    """
    Combine individual protein profile figures into one matplotlib grid figure.

    This is for on-screen display so Streamlit zoom opens the whole grid,
    not one individual protein at a time.
    """
    from PIL import Image
    from io import BytesIO
    import math

    valid_items = [
        (fig, gene)
        for fig, gene in zip(figures, gene_names)
        if fig is not None
    ]

    if not valid_items:
        return None

    n_items = len(valid_items)
    n_cols = max(1, int(n_cols))
    n_rows = math.ceil(n_items / n_cols)

    grid_fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * panel_width, n_rows * panel_height)
    )

    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, (gene_fig, gene_name) in zip(axes, valid_items):
        gene_fig.canvas.draw()

        buf = BytesIO()
        gene_fig.savefig(
            buf,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.20
        )
        buf.seek(0)

        img = Image.open(buf)
        ax.imshow(img)
        ax.axis("off")

        # Do not add a second title here; the individual plot already has the protein title.

    grid_fig.tight_layout(pad=0.25)
    return grid_fig


def build_profile_pdf_bytes_tp(
    figures: list,
    gene_names: list,
    n_cols: int = 5,
    n_rows: int = 2,
) -> bytes:
    """Build a multi-page protein-profile PDF entirely in memory."""
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image
    from io import BytesIO

    valid_items = [
        (fig, gene)
        for fig, gene in zip(figures, gene_names)
        if fig is not None
    ]
    if not valid_items:
        raise ValueError("No protein profile figures were available for PDF export.")

    plots_per_page = n_cols * n_rows
    pdf_buffer = BytesIO()

    with PdfPages(pdf_buffer) as pdf:
        for page_start in range(0, len(valid_items), plots_per_page):
            page_items = valid_items[page_start:page_start + plots_per_page]
            page_fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(n_cols * 4.4, n_rows * 4.2),
            )
            axes = np.array(axes).reshape(-1)

            for ax in axes:
                ax.axis("off")

            for ax, (gene_fig, _gene_name) in zip(axes, page_items):
                gene_fig.canvas.draw()
                image_buffer = BytesIO()
                gene_fig.savefig(
                    image_buffer,
                    format="png",
                    dpi=260,
                    bbox_inches="tight",
                    pad_inches=0.25,
                )
                image_buffer.seek(0)
                with Image.open(image_buffer) as img:
                    ax.imshow(img.copy())
                ax.axis("off")

            page_fig.tight_layout(pad=0.4)
            pdf.savefig(page_fig, bbox_inches="tight")
            plt.close(page_fig)

    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


def save_profile_pdf_to_downloads_tp(
    figures: list,
    gene_names: list,
    filename: str = "protein_profiles.pdf",
    n_cols: int = 5,
    n_rows: int = 2,
):
    """Backward-compatible local save wrapper around the in-memory PDF builder."""
    pdf_bytes = build_profile_pdf_bytes_tp(
        figures=figures,
        gene_names=gene_names,
        n_cols=n_cols,
        n_rows=n_rows,
    )
    return _save_bytes_to_downloads(pdf_bytes, filename)

def create_total_protein_waterfall_plot_tp(
    res: pd.DataFrame,
    gene_col_name: Optional[str],
    accession_col_name: Optional[str],
    metric: str,
    alpha_thr: float,
    fc_thresh: float,
    top_n: int = 10,
):
    """Create ranked total-protein waterfall plot with numbered significant labels.

    The returned label_df preserves protein and accession columns so the numbered
    points can be mapped back to proteins in the table/download.
    """
    wfl = res.dropna(subset=["log2FC", metric]).copy()
    wfl["log2FC"] = pd.to_numeric(wfl["log2FC"], errors="coerce")
    wfl[metric] = pd.to_numeric(wfl[metric], errors="coerce")
    wfl = wfl.dropna(subset=["log2FC", metric]).sort_values("log2FC", ascending=False).reset_index(drop=True)
    wfl["Rank"] = wfl.index + 1
    wfl["Status"] = "Not Significant"
    wfl.loc[(wfl["log2FC"] >= fc_thresh) & (wfl[metric] <= alpha_thr), "Status"] = "Up regulated"
    wfl.loc[(wfl["log2FC"] <= -fc_thresh) & (wfl[metric] <= alpha_thr), "Status"] = "Down regulated"
    top_up = wfl[wfl["Status"] == "Up regulated"].head(top_n)
    top_down = wfl[wfl["Status"] == "Down regulated"].sort_values("log2FC", ascending=True).head(top_n)
    label_df = pd.concat([top_up, top_down], ignore_index=True)
    wfl["Point_ID"] = np.nan
    if not label_df.empty:
        label_df = label_df.copy()
        label_df["Point_ID"] = range(1, len(label_df) + 1)
        wfl["Point_ID"] = wfl["Rank"].map(label_df.set_index("Rank")["Point_ID"].to_dict())
    fig, ax = plt.subplots(figsize=(9, 6.5))
    color_map = {"Up regulated": "#d62728", "Down regulated": "#1f77b4", "Not Significant": "#bdbdbd"}
    for status, sub in wfl.groupby("Status"):
        ax.scatter(sub["Rank"], sub["log2FC"], s=18, alpha=0.85, color=color_map.get(status, "#999999"), label=status)
    if not label_df.empty:
        x_span = max(float(wfl["Rank"].max() - wfl["Rank"].min()), 1.0)
        x_offset = x_span * 0.025
        y_span = max(float(wfl["log2FC"].max() - wfl["log2FC"].min()), 1.0)
        y_offset = y_span * 0.015
        for _, row in label_df.iterrows():
            x0 = row["Rank"]
            y0 = row["log2FC"]
            x1 = x0 + x_offset
            y1 = y0 + (y_offset if y0 >= 0 else -y_offset)
            ax.plot([x0, x1], [y0, y1], color="gray", linewidth=0.6, alpha=0.8)
            ax.text(x1, y1, str(int(row["Point_ID"])), fontsize=8, ha="left", va="center", fontweight="bold", color="black")
    ax.axhline(0, color="black", linewidth=1.0)
    ax.axhline(fc_thresh, color="gray", linestyle="--", linewidth=0.9)
    ax.axhline(-fc_thresh, color="gray", linestyle="--", linewidth=0.9)
    #ax.set_title(f"Total Protein Waterfall ({metric} < {alpha_thr}, |log2FC| > {fc_thresh})", fontweight="bold")
    ax.set_xlabel("Protein", fontweight="bold")
    ax.set_ylabel("log2 Fold-Change", fontweight="bold")
    ax.grid(False)
    ax.legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig, wfl, label_df



def render_professional_tp_start_panel(upload_disabled: bool = True):
    """
    Professional first-run landing panel for the Total Proteomics app.
    Mirrors the GlycoExplorer start experience without informal icons.
    """
    button_text = "Upload data to enable processing" if upload_disabled else "Process Data Matrix"

    st.markdown(
        """
        <style>
        .tp-start-wrap {
            margin-top: 1.4rem;
            margin-bottom: 1.5rem;
        }
        .tp-start-title {
            font-size: 1.85rem;
            line-height: 1.15;
            font-weight: 700;
            color: #005a9c;
            margin-bottom: 0.35rem;
            letter-spacing: -0.01em;
        }
        .tp-start-subtitle {
            font-size: 1.02rem;
            color: #2f3a45;
            margin-bottom: 1.6rem;
            max-width: 980px;
            line-height: 1.55;
        }
        .tp-step-card {
            border: 1px solid #d9e2ef;
            border-radius: 10px;
            padding: 22px 24px;
            background: #f8fbff;
            min-height: 138px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }
        .tp-step-number {
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
            vertical-align: middle;
        }
        .tp-step-heading {
            font-size: 1.15rem;
            font-weight: 700;
            color: #005a9c;
            vertical-align: middle;
        }
        .tp-step-body {
            color: #2f3a45;
            margin-left: 58px;
            margin-top: 8px;
            font-size: 0.96rem;
            line-height: 1.5;
        }
        .tp-step-arrow {
            text-align: center;
            font-size: 2.2rem;
            color: #8a94a6;
            padding-top: 45px;
            font-weight: 300;
        }
        .tp-info-box {
            background: #eef6ff;
            border-radius: 8px;
            padding: 16px 20px;
            margin-top: 24px;
            color: #004b8d;
            font-weight: 700;
            border: 1px solid #d8eaff;
        }
        .tp-info-box-sub {
            font-weight: 400;
            margin-top: 6px;
            color: #263746;
        }
        .tp-warning-box {
            background: #fff8e5;
            border-radius: 8px;
            padding: 16px 20px;
            margin-top: 14px;
            color: #8a5a00;
            font-weight: 700;
            border: 1px solid #f5e4b6;
        }
        .tp-warning-box-sub {
            font-weight: 400;
            margin-top: 6px;
            color: #5f4300;
        }
        .tp-muted-button {
            border: 1px solid #d9dfe8;
            border-radius: 7px;
            padding: 10px 18px;
            text-align: center;
            color: #8a94a6;
            font-weight: 600;
            margin-top: 28px;
            background: #ffffff;
        }
        .tp-active-button-note {
            border: 1px solid #b8d6f2;
            border-radius: 7px;
            padding: 10px 18px;
            text-align: center;
            color: #005a9c;
            font-weight: 700;
            margin-top: 28px;
            background: #f8fbff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="tp-start-wrap">
            <div class="tp-start-title">Start Here</div>
            <div class="tp-start-subtitle">
                Begin your total protein analysis by uploading the quantitative proteomics matrix from the left panel.
                Once uploaded, review the matrix, define groups, run QC, perform statistics, and export results.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, arrow_col, col2 = st.columns([0.46, 0.08, 0.46])

    with col1:
        st.markdown(
            """
            <div class="tp-step-card">
                <div>
                    <span class="tp-step-number">1</span>
                    <span class="tp-step-heading">Upload Data</span>
                </div>
                <div class="tp-step-body">
                    Upload your total proteomics matrix as a CSV or Excel file using the left sidebar.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with arrow_col:
        st.markdown("<div class='tp-step-arrow'>›</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            """
            <div class="tp-step-card">
                <div>
                    <span class="tp-step-number">2</span>
                    <span class="tp-step-heading">Review Matrix</span>
                </div>
                <div class="tp-step-body">
                    Confirm detected abundance columns, then proceed to group setup, QC, statistical analysis, and reporting.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="tp-info-box">
            Upload data using the left sidebar
            <div class="tp-info-box-sub">
                Required input: one CSV or Excel file containing protein identifiers and raw abundance/reporter intensity columns.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if upload_disabled:
        st.markdown(
            """
            <div class="tp-warning-box">
                No data uploaded yet
                <div class="tp-warning-box-sub">
                    Please upload a proteome data file to unlock the workflow.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    btn_class = "tp-muted-button" if upload_disabled else "tp-active-button-note"
    st.markdown(f"<div class='{btn_class}'>{button_text}</div>", unsafe_allow_html=True)



# -----------------------------------------------------------------------------
# App Helper Functions
# -----------------------------------------------------------------------------
def _uniq(seq):
    """Return unique items while preserving original order."""
    seen = set()
    out = []
    for x in seq or []:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _sanitize(selected, allowed):
    """Keep selected items that are still valid options."""
    allowed_set = set(allowed or [])
    return _uniq([x for x in (selected or []) if x in allowed_set])


def _file_signature(file) -> Optional[str]:
    """Return a stable signature for the uploaded file so stale state can be cleared."""
    return f"{file.name}_{file.size}" if file else None


def _reset_overview_flags():
    """Clear lower-level UI mode flags when switching workflow panels."""
    for k in ["qc_plot_type", "explore_mode_tp"]:
        ss.pop(k, None)


def _set_active_panel(active_key: str):
    """Open one workflow panel and close the others."""
    panel_keys = ["show_global_select", "show_overview", "show_suite", "show_report"]
    for k in panel_keys:
        ss[k] = (k == active_key)
    if active_key == "show_global_select":
        _reset_overview_flags()


def render_sidebar_controls():
    """Render sidebar upload, example dataset, and reset controls."""
    st.sidebar.markdown("### Total Proteomics")
    st.sidebar.caption("v1.0 | Bioinformatics Workflow")
    st.sidebar.divider()

    st.sidebar.markdown("**Upload Data**")
    file = st.sidebar.file_uploader("Upload proteome data (CSV or Excel)", type=["csv", "xlsx"])

    with st.sidebar.expander("Download Example Dataset", expanded=False):
        st.caption("Use this example file to see the required formatting and test the workflow.")
        script_dir = Path(__file__).resolve().parent
        example_path = script_dir / "assets" / "example" / "TP" / "Protein_abundance.csv"

        if example_path.exists():
            add_download_button(example_path.read_bytes(), "Protein_abundance_example.csv", label="Download Protein_abundance.csv", key="save_ex_tp", mime="text/csv")
        else:
            st.info("Example dataset not found in assets directory.")

    st.sidebar.divider()
    if st.sidebar.button(
        "Reset Session",
        type="secondary",
        help="Clear all current data, selections, and cache to start fresh."
    ):
        _reset_all_data(clear_cache=True)
        st.rerun()

    return file


def handle_file_change(file) -> None:
    """Clear derived state when a different input file is uploaded."""
    current_sig = _file_signature(file)
    if current_sig != ss.get("upload_sig"):
        _reset_all_data()
        ss["upload_sig"] = current_sig


def add_gene_id_and_row_id(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Add canonical GeneID and RowID columns without changing downstream logic."""
    raw_df = raw_df.copy()

    gene_col_match = next((c for c in raw_df.columns if re.search(r"(?i)^gene", str(c))), None)
    if gene_col_match:
        raw_df["GeneID"] = raw_df[gene_col_match]
    elif "Description" in raw_df.columns:
        raw_df["GeneID"] = raw_df["Description"].apply(
            lambda x: re.search(r"GN=([\w\-]+)", str(x)).group(1)
            if pd.notna(x) and re.search(r"GN=([\w\-]+)", str(x))
            else None
        )

    accession_col = next((c for c in ["Accession", "Master Protein Accessions"] if c in raw_df.columns), None)
    if "GeneID" in raw_df.columns and accession_col:
        raw_df["RowID"] = raw_df["GeneID"].astype(str) + "_" + raw_df[accession_col].astype(str)
    elif "GeneID" in raw_df.columns:
        raw_df["RowID"] = raw_df["GeneID"].astype(str)
    elif accession_col:
        raw_df["RowID"] = raw_df[accession_col].astype(str)
    else:
        raw_df["RowID"] = "Feature_" + raw_df.index.astype(str)

    return raw_df


def load_uploaded_proteomics_data(file) -> Optional[pd.DataFrame]:
    """Load uploaded CSV/Excel once and add RowID metadata."""
    if ss.get("raw_df") is None:
        try:
            raw_df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
            ss["raw_df"] = add_gene_id_and_row_id(raw_df)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None
    return ss["raw_df"]


def detect_numeric_columns_tp(df: pd.DataFrame) -> list:
    """Detect raw abundance/reporter/intensity columns using the original filtering rules."""
    raw_numeric_cols = [c for c in df.columns if re.search(r"(?i)abundance|reporter|intensity", str(c))]
    numeric_cols = [
        c for c in raw_numeric_cols
        if not re.search(r"(?i)normalized|grouped|scaled|s/n|corr|ratio|count|cv", str(c))
    ]
    if not numeric_cols and raw_numeric_cols:
        numeric_cols = [
            c for c in raw_numeric_cols
            if not re.search(r"(?i)normalized|scaled|s/n|corr|ratio|count|cv", str(c))
        ]
    return numeric_cols


def make_pretty_sample_labels(numeric_cols: list) -> dict:
    """Create unique display labels for sample columns."""
    pretty_labels = {}
    used_labels = set()
    for col in numeric_cols:
        base_label = clean_tmt_sample(col)
        if base_label in used_labels:
            i = 2
            new_label = f"{base_label} [{i}]"
            while new_label in used_labels:
                i += 1
                new_label = f"{base_label} [{i}]"
            base_label = new_label
        used_labels.add(base_label)
        pretty_labels[col] = base_label
    return pretty_labels


def render_uploaded_matrix_panel(df: pd.DataFrame, numeric_cols: list, downstream_active: bool) -> None:
    """Render uploaded matrix metrics, downloads, and preview table."""
    with st.expander("Integrated Quantitative Proteomics Matrix", expanded=not downstream_active):
        m1, m2 = st.columns(2)
        m1.metric("Total Proteins Detected", f"{df.shape[0]:,}")
        m2.metric("Total Sample Columns", len(numeric_cols))

        st.markdown("<br>", unsafe_allow_html=True)
        dl_col1, dl_col2, _ = st.columns([1.5, 1.5, 7])

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        with dl_col1:
            add_download_button(csv_bytes, "uploaded_proteomics_data.csv", label="Download CSV", key="save_uploaded_csv", mime="text/csv")

        buffer_excel = BytesIO()
        with pd.ExcelWriter(buffer_excel, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Merged_Data")
        excel_bytes = buffer_excel.getvalue()

        with dl_col2:
            add_download_button(excel_bytes, "uploaded_proteomics_data.xlsx", label="Download Excel", key="save_uploaded_excel", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, height=450)


def render_navigation_buttons() -> None:
    """Render the four main workflow buttons."""
    c1, c2, c3, c4 = st.columns(4)

    panels = [
        (c1, "show_global_select", "btn_global_select", "Data Filtering & Group Specification", "Assign samples to biological groups and set a control/baseline."),
        (c2, "show_overview", "btn_overview", "Data Integrity Check & Sanity Profiling", "Review preprocessing steps and Quality Control (QC) visualizations."),
        (c3, "show_suite", "btn_suite", "Statistical Analysis Suite", "Perform differential expression testing and visualize results (Volcano, Heatmaps, etc.)."),
        (c4, "show_report", "btn_report", "Results Download", "Compile and export a comprehensive PDF report and ZIP bundle of your analysis."),
    ]

    for col, state_key, button_key, label, help_text in panels:
        prefix = "▸ " if not ss[state_key] else "▾ "
        if col.button(prefix + label, use_container_width=True, key=button_key, help=help_text):
            if ss[state_key]:
                ss[state_key] = False
            else:
                _set_active_panel(state_key)


def get_pairwise_group_labels(default_control: str = "Control", default_test: str = "Test") -> tuple[str, str]:
    """Return control/test labels using the selected global reference group."""
    labels = ss.get("global_group_labels", []) or []
    control_idx = ss.get("global_control_group_idx", 0)

    if not labels:
        return default_control, default_test

    if control_idx >= len(labels):
        control_idx = 0

    control_label = labels[control_idx]
    test_candidates = [i for i in range(len(labels)) if i != control_idx]
    test_idx = test_candidates[0] if test_candidates else control_idx
    test_label = labels[test_idx]
    return control_label, test_label


def make_pairwise_sig_mask(res: pd.DataFrame, metric: str, alpha_thr: float, fc_thresh: float) -> pd.Series:
    """Create the same pairwise significance mask used across plots."""
    return (res[metric] <= alpha_thr) & (res["log2FC"].abs() >= fc_thresh)


def create_interactive_volcano_plot_tp(
    res: pd.DataFrame,
    sig_mask: pd.Series,
    gene_col_name: Optional[str],
    metric: str,
    alpha_thr: float,
    fc_thresh: float,
    control_label: str,
    test_label: str,
):
    """Create the interactive Plotly volcano plot."""
    plot_df = res.copy()
    plot_df["-log10(p)"] = -np.log10(
        pd.to_numeric(plot_df[metric], errors="coerce").clip(lower=1e-10)
    )
    plot_df["Sig"] = np.where(sig_mask, "Significant", "NS")

    x_axis_label = f"log<sub>2</sub> fold-change ({test_label}/{control_label})"

    fig = px.scatter(
        plot_df,
        x="log2FC",
        y="-log10(p)",
        color="Sig",
        hover_data=[gene_col_name] if gene_col_name else [],
        color_discrete_map={"Significant": "#d62728", "NS": "#a9a9a9"},
        labels={"log2FC": x_axis_label, "-log10(p)": f"-log10({metric})", "Sig": "Status"},
    )
    fig.add_hline(y=-np.log10(alpha_thr), line_dash="dash", line_color="black")
    fig.add_vline(x=fc_thresh, line_dash="dash", line_color="black")
    fig.add_vline(x=-fc_thresh, line_dash="dash", line_color="black")
    fig.update_xaxes(title_text=x_axis_label, showgrid=False, zeroline=False)
    fig.update_yaxes(title_text=f"-log10({metric})", showgrid=False, zeroline=False)
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=30, l=20, r=20, b=20),
        font=dict(size=12),
    )
    return fig


def create_interactive_ma_plot_tp(
    res: pd.DataFrame,
    sig_mask: pd.Series,
    gene_col_name: Optional[str],
    metric: str,
    fc_thresh: float,
    mean_cols: list,
    control_label: str,
    test_label: str,
):
    """Create the interactive Plotly MA plot."""
    plot_df = res.copy()
    for c in mean_cols:
        plot_df[c] = pd.to_numeric(plot_df[c], errors="coerce")
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df["log2FC"] = pd.to_numeric(plot_df["log2FC"], errors="coerce")
    plot_df["Average_Intensity"] = plot_df[mean_cols].mean(axis=1)
    plot_df["Sig"] = np.where(sig_mask, "Significant", "NS")
    plot_df = plot_df.dropna(subset=["Average_Intensity", "log2FC", metric]).copy()

    x_axis_label = "Average Intensity (log<sub>2</sub>)"
    y_axis_label = f"log<sub>2</sub> fold-change ({test_label}/{control_label})"

    fig = px.scatter(
        plot_df,
        x="Average_Intensity",
        y="log2FC",
        color="Sig",
        hover_data=[gene_col_name] if gene_col_name else [],
        color_discrete_map={"Significant": "#d62728", "NS": "#a9a9a9"},
        labels={"Average_Intensity": x_axis_label, "log2FC": y_axis_label, "Sig": "Status"},
    )
    fig.add_hline(y=0, line_color="black")
    fig.add_hline(y=fc_thresh, line_dash="dash", line_color="gray")
    fig.add_hline(y=-fc_thresh, line_dash="dash", line_color="gray")
    fig.update_xaxes(title_text=x_axis_label, showgrid=False, zeroline=False)
    fig.update_yaxes(title_text=y_axis_label, showgrid=False, zeroline=False)
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=40, l=20, r=20, b=20),
        font=dict(size=12),
    )
    return fig


# -----------------------------------------------------------------------------
# Shared common-library wrappers
# -----------------------------------------------------------------------------
# These wrappers keep the original function names used throughout this launcher
# while moving the actual plotting/export implementation into tp_common_utility.py.
def get_pairwise_group_labels(default_control: str = "Control", default_test: str = "Test") -> tuple[str, str]:
    return get_pairwise_group_labels_common(ss, default_control=default_control, default_test=default_test)


def make_pairwise_sig_mask(res: pd.DataFrame, metric: str, alpha_thr: float, fc_thresh: float) -> pd.Series:
    return make_pairwise_sig_mask_common(res, metric, alpha_thr, fc_thresh)


def create_static_volcano_plot_tp(
    res: pd.DataFrame,
    gene_col_name: Optional[str],
    metric: str,
    alpha_thr: float,
    fc_thresh: float,
    top_n_annot: int = 0,
    control_label: str = "Control",
    test_label: str = "Test",
):
    return create_static_volcano_plot_common(
        res=res,
        gene_col_name=gene_col_name,
        metric=metric,
        alpha_thr=alpha_thr,
        fc_thresh=fc_thresh,
        top_n_annot=top_n_annot,
        control_label=control_label,
        test_label=test_label,
        show_fc_lines=False,
        show_title=False,
    )


def create_interactive_volcano_plot_tp(
    res: pd.DataFrame,
    sig_mask: pd.Series,
    gene_col_name: Optional[str],
    metric: str,
    alpha_thr: float,
    fc_thresh: float,
    control_label: str,
    test_label: str,
):
    return create_interactive_volcano_plot_common(
        res=res,
        sig_mask=sig_mask,
        gene_col_name=gene_col_name,
        metric=metric,
        alpha_thr=alpha_thr,
        fc_thresh=fc_thresh,
        control_label=control_label,
        test_label=test_label,
        show_fc_lines=True,
    )


def create_static_ma_plot_tp(
    res: pd.DataFrame,
    gene_col_name: Optional[str],
    metric: str,
    alpha_thr: float,
    fc_thresh: float,
    mean_cols: list,
    top_n_annot: int = 0,
    control_label: str = "Control",
    test_label: str = "Test",
):
    return create_static_ma_plot_common(
        res=res,
        gene_col_name=gene_col_name,
        metric=metric,
        alpha_thr=alpha_thr,
        fc_thresh=fc_thresh,
        mean_cols=mean_cols,
        top_n_annot=top_n_annot,
        control_label=control_label,
        test_label=test_label,
        show_title=False,
    )


def create_interactive_ma_plot_tp(
    res: pd.DataFrame,
    sig_mask: pd.Series,
    gene_col_name: Optional[str],
    metric: str,
    fc_thresh: float,
    mean_cols: list,
    control_label: str,
    test_label: str,
):
    return create_interactive_ma_plot_common(
        res=res,
        sig_mask=sig_mask,
        gene_col_name=gene_col_name,
        metric=metric,
        fc_thresh=fc_thresh,
        mean_cols=mean_cols,
        control_label=control_label,
        test_label=test_label,
        show_title=False,
    )


def add_figure_download_buttons_tp(fig, base_filename: str, key_suffix: str = "", include_pdf: bool = True):
    return add_figure_download_buttons_common(
        fig=fig,
        base_filename=base_filename,
        key_suffix=key_suffix,
        include_pdf=include_pdf,
        save_to_downloads=False,
    )


def add_plotly_html_download_button_tp(fig, base_filename: str, key_suffix: str = ""):
    return add_plotly_html_download_button_common(
        fig=fig,
        base_filename=base_filename,
        key_suffix=key_suffix,
        save_to_downloads=False,
    )


# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------

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

    # Explicit application-specific override. Keep the GlycoExplorer variable as
    # a backward-compatible fallback for shared deployments.
    for env_name in (
        "TOTALPROTEOMICS_HOSTED",
        "TOTAL_PROTEOMICS_HOSTED",
        "GLYCOEXPLORER_HOSTED",
    ):
        explicit = _env_to_bool_for_downloads(os.environ.get(env_name))
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
    if any(os.environ.get(v) for v in hosted_env_vars):
        return True

    # Posit Connect installations do not always expose the same environment
    # variables. The deployed application path is a reliable final check.
    runtime_text = " ".join(
        [str(globals().get("__file__", "")), os.getcwd()]
    ).lower()
    return "rstudio-connect" in runtime_text or "/mnt/app" in runtime_text


def _get_downloads_dir():
    import os
    from pathlib import Path

    custom_dir = (
        os.environ.get("TOTALPROTEOMICS_DOWNLOAD_DIR", "").strip()
        or os.environ.get("TOTAL_PROTEOMICS_DOWNLOAD_DIR", "").strip()
        or os.environ.get("GLYCOEXPLORER_DOWNLOAD_DIR", "").strip()
    )
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


def add_hosted_safe_file_download_button(
    label,
    data,
    file_name,
    key,
    use_container_width=True,
    help_text=None,
    type="secondary",
):
    """
    Hosted/RConnect: return bytes through the browser with st.download_button.
    Local standalone: save directly to the local Downloads directory.
    """
    try:
        safe_name = _safe_download_filename(file_name)
        file_bytes = _coerce_to_bytes_for_downloads(data)
    except Exception as e:
        st.error(f"Could not prepare {file_name} for download: {e}")
        return False

    if _is_hosted_mode():
        return st.download_button(
            label=label,
            data=file_bytes,
            file_name=safe_name,
            mime=_guess_mime_from_filename(safe_name),
            key=key,
            help=help_text,
            type=type,
            use_container_width=use_container_width,
        )

    if st.button(
        label,
        key=key,
        help=help_text,
        type=type,
        use_container_width=use_container_width,
    ):
        try:
            outpath = _save_bytes_to_downloads(file_bytes, safe_name)
            st.success(f"Saved to: {outpath}")
            return True
        except Exception as e:
            st.error(f"Could not save {safe_name}: {e}")
    return False

def main():
    init_session_state_defaults()

    file = render_sidebar_controls()
    handle_file_change(file)

    if not file:
        render_professional_tp_start_panel(upload_disabled=True)
        st.markdown("<br>", unsafe_allow_html=True)
        faq_section()
        return

    df = load_uploaded_proteomics_data(file)
    if df is None:
        return

    numeric_cols = detect_numeric_columns_tp(df)
    if not numeric_cols:
        st.error("No valid raw 'Abundance' or 'Reporter' columns found. Check your file export settings.")
        st.stop()

    pretty_labels = make_pretty_sample_labels(numeric_cols)

    downstream_active = any(
        ss.get(k) for k in ["show_global_select", "show_overview", "show_suite", "show_report"]
    )
    render_uploaded_matrix_panel(df, numeric_cols, downstream_active)

    st.markdown("<br>", unsafe_allow_html=True)
    render_navigation_buttons()
    st.markdown("<br>", unsafe_allow_html=True)

    # =========================================================================
    # PILLAR 1: DATA FILTERING & Group Specification
    # =========================================================================
    if ss["show_global_select"]:

        left, right = st.columns([0.42, 0.58], gap="large")

        with left:
            with st.container(border=True):
                st.markdown("#### Define Experimental Groups")

                if "global_sample_cols_ui" not in ss:
                    default_samples = ss.get("global_sample_cols") or numeric_cols
                    ss["global_sample_cols_ui"] = _sanitize(default_samples, numeric_cols)

                ss["global_sample_cols_ui"] = _sanitize(ss["global_sample_cols_ui"], numeric_cols)

                sample_cols = st.multiselect(
                    "Active Samples",
                    options=numeric_cols,
                    format_func=lambda x: pretty_labels.get(x, x),
                    key="global_sample_cols_ui",
                )

                sample_cols = _sanitize(sample_cols, numeric_cols)

                if not sample_cols:
                    st.warning("Select at least 1 Active Sample column.")
                    ss["global_sample_cols"] = []
                    ss["global_group_samples"] = []
                    ss["global_group_labels"] = []
                    ss["global_all_groups"] = []
                    ss["batch_dict"] = {}
                else:
                    ss["global_sample_cols"] = sample_cols

                st.divider()

                if "global_n_groups_ui" not in ss:
                    ss["global_n_groups_ui"] = 2 if ss.get("global_n_groups", 2) == 2 else 3

                n_groups = st.radio(
                    "Number of groups",
                    [2, 3],
                    index=0 if ss["global_n_groups_ui"] == 2 else 1,
                    horizontal=True,
                    key="global_n_groups_ui",
                )
                ss["global_n_groups"] = n_groups

                n = len(sample_cols)
                d1 = sample_cols[: max(1, n // 3)]
                d2 = sample_cols[max(1, n // 3): max(2, 2 * n // 3)]
                d3 = sample_cols[max(2, 2 * n // 3):]

                g_groups = ss.get("global_group_samples") or []
                g_labels = ss.get("global_group_labels") or []

                if "global_grp1" not in ss:
                    grp1_default = (g_groups[0] if len(g_groups) > 0 else d1)
                    ss["global_grp1"] = _sanitize(grp1_default, sample_cols)
                ss["global_grp1"] = _sanitize(ss["global_grp1"], sample_cols)

                grp1 = st.multiselect(
                    "Group 1 columns",
                    options=sample_cols,
                    format_func=lambda x: pretty_labels.get(x, x),
                    key="global_grp1",
                )
                grp1 = _sanitize(grp1, sample_cols)

                if "global_g1_label" not in ss:
                    ss["global_g1_label"] = g_labels[0] if len(g_labels) > 0 else "Group1"
                g1_label = st.text_input("Group 1 label", value=ss["global_g1_label"], key="global_g1_label")

                options_g2 = [c for c in sample_cols if c not in set(grp1)]

                if "global_grp2" not in ss:
                    grp2_default = (g_groups[1] if len(g_groups) > 1 else d2)
                    ss["global_grp2"] = _sanitize(grp2_default, options_g2)
                ss["global_grp2"] = _sanitize(ss["global_grp2"], options_g2)

                grp2 = st.multiselect(
                    "Group 2 columns",
                    options=options_g2,
                    format_func=lambda x: pretty_labels.get(x, x),
                    key="global_grp2",
                )
                grp2 = _sanitize(grp2, options_g2)

                if "global_g2_label" not in ss:
                    ss["global_g2_label"] = g_labels[1] if len(g_labels) > 1 else "Group2"
                g2_label = st.text_input("Group 2 label", value=ss["global_g2_label"], key="global_g2_label")

                grp3, g3_label = [], None
                if n_groups == 3:
                    options_g3 = [c for c in sample_cols if c not in set(grp1) and c not in set(grp2)]

                    if "global_grp3" not in ss:
                        grp3_default = (g_groups[2] if len(g_groups) > 2 else d3)
                        ss["global_grp3"] = _sanitize(grp3_default, options_g3)
                    ss["global_grp3"] = _sanitize(ss["global_grp3"], options_g3)

                    grp3 = st.multiselect(
                        "Group 3 columns",
                        options=options_g3,
                        format_func=lambda x: pretty_labels.get(x, x),
                        key="global_grp3",
                    )
                    grp3 = _sanitize(grp3, options_g3)

                    if "global_g3_label" not in ss:
                        ss["global_g3_label"] = g_labels[2] if len(g_labels) > 2 else "Group3"
                    g3_label = st.text_input("Group 3 label", value=ss["global_g3_label"], key="global_g3_label")
                else:
                    ss.pop("global_grp3", None)
                    ss.pop("global_g3_label", None)

                grp1 = _sanitize(grp1, sample_cols)
                grp2 = _sanitize([c for c in grp2 if c not in set(grp1)], sample_cols)
                if n_groups == 3:
                    grp3 = _sanitize([c for c in grp3 if c not in set(grp1) and c not in set(grp2)], sample_cols)

                global_group_samples = [grp1, grp2] + ([grp3] if n_groups == 3 else [])
                global_group_labels  = [g1_label, g2_label] + ([g3_label] if n_groups == 3 else [])

                ss["global_group_samples"] = global_group_samples
                ss["global_group_labels"]  = global_group_labels
                ss["global_all_groups"]    = _uniq(grp1 + grp2 + (grp3 if n_groups == 3 else []))

                nonempty_groups = [(i, lbl, cols) for i, (lbl, cols) in enumerate(zip(global_group_labels, global_group_samples)) if cols]
                if nonempty_groups:
                    ss.setdefault("global_control_group_idx", nonempty_groups[0][0])
                    valid_idxs = [i for i, _, _ in nonempty_groups]
                    current = ss.get("global_control_group_idx", valid_idxs[0])
                    if current not in valid_idxs:
                        current = valid_idxs[0]

                    control_choice = st.selectbox(
                        "Global Reference (Baseline) Group",
                        options=valid_idxs,
                        format_func=lambda i: global_group_labels[i],
                        index=valid_idxs.index(current),
                        key="global_control_choice",
                    )
                    ss["global_control_group_idx"] = control_choice
                else:
                    ss["global_control_group_idx"] = 0
                    st.info("Assign samples to groups to select a global reference.")

            # ---- BATCH CORRECTION UI ----
            with st.container(border=True):
                st.markdown("#### Batch Effect Correction")

                batch_allowed_samples = ss.get("global_all_groups", [])
                batch_allowed_samples = _sanitize(
                    batch_allowed_samples,
                    sample_cols
                )

                if not batch_allowed_samples:
                    st.info(
                        "Assign samples to biological groups first "
                        "before enabling batch correction."
                    )
                else:
                    # ---------------------------------------------------------
                    # Persistent ON/OFF state
                    # ---------------------------------------------------------
                    # Use a temporary widget key and keep the permanent analysis
                    # setting separately. Widget keys can disappear while this
                    # workflow panel is hidden.
                    if "_tp_tmt_batch_toggle" not in ss:
                        ss["_tp_tmt_batch_toggle"] = bool(
                            ss.get("enable_batch_correction", False)
                        )

                    apply_batch = st.toggle(
                        "Apply Batch Effect Correction",
                        key="_tp_tmt_batch_toggle"
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
                                "Internal Reference Scaling (IRS)",
                                "Median Centering",
                                "Mean Centering",
                                "Standardize (Mean + Variance)"
                            ]

                            saved_method = ss.get(
                                "batch_method_ui",
                                "ComBat (Empirical Bayes)"
                            )
                            if saved_method not in batch_methods:
                                saved_method = "ComBat (Empirical Bayes)"

                            if "_tp_tmt_batch_method" not in ss:
                                ss["_tp_tmt_batch_method"] = saved_method

                            batch_method = st.selectbox(
                                "Batch Correction Method",
                                options=batch_methods,
                                help=(
                                    "IRS requires a common pool channel across "
                                    "multiplexes. ComBat requires neuroCombat."
                                ),
                                key="_tp_tmt_batch_method"
                            )
                            ss["batch_method_ui"] = batch_method

                            # -------------------------------------------------
                            # Persistent number of batches
                            # -------------------------------------------------
                            batch_count_options = [2, 3, 4]
                            saved_num_batches = int(
                                ss.get("batch_num_selected", 2)
                            )
                            if saved_num_batches not in batch_count_options:
                                saved_num_batches = 2

                            if "_tp_tmt_batch_num" not in ss:
                                ss["_tp_tmt_batch_num"] = saved_num_batches

                            num_batches = st.radio(
                                "Number of Batches",
                                batch_count_options,
                                horizontal=True,
                                key="_tp_tmt_batch_num"
                            )
                            ss["batch_num_selected"] = int(num_batches)

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
                                widget_key = f"_tp_tmt_batch_sel_{b}"

                                if widget_key in ss:
                                    prev_source = ss.get(widget_key, [])
                                else:
                                    prev_source = stored_batch_config.get(
                                        batch_name, []
                                    )

                                prev_selected = _sanitize(
                                    prev_source,
                                    batch_allowed_samples
                                )

                                available_for_batch = [
                                    c for c in batch_allowed_samples
                                    if c not in selected_so_far
                                    or c in prev_selected
                                ]

                                prev_selected = _sanitize(
                                    prev_selected,
                                    available_for_batch
                                )

                                # Re-seed the widget from permanent state when
                                # the Data Filtering panel is reconstructed.
                                ss[widget_key] = prev_selected

                                b_cols = st.multiselect(
                                    f"{batch_name} Samples",
                                    options=available_for_batch,
                                    format_func=lambda x: pretty_labels.get(x, x),
                                    key=widget_key
                                )

                                b_cols = _sanitize(
                                    b_cols,
                                    batch_allowed_samples
                                )
                                batch_dict[batch_name] = b_cols
                                selected_so_far = _uniq(
                                    selected_so_far + b_cols
                                )

                            # Save immediately; do not wait for the Run button.
                            ss["batch_config"] = {
                                k: list(v)
                                for k, v in batch_dict.items()
                            }

                            assigned_samples = _uniq(
                                [
                                    c
                                    for cols in batch_dict.values()
                                    for c in cols
                                ]
                            )

                            duplicated = [
                                c
                                for c in assigned_samples
                                if sum(
                                    c in cols
                                    for cols in batch_dict.values()
                                ) > 1
                            ]

                            unassigned_batch_samples = [
                                c
                                for c in batch_allowed_samples
                                if c not in assigned_samples
                            ]

                            nonempty_batches = {
                                k: v
                                for k, v in batch_dict.items()
                                if v
                            }

                            if duplicated:
                                st.error(
                                    "A sample cannot be assigned to more than "
                                    "one batch: "
                                    + ", ".join(
                                        [
                                            pretty_labels.get(c, c)
                                            for c in duplicated
                                        ]
                                    )
                                )

                            if unassigned_batch_samples:
                                st.warning(
                                    "These grouped samples are not assigned to "
                                    "any batch: "
                                    + ", ".join(
                                        [
                                            pretty_labels.get(c, c)
                                            for c in unassigned_batch_samples
                                        ]
                                    )
                                )

                            if len(nonempty_batches) < 2:
                                st.warning(
                                    "Assign samples to at least two non-empty "
                                    "batches."
                                )

                            # -------------------------------------------------
                            # IRS reference channels
                            # -------------------------------------------------
                            irs_ref_dict = {}
                            if batch_method == "Internal Reference Scaling (IRS)":
                                if nonempty_batches:
                                    st.markdown(
                                        "**Select the Reference/Pool Sample "
                                        "for each batch:**"
                                    )

                                    saved_irs = (
                                        ss.get("irs_ref_dict", {}) or {}
                                    )

                                    for b_idx, (b_name, b_cols) in enumerate(
                                        nonempty_batches.items()
                                    ):
                                        irs_key = f"_tp_tmt_irs_ref_{b_idx}"

                                        saved_ref = saved_irs.get(b_name)
                                        if saved_ref not in b_cols:
                                            saved_ref = b_cols[0]

                                        if irs_key not in ss:
                                            ss[irs_key] = saved_ref
                                        elif ss[irs_key] not in b_cols:
                                            ss[irs_key] = saved_ref

                                        irs_ref_dict[b_name] = st.selectbox(
                                            f"{b_name} Reference Channel",
                                            options=b_cols,
                                            format_func=lambda x: (
                                                pretty_labels.get(x, x)
                                            ),
                                            key=irs_key
                                        )

                                    ss["irs_ref_dict"] = dict(irs_ref_dict)
                            else:
                                irs_ref_dict = {}

                            current_signature = (
                                batch_method,
                                int(num_batches),
                                tuple(
                                    (name, tuple(cols))
                                    for name, cols in batch_dict.items()
                                ),
                                tuple(
                                    sorted(
                                        (
                                            str(k),
                                            str(v)
                                        )
                                        for k, v in irs_ref_dict.items()
                                    )
                                )
                            )

                            previous_signature = ss.get(
                                "batch_correction_run_signature"
                            )

                            if (
                                ss.get("batch_correction_requested", False)
                                and previous_signature is not None
                                and previous_signature != current_signature
                            ):
                                ss["batch_correction_requested"] = False
                                ss["batch_correction_applied"] = False
                                ss["batch_correction_applied_method"] = "None"
                                ss["expr_pre_batch"] = None
                                ss["expr_processed"] = None
                                ss["diff_results"] = None
                                ss["filtered_df"] = None
                                ss["filtered_ready"] = False

                                st.info(
                                    "Batch configuration changed. "
                                    "Click **Run Batch Correction** again."
                                )

                            can_run_batch = (
                                not duplicated
                                and not unassigned_batch_samples
                                and len(nonempty_batches) >= 2
                            )

                            if batch_method == "Internal Reference Scaling (IRS)":
                                can_run_batch = (
                                    can_run_batch
                                    and len(irs_ref_dict)
                                    == len(nonempty_batches)
                                )

                            if st.button(
                                "Run Batch Correction",
                                type="primary",
                                key="run_batch_correction_btn",
                                disabled=not can_run_batch,
                                help=(
                                    "Validate and lock this batch design. "
                                    "The correction is then applied with the "
                                    "selected preprocessing settings in the "
                                    "Data Integrity module."
                                )
                            ):
                                ss["batch_dict"] = {
                                    k: list(v)
                                    for k, v in nonempty_batches.items()
                                }
                                ss["batch_config"] = {
                                    k: list(v)
                                    for k, v in batch_dict.items()
                                }
                                if batch_method == "Internal Reference Scaling (IRS)":
                                    ss["irs_ref_dict"] = dict(irs_ref_dict)

                                ss["batch_correction_requested"] = True
                                ss["batch_correction_applied"] = False
                                ss["batch_correction_applied_method"] = batch_method
                                ss["batch_correction_run_signature"] = current_signature
                                ss["batch_correction_last_error"] = None

                                ss["expr_pre_batch"] = None
                                ss["expr_processed"] = None
                                ss["diff_results"] = None
                                ss["filtered_df"] = None
                                ss["filtered_ready"] = False

                                st.success(
                                    f"{batch_method} batch setup saved. "
                                    "Open Data Integrity Check & Sanity Profiling "
                                    "to apply the correction with the selected "
                                    "preprocessing settings."
                                )

                            elif ss.get("batch_correction_requested", False):
                                if ss.get("batch_correction_applied", False):
                                    st.success(
                                        f"{ss.get('batch_correction_applied_method', batch_method)} "
                                        "applied successfully. QC can compare "
                                        "before and after batch correction, and "
                                        "downstream analysis uses corrected abundances."
                                    )
                                else:
                                    st.info(
                                        f"{batch_method} batch setup is ready. "
                                        "Open Data Integrity Check & Sanity Profiling "
                                        "to apply the correction."
                                    )

                    else:
                        ss["batch_dict"] = {}
                        ss["irs_ref_dict"] = {}
                        ss["batch_method_ui"] = "None"
                        ss["batch_correction_requested"] = False
                        ss["batch_correction_applied"] = False
                        ss["batch_correction_applied_method"] = "None"
                        ss["batch_correction_run_signature"] = None
                        ss["expr_pre_batch"] = None
                        ss["expr_processed"] = None
                        ss["diff_results"] = None
                        ss["filtered_df"] = None
                        ss["filtered_ready"] = False

        with right:
            with st.container(border=True):
                st.markdown("#### Sample Mapping Summary")
                if not ss["global_all_groups"]:
                    st.info("No groups assigned yet. Select samples on the left.")
                else:
                    rows = []
                    for c in ss.get("global_sample_cols", []):
                        grp_name = "Unassigned"
                        grp_num = None
                        is_ctrl = False
                        
                        for i, g_cols in enumerate(ss.get("global_group_samples", [])):
                            if c in g_cols:
                                grp_name = ss["global_group_labels"][i] if i < len(ss["global_group_labels"]) else f"Group{i+1}"
                                grp_num = i + 1
                                is_ctrl = (ss.get("global_control_group_idx") is not None and i == ss["global_control_group_idx"])
                                break
                                
                        batch_name = "Unassigned"
                        if ss.get("enable_batch_correction"):
                            visible_batch_dict = (
                                ss.get("batch_config", {})
                                or ss.get("batch_dict", {})
                            )
                            for b_name, b_cols in visible_batch_dict.items():
                                if c in b_cols:
                                    batch_name = b_name
                                    break
                        else:
                            batch_name = "N/A"
                                
                        rows.append({
                            "Column": c,
                            "Sample": pretty_labels.get(c, c),
                            "Group": grp_name,
                            "Group #": grp_num,
                            "Batch": batch_name,
                            "Is_Reference": is_ctrl
                        })
                    
                    meta_df = pd.DataFrame(rows)
                    assigned_df = meta_df[meta_df["Group"] != "Unassigned"].copy()
                    
                    if assigned_df.empty:
                        st.info("No selected samples are assigned to a group yet.")
                    else:
                        st.dataframe(assigned_df, use_container_width=True, hide_index=True)

    # =========================================================================
    # PILLAR 2: DATA INTEGRITY CHECK & SANITY PROFILING
    # =========================================================================
    if ss["show_overview"]:
        
        if not ss["global_all_groups"]:
            st.warning("Please assign groups in Data Filtering & Group Specification first.")
            st.stop()

        with st.container(border=True):
            st.markdown("#### Global Matrix Preprocessing")
            
            st.markdown(
                "<p class='subtitle-text'>Configure how raw intensities are cleaned and transformed "
                "before generating Quality Control (QC) plots or passing data to the Statistical Suite.</p>",
                unsafe_allow_html=True
            )
            
            with st.expander("Preprocessing Parameter Definitions & Guide", expanded=False):
                st.markdown("""
                **Transformations (Adjusting the Data Shape):**
                * **Treat 0 as missing (NA):** In mass spectrometry, a true "0" usually means the protein/peptide was below the instrument's limit of detection. Treating 0s as true missing values (NA) prevents them from artificially dragging down group means and skewing variance calculations.
                * **Log2 transform:** Converts multiplicative intensity differences into additive, linear scales. This is a strict prerequisite for most parametric statistical tests (like t-tests and ANOVA) and helps stabilize variance between highly abundant and low-abundance proteins.
                * **Quantile normalize:** Forces the overall intensity distribution of every sample to be statistically identical. This is highly recommended for TMT multiplexed datasets or label-free runs with visible technical batch effects to ensure true comparability.
                """)
            st.divider()

            colA, colB = st.columns([0.5, 0.5])
            with colA:
                pre_zero = st.checkbox("Treat 0 as missing (NA)", value=ss.get("pre_treat_zero", True), key="pre_treat_zero")
            with colB:
                pre_log2 = st.checkbox("Log2 transform", value=ss.get("pre_log2", True), key="pre_log2")
                pre_quant = st.checkbox("Quantile normalize", value=ss.get("pre_quantile", True), key="pre_quantile")

            with st.spinner("Processing Matrix..."):
                expr_raw = (
                    df[["RowID"] + ss["global_all_groups"]]
                    .groupby("RowID")
                    .mean()
                    .apply(pd.to_numeric, errors="coerce")
                )
                expr_raw.rename(columns=pretty_labels, inplace=True)

                expr_base = expr_raw.copy()

                # 1. Zero Handling
                if pre_zero:
                    expr_base = expr_base.replace(0, np.nan)

                # 2. Log2 Transform
                if pre_log2:
                    mn = expr_base.min().min()
                    if pd.notna(mn) and mn <= 0:
                        expr_base = expr_base + abs(mn) + 1e-6
                    expr_base = np.log2(expr_base)

                batch_method = ss.get("batch_method_ui", "None")
                batch_requested = (
                    ss.get("enable_batch_correction", False)
                    and ss.get("batch_correction_requested", False)
                    and batch_method != "None"
                )

                expr_pre_batch = expr_base.copy()
                expr_proc = expr_base.copy()

                if batch_requested:
                    translated_batch_dict = {}
                    for b_name, b_cols in ss.get("batch_dict", {}).items():
                        translated_batch_dict[b_name] = [
                            pretty_labels.get(c, c)
                            for c in b_cols
                        ]

                    translated_irs_refs = {
                        k: pretty_labels.get(v, v)
                        for k, v in ss.get("irs_ref_dict", {}).items()
                    }

                    expr_proc = apply_batch_correction(
                        df=expr_proc,
                        batch_dict=translated_batch_dict,
                        method=batch_method,
                        is_log2=pre_log2,
                        irs_refs=translated_irs_refs
                    )

                    if ss.get("batch_correction_last_error"):
                        ss["batch_correction_applied"] = False
                    else:
                        ss["batch_correction_applied"] = True
                        ss["batch_correction_applied_method"] = batch_method
                else:
                    ss["batch_correction_applied"] = False
                    ss["batch_correction_applied_method"] = "None"

                # 4. Quantile Normalization
                # Apply the same normalization to both matrices so Before/After
                # PCA differs by batch correction rather than normalization.
                if pre_quant:
                    expr_pre_batch = quantile_normalize(expr_pre_batch)
                    expr_proc = quantile_normalize(expr_proc)

                ss["expr_raw"] = expr_raw
                ss["expr_pre_batch"] = expr_pre_batch
                ss["expr_processed"] = expr_proc

            if ss.get("batch_correction_applied", False):
                st.success(
                    f"{ss.get('batch_correction_applied_method', batch_method)} "
                    "applied successfully. QC can compare before and after "
                    "batch correction, and downstream analysis will use "
                    "corrected abundances."
                )
            elif (
                ss.get("enable_batch_correction", False)
                and not ss.get("batch_correction_requested", False)
            ):
                st.warning(
                    "Batch correction is enabled but has not been run. "
                    "Return to Data Filtering & Group Specification and click "
                    "**Run Batch Correction**."
                )

            st.success(
                f"Matrix processed! Tracking "
                f"{expr_proc.shape[0]} unique protein features."
            )
        
        st.divider()    
        st.markdown("#### Quality Control Plots")
        
        design_rows = []
        for i, cols in enumerate(ss["global_group_samples"]):
            for c in cols: design_rows.append({"Sample": pretty_labels[c], "Group": ss["global_group_labels"][i]})
        design_df = pd.DataFrame(design_rows)

        qc_target = expr_proc

        left_qc, right_qc = st.columns([0.35, 0.65], gap="large")
        
        with left_qc:
            with st.container(border=True):
                
                if ss.get("batch_correction_applied", False):
                    view_data = st.radio(
                        "Data Source:",
                        [
                            "After Batch Correction",
                            "Before Batch Correction",
                            "Raw (Un-transformed)"
                        ],
                        key="tmt_qc_data_source"
                    )

                    if view_data == "Before Batch Correction":
                        qc_target = ss["expr_pre_batch"]
                    elif view_data == "Raw (Un-transformed)":
                        qc_target = expr_raw
                    else:
                        qc_target = expr_proc

                elif pre_quant or pre_log2:
                    view_data = st.radio(
                        "Data Source:",
                        ["Processed", "Raw (Un-transformed)"],
                        key="tmt_qc_data_source"
                    )
                    if view_data == "Raw (Un-transformed)":
                        qc_target = expr_raw
                    else:
                        qc_target = expr_proc

                qc_mode = st.radio(
                    "Select QC Plot:", 
                    ["PCA", "Density", "Correlation Matrix", "QC Heatmap", "Boxplot", "Missing Values"],
                    key="qc_plot_type"
                )
                
                if qc_mode == "PCA":
                    st.divider()
                    st.markdown("**PCA Display Options**")
                    
                    data_for_pca = qc_target.copy().dropna(how="all")
                    n_comps = min(10, data_for_pca.shape[0], data_for_pca.shape[1])
                    n_comps = max(2, n_comps)
                    pc_cols = [f"PC{i+1}" for i in range(n_comps)]
                    
                    col_x, col_y = st.columns(2)
                    with col_x:
                        ss["pca_x"] = st.selectbox("X-axis Component", options=pc_cols, index=0, key="pca_x_ui")
                    with col_y:
                        ss["pca_y"] = st.selectbox("Y-axis Component", options=pc_cols, index=1 if len(pc_cols)>1 else 0, key="pca_y_ui")
                        
                    colA, colB = st.columns(2)
                    with colA:
                        ss["pca_hide_outliers"] = st.checkbox("Hide outliers", value=False, key="pca_hide_outliers_ui")
                        ss["pca_draw_ellipses"] = st.checkbox("Draw ellipses", value=True, key="pca_draw_ellipses_ui")
                    with colB:
                        ss["pca_label_outliers"] = st.checkbox("Label outliers", value=True, key="pca_label_outliers_ui")
                        
                    ss["pca_zoom_q"] = st.slider("Zoom (quantile)", 0.90, 1.0, 1.0, 0.01, key="pca_zoom_q_ui", help="Axis limits use these quantiles (visual zoom only).")
                    
                    st.markdown("**Group Colors**")
                    unique_groups = design_df["Group"].unique()
                    default_colors = sns.color_palette("Set2", n_colors=len(unique_groups)).as_hex()
                    
                    pca_palette = {}
                    cols = st.columns(len(unique_groups))
                    for idx, grp in enumerate(unique_groups):
                        with cols[idx % len(cols)]:
                            pca_palette[grp] = st.color_picker(grp, default_colors[idx], key=f"pca_color_{grp}_ui")
                    ss["pca_palette"] = pca_palette

                elif qc_mode == "QC Heatmap":
                    st.divider()
                    st.markdown("**Heatmap Options**")
                    topn = st.slider("Top N variable proteins", 10, 100, 50, key="qc_topn_proc")
                    
                    st.markdown("**Group Colors**")
                    unique_groups = design_df["Group"].unique()
                    default_colors = sns.color_palette("Set2", n_colors=len(unique_groups)).as_hex()
                    
                    qc_hm_palette = {}
                    for idx, grp in enumerate(unique_groups):
                        qc_hm_palette[grp] = st.color_picker(
                            f"Color for {grp}", 
                            default_colors[idx], 
                            key=f"qc_hm_color_{grp}"
                        )
                    ss["qc_hm_palette"] = qc_hm_palette

        with right_qc:
            if qc_mode == "PCA":
                short_groups = [
                    [pretty_labels[c] for c in gcols]
                    for gcols in ss["global_group_samples"]
                ]

                pca_title = "PCA of Samples"

                if ss.get("batch_correction_applied", False):
                    selected_source = ss.get(
                        "tmt_qc_data_source",
                        "After Batch Correction"
                    )

                    if selected_source == "Before Batch Correction":
                        pca_title = "PCA Before Batch Correction"
                    elif selected_source == "After Batch Correction":
                        pca_title = "PCA After Batch Correction"
                    elif selected_source == "Raw (Un-transformed)":
                        pca_title = "PCA of Raw TMT Abundances"

                    st.caption(
                        "Use **Data Source** on the left to switch between "
                        "Before Batch Correction and After Batch Correction."
                    )

                # Keep exactly one PCA plot on the page. create_pca_plot()
                # creates its own download buttons, so this avoids duplicate
                # Streamlit element keys.
                fig = create_pca_plot(
                    df=qc_target,
                    group_samples=short_groups,
                    group_labels=ss["global_group_labels"],
                    title=pca_title,
                    pc_x=ss.get("pca_x", "PC1"),
                    pc_y=ss.get("pca_y", "PC2"),
                    hide_outliers=ss.get("pca_hide_outliers", False),
                    label_outliers=ss.get("pca_label_outliers", True),
                    draw_ellipses=ss.get("pca_draw_ellipses", False),
                    zoom_q=ss.get("pca_zoom_q", 1.0),
                    palette=ss.get("pca_palette", None)
                )

            elif qc_mode == "Density":
                fig, ax = plt.subplots(figsize=(7, 5))
                for sample in qc_target.columns:
                    sns.kdeplot(qc_target[sample].dropna(), ax=ax, label=sample, lw=1.5, alpha=0.6)
                ax.set_title("Sample Density")
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
                st.pyplot(fig)
                
                if qc_target is expr_raw:
                    ss["report_density_raw_fig"] = fig
                else:
                    ss["report_density_proc_fig"] = fig
                    
                add_plot_download_buttons(fig, "density_plot")

            elif qc_mode == "Correlation Matrix":
                fig, ax = plt.subplots(figsize=(7, 6))
                sns.heatmap(qc_target.corr(), cmap="vlag", ax=ax, cbar_kws={"shrink": 0.5})
                st.pyplot(fig)
                
                ss["report_corr_fig"] = fig  
                add_plot_download_buttons(fig, "correlation_heatmap")

            elif qc_mode == "QC Heatmap":
                var = qc_target.var(axis=1, skipna=True)
                top_idx = var.nlargest(topn).index
                hm = (
                    qc_target.loc[top_idx]
                    .copy()
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna(axis=0, how="all")
                )

                z = (
                    hm.sub(hm.mean(axis=1), axis=0)
                    .div(hm.std(axis=1), axis=0)
                    .apply(lambda r: r.fillna(r.median()), axis=1)
                    .fillna(0.0)
                )

                # Remove RowID or any index/column axis labels from heatmap
                z.index.name = None
                z.columns.name = None

                if not np.isfinite(z.to_numpy()).all():
                    st.error("QC heatmap contains non-finite values.")
                else:
                    sample_to_group = design_df.set_index("Sample")["Group"]
                    unique_groups = design_df["Group"].unique()

                    fallback_colors = {
                        g: c for g, c in zip(
                            unique_groups,
                            sns.color_palette("Set2", len(unique_groups)).as_hex()
                        )
                    }

                    group_palette = ss.get("qc_hm_palette", fallback_colors)

                    col_colors = (
                        pd.Series(z.columns, index=z.columns)
                        .map(sample_to_group)
                        .map(group_palette)
                    )

                    fig = sns.clustermap(
                        z,
                        cmap="vlag",
                        col_colors=col_colors,
                        figsize=(10, 10),
                        xticklabels=True,
                        yticklabels=True,
                        tree_kws=dict(linewidths=0.5)
                    )

                    # Remove y-axis and x-axis labels such as RowID
                    fig.ax_heatmap.set_ylabel("")
                    fig.ax_heatmap.set_xlabel("")

                    plt.setp(fig.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8)
                    plt.setp(
                        fig.ax_heatmap.get_xticklabels(),
                        rotation=45,
                        horizontalalignment="right"
                    )

                    handles = [
                        mpatches.Patch(color=group_palette[g], label=g)
                        for g in unique_groups
                        if g in group_palette
                    ]

                    fig.ax_col_dendrogram.legend(
                        handles=handles,
                        title="Group",
                        loc="center left",
                        bbox_to_anchor=(1.02, 0.5),
                        frameon=True
                    )

                    st.pyplot(fig.fig)

                    ss["report_qc_heatmap_fig"] = fig.fig
                    add_plot_download_buttons(fig.fig, "qc_heatmap")

            elif qc_mode == "Boxplot":
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.boxplot(data=qc_target, ax=ax)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                st.pyplot(fig)
                
                ss["report_qc_boxplot_fig"] = fig
                add_plot_download_buttons(fig, "qc_boxplot")
                
            elif qc_mode == "Missing Values":
                miss = qc_target.isna().mean() * 100
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(x=miss.index, y=miss.values, ax=ax)
                ax.set_title("% Missing Values")
                plt.xticks(rotation=45, ha="right")
                st.pyplot(fig)
                
                ss["report_qc_missing_fig"] = fig
                add_plot_download_buttons(fig, "missing_values")

            elif qc_mode == "Dendrogram":
                X = qc_target.T.copy().replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
                X = X.loc[:, X.std(axis=0) > 0].apply(lambda c: c.fillna(c.median()), axis=0).fillna(0.0)
                if X.shape[0] < 2 or X.shape[1] < 2:
                    st.warning("Not enough data to build dendrogram.")
                else:
                    Z = linkage(X.to_numpy(), method="ward")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    dendrogram(Z, labels=X.index.tolist(), ax=ax)
                    plt.xticks(rotation=45, ha="right")
                    st.pyplot(fig)
                    
                    ss["report_dendrogram_fig"] = fig
                    add_plot_download_buttons(fig, "dendrogram")


    # =========================================================================
    # PILLAR 3: STATISTICAL ANALYSIS SUITE
    # =========================================================================
    if ss["show_suite"]:
        if ss.get("expr_processed") is None or ss["expr_processed"].empty:
            st.warning("Please verify your matrix in 'Data Integrity Check & Sanity Profiling' first.")
            st.stop()

        tabs = st.tabs(["Differential Analysis", "Results Visualizations"])

        with tabs[0]:
            expr = ss["expr_processed"]
            groups = [[pretty_labels[c] for c in gcols] for gcols in ss["global_group_samples"]]
            labels = ss["global_group_labels"]
            ctrl_idx = ss["global_control_group_idx"]

            active = [(lbl, cols, i) for i, (lbl, cols) in enumerate(zip(labels, groups)) if cols]
            
            with st.expander("Current global groups (read-only)", expanded=True):
                st.write(f"**Number of groups:** {ss['global_n_groups']}")
                control_label = labels[ctrl_idx] if ctrl_idx < len(labels) else "Unknown"
                st.write(f"**Reference group (Baseline):** {control_label}")
                for i, (lbl, cols) in enumerate(zip(labels, groups), start=1):
                    st.write(f"**Group {i} — {lbl}:** {len(cols)} samples")
                    st.caption(", ".join([str(c) for c in cols]) if cols else "(empty)")

            if len(active) < 2:
                st.error("Need at least 2 non-empty groups to run stats. Check Section ①.")
                st.stop()

            with st.container(border=True):
                st.markdown("#### Test Parameters and Execution")
                st.markdown(
                    "<p class='subtitle-text'>Configure the parameters for identifying differentially "
                    "abundant proteins. Expand the guide below for a detailed explanation of each setting.</p>",
                    unsafe_allow_html=True
                )
                
                with st.expander("Parameter Definitions & Guide", expanded=False):
                    st.markdown("""
                    **Preprocessing (Handling Missing Data & Noise):**
                    * **Replace 0/NA intensities (Zero-Imputation):** Log-fold calculations fail on true zeros. If enabled, the app calculates the minimum positive intensity for a given protein across all your samples, and replaces missing/zero values with 20% (1/5th) of that minimum value. 
                    * **Filter by presence (Drop Sparse Rows):** Removes proteins that are barely detected across your dataset to improve statistical power.
                    * **Presence threshold:** The minimum fraction of samples that must contain a valid intensity.
                    * **Presence rule:** Determines how the threshold above is applied (e.g., 'At least one group only' preserves on/off markers).

                    **Statistical Testing:**
                    * **Experimental design:** Select **Pairwise** to compare exactly two groups against each other, or **Three-group ANOVA** to test for any significant variance across three distinct groups simultaneously.
                    * **Statistical test (Pairwise only):** Choose between parametric (e.g., Student's t, Welch's t) or non-parametric tests (e.g., Mann-Whitney U, Wilcoxon).
                    """)

                can_pairwise = len(active) >= 2
                can_anova    = (len(active) == 3 and ss["global_n_groups"] == 3)

                with st.form("run_diff_form"):
                    st.markdown("### Preprocessing (before statistics)")

                    do_zero_impute = st.checkbox(
                        "Replace 0/NA intensities before differential",
                        value=True,
                        help="Row-wise: for each row, 0/NA → (minimum positive intensity in that row) / 5"
                    )
                    
                    do_presence_filter = st.checkbox("Filter by presence (drop sparse rows)", value=True)
                    presence_min_frac = st.slider("Presence threshold (fraction)", 0.05, 1.00, 0.40, 0.05)
                    
                    presence_rule_ui = st.radio(
                        "Presence rule",
                        options=[
                            "Total samples only",
                            "At least one group only",
                            "OR: Total samples OR one group",
                            "AND: Total samples AND one group",
                        ],
                        index=2
                    )
                    presence_rule_key = {
                        "Total samples only": "all_only",
                        "At least one group only": "group_only",
                        "OR: Total samples OR one group": "or",
                        "AND: Total samples AND one group": "and",
                    }[presence_rule_ui]

                    st.divider()
                    st.markdown("### Comparison")

                    comparison_choices = []
                    if can_pairwise: comparison_choices.append("Pairwise")
                    else: comparison_choices.append("Pairwise — need ≥2 non-empty groups")
                    
                    if can_anova: comparison_choices.append("Three-group ANOVA")
                    else: comparison_choices.append("Three-group ANOVA — need exactly 3 non-empty groups")

                    comparison_type = st.radio("Select experimental design:", comparison_choices, index=0)
                    selected_pair = None
                    specific_test = "Welch's t-test (Independent, Unequal Variance)"

                    if comparison_type.startswith("Pairwise"):
                        if can_pairwise:
                            pairs = {
                                f"{active[i][0]} vs {active[j][0]}": (active[i][1], active[j][1], active[i][0], active[j][0])
                                for i in range(len(active)) for j in range(i + 1, len(active))
                            }
                            col1, col2 = st.columns(2)
                            with col1:
                                selected_pair = st.selectbox("Select pairwise comparison:", list(pairs.keys()))
                            with col2:
                                specific_test = st.selectbox("Select statistical test:", [
                                    "Student's t-test (Independent, Equal Variance)",
                                    "Welch's t-test (Independent, Unequal Variance)",
                                    "Paired t-test (Matched Samples)",
                                    "Mann-Whitney U (Non-parametric Independent)",
                                    "Wilcoxon Signed-Rank (Non-parametric Paired)"
                                ], index=1)
                    else:
                        selected_pair = "Three-group ANOVA"

                    run_button = st.form_submit_button(
                        "Run Analysis", 
                        type="primary", 
                        help="Execute the selected statistical test on the grouped data."
                    )

                if run_button:
                    try:
                        meta_df = df.drop_duplicates(subset="RowID").set_index("RowID")
                        meta_cols = [c for c in ["GeneID", "Genes", "Description", "Accession", "Master Protein Accessions"] if c in meta_df.columns]
                        analysis_df = meta_df[meta_cols].join(expr, how="right").reset_index()

                        target_cols = [c for g in groups for c in g]
                        if do_presence_filter:
                            analysis_df = filter_by_presence_rule(
                                analysis_df, target_cols, groups, presence_min_frac, presence_rule_key
                            )
                        if do_zero_impute:
                            analysis_df = replace_zeros_rowwise_min_fraction(
                                analysis_df, target_cols, frac=0.2, replace_nan=True
                            )

                        if comparison_type.startswith("Pairwise"):
                            group1, group2, label1, label2 = pairs[selected_pair]
                            
                            if control_label == label1:
                                control_cols, test_cols = group1, group2
                                control_lab, test_lab = label1, label2
                            elif control_label == label2:
                                control_cols, test_cols = group2, group1
                                control_lab, test_lab = label2, label1
                            else:
                                control_cols, test_cols = group1, group2
                                control_lab, test_lab = label1, label2

                            diff_df = compute_diffexp_tp(
                                df=analysis_df, control_cols=control_cols, test_cols=test_cols,
                                control_label=control_lab, test_label=test_lab, test_type=specific_test
                            )
                        else:
                            diff_df = compute_diffexp_anova_tp(
                                df=analysis_df, 
                                group1_cols=groups[0], group2_cols=groups[1], group3_cols=groups[2],
                                group1_label=labels[0], group2_label=labels[1], group3_label=labels[2]
                            )
                        
                        ss["diff_results"] = diff_df
                        st.success(f"Differential analysis complete: {selected_pair}")
                    
                    except Exception as e:
                        st.error(f"Differential analysis failed: {e}")

            if ss.get("diff_results") is not None:
                st.success("Differential analysis complete. Results available below.")
                st.dataframe(ss["diff_results"], use_container_width=True)
                
                with st.expander("Download Full Dataset (Unfiltered)", expanded=False):
                    download_diff_df = ss["diff_results"].copy()
                    dl_col1, dl_col2 = st.columns(2)
                    
                    csv_diff = download_diff_df.to_csv(index=False).encode('utf-8')
                    with dl_col1:
                        add_bytes_download_button_tp(
                            label="Download Results (CSV)",
                            data=csv_diff,
                            filename="differential_analysis_results_totalproteome.csv",
                            key="save_diff_csv",
                            help_text="Save the complete differential analysis statistics table as a CSV file."
                        )

                    buffer_diff = BytesIO()
                    with pd.ExcelWriter(buffer_diff, engine='openpyxl') as writer:
                        download_diff_df.to_excel(writer, index=False, sheet_name='Differential_Results')
                    with dl_col2:
                        add_bytes_download_button_tp(
                            label="Download Results (Excel)",
                            data=buffer_diff.getvalue(),
                            filename="differential_analysis_results_totalproteome.xlsx",
                            key="save_diff_xlsx",
                            help_text="Save the complete differential analysis statistics table as an Excel file."
                        )

        # ---------------------------------------------------------------------
        # EXPLORE RESULTS TAB
        # ---------------------------------------------------------------------
        with tabs[1]:
            if ss.get("diff_results") is None:
                st.info("Run Differential Analysis first.")
            else:
                res = ss["diff_results"]
                expr = ss["expr_processed"]
                
                gene_col_name = next((c for c in ["GeneID", "Genes", "Gene", "Description"] if c in res.columns), None)
                accession_col_name = next((c for c in ["Accession", "Master Protein Accessions", "Protein Accessions", "Protein ID", "ProteinID", "UniProt", "Uniprot"] if c in res.columns), None)

                left, right = st.columns([0.35, 0.65], gap="large")

                for key in ["show_volcano", "show_waterfall", "show_ma", "show_density", "show_heatmap",  "show_profile"]:
                    ss.setdefault(key, False)

                with left:
                    with st.container(border=True):
                        
                        is_anova = "log2FC" not in res.columns
                        modes = ["Heatmap", "Protein Profile"] if is_anova else ["Volcano", "Waterfall", "MA Plot", "Heatmap", "Protein Profile"]

                        mode = st.radio(
                            "Select Analysis Module", 
                            modes, 
                            index=None, 
                            key="explore_mode_tp", 
                            horizontal=False,
                            label_visibility="collapsed"
                        )

                        flags = {
                            "Volcano": dict(show_volcano=True, show_waterfall=False, show_ma=False, show_density=False, show_heatmap=False, show_profile=False),
                            "Waterfall": dict(show_volcano=False, show_waterfall=True, show_ma=False, show_density=False, show_heatmap=False, show_profile=False),
                            "MA Plot": dict(show_volcano=False, show_waterfall=False, show_ma=True, show_density=False, show_heatmap=False, show_profile=False),
                            "Heatmap": dict(show_volcano=False, show_waterfall=False, show_ma=False, show_density=False, show_heatmap=True, show_profile=False),
                            "Protein Profile": dict(show_volcano=False, show_waterfall=False, show_ma=False, show_density=False, show_heatmap=False, show_profile=True),
                        }
                        
                        for k in ["show_volcano", "show_waterfall", "show_ma", "show_density", "show_heatmap", "show_profile"]:
                            ss[k] = False
                        
                        if mode is not None:
                            ss.update(flags.get(mode, {}))

                        # --- MODULE SPECIFIC OPTIONS ---
                        if ss.get("show_volcano", False):
                            st.markdown("**Volcano Plot Options**")
                            ss["volcano_plot_type"] = st.radio(
                                "Volcano display",
                                ["Static", "Interactive"],
                                horizontal=True,
                                key="volcano_plot_type_ui",
                                help="Interactive uses Plotly for hover/zoom. Static uses Matplotlib and can be exported as PNG/SVG/PDF."
                            )
                            metric_opts = ["q-value", "p-value"] if "q-value" in res.columns else ["p-value"]
                            ss["volcano_sig_col"] = st.radio(
                                "Significance metric",
                                metric_opts,
                                horizontal=True,
                                key="volcano_sig_metric_ui"
                            )
                            ss["volcano_alpha_thr"] = st.slider(
                                f"{ss['volcano_sig_col']} cutoff",
                                0.0001,
                                0.2,
                                0.05,
                                format="%.4f",
                                key="volcano_alpha_thr_ui"
                            )
                            ss["volcano_fc_thresh"] = st.slider(
                                "|log2FC| threshold",
                                0.0,
                                4.0,
                                1.0,
                                key="volcano_fc_thresh_ui"
                            )

                            sig_tmp = (res[ss["volcano_sig_col"]] <= ss["volcano_alpha_thr"]) & (
                                res["log2FC"].abs() >= ss["volcano_fc_thresh"]
                            )
                            max_annot = int(sig_tmp.sum())
                            ss["volcano_top_n_annot"] = st.slider(
                                "Number of significant proteins to annotate in downloaded static plot",
                                min_value=0,
                                max_value=max(0, min(max_annot, 100)),
                                value=min(10, max_annot),
                                step=1,
                                key="volcano_top_n_annot_ui",
                                help="Annotations are applied to the static Matplotlib volcano plot and exported PNG/SVG/PDF."
                            )

                        elif ss.get("show_waterfall", False):
                            st.markdown("**Waterfall Plot Options**")
                            metric_opts = ["q-value", "p-value"] if "q-value" in res.columns else ["p-value"]
                            ss["waterfall_sig_col"] = st.radio(
                                "Significance metric",
                                metric_opts,
                                horizontal=True,
                                key="waterfall_sig_metric_ui"
                            )
                            ss["waterfall_alpha_thr"] = st.slider(
                                f"{ss['waterfall_sig_col']} cutoff",
                                0.0001,
                                0.2,
                                0.05,
                                format="%.4f",
                                key="waterfall_alpha_thr_ui"
                            )
                            ss["waterfall_fc_thresh"] = st.slider(
                                "|log2FC| threshold",
                                0.0,
                                4.0,
                                1.0,
                                key="waterfall_fc_thresh_ui"
                            )
                            ss["waterfall_top_n"] = st.slider(
                                "Number of top up/down proteins to number",
                                3,
                                25,
                                10,
                                key="waterfall_top_n_ui"
                            )

                        elif ss.get("show_ma", False):
                            st.markdown("**MA Plot Options**")
                            ss["ma_plot_type"] = st.radio(
                                "MA plot display",
                                ["Static", "Interactive"],
                                horizontal=True,
                                key="ma_plot_type_ui",
                                help="Interactive uses Plotly for hover/zoom. Static uses Matplotlib and can be exported as PNG/SVG/PDF."
                            )
                            metric_opts = ["q-value", "p-value"] if "q-value" in res.columns else ["p-value"]
                            ss["ma_sig_col"] = st.radio("Significance metric", metric_opts, horizontal=True, key="ma_sig_metric_ui")
                            ss["ma_alpha_thr"] = st.slider(f"{ss['ma_sig_col']} cutoff", 0.0001, 0.2, 0.05, format="%.4f", key="ma_alpha_thr_ui")
                            ss["ma_fc_thresh"] = st.slider("|log2FC| threshold", 0.0, 4.0, 1.0, key="ma_fc_thresh_ui")
                            ss["ma_top_n"] = st.slider("Number of top proteins to annotate in static plot", 0, 50, 10, key="ma_top_n_ui")

                        elif ss.get("show_heatmap", False):
                            st.markdown("**Heatmap Options**")
                            metric_opts = ["q-value", "p-value"] if "q-value" in res.columns else ["p-value"]
                            ss["hm_sig_col"] = st.radio("Significance metric", metric_opts, horizontal=True)
                            ss["hm_alpha_thr"] = st.slider(f"{ss['hm_sig_col']} cutoff", 0.0001, 0.2, 0.05, format="%.4f")
                            if not is_anova:
                                ss["hm_fc_thresh"] = st.slider("|log2FC| threshold", 0.0, 4.0, 1.0)
                            else:
                                ss["hm_fc_thresh"] = 0.0
                                
                            ss["hm_top_n"] = st.slider("Top N for Heatmap", 5, 200, 25)
                            ss["hm_norm"] = st.radio("Normalize data", ["Z-score", "Raw Abundance"], horizontal=True)
                            ss["hm_cluster"] = st.selectbox("Clustering", ["Hierarchical", "K-Means", "None"])
                            
                            st.markdown("**Group Colors**")
                            active_groups = [g for g in ss.get("global_group_labels", []) if g]
                            default_colors = sns.color_palette("Set2", n_colors=max(1, len(active_groups))).as_hex()
                            diff_hm_palette = {}
                            for idx, grp in enumerate(active_groups):
                                diff_hm_palette[grp] = st.color_picker(f"Color for {grp}", default_colors[idx], key=f"diff_hm_color_{grp}")
                            ss["diff_hm_palette"] = diff_hm_palette

                        elif ss.get("show_profile", False):
                            st.markdown("**Protein Profile Options**")
                            metric_opts = ["q-value", "p-value"] if "q-value" in res.columns else ["p-value"]
                            ss["prof_sig_col"] = st.radio("Significance metric", metric_opts, horizontal=True)
                            ss["prof_alpha_thr"] = st.slider(f"{ss['prof_sig_col']} cutoff", 0.0001, 0.2, 0.05, format="%.4f")
                            if not is_anova:
                                ss["prof_fc_thresh"] = st.slider("|log2FC| threshold", 0.0, 4.0, 1.0)
                            else:
                                ss["prof_fc_thresh"] = 0.0
                                
                            st.markdown("**Group Colors**")
                            active_groups_prof = [g for g in ss.get("global_group_labels", []) if g]
                            default_colors_prof = sns.color_palette("Set2", n_colors=max(1, len(active_groups_prof))).as_hex()
                            
                            prof_palette = {}
                            for idx, grp in enumerate(active_groups_prof):
                                prof_palette[grp] = st.color_picker(f"{grp} Color", default_colors_prof[idx], key=f"prof_color_{grp}")
                            ss["prof_palette"] = prof_palette
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            # Filter the dropdown dynamically 
                            sig_mask = res[ss["prof_sig_col"]] <= ss["prof_alpha_thr"]
                            if "log2FC" in res.columns:
                                sig_mask = sig_mask & (res["log2FC"].abs() >= ss["prof_fc_thresh"])
                            
                            if not gene_col_name:
                                st.warning("Cannot map to a specific protein. Protein identifier column is missing.")
                                st.selectbox("Search for a Protein:", options=[], key="pick_gene")
                            else:
                                ss["prof_plot_mode"] = st.radio(
                                    "Protein profile mode",
                                    ["Single selected protein", "Top significant proteins"],
                                    horizontal=False,
                                    key="prof_plot_mode_ui",
                                    help="Use the original single-protein picker or automatically plot the top significant proteins."
                                )
                                sig_gene_opts = res.loc[sig_mask, gene_col_name].dropna().unique().tolist()
                                all_gene_opts = res[gene_col_name].dropna().unique().tolist()
                                opts = sorted(sig_gene_opts) + ["---"] + sorted(list(set(all_gene_opts) - set(sig_gene_opts)))

                                if ss["prof_plot_mode"] == "Single selected protein":
                                    st.selectbox("Search for a Protein:", options=opts, key="pick_gene")
                                else:
                                    max_prof_genes = max(1, min(len(sig_gene_opts), 50))
                                    ss["prof_top_n_genes"] = st.slider(
                                        "Number of significant proteins to plot",
                                        min_value=1,
                                        max_value=max_prof_genes,
                                        value=min(6, max_prof_genes),
                                        step=1,
                                        key="prof_top_n_genes_ui"
                                    )
                                    st.caption("Top proteins are selected by the chosen significance metric and exported as a multi-page PDF.")

                # --- RIGHT PANEL ---
                with right:
                    def display_sig_stats(mask):
                        sig_rows = res.loc[mask].copy()
                        n_features = int(mask.sum())

                        if gene_col_name and gene_col_name in res.columns:
                            sig_genes = (
                                sig_rows[gene_col_name]
                                .dropna()
                                .astype(str)
                                .replace("nan", np.nan)
                                .dropna()
                                .unique()
                                .tolist()
                            )
                            sig_genes = sorted(sig_genes)
                            n_genes = len(sig_genes)
                            st.success(
                                f"Significant features passing thresholds: **{n_features}** | "
                                f"Unique proteins: **{n_genes}**"
                            )
                        else:
                            sig_genes = []
                            n_genes = 0
                            st.success(f"Significant features passing thresholds: **{n_features}**")

                        if n_features > 0:
                            with st.expander("Export Significant Results", expanded=False):
                                st.caption(
                                    "The download includes all rows currently passing the selected p/q-value and |log2FC| thresholds. "
                                    "The protein list is de-duplicated and can be pasted into pathway tools such as STRING, DAVID, or Enrichr."
                                )

                                c1, c2 = st.columns(2)
                                with c1:
                                    add_bytes_download_button_tp(
                                        label="Download Significant Rows CSV",
                                        data=sig_rows.to_csv(index=False).encode("utf-8"),
                                        filename="significant_total_protein_results.csv",
                                        key=f"save_sig_rows_{ss.get('explore_mode_tp', 'plot')}",
                                        help_text="Save rows passing the current significance thresholds as CSV."
                                    )

                                if sig_genes:
                                    gene_txt = "\n".join(sig_genes)
                                    with c2:
                                        add_bytes_download_button_tp(
                                            label=f"Download Protein List TXT ({n_genes})",
                                            data=gene_txt.encode("utf-8"),
                                            filename="significant_total_protein_list.txt",
                                            key=f"save_sig_genes_{ss.get('explore_mode_tp', 'plot')}",
                                            help_text="Save the de-duplicated significant protein list as TXT."
                                        )
                                    st.code(gene_txt, language="text")

                    if mode is None:
                        st.info("Select a Plot mode on the left to begin.")

                    elif ss.get("show_volcano", False):
                        if "log2FC" not in res.columns:
                            st.warning("Volcano plot requires log2FC data (Pairwise Test).")
                        else:
                            metric = ss["volcano_sig_col"]
                            alpha_thr = ss["volcano_alpha_thr"]
                            fc_thresh = ss["volcano_fc_thresh"]
                            top_n_annot = int(ss.get("volcano_top_n_annot", 0))

                            sig_mask = make_pairwise_sig_mask(res, metric, alpha_thr, fc_thresh)
                            display_sig_stats(sig_mask)

                            volcano_plot_type = ss.get("volcano_plot_type", "Interactive")

                            if volcano_plot_type == "Interactive":
                                control_label, test_label = get_pairwise_group_labels()
                                fig = create_interactive_volcano_plot_tp(
                                    res=res,
                                    sig_mask=sig_mask,
                                    gene_col_name=gene_col_name,
                                    metric=metric,
                                    alpha_thr=alpha_thr,
                                    fc_thresh=fc_thresh,
                                    control_label=control_label,
                                    test_label=test_label,
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                add_plotly_html_download_button_tp(
                                    fig,
                                    "total_protein_volcano_interactive",
                                    "volcano_interactive"
                                )
                            else:
                                control_label, test_label = get_pairwise_group_labels()

                                static_volcano_fig = create_static_volcano_plot_tp(
                                    res=res,
                                    gene_col_name=gene_col_name,
                                    metric=metric,
                                    alpha_thr=alpha_thr,
                                    fc_thresh=fc_thresh,
                                    top_n_annot=top_n_annot,
                                    control_label=control_label,
                                    test_label=test_label,
                                )
                                st.pyplot(static_volcano_fig, use_container_width=True)
                                ss["report_volcano_fig"] = static_volcano_fig
                                add_figure_download_buttons_tp(
                                    static_volcano_fig,
                                    "total_protein_volcano_annotated",
                                    "volcano_static",
                                    include_pdf=True,
                                )

                    elif ss.get("show_waterfall", False):
                        if "log2FC" not in res.columns:
                            st.warning("Waterfall plot requires log2FC data (Pairwise Test).")
                        else:
                            metric = ss["waterfall_sig_col"]
                            alpha_thr = ss["waterfall_alpha_thr"]
                            fc_thresh = ss["waterfall_fc_thresh"]
                            top_n = int(ss.get("waterfall_top_n", 10))

                            sig_mask = (res[metric] <= alpha_thr) & (res["log2FC"].abs() >= fc_thresh)
                            display_sig_stats(sig_mask)

                            wf_fig, wf_df, wf_label_df = create_total_protein_waterfall_plot_tp(
                                res=res,
                                gene_col_name=gene_col_name,
                                accession_col_name=accession_col_name,
                                metric=metric,
                                alpha_thr=alpha_thr,
                                fc_thresh=fc_thresh,
                                top_n=top_n,
                                
                            )

                            st.pyplot(wf_fig, use_container_width=True)

                            png_buf = BytesIO()
                            wf_fig.savefig(png_buf, format="png", dpi=300, bbox_inches="tight")
                            png_buf.seek(0)

                            svg_buf = BytesIO()
                            wf_fig.savefig(svg_buf, format="svg", bbox_inches="tight")
                            svg_buf.seek(0)

                            if not wf_label_df.empty:
                                display_cols = ["Rank"]
                                if gene_col_name and gene_col_name in wf_label_df.columns:
                                    display_cols.append(gene_col_name)
                                if accession_col_name and accession_col_name in wf_label_df.columns:
                                    display_cols.append(accession_col_name)
                                for c in ["log2FC", metric, "Status"]:
                                    if c in wf_label_df.columns:
                                        display_cols.append(c)
                                st.markdown("##### Numbered proteins in waterfall")
                                numbered_wf_table = wf_label_df[display_cols].copy()
                                st.dataframe(
                                    numbered_wf_table,
                                    use_container_width=True,
                                    hide_index=True,
                                )

                                add_bytes_download_button_tp(
                                    label="Download Numbered Waterfall Table CSV",
                                    data=numbered_wf_table.to_csv(index=False).encode("utf-8"),
                                    filename="total_protein_waterfall_numbered_table.csv",
                                    key="save_waterfall_numbered_table_csv",
                                    help_text="Save the numbered waterfall table as CSV."
                                )

                            add_figure_download_buttons_tp(
                                wf_fig,
                                "total_protein_waterfall",
                                "waterfall_static",
                                include_pdf=True,
                            )


                    elif ss.get("show_ma", False):
                        if "log2FC" not in res.columns:
                            st.warning("MA Plot requires log2FC data.")
                        else:
                            metric = ss["ma_sig_col"]
                            alpha_thr = ss["ma_alpha_thr"]
                            fc_thresh = ss["ma_fc_thresh"]
                            top_n_ma_annot = ss["ma_top_n"]

                            sig_mask = make_pairwise_sig_mask(res, metric, alpha_thr, fc_thresh)
                            display_sig_stats(sig_mask)

                            mean_cols = [lbl for lbl in ss["global_group_labels"] if lbl in res.columns]

                            if len(mean_cols) >= 2:
                                ma_plot_type = ss.get("ma_plot_type", "Interactive")

                                control_label, test_label = get_pairwise_group_labels()

                                if ma_plot_type == "Interactive":
                                    fig = create_interactive_ma_plot_tp(
                                        res=res,
                                        sig_mask=sig_mask,
                                        gene_col_name=gene_col_name,
                                        metric=metric,
                                        fc_thresh=fc_thresh,
                                        mean_cols=mean_cols,
                                        control_label=control_label,
                                        test_label=test_label,
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    add_plotly_html_download_button_tp(
                                        fig,
                                        "ma_plot_interactive",
                                        "ma_interactive"
                                    )

                                else:
                                    fig, _plot_df = create_static_ma_plot_tp(
                                        res=res,
                                        gene_col_name=gene_col_name,
                                        metric=metric,
                                        alpha_thr=alpha_thr,
                                        fc_thresh=fc_thresh,
                                        mean_cols=mean_cols,
                                        top_n_annot=top_n_ma_annot,
                                        control_label=control_label,
                                        test_label=test_label,
                                    )
                                    st.pyplot(fig, use_container_width=True)
                                    ss["report_ma_fig"] = fig
                                    add_figure_download_buttons_tp(fig, "ma_plot", "ma_static", include_pdf=True)
                            else:
                                st.warning("Insufficient mean columns found for MA Plot.")

                    elif ss.get("show_heatmap", False):
                        metric = ss["hm_sig_col"]
                        alpha_thr = ss["hm_alpha_thr"]
                        fc_thresh = ss["hm_fc_thresh"]
                        top_n_hm = ss["hm_top_n"]
                        normalize_heatmap = ss["hm_norm"]
                        cluster_option = ss["hm_cluster"]

                        sig_mask = res[metric] <= alpha_thr

                        if not is_anova:
                            sig_mask = sig_mask & (res["log2FC"].abs() >= fc_thresh)

                        display_sig_stats(sig_mask)

                        if sig_mask.sum() < 2:
                            st.warning("Heatmap requires at least 2 significant features to perform hierarchical clustering.")
                        else:
                            heat_idx = res[sig_mask].nsmallest(top_n_hm, metric)["RowID"]
                            hm_data = expr.loc[heat_idx]

                            if normalize_heatmap == "Z-score":
                                df_data = (
                                    hm_data.sub(hm_data.mean(axis=1), axis=0)
                                    .div(hm_data.std(axis=1), axis=0)
                                    .fillna(0)
                                )
                            else:
                                df_data = hm_data.copy().fillna(0)

                            # Remove RowID or any index/column axis label from heatmap
                            df_data.index.name = None
                            df_data.columns.name = None

                            mapping_dict = {}

                            for grp_label, grp_cols in zip(ss["global_group_labels"], ss["global_group_samples"]):
                                for c in grp_cols:
                                    mapping_dict[pretty_labels[c]] = grp_label

                            sample_groups = pd.Series(df_data.columns).map(mapping_dict)
                            unique_groups = [g for g in ss["global_group_labels"] if g in sample_groups.values]

                            fallback_colors = {
                                g: c for g, c in zip(
                                    unique_groups,
                                    sns.color_palette("Set2", len(unique_groups)).as_hex()
                                )
                            }

                            group_palette = ss.get("diff_hm_palette", fallback_colors)
                            col_colors = sample_groups.map(group_palette).values

                            row_cluster = False
                            col_cluster = False

                            if cluster_option == "K-Means":
                                row_cluster = True
                                col_cluster = True
                            elif cluster_option == "Hierarchical":
                                row_cluster = True
                                col_cluster = True

                            n_rows = df_data.shape[0]
                            fig_height = max(10, n_rows * 0.4)

                            fig = sns.clustermap(
                                df_data,
                                figsize=(10, fig_height),
                                cmap="vlag" if normalize_heatmap == "Z-score" else "YlGnBu",
                                center=0 if normalize_heatmap == "Z-score" else None,
                                col_cluster=col_cluster,
                                row_cluster=row_cluster,
                                col_colors=col_colors,
                                yticklabels=True,
                                dendrogram_ratio=(0.15, 0.05) if row_cluster else 0.15,
                                colors_ratio=(0.02, 0.02)
                            )

                            # Remove visible axis labels such as RowID
                            fig.ax_heatmap.set_ylabel("")
                            fig.ax_heatmap.set_xlabel("")

                            if gene_col_name:
                                rowid_to_gene = res.set_index("RowID")[gene_col_name].to_dict()
                                new_labels = [
                                    str(rowid_to_gene.get(t.get_text(), t.get_text()))
                                    for t in fig.ax_heatmap.get_yticklabels()
                                ]

                                fig.ax_heatmap.set_yticklabels(
                                    new_labels,
                                    fontsize=10,
                                    fontweight="bold",
                                    rotation=0
                                )
                            else:
                                plt.setp(
                                    fig.ax_heatmap.get_yticklabels(),
                                    rotation=0,
                                    fontsize=10,
                                    fontweight="bold"
                                )

                            plt.setp(
                                fig.ax_heatmap.get_xticklabels(),
                                rotation=45,
                                horizontalalignment="right",
                                fontsize=10,
                                fontweight="bold"
                            )

                            cbar_label = "Z-score" if normalize_heatmap == "Z-score" else "Abundance"

                            fig.cax.set_ylabel(
                                cbar_label,
                                fontsize=12,
                                fontweight="bold"
                            )

                            handles = [
                                mpatches.Patch(color=group_palette[g], label=g)
                                for g in unique_groups
                                if g in group_palette
                            ]

                            fig.ax_col_dendrogram.legend(
                                handles=handles,
                                title="Group",
                                loc="center left",
                                bbox_to_anchor=(1.02, 0.5),
                                frameon=True
                            )

                            st.pyplot(fig.fig)

                            ss["report_heatmap_fig"] = fig.fig
                            add_plot_download_buttons(fig.fig, "differential_heatmap")

                    elif ss.get("show_profile", False):
                        metric = ss["prof_sig_col"]
                        alpha_thr = ss["prof_alpha_thr"]
                        fc_thresh = ss["prof_fc_thresh"]

                        sig_mask = res[metric] <= alpha_thr
                        if not is_anova:
                            sig_mask = sig_mask & (res["log2FC"].abs() >= fc_thresh)

                        display_sig_stats(sig_mask)

                        if not gene_col_name:
                            st.warning("Cannot draw protein profiles because the protein identifier column is missing.")
                        else:
                            profile_mode = ss.get("prof_plot_mode", "Single selected protein")
                            palette = ss.get("prof_palette", None)

                            if profile_mode == "Top significant proteins":
                                sig_profile_df = res.loc[sig_mask].copy()
                                if sig_profile_df.empty:
                                    st.warning("No significant proteins pass the selected thresholds.")
                                else:
                                    top_n_genes = int(ss.get("prof_top_n_genes", 6))
                                    top_gene_names = (
                                        sig_profile_df.dropna(subset=[gene_col_name])
                                        .sort_values(metric)
                                        [gene_col_name]
                                        .astype(str)
                                        .drop_duplicates()
                                        .head(top_n_genes)
                                        .tolist()
                                    )

                                    if not top_gene_names:
                                        st.warning("No protein names are available for the selected significant rows.")
                                    else:
                                        st.markdown(f"##### Protein profiles for top {len(top_gene_names)} significant proteins")
                                        profile_figs = []
                                        profile_gene_names = []

                                        for gene_name in top_gene_names:
                                            fig = create_protein_profile_figure_tp(
                                                res=res,
                                                expr=expr,
                                                gene_col_name=gene_col_name,
                                                gene_name=gene_name,
                                                group_labels=ss["global_group_labels"],
                                                group_samples=ss["global_group_samples"],
                                                pretty_labels=pretty_labels,
                                                palette=palette,
                                                annotate_stats=True,
                                                sig_metric=metric,
                                            )
                                            if fig is not None:
                                                profile_figs.append(fig)
                                                profile_gene_names.append(gene_name)

                                        if profile_figs:
                                            grid_fig = create_profile_grid_figure_tp(
                                                figures=profile_figs,
                                                gene_names=profile_gene_names,
                                                n_cols=4,
                                                panel_width=4.2,
                                                panel_height=3.8,
                                                dpi=220,
                                            )
                                            if grid_fig is not None:
                                                st.pyplot(grid_fig, use_container_width=True)
                                                ss["report_boxplot_fig"] = grid_fig
                                            else:
                                                ss["report_boxplot_fig"] = profile_figs[0]
                                            try:
                                                profile_pdf_bytes = build_profile_pdf_bytes_tp(
                                                    profile_figs,
                                                    profile_gene_names,
                                                    n_cols=5,
                                                    n_rows=2,
                                                )
                                                add_bytes_download_button_tp(
                                                    label="Download Protein Profiles PDF",
                                                    data=profile_pdf_bytes,
                                                    filename="top_protein_profiles.pdf",
                                                    key="download_top_profile_pdf",
                                                    help_text="Download all displayed protein profile plots as a multi-page PDF.",
                                                    type="secondary",
                                                )
                                            except Exception as e:
                                                st.error(f"Could not prepare protein profile PDF: {e}")
                                        else:
                                            st.warning("No profile plots could be generated for the selected proteins.")

                            else:
                                pick_gene = ss.get("pick_gene")
                                if pick_gene and pick_gene != "---":
                                    fig = create_protein_profile_figure_tp(
                                        res=res,
                                        expr=expr,
                                        gene_col_name=gene_col_name,
                                        gene_name=pick_gene,
                                        group_labels=ss["global_group_labels"],
                                        group_samples=ss["global_group_samples"],
                                        pretty_labels=pretty_labels,
                                        palette=palette,
                                        annotate_stats=True,
                                        sig_metric=metric,
                                    )

                                    if fig is None:
                                        st.warning("No expression values found for the selected protein.")
                                    else:
                                        st.pyplot(fig, use_container_width=True)
                                        ss["report_boxplot_fig"] = fig
                                        add_figure_download_buttons_tp(
                                            fig,
                                            f"protein_profile_{pick_gene}",
                                            f"protein_profile_{safe_filename(pick_gene)}",
                                            include_pdf=True,
                                        )


    # =========================================================================
    # PILLAR 4: RESULTS DOWNLOAD
    # =========================================================================
    if ss.get("show_report", False):
        with st.container(border=True):
            col_info, col_action = st.columns([0.65, 0.35], gap="large")

            with col_info:
                st.markdown(
                    "<p class='subtitle-text'>Generate a document summarizing your entire analysis workflow. "
                    "The compiled report includes:</p>", 
                    unsafe_allow_html=True
                )
                
                st.markdown(
                    """
                    - **Experimental Design:** Cohort definitions and sample mapping.
                    - **Data Integrity:** PCA, correlation matrices.
                    - **Differential Analysis:** Volcano plots, heatmaps, and statistical results.
                    - **Export Bundle:** A ZIP archive containing all report images and tabular outputs.
                    """
                )

            with col_action:
                st.write("")
                st.write("")

                def _fig_to_png_bytes(fig):
                    buf = BytesIO()
                    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                    buf.seek(0)
                    return buf.getvalue()

                def _add_fig_to_zip(zf, folder, name, fig):
                    try:
                        png_bytes = _fig_to_png_bytes(fig)
                        zf.writestr(f"{folder}/{name}.png", png_bytes)
                    finally:
                        try:
                            plt.close(fig)
                        except Exception:
                            pass

                def _build_report_zip_tp():
                    zip_buffer = BytesIO()

                    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                        if ss.get("final_pdf_bytes") is not None:
                            zf.writestr(
                                "report/final_report.pdf",
                                _coerce_to_bytes_for_downloads(ss["final_pdf_bytes"]),
                            )

                        if ss.get("raw_df") is not None and not ss["raw_df"].empty:
                            zf.writestr("tables/raw_data.csv", ss["raw_df"].to_csv(index=False))

                        if ss.get("expr_processed") is not None and not ss["expr_processed"].empty:
                            zf.writestr("tables/processed_matrix.csv", ss["expr_processed"].to_csv(index=True))

                        if ss.get("diff_results") is not None and not ss["diff_results"].empty:
                            zf.writestr("tables/differential_results.csv", ss["diff_results"].to_csv(index=False))

                        if ss.get("filtered_df") is not None and not ss["filtered_df"].empty:
                            zf.writestr("tables/filtered_results.csv", ss["filtered_df"].to_csv(index=False))

                        if (
                            ss.get("global_sample_cols")
                            and ss.get("global_group_samples") is not None
                            and ss.get("global_group_labels") is not None
                        ):
                            rows = []
                            group_samples = ss.get("global_group_samples", [])
                            group_labels = ss.get("global_group_labels", [])
                            control_idx = ss.get("global_control_group_idx", 0)

                            for col in ss.get("global_sample_cols", []):
                                grp_name = "Unassigned"
                                grp_num = None
                                for i, cols in enumerate(group_samples):
                                    if col in (cols or []):
                                        grp_name = group_labels[i] if i < len(group_labels) else f"Group{i+1}"
                                        grp_num = i + 1
                                        break
                                        
                                batch_name = "Unassigned"
                                if ss.get("enable_batch_correction"):
                                    for b_name, b_cols in ss.get("batch_dict", {}).items():
                                        if col in b_cols:
                                            batch_name = b_name
                                            break
                                else:
                                    batch_name = "N/A"

                                rows.append({
                                    "Column": col,
                                    "Sample": pretty_labels.get(col, col), 
                                    "Group": grp_name,
                                    "Group_Number": grp_num,
                                    "Batch": batch_name,
                                    "Is_Reference": (grp_num is not None and (grp_num - 1) == control_idx)
                                })

                            if rows:
                                sample_map_df = pd.DataFrame(rows)
                                zf.writestr("tables/sample_metadata_mapping.csv", sample_map_df.to_csv(index=False))

                        plot_keys = [
                            "report_pca_fig", "report_corr_fig", "report_density_raw_fig", "report_density_proc_fig",
                            "report_qc_boxplot_fig", "report_qc_missing_fig", "report_qc_heatmap_fig", 
                            "report_dendrogram_fig", "report_volcano_fig", "report_ma_fig", 
                            "report_log2fc_density_fig", "report_heatmap_fig", "report_boxplot_fig"
                        ]

                        for plot_key in plot_keys:
                            if plot_key in ss and ss[plot_key] is not None:
                                safe_name = plot_key.replace("report_", "").replace("_fig", "")
                                _add_fig_to_zip(zf, "plots", safe_name, ss[plot_key])

                        manifest_lines = [
                            "Total Proteomics Report Bundle",
                            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            "",
                            "Contents:",
                            "- report/final_report.pdf (if generated)",
                            "- tables/*.csv",
                            "- plots/*.png",
                        ]
                        zf.writestr("README.txt", "\n".join(manifest_lines))

                    zip_buffer.seek(0)
                    return zip_buffer.getvalue()

                if "final_pdf_bytes" not in ss:
                    if st.button("Compile PDF Report & ZIP Bundle", use_container_width=True, type="primary", help="Generate a PDF document and ZIP bundle containing all currently selected plots and parameters."):
                        with st.spinner("Compiling plots and generating PDF... this may take a moment."):
                            try:
                                from tp.report_utility import generate_final_pdf

                                generated_pdf = generate_final_pdf()
                                ss["final_pdf_bytes"] = _coerce_to_bytes_for_downloads(generated_pdf)
                                ss["final_report_zip_bytes"] = _build_report_zip_tp()
                                ss["final_report_timestamp"] = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                                st.rerun()

                            except Exception as e:
                                st.error(f"Failed to generate report: {e}")
                
                else:
                    st.success("Report generated successfully.")
                    
                    if "final_report_timestamp" not in ss:
                        ss["final_report_timestamp"] = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                    timestamp = ss["final_report_timestamp"]

                    pdf_bytes = _coerce_to_bytes_for_downloads(ss.get("final_pdf_bytes"))
                    st.caption(f"PDF size: {len(pdf_bytes) / 1024 / 1024:.2f} MB")

                    if "final_report_zip_bytes" not in ss or ss.get("final_report_zip_bytes") is None:
                        try:
                            ss["final_report_zip_bytes"] = _build_report_zip_tp()
                        except Exception as e:
                            ss["final_report_zip_bytes"] = None
                            st.warning(f"PDF is ready, but ZIP bundle could not be prepared: {e}")

                    zip_bytes = ss.get("final_report_zip_bytes")
                    if zip_bytes is not None:
                        zip_bytes = _coerce_to_bytes_for_downloads(zip_bytes)
                        st.caption(f"ZIP size: {len(zip_bytes) / 1024 / 1024:.2f} MB")

                    dl_col1, dl_col2 = st.columns(2)
                    with dl_col1:
                        add_bytes_download_button_tp(
                            label="Download PDF Report",
                            data=pdf_bytes,
                            filename=f"TotalProteomics_Report_{timestamp}.pdf",
                            key="save_final_report_pdf",
                            help_text="Download the compiled PDF report through the browser.",
                            type="primary",
                        )

                    with dl_col2:
                        if zip_bytes is not None:
                            add_bytes_download_button_tp(
                                label="Download ZIP Bundle",
                                data=zip_bytes,
                                filename=f"TotalProteomics_Report_Bundle_{timestamp}.zip",
                                key="save_final_report_zip",
                                help_text="Download the PDF, images, and tabular outputs as one ZIP archive.",
                                type="secondary",
                            )
                        else:
                            st.warning("ZIP bundle is not available. Regenerate the report.")

                    if st.button(
                        "Regenerate Report",
                        use_container_width=True,
                        help="Discard the current report and generate a new one with updated data and plots.",
                        key="regenerate_final_report",
                    ):
                        ss.pop("final_pdf_bytes", None)
                        ss.pop("final_report_zip_bytes", None)
                        ss.pop("final_report_timestamp", None)
                        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    faq_section()


# =============================================================================
# Optional shared download imports
# =============================================================================
# Import utility-only names, but keep this app's browser-safe implementations.
_tp_local_download_functions = {
    "add_hosted_safe_file_download_button": add_hosted_safe_file_download_button,
    "add_bytes_download_button_tp": add_bytes_download_button_tp,
    "add_figure_download_buttons_tp": add_figure_download_buttons_tp,
    "add_plotly_html_download_button_tp": add_plotly_html_download_button_tp,
    "save_profile_pdf_to_downloads_tp": save_profile_pdf_to_downloads_tp,
    "_save_bytes_to_downloads": _save_bytes_to_downloads,
    "_save_file_to_downloads": _save_file_to_downloads,
}

try:
    from download_utility import (
        add_download_button,
        add_file_download_buttons,
        add_bytes_download_button,
        add_dataframe_download_buttons,
        add_matplotlib_download_buttons,
        add_plot_download_buttons,
        add_plotly_download_buttons,
        add_figure_download_buttons_tp_lf,
        add_plotly_html_download_button_tp_lf,
        add_figure_download_buttons_common,
        add_plotly_html_download_button_common,
        save_profile_pdf_to_downloads_tp_lf,
        explain_download_location,
    )
except Exception:
    pass

# Do not allow a shared helper to replace the tested Total Proteomics download path.
globals().update(_tp_local_download_functions)

if __name__ == "__main__":
    if "session_initialized_tp" not in st.session_state:
        _reset_all_data(clear_cache=True)
        st.session_state["session_initialized_tp"] = True

    main()