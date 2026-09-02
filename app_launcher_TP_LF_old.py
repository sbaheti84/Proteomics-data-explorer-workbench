#!/usr/bin/env python3
"""
################################################################################
Total Proteome (Label-Free) Analysis Workflow
################################################################################
"""
__author__ = ("Saurabh Baheti: baheti.saurabh@mayo.edu")
import os
import io
import re
import uuid
import shutil
import subprocess
import tempfile
import sys
import numpy as np
import pandas as pd
import streamlit as st
import zipfile       
import datetime      
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib.patches as mpatches
from pathlib import Path
# Global plotting configurations to guarantee no grids
sns.set_style("white", {"axes.grid": False})
mpl.rcParams['axes.grid'] = False
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
    from faq.faq_TP_LF import faq_section
except ImportError:
    try:
        from faq.faq_TP import faq_section
    except ImportError:
        def faq_section():
            pass
# -----------------------------------------------------------------------------
# Streamlit Setup & CSS Injection
# -----------------------------------------------------------------------------
try_set_page_config(page_title="Total Proteomics LFQ | Mayo Clinic", layout="wide")
inject_custom_css()
ss = st.session_state


def _runtime_bundle_root_lfq() -> Path:
    """Return the directory that contains bundled runtime resources."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent


def get_bundled_r_environment() -> tuple[str, dict]:
    """
    Locate Rscript and construct the environment used by the LFQ R backend.

    Search order:
      1. RSCRIPT_PATH environment override.
      2. R runtime bundled by PyInstaller under <bundle>/R.
      3. R_HOME supplied by the host environment.
      4. Rscript available on PATH.
      5. Common macOS/Linux/Windows installation locations.

    For a frozen Windows build, the function also points R at the bundled
    package library under <bundle>/R_library.
    """
    bundle_root = _runtime_bundle_root_lfq()
    bundled_library = bundle_root / "R_library"

    candidates = []

    explicit_rscript = os.environ.get("RSCRIPT_PATH")
    if explicit_rscript:
        candidates.append((Path(explicit_rscript), False))

    # PyInstaller/standalone runtime layout.
    candidates.extend([
        (bundle_root / "R" / "bin" / "Rscript.exe", True),
        (bundle_root / "R" / "bin" / "x64" / "Rscript.exe", True),
        (bundle_root / "R" / "bin" / "Rscript", True),
    ])

    # Host-provided R_HOME.
    host_r_home = os.environ.get("R_HOME")
    if host_r_home:
        host_r_home_path = Path(host_r_home)
        candidates.extend([
            (host_r_home_path / "bin" / "Rscript.exe", False),
            (host_r_home_path / "bin" / "x64" / "Rscript.exe", False),
            (host_r_home_path / "bin" / "Rscript", False),
        ])

    # Normal PATH lookup.
    path_rscript = shutil.which("Rscript") or shutil.which("Rscript.exe")
    if path_rscript:
        candidates.append((Path(path_rscript), False))

    # Common non-PATH locations.
    candidates.extend([
        (Path("/Library/Frameworks/R.framework/Resources/bin/Rscript"), False),
        (Path("/opt/homebrew/bin/Rscript"), False),
        (Path("/usr/local/bin/Rscript"), False),
        (Path("/usr/bin/Rscript"), False),
        (Path(r"C:\Program Files\R\R-4.5.1\bin\Rscript.exe"), False),
        (Path(r"C:\Program Files\R\R-4.5.1\bin\x64\Rscript.exe"), False),
    ])

    rscript = None
    using_bundled_r = False
    for candidate, is_bundled in candidates:
        if candidate and candidate.exists():
            rscript = candidate.resolve()
            using_bundled_r = is_bundled
            break

    if rscript is None:
        raise RuntimeError(
            "Rscript was not found. For the standalone Windows build, make sure the "
            "PyInstaller bundle contains R/bin/Rscript.exe and R_library. For a normal "
            "installation, install R or set RSCRIPT_PATH/R_HOME."
        )

    env = os.environ.copy()

    # For the bundled runtime we know the R installation layout exactly.
    if using_bundled_r:
        if rscript.parent.name.lower() == "x64":
            r_home = rscript.parent.parent.parent
        else:
            r_home = rscript.parent.parent

        env["R_HOME"] = str(r_home)

        r_bin = r_home / "bin"
        r_x64 = r_bin / "x64"
        path_parts = [str(r_x64), str(r_bin), env.get("PATH", "")]
        env["PATH"] = os.pathsep.join([p for p in path_parts if p])

        if bundled_library.exists():
            # Keep the packaged project library first while R still retains its
            # base/recommended library under R_HOME.
            env["R_LIBS_USER"] = str(bundled_library)
            env["R_LIBS"] = str(bundled_library)
    else:
        # Do not infer R_HOME from /usr/bin/Rscript or similar symlinks. If the
        # host supplied R_HOME, preserve it; otherwise let R discover its own home.
        if host_r_home:
            env["R_HOME"] = str(Path(host_r_home).resolve())

        # If a packaged library happens to be present, expose it without forcing
        # a bundled R installation.
        if bundled_library.exists():
            env["R_LIBS_USER"] = str(bundled_library)
            env["R_LIBS"] = str(bundled_library)

        rscript_dir = str(rscript.parent)
        env["PATH"] = os.pathsep.join([rscript_dir, env.get("PATH", "")])

    return str(rscript), env

def _get_downloads_dir():
    """Return a safe Downloads folder path for macOS/Windows/Linux."""
    downloads = Path.home() / "Downloads"
    if downloads.exists():
        return downloads
    return Path.home()


def _safe_download_filename(filename):
    """Clean filename and avoid characters that can cause issues on macOS/Windows/Linux."""
    filename = str(filename).strip()
    filename = re.sub(r"[^A-Za-z0-9._-]+", "_", filename)
    filename = filename.strip("_")
    return filename or "downloaded_file"


def _save_bytes_to_downloads(data, filename):
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
def _reset_all_data(clear_cache: bool = False):
    """Clear all derived data so the suite cannot run on stale state."""
    ss["global_sample_cols"] = []
    ss["global_group_samples"] = []
    ss["global_group_labels"] = []
    ss["global_n_groups"] = 2
    ss["global_control_group_idx"] = 0
    ss["global_all_groups"] = []
    
    ss["global_batch_samples"] = []
    ss["global_batch_labels"] = []
    ss["global_n_batches"] = 1
    ss["enable_batch_correction"] = False
    ss["batch_method_ui"] = "None"
    ss["batch_correction_applied"] = False
    ss["batch_correction_applied_method"] = "None"
    ss["batch_correction_run_signature"] = None
    ss["batch_correction_last_error"] = None

    # Clear temporary batch UI widget state so Reset Session starts clean.
    ss.pop("_enable_batch_correction_widget", None)
    ss.pop("batch_method_ui_sel", None)
    ss.pop("num_batches_ui", None)
    for _k in list(ss.keys()):
        if str(_k).startswith("batch_sel_"):
            ss.pop(_k, None)

    ss["raw_df"] = None
    ss["expr_raw"] = None
    ss["expr_normalized"] = None
    ss["expr_imputed"] = None
    ss["expr_pre_batch"] = None
    ss["expr_processed"] = None
    ss["expr_final"] = None
    ss["preprocessing_applied"] = False
    ss["preprocessing_run_signature"] = None
    ss["imputation_applied_method"] = "None"
    ss["normalization_applied_method"] = "None"
    ss["preprocessing_dirty"] = False
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
        "global_batch_samples": [],
        "global_batch_labels": [],
        "global_n_batches": 1,
        "enable_batch_correction": False,
        "batch_method_ui": "None",
        "batch_correction_applied": False,
        "batch_correction_applied_method": "None",
        "batch_correction_run_signature": None,
        "batch_correction_last_error": None,
        "expr_raw": None,
        "expr_normalized": None,
        "expr_imputed": None,
        "expr_pre_batch": None,
        "expr_processed": None,
        "expr_final": None,
        "preprocessing_applied": False,
        "preprocessing_run_signature": None,
        "imputation_applied_method": "None",
        "normalization_applied_method": "None",
        "preprocessing_dirty": False,
        "pre_treat_zero": True,
        "pre_log2": True,
        "pre_presence_filter": True,
        "pre_presence_min_frac": 0.40,
        "pre_presence_rule": "or",
        "pre_normalization_method": "Median centering (log2)",
        "pre_imputation_family": "All methods",
        "pre_imputation_method": "K-nearest neighbor (KNN)",
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
def clean_lfq_sample(col: str) -> str:
    if col is None: return col
    s = str(col)
    s = re.sub(r"(?i)lfq\s*intensity\s*", "", s)
    s = re.sub(r"(?i)intensity\s*", "", s)
    s = re.sub(r"(?i)abundance\s*:\s*", "", s)
    s = s.replace(":", "").strip()
    return s.strip()
def apply_batch_correction(
    df: pd.DataFrame,
    batch_list: list[list[str]],
    method: str,
    biological_groups: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Apply technical batch correction to an already preprocessed log2 matrix.

    For ComBat, biological group labels are passed as a categorical covariate so
    the biological condition is preserved while estimating technical batch effects.
    """
    out = df.copy()
    ss["batch_correction_last_error"] = None

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
                out[valid] = (
                    out[valid]
                    .sub(b_mean, axis=0)
                    .div(b_std, axis=0)
                    .mul(global_std, axis=0)
                    .add(global_mean, axis=0)
                )
            elif len(valid) == 1:
                b_mean = out[valid].mean(axis=1)
                out[valid] = out[valid].sub(b_mean, axis=0).add(global_mean, axis=0)

    elif method == "ComBat (Empirical Bayes)":
        if not HAS_COMBAT:
            msg = "The 'neuroCombat' package is not installed. Run `pip install neuroCombat` to use ComBat."
            ss["batch_correction_last_error"] = msg
            st.error(msg)
            return out

        sample_to_batch = {}
        for idx, b_cols in enumerate(batch_list):
            for c in b_cols:
                if c in out.columns:
                    sample_to_batch[c] = idx + 1

        valid_cols = list(sample_to_batch.keys())
        if not valid_cols:
            return out

        dat = out[valid_cols].copy()
        if dat.isna().any().any():
            msg = (
                "ComBat requires a complete matrix in this workflow. Missing values remain after "
                "imputation; choose an imputation method that completes the matrix before running ComBat."
            )
            ss["batch_correction_last_error"] = msg
            st.error(msg)
            return out

        covars_dict = {"batch": [sample_to_batch[c] for c in valid_cols]}
        categorical_cols = []
        if biological_groups:
            group_values = [biological_groups.get(c, "Unknown") for c in valid_cols]
            if len(set(group_values)) > 1:
                covars_dict["biological_group"] = group_values
                categorical_cols = ["biological_group"]

        covars = pd.DataFrame(covars_dict, index=valid_cols)

        try:
            combat_res = neuroCombat(
                dat=dat,
                covars=covars,
                batch_col="batch",
                categorical_cols=categorical_cols,
            )
            corrected_df = pd.DataFrame(combat_res["data"], index=dat.index, columns=valid_cols)
            out[valid_cols] = corrected_df
        except Exception as e:
            msg = (
                f"ComBat correction failed: {e}. Check batch sizes and whether biological group "
                "is completely confounded with batch."
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
# LFQ normalization and NAguideR-compatible R imputation bridge
# -----------------------------------------------------------------------------
NAGUIDER_METHODS = [
    {"label": "Zero", "key": "zero", "family": "Single value", "package": None, "speed": "Fast"},
    {"label": "Minimum", "key": "minimum", "family": "Single value", "package": None, "speed": "Fast"},
    {"label": "Column median", "key": "colmedian", "family": "Single value", "package": "e1071", "speed": "Fast"},
    {"label": "Row median", "key": "rowmedian", "family": "Single value", "package": "e1071", "speed": "Fast"},
    {"label": "Deterministic minimum (MinDet)", "key": "mindet", "family": "Single value", "package": "imputeLCMD", "speed": "Fast"},
    {"label": "Stochastic minimum (MinProb)", "key": "minprob", "family": "Single value", "package": "imputeLCMD", "speed": "Fast"},
    {"label": "Perseus imputation (PI)", "key": "pi", "family": "Single value", "package": None, "speed": "Fast"},
    {"label": "Singular value decomposition (SVD)", "key": "svdmethod", "family": "Global structure", "package": "pcaMethods", "speed": "Medium"},
    {"label": "Maximum likelihood estimation (MLE)", "key": "mle", "family": "Global structure", "package": "norm", "speed": "Medium"},
    {"label": "Sequential imputation (ImpSeq)", "key": "impseq", "family": "Global structure", "package": "rrcovNA", "speed": "Medium"},
    {"label": "Robust sequential imputation (ImpSeqRob)", "key": "impseqrob", "family": "Global structure", "package": "rrcovNA", "speed": "Medium"},
    {"label": "Bayesian PCA (BPCA)", "key": "bpca", "family": "Global structure", "package": "pcaMethods", "speed": "Slow"},
    {"label": "K-nearest neighbor (KNN)", "key": "knnmethod", "family": "Local similarity", "package": "impute", "speed": "Medium"},
    {"label": "Sequential KNN (Seq-KNN)", "key": "seqknn", "family": "Local similarity", "package": "SeqKnn", "speed": "Medium"},
    {"label": "Quantile regression (QRILC)", "key": "qrilc", "family": "Local similarity", "package": "imputeLCMD", "speed": "Medium"},
    {"label": "Local least squares (LLS)", "key": "lls", "family": "Local similarity", "package": "pcaMethods", "speed": "Medium"},
    {"label": "Glmnet ridge regression (GRR)", "key": "grr", "family": "Local similarity", "package": "DreamAI", "speed": "Medium"},
    {"label": "MICE Bayesian linear regression (mice-norm)", "key": "mice-norm", "family": "Local similarity", "package": "mice", "speed": "Slow"},
    {"label": "Truncation KNN (trKNN)", "key": "trknn", "family": "Local similarity", "package": "NAguideR", "speed": "Slow"},
    {"label": "Iterative robust model (IRM)", "key": "irm", "family": "Local similarity", "package": "VIM", "speed": "Slow"},
    {"label": "Generalized Mass Spectrum (GMS)", "key": "gms", "family": "Local similarity", "package": "GMSimpute", "speed": "Slow"},
    {"label": "MICE classification/regression trees (mice-cart)", "key": "mice-cart", "family": "Local similarity", "package": "mice", "speed": "Slow"},
    {"label": "Random forest (RF)", "key": "rf", "family": "Local similarity", "package": "missForest", "speed": "Slow"},
]
NAGUIDER_BY_LABEL = {m["label"]: m for m in NAGUIDER_METHODS}

# -----------------------------------------------------------------------------
# TEMPORARY RCONNECT COMPATIBILITY MODE
# -----------------------------------------------------------------------------
# Keep the NAguideR method selector and R bridge in the application, but while
# R execution is unavailable on Posit/RStudio Connect, use the historical LFQ
# imputation rule: missing values = 1/5 of the smallest positive abundance for
# that protein. Set this flag to False to restore the selected NAguideR method.
TEMP_USE_LEGACY_ONE_FIFTH_IMPUTATION = True
LEGACY_ONE_FIFTH_FRACTION = 0.20


def normalize_lfq_log2(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Normalize a log2 LFQ matrix while preserving missing values."""
    out = df.copy().astype(float)
    if not method or method == "None":
        return out

    if method == "Median centering (log2)":
        centers = out.median(axis=0, skipna=True)
        target = float(np.nanmedian(centers.to_numpy(dtype=float)))
        return out.subtract(centers, axis=1).add(target)

    if method == "Mean centering (log2)":
        centers = out.mean(axis=0, skipna=True)
        target = float(np.nanmean(centers.to_numpy(dtype=float)))
        return out.subtract(centers, axis=1).add(target)

    raise ValueError(f"Unsupported normalization method: {method}")


def temporary_legacy_one_fifth_imputation_lfq(
    raw_df: pd.DataFrame,
    transformed_df: pd.DataFrame,
    normalized_df: pd.DataFrame,
    frac: float = LEGACY_ONE_FIFTH_FRACTION,
    log2_enabled: bool = True,
) -> pd.DataFrame:
    """
    Temporarily reproduce the historical LFQ imputation rule without R.

    For each protein (row), every missing/non-positive value is replaced by
    ``frac * smallest positive observed abundance``.  The replacement is
    calculated on the original abundance scale.

    When log2 preprocessing is enabled, the fill value is converted to log2 and
    then placed on the *existing normalized scale* using the same per-sample
    offset already applied to observed values.  This avoids the incorrect
    operation of multiplying a log2 abundance by 0.2 and also avoids changing
    the normalized observed values.
    """
    raw = raw_df.copy().astype(float)
    transformed = transformed_df.copy().astype(float)
    out = normalized_df.copy().astype(float)

    # Historical rule: row-wise minimum strictly positive abundance / 5.
    positive = raw.where(np.isfinite(raw) & (raw > 0))
    row_min_positive = positive.min(axis=1, skipna=True)
    fill_raw = row_min_positive * float(frac)

    if log2_enabled:
        fill_base = np.log2(fill_raw.where(fill_raw > 0))

        # Normalization in this workflow is a per-column additive shift on the
        # log2 scale. Recover that shift from observed values and apply it only
        # to the newly imputed entries.
        for col in out.columns:
            observed = transformed[col].notna() & out[col].notna()
            if observed.any():
                offset = float((out.loc[observed, col] - transformed.loc[observed, col]).median())
            else:
                offset = 0.0

            missing = out[col].isna() & fill_base.notna()
            if missing.any():
                out.loc[missing, col] = fill_base.loc[missing] + offset
    else:
        for col in out.columns:
            missing = out[col].isna() & fill_raw.notna()
            if missing.any():
                out.loc[missing, col] = fill_raw.loc[missing]

    return out


def _find_rscript_lfq() -> Optional[str]:
    """Return the Rscript executable used by the LFQ backend."""
    try:
        rscript, _ = get_bundled_r_environment()
        return rscript
    except Exception:
        return None


def _find_naguider_bridge_script_lfq() -> Optional[Path]:
    """Locate naguider_impute.R in source, deployed, or frozen-app layouts."""
    bundle_root = _runtime_bundle_root_lfq()
    source_root = Path(__file__).resolve().parent

    candidates = [
        bundle_root / "naguider_impute.R",
        bundle_root / "assets" / "r" / "naguider_impute.R",
        source_root / "naguider_impute.R",
        source_root / "assets" / "r" / "naguider_impute.R",
        Path.cwd() / "naguider_impute.R",
        Path.cwd() / "assets" / "r" / "naguider_impute.R",
    ]

    seen = set()
    for candidate in candidates:
        candidate = Path(candidate)
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate.resolve()
    return None


def check_naguider_r_environment_lfq(
    method_label: Optional[str] = None,
) -> tuple[bool, str]:
    """Check Rscript, the bridge script, and the R package required by a method."""
    try:
        rscript, r_env = get_bundled_r_environment()
    except Exception as e:
        return False, str(e)

    bridge = _find_naguider_bridge_script_lfq()
    if bridge is None:
        return (
            False,
            "naguider_impute.R was not found. The standalone build must bundle it "
            "at the application root (PyInstaller --add-data).",
        )

    method = NAGUIDER_BY_LABEL.get(method_label or "")
    pkgs = []
    if method and method.get("package"):
        pkgs.append(method["package"])
    if method and method.get("key") == "trknn" and "NAguideR" not in pkgs:
        pkgs.append("NAguideR")

    checks = [
        'cat("R_VERSION=", as.character(getRversion()), "\\n", sep="")',
        'cat("R_HOME=", R.home(), "\\n", sep="")',
        'cat("R_LIBS=", paste(.libPaths(), collapse="|"), "\\n", sep="")',
    ]
    checks.extend(
        [
            f'if (!requireNamespace("{p}", quietly=TRUE)) stop("MISSING:{p}")'
            for p in pkgs
        ]
    )
    r_expr = "; ".join(checks)

    try:
        res = subprocess.run(
            [rscript, "--vanilla", "-e", r_expr],
            capture_output=True,
            text=True,
            timeout=90,
            env=r_env,
        )
    except Exception as e:
        return False, f"Could not start Rscript: {e}"

    if res.returncode != 0:
        detail = (res.stderr or res.stdout or "Unknown R error").strip()
        return False, (
            "R is available, but the selected NAguideR method is not ready: "
            + detail[-2500:]
        )

    stdout = (res.stdout or "").strip().replace("\n", " | ")
    pkg_text = ", ".join(pkgs) if pkgs else "base R only"
    return True, (
        f"R backend ready ({pkg_text}). Rscript: {rscript}. "
        f"Bridge: {bridge}. {stdout}"
    )


def run_naguider_imputation_lfq(
    df: pd.DataFrame,
    method_label: str,
    pi_width: float = 0.3,
    pi_downshift: float = 1.8,
    rf_ntree: int = 20,
) -> pd.DataFrame:
    """Run one of NAguideR's 23 imputation methods through an Rscript subprocess."""
    if method_label == "None (leave missing values)":
        return df.copy()

    method = NAGUIDER_BY_LABEL.get(method_label)
    if method is None:
        raise ValueError(f"Unknown NAguideR method: {method_label}")

    try:
        rscript, r_env = get_bundled_r_environment()
    except Exception as e:
        raise RuntimeError(str(e)) from e

    bridge = _find_naguider_bridge_script_lfq()
    if bridge is None:
        raise RuntimeError(
            "naguider_impute.R was not found. In the standalone build, bundle it with "
            "`--add-data \"$ROOT\\naguider_impute.R;.\"`."
        )

    work = df.copy().astype(float)

    with tempfile.TemporaryDirectory(prefix="lfq_naguider_") as tmpdir:
        in_path = Path(tmpdir) / "input.tsv"
        out_path = Path(tmpdir) / "output.tsv"

        work.to_csv(
            in_path,
            sep="\t",
            index=True,
            index_label="RowID",
            na_rep="NA",
        )

        cmd = [
            rscript,
            "--vanilla",
            str(bridge),
            str(in_path),
            str(out_path),
            method["key"],
            str(float(pi_width)),
            str(float(pi_downshift)),
            str(int(rf_ntree)),
        ]

        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,
                env=r_env,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"NAguideR method '{method_label}' exceeded the 60-minute timeout."
            ) from e
        except Exception as e:
            raise RuntimeError(f"Could not launch the NAguideR R backend: {e}") from e

        if res.returncode != 0:
            detail = (res.stderr or res.stdout or "Unknown R error").strip()
            raise RuntimeError(
                f"NAguideR method '{method_label}' failed.\n{detail[-4000:]}"
            )

        if not out_path.exists():
            raise RuntimeError(
                "R completed without creating the imputed output matrix."
            )

        out = pd.read_csv(out_path, sep="\t", index_col=0)
        out.index = out.index.astype(str)
        expected_index = work.index.astype(str)
        out = out.reindex(index=expected_index, columns=work.columns)
        out.index = work.index
        out = out.apply(pd.to_numeric, errors="coerce")

    if out.shape != work.shape:
        raise RuntimeError(
            f"Imputed matrix shape changed from {work.shape} to {out.shape}."
        )

    return out

def _group_map_pretty_lfq(pretty_labels: dict) -> dict:
    """Map pretty sample names to biological group labels for ComBat preservation."""
    mapping = {}
    for label, cols in zip(ss.get("global_group_labels", []), ss.get("global_group_samples", [])):
        for col in cols:
            mapping[pretty_labels.get(col, col)] = label
    return mapping

# -----------------------------------------------------------------------------
# Differential Engine 
# -----------------------------------------------------------------------------
def compute_diffexp_tp_lf(
    df: pd.DataFrame,
    control_cols: list[str],
    test_cols: list[str],
    control_label: str = "Control",
    test_label: str = "Test",
    test_type: str = "Student's t-test (Independent, Equal Variance)",
    do_zero_impute: bool = False,
    zero_frac: float = 0.2,
    do_presence_filter: bool = True,
    presence_min_frac: float = 0.40,
    presence_rule: str = "or",
    min_n_per_group: int = 2,
    batch_samples: list[list[str]] = None,
    batch_method: str = "None"
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
            df=df, all_samples=all_cols, group_samples=[ctrl, test], 
            min_frac=presence_min_frac, presence_rule=presence_rule
        )
    if do_zero_impute:
        df = replace_zeros_rowwise_min_fraction(
            df=df, sample_cols=all_cols, frac=zero_frac, replace_nan=True
        )
    for c in all_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # --- BATCH CORRECTION ---
    if batch_method and batch_method != "None" and batch_samples:
        tmp_vals = df[all_cols].astype(float)
        tmp_vals[tmp_vals <= 0] = np.nan
        tmp_log = np.log2(tmp_vals)
        
        corrected_log = apply_batch_correction(tmp_log, batch_samples, batch_method)
        corrected_raw = (2 ** corrected_log).fillna(0)
        df[all_cols] = corrected_raw
    df[control_label] = df[ctrl].mean(axis=1)
    df[test_label] = df[test].mean(axis=1)
    
    pseudo = 0.0
    # Input abundances are already log2-transformed by the global preprocessing pipeline.
    # Therefore log2 fold-change is the difference between group mean log2 abundances.
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
def compute_diffexp_anova_tp_lf(
    df: pd.DataFrame,
    group1_cols: list[str], group2_cols: list[str], group3_cols: list[str],
    group1_label: str = "Group1", group2_label: str = "Group2", group3_label: str = "Group3",
    do_zero_impute: bool = False,
    zero_frac: float = 0.2,
    do_presence_filter: bool = True,
    presence_min_frac: float = 0.40,
    presence_rule: str = "or",
    batch_samples: list[list[str]] = None,
    batch_method: str = "None"
) -> pd.DataFrame:
    df = df.copy()
    g1 = [c for c in group1_cols if c in df.columns]
    g2 = [c for c in group2_cols if c in df.columns]
    g3 = [c for c in group3_cols if c in df.columns]
    all_cols = list(dict.fromkeys(g1 + g2 + g3))
    if not (g1 and g2 and g3):
        raise ValueError("ANOVA requires 3 non-empty groups.")
    for c in all_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if do_presence_filter:
        df = filter_by_presence_rule(
            df=df, all_samples=all_cols, group_samples=[g1, g2, g3], 
            min_frac=presence_min_frac, presence_rule=presence_rule
        )
    if do_zero_impute:
        df = replace_zeros_rowwise_min_fraction(
            df=df, sample_cols=all_cols, frac=zero_frac, replace_nan=True
        )
    for c in all_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # --- BATCH CORRECTION ---
    if batch_method and batch_method != "None" and batch_samples:
        tmp_vals = df[all_cols].astype(float)
        tmp_vals[tmp_vals <= 0] = np.nan
        tmp_log = np.log2(tmp_vals)
        
        corrected_log = apply_batch_correction(tmp_log, batch_samples, batch_method)
        corrected_raw = (2 ** corrected_log).fillna(0)
        df[all_cols] = corrected_raw
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

def create_static_volcano_plot_tp_lf(
    res: pd.DataFrame,
    gene_col_name: Optional[str],
    metric: str,
    alpha_thr: float,
    fc_thresh: float,
    top_n_annot: int = 0,
):
    """Create a static downloadable volcano plot for LFQ total proteomics.

    The interactive Plotly volcano stays clean. This static figure is used only
    for PNG/SVG downloads and can annotate the top N significant proteins.
    """
    plot_df = res.copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df["log2FC"] = pd.to_numeric(plot_df["log2FC"], errors="coerce")
    plot_df = plot_df.dropna(subset=["log2FC", metric]).copy()

    plot_df["-log10(metric)"] = -np.log10(plot_df[metric].clip(lower=1e-300))

    mask_up = (plot_df[metric] <= alpha_thr) & (plot_df["log2FC"] >= fc_thresh)
    mask_down = (plot_df[metric] <= alpha_thr) & (plot_df["log2FC"] <= -fc_thresh)
    mask_sig = mask_up | mask_down
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
        plot_df.loc[mask_down, "log2FC"],
        plot_df.loc[mask_down, "-log10(metric)"],
        color="#1f77b4",
        alpha=0.85,
        s=24,
        label="Down regulated",
    )

    ax.axhline(-np.log10(alpha_thr), color="black", linestyle="--", linewidth=1)
    #ax.axvline(fc_thresh, color="black", linestyle="--", linewidth=1)
    #ax.axvline(-fc_thresh, color="black", linestyle="--", linewidth=1)

    if top_n_annot and top_n_annot > 0 and gene_col_name and gene_col_name in plot_df.columns:
        annot_df = plot_df.loc[mask_sig].copy()
        if not annot_df.empty:
            annot_df["abs_log2FC"] = annot_df["log2FC"].abs()
            annot_df = annot_df.sort_values([metric, "abs_log2FC"], ascending=[True, False]).head(int(top_n_annot))

            texts = []
            x_span = max(float(plot_df["log2FC"].max() - plot_df["log2FC"].min()), 1.0)
            y_span = max(float(plot_df["-log10(metric)"].max() - plot_df["-log10(metric)"].min()), 1.0)

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
                ax.plot([x, x + dx], [y, y + dy], color="gray", linewidth=0.5, alpha=0.7)

            if HAS_ADJUST_TEXT and texts:
                adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.7))

    #ax.set_title(f"Volcano Plot ({metric} < {alpha_thr}, |log2FC| > {fc_thresh})", fontweight="bold")
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
   #ax.set_xlabel("log2 Fold-Change", fontweight="bold")
   #ax.set_ylabel(f"-log10({metric})", fontweight="bold")
    ax.grid(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig

def _figure_to_bytes_tp_lf(fig, fmt: str, dpi: int = 300) -> bytes:
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


def add_figure_download_buttons_tp_lf(fig, base_filename: str, key_suffix: str = "", include_pdf: bool = True):
    """
    Desktop-safe figure export buttons. Files are saved directly to the user's
    Downloads folder, which is more reliable for packaged Streamlit apps.
    """
    if fig is None:
        return

    if not key_suffix:
        key_suffix = str(uuid.uuid4())[:8]

    base_filename = _safe_download_filename(base_filename)
    png_bytes = _figure_to_bytes_tp_lf(fig, "png")
    svg_bytes = _figure_to_bytes_tp_lf(fig, "svg")
    pdf_bytes = _figure_to_bytes_tp_lf(fig, "pdf") if include_pdf else None

    cols = st.columns(3 if include_pdf else 2)

    with cols[0]:
        if st.button(
            "Download PNG",
            key=f"save_png_{base_filename}_{key_suffix}",
            use_container_width=True,
            help="Save PNG directly to your local Downloads folder."
        ):
            try:
                outpath = _save_bytes_to_downloads(png_bytes, f"{base_filename}.png")
                st.success(f"Saved PNG to: {outpath}")
            except Exception as e:
                st.error(f"Could not save PNG: {e}")

    with cols[1]:
        if st.button(
            "Download SVG",
            key=f"save_svg_{base_filename}_{key_suffix}",
            use_container_width=True,
            help="Save SVG directly to your local Downloads folder."
        ):
            try:
                outpath = _save_bytes_to_downloads(svg_bytes, f"{base_filename}.svg")
                st.success(f"Saved SVG to: {outpath}")
            except Exception as e:
                st.error(f"Could not save SVG: {e}")

    if include_pdf:
        with cols[2]:
            if st.button(
                "Download PDF",
                key=f"save_pdf_{base_filename}_{key_suffix}",
                use_container_width=True,
                help="Save PDF directly to your local Downloads folder."
            ):
                try:
                    outpath = _save_bytes_to_downloads(pdf_bytes, f"{base_filename}.pdf")
                    st.success(f"Saved PDF to: {outpath}")
                except Exception as e:
                    st.error(f"Could not save PDF: {e}")


def add_plotly_html_download_button_tp_lf(fig, base_filename: str, key_suffix: str = ""):
    """
    Save an interactive Plotly figure as a self-contained HTML file.
    """
    if fig is None:
        return
    if not key_suffix:
        key_suffix = str(uuid.uuid4())[:8]
    base_filename = _safe_download_filename(base_filename)
    try:
        html_bytes = fig.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8")
    except Exception as e:
        st.warning(f"Could not prepare interactive HTML export: {e}")
        return

    if st.button(
        "Download Interactive HTML",
        key=f"save_html_{base_filename}_{key_suffix}",
        use_container_width=True,
        help="Save the interactive Plotly figure to your local Downloads folder."
    ):
        try:
            outpath = _save_bytes_to_downloads(html_bytes, f"{base_filename}.html")
            st.success(f"Saved HTML to: {outpath}")
        except Exception as e:
            st.error(f"Could not save HTML: {e}")


def create_static_ma_plot_tp_lf(
    res: pd.DataFrame,
    gene_col_name: Optional[str],
    metric: str,
    alpha_thr: float,
    fc_thresh: float,
    mean_cols: list,
    top_n_annot: int = 0,
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
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="black", lw=0.5, alpha=0.6))

    ax.set_title("MA Plot", fontweight="bold")
    ax.set_xlabel("Average Intensity (Log2)", fontweight="bold")
    ax.set_ylabel("Log2 Fold Change", fontweight="bold")
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


def create_protein_profile_figure_tp_lf(
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
    gene_rows = res.loc[res[gene_col_name] == gene_name, "RowID"].dropna().tolist()

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

    # Manual star annotation using existing differential result.
    # This avoids missing/clipped statannotations in top-protein grids.
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
                y_range = ymax - ymin
                y = ymax + 0.08 * y_range
                h = 0.04 * y_range

                # For 2 groups, draw bracket between group 1 and group 2.
                # For 3+ groups, draw bracket from first to last group.
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

    ax.set_ylabel(
        "Normalized Abundance",
        fontweight="bold",
        fontsize=11
    )

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


def build_profile_pdf_bytes_tp_lf(
    figures: list,
    gene_names: list,
    n_cols: int = 5,
    n_rows: int = 2,
) -> bytes:
    """Build the LFQ protein-profile PDF entirely in memory."""
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

    pdf_buffer = BytesIO()
    plots_per_page = max(1, int(n_cols) * int(n_rows))

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


def save_profile_pdf_to_downloads_tp_lf(
    figures: list,
    gene_names: list,
    filename: str = "lfq_protein_profiles.pdf",
    n_cols: int = 5,
    n_rows: int = 2,
):
    """Backward-compatible local save wrapper around the in-memory PDF builder."""
    pdf_bytes = build_profile_pdf_bytes_tp_lf(
        figures=figures,
        gene_names=gene_names,
        n_cols=n_cols,
        n_rows=n_rows,
    )
    return _save_bytes_to_downloads(pdf_bytes, filename)


def create_profile_grid_figure_tp_lf(
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
        # Make sure manual stars/brackets are rendered before converting to image.
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

        # Do not add another title here; each individual plot already has one.

    grid_fig.tight_layout(pad=0.25)
    return grid_fig


def create_total_protein_waterfall_plot_tp_lf(
    res: pd.DataFrame,
    gene_col_name: Optional[str],
    accession_col_name: Optional[str],
    metric: str,
    alpha_thr: float,
    fc_thresh: float,
    top_n: int = 10,
):
    """Create ranked LFQ total-protein waterfall plot with numbered significant labels."""
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
    ax.set_title(f"LFQ Total Protein Waterfall ({metric} < {alpha_thr}, |log2FC| > {fc_thresh})", fontweight="bold")
    ax.set_xlabel("Protein", fontweight="bold")
    ax.set_ylabel("log2 Fold-Change", fontweight="bold")
    ax.grid(False)
    ax.legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig, wfl, label_df

def render_professional_tp_lf_start_panel(upload_disabled: bool = True):
    """Professional first-run landing panel for the Total Proteomics LFQ app."""
    button_text = "Upload data to enable processing" if upload_disabled else "Review LFQ Data Matrix"

    st.markdown(
        """
        <style>
        .tplf-start-wrap { margin-top: 1.4rem; margin-bottom: 1.5rem; }
        .tplf-start-title { font-size: 1.85rem; line-height: 1.15; font-weight: 700; color: #005a9c; margin-bottom: 0.35rem; letter-spacing: -0.01em; }
        .tplf-start-subtitle { font-size: 1.02rem; color: #2f3a45; margin-bottom: 1.6rem; max-width: 980px; line-height: 1.55; }
        .tplf-step-card { border: 1px solid #d9e2ef; border-radius: 10px; padding: 22px 24px; background: #f8fbff; min-height: 138px; box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04); }
        .tplf-step-number { width: 42px; height: 42px; border-radius: 50%; background: #005a9c; color: white; font-weight: 700; display: inline-flex; align-items: center; justify-content: center; margin-right: 14px; font-size: 1rem; vertical-align: middle; }
        .tplf-step-heading { font-size: 1.15rem; font-weight: 700; color: #005a9c; vertical-align: middle; }
        .tplf-step-body { color: #2f3a45; margin-left: 58px; margin-top: 8px; font-size: 0.96rem; line-height: 1.5; }
        .tplf-step-arrow { text-align: center; font-size: 2.2rem; color: #8a94a6; padding-top: 45px; font-weight: 300; }
        .tplf-info-box { background: #eef6ff; border-radius: 8px; padding: 16px 20px; margin-top: 24px; color: #004b8d; font-weight: 700; border: 1px solid #d8eaff; }
        .tplf-info-box-sub { font-weight: 400; margin-top: 6px; color: #263746; }
        .tplf-warning-box { background: #fff8e5; border-radius: 8px; padding: 16px 20px; margin-top: 14px; color: #8a5a00; font-weight: 700; border: 1px solid #f5e4b6; }
        .tplf-warning-box-sub { font-weight: 400; margin-top: 6px; color: #5f4300; }
        .tplf-muted-button { border: 1px solid #d9dfe8; border-radius: 7px; padding: 10px 18px; text-align: center; color: #8a94a6; font-weight: 600; margin-top: 28px; background: #ffffff; }
        .tplf-active-button-note { border: 1px solid #b8d6f2; border-radius: 7px; padding: 10px 18px; text-align: center; color: #005a9c; font-weight: 700; margin-top: 28px; background: #f8fbff; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="tplf-start-wrap">
            <div class="tplf-start-title">Start Here</div>
            <div class="tplf-start-subtitle">
                Begin your label-free total protein analysis by uploading the quantitative LFQ matrix from the left panel.
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
            <div class="tplf-step-card">
                <div>
                    <span class="tplf-step-number">1</span>
                    <span class="tplf-step-heading">Upload LFQ Data</span>
                </div>
                <div class="tplf-step-body">
                    Upload your label-free total proteomics matrix as CSV, TSV, TXT, or Excel using the left sidebar.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with arrow_col:
        st.markdown("<div class='tplf-step-arrow'>›</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            """
            <div class="tplf-step-card">
                <div>
                    <span class="tplf-step-number">2</span>
                    <span class="tplf-step-heading">Review Matrix</span>
                </div>
                <div class="tplf-step-body">
                    Confirm detected LFQ intensity columns, then proceed to group setup, normalization/imputation, QC, optional batch correction, statistics, and reporting.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="tplf-info-box">
            Upload data using the left sidebar
            <div class="tplf-info-box-sub">
                Required input: one LFQ protein matrix containing protein identifiers and raw LFQ abundance/intensity columns.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if upload_disabled:
        st.markdown(
            """
            <div class="tplf-warning-box">
                No data uploaded yet
                <div class="tplf-warning-box-sub">
                    Please upload an LFQ proteome data file to unlock the workflow.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    btn_class = "tplf-muted-button" if upload_disabled else "tplf-active-button-note"
    st.markdown(f"<div class='{btn_class}'>{button_text}</div>", unsafe_allow_html=True)



# -----------------------------------------------------------------------------
# Main-app helper functions
# -----------------------------------------------------------------------------
def _uniq_lf(seq):
    """Return items in original order while removing duplicates."""
    seen = set()
    out = []
    for x in seq or []:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _sanitize_lf(selected, allowed):
    """Keep only selected values that still exist in the allowed option list."""
    allowed_set = set(allowed or [])
    return _uniq_lf([x for x in (selected or []) if x in allowed_set])


def render_sidebar_controls_lf():
    """Render the LFQ sidebar controls and return the uploaded file object."""
    st.sidebar.markdown("### Total Proteomics (LFQ)")
    st.sidebar.caption("v1.0 | Bioinformatics LFQ Workflow")
    st.sidebar.divider()

    st.sidebar.markdown("**Upload Data**")
    file = st.sidebar.file_uploader(
        "Upload LFQ proteome data (CSV or Excel)",
        type=["csv", "xlsx", "tsv", "txt"]
    )

    with st.sidebar.expander("Download Example Dataset", expanded=False):
        st.caption("Use this example file to see the required formatting and test the workflow.")

        script_dir = Path(__file__).resolve().parent
        example_path = script_dir / "assets" / "example" / "TPLF" / "Protein_abundance_LFQ.csv"

        if example_path.exists():
            add_bytes_download_button_tp_lf(
                label="Download Protein_abundance_LFQ.csv",
                data=example_path.read_bytes(),
                filename="Protein_abundance_LFQ_example.csv",
                key="dl_ex_tp_lf",
                help_text="Download a sample LFQ dataset to test the workflow.",
            )
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


def handle_file_change_lf(file):
    """Reset derived state when a different uploaded file is selected."""
    current_sig = f"{file.name}_{file.size}" if file else None
    if current_sig != ss.get("upload_sig"):
        _reset_all_data()
        ss["upload_sig"] = current_sig


def read_lfq_uploaded_file(file) -> pd.DataFrame:
    """Read CSV, TSV/TXT, or Excel LFQ input without changing parsing behavior."""
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    if file.name.endswith((".tsv", ".txt")):
        return pd.read_csv(file, sep="\t")
    return pd.read_excel(file)


def add_gene_id_and_row_id_lf(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Add canonical GeneID and RowID columns used downstream."""
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

    accession_col = next(
        (
            c for c in [
                "Accession",
                "Master Protein Accessions",
                "Protein ID",
                "Protein.IDs"
            ]
            if c in raw_df.columns
        ),
        None
    )

    if "GeneID" in raw_df.columns and accession_col:
        raw_df["RowID"] = raw_df["GeneID"].astype(str) + "_" + raw_df[accession_col].astype(str)
    elif "GeneID" in raw_df.columns:
        raw_df["RowID"] = raw_df["GeneID"].astype(str)
    elif accession_col:
        raw_df["RowID"] = raw_df[accession_col].astype(str)
    else:
        raw_df["RowID"] = "Feature_" + raw_df.index.astype(str)

    return raw_df


def load_raw_lfq_data_lf(file):
    """Load the uploaded LFQ matrix once and cache it in session state."""
    if ss.get("raw_df") is not None:
        return ss["raw_df"]

    try:
        raw_df = read_lfq_uploaded_file(file)
        raw_df = add_gene_id_and_row_id_lf(raw_df)
        ss["raw_df"] = raw_df
        return raw_df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


def detect_lfq_numeric_columns(df: pd.DataFrame) -> list:
    """
    Detect LFQ sample columns using the same loose numeric strategy as the original app.
    Metadata-like columns are ignored before coercing numeric values.
    """
    meta_keywords = r"(?i)^(RowID|Gene|Description|Accession|Protein|Sequence|Peptide|Score|Coverage|FDR|q-value|p-value|#)"

    for col in df.columns:
        if not re.match(meta_keywords, str(col)):
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().sum() > 0:
                df[col] = coerced

    raw_numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c != "RowID"
    ]

    numeric_cols = [
        c for c in raw_numeric_cols
        if not re.search(r"(?i)normalized|grouped|scaled|s/n|corr|ratio|count|cv", str(c))
    ]

    return numeric_cols


def make_pretty_lfq_labels(numeric_cols: list) -> dict:
    """Create unique display labels for LFQ sample columns."""
    pretty_labels = {}
    used_labels = set()

    for col in numeric_cols:
        base_label = clean_lfq_sample(col)

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


def is_downstream_active_lf() -> bool:
    """Return whether any of the four main workflow panels is open."""
    return (
        ss.get("show_global_select")
        or ss.get("show_overview")
        or ss.get("show_suite")
        or ss.get("show_report")
    )


def render_lfq_matrix_panel(df: pd.DataFrame, numeric_cols: list):
    """Render uploaded LFQ matrix preview and original-data downloads."""
    with st.expander(
        "Integrated LFQ Quantitative Proteomics Matrix",
        expanded=not is_downstream_active_lf()
    ):
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Proteins Detected", f"{df.shape[0]:,}")
        m2.metric("Total LFQ Sample Columns", len(numeric_cols))
        m3.metric("Data Format", "LFQ Matrix")

        st.markdown("<br>", unsafe_allow_html=True)

        dl_col1, dl_col2, spacer = st.columns([1.5, 1.5, 7])

        with dl_col1:
            add_bytes_download_button_tp_lf(
                label="Download CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                filename="uploaded_lfq_proteomics_data.csv",
                key="download_uploaded_lfq_csv",
                help_text="Download the uploaded LFQ matrix as CSV.",
            )

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Raw_LFQ_Data")

        with dl_col2:
            add_bytes_download_button_tp_lf(
                label="Download Excel",
                data=buffer.getvalue(),
                filename="uploaded_lfq_proteomics_data.xlsx",
                key="download_uploaded_lfq_excel",
                help_text="Download the uploaded LFQ matrix as Excel.",
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, height=450)

    st.markdown("<br>", unsafe_allow_html=True)


def reset_overview_flags_lf():
    """Clear overview-specific UI state when leaving that panel."""
    for k in ["qc_plot_type", "explore_mode_tp"]:
        if k in ss:
            del ss[k]


def render_navigation_buttons_lf():
    """Render the four main workflow navigation buttons and update panel state."""
    c1, c2, c3, c4 = st.columns(4)

    if c1.button(
        ("▸ " if not ss["show_global_select"] else "▾ ") + "Data Filtering & Group Specification",
        use_container_width=True,
        key="btn_global_select",
        help="Assign samples to biological groups and set a control/baseline."
    ):
        ss["show_global_select"] = not ss["show_global_select"]
        if ss["show_global_select"]:
            ss["show_overview"] = False
            ss["show_suite"] = False
            ss["show_report"] = False
            reset_overview_flags_lf()

    if c2.button(
        ("▸ " if not ss["show_overview"] else "▾ ") + "Preprocessing, QC & Batch Correction",
        use_container_width=True,
        key="btn_overview",
        help="Normalize, impute, review QC, and optionally correct technical batch effects."
    ):
        ss["show_overview"] = not ss["show_overview"]
        if ss["show_overview"]:
            ss["show_suite"] = False
            ss["show_global_select"] = False
            ss["show_report"] = False

    if c3.button(
        ("▸ " if not ss["show_suite"] else "▾ ") + "Statistical Analysis Suite",
        use_container_width=True,
        key="btn_suite",
        help="Perform differential protein abundance testing and visualize results (Volcano, Heatmaps, etc.)."
    ):
        ss["show_suite"] = not ss["show_suite"]
        if ss["show_suite"]:
            ss["show_overview"] = False
            ss["show_global_select"] = False
            ss["show_report"] = False

    if c4.button(
        ("▸ " if not ss["show_report"] else "▾ ") + "Results Download",
        use_container_width=True,
        key="btn_report",
        help="Compile and export a comprehensive PDF report and ZIP bundle of your analysis."
    ):
        ss["show_report"] = not ss["show_report"]
        if ss["show_report"]:
            ss["show_overview"] = False
            ss["show_global_select"] = False
            ss["show_suite"] = False

    st.markdown("<br>", unsafe_allow_html=True)


def get_pairwise_group_labels_lf() -> tuple[str, str]:
    """Return the active test/control labels for pairwise plots."""
    labels = ss.get("global_group_labels", [])
    control_idx = ss.get("global_control_group_idx", 0)

    control_label = labels[control_idx] if 0 <= control_idx < len(labels) else "Control"
    test_idx = 1 if control_idx == 0 else 0
    test_label = labels[test_idx] if 0 <= test_idx < len(labels) else "Test"

    return control_label, test_label


def make_pairwise_sig_mask_lf(res: pd.DataFrame, metric: str, alpha_thr: float, fc_thresh: float) -> pd.Series:
    """Build the common significance mask used by pairwise LFQ plots."""
    return (res[metric] <= alpha_thr) & (res["log2FC"].abs() >= fc_thresh)



# -----------------------------------------------------------------------------
# Shared common-library wrappers
# -----------------------------------------------------------------------------
# These wrappers keep the original LFQ function names used throughout this launcher
# while moving the actual plotting/export implementation into tp_common_utility.py.
def get_pairwise_group_labels(default_control: str = "Control", default_test: str = "Test") -> tuple[str, str]:
    return get_pairwise_group_labels_common(ss, default_control=default_control, default_test=default_test)


def make_pairwise_sig_mask(res: pd.DataFrame, metric: str, alpha_thr: float, fc_thresh: float) -> pd.Series:
    return make_pairwise_sig_mask_common(res, metric, alpha_thr, fc_thresh)


def create_static_volcano_plot_tp_lf(
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


def create_interactive_volcano_plot_tp_lf(
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


def create_static_ma_plot_tp_lf(
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


def create_interactive_ma_plot_tp_lf(
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


def add_figure_download_buttons_tp_lf(fig, base_filename: str, key_suffix: str = "", include_pdf: bool = True):
    return add_figure_download_buttons_common(
        fig=fig,
        base_filename=base_filename,
        key_suffix=key_suffix,
        include_pdf=include_pdf,
        save_to_downloads=True,
    )


def add_plotly_html_download_button_tp_lf(fig, base_filename: str, key_suffix: str = ""):
    return add_plotly_html_download_button_common(
        fig=fig,
        base_filename=base_filename,
        key_suffix=key_suffix,
        save_to_downloads=True,
    )


# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------

# =============================================================================
# Browser-safe and desktop-safe downloads
# =============================================================================
# Posit/RStudio Connect runs Python on the server. Writing to Path.home()/Downloads
# writes inside the server container, not to the user's computer. Browser downloads
# must be supplied as bytes through st.download_button.

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

    for env_name in (
        "TOTALPROTEOMICS_LFQ_HOSTED",
        "TOTAL_PROTEOMICS_LFQ_HOSTED",
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

    runtime_text = " ".join(
        [str(globals().get("__file__", "")), os.getcwd()]
    ).lower()
    return "rstudio-connect" in runtime_text or "/mnt/app" in runtime_text


def _get_downloads_dir():
    import os
    from pathlib import Path

    custom_dir = (
        os.environ.get("TOTALPROTEOMICS_LFQ_DOWNLOAD_DIR", "").strip()
        or os.environ.get("TOTAL_PROTEOMICS_LFQ_DOWNLOAD_DIR", "").strip()
        or os.environ.get("TOTALPROTEOMICS_DOWNLOAD_DIR", "").strip()
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
    if lower.endswith(".tsv"):
        return "text/tab-separated-values"
    if lower.endswith(".txt"):
        return "text/plain"
    if lower.endswith(".xlsx"):
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if lower.endswith(".xls"):
        return "application/vnd.ms-excel"
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
    if hasattr(data, "getvalue"):
        value = data.getvalue()
        return value if isinstance(value, bytes) else bytes(value)
    return bytes(data)


def _save_bytes_to_downloads(data, filename):
    """
    Legacy local-desktop helper. Hosted applications must use browser downloads.
    """
    file_bytes = _coerce_to_bytes_for_downloads(data)
    safe_name = _safe_download_filename(filename)

    if _is_hosted_mode():
        raise RuntimeError(
            "Direct saving is disabled in hosted mode. "
            "Use add_bytes_download_button_tp_lf() instead."
        )

    outdir = _get_downloads_dir()
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / safe_name

    base_outpath = outpath
    counter = 1
    while outpath.exists():
        outpath = base_outpath.with_name(
            f"{base_outpath.stem}_{counter}{base_outpath.suffix}"
        )
        counter += 1

    outpath.write_bytes(file_bytes)
    return outpath


def _save_file_to_downloads(source_path, filename=None):
    source_path = Path(source_path)
    return _save_bytes_to_downloads(
        source_path.read_bytes(),
        filename or source_path.name,
    )


def add_bytes_download_button_tp_lf(
    label: str,
    data,
    filename: str,
    key: str,
    help_text: str = "Download file.",
    type: str = "secondary",
    use_container_width: bool = True,
):
    """Render a browser-native button for CSV/XLSX/TXT/HTML/PDF/ZIP/image bytes."""
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
        use_container_width=use_container_width,
    )
    if clicked:
        st.success(f"Downloaded {safe_name}")
    return clicked


def add_hosted_safe_file_download_button(
    label,
    data,
    file_name,
    key,
    use_container_width=True,
    help_text=None,
    type="secondary",
):
    """Compatibility wrapper that always returns the file through the browser."""
    return add_bytes_download_button_tp_lf(
        label=label,
        data=data,
        filename=file_name,
        key=key,
        help_text=help_text or "Download file.",
        type=type,
        use_container_width=use_container_width,
    )


def add_figure_download_buttons_tp_lf(
    fig,
    base_filename: str,
    key_suffix: str = "",
    include_pdf: bool = True,
):
    """Browser-native PNG/SVG/PDF exports for Matplotlib figures."""
    if fig is None:
        return

    if not key_suffix:
        key_suffix = uuid.uuid4().hex[:8]

    base_filename = _safe_download_filename(base_filename)
    png_bytes = _figure_to_bytes_tp_lf(fig, "png")
    svg_bytes = _figure_to_bytes_tp_lf(fig, "svg")
    pdf_bytes = _figure_to_bytes_tp_lf(fig, "pdf") if include_pdf else None

    cols = st.columns(3 if include_pdf else 2)

    with cols[0]:
        add_bytes_download_button_tp_lf(
            label="Download PNG",
            data=png_bytes,
            filename=f"{base_filename}.png",
            key=f"save_png_{base_filename}_{key_suffix}",
            help_text="Download the figure as a high-resolution PNG.",
        )

    with cols[1]:
        add_bytes_download_button_tp_lf(
            label="Download SVG",
            data=svg_bytes,
            filename=f"{base_filename}.svg",
            key=f"save_svg_{base_filename}_{key_suffix}",
            help_text="Download the figure as a vector SVG.",
        )

    if include_pdf:
        with cols[2]:
            add_bytes_download_button_tp_lf(
                label="Download PDF",
                data=pdf_bytes,
                filename=f"{base_filename}.pdf",
                key=f"save_pdf_{base_filename}_{key_suffix}",
                help_text="Download the figure as a PDF.",
            )


def add_plotly_html_download_button_tp_lf(
    fig,
    base_filename: str,
    key_suffix: str = "",
):
    """Browser-native interactive Plotly HTML export."""
    if fig is None:
        return
    if not key_suffix:
        key_suffix = uuid.uuid4().hex[:8]

    base_filename = _safe_download_filename(base_filename)
    try:
        html_bytes = fig.to_html(
            include_plotlyjs="cdn",
            full_html=True,
        ).encode("utf-8")
    except Exception as e:
        st.warning(f"Could not prepare interactive HTML export: {e}")
        return

    add_bytes_download_button_tp_lf(
        label="Download Interactive HTML",
        data=html_bytes,
        filename=f"{base_filename}.html",
        key=f"save_html_{base_filename}_{key_suffix}",
        help_text="Download the interactive Plotly figure as HTML.",
    )


def main():
    init_session_state_defaults()

    file = render_sidebar_controls_lf()
    handle_file_change_lf(file)

    if not file:
        render_professional_tp_lf_start_panel(upload_disabled=True)
        st.markdown("<br>", unsafe_allow_html=True)
        faq_section()
        return

    # Load data once
    df = load_raw_lfq_data_lf(file)
    if df is None:
        return

    
    numeric_cols = detect_lfq_numeric_columns(df)

    if not numeric_cols:
        st.error("No valid numeric sample columns found. Please ensure your file contains quantitative data.")
        st.stop()

    pretty_labels = make_pretty_lfq_labels(numeric_cols)

    render_lfq_matrix_panel(df, numeric_cols)

    # =========================================================================
    # THE 4-PILLAR NAVIGATION LAYOUT
    # =========================================================================
    render_navigation_buttons_lf()

    # =========================================================================
    # PILLAR 1: DATA FILTERING & Group Specification
    # =========================================================================
    if ss["show_global_select"]:
        _uniq = _uniq_lf
        _sanitize = _sanitize_lf
        left, right = st.columns([0.42, 0.58], gap="large")
        with left:
            with st.container(border=True):
                st.markdown("#### Define Experimental Groups")
                if "global_sample_cols_ui" not in ss:
                    default_samples = ss.get("global_sample_cols") or numeric_cols
                    ss["global_sample_cols_ui"] = _sanitize(default_samples, numeric_cols)
                ss["global_sample_cols_ui"] = _sanitize(ss["global_sample_cols_ui"], numeric_cols)
                sample_cols = st.multiselect(
                    "Active LFQ Samples",
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
                    ss["global_batch_samples"] = []
                    ss["global_batch_labels"] = []
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
                st.divider()
                st.info(
                    "Technical batch correction is configured after normalization, imputation, and QC in "
                    "**Preprocessing, QC & Batch Correction**. This keeps biological group definition separate "
                    "from technical correction."
                )

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
                                
                        batch_name = "N/A"
                        if ss.get("enable_batch_correction", False):
                            batch_name = "Unassigned"
                            for j, b_cols in enumerate(ss.get("global_batch_samples", [])):
                                if c in (b_cols or []):
                                    labels_tmp = ss.get("global_batch_labels", [])
                                    batch_name = labels_tmp[j] if j < len(labels_tmp) else f"Batch {j+1}"
                                    break
                                
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
    # PILLAR 2: PREPROCESSING, QC & OPTIONAL BATCH CORRECTION
    # =========================================================================
    if ss["show_overview"]:
        if not ss["global_all_groups"]:
            st.warning("Please assign groups in Data Filtering & Group Specification first.")
            st.stop()

        pretty_group_samples = [
            [pretty_labels.get(c, c) for c in gcols]
            for gcols in ss["global_group_samples"]
        ]
        all_pretty_samples = [pretty_labels.get(c, c) for c in ss["global_all_groups"]]

        expr_raw_current = (
            df[["RowID"] + ss["global_all_groups"]]
            .groupby("RowID")
            .mean()
            .apply(pd.to_numeric, errors="coerce")
        )
        expr_raw_current.rename(columns=pretty_labels, inplace=True)
        ss["expr_raw"] = expr_raw_current.copy()

        with st.container(border=True):
            st.markdown("#### Step 1 — Normalize and Impute LFQ Matrix")
            st.caption(
                "Preprocessing is performed once here and the resulting matrix is reused by QC, optional batch "
                "correction, and differential analysis. No second hidden imputation is performed during statistics."
            )

            with st.expander("Preprocessing definitions and NAguideR integration", expanded=False):
                st.markdown("""
                - **0 → NA:** zeros in LFQ intensity matrices usually represent non-detection rather than measured zero abundance.
                - **Presence filtering:** removes very sparse proteins before imputation.
                - **Log2:** stabilizes variance and makes differences in group means interpretable as log2 fold-changes.
                - **Normalization:** median or mean centering aligns sample-level distributions on the log2 scale.
                - **NAguideR imputation:** all 23 NAguideR methods are available through an Rscript backend.
                - **Batch correction:** is deliberately performed only after preprocessing and QC.
                """)

            pcol1, pcol2, pcol3 = st.columns(3, gap="large")
            with pcol1:
                pre_zero = st.checkbox(
                    "Treat 0 as missing (NA)",
                    value=bool(ss.get("pre_treat_zero", True)),
                    key="pre_treat_zero",
                )
                pre_presence_filter = st.checkbox(
                    "Filter sparse proteins before imputation",
                    value=bool(ss.get("pre_presence_filter", True)),
                    key="pre_presence_filter",
                )
                if pre_presence_filter:
                    presence_min_frac = st.slider(
                        "Presence threshold",
                        0.05, 1.00,
                        float(ss.get("pre_presence_min_frac", 0.40)),
                        0.05,
                        key="pre_presence_min_frac",
                    )
                    rule_labels = {
                        "OR: total samples OR one group": "or",
                        "Total samples only": "all_only",
                        "At least one group only": "group_only",
                        "AND: total samples AND one group": "and",
                    }
                    current_rule_key = ss.get("pre_presence_rule", "or")
                    default_rule_label = next((k for k, v in rule_labels.items() if v == current_rule_key), list(rule_labels)[0])
                    presence_rule_label = st.selectbox(
                        "Presence rule",
                        list(rule_labels.keys()),
                        index=list(rule_labels.keys()).index(default_rule_label),
                        key="pre_presence_rule_ui",
                    )
                    presence_rule_key = rule_labels[presence_rule_label]
                    ss["pre_presence_rule"] = presence_rule_key
                else:
                    presence_min_frac = float(ss.get("pre_presence_min_frac", 0.40))
                    presence_rule_key = ss.get("pre_presence_rule", "or")

            with pcol2:
                pre_log2 = st.checkbox(
                    "Log2 transform",
                    value=bool(ss.get("pre_log2", True)),
                    key="pre_log2",
                    help="Keep this enabled for differential analysis and batch correction.",
                )
                normalization_options = ["None", "Median centering (log2)", "Mean centering (log2)"]
                previous_norm = ss.get("pre_normalization_method", "Median centering (log2)")
                if previous_norm not in normalization_options:
                    previous_norm = "Median centering (log2)"
                normalization_method = st.selectbox(
                    "Normalization method",
                    normalization_options,
                    index=normalization_options.index(previous_norm),
                    key="pre_normalization_method",
                    disabled=not pre_log2,
                )
                if not pre_log2:
                    normalization_method = "None"
                    st.warning("Normalization and downstream differential analysis are designed for log2 LFQ values.")

            with pcol3:
                family_options = ["All methods", "Single value", "Global structure", "Local similarity"]
                previous_family = ss.get("pre_imputation_family", "All methods")
                if previous_family not in family_options:
                    previous_family = "All methods"
                imputation_family = st.selectbox(
                    "NAguideR method family",
                    family_options,
                    index=family_options.index(previous_family),
                    key="pre_imputation_family",
                )
                filtered_methods = [
                    m["label"] for m in NAGUIDER_METHODS
                    if imputation_family == "All methods" or m["family"] == imputation_family
                ]
                imputation_options = ["None (leave missing values)"] + filtered_methods
                previous_imp = ss.get("pre_imputation_method", "K-nearest neighbor (KNN)")
                if previous_imp not in imputation_options:
                    previous_imp = imputation_options[1] if len(imputation_options) > 1 else imputation_options[0]
                imputation_method = st.selectbox(
                    "Missing-value imputation",
                    imputation_options,
                    index=imputation_options.index(previous_imp),
                    key="pre_imputation_method",
                )
                if imputation_method != "None (leave missing values)":
                    mmeta = NAGUIDER_BY_LABEL[imputation_method]
                    st.caption(
                        f"Family: {mmeta['family']} · Speed: {mmeta['speed']} · "
                        f"R package: {mmeta['package'] or 'base R / NAguideR logic'}"
                    )
                    if TEMP_USE_LEGACY_ONE_FIFTH_IMPUTATION:
                        st.info(
                            "Temporary RConnect mode: the selected NAguideR method is kept in the UI, "
                            "but execution currently uses the historical 1/5 row-minimum imputation. "
                            "No R backend is required for preprocessing."
                        )

            pi_width = 0.3
            pi_downshift = 1.8
            rf_ntree = 20
            if imputation_method == "Perseus imputation (PI)":
                acol1, acol2 = st.columns(2)
                with acol1:
                    pi_width = st.number_input("PI width", min_value=0.01, max_value=2.0, value=0.3, step=0.05)
                with acol2:
                    pi_downshift = st.number_input("PI downshift", min_value=0.0, max_value=5.0, value=1.8, step=0.1)
            elif imputation_method == "Random forest (RF)":
                rf_ntree = st.number_input("RF number of trees", min_value=10, max_value=1000, value=20, step=10)

            setup_col, run_col = st.columns([0.35, 0.65])
            with setup_col:
                if st.button("Check R / NAguideR Setup", use_container_width=True, key="check_naguider_r_lfq"):
                    ok, msg = check_naguider_r_environment_lfq(
                        None if imputation_method == "None (leave missing values)" else imputation_method
                    )
                    (st.success if ok else st.error)(msg)
            with run_col:
                run_preprocessing = st.button(
                    "Run Normalization + Imputation",
                    type="primary",
                    use_container_width=True,
                    key="run_preprocessing_lfq",
                )

            preprocessing_signature = (
                tuple(ss.get("global_all_groups", [])),
                bool(pre_zero), bool(pre_presence_filter), float(presence_min_frac), str(presence_rule_key),
                bool(pre_log2), str(normalization_method), str(imputation_method),
                float(pi_width), float(pi_downshift), int(rf_ntree),
            )

            if run_preprocessing:
                try:
                    work = expr_raw_current.copy()
                    if pre_zero:
                        work = work.replace(0, np.nan)

                    if pre_presence_filter:
                        before_n = work.shape[0]
                        work = filter_by_presence_rule(
                            df=work,
                            all_samples=all_pretty_samples,
                            group_samples=pretty_group_samples,
                            min_frac=presence_min_frac,
                            presence_rule=presence_rule_key,
                        )
                        removed_n = before_n - work.shape[0]
                    else:
                        removed_n = 0

                    # Keep a copy on the original abundance scale so the temporary
                    # legacy imputation can reproduce the historical 1/5 rule correctly.
                    raw_for_imputation = work.copy()

                    if pre_log2:
                        nonpositive = int(((work <= 0) & work.notna()).sum().sum())
                        if nonpositive:
                            st.warning(f"{nonpositive:,} non-positive observed values were converted to NA before log2 transformation.")
                        work = work.where(work > 0)
                        work = np.log2(work)

                    transformed = work.copy()
                    normalized = normalize_lfq_log2(work, normalization_method if pre_log2 else "None")
                    missing_before = int(normalized.isna().sum().sum())

                    if imputation_method == "None (leave missing values)":
                        imputed = normalized.copy()
                        applied_imputation_method = "None (leave missing values)"
                    elif TEMP_USE_LEGACY_ONE_FIFTH_IMPUTATION:
                        with st.spinner("Running temporary legacy 1/5 minimum imputation ..."):
                            imputed = temporary_legacy_one_fifth_imputation_lfq(
                                raw_df=raw_for_imputation,
                                transformed_df=transformed,
                                normalized_df=normalized,
                                frac=LEGACY_ONE_FIFTH_FRACTION,
                                log2_enabled=bool(pre_log2),
                            )
                        applied_imputation_method = (
                            f"Legacy 1/5 row minimum (temporary; selected: {imputation_method})"
                        )
                    else:
                        with st.spinner(f"Running NAguideR method: {imputation_method} ..."):
                            imputed = run_naguider_imputation_lfq(
                                normalized,
                                method_label=imputation_method,
                                pi_width=pi_width,
                                pi_downshift=pi_downshift,
                                rf_ntree=int(rf_ntree),
                            )
                        applied_imputation_method = imputation_method
                    missing_after = int(imputed.isna().sum().sum())

                    ss["expr_normalized"] = normalized.copy()
                    ss["expr_imputed"] = imputed.copy()
                    ss["expr_pre_batch"] = imputed.copy()
                    ss["expr_processed"] = imputed.copy()
                    ss["expr_final"] = imputed.copy()
                    ss["preprocessing_applied"] = True
                    ss["preprocessing_run_signature"] = preprocessing_signature
                    ss["imputation_applied_method"] = applied_imputation_method
                    ss["normalization_applied_method"] = normalization_method if pre_log2 else "None"
                    ss["preprocessing_dirty"] = False
                    ss["batch_correction_applied"] = False
                    ss["batch_correction_applied_method"] = "None"
                    ss["batch_correction_run_signature"] = None
                    ss["diff_results"] = None
                    ss["filtered_df"] = None
                    ss["filtered_ready"] = False

                    st.success(
                        f"Preprocessing complete: {imputed.shape[0]:,} proteins retained; "
                        f"{removed_n:,} sparse proteins removed; missing values {missing_before:,} → {missing_after:,}."
                    )
                except Exception as e:
                    st.error(f"Preprocessing failed: {e}")

            if ss.get("preprocessing_applied", False):
                if ss.get("preprocessing_run_signature") != preprocessing_signature:
                    ss["preprocessing_dirty"] = True
                    st.warning("Preprocessing settings have changed. Click **Run Normalization + Imputation** to update the processed matrix before statistics.")
                else:
                    ss["preprocessing_dirty"] = False
                    st.success(
                        f"Current matrix: {ss.get('normalization_applied_method', 'None')} + "
                        f"{ss.get('imputation_applied_method', 'None')}"
                    )

        # ------------------------------------------------------------------
        # QC after normalization/imputation, with raw and before/after views.
        # ------------------------------------------------------------------
        st.divider()
        st.markdown("#### Step 2 — Quality Control")
        design_rows = []
        for i, cols in enumerate(ss["global_group_samples"]):
            for c in cols:
                design_rows.append({"Sample": pretty_labels[c], "Group": ss["global_group_labels"][i]})
        design_df = pd.DataFrame(design_rows)

        qc_sources = {"Raw selected abundances": expr_raw_current}
        if isinstance(ss.get("expr_normalized"), pd.DataFrame):
            qc_sources["Normalized (before imputation)"] = ss["expr_normalized"]
        if isinstance(ss.get("expr_imputed"), pd.DataFrame):
            qc_sources["Imputed (before batch correction)"] = ss["expr_imputed"]
        if ss.get("batch_correction_applied", False) and isinstance(ss.get("expr_processed"), pd.DataFrame):
            qc_sources["After batch correction"] = ss["expr_processed"]

        left_qc, right_qc = st.columns([0.35, 0.65], gap="large")
        with left_qc:
            with st.container(border=True):
                source_names = list(qc_sources.keys())
                current_source = ss.get("qc_data_source_lfq", source_names[-1])
                if current_source not in source_names:
                    current_source = source_names[-1]
                    ss["qc_data_source_lfq"] = current_source
                qc_source_name = st.selectbox(
                    "Data Source",
                    source_names,
                    index=source_names.index(current_source),
                    key="qc_data_source_lfq",
                )
                qc_target = qc_sources[qc_source_name]

                qc_mode = st.radio(
                    "Select QC Plot",
                    ["PCA", "Density", "Correlation Heatmap", "QC Heatmap", "Boxplot", "Missing Values", "Dendrogram"],
                    key="qc_plot_type",
                )

                topn = 50
                if qc_mode == "PCA":
                    st.divider()
                    st.markdown("**PCA Display Options**")
                    n_comps = max(2, min(10, qc_target.shape[0], qc_target.shape[1]))
                    pc_cols = [f"PC{i+1}" for i in range(n_comps)]
                    col_x, col_y = st.columns(2)
                    with col_x:
                        ss["pca_x"] = st.selectbox("X-axis Component", pc_cols, index=0, key="pca_x_ui")
                    with col_y:
                        ss["pca_y"] = st.selectbox("Y-axis Component", pc_cols, index=1 if len(pc_cols) > 1 else 0, key="pca_y_ui")
                    colA, colB = st.columns(2)
                    with colA:
                        ss["pca_hide_outliers"] = st.checkbox("Hide outliers", value=False, key="pca_hide_outliers_ui")
                        ss["pca_draw_ellipses"] = st.checkbox("Draw ellipses", value=True, key="pca_draw_ellipses_ui")
                    with colB:
                        ss["pca_label_outliers"] = st.checkbox("Label outliers", value=True, key="pca_label_outliers_ui")
                    ss["pca_zoom_q"] = st.slider("Zoom (quantile)", 0.90, 1.0, 1.0, 0.01, key="pca_zoom_q_ui")

                    unique_groups = design_df["Group"].unique()
                    default_colors = sns.color_palette("Set2", n_colors=max(1, len(unique_groups))).as_hex()
                    pca_palette = {}
                    if len(unique_groups):
                        st.markdown("**Group Colors**")
                        color_cols = st.columns(len(unique_groups))
                        for idx, grp in enumerate(unique_groups):
                            with color_cols[idx]:
                                pca_palette[grp] = st.color_picker(grp, default_colors[idx], key=f"pca_color_{grp}_ui")
                    ss["pca_palette"] = pca_palette

                elif qc_mode == "QC Heatmap":
                    topn = st.slider("Top N variable proteins", 10, 200, 50, key="qc_topn_proc")
                    unique_groups = design_df["Group"].unique()
                    default_colors = sns.color_palette("Set2", n_colors=max(1, len(unique_groups))).as_hex()
                    qc_hm_palette = {}
                    for idx, grp in enumerate(unique_groups):
                        qc_hm_palette[grp] = st.color_picker(
                            f"Color for {grp}", default_colors[idx], key=f"qc_hm_color_{grp}"
                        )
                    ss["qc_hm_palette"] = qc_hm_palette

        with right_qc:
            qc_target = qc_sources[qc_source_name]
            if qc_mode == "PCA":
                pca_input = qc_target.copy().replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="all")
                # PCA needs a complete matrix. Fill only for the QC calculation; the stored matrix is unchanged.
                pca_input = pca_input.apply(lambda r: r.fillna(r.median()), axis=1).dropna(axis=0, how="any")
                if pca_input.shape[0] < 2 or pca_input.shape[1] < 2:
                    st.warning("Not enough complete information to calculate PCA for this data source.")
                else:
                    create_pca_plot(
                        df=pca_input,
                        group_samples=pretty_group_samples,
                        group_labels=ss["global_group_labels"],
                        title=f"PCA — {qc_source_name}",
                        pc_x=ss.get("pca_x", "PC1"),
                        pc_y=ss.get("pca_y", "PC2"),
                        hide_outliers=ss.get("pca_hide_outliers", False),
                        label_outliers=ss.get("pca_label_outliers", True),
                        draw_ellipses=ss.get("pca_draw_ellipses", False),
                        zoom_q=ss.get("pca_zoom_q", 1.0),
                        palette=ss.get("pca_palette", None),
                    )

            elif qc_mode == "Density":
                fig, ax = plt.subplots(figsize=(7, 5))
                for sample in qc_target.columns:
                    vals = qc_target[sample].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(vals) >= 2:
                        sns.kdeplot(vals, ax=ax, label=sample, lw=1.5, alpha=0.6)
                ax.set_title(f"Sample LFQ Density — {qc_source_name}")
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
                st.pyplot(fig)
                ss["report_density_proc_fig"] = fig
                add_plot_download_buttons(fig, "density_plot")

            elif qc_mode == "Correlation Heatmap":
                fig, ax = plt.subplots(figsize=(7, 6))
                sns.heatmap(qc_target.corr(), cmap="vlag", ax=ax, cbar_kws={"shrink": 0.5})
                ax.set_title(qc_source_name)
                st.pyplot(fig)
                ss["report_corr_fig"] = fig
                add_plot_download_buttons(fig, "correlation_heatmap")

            elif qc_mode == "QC Heatmap":
                var = qc_target.var(axis=1, skipna=True)
                top_idx = var.nlargest(topn).index
                hm = qc_target.loc[top_idx].copy().replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="all")
                z = hm.sub(hm.mean(axis=1), axis=0).div(hm.std(axis=1), axis=0)
                z = z.apply(lambda r: r.fillna(r.median()), axis=1).fillna(0.0)
                if not np.isfinite(z.to_numpy()).all():
                    st.error("QC heatmap contains non-finite values.")
                else:
                    sample_to_group = design_df.set_index("Sample")["Group"]
                    unique_groups = design_df["Group"].unique()
                    fallback_colors = {
                        g: c for g, c in zip(unique_groups, sns.color_palette("Set2", len(unique_groups)).as_hex())
                    }
                    group_palette = ss.get("qc_hm_palette", fallback_colors)
                    col_colors = pd.Series(z.columns).map(sample_to_group).map(group_palette).values
                    fig = sns.clustermap(
                        z, cmap="vlag", col_colors=col_colors, figsize=(10, 10),
                        xticklabels=True, yticklabels=True, tree_kws=dict(linewidths=0.5),
                    )
                    plt.setp(fig.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8)
                    plt.setp(fig.ax_heatmap.get_xticklabels(), rotation=45, horizontalalignment="right")
                    handles = [
                        mpatches.Patch(color=group_palette[g], label=g)
                        for g in unique_groups if g in group_palette
                    ]
                    fig.ax_col_dendrogram.legend(
                        handles=handles, title="Group", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True
                    )
                    st.pyplot(fig.fig)
                    ss["report_qc_heatmap_fig"] = fig.fig
                    add_plot_download_buttons(fig.fig, "qc_heatmap")

            elif qc_mode == "Boxplot":
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.boxplot(data=qc_target, ax=ax)
                ax.set_title(qc_source_name)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                st.pyplot(fig)
                ss["report_qc_boxplot_fig"] = fig
                add_plot_download_buttons(fig, "qc_boxplot")

            elif qc_mode == "Missing Values":
                miss = qc_target.isna().mean() * 100
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(x=miss.index, y=miss.values, ax=ax)
                ax.set_title(f"% Missing Values — {qc_source_name}")
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
                    ax.set_title(qc_source_name)
                    plt.xticks(rotation=45, ha="right")
                    st.pyplot(fig)
                    ss["report_dendrogram_fig"] = fig
                    add_plot_download_buttons(fig, "dendrogram")

        # ------------------------------------------------------------------
        # Batch correction comes after normalization/imputation and initial QC.
        # ------------------------------------------------------------------
        st.divider()
        st.markdown("#### Step 3 — Optional Technical Batch Correction")
        st.caption(
            "Use this only for known technical batches/runs. Review the QC above first. "
            "ComBat preserves the biological group as a categorical covariate."
        )

        if not ss.get("preprocessing_applied", False) or not isinstance(ss.get("expr_imputed"), pd.DataFrame):
            st.info("Run normalization + imputation first. Batch correction is intentionally unavailable before preprocessing.")
        elif not ss.get("pre_log2", True):
            st.warning("Batch correction in this workflow requires log2-transformed abundances.")
        else:
            if "_enable_batch_correction_widget" not in ss:
                ss["_enable_batch_correction_widget"] = bool(ss.get("enable_batch_correction", False))
            apply_batch = st.toggle(
                "Apply Batch Effect Correction",
                key="_enable_batch_correction_widget",
                help="Define technical batches only after reviewing normalized/imputed QC.",
            )
            ss["enable_batch_correction"] = bool(apply_batch)

            if not apply_batch:
                ss["batch_correction_applied"] = False
                ss["batch_correction_applied_method"] = "None"
                ss["expr_pre_batch"] = ss["expr_imputed"].copy()
                ss["expr_processed"] = ss["expr_imputed"].copy()
                ss["expr_final"] = ss["expr_imputed"].copy()
                st.info("Batch correction is off. Differential analysis will use the normalized/imputed matrix.")
            else:
                bcol1, bcol2 = st.columns([0.35, 0.65], gap="large")
                with bcol1:
                    batch_methods = [
                        "ComBat (Empirical Bayes)",
                        "Median Centering",
                        "Mean Centering",
                        "Standardize (Mean + Variance)",
                    ]
                    previous_method = ss.get("batch_method_ui", "ComBat (Empirical Bayes)")
                    if previous_method not in batch_methods:
                        previous_method = batch_methods[0]
                    batch_method = st.selectbox(
                        "Batch Correction Method",
                        batch_methods,
                        index=batch_methods.index(previous_method),
                        key="batch_method_ui_sel",
                    )
                    ss["batch_method_ui"] = batch_method
                    previous_n = int(ss.get("global_n_batches", 2) or 2)
                    if previous_n not in [2, 3, 4]:
                        previous_n = 2
                    num_batches = st.radio(
                        "Number of Batches", [2, 3, 4], index=[2, 3, 4].index(previous_n), horizontal=True,
                        key="num_batches_ui",
                    )
                    ss["global_n_batches"] = num_batches
                    if batch_method == "ComBat (Empirical Bayes)":
                        st.info("Biological group labels are included in ComBat and protected from batch removal.")

                with bcol2:
                    batch_allowed_samples = list(ss.get("global_all_groups", []))
                    stored_batch_samples = ss.get("global_batch_samples", []) or []
                    batch_samples = []
                    selected_so_far = []
                    for b in range(num_batches):
                        key = f"batch_sel_{b}"
                        previous = ss.get(key, stored_batch_samples[b] if b < len(stored_batch_samples) else [])
                        previous = _sanitize_lf(previous, batch_allowed_samples)
                        available = [c for c in batch_allowed_samples if c not in selected_so_far or c in previous]
                        previous = _sanitize_lf(previous, available)
                        ss[key] = previous
                        b_cols = st.multiselect(
                            f"Batch {b + 1} Samples",
                            available,
                            format_func=lambda x: pretty_labels.get(x, x),
                            key=key,
                        )
                        b_cols = _sanitize_lf(b_cols, batch_allowed_samples)
                        batch_samples.append(b_cols)
                        selected_so_far = _uniq_lf(selected_so_far + b_cols)

                    assigned = _uniq_lf([c for cols in batch_samples for c in cols])
                    unassigned = [c for c in batch_allowed_samples if c not in assigned]
                    nonempty = [cols for cols in batch_samples if cols]
                    if unassigned:
                        st.warning("Unassigned samples: " + ", ".join(pretty_labels.get(c, c) for c in unassigned))
                    if len(nonempty) < 2:
                        st.warning("At least two non-empty technical batches are required.")

                    ss["global_batch_samples"] = batch_samples
                    ss["global_batch_labels"] = [f"Batch {i+1}" for i in range(num_batches)]

                    translated_batches = [
                        [pretty_labels.get(c, c) for c in cols]
                        for cols in batch_samples
                    ]
                    batch_signature = (
                        ss.get("preprocessing_run_signature"), batch_method, int(num_batches),
                        tuple(tuple(cols) for cols in batch_samples),
                    )

                    # Never keep using a correction generated from old batch assignments/methods.
                    if (
                        ss.get("batch_correction_applied", False)
                        and ss.get("batch_correction_run_signature") != batch_signature
                    ):
                        ss["batch_correction_applied"] = False
                        ss["batch_correction_applied_method"] = "None"
                        ss["batch_correction_run_signature"] = None
                        ss["expr_pre_batch"] = ss["expr_imputed"].copy()
                        ss["expr_processed"] = ss["expr_imputed"].copy()
                        ss["expr_final"] = ss["expr_imputed"].copy()
                        ss["diff_results"] = None
                        st.warning("Batch settings changed. The previous correction was removed; run batch correction again.")

                    can_run_batch = not unassigned and len(nonempty) >= 2
                    if st.button(
                        "Run Batch Correction",
                        type="primary",
                        disabled=not can_run_batch,
                        use_container_width=True,
                        key="run_batch_correction_btn_lfq",
                    ):
                        corrected = apply_batch_correction(
                            ss["expr_imputed"],
                            translated_batches,
                            method=batch_method,
                            biological_groups=_group_map_pretty_lfq(pretty_labels),
                        )
                        if not ss.get("batch_correction_last_error"):
                            ss["expr_pre_batch"] = ss["expr_imputed"].copy()
                            ss["expr_processed"] = corrected.copy()
                            ss["expr_final"] = corrected.copy()
                            ss["batch_correction_applied"] = True
                            ss["batch_correction_applied_method"] = batch_method
                            ss["batch_correction_run_signature"] = batch_signature
                            ss["diff_results"] = None
                            st.success(
                                f"{batch_method} applied. Select **After batch correction** in the QC Data Source above "
                                "to compare it with the pre-batch matrix."
                            )

                if ss.get("batch_correction_applied", False):
                    st.success(f"Current final matrix includes {ss.get('batch_correction_applied_method', 'batch correction')}.")

    # =========================================================================
    # PILLAR 3: STATISTICAL ANALYSIS SUITE
    # =========================================================================
    if ss["show_suite"]:
        if not ss.get("preprocessing_applied", False) or ss.get("expr_processed") is None or ss["expr_processed"].empty:
            st.warning("Run normalization + imputation in 'Preprocessing, QC & Batch Correction' first.")
            st.stop()

        applied_signature = ss.get("preprocessing_run_signature")
        current_sample_signature = tuple(ss.get("global_all_groups", []))
        if not applied_signature or tuple(applied_signature[0]) != current_sample_signature:
            st.error("The selected samples/groups changed after preprocessing. Re-run normalization + imputation before statistics.")
            st.stop()
        if ss.get("preprocessing_dirty", False):
            st.error("Preprocessing settings changed after the last run. Re-run normalization + imputation before statistics.")
            st.stop()

        if not ss.get("pre_log2", True):
            st.error("Differential analysis requires the log2 preprocessing option to be enabled.")
            st.stop()
        tabs = st.tabs(["Differential Analysis", "Results Visualization"])
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
                    **Preprocessing is locked before statistics.**
                    - Sparse-protein filtering, normalization, and missing-value imputation are performed in the preprocessing/QC section.
                    - Differential analysis does **not** re-impute or re-normalize the data.
                    - Pairwise tests operate on the final log2 abundance matrix, so log2 fold-change is the difference between group mean log2 abundances.
                    """)
                can_pairwise = len(active) >= 2
                can_anova = (len(active) == 3 and ss["global_n_groups"] == 3)
                with st.form("run_diff_form"):
                    st.markdown("### Final Preprocessed Matrix")
                    st.info(
                        f"Normalization: **{ss.get('normalization_applied_method', 'None')}** · "
                        f"Imputation: **{ss.get('imputation_applied_method', 'None')}** · "
                        f"Batch correction: **{ss.get('batch_correction_applied_method', 'None')}**"
                    )
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
                    batch_method = "None"
                    if ss.get("batch_correction_applied", False):
                        st.info(
                            "Differential analysis is using the batch-corrected matrix generated with "
                            f"**{ss.get('batch_correction_applied_method', 'batch correction')}**."
                        )

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
                            diff_df = compute_diffexp_tp_lf(
                                df=analysis_df, control_cols=control_cols, test_cols=test_cols,
                                control_label=control_lab, test_label=test_lab, test_type=specific_test,
                                do_zero_impute=False, zero_frac=0.2,
                                do_presence_filter=False, presence_min_frac=0.0,
                                presence_rule="all_only", min_n_per_group=2,
                                batch_samples=ss.get("global_batch_samples"),
                                batch_method=batch_method
                            )
                        else:
                            diff_df = compute_diffexp_anova_tp_lf(
                                df=analysis_df, 
                                group1_cols=groups[0], group2_cols=groups[1], group3_cols=groups[2],
                                group1_label=labels[0], group2_label=labels[1], group3_label=labels[2],
                                do_zero_impute=False, zero_frac=0.2,
                                do_presence_filter=False, presence_min_frac=0.0,
                                presence_rule="all_only",
                                batch_samples=ss.get("global_batch_samples"),
                                batch_method=batch_method
                            )
                        
                        ss["diff_results"] = diff_df
                        st.success(f"LFQ Differential analysis complete: {selected_pair}")
                    
                    except Exception as e:
                        st.error(f"Differential analysis failed: {e}")
            if ss.get("diff_results") is not None:
                st.success("Differential analysis complete. Results available below.")
                st.dataframe(ss["diff_results"], use_container_width=True)
                
                with st.expander("📥 Download Full Dataset (Unfiltered)", expanded=False):
                    download_diff_df = ss["diff_results"].copy()
                    dl_col1, dl_col2 = st.columns(2)
                    
                    csv_diff = download_diff_df.to_csv(index=False).encode("utf-8")
                    with dl_col1:
                        add_bytes_download_button_tp_lf(
                            label="Download Results (CSV)",
                            data=csv_diff,
                            filename="differential_analysis_results_lfq_totalproteome.csv",
                            key="dl_diff_csv",
                            help_text="Download the complete differential-analysis table as CSV.",
                        )

                    buffer_diff = BytesIO()
                    with pd.ExcelWriter(buffer_diff, engine="openpyxl") as writer:
                        download_diff_df.to_excel(
                            writer,
                            index=False,
                            sheet_name="LFQ_Differential_Results",
                        )
                    with dl_col2:
                        add_bytes_download_button_tp_lf(
                            label="Download Results (Excel)",
                            data=buffer_diff.getvalue(),
                            filename="differential_analysis_results_lfq_totalproteome.xlsx",
                            key="dl_diff_xlsx",
                            help_text="Download the complete differential-analysis table as Excel.",
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
                accession_col_name = next((c for c in ["Accession", "Master Protein Accessions", "Protein Accessions", "Protein ID", "ProteinID", "Protein.IDs", "UniProt", "Uniprot"] if c in res.columns), None)

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
                                st.warning("Cannot map to a specific protein. Protein/gene identifier column missing.")
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
                                    "The protein identifier list is de-duplicated and can be pasted into pathway tools such as STRING, DAVID, or Enrichr."
                                )

                                c1, c2 = st.columns(2)
                                with c1:
                                    add_bytes_download_button_tp_lf(
                                        label="Download Significant Rows CSV",
                                        data=sig_rows.to_csv(index=False).encode("utf-8"),
                                        filename="significant_total_protein_results.csv",
                                        key=f"dl_sig_rows_{ss.get('explore_mode_tp', 'plot')}",
                                        help_text="Download rows passing the active significance thresholds.",
                                    )

                                if sig_genes:
                                    gene_txt = "\n".join(sig_genes)
                                    with c2:
                                        add_bytes_download_button_tp_lf(
                                            label=f"Download Protein List TXT ({n_genes})",
                                            data=gene_txt.encode("utf-8"),
                                            filename="significant_total_protein_identifier_list.txt",
                                            key=f"dl_sig_genes_{ss.get('explore_mode_tp', 'plot')}",
                                            help_text="Download the de-duplicated significant protein list.",
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

                            sig_mask = (res[metric] <= alpha_thr) & (res["log2FC"].abs() >= fc_thresh)
                            display_sig_stats(sig_mask)

                            volcano_plot_type = ss.get("volcano_plot_type", "Interactive")

                            control_label, test_label = get_pairwise_group_labels()

                            if volcano_plot_type == "Interactive":
                                fig = create_interactive_volcano_plot_tp_lf(
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
                                add_plotly_html_download_button_tp_lf(fig, "lfq_total_protein_volcano_interactive", "volcano_interactive")

                            else:
                                static_volcano_fig = create_static_volcano_plot_tp_lf(
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
                                add_figure_download_buttons_tp_lf(
                                    static_volcano_fig,
                                    "lfq_total_protein_volcano_annotated",
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

                            wf_fig, wf_df, wf_label_df = create_total_protein_waterfall_plot_tp_lf(
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

                                add_bytes_download_button_tp_lf(
                                    label="Download Numbered Waterfall Table CSV",
                                    data=numbered_wf_table.to_csv(index=False).encode("utf-8"),
                                    filename="lfq_total_protein_waterfall_numbered_table.csv",
                                    key="dl_waterfall_numbered_table_csv",
                                    help_text="Download the numbered waterfall annotation table.",
                                )

                            plt.close(wf_fig)

                            dlc1, dlc2 = st.columns(2)
                            with dlc1:
                                add_bytes_download_button_tp_lf(
                                    label="Download PNG",
                                    data=png_buf.getvalue(),
                                    filename="lfq_total_protein_waterfall.png",
                                    key="dl_waterfall_png",
                                    help_text="Download the waterfall plot as PNG.",
                                )
                            with dlc2:
                                add_bytes_download_button_tp_lf(
                                    label="Download SVG",
                                    data=svg_buf.getvalue(),
                                    filename="lfq_total_protein_waterfall.svg",
                                    key="dl_waterfall_svg",
                                    help_text="Download the waterfall plot as SVG.",
                                )


                    elif ss.get("show_ma", False):
                        if "log2FC" not in res.columns:
                            st.warning("MA Plot requires log2FC data.")
                        else:
                            metric = ss["ma_sig_col"]
                            alpha_thr = ss["ma_alpha_thr"]
                            fc_thresh = ss["ma_fc_thresh"]
                            top_n_ma_annot = ss["ma_top_n"]
                            
                            sig_mask = res[metric] <= alpha_thr
                            sig_mask = sig_mask & (res["log2FC"].abs() >= fc_thresh)
                            display_sig_stats(sig_mask)

                            mean_cols = [lbl for lbl in ss["global_group_labels"] if lbl in res.columns]
                            if len(mean_cols) >= 2:
                                ma_plot_type = ss.get("ma_plot_type", "Interactive")

                                control_label, test_label = get_pairwise_group_labels()

                                if ma_plot_type == "Interactive":
                                    fig = create_interactive_ma_plot_tp_lf(
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
                                    add_plotly_html_download_button_tp_lf(fig, "ma_plot_interactive", "ma_interactive")

                                else:
                                    fig, _plot_df = create_static_ma_plot_tp_lf(
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
                                    add_figure_download_buttons_tp_lf(fig, "ma_plot", "ma_static", include_pdf=True)
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
                                df_data = hm_data.sub(hm_data.mean(axis=1), axis=0).div(hm_data.std(axis=1), axis=0).fillna(0)
                            else:
                                df_data = hm_data.copy().fillna(0)
                            
                            mapping_dict = {}
                            for grp_label, grp_cols in zip(ss["global_group_labels"], ss["global_group_samples"]):
                                for c in grp_cols:
                                    mapping_dict[pretty_labels[c]] = grp_label
                                    
                            sample_groups = pd.Series(df_data.columns).map(mapping_dict)
                            unique_groups = [g for g in ss["global_group_labels"] if g in sample_groups.values]
                            
                            fallback_colors = {g: c for g, c in zip(unique_groups, sns.color_palette("Set2", len(unique_groups)).as_hex())}
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
                                df_data, figsize=(10, fig_height),
                                cmap="vlag" if normalize_heatmap == "Z-score" else "YlGnBu", 
                                center=0 if normalize_heatmap == "Z-score" else None,
                                col_cluster=col_cluster, row_cluster=row_cluster,
                                col_colors=col_colors, yticklabels=True,
                                dendrogram_ratio=(0.15, 0.05) if row_cluster else 0.15,
                                colors_ratio=(0.02, 0.02)
                            )
                            
                            if gene_col_name:
                                rowid_to_gene = res.set_index("RowID")[gene_col_name].to_dict()
                                new_labels = [str(rowid_to_gene.get(t.get_text(), t.get_text())) for t in fig.ax_heatmap.get_yticklabels()]
                                fig.ax_heatmap.set_yticklabels(new_labels, fontsize=10, fontweight='bold', rotation=0)
                            else:
                                plt.setp(fig.ax_heatmap.get_yticklabels(), rotation=0, fontsize=10, fontweight='bold')
                                
                            plt.setp(fig.ax_heatmap.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=10, fontweight='bold')
                            
                            cbar_label = 'Z-score' if normalize_heatmap == "Z-score" else 'Abundance'
                            fig.cax.set_ylabel(cbar_label, fontsize=12, fontweight='bold')

                            handles = [mpatches.Patch(color=group_palette[g], label=g) for g in unique_groups if g in group_palette]
                            fig.ax_col_dendrogram.legend(handles=handles, title="Group", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
                            
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
                                        st.warning("No protein identifiers are available for the selected significant rows.")
                                    else:
                                        st.markdown(f"##### Protein profiles for top {len(top_gene_names)} significant proteins")
                                        profile_figs = []
                                        profile_gene_names = []

                                        for gene_name in top_gene_names:
                                            fig = create_protein_profile_figure_tp_lf(
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
                                            grid_fig = create_profile_grid_figure_tp_lf(
                                                figures=profile_figs,
                                                gene_names=profile_gene_names,
                                                n_cols=4,
                                                panel_width=4.2,
                                                panel_height=3.8,
                                                dpi=220,
                                            )

                                            if grid_fig is not None:
                                                # Display one combined figure so double-click/zoom opens the whole grid.
                                                st.pyplot(grid_fig, use_container_width=True)
                                                ss["report_boxplot_fig"] = grid_fig
                                            else:
                                                ss["report_boxplot_fig"] = profile_figs[0]

                                            try:
                                                profile_pdf_bytes = build_profile_pdf_bytes_tp_lf(
                                                    figures=profile_figs,
                                                    gene_names=profile_gene_names,
                                                    n_cols=5,
                                                    n_rows=2,
                                                )
                                                add_bytes_download_button_tp_lf(
                                                    label="Download Protein Profiles PDF",
                                                    data=profile_pdf_bytes,
                                                    filename="lfq_top_significant_protein_profiles.pdf",
                                                    key="download_top_profile_pdf",
                                                    help_text="Download all displayed protein profiles as a multi-page PDF.",
                                                )
                                            except Exception as e:
                                                st.error(f"Could not prepare protein-profile PDF: {e}")
                                        else:
                                            st.warning("No profile plots could be generated for the selected proteins.")

                            else:
                                pick_gene = ss.get("pick_gene")
                                if pick_gene and pick_gene != "---":
                                    fig = create_protein_profile_figure_tp_lf(
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
                                        st.warning("No protein abundance values found for the selected protein.")
                                    else:
                                        st.pyplot(fig, use_container_width=True)
                                        ss["report_boxplot_fig"] = fig
                                        add_figure_download_buttons_tp_lf(
                                            fig,
                                            f"lfq_protein_profile_{pick_gene}",
                                            f"lfq_protein_profile_{_safe_download_filename(pick_gene)}",
                                            include_pdf=True,
                                        )


    # =========================================================================
    # PILLAR 4: RESULTS DOWNLOAD
    # =========================================================================

    # ====================== FINAL ANALYSIS REPORT ======================
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
                    """Add a session-state figure without closing the source figure."""
                    try:
                        png_bytes = _fig_to_png_bytes(fig)
                        zf.writestr(f"{folder}/{name}.png", png_bytes)
                    except Exception as e:
                        zf.writestr(
                            f"{folder}/{name}_EXPORT_WARNING.txt",
                            f"Could not export this figure: {e}",
                        )
                def _build_report_zip_tp_lf():
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
                            
                            batch_samples = ss.get("global_batch_samples", [])
                            batch_labels = ss.get("global_batch_labels", [])
                            for col in ss.get("global_sample_cols", []):
                                grp_name = "Unassigned"
                                grp_num = None
                                for i, cols in enumerate(group_samples):
                                    if col in (cols or []):
                                        grp_name = group_labels[i] if i < len(group_labels) else f"Group{i+1}"
                                        grp_num = i + 1
                                        break
                                        
                                batch_name = "N/A"
                                for j, b_cols in enumerate(batch_samples):
                                    if col in (b_cols or []):
                                        batch_name = batch_labels[j] if j < len(batch_labels) else f"Batch{j+1}"
                                        break
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
                            "Total Proteomics LFQ Report Bundle",
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
                    if st.button(
                        "Compile PDF Report & ZIP Bundle",
                        use_container_width=True,
                        type="primary",
                        help="Generate a PDF document and ZIP bundle containing all currently selected plots and parameters.",
                    ):
                        with st.spinner("Compiling plots and generating PDF... this may take a moment."):
                            try:
                                from tp.report_utility import generate_final_pdf

                                generated_pdf = generate_final_pdf()
                                ss["final_pdf_bytes"] = _coerce_to_bytes_for_downloads(generated_pdf)
                                ss["final_report_zip_bytes"] = _build_report_zip_tp_lf()
                                ss["final_report_timestamp"] = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to generate report: {e}")

                else:
                    st.success("Report generated successfully.")

                    if "final_report_timestamp" not in ss:
                        ss["final_report_timestamp"] = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                    timestamp = ss["final_report_timestamp"]

                    try:
                        pdf_bytes = _coerce_to_bytes_for_downloads(ss.get("final_pdf_bytes"))
                        st.caption(f"PDF size: {len(pdf_bytes) / 1024 / 1024:.2f} MB")
                    except Exception as e:
                        pdf_bytes = None
                        st.error(f"PDF report data are invalid: {e}")

                    if "final_report_zip_bytes" not in ss or ss.get("final_report_zip_bytes") is None:
                        try:
                            ss["final_report_zip_bytes"] = _build_report_zip_tp_lf()
                        except Exception as e:
                            ss["final_report_zip_bytes"] = None
                            st.warning(f"PDF is ready, but ZIP bundle could not be prepared: {e}")

                    zip_bytes = ss.get("final_report_zip_bytes")
                    if zip_bytes is not None:
                        try:
                            zip_bytes = _coerce_to_bytes_for_downloads(zip_bytes)
                            st.caption(f"ZIP size: {len(zip_bytes) / 1024 / 1024:.2f} MB")
                        except Exception as e:
                            zip_bytes = None
                            st.warning(f"ZIP bundle data are invalid: {e}")

                    dl_col1, dl_col2 = st.columns(2)
                    with dl_col1:
                        if pdf_bytes is not None:
                            add_bytes_download_button_tp_lf(
                                label="Download PDF Report",
                                data=pdf_bytes,
                                filename=f"TotalProteomics_LFQ_Report_{timestamp}.pdf",
                                key="final_report_download",
                                help_text="Download the compiled PDF report through the browser.",
                                type="primary",
                            )

                    with dl_col2:
                        if zip_bytes is not None:
                            add_bytes_download_button_tp_lf(
                                label="Download ZIP Bundle",
                                data=zip_bytes,
                                filename=f"TotalProteomics_LFQ_Report_Bundle_{timestamp}.zip",
                                key="final_report_zip_download",
                                help_text="Download the PDF, images, and tables as one ZIP archive.",
                            )
                        else:
                            st.warning("ZIP bundle is not available. Regenerate the report.")

                    if st.button(
                        "Regenerate Report",
                        use_container_width=True,
                        help="Discard the current report and generate a new one with updated data/plots.",
                        key="regenerate_final_report",
                    ):
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
    faq_section()

# =============================================================================
# Optional shared download imports
# =============================================================================
# Import utility-only names, then restore this app's browser-safe implementations.
_tp_lf_local_download_functions = {
    "add_hosted_safe_file_download_button": add_hosted_safe_file_download_button,
    "add_bytes_download_button_tp_lf": add_bytes_download_button_tp_lf,
    "add_figure_download_buttons_tp_lf": add_figure_download_buttons_tp_lf,
    "add_plotly_html_download_button_tp_lf": add_plotly_html_download_button_tp_lf,
    "save_profile_pdf_to_downloads_tp_lf": save_profile_pdf_to_downloads_tp_lf,
    "_save_bytes_to_downloads": _save_bytes_to_downloads,
    "_save_file_to_downloads": _save_file_to_downloads,
}

try:
    from download_utility import (
        add_download_button,
        add_file_download_buttons,
        add_bytes_download_button,
        add_bytes_download_button_tp,
        add_dataframe_download_buttons,
        add_matplotlib_download_buttons,
        add_plot_download_buttons,
        add_plotly_download_buttons,
        add_figure_download_buttons_tp,
        add_plotly_html_download_button_tp,
        add_figure_download_buttons_common,
        add_plotly_html_download_button_common,
        save_profile_pdf_to_downloads_tp,
        explain_download_location,
    )
except Exception:
    pass

# Do not allow a shared helper to replace the tested LFQ browser-download path.
globals().update(_tp_lf_local_download_functions)

if __name__ == "__main__":
    if "session_initialized_tp_lf" not in st.session_state:
        _reset_all_data(clear_cache=True)
        st.session_state["session_initialized_tp_lf"] = True
    main()
