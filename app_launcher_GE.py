#!/usr/bin/env python3
"""
################################################################################
GlycoExplorer — Combined pGlyco + PD / MSFragger workflow.
################################################################################
"""
__author__ = ("Saurabh Baheti: baheti.saurabh@mayo.edu")

import re
import warnings
from pathlib import Path
import zipfile
import datetime
from io import BytesIO
from typing import Optional
import base64
import plotly.express as px

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from PIL import Image

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
from faq.faq_GE import *
from ge.waterfall import *

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = 500_000_000  # safety

# -----------------------------------------------------------------------------
# Streamlit page config
# -----------------------------------------------------------------------------
try_set_page_config(page_title="GlycoExplorer | Mayo Clinic", layout="wide")
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

# One-time session defaults
ss.setdefault("merged_ready", False)
ss.setdefault("merged_df", None)
ss.setdefault("diffexp_df", None)
ss.setdefault("filtered_df", None)
ss.setdefault("diff_done", False)
ss.setdefault("show_overview", False)
ss.setdefault("show_suite", False)
ss.setdefault("show_global_select", False)
ss.setdefault("upload_sig", None)
ss.setdefault("show_report", False)

ss.setdefault("merged_df_before_batch", None)
ss.setdefault("enable_batch_correction", False)

# Permanent batch-configuration state.
# These keys are deliberately separate from Streamlit widget keys so the
# configuration survives when the Data Filtering panel is not rendered.
ss.setdefault("batch_method_selected", "ComBat (Empirical Bayes)")
ss.setdefault("batch_num_selected", 2)
ss.setdefault("batch_config", {})
ss.setdefault("irs_ref_config", {})

# Explore tab flags
for k in [
    "run_pca", "run_corr", "run_peptide_freq",
    "run_abundance", "run_glycan_class", "run_overview_heatmap"
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

# Tool usage
ss.setdefault("tool_usage", {})

# Standardized report keys shared with ge.report_utility
ss.setdefault("waterfall_sig_col", "q-value")
ss.setdefault("waterfall_alpha", 0.05)
ss.setdefault("waterfall_logfc", 1.0)
ss.setdefault("waterfall_topn", 5)

# -----------------------------------------------------------------------------
# Statistical tests offered in the Statistical Analysis Suite
# -----------------------------------------------------------------------------
LIMMA_T = "limma moderated t-test (log2 scale)"
LIMMA_F = "limma moderated F-test (log2 scale)"
CLASSIC_ANOVA = "One-way ANOVA (classical F-test)"

# limma is listed first, so it is the default choice.
PAIRWISE_TESTS = [
    LIMMA_T,
    "Student's t-test (Independent, Equal Variance)",
    "Welch's t-test (Independent, Unequal Variance)",
    "Paired t-test (Matched Samples)",
    "Mann-Whitney U (Non-parametric Independent)",
    "Wilcoxon Signed-Rank (Non-parametric Paired)",
]
MULTIGROUP_TESTS = [LIMMA_F, CLASSIC_ANOVA]

TEST_HELP = {
    "Student's t-test (Independent, Equal Variance)":
        "Classic two-sample t-test on raw abundances. Assumes both groups have a similar spread.",
    "Welch's t-test (Independent, Unequal Variance)":
        "t-test on raw abundances that does not assume equal spread. Safer than Student's when one group is more variable.",
    "Paired t-test (Matched Samples)":
        "For matched samples (e.g. the same patient before and after). Channels are paired in the order they are "
        "listed in each group, so both groups need the same number of channels in matching order.",
    "Mann-Whitney U (Non-parametric Independent)":
        "Rank-based test with no normality assumption. With 3 vs 3 samples the smallest possible p-value is 0.1, "
        "so nothing can pass a 0.05 cutoff. Use it only with larger groups.",
    "Wilcoxon Signed-Rank (Non-parametric Paired)":
        "Rank-based paired test. Needs at least 6 pairs before any p-value can fall below 0.05.",
    LIMMA_T:
        "Recommended for TMT with few replicates. Works on log2 abundances and borrows variance information from "
        "all glycoforms (empirical Bayes), which gives stable results with 3-6 replicates. log2FC is the difference "
        "of mean log2 abundance, so it can differ slightly from the log2FC of the other tests.",
    CLASSIC_ANOVA:
        "One-way ANOVA on raw abundances: does any of the three groups differ? Glycoforms with no variation "
        "inside a group are skipped.",
    LIMMA_F:
        "limma's version of ANOVA on log2 abundances, with empirical Bayes variances. The p/q-value says whether "
        "any group differs. The table also gives log2FC and q-value for every pair of groups (e.g. B vs A, C vs A, "
        "C vs B), so one run answers which groups differ.",
}



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


def _save_bytes_to_downloads(data, filename):
    """
    Save bytes directly to the user's Downloads folder.
    This is more reliable than browser download inside packaged desktop apps.
    Existing files are not overwritten; a numeric suffix is added.
    """
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


def _save_file_to_downloads(source_path, filename=None):
    """
    Save an existing file to the user's Downloads folder.
    Used for bundled example datasets in the standalone desktop app.
    """
    source_path = Path(source_path)
    if filename is None:
        filename = source_path.name
    return _save_bytes_to_downloads(source_path.read_bytes(), filename)


# =============================================================================
# MSFRAGGER PARSING LOGIC
# =============================================================================

def _calculate_prosite(row):
    start_col = 'Protein Start' if 'Protein Start' in row else 'Start'
    
    if pd.isna(row.get('Assigned Modifications')) or pd.isna(row.get(start_col)):
        return "Unknown"
    
    try:
        start_pos = int(row[start_col])
    except ValueError:
        return "Unknown"
        
    matches = re.findall(r'(\d+)N\(([\d\.]+)\)', str(row['Assigned Modifications']))
    
    valid_sites = []
    for pos, mass in matches:
        if float(mass) > 300:
            prosite = start_pos + int(pos) - 1
            valid_sites.append(prosite)
            
    if not valid_sites:
        return "Unknown"
        
    valid_sites = sorted(list(set(valid_sites)))
    return "; ".join(map(str, valid_sites))
    
def glycan_to_composition(glycan_str):
    if pd.isna(glycan_str): return None
    glycan_part = str(glycan_str).split('%')[0].strip()
    matches = re.findall(r'([A-Za-z]+)\((\d+)\)', glycan_part)
    if not matches: return None
    return " ".join(f"{mono}:{count}" for mono, count in matches)

def _clean_glycan(glycan_str):
    if pd.isna(glycan_str): return None
    return str(glycan_str).split('%')[0].strip()

def detect_tmt_channels(psm_df):
    return [
        c for c in psm_df.columns
        if re.match(r'^Intensity \S+', c) and not c.startswith('Intensity NA')
    ]

def build_glyco_id_table(psm_df, peptide_df, tmt_channels):
    glyco = psm_df[psm_df['Glycan_Composition'].notna()].copy()
    
    agg_dict = {
        'Glycan_Composition': ('Glycan_Composition', 'first'),
        'PSM_Count':          ('Peptide', 'count'),
        'Best_Glycan_Score':  ('Glycan Score', 'max'),
        'Best_Glycan_Qvalue': ('Glycan q-value', 'min'),
        'ProSites':           ('ProSites', 'first')
    }
    
    for col in tmt_channels:
        ge_col_name = col.replace('Intensity ', 'Abundance: ')
        agg_dict[ge_col_name] = (col, 'mean')

    glycoform_table = (
        glyco.groupby(['Peptide', 'Glycan_Composition_Clean'])
        .agg(**agg_dict)
        .reset_index()
    )

    meta_cols = [c for c in ['Peptide', 'Protein Description', 'Protein ID', 'Gene'] if c in peptide_df.columns]
    meta = peptide_df[meta_cols].drop_duplicates('Peptide')
    glycoform_table = glycoform_table.merge(meta, on='Peptide', how='left')

    if 'Gene' in glycoform_table.columns:
        glycoform_table.rename(columns={'Gene': 'Genes'}, inplace=True)
    
    glycoform_table.rename(columns={'Glycan_Composition_Clean': 'Glycan(H,N,A,F)'}, inplace=True)
    glycoform_table['Glycan'] = glycoform_table['Glycan(H,N,A,F)']
    glycoform_table['GlycanClass'] = glycoform_table['Glycan(H,N,A,F)'].apply(classify_glycan)

    front_cols = [
        'Peptide', 'Protein Description', 'Protein ID', 'Genes',
        'ProSites', 'Glycan(H,N,A,F)', 'Glycan', 'GlycanClass',
        'Glycan_Composition', 'PSM_Count', 'Best_Glycan_Score', 'Best_Glycan_Qvalue'
    ]
    
    intensity_cols = [c for c in glycoform_table.columns if c.startswith('Abundance:')]
    
    return (glycoform_table[front_cols + intensity_cols]
            .sort_values('PSM_Count', ascending=False)
            .reset_index(drop=True))


def _reset_all_data(clear_cache: bool = False):
    ss["global_sample_cols"] = []
    ss["global_group_samples"] = []
    ss["global_group_labels"] = []
    ss["global_n_groups"] = 2
    ss["global_control_group_idx"] = 0
    ss["global_all_groups"] = []
    ss["batch_dict"] = {}
    ss["batch_config"] = {}
    ss["irs_ref_config"] = {}
    ss["batch_corrected"] = False
    ss["enable_batch_correction"] = False
    ss["batch_method_selected"] = "ComBat (Empirical Bayes)"
    ss["batch_num_selected"] = 2
    ss["merged_df_before_batch"] = None

    # Clear temporary Streamlit batch widget keys.
    for _k in list(ss.keys()):
        if (
            str(_k).startswith("_ge_batch_")
            or str(_k).startswith("_ge_irs_ref_")
            or str(_k).startswith("batch_sel_")
            or str(_k).startswith("irs_ref_")
        ):
            ss.pop(_k, None)

    ss["show_report"] = False
    ss["merged_ready"] = False
    ss["merged_df"] = None
    ss["diffexp_df"] = None
    ss["filtered_df"] = None
    ss["diff_done"] = False
    ss["filtered_ready"] = False

    for k in [
        "run_pca", "run_corr", "run_peptide_freq",
        "run_abundance", "run_glycan_class", "run_overview_heatmap"
    ]:
        ss[k] = False

    for key in [
        "show_volcano", "show_box", "show_heatmap",
        "show_box_gene", "show_lollipop"
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
    ss.pop("stat_method_label", None)
    ss["show_class_breakdown"] = False

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
        "merged_df": None,
        "diffexp_df": None,
        "design_df": None,
        "group_col": None,
        "batch_col": None,
        "tool_usage": {},
    }
    for k, v in defaults.items():
        ss.setdefault(k, v)


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------
def _plot_genes_cb():
    st.session_state["box_gene_plot"] = True

def _clear_genes_cb():
    st.session_state["box_gene_list"] = []
    st.session_state["box_gene_plot"] = False
    st.session_state["box_gene_cleared"] = True

def _plot_lollipop_cb():
    st.session_state["lollipop_gene_plot"] = True

def _clear_lollipop_cb():
    st.session_state["lollipop_gene_ids"] = ""
    st.session_state["lollipop_gene_list"] = []
    st.session_state["lollipop_gene_plot"] = False

def _reset_merge_state():
    ss["merged_df"] = None
    ss["merged_ready"] = False
    ss["show_overview"] = False
    ss["show_suite"] = False
    ss["show_global_select"] = False
    ss["show_report"] = False

    for k in [
        "run_pca", "run_corr", "run_overview_heatmap",
        "run_peptide_freq", "run_abundance", "run_glycan_class"
    ]:
        ss[k] = False


def _msfragger_to_pglyco(glycan_str):
    if pd.isna(glycan_str): return None
    clean_str = str(glycan_str).split('%')[0].strip()
    
    h = re.search(r'Hex\((\d+)\)', clean_str)
    n = re.search(r'HexNAc\((\d+)\)', clean_str)
    a = re.search(r'NeuAc\((\d+)\)', clean_str)
    f = re.search(r'Fuc\((\d+)\)', clean_str)
    
    h_count = h.group(1) if h else "0"
    n_count = n.group(1) if n else "0"
    a_count = a.group(1) if a else "0"
    f_count = f.group(1) if f else "0"
    
    return f"{h_count} {n_count} {a_count} {f_count}"



def render_professional_start_panel(pipeline_choice: str, merge_disabled: bool):
    """
    Professional landing panel shown before data are uploaded/processed.
    Keeps the first-user experience clean and removes informal icons/emojis.
    """
    if pipeline_choice == "pGlyco + Proteome Discoverer":
        upload_text = "Upload your pGlyco and Proteome Discoverer files using the left sidebar."
        required_text = "Required files: pGlyco identification file and Proteome Discoverer abundance file."
    else:
        upload_text = "Upload your MSFragger psm.tsv and peptide.tsv files using the left sidebar."
        required_text = "Required files: MSFragger psm.tsv and peptide.tsv."

    button_text = (
        "Upload data to enable processing"
        if merge_disabled
        else "Merge & Process Data Matrix"
    )

    st.markdown(
        """
        <style>
        .ge-start-wrap {
            margin-top: 1.4rem;
            margin-bottom: 1.5rem;
        }
        .ge-start-title {
            font-size: 1.85rem;
            line-height: 1.15;
            font-weight: 700;
            color: #005a9c;
            margin-bottom: 0.35rem;
            letter-spacing: -0.01em;
        }
        .ge-start-subtitle {
            font-size: 1.02rem;
            color: #2f3a45;
            margin-bottom: 1.6rem;
            max-width: 980px;
            line-height: 1.55;
        }
        .ge-step-card {
            border: 1px solid #d9e2ef;
            border-radius: 10px;
            padding: 22px 24px;
            background: #f8fbff;
            min-height: 138px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
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
            vertical-align: middle;
        }
        .ge-step-heading {
            font-size: 1.15rem;
            font-weight: 700;
            color: #005a9c;
            vertical-align: middle;
        }
        .ge-step-body {
            color: #2f3a45;
            margin-left: 58px;
            margin-top: 8px;
            font-size: 0.96rem;
            line-height: 1.5;
        }
        .ge-step-arrow {
            text-align: center;
            font-size: 2.2rem;
            color: #8a94a6;
            padding-top: 45px;
            font-weight: 300;
        }
        .ge-info-box {
            background: #eef6ff;
            border-radius: 8px;
            padding: 16px 20px;
            margin-top: 24px;
            color: #004b8d;
            font-weight: 700;
            border: 1px solid #d8eaff;
        }
        .ge-info-box-sub {
            font-weight: 400;
            margin-top: 6px;
            color: #263746;
        }
        .ge-warning-box {
            background: #fff8e5;
            border-radius: 8px;
            padding: 16px 20px;
            margin-top: 14px;
            color: #8a5a00;
            font-weight: 700;
            border: 1px solid #f5e4b6;
        }
        .ge-warning-box-sub {
            font-weight: 400;
            margin-top: 6px;
            color: #5f4300;
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
        .ge-active-button-note {
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
        <div class="ge-start-wrap">
            <div class="ge-start-title">Start Here</div>
            <div class="ge-start-subtitle">
                Begin your analysis by uploading the required data files from the left panel.
                Once uploaded, click the processing button to create the glycopeptide abundance matrix.
            </div>
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
        st.markdown("<div class='ge-step-arrow'>›</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            """
            <div class="ge-step-card">
                <div>
                    <span class="ge-step-number">2</span>
                    <span class="ge-step-heading">Process Data</span>
                </div>
                <div class="ge-step-body">
                    Click <b>Merge & Process Data Matrix</b> to create the analysis-ready glycopeptide table.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="ge-info-box">
            Upload data using the left sidebar
            <div class="ge-info-box-sub">{required_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if merge_disabled:
        st.markdown(
            """
            <div class="ge-warning-box">
                No data uploaded yet
                <div class="ge-warning-box-sub">
                    Please upload the required files to enable data processing.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    btn_class = "ge-muted-button" if merge_disabled else "ge-active-button-note"
    st.markdown(
        f"""
        <div class="{btn_class}">{button_text}</div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# MAIN
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


# =============================================================================
# Glycan class breakdown of differential results
# =============================================================================
def _glycan_trait(glycan_str) -> str:
    """Sialylation / fucosylation from a composition string "H N A F" or "H N A G F"."""
    nums = re.findall(r"\d+", str(glycan_str))
    if len(nums) == 4:
        _h, _n, a, f = map(int, nums)
        sia = a
    elif len(nums) == 5:
        _h, _n, a, g, f = map(int, nums)
        sia = a + g
    else:
        return "Unknown"
    if sia > 0 and f > 0:
        return "Sialylated + fucosylated"
    if sia > 0:
        return "Sialylated only"
    if f > 0:
        return "Fucosylated only"
    return "No sialic acid or fucose"


def render_glycan_class_breakdown(plot_df: pd.DataFrame, is_anova: bool, sig_col: str,
                                  alpha: float, fc_thr: float, group_by: str):
    """
    Which glycan classes / traits are over-represented among significant glycoforms?
    One-sided Fisher's exact test per class and direction, BH-adjusted.
    """
    from scipy.stats import fisher_exact
    from statsmodels.stats.multitest import multipletests

    df = plot_df.copy()
    if group_by == "Glycan class":
        if "GlycanClass" not in df.columns:
            st.info("These results have no GlycanClass column.")
            return
        df["_class"] = df["GlycanClass"].fillna("Unknown").astype(str)
    else:
        if "Glycan" not in df.columns:
            st.info("These results have no Glycan composition column.")
            return
        df["_class"] = df["Glycan"].apply(_glycan_trait)

    sig = pd.to_numeric(df[sig_col], errors="coerce") < alpha
    if is_anova:
        df["_status"] = np.where(sig, "Significant", "Not significant")
        directions = ["Significant"]
    else:
        lfc = pd.to_numeric(df["log2FC"], errors="coerce")
        df["_status"] = "Not significant"
        df.loc[sig & (lfc > fc_thr), "_status"] = "Up"
        df.loc[sig & (lfc < -fc_thr), "_status"] = "Down"
        directions = ["Up", "Down"]

    counts = pd.crosstab(df["_class"], df["_status"])
    for col in directions + ["Not significant"]:
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[directions + ["Not significant"]]
    counts["Total"] = counts.sum(axis=1)
    counts = counts.sort_values("Total", ascending=False)

    total = int(counts["Total"].sum())
    rows = []
    for cls, r in counts.iterrows():
        row = {"Class": cls, "Glycoforms": int(r["Total"])}
        for d in directions:
            in_cls_dir = int(r[d])
            in_cls_not = int(r["Total"]) - in_cls_dir
            out_dir = int(counts[d].sum()) - in_cls_dir
            out_not = (total - int(r["Total"])) - out_dir
            odds, pval = fisher_exact([[in_cls_dir, in_cls_not], [out_dir, out_not]], alternative="greater")
            row[f"{d} (n)"] = in_cls_dir
            row[f"{d} (% of class)"] = round(100.0 * in_cls_dir / max(1, int(r["Total"])), 1)
            row[f"{d} enrichment p"] = pval
        rows.append(row)
    table = pd.DataFrame(rows)

    pcols = [c for c in table.columns if c.endswith("enrichment p")]
    if pcols and len(table):
        allp = table[pcols].to_numpy().ravel()
        q = multipletests(allp, method="fdr_bh")[1].reshape(table[pcols].shape)
        for k, c in enumerate(pcols):
            table[c.replace("enrichment p", "enrichment q")] = q[:, k]

    n_hits = int(sum(counts[d].sum() for d in directions))
    if n_hits == 0:
        st.warning("No glycoform passes the current cutoffs, so there is nothing to compare. Loosen the cutoffs on the left.")

    # 100% stacked bars: share of each class that is up / down / not significant
    long = counts[directions + ["Not significant"]].reset_index().melt(
        id_vars="_class", var_name="Status", value_name="Glycoforms"
    )
    long["Percent"] = long.apply(
        lambda r: 100.0 * r["Glycoforms"] / max(1, int(counts.loc[r["_class"], "Total"])), axis=1
    )
    colors = {"Up": "#d62728", "Down": "#1f77b4", "Significant": "#8e44ad", "Not significant": "#d0d4da"}
    fig = px.bar(
        long, y="_class", x="Percent", color="Status", orientation="h",
        color_discrete_map=colors, text="Glycoforms",
        category_orders={"_class": list(counts.index), "Status": directions + ["Not significant"]},
        labels={"_class": group_by, "Percent": "% of glycoforms in class"},
    )
    fig.update_layout(
        barmode="stack", height=120 + 48 * len(counts), margin=dict(l=10, r=10, t=40, b=10),
        legend_title_text="", title=f"Share of significant glycoforms per {group_by.lower()}",
        plot_bgcolor="white",
    )
    fig.update_traces(textposition="inside")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Enrichment: one-sided Fisher's exact test of whether a class holds more significant glycoforms than "
        "expected from its size, Benjamini-Hochberg adjusted (q). It tells you which classes change more often "
        "than others, not how much they change."
    )
    show_cols = ["Class", "Glycoforms"] + [c for c in table.columns if c.endswith("(n)") or c.endswith("% of class)") or c.endswith("enrichment q")]
    st.dataframe(
        table[show_cols].style.format({c: "{:.3g}" for c in show_cols if c.endswith("enrichment q")}),
        use_container_width=True, hide_index=True,
    )
    add_hosted_safe_file_download_button(
        label="Download table (CSV)",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=f"glycan_{'class' if group_by == 'Glycan class' else 'trait'}_breakdown.csv",
        key="download_glycan_breakdown_csv",
        use_container_width=False,
    )


def main():
    init_session_state_defaults()

    st.sidebar.markdown("### GlycoExplorer (TMT)")
    st.sidebar.caption("v1.0 | Bioinformatics Workflow")
    st.sidebar.divider()

    species = st.sidebar.radio("Select Species", ["Human", "Mouse"], index=0)
    version_label = "v3"

    pipeline_choice = st.sidebar.radio(
        "Select Pipeline Engine",
        ["pGlyco + Proteome Discoverer", "MSFragger (TMT)"]
    )

    pglyco_upload, pd_upload, psm_upload, pep_upload = None, None, None, None

    if pipeline_choice == "pGlyco + Proteome Discoverer":
        pglyco_help = """
        **Required Columns:**
        - `Peptide`, `Genes`, `Proteins`, `ProSites`
        - `RawName`, `Scan`
        - `Glycan(H,N,A,F)` (Human)
        """
        pglyco_upload = st.sidebar.file_uploader(
            "Upload pGlyco Data",
            type=["csv", "txt", "tsv", "xlsx", "xls"],
            key="pglyco_csv",
            help=pglyco_help
        )

        pd_help = """
        **Required Columns:**
        - `Spectrum File`
        - `First Scan`
        - Numeric sample columns (e.g., `Abundance: Sample_1`)
        """
        pd_upload = st.sidebar.file_uploader(
            "Upload Proteome Discoverer Data",
            type=["csv", "txt", "tsv", "xlsx", "xls"],
            key="pd_csv",
            help=pd_help
        )
        
        merge_disabled = not (pglyco_upload and pd_upload)
        new_sig = _make_upload_sig(pglyco_upload, pd_upload)
    else:
        psm_help = """
        **Required Columns:**
        - `Total Glycan Composition`
        - `Intensity <channel>`
        """
        psm_upload = st.sidebar.file_uploader(
            "Upload MSFragger psm.tsv",
            type=["tsv", "txt"],
            key="msf_psm",
            help=psm_help
        )
        
        pep_help = """
        **Required Columns:**
        - `Peptide`
        - `Protein ID`
        - `Gene`
        """
        pep_upload = st.sidebar.file_uploader(
            "Upload MSFragger peptide.tsv",
            type=["tsv", "txt"],
            key="msf_pep",
            help=pep_help
        )
        
        merge_disabled = not (psm_upload and pep_upload)
        new_sig = _make_upload_sig(psm_upload, pep_upload)

    with st.sidebar.expander("Download Example Datasets", expanded=False):
        st.caption("Use these 10-row example files to see the required formatting and test the workflow.")

        script_dir = Path(__file__).resolve().parent
        example_dir = script_dir / "assets" / "example" / "GE"

        example_pglyco_path = example_dir / "pGlycoDB.csv"
        example_pd_path     = example_dir / "PD.csv"
        example_psm         = example_dir / "MSFragger_psm.tsv"
        example_pep         = example_dir / "MSFragger_pep.tsv"

        if pipeline_choice == "pGlyco + Proteome Discoverer":
            st.markdown("**pGlyco + Proteome Discoverer examples**")
            col1, col2 = st.columns(2)

            if example_pglyco_path.exists():
                with col1:
                    add_download_button(example_pglyco_path.read_bytes(), "pGlycoDB.csv", label="Download pGlycoDB", key=f"save_ex_pglyco_{pipeline_choice}", mime="text/csv")

            if example_pd_path.exists():
                with col2:
                    add_download_button(example_pd_path.read_bytes(), "PD.csv", label="Download PD.csv", key=f"save_ex_pd_{pipeline_choice}", mime="text/csv")

        elif pipeline_choice == "MSFragger (TMT)":
            st.markdown("**MSFragger TMT examples**")
            col1, col2 = st.columns(2)

            if example_psm.exists():
                with col1:
                    add_download_button(example_psm.read_bytes(), "MSFragger_psm.tsv", label="Download psm.tsv", key=f"save_ex_psm_{pipeline_choice}", mime="text/tab-separated-values")

            if example_pep.exists():
                with col2:
                    add_download_button(example_pep.read_bytes(), "MSFragger_pep.tsv", label="Download peptide.tsv", key=f"save_ex_pep_{pipeline_choice}", mime="text/tab-separated-values")

        # Also offer the other pipeline's examples (not the same files twice).
        if pipeline_choice != "MSFragger (TMT)":
            st.markdown("<div style='padding-top: 10px;'></div>", unsafe_allow_html=True)

            st.markdown("**MSFragger (TMT)**")
            col3, col4 = st.columns(2)

            if example_psm.exists():
                with col3:
                    add_download_button(example_psm.read_bytes(), "MSFragger_psm.tsv", label="Download psm.tsv", key="save_ex_psm_general", mime="text/tab-separated-values")

            if example_pep.exists():
                with col4:
                    add_download_button(example_pep.read_bytes(), "MSFragger_pep.tsv", label="Download peptide.tsv", key="save_ex_pep_general", mime="text/tab-separated-values")

    if st.sidebar.button(
        "Reset session data",
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
    main_merge_label = (
        "Merge & Process Data Matrix"
        if not merge_disabled
        else "Upload data to enable processing"
    )

    with st.container(border=True):
        if st.button(
            main_merge_label,
            type="primary",
            disabled=merge_disabled,
            key="btn_merge",
            help="Processes identifications and quantitative abundances based on the selected pipeline."
        ):
            if pipeline_choice == "pGlyco + Proteome Discoverer":
                if not (pglyco_upload and pd_upload):
                    st.error("Please upload both files before merging.")
                else:
                    try:
                        pglyco_df = read_data_file(pglyco_upload)
                        pd_df     = read_data_file(pd_upload)

                        required_pglyco_cols_h = [
                            "Peptide", "Genes", "Proteins",
                            "ProSites", "RawName", "Scan", "Glycan(H,N,A,F)"
                        ]
                        required_pglyco_cols_m = [
                            "Peptide", "Genes", "Proteins",
                            "ProSites", "RawName", "Scan", "Glycan(H,N,A,G,F)"
                        ]
                        required_pd_cols = ["Spectrum File", "First Scan"]

                        if species.lower() == "mouse":
                            missing_pglyco = [c for c in required_pglyco_cols_m if c not in pglyco_df.columns]
                        else:
                            missing_pglyco = [c for c in required_pglyco_cols_h if c not in pglyco_df.columns]
                        
                        if missing_pglyco:
                            st.error(f"pGlyco file is missing required columns: {', '.join(missing_pglyco)}")
                            st.stop()

                        missing_pd = [c for c in required_pd_cols if c not in pd_df.columns]
                        if missing_pd:
                            st.error(f"Proteome Discoverer file is missing required columns: {', '.join(missing_pd)}")
                            st.stop()

                        abundance_cols = [
                            c for c in pd_df.columns
                            if re.match(r"(?i)^abundance[s]?:\s*\d+", str(c))
                            and not re.search(r"(?i)normalized|scaled|s/n|corr", str(c))
                        ]
                        
                        if not abundance_cols:
                            st.warning("No 'Abundance' columns detected in PD file. Please confirm PD export settings.")

                        if pglyco_df.empty or pd_df.empty:
                            st.error("One or both uploaded files are empty or unreadable.")
                            st.stop()

                        if abundance_cols:
                            pd_df = pd_df[required_pd_cols + abundance_cols]
                        else:
                            pd_df = pd_df[required_pd_cols]
                            
                        merged = merge_pglyco_pd(pglyco_df, pd_df, version_label, species)

                        ss["merged_df"] = merged
                        ss["merged_ready"] = True
                        ss["diff_done"] = False
                        ss["diffexp_df"] = None
                        ss["filtered_df"] = None
                        ss["filtered_ready"] = False
                        ss["batch_corrected"] = False

                    except Exception as e:
                        st.error(f"Merge failed: {e}")

            elif pipeline_choice == "MSFragger (TMT)":
                if not (psm_upload and pep_upload):
                    st.error("Please upload both psm.tsv and peptide.tsv.")
                else:
                    try:
                        with st.spinner("Processing MSFragger Glyco-PSMs..."):
                            psm_df = pd.read_csv(psm_upload, sep='\t', low_memory=False)
                            peptide_df = pd.read_csv(pep_upload, sep='\t', low_memory=False)

                            psm_df['Glycan_Composition'] = psm_df['Total Glycan Composition'].apply(glycan_to_composition)
                            psm_df['Glycan_Composition_Clean'] = psm_df['Total Glycan Composition'].apply(_msfragger_to_pglyco)
                            psm_df['ProSites'] = psm_df.apply(_calculate_prosite, axis=1)
                            
                            tmt_channels = detect_tmt_channels(psm_df)
                            merged = build_glyco_id_table(psm_df, peptide_df, tmt_channels)
                            
                            abundance_cols = [c for c in merged.columns if c.startswith('Abundance:')]

                        if merged.empty:
                            st.error("No valid glycopeptides found after parsing.")
                        else:
                            ss["merged_df"] = merged
                            ss["merged_ready"] = True
                            ss["diff_done"] = False
                            ss["diffexp_df"] = None
                            ss["filtered_df"] = None
                            ss["filtered_ready"] = False
                            ss["batch_corrected"] = False
                            st.success(f"Successfully processed {len(merged)} unique MSFragger glycoforms!")
                    except Exception as e:
                        st.error(f"MSFragger Processing failed: {e}")

    downstream_active = (
        ss.get("show_global_select") or
        ss.get("show_overview") or
        ss.get("show_suite") or
        ss.get("show_report")
    )

    if not downstream_active and ss.get("merged_ready") and not df_is_empty(ss.get("merged_df")):
        merged_display_df = ss["merged_df"]

        abundance_cols_display = [
            c for c in merged_display_df.columns
            if pd.api.types.is_numeric_dtype(merged_display_df[c])
        ]

        with st.expander("Integrated Glycopeptide Abundance Matrix", expanded=True):
            m1, m2, m3, m4 = st.columns(4)

            m1.metric("Glycoforms", f"{len(merged_display_df):,}")
            m2.metric("Samples", len(abundance_cols_display))
            m3.metric(
                "Genes",
                merged_display_df["Genes"].nunique() if "Genes" in merged_display_df.columns else "N/A"
            )
            m4.metric(
                "Glycan Classes",
                merged_display_df["GlycanClass"].nunique() if "GlycanClass" in merged_display_df.columns else "N/A"
            )

            st.markdown("<br>", unsafe_allow_html=True)

            dl_col1, dl_col2, spacer = st.columns([1.5, 1.5, 7])

            csv_bytes = merged_display_df.to_csv(index=False).encode("utf-8")

            with dl_col1:
                add_hosted_safe_file_download_button(
                    label="Download CSV",
                    data=csv_bytes,
                    file_name="merged_output.csv",
                    key="download_merged_csv",
                    use_container_width=True,
                )

            buffer_excel = BytesIO()
            with pd.ExcelWriter(buffer_excel, engine="openpyxl") as writer:
                merged_display_df.to_excel(writer, index=False, sheet_name="Merged_Data")

            excel_bytes = buffer_excel.getvalue()

            with dl_col2:
                add_hosted_safe_file_download_button(
                    label="Download Excel",
                    data=excel_bytes,
                    file_name="merged_output.xlsx",
                    key="download_merged_excel",
                    use_container_width=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            tab_matrix, tab_classes = st.tabs(["Abundance matrix", "Glycan class composition"])
            with tab_matrix:
                show_grid(merged_display_df, page_size=15, height=450)
            with tab_classes:
                if "GlycanClass" in merged_display_df.columns:
                    st.caption(
                        "How the identified glycoforms are spread across glycan classes. This describes what was "
                        "identified, not data quality. To see which classes change between groups, run the "
                        "statistics and open **Glycan class breakdown** in Results Visualizations."
                    )
                    create_glycan_class_summary(merged_display_df, "GlycanClass")
                else:
                    st.info("This dataset has no GlycanClass column.")

    if not ss.get("merged_ready") or df_is_empty(ss.get("merged_df")):

        ss["show_overview"] = False
        ss["show_suite"] = False
        ss["show_global_select"] = False

        render_professional_start_panel(
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

    c1, c2, c3, c4 = st.columns(4)

    if c1.button(
        ("▸ " if not ss["show_global_select"] else "▾ ") + "Data Filtering & Group Specification",
        width='stretch',
        key="btn_global_select",
        help="Define experimental cohorts, map batches, and set the baseline reference group."
    ):
        ss["show_global_select"] = not ss["show_global_select"]
        if ss["show_global_select"]:
            ss["show_overview"] = False
            ss["show_suite"] = False
            ss["show_report"] = False
            _reset_overview_flags()
        else:
            _reset_overview_flags()

    if c2.button(
        ("▸ " if not ss["show_overview"] else "▾ ") + "Data Integrity Check & Sanity Profiling",
        width='stretch',
        key="btn_overview",
        help="Quality control before testing: PCA, sample correlation, sample clustering and per-sample abundance distributions."
    ):
        ss["show_overview"] = not ss["show_overview"]
        if ss["show_overview"]:
            ss["show_suite"] = False
            ss["show_global_select"] = False
            ss["show_report"] = False
        else:
            _reset_overview_flags()

    if c3.button(
        ("▸ " if not ss["show_suite"] else "▾ ") + "Statistical Analysis Suite",
        width='stretch',
        key="btn_suite",
        help="Pairwise tests, ANOVA or limma, then volcano, waterfall, box, heatmap, lollipop and glycan class plots."
    ):
        ss["show_suite"] = not ss["show_suite"]
        if ss["show_suite"]:
            ss["show_overview"] = False
            ss["show_global_select"] = False
            ss["show_report"] = False
            _reset_overview_flags()
        else:
            _reset_overview_flags()

    if c4.button(
        ("▸ " if not ss["show_report"] else "▾ ") + "Results Download",
        width='stretch',
        key="btn_report",
        help="Generate a comprehensive PDF report and a ZIP bundle of all tables and figures."
    ):
        ss["show_report"] = not ss["show_report"]
        if ss["show_report"]:
            ss["show_overview"] = False
            ss["show_global_select"] = False
            ss["show_suite"] = False
            _reset_overview_flags()
        else:
            _reset_overview_flags()

    # Step guide under the four buttons: what each step needs and whether it is done.
    _n_groups_ready = sum(1 for g in (ss.get("global_group_samples") or []) if g)
    c1.caption(
        f"Step 1 · :green[✓ {_n_groups_ready} groups defined]" if _n_groups_ready >= 2
        else "Step 1 · Start here: assign samples to at least 2 groups"
    )
    c2.caption("Step 2 · Check samples look as expected before testing")
    c3.caption(
        f"Step 3 · :green[✓ {ss.get('stat_method_label', 'Statistics done')}]" if ss.get("diff_done")
        else ("Step 3 · Ready to run statistics" if _n_groups_ready >= 2 else "Step 3 · Needs groups from step 1")
    )
    c4.caption("Step 4 · PDF report and ZIP of tables and figures" if ss.get("diff_done")
               else "Step 4 · Available after statistics")

    st.markdown("<br>", unsafe_allow_html=True)

    # ====================== DATA FILTERING & GROUP SPECIFICATION ======================
    if ss.get("show_global_select"):
        df = ss["merged_df"]

        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        numeric_cols = list(numeric_cols)
        numeric_cols = sorted(numeric_cols, key=lambda x: str(x))

        def clean_label(name: str) -> str:
            s = str(name)
            s = re.sub(r"^\s*Abundance\s*:\s*", "", s)
            if ":" in s:
                s = s.split(":", 1)[-1]
            return s.strip()

        pretty_labels = {c: clean_label(c) for c in numeric_cols}
        fmt = lambda x: pretty_labels.get(x, x)

        with st.expander("What do I set here?", icon=":material/help:"):
            st.markdown(
                "1. **Sample columns**: the TMT channels to analyse. Leave out empty channels. Leave out the "
                "pool/reference channel too, unless you need it for IRS batch correction.\n"
                "2. **Groups**: put every channel into a biological group (2 or 3 groups). The statistics "
                "compare these groups.\n"
                "3. **Reference group**: the baseline, usually the control. Fold changes are reported as "
                "*other group vs reference*, so a positive log2FC means higher than the reference.\n"
                "4. **Batch correction (optional)**: only if your samples were run in more than one TMT plex "
                "or batch. Every grouped channel must be in a batch, and each batch should contain more than "
                "one group. If a batch holds only one group, batch and biology cannot be separated."
            )

        def _uniq(seq):
            seen = set()
            out = []
            for x in seq:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            return out

        def _sanitize(selected, allowed):
            allowed_set = set(allowed)
            return _uniq([x for x in (selected or []) if x in allowed_set])

        left, right = st.columns([0.42, 0.58], gap="large")

        with left:
            with st.container(border=True):
                st.markdown("#### Define Experimental Groups")

                if "global_sample_cols_ui" not in ss:
                    default_samples = ss.get("global_sample_cols") or numeric_cols
                    ss["global_sample_cols_ui"] = _sanitize(default_samples, numeric_cols)

                ss["global_sample_cols_ui"] = _sanitize(ss["global_sample_cols_ui"], numeric_cols)

                sample_cols = st.multiselect(
                    "Sample/abundance columns (used everywhere)",
                    options=numeric_cols,
                    help="TMT channels used in QC, statistics and the report. Remove empty or pool channels.",
                    format_func=fmt,
                    key="global_sample_cols_ui",
                )

                sample_cols = _sanitize(sample_cols, numeric_cols)

                if not sample_cols:
                    st.warning("Select at least 1 sample column.")
                    ss["global_sample_cols"] = []
                    ss["global_group_samples"] = []
                    ss["global_group_labels"] = []
                    ss["global_all_groups"] = []
                else:
                    ss["global_sample_cols"] = sample_cols

                st.divider()

                if "global_n_groups_ui" not in ss:
                    ss["global_n_groups_ui"] = 2 if ss.get("global_n_groups", 2) == 2 else 3

                n_groups = st.radio(
                    "Number of groups",
                    [2, 3],
                    help="Choose 3 to unlock the three-group tests (ANOVA or limma F-test).",
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
                    format_func=fmt,
                    key="global_grp1",
                )
                grp1 = _sanitize(grp1, sample_cols)

                if "global_g1_label" not in ss:
                    ss["global_g1_label"] = g_labels[0] if len(g_labels) > 0 else "Group1"
                g1_label = st.text_input(
                    "Group 1 label",
                    value=ss["global_g1_label"],
                    key="global_g1_label"
                )

                options_g2 = [c for c in sample_cols if c not in set(grp1)]

                if "global_grp2" not in ss:
                    grp2_default = (g_groups[1] if len(g_groups) > 1 else d2)
                    ss["global_grp2"] = _sanitize(grp2_default, options_g2)
                ss["global_grp2"] = _sanitize(ss["global_grp2"], options_g2)

                grp2 = st.multiselect(
                    "Group 2 columns",
                    options=options_g2,
                    format_func=fmt,
                    key="global_grp2",
                )
                grp2 = _sanitize(grp2, options_g2)

                if "global_g2_label" not in ss:
                    ss["global_g2_label"] = g_labels[1] if len(g_labels) > 1 else "Group2"
                g2_label = st.text_input(
                    "Group 2 label",
                    value=ss["global_g2_label"],
                    key="global_g2_label"
                )

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
                        format_func=fmt,
                        key="global_grp3",
                    )
                    grp3 = _sanitize(grp3, options_g3)

                    if "global_g3_label" not in ss:
                        ss["global_g3_label"] = g_labels[2] if len(g_labels) > 2 else "Group3"
                    g3_label = st.text_input(
                        "Group 3 label",
                        value=ss["global_g3_label"],
                        key="global_g3_label"
                    )
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

                nonempty_groups = [
                    (i, lbl, cols)
                    for i, (lbl, cols) in enumerate(zip(global_group_labels, global_group_samples))
                    if cols
                ]
                
                if nonempty_groups:
                    ss.setdefault("global_control_group_idx", nonempty_groups[0][0])
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
                        key="global_control_choice",
                    )
                    ss["global_control_group_idx"] = control_choice
                    st.caption(f"Global reference set to: **{global_group_labels[control_choice]}**")
                else:
                    ss["global_control_group_idx"] = 0
                    st.info("Select at least one column in any group to set a global reference.")

            with st.container(border=True):
                st.markdown("#### Batch Effect Correction")

                batch_allowed_samples = ss.get("global_all_groups", [])
                batch_allowed_samples = _sanitize(batch_allowed_samples, sample_cols)

                if not batch_allowed_samples:
                    st.info("Assign samples to Group 1/2/3 first before enabling batch correction.")
                else:
                    # -------------------------------------------------------------
                    # Persistent toggle
                    # -------------------------------------------------------------
                    # A temporary widget key is used because Streamlit may remove
                    # widget keys when this panel is hidden. The permanent setting
                    # remains in ss["enable_batch_correction"].
                    if "_ge_batch_toggle" not in ss:
                        ss["_ge_batch_toggle"] = bool(
                            ss.get("enable_batch_correction", False)
                        )

                    apply_batch = st.toggle(
                        "Apply Batch Effect Correction",
                        key="_ge_batch_toggle"
                    )
                    ss["enable_batch_correction"] = bool(apply_batch)

                    if apply_batch:
                        with st.container(border=True):
                            st.markdown("##### Batch Setup")

                            # -----------------------------------------------------
                            # Persistent batch method
                            # -----------------------------------------------------
                            batch_methods = [
                                "ComBat (Empirical Bayes)",
                                "Median Centering",
                                "Internal Reference Scaling (IRS)"
                            ]

                            saved_method = ss.get(
                                "batch_method_selected",
                                batch_methods[0]
                            )
                            if saved_method not in batch_methods:
                                saved_method = batch_methods[0]

                            if "_ge_batch_method" not in ss:
                                ss["_ge_batch_method"] = saved_method

                            batch_method = st.selectbox(
                                "Batch Correction Method",
                                options=batch_methods,
                                help=(
                                    "ComBat requires neuroCombat. "
                                    "IRS requires a common pool channel across multiplexes."
                                ),
                                key="_ge_batch_method"
                            )
                            ss["batch_method_selected"] = batch_method

                            # -----------------------------------------------------
                            # Persistent number of batches
                            # -----------------------------------------------------
                            batch_count_options = [2, 3, 4]
                            saved_num_batches = int(
                                ss.get("batch_num_selected", 2)
                            )
                            if saved_num_batches not in batch_count_options:
                                saved_num_batches = 2

                            if "_ge_num_batches" not in ss:
                                ss["_ge_num_batches"] = saved_num_batches

                            num_batches = st.radio(
                                "Number of Batches",
                                batch_count_options,
                                horizontal=True,
                                key="_ge_num_batches"
                            )
                            ss["batch_num_selected"] = int(num_batches)

                            # -----------------------------------------------------
                            # Persistent batch sample assignments
                            # -----------------------------------------------------
                            # Prefer saved UI configuration. If this is an older
                            # session, fall back to the batches used for the last
                            # successful correction.
                            stored_batch_config = (
                                ss.get("batch_config", {})
                                or ss.get("batch_dict", {})
                                or {}
                            )

                            batch_dict = {}
                            selected_so_far = []

                            for b in range(num_batches):
                                batch_name = f"Batch {b+1}"
                                widget_key = f"_ge_batch_sel_{b}"

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

                                # Seed the widget from permanent state when the
                                # Data Filtering panel is rebuilt.
                                ss[widget_key] = prev_selected

                                b_cols = st.multiselect(
                                    f"{batch_name} Samples",
                                    options=available_for_batch,
                                    format_func=fmt,
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

                            # Save assignments immediately, not only after the
                            # correction button is clicked.
                            ss["batch_config"] = {
                                k: list(v) for k, v in batch_dict.items()
                            }

                            assigned_samples = _uniq(
                                [
                                    c
                                    for cols in batch_dict.values()
                                    for c in cols
                                ]
                            )

                            unassigned_batch_samples = [
                                c
                                for c in batch_allowed_samples
                                if c not in assigned_samples
                            ]

                            if unassigned_batch_samples:
                                st.warning(
                                    "These grouped samples are not assigned to any batch: "
                                    + ", ".join(
                                        [fmt(c) for c in unassigned_batch_samples]
                                    )
                                )

                            # -----------------------------------------------------
                            # Persistent IRS reference channels
                            # -----------------------------------------------------
                            irs_ref_dict = {}
                            if batch_method == "Internal Reference Scaling (IRS)":
                                nonempty_batches_irs = {
                                    k: v for k, v in batch_dict.items() if v
                                }

                                if nonempty_batches_irs:
                                    st.markdown(
                                        "**Select the Reference/Pool Sample for each batch:**"
                                    )

                                    saved_irs_refs = ss.get(
                                        "irs_ref_config", {}
                                    ) or {}

                                    for b_idx, (b_name, b_cols) in enumerate(
                                        nonempty_batches_irs.items()
                                    ):
                                        irs_widget_key = f"_ge_irs_ref_{b_idx}"

                                        saved_ref = saved_irs_refs.get(b_name)
                                        if saved_ref not in b_cols:
                                            saved_ref = b_cols[0]

                                        if irs_widget_key not in ss:
                                            ss[irs_widget_key] = saved_ref
                                        elif ss[irs_widget_key] not in b_cols:
                                            ss[irs_widget_key] = saved_ref

                                        irs_ref_dict[b_name] = st.selectbox(
                                            f"{b_name} Reference Channel",
                                            options=b_cols,
                                            format_func=fmt,
                                            key=irs_widget_key
                                        )

                                    ss["irs_ref_config"] = dict(irs_ref_dict)

                            # -----------------------------------------------------
                            # Run correction
                            # -----------------------------------------------------
                            if st.button(
                                "Run Batch Correction",
                                type="primary",
                                key="run_batch_correction_btn"
                            ):
                                assigned_samples = _uniq(
                                    [
                                        c
                                        for cols in batch_dict.values()
                                        for c in cols
                                    ]
                                )
                                nonempty_batches = {
                                    k: v
                                    for k, v in batch_dict.items()
                                    if v
                                }

                                duplicated = [
                                    c
                                    for c in assigned_samples
                                    if sum(
                                        c in cols
                                        for cols in batch_dict.values()
                                    ) > 1
                                ]

                                if duplicated:
                                    st.error(
                                        "A sample cannot be assigned to more than one batch: "
                                        + ", ".join(
                                            [fmt(c) for c in duplicated]
                                        )
                                    )

                                elif len(nonempty_batches) < 2:
                                    st.error(
                                        "Please assign samples to at least two non-empty batches."
                                    )

                                elif set(assigned_samples) != set(batch_allowed_samples):
                                    st.error(
                                        "Please assign every grouped sample to one batch before running correction."
                                    )

                                else:
                                    try:
                                        # Always retain the original uncorrected
                                        # matrix for PCA-before comparison and for
                                        # restoring the data if correction is turned off.
                                        if (
                                            ss.get("merged_df_before_batch") is None
                                            or not ss.get("batch_corrected", False)
                                        ):
                                            ss["merged_df_before_batch"] = (
                                                ss["merged_df"].copy()
                                            )

                                        input_df_for_batch = (
                                            ss["merged_df_before_batch"].copy()
                                        )

                                        if batch_method == "ComBat (Empirical Bayes)":
                                            # ComBat must know each sample's biological group,
                                            # otherwise it removes real group differences.
                                            group_map = {
                                                col: lbl
                                                for lbl, cols in zip(
                                                    ss.get("global_group_labels", []),
                                                    ss.get("global_group_samples", []),
                                                )
                                                for col in (cols or [])
                                            }
                                            corrected_df, combat_info = apply_combat_batch_correction(
                                                input_df_for_batch,
                                                nonempty_batches,
                                                batch_allowed_samples,
                                                group_map=group_map,
                                            )
                                            for _w in combat_info.get("warnings", []):
                                                st.warning(_w)

                                        elif batch_method == "Internal Reference Scaling (IRS)":
                                            corrected_df = apply_irs_batch_correction(
                                                input_df_for_batch,
                                                nonempty_batches,
                                                irs_ref_dict,
                                                batch_allowed_samples
                                            )

                                        else:
                                            corrected_df = apply_median_centering_batch_correction(
                                                input_df_for_batch,
                                                nonempty_batches,
                                                batch_allowed_samples
                                            )

                                        ss["merged_df"] = corrected_df
                                        ss["batch_dict"] = {
                                            k: list(v)
                                            for k, v in nonempty_batches.items()
                                        }
                                        ss["batch_config"] = {
                                            k: list(v)
                                            for k, v in batch_dict.items()
                                        }
                                        ss["batch_method_selected"] = batch_method
                                        ss["batch_num_selected"] = int(num_batches)
                                        ss["batch_corrected"] = True

                                        ss["diffexp_df"] = None
                                        ss["filtered_df"] = None
                                        ss["diff_done"] = False
                                        ss["filtered_ready"] = False

                                        st.success(
                                            f"{batch_method} applied successfully. "
                                            "QC PCA will show before and after batch correction, "
                                            "and downstream analysis will use corrected abundances."
                                        )

                                    except ImportError as ie:
                                        st.error(str(ie))
                                    except Exception as e:
                                        st.error(
                                            f"Batch correction failed: {e}"
                                        )

                    else:
                        # If correction had already been applied, turning the
                        # toggle OFF should restore the original data rather than
                        # merely changing the flag while leaving corrected values.
                        if (
                            ss.get("batch_corrected", False)
                            and isinstance(
                                ss.get("merged_df_before_batch"),
                                pd.DataFrame
                            )
                            and not ss["merged_df_before_batch"].empty
                        ):
                            ss["merged_df"] = (
                                ss["merged_df_before_batch"].copy()
                            )
                            ss["diffexp_df"] = None
                            ss["filtered_df"] = None
                            ss["diff_done"] = False
                            ss["filtered_ready"] = False

                        ss["batch_dict"] = {}
                        ss["batch_corrected"] = False

                        # Keep batch_config and merged_df_before_batch so the
                        # user's assignments can be restored if correction is
                        # turned back on during this session.

        with right:
            with st.container(border=True):
                st.markdown("#### Sample Mapping Summary")

                sample_cols = ss.get("global_sample_cols", [])
                group_samples = ss.get("global_group_samples", [])
                group_labels  = ss.get("global_group_labels", [])
                control_idx   = ss.get("global_control_group_idx", 0)
                batch_dict = (
                    ss.get("batch_config", {})
                    if ss.get("enable_batch_correction", False)
                    else ss.get("batch_dict", {})
                )

                rows = []
                for col in sample_cols:
                    grp_name = "Unassigned"
                    grp_num = None
                    for i, cols in enumerate(group_samples):
                        if col in (cols or []):
                            grp_name = group_labels[i] if i < len(group_labels) else f"Group{i+1}"
                            grp_num = i + 1
                            break

                    batch_name = "Unassigned"
                    for b_name, b_cols in batch_dict.items():
                        if col in b_cols:
                            batch_name = b_name
                            break

                    rows.append({
                        "Column": col,
                        "Sample": fmt(col),
                        "Group": grp_name,
                        "Batch": batch_name if ss.get("batch_corrected") else "N/A",
                        "Group_Number": grp_num,
                        "Is_Reference": (grp_num is not None and (grp_num - 1) == control_idx)
                    })

                meta_df = pd.DataFrame(rows)

                st.dataframe(meta_df, width='stretch', hide_index=True)

                # Pass CSV as bytes, not as a str. The centralized download utility
                # treats strings as possible file paths; a full CSV string can exceed
                # the OS filename-length limit and raise OSError: [Errno 63].
                sample_metadata_csv = meta_df.to_csv(index=False).encode("utf-8")

                add_hosted_safe_file_download_button(
                    label="Download sample metadata CSV",
                    data=sample_metadata_csv,
                    file_name="sample_metadata_mapping.csv",
                    key="download_sample_metadata_csv",
                    use_container_width=True,
                )

                if not meta_df.empty:
                    n_unassigned = (meta_df["Group"] == "Unassigned").sum()
                    if n_unassigned > 0:
                        st.warning(f"{n_unassigned} selected sample(s) are not assigned to any group.")

    # ====================== DATA INTEGRITY CHECK & SANITY PROFILING ======================
    if ss.get("show_overview"):
        df = ss.merged_df
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

        def clean_label(name: str) -> str:
            s = str(name)
            s = re.sub(r"^\s*Abundance\s*:\s*", "", s)
            if ":" in s:
                s = s.split(":", 1)[-1]
            return s.strip()

        pretty_labels = {c: clean_label(c) for c in numeric_cols}
        fmt = lambda x: pretty_labels.get(x, x)

        sample_cols = ss.get("global_sample_cols") or numeric_cols
        group_samples = ss.get("global_group_samples") or []
        group_labels  = ss.get("global_group_labels") or []
        n_groups      = ss.get("global_n_groups", 2)

        if not sample_cols:
            st.warning("No numeric abundance columns detected.")
        else:
            if not group_samples or not any(group_samples):
                n = len(sample_cols)
                d1 = sample_cols[: max(1, n // 3)]
                d2 = sample_cols[max(1, n // 3): max(2, 2 * n // 3)]
                d3 = sample_cols[max(2, 2 * n // 3):]

                if n_groups == 3:
                    group_samples = [d1, d2, d3]
                    group_labels  = ["Group1", "Group2", "Group3"]
                else:
                    mid = max(1, n // 2)
                    group_samples = [sample_cols[:mid], sample_cols[mid:]]
                    group_labels  = ["Group1", "Group2"]

            all_groups = ss.get("global_all_groups") or list(dict.fromkeys([c for g in group_samples for c in g]))
            if not all_groups:
                all_groups = sample_cols

            left, right = st.columns([0.42, 0.58], gap="large")

            with left:
                with st.container(border=True):

                    with st.expander("Assigned Sample Mapping", expanded=False):
                        sample_cols = ss.get("global_sample_cols", [])
                        
                        if not sample_cols:
                            st.info("No samples selected in Data Filtering & Group Specification.")
                        else:
                            group_samples = ss.get("global_group_samples", [])
                            group_labels  = ss.get("global_group_labels", [])
                            control_idx   = ss.get("global_control_group_idx", None)
                            batch_dict    = ss.get("batch_dict", {})

                            rows = []
                            for col in sample_cols:
                                grp_name = "Unassigned"
                                grp_num  = None
                                is_ctrl  = False

                                for i, enumerate_cols in enumerate(group_samples or []):
                                    if col in (enumerate_cols or []):
                                        grp_name = group_labels[i] if i < len(group_labels) else f"Group{i+1}"
                                        grp_num  = i + 1
                                        is_ctrl  = (control_idx is not None and i == control_idx)
                                        break
                                        
                                batch_name = "Unassigned"
                                for b_name, b_cols in batch_dict.items():
                                    if col in b_cols:
                                        batch_name = b_name
                                        break

                                rows.append({
                                    "Sample": fmt(col),
                                    "Group": grp_name,
                                    "Group #": grp_num,
                                    "Batch": batch_name if ss.get("batch_corrected") else "N/A",
                                    "Is Reference": is_ctrl
                                })

                            meta_df = pd.DataFrame(rows)
                            assigned_df = meta_df[meta_df["Group"] != "Unassigned"].copy()

                            n_hidden = (meta_df["Group"] == "Unassigned").sum()
                            if n_hidden > 0:
                                st.warning(f"Hiding {n_hidden} selected sample(s) that are not assigned to any group.")

                            if assigned_df.empty:
                                st.info("No selected samples are assigned to a group yet.")
                            else:
                                st.dataframe(assigned_df, width='stretch', hide_index=True)

                                if not assigned_df["Is Reference"].any():
                                    st.info("No reference group selected yet.")

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
                            "Abundance Density": "Abundance Distribution (per sample)"
                        }.get(m, m),
                    )

                    _qc_help = {
                        "Principal Component Analysis (PCA)":
                            "Each dot is a sample. Samples of the same group should sit together. A sample far from "
                            "its group may be an outlier (failed labelling, wrong channel). If samples separate by "
                            "batch or plex instead of by group, consider batch correction.",
                        "Correlation Matrix":
                            "Correlation of raw abundances between samples. Replicates of one group should correlate "
                            "highly. A sample that correlates poorly with all others needs a closer look before testing.",
                        "Sample Clustering":
                            "Clusters samples on the most variable glycoforms without using the group labels. If "
                            "samples cluster by group, biology dominates. If they cluster by batch or run order, "
                            "check for batch effects.",
                        "Abundance Density":
                            "log2 abundance distribution of each sample. In TMT data the curves should overlap. A curve "
                            "shifted left or right means that channel had less or more material loaded. The statistics "
                            "in this app do not correct for that, so check this before testing.",
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
                        data_for_pca = df[all_groups].dropna()
                        
                        if not data_for_pca.empty and len(data_for_pca.columns) > 0:
                            n_comps = min(10, data_for_pca.shape[0], data_for_pca.shape[1])
                            pc_cols = [f"PC{i+1}" for i in range(n_comps)]
                        else:
                            pc_cols = ["PC1", "PC2"]
                        
                        col_x, col_y = st.columns(2)
                        with col_x:
                            ss["pca_x"] = st.selectbox(
                                "X-axis Component",
                                options=pc_cols,
                                index=0,
                                key="pca_x_ui"
                            )
                        with col_y:
                            ss["pca_y"] = st.selectbox(
                                "Y-axis Component",
                                options=pc_cols,
                                index=1 if len(pc_cols) > 1 else 0,
                                key="pca_y_ui"
                            )
                            
                        ss["pca_draw_ellipses"] = st.checkbox(
                            "Draw confidence ellipses",
                            value=True,
                            key="pca_ellipses_ui"
                        )

                        pca_opt1, pca_opt2 = st.columns(2)
                        with pca_opt1:
                            ss["pca_hide_outliers"] = st.checkbox(
                                "Hide outliers",
                                value=False,
                                key="pca_hide_outliers_ui",
                                help="Remove IQR-based outliers from the displayed PCA plot."
                            )
                        with pca_opt2:
                            ss["pca_label_outliers"] = st.checkbox(
                                "Label outliers",
                                value=True,
                                key="pca_label_outliers_ui",
                                help="Add sample names next to detected outliers."
                            )

                        ss["pca_zoom_q"] = st.slider(
                            "Zoom view",
                            min_value=0.80,
                            max_value=1.00,
                            value=1.00,
                            step=0.01,
                            key="pca_zoom_q_ui",
                            help="1.00 shows the full PCA. Lower values zoom into the central samples."
                        )

                        st.markdown("**Customize Group Colors**")
                        default_palette = ["#b3e2cd", "#fdcdac", "#cbd5e8", "#decbe4", "#ccebc5", "#fed9a6"]
                        ss["pca_palette"] = {}
                        cols = st.columns(len(group_labels))
                        
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
                            "Top N most variable features used for clustering",
                            10, 300, 50,
                            key="overview_hm_topn_ui"
                        )
                        ss["overview_hm_norm"] = st.radio(
                            "Data Scaling",
                            ["Z-score", "Raw Abundance"],
                            key="overview_hm_norm_ui",
                            horizontal=True
                        )
                        st.caption("Samples are clustered without using group labels. Group colors are used only to color sample labels.")

                        st.markdown("**Customize Group Colors**")
                        default_cluster_palette = ["#b3e2cd", "#fdcdac", "#cbd5e8", "#decbe4", "#ccebc5", "#fed9a6"]
                        ss["cluster_palette"] = {}
                        if group_labels:
                            cluster_cols = st.columns(len(group_labels))
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
                        "Feature Frequency Profiling": dict(
                            run_pca=False, run_corr=False, run_overview_heatmap=False,
                            run_peptide_freq=True, run_abundance=False, run_glycan_class=False
                        ),
                        "Abundance Density": dict(
                            run_pca=False, run_corr=False, run_overview_heatmap=False,
                            run_peptide_freq=False, run_abundance=True, run_glycan_class=False
                        ),
                        "Glycan Classification Summary": dict(
                            run_pca=False, run_corr=False, run_overview_heatmap=False,
                            run_peptide_freq=False, run_abundance=False, run_glycan_class=True
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
                            and isinstance(ss.get("merged_df_before_batch"), pd.DataFrame)
                            and not ss["merged_df_before_batch"].empty
                        )

                        if batch_was_applied:
                            st.caption(
                                "Compare the same grouped samples before and after batch correction. "
                                "A useful correction should reduce technical batch separation without "
                                "removing the expected biological group structure."
                            )
                            st.markdown(
                                """
                                <span style="
                                    background-color:#e8f4ff;
                                    color:#005a9c;
                                    padding:5px 10px;
                                    border-radius:999px;
                                    font-size:0.85rem;
                                    font-weight:600;
                                ">
                                Batch correction applied
                                </span>
                                """,
                                unsafe_allow_html=True
                            )

                            st.markdown("<br>", unsafe_allow_html=True)

                            pca_col1, pca_col2 = st.columns(2, gap="large")

                            with pca_col1:
                                st.markdown("PCA Before Batch Correction")
                                create_pca_plot(
                                    df=ss["merged_df_before_batch"],
                                    group_samples=group_samples,
                                    group_labels=group_labels,
                                    title="PCA Before Batch Correction",
                                    pc_x=ss.get("pca_x", "PC1"),
                                    pc_y=ss.get("pca_y", "PC2"),
                                    draw_ellipses=ss.get("pca_draw_ellipses", False),
                                    hide_outliers=ss.get("pca_hide_outliers", False),
                                    label_outliers=ss.get("pca_label_outliers", True),
                                    zoom_q=ss.get("pca_zoom_q", 1.0),
                                    palette=ss.get("pca_palette", None),
                                    quant_mode="TMT",
                                    key_suffix="before_batch"
                                )

                            with pca_col2:
                                st.markdown("PCA After Batch Correction")
                                create_pca_plot(
                                    df=ss["merged_df"],
                                    group_samples=group_samples,
                                    group_labels=group_labels,
                                    title="PCA After Batch Correction",
                                    pc_x=ss.get("pca_x", "PC1"),
                                    pc_y=ss.get("pca_y", "PC2"),
                                    draw_ellipses=ss.get("pca_draw_ellipses", False),
                                    hide_outliers=ss.get("pca_hide_outliers", False),
                                    label_outliers=ss.get("pca_label_outliers", True),
                                    zoom_q=ss.get("pca_zoom_q", 1.0),
                                    palette=ss.get("pca_palette", None),
                                    quant_mode="TMT",
                                    key_suffix="after_batch"
                                )

                        else:
                            #st.markdown("### PCA")
                            create_pca_plot(
                                df=df,
                                group_samples=group_samples,
                                group_labels=group_labels,
                                title="PCA",
                                pc_x=ss.get("pca_x", "PC1"),
                                pc_y=ss.get("pca_y", "PC2"),
                                draw_ellipses=ss.get("pca_draw_ellipses", False),
                                    hide_outliers=ss.get("pca_hide_outliers", False),
                                    label_outliers=ss.get("pca_label_outliers", True),
                                    zoom_q=ss.get("pca_zoom_q", 1.0),
                                palette=ss.get("pca_palette", None),
                                quant_mode="TMT",
                                key_suffix="single_pca"
                            )

                    else:
                        st.info("No samples available for PCA.")

                elif ss.get("run_corr", False):
                    if len(all_groups) >= 2:
                        create_sample_corr_heatmap(df, all_groups)
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
                            quant_mode="TMT",
                            palette=ss.get("cluster_palette", None),
                            key_suffix="overview_cluster"
                        )
                    elif len(all_groups) >= 2:
                        # Fallback if samples were selected but not explicitly grouped.
                        fig_obj = create_sample_clustering(
                            df=df,
                            group_samples=[all_groups],
                            group_labels=["All selected samples"],
                            title="Sample clustering",
                            top_n=ss.get("overview_hm_topn", 50),
                            norm_type=ss.get("overview_hm_norm", "Z-score"),
                            quant_mode="TMT",
                            palette={"All selected samples": "#b3e2cd"},
                            key_suffix="overview_cluster"
                        )
                    else:
                        st.warning("Need at least two sample columns for sample clustering.")

                elif ss.get("run_abundance", False):
                    if len(all_groups) >= 1:
                        create_abundance_density(df, all_groups)
                        # The report looks for "report_density_fig".
                        ss["report_density_fig"] = ss.get("report_abundance_density_fig")
                    else:
                        st.info("No numeric sample columns detected for abundance density.")

                else:
                    st.info("Choose an Analysis Module on the left to begin.")

    # ====================== Step 4: Statistical Analysis Suite ======================
    if ss.get("show_suite"):
        tabs = st.tabs(["Differential Analysis", "Results Visualizations"])

        def _detect_sample_options(df: pd.DataFrame):
            candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            
            def _clean(name: str) -> str:
                s = str(name)
                patterns = [
                    r'^(?:Abundance)\s*:\s*(.+)$',
                    r'^(?:Abundance)\s*\([^)]*\)\s*:\s*(.+)$',
                    r'^(?:Abundances?)\s*\([^)]*\)\s*:\s*(.+)$',
                    r'^(?:Abundance)\s+(.+)$',
                ]
                for pat in patterns:
                    m = re.match(pat, s, flags=re.IGNORECASE)
                    if m:
                        return m.group(1).strip()
                return s

            display_to_actual = {}
            used = set()
            for actual in candidates:
                disp = _clean(actual)
                if disp in used:
                    i = 2
                    new = f"{disp} [{i}]"
                    while new in used:
                        i += 1
                        new = f"{disp} [{i}]"
                    disp = new
                used.add(disp)
                display_to_actual[disp] = actual

            return list(display_to_actual.keys()), display_to_actual

        def _unique(seq):
            seen = set()
            out = []
            for x in seq:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            return out

        # ------------------- TAB: Differential Analysis -------------------
        with tabs[0]:

            merged_df = ss.get("merged_df")
            
            if df_is_empty(merged_df):
                st.info("Upload and process data first.")
            else:
                group_samples = ss.get("global_group_samples", [])
                group_labels  = ss.get("global_group_labels", [])
                n_groups      = ss.get("global_n_groups", 2)
                ctrl_idx      = ss.get("global_control_group_idx", 0)

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
                        ss["global_control_group_idx"] = ctrl_idx

                    ss["group_samples"] = group_samples
                    ss["group_labels"]  = group_labels
                    ss["control_group_idx"] = ctrl_idx
                    ss["all_groups"] = list(dict.fromkeys([c for g in group_samples for c in g]))

                    with st.expander("Current Group Specification (read-only)", expanded=True):
                        st.write(f"**Number of groups:** {n_groups}")
                        st.write(f"**Reference group:** {group_labels[ctrl_idx]}")
                        for i, (lbl, cols) in enumerate(zip(group_labels, group_samples), start=1):
                            st.write(f"**Group {i} — {lbl}:** {len(cols)} samples")
                            st.caption(", ".join([str(c) for c in cols]) if cols else "(empty)")

                    ss.setdefault("compare_open", False)
                    ss.setdefault("filter_open", False)

                    if st.button(
                        ("▸ " if not ss["compare_open"] else "▾ ") + "Select Statistical Test",
                        key="btn_toggle_compare",
                        width='stretch',
                        help="Open the configuration panel to select the statistical test type and specific comparisons."
                    ):
                        ss["compare_open"] = not ss["compare_open"]

                    if not ss["compare_open"]:
                        st.info("Click Select Statistical Test to choose pairwise or ANOVA and run the analysis.")
                    else:
                        with st.container(border=True):
                            st.markdown("#### Test Parameters and Execution")

                            with st.expander("Which test should I use?", icon=":material/help:"):
                                st.markdown(
                                    "| Your design | Suggested test |\n"
                                    "|---|---|\n"
                                    "| 2 groups, independent samples, 3-6 replicates (typical TMT) | **limma moderated t-test** |\n"
                                    "| 2 groups, independent samples, 8+ replicates | Welch's t-test or limma |\n"
                                    "| Same subjects measured twice (matched channels) | Paired t-test |\n"
                                    "| 3 groups | **limma moderated F-test** (default), then read the pairwise q-values |\n\n"
                                    "**Why limma?** With few replicates, each glycoform's variance is estimated from only "
                                    "a handful of values, so ordinary t-tests produce many false hits (tiny variance by "
                                    "chance) and miss real ones. limma shrinks each variance towards the typical variance "
                                    "of all glycoforms, which makes the test both more reliable and more powerful.\n\n"
                                    "**Scale:** limma works on log2 abundances (zeros are treated as missing). The other "
                                    "tests run on raw abundances, so their p-values and log2FC can differ from limma's.\n\n"
                                    "**q-value** is the Benjamini-Hochberg adjusted p-value (false discovery rate). "
                                    "q < 0.05 means about 5% of the features you call significant are expected to be false."
                                )

                            nonempty = [(lbl, cols) for lbl, cols in zip(group_labels, group_samples) if cols]
                            nonempty_idx = [i for i, cols in enumerate(group_samples) if cols]
                            can_pairwise = len(nonempty) >= 2
                            can_anova    = (len(nonempty) == 3) and (n_groups == 3)
                            labels_unique = len({lbl for lbl, _ in nonempty}) == len(nonempty)

                            # No st.form here on purpose: inside a form the options below
                            # would not update when the design is changed.
                            DESIGN_TWO = "Two groups (pairwise)"
                            DESIGN_THREE = "All three groups (limma F-test or ANOVA)"
                            comparison_choices = [
                                DESIGN_TWO if can_pairwise else DESIGN_TWO + " — need ≥2 non-empty groups",
                                DESIGN_THREE if can_anova else DESIGN_THREE + " — need exactly 3 non-empty groups",
                            ]
                            # Labels change when groups change; drop a stale stored choice.
                            if ss.get("stat_design") not in comparison_choices:
                                ss.pop("stat_design", None)
                            comparison_type = st.radio(
                                "Select experimental design:",
                                comparison_choices,
                                # With three groups defined, start on the three-group design.
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
                                    st.warning("Please define at least two non-empty groups in the Data Filtering tab.")
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
                                            "Classical test on raw abundances. Only says whether any group differs.",
                                        ],
                                        help=TEST_HELP[LIMMA_F],
                                    )
                                else:
                                    st.warning("Define exactly three non-empty groups in Data Filtering & Group Specification.")

                            run_button = st.button(
                                "Run Analysis",
                                type="primary",
                                key="btn_run_diff",
                                help="Execute the selected statistical test to generate significance metrics (p-values, q-values, fold changes).",
                            )

                            if run_button:
                                ss["filter_open"] = False
                                # A new test makes any earlier filtered subset stale.
                                ss["filtered_df"] = None
                                ss["filtered_ready"] = False

                                if not labels_unique:
                                    st.error("Group labels must be different from each other. Rename the groups in Data Filtering & Group Specification.")

                                elif is_pairwise_design:
                                    if not can_pairwise or not selected_pair:
                                        st.error("Define at least two non-empty groups and pick a comparison.")
                                    else:
                                        i, j = pairs[selected_pair]
                                        if ctrl_idx == j:
                                            c_i, t_i = j, i
                                        else:
                                            c_i, t_i = i, j

                                        control_cols, test_cols = group_samples[c_i], group_samples[t_i]
                                        control_lab, test_lab = group_labels[c_i], group_labels[t_i]

                                        try:
                                            with st.spinner(f"Running {specific_test}..."):
                                                if specific_test == LIMMA_T:
                                                    diffexp_df = compute_diffexp_limma(
                                                        df=merged_df,
                                                        group_cols=[list(c) for c in group_samples],
                                                        group_labels=list(group_labels),
                                                        control_idx=c_i,
                                                        test_idx=t_i,
                                                        use_all_groups=limma_all_groups,
                                                    )
                                                else:
                                                    diffexp_df = compute_diffexp(
                                                        df=merged_df,
                                                        control_cols=control_cols,
                                                        test_cols=test_cols,
                                                        control_label=control_lab,
                                                        test_label=test_lab,
                                                        test_type=specific_test
                                                    )
                                        except Exception as e:
                                            st.error(f"The statistical test could not be run: {e}")
                                        else:
                                            method_label = f"{specific_test}: {test_lab} vs {control_lab}"
                                            if specific_test == LIMMA_T and limma_all_groups and len(nonempty) > 2:
                                                method_label += " (variance from all groups)"

                                            ss["pairwise_control_label"] = control_lab
                                            ss["pairwise_test_label"] = test_lab
                                            ss["pairwise_control_cols"] = control_cols
                                            ss["pairwise_test_cols"] = test_cols

                                            ss["stat_method_label"] = method_label
                                            ss["diffexp_df"] = diffexp_df
                                            ss["diff_done"]  = True

                                else:
                                    if not can_anova:
                                        st.error("Provide exactly three non-empty groups for ANOVA.")
                                    else:
                                        try:
                                            with st.spinner(f"Running {multigroup_test}..."):
                                                if multigroup_test == LIMMA_F:
                                                    diffexp_df = compute_diffexp_limma_f(
                                                        df=merged_df,
                                                        group_cols=[list(c) for c in group_samples],
                                                        group_labels=list(group_labels),
                                                        control_idx=ctrl_idx,
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
                                                        run_tukey=False
                                                    )
                                        except Exception as e:
                                            st.error(f"The statistical test could not be run: {e}")
                                        else:
                                            ss["stat_method_label"] = f"{multigroup_test}: {', '.join(group_labels[:3])}"
                                            ss["diffexp_df"] = diffexp_df
                                            ss["diff_done"]  = True


                    if ss.get("diff_done") and df_is_nonempty(ss.get("diffexp_df")):
                        _res = ss["diffexp_df"]
                        _n_sig = int((pd.to_numeric(_res.get("q-value"), errors="coerce") < 0.05).sum()) if "q-value" in _res.columns else 0
                        st.success(
                            f"**{ss.get('stat_method_label', 'Differential analysis complete')}**: "
                            f"{len(_res):,} glycoforms tested, {_n_sig:,} with q < 0.05."
                        )
                        if "AveExpr" in _res.columns:
                            if "F" in _res.columns:
                                st.caption(
                                    "limma columns: **AveExpr** = mean log2 abundance; **F** = moderated F statistic; "
                                    "**p-value / q-value** = does any group differ; **log2FC (B vs A)** and "
                                    "**q-value (B vs A)** = each pair of groups (moderated t-test). "
                                    "Group-name columns show the mean raw abundance."
                                )
                            else:
                                st.caption(
                                    "limma columns: **log2FC** = mean log2(test) − mean log2(reference); "
                                    "**AveExpr** = mean log2 abundance; **t** = moderated t statistic. "
                                    "Group-name columns show the mean raw abundance."
                                )
                        show_grid(ss["diffexp_df"], page_size=15, height=400)
                        
                        plot_df = ss["diffexp_df"]
                        
                        st.divider()
                        st.markdown("#### Filter & Download Results")
                        
                        with st.expander("Download Full Dataset (Unfiltered)", expanded=False):
                            dl_col1, dl_col2, spacer = st.columns([1.5, 1.5, 7])
                            
                            buffer_csv_diff = BytesIO()
                            ss["diffexp_df"].to_csv(buffer_csv_diff, index=False)
                            
                            with dl_col1:
                                add_hosted_safe_file_download_button(
                                    label="Download CSV",
                                    data=buffer_csv_diff.getvalue(),
                                    file_name="diffexp_results_glycoexplorer.csv",
                                    key="download_ge_diff_csv",
                                    use_container_width=True,
                                )

                            buffer_diff_excel = BytesIO()
                            with pd.ExcelWriter(buffer_diff_excel, engine='openpyxl') as writer:
                                ss["diffexp_df"].to_excel(writer, index=False, sheet_name='Differential_Results')
                            
                            with dl_col2:
                                add_hosted_safe_file_download_button(
                                    label="Download Excel",
                                    data=buffer_diff_excel.getvalue(),
                                    file_name="diffexp_results_glycoexplorer.xlsx",
                                    key="download_ge_diff_xlsx",
                                    use_container_width=True,
                                )

                        with st.container(border=True):
                            st.markdown("**Filter Significant Features**")
                            diff_df = ss["diffexp_df"]
                            is_anova = "log2FC" not in diff_df.columns

                            col1, col2 = st.columns(2)
                            with col1:
                                metric_opts = ["q-value", "p-value"] if "q-value" in diff_df.columns else ["p-value"]
                                sig_metric = st.selectbox(
                                    "Significance Metric",
                                    metric_opts,
                                    key="filter_metric",
                                    help="q-value (false discovery rate) is recommended. Raw p-values give many false hits when thousands of glycoforms are tested."
                                )
                                alpha_thr = st.number_input(
                                    f"{sig_metric} cutoff (<)",
                                    value=0.05,
                                    step=0.01,
                                    format="%.4f",
                                    key="filter_alpha"
                                )
                            with col2:
                                if not is_anova:
                                    fc_thr = st.number_input(
                                        "|log2FC| cutoff (>)",
                                        value=1.0,
                                        step=0.5,
                                        format="%.2f",
                                        key="filter_fc"
                                    )
                                else:
                                    fc_thr = 0.0
                                    st.info("Fold-change filtering is not applicable for ANOVA.")

                            if st.button("Apply Filters & View Subset", type="primary", key="btn_apply_filter"):
                                mask = diff_df[sig_metric] < alpha_thr
                                if not is_anova:
                                    mask = mask & (diff_df["log2FC"].abs() > fc_thr)
                                
                                ss["filtered_df"] = diff_df[mask].copy()
                                ss["filtered_ready"] = True
                                
                            if ss.get("filtered_ready") and ss.get("filtered_df") is not None:
                                f_df = ss["filtered_df"]
                                st.success(f"Retained **{len(f_df)}** significant features matching the criteria.")
                                
                                if not f_df.empty:
                                    show_grid(f_df, page_size=10, height=300)
                                    
                                    f_col1, f_col2, _ = st.columns([1.5, 1.5, 7])
                                    
                                    buf_f_csv = BytesIO()
                                    f_df.to_csv(buf_f_csv, index=False)
                                    with f_col1:
                                        add_hosted_safe_file_download_button(
                                            label="Download CSV",
                                            data=buf_f_csv.getvalue(),
                                            file_name="filtered_differential_results.csv",
                                            key="download_f_csv",
                                            use_container_width=True,
                                        )

                                    buf_f_xlsx = BytesIO()
                                    with pd.ExcelWriter(buf_f_xlsx, engine='openpyxl') as writer:
                                        f_df.to_excel(writer, index=False, sheet_name='Filtered_Results')
                                    with f_col2:
                                        add_hosted_safe_file_download_button(
                                            label="Download Excel",
                                            data=buf_f_xlsx.getvalue(),
                                            file_name="filtered_differential_results.xlsx",
                                            key="download_f_xlsx",
                                            use_container_width=True,
                                        )

        # ------------------- TAB: Results Visualizations -------------------
        with tabs[1]:
            if df_is_empty(ss.get("diffexp_df")):
                st.warning("Please complete the Differential Analysis first.")
            else:
                diffexp_df    = ss.get("diffexp_df", pd.DataFrame())
                filtered_df   = ss.get("filtered_df", pd.DataFrame())
                group_samples = ss.get("group_samples", [])
                group_labels  = ss.get("group_labels", [])

                is_anova = "log2FC" not in diffexp_df.columns

                ctrl_idx = ss.get("control_group_idx", 0)
                ctrl_idx = min(max(ctrl_idx, 0), len(group_samples) - 1) if group_samples else 0

                control_group = group_samples[ctrl_idx] if len(group_samples) > ctrl_idx else []
                control_label = group_labels[ctrl_idx] if len(group_labels) > ctrl_idx else "Reference"

                other_indices = [i for i in range(len(group_samples)) if i != ctrl_idx]
                test_a_group  = group_samples[other_indices[0]] if len(other_indices) > 0 else []
                test_a_label  = group_labels[other_indices[0]] if len(other_indices) > 0 else "Test"
                test_b_group  = group_samples[other_indices[1]] if len(other_indices) > 1 else []
                test_b_label  = group_labels[other_indices[1]] if len(other_indices) > 1 else "Test2"

                if not is_anova and ss.get("pairwise_control_cols") and ss.get("pairwise_test_cols"):
                    control_group = ss["pairwise_control_cols"]
                    test_a_group  = ss["pairwise_test_cols"]
                    control_label = ss.get("pairwise_control_label", control_label)
                    test_a_label  = ss.get("pairwise_test_label", test_a_label)

                plot_df = diffexp_df
                left, right = st.columns([0.42, 0.58], gap="large")

                for key in ["show_volcano", "show_waterfall", "show_box", "show_heatmap", "show_box_gene", "show_lollipop", "show_class_breakdown"]:
                    ss.setdefault(key, False)

                with left:
                    with st.container(border=True):

                        plot_modes_pairwise = ["Volcano", "Waterfall", "Box plot", "Heatmap", "Box plot (Gene)", "Lollipop plot", "Glycan class breakdown"]
                        plot_modes_anova    = ["Box plot", "Heatmap", "Box plot (Gene)", "Lollipop plot", "Glycan class breakdown"]
                        modes = plot_modes_anova if is_anova else plot_modes_pairwise

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
                                show_volcano=True,  show_waterfall=False, show_box=False, show_heatmap=False,
                                show_box_gene=False, show_lollipop=False
                            ),
                            "Waterfall": dict(
                                show_volcano=False, show_waterfall=True, show_box=False, show_heatmap=False,
                                show_box_gene=False, show_lollipop=False
                            ),
                            "Box plot": dict(
                                show_volcano=False, show_waterfall=False, show_box=True,  show_heatmap=False,
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
                            
                            sig_col = st.radio(
                                "Significance metric",
                                ["p-value", "q-value"],
                                key="volcano_sig_col",
                                horizontal=True
                            )
                            sig_threshold = st.slider(
                                f"{sig_col} threshold",
                                1e-5, 0.1, 0.05,
                                key="volcano_sig_threshold",
                                format="%.5f"
                            )
                            log2fc_threshold = st.slider("log2FC threshold", 0.5, 3.0, 1.0, key="volcano_logfc")
                            top_n = st.slider("Top N markers to annotate", 1, 50, 10, key="volcano_topn")
                            top_mode = st.radio(
                                "Highlight Top Hits:",
                                options=["all", "up", "down", "up-down"],
                                index=3,
                                key="volcano_topmode",
                                horizontal=True
                            )

                        if ss.get("show_waterfall", False):
                            st.markdown("**Waterfall plot options**")
                            metric_opts = ["q-value", "p-value"] if "q-value" in plot_df.columns else ["p-value"]

                            wfl_sig_col = st.radio(
                                "Significance metric",
                                metric_opts,
                                key="wfl_sig_col",
                                horizontal=True,
                            )
                            wfl_alpha_thr = st.slider(
                                f"{wfl_sig_col} threshold",
                                1e-5, 0.1, 0.05,
                                key="wfl_alpha_thr",
                                format="%.5f",
                            )
                            wfl_log2fc_thr = st.slider(
                                "log2FC threshold",
                                0.5, 3.0, 1.0,
                                key="wfl_log2fc_thr",
                            )
                            wfl_top_n = st.slider(
                                "Number of top up/down features to show",
                                3, 25, 5,
                                key="wfl_top_n",
                            )

                            ss["waterfall_sig_col"] = wfl_sig_col
                            ss["waterfall_alpha"] = wfl_alpha_thr
                            ss["waterfall_logfc"] = wfl_log2fc_thr
                            ss["waterfall_topn"] = wfl_top_n

                            st.caption(
                                "Waterfall is rendered full-width below so the structural table has enough room."
                            )

                        if ss.get("show_box", False):
                            st.markdown("**Box plot options**")
                            sig_col_box = st.radio(
                                "Significance metric",
                                ["p-value", "q-value"],
                                key="box_sig_col",
                                horizontal=True
                            )
                            top_n_box = st.slider("Top N markers to display", 1, 50, 10, key="box_topn")
                            
                            if not is_anova:
                                top_mode_box = st.radio(
                                    "Top Hits Direction:",
                                    options=["all", "up", "down", "up-down"],
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

                        if ss.get("show_heatmap", False):
                            st.markdown("**Heatmap options**")
                            sig_col_heatmap = st.radio(
                                "Significance metric",
                                ["p-value", "q-value"],
                                key="heatmap_sig_col",
                                horizontal=True
                            )
                            top_n_heatmap = st.slider(
                                "Top N markers to display",
                                1, max(10, len(plot_df)), 25,
                                key="heatmap_topn"
                            )
                            normalize_heatmap = st.radio(
                                "Normalize data",
                                ["Z-score", "Raw Abundance"],
                                key="heatmap_norm",
                                horizontal=True
                            )
                            cluster_option = st.selectbox(
                                "Clustering",
                                ["Hierarchical", "K-Means", "None"],
                                key="heatmap_cluster"
                            )
                            
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

                        if ss.get("show_box_gene", False):
                            st.markdown("**Box plots for selected Gene IDs**")
                            genes_text = st.text_area(
                                "Enter Gene IDs to plot (one per line or comma-separated)",
                                key="box_gene_ids",
                                placeholder="e.g.\nACTB\nGAPDH, MUC1\nTP53"
                            )
                            user_genes = [g.strip() for g in re.split(r"[,\n;\t ]+", genes_text) if g.strip()]
                            ss["box_gene_list"] = user_genes
                            ss.setdefault("box_gene_plot", False)

                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.button(
                                    "Plot selected genes",
                                    key="btn_plot_genes",
                                    type="primary",
                                    on_click=_plot_genes_cb,
                                    disabled=(len(user_genes) == 0),
                                    help="Generate comparative box plots for the specified Gene IDs."
                                )
                            with col_b:
                                st.button(
                                    "Clear selection",
                                    key="btn_clear_genes",
                                    on_click=_clear_genes_cb,
                                    help="Clear the entered Gene IDs and remove the generated plots."
                                )

                        if ss.get("show_class_breakdown", False):
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
                            gcb_metric_opts = ["q-value", "p-value"] if "q-value" in plot_df.columns else ["p-value"]
                            st.radio("Significance metric", gcb_metric_opts, key="gcb_sig_col", horizontal=True)
                            st.number_input("Cutoff (<)", value=0.05, min_value=0.0001, max_value=1.0,
                                            step=0.01, format="%.4f", key="gcb_alpha")
                            if not is_anova:
                                st.number_input("|log2FC| cutoff (>)", value=1.0, min_value=0.0, step=0.25,
                                                format="%.2f", key="gcb_fc")

                        if ss.get("show_lollipop", False):
                            st.markdown("**Lollipop plot options**")                            
                            
                            metric_opts = ["q-value", "p-value"] if "q-value" in plot_df.columns else ["p-value"]
                            ss["lollipop_sig_col"] = st.radio(
                                "Significance metric",
                                metric_opts,
                                key="lollipop_sig_col_ui",
                                horizontal=True
                            )
                            ss["lollipop_show_legend"] = st.checkbox(
                                "Distinguish glycans with shapes and legend",
                                value=False,
                                key="lollipop_legend_ui"
                            ) 
                            
                            if "Genes" in plot_df.columns:
                                all_genes_raw = plot_df["Genes"].dropna().astype(str).tolist()
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
                                    df=plot_df,
                                    sig_threshold=ss["volcano_sig_threshold"],
                                    log2fc_threshold=ss["volcano_logfc"],
                                    sig_col=ss["volcano_sig_col"],
                                    log2fc_col="log2FC",
                                    control_label=control_label,
                                    test_label=test_a_label,
                                )
                            else:
                                create_volcano_plot(
                                    df=plot_df,
                                    top_n=ss["volcano_topn"],
                                    sig_threshold=ss["volcano_sig_threshold"],
                                    log2fc_threshold=ss["volcano_logfc"],
                                    label_col="Genes",
                                    sig_col=ss["volcano_sig_col"],
                                    log2fc_col="log2FC",
                                    control_label=control_label,
                                    test_label=test_a_label,
                                    glycan_image_dir=glycan_image_dir,
                                    top_mode=ss.get("volcano_topmode", "up-down"),
                                )

                    elif ss.get("show_waterfall", False):
                        st.info("Waterfall output is shown full-width below this section.")

                    elif ss.get("show_box", False):
                        pval_col = "p-value" if ss["box_sig_col"] == "p-value" else "q-value"
                        if is_anova:
                            create_box_plots_anova(
                                df=plot_df,
                                pval_col=pval_col,
                                top_n=ss["box_topn"],
                                group_samples=group_samples,
                                group_labels=group_labels,
                                glycan_image_dir=glycan_image_dir,
                                quant_mode="TMT",
                                group_colors=ss.get("box_anova_colors")
                            )
                        else:
                            create_box_plots(
                                df=plot_df,
                                pval_col=pval_col,
                                top_n=ss["box_topn"],
                                control_cols=control_group,
                                test_cols=test_a_group,
                                control_label=control_label,
                                test_label=test_a_label,
                                glycan_image_dir=glycan_image_dir,
                                control_color=ss.get("box_color_ctrl", "#1f77b4"),
                                test_color=ss.get("box_color_test", "#d62728"),
                                quant_mode="TMT",
                                top_mode=ss.get("box_topmode", "up-down"),
                                log2fc_col="log2FC"
                            )

                    elif ss.get("show_heatmap", False):
                        pval_col_heatmap = ss["heatmap_sig_col"]
                        if is_anova:
                            create_heatmap_anova(
                                df=plot_df,
                                pval_col=pval_col_heatmap,
                                top_n=ss["heatmap_topn"],
                                group_samples=group_samples,
                                group_labels=group_labels,
                                normalize=ss["heatmap_norm"],
                                cluster_option=ss["heatmap_cluster"],
                                group_colors=ss.get("heatmap_anova_colors"),
                                quant_mode="TMT"
                            )
                        else:
                            create_heatmap(
                                df=plot_df,
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
                                quant_mode="TMT"
                            )

                    elif ss.get("show_box_gene", False):
                        user_genes = ss.get("box_gene_list", [])
                        if not ss.get("box_gene_plot", False):
                            st.info("Enter Gene IDs on the left and click Plot selected genes.")
                        else:
                            if not user_genes:
                                st.info("Please enter at least one Gene ID.")
                            elif "Genes" not in plot_df.columns:
                                st.error("The dataset is missing a 'Genes' column.")
                            else:
                                user_genes_lower = {g.lower() for g in user_genes}
                                genes_series = plot_df["Genes"].astype(str)
                                mask = genes_series.str.lower().isin(user_genes_lower)
                                sub_df = plot_df.loc[mask].copy()

                                if sub_df.empty:
                                    st.warning("No matching Gene IDs in the current dataset.")
                                else:
                                    pval_col = "p-value" if "p-value" in sub_df.columns else ("q-value" if "q-value" in sub_df.columns else None)
                                    if not pval_col:
                                        st.error("Neither 'p-value' nor 'q-value' column found.")
                                    else:
                                        top_n_rows = len(sub_df)

                                        if is_anova:
                                            if len(group_samples) >= 3:
                                                create_box_plots_top_n_anova(
                                                    df=sub_df, pval_col=pval_col, top_n=top_n_rows,
                                                    group1_cols=control_group, group2_cols=test_a_group, group3_cols=test_b_group,
                                                    group1_label=control_label, group2_label=test_a_label, group3_label=test_b_label,
                                                    glycan_image_dir=glycan_image_dir,
                                                )
                                            else:
                                                st.warning("ANOVA requires three groups.")
                                        else:
                                            create_box_plots_top_n(
                                                df=sub_df, pval_col=pval_col, top_n=top_n_rows,
                                                control_cols=control_group, test_cols=test_a_group,
                                                control_label=control_label, test_label=test_a_label,
                                                glycan_image_dir=glycan_image_dir,
                                            )

                    elif ss.get("show_class_breakdown", False):
                        render_glycan_class_breakdown(
                            plot_df=plot_df,
                            is_anova=is_anova,
                            sig_col=ss.get("gcb_sig_col", "q-value" if "q-value" in plot_df.columns else "p-value"),
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
                            missing = [c for c in required if c not in plot_df.columns]
                            
                            if missing:
                                st.error("Missing required columns for lollipop plot: " + ", ".join(missing))
                            else:
                                df_for_plot = plot_df.copy()
                                gene = genes[0]
                                
                                show_legend = ss.get("lollipop_show_legend", False)
                                
                                if show_legend:
                                    create_lollipop_legend(df_for_plot, gene, sig_col=sig_metric)
                                else:
                                    create_lollipop(df_for_plot, gene, sig_col=sig_metric)


    # ====================== FINAL ANALYSIS REPORT ======================

                if ss.get("show_waterfall", False):
                    st.markdown("---")
                    if is_anova:
                        st.info("Waterfall plot is only available for pairwise results with log2FC.")
                    else:
                        render_ge_structural_waterfall(
                            plot_df=plot_df,
                            glycan_image_dir=glycan_image_dir,
                            sig_col=ss.get("wfl_sig_col", "q-value" if "q-value" in plot_df.columns else "p-value"),
                            alpha_thr=ss.get("wfl_alpha_thr", 0.05),
                            log2fc_thr=ss.get("wfl_log2fc_thr", 1.0),
                            top_n=ss.get("wfl_top_n", 5),
                        )

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
                    - **Experimental Design:** Cohort definitions, sample mapping, reference group, and batch status.
                    - **Data Integrity:** PCA, correlation matrices, sample clustering, and glycan summaries.
                    - **Differential Analysis:** Top up/down glycopeptides, gene-level summaries, waterfall top features, and statistical tables.
                    - **Export Bundle:** A ZIP archive containing the PDF, report images, and tabular outputs.
                    """
                )

            with col_action:
                st.write("")
                st.write("")

                # ---------------------------
                # Helper functions
                # ---------------------------
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

                def _build_report_zip():
                    """
                    Creates a ZIP bundle with:
                    - PDF report (if available)
                    - key data tables
                    - sample mapping table
                    - generated plot images from available session-state figures
                    """
                    zip_buffer = BytesIO()

                    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                        
                        if ss.get("final_pdf_bytes") is not None:
                            zf.writestr(
                                "report/final_report.pdf",
                                ss["final_pdf_bytes"]
                            )

                        if ss.get("merged_df") is not None and not ss["merged_df"].empty:
                            zf.writestr(
                                "tables/merged_data.csv",
                                ss["merged_df"].to_csv(index=False)
                            )

                        if ss.get("diffexp_df") is not None and not ss["diffexp_df"].empty:
                            zf.writestr(
                                "tables/differential_results.csv",
                                ss["diffexp_df"].to_csv(index=False)
                            )

                        if ss.get("filtered_df") is not None and not ss["filtered_df"].empty:
                            zf.writestr(
                                "tables/filtered_results.csv",
                                ss["filtered_df"].to_csv(index=False)
                            )

                        # Sample mapping metadata
                        if (
                            ss.get("global_sample_cols")
                            and ss.get("global_group_samples") is not None
                            and ss.get("global_group_labels") is not None
                        ):
                            rows = []
                            group_samples = ss.get("global_group_samples", [])
                            group_labels = ss.get("global_group_labels", [])
                            control_idx = ss.get("global_control_group_idx", 0)
                            batch_dict  = ss.get("batch_dict", {})

                            def clean_label(name: str) -> str:
                                s = str(name)
                                s = re.sub(r"^\s*Abundance\s*:\s*", "", s)
                                if ":" in s:
                                    s = s.split(":", 1)[-1]
                                return s.strip()

                            for col in ss.get("global_sample_cols", []):
                                grp_name = "Unassigned"
                                grp_num = None
                                for i, cols in enumerate(group_samples):
                                    if col in (cols or []):
                                        grp_name = group_labels[i] if i < len(group_labels) else f"Group{i+1}"
                                        grp_num = i + 1
                                        break
                                        
                                batch_name = "Unassigned"
                                for b_name, b_cols in batch_dict.items():
                                    if col in b_cols:
                                        batch_name = b_name
                                        break

                                rows.append({
                                    "Column": col,
                                    "Sample": clean_label(col),
                                    "Group": grp_name,
                                    "Batch": batch_name if ss.get("batch_corrected") else "N/A",
                                    "Group_Number": grp_num,
                                    "Is_Reference": (grp_num is not None and (grp_num - 1) == control_idx)
                                })

                            if rows:
                                sample_map_df = pd.DataFrame(rows)
                                zf.writestr(
                                    "tables/sample_metadata_mapping.csv",
                                    sample_map_df.to_csv(index=False)
                                )

                        plot_keys = [
                            "report_pca_fig", "report_corr_fig", "report_density_fig",
                            "report_sample_clustering_fig", "report_overview_heatmap_fig", "report_qc_heatmap_fig",
                            "report_glycan_class_fig",
                            "report_volcano_fig", "report_waterfall_fig", "report_ma_fig",
                            "report_heatmap_fig", "report_boxplot_fig", "report_lollipop_fig"
                        ]

                        for plot_key in plot_keys:
                            if plot_key in ss and ss[plot_key] is not None:
                                safe_name = plot_key.replace("report_", "").replace("_fig", "")
                                _add_fig_to_zip(zf, "plots", safe_name, ss[plot_key])

                        manifest_lines = [
                            "GlycoExplorer Report Bundle",
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

                # ---------------------------
                # Main report actions
                # ---------------------------
                if "final_pdf_bytes" not in ss:
                    if st.button(
                        "Compile PDF Report",
                        width='stretch',
                        type="primary",
                        help="Gather all generated plots and statistics into a single formatted PDF document."
                    ):
                        with st.spinner("Compiling executive summary, plots, tables, and PDF report..."):
                            try:
                                from ge.report_utility import generate_final_pdf

                                ss["final_pdf_bytes"] = generate_final_pdf()
                                ss["final_report_zip_bytes"] = _build_report_zip()
                                ss["final_report_timestamp"] = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                                st.rerun()

                            except Exception as e:
                                st.error(f"Failed to generate report: {e}")

                else:
                    st.success("Report generated successfully.")

                    # Keep timestamp stable during the same generated report session.
                    if "final_report_timestamp" not in ss:
                        ss["final_report_timestamp"] = datetime.datetime.now().strftime("%Y%m%d_%H%M")

                    timestamp = ss["final_report_timestamp"]

                    pdf_filename = f"GlycoExplorer_TMT_Report_{timestamp}.pdf"
                    zip_filename = f"GlycoExplorer_TMT_Report_Bundle_{timestamp}.zip"

                    # Make sure PDF bytes are real bytes.
                    pdf_bytes = ss.get("final_pdf_bytes")
                    if isinstance(pdf_bytes, BytesIO):
                        pdf_bytes = pdf_bytes.getvalue()

                    if pdf_bytes is None:
                        st.error("PDF report bytes are missing. Please regenerate the report.")
                    else:
                        st.caption(f"PDF size: {len(pdf_bytes) / 1024 / 1024:.2f} MB")

                    # Build ZIP if missing.
                    if "final_report_zip_bytes" not in ss or ss.get("final_report_zip_bytes") is None:
                        try:
                            ss["final_report_zip_bytes"] = _build_report_zip()
                        except Exception as e:
                            ss["final_report_zip_bytes"] = None
                            st.warning(f"PDF is ready, but ZIP bundle could not be prepared: {e}")

                    zip_bytes = ss.get("final_report_zip_bytes")
                    if isinstance(zip_bytes, BytesIO):
                        zip_bytes = zip_bytes.getvalue()

                    if zip_bytes is not None:
                        st.caption(f"ZIP size: {len(zip_bytes) / 1024 / 1024:.2f} MB")

                    # ------------------------------------------------------------------
                    # Desktop-safe download layout
                    # ------------------------------------------------------------------
                    dl_col1, dl_col2 = st.columns(2)

                    with dl_col1:
                        if pdf_bytes is not None:
                            add_hosted_safe_file_download_button(
                                label="Download PDF",
                                data=pdf_bytes,
                                file_name=pdf_filename,
                                key="download_pdf_report",
                                use_container_width=True,
                            )

                    with dl_col2:
                        if zip_bytes is not None:
                            add_hosted_safe_file_download_button(
                                label="Download ZIP",
                                data=zip_bytes,
                                file_name=zip_filename,
                                key="download_zip_report",
                                use_container_width=True,
                            )
                        else:
                            st.warning("ZIP bundle is not available. Try regenerating the report.")

                    st.markdown("<br>", unsafe_allow_html=True)

                    if st.button(
                        "Regenerate Report",
                        width='stretch',
                        help="Clear the current report and recompile it with any updated parameters or new plots.",
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