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
    ss["batch_corrected"] = False
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

    st.sidebar.subheader("Glycan Image Directory")
    user_input = st.sidebar.text_input(
        "Enter folder path (optional):",
        value=str(default_dir) if default_dir.exists() else "",
        help="Leave blank to use the default path detected automatically."
    )

    glycan_image_dir = Path(user_input).expanduser().resolve() if user_input else default_dir
    if glycan_image_dir.is_dir():
        st.sidebar.success(f"Glycan image folder found:\n{glycan_image_dir}")
    else:
        st.sidebar.error(f"Folder not found:\n{glycan_image_dir}")
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
            show_grid(merged_display_df, page_size=15, height=450)

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
        help="Perform quality control using PCA, correlation matrix, sample clustering, and glycan distribution analysis."
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
        help="Run Pairwise T-tests or ANOVA and generate Volcano, Box, and Heatmap visualizations."
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
                    apply_batch = st.toggle(
                        "Apply Batch Effect Correction",
                        value=ss.get("enable_batch_correction", False),
                        key="enable_batch_correction"
                    )

                    if apply_batch:
                        batch_method = st.selectbox(
                            "Batch Correction Method",
                            options=["ComBat (Empirical Bayes)", "Median Centering", "Internal Reference Scaling (IRS)"],
                            help="ComBat requires neuroCombat. IRS requires a common pool channel across multiplexes.",
                            key="batch_method_ui"
                        )

                        num_batches = st.radio(
                            "Number of Batches",
                            [2, 3, 4],
                            horizontal=True,
                            key="num_batches_ui"
                        )

                        batch_dict = {}
                        selected_so_far = []

                        for b in range(num_batches):
                            prev_key = f"batch_sel_{b}"
                            prev_selected = _sanitize(ss.get(prev_key, []), batch_allowed_samples)

                            available_for_batch = [
                                c for c in batch_allowed_samples
                                if c not in selected_so_far or c in prev_selected
                            ]

                            prev_selected = _sanitize(prev_selected, available_for_batch)

                            b_cols = st.multiselect(
                                f"Batch {b+1} Samples",
                                options=available_for_batch,
                                default=prev_selected,
                                format_func=fmt,
                                key=prev_key
                            )

                            b_cols = _sanitize(b_cols, batch_allowed_samples)
                            batch_dict[f"Batch {b+1}"] = b_cols
                            selected_so_far = _uniq(selected_so_far + b_cols)

                        assigned_samples = _uniq([c for cols in batch_dict.values() for c in cols])
                        unassigned_batch_samples = [
                            c for c in batch_allowed_samples if c not in assigned_samples
                        ]

                        if unassigned_batch_samples:
                            st.warning(
                                "These grouped samples are not assigned to any batch: "
                                + ", ".join([fmt(c) for c in unassigned_batch_samples])
                            )

                        # Provide IRS Reference Dropdowns if IRS is chosen
                        irs_ref_dict = {}
                        if batch_method == "Internal Reference Scaling (IRS)":
                            nonempty_batches_irs = {k: v for k, v in batch_dict.items() if v}
                            if nonempty_batches_irs:
                                st.markdown("**Select the Reference/Pool Sample for each batch:**")
                                for b_name, b_cols in nonempty_batches_irs.items():
                                    irs_ref_dict[b_name] = st.selectbox(
                                        f"{b_name} Reference Channel",
                                        options=b_cols,
                                        format_func=fmt,
                                        key=f"irs_ref_{b_name}"
                                    )

                        if st.button("Run Batch Correction", type="primary", key="run_batch_correction_btn"):
                            assigned_samples = _uniq([c for cols in batch_dict.values() for c in cols])
                            nonempty_batches = {k: v for k, v in batch_dict.items() if v}

                            duplicated = [
                                c for c in assigned_samples
                                if sum(c in cols for cols in batch_dict.values()) > 1
                            ]

                            if duplicated:
                                st.error(
                                    "A sample cannot be assigned to more than one batch: "
                                    + ", ".join([fmt(c) for c in duplicated])
                                )

                            elif len(nonempty_batches) < 2:
                                st.error("Please assign samples to at least two non-empty batches.")

                            elif set(assigned_samples) != set(batch_allowed_samples):
                                st.error("Please assign every grouped sample to one batch before running correction.")

                            else:
                                try:
                                    # Save raw/pre-correction matrix only once
                                    if ss.get("merged_df_before_batch") is None or not ss.get("batch_corrected", False):
                                        ss["merged_df_before_batch"] = ss["merged_df"].copy()

                                    input_df_for_batch = ss["merged_df_before_batch"].copy()

                                    if batch_method == "ComBat (Empirical Bayes)":
                                        corrected_df = apply_combat_batch_correction(
                                            input_df_for_batch,
                                            nonempty_batches,
                                            batch_allowed_samples
                                        )
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
                                    ss["batch_dict"] = nonempty_batches
                                    ss["batch_corrected"] = True
                                    ss["diffexp_df"] = None
                                    ss["filtered_df"] = None
                                    ss["diff_done"] = False
                                    ss["filtered_ready"] = False

                                    st.success(
                                        f"{batch_method} applied successfully. Downstream analysis will use corrected abundances."
                                    )

                                except ImportError as ie:
                                    st.error(str(ie))
                                except Exception as e:
                                    st.error(f"Batch correction failed: {e}")

                    else:
                        ss["batch_dict"] = {}
                        ss["batch_corrected"] = False
                        ss["merged_df_before_batch"] = None

        with right:
            with st.container(border=True):
                st.markdown("#### Sample Mapping Summary")

                sample_cols = ss.get("global_sample_cols", [])
                group_samples = ss.get("global_group_samples", [])
                group_labels  = ss.get("global_group_labels", [])
                control_idx   = ss.get("global_control_group_idx", 0)
                batch_dict    = ss.get("batch_dict", {})

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

                add_hosted_safe_file_download_button(
                    label="Download sample metadata CSV",
                    data=meta_df.to_csv(index=False),
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
                            "Glycan Classification Summary"
                        ],
                        index=None,
                        key="overview_mode",
                        label_visibility="collapsed"
                    )

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
                    else:
                        st.info("No numeric sample columns detected for abundance density.")

                elif ss.get("run_glycan_class", False):
                    create_glycan_class_summary(df, "GlycanClass")

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

                            nonempty = [(lbl, cols) for lbl, cols in zip(group_labels, group_samples) if cols]
                            can_pairwise = len(nonempty) >= 2
                            can_anova    = (len(nonempty) == 3) and (n_groups == 3)

                            with st.form("run_diff_form"):

                                comparison_choices = []
                                if can_pairwise:
                                    comparison_choices.append("Pairwise")
                                else:
                                    comparison_choices.append("Pairwise — need ≥2 non-empty groups")

                                if can_anova:
                                    comparison_choices.append("Three-group ANOVA")
                                else:
                                    comparison_choices.append("Three-group ANOVA — need exactly 3 non-empty groups")

                                comparison_type = st.radio(
                                    "Select experimental design:",
                                    comparison_choices,
                                    index=0
                                )

                                selected_pair = None
                                pairs = {}
                                specific_test = "Student's t-test (Independent, Equal Variance)"

                                if comparison_type.startswith("Pairwise"):
                                    if not can_pairwise:
                                        st.warning("Please define at least two non-empty groups in the Data Filtering tab.")
                                    else:
                                        labels = [lbl for lbl, _ in nonempty]
                                        pairs = {
                                            f"{labels[i]} vs {labels[j]}": (
                                                nonempty[i][1],
                                                nonempty[j][1],
                                                labels[i],
                                                labels[j]
                                            )
                                            for i in range(len(labels))
                                            for j in range(i + 1, len(labels))
                                        }
                                        
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            selected_pair = st.selectbox("Select pairwise comparison:", list(pairs.keys()))
                                        
                                        with col2:
                                            test_options = [
                                                "Student's t-test (Independent, Equal Variance)",
                                                "Welch's t-test (Independent, Unequal Variance)",
                                                "Paired t-test (Matched Samples)",
                                                "Mann-Whitney U (Non-parametric Independent)",
                                                "Wilcoxon Signed-Rank (Non-parametric Paired)"
                                            ]
                                            specific_test = st.selectbox("Select statistical test:", test_options)
                                else:
                                    selected_pair = "Three-group ANOVA"

                                run_button = st.form_submit_button(
                                    "Run Analysis",
                                    type="primary",
                                    help="Execute the selected statistical test to generate significance metrics (e.g., p-values, q-values, fold changes)."
                                )

                            if run_button:
                                ss["filter_open"] = False

                                if comparison_type.startswith("Pairwise"):
                                    if not can_pairwise or not selected_pair:
                                        st.error("Define at least two non-empty groups and pick a comparison.")
                                    else:
                                        group1, group2, label1, label2 = pairs[selected_pair]
                                        ctrl_label = group_labels[ctrl_idx] if ctrl_idx < len(group_labels) else label1

                                        if ctrl_label == label1:
                                            control_cols, test_cols = group1, group2
                                            control_lab, test_lab = label1, label2
                                        elif ctrl_label == label2:
                                            control_cols, test_cols = group2, group1
                                            control_lab, test_lab = label2, label1
                                        else:
                                            control_cols, test_cols = group1, group2
                                            control_lab, test_lab = label1, label2

                                        diffexp_df = compute_diffexp(
                                            df=merged_df,
                                            control_cols=control_cols,
                                            test_cols=test_cols,
                                            control_label=control_lab,
                                            test_label=test_lab,
                                            test_type=specific_test
                                        )

                                        ss["pairwise_control_label"] = control_lab
                                        ss["pairwise_test_label"] = test_lab
                                        ss["pairwise_control_cols"] = control_cols
                                        ss["pairwise_test_cols"] = test_cols
                                        
                                        ss["diffexp_df"] = diffexp_df
                                        ss["diff_done"]  = True
                                        st.success(f"Differential analysis complete: {selected_pair}")

                                else:
                                    if not can_anova:
                                        st.error("Provide exactly three non-empty groups for ANOVA.")
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
                                        
                                        ss["diffexp_df"] = diffexp_df
                                        ss["diff_done"]  = True
                                        st.success(f"Differential analysis complete: {selected_pair}")


                    if ss.get("diff_done") and df_is_nonempty(ss.get("diffexp_df")):
                        st.success("Differential analysis complete. Results available below.")
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
                                    key="filter_metric"
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

                for key in ["show_volcano", "show_waterfall", "show_box", "show_heatmap", "show_box_gene", "show_lollipop"]:
                    ss.setdefault(key, False)

                with left:
                    with st.container(border=True):

                        plot_modes_pairwise = ["Volcano", "Waterfall", "Box plot", "Heatmap", "Box plot (Gene)", "Lollipop plot"]
                        plot_modes_anova    = ["Box plot", "Heatmap", "Box plot (Gene)", "Lollipop plot"]
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
                        }
                        
                        for k in ["show_volcano","show_waterfall","show_box","show_heatmap","show_box_gene","show_lollipop"]:
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