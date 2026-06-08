#!/usr/bin/env python3
"""
Glyco-Proteome Integrator (GPI) — biologically safer integration workflow
----------------------------------------------------------------------------
"""

__author__ = "Saurabh Baheti: baheti.saurabh@mayo.edu"

import io
import re
import uuid
import math
import zipfile
import base64
import datetime
import html as html_lib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import plotly.express as px
from scipy.stats import pearsonr, spearmanr, fisher_exact
from io import BytesIO
# FIX: removed duplicate `import base64` that was here
import tempfile
from PIL import Image

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    from faq.faq_GPI import faq_section
except Exception:
    def faq_section():
        st.info("FAQ module was not found. The main GPI workflow is still available.")

# -----------------------------------------------------------------------------
# Global style
# -----------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Nimbus Sans", "DejaVu Sans"],
    "font.weight": "bold",
    "font.size": 11,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
sns.set_style("whitegrid")

try:
    from upsetplot import UpSet, from_contents
    HAVE_UPSET = True
except Exception:
    HAVE_UPSET = False

try:
    st.set_page_config(page_title="Glyco-Proteome Integrator | Mayo Clinic", layout="wide")
except Exception:
    pass

ss = st.session_state

# -----------------------------------------------------------------------------
# CSS
# -----------------------------------------------------------------------------
def inject_custom_css():
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {padding-top: 1.6rem !important; max-width: 1700px;}
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 12px !important;
            border: 1px solid #e0e6ed !important;
            background-color: #ffffff;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02), 0 4px 12px rgba(0, 0, 0, 0.04) !important;
            transition: all 0.25s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
            padding: 10px !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05), 0 12px 24px rgba(0, 90, 156, 0.10) !important;
            border-color: #005a9c !important;
        }
        h1, h2, h3, h4 {
            color: #005a9c !important;
            font-weight: 600 !important;
            letter-spacing: -0.3px !important;
        }
        .subtitle-text {
            color: #5f6368 !important;
            font-size: 1.0rem;
            font-weight: 400;
        }
        .small-note {
            color: #5f6368 !important;
            font-size: 0.88rem;
        }
        .stButton>button {
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        .stButton>button:hover {
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
            transform: translateY(-1px);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_custom_css()

# -----------------------------------------------------------------------------
# State defaults
# -----------------------------------------------------------------------------
for _k in [
    "show_overview", "show_targeted", "show_advanced", "show_export",
    "upload_sig", "gpi_pdf_bytes", "gpi_zip_bytes"
]:
    if _k not in ss:
        ss[_k] = False if _k.startswith("show_") else None

def _reset_module_flags():
    for key in ["show_overview", "show_targeted", "show_advanced", "show_export"]:
        ss[key] = False

# -----------------------------------------------------------------------------
# Image Rendering Utilities (FIXED)
# -----------------------------------------------------------------------------
def find_glycan_image(glycan_str, img_dir):
    if not glycan_str or pd.isna(glycan_str): return None
    g_str = str(glycan_str).strip()
    
    candidates = [
        f"{g_str}.png",
        f"{g_str.replace(' ', '_')}.png",
        f"{g_str.replace(' ', '')}.png",
        f"Glycan_{g_str.replace(' ', '_')}.png"
    ]
    for c in candidates:
        p = img_dir / c
        if p.exists() and p.is_file():
            return p
    return None

def image_to_base64(img_path):
    if img_path and img_path.exists():
        with open(img_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{encoded}"
    return ""

def format_structural_html_table(df, img_dir, gene_col, pep_col, lfc_col, metric_col, metric_label):
    html = [
        "<style>",
        ".gpi-table { width: 100%; border-collapse: collapse; table-layout: fixed; font-family: sans-serif; font-size: 0.86em; }",
        ".gpi-table th { background-color: #f8f9fa; border-bottom: 2px solid #dee2e6; padding: 8px; text-align: center; color: #495057; }",
        ".gpi-table td { border-bottom: 1px solid #e9ecef; padding: 8px; text-align: center; vertical-align: middle; overflow-wrap: anywhere; }",
        ".gpi-table th:nth-child(1), .gpi-table td:nth-child(1) { width: 6%; }",
        ".gpi-table th:nth-child(2), .gpi-table td:nth-child(2) { width: 12%; }",
        ".gpi-table th:nth-child(3), .gpi-table td:nth-child(3) { width: 48%; }",
        ".gpi-table th:nth-child(4), .gpi-table td:nth-child(4) { width: 11%; }",
        ".gpi-table th:nth-child(5), .gpi-table td:nth-child(5) { width: 10%; }",
        ".gpi-table th:nth-child(6), .gpi-table td:nth-child(6) { width: 13%; }",
        ".gpi-point { font-size: 1.1em; font-weight: bold; color: #111; }",
        ".gpi-peptide { font-family: monospace; font-size: 0.92em; color: #333; margin-top: 4px; line-height: 1.25; }",
        ".gpi-annotation { color: #555; font-size: 0.82em; margin-top: 3px; margin-bottom: 3px; font-weight: 600; }",
        ".gpi-glycan-text { color: #333; font-size: 0.84em; margin-top: 4px; margin-bottom: 3px; font-weight: 700; overflow-wrap: anywhere; }",
        ".gpi-sig-up { color: #d62728; font-weight: bold; }",
        ".gpi-sig-down { color: #1f77b4; font-weight: bold; }",
        ".gpi-glyco-img { max-height: 72px; max-width: 220px; display: block; margin: 0 auto 4px auto; }",
        ".gpi-gray { color: #888; font-size: 0.8em; }",
        "</style>",
        "<table class='gpi-table'>",
        "<thead><tr>",
        "<th>No.</th>",
        "<th>Gene Symbol</th>",
        "<th>Peptide / Glycan</th>",
        "<th>Prosite</th>",
        "<th>log2FC</th>",
        f"<th>{metric_label}</th>",
        "</tr></thead>",
        "<tbody>"
    ]

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        point_num = row.get("Point_ID", i)

        try:
            point_num = int(point_num)
        except Exception:
            point_num = i

        gene_raw = str(row.get(gene_col, row.get("GeneKey", "Unknown")))
        pep_raw = str(row.get(pep_col, row.get("Peptide", "Unknown")))
        gene = html_lib.escape(gene_raw)
        pep = html_lib.escape(pep_raw)

        glycan = row.get(
            "Glycan(H,N,A,F)",
            row.get("Glycan", row.get("Glycan_Composition", ""))
        )
        glycan_raw = "" if pd.isna(glycan) else str(glycan)
        glycan_text = html_lib.escape(glycan_raw)

        site_raw = str(row.get("ProSites", row.get("Glycosite", "")))
        site = html_lib.escape(site_raw)

        lfc = float(row.get(lfc_col, 0.0))
        sig_val = float(row.get(metric_col, 1.0))
        sig_str = f"{sig_val:.2e}" if sig_val < 0.001 else f"{sig_val:.4f}"

        lfc_class = "gpi-sig-up" if lfc > 0 else "gpi-sig-down"
        lfc_str = f'<span class="{lfc_class}">{lfc:+.2f}</span>'

        img_path = find_glycan_image(glycan, img_dir)

        if img_path:
            b64 = image_to_base64(img_path)
            img_tag = f'<img src="{b64}" class="gpi-glyco-img">'
        else:
            img_tag = f'<span class="gpi-gray">[{glycan_text}]</span>' if glycan_text else ""

        pglyco_annotation_raw = str(row.get("P.Glyco.Site", row.get("pGlyco Annotation", row.get("GlycoAnnotation", ""))))
        pglyco_annotation = "" if pglyco_annotation_raw.lower() in ["nan", "none"] else html_lib.escape(pglyco_annotation_raw)

        pglyco_html = f'<div class="gpi-annotation"><b>pGlyco:</b> {pglyco_annotation}</div>' if pglyco_annotation else ""

        # IMPORTANT: keep this as a single non-indented HTML string.
        # Indented multi-line HTML inside a markdown-rendered table can be shown as literal code.
        # Display order: glycan image, glycan composition, pGlyco annotation, peptide sequence.
        pep_html = (
            f'<div class="gpi-glycan-wrap">{img_tag}</div>'
            f'<div class="gpi-glycan-text"><b></b> {glycan_text}</div>'
            f'{pglyco_html}'
            f'<div class="gpi-peptide">{pep}</div>'
        )

        html.append("<tr>")
        html.append(f"<td><span class='gpi-point'>{point_num}</span></td>")
        html.append(f"<td><strong>{gene}</strong></td>")
        html.append(f"<td>{pep_html}</td>")
        html.append(f"<td>{site}</td>")
        html.append(f"<td>{lfc_str}</td>")
        html.append(f"<td>{sig_str}</td>")
        html.append("</tr>")

    html.append("</tbody></table>")

    return "".join(html)

# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def _get_downloads_dir() -> Path:
    """Return a reliable local Downloads folder for macOS/Windows/Linux."""
    downloads = Path.home() / "Downloads"
    return downloads if downloads.exists() else Path.home()


def _safe_download_filename(filename: str) -> str:
    """Remove characters that break filenames on Windows/macOS/Linux."""
    filename = str(filename).strip()
    filename = re.sub(r"[^A-Za-z0-9._-]+", "_", filename)
    filename = filename.strip("_")
    return filename or "downloaded_file"


def _coerce_to_bytes(data):
    """Convert common file-like objects, strings, and paths to bytes."""
    if data is None:
        return None
    if isinstance(data, BytesIO):
        return data.getvalue()
    if isinstance(data, io.BytesIO):
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


def _save_bytes_to_downloads(data, filename: str) -> Path:
    """Save bytes directly to the local Downloads folder; avoids embedded-browser download failures."""
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


def _dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    return buffer.getvalue()


def add_dataframe_download_buttons(df: pd.DataFrame, base_filename: str, key_suffix: str = ""):
    """Desktop-safe CSV/Excel save buttons for dataframes."""
    if df is None or df.empty:
        return
    uid = key_suffix or str(uuid.uuid4())[:8]
    base_filename = _safe_download_filename(base_filename).rsplit(".", 1)[0]
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    excel_bytes = _dataframe_to_excel_bytes(df, sheet_name=base_filename[:31])

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save CSV to Downloads", key=f"save_csv_{base_filename}_{uid}", use_container_width=True):
            try:
                outpath = _save_bytes_to_downloads(csv_bytes, f"{base_filename}.csv")
                st.success(f"Saved CSV to: {outpath}")
            except Exception as e:
                st.error(f"Could not save CSV: {e}")
    with c2:
        if st.button("Save Excel to Downloads", key=f"save_xlsx_{base_filename}_{uid}", use_container_width=True):
            try:
                outpath = _save_bytes_to_downloads(excel_bytes, f"{base_filename}.xlsx")
                st.success(f"Saved Excel to: {outpath}")
            except Exception as e:
                st.error(f"Could not save Excel: {e}")


def add_plot_download_buttons(fig, base_filename: str = "plot"):
    """Desktop-safe PNG/SVG/PDF export buttons for Matplotlib figures."""
    uid = str(uuid.uuid4())[:8]
    base_filename = _safe_download_filename(base_filename).rsplit(".", 1)[0]

    png_buf = io.BytesIO()
    svg_buf = io.BytesIO()
    pdf_buf = io.BytesIO()
    fig.savefig(png_buf, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_buf, format="svg", bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_buf, format="pdf", bbox_inches="tight", facecolor="white")
    png_bytes = png_buf.getvalue()
    svg_bytes = svg_buf.getvalue()
    pdf_bytes = pdf_buf.getvalue()

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Save PNG", key=f"save_png_{base_filename}_{uid}", use_container_width=True):
            try:
                outpath = _save_bytes_to_downloads(png_bytes, f"{base_filename}.png")
                st.success(f"Saved PNG to: {outpath}")
            except Exception as e:
                st.error(f"Could not save PNG: {e}")
    with c2:
        if st.button("Save SVG", key=f"save_svg_{base_filename}_{uid}", use_container_width=True):
            try:
                outpath = _save_bytes_to_downloads(svg_bytes, f"{base_filename}.svg")
                st.success(f"Saved SVG to: {outpath}")
            except Exception as e:
                st.error(f"Could not save SVG: {e}")
    with c3:
        if st.button("Save PDF", key=f"save_pdf_{base_filename}_{uid}", use_container_width=True):
            try:
                outpath = _save_bytes_to_downloads(pdf_bytes, f"{base_filename}.pdf")
                st.success(f"Saved PDF to: {outpath}")
            except Exception as e:
                st.error(f"Could not save PDF: {e}")


def add_plotly_download_buttons(fig, base_filename: str = "interactive_plot"):
    """Save an interactive Plotly figure as self-contained HTML for desktop apps."""
    uid = str(uuid.uuid4())[:8]
    base_filename = _safe_download_filename(base_filename).rsplit(".", 1)[0]
    html_bytes = fig.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8")
    if st.button("Save Interactive HTML", key=f"save_html_{base_filename}_{uid}", use_container_width=True):
        try:
            outpath = _save_bytes_to_downloads(html_bytes, f"{base_filename}.html")
            st.success(f"Saved HTML to: {outpath}")
        except Exception as e:
            st.error(f"Could not save HTML: {e}")


@st.cache_data(show_spinner=False)
def read_table(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        if name.endswith(".tsv") or name.endswith(".txt"):
            return pd.read_csv(uploaded_file, sep="\t")
        return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading '{uploaded_file.name}': {e}")
        return pd.DataFrame()

def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def clean_tmt_sample(col: str) -> str:
    """Normalize common intensity column labels to comparable sample/channel names."""
    if col is None:
        return col
    s = str(col)
    s = re.sub(r"(?i)abundances?\s*\(grouped\):?", "", s)
    s = re.sub(r"(?i)abundances?:?", "", s)
    s = re.sub(r"(?i)reporter\s*", "", s)
    s = re.sub(r"(?i)^lfq[._\- ]?intensity[._\- :]?", "", s)
    s = re.sub(r"(?i)^pg\.quantity[._\- :]?", "", s)
    s = re.sub(r"(?i)^quantity[._\- :]?", "", s)
    s = re.sub(r"(?i)^intensity[._\- :]?", "", s)
    s = s.replace(":", "").strip()
    m = re.search(r"(12[6-9]|13[0-4])[NC]?", s)
    if m:
        return m.group(0)
    return s

def detect_gene_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["GeneID", "Genes", "Gene", "Gene Symbol", "GeneSymbol", "Gene Names"]:
        if c in df.columns:
            return c
    return None

def detect_log2fc_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if re.search(r"(?i)log2\s*fc|log2fc|log\s*2\s*fold", str(c)):
            return c
    return None

def detect_p_col(df: pd.DataFrame) -> Optional[str]:
    # FIX: use re.search instead of re.fullmatch to catch "P.Value", "p_value", etc.
    for c in df.columns:
        if re.search(r"(?i)^p[._\- ]?val", str(c).strip()):
            return c
    return None

def detect_q_col(df: pd.DataFrame) -> Optional[str]:
    # FIX: broadened pattern to catch adj_p_val, q_value, adj.P.Val (limma), etc.
    for c in df.columns:
        if re.search(r"(?i)(q[._\- ]?val|fdr|adj[._]?p)", str(c).strip()):
            return c
    return None

def detect_intensity_cols(df: pd.DataFrame) -> List[str]:
    """Auto-detect TMT, LFQ, Spectronaut, DIA-NN, and generic abundance columns.

    The older version only caught reporter channels and generic abundance columns.
    This broader detector supports common label-free formats while avoiding
    statistical columns such as log2FC, p-values, q-values, scores, and ratios.
    """
    cols = []
    exclude_patterns = [
        r"(?i)log2", r"(?i)fold", r"(?i)^p[._\- ]?val", r"(?i)^q[._\- ]?val",
        r"(?i)adj[._\- ]?p", r"(?i)fdr", r"(?i)ratio", r"(?i)score",
        r"(?i)prob", r"(?i)confid", r"(?i)error", r"(?i)cv$", r"(?i)percent",
    ]
    include_patterns = [
        r"(?i)abundance",
        r"(?i)^lfq[._\- ]?intensity",
        r"(?i)lfq[._\- ]?intensity",
        r"(?i)^intensity[._\- :]",
        r"(?i)^quantity[._\- :]",
        r"(?i)^(pg\.)?quantity",
        r"(?i)^ms[0-9]*[._\- ]?area",
        r"(?i)^reporter",
        r"(?i)^channel",
        r"(?i)^sample[._\- ].*(intensity|quantity|abundance)",
    ]
    for c in df.columns:
        s_col = str(c).strip()
        if any(re.search(pat, s_col) for pat in exclude_patterns):
            continue
        if re.fullmatch(r"(12[6-9]|13[0-4])[NC]?", s_col):
            cols.append(c)
            continue
        if any(re.search(pat, s_col) for pat in include_patterns):
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().sum() > 0:
                cols.append(c)
    return list(dict.fromkeys(cols))

def detect_feature_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def split_gene_tokens(value: object) -> List[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    tokens = re.split(r"[;,|/]", text)
    out = []
    for tok in tokens:
        tok = str(tok).strip().upper()
        if tok and tok != "NAN":
            out.append(tok)
    return list(dict.fromkeys(out))

def choose_primary_gene(value: object) -> Optional[str]:
    toks = split_gene_tokens(value)
    return toks[0] if toks else None

def standardize_gene(df: pd.DataFrame, gene_col: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    if gene_col and gene_col in out.columns:
        out["GeneTokens"] = out[gene_col].apply(split_gene_tokens)
        out["GeneKey"] = out[gene_col].apply(choose_primary_gene)
        out["GeneMultiplicity"] = out["GeneTokens"].apply(len)
        out["HasMultiGeneAssignment"] = out["GeneMultiplicity"] > 1
    else:
        out["GeneTokens"] = [[] for _ in range(len(out))]
        out["GeneKey"] = np.nan
        out["GeneMultiplicity"] = 0
        out["HasMultiGeneAssignment"] = False
    return out


def detect_glycosite_col(df: pd.DataFrame) -> Optional[str]:
    """Find a likely glycosite/localization column for site-level roll-up."""
    candidates = [
        "ProSites", "ProSite", "Glycosite", "GlycoSite", "Glycosylation Site",
        "Site", "Position", "Positions", "P.Glyco.Site", "GlycoSite_Position",
        "Protein Position", "Modified.Site", "Modified Site",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if re.search(r"(?i)(pro.?site|glyco.?site|glycosylation.*site|modified.*site|site.*position)", str(c)):
            return c
    return None


def _clean_site_value(value: object) -> str:
    if pd.isna(value):
        return "UnknownSite"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "na"}:
        return "UnknownSite"
    text = re.sub(r"\s+", "", text)
    return text


def add_glycosite_key(df: pd.DataFrame, site_col: Optional[str], peptide_col: Optional[str] = None) -> pd.DataFrame:
    """Create a biologically safer gene+site grouping key.

    If no site column is present, the function falls back to peptide sequence so
    glycopeptide-level heterogeneity is still not collapsed directly into gene.
    """
    out = df.copy()
    if site_col and site_col in out.columns:
        site_vals = out[site_col].apply(_clean_site_value)
    elif peptide_col and peptide_col in out.columns:
        site_vals = out[peptide_col].astype(str).str.strip().replace({"": "UnknownPeptide", "nan": "UnknownPeptide"})
    else:
        site_vals = pd.Series(["UnknownSite"] * len(out), index=out.index)
    out["Glycosite"] = site_vals
    out["GlycositeKey"] = out["GeneKey"].astype(str).str.upper().str.strip() + "|" + out["Glycosite"].astype(str)
    out.loc[out["GeneKey"].isna(), "GlycositeKey"] = np.nan
    return out


def _is_glycopeptide_row(row: pd.Series) -> bool:
    """Conservative detector to exclude non-glycopeptides from ratio matrices."""
    glyco_cols = [
        "Glycan(H,N,A,F)", "Glycan", "Glycan_Composition", "GlycoAnnotation",
        "pGlyco Annotation", "P.Glyco.Site", "Glycosite", "ProSites"
    ]
    for c in glyco_cols:
        if c in row.index:
            val = row.get(c)
            if pd.notna(val) and str(val).strip().lower() not in {"", "nan", "none", "unglycosylated", "unmodified"}:
                return True
    return False


def filter_glycopeptide_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = df.apply(_is_glycopeptide_row, axis=1)
    # If no rows can be confidently classified, keep the table and let the warning explain the limitation.
    return df.loc[mask].copy() if mask.any() else df.copy()


def median_center_intensity_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Median-center sample/channel intensity columns on log2 scale, then return linear scale."""
    x = df.apply(pd.to_numeric, errors="coerce").astype(float)
    x = x.where(x > 0)
    logx = np.log2(x)
    centered = logx.subtract(logx.median(axis=0, skipna=True), axis=1)
    # Preserve positive scale for downstream ratio computation.
    return np.power(2.0, centered)

def compute_significance_flags(df: pd.DataFrame, lfc_col: str, metric_col: str, alpha: float, lfc_thr: float) -> pd.Series:
    return (safe_numeric(df[lfc_col]).abs() >= lfc_thr) & (safe_numeric(df[metric_col]) <= alpha)

def majority_direction(values: pd.Series) -> str:
    vals = safe_numeric(values).dropna()
    if vals.empty:
        return "No data"
    pos = (vals > 0).sum()
    neg = (vals < 0).sum()
    if pos == neg:
        return "Mixed"
    return "Up" if pos > neg else "Down"

def directional_consistency(values: pd.Series) -> float:
    vals = safe_numeric(values).dropna()
    if vals.empty:
        return np.nan
    pos = (vals > 0).sum()
    neg = (vals < 0).sum()
    if pos + neg == 0:
        return np.nan
    return max(pos, neg) / (pos + neg)

def summarize_site_level_glyco(
    df: pd.DataFrame,
    gene_col: str,
    lfc_col: str,
    metric_col: str,
    peptide_col: Optional[str],
    alpha: float,
    lfc_thr: float,
) -> pd.DataFrame:
    """Summarize glycopeptides at gene+glycosite level before gene roll-up.

    Defensive fix: Streamlit cache / older call paths can sometimes pass a
    standardized glyco table that has GeneKey but not GlycositeKey. Build it
    here as a final safety net so the app never crashes with
    KeyError: 'GlycositeKey'.
    """
    if "GeneKey" not in df.columns:
        df = standardize_gene(df, gene_col)
    if "GlycositeKey" not in df.columns:
        df = add_glycosite_key(df, detect_glycosite_col(df), peptide_col)

    g = df.dropna(subset=["GeneKey", "GlycositeKey"]).copy()
    g = g[g["GeneKey"].astype(str).str.strip() != ""]
    g[lfc_col] = safe_numeric(g[lfc_col])
    g[metric_col] = safe_numeric(g[metric_col])
    g["Sig_this_row"] = compute_significance_flags(g, lfc_col, metric_col, alpha, lfc_thr)
    feat_col = peptide_col if peptide_col else gene_col

    def agg(group: pd.DataFrame) -> pd.Series:
        lfc_vals = safe_numeric(group[lfc_col]).dropna()
        metric_vals = safe_numeric(group[metric_col]).dropna()
        sig_group = group[group["Sig_this_row"]]
        sig_lfc_vals = safe_numeric(sig_group[lfc_col]).dropna()
        primary_vals = sig_lfc_vals if not sig_lfc_vals.empty else lfc_vals
        if not sig_lfc_vals.empty:
            summary_lfc = float(sig_lfc_vals.median())
        elif not lfc_vals.empty:
            summary_lfc = float(lfc_vals.median())
        else:
            summary_lfc = np.nan
        return pd.Series({
            "Gene": group["GeneKey"].iloc[0],
            "GeneName_glyco": group[gene_col].iloc[0],
            "GlycositeKey": str(group.name),
            "Glycosite": group["Glycosite"].iloc[0] if "Glycosite" in group.columns else "UnknownSite",
            "site_log2FC_glyco": summary_lfc,
            "site_metric_glyco_best": float(metric_vals.min()) if not metric_vals.empty else np.nan,
            "site_metric_glyco_median": float(metric_vals.median()) if not metric_vals.empty else np.nan,
            "site_glycopeptide_count": int(group.shape[0]),
            "site_unique_features": int(group[feat_col].astype(str).nunique()),
            "site_significant_peptides": int(sig_group.shape[0]),
            "site_direction": majority_direction(primary_vals),
            "site_consistency": directional_consistency(lfc_vals),
            "site_consistency_sig": directional_consistency(sig_lfc_vals) if not sig_lfc_vals.empty else np.nan,
            "site_multi_gene_rows": int(group["HasMultiGeneAssignment"].sum()),
            "site_has_multigene_assignment": bool(group["HasMultiGeneAssignment"].any()),
        })

    return g.groupby("GlycositeKey", dropna=True).apply(agg, include_groups=False).reset_index(drop=True)


def summarize_gene_level_glyco(
    df: pd.DataFrame,
    gene_col: str,
    lfc_col: str,
    metric_col: str,
    peptide_col: Optional[str],
    alpha: float,
    lfc_thr: float,
) -> pd.DataFrame:
    """Roll site-level summaries up to gene-level without hiding site heterogeneity."""
    site_df = summarize_site_level_glyco(df, gene_col, lfc_col, metric_col, peptide_col, alpha, lfc_thr)
    if site_df.empty:
        return site_df
    site_df["Site_sig"] = compute_significance_flags(site_df, "site_log2FC_glyco", "site_metric_glyco_best", alpha, lfc_thr)

    def agg(group: pd.DataFrame) -> pd.Series:
        lfc_vals = safe_numeric(group["site_log2FC_glyco"]).dropna()
        metric_vals = safe_numeric(group["site_metric_glyco_best"]).dropna()
        sig_group = group[group["Site_sig"]]
        sig_lfc_vals = safe_numeric(sig_group["site_log2FC_glyco"]).dropna()
        primary_vals = sig_lfc_vals if not sig_lfc_vals.empty else lfc_vals
        if not sig_lfc_vals.empty:
            summary_lfc = float(sig_lfc_vals.median())
        elif not lfc_vals.empty:
            summary_lfc = float(lfc_vals.median())
        else:
            summary_lfc = np.nan
        directions = group["site_direction"].dropna().astype(str)
        site_mixed = (directions.isin(["Up"]).any() and directions.isin(["Down"]).any()) or (directions == "Mixed").any()
        return pd.Series({
            "Gene": str(group.name),
            "GeneName_glyco": group["GeneName_glyco"].iloc[0],
            "log2FC_glyco": summary_lfc,
            "log2FC_glyco_mean_all": float(lfc_vals.mean()) if not lfc_vals.empty else np.nan,
            "log2FC_glyco_median_all": float(lfc_vals.median()) if not lfc_vals.empty else np.nan,
            "metric_glyco_best": float(metric_vals.min()) if not metric_vals.empty else np.nan,
            "metric_glyco_median": float(metric_vals.median()) if not metric_vals.empty else np.nan,
            "glyco_site_count": int(group["GlycositeKey"].nunique()),
            "glyco_site_keys": "; ".join(group["GlycositeKey"].astype(str).head(20)),
            "glyco_site_direction_mixed": bool(site_mixed),
            "glyco_peptide_count": int(group["site_glycopeptide_count"].sum()),
            "glyco_unique_features": int(group["site_unique_features"].sum()),
            "glyco_significant_peptides": int(group["site_significant_peptides"].sum()),
            "glyco_significant_sites": int(sig_group["GlycositeKey"].nunique()),
            "glyco_direction": majority_direction(primary_vals),
            "glyco_consistency": directional_consistency(lfc_vals),
            "glyco_consistency_sig": directional_consistency(sig_lfc_vals) if not sig_lfc_vals.empty else np.nan,
            "glyco_multi_gene_rows": int(group["site_multi_gene_rows"].sum()),
            "glyco_has_multigene_assignment": bool(group["site_has_multigene_assignment"].any()),
        })

    return site_df.groupby("Gene", dropna=True).apply(agg, include_groups=False).reset_index(drop=True)

def summarize_gene_level_prot(
    df: pd.DataFrame,
    gene_col: str,
    lfc_col: str,
    metric_col: str,
    protein_col: Optional[str],
    alpha: float,
    lfc_thr: float,
) -> pd.DataFrame:
    """Fast vectorized protein-level gene summary for large TP tables."""
    if "GeneKey" not in df.columns:
        df = standardize_gene(df, gene_col)

    p = df.dropna(subset=["GeneKey"]).copy()
    p = p[p["GeneKey"].astype(str).str.strip() != ""]
    if p.empty:
        return pd.DataFrame()

    p[lfc_col] = safe_numeric(p[lfc_col])
    p[metric_col] = safe_numeric(p[metric_col])
    p["Sig_this_row"] = compute_significance_flags(p, lfc_col, metric_col, alpha, lfc_thr)
    feat_col = protein_col if protein_col and protein_col in p.columns else gene_col

    grouped = p.groupby("GeneKey", dropna=True, sort=False)
    out = grouped.agg(
        GeneName_prot=(gene_col, "first"),
        log2FC_prot_mean_all=(lfc_col, "mean"),
        log2FC_prot_median_all=(lfc_col, "median"),
        metric_prot_best=(metric_col, "min"),
        metric_prot_median=(metric_col, "median"),
        prot_feature_count=(lfc_col, "size"),
        prot_unique_features=(feat_col, pd.Series.nunique),
        prot_significant_features=("Sig_this_row", "sum"),
    ).reset_index().rename(columns={"GeneKey": "Gene"})

    sig_median = p[p["Sig_this_row"]].groupby("GeneKey", sort=False)[lfc_col].median()
    out["log2FC_prot"] = out["Gene"].map(sig_median).combine_first(out["log2FC_prot_median_all"])

    all_pos = grouped[lfc_col].apply(lambda x: (safe_numeric(x) > 0).sum())
    all_neg = grouped[lfc_col].apply(lambda x: (safe_numeric(x) < 0).sum())
    sig_grouped = p[p["Sig_this_row"]].groupby("GeneKey", dropna=True, sort=False)
    sig_pos = sig_grouped[lfc_col].apply(lambda x: (safe_numeric(x) > 0).sum()) if p["Sig_this_row"].any() else pd.Series(dtype=float)
    sig_neg = sig_grouped[lfc_col].apply(lambda x: (safe_numeric(x) < 0).sum()) if p["Sig_this_row"].any() else pd.Series(dtype=float)

    def _direction_for_gene(gene: str) -> str:
        sp = int(sig_pos.get(gene, 0))
        sn = int(sig_neg.get(gene, 0))
        if sp + sn == 0:
            sp = int(all_pos.get(gene, 0))
            sn = int(all_neg.get(gene, 0))
        if sp + sn == 0:
            return "No data"
        if sp == sn:
            return "Mixed"
        return "Up" if sp > sn else "Down"

    def _consistency_for_gene(gene: str) -> float:
        ap = int(all_pos.get(gene, 0))
        an = int(all_neg.get(gene, 0))
        if ap + an == 0:
            return np.nan
        return max(ap, an) / (ap + an)

    out["prot_direction"] = out["Gene"].astype(str).map(_direction_for_gene)
    out["prot_consistency"] = out["Gene"].astype(str).map(_consistency_for_gene)
    return out

def classify_concordance(row: pd.Series) -> str:
    sg = bool(row.get("Sig_glyco", False))
    sp = bool(row.get("Sig_prot", False))
    dg = row.get("glyco_direction", "No data")
    dp = row.get("prot_direction", "No data")

    if sg and sp:
        if dg == "Up" and dp == "Up":
            return "Concordant Up"
        if dg == "Down" and dp == "Down":
            return "Concordant Down"
        if dg == "Up" and dp == "Down":
            return "Discordant: Glyco Up / Protein Down"
        if dg == "Down" and dp == "Up":
            return "Discordant: Glyco Down / Protein Up"
        return "Both Significant (Mixed Direction)"
    if sg:
        return f"Glyco-specific {dg}" if dg in {"Up", "Down"} else "Glyco-specific"
    if sp:
        return f"Protein-specific {dp}" if dp in {"Up", "Down"} else "Protein-specific"
    return "Non-significant"

def color_map_status() -> Dict[str, str]:
    return {
        "Concordant Up": "#1b9e77",
        "Concordant Down": "#66a61e",
        "Discordant: Glyco Up / Protein Down": "#d95f02",
        "Discordant: Glyco Down / Protein Up": "#7570b3",
        "Both Significant (Mixed Direction)": "#e7298a",
        "Glyco-specific Up": "#1f78b4",
        "Glyco-specific Down": "#a6cee3",
        "Protein-specific Up": "#e41a1c",
        "Protein-specific Down": "#fbb4ae",
        "Glyco-specific": "#1f78b4",
        "Protein-specific": "#e41a1c",
        "Non-significant": "#999999",
    }

def build_long_ratio_df(
    glyco_sub: pd.DataFrame,
    prot_sub: pd.DataFrame,
    g_int_cols: List[str],
    p_int_cols: List[str],
    min_intensity: float = 1.0,
    median_center: bool = False,
    glycopeptide_only: bool = True,
) -> pd.DataFrame:
    if glyco_sub.empty or prot_sub.empty or not g_int_cols or not p_int_cols:
        return pd.DataFrame()

    g_use = filter_glycopeptide_rows(glyco_sub) if glycopeptide_only else glyco_sub.copy()
    if g_use.empty:
        return pd.DataFrame()

    gtmp = g_use[g_int_cols].copy()
    ptmp = prot_sub[p_int_cols].copy()
    gtmp.columns = [clean_tmt_sample(c) for c in gtmp.columns]
    ptmp.columns = [clean_tmt_sample(c) for c in ptmp.columns]

    if median_center:
        gtmp = median_center_intensity_frame(gtmp)
        ptmp = median_center_intensity_frame(ptmp)

    mean_g = gtmp.apply(safe_numeric).mean(axis=0)
    mean_p = ptmp.apply(safe_numeric).mean(axis=0)

    shared = [c for c in mean_g.index if c in mean_p.index]
    rows = []
    for ch in shared:
        gv, pv = mean_g[ch], mean_p[ch]
        if pd.isna(gv) or pd.isna(pv) or gv <= 0 or pv <= 0:
            continue
        if gv < min_intensity or pv < min_intensity:
            continue
        rows.append({
            "Sample": ch,
            "GlycoIntensity": float(gv),
            "ProteinIntensity": float(pv),
            "Ratio": float(gv / pv),
            "log2Ratio": float(np.log2(gv / pv)),
        })
    return pd.DataFrame(rows)


def build_global_ratio_matrix(
    merged_gene: pd.DataFrame,
    glyco_df: pd.DataFrame,
    prot_df: pd.DataFrame,
    g_int_cols: List[str],
    p_int_cols: List[str],
    min_intensity: float = 1.0,
    median_center: bool = False,
    glycopeptide_only: bool = True,
) -> pd.DataFrame:
    """Fast gene x sample log2(Glyco/Protein) matrix.

    This avoids the slow previous implementation that scanned the full glyco
    and protein tables once for every gene. It pre-aggregates intensities by
    GeneKey and then computes ratios only for genes present in both omics.
    """
    if merged_gene.empty or glyco_df.empty or prot_df.empty or not g_int_cols or not p_int_cols:
        return pd.DataFrame()

    g_use = filter_glycopeptide_rows(glyco_df) if glycopeptide_only else glyco_df.copy()
    p_use = prot_df.copy()
    if g_use.empty or p_use.empty or "GeneKey" not in g_use.columns or "GeneKey" not in p_use.columns:
        return pd.DataFrame()

    def _gene_mean_matrix(df: pd.DataFrame, int_cols: List[str]) -> pd.DataFrame:
        x = df[int_cols].apply(pd.to_numeric, errors="coerce")
        x.columns = [clean_tmt_sample(c) for c in x.columns]
        # Collapse duplicate cleaned channels such as Abundance: 126 and 126.
        x = x.T.groupby(level=0).mean().T
        if median_center:
            x = median_center_intensity_frame(x)
        x["GeneKey"] = df["GeneKey"].astype(str).values
        return x.groupby("GeneKey", dropna=True, sort=False).mean(numeric_only=True)

    g_mean = _gene_mean_matrix(g_use, g_int_cols)
    p_mean = _gene_mean_matrix(p_use, p_int_cols)

    candidate_genes = set(merged_gene["Gene"].dropna().astype(str))
    shared_genes = [g for g in g_mean.index.astype(str) if g in p_mean.index.astype(str) and g in candidate_genes]
    shared_cols = [c for c in g_mean.columns if c in p_mean.columns]
    if not shared_genes or not shared_cols:
        return pd.DataFrame()

    g_sub = g_mean.loc[shared_genes, shared_cols].astype(float)
    p_sub = p_mean.loc[shared_genes, shared_cols].astype(float)
    valid = (g_sub > 0) & (p_sub > 0) & (g_sub >= min_intensity) & (p_sub >= min_intensity)
    ratio = np.log2(g_sub / p_sub).where(valid)
    ratio = ratio.dropna(axis=0, how="all")
    ratio.index.name = "Gene"
    return ratio

def build_ratio_consistency_table(ratio_mat: pd.DataFrame) -> pd.DataFrame:
    if ratio_mat is None or ratio_mat.empty:
        return pd.DataFrame()
    out = pd.DataFrame(index=ratio_mat.index)
    out["mean_log2Ratio"] = ratio_mat.mean(axis=1, skipna=True)
    out["median_log2Ratio"] = ratio_mat.median(axis=1, skipna=True)
    out["sd_log2Ratio"] = ratio_mat.std(axis=1, skipna=True)
    out["cv_abs_log2Ratio"] = out["sd_log2Ratio"] / out["mean_log2Ratio"].abs().replace(0, np.nan)
    out["n_samples"] = ratio_mat.notna().sum(axis=1)
    out["n_up_samples"] = (ratio_mat > 0).sum(axis=1)
    out["n_down_samples"] = (ratio_mat < 0).sum(axis=1)
    denom = (out["n_up_samples"] + out["n_down_samples"]).replace(0, np.nan)
    out["sample_direction_consistency"] = out[["n_up_samples", "n_down_samples"]].max(axis=1) / denom
    out["sample_direction"] = np.where(out["n_up_samples"] > out["n_down_samples"], "Up", np.where(out["n_down_samples"] > out["n_up_samples"], "Down", "Mixed"))
    return out.reset_index().rename(columns={"index": "Gene"})


def parse_gene_set_file(uploaded_file) -> Dict[str, set]:
    """Parse GMT or two-column CSV/TSV gene set files."""
    if uploaded_file is None:
        return {}
    raw = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    gene_sets: Dict[str, set] = {}
    name = uploaded_file.name.lower()
    if name.endswith(".gmt"):
        for line in raw.splitlines():
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                gene_sets[parts[0]] = {g.strip().upper() for g in parts[2:] if g.strip()}
    else:
        sep = "\t" if name.endswith((".tsv", ".txt")) else ","
        df = pd.read_csv(io.StringIO(raw), sep=sep)
        if df.shape[1] >= 2:
            set_col, gene_col = df.columns[:2]
            for set_name, sub in df.groupby(set_col):
                gene_sets[str(set_name)] = {str(g).strip().upper() for g in sub[gene_col].dropna() if str(g).strip()}
    return {k: v for k, v in gene_sets.items() if v}


def run_simple_fisher_enrichment(selected_genes: set, background_genes: set, gene_sets: Dict[str, set]) -> pd.DataFrame:
    selected_genes = {g.upper() for g in selected_genes if g}
    background_genes = {g.upper() for g in background_genes if g}
    if not selected_genes or not background_genes or not gene_sets:
        return pd.DataFrame()
    rows = []
    for name, genes in gene_sets.items():
        gs = {g.upper() for g in genes} & background_genes
        if not gs:
            continue
        a = len(selected_genes & gs)
        b = len(selected_genes - gs)
        c = len(gs - selected_genes)
        d = max(len(background_genes - selected_genes - gs), 0)
        if a == 0:
            continue
        odds, pval = fisher_exact([[a, b], [c, d]], alternative="greater")
        rows.append({"GeneSet": name, "Overlap": a, "SetSizeInBackground": len(gs), "SelectedGenes": len(selected_genes), "OddsRatio": odds, "p_value": pval, "Genes": ";".join(sorted(selected_genes & gs))})
    out = pd.DataFrame(rows).sort_values("p_value") if rows else pd.DataFrame()
    if not out.empty:
        out["FDR_BH"] = (out["p_value"] * len(out) / (np.arange(len(out)) + 1)).clip(upper=1.0)
    return out

def create_side_by_side_waterfall_html_table_export(
    wfl,
    disp_df,
    html_table,
    g_lfc_col,
    metric_col,
    sig_alpha,
    log2fc_thr,
    img_dir=None,
    gene_col=None,
    pep_col=None
):
    from io import BytesIO
    import base64
    import textwrap
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw, ImageFont
    import pandas as pd
    import numpy as np

    # -----------------------------
    # 1. Create waterfall plot PNG
    # -----------------------------
    fig, ax = plt.subplots(figsize=(8, 7))

    color_map = {
        "Up regulated": "#d62728",
        "Down regulated": "#1f77b4",
        "Non Significant": "#cccccc",
    }

    for status, sub in wfl.groupby("Status"):
        ax.scatter(
            sub["Rank"],
            sub[g_lfc_col],
            s=22,
            alpha=0.85,
            color=color_map.get(status, "#999999"),
            label=status
        )

    label_df = wfl.dropna(subset=["Point_ID"]).copy()

    if not label_df.empty:
        # Move numeric labels to the side of each point so they do not cover the dot.
        x_span = max(float(wfl["Rank"].max() - wfl["Rank"].min()), 1.0)
        x_offset = x_span * 0.05

        for _, row in label_df.iterrows():
            x0 = row["Rank"]
            y0 = row[g_lfc_col]
            x1 = x0 + x_offset

            ax.plot([x0, x1], [y0, y0], color="gray", linewidth=0.6, alpha=0.8)
            ax.text(
                x1,
                y0,
                str(int(row["Point_ID"])),
                fontsize=8,
                ha="left",
                va="center",
                fontweight="bold",
                color="black",
            )

    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(log2fc_thr, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(-log2fc_thr, color="gray", linestyle="--", linewidth=0.8)

    ax.set_title(
        f"Glycopeptide ({metric_col} < {sig_alpha}, |log2FC| > {log2fc_thr})"
    )
    ax.set_xlabel("Glycopeptide Rank")
    ax.set_ylabel("log2 Fold-Change")
    ax.grid(False)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    plot_png = BytesIO()
    fig.savefig(plot_png, format="png", dpi=300, bbox_inches="tight")
    plot_png.seek(0)
    plt.close(fig)

    plot_img = Image.open(plot_png).convert("RGB")

    # -----------------------------
    # 2. Build table as image
    # -----------------------------
    table_df = disp_df.copy()

    W = 1500
    row_h = 205
    header_h = 60
    pad = 18

    table_h = header_h + max(1, len(table_df)) * row_h + 40
    table_img = Image.new("RGB", (W, table_h), "white")
    draw = ImageDraw.Draw(table_img)

    try:
        font = ImageFont.truetype("Arial.ttf", 24)
        font_bold = ImageFont.truetype("Arial Bold.ttf", 26)
        font_small = ImageFont.truetype("Arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
        font_bold = ImageFont.load_default()
        font_small = ImageFont.load_default()

    cols = {
        "No.": 70,
        "Gene": 170,
        "Peptide / Glycan": 650,
        "Prosite": 180,
        "log2FC": 130,
        metric_col: 190,
    }

    x_positions = {}
    x = pad
    for c, width in cols.items():
        x_positions[c] = x
        x += width

    # Header
    draw.rectangle([0, 0, W, header_h], fill="#f8f9fa")

    for c, xpos in x_positions.items():
        draw.text((xpos, 18), c, fill="#495057", font=font_bold)

    y = header_h

    for _, row in table_df.iterrows():
        draw.line((0, y, W, y), fill="#e9ecef", width=2)

        point_num = int(row.get("Point_ID", 0)) if pd.notna(row.get("Point_ID", np.nan)) else ""
        gene_raw = str(row.get(gene_col, row.get("GeneKey", "Unknown")))
        pep_raw = str(row.get(pep_col, row.get("Peptide", "Unknown")))
        gene = html_lib.escape(gene_raw)
        pep = html_lib.escape(pep_raw)
        site_raw = str(row.get("ProSites", row.get("Glycosite", "")))
        site = html_lib.escape(site_raw)

        lfc = float(row.get(g_lfc_col, 0.0))
        sig_val = float(row.get(metric_col, 1.0))

        glycan = row.get(
            "Glycan(H,N,A,F)",
            row.get("Glycan", row.get("Glycan_Composition", ""))
        )
        glycan_text = "" if pd.isna(glycan) else str(glycan)

        lfc_color = "#d62728" if lfc > 0 else "#1f77b4"

        draw.text((x_positions["No."], y + 45), str(point_num), fill="black", font=font_bold)
        draw.text((x_positions["Gene"], y + 45), gene, fill="black", font=font_bold)

        pglyco_annotation = str(row.get("P.Glyco.Site", row.get("pGlyco Annotation", row.get("GlycoAnnotation", ""))))
        if pglyco_annotation.lower() in ["nan", "none"]:
            pglyco_annotation = ""

        # Glycan image first
        glycan_x = x_positions["Peptide / Glycan"]
        if img_dir is not None:
            img_path = find_glycan_image(glycan, img_dir)
            if img_path:
                try:
                    gly_img = Image.open(img_path).convert("RGBA")
                    gly_img.thumbnail((210, 58))
                    table_img.paste(gly_img, (glycan_x, y + 8), gly_img)
                except Exception:
                    pass

        # Glycan composition second, from glyco input table
        glycan_lines = textwrap.wrap(f"Glycan: {glycan_text}", width=54)
        for j, line in enumerate(glycan_lines[:2]):
            draw.text(
                (glycan_x, y + 70 + j * 22),
                line,
                fill="#333333",
                font=font_small
            )

        # pGlyco annotation third
        pglyco_y = y + 118
        if pglyco_annotation:
            pglyco_lines = textwrap.wrap(f"pGlyco: {pglyco_annotation}", width=54)
            for j, line in enumerate(pglyco_lines[:1]):
                draw.text(
                    (glycan_x, pglyco_y + j * 22),
                    line,
                    fill="#666666",
                    font=font_small
                )

        # Peptide name fourth
        pep_lines = textwrap.wrap(pep, width=54)
        for j, line in enumerate(pep_lines[:2]):
            draw.text(
                (glycan_x, y + 145 + j * 24),
                line,
                fill="black",
                font=font_small
            )

        draw.text((x_positions["Prosite"], y + 85), site, fill="black", font=font)
        draw.text((x_positions["log2FC"], y + 85), f"{lfc:+.2f}", fill=lfc_color, font=font_bold)
        draw.text((x_positions[metric_col], y + 85), f"{sig_val:.2e}", fill="black", font=font)

        y += row_h

    draw.line((0, y, W, y), fill="#e9ecef", width=2)

    # -----------------------------
    # 3. Combine side-by-side
    # -----------------------------
    target_h = max(plot_img.height, table_img.height)

    plot_ratio = target_h / plot_img.height
    plot_resized = plot_img.resize(
        (int(plot_img.width * plot_ratio), target_h)
    )

    table_ratio = target_h / table_img.height
    table_resized = table_img.resize(
        (int(table_img.width * table_ratio), target_h)
    )

    gap = 40
    combined_w = plot_resized.width + table_resized.width + gap
    combined_h = target_h

    combined = Image.new("RGB", (combined_w, combined_h), "white")
    combined.paste(plot_resized, (0, 0))
    combined.paste(table_resized, (plot_resized.width + gap, 0))

    png_out = BytesIO()
    combined.save(png_out, format="PNG", dpi=(300, 300))
    png_out.seek(0)

    # -----------------------------
    # 4. SVG fallback as embedded PNG
    # -----------------------------
    png_b64 = base64.b64encode(png_out.getvalue()).decode("utf-8")

    svg_text = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="{combined_w}" height="{combined_h}">
        <image href="data:image/png;base64,{png_b64}" width="{combined_w}" height="{combined_h}"/>
    </svg>
    """

    return png_out.getvalue(), svg_text.encode("utf-8")

@st.cache_data(show_spinner=False)
def build_summary_tables(
    glyco_df: pd.DataFrame,
    prot_df: pd.DataFrame,
    g_gene_col: str,
    g_lfc_col: str,
    g_metric_col: str,
    p_gene_col: str,
    p_lfc_col: str,
    p_metric_col: str,
    glyco_alpha: float,
    prot_alpha: float,
    glyco_lfc_thr: float,
    prot_lfc_thr: float,
    min_sig_peptides: int,
    min_glyco_consistency: float,
    peptide_col: Optional[str],
    protein_col: Optional[str],
    glycosite_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    glyco_df = standardize_gene(glyco_df, g_gene_col)
    glyco_df = add_glycosite_key(glyco_df, glycosite_col, peptide_col)
    prot_df = standardize_gene(prot_df, p_gene_col)

    glyco_gene = summarize_gene_level_glyco(
        glyco_df, g_gene_col, g_lfc_col, g_metric_col, peptide_col,
        glyco_alpha, glyco_lfc_thr
    )
    prot_gene = summarize_gene_level_prot(
        prot_df, p_gene_col, p_lfc_col, p_metric_col, protein_col,
        prot_alpha, prot_lfc_thr
    )

    merged_gene = pd.merge(glyco_gene, prot_gene, on="Gene", how="outer")

    merged_gene["Sig_glyco"] = (
        (safe_numeric(merged_gene["log2FC_glyco"]).abs() >= glyco_lfc_thr) &
        (safe_numeric(merged_gene["metric_glyco_best"]) <= glyco_alpha) &
        (safe_numeric(merged_gene["glyco_significant_peptides"]).fillna(0) >= min_sig_peptides) &
        (safe_numeric(merged_gene["glyco_consistency"]).fillna(0) >= min_glyco_consistency)
    )
    merged_gene["Sig_prot"] = (
        (safe_numeric(merged_gene["log2FC_prot"]).abs() >= prot_lfc_thr) &
        (safe_numeric(merged_gene["metric_prot_best"]) <= prot_alpha)
    )
    merged_gene["Status"] = merged_gene.apply(classify_concordance, axis=1)
    merged_gene["DisplayGeneName"] = merged_gene["GeneName_glyco"].fillna(merged_gene["GeneName_prot"]).fillna(merged_gene["Gene"])
    merged_gene["Delta_log2FC"] = safe_numeric(merged_gene["log2FC_glyco"]) - safe_numeric(merged_gene["log2FC_prot"])
    merged_gene["Abs_Delta_log2FC"] = merged_gene["Delta_log2FC"].abs()

    return glyco_df, prot_df, merged_gene

@dataclass
class FigurePack:
    figures: Dict[str, plt.Figure]
    captions: Dict[str, str]

def generate_standard_figures(
    merged_gene: pd.DataFrame,
    glyco_df: pd.DataFrame,
    prot_df: pd.DataFrame,
    g_int_cols: List[str],
    p_int_cols: List[str],
    glyco_lfc_thr: float,
    prot_lfc_thr: float,
    top_n_heatmap: int = 40,
    top_n_ratio: int = 25,
    min_intensity_ratio: float = 1.0,
    median_center_ratio: bool = False,
) -> FigurePack:
    figs: Dict[str, plt.Figure] = {}
    caps: Dict[str, str] = {}
    cmap = color_map_status()

    # 1. Status bar plot
    order = [
        "Concordant Up", "Concordant Down",
        "Discordant: Glyco Up / Protein Down", "Discordant: Glyco Down / Protein Up",
        "Both Significant (Mixed Direction)",
        "Glyco-specific Up", "Glyco-specific Down",
        "Protein-specific Up", "Protein-specific Down",
        "Non-significant",
    ]
    vc = merged_gene["Status"].fillna("Non-significant").value_counts()
    counts = pd.DataFrame({
        "Status": [status for status in order if int(vc.get(status, 0)) > 0],
        "Count": [int(vc.get(status, 0)) for status in order if int(vc.get(status, 0)) > 0],
    })
    fig1, ax1 = plt.subplots(figsize=(8, 4.8))
    bar_colors = [cmap.get(str(x), "#999999") or "#999999" for x in counts["Status"]]
    ax1.bar(counts["Status"].astype(str), counts["Count"].astype(int), color=bar_colors)
    ax1.set_title("Concordance Class Distribution", pad=12)
    ax1.set_xlabel("")
    ax1.set_ylabel("Genes")
    plt.setp(ax1.get_xticklabels(), rotation=35, ha="right")
    plt.tight_layout()
    figs["01_concordance_distribution"] = fig1
    caps["01_concordance_distribution"] = "Gene counts split into concordant, discordant, glyco-specific, and protein-specific classes."

    # 2. Global scatter
    sc = merged_gene.dropna(subset=["log2FC_glyco", "log2FC_prot"]).copy()
    if not sc.empty:
        fig2, ax2 = plt.subplots(figsize=(7.2, 6.2))
        sizes = np.clip(sc["glyco_peptide_count"].fillna(1).astype(float) * 18, 30, 220)
        for status, sub in sc.groupby("Status", dropna=False):
            idx = sub.index
            ax2.scatter(
                sub["log2FC_prot"],
                sub["log2FC_glyco"],
                s=sizes.loc[idx] if hasattr(sizes, "loc") else np.asarray(sizes),
                alpha=0.82,
                color=cmap.get(str(status), "#999999") or "#999999",
                label=str(status),
                edgecolors="none",
            )
        ax2.axvline(0, color="gray", ls="--", lw=1)
        ax2.axhline(0, color="gray", ls="--", lw=1)
        ax2.axvline(prot_lfc_thr, color="#444", ls=":", lw=1)
        ax2.axvline(-prot_lfc_thr, color="#444", ls=":", lw=1)
        ax2.axhline(glyco_lfc_thr, color="#444", ls=":", lw=1)
        ax2.axhline(-glyco_lfc_thr, color="#444", ls=":", lw=1)
        ax2.set_xlabel("Protein log2FC")
        ax2.set_ylabel("Glyco log2FC")
        ax2.set_title("Global Glyco vs Protein Fold-Change Concordance", pad=12)
        ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        figs["02_global_concordance_scatter"] = fig2
        caps["02_global_concordance_scatter"] = "Bubble size reflects glycopeptide support; dotted lines mark log2FC thresholds."

    # 3. Top discordant genes
    disc = merged_gene[merged_gene["Status"].str.contains("Discordant", na=False)].copy()
    if not disc.empty:
        disc = disc.sort_values("Abs_Delta_log2FC", ascending=False).head(20)
        fig3, ax3 = plt.subplots(figsize=(8, max(4.5, 0.32 * len(disc))))
        y = np.arange(len(disc))
        ax3.scatter(disc["log2FC_prot"], y, label="Protein", s=60)
        ax3.scatter(disc["log2FC_glyco"], y, label="Glyco", s=60)
        for i, (_, row) in enumerate(disc.iterrows()):
            ax3.plot([row["log2FC_prot"], row["log2FC_glyco"]], [i, i], lw=1.8, color="#888")
        ax3.axvline(0, color="black", ls="--", lw=1)
        ax3.set_yticks(y)
        ax3.set_yticklabels(disc["Gene"])
        ax3.set_xlabel("log2FC")
        ax3.set_title("Top Discordant Genes", pad=12)
        ax3.legend()
        plt.tight_layout()
        figs["03_top_discordant_genes"] = fig3
        caps["03_top_discordant_genes"] = "Genes with the largest glyco-versus-protein fold-change disagreement."

    # 4. Real clustermap
    hm = merged_gene[merged_gene["Status"] != "Non-significant"].dropna(subset=["log2FC_glyco", "log2FC_prot"]).copy()
    if not hm.empty:
        hm = hm.sort_values("Abs_Delta_log2FC", ascending=False).head(top_n_heatmap).set_index("Gene")[["log2FC_glyco", "log2FC_prot"]]
        hm.columns = ["Glyco log2FC", "Protein log2FC"]
        cg = sns.clustermap(hm, cmap="coolwarm", center=0, linewidths=0.2, figsize=(6.4, max(5, 0.22 * len(hm))), cbar_kws={"label": "log2FC"})
        cg.fig.suptitle("Hierarchical Clustering of Differential Signal", y=1.02)
        figs["04_hierarchical_clustermap"] = cg.fig
        caps["04_hierarchical_clustermap"] = "True hierarchical clustering across significant genes using glyco and protein log2FC values."

    # 5. Ratio heatmap
    ratio_mat = build_global_ratio_matrix(
        merged_gene, glyco_df, prot_df, g_int_cols, p_int_cols,
        min_intensity=min_intensity_ratio, median_center=median_center_ratio, glycopeptide_only=True
    )
    if not ratio_mat.empty:
        ranking = ratio_mat.mean(axis=1).sort_values(ascending=False).head(top_n_ratio).index
        ratio_top = ratio_mat.loc[ranking]
        fig5, ax5 = plt.subplots(figsize=(8.2, max(4.5, 0.28 * len(ratio_top))))
        sns.heatmap(ratio_top.fillna(0), cmap="vlag", center=0, ax=ax5, cbar_kws={"label": "log2(Glyco / Protein)", "shrink": 0.65})
        ax5.set_title("Relative Glyco-to-Protein Ratio Map", pad=12)
        ax5.set_ylabel("")
        plt.tight_layout()
        figs["05_log2_ratio_heatmap"] = fig5
        caps["05_log2_ratio_heatmap"] = "Relative glyco-to-protein ratio shown as log2(Glyco/Protein), not absolute occupancy."

    return FigurePack(figures=figs, captions=caps)

def generate_report_bundle(
    merged_df: pd.DataFrame,
    glyco_df: pd.DataFrame,
    prot_df: pd.DataFrame,
    figure_pack: FigurePack,
) -> Tuple[bytes, bytes]:
    zip_buf = io.BytesIO()
    pdf_buf = io.BytesIO()

    with PdfPages(pdf_buf) as pdf:
        title_page = plt.figure(figsize=(8.3, 6.2))
        title_page.clf()
        title_page.text(0.5, 0.72, "Glyco-Proteome Integrator (GPI)", ha="center", va="center", size=21, weight="bold", color="#005a9c")
        title_page.text(0.5, 0.60, "Evidence-aware multi-omic integration report", ha="center", va="center", size=14)
        title_page.text(0.5, 0.50, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ha="center", va="center", size=11, color="gray")
        title_page.text(0.5, 0.34, "Gene-level glyco calls are filtered using peptide support and directional consistency.", ha="center", va="center", size=11)
        pdf.savefig(title_page, bbox_inches="tight")
        plt.close(title_page)

        summary_page = plt.figure(figsize=(8.3, 6.2))
        summary_page.clf()
        ax = summary_page.add_axes([0.08, 0.08, 0.84, 0.84])
        ax.axis("off")
        lines = [
            f"Total glyco rows: {glyco_df.shape[0]:,}",
            f"Total proteome rows: {prot_df.shape[0]:,}",
            f"Integrated genes: {merged_df.shape[0]:,}",
            f"Concordant Up: {(merged_df['Status'] == 'Concordant Up').sum():,}",
            f"Concordant Down: {(merged_df['Status'] == 'Concordant Down').sum():,}",
            f"Discordant genes: {merged_df['Status'].str.contains('Discordant', na=False).sum():,}",
            f"Glyco-specific genes: {merged_df['Status'].str.contains('Glyco-specific', na=False).sum():,}",
            f"Protein-specific genes: {merged_df['Status'].str.contains('Protein-specific', na=False).sum():,}",
        ]
        ax.text(0.0, 1.0, "Analysis Summary", fontsize=18, fontweight="bold", va="top", color="#005a9c")
        for i, line in enumerate(lines):
            ax.text(0.02, 0.90 - i * 0.08, line, fontsize=12, va="top")
        pdf.savefig(summary_page, bbox_inches="tight")
        plt.close(summary_page)

        for name, fig in figure_pack.figures.items():
            pdf.savefig(fig, bbox_inches="tight")

    # FIX: pdf_bytes must be read AFTER the `with PdfPages` block closes (flushes).
    # Previously pdf_bytes was assigned inside the block, producing a truncated/corrupt PDF.
    pdf_bytes = pdf_buf.getvalue()

    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("GPI_Analysis_Report.pdf", pdf_bytes)
        zf.writestr("tables/gpi_integrated_results.csv", merged_df.to_csv(index=False))
        zf.writestr("tables/gpi_glyco_data.csv", glyco_df.to_csv(index=False))
        zf.writestr("tables/gpi_proteome_data.csv", prot_df.to_csv(index=False))
        for name, fig in figure_pack.figures.items():
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format="png", dpi=300, bbox_inches="tight")
            zf.writestr(f"plots/{name}.png", img_buf.getvalue())
        manifest = [
            "GPI Export Bundle",
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "This bundle regenerates standard figures from the current thresholds.",
            "The ratio heatmap represents relative log2(Glyco/Protein), not absolute glycan occupancy.",
            "Ratio plots filter likely non-glycopeptide rows when glycan/site annotations exist and can optionally median-center sample intensities before ratio calculation.",
        ]
        zf.writestr("README.txt", "\n".join(manifest))
    return pdf_bytes, zip_buf.getvalue()






def render_glycopeptide_volcano(
    glyco_df: pd.DataFrame,
    g_gene_col: str,
    g_peptide_col: Optional[str],
    g_lfc_col: str,
    default_metric_col: str,
    glyco_alpha: float,
    glyco_lfc_thr: float,
):
    st.markdown("#### Glycopeptide volcano plot")
    metric_cols = [c for c in glyco_df.columns if re.search(r"(?i)(^p[._\- ]?val|^q[._\- ]?val|adj[._]?p|fdr)", str(c))]
    if default_metric_col not in metric_cols and default_metric_col in glyco_df.columns:
        metric_cols.insert(0, default_metric_col)
    if not metric_cols:
        st.info("No p-value or q-value column was detected for volcano plotting.")
        return
    c1, c2, c3 = st.columns([0.35, 0.25, 0.25])
    with c1:
        metric_col = st.selectbox("Volcano significance metric", metric_cols, index=metric_cols.index(default_metric_col) if default_metric_col in metric_cols else 0, key="volcano_metric_col")
    with c2:
        alpha = st.number_input("Volcano p/q cutoff", min_value=1e-12, max_value=1.0, value=float(glyco_alpha), step=0.01, format="%.4f", key="volcano_alpha")
    with c3:
        lfc_thr = st.number_input("Volcano |log2FC| cutoff", min_value=0.0, value=float(glyco_lfc_thr), step=0.1, format="%.2f", key="volcano_lfc")

    vdf = glyco_df.copy()
    vdf[g_lfc_col] = safe_numeric(vdf[g_lfc_col])
    vdf[metric_col] = safe_numeric(vdf[metric_col])
    vdf = vdf.dropna(subset=[g_lfc_col, metric_col]).copy()
    vdf = vdf[vdf[metric_col] > 0].copy()
    if vdf.empty:
        st.info("No valid glycopeptide rows are available after filtering missing log2FC and metric values.")
        return
    vdf["neg_log10_metric"] = -np.log10(vdf[metric_col].clip(lower=1e-300))
    vdf["VolcanoStatus"] = "Not significant"
    vdf.loc[(vdf[g_lfc_col] >= lfc_thr) & (vdf[metric_col] <= alpha), "VolcanoStatus"] = "Up regulated"
    vdf.loc[(vdf[g_lfc_col] <= -lfc_thr) & (vdf[metric_col] <= alpha), "VolcanoStatus"] = "Down regulated"
    hover_cols = [c for c in [g_gene_col, g_peptide_col, "Glycosite", "Glycan(H,N,A,F)", "Glycan", metric_col] if c and c in vdf.columns]
    fig_vol = px.scatter(
        vdf,
        x=g_lfc_col,
        y="neg_log10_metric",
        color="VolcanoStatus",
        color_discrete_map={"Up regulated": "#d62728", "Down regulated": "#1f77b4", "Not significant": "#cccccc"},
        hover_data=hover_cols,
        labels={g_lfc_col: "log2 Fold-Change", "neg_log10_metric": f"-log10({metric_col})"},
        height=620,
    )
    fig_vol.add_vline(x=0, line_dash="solid", line_color="black")
    fig_vol.add_vline(x=lfc_thr, line_dash="dash", line_color="gray")
    fig_vol.add_vline(x=-lfc_thr, line_dash="dash", line_color="gray")
    fig_vol.add_hline(y=-np.log10(alpha), line_dash="dash", line_color="gray")
    fig_vol.update_layout(plot_bgcolor="white", paper_bgcolor="white", margin=dict(t=35, l=45, r=40, b=55))
    st.plotly_chart(fig_vol, use_container_width=True, config={"displaylogo": False})
    add_plotly_download_buttons(fig_vol, "glycopeptide_volcano_interactive")
    show_cols = [c for c in [g_gene_col, g_peptide_col, "Glycosite", g_lfc_col, metric_col, "VolcanoStatus"] if c and c in vdf.columns]
    st.dataframe(vdf.sort_values(metric_col).head(100)[show_cols], use_container_width=True, hide_index=True)
    add_dataframe_download_buttons(vdf[show_cols], "glycopeptide_volcano_table", key_suffix="glycopeptide_volcano")

def render_professional_gpi_start_panel(upload_disabled: bool = True):
    """Professional first-run landing panel for the Glyco-Proteome Integrator."""
    button_text = "Upload data to enable integration" if upload_disabled else "Run Integration"

    st.markdown(
        """
        <style>
        .gpi-start-wrap {
            margin-top: 1.4rem;
            margin-bottom: 1.5rem;
        }
        .gpi-start-title {
            font-size: 1.85rem;
            line-height: 1.15;
            font-weight: 700;
            color: #005a9c;
            margin-bottom: 0.35rem;
            letter-spacing: -0.01em;
        }
        .gpi-start-subtitle {
            font-size: 1.02rem;
            color: #2f3a45;
            margin-bottom: 1.6rem;
            max-width: 980px;
            line-height: 1.55;
        }
        .gpi-step-card {
            border: 1px solid #d9e2ef;
            border-radius: 10px;
            padding: 22px 24px;
            background: #f8fbff;
            min-height: 138px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }
        .gpi-step-number {
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
        .gpi-step-heading {
            font-size: 1.15rem;
            font-weight: 700;
            color: #005a9c;
            vertical-align: middle;
        }
        .gpi-step-body {
            color: #2f3a45;
            margin-left: 58px;
            margin-top: 8px;
            font-size: 0.96rem;
            line-height: 1.5;
        }
        .gpi-step-arrow {
            text-align: center;
            font-size: 2.2rem;
            color: #8a94a6;
            padding-top: 45px;
            font-weight: 300;
        }
        .gpi-info-box {
            background: #eef6ff;
            border-radius: 8px;
            padding: 16px 20px;
            margin-top: 24px;
            color: #004b8d;
            font-weight: 700;
            border: 1px solid #d8eaff;
        }
        .gpi-info-box-sub {
            font-weight: 400;
            margin-top: 6px;
            color: #263746;
        }
        .gpi-warning-box {
            background: #fff8e5;
            border-radius: 8px;
            padding: 16px 20px;
            margin-top: 14px;
            color: #8a5a00;
            font-weight: 700;
            border: 1px solid #f5e4b6;
        }
        .gpi-warning-box-sub {
            font-weight: 400;
            margin-top: 6px;
            color: #5f4300;
        }
        .gpi-muted-button {
            border: 1px solid #d9dfe8;
            border-radius: 7px;
            padding: 10px 18px;
            text-align: center;
            color: #8a94a6;
            font-weight: 600;
            margin-top: 28px;
            background: #ffffff;
        }
        .gpi-active-button-note {
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
        <div class="gpi-start-wrap">
            <div class="gpi-start-title">Start Here</div>
            <div class="gpi-start-subtitle">
                Begin your glyco-proteome integration by uploading differential result tables for both glycoproteomics and total proteomics.
                Once both files are available, the app will summarize matched genes, compare fold-change directions, and highlight concordant or discordant biology.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, arrow_col, col2 = st.columns([0.46, 0.08, 0.46])

    with col1:
        st.markdown(
            """
            <div class="gpi-step-card">
                <div>
                    <span class="gpi-step-number">1</span>
                    <span class="gpi-step-heading">Upload Results</span>
                </div>
                <div class="gpi-step-body">
                    Upload one glycoproteome differential table and one total proteome differential table using the left sidebar.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with arrow_col:
        st.markdown("<div class='gpi-step-arrow'>›</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            """
            <div class="gpi-step-card">
                <div>
                    <span class="gpi-step-number">2</span>
                    <span class="gpi-step-heading">Review Integration</span>
                </div>
                <div class="gpi-step-body">
                    Confirm detected columns, review concordance classes, inspect targeted gene evidence, and export reports.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="gpi-info-box">
            Upload data using the left sidebar
            <div class="gpi-info-box-sub">
                Required inputs: glycoproteome peptide-level differential results and total proteome protein-level differential results.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if upload_disabled:
        st.markdown(
            """
            <div class="gpi-warning-box">
                Waiting for input files
                <div class="gpi-warning-box-sub">
                    Please upload both result tables to enable the integration workflow.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    btn_class = "gpi-muted-button" if upload_disabled else "gpi-active-button-note"
    st.markdown(f"<div class='{btn_class}'>{button_text}</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------
def main():
    with st.sidebar:
        st.markdown("### GPI Workflow")
        st.caption("v1.0 | Bioinformatics Workflow")
        st.divider()

        st.markdown("**Upload differential result tables**")
        glyco_file = st.file_uploader("Glycoproteome results (peptide-level)", type=["csv", "tsv", "txt", "xlsx"], key="glyco_up")
        prot_file = st.file_uploader("Total proteome results (protein-level)", type=["csv", "tsv", "txt", "xlsx"], key="prot_up")

        current_sig = None
        if glyco_file and prot_file:
            current_sig = (glyco_file.name, glyco_file.size, prot_file.name, prot_file.size)

        st.divider()
        st.markdown("**1. Statistical thresholds**")
        metric_choice = st.radio("Significance metric", ["FDR (q-value)", "Nominal p-value"], index=0)
        use_q = metric_choice.startswith("FDR")

        glyco_alpha = st.number_input(
            "Glyco significance cutoff",
            min_value=1e-6,
            max_value=1.0,
            value=0.05,
            step=0.001,
            format="%.4f",
            help="A glycopeptide/gene passes the glyco side only when its selected p/q value is at or below this cutoff."
        )
        prot_alpha = st.number_input(
            "Proteome significance cutoff",
            min_value=1e-6,
            max_value=1.0,
            value=0.05,
            step=0.001,
            format="%.4f",
            help="A protein/gene passes the total-protein side only when its selected p/q value is at or below this cutoff."
        )
        glyco_lfc_thr = st.slider("Glyco |log2FC| threshold", 0.0, 5.0, 1.0, 0.1)
        prot_lfc_thr = st.slider("Proteome |log2FC| threshold", 0.0, 5.0, 1.0, 0.1)

        st.divider()
        st.markdown("**2. Glyco support filters**")
        min_sig_peptides = st.slider(
            "Minimum significant glycopeptides per gene",
            1,
            10,
            1,
            1,
            help="Increase this to require stronger glycopeptide-level support before a gene is called glyco-specific or concordant/discordant."
        )
        min_glyco_consistency = st.slider(
            "Minimum glyco directional consistency",
            0.50,
            1.00,
            0.67,
            0.01,
            help="Fraction of glycopeptides for a gene that must move in the same direction. Example: 0.67 means at least two-thirds agree."
        )
        min_intensity_ratio = st.number_input(
            "Minimum channel intensity for relative ratio plots",
            min_value=0.0,
            value=1.0,
            step=1.0,
            help="Only affects glyco/protein channel-level ratio visualizations. It does not change differential calls."
        )
        median_center_ratio = st.checkbox(
            "Median-center intensity columns before ratio plots",
            value=True,
            help="Recommended when input intensities are not already normalized for loading. This affects only ratio visualizations, not differential calls."
        )

        script_dir = Path(__file__).resolve().parent
        default_dir = script_dir / "assets" / "glycan_images"
        st.divider()
        st.markdown("**3. Asset paths**")
        user_input = st.text_input("Glycan Image Folder (Optional)", value=str(default_dir) if default_dir.exists() else "")
        glycan_image_dir = Path(user_input).expanduser().resolve() if user_input else default_dir
        if glycan_image_dir.is_dir():
            st.success(f"Image folder found: {glycan_image_dir.name}")
        else:
            # FIX: changed st.error to st.warning — the image folder is optional.
            # Showing an error here was misleading since the app works fine without glycan images.
            if user_input:
                st.warning("Image folder not found. Glycan structure images will not be rendered.")

        st.divider()
        if st.button("Reset Session Data", help="Clear uploads, cache, plots, and compiled reports."):
            st.cache_data.clear()
            st.session_state.clear()
            st.rerun()

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: center; font-size: 0.85em; color: #888;'>Developed in collaboration with the<br>Pandey Lab at Mayo Clinic</div>", unsafe_allow_html=True)

    glyco_df = read_table(glyco_file)
    prot_df = read_table(prot_file)

    #st.markdown("## Glyco-Proteome Integrator")
    #st.markdown("<p class='subtitle-text'>Integrate glycoproteome and total proteome differential results with evidence-aware gene summarization.</p>", unsafe_allow_html=True)

    if glyco_df.empty or prot_df.empty:
        render_professional_gpi_start_panel(upload_disabled=True)
        st.markdown("<br>", unsafe_allow_html=True)
        faq_section()
        return

    g_gene_col = detect_gene_col(glyco_df)
    g_lfc_col = detect_log2fc_col(glyco_df)
    g_p_col = detect_p_col(glyco_df)
    g_q_col = detect_q_col(glyco_df)
    g_int_cols = detect_intensity_cols(glyco_df)
    g_peptide_col = detect_feature_col(glyco_df, ["Peptide", "Annotated Sequence", "Sequence", "Peptide Sequence"])
    g_site_col = detect_glycosite_col(glyco_df)

    p_gene_col = detect_gene_col(prot_df)
    p_lfc_col = detect_log2fc_col(prot_df)
    p_p_col = detect_p_col(prot_df)
    p_q_col = detect_q_col(prot_df)
    p_int_cols = detect_intensity_cols(prot_df)
    p_protein_col = detect_feature_col(prot_df, ["Accession", "Protein", "Protein Accessions", "Master Protein Accessions"])

    for name, gene_col, lfc_col in [("Glyco", g_gene_col, g_lfc_col), ("Proteome", p_gene_col, p_lfc_col)]:
        if gene_col is None:
            st.error(f"Could not detect a gene identifier column in the {name} table.")
            return
        if lfc_col is None:
            st.error(f"Could not detect a log2FC column in the {name} table.")
            return

    g_metric_col = g_q_col if (use_q and g_q_col) else (g_p_col if g_p_col else g_q_col)
    p_metric_col = p_q_col if (use_q and p_q_col) else (p_p_col if p_p_col else p_q_col)
    if g_metric_col is None or p_metric_col is None:
        st.error("Could not detect p-value or q-value columns in one of the tables.")
        return

    glyco_df, prot_df, merged_gene = build_summary_tables(
        glyco_df, prot_df,
        g_gene_col, g_lfc_col, g_metric_col,
        p_gene_col, p_lfc_col, p_metric_col,
        glyco_alpha, prot_alpha,
        glyco_lfc_thr, prot_lfc_thr,
        min_sig_peptides, min_glyco_consistency,
        g_peptide_col, p_protein_col, g_site_col,
    )

    with st.expander("Detected input columns", expanded=False):
        cdet1, cdet2 = st.columns(2)
        with cdet1:
            st.markdown("**Glycoproteome**")
            st.write({
                "Gene": g_gene_col, "Glycosite/Site": g_site_col or "Not detected; falling back to peptide/unknown site",
                "Peptide": g_peptide_col, "log2FC": g_lfc_col, "Metric": g_metric_col,
                "Intensity columns detected": len(g_int_cols),
            })
        with cdet2:
            st.markdown("**Total proteome**")
            st.write({
                "Gene": p_gene_col, "Protein": p_protein_col, "log2FC": p_lfc_col,
                "Metric": p_metric_col, "Intensity columns detected": len(p_int_cols),
            })

    if current_sig is not None and current_sig != ss.get("upload_sig"):
        ss["upload_sig"] = current_sig
        _reset_module_flags()
        ss["show_overview"] = True
        ss["gpi_pdf_bytes"] = None
        ss["gpi_zip_bytes"] = None

    c1, c2, c3, c4 = st.columns(4)
    if c1.button(("▸ " if not ss["show_overview"] else "▾ ") + "Overview & Concordance", use_container_width=True):
        state = not ss["show_overview"]
        _reset_module_flags()
        ss["show_overview"] = state
    if c2.button(("▸ " if not ss["show_targeted"] else "▾ ") + "Targeted Gene Evidence", use_container_width=True):
        state = not ss["show_targeted"]
        _reset_module_flags()
        ss["show_targeted"] = state
    if c3.button(("▸ " if not ss["show_advanced"] else "▾ ") + "Advanced Views", use_container_width=True):
        state = not ss["show_advanced"]
        _reset_module_flags()
        ss["show_advanced"] = state
    if c4.button(("▸ " if not ss["show_export"] else "▾ ") + "Export", use_container_width=True):
        state = not ss["show_export"]
        _reset_module_flags()
        ss["show_export"] = state

    st.markdown("<br>", unsafe_allow_html=True)
    cmap = color_map_status()

    with st.expander("How to use this integration result", expanded=False):
        st.markdown(
            """
            **Recommended interpretation order**

            1. Start with **Overview & Concordance** to see whether glyco and total protein move together.
            2. Use **Targeted Gene Evidence** for individual genes; this is the most biologically interpretable view.
            3. Use **Advanced Views** only after thresholds are final. These plots are summary views, not primary evidence.

            **What the classes mean**
            - **Concordant**: glycopeptide signal and total protein signal move in the same direction.
            - **Discordant**: glycopeptide signal and total protein signal move in opposite directions. These are the most interesting candidates for altered glycosylation relative to protein abundance.
            - **Glyco-specific**: glycopeptides pass thresholds but total protein does not. This can suggest glycosylation-specific change, but should be checked at peptide/site level.
            - **Protein-specific**: total protein passes thresholds but glycopeptides do not.

            **Important caution**: the channel-level log2(Glyco/Protein) ratio is a **relative signal ratio, not glycan occupancy**. It can shift because of peptide detectability, ionization, missingness, or protein-loading effects. The app filters likely non-glycopeptide rows before ratio plots when glycan/site annotations are available, but the output should still be interpreted as relative evidence, not absolute occupancy.
            """
        )

    # ---------------------------------------------------------------------
    # Module 1: Overview
    # ---------------------------------------------------------------------
    if ss["show_overview"]:
        overview_subtab = st.radio(
            "Overview section",
            ["Integrated Gene Overview", "Global Glycopeptide Waterfall", "Global Glycopeptide Volcano"],
            horizontal=True,
            key="gpi_overview_subtab",
            label_visibility="collapsed"
        )

        if overview_subtab == "Integrated Gene Overview":
            with st.container(border=True):
                st.markdown("#### Multi-omic summary")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Glyco rows", f"{glyco_df.shape[0]:,}")
                m2.metric("Protein rows", f"{prot_df.shape[0]:,}")
                m3.metric("Integrated genes", f"{merged_gene.shape[0]:,}")
                m4.metric("Concordant genes", f"{merged_gene['Status'].isin(['Concordant Up','Concordant Down']).sum():,}")
                m5.metric("Discordant genes", f"{merged_gene['Status'].str.contains('Discordant', na=False).sum():,}")

            st.markdown("<br>", unsafe_allow_html=True)
            left, right = st.columns([0.42, 0.58], gap="large")

            with left:
                with st.container(border=True):
                    st.markdown("#### Concordance class distribution")
                    order = [
                        "Concordant Up", "Concordant Down",
                        "Discordant: Glyco Up / Protein Down", "Discordant: Glyco Down / Protein Up",
                        "Both Significant (Mixed Direction)",
                        "Glyco-specific Up", "Glyco-specific Down",
                        "Protein-specific Up", "Protein-specific Down",
                        "Non-significant",
                    ]
                    df_counts = merged_gene["Status"].value_counts().reindex(order).dropna().reset_index()
                    df_counts.columns = ["Status", "Count"]
                    fig_bar, ax_bar = plt.subplots(figsize=(7, 5.2))
                    sns.barplot(data=df_counts, x="Status", y="Count", ax=ax_bar, palette=[cmap.get(x, "#999") for x in df_counts["Status"]])
                    ax_bar.set_xlabel("")
                    ax_bar.set_ylabel("Genes")
                    ax_bar.set_title("Integrated status distribution", pad=12)
                    plt.setp(ax_bar.get_xticklabels(), rotation=35, ha="right")
                    plt.tight_layout()
                    st.pyplot(fig_bar)
                    add_plot_download_buttons(fig_bar, "gpi_status_distribution")

            with right:
                with st.container(border=True):
                    st.markdown("#### Global glyco vs protein concordance")
                    df_scatter = merged_gene.dropna(subset=["log2FC_glyco", "log2FC_prot"]).copy()
                    if df_scatter.empty:
                        st.warning("Insufficient overlap to generate the global concordance scatter.")
                    else:
                        df_scatter["BubbleSize"] = np.clip(df_scatter["glyco_peptide_count"].fillna(1) * 5, 8, 38)
                        fig_sc = px.scatter(
                            df_scatter,
                            x="log2FC_prot",
                            y="log2FC_glyco",
                            color="Status",
                            size="glyco_peptide_count",
                            hover_data=[
                                "Gene", "DisplayGeneName", "glyco_peptide_count",
                                "glyco_significant_peptides", "glyco_consistency", "metric_glyco_best", "metric_prot_best"
                            ],
                            labels={"log2FC_prot": "Protein log2FC", "log2FC_glyco": "Glyco log2FC"},
                            color_discrete_map=cmap,
                        )
                        fig_sc.add_vline(x=0, line_dash="dash", line_color="gray")
                        fig_sc.add_hline(y=0, line_dash="dash", line_color="gray")
                        fig_sc.add_vline(x=prot_lfc_thr, line_dash="dot", line_color="#444")
                        fig_sc.add_vline(x=-prot_lfc_thr, line_dash="dot", line_color="#444")
                        fig_sc.add_hline(y=glyco_lfc_thr, line_dash="dot", line_color="#444")
                        fig_sc.add_hline(y=-glyco_lfc_thr, line_dash="dot", line_color="#444")
                        fig_sc.update_layout(margin=dict(t=25, b=20, l=10, r=10), height=520)
                        st.plotly_chart(fig_sc, use_container_width=True)
                        add_plotly_download_buttons(fig_sc, "gpi_global_concordance_scatter")

                        cor_df = df_scatter[["log2FC_prot", "log2FC_glyco"]].dropna()
                        if cor_df.shape[0] > 2:
                            r_pear, p_pear = pearsonr(cor_df["log2FC_prot"], cor_df["log2FC_glyco"])
                            r_spear, p_spear = spearmanr(cor_df["log2FC_prot"], cor_df["log2FC_glyco"])
                            st.caption(
                                f"Pearson r = **{r_pear:.3f}** (p = **{p_pear:.2e}**) | "
                                f"Spearman ρ = **{r_spear:.3f}** (p = **{p_spear:.2e}**)."
                            )

            with st.container(border=True):
                st.markdown("#### Genes with strongest glyco-protein disagreement")
                top_disc = merged_gene.sort_values("Abs_Delta_log2FC", ascending=False).head(25).copy()
                show_cols = [
                    c for c in [
                        "Gene", "DisplayGeneName", "Status", "log2FC_glyco", "log2FC_prot", "Delta_log2FC",
                        "glyco_peptide_count", "glyco_significant_peptides", "glyco_consistency",
                        "metric_glyco_best", "metric_prot_best", "glyco_has_multigene_assignment"
                    ] if c in top_disc.columns
                ]
                st.dataframe(top_disc[show_cols], use_container_width=True, hide_index=True)
                add_dataframe_download_buttons(top_disc[show_cols], "gpi_top_disagreement_genes", key_suffix="top_disc")

            with st.container(border=True):
                st.markdown("#### Integrated gene result explorer")
                status_options = sorted(merged_gene["Status"].dropna().astype(str).unique().tolist())
                default_status = [s for s in status_options if s != "Non-significant"] or status_options
                selected_status = st.multiselect(
                    "Status classes to show",
                    options=status_options,
                    default=default_status,
                    key="gpi_overview_status_filter"
                )
                gene_search_overview = st.text_input("Search genes in integrated table", value="", key="gpi_overview_gene_search")
                max_rows_overview = st.slider("Rows to display", 25, 500, 100, 25, key="gpi_overview_rows")

                filtered_gene_view = merged_gene.copy()
                if selected_status:
                    filtered_gene_view = filtered_gene_view[filtered_gene_view["Status"].isin(selected_status)]
                if gene_search_overview.strip():
                    q = gene_search_overview.strip()
                    filtered_gene_view = filtered_gene_view[
                        filtered_gene_view["Gene"].astype(str).str.contains(q, case=False, na=False) |
                        filtered_gene_view["DisplayGeneName"].astype(str).str.contains(q, case=False, na=False)
                    ]

                explorer_cols = [
                    c for c in [
                        "Gene", "DisplayGeneName", "Status", "log2FC_glyco", "log2FC_prot",
                        "Delta_log2FC", "Abs_Delta_log2FC", "metric_glyco_best", "metric_prot_best",
                        "glyco_peptide_count", "glyco_significant_peptides", "glyco_consistency",
                        "glyco_direction", "prot_direction", "glyco_has_multigene_assignment"
                    ] if c in filtered_gene_view.columns
                ]
                filtered_gene_view = filtered_gene_view.sort_values("Abs_Delta_log2FC", ascending=False, na_position="last")
                st.dataframe(filtered_gene_view[explorer_cols].head(max_rows_overview), use_container_width=True, hide_index=True)
                add_dataframe_download_buttons(filtered_gene_view[explorer_cols], "gpi_integrated_gene_filtered", key_suffix="overview_filtered")

        if overview_subtab == "Global Glycopeptide Waterfall":
            #st.markdown("#### Structural Glycopeptide Waterfall")
            #st.caption(
            #    "A global view of individual glycopeptides ranked by log2 fold-change, "
            #    "paired with top driver feature structures."
            #)

            # -----------------------------
            # User controls
            # -----------------------------
            available_metric_cols = [
                c for c in ["p-value", "pvalue", "P.Value", "p_val", "q-value", "qvalue", "adj.P.Val", "FDR"]
                if c in glyco_df.columns
            ]

            if g_metric_col not in available_metric_cols and g_metric_col in glyco_df.columns:
                available_metric_cols.insert(0, g_metric_col)

            if not available_metric_cols:
                st.error("No p-value or q-value column found for significance filtering.")
                st.stop()

            selected_metric_col = st.radio(
                "Use significance metric",
                available_metric_cols,
                index=available_metric_cols.index(g_metric_col) if g_metric_col in available_metric_cols else 0,
                horizontal=True,
                key="wfl_metric_choice"
            )

            sig_alpha = st.number_input(
                f"{selected_metric_col} threshold",
                min_value=0.0,
                max_value=1.0,
                value=float(glyco_alpha),
                step=0.01,
                format="%.4f",
                key="wfl_alpha_threshold"
            )

            log2fc_thr = st.number_input(
                "log2FC threshold",
                min_value=0.0,
                value=float(glyco_lfc_thr),
                step=0.1,
                format="%.2f",
                key="wfl_log2fc_threshold"
            )

            # -----------------------------
            # Prepare waterfall data
            # -----------------------------
            wfl = glyco_df.dropna(subset=[g_lfc_col]).copy()
            wfl[g_lfc_col] = safe_numeric(wfl[g_lfc_col])
            wfl[selected_metric_col] = safe_numeric(wfl[selected_metric_col])

            wfl = wfl.dropna(subset=[g_lfc_col, selected_metric_col]).copy()
            wfl = wfl.sort_values(g_lfc_col, ascending=False).reset_index(drop=True)
            wfl["Rank"] = wfl.index + 1

            wfl["Status"] = "Non Significant"

            sig_up = (
                (wfl[g_lfc_col] >= log2fc_thr) &
                (wfl[selected_metric_col] <= sig_alpha)
            )

            sig_down = (
                (wfl[g_lfc_col] <= -log2fc_thr) &
                (wfl[selected_metric_col] <= sig_alpha)
            )

            wfl.loc[sig_up, "Status"] = "Up regulated"
            wfl.loc[sig_down, "Status"] = "Down regulated"

            col_plot, col_table = st.columns([0.45, 0.55], gap="large")

            with col_table:
                top_n_table = st.slider(
                    "Number of Top Features to Display",
                    3, 10, 5,
                    key="wfl_topn"
                )

                top_up = wfl[wfl["Status"] == "Up regulated"].head(top_n_table)

                top_down = (
                    wfl[wfl["Status"] == "Down regulated"]
                    .sort_values(g_lfc_col, ascending=True)
                    .head(top_n_table)
                )

                disp_df = pd.concat([top_up, top_down], ignore_index=True)

                wfl["Point_ID"] = np.nan

                if not disp_df.empty:
                    disp_df = disp_df.copy()
                    disp_df["Point_ID"] = range(1, len(disp_df) + 1)

                    id_map = disp_df.set_index("Rank")["Point_ID"].to_dict()
                    wfl["Point_ID"] = wfl["Rank"].map(id_map)

            # -----------------------------
            # Interactive Plotly waterfall
            # -----------------------------
            with col_plot:
                fig_wfl = px.scatter(
                    wfl,
                    x="Rank",
                    y=g_lfc_col,
                    color="Status",
                    color_discrete_map={
                        "Up regulated": "#d62728",
                        "Down regulated": "#1f77b4",
                        "Non Significant": "#cccccc",
                    },
                    hover_data=[
                        "Point_ID",
                        g_gene_col,
                        # FIX: g_peptide_col may be None; only include if present
                        *([g_peptide_col] if g_peptide_col else []),
                        g_lfc_col,
                        selected_metric_col,
                    ],
                    height=600,
                    title=f"Glycopeptide ({selected_metric_col} < {sig_alpha}, |log2FC| > {log2fc_thr})"
                )

                fig_wfl.add_hline(y=0, line_dash="solid", line_color="black")
                fig_wfl.add_hline(y=log2fc_thr, line_dash="dash", line_color="gray")
                fig_wfl.add_hline(y=-log2fc_thr, line_dash="dash", line_color="gray")

                # ✅ LABEL CODE MUST BE INSIDE THIS BLOCK
                label_df = wfl.dropna(subset=["Point_ID"]).copy()

                if not label_df.empty:
                    x_offset = (wfl["Rank"].max()) * 0.01
                    y_offset = (wfl[g_lfc_col].max() - wfl[g_lfc_col].min()) * 0.02

                    label_df["x_label"] = label_df["Rank"] + x_offset
                    label_df["y_label"] = label_df[g_lfc_col] + np.where(
                        label_df[g_lfc_col] > 0, y_offset, -y_offset
                    )

                    fig_wfl.add_scatter(
                        x=label_df["x_label"],
                        y=label_df["y_label"],
                        mode="text",
                        text=label_df["Point_ID"].astype(int).astype(str),
                        textfont=dict(size=12, color="black"),
                        showlegend=False,
                        hoverinfo="skip"
                    )

                    # connector lines (important for clarity)
                    for _, row in label_df.iterrows():
                        fig_wfl.add_shape(
                            type="line",
                            x0=row["Rank"],
                            y0=row[g_lfc_col],
                            x1=row["x_label"],
                            y1=row["y_label"],
                            line=dict(color="gray", width=1)
                        )

                fig_wfl.update_traces(marker=dict(size=5, opacity=0.85))

                max_rank = float(wfl["Rank"].max()) if not wfl.empty else 1.0
                fig_wfl.update_xaxes(range=[0, max_rank * 1.10])

                fig_wfl.update_layout(
                    xaxis_title="Glycopeptide",
                    yaxis_title="log2 Fold-Change",

                    # REMOVE GRID
                    xaxis=dict(
                        showgrid=False,
                        showline=True,
                        linewidth=2,
                        linecolor="black",
                        mirror=True,
                        ticks="outside",
                        tickwidth=1.5,
                        tickcolor="black"
                    ),
                    yaxis=dict(
                        showgrid=False,
                        showline=True,
                        linewidth=2,
                        linecolor="black",
                        mirror=True,
                        ticks="outside",
                        tickwidth=1.5,
                        tickcolor="black"
                    ),

                    legend=dict(
                        title="Status",
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.2,
                        bgcolor="rgba(255,255,255,0.85)"
                    ),

                    plot_bgcolor="white",
                    paper_bgcolor="white",

                    margin=dict(t=50, l=45, r=150, b=55)
                )

                st.plotly_chart(fig_wfl, use_container_width=True, config={"displaylogo": False})
                add_plotly_download_buttons(fig_wfl, "glycopeptide_waterfall_interactive")

            # -----------------------------
            # Formatted table + downloads
            # -----------------------------
            with col_table:
                if disp_df.empty:
                    st.info("No features meet the significance thresholds to display.")
                else:
                    st.markdown("##### Numbered Top Significant Features")

                    metric_label = "q-value" if "q" in selected_metric_col.lower() else "p-value"

                    html_table = format_structural_html_table(
                        disp_df,
                        glycan_image_dir,
                        g_gene_col,
                        g_peptide_col,
                        g_lfc_col,
                        selected_metric_col,
                        metric_label
                    )

                    st.markdown(html_table, unsafe_allow_html=True)

                    combined_png, combined_svg = create_side_by_side_waterfall_html_table_export(
                        wfl=wfl,
                        disp_df=disp_df,
                        html_table=html_table,
                        g_lfc_col=g_lfc_col,
                        metric_col=selected_metric_col,
                        sig_alpha=sig_alpha,
                        log2fc_thr=log2fc_thr,
                        img_dir=glycan_image_dir,
                        gene_col=g_gene_col,
                        pep_col=g_peptide_col
                    )

                    dcol1, dcol2, dcol3 = st.columns(3)

                    with dcol1:
                        if st.button("Save Combined PNG", key="save_wfl_table_png", use_container_width=True):
                            try:
                                outpath = _save_bytes_to_downloads(combined_png, "glycopeptide_waterfall_table.png")
                                st.success(f"Saved PNG to: {outpath}")
                            except Exception as e:
                                st.error(f"Could not save PNG: {e}")

                    with dcol2:
                        if st.button("Save Combined SVG", key="save_wfl_table_svg", use_container_width=True):
                            try:
                                outpath = _save_bytes_to_downloads(combined_svg, "glycopeptide_waterfall_table.svg")
                                st.success(f"Saved SVG to: {outpath}")
                            except Exception as e:
                                st.error(f"Could not save SVG: {e}")

                    with dcol3:
                        add_dataframe_download_buttons(disp_df, "glycopeptide_waterfall_top_features", key_suffix="wfl_top_features")

        if overview_subtab == "Global Glycopeptide Volcano":
            with st.container(border=True):
                render_glycopeptide_volcano(glyco_df, g_gene_col, g_peptide_col, g_lfc_col, g_metric_col, glyco_alpha, glyco_lfc_thr)

    # ---------------------------------------------------------------------
    # Module 2: Targeted evidence
    # ---------------------------------------------------------------------
    if ss["show_targeted"]:
        filters_left, filters_right = st.columns([0.35, 0.65], gap="large")
        with filters_left:
            with st.container(border=True):
                st.markdown("#### Gene selection")
                class_options = [
                    "All genes", "Concordant only", "Discordant only", "Glyco-specific only",
                    "Protein-specific only", "Non-significant only"
                ]
                class_choice = st.selectbox("Filter genes by class", class_options)

                gene_view = merged_gene.copy()
                if class_choice == "Concordant only":
                    gene_view = gene_view[gene_view["Status"].isin(["Concordant Up", "Concordant Down"])]
                elif class_choice == "Discordant only":
                    gene_view = gene_view[gene_view["Status"].str.contains("Discordant", na=False)]
                elif class_choice == "Glyco-specific only":
                    gene_view = gene_view[gene_view["Status"].str.contains("Glyco-specific", na=False)]
                elif class_choice == "Protein-specific only":
                    gene_view = gene_view[gene_view["Status"].str.contains("Protein-specific", na=False)]
                elif class_choice == "Non-significant only":
                    gene_view = gene_view[gene_view["Status"] == "Non-significant"]

                search_term = st.text_input("Search gene", value="")
                if search_term.strip():
                    q = search_term.strip().upper()
                    gene_view = gene_view[
                        gene_view["Gene"].astype(str).str.contains(q, case=False, na=False) |
                        gene_view["DisplayGeneName"].astype(str).str.contains(q, case=False, na=False)
                    ]

                gene_options = gene_view["Gene"].dropna().astype(str).tolist()
                if not gene_options:
                    st.warning("No genes available under the current filter.")
                else:
                    chosen_gene = st.selectbox("Select target gene", gene_options, index=0)

                    glyco_sub = glyco_df[glyco_df["GeneKey"] == chosen_gene].copy()
                    prot_sub = prot_df[prot_df["GeneKey"] == chosen_gene].copy()
                    gene_row = merged_gene[merged_gene["Gene"] == chosen_gene].head(1)
                    gene_name = gene_row["DisplayGeneName"].iloc[0] if not gene_row.empty else chosen_gene

                    st.divider()
                    st.markdown(f"**Selected gene:** {gene_name} (`{chosen_gene}`)")
                    if not gene_row.empty:
                        st.markdown(f"**Integrated status:** {gene_row['Status'].iloc[0]}")
                        st.markdown(f"**Glyco peptide count:** {int(gene_row['glyco_peptide_count'].fillna(0).iloc[0]) if 'glyco_peptide_count' in gene_row else glyco_sub.shape[0]}")
                        if 'glyco_consistency' in gene_row.columns:
                            gc = gene_row['glyco_consistency'].iloc[0]
                            st.markdown(f"**Glyco consistency:** {gc:.2f}" if pd.notna(gc) else "**Glyco consistency:** NA")

        with filters_right:
            if gene_options:
                tabs = st.tabs(["Peptide Evidence", "Channel Ratios", "Gene Summary", "Raw Tables"])

                with tabs[0]:
                    with st.container(border=True):
                        st.markdown("#### Peptide-level evidence")
                        if glyco_sub.empty:
                            st.info("No glycopeptide rows are available for this gene.")
                        else:
                            peptide_col = g_peptide_col if g_peptide_col else ("Peptide" if "Peptide" in glyco_sub.columns else glyco_sub.columns[0])
                            pe = glyco_sub[[peptide_col, g_lfc_col, g_metric_col]].copy()
                            pe[g_lfc_col] = safe_numeric(pe[g_lfc_col])
                            pe[g_metric_col] = safe_numeric(pe[g_metric_col])
                            pe = pe.dropna(subset=[g_lfc_col])
                            pe["Significant"] = (pe[g_lfc_col].abs() >= glyco_lfc_thr) & (pe[g_metric_col] <= glyco_alpha)
                            pe = pe.sort_values(g_lfc_col)

                            # -------------------------------------------------------------
                            # More intuitive peptide-level evidence plot
                            # -------------------------------------------------------------
                            from matplotlib.lines import Line2D
                            from matplotlib.patches import Patch

                            pe["Direction"] = np.where(pe[g_lfc_col] >= 0, "Up", "Down")
                            pe["Evidence_Class"] = "Not significant"
                            pe.loc[pe["Significant"] & (pe[g_lfc_col] >= 0), "Evidence_Class"] = "Significant glyco up"
                            pe.loc[pe["Significant"] & (pe[g_lfc_col] < 0), "Evidence_Class"] = "Significant glyco down"

                            evidence_colors = {
                                "Significant glyco up": "#d62728",
                                "Significant glyco down": "#1f77b4",
                                "Not significant": "#bdbdbd",
                            }
                            pe["Plot_Color"] = pe["Evidence_Class"].map(evidence_colors).fillna("#bdbdbd")
                            pe["Evidence_Strength"] = -np.log10(pe[g_metric_col].clip(lower=1e-300))
                            pe["Point_Size"] = np.clip(pe["Evidence_Strength"] * 24, 45, 260)

                            st.markdown(
                                f"""
                                **How to read this plot:** each dot is one glycopeptide for **{gene_name}**.  
                                Dots to the **right** are increased glycopeptides; dots to the **left** are decreased glycopeptides.  
                                The shaded center region is below the selected fold-change cutoff and should be treated as small-change/noise region.  
                                Larger dots have stronger statistical evidence because size is based on `-log10({g_metric_col})`.
                                """
                            )

                            fig_pep, ax_pep = plt.subplots(figsize=(10.5, max(4.0, 0.46 * len(pe))))
                            y = np.arange(len(pe))

                            # Shaded non-effect zone and threshold guides
                            ax_pep.axvspan(-glyco_lfc_thr, glyco_lfc_thr, color="#f2f2f2", alpha=0.9, zorder=0)
                            ax_pep.axvline(0, color="black", ls="--", lw=1.1, zorder=1)
                            ax_pep.axvline(glyco_lfc_thr, color="#444", ls=":", lw=1.2, zorder=1)
                            ax_pep.axvline(-glyco_lfc_thr, color="#444", ls=":", lw=1.2, zorder=1)

                            # Lollipop stems and points
                            stem_colors = np.where(pe[g_lfc_col] >= 0, "#d9a3a3", "#9eb6d8")
                            ax_pep.hlines(y=y, xmin=0, xmax=pe[g_lfc_col], color=stem_colors, lw=1.6, alpha=0.85, zorder=2)
                            ax_pep.scatter(
                                pe[g_lfc_col],
                                y,
                                c=pe["Plot_Color"],
                                s=pe["Point_Size"],
                                alpha=0.90,
                                edgecolors="black",
                                linewidths=0.35,
                                zorder=3,
                            )

                            # Gene-level summarized glyco signal
                            gene_summary_val = None
                            if not gene_row.empty and pd.notna(gene_row["log2FC_glyco"].iloc[0]):
                                gene_summary_val = float(gene_row["log2FC_glyco"].iloc[0])
                                ax_pep.axvline(
                                    gene_summary_val,
                                    color="#1b9e77",
                                    ls="-",
                                    lw=2.4,
                                    zorder=4,
                                )

                            # Keep labels readable for long peptide strings
                            def _short_peptide_label(x, max_len=58):
                                x = str(x)
                                return x if len(x) <= max_len else x[:max_len - 3] + "..."

                            ax_pep.set_yticks(y)
                            ax_pep.set_yticklabels([_short_peptide_label(x) for x in pe[peptide_col].astype(str)], fontsize=8)
                            ax_pep.set_xlabel("Glycopeptide log2 fold-change", fontweight="bold")
                            ax_pep.set_ylabel("Glycopeptide feature", fontweight="bold")
                            ax_pep.set_title("Peptide-level glycosylation evidence", pad=12, fontweight="bold")
                            ax_pep.grid(axis="x", linestyle="--", alpha=0.25)
                            ax_pep.grid(axis="y", visible=False)
                            ax_pep.spines["top"].set_visible(False)
                            ax_pep.spines["right"].set_visible(False)

                            # Legend explaining colors, lines, and shaded region
                            legend_handles = [
                                Line2D([0], [0], marker="o", color="white", label="Significant glyco up", markerfacecolor="#d62728", markeredgecolor="black", markersize=8),
                                Line2D([0], [0], marker="o", color="white", label="Significant glyco down", markerfacecolor="#1f77b4", markeredgecolor="black", markersize=8),
                                Line2D([0], [0], marker="o", color="white", label="Not significant", markerfacecolor="#bdbdbd", markeredgecolor="black", markersize=8),
                                Line2D([0], [0], color="#1b9e77", lw=2.4, label="Gene-level glyco summary" + (f" ({gene_summary_val:+.2f})" if gene_summary_val is not None else "")),
                                Line2D([0], [0], color="black", lw=1.1, ls="--", label="No change"),
                                Line2D([0], [0], color="#444", lw=1.2, ls=":", label="± log2FC cutoff"),
                                Patch(facecolor="#f2f2f2", edgecolor="none", label="Below FC cutoff"),
                            ]
                            ax_pep.legend(
                                handles=legend_handles,
                                title="Plot guide",
                                loc="upper left",
                                bbox_to_anchor=(1.02, 1.0),
                                frameon=True,
                                fontsize=8,
                                title_fontsize=9,
                            )

                            # Add a compact evidence summary below the plot
                            n_sig_up = int(((pe["Evidence_Class"] == "Significant glyco up")).sum())
                            n_sig_down = int(((pe["Evidence_Class"] == "Significant glyco down")).sum())
                            n_not_sig = int(((pe["Evidence_Class"] == "Not significant")).sum())

                            plt.tight_layout()
                            st.pyplot(fig_pep)
                            add_plot_download_buttons(fig_pep, f"{chosen_gene}_peptide_evidence")
                            st.caption(
                                f"Significant up: {n_sig_up} | Significant down: {n_sig_down} | "
                                f"Not significant: {n_not_sig}. Dot size represents -log10({g_metric_col})."
                            )

                with tabs[1]:
                    with st.container(border=True):
                        st.markdown("#### Channel-wise relative glyco-to-protein ratio")
                        ratio_df = build_long_ratio_df(
                            glyco_sub, prot_sub, g_int_cols, p_int_cols,
                            min_intensity=min_intensity_ratio, median_center=median_center_ratio, glycopeptide_only=True
                        )
                        if ratio_df.empty:
                            st.info("Insufficient matched intensity values to calculate channel-wise log2 ratios.")
                        else:
                            fig_ratio, ax_ratio = plt.subplots(figsize=(8.5, 4.5))
                            ax_ratio.plot(ratio_df["Sample"], ratio_df["log2Ratio"], marker="o", lw=2)
                            ax_ratio.axhline(0, color="black", ls="--", lw=1)
                            ax_ratio.set_ylabel("log2(Glyco / Protein)")
                            ax_ratio.set_xlabel("TMT channel")
                            ax_ratio.set_title("Relative glyco-to-protein ratio by channel", pad=12)
                            plt.setp(ax_ratio.get_xticklabels(), rotation=45, ha="right")
                            plt.tight_layout()
                            st.pyplot(fig_ratio)
                            add_plot_download_buttons(fig_ratio, f"{chosen_gene}_log2_ratio")
                            st.dataframe(ratio_df, use_container_width=True, hide_index=True)
                            add_dataframe_download_buttons(ratio_df, f"{chosen_gene}_relative_ratio_table", key_suffix=f"ratio_{chosen_gene}")
                            ratio_consistency = directional_consistency(ratio_df["log2Ratio"])
                            ratio_cv = ratio_df["log2Ratio"].std() / ratio_df["log2Ratio"].mean().abs() if ratio_df["log2Ratio"].mean() != 0 else np.nan
                            st.caption(
                                f"Relative signal ratio only — not glycan occupancy. "
                                f"Sample-direction consistency: {ratio_consistency:.2f}; CV on log2 ratios: {ratio_cv:.2f}. "
                                "Interpret cautiously because peptide detectability, ionization, and missingness can change this value."
                            )

                with tabs[2]:
                    with st.container(border=True):
                        st.markdown("#### Gene-level integration summary")
                        if gene_row.empty:
                            st.info("No gene-level summary available.")
                        else:
                            summary_cols = [
                                c for c in [
                                    "Gene", "DisplayGeneName", "Status",
                                    "log2FC_glyco", "log2FC_prot", "Delta_log2FC",
                                    "metric_glyco_best", "metric_prot_best",
                                    "glyco_site_count", "glyco_significant_sites", "glyco_site_direction_mixed",
                                    "glyco_peptide_count", "glyco_significant_peptides",
                                    "glyco_consistency", "glyco_consistency_sig",
                                    "glyco_direction", "prot_direction",
                                    "glyco_has_multigene_assignment"
                                ] if c in gene_row.columns
                            ]
                            summary_download_df = gene_row[summary_cols].T.rename(columns={gene_row.index[0]: "Value"})
                            st.dataframe(summary_download_df, use_container_width=True)
                            add_dataframe_download_buttons(summary_download_df.reset_index().rename(columns={"index": "Field"}), f"{chosen_gene}_gene_summary", key_suffix=f"summary_{chosen_gene}")

                with tabs[3]:
                    with st.container(border=True):
                        st.markdown("#### Raw evidence tables")
                        st.markdown("**Glycoproteome rows**")
                        st.dataframe(glyco_sub, use_container_width=True)
                        add_dataframe_download_buttons(glyco_sub, f"{chosen_gene}_glycoproteome_rows", key_suffix=f"glyco_rows_{chosen_gene}")
                        st.markdown("**Proteome rows**")
                        st.dataframe(prot_sub, use_container_width=True)
                        add_dataframe_download_buttons(prot_sub, f"{chosen_gene}_proteome_rows", key_suffix=f"prot_rows_{chosen_gene}")

    # ---------------------------------------------------------------------
    # Module 3: Advanced views
    # ---------------------------------------------------------------------
    if ss["show_advanced"]:
        left, right = st.columns([0.35, 0.65], gap="large")
        with left:
            with st.container(border=True):
                vis_choice = st.radio("Select plot", [
                    "Directional UpSet", "Hierarchical Clustermap", "Relative Ratio Heatmap", "Ratio Consistency Dot Plot", "Gene Set Enrichment", "Discordant Rank Plot"
                ])
                st.divider()
                top_n_hm = st.slider("Top N genes for clustering", 10, 200, min(40, max(10, merged_gene.shape[0])), 5)
                top_n_ratio = st.slider("Top N genes for ratio heatmap", 10, 100, 25, 5)

        with right:
            with st.container(border=True):
                if vis_choice == "Directional UpSet":
                    if not HAVE_UPSET:
                        st.info("Install `upsetplot` to enable this panel.")
                    else:
                        sets = {
                            "Glyco Up": set(merged_gene.loc[merged_gene["Status"].isin(["Concordant Up", "Glyco-specific Up", "Discordant: Glyco Up / Protein Down"]), "Gene"].dropna().astype(str)),
                            "Glyco Down": set(merged_gene.loc[merged_gene["Status"].isin(["Concordant Down", "Glyco-specific Down", "Discordant: Glyco Down / Protein Up"]), "Gene"].dropna().astype(str)),
                            "Protein Up": set(merged_gene.loc[merged_gene["Status"].isin(["Concordant Up", "Protein-specific Up", "Discordant: Glyco Down / Protein Up"]), "Gene"].dropna().astype(str)),
                            "Protein Down": set(merged_gene.loc[merged_gene["Status"].isin(["Concordant Down", "Protein-specific Down", "Discordant: Glyco Up / Protein Down"]), "Gene"].dropna().astype(str)),
                        }
                        sets = {k: v for k, v in sets.items() if len(v) > 0}
                        if not sets:
                            st.info("No significant directional sets under the current thresholds.")
                        else:
                            fig_up = plt.figure(figsize=(7.4, 5.3))
                            UpSet(from_contents(sets), show_counts=True, facecolor="black", shading_color="lightgray").plot(fig=fig_up)
                            st.pyplot(fig_up, use_container_width=True)
                            add_plot_download_buttons(fig_up, "directional_upset")
                            st.caption("Direction-aware set overlap is often more informative than a simple binary significant/non-significant overlap.")

                elif vis_choice == "Hierarchical Clustermap":
                    hm_subset = merged_gene[merged_gene["Status"] != "Non-significant"].dropna(subset=["log2FC_glyco", "log2FC_prot"]).copy()
                    if hm_subset.empty:
                        st.info("No significant features are available for clustering.")
                    else:
                        hm_subset = hm_subset.sort_values("Abs_Delta_log2FC", ascending=False).head(top_n_hm)
                        mat = hm_subset.set_index("Gene")[["log2FC_glyco", "log2FC_prot"]].copy()
                        mat.columns = ["Glyco log2FC", "Protein log2FC"]
                        cg = sns.clustermap(mat, cmap="coolwarm", center=0, linewidths=0.2, figsize=(7, max(5, 0.22 * len(mat))), cbar_kws={"label": "log2FC"})
                        cg.fig.suptitle("Hierarchical clustering of differential signal", y=1.02)
                        st.pyplot(cg.fig, use_container_width=True)
                        add_plot_download_buttons(cg.fig, "hierarchical_clustermap")

                elif vis_choice == "Relative Ratio Heatmap":
                    ratio_mat = build_global_ratio_matrix(
                        merged_gene, glyco_df, prot_df, g_int_cols, p_int_cols,
                        min_intensity=min_intensity_ratio, median_center=median_center_ratio, glycopeptide_only=True
                    )
                    if ratio_mat.empty:
                        st.info("No gene-by-channel ratio matrix could be created with the current inputs.")
                    else:
                        keep = ratio_mat.mean(axis=1).sort_values(ascending=False).head(top_n_ratio).index
                        ratio_top = ratio_mat.loc[keep]
                        fig_occ, ax_occ = plt.subplots(figsize=(8.5, max(4.5, 0.28 * len(ratio_top))))
                        sns.heatmap(ratio_top.fillna(0), cmap="vlag", center=0, ax=ax_occ, cbar_kws={"label": "log2(Glyco / Protein)", "shrink": 0.65})
                        ax_occ.set_ylabel("")
                        ax_occ.set_title("Relative glyco-to-protein ratio heatmap", pad=12)
                        plt.tight_layout()
                        st.pyplot(fig_occ)
                        add_plot_download_buttons(fig_occ, "relative_ratio_heatmap")
                        ratio_stats = build_ratio_consistency_table(ratio_mat)
                        st.caption(
                            "Positive values indicate relatively stronger glyco signal than protein signal for that gene/channel. "
                            "This is not glycan occupancy; it is a relative signal ratio after filtering likely non-glycopeptide rows where annotations permit."
                        )
                        with st.expander("Per-gene sample consistency table", expanded=False):
                            st.dataframe(ratio_stats, use_container_width=True, hide_index=True)
                            add_dataframe_download_buttons(ratio_stats, "relative_ratio_consistency_table", key_suffix="ratio_heatmap_consistency")

                elif vis_choice == "Ratio Consistency Dot Plot":
                    ratio_mat = build_global_ratio_matrix(
                        merged_gene, glyco_df, prot_df, g_int_cols, p_int_cols,
                        min_intensity=min_intensity_ratio, median_center=median_center_ratio, glycopeptide_only=True
                    )
                    ratio_stats = build_ratio_consistency_table(ratio_mat)
                    if ratio_stats.empty:
                        st.info("No matched channel/sample ratio data are available for consistency analysis.")
                    else:
                        ratio_stats = ratio_stats.merge(merged_gene[["Gene", "Status"]], on="Gene", how="left")
                        plot_df = ratio_stats.dropna(subset=["mean_log2Ratio", "sample_direction_consistency"]).copy()
                        fig_rc = px.scatter(
                            plot_df,
                            x="mean_log2Ratio",
                            y="sample_direction_consistency",
                            color="Status",
                            size="n_samples",
                            hover_data=["Gene", "n_up_samples", "n_down_samples", "sd_log2Ratio", "cv_abs_log2Ratio"],
                            labels={
                                "mean_log2Ratio": "Mean log2(Glyco / Protein)",
                                "sample_direction_consistency": "Per-sample direction consistency",
                            },
                            color_discrete_map=cmap,
                            height=560,
                        )
                        fig_rc.add_vline(x=0, line_dash="dash", line_color="gray")
                        fig_rc.add_hline(y=0.5, line_dash="dot", line_color="gray")
                        fig_rc.update_layout(plot_bgcolor="white", paper_bgcolor="white")
                        st.plotly_chart(fig_rc, use_container_width=True, config={"displaylogo": False})
                        add_plotly_download_buttons(fig_rc, "ratio_consistency_dotplot")
                        st.caption("This view flags genes where the average ratio hides sample disagreement. Values near 0.5 mean samples split between up and down.")
                        st.dataframe(ratio_stats.sort_values(["sample_direction_consistency", "sd_log2Ratio"], ascending=[True, False]), use_container_width=True, hide_index=True)
                        add_dataframe_download_buttons(ratio_stats, "ratio_consistency_table", key_suffix="ratio_consistency")

                elif vis_choice == "Gene Set Enrichment":
                    st.markdown("#### Simple gene-set enrichment")
                    st.caption("Upload a GMT file or a two-column CSV/TSV where column 1 is pathway/gene-set name and column 2 is gene symbol. Uses Fisher's exact test against genes present in the integrated table.")
                    gene_set_file = st.file_uploader("Gene set file (.gmt, .csv, .tsv, .txt)", type=["gmt", "csv", "tsv", "txt"], key="gpi_gene_set_file")
                    enrich_class = st.selectbox(
                        "Genes to test",
                        ["Discordant genes", "Glyco-specific genes", "Protein-specific genes", "Concordant genes", "All significant genes"],
                        key="gpi_enrich_class",
                    )
                    background = set(merged_gene["Gene"].dropna().astype(str).str.upper())
                    if enrich_class == "Discordant genes":
                        selected = set(merged_gene.loc[merged_gene["Status"].str.contains("Discordant", na=False), "Gene"].dropna().astype(str).str.upper())
                    elif enrich_class == "Glyco-specific genes":
                        selected = set(merged_gene.loc[merged_gene["Status"].str.contains("Glyco-specific", na=False), "Gene"].dropna().astype(str).str.upper())
                    elif enrich_class == "Protein-specific genes":
                        selected = set(merged_gene.loc[merged_gene["Status"].str.contains("Protein-specific", na=False), "Gene"].dropna().astype(str).str.upper())
                    elif enrich_class == "Concordant genes":
                        selected = set(merged_gene.loc[merged_gene["Status"].isin(["Concordant Up", "Concordant Down"]), "Gene"].dropna().astype(str).str.upper())
                    else:
                        selected = set(merged_gene.loc[merged_gene["Status"] != "Non-significant", "Gene"].dropna().astype(str).str.upper())
                    gene_sets = parse_gene_set_file(gene_set_file)
                    if gene_set_file is None:
                        st.info("Upload a gene set file to run enrichment. This avoids hard-coding pathway databases and makes the result reproducible offline.")
                    elif not gene_sets:
                        st.warning("Could not parse any gene sets from the uploaded file.")
                    elif not selected:
                        st.info("No genes are available for the selected class under the current thresholds.")
                    else:
                        enrich_df = run_simple_fisher_enrichment(selected, background, gene_sets)
                        if enrich_df.empty:
                            st.info("No enriched gene sets had at least one overlapping selected gene.")
                        else:
                            st.dataframe(enrich_df.head(100), use_container_width=True, hide_index=True)
                            add_dataframe_download_buttons(enrich_df, "gpi_gene_set_enrichment", key_suffix="gene_set_enrichment")
                            top_plot = enrich_df.head(20).copy()
                            top_plot["neg_log10_p"] = -np.log10(top_plot["p_value"].clip(lower=1e-300))
                            fig_en = px.bar(top_plot.iloc[::-1], x="neg_log10_p", y="GeneSet", orientation="h", hover_data=["Overlap", "SetSizeInBackground", "Genes"], labels={"neg_log10_p": "-log10(p-value)", "GeneSet": "Gene set"}, height=max(420, 24 * len(top_plot)))
                            st.plotly_chart(fig_en, use_container_width=True, config={"displaylogo": False})
                            add_plotly_download_buttons(fig_en, "gpi_gene_set_enrichment")

                elif vis_choice == "Discordant Rank Plot":
                    disc = merged_gene.sort_values("Abs_Delta_log2FC", ascending=False).head(20).copy()
                    if disc.empty:
                        st.info("No genes available.")
                    else:
                        fig_d, ax_d = plt.subplots(figsize=(8.5, max(4.5, 0.32 * len(disc))))
                        y = np.arange(len(disc))
                        ax_d.scatter(disc["log2FC_prot"], y, label="Protein", s=60)
                        ax_d.scatter(disc["log2FC_glyco"], y, label="Glyco", s=60)
                        for i, (_, row) in enumerate(disc.iterrows()):
                            ax_d.plot([row["log2FC_prot"], row["log2FC_glyco"]], [i, i], lw=1.6, color="#888")
                        ax_d.axvline(0, color="black", ls="--", lw=1)
                        ax_d.set_yticks(y)
                        ax_d.set_yticklabels(disc["Gene"])
                        ax_d.set_xlabel("log2FC")
                        ax_d.set_title("Genes ranked by glyco-protein divergence", pad=12)
                        ax_d.legend()
                        plt.tight_layout()
                        st.pyplot(fig_d)
                        add_plot_download_buttons(fig_d, "discordant_rank_plot")

    # ---------------------------------------------------------------------
    # Module 4: Export
    # ---------------------------------------------------------------------
    if ss["show_export"]:
        with st.container(border=True):
            col_info, col_action = st.columns([0.65, 0.35], gap="large")
            with col_info:
                st.markdown("#### Reproducible export bundle")
                st.markdown(
                    "<p class='subtitle-text'>The export bundle regenerates a standard report from the current thresholds and support filters.</p>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "- **PDF report** with standard overview figures\n"
                    "- **PNG plot assets** from the regenerated figure set\n"
                    "- **Integrated gene matrix** plus standardized input tables\n"
                    "- **README** describing the ratio interpretation"
                )
            with col_action:
                if ss.get("gpi_pdf_bytes") is None:
                    if st.button("Compile PDF & ZIP Bundle", type="primary", use_container_width=True):
                        with st.spinner("Generating report bundle..."):
                            figure_pack = generate_standard_figures(
                                merged_gene, glyco_df, prot_df,
                                g_int_cols, p_int_cols,
                                glyco_lfc_thr, prot_lfc_thr,
                                top_n_heatmap=40,
                                top_n_ratio=25,
                                min_intensity_ratio=min_intensity_ratio,
                                median_center_ratio=median_center_ratio,
                            )
                            pdf_b, zip_b = generate_report_bundle(merged_gene, glyco_df, prot_df, figure_pack)
                            ss["gpi_pdf_bytes"] = pdf_b
                            ss["gpi_zip_bytes"] = zip_b
                            st.rerun()
                else:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                    st.success("Report generated successfully.")
                    b1, b2 = st.columns(2)
                    with b1:
                        if st.button("Save PDF Report", key="save_gpi_pdf_report", use_container_width=True, type="primary"):
                            try:
                                outpath = _save_bytes_to_downloads(ss["gpi_pdf_bytes"], f"GPI_Report_{ts}.pdf")
                                st.success(f"Saved PDF to: {outpath}")
                            except Exception as e:
                                st.error(f"Could not save PDF: {e}")
                    with b2:
                        if st.button("Save ZIP Bundle", key="save_gpi_zip_bundle", use_container_width=True):
                            try:
                                outpath = _save_bytes_to_downloads(ss["gpi_zip_bytes"], f"GPI_Report_Bundle_{ts}.zip")
                                st.success(f"Saved ZIP to: {outpath}")
                            except Exception as e:
                                st.error(f"Could not save ZIP: {e}")
                    if st.button("Regenerate report", use_container_width=True):
                        ss["gpi_pdf_bytes"] = None
                        ss["gpi_zip_bytes"] = None
                        st.rerun()

    if not any([ss["show_overview"], ss["show_targeted"], ss["show_advanced"], ss["show_export"]]):
        faq_section()


if __name__ == "__main__":
    main()