#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import uuid
from io import BytesIO
from pathlib import Path

import streamlit as st
import matplotlib as mpl
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------
# Global Matplotlib Settings
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


def _get_downloads_dir() -> Path:
    """
    Return user's Downloads folder if available.
    Falls back to home directory.
    """
    downloads = Path.home() / "Downloads"
    if downloads.exists():
        return downloads
    return Path.home()


def _safe_filename(filename: str) -> str:
    """
    Remove characters that can cause problems in filenames.
    """
    filename = str(filename)
    filename = re.sub(r"[^\w\-.]+", "_", filename)
    filename = filename.strip("_")
    return filename or "plot"


def _save_bytes_to_downloads(data: bytes, filename: str) -> Path:
    """
    Save bytes directly to local Downloads folder.
    This is more reliable in packaged Streamlit desktop apps.
    """
    outdir = _get_downloads_dir()
    outpath = outdir / _safe_filename(filename)

    counter = 1
    original_outpath = outpath

    while outpath.exists():
        outpath = original_outpath.with_name(
            f"{original_outpath.stem}_{counter}{original_outpath.suffix}"
        )
        counter += 1

    outpath.write_bytes(data)
    return outpath


def add_plot_download_buttons(fig, base_filename: str = "plot", key_suffix: str = ""):
    """
    Add PNG and SVG export buttons for a matplotlib figure.

    In normal browser mode, Streamlit download buttons work.
    In packaged desktop mode, the embedded webview may open PNG/SVG/PDF
    inside the app instead of downloading. Therefore, this function also
    provides direct Save-to-Downloads buttons.
    """

    if not key_suffix:
        key_suffix = str(uuid.uuid4())[:8]

    base_filename = _safe_filename(base_filename)

    # ------------------------------------------------------------------
    # Generate PNG bytes
    # ------------------------------------------------------------------
    png_buf = BytesIO()
    fig.savefig(png_buf, format="png", dpi=300, bbox_inches="tight")
    png_bytes = png_buf.getvalue()

    # ------------------------------------------------------------------
    # Generate SVG bytes
    # ------------------------------------------------------------------
    svg_buf = BytesIO()
    fig.savefig(svg_buf, format="svg", dpi=300, bbox_inches="tight")
    svg_bytes = svg_buf.getvalue()

    png_filename = f"{base_filename}.png"
    svg_filename = f"{base_filename}.svg"

    # ------------------------------------------------------------------
    # Desktop-safe direct save buttons
    # ------------------------------------------------------------------
    save_col1, save_col2 = st.columns(2)

    with save_col1:
        if st.button(
            "Download PNG",
            key=f"save_png_{base_filename}_{key_suffix}",
            use_container_width=True,
            help="Directly save PNG to your local Downloads folder."
        ):
            try:
                outpath = _save_bytes_to_downloads(png_bytes, png_filename)
                st.success(f"Saved PNG to: {outpath}")
            except Exception as e:
                st.error(f"Could not save PNG: {e}")

    with save_col2:
        if st.button(
            "Download SVG",
            key=f"save_svg_{base_filename}_{key_suffix}",
            use_container_width=True,
            help="Directly save SVG to your local Downloads folder."
        ):
            try:
                outpath = _save_bytes_to_downloads(svg_bytes, svg_filename)
                st.success(f"Saved SVG to: {outpath}")
            except Exception as e:
                st.error(f"Could not save SVG: {e}")

    plt.close(fig)


def classify_glycan(glycan_str: str) -> str:
    """
    Classify glycans based on their structure.
    Returns 'Complex/Hybrid', 'Paucimannose', 'High-mannose', or 'Unknown'.
    """
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