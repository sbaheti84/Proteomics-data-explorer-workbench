#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import uuid
from io import BytesIO
from pathlib import Path
from typing import Union

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


# -------------------------------------------------------------------------
# Save / Download Mode Helpers
# -------------------------------------------------------------------------
_TRUE_VALUES = {"1", "true", "yes", "y", "on", "hosted", "connect"}
_FALSE_VALUES = {"0", "false", "no", "n", "off", "local", "standalone"}


def _env_to_bool(value):
    """
    Convert environment variable text to True/False/None.
    """
    if value is None:
        return None

    value = str(value).strip().lower()

    if value in _TRUE_VALUES:
        return True

    if value in _FALSE_VALUES:
        return False

    return None


def _is_hosted_mode() -> bool:
    """True when the app is not running on the user's own computer.

    Delegates so every download button in the project answers this the same way.
    See download_utility.is_hosted_mode.
    """
    from download_utility import is_hosted_mode as _shared

    return _shared()


def _get_save_mode() -> str:
    """How downloads should be delivered: dialog, browser, local or hybrid.

    Delegates so every download button in the project answers this the same way.
    See download_utility.get_save_mode.
    """
    from download_utility import get_save_mode as _shared

    return _shared()


# -------------------------------------------------------------------------
# File Path / Filename Helpers
# -------------------------------------------------------------------------
def _get_downloads_dir() -> Path:
    """The folder a direct save writes to on this computer.

    Delegates so every direct save resolves its destination the same way, and so
    none can write to a server. See download_utility._get_downloads_dir.
    """
    from download_utility import _get_downloads_dir as _shared

    return _shared()


def _safe_filename(filename: str) -> str:
    """
    Remove characters that can cause problems in filenames across Windows,
    macOS, Linux, browser downloads, and server environments.
    """

    filename = str(filename).strip()

    # Replace path separators and invalid Windows/macOS characters.
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1F]+', "_", filename)

    # Replace whitespace and other unusual characters.
    filename = re.sub(r"[^\w\-.]+", "_", filename)

    # Avoid empty or hidden-only names.
    filename = filename.strip("._")

    return filename or "plot"


def _coerce_to_bytes(data: Union[bytes, bytearray, BytesIO, str]) -> bytes:
    """
    Convert common file-like objects to bytes.
    """

    if isinstance(data, BytesIO):
        return data.getvalue()

    if isinstance(data, bytes):
        return data

    if isinstance(data, bytearray):
        return bytes(data)

    if isinstance(data, str):
        return data.encode("utf-8")

    raise TypeError(f"Unsupported data type for saving/downloading: {type(data)}")


def _as_save_label(label: str) -> str:
    """
    Reword a download button for the mode that asks where to put the file.

    "Download PNG" becomes "Save PNG as...", which is what the button now does.
    """
    text = str(label).strip()

    if text.lower().startswith("download"):
        rest = text[len("download"):].strip()
        return f"Save {rest} as..." if rest else "Save as..."

    return f"Save {text} as..." if text else "Save as..."


def _save_bytes_to_downloads(data: Union[bytes, bytearray, BytesIO, str], filename: str) -> Path:
    """
    Save bytes directly to the local Downloads folder.

    Use only for local standalone desktop mode.
    On RConnect/hosted mode, this writes to the server/container, not the user laptop.
    """

    outdir = _get_downloads_dir()
    outdir.mkdir(parents=True, exist_ok=True)

    safe_name = _safe_filename(filename)
    outpath = outdir / safe_name

    counter = 1
    original_outpath = outpath

    while outpath.exists():
        outpath = original_outpath.with_name(
            f"{original_outpath.stem}_{counter}{original_outpath.suffix}"
        )
        counter += 1

    outpath.write_bytes(_coerce_to_bytes(data))

    return outpath


# -------------------------------------------------------------------------
# General Download Button Helpers
# -------------------------------------------------------------------------
def add_browser_download_button(
    label: str,
    data: Union[bytes, bytearray, BytesIO, str],
    file_name: str,
    mime: str,
    key: str,
    use_container_width: bool = True,
):
    """
    Browser-based download.

    Works on:
        - RConnect/Posit Connect
        - Streamlit in normal browser
        - Standalone apps that open in the user's browser

    The browser controls the final folder. The app controls the suggested filename.
    """

    st.download_button(
        label=label,
        data=_coerce_to_bytes(data),
        file_name=_safe_filename(file_name),
        mime=mime,
        key=key,
        width=("stretch" if use_container_width else "content"),
    )


def add_direct_save_button(
    label: str,
    data: Union[bytes, bytearray, BytesIO, str],
    file_name: str,
    key: str,
    use_container_width: bool = True,
):
    """
    Direct local save button.

    Intended for standalone Mac/Windows packaged mode.
    Hidden automatically in hosted mode unless GLYCOEXPLORER_SAVE_MODE=local/hybrid.
    """

    if st.button(
        label,
        key=key,
        width=("stretch" if use_container_width else "content"),
        help="Standalone mode only: save directly to the local Downloads folder.",
    ):
        try:
            outpath = _save_bytes_to_downloads(data, file_name)
            st.success(f"Saved to: {outpath}")
        except Exception as e:
            st.error(f"Could not save file: {e}")


def add_file_download_buttons(
    data: Union[bytes, bytearray, BytesIO, str],
    file_name: str,
    mime: str,
    label: str = "Download file",
    key_suffix: str = "",
):
    """
    Universal helper for CSV, TSV, Excel, PDF, ZIP, PNG, SVG, etc.

    Hosted/RConnect:
        Browser download only.

    Standalone:
        Browser download plus direct Save-to-Downloads fallback.
    """

    if not key_suffix:
        key_suffix = str(uuid.uuid4())[:8]

    file_name = _safe_filename(file_name)
    mode = _get_save_mode()

    if mode in {"browser", "hybrid"}:
        add_browser_download_button(
            label=label,
            data=data,
            file_name=file_name,
            mime=mime,
            key=f"browser_download_{file_name}_{key_suffix}",
            use_container_width=True,
        )

    if mode in {"local", "hybrid", "dialog"}:
        # dialog opens a Save As box, so name the button for what it does.
        direct_label = label if mode == "local" else _as_save_label(label)

        add_direct_save_button(
            label=direct_label,
            data=data,
            file_name=file_name,
            key=f"direct_save_{file_name}_{key_suffix}",
            use_container_width=True,
        )


# -------------------------------------------------------------------------
# Plot Download Buttons
# -------------------------------------------------------------------------
def add_plot_download_buttons(fig, base_filename: str = "plot", key_suffix: str = ""):
    """
    Add PNG and SVG export buttons for a matplotlib figure.

    Hosted/RConnect behavior:
        - Shows browser Download PNG/SVG buttons only.
        - Does not write to /local/opt/rstudio-connect/... server paths.

    Standalone Windows/Mac behavior:
        - Shows browser Download PNG/SVG buttons.
        - Also shows Save PNG/SVG to Downloads fallback buttons.
    """

    if fig is None:
        st.warning("No figure available to download.")
        return

    if not key_suffix:
        key_suffix = str(uuid.uuid4())[:8]

    base_filename = _safe_filename(base_filename)
    mode = _get_save_mode()

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
    # Browser download buttons
    # ------------------------------------------------------------------
    if mode in {"browser", "hybrid"}:
        col1, col2 = st.columns(2)

        with col1:
            add_browser_download_button(
                label="Download PNG",
                data=png_bytes,
                file_name=png_filename,
                mime="image/png",
                key=f"browser_png_{base_filename}_{key_suffix}",
                use_container_width=True,
            )

        with col2:
            add_browser_download_button(
                label="Download SVG",
                data=svg_bytes,
                file_name=svg_filename,
                mime="image/svg+xml",
                key=f"browser_svg_{base_filename}_{key_suffix}",
                use_container_width=True,
            )

    # ------------------------------------------------------------------
    # Local direct save buttons
    # ------------------------------------------------------------------
    if mode in {"local", "hybrid", "dialog"}:
        save_col1, save_col2 = st.columns(2)

        asks = mode in {"hybrid", "dialog"}
        png_label = "Save PNG as..." if asks else "Download PNG"
        svg_label = "Save SVG as..." if asks else "Download SVG"

        with save_col1:
            add_direct_save_button(
                label=png_label,
                data=png_bytes,
                file_name=png_filename,
                key=f"direct_png_{base_filename}_{key_suffix}",
                use_container_width=True,
            )

        with save_col2:
            add_direct_save_button(
                label=svg_label,
                data=svg_bytes,
                file_name=svg_filename,
                key=f"direct_svg_{base_filename}_{key_suffix}",
                use_container_width=True,
            )

    plt.close(fig)


# -------------------------------------------------------------------------
# Glycan Classification
# -------------------------------------------------------------------------
def classify_glycan(glycan_str: str) -> str:
    """
    Classify glycans based on their structure.

    Expected format:
        H N A F

    Returns:
        High-mannose
        Paucimannose
        Complex/Hybrid
        Unknown
    """

    match = re.match(r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", str(glycan_str))

    if not match:
        return "Unknown"

    H, N, A, F = map(int, match.groups())

    if 4 <= H <= 12 and N == 2 and A == 0 and F in [0, 1]:
        return "High-mannose"

    if 0 <= H <= 3 and N in [1, 2] and A == 0 and F in [0, 1]:
        return "Paucimannose"

    return "Complex/Hybrid"

# =============================================================================
# Centralized browser-safe download overrides
# =============================================================================
try:
    from download_utility import (
        safe_filename,
        _safe_download_filename,
        coerce_to_bytes,
        _coerce_to_bytes,
        add_download_button,
        add_browser_download_button,
        add_direct_save_button,
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
