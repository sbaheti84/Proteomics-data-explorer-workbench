#!/usr/bin/env python3
"""
Central browser-safe download helpers for all GlycoExplorer / Total Protein / GPI Streamlit apps.

Key rule:
    Hosted/RConnect mode must use st.download_button because Python runs on the server.
    Local packaged .exe mode should not rely on browser downloads. Use a native Save As
    dialog or save directly to Downloads from the local Python process.
"""
from __future__ import annotations

import os
import re
import uuid
import mimetypes
import inspect
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
import streamlit as st

_TRUE_VALUES = {"1", "true", "yes", "y", "on", "hosted", "connect"}
_FALSE_VALUES = {"0", "false", "no", "n", "off", "local", "standalone"}


def env_to_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    value = str(value).strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    return None


def is_hosted_mode() -> bool:
    """Detect RConnect/Posit Connect hosted mode."""
    explicit = env_to_bool(os.environ.get("GLYCOEXPLORER_HOSTED"))
    if explicit is not None:
        return explicit
    if os.environ.get("POSIT_PRODUCT", "").upper() == "CONNECT":
        return True
    if os.environ.get("RSTUDIO_PRODUCT", "").upper() == "CONNECT":
        return True
    return any(os.environ.get(v) for v in [
        "CONNECT_CONTENT_GUID", "RSCONNECT_CONTENT_GUID", "RSCONNECT_APP_NAME",
        "CONNECT_SERVER", "POSIT_CONNECT", "RSTUDIO_CONNECT",
    ])


def get_save_mode() -> str:
    """
    Return browser/local/dialog.

    browser:
        Use Streamlit/browser download. Required for Posit Connect/RConnect.
    local:
        Save automatically to Downloads or GLYCOEXPLORER_DOWNLOAD_DIR.
    dialog:
        Local executable mode. Open a native Save As dialog and let the user
        choose the output path. This only works when the Streamlit process is
        running on the user's own computer.
    """
    mode = os.environ.get("GLYCOEXPLORER_SAVE_MODE", "browser").strip().lower()
    if mode in {"browser", "local", "dialog", "saveas", "save_as"}:
        if mode in {"saveas", "save_as"}:
            return "dialog"
        return mode
    return "browser"


def safe_filename(filename: str) -> str:
    filename = str(filename).strip()
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1F]+', "_", filename)
    filename = re.sub(r"[^A-Za-z0-9._-]+", "_", filename)
    filename = filename.strip("._")
    return filename or "downloaded_file"



def make_streamlit_key(*parts: Any, caller_depth: int = 2) -> str:
    """Build a deterministic, call-site-aware Streamlit key.

    This prevents DuplicateElementKey errors when several plots use the same
    helper default such as key_prefix="plotly". The key includes the requested
    filename/base name plus the caller file and line number.
    """
    clean_parts = [safe_filename(str(x)) for x in parts if x is not None and str(x) != ""]
    try:
        frame = inspect.stack()[caller_depth]
        loc = f"{Path(frame.filename).stem}_{frame.lineno}"
    except Exception:
        loc = uuid.uuid4().hex[:8]
    raw = "_".join(clean_parts + [loc]) or f"download_{uuid.uuid4().hex[:8]}"
    raw = safe_filename(raw)
    # Streamlit accepts long keys, but shorter keys are easier to debug.
    return raw[:180]

# Backward-compatible name used in older GE code.
def _safe_download_filename(filename: str) -> str:
    return safe_filename(filename)


def guess_mime(filename: str) -> str:
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
    if lower.endswith(".html") or lower.endswith(".htm"):
        return "text/html"
    return "application/octet-stream"


def coerce_to_bytes(data: Any) -> bytes:
    if data is None:
        raise ValueError("No data was provided for download.")
    if isinstance(data, BytesIO):
        return data.getvalue()
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray):
        return bytes(data)
    if isinstance(data, memoryview):
        return data.tobytes()
    if isinstance(data, Path):
        return data.read_bytes()
    if isinstance(data, str):
        possible_path = Path(data).expanduser()
        if possible_path.exists() and possible_path.is_file():
            return possible_path.read_bytes()
        return data.encode("utf-8")
    return bytes(data)

# Backward-compatible name used in older code.
def _coerce_to_bytes(data: Any) -> bytes:
    return coerce_to_bytes(data)


def dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    buffer = BytesIO()
    safe_sheet = str(sheet_name or "Sheet1")[:31]
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=safe_sheet)
    return buffer.getvalue()


def _get_downloads_dir() -> Path:
    """Local-only compatibility helper. Do not use in hosted mode."""
    custom_dir = os.environ.get("GLYCOEXPLORER_DOWNLOAD_DIR", "").strip()
    if custom_dir:
        p = Path(custom_dir).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p
    downloads = Path.home() / "Downloads"
    return downloads if downloads.exists() else Path.home()


def save_bytes_local_only(data: Any, filename: str) -> Path:
    """Local fallback only when GLYCOEXPLORER_SAVE_MODE=local and not hosted."""
    if is_hosted_mode():
        raise RuntimeError("Direct local saving is disabled in hosted/RConnect mode.")
    outdir = _get_downloads_dir()
    outdir.mkdir(parents=True, exist_ok=True)
    safe_name = safe_filename(filename)
    outpath = outdir / safe_name
    base = outpath
    n = 1
    while outpath.exists():
        outpath = base.with_name(f"{base.stem}_{n}{base.suffix}")
        n += 1
    outpath.write_bytes(coerce_to_bytes(data))
    return outpath


def _filetypes_for_dialog(filename: str):
    """Return native file-dialog filters based on extension."""
    ext = Path(filename).suffix.lower()
    mapping = {
        ".csv": [("CSV files", "*.csv")],
        ".tsv": [("TSV files", "*.tsv")],
        ".txt": [("Text files", "*.txt")],
        ".xlsx": [("Excel files", "*.xlsx")],
        ".xls": [("Excel files", "*.xls")],
        ".pdf": [("PDF files", "*.pdf")],
        ".zip": [("ZIP files", "*.zip")],
        ".png": [("PNG images", "*.png")],
        ".svg": [("SVG images", "*.svg")],
        ".html": [("HTML files", "*.html")],
        ".htm": [("HTML files", "*.htm")],
    }
    return mapping.get(ext, []) + [("All files", "*.*")]


def ask_user_save_path(default_filename: str) -> Optional[Path]:
    """Open a native Save As dialog for local executable mode."""
    if is_hosted_mode():
        raise RuntimeError("Save As dialog is disabled in hosted/RConnect mode.")

    safe_name = safe_filename(default_filename)
    ext = Path(safe_name).suffix

    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as e:
        raise RuntimeError(
            "Could not open a native Save As dialog because tkinter is not available. "
            "Install/ship tkinter with the executable, or use browser download mode."
        ) from e

    root = tk.Tk()
    root.withdraw()
    try:
        # Try to keep the dialog above the Streamlit/browser window.
        try:
            root.attributes("-topmost", True)
            root.update()
            root.lift()
            root.focus_force()
        except Exception:
            pass

        selected = filedialog.asksaveasfilename(
            title="Save file as",
            initialfile=safe_name,
            defaultextension=ext if ext else None,
            filetypes=_filetypes_for_dialog(safe_name),
        )
        if not selected:
            return None
        return Path(selected).expanduser()
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def save_bytes_with_dialog(data: Any, filename: str) -> Optional[Path]:
    """Ask the local user where to save a file, then write bytes there."""
    outpath = ask_user_save_path(filename)
    if outpath is None:
        return None
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_bytes(coerce_to_bytes(data))
    return outpath


def add_download_button(
    data: Any,
    file_name: str,
    label: Optional[str] = None,
    key: Optional[str] = None,
    mime: Optional[str] = None,
    use_container_width: bool = True,
    help: Optional[str] = None,
    type: str = "secondary",
):
    """Download/save helper for CSV/XLSX/PNG/SVG/PDF/HTML/ZIP/etc.

    Hosted/RConnect mode:
        Uses st.download_button because a server cannot open a local file picker
        on the user's machine.

    Local executable mode with GLYCOEXPLORER_SAVE_MODE=dialog:
        Uses a normal st.button plus tkinter.asksaveasfilename so the user can
        choose the exact save location.
    """
    safe_name = safe_filename(file_name)
    file_bytes = coerce_to_bytes(data)
    save_mode = get_save_mode()
    button_key = make_streamlit_key(key or "download", safe_name, caller_depth=3)

    if save_mode in {"dialog", "local"} and not is_hosted_mode():
        if st.button(
            label=label or f"Save {safe_name}",
            key=button_key,
            use_container_width=use_container_width,
            help=help,
            type=type,
        ):
            try:
                if save_mode == "dialog":
                    outpath = save_bytes_with_dialog(file_bytes, safe_name)
                    if outpath is None:
                        st.info("Save cancelled.")
                        return False
                else:
                    outpath = save_bytes_local_only(file_bytes, safe_name)
                st.success(f"Saved to: {outpath}")
                return True
            except Exception as e:
                st.error(f"Could not save file: {e}")
                return False
        return False

    return st.download_button(
        label=label or f"Download {safe_name}",
        data=file_bytes,
        file_name=safe_name,
        mime=mime or guess_mime(safe_name),
        key=button_key,
        use_container_width=use_container_width,
        help=help,
        type=type,
    )


def add_browser_download_button(label: str, data: Any, file_name: str, mime: str, key: str, use_container_width: bool = True):
    return add_download_button(data=data, file_name=file_name, label=label, key=key, mime=mime, use_container_width=use_container_width)


def add_direct_save_button(label: str, data: Any, file_name: str, key: str, use_container_width: bool = True):
    """Compatibility wrapper. Uses the configured save mode."""
    return add_download_button(
        data=data,
        file_name=file_name,
        label=label,
        key=key,
        use_container_width=use_container_width,
    )


def add_file_download_buttons(data: Any, file_name: str, mime: Optional[str] = None, label: str = "Download file", key_suffix: str = ""):
    safe_name = safe_filename(file_name)
    return add_download_button(
        data=data,
        file_name=safe_name,
        label=label,
        key=make_streamlit_key("download", safe_name, key_suffix or "file", caller_depth=3),
        mime=mime or guess_mime(safe_name),
    )


def add_hosted_safe_file_download_button(label, data, file_name, key, use_container_width=True):
    """Backward-compatible helper used in launchers. Always browser-safe by default."""
    return add_download_button(data=data, file_name=file_name, label=label, key=key, use_container_width=use_container_width)


def add_bytes_download_button(label: str, data: Any, filename: Optional[str] = None, file_name: Optional[str] = None, key: Optional[str] = None, key_suffix: str = "", help_text: Optional[str] = None, type: str = "secondary", use_container_width: bool = True):
    fname = filename or file_name or "downloaded_file"
    k = key or make_streamlit_key("download", safe_filename(fname), key_suffix or "bytes", caller_depth=2)
    return add_download_button(data=data, file_name=fname, label=label, key=k, help=help_text, type=type, use_container_width=use_container_width)

# Total-protein launcher alias.
def add_bytes_download_button_tp(label: str, data: Any, filename: str, key: str, help_text: str = "Download file.", type: str = "secondary"):
    return add_bytes_download_button(label=label, data=data, filename=filename, key=key, help_text=help_text, type=type)


def add_dataframe_download_buttons(
    df: pd.DataFrame,
    base_filename: str = "table",
    key_prefix: str = "table",
    sheet_name: str = "Data",
    key_suffix: str = "",
    csv_label: str = "Download CSV",
    excel_label: str = "Download Excel",
    include_csv: bool = True,
    include_excel: bool = True,
    use_container_width: bool = True,
    **kwargs,
):
    """CSV/XLSX dataframe export with backward-compatible arguments.

    Why this exists:
        Older launchers call this helper with ``key_suffix=...`` while newer
        code may use ``key_prefix=...``. The previous utility accepted only
        ``key_prefix`` and broke GPI with:
            TypeError: unexpected keyword argument 'key_suffix'

    RConnect behavior:
        Uses Streamlit browser downloads only. The user's browser decides the
        destination folder. To get a folder prompt, enable the browser setting
        'Ask where to save each file before downloading'.
    """
    if df is None or getattr(df, "empty", True):
        st.info("No table available for download.")
        return

    # Accept common legacy aliases without breaking callers.
    if "filename" in kwargs and not base_filename:
        base_filename = kwargs.get("filename")
    if "file_name" in kwargs and not base_filename:
        base_filename = kwargs.get("file_name")

    base = safe_filename(base_filename or "table").rsplit(".", 1)[0]
    # key_suffix is the historical GE/GPI argument; make it part of the key so
    # repeated tables on the same page do not collide.
    key_token = safe_filename(key_suffix or key_prefix or base)

    if include_csv and include_excel:
        col1, col2 = st.columns(2)
        with col1:
            add_download_button(
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=f"{base}.csv",
                label=csv_label,
                key=make_streamlit_key(key_token, base, "csv", caller_depth=3),
                mime="text/csv",
                use_container_width=use_container_width,
            )
        with col2:
            add_download_button(
                data=dataframe_to_excel_bytes(df, sheet_name=sheet_name),
                file_name=f"{base}.xlsx",
                label=excel_label,
                key=make_streamlit_key(key_token, base, "xlsx", caller_depth=3),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=use_container_width,
            )
        return

    if include_csv:
        add_download_button(
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"{base}.csv",
            label=csv_label,
            key=make_streamlit_key(key_token, base, "csv", caller_depth=3),
            mime="text/csv",
            use_container_width=use_container_width,
        )

    if include_excel:
        add_download_button(
            data=dataframe_to_excel_bytes(df, sheet_name=sheet_name),
            file_name=f"{base}.xlsx",
            label=excel_label,
            key=make_streamlit_key(key_token, base, "xlsx", caller_depth=3),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=use_container_width,
        )


def _figure_to_bytes(fig, fmt: str, dpi: int = 300) -> bytes:
    buf = BytesIO()
    fmt = fmt.lower().lstrip(".")
    kwargs = {"format": fmt, "bbox_inches": "tight", "facecolor": "white"}
    if fmt == "png":
        kwargs["dpi"] = dpi
    fig.savefig(buf, **kwargs)
    return buf.getvalue()


def add_matplotlib_download_buttons(fig, base_filename: str = "plot", key_prefix: str = "plot", key_suffix: str = "", include_png: bool = True, include_svg: bool = True, include_pdf: bool = True):
    if fig is None:
        st.warning("No figure available for download.")
        return
    base = safe_filename(base_filename).rsplit(".", 1)[0]
    kp = safe_filename(key_suffix or (base if key_prefix == "plotly" else key_prefix) or base)
    options = []
    if include_png:
        options.append(("PNG", "png", "image/png"))
    if include_svg:
        options.append(("SVG", "svg", "image/svg+xml"))
    if include_pdf:
        options.append(("PDF", "pdf", "application/pdf"))
    if not options:
        return
    cols = st.columns(len(options))
    for col, (label, fmt, mime) in zip(cols, options):
        with col:
            add_download_button(
                data=_figure_to_bytes(fig, fmt),
                file_name=f"{base}.{fmt}",
                label=f"Download {label}",
                key=make_streamlit_key(kp, base, fmt, caller_depth=3),
                mime=mime,
            )

# Historical GE name.
def add_plot_download_buttons(fig, base_filename: str = "plot", key_suffix: str = ""):
    return add_matplotlib_download_buttons(fig=fig, base_filename=base_filename, key_prefix=key_suffix or safe_filename(base_filename), include_png=True, include_svg=True, include_pdf=True)

# Historical TP name.
def add_figure_download_buttons_tp(fig, base_filename: str, key_suffix: str = "", include_pdf: bool = True):
    return add_matplotlib_download_buttons(fig=fig, base_filename=base_filename, key_prefix=key_suffix or safe_filename(base_filename), include_pdf=include_pdf)

def add_figure_download_buttons_tp_lf(fig, base_filename: str, key_suffix: str = "", include_pdf: bool = True):
    return add_figure_download_buttons_tp(fig, base_filename, key_suffix, include_pdf)


def add_plotly_download_buttons(fig, base_filename: str = "interactive_plot", key_prefix: str = "plotly", key_suffix: str = "", include_html: bool = True, include_png: bool = False, include_svg: bool = False):
    if fig is None:
        st.warning("No figure available for download.")
        return
    base = safe_filename(base_filename).rsplit(".", 1)[0]
    kp = safe_filename(key_suffix or (base if key_prefix == "plotly" else key_prefix) or base)
    buttons = []
    if include_html:
        try:
            html_bytes = fig.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8")
            buttons.append(("Interactive HTML", "html", "text/html", html_bytes))
        except Exception as e:
            st.warning(f"Could not prepare Plotly HTML export: {e}")
    for fmt, mime in [("png", "image/png"), ("svg", "image/svg+xml")]:
        if (fmt == "png" and include_png) or (fmt == "svg" and include_svg):
            try:
                data = fig.to_image(format=fmt, scale=2)
                buttons.append((fmt.upper(), fmt, mime, data))
            except Exception as e:
                st.warning(f"Could not prepare Plotly {fmt.upper()} export. Install kaleido if needed. {e}")
    if not buttons:
        return
    cols = st.columns(len(buttons))
    for col, (label, fmt, mime, data) in zip(cols, buttons):
        with col:
            add_download_button(
                data=data,
                file_name=f"{base}.{fmt}",
                label=f"Download {label}",
                key=make_streamlit_key(kp, base, fmt, caller_depth=3),
                mime=mime,
            )

# Historical TP names.
def add_plotly_html_download_button_tp(fig, base_filename: str, key_suffix: str = ""):
    return add_plotly_download_buttons(fig=fig, base_filename=base_filename, key_prefix=key_suffix or safe_filename(base_filename), include_html=True)

def add_plotly_html_download_button_tp_lf(fig, base_filename: str, key_suffix: str = ""):
    return add_plotly_html_download_button_tp(fig, base_filename, key_suffix)


def add_figure_download_buttons_common(fig, base_filename: str, key_suffix: str = "", include_pdf: bool = True, save_to_downloads: bool = False):
    return add_matplotlib_download_buttons(fig=fig, base_filename=base_filename, key_prefix=key_suffix or safe_filename(base_filename), include_pdf=include_pdf)


def add_plotly_html_download_button_common(fig, base_filename: str, key_suffix: str = "", save_to_downloads: bool = False):
    return add_plotly_html_download_button_tp(fig=fig, base_filename=base_filename, key_suffix=key_suffix)


def explain_download_location():
    st.caption(
        "Downloads are handled by your browser. To choose a folder every time, enable your browser setting "
        "'Ask where to save each file before downloading'. In RConnect, the app cannot directly open a local file picker."
    )

# Backward-compatible wrappers. These no longer write to server in hosted/browser mode.
def _save_bytes_to_downloads(data: Any, filename: str):
    save_mode = get_save_mode()
    if save_mode == "dialog" and not is_hosted_mode():
        return save_bytes_with_dialog(data, filename)
    if save_mode == "local" and not is_hosted_mode():
        return save_bytes_local_only(data, filename)
    add_download_button(data=data, file_name=filename, label=f"Download {safe_filename(filename)}", key=make_streamlit_key("compat_download", safe_filename(filename), caller_depth=2))
    return f"Browser download prepared for {safe_filename(filename)}"


def _save_file_to_downloads(source_path: Union[str, Path], filename: Optional[str] = None):
    source_path = Path(source_path)
    return _save_bytes_to_downloads(source_path.read_bytes(), filename or source_path.name)


def save_profile_pdf_to_downloads_tp(profile_figs, gene_names, filename: str = "top_protein_profiles.pdf"):
    """Compatibility: build a multipage PDF and expose it as a browser download."""
    from matplotlib.backends.backend_pdf import PdfPages
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        for fig in profile_figs or []:
            if fig is not None:
                pdf.savefig(fig, bbox_inches="tight")
    return _save_bytes_to_downloads(buf.getvalue(), filename)

def save_profile_pdf_to_downloads_tp_lf(profile_figs, gene_names, filename: str = "top_protein_profiles.pdf"):
    return save_profile_pdf_to_downloads_tp(profile_figs, gene_names, filename)
