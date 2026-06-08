#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Common plotting and utility helpers shared by Total Protein TMT and Total Protein LFQ apps.

Keep app-specific statistical engines in their own launchers. This module is for
shared UI/download helpers, group label helpers, and reusable MA/volcano plots.
"""

import re
import uuid
from io import BytesIO
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except Exception:
    HAS_ADJUST_TEXT = False


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------
def df_is_nonempty(df) -> bool:
    return isinstance(df, pd.DataFrame) and not df.empty


def safe_filename(filename: str) -> str:
    """Clean filename for Streamlit/browser or desktop app downloads."""
    filename = str(filename).strip()
    filename = re.sub(r"[^A-Za-z0-9._-]+", "_", filename)
    filename = filename.strip("_")
    return filename or "downloaded_file"


def get_downloads_dir() -> Path:
    downloads = Path.home() / "Downloads"
    return downloads if downloads.exists() else Path.home()


def save_bytes_to_downloads(data, filename: str) -> Path:
    """Save bytes directly to the user's local Downloads folder without overwriting."""
    outdir = get_downloads_dir()
    outpath = outdir / safe_filename(filename)

    if isinstance(data, BytesIO):
        data = data.getvalue()
    if isinstance(data, str):
        data = data.encode("utf-8")
    if not isinstance(data, (bytes, bytearray)):
        data = bytes(data)

    original_outpath = outpath
    counter = 1
    while outpath.exists():
        outpath = original_outpath.with_name(
            f"{original_outpath.stem}_{counter}{original_outpath.suffix}"
        )
        counter += 1

    outpath.write_bytes(data)
    return outpath


def figure_to_bytes(fig, fmt: str, dpi: int = 300) -> bytes:
    """Convert a matplotlib figure to bytes."""
    buf = BytesIO()
    fmt = fmt.lower()
    if fmt == "pdf":
        fig.savefig(buf, format="pdf", bbox_inches="tight")
    elif fmt == "svg":
        fig.savefig(buf, format="svg", bbox_inches="tight")
    else:
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return buf.getvalue()


def add_figure_download_buttons_common(
    fig,
    base_filename: str,
    key_suffix: str = "",
    include_pdf: bool = True,
    save_to_downloads: bool = False,
):
    """
    Shared figure export buttons.

    save_to_downloads=False:
        browser-native st.download_button, useful for web Streamlit.
    save_to_downloads=True:
        desktop-app style save-to-Downloads buttons, useful for packaged apps.
    """
    if fig is None:
        return

    if not key_suffix:
        key_suffix = str(uuid.uuid4())[:8]

    base_filename = safe_filename(base_filename)
    png_bytes = figure_to_bytes(fig, "png")
    svg_bytes = figure_to_bytes(fig, "svg")
    pdf_bytes = figure_to_bytes(fig, "pdf") if include_pdf else None

    cols = st.columns(3 if include_pdf else 2)

    def _button(label, data, filename, mime, key, help_text):
        if save_to_downloads:
            clicked = st.button(label, key=key, use_container_width=True, help=help_text)
            if clicked:
                try:
                    outpath = save_bytes_to_downloads(data, filename)
                    st.success(f"Saved {filename} to: {outpath}")
                except Exception as e:
                    st.error(f"Could not save {filename}: {e}")
        else:
            clicked = st.download_button(
                label=label,
                data=data,
                file_name=filename,
                mime=mime,
                key=key,
                use_container_width=True,
                help=help_text,
            )
            if clicked:
                st.success(f"Downloaded {filename}")

    with cols[0]:
        _button(
            "Download PNG",
            png_bytes,
            f"{base_filename}.png",
            "image/png",
            f"save_png_{base_filename}_{key_suffix}",
            "Save PNG file.",
        )

    with cols[1]:
        _button(
            "Download SVG",
            svg_bytes,
            f"{base_filename}.svg",
            "image/svg+xml",
            f"save_svg_{base_filename}_{key_suffix}",
            "Save SVG file.",
        )

    if include_pdf:
        with cols[2]:
            _button(
                "Download PDF",
                pdf_bytes,
                f"{base_filename}.pdf",
                "application/pdf",
                f"save_pdf_{base_filename}_{key_suffix}",
                "Save PDF file.",
            )


def add_plotly_html_download_button_common(
    fig,
    base_filename: str,
    key_suffix: str = "",
    save_to_downloads: bool = False,
):
    """Shared Plotly HTML export button."""
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

    filename = f"{base_filename}.html"
    key = f"save_html_{base_filename}_{key_suffix}"

    if save_to_downloads:
        if st.button(
            "Download Interactive HTML",
            key=key,
            use_container_width=True,
            help="Save the interactive Plotly figure to your local Downloads folder.",
        ):
            try:
                outpath = save_bytes_to_downloads(html_bytes, filename)
                st.success(f"Saved HTML to: {outpath}")
            except Exception as e:
                st.error(f"Could not save HTML: {e}")
    else:
        clicked = st.download_button(
            label="Download Interactive HTML",
            data=html_bytes,
            file_name=filename,
            mime="text/html",
            key=key,
            use_container_width=True,
        )
        if clicked:
            st.success(f"Downloaded {filename}")


# -----------------------------------------------------------------------------
# Shared data / group helpers
# -----------------------------------------------------------------------------
def add_gene_id_and_row_id(
    raw_df: pd.DataFrame,
    accession_candidates: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Add GeneID and RowID using the same logic used by both Total Protein apps.
    """
    raw_df = raw_df.copy()
    accession_candidates = accession_candidates or [
        "Accession",
        "Master Protein Accessions",
        "Protein ID",
        "Protein.IDs",
    ]

    gene_col_match = next((c for c in raw_df.columns if re.search(r"(?i)^gene", str(c))), None)

    if gene_col_match:
        raw_df["GeneID"] = raw_df[gene_col_match]
    elif "Description" in raw_df.columns:
        raw_df["GeneID"] = raw_df["Description"].apply(
            lambda x: re.search(r"GN=([\w\-]+)", str(x)).group(1)
            if pd.notna(x) and re.search(r"GN=([\w\-]+)", str(x))
            else None
        )

    accession_col = next((c for c in accession_candidates if c in raw_df.columns), None)

    if "GeneID" in raw_df.columns and accession_col:
        raw_df["RowID"] = raw_df["GeneID"].astype(str) + "_" + raw_df[accession_col].astype(str)
    elif "GeneID" in raw_df.columns:
        raw_df["RowID"] = raw_df["GeneID"].astype(str)
    elif accession_col:
        raw_df["RowID"] = raw_df[accession_col].astype(str)
    else:
        raw_df["RowID"] = "Feature_" + raw_df.index.astype(str)

    return raw_df


def make_pretty_sample_labels(sample_cols: list, clean_func: Callable[[str], str]) -> dict:
    """Create unique display labels from raw sample column names."""
    pretty_labels = {}
    used_labels = set()

    for col in sample_cols:
        base_label = clean_func(col)
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


def get_pairwise_group_labels_common(
    ss,
    default_control: str = "Control",
    default_test: str = "Test",
) -> tuple[str, str]:
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


def make_pairwise_sig_mask_common(
    res: pd.DataFrame,
    metric: str,
    alpha_thr: float,
    fc_thresh: float,
) -> pd.Series:
    """Create the same pairwise significance mask used across plots."""
    return (res[metric] <= alpha_thr) & (res["log2FC"].abs() >= fc_thresh)


# -----------------------------------------------------------------------------
# Shared MA / Volcano plots
# -----------------------------------------------------------------------------
def _prepare_volcano_df(res: pd.DataFrame, metric: str) -> pd.DataFrame:
    plot_df = res.copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df["log2FC"] = pd.to_numeric(plot_df["log2FC"], errors="coerce")
    plot_df = plot_df.dropna(subset=["log2FC", metric]).copy()
    plot_df["-log10(metric)"] = -np.log10(plot_df[metric].clip(lower=1e-300))
    return plot_df


def create_static_volcano_plot_common(
    res: pd.DataFrame,
    gene_col_name: Optional[str],
    metric: str,
    alpha_thr: float,
    fc_thresh: float,
    top_n_annot: int = 0,
    control_label: str = "Control",
    test_label: str = "Test",
    show_fc_lines: bool = False,
    show_title: bool = False,
):
    """Create a reusable static volcano plot for Total Protein TMT and LFQ."""
    plot_df = _prepare_volcano_df(res, metric)

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
    if show_fc_lines:
        ax.axvline(fc_thresh, color="black", linestyle="--", linewidth=1)
        ax.axvline(-fc_thresh, color="black", linestyle="--", linewidth=1)

    if top_n_annot and top_n_annot > 0 and gene_col_name and gene_col_name in plot_df.columns:
        annot_df = plot_df.loc[mask_sig].copy()
        if not annot_df.empty:
            annot_df["abs_log2FC"] = annot_df["log2FC"].abs()
            annot_df = (
                annot_df.sort_values([metric, "abs_log2FC"], ascending=[True, False])
                .head(int(top_n_annot))
            )

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
                adjust_text(
                    texts,
                    ax=ax,
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.7),
                )

    if show_title:
        ax.set_title(
            f"Volcano Plot ({metric} < {alpha_thr}, |log2FC| > {fc_thresh})",
            fontweight="bold",
        )

    ax.set_xlabel(
        fr"log$_2$ fold-change ({test_label}/{control_label})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel(
        fr"$-$log$_{{10}}$ ({metric})",
        fontsize=12,
        fontweight="bold",
    )

    ax.grid(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    return fig


def create_interactive_volcano_plot_common(
    res: pd.DataFrame,
    sig_mask: pd.Series,
    gene_col_name: Optional[str],
    metric: str,
    alpha_thr: float,
    fc_thresh: float,
    control_label: str,
    test_label: str,
    show_fc_lines: bool = True,
):
    """Create a reusable interactive Plotly volcano plot."""
    plot_df = res.copy()
    plot_df["-log10(p)"] = -np.log10(
        pd.to_numeric(plot_df[metric], errors="coerce").clip(lower=1e-10)
    )
    plot_df["Sig"] = np.where(sig_mask, "Significant", "NS")

    x_axis_label = f"log<sub>2</sub> fold-change ({test_label}/{control_label})"
    y_axis_label = f"-log10({metric})"

    fig = px.scatter(
        plot_df,
        x="log2FC",
        y="-log10(p)",
        color="Sig",
        hover_data=[gene_col_name] if gene_col_name else [],
        color_discrete_map={"Significant": "#d62728", "NS": "#a9a9a9"},
        labels={"log2FC": x_axis_label, "-log10(p)": y_axis_label, "Sig": "Status"},
    )

    fig.add_hline(y=-np.log10(alpha_thr), line_dash="dash", line_color="black")
    if show_fc_lines:
        fig.add_vline(x=fc_thresh, line_dash="dash", line_color="black")
        fig.add_vline(x=-fc_thresh, line_dash="dash", line_color="black")

    fig.update_xaxes(title_text=x_axis_label, showgrid=False, zeroline=False)
    fig.update_yaxes(title_text=y_axis_label, showgrid=False, zeroline=False)
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=30, l=20, r=20, b=20),
        font=dict(size=12),
    )
    return fig


def _prepare_ma_df(res: pd.DataFrame, metric: str, mean_cols: list) -> pd.DataFrame:
    plot_df = res.copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df["log2FC"] = pd.to_numeric(plot_df["log2FC"], errors="coerce")

    for c in mean_cols:
        plot_df[c] = pd.to_numeric(plot_df[c], errors="coerce")

    plot_df["Average_Intensity"] = plot_df[mean_cols].mean(axis=1)
    plot_df = plot_df.dropna(subset=["Average_Intensity", "log2FC", metric]).copy()
    return plot_df


def create_static_ma_plot_common(
    res: pd.DataFrame,
    gene_col_name: Optional[str],
    metric: str,
    alpha_thr: float,
    fc_thresh: float,
    mean_cols: list,
    top_n_annot: int = 0,
    control_label: str = "Control",
    test_label: str = "Test",
    show_title: bool = False,
):
    """Create a reusable static MA plot for Total Protein TMT and LFQ."""
    plot_df = _prepare_ma_df(res, metric, mean_cols)
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

    if show_title:
        ax.set_title("MA Plot", fontweight="bold")

    ax.set_xlabel(
        r"Average Intensity (log$_2$)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel(
        fr"log$_2$ fold-change ({test_label}/{control_label})",
        fontsize=12,
        fontweight="bold",
    )

    ax.grid(False)
    ax.legend(frameon=False, fontsize=8, loc="best")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    return fig, plot_df


def create_interactive_ma_plot_common(
    res: pd.DataFrame,
    sig_mask: pd.Series,
    gene_col_name: Optional[str],
    metric: str,
    fc_thresh: float,
    mean_cols: list,
    control_label: str,
    test_label: str,
    show_title: bool = False,
):
    """Create a reusable interactive Plotly MA plot."""
    plot_df = _prepare_ma_df(res, metric, mean_cols)
    plot_df["Sig"] = np.where(sig_mask.loc[plot_df.index] if hasattr(sig_mask, "loc") else sig_mask, "Significant", "NS")

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

    layout_kwargs = dict(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=40, l=20, r=20, b=20),
        font=dict(size=12),
    )
    if show_title:
        layout_kwargs["title"] = "MA Plot"

    fig.update_layout(**layout_kwargs)
    return fig
