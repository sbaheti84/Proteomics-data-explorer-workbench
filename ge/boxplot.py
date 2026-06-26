#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from io import BytesIO
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from download_utility import add_download_button, add_matplotlib_download_buttons, add_plotly_download_buttons
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from PIL import Image
from statannotations.Annotator import Annotator
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backends.backend_pdf import PdfPages

# Import the centralized helper
from ge.misc_utility import *
# -------------------------------------------------------------------------
# Shared helpers
# -------------------------------------------------------------------------

def _safe_gp_parts(glycopeptide: str):
    parts = str(glycopeptide).split("--")
    gene = parts[0] if len(parts) > 0 else ""
    site = parts[1] if len(parts) > 1 else ""
    glycan = parts[2] if len(parts) > 2 else ""
    return gene, site, glycan


def _prepare_plot_matrix(
    df: pd.DataFrame,
    cols: list[str],
    quant_mode: str = "TMT",
    lfq_zero_as_na: bool = True,
    lfq_pseudocount: float = 1.0,
):
    """
    TMT  -> row-scaled Relative Abundance (%)
    LFQ  -> Log2 LFQ Intensity
    """
    quant_mode = str(quant_mode).strip().upper()
    X = df[cols].apply(pd.to_numeric, errors="coerce").copy()

    if quant_mode == "LFQ":
        if lfq_zero_as_na:
            X = X.replace(0, np.nan)
            X = np.log2(X)
        else:
            X = np.log2(X + float(lfq_pseudocount))

        y_col = "Log2 LFQ Intensity"
        y_label = "Log2 LFQ Intensity"

        finite_vals = X.to_numpy(dtype=float)
        finite_vals = finite_vals[np.isfinite(finite_vals)]
        if finite_vals.size > 0:
            ymin = float(np.floor(finite_vals.min()) - 0.5)
            ymax = float(np.ceil(finite_vals.max()) + 0.5)
            y_limits = (ymin, ymax)
        else:
            y_limits = None
        y_ticks = None

    else:
        row_max = X.max(axis=1).replace(0, pd.NA)
        X = X.div(row_max, axis=0) * 100.0

        y_col = "Relative Abundance (%)"
        y_label = "Relative Abundance (%)"
        y_limits = (-5, 120)
        y_ticks = [0, 20, 40, 60, 80, 100]

    return X, y_col, y_label, y_limits, y_ticks


def _add_glycan_image(
    ax,
    glycan_image_dir: str,
    glycan: str,
    xy=(0.5, 1.38),
    zoom=0.18,
    crop=False,
    upscale=4
):
    image_path = Path(glycan_image_dir) / f"{glycan}.png"

    if image_path.is_file():
        try:
            image = Image.open(image_path).convert("RGBA")

            if crop:
                image = image.crop((10, 10, image.width - 10, image.height - 10))

            # Upscale before embedding to reduce blur in PDF zoom
            if upscale and upscale > 1:
                image = image.resize(
                    (image.width * upscale, image.height * upscale),
                    Image.Resampling.LANCZOS
                )

            glycan_img = OffsetImage(
                image,
                zoom=zoom / upscale,
                resample=True,
                interpolation="lanczos"
            )

            ab = AnnotationBbox(
                glycan_img,
                xy,
                frameon=False,
                box_alignment=(0.5, 0.5),
                xycoords="axes fraction",
            )
            ax.add_artist(ab)

        except Exception:
            pass


def _style_axis(
    ax,
    y_label: str,
    show_ylabel: bool,
    xticklabels: list[str],
    y_limits=None,
    y_ticks=None,
    rotation: int = 0,
):
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)

    ax.set_xlabel("")
    ax.set_xticklabels(xticklabels, fontsize=12, fontweight="bold", rotation=rotation)

    if show_ylabel:
        ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
    else:
        ax.set_ylabel("")

    ax.grid(False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_linewidth(1.0)
    
    # Ensured ticks are visible on axes
    ax.tick_params(axis="both", labelsize=12, left=True, bottom=True)


def _finalize_pdf_download(pdf_buf: BytesIO, label: str, file_name: str):
    pdf_buf.seek(0)
    pdf_bytes = pdf_buf.getvalue()
    add_download_button(
        data=pdf_bytes,
        file_name=file_name,
        label=label,
        key=f"download_pdf_{str(file_name).replace('.pdf', '')}",
        mime="application/pdf",
    )


# -------------------------------------------------------------------------
# 1. create_box_plots
# -------------------------------------------------------------------------

def create_box_plots(
    df: pd.DataFrame,
    pval_col: str,
    top_n: int,
    control_cols: list[str],
    test_cols: list[str],
    control_label: str = "Control",
    test_label: str = "Test",
    glycan_image_dir: str = "glycan_images",
    control_color: str = "#1f77b4",
    test_color: str = "#d62728",
    quant_mode: str = "TMT",
    lfq_zero_as_na: bool = True,
    lfq_pseudocount: float = 1.0,
    top_mode: str = "up-down",
    log2fc_col: str = "log2FC",
) -> None:
    
    df_valid = df.dropna(subset=[pval_col]).copy()
    
    if top_mode == "up" and log2fc_col in df_valid.columns:
        top_hits = df_valid[df_valid[log2fc_col] > 0].nsmallest(top_n, pval_col).copy()
    elif top_mode == "down" and log2fc_col in df_valid.columns:
        top_hits = df_valid[df_valid[log2fc_col] < 0].nsmallest(top_n, pval_col).copy()
    elif top_mode == "up-down" and log2fc_col in df_valid.columns:
        half_n, extra = top_n // 2, top_n % 2
        top_up = df_valid[df_valid[log2fc_col] > 0].nsmallest(half_n + extra, pval_col).copy()
        top_down = df_valid[df_valid[log2fc_col] < 0].nsmallest(half_n, pval_col).copy()
        top_hits = pd.concat([top_up, top_down], ignore_index=True)
    else:
        top_hits = df_valid.nsmallest(top_n, pval_col).copy()

    if top_hits.empty:
        st.warning("No significant hits available to generate box plots.")
        return

    top_hits["Glycopeptide"] = (
        top_hits["Genes"].astype(str) + "--" +
        top_hits["ProSites"].astype(str) + "--" +
        top_hits["Glycan"].astype(str)
    )

    abundance_cols = test_cols + control_cols
    plot_vals, y_col, y_label, y_limits, y_ticks = _prepare_plot_matrix(
        top_hits,
        abundance_cols,
        quant_mode=quant_mode,
        lfq_zero_as_na=lfq_zero_as_na,
        lfq_pseudocount=lfq_pseudocount,
    )
    plot_vals["Glycopeptide"] = top_hits["Glycopeptide"].values

    df_melt = pd.DataFrame()

    for column in control_cols:
        df_new = plot_vals[["Glycopeptide", column]].copy()
        df_new.rename(columns={column: y_col}, inplace=True)
        df_new["Group"] = control_label
        df_melt = pd.concat([df_melt, df_new], axis=0, ignore_index=True)

    for column in test_cols:
        df_new = plot_vals[["Glycopeptide", column]].copy()
        df_new.rename(columns={column: y_col}, inplace=True)
        df_new["Group"] = test_label
        df_melt = pd.concat([df_melt, df_new], axis=0, ignore_index=True)

    glycopeptide_order = list(df_melt["Glycopeptide"].dropna().unique())

    pdf_buf = BytesIO()
    pdf = PdfPages(pdf_buf)

    cols_per_row = 5
    plots_per_page = 10
    last_fig = None
    group_order = [control_label, test_label]

    for start in range(0, len(glycopeptide_order), plots_per_page):
        end = min(start + plots_per_page, len(glycopeptide_order))
        current_peptides = glycopeptide_order[start:end]
        n_rows = math.ceil(len(current_peptides) / cols_per_row)

        fig = plt.figure(figsize=(cols_per_row * 3.2, n_rows * 5.0))
        gs = gridspec.GridSpec(n_rows, cols_per_row, figure=fig)

        for idx, glycopeptide in enumerate(current_peptides):
            row = idx // cols_per_row
            col = idx % cols_per_row
            ax = fig.add_subplot(gs[row, col])

            subset = df_melt[df_melt["Glycopeptide"] == glycopeptide].dropna(subset=[y_col]).copy()
            label_gene, label_site, label_glycan = _safe_gp_parts(glycopeptide)

            custom_palette = {control_label: control_color, test_label: test_color}

            sns.boxplot(
                x="Group", y=y_col, data=subset,
                palette=custom_palette, ax=ax, width=0.5, fliersize=0,
                order=group_order
            )
            sns.stripplot(
                x="Group", y=y_col, data=subset,
                color="black", dodge=False, size=4, alpha=0.5, ax=ax, jitter=0.08,
                order=group_order
            )

            try:
                pairs = [(control_label, test_label)]
                annotator = Annotator(ax, pairs, data=subset, x="Group", y=y_col, order=group_order)
                annotator.configure(test="t-test_ind", text_format="star", loc="inside", verbose=0)
                annotator.apply_and_annotate()
            except Exception:
                pass

            ax.set_xticks([0, 1])
            _style_axis(
                ax,
                y_label=y_label,
                show_ylabel=(col == 0),
                xticklabels=[control_label, test_label],
                y_limits=y_limits,
                y_ticks=y_ticks,
                rotation=0,
            )

            text_y = 1.06
            ax.text(
                0.5, text_y, f"{label_gene}-",
                transform=ax.transAxes, ha="right",
                fontsize=12, fontweight="bold", color="black"
            )

            ax.text(
                0.5, text_y, fr"Asn$^{{{label_site}}}$",
                transform=ax.transAxes, ha="left",
                fontsize=12, fontweight="bold", color="red"
            )

            # Glycan composition text, e.g. 4 3 2 1
            ax.text(
                0.5, 1.18, str(label_glycan),
                transform=ax.transAxes, ha="center",
                fontsize=10, fontweight="bold", color="black"
            )

            _add_glycan_image(
                ax, glycan_image_dir, label_glycan,
                xy=(0.5, 1.50), zoom=0.15, crop=False
            )

        for k in range(len(current_peptides), n_rows * cols_per_row):
            fig.add_subplot(gs[k // cols_per_row, k % cols_per_row]).axis("off")

        fig.tight_layout()
        fig.subplots_adjust(top=0.80, bottom=0.15, hspace=1.3)

        legend_text = "Significance Level:  ns (p > 0.05)   * (p ≤ 0.05)   ** (p ≤ 0.01)   *** (p ≤ 0.001)   **** (p ≤ 0.0001)"
        fig.text(0.5, 0.02, legend_text, ha="center", va="bottom", fontsize=12, fontweight="bold", color="black")

        st.pyplot(fig)
        pdf.savefig(fig, bbox_inches="tight", dpi=600)
        last_fig = fig
        plt.close(fig)

    pdf.close()

    out_name = "boxplot_grid_layout_lfq.pdf" if str(quant_mode).upper() == "LFQ" else "boxplot_grid_layout_tmt.pdf"
    _finalize_pdf_download(pdf_buf, "📥 Download All Box Plots (PDF)", out_name)

    if last_fig is not None:
        st.session_state["report_boxplot_fig"] = last_fig


# -------------------------------------------------------------------------
# 2. create_box_plots_anova
# -------------------------------------------------------------------------

def create_box_plots_anova(
    df: pd.DataFrame,
    pval_col: str,
    top_n: int,
    group_samples: list[list[str]],
    group_labels: list[str],
    glycan_image_dir: str = "glycan_images",
    quant_mode: str = "TMT",
    lfq_zero_as_na: bool = True,
    lfq_pseudocount: float = 1.0,
    group_colors: dict = None,
) -> None:

    all_sample_cols = [s for group in group_samples for s in group]

    top_hits = df.nsmallest(top_n, pval_col).copy()
    top_hits["Glycopeptide"] = (
        top_hits["Genes"].astype(str) + "--" +
        top_hits["ProSites"].astype(str) + "--" +
        top_hits["Glycan"].astype(str)
    )
    glycopeptide_order = top_hits["Glycopeptide"].unique().tolist()

    plot_vals, y_col, y_label, y_limits, y_ticks = _prepare_plot_matrix(
        top_hits,
        all_sample_cols,
        quant_mode=quant_mode,
        lfq_zero_as_na=lfq_zero_as_na,
        lfq_pseudocount=lfq_pseudocount,
    )
    plot_vals["Glycopeptide"] = top_hits["Glycopeptide"].values

    melt_parts = []
    for group, label in zip(group_samples, group_labels):
        df_tmp = plot_vals[group + ["Glycopeptide"]].melt(
            id_vars="Glycopeptide",
            value_name=y_col,
            var_name="Sample"
        )
        df_tmp["Group"] = label
        melt_parts.append(df_tmp)

    df_melt = pd.concat(melt_parts, ignore_index=True).dropna(subset=[y_col]).copy()

    pdf_buf = BytesIO()
    pdf = PdfPages(pdf_buf)

    cols_per_row = 5
    plots_per_page = 10
    
    if group_colors is not None:
        palette = group_colors
    else:
        palette = dict(zip(group_labels, sns.color_palette("Set2", len(group_labels))))
    
    pairs = list(combinations(group_labels, 2))
    last_fig = None
    group_order = group_labels

    for start in range(0, len(glycopeptide_order), plots_per_page):
        end = min(start + plots_per_page, len(glycopeptide_order))
        current_peptides = glycopeptide_order[start:end]
        n_rows = math.ceil(len(current_peptides) / cols_per_row)

        fig = plt.figure(figsize=(cols_per_row * 3.2, n_rows * 5.0))
        gs = gridspec.GridSpec(n_rows, cols_per_row, figure=fig)

        for idx, glycopeptide in enumerate(current_peptides):
            row, col = divmod(idx, cols_per_row)
            ax = fig.add_subplot(gs[row, col])

            subset = df_melt[df_melt["Glycopeptide"] == glycopeptide].copy()
            gene, site, glycan = _safe_gp_parts(glycopeptide)

            sns.boxplot(
                x="Group", y=y_col, data=subset, palette=palette, 
                ax=ax, width=0.5, fliersize=0, order=group_order
            )
            sns.stripplot(
                x="Group", y=y_col, data=subset, color="black", 
                jitter=0.08, size=4, alpha=0.5, ax=ax, order=group_order
            )

            ax.set_xticks(range(len(group_labels)))
            _style_axis(
                ax,
                y_label=y_label,
                show_ylabel=(col == 0),
                xticklabels=group_labels,
                y_limits=y_limits,
                y_ticks=y_ticks,
                rotation=30,
            )

            if len(group_labels) >= 2:
                try:
                    annotator = Annotator(ax, pairs, data=subset, x="Group", y=y_col, order=group_order)
                    annotator.configure(test="t-test_ind", text_format="star", loc="inside", verbose=0)
                    annotator.apply_and_annotate()
                except Exception:
                    pass

            text_y = 1.06
            ax.text(
                0.5, text_y, f"{gene}-",
                transform=ax.transAxes, ha="right",
                fontsize=12, fontweight="bold", color="black"
            )

            ax.text(
                0.5, text_y, fr"Asn$^{{{site}}}$",
                transform=ax.transAxes, ha="left",
                fontsize=12, fontweight="bold", color="red"
            )

            # Glycan composition text, e.g. 4 3 2 1
            ax.text(
                0.5, 1.18, str(glycan),
                transform=ax.transAxes, ha="center",
                fontsize=10, fontweight="bold", color="black"
            )
            
            try:
                pval_anova = top_hits.loc[top_hits["Glycopeptide"] == glycopeptide, pval_col].values[0]
                ax.text(0.5, 1.28, f"ANOVA p={pval_anova:.2g}", ha="center", transform=ax.transAxes, fontsize=10, fontweight="bold", color="black")
            except Exception:
                pass

            _add_glycan_image(
                ax, glycan_image_dir, glycan,
                xy=(0.5, 1.55), zoom=0.15, crop=True
            )

        for k in range(len(current_peptides), n_rows * cols_per_row):
            fig.add_subplot(gs[k // cols_per_row, k % cols_per_row]).axis("off")

        fig.tight_layout()
        fig.subplots_adjust(top=0.80, bottom=0.15, hspace=1.4)

        legend_text = "Significance Level:  ns (p > 0.05)   * (p ≤ 0.05)   ** (p ≤ 0.01)   *** (p ≤ 0.001)   **** (p ≤ 0.0001)"
        fig.text(0.5, 0.02, legend_text, ha="center", va="bottom", fontsize=12, fontweight="bold", color="black")

        st.pyplot(fig)
        pdf.savefig(fig, bbox_inches="tight")
        last_fig = fig
        plt.close(fig)

    pdf.close()

    out_name = "anova_boxplot_grid_layout_lfq.pdf" if str(quant_mode).upper() == "LFQ" else "anova_boxplot_grid_layout_tmt.pdf"
    _finalize_pdf_download(pdf_buf, "📥 Download All Box Plots (PDF)", out_name)

    if last_fig is not None:
        st.session_state["report_boxplot_fig"] = last_fig




# -------------------------------------------------------------------------
# 5. create_box_plots_top_n
# -------------------------------------------------------------------------

def create_box_plots_top_n(
    df: pd.DataFrame,
    pval_col: str,
    top_n: int,
    control_cols: list[str],
    test_cols: list[str],
    control_label: str = "Control",
    test_label: str = "Test",
    glycan_image_dir: str = "glycan_images",
    quant_mode: str = "TMT",
    lfq_zero_as_na: bool = True,
    lfq_pseudocount: float = 1.0,
) -> None:
    top_hits = df.nsmallest(top_n, pval_col)
    selected_genes = top_hits["Genes"].unique()
    df_top = df[df["Genes"].isin(selected_genes)].copy()
    df_top["Glycopeptide"] = (
        df_top["Genes"].astype(str) + "--" +
        df_top["ProSites"].astype(str) + "--" +
        df_top["Glycan"].astype(str)
    )

    abundance_cols = control_cols + test_cols
    plot_vals, y_col, y_label, y_limits, y_ticks = _prepare_plot_matrix(
        df_top,
        abundance_cols,
        quant_mode=quant_mode,
        lfq_zero_as_na=lfq_zero_as_na,
        lfq_pseudocount=lfq_pseudocount,
    )
    plot_vals["Genes"] = df_top["Genes"].values
    plot_vals["Glycopeptide"] = df_top["Glycopeptide"].values

    pdf_buffer = BytesIO()
    custom_palette = {control_label: "blue", test_label: "red"}
    group_order = [control_label, test_label]

    with st.status("Generating boxplots for selected genes...", expanded=True) as status:
        with PdfPages(pdf_buffer) as pdf:
            for gene in selected_genes:
                df_gene = plot_vals[plot_vals["Genes"] == gene]
                glycopeptides = df_gene["Glycopeptide"].unique()
                n = len(glycopeptides)
                ncols = min(5, n)
                nrows = (n + ncols - 1) // ncols

                fig, axs = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 7.5 * nrows))
                if nrows == 1 and ncols == 1:
                    axs = np.array([[axs]])
                elif nrows == 1:
                    axs = np.array([axs])
                elif ncols == 1:
                    axs = np.array([[ax] for ax in axs])

                for i, glyco in enumerate(glycopeptides):
                    df_subset = df_gene[df_gene["Glycopeptide"] == glyco]
                    df_melt = pd.melt(
                        df_subset,
                        id_vars=["Glycopeptide", "Genes"],
                        value_vars=control_cols + test_cols,
                        var_name="Sample",
                        value_name=y_col
                    ).dropna(subset=[y_col])
                    df_melt["Group"] = df_melt["Sample"].apply(lambda x: test_label if x in test_cols else control_label)

                    row = i // ncols
                    col = i % ncols
                    ax = axs[row][col]

                    sns.boxplot(
                        data=df_melt, x="Group", y=y_col, ax=ax, 
                        palette=custom_palette, fliersize=0, order=group_order
                    )
                    sns.stripplot(
                        data=df_melt, x="Group", y=y_col, color="black", 
                        size=4, jitter=0.08, ax=ax, order=group_order
                    )

                    try:
                        pairs = [(control_label, test_label)]
                        annotator = Annotator(ax, pairs, data=df_melt, x="Group", y=y_col, order=group_order)
                        annotator.configure(test="t-test_ind", text_format="star", loc="inside", verbose=0)
                        annotator.apply_and_annotate()
                    except Exception:
                        pass

                    ax.set_xticks([0, 1])
                    _style_axis(
                        ax,
                        y_label=y_label,
                        show_ylabel=(col == 0),
                        xticklabels=[control_label, test_label],
                        y_limits=y_limits,
                        y_ticks=y_ticks,
                        rotation=0,
                    )

                    gene_name, site, glycan = _safe_gp_parts(glyco)

                    text_y = 1.06
                    ax.text(
                        0.5, text_y, f"{gene_name}-",
                        transform=ax.transAxes, ha="right",
                        fontsize=12, fontweight="bold", color="black"
                    )

                    ax.text(
                        0.5, text_y, fr"Asn$^{{{site}}}$",
                        transform=ax.transAxes, ha="left",
                        fontsize=12, fontweight="bold", color="red"
                    )

                    # Glycan composition text, e.g. 4 3 2 1
                    ax.text(
                        0.5, 1.18, str(glycan),
                        transform=ax.transAxes, ha="center",
                        fontsize=10, fontweight="bold", color="black"
                    )

                    _add_glycan_image(
                        ax, glycan_image_dir, glycan,
                        xy=(0.5, 1.50), zoom=0.15, crop=True
                    )

                for j in range(i + 1, nrows * ncols):
                    row = j // ncols
                    col = j % ncols
                    axs[row][col].axis("off")

                fig.suptitle(f"Gene: {gene}", fontsize=18, fontweight="bold", y=0.98)
                fig.tight_layout(rect=[0, 0.05, 1, 0.95], pad=3.5, h_pad=8.0, w_pad=2.5)

                legend_text = "Significance Level:  ns (p > 0.05)   * (p ≤ 0.05)   ** (p ≤ 0.01)   *** (p ≤ 0.001)   **** (p ≤ 0.0001)"
                fig.text(0.5, 0.02, legend_text, ha="center", va="bottom", fontsize=12, fontweight="bold", color="black")

                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        status.update(label="✅ Boxplots generated successfully.", state="complete")

    out_name = "gene_glycopeptide_boxplots_lfq.pdf" if str(quant_mode).upper() == "LFQ" else "gene_glycopeptide_boxplots_tmt.pdf"
    _finalize_pdf_download(pdf_buffer, "📥 Download Gene-wise Glycopeptide Boxplots (PDF)", out_name)


# -------------------------------------------------------------------------
# 6. create_box_plots_top_n_anova
# -------------------------------------------------------------------------

def create_box_plots_top_n_anova(
    df: pd.DataFrame,
    pval_col: str,
    top_n: int,
    group1_cols: list[str],
    group2_cols: list[str],
    group3_cols: list[str],
    group1_label: str,
    group2_label: str,
    group3_label: str,
    glycan_image_dir: str = "glycan_images",
    quant_mode: str = "TMT",
    lfq_zero_as_na: bool = True,
    lfq_pseudocount: float = 1.0,
) -> None:

    top_hits = df.nsmallest(top_n, pval_col)
    selected_genes = top_hits["Genes"].unique()
    df_top = df[df["Genes"].isin(selected_genes)].copy()
    df_top["Glycopeptide"] = (
        df_top["Genes"].astype(str) + "--" +
        df_top["ProSites"].astype(str) + "--" +
        df_top["Glycan"].astype(str)
    )

    all_cols = group1_cols + group2_cols + group3_cols
    plot_vals, y_col, y_label, y_limits, y_ticks = _prepare_plot_matrix(
        df_top,
        all_cols,
        quant_mode=quant_mode,
        lfq_zero_as_na=lfq_zero_as_na,
        lfq_pseudocount=lfq_pseudocount,
    )
    plot_vals["Genes"] = df_top["Genes"].values
    plot_vals["Glycopeptide"] = df_top["Glycopeptide"].values

    pdf_buffer = BytesIO()
    group_map = {col: group1_label for col in group1_cols}
    group_map.update({col: group2_label for col in group2_cols})
    group_map.update({col: group3_label for col in group3_cols})

    group_order = [group1_label, group2_label, group3_label]
    palette = dict(zip(group_order, sns.color_palette("Set2", 3)))

    with st.status("Generating boxplots for selected genes...", expanded=True) as status:
        with PdfPages(pdf_buffer) as pdf:
            for gene in selected_genes:
                df_gene = plot_vals[plot_vals["Genes"] == gene]
                glycopeptides = df_gene["Glycopeptide"].unique()
                n = len(glycopeptides)
                ncols = min(5, n)
                nrows = (n + ncols - 1) // ncols

                fig, axs = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 7.5 * nrows))
                if nrows == 1 and ncols == 1:
                    axs = np.array([[axs]])
                elif nrows == 1:
                    axs = np.array([axs])
                elif ncols == 1:
                    axs = np.array([[ax] for ax in axs])

                for i, glyco in enumerate(glycopeptides):
                    df_subset = df_gene[df_gene["Glycopeptide"] == glyco]
                    df_melt = pd.melt(
                        df_subset,
                        id_vars=["Glycopeptide", "Genes"],
                        value_vars=all_cols,
                        var_name="Sample",
                        value_name=y_col
                    ).dropna(subset=[y_col])
                    df_melt["Group"] = df_melt["Sample"].map(group_map)

                    row = i // ncols
                    col = i % ncols
                    ax = axs[row][col]

                    sns.boxplot(
                        data=df_melt, x="Group", y=y_col, ax=ax, 
                        palette=palette, fliersize=0, order=group_order
                    )
                    sns.stripplot(
                        data=df_melt, x="Group", y=y_col, ax=ax, 
                        color="black", size=4, jitter=0.08, order=group_order
                    )

                    pairs = [(group1_label, group2_label), (group1_label, group3_label), (group2_label, group3_label)]
                    try:
                        annotator = Annotator(ax, pairs, data=df_melt, x="Group", y=y_col, order=group_order)
                        annotator.configure(test="t-test_ind", text_format="star", loc="inside", verbose=0)
                        annotator.apply_and_annotate()
                    except Exception:
                        pass

                    ax.set_xticks(range(3))
                    _style_axis(
                        ax,
                        y_label=y_label,
                        show_ylabel=(col == 0),
                        xticklabels=[group1_label, group2_label, group3_label],
                        y_limits=y_limits,
                        y_ticks=y_ticks,
                        rotation=30,
                    )

                    gene_name, site, glycan = _safe_gp_parts(glyco)

                    text_y = 1.06
                    ax.text(
                        0.5, text_y, f"{gene_name}-",
                        transform=ax.transAxes, ha="right",
                        fontsize=12, fontweight="bold", color="black"
                    )

                    ax.text(
                        0.5, text_y, fr"Asn$^{{{site}}}$",
                        transform=ax.transAxes, ha="left",
                        fontsize=12, fontweight="bold", color="red"
                    )

                    # Glycan composition text, e.g. 4 3 2 1
                    ax.text(
                        0.5, 1.18, str(glycan),
                        transform=ax.transAxes, ha="center",
                        fontsize=10, fontweight="bold", color="black"
                    )

                    _add_glycan_image(
                        ax, glycan_image_dir, glycan,
                        xy=(0.5, 1.50), zoom=0.15, crop=True
                    )

                for j in range(i + 1, nrows * ncols):
                    row = j // ncols
                    col = j % ncols
                    axs[row][col].axis("off")

                fig.suptitle(f"Gene: {gene}", fontsize=18, fontweight="bold", y=0.98)
                fig.tight_layout(rect=[0, 0.05, 1, 0.95], pad=3.5, h_pad=8.0, w_pad=2.5)

                legend_text = "Significance Level:  ns (p > 0.05)   * (p ≤ 0.05)   ** (p ≤ 0.01)   *** (p ≤ 0.001)   **** (p ≤ 0.0001)"
                fig.text(0.5, 0.02, legend_text, ha="center", va="bottom", fontsize=12, fontweight="bold", color="black")

                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        status.update(label="✅ Boxplots generated successfully.", state="complete")

    out_name = "gene_glycopeptide_boxplots_anova_lfq.pdf" if str(quant_mode).upper() == "LFQ" else "gene_glycopeptide_boxplots_anova_tmt.pdf"
    _finalize_pdf_download(pdf_buffer, "📥 Download Gene-wise Glycopeptide Boxplots (PDF)", out_name)