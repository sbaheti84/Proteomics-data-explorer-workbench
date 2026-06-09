#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import matplotlib.lines as mlines
import re

# Import the centralized helper
from ge.misc_utility import add_plot_download_buttons


def get_glycan_legend_handles():
    """
    Helper to generate custom legend handles matching CFG/SNFG glycan notation shapes/colors.
    """
    return [
        mlines.Line2D([], [], marker='o', linestyle='None', markerfacecolor='#00a651', markeredgecolor='black', markersize=10, label='Mannose'),
        mlines.Line2D([], [], marker='o', linestyle='None', markerfacecolor='#ffc20e', markeredgecolor='black', markersize=10, label='Galactose'),
        mlines.Line2D([], [], marker='s', linestyle='None', markerfacecolor='#004a80', markeredgecolor='black', markersize=10, label='N-acetylglucosamine'),
        mlines.Line2D([], [], marker='D', linestyle='None', markerfacecolor='#a11a4d', markeredgecolor='black', markersize=10, label='N-acetylneuraminic acid'),
        mlines.Line2D([], [], marker='>', linestyle='None', markerfacecolor='#e81c24', markeredgecolor='black', markersize=10, label='Fucose'),
        mlines.Line2D([], [], marker='D', linestyle='None', markerfacecolor='#00aeef', markeredgecolor='black', markersize=10, label='N-glycolylneuraminic acid')
    ]

def get_glycan_image_candidates(label_glycan):
    """
    Return possible glycan image labels to try.

    This fixes mouse-style 5-number glycans when the glycan image library
    stores 4-number filenames.

    Expected mouse order for 5 numbers:
        Hex HexNAc NeuAc Fuc NeuGc

    Examples:
        5 3 2 0 1 -> tries 5 3 2 1
        4 5 2 0 1 -> tries 4 5 2 1
        5 4 2 0 0 -> tries 5 4 2 0
        4 2 0 0 0 -> tries 4 2 0 0

    The original glycan label is still returned first, so existing exact image
    names continue to work. This only affects image lookup, not plot labels.
    """
    original = str(label_glycan).strip()
    parts = original.split()

    candidates = [original]

    if len(parts) == 5:
        # Drop the 4th value, keeping NeuGc as the 4th image component.
        # Example: 5 3 2 0 1 -> 5 3 2 1
        candidates.append(" ".join([parts[0], parts[1], parts[2], parts[4]]))

        # Also try the first four values.
        # Example: 5 4 2 0 0 -> 5 4 2 0
        candidates.append(" ".join(parts[:4]))

    # Also try underscore-separated versions in case files were saved that way.
    expanded_candidates = []
    for cand in candidates:
        expanded_candidates.append(cand)
        expanded_candidates.append(cand.replace(" ", "_"))

    # Remove duplicates while preserving order.
    clean_candidates = []
    for cand in expanded_candidates:
        if cand and cand not in clean_candidates:
            clean_candidates.append(cand)

    return clean_candidates


def find_glycan_image_path(glycan_image_dir, label_glycan):
    """
    Try multiple possible glycan image filenames and return the first match.

    Returns:
        (Path, image_label) if found, otherwise (None, None)
    """
    glycan_image_dir = Path(glycan_image_dir)

    for image_label in get_glycan_image_candidates(label_glycan):
        image_path = glycan_image_dir / f"{image_label}.png"
        if image_path.is_file():
            return image_path, image_label

    return None, None

def create_volcano_plot(
    df: pd.DataFrame,
    top_n: int,
    sig_threshold: float,
    log2fc_threshold: float,
    label_col: str,
    sig_col: str = "p-value",
    log2fc_col: str = "log2FC",
    control_label: str = "Control",
    test_label: str = "Test",
    glycan_image_dir: str = "glycan_images",
    top_mode: str = "all"
) -> None:
    """
    Polished volcano plot with fan-shaped annotations.

    Features:
    - Keeps label format consistent.
    - First ProSite is red.
    - Additional ProSites are blue and closer to the red Asn.
    - Glycan image is always placed above the red Asn region.
    - Annotation lines stop below the label block.
    - Fan structure is preserved while pushing labels outward.
    """

    glycan_orig = "Glycan"
    df = df.copy()

    # ------------------------------------------------------------------
    # Required columns
    # ------------------------------------------------------------------
    required_cols = [glycan_orig, log2fc_col, sig_col, label_col, "Genes", "ProSites"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        return

    # ------------------------------------------------------------------
    # Clean input
    # ------------------------------------------------------------------
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[log2fc_col, sig_col]).copy()
    df = df[df[sig_col] > 0].copy()

    if df.empty:
        st.warning("No valid rows available for volcano plot after filtering.")
        return

    # ------------------------------------------------------------------
    # Color points
    # ------------------------------------------------------------------
    df["color"] = "gray"
    df["-log10(sig_col)"] = -np.log10(df[sig_col])

    df.loc[
        (df[log2fc_col] > log2fc_threshold) &
        (df[sig_col] < sig_threshold),
        "color"
    ] = "red"

    df.loc[
        (df[log2fc_col] < -log2fc_threshold) &
        (df[sig_col] < sig_threshold),
        "color"
    ] = "blue"

    # ------------------------------------------------------------------
    # Select top hits
    # ------------------------------------------------------------------
    if top_mode == "up":
        top_hits = df[
            (df[sig_col] < sig_threshold) &
            (df[log2fc_col] > log2fc_threshold)
        ].nsmallest(top_n, sig_col).copy()

    elif top_mode == "down":
        top_hits = df[
            (df[sig_col] < sig_threshold) &
            (df[log2fc_col] < -log2fc_threshold)
        ].nsmallest(top_n, sig_col).copy()

    elif top_mode == "up-down":
        half_n, extra = top_n // 2, top_n % 2

        top_up = df[
            (df[sig_col] < sig_threshold) &
            (df[log2fc_col] > log2fc_threshold)
        ].nsmallest(half_n + extra, sig_col).copy()

        top_down = df[
            (df[sig_col] < sig_threshold) &
            (df[log2fc_col] < -log2fc_threshold)
        ].nsmallest(half_n, sig_col).copy()

        top_hits = pd.concat([top_up, top_down], ignore_index=True)

    else:
        top_hits = df[
            (df[sig_col] < sig_threshold) &
            (df[log2fc_col].abs() > log2fc_threshold)
        ].nsmallest(top_n, sig_col).copy()

    if top_hits.empty:
        st.warning("No significant hits passed the selected thresholds.")
        return

    top_hits["-log10(sig_col)"] = -np.log10(top_hits[sig_col])

    plot_df = df[[log2fc_col, sig_col, "color", "-log10(sig_col)"]].copy()

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def split_prosites(prosite_text):
        s = str(prosite_text).strip()
        if not s or s.lower() in {"nan", "none", "na", "n/a", "unknown"}:
            return [s]
        sites = [x.strip() for x in re.split(r"[;,|/]+", s) if x.strip()]
        return sites if sites else [s]

    def assign_anchors_optimal(hits_df, anchors_list):
        if hits_df.empty or not anchors_list:
            return hits_df, anchors_list

        pts = hits_df[[log2fc_col, "-log10(sig_col)"]].values
        anchors_arr = np.array(anchors_list)

        cost = cdist(pts, anchors_arr, metric="euclidean")
        row_idx, col_idx = linear_sum_assignment(cost)

        matched_hits = hits_df.iloc[row_idx].reset_index(drop=True)
        matched_anchors = [anchors_list[i] for i in col_idx]

        return matched_hits, matched_anchors

    def gently_spread_y(anchors, min_gap):
        """
        Slight vertical spreading while keeping overall fan structure.
        """
        if not anchors:
            return anchors

        order = np.argsort([a[1] for a in anchors])
        sorted_pts = [anchors[i] for i in order]

        adjusted = [sorted_pts[0]]
        for x, y in sorted_pts[1:]:
            prev_x, prev_y = adjusted[-1]
            if y - prev_y < min_gap:
                y = prev_y + min_gap
            adjusted.append((x, y))

        out = [None] * len(anchors)
        for idx, orig_idx in enumerate(order):
            out[orig_idx] = adjusted[idx]
        return out

    # ------------------------------------------------------------------
    # Fan layout math
    # ------------------------------------------------------------------
    cx = 0.0
    cy = -np.log10(sig_threshold)

    left_hits = top_hits[top_hits[log2fc_col] < 0].reset_index(drop=True)
    right_hits = top_hits[top_hits[log2fc_col] >= 0].reset_index(drop=True)

    x_min_val = float(plot_df[log2fc_col].min())
    x_max_val = float(plot_df[log2fc_col].max())
    y_max_val = float(plot_df["-log10(sig_col)"].max())

    # outward fan
    fan_push_x = 1.8 + (0.14 * top_n)
    fan_push_y = 1.6 + (0.18 * top_n)

    rx_left = abs(x_min_val) + fan_push_x + (0.05 * len(left_hits))
    rx_right = abs(x_max_val) + fan_push_x + (0.05 * len(right_hits))
    ry = (y_max_val - cy) + fan_push_y

    # Make the fan wider when top_n is larger
    fan_extra = min(0.22, 0.006 * top_n)

    left_start = np.pi - (0.18 + fan_extra)
    left_end   = np.pi / 2 + (0.16 + fan_extra)

    right_start = np.pi / 2 - (0.16 + fan_extra)
    right_end   = 0.18 + fan_extra

    left_angles = (
        np.linspace(left_start, left_end, len(left_hits))
        if not left_hits.empty else []
    )

    right_angles = (
        np.linspace(right_start, right_end, len(right_hits))
        if not right_hits.empty else []
    )

    left_anchors = [
        (cx + rx_left * np.cos(a), cy + ry * np.sin(a))
        for a in left_angles
    ]
    right_anchors = [
        (cx + rx_right * np.cos(a), cy + ry * np.sin(a))
        for a in right_angles
    ]

    left_hits, left_anchors = assign_anchors_optimal(left_hits, left_anchors)
    right_hits, right_anchors = assign_anchors_optimal(right_hits, right_anchors)

    min_gap = 0.09 + (0.010 * top_n)
    left_anchors = gently_spread_y(left_anchors, min_gap)
    right_anchors = gently_spread_y(right_anchors, min_gap)

    # ------------------------------------------------------------------
    # Figure setup
    # ------------------------------------------------------------------
    x_left_bound = cx - rx_left - 2.0
    x_right_bound = cx + rx_right + 2.0
    y_lower = 0

    all_anchor_y = [p[1] for p in (left_anchors + right_anchors)] if (left_anchors or right_anchors) else [cy + ry]
    y_upper = max(all_anchor_y) + 2.5

    fig_width = 14.5
    fig_height = 8.5 + (0.14 * top_n)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # ------------------------------------------------------------------
    # Scatter
    # ------------------------------------------------------------------
    ax.scatter(
        plot_df[log2fc_col],
        plot_df["-log10(sig_col)"],
        c=plot_df["color"],
        s=15,
        alpha=0.6,
        edgecolors="white",
        linewidth=0.4,
        zorder=2
    )

    ax.axhline(
        cy,
        color="black",
        linestyle="dashed",
        linewidth=1.2,
        zorder=1
    )

    # ------------------------------------------------------------------
    # Drawing settings
    # ------------------------------------------------------------------
    y_range = y_upper - y_lower
    text_spacing = y_range * 0.017

    if top_n <= 15:
        image_zoom = 0.175
    elif top_n <= 30:
        image_zoom = 0.155
    else:
        image_zoom = 0.135

    def draw_multisite_label(anchor_x, text_y, label_gene, label_site):
        """
        Draw:
            GENE-Asn267
        or
            GENE-Asn267-Asn295-Asn301

        First Asn = red
        Remaining Asn labels = blue
        """
        sites = split_prosites(label_site)

        # gene text ends at anchor_x
        ax.text(
            anchor_x,
            text_y,
            f"{label_gene}-",
            fontsize=10,
            fontweight="bold",
            color="black",
            ha="right",
            va="bottom",
            zorder=6
        )

        first_site = sites[0] if sites else str(label_site)

        ax.text(
            anchor_x,
            text_y,
            f"Asn$^{{{first_site}}}$",
            fontsize=10,
            fontweight="bold",
            color="red",
            ha="left",
            va="bottom",
            zorder=6
        )

        # tighter spacing for blue labels
        offset_pts = 26 + (3 * len(str(first_site)))

        for extra_site in sites[1:]:
            ax.annotate(
                f"-Asn$^{{{extra_site}}}$",
                xy=(anchor_x, text_y),
                xycoords="data",
                xytext=(offset_pts, 0),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
                color="blue",
                ha="left",
                va="bottom",
                zorder=6
            )
            offset_pts += 30 + (3 * len(str(extra_site)))

    def draw_glycan_fallback_text(anchor_x, text_y, label_glycan):
        ax.annotate(
            str(label_glycan),
            xy=(anchor_x, text_y),
            xycoords="data",
            xytext=(24, 16),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            color="black",
            ha="center",
            va="bottom",
            zorder=5
        )

    def draw_side(side_hits: pd.DataFrame, anchors, side="right"):
        if side_hits.empty:
            return

        for (_, row), (anchor_x, anchor_y) in zip(side_hits.iterrows(), anchors):
            x = float(row[log2fc_col])
            y = float(row["-log10(sig_col)"])

            label_gene = str(row["Genes"])
            label_site = str(row["ProSites"])
            label_glycan = str(row[glycan_orig])

            text_y = anchor_y + (text_spacing * 0.20)
            line_end_y = text_y - (text_spacing * 0.24)

            # line ends just below label
            ax.annotate(
                "",
                xy=(anchor_x, line_end_y),
                xytext=(x, y),
                arrowprops=dict(
                    arrowstyle="-|>",
                    lw=0.9,
                    color="gray",
                    shrinkA=2,
                    shrinkB=0,
                    mutation_scale=10
                ),
                zorder=3
            )

            draw_multisite_label(
                anchor_x=anchor_x,
                text_y=text_y,
                label_gene=label_gene,
                label_site=label_site
            )

            glycan_img_path, image_glycan_label = find_glycan_image_path(
                glycan_image_dir, label_glycan
            )

            # glycan stays above red Asn
            glycan_xybox = (24, 16)

            if glycan_img_path is not None:
                try:
                    img = Image.open(glycan_img_path)
                    imagebox = OffsetImage(img, zoom=image_zoom)
                    ab = AnnotationBbox(
                        imagebox,
                        (anchor_x, text_y),
                        xybox=glycan_xybox,
                        xycoords="data",
                        boxcoords="offset points",
                        frameon=False,
                        box_alignment=(0.5, 0),
                        clip_on=False,
                        zorder=4
                    )
                    ax.add_artist(ab)
                except Exception:
                    draw_glycan_fallback_text(anchor_x, text_y, label_glycan)
            else:
                draw_glycan_fallback_text(anchor_x, text_y, label_glycan)

    draw_side(left_hits, left_anchors, side="left")
    draw_side(right_hits, right_anchors, side="right")

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------
    ax.set_xlabel(
        fr"log$_2$ fold-change ({test_label}/{control_label})",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_ylabel(
        fr"$-$log$_{{10}}$ ({sig_col})",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_ylim(y_lower, y_upper)
    ax.set_xlim(x_left_bound, x_right_bound)

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=12, left=True, bottom=True)

    ax.text(
        0.05,
        -0.08,
        r"$\bf{Asn}^{ABC}$ = significant glycosylation site",
        transform=ax.transAxes,
        fontsize=10,
        color="red",
        va="top",
        ha="left"
    )

    ax.text(
        0.40,
        -0.08,
        r"$\bf{Asn}^{XYZ}$ = second/unoccupied site from J peptide",
        transform=ax.transAxes,
        fontsize=10,
        color="blue",
        va="top",
        ha="left"
    )

    ax.legend(
        handles=get_glycan_legend_handles(),
        loc="upper left",
        bbox_to_anchor=(1.02, 0.4),
        frameon=False,
        fontsize=10,
        labelspacing=0.6,
        handletextpad=0.4
    )

    plt.tight_layout()
    st.pyplot(fig)

    add_plot_download_buttons(
        fig,
        base_filename="volcano_plot",
        key_suffix="static_volcano_plot"
    )

    st.session_state["report_volcano_fig"] = fig

def create_interactive_volcano_plot(
    df: pd.DataFrame,
    sig_threshold: float,
    log2fc_threshold: float,
    sig_col: str = "p-value",
    log2fc_col: str = "log2FC",
    control_label: str = "Control",
    test_label: str = "Test",
) -> None:
    """
    Interactive volcano plot for exploration.

    This is designed for hover/zoom/search behavior.
    The static volcano should remain the default for publication/report figures.
    """

    try:
        import plotly.express as px
    except ImportError:
        st.error("Plotly is required for the interactive volcano plot. Please install plotly.")
        return

    required_cols = [log2fc_col, sig_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns for interactive volcano plot: {', '.join(missing_cols)}")
        return

    plot_df = df.copy()
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)
    plot_df = plot_df.dropna(subset=[log2fc_col, sig_col]).copy()
    plot_df = plot_df[plot_df[sig_col] > 0].copy()

    if plot_df.empty:
        st.warning("No valid rows available for interactive volcano plot.")
        return

    plot_df["minus_log10_sig"] = -np.log10(plot_df[sig_col])

    plot_df["Regulation_Display"] = "Not significant"
    plot_df.loc[
        (plot_df[log2fc_col] > log2fc_threshold) &
        (plot_df[sig_col] < sig_threshold),
        "Regulation_Display"
    ] = "Upregulated"

    plot_df.loc[
        (plot_df[log2fc_col] < -log2fc_threshold) &
        (plot_df[sig_col] < sig_threshold),
        "Regulation_Display"
    ] = "Downregulated"

    hover_cols = []
    for c in [
        "Genes",
        "Peptide",
        "ProSites",
        "Glycan",
        "GlycanClass",
        log2fc_col,
        sig_col,
        "q-value",
        "p-value",
        "Regulation",
    ]:
        if c in plot_df.columns and c not in hover_cols:
            hover_cols.append(c)

    fig = px.scatter(
        plot_df,
        x=log2fc_col,
        y="minus_log10_sig",
        color="Regulation_Display",
        hover_data=hover_cols,
        custom_data=hover_cols,
        labels={
            log2fc_col: f"log2 fold-change ({test_label}/{control_label})",
            "minus_log10_sig": f"-log10({sig_col})",
            "Regulation_Display": "Regulation",
        },
        #title="Interactive Volcano Plot",
    )

    fig.add_hline(
        y=-np.log10(sig_threshold),
        line_dash="dash",
        line_color="black",
        annotation_text=f"{sig_col} = {sig_threshold:g}",
        annotation_position="top left",
    )

    fig.add_vline(
        x=log2fc_threshold,
        line_dash="dash",
        line_color="gray",
    )

    fig.add_vline(
        x=-log2fc_threshold,
        line_dash="dash",
        line_color="gray",
    )

    fig.update_traces(
        marker=dict(size=7, opacity=0.75),
        selector=dict(mode="markers"),
    )

    fig.update_layout(
        height=720,
        template="plotly_white",
        legend_title_text="Regulation",
        margin=dict(l=40, r=40, t=70, b=40),
    )

    st.plotly_chart(fig, width="stretch")

    st.session_state["report_interactive_volcano_fig"] = fig