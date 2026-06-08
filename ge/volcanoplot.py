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
    Volcano plot with radial fanned-out annotations.
    Uses bipartite matching to reduce annotation overlap.
    If a glycan image is not available, the glycan label is shown as text.
    """

    glycan_orig = "Glycan"
    df = df.copy()

    # -----------------------------
    # Required columns & Input Clean
    # -----------------------------
    required_cols = [glycan_orig, log2fc_col, sig_col, label_col, "Genes", "ProSites"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        return

    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[log2fc_col, sig_col]
    ).copy()

    df = df[df[sig_col] > 0].copy()

    if df.empty:
        st.warning("No valid rows available for volcano plot after filtering.")
        return

    # -----------------------------
    # Plot values & Color
    # -----------------------------
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

    # -----------------------------
    # Top hit selection
    # -----------------------------
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

    top_hits["Glycopeptide"] = (
        top_hits["Genes"].astype(str) + "--" +
        top_hits["ProSites"].astype(str) + "--" +
        top_hits[glycan_orig].astype(str)
    )

    plot_df = df[[log2fc_col, sig_col, "color", "-log10(sig_col)"]].copy()

    # -----------------------------
    # Radial Fan-Out Layout Math
    # -----------------------------
    cx = 0.0
    cy = -np.log10(sig_threshold)

    left_hits = top_hits[top_hits[log2fc_col] < 0].reset_index(drop=True)
    right_hits = top_hits[top_hits[log2fc_col] >= 0].reset_index(drop=True)

    x_min_val = float(plot_df[log2fc_col].min())
    x_max_val = float(plot_df[log2fc_col].max())
    y_max_val = float(plot_df["-log10(sig_col)"].max())

    rx_left = abs(x_min_val) + 0.5 + (0.05 * len(left_hits))
    rx_right = abs(x_max_val) + 0.5 + (0.05 * len(right_hits))
    ry = (y_max_val - cy) + 0.5 + (0.05 * top_n)

    left_angles = (
        np.linspace(np.pi - 0.15, np.pi / 2 + 0.15, len(left_hits))
        if not left_hits.empty else []
    )

    right_angles = (
        np.linspace(np.pi / 2 - 0.15, 0.15, len(right_hits))
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

    left_hits, left_anchors = assign_anchors_optimal(left_hits, left_anchors)
    right_hits, right_anchors = assign_anchors_optimal(right_hits, right_anchors)

    # -----------------------------
    # Figure setup
    # -----------------------------
    x_left_bound = cx - rx_left - 1.8
    x_right_bound = cx + rx_right + 1.8
    y_lower = 0
    y_upper = (cy + ry) + 2.8

    fig_width = 14
    fig_height = 8 + (0.1 * top_n)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # -----------------------------
    # Scatter & Threshold Line
    # -----------------------------
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

    # -----------------------------
    # Drawing Helper
    # -----------------------------
    y_range = y_upper - y_lower
    text_spacing = y_range * 0.02
    image_zoom = 0.175

    def draw_glycan_label(anchor_x, img_y, label_glycan):
        """
        Fallback text when glycan image is missing or unreadable.
        """
        ax.text(
            anchor_x,
            img_y,
            str(label_glycan),
            fontsize=10,
            fontweight="bold",
            color="black",
            ha="center",
            va="bottom",
            zorder=5
        )

    def draw_side(side_hits: pd.DataFrame, anchors):
        if side_hits.empty: return

        # Notice we are using the raw 'anchor_y' again to preserve the perfect fan arch
        for (_, row), (anchor_x, anchor_y) in zip(side_hits.iterrows(), anchors):
            x = float(row[log2fc_col])
            y = float(row["-log10(sig_col)"])
            label_gene, label_site, label_glycan = row["Glycopeptide"].split("--", 2)

            # Draw connector arrow.
            # Important:
            # shrinkA controls the gap near the label/glycan side.
            # This prevents the line from cutting through labels when annotations come
            # from the left or right side.
            dx = abs(anchor_x - x)
            dy = abs(anchor_y - y)

            # Side labels need a larger gap because the arrow is more horizontal
            # and can pass through the gene/Asn label.
            if dx > dy:
                label_gap = 55
            else:
                label_gap = 25

            ax.annotate(
                "",
                xy=(x, y),                    # Arrow head at data point
                xytext=(anchor_x, anchor_y),  # Label/glycan side
                arrowprops=dict(
                    arrowstyle="-|>",
                    lw=0.9,
                    color="gray",
                    shrinkA=label_gap,        # gap near label/glycan
                    shrinkB=2,                # tiny gap near point
                    mutation_scale=10
                ),
                zorder=3
            )

            # 2. Add a slight gap above the arrow tail for the text
            text_y = anchor_y + (text_spacing * 0.4)

            # 3. Gene name (Black, right-aligned so it ends exactly at the anchor)
            ax.text(
                anchor_x, text_y, f"{label_gene}-", 
                fontsize=10, fontweight="bold", color="black", 
                ha="right", va="bottom", zorder=5
            )

            # 4. Asn site (Red, left-aligned so it starts exactly at the anchor)
            ax.text(
                anchor_x, text_y, f"Asn$^{{{label_site}}}$", 
                fontsize=10, color="red", fontweight="bold", 
                ha="left", va="bottom", zorder=5
            )

            # 5. Glycan Image positioned above the Asn text
            img_y = text_y + (text_spacing * 1.5)
            glycan_img_path, image_glycan_label = find_glycan_image_path(
                glycan_image_dir, label_glycan
            )

            if glycan_img_path is not None:
                try:
                    img = Image.open(glycan_img_path)
                    
                    # NOTE: Uncomment the crop line below if pasting into the userfile version!
                    # cropped_img = img.crop((10, 10, img.width - 10, img.height - 10))
                    # imagebox = OffsetImage(cropped_img, zoom=image_zoom)
                    
                    imagebox = OffsetImage(img, zoom=image_zoom)
                    
                    # box_alignment=(0.1, 0) places the image slightly right to sit over "Asn"
                    ab = AnnotationBbox(
                        imagebox, (anchor_x, img_y), frameon=False, 
                        box_alignment=(0.1, 0), clip_on=False, zorder=4
                    )
                    ax.add_artist(ab)
                except Exception as e:
                    ax.text(
                        anchor_x, img_y, str(label_glycan),
                        fontsize=10, fontweight="bold", color="black",
                        ha="left", va="bottom", zorder=5
                    )
            else:
                ax.text(
                    anchor_x, img_y, str(label_glycan),
                    fontsize=10, fontweight="bold", color="black",
                    ha="left", va="bottom", zorder=5
                )

    draw_side(left_hits, left_anchors)
    draw_side(right_hits, right_anchors)

    # -----------------------------
    # Styling
    # -----------------------------
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

    yticks = ax.get_yticks()
    ax.set_yticks([y for y in yticks if not np.isclose(y, 0)])

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(
        0.05,
        -0.08,
        r"$\bf{Asn}^{ABC}$ = glycosylated site",
        transform=ax.transAxes,
        fontsize=10,
        color="red",
        va="top",
        ha="left"
    )

    ax.text(
        0.35,
        -0.08,
        r"$\bf{Asn}^{XYZ}$ = unoccupied glycosylation site",
        transform=ax.transAxes,
        fontsize=10,
        color="blue",
        va="top",
        ha="left"
    )

    # --- Added Legend Section ---
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

