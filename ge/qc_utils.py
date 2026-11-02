#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import streamlit as st
from download_utility import help_popover  # help text behind a small button

# Import the centralized helper
from ge.misc_utility import *


def draw_confidence_ellipse(x, y, ax, color, n_std=2.0, **kwargs):
    """
    Draw an ellipse representing the covariance of (x, y).
    n_std: number of standard deviations (2 = ~95% CI for Gaussian).
    """
    if len(x) < 2:
        return

    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)

    ellipse = Ellipse(
        (np.mean(x), np.mean(y)),
        width, height,
        angle=theta,
        facecolor=color, alpha=0.2,
        edgecolor="black", lw=1.5, **kwargs
    )
    ax.add_patch(ellipse)


def create_pca_plot(
    df: pd.DataFrame,
    group_samples: list[list],
    group_labels: list[str],
    title: str = "PCA",
    pc_x: str = "PC1",
    pc_y: str = "PC2",
    hide_outliers: bool = False,
    label_outliers: bool = True,
    draw_ellipses: bool = False,
    zoom_q: float = 1.0,
    palette: dict = None,
    quant_mode: str = "LFQ",
    lfq_zero_as_na: bool = True,
    lfq_pseudocount: float = 1.0,
    key_suffix: str = ""
):
    """
    Create PCA plot for sample groups.

    Added features:
      - robust outlier detection using IQR on the plotted PCs
      - optional outlier hiding
      - optional sample-name labels for outliers
      - optional quantile zoom
      - optional group confidence ellipses
      - selectable PCs such as PC1 vs PC2, PC1 vs PC3, etc.

    LFQ:
        uses log2-transformed intensity values
    TMT:
        uses values as provided
    """

    if df is None or df.empty:
        st.warning("Expression data is empty.")
        return None

    # ---------------------------------------------------------
    # Flatten sample list while preserving group/sample order
    # ---------------------------------------------------------
    all_samples = []
    sample_to_group = {}

    for label, samples in zip(group_labels, group_samples):
        for sample in samples:
            if sample not in sample_to_group:
                sample_to_group[sample] = label
                all_samples.append(sample)

    # Keep only sample columns present in the dataframe
    all_samples = [s for s in all_samples if s in df.columns]

    if len(all_samples) < 2:
        st.warning("Need at least 2 valid sample columns for PCA.")
        return None

    # ---------------------------------------------------------
    # Data preprocessing
    # ---------------------------------------------------------
    data = df[all_samples].copy()
    for col in all_samples:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    quant_mode = str(quant_mode).strip().upper()

    if quant_mode == "LFQ":
        if lfq_zero_as_na:
            data = data.replace(0, np.nan)
            data = np.log2(data)
        else:
            data = np.log2(data + float(lfq_pseudocount))

    # Remove features with all missing values
    data = data.dropna(how="all")
    if data.empty:
        st.warning("Not enough data to perform PCA after removing empty rows.")
        return None

    # Fill remaining missing values by feature/row median before transpose
    row_median = data.median(axis=1)
    data = data.apply(lambda col: col.fillna(row_median))

    # samples x features
    data_t = data.T

    # Remove features still missing or with no variance
    data_t = data_t.dropna(axis=1, how="all")
    non_constant_cols = data_t.columns[data_t.nunique(dropna=True) > 1]
    data_t = data_t[non_constant_cols]

    if data_t.empty or data_t.shape[0] < 2 or data_t.shape[1] < 1:
        st.warning("Not enough informative data to perform PCA.")
        return None

    # ---------------------------------------------------------
    # PCA
    # ---------------------------------------------------------
    x_scaled = StandardScaler(with_mean=True, with_std=True).fit_transform(data_t)

    n_comps = min(10, x_scaled.shape[0], x_scaled.shape[1])
    if n_comps < 2:
        st.warning("Not enough components to plot PCA. PCA requires at least 2 components.")
        return None

    pca_model = PCA(n_components=n_comps, random_state=0)
    scores = pca_model.fit_transform(x_scaled)

    pc_cols = [f"PC{i + 1}" for i in range(n_comps)]
    pc_vars = {
        f"PC{i + 1}": pca_model.explained_variance_ratio_[i] * 100
        for i in range(n_comps)
    }

    pc_df = pd.DataFrame(scores, columns=pc_cols)
    pc_df["Sample"] = list(data_t.index)
    pc_df["Group"] = pc_df["Sample"].map(sample_to_group)
    pc_df = pc_df.dropna(subset=["Group"]).copy()

    if pc_df.empty:
        st.warning("No valid grouped samples remained for PCA plotting.")
        return None

    # Validate requested PCs
    if pc_x not in pc_cols:
        pc_x = pc_cols[0]
    if pc_y not in pc_cols:
        pc_y = pc_cols[1] if len(pc_cols) > 1 else pc_cols[0]

    # ---------------------------------------------------------
    # Outlier detection using robust IQR on selected PCs
    # ---------------------------------------------------------
    def iqr_outlier(series, k=2.0):
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            return pd.Series(False, index=series.index)
        lo, hi = q1 - k * iqr, q3 + k * iqr
        return (series < lo) | (series > hi)

    pc_df["Outlier"] = (
        iqr_outlier(pc_df[pc_x], k=2.0) |
        iqr_outlier(pc_df[pc_y], k=2.0)
    )

    plot_df = pc_df[~pc_df["Outlier"]].copy() if hide_outliers else pc_df.copy()
    if plot_df.empty:
        st.warning("All samples were marked as outliers. Showing all samples instead.")
        plot_df = pc_df.copy()

    # ---------------------------------------------------------
    # Palette
    # ---------------------------------------------------------
    if palette is None:
        unique_groups = list(dict.fromkeys([g for g in group_labels if g in pc_df["Group"].unique()]))
        if not unique_groups:
            unique_groups = list(pc_df["Group"].dropna().unique())
        default_colors = sns.color_palette("Set2", n_colors=max(1, len(unique_groups))).as_hex()
        palette = {
            group: default_colors[idx % len(default_colors)]
            for idx, group in enumerate(unique_groups)
        }

    # Make sure every plotted group has a color
    missing_groups = [g for g in plot_df["Group"].dropna().unique() if g not in palette]
    if missing_groups:
        extra_colors = sns.color_palette("husl", n_colors=len(missing_groups)).as_hex()
        for g, c in zip(missing_groups, extra_colors):
            palette[g] = c

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 6))

    sns.scatterplot(
        data=plot_df,
        x=pc_x,
        y=pc_y,
        hue="Group",
        palette=palette,
        s=100,
        edgecolor="black",
        linewidth=1.0,
        ax=ax,
        zorder=3
    )

    # Confidence ellipses
    if draw_ellipses:
        for group, color in palette.items():
            group_data = plot_df[plot_df["Group"] == group]
            if len(group_data) >= 3:
                x_vals = group_data[pc_x].astype(float).values
                y_vals = group_data[pc_y].astype(float).values
                try:
                    draw_confidence_ellipse(x_vals, y_vals, ax, color, n_std=2.0, zorder=1)
                except Exception:
                    pass

        ax.relim()
        ax.autoscale_view()

    # Label outliers with sample names
    if label_outliers and not hide_outliers:
        outlier_df = pc_df[pc_df["Outlier"]].copy()
        for _, row in outlier_df.iterrows():
            ax.text(
                row[pc_x],
                row[pc_y],
                str(row["Sample"]),
                fontsize=9,
                ha="left",
                va="bottom",
                zorder=4
            )

    # Quantile zoom based on plotted points
    zoom_q = float(zoom_q)
    if len(plot_df) >= 3 and zoom_q < 1.0:
        zoom_q = min(max(zoom_q, 0.50), 1.0)
        qlo = 1.0 - zoom_q

        xlo, xhi = plot_df[pc_x].quantile(qlo), plot_df[pc_x].quantile(zoom_q)
        ylo, yhi = plot_df[pc_y].quantile(qlo), plot_df[pc_y].quantile(zoom_q)

        xpad = (xhi - xlo) * 0.08 if xhi > xlo else 1.0
        ypad = (yhi - ylo) * 0.08 if yhi > ylo else 1.0

        ax.set_xlim(xlo - xpad, xhi + xpad)
        ax.set_ylim(ylo - ypad, yhi + ypad)
    elif draw_ellipses:
        ax.relim()
        ax.autoscale_view()

    # Labels and styling
    title_suffix = "LFQ" if quant_mode == "LFQ" else "TMT"
    final_title = title if title else f"PCA ({title_suffix})"

    #ax.set_title(final_title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel(f"{pc_x} ({pc_vars.get(pc_x, 0):.1f}%)", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"{pc_y} ({pc_vars.get(pc_y, 0):.1f}%)", fontsize=12, fontweight="bold")
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=12)

    ax.legend(
        title="Group",
        title_fontsize=12,
        fontsize=12,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5)
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    st.pyplot(fig)

    if "add_plot_download_buttons" in globals():
        add_plot_download_buttons(
            fig,
            base_filename=f"pca_plot_{quant_mode}",
            key_suffix=key_suffix
        )

    # Outlier table
    if pc_df["Outlier"].any():
        outlier_table = (
            pc_df.loc[pc_df["Outlier"], ["Sample", "Group", pc_x, pc_y]]
            .sort_values(pc_x, ascending=False)
            .reset_index(drop=True)
        )
        st.caption("Outliers detected. Check these samples for loading, normalization, missingness, or batch effects.")
        st.dataframe(outlier_table, use_container_width=True, hide_index=True)

    st.session_state["report_pca_fig"] = fig
    st.session_state["report_pca_scores"] = pc_df

    return fig


def create_sample_clustering(
    df: pd.DataFrame,
    group_samples: list[list],
    group_labels: list[str],
    title: str = "Sample clustering",
    top_n: int = 50,
    norm_type: str = "Z-score",
    quant_mode: str = "TMT",
    palette: dict = None,
    lfq_zero_as_na: bool = True,
    lfq_pseudocount: float = 1.0,
    distance_metric: str = "correlation",
    linkage_method: str = "average",
    key_suffix: str = ""
):
    """
    Create sample-only hierarchical clustering dendrogram.

    This does NOT draw a feature heatmap. It only shows how samples cluster
    based on abundance profiles. Group labels are used only for coloring sample
    names and legend entries, not for clustering.
    """

    if df is None or df.empty:
        st.warning("Expression data is empty.")
        return None

    # ---------------------------------------------------------
    # Flatten samples while preserving user-defined group/sample order
    # ---------------------------------------------------------
    all_samples = []
    sample_to_group = {}

    for label, samples in zip(group_labels or [], group_samples or []):
        for sample in samples or []:
            if sample not in sample_to_group:
                sample_to_group[sample] = label
                all_samples.append(sample)

    # Keep only columns present in the dataframe
    all_samples = [s for s in all_samples if s in df.columns]

    if len(all_samples) < 2:
        st.warning("Need at least 2 valid sample columns for sample clustering.")
        return None

    # ---------------------------------------------------------
    # Prepare abundance matrix
    # ---------------------------------------------------------
    data = df[all_samples].copy()

    for col in all_samples:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    quant_mode = str(quant_mode).strip().upper()

    if quant_mode == "LFQ":
        if lfq_zero_as_na:
            data = data.replace(0, np.nan)
            data = np.log2(data)
        else:
            data = np.log2(data + float(lfq_pseudocount))

    # Remove features with no data
    data = data.dropna(how="all")

    if data.empty:
        st.warning("Not enough data for sample clustering after removing empty rows.")
        return None

    # Impute feature-level missing values using row median
    row_median = data.median(axis=1)
    data = data.apply(lambda col: col.fillna(row_median))

    # Transpose: samples x features
    data_t = data.T

    # Remove features still all missing
    data_t = data_t.dropna(axis=1, how="all")

    # Remove zero-variance features
    non_constant_cols = data_t.columns[data_t.nunique(dropna=True) > 1]
    data_t = data_t[non_constant_cols]

    if data_t.empty or data_t.shape[0] < 2 or data_t.shape[1] < 1:
        st.warning("Not enough informative data to cluster samples.")
        return None

    # ---------------------------------------------------------
    # Keep top most-variable features
    # ---------------------------------------------------------
    try:
        top_n = int(top_n)
    except Exception:
        top_n = 50

    top_n = max(2, min(top_n, data_t.shape[1]))

    variances = data_t.var(axis=0, skipna=True).sort_values(ascending=False)
    data_t = data_t.loc[:, variances.head(top_n).index]

    # ---------------------------------------------------------
    # Normalize if requested
    # ---------------------------------------------------------
    norm_type_clean = str(norm_type or "Z-score").strip().lower()

    if norm_type_clean.startswith("z"):
        x = StandardScaler(with_mean=True, with_std=True).fit_transform(data_t.values)
    else:
        x = data_t.values.astype(float)

    # ---------------------------------------------------------
    # Hierarchical clustering
    # ---------------------------------------------------------
    try:
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import pdist
    except Exception:
        st.error("Sample clustering requires scipy. Install it with: pip install scipy")
        return None

    try:
        distances = pdist(x, metric=distance_metric)

        if np.any(~np.isfinite(distances)):
            distances = pdist(np.nan_to_num(x, nan=0.0), metric="euclidean")
            distance_metric_used = "euclidean"
        else:
            distance_metric_used = distance_metric

        z = linkage(distances, method=linkage_method)

    except Exception:
        # Correlation distance can fail if a sample has no variance
        z = linkage(
            pdist(np.nan_to_num(x, nan=0.0), metric="euclidean"),
            method="average"
        )
        distance_metric_used = "euclidean"
        linkage_method = "average"

    # ---------------------------------------------------------
    # Palette
    # ---------------------------------------------------------
    if palette is None:
        unique_groups = list(dict.fromkeys([g for g in group_labels if g is not None]))

        if not unique_groups:
            unique_groups = ["Samples"]

        default_colors = sns.color_palette(
            "Set2",
            n_colors=max(1, len(unique_groups))
        ).as_hex()

        palette = {
            group: default_colors[idx % len(default_colors)]
            for idx, group in enumerate(unique_groups)
        }

    observed_groups = list(
        dict.fromkeys([sample_to_group.get(s, "Unassigned") for s in all_samples])
    )

    missing_groups = [g for g in observed_groups if g not in palette]

    if missing_groups:
        extra_colors = sns.color_palette("husl", n_colors=len(missing_groups)).as_hex()
        for group, color in zip(missing_groups, extra_colors):
            palette[group] = color

    # ---------------------------------------------------------
    # Plot dendrogram only
    # ---------------------------------------------------------
    fig_width = max(9.5, min(28, 0.48 * len(all_samples) + 5))
    fig_height = 6.8

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    display_labels = [
        str(s).replace("Abundance: ", "").strip()
        for s in all_samples
    ]

    dendro = dendrogram(
        z,
        labels=display_labels,
        leaf_rotation=75,
        leaf_font_size=max(7, min(11, 13 - len(all_samples) // 8)),
        ax=ax,
        color_threshold=None,
        above_threshold_color="#555555",
    )

    # ---------------------------------------------------------
    # Force ticks and labels to display
    # ---------------------------------------------------------
    ax.tick_params(
        axis="x",
        which="major",
        bottom=True,
        top=False,
        labelbottom=True,
        labeltop=False,
        length=5,
        width=1,
        pad=6,
    )

    ax.tick_params(
        axis="y",
        which="major",
        left=True,
        right=False,
        labelleft=True,
        length=5,
        width=1,
        pad=4,
    )

    ax.set_xlabel("Samples", fontsize=12, fontweight="bold", labelpad=18)
    ax.set_ylabel(
        f"Distance ({distance_metric_used})",
        fontsize=12,
        fontweight="bold"
    )

    #ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    # Keep only x-axis and y-axis; remove plot box
    ax.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    ax.tick_params(axis="x", bottom=True, top=False)
    ax.tick_params(axis="y", left=True, right=False)

    # ---------------------------------------------------------
    # Color leaf labels by sample group
    # ---------------------------------------------------------
    label_to_original = {
        str(s).replace("Abundance: ", "").strip(): s
        for s in all_samples
    }

    for tick in ax.get_xticklabels():
        shown_label = tick.get_text()
        original_sample = label_to_original.get(shown_label, shown_label)
        group = sample_to_group.get(original_sample, "Unassigned")

        tick.set_color(palette.get(group, "#333333"))
        tick.set_fontweight("bold")
        tick.set_rotation(75)
        tick.set_ha("right")
        tick.set_va("top")

    # ---------------------------------------------------------
    # Group legend
    # ---------------------------------------------------------
    import matplotlib.patches as mpatches

    handles = []
    for group in observed_groups:
        handles.append(
            mpatches.Patch(
                color=palette.get(group, "#333333"),
                label=str(group)
            )
        )

    if handles:
        ax.legend(
            handles=handles,
            title="Group",
            title_fontsize=11,
            fontsize=10,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
        )

    # Do not rely only on tight_layout; rotated dendrogram labels get clipped.
    fig.subplots_adjust(
        left=0.08,
        right=0.82,
        bottom=0.38,
        top=0.90,
    )

    st.pyplot(fig)

    # ---------------------------------------------------------
    # Download buttons
    # ---------------------------------------------------------
    if "add_plot_download_buttons" in globals():
        add_plot_download_buttons(
            fig,
            base_filename="sample_clustering_dendrogram",
            key_suffix=key_suffix or "sample_clustering"
        )

    # ---------------------------------------------------------
    # Clustered sample order table
    # ---------------------------------------------------------
    clustered_labels = dendro.get("ivl", [])
    clustered_samples = [
        label_to_original.get(label, label)
        for label in clustered_labels
    ]

    order_df = pd.DataFrame({
        "Cluster_Order": list(range(1, len(clustered_samples) + 1)),
        "Sample": clustered_samples,
        "Display_Label": clustered_labels,
        "Group": [
            sample_to_group.get(sample, "Unassigned")
            for sample in clustered_samples
        ],
    })

    with st.expander("Clustered sample order", expanded=False):
        st.dataframe(order_df, use_container_width=True, hide_index=True)

    st.session_state["report_sample_clustering_fig"] = fig
    st.session_state["report_sample_clustering_order"] = order_df

    return fig

def create_sample_corr_heatmap(df: pd.DataFrame, sample_cols: list[str]):
    """
    Create a sample-to-sample correlation heatmap.
    """
    clean_cols = [col.replace("Abundance: ", "").strip() for col in sample_cols]
    rename_map = dict(zip(sample_cols, clean_cols))
    df = df.rename(columns=rename_map)

    # Pearson correlation of log2 values. Zeros mean "not measured" and are left out
    # (pairwise), so a few very bright glycoforms or many zeros cannot dominate.
    _vals = df[clean_cols].apply(pd.to_numeric, errors="coerce")
    _vals = np.log2(_vals.where(_vals > 0))
    corr = _vals.corr(min_periods=3)
    
    # --- DYNAMIC FONT SIZING FIX ---
    num_samples = len(clean_cols)
    dynamic_font_size = max(5, 12 - int(num_samples / 1.5))
    show_annotations = num_samples <= 20
    # -------------------------------

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, 
        annot=show_annotations, 
        cmap='vlag', 
        fmt=".2f", 
        ax=ax,
        annot_kws={"size": dynamic_font_size, "weight": "bold"},
        cbar_kws={"shrink": 0.5}  
    )
    
    axis_label_size = max(6, 12 - int(num_samples / 3))
    
    # Force physical tick marks to show on both axes
    ax.tick_params(axis='both', labelsize=axis_label_size, left=True, bottom=True, color='black', width=1) 
    
    # Format colorbar labels
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10, left=False, right=True) # Ensure colorbar ticks are visible too
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Assuming you have this helper function defined elsewhere
    add_plot_download_buttons(fig, base_filename="sample_corr_heatmap", key_suffix="static_corr_matrix")
    st.session_state["report_corr_fig"] = fig


def create_abundance_density(df: pd.DataFrame, abundance_cols: list[str]):
    """
    Create a density plot of Log2 Abundances for all samples.
    """
    renamed_cols = {col: col.replace("Abundance: ", "") for col in abundance_cols}
    df = df.rename(columns=renamed_cols)
    sample_cols = list(renamed_cols.values())

    # Convert to numeric and log2 transform. Zeros mean "not measured" and are left
    # out; with log2(x + 1) they formed a false peak at 0 that squashed the real curves.
    df_clean = df[sample_cols].apply(pd.to_numeric, errors='coerce')
    df_log = np.log2(df_clean.where(df_clean > 0))

    # Melt for long-form plotting
    df_long = df_log.melt(var_name='Sample', value_name='Log2 Abundance')
    df_long = df_long.dropna()
    df_long['Sample'] = df_long['Sample'].astype(str)

    # Plot - increased width slightly to make room for external legend
    fig, ax = plt.subplots(figsize=(8.5, 6))
    unique_samples = df_long['Sample'].unique()
    
    # Dynamic palette assignment based on number of samples
    palette = sns.color_palette("husl", n_colors=len(unique_samples)) 
    
    sns.kdeplot(
        data=df_long,
        x='Log2 Abundance',
        hue='Sample',
        common_norm=False,
        linewidth=2.0, # Made lines slightly thicker for visibility
        alpha=0.7,
        palette=palette,
        ax=ax
    )
    
    ax.set_title("Log2 Abundance Density per Sample (measured values)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Log2 Abundance", fontsize=12, fontweight='bold')
    ax.set_ylabel("Density", fontsize=12, fontweight='bold')
    
    # Force physical tick marks to show on both axes
    ax.tick_params(axis='both', labelsize=12, left=True, bottom=True, color='black', width=1.2)
    
    # Ensure spines are clearly visible
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    
    # Legend formatting - Move the EXISTING Seaborn legend instead of creating a new empty one
    sns.move_legend(
        ax, "center left", 
        bbox_to_anchor=(1.02, 0.5), 
        title="Sample", 
        title_fontsize=12, 
        fontsize=10, 
        frameon=True
    )
    
    # Instruct tight_layout to explicitly reserve the right 15% of the figure for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    st.pyplot(fig)
    add_plot_download_buttons(fig, base_filename="abundance_density")
    st.session_state["report_abundance_density_fig"] = fig


def create_glycan_class_summary(df: pd.DataFrame, glycan_col: str = "GlycanClass") -> pd.DataFrame:
    """
    Compact glycan class composition: one table with an in-cell percentage bar.

    A small bar chart is still built for the PDF report and the download buttons,
    but it is not drawn on the page (it used to fill the whole width).
    """
    class_counts = df[glycan_col].value_counts(dropna=False)
    class_percent = class_counts / class_counts.sum() * 100

    summary_df = pd.DataFrame({
        "Glycan Class": class_counts.index.astype(str),
        "Count": class_counts.values,
        "Percent (%)": class_percent.values.round(2)
    })

    table_col, _ = st.columns([0.62, 0.38])
    with table_col:
        st.dataframe(
            summary_df,
            hide_index=True,
            use_container_width=True,
            height=min(35 * (len(summary_df) + 1) + 3, 360),
            column_config={
                "Count": st.column_config.NumberColumn("Glycoforms", format="%d"),
                "Percent (%)": st.column_config.ProgressColumn(
                    "Share of glycoforms", format="%.1f%%", min_value=0, max_value=100,
                ),
            },
        )

    # Small figure for the report and for downloads only.
    sns.set_style("white")
    plot_df = summary_df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.2, 0.42 * len(plot_df) + 0.9))
    ax.barh(plot_df["Glycan Class"], plot_df["Count"], color="#005a9c", height=0.6)
    for y, (cnt, pct) in enumerate(zip(plot_df["Count"], plot_df["Percent (%)"])):
        ax.text(cnt, y, f"  {cnt} ({pct:.1f}%)", va="center", fontsize=8, color="#333333")
    ax.set_xlabel("Glycoforms", fontsize=9)
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=8)
    ax.set_xlim(0, plot_df["Count"].max() * 1.3 if len(plot_df) else 1)
    sns.despine(ax=ax)
    ax.grid(False)
    plt.tight_layout()
    st.session_state["report_glycan_class_fig"] = fig

    with table_col:
        with st.popover("Download as figure", icon=":material/download:"):
            add_plot_download_buttons(
                fig,
                base_filename="glycan_class_distribution",
                key_suffix="static_glycan_class"
            )

    return summary_df

# =============================================================================
# Glycan class breakdown of differential results
# =============================================================================
def _glycan_trait(glycan_str) -> str:
    """Sialylation / fucosylation from a composition string "H N A F" or "H N A G F"."""
    import re
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
    import re
    import plotly.express as px
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

    help_popover(
        "Enrichment: one-sided Fisher's exact test of whether a class holds more significant glycoforms than "
        "expected from its size, Benjamini-Hochberg adjusted (q). It tells you which classes change more often "
        "than others, not how much they change.", label="How to read this"
    )
    show_cols = ["Class", "Glycoforms"] + [c for c in table.columns if c.endswith("(n)") or c.endswith("% of class)") or c.endswith("enrichment q")]
    st.dataframe(
        table[show_cols].style.format({c: "{:.3g}" for c in show_cols if c.endswith("enrichment q")}),
        use_container_width=True, hide_index=True,
    )
    csv_bytes = table.to_csv(index=False).encode("utf-8")
    csv_name = f"glycan_{'class' if group_by == 'Glycan class' else 'trait'}_breakdown.csv"
    try:
        from download_utility import add_hosted_safe_file_download_button as _dl
        _dl(label="Download table (CSV)", data=csv_bytes, file_name=csv_name,
            key="download_glycan_breakdown_csv", use_container_width=False)
    except Exception:
        st.download_button("Download table (CSV)", csv_bytes, file_name=csv_name,
                           mime="text/csv", key="download_glycan_breakdown_csv_fallback")


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


# =============================================================================
# TMT channel loading check and optional normalization (GlycoExplorer TMT)
# =============================================================================
GE_NORM_NONE = "None (raw abundances)"
GE_NORM_MEDIAN = "Median normalization (glycopeptides)"
GE_NORM_TOTAL = "Total intensity (sample loading)"
GE_NORM_PROTEOME = "Factors from a total-proteome table (same TMT pool)"
GE_NORM_METHODS = [GE_NORM_NONE, GE_NORM_MEDIAN, GE_NORM_TOTAL, GE_NORM_PROTEOME]
GE_NORM_HELP = {
    GE_NORM_NONE: "Statistics use the abundances exactly as exported. Fine when the channel medians already line up.",
    GE_NORM_MEDIAN: "Scales each channel so its median glycoform abundance matches the others. Assumes most "
                    "glycoforms do not change; a real global shift (for example more sialylation everywhere) "
                    "would be removed too.",
    GE_NORM_TOTAL: "Classic TMT sample-loading normalization: scales each channel to the same total signal. "
                   "Same assumption as median normalization, and more influenced by the few brightest glycoforms.",
    GE_NORM_PROTEOME: "Uses loading factors measured on the proteins of the same TMT pool (for example the "
                      "unenriched or total-proteome run of the same labelled samples). Does not assume that "
                      "glycosylation is unchanged. Upload the protein table with the same TMT channels.",
}


def ge_channel_key(col) -> str:
    """TMT channel code (126, 127N, ... 135N) if present in a column name, else the cleaned name."""
    s = str(col)
    m = re.search(r"(?<![0-9])(12[6-9]|13[0-5])[NC]?(?![0-9A-Za-z])", s)
    if m:
        return m.group(0).upper()
    s = re.sub(r"(?i)^\s*abundances?\s*(\(grouped\))?\s*:?\s*", "", s)
    return s.strip().lower()


def _log2_positive(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    X = df[cols].apply(pd.to_numeric, errors="coerce").astype(float)
    return np.log2(X.where(X > 0))


def _rows_for_loading(L: pd.DataFrame, min_rows: int = 30) -> tuple:
    """Rows quantified in every non-empty channel (fair comparison); falls back to all rows."""
    nonempty = [c for c in L.columns if L[c].notna().sum() >= 10]
    complete = L[nonempty].dropna() if nonempty else L.iloc[0:0]
    if len(complete) >= min_rows:
        return L.loc[complete.index], len(complete), True
    return L, len(complete), False


def ge_channel_medians(df: pd.DataFrame, cols: list) -> pd.Series:
    """Median log2 abundance per channel, on glycoforms quantified in all channels."""
    L = _log2_positive(df, cols)
    base, _, _ = _rows_for_loading(L)
    return base.median(axis=0)


def compute_ge_norm_factors(raw_df: pd.DataFrame, cols: list, method: str,
                            proteome_df: "pd.DataFrame | None" = None) -> tuple:
    """Linear multiplier per channel (1.0 = unchanged) and a short description.

    Zeros stay zero (they mean 'not quantified'), so only measured values are scaled.
    """
    factors = pd.Series(1.0, index=list(cols), dtype=float)
    info = {"rows_used": 0, "complete_rows": False, "unmatched": [], "message": ""}
    if method == GE_NORM_NONE or not cols:
        info["message"] = "no normalization"
        return factors, info

    if method in (GE_NORM_MEDIAN, GE_NORM_TOTAL):
        L = _log2_positive(raw_df, cols)
        base, n_complete, used_complete = _rows_for_loading(L)
        info["rows_used"] = int(len(base))
        info["complete_rows"] = bool(used_complete)
        if method == GE_NORM_MEDIAN:
            med = base.median(axis=0)
            target = float(np.nanmean(med.to_numpy()))
            f = np.power(2.0, target - med)
        else:
            lin = np.power(2.0, base)
            tot = lin.sum(axis=0, min_count=1)
            target = float(np.nanmean(tot.to_numpy()))
            f = target / tot
        factors = f.replace([np.inf, -np.inf], np.nan).fillna(1.0).astype(float)
        info["message"] = (f"{'median' if method == GE_NORM_MEDIAN else 'total intensity'} of "
                          f"{info['rows_used']:,} glycoforms"
                          + (" quantified in every channel" if used_complete else " (all quantified values)"))
        return factors.reindex(cols).fillna(1.0), info

    if method == GE_NORM_PROTEOME:
        if proteome_df is None or proteome_df.empty:
            raise ValueError("Upload the total-proteome table first.")
        prot_num = [c for c in proteome_df.columns
                    if pd.to_numeric(proteome_df[c], errors="coerce").notna().sum() > 0
                    and not re.search(r"(?i)normalized|scaled|ratio|grouped cv|count|p-value|q-value|score|coverage|length|mw|\[kda\]", str(c))]
        key_to_prot = {}
        for c in prot_num:
            k = ge_channel_key(c)
            if re.fullmatch(r"(12[6-9]|13[0-5])[NC]?", k) or re.search(r"(?i)abundance", str(c)):
                key_to_prot.setdefault(k, c)
        mapping = {}
        # Empty glyco channels (nothing quantified) keep factor 1.0 and need no protein partner.
        live = [c for c in cols if (pd.to_numeric(raw_df[c], errors="coerce") > 0).any()]
        for c in live:
            k = ge_channel_key(c)
            if k in key_to_prot:
                mapping[c] = key_to_prot[k]
            else:
                info["unmatched"].append(str(c))
        if info["unmatched"]:
            raise ValueError(
                "These glyco channels have no matching channel in the protein table: "
                + ", ".join(info["unmatched"][:12])
                + ". Both tables must use the same TMT channel names (126, 127N, ...)."
            )
        P = _log2_positive(proteome_df, list(dict.fromkeys(mapping.values())))
        empty_prot = [c for c in P.columns if P[c].notna().sum() == 0]
        if empty_prot:
            raise ValueError("These channels have no values in the protein table: "
                             + ", ".join(map(str, empty_prot[:12])) + ".")
        P = P.dropna()
        if len(P) < 50:
            raise ValueError(f"Only {len(P)} proteins are quantified in every channel; at least 50 are needed.")
        ratio = P.sub(P.mean(axis=1), axis=0)
        med_ratio = ratio.median(axis=0)
        info["rows_used"] = int(len(P))
        info["complete_rows"] = True
        for c, pc in mapping.items():
            factors[c] = float(2.0 ** (-med_ratio[pc]))
        info["message"] = f"median protein ratio of {len(P):,} proteins quantified in every channel"
        return factors, info

    raise ValueError(f"Unknown normalization method: {method}")


def apply_ge_normalization(raw_df: pd.DataFrame, factors: pd.Series) -> pd.DataFrame:
    out = raw_df.copy()
    for c, f in factors.items():
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce") * float(f)
    return out


def ge_loading_summary(raw_df: pd.DataFrame, norm_df: pd.DataFrame, cols: list, fmt=None,
                       group_of: "dict | None" = None, factors: "pd.Series | None" = None,
                       unit_label: str = "Channel") -> pd.DataFrame:
    fmt = fmt or (lambda x: str(x))
    group_of = group_of or {}
    before = ge_channel_medians(raw_df, cols)
    after = ge_channel_medians(norm_df, cols)
    ref = float(np.nanmean(before.to_numpy()))
    rows = []
    for c in cols:
        rows.append({
            unit_label: fmt(c),
            "Group": group_of.get(c, ""),
            "Median log2 before": round(float(before.get(c, np.nan)), 3),
            "Offset before (log2)": round(float(before.get(c, np.nan) - ref), 3),
            "Factor applied": round(float(factors.get(c, 1.0)) if factors is not None else 1.0, 3),
            "Median log2 after": round(float(after.get(c, np.nan)), 3),
        })
    return pd.DataFrame(rows)


def ge_group_offsets(df: pd.DataFrame, group_samples: list, group_labels: list, ref_idx: int) -> pd.DataFrame:
    """Mean channel median per group, relative to the reference group (log2)."""
    med_all = ge_channel_medians(df, [c for g in group_samples for c in (g or [])])
    means = {}
    for lbl, cols in zip(group_labels, group_samples):
        cols = [c for c in (cols or []) if c in med_all.index]
        if cols:
            means[lbl] = float(med_all[cols].mean())
    if not means or ref_idx >= len(group_labels) or group_labels[ref_idx] not in means:
        return pd.DataFrame()
    ref = means[group_labels[ref_idx]]
    return pd.DataFrame([{"Group": k, "Offset vs reference (log2)": round(v - ref, 3),
                          "Same as % change": f"{(2 ** (v - ref) - 1) * 100:+.0f}%"} for k, v in means.items()])


def ge_norm_boxplot_figure(raw_df: pd.DataFrame, norm_df: "pd.DataFrame | None", cols: list, fmt=None,
                           group_of: "dict | None" = None, palette: "dict | None" = None, after_title: str = "",
                           panels: "list | None" = None, ylabel: str = "log2 abundance"):
    """Per-channel log2 abundance boxplots: before, and after normalization when one is applied.

    `panels` (list of (title, DataFrame)) replaces the default before/after pair, e.g. to add
    an "after batch correction" panel in the label-free app.
    """
    fmt = fmt or (lambda x: str(x))
    group_of = group_of or {}
    groups = list(dict.fromkeys(group_of.get(c, "") for c in cols))
    default_cols = ["#8dd3c7", "#fdb462", "#bebada", "#fb8072", "#80b1d3", "#b3de69"]
    palette = palette or {g: default_cols[i % len(default_cols)] for i, g in enumerate(groups)}
    if not panels:
        panels = [("Before normalization (raw abundances)", raw_df)]
        if norm_df is not None:
            panels.append((after_title or "After normalization", norm_df))
    width = max(6.0, 0.42 * len(cols) + 1.8)
    fig, axes = plt.subplots(len(panels), 1, figsize=(width, 2.6 * len(panels) + 0.4), sharex=True, squeeze=False)
    for ax, (title, data) in zip(axes[:, 0], panels):
        # Same rows as the medians and the normalization factors: glycoforms measured in
        # every sample when there are enough, so the boxes line up exactly when the medians do.
        L, _n_complete, _used_complete = _rows_for_loading(_log2_positive(data, cols))
        vals = [L[c].dropna().to_numpy() for c in cols]
        bp = ax.boxplot(vals, showfliers=False, patch_artist=True, widths=0.6)
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(palette.get(group_of.get(c, ""), "#d9d9d9"))
            patch.set_edgecolor("#555555")
        for med in bp["medians"]:
            med.set_color("#222222")
        overall = float(np.nanmedian(np.concatenate([v for v in vals if len(v)]))) if any(len(v) for v in vals) else np.nan
        if np.isfinite(overall):
            ax.axhline(overall, color="#c0392b", linestyle="--", linewidth=0.8)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        sns.despine(ax=ax)
        ax.grid(False)
    axes[-1, 0].set_xticks(range(1, len(cols) + 1))
    axes[-1, 0].set_xticklabels([fmt(c) for c in cols], rotation=45, ha="right", fontsize=7)
    if groups and any(groups):
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=palette[g], edgecolor="#555555", label=g) for g in groups if g]
        axes[0, 0].legend(handles=handles, fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    return fig


# =============================================================================
# Missing values (label-free QC)
# =============================================================================
def missing_value_table(df: pd.DataFrame, cols: list, fmt=None, group_of: "dict | None" = None) -> pd.DataFrame:
    """Per-sample count and percent of glycoforms not measured (0 or empty)."""
    fmt = fmt or (lambda x: str(x))
    group_of = group_of or {}
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    miss = ~(X > 0)
    rows = []
    for c in cols:
        rows.append({
            "Sample": fmt(c),
            "Group": group_of.get(c, ""),
            "Measured": int((~miss[c]).sum()),
            "Missing": int(miss[c].sum()),
            "Missing (%)": round(100.0 * float(miss[c].mean()), 1) if len(X) else np.nan,
        })
    return pd.DataFrame(rows)


def missing_values_figure(df: pd.DataFrame, cols: list, fmt=None, group_of: "dict | None" = None,
                          palette: "dict | None" = None):
    """Left: % of glycoforms missing in each sample. Right: % missing by intensity quartile.

    In label-free data, low-intensity glycoforms are missing far more often. A rising curve on the
    right means values are mostly missing because they were below detection, which is what a
    minimum-value replacement assumes.
    """
    fmt = fmt or (lambda x: str(x))
    group_of = group_of or {}
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    miss = ~(X > 0)
    pct = 100.0 * miss.mean(axis=0)
    groups = list(dict.fromkeys(group_of.get(c, "") for c in cols))
    default_cols = ["#8dd3c7", "#fdb462", "#bebada", "#fb8072", "#80b1d3", "#b3de69"]
    palette = palette or {g: default_cols[i % len(default_cols)] for i, g in enumerate(groups)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(6.4, 0.26 * len(cols) + 3.8), 3.4),
                                   gridspec_kw={"width_ratios": [max(2.0, 0.26 * len(cols)), 1.9]})
    ax1.bar(range(len(cols)), pct.values,
            color=[palette.get(group_of.get(c, ""), "#d9d9d9") for c in cols], edgecolor="#555555", linewidth=0.6)
    ax1.set_xticks(range(len(cols)))
    ax1.set_xticklabels([fmt(c) for c in cols], rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel("Glycoforms missing (%)", fontsize=8)
    ax1.margins(x=0.01)
    ax1.set_title(f"Missing values per sample  ·  overall {100.0 * float(miss.values.mean()):.1f}%",
                  fontsize=9, loc="left")
    ax1.tick_params(axis="y", labelsize=7)
    ax1.set_ylim(0, max(5.0, float(pct.max()) * 1.35 if len(pct) else 5.0))
    if groups and any(groups):
        from matplotlib.patches import Patch
        ax1.legend(handles=[Patch(facecolor=palette[g], edgecolor="#555555", label=g) for g in groups if g],
                   fontsize=7, frameon=False, loc="upper left", ncol=min(3, len([g for g in groups if g])))

    L = np.log2(X.where(X > 0))
    mean_int = L.mean(axis=1)
    ok = mean_int.notna()
    if ok.sum() >= 8:
        q = pd.qcut(mean_int[ok], 4, labels=["Q1\n(lowest)", "Q2", "Q3", "Q4\n(highest)"], duplicates="drop")
        frac = miss[ok].mean(axis=1).groupby(q, observed=False).mean() * 100.0
        ax2.plot(range(len(frac)), frac.values, marker="o", color="#2c3e50")
        ax2.set_xticks(range(len(frac)))
        ax2.set_xticklabels([str(i) for i in frac.index], fontsize=7)
        ax2.set_ylim(0, max(5.0, float(np.nanmax(frac.values)) * 1.2))
    else:
        ax2.text(0.5, 0.5, "Too few glycoforms", ha="center", va="center", transform=ax2.transAxes, fontsize=8)
    ax2.set_title("Missing vs intensity", fontsize=9, loc="left")
    ax2.set_xlabel("Quartile of mean log2 intensity", fontsize=8)
    ax2.set_ylabel("Missing (%)", fontsize=8)
    ax2.tick_params(axis="y", labelsize=7)
    for ax in (ax1, ax2):
        sns.despine(ax=ax)
        ax.grid(False)
    plt.tight_layout()
    return fig
