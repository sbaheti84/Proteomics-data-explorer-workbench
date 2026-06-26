#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import streamlit as st

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

    # Compute correlation
    corr = df[clean_cols].corr()
    
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

    # Convert to numeric and Log2 transform
    df_clean = df[sample_cols].apply(pd.to_numeric, errors='coerce')
    df_log = np.log2(df_clean + 1)

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
    
    ax.set_title("Log2 Abundance Density per Sample", fontsize=14, fontweight='bold', pad=15)
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
    Generate a summary table and bar chart for Glycan Classes.
    """
    class_counts = df[glycan_col].value_counts(dropna=False)
    class_percent = class_counts / class_counts.sum() * 100

    summary_df = pd.DataFrame({
        "Glycan Class": class_counts.index.astype(str),
        "Count": class_counts.values,
        "Percent (%)": class_percent.values.round(2)
    })

    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.barplot(
        data=summary_df,
        x="Glycan Class",
        y="Count",
        color="#005a9c",
        ax=ax
    )

    ax.grid(False)
    sns.despine(ax=ax)

    for container in ax.containers:
        ax.bar_label(
            container,
            padding=3,
            fontsize=10,
            color="#333333",
            fontweight="bold"
        )

    ax.set_xlabel("Glycan Class", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12, fontweight="bold")

    # X-axis ticks
    ax.tick_params(
        axis="x",
        which="major",
        bottom=True,
        top=False,
        labelbottom=True,
        length=5,
        width=1.2,
        labelsize=12,
        rotation=45
    )

    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")

    # Y-axis ticks
    ax.tick_params(
        axis="y",
        which="major",
        left=True,
        right=False,
        labelleft=True,
        length=5,
        width=1.2,
        labelsize=12
    )

    # Keep only x and y axis lines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    plt.tight_layout()
    st.pyplot(fig)

    st.session_state["report_glycan_class_fig"] = fig

    add_plot_download_buttons(
        fig,
        base_filename="glycan_class_distribution",
        key_suffix="static_glycan_class"
    )

    return summary_df

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
