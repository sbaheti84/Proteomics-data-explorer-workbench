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
    pc_x: str = "PC1",
    pc_y: str = "PC2",
    draw_ellipses: bool = False,
    palette: dict = None,
    quant_mode: str = "LFQ",
    lfq_zero_as_na: bool = True,
    lfq_pseudocount: float = 1.0,
    key_suffix: str = ""
):
    """
    Create PCA plot for sample groups, with optional confidence ellipses,
    allowing users to select which Principal Components to plot.

    LFQ:
        uses log2-transformed intensity values
    TMT:
        uses values as provided
    """

    all_samples = [s for group in group_samples for s in group]
    if not all_samples:
        st.error("No sample columns were provided for PCA.")
        return

    data = df[all_samples].copy()
    for col in all_samples:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    quant_mode = str(quant_mode).strip().upper()

    # -----------------------------
    # Quant-mode aware preprocessing
    # -----------------------------
    if quant_mode == "LFQ":
        if lfq_zero_as_na:
            data = data.replace(0, np.nan)
            data = np.log2(data)
        else:
            data = np.log2(data + float(lfq_pseudocount))

    # Remove rows with all missing values
    data = data.dropna(how="all")

    # For PCA, transpose so samples are rows and features are columns
    data_t = data.T

    # Fill remaining missing values feature-wise after transpose
    # Each column in data_t is a feature/glycopeptide
    data_t = data_t.apply(lambda col: col.fillna(col.median()), axis=0)

    # Remove columns still all-NA or zero-variance
    data_t = data_t.dropna(axis=1, how="all")
    if data_t.empty:
        st.error("Not enough data to perform PCA after preprocessing.")
        return

    non_constant_cols = data_t.columns[data_t.nunique(dropna=True) > 1]
    data_t = data_t[non_constant_cols]

    if data_t.empty or data_t.shape[0] < 2 or data_t.shape[1] < 1:
        st.error("Not enough informative data to perform PCA.")
        return

    # Standardize data
    x_scaled = StandardScaler().fit_transform(data_t)

    # Calculate PCA
    n_comps = min(10, x_scaled.shape[0], x_scaled.shape[1])
    if n_comps < 1:
        st.error("Could not compute PCA because the processed matrix is too small.")
        return

    pca_model = PCA(n_components=n_comps)
    pca = pca_model.fit_transform(x_scaled)

    # Store variance and create column names
    pc_cols = [f"PC{i+1}" for i in range(n_comps)]
    pc_vars = {
        f"PC{i+1}": pca_model.explained_variance_ratio_[i] * 100
        for i in range(n_comps)
    }

    # Build result DataFrame
    pc_df = pd.DataFrame(pca, columns=pc_cols, index=data_t.index)

    group_column = []
    sample_order = []
    for label, samples in zip(group_labels, group_samples):
        group_column.extend([label] * len(samples))
        sample_order.extend(samples)

    # Align sample labels to actual PCA rows
    sample_to_group = {}
    for label, samples in zip(group_labels, group_samples):
        for s in samples:
            sample_to_group[s] = label

    pc_df["Sample"] = pc_df.index
    pc_df["Group"] = pc_df["Sample"].map(sample_to_group)

    if pc_df["Group"].isna().any():
        pc_df = pc_df.dropna(subset=["Group"]).copy()

    if pc_df.empty:
        st.error("No valid grouped samples remained for PCA plotting.")
        return

    # Apply Custom Palette
    if palette is None:
        default_palette = ["#b3e2cd", "#fdcdac", "#cbd5e8", "#decbe4", "#ccebc5", "#fed9a6"]
        unique_groups = list(dict.fromkeys(group_labels))
        palette = {
            label: default_palette[i % len(default_palette)]
            for i, label in enumerate(unique_groups)
        }

    # Validate requested PCs
    if pc_x not in pc_cols:
        pc_x = pc_cols[0]
    if pc_y not in pc_cols:
        pc_y = pc_cols[1] if len(pc_cols) > 1 else pc_cols[0]

    # Plot
    fig, ax = plt.subplots(figsize=(8.5, 6))
    sns.scatterplot(
        data=pc_df,
        x=pc_x,
        y=pc_y,
        hue="Group",
        palette=palette,
        s=100,
        edgecolor="black",
        linewidth=1.2,
        ax=ax
    )

    # Draw ellipses
    if draw_ellipses:
        for group, color in palette.items():
            group_data = pc_df[pc_df["Group"] == group]
            if len(group_data) >= 2:
                draw_confidence_ellipse(group_data[pc_x], group_data[pc_y], ax, color)

        ax.relim()
        ax.autoscale_view()

    title_suffix = "LFQ" if quant_mode == "LFQ" else "TMT"
    ax.set_title(f"PCA ({title_suffix})", fontsize=14, fontweight="bold", pad=15)
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
        add_plot_download_buttons(fig, base_filename=f"pca_plot_{quant_mode}", key_suffix=key_suffix)

    st.session_state["report_pca_fig"] = fig

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
    ax.tick_params(axis='both', labelsize=axis_label_size) 
    ax.set_title("Sample-Sample Correlation Heatmap", fontsize=14, fontweight="bold", pad=15)
    
    # Format colorbar labels
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    
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
    ax.tick_params(axis='both', labelsize=12)
    
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

    # Safely construct DataFrame regardless of Pandas version
    summary_df = pd.DataFrame({
        "Glycan Class": class_counts.index.astype(str),
        "Count": class_counts.values,
        "Percent (%)": class_percent.values.round(2)
    })

    # Display table (utilize full container width for a cleaner Streamlit layout)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # ---------------------------------------------------------
    # Plot bar chart
    # ---------------------------------------------------------
    # Force white background and ensure no seaborn grid styling carries over
    sns.set_style("white")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot using a consistent, professional blue color
    sns.barplot(data=summary_df, x="Glycan Class", y="Count", color='#005a9c', ax=ax)
    
    # Explicitly remove background grid
    ax.grid(False)
    
    # Despine to remove top and right borders
    sns.despine(ax=ax)
    
    # Add count labels directly on top of the bars
    for container in ax.containers:
        ax.bar_label(container, padding=3, fontsize=10, color='#333333', fontweight='bold')
    
    # Typography and labels
    ax.set_title("Distribution of Glycan Classes", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Glycan Class", fontsize=12, fontweight='bold')
    ax.set_ylabel("Count", fontsize=12, fontweight='bold')
    
    # Rotate x-labels and align right to prevent text overlap
    ax.tick_params(axis='x', rotation=45, labelsize=11)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
        
    ax.tick_params(axis='y', labelsize=11)
    
    plt.tight_layout()
    st.pyplot(fig)
    st.session_state["report_glycan_class_fig"] = fig
    add_plot_download_buttons(fig, base_filename="glycan_class_distribution", key_suffix="static_glycan_class")
    
    return summary_df

