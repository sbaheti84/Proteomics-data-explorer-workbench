#!/usr/bin/env python
# -*- coding: utf-8 -*-

from io import BytesIO
import re
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.patches import Patch
from PIL import Image

# Import the centralized helper
from ge.misc_utility import add_plot_download_buttons


# -------------------------------------------------------------------------
# Shared helpers
# -------------------------------------------------------------------------

def _clean_sample_names(cols):
    return [str(col).replace("Abundance: ", "").strip() for col in cols]


def _clean_label(name: str) -> str:
    s = str(name)
    s = re.sub(r"^\s*Abundance\s*:\s*", "", s)
    return s.split(":", 1)[-1].strip() if ":" in s else s.strip()


def _convert_to_string(comp_str):
    labels = ["Hex", "HexNAc", "NeuAc", "Fuc"]
    try:
        numbers = list(map(int, str(comp_str).split()))
        return "".join(f"{label}{count}" for label, count in zip(labels, numbers) if count > 0)
    except Exception:
        return str(comp_str)


def _prepare_plot_matrix(
    df: pd.DataFrame,
    cols: list[str],
    quant_mode: str = "TMT",
    lfq_zero_as_na: bool = True,
    lfq_pseudocount: float = 1.0,
):
    """
    TMT -> row-scaled Relative Abundance (%)
    LFQ -> Log2 LFQ Intensity
    """
    quant_mode = str(quant_mode).strip().upper()
    X = df[cols].apply(pd.to_numeric, errors="coerce").copy()

    if quant_mode == "LFQ":
        if lfq_zero_as_na:
            X = X.replace(0, np.nan)
            X = np.log2(X)
        else:
            X = np.log2(X + float(lfq_pseudocount))

        y_label = "Log2 LFQ Intensity"
    else:
        row_max = X.max(axis=1).replace(0, pd.NA)
        X = X.div(row_max, axis=0) * 100.0
        y_label = "Relative Abundance (%)"

    return X, y_label


def _normalize_heatmap_data(df_data: pd.DataFrame, normalize: str):
    normalize = str(normalize).strip()
    if normalize == "Z-score":
        out = df_data.sub(df_data.mean(axis=1), axis=0).div(df_data.std(axis=1), axis=0)
        cbar_label = "Z-score"
        cmap = "vlag"
        center = 0
    else:
        out = df_data.copy()
        cbar_label = None  # set later from quant mode
        cmap = "YlGnBu"
        center = None

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(how="all", axis=0)
    out = out.dropna(how="all", axis=1)
    
    out = out.fillna(0) 
    
    return out, cbar_label, cmap, center


def _plot_elbow_curve(data, max_k=10, checkbox_key="zoom_heatmap"):
    inertias = []
    if data.shape[0] < 2:
        st.info("Not enough rows to calculate an elbow curve.")
        return

    max_k = min(max_k, max(1, data.shape[0]))
    ks = range(1, max_k + 1)

    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(list(ks), inertias, marker="o", markersize=6, linewidth=2)
    ax.set_xlabel("k", fontsize=12, fontweight="bold")
    ax.set_ylabel("Inertia", fontsize=12, fontweight="bold")
    ax.set_title("Elbow Curve", fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", labelsize=12)
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    thumb = Image.open(buf).resize((250, 200))
    st.image(thumb, caption="Click to enlarge", use_container_width=False)

    show_zoom = st.checkbox("Show full-size elbow plot", key=checkbox_key)
    if show_zoom:
        st.image(buf, caption="Elbow Plot (Zoomed)", use_container_width=True)

    plt.close(fig)


def _finalize_clustermap(
    g,
    cbar_label: str,
    title: str,
    legend_handles,
    cax_position,
    adjust_kwargs,
):
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(),
        fontsize=12,
        fontweight="bold",
        rotation=0
    )
    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(),
        fontsize=12,
        fontweight="bold",
        rotation=45,
        ha="right"
    )

    g.ax_heatmap.set_xlabel("Samples", fontsize=14, fontweight="bold", labelpad=15)
    g.ax_heatmap.set_ylabel("Glycopeptides", fontsize=14, fontweight="bold", labelpad=15)

    g.cax.set_title(cbar_label, fontsize=12, fontweight="bold", pad=12)
    g.cax.set_ylabel("")
    g.cax.tick_params(labelsize=10)
    g.cax.set_position(cax_position)

    if legend_handles:
        g.ax_heatmap.legend(
            handles=legend_handles,
            title="Group",
            loc="upper left",
            bbox_to_anchor=(0.95, 1.2),
            fontsize=12,
            title_fontsize=12,
            frameon=True
        )

    g.fig.subplots_adjust(**adjust_kwargs)
    g.fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    st.pyplot(g.fig)


# -------------------------------------------------------------------------
# 1. create_heatmap
# -------------------------------------------------------------------------

def create_heatmap(
    df: pd.DataFrame,
    pval_col: str,
    top_n: int,
    control_cols: list[str],
    test_cols: list[str],
    control_label: str = "Reference",
    test_label: str = "Test",
    normalize: str = "Z-score",
    cluster_option: str = "K-Means",
    control_color: str = "#87CEEB",
    test_color: str = "#FA8072",
    quant_mode: str = "TMT",
    lfq_zero_as_na: bool = True,
    lfq_pseudocount: float = 1.0,
) -> None:

    control_cols_clean = _clean_sample_names(control_cols)
    test_cols_clean = _clean_sample_names(test_cols)
    rename_map = dict(zip(control_cols + test_cols, control_cols_clean + test_cols_clean))
    df = df.rename(columns=rename_map).copy()

    df["glycan_name"] = df["Glycan"].apply(_convert_to_string)
    df["Glycopeptide"] = (
        df["Genes"].astype(str) + "-" +
        df["ProSites"].astype(str) + "-" +
        df["glycan_name"].astype(str)
    )

    top_hits = df.nsmallest(top_n, pval_col).drop_duplicates(subset="Glycopeptide").copy()
    selected_cols = test_cols_clean + control_cols_clean

    plot_vals, raw_label = _prepare_plot_matrix(
        top_hits,
        selected_cols,
        quant_mode=quant_mode,
        lfq_zero_as_na=lfq_zero_as_na,
        lfq_pseudocount=lfq_pseudocount,
    )
    plot_vals.index = top_hits["Glycopeptide"]

    df_meta = pd.concat([
        pd.DataFrame({"Sample_ID": test_cols_clean, "Group": test_label}),
        pd.DataFrame({"Sample_ID": control_cols_clean, "Group": control_label})
    ], ignore_index=True).set_index("Sample_ID")

    df_meta = df_meta.loc[plot_vals.columns]

    df_data, cbar_label, cmap, center = _normalize_heatmap_data(plot_vals, normalize=normalize)
    if cbar_label is None:
        cbar_label = raw_label

    if df_data.empty or df_data.shape[0] < 1:
        st.warning("Not enough data available to generate the heatmap.")
        return

    group_palette = {control_label: control_color, test_label: test_color}
    col_colors = df_meta["Group"].map(group_palette)

    if cluster_option == "K-Means":
        st.subheader("Elbow Plot to Select Number of Clusters")
        _plot_elbow_curve(df_data, max_k=10, checkbox_key="zoom_standard")
        selected_k = st.slider(
            "Select number of clusters (k)",
            min_value=1,
            max_value=min(len(df_data), 10),
            value=min(4, max(1, len(df_data))),
            key="heatmap_k"
        )
        kmeans = KMeans(n_clusters=selected_k, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(df_data)
        sorted_idx = pd.Series(cluster_labels, index=df_data.index).sort_values().index
        df_data = df_data.loc[sorted_idx]
        row_cluster = False
        col_cluster = True

    elif cluster_option == "Hierarchical":
        row_cluster = True
        col_cluster = True
    else:
        row_cluster = False
        col_cluster = False

    n_rows = df_data.shape[0]
    fig_height = max(10, n_rows * 0.4)

    g = sns.clustermap(
        df_data,
        figsize=(14, fig_height),
        dendrogram_ratio=(0.20, 0.05),
        colors_ratio=(0.02, 0.02),
        cmap=cmap,
        center=center,
        col_cluster=col_cluster,
        row_cluster=row_cluster,
        col_colors=col_colors,
        tree_kws={"linewidth": 1.5, "color": "#333333"}
    )

    legend_handles = [Patch(color=group_palette[grp], label=grp) for grp in group_palette]

    _finalize_clustermap(
        g=g,
        cbar_label=cbar_label,
        title="",
        legend_handles=legend_handles,
        cax_position=[0.02, 0.85, 0.02, 0.10],
        adjust_kwargs=dict(left=0.10, right=0.65, top=0.90, bottom=0.15),
    )

    add_plot_download_buttons(g.fig, base_filename="heatmap_plot", key_suffix="static_heatmap_plot")
    st.session_state["report_heatmap_fig"] = g.fig


# -------------------------------------------------------------------------
# 2. create_heatmap_anova
# -------------------------------------------------------------------------

def create_heatmap_anova(
    df: pd.DataFrame,
    pval_col: str,
    top_n: int,
    group_samples: list[list[str]],
    group_labels: list[str],
    normalize: str = "Z-score",
    cluster_option: str = "K-Means",
    group_colors: dict = None,
    quant_mode: str = "TMT",
    lfq_zero_as_na: bool = True,
    lfq_pseudocount: float = 1.0,
) -> None:

    all_cols = [item for sublist in group_samples for item in sublist]
    all_cols_clean = _clean_sample_names(all_cols)
    rename_map = dict(zip(all_cols, all_cols_clean))
    df = df.rename(columns=rename_map).copy()

    df["glycan_name"] = df["Glycan"].apply(_convert_to_string)
    df["Glycopeptide"] = (
        df["Genes"].astype(str) + "-" +
        df["ProSites"].astype(str) + "-" +
        df["glycan_name"].astype(str)
    )

    top_hits = df.nsmallest(top_n, pval_col).drop_duplicates(subset="Glycopeptide").copy()
    clean_group_cols = [_clean_sample_names(g) for g in group_samples]
    selected_cols = [col for g in clean_group_cols for col in g]

    plot_vals, raw_label = _prepare_plot_matrix(
        top_hits,
        selected_cols,
        quant_mode=quant_mode,
        lfq_zero_as_na=lfq_zero_as_na,
        lfq_pseudocount=lfq_pseudocount,
    )
    plot_vals.index = top_hits["Glycopeptide"]

    df_meta = pd.DataFrame({
        "Sample_ID": sum(clean_group_cols, []),
        "Group": sum([[label] * len(g) for label, g in zip(group_labels, clean_group_cols)], [])
    }).set_index("Sample_ID").loc[plot_vals.columns]

    df_data, cbar_label, cmap, center = _normalize_heatmap_data(plot_vals, normalize=normalize)
    if cbar_label is None:
        cbar_label = raw_label

    if df_data.empty or df_data.shape[0] < 1:
        st.warning("Not enough data available to generate the ANOVA heatmap.")
        return

    if group_colors is not None:
        group_palette = group_colors
    else:
        group_palette = dict(zip(group_labels, sns.color_palette("Set2", len(group_labels))))

    col_colors = df_meta["Group"].map(group_palette)

    if cluster_option == "K-Means":
        st.subheader("Elbow Plot to Select Number of Clusters")
        _plot_elbow_curve(df_data, max_k=10, checkbox_key="zoom_anova")
        selected_k = st.slider(
            "Select number of clusters (k)",
            min_value=1,
            max_value=min(len(df_data), 10),
            value=min(4, max(1, len(df_data))),
            key="heatmap_k_anova"
        )
        kmeans = KMeans(n_clusters=selected_k, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(df_data)
        sorted_idx = pd.Series(cluster_labels, index=df_data.index).sort_values().index
        df_data = df_data.loc[sorted_idx]
        row_cluster = False
        col_cluster = True

    elif cluster_option == "Hierarchical":
        row_cluster = True
        col_cluster = True
    else:
        row_cluster = False
        col_cluster = False

    n_rows = df_data.shape[0]
    fig_height = max(10, n_rows * 0.4)

    g = sns.clustermap(
        df_data,
        figsize=(14, fig_height),
        dendrogram_ratio=(0.20, 0.05),
        colors_ratio=(0.02, 0.02),
        cmap=cmap,
        center=center,
        col_cluster=col_cluster,
        row_cluster=row_cluster,
        col_colors=col_colors,
        tree_kws={"linewidth": 1.5, "color": "#333333"}
    )

    legend_handles = [Patch(color=group_palette[grp], label=grp) for grp in group_palette]

    _finalize_clustermap(
        g=g,
        cbar_label=cbar_label,
        title="ANOVA Differential Expression Heatmap",
        legend_handles=legend_handles,
        cax_position=[0.02, 0.85, 0.02, 0.10],
        adjust_kwargs=dict(left=0.10, right=0.65, top=0.90, bottom=0.15),
    )

    add_plot_download_buttons(g.fig, base_filename="heatmap_plot_anova", key_suffix="static_heatmap_plot")
    st.session_state["report_heatmap_fig"] = g.fig


# -------------------------------------------------------------------------
# 3. create_overview_heatmap
# -------------------------------------------------------------------------

def create_overview_heatmap(
    df: pd.DataFrame,
    sample_cols: list[str],
    top_n: int = 20,
    norm_type: str = "Z-score",
    quant_mode: str = "LFQ",
    lfq_zero_as_na: bool = True,
    lfq_pseudocount: float = 1.0,
):
    """
    Creates an overview clustermap of the top N most variable features.
    Supports LFQ and TMT raw-scale handling.
    """

    data = df.dropna(subset=sample_cols).copy()
    for col in sample_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=sample_cols)
    if data.empty:
        st.warning("Not enough numeric data to calculate feature variance.")
        return None

    transformed_vals, raw_label = _prepare_plot_matrix(
        data,
        sample_cols,
        quant_mode=quant_mode,
        lfq_zero_as_na=lfq_zero_as_na,
        lfq_pseudocount=lfq_pseudocount,
    )

    variance_df = transformed_vals.copy()
    variance_df["Variance"] = variance_df.var(axis=1)
    top_idx = variance_df.sort_values("Variance", ascending=False).head(top_n).index

    top_data = data.loc[top_idx].copy()
    hm_data = transformed_vals.loc[top_idx].copy()

    if "Peptide" in top_data.columns:
        top_data["Peptide"] = top_data["Peptide"].astype(str)
        if top_data["Peptide"].duplicated().any():
            top_data["Peptide"] = top_data["Peptide"] + "_" + top_data.groupby("Peptide").cumcount().astype(str)
        hm_data.index = top_data["Peptide"]
    else:
        hm_data.index = [f"Feature_{i+1}" for i in range(len(hm_data))]

    hm_data.columns = [_clean_label(c) for c in hm_data.columns]
    hm_data = hm_data.astype(float)

    if norm_type == "Z-score":
        hm_plot = hm_data.sub(hm_data.mean(axis=1), axis=0).div(hm_data.std(axis=1), axis=0)
        cmap = "vlag"
        center = 0
        cbar_title = "Z-score"
    else:
        hm_plot = hm_data.copy()
        cmap = "YlGnBu"
        center = None
        cbar_title = raw_label

    hm_plot = hm_plot.replace([np.inf, -np.inf], np.nan)
    hm_plot = hm_plot.dropna(how="all", axis=0).dropna(how="all", axis=1)
    
    hm_plot = hm_plot.fillna(0)

    if hm_plot.empty:
        st.warning("Overview heatmap matrix is empty after preprocessing.")
        return None

    n_rows = hm_plot.shape[0]
    fig_height = max(8, n_rows * 0.35)

    try:
        fig = sns.clustermap(
            hm_plot,
            cmap=cmap,
            center=center,
            figsize=(12, fig_height),
            linewidths=0.0,
            col_cluster=True,
            row_cluster=True,
            tree_kws={"linewidth": 1.5, "color": "#333333"}
        )

        fig.ax_heatmap.set_yticklabels(
            fig.ax_heatmap.get_yticklabels(),
            fontsize=10,
            fontweight="bold",
            rotation=0
        )

        fig.ax_heatmap.set_xticklabels(
            fig.ax_heatmap.get_xticklabels(),
            fontsize=12,
            fontweight="bold",
            rotation=45,
            ha="right"
        )

        fig.ax_heatmap.set_ylabel("Features", fontweight="bold", labelpad=15)
        fig.ax_heatmap.set_xlabel("Samples", fontweight="bold", labelpad=15)

        fig.fig.subplots_adjust(right=0.85, top=0.92)

        cpos = fig.cax.get_position()
        new_width = cpos.width * 0.5
        new_height = cpos.height * 0.5
        fig.cax.set_position([0.88, 0.85, new_width, new_height])
        fig.cax.tick_params(labelsize=8)
        fig.cax.set_title(cbar_title, fontsize=10, fontweight="bold", pad=10)

        fig.fig.suptitle(
            f"Overview Heatmap (Top {top_n} Variable Features)",
            fontsize=14, fontweight="bold",y=0.98
        )

        st.pyplot(fig.fig)

        quant_mode_str = str(quant_mode).lower()

        if "add_plot_download_buttons" in globals():
            add_plot_download_buttons(
                fig.fig,
                base_filename=f"overview_heatmap_{quant_mode_str}",
                key_suffix=f"overview_heatmap_{quant_mode_str}"
            )

        st.session_state["report_overview_heatmap_fig"] = fig.fig

        return fig.fig

    except Exception as e:
        st.error(f"Clustermap plotting failed: {e}")
        return None