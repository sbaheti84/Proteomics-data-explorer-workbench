import uuid
import re
from io import BytesIO
from typing import List
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# -------------------------------------------------------------------------
# Global Matplotlib Settings (Centralized)
# -------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Nimbus Sans", "DejaVu Sans"],
    "font.weight": "bold",
    "font.size": 12,          # Enforce global base size
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "svg.fonttype": "none",   # keep SVG text editable
    "pdf.fonttype": 42,       # better editable text in Illustrator
    "ps.fonttype": 42
})

def _get_downloads_dir() -> Path:
    """
    Return user's Downloads folder if available.
    Falls back to home directory.
    """
    downloads = Path.home() / "Downloads"
    if downloads.exists():
        return downloads
    return Path.home()


def safe_filename(filename: str) -> str:
    """
    Remove characters that can cause problems in filenames.
    """
    filename = str(filename)
    filename = re.sub(r"[^\w\-.]+", "_", filename)
    filename = filename.strip("_")
    return filename or "plot"


def _save_bytes_to_downloads(data: bytes, filename: str) -> Path:
    """
    Save bytes directly to local Downloads folder.
    This is more reliable in packaged Streamlit desktop apps.
    """
    outdir = _get_downloads_dir()
    outpath = outdir / safe_filename(filename)  # Updated here

    counter = 1
    original_outpath = outpath

    while outpath.exists():
        outpath = original_outpath.with_name(
            f"{original_outpath.stem}_{counter}{original_outpath.suffix}"
        )
        counter += 1

    outpath.write_bytes(data)
    return outpath


def add_plot_download_buttons(fig, base_filename: str = "plot", key_suffix: str = ""):
    """Add PNG and SVG native Streamlit download buttons for a matplotlib figure."""
    if not key_suffix:
        key_suffix = str(uuid.uuid4())[:8]

    base_filename = safe_filename(base_filename)

    # Generate PNG bytes
    png_buf = BytesIO()
    fig.savefig(png_buf, format="png", dpi=300, bbox_inches="tight")
    png_bytes = png_buf.getvalue()

    # Generate SVG bytes
    svg_buf = BytesIO()
    fig.savefig(svg_buf, format="svg", dpi=300, bbox_inches="tight")
    svg_bytes = svg_buf.getvalue()

    save_col1, save_col2 = st.columns(2)

    with save_col1:
        # Capture the button click in a variable
        png_clicked = st.download_button(
            label="Download PNG",
            data=png_bytes,
            file_name=f"{base_filename}.png",
            mime="image/png",
            key=f"save_png_{base_filename}_{key_suffix}",
            use_container_width=True
        )
        # Show message if clicked
        if png_clicked:
            st.success(f"Downloaded {base_filename}.png")

    with save_col2:
        # Capture the button click in a variable
        svg_clicked = st.download_button(
            label="Download SVG",
            data=svg_bytes,
            file_name=f"{base_filename}.svg",
            mime="image/svg+xml",
            key=f"save_svg_{base_filename}_{key_suffix}",
            use_container_width=True
        )
        # Show message if clicked
        if svg_clicked:
            st.success(f"Downloaded {base_filename}.svg")

    plt.close(fig)

def create_pca_plot(
    df: pd.DataFrame,
    group_samples: List[List[str]],
    group_labels: List[str],
    title: str = "PCA",
    pc_x: str = "PC1",
    pc_y: str = "PC2",
    hide_outliers: bool = False,
    label_outliers: bool = True,
    draw_ellipses: bool = False,
    zoom_q: float = 1.0,
    palette: dict = None
):
    if df is None or df.empty:
        st.warning("Expression data is empty.")
        return None

    # Flatten sample list preserving order
    all_samples = []
    group_of_sample = {}
    for lbl, cols in zip(group_labels, group_samples):
        for c in cols:
            if c not in group_of_sample:
                all_samples.append(c)
                group_of_sample[c] = lbl

    # Keep only samples that exist
    all_samples = [s for s in all_samples if s in df.columns]
    if len(all_samples) < 2:
        st.warning("Need at least 2 samples for PCA.")
        return None

    # Subset to selected samples; drop rows with all missing in those samples
    data = df[all_samples].copy()
    data = data.dropna(how="all")

    # Fill remaining NaNs with row-median (PCA can't handle NaNs)
    row_median = data.median(axis=1)
    data = data.apply(lambda col: col.fillna(row_median))

    # samples × features
    X = data.T.values

    # scale
    X_scaled = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

    # PCA Calculation (Dynamic Components)
    n_comps = min(10, X_scaled.shape[0], X_scaled.shape[1])
    if n_comps < 2:
        st.warning("Not enough components to plot PCA (requires at least 2).")
        return None

    pca = PCA(n_components=n_comps, random_state=0)
    scores = pca.fit_transform(X_scaled)
    
    pc_cols = [f"PC{i+1}" for i in range(n_comps)]
    pc_vars = {f"PC{i+1}": pca.explained_variance_ratio_[i] * 100 for i in range(n_comps)}

    pc_df = pd.DataFrame(scores, columns=pc_cols)
    pc_df["Sample"] = all_samples
    pc_df["Group"] = [group_of_sample[s] for s in all_samples]

    # Validate requested PCs exist in case of dataset changes
    if pc_x not in pc_cols:
        pc_x = pc_cols[0]
    if pc_y not in pc_cols:
        pc_y = pc_cols[1] if len(pc_cols) > 1 else pc_cols[0]

    # Outlier detection (robust IQR) mapped to selected PCs
    def iqr_outlier(s, k=1.5):
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        return (s < lo) | (s > hi)

    pc_df["Outlier"] = iqr_outlier(pc_df[pc_x], k=2.0) | iqr_outlier(pc_df[pc_y], k=2.0)

    # Fallback Palette if not provided by UI
    if palette is None:
        unique_groups = list(dict.fromkeys(group_labels))
        default_colors = sns.color_palette("Set2", n_colors=len(unique_groups)).as_hex()
        palette = {grp: default_colors[idx] for idx, grp in enumerate(unique_groups)}

    plot_df = pc_df[~pc_df["Outlier"]].copy() if hide_outliers else pc_df.copy()

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.scatterplot(
        data=plot_df,
        x=pc_x, y=pc_y,
        hue="Group",
        palette=palette,  
        s=95,
        edgecolor="black",
        linewidth=0.8,
        ax=ax,
        zorder=3
    )

    # Confidence Ellipses (95%)
    if draw_ellipses:
        for group, color in palette.items():
            group_data = plot_df[plot_df["Group"] == group]
            x_vals = group_data[pc_x].values
            y_vals = group_data[pc_y].values
            
            if len(x_vals) >= 3:
                cov = np.cov(x_vals, y_vals)
                if cov[0, 0] > 0 and cov[1, 1] > 0:
                    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
                    ell_radius_x = np.sqrt(1 + pearson)
                    ell_radius_y = np.sqrt(1 - pearson)
                    
                    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                                      facecolor=color, edgecolor=color, alpha=0.15, linewidth=2, zorder=1)
                    
                    scale_x = np.sqrt(cov[0, 0]) * 2.0  
                    scale_y = np.sqrt(cov[1, 1]) * 2.0
                    mean_x, mean_y = np.mean(x_vals), np.mean(y_vals)
                    
                    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
                    ellipse.set_transform(transf + ax.transData)
                    ax.add_patch(ellipse)

    if label_outliers:
        lab_df = plot_df[plot_df["Outlier"]] if not hide_outliers else pd.DataFrame()
        for _, r in lab_df.iterrows():
            ax.text(r[pc_x], r[pc_y], str(r["Sample"]), fontsize=9, ha="left", va="bottom", zorder=4)

    # View Limits mapped to selected PCs
    if len(plot_df) >= 3 and zoom_q < 1.0:
        qlo = 1 - zoom_q
        xlo, xhi = plot_df[pc_x].quantile(qlo), plot_df[pc_x].quantile(zoom_q)
        ylo, yhi = plot_df[pc_y].quantile(qlo), plot_df[pc_y].quantile(zoom_q)

        xpad = (xhi - xlo) * 0.08 if xhi > xlo else 1.0
        ypad = (yhi - ylo) * 0.08 if yhi > ylo else 1.0
        ax.set_xlim(xlo - xpad, xhi + xpad)
        ax.set_ylim(ylo - ypad, yhi + ypad)
    elif draw_ellipses:
        ax.relim()
        ax.autoscale_view()

    ax.set_title(title, fontsize=15)
    ax.set_xlabel(f"{pc_x} ({pc_vars[pc_x]:.1f}%)")
    ax.set_ylabel(f"{pc_y} ({pc_vars[pc_y]:.1f}%)")

    # Removed Grid!
    ax.grid(False) 
    
    plt.legend(title="Group", title_fontsize=12, fontsize=11, loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.tight_layout()

    st.pyplot(fig)
    add_plot_download_buttons(fig, "pca_plot")

    if pc_df["Outlier"].any():
        out = pc_df.loc[pc_df["Outlier"], ["Sample", "Group", pc_x, pc_y]].sort_values(pc_x, ascending=False)
        st.caption("Outliers detected (check these for loading / normalization / missingness issues):")
        st.dataframe(out, use_container_width=True, hide_index=True)

    st.session_state["report_pca_fig"] = fig
    return fig