from typing import Optional
from pathlib import Path
import base64
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

from PIL import Image, ImageDraw, ImageFont
from ge.misc_utility import _save_bytes_to_downloads

# -----------------------------------------------------------------------------
# Structural waterfall helpers for Results Visualizations
# -----------------------------------------------------------------------------
def _ge_first_existing_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _ge_find_glycan_image(glycan_str, img_dir: Path):
    if glycan_str is None or pd.isna(glycan_str):
        return None
    g_str = str(glycan_str).strip()
    if not g_str or g_str.lower() == "nan":
        return None

    candidates = [
        f"{g_str}.png",
        f"{g_str.replace(' ', '_')}.png",
        f"{g_str.replace(' ', '')}.png",
        f"Glycan_{g_str.replace(' ', '_')}.png",
    ]
    for fname in candidates:
        p = Path(img_dir) / fname
        if p.exists() and p.is_file():
            return p
    return None


def _ge_image_to_base64(img_path):
    if img_path and Path(img_path).exists():
        with open(img_path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")
    return ""


def _ge_clean_text(x, default=""):
    if x is None or pd.isna(x):
        return default
    s = str(x)
    return "" if s.lower() == "nan" else s


def _ge_format_structural_waterfall_table(
    df: pd.DataFrame,
    glycan_image_dir: Path,
    gene_col: Optional[str],
    peptide_col: Optional[str],
    glycan_col: Optional[str],
    prosite_col: Optional[str],
    annotation_col: Optional[str],
    lfc_col: str,
    metric_col: str,
    metric_label: str,
) -> str:
    html = [
        "<style>",
        ".ge-wfl-table { width: 100%; border-collapse: collapse; font-family: sans-serif; font-size: 0.88em; table-layout: fixed; }",
        ".ge-wfl-table th { background-color: #f8f9fa; border-bottom: 2px solid #dee2e6; padding: 8px; text-align: center; color: #495057; }",
        ".ge-wfl-table td { border-bottom: 1px solid #e9ecef; padding: 8px; text-align: center; vertical-align: middle; word-wrap: break-word; }",
        ".ge-wfl-no { font-size: 1.1em; font-weight: 700; color: #111; }",
        ".ge-wfl-gene { font-weight: 700; color: #111; }",
        ".ge-wfl-pep { font-family: monospace; font-size: 0.95em; color: #333; margin-top: 4px; }",
        ".ge-wfl-small { color: #666; font-size: 0.82em; margin-top: 3px; }",
        ".ge-wfl-img { max-height: 58px; display: block; margin: 0 auto 4px auto; }",
        ".ge-wfl-up { color: #d62728; font-weight: 700; }",
        ".ge-wfl-down { color: #1f77b4; font-weight: 700; }",
        "</style>",
        "<table class='ge-wfl-table'>",
        "<thead><tr>",
        "<th style='width:7%;'>No.</th>",
        "<th style='width:13%;'>Gene</th>",
        "<th style='width:13%;'>Glycan</th>",
        "<th style='width:12%;'>ProSites</th>",
        "<th style='width:10%;'>log2FC</th>",
        f"<th style='width:14%;'>{metric_label}</th>",
        "</tr></thead><tbody>",
    ]

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        point_id = row.get("Point_ID", i)
        try:
            point_id = int(point_id)
        except Exception:
            point_id = i

        gene = _ge_clean_text(row.get(gene_col, "")) if gene_col else ""
        peptide = _ge_clean_text(row.get(peptide_col, "")) if peptide_col else ""
        glycan = row.get(glycan_col, "") if glycan_col else ""
        glycan_text = _ge_clean_text(glycan)
        prosite = _ge_clean_text(row.get(prosite_col, "")) if prosite_col else ""
        annot = _ge_clean_text(row.get(annotation_col, "")) if annotation_col else ""

        try:
            lfc = float(row.get(lfc_col, 0.0))
        except Exception:
            lfc = 0.0
        try:
            metric_val = float(row.get(metric_col, np.nan))
        except Exception:
            metric_val = np.nan
        metric_str = f"{metric_val:.2e}" if pd.notna(metric_val) and metric_val < 0.001 else (f"{metric_val:.4f}" if pd.notna(metric_val) else "")

        lfc_class = "ge-wfl-up" if lfc >= 0 else "ge-wfl-down"
        img_path = _ge_find_glycan_image(glycan_text, glycan_image_dir)
        img_tag = ""
        if img_path:
            img_tag = f"<img src='{_ge_image_to_base64(img_path)}' class='ge-wfl-img'>"

        feature_html = (
            f"<div>{img_tag}</div>"
            f"<div class='ge-wfl-pep' style='font-size:1.05em; font-weight:700; margin-top:4px;'>{glycan_text}</div>"
        )

        html.extend([
            "<tr>",
            f"<td><span class='ge-wfl-no'>{point_id}</span></td>",
            f"<td><span class='ge-wfl-gene'>{gene}</span></td>",
            f"<td>{feature_html}</td>",
            f"<td>{prosite}</td>",
            f"<td><span class='{lfc_class}'>{lfc:+.2f}</span></td>",
            f"<td>{metric_str}</td>",
            "</tr>",
        ])

    html.append("</tbody></table>")
    return "".join(html)


def _ge_create_waterfall_export(
    wfl: pd.DataFrame,
    label_df: pd.DataFrame,
    glycan_image_dir: Path,
    gene_col: Optional[str],
    peptide_col: Optional[str],
    glycan_col: Optional[str],
    prosite_col: Optional[str],
    annotation_col: Optional[str],
    lfc_col: str,
    metric_col: str,
    metric_label: str,
    alpha_thr: float,
    log2fc_thr: float,
):
    from PIL import ImageDraw, ImageFont
    import textwrap

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    color_map = {
        "Up regulated": "#d62728",
        "Down regulated": "#1f77b4",
        "Not Significant": "#bdbdbd",
    }
    for status, sub in wfl.groupby("Status"):
        ax.scatter(
            sub["Rank"], sub[lfc_col],
            s=18, alpha=0.85,
            color=color_map.get(status, "#999999"),
            label=status,
        )

    if not label_df.empty:
        x_span = max(float(wfl["Rank"].max() - wfl["Rank"].min()), 1.0)
        y_span = max(float(wfl[lfc_col].max() - wfl[lfc_col].min()), 1.0)

        x_offset = x_span * 0.012
        y_offset = y_span * 0.018

        label_plot = label_df.copy()
        label_plot["x_label"] = label_plot["Rank"] + x_offset
        label_plot["y_label"] = label_plot[lfc_col] + np.where(
            label_plot[lfc_col] > 0,
            y_offset,
            -y_offset
        )

        for _, row in label_plot.iterrows():
            ax.plot(
                [row["Rank"], row["x_label"]],
                [row[lfc_col], row["y_label"]],
                color="gray",
                linewidth=0.6,
                alpha=0.8
            )

            ax.text(
                row["x_label"],
                row["y_label"],
                str(int(row["Point_ID"])),
                fontsize=6,
                ha="left",
                va="center",
                fontweight="bold"
            )

    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(log2fc_thr, color="gray", linestyle="--", linewidth=0.9)
    ax.axhline(-log2fc_thr, color="gray", linestyle="--", linewidth=0.9)
    #ax.set_title(f"({metric_label} < {alpha_thr}, |log2FC| > {log2fc_thr})", fontweight="bold")
    ax.set_xlabel("Glycopeptide", fontweight="bold")
    ax.set_ylabel("log2 Fold-Change", fontweight="bold")
    ax.grid(False)
    ax.legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    plot_buf = BytesIO()
    fig.savefig(plot_buf, format="png", dpi=300, bbox_inches="tight")
    plot_buf.seek(0)
    plt.close(fig)
    plot_img = Image.open(plot_buf).convert("RGB")

    table_df = label_df.copy()
  #  W = sum(cols.values()) + pad * 2
    row_h = 120
    header_h = 64
    pad = 18
    
    try:
        font = ImageFont.truetype("Arial.ttf", 22)
        font_bold = ImageFont.truetype("Arial Bold.ttf", 24)
        font_small = ImageFont.truetype("Arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
        font_bold = ImageFont.load_default()
        font_small = ImageFont.load_default()

    cols = {
        "No.": 70,
        "Gene": 150,
        "Glycan": 200,
        "ProSites": 150,
        "log2FC": 120,
        metric_label: 90,
    }

    W = sum(cols.values()) + pad * 2
    table_h = header_h + max(1, len(table_df)) * row_h + 25
    table_img = Image.new("RGB", (W, table_h), "white")
    draw = ImageDraw.Draw(table_img)

    x_positions = {}
    x = pad
    for c, width in cols.items():
        x_positions[c] = x
        x += width

    draw.rectangle([0, 0, W, header_h], fill="#f8f9fa")
    for c, xpos in x_positions.items():
        draw.text((xpos, 18), c, fill="#495057", font=font_bold)

    y = header_h
    for _, row in table_df.iterrows():
        draw.line((0, y, W, y), fill="#e9ecef", width=2)
        point_id = int(row.get("Point_ID", 0)) if pd.notna(row.get("Point_ID", np.nan)) else ""
        gene = _ge_clean_text(row.get(gene_col, "")) if gene_col else ""
        peptide = _ge_clean_text(row.get(peptide_col, "")) if peptide_col else ""
        glycan = row.get(glycan_col, "") if glycan_col else ""
        glycan_text = _ge_clean_text(glycan)
        prosite = _ge_clean_text(row.get(prosite_col, "")) if prosite_col else ""
        annot = _ge_clean_text(row.get(annotation_col, "")) if annotation_col else ""
        lfc = float(row.get(lfc_col, 0.0))
        metric_val = float(row.get(metric_col, np.nan)) if pd.notna(row.get(metric_col, np.nan)) else np.nan
        metric_str = f"{metric_val:.2e}" if pd.notna(metric_val) and metric_val < 0.001 else (f"{metric_val:.4f}" if pd.notna(metric_val) else "")
        lfc_color = "#d62728" if lfc >= 0 else "#1f77b4"

        draw.text((x_positions["No."], y + 45), str(point_id), fill="black", font=font_bold)
        draw.text((x_positions["Gene"], y + 45), gene[:16], fill="black", font=font_bold)

        gx = x_positions["Glycan"]
        img_path = _ge_find_glycan_image(glycan_text, glycan_image_dir)
        if img_path:
            try:
                gly_img = Image.open(img_path).convert("RGBA")
                gly_img.thumbnail((190, 58))
                table_img.paste(gly_img, (gx, y + 8), gly_img)
            except Exception:
                pass

        # Only show the glycan composition text below the structure.
        draw.text((gx, y + 72), glycan_text[:60], fill="black", font=font_bold)

        draw.text((x_positions["ProSites"], y + 45), prosite[:18], fill="black", font=font)
        draw.text((x_positions["log2FC"], y + 45), f"{lfc:+.2f}", fill=lfc_color, font=font_bold)
        draw.text((x_positions[metric_label], y + 45), metric_str, fill="black", font=font)
        y += row_h
    draw.line((0, y, W, y), fill="#e9ecef", width=2)

    target_h = max(plot_img.height, table_img.height)
    plot_resized = plot_img.resize((int(plot_img.width * target_h / plot_img.height), target_h))
    table_resized = table_img.resize((int(table_img.width * target_h / table_img.height), target_h))
    gap = 40
    combined_w = plot_resized.width + table_resized.width + gap
    combined = Image.new("RGB", (combined_w, target_h), "white")
    combined.paste(plot_resized, (0, 0))
    combined.paste(table_resized, (plot_resized.width + gap, 0))

    png_out = BytesIO()
    combined.save(png_out, format="PNG", dpi=(300, 300))
    png_out.seek(0)
    png_bytes = png_out.getvalue()
    svg_b64 = base64.b64encode(png_bytes).decode("utf-8")
    svg_text = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="{combined_w}" height="{target_h}">
        <image href="data:image/png;base64,{svg_b64}" width="{combined_w}" height="{target_h}"/>
    </svg>
    """.encode("utf-8")
    return png_bytes, svg_text


def render_ge_structural_waterfall(
    plot_df: pd.DataFrame,
    glycan_image_dir: Path,
    sig_col: str,
    alpha_thr: float,
    log2fc_thr: float,
    top_n: int,
):
    if "log2FC" not in plot_df.columns:
        st.info("Waterfall plot is only available for pairwise results with log2FC.")
        return
    if sig_col not in plot_df.columns:
        st.error(f"Missing significance column: {sig_col}")
        return

    gene_col = _ge_first_existing_col(plot_df, ["Genes", "Gene", "GeneID", "Gene Symbol"])
    peptide_col = _ge_first_existing_col(plot_df, ["Peptide", "Annotated Sequence", "Sequence", "Peptide Sequence"])
    glycan_col = _ge_first_existing_col(plot_df, ["Glycan(H,N,A,F)", "Glycan(H,N,A,G,F)", "Glycan", "Glycan_Composition", "Glycan_Composition_Clean"])
    prosite_col = _ge_first_existing_col(plot_df, ["ProSites", "Glycosite", "GlycoSite", "Glycosites"])
    annotation_col = _ge_first_existing_col(plot_df, ["pGlyco Annotation", "P.Glyco.Site", "GlycoAnnotation", "Glycan_Composition", "GlycanClass"])
    metric_label = "q-value" if "q" in sig_col.lower() else "p-value"

    wfl = plot_df.copy()
    wfl["log2FC"] = pd.to_numeric(wfl["log2FC"], errors="coerce")
    wfl[sig_col] = pd.to_numeric(wfl[sig_col], errors="coerce")
    wfl = wfl.dropna(subset=["log2FC", sig_col]).sort_values("log2FC", ascending=False).reset_index(drop=True)
    wfl["Rank"] = wfl.index + 1
    wfl["Status"] = "Not Significant"
    wfl.loc[(wfl["log2FC"] >= log2fc_thr) & (wfl[sig_col] <= alpha_thr), "Status"] = "Up regulated"
    wfl.loc[(wfl["log2FC"] <= -log2fc_thr) & (wfl[sig_col] <= alpha_thr), "Status"] = "Down regulated"

    top_up = wfl[wfl["Status"] == "Up regulated"].head(top_n)
    top_down = wfl[wfl["Status"] == "Down regulated"].sort_values("log2FC", ascending=True).head(top_n)
    label_df = pd.concat([top_up, top_down], ignore_index=True)
    wfl["Point_ID"] = np.nan
    if not label_df.empty:
        label_df = label_df.copy()
        label_df["Point_ID"] = range(1, len(label_df) + 1)
        wfl["Point_ID"] = wfl["Rank"].map(label_df.set_index("Rank")["Point_ID"].to_dict())

    col_plot, col_table = st.columns([0.56, 0.44], gap="large")

    with col_plot:
        hover_cols = [c for c in [gene_col, peptide_col, glycan_col, prosite_col, "log2FC", sig_col] if c and c in wfl.columns]
        fig_wfl = px.scatter(
            wfl,
            x="Rank",
            y="log2FC",
            color="Status",
            color_discrete_map={
                "Up regulated": "#d62728",
                "Down regulated": "#1f77b4",
                "Not Significant": "#bdbdbd",
            },
            hover_data=hover_cols,
            height=600
        )
        fig_wfl.add_hline(y=0, line_dash="solid", line_color="black")
        fig_wfl.add_hline(y=log2fc_thr, line_dash="dash", line_color="gray")
        fig_wfl.add_hline(y=-log2fc_thr, line_dash="dash", line_color="gray")

        if not label_df.empty:
            x_offset = max(float(wfl["Rank"].max()), 1.0) * 0.012
            y_offset = max(float(wfl["log2FC"].max() - wfl["log2FC"].min()), 1.0) * 0.018
            label_plot = wfl.dropna(subset=["Point_ID"]).copy()
            label_plot["x_label"] = label_plot["Rank"] + x_offset
            label_plot["y_label"] = label_plot["log2FC"] + np.where(label_plot["log2FC"] > 0, y_offset, -y_offset)
            fig_wfl.add_scatter(
                x=label_plot["x_label"],
                y=label_plot["y_label"],
                mode="text",
                text=label_plot["Point_ID"].astype(int).astype(str),
                textfont=dict(size=12, color="black"),
                showlegend=False,
                hoverinfo="skip",
            )
            for _, row in label_plot.iterrows():
                fig_wfl.add_shape(
                    type="line",
                    x0=row["Rank"], y0=row["log2FC"],
                    x1=row["x_label"], y1=row["y_label"],
                    line=dict(color="gray", width=1),
                )

        max_rank = float(wfl["Rank"].max()) if not wfl.empty else 1.0
        fig_wfl.update_xaxes(range=[0, max_rank * 1.06])
        fig_wfl.update_traces(marker=dict(size=5, opacity=0.85))
        fig_wfl.update_layout(
            xaxis_title="Glycopeptide",
            yaxis_title="log2 Fold-Change",
            xaxis=dict(showgrid=False, showline=True, linewidth=2, linecolor="black", mirror=True, ticks="outside"),
            yaxis=dict(showgrid=False, showline=True, linewidth=2, linecolor="black", mirror=True, ticks="outside"),
            legend=dict(title="Status", orientation="v", yanchor="top", y=1, xanchor="left", x=1.12),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(t=45, l=20, r=110, b=45),
        )
        st.plotly_chart(fig_wfl, use_container_width=True)

    with col_table:
        if label_df.empty:
            st.info("No features meet the selected waterfall thresholds.")
        else:
            #st.markdown("##### Numbered Top Significant Glycopeptides")
            html_table = _ge_format_structural_waterfall_table(
                label_df, glycan_image_dir,
                gene_col, peptide_col, glycan_col, prosite_col, annotation_col,
                "log2FC", sig_col, metric_label,
            )
            st.markdown(html_table, unsafe_allow_html=True)

            export_cols = ["Point_ID"] + [c for c in [gene_col, peptide_col, glycan_col, prosite_col, annotation_col, "log2FC", sig_col, "Status", "Rank"] if c and c in label_df.columns]
            csv_bytes = label_df[export_cols].to_csv(index=False).encode("utf-8")
            png_bytes, svg_bytes = _ge_create_waterfall_export(
                wfl, label_df, glycan_image_dir,
                gene_col, peptide_col, glycan_col, prosite_col, annotation_col,
                "log2FC", sig_col, metric_label, alpha_thr, log2fc_thr,
            )
            d1, d2, d3 = st.columns(3)
            with d1:
                if st.button("Download PNG", key="ge_wfl_png", use_container_width=True, help="Directly save PNG to your local Downloads folder."):
                    try:
                        outpath = _save_bytes_to_downloads(png_bytes, "glycoexplorer_waterfall_table.png")
                        st.success(f"Saved PNG to: {outpath}")
                    except Exception as e:
                        st.error(f"Could not save PNG: {e}")
            with d2:
                if st.button("Download SVG", key="ge_wfl_svg", use_container_width=True, help="Directly save SVG to your local Downloads folder."):
                    try:
                        outpath = _save_bytes_to_downloads(svg_bytes, "glycoexplorer_waterfall_table.svg")
                        st.success(f"Saved SVG to: {outpath}")
                    except Exception as e:
                        st.error(f"Could not save SVG: {e}")
            with d3:
                if st.button("Download CSV", key="ge_wfl_csv", use_container_width=True, help="Directly save CSV to your local Downloads folder."):
                    try:
                        outpath = _save_bytes_to_downloads(csv_bytes, "glycoexplorer_waterfall_top_features.csv")
                        st.success(f"Saved CSV to: {outpath}")
                    except Exception as e:
                        st.error(f"Could not save CSV: {e}")

