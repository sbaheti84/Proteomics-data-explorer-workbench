from typing import Optional
from pathlib import Path
import base64
from io import BytesIO
import re
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ge.misc_utility import _save_bytes_to_downloads

# =============================================================================
# STRUCTURAL WATERFALL HELPERS FOR LFQ GLYCOPEPTIDES
# =============================================================================

def ge_lf_find_glycan_image(glycan_str, img_dir):
    """Find glycan PNG from common pGlyco composition naming styles."""
    if glycan_str is None or pd.isna(glycan_str):
        return None

    g = str(glycan_str).strip()
    if not g or g.lower() == "nan":
        return None

    nums = re.findall(r"\d+", g)
    candidates = [
        f"{g}.png",
        f"{g.replace(' ', '_')}.png",
        f"{g.replace(' ', '')}.png",
        f"Glycan_{g.replace(' ', '_')}.png",
    ]

    if len(nums) >= 4:
        h, n, a, f = nums[:4]
        candidates.extend([
            f"{h} {n} {a} {f}.png",
            f"{h}_{n}_{a}_{f}.png",
            f"H{h}_N{n}_A{a}_F{f}.png",
            f"Glycan_{h}_{n}_{a}_{f}.png",
            f"Glycan_H{h}_N{n}_A{a}_F{f}.png",
        ])

    img_dir = Path(img_dir)
    for c in candidates:
        p = img_dir / c
        if p.exists() and p.is_file():
            return p
    return None


def ge_lf_image_to_base64(img_path):
    if img_path and Path(img_path).exists():
        with open(img_path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")
    return ""


def ge_lf_first_existing_col(df: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def ge_lf_clean_text(x, default=""):
    if x is None or pd.isna(x):
        return default
    s = str(x)
    return "" if s.lower() == "nan" else s


def ge_lf_format_metric(value) -> str:
    val = pd.to_numeric(value, errors="coerce")
    if pd.isna(val):
        return ""
    return f"{val:.2e}" if val < 0.001 else f"{val:.4f}"


def ge_lf_format_structural_waterfall_table(
    df,
    img_dir,
    gene_col,
    pep_col,
    lfc_col,
    metric_col,
    metric_label,
):
    """HTML table shown inside Streamlit."""
    glycan_col = ge_lf_first_existing_col(
        df,
        ["Glycan(H,N,A,F)", "Glycan(H,N,A,G,F)", "Glycan", "Glycan_Composition", "Glycan_Composition_Clean"],
    )
    site_col = ge_lf_first_existing_col(df, ["ProSites", "Glycosite", "GlycoSite", "prosite", "ProSites_norm"])
    protein_col = ge_lf_first_existing_col(df, ["Proteins", "Protein ID", "Protein", "Protein Description"])

    html = [
        "<style>",
        ".lfw-table { width:100%; border-collapse:collapse; font-family:sans-serif; font-size:0.88em; table-layout:fixed; }",
        ".lfw-table th { background:#f8f9fa; border-bottom:2px solid #dee2e6; padding:8px; text-align:center; color:#495057; }",
        ".lfw-table td { border-bottom:1px solid #e9ecef; padding:8px; text-align:center; vertical-align:middle; word-wrap:break-word; }",
        ".lfw-no { font-size:1.1em; font-weight:700; color:#111; }",
        ".lfw-gene { font-weight:700; color:#111; }",
        ".lfw-img { max-height:58px; display:block; margin:0 auto 4px auto; }",
        ".lfw-glycan { font-family:monospace; font-size:1.05em; font-weight:700; color:#111; text-align:center; }",
        ".lfw-small { color:#555; font-size:0.82em; margin-top:3px; word-break:break-word; }",
        ".lfw-up { color:#d62728; font-weight:700; }",
        ".lfw-down { color:#1f77b4; font-weight:700; }",
        "</style>",
        "<table class='lfw-table'>",
        "<thead><tr>",
        "<th style='width:7%;'>No.</th>",
        "<th style='width:15%;'>Gene/Peptide</th>",
        "<th style='width:18%;'>Protein</th>",
        "<th style='width:16%;'>Glycan</th>",
        "<th style='width:11%;'>ProSites</th>",
        "<th style='width:10%;'>log2FC</th>",
        f"<th style='width:11%;'>{metric_label}</th>",
        "</tr></thead><tbody>",
    ]

    for _, row in df.iterrows():
        point_id = row.get("Point_ID", "")
        try:
            point_id = int(point_id)
        except Exception:
            point_id = ""

        gene = ge_lf_clean_text(row.get(gene_col, row.get("Genes", ""))) if gene_col else ge_lf_clean_text(row.get("Genes", ""))
        pep = ge_lf_clean_text(row.get(pep_col, row.get("Peptide", ""))) if pep_col else ge_lf_clean_text(row.get("Peptide", ""))
        protein = ge_lf_clean_text(row.get(protein_col, "")) if protein_col else ""
        protein_short = protein[:25] + "…" if len(protein) > 26 else protein
        site = ge_lf_clean_text(row.get(site_col, "")) if site_col else ""
        glycan = row.get(glycan_col, "") if glycan_col else ""
        glycan_text = ge_lf_clean_text(glycan)

        lfc = pd.to_numeric(row.get(lfc_col, np.nan), errors="coerce")
        lfc_class = "lfw-up" if pd.notna(lfc) and lfc >= 0 else "lfw-down"
        lfc_str = "" if pd.isna(lfc) else f"{lfc:+.2f}"
        sig_str = ge_lf_format_metric(row.get(metric_col, np.nan))

        img_path = ge_lf_find_glycan_image(glycan, img_dir)
        img_tag = f"<img src='{ge_lf_image_to_base64(img_path)}' class='lfw-img'>" if img_path else ""
        glycan_html = f"<div>{img_tag}</div><div class='lfw-glycan'>{glycan_text}</div>"

        html.extend([
            "<tr>",
            f"<td><span class='lfw-no'>{point_id}</span></td>",
            f"<td><span class='lfw-gene'>{gene}</span><div class='lfw-small'>{pep}</div></td>",
            f"<td><div class='lfw-small'>{protein_short}</div></td>",
            f"<td>{glycan_html}</td>",
            f"<td>{site}</td>",
            f"<td><span class='{lfc_class}'>{lfc_str}</span></td>",
            f"<td>{sig_str}</td>",
            "</tr>",
        ])

    html.append("</tbody></table>")
    return "".join(html)


def ge_lf_create_combined_waterfall_png_svg(
    wfl,
    disp_df,
    img_dir,
    gene_col,
    pep_col,
    lfc_col,
    metric_col,
    metric_label,
    alpha_thr,
    fc_thr,
):
    """Static side-by-side waterfall + structural table for downloads."""

    # ------------------------------------------------------------------
    # Waterfall plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    colors = {
        "Up regulated": "#d62728",
        "Down regulated": "#1f77b4",
        "Not Significant": "#bdbdbd",
    }

    for status, sub in wfl.groupby("Status"):
        ax.scatter(
            sub["Rank"],
            sub[lfc_col],
            s=18,
            alpha=0.85,
            color=colors.get(status, "#999999"),
            label=status,
        )

    label_df = wfl.dropna(subset=["Point_ID"]).copy()
    if not label_df.empty:
        x_span = max(float(wfl["Rank"].max() - wfl["Rank"].min()), 1.0)
        y_span = max(float(wfl[lfc_col].max() - wfl[lfc_col].min()), 1.0)
        x_offset = x_span * 0.012
        y_offset = y_span * 0.018

        label_df["x_label"] = label_df["Rank"] + x_offset
        label_df["y_label"] = label_df[lfc_col] + np.where(label_df[lfc_col] > 0, y_offset, -y_offset)

        for _, row in label_df.iterrows():
            ax.plot(
                [row["Rank"], row["x_label"]],
                [row[lfc_col], row["y_label"]],
                color="gray",
                linewidth=0.6,
                alpha=0.8,
            )
            ax.text(
                row["x_label"],
                row["y_label"],
                str(int(row["Point_ID"])),
                fontsize=6,
                ha="left",
                va="center",
                fontweight="bold",
            )

    ax.axhline(0, color="black", linewidth=1.0)
    ax.axhline(fc_thr, color="gray", linestyle="--", linewidth=0.9)
    ax.axhline(-fc_thr, color="gray", linestyle="--", linewidth=0.9)
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

    # ------------------------------------------------------------------
    # Table image
    # ------------------------------------------------------------------
    glycan_col = ge_lf_first_existing_col(
        disp_df,
        ["Glycan(H,N,A,F)", "Glycan(H,N,A,G,F)", "Glycan", "Glycan_Composition", "Glycan_Composition_Clean"],
    )
    site_col = ge_lf_first_existing_col(disp_df, ["ProSites", "Glycosite", "GlycoSite", "ProSites_norm"])
    protein_col = ge_lf_first_existing_col(disp_df, ["Proteins", "Protein ID", "Protein", "Protein Description"])

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
        "No.": 60,
        "Gene/Peptide": 280,
        "Protein": 200,
        "Glycan": 200,
        "ProSites": 100,
        "log2FC": 100,
        metric_label: 120,
    }

    # Dynamic width: table ends right after q-value / p-value column.
    W = sum(cols.values()) + pad * 2
    table_h = header_h + max(1, len(disp_df)) * row_h + 25
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
    for _, row in disp_df.iterrows():
        draw.line((0, y, W, y), fill="#e9ecef", width=2)

        point = row.get("Point_ID", "")
        try:
            point = int(point)
        except Exception:
            point = ""

        gene = ge_lf_clean_text(row.get(gene_col, row.get("Genes", ""))) if gene_col else ge_lf_clean_text(row.get("Genes", ""))
        pep = ge_lf_clean_text(row.get(pep_col, row.get("Peptide", ""))) if pep_col else ge_lf_clean_text(row.get("Peptide", ""))
        protein = ge_lf_clean_text(row.get(protein_col, "")) if protein_col else ""
        glycan = row.get(glycan_col, "") if glycan_col else ""
        glycan_text = ge_lf_clean_text(glycan)
        site = ge_lf_clean_text(row.get(site_col, "")) if site_col else ""
        lfc = pd.to_numeric(row.get(lfc_col, np.nan), errors="coerce")
        sig_val = pd.to_numeric(row.get(metric_col, np.nan), errors="coerce")
        lfc_color = "#d62728" if pd.notna(lfc) and lfc >= 0 else "#1f77b4"

        draw.text((x_positions["No."], y + 45), str(point), fill="black", font=font_bold)

        draw.text((x_positions["Gene/Peptide"], y + 18), gene[:22], fill="black", font=font_bold)
        for j, line in enumerate(textwrap.wrap(pep, width=27)[:2]):
            draw.text((x_positions["Gene/Peptide"], y + 52 + j * 20), line, fill="#555555", font=font_small)

        for j, line in enumerate(textwrap.wrap(protein, width=24)[:3]):
            draw.text((x_positions["Protein"], y + 22 + j * 20), line, fill="#333333", font=font_small)

        gx = x_positions["Glycan"]
        img_path = ge_lf_find_glycan_image(glycan, img_dir)
        if img_path:
            try:
                gly_img = Image.open(img_path).convert("RGBA")
                gly_img.thumbnail((120, 50))
                table_img.paste(gly_img, (gx + 35, y + 8), gly_img)
            except Exception:
                pass

        draw.text((gx + 35, y + 72), glycan_text[:40], fill="black", font=font)
        draw.text((x_positions["ProSites"], y + 45), site[:12], fill="black", font=font)
        draw.text((x_positions["log2FC"] + 5, y + 45), "" if pd.isna(lfc) else f"{lfc:+.2f}", fill=lfc_color, font=font_bold)
        draw.text((x_positions[metric_label] + 5, y + 45), ge_lf_format_metric(sig_val), fill="black", font=font_small)

        y += row_h

    draw.line((0, y, W, y), fill="#e9ecef", width=2)

    target_h = max(plot_img.height, table_img.height)
    plot_resized = plot_img.resize((int(plot_img.width * target_h / plot_img.height), target_h))
    table_resized = table_img.resize((int(table_img.width * target_h / table_img.height), target_h))

    gap = 30
    combined_w = plot_resized.width + table_resized.width + gap
    combined = Image.new("RGB", (combined_w, target_h), "white")
    combined.paste(plot_resized, (0, 0))
    combined.paste(table_resized, (plot_resized.width + gap, 0))

    png_out = BytesIO()
    combined.save(png_out, format="PNG", dpi=(300, 300))
    png_out.seek(0)
    png_bytes = png_out.getvalue()

    # This SVG wraps the PNG so Streamlit downloads one file consistently.
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="{combined_w}" height="{target_h}">
        <image href="data:image/png;base64,{b64}" width="{combined_w}" height="{target_h}"/>
    </svg>
    """.encode("utf-8")

    return png_bytes, svg


def render_ge_lf_structural_waterfall(
    diff_df,
    glycan_image_dir,
    sig_col="q-value",
    alpha_thr=0.05,
    fc_thr=1.0,
    top_n=5,
):
    if "log2FC" not in diff_df.columns:
        st.info("Waterfall plot is only available for pairwise results with log2FC.")
        return

    required = ["log2FC", sig_col]
    missing = [c for c in required if c not in diff_df.columns]
    if missing:
        st.error("Missing required columns for waterfall: " + ", ".join(missing))
        return

    lfc_col = "log2FC"
    metric_col = sig_col
    metric_label = "q-value" if "q" in str(sig_col).lower() else "p-value"
    gene_col = ge_lf_first_existing_col(diff_df, ["Genes", "Gene", "GeneID", "Gene Symbol"])
    pep_col = ge_lf_first_existing_col(diff_df, ["Peptide", "Glycopeptide", "Annotated Sequence", "Sequence", "Peptide Sequence"])

    wfl = diff_df.copy()
    wfl[lfc_col] = pd.to_numeric(wfl[lfc_col], errors="coerce")
    wfl[metric_col] = pd.to_numeric(wfl[metric_col], errors="coerce")
    wfl = wfl.dropna(subset=[lfc_col, metric_col]).sort_values(lfc_col, ascending=False).reset_index(drop=True)

    if wfl.empty:
        st.info("No valid rows available for the waterfall plot.")
        return

    wfl["Rank"] = wfl.index + 1
    wfl["Status"] = "Not Significant"
    wfl.loc[(wfl[lfc_col] >= fc_thr) & (wfl[metric_col] <= alpha_thr), "Status"] = "Up regulated"
    wfl.loc[(wfl[lfc_col] <= -fc_thr) & (wfl[metric_col] <= alpha_thr), "Status"] = "Down regulated"

    top_up = wfl[wfl["Status"] == "Up regulated"].head(top_n)
    top_down = wfl[wfl["Status"] == "Down regulated"].sort_values(lfc_col, ascending=True).head(top_n)
    disp_df = pd.concat([top_up, top_down], ignore_index=True)

    wfl["Point_ID"] = np.nan
    if not disp_df.empty:
        disp_df = disp_df.copy()
        disp_df["Point_ID"] = range(1, len(disp_df) + 1)
        wfl["Point_ID"] = wfl["Rank"].map(disp_df.set_index("Rank")["Point_ID"].to_dict())

    col_plot, col_table = st.columns([0.56, 0.44], gap="large")

    with col_plot:
        hover_cols = [
            c for c in ["Point_ID", gene_col, pep_col, "Glycan", "ProSites", lfc_col, metric_col]
            if c and c in wfl.columns
        ]

        fig_wfl = px.scatter(
            wfl,
            x="Rank",
            y=lfc_col,
            color="Status",
            color_discrete_map={
                "Up regulated": "#d62728",
                "Down regulated": "#1f77b4",
                "Not Significant": "#bdbdbd",
            },
            hover_data=hover_cols,
            height=600,
        )

        fig_wfl.add_hline(y=0, line_dash="solid", line_color="black")
        fig_wfl.add_hline(y=fc_thr, line_dash="dash", line_color="gray")
        fig_wfl.add_hline(y=-fc_thr, line_dash="dash", line_color="gray")

        label_df = wfl.dropna(subset=["Point_ID"]).copy()
        if not label_df.empty:
            x_offset = max(float(wfl["Rank"].max()), 1.0) * 0.012
            y_offset = max(float(wfl[lfc_col].max() - wfl[lfc_col].min()), 1.0) * 0.018
            label_df["x_label"] = label_df["Rank"] + x_offset
            label_df["y_label"] = label_df[lfc_col] + np.where(label_df[lfc_col] > 0, y_offset, -y_offset)

            fig_wfl.add_scatter(
                x=label_df["x_label"],
                y=label_df["y_label"],
                mode="text",
                text=label_df["Point_ID"].astype(int).astype(str),
                textfont=dict(size=12, color="black"),
                showlegend=False,
                hoverinfo="skip",
            )

            for _, row in label_df.iterrows():
                fig_wfl.add_shape(
                    type="line",
                    x0=row["Rank"],
                    y0=row[lfc_col],
                    x1=row["x_label"],
                    y1=row["y_label"],
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
        if disp_df.empty:
            st.info("No features meet the selected waterfall thresholds.")
            return

        html = ge_lf_format_structural_waterfall_table(
            disp_df,
            glycan_image_dir,
            gene_col,
            pep_col,
            lfc_col,
            metric_col,
            metric_label,
        )
        st.markdown(html, unsafe_allow_html=True)

        combined_png, combined_svg = ge_lf_create_combined_waterfall_png_svg(
            wfl,
            disp_df,
            glycan_image_dir,
            gene_col,
            pep_col,
            lfc_col,
            metric_col,
            metric_label,
            alpha_thr,
            fc_thr,
        )

        d1, d2, d3 = st.columns(3)
        with d1:
            if st.button("Save PNG", key="lfq_wfl_png", use_container_width=True, help="Directly save PNG to your local Downloads folder."):
                try:
                    outpath = _save_bytes_to_downloads(combined_png, "lfq_glycopeptide_waterfall_table.png")
                    st.success(f"Saved PNG to: {outpath}")
                except Exception as e:
                    st.error(f"Could not save PNG: {e}")
        with d2:
            if st.button("Save SVG", key="lfq_wfl_svg", use_container_width=True, help="Directly save SVG to your local Downloads folder."):
                try:
                    outpath = _save_bytes_to_downloads(combined_svg, "lfq_glycopeptide_waterfall_table.svg")
                    st.success(f"Saved SVG to: {outpath}")
                except Exception as e:
                    st.error(f"Could not save SVG: {e}")
        with d3:
            if st.button("Save Table CSV", key="lfq_wfl_csv", use_container_width=True, help="Directly save CSV to your local Downloads folder."):
                try:
                    csv_bytes = disp_df.to_csv(index=False).encode("utf-8")
                    outpath = _save_bytes_to_downloads(csv_bytes, "lfq_waterfall_top_features.csv")
                    st.success(f"Saved CSV to: {outpath}")
                except Exception as e:
                    st.error(f"Could not save CSV: {e}")
