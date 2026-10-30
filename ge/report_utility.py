#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GlycoExplorer report utilities.

"""

import datetime
import os
import tempfile
import re
from typing import List, Dict, Any, Optional

from fpdf import FPDF
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# =============================================================================
# PDF CLASS
# =============================================================================

class AnalysisReport(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 16)
        self.set_text_color(0, 90, 156)  # Mayo Blue
        self.cell(0, 10, "GlycoExplorer | Final Analysis Report", ln=True, align="C")

        self.set_font("helvetica", "I", 8)
        self.set_text_color(100)
        self.cell(
            0,
            5,
            f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
            ln=True,
            align="C",
        )
        self.ln(6)

    def footer(self):
        self.set_y(-12)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(120)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")

    def section_title(self, title: str):
        self.set_font("helvetica", "B", 14)
        self.set_text_color(0, 90, 156)
        self.cell(0, 9, title, ln=True)
        self.ln(2)

    def subsection_title(self, title: str):
        self.set_font("helvetica", "B", 11)
        self.set_text_color(0, 90, 156)
        self.cell(0, 7, title, ln=True)
        self.ln(1)

    def section_text(self, text: str, font_size: int = 10):
        self.set_font("helvetica", "", font_size)
        self.set_text_color(0)
        self.multi_cell(0, 5.5, str(text))
        self.ln(1)

    def add_kv_lines(self, lines: List[str], font_size: int = 10):
        self.set_font("helvetica", "", font_size)
        self.set_text_color(0)
        for line in lines:
            self.multi_cell(0, 5.5, str(line))
        self.ln(3)


# =============================================================================
# SMALL HELPERS
# =============================================================================

def _clean_sample_label(name: str) -> str:
    s = str(name)
    patterns = [
        r'^(?:Abundance)\s*:\s*(.+)$',
        r'^(?:Abundance)\s*\([^)]*\)\s*:\s*(.+)$',
        r'^(?:Abundances?)\s*\([^)]*\)\s*:\s*(.+)$',
        r'^(?:Abundance)\s+(.+)$',
    ]
    for pat in patterns:
        m = re.match(pat, s, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
            
    if ":" in s:
        s = s.split(":", 1)[-1].strip()
    return s.strip()


def _as_bool_yes_no(x) -> str:
    return "Yes" if bool(x) else "No"


def _get_batch_status(ss) -> tuple[bool, str, dict]:
    """Return (batch_applied, display_status, sample_to_batch). Works for LFQ and TMT session keys."""
    batch_corrected = bool(ss.get("batch_corrected", False))
    run_combat = bool(ss.get("run_combat", False))
    batch_applied = batch_corrected or run_combat

    method = ss.get("batch_method_ui", None) or ss.get("batch_method", None)
    if run_combat and not method:
        method = "ComBat (Empirical Bayes)"

    if batch_applied:
        status = f"Applied ({method})" if method else "Applied"
    else:
        status = "Not Applied"

    sample_to_batch = {}

    # Newer apps use batch_dict: {Batch 1: [sample columns], ...}
    batch_dict = ss.get("batch_dict", {}) or {}
    if isinstance(batch_dict, dict):
        for b_name, cols in batch_dict.items():
            for c in cols or []:
                sample_to_batch[str(c)] = str(b_name)
                sample_to_batch[c] = str(b_name)

    # Legacy ComBat path may use batch_mapping_df with Sample/Batch columns
    batch_df = ss.get("batch_mapping_df", pd.DataFrame())
    if isinstance(batch_df, pd.DataFrame) and not batch_df.empty and {"Sample", "Batch"}.issubset(batch_df.columns):
        sample_to_batch.update(dict(zip(batch_df["Sample"].astype(str), batch_df["Batch"].astype(str))))

    return batch_applied, status, sample_to_batch


def _build_sample_mapping_df(ss) -> pd.DataFrame:
    sample_cols = ss.get("global_sample_cols", []) or []
    group_samples = ss.get("global_group_samples", []) or []
    group_labels = ss.get("global_group_labels", []) or []
    control_idx = ss.get("global_control_group_idx", 0)
    batch_applied, _, sample_to_batch = _get_batch_status(ss)

    rows = []
    assigned_samples = {c for cols in group_samples for c in (cols or [])}

    for col in sample_cols:
        if assigned_samples and col not in assigned_samples:
            continue

        group_name = "Unassigned"
        group_num = None
        is_reference = False

        for i, cols in enumerate(group_samples):
            if col in (cols or []):
                group_name = group_labels[i] if i < len(group_labels) else f"Group{i + 1}"
                group_num = i + 1
                is_reference = (i == control_idx)
                break

        row_data = {
            "Column": str(col),
            "Sample": _clean_sample_label(col),
            "Group": group_name,
            "Group_Number": "" if group_num is None else str(group_num),
            "Is_Reference": _as_bool_yes_no(is_reference),
        }

        if batch_applied:
            row_data["Batch"] = sample_to_batch.get(col, sample_to_batch.get(str(col), "N/A"))

        rows.append(row_data)

    return pd.DataFrame(rows)


def _add_sample_mapping_table(pdf: AnalysisReport, mapping_df: pd.DataFrame, batch_applied: bool = False):
    pdf.add_page()
    pdf.section_title("2. Sample Mapping")

    if mapping_df is None or mapping_df.empty:
        pdf.section_text("No sample mapping information was available in session state.")
        return

    pdf.section_text(
        "This table shows the samples included in the active analysis, their assigned groups, "
        "reference status, and batch assignment when batch correction was applied.",
        font_size=9,
    )

    if batch_applied and "Batch" in mapping_df.columns:
        col_names = ["Sample", "Group", "Group_Number", "Batch", "Is_Reference"]
        headers = ["Sample", "Group", "Group #", "Batch", "Reference"]
        widths = [72, 48, 18, 25, 24]
    else:
        col_names = ["Sample", "Group", "Group_Number", "Is_Reference"]
        headers = ["Sample", "Group", "Group #", "Reference"]
        widths = [86, 55, 20, 25]

    row_h = 7
    pdf.set_font("helvetica", "B", 9)
    pdf.set_fill_color(230, 240, 250)
    pdf.set_text_color(0)
    for header, width in zip(headers, widths):
        pdf.cell(width, row_h, header, border=1, align="C", fill=True)
    pdf.ln(row_h)

    pdf.set_font("helvetica", "", 8)
    table_df = mapping_df[[c for c in col_names if c in mapping_df.columns]].copy()

    for _, row in table_df.iterrows():
        if pdf.get_y() > 265:
            pdf.add_page()
            pdf.section_title("2. Sample Mapping (continued)")
            pdf.set_font("helvetica", "B", 9)
            pdf.set_fill_color(230, 240, 250)
            for header, width in zip(headers, widths):
                pdf.cell(width, row_h, header, border=1, align="C", fill=True)
            pdf.ln(row_h)
            pdf.set_font("helvetica", "", 8)

        values = [str(row.get(c, "")) for c in col_names]
        if values:
            values[0] = values[0][:42] + "..." if len(values[0]) > 45 else values[0]
        if len(values) > 1:
            values[1] = values[1][:24] + "..." if len(values[1]) > 27 else values[1]

        for value, width in zip(values, widths):
            pdf.cell(width, row_h, value, border=1)
        pdf.ln(row_h)

    pdf.ln(4)


# =============================================================================
# TABLE HELPERS
# =============================================================================

def _row_height(pdf, data, col_widths, line_h=4):
    max_lines = 1
    for text, width in zip(data, col_widths):
        try:
            lines = pdf.multi_cell(width, line_h, str(text), split_only=True)
            max_lines = max(max_lines, len(lines))
        except Exception:
            max_lines = max(max_lines, 1)
    return max_lines * line_h


def _format_numeric_for_table(series: pd.Series, decimals: int = 4) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").map(
        lambda x: f"{x:.{decimals}f}" if pd.notnull(x) else ""
    )


def _clip_text(x, max_len: int = 80) -> str:
    s = "" if pd.isna(x) else str(x)
    return s[: max_len - 3] + "..." if len(s) > max_len else s


def _add_dataframe_table(pdf, df, title, max_rows=25, note: Optional[str] = None):
    pdf.add_page()
    pdf.section_title(title)

    if note:
        pdf.section_text(note, font_size=9)

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        pdf.section_text("No table data available.")
        return

    df = df.head(max_rows).copy()

    # Keep table readable: limit to common informative columns first, then any remaining.
    preferred = [
        "Point_ID", "Gene", "Genes", "Peptide", "Glycan", "Glycan(H,N,A,F)", "GlycanClass",
        "ProSites", "Proteins", "log2FC", "Mean_log2FC", "p-value", "q-value",
        "adj.p-value", "Best_q_value", "Regulation", "Status", "Rank",
        "Significant_Peptides", "Significant_Features", "Upregulated_Hits", "Downregulated_Hits",
    ]
    ordered = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[ordered]

    for col in [
        "log2FC", "p-value", "q-value", "adj.p-value",
        "Mean_log2FC", "Best_q_value"
    ]:
        if col in df.columns:
            df[col] = _format_numeric_for_table(df[col], decimals=4)

    # Clip long text fields to avoid huge rows.
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            max_len = 55 if col in ["Peptide", "Proteins"] else 38
            df[col] = df[col].map(lambda x: _clip_text(x, max_len=max_len))

    col_names = df.columns.tolist()

    col_width_map = {
        "Point_ID": 14,
        "Gene": 22,
        "Genes": 22,
        "Peptide": 34,
        "Glycan": 20,
        "Glycan(H,N,A,F)": 22,
        "GlycanClass": 20,
        "ProSites": 16,
        "Proteins": 28,
        "log2FC": 17,
        "Mean_log2FC": 19,
        "p-value": 18,
        "q-value": 18,
        "adj.p-value": 18,
        "Best_q_value": 20,
        "Regulation": 25,
        "Status": 24,
        "Rank": 14,
        "Significant_Peptides": 28,
        "Significant_Features": 28,
        "Upregulated_Hits": 24,
        "Downregulated_Hits": 26,
    }
    col_widths = [col_width_map.get(col, 20) for col in col_names]

    # Scale if too wide for A4 portrait width.
    max_total = 186
    total = sum(col_widths)
    if total > max_total:
        scale = max_total / total
        col_widths = [max(10, w * scale) for w in col_widths]

    header_h = 6
    line_h = 4

    pdf.set_font("helvetica", "B", 6.8)
    pdf.set_fill_color(230, 240, 250)
    pdf.set_text_color(0)

    for col, w in zip(col_names, col_widths):
        pdf.cell(w, header_h, str(col), border=1, align="C", fill=True)
    pdf.ln()

    pdf.set_font("helvetica", "", 6)

    for _, row in df.iterrows():
        row_data = [str(row[col]) for col in col_names]
        h = _row_height(pdf, row_data, col_widths, line_h=line_h)

        if pdf.get_y() + h > 270:
            pdf.add_page()
            pdf.section_title(f"{title} (continued)")
            pdf.set_font("helvetica", "B", 6.8)
            pdf.set_fill_color(230, 240, 250)
            for col, w in zip(col_names, col_widths):
                pdf.cell(w, header_h, str(col), border=1, align="C", fill=True)
            pdf.ln()
            pdf.set_font("helvetica", "", 6)

        y0 = pdf.get_y()
        x_start = pdf.get_x()
        for text, w in zip(row_data, col_widths):
            x0 = pdf.get_x()
            pdf.multi_cell(w, line_h, text, border=1)
            pdf.set_xy(x0 + w, y0)
        pdf.set_xy(x_start, y0 + h)


# =============================================================================
# FIGURE HELPERS
# =============================================================================

def _save_fig_to_temp(fig, dpi=180) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
        temp_img_path = tmp_img.name

    if hasattr(fig, "savefig"):
        fig.savefig(temp_img_path, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    elif hasattr(fig, "write_image"):
        # Requires kaleido for plotly. If not installed, this raises and will be reported in PDF.
        fig.write_image(temp_img_path, scale=2)
    else:
        raise TypeError("Unsupported figure type. Expected matplotlib figure or plotly figure.")

    return temp_img_path


def _figure_interpretation(title: str) -> str:
    t = title.lower()
    if "pca" in t or "principal component" in t:
        return "Interpretation: Samples that cluster together have more similar glycopeptide abundance profiles. Separation by group supports biological differences; separation by batch suggests technical effects."
    if "correlation" in t:
        return "Interpretation: High within-group correlation supports sample reproducibility. Low-correlation samples should be reviewed for outlier behavior or technical issues."
    if "volcano" in t:
        return "Interpretation: Points farther left or right have larger fold changes. Points higher on the plot are more statistically significant."
    if "waterfall" in t:
        return "Interpretation: Glycopeptides are ranked by fold change. Numbered features represent the strongest significant up- and down-regulated changes under the selected thresholds."
    if "heatmap" in t:
        return "Interpretation: Clusters indicate glycopeptides and samples with similar abundance patterns. Group-specific clusters may highlight coordinated biological changes."
    if "glycan classification" in t or "glycan class" in t:
        return "Interpretation: This plot summarizes the distribution of glycan classes and helps assess whether expected glycan biology is represented in the dataset."
    if "box" in t:
        return "Interpretation: Box plots show sample-level abundance distributions and help determine whether a result is consistent across replicates or driven by an outlier."
    if "lollipop" in t:
        return "Interpretation: Lollipop plots map glycosylation changes onto protein sites, helping identify site-specific glycoform regulation."
    return ""


def _add_figure_page(pdf: AnalysisReport, fig, title: str, subtitle: str = None):
    pdf.add_page()
    pdf.section_title(title)

    interpretation = _figure_interpretation(title)
    if interpretation:
        pdf.section_text(interpretation, font_size=9)

    if subtitle:
        pdf.section_text(subtitle, font_size=9)

    temp_img_path = None
    try:
        temp_img_path = _save_fig_to_temp(fig, dpi=180)

        top_y = pdf.get_y() + 2
        max_w = 186
        max_h = 255 - top_y

        from PIL import Image
        with Image.open(temp_img_path) as im:
            img_w, img_h = im.size

        scale = min(max_w / img_w, max_h / img_h)
        draw_w = img_w * scale
        draw_h = img_h * scale

        x = (210 - draw_w) / 2
        y = top_y
        pdf.image(temp_img_path, x=x, y=y, w=draw_w, h=draw_h)

    finally:
        if temp_img_path and os.path.exists(temp_img_path):
            os.remove(temp_img_path)


def _collect_report_figures(ss) -> List[Dict[str, Any]]:
    # Added 'report_sample_clustering_fig' and 'report_waterfall_fig' to the list
    figure_specs = [
        ("report_pca_fig", "3. Principal Component Analysis (PCA)"),
        ("report_corr_fig", "4. Sample Correlation Heatmap"),
        ("report_density_fig", "5. Abundance Density Plot"),
        ("report_sample_clustering_fig", "6. Sample Clustering"),
        ("report_glycan_class_fig", "7. Glycan Classification Summary"),
        ("report_volcano_fig", "8. Differential Volcano Plot"),
        ("report_waterfall_fig", "9. Waterfall Plot of Differential Glycopeptides"),
        ("report_ma_fig", "10. MA Plot"),
        ("report_heatmap_fig", "11. Clustered Heatmap of Top Hits"),
        ("report_boxplot_fig", "12. Differential Box Plot"),
        ("report_lollipop_fig", "13. Glycosite Lollipop Plot"),
    ]

    found = []
    for key, title in figure_specs:
        try:
            # Check for the key and ensure it's not None
            if key in ss and ss[key] is not None:
                found.append({"key": key, "title": title})
        except Exception:
            pass
    return found


# =============================================================================
# DIFFERENTIAL SUMMARY HELPERS
# =============================================================================

def _best_q_col(df: pd.DataFrame):
    for col in ["q-value", "adj.p-value", "p-value"]:
        if col in df.columns:
            return col
    return None


def _clean_gene_tokens(val) -> list:
    if pd.isna(val):
        return []
    s = str(val).strip()
    if not s:
        return []
    s = s.replace(";", ",")
    return [g.strip() for g in s.split(",") if g.strip()]


def _build_diff_summary(diff_df: pd.DataFrame) -> dict:
    out = {
        "is_anova": True,
        "n_rows": 0,
        "n_sig": 0,
        "n_up": 0,
        "n_down": 0,
        "top_up": pd.DataFrame(),
        "top_down": pd.DataFrame(),
        "top_sig": pd.DataFrame(),
        "sig_gene_summary": pd.DataFrame(),
        "top_gene_hits": pd.DataFrame(),
    }

    if diff_df is None or not isinstance(diff_df, pd.DataFrame) or diff_df.empty:
        return out

    df = diff_df.copy()
    out["n_rows"] = len(df)

    for col in ["log2FC", "p-value", "q-value", "adj.p-value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    q_col = _best_q_col(df)
    out["is_anova"] = "log2FC" not in df.columns

    if q_col:
        sig_df = df[df[q_col] < 0.05].copy()
    else:
        sig_df = df.copy()

    out["n_sig"] = len(sig_df)

    if not out["is_anova"]:
        if "Regulation" in df.columns:
            reg = df["Regulation"].astype(str).str.lower()
            out["n_up"] = int(reg.str.contains("up").sum())
            out["n_down"] = int(reg.str.contains("down").sum())
        elif q_col and "log2FC" in df.columns:
            out["n_up"] = int(((df["log2FC"] > 0) & (df[q_col] < 0.05)).sum())
            out["n_down"] = int(((df["log2FC"] < 0) & (df[q_col] < 0.05)).sum())

        pairwise_cols = [
            c for c in [
                "Peptide", "Glycan", "Glycan(H,N,A,F)", "GlycanClass", "Genes", "ProSites",
                "log2FC", "p-value", "q-value", "adj.p-value", "Regulation"
            ] if c in df.columns
        ]

        if "log2FC" in df.columns:
            up_df = sig_df[sig_df["log2FC"] > 0].copy()
            down_df = sig_df[sig_df["log2FC"] < 0].copy()

            if not up_df.empty:
                sort_cols = ["log2FC"]
                ascending = [False]
                if q_col in up_df.columns:
                    sort_cols.append(q_col)
                    ascending.append(True)
                out["top_up"] = up_df.sort_values(sort_cols, ascending=ascending)[pairwise_cols].head(15)

            if not down_df.empty:
                down_df["_abs_fc"] = down_df["log2FC"].abs()
                sort_cols = ["_abs_fc"]
                ascending = [False]
                if q_col in down_df.columns:
                    sort_cols.append(q_col)
                    ascending.append(True)
                out["top_down"] = (
                    down_df.sort_values(sort_cols, ascending=ascending)
                    .drop(columns="_abs_fc", errors="ignore")[pairwise_cols]
                    .head(15)
                )

        if "Genes" in sig_df.columns:
            gene_rows = []
            for _, row in sig_df.iterrows():
                for gene in _clean_gene_tokens(row.get("Genes", "")):
                    gene_rows.append(
                        {
                            "Gene": gene,
                            "Peptide": row.get("Peptide", ""),
                            "log2FC": row.get("log2FC", np.nan),
                            "q-value": row.get("q-value", row.get("adj.p-value", row.get("p-value", np.nan))),
                            "Regulation": row.get("Regulation", ""),
                        }
                    )

            if gene_rows:
                gene_long = pd.DataFrame(gene_rows)
                out["sig_gene_summary"] = (
                    gene_long.groupby("Gene", dropna=True)
                    .agg(
                        Significant_Peptides=("Peptide", "nunique"),
                        Best_q_value=("q-value", "min"),
                        Mean_log2FC=("log2FC", "mean"),
                        Upregulated_Hits=("Regulation", lambda x: int(x.astype(str).str.lower().str.contains("up").sum())),
                        Downregulated_Hits=("Regulation", lambda x: int(x.astype(str).str.lower().str.contains("down").sum())),
                    )
                    .reset_index()
                    .sort_values(["Significant_Peptides", "Best_q_value"], ascending=[False, True])
                    .head(20)
                )
                out["top_gene_hits"] = (
                    gene_long.sort_values(["q-value"], ascending=True)
                    [["Gene", "Peptide", "log2FC", "q-value", "Regulation"]]
                    .head(20)
                )

    else:
        anova_cols = [
            c for c in [
                "Peptide", "Glycan", "Glycan(H,N,A,F)", "GlycanClass", "Genes", "ProSites",
                "p-value", "q-value", "adj.p-value", "Regulation"
            ] if c in df.columns
        ]

        if q_col and not sig_df.empty:
            out["top_sig"] = sig_df.sort_values(q_col, ascending=True)[anova_cols].head(20)

        if "Genes" in sig_df.columns and not sig_df.empty:
            gene_rows = []
            for _, row in sig_df.iterrows():
                for gene in _clean_gene_tokens(row.get("Genes", "")):
                    gene_rows.append(
                        {
                            "Gene": gene,
                            "Peptide": row.get("Peptide", ""),
                            "q-value": row.get("q-value", row.get("adj.p-value", row.get("p-value", np.nan))),
                        }
                    )

            if gene_rows:
                gene_long = pd.DataFrame(gene_rows)
                out["sig_gene_summary"] = (
                    gene_long.groupby("Gene", dropna=True)
                    .agg(
                        Significant_Features=("Peptide", "nunique"),
                        Best_q_value=("q-value", "min"),
                    )
                    .reset_index()
                    .sort_values(["Significant_Features", "Best_q_value"], ascending=[False, True])
                    .head(20)
                )
                out["top_gene_hits"] = (
                    gene_long.sort_values(["q-value"], ascending=True)
                    [["Gene", "Peptide", "q-value"]]
                    .head(20)
                )

    return out


def _build_waterfall_top_table(diff_df: pd.DataFrame, ss) -> pd.DataFrame:
    if diff_df is None or not isinstance(diff_df, pd.DataFrame) or diff_df.empty or "log2FC" not in diff_df.columns:
        return pd.DataFrame()

    sig_col = ss.get("lfq_wfl_sig_col", None) or ss.get("waterfall_sig_col", None)
    if not sig_col or sig_col not in diff_df.columns:
        sig_col = "q-value" if "q-value" in diff_df.columns else ("p-value" if "p-value" in diff_df.columns else None)
    if not sig_col:
        return pd.DataFrame()

    alpha_thr = float(ss.get("lfq_wfl_alpha", ss.get("waterfall_alpha", 0.05)))
    fc_thr = float(ss.get("lfq_wfl_fc", ss.get("waterfall_logfc", 1.0)))
    top_n = int(ss.get("lfq_wfl_topn", ss.get("waterfall_topn", 5)))

    df = diff_df.copy()
    df["log2FC"] = pd.to_numeric(df["log2FC"], errors="coerce")
    df[sig_col] = pd.to_numeric(df[sig_col], errors="coerce")
    df = df.dropna(subset=["log2FC", sig_col]).sort_values("log2FC", ascending=False).reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()

    df["Rank"] = df.index + 1
    df["Status"] = "Not Significant"
    df.loc[(df["log2FC"] >= fc_thr) & (df[sig_col] <= alpha_thr), "Status"] = "Up regulated"
    df.loc[(df["log2FC"] <= -fc_thr) & (df[sig_col] <= alpha_thr), "Status"] = "Down regulated"

    top_up = df[df["Status"] == "Up regulated"].head(top_n)
    top_down = df[df["Status"] == "Down regulated"].sort_values("log2FC", ascending=True).head(top_n)
    out = pd.concat([top_up, top_down], ignore_index=True)
    if out.empty:
        return pd.DataFrame()

    out = out.copy()
    out["Point_ID"] = range(1, len(out) + 1)
    keep = [
        c for c in [
            "Point_ID", "Genes", "Peptide", "Glycan", "Glycan(H,N,A,F)", "GlycanClass", "ProSites",
            "log2FC", sig_col, "Status", "Rank"
        ] if c in out.columns
    ]
    return out[keep]


# =============================================================================
# MAIN PDF GENERATOR
# =============================================================================

def generate_final_pdf():
    ss = st.session_state
    pdf = AnalysisReport()
    pdf.set_auto_page_break(auto=True, margin=15)

    group_labels = ss.get("global_group_labels", []) or []
    control_idx = ss.get("global_control_group_idx", 0)
    ref_grp = group_labels[control_idx] if group_labels and control_idx < len(group_labels) else "N/A"

    merged_df = ss.get("merged_df", None)
    diff_df = ss.get("diffexp_df", None)
    diff_summary = _build_diff_summary(diff_df)

    is_anova = isinstance(diff_df, pd.DataFrame) and "log2FC" not in diff_df.columns
    # Name the exact test when the app recorded it (GlycoExplorer TMT does);
    # fall back to the generic description otherwise.
    stat_test = ss.get("stat_method_label") or ("Three-group ANOVA" if is_anova else "Pairwise comparison")
    batch_applied, batch_status, _ = _get_batch_status(ss)

    # -------------------------------------------------------------------------
    # 1. Executive Summary
    # -------------------------------------------------------------------------
    pdf.add_page()
    pdf.section_title("1. Executive Summary")

    pdf.section_text(
        "This report summarizes the glycopeptide analysis workflow, including sample grouping, "
        "quality-control visualizations, differential testing, and key glycopeptide/gene-level findings.",
        font_size=10,
    )

    summary_lines = [
        f"- Total glycopeptides in active matrix: {len(merged_df):,}" if isinstance(merged_df, pd.DataFrame) else "- Total glycopeptides in active matrix: N/A",
        f"- Differential result rows: {diff_summary.get('n_rows', 0):,}",
        f"- Significant features (q < 0.05): {diff_summary.get('n_sig', 0):,}",
        f"- Statistical analysis: {stat_test}",
        f"- Reference group: {ref_grp}",
        f"- Batch correction: {batch_status}",
    ]

    if not diff_summary.get("is_anova", True):
        summary_lines.extend([
            f"- Upregulated features: {diff_summary.get('n_up', 0):,}",
            f"- Downregulated features: {diff_summary.get('n_down', 0):,}",
        ])

    if isinstance(diff_summary.get("sig_gene_summary"), pd.DataFrame) and not diff_summary["sig_gene_summary"].empty:
        summary_lines.append(f"- Significant genes represented: {len(diff_summary['sig_gene_summary']):,}")

    pdf.add_kv_lines(summary_lines, font_size=10)

    # -------------------------------------------------------------------------
    # 2. Experimental Design & Parameters
    # -------------------------------------------------------------------------
    pdf.add_page()
    pdf.section_title("2. Experimental Design & Parameters")

    analysis_mode = "Label-Free Quantification (LFQ)" if ss.get("raw_merged_df", None) is not None else "TMT / Glycopeptide Quantification"
    params = [
        f"Analysis Mode: {analysis_mode}",
        "Species: Human",
        f"Statistical Test Used: {stat_test}",
        f"Global Reference Group: {ref_grp}",
        f"Number of Groups: {ss.get('global_n_groups', 'N/A')}",
        f"Batch Effect Correction: {batch_status}",
        f"Total Glycopeptides Analyzed: {len(merged_df) if isinstance(merged_df, pd.DataFrame) else 'N/A'}",
        f"Differential Result Rows: {len(diff_df) if isinstance(diff_df, pd.DataFrame) else 'N/A'}",
    ]
    pdf.add_kv_lines([f"- {p}" for p in params], font_size=10)

    pdf.subsection_title("Visualization Thresholds & Settings")
    thresholds = []
    
    # Volcano
    v_metric = ss.get("volcano_sig_col", ss.get("lfq_volcano_sig_col", "p-value"))
    v_sig = ss.get("volcano_sig_threshold", ss.get("lfq_volcano_sig_threshold", 0.05))
    v_fc = ss.get("volcano_logfc", ss.get("lfq_volcano_logfc", 1.0))
    v_topn = ss.get("volcano_topn", ss.get("lfq_volcano_topn", 10))
    v_mode = ss.get("volcano_topmode", ss.get("lfq_volcano_topmode", "up-down"))
    thresholds.append(f"Volcano Plot: {v_metric} < {v_sig}, |log2FC| > {v_fc}; top labels = {v_topn} ({v_mode})")

    # Heatmap
    h_norm = ss.get("heatmap_norm", ss.get("overview_hm_norm", "Z-score"))
    h_clust = ss.get("heatmap_cluster", "Hierarchical")
    h_topn = ss.get("heatmap_topn", ss.get("overview_hm_topn", 25))
    thresholds.append(f"Heatmap: Normalization = {h_norm}, Clustering = {h_clust}, Top N = {h_topn}")

    # Box Plot
    b_topn = ss.get("box_topn", 10)
    b_metric = ss.get("box_sig_col", "p-value")
    b_mode = ss.get("box_topmode", "up-down")
    if is_anova:
        thresholds.append(f"Box Plot: Top {b_topn} features by {b_metric}")
    else:
        thresholds.append(f"Box Plot: Top {b_topn} features by {b_metric} ({b_mode})")

    # Waterfall
    wf_sig = ss.get("lfq_wfl_sig_col", ss.get("waterfall_sig_col", "q-value"))
    wf_alpha = ss.get("lfq_wfl_alpha", ss.get("waterfall_alpha", 0.05))
    wf_fc = ss.get("lfq_wfl_fc", ss.get("waterfall_logfc", 1.0))
    wf_topn = ss.get("lfq_wfl_topn", ss.get("waterfall_topn", 5))
    thresholds.append(f"Waterfall: {wf_sig} < {wf_alpha}, |log2FC| > {wf_fc}; top N per direction = {wf_topn}")

    # Lollipop
    l_genes = ss.get("lollipop_gene_list", [])
    if l_genes:
        l_sig = ss.get("lollipop_sig_col", "q-value")
        l_leg = "Yes" if ss.get("lollipop_show_legend") else "No"
        thresholds.append(f"Lollipop Plot: Metric = {l_sig}, Gene(s) = {', '.join(l_genes)}, Shapes = {l_leg}")
        
    # Box Plot Gene
    box_genes = ss.get("box_gene_list", [])
    if box_genes:
        thresholds.append(f"Box Plot (Gene): Gene(s) = {', '.join(box_genes)}")

    pdf.add_kv_lines([f"- {t}" for t in thresholds], font_size=9)

    # -------------------------------------------------------------------------
    # 3. Sample Mapping
    # -------------------------------------------------------------------------
    mapping_df = _build_sample_mapping_df(ss)
    _add_sample_mapping_table(pdf, mapping_df, batch_applied=batch_applied)

    # -------------------------------------------------------------------------
    # Differential summaries and tables
    # -------------------------------------------------------------------------
    if isinstance(diff_df, pd.DataFrame) and not diff_df.empty:
        pdf.add_page()
        pdf.section_title("Differential Analysis Summary")
        pdf.section_text(
            "Differential analysis identifies glycopeptides that differ between experimental groups. "
            "P-values are adjusted for multiple testing when q-values are available.",
            font_size=10,
        )

        summary_lines = [
            f"- Total tested features: {diff_summary['n_rows']:,}",
            f"- Significant features (q < 0.05): {diff_summary['n_sig']:,}",
        ]
        if diff_summary["is_anova"]:
            summary_lines.append("- Result type: ANOVA/global group difference; pairwise direction is not shown by default.")
        else:
            summary_lines.append(f"- Upregulated features: {diff_summary['n_up']:,}")
            summary_lines.append(f"- Downregulated features: {diff_summary['n_down']:,}")
        pdf.add_kv_lines(summary_lines, font_size=10)

        if diff_summary["is_anova"]:
            if not diff_summary["top_sig"].empty:
                _add_dataframe_table(
                    pdf,
                    diff_summary["top_sig"],
                    title="Top Significant Features (ANOVA)",
                    max_rows=20,
                )
        else:
            if not diff_summary["top_up"].empty:
                _add_dataframe_table(
                    pdf,
                    diff_summary["top_up"],
                    title="Top Upregulated Glycopeptides",
                    max_rows=15,
                )
            if not diff_summary["top_down"].empty:
                _add_dataframe_table(
                    pdf,
                    diff_summary["top_down"],
                    title="Top Downregulated Glycopeptides",
                    max_rows=15,
                )

        waterfall_table = _build_waterfall_top_table(diff_df, ss)
        if not waterfall_table.empty:
            _add_dataframe_table(
                pdf,
                waterfall_table,
                title="Waterfall Top Features",
                max_rows=20,
                note="These are the numbered features that would appear in the waterfall plot table under the selected waterfall thresholds.",
            )

        if not diff_summary["sig_gene_summary"].empty:
            _add_dataframe_table(
                pdf,
                diff_summary["sig_gene_summary"],
                title="Gene-Level Summary",
                max_rows=20,
            )

        if not diff_summary["top_gene_hits"].empty:
            _add_dataframe_table(
                pdf,
                diff_summary["top_gene_hits"],
                title="Top Gene-Associated Significant Hits",
                max_rows=20,
            )

        important_cols = [
            c for c in [
                "Peptide", "Glycan", "Glycan(H,N,A,F)", "GlycanClass", "Genes", "ProSites",
                "log2FC", "p-value", "q-value", "adj.p-value", "Regulation"
            ] if c in diff_df.columns
        ]
        if important_cols:
            df_to_plot = diff_df[important_cols].copy()
            sort_by = [c for c in ["q-value", "adj.p-value", "p-value"] if c in df_to_plot.columns]
            if sort_by:
                df_to_plot = df_to_plot.sort_values(by=sort_by, ascending=True)
            _add_dataframe_table(
                pdf,
                df_to_plot,
                title="Top Differential Results",
                max_rows=25,
            )

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    report_figures = _collect_report_figures(ss)
    for item in report_figures:
        key = item["key"]
        title = item["title"]
        try:
            _add_figure_page(pdf, ss[key], title)
        except Exception as e:
            pdf.add_page()
            pdf.section_title(title)
            pdf.set_font("helvetica", "I", 10)
            pdf.set_text_color(200, 0, 0)
            pdf.multi_cell(0, 6, f"Error including figure '{key}': {e}")
            pdf.set_text_color(0)

    # -------------------------------------------------------------------------
    # Final notes
    # -------------------------------------------------------------------------
    pdf.add_page()
    pdf.section_title("Report Notes")
    pdf.section_text(
        "This report is generated from the plots and tables available in the current Streamlit session. "
        "If a plot is missing, generate or open that visualization in the app before creating the report again. "
        "All reported values should be reviewed with the original data and experimental design before publication-level interpretation.",
        font_size=9,
    )

    out = pdf.output(dest="S")
    if isinstance(out, str):
        return out.encode("latin-1")
    return bytes(out)