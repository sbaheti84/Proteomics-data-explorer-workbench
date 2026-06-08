#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import os
import tempfile
from typing import List, Dict, Any

from fpdf import FPDF
import numpy as np
import pandas as pd
import streamlit as st


class AnalysisReport(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 16)
        self.set_text_color(0, 90, 156)
        self.cell(0, 10, "Total Proteomics | Analysis Summary Report", ln=True, align="C")

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

    def section_title(self, title):
        self.set_font("helvetica", "B", 14)
        self.set_text_color(0, 90, 156)
        self.cell(0, 10, title, ln=True)
        self.ln(2)

    def section_text(self, text, font_size=10):
        self.set_font("helvetica", "", font_size)
        self.set_text_color(0)
        self.multi_cell(0, 6, str(text))
        self.ln(1)

    def add_kv_lines(self, lines: List[str], font_size=10):
        self.set_font("helvetica", "", font_size)
        self.set_text_color(0)
        for line in lines:
            self.cell(0, 6, str(line), ln=True)
        self.ln(3)


def _clean_sample_label(name: str) -> str:
    s = str(name)
    s = s.replace("Abundance:", "").replace("Reporter", "").replace("Intensity", "").strip()
    if ":" in s:
        s = s.split(":", 1)[-1].strip()
    return s


def _build_sample_mapping_df(ss) -> pd.DataFrame:
    sample_cols = ss.get("global_sample_cols", []) or []
    group_samples = ss.get("global_group_samples", []) or []
    group_labels = ss.get("global_group_labels", []) or []
    control_idx = ss.get("global_control_group_idx", 0)
    batch_dict = ss.get("batch_dict", {}) or {}
    batch_enabled = bool(ss.get("enable_batch_correction", False) or ss.get("batch_corrected", False))

    rows = []
    assigned_samples = {c for cols in group_samples for c in (cols or [])}

    for col in sample_cols:
        if col not in assigned_samples:
            continue

        for i, cols in enumerate(group_samples):
            if col in (cols or []):
                batch_name = "N/A"
                if batch_enabled:
                    batch_name = "Unassigned"
                    for b_name, b_cols in batch_dict.items():
                        if col in (b_cols or []):
                            batch_name = b_name
                            break

                rows.append(
                    {
                        "Column": str(col),
                        "Sample": _clean_sample_label(col),
                        "Group": group_labels[i] if i < len(group_labels) else f"Group{i+1}",
                        "Group_Number": str(i + 1),
                        "Batch": batch_name,
                        "Is_Reference": "Yes" if i == control_idx else "No",
                    }
                )
                break

    return pd.DataFrame(rows)


def _add_sample_mapping_table(pdf: AnalysisReport, mapping_df: pd.DataFrame):
    pdf.add_page()
    pdf.section_title("2. Sample Mapping")

    if mapping_df.empty:
        pdf.section_text("No sample mapping information was available in session state.")
        return

    pdf.section_text(
        "This table shows the samples used in the active analysis, their assigned groups, reference-group status, "
        "and batch assignment when batch correction was configured.",
        font_size=9,
    )

    include_batch = "Batch" in mapping_df.columns and mapping_df["Batch"].astype(str).ne("N/A").any()
    if include_batch:
        col_names = ["Sample", "Group", "Group_Number", "Batch", "Is_Reference"]
        headers = ["Sample", "Group", "Group #", "Batch", "Reference"]
        widths = [70, 48, 18, 24, 25]
    else:
        col_names = ["Sample", "Group", "Group_Number", "Is_Reference"]
        headers = ["Sample", "Group", "Group #", "Reference"]
        widths = [85, 55, 20, 25]

    row_h = 7
    pdf.set_font("helvetica", "B", 9)
    pdf.set_fill_color(230, 240, 250)
    pdf.set_text_color(0)

    for header, width in zip(headers, widths):
        pdf.cell(width, row_h, header, border=1, align="C", fill=True)
    pdf.ln(row_h)

    pdf.set_font("helvetica", "", 8)

    for _, row in mapping_df[col_names].iterrows():
        if pdf.get_y() > 265:
            pdf.add_page()
            pdf.section_title("2. Sample Mapping (continued)")
            pdf.set_font("helvetica", "B", 9)
            pdf.set_fill_color(230, 240, 250)
            for header, width in zip(headers, widths):
                pdf.cell(width, row_h, header, border=1, align="C", fill=True)
            pdf.ln(row_h)
            pdf.set_font("helvetica", "", 8)

        values = [str(row[c]) for c in col_names]
        values[0] = values[0][:42] + "..." if len(values[0]) > 45 else values[0]
        values[1] = values[1][:24] + "..." if len(values[1]) > 27 else values[1]

        for value, width in zip(values, widths):
            pdf.cell(width, row_h, value, border=1)
        pdf.ln(row_h)

    pdf.ln(4)


def _row_height(pdf, data, col_widths, line_h=4):
    max_lines = 1
    for text, width in zip(data, col_widths):
        lines = pdf.multi_cell(width, line_h, str(text), split_only=True)
        max_lines = max(max_lines, len(lines))
    return max_lines * line_h


def _format_numeric_for_table(series: pd.Series, decimals: int = 4) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").map(
        lambda x: f"{x:.{decimals}f}" if pd.notnull(x) else ""
    )


def _add_dataframe_table(pdf, df, title, max_rows=25):
    pdf.add_page()
    pdf.section_title(title)

    if df is None or df.empty:
        pdf.section_text("No table data available.")
        return

    df = df.head(max_rows).copy()
    col_names = df.columns.tolist()

    for col in ["log2FC", "p-value", "q-value", "adj.p-value", "Mean_log2FC", "Best_q_value"]:
        if col in df.columns:
            df[col] = _format_numeric_for_table(df[col], decimals=4)

    col_width_map = {
        "RowID": 28,
        "GeneID": 20,
        "Genes": 20,
        "Gene": 20,
        "Accession": 25,
        "Master Protein Accessions": 28,
        "Description": 45,
        "log2FC": 15,
        "Mean_log2FC": 18,
        "p-value": 16,
        "q-value": 16,
        "adj.p-value": 16,
        "Best_q_value": 18,
        "Regulation": 25,
        "Significant_Proteins": 28,
        "Significant_Features": 28,
        "Upregulated_Hits": 24,
        "Downregulated_Hits": 26,
    }
    col_widths = [col_width_map.get(col, 20) for col in col_names]

    # Scale down if table is too wide.
    total_w = sum(col_widths)
    if total_w > 185:
        scale = 185 / total_w
        col_widths = [max(12, w * scale) for w in col_widths]

    header_h = 6
    line_h = 4

    pdf.set_font("helvetica", "B", 7)
    pdf.set_fill_color(230, 240, 250)

    for col, w in zip(col_names, col_widths):
        pdf.cell(w, header_h, str(col)[:24], border=1, align="C", fill=True)
    pdf.ln()

    pdf.set_font("helvetica", "", 6)

    for _, row in df.iterrows():
        row_data = [str(row[col]) for col in col_names]

        for idx, col in enumerate(col_names):
            if col in {"Description", "RowID", "Master Protein Accessions"} and len(row_data[idx]) > 55:
                row_data[idx] = row_data[idx][:52] + "..."

        h = _row_height(pdf, row_data, col_widths, line_h=line_h)

        if pdf.get_y() + h > 270:
            pdf.add_page()
            pdf.section_title(f"{title} (continued)")
            pdf.set_font("helvetica", "B", 7)
            pdf.set_fill_color(230, 240, 250)
            for col, w in zip(col_names, col_widths):
                pdf.cell(w, header_h, str(col)[:24], border=1, align="C", fill=True)
            pdf.ln()
            pdf.set_font("helvetica", "", 6)

        y0 = pdf.get_y()
        for text, w in zip(row_data, col_widths):
            x0 = pdf.get_x()
            pdf.multi_cell(w, line_h, text, border=1)
            pdf.set_xy(x0 + w, y0)
        pdf.ln(h)


def _save_fig_to_temp(fig, dpi=180) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
        temp_img_path = tmp_img.name
    fig.savefig(temp_img_path, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    return temp_img_path


def _interpretation_for_title(title: str) -> str:
    t = title.lower()
    if "pca" in t:
        return "PCA shows overall sample similarity. Replicates should cluster together; separation by batch rather than biology should be reviewed."
    if "correlation" in t:
        return "Higher correlation indicates stronger similarity between samples. Low-correlation samples may represent outliers or technical issues."
    if "density" in t:
        return "Density plots compare sample-level intensity distributions and help identify shifts caused by loading, normalization, or batch effects."
    if "missing" in t:
        return "Missing-value plots show detection completeness across samples. High missingness may reduce statistical power."
    if "heatmap" in t:
        return "Heatmaps show patterns of protein abundance across samples. Clusters can reveal biological grouping, outliers, or batch structure."
    if "dendrogram" in t or "clustering" in t:
        return "Hierarchical clustering shows sample relatedness based on total protein profiles."
    if "volcano" in t:
        return "Volcano plots combine fold-change and significance. Proteins farther left or right have larger abundance changes; higher points are more significant."
    if "ma plot" in t or "ma " in t:
        return "MA plots show fold-change against average abundance and help identify abundance-dependent bias."
    if "waterfall" in t:
        return "Waterfall plots rank proteins by fold-change and highlight the strongest significant up- and down-regulated proteins."
    if "box" in t or "profile" in t:
        return "Box/profile plots show abundance distributions for selected proteins and help verify whether a result is consistent across samples."
    if "log2 fold" in t:
        return "The log2 fold-change distribution summarizes the global direction and magnitude of protein abundance changes."
    return ""


def _add_figure_page(pdf: AnalysisReport, fig, title: str, subtitle: str = None):
    pdf.add_page()
    pdf.section_title(title)

    interpretation = _interpretation_for_title(title)
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
    figure_specs = [
        ("report_pca_no_ellipse_fig", "3a. Principal Component Analysis (PCA) - No Ellipse"),
        ("report_pca_ellipse_fig", "3b. Principal Component Analysis (PCA) - With Ellipse"),
        ("report_pca_fig", "3. Principal Component Analysis (PCA)"),
        ("report_corr_fig", "4. Sample Correlation Heatmap"),
        ("report_density_raw_fig", "5a. Sample Density (Raw Un-transformed)"),
        ("report_density_proc_fig", "5b. Sample Density (Processed)"),
        ("report_qc_boxplot_fig", "6. QC Sample Boxplot"),
        ("report_qc_missing_fig", "7. Missing Values Distribution"),
        ("report_qc_heatmap_fig", "8. QC Expression Heatmap"),
        ("report_dendrogram_fig", "9. Hierarchical Clustering Dendrogram"),
        ("report_volcano_fig", "10. Differential Expression Volcano Plot"),
        ("report_ma_fig", "11. MA Plot"),
        ("report_log2fc_density_fig", "12. Log2 Fold Change Distribution"),
        ("report_heatmap_fig", "13. Clustered Heatmap of Top Differential Hits"),
        ("report_waterfall_fig", "14. Total Protein Waterfall Plot"),
        ("report_boxplot_fig", "15. Protein Profile Box Plot"),
    ]

    found = []
    seen_titles = set()
    for key, title in figure_specs:
        if key in ss and ss[key] is not None:
            if title not in seen_titles:
                found.append({"key": key, "title": title})
                seen_titles.add(title)
    return found


def _recalculate_regulation(df: pd.DataFrame, metric: str, alpha: float, fc: float) -> pd.DataFrame:
    df = df.copy()
    if metric not in df.columns:
        return df

    is_anova = "log2FC" not in df.columns
    df["Regulation"] = "Not Significant"
    df[metric] = pd.to_numeric(df[metric], errors="coerce")

    if not is_anova:
        df["log2FC"] = pd.to_numeric(df["log2FC"], errors="coerce")
        df.loc[(df[metric] <= alpha) & (df["log2FC"] >= fc), "Regulation"] = "Upregulated"
        df.loc[(df[metric] <= alpha) & (df["log2FC"] <= -fc), "Regulation"] = "Downregulated"
    else:
        df.loc[df[metric] <= alpha, "Regulation"] = "Significant"

    return df


def _best_metric(df: pd.DataFrame) -> str:
    for col in ["q-value", "adj.p-value", "p-value"]:
        if isinstance(df, pd.DataFrame) and col in df.columns:
            return col
    return "p-value"


def _build_diff_summary(diff_df: pd.DataFrame, metric: str) -> dict:
    out = {
        "is_anova": True,
        "n_rows": 0,
        "n_sig": 0,
        "n_up": 0,
        "n_down": 0,
        "top_up": pd.DataFrame(),
        "top_down": pd.DataFrame(),
        "top_sig": pd.DataFrame(),
        "protein_summary": pd.DataFrame(),
    }

    if diff_df is None or not isinstance(diff_df, pd.DataFrame) or diff_df.empty:
        return out

    df = diff_df.copy()
    out["n_rows"] = len(df)
    out["is_anova"] = "log2FC" not in df.columns

    if "Regulation" in df.columns:
        sig_df = df[df["Regulation"] != "Not Significant"].copy()
    elif metric in df.columns:
        sig_df = df[pd.to_numeric(df[metric], errors="coerce") <= 0.05].copy()
    else:
        sig_df = pd.DataFrame()

    out["n_sig"] = len(sig_df)

    pairwise_cols = [
        c for c in [
            "RowID", "GeneID", "Genes", "Accession", "Master Protein Accessions", "Description",
            "log2FC", "p-value", "q-value", "adj.p-value", "Regulation"
        ] if c in df.columns
    ]

    if not out["is_anova"]:
        out["n_up"] = int((df.get("Regulation", pd.Series(dtype=str)) == "Upregulated").sum())
        out["n_down"] = int((df.get("Regulation", pd.Series(dtype=str)) == "Downregulated").sum())

        up_df = df[df["Regulation"] == "Upregulated"].copy() if "Regulation" in df.columns else pd.DataFrame()
        down_df = df[df["Regulation"] == "Downregulated"].copy() if "Regulation" in df.columns else pd.DataFrame()

        sort_metric = metric if metric in df.columns else _best_metric(df)
        if not up_df.empty:
            up_df[sort_metric] = pd.to_numeric(up_df[sort_metric], errors="coerce")
            up_df["_abs_fc"] = pd.to_numeric(up_df.get("log2FC", np.nan), errors="coerce").abs()
            out["top_up"] = (
                up_df.sort_values([sort_metric, "_abs_fc"], ascending=[True, False])
                .drop(columns="_abs_fc", errors="ignore")[pairwise_cols]
                .head(20)
            )

        if not down_df.empty:
            down_df[sort_metric] = pd.to_numeric(down_df[sort_metric], errors="coerce")
            down_df["_abs_fc"] = pd.to_numeric(down_df.get("log2FC", np.nan), errors="coerce").abs()
            out["top_down"] = (
                down_df.sort_values([sort_metric, "_abs_fc"], ascending=[True, False])
                .drop(columns="_abs_fc", errors="ignore")[pairwise_cols]
                .head(20)
            )
    else:
        anova_cols = [
            c for c in [
                "RowID", "GeneID", "Genes", "Accession", "Master Protein Accessions", "Description",
                "p-value", "q-value", "adj.p-value", "Regulation"
            ] if c in df.columns
        ]
        if not sig_df.empty:
            sort_metric = metric if metric in sig_df.columns else _best_metric(sig_df)
            sig_df[sort_metric] = pd.to_numeric(sig_df[sort_metric], errors="coerce")
            out["top_sig"] = sig_df.sort_values(sort_metric, ascending=True)[anova_cols].head(25)

    gene_col = next((c for c in ["GeneID", "Genes", "Gene"] if c in sig_df.columns), None)
    if gene_col and not sig_df.empty:
        tmp = sig_df.copy()
        tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce") if metric in tmp.columns else np.nan
        agg = tmp.groupby(gene_col, dropna=True).agg(
            Significant_Proteins=(gene_col, "size"),
            Best_q_value=(metric, "min") if metric in tmp.columns else (gene_col, "size"),
        ).reset_index().rename(columns={gene_col: "Gene"})
        if "log2FC" in tmp.columns:
            fc_map = tmp.groupby(gene_col)["log2FC"].mean().to_dict()
            agg["Mean_log2FC"] = agg["Gene"].map(fc_map)
        out["protein_summary"] = agg.sort_values(["Significant_Proteins", "Best_q_value"], ascending=[False, True]).head(20)

    return out


def generate_final_pdf():
    ss = st.session_state
    pdf = AnalysisReport()
    pdf.set_auto_page_break(auto=True, margin=15)

    merged_df = ss.get("merged_df", None)
    if merged_df is None:
        merged_df = ss.get("raw_df", None)

    diff_df_raw = ss.get("diffexp_df", None)
    if diff_df_raw is None:
        diff_df_raw = ss.get("diff_results", None)

    metric_default = _best_metric(diff_df_raw) if isinstance(diff_df_raw, pd.DataFrame) else "q-value"
    metric = ss.get("filter_metric", metric_default)
    alpha_thr = ss.get("filter_alpha", 0.05)
    fc_thresh = ss.get("filter_fc", 1.0)

    diff_df = _recalculate_regulation(diff_df_raw, metric, alpha_thr, fc_thresh) if isinstance(diff_df_raw, pd.DataFrame) else None
    diff_summary = _build_diff_summary(diff_df, metric)

    group_labels = ss.get("global_group_labels", [])
    control_idx = ss.get("global_control_group_idx", 0)
    ref_grp = group_labels[control_idx] if group_labels and control_idx < len(group_labels) else "N/A"
    is_anova = isinstance(diff_df, pd.DataFrame) and "log2FC" not in diff_df.columns
    stat_test = "Three-group ANOVA" if is_anova else "Pairwise differential test"

    # ---------------------------------------------------------
    # 1. Experimental design
    # ---------------------------------------------------------
    pdf.add_page()
    pdf.section_title("1. Experimental Design & Parameters")

    batch_method = ss.get("batch_method_ui", ss.get("batch_method_ui_sel", "None"))
    batch_status = "Applied/configured" if ss.get("enable_batch_correction", False) else "Not Applied"
    if batch_status != "Not Applied" and batch_method:
        batch_status = f"{batch_status} ({batch_method})"

    params = [
        f"Analysis Type: Total Proteomics",
        f"Statistical Test Used: {stat_test}",
        f"Global Reference Group: {ref_grp}",
        f"Number of Groups: {ss.get('global_n_groups', 'N/A')}",
        f"Batch Effect Correction: {batch_status}",
        f"Total Proteins Analyzed: {len(merged_df) if isinstance(merged_df, pd.DataFrame) else 'N/A'}",
        f"Differential Result Rows: {len(diff_df) if isinstance(diff_df, pd.DataFrame) else 'N/A'}",
    ]
    pdf.add_kv_lines([f"- {p}" for p in params], font_size=10)

    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(0, 90, 156)
    pdf.cell(0, 8, "Data Visualization Thresholds & Settings", ln=True)
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(0)

    thresholds = [f"Global Significance Cutoff ({metric}): <= {alpha_thr}"]
    if not is_anova:
        thresholds.append(f"Global Fold-Change Cutoff (|log2FC|): >= {fc_thresh}")
    thresholds.append(f"Heatmap Configuration: Normalization = {ss.get('heatmap_norm', 'Z-score')}, Clustering = {ss.get('heatmap_cluster', 'Hierarchical')}")
    pdf.add_kv_lines([f"- {t}" for t in thresholds], font_size=10)

    # ---------------------------------------------------------
    # Executive summary
    # ---------------------------------------------------------
    pdf.add_page()
    pdf.section_title("Executive Summary")
    pdf.section_text(
        "This report summarizes total protein abundance differences across experimental groups. "
        "The summary below highlights statistically significant proteins using the thresholds selected in the application.",
        font_size=10,
    )

    summary_lines = [
        f"Total tested proteins: {diff_summary['n_rows']:,}",
        f"Significant proteins ({metric} <= {alpha_thr}): {diff_summary['n_sig']:,}",
    ]
    if not diff_summary["is_anova"]:
        summary_lines.extend([
            f"Upregulated proteins: {diff_summary['n_up']:,}",
            f"Downregulated proteins: {diff_summary['n_down']:,}",
        ])
    pdf.add_kv_lines([f"- {x}" for x in summary_lines], font_size=10)

    # ---------------------------------------------------------
    # Tables
    # ---------------------------------------------------------
    mapping_df = _build_sample_mapping_df(ss)
    _add_sample_mapping_table(pdf, mapping_df)

    if isinstance(diff_df, pd.DataFrame) and not diff_df.empty:
        if diff_summary["is_anova"]:
            if not diff_summary["top_sig"].empty:
                _add_dataframe_table(pdf, diff_summary["top_sig"], title="Top Significant Proteins (ANOVA)", max_rows=25)
        else:
            if not diff_summary["top_up"].empty:
                _add_dataframe_table(pdf, diff_summary["top_up"], title="Top Upregulated Proteins", max_rows=20)
            if not diff_summary["top_down"].empty:
                _add_dataframe_table(pdf, diff_summary["top_down"], title="Top Downregulated Proteins", max_rows=20)

        if not diff_summary["protein_summary"].empty:
            _add_dataframe_table(pdf, diff_summary["protein_summary"], title="Gene-Level Protein Summary", max_rows=20)

        important_cols = [
            c for c in [
                "RowID", "GeneID", "Genes", "Accession", "Master Protein Accessions", "Description",
                "log2FC", "p-value", "q-value", "adj.p-value", "Regulation"
            ] if c in diff_df.columns
        ]
        if important_cols:
            df_to_plot = diff_df[important_cols].copy()
            sort_by = []
            ascending = []
            if metric in df_to_plot.columns:
                df_to_plot[metric] = pd.to_numeric(df_to_plot[metric], errors="coerce")
                sort_by.append(metric)
                ascending.append(True)
            if "log2FC" in df_to_plot.columns:
                df_to_plot["_abs_fc"] = pd.to_numeric(df_to_plot["log2FC"], errors="coerce").abs()
                sort_by.append("_abs_fc")
                ascending.append(False)
            if sort_by:
                df_to_plot = df_to_plot.sort_values(by=sort_by, ascending=ascending)
            df_to_plot = df_to_plot.drop(columns="_abs_fc", errors="ignore")
            _add_dataframe_table(pdf, df_to_plot, title="Top Overall Differential Results", max_rows=30)

    # ---------------------------------------------------------
    # Figures
    # ---------------------------------------------------------
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

    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    return pdf_bytes
