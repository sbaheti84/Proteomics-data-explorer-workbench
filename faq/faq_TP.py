#!/usr/bin/env python
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd


def faq_section():
    with st.expander("Help, Quick Start & FAQ", expanded=False):

        st.markdown("## Quick Start for Investigators")
        st.caption(
            "A simple guide for reviewing total protein abundance results without needing to understand every technical setting."
        )

        q1, q2, q3 = st.columns(3)

        with q1:
            st.markdown(
                '''
                ### 1. Upload Data
                Upload one total protein abundance matrix.

                Supported file types:
                - CSV
                - Excel XLSX

                The app expects protein identifiers plus raw quantitative columns containing terms such as `Abundance`, `Reporter`, or `Intensity`.
                '''
            )

        with q2:
            st.markdown(
                '''
                ### 2. Define Groups
                Go to **Data Filtering & Group Specification**.

                Select:
                - Active sample columns
                - Group labels
                - Control/reference group
                - Optional batch correction
                '''
            )

        with q3:
            st.markdown(
                '''
                ### 3. Review Results
                Check:

                - PCA and QC plots
                - Differential results
                - Volcano plot
                - MA plot
                - Heatmap
                - Waterfall plot
                - Protein profile plots
                - PDF/ZIP export
                '''
            )

        st.info(
            "Recommended workflow: upload data → define groups and reference group → run preprocessing/QC → run statistics → review volcano, MA, waterfall, heatmap, and protein profile plots → export the report."
        )

        st.markdown("### How to Interpret the Main Outputs")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown(
                '''
                #### PCA / QC Plots
                Use these first to check whether samples behave as expected.

                Good signs:
                - Replicates cluster together
                - No obvious outlier sample
                - Groups separate by biology, not batch
                '''
            )

            st.markdown(
                '''
                #### Volcano Plot
                Best for two-group comparisons.

                - X-axis = log2 fold-change
                - Y-axis = statistical significance
                - Red/blue points indicate significant up/down regulation
                - Annotation controls label top proteins
                '''
            )

        with c2:
            st.markdown(
                '''
                #### MA Plot
                Shows abundance-dependent changes.

                - X-axis = average intensity
                - Y-axis = log2 fold-change
                - Useful for checking intensity-related bias
                '''
            )

            st.markdown(
                '''
                #### Protein Profile / Box Plot
                Use this to validate individual protein findings.

                - Confirms sample-level abundance patterns
                - Shows group distribution
                - Helps identify outlier-driven results
                - Top profiles can be exported as multi-panel PDF grids
                '''
            )

        st.divider()

        faqs = pd.DataFrame([
            {
                "Category": "General",
                "Question": "What is the Total Proteomics app?",
                "Answer": (
                    "The Total Proteomics app analyzes total protein abundance matrices. It supports sample selection, "
                    "group assignment, optional batch correction, preprocessing, QC, differential testing, volcano plots, "
                    "MA plots, heatmaps, waterfall plots, protein profile plots, and final report/ZIP export."
                ),
            },
            {
                "Category": "General",
                "Question": "Is this app for TMT or label-free data?",
                "Answer": (
                    "This FAQ is for the Total Protein TMT-style app that expects a single protein abundance matrix with "
                    "raw abundance/reporter/intensity columns. The label-free total protein app is handled separately."
                ),
            },
            {
                "Category": "Input Data",
                "Question": "What kind of file do I need to upload?",
                "Answer": (
                    "Upload one total proteomics matrix as CSV or Excel XLSX. The matrix should contain protein or gene identifiers "
                    "and raw quantitative sample columns."
                ),
            },
            {
                "Category": "Input Data",
                "Question": "Which columns are required?",
                "Answer": (
                    "The app can use protein/gene identifier columns such as `Genes`, `GeneID`, `Description`, `Accession`, "
                    "or `Master Protein Accessions`. Quantitative sample columns should contain terms such as `Abundance`, "
                    "`Reporter`, or `Intensity`."
                ),
            },
            {
                "Category": "Input Data",
                "Question": "Why are my abundance columns not detected?",
                "Answer": (
                    "The app searches for columns containing `Abundance`, `Reporter`, or `Intensity`. It intentionally ignores "
                    "columns that look pre-normalized or non-quantitative, including names containing `Normalized`, `Grouped`, "
                    "`Scaled`, `S/N`, `Corr`, `Ratio`, `Count`, or `CV`. Check column names if expected samples are missing."
                ),
            },
            {
                "Category": "Input Data",
                "Question": "How are GeneID and RowID created?",
                "Answer": (
                    "If a gene-like column is available, the app uses it as `GeneID`. If an accession column is also present, "
                    "`RowID` is built from gene plus accession. Otherwise, the app falls back to accession or a generic feature ID."
                ),
            },
            {
                "Category": "Input Data",
                "Question": "What does the example dataset do?",
                "Answer": (
                    "The example dataset is a small total protein abundance matrix that can be saved from the sidebar and used to test "
                    "the workflow formatting, grouping, QC, statistics, and exports."
                ),
            },
            {
                "Category": "Sample Setup",
                "Question": "How do I define groups?",
                "Answer": (
                    "Use **Data Filtering & Group Specification**. Select the active sample columns, then assign samples to Group 1, "
                    "Group 2, and optionally Group 3. Set the global reference/control group before running differential analysis."
                ),
            },
            {
                "Category": "Sample Setup",
                "Question": "What is the reference group?",
                "Answer": (
                    "The reference group controls fold-change direction. For example, if Control is the reference group, positive "
                    "log2FC means the comparison/test group is higher than Control."
                ),
            },
            {
                "Category": "Sample Setup",
                "Question": "Can I analyze three groups?",
                "Answer": (
                    "Yes. If three groups are selected, the app performs one-way ANOVA. ANOVA tests whether any group differs, "
                    "but it does not provide a single pairwise log2FC direction like a two-group comparison."
                ),
            },
            {
                "Category": "Preprocessing",
                "Question": "What preprocessing options are available?",
                "Answer": (
                    "The app includes common total-protein preprocessing options such as treating zeros as missing, log2 transformation, "
                    "row-wise zero/NA replacement, presence filtering, and optional normalization/QC views depending on the selected panel."
                ),
            },
            {
                "Category": "Preprocessing",
                "Question": "What does Treat 0 as missing mean?",
                "Answer": (
                    "In mass spectrometry, zero often means the protein was not detected rather than truly absent. Treating zeros as missing "
                    "prevents them from artificially lowering group means and distorting variance."
                ),
            },
            {
                "Category": "Preprocessing",
                "Question": "Should I use log2 transformation?",
                "Answer": (
                    "Usually yes for abundance data. Log2 transformation stabilizes variance and makes fold-change interpretation easier. "
                    "For TMT-style total protein data, differential log2FC is calculated as test-group mean minus control-group mean on the processed scale."
                ),
            },
            {
                "Category": "Preprocessing",
                "Question": "How does zero replacement or imputation work?",
                "Answer": (
                    "When enabled, the app can replace zero or missing values using a small fraction of the minimum positive value in that protein row. "
                    "This helps avoid unstable fold-change and statistical calculations caused by zeros."
                ),
            },
            {
                "Category": "Preprocessing",
                "Question": "How does the presence filter work?",
                "Answer": (
                    "The presence filter removes sparse proteins before statistics. You can keep proteins detected across all samples, detected in at least "
                    "one group, or use OR/AND combinations of those rules."
                ),
            },
            {
                "Category": "Batch Correction",
                "Question": "When should I use batch correction?",
                "Answer": (
                    "Use batch correction when PCA or QC plots show separation by technical batch, TMT plex, preparation day, or instrument run rather than biology. "
                    "Avoid unnecessary correction when no batch effect is visible."
                ),
            },
            {
                "Category": "Batch Correction",
                "Question": "Which batch correction methods are available?",
                "Answer": (
                    "The app provides batch correction options including:\n\n"
                    "- **ComBat (Empirical Bayes)**\n"
                    "- **Median Centering**\n"
                    "- **Internal Reference Scaling (IRS)**\n\n"
                    "IRS is useful when each TMT batch/plex has a common pool/reference channel."
                ),
            },
            {
                "Category": "Batch Correction",
                "Question": "What is Internal Reference Scaling?",
                "Answer": (
                    "Internal Reference Scaling uses a common reference or pool channel within each batch to align batches. "
                    "Select the reference channel for each batch when using IRS."
                ),
            },
            {
                "Category": "Batch Correction",
                "Question": "What happens after batch correction?",
                "Answer": (
                    "After batch correction, downstream QC and statistics use the corrected abundance values. Differential results are reset so they can be re-run on the corrected matrix."
                ),
            },
            {
                "Category": "QC",
                "Question": "What QC modules are available?",
                "Answer": (
                    "The **Data Integrity Check & Sanity Profiling** section includes QC views such as PCA, density/distribution plots, "
                    "sample boxplots, missing-value summaries, correlation heatmaps, QC heatmaps, and dendrogram-style clustering views."
                ),
            },
            {
                "Category": "QC",
                "Question": "What does a good PCA look like?",
                "Answer": (
                    "Replicates should usually cluster together. If samples separate mostly by batch instead of biological group, "
                    "review sample mapping and consider batch correction."
                ),
            },
            {
                "Category": "QC",
                "Question": "How should I interpret correlation heatmaps?",
                "Answer": (
                    "Samples from the same biological group or replicate set should generally show higher correlation. A sample with low correlation to all others may be an outlier."
                ),
            },
            {
                "Category": "QC",
                "Question": "Why do some heatmaps show RowID?",
                "Answer": (
                    "`RowID` is the internal feature identifier. In updated plotting code, the RowID axis title should be removed while keeping useful protein/gene row labels."
                ),
            },
            {
                "Category": "Differential Analysis",
                "Question": "Which statistical tests are supported?",
                "Answer": (
                    "For two-group comparisons, the app supports:\n\n"
                    "- Student's t-test\n"
                    "- Welch's t-test\n"
                    "- Paired t-test\n"
                    "- Mann-Whitney U test\n"
                    "- Wilcoxon signed-rank test\n\n"
                    "For three groups, the app performs one-way ANOVA."
                ),
            },
            {
                "Category": "Differential Analysis",
                "Question": "What multiple-testing correction is used?",
                "Answer": (
                    "The app uses Benjamini-Hochberg false discovery rate correction and reports adjusted results as `q-value`."
                ),
            },
            {
                "Category": "Differential Analysis",
                "Question": "What thresholds should I use?",
                "Answer": (
                    "A common starting point is:\n\n"
                    "- `q-value < 0.05`\n"
                    "- `|log2FC| > 1`\n\n"
                    "For exploratory analysis, `q-value < 0.1` or `|log2FC| > 0.5` may be useful."
                ),
            },
            {
                "Category": "Differential Analysis",
                "Question": "How is log2FC calculated?",
                "Answer": (
                    "For pairwise TMT-style total protein analysis, log2FC is calculated as the test-group mean minus the control-group mean. "
                    "Axis labels use the selected comparison format, for example `log2 fold-change (Case/Control)`."
                ),
            },
            {
                "Category": "Differential Analysis",
                "Question": "What happens when I use ANOVA?",
                "Answer": (
                    "ANOVA tests whether any group differs across three groups. It does not produce a single pairwise direction, so pairwise-style "
                    "volcano, MA, and waterfall interpretation should be used mainly for two-group comparisons."
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "Which result visualizations are available?",
                "Answer": (
                    "The app provides Volcano, MA plot, Differential Heatmap, Waterfall plot, and Protein Profile plots. "
                    "Protein profile plots can be shown for selected genes/proteins and for top significant proteins."
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "How should I interpret the volcano plot?",
                "Answer": (
                    "The volcano plot shows log2 fold-change on the x-axis and statistical significance on the y-axis. "
                    "Proteins farther left or right have larger abundance changes; proteins higher up are more statistically significant."
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "How should I interpret the MA plot?",
                "Answer": (
                    "The MA plot shows average abundance on the x-axis and log2 fold-change on the y-axis. "
                    "It helps identify whether fold-change patterns depend on protein abundance level."
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "What is the total protein waterfall plot?",
                "Answer": (
                    "The waterfall plot ranks proteins by log2FC and labels top significant upregulated and downregulated proteins. "
                    "It is useful for PI-facing summaries and quick review of strongest changes."
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "Why is my waterfall plot empty?",
                "Answer": (
                    "Possible reasons include no proteins passing significance thresholds, overly strict fold-change cutoff, incorrect group assignment, "
                    "missing pairwise differential results, or differential analysis not being run yet."
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "How do top protein profile plots work?",
                "Answer": (
                    "Top protein profile plots show box/strip plots for selected significant proteins. In the updated app, the on-screen top profiles are combined "
                    "into one grid figure, so zoom opens the whole grid. PDF export saves the top profiles as multi-panel pages."
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "Why do significance stars in protein profiles come from the selected metric?",
                "Answer": (
                    "For top protein profiles, stars are drawn from the selected significance metric, such as p-value or q-value, so they match the same filter used "
                    "to select the top significant proteins."
                ),
            },
            {
                "Category": "Export",
                "Question": "What can I export?",
                "Answer": (
                    "The app can export uploaded/processed tables, differential results, static figures, interactive HTML plots, final PDF reports, and ZIP bundles when available."
                ),
            },
            {
                "Category": "Export",
                "Question": "What is included in the final report?",
                "Answer": (
                    "The PDF report includes available session results such as sample mapping, QC plots, differential tables, volcano/MA/heatmap/waterfall plots, "
                    "and generated protein profile figures."
                ),
            },
            {
                "Category": "Export",
                "Question": "Why are some plots missing from the PDF or ZIP?",
                "Answer": (
                    "Reports are session-driven. Only plots generated during the current session are included. Open or generate each desired plot before creating the final report."
                ),
            },
            {
                "Category": "Export",
                "Question": "Where are files saved?",
                "Answer": (
                    "Some buttons use native Streamlit downloads, while some desktop-safe save buttons write directly to the user's Downloads folder. "
                    "This is designed to work better in packaged Streamlit desktop applications."
                ),
            },
            {
                "Category": "Troubleshooting",
                "Question": "Why did analysis fail?",
                "Answer": (
                    "Common causes include missing abundance columns, non-numeric sample values, empty groups, too few samples per group, missing identifier columns, "
                    "or running downstream plots before differential analysis has completed."
                ),
            },
            {
                "Category": "Troubleshooting",
                "Question": "Why did batch correction fail?",
                "Answer": (
                    "Batch correction can fail if batches are empty, if too few samples are assigned, if samples are assigned to multiple batches, "
                    "or if ComBat is selected but the `neuroCombat` package is not installed."
                ),
            },
            {
                "Category": "Troubleshooting",
                "Question": "Why is the app slow with large files?",
                "Answer": (
                    "Large protein matrices can take time for preprocessing, clustering, heatmaps, and report generation. Filtering sparse rows and limiting top-N heatmaps "
                    "or top protein profiles can improve speed."
                ),
            },
        ])

        st.markdown("## Searchable FAQ")

        search_col, cat_col = st.columns([0.65, 0.35])

        with search_col:
            search = st.text_input(
                "Search FAQ",
                placeholder="Search: PCA, batch, volcano, MA plot, waterfall, protein profile, report...",
                key="tp_faq_search",
            )

        with cat_col:
            categories = ["All"] + sorted(faqs["Category"].unique().tolist())
            selected_category = st.selectbox(
                "Category",
                categories,
                key="tp_faq_category_filter",
            )

        filtered = faqs.copy()

        if selected_category != "All":
            filtered = filtered[filtered["Category"] == selected_category]

        if search:
            q = search.lower().strip()
            filtered = filtered[
                filtered["Category"].str.lower().str.contains(q, na=False)
                | filtered["Question"].str.lower().str.contains(q, na=False)
                | filtered["Answer"].str.lower().str.contains(q, na=False)
            ]

        st.caption(f"Showing {len(filtered)} FAQ item(s).")

        if filtered.empty:
            st.info("No matching FAQ entries found.")
            return

        category_order = [
            "General",
            "Input Data",
            "Sample Setup",
            "Preprocessing",
            "Batch Correction",
            "QC",
            "Differential Analysis",
            "Visualizations",
            "Export",
            "Troubleshooting",
        ]

        for category in category_order:
            cat_df = filtered[filtered["Category"] == category]
            if cat_df.empty:
                continue

            st.markdown(f"### {category}")
            for _, row in cat_df.iterrows():
                with st.expander(row["Question"], expanded=False):
                    st.markdown(row["Answer"])

        extra_categories = [c for c in filtered["Category"].unique() if c not in category_order]
        for category in extra_categories:
            cat_df = filtered[filtered["Category"] == category]
            st.markdown(f"### {category}")
            for _, row in cat_df.iterrows():
                with st.expander(row["Question"], expanded=False):
                    st.markdown(row["Answer"])
