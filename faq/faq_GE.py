#!/usr/bin/env python
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd


def faq_section():
    with st.expander("Help, Quick Start & FAQ", expanded=False):

        # -----------------------------
        # PI-FRIENDLY QUICK START
        # -----------------------------
        st.markdown("## Quick Start for Investigators")
        st.caption(
            "A simple guide for reviewing glycopeptide results without needing to understand every technical setting."
        )

        q1, q2, q3 = st.columns(3)

        with q1:
            st.markdown(
                """
                ### 1. Upload Data
                Select the pipeline engine in the sidebar and upload either:

                **pGlyco + Proteome Discoverer**
                - pGlyco identification file
                - Proteome Discoverer abundance file

                **or MSFragger TMT**
                - `psm.tsv`
                - `peptide.tsv`

                Also confirm the **species** and glycan image directory.
                """
            )

        with q2:
            st.markdown(
                """
                ### 2. Define Groups
                Go to **Data Filtering & Group Specification**.

                Select:
                - Sample/abundance columns
                - Group labels
                - Control/reference group
                - Optional batch correction

                Only assigned samples are used downstream.
                """
            )

        with q3:
            st.markdown(
                """
                ### 3. Review Results
                Check:

                - PCA
                - Correlation matrix
                - Sample clustering heatmap
                - Glycan classification summary
                - Differential results
                - Volcano, waterfall, heatmap, box, and lollipop plots
                - PDF/ZIP export
                """
            )

        st.info(
            "Recommended workflow: upload and process files, define groups, review QC, run statistics, apply filters, then inspect volcano, waterfall, heatmap, box, and lollipop plots before exporting the PDF/ZIP report."
        )

        st.markdown("### How to Interpret the Main Outputs")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown(
                """
                #### PCA / Correlation Matrix
                Use these first to check whether samples behave as expected.

                Good signs:
                - Replicates cluster together
                - No obvious outlier sample
                - Groups separate by biology rather than batch
                """
            )

            st.markdown(
                """
                #### Volcano Plot
                Best for two-group comparisons.

                - Left/right = fold-change direction
                - Higher = more statistically significant
                - Toggle between Static and Interactive modes for exploration.
                """
            )

        with c2:
            st.markdown(
                """
                #### Waterfall Plot
                Best for quickly seeing the strongest glycopeptide changes.

                It shows:
                - Upregulated glycopeptides
                - Downregulated glycopeptides
                - Ranked features with a numbered table
                - Glycan/site information when available
                """
            )

            st.markdown(
                """
                #### Box / Lollipop Plots
                Use these to validate individual findings.

                - Box plot: sample-level abundance pattern (can filter by up/down hits).
                - Box plot (Gene): selected gene-level feature review
                - Lollipop: site-level glycosylation changes across a protein
                """
            )

        st.divider()

        # -----------------------------
        # FAQ DATA
        # -----------------------------
        faqs = pd.DataFrame([
            {
                "Category": "General",
                "Question": "What is GlycoExplorer?",
                "Answer": (
                    "GlycoExplorer is a Streamlit tool for integrating glycopeptide identification results "
                    "with quantitative TMT abundance data. It supports file merging, species-aware glycan parsing, "
                    "sample grouping, optional batch correction, QC, differential analysis, glycan visualization, "
                    "waterfall plots, box plots, lollipop plots, and PDF/ZIP export."
                ),
            },
            {
                "Category": "General",
                "Question": "Which workflows are supported?",
                "Answer": (
                    "This GlycoExplorer app supports two TMT workflows:\n\n"
                    "1. **pGlyco + Proteome Discoverer**\n"
                    "2. **MSFragger TMT** using `psm.tsv` and `peptide.tsv`\n\n"
                    "Label-free total protein workflows should be handled in the separate LFQ module."
                ),
            },
            {
                "Category": "General",
                "Question": "Why do I need to select species?",
                "Answer": (
                    "The species selection controls the expected pGlyco glycan composition column. "
                    "For Human data, the app expects `Glycan(H,N,A,F)`. For Mouse data, the app expects "
                    "`Glycan(H,N,A,G,F)`."
                ),
            },
            {
                "Category": "General",
                "Question": "What is the glycan image directory?",
                "Answer": (
                    "The app uses the glycan image directory to display glycan structures in plots and tables. "
                    "If the folder path is missing or incorrect, the app will stop and ask for a valid folder path. "
                    "If a specific glycan image is not available, some plots may fall back to text composition labels."
                ),
            },
            {
                "Category": "Input Files",
                "Question": "What files do I need for pGlyco + Proteome Discoverer?",
                "Answer": (
                    "**pGlyco file:** requires `Peptide`, `Genes`, `Proteins`, `ProSites`, `RawName`, `Scan`, "
                    "and the species-specific glycan column:\n\n"
                    "- Human: `Glycan(H,N,A,F)`\n"
                    "- Mouse: `Glycan(H,N,A,G,F)`\n\n"
                    "**Proteome Discoverer file:** requires `Spectrum File`, `First Scan`, and TMT abundance columns."
                ),
            },
            {
                "Category": "Input Files",
                "Question": "What files do I need for MSFragger TMT?",
                "Answer": (
                    "Upload both:\n\n"
                    "1. **psm.tsv** with `Total Glycan Composition`, `Assigned Modifications`, a protein start column "
                    "such as `Protein Start` or `Start`, and TMT `Intensity <channel>` columns.\n"
                    "2. **peptide.tsv** with `Peptide`, `Protein ID`, and `Gene`."
                ),
            },
            {
                "Category": "Input Files",
                "Question": "Why are my abundance columns not detected?",
                "Answer": (
                    "For Proteome Discoverer files, the app looks for columns beginning with `Abundance:` or `Abundances:` "
                    "followed by a numeric TMT channel/sample label. Normalized, scaled, S/N, corrected, ratio, count, "
                    "and CV-like columns are ignored.\n\n"
                    "For MSFragger TMT, the app expects columns starting with `Intensity`, excluding `Intensity NA` columns."
                ),
            },
            {
                "Category": "Merging",
                "Question": "How does the pGlyco + PD merge work?",
                "Answer": (
                    "The app matches pGlyco IDs to Proteome Discoverer abundance values using scan-level information:\n\n"
                    "- pGlyco: `RawName` and `Scan`\n"
                    "- Proteome Discoverer: `Spectrum File` and `First Scan`\n\n"
                    "The matched data are then summarized into an analysis-ready glycopeptide abundance matrix."
                ),
            },
            {
                "Category": "Merging",
                "Question": "How does the MSFragger TMT workflow work?",
                "Answer": (
                    "The app reads `psm.tsv`, parses glycan compositions, estimates ProSites from modified N-sites, "
                    "detects TMT intensity channels, and joins peptide-level metadata from `peptide.tsv`. "
                    "It then builds a glycoform-level abundance table for downstream analysis."
                ),
            },
            {
                "Category": "Merging",
                "Question": "Why did the merged table have fewer rows than expected?",
                "Answer": (
                    "Common reasons:\n\n"
                    "- Raw file names do not match\n"
                    "- Scan numbers do not match\n"
                    "- Some scans are missing from the Proteome Discoverer output\n"
                    "- Duplicate glycopeptide IDs were collapsed or summarized\n"
                    "- Abundance/intensity columns were not detected\n"
                    "- MSFragger rows did not contain valid glycan composition information"
                ),
            },
            {
                "Category": "Sample Setup",
                "Question": "How do I define groups?",
                "Answer": (
                    "Use **Data Filtering & Group Specification**. Select the sample/abundance columns, then assign samples "
                    "to Group 1, Group 2, or Group 3. Only assigned samples are used downstream."
                ),
            },
            {
                "Category": "Sample Setup",
                "Question": "What is the reference group?",
                "Answer": (
                    "The reference group controls fold-change direction. For example, if Control is the reference, "
                    "positive log2FC means higher abundance in the comparison group relative to Control."
                ),
            },
            {
                "Category": "Sample Setup",
                "Question": "What happens if samples are unassigned?",
                "Answer": (
                    "Unassigned samples are not used in the downstream grouped analysis. The sample mapping summary shows "
                    "which samples are assigned, unassigned, assigned to batches, and marked as the reference group."
                ),
            },
            {
                "Category": "Batch Correction",
                "Question": "When should I use batch correction?",
                "Answer": (
                    "Use batch correction when samples were processed across different TMT plexes, preparation days, "
                    "instrument runs, or acquisition batches and PCA/correlation suggests that batch is stronger than biology."
                ),
            },
            {
                "Category": "Batch Correction",
                "Question": "Which batch correction methods are available?",
                "Answer": (
                    "The app supports:\n\n"
                    "- **ComBat (Empirical Bayes)**: useful for multi-batch correction when assumptions are met.\n"
                    "- **Median Centering**: a simpler global centering approach.\n"
                    "- **Internal Reference Scaling (IRS)**: useful when each batch/plex contains a reference or pooled channel."
                ),
            },
            {
                "Category": "Batch Correction",
                "Question": "How do I enable batch correction?",
                "Answer": (
                    "Turn on **Apply Batch Effect Correction**, select the method, choose the number of batches, and assign "
                    "each grouped sample to exactly one batch. For IRS, select the reference/pool sample for each batch."
                ),
            },
            {
                "Category": "Batch Correction",
                "Question": "What happens if batch correction is turned off?",
                "Answer": (
                    "The app uses the original abundance values. This is recommended when all samples were processed together "
                    "or no batch effect is visible."
                ),
            },
            {
                "Category": "QC",
                "Question": "What QC modules are available?",
                "Answer": (
                    "The **Data Integrity Check & Sanity Profiling** section currently includes:\n\n"
                    "- Principal Component Analysis (PCA)\n"
                    "- Correlation Matrix\n"
                    "- Sample Clustering\n"
                    "- Glycan Classification Summary"
                ),
            },
            {
                "Category": "QC",
                "Question": "What does a good PCA look like?",
                "Answer": (
                    "Replicates should usually cluster together. Strong separation by batch instead of group suggests "
                    "a technical effect that should be reviewed before interpreting biological results."
                ),
            },
            {
                "Category": "QC",
                "Question": "How should I use the sample clustering heatmap?",
                "Answer": (
                    "The sample clustering heatmap shows the top most-variable features across selected samples. "
                    "Use Z-score scaling to compare relative patterns across samples, or raw abundance to inspect magnitude."
                ),
            },
            {
                "Category": "QC",
                "Question": "What does the glycan classification summary show?",
                "Answer": (
                    "The glycan classification summary groups features by glycan class and helps identify whether specific "
                    "glycan classes dominate the dataset or differ across groups."
                ),
            },
            {
                "Category": "Differential Analysis",
                "Question": "Which statistical test should I use?",
                "Answer": (
                    "For two groups, the app supports Student's t-test, Welch's t-test, paired t-test, Mann-Whitney U, "
                    "and Wilcoxon signed-rank. Welch's t-test is often safer when variances differ. "
                    "For three groups, the app uses ANOVA."
                ),
            },
            {
                "Category": "Differential Analysis",
                "Question": "What happens when I select 3 groups?",
                "Answer": (
                    "The app performs three-group ANOVA. ANOVA detects whether any group differs, but it does not provide "
                    "one simple pairwise log2FC direction. Volcano and waterfall plots are mainly for two-group comparisons, "
                    "so ANOVA mode focuses on box plots, heatmaps, gene-level box plots, and lollipop plots."
                ),
            },
            {
                "Category": "Differential Analysis",
                "Question": "What thresholds should I use?",
                "Answer": (
                    "Common starting thresholds are:\n\n"
                    "- q-value < 0.05\n"
                    "- absolute log2FC > 1\n\n"
                    "For exploratory review, q-value < 0.1 or absolute log2FC > 0.5 can be useful."
                ),
            },
            {
                "Category": "Differential Analysis",
                "Question": "What does Apply Filters & View Subset do?",
                "Answer": (
                    "This filters the differential results table using the selected significance metric and cutoff. "
                    "For pairwise analysis it also applies the absolute log2FC cutoff. For ANOVA, fold-change filtering "
                    "is not applicable."
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "Which result visualization modules are available?",
                "Answer": (
                    "For pairwise comparisons, the app provides Volcano, Waterfall, Box plot, Heatmap, Box plot (Gene), "
                    "and Lollipop plot. For ANOVA, the app provides Box plot, Heatmap, Box plot (Gene), and Lollipop plot."
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "How should I interpret the volcano plot?",
                "Answer": (
                    "The volcano plot shows log2 fold-change on the x-axis and significance on the y-axis. "
                    "Points farther left or right have larger abundance changes. Points higher up are more significant. "
                    "You can view it as a Static plot or an Interactive plot for data exploration."
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "How does volcano annotation work?",
                "Answer": (
                    "Use the annotation slider to control how many top markers are labeled. The top-hit option can highlight "
                    "all top markers, upregulated markers, downregulated markers, or both up and down markers."
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "What is the waterfall plot?",
                "Answer": (
                    "The waterfall plot ranks glycopeptides by log2FC and highlights significantly up- and down-regulated features. "
                    "It includes a numbered table with gene, glycan, glycosite, log2FC, and p/q-value information."
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "Why is my waterfall plot empty?",
                "Answer": (
                    "Possible reasons:\n\n"
                    "- No features pass the significance threshold\n"
                    "- Fold-change cutoff is too strict\n"
                    "- Group assignment is incorrect\n"
                    "- Differential analysis has not been run\n"
                    "- The current analysis is ANOVA, where waterfall is not shown"
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "Why are glycan images missing?",
                "Answer": (
                    "Glycan images will only appear if the image directory exists and filenames match the glycan composition "
                    "used by the app. Confirm the glycan image directory in the sidebar."
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "How should I interpret box plots?",
                "Answer": (
                    "Box plots show abundance distributions across groups for selected features. They help confirm whether "
                    "a result is consistent across samples or driven by an outlier. You can also filter the displayed top hits by direction (up/down)."
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "How should I interpret the lollipop plot?",
                "Answer": (
                    "The lollipop plot maps glycosylation changes to protein site positions. It helps identify site-specific "
                    "glycosylation changes within a gene. The option to distinguish glycans with shapes and legend can help "
                    "separate different glycan compositions at the same or nearby sites."
                ),
            },
            {
                "Category": "MSFragger",
                "Question": "How are MSFragger glycan compositions parsed?",
                "Answer": (
                    "The app reads `Total Glycan Composition`, extracts Hex, HexNAc, NeuAc, and Fuc counts, and converts them "
                    "into the same general glycan composition format used by the pGlyco workflow."
                ),
            },
            {
                "Category": "MSFragger",
                "Question": "How are ProSites estimated for MSFragger?",
                "Answer": (
                    "The app estimates ProSites using the protein start position plus modified N-site positions from "
                    "`Assigned Modifications`. Large modification masses are treated as likely glycan modifications."
                ),
            },
            {
                "Category": "Export",
                "Question": "What is included in the PDF report?",
                "Answer": (
                    "The report includes available session results such as sample mapping, QC plots, differential tables, "
                    "volcano plots, heatmaps, box plots, waterfall plots, lollipop plots, and selected feature plots when they "
                    "were generated during the session."
                ),
            },
            {
                "Category": "Export",
                "Question": "What is included in the ZIP bundle?",
                "Answer": (
                    "The ZIP bundle includes generated report content and session outputs such as tables and plots. "
                    "The exact content depends on which analyses and visualizations were run before export."
                ),
            },
            {
                "Category": "Export",
                "Question": "Why are some plots missing from the PDF or ZIP?",
                "Answer": (
                    "Exports are session-driven. Only plots that were generated during the current session are included. "
                    "Open or run the plots before generating the report."
                ),
            },
            {
                "Category": "Troubleshooting",
                "Question": "Why did report generation fail?",
                "Answer": (
                    "Report generation can fail if a required plot, table, or column is missing. Re-run the relevant plots "
                    "and confirm that differential analysis completed successfully."
                ),
            },
            {
                "Category": "Troubleshooting",
                "Question": "Why is the app slow with large files?",
                "Answer": (
                    "Large glycoproteomics tables can take time to merge, filter, plot, and export. Use filtered input files, "
                    "remove unnecessary columns, and generate targeted plots when possible."
                ),
            },
            {
                "Category": "Troubleshooting",
                "Question": "Why did batch correction fail?",
                "Answer": (
                    "Common reasons include too few samples per batch, unassigned samples, duplicated sample assignments, "
                    "missing IRS reference channels, or `neuroCombat` not being installed when using ComBat."
                ),
            },
        ])

        # -----------------------------
        # CLEAN FAQ UI
        # -----------------------------
        st.markdown("## Searchable FAQ")

        search_col, cat_col = st.columns([0.65, 0.35])

        with search_col:
            search = st.text_input(
                "Search FAQ",
                placeholder="Search: species, batch, PCA, volcano, waterfall, MSFragger, report...",
                key="faq_search",
            )

        with cat_col:
            categories = ["All"] + sorted(faqs["Category"].unique().tolist())
            selected_category = st.selectbox(
                "Category",
                categories,
                key="faq_category_filter",
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
            "Input Files",
            "Merging",
            "Sample Setup",
            "Batch Correction",
            "QC",
            "Differential Analysis",
            "Visualizations",
            "MSFragger",
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

        extra_categories = [
            c for c in filtered["Category"].unique()
            if c not in category_order
        ]

        for category in extra_categories:
            cat_df = filtered[filtered["Category"] == category]

            st.markdown(f"### {category}")

            for _, row in cat_df.iterrows():
                with st.expander(row["Question"], expanded=False):
                    st.markdown(row["Answer"])