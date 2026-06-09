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
            "A simple guide for reviewing label-free glycopeptide results without needing to understand every technical setting."
        )

        q1, q2, q3 = st.columns(3)

        with q1:
            st.markdown(
                """
                ### 1. Upload Data
                Choose one LFQ workflow:

                **pGlyco (LFQ)**
                - One pGlyco LFQ file
                - Raw `IsotopeArea` intensities

                **MSFragger (LFQ)**
                - Multiple TSV/TXT files
                - One file per sample/run
                """
            )

        with q2:
            st.markdown(
                """
                ### 2. Compile and Group
                Click **Compile LFQ Data Matrix**, then go to
                **Data Filtering & Group Specification**.

                Select:
                - Sample abundance columns
                - Group labels
                - Control/reference group
                - Optional batch correction
                """
            )

        with q3:
            st.markdown(
                """
                ### 3. Review and Export
                Check:

                - PCA and correlation plots
                - Sample clustering heatmap
                - Glycan class summary
                - Differential results
                - Volcano / waterfall
                - Box / heatmap / lollipop plots
                - PDF/ZIP export
                """
            )

        st.info(
            "Recommended workflow: compile the LFQ matrix → define groups → check QC → run statistics → review volcano/waterfall/heatmap → validate selected genes or sites with box and lollipop plots."
        )

        st.markdown("### How to Interpret the Main Outputs")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown(
                """
                #### PCA / Correlation
                Use these first to confirm that samples behave as expected.

                Good signs:
                - Replicates cluster together
                - No obvious outlier sample
                - Groups separate by biology, not technical batch
                """
            )

            st.markdown(
                """
                #### Volcano Plot
                Best for two-group comparisons.

                - X-axis = log2 fold-change
                - Y-axis = statistical significance
                - Toggle between Static and Interactive modes for exploration.
                """
            )

        with c2:
            st.markdown(
                """
                #### Waterfall Plot
                Ranks glycopeptides by log2 fold-change.

                It highlights:
                - Significantly upregulated glycopeptides
                - Significantly downregulated glycopeptides
                - Numbered top hits with supporting table
                """
            )

            st.markdown(
                """
                #### Box / Lollipop Plots
                Use these to validate individual findings.

                - Box plot: sample-level abundance pattern (can filter by up/down hits).
                - Box plot (Gene): selected gene-level review.
                - Lollipop: glycosylation site-level changes across a protein.
                """
            )

        st.divider()

        # -----------------------------
        # FAQ DATA
        # -----------------------------
        faqs = pd.DataFrame([
            {
                "Category": "General",
                "Question": "What is GlycoExplorer LFQ?",
                "Answer": (
                    "GlycoExplorer LFQ is a Streamlit application for label-free glycopeptide analysis. "
                    "It compiles pGlyco LFQ or MSFragger LFQ outputs into an abundance matrix, supports sample grouping, "
                    "optional batch correction, QC, differential analysis, glycan visualization, waterfall plots, "
                    "box plots, heatmaps, lollipop plots, and report export."
                ),
            },
            {
                "Category": "General",
                "Question": "Which LFQ workflows are supported?",
                "Answer": (
                    "The app supports two label-free glycoproteomics workflows:\n\n"
                    "1. **pGlyco (LFQ)** using one pGlyco LFQ file.\n"
                    "2. **MSFragger (LFQ)** using multiple TSV/TXT files, typically one file per sample/run."
                ),
            },
            {
                "Category": "General",
                "Question": "Is this the same as the TMT GlycoExplorer app?",
                "Answer": (
                    "No. This FAQ is for the **label-free** GlycoExplorer app. "
                    "The TMT app uses pGlyco + Proteome Discoverer or MSFragger TMT files. "
                    "This LFQ app uses pGlyco LFQ raw `IsotopeArea` values or multiple MSFragger LFQ sample-run files."
                ),
            },

            # ---------------- INPUT FILES ----------------
            {
                "Category": "Input Files",
                "Question": "What files do I need for pGlyco LFQ?",
                "Answer": (
                    "Upload one pGlyco LFQ file as CSV, TSV/TXT, XLSX, or XLS.\n\n"
                    "Required columns include:\n"
                    "- `Peptide`\n"
                    "- `Genes`\n"
                    "- `Proteins`\n"
                    "- `ProSites`\n"
                    "- `RawName`\n"
                    "- `IsotopeArea`\n"
                    "- `Glycan(H,N,A,F)` or corrected/equivalent glycan column\n\n"
                    "The app expects **raw intensity** values in `IsotopeArea`, not log-transformed values."
                ),
            },
            {
                "Category": "Input Files",
                "Question": "What files do I need for MSFragger LFQ?",
                "Answer": (
                    "Upload multiple MSFragger TSV/TXT files, usually one file per sample or run.\n\n"
                    "Required or expected columns include:\n"
                    "- `Total Glycan Composition`\n"
                    "- `Intensity` or another intensity-like column\n"
                    "- `Peptide`\n"
                    "- `Protein ID`\n"
                    "- `Gene` when available\n"
                    "- `Assigned Modifications` and protein start position when site estimation is needed"
                ),
            },
            {
                "Category": "Input Files",
                "Question": "Why do I get 'IsotopeArea appears log-transformed'?",
                "Answer": (
                    "The app checks whether `IsotopeArea` values look too small for raw intensities. "
                    "If the maximum value is very low, the file may already be log-transformed. "
                    "For pGlyco LFQ, export raw `IsotopeArea` intensities so normalization, fold-change, and imputation work correctly."
                ),
            },
            {
                "Category": "Input Files",
                "Question": "Why are my sample columns not detected?",
                "Answer": (
                    "For pGlyco LFQ, samples are created from the `RawName` field when the app pivots the matrix. "
                    "For MSFragger LFQ, the app creates abundance columns from uploaded sample-run files and intensity columns. "
                    "Check that intensity values are numeric and that expected fields such as `RawName`, `IsotopeArea`, "
                    "`Total Glycan Composition`, and `Intensity` are present."
                ),
            },
            {
                "Category": "Input Files",
                "Question": "What does the glycan image directory do?",
                "Answer": (
                    "The app asks for a glycan image directory so waterfall, box, and related glycan visualizations can display structures. "
                    "If images are missing or filenames do not match the glycan composition, plots can still run but may show text-based glycan labels instead of structures."
                ),
            },

            # ---------------- PROCESSING ----------------
            {
                "Category": "Processing",
                "Question": "What does Compile LFQ Data Matrix do?",
                "Answer": (
                    "The compile step converts raw LFQ identification/intensity files into an analysis-ready glycopeptide abundance matrix. "
                    "After this step, you can define groups, run QC, perform differential analysis, and export results."
                ),
            },
            {
                "Category": "Processing",
                "Question": "How is the pGlyco LFQ matrix created?",
                "Answer": (
                    "The app validates required pGlyco columns, checks raw `IsotopeArea`, creates glycan and glycopeptide identifiers, "
                    "groups by sample/raw file and glycopeptide identity, and pivots the data so samples become abundance columns."
                ),
            },
            {
                "Category": "Processing",
                "Question": "How is the MSFragger LFQ matrix created?",
                "Answer": (
                    "The app reads multiple MSFragger files, extracts glycan composition, estimates ProSites when possible, "
                    "summarizes glycopeptide intensity for each sample/run, and outer-joins all sample files into one LFQ matrix."
                ),
            },
            {
                "Category": "Processing",
                "Question": "How are duplicate glycopeptide entries handled?",
                "Answer": (
                    "The app summarizes duplicate entries by glycopeptide identity and sample. "
                    "For pGlyco LFQ, raw isotope areas are summarized during matrix construction. "
                    "For MSFragger LFQ, sample-run intensities are summarized and joined across files."
                ),
            },

            # ---------------- SAMPLE SETUP ----------------
            {
                "Category": "Sample Setup",
                "Question": "How do I define groups?",
                "Answer": (
                    "Use **Data Filtering & Group Specification**. Select sample abundance columns, assign them to Group 1, Group 2, "
                    "and optionally Group 3, then choose the global reference/control group."
                ),
            },
            {
                "Category": "Sample Setup",
                "Question": "What is the reference group?",
                "Answer": (
                    "The reference group controls fold-change direction. For example, if Control is the reference group, "
                    "positive log2FC means the comparison/test group has higher abundance relative to Control."
                ),
            },
            {
                "Category": "Sample Setup",
                "Question": "Can I analyze three groups?",
                "Answer": (
                    "Yes. When three groups are defined, the app performs ANOVA. "
                    "ANOVA identifies features that differ across groups, but it does not have a simple two-group log2FC direction. "
                    "Volcano and waterfall views are mainly for pairwise comparisons."
                ),
            },

            # ---------------- BATCH CORRECTION ----------------
            {
                "Category": "Batch Correction",
                "Question": "When should I use batch correction?",
                "Answer": (
                    "Use batch correction if samples were processed on different days, runs, preparation batches, or instruments "
                    "and QC plots show separation by batch rather than biological group."
                ),
            },
            {
                "Category": "Batch Correction",
                "Question": "Which batch correction methods are available?",
                "Answer": (
                    "The LFQ app provides:\n\n"
                    "- **ComBat (Empirical Bayes)**\n"
                    "- **Median Centering**\n"
                    "- **Mean Centering**\n"
                    "- **Standardize (Mean + Variance)**\n\n"
                    "ComBat requires the `neuroCombat` package."
                ),
            },
            {
                "Category": "Batch Correction",
                "Question": "How does batch correction work in this LFQ app?",
                "Answer": (
                    "The app applies batch correction on a log2-transformed version of the selected sample abundance matrix, "
                    "then converts values back to the raw scale for downstream analysis. "
                    "After correction, downstream QC and statistics use the corrected abundance values."
                ),
            },
            {
                "Category": "Batch Correction",
                "Question": "What happens if I turn batch correction off?",
                "Answer": (
                    "The app restores or keeps the raw uncorrected LFQ matrix. "
                    "This is appropriate when no strong batch effect is visible or all samples were processed together."
                ),
            },

            # ---------------- QC ----------------
            {
                "Category": "QC",
                "Question": "What QC modules are available?",
                "Answer": (
                    "The **Data Integrity Check & Sanity Profiling** section includes:\n\n"
                    "- Principal Component Analysis (PCA)\n"
                    "- Correlation Matrix\n"
                    "- Sample Clustering\n"
                    "- Glycan Classification Summary"
                ),
            },
            {
                "Category": "QC",
                "Question": "What preprocessing options should I check before QC?",
                "Answer": (
                    "The app includes LFQ preprocessing options such as treating zeros as missing values and log2 transformation. "
                    "These steps help avoid artificial zeros, stabilize variance, and make PCA, clustering, and statistical tests more interpretable."
                ),
            },
            {
                "Category": "QC",
                "Question": "What does a good PCA look like?",
                "Answer": (
                    "Replicates should usually cluster together. If samples separate mostly by batch instead of group, "
                    "review the batch setup and consider batch correction."
                ),
            },
            {
                "Category": "QC",
                "Question": "How should I interpret the correlation matrix?",
                "Answer": (
                    "Samples from the same biological group or replicate set should generally show higher correlation. "
                    "A sample with unusually low correlation to all others may be an outlier or may have a technical issue."
                ),
            },
            {
                "Category": "QC",
                "Question": "What does the glycan classification summary show?",
                "Answer": (
                    "It summarizes glycopeptides by glycan class, such as high-mannose, paucimannose, or complex/hybrid, "
                    "based on the glycan composition detected by the app."
                ),
            },

            # ---------------- DIFFERENTIAL ANALYSIS ----------------
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
                    "For three groups, the app uses ANOVA."
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
                "Question": "What do the zero imputation and presence filters do?",
                "Answer": (
                    "Zero/NA imputation replaces missing or zero LFQ values using a row-wise small fraction of the observed minimum. "
                    "Presence filtering keeps features that are observed in enough samples overall or within groups. "
                    "These options reduce unstable statistics caused by sparse LFQ data."
                ),
            },
            {
                "Category": "Differential Analysis",
                "Question": "What thresholds should I start with?",
                "Answer": (
                    "Common starting thresholds are:\n\n"
                    "- `q-value < 0.05`\n"
                    "- `|log2FC| > 1`\n\n"
                    "For exploratory analysis, you may relax to `q-value < 0.1` or `|log2FC| > 0.5`, depending on study goals."
                ),
            },
            {
                "Category": "Differential Analysis",
                "Question": "What does Apply Filters & View Subset do?",
                "Answer": (
                    "After differential analysis, this filters the results table using the selected significance metric and fold-change cutoff. "
                    "For ANOVA, fold-change filtering is not applied because ANOVA does not produce a simple pairwise fold-change direction."
                ),
            },

            # ---------------- VISUALIZATIONS ----------------
            {
                "Category": "Visualizations",
                "Question": "Which result visualizations are available?",
                "Answer": (
                    "For pairwise results, available modules include Volcano, Waterfall, Box plot, Heatmap, Box plot (Gene), and Lollipop plot. "
                    "For ANOVA, the app focuses on Box plot, Heatmap, Box plot (Gene), and Lollipop plot."
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "How do I interpret the volcano plot?",
                "Answer": (
                    "The volcano plot shows log2 fold-change on the x-axis and statistical significance on the y-axis. "
                    "Points farther left or right have larger abundance changes; points higher up have stronger significance. "
                    "You can toggle between a Static image and an Interactive plot to hover over individual points."
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "Why is volcano not available for ANOVA?",
                "Answer": (
                    "ANOVA tests whether any of three groups differ. It does not produce one pairwise log2FC direction, "
                    "so volcano plots are mainly used for two-group pairwise comparisons."
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "What is the waterfall plot?",
                "Answer": (
                    "The waterfall plot ranks glycopeptides by log2FC and highlights significantly up- and down-regulated features. "
                    "It can also display supporting glycan/gene/site information for top hits."
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "Why is my waterfall plot empty?",
                "Answer": (
                    "Possible reasons include:\n\n"
                    "- No glycopeptides pass the selected significance threshold\n"
                    "- Fold-change cutoff is too strict\n"
                    "- Group assignments are incorrect\n"
                    "- Differential analysis has not been run\n"
                    "- You are viewing an ANOVA result rather than pairwise results"
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "How should I interpret box plots?",
                "Answer": (
                    "Box plots show abundance distributions across groups. They help confirm whether a differential result is consistent "
                    "across samples or driven by one outlier. For pairwise tests, you can highlight specifically up, down, or all top hits."
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "How should I interpret the lollipop plot?",
                "Answer": (
                    "The lollipop plot maps glycosylation changes to protein site positions. It is useful for identifying site-specific "
                    "changes within a gene or protein."
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "Why are glycan images missing?",
                "Answer": (
                    "Glycan images require a valid glycan image directory and matching image filenames. "
                    "If an image is not found, the app may show a text glycan composition instead."
                ),
            },

            # ---------------- MSFRAGGER ----------------
            {
                "Category": "MSFragger",
                "Question": "How are MSFragger glycan compositions parsed?",
                "Answer": (
                    "The app reads `Total Glycan Composition`, extracts Hex, HexNAc, NeuAc, and Fuc counts, "
                    "and converts them to the pGlyco-style H N A F composition used in downstream plots."
                ),
            },
            {
                "Category": "MSFragger",
                "Question": "How are ProSites estimated for MSFragger LFQ?",
                "Answer": (
                    "When possible, the app uses the protein start position and modified N-site positions from `Assigned Modifications`. "
                    "Large N-linked modification masses are treated as likely glycan modifications."
                ),
            },

            # ---------------- EXPORT ----------------
            {
                "Category": "Export",
                "Question": "What can I export?",
                "Answer": (
                    "The app can save the compiled LFQ master matrix as CSV or Excel and can export generated plots and reports. "
                    "The report/ZIP bundle is session-driven and includes results that were generated during the current session."
                ),
            },
            {
                "Category": "Export",
                "Question": "Why are some plots missing from the PDF or ZIP?",
                "Answer": (
                    "Exports are session-driven. Only plots generated during the current session are available for the report/ZIP. "
                    "Open or run the relevant PCA, heatmap, volcano, waterfall, box, or lollipop plot before exporting."
                ),
            },
            {
                "Category": "Export",
                "Question": "Where are files saved in the desktop app?",
                "Answer": (
                    "Many save buttons write directly to the user's Downloads folder. "
                    "This behavior is designed to be more reliable in packaged desktop Streamlit applications."
                ),
            },

            # ---------------- TROUBLESHOOTING ----------------
            {
                "Category": "Troubleshooting",
                "Question": "Why did matrix compilation fail?",
                "Answer": (
                    "Common causes include missing required columns, non-numeric intensity values, log-transformed pGlyco `IsotopeArea`, "
                    "empty uploaded files, or MSFragger files without usable glycan compositions."
                ),
            },
            {
                "Category": "Troubleshooting",
                "Question": "Why did batch correction fail?",
                "Answer": (
                    "Batch correction can fail if batches are empty, have too few samples, contain too many missing values, "
                    "or if `neuroCombat` is not installed when using ComBat."
                ),
            },
            {
                "Category": "Troubleshooting",
                "Question": "Why did report generation fail?",
                "Answer": (
                    "Report generation may fail if expected plots, tables, or columns are missing. "
                    "Re-run the relevant plots and confirm differential analysis completed successfully before exporting."
                ),
            },
            {
                "Category": "Troubleshooting",
                "Question": "Why is the app slow with large files?",
                "Answer": (
                    "Large LFQ glycoproteomics tables can take time to compile, filter, plot, and export. "
                    "Use targeted visualizations, remove unnecessary columns, and filter to the features needed for review."
                ),
            },
        ])

        # -----------------------------
        # CLEAN SEARCHABLE FAQ UI
        # -----------------------------
        st.markdown("## Searchable FAQ")

        search_col, cat_col = st.columns([0.65, 0.35])

        with search_col:
            search = st.text_input(
                "Search FAQ",
                placeholder="Search: pGlyco, MSFragger, IsotopeArea, batch, PCA, volcano, waterfall, report...",
                key="faq_search_lfq",
            )

        with cat_col:
            categories = ["All"] + sorted(faqs["Category"].unique().tolist())
            selected_category = st.selectbox(
                "Category",
                categories,
                key="faq_category_filter_lfq",
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
            "Processing",
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