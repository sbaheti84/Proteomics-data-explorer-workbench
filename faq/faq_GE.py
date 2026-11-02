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

                Then open **Data Integrity Check** and look at
                **Channel Loading & Normalization** (default: none).

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
                - Channel medians line up (Channel Loading & Normalization)
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
                    "sample grouping, a channel-loading check with optional normalization, QC, differential analysis, glycan visualization, "
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
                    "which samples are assigned, unassigned, and marked as the reference group."
                ),
            },
            {
                "Category": "Normalization",
                "Question": "Is there batch correction?",
                "Answer": (
                    "No. GlycoExplorer (TMT) analyses **one TMT plex at a time**: all channels of a plex are labelled, pooled "
                    "and measured together, so there is no batch effect between them to correct. The merge adds up each "
                    "glycoform over all raw files of the upload, so every file must come from the same plex (fractions of "
                    "one plex are fine). If a study spans several plexes, analyse each plex separately."
                ),
            },
            {
                "Category": "Normalization",
                "Question": "Do I need to normalize?",
                "Answer": (
                    "Often not. Open **Data Integrity Check → Channel Loading & Normalization**. Each box is one channel. "
                    "If the boxes line up, keep the default **None**. If one channel is clearly higher or lower, that "
                    "sample had more or less material loaded and the app shows a warning; normalizing removes that "
                    "difference. **Per-channel details** under the plot lists each channel's median and whether a whole "
                    "group is offset from the reference group (such an offset shifts every fold change of that group)."
                ),
            },
            {
                "Category": "Normalization",
                "Question": "Which normalization should I choose?",
                "Answer": (
                    "- **None** (default): abundances as exported.\n"
                    "- **Median normalization (glycopeptides)**: makes the channel medians equal. Assumes most glycoforms do "
                    "not change; a real global change (for example more sialylation everywhere) would be removed too.\n"
                    "- **Total intensity (sample loading)**: classic TMT normalization to equal total signal; same assumption, "
                    "more influenced by the brightest glycoforms.\n"
                    "- **Factors from a total-proteome table (same TMT pool)**: the safest choice when you have it. Loading "
                    "factors come from the proteins of the same labelled samples, so glycosylation changes are kept. Upload "
                    "the protein table with the same TMT channel names.\n\n"
                    "The plot shows the channels before and after. Zeros (not quantified) are never changed."
                ),
            },
            {
                "Category": "Normalization",
                "Question": "What happens to my results when I change the normalization?",
                "Answer": (
                    "QC plots update straight away and earlier statistics are cleared, so they cannot be mixed up with the new "
                    "matrix. Run the statistics again. The method used is printed in the statistics panel and in the PDF report."
                ),
            },
            {
                "Category": "QC",
                "Question": "What QC modules are available?",
                "Answer": (
                    "The **Data Integrity Check & Sanity Profiling** section has two blocks:\n\n"
                    "- **Channel Loading & Normalization** (top): channel box plot before/after and the "
                    "normalization choice (default: none)\n"
                    "- **Quality Control Plots** (below): PCA, Correlation Matrix, Sample Clustering and "
                    "Abundance Distribution (per sample). These use the normalization chosen above."
                ),
            },
            {
                "Category": "QC",
                "Question": "What does a good PCA look like?",
                "Answer": (
                    "Replicates should usually cluster together. A sample far from its group may be an outlier (failed "
                    "labelling, wrong channel or unequal loading) and should be reviewed before interpreting results."
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
                "Category": "Data Overview",
                "Question": "Where is the glycan classification summary?",
                "Answer": (
                    "It describes what was identified, not data quality, so it now sits next to the merged matrix: "
                    "open the **Glycan class composition** tab under *Integrated Glycopeptide Abundance Matrix*. "
                    "To see which glycan classes or traits (sialylation, fucosylation) change between groups, run the "
                    "statistics and choose **Glycan class breakdown** in Results Visualizations."
                ),
            },
            {
                "Category": "Differential Analysis",
                "Question": "Which statistical test should I use?",
                "Answer": (
                    "For two groups: limma moderated t-test, Student's t-test, Welch's t-test, paired t-test, "
                    "Mann-Whitney U and Wilcoxon signed-rank. For three groups: one-way ANOVA or the limma moderated "
                    "F-test.\n\n"
                    "With the 3-6 replicates typical of TMT experiments, **limma** is usually the best choice: it "
                    "borrows variance information from all glycoforms, which makes it more reliable and more powerful "
                    "than an ordinary t-test. Use the paired t-test only for matched samples. Rank-based tests "
                    "(Mann-Whitney, Wilcoxon) need larger groups to reach small p-values."
                ),
            },
            {
                "Category": "Differential Analysis",
                "Question": "How does the limma option work, and why can its log2FC differ from the t-test?",
                "Answer": (
                    "limma fits a linear model to the **log2** abundance of every glycoform (zeros count as missing), "
                    "then shrinks each glycoform's variance towards the typical variance of all glycoforms "
                    "(empirical Bayes). This gives the moderated t- and F-statistics used in most proteomics papers. "
                    "The implementation is a Python port of R limma's lmFit, contrasts.fit and eBayes, and gives the "
                    "same numbers as R limma.\n\n"
                    "limma's log2FC is mean(log2 test) minus mean(log2 reference). The other tests run on raw "
                    "abundances and report log2 of the ratio of mean abundances, so the two fold changes can differ, "
                    "most for glycoforms with very skewed values."
                ),
            },
            {
                "Category": "Differential Analysis",
                "Question": "What happens when I select 3 groups?",
                "Answer": (
                    "Choose **All three groups** and the limma moderated F-test (default) or one-way ANOVA. Both test "
                    "whether any group differs. The limma F-test also reports log2FC and q-value for every pair of groups "
                    "(B vs A, C vs A, C vs B) in the same table. There is no single log2FC direction for the overall test. Volcano and waterfall plots are mainly for two-group comparisons, "
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
                "Question": "Total-proteome normalization says channels do not match. Why?",
                "Answer": (
                    "The protein table must have one abundance column for every glyco channel, with the same TMT channel "
                    "names (126, 127N, ...). Check that it comes from the same TMT pool and that no channel was removed."
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
                placeholder="Search: species, normalization, PCA, volcano, waterfall, MSFragger, report...",
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
            "Normalization",
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