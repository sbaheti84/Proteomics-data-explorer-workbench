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
            "Use this section as a simple guide for integrating glycoproteome and total proteome differential results."
        )

        q1, q2, q3 = st.columns(3)

        with q1:
            st.markdown(
                """
                ### 1. Upload Results
                Upload two differential result tables:

                **Glycoproteome**
                - Peptide/glycopeptide-level table
                - Gene, log2FC, p/q-value columns

                **Total Proteome**
                - Protein-level table
                - Gene, log2FC, p/q-value columns
                """
            )

        with q2:
            st.markdown(
                """
                ### 2. Set Thresholds
                Define:

                - Significance metric
                - Glyco q/p cutoff
                - Protein q/p cutoff
                - log2FC cutoffs
                - Minimum glycopeptide support
                - Directional consistency
                """
            )

        with q3:
            st.markdown(
                """
                ### 3. Review Integration
                Check:

                - Concordance classes
                - Glyco vs protein scatter
                - Discordant genes
                - Glycopeptide waterfall
                - Targeted gene evidence
                - Export bundle
                """
            )

        st.info(
            "Recommended workflow: upload both result tables, confirm detected columns, review concordance overview, inspect discordant genes, then use targeted gene evidence for biological interpretation."
        )

        st.markdown("### How to Interpret the Main Outputs")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown(
                """
                #### Concordance Classes
                These summarize whether glycosylation and total protein abundance move together or separately.

                - **Concordant:** glyco and protein change in the same direction
                - **Discordant:** glyco and protein change in opposite directions
                - **Glyco-specific:** glyco changes without protein-level change
                - **Protein-specific:** protein changes without glyco-level change
                """
            )

            st.markdown(
                """
                #### Global Scatter
                This compares protein log2FC against glyco log2FC for each gene.

                Points far from the diagonal or in opposite quadrants are often the most biologically interesting.
                """
            )

        with c2:
            st.markdown(
                """
                #### Glycopeptide Waterfall
                This ranks individual glycopeptides by fold change and highlights strongest glyco drivers.

                Use this to identify which glycopeptides support a gene-level glyco call.
                """
            )

            st.markdown(
                """
                #### Targeted Gene Evidence
                Use this when a PI asks about one gene.

                It shows glycopeptide-level support, protein-level change, ratio behavior, and whether the evidence is consistent.
                """
            )

        st.divider()

        # -----------------------------
        # FAQ DATA
        # -----------------------------
        faqs = pd.DataFrame([
            {
                "Category": "General",
                "Question": "What does the Glyco-Proteome Integrator do?",
                "Answer": (
                    "The Glyco-Proteome Integrator combines glycoproteome differential results with total proteome differential results. "
                    "It summarizes evidence at the gene level and helps separate true glycosylation-specific changes from changes that may simply reflect altered protein abundance."
                ),
            },
            {
                "Category": "General",
                "Question": "What question does this app help answer?",
                "Answer": (
                    "It helps answer whether a glycopeptide change is likely driven by glycosylation biology, total protein abundance, or both. "
                    "This is useful when interpreting glycoproteomics results alongside matched proteomics data."
                ),
            },
            {
                "Category": "Input Files",
                "Question": "What files do I need to upload?",
                "Answer": (
                    "Upload two differential result tables:\n\n"
                    "1. A glycoproteome result table, usually peptide or glycopeptide level.\n"
                    "2. A total proteome result table, usually protein level.\n\n"
                    "Both files should contain gene identifiers, log2 fold-change values, and p-value or q-value columns. "
                    "The easiest source is **Download Full Dataset (Unfiltered)** in the Statistical Analysis Suite of "
                    "GlycoExplorer and of Total Proteome.\n\n"
                    "**Both tables must describe the same comparison in the same direction** (for example Treated vs Control "
                    "in both). If one is Treated vs Control and the other Control vs Treated, every gene looks discordant.\n\n"
                    "To try the app first, use **Download example files** in the sidebar (a small synthetic pair)."
                ),
            },
            {
                "Category": "Input Files",
                "Question": "What file formats are supported?",
                "Answer": (
                    "The app supports CSV, TSV, TXT, and Excel files. It automatically detects common gene, log2FC, p-value, q-value, and abundance/intensity columns."
                ),
            },
            {
                "Category": "Input Files",
                "Question": "What columns are required?",
                "Answer": (
                    "At minimum, each table needs:\n\n"
                    "- A gene identifier column, such as `Gene`, `Genes`, `GeneID`, or `Gene Symbol`\n"
                    "- A log2 fold-change column\n"
                    "- A p-value or q-value/FDR column\n\n"
                    "For ratio plots, intensity or abundance columns are also needed."
                ),
            },
            {
                "Category": "Input Files",
                "Question": "How are sample IDs and TMT channels parsed?",
                "Answer": (
                    "The app cleans common abundance and reporter channel column names, including channels like 126, 127N, 134C, and sample IDs with underscores. "
                    "This helps align glyco and protein intensity columns for ratio-style visualizations."
                ),
            },
            {
                "Category": "Column Detection",
                "Question": "What if the app cannot detect my gene or log2FC column?",
                "Answer": (
                    "Open **Input columns — auto-detected, change if wrong** at the top of the results and pick the right "
                    "gene and log2FC columns for each table. The significance column used is shown under each table. "
                    "Detection looks for terms such as `Gene`, `Genes`, `GeneID`, `log2FC`, `p-value`, `q-value`, `FDR` or `adj.P.Val`."
                ),
            },
            {
                "Category": "Column Detection",
                "Question": "My glyco or protein result has several log2FC columns. Which one is used?",
                "Answer": (
                    "Three-group results (limma F-test) contain one log2FC and q-value per pair, for example `log2FC (B vs A)`. "
                    "The app asks you to choose the pair under **Input columns** and then uses the matching `q-value (B vs A)`, "
                    "not the overall F-test p-value, which answers a different question. Choose the same pair in both tables."
                ),
            },
            {
                "Category": "Methodology",
                "Question": "How are glycopeptides summarized to genes?",
                "Answer": (
                    "The app uses an evidence-aware gene summary. For each gene, it tracks glycopeptide count, significant glycopeptide count, best significance value, median or mean log2FC, and directional consistency. "
                    "This is safer than using only one best glycopeptide or a simple average."
                ),
            },
            {
                "Category": "Methodology",
                "Question": "What is directional consistency?",
                "Answer": (
                    "Directional consistency measures how many glycopeptides for a gene move in the same direction. "
                    "For example, a value of 0.80 means 80% of non-zero glycopeptide fold changes agree in direction. Higher values indicate stronger support."
                ),
            },
            {
                "Category": "Methodology",
                "Question": "Why is minimum significant glycopeptide support important?",
                "Answer": (
                    "A gene-level glyco call based on multiple significant glycopeptides is usually more reliable than a call based on one peptide. "
                    "The minimum support filter lets you require stronger peptide-level evidence before calling a gene glyco-significant."
                ),
            },
            {
                "Category": "Methodology",
                "Question": "How are peptides mapped to multiple genes handled?",
                "Answer": (
                    "The app uses the first parsed gene as the integration key, but it flags multi-gene assignments. "
                    "These rows should be reviewed carefully because shared peptides can make biological interpretation less specific."
                ),
            },
            {
                "Category": "Interpretation",
                "Question": "What do the status labels mean?",
                "Answer": (
                    "**Status** is the simple significance pattern: Concordant (both sides significant, same direction), Discordant "
                    "(opposite directions), Glyco-specific / Protein-specific (only one side passed its cutoffs) and "
                    "**Glyco-only, protein not measured** (glyco passed, the gene is not in the protein table).\n\n"
                    "**Biological_Interpretation** adds the size of the change: it compares glyco log2FC with protein log2FC using the "
                    "*Glyco − protein |log2FC| difference* cutoff in the sidebar (default 0.5). See *How to read the integration* on the results page."
                ),
            },
            {
                "Category": "Interpretation",
                "Question": "What does glyco-specific mean?",
                "Answer": (
                    "Glycosylation-specific means the glyco change passes its thresholds, the protein **was measured**, and the protein "
                    "did not change by a similar amount (difference above the glyco − protein cutoff).\n\n"
                    "Two cases are deliberately **not** called glycosylation-specific:\n\n"
                    "- **Glyco change matches protein trend:** the protein moved by about the same amount but missed its p/q cutoff. "
                    "The glyco change is probably just protein abundance.\n"
                    "- **Glyco change, protein not measured:** the gene is not in the protein table, so protein abundance cannot be ruled out. "
                    "These stay high priority, but need protein evidence (for example a western blot) before calling them glycosylation-specific."
                ),
            },
            {
                "Category": "Interpretation",
                "Question": "What does discordant mean?",
                "Answer": (
                    "Discordant means glyco and total protein change in opposite directions. These cases are often biologically interesting, but they should be reviewed carefully using targeted gene evidence and peptide-level support."
                ),
            },
            {
                "Category": "Interpretation",
                "Question": "Is the ratio heatmap true glycan occupancy?",
                "Answer": (
                    "No. The ratio heatmap shows relative log2(Glyco / Protein). It is useful for visualizing relative glyco-to-protein behavior, but it should not be interpreted as absolute biochemical site occupancy."
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "How should I interpret the global glyco vs protein scatter plot?",
                "Answer": (
                    "The x-axis shows total protein log2FC and the y-axis shows glyco log2FC. Points near the diagonal show similar behavior across both layers. Points in opposite quadrants or far from the diagonal may indicate discordant or glyco-specific biology."
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "What is the glycopeptide waterfall plot?",
                "Answer": (
                    "The waterfall plot ranks individual glycopeptides by log2FC and highlights the strongest significant upregulated and downregulated glycopeptides. "
                    "The numbered labels connect plot points to the detailed feature table."
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "How do I get glycan structures to show in the waterfall plot?",
                "Answer": (
                    "Provide the path to a folder containing glycan PNG images in the sidebar. Image filenames should match the glycan composition text as closely as possible. "
                    "The app tries several filename patterns, including spaces, underscores, and `Glycan_` prefixes."
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "Why is the Directional UpSet plot not generating?",
                "Answer": (
                    "The Directional UpSet plot requires the `upsetplot` Python package. Install it in the same environment running Streamlit if this plot does not appear."
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "Is the advanced heatmap actually clustered?",
                "Answer": (
                    "Yes. The advanced clustering view uses hierarchical clustering, not just a ranked heatmap. It clusters genes based on glyco and protein fold-change patterns."
                ),
            },
            {
                "Category": "Export",
                "Question": "What is included in the export bundle?",
                "Answer": (
                    "The ZIP export includes a PDF report, high-resolution PNG plots, integrated result tables, glyco and protein input tables after processing, and a README describing the run."
                ),
            },
            {
                "Category": "Export",
                "Question": "Why are some plots missing from the report?",
                "Answer": (
                    "Some plots are generated from the current thresholds and available data. If a plot has no valid input, for example no overlapping genes or no significant features, it may be skipped."
                ),
            },
            {
                "Category": "Troubleshooting",
                "Question": "Why did integration return very few overlapping genes?",
                "Answer": (
                    "Common reasons include inconsistent gene symbols between glyco and protein tables, multi-gene peptide assignments, missing gene columns, or different identifier types. "
                    "Standardizing gene symbols before upload usually improves overlap."
                ),
            },
            {
                "Category": "Troubleshooting",
                "Question": "Almost every gene is discordant. Why?",
                "Answer": (
                    "The two tables are probably compared in opposite directions (Treated vs Control in one, Control vs Treated in the "
                    "other). Re-export one of them with the same reference group, or pick the matching log2FC column under Input columns."
                ),
            },
            {
                "Category": "Troubleshooting",
                "Question": "Why is the ratio heatmap empty?",
                "Answer": (
                    "The ratio heatmap requires matched intensity/abundance columns in both glyco and protein tables. It can also be empty if the minimum intensity threshold is too high or if sample/channel names do not match after cleaning."
                ),
            },
            {
                "Category": "Troubleshooting",
                "Question": "Why are glycan images missing?",
                "Answer": (
                    "The most common causes are a missing image folder, filename mismatch, or glycan composition formatting differences between the table and image files. Check the folder path and filename convention."
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
                placeholder="Search: concordance, discordant, waterfall, ratio, glycan images...",
                key="gpi_faq_search",
            )

        with cat_col:
            categories = ["All"] + sorted(faqs["Category"].unique().tolist())
            selected_category = st.selectbox(
                "Category",
                categories,
                key="gpi_faq_category_filter",
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
            "Column Detection",
            "Methodology",
            "Interpretation",
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

