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
            "A simple guide for reviewing label-free total protein results without needing to understand every technical setting."
        )

        q1, q2, q3 = st.columns(3)

        with q1:
            st.markdown(
                '''
                ### 1. Upload Data
                Upload one LFQ total proteomics matrix.

                Supported formats:
                - CSV
                - TSV / TXT
                - Excel XLSX

                The file should include protein identifiers and numeric LFQ abundance/intensity columns.
                '''
            )

        with q2:
            st.markdown(
                '''
                ### 2. Define Groups
                Go to **Data Filtering & Group Specification**.

                Select:
                - Active LFQ sample columns (LFQ columns are pre-selected)
                - 2 or 3 groups
                - Group labels
                - Reference/control group

                Then open **Preprocessing, QC & Batch Correction**, pick the
                imputation method and click **Run Normalization + Imputation**.
                '''
            )

        with q3:
            st.markdown(
                '''
                ### 3. Review Results
                Check:

                - PCA and QC plots
                - Correlation heatmap
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
            "Recommended workflow: upload matrix → define groups and reference group → normalize + impute (your choice of method) "
            "→ check QC → optional batch correction → limma (recommended) or t-test → review volcano, MA, waterfall, heatmap "
            "and protein profiles → export report."
        )

        st.markdown("### How to Interpret the Main Outputs")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown(
                '''
                #### PCA / Correlation
                Use these first to check whether samples behave as expected.

                Good signs:
                - Replicates cluster together
                - No obvious outlier sample
                - Samples separate by biology, not batch
                '''
            )

            st.markdown(
                '''
                #### Volcano / MA Plot
                Best for two-group comparisons.

                - Volcano shows fold-change and significance
                - MA plot shows fold-change versus average abundance
                - Use annotation controls to label top proteins
                '''
            )

        with c2:
            st.markdown(
                '''
                #### Waterfall Plot
                Best for quickly seeing the strongest protein-level changes.

                It shows:
                - Upregulated proteins
                - Downregulated proteins
                - Ranked log2 fold-change
                - Numbered table for top hits
                '''
            )

            st.markdown(
                '''
                #### Protein Profile / Box Plot
                Use these to validate individual proteins.

                - Confirms whether a finding is consistent across samples
                - Helps detect if a result is driven by one outlier
                - Top profiles can be viewed/exported as multi-panel grids
                '''
            )

        st.divider()

        # -----------------------------
        # FAQ DATA
        # -----------------------------
        faqs = pd.DataFrame([
            {
                "Category": "General",
                "Question": "What is the Total Protein LFQ app?",
                "Answer": (
                    "The Total Protein LFQ app analyzes label-free total proteomics abundance matrices. "
                    "It supports sample grouping, preprocessing, QC, optional batch correction, differential analysis, "
                    "volcano plots, MA plots, heatmaps, waterfall plots, protein profile plots, and final PDF/ZIP export."
                ),
            },
            {
                "Category": "General",
                "Question": "How is this different from the Total Protein TMT app?",
                "Answer": (
                    "This app is for label-free total protein matrices. It accepts CSV, TSV/TXT, and Excel files and detects numeric LFQ sample columns. "
                    "The TMT app is focused on abundance/reporter/intensity matrix exports from TMT-style workflows."
                ),
            },
            {
                "Category": "Input Files",
                "Question": "What kind of file do I need to upload?",
                "Answer": (
                    "Upload one total protein LFQ matrix. The app supports CSV, TSV/TXT, and Excel XLSX formats. "
                    "The file should contain protein identifiers and numeric LFQ intensity or abundance sample columns."
                ),
            },
            {
                "Category": "Input Files",
                "Question": "Which identifier columns are supported?",
                "Answer": (
                    "The app can use identifier columns such as `GeneID`, `Genes`, `Gene names`, `Description`, `Accession`, `Protein ID`, "
                    "`Protein.IDs`, or `Master Protein Accessions`. It creates internal `GeneID` and `RowID` fields for downstream plotting.\n\n"
                    "Common exports work directly: MaxQuant `proteinGroups.txt` (the `LFQ intensity` columns are pre-selected), "
                    "FragPipe `combined_protein.tsv` (the `MaxLFQ Intensity` columns are pre-selected) and DIA-NN `pg_matrix.tsv` "
                    "(file paths in column names are shortened to the file name)."
                ),
            },
            {
                "Category": "Input Files",
                "Question": "How does the app detect LFQ sample columns?",
                "Answer": (
                    "The app converts non-metadata columns to numeric where possible, then keeps numeric columns as candidate sample columns. "
                    "It excludes obvious metadata or statistical columns such as normalized values, ratios, counts, CV, S/N, and correlation columns."
                ),
            },
            {
                "Category": "Input Files",
                "Question": "Why did the app drop my abundance columns?",
                "Answer": (
                    "Possible reasons include:\n\n"
                    "- the columns are not numeric\n"
                    "- headers contain filtered keywords such as `normalized`, `scaled`, `ratio`, `count`, `cv`, or `S/N`\n"
                    "- the file has unexpected formatting\n\n"
                    "Check your input headers and confirm that LFQ intensity columns contain numeric values."
                ),
            },
            {
                "Category": "Input Files",
                "Question": "What does the example dataset do?",
                "Answer": (
                    "The sidebar example (`Protein_abundance_LFQ.xlsx`) shows the expected layout: identifier columns followed by one "
                    "intensity column per sample. It is very small (16 proteins), so use it to check formatting, not to judge statistics."
                ),
            },

            # ---------------- SAMPLE SETUP ----------------
            {
                "Category": "Sample Setup",
                "Question": "How do I define groups?",
                "Answer": (
                    "Use **Data Filtering & Group Specification**. Select active LFQ sample columns, choose 2 or 3 groups, assign samples to groups, "
                    "and set the global reference/control group."
                ),
            },
            {
                "Category": "Sample Setup",
                "Question": "What is the reference group?",
                "Answer": (
                    "The reference group controls fold-change direction. Positive log2FC means higher abundance in the comparison/test group relative to the selected reference group."
                ),
            },
            {
                "Category": "Sample Setup",
                "Question": "Can I analyze three groups?",
                "Answer": (
                    "Yes. With three groups you can either test one pair at a time (volcano, waterfall and MA plots) or test all three "
                    "together with the limma F-test or one-way ANOVA. The limma F-test also reports log2FC and q-value for every pair."
                ),
            },

            # ---------------- PREPROCESSING ----------------
            {
                "Category": "Preprocessing",
                "Question": "Should I treat zeros as missing values?",
                "Answer": (
                    "Usually yes for LFQ data. In mass spectrometry, zero often means a protein was below detection rather than truly absent. "
                    "Treating zeros as missing values avoids artificially lowering group means."
                ),
            },
            {
                "Category": "Preprocessing",
                "Question": "Should I use log2 transformation?",
                "Answer": (
                    "Usually yes. Log2 transformation makes intensity distributions more suitable for t-tests/ANOVA and makes fold-change interpretation easier."
                ),
            },
            {
                "Category": "Imputation",
                "Question": "Which imputation method should I use?",
                "Answer": (
                    "It depends on why values are missing. Open *Help: what each step does* in step 1: the bar chart shows the "
                    "% missing for low- to high-intensity proteins.\n\n"
                    "- **Missing mostly in low-intensity proteins** (typical for DDA label-free): a low-end method — "
                    "Perseus (PI), MinProb, MinDet, QRILC, or the 1/5 minimum rule.\n"
                    "- **Missing looks random** (often DIA, or few missing values): a similarity method — KNN, BPCA, SVD, LLS or RF.\n"
                    "- **You use limma and prefer not to invent values:** choose *None*. limma uses only measured values.\n\n"
                    "Avoid Zero, Column median and Row median for differential analysis; they distort fold changes."
                ),
            },
            {
                "Category": "Imputation",
                "Question": "What is the '1/5 of protein minimum (previous default)' method?",
                "Answer": (
                    "Each missing value of a protein is set to 1/5 of that protein's smallest measured intensity (on the log2 scale: "
                    "its smallest value minus 2.32), then shifted by the same normalization as the sample. It is what this app used "
                    "before, so it is the default to keep results comparable.\n\n"
                    "Its weakness: all filled values of a protein are the same number. If a protein is missing in a whole group, that "
                    "group has zero spread and the t-test can look more significant than it is. Perseus (PI) or MinProb add realistic "
                    "noise and avoid this."
                ),
            },
            {
                "Category": "Imputation",
                "Question": "What do 'built-in' and 'needs R' mean?",
                "Answer": (
                    "**Built-in** methods (1/5 minimum, Perseus, MinProb, MinDet, medians, minimum, zero) run inside the app and work "
                    "everywhere, including hosted servers. They follow the same formulas as NAguideR/imputeLCMD/Perseus; the deterministic "
                    "ones give the same numbers as R. Perseus and MinProb draw random numbers with a fixed seed, so a re-run gives the same "
                    "values, but not the same random numbers as R.\n\n"
                    "**Needs R** methods (KNN, BPCA, SVD, QRILC, RF, ...) run the NAguideR code in R. The Windows, macOS and Docker builds "
                    "include R and these packages. On a server without R these methods are shown as unavailable and the Run button is "
                    "disabled; the app never replaces your choice with another method."
                ),
            },
            {
                "Category": "Imputation",
                "Question": "Earlier results said 'Temporary RConnect mode'. What was that?",
                "Answer": (
                    "An earlier version always used the 1/5 minimum rule, whatever method was selected, because R was not available on "
                    "the hosted server. That mode has been removed: the method you pick is the method that runs, and the report records "
                    "it. To reproduce those earlier results, choose **1/5 of protein minimum (previous default)**."
                ),
            },
            {
                "Category": "Imputation",
                "Question": "Can I run statistics without imputation?",
                "Answer": (
                    "Yes. Choose *None (leave missing values)*. limma and the t-tests then use only measured values; proteins with too few "
                    "measured values in a compared group get no p-value. PCA and heatmaps fill gaps temporarily for drawing only. "
                    "ComBat batch correction needs a complete matrix, so it is not possible without imputation."
                ),
            },
            {
                "Category": "Preprocessing",
                "Question": "Are contaminants and decoys removed?",
                "Answer": (
                    "Yes, when the file marks them: MaxQuant `Reverse`, `Potential contaminant` and `Only identified by site` columns "
                    "(value `+`), and protein IDs starting with `CON__`, `REV__`, `contam_` or `rev_`. The option appears in step 1 "
                    "only when such rows exist, and it is on by default."
                ),
            },
            {
                "Category": "Preprocessing",
                "Question": "In which order are the preprocessing steps applied?",
                "Answer": (
                    "Remove contaminants → 0 becomes missing → drop sparse proteins → log2 → normalize → impute. Then QC, then optional "
                    "batch correction. Statistics use this final matrix and never re-impute or re-normalize."
                ),
            },
            {
                "Category": "Preprocessing",
                "Question": "How does the presence filter work?",
                "Answer": (
                    "The presence filter removes sparse proteins before imputation. You can require a protein to be present in a minimum fraction of total samples, "
                    "in at least one group, or by combined AND/OR rules. 'At least one group' keeps proteins that are present in one "
                    "condition and absent in another."
                ),
            },

            # ---------------- BATCH CORRECTION ----------------
            {
                "Category": "Batch Correction",
                "Question": "When should I use batch correction?",
                "Answer": (
                    "Use batch correction only when samples were processed across different runs, preparation days, instruments, or technical batches "
                    "and QC plots suggest batch-driven separation."
                ),
            },
            {
                "Category": "Batch Correction",
                "Question": "Which batch correction methods are available?",
                "Answer": (
                    "The LFQ app supports:\n\n"
                    "- **ComBat (Empirical Bayes)**\n"
                    "- **Median Centering**\n"
                    "- **Mean Centering**\n"
                    "- **Standardize (Mean + Variance)**\n\n"
                    "ComBat requires the `neuroCombat` package and works best when batches have enough samples."
                ),
            },
            {
                "Category": "Batch Correction",
                "Question": "How is batch correction applied in the LFQ app?",
                "Answer": (
                    "Batch correction runs after normalization, imputation and QC, on the log2 matrix. The corrected log2 matrix is used "
                    "directly for statistics. ComBat keeps the biological group as a covariate so that group differences are not removed."
                ),
            },
            {
                "Category": "Batch Correction",
                "Question": "What should I check after batch correction?",
                "Answer": (
                    "Check PCA, correlation heatmap, density plots, and boxplots. Batch correction should reduce technical separation without removing meaningful biological group separation."
                ),
            },

            # ---------------- QC ----------------
            {
                "Category": "QC",
                "Question": "What QC plots should I review first?",
                "Answer": (
                    "Start with PCA, density/distribution plots, sample boxplots, correlation heatmap, missing-value summaries, QC heatmap, and clustering/dendrogram views."
                ),
            },
            {
                "Category": "QC",
                "Question": "How should I interpret the PCA plot?",
                "Answer": (
                    "PCA shows global similarity between samples. Replicates should usually cluster together. If samples cluster by technical batch instead of biological group, "
                    "review batch correction or preprocessing choices."
                ),
            },
            {
                "Category": "QC",
                "Question": "What does the missing values plot show?",
                "Answer": (
                    "It shows the percentage or pattern of missing values per sample. Samples with unusually high missingness may need closer review before statistical testing."
                ),
            },
            {
                "Category": "QC",
                "Question": "Why do some heatmaps show RowID?",
                "Answer": (
                    "`RowID` is the internal identifier (gene + accession). It keeps proteins with the same gene name apart."
                ),
            },

            # ---------------- DIFFERENTIAL ANALYSIS ----------------
            {
                "Category": "Differential Analysis",
                "Question": "What statistical tests are used?",
                "Answer": (
                    "For two groups: the limma moderated t-test (recommended), Student's t-test, Welch's t-test, paired t-test, Mann-Whitney U "
                    "and Wilcoxon signed-rank. For three groups: the limma moderated F-test (recommended) or one-way ANOVA. "
                    "P-values are adjusted to q-values with Benjamini-Hochberg FDR over the proteins that could be tested."
                ),
            },
            {
                "Category": "Differential Analysis",
                "Question": "Why is limma recommended?",
                "Answer": (
                    "Label-free studies usually have 3-6 samples per group. With so few values, an ordinary t-test often finds a tiny "
                    "variance by chance (false hits) or a large one (missed hits). limma shrinks each protein's variance towards the typical "
                    "variance of all proteins (empirical Bayes). This makes results more stable and more powerful. It also handles missing "
                    "values by using only the measured ones. The app's limma code gives the same results as R limma."
                ),
            },
            {
                "Category": "Differential Analysis",
                "Question": "How is log2FC calculated for LFQ?",
                "Answer": (
                    "The matrix is already log2, so log2FC = mean log2 abundance of the test group minus mean log2 abundance of the "
                    "reference group. All tests use the same definition."
                ),
            },
            {
                "Category": "Differential Analysis",
                "Question": "What thresholds should I use?",
                "Answer": (
                    "A common starting point is `q-value < 0.05` and `|log2FC| > 1`. "
                    "For exploratory work, `q-value < 0.1` or `|log2FC| > 0.5` may be useful."
                ),
            },
            {
                "Category": "Differential Analysis",
                "Question": "What happens when I select three groups?",
                "Answer": (
                    "Choose *Two groups (pairwise)* to compare one pair (with fold change and volcano), or *All three groups* for the "
                    "limma F-test / ANOVA, which tests whether any group differs. The limma F-test table also includes each pair's "
                    "log2FC and q-value."
                ),
            },

            # ---------------- VISUALIZATIONS ----------------
            {
                "Category": "Visualizations",
                "Question": "Which result visualizations are available?",
                "Answer": (
                    "The app provides Volcano, MA plot, Differential Heatmap, Waterfall plot, and Protein Profile plots. "
                    "Protein profiles can be generated for selected proteins or top significant proteins."
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "How do I interpret the volcano plot?",
                "Answer": (
                    "The x-axis shows log2 fold-change and the y-axis shows statistical significance. Proteins farther left or right have larger changes, "
                    "and proteins higher up are more significant."
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "How do I interpret the MA plot?",
                "Answer": (
                    "The MA plot shows average abundance on the x-axis and log2 fold-change on the y-axis. "
                    "It helps reveal whether fold-change patterns depend on abundance level."
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "What is the waterfall plot?",
                "Answer": (
                    "The waterfall plot ranks proteins by log2 fold-change and highlights the strongest upregulated and downregulated proteins. "
                    "Numbered labels connect plotted points to the results table."
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "Why is my waterfall plot empty?",
                "Answer": (
                    "Possible reasons include:\n\n"
                    "- no proteins pass the selected significance threshold\n"
                    "- fold-change cutoff is too strict\n"
                    "- differential analysis has not been run\n"
                    "- group assignment is incomplete\n"
                    "- an ANOVA result is being interpreted like a pairwise result"
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "How do top protein profile plots work?",
                "Answer": (
                    "Top protein profile plots show box/strip plots for selected significant proteins. In the updated app, the on-screen top profiles are combined into one grid figure, "
                    "so zoom opens the whole grid. PDF export saves profiles as multi-panel pages."
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "Why do significance stars in protein profiles come from the selected metric?",
                "Answer": (
                    "For top protein profiles, stars are drawn from the selected significance metric, such as p-value or q-value. "
                    "This makes the profile stars match the same filter used to select the top significant proteins."
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "Can I edit the generated plots?",
                "Answer": (
                    "Yes. Download SVG files for editable figures that can be opened in Adobe Illustrator or similar vector editing tools."
                ),
            },

            # ---------------- EXPORT ----------------
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
                    "The report includes available session results such as experimental setup, sample mapping, QC plots, differential summaries, top protein tables, "
                    "volcano/MA/heatmap/waterfall plots, and selected protein profiles."
                ),
            },
            {
                "Category": "Export",
                "Question": "Why are some plots missing from my PDF or ZIP?",
                "Answer": (
                    "Exports are session-driven. Only plots generated during the current session are included. Open or run the desired plots before creating the final report."
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

            # ---------------- TROUBLESHOOTING ----------------
            {
                "Category": "Troubleshooting",
                "Question": "Why did differential analysis fail?",
                "Answer": (
                    "Common reasons include too few samples per group, non-numeric sample columns, all-missing proteins after filtering, incomplete group assignments, "
                    "or running downstream plots before differential analysis has completed."
                ),
            },
            {
                "Category": "Troubleshooting",
                "Question": "Why did batch correction fail?",
                "Answer": (
                    "Batch correction can fail if batches are empty, if too few samples are assigned, if biological group and batch "
                    "are the same split (confounded), or if ComBat is selected but the `neuroCombat` package is not installed. ComBat also "
                    "needs a complete matrix: choose an imputation method other than *None*."
                ),
            },
            {
                "Category": "Troubleshooting",
                "Question": "The Run Normalization + Imputation button is greyed out. Why?",
                "Answer": (
                    "The selected method needs R and R (or the method's R package) is not available here. The message under the button "
                    "says which. Pick a built-in method, or use the Windows/macOS/Docker build, which includes R."
                ),
            },
            {
                "Category": "Troubleshooting",
                "Question": "Statistics say 'Preprocessing settings changed'. What do I do?",
                "Answer": (
                    "You changed a step-1 setting after the last run. Click **Run Normalization + Imputation** again so the statistics "
                    "use the matrix that matches the settings shown."
                ),
            },
            {
                "Category": "Troubleshooting",
                "Question": "Why is the app slow with large LFQ files?",
                "Answer": (
                    "Large LFQ matrices can take time to preprocess, plot, cluster, and export. Filter unnecessary columns before upload, "
                    "limit top-N heatmaps/profiles, and use targeted plots when possible."
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
                placeholder="Search: LFQ, PCA, batch, volcano, MA plot, waterfall, protein profile, report...",
                key="tp_lf_faq_search",
            )

        with cat_col:
            categories = ["All"] + sorted(faqs["Category"].unique().tolist())
            selected_category = st.selectbox(
                "Category",
                categories,
                key="tp_lf_faq_category_filter",
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
            "Sample Setup",
            "Preprocessing",
            "Imputation",
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
