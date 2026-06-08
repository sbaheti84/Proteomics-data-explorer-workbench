#!/usr/bin/env python
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd


def faq_section():
    with st.expander("Help, Quick Start & FAQ", expanded=False):

        # =============================
        # QUICK START
        # =============================
        st.markdown("## Quick Start")
        st.caption(
            "A practical guide for extracting and reviewing target fragment ions from mzML or MGF files."
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(
                """
                ### 1. Upload Raw Data
                Upload one raw MS file:

                - `.mzML`
                - `.mgf`

                mzML supports MS1, MS2, or both. MGF is usually MS2-only.
                """
            )

        with c2:
            st.markdown(
                """
                ### 2. Define Targets
                Enter target m/z values in the sidebar.

                Optional:
                - Upload ion annotation table
                - Select MS level
                - Adjust ppm tolerance
                - Set noise filter
                """
            )

        with c3:
            st.markdown(
                """
                ### 3. Review & Export
                Use the workflow panels to review:

                - File metadata
                - Raw spectra
                - Extracted ion matrix
                - PI summary tables
                - RT profiles
                - CSV/JSON exports
                """
            )

        st.info(
            "Recommended workflow: first inspect file metadata, then preview representative spectra, then run targeted extraction and review the ion summary before exporting."
        )

        st.markdown("### How to Interpret the Main Outputs")

        i1, i2 = st.columns(2)

        with i1:
            st.markdown(
                """
                #### Data Integrity & Overview
                Confirms the file was read correctly and shows basic metadata such as total spectra, MS levels, and RT range.

                Use this first to decide whether to extract from MS1, MS2, or both.
                """
            )

            st.markdown(
                """
                #### Raw Spectral Profiling
                Lets you preview representative spectra before extraction.

                This helps confirm that the target ions are present in the expected m/z region.
                """
            )

        with i2:
            st.markdown(
                """
                #### Targeted Extraction
                Extracts target m/z intensities across spectra using the selected ppm tolerance and peak-picking rule.

                The output matrix includes scan metadata, intensities, and detection flags.
                """
            )

            st.markdown(
                """
                #### PI Summary
                Summarizes detection frequency, confidence score, RT profiles, co-detection, and run-level metrics.

                This is the most useful section for quick biological interpretation.
                """
            )

        st.divider()

        # =============================
        # FAQ DATA
        # =============================
        faqs = pd.DataFrame([
            {
                "Category": "General",
                "Question": "What does the Fragment Ion Extractor do?",
                "Answer": (
                    "The Fragment Ion Extractor scans mzML or MGF files for user-defined target m/z values. "
                    "It extracts ion intensities across spectra, applies optional noise filtering, summarizes detection confidence, "
                    "and exports scan-level and ion-level result tables."
                ),
            },
            {
                "Category": "General",
                "Question": "Who is this tool useful for?",
                "Answer": (
                    "The tool is useful when a PI or analyst wants to check whether specific diagnostic, glycan, oxonium, "
                    "or fragment ions are detectable across spectra without manually opening every raw file."
                ),
            },
            {
                "Category": "Input Files",
                "Question": "What file formats are supported?",
                "Answer": (
                    "The app supports `.mzML` and `.mgf` files. mzML files may contain MS1 and MS2 scans. "
                    "MGF files usually contain MS2 spectra only."
                ),
            },
            {
                "Category": "Input Files",
                "Question": "Do I need centroided data?",
                "Answer": (
                    "Centroided data is usually preferred for targeted fragment extraction because peak picking is cleaner and faster. "
                    "Profile-mode data may still work, but it can produce broader peak windows and larger files."
                ),
            },
            {
                "Category": "Input Files",
                "Question": "What is the optional ion annotation table?",
                "Answer": (
                    "The optional annotation table maps target ions to readable labels. It should contain two columns:\n\n"
                    "- `Ion`\n"
                    "- `Annotation`\n\n"
                    "This is useful for labeling ions such as oxonium ions, Y ions, diagnostic fragments, or custom target ions."
                ),
            },
            {
                "Category": "MS Level Selection",
                "Question": "What does MS1, MS2, or Both mean?",
                "Answer": (
                    "For mzML files, you can choose which spectra are searched:\n\n"
                    "- **MS1:** precursor-level survey scans\n"
                    "- **MS2:** fragmentation spectra\n"
                    "- **Both:** all available spectra\n\n"
                    "For fragment or oxonium ion searches, MS2 is usually the most relevant choice."
                ),
            },
            {
                "Category": "Extraction Parameters",
                "Question": "What is mass tolerance in ppm?",
                "Answer": (
                    "Mass tolerance defines the search window around each target m/z. For example, 10 ppm at m/z 1000 gives "
                    "a search window of approximately ±0.01 Da. Smaller values are stricter; larger values are more permissive."
                ),
            },
            {
                "Category": "Extraction Parameters",
                "Question": "Should I use closest m/z or maximum intensity?",
                "Answer": (
                    "Use **closest m/z** when you trust mass accuracy and want the peak nearest to the target. "
                    "Use **maximum intensity** when you want the strongest peak within the ppm window, which may be helpful for noisy data."
                ),
            },
            {
                "Category": "Extraction Parameters",
                "Question": "What is the intensity noise threshold?",
                "Answer": (
                    "The noise threshold removes weak peaks based on a percentage of the base peak intensity (BPI) in each spectrum. "
                    "Signals below this threshold are set to missing in the `_nf` noise-filtered columns."
                ),
            },
            {
                "Category": "Extraction Parameters",
                "Question": "What is the difference between raw and noise-filtered outputs?",
                "Answer": (
                    "Raw columns report detected intensities inside the ppm window. Noise-filtered columns, marked with `_nf`, "
                    "keep only signals that pass the selected BPI-based noise threshold."
                ),
            },
            {
                "Category": "Results",
                "Question": "What does the extracted per-spectrum table show?",
                "Answer": (
                    "The per-spectrum table contains scan-level metadata and extracted target ion intensities. It includes fields such as "
                    "scan number, retention time, MS level, base peak intensity, precursor m/z, and target ion columns."
                ),
            },
            {
                "Category": "Results",
                "Question": "What does the quantitative pivot matrix show?",
                "Answer": (
                    "The pivot matrix summarizes target ion intensities by retention time and scan number. This is the main export table "
                    "for downstream review, filtering, and plotting."
                ),
            },
            {
                "Category": "Results",
                "Question": "How do I interpret the status flags?",
                "Answer": (
                    "The matrix includes several flags:\n\n"
                    "- `flag_all_ions_present`: all selected target ions were detected in that scan\n"
                    "- `flag_noise_filtered`: at least one detected signal was removed by the noise filter\n"
                    "- `raw_all_but_nf_filters_some`: all ions were present in raw data, but the noise filter removed at least one"
                ),
            },
            {
                "Category": "PI Summary",
                "Question": "What is the ion summary table?",
                "Answer": (
                    "The ion summary table reports detection percentage, mean intensity, median intensity, maximum intensity, CV, "
                    "noise-filtered percentage, confidence score, and confidence label for each target ion."
                ),
            },
            {
                "Category": "PI Summary",
                "Question": "How is the confidence score calculated?",
                "Answer": (
                    "The confidence score starts from the detection frequency and is penalized when signals are frequently suppressed by "
                    "the noise filter. Higher scores indicate more consistent and robust detection."
                ),
            },
            {
                "Category": "PI Summary",
                "Question": "How should I interpret High, Moderate, and Low confidence ions?",
                "Answer": (
                    "High-confidence ions are detected consistently and survive noise filtering. Moderate-confidence ions are detected but less consistently. "
                    "Low-confidence ions may be weak, rare, noisy, or absent in the selected spectra."
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "What does detection frequency show?",
                "Answer": (
                    "Detection frequency shows the percentage of spectra where each target ion was detected. It helps quickly identify ions "
                    "that are consistently present versus rarely observed."
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "What does the co-detection heatmap show?",
                "Answer": (
                    "The co-detection heatmap shows whether target ions tend to appear together in the same scans. Values close to 1 mean "
                    "two ions are usually co-detected. Values near 0 mean they rarely appear together."
                ),
            },
            {
                "Category": "Visualizations",
                "Question": "What do retention time profiles show?",
                "Answer": (
                    "Retention time profiles plot ion intensity across chromatography time. They help determine whether target ions form "
                    "clear elution peaks or appear as scattered background signal."
                ),
            },
            {
                "Category": "Export",
                "Question": "What files can I download?",
                "Answer": (
                    "The app exports:\n\n"
                    "- Matrix with flags CSV\n"
                    "- Per-spectrum extraction CSV\n"
                    "- Run parameter JSON\n"
                    "- Ion summary CSV\n"
                    "- Run summary CSV"
                ),
            },
            {
                "Category": "Export",
                "Question": "Why is the parameter JSON useful?",
                "Answer": (
                    "The JSON file records the exact ppm tolerance, target ions, noise filter, MS level, and peak-picking settings used for extraction. "
                    "This makes the analysis reproducible."
                ),
            },
            {
                "Category": "Troubleshooting",
                "Question": "Why do I see no extracted signals?",
                "Answer": (
                    "Common reasons include:\n\n"
                    "- Wrong MS level selected\n"
                    "- Target m/z values are not present\n"
                    "- PPM tolerance is too strict\n"
                    "- Data are profile-mode or poorly centroided\n"
                    "- Noise threshold is too high"
                ),
            },
            {
                "Category": "Troubleshooting",
                "Question": "Why are many values missing in noise-filtered columns?",
                "Answer": (
                    "The selected noise threshold may be too strict. Try lowering the BPI percentage or reviewing raw columns first to see whether "
                    "the signal is present before filtering."
                ),
            },
            {
                "Category": "Troubleshooting",
                "Question": "Why does MGF show fewer metadata fields than mzML?",
                "Answer": (
                    "MGF files usually contain simplified MS2 spectra and fewer instrument metadata fields. mzML files preserve richer metadata, "
                    "including MS levels, precursor information, scan IDs, and chromatographic details."
                ),
            },
        ])

        # =============================
        # CLEAN SEARCHABLE FAQ UI
        # =============================
        st.markdown("## Searchable FAQ")

        search_col, category_col = st.columns([0.65, 0.35])

        with search_col:
            search = st.text_input(
                "Search FAQ",
                placeholder="Search: ppm, MS2, BPI, noise filter, confidence, export...",
                key="faq_search_box",
            )

        with category_col:
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
            "MS Level Selection",
            "Extraction Parameters",
            "Results",
            "PI Summary",
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
