#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import streamlit as st
from scipy.stats import ttest_ind, ttest_rel, mannwhitneyu, wilcoxon, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

# -----------------------------------------------------------------------------
# Differential analysis
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_diffexp(df: pd.DataFrame, 
                    control_cols: list[str], 
                    test_cols: list[str],
                    control_label: str = "Reference", 
                    test_label: str = "Test",
                    test_type: str = "Student's t-test (Independent, Equal Variance)") -> pd.DataFrame:
    
    df = df.copy()
    for col in control_cols + test_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[test_label] = df[test_cols].mean(axis=1).round(3)
    df[control_label] = df[control_cols].mean(axis=1).round(3)

    df["log2FC"] = (np.log2(df[test_cols].mean(axis=1) + 1) - np.log2(df[control_cols].mean(axis=1) + 1))

    pvals = []
    
    # Check if the user selected a paired test
    is_paired = "Paired" in test_type

    for _, row in df.iterrows():
        v1_raw = row[control_cols].astype(float).values
        v2_raw = row[test_cols].astype(float).values
        
        if is_paired:
            # For paired tests, we must drop NaNs pairwise to keep samples aligned
            if len(v1_raw) == len(v2_raw):
                mask = ~np.isnan(v1_raw) & ~np.isnan(v2_raw)
                v1 = v1_raw[mask]
                v2 = v2_raw[mask]
            else:
                # If groups have different numbers of columns, paired tests are invalid
                v1, v2 = [], [] 
        else:
            # For independent tests, drop NaNs independently
            v1 = v1_raw[~np.isnan(v1_raw)]
            v2 = v2_raw[~np.isnan(v2_raw)]

        if len(v1) > 1 and len(v2) > 1:
            try:
                if test_type == "Student's t-test (Independent, Equal Variance)":
                    pval = ttest_ind(v1, v2, equal_var=True, nan_policy="omit")[1]
                    
                elif test_type == "Welch's t-test (Independent, Unequal Variance)":
                    pval = ttest_ind(v1, v2, equal_var=False, nan_policy="omit")[1]
                    
                elif test_type == "Paired t-test (Matched Samples)":
                    pval = ttest_rel(v1, v2, nan_policy="omit")[1]
                    
                elif test_type == "Mann-Whitney U (Non-parametric Independent)":
                    pval = mannwhitneyu(v1, v2, nan_policy="omit")[1]
                    
                elif test_type == "Wilcoxon Signed-Rank (Non-parametric Paired)":
                    # Wilcoxon can throw errors if all differences are exactly zero
                    pval = wilcoxon(v1, v2, nan_policy="omit")[1] 
                    
                else:
                    pval = np.nan
            except Exception:
                pval = np.nan # Catch cases where variance is exactly zero
        else:
            pval = np.nan
            
        pvals.append(pval)

    df["p-value"] = pvals
    df["adj.p-value"] = multipletests(df["p-value"].fillna(1.0), method="fdr_bh")[1]
    df["q-value"] = df["adj.p-value"]

    df["Regulation"] = "Not Significant"
    df.loc[(df["adj.p-value"] < 0.05) & (df["log2FC"] > 1), "Regulation"] = "Upregulated"
    df.loc[(df["adj.p-value"] < 0.05) & (df["log2FC"] < -1), "Regulation"] = "Downregulated"

    regulation_order = {"Upregulated": 0, "Downregulated": 1, "Not Significant": 2}
    df["reg_sort"] = df["Regulation"].map(regulation_order)
    sorted_df = df.sort_values(by=["reg_sort", "adj.p-value"], ascending=[True, True]).drop(columns="reg_sort")

    metadata_cols = ["Peptide", "Glycan", "Genes", "Proteins", "ProSites", "GlycanClass"]
    computed_cols = [control_label, test_label, "log2FC", "p-value", "q-value", "Regulation"]
    keep_cols = [c for c in metadata_cols if c in df.columns] + computed_cols + control_cols + test_cols
    
    return sorted_df[keep_cols]


def compute_diffexp_anova(df: pd.DataFrame,
                          group1_cols: list[str], group2_cols: list[str], group3_cols: list[str],
                          group1_label: str = "Group1", group2_label: str = "Group2", group3_label: str = "Group3",
                          run_tukey: bool = False, tukey_limit: int = 100) -> pd.DataFrame:
    df = df.copy()
    for col in group1_cols + group2_cols + group3_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[group1_label] = df[group1_cols].mean(axis=1).round(3)
    df[group2_label] = df[group2_cols].mean(axis=1).round(3)
    df[group3_label] = df[group3_cols].mean(axis=1).round(3)

    pvals = []
    for _, row in df.iterrows():
        v1 = row[group1_cols].dropna().astype(float)
        v2 = row[group2_cols].dropna().astype(float)
        v3 = row[group3_cols].dropna().astype(float)
        if len(v1) > 1 and len(v2) > 1 and len(v3) > 1 and np.var(v1) > 0 and np.var(v2) > 0 and np.var(v3) > 0:
            try:
                pvals.append(f_oneway(v1, v2, v3)[1])
            except Exception:
                pvals.append(np.nan)
        else:
            pvals.append(np.nan)

    df["p-value"] = pvals
    df["adj.p-value"] = multipletests(df["p-value"].fillna(1.0), method="fdr_bh")[1]
    df["q-value"] = df["adj.p-value"]
    df["Regulation"] = np.where(df["q-value"] < 0.05, "Significant", "Not Significant")

    if run_tukey:
        tukey_results = {
            f"{group1_label}_vs_{group2_label}": [],
            f"{group1_label}_vs_{group3_label}": [],
            f"{group2_label}_vs_{group3_label}": [],
        }
        sig_df = df[df["q-value"] < 0.05].copy().head(tukey_limit)
        sig_indices = sig_df.index

        progress_bar = st.progress(0, text="Running Tukey HSD comparisons...")
        for i, idx in enumerate(sig_indices):
            row = df.loc[idx]
            v1 = row[group1_cols].dropna().astype(float)
            v2 = row[group2_cols].dropna().astype(float)
            v3 = row[group3_cols].dropna().astype(float)

            if len(v1) > 1 and len(v2) > 1 and len(v3) > 1:
                try:
                    data = np.concatenate([v1, v2, v3])
                    groups = [group1_label]*len(v1) + [group2_label]*len(v2) + [group3_label]*len(v3)
                    tukey = pairwise_tukeyhsd(endog=data, groups=groups, alpha=0.05)
                    result_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])

                    for comp in tukey_results:
                        g1, g2 = comp.split("_vs_")
                        match = result_df[((result_df["group1"] == g1) & (result_df["group2"] == g2)) |
                                          ((result_df["group1"] == g2) & (result_df["group2"] == g1))]
                        pval = match["p-adj"].values[0] if not match.empty else np.nan
                        tukey_results[comp].append((idx, pval))
                except Exception:
                    pass

            progress_bar.progress((i + 1) / max(1, len(sig_indices)), text=f"Tukey test {i+1} of {len(sig_indices)}")
        progress_bar.empty()

        for comp, values in tukey_results.items():
            col = f"Tukey_{comp}_pval"
            df[col] = np.nan
            for ridx, pval in values:
                df.loc[ridx, col] = pval

    metadata_cols = ["Peptide", "Glycan", "Genes", "Proteins", "ProSites", "GlycanClass"]
    computed_cols = [group1_label, group2_label, group3_label, "p-value", "q-value", "Regulation"] + \
                    [c for c in df.columns if c.startswith("Tukey_")]
    keep_cols = [c for c in metadata_cols if c in df.columns] + computed_cols + group1_cols + group2_cols + group3_cols
    df = df.sort_values(by="adj.p-value")
    return df[keep_cols]


def apply_tukey_to_filtered(full_df: pd.DataFrame, filtered_peptides: list[str],
                            group1_cols: list[str], group2_cols: list[str], group3_cols: list[str],
                            group1_label: str, group2_label: str, group3_label: str) -> pd.DataFrame:
    subset_df = full_df[full_df["Peptide"].isin(filtered_peptides)].copy().reset_index(drop=True)
    tukey_df = compute_diffexp_anova(
        df=subset_df,
        group1_cols=group1_cols, group2_cols=group2_cols, group3_cols=group3_cols,
        group1_label=group1_label, group2_label=group2_label, group3_label=group3_label,
        run_tukey=True, tukey_limit=len(subset_df)
    )
    tukey_cols = [col for col in tukey_df.columns if col.startswith("Tukey_")]
    return tukey_df[["Peptide"] + tukey_cols].drop_duplicates(subset="Peptide")
