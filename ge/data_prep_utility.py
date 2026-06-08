#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
import streamlit as st
from neuroCombat import neuroCombat

# Import the classification helper
from ge.misc_utility import *

@st.cache_data(show_spinner=False)
def read_data_file(uploaded_file) -> pd.DataFrame:
    """Smart reader that handles CSV, TSV, TXT, XLSX, and XLS formats."""
    name = uploaded_file.name.lower()
    if name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif name.endswith(('.tsv', '.txt')):
        return pd.read_csv(uploaded_file, sep='\t')
    elif name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)

def clean_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns.astype(str)
                  .str.replace("\ufeff", "", regex=False)
                  .str.replace("\xa0", " ", regex=False)
                  .str.strip()
                  .str.replace(r"\s+", " ", regex=True))
    return df

def clean_key_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = (df[c].astype(str)
                     .str.replace("\ufeff", "", regex=False)
                     .str.replace("\xa0", " ", regex=False)
                     .str.strip()
                     .str.replace(r"\s+", " ", regex=True))
    return df

# -----------------------------------------------------------------------------
# Merge pGlyco + PD
# -----------------------------------------------------------------------------
def merge_pglyco_pd(pglyco_df: pd.DataFrame, pd_df: pd.DataFrame, version_label: str, species: str) -> pd.DataFrame:
    def parse_glycan_string(s, n_parts):
        if pd.isna(s):
            return [0] * n_parts
        s = str(s).strip()
        nums = re.findall(r"\d+", s)
        if len(nums) == n_parts:
            return [int(x) for x in nums]
        elif " " in s:
            try:
                vals = [int(x) for x in s.split()]
                while len(vals) < n_parts:
                    vals.append(0)
                return vals[:n_parts]
            except Exception:
                return [0] * n_parts
        return [0] * n_parts

    pglyco_df = clean_colnames(pglyco_df)
    pd_df     = clean_colnames(pd_df)

    if species.lower() == "mouse" or "Glycan(H,N,A,G,F)" in pglyco_df.columns:
        glycan_col = "Glycan(H,N,A,G,F)"
        corr_col = "CorrectedGlycan(H,N,A,G,F)"

        if corr_col in pglyco_df.columns:
            pglyco_df["__glycan_source__"] = np.where(
                pglyco_df[corr_col].astype(str).str.contains(r"\d", na=False),
                pglyco_df[corr_col],
                pglyco_df[glycan_col]
            )
            glycan_col = "__glycan_source__"

        glycan_expanded = pglyco_df[glycan_col].apply(lambda s: parse_glycan_string(s, 5))
        glycan_df = pd.DataFrame(glycan_expanded.tolist(), columns=["H", "N", "A", "G", "F"])
        pglyco_df = pd.concat([pglyco_df.reset_index(drop=True), glycan_df], axis=1)
        pglyco_df["Glycan"] = pglyco_df[["H", "N", "A", "G", "F"]].astype(str).agg(" ".join, axis=1)
    else:
        glycan_col = "Glycan(H,N,A,F)"
        corr_col = "CorrectedGlycan(H,N,A,F)"

        if corr_col in pglyco_df.columns:
            pglyco_df["__glycan_source__"] = np.where(
                pglyco_df[corr_col].astype(str).str.contains(r"\d", na=False),
                pglyco_df[corr_col],
                pglyco_df[glycan_col]
            )
            glycan_col = "__glycan_source__"

        glycan_expanded = pglyco_df[glycan_col].apply(lambda s: parse_glycan_string(s, 4))
        glycan_df = pd.DataFrame(glycan_expanded.tolist(), columns=["H", "N", "A", "F"])
        pglyco_df = pd.concat([pglyco_df.reset_index(drop=True), glycan_df], axis=1)
        pglyco_df["Glycan"] = pglyco_df[["H", "N", "A", "F"]].astype(str).agg(" ".join, axis=1)

    glycan_orig = "Glycan"

    pglyco_df = clean_key_cols(pglyco_df, ["RawName", "Peptide", "Genes", "Proteins", "ProSites"])
    pd_df     = clean_key_cols(pd_df, ["Spectrum File"])

    if "Scan" in pglyco_df.columns:
        pglyco_df["Scan"] = pd.to_numeric(pglyco_df["Scan"], errors="coerce").astype("Int64")
    if "First Scan" in pd_df.columns:
        pd_df["First Scan"] = pd.to_numeric(pd_df["First Scan"], errors="coerce").astype("Int64")

    if "Spectrum File" in pd_df.columns:
        pd_df["Spectrum File"] = pd_df["Spectrum File"].astype(str).str.replace(r"\.raw$", "", regex=True).str.strip()
    if "RawName" in pglyco_df.columns:
        pglyco_df["RawName"] = pglyco_df["RawName"].astype(str).str.strip()

    pglyco_df["CompositeAnnotation"] = (
        pglyco_df["Peptide"].astype(str) + "--" +
        pglyco_df[glycan_orig].astype(str) + "--" +
        pglyco_df["Genes"].astype(str) + "--" +
        pglyco_df["Proteins"].astype(str) + "--" +
        pglyco_df["ProSites"].astype(str)
    )

    if "TotalScore" in pglyco_df.columns:
        pglyco_df["__TotalScore__"] = pd.to_numeric(pglyco_df["TotalScore"], errors="coerce")
        pglyco_df = pglyco_df.sort_values(
            by=["RawName", "Scan", "__TotalScore__"],
            ascending=[True, True, False],
            kind="mergesort"
        )
        pglyco_df = pglyco_df.drop_duplicates(subset=["RawName", "Scan"], keep="first").drop(columns=["__TotalScore__"], errors="ignore")
    else:
        pglyco_df = pglyco_df.drop_duplicates(subset=["RawName", "Scan"], keep="first")

    pglyco_df["Scan"] = pd.to_numeric(pglyco_df["Scan"], errors="coerce").astype("Int64")
    pd_df["First Scan"] = pd.to_numeric(pd_df["First Scan"], errors="coerce").astype("Int64")

    abundance_cols = [c for c in pd_df.columns if c.startswith("Abundance") or re.match(r"(?i)^abundance[s]?:\s*\d+", str(c))]
    abundance_cols = [c for c in abundance_cols if not re.search(r"(?i)normalized|scaled|s/n|corr", str(c))]

    if abundance_cols:
        pd_df = pd_df.groupby(["Spectrum File", "First Scan"], as_index=False)[abundance_cols].sum()

    merged = pd.merge(
        pglyco_df,
        pd_df,
        left_on=["RawName", "Scan"],
        right_on=["Spectrum File", "First Scan"],
        how="left"
    )

    if "CompositeAnnotation" not in merged.columns:
        raise KeyError("CompositeAnnotation lost during merging.")

    max_scans = 5
    if "GlySpec" in merged.columns:
        scan_lists = merged.groupby("CompositeAnnotation")["GlySpec"].apply(list).reset_index()
        scan_expanded = scan_lists["GlySpec"].apply(lambda x: pd.Series(x[:max_scans]))
        scan_expanded.columns = [f"Scan_{i+1}" for i in range(scan_expanded.shape[1])]
        scan_expanded["CompositeAnnotation"] = scan_lists["CompositeAnnotation"]
    else:
        scan_expanded = pd.DataFrame(columns=["CompositeAnnotation"])

    if abundance_cols:
        aggregated = merged.groupby("CompositeAnnotation", dropna=False)[abundance_cols].sum().reset_index()
        aggregated[abundance_cols] = aggregated[abundance_cols].astype(float).round(2)
    else:
        aggregated = merged[["CompositeAnnotation"]].copy()

    split_cols = aggregated["CompositeAnnotation"].str.split("--", n=4, expand=True)
    split_cols.columns = ["Peptide", glycan_orig, "Genes", "Proteins", "ProSites"]
    split_cols["CompositeAnnotation"] = aggregated["CompositeAnnotation"]
    
    split_cols["GlycanClass"] = split_cols[glycan_orig].apply(classify_glycan)

    result = pd.concat([split_cols, aggregated[abundance_cols] if abundance_cols else pd.DataFrame()], axis=1)
    if not scan_expanded.empty:
        result = pd.merge(result, scan_expanded, on="CompositeAnnotation", how="left")

    result = result.drop(columns=["CompositeAnnotation"], errors="ignore")
    return result

def apply_combat_correction(df, sample_cols, batch_mapping):
    """
    Applies empirical Bayes batch correction using neuroCombat.
    """
    # 1. Isolate the abundance matrix
    data_matrix = df[sample_cols].copy()
    
    # 2. Handle missing values (ComBat requires complete cases or imputed data)
    # Simple median imputation for the sake of the algorithm, or drop NAs
    data_matrix = data_matrix.fillna(data_matrix.median()) 
    
    # 3. Prepare batch array
    # Ensure the order matches the sample_cols
    batch_array = batch_mapping.set_index("Sample").loc[sample_cols]["Batch"].values
    
    # 4. Run neuroCombat
    combat_results = neuroCombat(
        dat=data_matrix,
        covars=pd.DataFrame({"Batch": batch_array}),
        batch_col="Batch"
    )
    
    # 5. Reintegrate corrected values back into the dataframe
    corrected_df = df.copy()
    corrected_df[sample_cols] = combat_results["data"]
    
    return corrected_df