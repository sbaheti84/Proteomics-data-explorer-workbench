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


# =============================================================================
# limma (linear models + empirical Bayes moderated statistics)
#
# A pure numpy/scipy port of the parts of R limma used for group comparisons:
#   lmFit (group-means design, missing values allowed per feature)
#   contrasts.fit
#   eBayes (squeezeVar / fitFDist, moderated t and moderated F)
#   topTable-style BH adjustment (NA p-values are left out, as in p.adjust)
#
# Validated against R limma 3.58 (eBayes with trend=FALSE, robust=FALSE).
# limma >= 3.61 uses a different prior estimate *only* when features have
# different residual degrees of freedom (i.e. missing values); with complete
# data the results are identical.
#
# Input must be on the log2 scale. The wrappers below log2-transform the
# raw abundances (values <= 0 become missing).
# =============================================================================
from scipy.special import digamma as _digamma, polygamma as _polygamma
from scipy import stats as _sstats


def _trigamma(x):
    return _polygamma(1, x)


def _trigamma_inverse(x: float) -> float:
    """Solve trigamma(y) = x for y (Newton iteration, as in limma::trigammaInverse)."""
    if not np.isfinite(x) or x < 0:
        return np.nan
    if x > 1e7:
        return 1.0 / np.sqrt(x)
    if x < 1e-6:
        return 1.0 / x
    y = 0.5 + 1.0 / x
    for _ in range(51):
        tri = _trigamma(y)
        dif = tri * (1.0 - tri / x) / _polygamma(2, y)
        y = y + dif
        if -dif / y < 1e-8:
            break
    return float(y)


def _limma_fit_f_dist(x, df1):
    """Moment estimation of the scaled F prior (limma::fitFDist, no covariate)."""
    x = np.asarray(x, dtype=float)
    df1 = np.broadcast_to(np.asarray(df1, dtype=float), x.shape)
    n = x.size
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(x[0]), 0.0

    ok = np.isfinite(df1) & (df1 > 1e-15) & np.isfinite(x) & (x > -1e-15)
    nok = int(ok.sum())
    if nok == 0:
        return np.nan, np.nan
    if nok == 1:
        return float(x[ok][0]), 0.0

    x = np.maximum(x[ok], 0.0)
    d = df1[ok]
    m = np.median(x)
    if m == 0:
        m = 1.0  # limma: "more than half of residual variances are exactly zero"
    x = np.maximum(x, 1e-5 * m)

    e = np.log(x) - _digamma(d / 2.0) + np.log(d / 2.0)
    emean = e.mean()
    evar = ((e - emean) ** 2).sum() / (nok - 1)
    evar = evar - _trigamma(d / 2.0).mean()

    if evar > 0:
        df2 = 2.0 * _trigamma_inverse(evar)
        s20 = float(np.exp(emean + _digamma(df2 / 2.0) - np.log(df2 / 2.0)))
    else:
        df2 = np.inf
        s20 = float(x.mean())
    return s20, df2


def _limma_squeeze_var(var, df):
    """Posterior variances (limma::squeezeVar, no covariate, not robust)."""
    var = np.asarray(var, dtype=float).copy()
    df = np.asarray(df, dtype=float)
    var[df == 0] = 0.0
    s0, d0 = _limma_fit_f_dist(var, df)
    if np.isnan(d0):
        raise ValueError("limma could not estimate the prior degrees of freedom.")
    if np.isfinite(d0):
        var_post = (df * var + d0 * s0) / (df + d0)
    else:
        var_post = np.full_like(var, s0)
    return var_post, s0, d0


def _bh_adjust(p):
    """Benjamini-Hochberg like R p.adjust: missing p-values stay missing."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return out


def limma_group_test(log2_values: np.ndarray, groups: np.ndarray, contrasts: np.ndarray) -> dict:
    """
    limma for a group-means design: y ~ 0 + group, then contrasts, then eBayes.

    log2_values : (n_features, n_samples) log2 data, NaN = missing
    groups      : (n_samples,) integer group index 0..G-1
    contrasts   : (G, K) contrast matrix (e.g. test - reference)

    Returns a dict of arrays: logFC (n,K), t (n,K), p (n,K), AveExpr (n,),
    F, F_p (n,), df_prior, s2_prior, df_residual (n,), df_total (n,).
    """
    Y = np.asarray(log2_values, dtype=float)
    groups = np.asarray(groups)
    C = np.asarray(contrasts, dtype=float)
    if C.ndim == 1:
        C = C[:, None]
    G = C.shape[0]
    n_feat = Y.shape[0]

    # ---- lmFit: per-feature group means with missing values ----------------
    obs = np.isfinite(Y)
    n_obs = np.zeros((n_feat, G))
    sums = np.zeros((n_feat, G))
    for g in range(G):
        cols = groups == g
        n_obs[:, g] = obs[:, cols].sum(axis=1)
        sums[:, g] = np.where(obs[:, cols], Y[:, cols], 0.0).sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        coef = np.where(n_obs > 0, sums / n_obs, np.nan)
        fitted = coef[:, groups]
        resid = np.where(obs, Y - fitted, 0.0)
        rss = (resid ** 2).sum(axis=1)
        df_res = obs.sum(axis=1) - (n_obs > 0).sum(axis=1)
        sigma2 = np.where(df_res > 0, rss / np.maximum(df_res, 1), np.nan)
        stdev_unscaled = np.where(n_obs > 0, 1.0 / np.sqrt(n_obs), np.nan)

    n_obs_row = obs.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        amean = np.where(n_obs_row > 0, np.where(obs, Y, 0.0).sum(axis=1) / n_obs_row, np.nan)

    # ---- contrasts.fit (group means are uncorrelated -> orthogonal) --------
    na_coef = ~np.isfinite(coef)
    coef0 = np.where(na_coef, 0.0, coef)
    su0 = np.where(na_coef, 1e30, stdev_unscaled)
    logfc = coef0 @ C
    su_c = np.sqrt((su0 ** 2) @ (C ** 2))
    bad = su_c > 1e20
    logfc[bad] = np.nan
    su_c[bad] = np.nan

    # Design-level covariance of the contrasts (for the F-test)
    n_design = np.array([(groups == g).sum() for g in range(G)], dtype=float)
    cov_c = C.T @ np.diag(1.0 / n_design) @ C

    # ---- eBayes --------------------------------------------------------------
    s2_post, s2_prior, df_prior = _limma_squeeze_var(sigma2, df_res)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = logfc / su_c / np.sqrt(s2_post)[:, None]
    df_total = df_res + df_prior
    df_total = np.minimum(df_total, np.nansum(df_res))
    p = 2.0 * _sstats.t.sf(np.abs(t), df=df_total[:, None])

    # Moderated F across all contrasts (limma::classifyTestsF, fstat.only)
    K = C.shape[1]
    df2 = df_res + df_prior
    if K == 1:
        F = t[:, 0] ** 2
        df1 = 1
    else:
        sd = np.sqrt(np.diag(cov_c))
        cor = cov_c / np.outer(sd, sd)
        evals, evecs = np.linalg.eigh(cor)
        order = np.argsort(evals)[::-1]
        evals, evecs = evals[order], evecs[:, order]
        r = int((evals / evals[0] > 1e-8).sum())
        Q = evecs[:, :r] / np.sqrt(evals[:r]) / np.sqrt(r)
        F = ((t @ Q) ** 2).sum(axis=1)
        df1 = r
    with np.errstate(invalid="ignore"):
        if np.isfinite(df_prior):
            F_p = _sstats.f.sf(F, df1, df2)
        else:
            F_p = _sstats.chi2.sf(df1 * F, df1)

    return {
        "logFC": logfc,
        "t": t,
        "p": p,
        "AveExpr": amean,
        "F": F,
        "F_p": F_p,
        "df_prior": df_prior,
        "s2_prior": s2_prior,
        "df_residual": df_res,
        "df_total": df_total,
    }


def _log2_matrix(df: pd.DataFrame, cols: list) -> np.ndarray:
    """Raw abundances -> log2. Zero, negative or missing values become NaN."""
    vals = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(vals > 0, np.log2(vals), np.nan)


@st.cache_data(show_spinner=False)
def compute_diffexp_limma(df: pd.DataFrame,
                          group_cols: list,
                          group_labels: list,
                          control_idx: int,
                          test_idx: int,
                          use_all_groups: bool = True) -> pd.DataFrame:
    """
    Two-group comparison with limma's moderated t-test.

    Columns match compute_diffexp so every downstream plot works unchanged:
    <control mean>, <test mean>, log2FC, p-value, q-value, Regulation, plus
    AveExpr (mean log2 abundance) and t (moderated t).

    log2FC = mean(log2 test) - mean(log2 control)  (difference of log2 means).
    When use_all_groups is True and a third group exists, it is kept in the
    model so the variance is estimated from all samples (standard limma use).
    """
    df = df.copy()
    keep = [i for i, cols in enumerate(group_cols) if cols]
    if not use_all_groups:
        keep = [control_idx, test_idx]
    if control_idx not in keep or test_idx not in keep:
        raise ValueError("Both compared groups must contain samples.")

    sample_cols, groups = [], []
    for new_i, gi in enumerate(keep):
        for c in group_cols[gi]:
            sample_cols.append(c)
            groups.append(new_i)
    groups = np.array(groups)

    contrast = np.zeros(len(keep))
    contrast[keep.index(test_idx)] = 1.0
    contrast[keep.index(control_idx)] = -1.0

    res = limma_group_test(_log2_matrix(df, sample_cols), groups, contrast[:, None])

    control_cols = list(group_cols[control_idx])
    test_cols = list(group_cols[test_idx])
    control_label = group_labels[control_idx]
    test_label = group_labels[test_idx]

    for col in control_cols + test_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[test_label] = df[test_cols].mean(axis=1).round(3)
    df[control_label] = df[control_cols].mean(axis=1).round(3)

    df["log2FC"] = res["logFC"][:, 0]
    df["AveExpr"] = res["AveExpr"]
    df["t"] = res["t"][:, 0]
    df["p-value"] = res["p"][:, 0]
    df["adj.p-value"] = _bh_adjust(df["p-value"].to_numpy())
    df["q-value"] = df["adj.p-value"]

    df["Regulation"] = "Not Significant"
    df.loc[(df["adj.p-value"] < 0.05) & (df["log2FC"] > 1), "Regulation"] = "Upregulated"
    df.loc[(df["adj.p-value"] < 0.05) & (df["log2FC"] < -1), "Regulation"] = "Downregulated"

    regulation_order = {"Upregulated": 0, "Downregulated": 1, "Not Significant": 2}
    df["reg_sort"] = df["Regulation"].map(regulation_order)
    df = df.sort_values(by=["reg_sort", "adj.p-value"], ascending=[True, True],
                        na_position="last").drop(columns="reg_sort")

    metadata_cols = ["Peptide", "Glycan", "Genes", "Proteins", "ProSites", "GlycanClass"]
    computed_cols = [control_label, test_label, "log2FC", "AveExpr", "t",
                     "p-value", "q-value", "Regulation"]
    keep_cols = [c for c in metadata_cols if c in df.columns] + computed_cols + control_cols + test_cols
    df.attrs["limma"] = {"df_prior": float(res["df_prior"]), "s2_prior": float(res["s2_prior"])}
    return df[keep_cols]


@st.cache_data(show_spinner=False)
def compute_diffexp_limma_f(df: pd.DataFrame,
                            group_cols: list,
                            group_labels: list,
                            control_idx: int = 0) -> pd.DataFrame:
    """
    Multi-group comparison with limma's moderated F-test (the limma
    alternative to one-way ANOVA). Also reports, for every group, the
    log2FC and BH-adjusted moderated-t q-value against the reference group.

    The output deliberately has no plain "log2FC" column, so the app treats
    it like ANOVA results (box plots, heatmaps, gene box plots, lollipop).
    """
    df = df.copy()
    keep = [i for i, cols in enumerate(group_cols) if cols]
    if len(keep) < 2:
        raise ValueError("limma F-test needs at least two non-empty groups.")
    if control_idx not in keep:
        control_idx = keep[0]

    sample_cols, groups = [], []
    for new_i, gi in enumerate(keep):
        for c in group_cols[gi]:
            sample_cols.append(c)
            groups.append(new_i)
    groups = np.array(groups)

    others = [gi for gi in keep if gi != control_idx]
    C = np.zeros((len(keep), len(others)))
    for k, gi in enumerate(others):
        C[keep.index(gi), k] = 1.0
        C[keep.index(control_idx), k] = -1.0

    res = limma_group_test(_log2_matrix(df, sample_cols), groups, C)

    all_cols = []
    for gi in keep:
        cols = list(group_cols[gi])
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[group_labels[gi]] = df[cols].mean(axis=1).round(3)
        all_cols += cols

    ref = group_labels[control_idx]
    df["AveExpr"] = res["AveExpr"]
    df["F"] = res["F"]
    df["p-value"] = res["F_p"]
    df["adj.p-value"] = _bh_adjust(df["p-value"].to_numpy())
    df["q-value"] = df["adj.p-value"]
    df["Regulation"] = np.where(df["q-value"] < 0.05, "Significant", "Not Significant")

    contrast_cols = []
    for k, gi in enumerate(others):
        name = f"{group_labels[gi]} vs {ref}"
        df[f"log2FC ({name})"] = res["logFC"][:, k]
        df[f"q-value ({name})"] = _bh_adjust(res["p"][:, k])
        contrast_cols += [f"log2FC ({name})", f"q-value ({name})"]

    metadata_cols = ["Peptide", "Glycan", "Genes", "Proteins", "ProSites", "GlycanClass"]
    computed_cols = [group_labels[gi] for gi in keep] + ["AveExpr", "F", "p-value", "q-value", "Regulation"] + contrast_cols
    keep_cols = [c for c in metadata_cols if c in df.columns] + computed_cols + all_cols
    df = df.sort_values(by="adj.p-value", na_position="last")
    df.attrs["limma"] = {"df_prior": float(res["df_prior"]), "s2_prior": float(res["s2_prior"])}
    return df[keep_cols]
