import pandas as pd
import numpy as np

# =============================================================================
# BATCH CORRECTION UTILITY
# =============================================================================

def apply_irs_batch_correction(df, batch_dict, irs_refs, sample_cols):
    """
    Applies Internal Reference Scaling (IRS) batch correction.
    Normalizes batches based on a common reference/pool channel present in each multiplex.
    """
    corr_df = df.copy()
    
    for c in sample_cols:
        corr_df[c] = pd.to_numeric(corr_df[c], errors='coerce')

    # Extract the reference columns across all batches
    ref_cols = [irs_refs[b] for b in batch_dict.keys() if b in irs_refs]
    
    # Calculate the global average for each feature across the reference channels
    global_ref_mean = corr_df[ref_cols].mean(axis=1)

    for b_name, cols in batch_dict.items():
        if not cols:
            continue
            
        b_ref_col = irs_refs.get(b_name)
        if not b_ref_col:
            continue
            
        b_ref_intensities = corr_df[b_ref_col]
        
        # Calculate row-wise scaling factor for this batch
        # Add epsilon to avoid division by zero if a reference channel is missing a feature
        scaling_factor = global_ref_mean / (b_ref_intensities + 1e-6)
        
        # Apply scaling to all samples within this batch
        for col in cols:
            corr_df[col] = corr_df[col] * scaling_factor

    return corr_df

def apply_median_centering_batch_correction(df, batch_dict, sample_cols):
    """
    Applies median centering to align batch distributions.
    Scales features in each batch so their median matches the global median.
    """
    corr_df = df.copy()
    
    for c in sample_cols:
        corr_df[c] = pd.to_numeric(corr_df[c], errors='coerce')

    global_med = corr_df[sample_cols].median(axis=1)

    for batch_name, cols in batch_dict.items():
        if not cols:
            continue
        batch_med = corr_df[cols].median(axis=1)
        
        for col in cols:
            # Use ratio scaling to prevent negative TMT intensities
            corr_df[col] = corr_df[col] / (batch_med + 1e-6) * global_med

    return corr_df

# =============================================================================
# Confounding guard
# =============================================================================
def check_batch_group_confounding(batch_labels, group_labels):
    """Inspect whether batch and biological group can be separated at all.
 
    Returns (status, message) where status is one of:
      "ok"          — every batch contains every group
      "partial"     — some batch/group cells are empty or badly unbalanced
      "confounded"  — group is nested in batch; the design cannot be corrected
    """
    tab = pd.crosstab(pd.Series(batch_labels, name="batch"),
                      pd.Series(group_labels, name="group"))
 
    groups_per_batch = (tab > 0).sum(axis=1)
    batches_per_group = (tab > 0).sum(axis=0)
 
    if (groups_per_batch <= 1).all():
        return "confounded", (
            "Batch and biological group are completely confounded — each batch "
            "contains only one group. No batch-correction method can separate "
            "the two. Any 'correction' here removes your biological effect. "
            f"Layout:\n{tab.to_string()}"
        )
 
    if (batches_per_group <= 1).any():
        lost = list(batches_per_group[batches_per_group <= 1].index)
        return "confounded", (
            f"Group(s) {lost} appear in only one batch. Their effect cannot be "
            f"distinguished from that batch's offset.\nLayout:\n{tab.to_string()}"
        )
 
    if (tab == 0).any().any() or (tab.min().min() < 2):
        return "partial", (
            "Batch and group are unbalanced — some batch/group combinations have "
            "fewer than 2 samples. ComBat will run, but effect sizes will be "
            f"partially shrunk. Layout:\n{tab.to_string()}"
        )
 
    return "ok", f"Batch and group are adequately crossed.\n{tab.to_string()}"
 
 
# =============================================================================
# ComBat
# =============================================================================
def apply_combat_batch_correction(
    df,
    batch_dict,
    sample_cols,
    group_map=None,
    ref_batch=None,
    return_linear=True,
    strict=True,
):
    """ComBat batch correction that preserves the biological group.
 
    Parameters
    ----------
    df : DataFrame
        Feature x sample table. Sample columns hold linear abundances.
    batch_dict : dict[str, list[str]]
        Batch name -> list of sample column names.
    sample_cols : list[str]
        Samples eligible for correction.
    group_map : dict[str, str]
        Sample column name -> biological group label. REQUIRED. Omitting this
        is the bug this function exists to fix, so it is not silently optional.
    ref_batch : str, optional
        Anchor all batches to this one instead of the grand mean.
    return_linear : bool
        Round-trip back to linear scale via 2**x for backward compatibility with
        the existing downstream code. Set False once the downstream statistics
        work on the log2 scale, which is where they belong.
    strict : bool
        Refuse to run on a partially confounded design. Set False to proceed
        with a warning.
 
    Returns
    -------
    (corrected_df, info) where info carries the confounding report and warnings.
    """
    try:
        from neuroCombat import neuroCombat
    except ImportError:
        raise ImportError(
            "ComBat requires neuroCombat. Install with: pip install neuroCombat"
        )
 
    if group_map is None:
        raise ValueError(
            "apply_combat_batch_correction now requires `group_map` "
            "(sample column -> biological group label).\n\n"
            "Running ComBat without the biological covariate removes real group "
            "differences whenever group and batch are unbalanced. Build it from "
            "the global group selection, e.g.:\n\n"
            "    group_map = {\n"
            "        col: label\n"
            "        for label, cols in zip(ss['global_group_labels'],\n"
            "                               ss['global_group_samples'])\n"
            "        for col in cols\n"
            "    }\n"
        )
 
    corr_df = df.copy()
    info = {"warnings": []}
 
    # ---- order samples and collect batch labels -----------------------------
    col_order, batch_list = [], []
    for b_name, cols in batch_dict.items():
        for c in cols:
            if c in sample_cols and c not in col_order:
                col_order.append(c)
                batch_list.append(str(b_name))
 
    if len(col_order) < 2:
        raise ValueError("ComBat requires at least 2 samples assigned to batches.")
    if len(set(batch_list)) < 2:
        raise ValueError("ComBat requires at least 2 batches.")
 
    # ---- collect group labels in the same order -----------------------------
    missing_group = [c for c in col_order if c not in group_map]
    if missing_group:
        raise ValueError(
            "These batched samples have no biological group assigned: "
            f"{missing_group}. Assign every sample to a group before correcting."
        )
    group_list = [str(group_map[c]) for c in col_order]
 
    if len(set(group_list)) < 2:
        info["warnings"].append(
            "Only one biological group present; ComBat is running without a "
            "group covariate because there is nothing to preserve."
        )
        use_group = False
    else:
        use_group = True
 
    # ---- confounding check --------------------------------------------------
    status, message = check_batch_group_confounding(batch_list, group_list)
    info["confounding_status"] = status
    info["confounding_report"] = message
 
    if status == "confounded":
        raise ValueError(
            "Batch correction refused — the design cannot support it.\n\n" + message
        )
    if status == "partial":
        if strict:
            raise ValueError(
                "Batch correction halted on an unbalanced design.\n\n" + message +
                "\n\nRe-run with strict=False to proceed anyway, and report the "
                "imbalance alongside the results."
            )
        info["warnings"].append(message)
 
    # ---- log2 scale ---------------------------------------------------------
    df_expr = corr_df[col_order].apply(pd.to_numeric, errors="coerce")
    df_log = np.log2(df_expr.mask(df_expr <= 0))
    na_mask = df_log.isna()
 
    # Drop features with no usable data, then fill remaining gaps only so the
    # linear algebra runs. Fills are removed again afterwards.
    usable = df_log.notna().sum(axis=1) >= max(2, len(col_order) // 2)
    if not usable.any():
        raise ValueError(
            "No features have enough non-missing values for ComBat. "
            "Apply presence filtering first."
        )
    if (~usable).any():
        info["warnings"].append(
            f"{int((~usable).sum())} features had too few values for ComBat and "
            "were left uncorrected."
        )
 
    work = df_log.loc[usable].copy()
    # Fill within batch, so an absent batch is not handed another batch's median.
    for b in set(batch_list):
        b_cols = [c for c, bb in zip(col_order, batch_list) if bb == b]
        work[b_cols] = work[b_cols].apply(lambda r: r.fillna(r.median()), axis=1)
    work = work.apply(lambda r: r.fillna(r.median()), axis=1)
    work = work.fillna(work.stack().min() if work.notna().any().any() else 0.0)
 
    # ---- run ----------------------------------------------------------------
    covars = pd.DataFrame({"batch": batch_list}, index=col_order)
    categorical = None
    if use_group:
        covars["group"] = group_list
        categorical = ["group"]
 
    kwargs = {}
    if ref_batch is not None:
        kwargs["ref_batch"] = str(ref_batch)
 
    try:
        res = neuroCombat(
            dat=work,
            covars=covars,
            batch_col="batch",
            categorical_cols=categorical,
            **kwargs,
        )
    except np.linalg.LinAlgError as e:
        raise ValueError(
            "ComBat could not fit the model — the batch/group design matrix is "
            "singular, which means batch and group are not separable.\n\n"
            f"{message}\n\nUnderlying error: {e}"
        )
 
    corrected_log = pd.DataFrame(res["data"], index=work.index, columns=col_order)
 
    # ---- reassemble, restoring true missingness -----------------------------
    out_log = df_log.copy()
    out_log.loc[work.index, col_order] = corrected_log
    out_log[na_mask] = np.nan
 
    corr_df[col_order] = np.exp2(out_log) if return_linear else out_log
 
    info["n_features_corrected"] = int(usable.sum())
    info["group_covariate_used"] = use_group
    info["ref_batch"] = ref_batch
    return corr_df, info