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

def apply_combat_batch_correction(df, batch_dict, sample_cols):
    """
    Applies ComBat batch correction using neuroCombat.
    """
    try:
        from neuroCombat import neuroCombat
    except ImportError:
        raise ImportError(
            "ComBat requires neuroCombat. Install with: pip install neuroCombat"
        )

    corr_df = df.copy()

    # Keep only selected/assigned samples that are in batch_dict
    col_order = []
    batch_list = []

    for b_name, cols in batch_dict.items():
        for c in cols:
            if c in sample_cols and c not in col_order:
                col_order.append(c)
                batch_list.append(b_name)

    if len(col_order) < 2:
        raise ValueError("ComBat requires at least 2 samples assigned to batches.")

    if len(set(batch_list)) < 2:
        raise ValueError("ComBat requires at least 2 batches.")

    df_expr = corr_df[col_order].apply(pd.to_numeric, errors="coerce")

    df_log = np.log2(df_expr.replace(0, np.nan))

    row_medians = df_log.median(axis=1)
    df_log = df_log.apply(lambda row: row.fillna(row_medians[row.name]), axis=1)

    global_fill = df_log.min().min()
    if pd.isna(global_fill):
        global_fill = 0
    df_log = df_log.fillna(global_fill)

    covars = pd.DataFrame({
        "batch": batch_list
    }, index=col_order)

    combat_res = neuroCombat(
        dat=df_log,
        covars=covars,
        batch_col="batch"
    )

    corrected_log = pd.DataFrame(
        combat_res["data"],
        index=df_log.index,
        columns=col_order
    )

    corr_df[col_order] = np.exp2(corrected_log)

    return corr_df