"""
Glycopeptide abundance normalized to a denominator measured in the same samples.

Two questions, one calculation
------------------------------
A change in a glycopeptide can come from the protein becoming more abundant, or
from the protein carrying more glycan at that site. Telling those apart needs a
denominator measured in the same samples:

  parent protein     answers "is the protein simply more abundant?"
  naked peptide      answers "did occupancy at this site change?", and is the
                     better denominator where it exists, because the same
                     sequence shares digestion efficiency and ionisation with
                     the glycosylated form and is specific to the site

Why the division happens per sample
-----------------------------------
Dividing per sample and then comparing groups gives exactly the same fold change
as dividing the two group fold changes:

    mean_test(log GP - log D) - mean_ctrl(log GP - log D)
      = [mean_test(log GP) - mean_ctrl(log GP)] - [mean_test(log D) - mean_ctrl(log D)]

The point estimate is identical to the last decimal. What differs is the spread.
A channel that was loaded slightly light, or digested a little longer, moves the
glycopeptide and its denominator together, and dividing inside the sample cancels
that. Averaging first throws the cancellation away. The gain is large when both
come from the same labelled set and nil when they do not, so this module is for
the former: one TMT set split into a glyco-enriched and a total fraction.

What the numbers do and do not mean
-----------------------------------
A glyco-enriched fraction and a total fraction sit on different scales, and the
scale factor is unknown. So the ratio itself carries an arbitrary unit and is not
an occupancy fraction. Its change between groups is what can be read, because the
unknown factor is the same in both groups and cancels. Columns here are named to
keep that distinction visible.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

__all__ = [
    "N_SEQUON",
    "find_abundance_columns",
    "channel_label",
    "match_channels",
    "strip_peptide_sequence",
    "count_sequons",
    "aggregate_peptide_abundances",
    "build_protein_denominator",
    "build_peptide_denominator",
    "normalize_to_denominator",
    "compare_groups",
]


# N-X-S/T, where X is any residue except proline.
N_SEQUON = re.compile(r"N[^P][ST]")

_ABUNDANCE_PREFIX = re.compile(r"(?i)^\s*(abundances?|intensity|intensities)\s*(\([^)]*\))?\s*:\s*")
# Case-insensitive: one export may write 131C and another 131c, and the two have
# to reduce to the same label or the channels will not pair.
_TMT_CHANNEL = re.compile(r"(?<![0-9])(12[6-9]|13[0-5])([NC]?)(?![0-9A-Za-z])", re.IGNORECASE)


def find_abundance_columns(df: pd.DataFrame) -> list[str]:
    """
    Columns holding per-sample quantitation.

    Matches the shapes Proteome Discoverer and the apps in this project produce,
    such as "Abundance: 127N" and "Abundances (Grouped): Control_1". A column
    only qualifies if it is numeric, so a text column that happens to be named
    like one is left out.
    """
    out = []
    for col in df.columns:
        name = str(col)
        looks_right = bool(_ABUNDANCE_PREFIX.match(name)) or bool(_TMT_CHANNEL.search(name))
        if looks_right and pd.api.types.is_numeric_dtype(df[col]):
            out.append(col)
    return out


def channel_label(col) -> str:
    """
    The part of a column name that identifies the sample.

    "Abundance: 127N" and "Abundances (Grouped): 127N" both reduce to "127N", so
    the same channel can be recognised across two files that were exported with
    different settings.
    """
    name = _ABUNDANCE_PREFIX.sub("", str(col)).strip()

    found = _TMT_CHANNEL.search(name)
    if found:
        return (found.group(1) + found.group(2)).upper()

    return name.lower()


def match_channels(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_columns: list[str] | None = None,
    right_columns: list[str] | None = None,
) -> tuple[dict, list, list]:
    """
    Pair the quantitation columns of two files by the sample they describe.

    Returns the pairing as {left_column: right_column}, then the left-only and
    right-only labels. The leftovers are worth showing: a channel present in one
    file and not the other cannot be used, and a pairing that covers only part of
    the set usually means the two exports do not describe the same samples.

    Pass left_columns and right_columns to use a caller's own column detection,
    so this agrees with the rest of whatever application it runs in instead of
    deciding separately.
    """
    left_names = left_columns if left_columns is not None else find_abundance_columns(left)
    right_names = right_columns if right_columns is not None else find_abundance_columns(right)

    left_cols = {channel_label(c): c for c in left_names}
    right_cols = {channel_label(c): c for c in right_names}

    shared = sorted(set(left_cols) & set(right_cols))

    pairs = {left_cols[k]: right_cols[k] for k in shared}
    only_left = sorted(set(left_cols) - set(right_cols))
    only_right = sorted(set(right_cols) - set(left_cols))

    return pairs, only_left, only_right


# Proteome Discoverer writes sequences as "[K].NVTEELKM(Oxidation)R.[G]": flanking
# residues in brackets, modifications in parentheses, and sometimes lower case for
# the flanks.
_FLANK = re.compile(r"^\s*\[?[A-Za-z-]\]?\.|\.\[?[A-Za-z-]\]?\s*$")
_MODIFICATION = re.compile(r"\([^)]*\)|\[[^\]]*\]|\{[^}]*\}")


def strip_peptide_sequence(seq) -> str:
    """
    Bare amino acid sequence, for matching one file's peptides against another's.

    Removes the flanking residues and modification annotations Proteome Discoverer
    adds, then upper-cases what is left. Anything that is not a residue letter is
    dropped, so charge states and stray punctuation do not prevent a match.
    """
    text = str(seq or "")

    text = _MODIFICATION.sub("", text)
    text = _FLANK.sub("", text)
    text = _FLANK.sub("", text)          # once more: both ends may be annotated

    return re.sub(r"[^A-Za-z]", "", text).upper()


def count_sequons(seq) -> int:
    """
    How many N-X-S/T sites the sequence contains.

    Two or more is worth knowing about: the unglycosylated form of such a peptide
    cannot say which of its sites changed, so those rows are reported but should
    not be read as site-specific.
    """
    return len(N_SEQUON.findall(strip_peptide_sequence(seq)))


def aggregate_peptide_abundances(
    peptides: pd.DataFrame,
    sequence_col: str,
    abundance_cols: list[str],
) -> pd.DataFrame:
    """
    One row per bare sequence, summing the forms that share it.

    A sequence appears several times in a peptide-level export: different charge
    states, a missed cleavage, an oxidised methionine. Those are the same piece of
    protein measured in pieces, so they are summed rather than averaged, which
    keeps the result proportional to how much of that peptide was present.
    """
    work = peptides.copy()
    work["_sequence"] = work[sequence_col].map(strip_peptide_sequence)
    work = work[work["_sequence"].str.len() > 0]

    for col in abundance_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    grouped = work.groupby("_sequence", sort=True)[abundance_cols].sum(min_count=1)

    grouped["forms_summed"] = work.groupby("_sequence", sort=True).size()
    grouped["sequons_in_peptide"] = [count_sequons(s) for s in grouped.index]

    return grouped.reset_index()


def _first_present(df: pd.DataFrame, names) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _split_ids(value) -> list[str]:
    """Identifier cells often hold several accessions separated by ; or , or |."""
    return [p.strip() for p in re.split(r"[;,|/]", str(value or "")) if p.strip()]


def build_protein_denominator(
    glyco: pd.DataFrame,
    protein: pd.DataFrame,
    channel_pairs: dict,
) -> pd.DataFrame:
    """
    A parent-protein abundance for each glycopeptide row.

    Matched on accession where both files carry one, since a gene symbol can cover
    several accessions, and on gene symbol otherwise. Where a glycopeptide matches
    more than one protein row the median is taken, and the count is reported so a
    row resting on an ambiguous match can be spotted.

    Returns the glycopeptide index, the denominator columns renamed to the
    glycopeptide's own channel names, and how the match was made.
    """
    glyco_id = _first_present(glyco, ["Proteins", "Protein", "Accession", "Master Protein Accessions"])
    prot_id = _first_present(protein, ["Accession", "Master Protein Accessions", "Proteins", "Protein"])

    glyco_gene = _first_present(glyco, ["Genes", "Gene", "GeneID"])
    prot_gene = _first_present(protein, ["Genes", "Gene", "GeneID"])

    right_cols = list(channel_pairs.values())

    # One row per identifier, so a protein listed under several accessions can be
    # found by any of them.
    def exploded(df, id_col):
        if id_col is None:
            return None
        work = df[[id_col] + right_cols].copy()
        work["_key"] = work[id_col].map(_split_ids)
        work = work.explode("_key")
        work["_key"] = work["_key"].astype(str).str.strip().str.upper()
        work = work[work["_key"].str.len() > 0]
        for col in right_cols:
            work[col] = pd.to_numeric(work[col], errors="coerce")
        counts = work.groupby("_key", sort=True).size().rename("protein_rows_matched")
        medians = work.groupby("_key", sort=True)[right_cols].median()
        return medians.join(counts)

    by_accession = exploded(protein, prot_id)
    by_gene = exploded(protein, prot_gene)

    rows = []
    for idx, row in glyco.iterrows():
        hit, how = None, "none"

        if by_accession is not None and glyco_id is not None:
            for key in _split_ids(row.get(glyco_id)):
                if key.upper() in by_accession.index:
                    hit, how = by_accession.loc[key.upper()], "accession"
                    break

        if hit is None and by_gene is not None and glyco_gene is not None:
            for key in _split_ids(row.get(glyco_gene)):
                if key.upper() in by_gene.index:
                    hit, how = by_gene.loc[key.upper()], "gene"
                    break

        record = {"_glyco_index": idx, "denominator_match": how}
        if hit is not None:
            # Renamed to the glycopeptide's channel names so the two line up.
            for left_col, right_col in channel_pairs.items():
                record[left_col] = hit[right_col]
            record["protein_rows_matched"] = int(hit["protein_rows_matched"])
        rows.append(record)

    return pd.DataFrame(rows).set_index("_glyco_index")


def build_peptide_denominator(
    glyco: pd.DataFrame,
    peptides_aggregated: pd.DataFrame,
    channel_pairs: dict,
    peptide_col: str = "Peptide",
) -> pd.DataFrame:
    """
    The unglycosylated abundance of each glycopeptide's own backbone sequence.

    Matched on the bare sequence, so a glycopeptide backbone is paired with the
    same sequence measured without glycan in the total fraction. A sequence with
    no match is left empty: unglycosylated forms of sequon peptides are often not
    detected, and most datasets will carry a denominator for only some sites.
    """
    lookup = peptides_aggregated.set_index("_sequence")
    right_cols = list(channel_pairs.values())

    rows = []
    for idx, row in glyco.iterrows():
        sequence = strip_peptide_sequence(row.get(peptide_col))
        record = {"_glyco_index": idx, "denominator_match": "none", "backbone": sequence}

        if sequence and sequence in lookup.index:
            hit = lookup.loc[sequence]
            if isinstance(hit, pd.DataFrame):      # duplicate index, take the first
                hit = hit.iloc[0]
            for left_col, right_col in channel_pairs.items():
                record[left_col] = hit[right_col]
            record["denominator_match"] = "sequence"
            record["forms_summed"] = int(hit.get("forms_summed", 0) or 0)
            record["sequons_in_peptide"] = int(hit.get("sequons_in_peptide", 0) or 0)

        rows.append(record)

    return pd.DataFrame(rows).set_index("_glyco_index")


def normalize_to_denominator(
    glyco: pd.DataFrame,
    denominator: pd.DataFrame,
    channel_cols: list[str],
) -> pd.DataFrame:
    """
    log2(glycopeptide) - log2(denominator), one channel at a time.

    Worked in log space because that is where the subtraction belongs, and because
    it keeps the result symmetric: a halving and a doubling come out the same size
    with opposite signs.

    A channel is left empty unless both values are present and above zero. Zero
    and missing are not treated as small numbers: dividing by one would invent a
    very large ratio from no measurement at all.
    """
    out = pd.DataFrame(index=glyco.index)

    for col in channel_cols:
        num = pd.to_numeric(glyco[col], errors="coerce").reindex(glyco.index)
        den = pd.to_numeric(denominator.get(col), errors="coerce")
        den = den.reindex(glyco.index) if den is not None else pd.Series(np.nan, index=glyco.index)

        usable = (num > 0) & (den > 0) & np.isfinite(num) & np.isfinite(den)
        out[col] = np.where(usable, np.log2(num.where(usable)) - np.log2(den.where(usable)), np.nan)

    return out


def compare_groups(
    ratios: pd.DataFrame,
    control_cols: list[str],
    test_cols: list[str],
    min_per_group: int = 2,
) -> pd.DataFrame:
    """
    Compare the normalized values between two groups.

    The values are already in log space, so the fold change is a difference of
    means and no further logging is done. Welch's t-test is used rather than the
    equal-variance form, because a normalized ratio inherits spread from both the
    glycopeptide and its denominator and there is no reason to expect the two
    groups to match. Benjamini-Hochberg is applied across the rows that could be
    tested.

    Rows with fewer than min_per_group usable channels on either side get a fold
    change where possible but no p-value, rather than a p-value computed from one
    measurement.
    """
    from scipy import stats

    ctrl = ratios[control_cols].astype(float)
    test = ratios[test_cols].astype(float)

    n_ctrl = ctrl.notna().sum(axis=1)
    n_test = test.notna().sum(axis=1)

    result = pd.DataFrame(index=ratios.index)
    result["normalized_log2FC"] = test.mean(axis=1) - ctrl.mean(axis=1)
    result["channels_control"] = n_ctrl
    result["channels_test"] = n_test

    testable = (n_ctrl >= min_per_group) & (n_test >= min_per_group)

    pvals = pd.Series(np.nan, index=ratios.index)
    if testable.any():
        import warnings

        with warnings.catch_warnings():
            # A row whose channels are all but identical has no spread to test.
            # SciPy warns and returns something unusable, which is filtered below.
            warnings.simplefilter("ignore")
            stat, p = stats.ttest_ind(
                test[testable], ctrl[testable],
                axis=1, equal_var=False, nan_policy="omit",
            )
        p = np.asarray(p, dtype=float)
        # Leave a row without a p-value rather than reporting one the data cannot
        # support. Two cases: SciPy could not compute one, or the channels within
        # both groups are identical, which makes the standard error zero and the
        # p-value exactly zero. The second is the dangerous one, since a p of zero
        # sorts to the top of the results while resting on no spread at all.
        spread = (
            test[testable].var(axis=1, ddof=1).fillna(0).to_numpy()
            + ctrl[testable].var(axis=1, ddof=1).fillna(0).to_numpy()
        )
        p = np.where(np.isfinite(p) & (spread > 0), p, np.nan)
        pvals.loc[testable] = p
    result["p-value"] = pvals

    # Benjamini-Hochberg over the rows that were actually tested.
    result["q-value"] = np.nan
    have_p = result["p-value"].notna()
    if have_p.any():
        p = result.loc[have_p, "p-value"].to_numpy(dtype=float)
        order = np.argsort(p)
        ranked = p[order]
        m = len(ranked)
        adjusted = ranked * m / np.arange(1, m + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        q = np.empty(m)
        q[order] = np.clip(adjusted, 0, 1)
        result.loc[have_p, "q-value"] = q

    return result