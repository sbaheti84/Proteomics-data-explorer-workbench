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
    "describe_abundance_columns",
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


# Columns that name a channel but hold something other than its abundance.
# Proteome Discoverer writes "Abundances (Grouped) CV [%]: 126" alongside
# "Abundances (Grouped): 126", and both carry the channel in the name. Taking the
# first for the second would quietly use a coefficient of variation as a
# quantity, and because both reduce to the same channel label, whichever came
# last would win.
_NOT_QUANTITATION = re.compile(
    r"(?i)(\bcv\b|\[%\]|percent|occupancy|\bratio\b|\bcount\b|^#|found\s+in\s+sample"
    r"|\bscore\b|\bp[-_. ]?value\b|\bq[-_. ]?value\b|log2|fold|theo\.|\bpep\b)"
)

# A sample column cannot be relied on to carry its reporter channel, or the word
# abundance, or any other recognisable marker: an analysed table is usually
# renamed to whatever the sample sheet called the samples. So there has to be a
# way to find the quantitation columns without reading the names at all, and that
# means being able to rule out, by name, everything a results table puts beside
# them. This list is deliberately long. It is only consulted on that last-resort
# path, so a name it wrongly catches costs nothing where the ordinary detection
# works, and the module always reports which columns it took.
_STATISTIC_OR_ANNOTATION = re.compile(
    r"(?i)("
    r"\bcv\b|\[%\]|percent|occupancy|\bratio\b|\bcount\b|^#|found\s+in\s+sample"
    r"|\bscore\b|log2|log10|neglog|\bfold\b|^fc\b|\bfc$|\bfc[-_. /]|theo\.|\bpep\b"
    r"|\bt[-_. ]?test\b|\banova\b|\bwelch\b|\blimma\b"
    r"|\bp[-_. ]?val|\bq[-_. ]?val|\bfdr\b|\badj|\bsignif|\balpha\b"
    r"|\bmean\b|\bmedian\b|\bsd\b|\bstdev\b|\bstd\b|\bsem\b|\bse\b|\bvariance\b"
    r"|\bz[-_. ]?score\b|\bcorrel|\brho\b|\bslope\b|\bintercept\b"
    r"|confid|\bprob|\berror\b|\bmissed\b|cleavage|\bcharge\b|\bm/?z\b"
    r"|retention|\brt\b|\bmass\b|\bda\b|\bppm\b|\bpsm|\bspectr|\bscan\b"
    r"|sequence|modificat|\bposition\b|\bstart\b|\bend\b|\blength\b"
    r"|accession|\bprotein|\bgene|\bentry\b|descript|\bchecked\b|\bquan\s+info\b"
    r"|\bcoverage\b|\bunique\b|\brank\b|\bindex\b|\bid$|\bid[-_. ]"
    r")"
)


def find_abundance_columns(df: pd.DataFrame, by_name_only: bool = False) -> list[str]:
    """
    Columns holding per-sample quantitation.

    First by name, which recognises the shapes Proteome Discoverer and the apps in
    this project produce: "Abundance: 127N", "Abundances (Grouped): Control_1",
    and the renamed sample columns of an analysed table, "C1-2180-126".

    A column qualifies only if it is numeric and its name does not say it holds
    something else, which keeps the CV, occupancy and count columns that sit
    beside the real ones from being mistaken for them.

    Where that finds nothing, the names carry no marker at all, which is the
    ordinary case for a table renamed to its sample sheet: "C1", "MPI_3",
    "Patient 4". Then the columns are found by what they contain instead.
    Pass by_name_only=True to skip that and see only what the names give.
    """
    out = []
    for col in df.columns:
        name = str(col)

        if _NOT_QUANTITATION.search(name):
            continue

        looks_right = bool(_ABUNDANCE_PREFIX.match(name)) or bool(_TMT_CHANNEL.search(name))
        if looks_right and pd.api.types.is_numeric_dtype(df[col]):
            out.append(col)

    if out or by_name_only:
        return out

    return _find_abundance_columns_by_content(df)


# A plex is never one or two channels, so a smaller group of look-alike columns is
# more likely a handful of statistics than a set of samples.
_MIN_SAMPLE_COLUMNS = 3


def _find_abundance_columns_by_content(df: pd.DataFrame) -> list[str]:
    """
    Quantitation columns found without reading their names for meaning.

    Two properties separate samples from everything else in a results table.
    They hold abundances, so the values are non-negative and spread over orders
    of magnitude rather than sitting in a narrow band. And they are repeats of
    one measurement across channels, so they resemble each other: their typical
    values land within a factor of a few, while a fold change, a p-value and a
    significance score each sit somewhere different entirely.

    So: drop anything whose name says it is a statistic or an annotation, keep
    the numeric remainder that looks like an abundance, then take the largest
    group of columns of similar magnitude. A group below _MIN_SAMPLE_COLUMNS is
    rejected rather than guessed at.
    """
    candidates = {}
    for col in df.columns:
        if _STATISTIC_OR_ANNOTATION.search(str(col)):
            continue

        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(values) < 2 or (values < 0).any():
            continue

        positive = values[values > 0]
        if positive.empty:
            continue

        # A p-value, a q-value or a proportion never leaves [0, 1]. An abundance
        # does, on a raw scale and on a log2 scale alike, so this rules out that
        # whole family without assuming anything about how wide the real spread
        # is. Width is not tested: log2-transformed abundances sit in a band
        # narrower than any threshold worth setting.
        if float(values.max()) <= 1.0:
            continue

        candidates[col] = float(np.median(positive))

    if len(candidates) < _MIN_SAMPLE_COLUMNS:
        return []

    # Split the columns into clusters of similar magnitude by sorting them and
    # cutting wherever the step up to the next one exceeds a factor of 32. That
    # factor is wider than the channels of one plex differ from each other and far
    # narrower than the gap between an abundance and any statistic beside it, so
    # the samples come out as one cluster whatever their number.
    ordered = sorted(candidates.items(), key=lambda kv: kv[1])
    clusters = [[ordered[0]]]
    for previous, current in zip(ordered, ordered[1:]):
        if current[1] / previous[1] > 32.0:
            clusters.append([current])
        else:
            clusters[-1].append(current)

    groups = [c for c in clusters if len(c) >= _MIN_SAMPLE_COLUMNS]
    if not groups:
        return []

    # Of the groups big enough to be a plex, take the one holding the largest
    # numbers. In a results table the abundances are the large values and the
    # statistics beside them are small, on a raw scale and a log2 scale alike, so
    # magnitude separates the two more reliably than how many columns each has:
    # a table can carry more statistics than samples.
    best = max(groups, key=lambda g: (np.median([v for _, v in g]), len(g)))

    chosen = {col for col, _ in best}
    return [col for col in df.columns if col in chosen]


def describe_abundance_columns(df: pd.DataFrame) -> tuple[list[str], str]:
    """
    The quantitation columns, and how certain that answer is.

    Returns "name" where the column names identified themselves, which can be
    relied on; "content" where the names said nothing and the columns were picked
    by what they hold, which is a suggestion and can be wrong when a table
    carries more unlabelled statistics than samples; and "none" where neither
    found anything. A caller with a user in front of it should show the columns
    and let them be corrected whenever this says "content".
    """
    by_name = find_abundance_columns(df, by_name_only=True)
    if by_name:
        return by_name, "name"

    by_content = _find_abundance_columns_by_content(df)
    return (by_content, "content") if by_content else ([], "none")


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
    allow_positional: bool = False,
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

    Three things are tried, in descending order of how much they can be trusted.
    The reporter channel where the names carry one, since that identifies the
    sample regardless of what the column was renamed to. The column name itself
    where both files were renamed from the same sample sheet, which is an exact
    match and just as safe. And, only with allow_positional, the order the
    columns appear in, which is a guess: it is right when both files came out of
    one pipeline and wrong without warning when a channel was dropped from one of
    them. Leave it off unless whoever is running this has confirmed the two files
    list the same samples in the same order.
    """
    left_names = left_columns if left_columns is not None else find_abundance_columns(left)
    right_names = right_columns if right_columns is not None else find_abundance_columns(right)

    left_cols = {channel_label(c): c for c in left_names}
    right_cols = {channel_label(c): c for c in right_names}

    shared = sorted(set(left_cols) & set(right_cols))

    pairs = {left_cols[k]: right_cols[k] for k in shared}
    only_left = sorted(set(left_cols) - set(right_cols))
    only_right = sorted(set(right_cols) - set(left_cols))

    extra = _pair_bare_masses(left_cols, right_cols, only_left, only_right)
    for left_label, right_label in extra.items():
        pairs[left_cols[left_label]] = right_cols[right_label]
        only_left.remove(left_label)
        only_right.remove(right_label)

    if allow_positional and not pairs and len(left_names) == len(right_names):
        # Nothing in the names lined up. Fall back to the order the columns appear
        # in, which only the caller can vouch for. Equal counts are required: an
        # unequal pair of files cannot be aligned this way at all, and quietly
        # truncating the longer one would pair the wrong samples.
        pairs = dict(zip(list(left_names), list(right_names)))
        only_left, only_right = [], []

    return pairs, only_left, only_right


def _pair_bare_masses(left_cols, right_cols, only_left, only_right) -> dict:
    """
    Pair a channel written without its light/heavy letter against the one
    channel that carries it.

    One file can name the last reagent of a 16-plex "134" and another "134N",
    and the two are the same sample. Pairing them is only safe where there is
    no choice to make, so a mass is paired this way only when one side has the
    bare number, the other side has exactly one lettered form of that mass, and
    neither label was already paired. Where both the N and the C form are
    waiting, nothing is guessed.
    """
    matched = {}

    def by_mass(labels):
        out = {}
        for label in labels:
            found = _TMT_CHANNEL.search(label)
            if found:
                out.setdefault(found.group(1), []).append(label)
        return out

    for bare_side, lettered_side, flip in (
        (only_left, only_right, False),
        (only_right, only_left, True),
    ):
        lettered_by_mass = by_mass(lettered_side)
        for label in bare_side:
            if label in matched or label in matched.values():
                continue
            found = _TMT_CHANNEL.search(label)
            if not found or found.group(2):            # already carries N or C
                continue
            candidates = [
                c for c in lettered_by_mass.get(found.group(1), [])
                if c not in matched and c not in matched.values()
            ]
            if len(candidates) != 1:                   # 0 to pair, or an ambiguous 2
                continue
            other = candidates[0]
            matched[other if flip else label] = label if flip else other

    # keyed left -> right, which is the direction the caller indexes
    return {k: v for k, v in matched.items() if k in left_cols and v in right_cols}


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

    # Cut at the first separator rather than deleting it. A glycopeptide table
    # often carries the glycan, the site or the accession in the same cell as the
    # backbone, and simply dropping everything that is not a letter welded those
    # onto the sequence: "EEQYNSTYR_HexNAc(4)Hex(5)" became "EEQYNSTYRHEXNACHEX",
    # which matches no peptide anywhere, so every site silently lost its
    # denominator. No amino acid sequence contains any of these characters, so
    # cutting here cannot shorten a plain sequence.
    text = re.split(r"[_/|+,;:\s]", text.strip(), maxsplit=1)[0]

    return re.sub(r"[^A-Za-z]", "", text).upper()


def sequence_variants(seq) -> list[tuple[str, str]]:
    """
    The forms one backbone can take in a peptide file, and what each would mean.

    The same peptide is not always written the same way in both files, and a
    single character of difference leaves a site with no denominator and nothing
    said about it. Three forms are worth trying, in descending order of how
    certain a match is:

    "sequence"   the sequence itself, which is what a total proteome run reports,
                 since nothing about the peptide changes when it carries no glycan.
    "marked"     the sequence with a site marker turned back into asparagine.
                 Some software writes the glycosylated residue as J rather than N.
    "deamidated" the sequence with a sequon asparagine written as aspartate. Where
                 glycans were released with PNGase F the formerly glycosylated
                 asparagine is deamidated, so the released peptide is one dalton
                 heavier and reads D where the glycopeptide read N. Only sequon
                 asparagines are swapped, one at a time, which keeps this from
                 pairing peptides that merely look alike.

    The caller records which form matched, because they do not mean the same
    thing: a deamidated match says the glycan was cut off, not that the site was
    unoccupied, and that changes how the ratio should be read.
    """
    base = strip_peptide_sequence(seq)
    if not base:
        return []

    out = [(base, "sequence")]

    unmarked = base.replace("J", "N")
    if unmarked != base:
        out.append((unmarked, "marked"))

    for candidate in (base, unmarked):
        for found in N_SEQUON.finditer(candidate):
            at = found.start()
            swapped = candidate[:at] + "D" + candidate[at + 1:]
            if swapped not in [s for s, _ in out]:
                out.append((swapped, "deamidated"))

    return out


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

        # Try the plain sequence first and the rewritten forms only after it, so a
        # site that has a straightforward match is never credited to a weaker rule.
        hit, how = None, "none"
        for candidate, rule in sequence_variants(sequence):
            if candidate in lookup.index:
                hit, how = lookup.loc[candidate], rule
                break

        if hit is not None:
            if isinstance(hit, pd.DataFrame):      # duplicate index, take the first
                hit = hit.iloc[0]
            for left_col, right_col in channel_pairs.items():
                record[left_col] = hit[right_col]
            record["denominator_match"] = how
            record["forms_summed"] = int(hit.get("forms_summed", 0) or 0)
            record["sequons_in_peptide"] = int(hit.get("sequons_in_peptide", 0) or 0)

        rows.append(record)

    return pd.DataFrame(rows).set_index("_glyco_index")


def _column_as_series(frame: pd.DataFrame, col, index=None) -> pd.Series:
    """
    One numeric column, as a Series on the given index, whatever the frame holds.

    Reading a column straight out has three ways of not giving back a Series, and
    each produced a different failure further on. A column the frame does not
    have at all comes back as None, and pd.to_numeric turns that into a bare nan
    whose "is not None" check then passes, so the next call fell over on a float
    rather than saying the column was missing. A heading the file repeats gives a
    frame instead of a column. And a Series rather than a frame gives the single
    value stored under that label.

    A column that is absent is a column of nothing, which is what the caller
    wants here: it leaves that channel empty instead of stopping the run.
    """
    if index is None:
        index = frame.index

    if isinstance(frame, pd.Series) or col not in getattr(frame, "columns", []):
        return pd.Series(np.nan, index=index, dtype="float64")

    values = frame.loc[:, col]
    if isinstance(values, pd.DataFrame):          # the heading occurs more than once
        values = values.iloc[:, 0]

    return pd.to_numeric(values, errors="coerce").reindex(index)


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
        num = _column_as_series(glyco, col)
        den = _column_as_series(denominator, col)

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