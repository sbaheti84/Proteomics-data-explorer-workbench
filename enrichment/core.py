"""
Pathway enrichment for differential results: the calculations (no Streamlit).

Two methods, both run with gseapy on gene-set files shipped in gene_sets/
(see build_gene_sets.py):

Over-representation analysis (ORA)
    Are proteins that changed significantly over-represented in a gene set?
    One-sided hypergeometric test, Benjamini-Hochberg adjusted. Conventions match
    clusterProfiler:
      - background (universe) = proteins tested in THIS dataset that have at
        least one annotation in the library. Never the whole genome: a
        proteomics or glycoproteomics experiment only sees part of the proteome,
        and a genome background makes "extracellular", "membrane", "lysosome"
        etc. look enriched in almost every dataset.
      - gene sets are restricted to the background before the size limits.
      - only gene sets with at least one significant protein are tested.

GSEA (pre-ranked)
    Uses every tested protein, ranked by sign(log2FC) x -log10(p-value).
    Fixed random seed, so results are reproducible.

Proteins, not rows
    Each protein is counted once. For glyco data, a protein with many glycoforms
    would otherwise get many chances to look significant. A protein counts as
    "up" if any of its rows is significantly up (likewise "down"); for GSEA its
    score comes from its most significant row.

Identifiers
    UniProt accessions are read from the protein column (isoform suffix removed;
    for protein groups the first accession with an annotation is used). Rows
    without an accession fall back to the gene symbol.
"""
from __future__ import annotations

import functools
import gzip
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

GSEA_SEED = 42
DEFAULT_MIN_SIZE = 5          # over-representation
DEFAULT_MAX_SIZE = 500
GSEA_MIN_SIZE = 15            # GSEA's own default; also keeps run time reasonable
GSEA_THREADS = max(1, min(8, os.cpu_count() or 1))  # results are identical for any thread count

PROTEIN_COLUMNS = [
    "Accession", "Master Protein Accessions", "Protein Accessions", "Protein ID", "Protein IDs",
    "Majority protein IDs", "Leading razor protein", "Proteins", "Protein", "ProteinID", "UniProt", "Uniprot",
]
GENE_COLUMNS = ["Genes", "Gene", "GeneID", "Gene Symbol", "GeneSymbol", "Gene Names", "Gene names", "Gene name"]
DESCRIPTION_COLUMNS = ["Description", "Protein Description", "Fasta headers", "Protein names"]

# UniProt accession format (https://www.uniprot.org/help/accession_numbers), optional isoform.
ACCESSION_RE = re.compile(
    r"(?<![A-Z0-9])([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-\d+)?(?![A-Z0-9])"
)
_SPLIT_GENES = re.compile(r"[;,|/\s]+")
_ID_IN_TERM = re.compile(r"\(([^()]+)\)\s*$")


# ----------------------------------------------------------------------------- gene-set files
def gene_set_dir() -> str:
    """gene_sets/ next to app.py (also inside the packaged app)."""
    base = getattr(sys, "_MEIPASS", None)
    if not base:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.environ.get("WORKBENCH_GENE_SETS", os.path.join(base, "gene_sets"))


def load_manifest() -> Optional[dict]:
    path = os.path.join(gene_set_dir(), "manifest.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def species_options(manifest: dict) -> List[str]:
    return list((manifest or {}).get("species", {}).keys())


def library_options(manifest: dict, species: str) -> Dict[str, str]:
    """{library id: title} for a species."""
    libs = (manifest or {}).get("species", {}).get(species, {}).get("libraries", {})
    return {k: v.get("title", k) for k, v in libs.items() if v.get("terms", 0) > 0}


@functools.lru_cache(maxsize=32)
def load_gene_sets(species: str, library: str) -> Dict[str, Tuple[str, ...]]:
    """{"Term name (ID)": (accession, ...)}"""
    manifest = load_manifest() or {}
    info = manifest.get("species", {}).get(species, {}).get("libraries", {}).get(library)
    if not info:
        raise KeyError(f"No gene-set library {library!r} for {species}.")
    path = os.path.join(gene_set_dir(), *info["file"].split("/"))
    sets = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                sets[parts[0]] = tuple(parts[2:])
    return sets


@dataclass(frozen=True)
class NameIndex:
    symbol_of: Dict[str, str]          # accession -> gene symbol
    acc_of_symbol: Dict[str, str]      # SYMBOL (upper case) -> accession, unambiguous only
    acc_of_alias: Dict[str, str]       # synonym / entry name (upper) -> accession, unambiguous only


@functools.lru_cache(maxsize=4)
def load_name_index(species: str) -> NameIndex:
    path = os.path.join(gene_set_dir(), species, "protein_names.tsv.gz")
    symbol_of, sym_hits, alias_hits = {}, {}, {}
    if os.path.exists(path):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            next(f, None)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                acc, sym = parts[0], parts[1]
                symbol_of[acc] = sym
                if sym:
                    sym_hits.setdefault(sym.upper(), set()).add(acc)
                for a in (parts[2].split("|") if len(parts) > 2 and parts[2] else []):
                    if a:
                        alias_hits.setdefault(a.upper(), set()).add(acc)
    one = lambda d: {k: next(iter(v)) for k, v in d.items() if len(v) == 1}
    return NameIndex(symbol_of, one(sym_hits), one(alias_hits))


# ----------------------------------------------------------------------------- columns and IDs
def first_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    return next((c for c in candidates if c in df.columns), None)


@dataclass
class Comparison:
    label: str
    lfc: Optional[str]
    p: Optional[str]
    q: Optional[str]


def find_comparisons(df: pd.DataFrame) -> List[Comparison]:
    """The main test, plus any per-pair columns 'log2FC (B vs A)' / 'q-value (B vs A)'."""
    out = []
    if "p-value" in df.columns or "q-value" in df.columns:
        lfc = "log2FC" if "log2FC" in df.columns else None
        label = "Main test" if lfc else "Main test (any group differs)"
        out.append(Comparison(label, lfc, "p-value" if "p-value" in df.columns else None,
                              "q-value" if "q-value" in df.columns else None))
    for c in df.columns:
        m = re.fullmatch(r"log2FC \((.+)\)", str(c))
        if m:
            name = m.group(1)
            q = f"q-value ({name})" if f"q-value ({name})" in df.columns else None
            p = f"p-value ({name})" if f"p-value ({name})" in df.columns else None
            if q or p:
                out.append(Comparison(name, c, p, q))
    return out


def detect_species(df: pd.DataFrame) -> Optional[str]:
    cols = [c for c in PROTEIN_COLUMNS + DESCRIPTION_COLUMNS + ["Entry Name"] if c in df.columns]
    if not cols:
        return None
    text = " ".join(df[cols].astype(str).head(500).fillna("").values.ravel().tolist()).upper()
    human = len(re.findall(r"_HUMAN\b|\bOX=9606\b|HOMO SAPIENS", text))
    mouse = len(re.findall(r"_MOUSE\b|\bOX=10090\b|MUS MUSCULUS", text))
    if human == mouse == 0:
        return None
    return "human" if human >= mouse else "mouse"


def accessions_in(value) -> List[str]:
    """All UniProt accessions in a cell, isoform suffix removed, in order."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return list(dict.fromkeys(ACCESSION_RE.findall(str(value).upper())))


def map_to_accessions(df: pd.DataFrame, names: NameIndex, known: Optional[set] = None) -> Tuple[pd.Series, dict]:
    """One accession per row (None if it cannot be mapped) and a short mapping report."""
    known = known if known is not None else set(names.symbol_of)
    prot_col = first_column(df, PROTEIN_COLUMNS)
    gene_col = first_column(df, GENE_COLUMNS)
    entry_col = "Entry Name" if "Entry Name" in df.columns else None
    out, how = [], {"accession": 0, "gene symbol": 0, "not mapped": 0}
    for i in range(len(df)):
        acc = None
        if prot_col is not None:
            accs = accessions_in(df[prot_col].iat[i])
            if accs:
                acc = next((a for a in accs if a in known), accs[0])
                how["accession"] += 1
        if acc is None:
            for col in (gene_col, entry_col):
                if col is None:
                    continue
                for token in _SPLIT_GENES.split(str(df[col].iat[i]).strip()):
                    t = token.upper()
                    if not t or t in {"NAN", "NONE"}:
                        continue
                    acc = names.acc_of_symbol.get(t) or names.acc_of_alias.get(t)
                    if acc:
                        break
                if acc:
                    how["gene symbol"] += 1
                    break
        if acc is None:
            how["not mapped"] += 1
        out.append(acc)
    report = {"rows": len(df), "protein column": prot_col, "gene column": gene_col, **how}
    return pd.Series(out, index=df.index, dtype=object), report


# ----------------------------------------------------------------------------- protein-level table
@dataclass
class Selection:
    sig_col: str          # column used for the cutoff (q-value or p-value)
    alpha: float
    lfc_threshold: float  # |log2FC| must be > this (ignored without a log2FC column)


def protein_table(df: pd.DataFrame, acc: pd.Series, comp: Comparison, sel: Selection,
                  group: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    One row per protein (per glycan group, if `group` is given) with:
    tested, up, down, changed, score (for GSEA), n_rows.
    Rows without an accession or without a test result are left out.
    """
    sig = pd.to_numeric(df[sel.sig_col], errors="coerce")
    lfc = pd.to_numeric(df[comp.lfc], errors="coerce") if comp.lfc else pd.Series(np.nan, index=df.index)
    pcol = comp.p or comp.q
    p = pd.to_numeric(df[pcol], errors="coerce")
    t = pd.DataFrame({"acc": acc, "sig": sig, "lfc": lfc, "p": p})
    if group is not None:
        t["group"] = group.astype(str).values
    t = t[t["acc"].notna() & t["sig"].notna()]
    hit = t["sig"] < sel.alpha
    if comp.lfc:
        t["up"] = hit & (t["lfc"] > sel.lfc_threshold)
        t["down"] = hit & (t["lfc"] < -sel.lfc_threshold)
        t["changed"] = t["up"] | t["down"]
        safe_p = t["p"].clip(lower=1e-300)
        t["score"] = np.sign(t["lfc"]) * -np.log10(safe_p)
    else:
        t["up"] = False
        t["down"] = False
        t["changed"] = hit
        t["score"] = np.nan
    keys = ["acc", "group"] if group is not None else ["acc"]
    t = t.sort_values("p", na_position="last")
    agg = t.groupby(keys, sort=False).agg(
        up=("up", "any"), down=("down", "any"), changed=("changed", "any"),
        score=("score", "first"), lfc=("lfc", "first"), p=("p", "first"), n_rows=("sig", "size"),
    ).reset_index()
    return agg


# ----------------------------------------------------------------------------- statistics
def _term_id(term: str) -> str:
    m = _ID_IN_TERM.search(term)
    return m.group(1) if m else ""


def _term_name(term: str) -> str:
    return _ID_IN_TERM.sub("", term).strip()


def _symbols(accs: Iterable[str], names: NameIndex) -> str:
    return ", ".join(sorted({names.symbol_of.get(a) or a for a in accs if a}))


def run_ora(query: Iterable[str], universe: Iterable[str], gene_sets: Dict[str, Sequence[str]],
            names: NameIndex, min_size: int = DEFAULT_MIN_SIZE, max_size: int = DEFAULT_MAX_SIZE) -> Tuple[pd.DataFrame, dict]:
    """Over-representation analysis with gseapy. Returns (table, counts)."""
    import gseapy

    annotated_all = set().union(*gene_sets.values()) if gene_sets else set()
    background = set(universe) & annotated_all
    q = set(query) & background
    sets = {}
    for term, members in gene_sets.items():
        m = background.intersection(members)
        if min_size <= len(m) <= max_size:
            sets[term] = sorted(m)
    counts = {"universe": len(set(universe)), "background": len(background), "query": len(set(query)),
              "query annotated": len(q), "sets tested": 0}
    empty = pd.DataFrame(columns=["Term", "ID", "Set size", "Hits", "Expected", "Fold enrichment",
                                  "P-value", "Adjusted P-value", "Proteins"])
    if not q or not sets:
        return empty, counts
    enr = gseapy.enrich(gene_list=sorted(q), gene_sets=sets, background=background,
                        outdir=None, no_plot=True, verbose=False)
    res = getattr(enr, "results", None)
    if res is None or len(res) == 0:
        return empty, counts
    res = res.copy()
    counts["sets tested"] = len(res)
    hits = res["Overlap"].str.split("/", expand=True).astype(int)
    n_bg, n_q = len(background), len(q)
    out = pd.DataFrame({
        "Term": res["Term"].map(_term_name),
        "ID": res["Term"].map(_term_id),
        "Set size": hits[1],
        "Hits": hits[0],
    })
    out["Expected"] = out["Set size"] * n_q / n_bg
    out["Fold enrichment"] = out["Hits"] / out["Expected"]
    out["P-value"] = res["P-value"].astype(float)
    out["Adjusted P-value"] = res["Adjusted P-value"].astype(float)
    out["Proteins"] = res["Genes"].map(lambda g: _symbols(str(g).split(";"), names))
    out = out.sort_values(["Adjusted P-value", "P-value"]).reset_index(drop=True)
    return out, counts


def run_gsea(scores: pd.Series, gene_sets: Dict[str, Sequence[str]], names: NameIndex,
             min_size: int = GSEA_MIN_SIZE, max_size: int = DEFAULT_MAX_SIZE,
             permutations: int = 1000, seed: int = GSEA_SEED) -> Tuple[pd.DataFrame, dict]:
    """Pre-ranked GSEA with gseapy. `scores` is indexed by accession."""
    import gseapy

    rnk = scores.dropna()
    rnk = rnk[~rnk.index.duplicated()].sort_values(ascending=False)
    ranked = set(rnk.index)
    sets = {}
    for term, members in gene_sets.items():
        m = ranked.intersection(members)
        if min_size <= len(m) <= max_size:
            sets[term] = sorted(m)
    counts = {"ranked proteins": len(rnk), "ranked annotated": len(ranked & set().union(*gene_sets.values())) if gene_sets else 0,
              "sets tested": len(sets)}
    empty = pd.DataFrame(columns=["Term", "ID", "Set size", "NES", "ES", "P-value", "FDR q-value", "Leading-edge proteins"])
    if len(rnk) < 10 or not sets:
        return empty, counts
    pre = gseapy.prerank(rnk=rnk, gene_sets=sets, min_size=min_size, max_size=max_size,
                         permutation_num=int(permutations), seed=int(seed), threads=GSEA_THREADS,
                         outdir=None, no_plot=True, verbose=False)
    res = pre.res2d.copy()
    if res.empty:
        return empty, counts
    out = pd.DataFrame({
        "Term": res["Term"].map(_term_name),
        "ID": res["Term"].map(_term_id),
        "Set size": res["Term"].map(lambda t: len(sets.get(t, ()))),
        "NES": pd.to_numeric(res["NES"], errors="coerce"),
        "ES": pd.to_numeric(res["ES"], errors="coerce"),
        "P-value": pd.to_numeric(res["NOM p-val"], errors="coerce"),
        "FDR q-value": pd.to_numeric(res["FDR q-val"], errors="coerce"),
        "Leading-edge proteins": res["Lead_genes"].map(lambda g: _symbols(str(g).split(";"), names)),
    })
    out = out.sort_values(["FDR q-value", "P-value"]).reset_index(drop=True)
    return out, counts


# ----------------------------------------------------------------------------- glycan groups
def glycan_groups(df: pd.DataFrame, how: str) -> Optional[pd.Series]:
    """'Glycan class' uses the GlycanClass column; 'Sialylation / fucosylation' uses the composition."""
    if how == "Glycan class":
        return df["GlycanClass"].fillna("Unknown").astype(str) if "GlycanClass" in df.columns else None
    col = first_column(df, ["Glycan(H,N,A,F)", "Glycan(H,N,A,G,F)", "Glycan", "GlycanComposition"])
    if col is None:
        return None
    from ge.qc_utils import _glycan_trait  # same rule as the glycan class breakdown plot
    return df[col].map(_glycan_trait)


# ----------------------------------------------------------------------------- methods text
def methods_paragraph(settings: dict, manifest: dict, counts: dict) -> str:
    sp = settings["species"]
    libs = manifest["species"][sp]["libraries"]
    lib_txt = "; ".join(f"{libs[l]['title']} ({libs[l]['source']})" for l in settings["libraries"])
    import gseapy
    common = (
        f"Pathway enrichment used gseapy {gseapy.__version__} (Fang et al., Bioinformatics 2023) with gene sets "
        f"keyed by UniProt accession for {sp}: {lib_txt}. Gene sets were built on {manifest.get('built', 'an unknown date')} "
        "(GO annotations propagated to parent terms over is_a and part_of; NOT annotations excluded). "
        "Proteins were mapped by UniProt accession (isoform suffix removed; first annotated accession of a protein group), "
        "or by gene symbol when no accession was given, and each protein was counted once"
    )
    if settings.get("glyco"):
        common += " regardless of how many glycoforms it had"
    common += ". "
    size = f"Gene sets with {settings['min_size']}-{settings['max_size']} proteins in the background were tested. "
    if settings["method"] == "ORA":
        thr = f"{settings['sig_col']} < {settings['alpha']}"
        if settings.get("lfc_col"):
            thr += f" and |log2FC| > {settings['lfc_threshold']}"
        txt = (
            common
            + f"Over-representation of significantly changed proteins ({thr}; comparison: {settings['comparison']}; "
            f"direction: {settings['direction'].lower()}) was tested with a one-sided hypergeometric test "
            "against a background of all proteins tested in this dataset that had at least one annotation in the "
            "library. " + size + "P-values were adjusted with the Benjamini-Hochberg method"
        )
        if settings.get("group_by"):
            txt += (
                f". The analysis was run separately for each glycan group ({settings['group_by'].lower()}): "
                "the query was proteins with at least one significant glycoform in that group, and the background "
                "was proteins with at least one tested glycoform in that group"
            )
        return txt + "."
    return (
        common
        + "Pre-ranked gene set enrichment analysis (GSEA; Subramanian et al., PNAS 2005) ranked all tested proteins by "
        f"sign(log2FC) x -log10({settings.get('rank_col', 'p-value')}), using each protein's most significant row "
        f"(comparison: {settings['comparison']}). " + size
        + f"Significance was estimated with {settings['permutations']} gene-set permutations (random seed {settings['seed']}); "
        "FDR q-values are reported."
    )
