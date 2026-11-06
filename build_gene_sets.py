"""
Build the gene-set files used by the Pathway Enrichment tab.

    python build_gene_sets.py                       # download, build gene_sets/, delete downloads
    python build_gene_sets.py --keep-downloads DIR  # keep the downloads in DIR for a re-run
    python build_gene_sets.py --source-dir DIR      # use files you already downloaded

Run it when you want newer annotations, then commit ONLY the gene_sets/ folder
(a few MB). The source files are several hundred MB: by default they are
downloaded to a temporary folder outside the repository and deleted at the end,
so they can never end up in git. The GitHub workflow "Update gene sets" runs
this for you, so nothing is downloaded to your computer at all.

Sources (official, free to redistribute with attribution):
  GO ontology      https://purl.obolibrary.org/obo/go/go-basic.obo
  GO annotations   EBI GOA, UniProt-keyed: goa_human.gaf.gz, goa_mouse.gaf.gz
  Reactome         UniProt2Reactome_All_Levels.txt (all pathway levels)

What it does:
  - GO: an annotation to a term also counts for all of that term's parent terms
    (is_a and part_of), the standard "true path" rule. "NOT" annotations are
    skipped. Evidence codes are not filtered.
  - Reactome: proteins are already mapped to every pathway level.
  - Proteins are UniProt accessions (isoform suffix removed), which match
    proteomics output better than gene symbols.
  - Terms with fewer than MIN_TERM_SIZE or more than MAX_TERM_SIZE proteins are
    dropped (the app applies its own, narrower size limits at run time).

Output (gene_sets/):
  manifest.json                 versions, sources and counts (shown in the app and
                                written into the methods paragraph)
  <species>/<LIB>.gmt.gz        one gene set per line: name, ID, accessions...
  <species>/protein_names.tsv.gz  accession, gene symbol, synonyms
Files are gzipped without timestamps, so rebuilding unchanged data gives
identical files (no git noise).
"""
import argparse
import collections
import datetime as _dt
import gzip
import io
import json
import os
import re
import sys
import tempfile
import urllib.request

BUILDER_VERSION = 1
MIN_TERM_SIZE = 3
MAX_TERM_SIZE = 3000

SPECIES = {
    "human": {"taxon": "9606", "reactome_name": "Homo sapiens", "goa": "HUMAN/goa_human.gaf.gz"},
    "mouse": {"taxon": "10090", "reactome_name": "Mus musculus", "goa": "MOUSE/goa_mouse.gaf.gz"},
}
URLS = {
    "go_obo": "https://purl.obolibrary.org/obo/go/go-basic.obo",
    "goa_base": "https://ftp.ebi.ac.uk/pub/databases/GO/goa/",
    "reactome": "https://reactome.org/download/current/UniProt2Reactome_All_Levels.txt",
    "reactome_version": "https://reactome.org/ContentService/data/database/version",
}
# The three root terms contain every annotated protein, so they say nothing.
GO_ROOTS = {"GO:0008150", "GO:0003674", "GO:0005575"}
GO_LIBS = {  # namespace -> (library id, title)
    "biological_process": ("GO_BP", "GO Biological Process"),
    "molecular_function": ("GO_MF", "GO Molecular Function"),
    "cellular_component": ("GO_CC", "GO Cellular Component"),
}
ISOFORM = re.compile(r"-\d+$")
GAF_DATE = re.compile(r"^!\s*(?:date-generated|generated)\s*:\s*(.+)$", re.IGNORECASE)


# ----------------------------------------------------------------------------- download
def _fetch(url: str, dest: str) -> str:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  using cached {os.path.basename(dest)}")
        return dest
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "ProteomicsWorkbench-gene-set-builder"})
    tmp = dest + ".part"
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)
    return dest


def get_sources(source_dir: str, download: bool) -> dict:
    os.makedirs(source_dir, exist_ok=True)
    files = {
        "go_obo": os.path.join(source_dir, "go-basic.obo"),
        "reactome": os.path.join(source_dir, "UniProt2Reactome_All_Levels.txt"),
        "reactome_version": os.path.join(source_dir, "reactome_version.txt"),
    }
    for sp, info in SPECIES.items():
        files[f"goa_{sp}"] = os.path.join(source_dir, os.path.basename(info["goa"]))
    if download:
        _fetch(URLS["go_obo"], files["go_obo"])
        _fetch(URLS["reactome"], files["reactome"])
        try:
            _fetch(URLS["reactome_version"], files["reactome_version"])
        except Exception as e:  # the version number is nice to have, not essential
            print(f"  could not get the Reactome version ({e})")
        for sp, info in SPECIES.items():
            _fetch(URLS["goa_base"] + info["goa"], files[f"goa_{sp}"])
    missing = [p for k, p in files.items() if k != "reactome_version" and not os.path.exists(p)]
    if missing:
        sys.exit("Missing source files:\n  " + "\n  ".join(missing))
    return files


# ----------------------------------------------------------------------------- GO
def parse_obo(path: str):
    """Return (terms, alt_ids, data_version). terms[id] = {name, namespace, parents}."""
    terms, alt, version = {}, {}, ""
    cur = None
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("data-version:") and not terms and cur is None:
                version = line.split(":", 1)[1].strip()
            if line.startswith("["):
                cur = {"parents": set(), "obsolete": False} if line == "[Term]" else None
                continue
            if cur is None or not line or line.startswith("!"):
                continue
            key, _, val = line.partition(": ")
            val = val.split(" ! ")[0].strip()
            if key == "id":
                cur["id"] = val
                terms[val] = cur
            elif key == "name":
                cur["name"] = val
            elif key == "namespace":
                cur["namespace"] = val
            elif key == "alt_id":
                alt[val] = cur.get("id")
            elif key == "is_a":
                cur["parents"].add(val.split()[0])
            elif key == "relationship":
                rel, _, target = val.partition(" ")
                if rel == "part_of":
                    cur["parents"].add(target.split()[0])
            elif key == "is_obsolete" and val == "true":
                cur["obsolete"] = True
    terms = {k: v for k, v in terms.items() if not v["obsolete"] and "namespace" in v}
    alt = {a: p for a, p in alt.items() if p in terms}
    return terms, alt, version


def ancestors_fn(terms: dict):
    cache = {}

    def anc(term_id: str) -> frozenset:
        if term_id in cache:
            return cache[term_id]
        stack, seen = [term_id], set()
        while stack:
            t = stack.pop()
            if t in seen or t not in terms:
                continue
            seen.add(t)
            stack.extend(terms[t]["parents"])
        # stay inside the term's own ontology (BP, MF or CC)
        ns = terms[term_id]["namespace"]
        cache[term_id] = frozenset(x for x in seen if terms[x]["namespace"] == ns)
        return cache[term_id]

    return anc


def parse_gaf(path: str, taxon: str, terms: dict, alt: dict):
    """Return (go_sets {namespace: {term: set(acc)}}, names {acc: (symbol, synonyms)}, generated)."""
    anc = ancestors_fn(terms)
    sets = {ns: collections.defaultdict(set) for ns in GO_LIBS}
    names, generated = {}, ""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!"):
                m = GAF_DATE.match(line)
                if m and not generated:
                    generated = m.group(1).strip()[:10]
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 13 or c[0] != "UniProtKB":
                continue
            if c[12].split("|")[0].strip() != f"taxon:{taxon}":
                continue
            acc = ISOFORM.sub("", c[1].strip())
            symbol = c[2].strip()
            syn = c[10].strip() if len(c) > 10 else ""
            if acc not in names:
                names[acc] = (symbol, syn)
            if "NOT" in c[3].split("|"):
                continue
            go = alt.get(c[4], c[4])
            if go not in terms:
                continue
            ns = terms[go]["namespace"]
            if ns not in sets:
                continue
            for t in anc(go):
                if t not in GO_ROOTS:
                    sets[ns][t].add(acc)
    return sets, names, generated


# ----------------------------------------------------------------------------- Reactome
def parse_reactome(path: str, species_name: str):
    sets, titles = collections.defaultdict(set), {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) < 6 or c[5].strip() != species_name:
                continue
            acc = ISOFORM.sub("", c[0].strip())
            pid = c[1].strip()
            sets[pid].add(acc)
            titles[pid] = c[3].strip()
    return sets, titles


# ----------------------------------------------------------------------------- write
def _gzip_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # mtime=0 and no file name: identical input gives an identical file.
    with open(path, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
            gz.write(text.encode("utf-8"))


def write_gmt(path: str, sets: dict, titles: dict) -> tuple:
    lines, prots = [], set()
    for tid in sorted(sets):
        accs = sets[tid]
        if not (MIN_TERM_SIZE <= len(accs) <= MAX_TERM_SIZE):
            continue
        name = titles[tid].replace("\t", " ")
        lines.append("\t".join([f"{name} ({tid})", tid] + sorted(accs)))
        prots |= accs
    _gzip_text(path, "\n".join(lines) + "\n")
    return len(lines), len(prots)


def build(out_dir: str, files: dict) -> dict:
    print("Reading GO ontology...")
    terms, alt, go_version = parse_obo(files["go_obo"])
    print(f"  {len(terms):,} current GO terms ({go_version or 'version not stated'})")
    reactome_version = ""
    if os.path.exists(files["reactome_version"]):
        reactome_version = open(files["reactome_version"], encoding="utf-8").read().strip()

    manifest = {
        "builder_version": BUILDER_VERSION,
        "built": _dt.date.today().isoformat(),
        "term_size_limits": [MIN_TERM_SIZE, MAX_TERM_SIZE],
        "id_type": "UniProt accession",
        "species": {},
    }
    for sp, info in SPECIES.items():
        print(f"{sp}: reading GO annotations...")
        go_sets, names, generated = parse_gaf(files[f"goa_{sp}"], info["taxon"], terms, alt)
        libs = {}
        for ns, (lib, title) in GO_LIBS.items():
            n_terms, n_prot = write_gmt(os.path.join(out_dir, sp, f"{lib}.gmt.gz"), go_sets[ns],
                                        {t: terms[t]["name"] for t in go_sets[ns]})
            libs[lib] = {
                "title": title, "file": f"{sp}/{lib}.gmt.gz", "terms": n_terms, "proteins": n_prot,
                "source": f"Gene Ontology ({go_version or 'go-basic.obo'}); annotations: EBI GOA "
                          f"{os.path.basename(info['goa'])}" + (f" ({generated})" if generated else ""),
                "citation": "Ashburner M, et al. Nat Genet. 2000;25(1):25-29; Gene Ontology Consortium. "
                            "The Gene Ontology knowledgebase in 2023. Genetics. 2023;224(1):iyad031.",
            }
            print(f"  {lib}: {n_terms:,} terms, {n_prot:,} proteins")
        print(f"{sp}: reading Reactome...")
        r_sets, r_titles = parse_reactome(files["reactome"], info["reactome_name"])
        n_terms, n_prot = write_gmt(os.path.join(out_dir, sp, "Reactome.gmt.gz"), r_sets, r_titles)
        libs["Reactome"] = {
            "title": "Reactome pathways", "file": f"{sp}/Reactome.gmt.gz", "terms": n_terms, "proteins": n_prot,
            "source": "Reactome" + (f" release {reactome_version}" if reactome_version else "")
                      + " (UniProt2Reactome_All_Levels)"
                      + (" - mouse pathways are inferred from human by orthology" if sp == "mouse" else ""),
            "citation": "Milacic M, et al. The Reactome Pathway Knowledgebase 2024. Nucleic Acids Res. 2024;52(D1):D672-D678.",
        }
        print(f"  Reactome: {n_terms:,} pathways, {n_prot:,} proteins")
        rows = ["accession\tsymbol\tsynonyms"] + [
            f"{a}\t{s}\t{syn}" for a, (s, syn) in sorted(names.items())
        ]
        _gzip_text(os.path.join(out_dir, sp, "protein_names.tsv.gz"), "\n".join(rows) + "\n")
        manifest["species"][sp] = {"taxon": info["taxon"], "libraries": libs}

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"Wrote {out_dir}/manifest.json")
    return manifest


def _inside_git_repo(path: str) -> bool:
    p = os.path.abspath(path)
    while True:
        if os.path.isdir(os.path.join(p, ".git")):
            return True
        parent = os.path.dirname(p)
        if parent == p:
            return False
        p = parent


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="gene_sets", help="output folder (default: gene_sets)")
    ap.add_argument("--source-dir", default=None,
                    help="folder with already-downloaded source files; nothing is downloaded")
    ap.add_argument("--keep-downloads", "--cache", dest="keep", default=None, metavar="DIR",
                    help="keep the downloaded source files in DIR so a re-run does not download again. "
                         "Use a folder OUTSIDE the repository (the files are several hundred MB).")
    a = ap.parse_args(argv)

    if a.source_dir:
        build(a.out, get_sources(a.source_dir, download=False))
    elif a.keep:
        if _inside_git_repo(a.keep):
            print(f"WARNING: {a.keep} is inside a git repository. Do not commit it: the source files are "
                  "several hundred MB and GitHub rejects files over 100 MB.")
        build(a.out, get_sources(a.keep, download=True))
    else:
        with tempfile.TemporaryDirectory(prefix="gene_set_sources_") as tmp:
            print(f"Downloading source files to a temporary folder: {tmp}")
            build(a.out, get_sources(tmp, download=True))
        print("Deleted the downloaded source files (only gene_sets/ is kept).")


if __name__ == "__main__":
    main()