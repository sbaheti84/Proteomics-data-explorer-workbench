"""
"Pathway Enrichment" tab, shared by every app.

    from enrichment.ui import render_pathway_enrichment
    render_pathway_enrichment(results_df, app_key="proteome-tmt")
    render_pathway_enrichment(results_df, app_key="glyco-tmt", glyco=True)

The calculations are in enrichment/core.py.
"""
from __future__ import annotations

import io
import logging
import traceback

import numpy as np
import pandas as pd
import streamlit as st

from enrichment import core

LOG = logging.getLogger("workbench.enrichment")

ORA = "Over-representation (significant proteins)"
GSEA = "GSEA (all proteins, ranked)"
DIRECTIONS = ["Up and down separately", "Up only", "Down only", "Up and down together"]
GROUP_LEVELS = ["Proteins (any significant glycoform)", "Separately for each glycan group"]


def _k(app_key: str, name: str) -> str:
    return f"pe_{app_key}_{name}"


def _explainer(glyco: bool) -> None:
    parts = [
        "**Over-representation** asks: among the proteins that changed significantly, are some pathways or "
        "GO terms more common than expected? It compares them with a **background of all proteins tested in "
        "this dataset** (that have at least one annotation), not the whole genome. A whole-genome background "
        "makes membrane, extracellular and lysosomal terms look enriched in almost any proteomics dataset.",
        "**GSEA** uses every tested protein, ranked from most up to most down "
        "(sign of log2FC × −log10 p-value). It finds pathways whose proteins sit near the top or bottom of the "
        "list, without a cutoff. Positive NES = shifted up, negative NES = shifted down.",
        "**Each protein counts once.**",
    ]
    if glyco:
        parts[-1] += (" A protein with many glycoforms counts as *up* if any glycoform is significantly up "
                      "(likewise *down*). Without this, heavily glycosylated proteins would dominate.")
        parts.append(
            "**By glycan group**: the test is run separately for, e.g., sialylated glycoforms, against the "
            "proteins that carry that kind of glycan in this dataset. The idea comes from the glycoverse "
            "*glyfun* package (glycan-centric enrichment)."
        )
    parts.append("Gene sets: GO (Biological Process, Molecular Function, Cellular Component) and Reactome, "
                 "matched by UniProt accession. Versions are listed under *Methods text* below the results.")
    with st.expander("What this does and how to read it", icon=":material/help:"):
        st.markdown("\n\n".join(parts))


def _mapping_note(report: dict, n_background: int, library_titles: str) -> None:
    rows = report["rows"]
    mapped = report["accession"] + report["gene symbol"]
    frac = mapped / rows if rows else 0
    st.caption(
        f"Identifiers: {mapped:,} of {rows:,} result rows matched to a UniProt protein "
        f"({report['accession']:,} by accession, {report['gene symbol']:,} by gene symbol). "
        f"Background: {n_background:,} tested proteins with an annotation in {library_titles}."
    )
    if rows and frac < 0.5:
        st.warning(
            "Fewer than half of the rows could be matched to a protein. Check that the species is right and that "
            f"the results have a protein accession column (looked for: {', '.join(core.PROTEIN_COLUMNS[:6])}, ...) "
            "or a gene symbol column."
        )
    if 0 < n_background < 100:
        st.warning(f"Only {n_background} background proteins: enrichment results will be weak and unstable.")


def _bar_chart(table: pd.DataFrame, method: str, top_n: int, color_col: str):
    import plotly.express as px

    if table.empty:
        return None
    if method == ORA:
        d = table.nsmallest(top_n, ["Adjusted P-value", "P-value"]).copy()
        d["−log10 adjusted p"] = -np.log10(d["Adjusted P-value"].clip(lower=1e-300))
        x, hover = "−log10 adjusted p", ["Hits", "Set size", "Fold enrichment", "Adjusted P-value"]
    else:
        d = table.nsmallest(top_n, ["FDR q-value", "P-value"]).copy()
        x, hover = "NES", ["Set size", "FDR q-value"]
    d["Label"] = d["Term"].str.slice(0, 70) + " (" + d["ID"] + ")"
    d = d.iloc[::-1]
    fig = px.bar(d, x=x, y="Label", orientation="h", color=color_col if color_col in d.columns else None,
                 hover_data=[c for c in hover if c in d.columns])
    fig.update_layout(height=max(320, 26 * len(d) + 120), yaxis_title=None, legend_title=None,
                      margin=dict(l=10, r=10, t=30, b=10))
    if method == ORA:
        fig.add_vline(x=-np.log10(0.05), line_dash="dot", line_color="grey")
    return fig


def _compute(results: pd.DataFrame, s: dict) -> dict:
    """Run the analysis described by settings `s`. Returns tables + counts."""
    species = s["species"]
    names = core.load_name_index(species)
    lib_sets = {lib: core.load_gene_sets(species, lib) for lib in s["libraries"]}
    known = set(names.symbol_of)
    for gs in lib_sets.values():
        for members in gs.values():
            known.update(members)
    acc, report = core.map_to_accessions(results, names, known)
    comp = core.Comparison(s["comparison"], s.get("lfc_col"), s.get("p_col"), s.get("q_col"))
    sel = core.Selection(s.get("sig_col") or comp.q or comp.p, float(s.get("alpha", 0.05)),
                         float(s.get("lfc_threshold", 0.0)))
    groups = core.glycan_groups(results, s["group_by"]) if s.get("group_by") else None
    prot = core.protein_table(results, acc, comp, sel, groups)

    tables, counts = [], []
    if s["method"] == "ORA":
        if s.get("lfc_col"):
            dirs = {"Up and down separately": ["Up", "Down"], "Up only": ["Up"], "Down only": ["Down"],
                    "Up and down together": ["Up or down"]}[s["direction"]]
        else:
            dirs = ["Changed"]
        flag = {"Up": "up", "Down": "down", "Up or down": "changed", "Changed": "changed"}
        group_values = sorted(prot["group"].unique()) if groups is not None else [None]
        for g in group_values:
            sub = prot[prot["group"] == g] if g is not None else prot
            universe = set(sub["acc"])
            for d in dirs:
                query = set(sub.loc[sub[flag[d]], "acc"])
                for lib, gs in lib_sets.items():
                    tab, cnt = core.run_ora(query, universe, gs, names, s["min_size"], s["max_size"])
                    tab.insert(0, "Library", s["library_titles"][lib])
                    tab.insert(0, "Direction", d)
                    if g is not None:
                        tab.insert(0, "Glycan group", g)
                    tables.append(tab)
                    counts.append({"Glycan group": g, "Direction": d, "Library": s["library_titles"][lib], **cnt})
    else:
        scores = prot.set_index("acc")["score"]
        for lib, gs in lib_sets.items():
            tab, cnt = core.run_gsea(scores, gs, names, s["min_size"], s["max_size"],
                                     s["permutations"], s["seed"])
            tab.insert(0, "Library", s["library_titles"][lib])
            tables.append(tab)
            counts.append({"Library": s["library_titles"][lib], **cnt})
    table = pd.concat([t for t in tables if not t.empty], ignore_index=True) if any(not t.empty for t in tables) \
        else (tables[0] if tables else pd.DataFrame())
    n_bg = max((c.get("background", c.get("ranked annotated", 0)) for c in counts), default=0)
    return {"table": table, "counts": pd.DataFrame(counts), "report": report, "n_background": n_bg,
            "n_proteins": int(prot["acc"].nunique())}


def render_pathway_enrichment(results, app_key: str, glyco: bool = False) -> None:
    ss = st.session_state
    k = lambda n: _k(app_key, n)

    st.markdown("#### Pathway Enrichment")
    _explainer(glyco)

    if results is None or not isinstance(results, pd.DataFrame) or results.empty:
        st.info("Run Differential Analysis first.")
        return
    manifest = core.load_manifest()
    if not manifest or not core.species_options(manifest):
        st.warning(
            "The gene-set files are missing from this installation (gene_sets/manifest.json). "
            "They are created by running build_gene_sets.py, or the *Update gene sets* workflow on GitHub."
        )
        return
    comps = core.find_comparisons(results)
    if not comps:
        st.info("These results have no p-value or q-value column, so enrichment cannot run.")
        return

    with st.container(border=True):
        st.markdown("##### Settings")
        species_opts = core.species_options(manifest)
        detected = core.detect_species(results)
        c1, c2 = st.columns(2)
        species = c1.selectbox(
            "Species", species_opts,
            index=species_opts.index(detected) if detected in species_opts else 0,
            format_func=str.capitalize, key=k("species"),
            help="Detected from the protein IDs or descriptions when possible "
                 f"({'detected: ' + detected if detected else 'not detected'}).",
        )
        lib_opts = core.library_options(manifest, species)
        libraries = c2.multiselect(
            "Gene-set libraries", list(lib_opts), default=[x for x in ("GO_BP", "Reactome") if x in lib_opts],
            format_func=lambda x: lib_opts.get(x, x), key=k("libs"),
        )

        c3, c4 = st.columns(2)
        labels = [c.label for c in comps]
        comp_label = c3.selectbox("Comparison", labels, key=k("comp"),
                                  help="Main test, or one pair of groups when the results include per-pair columns.")
        comp = comps[labels.index(comp_label)]
        method_opts = [ORA] + ([GSEA] if comp.lfc else [])
        method = c4.radio("Method", method_opts, key=k("method"),
                          help="GSEA needs a log2FC, so it is not offered for the 'any group differs' test.")

        settings = {"species": species, "libraries": list(libraries), "comparison": comp.label,
                    "lfc_col": comp.lfc, "p_col": comp.p, "q_col": comp.q, "glyco": glyco,
                    "library_titles": {l: lib_opts.get(l, l) for l in libraries}}
        if method == ORA:
            c5, c6, c7, c8 = st.columns(4)
            sig_choices = [x for x in (comp.q, comp.p) if x]
            settings["method"] = "ORA"
            settings["sig_col"] = c5.selectbox("Significance column", sig_choices, key=k("sig"))
            settings["alpha"] = float(c6.number_input("Cutoff (<)", min_value=0.0001, max_value=0.5, value=0.05,
                                                      step=0.01, format="%.4f", key=k("alpha")))
            if comp.lfc:
                settings["lfc_threshold"] = float(c7.number_input("|log2FC| >", min_value=0.0, max_value=10.0,
                                                                  value=1.0, step=0.1, key=k("lfc")))
                settings["direction"] = c8.selectbox("Direction", DIRECTIONS, key=k("dir"))
            else:
                settings["direction"] = "Changed (any direction)"
            if glyco:
                g1, g2 = st.columns(2)
                level = g1.radio("Count", GROUP_LEVELS, key=k("level"),
                                 help="By glycan group runs the test separately for each kind of glycan.")
                if level == GROUP_LEVELS[1]:
                    group_opts = (["Glycan class"] if "GlycanClass" in results.columns else []) + \
                                 ["Sialylation / fucosylation"]
                    settings["group_by"] = g2.selectbox("Glycan groups", group_opts, key=k("groupby"))
        else:
            settings["method"] = "GSEA"
            settings["rank_col"] = comp.p or comp.q
            settings["permutations"] = int(st.number_input("Permutations", min_value=100, max_value=10000,
                                                           value=1000, step=100, key=k("perm")))
            settings["seed"] = core.GSEA_SEED
            st.caption("GSEA is slower than over-representation: GO Biological Process can take a few minutes. "
                       "Reactome is quicker.")
        c9, c10 = st.columns(2)
        is_gsea = settings["method"] == "GSEA"
        settings["min_size"] = int(c9.number_input(
            "Smallest gene set (proteins)", 2, 1000, core.GSEA_MIN_SIZE if is_gsea else core.DEFAULT_MIN_SIZE,
            key=k("min_gsea" if is_gsea else "min"),
            help="Counted among proteins in this dataset. Defaults: 5 for over-representation, 15 for GSEA."))
        settings["max_size"] = int(c10.number_input("Largest gene set (proteins)", 10, 5000, core.DEFAULT_MAX_SIZE,
                                                    key=k("max")))
        run = st.button("Run enrichment", type="primary", key=k("run"), disabled=not libraries)

    if run:
        try:
            with st.spinner("Running enrichment..."):
                out = _compute(results, settings)
            ss[k("result")] = {"settings": settings, **out}
            LOG.info("Enrichment (%s): %s, %d result rows", app_key, settings["method"], len(out["table"]))
        except Exception as e:
            LOG.error("Enrichment failed (%s):\n%s", app_key, traceback.format_exc())
            st.error(f"Enrichment could not run: {e}")

    res = ss.get(k("result"))
    if not res:
        return
    s, table = res["settings"], res["table"]
    if s != settings:
        st.caption(":material/info: Settings changed since these results. Press **Run enrichment** to update.")

    _mapping_note(res["report"], res["n_background"], " / ".join(s["library_titles"].values()))
    counts = res.get("counts")
    if s["method"] == "ORA" and counts is not None and not counts.empty and "query annotated" in counts:
        small = counts[counts["query annotated"] < 5]
        if len(small):
            worst = small.iloc[0]
            what = " ".join(str(x) for x in (worst.get("Glycan group"), worst.get("Direction")) if x and x == x)
            st.info(
                f"Only {int(worst['query annotated'])} significant protein(s) in at least one list ({what}). "
                "That is too few for over-representation to find anything. Classical t-tests with few "
                "replicates often leave few proteins below a q-value cutoff; limma usually finds more. "
                "GSEA uses every protein and needs no cutoff."
            )
    qcol = "Adjusted P-value" if s["method"] == "ORA" else "FDR q-value"
    if table is None or table.empty:
        st.info("No gene set had a significant protein (or none passed the size limits). Try a looser cutoff, "
                "both directions together, or a smaller minimum gene-set size.")
    else:
        n_sig = int((table[qcol] < 0.05).sum())
        st.success(f"{n_sig:,} of {len(table):,} tested gene sets have {qcol.lower()} < 0.05.")
        f1, f2 = st.columns([0.6, 0.4])
        show_all = f1.toggle(f"Show all tested gene sets (not only {qcol.lower()} < 0.05)", key=k("all"))
        top_n = f2.slider("Bars in plot", 5, 40, 15, key=k("topn"))
        shown = table if show_all else table[table[qcol] < 0.05]
        fmt = {c: st.column_config.NumberColumn(format="%.2e") for c in
               ("P-value", "Adjusted P-value", "FDR q-value") if c in shown.columns}
        fmt.update({c: st.column_config.NumberColumn(format="%.2f") for c in
                    ("Expected", "Fold enrichment", "NES", "ES") if c in shown.columns})
        st.dataframe(shown, width="stretch", hide_index=True, column_config=fmt)
        color = "Direction" if s["method"] == "ORA" and s.get("direction") == DIRECTIONS[0] else "Library"
        if s.get("group_by"):
            color = "Glycan group"
        fig = _bar_chart(shown if not shown.empty else table, ORA if s["method"] == "ORA" else GSEA, top_n, color)
        if fig is not None:
            st.plotly_chart(fig, width="stretch", key=k("plot"))

        from download_utility import add_dataframe_download_buttons, add_plotly_download_buttons
        st.markdown("###### Download")
        add_dataframe_download_buttons(table, base_filename=f"pathway_enrichment_{app_key}",
                                       key_suffix=k("dl_table"), sheet_name="Enrichment")
        if fig is not None:
            add_plotly_download_buttons(fig, base_filename=f"pathway_enrichment_{app_key}_plot",
                                        key_suffix=k("dl_plot"))

    with st.expander("Methods text and gene-set versions"):
        text = core.methods_paragraph(s, manifest, {})
        st.code(text, language=None, wrap_lines=True)
        libs = manifest["species"][s["species"]]["libraries"]
        st.markdown("\n".join(f"- **{libs[l]['title']}**: {libs[l]['source']}. {libs[l].get('citation', '')}"
                              for l in s["libraries"]))
        if res.get("counts") is not None and not res["counts"].empty:
            st.markdown("Counts per test")
            st.dataframe(res["counts"].dropna(axis=1, how="all"), width="stretch", hide_index=True)
