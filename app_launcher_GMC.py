#!/usr/bin/env python3
"""
Glycopeptide Mass Calculator (GMC)
----------------------------------
In-silico fragmentation and mass derivation for glycopeptides.
"""

__author__ = ("Saurabh Baheti: baheti.saurabh@mayo.edu")

import streamlit as st
from collections import Counter
import re
import matplotlib.pyplot as plt
import pandas as pd
import io
from itertools import product
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# Import the custom FAQ module
from faq.faq_GMC import faq_section

# -----------------------------------------------------------------------------
# Streamlit page config & CSS
# -----------------------------------------------------------------------------
try:
    st.set_page_config(
        page_title="Glycopeptide Mass Calculator | Mayo Clinic",
        layout="wide",
    )
except Exception:
    pass

def inject_custom_css():
    st.markdown("""
        <style>
        /* Hide Streamlit Cruft */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {padding-top: 2rem !important; max-width: 1600px;}

        /* Material Design Card Styling */
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 12px !important;
            border: 1px solid #e0e6ed !important;
            background-color: #ffffff;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02), 0 4px 12px rgba(0, 0, 0, 0.04) !important;
            transition: all 0.25s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
            padding: 10px !important;
        }

        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05), 0 12px 24px rgba(0, 90, 156, 0.1) !important;
            border-color: #005a9c !important;
        }

        h1, h2, h3, h4 {
            color: #005a9c !important;
            font-weight: 600 !important;
            letter-spacing: -0.5px !important;
        }
        
        .subtitle-text {
            color: #5f6368 !important;
            font-size: 1.1em;
            font-weight: 400;
        }

        /* Modernized Buttons */
        .stButton>button {
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        .stButton>button:hover {
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            transform: translateY(-1px);
        }
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ---------------------------
# Monoisotopic atomic masses
# ---------------------------
ATOM_MASS = {
    'H': 1.007825,
    'C': 12.000000,
    'N': 14.003074,
    'O': 15.994915,
    'S': 31.972071
}

# Glycan building blocks
GLYCAN_MASS = {
    'H': 162.05282,   # Hex
    'N': 203.07937,   # HexNAc
    'F': 146.05791,   # Fucose
    'S': 291.09542,   # Sialic Acid
    'A': 291.09542    # NeuAc
}

# Amino acid residue compositions (after peptide bond formation)
AA_COMPOSITION = {
    'A': Counter({'C':3,'H':5,'N':1,'O':1}),
    'R': Counter({'C':6,'H':12,'N':4,'O':1}),
    'N': Counter({'C':4,'H':6,'N':2,'O':2}),
    'D': Counter({'C':4,'H':5,'N':1,'O':3}),
    'C': Counter({'C':3,'H':5,'N':1,'O':1,'S':1}),
    'E': Counter({'C':5,'H':7,'N':1,'O':3}),
    'Q': Counter({'C':5,'H':8,'N':2,'O':2}),
    'G': Counter({'C':2,'H':3,'N':1,'O':1}),
    'H': Counter({'C':6,'H':7,'N':3,'O':1}),
    'I': Counter({'C':6,'H':11,'N':1,'O':1}),
    'L': Counter({'C':6,'H':11,'N':1,'O':1}),
    'K': Counter({'C':6,'H':12,'N':2,'O':1}),
    'M': Counter({'C':5,'H':9,'N':1,'O':1,'S':1}),
    'F': Counter({'C':9,'H':9,'N':1,'O':1}),
    'P': Counter({'C':5,'H':7,'N':1,'O':1}),
    'S': Counter({'C':3,'H':5,'N':1,'O':2}),
    'T': Counter({'C':4,'H':7,'N':1,'O':2}),
    'W': Counter({'C':11,'H':10,'N':2,'O':1}),
    'Y': Counter({'C':9,'H':9,'N':1,'O':2}),
    'V': Counter({'C':5,'H':9,'N':1,'O':1})
}

PROTON_MASS = 1.007276

# ---------------------------
# Helper Functions
# ---------------------------
MOD_MASS = {
    'Carbamidomethyl(C)': 57.021464,
}

@dataclass
class Node:
    sym: str
    children: List["Node"] = field(default_factory=list)

    def clone(self) -> "Node":
        return Node(self.sym, [c.clone() for c in self.children])

def sanitize_pglyco(s: str) -> str:
    s = s.strip()
    while s and s[0] == "(":
        s = s[1:]
    opens, closes = s.count("("), s.count(")")
    while closes > opens and s.endswith(")"):
        s = s[:-1]
        closes -= 1
    return s

def parse_pglyco(s: str) -> Node:
    i = 0
    def parse(i: int) -> Tuple[Node, int]:
        if i >= len(s) or not s[i].isalpha():
            raise ValueError(f"Expected residue at {i}")
        n = Node(s[i])
        i += 1
        while i < len(s) and s[i] == "(":
            i += 1
            ch, i = parse(i)
            if i >= len(s) or s[i] != ")":
                raise ValueError(f"Expected ')' at {i}")
            n.children.append(ch)
            i += 1
        return n, i
    root, end = parse(0)
    if end != len(s):
        raise ValueError("Extra characters in glycan string")
    return root

def to_string(n: Optional[Node]) -> str:
    if n is None:
        return ""
    return n.sym if not n.children else n.sym + "".join(f"({to_string(c)})" for c in n.children)

def longest_path(root: Node) -> List[Node]:
    best = []
    def dfs(cur, path):
        nonlocal best
        path.append(cur)
        if not cur.children:
            if len(path) > len(best):
                best = path.copy()
        else:
            for ch in cur.children:
                dfs(ch, path)
        path.pop()
    dfs(root, [])
    return best

def find_branchpoint(root: Node) -> Tuple[Node, List[int]]:
    lp = longest_path(root)
    idx_path = []
    for depth, node in enumerate(lp):
        if len(node.children) >= 2:
            return node, idx_path.copy()
        if depth + 1 < len(lp):
            nxt = lp[depth + 1]
            for j, ch in enumerate(node.children):
                if ch is nxt:
                    idx_path.append(j)
                    break
    def dfs(n, path):
        if len(n.children) >= 2:
            return n, path
        for i, ch in enumerate(n.children):
            r = dfs(ch, path + [i])
            if r:
                return r
        return None
    bp = dfs(root, [])
    if not bp:
        raise ValueError("No branchpoint found")
    return bp

def all_node_paths(n: Node, base:Tuple[int,...]=()) -> List[Tuple[int,...]]:
    paths = [base]
    for i,ch in enumerate(n.children):
        paths.extend(all_node_paths(ch, base+(i,)))
    return paths

def remove_at_path(subtree: Node, path: Tuple[int,...]) -> Optional[Node]:
    if not path:
        return None
    new_root = subtree.clone()
    cur = new_root
    for idx in path[:-1]:
        cur = cur.children[idx]
    del cur.children[path[-1]]
    return new_root

def arm_drop_variants(arm_root: Node) -> List[Optional[Node]]:
    seen = set()
    variants = []
    for p in all_node_paths(arm_root, ()):
        new_arm = remove_at_path(arm_root, p)
        key = "__NONE__" if new_arm is None else to_string(new_arm)
        if key not in seen:
            seen.add(key)
            variants.append(new_arm)
    if to_string(arm_root) not in seen:
        variants.append(arm_root.clone())
    return variants

def apply_arm_choices(root: Node, bp_path: List[int],
                      left_idx:int,left_arm_new:Optional[Node],
                      right_idx:int,right_arm_new:Optional[Node]) -> Node:
    r = root.clone()
    cur = r
    for idx in bp_path:
        cur = cur.children[idx]
    new_children = []
    for i,ch in enumerate(cur.children):
        if i == left_idx:
            if left_arm_new is not None:
                new_children.append(left_arm_new.clone())
        elif i == right_idx:
            if right_arm_new is not None:
                new_children.append(right_arm_new.clone())
        else:
            new_children.append(ch.clone())
    cur.children = new_children
    return r

def to_HNFAS(glycan: str) -> str:
    counts = Counter([ch for ch in glycan if ch in ["H","N","F","A"]])
    return (f"H{counts['H']}"
            f"N{counts['N']}"
            f"F{counts['F']}"
            f"A{counts['A']}")

def root_stub_variants(root: Node) -> set[str]:
    variants = set()
    variants.add(root.sym)
    for ch in root.children:
        variants.add(root.sym + f"({ch.sym})")
    return variants

def generate_fragments(glycan: str) -> List[Tuple[str, str]]:
    root = parse_pglyco(sanitize_pglyco(glycan))
    original = to_string(root)
    out = set()
    out |= root_stub_variants(root)

    try:
        bp, bp_path = find_branchpoint(root)
        if len(bp.children) >= 2:
            L_idx, R_idx = 0, 1
            L_opts = arm_drop_variants(bp.children[L_idx])
            R_opts = arm_drop_variants(bp.children[R_idx])

            for L in L_opts:
                for R in R_opts:
                    t = apply_arm_choices(root, bp_path, L_idx, L, R_idx, R)
                    s = to_string(t)
                    if s != original:
                        out.add(s)
    except ValueError:
        pass

    if not out or out == root_stub_variants(root):
        s = sanitize_pglyco(glycan)
        fragments = []
        while True:
            last_open = s.rfind("(")
            last_close = s.find(")", last_open)
            if last_open == -1 or last_close == -1:
                break
            truncated = s[:last_open] + s[last_close + 1:]
            truncated = truncated.strip()
            if truncated and truncated != s:
                fragments.append(truncated)
            s = truncated
        for f in fragments:
            temp = f
            while "(" in temp:
                lo = temp.rfind("(")
                lc = temp.find(")", lo)
                if lo == -1 or lc == -1:
                    break
                sub = temp[:lo] + temp[lc+1:]
                sub = sub.strip()
                if sub and sub != temp:
                    fragments.append(sub)
                temp = sub
        out.update(set(fragments))

    final = []
    for frag in sorted(out, key=lambda x: (len(x), x)):
        try:
            final.append((frag, to_HNFAS(frag)))
        except Exception:
            continue
    return final

def generate_unique_fragments_with_fucose(glycan_str: str):
    results = []
    seen = set()

    if "(F)" in glycan_str:
        glycan_no_fuc = glycan_str.replace("(F)", "")
        frags_without = generate_fragments(glycan_no_fuc)
        for frag, hnfas in frags_without:
            if (frag, hnfas) not in seen:
                seen.add((frag, hnfas))
                results.append((frag, hnfas, "Without F"))

        frags_with = generate_fragments(glycan_str)
        for frag, hnfas in frags_with:
            if "F0" in hnfas:
                continue
            if (frag, hnfas) not in seen:
                seen.add((frag, hnfas))
                results.append((frag, hnfas, "With F"))
    else:
        frags = generate_fragments(glycan_str)
        for frag, hnfas in frags:
            if (frag, hnfas) not in seen:
                seen.add((frag, hnfas))
                results.append((frag, hnfas, "Without F"))

    count_with = len([1 for _, _, src in results if src == "With F"])
    count_without = len([1 for _, _, src in results if src == "Without F"])
    total = len(results)

    return results, count_with, count_without, total

def peptide_composition(seq):
    comp = Counter()
    for aa in seq:
        if aa not in AA_COMPOSITION:
            raise ValueError(f"Unknown amino acid: {aa}")
        comp += AA_COMPOSITION[aa]
    comp += Counter({'H':2,'O':1})
    return comp

def composition_mass(comp):
    return sum(ATOM_MASS[e]*n for e,n in comp.items())

def glycan_mass(glycan_str) -> float:
    total = 0.0
    matches = re.findall(r'([HNFAS])(\d+)', glycan_str)
    for symbol, count in matches:
        if symbol in GLYCAN_MASS:
            total += GLYCAN_MASS[symbol] * int(count)
    return total

def parse_pglyco_annotation(pglyco_str):
    comp = Counter()
    for m in re.findall(r'[HNFA]', pglyco_str):
        comp[m] += 1
    return "".join(f"{k}{comp[k]}" for k in "HNFA" if comp[k] > 0)

def mz_values(neutral_mass, charges=(1,2,3)):
    return {z: (neutral_mass + z*PROTON_MASS)/z for z in charges}

def aa_mass(aa):
    comp = AA_COMPOSITION[aa]
    return sum(ATOM_MASS[e]*n for e,n in comp.items())

def fragment_ions(seq, fixed_mods=True):
    n = len(seq)
    ions = []
    residue_masses = []
    for aa in seq:
        base_mass = aa_mass(aa)
        if aa == "C" and fixed_mods:
            base_mass += 57.02146
        residue_masses.append(base_mass)

    for i in range(1, n):
        b_mass = sum(residue_masses[:i])
        ions.append((f"b{i}", i, seq[:i],
                     round((b_mass+PROTON_MASS)/1,2),
                     round((b_mass+2*PROTON_MASS)/2,2),
                     round((b_mass+3*PROTON_MASS)/3,2)))
        y_mass = sum(residue_masses[i:]) + 18.01056
        ions.append((f"y{n-i}", n-i, seq[i:],
                     round((y_mass+PROTON_MASS)/1,2),
                     round((y_mass+2*PROTON_MASS)/2,2),
                     round((y_mass+3*PROTON_MASS)/3,2)))
    return ions

def oxonium_ions(glycan_str=""):
    ions = {
        "Ox_204.09": 204.09,
        "Ox_366.14": 366.14
    }
    glycan_upper = glycan_str.upper()
    has_sialic = any(k in glycan_upper for k in ["A", "S"])
    has_fucose = "F" in glycan_upper

    if has_sialic:
        ions.update({"Ox_274.09": 274.09, "Ox_292.10": 292.10, "Ox_657.23": 657.23})
    if has_fucose:
        ions.update({"Ox_350.15": 350.15, "Ox_512.19": 512.19})
    return [(k, v) for k, v in ions.items()]

def peptide_mass(seq, cam_cys=True):
    base = composition_mass(peptide_composition(seq))
    if cam_cys:
        base += seq.count("C") * 57.021464
    return base

def parse_pglyco_tree_to_counts(s: str) -> Counter:
    comp = Counter()
    def recurse(t):
        i=0
        while i<len(t):
            c = t[i]
            if c in "HNFA":
                comp[c]+=1
                i+=1
            elif c=="(":
                depth,j=1,i+1
                while j<len(t) and depth>0:
                    if t[j]=="(": depth+=1
                    elif t[j]==")": depth-=1
                    j+=1
                recurse(t[i+1:j-1])
                i=j
            else:
                i+=1
    if s.startswith("(") and s.endswith(")"):
        s=s[1:-1]
    recurse(s)
    return Counter({k:comp[k] for k in ("H","N","A","F") if comp[k]>0})

def glycan_mass_from_counts(c: Counter):
    return sum(c.get(k,0)*GLYCAN_MASS[k] for k in GLYCAN_MASS)

def counts_to_string(c: Counter) -> str:
    return "".join(f"{k}{c[k]}" for k in "HNAF" if c.get(k,0)>0)

def enumerate_glycoforms(seq, glycan_tree, charges=(1,2,3), cam_cys=True):
    pep = peptide_mass(seq, cam_cys=cam_cys)
    comp_full = parse_pglyco_tree_to_counts(glycan_tree)
    rows=[]
    for H,N,A,F in product(range(comp_full["H"]+1),
                           range(comp_full["N"]+1),
                           range(comp_full["A"]+1),
                           range(comp_full["F"]+1)):
        c=Counter({"H":H,"N":N,"A":A,"F":F})
        gmass=glycan_mass_from_counts(c)
        neutral=pep+gmass
        label="Y+"+counts_to_string(c)
        row={"Name":label}
        for z in charges:
            row[f"m/z ({z}+)" ]=round((neutral+z*PROTON_MASS)/z,2)
        rows.append(row)
    return pd.DataFrame(rows)

def filter_plausible_glycoforms(df: pd.DataFrame) -> pd.DataFrame:
    parts = df["Name"].str.extract(r'H(\d+)N(\d+)A(\d+)F(\d+)').fillna(0).astype(int)
    parts.columns = ["H", "N", "A", "F"]
    mask = []
    for h, n, a, f in parts.itertuples(index=False):
        if n in (0, 1):
            mask.append(True)
            continue
        if n < 2:
            mask.append(False)
            continue
        if a > 0 and not (h >= 4 and n >= 3):
            mask.append(False)
            continue
        mask.append(True)
    return df[pd.Series(mask, index=df.index)].reset_index(drop=True)

def export_to_excel(dfs, peptide, glycan, gene_site=""):
    pep_mass = peptide_mass(peptide, cam_cys=("C" in peptide))
    gly_mass = glycan_mass(glycan)
    total_mass = pep_mass + gly_mass
    molecule_name = f"{gene_site}_{peptide}_{glycan}" if gene_site else f"UNSPECIFIED_{glycan}"
    output = io.BytesIO()

    precursor_rows = []
    mz_dict = mz_values(total_mass, charges=(1,2,3,4,5))
    for z, mz in mz_dict.items():
        precursor_rows.append({
            "Peptide": peptide,
            "Glycan": glycan,
            "Charge": z,
            "m/z": round(mz, 4)
        })
    df_precursor = pd.DataFrame(precursor_rows)

    all_rows = []
    def add_fragment_rows(df, ion_col, frag_mz_cols, ion_label_func=None):
        if ion_label_func is None:
            ion_label_func = lambda r, col=ion_col: r[col]
        for _, row in df.iterrows():
            for frag_charge, frag_mz in frag_mz_cols(row).items():
                for prec_charge in range(1, 6):
                    prec_mz = mz_values(total_mass, charges=(prec_charge,))[prec_charge]
                    all_rows.append({
                        "Molecule": molecule_name,
                        "Gene_Site": gene_site,
                        "Precursor m/z": round(prec_mz, 4),
                        "Precursor charge": prec_charge,
                        "Fragment ion": ion_label_func(row),
                        "Fragment m/z": round(frag_mz, 2),
                        "Fragment charge": frag_charge
                    })

    if "Backbone" in dfs:
        add_fragment_rows(dfs["Backbone"], "Ion", lambda r: {1: r["m/z+1"], 2: r["m/z+2"], 3: r["m/z+3"]})
    if "Backbone+HexNAc" in dfs:
        add_fragment_rows(dfs["Backbone+HexNAc"], "Ion", lambda r: {1: r["m/z+1"], 2: r["m/z+2"], 3: r["m/z+3"]})
    if "WholePeptide+GlycanFrags" in dfs:
        add_fragment_rows(dfs["WholePeptide+GlycanFrags"], "Fragment (HNFAS)", 
                          lambda r: {1: r["m/z (z=1)"], 2: r["m/z (z=2)"], 3: r["m/z (z=3)"]},
                          ion_label_func=lambda r: f"Y+{r['Fragment (HNFAS)'].replace('Y+','')}")
    if "Oxonium" in dfs:
        for _, row in dfs["Oxonium"].iterrows():
            prec_mz = mz_values(total_mass, charges=(1,))[1]
            all_rows.append({
                "Molecule": molecule_name,
                "Gene_Site": gene_site,
                "Precursor m/z": round(prec_mz, 4),
                "Precursor charge": 1,
                "Fragment ion": row["Ion"],
                "Fragment m/z": round(row["m/z"], 2),
                "Fragment charge": 1
            })

    df_all = pd.DataFrame(all_rows)
    df_all["_orig_idx"] = df_all.index
    df_all = df_all.sort_values(by=["Precursor charge", "_orig_idx"], ascending=[True, True]).drop(columns="_orig_idx").reset_index(drop=True)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, df in dfs.items():
            df.to_excel(writer, sheet_name=name, index=False)
        df_precursor.to_excel(writer, sheet_name="PrecursorMass", index=False)
        df_all.to_excel(writer, sheet_name="skylineInput", index=False)

    file_name = f"{molecule_name}_{round(total_mass,2)}.xlsx"
    return output.getvalue(), file_name

def plot_fragmentation(seq):
    n = len(seq)
    fig, ax = plt.subplots(figsize=(max(4, len(seq)*0.6), 2))
    for i, aa in enumerate(seq, start=1):
        ax.text(i, 0, aa, ha='center', va='center', fontsize=14, fontweight='bold')
    for i in range(1, n):
        ax.annotate(f"b{i}", xy=(i+0.5, 0.3), xytext=(i+0.5, 0.8),
                    arrowprops=dict(arrowstyle='-|>', lw=1.2, color='#d62728'),
                    ha='center', fontsize=10, color='#d62728', fontweight='bold')
        ax.annotate(f"y{n-i}", xy=(i+0.5, -0.3), xytext=(i+0.5, -0.8),
                    arrowprops=dict(arrowstyle='-|>', lw=1.2, color='#1f77b4'),
                    ha='center', fontsize=10, color='#1f77b4', fontweight='bold')
    ax.set_xlim(0.5, n+0.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')
    return fig

def find_nglyc_sites(seq):
    sites = []
    for i in range(len(seq)-2):
        if seq[i]=="N" and seq[i+1]!="P" and seq[i+2] in ("S","T"):
            sites.append(i+1)
    return sites

def fragment_ions_with_hexnac_stub(seq, gly_site, fixed_mods=True):
    ions = fragment_ions(seq, fixed_mods=fixed_mods)
    stub_mass = GLYCAN_MASS["N"]
    stub_ions = []
    for ion, pos, subseq, m1, m2, m3 in ions:
        add_stub = False
        if ion.startswith("b") and pos >= gly_site:
            add_stub = True
        elif ion.startswith("y"):
            c_term_start = len(seq) - (int(ion[1:])) + 1
            if c_term_start <= gly_site:
                add_stub = True
        if add_stub:
            stub_ions.append((
                ion + "+HexNAc",
                round((m1*1 - PROTON_MASS + stub_mass + PROTON_MASS)/1,2),
                round((m2*2 - PROTON_MASS + stub_mass + PROTON_MASS)/2,2),
                round((m3*3 - PROTON_MASS + stub_mass + PROTON_MASS)/3,2)
            ))
    return stub_ions

def normalize_labels(df, col="Name"):
    def expand_label(label):
        counts = {"H": 0, "N": 0, "A": 0, "F": 0}
        matches = re.findall(r'([HNAF])(\d+)', label)
        for sym, num in matches:
            counts[sym] = int(num)
        return f"Y+H{counts['H']}N{counts['N']}A{counts['A']}F{counts['F']}"
    df[col] = df[col].apply(expand_label)
    return df

# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main():
    #st.title("Glycopeptide Mass Calculator (GMC)")
    #st.markdown("<p class='subtitle-text'>In-silico Fragmentation & Precursor Derivation Workflow</p>", unsafe_allow_html=True)

    # ============================================================
    # Sidebar - Workflow Configuration
    # ============================================================
    with st.sidebar:
        st.markdown("### Molecular Configuration")
        st.caption("Define sequence and glycan parameters")
        st.divider()

        peptide = st.text_input(
            "Peptide Sequence", 
            value="CGLVPVLAENYNK",
            help="Enter the unmodified amino acid sequence. Carbamidomethylation on Cysteine is applied automatically."
        ).strip().upper()

        glycan_in = st.text_input(
            "Glycan Tree (pGlyco Format)", 
            value="(N(F)(N(H(H(N(H(A))))(H(N(H(A))))))))",
            help="Enter the glycan topology strictly using the nested parentheses pGlyco format."
        ).strip()

        gene_site = st.text_input(
            "Gene & Site Identifier", 
            value="TF-432",
            help="Optional identifier to append to exported matrices (e.g., TargetProtein-N123)."
        ).strip().upper()

        st.divider()
        st.markdown("**Glycosylation Site Definition**")
        
        sites = find_nglyc_sites(peptide)
        chosen_site = None
        
        if sites:
            chosen_site = st.selectbox(
                "Detected N-x-S/T Sequons", 
                options=sites,
                help="Select the specific Asparagine (N) residue index hosting the glycan."
            )
        else:
            st.warning("No canonical N-linked sequons detected in sequence.")
            manual_site = st.text_input("Manual N-site Index", help="Specify the 1-based index of the modified residue.")
            if manual_site:
                clean_site = manual_site.strip().upper().replace("N", "")
                if clean_site.isdigit():
                    chosen_site = int(clean_site)

    # ============================================================
    # Parse & Display Molecular Characteristics
    # ============================================================
    glycan = parse_pglyco_annotation(glycan_in) if "(" in glycan_in else glycan_in
    pep_mass = peptide_mass(peptide, cam_cys=("C" in peptide))
    gly_mass = glycan_mass(glycan)
    total_mass = pep_mass + gly_mass

    with st.container(border=True):
        st.markdown("#### Molecular Mass Characteristics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Peptide Backbone Mass", f"{pep_mass:.4f} Da")
        m2.metric("Intact Glycan Mass", f"{gly_mass:.4f} Da")
        m3.metric("Total Precursor Mass", f"{total_mass:.4f} Da")

    if not chosen_site:
        st.info("Define a glycosylation site in the sidebar configuration to generate the fragmentation library.")
        return

    # ============================================================
    # Fragmentation Library Generation
    # ============================================================
    with st.container(border=True):
        st.markdown("#### Theoretical Fragmentation Topology")
        fig = plot_fragmentation(peptide)
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("#### In-silico Fragmentation Library")
        
        # Calculate all fragments silently
        ions = fragment_ions(peptide)
        df_backbone = pd.DataFrame(ions, columns=["Ion", "Position", "Sequence", "m/z+1", "m/z+2", "m/z+3"])
        
        df_stub = pd.DataFrame(fragment_ions_with_hexnac_stub(peptide, chosen_site), columns=["Ion", "m/z+1", "m/z+2", "m/z+3"])
        
        frags_comb, c_with, c_without, c_total = generate_unique_fragments_with_fucose(glycan_in)
        
        rows = []
        for frag, hnfas, src in frags_comb:
            g_m = glycan_mass(hnfas)
            gp_m = pep_mass + g_m
            rows.append({
                "Fragment (Tree)": frag,
                "Fragment (HNFAS)": f"Y+{hnfas}",
                "Source": src,
                "Glycan Mass": round(g_m, 4),
                "Glycopeptide Mass": round(gp_m, 4),
                "m/z (z=1)": round(gp_m / 1, 2),
                "m/z (z=2)": round(gp_m / 2, 2),
                "m/z (z=3)": round(gp_m / 3, 2),
            })
        df_frag = pd.DataFrame(rows)
        
        df_ox = pd.DataFrame(oxonium_ions(glycan_in), columns=["Ion", "m/z"])

        # Tabbed Navigation
        tabs = st.tabs([
            "Peptide Backbone (b/y-ions)", 
            "HexNAc Stub Ions", 
            "Intact Glycopeptide (Y-ions)", 
            "Oxonium Diagnostics"
        ])

        with tabs[0]:
            st.markdown("**Unmodified Peptide Fragments**")
            st.dataframe(df_backbone, use_container_width=True, hide_index=True)

        with tabs[1]:
            st.markdown("**Site-Specific Backbone + HexNAc Modification**")
            st.dataframe(df_stub, use_container_width=True, hide_index=True)

        with tabs[2]:
            st.markdown("**Y-ion Series (Peptide + Truncated Glycan)**")
            st.caption(f"Calculated unique fragments: {c_total} (With Fucose: {c_with} | Without: {c_without})")
            st.dataframe(df_frag, use_container_width=True, hide_index=True)

        with tabs[3]:
            st.markdown("**Diagnostic Oxonium Reporter Ions**")
            st.dataframe(df_ox, use_container_width=True, hide_index=True)

    # ============================================================
    # Export Module
    # ============================================================
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        col_info, col_act = st.columns([0.7, 0.3])
        with col_info:
            st.markdown("#### Export Analysis Artifacts")
            st.markdown(
                "<p class='subtitle-text'>Download the compiled fragmentation library mapped across all calculated charge states.</p>", 
                unsafe_allow_html=True
            )
            st.caption("Generates a structured `.xlsx` file suitable for Skyline transition lists and downstream targeted extraction.")
        
        with col_act:
            st.write("")
            dfs_dict = {
                "Backbone": df_backbone, 
                "Backbone+HexNAc": df_stub,
                "WholePeptide+GlycanFrags": df_frag, 
                "Oxonium": df_ox
            }
            excel_data, file_name = export_to_excel(dfs_dict, peptide, glycan, gene_site)
            
            st.download_button(
                label="Download Skyline Matrix (Excel)", 
                data=excel_data, 
                file_name=file_name,
                use_container_width=True,
                type="primary",
                help="Compile and export the in-silico transition lists for all backbone, stub, Y-ion, and oxonium fragments."
            )

    # Append the FAQ section to the bottom of the app
    faq_section()

if __name__ == "__main__":
    main()