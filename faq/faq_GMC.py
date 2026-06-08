import streamlit as st

def faq_section():
    with st.expander("Frequently Asked Questions & Methodology", expanded=False):
        st.caption("Expand for analytical methodology, interpretation notes, and app operations.")
        
        items = [
            # General Information
            ("General Information", "What is the Glycopeptide Mass Calculator (GMC)?", "GMC is a workflow application designed for the in-silico fragmentation and mass derivation of glycopeptides."),
            ("General Information", "What downstream applications is this tool built for?", "The tool compiles fragmentation libraries into a structured `.xlsx` file designed specifically for Skyline transition lists and targeted extraction."),
            
            # Sequence & Configuration Inputs
            ("Sequence & Configuration Inputs", "How should I format the glycan tree structure?", "The glycan topology must be entered strictly using the nested parentheses pGlyco format. An example of this format is `(N(F)(N(H(H(N(H(A))))(H(N(H(A))))))))`."),
            ("Sequence & Configuration Inputs", "Are peptide modifications automatically calculated?", "Yes, Carbamidomethylation on Cysteine (C) is applied automatically as a fixed modification, adding a mass of 57.021464 Da to every Cysteine residue within the inputted sequence."),
            ("Sequence & Configuration Inputs", "How does the app determine the correct glycosylation site?", "The application automatically scans the peptide sequence for canonical N-linked sequons (an Asparagine (N), followed by any amino acid that is not Proline (P), followed by either Serine (S) or Threonine (T))."),
            ("Sequence & Configuration Inputs", "What if my sequence doesn't contain a standard N-linked sequon?", "If no canonical sequon is detected, the app provides a manual override where you can specify the 1-based index of the modified residue."),

            # In-silico Fragmentation Libraries
            ("In-silico Fragmentation Libraries", "What are HexNAc Stub Ions?", "These are site-specific backbone fragments (b and y ions) that retain a single HexNAc mass. The calculator specifically adds the HexNAc mass (203.07937 Da) to fragments that contain the defined glycosylation site."),
            ("In-silico Fragmentation Libraries", "What does the 'Intact Glycopeptide (Y-ions)' tab represent?", "This series represents the intact peptide backbone attached to various truncated glycan fragments resulting from branch breakages. The tool calculates these dynamically and tracks whether the resulting fragment retains a Fucose residue or not."),
            ("In-silico Fragmentation Libraries", "Which Oxonium Diagnostic reporter ions are generated?", "The calculator always generates base oxonium ions at 204.09 m/z and 366.14 m/z. If the glycan contains Sialic Acid ('A' or 'S'), diagnostic ions at 274.09, 292.10, and 657.23 m/z are dynamically added. If it contains Fucose ('F'), ions at 350.15 and 512.19 m/z are added."),

            # Exporting Data
            ("Exporting Data", "What is included in the Skyline Matrix Excel export?", "The export maps your in-silico transition lists across all calculated charge states. It generates individual sheets for the Backbone, Backbone+HexNAc, WholePeptide+GlycanFrags, Oxonium ions, and PrecursorMass, merging them into a final 'skylineInput' sheet formatted for direct use in Skyline.")
        ]
        
        current_cat = None
        for cat, q, a in items:
            if cat != current_cat:
                if current_cat is not None:
                    st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"##### {cat}")
                current_cat = cat
                
            with st.expander(q, expanded=False):
                st.markdown(a)
                
        st.markdown("---")
        st.caption("Contact: **baheti.saurabh@mayo.edu**")