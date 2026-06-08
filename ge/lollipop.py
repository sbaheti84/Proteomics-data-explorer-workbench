#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Centralized helper for fallback download options if needed
try:
    from ge.misc_utility import add_plot_download_buttons
except ImportError:
    pass


def create_lollipop_interactive(df, gene: str, glycan_col: str = "Glycan", sig_col: str = "q-value", use_legend: bool = True):
    """
    Interactive Horizontal Lollipop plot using Plotly.
    Maintains continuous color-coding based on significance (-log10 q-value),
    handles missing library structures dynamically, and includes standalone PNG/SVG download buttons.
    """
    # ---------- Filter ----------
    subset = df[df["Genes"].str.contains(gene, case=False, na=False)].copy()
    if subset.empty:
        st.warning(f"No peptides found for gene: {gene}")
        return

    # ---------- Parse numeric ProSite ----------
    def _parse_site(val):
        nums = re.findall(r"\d+", str(val))
        return int(nums[0]) if nums else np.nan

    subset["ProSiteNum"] = subset["ProSites"].apply(_parse_site)
    subset = subset.dropna(subset=["ProSiteNum"])
    subset["neglog10q"] = -np.log10(subset[sig_col].astype(float).replace(0, np.nan)).fillna(0)
    subset = subset.sort_values("ProSiteNum")
    unique_sites = sorted(subset["ProSiteNum"].unique())

    # Get global min/max for significance to fix the colorbar range
    min_sig = subset["neglog10q"].min()
    max_sig = subset["neglog10q"].max()
    if min_sig == max_sig:
        max_sig += 1.0

    # ---------- Dynamic Horizontal Positioning ----------
    GLYCAN_SPACING = 0.35  
    MIN_SITE_GAP = 1.2     

    site_spreads = {}
    for site, group in subset.groupby("ProSiteNum"):
        up_count = len(group[group["log2FC"] > 0.01])
        down_count = len(group[group["log2FC"] < -0.01])
        max_count = max(up_count, down_count)
        site_spreads[site] = max(0.2, (max_count - 1) * GLYCAN_SPACING)

    site_map = {}
    current_x = 0
    for i, site in enumerate(unique_sites):
        if i == 0:
            site_map[site] = 0
        else:
            prev_site = unique_sites[i-1]
            dist = (site_spreads[prev_site] / 2) + (site_spreads[site] / 2) + MIN_SITE_GAP
            current_x += dist
            site_map[site] = current_x

    actual_min_x = site_map[unique_sites[0]] - (site_spreads[unique_sites[0]] / 2)
    actual_max_x = site_map[unique_sites[-1]] + (site_spreads[unique_sites[-1]] / 2)

    # ---------- Adaptive scaling ----------
    BASE_LENGTH = 0.1
    MAX_STALK = 2.8
    GAP = 0.8  
    
    fc_max = np.nanmax(np.abs(subset["log2FC"])) if len(subset) else 1.0
    SCALE = MAX_STALK / fc_max 

    # Initialize Plotly Figure
    fig = go.Figure()

    # Protein backbone line
    fig.add_shape(type="line",
                  x0=actual_min_x - 1.0, y0=0, x1=actual_max_x + 1.0, y1=0,
                  line=dict(color="black", width=4), layer="above")

    # Dynamic Map markers for Legend shapes
    available_markers = ['circle', 'square', 'triangle-up', 'diamond', 'triangle-down', 'hexagram', 'star']
    
    # Process compositions and assign fallbacks where structure is missing
    processed_glycans = []
    for _, row in subset.iterrows():
        glycan_val = str(row.get(glycan_col, ""))
        if pd.isna(row.get(glycan_col)) or glycan_val.strip() == "" or "not available" in glycan_val.lower() or "unknown" in glycan_val.lower():
            h = int(row.get('Hex', 0)) if 'Hex' in row else 0
            n = int(row.get('HexNAc', 0)) if 'HexNAc' in row else 0
            f = int(row.get('Fuc', 0)) if 'Fuc' in row else 0
            a = int(row.get('NeuAc', 0)) if 'NeuAc' in row else 0
            glycan_val = f"{h} {n} {f} {a}"
        processed_glycans.append(glycan_val)
        
    subset["DisplayGlycan"] = processed_glycans
    unique_glycans = sorted(list(subset["DisplayGlycan"].unique()))
    marker_map = {g: available_markers[i % len(available_markers)] for i, g in enumerate(unique_glycans)}

    # Track if colorbar is added to prevent duplicates
    colorbar_added = False

    # ---------- Draw per site ----------
    for site, group in subset.groupby("ProSiteNum"):
        xpos = site_map[site]
        up = group[group["log2FC"] > 0.01]
        down = group[group["log2FC"] < -0.01]

        for direction, side_group in [(1, up), (-1, down)]:
            count = len(side_group)
            if count == 0:
                continue

            fan_width = (count - 1) * GLYCAN_SPACING
            x_ends = np.linspace(xpos - fan_width / 2, xpos + fan_width / 2, count) if count > 1 else [xpos]

            for (_, row), x_e in zip(side_group.iterrows(), x_ends):
                log2fc = float(row["log2FC"])
                qsig = float(row["neglog10q"])
                glycan_display = row["DisplayGlycan"]

                y_start = direction * BASE_LENGTH
                y_kink = direction * GAP 
                actual_length = abs(log2fc) * SCALE
                y_end = y_kink + direction * actual_length

                # Draw Stalk Line
                fig.add_trace(go.Scatter(
                    x=[xpos, x_e, x_e],
                    y=[y_start, y_kink, y_end],
                    mode='lines',
                    line=dict(color='black', width=1.2),
                    hoverinfo='skip',
                    showlegend=False
                ))

                # Setup colorbar configuration for the first leaf plotted
                marker_dict = dict(
                    size=15,
                    symbol=marker_map.get(glycan_display, 'circle') if use_legend else 'circle',
                    color=[qsig],
                    cmin=min_sig,
                    cmax=max_sig,
                    colorscale='Viridis',
                    reversescale=True,
                    line=dict(color='black', width=1)
                )

                if not colorbar_added:
                    marker_dict['showscale'] = True
                    marker_dict['colorbar'] = dict(
                        title=dict(text=f"<b>-log<sub>10</sub>({sig_col})</b>", font=dict(size=11)),
                        thickness=15,
                        len=0.5,
                        yanchor="middle",
                        y=0.5,
                        xanchor="left",
                        x=1.22 if use_legend else 1.05
                    )
                    colorbar_added = True
                else:
                    marker_dict['showscale'] = False

                # Draw Interactive Lollipop Head (Leaf)
                fig.add_trace(go.Scatter(
                    x=[x_e],
                    y=[y_end],
                    mode='markers',
                    name=glycan_display if use_legend else "",
                    legendgroup=glycan_display if use_legend else "",
                    showlegend=False,  
                    marker=marker_dict,
                    hovertemplate=f"<b>Composition:</b> {glycan_display}<br><b>log2FC:</b> {log2fc:.2f}<br><b>-log10(q):</b> {qsig:.2f}<extra></extra>"
                ))

    # Add centralized clean Mock Legend Elements for shapes
    if use_legend:
        for g_val in unique_glycans:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=11, symbol=marker_map.get(g_val, 'circle'), color='gray', line=dict(color='black')),
                name=g_val,
                showlegend=True
            ))

    # ---------- Dynamic Split Axis Formatting ----------
    if fc_max <= 1.0: step = 0.2
    elif fc_max <= 2.5: step = 0.5
    elif fc_max <= 5.0: step = 1.0
    else: step = 2.0

    raw_ticks = np.arange(0, fc_max + step, step)
    pos_ticks = [t * SCALE + GAP for t in raw_ticks]
    neg_ticks = [-t * SCALE - GAP for t in raw_ticks if t != 0]
    all_ticks = sorted(neg_ticks + pos_ticks)
    
    tick_labels = []
    for tick in all_ticks:
        val = (tick - GAP) / SCALE if tick > 0 else (tick + GAP) / SCALE
        if abs(val) < 0.001: val = 0.0
        tick_labels.append(f"{val:g}")

    # Layout styling updates
    fig.update_layout(
        title=dict(text=f"<b>{gene}</b>", font=dict(size=18), xanchor="center", x=0.5, y=0.95,yanchor="top"),
        xaxis=dict(
            title=dict(text="<b>Glycosylation site</b>", font=dict(size=13)),  
            tickmode='array',
            tickvals=list(site_map.values()),
            ticktext=unique_sites,
            tickangle=-90,
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            title=dict(text="<b>log<sub>2</sub>(Fold Change)</b>", font=dict(size=13)),
            tickmode='array',
            tickvals=all_ticks,
            ticktext=tick_labels,
            showgrid=False,
            zeroline=False
        ),
        template="plotly_white",
        height=620,
        margin=dict(l=70, r=40, t=60, b=90),
        legend=dict(
            title=dict(text=f"<b>{glycan_col}</b>", font=dict(size=12)),
            orientation="v",
            yanchor="middle",  
            y=0.5,
            xanchor="left",
            x=1.03
        )
    )

    # Render Plotly graph in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # ---------- Restored Standalone Download Interface ----------
    try:
        col1, col2 = st.columns(2)
        
        # Build image payloads using engine background library (Kaleido)
        png_data = fig.to_image(format="png", width=1200, height=650, scale=2)
        svg_data = fig.to_image(format="svg", width=1200, height=650)
        
        with col1:
            st.download_button(
                label="📥 Download PNG",
                data=png_data,
                fileName=f"{gene}_lollipop.png",
                mime="image/png"
            )
        with col2:
            st.download_button(
                label="📥 Download SVG",
                data=svg_data,
                fileName=f"{gene}_lollipop.svg",
                mime="image/svg+xml"
            )
    except Exception as e:
        # Fallback handling if kaleido isn't properly built inside environment setup
        st.info("Interactive modebar export available via camera icon on top right corner of graph.")


def create_lollipop(df, gene: str, sig_col: str = "q-value"):
    """Original signature legacy wrapper."""
    create_lollipop_interactive(df, gene, sig_col=sig_col, use_legend=False)


def create_lollipop_legend(df: pd.DataFrame, gene: str, glycan_col: str = "Glycan", sig_col: str = "q-value"):
    """Updated legend wrapper using fallback compositions on top."""
    create_lollipop_interactive(df, gene, glycan_col=glycan_col, sig_col=sig_col, use_legend=True)