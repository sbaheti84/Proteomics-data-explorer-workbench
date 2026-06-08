import streamlit as st
import pandas as pd
from streamlit.errors import StreamlitAPIException

try:
    from st_aggrid import AgGrid, GridOptionsBuilder
except Exception:
    AgGrid = None
    GridOptionsBuilder = None

def try_set_page_config(*args, **kwargs):
    try:
        st.set_page_config(*args, **kwargs)
    except StreamlitAPIException:
        pass

# =============================================================================
# CSS Material Design Integration
# =============================================================================
def inject_custom_css():
    st.markdown("""
        <style>
        /* Hide Streamlit Cruft */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {padding-top: 2rem !important; max-width: 1600px;}

        /* Material Design Card Styling for border=True containers */
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 12px !important;
            border: 1px solid #e0e6ed !important;
            background-color: #ffffff;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02), 0 4px 12px rgba(0, 0, 0, 0.04) !important;
            transition: all 0.25s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
            padding: 10px !important;
        }

        /* Card Hover Animation */
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05), 0 12px 24px rgba(0, 90, 156, 0.1) !important;
            border-color: #005a9c !important;
        }

        /* Typography Harmonization */
        h1, h2, h3, h4 {
            color: #005a9c !important;
            font-weight: 600 !important;
            letter-spacing: -0.5px !important;
        }
        
        /* Softer subtile text */
        .subtitle-text {
            color: #5f6368 !important;
            font-size: 1.1em;
            font-weight: 400;
        }

        /* Upgrade Tabs to look like modern web tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 16px;
            border-bottom: 1px solid #e0e6ed;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 6px 6px 0px 0px;
            padding-top: 10px;
            padding-bottom: 10px;
            color: #5f6368;
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            background-color: #f0f7ff !important;
            border-bottom: 3px solid #005a9c !important;
            color: #005a9c !important;
            font-weight: 600 !important;
        }

        /* Clean DataFrames */
        [data-testid="stDataFrame"] {
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #e0e6ed;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
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
        
        /* Expanders */
        .streamlit-expanderHeader {
            font-weight: 600 !important;
            border-bottom: 1px solid #f0f4f8;
            color: #202124 !important;
        }
        </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------
def show_grid(df, page_size=20, height=500):
    if GridOptionsBuilder and AgGrid:
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=page_size)
        gb.configure_default_column(resizable=True, filterable=True, sortable=True)
        grid_options = gb.build()
        AgGrid(
            df,
            gridOptions=grid_options,
            fit_columns_on_grid_load=False,
            enable_enterprise_modules=False,
            allow_unsafe_jscode=True,
            height=height,
            width="100%",
            theme="material",
        )
    else:
        st.dataframe(df, use_container_width=True, height=height)

def df_is_nonempty(df):
    return isinstance(df, pd.DataFrame) and not df.empty

def df_is_empty(df):
    return not df_is_nonempty(df)