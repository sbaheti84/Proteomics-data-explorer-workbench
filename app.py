#!/usr/bin/env python3

import os
import sys
import time
import importlib
import importlib.util
import traceback

# Desktop/PyInstaller build: browser download buttons are unreliable in the embedded
# Streamlit desktop/webview runtime. Force local Save As mode for the packaged exe.
if getattr(sys, "frozen", False) or hasattr(sys, "_MEIPASS"):
    os.environ.setdefault("GLYCOEXPLORER_HOSTED", "false")
    os.environ.setdefault("GLYCOEXPLORER_SAVE_MODE", "dialog")

import streamlit as st


# =============================================================================
# PYINSTALLER / STREAMLIT-DESKTOP COMPATIBILITY
# =============================================================================
def get_resource_path(relative_path=""):
    """
    Return an absolute path that works both during normal Streamlit execution
    and inside a PyInstaller / streamlit-desktop-app packaged build.
    """
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.normpath(os.path.join(base_path, relative_path))


# Runtime root directory used by both local execution and packaged builds.
current_dir = get_resource_path("")
bundle_dir = getattr(sys, "_MEIPASS", current_dir)

for p in [current_dir, bundle_dir, os.path.dirname(os.path.abspath(__file__))]:
    if p and p not in sys.path:
        sys.path.insert(0, p)


def resource_path(*parts):
    """
    Compatibility wrapper used by this dashboard for assets.
    """
    return get_resource_path(os.path.join(*parts))


def load_module_safely(module_name: str):
    """
    Load <module_name>.py dynamically from the runtime root directory.

    This follows the same approach used in the Lipidomics Workbench launcher and
    is more robust for PyInstaller/streamlit-desktop-app standalone builds than
    a plain importlib.import_module call.
    """
    module_file = get_resource_path(f"{module_name}.py")

    if not os.path.exists(module_file):
        files_found = []
        try:
            files_found = os.listdir(current_dir)
        except Exception:
            pass

        raise FileNotFoundError(
            f"Could not find expected launcher file:\n{module_file}\n\n"
            f"Files found in app directory:\n{files_found}"
        )

    # Reload cleanly when switching between apps during the same Streamlit session.
    if module_name in sys.modules:
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, module_file)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build import spec for: {module_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"While loading '{module_name}.py', Python could not import: '{e.name}'. "
            f"The file exists, but one dependency/module is missing."
        ) from e

    return module


# =============================================================================
# CONFIGURATION
# =============================================================================
DASHBOARD_OPTION = "-- Select an app --"

APP_MODULES = {
    "GlycoExplorer (TMT)": {
        "module": "app_launcher_GE",
        "desc": "Analyze and visualize TMT glycoproteomics results.",
        "category": "GlycoProteomics",
    },
    "GlycoExplorer (Label-Free)": {
        "module": "app_launcher_GE_LF",
        "desc": "Analyze and visualize label-free glycoproteomics results.",
        "category": "GlycoProteomics",
    },
    "Total Proteome (TMT)": {
        "module": "app_launcher_TP",
        "desc": "Perform proteome-level QC, normalization, and reporting.",
        "category": "Proteomics",
    },
    "Total Proteome (Label-Free)": {
        "module": "app_launcher_TP_LF",
        "desc": "Perform label-free QC, normalization, and reporting.",
        "category": "Proteomics",
    },
    "Glyco-Proteome Integrator": {
        "module": "app_launcher_GPI",
        "desc": "Integrate glycoprotein and total proteome results.",
        "category": "Integration",
    },
    "Glycopeptide Mass Calculator": {
        "module": "app_launcher_GMC",
        "desc": "Compute peptide/glycopeptide masses and fragmentation ions.",
        "category": "Utilities",
    },
    "Fragment Ion Extractor": {
        "module": "app_launcher_FIIE",
        "desc": "Extract specific fragment ions and analyze mzML/mgf data.",
        "category": "Utilities",
    },
}


# =============================================================================
# PAGE SETUP
# =============================================================================
st.set_page_config(
    page_title="Proteomics Workbench | Mayo Clinic",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_dashboard_css():
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}

        .block-container {
            padding-top: 2rem;
            max-width: 1400px;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 12px !important;
            border: 1px solid #e0e6ed !important;
            background-color: #ffffff;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02),
                        0 4px 12px rgba(0, 0, 0, 0.04) !important;
            transition: all 0.25s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
            padding: 10px !important;
        }

        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05),
                        0 12px 24px rgba(0, 90, 156, 0.10) !important;
            transform: translateY(-4px);
            border-color: #005a9c !important;
        }

        h1 {
            font-weight: 700 !important;
            letter-spacing: -0.5px !important;
        }

        h3 {
            color: #202124 !important;
            font-weight: 600 !important;
            margin-bottom: 0.5rem !important;
        }

        h4 {
            color: #005a9c !important;
            font-weight: 600 !important;
            margin-bottom: 0.2rem !important;
        }

        .card-text {
            color: #5f6368 !important;
            font-size: 0.95rem;
            line-height: 1.4;
            margin-bottom: 15px;
        }

        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
            border-right: 1px solid #e0e6ed;
        }

        .active-app-title {
            color: #005a9c;
            font-weight: 700;
            margin-bottom: 0px;
        }

        .loading-card {
            border: 1px solid #d9e2ef;
            border-radius: 14px;
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            padding: 34px 38px;
            margin-top: 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 16px rgba(0, 75, 135, 0.08);
        }

        .loading-title {
            color: #004b87;
            font-size: 1.35rem;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .loading-subtitle {
            color: #475569;
            font-size: 1rem;
            line-height: 1.55;
            margin-bottom: 18px;
        }

        .loading-note {
            color: #64748b;
            font-size: 0.9rem;
            margin-top: 12px;
        }

        .loader {
            border: 5px solid #e5edf5;
            border-top: 5px solid #004b87;
            border-radius: 50%;
            width: 42px;
            height: 42px;
            animation: spin 0.9s linear infinite;
            margin-bottom: 18px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_dashboard_css()


def render_loading_panel(app_name: str, message: str = "Preparing application"):
    """
    Show a professional loading panel while the selected app module is being imported.
    """
    st.markdown(
        f"""
        <div class="loading-card">
            <div class="loader"></div>
            <div class="loading-title">{message}</div>
            <div class="loading-subtitle">
                Loading <b>{app_name}</b>. The first launch may take a few moments
                while Python modules and visualization libraries are initialized.
            </div>
            <div class="loading-note">
                Please keep this window open. The application will continue automatically.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# SESSION STATE
# =============================================================================
if "current_app" not in st.session_state:
    st.session_state.current_app = DASHBOARD_OPTION

if "pending_app" not in st.session_state:
    st.session_state.pending_app = None

if "last_loaded_module" not in st.session_state:
    st.session_state.last_loaded_module = None

if "last_import_seconds" not in st.session_state:
    st.session_state.last_import_seconds = None

if "last_render_seconds" not in st.session_state:
    st.session_state.last_render_seconds = None


def go_to_app(app_name: str):
    """
    Route from dashboard cards or Back button.

    Important:
    Do NOT modify st.session_state.sidebar_nav here.
    That key belongs to the sidebar selectbox widget, and Streamlit does not
    allow changing widget-backed session state after the widget is created.
    """
    st.session_state.pending_app = app_name
    st.session_state.current_app = app_name
    st.rerun()


def update_app_from_sidebar():
    """
    Sidebar selectbox callback.
    This is the only place sidebar_nav should control current_app.
    """
    selected = st.session_state.sidebar_nav
    st.session_state.pending_app = selected
    st.session_state.current_app = selected


valid_apps = [DASHBOARD_OPTION] + list(APP_MODULES.keys())

if st.session_state.current_app not in valid_apps:
    st.session_state.current_app = DASHBOARD_OPTION

if st.session_state.pending_app not in valid_apps:
    st.session_state.pending_app = None


# =============================================================================
# SIDEBAR
# =============================================================================
for logo_name in ["pandeylab_logo.png", "pandyelab_logo.png", "logo.png"]:
    logo_path = resource_path("assets", logo_name)
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, width=160)
        break

st.sidebar.title("Workbench Menu")

nav_options = [DASHBOARD_OPTION] + list(APP_MODULES.keys())

try:
    current_index = nav_options.index(st.session_state.current_app)
except ValueError:
    current_index = 0
    st.session_state.current_app = DASHBOARD_OPTION

st.sidebar.selectbox(
    "Navigation",
    options=nav_options,
    index=current_index,
    key="sidebar_nav",
    on_change=update_app_from_sidebar,
)

st.sidebar.markdown("---")

if st.sidebar.button("Reset App & Clear Cache", use_container_width=True):
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
    except Exception:
        pass

    for key in list(st.session_state.keys()):
        del st.session_state[key]

    st.rerun()


# =============================================================================
# DASHBOARD
# =============================================================================
def render_category_cards(column, category_name):
    with column:
        st.markdown(f"### {category_name}")

        for app_name, info in APP_MODULES.items():
            if info["category"] != category_name:
                continue

            with st.container(border=True):
                st.markdown(f"#### {app_name}")
                st.markdown(
                    f"<p class='card-text'>{info['desc']}</p>",
                    unsafe_allow_html=True,
                )

                if st.button(
                    "Open Application",
                    key=f"btn_{app_name}",
                    use_container_width=True,
                ):
                    go_to_app(app_name)


def render_dashboard():
    st.markdown(
        """
        <h1 style='text-align: center; color: #005a9c; margin-bottom: 0px;'>
            Proteomics data explorer workbench
        </h1>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p style='text-align: center; font-size: 1.15em; color: #5f6368;
                  margin-top: 8px; font-weight: 400;'>
            An integrated analytical suite for glycoprotein, proteome,
            and multi-omics discovery.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")

    render_category_cards(col1, "GlycoProteomics")
    render_category_cards(col2, "Proteomics")

    with col3:
        st.markdown("### Integration")

        for app_name, info in APP_MODULES.items():
            if info["category"] == "Integration":
                with st.container(border=True):
                    st.markdown(f"#### {app_name}")
                    st.markdown(
                        f"<p class='card-text'>{info['desc']}</p>",
                        unsafe_allow_html=True,
                    )

                    if st.button(
                        "Open Application",
                        key=f"btn_{app_name}",
                        use_container_width=True,
                    ):
                        go_to_app(app_name)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### Utilities")

        for app_name, info in APP_MODULES.items():
            if info["category"] == "Utilities":
                with st.container(border=True):
                    st.markdown(f"#### {app_name}")
                    st.markdown(
                        f"<p class='card-text'>{info['desc']}</p>",
                        unsafe_allow_html=True,
                    )

                    if st.button(
                        "Open Application",
                        key=f"btn_{app_name}",
                        use_container_width=True,
                    ):
                        go_to_app(app_name)

    st.markdown(
        "<br><hr style='margin-top: 40px; border-color: #e0e6ed;'>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p style='text-align: center; font-size: 0.85em; color: #9aa0a6;'>
            Developed and maintained by Pandey Lab • Version 1.0
        </p>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# ACTIVE APP LOADER
# =============================================================================
def render_active_app():
    app_choice = st.session_state.current_app

    if app_choice not in APP_MODULES:
        st.error(f"Unknown app selection: {app_choice}")
        if st.button("Back to Dashboard", use_container_width=True):
            go_to_app(DASHBOARD_OPTION)
        return

    app_meta = APP_MODULES[app_choice]
    module_name = app_meta["module"]

    header_col, back_col = st.columns([0.78, 0.22])

    with header_col:
        st.markdown(
            f"<h3 class='active-app-title'>{app_choice}</h3>",
            unsafe_allow_html=True,
        )
        st.caption(app_meta["desc"])

    with back_col:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Back to Dashboard", use_container_width=True):
            go_to_app(DASHBOARD_OPTION)

    st.markdown(
        "<hr style='margin-top: 0; margin-bottom: 25px; border-color: #e0e6ed;'>",
        unsafe_allow_html=True,
    )

    loading_placeholder = st.empty()

    if st.session_state.get("pending_app") == app_choice:
        with loading_placeholder.container():
            render_loading_panel(app_choice)

    # ------------------------------------------------------------
    # Step 1: Load selected launcher module from file
    # ------------------------------------------------------------
    try:
        import_start = time.perf_counter()
        app_module = load_module_safely(module_name)
        import_seconds = time.perf_counter() - import_start

        st.session_state.last_loaded_module = module_name
        st.session_state.last_import_seconds = import_seconds

        loading_placeholder.empty()

    except FileNotFoundError as e:
        loading_placeholder.empty()

        expected_module_path = get_resource_path(f"{module_name}.py")
        st.error(f"Could not find launcher module file: `{expected_module_path}`")
        st.info(
            "Make sure the launcher .py file is bundled in your PyInstaller/streamlit-desktop-app build. "
            "Add it with `--add-data` or ensure it is present in the runtime folder."
        )
        st.code(f'--add-data "{module_name}.py;."', language="bash")

        with st.expander("Technical details", expanded=False):
            st.code(str(e), language="text")
            st.code(traceback.format_exc(), language="text")

        return

    except ModuleNotFoundError as e:
        loading_placeholder.empty()

        st.error("A required Python package or internal helper module is missing.")
        st.warning(
            f"The launcher file `{module_name}.py` exists, but one of its imports failed. "
            "For packaged builds, check hidden imports and bundled folders such as `ge`, `tp`, `faq`, and `assets`."
        )
        st.code(str(e), language="text")

        with st.expander("Technical details", expanded=False):
            st.code(traceback.format_exc(), language="text")

        return

    except Exception as e:
        loading_placeholder.empty()

        st.exception(e)
        st.warning(f"Unexpected error while loading `{module_name}.py`.")

        with st.expander("Technical traceback", expanded=False):
            st.code(traceback.format_exc(), language="text")

        return

    # ------------------------------------------------------------
    # Step 2: Check main()
    # ------------------------------------------------------------
    if not hasattr(app_module, "main") or not callable(app_module.main):
        st.session_state.pending_app = None

        st.error(f"Module `{module_name}` does not define a callable `main()` function.")
        st.info(
            "Each launcher file should contain:\n\n"
            "def main():\n"
            "    ...your Streamlit code..."
        )
        return

    # ------------------------------------------------------------
    # Step 3: Run selected app
    # ------------------------------------------------------------
    try:
        render_start = time.perf_counter()

        if st.session_state.get("pending_app") == app_choice:
            with st.spinner(f"Opening {app_choice}..."):
                app_module.main()
        else:
            app_module.main()

        render_seconds = time.perf_counter() - render_start

        st.session_state.last_render_seconds = render_seconds
        st.session_state.pending_app = None

    except ModuleNotFoundError as e:
        st.session_state.pending_app = None

        st.error("A required module inside the selected app is missing.")

        st.warning(
            f"The launcher `{module_name}` loaded correctly, but something inside "
            f"`{module_name}.main()` tried to import another missing module."
        )

        st.code(str(e), language="text")

        st.info(
            "This usually means one of your helper packages/folders was not bundled, "
            "for example `ge`, `tp`, `faq`, or another dependency."
        )

        with st.expander("Technical traceback", expanded=False):
            st.code(traceback.format_exc(), language="text")

    except Exception as e:
        st.session_state.pending_app = None

        st.exception(e)
        st.warning(f"Unexpected error while running `{module_name}.main()`.")

        with st.expander("Technical traceback", expanded=False):
            st.code(traceback.format_exc(), language="text")


# =============================================================================
# MAIN ROUTING
# =============================================================================
if st.session_state.current_app == DASHBOARD_OPTION:
    render_dashboard()
else:
    render_active_app()