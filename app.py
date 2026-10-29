#!/usr/bin/env python3
"""
Proteomics data explorer workbench: entry point.

This file only does three things:
  1. Navigation between the apps (each app lives in app_launcher_*.py).
  2. The home page, which helps researchers pick the right app for their data.
  3. Keeping apps isolated from each other and showing clear errors.

All analysis code stays in the launcher files and the ge/ tp/ faq/ packages.
"""

import html
import importlib.util
import os
import platform
import sys
import traceback
import urllib.parse
from datetime import datetime

# Desktop/PyInstaller build: browser download buttons are unreliable in the embedded
# Streamlit desktop/webview runtime. Force local Save As mode for the packaged exe.
if getattr(sys, "frozen", False) or hasattr(sys, "_MEIPASS"):
    os.environ.setdefault("GLYCOEXPLORER_HOSTED", "false")
    os.environ.setdefault("GLYCOEXPLORER_SAVE_MODE", "dialog")

import streamlit as st


# =============================================================================
# SETTINGS (edit these)
# =============================================================================
APP_VERSION = "1.0"
LAB_NAME = "Pandey Lab"

# Where the "Report a problem" button points. Use a mailto: link instead if
# your users do not have GitHub accounts, e.g. "mailto:someone@example.org".
SUPPORT_URL = "https://github.com/sbaheti84/Proteomics-data-explorer-workbench/issues/new"

# Session-state keys owned by this file start with this prefix.
# Everything else belongs to the app that is currently open.
WB = "_wb_"


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
    """Compatibility wrapper used by this dashboard for assets."""
    return get_resource_path(os.path.join(*parts))


def load_module_safely(module_name: str):
    """
    Load <module_name>.py dynamically from the runtime root directory.

    The launcher files run their top-level code (page config, CSS, session
    defaults) on every Streamlit rerun, so the module is executed fresh each
    time, exactly as before.
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
            f"The file exists, but one dependency/module is missing.",
            name=e.name,
        ) from e

    return module


# =============================================================================
# APP REGISTRY
#
# To add, hide or rename an app, edit this list only.
#   key      -> stable id, also the page address (e.g. .../proteome-lfq)
#   nav      -> short label for the sidebar menu (the sidebar is narrow)
#   omics    -> "glyco" | "proteome" | "integration"  (home page row)
#   quant    -> "TMT" | "Label-free" | ""              (badge on the card)
#   inputs   -> what the researcher must upload, in plain words
# =============================================================================
APPS = [
    {
        "key": "glyco-tmt",
        "nav": "Glyco · TMT",
        "title": "GlycoExplorer (TMT)",
        "module": "app_launcher_GE",
        "omics": "glyco",
        "quant": "TMT",
        "icon": ":material/bubble_chart:",
        "desc": "QC, normalization, differential analysis and plots for TMT glycoproteomics.",
        "inputs": "pGlyco + Proteome Discoverer exports, or MSFragger psm.tsv + peptide.tsv",
    },
    {
        "key": "glyco-lfq",
        "nav": "Glyco · Label-free",
        "title": "GlycoExplorer (Label-Free)",
        "module": "app_launcher_GE_LF",
        "omics": "glyco",
        "quant": "Label-free",
        "icon": ":material/bubble_chart:",
        "desc": "The same glyco workflow for label-free data, built from per-sample runs.",
        "inputs": "pGlyco LFQ matrix, or one MSFragger .tsv per sample run",
    },
    {
        "key": "proteome-tmt",
        "nav": "Proteome · TMT",
        "title": "Total Proteome (TMT)",
        "module": "app_launcher_TP",
        "omics": "proteome",
        "quant": "TMT",
        "icon": ":material/biotech:",
        "desc": "Protein-level QC, normalization, batch correction, statistics and report.",
        "inputs": "Protein abundance table (CSV or Excel)",
    },
    {
        "key": "proteome-lfq",
        "nav": "Proteome · Label-free",
        "title": "Total Proteome (Label-Free)",
        # NOTE: this points to the *_old* launcher, as in the previous app.py.
        "module": "app_launcher_TP_LF_old",
        "omics": "proteome",
        "quant": "Label-free",
        "icon": ":material/biotech:",
        "desc": "Label-free protein workflow with missing-value filtering and NAguideR imputation.",
        "inputs": "LFQ protein intensity table (CSV, TSV or Excel)",
    },
    {
        "key": "integrator",
        "nav": "Integrator",
        "title": "Glyco-Proteome Integrator",
        "module": "app_launcher_GPI",
        "omics": "integration",
        "quant": "",
        "icon": ":material/merge:",
        "desc": "Compare glycosite changes with protein changes to separate glycosylation "
                "effects from protein abundance effects.",
        "inputs": "Glyco results + protein results, each with gene ID, log2FC and p- or q-value",
    },
]
APP_BY_KEY = {a["key"]: a for a in APPS}

ROWS = [
    ("glyco", "Glycoproteomics", "Glycopeptide and glycosite-level data"),
    ("proteome", "Total proteome", "Protein-level abundance data"),
    ("integration", "Integration", "You already have glyco and protein results"),
]


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
        footer {visibility: hidden;}

        /* Clean desktop look: hide the top bar ... */
        header[data-testid="stHeader"] {visibility: hidden;}
        /* ... but never hide the button that re-opens a collapsed sidebar
           (all upload controls live there) or the "app is busy" indicator.
           Some launchers also hide the header; this rule wins over theirs. */
        [data-testid="stExpandSidebarButton"],
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="stStatusWidget"] {
            visibility: visible !important;
        }

        .block-container {
            padding-top: 2rem;
            max-width: 1400px;
        }

        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
            border-right: 1px solid #e0e6ed;
        }

        /* Compact app list in the sidebar */
        [data-testid="stSidebar"] [data-testid="stPageLink"] a {
            padding-top: 0.2rem;
            padding-bottom: 0.2rem;
        }

        /* Home page cards only (scoped by container key, so the hover
           effect never leaks into the panels inside the apps). */
        [class*="st-key-wb_card_"] {
            background-color: #ffffff;
            border-radius: 12px !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02),
                        0 4px 12px rgba(0, 0, 0, 0.04);
            transition: box-shadow 0.2s ease, border-color 0.2s ease,
                        transform 0.2s ease;
        }
        [class*="st-key-wb_card_"]:hover {
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05),
                        0 12px 24px rgba(0, 90, 156, 0.10);
            border-color: #005a9c !important;
            transform: translateY(-2px);
        }

        .wb-card-title {
            color: #005a9c;
            font-weight: 650;
            font-size: 1.15rem;
            margin: 0 0 0.15rem 0;
        }
        .wb-card-text {
            color: #5f6368;
            font-size: 0.95rem;
            line-height: 1.45;
            margin: 0;
        }
        .wb-card-input {
            color: #475569;
            font-size: 0.85rem;
            line-height: 1.4;
            margin: 0.1rem 0 0.2rem 0;
        }
        .wb-row-title {
            color: #202124;
            font-weight: 650;
            font-size: 1.25rem;
            margin: 0;
        }
        .wb-row-sub {
            color: #6b7280;
            font-size: 0.9rem;
            margin: 0 0 0.3rem 0;
        }

        .wb-app-title {
            color: #005a9c;
            font-weight: 700;
            margin-bottom: 0;
        }
        .wb-rule {
            margin-top: 0;
            margin-bottom: 25px;
            border-color: #e0e6ed;
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


def render_loading_panel(app_title: str):
    """Shown while an app's code is loaded for the first time in this session."""
    st.markdown(
        f"""
        <div class="loading-card">
            <div class="loader"></div>
            <div class="loading-title">Preparing application</div>
            <div class="loading-subtitle">
                Loading <b>{html.escape(app_title)}</b>. The first launch may take a few
                moments while Python modules and visualization libraries are initialized.
            </div>
            <div class="loading-note">
                Please keep this window open. The application will continue automatically.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# ERROR REPORTING
# =============================================================================
def build_error_report(app: dict, stage: str, exc: BaseException) -> str:
    """Plain-text report a researcher can copy and send to the maintainer."""
    lines = [
        f"Proteomics Workbench {APP_VERSION}",
        f"App:        {app['title']} ({app['module']}.py)",
        f"Stage:      {stage}",
        f"Time:       {datetime.now().isoformat(timespec='seconds')}",
        f"Packaged:   {'yes (desktop build)' if getattr(sys, 'frozen', False) else 'no'}",
        f"Python:     {platform.python_version()}",
        f"Streamlit:  {st.__version__}",
        f"System:     {platform.platform()}",
        "",
        f"Error:      {type(exc).__name__}: {exc}",
    ]
    if isinstance(exc, FileNotFoundError):
        lines += [
            "",
            "Developer hint: the launcher file is not in the build. Add it with",
            f'  --add-data "{app["module"]}.py;."',
        ]
    elif isinstance(exc, ModuleNotFoundError):
        lines += [
            "",
            "Developer hint: check PyInstaller hidden imports and that the ge/, tp/,",
            "faq/ and assets/ folders are bundled.",
        ]
    lines += ["", "Traceback:", "".join(traceback.format_exception(exc)).rstrip()]
    return "\n".join(lines)


def render_failure(app: dict, stage: str, exc: BaseException):
    """Researcher-friendly error: what happened, what to do, and a copyable report."""
    if isinstance(exc, FileNotFoundError):
        what = "Part of this app is missing from the installation."
        todo = "Download the latest version of the workbench and try again."
    elif isinstance(exc, ModuleNotFoundError):
        missing = getattr(exc, "name", None) or getattr(exc.__cause__, "name", None) or "unknown"
        what = f"A required component (`{missing}`) is missing from the installation."
        todo = "Download the latest version of the workbench and try again."
    else:
        what = "Something went wrong inside the app."
        todo = (
            "Try again. If it keeps happening, use **Help → Restart workbench** in the "
            "sidebar, then report the problem with the error report below."
        )

    st.error(f"**{app['title']} could not {stage}.** {what}", icon=":material/error:")
    st.markdown(todo)

    issue_title = urllib.parse.quote(f"[{app['title']}] {type(exc).__name__} while trying to {stage}")
    issue_body = urllib.parse.quote("Please paste the error report from the app here.\n\nWhat were you doing?\n")
    if SUPPORT_URL.startswith("mailto:"):
        report_url = f"{SUPPORT_URL}?subject={issue_title}"
    else:
        report_url = f"{SUPPORT_URL}?title={issue_title}&body={issue_body}"
    st.link_button("Report this problem", report_url, icon=":material/bug_report:")

    with st.expander("Error report (copy this and include it in your message)"):
        st.code(build_error_report(app, stage, exc), language="text")


# =============================================================================
# APP ISOLATION
#
# All apps share one st.session_state, and several use the same key names
# (e.g. TP and TP-LFQ share expr_processed, diff_results, final_pdf_bytes).
# The apps only reset when the uploaded file *changes*, so uploading the same
# file in two apps could show one app's results inside the other.
# When the researcher switches app, we clear the previous app's state.
# (Switching already lost uploads before, because Streamlit forgets file
# uploads of widgets that are no longer on screen.)
# =============================================================================
def isolate_app_state(page_key: str):
    previous = st.session_state.get(WB + "active_page")
    if previous is not None and previous != page_key:
        for key in list(st.session_state.keys()):
            if not str(key).startswith(WB):
                del st.session_state[key]
    st.session_state[WB + "active_page"] = page_key


def restart_workbench():
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
    except Exception:
        pass
    for key in list(st.session_state.keys()):
        del st.session_state[key]


# =============================================================================
# PAGES
# =============================================================================
def render_app_card(app: dict):
    with st.container(border=True, key=f"wb_card_{app['key'].replace('-', '_')}"):
        title_col, badge_col = st.columns([0.72, 0.28], vertical_alignment="center")
        with title_col:
            st.markdown(
                f"<p class='wb-card-title'>{html.escape(app['title'])}</p>",
                unsafe_allow_html=True,
            )
        if app["quant"]:
            with badge_col:
                color = "blue" if app["quant"] == "TMT" else "green"
                st.badge(app["quant"], color=color)

        st.markdown(
            f"<p class='wb-card-text'>{html.escape(app['desc'])}</p>"
            f"<p class='wb-card-input'><b>You upload:</b> {html.escape(app['inputs'])}</p>",
            unsafe_allow_html=True,
        )

        if st.button(
            "Open",
            key=f"{WB}open_{app['key']}",
            icon=":material/arrow_forward:",
            help=f"Open {app['title']}",
            width="stretch",
        ):
            st.switch_page(PAGES[app["key"]])


def render_home():
    st.markdown(
        """
        <h1 style='text-align: center; color: #005a9c; margin-bottom: 0px;'>
            Proteomics data explorer workbench
        </h1>
        <p style='text-align: center; font-size: 1.15em; color: #5f6368;
                  margin-top: 8px; margin-bottom: 1.6rem; font-weight: 400;'>
            Pick the app that matches your data: what you measured, and how it was quantified.
        </p>
        """,
        unsafe_allow_html=True,
    )

    for omics, row_title, row_sub in ROWS:
        row_apps = [a for a in APPS if a["omics"] == omics]
        if not row_apps:
            continue

        label_col, cards_col = st.columns([0.2, 0.8], gap="large")
        with label_col:
            st.markdown(
                f"<p class='wb-row-title'>{row_title}</p>"
                f"<p class='wb-row-sub'>{row_sub}</p>",
                unsafe_allow_html=True,
            )
        with cards_col:
            card_cols = st.columns(2, gap="medium")
            for i, app in enumerate(row_apps):
                with card_cols[i % 2]:
                    render_app_card(app)
        st.write("")

    st.divider()
    tip1, tip2, tip3 = st.columns(3, gap="large")
    tip1.markdown(
        ":material/description: **Not sure about the file format?**  \n"
        "Open the app and look in its sidebar for example files."
    )
    tip2.markdown(
        ":material/swap_horiz: **Switching apps starts fresh.**  \n"
        "Download your tables, plots and report before you switch."
    )
    tip3.markdown(
        ":material/help: **Something not working?**  \n"
        "Use **Help** in the sidebar to report a problem."
    )

    st.markdown(
        f"""
        <p style='text-align: center; font-size: 0.85em; color: #9aa0a6; margin-top: 2.5rem;'>
            Developed and maintained by {html.escape(LAB_NAME)} • Version {html.escape(APP_VERSION)}
        </p>
        """,
        unsafe_allow_html=True,
    )


def render_app(app: dict):
    head_col, back_col = st.columns([0.78, 0.22], vertical_alignment="center")
    with head_col:
        st.markdown(
            f"<h3 class='wb-app-title'>{html.escape(app['title'])}</h3>",
            unsafe_allow_html=True,
        )
        st.caption(app["desc"])
    with back_col:
        if st.button("All apps", key=f"{WB}back", icon=":material/apps:", width="stretch"):
            st.switch_page(HOME_PAGE)

    st.markdown("<hr class='wb-rule'>", unsafe_allow_html=True)

    first_open = st.session_state.get(WB + "loaded") != app["key"]
    loading_slot = st.empty()
    if first_open:
        with loading_slot.container():
            render_loading_panel(app["title"])

    # Step 1: load the launcher file
    try:
        module = load_module_safely(app["module"])
    except Exception as exc:
        loading_slot.empty()
        render_failure(app, "start", exc)
        return
    loading_slot.empty()

    main = getattr(module, "main", None)
    if not callable(main):
        render_failure(
            app,
            "start",
            AttributeError(f"{app['module']}.py does not define a callable main() function."),
        )
        return

    st.session_state[WB + "loaded"] = app["key"]

    # Step 2: run the app. st.stop()/st.rerun()/st.switch_page() are not
    # Exceptions, so they pass through untouched.
    try:
        main()
    except Exception as exc:
        render_failure(app, "run", exc)


def _make_app_page(app: dict):
    def _page():
        render_app(app)

    _page.__name__ = "page_" + app["key"].replace("-", "_")
    return st.Page(_page, title=app["title"], icon=app["icon"], url_path=app["key"])


HOME_PAGE = st.Page(render_home, title="Home", icon=":material/home:", default=True)
PAGES = {a["key"]: _make_app_page(a) for a in APPS}

# position="hidden": we draw the menu ourselves so the lab logo stays on top.
current_page = st.navigation([HOME_PAGE, *PAGES.values()], position="hidden")
isolate_app_state(current_page.url_path or "home")


# =============================================================================
# SIDEBAR (workbench part; the open app adds its own controls below)
# =============================================================================
def render_app_menu():
    """List of all apps, grouped the same way as the home page."""
    st.page_link(HOME_PAGE, label="All apps", icon=":material/home:")
    for omics, row_title, _ in ROWS:
        row_apps = [a for a in APPS if a["omics"] == omics]
        if row_apps:
            st.caption(row_title)
            for app in row_apps:
                st.page_link(PAGES[app["key"]], label=app["nav"], icon=app["icon"])


def render_help():
    st.markdown(f"**Proteomics Workbench {APP_VERSION}**")
    st.caption(
        f"Python {platform.python_version()} • Streamlit {st.__version__} • "
        f"{'Desktop build' if getattr(sys, 'frozen', False) else 'Hosted / source'}"
    )
    st.link_button(
        "Report a problem",
        SUPPORT_URL,
        icon=":material/bug_report:",
        width="stretch",
    )

    st.divider()
    st.markdown("**Restart workbench**")
    st.caption(
        "Clears uploaded files, results and cached calculations in every app. "
        "To clear just one analysis, use that app's own reset button."
    )
    # No WB prefix on purpose: the tick is forgotten when you change app.
    sure = st.checkbox("I have saved what I need", key="wb_restart_ok")
    if st.button(
        "Restart workbench",
        key=WB + "restart",
        type="primary",
        disabled=not sure,
        icon=":material/restart_alt:",
        width="stretch",
    ):
        restart_workbench()
        st.switch_page(HOME_PAGE)


def render_sidebar(on_home: bool):
    """
    Home page: logo + full app list (nothing else is in the sidebar).
    Inside an app: small logo + two menu buttons, so the app's own upload
    controls start near the top of the sidebar instead of below the fold.
    """
    with st.sidebar:
        for logo_name in ["pandeylab_logo.png", "pandyelab_logo.png", "logo.png"]:
            logo_path = resource_path("assets", logo_name)
            if os.path.exists(logo_path):
                st.image(logo_path, width=110 if on_home else 72)
                break

        if on_home:
            render_app_menu()
            with st.popover("Help", icon=":material/help:", width="stretch"):
                render_help()
        else:
            with st.popover("Switch app", icon=":material/apps:", width="stretch"):
                render_app_menu()
            with st.popover("Help", icon=":material/help:", width="stretch"):
                render_help()
            st.divider()


render_sidebar(on_home=(current_page.url_path or "home") == "home")
current_page.run()
