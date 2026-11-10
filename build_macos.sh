#!/usr/bin/env bash
# Build the macOS app (ProteomicsWorkbench.app).
#
# Mac version of build_windows.ps1. Differences:
#   - --add-data uses ":" (not ";") between source and destination.
#   - --windowed is required, otherwise PyInstaller makes a bare executable
#     that opens in Terminal instead of a double-clickable .app.
#   - Only the R *package library* is bundled (r_bundle/library -> R_library).
#     The R runtime itself is NOT bundled: users install R 4.5.x from CRAN.
#     The app already looks for /Library/Frameworks/R.framework/.../Rscript.

set -euo pipefail

APP_NAME="ProteomicsWorkbench"
BUNDLE_ID="org.pandeylab.proteomicsworkbench"
ROOT="$(pwd)"

# Build-time defaults for the packaged app. app.py also sets these at runtime
# when frozen mode is detected.
export GLYCOEXPLORER_HOSTED="false"
export GLYCOEXPLORER_SAVE_MODE="dialog"

echo "Repository root: $ROOT"
echo "Cleaning old builds..."
rm -rf build dist "$APP_NAME.spec"

# Folders PyInstaller expects.
for d in .streamlit assets faq ge tp; do
  mkdir -p "$ROOT/$d"
done

# PyInstaller can fail on an empty data directory.
[ -f "$ROOT/assets/.keep" ] || echo "placeholder so PyInstaller can bundle assets directory" > "$ROOT/assets/.keep"

[ -f "$ROOT/desktop_launcher.py" ] || { echo "ERROR: desktop_launcher.py is missing."; exit 1; }
[ -f "$ROOT/gene_sets/manifest.json" ] || { echo "ERROR: gene_sets/manifest.json is missing. Run the 'Update gene sets' workflow on GitHub (or python build_gene_sets.py) first."; exit 1; }
[ -f "$ROOT/enrichment/ui.py" ] || { echo "ERROR: enrichment/ui.py is missing."; exit 1; }
[ -f "$ROOT/download_utility.py" ] || { echo "ERROR: download_utility.py is missing."; exit 1; }
[ -d "$ROOT/r_bundle/library" ]    || { echo "ERROR: r_bundle/library is missing. Run the 'Stage R package library' step first."; exit 1; }

# Optional Mac icon: put a .icns file at assets/app_icon.icns to use it.
ICON_ARGS=()
if [ -f "$ROOT/assets/app_icon.icns" ]; then
  ICON_ARGS=(--icon "$ROOT/assets/app_icon.icns")
fi

echo "Building macOS app..."

# Fewer files = much faster unzip. "--collect-all" also copied every library's
# .py source next to the compiled archive (~6,800 duplicate files), so each
# library now uses --collect-submodules/--collect-data/--collect-binaries
# (same code and data, no duplicate sources). IPython/jedi are developer tools
# the app never uses (~1,800 files).
# Build with PyInstaller directly, around our own desktop_launcher.py.
# (The wrapper made by "streamlit-desktop-app build" gave the server only
# 10 seconds to start and then quit. Streamlit settings now live in
# desktop_launcher.py.)
python -m PyInstaller \
  --name "$APP_NAME" \
  --noconfirm \
    --exclude-module IPython --exclude-module jedi --exclude-module parso \
  ${ICON_ARGS[@]+"${ICON_ARGS[@]}"} \
    --onedir \
    --windowed \
    --osx-bundle-identifier "$BUNDLE_ID" \
    --paths "$ROOT" \
    --paths "$ROOT/ge" \
    --paths "$ROOT/tp" \
    --paths "$ROOT/faq" \
    --add-data "$ROOT/app.py:." \
    --copy-metadata streamlit \
    --hidden-import streamlit.web.cli \
    --add-data "$ROOT/app_launcher_*.py:." \
    --add-data "$ROOT/download_utility.py:." \
    --add-data "$ROOT/workbench_log.py:." \
    --runtime-hook "$ROOT/workbench_log.py" \
    --hidden-import workbench_log \
    --add-data "$ROOT/naguider_impute.R:." \
    --hidden-import download_utility \
    --hidden-import app_launcher_GE \
    --hidden-import app_launcher_GE_LF \
    --hidden-import app_launcher_TP \
    --hidden-import app_launcher_TP_LF \
    --hidden-import app_launcher_GMC \
    --hidden-import app_launcher_GPI \
    --hidden-import faq.faq_GE \
    --hidden-import faq.faq_GE_LF \
    --hidden-import faq.faq_TP \
    --hidden-import faq.faq_TP_LF \
    --hidden-import faq.faq_GMC \
    --hidden-import faq.faq_GPI \
    --hidden-import ge.batch_correcttion \
    --hidden-import ge.boxplot \
    --hidden-import ge.data_prep_utility \
    --hidden-import ge.diff_results_utility \
    --hidden-import ge.heatmap \
    --hidden-import ge.lollipop \
    --hidden-import ge.misc_utility \
    --hidden-import ge.qc_utils \
    --hidden-import ge.report_utility \
    --hidden-import ge.ui_utility \
    --hidden-import ge.volcanoplot \
    --hidden-import ge.waterfall \
    --hidden-import ge.waterfall_lf \
    --hidden-import tp.qc_utility \
    --hidden-import tp.report_utility \
    --hidden-import tp.tp_common_utility \
    --hidden-import tp.ui_utility \
    --hidden-import matplotlib.backends.backend_svg \
    --hidden-import matplotlib.backends.backend_pdf \
    --hidden-import matplotlib.backends.backend_agg \
    --add-data "$ROOT/.streamlit:.streamlit" \
    --add-data "$ROOT/assets:assets" \
    --add-data "$ROOT/faq:faq" \
    --add-data "$ROOT/ge:ge" \
    --add-data "$ROOT/tp:tp" \
    --add-data "$ROOT/gene_sets:gene_sets" \
    --hidden-import enrichment.core \
    --hidden-import enrichment.ui \
    --hidden-import figure_utils \
    --collect-submodules gseapy --collect-data gseapy --collect-binaries gseapy --copy-metadata gseapy \
    --collect-submodules streamlit --collect-data streamlit --collect-binaries streamlit --copy-metadata streamlit \
    --collect-submodules plotly --collect-data plotly --collect-binaries plotly --copy-metadata plotly \
    --collect-submodules openpyxl --collect-data openpyxl --collect-binaries openpyxl --copy-metadata openpyxl \
    --collect-submodules matplotlib --collect-data matplotlib --collect-binaries matplotlib --copy-metadata matplotlib \
    --collect-submodules scipy --collect-data scipy --collect-binaries scipy --copy-metadata scipy \
    --collect-submodules pandas --collect-data pandas --collect-binaries pandas --copy-metadata pandas \
    --collect-submodules numpy --collect-data numpy --collect-binaries numpy --copy-metadata numpy \
    --collect-submodules sklearn --collect-data sklearn --collect-binaries sklearn --copy-metadata scikit-learn \
    --collect-submodules statsmodels --collect-data statsmodels --collect-binaries statsmodels --copy-metadata statsmodels \
    --collect-submodules PIL --collect-data PIL --collect-binaries PIL --copy-metadata pillow \
    --collect-submodules streamlit \
    --collect-submodules plotly \
    "$ROOT/desktop_launcher.py"

APP="$ROOT/dist/$APP_NAME.app"
[ -d "$APP" ] || { echo "ERROR: $APP was not created."; exit 1; }

# ---------------------------------------------------------------------------
# Add the R package library AFTER PyInstaller.
#
# If R_library goes through --add-data, PyInstaller treats the packages' .so
# files as binaries, copies libR.dylib into the app and rewrites the .so files
# to use that copy. R then ends up with two libR copies loaded -> crashes.
# Copying afterwards keeps the .so files pointing at the user's installed
# /Library/Frameworks/R.framework (which is what we want).
#
# The app code looks for <sys._MEIPASS>/R_library. In a PyInstaller .app,
# sys._MEIPASS is Contents/Frameworks, so link it from there (same layout
# PyInstaller itself uses for data folders).
# ---------------------------------------------------------------------------
echo "Adding R package library to the app..."
rm -rf "$APP/Contents/Resources/R_library" "$APP/Contents/Frameworks/R_library"
cp -R "$ROOT/r_bundle/library" "$APP/Contents/Resources/R_library"
ln -s "../Resources/R_library" "$APP/Contents/Frameworks/R_library"

echo "Build complete."
echo "Output: dist/$APP_NAME.app"