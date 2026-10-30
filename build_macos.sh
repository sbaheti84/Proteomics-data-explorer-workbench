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

[ -f "$ROOT/download_utility.py" ] || { echo "ERROR: download_utility.py is missing."; exit 1; }
[ -d "$ROOT/r_bundle/library" ]    || { echo "ERROR: r_bundle/library is missing. Run the 'Stage R package library' step first."; exit 1; }

# Optional Mac icon: put a .icns file at assets/app_icon.icns to use it.
ICON_ARGS=()
if [ -f "$ROOT/assets/app_icon.icns" ]; then
  ICON_ARGS=(--icon "$ROOT/assets/app_icon.icns")
fi

echo "Building macOS app..."

streamlit-desktop-app build app.py \
  --name "$APP_NAME" \
  ${ICON_ARGS[@]+"${ICON_ARGS[@]}"} \
  --streamlit-options \
    --server.maxUploadSize=1024 \
    --server.enableStaticServing=true \
    --theme.base=light \
    "--theme.primaryColor=#004b87" \
    "--theme.backgroundColor=#ffffff" \
    "--theme.secondaryBackgroundColor=#f0f4f8" \
    "--theme.textColor=#1e293b" \
    "--theme.font=sans serif" \
  --pyinstaller-options \
    --onedir \
    --windowed \
    --osx-bundle-identifier "$BUNDLE_ID" \
    --paths "$ROOT" \
    --paths "$ROOT/ge" \
    --paths "$ROOT/tp" \
    --paths "$ROOT/faq" \
    --add-data "$ROOT/app_launcher_*.py:." \
    --add-data "$ROOT/download_utility.py:." \
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
    --collect-all streamlit \
    --collect-all plotly \
    --collect-all openpyxl \
    --collect-all matplotlib \
    --collect-all scipy \
    --collect-all pandas \
    --collect-all numpy \
    --collect-all sklearn \
    --collect-all statsmodels \
    --collect-all PIL \
    --collect-submodules streamlit \
    --collect-submodules plotly \
    --noconfirm

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
