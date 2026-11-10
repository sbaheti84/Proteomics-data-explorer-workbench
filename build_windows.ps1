$ErrorActionPreference = "Stop"

$APP_NAME = "ProteomicsWorkbench"
$ROOT = (Get-Location).Path

# Build-time defaults for the packaged executable. app.py also sets these at runtime
# when PyInstaller/frozen mode is detected.
$env:GLYCOEXPLORER_HOSTED = "false"
$env:GLYCOEXPLORER_SAVE_MODE = "dialog"

Write-Host "Repository root: $ROOT" 
Write-Host "Cleaning old builds..."
Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue
Remove-Item -Force "$APP_NAME.spec" -ErrorAction SilentlyContinue

# Ensure folders expected by PyInstaller exist in GitHub runner
$requiredDirs = @(".streamlit", "assets", "faq", "ge", "tp")
foreach ($d in $requiredDirs) {
    $p = Join-Path $ROOT $d
    if (!(Test-Path -Path $p)) {
        Write-Host "Creating missing directory: $d"
        New-Item -ItemType Directory -Force -Path $p | Out-Null
    }
}

# PyInstaller can fail on an empty data directory. Add harmless placeholders.
$placeholder = Join-Path $ROOT "assets\.keep"
if (!(Test-Path $placeholder)) {
    "placeholder so PyInstaller can bundle assets directory" | Out-File -FilePath $placeholder -Encoding utf8
}

# Optional logo/example folders are bundled only if they exist, but this keeps assets stable.
Write-Host "Checking expected paths..."
if (!(Test-Path (Join-Path $ROOT "slim_bundle.py"))) { throw "slim_bundle.py is missing." }
if (!(Test-Path (Join-Path $ROOT "gene_sets\manifest.json"))) { throw "gene_sets\manifest.json is missing. Run the 'Update gene sets' workflow on GitHub (or python build_gene_sets.py) first." }
if (!(Test-Path (Join-Path $ROOT "enrichment\ui.py"))) { throw "enrichment\ui.py is missing." }
if (!(Test-Path (Join-Path $ROOT "desktop_launcher.py"))) { throw "desktop_launcher.py is missing (entry point for the desktop app)." }
if (!(Test-Path (Join-Path $ROOT "workbench_log.py"))) { throw "workbench_log.py is missing (log file for the desktop app)." }
if (!(Test-Path (Join-Path $ROOT "download_utility.py"))) { throw "download_utility.py is missing. Put the fixed file at repository root before building." }
foreach ($d in $requiredDirs) {
    $p = Join-Path $ROOT $d
    if (!(Test-Path $p)) { throw "Required path missing after setup: $p" }
    Write-Host "Found: $p"
}

Write-Host "Top-level files/folders before build:"
Get-ChildItem -Force $ROOT | Select-Object Name, Mode, Length | Format-Table -AutoSize

Write-Host "Building Windows app..."

# Fewer files = much faster unzip. "--collect-all" also copied every library's
# .py source next to the compiled archive (~6,800 duplicate files), so each
# library now uses --collect-submodules/--collect-data/--collect-binaries
# (same code and data, no duplicate sources). IPython/jedi are developer tools
# the app never uses (~1,800 files).
# Build with PyInstaller directly, around our own desktop_launcher.py.
# (The wrapper made by "streamlit-desktop-app build" gave the server only
# 10 seconds to start, then quit with "[PYI-49000] Failed to execute script".
# Streamlit settings now live in desktop_launcher.py.)
python -m PyInstaller `
  --name $APP_NAME `
  --noconfirm `
    --exclude-module IPython --exclude-module jedi --exclude-module parso `
    --onedir `
    --paths "$ROOT" `
    --paths "$(Join-Path $ROOT 'ge')" `
    --paths "$(Join-Path $ROOT 'tp')" `
    --paths "$(Join-Path $ROOT 'faq')" `
    --add-data "$ROOT\app.py;." `
    --copy-metadata streamlit `
    --hidden-import streamlit.web.cli `
    --add-data "$ROOT\app_launcher_*.py;." `
    --add-data "$ROOT\download_utility.py;." `
    --add-data "$ROOT\workbench_log.py;." `
    --runtime-hook "$ROOT\workbench_log.py" `
    --hidden-import workbench_log `
    --add-data "$ROOT\naguider_impute.R;." `
    --add-data "$ROOT\r_bundle\R;R" `
    --add-data "$ROOT\r_bundle\library;R_library" `
    --hidden-import download_utility `
    --hidden-import tkinter `
    --hidden-import tkinter.filedialog `
    --hidden-import tkinter.messagebox `
    --hidden-import _tkinter `
    --hidden-import app_launcher_GE `
    --hidden-import app_launcher_GE_LF `
    --hidden-import app_launcher_TP `
    --hidden-import app_launcher_TP_LF `
    --hidden-import app_launcher_GMC `
    --hidden-import app_launcher_GPI `
    --hidden-import app_launcher_FIIE `
    --hidden-import faq.faq_GE `
    --hidden-import faq.faq_GE_LF `
    --hidden-import faq.faq_TP `
    --hidden-import faq.faq_TP_LF `
    --hidden-import faq.faq_GMC `
    --hidden-import faq.faq_GPI `
    --hidden-import faq.faq_FIIE `
    --hidden-import ge.batch_correcttion `
    --hidden-import ge.boxplot `
    --hidden-import ge.data_prep_utility `
    --hidden-import ge.diff_results_utility `
    --hidden-import ge.heatmap `
    --hidden-import ge.lollipop `
    --hidden-import ge.misc_utility `
    --hidden-import ge.qc_utils `
    --hidden-import ge.report_utility `
    --hidden-import ge.ui_utility `
    --hidden-import ge.volcanoplot `
    --hidden-import ge.waterfall `
    --hidden-import ge.waterfall_lf `
    --hidden-import tp.qc_utility `
    --hidden-import tp.report_utility `
    --hidden-import tp.tp_common_utility `
    --hidden-import tp.ui_utility `
    --hidden-import matplotlib.backends.backend_svg `
    --hidden-import matplotlib.backends.backend_pdf `
    --hidden-import matplotlib.backends.backend_agg `
    --add-data "$(Join-Path $ROOT '.streamlit');.streamlit" `
    --add-data "$(Join-Path $ROOT 'assets');assets" `
    --add-data "$(Join-Path $ROOT 'faq');faq" `
    --add-data "$(Join-Path $ROOT 'ge');ge" `
    --add-data "$(Join-Path $ROOT 'tp');tp" `
    --add-data "$(Join-Path $ROOT 'gene_sets');gene_sets" `
    --hidden-import enrichment.core `
    --hidden-import enrichment.ui `
    --hidden-import figure_utils `
    --collect-submodules gseapy --collect-data gseapy --collect-binaries gseapy --copy-metadata gseapy `
    --collect-submodules streamlit --collect-data streamlit --collect-binaries streamlit --copy-metadata streamlit `
    --collect-submodules plotly --collect-data plotly --collect-binaries plotly --copy-metadata plotly `
    --collect-submodules openpyxl --collect-data openpyxl --collect-binaries openpyxl --copy-metadata openpyxl `
    --collect-submodules matplotlib --collect-data matplotlib --collect-binaries matplotlib --copy-metadata matplotlib `
    --collect-submodules scipy --collect-data scipy --collect-binaries scipy --copy-metadata scipy `
    --collect-submodules pandas --collect-data pandas --collect-binaries pandas --copy-metadata pandas `
    --collect-submodules numpy --collect-data numpy --collect-binaries numpy --copy-metadata numpy `
    --collect-submodules sklearn --collect-data sklearn --collect-binaries sklearn --copy-metadata scikit-learn `
    --collect-submodules statsmodels --collect-data statsmodels --collect-binaries statsmodels --copy-metadata statsmodels `
    --collect-submodules PIL --collect-data PIL --collect-binaries PIL --copy-metadata pillow `
    --collect-submodules streamlit `
    --collect-submodules plotly `
    "$ROOT\desktop_launcher.py"

if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }

# Remove test folders, type-hint files and C++ headers the app never loads.
Write-Host "Removing files the app does not use..."
python "$ROOT\slim_bundle.py" "dist\$APP_NAME\_internal"
if ($LASTEXITCODE -ne 0) { throw "slim_bundle.py failed with exit code $LASTEXITCODE" }

Write-Host "Build complete."
Write-Host "Output folder: dist\$APP_NAME"