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
if (!(Test-Path (Join-Path $ROOT "download_utility.py"))) { throw "download_utility.py is missing. Put the fixed file at repository root before building." }
foreach ($d in $requiredDirs) {
    $p = Join-Path $ROOT $d
    if (!(Test-Path $p)) { throw "Required path missing after setup: $p" }
    Write-Host "Found: $p"
}

Write-Host "Top-level files/folders before build:"
Get-ChildItem -Force $ROOT | Select-Object Name, Mode, Length | Format-Table -AutoSize

Write-Host "Building Windows app..."

streamlit-desktop-app build app.py `
  --name $APP_NAME `
  --streamlit-options `
    --server.maxUploadSize=1024 `
    --server.enableStaticServing=true `
    --theme.base=light `
    "--theme.primaryColor=#004b87" `
    "--theme.backgroundColor=#ffffff" `
    "--theme.secondaryBackgroundColor=#f0f4f8" `
    "--theme.textColor=#1e293b" `
    "--theme.font=sans serif" `
  --pyinstaller-options `
    --onedir `
    --paths "$ROOT" `
    --paths "$(Join-Path $ROOT 'ge')" `
    --paths "$(Join-Path $ROOT 'tp')" `
    --paths "$(Join-Path $ROOT 'faq')" `
    --add-data "$ROOT\app_launcher_*.py;." `
    --add-data "$ROOT\total_proteome_lfq_naguider.py;." `
    --add-data "$ROOT\download_utility.py;." `
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
    --hidden-import app_launcher_FILE `
    --hidden-import faq.faq_GE `
    --hidden-import faq.faq_GE_LF `
    --hidden-import faq.faq_TP `
    --hidden-import faq.faq_TP_LF `
    --hidden-import faq.faq_GMC `
    --hidden-import faq.faq_GPI `
    --hidden-import faq.faq_FILE `
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
    --collect-all streamlit `
    --collect-all plotly `
    --collect-all openpyxl `
    --collect-all matplotlib `
    --collect-all scipy `
    --collect-all pandas `
    --collect-all numpy `
    --collect-all sklearn `
    --collect-all statsmodels `
    --collect-all PIL `
    --collect-submodules streamlit `
    --collect-submodules plotly `
    --noconfirm

Write-Host "Build complete."
Write-Host "Output folder: dist\$APP_NAME"
