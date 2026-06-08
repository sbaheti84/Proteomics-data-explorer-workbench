$ErrorActionPreference = "Stop"

$APP_NAME = "ProteomicsWorkbench"

Write-Host "Cleaning old builds..."
Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue
Remove-Item -Force "$APP_NAME.spec" -ErrorAction SilentlyContinue

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
    --onefile `
    --paths "." `
    --paths "ge" `
    --paths "tp" `
    --paths "faq" `
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
    --add-data ".streamlit;.streamlit" `
    --add-data "assets;assets" `
    --add-data "faq;faq" `
    --add-data "ge;ge" `
    --add-data "tp;tp" `
    --add-data "pandyelab_logo.png;." `
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