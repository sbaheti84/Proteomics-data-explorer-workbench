# run.ps1 - start Proteomics Data Explorer Workbench (Windows).
# Run setup.bat once before using this. Easiest: double-click run.bat.
#
# Extra options are passed to Streamlit, for example:
#   .\run.bat --server.port=8502          use another port
#   .\run.bat --server.headless=true      do not open a browser
#
# No param() block on purpose: Streamlit options start with "--", and a param
# block would make PowerShell try to read them as its own parameters.

$ErrorActionPreference = "Stop"

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT   # Streamlit reads .streamlit\config.toml from the current folder

$VPY = Join-Path $ROOT ".venv\Scripts\python.exe"
if (-not (Test-Path $VPY)) {
    Write-Host "The Python environment is missing. Run setup.bat first." -ForegroundColor Red
    exit 1
}

# Use the R that built R_library\ (recorded by setup.ps1), unless overridden.
$rFile = Join-Path $ROOT "R_library\.rscript_path"
if (-not $env:RSCRIPT_PATH -and (Test-Path $rFile)) {
    $env:RSCRIPT_PATH = (Get-Content -Raw $rFile).Trim()
}

$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"

# Streamlit logs to stderr; do not let PowerShell 5.1 treat that as an error.
$ErrorActionPreference = "Continue"
& $VPY -m streamlit run app.py @args
exit $LASTEXITCODE
