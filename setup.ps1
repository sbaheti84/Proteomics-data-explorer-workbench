<#
.SYNOPSIS
  One-time setup to run Proteomics Data Explorer Workbench from source on Windows.

.DESCRIPTION
  1. Finds Python 3.12 (3.11 also works), R (4.5.x recommended) and Rtools.
  2. Creates a Python environment in .venv and installs requirements.txt.
  3. Installs the R packages (NAguideR, SeqKnn, GMSimpute, DreamAI and their
     dependencies) into .\R_library. This library lives inside this folder, so
     your normal R setup is not changed. The app finds it by itself.
  4. Checks that the Python and R packages load.

  Safe to run again: packages that are already installed are skipped.

  Easiest way to run it: double-click setup.bat, or in a terminal:
    powershell -ExecutionPolicy Bypass -File .\setup.ps1

.PARAMETER InstallPrereqs
  Use winget to install whatever is missing: Python 3.12, R 4.5 and Rtools45.

.PARAMETER SkipR
  Set up Python only. Everything works except NAguideR imputation in the
  Total Proteomics label-free app.

.PARAMETER Python
  Path to python.exe to use instead of searching.

.PARAMETER Rscript
  Path to Rscript.exe to use instead of searching.

.PARAMETER RecreateVenv
  Delete .venv and build it again from scratch.
#>
[CmdletBinding()]
param(
    [switch]$InstallPrereqs,
    [switch]$SkipR,
    [string]$Python = "",
    [string]$Rscript = "",
    [switch]$RecreateVenv
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT
[Environment]::CurrentDirectory = $ROOT   # R uses relative paths (r_packages\...)

$VENV_DIR  = Join-Path $ROOT ".venv"
$VPY       = Join-Path $VENV_DIR "Scripts\python.exe"
$R_LIB_DIR = Join-Path $ROOT "R_library"   # the app looks for <app folder>\R_library
$TESTED_R  = "4.5"                         # R version used by the official builds
$WINGET_R_VERSION = "4.5.3"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
function Write-Step($msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "  OK  $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "  WARNING: $msg" -ForegroundColor Yellow }
function Fail($msg) {
    Write-Host ""
    Write-Host "ERROR: $msg" -ForegroundColor Red
    exit 1
}

# Run a program and show its output. Returns the exit code.
# Windows PowerShell 5.1 turns anything a program writes to stderr into an
# error when ErrorActionPreference is Stop (pip and R write progress there),
# so relax it around native calls.
function Invoke-Native([string]$Exe, [string[]]$Arguments) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $global:LASTEXITCODE = 0
        & $Exe @Arguments | Out-Host
        return $global:LASTEXITCODE
    } finally {
        $ErrorActionPreference = $old
    }
}

# Run a program and return its trimmed stdout, or $null if it failed.
function Get-NativeOutput([string]$Exe, [string[]]$Arguments) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $global:LASTEXITCODE = -1
        $out = & $Exe @Arguments 2>$null
        if ($global:LASTEXITCODE -ne 0) { return $null }
        return (($out | Out-String).Trim())
    } catch {
        return $null
    } finally {
        $ErrorActionPreference = $old
    }
}

function Test-Winget { return [bool](Get-Command winget.exe -ErrorAction SilentlyContinue) }

function Install-WithWinget([string]$Id, [string]$Version = "", [string[]]$Extra = @()) {
    if (-not (Test-Winget)) {
        Fail "winget is not available on this computer. Install the missing program by hand (links above) and run setup again."
    }
    $wingetArgs = @("install", "-e", "--id", $Id, "--accept-package-agreements", "--accept-source-agreements") + $Extra
    if ($Version) { $wingetArgs += @("--version", $Version) }
    Write-Host "  winget $($wingetArgs -join ' ')"
    return (Invoke-Native "winget.exe" $wingetArgs)
}

# Note on quoting: Windows PowerShell 5.1 drops double quotes inside arguments
# passed to programs, so the Python/R snippets below only use single quotes.
$PY_VERSION_CODE = "import sys; print('%d.%d' % sys.version_info[:2])"
$PY_EXE_CODE     = "import sys; print(sys.executable)"

function Get-PyVersion([string]$Exe) { return (Get-NativeOutput $Exe @("-c", $PY_VERSION_CODE)) }
# 3.13+ is not supported: requirements.txt pins pipreqs==0.5.0, which needs Python < 3.13.
function Test-PySupported([string]$Ver) { return (@("3.11", "3.12") -contains $Ver) }

function Find-Python {
    # 1. The py launcher knows every python.org install.
    $launchers = @(
        (Get-Command py.exe -ErrorAction SilentlyContinue | ForEach-Object { $_.Source }),
        "$env:LOCALAPPDATA\Programs\Python\Launcher\py.exe",
        "$env:WINDIR\py.exe"
    ) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique

    foreach ($launcher in $launchers) {
        foreach ($v in @("3.12", "3.11")) {
            $exe = Get-NativeOutput $launcher @("-$v", "-c", $PY_EXE_CODE)
            if ($exe -and (Test-Path $exe)) { return $exe }
        }
    }

    # 2. Known install folders and PATH. The "python.exe" Microsoft Store stub
    #    fails the version check, so it is skipped automatically.
    $candidates = @()
    foreach ($v in @("312", "311")) {
        $candidates += "$env:LOCALAPPDATA\Programs\Python\Python$v\python.exe"
        $candidates += "$env:ProgramFiles\Python$v\python.exe"
    }
    $candidates += (Get-Command python.exe -All -ErrorAction SilentlyContinue | ForEach-Object { $_.Source })

    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c)) {
            $ver = Get-PyVersion $c
            if ($ver -and (Test-PySupported $ver)) { return $c }
        }
    }
    return $null
}

function Find-Rscript {
    $found = @()

    if ($env:RSCRIPT_PATH -and (Test-Path $env:RSCRIPT_PATH)) { return $env:RSCRIPT_PATH }

    # Every R in the standard install folders, newest first.
    foreach ($base in @("$env:ProgramFiles\R", "$env:LOCALAPPDATA\Programs\R")) {
        if (Test-Path $base) {
            Get-ChildItem -Path $base -Directory -Filter "R-*" -ErrorAction SilentlyContinue | ForEach-Object {
                $exe = Join-Path $_.FullName "bin\Rscript.exe"
                if (Test-Path $exe) {
                    $verText = $_.Name.Substring(2)
                    $ver = $null
                    if ([version]::TryParse($verText, [ref]$ver)) {
                        $found += [pscustomobject]@{ Path = $exe; Version = $ver }
                    }
                }
            }
        }
    }

    # Prefer the newest 4.5.x (what the official builds use), then the newest R.
    $preferred = $found | Where-Object { "$($_.Version.Major).$($_.Version.Minor)" -eq $TESTED_R } |
        Sort-Object Version -Descending | Select-Object -First 1
    if ($preferred) { return $preferred.Path }
    $newest = $found | Sort-Object Version -Descending | Select-Object -First 1
    if ($newest) { return $newest.Path }

    $onPath = Get-Command Rscript.exe -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    return $null
}

function Test-Rtools {
    foreach ($name in @("RTOOLS45_HOME", "RTOOLS46_HOME", "RTOOLS47_HOME")) {
        $p = [Environment]::GetEnvironmentVariable($name)
        if ($p -and (Test-Path (Join-Path $p "usr\bin\make.exe"))) { return $true }
    }
    foreach ($p in @("C:\rtools45", "C:\rtools46", "C:\rtools47")) {
        if (Test-Path (Join-Path $p "usr\bin\make.exe")) { return $true }
    }
    return $false
}

Write-Host "Proteomics Data Explorer Workbench - setup" -ForegroundColor White
Write-Host "Folder: $ROOT"

# -----------------------------------------------------------------------------
# 1. Python
# -----------------------------------------------------------------------------
Write-Step "Checking Python"

if ($Python) {
    if (-not (Test-Path $Python)) { Fail "Python not found: $Python" }
    $pyVer = Get-PyVersion $Python
    if (-not $pyVer) { Fail "Could not run $Python" }
    if (-not (Test-PySupported $pyVer)) {
        Fail "$Python is Python $pyVer. The app needs Python 3.12 (3.11 also works)."
    }
} else {
    $Python = Find-Python
    if (-not $Python -and $InstallPrereqs) {
        Write-Host "  Python 3.12 not found. Installing it with winget..."
        $code = Install-WithWinget "Python.Python.3.12" "" @("--scope", "user")
        if ($code -ne 0) { Fail "winget could not install Python 3.12 (exit code $code)." }
        $Python = Find-Python
    }
    if (-not $Python) {
        Fail ("Python 3.12 was not found (3.11 also works).`n" +
              "Install Python 3.12 from https://www.python.org/downloads/windows/ , or run:`n" +
              "    setup.bat -InstallPrereqs`n" +
              "Then run setup again. Python somewhere unusual? setup.bat -Python C:\path\to\python.exe")
    }
    $pyVer = Get-PyVersion $Python
}
Write-Ok "Python $pyVer ($Python)"
if ($pyVer -ne "3.12") { Write-Warn "Python 3.12 is what the official builds use. $pyVer works, but 3.12 is the safest choice." }

# -----------------------------------------------------------------------------
# 2. Python environment + packages
# -----------------------------------------------------------------------------
Write-Step "Setting up the Python environment (.venv)"

if ((Test-Path $VENV_DIR) -and -not $RecreateVenv) {
    $venvVer = $null
    if (Test-Path $VPY) { $venvVer = Get-PyVersion $VPY }
    if (-not $venvVer -or -not (Test-PySupported $venvVer)) {
        Write-Warn "Existing .venv is broken or uses an unsupported Python. Rebuilding it."
        $RecreateVenv = $true
    }
}
if ($RecreateVenv -and (Test-Path $VENV_DIR)) {
    Remove-Item -Recurse -Force $VENV_DIR
}

if (-not (Test-Path $VPY)) {
    $code = Invoke-Native $Python @("-m", "venv", $VENV_DIR)
    if ($code -ne 0 -or -not (Test-Path $VPY)) { Fail "Could not create the Python environment in $VENV_DIR" }
    Write-Ok "Created .venv"
} else {
    Write-Ok "Using existing .venv (Python $(Get-PyVersion $VPY))"
}

$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
$code = Invoke-Native $VPY @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
if ($code -ne 0) { Fail "Upgrading pip failed (exit code $code)." }
$code = Invoke-Native $VPY @("-m", "pip", "install", "-r", (Join-Path $ROOT "requirements.txt"))
if ($code -ne 0) {
    Fail ("Installing Python packages failed. Scroll up for the first error from pip.`n" +
          "If it mentions a path that is too long, move this folder somewhere short (e.g. C:\pdew) and run setup again.")
}

$checkCode = @'
import contextlib, importlib, io
mods = ['streamlit', 'pandas', 'numpy', 'scipy', 'sklearn', 'statsmodels',
        'plotly', 'matplotlib', 'seaborn', 'openpyxl', 'pyteomics', 'st_aggrid',
        'statannotations', 'neuroCombat', 'upsetplot', 'fpdf', 'adjustText', 'PIL']
bad = []
# Hide Streamlit's harmless "missing ScriptRunContext" log line.
with contextlib.redirect_stderr(io.StringIO()):
    for m in mods:
        try:
            importlib.import_module(m)
        except Exception as e:
            bad.append(m + ': ' + str(e))
if bad:
    print('  Failed imports:')
    for b in bad:
        print('   - ' + b)
    raise SystemExit(1)
import streamlit
print('  All %d key Python packages import (streamlit %s).' % (len(mods), streamlit.__version__))
'@
$checkFile = Join-Path $env:TEMP "pdew_import_check.py"
Set-Content -Path $checkFile -Value $checkCode -Encoding ASCII
$code = Invoke-Native $VPY @($checkFile)
Remove-Item -Force $checkFile -ErrorAction SilentlyContinue
if ($code -ne 0) { Fail "Some Python packages did not import (see above)." }
Write-Ok "Python packages installed"

# -----------------------------------------------------------------------------
# 3. R + R packages
# -----------------------------------------------------------------------------
if ($SkipR) {
    Write-Step "Skipping R (-SkipR)"
    Write-Warn "NAguideR imputation in the Total Proteomics label-free app will not work until you run setup without -SkipR."
} else {
    Write-Step "Checking R"

    if (-not $Rscript) { $Rscript = Find-Rscript }
    if (-not $Rscript -and $InstallPrereqs) {
        Write-Host "  R not found. Installing R $WINGET_R_VERSION with winget (Windows may ask for permission)..."
        $code = Install-WithWinget "RProject.R" $WINGET_R_VERSION
        if ($code -ne 0) {
            Write-Warn "R $WINGET_R_VERSION is not available in winget. Installing the newest R instead."
            $code = Install-WithWinget "RProject.R"
            if ($code -ne 0) { Fail "winget could not install R (exit code $code)." }
        }
        $Rscript = Find-Rscript
    }
    if (-not $Rscript) {
        Fail ("R was not found. Install R 4.5 from https://cran.r-project.org/bin/windows/base/old/ (4.5.3), or run:`n" +
              "    setup.bat -InstallPrereqs`n" +
              "R somewhere unusual? setup.bat -Rscript 'C:\path\to\Rscript.exe'`n" +
              "To set up without R (no NAguideR imputation): setup.bat -SkipR")
    }
    if (-not (Test-Path $Rscript)) { Fail "Rscript not found: $Rscript" }

    $rVer = Get-NativeOutput $Rscript @("--vanilla", "-e", "cat(paste(R.version[['major']], R.version[['minor']], sep='.'))")
    if (-not $rVer) { Fail "Could not run $Rscript" }
    Write-Ok "R $rVer ($Rscript)"
    $rParts = $rVer.Split(".")
    $rMajorMinor = "$($rParts[0]).$($rParts[1])"
    if ([version]"$rMajorMinor.0" -lt [version]"4.3.0") { Fail "R $rVer is too old. Install R $TESTED_R.x." }
    if ($rMajorMinor -ne $TESTED_R) {
        Write-Warn "The official builds use R $TESTED_R.x. R $rVer may work, but if a package fails, install R $TESTED_R.x."
    }

    # DreamAI has C++ code, so the first install needs Rtools.
    if (-not (Test-Path (Join-Path $R_LIB_DIR "DreamAI"))) {
        if (-not (Test-Rtools) -and $InstallPrereqs) {
            Write-Host "  Rtools not found. Installing Rtools45 with winget..."
            $code = Install-WithWinget "RProject.Rtools"
            if ($code -ne 0) { Fail "winget could not install Rtools (exit code $code)." }
        }
        if (-not (Test-Rtools)) {
            Fail ("Rtools45 (the compiler R uses on Windows) was not found. One R package (DreamAI) has to be compiled.`n" +
                  "Install it from https://cran.r-project.org/bin/windows/Rtools/rtools45/rtools.html (keep the default folder C:\rtools45),`n" +
                  "or run: setup.bat -InstallPrereqs")
        }
        Write-Ok "Rtools found"
    }

    if (-not (Get-Command git.exe -ErrorAction SilentlyContinue)) {
        Fail "git was not found. It is needed to download the DreamAI R package. Install Git for Windows: https://git-scm.com/download/win"
    }

    Write-Step "Installing R packages into R_library\ (first run: about 5-20 minutes)"
    New-Item -ItemType Directory -Force -Path $R_LIB_DIR | Out-Null

    # The same library settings the app uses at run time, so what is installed
    # here is exactly what the app sees. --vanilla keeps the repo's renv
    # .Rprofile out of the way.
    $saved = @{ R_LIBS = $env:R_LIBS; R_LIBS_USER = $env:R_LIBS_USER; R_NCPUS = $env:R_NCPUS }
    $env:R_LIBS = $R_LIB_DIR
    $env:R_LIBS_USER = $R_LIB_DIR
    if (-not $env:R_NCPUS) { $env:R_NCPUS = "$([Environment]::ProcessorCount)" }
    try {
        $code = Invoke-Native $Rscript @("--vanilla", (Join-Path $ROOT "install_naguider_r_dependencies.R"))
    } finally {
        $env:R_LIBS = $saved.R_LIBS
        $env:R_LIBS_USER = $saved.R_LIBS_USER
        $env:R_NCPUS = $saved.R_NCPUS
    }
    if ($code -ne 0) {
        Fail ("R package installation failed (exit code $code). The error is in the last lines above.`n" +
              "Common causes: no internet access to CRAN/GitHub, or Rtools missing. Re-running setup continues where it stopped.")
    }

    # Remember which R built this library. run.bat passes it to the app so a
    # different R cannot pick up packages built for another version.
    Set-Content -Path (Join-Path $R_LIB_DIR ".rscript_path") -Value $Rscript -Encoding UTF8
    Write-Ok "R packages installed in R_library\"
}

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
Write-Step "Setup complete"
Write-Host "Start the app by double-clicking run.bat, or in a terminal:"
Write-Host ""
Write-Host "    .\run.bat"
Write-Host ""
Write-Host "It opens in your browser. If it does not, go to http://localhost:8501"
