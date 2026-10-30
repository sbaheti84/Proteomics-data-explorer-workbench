#!/usr/bin/env bash
# =============================================================================
# setup.sh - one-time setup to run Proteomics Data Explorer Workbench from
# source on macOS or Linux. (Windows: run setup.bat instead.)
#
#   git clone https://github.com/sbaheti84/Proteomics-data-explorer-workbench.git
#   cd Proteomics-data-explorer-workbench
#   ./setup.sh
#   ./run.sh
#
# What it does
#   1. Finds Python 3.12 (3.11 also works) and R (4.5.x recommended).
#   2. Creates a Python environment in .venv/ and installs requirements.txt.
#   3. Installs the R packages (NAguideR, SeqKnn, GMSimpute, DreamAI and their
#      dependencies) into ./R_library. This library lives inside this folder,
#      so your normal R setup is not changed. The app finds it by itself.
#   4. Checks that the Python and R packages load.
#
# It is safe to run again. Packages that are already installed are skipped.
# Run ./setup.sh --help for options.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV_DIR="$ROOT/.venv"
R_LIB_DIR="$ROOT/R_library"   # the app looks for <app folder>/R_library
R_LOG="$ROOT/setup-r.log"
TESTED_R="4.5"                # R version used by the Windows/macOS/Docker builds

SKIP_R=0
SYSTEM_DEPS=0
RECREATE_VENV=0
PYTHON_BIN=""
RSCRIPT_BIN="${RSCRIPT_PATH:-}"

# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------
if [ -t 1 ]; then
  BOLD=$'\033[1m'; RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RESET=$'\033[0m'
else
  BOLD=""; RED=""; GREEN=""; YELLOW=""; RESET=""
fi
step() { printf '\n%s==> %s%s\n' "$BOLD" "$*" "$RESET"; }
ok()   { printf '%s  OK%s  %s\n' "$GREEN" "$RESET" "$*"; }
warn() { printf '%s  WARNING:%s %s\n' "$YELLOW" "$RESET" "$*" >&2; }
die()  { printf '\n%sERROR:%s %s\n' "$RED" "$RESET" "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage: ./setup.sh [options]

Options
  --skip-r          Set up Python only. Everything works except NAguideR
                    imputation in the Total Proteomics label-free app.
  --system-deps     Debian/Ubuntu only: apt-get install the compilers and
                    libraries that R packages need (asks for your password).
  --python PATH     Use this Python instead of searching for one.
  --rscript PATH    Use this Rscript instead of searching for one.
  --recreate-venv   Delete .venv and build it again from scratch.
  -h, --help        Show this help.

Environment variables
  CRAN_REPO   CRAN mirror for R packages. On Ubuntu/Debian the default is
              Posit Package Manager, which has ready-built Linux packages
              (much faster than compiling).
  R_NCPUS     How many R packages to build in parallel (default: CPU count).
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-r)        SKIP_R=1 ;;
    --system-deps)   SYSTEM_DEPS=1 ;;
    --recreate-venv) RECREATE_VENV=1 ;;
    --python)        [ $# -ge 2 ] || die "--python needs a path"; PYTHON_BIN="$2"; shift ;;
    --rscript)       [ $# -ge 2 ] || die "--rscript needs a path"; RSCRIPT_BIN="$2"; shift ;;
    -h|--help)       usage; exit 0 ;;
    *)               die "Unknown option: $1 (see ./setup.sh --help)" ;;
  esac
  shift
done

case "$(uname -s)" in
  Darwin) PLATFORM="macos" ;;
  Linux)  PLATFORM="linux" ;;
  *)      die "This script is for macOS and Linux. On Windows, run setup.bat." ;;
esac

printf '%sProteomics Data Explorer Workbench - setup%s\n' "$BOLD" "$RESET"
echo "Folder:   $ROOT"
echo "Platform: $PLATFORM ($(uname -m))"

# -----------------------------------------------------------------------------
# 0. System libraries (optional, Debian/Ubuntu)
#    Same list as the Dockerfile, which is known to build every R package.
# -----------------------------------------------------------------------------
APT_PACKAGES=(
  build-essential gcc g++ gfortran make git curl ca-certificates pkg-config cmake
  python3-venv
  libcurl4-openssl-dev libssl-dev libxml2-dev libgit2-dev libfontconfig1-dev
  libfreetype6-dev libpng-dev libjpeg-dev libtiff-dev libharfbuzz-dev
  libfribidi-dev libcairo2-dev libxt-dev libicu-dev libblas-dev liblapack-dev
  libuv1-dev libbz2-dev liblzma-dev libpcre2-dev libreadline-dev zlib1g-dev
  libzstd-dev libncurses-dev libglpk-dev libnlopt-dev libgsl-dev libudunits2-dev
  libgdal-dev libgeos-dev libproj-dev
)

if [ "$SYSTEM_DEPS" -eq 1 ]; then
  step "Installing system libraries (apt-get)"
  if [ "$PLATFORM" != "linux" ] || ! command -v apt-get >/dev/null 2>&1; then
    die "--system-deps only works on Debian/Ubuntu (apt-get). Install the equivalent packages with your package manager."
  fi
  SUDO=""
  [ "$(id -u)" -ne 0 ] && SUDO="sudo"
  $SUDO apt-get update
  $SUDO apt-get install -y --no-install-recommends "${APT_PACKAGES[@]}"
  ok "System libraries installed"
fi

# -----------------------------------------------------------------------------
# 1. Python
# -----------------------------------------------------------------------------
step "Checking Python"

py_version() { "$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null; }
# 3.13+ is not supported: requirements.txt pins pipreqs==0.5.0, which needs Python < 3.13.
py_supported() { case "$1" in 3.11|3.12) return 0 ;; *) return 1 ;; esac; }

if [ -n "$PYTHON_BIN" ]; then
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python not found: $PYTHON_BIN"
  PY_VER="$(py_version "$PYTHON_BIN")" || die "Could not run $PYTHON_BIN"
  py_supported "$PY_VER" \
    || die "$PYTHON_BIN is Python $PY_VER. The app needs Python 3.12 (3.11 also works)."
else
  for candidate in python3.12 python3.11 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      v="$(py_version "$candidate" || true)"
      if [ -n "$v" ] && py_supported "$v"; then
        PYTHON_BIN="$(command -v "$candidate")"
        PY_VER="$v"
        break
      fi
    fi
  done
fi

if [ -z "$PYTHON_BIN" ]; then
  if [ "$PLATFORM" = "macos" ]; then
    hint="Install it from https://www.python.org/downloads/ (3.12) or with Homebrew: brew install python@3.12"
  else
    hint="Install it with your package manager, e.g. Ubuntu/Debian: sudo apt install python3.12 python3.12-venv"
  fi
  die "Python 3.12 was not found (3.11 also works). $hint
Then run ./setup.sh again. If Python is installed somewhere unusual: ./setup.sh --python /path/to/python3.12"
fi
ok "Python $PY_VER ($PYTHON_BIN)"
[ "$PY_VER" = "3.12" ] || warn "Python 3.12 is what the official builds use. $PY_VER works, but 3.12 is the safest choice."

# -----------------------------------------------------------------------------
# 2. Python environment + packages
# -----------------------------------------------------------------------------
step "Setting up the Python environment (.venv)"

VPY="$VENV_DIR/bin/python"

if [ -d "$VENV_DIR" ] && [ "$RECREATE_VENV" -eq 0 ]; then
  venv_ver="$(py_version "$VPY" || true)"
  if [ -z "$venv_ver" ] || ! py_supported "$venv_ver"; then
    warn "Existing .venv is broken or uses an unsupported Python (${venv_ver:-unknown}). Rebuilding it."
    RECREATE_VENV=1
  fi
fi
if [ "$RECREATE_VENV" -eq 1 ] && [ -d "$VENV_DIR" ]; then
  rm -rf "$VENV_DIR"
fi

if [ ! -x "$VPY" ]; then
  if ! "$PYTHON_BIN" -m venv "$VENV_DIR"; then
    rm -rf "$VENV_DIR"
    die "Could not create the Python environment.
On Ubuntu/Debian this usually means the venv module is missing: sudo apt install python${PY_VER}-venv"
  fi
  ok "Created .venv"
else
  ok "Using existing .venv (Python $(py_version "$VPY"))"
fi

export PIP_DISABLE_PIP_VERSION_CHECK=1
"$VPY" -m pip install --upgrade pip setuptools wheel
"$VPY" -m pip install -r "$ROOT/requirements.txt" \
  || die "Installing Python packages failed. Scroll up for the first error from pip."

"$VPY" - <<'PYCHECK' || die "Some Python packages did not import (see above)."
import contextlib, importlib, io
mods = ["streamlit", "pandas", "numpy", "scipy", "sklearn", "statsmodels",
        "plotly", "matplotlib", "seaborn", "openpyxl", "pyteomics", "st_aggrid",
        "statannotations", "neuroCombat", "upsetplot", "fpdf", "adjustText", "PIL"]
bad = []
# Hide Streamlit's harmless "missing ScriptRunContext" log line.
with contextlib.redirect_stderr(io.StringIO()):
    for m in mods:
        try:
            importlib.import_module(m)
        except Exception as e:  # report every failure, not just the first
            bad.append(f"{m}: {e}")
if bad:
    print("  Failed imports:\n   - " + "\n   - ".join(bad))
    raise SystemExit(1)
import streamlit
print(f"  All {len(mods)} key Python packages import (streamlit {streamlit.__version__}).")
PYCHECK
ok "Python packages installed"

# -----------------------------------------------------------------------------
# 3. R + R packages
# -----------------------------------------------------------------------------
if [ "$SKIP_R" -eq 1 ]; then
  step "Skipping R (--skip-r)"
  warn "NAguideR imputation in the Total Proteomics label-free app will not work until you run ./setup.sh without --skip-r."
else
  step "Checking R"

  if [ -z "$RSCRIPT_BIN" ]; then
    for candidate in \
        "$(command -v Rscript 2>/dev/null || true)" \
        /Library/Frameworks/R.framework/Resources/bin/Rscript \
        /opt/homebrew/bin/Rscript \
        /usr/local/bin/Rscript \
        /usr/bin/Rscript; do
      if [ -n "$candidate" ] && [ -x "$candidate" ]; then
        RSCRIPT_BIN="$candidate"
        break
      fi
    done
  fi

  if [ -z "$RSCRIPT_BIN" ]; then
    if [ "$PLATFORM" = "macos" ]; then
      if [ "$(uname -m)" = "arm64" ]; then pkg="big-sur-arm64/base/R-4.5.3-arm64.pkg"; else pkg="big-sur-x86_64/base/R-4.5.3-x86_64.pkg"; fi
      hint="Install R 4.5 from CRAN: https://cran.r-project.org/bin/macosx/$pkg"
    else
      hint="Install R 4.5 from CRAN: https://cloud.r-project.org/bin/linux/
(Quick option on Ubuntu/Debian: sudo apt install r-base r-base-dev, but Ubuntu's own R is older than 4.5.)"
    fi
    die "R was not found. $hint
Then run ./setup.sh again. If R is installed somewhere unusual: ./setup.sh --rscript /path/to/Rscript
To set up without R (no NAguideR imputation): ./setup.sh --skip-r"
  fi
  [ -x "$RSCRIPT_BIN" ] || die "Rscript not found or not executable: $RSCRIPT_BIN"

  # shellcheck disable=SC2016  # $ here is R code, not shell
  R_VER="$("$RSCRIPT_BIN" --vanilla -e 'cat(R.version$major, ".", R.version$minor, sep="")' 2>/dev/null)" \
    || die "Could not run $RSCRIPT_BIN"
  R_MAJOR_MINOR="$(echo "$R_VER" | cut -d. -f1,2)"
  ok "R $R_VER ($RSCRIPT_BIN)"
  case "$R_MAJOR_MINOR" in
    "$TESTED_R") ;;
    3.*|4.[0-2]) die "R $R_VER is too old. Install R $TESTED_R.x from CRAN." ;;
    *) warn "The official builds use R $TESTED_R.x. R $R_VER may work, but if a package fails, install R $TESTED_R.x." ;;
  esac

  # First run only: some packages must be compiled. DreamAI (GitHub) has C++
  # code everywhere; on Linux, Bioconductor's impute also has Fortran code.
  if [ ! -d "$R_LIB_DIR/DreamAI" ]; then
    if [ "$PLATFORM" = "macos" ]; then
      xcode-select -p >/dev/null 2>&1 \
        || die "Apple's command line tools (C/C++ compiler) are missing. One R package (DreamAI) has to be compiled.
Run:  xcode-select --install   then run ./setup.sh again."
    else
      missing=""
      for tool in gcc g++ gfortran make; do command -v "$tool" >/dev/null 2>&1 || missing="$missing $tool"; done
      [ -z "$missing" ] || die "Missing build tools:$missing. Some R packages have to be compiled.
Ubuntu/Debian: run ./setup.sh --system-deps   (or: sudo apt install build-essential gfortran)"
    fi
  fi

  # CRAN mirror. Linux gets ready-built packages from Posit Package Manager when
  # the distribution is supported; otherwise everything is compiled from source.
  if [ -z "${CRAN_REPO:-}" ] && [ "$PLATFORM" = "linux" ] && [ -r /etc/os-release ]; then
    codename="$(. /etc/os-release && echo "${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}")"
    case "$codename" in
      focal|jammy|noble|bullseye|bookworm)
        CRAN_REPO="https://packagemanager.posit.co/cran/__linux__/$codename/latest" ;;
    esac
  fi
  CRAN_REPO="${CRAN_REPO:-https://cloud.r-project.org}"
  R_NCPUS="${R_NCPUS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)}"

  step "Installing R packages into R_library/ (first run: about 5-40 minutes)"
  echo "  CRAN mirror: $CRAN_REPO"
  echo "  Full log:    $R_LOG"
  mkdir -p "$R_LIB_DIR"

  # The same library settings the app uses at run time, so what is installed
  # here is exactly what the app sees. --vanilla keeps the repo's renv
  # .Rprofile out of the way.
  if ! R_LIBS="$R_LIB_DIR" R_LIBS_USER="$R_LIB_DIR" \
       CRAN_REPO="$CRAN_REPO" R_NCPUS="$R_NCPUS" \
       "$RSCRIPT_BIN" --vanilla "$ROOT/install_naguider_r_dependencies.R" 2>&1 | tee "$R_LOG"; then
    die "R package installation failed. The error is near the end of $R_LOG.
Common causes: no internet access to CRAN/GitHub, or missing system libraries
(Linux: ./setup.sh --system-deps). Re-running ./setup.sh continues where it stopped."
  fi

  # Remember which R built this library. run.sh passes it to the app so a
  # different R on PATH cannot pick up packages built for another version.
  printf '%s\n' "$RSCRIPT_BIN" > "$R_LIB_DIR/.rscript_path"
  ok "R packages installed in R_library/"
fi

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
step "Setup complete"
echo "Start the app with:"
echo
echo "    ./run.sh"
echo
echo "It opens in your browser. If it does not, go to http://localhost:8501"
