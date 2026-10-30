#!/usr/bin/env bash
# run.sh - start Proteomics Data Explorer Workbench (macOS / Linux).
# Run ./setup.sh once before using this.
#
# Extra options are passed to Streamlit, for example:
#   ./run.sh --server.port 8502            use another port
#   ./run.sh --server.headless true        do not open a browser
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"   # Streamlit reads .streamlit/config.toml from the current folder

VPY="$ROOT/.venv/bin/python"
if [ ! -x "$VPY" ]; then
  echo "The Python environment is missing. Run ./setup.sh first." >&2
  exit 1
fi

# Use the R that built R_library/ (recorded by setup.sh), unless overridden.
if [ -z "${RSCRIPT_PATH:-}" ] && [ -f "$ROOT/R_library/.rscript_path" ]; then
  RSCRIPT_PATH="$(cat "$ROOT/R_library/.rscript_path")"
  export RSCRIPT_PATH
fi

export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
exec "$VPY" -m streamlit run app.py "$@"
