# Proteomics Data Explorer Workbench

A Streamlit workbench for proteomics data, with five apps:

- **GlycoExplorer (TMT)** and **GlycoExplorer (Label-Free)**: glycoproteomics QC, statistics and plots
- **Total Proteome (TMT)** and **Total Proteome (Label-Free)**: protein-level QC, normalization, batch correction, statistics and reports
- **Glyco-Proteome Integrator**: compares glycosite changes with protein changes

Example input files are in `assets/example/`.

## Three ways to use it

| Way | Best for | How |
|---|---|---|
| Desktop app | Most users on Windows or macOS | Download the ZIP from [Releases](https://github.com/sbaheti84/Proteomics-data-explorer-workbench/releases) |
| From source | Developers, Linux users, anyone who wants the latest code | `git clone`, then the setup script (below) |
| Docker | Servers and shared installs | `docker run -p 8501:8501 ghcr.io/sbaheti84/proteomics-data-explorer-workbench:latest` |

---

## Install from source

### What you need first

| Tool | Version | Needed for |
|---|---|---|
| Git | any | Downloading the code (and one R package) |
| Python | **3.12** (3.11 also works; **3.13 and newer do not**) | The whole app |
| R | **4.5.x** recommended | Only NAguideR imputation in **Total Proteome (Label-Free)**. Skip it with `--skip-r` / `-SkipR` if you do not need that |
| C/C++ compiler | see below | One R package (DreamAI) must be compiled |

Compiler per system:

- **Windows:** [Rtools45](https://cran.r-project.org/bin/windows/Rtools/rtools45/rtools.html) (keep the default folder `C:\rtools45`)
- **macOS:** Apple's command line tools: run `xcode-select --install`
- **Linux (Ubuntu/Debian):** `./setup.sh --system-deps` installs everything needed

Disk space: about 1 GB for Python packages, plus about 1 GB for R packages.

### macOS and Linux

```bash
git clone https://github.com/sbaheti84/Proteomics-data-explorer-workbench.git
cd Proteomics-data-explorer-workbench
./setup.sh
./run.sh
```

On Ubuntu/Debian, run `./setup.sh --system-deps` the first time. It installs the compilers and system libraries the R packages need (it asks for your password).

### Windows

```bat
git clone https://github.com/sbaheti84/Proteomics-data-explorer-workbench.git
cd Proteomics-data-explorer-workbench
setup.bat
run.bat
```

You can also double-click `setup.bat` and then `run.bat` in File Explorer.

Missing Python, R or Rtools? Run `setup.bat -InstallPrereqs` to install them with winget (Windows may ask for permission).

### What happens

`setup` does four things. It is safe to run again; finished steps are skipped.

1. Finds Python and R, and checks their versions.
2. Creates a Python environment in `.venv/` and installs `requirements.txt` into it.
3. Installs the R packages (NAguideR, SeqKnn, GMSimpute, DreamAI and their dependencies) into `R_library/` inside the project folder. Your normal R setup is not changed, and the app finds `R_library/` by itself.
4. Checks that the Python and R packages load.

The first run takes about 5 to 40 minutes, mostly for R packages. Linux takes longest.

`run` starts the app and opens it in your browser. If the browser does not open, go to <http://localhost:8501>. To stop the app, press `Ctrl+C` in the terminal (or close the `run.bat` window).

### Options

| macOS / Linux | Windows | What it does |
|---|---|---|
| `--skip-r` | `-SkipR` | Python only. Everything works except NAguideR imputation |
| `--system-deps` | — | Ubuntu/Debian: install compilers and system libraries |
| — | `-InstallPrereqs` | Install missing Python 3.12, R 4.5 and Rtools45 with winget |
| `--python PATH` | `-Python PATH` | Use this Python |
| `--rscript PATH` | `-Rscript PATH` | Use this R |
| `--recreate-venv` | `-RecreateVenv` | Rebuild the Python environment from scratch |

To pass options to Streamlit, add them after `run`, for example `./run.sh --server.port=8502`.

### Updating

```bash
git pull
./setup.sh        # Windows: setup.bat
```

### Uninstalling

Delete the project folder. Everything the setup installs lives inside it (`.venv/` and `R_library/`).

### Troubleshooting

- **"Python 3.12 was not found" but I have Python 3.13.** The pinned packages in `requirements.txt` do not support 3.13 yet. Install 3.12 next to it (python.org, or `brew install python@3.12`), then run setup again.
- **"Permission denied" when running `./setup.sh`.** Run `chmod +x setup.sh run.sh` once, or use `bash setup.sh`.
- **R packages fail to compile on Linux.** Run `./setup.sh --system-deps`. The first error is near the end of `setup-r.log`.
- **You changed your R version.** Packages built for one R version may not load in another. Delete `R_library/` and run setup again.
- **Windows: pip fails with a "path too long" error.** Move the project to a short path such as `C:\pdew`, then run `setup.bat -RecreateVenv`.
- **Port 8501 is already in use.** Run `./run.sh --server.port=8502` (Windows: `run.bat --server.port=8502`).
- **NAguideR methods say R is not ready.** Start the app with `run.sh` / `run.bat`, not with `streamlit run app.py`. The run script tells the app which R built `R_library/`.

---

## For maintainers

- `install_naguider_r_dependencies.R` is shared by `setup.sh`, `setup.ps1`, the GitHub Actions builds and the Dockerfile. It reads two optional environment variables, `CRAN_REPO` and `R_NCPUS`. When they are not set, it behaves exactly as before.
- Desktop builds: run the **Build Windows App** and **Build macOS App** workflows in GitHub Actions. They publish to Releases.
- `R_library/` is ignored by git and Docker. Do not let it into the Docker image: the app uses `<app folder>/R_library` whenever it exists.
