---
name: proteomics-workbench-maintainer
description: Review, debug, modify, build, and validate the Proteomics Data Explorer Workbench. Use for Streamlit application changes, Total Proteome or GlycoExplorer modules, LFQ preprocessing, NAguideR/R integration, Docker, Windows PyInstaller builds, GitHub Actions, downloads, plotting, and dependency issues in this repository.
---

# Proteomics Workbench Maintainer

Work specifically within the Proteomics-data-explorer-workbench repository.

Before changing code, inspect the relevant files and trace the existing implementation.

Do not rewrite unrelated modules.

## Repository architecture

Main application:
- app.py

Application launchers include:
- app_launcher_TP.py
- app_launcher_TP_LF.py
- app_launcher_GE.py
- app_launcher_GE_LF.py
- app_launcher_GMC.py
- app_launcher_GPI.py
- app_launcher_FIIE.py

Shared code:
- tp/ = total proteomics utilities
- ge/ = glyco explorer utilities
- faq/ = FAQ modules
- assets/ = application assets

R / NAguideR:
- naguider_impute.R
- install_naguider_r_dependencies.R
- r_packages/
- renv.lock
- _dependencies.R

Deployment:
- Dockerfile
- build_windows.ps1
- .github/workflows/

## General modification rules

Before editing:

1. Identify the launcher or shared utility responsible for the requested behavior.
2. Check whether the function is shared by TMT and LFQ workflows.
3. Avoid duplicating existing helper functions.
4. Preserve existing Streamlit session-state behavior unless the task requires changing it.
5. Do not modify unrelated applications.

After editing:

1. Run Python syntax checks.
2. Search for duplicate implementations.
3. Report which files changed.
4. Report what was tested.
5. Report what could not be tested.

## LFQ scientific workflow

For Total Proteome LFQ, preserve this processing order:

raw abundance
→ sample/group selection
→ presence filtering
→ zero to missing
→ log2 transformation
→ normalization
→ imputation
→ QC
→ optional batch correction
→ final processed matrix
→ differential analysis

Do not silently change this order.

## log2 fold-change

The processed LFQ matrix is already log2 transformed.

Therefore:

log2FC =
mean(log2 test abundance)
-
mean(log2 control abundance)

Do not apply log2 again.

Do not calculate:

log2(mean(test)) - log2(mean(control))

on an already log2-transformed matrix.

Do not add a +1 pseudocount to the already log2-transformed matrix.

## Missing-value handling

Presence filtering must occur before imputation.

Do not perform another hidden imputation step inside differential analysis after global imputation.

Keep matrix states separate when possible:

- expr_raw
- expr_normalized
- expr_imputed
- expr_pre_batch
- expr_processed / expr_final

## NAguideR

Python calls NAguideR methods through:

naguider_impute.R

Important package names:

- NAguideR
- GMSimpute
- SeqKnn
- DreamAI
- impute
- pcaMethods
- imputeLCMD
- missForest
- mice
- VIM
- rrcovNA

GMSimpute is the correct capitalization.

Do not change it to GMSImpute.

DreamAI installation may require its package source location rather than treating it as a normal CRAN package.

## Batch correction

Batch correction occurs after preprocessing/imputation.

For ComBat:

- require a complete matrix;
- preserve biological group;
- warn if biology is confounded with batch;
- do not reconstruct raw abundance solely for batch correction.

## Windows build

The standalone Windows build contains Python and R.

Use:

- Python 3.12.8
- R 4.5.x
- streamlit-desktop-app
- PyInstaller
- bundled R runtime
- bundled R package library
- bundled naguider_impute.R

Prefer PyInstaller --onedir for the R-enabled application.

The finished application must not require the end user to separately install Python or R.

When troubleshooting Windows builds, inspect:

- .github/workflows/
- build_windows.ps1
- requirements.txt
- R installation steps
- r_packages/
- bundled R paths

## Docker

Docker is a hosted/server environment.

Docker should contain:

- Python
- R
- Rscript
- required R packages
- NAguideR bridge
- application source

Hosted mode should use browser-based downloads rather than desktop file dialogs.

Do not apply PyInstaller-specific assumptions to Docker.

## GitHub Actions

Keep dependency installation and application packaging separate.

Windows workflow:
- set up Python
- set up R
- install/verify R packages
- stage bundled R
- install Python packages
- build application
- package artifact

Docker workflow:
- build Dockerfile
- verify image
- optionally publish image

When a GitHub Actions job fails, identify the first real error rather than treating later exit-code messages as the root cause.

## Plotting

Preserve existing project plotting conventions unless asked to redesign them.

When changing a plot:
- preserve downloadable outputs;
- preserve interactive/static behavior where applicable;
- avoid changing unrelated plots.

## Safety checks before completion

For LFQ changes specifically search for:

- np.log2
- log2(
- pseudocount
- + 1
- replace_zeros
- imputation
- batch correction
- expr_processed
- expr_final

Check for:
- double log transformation;
- duplicate imputation;
- accidental pseudocounts;
- sample-column misalignment;
- unintended changes to raw data;
- batch correction before imputation.

## Response format

When completing a repository task, summarize:

1. Root cause or requested change.
2. Files modified.
3. Important implementation details.
4. Validation performed.
5. Remaining limitations.