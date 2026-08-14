#!/usr/bin/env Rscript

# Install the R backend used by the Streamlit LFQ + NAguideR integration.
# This installs NAguideR plus the method-specific packages required by the
# 23 imputation choices exposed in the Python application.

options(repos = c(CRAN = "https://cloud.r-project.org"))

install_cran_if_missing <- function(pkgs) {
  missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) {
    message("Installing CRAN packages: ", paste(missing, collapse = ", "))
    install.packages(missing, dependencies = NA)
  }
}

install_cran_if_missing(c("remotes", "BiocManager"))

# Bioconductor packages used directly by NAguideR methods.
bioc_pkgs <- c("impute", "pcaMethods")
missing_bioc <- bioc_pkgs[!vapply(bioc_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_bioc)) {
  message("Installing Bioconductor packages: ", paste(missing_bioc, collapse = ", "))
  BiocManager::install(missing_bioc, ask = FALSE, update = FALSE)
}

# Core CRAN packages called directly by the Streamlit R bridge.
install_cran_if_missing(c(
  "e1071", "norm", "imputeLCMD", "VIM", "rrcovNA", "mice", "missForest"
))

# NAguideR documents these archived/modified packages for local installation.
if (!requireNamespace("SeqKnn", quietly = TRUE)) {
  message("Installing SeqKnn from the NAguideR repository ...")
  remotes::install_url(
    "https://raw.githubusercontent.com/wangshisheng/NAguideR/master/SeqKnn_1.0.1.tar.gz",
    upgrade = "never",
    dependencies = NA
  )
}

if (!requireNamespace("GMSImpute", quietly = TRUE)) {
  message("Installing NAguideR-modified GMSImpute ...")
  remotes::install_url(
    "https://raw.githubusercontent.com/wangshisheng/NAguideR/master/GMSimpute_0.0.1.1.tar.gz",
    upgrade = "never",
    dependencies = NA
  )
}

if (!requireNamespace("DreamAI", quietly = TRUE)) {
  message("Installing DreamAI from GitHub ...")
  remotes::install_github(
    "WangLab-MSSM/DreamAI/Code",
    upgrade = "never",
    dependencies = NA
  )
}

# Install the NAguideR package after SeqKnn and the Bioconductor dependencies
# are present. NAguideR itself is used by the trKNN implementation and provides
# provenance for the R-backed workflow. Its normal Imports are installed here.
if (!requireNamespace("NAguideR", quietly = TRUE)) {
  message("Installing NAguideR from GitHub ...")
  remotes::install_github(
    "wangshisheng/NAguideR",
    upgrade = "never",
    dependencies = NA
  )
}

cat("\nNAguideR LFQ backend package check:\n")
check <- c(
  "NAguideR", "e1071", "impute", "SeqKnn", "pcaMethods", "norm",
  "imputeLCMD", "VIM", "rrcovNA", "mice", "missForest", "GMSimpute", "DreamAI"
)
status <- vapply(check, requireNamespace, logical(1), quietly = TRUE)
for (p in check) {
  cat(sprintf("  %-12s %s\n", p, if (status[[p]]) "OK" else "MISSING"))
}

if (!all(status)) {
  missing <- names(status)[!status]
  stop("Installation finished with missing packages: ", paste(missing, collapse = ", "))
}

cat("\nAll R packages required by the LFQ NAguideR backend are available.\n")
