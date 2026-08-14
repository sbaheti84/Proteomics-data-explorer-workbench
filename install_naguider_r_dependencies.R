options(
  repos = c(
    CRAN = "https://cloud.r-project.org"
  )
)

cat("R version:\n")
print(R.version.string)

# ------------------------------------------------------------
# Basic installer packages
# ------------------------------------------------------------

cran_bootstrap <- c(
  "remotes",
  "BiocManager"
)

for (pkg in cran_bootstrap) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}


# ------------------------------------------------------------
# CRAN packages
# ------------------------------------------------------------

cran_packages <- c(
  "e1071",
  "norm",
  "imputeLCMD",
  "VIM",
  "rrcovNA",
  "mice",
  "missForest",
  "randomForest",
  "glmnet",
  "foreach",
  "itertools",
  "iterators"
)

for (pkg in cran_packages) {

  if (!requireNamespace(pkg, quietly = TRUE)) {

    cat("Installing CRAN package:", pkg, "\n")

    tryCatch(
      install.packages(pkg),
      error = function(e) {
        cat("FAILED:", pkg, "\n")
        cat(conditionMessage(e), "\n")
      }
    )
  }
}


# ------------------------------------------------------------
# Bioconductor packages
# ------------------------------------------------------------

bioc_packages <- c(
  "impute",
  "pcaMethods"
)

for (pkg in bioc_packages) {

  if (!requireNamespace(pkg, quietly = TRUE)) {

    cat("Installing Bioconductor package:", pkg, "\n")

    tryCatch(
      BiocManager::install(
        pkg,
        ask = FALSE,
        update = FALSE
      ),
      error = function(e) {
        cat("FAILED:", pkg, "\n")
        cat(conditionMessage(e), "\n")
      }
    )
  }
}


# ------------------------------------------------------------
# NAguideR special source packages
# ------------------------------------------------------------

gms_file <- file.path(
  getwd(),
  "r_packages",
  "GMSimpute_0.0.1.1.tar.gz"
)

seqknn_file <- file.path(
  getwd(),
  "r_packages",
  "SeqKnn_1.0.1.tar.gz"
)


if (!requireNamespace("GMSimpute", quietly = TRUE)) {

  if (!file.exists(gms_file)) {
    stop(
      paste(
        "Missing required file:",
        gms_file
      )
    )
  }

  cat("Installing GMSimpute from:", gms_file, "\n")

  install.packages(
    gms_file,
    repos = NULL,
    type = "source"
  )
}


if (!requireNamespace("SeqKnn", quietly = TRUE)) {

  if (!file.exists(seqknn_file)) {
    stop(
      paste(
        "Missing required file:",
        seqknn_file
      )
    )
  }

  cat("Installing SeqKnn from:", seqknn_file, "\n")

  install.packages(
    seqknn_file,
    repos = NULL,
    type = "source"
  )
}


# ------------------------------------------------------------
# DreamAI
#
# IMPORTANT:
# Package is inside /Code in the GitHub repository.
# ------------------------------------------------------------

if (!requireNamespace("DreamAI", quietly = TRUE)) {

  cat("Installing DreamAI from GitHub...\n")

  remotes::install_github(
    "WangLab-MSSM/DreamAI/Code",
    dependencies = TRUE,
    upgrade = "never"
  )
}


# ------------------------------------------------------------
# NAguideR
# ------------------------------------------------------------

if (!requireNamespace("NAguideR", quietly = TRUE)) {

  cat("Installing NAguideR...\n")

  remotes::install_github(
    "wangshisheng/NAguideR",
    dependencies = TRUE,
    upgrade = "never"
  )
}


# ------------------------------------------------------------
# Final validation
# ------------------------------------------------------------

required_packages <- c(
  "NAguideR",
  "e1071",
  "impute",
  "SeqKnn",
  "pcaMethods",
  "norm",
  "imputeLCMD",
  "VIM",
  "rrcovNA",
  "mice",
  "missForest",
  "GMSimpute",
  "DreamAI"
)


cat("\n========================================\n")
cat("Final package validation\n")
cat("========================================\n")

failed <- character()

for (pkg in required_packages) {

  ok <- requireNamespace(
    pkg,
    quietly = TRUE
  )

  cat(
    sprintf(
      "%-15s %s\n",
      pkg,
      if (ok) "OK" else "MISSING"
    )
  )

  if (!ok) {
    failed <- c(failed, pkg)
  }
}


if (length(failed) > 0) {

  stop(
    paste(
      "Failed to install:",
      paste(failed, collapse = ", ")
    )
  )
}

cat("\nAll required R packages are installed.\n")