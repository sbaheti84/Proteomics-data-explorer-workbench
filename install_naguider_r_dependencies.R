# ============================================================
# install_naguider_r_dependencies.R
#
# Install all R dependencies required by:
#   - NAguideR
#   - SeqKnn
#   - GMSimpute
#   - DreamAI
#
# Designed for:
#   - GitHub Actions
#   - Windows
#   - R 4.5.x
#   - Bundling into PyInstaller application
#
# Required repository files:
#
#   r_packages/SeqKnn_1.0.1.tar.gz
#   r_packages/GMSimpute_0.0.1.1.tar.gz
#
# ============================================================


# ------------------------------------------------------------
# Global options
# ------------------------------------------------------------

options(
    repos = c(
        CRAN = "https://cloud.r-project.org"
    ),
    Ncpus = 2,
    timeout = 1200
)

cat("\n")
cat("============================================================\n")
cat("NAguideR dependency installation\n")
cat("============================================================\n")

cat("\nR version:\n")
cat(R.version.string, "\n")

cat("\nPlatform:\n")
cat(R.version$platform, "\n")

cat("\nR_HOME:\n")
cat(R.home(), "\n")

cat("\nLibrary paths:\n")
print(.libPaths())

cat("\nCRAN repository:\n")
print(getOption("repos"))


# ============================================================
# Helper functions
# ============================================================


install_cran_package <- function(pkg) {

    if (requireNamespace(pkg, quietly = TRUE)) {

        version <- tryCatch(
            as.character(packageVersion(pkg)),
            error = function(e) "unknown"
        )

        cat(
            sprintf(
                "CRAN %-20s already installed [%s]\n",
                pkg,
                version
            )
        )

        return(invisible(TRUE))
    }

    cat("\n")
    cat("------------------------------------------------------------\n")
    cat("Installing CRAN package:", pkg, "\n")
    cat("------------------------------------------------------------\n")

    tryCatch(
        {
            install.packages(
                pkg,
                dependencies = c(
                    "Depends",
                    "Imports",
                    "LinkingTo"
                )
            )
        },
        error = function(e) {

            cat("\nFAILED installing CRAN package:", pkg, "\n")
            cat("Error:", conditionMessage(e), "\n")

            stop(
                paste(
                    "Failed to install CRAN package:",
                    pkg
                ),
                call. = FALSE
            )
        }
    )

    if (!requireNamespace(pkg, quietly = TRUE)) {

        stop(
            paste(
                "CRAN package installation completed but",
                pkg,
                "cannot be loaded."
            ),
            call. = FALSE
        )
    }

    cat(
        "Successfully installed:",
        pkg,
        as.character(packageVersion(pkg)),
        "\n"
    )

    invisible(TRUE)
}


install_bioc_package <- function(pkg) {

    if (requireNamespace(pkg, quietly = TRUE)) {

        version <- tryCatch(
            as.character(packageVersion(pkg)),
            error = function(e) "unknown"
        )

        cat(
            sprintf(
                "Bioc %-20s already installed [%s]\n",
                pkg,
                version
            )
        )

        return(invisible(TRUE))
    }

    cat("\n")
    cat("------------------------------------------------------------\n")
    cat("Installing Bioconductor package:", pkg, "\n")
    cat("------------------------------------------------------------\n")

    tryCatch(
        {
            BiocManager::install(
                pkg,
                ask = FALSE,
                update = FALSE
            )
        },
        error = function(e) {

            cat(
                "\nFAILED installing Bioconductor package:",
                pkg,
                "\n"
            )

            cat(
                "Error:",
                conditionMessage(e),
                "\n"
            )

            stop(
                paste(
                    "Failed to install Bioconductor package:",
                    pkg
                ),
                call. = FALSE
            )
        }
    )

    if (!requireNamespace(pkg, quietly = TRUE)) {

        stop(
            paste(
                "Bioconductor package installation completed but",
                pkg,
                "cannot be loaded."
            ),
            call. = FALSE
        )
    }

    cat(
        "Successfully installed:",
        pkg,
        as.character(packageVersion(pkg)),
        "\n"
    )

    invisible(TRUE)
}


install_local_package <- function(pkg, file) {

    if (requireNamespace(pkg, quietly = TRUE)) {

        version <- tryCatch(
            as.character(packageVersion(pkg)),
            error = function(e) "unknown"
        )

        cat(
            sprintf(
                "LOCAL %-19s already installed [%s]\n",
                pkg,
                version
            )
        )

        return(invisible(TRUE))
    }

    if (!file.exists(file)) {

        stop(
            paste(
                "Required local R package does not exist:",
                file
            ),
            call. = FALSE
        )
    }

    file <- normalizePath(
        file,
        winslash = "/",
        mustWork = TRUE
    )

    cat("\n")
    cat("------------------------------------------------------------\n")
    cat("Installing local package:", pkg, "\n")
    cat("Source:", file, "\n")
    cat("------------------------------------------------------------\n")

    tryCatch(
        {
            install.packages(
                file,
                repos = NULL,
                type = "source"
            )
        },
        error = function(e) {

            cat(
                "\nFAILED installing local package:",
                pkg,
                "\n"
            )

            cat(
                "Source:",
                file,
                "\n"
            )

            cat(
                "Error:",
                conditionMessage(e),
                "\n"
            )

            stop(
                paste(
                    "Failed to install local package:",
                    pkg
                ),
                call. = FALSE
            )
        }
    )

    if (!requireNamespace(pkg, quietly = TRUE)) {

        stop(
            paste(
                "Local package installation completed but",
                pkg,
                "cannot be loaded."
            ),
            call. = FALSE
        )
    }

    cat(
        "Successfully installed:",
        pkg,
        as.character(packageVersion(pkg)),
        "\n"
    )

    invisible(TRUE)
}


# ============================================================
# Verify source tarballs exist
# ============================================================

cat("\n")
cat("============================================================\n")
cat("Checking local source packages\n")
cat("============================================================\n")


seqknn_file <- file.path(
    "r_packages",
    "SeqKnn_1.0.1.tar.gz"
)

gmsimpute_file <- file.path(
    "r_packages",
    "GMSimpute_0.0.1.1.tar.gz"
)


required_files <- c(
    seqknn_file,
    gmsimpute_file
)


for (file in required_files) {

    if (!file.exists(file)) {

        stop(
            paste(
                "Required file is missing:",
                file
            ),
            call. = FALSE
        )
    }

    cat(
        "Found:",
        normalizePath(
            file,
            winslash = "/",
            mustWork = TRUE
        ),
        "\n"
    )
}


# ============================================================
# Bootstrap package managers
# ============================================================

cat("\n")
cat("============================================================\n")
cat("Installing package managers\n")
cat("============================================================\n")


install_cran_package("remotes")

install_cran_package("BiocManager")


# ============================================================
# Bioconductor dependencies
#
# NAguideR DESCRIPTION requires:
#   impute
#   pcaMethods
# ============================================================

cat("\n")
cat("============================================================\n")
cat("Installing Bioconductor dependencies\n")
cat("============================================================\n")


bioc_packages <- c(
    "impute",
    "pcaMethods"
)


for (pkg in bioc_packages) {

    install_bioc_package(pkg)

}


# ============================================================
# CRAN dependencies
#
# Includes:
#   NAguideR dependencies
#   DreamAI dependencies
#
# We explicitly install them before install_github() so
# remotes does not have to guess or upgrade everything.
# ============================================================

cat("\n")
cat("============================================================\n")
cat("Installing CRAN dependencies\n")
cat("============================================================\n")


cran_packages <- c(

    # --------------------------------------------------------
    # NAguideR UI / application
    # --------------------------------------------------------

    "shiny",
    "shinyBS",
    "shinyjs",
    "shinyWidgets",
    "DT",
    "gdata",
    "ggplot2",
    "ggsci",
    "openxlsx",
    "data.table",
    "tidyverse",
    "ggExtra",
    "cowplot",
    "Amelia",

    # --------------------------------------------------------
    # NAguideR imputation dependencies
    # --------------------------------------------------------

    "e1071",
    "norm",
    "imputeLCMD",
    "VIM",
    "rrcovNA",
    "mice",
    "missForest",

    # --------------------------------------------------------
    # DreamAI dependencies
    # --------------------------------------------------------

    "randomForest",
    "glmnet",
    "foreach",
    "itertools",
    "iterators",
    "Rcpp",

    # These normally come with R, but check explicitly because
    # DreamAI lists them as Depends.
    "cluster",
    "survival",
    "Matrix"
)


for (pkg in unique(cran_packages)) {

    install_cran_package(pkg)

}


# ============================================================
# Install SeqKnn
#
# NAguideR recommends the old source archive because SeqKnn
# is no longer reliably installable by the normal package
# installation route.
# ============================================================

cat("\n")
cat("============================================================\n")
cat("Installing SeqKnn\n")
cat("============================================================\n")


install_local_package(
    pkg = "SeqKnn",
    file = seqknn_file
)


# ============================================================
# Install modified GMSImpute
#
# NAguideR distributes a modified GMSImpute source archive.
# Use that exact local package instead of another version.
# ============================================================

cat("\n")
cat("============================================================\n")
cat("Installing GMSImpute\n")
cat("============================================================\n")


install_local_package(
    pkg = "GMSimpute",
    file = gmsimpute_file
)


# ============================================================
# Install DreamAI
#
# IMPORTANT:
#
# DreamAI is NOT an R package at the repository root.
#
# Correct package location:
#
#   WangLab-MSSM/DreamAI
#                      |
#                      +-- Code/
#                          DESCRIPTION
#
# Therefore subdir = "Code" is REQUIRED.
# ============================================================

cat("\n")
cat("============================================================\n")
cat("Installing DreamAI\n")
cat("============================================================\n")


if (requireNamespace("DreamAI", quietly = TRUE)) {

    cat(
        "DreamAI already installed:",
        as.character(packageVersion("DreamAI")),
        "\n"
    )

} else {

    cat(
        "Installing DreamAI from ",
        "WangLab-MSSM/DreamAI, subdir=Code\n",
        sep = ""
    )

    tryCatch(
        {

            remotes::install_github(
                repo = "WangLab-MSSM/DreamAI",
                subdir = "Code",
                dependencies = FALSE,
                upgrade = "never",
                force = TRUE,
                quiet = FALSE
            )

        },
        error = function(e) {

            cat("\n")
            cat("============================================================\n")
            cat("DREAMAI INSTALLATION FAILED\n")
            cat("============================================================\n")

            cat(
                "Error:",
                conditionMessage(e),
                "\n"
            )

            cat(
                "\nExpected GitHub package location:\n"
            )

            cat(
                "WangLab-MSSM/DreamAI/Code\n"
            )

            stop(
                "DreamAI installation failed.",
                call. = FALSE
            )
        }
    )
}


if (!requireNamespace("DreamAI", quietly = TRUE)) {

    stop(
        paste(
            "DreamAI GitHub installation completed",
            "but DreamAI cannot be loaded."
        ),
        call. = FALSE
    )
}


cat(
    "DreamAI:",
    as.character(packageVersion("DreamAI")),
    "OK\n"
)


# ============================================================
# Install NAguideR
#
# Dependencies are FALSE intentionally.
#
# We already installed its dependencies above, including
# SeqKnn from the required local source package.
#
# This prevents remotes from trying to replace our special
# local packages.
# ============================================================

cat("\n")
cat("============================================================\n")
cat("Installing NAguideR\n")
cat("============================================================\n")


if (requireNamespace("NAguideR", quietly = TRUE)) {

    cat(
        "NAguideR already installed:",
        as.character(packageVersion("NAguideR")),
        "\n"
    )

} else {

    cat(
        "Installing NAguideR from GitHub...\n"
    )

    tryCatch(
        {

            remotes::install_github(
                repo = "wangshisheng/NAguideR",
                dependencies = FALSE,
                upgrade = "never",
                force = TRUE,
                quiet = FALSE
            )

        },
        error = function(e) {

            cat("\n")
            cat("============================================================\n")
            cat("NAGUIDER INSTALLATION FAILED\n")
            cat("============================================================\n")

            cat(
                "Error:",
                conditionMessage(e),
                "\n"
            )

            stop(
                "NAguideR installation failed.",
                call. = FALSE
            )
        }
    )
}


if (!requireNamespace("NAguideR", quietly = TRUE)) {

    stop(
        paste(
            "NAguideR installation completed",
            "but NAguideR cannot be loaded."
        ),
        call. = FALSE
    )
}


cat(
    "NAguideR:",
    as.character(packageVersion("NAguideR")),
    "OK\n"
)


# ============================================================
# Final validation
# ============================================================

cat("\n")
cat("============================================================\n")
cat("FINAL PACKAGE VALIDATION\n")
cat("============================================================\n")


required_packages <- c(

    # Main package
    "NAguideR",

    # Core dependencies
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

    # Special imputation packages
    "GMSimpute",
    "DreamAI",

    # DreamAI dependencies
    "randomForest",
    "glmnet",
    "foreach",
    "itertools",
    "iterators",
    "Rcpp",

    # NAguideR UI dependencies
    "shiny",
    "shinyBS",
    "shinyjs",
    "shinyWidgets",
    "DT",
    "gdata",
    "ggplot2",
    "ggsci",
    "openxlsx",
    "data.table",
    "tidyverse",
    "ggExtra",
    "cowplot",
    "Amelia"
)


failed_packages <- character()


cat("\n")

for (pkg in required_packages) {

    installed <- requireNamespace(
        pkg,
        quietly = TRUE
    )

    if (installed) {

        version <- tryCatch(
            as.character(packageVersion(pkg)),
            error = function(e) "unknown"
        )

        cat(
            sprintf(
                "%-20s %-10s %s\n",
                pkg,
                "OK",
                version
            )
        )

    } else {

        cat(
            sprintf(
                "%-20s %-10s\n",
                pkg,
                "MISSING"
            )
        )

        failed_packages <- c(
            failed_packages,
            pkg
        )
    }
}


# ============================================================
# Fail the GitHub Action if anything is missing
# ============================================================

if (length(failed_packages) > 0) {

    cat("\n")
    cat("============================================================\n")
    cat("DEPENDENCY INSTALLATION FAILED\n")
    cat("============================================================\n")

    cat(
        "Missing packages:\n"
    )

    for (pkg in failed_packages) {
        cat("  -", pkg, "\n")
    }

    stop(
        paste(
            "Missing R packages:",
            paste(
                failed_packages,
                collapse = ", "
            )
        ),
        call. = FALSE
    )
}


# ============================================================
# Test that the important namespaces actually load
# ============================================================

cat("\n")
cat("============================================================\n")
cat("Testing important namespaces\n")
cat("============================================================\n")


namespace_test <- c(
    "NAguideR",
    "SeqKnn",
    "GMSimpute",
    "DreamAI",
    "missForest"
)


for (pkg in namespace_test) {

    cat(
        "Loading namespace:",
        pkg,
        "... "
    )

    tryCatch(
        {

            loadNamespace(pkg)

            cat("OK\n")

        },
        error = function(e) {

            cat("FAILED\n")

            cat(
                "Error:",
                conditionMessage(e),
                "\n"
            )

            stop(
                paste(
                    "Namespace test failed:",
                    pkg
                ),
                call. = FALSE
            )
        }
    )
}


# ============================================================
# Session information
# ============================================================

cat("\n")
cat("============================================================\n")
cat("SESSION INFORMATION\n")
cat("============================================================\n")

print(sessionInfo())


# ============================================================
# Success
# ============================================================

cat("\n")
cat("============================================================\n")
cat("SUCCESS\n")
cat("============================================================\n")

cat(
    "All NAguideR dependencies installed successfully.\n"
)

cat(
    "NAguideR version:",
    as.character(packageVersion("NAguideR")),
    "\n"
)

cat(
    "DreamAI version:",
    as.character(packageVersion("DreamAI")),
    "\n"
)

cat(
    "GMSImpute version:",
    as.character(packageVersion("GMSimpute")),
    "\n"
)

cat(
    "SeqKnn version:",
    as.character(packageVersion("SeqKnn")),
    "\n"
)

cat("============================================================\n")