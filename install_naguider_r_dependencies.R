# ============================================================
# install_naguider_r_dependencies.R
#
# Complete R dependency installer for:
#   Proteomics-data-explorer-workbench
#   NAguideR
#   GMSImpute
#   SeqKnn
#   DreamAI
#
# Expected files:
#
#   r_packages/GMSimpute_0.0.1.1.tar.gz
#   r_packages/SeqKnn_1.0.1.tar.gz
#
# Tested/designed for:
#   Windows GitHub Actions
#   R 4.5.x
# ============================================================


# ============================================================
# Global configuration
# ============================================================

options(
    repos = c(
        CRAN = "https://cloud.r-project.org"
    ),
    timeout = 1800,
    Ncpus = 2
)

cat("\n")
cat("============================================================\n")
cat(" NAguideR R Dependency Installation\n")
cat("============================================================\n")

cat("\nR version:\n")
cat(R.version.string, "\n")

cat("\nPlatform:\n")
cat(R.version$platform, "\n")

cat("\nR_HOME:\n")
cat(R.home(), "\n")

cat("\nR library paths:\n")
print(.libPaths())

cat("\nCRAN repository:\n")
print(getOption("repos"))


# ============================================================
# Helper: package installed?
# ============================================================

package_ok <- function(pkg) {

    requireNamespace(
        pkg,
        quietly = TRUE
    )
}


# ============================================================
# Helper: package version
# ============================================================

package_version_safe <- function(pkg) {

    tryCatch(
        as.character(packageVersion(pkg)),
        error = function(e) "unknown"
    )
}


# ============================================================
# Helper: install CRAN package
# ============================================================

install_cran <- function(pkg) {

    if (package_ok(pkg)) {

        cat(
            sprintf(
                "%-22s already installed [%s]\n",
                pkg,
                package_version_safe(pkg)
            )
        )

        return(invisible(TRUE))
    }

    cat("\n")
    cat("------------------------------------------------------------\n")
    cat("Installing CRAN package:", pkg, "\n")
    cat("------------------------------------------------------------\n")

    result <- tryCatch(
        {

            install.packages(
                pkg,
                dependencies = c(
                    "Depends",
                    "Imports",
                    "LinkingTo"
                )
            )

            TRUE

        },
        error = function(e) {

            cat("\nERROR installing:", pkg, "\n")
            cat(conditionMessage(e), "\n")

            FALSE
        }
    )

    if (!result || !package_ok(pkg)) {

        stop(
            paste0(
                "CRAN package installation failed: ",
                pkg
            ),
            call. = FALSE
        )
    }

    cat(
        "Installed:",
        pkg,
        package_version_safe(pkg),
        "\n"
    )

    invisible(TRUE)
}


# ============================================================
# Bootstrap BiocManager and remotes
# ============================================================

cat("\n")
cat("============================================================\n")
cat(" Installing package managers\n")
cat("============================================================\n")

install_cran("remotes")
install_cran("BiocManager")


# ============================================================
# Helper: install Bioconductor package
# ============================================================

install_bioc <- function(pkg) {

    if (package_ok(pkg)) {

        cat(
            sprintf(
                "%-22s already installed [%s]\n",
                pkg,
                package_version_safe(pkg)
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

            cat("\nBioconductor installation error\n")
            cat("Package:", pkg, "\n")
            cat("Error:", conditionMessage(e), "\n")

            stop(
                paste(
                    "Failed to install Bioconductor package:",
                    pkg
                ),
                call. = FALSE
            )
        }
    )

    if (!package_ok(pkg)) {

        stop(
            paste(
                "Bioconductor package is still unavailable:",
                pkg
            ),
            call. = FALSE
        )
    }

    cat(
        "Installed:",
        pkg,
        package_version_safe(pkg),
        "\n"
    )

    invisible(TRUE)
}


# ============================================================
# Verify local package source files
# ============================================================

cat("\n")
cat("============================================================\n")
cat(" Checking local R source packages\n")
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


for (f in required_files) {

    if (!file.exists(f)) {

        stop(
            paste(
                "Required R package archive is missing:",
                f
            ),
            call. = FALSE
        )
    }

    cat(
        "Found:",
        normalizePath(
            f,
            winslash = "/",
            mustWork = TRUE
        ),
        "\n"
    )
}


# ============================================================
# Bioconductor packages
# ============================================================

cat("\n")
cat("============================================================\n")
cat(" Installing Bioconductor dependencies\n")
cat("============================================================\n")


bioc_packages <- c(
    "impute",
    "pcaMethods"
)


for (pkg in bioc_packages) {

    install_bioc(pkg)
}


# ============================================================
# CRAN packages
# ============================================================

cat("\n")
cat("============================================================\n")
cat(" Installing CRAN dependencies\n")
cat("============================================================\n")


cran_packages <- c(

    # --------------------------------------------------------
    # NAguideR / imputation
    # --------------------------------------------------------

    "e1071",
    "norm",
    "imputeLCMD",
    "VIM",
    "rrcovNA",
    "mice",
    "missForest",

    # --------------------------------------------------------
    # GMSImpute
    # IMPORTANT: reshape2 is required
    # --------------------------------------------------------

    "glmnet",
    "ggplot2",
    "reshape2",

    # --------------------------------------------------------
    # DreamAI
    # --------------------------------------------------------

    "randomForest",
    "foreach",
    "itertools",
    "iterators",
    "Rcpp",

    # --------------------------------------------------------
    # NAguideR / Shiny UI
    # --------------------------------------------------------

    "shiny",
    "shinyBS",
    "shinyjs",
    "shinyWidgets",
    "DT",
    "gdata",
    "ggsci",
    "openxlsx",
    "data.table",
    "tidyverse",
    "ggExtra",
    "cowplot",
    "Amelia"
)


for (pkg in unique(cran_packages)) {

    install_cran(pkg)
}


# ============================================================
# Verify GMSImpute dependencies BEFORE installation
# ============================================================

cat("\n")
cat("============================================================\n")
cat(" Checking GMSImpute dependencies\n")
cat("============================================================\n")


gms_dependencies <- c(
    "glmnet",
    "ggplot2",
    "reshape2"
)


for (pkg in gms_dependencies) {

    if (!package_ok(pkg)) {

        stop(
            paste(
                "Missing required GMSImpute dependency:",
                pkg
            ),
            call. = FALSE
        )
    }

    cat(
        sprintf(
            "%-15s OK [%s]\n",
            pkg,
            package_version_safe(pkg)
        )
    )
}


# ============================================================
# Helper: install local source package
# ============================================================

install_local_source <- function(pkg, archive) {

    if (package_ok(pkg)) {

        cat(
            sprintf(
                "%-22s already installed [%s]\n",
                pkg,
                package_version_safe(pkg)
            )
        )

        return(invisible(TRUE))
    }

    if (!file.exists(archive)) {

        stop(
            paste(
                "Source archive not found:",
                archive
            ),
            call. = FALSE
        )
    }

    archive <- normalizePath(
        archive,
        winslash = "/",
        mustWork = TRUE
    )

    cat("\n")
    cat("------------------------------------------------------------\n")
    cat("Installing local package:", pkg, "\n")
    cat("Archive:", archive, "\n")
    cat("------------------------------------------------------------\n")

    # --------------------------------------------------------
    # install.packages() can sometimes emit warnings instead
    # of a clean error. Capture those warnings so GitHub logs
    # contain the useful information.
    # --------------------------------------------------------

    warnings_seen <- character()

    result <- tryCatch(

        withCallingHandlers(
            {

                install.packages(
                    archive,
                    repos = NULL,
                    type = "source"
                )

                TRUE
            },

            warning = function(w) {

                warning_text <- conditionMessage(w)

                warnings_seen <<- c(
                    warnings_seen,
                    warning_text
                )

                cat(
                    "\nINSTALL WARNING:",
                    warning_text,
                    "\n"
                )

                invokeRestart("muffleWarning")
            }
        ),

        error = function(e) {

            cat("\n")
            cat("============================================================\n")
            cat("LOCAL PACKAGE INSTALL ERROR\n")
            cat("============================================================\n")

            cat("Package:", pkg, "\n")
            cat("Archive:", archive, "\n")
            cat("Error:", conditionMessage(e), "\n")

            FALSE
        }
    )


    # --------------------------------------------------------
    # Check whether R actually registered package
    # --------------------------------------------------------

    installed_names <- rownames(
        installed.packages()
    )

    if (!(pkg %in% installed_names)) {

        cat("\n")
        cat("============================================================\n")
        cat("PACKAGE WAS NOT INSTALLED\n")
        cat("============================================================\n")

        cat(
            "Package:",
            pkg,
            "\n"
        )

        if (length(warnings_seen) > 0) {

            cat("\nWarnings generated during installation:\n")

            for (w in unique(warnings_seen)) {
                cat(" -", w, "\n")
            }
        }

        stop(
            paste(
                "Installation failed for local package:",
                pkg
            ),
            call. = FALSE
        )
    }


    # --------------------------------------------------------
    # Test namespace explicitly
    # --------------------------------------------------------

    cat(
        "\nTesting namespace for",
        pkg,
        "...\n"
    )

    namespace_result <- tryCatch(
        {

            loadNamespace(pkg)

            TRUE

        },
        error = function(e) {

            cat("\n")
            cat("============================================================\n")
            cat("PACKAGE NAMESPACE LOAD FAILED\n")
            cat("============================================================\n")

            cat("Package:", pkg, "\n")
            cat("Version:", package_version_safe(pkg), "\n")
            cat("Error:", conditionMessage(e), "\n")

            cat("\nLibrary paths:\n")
            print(.libPaths())

            cat("\nPackage installation location:\n")

            pkg_location <- tryCatch(
                find.package(pkg),
                error = function(e) NA_character_
            )

            print(pkg_location)

            FALSE
        }
    )


    if (!namespace_result) {

        stop(
            paste(
                "Package installed but namespace cannot be loaded:",
                pkg
            ),
            call. = FALSE
        )
    }


    cat(
        "Successfully installed and loaded:",
        pkg,
        package_version_safe(pkg),
        "\n"
    )

    invisible(TRUE)
}


# ============================================================
# Install SeqKnn
# ============================================================

cat("\n")
cat("============================================================\n")
cat(" Installing SeqKnn\n")
cat("============================================================\n")


install_local_source(
    pkg = "SeqKnn",
    archive = seqknn_file
)


# ============================================================
# Install GMSImpute
# ============================================================

cat("\n")
cat("============================================================\n")
cat(" Installing GMSImpute\n")
cat("============================================================\n")


install_local_source(
    pkg = "GMSimpute",
    archive = gmsimpute_file
)


# ============================================================
# EXTRA GMSImpute validation
# ============================================================

cat("\n")
cat("============================================================\n")
cat(" Validating GMSImpute\n")
cat("============================================================\n")


if (!package_ok("GMSimpute")) {

    cat("\nGMSimpute dependencies:\n")

    for (pkg in gms_dependencies) {

        cat(
            sprintf(
                "%-15s %s\n",
                pkg,
                if (package_ok(pkg)) {
                    paste(
                        "OK",
                        package_version_safe(pkg)
                    )
                } else {
                    "MISSING"
                }
            )
        )
    }

    stop(
        "GMSimpute namespace cannot be loaded.",
        call. = FALSE
    )
}


cat(
    "GMSImpute:",
    package_version_safe("GMSimpute"),
    "OK\n"
)


# ============================================================
# DreamAI
# ============================================================

cat("\n")
cat("============================================================\n")
cat(" Installing DreamAI\n")
cat("============================================================\n")


if (package_ok("DreamAI")) {

    cat(
        "DreamAI already installed:",
        package_version_safe("DreamAI"),
        "\n"
    )

} else {

    cat("Installing DreamAI using git clone...\n")

    dreamai_tmp <- tempfile("DreamAI_")

    clone_status <- system2(
        "git",
        c(
            "clone",
            "--depth", "1",
            "https://github.com/WangLab-MSSM/DreamAI.git",
            dreamai_tmp
        )
    )

    if (clone_status != 0) {
        stop(
            "Failed to clone DreamAI repository.",
            call. = FALSE
        )
    }

    dreamai_pkg <- file.path(
        dreamai_tmp,
        "Code"
    )

    if (!file.exists(file.path(dreamai_pkg, "DESCRIPTION"))) {
        stop(
            paste(
                "DreamAI package DESCRIPTION not found:",
                dreamai_pkg
            ),
            call. = FALSE
        )
    }

    cat(
        "DreamAI package source:",
        dreamai_pkg,
        "\n"
    )

    tryCatch(
        {

            install.packages(
                dreamai_pkg,
                repos = NULL,
                type = "source"
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

            stop(
                "DreamAI installation failed.",
                call. = FALSE
            )
        }
    )

    unlink(
        dreamai_tmp,
        recursive = TRUE,
        force = TRUE
    )
}


if (!package_ok("DreamAI")) {

    stop(
        paste(
            "DreamAI installation finished but",
            "the DreamAI namespace cannot be loaded."
        ),
        call. = FALSE
    )
}


cat(
    "DreamAI:",
    package_version_safe("DreamAI"),
    "OK\n"
)


# ============================================================
# Install NAguideR
# ============================================================

cat("\n")
cat("============================================================\n")
cat(" Installing NAguideR\n")
cat("============================================================\n")


if (package_ok("NAguideR")) {

    cat(
        "NAguideR already installed:",
        package_version_safe("NAguideR"),
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


if (!package_ok("NAguideR")) {

    stop(
        paste(
            "NAguideR installation finished but",
            "the NAguideR namespace cannot be loaded."
        ),
        call. = FALSE
    )
}


cat(
    "NAguideR:",
    package_version_safe("NAguideR"),
    "OK\n"
)


# ============================================================
# Final package validation
# ============================================================

cat("\n")
cat("============================================================\n")
cat(" FINAL PACKAGE VALIDATION\n")
cat("============================================================\n")


required_packages <- c(

    # Main application
    "NAguideR",

    # NAguideR methods
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

    # Special packages
    "GMSimpute",
    "DreamAI",

    # GMSImpute
    "glmnet",
    "ggplot2",
    "reshape2",

    # DreamAI
    "randomForest",
    "foreach",
    "itertools",
    "iterators",
    "Rcpp",

    # UI
    "shiny",
    "shinyBS",
    "shinyjs",
    "shinyWidgets",
    "DT",
    "gdata",
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

    ok <- package_ok(pkg)

    if (ok) {

        cat(
            sprintf(
                "%-20s %-8s %s\n",
                pkg,
                "OK",
                package_version_safe(pkg)
            )
        )

    } else {

        cat(
            sprintf(
                "%-20s %-8s\n",
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
# Stop workflow if anything is missing
# ============================================================

if (length(failed_packages) > 0) {

    cat("\n")
    cat("============================================================\n")
    cat(" DEPENDENCY INSTALLATION FAILED\n")
    cat("============================================================\n")

    cat("\nMissing packages:\n")

    for (pkg in failed_packages) {

        cat(
            "  -",
            pkg,
            "\n"
        )
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
# Test important namespaces
# ============================================================

cat("\n")
cat("============================================================\n")
cat(" TESTING IMPORTANT NAMESPACES\n")
cat("============================================================\n")


namespace_packages <- c(
    "NAguideR",
    "SeqKnn",
    "GMSimpute",
    "DreamAI",
    "missForest"
)


for (pkg in namespace_packages) {

    cat(
        sprintf(
            "%-20s ",
            pkg
        )
    )

    result <- tryCatch(
        {

            loadNamespace(pkg)

            TRUE

        },
        error = function(e) {

            cat("\n")

            cat(
                "Namespace error:",
                conditionMessage(e),
                "\n"
            )

            FALSE
        }
    )

    if (!result) {

        stop(
            paste(
                "Namespace validation failed:",
                pkg
            ),
            call. = FALSE
        )
    }

    cat("OK\n")
}


# ============================================================
# Session information
# ============================================================

cat("\n")
cat("============================================================\n")
cat(" SESSION INFORMATION\n")
cat("============================================================\n")

print(
    sessionInfo()
)


# ============================================================
# Final success
# ============================================================

cat("\n")
cat("============================================================\n")
cat(" SUCCESS\n")
cat("============================================================\n")

cat(
    "All NAguideR dependencies installed successfully.\n\n"
)

cat(
    "NAguideR   :",
    package_version_safe("NAguideR"),
    "\n"
)

cat(
    "GMSImpute :",
    package_version_safe("GMSimpute"),
    "\n"
)

cat(
    "SeqKnn    :",
    package_version_safe("SeqKnn"),
    "\n"
)

cat(
    "DreamAI   :",
    package_version_safe("DreamAI"),
    "\n"
)

cat("\n")
cat("============================================================\n")