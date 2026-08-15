# ============================================================
# Proteomics Data Explorer Workbench
#
# Python 3.12
# R 4.5.x
# Streamlit
# NAguideR
# GMSImpute
# SeqKnn
# DreamAI
# ============================================================

FROM python:3.12-slim-trixie


# ============================================================
# Environment
# ============================================================

ENV DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# Application behavior
ENV GLYCOEXPLORER_HOSTED=true
ENV GLYCOEXPLORER_SAVE_MODE=browser

# Streamlit
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app


# ============================================================
# Linux + R system dependencies
# ============================================================
#
# r-base / r-base-dev:
#     R runtime + headers
#
# build-essential/gfortran:
#     required for compiled R packages
#
# libcurl/libssl/libxml2:
#     required by many CRAN packages
#
# font/image libraries:
#     required by ggplot2/ragg/systemfonts etc.
#
# BLAS/LAPACK:
#     required by glmnet and numerical packages
#
# ============================================================

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        r-base \
        r-base-dev \
        build-essential \
        gcc \
        g++ \
        gfortran \
        make \
        git \
        curl \
        wget \
        ca-certificates \
        pkg-config \
        libcurl4-openssl-dev \
        libssl-dev \
        libxml2-dev \
        libgit2-dev \
        libfontconfig1-dev \
        libfreetype6-dev \
        libpng-dev \
        libjpeg-dev \
        libtiff-dev \
        libharfbuzz-dev \
        libfribidi-dev \
        libicu-dev \
        libblas-dev \
        liblapack-dev \
        libbz2-dev \
        liblzma-dev \
        libpcre2-dev \
        libreadline-dev \
        zlib1g-dev \
        libzstd-dev \
        libncurses-dev \
    && rm -rf /var/lib/apt/lists/*


# ============================================================
# Verify Python and R
# ============================================================

RUN echo "========================================" && \
    echo "Python:" && \
    python --version && \
    echo "========================================" && \
    echo "R:" && \
    Rscript --version && \
    echo "========================================" && \
    echo "R library paths:" && \
    Rscript -e "print(.libPaths())" && \
    echo "========================================"


# ============================================================
# Python dependencies
#
# Copy requirements separately so Docker can cache this layer.
# ============================================================

COPY requirements.txt /app/requirements.txt


RUN python -m pip install --upgrade \
        pip \
        setuptools \
        wheel \
    && \
    python -m pip install \
        -r /app/requirements.txt


# ============================================================
# R source packages
#
# IMPORTANT:
#
# These MUST be copied BEFORE running
# install_naguider_r_dependencies.R.
#
# The installer expects:
#
# /app/r_packages/GMSimpute_0.0.1.1.tar.gz
# /app/r_packages/SeqKnn_1.0.1.tar.gz
#
# ============================================================

COPY r_packages/ /app/r_packages/


# ============================================================
# Verify local R archives
# ============================================================

RUN echo "========================================" && \
    echo "Checking R source packages" && \
    echo "========================================" && \
    ls -lah /app/r_packages/ && \
    test -f /app/r_packages/GMSimpute_0.0.1.1.tar.gz && \
    test -f /app/r_packages/SeqKnn_1.0.1.tar.gz && \
    echo "Required R source archives found."


# ============================================================
# R / NAguideR dependency installer
# ============================================================

COPY install_naguider_r_dependencies.R \
     /app/install_naguider_r_dependencies.R


# ============================================================
# Install R packages
#
# Working directory is /app, so relative paths such as:
#
# r_packages/GMSimpute_0.0.1.1.tar.gz
#
# work correctly.
# ============================================================

RUN Rscript /app/install_naguider_r_dependencies.R


# ============================================================
# Validate important R packages immediately
# ============================================================

RUN Rscript -e "\
pkgs <- c( \
    'NAguideR', \
    'SeqKnn', \
    'GMSimpute', \
    'DreamAI', \
    'impute', \
    'pcaMethods', \
    'missForest', \
    'glmnet', \
    'ggplot2', \
    'reshape2', \
    'shiny' \
); \
cat('\n========================================\n'); \
cat('R package validation\n'); \
cat('========================================\n'); \
failed <- character(); \
for (p in pkgs) { \
    ok <- requireNamespace(p, quietly=TRUE); \
    if (ok) { \
        cat(sprintf('%-20s OK [%s]\n', \
            p, \
            as.character(packageVersion(p)))); \
    } else { \
        cat(sprintf('%-20s MISSING\n', p)); \
        failed <- c(failed, p); \
    } \
}; \
if (length(failed) > 0) { \
    stop(paste('Missing R packages:', \
        paste(failed, collapse=', '))); \
}; \
cat('========================================\n'); \
cat('All required R packages available.\n'); \
cat('========================================\n');"


# ============================================================
# Copy application
#
# Do this AFTER dependency installation.
#
# Changing app source code then does not force Docker to
# reinstall all Python/R dependencies every build.
# ============================================================

COPY . /app


# ============================================================
# Final application checks
# ============================================================

RUN echo "========================================" && \
    echo "Final environment" && \
    echo "========================================" && \
    echo "Python:" && \
    python --version && \
    echo && \
    echo "R:" && \
    Rscript --version && \
    echo && \
    echo "Streamlit:" && \
    streamlit version && \
    echo && \
    echo "Application files:" && \
    ls -lah /app && \
    echo "========================================"


# ============================================================
# Streamlit port
# ============================================================

EXPOSE 8501


# ============================================================
# Health check
# ============================================================

HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=60s \
    --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1


# ============================================================
# Start Streamlit
# ============================================================

CMD [
    "streamlit",
    "run",
    "app.py",
    "--server.address=0.0.0.0",
    "--server.port=8501",
    "--server.headless=true",
    "--server.maxUploadSize=1024",
    "--browser.gatherUsageStats=false"
]