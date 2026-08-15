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

ENV GLYCOEXPLORER_HOSTED=true
ENV GLYCOEXPLORER_SAVE_MODE=browser

ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app


# ============================================================
# Linux + R dependencies
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
        cmake \
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
        libcairo2-dev \
        libxt-dev \
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
        libglpk-dev \
        libnlopt-dev \
    && rm -rf /var/lib/apt/lists/*


# ============================================================
# Verify base environment
# ============================================================

RUN echo "========================================" && \
    echo "Python version:" && \
    python --version && \
    echo "========================================" && \
    echo "R version:" && \
    Rscript --version && \
    echo "========================================" && \
    echo "R library paths:" && \
    Rscript -e "print(.libPaths())" && \
    echo "========================================"


# ============================================================
# Python dependencies
#
# Copy requirements separately so this layer is cached unless
# requirements.txt changes.
# ============================================================

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade \
        pip \
        setuptools \
        wheel && \
    python -m pip install \
        -r /app/requirements.txt


# ============================================================
# Copy local R source packages BEFORE running R installer
#
# Required:
#
# r_packages/GMSimpute_0.0.1.1.tar.gz
# r_packages/SeqKnn_1.0.1.tar.gz
# ============================================================

COPY r_packages/ /app/r_packages/


# ============================================================
# Verify local R source packages
# ============================================================

RUN echo "========================================" && \
    echo "Checking local R packages" && \
    echo "========================================" && \
    ls -lah /app/r_packages/ && \
    test -f /app/r_packages/GMSimpute_0.0.1.1.tar.gz && \
    test -f /app/r_packages/SeqKnn_1.0.1.tar.gz && \
    echo "Required R package archives found."


# ============================================================
# Copy R dependency installer
# ============================================================

COPY install_naguider_r_dependencies.R /app/install_naguider_r_dependencies.R


# ============================================================
# Install NAguideR / R dependencies
#
# WORKDIR is already /app, therefore relative paths such as:
#
# r_packages/GMSimpute_0.0.1.1.tar.gz
#
# work correctly.
# ============================================================

RUN echo "========================================" && \
    echo "Installing R dependencies" && \
    echo "========================================" && \
    Rscript /app/install_naguider_r_dependencies.R


# ============================================================
# Validate important R packages
# ============================================================

RUN Rscript -e "\
pkgs <- c( \
'NAguideR', \
'SeqKnn', \
'GMSimpute', \
'DreamAI', \
'e1071', \
'impute', \
'pcaMethods', \
'norm', \
'imputeLCMD', \
'VIM', \
'rrcovNA', \
'mice', \
'missForest', \
'glmnet', \
'ggplot2', \
'reshape2', \
'randomForest', \
'shiny' \
); \
cat('\n========================================\n'); \
cat('R package validation\n'); \
cat('========================================\n'); \
failed <- character(); \
for (p in pkgs) { \
    ok <- requireNamespace(p, quietly = TRUE); \
    if (ok) { \
        cat(sprintf('%-20s OK [%s]\n', p, as.character(packageVersion(p)))); \
    } else { \
        cat(sprintf('%-20s MISSING\n', p)); \
        failed <- c(failed, p); \
    } \
}; \
if (length(failed) > 0) { \
    stop(paste('Missing R packages:', paste(failed, collapse=', '))); \
}; \
cat('========================================\n'); \
cat('All required R packages available.\n'); \
cat('========================================\n');"


# ============================================================
# Copy complete application
#
# This is deliberately AFTER dependency installation.
#
# Application code changes will therefore not force all R and
# Python packages to reinstall on every Docker build.
# ============================================================

COPY . /app


# ============================================================
# Verify important application files
# ============================================================

RUN echo "========================================" && \
    echo "Checking application files" && \
    echo "========================================" && \
    test -f /app/app.py && \
    test -f /app/naguider_impute.R && \
    echo "app.py: FOUND" && \
    echo "naguider_impute.R: FOUND"


# ============================================================
# Final environment check
# ============================================================

RUN echo "========================================" && \
    echo "Final Docker environment" && \
    echo "========================================" && \
    echo "Python:" && \
    python --version && \
    echo "" && \
    echo "R:" && \
    Rscript --version && \
    echo "" && \
    echo "Streamlit:" && \
    streamlit version && \
    echo "" && \
    echo "Application directory:" && \
    ls -lah /app && \
    echo "========================================"


# ============================================================
# Streamlit port
# ============================================================

EXPOSE 8501


# ============================================================
# Health check
# ============================================================

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1


# ============================================================
# Start application
#
# IMPORTANT:
# Keep JSON-form CMD on ONE Dockerfile instruction.
# ============================================================

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--server.maxUploadSize=1024", "--browser.gatherUsageStats=false"]