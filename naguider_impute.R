#!/usr/bin/env Rscript

# NAguideR-compatible single-method imputation bridge for the Python Streamlit LFQ app.
# The method implementations mirror the NAguideR app.R nafunctions() logic.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript naguider_impute.R <input.tsv> <output.tsv> <method> [pi_width] [pi_downshift] [rf_ntree]")
}

input_file <- args[[1]]
output_file <- args[[2]]
method <- tolower(args[[3]])
pi_width <- if (length(args) >= 4) as.numeric(args[[4]]) else 0.3
pi_downshift <- if (length(args) >= 5) as.numeric(args[[5]]) else 1.8
rf_ntree <- if (length(args) >= 6) as.integer(args[[6]]) else 20L

need_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(sprintf("Required R package '%s' is not installed for method '%s'.", pkg, method))
  }
}

x <- read.delim(input_file, row.names = 1, check.names = FALSE, stringsAsFactors = FALSE)
x[] <- lapply(x, function(z) as.numeric(as.character(z)))
df1 <- as.data.frame(x, check.names = FALSE)

if (!anyNA(df1)) {
  write.table(df1, output_file, sep = "\t", quote = FALSE, col.names = NA)
  quit(save = "no", status = 0)
}

set.seed(1234)

if (method == "zero") {
  df <- df1
  df[is.na(df)] <- 0

} else if (method == "minimum") {
  df <- df1
  minval <- suppressWarnings(min(as.matrix(df1), na.rm = TRUE))
  if (!is.finite(minval)) stop("Minimum imputation failed: no finite observed values are available.")
  df[is.na(df)] <- minval

} else if (method == "colmedian") {
  need_pkg("e1071")
  df <- e1071::impute(df1, what = "median")

} else if (method == "rowmedian") {
  need_pkg("e1071")
  df <- t(e1071::impute(t(df1), what = "median"))

} else if (method == "knnmethod") {
  need_pkg("impute")
  df <- impute::impute.knn(as.matrix(df1), k = 10, rowmax = 1, colmax = 1)$data

} else if (method == "seqknn") {
  need_pkg("SeqKnn")
  df <- SeqKnn::SeqKNN(df1, k = 10)

} else if (method == "bpca") {
  need_pkg("pcaMethods")
  npcs <- max(1, min(ncol(df1) - 1, nrow(df1) - 1))
  fit <- pcaMethods::pca(as.matrix(df1), nPcs = npcs, method = "bpca", maxSteps = 100)
  df <- pcaMethods::completeObs(fit)

} else if (method == "svdmethod") {
  need_pkg("pcaMethods")
  npcs <- max(1, min(ncol(df1) - 1, nrow(df1) - 1))
  fit <- pcaMethods::pca(as.matrix(df1), nPcs = npcs, method = "svdImpute")
  df <- pcaMethods::completeObs(fit)

} else if (method == "lls") {
  need_pkg("pcaMethods")
  fit <- pcaMethods::llsImpute(t(df1), k = 10)
  df <- t(pcaMethods::completeObs(fit))

} else if (method == "mle") {
  need_pkg("norm")
  xxm <- as.matrix(df1)
  ss0 <- norm::prelim.norm(xxm)
  thx <- norm::em.norm(ss0, showits = FALSE)
  norm::rngseed(123)
  df <- norm::imp.norm(ss0, thx, xxm)

} else if (method == "qrilc") {
  need_pkg("imputeLCMD")
  xxm <- t(df1)
  df <- t(imputeLCMD::impute.QRILC(xxm, tune.sigma = 1)[[1]])

} else if (method == "mindet") {
  need_pkg("imputeLCMD")
  df <- imputeLCMD::impute.MinDet(as.matrix(df1), q = 0.01)

} else if (method == "minprob") {
  need_pkg("imputeLCMD")
  df <- imputeLCMD::impute.MinProb(as.matrix(df1), q = 0.01, tune.sigma = 1)

} else if (method == "irm") {
  need_pkg("VIM")
  df <- VIM::irmi(df1, trace = FALSE, imp_var = FALSE)
  rownames(df) <- rownames(df1)

} else if (method == "impseq") {
  need_pkg("rrcovNA")
  df <- rrcovNA::impSeq(df1)

} else if (method == "impseqrob") {
  need_pkg("rrcovNA")
  df <- rrcovNA::impSeqRob(df1, alpha = 0.9)$x

} else if (method == "mice-norm") {
  need_pkg("mice")
  minum <- 5
  mids <- mice::mice(df1, m = minum, seed = 1234, method = "norm", printFlag = FALSE)
  acc <- 0
  for (i in seq_len(minum)) acc <- mice::complete(mids, action = i) + acc
  df <- acc / minum
  rownames(df) <- rownames(df1)

} else if (method == "mice-cart") {
  need_pkg("mice")
  minum <- 5
  mids <- mice::mice(df1, m = minum, seed = 1234, method = "cart", printFlag = FALSE)
  acc <- 0
  for (i in seq_len(minum)) acc <- mice::complete(mids, action = i) + acc
  df <- acc / minum
  rownames(df) <- rownames(df1)

} else if (method == "trknn") {
  need_pkg("NAguideR")
  tr_file <- file.path(system.file("NAguideRapp", package = "NAguideR"), "Trunc_KNN", "Imput_funcs.r")
  if (!file.exists(tr_file)) stop("NAguideR Trunc_KNN helper file was not found in the installed package.")
  source(tr_file, local = TRUE)
  df <- imputeKNN(as.matrix(df1), k = 10, distance = "truncation", perc = 0)

} else if (method == "rf") {
  need_pkg("missForest")
  mtry_val <- max(1L, floor(nrow(df1)^(1/3)))
  fit <- missForest::missForest(t(df1), maxiter = 10, ntree = rf_ntree, mtry = mtry_val, verbose = FALSE)
  df <- t(fit$ximp)

} else if (method == "pi") {
  df <- df1
  for (i in seq_len(ncol(df1))) {
    temp <- df1[[i]]
    obs <- temp[!is.na(temp)]
    if (sum(is.na(temp)) > 0) {
      if (length(obs) < 2 || !is.finite(sd(obs))) stop(sprintf("PI requires at least two observed values in column '%s'.", colnames(df1)[i]))
      temp_sd <- pi_width * sd(obs)
      temp_mean <- mean(obs) - pi_downshift * sd(obs)
      temp[is.na(temp)] <- rnorm(sum(is.na(temp)), mean = temp_mean, sd = temp_sd)
      df[[i]] <- temp
    }
  }

} else if (method == "grr") {
  need_pkg("DreamAI")
  df <- DreamAI::impute.RegImpute(
    data = as.matrix(df1), fillmethod = "row_mean",
    maxiter_RegImpute = 10, conv_nrmse = 1e-03
  )

} else if (method == "gms") {
  need_pkg("GMSimpute")
  df <- GMSimpute::GMS.Lasso(df1, nfolds = 3, log.scale = FALSE, TS.Lasso = TRUE)

} else {
  stop(sprintf("Unsupported NAguideR method: %s", method))
}

df <- as.data.frame(df, check.names = FALSE)
rownames(df) <- rownames(df1)
colnames(df) <- colnames(df1)

if (any(!is.finite(as.matrix(df)))) {
  stop(sprintf("Method '%s' returned non-finite values.", method))
}
if (anyNA(df)) {
  stop(sprintf("Method '%s' finished but missing values remain.", method))
}

write.table(df, output_file, sep = "\t", quote = FALSE, col.names = NA)
