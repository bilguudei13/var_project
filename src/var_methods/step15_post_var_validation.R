
# step15_post_var_validation.R
#
# GARCH-Copula VaR Project — Post-VaR Validation
#
#   Layer A — Pooled in-sample re-validation (consolidates Step 7 results)
#   Layer B — Per rolling window re-validation (out-of-sample stability)
#   Layer C — Exception-based residual diagnostics
#   Layer D — Variance targeting diagnostic
#
# REQUIRES: source("steps_9_to_12_copula_var.R") with the updated rolling_var()
#   that attaches fits, window, and step as attributes to results$var_rolling.
#   Uses from global workspace: results, factor_names, factors_mat,
#   FIG_DIR, TBL_DIR, save_png()
#


# Safety checks
if (!exists("results") || is.null(results$var_rolling))
  stop("Run source('steps_9_to_12_copula_var.R') before this file.")
if (is.null(attr(results$var_rolling, "fits")))
  stop(paste("attr(results$var_rolling, 'fits') is NULL.",
             "Re-run steps_9_to_12_copula_var.R with the updated rolling_var()."))
if (!exists("FIG_DIR"))
  stop("FIG_DIR not found — source steps_3_to_8_marginal_garch.R first.")

rolling_fits <- attr(results$var_rolling, "fits")
n_windows    <- length(rolling_fits)
window       <- attr(results$var_rolling, "window")
step_size    <- attr(results$var_rolling, "step")
T_total      <- nrow(factors_mat)
K            <- length(factor_names)

if (n_windows != nrow(results$var_rolling))
  warning(sprintf(
    "n_windows (%d) != nrow(var_rolling) (%d) — some iterations were skipped.",
    n_windows, nrow(results$var_rolling)))

results$post_validation <- list()

# Per-factor GARCH order helpers (used in Layer B to mirror Step 7 spec)
get_go   <- function(fct) {
  g <- results$garch_fit[[fct]]
  c(g@model$modelinc["alpha"], g@model$modelinc["beta"])
}
get_arma <- function(fct) results$arma_order[[fct]][c(1, 3)]



# validate_one_fit() — re-runs all 6 criteria on a single ugarchfit object.
# Exact mirror of Step 7 so Layer B results are directly comparable to Layer A.

validate_one_fit <- function(fit, y_window, p_ar, q_ma, go) {
  Zh <- as.numeric(residuals(fit, standardize = TRUE))

  wlb <- function(x, fitdf) {
    res <- suppressWarnings(tryCatch(
      Weighted.Box.test(x, lag = 10, type = "Ljung-Box", fitdf = fitdf),
      error = function(e) NULL))
    if (is.null(res) || is.na(res$p.value))
      Box.test(x, lag = 10, type = "Ljung-Box", fitdf = fitdf)
    else res
  }

  c1      <- wlb(Zh,    fitdf = p_ar + q_ma)
  c2a     <- wlb(Zh^2,  fitdf = go[1] + go[2])
  c2b     <- tryCatch(ArchTest(Zh, lags = 10),
                      error = function(e) list(p.value = 1))
  gof_tbl <- tryCatch(gof(fit, groups = c(20, 30, 40, 50)),
                      error = function(e) NULL)
  uv      <- tryCatch(as.numeric(uncvariance(fit)), error = function(e) NA_real_)
  ev      <- var(y_window, na.rm = TRUE)
  ratio   <- if (!is.na(uv) && ev > 0) uv / ev else NA_real_
  sb      <- tryCatch(signbias(fit),
                      error = function(e) list(prob = c(1, 1, 1)))
  ny      <- tryCatch(nyblom(fit),
                      error = function(e) list(JointStat = 0, JointCritical = c(0, Inf)))

  c(C1 = isTRUE(c1$p.value  > 0.05),
    C2 = isTRUE(c2a$p.value > 0.05 && c2b$p.value > 0.05),
    C3 = if (is.null(gof_tbl)) NA_real_
         else as.numeric(mean(gof_tbl[, "p-value(g-1)"] > 0.05, na.rm = TRUE) >= 0.5),
    C4 = isTRUE(!is.na(ratio) && ratio > 0.75 && ratio < 1.25),
    C5 = isTRUE(all(sb$prob > 0.05)),
    C6 = isTRUE(ny$JointStat < ny$JointCritical[2]))
}



# Layer A — Pooled in-sample re-validation
#
# No refitting — consolidates results$garch_valid[[fct]] from Step 7.
# Useful as a single-glance summary of which factors passed each criterion
# on the full in-sample window, before checking rolling stability in Layer B.


poolA <- data.frame(
  Factor    = factor_names,
  C1_LB_Z   = sapply(factor_names, function(f) isTRUE(results$garch_valid[[f]]$Pass[1])),
  C2_LB_Z2  = sapply(factor_names, function(f) isTRUE(results$garch_valid[[f]]$Pass[2])),
  C3_GoF    = sapply(factor_names, function(f) isTRUE(results$garch_valid[[f]]$Pass[3])),
  C4_Var    = sapply(factor_names, function(f) isTRUE(results$garch_valid[[f]]$Pass[4])),
  C5_SignBs = sapply(factor_names, function(f) isTRUE(results$garch_valid[[f]]$Pass[5])),
  C6_Nyblom = sapply(factor_names, function(f) isTRUE(results$garch_valid[[f]]$Pass[6])),
  stringsAsFactors = FALSE)
poolA$AllPass <- poolA$C1_LB_Z & poolA$C2_LB_Z2 & poolA$C3_GoF &
                 poolA$C4_Var  & poolA$C5_SignBs & poolA$C6_Nyblom

write.csv(poolA, file.path(TBL_DIR, "step15_layerA_pooled.csv"), row.names = FALSE)
results$post_validation$layerA <- poolA

save_png("step15a_pooled_validation.png", quote({
  crit_cols <- c("C1_LB_Z","C2_LB_Z2","C3_GoF","C4_Var","C5_SignBs","C6_Nyblom","AllPass")
  mat <- t(as.matrix(poolA[, crit_cols]) * 1L)   # crit x factor
  par(mar = c(3, 9, 4, 2))
  image(seq_len(nrow(mat)), seq_len(ncol(mat)),
        mat, col = c("#d73027","#1a9850"),
        axes = FALSE, zlim = c(0, 1),
        main = "Layer A — In-sample validation (green = PASS)",
        xlab = "", ylab = "")
  axis(1, at = seq_len(nrow(mat)), labels = crit_cols, las = 2, cex.axis = 0.75)
  axis(2, at = seq_len(ncol(mat)), labels = poolA$Factor, las = 1, cex.axis = 0.85)
  for (ci in seq_len(nrow(mat)))
    for (fi in seq_len(ncol(mat)))
      text(ci, fi, if (isTRUE(mat[ci, fi] == 1)) "PASS" else "FAIL",
           col = "white", cex = 0.75, font = 2)
}), w = max(9, 1.4 * 7), h = max(4, 0.8 * K))

# Layer A output block
cat("\nLayer A — in-sample criteria pass/fail:\n")
print(poolA, row.names = FALSE)
cat("Layer A complete.\n")



# Layer B — Rolling-window re-validation
#
# Reads attr(results$var_rolling, "fits") and re-runs 6 criteria per window.
# Pass rates below ~70% for a criterion across windows indicate persistent
# structural misspecification, not just occasional bad luck.


if (is.null(window) || is.null(step_size)) {
  window    <- 500L
  step_size <- max(1L, floor((T_total - window) / max(n_windows - 1, 1)))
  cat(sprintf("WARNING: window/step attrs missing — inferring window=%d step=%d\n",
              window, step_size))
}
rows_out   <- seq(window, T_total - 1, by = step_size)
rows_out   <- rows_out[seq_len(min(length(rows_out), n_windows))]
crit_names <- c("C1","C2","C3","C4","C5","C6")

pass_arr <- array(NA_real_, dim = c(length(rows_out), K, 6),
                  dimnames = list(NULL, factor_names, crit_names))

cat(sprintf("Validating %d rolling windows x %d factors...\n",
            length(rows_out), K))

for (iter in seq_along(rows_out)) {
  fits_iter <- rolling_fits[[iter]]
  if (is.null(fits_iter)) next
  t       <- rows_out[iter]
  win_idx <- (t - window + 1):t

  for (fi in seq_len(K)) {
    fct <- factor_names[fi]
    fw  <- fits_iter[[fct]]
    if (is.null(fw)) next
    y_win <- as.numeric(factors_mat[win_idx, fct])
    go    <- get_go(fct)
    ord   <- get_arma(fct)
    res   <- tryCatch(
      validate_one_fit(fw, y_win, p_ar = ord[1], q_ma = ord[2], go = go),
      error = function(e) rep(NA_real_, 6))
    pass_arr[iter, fi, ] <- as.numeric(res)
  }
  if (iter %% 10 == 0)
    cat(sprintf("  iter %d/%d  (t=%d)\n", iter, length(rows_out), rows_out[iter]))
}

poolB <- data.frame(
  Factor = factor_names,
  C1_pct = sapply(seq_len(K), function(fi) round(100 * mean(pass_arr[, fi, "C1"], na.rm = TRUE), 1)),
  C2_pct = sapply(seq_len(K), function(fi) round(100 * mean(pass_arr[, fi, "C2"], na.rm = TRUE), 1)),
  C3_pct = sapply(seq_len(K), function(fi) round(100 * mean(pass_arr[, fi, "C3"], na.rm = TRUE), 1)),
  C4_pct = sapply(seq_len(K), function(fi) round(100 * mean(pass_arr[, fi, "C4"], na.rm = TRUE), 1)),
  C5_pct = sapply(seq_len(K), function(fi) round(100 * mean(pass_arr[, fi, "C5"], na.rm = TRUE), 1)),
  C6_pct = sapply(seq_len(K), function(fi) round(100 * mean(pass_arr[, fi, "C6"], na.rm = TRUE), 1)),
  stringsAsFactors = FALSE)

write.csv(poolB, file.path(TBL_DIR, "step15_layerB_rolling_passrates.csv"), row.names = FALSE)
results$post_validation$layerB <- list(pass_arr = pass_arr, poolB = poolB)

save_png("step15b_rolling_passrates.png", quote({
  pct_mat <- as.matrix(poolB[, paste0("C", 1:6, "_pct")])
  rownames(pct_mat) <- poolB$Factor
  colnames(pct_mat) <- paste0("C", 1:6)
  pal <- c("#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b")[seq_len(K)]
  par(mar = c(5, 5, 4, 9))
  mp <- barplot(t(pct_mat), beside = TRUE, col = pal,
                ylim = c(0, 115),
                names.arg = colnames(pct_mat),
                xlab = "Criterion", ylab = "Pass rate (%)",
                main = "Layer B — Rolling window pass rates by criterion")
  abline(h = 90, col = "darkred", lwd = 2, lty = 2)
  text(mean(mp), 93, "90% threshold", col = "darkred", cex = 0.8, adj = 0)
  legend("topright", legend = poolB$Factor, fill = pal, bty = "n", cex = 0.8,
         inset = c(-0.18, 0), xpd = TRUE)
}), w = max(10, 2 * K), h = 6)

save_png("step15c_rolling_pass_heatmap.png", quote({
  nc_p <- min(K, 3L)
  nr_p <- ceiling(K / nc_p)
  par(mfrow = c(nr_p, nc_p), mar = c(3, 5, 3, 1))
  for (fi in seq_len(K)) {
    fct    <- factor_names[fi]
    mat_fi <- pass_arr[, fi, ]   # n_windows x 6
    image(seq_len(nrow(mat_fi)), seq_len(6),
          mat_fi, col = c("#d73027","#1a9850"),
          axes = FALSE, zlim = c(0, 1),
          main = paste("Layer B —", fct),
          xlab = "Window", ylab = "")
    axis(1, at = pretty(seq_len(nrow(mat_fi))), cex.axis = 0.7)
    axis(2, at = 1:6, labels = crit_names, las = 1, cex.axis = 0.75)
  }
}), w = 5 * min(K, 3L), h = 5 * ceiling(K / 3L))

# Layer B output block
cat("\nLayer B — rolling window pass rates (%):\n")
print(poolB, row.names = FALSE)
cat("Layer B complete.\n")



# Layer C — Exception-based residual diagnostics
#
# Two tests on the rolling VaR exception series:
#   1. Wald-Wolfowitz runs test: are exceptions independent over time?
#      Clustered exceptions (low p) indicate regime persistence not captured.
#   2. Exception magnitude: how far do losses exceed VaR when they breach it?
#      Large mean excess suggests fat tails in the residual not in the model.


rv <- results$var_rolling

# Helper functions — inline cat() kept since they print the primary per-alpha result
runs_test_ww <- function(ex_vec, tag) {
  ex <- as.integer(na.omit(as.logical(ex_vec)))
  n  <- length(ex); n1 <- sum(ex); n0 <- n - n1
  if (n1 == 0 || n0 == 0) {
    cat(sprintf("  %s VaR: %d exceptions — runs test not applicable\n", tag, n1))
    return(list(n = n, n1 = n1, n0 = n0, runs = NA_real_,
                mu_R = NA_real_, z = NA_real_, p = NA_real_,
                pass = NA, note = "no exceptions or all exceptions"))
  }
  runs <- 1L + sum(diff(ex) != 0L)
  mu_R <- 2 * n0 * n1 / n + 1
  sd_R <- sqrt(2 * n0 * n1 * (2 * n0 * n1 - n0 - n1) / (n^2 * (n - 1)))
  z    <- (runs - mu_R) / sd_R
  p    <- 2 * pnorm(-abs(z))
  cat(sprintf("  %s VaR: n=%d exceptions=%d runs=%d mu_R=%.2f z=%.3f p=%.4f -> %s\n",
              tag, n, n1, runs, mu_R, z, p,
              if (p > 0.05) "PASS (independent)" else "FAIL (clustered)"))
  list(n = n, n1 = n1, n0 = n0, runs = runs, mu_R = mu_R,
       z = z, p = p, pass = p > 0.05, note = "")
}

exc_magnitude <- function(ex_col, pnl_col, var_col, tag) {
  ex_idx <- which(as.logical(ex_col) & !is.na(ex_col))
  if (length(ex_idx) == 0) {
    cat(sprintf("  %s VaR: no exceptions — magnitude analysis skipped\n", tag))
    return(data.frame(alpha = tag, count = 0L,
                      mean_excess = NA_real_, median_excess = NA_real_,
                      max_excess = NA_real_, p95_excess = NA_real_,
                      stringsAsFactors = FALSE))
  }
  realised  <- -pnl_col[ex_idx]
  threshold <- var_col[ex_idx]
  excess    <- (realised / threshold) - 1
  cat(sprintf("  %s VaR: %d exceptions | mean_excess=%.2f%% median=%.2f%% max=%.2f%%\n",
              tag, length(ex_idx),
              100 * mean(excess, na.rm = TRUE),
              100 * median(excess, na.rm = TRUE),
              100 * max(excess, na.rm = TRUE)))
  data.frame(alpha         = tag,
             count         = length(ex_idx),
             mean_excess   = round(100 * mean(excess,           na.rm = TRUE), 2),
             median_excess = round(100 * median(excess,         na.rm = TRUE), 2),
             max_excess    = round(100 * max(excess,            na.rm = TRUE), 2),
             p95_excess    = round(100 * quantile(excess, 0.95, na.rm = TRUE), 2),
             stringsAsFactors = FALSE)
}

cat("\nWald-Wolfowitz runs test (H0: exceptions independent in time):\n")
r95 <- runs_test_ww(rv$exception_95, "95%")
r99 <- runs_test_ww(rv$exception_99, "99%")

cat("\nException magnitude (excess = realised/VaR - 1):\n")
m95 <- exc_magnitude(rv$exception_95, rv$realised_pnl, rv$VaR_95, "95%")
m99 <- exc_magnitude(rv$exception_99, rv$realised_pnl, rv$VaR_99, "99%")

layerC_tbl <- data.frame(
  alpha           = c("95%", "99%"),
  n               = c(r95$n,   r99$n),
  exceptions      = c(r95$n1,  r99$n1),
  expected        = c(round(0.05 * r95$n, 1), round(0.01 * r99$n, 1)),
  runs            = c(r95$runs,  r99$runs),
  runs_z          = c(round(r95$z, 3), round(r99$z, 3)),
  runs_p          = c(round(r95$p, 4), round(r99$p, 4)),
  runs_pass       = c(r95$pass, r99$pass),
  mean_excess_pct = c(m95$mean_excess, m99$mean_excess),
  stringsAsFactors = FALSE)

write.csv(layerC_tbl, file.path(TBL_DIR, "step15_layerC_exception_diag.csv"), row.names = FALSE)
results$post_validation$layerC <- layerC_tbl

save_png("step15d_exception_strip.png", quote({
  n_rv     <- nrow(rv)
  ex95_idx <- which(as.logical(rv$exception_95))
  ex99_idx <- which(as.logical(rv$exception_99))
  par(mar = c(4, 3, 3, 1))
  plot(seq_len(n_rv), rep(1, n_rv), type = "n", ylim = c(0.5, 1.5),
       xlab = "Window index", ylab = "", yaxt = "n",
       main = "Layer C — Exception strip (orange=95% breach, darkred=99%)")
  if (length(ex95_idx)) abline(v = ex95_idx, col = "#e84118", lwd = 1.5)
  if (length(ex99_idx)) abline(v = ex99_idx, col = "#6b0000", lwd = 3.0)
  legend("topright",
         legend = c(sprintf("95%% (n=%d)", length(ex95_idx)),
                    sprintf("99%% (n=%d)", length(ex99_idx))),
         col = c("#e84118","#6b0000"), lwd = c(1.5, 3), bty = "n")
}), w = 12, h = 3)

save_png("step15e_exception_magnitude.png", quote({
  par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
  for (cfg in list(list("95%", rv$exception_95, rv$VaR_95),
                   list("99%", rv$exception_99, rv$VaR_99))) {
    tag    <- cfg[[1]]; ex_col <- cfg[[2]]; var_col <- cfg[[3]]
    ex_idx <- which(as.logical(ex_col) & !is.na(ex_col))
    if (length(ex_idx) < 2) {
      plot.new(); title(paste0("VaR ", tag, " — insufficient exceptions")); next
    }
    excess <- (-rv$realised_pnl[ex_idx] / var_col[ex_idx]) - 1
    me     <- mean(excess, na.rm = TRUE)
    hist(excess, breaks = max(5L, length(excess) %/% 3L),
         probability = TRUE, col = "grey80", border = "white",
         main = paste0("Layer C — Excess beyond VaR ", tag),
         xlab = "Excess ratio (realised/VaR - 1)")
    abline(v = 0,  col = "steelblue", lwd = 2, lty = 2)
    abline(v = me, col = "darkred",   lwd = 2)
    legend("topright",
           legend = c("threshold (0)", sprintf("mean (%.2f%%)", 100 * me)),
           col = c("steelblue","darkred"), lty = 2:1, lwd = 2, bty = "n", cex = 0.8)
  }
}), w = 10, h = 5)

# Layer C output block
cat("\nLayer C consolidated:\n")
print(layerC_tbl, row.names = FALSE)
cat("Layer C complete.\n")



# Layer D — Variance targeting diagnostic
#
# Variance targeting (VT) pins the GARCH unconditional variance to the sample
# variance. When the model-implied unconditional variance deviates by more
# than 5%, VT either did not converge or the model structure (e.g. apARCH's
# delta-power constraint) prevents correct targeting in variance space.


poolD <- do.call(rbind, lapply(factor_names, function(fct) {
  vc        <- results$variance_check[[fct]]
  fit       <- results$garch_fit[[fct]]
  cf        <- coef(fit)
  per       <- vc$persistence
  garch_mdl <- if (!is.null(results$garch_model) && !is.null(results$garch_model[[fct]]))
                 results$garch_model[[fct]] else "sGARCH"

  emp_sd   <- sqrt(max(vc$empirical, 0))
  model_sd <- if (!is.na(vc$modelled) && vc$modelled >= 0) sqrt(vc$modelled) else NA_real_
  ratio    <- vc$ratio
  dev      <- if (!is.na(ratio)) 100 * (ratio - 1) else NA_real_
  alpha0_now <- if ("omega" %in% names(cf)) unname(cf["omega"]) else NA_real_
  # For sGARCH the VT target is (1-alpha-beta)*sigma2_emp.
  # For gjrGARCH/eGARCH/apARCH use rugarch's uncvariance() directly.
  alpha0_vt <- if (garch_mdl == "sGARCH") {
    if (!is.na(per) && !is.na(vc$empirical)) (1 - per) * vc$empirical else NA_real_
  } else {
    tryCatch(as.numeric(uncvariance(fit)), error = function(e) NA_real_)
  }
  vt_needed <- isTRUE(!is.na(dev) && abs(dev) > 5)

  data.frame(
    Factor         = fct,
    GARCHModel     = garch_mdl,
    EmpSd_pct      = round(100 * emp_sd, 4),
    ModelSd_pct    = round(100 * ifelse(is.na(model_sd), NA_real_, model_sd), 4),
    Direction      = if (!is.na(ratio)) ifelse(ratio > 1, "model > obs", "model < obs")
                     else "unknown",
    Deviation_pct  = round(dev, 2),
    Alpha0_now     = signif(alpha0_now, 4),
    Alpha0_target  = signif(alpha0_vt,  4),
    VT_deviation_flag = vt_needed,
    stringsAsFactors = FALSE)
}))

# Flag factors with large VT deviation — inline since these are primary warnings
for (i in seq_len(nrow(poolD))) {
  if (!isTRUE(poolD$VT_deviation_flag[i])) next
  fct     <- poolD$Factor[i]
  dir_txt <- if (poolD$Direction[i] == "model > obs") "OVERSTATES" else "UNDERSTATES"
  cat(sprintf("\n-> Factor %s [%s]: model %s variance by %.1f%% despite variance.targeting=TRUE.\n",
              fct, poolD$GARCHModel[i], dir_txt, abs(poolD$Deviation_pct[i])))
  cat("   This suggests a convergence issue with variance targeting.\n")
  cat("   Consider: (a) checking optimizer convergence, (b) trying a different solver,\n")
  cat("   (c) switching model type in model_candidates.\n")
}

write.csv(poolD, file.path(TBL_DIR, "step15_layerD_variance_targeting.csv"), row.names = FALSE)
results$post_validation$layerD <- poolD

save_png("step15f_variance_targeting.png", quote({
  sd_mat <- rbind(poolD$EmpSd_pct, poolD$ModelSd_pct)
  colnames(sd_mat) <- poolD$Factor
  par(mar = c(5, 5, 4, 2))
  y_top <- max(sd_mat, na.rm = TRUE) * 1.3
  mp <- barplot(sd_mat, beside = TRUE,
                col = c("steelblue","darkred"),
                ylim = c(0, y_top),
                names.arg = poolD$Factor,
                xlab = "Factor", ylab = "Unconditional standard deviation (%)",
                main = "Layer D — Empirical vs modelled unconditional sigma")
  legend("topright", legend = c("Empirical sigma","Modelled sigma"),
         fill = c("steelblue","darkred"), bty = "n")
  for (j in seq_len(ncol(sd_mat))) {
    dev_j <- poolD$Deviation_pct[j]
    if (!is.na(dev_j)) {
      x_pos <- mean(mp[, j])
      y_pos <- max(sd_mat[, j], na.rm = TRUE) + 0.04 * max(sd_mat, na.rm = TRUE)
      text(x_pos, y_pos, sprintf("%+.1f%%", dev_j),
           col  = if (abs(dev_j) > 5) "darkred" else "grey40",
           cex  = 0.85,
           font = if (abs(dev_j) > 5) 2L else 1L)
    }
  }
}), w = max(8, 2 * K), h = 6)

# Layer D output block
cat("\nLayer D — variance targeting diagnostic:\n")
print(poolD, row.names = FALSE)
cat("Layer D complete.\n")


##############################################################################
# Summary
##############################################################################

A_allpass <- paste(sapply(seq_len(nrow(poolA)), function(i)
  sprintf("%s %s", poolA$Factor[i], if (isTRUE(poolA$AllPass[i])) "TRUE" else "FALSE")),
  collapse = " | ")

pct_vals   <- unlist(poolB[, paste0("C", 1:6, "_pct")])
col_means  <- colMeans(as.matrix(poolB[, paste0("C", 1:6, "_pct")]), na.rm = TRUE)
min_pct    <- min(pct_vals, na.rm = TRUE)
min_crit   <- paste0("C", which.min(col_means))
med_pct    <- median(pct_vals, na.rm = TRUE)

fmt_c <- function(r) sprintf(
  "%d exc (exp %.1f), runs p=%s, mean excess=%.2f%%",
  r$exceptions, r$expected,
  ifelse(is.na(r$runs_p), "N/A", as.character(round(r$runs_p, 4))),
  ifelse(is.na(r$mean_excess_pct), 0, r$mean_excess_pct))

vt_fcts <- poolD$Factor[isTRUE(poolD$VT_deviation_flag) | poolD$VT_deviation_flag == TRUE]
vt_str  <- if (length(vt_fcts) == 0) "None (all within +/-5%)" else
  paste(sapply(vt_fcts, function(f) {
    r <- poolD[poolD$Factor == f, ]
    sprintf("%s/%s (dev %+.1f%%)", f, r$GARCHModel, r$Deviation_pct)
  }), collapse = ", ")

cat("\nStep 15 — post-VaR validation summary\n")
cat(sprintf("  Layer A -- in-sample (%d factors x 6 criteria):\n", K))
cat(sprintf("    All criteria pass: %s\n", A_allpass))
cat(sprintf("  Layer B -- rolling (%d windows):\n", length(rows_out)))
cat(sprintf("    Min pass rate per criterion: %s (%.0f%%)\n", min_crit, min_pct))
cat(sprintf("    Median pass rate: %.0f%%\n", med_pct))
cat("  Layer C -- exception diagnostics:\n")
cat(sprintf("    95%% VaR: %s\n", fmt_c(layerC_tbl[1, ])))
cat(sprintf("    99%% VaR: %s\n", fmt_c(layerC_tbl[2, ])))
cat("  Layer D -- variance targeting:\n")
cat(sprintf("    Deviation flags: %s\n", vt_str))

# Unified summary CSV
sum_rows <- list()

for (i in seq_len(nrow(poolA)))
  sum_rows[[length(sum_rows) + 1]] <- data.frame(
    Layer = "A",
    Item  = paste(poolA$Factor[i], "all criteria"),
    Value = as.character(poolA$AllPass[i]),
    Pass  = isTRUE(poolA$AllPass[i]),
    stringsAsFactors = FALSE)

sum_rows[[length(sum_rows) + 1]] <- data.frame(
  Layer = "B", Item = "Min pass rate (any criterion/factor)",
  Value = sprintf("%.2f", min_pct / 100), Pass = min_pct >= 70,
  stringsAsFactors = FALSE)
sum_rows[[length(sum_rows) + 1]] <- data.frame(
  Layer = "B", Item = "Median pass rate",
  Value = sprintf("%.2f", med_pct / 100), Pass = med_pct >= 90,
  stringsAsFactors = FALSE)

for (i in 1:2) {
  p <- layerC_tbl$runs_p[i]
  sum_rows[[length(sum_rows) + 1]] <- data.frame(
    Layer = "C",
    Item  = paste(layerC_tbl$alpha[i], "VaR runs test p"),
    Value = ifelse(is.na(p), "N/A", sprintf("%.4f", p)),
    Pass  = isTRUE(!is.na(p) && p > 0.05),
    stringsAsFactors = FALSE)
}

for (i in seq_len(nrow(poolD)))
  sum_rows[[length(sum_rows) + 1]] <- data.frame(
    Layer = "D",
    Item  = paste(poolD$Factor[i], "VT dev >5% flag"),
    Value = as.character(poolD$VT_deviation_flag[i]),
    Pass  = !isTRUE(poolD$VT_deviation_flag[i]),
    stringsAsFactors = FALSE)

summary_csv <- do.call(rbind, sum_rows)
write.csv(summary_csv, file.path(TBL_DIR, "step15_summary.csv"), row.names = FALSE)
results$post_validation$summary <- summary_csv

cat(sprintf("\nFigures -> %s/  |  Tables -> %s/\n", FIG_DIR, TBL_DIR))
cat("results$post_validation: layerA | layerB | layerC | layerD | summary\n")
cat("Step 15 complete.\n")
