
# steps_3_to_8_marginal_garch_fixed.R
#
# GARCH-Copula VaR — Phase B: Marginal GARCH models (Steps 3-8)
#
#   Step 3  — ADF stationarity test
#   Step 4  — ARMA mean model selection
#   Step 5  — ARCH-effect tests
#   Step 6  — GARCH model and innovation distribution selection
#   Step 7  — 6-criteria validation of the selected GARCH model
#   Step 8  — Diagnostic residual distribution fitting on Ẑ_t
#
# Usage: source("steps_3_to_8_marginal_garch_fixed.R")
# The 'results' list is passed to steps_9_to_12_copula_var.R automatically.

# Force Copula Engine to use Vine Copula in the subsequent pipeline
options(manual_copula_engine = "vine")

# Load packages; install any that are missing
pkgs <- c("xts","zoo","moments","tseries","forecast","FinTS",
          "rugarch","gamlss","gamlss.dist","gamlss.add","copula",
          "WeightedPortTest")
miss <- setdiff(pkgs, rownames(installed.packages()))
if (length(miss)) install.packages(miss, dependencies = TRUE,
                                   repos = "https://cloud.r-project.org")
suppressPackageStartupMessages(lapply(pkgs, library, character.only = TRUE))


# Output directories and helper for saving plots to PNG
FIG_DIR <- "outputs/figures/GARCH"
TBL_DIR <- "outputs/tables/GARCH"
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(TBL_DIR, recursive = TRUE, showWarnings = FALSE)

save_png <- function(name, expr, w = 10, h = 6) {
  f <- file.path(FIG_DIR, name)
  png(f, width = w * 100, height = h * 100, res = 100)
  on.exit(dev.off(), add = TRUE)
  eval(expr, parent.frame())
}


# Load risk factors from compute_returns.py output.
# Columns produced by compute_returns.py include:
#   SPY_log_return, DGS10_change, GLD_log_return, EURUSD_log_return,
#   SPY_level_change, VIX_change.
#
# Methodological fix: SPY_level_change is deterministic given SPY_log_return
# and the current SPY level. It is therefore excluded as a separate stochastic
# copula/GARCH factor and reconstructed later as spot_t * (exp(r_SPY)-1).
raw          <- read.csv("data/processed/risk_factors.csv", header = TRUE,
                         stringsAsFactors = FALSE, check.names = FALSE)
factor_dates <- as.Date(raw[, 1])
all_factor_names <- colnames(raw)[-1]
factor_names <- setdiff(all_factor_names, "SPY_level_change")

mat_raw        <- as.matrix(raw[, factor_names, drop = FALSE])
class(mat_raw) <- "numeric"
ok             <- complete.cases(mat_raw)
factors_mat    <- mat_raw[ok, , drop = FALSE]
rownames(factors_mat) <- as.character(factor_dates[ok])
factors_xts    <- xts(factors_mat, order.by = factor_dates[ok])

# Keep factors_list only for backwards compatibility, but build it from the
# common complete-case matrix so every factor has the exact same time index.
factors_list <- setNames(lapply(factor_names, function(c)
  as.numeric(factors_mat[, c])), factor_names)

cat("Loaded stochastic factors:", paste(factor_names, collapse = ", "),
    "| rows:", nrow(factors_mat), "\n")
if ("SPY_level_change" %in% all_factor_names)
  cat("Excluded SPY_level_change from stochastic modelling; it is derived from SPY_log_return in P&L mapping.\n")


# Results containers — all Step 3-8 outputs are stored here.
# Steps 9-12 consume results$garch_fit and use rank-based PITs for the main copula.
# results$pit_family/results$pit_params are kept for Step-8/Step-9 diagnostics only.
results <- list(
  adf                   = list(),   # Step 3
  arma_order            = list(),   # Step 4
  arma_fit              = list(),   # Step 4
  arma_resid            = list(),   # Step 4
  arch_test             = list(),   # Step 5
  garch_dist_comparison = list(),   # Step 6 — full candidate table (Model x Dist)
  garch_dist_choice     = list(),   # Step 6 — chosen innovation distribution string
  garch_model           = list(),   # Step 6 — chosen GARCH model type string
  garch_fit             = list(),   # Step 6 — chosen ugarchfit object
  variance_check        = list(),   # Step 7 — C4 detail
  garch_valid           = list(),   # Step 7 — 6-criteria pass/fail table
  pit_family            = list(),   # Step 8 diagnostic — chosen gamlss family name
  pit_params            = list(),   # Step 8 diagnostic — chosen family parameters
  pit_comparison        = list()    # Step 8 diagnostic — full candidate table
)


# Main loop: Steps 3-8 for each risk factor.
# All intermediate outputs (prints, tables) are collected and shown together
# at the end of each factor iteration

for (fct in factor_names) {

  # Use the common complete-case matrix. This prevents silent time-index
  # misalignment that would arise from removing NAs separately by column.
  y <- as.numeric(factors_mat[, fct])
  n <- length(y)


  # Step 3 — ADF stationarity test
  # Returns must be stationary for ARMA-GARCH modelling. ADF H0: unit root.
  # We want p < 0.05 (reject unit root) to proceed.
  adf <- adf.test(y)
  results$adf[[fct]] <- adf

  save_png(paste0("garch_step3_adf_", fct, ".png"), quote({
    par(mfrow = c(2, 1), mar = c(4, 4, 3, 1))
    plot(y, type = "l", col = "steelblue",
         main = paste0("Returns — ", fct, "  (ADF p=", round(adf$p.value, 4), ")"),
         xlab = "t", ylab = "log-ret")
    abline(h = 0, col = "grey60", lty = 2)
    rm <- stats::filter(y, rep(1/60, 60), sides = 2)
    rs <- sqrt(stats::filter((y - mean(y))^2, rep(1/60, 60), sides = 2))
    plot(rm, type = "l", col = "darkred",
         ylim = range(c(rm, rs), na.rm = TRUE),
         main = "60-obs rolling mean (red) & sd (blue)", xlab = "t", ylab = "")
    lines(rs, col = "steelblue"); abline(h = 0, col = "grey60", lty = 2)
  }))


  # Step 4 — ARMA mean model selection
  # auto.arima picks lowest-AIC ARMA(p,q) with d=0 (returns are already stationary).
  # Ljung-Box on residuals: want p > 0.05 (no remaining autocorrelation).
  # ARMA residuals become the input series to the GARCH model in Step 6.
  af  <- auto.arima(y, max.p = 5, max.q = 5, max.d = 0, stationary = TRUE,
                    seasonal = FALSE, ic = "aic",
                    stepwise = FALSE, approximation = FALSE)
  ord <- arimaorder(af)
  lb4 <- Box.test(residuals(af), lag = 10, type = "Ljung-Box",
                  fitdf = ord[1] + ord[3])
  results$arma_order[[fct]] <- ord
  results$arma_fit[[fct]]   <- af
  results$arma_resid[[fct]] <- residuals(af)

  save_png(paste0("garch_step4_arma_", fct, ".png"), quote({
    par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
    Acf(y,  lag.max = 40, main = paste("ACF returns —", fct))
    Pacf(y, lag.max = 40, main = paste("PACF returns —", fct))
    Acf(residuals(af),  lag.max = 40,
        main = paste0("ACF resid ARMA(", ord[1], ",", ord[3], ")"))
    Pacf(residuals(af), lag.max = 40, main = "PACF resid")
  }), w = 12, h = 8)


  # Step 5 — ARCH-effect tests
  # Engle ARCH-LM and Ljung-Box on squared ARMA residuals.
  # Both tests must reject (p < 0.05) to confirm volatility clustering.
  ri   <- residuals(af)
  arch <- ArchTest(ri, lags = 10)
  lb5  <- Box.test(ri^2, lag = 10, type = "Ljung-Box")
  results$arch_test[[fct]] <- list(arch = arch, lb_sq = lb5)

  save_png(paste0("garch_step5_arch_", fct, ".png"), quote({
    par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
    plot(ri,    type = "l", col = "steelblue",
         main = paste("Residuals —", fct), xlab = "t", ylab = "")
    abline(h = 0, col = "grey60", lty = 2)
    plot(ri^2,  type = "l", col = "darkred",
         main = "Squared residuals", xlab = "t", ylab = "")
    Acf(ri,   lag.max = 40, main = "ACF residuals")
    Acf(ri^2, lag.max = 40,
        main = paste0("ACF squared  (LB p=", round(lb5$p.value, 4), ")"))
  }), w = 12, h = 8)


  # GARCH order: default (1,1); override via manual_garch_order before sourcing.
  go   <- if (exists("manual_garch_order") && !is.null(manual_garch_order[[fct]]))
            manual_garch_order[[fct]] else c(1, 1)
  p_ar <- ord[1]; q_ma <- ord[3]


  # Step 6 — GARCH innovation distribution selection
  #
  # We fit full ARMA-GARCH models (not just GARCH in isolation) with different
  # innovation distributions and compare via AIC + Pearson GoF. This is the
  # correct approach because the standardised residuals Ẑ_t are the actual
  # innovation series — their distribution, not the raw return distribution,
  # matters for VaR accuracy.
  #
  # Selection rule: lowest AIC among GoF-passing candidates.
  # If nothing passes GoF, fall back to global lowest AIC with a warning.
  # Override: set manual_garch_dist <- list(<factor> = "sstd") before sourcing.
  # Override: set manual_garch_model <- list(<factor> = "gjrGARCH") to restrict types.

  model_candidates <- if (exists("manual_garch_model") && !is.null(manual_garch_model[[fct]]))
                        manual_garch_model[[fct]]
                      else
                        c("sGARCH", "gjrGARCH", "eGARCH", "apARCH")

  garch_dists <- c("norm", "std", "sstd", "ged", "sged", "nig", "jsu")

  comparison_rows <- list()
  succ_fits       <- list()   # keyed by "model-dist"

  cat(sprintf("\nStep 6 — fitting %d model type(s) x %d distributions for %s\n",
              length(model_candidates), length(garch_dists), fct))

  for (mdl in model_candidates) {
    cat(sprintf("  [%s]", mdl))
    for (dist_c in garch_dists) {
      combo <- tryCatch({
        spec_g <- ugarchspec(
          variance.model     = list(model = mdl, garchOrder = go,
                                    variance.targeting = FALSE),
          mean.model         = list(armaOrder = c(p_ar, q_ma), include.mean = TRUE),
          distribution.model = dist_c)
        fit_g  <- ugarchfit(spec = spec_g, data = y, solver = "hybrid")
        ic_g   <- infocriteria(fit_g)
        gof_g  <- gof(fit_g, groups = c(20, 30, 40, 50))
        list(Model        = mdl,
             Distribution = dist_c,
             fit          = fit_g,
             logLik       = likelihood(fit_g),
             AIC          = ic_g[1],
             BIC          = ic_g[2],
             GoF_p_min    = min(gof_g[, "p-value(g-1)"], na.rm = TRUE),
             GoF_p_mean   = mean(gof_g[, "p-value(g-1)"], na.rm = TRUE),
             Converged    = TRUE,
             ok           = TRUE)
      }, error = function(e) {
        list(Model = mdl, Distribution = dist_c, ok = FALSE)
      })

      key <- paste(mdl, dist_c, sep = "-")
      if (combo$ok) succ_fits[[key]] <- combo$fit
      comparison_rows[[length(comparison_rows) + 1]] <- data.frame(
        Model        = combo$Model,
        Distribution = combo$Distribution,
        logLik       = if (combo$ok) round(combo$logLik, 2) else NA_real_,
        AIC          = if (combo$ok) round(combo$AIC,    4) else NA_real_,
        BIC          = if (combo$ok) round(combo$BIC,    4) else NA_real_,
        GoF_p_min    = if (combo$ok) round(combo$GoF_p_min, 4) else NA_real_,
        GoF_p_mean   = if (combo$ok) round(combo$GoF_p_mean, 4) else NA_real_,
        GoF_pass     = if (combo$ok) combo$GoF_p_min > 0.05 else FALSE,
        Converged    = combo$ok,
        stringsAsFactors = FALSE)
      cat(if (combo$ok) " ." else " x")
    }
    cat("\n")
  }

  if (length(succ_fits) == 0) {
    cat("All GARCH fits failed for", fct, "- skipping.\n"); next
  }

  comparison_df <- do.call(rbind, comparison_rows)
  comparison_df <- comparison_df[order(comparison_df$AIC, na.last = TRUE), ]
  rownames(comparison_df) <- NULL
  write.csv(comparison_df,
            file.path(TBL_DIR, paste0("step6_dist_comparison_", fct, ".csv")),
            row.names = FALSE)
  results$garch_dist_comparison[[fct]] <- comparison_df

  # Pick winner; build sel_msg for the output block at end of loop
  cmp_pass <- comparison_df[!is.na(comparison_df$AIC) & comparison_df$GoF_pass, ]
  if (exists("manual_garch_dist") && !is.null(manual_garch_dist[[fct]])) {
    gd     <- manual_garch_dist[[fct]]
    cmp_gd <- comparison_df[!is.na(comparison_df$AIC) &
                               comparison_df$Distribution == gd, ]
    gm     <- if (nrow(cmp_gd) > 0) cmp_gd$Model[1] else model_candidates[1]
    sel_msg <- paste0("manual override: ", gm, " / ", gd)
  } else if (nrow(cmp_pass) > 0) {
    gm <- cmp_pass$Model[1]
    gd <- cmp_pass$Distribution[1]
    sel_msg <- paste0(gm, " / ", gd, " (lowest AIC + GoF pass)")
  } else {
    cmp_ok  <- comparison_df[!is.na(comparison_df$AIC), ]
    best_r  <- cmp_ok[which.min(cmp_ok$AIC), ]
    gm <- best_r$Model; gd <- best_r$Distribution
    sel_msg <- paste0("WARNING: no GoF pass - fallback to lowest AIC: ", gm, " / ", gd)
  }
  results$garch_dist_choice[[fct]] <- gd
  results$garch_model[[fct]]       <- gm

  key_chosen       <- paste(gm, gd, sep = "-")
  fit_g            <- succ_fits[[key_chosen]]
  results$garch_fit[[fct]] <- fit_g

  # AIC heatmap: blue = lowest (best), red = highest (worst); winner marked with star
  save_png(paste0("step6a_dist_bars_", fct, ".png"), quote({
    models_u <- model_candidates
    dists_u  <- garch_dists
    nm <- length(models_u); nd <- length(dists_u)
    aic_mat <- matrix(NA_real_, nrow = nm, ncol = nd,
                      dimnames = list(models_u, dists_u))
    for (ri in seq_len(nrow(comparison_df))) {
      m <- comparison_df$Model[ri]; d <- comparison_df$Distribution[ri]
      if (m %in% models_u && d %in% dists_u)
        aic_mat[m, d] <- comparison_df$AIC[ri]
    }
    aic_vec  <- as.vector(aic_mat)
    aic_rng  <- range(aic_vec, na.rm = TRUE)
    aic_norm <- if (diff(aic_rng) > 0)
                  (aic_mat - aic_rng[1]) / diff(aic_rng)
                else
                  matrix(0.5, nm, nd, dimnames = list(models_u, dists_u))
    pal     <- colorRampPalette(c("#2166ac", "#f7f7f7", "#d73027"))(101)
    col_idx <- pmin(pmax(round(aic_norm * 100) + 1L, 1L), 101L)
    col_mat <- matrix(pal[col_idx], nrow = nm, ncol = nd)
    col_mat[is.na(aic_mat)] <- "grey80"
    par(mar = c(6, 8, 5, 2))
    plot(0, type = "n", xlim = c(0.5, nd + 0.5), ylim = c(0.5, nm + 0.5),
         xaxt = "n", yaxt = "n",
         main = paste0("Step 6 — GARCH AIC heatmap (blue=best) — ", fct),
         xlab = "", ylab = "")
    axis(1, at = seq_len(nd), labels = dists_u, las = 2, cex.axis = 0.85)
    axis(2, at = seq_len(nm), labels = models_u, las = 1, cex.axis = 0.85)
    mtext("Distribution", side = 1, line = 4.5)
    mtext("Model",        side = 2, line = 6.5)
    for (mi in seq_len(nm)) {
      for (di in seq_len(nd)) {
        rect(di - 0.5, mi - 0.5, di + 0.5, mi + 0.5,
             col = col_mat[mi, di], border = "white", lwd = 1.5)
        aval    <- aic_mat[models_u[mi], dists_u[di]]
        is_best <- (models_u[mi] == gm && dists_u[di] == gd)
        lbl <- if (!is.na(aval))
                 paste0(if (is_best) "* " else "", round(aval, 3))
               else "-"
        text(di, mi, lbl, cex = 0.65,
             col  = "black",
             font = if (is_best) 2L else 1L)
      }
    }
  }), w = max(10, length(garch_dists) * 1.4),
      h = max(5,  length(model_candidates) * 1.3))

  # QQ grid for all distributions of the winning model type; winner in darkred
  save_png(paste0("step6b_zhat_qq_", fct, ".png"), quote({
    gm_rows <- comparison_df[comparison_df$Model == gm & !is.na(comparison_df$AIC), ]
    gm_succ <- Filter(Negate(is.null), lapply(seq_len(nrow(gm_rows)), function(i) {
      k     <- paste(gm_rows$Model[i], gm_rows$Distribution[i], sep = "-")
      fit_k <- succ_fits[[k]]
      if (is.null(fit_k)) return(NULL)
      list(dist = gm_rows$Distribution[i], fit = fit_k, AIC = gm_rows$AIC[i])
    }))
    nc_qq <- 4; nr_qq <- max(1L, ceiling(length(gm_succ) / nc_qq))
    par(mfrow = c(nr_qq, nc_qq), mar = c(4, 4, 3, 1))
    for (x in gm_succ) {
      Zh_c <- sort(as.numeric(residuals(x$fit, standardize = TRUE)))
      n_c  <- length(Zh_c)
      pp   <- (1:n_c) / (n_c + 1)
      cf   <- coef(x$fit)
      shp  <- if ("shape" %in% names(cf)) unname(cf["shape"]) else NULL
      skw  <- if ("skew"  %in% names(cf)) unname(cf["skew"])  else NULL
      qth  <- tryCatch(qdist(x$dist, p = pp, shape = shp, skew = skw),
                       error = function(e) qnorm(pp))
      ch   <- identical(x$dist, gd)
      plot(qth, Zh_c, pch = 20, cex = 0.5,
           col  = if (ch) "darkred" else "steelblue",
           main = paste0(if (ch) "* " else "", x$dist,
                         "  AIC=", round(x$AIC, 2)),
           xlab = "theoretical", ylab = "empirical")
      abline(0, 1, col = "red", lwd = 1.5)
      if (ch) box(col = "darkred", lwd = 2)
    }
  }), w = 16, h = 8)


  # Step 7 — 6-criteria validation on the selected GARCH model
  #
  # No refitting — reuses results$garch_fit[[fct]] from Step 6.
  # C1/C2 use Weighted.Box.test, which correctly adjusts degrees of freedom
  # for the estimated ARMA and GARCH parameters. Plain Box.test underestimates
  # df and over-rejects for fitted models (see WeightedPortTest documentation).
  Zh  <- as.numeric(residuals(fit_g, standardize = TRUE))
  sig <- as.numeric(sigma(fit_g))

  # Weighted LB helper: falls back to plain Box.test if gamma approx gives NaN
  wlb <- function(x, fitdf) {
    res <- suppressWarnings(tryCatch(
      Weighted.Box.test(x, lag = 10, type = "Ljung-Box", fitdf = fitdf),
      error = function(e) NULL))
    if (is.null(res) || is.na(res$p.value))
      Box.test(x, lag = 10, type = "Ljung-Box", fitdf = fitdf)
    else
      res
  }

  # C1: no autocorrelation in standardised residuals (df = ARMA params)
  c1  <- wlb(Zh,   fitdf = p_ar + q_ma)
  # C2: no remaining ARCH effects in squared residuals (df = GARCH params)
  c2a <- wlb(Zh^2, fitdf = go[1] + go[2])
  c2b <- ArchTest(Zh, lags = 10)
  # C3: innovation distribution fits the standardised residuals
  gof_tbl <- gof(fit_g, groups = c(20, 30, 40, 50))
  # C4: modelled unconditional variance matches empirical variance (want ratio in [0.75, 1.25])
  uv  <- tryCatch(as.numeric(uncvariance(fit_g)), error = function(e) NA_real_)
  ev  <- var(y); ratio <- uv / ev
  per <- tryCatch(as.numeric(persistence(fit_g)), error = function(e) NA_real_)
  # C5: no sign bias (symmetric news impact)
  sb  <- signbias(fit_g)
  # C6: Nyblom parameter stability over the full sample
  ny  <- nyblom(fit_g)

  v1 <- c1$p.value  > 0.05
  v2 <- c2a$p.value > 0.05 && c2b$p.value > 0.05
  v3 <- mean(gof_tbl[, "p-value(g-1)"] > 0.05) >= 0.5
  v4 <- !is.na(ratio) && ratio > 0.75 && ratio < 1.25
  v5 <- all(sb$prob > 0.05)
  v6 <- ny$JointStat < ny$JointCritical[2]

  verdict <- data.frame(
    Criterion = c("C1 WLB Z", "C2 WLB+ARCH Z2", "C3 GoF innov",
                  "C4 Uncond var", "C5 Sign Bias", "C6 Nyblom"),
    Value     = c(round(c1$p.value, 4),
                  round(min(c2a$p.value, c2b$p.value), 4),
                  round(mean(gof_tbl[, "p-value(g-1)"]), 4),
                  round(ratio, 3),
                  round(min(sb$prob), 4),
                  round(ny$JointStat, 4)),
    Pass      = c(v1, v2, v3, v4, v5, v6),
    stringsAsFactors = FALSE)

  # Stored for Step 15 Layer D (variance targeting diagnostic)
  results$variance_check[[fct]] <- list(empirical   = ev,
                                        modelled    = uv,
                                        ratio       = ratio,
                                        deviation   = ratio - 1,
                                        persistence = per,
                                        pass        = v4)
  # Stored for Step 15 Layer A/B (pooled and rolling re-validation)
  results$garch_valid[[fct]] <- verdict

  save_png(paste0("garch_step7_", fct, ".png"), quote({
    par(mfrow = c(2, 3), mar = c(4, 4, 3, 1))
    plot(y, type = "l", col = "grey60",
         main = sprintf("%s(%d,%d)-%s | %s", gm, go[1], go[2], gd, fct),
         xlab = "t", ylab = "ret")
    lines( 2 * sig, col = "red"); lines(-2 * sig, col = "red")
    plot(Zh, type = "l", col = "steelblue",
         main = "Std residuals Zh_t", xlab = "t", ylab = "Zh")
    abline(h = 0, col = "grey60", lty = 2)
    hist(Zh, breaks = 40, probability = TRUE, col = "grey85", border = "white",
         main = "Zh_t vs N(0,1)", xlab = "Zh")
    curve(dnorm(x), add = TRUE, col = "red", lwd = 2)
    Acf(Zh,   lag.max = 40,
        main = paste0("ACF Zh  (WLB p=", round(c1$p.value, 3), ")"))
    Acf(Zh^2, lag.max = 40,
        main = paste0("ACF Zh^2  (WLB p=", round(c2a$p.value, 3), ")"))
    plot(sig^2, type = "l", col = "grey50",
         main = sprintf("Variance ratio=%.2f  dev=%+.1f%%",
                        ratio, 100 * (ratio - 1)),
         xlab = "t", ylab = "sigma^2")
    abline(h = ev, col = "blue",    lwd = 2)
    abline(h = uv, col = "darkred", lwd = 2, lty = 2)
    legend("topright", legend = c("cond sigma^2", "emp var", "model uncond"),
           col = c("grey50", "blue", "darkred"), lty = c(1, 1, 2),
           lwd = 2, bty = "n", cex = 0.75)
  }), w = 15, h = 9)


  # Step 8 — Diagnostic parametric residual distribution fit
  #
  # This block fits parametric gamlss families to the standardised GARCH
  # residuals Zh_t for diagnostics and reporting only. The main VaR pipeline
  # in Steps 9-12 uses rank-based PITs and empirical residual quantiles, so it
  # does not depend on this second parametric residual layer.
  # Selection: lowest AIC among KS-passing families.
  # Override: set manual_pit_dist <- list(<factor> = "SST") before sourcing.
  pit_cands <- c("NO", "TF", "LO", "SST", "ST3", "JSU", "GED")

  pit_fit_one <- function(data, fam) {
    tryCatch({
      m    <- gamlssML(data, family = fam, trace = FALSE)
      pars <- Filter(Negate(is.null),
                     list(mu = m$mu, sigma = m$sigma, nu = m$nu, tau = m$tau))
      pars <- pars[!is.na(unlist(pars))]
      set.seed(42)
      sim  <- do.call(match.fun(paste0("r", fam)), c(list(n = 5000), pars))
      ks   <- suppressWarnings(ks.test(data, sim))
      list(family = fam, fit = m, params = pars,
           AIC    = AIC(m),   BIC    = BIC(m),
           logLik = as.numeric(logLik(m)),
           KS_D   = as.numeric(ks$statistic),
           KS_p   = as.numeric(ks$p.value),
           ok     = TRUE)
    }, error = function(e) list(family = fam, ok = FALSE))
  }

  cat("Fitting diagnostic residual distributions for", fct, "...")
  pit_fits <- lapply(pit_cands, function(f) { cat(" ", f); pit_fit_one(Zh, f) })
  cat("\n")

  pit_succ <- Filter(function(x) isTRUE(x$ok), pit_fits)

  pit_cmp <- do.call(rbind, lapply(pit_succ, function(x)
    data.frame(Family  = x$family,
               logLik  = round(x$logLik, 2),
               AIC     = round(x$AIC,    2),
               BIC     = round(x$BIC,    2),
               KS_D    = round(x$KS_D,   4),
               KS_p    = round(x$KS_p,   4),
               KS_pass = x$KS_p > 0.05,
               stringsAsFactors = FALSE)))
  pit_cmp <- pit_cmp[order(pit_cmp$AIC), ]; rownames(pit_cmp) <- NULL
  write.csv(pit_cmp,
            file.path(TBL_DIR, paste0("step8_pit_comparison_", fct, ".csv")),
            row.names = FALSE)
  results$pit_comparison[[fct]] <- pit_cmp

  pit_pass <- pit_cmp[pit_cmp$KS_pass, ]
  pfam <- if (exists("manual_pit_dist") && !is.null(manual_pit_dist[[fct]]))
            manual_pit_dist[[fct]] else
          if (nrow(pit_pass) > 0) pit_pass$Family[1] else pit_cmp$Family[1]

  pci <- which(sapply(pit_succ, function(x) x$family) == pfam)
  results$pit_family[[fct]] <- pfam
  results$pit_params[[fct]] <- pit_succ[[pci]]$params

  save_png(paste0("step8a_pit_density_", fct, ".png"), quote({
    pal <- c("#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2")
    hist(Zh, breaks = 50, probability = TRUE, col = "grey90", border = "white",
         main = paste0("Step 8 — Diagnostic residual density fits — ", fct), xlab = "Zh")
    for (i in seq_along(pit_succ)) {
      x  <- pit_succ[[i]]; ch <- identical(x$family, pfam)
      curve(do.call(match.fun(paste0("d", x$family)), c(list(x = z), x$params)),
            xname = "z", add = TRUE, col = pal[((i - 1) %% length(pal)) + 1],
            lwd = if (ch) 3 else 1, lty = if (ch) 1 else 3)
    }
    curve(dnorm(z), xname = "z", add = TRUE, col = "red", lwd = 2, lty = 2)
    legend("topleft",
           legend = c(sapply(pit_succ, function(x)
             paste0(if (identical(x$family, pfam)) "* " else "",
                    x$family, "  AIC=", round(x$AIC, 1))), "N(0,1) ref"),
           col = c(pal[seq_along(pit_succ)], "red"),
           lty = c(rep(1, length(pit_succ)), 2), cex = 0.75, bty = "n")
  }), w = 10, h = 6)

  save_png(paste0("step8b_pit_qq_", fct, ".png"), quote({
    nc <- 4; nr <- ceiling(length(pit_succ) / nc)
    par(mfrow = c(nr, nc), mar = c(4, 4, 3, 1))
    for (x in pit_succ) {
      set.seed(42)
      sim <- do.call(match.fun(paste0("r", x$family)),
                     c(list(n = length(Zh)), x$params))
      ch  <- identical(x$family, pfam)
      qqplot(sort(sim), sort(Zh),
             main = paste0(if (ch) "* " else "", x$family,
                           "  KS p=", round(x$KS_p, 3)),
             xlab = "fitted", ylab = "empirical",
             col  = if (ch) "darkred" else "steelblue", pch = 20, cex = 0.6)
      abline(0, 1, col = "red", lwd = 1.5)
      if (ch) box(col = "darkred", lwd = 2)
    }
  }), w = 14, h = 4 * ceiling(length(pit_succ) / 4))


  # Per-factor output block — all Steps 3-8 results printed together
  cat("\nResults for", fct, "\n")
  cat("  ADF p =", round(adf$p.value, 4),
      "->", if (adf$p.value < 0.05) "stationary" else "non-stationary", "\n")
  cat("  ARMA(", ord[1], ",", ord[3], ")  LB residual p =", round(lb4$p.value, 4),
      if (lb4$p.value > 0.05) "" else "  [autocorrelation remains]", "\n")
  cat("  ARCH LM p =", round(arch$p.value, 4),
      "  LBsq p =", round(lb5$p.value, 4),
      "->", if (arch$p.value < 0.05 && lb5$p.value < 0.05)
              "ARCH confirmed" else "weak evidence", "\n")
  cat("\n  Step 6 distribution comparison (sorted by AIC):\n")
  print(comparison_df)
  cat("  Selected:", sel_msg, "\n")
  cat("\n  Step 7 validation:\n")
  print(verdict, row.names = FALSE)
  if (isFALSE(v5)) cat("  Note: sign bias failed (residual asymmetry may be intrinsic)\n")
  if (isFALSE(v3)) cat("  Note: GoF failed - try dist='sstd' or 'nig'\n")
  if (isFALSE(v6)) cat("  Note: Nyblom failed - consider rolling re-estimation\n")
  cat("\n  Step 8 diagnostic residual distributions (sorted by AIC):\n")
  print(pit_cmp)
  cat("  PIT family selected:", pfam, "\n")

}  # end per-factor loop


# Final summary across all factors
# Table 1, 2, 3 are written to CSV and printed once the loop is done.

# Table 1: model spec + 6-criteria pass/fail
summary_tbl <- data.frame(
  Factor     = factor_names,
  ARMA       = sapply(factor_names, function(f) {
    o <- results$arma_order[[f]]; paste0("(", o[1], ",", o[3], ")") }),
  GARCHspec  = sapply(factor_names, function(f) {
    g <- results$garch_fit[[f]]; if (is.null(g)) "-" else {
      m <- g@model
      sprintf("%s(%d,%d)", m$modeldesc$vmodel,
              m$modelinc["alpha"], m$modelinc["beta"]) }}),
  GarchInnov = sapply(factor_names, function(f)
    if (is.null(results$garch_dist_choice[[f]])) "-"
    else results$garch_dist_choice[[f]]),
  PitFamily  = sapply(factor_names, function(f)
    if (is.null(results$pit_family[[f]])) "-"
    else results$pit_family[[f]]),
  C1 = sapply(factor_names, function(f) results$garch_valid[[f]]$Pass[1]),
  C2 = sapply(factor_names, function(f) results$garch_valid[[f]]$Pass[2]),
  C3 = sapply(factor_names, function(f) results$garch_valid[[f]]$Pass[3]),
  C4 = sapply(factor_names, function(f) results$garch_valid[[f]]$Pass[4]),
  C5 = sapply(factor_names, function(f) results$garch_valid[[f]]$Pass[5]),
  C6 = sapply(factor_names, function(f) results$garch_valid[[f]]$Pass[6]),
  stringsAsFactors = FALSE)
cat("\nTable 1: model spec + 6-criteria pass/fail\n")
print(summary_tbl, row.names = FALSE)
write.csv(summary_tbl, file.path(TBL_DIR, "garch_summary.csv"), row.names = FALSE)

# Table 2: unconditional variance check
# Ratio = modelled/empirical. 1.00 is perfect; |deviation| > 25% is a red flag.
var_tbl <- data.frame(
  Factor      = factor_names,
  EmpVar      = sapply(factor_names, function(f)
    sprintf("%.3e", results$variance_check[[f]]$empirical)),
  ModelVar    = sapply(factor_names, function(f)
    sprintf("%.3e", results$variance_check[[f]]$modelled)),
  Ratio       = sapply(factor_names, function(f)
    round(results$variance_check[[f]]$ratio, 3)),
  Deviation   = sapply(factor_names, function(f)
    sprintf("%+.1f%%", 100 * results$variance_check[[f]]$deviation)),
  Persistence = sapply(factor_names, function(f)
    round(results$variance_check[[f]]$persistence, 3)),
  C4_Pass     = sapply(factor_names, function(f)
    results$variance_check[[f]]$pass),
  stringsAsFactors = FALSE)
cat("\nTable 2: observed vs modelled unconditional variance\n")
print(var_tbl, row.names = FALSE)
cat("  Ratio = Modelled/Empirical.  1.00 is perfect.\n")
cat("  |Deviation| > 25% -> mis-specified long-run risk (dangerous for VaR).\n")
cat("  Persistence near 1 -> near-IGARCH; variance barely mean-reverts.\n")
write.csv(var_tbl, file.path(TBL_DIR, "garch_variance_check.csv"), row.names = FALSE)

# Table 3: top-3 GARCH distribution candidates per factor
top3_list <- lapply(factor_names, function(f) {
  cmp <- results$garch_dist_comparison[[f]]
  if (is.null(cmp)) return(NULL)
  top <- head(cmp[!is.na(cmp$AIC), ], 3)
  top$Factor <- f
  top[, intersect(c("Factor", "Model", "Distribution", "AIC", "BIC",
                    "GoF_p_min", "GoF_p_mean", "GoF_pass"), names(top))]
})
top3_tbl <- do.call(rbind, Filter(Negate(is.null), top3_list))
rownames(top3_tbl) <- NULL
cat("\nTable 3: GARCH distribution selection — top 3 by AIC\n")
print(top3_tbl, row.names = FALSE)
write.csv(top3_tbl, file.path(TBL_DIR, "garch_dist_selection_top3.csv"),
          row.names = FALSE)

cat("\nOverride hooks (set before sourcing):\n")
cat("  manual_garch_order <- list(<factor> = c(2,1))      # GARCH lag order\n")
cat("  manual_garch_model <- list(<factor> = 'gjrGARCH')  # variance model type\n")
cat("  manual_garch_dist  <- list(<factor> = 'sstd')      # GARCH innovation dist\n")
cat("  manual_pit_dist    <- list(<factor> = 'SST')       # PIT margin for copula\n")

cat("\nFigures -> outputs/figures/  |  Tables -> outputs/tables/\n")
cat("'results' list in workspace -> used by steps_9_to_12_copula_var.R\n")

# =============================================================================
# steps_9_to_12_copula_var.R
#
# GARCH-Copula VaR Project — Phase C (Copula) + Phase D (VaR)
#
#   Step  9  — PIT: transform Ẑ_t into pseudo-observations U_t ∈ (0,1)
#   Step 10  — Vine copula fit for dependence modelling
#   Step 11  — Monte Carlo simulation of portfolio P&L
#   Step 12  — Rolling VaR (input for Step 13 backtesting)
#   Step 14  — Sensitivity / stress analysis
#
# REQUIRES: source("steps_3_to_8_marginal_garch.R") first.
#   Uses from global workspace: results, factor_names, factors_mat,
#   factors_list, factor_dates, FIG_DIR, TBL_DIR, save_png()
#
# Step 13 (Kupiec / Christoffersen / traffic light) lives in a separate
# backtesting script. This file only produces results$var_rolling for it.

Sys.setenv(RETICULATE_PYTHON = "C:/Users/benlu/AppData/Local/Programs/Python/Python313/python.exe")

suppressPackageStartupMessages(library(copula))
suppressPackageStartupMessages(library(reticulate))
suppressPackageStartupMessages(library(rvinecopulib))
# Vine copulas are used exclusively for both static and rolling VaR.

# rvinecopulib::rvinecop() returns an unnamed matrix; ensure_colnames() guards
# every call site that subsequently indexes by factor name strings.
ensure_colnames <- function(mat, names) {
  if (is.null(colnames(mat))) colnames(mat) <- names
  mat
}

# Semiparametric innovation inverse: maps U(0,1) draws back to standardised
# GARCH residuals using the empirical residual distribution. The output is
# re-centred/re-scaled to preserve the GARCH convention E[Z]=0, Var[Z]=1.
empirical_innov_quantile <- function(u, z) {
  u <- pmin(pmax(as.numeric(u), 1e-6), 1 - 1e-6)
  z <- as.numeric(z[is.finite(z)])
  if (length(z) < 10L) return(rep(NA_real_, length(u)))
  qs <- as.numeric(quantile(z, probs = u, type = 8, na.rm = TRUE))
  if (length(qs) > 1L && is.finite(sd(qs, na.rm = TRUE)) && sd(qs, na.rm = TRUE) > 0) {
    qs <- (qs - mean(qs, na.rm = TRUE)) / sd(qs, na.rm = TRUE)
  }
  qs
}

if (!exists("results") || length(results$garch_fit) == 0)
  stop("Run source('steps_3_to_8_marginal_garch.R') before this file.")
if (!exists("FIG_DIR"))
  stop("FIG_DIR not found — source steps_3_to_8_marginal_garch.R first.")

K <- length(factor_names)


# RETICULATE SETUP — portfolio pricing bridge

reticulate::use_python("C:/Users/benlu/AppData/Local/Programs/Python/Python313/python.exe", required = TRUE)
reticulate::source_python("src/data/portfolio_pricing.py")

cfg_py          <- reticulate::import_from_path("config", path = "src/data",
                                                convert = TRUE)
IRS_NOTIONAL    <- as.numeric(cfg_py$IRS_NOTIONAL)      # 1_000_000
IRS_FIXED_RATE  <- as.numeric(cfg_py$IRS_FIXED_RATE)    # 0.03
STRADDLE_SHARES <- as.numeric(cfg_py$STRADDLE_SHARES)   # 2000
STRADDLE_DAYS   <- as.integer(cfg_py$STRADDLE_DAYS)     # 30
V0              <- as.numeric(cfg_py$V0)                 # 1_000_000
# Linear position notional per asset (V0 × weight = 250 000 each)
SPY_NOTIONAL_USD    <- V0 / 4
GLD_NOTIONAL_USD    <- V0 / 4
EURUSD_NOTIONAL_USD <- V0 / 4
# IEF weight is captured by the IRS position; no separate IEF notional needed.

# Load raw level data and align to factors_mat row dates
.raw_prices <- read.csv("data/raw/prices.csv", stringsAsFactors = FALSE)
.raw_prices$Date <- as.Date(.raw_prices$Date)
.raw_vix    <- read.csv("data/raw/vix.csv",    stringsAsFactors = FALSE)
.raw_vix$Date <- as.Date(.raw_vix[, 1])
colnames(.raw_vix)[2] <- "VIX"
.raw_dgs10  <- read.csv("data/raw/dgs10.csv",  stringsAsFactors = FALSE)
.raw_dgs10$Date <- as.Date(.raw_dgs10[, 1])
colnames(.raw_dgs10)[2] <- "DGS10"

fct_dates <- as.Date(rownames(factors_mat))

.align <- function(df, val_col) {
  v <- df[[val_col]][match(fct_dates, df$Date)]
  for (i in which(is.na(v))) v[i] <- v[max(which(!is.na(v[seq_len(i)])), 1)]
  v
}
spy_lev   <- .align(.raw_prices, "SPY")    # SPY price levels (USD)
dgs10_lev <- .align(.raw_dgs10,  "DGS10") # 10Y yield, raw % (4.37 = 4.37%)
vix_lev   <- .align(.raw_vix,    "VIX")   # VIX index points  (18.0 = 18%)

# Build rolling straddle state in R (mirrors build_straddle_state in Python)
straddle_df <- local({
  n <- length(spy_lev)
  strikes <- numeric(n); tenors <- numeric(n)
  current_strike <- spy_lev[1]; days_held <- 0L
  for (i in seq_len(n)) {
    if (i > 1L && days_held >= STRADDLE_DAYS) {
      current_strike <- spy_lev[i]; days_held <- 0L
    }
    strikes[i] <- current_strike
    tenors[i]  <- max((STRADDLE_DAYS - days_held) / 252, 1 / 252)
    days_held  <- days_held + 1L
  }
  data.frame(strike_spy = strikes, tenor_years = tenors)
})

spot_today     <- tail(spy_lev,   1)
yield_today    <- tail(dgs10_lev, 1) / 100   # raw % -> decimal
vix_today      <- tail(vix_lev,   1)          # VIX index points
last_strike    <- tail(straddle_df$strike_spy, 1)
last_tenor     <- tail(straddle_df$tenor_years, 1)
forecast_tenor <- max(last_tenor - 1 / 252, 1 / 252)

# Load historical total P&L for realised backtest returns
total_pnl_hist      <- read.csv("data/processed/total_portfolio_pnl.csv",
                                stringsAsFactors = FALSE)
total_pnl_hist$Date <- as.Date(total_pnl_hist[, 1])

# map_factors_to_pnl — core pricing function
# r_sim : N x K matrix of simulated GARCH innovations (factor shocks)
# Returns N-length vector of portfolio P&L in USD.
#
# Factor units in risk_factors.csv (verified from compute_returns.py):
#   SPY_log_return    — decimal log-return
#   DGS10_change      — raw % pt change (-0.01 = -1 bp); divide by 100 for decimal
#   GLD_log_return    — decimal log-return
#   EURUSD_log_return — decimal log-return
#   VIX_change        — VIX index point change (0.23 = +0.23 VIX pts); divide by 100 for sigma
#
# SPY_level_change is intentionally NOT simulated as a separate stochastic
# factor. It is implied by SPY_log_return via spot_scen = spot_today * exp(r_SPY).
map_factors_to_pnl <- function(r_sim,
                                spot_today,
                                yield_today,    # decimal (e.g. 0.0437)
                                vix_today,      # VIX index points (e.g. 18.0)
                                strike,
                                tenor) {
  spy_ret   <- r_sim[, "SPY_log_return"]
  gld_ret   <- r_sim[, "GLD_log_return"]
  fx_ret    <- r_sim[, "EURUSD_log_return"]
  dgs10_chg <- r_sim[, "DGS10_change"] / 100   # % pts -> decimal yield delta
  vix_chg   <- r_sim[, "VIX_change"]           # VIX index pts delta

  ## 1) Linear leg: log-return x notional
  pnl_linear <- SPY_NOTIONAL_USD    * spy_ret +
                GLD_NOTIONAL_USD    * gld_ret +
                EURUSD_NOTIONAL_USD * fx_ret

  ## 2) IRS leg: full mark-to-market repricing (vectorised over N scenarios)
  yield_scen <- yield_today + dgs10_chg
  pnl_irs    <- as.numeric(price_irs(IRS_NOTIONAL, IRS_FIXED_RATE, yield_scen, 10L)[[1]]) -
                as.numeric(price_irs(IRS_NOTIONAL, IRS_FIXED_RATE, yield_today, 10L)[[1]])

  ## 3) Straddle leg: Black-Scholes repricing
  # Methodological fix: use the same SPY scenario for linear and option P&L.
  # Do not simulate SPY_level_change separately.
  spot_scen  <- spot_today * exp(spy_ret)
  sigma_scen <- pmax((vix_today + vix_chg) / 100, 1e-6)
  sigma_base <- vix_today / 100

  px_scen <- as.numeric(price_straddle_position(
    spot_scen, strike, tenor, yield_scen, sigma_scen))
  px_base <- as.numeric(price_straddle_position(
    spot_today, strike, tenor, yield_today, sigma_base))

  pnl_straddle <- (px_scen - px_base) * STRADDLE_SHARES

  as.numeric(pnl_linear + pnl_irs + pnl_straddle)
}


# Step 9 — PIT: Probability Integral Transform
#
# Sklar's theorem: any joint distribution = copula(marginals). The copula
# models ONLY the dependence structure; the marginal shape is factored out
# first. Applying U_i,t = F_hat_i(Z_hat_i,t) maps each innovation series
# into a Uniform(0,1) series. If U_i is NOT uniform, F_hat_i is misspecified
# and the copula fit will be unreliable downstream (Step 10).
# KS test: H0: U_i ~ Uniform(0,1). We WANT to fail to reject (p > 0.05).

# 9a. Align standardised GARCH residuals to a common T x K matrix.
# rugarch drops a few leading obs for ARMA warm-up; series can differ in length.
zhat_list <- lapply(factor_names, function(fct)
  as.numeric(residuals(results$garch_fit[[fct]], standardize = TRUE)))

T_min    <- min(sapply(zhat_list, length))
zhat_mat <- do.call(cbind, lapply(zhat_list, tail, T_min))
colnames(zhat_mat) <- factor_names
results$zhat_mat   <- zhat_mat

# 9b. Diagnostic parametric PIT — apply Step-8 diagnostic residual CDFs
U_param <- matrix(NA_real_, nrow = T_min, ncol = K,
                  dimnames = list(NULL, factor_names))
for (fct in factor_names) {
  pfun           <- match.fun(paste0("p", results$pit_family[[fct]]))
  U_param[, fct] <- do.call(pfun, c(list(q = zhat_mat[, fct]),
                                     results$pit_params[[fct]]))
}

# 9c. Non-parametric PIT — rank-based pseudo-observations, no marginal assumption
U_emp            <- pobs(zhat_mat)
results$U_param  <- U_param
results$U_emp    <- U_emp
# Use rank-based PITs for the actual copula fit. This avoids making VaR depend
# on a second parametric GAMLSS layer fitted to already standardised residuals.
U_model          <- U_emp
colnames(U_model) <- factor_names
results$U_model  <- U_model

# Collect KS results now; print after plots so all Step 9 output appears together
ks_results <- lapply(setNames(factor_names, factor_names), function(fct)
  ks.test(U_param[, fct], "punif"))

# 9d. Plots
save_png("step9a_uniform_histograms.png", quote({
  par(mfrow = c(2, K), mar = c(3, 3, 2, 1))
  for (fct in factor_names) {
    hist(U_param[, fct], breaks = 25, probability = TRUE,
         col = "steelblue", border = "white",
         main = paste0("Param PIT\n", fct), xlab = "U", ylab = "")
    abline(h = 1, col = "red", lwd = 2, lty = 2)   # true Uniform density
  }
  for (fct in factor_names) {
    hist(U_emp[, fct], breaks = 25, probability = TRUE,
         col = "grey70", border = "white",
         main = paste0("Rank PIT\n", fct), xlab = "U", ylab = "")
    abline(h = 1, col = "red", lwd = 2, lty = 2)
  }
}), w = 4 * K, h = 8)

save_png("step9b_pairwise_pseudos.png", quote({
  pairs(U_param, pch = ".", col = "steelblue",
        main = "Step 9 — Pairwise pseudo-obs (parametric PIT)")
}), w = 10, h = 10)

# Step 9 output block
cat(sprintf("\nStep 9: aligned Z matrix %d x %d\n", T_min, K))
cat("KS uniformity test (H0: U_i ~ Uniform, want p > 0.05):\n")
for (fct in factor_names) {
  ks <- ks_results[[fct]]
  cat(sprintf("  %-25s  p=%.4f  mean=%.4f  sd=%.4f  %s\n",
              fct, ks$p.value,
              mean(U_param[, fct]), sd(U_param[, fct]),
              if (ks$p.value > 0.05) "PASS (uniform)" else
                "FAIL — diagnostic only; main Vine uses rank PIT"))
}
cat("Step 9 complete.\n")


# Step 10 — Vine copula fit for dependence modelling
#
# Methodological choice: the dependence model is specified directly as a Vine
# copula. This is used consistently for both the static VaR simulation and the
# rolling VaR backtest. No Gaussian/t/Clayton/Gumbel/Frank single-copula
# selection is performed as the main model, because a single global copula is too
# restrictive for a multi-factor portfolio with heterogeneous pairwise and tail
# dependence.
#
# The Vine structure and pair-copula families are selected by AIC within
# rvinecopulib. Pair families allowed below include elliptical and Archimedean
# building blocks, but only as pair-copulas inside the Vine structure.

vine_family_set <- c("gaussian", "t", "clayton", "gumbel", "frank", "joe")

cat("Fitting Vine copula as the exclusive dependence model...\n")
set.seed(42)
vine_fit <- tryCatch(
  rvinecopulib::vinecop(
    U_model,
    family_set = vine_family_set,
    par_method = "mle",
    selcrit    = "aic",
    cores      = 1L
  ),
  error = function(e) {
    stop(sprintf("Vine copula fit failed: %s", e$message))
  }
)

vine_loglik <- tryCatch(as.numeric(logLik(vine_fit)), error = function(e) NA_real_)
vine_aic    <- tryCatch(as.numeric(AIC(vine_fit)),    error = function(e) NA_real_)
vine_bic    <- tryCatch(as.numeric(BIC(vine_fit)),    error = function(e) NA_real_)

cop_cmp <- data.frame(
  Model     = "Vine copula",
  FamilySet = paste(vine_family_set, collapse = ", "),
  logLik    = round(vine_loglik, 2),
  AIC       = round(vine_aic, 2),
  BIC       = round(vine_bic, 2),
  MainModel = TRUE,
  stringsAsFactors = FALSE
)

cat("\nVine copula fit summary:\n")
print(cop_cmp, row.names = FALSE)
write.csv(cop_cmp, file.path(TBL_DIR, "step10_vine_copula_fit.csv"), row.names = FALSE)

chosen_cop_name <- "Vine"
results$copula_chosen    <- chosen_cop_name
results$copula_fit       <- vine_fit
results$vine_copula_fit  <- vine_fit
results$copula_comparison <- cop_cmp
results$vine_family_set  <- vine_family_set
results$vine_AIC         <- vine_aic
results$vine_BIC         <- vine_bic
results$vine_logLik      <- vine_loglik

save_png("step10a_vine_fit_information_criteria.png", quote({
  vals <- c(AIC = vine_aic, BIC = vine_bic)
  par(mar = c(4, 5, 3, 1))
  barplot(vals, horiz = TRUE, las = 1,
          main = "Step 10 — Vine copula information criteria",
          xlab = "Information criterion")
}), w = 8, h = 4)

save_png("step10b_vine_simulated_vs_empirical.png", quote({
  set.seed(1)
  sim_u <- ensure_colnames(rvinecopulib::rvinecop(nrow(U_model), results$vine_copula_fit), factor_names)
  pairs_list <- combn(factor_names, 2, simplify = FALSE)
  n_pairs <- length(pairs_list)
  nc <- min(3, n_pairs); nr <- ceiling(n_pairs / nc)
  par(mfrow = c(nr, nc), mar = c(3, 3, 2, 1))
  for (p in pairs_list) {
    idx_e <- sample(nrow(U_model), min(500, nrow(U_model)))
    idx_s <- sample(nrow(sim_u),  min(500, nrow(sim_u)))
    plot(sim_u[idx_s, p[1]], sim_u[idx_s, p[2]],
         pch = ".", col = "steelblue",
         main = paste(p[1], "vs", p[2]), xlab = p[1], ylab = p[2])
    points(U_model[idx_e, p[1]], U_model[idx_e, p[2]],
           pch = ".", col = "darkred")
    legend("topleft", legend = c("vine simulated", "empirical"),
           col = c("steelblue", "darkred"), pch = 20, bty = "n", cex = 0.8)
  }
}), w = 12, h = 4 * ceiling(length(combn(factor_names, 2, simplify = FALSE)) / 3))

cat("Step 10 complete. Vine copula will be used exclusively downstream.\n")


# Step 11 — Monte Carlo simulation of portfolio P&L
#
# With a fitted copula + fitted GARCH marginals, we generate N joint scenarios:
#   1. Draw copula-dependent Uniforms U* from the fitted copula.
#   2. Invert U*_i through the marginal quantile Q_hat_i -> Z*_i.
#   3. Scale by one-step-ahead GARCH forecast: r*_i = mu_i + sigma_i * Z*_i.
#   4. Map to portfolio P&L via the three-leg pricing bridge.
# The quantile of this P&L distribution is the model-consistent VaR.

# Consistency check: correlate reconstructed P&L against compute_pnl.py output.
# High correlation (> 0.95) confirms the pricing bridge uses the same conventions.
.n_chk  <- min(100L, nrow(factors_mat) - 1L)
.recon  <- vapply(seq_len(.n_chk), function(t)
  map_factors_to_pnl(factors_mat[t, , drop = FALSE],
                     spot_today  = spy_lev[t],
                     yield_today = dgs10_lev[t] / 100,
                     vix_today   = vix_lev[t],
                     strike      = straddle_df$strike_spy[t],
                     tenor       = straddle_df$tenor_years[t]),
  numeric(1))
.pnl_dates  <- total_pnl_hist$Date
.actual_idx <- match(fct_dates[seq_len(.n_chk)], .pnl_dates)
.actual     <- total_pnl_hist$pnl_total[.actual_idx]
.ok         <- !is.na(.actual)
# warning() fires immediately if correlation is low; cat() is deferred to output block
.cor_msg <- if (sum(.ok) >= 10L) {
  .cor <- cor(.recon[.ok], .actual[.ok])
  if (.cor < 0.95)
    warning(sprintf("Reconstruction correlation %.4f < 0.95 — check factor scaling!", .cor))
  sprintf("Reconstruction correlation (n=%d): %.4f", sum(.ok), .cor)
} else {
  "Too few overlapping dates for consistency check."
}

N <- 50000
set.seed(42)
cat(sprintf("Simulating %d scenarios from Vine copula...\n", N))

# 11a. Simulate Vine-copula-dependent Uniforms
PseudoSim <- rvinecopulib::rvinecop(N, results$vine_copula_fit)   # N x K
colnames(PseudoSim) <- factor_names

# 11b. Invert through empirical standardised-residual quantiles.
# This is a filtered historical simulation marginal step: GARCH supplies the
# conditional mean/volatility, while the empirical residual distribution supplies
# the innovation tails without imposing a second parametric GAMLSS model.
Zstar <- matrix(NA_real_, nrow = N, ncol = K, dimnames = list(NULL, factor_names))
for (fct in factor_names) {
  Zstar[, fct] <- empirical_innov_quantile(PseudoSim[, fct], zhat_mat[, fct])
}

# 11c. GARCH one-step-ahead volatility and mean forecasts
sigma_t1 <- setNames(numeric(K), factor_names)
mu_t1    <- setNames(numeric(K), factor_names)
for (fct in factor_names) {
  fc            <- ugarchforecast(results$garch_fit[[fct]], n.ahead = 1)
  sigma_t1[fct] <- as.numeric(sigma(fc))
  mu_t1[fct]    <- as.numeric(fitted(fc))
}

# 11d. Simulated returns: r*_i = mu_i + sigma_i * Z*_i
r_sim <- sweep(Zstar, 2, sigma_t1, "*") + rep(mu_t1, each = N)

# 11e. Portfolio mapping via real pricing (reticulate)
w <- if (exists("manual_weights") && length(manual_weights) == K)
       manual_weights / sum(manual_weights) else rep(1/K, K)
names(w) <- factor_names

scenario_PnL <- map_factors_to_pnl(r_sim,
                                    spot_today  = spot_today,
                                    yield_today = yield_today,
                                    vix_today   = vix_today,
                                    strike      = last_strike,
                                    tenor       = forecast_tenor)

results$scenario_PnL      <- scenario_PnL
results$sigma_t1          <- sigma_t1
results$mu_t1             <- mu_t1
results$portfolio_weights <- w

# 11f. Plots
save_png("step11a_simulated_returns.png", quote({
  pairs_list <- combn(factor_names, 2, simplify = FALSE)
  show_pairs <- pairs_list[seq_len(min(6, length(pairs_list)))]
  nc <- min(3, length(show_pairs)); nr <- ceiling(length(show_pairs) / nc)
  par(mfrow = c(nr, nc), mar = c(3, 3, 2, 1))
  for (p in show_pairs) {
    s_idx <- sample(N, 500); e_idx <- sample(nrow(factors_mat), 500)
    plot(r_sim[s_idx, p[1]], r_sim[s_idx, p[2]],
         pch = ".", col = "steelblue",
         main = paste(p[1], "vs", p[2]), xlab = p[1], ylab = p[2])
    points(factors_mat[e_idx, p[1]], factors_mat[e_idx, p[2]],
           pch = ".", col = "darkred")
    legend("topleft", c("simulated","empirical"),
           col = c("steelblue","darkred"), pch = 20, bty = "n", cex = 0.8)
  }
}), w = 12, h = 4 * ceiling(min(6, K * (K-1) / 2) / 3))

save_png("step11b_pnl_distribution.png", quote({
  par(mar = c(4, 4, 3, 1))
  hist(scenario_PnL, breaks = 80, probability = TRUE,
       col = "grey80", border = "white",
       main = "Step 11 — Simulated portfolio P&L", xlab = "P&L (USD)")
  abline(v = quantile(scenario_PnL, 0.05, na.rm = TRUE), col = "orange",  lwd = 2)
  abline(v = quantile(scenario_PnL, 0.01, na.rm = TRUE), col = "darkred", lwd = 2)
  abline(v = 0, col = "grey40", lty = 2)
  legend("topleft", c("5% quantile (VaR 95%)", "1% quantile (VaR 99%)"),
         col = c("orange","darkred"), lty = 1, lwd = 2, bty = "n")
}), w = 10, h = 6)

# Static VaR — placed here since scenario_PnL is ready.
# VaR_norm provides a Normal-distribution benchmark; the gap shows how much
# fat tails and copula dependence add to the capital requirement.
port_ret_hist <- as.numeric(na.omit(total_pnl_hist$pnl_total[
  match(fct_dates, total_pnl_hist$Date)]))
var_static <- list()
for (a in c(0.95, 0.99)) {
  tag      <- as.integer(a * 100)
  VaR_mc   <- -as.numeric(quantile(scenario_PnL, 1 - a, na.rm = TRUE))
  tail_pnl <- scenario_PnL[scenario_PnL <= -VaR_mc]
  ES_mc    <- if (length(tail_pnl) > 0) -mean(tail_pnl) else NA_real_
  VaR_norm <- -(qnorm(1 - a) * sd(port_ret_hist) + mean(port_ret_hist))
  var_static[[paste0("VaR_", tag)]]      <- VaR_mc
  var_static[[paste0("ES_",  tag)]]      <- ES_mc
  var_static[[paste0("VaR_norm_", tag)]] <- VaR_norm
}
results$var_static <- var_static

# Step 11 output block
cat(.cor_msg, "\n")
cat("GARCH one-step-ahead forecasts:\n")
for (fct in factor_names) {
  cat(sprintf("  %-25s  mu=%+.5f  sigma=%.5f\n", fct, mu_t1[fct], sigma_t1[fct]))
}
cat(sprintf("Static VaR_95 = %.2f  ES_95 = %.2f  (Normal benchmark: %.2f)\n",
            var_static$VaR_95, var_static$ES_95, var_static$VaR_norm_95))
cat(sprintf("Static VaR_99 = %.2f  ES_99 = %.2f  (Normal benchmark: %.2f)\n",
            var_static$VaR_99, var_static$ES_99, var_static$VaR_norm_99))
cat("Step 11 complete.\n")



# Step 12 — Rolling VaR estimation
#
# Static VaR (Step 11) gives today's risk level from a full-sample fit.
# The rolling VaR tracks how risk evolves through time — essential for
# detecting risk build-up before crises. The output (results$var_rolling)
# is the direct input to Step 13 backtesting (Kupiec / Christoffersen).
# We produce the series here; we do NOT run any backtest in this file.



# Vine-only rolling engine with periodic recalibration
#
# Daily VaR is produced with step = 1. To reduce runtime while preserving
# walk-forward validity, marginal GARCH models and the Vine copula are
# re-estimated every refit_every VaR days. Between refits, mu_t is held constant
# and sigma_t is recursively updated by the fitted GARCH(1,1) equation.

if (!requireNamespace("rvinecopulib", quietly = TRUE))
  stop("Package 'rvinecopulib' is required for Vine-only rolling VaR.")

vine_family_set <- c("gaussian", "t", "clayton", "gumbel", "frank", "joe")
cat("Rolling copula engine: vine only (rvinecopulib)\n")

empirical_innov_quantile <- function(u, z) {
  u <- pmin(pmax(as.numeric(u), 1e-6), 1 - 1e-6)
  z <- as.numeric(z[is.finite(z)])
  if (length(z) < 10L) return(rep(NA_real_, length(u)))
  qs <- as.numeric(quantile(z, probs = u, type = 8, na.rm = TRUE))
  if (length(qs) > 1L && is.finite(sd(qs, na.rm = TRUE)) && sd(qs, na.rm = TRUE) > 0) {
    qs <- (qs - mean(qs, na.rm = TRUE)) / sd(qs, na.rm = TRUE)
  }
  qs
}

.extract_first_coef <- function(cf, pattern, default = 0) {
  nm <- grep(pattern, names(cf), value = TRUE)
  if (length(nm) < 1L) return(default)
  as.numeric(cf[nm[1L]])
}

.advance_sigma_state <- function(cached, t) {
  # Daily volatility recursion between refits.
  # The conditional mean mu_t is held constant between refits. The conditional
  # volatility sigma_t is updated according to the selected variance family:
  #   - sGARCH   : sigma^2 recursion
  #   - gjrGARCH : asymmetric sigma^2 recursion
  #   - apARCH   : sigma^delta recursion
  for (i in seq_len(K)) {
    fct    <- factor_names[i]
    model  <- cached$model_win[i]
    eps_t  <- as.numeric(factors_mat[t, fct]) - cached$mu_win[i]
    omega  <- cached$omega_win[i]
    alpha1 <- cached$alpha_win[i]
    beta1  <- cached$beta_win[i]
    gamma1 <- cached$gamma_win[i]
    delta  <- cached$delta_win[i]

    if (!is.finite(eps_t) || !is.finite(omega) ||
        !is.finite(alpha1) || !is.finite(beta1) || !is.finite(gamma1)) {
      next
    }

    if (identical(model, "gjrGARCH")) {
      sigma2_old <- max(cached$sigma_win[i]^2, 1e-12)
      sigma2_new <- omega +
        alpha1 * eps_t^2 +
        gamma1 * as.numeric(eps_t < 0) * eps_t^2 +
        beta1  * sigma2_old
      sigma2_new <- max(as.numeric(sigma2_new), 1e-12)
      cached$sigma_win[i]    <- sqrt(sigma2_new)
      cached$sigma2_state[i] <- sigma2_new

    } else if (identical(model, "apARCH")) {
      # APARCH(1,1): sigma_t^delta = omega + alpha*(|eps|-gamma*eps)^delta + beta*sigma_{t-1}^delta
      if (!is.finite(delta) || delta <= 0) delta <- 2
      sigma_old <- max(cached$sigma_win[i], 1e-6)
      term <- abs(eps_t) - gamma1 * eps_t
      term <- max(as.numeric(term), 1e-12)
      sigma_delta_new <- omega +
        alpha1 * term^delta +
        beta1  * sigma_old^delta
      sigma_delta_new <- max(as.numeric(sigma_delta_new), 1e-12)
      sigma_new <- sigma_delta_new^(1 / delta)
      cached$sigma_win[i]    <- max(as.numeric(sigma_new), 1e-6)
      cached$sigma2_state[i] <- cached$sigma_win[i]^2

    } else if (identical(model, "sGARCH")) {
      sigma2_old <- max(cached$sigma_win[i]^2, 1e-12)
      sigma2_new <- omega + alpha1 * eps_t^2 + beta1 * sigma2_old
      sigma2_new <- max(as.numeric(sigma2_new), 1e-12)
      cached$sigma_win[i]    <- sqrt(sigma2_new)
      cached$sigma2_state[i] <- sigma2_new

    } else {
      warning(sprintf(
        "Unsupported rolling recursion for model '%s' on factor '%s'; using sGARCH-style fallback.",
        model, fct
      ))
      sigma2_old <- max(cached$sigma_win[i]^2, 1e-12)
      sigma2_new <- omega + alpha1 * eps_t^2 + beta1 * sigma2_old
      sigma2_new <- max(as.numeric(sigma2_new), 1e-12)
      cached$sigma_win[i]    <- sqrt(sigma2_new)
      cached$sigma2_state[i] <- sigma2_new
    }
  }
  cached
}

rolling_var <- function(window = 250,
                        step = 1,
                        alpha_vec = c(0.95, 0.99),
                        N_sim = 5000,
                        refit_every = 50) {
  T_total  <- nrow(factors_mat)
  rows_out <- seq(window, T_total - 1, by = step)

  out <- data.frame(
    date         = as.Date(character()),
    VaR_95       = numeric(),
    VaR_99       = numeric(),
    realised_pnl = numeric(),
    exception_95 = logical(),
    exception_99 = logical(),
    cop_engine   = character(),
    refit_used   = logical(),
    refit_id     = integer(),
    stringsAsFactors = FALSE
  )

  cat(sprintf(
    "Rolling VaR: window=%d  step=%d  N_sim=%d  refit_every=%d  iterations=%d\n",
    window, step, N_sim, refit_every, length(rows_out)
  ))

  cached <- list(
    valid        = FALSE,
    refit_id     = 0L,
    zhat_win     = NULL,
    vine_w       = NULL,
    mu_win       = NULL,
    sigma_win    = NULL,
    omega_win    = NULL,
    alpha_win    = NULL,
    beta_win     = NULL,
    gamma_win    = NULL,
    delta_win    = NULL,
    model_win    = NULL,
    sigma2_state = NULL,
    rolling_fits = list()
  )

  for (iter in seq_along(rows_out)) {
    t       <- rows_out[iter]
    win_idx <- (t - window + 1):t
    X_win   <- factors_mat[win_idx, , drop = FALSE]

    do_refit <- (!isTRUE(cached$valid)) || ((iter - 1L) %% refit_every == 0L)

    if (iter %% 10 == 0 || do_refit) {
      cat(sprintf(
        "  iter %d/%d  t=%d  %s\n",
        iter, length(rows_out), t,
        if (do_refit) "[REFIT]" else "[reuse + model-specific GARCH recursion]"
      ))
    }

    if (do_refit) {
      zhat_win  <- matrix(NA_real_, window, K)
      sigma_win <- setNames(numeric(K), factor_names)
      mu_win    <- setNames(numeric(K), factor_names)
      omega_win <- setNames(numeric(K), factor_names)
      alpha_win <- setNames(numeric(K), factor_names)
      beta_win  <- setNames(numeric(K), factor_names)
      gamma_win <- setNames(numeric(K), factor_names)
      delta_win <- setNames(rep(NA_real_, K), factor_names)
      model_win <- setNames(character(K), factor_names)
      rfits_this_refit <- list()
      fits_ok <- TRUE

      for (i in seq_len(K)) {
        fct <- factor_names[i]
        g   <- results$garch_fit[[fct]]

        sp <- ugarchspec(
          variance.model = list(
            model              = g@model$modeldesc$vmodel,
            garchOrder         = c(g@model$modelinc["alpha"],
                                   g@model$modelinc["beta"]),
            variance.targeting = FALSE
          ),
          mean.model = list(
            armaOrder    = results$arma_order[[fct]][c(1, 3)],
            include.mean = TRUE
          ),
          distribution.model = g@model$modeldesc$distribution
        )

        fw <- tryCatch(
          ugarchfit(spec = sp, data = X_win[, fct], solver = "hybrid"),
          error = function(e) NULL
        )
        if (is.null(fw)) { fits_ok <- FALSE; break }

        fc_w <- tryCatch(ugarchforecast(fw, n.ahead = 1), error = function(e) NULL)
        if (is.null(fc_w)) { fits_ok <- FALSE; break }

        sig <- as.numeric(sigma(fc_w))
        mu  <- as.numeric(fitted(fc_w))
        if (!is.finite(sig) || !is.finite(mu)) { fits_ok <- FALSE; break }

        cf <- coef(fw)
        omega <- if ("omega" %in% names(cf)) as.numeric(cf["omega"]) else NA_real_
        alpha1 <- .extract_first_coef(cf, "^alpha", default = 0)
        beta1  <- .extract_first_coef(cf, "^beta",  default = 0)
        gamma1 <- .extract_first_coef(cf, "^gamma", default = 0)
        delta  <- if ("delta" %in% names(cf)) as.numeric(cf["delta"]) else NA_real_
        model_type <- fw@model$modeldesc$vmodel

        if (!is.finite(omega) || !is.finite(alpha1) ||
            !is.finite(beta1) || !is.finite(gamma1)) {
          fits_ok <- FALSE
          break
        }

        zhat_i <- tail(as.numeric(residuals(fw, standardize = TRUE)), window)
        if (length(zhat_i) != window || any(!is.finite(zhat_i))) {
          fits_ok <- FALSE
          break
        }

        zhat_win[, i] <- zhat_i
        sigma_win[i]  <- sig
        mu_win[i]     <- mu
        omega_win[i]  <- omega
        alpha_win[i]  <- alpha1
        beta_win[i]   <- beta1
        gamma_win[i]  <- gamma1
        delta_win[i]  <- delta
        model_win[i]  <- model_type
        rfits_this_refit[[fct]] <- fw
      }

      if (fits_ok) {
        colnames(zhat_win) <- factor_names

        U_win <- tryCatch(pobs(zhat_win), error = function(e) NULL)
        if (!is.null(U_win)) {
          colnames(U_win) <- factor_names
          vine_w <- tryCatch(
            rvinecopulib::vinecop(
              U_win,
              family_set = vine_family_set,
              par_method = "mle",
              selcrit    = "aic",
              cores      = 1L
            ),
            error = function(e) NULL
          )
        } else {
          vine_w <- NULL
        }

        if (!is.null(vine_w)) {
          cached$valid        <- TRUE
          cached$refit_id     <- cached$refit_id + 1L
          cached$zhat_win     <- zhat_win
          cached$vine_w       <- vine_w
          cached$mu_win       <- mu_win
          cached$sigma_win    <- sigma_win
          cached$omega_win    <- omega_win
          cached$alpha_win    <- alpha_win
          cached$beta_win     <- beta_win
          cached$gamma_win    <- gamma_win
          cached$delta_win    <- delta_win
          cached$model_win    <- model_win
          cached$sigma2_state <- sigma_win^2
          cached$rolling_fits[[cached$refit_id]] <- rfits_this_refit
        } else {
          fits_ok <- FALSE
        }
      }

      if (!fits_ok) {
        warning(sprintf(
          "Rolling refit failed at iter=%d, t=%d. %s",
          iter, t,
          if (isTRUE(cached$valid)) "Reusing previous valid model." else "No valid previous model; skipping date."
        ))
        if (isTRUE(cached$valid)) {
          do_refit <- FALSE
          cached <- .advance_sigma_state(cached, t)
        } else {
          next
        }
      }

    } else {
      cached <- .advance_sigma_state(cached, t)
    }

    if (!isTRUE(cached$valid)) next

    set.seed(t)
    PS <- tryCatch(rvinecopulib::rvinecop(N_sim, cached$vine_w), error = function(e) NULL)
    if (is.null(PS)) next
    colnames(PS) <- factor_names

    Zs <- matrix(NA_real_, N_sim, K, dimnames = list(NULL, factor_names))
    for (i in seq_len(K)) {
      Zs[, i] <- empirical_innov_quantile(PS[, i], cached$zhat_win[, i])
    }

    r_s <- sweep(Zs, 2, cached$sigma_win, "*") + rep(cached$mu_win, each = N_sim)
    colnames(r_s) <- factor_names

    spot_w   <- spy_lev[t]
    yield_w  <- dgs10_lev[t] / 100
    vix_w    <- vix_lev[t]
    strike_w <- straddle_df$strike_spy[t]
    tenor_w  <- max(straddle_df$tenor_years[t] - 1 / 252, 1 / 252)

    pnl_s <- map_factors_to_pnl(r_s, spot_w, yield_w, vix_w, strike_w, tenor_w)

    VaR95 <- -as.numeric(quantile(pnl_s, 0.05, na.rm = TRUE))
    VaR99 <- -as.numeric(quantile(pnl_s, 0.01, na.rm = TRUE))

    dates   <- rownames(factors_mat)
    date_t1 <- as.Date(if (!is.null(dates)) dates[t + 1] else NA)
    idx_pnl <- match(date_t1, total_pnl_hist$Date)
    ret_t1  <- if (!is.na(idx_pnl)) total_pnl_hist$pnl_total[idx_pnl] else NA_real_

    out <- rbind(out, data.frame(
      date         = date_t1,
      VaR_95       = VaR95,
      VaR_99       = VaR99,
      realised_pnl = ret_t1,
      exception_95 = if (!is.na(ret_t1)) -ret_t1 > VaR95 else NA,
      exception_99 = if (!is.na(ret_t1)) -ret_t1 > VaR99 else NA,
      cop_engine   = "vine",
      refit_used   = do_refit,
      refit_id     = cached$refit_id,
      stringsAsFactors = FALSE
    ))
  }

  attr(out, "fits")        <- cached$rolling_fits
  attr(out, "window")      <- window
  attr(out, "step")        <- step
  attr(out, "refit_every") <- refit_every
  results$rolling_refit_every <- refit_every
  out
}

# Fixed rolling-window choice.
# 250 trading days roughly correspond to one trading year. The window length is
# imposed ex ante; no pseudo window-selection exercise is run.
fixed_window <- 250
optimal_window <- fixed_window
results$fixed_window   <- fixed_window
results$optimal_window <- optimal_window

cat(sprintf("Fixed rolling window: %d trading days\n", optimal_window))
cat(sprintf("Running final daily rolling VaR backtest with fixed window=%d, step=1, refit_every=50...\n",
            optimal_window))
results$var_rolling <- rolling_var(
  window      = optimal_window,
  step        = 1,
  N_sim       = 5000,
  refit_every = 50
)

write.csv(results$var_rolling,
          file.path(TBL_DIR, "step12_rolling_var.csv"), row.names = FALSE)
cat(sprintf("Rolling complete: %d rows | exceptions: %d (95%%), %d (99%%)\n",
            nrow(results$var_rolling),
            sum(results$var_rolling$exception_95, na.rm = TRUE),
            sum(results$var_rolling$exception_99, na.rm = TRUE)))
results$rolling_engine_summary <- table(results$var_rolling$cop_engine)
cat("Copula engine counts:\n"); print(results$rolling_engine_summary)

save_png("step12a_rolling_var.png", quote({
  rv  <- results$var_rolling
  idx <- seq_len(nrow(rv))
  par(mar = c(4, 4, 3, 1))
  plot(idx, rv$realised_pnl, type = "l", col = "grey60",
       ylim = range(c(rv$realised_pnl, -rv$VaR_99), na.rm = TRUE),
       main = "Step 12 — Rolling VaR vs realised portfolio P&L",
       xlab = "Rolling window index", ylab = "P&L / -VaR")
  lines(idx, -rv$VaR_95, col = "orange",  lwd = 2)
  lines(idx, -rv$VaR_99, col = "darkred", lwd = 2)
  ex95 <- which(rv$exception_95)
  ex99 <- which(rv$exception_99)
  if (length(ex95)) points(ex95, rv$realised_pnl[ex95],
                            col = "orange",  pch = 19, cex = 0.8)
  if (length(ex99)) points(ex99, rv$realised_pnl[ex99],
                            col = "darkred", pch = 19, cex = 0.8)
  legend("bottomright",
         legend = c("realised P&L", "-VaR 95%", "-VaR 99%", "exception"),
         col = c("grey60","orange","darkred","darkred"),
         lty = c(1,1,1,NA), pch = c(NA,NA,NA,19), lwd = 2, bty = "n")
}), w = 12, h = 6)
cat("Step 12 complete.\n")


# Step 14 — Sensitivity / stress analysis
#
# No model assumption is certain. Sensitivity analysis reveals which assumptions
# drive VaR the most — these deserve the most scrutiny. We change ONE input at
# a time, hold all else fixed, recompute VaR_95/99, and report the delta vs
# the baseline. The tornado plot in step14_tornado.png summarises the results.
#
#
#base_95 <- var_static$VaR_95
#base_99 <- var_static$VaR_99
#sens_rows <- list()
#
## 14a. Copula family sensitivity — swap copula, keep GARCH forecasts fixed
#cat("Scenario 1: copula family swap...\n")
#for (nm in names(succ_cops)) {
#  if (nm == chosen_cop_name) next
#  set.seed(42)
#  PS_alt <- rCopula(N, succ_cops[[nm]]$fit@copula)
#  Zs_alt <- matrix(NA_real_, N, K)
#  for (i in seq_len(K)) {
#    qfun       <- match.fun(paste0("q", results$pit_family[[factor_names[i]]]))
#    Zs_alt[,i] <- do.call(qfun, c(list(p = PS_alt[, i]),
#                                    results$pit_params[[factor_names[i]]]))
#  }
#  pnl_alt <- rowSums(sweep(exp(sweep(Zs_alt, 2, sigma_t1, "*") +
#                                 rep(mu_t1, each = N)), 2, w, "*")) - sum(w)
#  v95 <- -as.numeric(quantile(pnl_alt, 0.05, na.rm = TRUE))
#  v99 <- -as.numeric(quantile(pnl_alt, 0.01, na.rm = TRUE))
#  sens_rows[[length(sens_rows) + 1]] <- data.frame(
#    Scenario    = paste0("Copula: ", nm),
#    VaR_95      = round(v95, 6), VaR_99 = round(v99, 6),
#    DeltaVaR_95 = round(v95 - base_95, 6),
#    DeltaVaR_99 = round(v99 - base_99, 6),
#    stringsAsFactors = FALSE)
#}
#
# 14b. Gaussian GARCH innovations — refit each factor with norm, compare VaR
#cat("Scenario 2: Gaussian GARCH innovations...\n")
#sigma_norm <- setNames(numeric(K), factor_names)
#mu_norm    <- setNames(numeric(K), factor_names)
#norm_ok    <- TRUE
#for (fct in factor_names) {
#  g  <- results$garch_fit[[fct]]
#  sp <- ugarchspec(
#    variance.model     = list(
#      model      = g@model$modeldesc$vmodel,
#      garchOrder = c(g@model$modelinc["alpha"], g@model$modelinc["beta"])),
#    mean.model         = list(
#      armaOrder    = results$arma_order[[fct]][c(1, 3)],
#      include.mean = TRUE),
#    distribution.model = "norm")
#  fn <- tryCatch(
#    ugarchfit(spec = sp, data = factors_list[[fct]], solver = "hybrid"),
#    error = function(e) NULL)
#  if (is.null(fn)) { norm_ok <- FALSE; break }
#  fc_n            <- ugarchforecast(fn, n.ahead = 1)
#  sigma_norm[fct] <- as.numeric(sigma(fc_n))
#  mu_norm[fct]    <- as.numeric(fitted(fc_n))
#}
#if (norm_ok) {
#  set.seed(42)
#  Zs_n  <- qnorm(rCopula(N, results$copula_fit@copula))
#  pnl_n <- rowSums(sweep(exp(sweep(Zs_n, 2, sigma_norm, "*") +
#                               rep(mu_norm, each = N)), 2, w, "*")) - sum(w)
#  v95 <- -as.numeric(quantile(pnl_n, 0.05, na.rm = TRUE))
#  v99 <- -as.numeric(quantile(pnl_n, 0.01, na.rm = TRUE))
#  sens_rows[[length(sens_rows) + 1]] <- data.frame(
#    Scenario    = "GARCH innov: norm",
#    VaR_95      = round(v95, 6), VaR_99 = round(v99, 6),
#    DeltaVaR_95 = round(v95 - base_95, 6),
#    DeltaVaR_99 = round(v99 - base_99, 6),
#    stringsAsFactors = FALSE)
#}

# 14c. Portfolio weights
#cat("Scenario 3: portfolio weights...\n")
#w_scenarios <- if (exists("manual_weights_scenarios") &&
#                     is.list(manual_weights_scenarios))
#  manual_weights_scenarios else
#  list("equal"        = rep(1/K, K),
#       "concentrated" = c(0.5, rep(0.5 / (K - 1), K - 1)),
#       "diversified"  = c(0.3, 0.3, rep(0.4 / max(K - 2, 1), K - 2)))
#
#for (wname in names(w_scenarios)) {
#  ww     <- w_scenarios[[wname]]; ww <- ww / sum(ww)
#  pnl_ww <- rowSums(sweep(exp(r_sim), 2, ww, "*")) - sum(ww)
#  v95 <- -as.numeric(quantile(pnl_ww, 0.05, na.rm = TRUE))
#  v99 <- -as.numeric(quantile(pnl_ww, 0.01, na.rm = TRUE))
#  sens_rows[[length(sens_rows) + 1]] <- data.frame(
#    Scenario    = paste0("Weights: ", wname),
#    VaR_95      = round(v95, 6), VaR_99 = round(v99, 6),
#    DeltaVaR_95 = round(v95 - base_95, 6),
#    DeltaVaR_99 = round(v99 - base_99, 6),
#    stringsAsFactors = FALSE)
#}

# 14d. Vine-Copula robustness check
# Fits an R-vine on the same rank-based pseudo-observations. Only the copula
# structure varies — GARCH forecasts and empirical innovation margins are identical — so
# any VaR difference isolates the structural copula assumption.
#
# rvinecopulib runs inside a separate Rscript subprocess so any C-level crash
# in the vine library does not kill the parent session.
#cat("Scenario 4: Vine-Copula robustness check...\n")
#
#vine_result <- local({
#  tmp_in   <- gsub("\\\\", "/", file.path(tempdir(), "vine_input.rds"))
#  tmp_out  <- gsub("\\\\", "/", file.path(tempdir(), "vine_output.rds"))
#  vine_rsc <- gsub("\\\\", "/", file.path(tempdir(), "run_vine.R"))
#  if (file.exists(tmp_out)) file.remove(tmp_out)
#
#  saveRDS(list(U = U_param, N = N), tmp_in)
#
#  vine_code <- sprintf('
#suppressMessages(library(rvinecopulib))
#inp <- readRDS("%s")
#U   <- inp$U; N <- inp$N
#set_num_threads(1)
#vfit <- vinecop(U,
#  family_set = c("gaussian", "t", "clayton", "gumbel", "frank", "joe"),
#  par_method = "mle", selcrit = "aic", cores = 1)
#set.seed(42)
#sim <- rvinecop(N, vfit)
#ll   <- tryCatch(as.numeric(logLik(vfit)), error = function(e) NA_real_)
#np   <- tryCatch(attr(logLik(vfit), "df"),  error = function(e) NULL)
#vc_aic <- if (!is.null(np) && !is.na(ll)) -2 * ll + 2 * np else NA_real_
#saveRDS(list(ok = TRUE, sim = sim, aic = vc_aic), "%s")
#cat("vine_ok\\n")
#', tmp_in, tmp_out)
#
#  writeLines(vine_code, vine_rsc)
#
#  rscript_exe <- file.path(R.home("bin"), "Rscript.exe")
#  if (!file.exists(rscript_exe)) rscript_exe <- "Rscript"
#  cmd <- sprintf('"%s" --vanilla "%s"', rscript_exe, vine_rsc)
#
#  cat("  Launching vine subprocess (timeout 60s)...\n")
#  ec <- system(cmd, wait = TRUE, timeout = 60,
#               ignore.stdout = FALSE, ignore.stderr = TRUE)
#
#  if (ec == 0 && file.exists(tmp_out)) {
#    res <- readRDS(tmp_out)
#    cat("  Vine subprocess completed (AIC =", round(res$aic, 1), ")\n")
#    res
#  } else {
#    cat(sprintf("  Vine subprocess failed (exit %d) — skipping vine check.\n", ec))
#    list(ok = FALSE)
#  }
#})
#
#if (isTRUE(vine_result$ok)) {
#  PS_vine <- vine_result$sim
#
#  Zs_vine <- matrix(NA_real_, N, K, dimnames = list(NULL, factor_names))
#  for (i in seq_len(K)) {
#    qfun         <- match.fun(paste0("q", results$pit_family[[factor_names[i]]]))
#    Zs_vine[, i] <- do.call(qfun, c(list(p = PS_vine[, i]),
#                                     results$pit_params[[factor_names[i]]]))
#  }
#
#  r_sim_vine <- sweep(Zs_vine, 2, sigma_t1, "*") + rep(mu_t1, each = N)
#  pnl_vine   <- rowSums(sweep(exp(r_sim_vine), 2, w, "*")) - sum(w)
#
#  v95 <- -as.numeric(quantile(pnl_vine, 0.05, na.rm = TRUE))
#  v99 <- -as.numeric(quantile(pnl_vine, 0.01, na.rm = TRUE))
#
#  results$vine_fit    <- list(aic = vine_result$aic)   # fit object lives in subprocess
#  results$vine_pnl    <- pnl_vine
#  results$vine_VaR_95 <- v95
#  results$vine_VaR_99 <- v99
#  results$vine_AIC    <- vine_result$aic
#
#  pct_95 <- 100 * (v95 - base_95) / base_95
#  pct_99 <- 100 * (v99 - base_99) / base_99
#  cat(sprintf("  Vine vs %s baseline: dVaR_95 = %+.2f%%, dVaR_99 = %+.2f%%\n",
#              chosen_cop_name, pct_95, pct_99))
#  if (max(abs(pct_95), abs(pct_99)) < 5) {
#    cat("  -> Single-copula assumption ROBUST (< 5% deviation from vine).\n")
#  } else if (max(abs(pct_95), abs(pct_99)) < 15) {
#    cat("  -> Moderate difference. Report both VaR values in the report.\n")
#  } else {
#    cat("  -> LARGE difference. Single-copula likely misspecified — consider vine.\n")
#  }
#
#  sens_rows[[length(sens_rows) + 1]] <- data.frame(
#    Scenario    = "Vine copula",
#    VaR_95      = round(v95, 6),
#    VaR_99      = round(v99, 6),
#    DeltaVaR_95 = round(v95 - base_95, 6),
#    DeltaVaR_99 = round(v99 - base_99, 6),
#    stringsAsFactors = FALSE)
#
#  save_png("step14e_vine_vs_baseline_tails.png", quote({
#    par(mar = c(4, 4, 3, 1))
#    tail_cut <- quantile(c(scenario_PnL, pnl_vine), 0.10)
#    sp_tail  <- scenario_PnL[scenario_PnL <= tail_cut]
#    vp_tail  <- pnl_vine[pnl_vine <= tail_cut]
#    xr <- range(c(sp_tail, vp_tail))
#    plot(density(sp_tail), col = "steelblue", lwd = 2, xlim = xr,
#         main = sprintf("Left-tail P&L: %s vs Vine", chosen_cop_name),
#         xlab = "P&L (left 10% tail)")
#    lines(density(vp_tail), col = "darkred", lwd = 2)
#    abline(v = -base_95, col = "steelblue", lty = 2)
#    abline(v = -v95,     col = "darkred",   lty = 2)
#    legend("topleft",
#           legend = c(chosen_cop_name, "Vine",
#                      sprintf("%s VaR_95", chosen_cop_name), "Vine VaR_95"),
#           col = c("steelblue", "darkred", "steelblue", "darkred"),
#           lty = c(1, 1, 2, 2), lwd = 2, bty = "n")
#  }), w = 10, h = 6)
#} else {
#  results$vine_fit    <- NULL
#  results$vine_pnl    <- NULL
#  results$vine_VaR_95 <- NA_real_
#  results$vine_VaR_99 <- NA_real_
#  results$vine_AIC    <- NA_real_
#}
#
## 14e. Assemble, print, save sensitivity table
#sens_tbl <- rbind(
#  data.frame(Scenario    = paste0("BASELINE (", chosen_cop_name, ")"),
#             VaR_95      = round(base_95, 6), VaR_99 = round(base_99, 6),
#             DeltaVaR_95 = 0, DeltaVaR_99 = 0, stringsAsFactors = FALSE),
#  do.call(rbind, sens_rows))
#rownames(sens_tbl) <- NULL
#cat("\nSensitivity table:\n"); print(sens_tbl, row.names = FALSE)
#write.csv(sens_tbl, file.path(TBL_DIR, "step14_sensitivity.csv"), row.names = FALSE)
#results$sensitivity_table <- sens_tbl

#save_png("step14_tornado.png", quote({
#  tbl <- sens_tbl[-1, ]   # drop baseline row
#  oi  <- order(abs(tbl$DeltaVaR_95))
#  par(mar = c(4, 14, 3, 1))
#  bcols <- ifelse(tbl$DeltaVaR_95[oi] >= 0, "darkred", "steelblue")
#  barplot(tbl$DeltaVaR_95[oi], horiz = TRUE,
#          names.arg = tbl$Scenario[oi], las = 1, col = bcols,
#          main = "Step 14 — VaR 95% sensitivity (delta vs baseline)",
#          xlab = "DeltaVaR_95")
#  abline(v = 0, lwd = 1)
#}), w = 12, h = max(5, 0.5 * nrow(sens_tbl)))
#cat("Step 14 complete.\n")
#
#
#
# Pipeline complete

cat("\nPipeline complete — Steps 9-12 + 14\n")
cat("results objects produced:\n")
cat("  results$zhat_mat         — aligned T x K standardised residuals\n")
cat("  results$U_param          — T x K parametric pseudo-observations\n")
cat("  results$U_emp            — T x K rank-based pseudo-observations\n")
cat("  results$copula_chosen    — dependence model name; fixed to Vine\n")
cat("  results$vine_copula_fit  — fitted Vine copula object used for static VaR\n")
cat("  results$scenario_PnL     — N simulated P&L values\n")
cat("  results$var_static       — VaR_95/99, ES_95/99, VaR_norm_95/99\n")
cat("  results$optimal_window   — fixed rolling window size, hard-coded to 250\n")
cat("  results$var_rolling      — data.frame (Step 13 backtesting input)\n")
cat("  results$sensitivity_table      — Step 14 scenario comparison\n")
cat("  results$vine_AIC/BIC/logLik    — Vine fit information criteria\n")
cat("  results$rolling_engine_summary — counts of Vine rolling VaR rows\n")
cat(sprintf("\nFigures -> %s/  |  Tables -> %s/\n", FIG_DIR, TBL_DIR))
cat("Step 13: source your backtesting script — it reads results$var_rolling\n")
cat("  Schema: date | VaR_95 | VaR_99 | realised_pnl | exception_95 | exception_99\n")

###################################################
# EXTENSION Plot: GARCH-Copula VaR 95% vs PnL Plot
###################################################
save_png("garch_copula_var95_vs_pnl.png", quote({
  rv <- results$var_rolling
  
  # Ensure dates are available (using the row names if it doesn't have a date column)
  if ("date" %in% colnames(rv)) {
    idx <- as.Date(rv$date)
  } else {
    idx <- as.Date(rownames(rv))
  }
  
  # The actual loss is usually defined as -PnL. VaR is typically expressed as a positive number.
  var_s <- rv$VaR_95
  actual_loss <- -rv$realised_pnl 
  ex95 <- which(rv$exception_95)
  
  # Set margins and plot layout
  par(mar = c(4, 4, 3, 1))
  
  # Set y-limits: 0 to the maximum of VaR or actual_loss, plus some padding
  ymax <- max(c(var_s, actual_loss), na.rm = TRUE) * 1.05
  ymin <- min(c(var_s, actual_loss, 0), na.rm = TRUE) * 1.05
  
  # Initialize plot
  plot(idx, var_s, type = "n", ylim = c(ymin, ymax),
       main = "GARCH-Vine-Copula Conditional VaR 95% vs Portfolio PnL",
       xlab = "Date", ylab = "USD", font.main = 2, cex.main = 1.1)
  
  # 1. Fill between (using polygon)
  # X coordinates: go forward along idx, then backward
  # Y coordinates: go forward along var_s, then backward along 0s
  polygon(c(idx, rev(idx)), c(var_s, rep(0, length(idx))), 
          col = adjustcolor("#0288D1", alpha.f = 0.12), border = NA)
  
  # 2. Plot VaR line
  lines(idx, var_s, col = "#0288D1", lwd = 2)
  
  # 3. Plot Actual Loss
  lines(idx, actual_loss, col = adjustcolor("#90A4AE", alpha.f = 0.7), lwd = 1)
  
  # 4. Plot Exceptions
  if (length(ex95) > 0) {
    points(idx[ex95], actual_loss[ex95], col = "#F44336", pch = 19, cex = 0.8)
  }
  
  # 5. Add horizontal line at 0
  abline(h = 0, col = "black", lty = 2, lwd = 1)
  
  # 6. Add legend
  N <- nrow(rv)
  rate <- length(ex95) / N * 100
  legend("topleft", 
         legend = c("GARCH-Copula 95% VaR", 
                    "PnL (Actual Loss)", 
                    sprintf("95%% Exceptions N=%d (%.2f%%)", N, rate)),
         col = c("#0288D1", "#90A4AE", "#F44336"),
         lty = c(1, 1, NA), lwd = c(2, 1, NA),
         pch = c(NA, NA, 19), 
         bty = "n", cex = 0.85)
}), w = 12, h = 6)