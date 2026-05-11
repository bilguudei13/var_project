
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
FIG_DIR <- "outputs/figures"
TBL_DIR <- "outputs/tables"
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
