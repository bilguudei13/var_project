# =============================================================================
# steps_9_to_12_copula_var.R
#
# GARCH-Copula VaR Project — Phase C (Copula) + Phase D (VaR)
#
#   Step  9  — PIT: transform Ẑ_t into pseudo-observations U_t ∈ (0,1)
#   Step 10  — Copula family selection + Cramér-von-Mises GoF test
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
# =============================================================================
Sys.setenv(RETICULATE_PYTHON = "C:/Users/benlu/AppData/Local/Programs/Python/Python313/python.exe")

suppressPackageStartupMessages(library(copula))
suppressPackageStartupMessages(library(reticulate))
# rvinecopulib is loaded inside a subprocess in Step 14d to isolate potential crashes.

# rCopula() returns an unnamed matrix; ensure_colnames() guards every call site
# that subsequently indexes by factor name strings.
ensure_colnames <- function(mat, names) {
  if (is.null(colnames(mat))) colnames(mat) <- names
  mat
}

if (!exists("results") || length(results$garch_fit) == 0)
  stop("Run source('steps_3_to_8_marginal_garch.R') before this file.")
if (!exists("FIG_DIR"))
  stop("FIG_DIR not found — source steps_3_to_8_marginal_garch.R first.")

K <- length(factor_names)

# =============================================================================
# RETICULATE SETUP — portfolio pricing bridge
# =============================================================================
reticulate::use_python(Sys.which("python3"), required = FALSE)
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
#   SPY_level_change  — USD price change
#   VIX_change        — VIX index point change (0.23 = +0.23 VIX pts); divide by 100 for sigma
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
  spot_scen  <- spot_today + r_sim[, "SPY_level_change"]
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

# 9b. Parametric PIT — apply fitted marginal CDF F_hat_i from Step 8
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
                "FAIL — check marginal in Step 8"))
}
cat("Step 9 complete.\n")


# Step 10 — Copula family selection + Cramér-von-Mises GoF test
#
# The Gaussian copula has zero tail dependence — it underestimates the
# probability of joint extreme losses. The t-copula has symmetric tail
# dependence, appropriate for multi-asset crises. Archimedean families
# (Clayton, Gumbel, Frank) impose a single-parameter structure that may
# not fit 6-dimensional mixed dependence.
#
# We fit five candidates and pick the lowest AIC that is NOT rejected by
# the CvM GoF test. CvM H0: data come from the fitted copula. One-sided
# rejection: only the upper tail of the bootstrap null distribution counts
# (McNeil, Frey & Embrechts 2015, §7.4; Genest & Remillard 2008).

# CvM GoF helper — subsamples T_sub rows to keep O(T^2) loops feasible
cvm_gof <- function(cop_fitted, U_data, nSim = 500, T_sub = 200) {
  set.seed(42)
  idx <- sort(sample(nrow(U_data), min(T_sub, nrow(U_data))))
  U_s <- U_data[idx, ]; n_s <- nrow(U_s)

  # Empirical copula C_n(u): fraction of rows dominated by u coordinate-wise
  emp_c <- function(U_mat, u) mean(apply(U_mat, 1, function(r) all(r <= u)))

  # Reference simulation from the fitted copula
  SP1 <- rCopula(n_s, cop_fitted@copula)
  sc1 <- sapply(seq_len(n_s), function(i) emp_c(SP1, U_s[i, ]))

  # H0 null distribution: CvM distance between two copula simulations
  H0dist <- numeric(nSim)
  for (s in seq_len(nSim)) {
    if (s %% 50 == 0) cat(sprintf("    bootstrap %d/%d\n", s, nSim))
    SP2       <- rCopula(n_s, cop_fitted@copula)
    sc2       <- sapply(seq_len(n_s), function(i) emp_c(SP2, U_s[i, ]))
    H0dist[s] <- sum((sc2 - sc1)^2)
  }

  # Actual test statistic: empirical copula vs simulated copula
  ec   <- sapply(seq_len(n_s), function(i) emp_c(U_s, U_s[i, ]))
  stat <- sum((ec - sc1)^2)
  ci95 <- as.numeric(quantile(H0dist, 0.95))
  list(statistic = stat,
       CI_low    = as.numeric(quantile(H0dist, 0.025)),  # informational only
       CI_high   = ci95,                                 # upper-tail rejection threshold
       H0dist    = H0dist,
       reject    = stat > ci95)
}

# 10a. Fit candidate copulas
cat("Fitting copula candidates (this may take several minutes)...\n")
cop_specs <- list(
  gaussian = normalCopula(dim = K, dispstr = "un"),
  t        = tCopula(dim = K, dispstr = "un", df.fixed = FALSE),
  gumbel   = gumbelCopula(dim = K),
  clayton  = claytonCopula(dim = K),
  frank    = frankCopula(dim = K)
)

cop_fits <- lapply(names(cop_specs), function(nm) {
  cat(sprintf("  %s...\n", nm))
  tryCatch({
    fit_c <- fitCopula(cop_specs[[nm]], U_param, method = "mpl")
    ic    <- c(logLik = as.numeric(logLik(fit_c)),
               AIC    = AIC(fit_c),
               BIC    = BIC(fit_c))
    cat(sprintf("    logLik=%.2f  AIC=%.2f  Running CvM...\n",
                ic["logLik"], ic["AIC"]))
    gof <- cvm_gof(fit_c, U_param, nSim = 500, T_sub = 200)
    cat(sprintf("    CvM stat=%.4f  95pct-threshold=%.4f  reject=%s\n",
                gof$statistic, gof$CI_high,
                if (gof$reject) "YES" else "NO"))
    list(name = nm, fit = fit_c,
         logLik = ic["logLik"], AIC = ic["AIC"], BIC = ic["BIC"],
         CvM = gof$statistic, CI_low = gof$CI_low, CI_high = gof$CI_high,
         Reject = gof$reject, H0dist = gof$H0dist, ok = TRUE)
  }, error = function(e) {
    cat(sprintf("    FAILED: %s\n", e$message))
    list(name = nm, ok = FALSE)
  })
})
names(cop_fits) <- names(cop_specs)

succ_cops <- Filter(function(x) isTRUE(x$ok), cop_fits)

# 10b. Comparison table — printed after all fits so output is consolidated
cop_cmp <- do.call(rbind, lapply(succ_cops, function(x)
  data.frame(Family    = x$name,
             logLik    = round(x$logLik, 2),
             AIC       = round(x$AIC, 2),
             BIC       = round(x$BIC, 2),
             CvM_stat  = round(x$CvM, 4),
             CvM_CIlo  = round(x$CI_low, 4),
             CvM_CIhi  = round(x$CI_high, 4),
             Reject    = x$Reject,
             Converged = TRUE,
             stringsAsFactors = FALSE)))
cop_cmp <- cop_cmp[order(cop_cmp$AIC), ]; rownames(cop_cmp) <- NULL
cat("\nCopula comparison (sorted by AIC):\n"); print(cop_cmp)
write.csv(cop_cmp, file.path(TBL_DIR, "step10_copula_comparison.csv"),
          row.names = FALSE)
results$copula_comparison <- cop_cmp

# Selection: lowest AIC among non-rejected; fall back to lowest AIC if all fail
passed_cops <- cop_cmp[!cop_cmp$Reject, ]
if (exists("manual_copula") && !is.null(manual_copula)) {
  chosen_cop_name <- manual_copula
  cat(sprintf("-> Manual override: %s\n", chosen_cop_name))
} else if (nrow(passed_cops) > 0) {
  chosen_cop_name <- passed_cops$Family[1]
  cat(sprintf("-> Selected (lowest AIC, CvM not rejected): %s\n", chosen_cop_name))
} else {
  chosen_cop_name <- cop_cmp$Family[1]
  cat(sprintf("-> WARNING: all copulas rejected — using lowest AIC: %s\n",
              chosen_cop_name))
}
results$copula_chosen <- chosen_cop_name
results$copula_fit    <- succ_cops[[chosen_cop_name]]$fit

# 10c. Plots
save_png("step10a_copula_aic.png", quote({
  oi    <- order(cop_cmp$AIC, decreasing = TRUE)
  bcols <- ifelse(cop_cmp$Family[oi] == chosen_cop_name, "darkred", "steelblue")
  par(mar = c(4, 8, 3, 1))
  barplot(cop_cmp$AIC[oi], horiz = TRUE,
          names.arg = cop_cmp$Family[oi], las = 1, col = bcols,
          main = "Step 10 — Copula AIC", xlab = "AIC (lower = better)")
  abline(v = min(cop_cmp$AIC), col = "grey60", lty = 2)
}), w = 8, h = 5)

invisible(lapply(names(succ_cops), function(nm) {
  x <- succ_cops[[nm]]
  save_png(paste0("step10b_cvm_nulldist_", nm, ".png"), quote({
    hist(x$H0dist, breaks = 30, col = "grey80", border = "white",
         main = paste0("CvM null distribution — ", nm, " copula"),
         xlab = "Bootstrap CvM statistic")
    abline(v = x$CvM,     col = "darkred", lwd = 2)
    abline(v = x$CI_low,  col = "blue",    lwd = 1, lty = 2)
    abline(v = x$CI_high, col = "blue",    lwd = 1, lty = 2)
    legend("topright", legend = c("actual stat", "95% CI"),
           col = c("darkred", "blue"), lty = c(1, 2), lwd = 2, bty = "n")
  }), w = 7, h = 5)
}))

save_png("step10c_simulated_vs_empirical.png", quote({
  set.seed(1)
  sim_u <- ensure_colnames(rCopula(nrow(U_param), results$copula_fit@copula), factor_names)
  pairs_list <- combn(factor_names, 2, simplify = FALSE)
  n_pairs <- length(pairs_list)
  nc <- min(3, n_pairs); nr <- ceiling(n_pairs / nc)
  par(mfrow = c(nr, nc), mar = c(3, 3, 2, 1))
  for (p in pairs_list) {
    idx_e <- sample(nrow(U_param), min(500, nrow(U_param)))
    plot(sim_u[sample(nrow(sim_u), 500), p[1]],
         sim_u[sample(nrow(sim_u), 500), p[2]],
         pch = ".", col = "steelblue",
         main = paste(p[1], "vs", p[2]), xlab = p[1], ylab = p[2])
    points(U_param[idx_e, p[1]], U_param[idx_e, p[2]],
           pch = ".", col = "darkred")
    legend("topleft", legend = c("simulated", "empirical"),
           col = c("steelblue", "darkred"), pch = 20, bty = "n", cex = 0.8)
  }
}), w = 12, h = 4 * ceiling(length(combn(factor_names, 2, simplify = FALSE)) / 3))
cat("Step 10 complete.\n")


###############################################################################
# Step 11 — Monte Carlo simulation of portfolio P&L
#
# With a fitted copula + fitted GARCH marginals, we generate N joint scenarios:
#   1. Draw copula-dependent Uniforms U* from the fitted copula.
#   2. Invert U*_i through the marginal quantile Q_hat_i -> Z*_i.
#   3. Scale by one-step-ahead GARCH forecast: r*_i = mu_i + sigma_i * Z*_i.
#   4. Map to portfolio P&L via the three-leg pricing bridge.
# The quantile of this P&L distribution is the model-consistent VaR.
###############################################################################

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
cat(sprintf("Simulating %d scenarios from %s copula...\n", N, chosen_cop_name))

# 11a. Simulate copula-dependent Uniforms
PseudoSim <- rCopula(N, results$copula_fit@copula)   # N x K
colnames(PseudoSim) <- factor_names

# 11b. Invert through marginal quantile functions Q_hat_i from Step 8
Zstar <- matrix(NA_real_, nrow = N, ncol = K, dimnames = list(NULL, factor_names))
for (fct in factor_names) {
  qfun        <- match.fun(paste0("q", results$pit_family[[fct]]))
  Zstar[, fct] <- do.call(qfun, c(list(p = PseudoSim[, fct]),
                                   results$pit_params[[fct]]))
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
       main = "Step 11 — Simulated portfolio P&L", xlab = "P&L (EUR)")
  abline(v = quantile(scenario_PnL, 0.05), col = "orange",  lwd = 2)
  abline(v = quantile(scenario_PnL, 0.01), col = "darkred", lwd = 2)
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
  VaR_mc   <- -as.numeric(quantile(scenario_PnL, 1 - a))
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


###############################################################################
# Step 12 — Rolling VaR estimation
#
# Static VaR (Step 11) gives today's risk level from a full-sample fit.
# The rolling VaR tracks how risk evolves through time — essential for
# detecting risk build-up before crises. The output (results$var_rolling)
# is the direct input to Step 13 backtesting (Kupiec / Christoffersen).
# We produce the series here; we do NOT run any backtest in this file.
###############################################################################

# Vine engine is preferred when rvinecopulib is installed (allows pair-specific
# tail dependence); single-copula path is the fallback or can be forced via
# options(manual_copula_engine = "single").
use_vine_rolling <- (is.null(getOption("manual_copula_engine")) ||
                     getOption("manual_copula_engine") == "vine") &&
                    requireNamespace("rvinecopulib", quietly = TRUE)
cat(sprintf("Rolling copula engine: %s\n",
            if (use_vine_rolling) "vine (rvinecopulib)" else "single parametric"))

rolling_var <- function(window = 500, step = 50,
                        alpha_vec = c(0.95, 0.99), N_sim = 5000) {
  T_total  <- nrow(factors_mat)
  rows_out <- seq(window, T_total - 1, by = step)
  out <- data.frame(date         = as.Date(character()),
                    VaR_95       = numeric(), VaR_99       = numeric(),
                    realised_pnl = numeric(),
                    exception_95 = logical(), exception_99 = logical(),
                    stringsAsFactors = FALSE)

  cat(sprintf("Rolling VaR: window=%d  step=%d  N_sim=%d  iterations=%d\n",
              window, step, N_sim, length(rows_out)))

  rolling_fits <- list()
  for (iter in seq_along(rows_out)) {
    t       <- rows_out[iter]
    win_idx <- (t - window + 1):t
    X_win   <- factors_mat[win_idx, , drop = FALSE]

    if (iter %% 10 == 0)
      cat(sprintf("  iter %d/%d  t=%d\n", iter, length(rows_out), t))

    # Refit GARCH per factor on the rolling window, extract Z_hat and forecast
    zhat_win  <- matrix(NA_real_, window, K)
    sigma_win <- setNames(numeric(K), factor_names)
    mu_win    <- setNames(numeric(K), factor_names)
    fits_ok   <- TRUE
    rfits_this_window <- list()
    for (i in seq_len(K)) {
      fct <- factor_names[i]
      g   <- results$garch_fit[[fct]]
      sp  <- ugarchspec(
        variance.model     = list(
          model              = g@model$modeldesc$vmodel,
          garchOrder         = c(g@model$modelinc["alpha"], g@model$modelinc["beta"]),
          variance.targeting = TRUE),
        mean.model         = list(
          armaOrder    = results$arma_order[[fct]][c(1, 3)],
          include.mean = TRUE),
        distribution.model = g@model$modeldesc$distribution)
      fw <- tryCatch(ugarchfit(spec = sp, data = X_win[, fct], solver = "hybrid"),
                     error = function(e) NULL)
      if (is.null(fw)) { fits_ok <- FALSE; break }
      zhat_win[, i] <- tail(as.numeric(residuals(fw, standardize = TRUE)), window)
      fc_w          <- ugarchforecast(fw, n.ahead = 1)
      sigma_win[i]  <- as.numeric(sigma(fc_w))
      mu_win[i]     <- as.numeric(fitted(fc_w))
      rfits_this_window[[factor_names[i]]] <- fw
    }
    if (!fits_ok) next
    rolling_fits[[iter]] <- rfits_this_window

    # pobs pseudo-observations for the copula fit on this window
    U_win <- tryCatch(pobs(zhat_win), error = function(e) NULL)
    if (is.null(U_win)) next
    colnames(U_win) <- factor_names

    # Copula simulation: vine (default) or single-copula fallback
    set.seed(t)
    cop_engine_used <- "single"
    if (use_vine_rolling) {
      vine_w <- tryCatch(
        rvinecopulib::vinecop(U_win,
          family_set = c("gaussian", "t", "clayton", "gumbel", "frank", "joe"),
          par_method = "mle", selcrit = "aic", cores = 1L),
        error = function(e) NULL)
      if (!is.null(vine_w)) {
        PS              <- rvinecopulib::rvinecop(N_sim, vine_w)
        cop_engine_used <- "vine"
      } else {
        cop_win <- tryCatch(
          fitCopula(results$copula_fit@copula, U_win, method = "mpl"),
          error = function(e) NULL)
        if (is.null(cop_win)) next
        PS              <- rCopula(N_sim, cop_win@copula)
        cop_engine_used <- "single_fallback"
      }
    } else {
      cop_win <- tryCatch(
        fitCopula(results$copula_fit@copula, U_win, method = "mpl"),
        error = function(e) NULL)
      if (is.null(cop_win)) next
      PS <- rCopula(N_sim, cop_win@copula)
    }
    colnames(PS) <- factor_names

    # Simulate returns -> real portfolio P&L via pricing bridge
    Zs <- matrix(NA_real_, N_sim, K)
    for (i in seq_len(K)) {
      qfun    <- match.fun(paste0("q", results$pit_family[[factor_names[i]]]))
      Zs[, i] <- do.call(qfun, c(list(p = PS[, i]),
                                   results$pit_params[[factor_names[i]]]))
    }
    r_s       <- sweep(Zs, 2, sigma_win, "*") + rep(mu_win, each = N_sim)
    colnames(r_s) <- factor_names
    spot_w    <- spy_lev[t]
    yield_w   <- dgs10_lev[t] / 100
    vix_w     <- vix_lev[t]
    strike_w  <- straddle_df$strike_spy[t]
    tenor_w   <- max(straddle_df$tenor_years[t] - 1 / 252, 1 / 252)
    pnl_s     <- map_factors_to_pnl(r_s, spot_w, yield_w, vix_w, strike_w, tenor_w)
    VaR95 <- -as.numeric(quantile(pnl_s, 0.05))
    VaR99 <- -as.numeric(quantile(pnl_s, 0.01))

    # Realised P&L on day t+1 from historical total_portfolio_pnl.csv
    dates   <- rownames(factors_mat)
    date_t1 <- as.Date(if (!is.null(dates)) dates[t + 1] else NA)
    idx_pnl <- match(date_t1, total_pnl_hist$Date)
    ret_t1  <- if (!is.na(idx_pnl)) total_pnl_hist$pnl_total[idx_pnl] else NA_real_

    out <- rbind(out, data.frame(
      date         = date_t1,
      VaR_95       = VaR95, VaR_99 = VaR99,
      realised_pnl = ret_t1,
      exception_95 = if (!is.na(ret_t1)) -ret_t1 > VaR95 else NA,
      exception_99 = if (!is.na(ret_t1)) -ret_t1 > VaR99 else NA,
      cop_engine   = cop_engine_used,
      stringsAsFactors = FALSE))
  }
  attr(out, "fits")   <- rolling_fits
  attr(out, "window") <- window
  attr(out, "step")   <- step
  out
}

# Window selection — Kupiec UC + Christoffersen CC + Tick Loss
#
# Different window lengths trade recency against parameter stability.
# We test candidates on a common out-of-sample horizon for direct comparison.
# Override: set  manual_window <- 500  before sourcing to hard-code a choice.
#
# Scoring: UC pass (3 pts) + CC pass (3 pts) + tick-loss rank (1..n_windows).
# Summed across both alpha levels; highest composite score = optimal window.

win_candidates <- c(250, 375, 500, 750, 1000)

kupiec_uc <- function(exc_vec, alpha) {
  exc_vec <- na.omit(as.logical(exc_vec))
  n <- length(exc_vec); x <- sum(exc_vec)
  p <- 1 - alpha; p_hat <- x / n
  if (x == 0 || x == n)
    return(list(violations = x, n = n, rate = p_hat, stat = NA, p = NA))
  LR <- -2 * (x * log(p / p_hat) + (n - x) * log((1 - p) / (1 - p_hat)))
  list(violations = x, n = n, rate = p_hat,
       stat = LR, p = pchisq(LR, df = 1, lower.tail = FALSE))
}

christoffersen_cc <- function(exc_vec, alpha) {
  exc_vec <- na.omit(as.logical(exc_vec)) * 1L
  T00 <- sum(exc_vec[-length(exc_vec)] == 0 & exc_vec[-1] == 0)
  T01 <- sum(exc_vec[-length(exc_vec)] == 0 & exc_vec[-1] == 1)
  T10 <- sum(exc_vec[-length(exc_vec)] == 1 & exc_vec[-1] == 0)
  T11 <- sum(exc_vec[-length(exc_vec)] == 1 & exc_vec[-1] == 1)
  pi_hat <- (T01 + T11) / max(T00 + T01 + T10 + T11, 1)
  pi01   <- if ((T00 + T01) > 0) T01 / (T00 + T01) else 0
  pi11   <- if ((T10 + T11) > 0) T11 / (T10 + T11) else 0
  log_L_ind  <- (T00 + T10) * log(max(1 - pi_hat, 1e-12)) +
                (T01 + T11) * log(max(pi_hat, 1e-12))
  log_L_full <- T00 * log(max(1 - pi01, 1e-12)) + T01 * log(max(pi01, 1e-12)) +
                T10 * log(max(1 - pi11, 1e-12)) + T11 * log(max(pi11, 1e-12))
  LR_ind <- max(-2 * (log_L_ind - log_L_full), 0)
  uc     <- kupiec_uc(exc_vec, alpha)
  LR_cc  <- (if (!is.na(uc$stat)) uc$stat else 0) + LR_ind
  list(LR_ind = LR_ind, p_ind = pchisq(LR_ind, df = 1, lower.tail = FALSE),
       LR_cc  = LR_cc,  p_cc  = pchisq(LR_cc,  df = 2, lower.tail = FALSE))
}

tick_loss <- function(pnl_vec, var_vec, alpha) {
  L <- -pnl_vec                          # realised losses (positive = bad)
  u <- L - var_vec                       # forecast error
  mean(u * (alpha - as.numeric(u < 0)), na.rm = TRUE)
}

cat("Running window selection candidates (this takes several minutes)...\n")
rv_by_window <- setNames(
  lapply(win_candidates, function(win) {
    if (win >= nrow(factors_mat) - 1) {
      cat(sprintf("  window=%d skipped (not enough data)\n", win)); return(NULL)
    }
    cat(sprintf("  window=%d ...\n", win))
    rolling_var(window = win, step = 50, N_sim = 2000)
  }),
  as.character(win_candidates))

# Restrict comparison to observations shared by all window candidates
valid_wins   <- win_candidates[!sapply(rv_by_window, is.null)]
common_start <- Reduce(max, lapply(rv_by_window[as.character(valid_wins)],
                                   function(rv) min(rv$date, na.rm = TRUE)))

eval_window <- function(rv, win) {
  rv <- rv[!is.na(rv$date) & rv$date >= common_start, ]
  if (nrow(rv) < 20) return(NULL)
  do.call(rbind, lapply(list(c("95", 0.95), c("99", 0.99)), function(cfg) {
    tag  <- cfg[1]; alph <- as.numeric(cfg[2])
    exc  <- rv[[paste0("exception_", tag)]]
    uc   <- kupiec_uc(exc, alph)
    cc   <- christoffersen_cc(exc, alph)
    tl   <- tick_loss(rv$realised_pnl, rv[[paste0("VaR_", tag)]], alph)
    data.frame(window       = win,
               alpha        = paste0(tag, "%"),
               n            = uc$n,
               violations   = uc$violations,
               viol_rate    = round(uc$rate, 4),
               nominal_rate = 1 - alph,
               UC_stat      = round(uc$stat, 3),
               UC_p         = round(uc$p, 4),
               UC_pass      = !is.na(uc$p) && uc$p > 0.05,
               CC_stat      = round(cc$LR_cc, 3),
               CC_p         = round(cc$p_cc, 4),
               CC_pass      = !is.na(cc$p_cc) && cc$p_cc > 0.05,
               Ind_p        = round(cc$p_ind, 4),
               TickLoss     = round(tl, 6),
               stringsAsFactors = FALSE)
  }))
}

win_eval_tbl <- do.call(rbind, lapply(valid_wins, function(win) {
  rv <- rv_by_window[[as.character(win)]]
  if (is.null(rv) || nrow(rv) == 0) return(NULL)
  eval_window(rv, win)
}))
rownames(win_eval_tbl) <- NULL

# Composite score
win_scored <- win_eval_tbl
win_scored$score <- 3 * win_scored$UC_pass + 3 * win_scored$CC_pass
for (a in unique(win_scored$alpha)) {
  idx <- which(win_scored$alpha == a)
  n_w <- length(idx)
  win_scored$score[idx] <- win_scored$score[idx] +
    (n_w + 1 - rank(win_scored$TickLoss[idx]))
}
win_total <- aggregate(score ~ window, data = win_scored, FUN = sum)
win_total <- win_total[order(-win_total$score), ]

if (exists("manual_window") && !is.null(manual_window)) {
  optimal_window <- manual_window
  cat(sprintf("-> Manual override: window = %d\n", optimal_window))
} else {
  optimal_window <- win_total$window[1]
  cat(sprintf("-> Optimal window: %d  (composite score = %d)\n",
              optimal_window, win_total$score[1]))
}
results$window_selection <- win_eval_tbl
results$window_scores    <- win_total
results$optimal_window   <- optimal_window

# Window selection output block
cat(sprintf("Common evaluation horizon: %s\n", common_start))
cat("Window selection (Kupiec UC + Christoffersen CC + Tick Loss):\n")
print(win_eval_tbl, row.names = FALSE)
write.csv(win_eval_tbl,
          file.path(TBL_DIR, "step12b_window_selection.csv"), row.names = FALSE)

# Plot A: composite score bar chart
save_png("step12b_window_scores.png", quote({
  oi    <- order(win_total$score)
  bcols <- ifelse(win_total$window[oi] == optimal_window, "darkred", "steelblue")
  par(mar = c(4, 6, 3, 1))
  barplot(win_total$score[oi], horiz = TRUE,
          names.arg = paste0("w=", win_total$window[oi]),
          las = 1, col = bcols,
          main = "Step 12b — Window composite score (higher = better)",
          xlab = "Score  (UC pass + CC pass + tick-loss rank)")
  abline(v = max(win_total$score), col = "grey60", lty = 2)
}), w = 8, h = 5)

# Plot B: violation rate vs nominal per window
save_png("step12b_violation_rates.png", quote({
  par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
  for (tag in c("95%", "99%")) {
    sub   <- win_eval_tbl[win_eval_tbl$alpha == tag, ]
    bcols <- ifelse(sub$UC_pass, "steelblue", "darkred")
    barplot(sub$viol_rate * 100, names.arg = paste0("w=", sub$window),
            col = bcols, las = 2,
            main = paste0("Violation rate — VaR ", tag),
            ylab = "Observed violation rate (%)")
    abline(h = sub$nominal_rate[1] * 100, col = "black", lwd = 2, lty = 2)
    legend("topright", legend = c("UC pass", "UC fail", "nominal"),
           fill = c("steelblue", "darkred", NA),
           lty = c(NA, NA, 2), lwd = c(NA, NA, 2),
           col = c(NA, NA, "black"), bty = "n", cex = 0.8)
  }
}), w = 10, h = 5)

# Plot C: tick loss per window
save_png("step12b_tick_loss.png", quote({
  par(mfrow = c(1, 2), mar = c(4, 5, 3, 1))
  for (tag in c("95%", "99%")) {
    sub   <- win_eval_tbl[win_eval_tbl$alpha == tag, ]
    oi    <- order(sub$TickLoss, decreasing = TRUE)
    bcols <- ifelse(sub$window[oi] == optimal_window, "darkred", "steelblue")
    barplot(sub$TickLoss[oi], horiz = TRUE,
            names.arg = paste0("w=", sub$window[oi]),
            las = 1, col = bcols,
            main = paste0("Tick loss — VaR ", tag, "\n(lower = better)"),
            xlab = "Mean quantile tick loss")
    abline(v = min(sub$TickLoss), col = "grey60", lty = 2)
  }
}), w = 10, h = 5)

cat(sprintf("Running main rolling VaR with optimal window=%d...\n", optimal_window))
results$var_rolling <- rolling_var(window = optimal_window, step = 50)
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


###############################################################################
# Step 14 — Sensitivity / stress analysis
#
# No model assumption is certain. Sensitivity analysis reveals which assumptions
# drive VaR the most — these deserve the most scrutiny. We change ONE input at
# a time, hold all else fixed, recompute VaR_95/99, and report the delta vs
# the baseline. The tornado plot in step14_tornado.png summarises the results.
###############################################################################

base_95 <- var_static$VaR_95
base_99 <- var_static$VaR_99
sens_rows <- list()

# 14a. Copula family sensitivity — swap copula, keep GARCH forecasts fixed
cat("Scenario 1: copula family swap...\n")
for (nm in names(succ_cops)) {
  if (nm == chosen_cop_name) next
  set.seed(42)
  PS_alt <- rCopula(N, succ_cops[[nm]]$fit@copula)
  Zs_alt <- matrix(NA_real_, N, K)
  for (i in seq_len(K)) {
    qfun       <- match.fun(paste0("q", results$pit_family[[factor_names[i]]]))
    Zs_alt[,i] <- do.call(qfun, c(list(p = PS_alt[, i]),
                                    results$pit_params[[factor_names[i]]]))
  }
  pnl_alt <- rowSums(sweep(exp(sweep(Zs_alt, 2, sigma_t1, "*") +
                                 rep(mu_t1, each = N)), 2, w, "*")) - sum(w)
  v95 <- -as.numeric(quantile(pnl_alt, 0.05))
  v99 <- -as.numeric(quantile(pnl_alt, 0.01))
  sens_rows[[length(sens_rows) + 1]] <- data.frame(
    Scenario    = paste0("Copula: ", nm),
    VaR_95      = round(v95, 6), VaR_99 = round(v99, 6),
    DeltaVaR_95 = round(v95 - base_95, 6),
    DeltaVaR_99 = round(v99 - base_99, 6),
    stringsAsFactors = FALSE)
}

# 14b. Gaussian GARCH innovations — refit each factor with norm, compare VaR
cat("Scenario 2: Gaussian GARCH innovations...\n")
sigma_norm <- setNames(numeric(K), factor_names)
mu_norm    <- setNames(numeric(K), factor_names)
norm_ok    <- TRUE
for (fct in factor_names) {
  g  <- results$garch_fit[[fct]]
  sp <- ugarchspec(
    variance.model     = list(
      model      = g@model$modeldesc$vmodel,
      garchOrder = c(g@model$modelinc["alpha"], g@model$modelinc["beta"])),
    mean.model         = list(
      armaOrder    = results$arma_order[[fct]][c(1, 3)],
      include.mean = TRUE),
    distribution.model = "norm")
  fn <- tryCatch(
    ugarchfit(spec = sp, data = factors_list[[fct]], solver = "hybrid"),
    error = function(e) NULL)
  if (is.null(fn)) { norm_ok <- FALSE; break }
  fc_n            <- ugarchforecast(fn, n.ahead = 1)
  sigma_norm[fct] <- as.numeric(sigma(fc_n))
  mu_norm[fct]    <- as.numeric(fitted(fc_n))
}
if (norm_ok) {
  set.seed(42)
  Zs_n  <- qnorm(rCopula(N, results$copula_fit@copula))
  pnl_n <- rowSums(sweep(exp(sweep(Zs_n, 2, sigma_norm, "*") +
                               rep(mu_norm, each = N)), 2, w, "*")) - sum(w)
  v95 <- -as.numeric(quantile(pnl_n, 0.05))
  v99 <- -as.numeric(quantile(pnl_n, 0.01))
  sens_rows[[length(sens_rows) + 1]] <- data.frame(
    Scenario    = "GARCH innov: norm",
    VaR_95      = round(v95, 6), VaR_99 = round(v99, 6),
    DeltaVaR_95 = round(v95 - base_95, 6),
    DeltaVaR_99 = round(v99 - base_99, 6),
    stringsAsFactors = FALSE)
}

# 14c. Portfolio weights
cat("Scenario 3: portfolio weights...\n")
w_scenarios <- if (exists("manual_weights_scenarios") &&
                     is.list(manual_weights_scenarios))
  manual_weights_scenarios else
  list("equal"        = rep(1/K, K),
       "concentrated" = c(0.5, rep(0.5 / (K - 1), K - 1)),
       "diversified"  = c(0.3, 0.3, rep(0.4 / max(K - 2, 1), K - 2)))

for (wname in names(w_scenarios)) {
  ww     <- w_scenarios[[wname]]; ww <- ww / sum(ww)
  pnl_ww <- rowSums(sweep(exp(r_sim), 2, ww, "*")) - sum(ww)
  v95 <- -as.numeric(quantile(pnl_ww, 0.05))
  v99 <- -as.numeric(quantile(pnl_ww, 0.01))
  sens_rows[[length(sens_rows) + 1]] <- data.frame(
    Scenario    = paste0("Weights: ", wname),
    VaR_95      = round(v95, 6), VaR_99 = round(v99, 6),
    DeltaVaR_95 = round(v95 - base_95, 6),
    DeltaVaR_99 = round(v99 - base_99, 6),
    stringsAsFactors = FALSE)
}

# 14d. Vine-Copula robustness check
# Fits an R-vine on the same U_param pseudo-observations. Only the copula
# structure varies — GARCH forecasts and PIT marginals are identical — so
# any VaR difference isolates the structural copula assumption.
#
# rvinecopulib runs inside a separate Rscript subprocess so any C-level crash
# in the vine library does not kill the parent session.
cat("Scenario 4: Vine-Copula robustness check...\n")

vine_result <- local({
  tmp_in   <- gsub("\\\\", "/", file.path(tempdir(), "vine_input.rds"))
  tmp_out  <- gsub("\\\\", "/", file.path(tempdir(), "vine_output.rds"))
  vine_rsc <- gsub("\\\\", "/", file.path(tempdir(), "run_vine.R"))
  if (file.exists(tmp_out)) file.remove(tmp_out)

  saveRDS(list(U = U_param, N = N), tmp_in)

  vine_code <- sprintf('
suppressMessages(library(rvinecopulib))
inp <- readRDS("%s")
U   <- inp$U; N <- inp$N
set_num_threads(1)
vfit <- vinecop(U,
  family_set = c("gaussian", "t", "clayton", "gumbel", "frank", "joe"),
  par_method = "mle", selcrit = "aic", cores = 1)
set.seed(42)
sim <- rvinecop(N, vfit)
ll   <- tryCatch(as.numeric(logLik(vfit)), error = function(e) NA_real_)
np   <- tryCatch(attr(logLik(vfit), "df"),  error = function(e) NULL)
vc_aic <- if (!is.null(np) && !is.na(ll)) -2 * ll + 2 * np else NA_real_
saveRDS(list(ok = TRUE, sim = sim, aic = vc_aic), "%s")
cat("vine_ok\\n")
', tmp_in, tmp_out)

  writeLines(vine_code, vine_rsc)

  rscript_exe <- file.path(R.home("bin"), "Rscript.exe")
  if (!file.exists(rscript_exe)) rscript_exe <- "Rscript"
  cmd <- sprintf('"%s" --vanilla "%s"', rscript_exe, vine_rsc)

  cat("  Launching vine subprocess (timeout 60s)...\n")
  ec <- system(cmd, wait = TRUE, timeout = 60,
               ignore.stdout = FALSE, ignore.stderr = TRUE)

  if (ec == 0 && file.exists(tmp_out)) {
    res <- readRDS(tmp_out)
    cat("  Vine subprocess completed (AIC =", round(res$aic, 1), ")\n")
    res
  } else {
    cat(sprintf("  Vine subprocess failed (exit %d) — skipping vine check.\n", ec))
    list(ok = FALSE)
  }
})

if (isTRUE(vine_result$ok)) {
  PS_vine <- vine_result$sim

  Zs_vine <- matrix(NA_real_, N, K, dimnames = list(NULL, factor_names))
  for (i in seq_len(K)) {
    qfun         <- match.fun(paste0("q", results$pit_family[[factor_names[i]]]))
    Zs_vine[, i] <- do.call(qfun, c(list(p = PS_vine[, i]),
                                     results$pit_params[[factor_names[i]]]))
  }

  r_sim_vine <- sweep(Zs_vine, 2, sigma_t1, "*") + rep(mu_t1, each = N)
  pnl_vine   <- rowSums(sweep(exp(r_sim_vine), 2, w, "*")) - sum(w)

  v95 <- -as.numeric(quantile(pnl_vine, 0.05))
  v99 <- -as.numeric(quantile(pnl_vine, 0.01))

  results$vine_fit    <- list(aic = vine_result$aic)   # fit object lives in subprocess
  results$vine_pnl    <- pnl_vine
  results$vine_VaR_95 <- v95
  results$vine_VaR_99 <- v99
  results$vine_AIC    <- vine_result$aic

  pct_95 <- 100 * (v95 - base_95) / base_95
  pct_99 <- 100 * (v99 - base_99) / base_99
  cat(sprintf("  Vine vs %s baseline: dVaR_95 = %+.2f%%, dVaR_99 = %+.2f%%\n",
              chosen_cop_name, pct_95, pct_99))
  if (max(abs(pct_95), abs(pct_99)) < 5) {
    cat("  -> Single-copula assumption ROBUST (< 5% deviation from vine).\n")
  } else if (max(abs(pct_95), abs(pct_99)) < 15) {
    cat("  -> Moderate difference. Report both VaR values in the report.\n")
  } else {
    cat("  -> LARGE difference. Single-copula likely misspecified — consider vine.\n")
  }

  sens_rows[[length(sens_rows) + 1]] <- data.frame(
    Scenario    = "Vine copula",
    VaR_95      = round(v95, 6),
    VaR_99      = round(v99, 6),
    DeltaVaR_95 = round(v95 - base_95, 6),
    DeltaVaR_99 = round(v99 - base_99, 6),
    stringsAsFactors = FALSE)

  save_png("step14e_vine_vs_baseline_tails.png", quote({
    par(mar = c(4, 4, 3, 1))
    tail_cut <- quantile(c(scenario_PnL, pnl_vine), 0.10)
    sp_tail  <- scenario_PnL[scenario_PnL <= tail_cut]
    vp_tail  <- pnl_vine[pnl_vine <= tail_cut]
    xr <- range(c(sp_tail, vp_tail))
    plot(density(sp_tail), col = "steelblue", lwd = 2, xlim = xr,
         main = sprintf("Left-tail P&L: %s vs Vine", chosen_cop_name),
         xlab = "P&L (left 10% tail)")
    lines(density(vp_tail), col = "darkred", lwd = 2)
    abline(v = -base_95, col = "steelblue", lty = 2)
    abline(v = -v95,     col = "darkred",   lty = 2)
    legend("topleft",
           legend = c(chosen_cop_name, "Vine",
                      sprintf("%s VaR_95", chosen_cop_name), "Vine VaR_95"),
           col = c("steelblue", "darkred", "steelblue", "darkred"),
           lty = c(1, 1, 2, 2), lwd = 2, bty = "n")
  }), w = 10, h = 6)
} else {
  results$vine_fit    <- NULL
  results$vine_pnl    <- NULL
  results$vine_VaR_95 <- NA_real_
  results$vine_VaR_99 <- NA_real_
  results$vine_AIC    <- NA_real_
}

# 14e. Assemble, print, save sensitivity table
sens_tbl <- rbind(
  data.frame(Scenario    = paste0("BASELINE (", chosen_cop_name, ")"),
             VaR_95      = round(base_95, 6), VaR_99 = round(base_99, 6),
             DeltaVaR_95 = 0, DeltaVaR_99 = 0, stringsAsFactors = FALSE),
  do.call(rbind, sens_rows))
rownames(sens_tbl) <- NULL
cat("\nSensitivity table:\n"); print(sens_tbl, row.names = FALSE)
write.csv(sens_tbl, file.path(TBL_DIR, "step14_sensitivity.csv"), row.names = FALSE)
results$sensitivity_table <- sens_tbl

save_png("step14_tornado.png", quote({
  tbl <- sens_tbl[-1, ]   # drop baseline row
  oi  <- order(abs(tbl$DeltaVaR_95))
  par(mar = c(4, 14, 3, 1))
  bcols <- ifelse(tbl$DeltaVaR_95[oi] >= 0, "darkred", "steelblue")
  barplot(tbl$DeltaVaR_95[oi], horiz = TRUE,
          names.arg = tbl$Scenario[oi], las = 1, col = bcols,
          main = "Step 14 — VaR 95% sensitivity (delta vs baseline)",
          xlab = "DeltaVaR_95")
  abline(v = 0, lwd = 1)
}), w = 12, h = max(5, 0.5 * nrow(sens_tbl)))
cat("Step 14 complete.\n")


###############################################################################
# Pipeline complete
###############################################################################
cat("\nPipeline complete — Steps 9-12 + 14\n")
cat("results objects produced:\n")
cat("  results$zhat_mat         — aligned T x K standardised residuals\n")
cat("  results$U_param          — T x K parametric pseudo-observations\n")
cat("  results$U_emp            — T x K rank-based pseudo-observations\n")
cat("  results$copula_chosen    — chosen copula family name\n")
cat("  results$copula_fit       — fitted copula object\n")
cat("  results$scenario_PnL     — N simulated P&L values\n")
cat("  results$var_static       — VaR_95/99, ES_95/99, VaR_norm_95/99\n")
cat("  results$window_selection — Kupiec/CC/TickLoss table per window candidate\n")
cat("  results$window_scores    — composite scores + optimal_window selection\n")
cat("  results$optimal_window   — statistically selected rolling window size\n")
cat("  results$var_rolling      — data.frame (Step 13 backtesting input)\n")
cat("  results$sensitivity_table      — Step 14 scenario comparison\n")
cat("  results$vine_fit               — vine AIC metadata (if subprocess succeeded)\n")
cat("  results$vine_VaR_95/99         — vine-copula VaR (robustness check)\n")
cat("  results$vine_AIC               — vine AIC (compare against single copula AIC)\n")
cat("  results$rolling_engine_summary — counts of vine/single_fallback/single windows\n")
cat(sprintf("\nFigures -> %s/  |  Tables -> %s/\n", FIG_DIR, TBL_DIR))
cat("Step 13: source your backtesting script — it reads results$var_rolling\n")
cat("  Schema: date | VaR_95 | VaR_99 | realised_pnl | exception_95 | exception_99\n")
