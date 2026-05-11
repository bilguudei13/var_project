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