# =============================================================================
# steps_9_to_12_copula_var.R
#
# GARCH-Copula VaR Project — Phase C (Copula) + Phase D (VaR)
#
#   Step  9  — PIT: transform Ẑ_t into pseudo-observations U_t ∈ (0,1)
#   Step 10  — Copula family selection + Cramér-von-Mises GoF test
#   Step 11  — Monte Carlo simulation of portfolio P&L
#   Step 12  — Static VaR/ES + rolling VaR (input for Step 13 backtesting)
#   Step 14  — Sensitivity / stress analysis
#
# REQUIRES: source("steps_3_to_8_marginal_garch.R") first.
#   Uses from global workspace: results, factor_names, factors_mat,
#   factors_list, factor_dates, FIG_DIR, TBL_DIR, save_png(), hdr()
#
# Step 13 (Kupiec / Christoffersen / traffic light) lives in a separate
# backtesting script. This file only produces results$var_rolling for it.
# =============================================================================

library(copula)    # fitCopula, rCopula, pobs — all else inherited from Step 8

# ── Safety check ──────────────────────────────────────────────────────────────
if (!exists("results") || length(results$garch_fit) == 0)
  stop("Run source('steps_3_to_8_marginal_garch.R') before this file.")
if (!exists("FIG_DIR"))
  stop("FIG_DIR not found — source steps_3_to_8_marginal_garch.R first.")

K <- length(factor_names)


###############################################################################
# STEP 9 — PIT: Probability Integral Transform
#
# WHY: Sklar's theorem: any joint distribution = copula( marginals ).
# We already fitted marginals F̂_i in Step 8 (results$pit_family / pit_params).
# Applying U_i,t = F̂_i( Ẑ_i,t ) maps each innovation series into a
# Uniform(0,1) series — this is the PIT. The copula then models ONLY the
# dependence between the uniforms, with the marginal shape factored out.
#
# Sanity check: if U_i,t is NOT uniform, F̂_i is wrong and the copula fit
# will be compromised. We test with KS: H0: U_i ~ Uniform(0,1).
# We WANT to FAIL to reject (p > 0.05).
###############################################################################
hdr("Step 9 · PIT — pseudo-observations")

# ── 9a. Align standardised GARCH residuals to a common T × K matrix ──────────
# rugarch may drop a few leading observations for ARMA warm-up, so series
# can differ in length by a handful of rows. We truncate to the shortest.
zhat_list <- lapply(factor_names, function(fct)
  as.numeric(residuals(results$garch_fit[[fct]], standardize = TRUE)))

T_min    <- min(sapply(zhat_list, length))
zhat_mat <- do.call(cbind, lapply(zhat_list, tail, T_min))
colnames(zhat_mat) <- factor_names
results$zhat_mat   <- zhat_mat
cat(sprintf("Aligned Ẑ_t matrix: %d obs × %d factors\n", T_min, K))

# ── 9b. Parametric PIT — apply fitted marginal CDF F̂_i from Step 8 ──────────
# U_param[t,i] = F̂_i( Ẑ[t,i] ) using the gamlss p-function for pit_family.
U_param <- matrix(NA_real_, nrow = T_min, ncol = K,
                  dimnames = list(NULL, factor_names))
for (fct in factor_names) {
  pfun           <- match.fun(paste0("p", results$pit_family[[fct]]))
  U_param[, fct] <- do.call(pfun, c(list(q = zhat_mat[, fct]),
                                     results$pit_params[[fct]]))
}

# ── 9c. Non-parametric PIT — Deheuvels pseudo-observations (rank-based) ──────
# pobs() maps each column to ranks / (n+1) ∈ (0,1). No marginal assumption.
# Useful as a robustness check when fitting the copula in Step 10.
U_emp            <- pobs(zhat_mat)
results$U_param  <- U_param
results$U_emp    <- U_emp

# ── 9d. KS uniformity test per factor ────────────────────────────────────────
cat("\n─── KS test: H0: U_i ~ Uniform(0,1)  (want p > 0.05) ───\n")
for (fct in factor_names) {
  ks <- ks.test(U_param[, fct], "punif")
  cat(sprintf("  %-25s  p=%.4f  mean=%.4f  sd=%.4f  → %s\n",
              fct, ks$p.value,
              mean(U_param[, fct]), sd(U_param[, fct]),
              if (ks$p.value > 0.05) "PASS (uniform)" else
                                     "FAIL — check marginal in Step 8"))
}

# ── 9e. Plots ─────────────────────────────────────────────────────────────────
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
cat("Step 9 complete.\n")


###############################################################################
# STEP 10 — Copula family selection and Cramér-von-Mises GoF test
#
# WHY: The Gaussian copula has ZERO tail dependence — it systematically
# underestimates the probability of joint extreme losses, making it
# dangerous for VaR. The Student-t copula has symmetric tail dependence.
# Gumbel captures upper-tail co-movement; Clayton, lower-tail.
# We fit five candidates and pick the one with best AIC that is also NOT
# rejected by the Cramér-von-Mises (CvM) GoF test.
#
# CvM GoF test:
#   H0: the data come from the fitted copula.
#   We want to FAIL to reject (actual statistic INSIDE the 95% bootstrap CI).
#   Implementation: parametric bootstrap on the CvM distance between the
#   empirical copula and the simulated copula, following the procedure
#   described in the lecture notes.
###############################################################################
hdr("Step 10 · Copula selection")

# ── CvM GoF helper ────────────────────────────────────────────────────────────
# Subsamples T_sub rows (default 200) to keep O(T²) loops feasible.
cvm_gof <- function(cop_fitted, U_data, nSim = 500, T_sub = 200) {
  set.seed(42)
  idx <- sort(sample(nrow(U_data), min(T_sub, nrow(U_data))))
  U_s <- U_data[idx, ]; n_s <- nrow(U_s)

  # Empirical copula C_n(u): fraction of rows in U_mat dominated by u coord-wise
  emp_c <- function(U_mat, u) mean(apply(U_mat, 1, function(r) all(r <= u)))

  # Simulate "reference" sample once from the fitted copula
  SP1 <- rCopula(n_s, cop_fitted@copula)
  # Precompute SimulatedCopula(U_s[i,]) for all query points
  sc1 <- sapply(seq_len(n_s), function(i) emp_c(SP1, U_s[i, ]))

  # H0 null distribution: distance between two copula simulations
  H0dist <- numeric(nSim)
  for (s in seq_len(nSim)) {
    if (s %% 50 == 0) cat(sprintf("    bootstrap %d/%d\n", s, nSim))
    SP2       <- rCopula(n_s, cop_fitted@copula)
    sc2       <- sapply(seq_len(n_s), function(i) emp_c(SP2, U_s[i, ]))
    H0dist[s] <- sum((sc2 - sc1)^2)
  }

  # Actual test statistic: EmpiricalCopula vs SimulatedCopula
  ec   <- sapply(seq_len(n_s), function(i) emp_c(U_s, U_s[i, ]))
  stat <- sum((ec - sc1)^2)
  ci   <- quantile(H0dist, c(0.025, 0.975))
  list(statistic = stat,
       CI_low    = as.numeric(ci[1]),
       CI_high   = as.numeric(ci[2]),
       H0dist    = H0dist,
       reject    = stat < ci[1] || stat > ci[2])
}

# ── 10a. Fit candidate copulas ────────────────────────────────────────────────
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
    cat(sprintf("    CvM stat=%.4f  95%%CI=[%.4f,%.4f]  reject=%s\n",
                gof$statistic, gof$CI_low, gof$CI_high,
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

# ── 10b. Comparison table ─────────────────────────────────────────────────────
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
cat("\n─── Copula comparison (sorted by AIC) ───\n"); print(cop_cmp)
write.csv(cop_cmp, file.path(TBL_DIR, "step10_copula_comparison.csv"),
          row.names = FALSE)
results$copula_comparison <- cop_cmp

# Selection: lowest AIC among non-rejected; fall back if all rejected
passed_cops <- cop_cmp[!cop_cmp$Reject, ]
if (exists("manual_copula") && !is.null(manual_copula)) {
  chosen_cop_name <- manual_copula
  cat(sprintf("→ Manual override: %s\n", chosen_cop_name))
} else if (nrow(passed_cops) > 0) {
  chosen_cop_name <- passed_cops$Family[1]
  cat(sprintf("→ Auto pick (lowest AIC, CvM not rejected): %s\n", chosen_cop_name))
} else {
  chosen_cop_name <- cop_cmp$Family[1]
  cat(sprintf("→ WARNING: all copulas rejected — using lowest AIC: %s\n",
              chosen_cop_name))
}
results$copula_chosen <- chosen_cop_name
results$copula_fit    <- succ_cops[[chosen_cop_name]]$fit

# ── 10c. Plots ────────────────────────────────────────────────────────────────
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
  sim_u <- rCopula(nrow(U_param), results$copula_fit@copula)
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
# STEP 11 — Monte Carlo simulation of portfolio P&L
#
# WHY: With a fitted copula + fitted GARCH marginals, we generate N joint
# scenarios for tomorrow's factor returns:
#   1. Draw copula-dependent Uniforms U* from the fitted copula.
#   2. Invert each U*_i through the marginal quantile function Q̂_i → Ẑ*.
#   3. Scale by the one-step-ahead GARCH forecast: r*_i = μ̂_i + σ̂_i * Ẑ*_i.
#   4. Map to portfolio P&L via weights.
# The quantile of the P&L distribution gives the model-consistent VaR.
###############################################################################
hdr("Step 11 · Monte Carlo P&L simulation")

N <- 50000
set.seed(42)
cat(sprintf("Simulating %d scenarios from %s copula...\n", N, chosen_cop_name))

# ── 11a. Simulate copula-dependent Uniforms ───────────────────────────────────
PseudoSim <- rCopula(N, results$copula_fit@copula)   # N × K
colnames(PseudoSim) <- factor_names

# ── 11b. Invert through marginal quantile functions Q̂_i from Step 8 ──────────
Zstar <- matrix(NA_real_, nrow = N, ncol = K, dimnames = list(NULL, factor_names))
for (fct in factor_names) {
  qfun        <- match.fun(paste0("q", results$pit_family[[fct]]))
  Zstar[, fct] <- do.call(qfun, c(list(p = PseudoSim[, fct]),
                                   results$pit_params[[fct]]))
}

# ── 11c. GARCH one-step-ahead volatility and mean forecasts ──────────────────
sigma_t1 <- setNames(numeric(K), factor_names)
mu_t1    <- setNames(numeric(K), factor_names)
for (fct in factor_names) {
  fc           <- ugarchforecast(results$garch_fit[[fct]], n.ahead = 1)
  sigma_t1[fct] <- as.numeric(sigma(fc))
  mu_t1[fct]    <- as.numeric(fitted(fc))
  cat(sprintf("  %-25s  μ̂=%+.5f  σ̂=%.5f\n", fct, mu_t1[fct], sigma_t1[fct]))
}

# ── 11d. Simulated returns: r*_i = μ̂_i + σ̂_i × Ẑ*_i ────────────────────────
r_sim <- sweep(Zstar, 2, sigma_t1, "*") + rep(mu_t1, each = N)

# ── 11e. Portfolio mapping ────────────────────────────────────────────────────
# Default: equal weights. Override: set manual_weights before sourcing.
w <- if (exists("manual_weights") && length(manual_weights) == K)
       manual_weights / sum(manual_weights)
     else
       rep(1/K, K)
names(w) <- factor_names
cat(sprintf("Weights: %s\n",
            paste(sprintf("%s=%.3f", factor_names, w), collapse = "  ")))

# Portfolio value after one day (log-return approach; w sums to 1 EUR)
scenario_PnL   <- rowSums(sweep(exp(r_sim), 2, w, "*")) - sum(w)

results$scenario_PnL      <- scenario_PnL
results$sigma_t1          <- sigma_t1
results$mu_t1             <- mu_t1
results$portfolio_weights <- w

# ── 11f. Plots ────────────────────────────────────────────────────────────────
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
cat("Step 11 complete.\n")


###############################################################################
# STEP 12 — VaR, ES, and rolling estimation
#
# WHY: A single-shot VaR tells you today's risk level. The ROLLING VaR shows
# how that level changes through time — crucial for detecting risk build-up
# before a crisis. The rolling series (results$var_rolling) is the direct
# input to the Step 13 backtesting script (Kupiec / Christoffersen tests).
# We produce inputs; we do NOT run any backtest here.
#
# Exception: a day t+1 is an "exception" when the realised loss exceeds VaR.
# Step 13 counts exceptions and tests whether their frequency is consistent
# with the stated confidence level.
###############################################################################
hdr("Step 12 · VaR, ES, rolling estimation")

# ── 12a. Static VaR and ES (one-shot, full sample) ───────────────────────────
port_ret_hist <- as.numeric(factors_mat %*% w)   # historical portfolio log-returns
var_static    <- list()

cat("\n─── Static VaR & ES from Monte Carlo P&L ───\n")
for (a in c(0.95, 0.99)) {
  tag    <- as.integer(a * 100)
  VaR_mc <- -as.numeric(quantile(scenario_PnL, 1 - a))
  tail   <- scenario_PnL[scenario_PnL <= -VaR_mc]
  ES_mc  <- if (length(tail) > 0) -mean(tail) else NA_real_
  # Normal benchmark — helps grade model value vs naive assumption
  VaR_norm <- -(qnorm(1 - a) * sd(port_ret_hist) + mean(port_ret_hist))
  var_static[[paste0("VaR_", tag)]]      <- VaR_mc
  var_static[[paste0("ES_",  tag)]]      <- ES_mc
  var_static[[paste0("VaR_norm_", tag)]] <- VaR_norm
  cat(sprintf("  α=%d%%  VaR(MC)=%+.6f  ES(MC)=%+.6f  VaR(Normal)=%+.6f\n",
              tag, VaR_mc, ES_mc, VaR_norm))
}
results$var_static <- var_static

# ── 12b. Rolling VaR ──────────────────────────────────────────────────────────
# For each rolling window ending at row t:
#   1. Refit GARCH per factor on rows [(t-window+1):t] with the same spec.
#   2. pobs() pseudo-observations (non-parametric — faster than full Step 8 refit).
#   3. Refit copula (same family). 4. Simulate N_sim scenarios → VaR.
#   5. Record forecast date, VaR, and realised P&L for day t+1.

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

  for (iter in seq_along(rows_out)) {
    t       <- rows_out[iter]
    win_idx <- (t - window + 1):t
    X_win   <- factors_mat[win_idx, , drop = FALSE]

    if (iter %% 10 == 0)
      cat(sprintf("  iter %d/%d  t=%d\n", iter, length(rows_out), t))

    # 1. Refit GARCH per factor, extract Ẑ and one-step forecast
    zhat_win  <- matrix(NA_real_, window, K)
    sigma_win <- setNames(numeric(K), factor_names)
    mu_win    <- setNames(numeric(K), factor_names)
    fits_ok   <- TRUE
    for (i in seq_len(K)) {
      fct <- factor_names[i]
      g   <- results$garch_fit[[fct]]
      sp  <- ugarchspec(
        variance.model     = list(
          model      = g@model$modeldesc$vmodel,
          garchOrder = c(g@model$modelinc["alpha"], g@model$modelinc["beta"])),
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
    }
    if (!fits_ok) next

    # 2-3. pobs + refit copula
    U_win   <- tryCatch(pobs(zhat_win), error = function(e) NULL)
    if (is.null(U_win)) next
    cop_win <- tryCatch(
      fitCopula(results$copula_fit@copula, U_win, method = "mpl"),
      error = function(e) NULL)
    if (is.null(cop_win)) next

    # 4. Simulate and compute VaR
    set.seed(t)
    PS  <- rCopula(N_sim, cop_win@copula)
    Zs  <- matrix(NA_real_, N_sim, K)
    for (i in seq_len(K)) {
      qfun    <- match.fun(paste0("q", results$pit_family[[factor_names[i]]]))
      Zs[, i] <- do.call(qfun, c(list(p = PS[, i]),
                                   results$pit_params[[factor_names[i]]]))
    }
    r_s   <- sweep(Zs, 2, sigma_win, "*") + rep(mu_win, each = N_sim)
    pnl_s <- rowSums(sweep(exp(r_s), 2, w, "*")) - sum(w)
    VaR95 <- -as.numeric(quantile(pnl_s, 0.05))
    VaR99 <- -as.numeric(quantile(pnl_s, 0.01))

    # 5. Realised P&L on day t+1
    ret_t1 <- sum(factors_mat[t + 1, ] * w)   # weighted log-return ≈ P&L
    dates  <- rownames(factors_mat)
    out <- rbind(out, data.frame(
      date         = as.Date(if (!is.null(dates)) dates[t + 1] else NA),
      VaR_95       = VaR95, VaR_99 = VaR99,
      realised_pnl = ret_t1,
      exception_95 = -ret_t1 > VaR95,
      exception_99 = -ret_t1 > VaR99,
      stringsAsFactors = FALSE))
  }
  out
}

cat("Running rolling VaR (window=500, step=50)...\n")
results$var_rolling <- rolling_var(window = 500, step = 50)
write.csv(results$var_rolling,
          file.path(TBL_DIR, "step12_rolling_var.csv"), row.names = FALSE)
cat(sprintf("Rolling complete: %d rows | exceptions: %d (95%%), %d (99%%)\n",
            nrow(results$var_rolling),
            sum(results$var_rolling$exception_95, na.rm = TRUE),
            sum(results$var_rolling$exception_99, na.rm = TRUE)))

# ── 12c. Rolling VaR plot ─────────────────────────────────────────────────────
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
# STEP 14 — Sensitivity / stress analysis
#
# WHY: No model assumption is certain. Sensitivity analysis reveals which
# assumptions drive VaR the most (these deserve most scrutiny) and which are
# benign. Methodology: change ONE input at a time, hold all else fixed,
# recompute VaR_95/99, report the delta vs the baseline.
###############################################################################
hdr("Step 14 · Sensitivity analysis")

base_95 <- var_static$VaR_95
base_99 <- var_static$VaR_99
sens_rows <- list()

# ── 14a. Copula family sensitivity ────────────────────────────────────────────
# Swap copula family, keep everything else fixed.
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

# ── 14b. Gaussian GARCH innovations ───────────────────────────────────────────
# Refit each factor's GARCH with norm innovations, re-forecast sigma/mu.
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
  fc_n          <- ugarchforecast(fn, n.ahead = 1)
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

# ── 14c. Rolling window length ────────────────────────────────────────────────
cat("Scenario 3: rolling window 250 vs 1000...\n")
for (win in c(250, 1000)) {
  if (win >= nrow(factors_mat) - 1) next
  rv_w <- rolling_var(window = win, step = 100, N_sim = 2000)
  if (nrow(rv_w) == 0) next
  v95 <- mean(rv_w$VaR_95, na.rm = TRUE)
  v99 <- mean(rv_w$VaR_99, na.rm = TRUE)
  sens_rows[[length(sens_rows) + 1]] <- data.frame(
    Scenario    = paste0("Window: ", win),
    VaR_95      = round(v95, 6), VaR_99 = round(v99, 6),
    DeltaVaR_95 = round(v95 - base_95, 6),
    DeltaVaR_99 = round(v99 - base_99, 6),
    stringsAsFactors = FALSE)
}

# ── 14d. Portfolio weights ─────────────────────────────────────────────────────
cat("Scenario 4: portfolio weights...\n")
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

# ── 14e. Assemble, print, save ────────────────────────────────────────────────
sens_tbl <- rbind(
  data.frame(Scenario    = paste0("BASELINE (", chosen_cop_name, ")"),
             VaR_95      = round(base_95, 6), VaR_99 = round(base_99, 6),
             DeltaVaR_95 = 0, DeltaVaR_99 = 0, stringsAsFactors = FALSE),
  do.call(rbind, sens_rows))
rownames(sens_tbl) <- NULL
cat("\n─── Sensitivity table ───\n"); print(sens_tbl, row.names = FALSE)
write.csv(sens_tbl, file.path(TBL_DIR, "step14_sensitivity.csv"), row.names = FALSE)
results$sensitivity_table <- sens_tbl

save_png("step14_tornado.png", quote({
  tbl <- sens_tbl[-1, ]   # drop baseline row
  oi  <- order(abs(tbl$DeltaVaR_95))
  par(mar = c(4, 14, 3, 1))
  bcols <- ifelse(tbl$DeltaVaR_95[oi] >= 0, "darkred", "steelblue")
  barplot(tbl$DeltaVaR_95[oi], horiz = TRUE,
          names.arg = tbl$Scenario[oi], las = 1, col = bcols,
          main = "Step 14 — VaR 95% sensitivity (Δ vs baseline)",
          xlab = "ΔVaR_95")
  abline(v = 0, lwd = 1)
}), w = 12, h = max(5, 0.5 * nrow(sens_tbl)))
cat("Step 14 complete.\n")


###############################################################################
#  FINAL STATUS
###############################################################################
hdr("Pipeline complete — Steps 9-12 + 14")
cat("results$zhat_mat         — aligned T×K standardised residuals\n")
cat("results$U_param          — T×K parametric pseudo-observations\n")
cat("results$U_emp            — T×K rank-based pseudo-observations\n")
cat("results$copula_chosen    — chosen copula family name\n")
cat("results$copula_fit       — fitted copula object\n")
cat("results$scenario_PnL     — N simulated P&L values\n")
cat("results$var_static       — VaR_95/99, ES_95/99, VaR_norm_95/99\n")
cat("results$var_rolling      — data.frame (Step 13 backtesting input)\n")
cat("results$sensitivity_table — Step 14 scenario comparison\n")
cat(sprintf("\nFigures → %s/  |  Tables → %s/\n", FIG_DIR, TBL_DIR))
cat("Step 13: source your backtesting script — it reads results$var_rolling\n")
cat("  Schema: date | VaR_95 | VaR_99 | realised_pnl | exception_95 | exception_99\n")
