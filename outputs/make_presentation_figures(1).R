################################################################################
# make_presentation_figures.R
#
# Generates the 7 PNG figures for the GARCH-Copula VaR presentation.
#
# USAGE
#   setwd("path/to/var_project")
#
#   # (a) After sourcing the pipeline (uses `results` from workspace):
#   source("src/var_methods/steps_3_to_8_marginal_garch.R")
#   source("src/var_methods/steps_9_to_12_copula_var.R")
#   source("src/var_methods/make_presentation_figures.R")
#
#   # (b) Standalone (uses only CSVs on disk):
#   source("src/var_methods/make_presentation_figures.R")
#
# INPUTS
#   outputs/tables/step12_rolling_var.csv      -> sparse rolling VaR (1 / 50d)
#   data/processed/total_portfolio_pnl.csv     -> daily P&L (forward-filled VaR)
#   outputs/tables/garch_summary.csv           -> C1-C6 pass/fail
#   data/processed/risk_factors.csv            -> sigma fallback / SPY for HMM
#
# OUTPUTS
#   outputs/figures/presentation/fig1..fig7  *.png
#
# KEY FIXES VS. PREVIOUS VERSION
#   - Figures 1 & 4 now compute DAILY exceptions by forward-filling the
#     sparse (step=50) rolling VaR onto the full daily P&L history. This
#     reproduces the 4,269-obs / 51-exception backtest from the analysis doc.
#   - Figure 2 used to receive a *vector* for the `main =` arg (because
#     df$factor was a recycled column of length N); now passes a scalar
#     `as.character(df$factor[1])`. Eliminates the title-stacking artifact.
#   - All Unicode subscripts / Greek replaced with plotmath expression() so
#     the figures render correctly with Windows R's default PNG driver
#     (no more missing-glyph boxes).
#   - Figure 6 node labels moved OUTSIDE the circles + shortened so they
#     no longer get clipped.
################################################################################

# -- output directory ---------------------------------------------------------
OUT_DIR <- "outputs/figures/presentation"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# -- palette (matches the slide deck) ----------------------------------------
THEME <- list(
  navy    = "#1E3A6F",
  blue    = "#4A7BC8",
  blueL   = "#8FB4E8",
  pass    = "#2D8F5F",
  passBg  = "#D4F5DC",
  fail    = "#C53030",
  failBg  = "#FBD9D9",
  warn    = "#D97706",
  greyL   = "#D6DCE6",
  text    = "#1A1A2E",
  muted   = "#5A6478",
  crisis  = "#FFE3E3"
)

# -- helper: save_fig --------------------------------------------------------
# Try cairo PNG (better Unicode + anti-aliasing); fall back to default if not available.
.png_type <- if (capabilities("cairo")) "cairo-png" else NA
save_fig <- function(name, expr, w = 11, h = 5.5) {
  f <- file.path(OUT_DIR, name)
  if (!is.na(.png_type))
    png(filename = f, width = w * 130, height = h * 130, res = 130, type = .png_type)
  else
    png(filename = f, width = w * 130, height = h * 130, res = 130)
  on.exit(dev.off())
  par(family = "sans", col.lab = THEME$text, col.axis = THEME$muted,
      col.main = THEME$navy, bg = "white", fg = THEME$muted)
  eval(expr, envir = parent.frame())
  cat("  saved ->", f, "\n")
}

# -- input availability ------------------------------------------------------
have_results <- exists("results") && is.list(results) &&
                !is.null(results$garch_fit) && length(results$garch_fit) > 0

f_rolling   <- "outputs/tables/step12_rolling_var.csv"
f_pnl_daily <- "data/processed/total_portfolio_pnl.csv"
f_garchsum  <- "outputs/tables/garch_summary.csv"
f_factors   <- "data/processed/risk_factors.csv"

cat("Presentation figures - input availability:\n")
cat("  workspace `results` :", have_results,             "\n")
cat("  rolling VaR CSV     :", file.exists(f_rolling),   "\n")
cat("  daily PnL CSV       :", file.exists(f_pnl_daily), "\n")
cat("  GARCH summary CSV   :", file.exists(f_garchsum),  "\n")
cat("  risk factors CSV    :", file.exists(f_factors),   "\n\n")

FACTOR_NAMES_DEFAULT <- c("SPY_log_return", "DGS10_change", "GLD_log_return",
                          "EURUSD_log_return", "SPY_level_change", "VIX_change")


################################################################################
# DAILY BACKTEST CONSTRUCTION  (used by Figures 1 and 4)
#
# step12_rolling_var.csv has one VaR estimate every 50 days (the refit cadence).
# To produce a daily exception series matching the analysis report (4,269 obs,
# 51 exceptions), we forward-fill the VaR onto the daily P&L grid:
#   - For each daily P&L date, find the most recent rolling-VaR estimate.
#   - Exception <=> -pnl_total > VaR_95 (loss exceeds the threshold).
################################################################################
bt <- NULL   # full daily backtest data.frame: date, pnl, VaR_95, VaR_99, ex95, ex99
if (file.exists(f_rolling) && file.exists(f_pnl_daily)) {
  rv <- read.csv(f_rolling, stringsAsFactors = FALSE)
  rv$date <- as.Date(rv$date)
  rv <- rv[order(rv$date), ]

  pnld <- read.csv(f_pnl_daily, stringsAsFactors = FALSE)
  pnl_date_col <- if ("Date" %in% colnames(pnld)) "Date" else colnames(pnld)[1]
  pnl_val_col  <- if ("pnl_total" %in% colnames(pnld)) "pnl_total"
                  else if ("total_pnl" %in% colnames(pnld)) "total_pnl"
                  else colnames(pnld)[2]
  pnld$Date     <- as.Date(pnld[[pnl_date_col]])
  pnld$pnl      <- as.numeric(pnld[[pnl_val_col]])
  pnld <- pnld[order(pnld$Date), c("Date", "pnl")]
  pnld <- pnld[!is.na(pnld$pnl), ]

  # Use only dates >= first rolling-VaR date (no extrapolation before)
  pnld <- pnld[pnld$Date >= min(rv$date, na.rm = TRUE), ]

  # Forward-fill VaR onto daily grid (step function, no extrap left, repeat right)
  rv_num <- as.numeric(rv$date)
  pn_num <- as.numeric(pnld$Date)
  v95 <- approx(rv_num, rv$VaR_95, xout = pn_num,
                method = "constant", f = 0, rule = 2)$y
  v99 <- approx(rv_num, rv$VaR_99, xout = pn_num,
                method = "constant", f = 0, rule = 2)$y

  bt <- data.frame(
    date   = pnld$Date,
    pnl    = pnld$pnl,
    VaR_95 = v95,
    VaR_99 = v99,
    stringsAsFactors = FALSE
  )
  bt$ex95 <- -bt$pnl > bt$VaR_95
  bt$ex99 <- -bt$pnl > bt$VaR_99
  bt$year <- as.integer(format(bt$date, "%Y"))

  cat(sprintf("Daily backtest constructed: %d obs, %d exceptions (95%%), %d (99%%)\n",
              nrow(bt), sum(bt$ex95, na.rm = TRUE), sum(bt$ex99, na.rm = TRUE)))
} else {
  cat("Daily backtest: SKIPPED (missing rolling-VaR or daily-PnL CSV)\n")
}


################################################################################
# FIGURE 1 - Rolling VaR_95 vs. Daily Portfolio P&L
################################################################################
if (!is.null(bt)) {
  n_ex95 <- sum(bt$ex95, na.rm = TRUE)
  n_2008 <- sum(bt$year == 2008 & bt$ex95, na.rm = TRUE)
  n_2020 <- sum(bt$year == 2020 & bt$ex95, na.rm = TRUE)

  save_fig("fig1_rolling_var_backtest.png", quote({
    par(mar = c(3.5, 4.5, 3.0, 1.0))
    yrng <- range(c(bt$pnl, -bt$VaR_95), na.rm = TRUE) * 1.06

    plot(bt$date, bt$pnl, type = "n", ylim = yrng,
         xlab = "", ylab = "Daily P&L  (USD)",
         main = bquote("Rolling " * VaR[95] * " vs. Actual Portfolio P&L  |  "
                       * .(n_ex95) * " exceptions  (2008: " * .(n_2008)
                       * ", 2020: " * .(n_2020) * ")"),
         cex.main = 1.05, font.main = 2, cex.lab = 1.05, xaxs = "i")

    usr <- par("usr")
    rect(as.Date("2007-10-01"), usr[3], as.Date("2009-06-30"), usr[4],
         col = THEME$crisis, border = NA)
    rect(as.Date("2020-02-15"), usr[3], as.Date("2020-09-30"), usr[4],
         col = THEME$crisis, border = NA)
    text(as.Date("2008-08-15"), usr[4] * 0.92, "GFC",
         col = THEME$fail, cex = 0.85, font = 3)
    text(as.Date("2020-06-01"), usr[4] * 0.92, "COVID",
         col = THEME$fail, cex = 0.85, font = 3)

    grid(nx = NA, ny = NULL, col = THEME$greyL, lty = 3)
    abline(h = 0, col = THEME$greyL, lwd = 0.6)

    lines(bt$date, bt$pnl,    col = "grey55",   lwd = 0.6)
    lines(bt$date, -bt$VaR_95, col = THEME$blue, lwd = 1.4)

    ex_idx <- which(bt$ex95)
    if (length(ex_idx))
      points(bt$date[ex_idx], bt$pnl[ex_idx],
             col = THEME$fail, pch = 19, cex = 0.55)

    legend("bottomleft",
           legend = c("Realised daily P&L",
                      expression(paste("\u2212", VaR[95], " threshold")),
                      "Exception"),
           col    = c("grey55", THEME$blue, THEME$fail),
           lty    = c(1, 1, NA),
           pch    = c(NA, NA, 19),
           lwd    = c(0.7, 1.4, NA),
           bty = "o", bg = "white", box.col = THEME$greyL, cex = 0.85)
    box(col = THEME$muted)
  }), w = 11, h = 5.0)
}


################################################################################
# FIGURE 2 - GARCH conditional volatility sigma_t for 6 factors
#
# Bug-fix: previous version passed the recycled vector df$factor (length N)
# as the `main` argument. R's PNG driver rendered the title once per element,
# producing stacked text. Now passes scalar `df$factor[1]`.
################################################################################
sigma_panel <- NULL
if (have_results) {
  fns <- if (exists("factor_names")) factor_names else FACTOR_NAMES_DEFAULT
  dts <- if (exists("factor_dates")) as.Date(factor_dates) else NULL
  sigma_panel <- lapply(fns, function(fct) {
    g <- results$garch_fit[[fct]]
    if (is.null(g)) return(NULL)
    s <- as.numeric(rugarch::sigma(g))
    d <- if (!is.null(dts)) tail(dts, length(s)) else seq_along(s)
    list(date = d, sigma = s, factor = fct)   # list, NOT data.frame -> avoid recycling
  })
  sigma_panel <- Filter(Negate(is.null), sigma_panel)
  cat("Figure 2 - using results$garch_fit (true conditional sigma)\n")
} else if (file.exists(f_factors)) {
  raw <- read.csv(f_factors, stringsAsFactors = FALSE, check.names = FALSE)
  dts <- as.Date(raw[, 1])
  fns <- intersect(FACTOR_NAMES_DEFAULT, colnames(raw))
  roll_sd <- function(x, w = 21) {
    n <- length(x); out <- rep(NA_real_, n)
    for (i in w:n) out[i] <- sd(x[(i - w + 1):i], na.rm = TRUE)
    out
  }
  sigma_panel <- lapply(fns, function(fct)
    list(date = dts, sigma = roll_sd(raw[[fct]]), factor = fct))
  cat("Figure 2 - using 21-day rolling sd as sigma_t proxy\n")
} else {
  cat("Figure 2 - SKIPPED (no sigma source available)\n")
}

if (!is.null(sigma_panel) && length(sigma_panel) > 0) {
  save_fig("fig2_garch_conditional_sigma.png", quote({
    par(mfrow = c(2, 3), mar = c(2.6, 4.0, 2.4, 1.0), oma = c(0, 0, 2.4, 0))
    for (k in seq_along(sigma_panel)) {
      sp <- sigma_panel[[k]]
      yr <- range(sp$sigma, na.rm = TRUE)
      # IMPORTANT: scalar title only
      panel_title <- gsub("_", " ", as.character(sp$factor))[1]

      plot(sp$date, sp$sigma, type = "n", ylim = yr,
           xlab = "", ylab = "", main = "",
           xaxs = "i")
      title(main = panel_title, line = 0.5,
            cex.main = 0.95, font.main = 2, col.main = THEME$navy)
      mtext(text = expression(sigma[t]),
            side = 2, line = 2.6, las = 0, cex = 0.85, col = THEME$text)

      usr <- par("usr")
      rect(as.Date("2007-10-01"), usr[3], as.Date("2009-06-30"), usr[4],
           col = THEME$crisis, border = NA)
      rect(as.Date("2020-02-15"), usr[3], as.Date("2020-09-30"), usr[4],
           col = THEME$crisis, border = NA)
      grid(nx = NA, ny = NULL, col = THEME$greyL, lty = 3)
      lines(sp$date, sp$sigma, col = THEME$navy, lwd = 0.9)
      box(col = THEME$muted)
    }
    mtext(expression(bold("GARCH conditional volatility  ") *
                     bold(sigma[t]) *
                     bold("  -  six risk factors")),
          outer = TRUE, side = 3, line = 0.4,
          col = THEME$navy, cex = 1.15)
  }), w = 13.5, h = 7.0)
}


################################################################################
# FIGURE 3 - t-Copula simulated vs. empirical pseudo-observations
################################################################################
U_emp <- NULL; U_sim <- NULL; fn_pairs <- NULL
if (have_results && !is.null(results$U_param)) {
  U_emp <- results$U_param
  fn_pairs <- colnames(U_emp)
  if (!is.null(results$copula_fit) && requireNamespace("copula", quietly = TRUE)) {
    set.seed(42)
    U_sim <- copula::rCopula(nrow(U_emp), results$copula_fit@copula)
    if (is.null(colnames(U_sim))) colnames(U_sim) <- fn_pairs
  }
  cat("Figure 3 - using results$U_param",
      if (!is.null(U_sim)) "+ results$copula_fit" else "", "\n")
} else if (file.exists(f_factors)) {
  raw <- read.csv(f_factors, stringsAsFactors = FALSE, check.names = FALSE)
  X <- as.matrix(raw[, intersect(FACTOR_NAMES_DEFAULT, colnames(raw))])
  X <- X[complete.cases(X), , drop = FALSE]
  U_emp <- apply(X, 2, function(x) rank(x) / (length(x) + 1))
  fn_pairs <- colnames(U_emp)
  if (requireNamespace("copula", quietly = TRUE)) {
    tc <- tryCatch(
      copula::fitCopula(copula::tCopula(dim = ncol(U_emp), dispstr = "un"),
                        U_emp, method = "mpl"),
      error = function(e) NULL)
    if (!is.null(tc)) {
      set.seed(42)
      U_sim <- copula::rCopula(nrow(U_emp), tc@copula)
      colnames(U_sim) <- fn_pairs
    }
  }
  cat("Figure 3 - using pobs(risk_factors) fallback\n")
}

if (!is.null(U_emp)) {
  pairs_to_show <- list(
    c("SPY_log_return",    "VIX_change"),
    c("SPY_log_return",    "DGS10_change"),
    c("SPY_log_return",    "GLD_log_return"),
    c("EURUSD_log_return", "DGS10_change")
  )
  pairs_to_show <- Filter(function(p) all(p %in% fn_pairs), pairs_to_show)

  if (length(pairs_to_show) > 0) {
    save_fig("fig3_copula_sim_vs_empirical.png", quote({
      par(mfrow = c(2, 2), mar = c(3.8, 3.8, 2.5, 1.0), oma = c(0, 0, 2.2, 0))
      n_emp <- nrow(U_emp)
      idx_e <- sample(n_emp, min(800, n_emp))
      for (k in seq_along(pairs_to_show)) {
        p <- pairs_to_show[[k]]
        nm1 <- gsub("_", " ", p[1])
        nm2 <- gsub("_", " ", p[2])
        plot(NA, xlim = c(0, 1), ylim = c(0, 1),
             xlab = paste0("U [ ", nm1, " ]"),
             ylab = paste0("U [ ", nm2, " ]"),
             main = paste0(nm1, "  vs  ", nm2),
             cex.main = 0.95, font.main = 2, col.main = THEME$navy)
        grid(col = THEME$greyL, lty = 3)
        if (!is.null(U_sim)) {
          idx_s <- sample(nrow(U_sim), min(800, nrow(U_sim)))
          points(U_sim[idx_s, p[1]], U_sim[idx_s, p[2]],
                 col = adjustcolor(THEME$blueL, 0.45), pch = 19, cex = 0.5)
        }
        points(U_emp[idx_e, p[1]], U_emp[idx_e, p[2]],
               col = adjustcolor(THEME$navy, 0.7), pch = 20, cex = 0.55)
        if (k == 1) {
          legend("topleft",
                 legend = c("empirical U", "simulated t-copula"),
                 col    = c(THEME$navy, THEME$blueL),
                 pch    = c(20, 19),
                 bty = "o", bg = "white", box.col = THEME$greyL, cex = 0.78)
        }
        box(col = THEME$muted)
      }
      mtext("t-Copula  -  simulated vs. empirical pseudo-observations",
            outer = TRUE, side = 3, line = 0.5,
            font = 2, col = THEME$navy, cex = 1.15)
    }), w = 10, h = 8.5)
  }
}


################################################################################
# FIGURE 4 - Year-by-Year exception rate vs. 5% target  (DAILY backtest)
################################################################################
if (!is.null(bt)) {
  yrs <- sort(unique(bt$year))
  py <- data.frame(
    year  = yrs,
    n_ex  = sapply(yrs, function(y) sum(bt$ex95[bt$year == y], na.rm = TRUE)),
    n_obs = sapply(yrs, function(y) sum(!is.na(bt$ex95[bt$year == y]))),
    stringsAsFactors = FALSE)
  py$rate_pct <- 100 * py$n_ex / pmax(py$n_obs, 1)

  save_fig("fig4_yearly_exception_rates.png", quote({
    par(mar = c(3.6, 4.2, 2.8, 1.0))
    bar_col <- ifelse(py$rate_pct > 5,  THEME$fail,
              ifelse(py$rate_pct < 1,   THEME$blueL,
                                        THEME$navy))
    ymax <- max(13, max(py$rate_pct, na.rm = TRUE) * 1.20)
    bp <- barplot(py$rate_pct,
                  names.arg = py$year,
                  col = bar_col, border = "white", las = 1,
                  ylim = c(0, ymax),
                  ylab = "Exception rate  (%)",
                  main = expression(bold("Year-by-Year ") *
                                    bold(VaR[95]) *
                                    bold(" Exception Rate vs. 5% Target")),
                  cex.names = 0.85, cex.lab = 1.05, font.main = 2,
                  col.main = THEME$navy)
    abline(h = 5, col = THEME$fail, lty = 2, lwd = 1.6)
    text(par("usr")[2], 5.4, "5% target",
         pos = 2, col = THEME$fail, cex = 0.85, font = 3)
    for (i in seq_along(bp)) {
      v <- py$rate_pct[i]
      if (v > 0.05)
        text(bp[i], v + ymax * 0.025, sprintf("%.1f", v),
             cex = 0.72, col = THEME$text)
    }
    grid(nx = NA, ny = NULL, col = THEME$greyL, lty = 3)
    box(col = THEME$muted)
  }), w = 11, h = 5.0)
}


################################################################################
# FIGURE 5 - C1-C6 validation heatmap (6 factors x 6 criteria)
################################################################################
if (file.exists(f_garchsum)) {
  gs <- read.csv(f_garchsum, stringsAsFactors = FALSE)
  crit_cols <- c("C1", "C2", "C3", "C4", "C5", "C6")
  if (all(crit_cols %in% colnames(gs))) {
    M_raw <- as.matrix(gs[, crit_cols])
    M <- matrix(as.logical(as.vector(M_raw)),
                nrow = nrow(M_raw), ncol = ncol(M_raw),
                dimnames = list(gs$Factor, crit_cols))

    save_fig("fig5_c1_c6_heatmap.png", quote({
      par(mar = c(4.2, 9.5, 3.5, 6.0))
      nr <- nrow(M); nc <- ncol(M)
      plot(NA, xlim = c(0.5, nc + 0.5), ylim = c(0.5, nr + 0.5),
           xaxt = "n", yaxt = "n", xlab = "", ylab = "", bty = "n",
           main = "GARCH validation  -  C1 to C6  pass / fail per factor",
           cex.main = 1.1, font.main = 2, col.main = THEME$navy)
      for (i in seq_len(nr)) {
        for (j in seq_len(nc)) {
          pass <- isTRUE(M[i, j])
          rect(j - 0.45, nr - i + 0.55,
               j + 0.45, nr - i + 1.45,
               col    = if (pass) THEME$passBg else THEME$failBg,
               border = if (pass) THEME$pass   else THEME$fail,
               lwd = 1.4)
          text(j, nr - i + 1,
               if (pass) "v" else "x",
               col = if (pass) THEME$pass else THEME$fail,
               cex = 1.5, font = 2)
        }
      }
      axis(1, at = seq_len(nc), labels = crit_cols, las = 1,
           tick = FALSE, line = -0.4, cex.axis = 1.0, col.axis = THEME$navy)
      axis(2, at = seq(nr, 1), labels = gsub("_", " ", rownames(M)), las = 1,
           tick = FALSE, line = -0.4, cex.axis = 0.95, col.axis = THEME$text)
      col_rate <- colSums(M, na.rm = TRUE)
      mtext(side = 1, line = 2.2, at = seq_len(nc),
            text = sprintf("%d/%d", col_rate, nr),
            cex = 0.85, col = THEME$muted)
      mtext("pass rate ->", side = 1, line = 2.2, at = 0.4, adj = 1,
            cex = 0.85, col = THEME$muted, font = 3)
      legend(x = nc + 0.7, y = nr + 0.2,
             legend = c("pass", "fail"),
             fill   = c(THEME$passBg, THEME$failBg),
             border = c(THEME$pass,   THEME$fail),
             bty = "n", xpd = NA, cex = 0.95)
    }), w = 10, h = 6.0)
  }
}


################################################################################
# FIGURE 6 - Vine-Copula tree structure (schematic)
#
# Labels moved OUTSIDE the node circles + shorter names so they no longer
# get clipped.
################################################################################
save_fig("fig6_vine_copula_schematic.png", quote({
  fns <- c("SPY", "DGS10", "GLD", "EURUSD", "SPYlv", "VIX")
  K <- length(fns)
  n_trees <- K - 1   # 5

  par(mar = c(3.2, 3.5, 3.0, 2.0))
  plot(NA, xlim = c(0.3, K + 0.9), ylim = c(0.3, n_trees + 1.2),
       xaxt = "n", yaxt = "n", xlab = "", ylab = "", bty = "n",
       main = "D-Vine Copula  -  pair-specific tail dependence  (6 factors, 5 trees, 15 pair-copulae)",
       cex.main = 1.05, font.main = 2, col.main = THEME$navy)

  for (tr in seq_len(n_trees)) {
    text(0.30, n_trees - tr + 1, paste0("T", tr),
         pos = 4, col = THEME$navy, font = 2, cex = 1.0)
  }

  for (tr in seq_len(n_trees)) {
    y       <- n_trees - tr + 1
    n_nodes <- K - tr + 1
    x_pos   <- seq(1.1, K - 0.1, length.out = n_nodes)

    # Edges (lines + edge midpoint circle)
    for (k in seq_len(n_nodes - 1)) {
      lines(c(x_pos[k], x_pos[k + 1]), c(y, y),
            col = THEME$blue, lwd = 2.4 - 0.30 * tr)
      mid_x <- (x_pos[k] + x_pos[k + 1]) / 2
      points(mid_x, y, pch = 21,
             bg = "white", col = THEME$blue, cex = 0.7, lwd = 1.4)
    }

    # Nodes
    if (tr == 1) {
      nl <- fns
    } else {
      nl <- paste0("e", seq_len(n_nodes))
    }
    for (i in seq_along(x_pos)) {
      points(x_pos[i], y, pch = 21,
             bg  = if (tr == 1) THEME$navy else "white",
             col = THEME$navy, lwd = 2,
             cex = if (tr == 1) 2.6 else 2.2)
      # Label BELOW node for T1 (high-info), inside for higher trees (low-info)
      if (tr == 1) {
        text(x_pos[i], y - 0.30, nl[i],
             col = THEME$navy, cex = 0.80, font = 2)
      } else {
        text(x_pos[i], y, nl[i],
             col = THEME$navy, cex = 0.65, font = 2)
      }
    }
  }

  mtext(side = 1, line = 0.7,
        text = "Each edge = one bivariate copula  (t / Clayton / Gumbel / Frank / Joe ...) fitted independently",
        col = THEME$muted, cex = 0.85, font = 3)
}), w = 11.5, h = 6.2)


################################################################################
# FIGURE 7 - Markov-Switching GARCH: smoothed P(regime = crisis)
# Tries depmixS4 (2-state Gaussian HMM); falls back to 21-day vol-quantile.
################################################################################
spy_ret <- NULL; spy_dates <- NULL
if (have_results && !is.null(results$arma_resid[["SPY_log_return"]])) {
  spy_ret <- as.numeric(results$arma_resid[["SPY_log_return"]])
  spy_dates <- if (exists("factor_dates"))
                 tail(as.Date(factor_dates), length(spy_ret))
               else seq_along(spy_ret)
} else if (file.exists(f_factors)) {
  raw <- read.csv(f_factors, stringsAsFactors = FALSE, check.names = FALSE)
  if ("SPY_log_return" %in% colnames(raw)) {
    spy_ret   <- as.numeric(raw$SPY_log_return)
    spy_dates <- as.Date(raw[, 1])
    ok        <- !is.na(spy_ret)
    spy_ret   <- spy_ret[ok]
    spy_dates <- spy_dates[ok]
  }
}

regime_p <- NULL
if (!is.null(spy_ret) && length(spy_ret) > 100) {
  if (requireNamespace("depmixS4", quietly = TRUE)) {
    regime_p <- tryCatch({
      df_hmm <- data.frame(ret = spy_ret)
      m <- depmixS4::depmix(ret ~ 1, nstates = 2, data = df_hmm,
                            family = gaussian())
      set.seed(1)
      fm <- depmixS4::fit(m, verbose = FALSE)
      pp <- depmixS4::posterior(fm, type = "smoothing")
      state_seq <- apply(pp[, c("S1", "S2"), drop = FALSE], 1, which.max)
      v1 <- var(spy_ret[state_seq == 1])
      v2 <- var(spy_ret[state_seq == 2])
      crisis_state <- if (!is.na(v2) && !is.na(v1) && v2 > v1) 2 else 1
      pp[, paste0("S", crisis_state)]
    }, error = function(e) NULL)
    if (!is.null(regime_p))
      cat("Figure 7 - using depmixS4 2-state HMM\n")
  }
  if (is.null(regime_p)) {
    w <- 21
    rsd <- rep(NA_real_, length(spy_ret))
    for (i in w:length(spy_ret))
      rsd[i] <- sd(spy_ret[(i - w + 1):i], na.rm = TRUE)
    q50 <- quantile(rsd, 0.50, na.rm = TRUE)
    q95 <- quantile(rsd, 0.95, na.rm = TRUE)
    z   <- (rsd - q50) / max(q95 - q50, 1e-9)
    regime_p <- 1 / (1 + exp(-2.5 * z))
    regime_p[is.na(regime_p)] <- 0
    cat("Figure 7 - using 21-day vol-quantile proxy\n")
  }

  save_fig("fig7_regime_probabilities.png", quote({
    par(mar = c(3.5, 4.5, 3.0, 1.0))
    plot(spy_dates, regime_p, type = "n", ylim = c(0, 1),
         xlab = "", ylab = "P(crisis state)",
         main = "Markov-Switching GARCH  -  smoothed crisis-state probability  (proposed)",
         cex.main = 1.1, font.main = 2, col.main = THEME$navy,
         xaxs = "i", cex.lab = 1.05)
    usr <- par("usr")
    rect(as.Date("2007-10-01"), usr[3], as.Date("2009-06-30"), usr[4],
         col = THEME$crisis, border = NA)
    rect(as.Date("2011-07-15"), usr[3], as.Date("2011-12-31"), usr[4],
         col = THEME$crisis, border = NA)
    rect(as.Date("2020-02-15"), usr[3], as.Date("2020-09-30"), usr[4],
         col = THEME$crisis, border = NA)
    text(as.Date("2008-08-01"), 0.95, "GFC",
         col = THEME$fail, cex = 0.85, font = 3)
    text(as.Date("2011-09-15"), 0.95, "EU debt",
         col = THEME$fail, cex = 0.85, font = 3)
    text(as.Date("2020-05-15"), 0.95, "COVID",
         col = THEME$fail, cex = 0.85, font = 3)
    abline(h = 0.5, col = THEME$greyL, lty = 2)
    grid(nx = NA, ny = NULL, col = THEME$greyL, lty = 3)
    polygon(c(spy_dates, rev(spy_dates)),
            c(regime_p,  rep(0, length(regime_p))),
            col = adjustcolor(THEME$fail, 0.18), border = NA)
    lines(spy_dates, regime_p, col = THEME$fail, lwd = 1.4)
    legend("topleft",
           legend = c("P(crisis state)", "0.5 decision boundary"),
           col = c(THEME$fail, THEME$greyL),
           lty = c(1, 2), lwd = c(1.6, 1),
           bty = "o", bg = "white", box.col = THEME$greyL, cex = 0.85)
    box(col = THEME$muted)
  }), w = 11, h = 5.0)
}


################################################################################
cat("\nAll figures saved to:", normalizePath(OUT_DIR, mustWork = FALSE), "\n")
invisible(list.files(OUT_DIR, pattern = "\\.png$"))
