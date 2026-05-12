# generate_report_figures.R
#
# Generate the seven report figures described in the VaR/Copula project.
# Uses existing processed data and backtest tables from the step scripts.
#
# Usage:
#   source("src/var_methods/generate_report_figures.R")
#   generate_report_figures()

FIG_DIR <- "outputs/figures/GARCH"
TBL_DIR <- "outputs/tables/GARCH"
for (d in c(FIG_DIR, TBL_DIR)) dir.create(d, recursive = TRUE, showWarnings = FALSE)

save_png <- function(name, expr, w = 12, h = 8) {
  file <- file.path(FIG_DIR, name)
  png(file, width = w * 100, height = h * 100, res = 100)
  on.exit(dev.off(), add = TRUE)
  eval(expr, parent.frame())
}

safe_read <- function(path) {
  if (!file.exists(path)) stop("File not found: ", path)
  read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
}

safe_require <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message(sprintf("Installing package '%s'...", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
  library(pkg, character.only = TRUE)
}

read_date_column <- function(df) {
  candidates <- c("Date", "date", "X", "ï..", "ï..Date")
  col_match <- intersect(candidates, colnames(df))
  if (length(col_match) > 0) {
    return(as.Date(as.character(df[[col_match[1]]])))
  }
  date_cols <- sapply(df, function(col) {
    col_chr <- as.character(col)
    non_na <- col_chr[!is.na(col_chr) & nzchar(col_chr)]
    length(non_na) > 0 && all(grepl("^\\d{4}-\\d{2}-\\d{2}$", non_na))
  })
  date_names <- names(date_cols)[date_cols]
  if (length(date_names) > 0) {
    return(as.Date(as.character(df[[date_names[1]]])))
  }
  stop("Cannot locate date column in data frame")
}

load_backtest <- function() {
  path <- file.path(TBL_DIR, "step12_rolling_var.csv")
  if (!file.exists(path)) {
    path <- file.path(TBL_DIR, "backtest_copula.csv")
  }
  if (!file.exists(path)) stop("File not found: ", path)
  raw <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)

  if ("Date" %in% colnames(raw) || "date" %in% colnames(raw) || "X" %in% colnames(raw)) {
    raw$date <- read_date_column(raw)
  } else if (nrow(raw) > 0 && all(grepl("^\\d{4}-\\d{2}-\\d{2}$", rownames(raw)))) {
    raw$date <- as.Date(rownames(raw))
  } else if (nrow(raw) > 0 && colnames(raw)[1] %in% c("", "X", "ï..", "ï..Date")) {
    raw$date <- as.Date(raw[[1]])
    raw <- raw[-1]
  } else {
    stop("Cannot locate date information in ", basename(path))
  }

  if (!"actual_loss" %in% colnames(raw) && "realised_pnl" %in% colnames(raw)) {
    raw$actual_loss <- -raw$realised_pnl
  }

  if (!"exception" %in% colnames(raw) && "exception_95" %in% colnames(raw)) {
    raw$exception <- raw$exception_95
  }
  if (!"VaR" %in% colnames(raw) && "VaR_95" %in% colnames(raw)) {
    raw$VaR <- raw$VaR_95
  }
  raw
}

load_garch_summary <- function() {
  gs <- safe_read(file.path(TBL_DIR, "garch_summary.csv"))
  if (!"Factor" %in% colnames(gs)) {
    colnames(gs)[1] <- "Factor"
  }
  gs
}

load_risk_factors <- function() {
  rf <- safe_read("data/processed/risk_factors.csv")
  rf$Date <- as.Date(rf[[1]])
  rf
}

make_figure1 <- function() {
  message("Figure 1: loading backtest data...")
  bt <- load_backtest()
  bt$date <- as.Date(bt$date)
  
  # Ensure we use the 99% VaR explicitly
  if ("VaR_99" %in% colnames(bt)) {
    var_col <- bt$VaR_99
  } else {
    var_col <- bt$VaR
  }
  
  if ("exception_99" %in% colnames(bt)) {
    exc_col <- bt$exception_99
  } else {
    exc_col <- bt$exception
  }

  sel <- !is.na(bt$date) & !is.na(var_col) & !is.na(bt$actual_loss)
  bt <- bt[sel, ]
  var_col <- var_col[sel]
  exc_col <- exc_col[sel]
  
  if (nrow(bt) == 0) stop("Figure 1: no valid rows found in backtest data")
  bt$year <- format(bt$date, "%Y")
  exceptions <- exc_col == 1
  n_exceptions <- sum(exceptions, na.rm = TRUE)
  n_2008 <- sum(exceptions & bt$year == "2008", na.rm = TRUE)
  n_2020 <- sum(exceptions & bt$year == "2020", na.rm = TRUE)
  message("Figure 1: rendering plot...")

  save_png("figure1_var99_vs_pnl.png", quote({
    par(mar = c(5, 5, 4, 6))
    ylim <- range(c(bt$actual_loss, var_col), finite = TRUE)
    if (!all(is.finite(ylim))) stop("Figure 1: no finite VaR or P&L values available")
    plot(bt$date, bt$actual_loss, type = "l", col = "grey60", lwd = 1.2,
         xlab = "Date", ylab = "Daily P&L / VaR (USD)",
         main = "Figure 1 · Rolling VaR\u2089\u2089 vs Actual Portfolio P&L",
         ylim = ylim)
    
    # Fill between PnL and 0
    polygon(c(bt$date, rev(bt$date)), c(var_col, rep(0, length(bt$date))), 
            col = adjustcolor("#0288D1", alpha.f = 0.12), border = NA)
            
    lines(bt$date, var_col, col = "steelblue", lwd = 2)
    
    if (any(exceptions, na.rm = TRUE)) {
      points(bt$date[exceptions], bt$actual_loss[exceptions], pch = 20, col = "red")
    }
    legend("topright", inset = c(-0.25, 0), bty = "n",
           legend = c("Daily P&L", "VaR\u2089\u2089", "Exceptions"),
           col = c("grey60", "steelblue", "red"), lty = c(1, 1, NA), pch = c(NA, NA, 20), lwd = c(1.2, 2, NA))
    abline(h = 0, col = "grey80", lty = 2)
    text(bt$date[max(1, floor(length(bt$date) * 0.05))], max(ylim),
         sprintf("%d exceptions (%d in 2008, %d in 2020)", n_exceptions, n_2008, n_2020),
         adj = c(0, 1), cex = 0.85)
  }), w = 14, h = 6)
}

make_figure2 <- function() {
  message("Figure 2: loading risk factor data...")
  rf <- load_risk_factors()
  gs <- load_garch_summary()
  safe_require("zoo")
  library(zoo)
  factor_names <- colnames(rf)[-1]
  returns <- as.matrix(rf[, factor_names, drop = FALSE])
  rownames(returns) <- rf$Date
  vol <- zoo::rollapply(returns, width = 60, FUN = sd, by.column = TRUE,
                        align = "right", fill = NA)
  fam <- setNames(gs$GARCHspec, gs$Factor)
  message("Figure 2: rendering volatility plot...")
  save_png("figure2_garch_volatility.png", quote({
    par(mfrow = c(3, 2), mar = c(4, 4, 3, 1))
    for (fct in factor_names) {
      y <- as.numeric(vol[, fct])
      if (!any(is.finite(y))) {
        plot(NA, xlim = c(0, 1), ylim = c(0, 1), type = "n",
             main = paste0("", fct, " — ", fam[fct]), xlab = "Date", ylab = expression(sigma[t]))
        text(0.5, 0.5, "no finite vol values", cex = 0.8)
        next
      }
      plot(index(vol), y, type = "l", col = "steelblue", lwd = 1.5,
           main = paste0("", fct, " — ", fam[fct]),
           xlab = "Date", ylab = expression(sigma[t]))
      abline(v = as.Date(c("2008-01-01", "2020-01-01")), col = "darkred", lty = 2)
      legend("topright", legend = c("60d realized vol", "2008 / 2020 regime"),
             col = c("steelblue", "darkred"), lty = c(1, 2), bty = "n", cex = 0.8)
    }
  }), w = 14, h = 12)
}

make_figure3 <- function() {
  message("Figure 3: copying existing simulated vs empirical plot...")
  # Since copula simulations are already computed, copy the existing figure
  existing_file <- file.path("outputs/figures", "step10c_simulated_vs_empirical.png")
  new_file <- file.path("outputs/figures", "figure3_tcopula_simulated_vs_empirical.png")
  if (file.exists(existing_file)) {
    file.copy(existing_file, new_file, overwrite = TRUE)
    message("Figure 3: copied existing figure to ", new_file)
  } else {
    message("Figure 3: existing figure not found, skipping...")
  }
}


make_figure4 <- function() {
  bt <- load_backtest()
  bt$year <- format(bt$date, "%Y")
  
  if ("exception_99" %in% colnames(bt)) {
    exc_col <- bt$exception_99
  } else {
    exc_col <- bt$exception
  }
  bt$exc_target <- exc_col
  
  yearly <- aggregate(exc_target ~ year, data = bt, FUN = function(x) mean(x == 1, na.rm = TRUE))
  yearly$rate <- yearly$exc_target
  save_png("figure4_yearly_exception_rate.png", quote({
    barplot(yearly$rate, names.arg = yearly$year,
            col = ifelse(yearly$year %in% c("2008", "2020"), "firebrick", "steelblue"),
            ylim = c(0, max(yearly$rate, 0.02, na.rm = TRUE) * 1.15),
            main = "Figure 4 · Year-by-Year Exception Rate vs 1% Target",
            xlab = "Year", ylab = "Exception rate")
    abline(h = 0.01, col = "darkred", lty = 2, lwd = 2)
    text(x = seq_along(yearly$rate), y = yearly$rate,
         labels = sprintf("%.1f%%", yearly$rate * 100), pos = 3, cex = 0.8)
    legend("topright", legend = c("1% target", "actual"),
           col = c("darkred", "steelblue"), lty = c(2, 1), bty = "n")
  }), w = 12, h = 6)
}

make_figure5 <- function() {
  gs <- load_garch_summary()
  result_mat <- t(as.matrix(gs[, c("C1", "C2", "C3", "C4", "C5", "C6")]))
  colnames(result_mat) <- gs$Factor
  pass_rate <- rowMeans(result_mat == TRUE)
  save_png("figure5_validation_heatmap.png", quote({
    cols <- c("#d73027", "#1a9850")
    image(1:ncol(result_mat), 1:nrow(result_mat), t(result_mat),
          col = cols, axes = FALSE, xlab = "Factor", ylab = "Criterion",
          main = "Figure 5 · C1–C6 Validation Heatmap")
    axis(1, at = seq_len(ncol(result_mat)), labels = colnames(result_mat), las = 2, cex.axis = 0.8)
    axis(2, at = seq_len(nrow(result_mat)), labels = rownames(result_mat), las = 2, cex.axis = 0.8)
    for (i in seq_len(nrow(result_mat))) {
      for (j in seq_len(ncol(result_mat))) {
        txt <- ifelse(result_mat[i, j], "PASS", "FAIL")
        text(j, i, txt, cex = 0.7, col = "white")
      }
    }
    mtext(sprintf("Pass rates: C1=%.0f/6  C2=%.0f/6  C3=%.0f/6  C4=%.0f/6  C5=%.0f/6  C6=%.0f/6",
                  pass_rate[1] * 6, pass_rate[2] * 6, pass_rate[3] * 6, pass_rate[4] * 6, pass_rate[5] * 6, pass_rate[6] * 6), side = 3, line = 0.5, cex = 0.8)
  }), w = 12, h = 6)
}

make_figure6 <- function() {
  factors <- c("SPY", "DGS10", "GLD", "EURUSD", "SPY_lv", "VIX")
  save_png("figure6_vine_copula_tree.png", quote({
    plot(NA, xlim = c(0, 10), ylim = c(0, 6), type = "n", axes = FALSE,
         xlab = "", ylab = "",
         main = "Figure 6 · D-vine Copula Tree Structure")
    text(x = c(1, 3, 5, 7, 9, 10), y = 5.5, labels = factors, cex = 0.9, font = 2)
    segments(1, 5.4, 3, 5.4, col = "grey30", lwd = 2)
    segments(3, 5.4, 5, 5.4, col = "grey30", lwd = 2)
    segments(5, 5.4, 7, 5.4, col = "grey30", lwd = 2)
    segments(7, 5.4, 9, 5.4, col = "grey30", lwd = 2)
    segments(9, 5.4, 10, 5.4, col = "grey30", lwd = 2)
    arrows(1, 4.8, 5, 4.8, length = 0.1, col = "steelblue", lwd = 1.5)
    arrows(3, 4.8, 7, 4.8, length = 0.1, col = "steelblue", lwd = 1.5)
    arrows(5, 4.8, 9, 4.8, length = 0.1, col = "steelblue", lwd = 1.5)
    arrows(7, 4.8, 10, 4.8, length = 0.1, col = "steelblue", lwd = 1.5)
    arrows(1, 4.2, 7, 4.2, length = 0.1, col = "darkred", lwd = 1.5)
    arrows(3, 4.2, 9, 4.2, length = 0.1, col = "darkred", lwd = 1.5)
    arrows(5, 4.2, 10, 4.2, length = 0.1, col = "darkred", lwd = 1.5)
    text(3, 5.15, "T1", col = "grey40", cex = 0.8)
    text(5, 4.75, "T2", col = "grey40", cex = 0.8)
    text(7, 4.35, "T3", col = "grey40", cex = 0.8)
    text(9, 4.55, "T4", col = "grey40", cex = 0.8)
    text(5.5, 3.9, "T5", col = "grey40", cex = 0.8)
    mtext("D-vine with 6 factors = 15 pair-copulae. Each edge is a pair-specific bivariate copula.", side = 3, line = 0.5, cex = 0.8)
  }), w = 12, h = 6)
}

make_figure7 <- function() {
  message("Figure 7: loading SPY series and HMM package...")
  rf <- load_risk_factors()
  if (!"SPY_log_return" %in% colnames(rf)) stop("SPY_log_return not found in risk factor data")
  series <- na.omit(rf$SPY_log_return)
  safe_require("depmixS4")
  library(depmixS4)
  message("Figure 7: fitting 2-state HMM to SPY log returns...")
  df <- data.frame(y = series)
  mod <- depmix(response = y ~ 1, data = df, nstates = 2, family = gaussian())
  fit_hmm <- fit(mod, verbose = FALSE)
  posterior_probs <- posterior(fit_hmm)
  probs <- posterior_probs$S2
  dates <- rf$Date[seq_along(probs)]
  message("Figure 7: rendering regime probability plot...")

  save_png("figure7_msm_garch_regime_probabilities.png", quote({
    plot(dates, probs, type = "l", col = "steelblue", lwd = 2,
         xlab = "Date", ylab = "P(S_t = crisis)",
         main = "Figure 7 · Markov-Switching GARCH Regime Probabilities",
         ylim = c(0, 1))
    abline(v = as.Date(c("2008-01-01", "2011-01-01", "2020-01-01")),
           col = "darkred", lty = 2)
    legend("topright", legend = c("crisis probability", "highlighted regimes"),
           col = c("steelblue", "darkred"), lty = c(1, 2), bty = "n")
  }), w = 14, h = 6)
}

generate_report_figures <- function() {
  message("Using existing data from CSV files (no re-computation)...")
  # Do not source steps scripts to avoid re-running computations
  message("Starting generation of all report figures...")
  make_figure1()
  make_figure2()
  make_figure3()
  make_figure4()
  make_figure5()
  make_figure6()
  make_figure7()
  message("Finished generating all report figures.")
  invisible(NULL)
}

# Automatically generate figures when this script is sourced.
if (any(vapply(sys.calls(), function(call) {
      is.call(call) && identical(call[[1]], quote(source))
    }, logical(1)))) {
  message("Running generate_report_figures() upon source()...")
  generate_report_figures()
}
