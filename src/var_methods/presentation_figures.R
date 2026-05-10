# presentation_figures.R
#
# Runs after step15_post_var_validation.R.
# Reads results list + outputs/tables/ CSVs.
# Writes 5 presentation-ready PNGs to outputs/presentation/.

# =============================================================================
# Style constants
# =============================================================================
PRES_DIR <- "outputs/presentation"
dir.create(PRES_DIR, recursive = TRUE, showWarnings = FALSE)

COL_OK      <- "#2E8B57"   # green — what works / selected
COL_BAD     <- "#C44536"   # red   — failure / crisis
COL_NEUTRAL <- "#4A6FA5"   # blue  — neutral data / baseline
COL_GREY    <- "#7A7A7A"   # grey  — secondary lines / annotations
COL_BG      <- "#F5F5F5"   # light grey — background highlights

CEX_TITLE  <- 1.6
CEX_LABEL  <- 1.3
CEX_AXIS   <- 1.2
CEX_LEGEND <- 1.1

save_pres <- function(name, expr, w = 10, h = 6) {
  path <- file.path(PRES_DIR, name)
  png(path, width = w * 300, height = h * 300, res = 300)
  par(cex.main = CEX_TITLE, cex.lab = CEX_LABEL,
      cex.axis = CEX_AXIS, mar = c(5, 5, 4, 2))
  eval(expr)
  dev.off()
  cat(sprintf("Saved: %s\n", path))
}

# =============================================================================
# Safety check + shared data prep
# =============================================================================
if (!exists("results") || is.null(results$var_rolling))
  stop("Run the full pipeline first (steps 3-12 and step15).")

rv <- results$var_rolling
rv$date <- as.Date(rv$date)
rv <- rv[!is.na(rv$date) & !is.na(rv$realised_pnl), ]

# Short labels for factor names (used in the heatmap y-axis)
short_names <- setNames(
  c("SPY ret", "DGS10", "GLD ret", "EURUSD", "SPY lev", "VIX"),
  factor_names)


# =============================================================================
# plot_hook — 2008: cumulative observed vs expected VaR exceptions
#
# The cumulative exception count climbs far above the expected 5%-line from
# September onward, isolating Lehman week as the inflection point. The gap
# at year-end (observed vs expected) is the central headline number for the
# model failure narrative.
# =============================================================================
plot_hook <- function(sc = 1) {
  yr2008 <- rv[format(rv$date, "%Y") == "2008", ]
  yr2008 <- yr2008[order(yr2008$date), ]

  if (nrow(yr2008) == 0) {
    plot.new(); title("Keine 2008 Daten im rollenden Fenster"); return(invisible())
  }

  n_days       <- nrow(yr2008)
  cum_observed <- cumsum(ifelse(is.na(yr2008$exception_95), 0L,
                                as.integer(yr2008$exception_95)))
  cum_expected <- 0.05 * seq_len(n_days)

  ylim_max <- max(cum_observed, cum_expected, na.rm = TRUE) * 1.15

  par(mar = if (sc >= 1) c(5, 5, 4, 5) else c(4, 4, 3, 4))
  plot(yr2008$date, cum_observed,
       type = "l", col = COL_BAD, lwd = 3,
       ylim = c(0, ylim_max),
       xlab = "Datum",
       ylab = "Kumulierte Exceptions",
       main = "2008 — Krise des VaR-Modells",
       xaxt = "n",
       cex.main = CEX_TITLE * sc, cex.lab = CEX_LABEL * sc, cex.axis = CEX_AXIS * sc)

  axis.Date(1, at = seq(min(yr2008$date), max(yr2008$date), by = "month"),
            format = "%b", cex.axis = CEX_AXIS * sc)

  lines(yr2008$date, cum_expected, col = COL_GREY, lwd = 2, lty = 2)

  # Lehman event: exceptions accelerate sharply from this date
  lehman_date <- as.Date("2008-09-15")
  if (lehman_date %in% yr2008$date) {
    abline(v = lehman_date, col = COL_GREY, lty = 3)
    text(lehman_date, ylim_max * 0.95, "Lehman",
         pos = 4, cex = CEX_LABEL * sc, col = COL_GREY)
  }

  # Final annotation: observed vs expected at year-end
  final_obs <- tail(cum_observed, 1)
  final_exp <- tail(cum_expected, 1)
  text(tail(yr2008$date, 1), final_obs,
       sprintf("%d Exc.\n(erw. %.0f)", final_obs, final_exp),
       pos = 2, cex = CEX_LABEL * sc * 0.9, col = COL_BAD)

  legend("topleft",
         legend = c("Beobachtete kumulierte Exceptions",
                    "Erwartung bei 5%-VaR"),
         col = c(COL_BAD, COL_GREY),
         lty = c(1, 2), lwd = c(3, 2),
         bty = "n", cex = CEX_LEGEND * sc)
}


# =============================================================================
# plot_copula_aic — copula family selection by AIC and CvM GoF
#
# Delta AIC bars (0 = best) so bar lengths are directly comparable.
# Archimedean families (Clayton, Gumbel, Frank) have such large deltas that
# bars are clipped at 2x the Gaussian delta with ">>" annotation.
# =============================================================================
plot_copula_aic <- function(sc = 1) {
  cc <- tryCatch({
    csv_p <- file.path("outputs", "tables", "step10_copula_comparison.csv")
    df <- if (file.exists(csv_p)) read.csv(csv_p, stringsAsFactors = FALSE)
          else if (!is.null(results$copula_comparison)) results$copula_comparison
          else stop("no copula data")
    df[order(df$AIC), ]
  }, error = function(e) {
    message("plot_copula_aic: ", e$message); NULL
  })

  if (is.null(cc) || nrow(cc) == 0) {
    plot.new(); title("Copula data not available"); return(invisible())
  }

  chosen <- if (!is.null(results$copula_chosen)) results$copula_chosen else cc$Family[1]

  delta_aic  <- cc$AIC - min(cc$AIC)     # 0 = best family

  # Clip extreme bars so chart is readable; annotate clipped bars with ">>"
  non_reject_delta <- delta_aic[!cc$Reject]
  clip_at <- if (length(non_reject_delta) > 1)
    max(non_reject_delta) * 2.2 else max(delta_aic) * 0.5
  plot_delta <- pmin(delta_aic, clip_at)
  clipped    <- delta_aic > clip_at

  reject_sym <- ifelse(cc$Reject, "✗", "✓")   # ✗ ✓
  bar_cols   <- ifelse(cc$Family == chosen, COL_OK,
                       ifelse(cc$Reject, COL_BAD, COL_GREY))
  fam_labels <- paste0(cc$Family, "  ", reject_sym)

  par(mar = if (sc >= 1) c(7, 9, 4, 2) else c(5, 7, 3, 1.5))
  mp <- barplot(plot_delta, horiz = TRUE, names.arg = fam_labels,
                las = 1, col = bar_cols, border = NA,
                xlim = c(0, clip_at * 1.25),
                xlab = "Delta AIC zum besten Modell (niedriger = besser)",
                main = "Copula-Familienselektion — t-Copula gewinnt klar",
                cex.main = CEX_TITLE * sc, cex.lab = CEX_LABEL * sc,
                cex.axis = CEX_AXIS * sc, cex.names = CEX_LABEL * sc)

  # AIC value labels at bar ends
  for (i in seq_along(delta_aic)) {
    label <- formatC(cc$AIC[i], format = "f", digits = 0, big.mark = ",")
    x_pos <- plot_delta[i] + clip_at * 0.03
    if (clipped[i]) label <- paste0(">> ", label)
    text(x_pos, mp[i], labels = label,
         adj = 0, cex = CEX_AXIS * sc * 0.88, col = "grey20")
  }

  # Caption with AIC difference and GoF interpretation
  t_aic <- cc$AIC[tolower(cc$Family) == "t"]
  g_aic <- cc$AIC[tolower(cc$Family) == "gaussian"]
  diff_str <- if (length(t_aic) == 1 && length(g_aic) == 1)
    sprintf("AIC-Diff. t vs. Gaussian = %.0f  |  ", abs(t_aic - g_aic)) else ""
  mtext(paste0(diff_str, "✓ = CvM GoF nicht abgelehnt  "  ,
               "✗ = abgelehnt"),
        side = 1, line = if (sc >= 1) 5.5 else 3.5,
        cex = 0.82 * sc, col = COL_GREY)
}


# =============================================================================
# plot_yearly_exceptions — annual VaR violation rates
#
# Shows whether exception frequency matches the nominal 5% target each year.
# Crisis years (2008, 2020) are shown in red; annotation gives exact rates.
# =============================================================================
plot_yearly_exceptions <- function(sc = 1) {
  rv_e <- rv[!is.na(rv$exception_95), ]
  if (nrow(rv_e) == 0) { plot.new(); title("Keine Exception-Daten"); return(invisible()) }

  rv_e$year <- as.integer(format(rv_e$date, "%Y"))
  yr_rates  <- tapply(as.numeric(rv_e$exception_95), rv_e$year, mean, na.rm = TRUE)
  years     <- as.integer(names(yr_rates))
  rates_pct <- as.numeric(yr_rates) * 100

  crisis_yrs <- c(2008, 2020)
  bar_cols   <- ifelse(years %in% crisis_yrs, COL_BAD, COL_NEUTRAL)
  ylim_top   <- max(rates_pct, 7, na.rm = TRUE) * 1.28

  par(mar = if (sc >= 1) c(5.5, 5, 4, 2) else c(4.5, 4, 3, 1.5))
  mp <- barplot(rates_pct,
                names.arg = years, las = 2, col = bar_cols, border = NA,
                ylim = c(0, ylim_top),
                xlab = "Jahr", ylab = "VaR-Verletzungsrate (%)",
                main = "Jährl. VaR-Verletzungsraten — Krisen vs. ruhige Phasen",
                cex.main = CEX_TITLE * sc, cex.lab = CEX_LABEL * sc,
                cex.axis = CEX_AXIS * sc, cex.names = CEX_AXIS * sc * 0.9)

  abline(h = 5, col = COL_GREY, lwd = 2, lty = 2)
  text(par("usr")[2] * 0.98, 5 + ylim_top * 0.04, "5% erwartet",
       adj = 1, cex = 0.9 * sc, col = COL_GREY)

  # Annotate crisis years with exact rate
  for (yr in crisis_yrs) {
    idx_y <- which(years == yr)
    if (length(idx_y) == 1 && !is.na(rates_pct[idx_y]) && rates_pct[idx_y] > 0.1)
      text(mp[idx_y], rates_pct[idx_y] + ylim_top * 0.05,
           sprintf("%.1f%%", rates_pct[idx_y]),
           cex = 1.0 * sc, font = 2, col = COL_BAD)
  }

  legend("topright",
         legend = c("Krisenjahre (2008, 2020)", "Ruhige Phasen", "Erwartet (5%)"),
         fill   = c(COL_BAD, COL_NEUTRAL, NA), border = NA,
         lty    = c(NA, NA, 2), lwd = c(NA, NA, 2),
         col    = c(NA, NA, COL_GREY), bty = "n", cex = CEX_LEGEND * sc)
}


# =============================================================================
# plot_pass_fail — 6×6 GARCH validation heatmap
#
# Each cell shows whether a factor (row) passed a criterion (column) on the
# full in-sample window. Margin percentages summarise by criterion and factor.
# =============================================================================
plot_pass_fail <- function(sc = 1) {
  n_f        <- length(factor_names)
  n_c        <- 6
  crit_names <- paste0("C", 1:6)

  pass_mat <- tryCatch({
    m <- matrix(NA_real_, nrow = n_f, ncol = n_c,
                dimnames = list(short_names[factor_names], crit_names))
    for (fi in seq_len(n_f)) {
      vld <- results$garch_valid[[factor_names[fi]]]
      if (!is.null(vld) && !is.null(vld$Pass) && length(vld$Pass) >= 6)
        m[fi, ] <- as.numeric(as.logical(vld$Pass[1:6]))
    }
    m
  }, error = function(e) { message("plot_pass_fail: ", e$message); NULL })

  if (is.null(pass_mat)) {
    plot.new(); title("Validation data not available"); return(invisible())
  }

  row_rates <- rowMeans(pass_mat, na.rm = TRUE)   # per factor
  col_rates <- colMeans(pass_mat, na.rm = TRUE)   # per criterion

  par(mar = if (sc >= 1) c(8, 7, 4, 5) else c(6, 5.5, 3, 4))
  image(seq_len(n_c), seq_len(n_f),
        t(pass_mat),   # x=criteria, y=factors (bottom-to-top)
        col = c(COL_BAD, COL_OK), axes = FALSE, zlim = c(0, 1),
        main = "GARCH-Validierung — 6 Kriterien × 6 Faktoren",
        xlab = "", ylab = "",
        cex.main = CEX_TITLE * sc)

  axis(1, at = seq_len(n_c), labels = crit_names,
       las = 1, cex.axis = CEX_AXIS * sc)
  axis(2, at = seq_len(n_f), labels = rownames(pass_mat),
       las = 1, cex.axis = CEX_AXIS * sc * 0.9)

  for (ci in seq_len(n_c))
    for (fi in seq_len(n_f)) {
      val <- t(pass_mat)[ci, fi]
      if (!is.na(val))
        text(ci, fi, if (val == 1) "PASS" else "FAIL",
             col = "white", cex = 0.85 * sc, font = 2)
    }

  # Column pass rates along the bottom
  mtext(sprintf("%.0f%%", col_rates * 100), at = seq_len(n_c), side = 1,
        line = if (sc >= 1) 2.2 else 1.7, cex = 0.92 * sc, font = 2,
        col = ifelse(col_rates >= 0.5, COL_OK, COL_BAD))
  mtext("Pass-R.:", side = 1, at = 0.3,
        line = if (sc >= 1) 2.2 else 1.7, cex = 0.80 * sc, adj = 0)

  # Row pass rates on the right margin
  mtext(sprintf("%.0f%%", row_rates * 100), at = seq_len(n_f), side = 4,
        las = 1, line = 0.3, cex = 0.92 * sc, font = 2,
        col = ifelse(row_rates >= 0.5, COL_OK, COL_BAD))

  # Caption
  mtext("Grün = bestanden  |  Rot = abgelehnt (p ≤ 0.05)",
        side = 1, line = if (sc >= 1) 5.5 else 4,
        cex = 0.78 * sc, col = COL_GREY)
}


# =============================================================================
# Save individual presentation slides
# =============================================================================
save_pres("slide1_hook_2008_crisis.png",   quote(plot_hook()),              w = 12, h = 7)
save_pres("slide2_copula_aic.png",         quote(plot_copula_aic()),        w = 11, h = 6)
save_pres("slide3_yearly_exceptions.png",  quote(plot_yearly_exceptions()), w = 12, h = 6)
save_pres("slide3_pass_fail_summary.png",  quote(plot_pass_fail()),         w = 10, h = 7)


# =============================================================================
# Appendix: 2×2 combined overview (16×10 inches, 300 dpi)
# =============================================================================
app_path <- file.path(PRES_DIR, "appendix_full_overview.png")
png(app_path, width = 16 * 300, height = 10 * 300, res = 300)
par(mfrow = c(2, 2), oma = c(1, 1, 3, 1))
plot_hook(sc = 0.75)
plot_copula_aic(sc = 0.75)
plot_yearly_exceptions(sc = 0.75)
plot_pass_fail(sc = 0.75)
mtext("Anhang — Zusammenfassung der Hauptresultate",
      outer = TRUE, cex = 1.6, font = 2, line = 1)
dev.off()
cat(sprintf("Saved: %s\n", app_path))

cat(sprintf("\n5 files written to %s/\n", PRES_DIR))
cat("Contents:\n")
for (f in list.files(PRES_DIR, pattern = "\\.png$")) cat(sprintf("  %s\n", f))
