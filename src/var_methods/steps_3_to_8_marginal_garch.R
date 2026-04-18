# ── 0. Install & load packages ────────────────────────────────────────────────
pkgs <- c("xts","zoo","moments","tseries","forecast","FinTS",
          "rugarch","gamlss","gamlss.dist","gamlss.add","copula")
miss <- setdiff(pkgs, rownames(installed.packages()))
if (length(miss)) install.packages(miss, dependencies = TRUE,
                                   repos = "https://cloud.r-project.org")
suppressPackageStartupMessages(lapply(pkgs, library, character.only = TRUE))

# ── 1. Output dirs ─────────────────────────────────────────────────────────────
FIG_DIR <- "outputs/figures"
TBL_DIR <- "outputs/tables"
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(TBL_DIR, recursive = TRUE, showWarnings = FALSE)

save_png <- function(name, expr, w = 10, h = 6) {
  f <- file.path(FIG_DIR, name)
  png(f, width = w * 100, height = h * 100, res = 100)
  eval(expr); dev.off(); eval(expr)
}

hdr <- function(txt) cat("\n", strrep("═", 70), "\n  ", txt, "\n",
                         strrep("═", 70), "\n", sep = "")

# ── 2. Load risk factors ───────────────────────────────────────────────────────
raw          <- read.csv("data/processed/risk_factors.csv", header = TRUE,
                         stringsAsFactors = FALSE, check.names = FALSE)
factor_dates <- as.Date(raw[, 1])
factor_names <- colnames(raw)[-1]

factors_list <- setNames(lapply(factor_names, function(c) {
  x <- as.numeric(raw[[c]]); x[!is.na(x)]
}), factor_names)

mat_raw       <- as.matrix(raw[, factor_names, drop = FALSE])
class(mat_raw) <- "numeric"
ok            <- complete.cases(mat_raw)
factors_mat   <- mat_raw[ok, , drop = FALSE]
rownames(factors_mat) <- as.character(factor_dates[ok])
factors_xts   <- xts(factors_mat, order.by = factor_dates[ok])

cat("Loaded:", paste(factor_names, collapse = ", "),
    "| rows:", nrow(factors_mat), "\n")

# ── 3–7. Per-factor pipeline ───────────────────────────────────────────────────
results <- list(adf=list(), arma_order=list(), arma_fit=list(),
                arma_resid=list(), arch_test=list(), dist_fit=list(),
                dist_family=list(), dist_params=list(), dist_table=list(),
                garch_fit=list(), variance_check=list(), garch_valid=list())

for (fct in factor_names) {
  cat("\n\n", strrep("█", 70), "\n  FACTOR: ", fct, "\n",
      strrep("█", 70), "\n", sep = "")
  y <- factors_list[[fct]]; n <- length(y)

  # ── Step 3: ADF ───────────────────────────────────────────────────────────────
  hdr(paste("Step 3 · ADF —", fct))
  adf <- adf.test(y); print(adf)
  results$adf[[fct]] <- adf
  cat("→", if (adf$p.value < 0.05) "STATIONARY (p<0.05)" else
            "MAY BE NON-STATIONARY", "\n")

  save_png(paste0("garch_step3_adf_", fct, ".png"), quote({
    par(mfrow = c(2,1), mar = c(4,4,3,1))
    plot(y, type="l", col="steelblue",
         main=paste0("Returns — ", fct, "  (ADF p=", round(adf$p.value,4), ")"),
         xlab="t", ylab="log-ret")
    abline(h=0, col="grey60", lty=2)
    rm <- stats::filter(y, rep(1/60,60), sides=2)
    rs <- sqrt(stats::filter((y-mean(y))^2, rep(1/60,60), sides=2))
    plot(rm, type="l", col="darkred", ylim=range(c(rm,rs), na.rm=TRUE),
         main="60-obs rolling mean (red) & sd (blue)", xlab="t", ylab="")
    lines(rs, col="steelblue"); abline(h=0, col="grey60", lty=2)
  }))

  # ── Step 4: ARMA ──────────────────────────────────────────────────────────────
  hdr(paste("Step 4 · ARMA —", fct))
  af  <- auto.arima(y, max.p=5, max.q=5, max.d=0, stationary=TRUE,
                    seasonal=FALSE, ic="aic", stepwise=FALSE, approximation=FALSE)
  ord <- arimaorder(af)
  results$arma_order[[fct]] <- ord; results$arma_fit[[fct]] <- af
  results$arma_resid[[fct]] <- residuals(af)
  lb4 <- Box.test(residuals(af), lag=10, type="Ljung-Box", fitdf=ord[1]+ord[3])
  cat(sprintf("ARMA(%d,%d) | LB p=%.4f → %s\n", ord[1], ord[3], lb4$p.value,
              if (lb4$p.value>0.05) "OK" else "autocorrelation remains"))

  save_png(paste0("garch_step4_arma_", fct, ".png"), quote({
    par(mfrow=c(2,2), mar=c(4,4,3,1))
    Acf(y,  lag.max=40, main=paste("ACF returns —", fct))
    Pacf(y, lag.max=40, main=paste("PACF returns —", fct))
    Acf(residuals(af),  lag.max=40, main=paste0("ACF resid ARMA(",ord[1],",",ord[3],")"))
    Pacf(residuals(af), lag.max=40, main="PACF resid")
  }), w=12, h=8)

  # ── Step 5: ARCH ──────────────────────────────────────────────────────────────
  hdr(paste("Step 5 · ARCH —", fct))
  ri   <- residuals(af)
  arch <- ArchTest(ri, lags=10)
  lb5  <- Box.test(ri^2, lag=10, type="Ljung-Box")
  print(arch); print(lb5)
  results$arch_test[[fct]] <- list(arch=arch, lb_sq=lb5)
  cat("→", if (arch$p.value<0.05 && lb5$p.value<0.05)
            "ARCH CONFIRMED — use GARCH" else "Weak ARCH evidence", "\n")

  save_png(paste0("garch_step5_arch_", fct, ".png"), quote({
    par(mfrow=c(2,2), mar=c(4,4,3,1))
    plot(ri,    type="l", col="steelblue", main=paste("Residuals —",fct), xlab="t", ylab="")
    abline(h=0, col="grey60", lty=2)
    plot(ri^2,  type="l", col="darkred",  main="Squared residuals",      xlab="t", ylab="")
    Acf(ri,    lag.max=40, main="ACF residuals")
    Acf(ri^2,  lag.max=40, main=paste0("ACF squared  (LB p=",round(lb5$p.value,4),")"))
  }), w=12, h=8)

  # ── Step 6: Marginal distribution ────────────────────────────────────────────
  hdr(paste("Step 6 · Marginal dist —", fct))
  cands <- c("NO","TF","GT","LO","SST","ST3","JSU","GED","NET")

  fit_one <- function(data, fam) tryCatch({
    m    <- gamlssML(data, family=fam, trace=FALSE)
    pars <- Filter(Negate(is.null),
                   list(mu=m$mu, sigma=m$sigma, nu=m$nu, tau=m$tau))
    pars <- pars[!is.na(unlist(pars))]
    set.seed(42)
    sim  <- do.call(match.fun(paste0("r",fam)), c(list(n=5000), pars))
    ks   <- suppressWarnings(ks.test(data, sim))
    list(family=fam, fit=m, params=pars, AIC=AIC(m), BIC=BIC(m),
         logLik=as.numeric(logLik(m)), KS_D=as.numeric(ks$statistic),
         KS_p=as.numeric(ks$p.value), ok=TRUE)
  }, error=function(e) list(family=fam, ok=FALSE))

  succ <- Filter(function(x) isTRUE(x$ok),
                 lapply(cands, function(f) { cat(" ",f); fit_one(y,f) }))
  cat("\n")

  cmp <- do.call(rbind, lapply(succ, function(x)
    data.frame(Family=x$family, logLik=round(x$logLik,2), AIC=round(x$AIC,2),
               BIC=round(x$BIC,2), KS_D=round(x$KS_D,4),
               KS_p=round(x$KS_p,4), KS_pass=x$KS_p>0.05,
               stringsAsFactors=FALSE)))
  cmp <- cmp[order(cmp$AIC), ]; rownames(cmp) <- NULL
  print(cmp)
  write.csv(cmp, file.path(TBL_DIR, paste0("garch_step6_dist_", fct, ".csv")),
            row.names=FALSE)

  passing <- cmp[cmp$KS_pass, ]
  fam <- if (exists("manual_dist_choice") && !is.null(manual_dist_choice[[fct]]))
    manual_dist_choice[[fct]] else if (nrow(passing)>0) passing$Family[1] else cmp$Family[1]
  cat(sprintf("→ Chosen: %s\n", fam))

  ci  <- which(sapply(succ, function(x) x$family) == fam)
  fd  <- succ[[ci]]$fit; fp <- succ[[ci]]$params
  results$dist_fit[[fct]] <- fd; results$dist_family[[fct]] <- fam
  results$dist_params[[fct]] <- fp; results$dist_table[[fct]] <- cmp

  save_png(paste0("garch_step6a_densities_", fct, ".png"), quote({
    pal <- c("#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
             "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf")
    hist(y, breaks=60, probability=TRUE, col="grey90", border="white",
         main=paste0("Density fits — ", fct), xlab="log-ret")
    for (i in seq_along(succ)) {
      x <- succ[[i]]; chosen <- identical(x$family, fam)
      curve(do.call(match.fun(paste0("d",x$family)), c(list(x=z), x$params)),
            xname="z", add=TRUE, col=pal[((i-1)%%length(pal))+1],
            lwd=if(chosen) 3 else 1, lty=if(chosen) 1 else 3)
    }
    curve(dnorm(z, mean(y), sd(y)), xname="z", add=TRUE, col="red", lwd=2, lty=2)
    legend("topleft", legend=c(sapply(succ,function(x)
      paste0(if(identical(x$family,fam))"★ " else "",x$family,
             " AIC=",round(x$AIC,1))),"Normal ref"),
      col=c(pal[seq_along(succ)],"red"), lty=1, lwd=1, cex=0.7, bty="n")
  }), w=12, h=7)

  save_png(paste0("garch_step6b_qq_", fct, ".png"), quote({
    nc <- 3; nr <- ceiling(length(succ)/nc)
    par(mfrow=c(nr,nc), mar=c(4,4,3,1))
    for (x in succ) {
      set.seed(42)
      sim <- do.call(match.fun(paste0("r",x$family)), c(list(n=length(y)), x$params))
      qqplot(sort(sim), sort(y),
             main=paste0(if(identical(x$family,fam))"★ " else "",x$family,
                         " KS p=",round(x$KS_p,3)),
             xlab="fitted", ylab="empirical",
             col=if(identical(x$family,fam))"darkred" else "steelblue", pch=20, cex=0.6)
      abline(0,1,col="red",lwd=1.5)
    }
  }), w=14, h=4*ceiling(length(succ)/3))

  # ── Step 7: GARCH ────────────────────────────────────────────────────────────
  hdr(paste("Step 7 · GARCH —", fct))
  go <- if (exists("manual_garch_order") && !is.null(manual_garch_order[[fct]]))
    manual_garch_order[[fct]] else c(1,1)
  gm <- if (exists("manual_garch_model") && !is.null(manual_garch_model[[fct]]))
    manual_garch_model[[fct]] else "sGARCH"
  gd <- if (exists("manual_garch_dist")  && !is.null(manual_garch_dist[[fct]]))
    manual_garch_dist[[fct]]  else "std"
  cat(sprintf("Spec: %s(%d,%d)-%s\n", gm, go[1], go[2], gd))

  spec_g <- ugarchspec(
    variance.model     = list(model=gm, garchOrder=go),
    mean.model         = list(armaOrder=c(ord[1],ord[3]), include.mean=TRUE),
    distribution.model = gd)
  fit_g <- tryCatch(ugarchfit(spec=spec_g, data=y, solver="hybrid"),
                    error=function(e){message(e$message); NULL})
  if (is.null(fit_g)) { cat("Fit failed — skipping.\n"); next }
  show(fit_g)
  results$garch_fit[[fct]] <- fit_g

  Zh  <- as.numeric(residuals(fit_g, standardize=TRUE))
  sig <- as.numeric(sigma(fit_g))

  c1  <- Box.test(Zh,    lag=10, type="Ljung-Box")
  c2a <- Box.test(Zh^2,  lag=10, type="Ljung-Box")
  c2b <- ArchTest(Zh, lags=10)
  gof_tbl <- gof(fit_g, groups=c(20,30,40,50))
  uv  <- tryCatch(as.numeric(uncvariance(fit_g)), error=function(e) NA_real_)
  ev  <- var(y); ratio <- uv/ev
  per <- tryCatch(as.numeric(persistence(fit_g)), error=function(e) NA_real_)
  sb  <- signbias(fit_g)
  ny  <- nyblom(fit_g)

  v1 <- c1$p.value  > 0.05
  v2 <- c2a$p.value > 0.05 && c2b$p.value > 0.05
  v3 <- mean(gof_tbl[,"p-value(g-1)"] > 0.05) >= 0.5
  v4 <- !is.na(ratio) && ratio > 0.75 && ratio < 1.25
  v5 <- all(sb$prob  > 0.05)
  v6 <- ny$JointStat < ny$JointCritical[2]

  verdict <- data.frame(
    Criterion = c("C1 LB Ẑ","C2 LB+ARCH Ẑ²","C3 GoF innov",
                  "C4 Uncond var","C5 Sign Bias","C6 Nyblom"),
    Value     = c(round(c1$p.value,4), round(min(c2a$p.value,c2b$p.value),4),
                  round(mean(gof_tbl[,"p-value(g-1)"]),4),
                  round(ratio,3), round(min(sb$prob),4), round(ny$JointStat,4)),
    Pass      = c(v1,v2,v3,v4,v5,v6), stringsAsFactors=FALSE)
  print(verdict, row.names=FALSE)

  results$variance_check[[fct]] <- list(empirical=ev, modelled=uv, ratio=ratio,
                                        deviation=ratio-1, persistence=per, pass=v4)
  results$garch_valid[[fct]] <- verdict

  if (!v5) cat("→ Sign Bias failed — consider gjrGARCH\n")
  if (!v3) cat("→ GoF failed — try dist='sstd' or 'nig'\n")
  if (!v6) cat("→ Nyblom failed — consider rolling re-estimation\n")

  save_png(paste0("garch_step7_", fct, ".png"), quote({
    par(mfrow=c(2,3), mar=c(4,4,3,1))
    plot(y, type="l", col="grey60",
         main=sprintf("%s(%d,%d)-%s | %s", gm,go[1],go[2],gd,fct),
         xlab="t", ylab="ret")
    lines(2*sig, col="red"); lines(-2*sig, col="red")
    plot(Zh, type="l", col="steelblue", main="Std residuals Ẑ_t", xlab="t", ylab="Ẑ")
    abline(h=0, col="grey60", lty=2)
    hist(Zh, breaks=40, probability=TRUE, col="grey85", border="white",
         main="Ẑ_t vs N(0,1)", xlab="Ẑ")
    curve(dnorm(x), add=TRUE, col="red", lwd=2)
    Acf(Zh,    lag.max=40, main=paste0("ACF Ẑ  (p=",round(c1$p.value,3),")"))
    Acf(Zh^2,  lag.max=40, main=paste0("ACF Ẑ²  (p=",round(c2a$p.value,3),")"))
    plot(sig^2, type="l", col="grey50",
         main=sprintf("Variance: ratio=%.2f  dev=%+.1f%%", ratio, 100*(ratio-1)),
         xlab="t", ylab="σ̂²")
    abline(h=ev, col="blue", lwd=2); abline(h=uv, col="darkred", lwd=2, lty=2)
    legend("topright", legend=c("cond σ̂²","emp var","model uncond"),
           col=c("grey50","blue","darkred"), lty=c(1,1,2), lwd=2, bty="n", cex=0.75)
  }), w=15, h=9)

  cat("══ Done:", fct, "══\n")
}

# ── 8. Final summary tables ────────────────────────────────────────────────────
hdr("FINAL SUMMARY")

summary_tbl <- data.frame(
  Factor    = factor_names,
  ARMA      = sapply(factor_names, function(f) {
    o <- results$arma_order[[f]]; paste0("(",o[1],",",o[3],")") }),
  GARCHspec = sapply(factor_names, function(f) {
    g <- results$garch_fit[[f]]; if(is.null(g)) "—" else {
      m <- g@model
      sprintf("%s(%d,%d)-%s", m$modeldesc$vmodel,
              m$modelinc["alpha"], m$modelinc["beta"], m$modeldesc$distribution) }}),
  Dist      = sapply(factor_names, function(f) results$dist_family[[f]]),
  C1=sapply(factor_names,function(f) results$garch_valid[[f]]$Pass[1]),
  C2=sapply(factor_names,function(f) results$garch_valid[[f]]$Pass[2]),
  C3=sapply(factor_names,function(f) results$garch_valid[[f]]$Pass[3]),
  C4=sapply(factor_names,function(f) results$garch_valid[[f]]$Pass[4]),
  C5=sapply(factor_names,function(f) results$garch_valid[[f]]$Pass[5]),
  C6=sapply(factor_names,function(f) results$garch_valid[[f]]$Pass[6]),
  stringsAsFactors=FALSE)
print(summary_tbl, row.names=FALSE)
write.csv(summary_tbl, file.path(TBL_DIR, "garch_summary.csv"), row.names=FALSE)

var_tbl <- data.frame(
  Factor      = factor_names,
  EmpVar      = sapply(factor_names,function(f) sprintf("%.3e",results$variance_check[[f]]$empirical)),
  ModelVar    = sapply(factor_names,function(f) sprintf("%.3e",results$variance_check[[f]]$modelled)),
  Ratio       = sapply(factor_names,function(f) round(results$variance_check[[f]]$ratio,3)),
  Deviation   = sapply(factor_names,function(f) sprintf("%+.1f%%",100*results$variance_check[[f]]$deviation)),
  Persistence = sapply(factor_names,function(f) round(results$variance_check[[f]]$persistence,3)),
  C4_Pass     = sapply(factor_names,function(f) results$variance_check[[f]]$pass),
  stringsAsFactors=FALSE)
print(var_tbl, row.names=FALSE)
write.csv(var_tbl, file.path(TBL_DIR, "garch_variance_check.csv"), row.names=FALSE)

cat("\nFigures → outputs/figures/  |  Tables → outputs/tables/\n")
cat("'results' list in workspace — use for steps 8–13.\n")