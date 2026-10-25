# ============================================================================
#  Lab 5  激励与可辨识性：一条长轨迹 vs 多条短轨迹
#  运行： Rscript lab5_excitation.R          （约 2 分钟）
#  随机稀疏图 + 自环三次阻尼（ODEmodel v1 仿真），p = 10
#  多轨迹拟合按 ODEmodel/method.MD §11.3 的做法：同一张图、不同初值，按行堆叠
# ============================================================================
source("mtode_core.R")

# %% [traj] 同一张图，从 R 个不同初值出发各积一条轨迹
make_runs <- function(g, R, n, snr = 30) {
  lapply(seq_len(R), function(r)
    sim_traj(g, runif(g$p, -2, 2), n = n, snr = snr, Tend = 5))
}

# %% [stack] 多轨迹：每条轨迹各自构造，再按行堆叠成一个回归
fit_multi <- function(runs, method = "weak", n_order = 5) {
  p <- ncol(runs[[1]]$obs_X)
  s <- runs[[1]]$times / max(runs[[1]]$times)      # 时间归一到 [0,1]
  parts <- lapply(runs, function(d) {
    xs  <- sapply(1:p, function(j)
      predict(smooth.spline(s, d$obs_X[, j]), s)$y)
    phi <- do.call(cbind, lapply(1:p, function(i)
      poly_basis_1d(xs[, i], n_order)))
    if (method == "weak") build_weak(phi, d$obs_X, s)
    else build_integral(phi, d$obs_X, s)
  })
  X  <- do.call(rbind, lapply(parts, `[[`, "X"))
  Y  <- do.call(rbind, lapply(parts, `[[`, "Y"))
  D  <- unlist(lapply(parts, `[[`, "D"))
  id <- rep(seq_along(parts), sapply(parts, function(b) nrow(b$X)))
  Z  <- if (method == "integral")                  # 每条轨迹一列初值
    cbind(outer(id, seq_along(parts), "==") * 1, D)
  else cbind(D)                                    # 截距 c_j 全部共享
  qz <- qr(Z)
  Xs <- scale(qr.resid(qz, X));  ys <- scale(qr.resid(qz, Y))
  grp <- rep(1:p, each = n_order)
  adj <- t(sapply(1:p, function(j) {
    f <- ADSIHT(Xs, ys[, j], group = grp, kappa = 0.95,
                ic.coef = 3, ic.scale = 3, max_iter = 100)
    b <- f$beta[, which.min(f$ic)]
    tapply(b != 0, grp, any) * 1
  }))
  adj
}

# %% [metric] 非对角指标 + 自环召回
off_metrics <- function(adj, truth) {
  off <- row(truth) != col(truth)
  m <- calculate_metrics(adj[off], truth[off])
  c(m, self = mean(diag(adj) == 1))
}

if (sys.nframe() == 0) {
  options(width = 110)
  designs <- list("1 × 400" = c(1, 400), "8 × 50" = c(8, 50),
                  "4 × 200" = c(4, 200), "8 × 200" = c(8, 200))

  cat("=== [single] ===\n")
  set.seed(101)
  g  <- sim_graph(p = 10)
  r1 <- make_runs(g, 1, 400)[[1]]
  half <- r1$real_X[201:400, ]
  sd_late <- apply(half, 2, sd) / apply(r1$real_X, 2, sd)
  cat(sprintf("p = 10，每个方程 1 个自环 + 2 个父节点；单条轨迹 t ∈ [0, 5]，400 个点\n"))
  cat(sprintf("后半段波动 / 全程波动（各变量）：%s\n",
              paste(sprintf("%.2f", sd_late), collapse = " ")))
  cat(sprintf("后半段几乎不动（比值 < 0.1）的变量：%d / 10\n", sum(sd_late < 0.1)))
  fs <- MTODE(r1$obs_X, r1$times, method = "weak", joint = FALSE)
  fm <- fit_multi(list(r1), "weak")
  cat(sprintf("单轨迹时 fit_multi 与 MTODE() 选出的结构是否一致：%s\n",
              identical(unname(fs$adj_main), unname(fm))))
  r2 <- 1 - colSums((fs$x_est - r1$real_X)^2) /
            colSums(scale(r1$real_X, scale = FALSE)^2)
  m1 <- off_metrics(fs$adj_main, g$adj_main)
  cat(sprintf("弱形式：轨迹 R² 中位数 = %.3f，但非对角 MCC = %.2f，自环召回 = %.1f\n",
              median(r2), m1["MCC"], m1["self"]))

  cat("\n=== [design] ===\n")
  res <- do.call(rbind, lapply(1:5, function(rep) {
    set.seed(200 + rep)
    g <- sim_graph(p = 10)
    do.call(rbind, lapply(names(designs), function(nm) {
      d <- designs[[nm]]
      runs <- make_runs(g, d[1], d[2])
      do.call(rbind, lapply(c("integral", "weak"), function(m)
        data.frame(rep = rep, design = nm, method = m,
                   t(off_metrics(fit_multi(runs, m), g$adj_main)))))
    }))
  }))
  agg <- aggregate(cbind(TPR, FPR, MCC, self) ~ method + design, res, mean)
  agg <- agg[order(agg$method, match(agg$design, names(designs))), ]
  agg[, 3:6] <- round(agg[, 3:6], 2)
  cat("5 张随机图（种子 201–205），非对角指标与自环召回的均值：\n")
  print(agg, row.names = FALSE)
  write.csv(res, "out/lab5_design.csv", row.names = FALSE)
}
