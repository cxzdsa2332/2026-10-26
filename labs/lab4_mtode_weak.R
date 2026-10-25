# ============================================================================
#  Lab 4  MTODE 全流程：积分形式 vs 弱形式（6 变量基准系统）
#  运行： Rscript lab4_mtode_weak.R          （约 1 分钟）
#  核心函数在 mtode_core.R（摘自 ODEmodel 项目的 MTODE v1）
# ============================================================================
source("mtode_core.R")

# %% [data] 6 变量基准系统：持续振荡，真函数 ∈ {x², x³}，9 条有向边
set.seed(1000)
dat <- bench_data(n = 200, snr = 30)        # 方差意义 SNR = 30
# dat$obs_X：200 个时间点 × 6 个变量；dat$adj_main：真实邻接矩阵

# %% [fit] 同一份数据，两种消去导数的方式各拟合一次
fA <- MTODE(dat$obs_X, dat$times, method = "integral",
            joint = FALSE, rank_tol = 1e-2)          # 方案 A
fB <- MTODE(dat$obs_X, dat$times, method = "weak",
            joint = FALSE, rank_tol = 1e-2)          # 方案 B
M <- rbind(integral = eval_fit(fA, dat), weak = eval_fit(fB, dat))

# %% [rows] 为什么弱形式误选少：设计阵里相隔 h 行的两行有多相关
lag_cor <- function(X, h) {                 # 相隔 h 行的相关，取中位数
  X <- scale(X)
  median(sapply(1:(nrow(X) - h), function(k) cor(X[k, ], X[k + h, ])))
}

if (sys.nframe() == 0) {
  options(width = 110)
  cat("=== [data] ===\n")
  cat(sprintf("p = %d 个变量，n = %d 个时间点，t ∈ [0, 3]；真实有向边 %d 条（无自环）\n",
              dat$p, nrow(dat$obs_X), sum(dat$adj_main)))
  cat("真实邻接矩阵（行 = 方程 j，列 = 变量 i；1 表示 x_i 进入方程 j）：\n")
  print(dat$adj_main)

  cat("\n=== [fit] ===\n")
  print(round(M[, c("TPR_off", "FPR_off", "MCC_off", "c_mae", "r2_med")], 3))
  cat("弱形式估出的邻接矩阵：\n"); print(fB$adj_main)

  cat("\n=== [rows] ===\n")
  ## 换算到同一时间间隔：积分形式 1 行 = 1 个时间点，弱形式 1 行 ≈ 2 个时间点
  for (h in c(1, 5, 10, 20))
    cat(sprintf("相隔 Δt ≈ %.2f：积分形式 %.3f，弱形式 %.3f\n",
                3 * 2 * h / 199, lag_cor(fA$design$X, 2 * h),
                lag_cor(fB$design$X, h)))
  cat("弱形式测试函数支撑半径 r = 0.03 × 3 = 0.09；",
      "两行中心相隔超过 2r = 0.18 后支撑不再重叠\n")
  cat(sprintf("行数：积分形式 %d 行（每个时间点一行），弱形式 %d 行（每个测试函数一行）\n",
              nrow(fA$design$X), nrow(fB$design$X)))

  cat("\n=== [coef] ===\n")
  f0 <- MTODE(dat$obs_X, dat$times, method = "integral", joint = FALSE)
  mx <- function(f) round(apply(abs(f$beta), 2, max), 1)
  cat("各方程 max|β|（积分形式）——不做截断：\n"); print(mx(f0))
  cat("做选择后秩截断重拟合（rank_tol = 1e-2）：\n"); print(mx(fA))
  cat(sprintf("两者选出的结构是否相同：%s；轨迹 R² 中位数：%.4f vs %.4f\n",
              identical(f0$adj_main, fA$adj_main),
              eval_fit(f0, dat)["r2_med"], eval_fit(fA, dat)["r2_med"]))
  cat("截距 c_j：真值 vs 弱形式估计\n")
  print(round(rbind(true = dat$cvec, weak = fB$cvec), 3))

  cat("\n=== [rep] ===\n")
  res <- do.call(rbind, lapply(1:10, function(r) {
    set.seed(5000 + r)
    d <- bench_data(n = 200, snr = 30)
    do.call(rbind, lapply(c("integral", "weak"), function(m) {
      f <- MTODE(d$obs_X, d$times, method = m, joint = FALSE, rank_tol = 1e-2)
      data.frame(rep = r, method = m, t(eval_fit(f, d)))
    }))
  }))
  agg <- aggregate(cbind(TPR_off, FPR_off, MCC_off) ~ method, res, mean)
  agg[, -1] <- round(agg[, -1], 3)
  cat("10 个新种子（5001–5010），非对角指标均值：\n"); print(agg, row.names = FALSE)
  write.csv(res, "out/lab4_compare.csv", row.names = FALSE)

  ## 轨迹分解（方程 5：x2、x6 两个父节点），供 element/09_labs.R 作图
  j <- 5
  Zt <- sapply(1:6, function(i) {
    th <- dat$thetas[3 * (i - 1) + 1:3, j]
    xi <- dat$real_X[, i]
    cumtrapz(dat$times, cbind(xi, xi^2, xi^3) %*% th)
  })
  Ze <- apply(fB$f_main_est[[j]], 2, function(col) cumtrapz(dat$times, col))
  dec <- data.frame(t = dat$times, obs = dat$obs_X[, j],
                    true = dat$real_X[, j], est = fB$x_est[, j],
                    true_x2 = Zt[, 2], true_x6 = Zt[, 6],
                    est_x2 = Ze[, 2], est_x6 = Ze[, 6])
  write.csv(dec, "out/lab4_decomp.csv", row.names = FALSE)
}
