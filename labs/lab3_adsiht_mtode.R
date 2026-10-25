# ============================================================================
#  Lab 3  用作者自己的 R 包 ADSIHT 复现 MTODE 的仿真（ref/main.tex 第 5 节 Simulation）
#  运行： Rscript lab3_adsiht_mtode.R        （约 3 分钟）
#  设定照抄 main.tex：20 个耦合 ODE，ψ(x) = (cos(x/2), x³)，t ∈ [0,1]，T = 100 个时间点
#  ⚠ 课堂复现：每个 SNR 只重复 6 次（论文 50 次），数字不可当作论文结果引用
# ============================================================================
suppressPackageStartupMessages({ library(deSolve); library(ADSIHT); library(glmnet) })
options(width = 110)

# %% [theta] main.tex §5.2 方程组的 48 个非零 θ_ji：(j, i, θ_ji1, θ_ji2)
th <- rbind(
  c(1,1,-2,0),  c(1,2,0,7),   c(2,2,-2,0),  c(2,1,0,-11), c(2,6,5,0),
  c(3,3,-2,0),  c(3,4,0,4),   c(4,4,-2,0),  c(4,3,0,-2),  c(5,5,-2,0),
  c(5,6,0,-2),  c(5,4,0,-5),  c(6,6,-2,0),  c(6,5,0,3),   c(7,7,-2,0),
  c(7,8,0,5),   c(8,8,-2,0),  c(8,7,0,-5),  c(9,9,-2,0),  c(9,7,0,12),
  c(9,6,-2,0),  c(10,10,-2,0),c(10,7,2,0),  c(11,11,-2,0),c(11,7,0,-5),
  c(11,1,4,0),  c(12,12,-2,0),c(12,7,5,0),  c(13,13,-2,0),c(13,7,0,-8),
  c(13,18,3,0), c(14,14,-2,0),c(14,7,-6,0), c(14,16,0,-1),c(15,15,-2,0),
  c(15,16,0,-4),c(16,16,-2,0),c(16,15,0,4), c(17,17,-2,0),c(17,18,-3,0),
  c(17,19,0,-2),c(18,18,-2,0),c(18,17,0,6), c(19,19,-2,0),c(19,20,0,-2),
  c(20,20,-2,0),c(20,19,0,2), c(20,15,0,-2))
x0 <- c(-1.1, -0.5, 3.1, 0.1, 0.3, 3.4, 0.9, -2.5, -1.4, -0.9,
        2.4, 0.7, 0.8, 0.2, -1.1, 3.6, 1.0, -3.9, 1.4, -0.9)

# %% [truth] 真实系统：X_j' = Σ_i ψ(X_i)ᵀ θ_ji，lsoda 求解
p <- 20; M <- 2
psi <- function(x) cbind(cos(x / 2), x^3)       # 基函数，K = M = 2
Theta <- array(0, c(p, p, M))                   # Theta[j, i, m]：i → j
for (r in seq_len(nrow(th))) Theta[th[r, 1], th[r, 2], ] <- th[r, 3:4]
E_true <- apply(Theta != 0, c(1, 2), any)       # 真边（含自作用）
rhs <- function(t, x, parms)
  list(as.vector(Theta[, , 1] %*% cos(x / 2) + Theta[, , 2] %*% x^3))
tt <- seq(0, 1, length.out = 100)
X <- lsoda(x0, tt, rhs, NULL)[, -1]             # 100 × 20 真实轨道


# %% [design] 两步配点：平滑样条 → 基函数的时间积分 Ψ̂_i(t)
make_design <- function(Y) {
  # 细网格：每个采样间隔再分 10 份，观测时刻恰好落在格点上
  fine <- seq(0, 1, length.out = 10 * (length(tt) - 1) + 1)
  idx  <- seq(1, length(fine), by = 10)
  sp   <- lapply(1:p, function(i) smooth.spline(tt, Y[, i]))  # GCV 选 λ
  Xh   <- sapply(sp, function(s) predict(s, fine)$y)
  dXh  <- sapply(sp, function(s) predict(s, tt, deriv = 1)$y)  # 微分形式用
  cumtrap <- function(f)                                      # 累积梯形积分
    c(0, cumsum((f[-1] + f[-length(f)]) / 2 * diff(fine)))
  # Ψ̂_i(t) = ∫₀ᵗ ψ(X̂_i(u)) du，逐个调控者 i 拼成 100 × 40
  Psi_int <- do.call(cbind, lapply(1:p, function(i)
    apply(psi(Xh[, i]), 2, cumtrap)[idx, ]))
  Psi_dif <- do.call(cbind, lapply(1:p, function(i) psi(Xh[idx, i])))
  list(int = Psi_int, dif = Psi_dif, dY = dXh)
}
group <- rep(1:p, each = M)          # 第 i 组 = 调控者 i 的 M 个系数


# %% [fit] 三种估计器：ADSIHT（逐方程）、ADSIHT（多任务联合）、Lasso
edges_adsiht <- function(D, R) {     # R：100 × p 响应矩阵
  t(sapply(1:p, function(j) {
    fit <- ADSIHT(D, R[, j], group, ic.type = "dsic")
    b <- fit$beta[, which.min(fit$ic)]   # DSIC 选 s₀；λ 由停时确定
    tapply(b != 0, group, any)           # 组非零 ⇔ 有边
  }))
}
# 多任务（main.tex 算法 1）：标准化 → 拼块对角 → 组 = (j, i) → 联合求解
# 注：ADSIHT 0.2.1 的 ADSIHT.ML() 不读 group_list 参数，故不用它
edges_joint <- function(D, R) {
  Xb <- as.matrix(Matrix::bdiag(rep(list(scale(D)), p)))  # 2000 × 800
  gb <- rep(1:(p * p), each = M)
  fit <- ADSIHT(Xb, as.vector(scale(R)), gb, ic.type = "dsic")
  b <- fit$beta[, which.min(fit$ic)]
  matrix(tapply(b != 0, gb, any), p, p, byrow = TRUE)
}
edges_lasso <- function(D, R) {      # 对照：glmnet + 5 折交叉验证
  t(sapply(1:p, function(j) {
    cv <- suppressWarnings(cv.glmnet(D, R[, j], nfolds = 5))
    b <- as.vector(coef(cv, s = "lambda.min"))[-1]
    tapply(b != 0, group, any)
  }))
}

# %% [metric] TPR / FPR / MCC：在全部 p × p = 400 个有向对上计算
metric <- function(E) {
  TP <- sum(E & E_true);  FP <- sum(E & !E_true)
  FN <- sum(!E & E_true); TN <- sum(!E & !E_true)
  d <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
  c(TPR = TP / (TP + FN), FPR = FP / (FP + TN),
    MCC = if (d > 0) (TP * TN - FP * FN) / d else 0)
}

noisy <- function(snr) {             # main.tex 的 SNR 定义
  sig <- sqrt(apply(X, 2, var) / 10^(snr / 10))
  X + sweep(matrix(rnorm(length(X)), nrow(X)), 2, sig, `*`)
}


# ---------------------------------------------------------------------------
cat("=== [truth] ===\n")
cat(sprintf("p = %d 个节点，M = %d 个基函数 ⇒ 每个方程 p·M = %d 个候选系数；n = %d 个时间点\n",
            p, M, p * M, length(tt)))
cat(sprintf("真实有向边 %d / %d（其中自作用 %d 条）；轨道取值范围 [%.1f, %.1f]\n",
            sum(E_true), p * p, sum(diag(E_true)), min(X), max(X)))

cat("\n=== [one] ===\n")
set.seed(7)
Y <- noisy(20); D <- make_design(Y)
cat("SNR = 20 dB，逐方程比较选中的调控者集合：\n")
for (j in c(2, 13, 20)) {
  fit <- ADSIHT(D$int, Y[, j], group, ic.type = "dsic")
  b <- fit$beta[, which.min(fit$ic)]
  cv <- suppressWarnings(cv.glmnet(D$int, Y[, j], nfolds = 5))
  bl <- as.vector(coef(cv, s = "lambda.min"))[-1]
  cat(sprintf("方程 %2d  真实 {%s}\n", j, paste(which(E_true[j, ]), collapse = ",")))
  cat(sprintf("         ADSIHT {%s}\n", paste(which(tapply(b != 0, group, any)), collapse = ",")))
  cat(sprintf("         Lasso  {%s}\n", paste(which(tapply(bl != 0, group, any)), collapse = ",")))
}

cat("\n=== [compare] ===\n")
reps <- 6; snrs <- c(1, 5, 10, 20)
res <- list()
for (snr in snrs) for (r in 1:reps) {
  set.seed(1000 * snr + r)
  Y <- noisy(snr); D <- make_design(Y)
  res[[length(res) + 1]] <- rbind(
    data.frame(snr, rep = r, method = "积分+ADSIHT",       t(metric(edges_adsiht(D$int, Y)))),
    data.frame(snr, rep = r, method = "积分+ADSIHT多任务", t(metric(edges_joint(D$int, Y)))),
    data.frame(snr, rep = r, method = "积分+Lasso",        t(metric(edges_lasso(D$int, Y)))),
    data.frame(snr, rep = r, method = "微分+ADSIHT",       t(metric(edges_adsiht(D$dif, D$dY)))))
}
res <- do.call(rbind, res)
agg <- aggregate(cbind(TPR, FPR, MCC) ~ method + snr, res, mean)
for (s in snrs) {
  cat(sprintf("--- SNR = %d dB（%d 次重复平均）---\n", s, reps))
  a <- agg[agg$snr == s, ]
  for (k in seq_len(nrow(a)))
    cat(sprintf("  TPR=%.3f  FPR=%.3f  MCC=%.3f   ← %s\n", a$TPR[k], a$FPR[k], a$MCC[k], a$method[k]))
}
dir.create("out", showWarnings = FALSE)
write.csv(res, "out/lab3_compare.csv", row.names = FALSE, fileEncoding = "UTF-8")
