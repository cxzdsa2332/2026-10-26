# ============================================================================
#  MTODE 核心函数（Lab 4、Lab 5 共用，source 本文件即可）
#  摘自 ODEmodel 项目 myscript_intercrpt_v1.R（MTODE v1）：
#    模型、基函数、积分 / 弱形式两种构造、投影 + ADSIHT 选择、系数还原、
#    仿真与评估。去掉了绘图与并行网格函数，算法逻辑未改。
#  模型：dx_j/dt = c_j + sum_i f_ij(x_i),  f_ij(0) = 0
#  依赖：deSolve ADSIHT Matrix pracma
# ============================================================================
suppressMessages({
  library(deSolve); library(ADSIHT); library(Matrix); library(pracma)
})

## ---- 基函数 ----------------------------------------------------------------
## 一元基 function(x) -> length(x) x m 矩阵，每个变量 m 列构成一个 group。
## MTODE 内部对 c(0, x) 求值后减去 x = 0 那一行，所以任何基都满足 f(0) = 0
poly_basis_1d <- function(x, degree) {
  X <- outer(as.numeric(x), seq_len(degree), "^")
  colnames(X) <- paste0("x^", seq_len(degree))
  X
}

## 三角：sin / cos(k pi x / 2L)，k = 1..K，L = max|x|（数据只占半个周期）
trig_basis_1d <- function(x, K) {
  x <- as.numeric(x)
  w <- outer(x, seq_len(K)) * pi / (2 * max(abs(x)))
  X <- cbind(sin(w), cos(w))
  colnames(X) <- c(paste0("sin", seq_len(K)), paste0("cos", seq_len(K)))
  X
}

make_basis <- function(basis, n_order) {
  if (is.function(basis)) return(basis)
  switch(match.arg(basis, c("poly", "trig")),
         poly = function(x) poly_basis_1d(x, n_order),
         trig = function(x) trig_basis_1d(x, n_order))
}

## ---- 仿真 1：随机稀疏图 + 自环（ODEmodel 的 v1 仿真） ----------------------
## dx_j/dt = c_j - a_j x_j - b_j x_j^3 + sum_{i in pa(j)} (th1 x_i + th2 x_i^2)
## 自环三次阻尼保证有界；每个方程随机 indeg 个父节点
sim_graph <- function(p = 10, indeg = 2) {
  thetas <- matrix(0, 3 * p, p)
  for (j in 1:p) {
    thetas[3 * (j - 1) + 1, j] <- -runif(1, 0.5, 1.5)
    thetas[3 * (j - 1) + 3, j] <- -runif(1, 0.5, 1)
    for (i in sample(setdiff(1:p, j), indeg))
      thetas[3 * (i - 1) + 1:2, j] <-
        sample(c(-1, 1), 2, replace = TRUE) * runif(2, 0.5, 1.5)
  }
  grp <- rep(1:p, each = 3)
  adj <- t(sapply(1:p, function(j) tapply(thetas[, j] != 0, grp, any) * 1))
  dimnames(adj) <- list(paste0("eq", 1:p), paste0("x", 1:p))
  list(p = p, thetas = thetas, cvec = runif(p, -0.5, 0.5), adj_main = adj)
}

## 给定一张图，从初值 x0 出发积分一条轨迹并加测量噪声（方差 = Var(x)/snr）
sim_traj <- function(g, x0, n = 200, snr = 30, Tend = 5) {
  times <- seq(0, Tend, length.out = n)
  mod <- function(t, x, P)
    list(P$cvec + as.vector(crossprod(P$thetas, as.vector(rbind(x, x^2, x^3)))))
  X <- ode(y = x0, times = times, func = mod, parms = g,
           method = "lsoda")[, -1, drop = FALSE]
  sd <- sqrt(apply(X, 2, var) / snr)
  obs <- X + matrix(rnorm(n * g$p, sd = rep(sd, each = n)), n, g$p)
  list(times = times, real_X = X, obs_X = obs)
}

## 与 ODEmodel 的 sim_data() 等价：一张随机图 + 一条轨迹
sim_data <- function(p = 10, n = 200, snr = 30, Tend = 5, indeg = 2,
                     x0_range = 2) {
  g <- sim_graph(p, indeg)
  c(g, sim_traj(g, runif(p, -x0_range, x0_range), n, snr, Tend))
}

## ---- 仿真 2：6 变量基准系统（ODEmodel v0，持续振荡，f_ij ∈ {x^2, x^3}） ----
## thetas 行 = (变量 i, 次数 1..3)，列 = 方程 j；无自环，9 条有向边
bench_data <- function(n = 200, snr = 30) {
  p <- 6
  cvec <- c(0.3, -0.2, 0.5, -0.3, -0.4, 0.2)
  th <- matrix(0, 3 * p, p)
  put <- function(i, d, j, v) th[3 * (i - 1) + d, j] <<- v
  put(2, 3, 1,  2); put(3, 2, 1,  1); put(1, 3, 2, -3)
  put(4, 3, 3,  2); put(6, 3, 3,  2); put(3, 3, 4, -2)
  put(2, 3, 5, -2); put(6, 3, 5,  5); put(5, 3, 6, -5)
  g <- list(p = p, thetas = th, cvec = cvec)
  grp <- rep(1:p, each = 3)
  g$adj_main <- t(sapply(1:p, function(j) tapply(th[, j] != 0, grp, any) * 1))
  dimnames(g$adj_main) <- list(paste0("eq", 1:p), paste0("x", 1:p))
  x0 <- c(0.8, -0.8, 0.7, -0.7, -0.6, 0.6)
  c(g, sim_traj(g, x0, n, snr, Tend = 3))
}

## ---- 两种消去导数的构造：造 (设计阵 X, 响应 Y, 漂移列 D) -----------------
trapz_weights <- function(times) {
  n <- length(times)
  c(diff(times[1:2]), times[3:n] - times[1:(n - 2)],
    diff(times[(n - 1):n])) / 2
}

# %% [integral] 方案 A 积分形式：设计阵 = 基函数的累积积分，响应 = 观测本身
build_integral <- function(phi, x_obs, times) {
  list(X = apply(phi, 2, function(col) cumtrapz(times, col)),
       Y = x_obs,                   # x_j(t) = x_j(0) + c_j t + ∫Φ β
       D = times - times[1])        # 漂移列 t：系数是截距 c_j
}

# %% [weak] 方案 B 弱形式：乘紧支测试函数 ψ_k，分部积分把导数挪走
build_weak <- function(phi, x_obs, times, K = 100, r_frac = 0.03, q = 7) {
  n <- length(times)
  r <- r_frac * diff(range(times))                 # 支撑半径
  a <- seq(min(times) + r, max(times) - r, length.out = K)
  w <- trapz_weights(times)
  W <- dW <- matrix(0, K, n)
  for (k in seq_len(K)) {
    u  <- (times - a[k]) / r
    ii <- abs(u) < 1
    W[k, ii]  <- (1 - u[ii]^2)^q                   # ψ_k
    dW[k, ii] <- -2 * q / r * u[ii] * (1 - u[ii]^2)^(q - 1)   # ψ_k'
    s <- sqrt(sum(w * W[k, ]^2))                   # L² 归一
    W[k, ] <- W[k, ] / s;  dW[k, ] <- dW[k, ] / s
  }
  Wq  <- sweep(W,  2, w, "*")                      # 行积分算子
  dWq <- sweep(dW, 2, w, "*")
  list(X = Wq %*% phi,                             # ∫ ψ_k Φ dt
       Y = -(dWq %*% x_obs),                       # −∫ ψ_k' x_j dt
       D = as.vector(Wq %*% rep(1, n)))            # ∫ ψ_k dt
}

# %% [mtode] 主拟合函数 MTODE()（完整流程见第 6 章）
## basis   : "poly" / "trig" 或自定义 function(x)
## joint   : TRUE = p 个方程块对角联合求解；FALSE = 逐方程单独求解
## rank_tol: 选择后秩截断重拟合的阈值 (d_k / d_1)，NULL = 不截断
MTODE <- function(x_obs, times, n_order = 5, basis = "poly",
                  method = c("integral", "weak"),
                  kappa = 0.95, ic.coef = 3, ic.scale = 3,
                  joint = TRUE, rank_tol = NULL, ...) {
  method <- match.arg(method)
  p <- ncol(x_obs)

  ## 1. 时间归一到 [0,1]；系数整体被放大 T 倍，最后除回
  Tspan <- diff(range(times))
  times <- (times - min(times)) / Tspan

  ## 2. 逐列平滑（GCV），只用于构造非线性基
  sm <- lapply(1:p, function(j) smooth.spline(times, as.numeric(x_obs[, j])))
  x_smooth <- sapply(sm, function(f) predict(f, x = times)$y)

  ## 3. 构基：对 c(0, x) 求值再减首行 => f_ij(0) = 0
  bfun <- make_basis(basis, n_order)
  phi_list <- lapply(1:p, function(i) {
    B <- as.matrix(bfun(c(0, x_smooth[, i])))
    sweep(B[-1, , drop = FALSE], 2, B[1, ], "-")
  })
  phi   <- do.call(cbind, phi_list)
  group <- rep(1:p, sapply(phi_list, ncol))

  ## 4. 造设计阵与响应
  bd <- switch(method,
               integral = build_integral(phi, x_obs, times),
               weak     = build_weak(phi, x_obs, times, ...))

  ## 5. 投影掉免惩罚列 Z，再按列缩放（FWL：等价于 Z 不受惩罚）
  Z  <- if (method == "integral") cbind(1, bd$D) else cbind(bd$D)
  qz <- qr(Z)
  Xs <- qr.resid(qz, bd$X)
  sX <- sqrt(colSums(Xs^2) / (nrow(Xs) - 1));  Xs <- sweep(Xs, 2, sX, "/")
  y_s <- qr.resid(qz, bd$Y)
  sY  <- sqrt(colSums(y_s^2) / (nrow(y_s) - 1)); y_s <- sweep(y_s, 2, sY, "/")

  ## 6. 组稀疏选择（ADSIHT，按信息准则取路径上的最优点）
  fit_one <- function(X, y, grp) {
    f <- ADSIHT(X, y, group = grp, kappa = kappa, ic.coef = ic.coef,
                ic.scale = ic.scale, max_iter = 100)
    f$beta[, which.min(f$ic)]
  }
  if (joint) {
    X_design <- as.matrix(bdiag(rep(list(Xs), p)))
    groups   <- as.vector(outer(group, (0:(p - 1)) * p, "+"))
    b_std <- matrix(fit_one(X_design, as.vector(y_s), groups), ncol = p)
  } else {
    b_std <- sapply(1:p, function(j) fit_one(Xs, y_s[, j], group))
  }

  ## 6b. 选择后秩截断重拟合：支撑不变，只重估系数
  if (!is.null(rank_tol)) for (j in 1:p) {
    S <- which(b_std[, j] != 0)
    if (!length(S)) next
    sv <- svd(Xs[, S, drop = FALSE])
    r  <- sv$d / sv$d[1] > rank_tol
    b_std[S, j] <- sv$v[, r, drop = FALSE] %*%
      (crossprod(sv$u[, r, drop = FALSE], y_s[, j]) / sv$d[r])
  }

  ## 7. 还原系数到原始尺度；8. 残差回归到 Z 上取回截距与初值
  beta <- sweep(sweep(b_std, 1, sX, "/"), 2, sY / Tspan, "*")
  gam  <- qr.coef(qz, bd$Y - bd$X %*% (beta * Tspan))
  cvec <- gam[nrow(gam), ] / Tspan
  phi_int <- apply(phi, 2, function(col) cumtrapz(times, col))
  x_int <- phi_int %*% (beta * Tspan) + outer(times, cvec * Tspan)
  beta0 <- if (method == "integral") gam[1, ] else colMeans(x_obs - x_int)

  ## 9. 邻接矩阵：组内系数是否全零
  idx <- split(seq_len(ncol(phi)), group)
  adj <- matrix(0, p, p)
  for (j in 1:p)
    adj[j, which(sapply(idx, function(cl) any(beta[cl, j] != 0)))] <- 1
  dimnames(adj) <- list(paste0("eq", 1:p), paste0("x", 1:p))

  ## 10. 分组贡献 f_ij(x_i(t))、导数、重构轨迹
  f_main_est <- lapply(1:p, function(j) {
    est <- sapply(idx, function(cl) phi[, cl, drop = FALSE] %*% beta[cl, j])
    colnames(est) <- paste0("x", 1:p)
    est
  })
  list(method = method, beta = beta, beta0 = beta0, cvec = cvec,
       adj_main = adj, f_main_est = f_main_est,
       dx_est = sweep(phi %*% beta, 2, cvec, "+"),
       x_est  = sweep(x_int, 2, beta0, "+"), x_smooth = x_smooth,
       design = bd)
}

## ---- 评估 ------------------------------------------------------------------
calculate_metrics <- function(predicted, true) {
  predicted <- as.integer(predicted); true <- as.integer(true)
  TP <- sum(predicted == 1 & true == 1); FP <- sum(predicted == 1 & true == 0)
  TN <- sum(predicted == 0 & true == 0); FN <- sum(predicted == 0 & true == 1)
  den <- sqrt(as.numeric(TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
  c(TPR = if (TP + FN == 0) 0 else TP / (TP + FN),
    FPR = if (FP + TN == 0) 0 else FP / (FP + TN),
    MCC = if (den == 0) 0 else (TP * TN - FP * FN) / den)
}

## 自环恒为真，混进总体指标会抬高 TPR、稀释 FPR，另报非对角指标
eval_fit <- function(f, dat) {
  off <- row(dat$adj_main) != col(dat$adj_main)
  m_off <- calculate_metrics(f$adj_main[off], dat$adj_main[off])
  names(m_off) <- paste0(names(m_off), "_off")
  r2 <- 1 - colSums((f$x_est - dat$real_X)^2) /
            colSums(scale(dat$real_X, scale = FALSE)^2)
  c(calculate_metrics(f$adj_main, dat$adj_main), m_off,
    c_mae  = median(abs(f$cvec - dat$cvec)),
    r2_med = median(r2))
}
