# ============================================================================
#  Lab 0  造一个网络、造一批数据 —— 以及「相关 ≠ 结构」
#  运行： python lab0_make_data.py        （约 5 秒）
#  幻灯片按 "# %% [标签]" 抽取代码段，按 "=== [标签] ===" 抽取输出段
#  模型：确定性 ODE + 测量噪声（与 MTODE 的设定一致）
# ============================================================================
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


# %% [net] 随机稀疏网络：dx_i/dt = c_i − a_i x_i + Σ_j W_ij tanh(x_j)
def make_net(N=8, p=0.25, seed=0):
    r = np.random.default_rng(seed)
    mask = (r.random((N, N)) < p) & ~np.eye(N, dtype=bool)
    W = mask * r.choice([-1., 1.], (N, N)) * r.uniform(1, 2, (N, N))
    a = r.uniform(1.0, 2.0, N)               # 自衰减：保证有界
    c = r.uniform(-0.5, 0.5, N)              # 截距：基础合成/降解
    return W, a, c


# %% [ode] 正问题：已知结构 → 数值积分得到真实轨迹 x(t)
def simulate(W, a, c, x0, T=5.0, L=100):
    rhs = lambda t, x: c - a * x + W @ np.tanh(x)
    tt = np.linspace(0, T, L)
    sol = solve_ivp(rhs, (0, T), x0, t_eval=tt, rtol=1e-9)
    return tt, sol.y.T                       # L × N


# %% [obs] 观测模型：y = x + ε，ε 是测量噪声（按信噪比 SNR 设定）
def observe(X, snr_db, rng):
    sd = np.sqrt(X.var(0) / 10 ** (snr_db / 10))
    return X + rng.normal(0, 1, X.shape) * sd


# %% [corr] 最常见的「第一反应」：算相关矩阵，相关大就连边
def corr_edges(Y, thr=0.3):
    R = np.corrcoef(Y.T)                     # N×N，必然对称
    E = (np.abs(R) > thr) & ~np.eye(len(R), dtype=bool)
    return R, E


def score(E, W):
    off = ~np.eye(len(W), dtype=bool)
    T = (W != 0) & off                       # 真边（不含对角）
    tp = (E & T).sum()
    fp = (E & ~T & off).sum()
    return tp, T.sum(), fp, (~T & off).sum()


# %% [eq] 平衡态：x* 满足 F(x*) = 0；雅可比矩阵的特征值决定回不回得去、多快
def equilibrium(W, a, c, x_start):
    F = lambda x: c - a * x + W @ np.tanh(x)
    xs = fsolve(F, x_start, xtol=1e-12)      # 解 F(x*) = 0
    J = -np.diag(a) + W * (1 - np.tanh(xs) ** 2)  # J_ij = ∂F_i/∂x_j
    return xs, np.abs(F(xs)).max(), np.linalg.eigvals(J)


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True, linewidth=110)
    N, R_traj = 8, 10
    W, a, c = make_net(N)
    A = W - np.diag(a) + 0.0                 # 对角写入自衰减 −a_i

    print("=== [net] ===")
    print("真实耦合 W（行 i = 被影响者，列 j = 施加影响者），对角写入 −a_i：")
    print(A)
    nE = int((W != 0).sum())
    print(f"非对角真边 {nE} 条 / {N*(N-1)} 格；截距 c = {c}")

    print("\n=== [run] ===")
    rng = np.random.default_rng(1)
    trajs, x0s = [], []
    for r in range(R_traj):                  # 同一个网络、R 个不同初值
        x0s.append(rng.uniform(-3, 3, N))
        tt, X = simulate(W, a, c, x0s[-1])
        trajs.append(X)
    X1 = trajs[0]
    print(f"单条轨迹 X.shape = {X1.shape}（L 个时刻 × N 个节点），共 {R_traj} 条")
    drift = np.abs(X1[-1] - X1[-11]).max()
    print(f"第 1 条轨迹最后 10 个时刻的最大变化 = {drift:.1e}"
          "  ← 耗散系统很快停到平衡点")

    print("\n=== [corr] ===")
    rng = np.random.default_rng(2)
    Y = np.vstack([observe(X, 20, rng) for X in trajs])
    Rm, E = corr_edges(Y, thr=0.3)
    tp, P, fp, Nn = score(E, W)
    print("相关矩阵 R（R 对称：R[i,j] == R[j,i]）：")
    print(Rm)
    print(f"|R| > 0.3 判为有边：抓到真边 {tp}/{P}，误报 {fp}/{Nn}"
          f"  →  TPR = {tp/P:.2f}, FPR = {fp/Nn:.2f}")
    asym = ((W != 0) & (W.T == 0)).sum()
    print(f"真实网络里有 {asym} 条「单向边」——相关矩阵从构造上就无法区分 j→i 与 i→j")
    np.savetxt("out/lab0_R.csv", Rm, delimiter=",", fmt="%.4f")
    np.savetxt("out/lab0_A.csv", A, delimiter=",", fmt="%.4f")

    print("\n=== [deriv] ===")
    rhs_true = np.array([c - a * x + W @ np.tanh(x) for x in X1])
    for snr in (40, 20, 10):
        y = observe(X1, snr, np.random.default_rng(3))
        d = np.gradient(y, tt, axis=0)       # 直接对观测做差分
        e_y = np.sqrt(((y - X1) ** 2).mean())
        e_d = np.sqrt(((d - rhs_true) ** 2).mean())
        print(f"SNR = {snr:>2} dB：状态误差 RMSE = {e_y:.3f}，"
              f"差分导数误差 RMSE = {e_d:.3f}（放大 {e_d/e_y:4.1f} 倍）")

    print("\n=== [eq] ===")
    import pandas as pd
    T_long = 30.0                            # 观测只到 t = 5，这里延长到 30
    longs = [simulate(W, a, c, x0, T=T_long, L=601) for x0 in x0s]
    tl = longs[0][0]
    eqs, lab = [], []                        # 每条轨迹从终点出发解 F(x*) = 0，再归类
    for _, X in longs:
        xs, res, lam = equilibrium(W, a, c, X[-1])
        hit = [k for k, e in enumerate(eqs) if np.abs(e[0] - xs).max() < 1e-6]
        if hit:
            lab.append(hit[0])
        else:
            eqs.append((xs, res, lam)); lab.append(len(eqs) - 1)
    names = "ABCDEFGH"
    print(f"10 条轨迹最终落到 {len(eqs)} 个不同的平衡点（多稳态：去哪个取决于初值）")
    for k, (xs, res, lam) in enumerate(eqs):
        who = [r + 1 for r in range(R_traj) if lab[r] == k]
        lm = lam.real.max()
        print(f"平衡点 {names[k]} = {xs}")
        print(f"   残差 max|F| = {res:.0e}；雅可比特征值最大实部 = {lm:.3f}"
              f"（< 0 ⇒ 稳定），时间常数 {1 / abs(lm):.2f}；第 {who} 条轨迹去这里")
    dist = np.array([np.linalg.norm(X - eqs[lab[r]][0], axis=1)
                     for r, (_, X) in enumerate(longs)])
    for tq in (0, 2.5, 5, 10, 20, 30):
        k = np.argmin(np.abs(tl - tq))
        print(f"t = {tq:>4}：各轨迹到自己平衡点的距离，中位数 {np.median(dist[:, k]):.1e}，"
              f"最远 {dist[:, k].max():.1e}")
    k5 = np.argmin(np.abs(tl - 5))
    print(f"观测窗口末端 t = 5：距离只剩初始的 {np.median(dist[:, k5] / dist[:, 0]):.1%}"
          f"（中位数）；t ≥ 20 后约 1e-6 是数值积分精度的底噪")
    show = [lab.index(k) for k in range(len(eqs))]   # 每个平衡点挑第一条代表轨迹
    rows = [(r + 1, names[lab[r]], t, i, x) for r in show
            for t, row in zip(tl, longs[r][1]) for i, x in enumerate(row)]
    pd.DataFrame(rows, columns=["traj", "eq", "t", "node", "x"]
                 ).to_csv("out/lab0_traj.csv", index=False)
    L0 = len(tt)
    rows = [(r + 1, t, i, y) for r in show
            for t, row in zip(tt, Y[r * L0:(r + 1) * L0]) for i, y in enumerate(row)]
    pd.DataFrame(rows, columns=["traj", "t", "node", "y"]
                 ).to_csv("out/lab0_obs.csv", index=False)
    pd.DataFrame([(names[k], i, x, e[2].real.max()) for k, e in enumerate(eqs)
                  for i, x in enumerate(e[0])], columns=["eq", "node", "xstar", "lam_max"]
                 ).to_csv("out/lab0_eq.csv", index=False)
    pd.DataFrame([(r + 1, names[lab[r]], t, d) for r in range(R_traj)
                  for t, d in zip(tl, dist[r])], columns=["traj", "eq", "t", "dist"]
                 ).to_csv("out/lab0_dist.csv", index=False)
