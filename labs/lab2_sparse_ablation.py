# ============================================================================
#  Lab 2  消融实验：微分 vs 积分、Lasso vs 组硬阈值
#  运行： python lab2_sparse_ablation.py      （约 1 分钟，LassoCV 占大头）
#  同一份数据、同一个字典，只换两个设计选择 —— 差异全部来自这两个选择
# ============================================================================
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.interpolate import make_smoothing_spline
from sklearn.linear_model import LassoCV

# %% [setup] 真实系统 dx_i/dt = −a_i x_i + Σ_j W_ij tanh(x_j)
N, K, R, L, T = 10, 5, 8, 30, 3.0       # 节点、基函数、轨迹、时间点、时长
PSI = [lambda x: x, lambda x: x**2, lambda x: x**3,
       np.tanh, np.sin]                 # 字典 ψ，K = 5
rng = np.random.default_rng(7)
a = 1.0 + rng.random(N)
W = np.zeros((N, N))
edges = [(0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4),
         (6, 0), (7, 2), (8, 4), (9, 6), (0, 9), (3, 7)]
for i, j in edges:
    W[i, j] = rng.choice([-1., 1.]) * (1.5 + rng.random())
rhs = lambda t, x: -a * x + W @ np.tanh(x)
tt = np.linspace(0, T, L)
X0s = rng.normal(0, 2.5, (R, N))        # 8 个不同初值
Xs_true = [solve_ivp(rhs, (0, T), x0, t_eval=tt, rtol=1e-10).y.T
           for x0 in X0s]
Etrue = (W != 0) | np.eye(N, dtype=bool)  # 真边，含自衰减对角


# %% [build] 平滑 → 微分形式 (Pd, dY) 与积分形式 (Pi, Yc)
def build(snr_db, seed):
    r = np.random.default_rng(seed)
    Pd, Pi, dY, Yc = [], [], [], []
    for Xt in Xs_true:
        s2 = Xt.var(0) / 10**(snr_db / 10)          # SNR → 噪声方差
        sd = np.sqrt(np.maximum(s2, 1e-12))
        Y = Xt + r.normal(0, sd, Xt.shape)          # 测量噪声
        Xh = np.column_stack([make_smoothing_spline(tt, Y[:, i])(tt)
                              for i in range(N)])   # 平滑样条
        P = np.column_stack([f(Xh[:, i])
                             for i in range(N) for f in PSI])  # ψ(X̂)
        Pd.append(P)                                # 微分：ψ(X̂(t))
        Pi.append(cumulative_trapezoid(P, tt, axis=0,
                                       initial=0.)) # 积分：∫₀ᵗ ψ(X̂)
        dY.append(np.gradient(Xh, tt, axis=0))      # 响应：X̂′(t)
        Yc.append(Xh - Xh[0])                       # 响应：X̂(t) − X̂(0)
    return np.vstack(Pd), np.vstack(Pi), np.vstack(dY), np.vstack(Yc)


# %% [fit] 每个方程一次 LassoCV；组范数 ‖θ_ij,·‖₂ 就是边强度
def group_norms(D, resp):
    G = np.zeros((N, N))
    for i in range(N):
        c = LassoCV(cv=5, max_iter=50000).fit(D, resp[:, i]).coef_
        G[i] = np.linalg.norm(c.reshape(N, K), axis=1)
    return G


def select(G, th=0.0):
    """th = 0：纯 Lasso（组范数非零就算有边）
       th > 0：再叠一层组硬阈值，组范数 > th × 该行最大值才算"""
    return G > th * G.max(axis=1, keepdims=True)


# %% [mcc] 指标：TPR、FPR、MCC（类别不平衡时最可靠的单一指标）
def mcc_set(E):
    TP = (E & Etrue).sum(); FP = (E & ~Etrue).sum()
    FN = (~E & Etrue).sum(); TN = (~E & ~Etrue).sum()
    d = np.sqrt(float(TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = (TP * TN - FP * FN) / d if d else 0.
    return TP / max(TP + FN, 1), FP / max(FP + TN, 1), mcc


if __name__ == "__main__":
    print("=== [setup] ===")
    print(f"真实边 {Etrue.sum()}/{N*N}（含 {N} 个自衰减对角）；每条方程候选系数 N·K = {N*K}；"
          f"样本 R·L = {R*L}")
    sweep = {}
    print("\n=== [ablation] ===")
    for snr in [2, 10, 20]:
        Pd, Pi, dY, Yc = build(snr, 100 + snr)
        Gd, Gi = group_norms(Pd, dY), group_norms(Pi, Yc)
        rows = [('微分+Lasso', mcc_set(select(Gd))), ('积分+Lasso', mcc_set(select(Gi))),
                ('积分+组硬阈值', mcc_set(select(Gi, 0.1)))]
        print(f"--- SNR = {snr} dB ---")
        for n, (t_, f_, m_) in rows:
            print(f"  TPR={t_:.3f}  FPR={f_:.3f}  MCC={m_:.3f}   ← {n}")
        sweep[snr] = [(th, *mcc_set(select(Gi, th))) for th in np.arange(0, 0.55, 0.05)]

    print("\n=== [sweep] ===")
    print("积分形式 + 组硬阈值：阈值 th（相对每行最大组范数）对 MCC 的影响")
    print("  th   " + "".join(f"  SNR={s:>2}dB" for s in sweep))
    for k in range(len(sweep[2])):
        print(f" {sweep[2][k][0]:.2f} " + "".join(f"   {sweep[s][k][3]:7.3f}" for s in sweep))
    for s in sweep:
        best = max(sweep[s], key=lambda r: r[3])
        print(f"SNR={s:>2} dB：最优 th = {best[0]:.2f}，MCC = {best[3]:.3f}"
              f"（TPR {best[1]:.2f} / FPR {best[2]:.2f}）")
