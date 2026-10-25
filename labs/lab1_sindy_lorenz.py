# ============================================================================
#  Lab 1  字典回归与序贯阈值（STLSQ）：从混沌轨道里捞出 Lorenz 方程
#  运行： python lab1_sindy_lorenz.py         （约 5 秒）
#  纯 numpy / scipy 实现，不需要 pysindy
# ============================================================================
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
from itertools import combinations_with_replacement

TRUE = np.zeros((10, 3))       # 真值系数表（行 = 字典项，列 = x y z 三个方程）


# %% [sim] 造数据：Lorenz 系统，σ=10, ρ=28, β=8/3
sigma, rho, beta = 10., 28., 8 / 3
f = lambda t, s: [sigma * (s[1] - s[0]),
                  s[0] * (rho - s[2]) - s[1],
                  s[0] * s[1] - beta * s[2]]
t = np.arange(0, 25, 0.002)
X = solve_ivp(f, (t[0], t[-1]), [-8, 7, 27], t_eval=t,
              rtol=1e-10, atol=1e-12).y.T          # 12500 × 3


# %% [dict] 建字典：所有 ≤2 阶单项式 → 10 列
def poly_dict(X, order=2):
    n, d = X.shape
    cols, names = [np.ones(n)], ['1']
    lbl = ['x', 'y', 'z']
    for o in range(1, order + 1):
        for c in combinations_with_replacement(range(d), o):
            col = np.ones(n)
            for i in c:
                col = col * X[:, i]
            cols.append(col); names.append(''.join(lbl[i] for i in c))
    return np.column_stack(cols), names


# %% [stlsq] 序贯阈值最小二乘：置零小系数 → 在剩余支撑上重做最小二乘
def stlsq(Theta, dX, thr=0.1, n_iter=10):
    Xi = np.linalg.lstsq(Theta, dX, rcond=None)[0]
    for _ in range(n_iter):
        small = np.abs(Xi) < thr               # 找出小系数
        Xi[small] = 0                          # 置零
        for j in range(dX.shape[1]):           # 逐个方程
            big = ~small[:, j]
            # 只在幸存的列上重新最小二乘
            Xi[big, j] = np.linalg.lstsq(Theta[:, big], dX[:, j],
                                         rcond=None)[0]
    return Xi


# %% [noise] 加测量噪声 → 直接差分 vs 先平滑（Savitzky–Golay）再求导
def recover(Xn, smooth):
    if smooth:
        sg = dict(window_length=51, polyorder=3, axis=0)
        Xs = savgol_filter(Xn, **sg)
        dX = savgol_filter(Xn, deriv=1, delta=0.002, **sg)
    else:
        Xs, dX = Xn, np.gradient(Xn, t, axis=0)
    Th, _ = poly_dict(Xs)
    return stlsq(Th, dX)


# %% [print] 打印辅助（不上幻灯片）
def show(Xi, names):
    for j, v in enumerate('xyz'):
        terms = [f"{Xi[i, j]:+.3f} {names[i]}" for i in range(len(names)) if Xi[i, j] != 0]
        print(f"  d{v}/dt = " + "  ".join(terms))


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=110)
    Theta, names = poly_dict(X, order=2)
    for (i, j, v) in [('x', 0, -10), ('y', 0, 10), ('x', 1, 28), ('y', 1, -1),
                      ('xz', 1, -1), ('z', 2, -8 / 3), ('xy', 2, 1)]:
        TRUE[names.index(i), j] = v

    print("=== [dict] ===")
    print(f"字典列名（K = {len(names)}）：", names)
    print(f"设计矩阵 Θ 形状 = {Theta.shape}，cond(Θ) = {np.linalg.cond(Theta):.1e}")

    print("\n=== [ols] ===")
    dX = np.gradient(X, t, axis=0)                      # 数值求导
    Xi_ols = np.linalg.lstsq(Theta, dX, rcond=None)[0]
    print(f"普通最小二乘：非零项 {int((np.abs(Xi_ols) > 1e-8).sum())} / 30")
    print("  dx/dt 的 10 个系数 =", np.round(Xi_ols[:, 0], 3))

    print("\n=== [stlsq] ===")
    Xi = stlsq(Theta, dX)
    print(f"STLSQ：非零项 {int((Xi != 0).sum())} / 30")
    show(Xi, names)
    print("  真值：dx/dt = −10x + 10y | dy/dt = 28x − y − xz | dz/dt = −2.667z + xy")

    print("\n=== [noise] ===")
    rng = np.random.default_rng(0)
    rows = []
    print("噪声幅度   方法       非零项  错项  最大系数误差")
    for eta in [0.001, 0.01, 0.05, 0.10, 0.20]:
        Xn = X + eta * X.std(0) * rng.normal(size=X.shape)
        for smooth, lab in [(False, "直接差分"), (True, "先平滑  ")]:
            Xi_n = recover(Xn, smooth)
            wrong = int(((Xi_n != 0) != (TRUE != 0)).sum())
            err = np.abs(Xi_n - TRUE).max()
            print(f"  {eta*100:>4.1f}%   {lab}    {int((Xi_n != 0).sum()):>2}     "
                  f"{wrong:>2}    {err:8.3f}")
            rows.append((eta, int(smooth), int((Xi_n != 0).sum()), wrong, err))
    np.savetxt("out/lab1_noise.csv", rows, delimiter=",", header="eta,smooth,nnz,wrong,err",
               comments="", fmt="%.6g")
