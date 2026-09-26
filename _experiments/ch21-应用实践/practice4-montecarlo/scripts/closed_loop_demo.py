# closed_loop_demo.py：异方差下 OLS 检验水平失真、稳健标准误修正与 wild bootstrap 恢复
# 复现自第 21 章 21.5.4.1 节
import numpy as np
from scipy import stats

R, B, J = 1000, 199, 1          # 重复次数、bootstrap 次数、待检验系数下标
n = 30                          # 小样本 + 高杠杆设计，失真最明显的组合
rng = np.random.default_rng(20260105)   # 固定种子
rej = {k: 0 for k in ("OLS", "HC0", "HC3", "WB")}

for r in range(R):
    x1 = np.exp(rng.normal(size=n)) - np.exp(0.5)      # 对数正态，右尾重，产生高杠杆点
    X = np.column_stack([np.ones(n), x1, rng.normal(size=n)])
    y = X @ np.array([1.0, 0.0, 0.5]) + np.sqrt(1 + 3 * x1 ** 2) * rng.normal(size=n)
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    e = y - X @ b
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    crit = stats.t.ppf(0.975, n - 3)
    hinv2 = 1.0 / (1.0 - h) ** 2                        # 杠杆值修正因子
    for k, w in (("OLS", (e @ e) / (n - 3)),
                 ("HC0", e ** 2),
                 ("HC3", e ** 2 * hinv2)):
        se = np.sqrt(w * XtX_inv[J, J]) if k == "OLS" else \
             np.sqrt((XtX_inv @ (X.T @ (X * w[:, None])) @ XtX_inv)[J, J])
        rej[k] += int(abs(b[J] / se) > crit)
    # wild bootstrap：原假设下重抽样，统计量与被精化的检验同为 HC3 口径
    Xk = X[:, [0, 2]]
    br = np.linalg.pinv(Xk.T @ Xk) @ (Xk.T @ y)
    mu = Xk @ br                                        # 原假设下的均值结构
    er = y - mu
    Wg = rng.integers(0, 2, size=(n, B)) * 2 - 1        # Rademacher 权重
    Ys = mu[:, None] + er[:, None] * Wg
    Bs = XtX_inv @ (X.T @ Ys)
    Es = Ys - X @ Bs
    t_star = Bs[J] / np.sqrt((XtX_inv @ np.einsum("ia,im,ib->mab", X, Es**2 * hinv2[:, None], X)
                              @ XtX_inv)[:, J, J])
    t_obs = b[J] / np.sqrt((XtX_inv @ (X.T @ (X * (e**2 * hinv2)[:, None])) @ XtX_inv)[J, J])
    pval = (1 + np.sum(np.abs(t_star) >= abs(t_obs))) / (B + 1)
    rej["WB"] += int(pval < 0.05)

print(f"名义水平 0.05，R = {R}，B = {B}，n = {n}，lambda = 3，高杠杆设计")
for k, v in rej.items():
    p = v / R
    print(f"{k}: 拒绝率 {p:.3f}（蒙特卡洛标准误 {np.sqrt(p * (1 - p) / R):.3f}）")
