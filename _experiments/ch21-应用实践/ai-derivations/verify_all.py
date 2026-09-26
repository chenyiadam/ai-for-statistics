# verify_all.py：ai-derivations 各命题的数值验证（2026-09-25 实测）
import numpy as np
import statsmodels.api as sm

print("== 命题 A：常规方差估计 / 真实方差（设计 B、lam=3）==")
rng = np.random.default_rng(20260108)
ratios = []
for _ in range(2000):
    n = 50
    x1 = np.exp(rng.normal(size=n)) - np.exp(0.5)
    X = np.column_stack([np.ones(n), x1, rng.normal(size=n)])
    omega = 1.0 + 3.0 * x1 ** 2
    XtXi = np.linalg.pinv(X.T @ X)
    V = (X.T * omega) @ X
    true_var = (XtXi @ V @ XtXi)[1, 1]
    naive = np.mean(omega) * XtXi[1, 1]
    ratios.append(naive / true_var)
print("naive/true 中位数:", round(float(np.median(ratios)), 3))

print("== 命题 B：删除残差关系 + HC3 vs OLS se ==")
rng = np.random.default_rng(20260109)
n = 30
x1 = np.exp(rng.normal(size=n)) - np.exp(0.5)
X = np.column_stack([np.ones(n), x1, rng.normal(size=n)])
y = X @ np.array([1.0, 0.3, 0.5]) + np.sqrt(1 + 3 * x1 ** 2) * rng.normal(size=n)
b = np.linalg.pinv(X.T @ X) @ (X.T @ y)
e = y - X @ b
h = np.einsum("ij,jk,ik->i", X, np.linalg.pinv(X.T @ X), X)
looresid = np.array([(y[i] - X[i] @ np.linalg.pinv(np.delete(X, i, 0).T @ np.delete(X, i, 0))
                      @ (np.delete(X, i, 0).T @ np.delete(y, i))) for i in range(n)])
print("(B1) 最大偏差:", float(np.max(np.abs(looresid - e / (1 - h)))))
r = sm.OLS(y, X).fit()
print("HC3 se:", round(float(r.get_robustcov_results("HC3").bse[1]), 4),
      " OLS se:", round(float(r.bse[1]), 4))

print("== 命题 C：参数量公式 + CE 梯度 ==")
V, T, d, L = 8000, 256, 256, 6
P = V*d + T*d + L*(4*(d*d + d) + 8*d*d + 5*d + 4*d) + 2*d
print("公式参数量:", P)
rng = np.random.default_rng(0)
z = rng.normal(size=5); yy = 2
sm_ = np.exp(z - z.max()); sm_ /= sm_.sum()
eps = 1e-6
f = lambda zz: -np.log(np.exp(zz[yy]-zz.max())/np.exp(zz-zz.max()).sum())
grad_num = np.array([(f(z+eps*np.eye(5)[k]) - f(z-eps*np.eye(5)[k]))/(2*eps) for k in range(5)])
grad_ana = sm_.copy(); grad_ana[yy] -= 1
print("梯度解析-数值最大偏差:", float(np.max(np.abs(grad_ana - grad_num))))
print("ln(8000) =", round(float(np.log(8000)), 3))

print("== 命题 D：帽子矩阵性质 + 2p/n 阈值覆盖率 ==")
rng = np.random.default_rng(20260110)
for design in ("A", "B", "C"):
    hi_frac = []
    for _ in range(300):
        n, p = 50, 3
        if design == "A":
            x1 = rng.normal(size=n)
        elif design == "B":
            x1 = np.exp(rng.normal(size=n)) - np.exp(0.5)
        else:
            x1 = (rng.random(n) < 0.1).astype(float)
        X = np.column_stack([np.ones(n), x1, rng.normal(size=n)])
        if np.linalg.matrix_rank(X) < p:
            continue
        H = X @ np.linalg.pinv(X.T @ X) @ X.T
        hh = np.diag(H)
        assert abs(np.trace(H) - p) < 1e-8
        assert hh.min() > -1e-10 and hh.max() <= 1 + 1e-10
        hi_frac.append(float((hh > 2 * p / n).mean()))
    print(f"设计 {design}: h>2p/n 的平均比例 = {np.mean(hi_frac):.3f}")

print("== 命题 E：数值支撑引用 practice4 蒙特卡洛结果（见 runs/），无独立复算 ==")
