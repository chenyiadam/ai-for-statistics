# handcheck.py：小样本手算核对 + 边界检查（复现自第 21 章 21.5.4 核对块）
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))   # 使 from sim import hc_se 可用

import numpy as np
import statsmodels.api as sm
from sim import hc_se

X = np.column_stack([np.ones(5), np.array([1., 2., 3., 4., 8.])])   # 含一个高杠杆点
y = np.array([1.2, 2.1, 2.8, 4.2, 7.5])
r = sm.OLS(y, X).fit()
print("sm  HC3:", r.get_robustcov_results("HC3").bse)

XtX_inv = np.linalg.pinv(X.T @ X); b = XtX_inv @ (X.T @ y); e = (y - X @ b)[:, None]
print("own HC3:", hc_se(X, e, XtX_inv, "HC3", 1))       # 应与上一行一致

# 边界测试
print("h_ii 之和:", np.einsum('ij,jk,ik->i', X, XtX_inv, X).sum(), "应等于 p =", X.shape[1])
print("lam=0 时 sigma 常数:", np.unique(np.sqrt(1 + 0.0 * X[:, 1] ** 2)))
