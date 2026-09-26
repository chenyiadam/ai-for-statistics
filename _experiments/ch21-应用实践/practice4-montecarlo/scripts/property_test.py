# property_test.py：验证代码所依赖的代数性质在小规模随机情形下恒成立
# 复现自第 21 章 21.5.6 节
import numpy as np
rng = np.random.default_rng(20260104)
for _ in range(200):
    n = rng.integers(6, 60); p = rng.integers(2, 6)
    X = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
    if np.linalg.matrix_rank(X) < p:
        continue
    XtX_inv = np.linalg.pinv(X.T @ X)
    H = X @ XtX_inv @ X.T
    h = np.diag(H)
    assert np.allclose(H @ H, H, atol=1e-8)                 # 幂等
    assert np.allclose(H, H.T, atol=1e-8)                   # 对称
    assert abs(np.trace(H) - p) < 1e-8                      # 迹等于秩
    assert (h >= -1e-10).all() and (h <= 1 + 1e-10).all()   # 0 <= h_ii <= 1
    y = rng.normal(size=n)
    b = XtX_inv @ (X.T @ y); e = y - X @ b
    assert np.allclose(X.T @ e, 0, atol=1e-8)               # 残差与列空间正交
print("属性测试通过：帽子矩阵与残差的代数性质成立")
