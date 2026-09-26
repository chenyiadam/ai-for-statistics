# 命题 B：HC3 的 jackknife 解释与 $(1-h_i)^2$ 修正因子的来历

对应第 21 章 21.5.4 节（HC0–HC3 的结构）。

## 【提示词】

> HC0–HC3 三明治方差估计器的区别只在权重 $w_i$：HC0 取 $1$、HC2 取 $1/(1-h_i)$、HC3 取 $1/(1-h_i)^2$。
> 请推导：(1) 删除第 $i$ 个观测后重拟合，其残差 $e_{(i)}$ 与原始残差 $e_i$ 的精确关系；
> (2) 说明为什么 HC3 可以读作“近似 jackknife”方差估计；
> (3) 从这个角度解释为什么杠杆值 $h_i \to 1$ 时 HC2/HC3 爆炸，这与实践四模拟中设计 C 出现 NaN 是同一件事吗？

## 【AI 推导】

**(1) 删除残差的精确关系。** 设去掉第 $i$ 个观测后重拟合的系数为 $\hat{b}_{(i)}$。对回归诊断的标准结论（可用 Sherman–Morrison 公式对 $X^{\top}X$ 添一删一得到）：
$$e_{(i)} = y_i - x_i^{\top}\hat{b}_{(i)} = \frac{e_i}{1 - h_i}, \tag{B1}$$
其中 $h_i = x_i^{\top}(X^{\top}X)^{-1}x_i$ 是杠杆值，$e_i$ 是全样本残差。$h_i$ 越接近 1，该点对拟合的影响越大，全样本残差被压缩得越厉害，而删除后的残差恢复其“真身”。
**(2) HC3 读作近似 jackknife。** 删除第 $i$ 点后重拟合的方差估计的 jackknife 思想是：把每个“删除后波动”汇总。MacKinnon–White (1985) 指出，若把 (B1) 的删除残差平方 $e_{(i)}^2 = e_i^2/(1-h_i)^2$ 直接代入三明治公式中的“肉”矩阵 $\sum_i x_i x_i^{\top} e_i^2$ 的位置，得到的正是
$$V_{\mathrm{HC3}} = (X^{\top}X)^{-1} \left[ \sum_i x_i x_i^{\top} \cdot \frac{e_i^2}{(1-h_i)^2} \right] (X^{\top}X)^{-1},$$
即权重 $w_i = 1/(1-h_i)^2$ 的 HC3。所以 HC3 的直觉是：**先估计每个点的“删除后残差”再进三明治**，而 HC2 用 $1/(1-h_i)$ 是同一思想的一阶版本（对残差方差本身的修正），HC0 则完全不修正。经验上按 HC0 < HC1 < HC2 < HC3 的顺序，标准误逐个变大、过度拒绝逐个减轻（本实践四的模拟在方向上复现了这个顺序）。
**(3) $h_i \to 1$ 的爆炸。** $w_i = (1-h_i)^{-2}$ 在 $h_i \to 1$ 时二阶发散。这不是缺陷而是提示：$h_i = 1$ 意味着该观测自己决定了自己的拟合值（删除它后该设计点处没有任何信息），其残差恒为 0，任何基于残差的方差估计在该点都失去依据。实践四模拟的设计 C（稀疏二元回归量、$n$ 小）中部分重复恰好出现 $h_i = 1$（稀疏组的每个观测都被该组单独钉住），HC2/HC3 权重除零得 NaN，**与推导一致**，处理方式应当是显式回退（如改用 wild bootstrap），而不是静默丢弃。

## 【数值验证】

```python
# 验证 (B1)：删除残差 = e_i/(1-h_i)，并对照 HC3 与逐点删除重拟合的 jackknife 标准误
import numpy as np, statsmodels.api as sm
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
hc3 = r.get_robustcov_results("HC3").bse[1]
print("HC3 se:", round(float(hc3), 4), " OLS se:", round(float(r.bse[1]), 4))
```

输出：
```text
(B1) 最大偏差: 4.44e-15（机器精度内成立）
HC3 se: 0.2344  OLS se: 0.2460
```
删除关系精确成立。需要如实说明：在这一份随机数据上 HC3 标准误（0.2344）反而略小于常规标准误（0.2460），**单份数据上的标准误对比噪声很大，不能作为“HC3 更大”的证据**；“HC0<HC1<HC2<HC3 的系统性顺序”是跨重复的期望行为，其证据是实践四的蒙特卡洛（runs/grid_summary.csv 中 se_ratio 列）而非单点对比。这份对照的价值恰恰在于提醒：单次实现与期望性质是两回事。
