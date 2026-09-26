# 命题 A：异方差下常规标准误的偏误方向与杠杆值的放大作用
对应第 21 章 21.5.4 节（实践四的理论背景）。

## 【提示词】

> 设线性模型 $Y = X\beta + \varepsilon$，$\mathbb{E}[\varepsilon\varepsilon^{\top}] = \Omega$（对角，不必相等），即存在异方差。
> 请推导：(1) OLS 估计量第 $j$ 个系数的真实方差表达式；
> (2) 常规标准误（同方差假设下）对真实方差的高估/低估取决于什么？
> 请给出明确的方向性结论，并说明它如何解释“高杠杆点 + 与杠杆绑定的方差”会让常规标准误严重低估。

## 【AI 推导】
OLS 估计量 $\hat{b} = (X^{\top}X)^{-1}X^{\top}Y$，于是
$$\hat{b} - \beta = (X^{\top}X)^{-1}X^{\top}\varepsilon,$$
第 $j$ 个系数的真实方差（以 $X$ 为条件，即设计矩阵固定）为
$$\mathrm{Var}(\hat{b}_j \mid X) = \left[(X^{\top}X)^{-1} X^{\top} \Omega X (X^{\top}X)^{-1}\right]_{jj}. \tag{A1}$$
常规标准误把 $\Omega$ 换成 $\sigma^2 I$，对应估计
$$\widehat{\mathrm{Var}}(\hat{b}_j) = s^2 \left[(X^{\top}X)^{-1}\right]_{jj}, \qquad s^2 = \mathrm{RSS}/(n-p). \tag{A2}$$
二者之比为
$$\frac{(A2)}{(A1)} = \frac{\hat{\sigma}^2 \left[(X^{\top}X)^{-1}\right]_{jj}}{\left[(X^{\top}X)^{-1}X^{\top}\Omega X (X^{\top}X)^{-1}\right]_{jj}}.$$
把 $\Omega = \mathrm{diag}(\omega_1, \dots, \omega_n)$ 代入，$X^{\top}\Omega X = \sum_i \omega_i x_i x_i^{\top}$ 是 $X^{\top}X$ 的“加权版本”。由矩阵的谱性质，该比值可以写成 $\omega_i$ 关于某种权重的加权平均；关键在于**每个系数的方差是方差 $\omega_i$ 的特定加权平均**，权重由设计决定：
$$\mathrm{Var}(\hat{b}_j \mid X) = \sum_i v_{ij}^2\, \omega_i,$$
其中 $v_{ij}$ 是矩阵 $P_v = (X^{\top}X)^{-1}X^{\top}$ 的第 $(j, i)$ 元素的平方（即 "Whitesell 权重"：$\hat{b}_j = \sum_i v_{ij} y_i$，$v_{ij}$ 是把 $y_i$ 映到 $\hat{b}_j$ 的线性组合系数）。
方向性结论：
1. 若 $\omega_i$ 与 $x_i$ 的杠杆无关（同方差），加权平均数恰为 $\sigma^2 \left[(X^{\top}X)^{-1}\right]_{jj}$（因为 $\sum_i v_{ij}^2 = \left[(X^{\top}X)^{-1}\right]_{jj}$），常规标准误无偏（差一个自由度修正）。
2. 若 $\omega_i = \sigma^2(1 + \lambda x_{i1}^2)$ 且 $v_{ij}$ 的质量集中在高杠杆点（设计 B：$x_1$ 对数正态右尾重），则 $\mathrm{Var}(\hat{b}_j)$ 的“真实加权平均”被大 $\omega_i$ 主导，而 $(A2)$ 中的 $s^2$ 是全体残差平方的平均，它把小方差点也摊了进来。**真实方差 > 常规估计**，常规标准误低估，检验过度拒绝。
3. 直观地：$\hat{b}_j$ 主要由少数高杠杆观测决定，而那些观测恰好方差最大时，“决定系数估计的那几个点最不准”，平均化的 $s^2$ 看不见这种局部失准。
自由度修正（$s^2$ 除以 $n-p$）带来的低估放大：残差 $e_i = (1 - h_i)\varepsilon_i - $ 偏差项，高杠杆点的残差被 $(1 - h_i)$ 压缩，其大方差在 $s^2$ 中被系统性抹掉，进一步加剧低估。

## 【数值验证】

```python
# 验证 (A1)：直接算真实方差（对已知 Ω）与常规估计之比，设计 B、lam=3
import numpy as np
rng = np.random.default_rng(20260108)
ratios = []
for _ in range(2000):
    n = 50
    x1 = np.exp(rng.normal(size=n)) - np.exp(0.5)
    X = np.column_stack([np.ones(n), x1, rng.normal(size=n)])
    omega = 1.0 + 3.0 * x1 ** 2                      # 真实方差
    XtXi = np.linalg.pinv(X.T @ X)
    V = (X.T * omega) @ X
    true_var = (XtXi @ V @ XtXi)[1, 1]
    naive = np.mean(omega) * XtXi[1, 1]              # s² 的期望 = mean(omega)
    ratios.append(naive / true_var)
print("naive/true 中位数:", round(float(np.median(ratios)), 3), "（<1 即常规标准误低估）")
```

输出：
```text
naive/true 中位数: 0.126（<1 即常规标准误低估）
```
即常规方差估计的中位数只有真实值的约 13%，对应标准误低估约 65%，方向与模拟中 OLS 拒绝率被推高到 0.35 量级的观察一致，且低估幅度足以解释该现象。**注意**：以上推导是条件方差层面的期望比较，单次模拟的拒绝率还受检验统计量分布形状影响，二者一致但不等价。
