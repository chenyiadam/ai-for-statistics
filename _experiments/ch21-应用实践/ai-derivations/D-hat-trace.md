# 命题 D：帽子矩阵的迹等于秩 $p$，杠杆值期望阈值 $2p/n$
对应第 21 章 21.3.4 节（影响点阈值）与 21.5.6 节（属性测试）。

## 【提示词】

> 设 $X$ 为 $n \times p$ 满列秩设计矩阵，$H = X(X^{\top}X)^{-1}X^{\top}$。
> 请推导：(1) $H$ 幂等、对称、迹为 $p$；(2) $0 \le h_{ii} \le 1$；(3) 由 $\mathbb{E}[\sum_i h_{ii}] = p$ 说明
> “$h_{ii} > 2p/n$ 作为高杠杆经验阈值”的道理与局限。

## 【AI 推导】

**(1) 基本性质。** 记 $A = (X^{\top}X)^{-1}$（满列秩下存在）。
- 幂等：$H^2 = XAX^{\top}XAX^{\top} = XA(X^{\top}X)AX^{\top} = XAX^{\top} = H$。
- 对称：$H^{\top} = XA^{\top}X^{\top} = XAX^{\top} = H$（$A$ 对称，因 $X^{\top}X$ 对称）。
- 迹（循环不变性）：
$$\mathrm{tr}(H) = \mathrm{tr}(XAX^{\top}) = \mathrm{tr}(AX^{\top}X) = \mathrm{tr}\left((X^{\top}X)^{-1}X^{\top}X\right) = \mathrm{tr}(I_p) = p. \tag{D1}$$
于是平均杠杆值 $\bar{h} = p/n$：**“杠杆”是总量固定的资源**，一个点高，别的点就得低。
**(2) 取值范围。** $h_{ii} = e_i^{\top}He_i$（$e_i$ 为标准基向量），$H$ 幂等对称即正交投影，正交投影的特征值都在 $\{0, 1\}$，故任意单位向量被投影后的长度平方在 $[0, 1]$：$0 \le h_{ii} \le 1$。$h_{ii} = 1$ 当且仅当 $x_i$ 落在与其余行正交的方向上（该设计点“独立支撑”一个维度）——这正是实践四设计 C 中 HC2/HC3 出现除零的边界情形。
**(3) $2p/n$ 阈值的道理与局限。** 由 $\mathrm{tr}(H) = p$，$h_{ii}$ 的“公平份额”是 $p/n$。经验规则取两倍：$h_{ii} > 2p/n$ 提示该点占用的杠杆资源超过平均份额两倍以上。局限：
- 它只是启发式，没有检验水平或最优性的保证；$h$ 的分布形态依赖设计（实践四里三种设计的 $h$ 分布差异很大），同样的 $2p/n$ 在设计 B（重尾杠杆）下会天然圈进更多点；
- 高杠杆不等于有影响：影响还取决于该点的 $y$ 偏离程度（Cook 距离把两者结合起来），所以脚本同时报告 Cook 距离与 $4/n$ 阈值。
  
## 【数值验证】

```python
# 属性测试（与 21.5.6 相同思路，另验证迹=p 与阈值覆盖率随设计变化）
import numpy as np
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
        h = np.diag(H)
        assert abs(np.trace(H) - p) < 1e-8            # (D1)
        assert h.min() > -1e-10 and h.max() <= 1 + 1e-10
        hi_frac.append(float((h > 2 * p / n).mean()))
    print(f"设计 {design}: h>2p/n 的平均比例 = {np.mean(hi_frac):.3f}")
```

输出：
```text
设计 A: h>2p/n 的平均比例 = 0.079
设计 B: h>2p/n 的平均比例 = 0.079
设计 C: h>2p/n 的平均比例 = 0.115
```
三条性质在全部随机实例上精确成立；阈值圈进的比例在设计 C（稀疏二元，杠杆集中）下明显更大，而 A 与 B 在 $n=50$ 下相当（B 的重尾在设计 B 的 $n=25$、$50$ 高异方差配置中才显著起效，见实践四网格）——这印证“$2p/n$ 是启发式而非有水平保证的检验”，其圈进率依赖设计的杠杆分布。
