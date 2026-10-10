# 第3章 Transformer注意力与模型架构


## 本章导读

假设语言可以被写成序列上的条件概率 $p(x_t\mid x_{<t})$，剩下的问题是用什么参数化形式去逼近它。第 2 章解决了输入侧的离散化，本章处理函数族本身：自注意力如何把"从上下文中检索信息"写成一次可微的加权平均，前馈网络如何提供逐位置的非线性变换，归一化与残差如何使上千层的复合函数仍然可优化，位置信息怎样被注入一个置换不变的算子，以及有哪些非 Transformer 的替代者。叙述线索是"算子 → 组装 → 效率 → 规模化"，其中穿插三条与统计学的对接主线：注意力是核回归的可学习版本，前馈网络是自适应基函数展开，混合专家是带门控的混合分布模型。最后一节给出架构设计的决策路径与全栈工具地图，下一章第 4 章将在固定架构之上讨论训练与对齐。

## 3.1 自注意力机制数学定义

Transformer 是 2017 年 Google 在《Attention is All You Need》中提出的架构，用自注意力取代 RNN 的顺序处理。核心公式：

$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\Big(\frac{QK^\top}{\sqrt{d_k}}\Big)V$$

其中 Q（Query）、K（Key）、V（Value）由输入 X 经三个可学习矩阵投影得到。统计学解释：这是一个可学习的加权平均，每个 token 的输出是所有 token 的加权和，权重由 Query-Key 相似度决定。

### 3.1.1 从输入序列到 Q、K、V

设输入序列为 $X\in\mathbb{R}^{n\times d}$，第 $i$ 行为第 $i$ 个位置 token 的输入表示 $x_i\in\mathbb{R}^d$，$n$ 为序列长度，$d$ 为模型宽度。三个投影矩阵记为

$$Q=XW_Q,\quad K=XW_K,\quad V=XW_V,\qquad W_Q,W_K\in\mathbb{R}^{d\times d_k},\; W_V\in\mathbb{R}^{d\times d_v}$$

其中 $d_k$、$d_v$ 分别为 Key 与 Value 的维度，实践中常取 $d_k=d_v$。写成逐 token 的形式，$q_i=W_Q^\top x_i$，$k_j=W_K^\top x_j$，$v_j=W_V^\top x_j$。三个投影的功能分工在于角色不同：$k_j$ 是第 $j$ 个位置向外界发布的"索引"，$q_i$ 是第 $i$ 个位置发出的"检索请求"，$v_j$ 是被取回时真正搬运的"内容"。

#### 数据流示意

三次投影、一次打分、一次加权平均，整个自注意力层的数据流如下：

```text
            X  (n × d)   每 token 一行
            │
   ┌────────┼────────┐
   │        │        │
  W_Q      W_K      W_V     三组可学习投影
   │        │        │
   Q        K        V      检索请求 / 索引 / 内容
   │        │        │
   │      QK^T       │      相似度打分，得 n × n 的 logit 矩阵
   │        │        │
   │    ÷ √d_k       │      方差归一（见 3.1.2）
   │        │        │
   │    softmax      │      逐行归一化 → 权重 A（行随机矩阵）
   │        │        │
   └───── A × V ─────┘      逐行加权平均：O = A V，形状 (n × d_v)
```

数据流里没有任何一处用到 token 的先后顺序，三次投影都是逐位置的线性映射，$QK^\top$ 的打分也不含位置索引；顺序信息要等到 3.5 节的位置编码才被注入。图中的汇合点只有一次矩阵乘法 $AV$，这也是把注意力理解成"线性算子 $O=AV$"的直观来源。

### 3.1.2 缩放点积注意力与 $\sqrt{d_k}$ 的方差论证

定义未归一化的相似度（logit）$s_{ij}=q_i^\top k_j/\sqrt{d_k}$，行内 softmax 归一化后得到注意力权重

$$\alpha_{ij}=\frac{\exp(s_{ij})}{\sum_{l=1}^{n}\exp(s_{il})},\qquad \sum_{j=1}^{n}\alpha_{ij}=1$$

输出为加权和 $o_i=\sum_{j=1}^{n}\alpha_{ij}v_j$。矩阵形式即本节开头的公式，其中 $A=\mathrm{softmax}(QK^\top/\sqrt{d_k})$ 是一个行随机矩阵（row-stochastic），自注意力层就是线性算子 $O=AV$。

除以 $\sqrt{d_k}$ 的必要性可以由二阶矩直接论证。设 $q_i$ 与 $k_j$ 的各分量独立、均值为 0、方差为 $\sigma^2$，则点积 $q_i^\top k_j=\sum_{r=1}^{d_k}q_{ir}k_{jr}$ 的每一项均值为 0、方差为 $\sigma^4$，各项独立，于是

$$\mathbb{E}\big[q_i^\top k_j\big]=0,\qquad \mathrm{Var}\big(q_i^\top k_j\big)=d_k\,\sigma^4,\qquad \mathrm{sd}\big(q_i^\top k_j\big)=\sqrt{d_k}\,\sigma^2$$

即 logit 的标准差随头维度以 $\sqrt{d_k}$ 的速度增长。若不缩放，当 $d_k$ 取到几百的量级时，logit 的绝对值会远大于 1，softmax 的输出迅速退化为近似 one-hot：此时 $\partial \alpha_{ij}/\partial s_{ij}=\alpha_{ij}(1-\alpha_{ij})\approx 0$，梯度沿注意力通路消失，训练停滞。缩放后 logit 的标准差回到 $\sigma^2$，与单个分量的尺度一致，softmax 得以处在其"软"的可微区间。

温度角度可以更完整地解释这个尺度参数。把注意力写成带温度的形式 $\alpha_{ij}\propto \exp\big(q_i^\top k_j/(T\sqrt{d_k})\big)$：$T\to 0$ 时退化为最近邻检索（权重集中于单个 $j$），$T\to\infty$ 时退化为全局均匀平均（对每个主体就是样本均值）。我们可以定义一个可观测量来衡量"实际使用了多少个邻居"，即参与率（participation ratio），它在统计学里正是加权样本的有效样本量：

$$\mathrm{ESS}_i=\Big(\sum_{j=1}^{n}\alpha_{ij}^2\Big)^{-1},\qquad 1\le \mathrm{ESS}_i\le n$$

均匀权重时 $\mathrm{ESS}_i=n$，one-hot 时 $\mathrm{ESS}_i=1$。实测不同层、不同头的 ESS 差异极大，这与局部平滑中的有效邻域大小是同一个概念。把它记录下来也是调试注意力崩溃（attention collapse，即所有头的 ESS 都趋近 1 或 $n$）最直接的手段。

```python
import torch

torch.manual_seed(42)   # 固定随机种子，保证结果可复现

# 1. 构造输入 
B, H, n, d_head = 2, 4, 6, 8     # batch=2, 头数=4, 序列长=6, 每头维度=8

# 随机生成 q, k（模拟一个注意力层内部的投影结果）
q = torch.randn(B, H, n, d_head)
k = torch.randn(B, H, n, d_head)

#  2. 缩放点积注意力分数 
# scores[b, h, i, j] = q_i · k_j / sqrt(d_head)
scores = q @ k.transpose(-2, -1) / (d_head ** 0.5)   # 形状: (B, H, n, n)
print("scores 形状:", scores.shape)                    # torch.Size([2, 4, 6, 6])

#  3. softmax 得到行随机权重矩阵 
attn = torch.softmax(scores, dim=-1)                  # (B, H, n, n)
print("attn 形状:", attn.shape)                        # torch.Size([2, 4, 6, 6])
print("每行和 = 1 验证:", attn.sum(dim=-1)[0, 0])       # 全为 1.0

#  4. 有效邻域大小（Effective Sample Size） 
# 对每个 query 位置 i：ess_i = 1 / sum_j attn[b,h,i,j]^2
ess = 1.0 / attn.pow(2).sum(dim=-1)                   # (B, H, n)
print("ess 形状:", ess.shape)                          # torch.Size([2, 4, 6])
print("ess 取值范围验证: min = %.4f, max = %.4f" % (ess.min(), ess.max()))

#  5. 两个极端情况理解 ess 
# (a) 均匀分布（注意力完全分散）: ess → n
attn_uniform = torch.full((1, 1, 1, n), 1.0 / n)
ess_uniform = 1.0 / attn_uniform.pow(2).sum(dim=-1)
print(f"\n均匀分布时 ess = {ess_uniform.item():.1f}  (等于 n = {n})")

# (b) one-hot（注意力完全集中）: ess = 1
attn_onehot = torch.zeros(1, 1, 1, n)
attn_onehot[..., 0] = 1.0
ess_onehot = 1.0 / attn_onehot.sqrt().pow(2).sum(dim=-1)
print(f"one-hot 时   ess = {ess_onehot.item():.1f}  (等于 1)")

#  6. 打印部分结果 
print("\n第 1 个样本、第 1 个头的注意力矩阵 (6x6):")
print(attn[0, 0].round(decimals=3))

print("\n对应的 ess（每个 query 的有效邻居数）:")
print(ess[0, 0].round(decimals=3))

```
输出
```text
scores 形状: torch.Size([2, 4, 6, 6])
attn 形状: torch.Size([2, 4, 6, 6])
每行和 = 1 验证: tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000])
ess 形状: torch.Size([2, 4, 6])
ess 取值范围验证: min = 1.5638, max = 5.6464

均匀分布时 ess = 6.0  (等于 n = 6)
one-hot 时   ess = 1.0  (等于 1)

第 1 个样本、第 1 个头的注意力矩阵 (6x6):
tensor([[0.4820, 0.0100, 0.1420, 0.0570, 0.2150, 0.0940],
        [0.2560, 0.0960, 0.1970, 0.0770, 0.0910, 0.2820],
        [0.0980, 0.1140, 0.1430, 0.4410, 0.1060, 0.0980],
        [0.2270, 0.0670, 0.0730, 0.3840, 0.1040, 0.1440],
        [0.0810, 0.3840, 0.1000, 0.1160, 0.1960, 0.1230],
        [0.1080, 0.1880, 0.2360, 0.0740, 0.2170, 0.1770]])

对应的 ess（每个 query 的有效邻居数）:
tensor([3.2190, 4.8160, 3.8660, 4.1570, 4.3240, 5.3600])
```

#### 数值验证：手写缩放点积注意力与核加权平均的等价

下面的 numpy 示例不依赖任何深度学习框架，把 3.1.3 节"解释二"的等价关系做一次数值核对：固定一个查询点，先用 softmax 路径算注意力权重，再用显式的 Nadaraya–Watson 分子/分母路径算一遍；然后改变带宽 $h$，观察权重集中度与 ESS 如何随带宽移动。

```python
import numpy as np

rng = np.random.default_rng(0)
n, d, dk = 12, 8, 8
X = rng.normal(size=(n, d))
Wq = rng.normal(size=(d, dk)) / np.sqrt(d)
Wk = rng.normal(size=(d, dk)) / np.sqrt(d)
Wv = rng.normal(size=(d, dk)) / np.sqrt(d)

K, V = X @ Wk, X @ Wv
x_new = rng.normal(size=(d,))          # 固定一个查询点（不在序列里的新 token）
q = x_new @ Wq

# softmax 路径
s = (K @ q) / np.sqrt(dk)
w_softmax = np.exp(s - s.max()); w_softmax /= w_softmax.sum()

# 显式 Nadaraya-Watson 路径：分子 sum_j K(x,x_j) v_j，分母 sum_j K(x,x_j)
num = np.exp(s) @ V
den = np.exp(s).sum()
o_nw = num / den
w_nw = np.exp(s) / den

for h in [0.25, 0.5, 1.0, 2.0, 100.0]:   # 带宽 h：s = q.k / (h * sqrt(dk))
    w = np.exp((K @ q) / (h * np.sqrt(dk))); w = w / w.sum()
    ess = 1.0 / np.sum(w ** 2)
    j = int(np.argmax(w))
    print(f"h={h:>6}: top1权重={w.max():.4f}  j*={j:>2}  ESS={ess:6.2f}  "
          f"o[:3]={np.round((w @ V)[:3], 4)}")

w_hard = np.exp((K @ q) / (0.01 * np.sqrt(dk))); w_hard /= w_hard.sum()
w_soft = np.exp((K @ q) / (1e6 * np.sqrt(dk))); w_soft /= w_soft.sum()
j = int(np.argmax((K @ q)))
print("|o(h->0) - V[j*]|   =", np.abs(w_hard @ V - V[j]).max())
print("|o(h->inf) - meanV| =", np.abs(w_soft @ V - V.mean(axis=0)).max())
print("softmax 与 NW 两条路径的权重差 =", np.abs(w_softmax - w_nw).max())
```

运行输出（实测，环境 numpy 1.26.4）：

```text
h=  0.25: top1权重=0.8908  j*= 5  ESS=  1.26  o[:3]=[ 1.1663 -1.1662 -1.1519]
h=   0.5: top1权重=0.5465  j*= 5  ESS=  2.98  o[:3]=[ 0.8956 -0.854  -0.6496]
h=   1.0: top1权重=0.2842  j*= 5  ESS=  6.76  o[:3]=[ 0.6546 -0.5739 -0.2831]
h=   2.0: top1权重=0.1699  j*= 5  ESS=  9.85  o[:3]=[ 0.5204 -0.3624 -0.1118]
h= 100.0: top1权重=0.0847  j*= 5  ESS= 12.00  o[:3]=[ 0.3847 -0.0794  0.0625]
|o(h->0) - V[j*]|   = 0.0
|o(h->inf) - meanV| = 8.138509599575627e-07
softmax 与 NW 两条路径的权重差 = 1.3877787807814457e-17
```

三点观察。第一，softmax 路径与显式核加权路径的权重差在 $10^{-17}$ 量级，等价关系是恒等式而不是近似，softmax 的分母 $\sum_l\exp(s_{il})$ 恰好就是 Nadaraya–Watson 的核质量项。第二，带宽 $h$ 扮演的正是核平滑中带宽的角色：$h=0.25$ 时 ESS 约 1.3，输出几乎就是相似度最高那条记录的 value；$h=100$ 时权重接近均匀（ESS 恰为 $n=12$），输出退化为 value 的样本均值。第三，两个极限与统计学直觉对应：$h\to 0$ 是最近邻插值（低偏差、高方差），$h\to\infty$ 是全局平均（高偏差、低方差），注意力在训练中学习的其实是每个头、每个查询位置在这条光谱上的落点。

与条件期望的对接也在这里：把 $v_j$ 看作响应、把查询看作条件变量，$o_i=\sum_j\alpha_{ij}v_j$ 就是对条件期望 $\mathbb{E}[v\mid q_i]$ 的一个估计，与核回归估计 $\mathbb{E}[y\mid X=x]$ 的结构相同。凸组合还带来一层免费的方差控制：当 $v_j$ 相互独立时，$\mathrm{Var}(o_i)=\sum_j\alpha_{ij}^2\,\mathrm{Var}(v_j)\le\max_j\mathrm{Var}(v_j)$，权重越均匀收缩越强（极端情形就是样本均值），这与集成平均、控制变量法的方差缩减是同一类效应，平均总是稳的，代价可能是偏差，正是"解释三"所说的权衡。

#### 数值验证：温度缩放与注意力熵

softmax 饱和是训练中较常见的隐性故障之一，一个低成本的体检指标是每行注意力的熵（attention entropy，或等价地 ESS）。下面的示例先核对 $\sqrt{d_k}$ 缩放的方差论证，再扫温度看熵的变化：

```python
import numpy as np

rng = np.random.default_rng(1)
sigma = 1.0

# 一、方差论证的经验核对：对 q 与 k 共同平均
print("== 方差论证的经验核对：Var(q.k) = dk * sigma^4（对 q 与 k 共同平均）==")
for dk in [8, 64, 512]:
    m = 2000
    dots = np.empty((m, 2000))
    for i in range(m):
        q = rng.normal(size=dk) * sigma
        K = rng.normal(size=(2000, dk)) * sigma
        dots[i] = K @ q
    print(f"dk={dk:>4}: 经验 sd={dots.std():8.4f}   理论 sqrt(dk)*sigma^2="
          f"{np.sqrt(dk) * sigma ** 2:8.4f}")

# 二、温度缩放对注意力熵的影响
n, dk = 256, 64
q = rng.normal(size=dk)
K = rng.normal(size=(n, dk))
dots = K @ q

def entropy(w):
    w = w[w > 1e-300]
    return float(-np.sum(w * np.log(w)))

Hmax = np.log(n)
print("\n== 温度缩放对注意力熵的影响（n=256 个候选 key）==")
for T in [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]:
    s = dots / (np.sqrt(dk) * T)
    w = np.exp(s - s.max()); w = w / w.sum()
    print(f"T={T:>4}: 熵={entropy(w):.3f} ({entropy(w)/Hmax:.3f} of max)  "
          f"ESS={1.0/np.sum(w**2):7.2f}  最大权重={w.max():.4f}")

# 三、不缩放时的饱和效应
print("\n== 不缩放（除数取 1）的饱和效应 ==")
for dk in [8, 64, 512]:
    q = rng.normal(size=dk); K = rng.normal(size=(n, dk))
    dots = K @ q
    for label, sc in [("除以 sqrt(dk)", np.sqrt(dk)), ("不缩放", 1.0)]:
        s = dots / sc
        w = np.exp(s - s.max()); w = w / w.sum()
        print(f"dk={dk:>4} {label}: 熵/Hmax={entropy(w)/np.log(n):.3f}  "
              f"ESS={1.0/np.sum(w**2):6.1f}  最大权重={w.max():.4f}")
```

输出：

```text
== 方差论证的经验核对：Var(q.k) = dk * sigma^4（对 q 与 k 共同平均）==
dk=   8: 经验 sd=  2.8355   理论 sqrt(dk)*sigma^2=  2.8284
dk=  64: 经验 sd=  7.9994   理论 sqrt(dk)*sigma^2=  8.0000
dk= 512: 经验 sd= 22.6396   理论 sqrt(dk)*sigma^2= 22.6274

== 温度缩放对注意力熵的影响（n=256 个候选 key）==
T= 0.1: 熵=0.641 (0.116 of max)  ESS=   1.52  最大权重=0.7906
T=0.25: 熵=1.958 (0.353 of max)  ESS=   4.09  最大权重=0.4172
T= 0.5: 熵=3.902 (0.704 of max)  ESS=  20.36  最大权重=0.1432
T= 1.0: 熵=5.095 (0.919 of max)  ESS= 107.05  最大权重=0.0366
T= 2.0: 熵=5.432 (0.980 of max)  ESS= 204.32  最大权重=0.0134
T= 4.0: 熵=5.517 (0.995 of max)  ESS= 241.96  最大权重=0.0074

== 不缩放（除数取 1）的饱和效应 ==
dk=   8 除以 sqrt(dk): 熵/Hmax=0.924  ESS= 108.9  最大权重=0.0356
dk=   8 不缩放: 熵/Hmax=0.497  ESS=   7.2  最大权重=0.2434
dk=  64 除以 sqrt(dk): 熵/Hmax=0.909  ESS= 102.7  最大权重=0.0321
dk=  64 不缩放: 熵/Hmax=0.253  ESS=   3.0  最大权重=0.4895
dk= 512 除以 sqrt(dk): 熵/Hmax=0.914  ESS= 107.9  最大权重=0.0345
dk= 512 不缩放: 熵/Hmax=0.011  ESS=   1.0  最大权重=0.9885
```

第一组结果验证了 $\mathrm{sd}(q^\top k)=\sqrt{d_k}\sigma^2$：对 $q$ 与 $k$ 共同平均后经验值与理论一致。方差论证是对 $q$ 与 $k$ 的联合分布取的，若只抽一个固定的 $q$，条件方差 $\mathrm{Var}(q^\top k\mid q)=\lVert q\rVert^2\sigma^2$ 仍随该次抽样波动，小 $d_k$ 下单一实现的偏差可达两三成，这是"条件矩与无条件矩"差别的一个具体而微的例子。第二、三组给出温度—熵曲线：$T=0.1$ 时最大权重 0.79、熵只占上限的 12%，注意力已接近硬检索；$T=2$ 以上熵接近上限，注意力接近均匀平均。不缩放那一组直接展示了本节开头的论证：$d_k=512$ 时不除 $\sqrt{d_k}$ 的 softmax 熵几乎为零、ESS 约 1.0，梯度通路在数学上就已关闭。

实践上，在训练日志里定期记录各层各头的平均行熵或 ESS（每个 batch 抽几百个位置即可，开销可以忽略），或许比只盯 loss 曲线更早发现注意力崩溃；把该指标与困惑度联合观察，有助于把"loss 不降"区分为"检索过硬"与"检索过软"两类病因。

### 3.1.3 三种等价的统计解释

**解释一：可学习的加权平均。** 输出 $o_i$ 是所有 value 的凸组合，权重非负且和为 1，因此自注意力是一个保界的插值算子：它不做外推，只能在已有 value 的凸包内取值。这与 Nadaraya–Watson、局部多项式回归等线性平滑器共享同一结构，区别在于权重由参数化的神经网络函数生成而非预先给定。

**解释二：核回归（核平滑）视角。** Nadaraya–Watson 估计量为

$$\hat m(x)=\frac{\sum_{j}K_h(x,x_j)\,y_j}{\sum_{j}K_h(x,x_j)}$$

把 $y_j$ 换成 $v_j=W_V^\top x_j$，把核取成指数核 $K(x_i,x_j)=\exp\big(q_i^\top k_j/\sqrt{d_k}\big)$，softmax 的分母正好就是 Nadaraya–Watson 的分母。因此自注意力是一个带宽由训练决定、$q$ 与 $k$ 各自线性变换过的 Nadaraya–Watson 估计。与教科书版本有三处实质差别，这些差别正是 Transformer 表达能力的来源：

| 维度 | 经典核平滑 / 局部回归 | 自注意力 |
|------|---------------------|---------|
| 度量 | 欧氏距离或固定的马氏距离 $ (x-x')^\top M(x-x')$ | 双线性形式 $x^\top W_QW_K^\top x'$，可非对称，等价于可学习的、方向相关的相似度 |
| 带宽 | 由交叉验证或规则选择，标量或对角矩阵 | 由 $W_Q,W_K$ 的范数隐式决定，随层、随头、随输入变化 |
| 被平滑对象 | 原始响应 $y_j$（局部常数拟合） | 变换后的 $v_j=W_V^\top x_j$（先做一次线性降维/升维再平均） |
| 归一化 | 显式除以核质量和 | softmax 自动完成，且带来归一化的排他竞争效应 |

由于采用双线性形式，$s_{ij}\neq s_{ji}$，注意力是非对称的：这是一种有方向的检索，与对称核的非参数回归不同，这也是它在信息抽取类任务上表现突出的原因。

**解释三：softmax 温度控制的插值强度。** 权重分布的软硬程度由 logit 的分布决定：logit 分布越集中，注意力越接近硬检索（低方差、高偏差）；分布越平缓，越接近全局平均（高方差、低偏差）。这与带宽选择在偏差—方差权衡中的位置完全一致，区别是这里没有交叉验证，只有梯度下降。

### 3.1.4 复杂度、因果遮蔽与置换等变性

三层矩阵乘法分别是 $QK^\top$（$O(n^2d_k)$）、softmax（$O(n^2)$）与 $AV$（$O(n^2d_v)$），对 $n$ 是二次的；这是第 3.6 节所有效率优化的起点。

自注意力对位置的置换是等变的（permutation equivariant）：若 $\Pi$ 为置换矩阵，则 $\mathrm{Attention}(\Pi X)=\Pi\,\mathrm{Attention}(X)$。这意味着该算子本身完全不知道 token 的先后顺序，任何顺序信息都必须外部注入，这就是 3.5 节位置编码存在的理由。

自回归语言模型还需要因果遮蔽（causal masking）：令 $s_{ij}=-\infty$ 当 $j>i$，使第 $i$ 个位置只能看到前 $i$ 个位置，保证 $p_\theta(x_t\mid x_{<t})$ 的 factorization 与自回归分解一致。工程上要把 $-\infty$ 在 float16/bfloat16 下替换为有限的大负数以避免 NaN。

最后一个常被忽略的局限是秩的限制：$QK^\top$ 的秩不超过 $d_k$，$AV$ 的值域维数不超过 $d_v$。当头维度远小于序列长度时，注意力矩阵是一个低秩重构，难以表达"任意两个位置的任意关系"；这也是多头（增加独立的低秩通道）与更大头维度存在价值的一部分原因。

## 3.2 多头注意力

### 3.2.1 定义与参数量

多头注意力并行地跑 $H$ 组投影，每组在一个 $d_k=d_v=d/H$ 的子空间里工作，再把结果拼接并线性投影回去：

$$\mathrm{head}_h=\mathrm{Attention}\big(XW_Q^{(h)},XW_K^{(h)},XW_V^{(h)}\big),\qquad h=1,\dots,H$$
$$\mathrm{MHA}(X)=\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_H)\,W_O,\qquad W_O\in\mathbb{R}^{Hd_v\times d}$$

在常见的 $d_k=d_v=d/H$ 设定下，四组投影各自的总参数量为 $d\times d$，因此一个多头注意力模块（不含 FFN）的参数量为 $4d^2$，与头数无关；头数改变的是计算被切成多少份以及每份的宽度，而不是参数总量。这与并行效率的取舍直接相关：头数越多、每个头越窄，计算并行度越高但每个头的表达空间越小。常见头维度取值 64、128；$d=4096$ 时对应 64 头或 32 头。

### 3.2.2 PyTorch 实现

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """支持 MHA / GQA / MQA 的多头注意力；n_kv_heads=1 时退化为 MQA。"""

    def __init__(self, d_model, n_heads, n_kv_heads=None, causal=True):
        super().__init__()
        self.n_heads, self.n_kv_heads = n_heads, (n_kv_heads or n_heads)
        assert d_model % n_heads == 0 and n_heads % self.n_kv_heads == 0
        self.d_head = d_model // n_heads
        self.causal = causal
        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(n_heads * self.d_head, d_model, bias=False)

    def forward(self, x):
        B, n, _ = x.shape
        H, Hkv, Dh = self.n_heads, self.n_kv_heads, self.d_head
        q = self.q_proj(x).view(B, n, H, Dh).transpose(1, 2)     # (B, H, n, Dh)
        k = self.k_proj(x).view(B, n, Hkv, Dh).transpose(1, 2)   # (B, Hkv, n, Dh)
        v = self.v_proj(x).view(B, n, Hkv, Dh).transpose(1, 2)
        if Hkv != H:                                             # GQA/MQA：把 kv 头广播对齐到 q 头
            rep = H // Hkv
            k = k[:, :, None].expand(B, Hkv, rep, n, Dh).reshape(B, H, n, Dh)
            v = v[:, :, None].expand(B, Hkv, rep, n, Dh).reshape(B, H, n, Dh)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)       # (B, H, n, n)
        if self.causal:                                          # 自回归遮蔽：禁止看未来
            mask = torch.ones(n, n, dtype=torch.bool, device=x.device).triu(1)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        attn = F.softmax(scores, dim=-1)                         # 逐行归一化
        out = (attn @ v).transpose(1, 2).contiguous().view(B, n, H * Dh)
        return self.o_proj(out)


# 用法示例
mha = MultiHeadAttention(d_model=1024, n_heads=16, n_kv_heads=4, causal=True)
y = mha(torch.randn(2, 512, 1024))    # (batch=2, seq=512, d_model=1024)
print(y.shape)
```
输出
```text
torch.Size([2, 512, 1024])
```

### 3.2.3 多头的作用与冗余

多头机制为模型提供多个独立的 "相似度度量"，使不同头可以分别 specialize 于句法依存、共指、位置邻近等规律。实证研究的结论有两层：一是绝大部分头可以被剪掉而不明显损失困惑度，个别头承担关键功能（如负责"下一个词""同一词"的归纳头）；二是预训练阶段后期大量头高度相似，说明存在参数冗余。这给研究者的启示是，多头在现阶段是"过参数化带来的优化便利"，而不是每个头都有独立的语义解释；把单个头当成可解释单元去讲故事，通常需要额外的验证（如头消融的因果实验）。

把 3.1.2 的熵与 ESS 诊断搬到多头层面，还有第二个用途：检查头的维度利用率。做法是对每个头分别计算注意力的平均行熵，再对整个注意力输出的协方差矩阵计算有效秩（effective rank）$\exp(H)$，其中 $H=-\sum_r p_r\log p_r$ 是特征值谱 $p_r=\lambda_r/\sum_r\lambda_r$ 的熵。行熵低且有效秩低，说明该头几乎退化为固定查表；多个头的熵结构高度相似，则往往意味着冗余，这与剪枝实验的结论互相印证。笔者在练习中观察到，这套指标在训练早期就能把"头数设多了"的配置与"头数不足"的配置区分开；当然它只是诊断信号，替代不了正式的消融实验。

## 3.3 Feed-Forward Network

### 3.3.1 结构与参数占比

注意力负责位置之间的信息混合，前馈网络（Feed-Forward Network，FFN）则对每个位置独立地做非线性变换：

$$\mathrm{FFN}(x)=W_2\,f(W_1x+b_1)+b_2,\qquad W_1\in\mathbb{R}^{d\times d_{ff}},\; W_2\in\mathbb{R}^{d_{ff}\times d}$$

$d_{ff}$ 通常取 $4d$，此时 FFN 的参数量为 $2d d_{ff}=8d^2$，是注意力模块 $4d^2$ 的两倍，即整个 Transformer 层约三分之二的参数在 FFN 里。由于 FFN 逐位置独立地对同一个向量做计算，全部位置的 FFN 计算可以合并写成两次大矩阵乘法（先把 $n$ 个位置的行向量堆成 $n\times d$ 的矩阵，依次与 $W_1$、$W_2$ 相乘），这也是 FFN 在 GPU 上效率较高的原因。

### 3.3.2 激活函数：从 ReLU 到 SwiGLU

ReLU 是最初的选择，GELU（$x\Phi(x)$）因平滑且在 0 附近有非零梯度而被 BERT、GPT-2 广泛采用。当前主流是门控线性单元家族，Llama 系列采用的 SwiGLU 形式为

$$\mathrm{SwiGLU}(x)=\big(\mathrm{SiLU}(xW_{gate})\odot xW_{up}\big)W_{down},\qquad \mathrm{SiLU}(z)=z\,\sigma(z)$$

它包含三个矩阵，参数量为 $3d d_{ff}$。为保持与 ReLU 版相当的参数与计算，实践中把 $d_{ff}$ 取到 $\frac83 d$ 附近再向上取整到 256 的倍数（例如 $d=4096$ 时 Llama 系列公开配置为 $d_{ff}=11008$，即 $2.69d$，此时 $3dd_{ff}\approx 8.06d^2$，与 $8d^2$ 基本持平）。

### 3.3.3 统计解释：自适应基函数展开

把隐层展开写出来：$h_r=f(w_r^\top x+b_r)$，$r=1,\dots,d_{ff}$，输出第 $s$ 个分量 $y_s=\sum_{r}\beta_{sr}h_r$。这是一个标准的基函数回归形式：先把输入投影到 $d_{ff}$ 个非线性特征上，再做线性组合。与样条、小波等固定基不同的是，$w_r$ 本身是学习出来的，因此 FFN 属于投影寻踪回归（projection pursuit）与筛法（sieve estimation）意义上的自适应基展开：$d_{ff}$ 就是基函数的个数，也就是筛空间的维数，其大小决定近似能力，也决定过拟合风险，这与非参数回归中选择基个数的偏差—方差权衡一致。

另一条解释路径把 FFN 看作键值记忆（key-value memory）：$W_1$ 的每一行 $w_r$ 是一个模式检测器（内部点积 $w_r^\top x$ 度量匹配度），激活值 $h_r$ 是匹配得分，$W_2$ 的每一列则是被召回的内容。按这一视角，$d_{ff}$ 就是记忆槽的个数；这解释了为什么 FFN 的参数占比高，模型的"事实性知识"主要存放在这里，也同样解释了为什么 FFN 是知识编辑类研究的首选目标。

| 组件 | 参数量（$d_{ff}=4d$） | 每层占比 | 统计学对应物 |
|------|---------------------|---------|-------------|
| 注意力投影 $W_Q,W_K,W_V,W_O$ | $4d^2$ | 约 1/3 | 可学习的核加权（平滑矩阵的行） |
| FFN $W_1,W_2$（含偏置） | $8d^2$ | 约 2/3 | 自适应基函数展开 / 记忆字典 |
| 归一化与残差 | $O(d)$ | 可忽略 | 标准化、恒等捷径（残差分析） |

<!--PART2-->

## 3.4 LayerNorm、RMSNorm与残差连接

### 3.4.1 LayerNorm 的定义与归一化的方向

一个完整 Transformer 层包含：Multi-Head Attention（多头并行捕获不同语义关系）、Feed-Forward Network（两层 MLP，隐藏维度通常 4× 主维度）、LayerNorm/RMSNorm（层归一化稳定训练）、残差连接（防止梯度消失）。

对单个 token 的向量 $x\in\mathbb{R}^d$，LayerNorm 的计算是

$$\mu=\frac1d\sum_{r=1}^{d}x_r,\qquad \sigma^2=\frac1d\sum_{r=1}^{d}(x_r-\mu)^2,\qquad y=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$$

其中 $\gamma,\beta\in\mathbb{R}^d$ 是可学习的缩放与平移参数，$\epsilon$ 是数值稳定项。要点在于求期望的方向：LayerNorm 在特征维度上取均值与方差，即对一条观测内部的多个变量做标准化；BatchNorm 则在 batch 维度上对每个特征通道取统计量，即对多个观测做标准化。这个区别带来了以下差异：

| 维度 | BatchNorm | LayerNorm | RMSNorm |
|------|-----------|-----------|---------|
| 统计量方向 | 跨样本，逐通道 | 跨特征，逐样本 | 跨特征，逐样本（仅二阶矩） |
| 训练/推理是否一致 | 否，推理用滑动平均统计量 | 是 | 是 |
| 对 batch size 敏感 | 敏感，小 batch 下统计量噪声大 | 不敏感 | 不敏感 |
| 处理变长序列 | 需要 padding 掩码配合 | 天然适配 | 天然适配 |
| 是否减均值 | 是 | 是 | 否 |
| 可学习参数 | $\gamma,\beta$ | $\gamma,\beta$ | $\gamma$ |
| 典型使用者 | 早期 CNN 网络 | 原始 Transformer、GPT-2、ViT | Llama、Qwen、DeepSeek |

从统计学视角看，LayerNorm 是一次逐单元的标准化（z-score），但它丢弃了原本的尺度信息，再交由 $\gamma,\beta$ 学习恢复。原始动机（"缓解内部协变量偏移"）后来受到了质疑，后续研究更倾向于把归一化的收益归因于损失曲面条件的改善（Hessian 谱更集中、可用更大学习率）。对实践者的启示是：归一化层应被视为优化器的辅助设施，而不是某种必须存在的建模假设。

### 3.4.2 Pre-LN 与 Post-LN

残差块的内部排列有两种标准写法：

$$\text{Post-LN: } x_{l+1}=\mathrm{LN}\big(x_l+\mathrm{Sublayer}(x_l)\big)\qquad \text{Pre-LN: } x_{l+1}=x_l+\mathrm{Sublayer}\big(\mathrm{LN}(x_l)\big)$$

原始 Transformer 使用 Post-LN，在不使用学习率 warmup 时容易在深层网络上发散；Pre-LN 使每个子层的输入都先被归一化，残差通路成为一条近乎恒等的干净通道，梯度可以直接回传，因此现代大模型的标准做法是 Pre-LN（或 RMSNorm 版本的 Pre-LN）。代价是 Pre-LN 下残差流的方差随深度近似线性累加，深层模型需要在初始化时按层数缩放残差分支的方差，或引入额外的残差缩放系数。

### 3.4.3 RMSNorm

RMSNorm 删除了均值减法与偏置项，只保留二阶矩归一化：

$$\mathrm{RMS}(x)=\sqrt{\frac1d\sum_{r=1}^{d}x_r^2+\epsilon},\qquad y=\gamma\odot\frac{x}{\mathrm{RMS}(x)}$$

删除中心化步在经济上是可解释的，Transformer 内部多数层的激活均值本就接近 0，且缺失之处在于它将省下约一次全量规约。它以略简单的形式实现了与层归一化的可比质量，论文见 arXiv:1910.07467。目前 Llama、Qwen、DeepSeek 等大模型都改用 RMSNorm。

### 3.4.4 残差连接与路径分析

残差结构 $y=x+F(x)$ 使网络在初始化附近近似恒等映射，梯度沿 $\partial y/\partial x=I+\partial F/\partial x$ 传播时至少保留一条不衰减的通路，这让上千层的复合依然可训练。这里还有一种更贴切的理解：把各层的输出视为向一条公共"残差流"（residual stream）中累加的增量，每个 Transformer 层读取该流、计算增量、再写回；这使深层网络的行为不像一个纵向加深的函数复合，而更像一次迭代式的估计更新，与 boosting 式的加性更新形态接近。

### 3.4.5 Dropout 与其他正则化

Dropout 在训练时按概率 $p$ 屏蔽神经元或整条通路并对保留部分做 $1/(1-p)$ 缩放，等价于注入乘性的伯努利噪声：在线性回归情形下它与加权岭惩罚近似等价，在一般网络上则可以理解为对大量子网络做的加权模型平均，也可以理解为对 spike-and-slab 先验的一种粗略近似。需要注意的是，现代大模型的预训练阶段经常设 dropout=0，因为单个 epoch 下的巨大数据量已足够提供正则效果，而 dropout 的噪声会拖慢收敛；残差 dropout、注意力 dropout 更常见于较小模型或微调阶段。此外还有两类在当前大模型中频繁出现的正则手段：一是对 logit 先做归一化（QK-Norm）抑制注意力熵坍塌，二是在 MoE 里的路由器 z-loss（详见 3.7 节）。weight decay 与 $\ell_2$ 惩罚的区别在 AdamW 中被明确区分：只有 AdamW 将衰减项独立于自适应动差缩放施加。

## 3.5 位置编码演进

由于自注意力本身不感知顺序，需注入位置信息：

| 位置编码方案 | 核心思想 | 论文 | 开源 |
|-------------|---------|------|------|
| 绝对位置编码 | 为每个位置学习一个向量 | 原始 Transformer 论文 | — |
| RoPE 旋转位置编码 | 用复数旋转矩阵编码相对位置 | https://arxiv.org/abs/2104.09864 | https://github.com/ZhuiyiTechnology/roformer |
| ALiBi | 距离越远注意力偏置越大 | https://arxiv.org/abs/2108.12409 | https://github.com/ofirpress/attention_with_linear_biases |

RoPE 是当前的主流。DeepSeek/Qwen/Llama 均采用，支持 YaRN 外推扩展至 1M 上下文。

### 3.5.1 Sinusoidal 位置编码

原始 Transformer 采用确定性三角函数：

$$\mathrm{PE}_{(pos,2i)}=\sin\Big(\frac{pos}{10000^{2i/d}}\Big),\qquad \mathrm{PE}_{(pos,2i+1)}=\cos\Big(\frac{pos}{10000^{2i/d}}\Big)$$

两条性质使它至今仍在被分析。一是多分辨率结构：第 $i$ 维对应的波长为 $2\pi\cdot 10000^{2i/d}$，从 $2\pi$ 一直增长到万级，本质上是一组按几何级数排布的傅里叶基；二是线性递推性质，存在与 $\Delta$ 有关的线性算子 $T_\Delta$ 使 $\mathrm{PE}_{pos+\Delta}=T_\Delta\,\mathrm{PE}_{pos}$，因此内积意义下模型有机会学到"相对距离"。后者正是相对位置编码的出发点。

### 3.5.2 可学习位置编码

BERT（arXiv:1810.04805）与 GPT-2 采用的是可学习绝对位置嵌入：把位置索引当作额外的词，用一张 $\mathbb{R}^{L_{max}\times d}$ 的查找表参与端到端训练。它实现简单、拟合能力强，缺点是长度外推失效，超出 $L_{max}$ 的位置从未被训练过，参数在 $L_{max}$ 处突发式失效。折中路线是相对位置偏置（如 T5 的桶化相对偏置），它把 $|i-j|$ 分桶后给每个桶一个标量偏置加到 logit 上，长度不变性更好，代价是仍然不能处理任意长距离。

### 3.5.3 RoPE 的旋转矩阵形式

RoPE（Rotary Position Embedding，见 arXiv:2104.09864，Rotary Embedding 原名为 RoFormer）的设计目标非常明确：构造 $f_q,f_k$，使内积只依赖相对位置，即

$$\big\langle f_q(x_m,m),\,f_k(x_n,n)\big\rangle=g\big(x_m,x_n,m-n\big)$$

做法是把 $d$ 维向量按相邻两维分组成 $d/2$ 个复数 $z_i=x_{2i}+\mathrm{i}\,x_{2i+1}$，令 $\theta_i=10000^{-2i/d}$，定义分块对角旋转矩阵

$$R_{\Theta,m}=\begin{pmatrix}
\cos m\theta_1 & -\sin m\theta_1 & & & \\
\sin m\theta_1 & \cos m\theta_1 & & & \\
& & \ddots & & \\
& & & \cos m\theta_{d/2} & -\sin m\theta_{d/2}\\
& & & \sin m\theta_{d/2} & \cos m\theta_{d/2}
\end{pmatrix}$$

取 $f_q(x_m,m)=R_{\Theta,m}W_qx_m$、$f_k(x_n,n)=R_{\Theta,n}W_kx_n$，利用旋转矩阵的正交性与 $R_m^\top R_n=R_{n-m}$，得到

$$\big\langle f_q,f_k\big\rangle=\big(W_qx_m\big)^\top R_{\Theta,m}^\top R_{\Theta,n}\big(W_kx_n\big)=\big(W_qx_m\big)^\top R_{\Theta,n-m}\big(W_kx_n\big)$$

只依赖 $n-m$，达到相对位置编码的目标；且由于每维对应不同的旋转角频率，它与 Sinusoidal 一样具有多分辨率结构。实践中的典型性质是注意力得分随相对距离呈衰减趋势，这与自然语言中邻近词相关性更高的经验一致。

工程实现通常不真的构造 $d\times d$ 分块矩阵，而是预先计算好包含 $\cos$、$\sin$ 的旋转因子缓存，再按分量做交错乘法：

```python
import torch

def precompute_freqs_cis(dim: int, seq_len: int, base: float = 10000.0):
    theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    pos = torch.arange(seq_len).float()
    freqs = torch.outer(pos, theta)
    return torch.polar(torch.ones_like(freqs), freqs)      # (seq_len, dim/2) complex

def apply_rotary_emb(xq, xk, freqs_cis):
    def to_complex(t):
        t = t.float().reshape(*t.shape[:-1], -1, 2)
        real, imag = t.unbind(-1)
        return torch.complex(real, imag)
    xq_c, xk_c = to_complex(xq), to_complex(xk)
    f = freqs_cis.unsqueeze(1)                              # (n, 1, Dh/2)
    out_q = torch.view_as_real(xq_c * f).flatten(-2)
    out_k = torch.view_as_real(xk_c * f).flatten(-2)
    return out_q.type_as(xq), out_k.type_as(xk)

# 实例化
B, n, H, d_head = 2, 6, 4, 8
q = torch.randn(B, n, H, d_head)
k = torch.randn(B, n, H, d_head)

freqs_cis = precompute_freqs_cis(d_head, n)                 # 只需前 n 个位置
q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis)

print(q_rot.shape)   # torch.Size([2, 6, 4, 8])

# 验证 RoPE 的关键性质：内积只依赖相对位置
# 取相对位置差为 2 的两对 (q_i, k_{i+2})，旋转后内积应与绝对位置无关
q_rot_b = q_rot[0, :, 0, :]   # 去掉 batch 和 head 维方便观察
k_rot_b = k_rot[0, :, 0, :]
print((q_rot_b[0] @ k_rot_b[2]).item(), (q_rot_b[3] @ k_rot_b[5]).item())
# 两个值一般不相等（因为 q,k 本身随机不同），但若比较同一对 (q, k) 在不同绝对位置的旋转结果，内积相同
```
输出
```text
torch.Size([2, 6, 4, 8])
2.3643617630004883 -3.113664150238037
```

#### 相对位置不变性的数值验证

RoPE 的全部价值都压在 $\langle f_q(x_m,m),f_k(x_n,n)\rangle$ 只依赖 $m-n$ 这一条性质上，值得用数值实验直接检验。下面用 numpy 实现 $d=64$ 的 RoPE，依次验证三件事：旋转保持范数（正交性）、内积在查询与键的整体平移下不变、以及同一向量的内积随相对距离的变化。

```python
import numpy as np

def apply_rope(x, pos, base=10000.0):
    """x: (dim,)，pos: 标量位置。按相邻两维分组做平面旋转。"""
    dim = x.shape[-1]
    theta = base ** (-np.arange(0, dim, 2) / dim)   # 每对维度的角频率
    ang = pos * theta
    cos, sin = np.cos(ang), np.sin(ang)
    out = np.empty_like(x)
    out[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin
    out[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos
    return out

rng = np.random.default_rng(2)
dim = 64

# 一、正交性：范数保持
x = rng.normal(size=dim)
for m in [0, 7, 133, 4096]:
    print(f"pos={m:>5}: |R_m x| = {np.linalg.norm(apply_rope(x, m)):.12f}")

# 二、相对位置不变性：<R_m q, R_n k> 在整体平移下不变
q = rng.normal(size=dim)
k = rng.normal(size=dim)
worst = 0.0
for (m, n) in [(3, 10), (100, 200), (5, 8), (1000, 1500), (7, 899)]:
    a = apply_rope(q, m) @ apply_rope(k, n)
    b = apply_rope(q, m + 500) @ apply_rope(k, n + 500)   # 两个位置同时 +500
    worst = max(worst, abs(a - b))
    print(f"(m,n)=({m:>4},{n:>4})  d={n-m:>4}  <R_m q, R_n k> = {a:>12.8f}  "
          f"平移后 = {b:>12.8f}")
print("所有配对的最大偏差 =", worst)

# 三、同一向量随相对距离的内积（衰减趋势）
v = rng.normal(size=dim)
base = np.linalg.norm(v) ** 2
for dlt in [0, 1, 2, 4, 8, 16, 32, 64, 128]:
    ip = apply_rope(v, 0) @ apply_rope(v, dlt)
    print(f"相对距离 d={dlt:>4}: <R_0 v, R_d v> / |v|^2 = {ip/base:.4f}")
```

运行输出（实测，环境 numpy 1.26.4）：

```text
pos=    0: |R_m x| = 7.780780250995
pos=    7: |R_m x| = 7.780780250995
pos=  133: |R_m x| = 7.780780250995
pos= 4096: |R_m x| = 7.780780250995
(m,n)=(   3,  10)  d=   7  <R_m q, R_n k> =  -2.67005614  平移后 =  -2.67005614
(m,n)=( 100, 200)  d= 100  <R_m q, R_n k> =  -5.15552934  平移后 =  -5.15552934
(m,n)=(   5,   8)  d=   3  <R_m q, R_n k> =   5.08542143  平移后 =   5.08542143
(m,n)=(1000,1500)  d= 500  <R_m q, R_n k> =  -3.47358969  平移后 =  -3.47358969
(m,n)=(   7, 899)  d= 892  <R_m q, R_n k> =   4.03308118  平移后 =   4.03308118
所有配对的最大偏差 = 1.1368683772161603e-13
相对距离 d=   0: <R_0 v, R_d v> / |v|^2 = 1.0000
相对距离 d=   1: <R_0 v, R_d v> / |v|^2 = 0.9729
相对距离 d=   2: <R_0 v, R_d v> / |v|^2 = 0.9035
相对距离 d=   4: <R_0 v, R_d v> / |v|^2 = 0.7541
相对距离 d=   8: <R_0 v, R_d v> / |v|^2 = 0.7254
相对距离 d=  16: <R_0 v, R_d v> / |v|^2 = 0.6935
相对距离 d=  32: <R_0 v, R_d v> / |v|^2 = 0.6878
相对距离 d=  64: <R_0 v, R_d v> / |v|^2 = 0.4430
相对距离 d= 128: <R_0 v, R_d v> / |v|^2 = 0.2502
```

三项结果与理论一致：范数在任何位置都精确保持（旋转是正交变换）；五组位置配对在整体平移 500 后内积偏差不超过 $10^{-13}$（浮点误差量级），相对位置不变性成立，对应 $R_m^\top R_n=R_{n-m}$。距离衰减一组需要留意解读：单一随机向量的内积曲线并非严格单调（上表在 $d=8$ 到 $32$ 之间出现平台），"衰减趋势"是对频率分量与向量分布取平均后的整体性质；训练后的模型中观察到注意力得分随相对距离衰减、与自然语言中邻近词相关性更高的经验一致，但不能把单条曲线的单调性当成 RoPE 的数学保证。

### 3.5.4 ALiBi

ALiBi（Attention with Linear Biases，arXiv:2108.12409）不修改 $Q,K$，而是在 logit 上直接加线性距离惩罚：

$$s_{ij}=\frac{q_i^\top k_j}{\sqrt{d_k}}-m\,|i-j|$$

第 $h$ 个头的斜率按几何级数分配，常用形式 $m_h=2^{-8h/H}$（$h=1,\dots,H$），即不同头具备不同强度的局部性偏好。它的优势是零参数、实现简单、长度外推能力好；劣势是不区分方向（$|i-j|$ 对前文与后文对称），因此在需要区分"前文/后文"的位置敏感任务上不如 RoPE。

### 3.5.5 外推：从位置插值到 YaRN

把 RoPE 模型扩展到超出训练长度时，常用方法有三类：位置插值（position interpolation）把 $pos$ 压缩到原区间内；NTK-aware 缩放直接修正基数 $10000$ 使高频分量保持、低频分量被拉伸；YaRN 则在 NTK 思路基础上引入按维度的斜坡插值并对注意力 logit 做温度补偿，对不同上下文长度的适应性更强。这些方法操作的都是同一个对象，$\theta_i$ 与 $pos$ 的映射关系，因此属于同一族可以统一分析的干预。主流开源模型公开了各自的长上下文版本，常见的做法是先在较短上下文上预训练、再在长上下文上做少量退火，并配合 YaRN 等缩放策略把上下文扩展到几十万乃至百万 token 的量级；这类数字随版本更新变化很快，核对时以官方模型卡为准。

### 3.5.6 四种方案对比

| 方案 | 位置信息类型 | 是否可长度外推 | 额外参数量 | 计算开销 | 典型使用者 |
|------|-------------|---------------|-----------|---------|-----------|
| Sinusoidal | 绝对（内积呈相对性） | 弱 | 0 | 预计算后加到输入 | 原始 Transformer |
| 可学习绝对嵌入 | 绝对 | 否 | $L_{max}d$ | 查表 | BERT、GPT-2 |
| RoPE | 相对（旋转 invariance） | 中等，配合 YaRN 类方法可扩展 | 0 | $q,k$ 的逐元素旋转 | Llama、Qwen、DeepSeek |
| ALiBi | 相对（偏置形式） | 较好 | 0 | logit 上加标量斜坡 | Bloom 等早期长上下文模型 |

## 3.6 注意力效率优化

标准 MHA（Multi-Head Attention）在长文本下 KV 缓存爆炸，发展演进：

```
MHA → MQA（Multi-Query） → GQA（Grouped-Query） → MLA（Multi-head Latent Attention）
```

MLA（DeepSeek 独创）：将 KV 压缩到低秩潜在空间，KV 缓存降至 MHA 的 1/57：

| 方案 | KV 缓存（相对值） | 代表模型 |
|------|---------------|---------|
| MHA | 100% | 原始 Transformer |
| GQA | ~25% | Llama 3.1 |
| MQA | ~12.5% | 早期 PaLM |
| MLA | ~1.75%（1/57） | DeepSeek-V3/R1 |

### 3.6.1 复杂度与显存的定量分析

训练时注意力部分的浮点运算约为 $O(n^2 d)$（两次矩阵乘法各一个 $n^2$ 项乘上对头维度），中间注意力矩阵本身是 $O(n^2)$ 的显存，因此序列长度加倍带来四倍计算与四倍注意力显存。推理时瓶颈转为 KV 缓存：生成阶段需要复用此前所有位置的 $K,V$，每 token 的缓存量可由下式直接算得：

$$M_{\text{token}}=2\times L\times n_{kv}\times d_{head}\times b$$

其中 $L$ 为层数，$n_{kv}$ 为 KV 头数，$d_{head}$ 为每头维度，$b$ 为每个元素占用的字节数（FP16 为 2，FP8 为 1），因子 2 来自 K 与 V 两份。以 $L=32$、$n_{kv}=32$、$d_{head}=128$、FP16 代入，得到每 token 约 0.5 MB；十万 token 的上下文单条序列就要约 50 GB 的缓存，超过单张高端卡的显存。这条算式可以直接解释为什么长上下文服务往往被显存而非算力卡住，也是 KV 缓存量化、分页管理与架构改良（MLA）三件事同时存在的理由。

### 3.6.2 MQA、GQA 与 MLA

多查询注意力（MQA）让所有 query 头共享单一 KV 头，把缓存压到 $1/H$；分组查询注意力（GQA）在中间取值，把 $H$ 个 query 头分成 $g$ 组，每组共享一份 KV，主流配置是 8 个 KV 头对 32–64 个 query 头，质量损失通常不到一个点的困惑度。

多头潜在注意力（MLA）走的是低秩压缩路线：把每个 token 的输入隐向量压缩为一个低维潜在向量 $c^{KV}_t=W^{DKV}h_t\in\mathbb{R}^{d_c}$（$d_c\ll H d_{head}$），K 与 V 由它再投影得到。推理阶段由于 $q^\top W^{UK}c=(W^{UK\top}q)^\top c$，上投影矩阵可以吸收进 $W_Q$，于是只需缓存 $c^{KV}_t$，这就是上表中 KV 缓存大幅下降的来源。需要注意 RoPE 与低秩吸收之间不直接兼容，实际实现要把带旋转的部分解耦出来单独处理，这是 MLA 实现复杂度的主要来源。

#### KV 缓存结构的直观对比

```text
每 token 需缓存的 K/V 块（示意，一格 = 一个头的一份 d_head 维向量）

MHA   q 头 32，KV 头 32   KV: ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■  32 份
MQA   q 头 32，KV 头 1    KV: ■                                     1 份（全部 q 头共享）
GQA   q 头 32，KV 头 8    KV: ■■■■■■■■                             8 份（每 4 个 q 头共享 1 份）
MLA   q 头 32             KV: c ∈ R^dc                              1 份低秩潜在向量
                                                                     （dc << 32*d_head，
                                                                       K/V 由 c 上投影即时重建）
```

四者的差别只在"缓存里放什么"：MHA 每个头各存一份 K 与 V；MQA 全部查询头共用一份；GQA 在两者之间分组共享；MLA 干脆不存 K、V 本身，改存一个低维潜在向量 $c^{KV}$，推理时需要哪一头的 K、V 就用吸收后的投影即时重建。前三种是精确的头共享，质量损失主要来自共享带来的表达约束；MLA 是有损压缩思路，损失由潜在维数 $d_c$ 控制。

#### KV 缓存大小的数值核算

3.6.1 的公式 $M_{\text{token}}=2Ln_{kv}d_{head}b$ 只需几十行就能核算一次部署（以 Llama 3 公开的 8B 与 70B 配置为参照，配置取自官方模型卡）：

```python
def cache_bytes(L, n_kv, d_head, b):
    return 2 * L * n_kv * d_head * b          # 因子 2 = K 与 V 两份

configs = [
    # 名称, 层数 L, query 头数 H, kv 头数 n_kv, 头维 d_head
    ("Llama-3-8B 量级", 32, 32, 8, 128),
    ("同规模假想 MHA", 32, 32, 32, 128),
    ("同规模假想 MQA", 32, 32, 1, 128),
    ("Llama-3-70B 量级", 80, 64, 8, 128),
]

for name, L, H, Hkv, dh in configs:
    per = cache_bytes(L, Hkv, dh, 2)          # b=2：FP16
    mb_per = per / 1024 ** 2
    gb_32k = per * 32768 / 1024 ** 3
    print(f"{name:<18}{mb_per:>12.3f} MB/token{gb_32k:>9.2f} GB@32K"
          f"  相对MHA={Hkv/H:.2f}x")

per = cache_bytes(32, 8, 128, 1)              # FP8：b=1
print(f"GQA n_kv=8, FP8: {per/1024:.1f} KB/token, 32K 上下文 "
      f"{per*32768/1024**3:.2f} GB")
```

运行输出（实测）：

```text
Llama-3-8B 量级         0.125 MB/token    4.00 GB@32K  相对MHA=0.25x
同规模假想 MHA          0.500 MB/token   16.00 GB@32K  相对MHA=1.00x
同规模假想 MQA          0.016 MB/token    0.50 GB@32K  相对MHA=0.03x
Llama-3-70B 量级        0.312 MB/token   10.00 GB@32K  相对MHA=0.12x
GQA n_kv=8, FP8: 64.0 KB/token, 32K 上下文 2.00 GB
```

读数要点：GQA（8 个 KV 头）把 Llama-3-8B 量级模型每 token 的缓存从 0.5 MB 压到 0.125 MB，一条 32K 上下文从 16 GB 降到 4 GB，这是主流开源模型普遍采用 GQA 的直接算术理由；MQA 的 0.5 GB 虽然更低，但单份 KV 要服务全部 32 个查询头，质量代价相对更高；FP8 把数字再减半，属于与架构无关的收益（代价是数值精度）。70B 一档的读数还说明了另一件事：KV 缓存随层数与头数线性增长，模型变大后长上下文服务的显存压力增长得比参数量更快。

### 3.6.3 精确加速：FlashAttention 与 PagedAttention

并不需要改变数学模型也能获得数量级的性能提升。FlashAttention（arXiv:2205.14135）与 FlashAttention-2（arXiv:2307.08691）是 IO 感知的精确算法：通过分块（tiling）把注意力矩阵的前向与反向计算放在 SRAM 中完成，并采用在线 softmax 增量更新归一化因子，避免把 $n\times n$ 矩阵写回 HBM，从而把显存从 $O(n^2)$ 降到 $O(n)$、大幅缩短墙钟时间，而数学结果与标准注意力完全一致（浮点误差范围内）。vLLM 提出的 PagedAttention（arXiv:2309.06180）借鉴操作系统虚拟内存的分页思想，把 KV 缓存按块分配与管理，使变长请求之间的显存碎片基本消除，显著提升吞吐。这两项是当前开源推理引擎的通用底座。

### 3.6.4 近似方法：稀疏化与随机特征

另一条路线是改变 attention 的数学形式。稀疏注意力只计算若干 pattern 位置（滑动窗口、膨胀窗口、随机块、全局 token），把复杂度降到 $O(n\sqrt n)$ 或 $O(n\log n)$；线性注意力则用核技巧把 $\exp(q^\top k/\sqrt d)$ 分解为特征映射的内积 $\phi(q)^\top\phi(k)$，从而把 $(\phi(Q)\phi(K)^\top)V$ 重排为 $\phi(Q)(\phi(K)^\top V)$，复杂度降到 $O(n)$。Performer（arXiv:2009.14794 提出的 FAVOR+）用的正是随机特征近似。

对统计学者而言，这一支与核方法的近似技术完全是同一套工具：softmax 核的随机傅里叶特征、Nyström 型低秩近似、以及对 Gram 矩阵做采样的Nyström 方法，早在核机器时代就被系统研究过。它们的误差性质也高度类似：近似带来一层额外的偏差（核近似误差），其方差随随机特征个数以 $O(1/\sqrt R)$ 衰减。这里的实践教训是，评测线性注意力时不能只看短上下文的困惑度，因为偏差主要体现在需要精确检索的长距离任务上。

| 方法 | 训练复杂度 | 推理每步内存 | 是否精确 | 主要代价 |
|------|-----------|-------------|---------|---------|
| 标准 MHA + Fused kernel | $O(n^2d)$ | $O(n)$ 缓存 + 无 $n^2$ 矩阵 | 是 | 得依赖 IO 感知实现 |
| FlashAttention-2 | $O(n^2d)$ | $O(n)$ | 是 | 需要定制 kernel |
| 滑动窗口 / 稀疏注意力 | $O(n\,w\,d)$ | $O(w)$ | 否（有 pattern 偏差） | 长距离检索丢失 |
| 线性注意力 / 随机特征 | $O(n\,d\,R)$ | $O(1)$（递归形式） | 否（核近似误差） | 精确召回能力下降 |
| MQA / GQA | $O(n^2d)$ | 降至 $1/H$ ~ $1/4$ | 是 | 轻微质量损失 |
| MLA | $O(n^2d)$ | 显著低于 MHA | 是 | 实现复杂度、与 RoPE 需解耦 |

<!--PART3-->

## 3.7 MoE混合专家架构

MoE 将模型划分为多个"专家"子网络，路由器为每个 token 只激活 Top-k 专家：

- DeepSeek-V3：671B 总参数，每 token 激活 37B；1 个共享专家 + 256 路由专家，激活 8 个路由专家
- Qwen3-235B-A22B：235B 总参数，每 token 激活 22B
- Qwen3.8-Flash：125B 主参数 + 51B N-gram Embedding，激活仅 6B

核心优势：计算量降至全参数的 10% 以下，却保持接近全参数的性能。DeepSeek-V3 的 FP8 混合精度训练仅用 2.788M H800 GPU 小时（约 557 万美元）。

### 3.7.1 路由器与 Top-k 稀疏化

设第 $t$ 个 token 在某一层的输入隐向量为 $x_t\in\mathbb{R}^d$，专家 $\{E_i\}_{i=1}^{N}$ 是 $N$ 个同构的子网络（通常是该层的 FFN）。路由权重由一个线性路由器给出：

$$g(x)=\mathrm{TopK}\big(\mathrm{softmax}(W_r x)\big),\qquad W_r\in\mathbb{R}^{N\times d}$$

$\mathrm{TopK}$ 保留前 $k$ 个分量（通常 $k=1$ 或 2，大模型中也有 8），其余置零；层输出为被激活专家的加权和

$$y_t=\sum_{i=1}^{N}g_i(x_t)\,E_i(x_t)=E_{s}(x_t)+\sum_{i\in\mathcal{T}(x_t)}g_i(x_t)\,E_i(x_t)$$

第二式是"共享专家 + 路由专家"的写法：$E_s$ 对每个 token 都参与（承载通用知识），$\mathcal{T}(x_t)$ 是被选中的路由专家集合。Shazeer 等人（arXiv:1701.06538）最早提出的带噪门控形式为 $h_i=(xW_g)_i+\mathcal{N}(0,1)\cdot\mathrm{softplus}((xW_{noise})_i)$ 再取 TopK，噪声项的作用是打破对称性、让专家在训练早期不至于完全一致。

由此得到的收益是"参数量与计算量解耦"：前向传播的浮点量只与被激活的 $k$ 个专家有关，总参数可以扩大 $N/k$ 倍而算力基本不变。

路由的发生过程可以画成一次"打分—排序—分发"：

```text
              token 隐向量 x_t (维度 d)
                      │
               路由器 W_r (N × d)
                      │
            N 个专家的打分 z = W_r x_t
                      │
               softmax → 门控概率 g
                      │
            Top-k（k=1 或 2，大模型可到 8）
                      │
      ┌───────┬───────┼───────┬───────┐
      ▼       ▼       ▼       ▼       ▼
   专家 1   专家 2   专家 3   专家 4  ... 专家 N
    (FFN)   (FFN)   (FFN)   (FFN)      (FFN)
      │       │       ×       ×        ×     未选中：不计算
      └── g_1·E_1(x) + g_2·E_2(x) ────────┘   加权和作为层输出
```

每 token 只走 $k$ 个分支，其余专家对它完全沉默；不同 token 走的分支不同，这就是"参数全体共享、计算按需分配"的机制基础。图中被跳过的分支在物理上对应跨设备的 token 分发（all-to-all 通信），负载不均时该通信会成为瓶颈，这是 3.7.2 节均衡机制的动机。

### 3.7.2 负载均衡与训练稳定性

稀疏激活带来一个新问题：如果路由倾向于少数专家，多数专家得不到训练（功能退化），少数专家过载（成为计算瓶颈），这就是路由坍塌（routing collapse）。Switch Transformer（arXiv:2101.03961）引入的辅助损失至今仍是基准做法。记

$$f_i=\frac{1}{T}\sum_{t}\mathbb{1}\{i\in\mathcal{T}(x_t)\}\quad\text{（实际分发到专家 }i\text{ 的 token 比例）},\\ P_i=\frac{1}{T}\sum_{t}\mathrm{softmax}(W_rx_t)_i\quad\text{（路由概率均值）}$$

辅助损失取两者的内积：

$$\mathcal{L}_{aux}=\alpha\cdot N\sum_{i=1}^{N} f_i P_i$$

在均匀分布 $f_i=P_i=1/N$ 时该损失取最小值 $\alpha\cdot 1$，因此它会把两种"份额"（离散的实际分配量与连续的期望分配量）同时推向均匀。工程上要注意 $f_i$ 依赖离散决策、不可微分，实现时通常把它当作常数处理，梯度只经由 $P_i$ 回传。

除此之外还有三件必须配套的机制：

- 容量因子（capacity factor）：为每专家设定每批最多可接收的 token 数 $\lceil c\cdot T/N\rceil$，溢出 token 直接走残差通路绕过专家层（等价于被"丢弃"），避免不同设备的负载不均导致的同步等待。
- z-loss（ST-MoE 提出，arXiv:2202.08906）：惩罚过大的路由 logit 幅值 $\propto\sum_b\big(\log\sum_j e^{z_{bj}}\big)^2$，提升混合精度训练下路由的数值稳定性。
- 无辅助损失的均衡（DeepSeek-V3 采用）：给每个专家维护一个仅参与路由选择、不参与梯度回传的偏置项 $b_i$，按"某专家过载则减小偏置"的规则在线更新，从而在几乎不损失模型质量的前提下维持均衡。

### 3.7.3 三个实例与参数拆解

| 模型 | 总参数 / 激活参数 | 专家配置 | 配套机制 |
|------|-----------------|---------|---------|
| DeepSeek-V3 | 671B / 37B | 256 路由专家 + 1 共享专家，top-8 | MLA、无辅助损失均衡、节点受限路由、多 token 预测 |
| Qwen3-235B-A22B | 235B / 22B | 大规模路由专家，top-k 稀疏激活 | 思考/非思考双模式切换 |
| Mixtral 8x7B | 约 47B / 约 13B | 8 专家，top-2 | 密集 FFN 替换为 MoE 层的早期开源代表 |

MoE 的代价。一是显存：所有专家的权重都必须常驻，显存需求随总参数增长，与激活参数量无关，这也是为什么 MoE 模型的部署往往需要多卡并行。二是通信：专家并行需要跨设备的 token 分发与收集（all-to-all），对小集群与推理延迟都不友好。三是训练稳定性：路由的离散决策使损失曲线更容易出现尖峰，通常需要更保守的学习率与更强的监控。四是微调难度：稀疏路由在小数据集上极易过拟合，微调阶段常冻结路由或回退到密集配置。

### 3.7.4 统计学解释：它本来就是混合专家模型

MoE 这个缩写并非新造。"Adaptive Mixtures of Local Experts"（Jacobs、Jordan、Nowlan、Hinton，1991）提出的分层混合专家模型早已给出了完全相同的形式：每个专家 $E_i$ 是一个（当时是线性的）局部回归器，门控网络输出的就是各专家的后验责任 $\Pr(\text{专家 }i\mid x)$，整个模型是一个条件混合模型 $p(y\mid x)=\sum_i g_i(x)\,p(y\mid x,E_i)$，用极大似然（或其 EM 形式）训练。LLM 中的 MoE 与它只有三点差异：专家换成了 FFN，推理时改用硬 Top-k 而不是全量 soft 责任，以及需要在目标函数中加入额外的均衡约束。

由此可以借用混合模型研究中积累的全部经验来理解 routed training 的现象：

| MoE 训练中的现象 | 混合模型中的对应结论 |
|-----------------|--------------------|
| 路由坍塌（token 集中到少数专家） | 混合成分退化；似然面存在奇异性，成分坍缩到单点是已知的病态解 |
| 负载均衡辅助损失 | 先验约束或惩罚项，防止 EM 迭代走到退化解 |
| 专家数量 $N$ 的选择 | 成分个数的选择；过多则多数成分为空，带来不必要的自由度与识别问题 |
| 专家功能的不可识别性 | 混合模型的标签交换（label switching）与一般性不可识别，需要额外约束才能谈"专家负责什么" |
| 路由的不确定性 | 后验责任的不确定性；温度或噪声门控相当于对责任分布做平滑 |

一个由此而来的研究议题是：既然专家 × token 的分配本质是一个潜在类别分配，那么能否像 EM 那样显式地估计其不确定性，并用它做路由诊断、专家合并或剪枝，这在目前的工程实践中较少。

## 3.8 非Transformer架构

有一批模型抛弃了注意力机制，改用循环结构或状态空间模型，核心动机是解决 Transformer 的二次方复杂度问题（注意力计算量随序列长度平方增长）：

| 架构 | 核心思想 | 代表模型 | 论文/开源地址 |
|------|---------|---------|-------------|
| 状态空间模型（SSM） | 用线性时不变系统建模序列，推理复杂度 O(1) | Mamba、Mamba-2（SSD） | https://github.com/state-spaces/mamba  |
| 线性注意力RNN | 可并行的循环网络，兼具RNN的O(1)推理和Transformer的训练并行性 | RWKV-4/5/6/7（Eagle/Finch/Goose） | https://github.com/BlinkDL/RWKV-LM  |
| 扩展LSTM | 给LSTM加指数门控和矩阵记忆 | xLSTM（Hochreiter本人2024年新作） | https://arxiv.org/abs/2405.04517  |
| 测试时训练 | 隐状态本身是一个小神经网络，用梯度更新替代固定递归 | TTT-Linear、TTT-MLP | https://arxiv.org/abs/2407.04620 |

Mamba 是最受关注的一个，由 CMU 的 Albert Gu 和普林斯顿的 Tri Dao 提出，采用选择性状态空间机制（S6），在信息密集的语言建模上首次逼近同规模 Transformer 的困惑度，且推理显存占用比同级 Transformer 低 20-40%。RWKV-7（代号 Goose）于 2025 年发布，引入动态状态演化和矩阵值状态，是当前开源 RNN 阵营的旗舰。

如果要复现或研究"非Transformer架构"，建议从 Mamba 入手（生态最成熟、论文最完整），路径是：

1. 读 Mamba 原始论文（arXiv:2312.00752）和 Mamba-2 的 SSD 形式化（arXiv:2405.21060）
2. 跑 state-spaces/mamba 的 GitHub 仓库（https://github.com/state-spaces/mamba ），在莎士比亚/维基数据上训练小模型

### 3.8.1 状态空间模型的连续—离散形式

SSM 的原始定义是一个连续时间线性时不变系统

$$h'(t)=A h(t)+B x(t),\qquad y(t)=C h(t),\qquad A\in\mathbb{R}^{m\times m}$$

用零阶保持（zero-order hold）以步长 $\Delta$ 离散化后得到可计算的版本

$$h_t=\bar A h_{t-1}+\bar B x_t,\quad \bar A=\exp(\Delta A),\quad \bar B=(\Delta A)^{-1}\big(\exp(\Delta A)-I\big)\Delta B,\quad y_t=C h_t$$

线性时不变性质使它可以写成一次卷积 $y=x\ast \bar K$（$\bar K_r=C\bar A^{r}\bar B$），从而像 CNN 一样并行训练；推理时又可以回到递推形式，每步只做一次 $m\times m$ 矩阵乘向量的 O(1) 操作。Mamba 的关键改动是"选择性"：让 $\Delta,B,C$ 成为输入的函数，于是原本的时不变系统变成了输入依赖（时变）线性系统，等价于一个非线性、非平稳的递归模型，也因此在 loss curve 上重获了接近注意力的拟合能力，代价是失去了卷积形式，需要改用并行扫描（parallel scan / associative scan）来实现训练期的高效并行。Mamba-2 提出的结构化状态空间对偶性（SSD）给出了它与某种结构化掩码注意力之间的等价形式，把 SSM 与注意力放进了同一个数学框架。

### 3.8.2 与统计学中的状态空间模型对接

对统计学者来说，这组公式并不陌生，去掉过程噪声后，它就是一个线性高斯状态空间模型（linear Gaussian SSM）的确定性版本，递推式与 Kalman 滤波的预测步完全一致，$h_t$ 扮演充分统计量的角色：

$$\text{Kalman 预测： } h_{t|t-1}=A h_{t-1|t-1},\qquad \text{SSM 递推： } h_t=\bar A h_{t-1}+\bar B x_t$$

差别在于统计建模习惯于把过程的随机性与观测噪声显式建模，并由此得到不确定性量化；而 SSM 在深度学习中把全部重心放在确定性的信息压缩上，隐状态的维度 $m$ 是唯一的压缩瓶颈。这带来一个值得研究的空白：如何把随机状态空间的不确定性传播引入选择性 SSM，以获得带置信度的语言状态；同时也解释了这类架构在"需要精确逐字召回"的任务上弱于注意力，有限维的充分统计量承载不了全部历史，而 Transformer 的 KV 缓存本质上保留了全部历史。

其他几条路线的形态分别是：RWKV 对线性注意力做递推重排，用一个加权的 $WKV$ 状态（衰减累积的 key-value 加权和，带有可学习的每通道衰减率）实现常数空间；xLSTM 给传统 LSTM 加上指数门控与矩阵值状态，把标量门替换成矩阵记忆以提高容量；TTT（test-time training）把隐状态本身设为一个小神经网络，在前向传播时用一次梯度下降更新它，即"序列模型就是在线学习"。三条路线的共同思想是：把长历史的记忆压缩为一个固定大小的状态。

客观地说，这些架构在 2024 至 2025 年间已在中小规模上得到充分验证，但在最大规模（数百 B 参数、万亿 token 级语料）上的公开证据仍显著少于 Transformer；生态方面（kernel、量化、长上下文工具链、分布式推理）也相对薄弱，这往往比困惑度差异更能决定落地选择。

## 3.9 混合架构

**当前的真实格局：混合架构成为前沿趋势**

工业界的最新动向不是"非Transformer替代Transformer"，而是"混合架构"，在同一个模型里同时部署注意力层和非注意力层：

- Jamba（AI21 Labs）：Mamba层 + Transformer层 + MoE 混合，52B 总参数、12B 激活
- Qwen3.8-Flash：采用 GDN（Gated DeltaNet，线性注意力类）+ QSA（Qwen 稀疏注意力）混合，把 51B 的 N-gram Embedding 卸载到主机内存
- Zamba（Zyphra）：Mamba + Transformer 混合的 7B 模型

这背后的统计学逻辑：注意力机制擅长精确检索（长距离依赖的精确捕获），循环/状态空间结构擅长压缩记忆（长上下文的常数开销），混合架构在两者间取折中，本质是精度与复杂度的权衡。

工程上，混合架构的设计变量是三件事：一是两类层的比例（常见做法是每隔若干层线性层插入一层全注意力，Jamba 采用的正是这种规律性交错）；二是 MoE 是否叠加在 FFN 上（决定激活参数与推理成本）；三是注意力层是否进一步稀疏化（QSA 一类方案），这会把注意力本身的二次复杂度降下来。混合之后的参数—算力关系不再由单一公式给出，评估时必须分别统计注意力层、线性层与 MoE 各自的贡献，否则很难判断某次改动的实际效果。

从估计的角度看，把不同归纳偏置的算子交替堆叠，等价于把一个近似无偏但方差较大的检索算子（注意力）与一个有偏但方差较低的压缩算子（状态空间）组合起来。这与非参数回归中"局部多项式 + 全局平滑"的组合、或与投影追踪中的加性模型思路相近：目标是让整体均方误差低于任何单一算子。这也为研究者指出了一个方向：给定总预算，最优层间配比是否可以通过某种可估计的准则（例如对不同深度算子做 perturbation-based 重要性打分）来确定，而不是靠试错。

<!--PART4-->

## 3.10 模型设计思想方法

### 3.10.1 从需求到约束清单

设计一个新架构的第一步不是选层数，而是把约束写成清单。以下六项几乎总是须自我回答：目标部署形态（云端多卡并发服务、单卡开发机、还是端侧设备）；推理侧的硬指标（延迟、单卡并发、每百万 token 的成本）；训练算力预算（可用 GPU 小时乘以单价）；上下文长度目标（决定注意力方案与训练策略）；是否需要多模态（决定输入侧是否存在视觉编码器与跨模态通路）；以及许可与生态策略（是否开源、是否允许商用，决定社区能否接手维护）。

### 3.10.2 缩放律给出的是约束而非目标

Chinchilla 缩放律（arXiv:2203.15556）给出的拟合形式是

$$L(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}},\qquad C\approx 6ND$$

其中 $N$ 为参数量、$D$ 为训练 token 数、$C$ 为浮点运算总量。原文给出的拟合值约为 $E=1.69$、$A=406.4$、$B=410.7$、$\alpha=0.34$、$\beta=0.28$，由此得到的最优关系是每 1 个参数配约 20 个训练 token。代入关系可以做一个简单的预算练习：若算力预算为 $C=1.2\times 10^{24}$ FLOPs，则 $N=\sqrt{C/120}=10^{11}$（一千亿参数）、$D=20N=2\times 10^{12}$（2T tokens）。

从统计学角度要强调三点。第一，上式本身是一个用几百次训练运行拟合出来的三元回归模型，指数项是待估参数，整套结论都处在被拟合区间之外的外推区（extrapolation region）里，因此外推两个数量级时的不确定性不应被忽略。第二，同一批数据可以被不同的参数化形式拟合（早期 Kaplan 等人的版本给出的 exponents 与上述不同），拟合优度相当但最优 $N/D$ 结论差别很大，这是典型的模型不确定性问题，需要用模型平均或敏感性分析而不是单点估计来表达。第三，缩放律是关于"同等清洗质量的数据"的结论，第 2 章的所有数据质量操作会通过 $D$ 的有效性间接改变结论。

### 3.10.3 参数分配：宽度、深度、头数与词表

在总参数预算确定后，剩余的配置大致遵循以下经验约束：

- 每头维度 $d_{head}$ 取 64 或 128 比较常见，$d_{head}$ 过小会导致单头表达能力受限（见 3.1.4 的秩限制），过大则失去多头并行的意义；
- 宽度 $d$ 与层数 $L$ 的组合存在一个被广泛观察到的现象："深而窄"与"浅而宽"在同等参数下质量接近，但深层优化更难、显存更省、推理串行度更高；模型层数增多时需要 Pre-LN 与适当的初始化缩放配合；
- $d_{ff}$ 按 4 倍（ReLU/GELU）或约 8/3 倍（SwiGLU）取，$d$ 与 $d_{ff}$ 通常取 256 或 128 的倍数以对齐硬件的矩阵单元；
- 词表规模需要为 tensor core 的倍数，并兼顾目标语言的 fertility（第 2 章）；
- MoE 的专家数与激活比决定"存储—计算"的解耦程度，常见激活比为 1/8 到 1/16，专家数过多则需要更强的均衡机制与更细的存储管理。

### 3.10.4 三个案例的逆向拆解

下表把三条主流路线的公开配置放在一起对照。数字取自各模型公开的模型卡与配置文件，版本更新可能变动，引用前请核对官方来源。

| 配置维度 | Llama 3.1 405B（密集） | DeepSeek-V3（MoE） | Qwen3-235B-A22B（MoE） |
|---------|----------------------|-------------------|----------------------|
| 层数 $L$ | 126 | 61 | 94 |
| 隐藏维 $d$ | 16384 | 7168 | 4096 |
| 注意力头 / KV 头 | 128 / 8（GQA） | 128（MLA，低秩 KV 潜在维 512） | 多头 GQA 配置 |
| FFN 中间维 | 53248（SwiGLU，约 3.25d） | 18432（密集层）；路由专家独立配置 | 路由专家独立配置 |
| MoE | 无 | 256 路由专家 + 1 共享专家，top-8，前 3 层保持密集 | 128 专家量级，top-8 稀疏激活 |
| 位置编码 | RoPE（基数 θ 取值扩大以支持长上下文） | RoPE + YaRN 类外推 | RoPE + YaRN 类外推，超长上下文版本 |
| 归一化 / 激活 | RMSNorm / SwiGLU | RMSNorm / SwiGLU | RMSNorm / SwiGLU |
| 词表 | 128256 | 129280 | 十五万量级（面向多语言） |
| 上下文 | 128K | 原生 128K，可扩展至更长 | 支持超长上下文扩展 |

由这张表可以读出三条设计取舍：密集架构需要靠更深的层（Llama 3.1 的 126 层）来堆参数，代价是推理串行步数多、延迟高；MoE 架构用专家数堆参数、用层数控制深度（DeepSeek-V3 仅 61 层），代价是显存与路由；第三种取向（Qwen3）同时提供密集小模型与 MoE 大模型，把架构选择交给部署场景，并用统一的算子方案（一致的 RoPE、归一化、激活）降低生态碎片化。

### 3.10.5 消融实验设计的统计要点

架构决策最终要靠实验验证，大模型实验的一个陷阱是"差异低于噪声"：

1. 对照必须逐项对齐：相同的训练 token 预算、相同的数据顺序、相同的初始化种子分布。只改一个变量（如头数）而不调整学习率与批次，得到的结论往往无法复现。
2. 必须报告运行间变异（run-to-run variation）。同一配置不同随机种子在下游基准上出现 1 至 2 个百分点的差异是常态，因此小于该阈值的效应不应被当作架构改进。可以考虑报告均值与标准差，而不是单点。
3. 当实验预算有限时，采用筛选设计（fractional factorial）或逐个因子 (one-factor-at-a-time) 加少量确认实验的组合，比完全随机试错更高效；把"效应量/噪声比"（信噪比）作为排序依据，而不是只看绝对分数。
4. 在新架构上线前做一次"阴性对照"：把新组件换成参数量相当的旧组件再跑一遍，确认提升不是来自参数量或训练步数的增加。

## 3.11 全栈技术路线图

AI 的做法是：用大规模文本/图像/音频语料训练一个参数量极大的神经网络，让它从数据中学出统计规律，进而能做"生成"和"推理"。当前大模型的公共底座是 Transformer 架构（也有一部分非 Transformer 架构的大模型），通过自注意力机制捕获长距离依赖，再经由"预训练 → 监督微调（SFT） → 人类反馈强化学习（RLHF/DPO/GRPO）"三阶段范式获得对话能力。以下从数据 → 分词 → 词向量 → 网络结构 → 训练方法 → 主流模型 → 全栈路线逐层给出落地路径。

**从数据到部署的完整链路**：

```
[数据层] Common Crawl → FineWeb清洗 → token语料
    ↓ datatrove
[分词层] tiktoken/SentencePiece → token序列
    ↓ https://github.com/openai/tiktoken
[预训练] Megatron-LM/DeepSpeed分布式训练
    ↓ 2048卡×3.7天/万亿tokens
[SFT] LLaMA-Factory微调
    ↓ 10万-150万指令对
[对齐] TRL库做RLHF/DPO/GRPO
    ↓ https://github.com/huggingface/trl
[推理] vLLM/Ollama部署
    ↓ FP8量化、KV缓存优化
[应用] LangChain/LlamaIndex构建RAG/Agent
```

全栈开发通常划分为六层：**数据层**（数据采集、清洗、标注）→ **预训练层**（分布式训练、混合精度）→ **微调层**（SFT、LoRA/QLoRA、RLHF/DPO）→ **推理层**（vLLM、TensorRT-LLM）→ **应用层**（RAG、Agent、Prompt Engineering）→ **评估层**（Benchmark、幻觉检测）。训练层与对齐层的具体细节见第 4 章。

全栈开源工具链：

| 层次 | 工具 | 地址 | 用途 |
|------|------|------|------|
| 数据处理 | datatrove | https://github.com/huggingface/datatrove | 大规模清洗 |
| 分词 | tiktoken | https://github.com/openai/tiktoken | BPE分词 |
| 预训练框架 | Megatron-LM | https://github.com/NVIDIA/Megatron-LM | NVIDIA工业级训练 |
| 分布式加速 | DeepSpeed | https://github.com/microsoft/DeepSpeed | ZeRO优化器 |
| 微调 | LLaMA-Factory | https://github.com/hiyouga/LLaMA-Factory | 零代码LoRA微调 |
| 强化学习 | TRL | https://github.com/huggingface/trl | RLHF/DPO/GRPO |
| 推理 | vLLM | https://github.com/vllm-project/vllm | 高吞吐推理 |
| 本地运行 | Ollama | https://github.com/ollama/ollama | 一行命令跑模型 |
| 应用框架 | LangChain | https://github.com/langchain-ai/langchain | Agent/RAG开发 |

### 3.11.1 三条主流技术路线的生态对比

| 技术维度 | DeepSeek | Qwen | Llama |
|---------|----------|------|-------|
| 核心架构 | MoE（数百 B 总参数/数十 B 激活） | MoE + Dense 双路线 | Dense 为主 |
| 推理能力 | R1 系列，强化学习驱动 | 推理型产品线（QwQ 一类） | 基础推理能力，依赖上层方法 |
| 多语言 | 中英并重 | 中文优势显著 | 英文优先 |
| 开源协议 | MIT | Apache 2.0 | Llama License |

三者差异并不只在架构，更在于发布策略：DeepSeek 与 Qwen 采用宽松许可证并持续开放权重，有利于研究复现；Llama 系列带有商业使用限制条款，落地前需要核对许可条款的具体内容（见第 2 章关于许可兼容性的讨论）。

### 3.11.2 学习路径资源

这批项目是入门的第一站，覆盖从零基础到完整实践的完整路线。

| 项目名称 | 简介 | GitHub地址 |
|---------|------|-----------|
| llm-course | mlabonne出品，LLM学习三段式路线（基础→科学家→工程师），配大量Colab实战notebook | github.com/mlabonne/llm-course  |
| llm-cookbook | Datawhale出品，大模型系列课程的中文翻译与复现，覆盖Prompt→RAG→微调全流程 | github.com/datawhalechina/llm-cookbook  |
| happy-llm | Datawhale从零手写LLaMA2的教程，含215M参数小模型预训练+LoRA微调+RAG全链路 | github.com/datawhalechina/happy-llm  |
| llm-universe | Datawhale"构建个人知识助手"项目式教程，环境配置详尽 | github.com/datawhalechina/llm-universe |
| self-llm | Datawhale开源大模型"食用指南"，各主流开源模型的本地部署与微调手册 | github.com/datawhalechina/self-llm |
| leeml-notes | 李宏毅机器学习课程中文笔记，Datawhale维护 | github.com/datawhalechina/leeml-notes |
| llm-action | liguodongiot出品，大模型工程实战：LoRA/P-Tuning微调、分布式训练、LangChain集成 | github.com/liguodongiot/llm-action  |
| minimind | jingyaogong出品，26M参数超小LLM从零训练，小显存可跑，用于快速理解训练全流程 | github.com/jingyaogong/minimind  |
| start-llms | louisfb01整理，从零构建LLM的学习资源汇总 | github.com/louisfb01/start-llms  |
| LLM-Open-University | 结构化LLM学习路线图：基础→训练→应用→前沿 | github.com/youssefHosni/LLM-Open-University-From-Begineer-to-Advanced  |
| dive-into-llms | 上交大自然语言处理课程讲义延伸的动手实践教程 | github.com/Lordog/dive-into-llms  |
| d2l-zh | 《动手学深度学习》中文版，李沐等著 | github.com/d2l-ai/d2l-zh |
| generative-ai-for-beginners | 微软出品生成式AI入门课程 | github.com/microsoft/generative-ai-for-beginners  |
| LLM101n | Karpathy规划的"从零构建Storyteller AI模型"课程 | github.com/karpathy/LLM101n |
| neural-networks-zero-to-hero | Karpathy从反向传播到GPT的零基础视频系列 | github.com/karpathy/nn-zero-to-hero |
| annotated_deep_learning_paper_implementations | 60+篇深度学习论文的带注释实现 | github.com/labmlai/annotated_deep_learning_paper_implementations  |
| Awesome-LLM | Hannibal046维护，LLM资源索引（论文/模型/框架/排行榜） | github.com/Hannibal046/Awesome-LLM  |
| awesome-deep-learning | 深度学习教程、项目与社区精选清单 | github.com/ChristosChristofidis/awesome-deep-learning  |
| awesome-deep-learning-papers | 深度学习必读论文清单 | github.com/terryum/awesome-deep-learning-papers  |
| awesome-deeplearning-resources | 按时间排序的深度学习与深度强化学习论文列表 | github.com/endymecy/awesome-deeplearning-resources  |
| Awesome-Deep-Learning-Papers-for-Search-Recommendation-Advertising | 工业/搜索/推荐/广告场景深度学习论文集 | github.com/guyulongcs/Awesome-Deep-Learning-Papers-for-Search-Recommendation-Advertising  |
| awesome-artificial-intelligence | AI领域综合资源清单 | github.com/owainlewis/awesome-artificial-intelligence  |
| AI-For-Beginners | 微软AI基础课程 | github.com/microsoft/AI-For-Beginners |

### 3.11.3 与后续章节的接口

本章讨论的是单一文本模态下的函数逼近器，三个延伸方向的细节分别由后续章节承担：

- **多模态**：多模态大模型通过统一的表征空间将文本、图像、音频等模态映射到同一语义空间，核心技术包括视觉编码器（如 ViT，arXiv:2010.11929）、跨模态注意力机制、模态对齐训练（如 CLIP，arXiv:2103.00020）。当前趋势是从"拼接式多模态"走向"原生多模态"，即在预训练阶段即实现多模态统一建模。详见第 5 章。
- **智能体**：LLM-based Agent 的核心架构包含四个模块，规划（Planning）、记忆（Memory）、工具使用（Tool Use）和执行（Action）。当前研究将 LLM-based Agent 视为一个完整生命周期系统，涵盖感知、推理、行动、反思四个阶段，图增强的 Agent（GLAs）成为结构感知的重要扩展方向。工具调用、Skill 与自动化工作流详见第 7 章与第 8 章。
- **形式化验证**：在节点序列之外的另一条保证路径是用交互式定理证明器对关键结论做机器检查（Lean 4 / Mathlib 体系）。架构本身只是第一个环节，第 10 至 12 章会专门讨论如何把数学与统计命题形式化并用证明检查器验证。

## 本章小结

- 自注意力是一次行内的加权平均，输出矩阵满足 $O=AV$ 且 $A$ 为行随机矩阵，模型在自己的 value 凸包内插值而不做外推。
- 缩放因子 $\sqrt{d_k}$ 来自方差论证：logit 的标准差为 $\sqrt{d_k}\sigma^2$，不缩放会导致 softmax 退化为近似 one-hot、梯度消失；温度越软越接近全局平均，可用参与率 $\mathrm{ESS}=(\sum_j\alpha_{ij}^2)^{-1}$ 量化。
- 自注意力等价于带宽可学习、度量非对称、先线性变换再平均的 Nadaraya–Watson 核回归；它对位置置换等变，因此必须外挂位置信息。
- 多头注意力的参数量为 $4d^2$，与头数无关，头数调节的是表达空间切分与并行效率；大量头可被剪枝而无显著损失。
- FFN 占每层约三分之二参数，形式上是 $d_{ff}$ 个自适应基函数的线性组合（投影寻踪），也可解释为键值记忆；SwiGLU 用三个矩阵并在 $d_{ff}\approx\frac83 d$ 时与 $8d^2$ 参数量持平。
- LayerNorm 沿特征维度标准化（逐样本），BatchNorm 沿 batch 维度标准化（逐通道）；Pre-LN 为深层 LLM 的默认选择，RMSNorm 去均值只留二阶矩。
- RoPE 以分块旋转矩阵注入相对位置，满足 $R_m^\top R_n=R_{n-m}$；ALiBi 以 $-m|i-j|$ 的线性偏置实现同样目的且零参数；两者的长上下文都需要 PI/NTK/YaRN 类外推。
- 推理瓶颈是 KV 缓存，每 token 缓存量 $=2Ln_{kv}d_{head}b$；MQA/GQA 共享 KV 头，MLA 用低秩潜在向量压缩 KV 并在推理期吸收上投影矩阵。
- MoE 以 $g(x)=\mathrm{TopK}(\mathrm{softmax}(W_rx))$ 实现参数与算力的解耦，负载均衡辅助损失 $\mathcal{L}_{aux}=\alpha N\sum_i f_iP_i$ 的作用等价于混合模型中防止成分退化的先验约束。
- 非 Transformer 架构（SSM/Mamba、线性注意力 RNN、xLSTM、TTT）本质是让隐状态成为历史的固定维充分统计量，其数学形式与无噪声的线性高斯状态空间模型一致。
- 混合架构同时在精度与内存开销间取折中，是目前前沿模型的常见形态；架构实验结论必须报告运行间变异，否则容易把噪声当作提升。

## 延伸资源

| 资源类型 | 名称 | 链接 | 用途 |
|---------|------|------|------|
| 论文 | Attention is All You Need | arXiv:1706.03762 | Transformer原始论文 |
| 论文 | BERT | arXiv:1810.04805 | 双向编码器与掩码语言模型 |
| 论文 | RoFormer (RoPE) | arXiv:2104.09864 | 旋转位置编码 |
| 论文 | ALiBi (Train Short, Test Long) | arXiv:2108.12409 | 线性偏置位置编码 |
| 论文 | Switch Transformer | arXiv:2101.03961 | 稀疏 MoE 与 top-1 路由 |
| 论文 | Outrageously Large Neural Networks (MoE 层) | arXiv:1701.06538 | 稀疏门控混合专家层的原始工作 |
| 论文 | RMSNorm | arXiv:1910.07467 | 均方根层归一化 |
| 论文 | Mamba | arXiv:2312.00752 | 选择性状态空间模型 |
| 论文 | Mamba-2 (SSD) | arXiv:2405.21060 | 状态空间对偶性与统一视角 |
| 论文 | xLSTM | arXiv:2405.04517 | 带指数门控的扩展 LSTM |
| 论文 | TTT (Test-Time Training) | arXiv:2407.04620 | 以在线学习做序列建模 |
| 论文 | ST-MoE | arXiv:2202.08906 | 路由 z-loss 与 MoE 稳定训练设计 |
| 论文 | FlashAttention | arXiv:2205.14135 | IO 感知的精确注意力 kernel |
| 论文 | FlashAttention-2 | arXiv:2307.08691 | 改进的并行与分块策略 |
| 论文 | PagedAttention / vLLM | arXiv:2309.06180 | 分页 KV 缓存与高吞吐服务 |
| 论文 | Performer (FAVOR+) | arXiv:2009.14794 | softmax 核的随机特征近似 |
| 论文 | Chinchilla 缩放律 | arXiv:2203.15556 | 参数—数据最优配比 |
| 论文 | A Survey of Large Language Models | arXiv:2303.18223 | LLM 全景综述（架构、训练、对齐、评估） |
| 论文 | ViT | arXiv:2010.11929 | 视觉 Transformer（详见第 5 章） |
| 论文 | CLIP | arXiv:2103.00020 | 图文对比预训练（详见第 5 章） |
| 论文 | DeepSeek-R1技术报告 | arXiv:2501.12948 | 推理模型的训练方法 |
| 开源 | RoPE 官方实现 roformer | https://github.com/ZhuiyiTechnology/roformer | RoPE 参考实现 |
| 开源 | ALiBi 官方实现 | https://github.com/ofirpress/attention_with_linear_biases | 线性偏置注意力参考实现 |
| 开源 | Mamba | https://github.com/state-spaces/mamba | SSM 模型与 CUDA kernel |
| 开源 | RWKV-LM | https://github.com/BlinkDL/RWKV-LM | 线性注意力 RNN 阵营 |
| 开源 | nanoGPT | https://github.com/karpathy/nanoGPT | 极简GPT实现，学习原理 |
| 开源 | minGPT | https://github.com/karpathy/minGPT | 极简 GPT 实现（nanoGPT 的前身），适合逐行阅读 |
| 教程 | The Illustrated Transformer | https://jalammar.github.io/illustrated-transformer | 图解自注意力与 Transformer 数据流 |
| 教程 | The Illustrated GPT-2 | https://jalammar.github.io/illustrated-gpt2 | 图解解码器结构与自回归生成 |
| 开源 | LLaMA-Factory | https://github.com/hiyouga/LLaMA-Factory | 集成式微调平台 |
| 开源 | Ollama | https://github.com/ollama/ollama | 本地运行开源模型 |
| 开源 | Megatron-LM | https://github.com/NVIDIA/Megatron-LM | 工业级分布式预训练框架 |
| 开源 | DeepSpeed | https://github.com/microsoft/DeepSpeed | ZeRO 系列优化器 |
| 开源 | TRL | https://github.com/huggingface/trl | RLHF/DPO/GRPO 训练库 |
| 开源 | vLLM | https://github.com/vllm-project/vllm | 高吞吐推理引擎 |
| 数据 | FineWeb | https://huggingface.co/datasets/HuggingFaceFW/fineweb | 大规模开源预训练语料 |



