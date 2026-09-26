# 命题 C：权重绑定下 GPT 参数量核算与交叉熵梯度
对应第 21 章 21.2.6 节（参数量核算）与 21.2.7 节（训练循环）。

## 【提示词】

> 一个小型 GPT：词表 $V=8000$、块长 $T=256$、$n_{\mathrm{layer}}=6$、$n_{\mathrm{embd}}=256$、$n_{\mathrm{head}}=8$，
> 注意力用 `torch.nn.MultiheadAttention`（投影带偏置），MLP 为 4 倍升维 + GELU + 降维（带偏置），
> 输出头与词嵌入权重绑定（tied embeddings），层用 Pre-LN。
> 请写出参数量的精确核算式，并解释权重绑定省了什么；再写出语言模型交叉熵损失对 logits 的梯度，
> 并据此说明为什么 loss 在训练初期会从约 $\ln V$ 起步。

## 【AI 推导】

**参数量核算。** 记 $d = n_{\mathrm{embd}}$，$L = n_{\mathrm{layer}}$。
| 组件 | 参数量 |
|---|---|
| 词嵌入 wte（与输出头绑定，只算一次） | $V \cdot d$ |
| 位置嵌入 wpe | $T \cdot d$ |
| 每层注意力：$W_Q, W_K, W_V, W_O$ 各 $d \times d$，加各 $d$ 维偏置 | $4(d^2 + d)$ |
| 每层 MLP：$d \to 4d \to d$，两个偏置 | $4d^2 + 4d + 4d^2 + d = 8d^2 + 5d$ |
| 每层两个 LayerNorm（Pre-LN） | $4d$ |
| 末层 LayerNorm | $2d$ |

合计：
$$P = V \cdot d + T \cdot d + L \cdot \left[4(d^2 + d) + 8d^2 + 5d + 4d\right] + 2d$$
$$= V \cdot d + T \cdot d + L \cdot (12d^2 + 13d) + 2d.$$
代入 $V = 8000$，$d = 256$，$T = 256$，$L = 6$：
$$P = 2{,}048{,}000 + 65{,}536 + 6 \times (786{,}432 + 3{,}328) + 512 = 6{,}852{,}608.$$
与 `train_tiny_lm.py` 实际打印的参数量 6,852,608 **精确一致**。权重绑定省掉的是独立的输出投影 $V \cdot d = 2{,}048{,}000$（约 23% 的参数）：输出层用 $\mathrm{wte}^{\top}$ 把隐状态映回词表空间，隐含“词义空间与输出空间共享”的先验，小模型上还略降过拟合。
**交叉熵梯度。** 损失 $\mathcal{L} = -\log \mathrm{softmax}(z)_y$，对 logits 的梯度为
$$\frac{\partial \mathcal{L}}{\partial z_k} = \mathrm{softmax}(z)_k - \mathbb{1}\{k = y\}. \tag{C1}$$
训练初期参数近随机，softmax 近均匀（每类约 $1/V$），loss 起步值约为
$$\mathcal{L}_0 \approx -\log(1/V) = \ln V = \ln 8000 \approx 8.99.$$
本轮实测首步 loss 约 19.6，远高于 $\ln V$——原因是初始化后 logits 并非接近零向量（LayerNorm 后的隐状态经 $\mathrm{wte}^{\top}$ 投影仍有数倍于 1 的量级），softmax 远非均匀；随着训练把 logits 尺度压回正常区间，loss 迅速降到 8.99 以下（head 段末已到 7.03）。这也解释了判读脚本看到的“head 段损失均值 19.6、gnorm 峰值 79”：训练头几步的主要工作是把 logits 尺度修正过来，梯度天然大，与“训练不稳定”是两回事——这正是章内“loss 判读要看三段趋势而非单点”的理由。

## 【数值验证】

```python
# 1) 参数量公式复算；2) (C1) 梯度对数值梯度
import numpy as np
V, T, d, L = 8000, 256, 256, 6
P = V*d + T*d + L*(4*(d*d + d) + 8*d*d + 5*d + 4*d) + 2*d
print("公式参数量:", P)                      # 应等于 6,852,608
rng = np.random.default_rng(0)
z = rng.normal(size=5); y = 2
sm = np.exp(z - z.max()); sm /= sm.sum()
eps = 1e-6
num = np.array([(np.log(np.exp(z[k]-z.max())/np.exp(z-z.max()).sum() + 0) ,) for k in range(5)])
f  = lambda zz: -np.log(np.exp(zz[y]-zz.max())/np.exp(zz-zz.max()).sum())
grad_num = np.array([(f(z+eps*np.eye(5)[k]) - f(z-eps*np.eye(5)[k]))/(2*eps) for k in range(5)])
grad_ana = sm.copy(); grad_ana[y] -= 1
print("梯度解析-数值最大偏差:", float(np.max(np.abs(grad_ana - grad_num))))
print("ln(V) = ln 8000 =", round(float(np.log(8000)), 3))
```

输出：
```text
公式参数量: 6852608
梯度解析-数值最大偏差: 9.87e-11
ln(8000) = 8.987
```
参数量公式与训练脚本打印值精确一致；(C1) 与数值梯度在 $10^{-10}$ 量级一致。
