# -*- coding: utf-8 -*-
"""生成项目封面图 assets/cover.png（1600x800，深色学术风）。"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = r"I:\mydesk\ai-for-statistics\assets"
os.makedirs(OUT, exist_ok=True)

FP_CN = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
FP_EN = FontProperties(family="DejaVu Sans", weight="bold")

# 配色
BG_TOP = "#0A1729"
BG_BOT = "#16344F"
CYAN = "#5EEAD4"
BLUE = "#60A5FA"
GOLD = "#F5C86B"
WHITE = "#FFFFFF"
GREY = "#9FB4CC"

fig = plt.figure(figsize=(16, 8), dpi=100)
fig.patch.set_facecolor(BG_TOP)

# ---------- 背景渐变 ----------
ax = fig.add_axes([0, 0, 1, 1])
ax.set_axis_off()
grad = np.linspace(0, 1, 512).reshape(-1, 1) * np.ones((1, 512))
cmap_bg = LinearSegmentedColormap.from_list("bg", [BG_TOP, BG_BOT])
ax.imshow(grad.T, extent=[0, 1, 0, 1], aspect="auto", cmap=cmap_bg, origin="lower", zorder=0)

# 背景网格
for i in np.linspace(0.05, 0.98, 26):
    ax.plot([i, i], [0.02, 0.98], color="#FFFFFF", alpha=0.030, lw=0.6, zorder=1)
for j in np.linspace(0.05, 0.98, 13):
    ax.plot([0.02, 0.98], [j, j], color="#FFFFFF", alpha=0.030, lw=0.6, zorder=1)

# ---------- 左上角标签条 ----------
ax.add_patch(FancyBboxPatch(
    (0.055, 0.845), 0.208, 0.052,
    boxstyle="round,pad=0.004,rounding_size=0.012",
    transform=ax.transAxes, facecolor=CYAN, alpha=0.14,
    edgecolor=CYAN, linewidth=1.1, zorder=3))
ax.text(0.068, 0.871, "OPEN SOURCE HANDBOOK", transform=ax.transAxes,
        fontsize=11, color=CYAN, fontproperties=FP_EN, va="center", zorder=4)
ax.text(0.272, 0.871, "v1.1  ·  21 chapters  ·  CC BY-NC-SA 4.0", transform=ax.transAxes,
        fontsize=11.5, color=GREY, va="center", zorder=4)

# ---------- 主标题 ----------
ax.text(0.055, 0.700, "AI 赋能统计研究", transform=ax.transAxes,
        fontsize=52, color=WHITE, fontproperties=FP_CN,
        va="center", zorder=5, fontweight="bold")

# 标题下的强调短线
ax.add_patch(Rectangle((0.057, 0.638), 0.072, 0.0075,
                       transform=ax.transAxes, facecolor=CYAN, zorder=5))

ax.text(0.055, 0.575, "原理 · 工作流 · 应用实践", transform=ax.transAxes,
        fontsize=25, color=CYAN, fontproperties=FP_CN, va="center", zorder=5)

ax.text(0.055, 0.505, "AI for Statistical Research", transform=ax.transAxes,
        fontsize=17, color=WHITE, fontproperties=FP_EN, va="center", zorder=5)
ax.text(0.055, 0.452, "Principles, Workflows, and Applications", transform=ax.transAxes,
        fontsize=17, color=GREY, fontproperties=FP_EN, va="center", zorder=5)

# ---------- 定位语 ----------
ax.plot([0.057, 0.575], [0.395, 0.395], color="#FFFFFF", alpha=0.18, lw=1.0,
        transform=ax.transAxes, zorder=5)
ax.text(0.055, 0.340, "面向具有统计背景的AI初学者", transform=ax.transAxes,
        fontsize=19, color=GOLD, fontproperties=FP_CN, va="center", zorder=5)

# ---------- 三条主线 chips ----------
chips = ["什么是 AI", "怎么用 AI", "怎么研究 AI"]
cx = [0.057, 0.196, 0.352]
w = [0.115, 0.115, 0.138]
for x, ww, t in zip(cx, w, chips):
    ax.add_patch(FancyBboxPatch(
        (x, 0.255), ww, 0.058,
        boxstyle="round,pad=0.003,rounding_size=0.014",
        transform=ax.transAxes, facecolor="#FFFFFF", alpha=0.06,
        edgecolor="none", linewidth=0.0, zorder=4))
    ax.add_patch(FancyBboxPatch(
        (x, 0.255), ww, 0.058,
        boxstyle="round,pad=0.003,rounding_size=0.014",
        transform=ax.transAxes, facecolor="none",
        edgecolor=BLUE, linewidth=1.0, alpha=0.55, zorder=5))
    ax.text(x + ww / 2, 0.284, t, transform=ax.transAxes,
            fontsize=14.5, color=WHITE, fontproperties=FP_CN,
            ha="center", va="center", zorder=6)

# ---------- 底部作者 ----------
ax.plot([0.057, 0.575], [0.185, 0.185], color="#FFFFFF", alpha=0.12, lw=1.0,
        transform=ax.transAxes, zorder=5)
ax.text(0.055, 0.133, "chenyiadam", transform=ax.transAxes,
        fontsize=16, color=WHITE, fontproperties=FP_EN, va="center", zorder=5)
ax.text(0.178, 0.133, "云南大学 Yunnan University", transform=ax.transAxes,
        fontsize=12.5, color=GREY, fontproperties=FP_CN, va="center", zorder=5)
ax.text(0.415, 0.133, "zyk111bj", transform=ax.transAxes,
        fontsize=16, color=WHITE, fontproperties=FP_EN, va="center", zorder=5)
ax.text(0.512, 0.133, "西安交通大学 Xi'an Jiaotong University", transform=ax.transAxes,
        fontsize=12.5, color=GREY, fontproperties=FP_CN, va="center", zorder=5)

# ================= 右侧装饰区 =================
# --- 注意力热力图 ---
rng = np.random.default_rng(7)
n = 16
W = rng.normal(0, 1, (n, n))
W = W + np.eye(n) * 2.2                       # 对角自匹配
W = np.tril(W) + np.tril(W, -1).T * 0.35      # 因果结构
A = np.exp(W - W.max(axis=1, keepdims=True))
A = A / A.sum(axis=1, keepdims=True)

axh = fig.add_axes([0.628, 0.545, 0.315, 0.315])
axh.imshow(A, cmap=LinearSegmentedColormap.from_list(
    "atn", ["#0B2038", "#1D5C7A", "#3FA7B8", "#8FE3D0", "#EAFBF6"]),
    interpolation="nearest", aspect="equal")
axh.set_xticks([]); axh.set_yticks([])
for s in axh.spines.values():
    s.set_color("#5EEAD4"); s.set_linewidth(1.0); s.set_alpha(0.35)
ax.text(0.628, 0.885, "self-attention  ·  softmax weights", transform=ax.transAxes,
        fontsize=10.5, color="#7FD8CB", fontproperties=FP_EN, va="center", zorder=6)

# --- 缩放律曲线 + 外推置信带 ---
axc = fig.add_axes([0.628, 0.145, 0.315, 0.315])
x = np.linspace(0, 4.2, 300)
y = -0.062 * x + 1.15
band = 0.045 + 0.052 * np.maximum(x - 1.6, 0) ** 1.35
axc.fill_between(x, y - band, y + band, color=BLUE, alpha=0.16, lw=0)
axc.plot(x[:185], y[:185], color=CYAN, lw=2.4)
axc.plot(x[184:], y[184:], color=CYAN, lw=2.4, ls="--", alpha=0.55)
axc.scatter([1.6], [y[114]], s=34, color=GOLD, zorder=5)
axc.set_xlim(0, 4.2); axc.set_ylim(0.55, 1.30)
axc.set_xticks([]); axc.set_yticks([])
axc.set_facecolor("none")
for s in axc.spines.values():
    s.set_color("#5EEAD4"); s.set_linewidth(1.0); s.set_alpha(0.35)
axc.tick_params(colors=GREY)
ax.text(0.628, 0.475, "scaling laws  ·  extrapolation uncertainty", transform=ax.transAxes,
        fontsize=10.5, color="#7FD8CB", fontproperties=FP_EN, va="center", zorder=6)
ax.text(0.948, 0.487, "$L(N)\\,=\\,aN^{-\\alpha}$", transform=ax.transAxes,
        fontsize=12, color=GREY, ha="right", va="center", zorder=6)

# --- 公式装饰 ---
ax.text(0.628, 0.058,
        r"$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(QK^{\top}/\sqrt{d_k}\right)V$",
        transform=ax.transAxes, fontsize=12.5, color="#6E88A6", va="center", zorder=6)
ax.text(0.628, 0.022, "verify,  not  trust", transform=ax.transAxes,
        fontsize=11, color="#4B6188", fontproperties=FP_EN, va="center", zorder=6)

fig.savefig(os.path.join(OUT, "cover.png"), dpi=100, facecolor=BG_TOP,
            bbox_inches=None, pad_inches=0)
print("saved:", os.path.join(OUT, "cover.png"))
