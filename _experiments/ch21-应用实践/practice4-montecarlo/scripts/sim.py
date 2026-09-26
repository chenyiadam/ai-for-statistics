#!/usr/bin/env python3
"""异方差下 t 检验的水平失真：HC0–HC3 与 wild bootstrap 的蒙特卡洛比较。

复现自《AI 赋能统计研究》第 21 章 21.5.4 节代码块。
用法：
    python sim.py --n 50 --design B --lam 3.0 --R 2000 --Bboot 199 \
                  --seed 20260101 --out runs/n50_B_lam3.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
from scipy import stats

J = 1  # 待检验的系数下标：0 为截距，1 为 x1


def make_X(rng, n: int, design: str) -> np.ndarray:
    one = np.ones(n)
    if design == "A":                       # 标准正态设计，杠杆值均匀
        x1, x2 = rng.normal(size=n), rng.normal(size=n)
    elif design == "B":                     # 高杠杆设计：x1 对数正态，右尾重
        x1 = np.exp(rng.normal(size=n)) - np.exp(0.5)
        x2 = rng.normal(size=n)
    elif design == "C":                     # 含一个稀疏二元变量
        x1 = (rng.random(n) < 0.1).astype(float)
        x2 = rng.normal(size=n)
    else:
        raise ValueError(design)
    return np.column_stack([one, x1, x2])


def draw_errors(rng, n: int, dist: str) -> np.ndarray:
    if dist == "normal":
        z = rng.normal(size=n)
    elif dist == "t5":
        z = rng.standard_t(5, size=n) / np.sqrt(5 / 3)   # 标准化到单位方差
    elif dist == "lognormal":
        z = (np.exp(rng.normal(size=n)) - np.exp(0.5)) / np.sqrt(np.e ** 2 - np.e)
    else:
        raise ValueError(dist)
    return z


def fit(X: np.ndarray, Y: np.ndarray):
    """Y 可以是 n×m 矩阵：单列是常规拟合，多列用于一次算完所有 bootstrap 复制。"""
    XtX_inv = np.linalg.pinv(X.T @ X)
    B = XtX_inv @ (X.T @ Y)
    return B, Y - X @ B, XtX_inv


def hc_se(X: np.ndarray, E: np.ndarray, XtX_inv: np.ndarray, kind: str, j: int = J) -> np.ndarray:
    """返回第 j 个系数在多种 HC 形式下的标准误；E 为 n×m，返回长度 m 的向量。"""
    n, p = X.shape
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)          # 杠杆值 h_ii
    u = E ** 2
    if kind == "OLS":
        s2 = u.sum(axis=0) / (n - p)
        V = s2 * XtX_inv[j, j]
        return np.sqrt(np.repeat(V, E.shape[1]))
    if kind == "HC0":
        w = np.ones(n)
    elif kind == "HC1":
        w = np.full(n, n / (n - p))
    elif kind == "HC2":
        w = 1.0 / (1.0 - h)
    elif kind == "HC3":
        w = 1.0 / (1.0 - h) ** 2
    else:
        raise ValueError(kind)
    W = u * w[:, None]                                    # n×m
    M = np.einsum("ia,im,ib->mab", X, W, X)               # m×p×p 肉矩阵
    V = np.einsum("ac,mcd,db->mab", XtX_inv, M, XtX_inv)  # m×p×p 三明治
    return np.sqrt(V[:, j, j])


def wild_bootstrap_p(X, y, rng, B: int = 199, kind: str = "HC3", j: int = J) -> float:
    """Rademacher wild bootstrap：在原假设下重抽样，统计量与待精化的检验同口径。"""
    n, p = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    b, e, _ = fit(X, y[:, None])
    keep = [k for k in range(p) if k != j]
    br, er, _ = fit(X[:, keep], y[:, None])               # 受约束拟合
    mu = X[:, keep] @ br[:, 0]                            # 原假设下的均值结构
    t_obs = b[j, 0] / hc_se(X, e, XtX_inv, kind, j)[0]
    Wg = rng.integers(0, 2, size=(n, B)) * 2 - 1          # Rademacher 权重
    Bs, Es, _ = fit(X, mu[:, None] + er * Wg)
    t_star = Bs[j, :] / hc_se(X, Es, XtX_inv, kind, j)
    return (1 + np.sum(np.abs(t_star) >= abs(t_obs))) / (B + 1)


def one_rep(rng, n, design, lam, effect, dist, Bboot, kinds):
    X = make_X(rng, n, design)
    beta = np.array([1.0, effect, 0.5])
    sigma = np.sqrt(1.0 + lam * X[:, 1] ** 2)             # 异方差与 x1 绑定
    y = X @ beta + sigma * draw_errors(rng, n, dist)
    XtX_inv = np.linalg.pinv(X.T @ X)
    b, e, _ = fit(X, y[:, None])
    df = n - X.shape[1]
    crit = stats.t.ppf(0.975, df)                          # 双侧 5%，t_{n-p} 临界值
    out = {"beta_hat": float(b[J, 0])}
    for k in kinds:
        se = hc_se(X, e, XtX_inv, k, J)[0]
        out[f"se_{k}"] = float(se)
        out[f"rej_{k}"] = int(abs(b[J, 0] / se) > crit)
        out[f"cov_{k}"] = int(abs(b[J, 0] - beta[J]) <= crit * se)
    out["rej_WB"] = int(wild_bootstrap_p(X, y, rng, Bboot) < 0.05)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--design", default="A", choices=["A", "B", "C"])
    ap.add_argument("--lam", type=float, default=1.0, help="异方差强度")
    ap.add_argument("--dist", default="normal", choices=["normal", "t5", "lognormal"])
    ap.add_argument("--effect", type=float, default=0.0, help="beta_1 真值；0 用于测水平")
    ap.add_argument("--R", type=int, default=2000, help="蒙特卡洛重复次数")
    ap.add_argument("--Bboot", type=int, default=199)
    ap.add_argument("--seed", type=int, default=20260101)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    kinds = ["OLS", "HC0", "HC1", "HC2", "HC3"]
    children = np.random.SeedSequence(args.seed).spawn(args.R)   # 与并行顺序无关
    rows, bh, ses = [], [], {k: [] for k in kinds}
    t0 = time.time()
    for r in range(args.R):
        o = one_rep(np.random.default_rng(children[r]), args.n, args.design,
                    args.lam, args.effect, args.dist, args.Bboot, kinds)
        rows.append({k: v for k, v in o.items() if not k.startswith(("se_", "beta_"))})
        bh.append(o["beta_hat"])
        for k in kinds:
            ses[k].append(o[f"se_{k}"])
    bh = np.asarray(bh)

    def rate(key: str) -> dict:
        p = float(np.mean([r[key] for r in rows]))
        mcse = float(np.sqrt(p * (1 - p) / args.R))
        return {"rate": round(p, 4), "mcse": round(mcse, 4),
                "lo": round(p - 1.96 * mcse, 4), "hi": round(p + 1.96 * mcse, 4)}

    res = {"config": vars(args), "R": args.R,
           "reject": {k: rate(f"rej_{k}") for k in kinds} | {"WB": rate("rej_WB")},
           "coverage": {k: rate(f"cov_{k}") for k in kinds},
           "se_ratio": {k: round(float(np.median(ses[k]) / bh.std(ddof=1)), 4) for k in kinds},
           "meta": {"python": platform.python_version(), "numpy": np.__version__,
                    "seed": args.seed, "wall_sec": round(time.time() - t0, 1),
                    "config_sha": hashlib.sha256(json.dumps(vars(args), sort_keys=True)
                                                 .encode()).hexdigest()[:16]}}
    s = json.dumps(res, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(s, encoding="utf-8")
    print(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
