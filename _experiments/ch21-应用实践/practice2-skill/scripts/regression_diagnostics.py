#!/usr/bin/env python3
"""回归诊断与稳健性检查。

用法：
    python regression_diagnostics.py --data data.csv --formula "y ~ x1 + x2" \
        [--cluster g] [--alpha 0.05] [--out diag.json]

输出：JSON；默认打印到 stdout，指定 --out 时同时落盘。退出码 0 正常，2 为输入错误。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

VERSION = "0.3.0"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def se_table(res, data: pd.DataFrame, cluster: str | None, terms: list[str]) -> dict:
    """并列输出多种标准误，用于判断结论对方差估计方式的敏感度。"""
    kinds = ["nonrobust", "HC0", "HC1", "HC2", "HC3"]
    out = {}
    for k in kinds:
        if k == "nonrobust":
            # 常规标准误直接取 fit() 默认结果：nonrobust 不是 get_robustcov_results
            # 的合法 cov_type，传进去会直接报错
            se, pv = res.bse, res.pvalues
        else:
            r = res.get_robustcov_results(cov_type=k, use_t=True)
            se, pv = r.bse, r.pvalues
        out[k] = {"se": dict(zip(terms, np.asarray(se).ravel().tolist())),
                  "p": dict(zip(terms, np.asarray(pv).ravel().tolist()))}
    if cluster is not None:
        r = res.get_robustcov_results(
            cov_type="cluster", use_t=True,
            groups=data[cluster].astype("category").cat.codes.to_numpy())
        out["cluster"] = {"se": dict(zip(terms, np.asarray(r.bse).ravel().tolist())),
                          "p": dict(zip(terms, np.asarray(r.pvalues).ravel().tolist())),
                          "n_groups": int(data[cluster].nunique())}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--formula", required=True)
    ap.add_argument("--cluster", default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = Path(args.data)
    if not path.exists():
        print(f"[error] 数据文件不存在: {path}", file=sys.stderr)
        return 2
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)

    n_raw = len(df)
    cols = [c.strip() for c in args.formula.split("~")[1].split("+") if c.strip()]
    used = [args.formula.split("~")[0].strip()] + cols + ([args.cluster] if args.cluster else [])
    missing = [c for c in used if c not in df.columns]
    if missing:
        print(f"[error] 数据中缺少变量: {missing}", file=sys.stderr)
        return 2

    df = df[used].dropna().reset_index(drop=True)   # 重置索引，避免影响点位置与标签错位
    n_drop = n_raw - len(df)
    mod = smf.ols(args.formula, data=df)
    res = mod.fit()
    exog_names = list(res.model.exog_names)
    p = len(exog_names)
    n = int(res.nobs)
    if n < p + 2:
        print(f"[error] 样本量 {n} 不足，需至少 {p + 2}", file=sys.stderr)
        return 2
    # 异方差检验
    bp_lm, bp_p, bp_f, bp_fp = het_breuschpagan(res.resid, res.model.exog)
    white_lm, white_p, _, _ = het_white(res.resid, res.model.exog)
    jb, jb_p, skew, kurt = jarque_bera(res.resid)

    # 共线性：对设计矩阵各列计算 VIF，截距列不报告
    X = res.model.exog
    vif = {}
    for j, name in enumerate(exog_names):
        if name.lower() in ("intercept", "const"):
            continue
        try:
            vif[name] = float(variance_inflation_factor(X, j))
        except Exception:
            vif[name] = None

    # 影响点
    infl = OLSInfluence(res)
    h = np.asarray(infl.hat_matrix_diag)
    cooks = np.asarray(infl.cooks_distance[0])
    stud = np.asarray(infl.resid_studentized_external)
    lev_thr, cook_thr = 2.0 * p / n, 4.0 / n
    infl_points = [int(i) for i in np.where((h > lev_thr) | (cooks > cook_thr) | (np.abs(stud) > 3))[0]]

    # 剔除影响点后的系数变化（只报告，不执行剔除）
    refit = {}
    if infl_points:
        r2 = smf.ols(args.formula, data=df.drop(df.index[infl_points])).fit()
        refit = {"n_dropped": len(infl_points),
                 "coef_change_pct": {k: float(100 * (r2.params.get(k, np.nan) / res.params.get(k, np.nan) - 1))
                                     for k in res.params.index if abs(res.params.get(k, 0)) > 1e-12}}

    recs = []
    if bp_p < args.alpha:
        recs.append({"code": "USE_ROBUST_SE",
                     "text": f"Breusch-Pagan 检验 p = {bp_p:.4g} < {args.alpha}，"
                             f"建议报告 HC3 稳健标准误（n < 50 时改用 wild bootstrap）。"})
    if args.cluster is not None:
        G = int(df[args.cluster].nunique())
        txt = f"已按 {args.cluster} 聚类，簇数 G = {G}。"
        if G < 40:
            txt += " G < 40 时聚类稳健标准误在小样本下可能过度拒绝，建议 wild cluster bootstrap。"
        recs.append({"code": "CLUSTER_ROBUST", "text": txt})
    if vif and max(v for v in vif.values() if v is not None) > 10:
        recs.append({"code": "COLLINEARITY", "text": "存在 VIF > 10 的变量，系数估计对设定敏感。"})
    if infl_points:
        recs.append({"code": "INFLUENTIAL", "text": f"检出 {len(infl_points)} 个影响点，剔除后系数变化见 influence.refit。"})
    if jb_p < args.alpha and n < 50:
        recs.append({"code": "SMALL_SAMPLE", "text": f"残差非正态（Jarque-Bera p = {jb_p:.4g}）且 n = {n}，"
                                                    f"依赖渐近正态的推断不可靠。"})
    caveats = ["本脚本只做诊断，不做模型选择",
               "统计显著不等于因果效应",
               "Durbin-Watson 仅作提示，不能替代 HAC 标准误"]

    payload = {
        "meta": {"skill": "regression-diagnostics", "version": VERSION,
                 "python": platform.python_version(), "statsmodels": statsmodels.__version__,
                 "numpy": np.__version__, "data_sha256": sha256(path)[:16],
                 "formula": args.formula, "alpha": args.alpha},
        "sample": {"n_raw": n_raw, "n_used": n, "n_dropped_missing": int(n_drop)},
        "fit": {"params": {k: float(v) for k, v in res.params.items()},
                "bse": dict(zip(exog_names, res.bse.tolist())),
                "pvalues": dict(zip(exog_names, res.pvalues.tolist())),
                "r2": float(res.rsquared), "n_params": p},
        "robust_se": se_table(res, df, args.cluster, exog_names),
        "diagnostics": {"breusch_pagan": {"lm": float(bp_lm), "p_value": float(bp_p)},
                        "white": {"lm": float(white_lm), "p_value": float(white_p)},
                        "jarque_bera": {"stat": float(jb), "p_value": float(jb_p),
                                        "skew": float(skew), "kurtosis": float(kurt)},
                        "durbin_watson": float(durbin_watson(res.resid)),
                        "vif": vif},
        "influence": {"leverage_threshold": lev_thr, "cooks_threshold": cook_thr,
                      "n_flagged": len(infl_points),
                      "index_sample": infl_points[:20], "refit": refit},
        "recommendations": recs,
        "caveats": caveats,
    }
    s = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(s, encoding="utf-8")
    print(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
