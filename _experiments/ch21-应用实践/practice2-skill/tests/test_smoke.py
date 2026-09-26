# tests/test_smoke.py：用已知 DGP 验证诊断是否被正确触发
# 前三例复现自第 21 章 21.3.5 节；后两例对应正文测试表中的"输入错误"与"小样本"行
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
SCRIPT = HERE.parent / "scripts" / "regression_diagnostics.py"


def run(csv: Path, formula: str, cluster: str | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(SCRIPT), "--data", str(csv), "--formula", formula]
    if cluster:
        cmd += ["--cluster", cluster]
    return subprocess.run(cmd, capture_output=True, text=True)


def run_ok(csv: Path, formula: str, cluster: str | None = None) -> dict:
    r = run(csv, formula, cluster)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def case_heteroskedastic():
    """乘性异方差：BP 检验应当检出。"""
    rng = np.random.default_rng(20260101)
    n = 200
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = 1.0 + 0.5 * x1 + 0.3 * x2 + np.exp(x1) * rng.normal(size=n)
    p = HERE / "case_hetero.csv"
    pd.DataFrame({"y": y, "x1": x1, "x2": x2}).to_csv(p, index=False)
    d = run_ok(p, "y ~ x1 + x2")
    assert d["diagnostics"]["breusch_pagan"]["p_value"] < 0.05, "异方差未被检出"
    assert any(r["code"] == "USE_ROBUST_SE" for r in d["recommendations"])


def case_collinear():
    """近共线：VIF 应当超过 10。"""
    rng = np.random.default_rng(20260102)
    n = 200
    x1 = rng.normal(size=n)
    x2 = x1 + rng.normal(scale=0.01, size=n)      # 与 x1 几乎共线
    y = 1.0 + 0.5 * x1 + 0.0 * x2 + rng.normal(size=n)
    p = HERE / "case_collinear.csv"
    pd.DataFrame({"y": y, "x1": x1, "x2": x2}).to_csv(p, index=False)
    d = run_ok(p, "y ~ x1 + x2")
    assert max(v for v in d["diagnostics"]["vif"].values() if v) > 10, "共线性未被检出"
    assert any(r["code"] == "COLLINEARITY" for r in d["recommendations"])


def case_clustered():
    """聚类结构：簇数被正确记录，小簇数触发警告。"""
    rng = np.random.default_rng(20260103)
    G, m = 20, 10
    g = np.repeat(np.arange(G), m)
    u = rng.normal(scale=1.0, size=G)[g]           # 簇随机效应
    x = rng.normal(size=G * m)
    y = 1.0 + 0.4 * x + u + rng.normal(size=G * m)
    p = HERE / "case_cluster.csv"
    pd.DataFrame({"y": y, "x": x, "g": g}).to_csv(p, index=False)
    d = run_ok(p, "y ~ x", cluster="g")
    assert d["robust_se"]["cluster"]["n_groups"] == G
    assert any(r["code"] == "CLUSTER_ROBUST" for r in d["recommendations"])


def case_input_error():
    """公式引用不存在的列：退出码 2，错误信息可读。"""
    rng = np.random.default_rng(20260106)
    n = 50
    p = HERE / "case_ok.csv"
    pd.DataFrame({"y": rng.normal(size=n), "x1": rng.normal(size=n)}).to_csv(p, index=False)
    r = run(p, "y ~ nosuchcol")
    assert r.returncode == 2, f"预期退出码 2，实际 {r.returncode}"
    assert "缺少变量" in r.stderr


def case_small_sample():
    """n=12、参数 3 个：正常输出且触发 SMALL_SAMPLE 提示。

    注：初版用 t3 误差在 n=12 下未被 JB 拒绝（小样本下该检验功效不足），
    注入一个极端离群点后稳定触发——这本身演示了小样本渐近检验的局限与影响点的作用。
    """
    rng = np.random.default_rng(20260107)
    n = 12
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = 1.0 + 0.5 * x1 + 0.3 * x2 + np.exp(rng.normal(size=n)) * 0.8
    y[3] += 40.0   # 极端离群点：确定性地使 JB 拒绝正态
    p = HERE / "case_small.csv"
    pd.DataFrame({"y": y, "x1": x1, "x2": x2}).to_csv(p, index=False)
    d = run_ok(p, "y ~ x1 + x2")
    assert d["sample"]["n_used"] == n
    assert d["diagnostics"]["jarque_bera"]["p_value"] < 0.05, "JB 未拒绝（记录于日志）"
    assert any(r["code"] == "SMALL_SAMPLE" for r in d["recommendations"]), \
        "小样本提示未被触发"


if __name__ == "__main__":
    for f in (case_heteroskedastic, case_collinear, case_clustered,
              case_input_error, case_small_sample):
        f()
        print("ok:", f.__name__)
