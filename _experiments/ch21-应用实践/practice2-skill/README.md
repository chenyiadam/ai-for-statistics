# 实践二复现说明（科研 Skill：回归诊断）

## 目录
- `SKILL.md` — Skill 定义（复现自第 21 章 21.3.3 节）
- `scripts/regression_diagnostics.py` — 诊断脚本（21.3.4 节，含 nonrobust 修法与 reset_index 修复）
- `tests/test_smoke.py` — 5 个测试用例（21.3.5 节三例 + 测试表中"输入错误""小样本"两例）
- `runs/` — 证据存档

## 运行命令

```bash
PY=D:/Anaconda/envs/fed_clean_v2/python.exe
cd "I:/mydesk/ai-for-statistics/_experiments/ch21-应用实践/practice2-skill"

# 全部测试（5 例）
$PY tests/test_smoke.py

# 单次完整诊断（异方差数据），JSON 落盘
$PY scripts/regression_diagnostics.py --data tests/case_hetero.csv --formula "y ~ x1 + x2" --out runs/diag_hetero.json
```

## 实测（本机，Python 3.10.20 / statsmodels 0.15.0）

- `runs/test_smoke.log`：5 例全部 ok。
- `runs/diag_hetero.json`：BP 检验 p = 3.04e-05，建议码 `USE_ROBUST_SE` 与 `INFLUENTIAL` 被触发。
- 测试迭代记录：`case_small_sample` 初版用 t3 误差在 n=12 下 JB 未拒绝（小样本下 JB 功效不足，
  这正是正文"渐近检验不可靠"提示的实证），改为注入极端离群点后稳定触发，修改过程已写入测试注释。

## 已知边界

- 公式解析只支持 `+` 连接的简单形式（交互项/变换/分类变量需扩展）。
- 脚本只做诊断不改模型；剔除影响点仅报告变化量。
