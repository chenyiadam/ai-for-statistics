# 实践四复现说明（异方差检验水平失真的蒙特卡洛闭环）

## 目录
- `scripts/sim.py` — 主模拟脚本（21.5.4 节）
- `scripts/closed_loop_demo.py` — 最小闭环演示（21.5.4.1 节）
- `scripts/handcheck.py` — 小样本手算核对与边界检查（21.5.4 核对块）
- `scripts/property_test.py` — 帽子矩阵代数性质属性测试（21.5.6 节）
- `scripts/aggregate.py` — 网格汇总（21.5.5 节）
- `runs/` — 全部证据

## 运行命令（本目录下）

```bash
PY=D:/Anaconda/envs/fed_clean_v2/python.exe
$PY scripts/handcheck.py
$PY scripts/closed_loop_demo.py
$PY scripts/property_test.py
# 全网格（36 个水平配置 + 1 个功效配置，本机串行 5 分 30 秒）
for n in 25 50 100 400; do for d in A B C; do for l in 0 1 3; do
  $PY scripts/sim.py --n $n --design $d --lam $l --R 2000 --Bboot 199 \
    --seed 20260101 --out runs/n${n}_${d}_lam${l}.json
done; done; done
$PY scripts/sim.py --n 50 --design B --lam 3.0 --effect 0.3 --R 2000 --Bboot 199 \
  --seed 20260101 --out runs/power_n50_B.json
$PY scripts/aggregate.py
```

## 实测（Python 3.10.20 / NumPy 1.26.4 / SciPy 1.15.3）

- `runs/quick_checks.log`：
  - 手算核对：statsmodels 与自实现 HC3 在小样本上一致（0.0622），杠杆值之和精确等于 p=2；
  - 闭环演示（R=1000，B=199，n=30，λ=3，高杠杆设计）：OLS 0.486 / HC0 0.353 / HC3 0.117 / WB 0.057——
    与第 21 章 21.5.4.1 节记载完全一致；
  - 属性测试 200 例全部通过。
- `runs/n*_*.json`（36 个配置）+ `runs/power_n50_B.json` + `runs/grid_summary.csv`（216 行）
  + `runs/grid_table.log`（透视表）。
- 结果方向与正文定性预期一致：λ=0 时各方法接近 0.05；设计 B（高杠杆）下 OLS 失真最重
  （n=25、λ=3 时 0.354），HC3 压至 0.106，WB 恢复到 0.063；n=400 时各方法普遍收敛向名义水平。

## 已知现象与边界

- 设计 C（稀疏二元）在小 n 下部分重复的杠杆值 $h_{ii}=1$，HC2/HC3 权重除零产生 NaN，
  该重复按"不拒绝"计入（NaN 与临界值比较为 False）。这是杠杆值达到 1 时 HC2/HC3 定义失效的
  真实边界现象，不是实现错误；严格处理应在该重复回退到 HC0 或 wild bootstrap。
- 结论只适用于本模拟覆盖的 $(n, \text{设计}, \lambda)$ 组合，不构成方法的一般性质。
