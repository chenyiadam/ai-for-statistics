# 第7章 Agent、工具调用与Skill自动化

> **本章定位**：回答"如何把一次性的对话框问答变成可复用、可验证、可交接的科研工作流"。
> **关键词**：Agent、ReAct 范式、Function Calling、MCP、Skill、SKILL.md、渐进式披露、动态工具检索

## 本章导读

前六章讨论的是模型本身：数据从哪来、架构怎么搭、训练与对齐怎样进行、推理时如何解码。模型再强，也只是一个"输入文本、输出文本"的映射；它不会主动读取实验室服务器上的 `experiment.csv`，不会运行一遍 `Rscript`，也不会在发现 p 值算错时回头修改代码。把模型接到真实科研环境上，需要三件东西：一个能自行规划并调用工具的**执行体**（Agent），一套让工具可被发现和被调用的**接口协议**（Function Calling 与 MCP），以及一种把专家流程固化下来、可复用可验证的**封装格式**（Skill）。

本章沿"原理—范式—协议—封装—部署—检索—协作—实践"的顺序展开。7.1 到 7.4 拆解 Agent 的四个组成模块与主流框架；7.5 讨论 MCP 与函数调用，这是 Skill 得以运行的底层机制；7.6 到 7.9 给出 Skill 的定义、SKILL.md 规范、三级加载机制及其与 CLAUDE.md、MCP、Hooks、Subagents 的分工；7.10 与 7.11 是操作层面，给出可直接复制的科研 Skill 模板与本地部署方式；7.12 与 7.16 汇总开源资源；7.13 单独讨论动态工具检索，并把它与统计学中的多重检验与选择性推断问题联系起来；7.14 与 7.15 处理多智能体协作与本地自动化落地。

本章与第 6 章（提示词工程）是继承关系：提示词解决"怎么说"，Skill 解决"怎么做"。与第 8 章（AI 编程与科研自动化）在工具链上重叠，但本章的重心在封装格式与自动化机制，编程智能体的具体使用详见第 8 章；形式化验证与 Lean 相关的内容属第 10 章与第 11 章，本章仅在 7.13 借用其中的检索思想作类比。

## 7.1 Agent原理：规划、记忆、工具、执行

### 7.1.1 从语言模型到智能体

语言模型给出的是条件分布 $P_\theta(y \mid x)$：给定上文 $x$，采样续写 $y$。单次调用是一次无状态的映射。智能体（Agent）在此基础上增加了一个闭环：模型输出不再只是给使用者看的文本，还可以是对外部世界产生作用的**动作**；动作的结果作为新的观测回灌到上下文中，模型据此决定下一步动作，直到满足终止条件。

形式化地，设任务为 $q$，工具集为 $\mathcal{A}$，环境为 $\mathcal{E}$。在第 $t$ 步，模型依据历史
$$h_t = (q, a_1, o_1, \ldots, a_{t-1}, o_{t-1})$$
输出动作 $a_t \sim \pi_\theta(a_t \mid h_t)$，其中 $a_t$ 要么是一个工具调用（含参数），要么是终止并给出最终答案。环境返回观测 $o_t = \mathcal{E}(a_t)$，历史更新为 $h_{t+1} = (h_t, a_t, o_t)$。序列 $\tau = (a_1, o_1, \ldots, a_T)$ 称为一条**轨迹**（trajectory）。整个过程可以视为一个策略 $\pi_\theta$ 在部分可观测环境中的序贯决策问题。

这一形式化对统计学者并不陌生：它与序贯实验设计、序贯检验（sequential testing）的结构一致——每一步根据已积累的观测决定下一步"采样"什么，代价是轨迹之间存在强相关，事后对轨迹做统计汇总时必须考虑这种依赖性。

### 7.1.2 四个组成模块

工程文献通常把基于语言模型的智能体拆为四个模块：

| 模块 | 职责 | 常见实现 | 失效表现 |
|------|------|----------|----------|
| 规划（Planning） | 把目标分解为子任务，决定执行顺序 | 思维链分解、任务列表、图结构编排（LangGraph） | 分解粒度不当，子任务之间循环依赖 |
| 记忆（Memory） | 保存短期上下文与长期知识 | 上下文窗口、向量数据库、外部文件、Skill 库 | 早期信息被截断，长期记忆检索错误 |
| 工具（Tools） | 扩展模型能力的外部接口 | Function Calling、MCP 服务器、代码沙箱 | 选错工具、参数格式不合法 |
| 执行与反思（Execution & Reflection） | 调用工具、解析结果、检查并纠错 | 运行时循环、断言脚本、自检提示 | 错误沿轨迹传播且不再被回头修正 |

四个模块之间存在明确的依赖关系：规划决定调用哪个工具，记忆提供调用所需的参数（例如数据文件路径、上次运行的随机种子），执行结果反过来修正记忆与规划。**反思**（reflection）是把"执行—检查—修正"显式化为一个模块，而不是寄希望于模型在一次生成中自我纠正。

### 7.1.3 记忆的三个层次与"程序性记忆"

认知科学把记忆分为情景记忆、语义记忆与程序性记忆，这一划分在智能体工程中被直接沿用：

1. **情景记忆**：本次会话的对话历史，位于上下文窗口内，容量受限于窗口长度。
2. **语义记忆**：外部事实库，通常用向量数据库检索（RAG），对应"知道什么"。
3. **程序性记忆**：如何做一件事的步骤，对应"会做什么"。**Skill 库即程序性记忆**。

把 Skill 理解为程序性记忆，有助于理解它在上下文预算上的特殊性：程序性记忆的"条目"数量可以很大（一个人会做上千件事），但任意时刻只需要激活其中一两条。这直接导向 7.8 讨论的渐进式披露设计。

从统计角度看，程序性记忆的积累可以类比**经验贝叶斯**（empirical Bayes）中的先验更新：Skill 库是从历史任务中学到的"先验流程"，遇到新任务时先按相似度检索一个流程（先验），再用本次任务的实际观测（似然）修正执行细节，成功的流程被写回库中，成为下一次的先验。

### 7.1.4 图增强智能体

线性循环（thought → action → observation 直到终止）的表达能力有限：无法表达"并行执行三个独立分析""失败则回退到备份方案""两个分支结果投票"。**图增强**（graph-augmented）智能体把执行流程显式建模为有向图，节点是子任务或工具调用，边是控制流与数据流，支持条件分支、循环与并行。LangGraph 是这一思路的代表实现（7.3）。

图结构的代价是编排成本上升：需要显式声明状态模式（state schema）、节点签名与边条件，调试难度高于线性循环。对科研流水线而言，图结构适合"步骤固定、分支明确"的任务（如"读数据 → 缺失值诊断 → 按正态性分流检验 → 汇总"），而开放探索性分析更适合线性循环。

### 7.1.5 评估智能体时的统计问题

Agent 的评测结果是**随机变量的估计值**，不是确定性结论。若某基准包含 $n$ 条任务，观测成功率为 $\hat p$，则其标准误为 $\sqrt{\hat p(1-\hat p)/n}$。在 $\hat p \approx 0.7$、$n = 100$ 的典型设置下，标准误约 0.046，即 95% 置信区间宽度接近 ±9 个百分点。这意味着两个框架在 100 条任务上的成功率相差几个百分点，通常不足以判定优劣。

进一步，同一框架在两条任务上的成功事件并不独立（共享同一套工具、同一提示模板、同一模型版本），真实方差大于二项分布给出的值，需要用聚类稳健方差或在任务族层面做 bootstrap。这一节给出的结论在 7.13、7.14 会反复用到：**Agent 的改进必须在足够多的、彼此异质的任务上验证，单次演示不构成证据**。

### 7.1.6 多步任务成功率的分解与中间校验的收益

7.1.5 把镜头对准跨任务的评测，本节把镜头拉近到单条任务内部：一条多步轨迹为什么比单次调用不可靠得多，以及在步骤之间插入校验能换回多少可靠性。这是全书主线"一个输出凭什么可信"在 Agent 场景下的第一道算术。

最简单的分解模型：任务由 $k$ 个步骤串联，每步一次尝试正确的概率为 $p$，各步相互独立，且任何一步出错都会使最终结果出错。按乘法法则，端到端正确率为

$$P(\text{任务正确}) = p^{k}$$

这是一个对链路长度极其敏感的几何衰减。下表给出三档单步成功率下的数值（由本节脚本解析部分直接算出）：

| $p \backslash k$ | 1 | 2 | 3 | 5 | 8 | 10 |
|------|------|------|------|------|------|------|
| 0.90 | 0.900 | 0.810 | 0.729 | 0.590 | 0.430 | 0.349 |
| 0.95 | 0.950 | 0.902 | 0.857 | 0.774 | 0.663 | 0.599 |
| 0.99 | 0.990 | 0.980 | 0.970 | 0.951 | 0.923 | 0.904 |

单步成功率 0.95 的智能体，"看起来每步都挺可靠"，但跑 5 步只剩 0.774，10 步不足 0.60。这与统计模拟中"每步近似正确、累计后面目全非"的体验同源：单步误差在链路上是相乘而不是相加的。对科研工作流的直接含义是，自动化产出的可信性下界由 $p^k$ 给出，链路越长，单步可靠性越不足以担保最终结论。

改变这一衰减曲线最便宜的手段不是换更强的模型，而是在关键步骤之间部署**中间校验**（intermediate verification）。设每步之后有一个检查点，以概率 $c$ 发现该步的错误；发现则重试一次，重试正确的概率仍为 $p$，重试仍失败则任务中止；未被发现的错误流入下游。此时单步"最终正确"的概率为

$$q \;=\; p + (1-p)\,c\,p$$

端到端正确率变为 $q^k$。下表取 $c = 0.8$：

| $p \backslash k$ | 1 | 2 | 3 | 5 | 8 | 10 |
|------|------|------|------|------|------|------|
| 0.90 | 0.972 | 0.945 | 0.918 | 0.868 | 0.797 | 0.753 |
| 0.95 | 0.988 | 0.976 | 0.964 | 0.941 | 0.908 | 0.886 |
| 0.99 | 0.998 | 0.996 | 0.994 | 0.990 | 0.983 | 0.979 |

对比两表：$p = 0.95$、$k = 5$ 时，端到端正确率从 0.774 提到 0.941。校验器的作用不是让模型更聪明，而是把"错误沿链传播"截断在发生的步骤本地。下面的脚本同时给出解析值与蒙特卡洛核验：

```python
# agent_step_decay.py：多步任务端到端成功率的分解模拟
import numpy as np

K_LIST = [1, 2, 3, 5, 8, 10]
N_SIM, SEED = 20000, 20260101


def step_success(p, c):
    """部署校验后单步"最终正确"的概率：一次成功，或失败被发现后重试成功。"""
    return p + (1 - p) * c * p


def simulate_task(p, k, c, rng):
    """返回一条 k 步任务的结局：correct / silent / detected。"""
    for _ in range(k):
        if rng.random() < p:
            continue                      # 该步一次成功
        if rng.random() < c:              # 失败被中间校验发现
            if rng.random() < p:          # 重试成功
                continue
            return "detected"             # 重试仍失败，任务中止
        return "silent"                   # 失败未被察觉，错误流入下游
    return "correct"


rng = np.random.default_rng(SEED)

for c in [0.0, 0.8]:
    print(f"--- 每步校验发现错误的概率 c={c} ---")
    print("p\\k  " + "".join(f"{k:>8}" for k in K_LIST))
    for p in [0.90, 0.95, 0.99]:
        cells = "".join(f"{step_success(p, c) ** k:>8.3f}" for k in K_LIST)
        print(f"{p:.2f} {cells}")

print("--- p=0.95、无校验时的衰减（ASCII 柱状） ---")
for k in K_LIST:
    v = 0.95 ** k
    print(f"k={k:>2}  {v:6.3f}  |{'#' * round(v * 50)}")

print("--- 蒙特卡洛核验：模拟 vs 解析 ---")
for p, k, c in [(0.95, 5, 0.0), (0.95, 5, 0.8), (0.90, 10, 0.8)]:
    est = sum(simulate_task(p, k, c, rng) == "correct"
              for _ in range(N_SIM)) / N_SIM
    se = (est * (1 - est) / N_SIM) ** 0.5
    print(f"p={p:.2f} k={k:>2} c={c:.1f} | 解析 {step_success(p, c) ** k:.4f}"
          f" | 模拟 {est:.4f} (SE {se:.4f})")
```

实测输出（Python 3.10.20，numpy 1.26.4）：

```text
--- 每步校验发现错误的概率 c=0.0 ---
p\k         1       2       3       5       8      10
0.90    0.900   0.810   0.729   0.590   0.430   0.349
0.95    0.950   0.902   0.857   0.774   0.663   0.599
0.99    0.990   0.980   0.970   0.951   0.923   0.904
--- 每步校验发现错误的概率 c=0.8 ---
p\k         1       2       3       5       8      10
0.90    0.972   0.945   0.918   0.868   0.797   0.753
0.95    0.988   0.976   0.964   0.941   0.908   0.886
0.99    0.998   0.996   0.994   0.990   0.983   0.979
--- p=0.95、无校验时的衰减（ASCII 柱状） ---
k= 1   0.950  |################################################
k= 2   0.902  |#############################################
k= 3   0.857  |###########################################
k= 5   0.774  |#######################################
k= 8   0.663  |#################################
k=10   0.599  |##############################
--- 蒙特卡洛核验：模拟 vs 解析 ---
p=0.95 k= 5 c=0.0 | 解析 0.7738 | 模拟 0.7699 (SE 0.0030)
p=0.95 k= 5 c=0.8 | 解析 0.9414 | 模拟 0.9397 (SE 0.0017)
p=0.90 k=10 c=0.8 | 解析 0.7528 | 模拟 0.7517 (SE 0.0031)
```

三组核验中模拟值与解析值之差都在 1.5 个标准误以内，与蒙特卡洛误差的预期一致。

这个模型有两处偏乐观的假设，做敏感性分析时应当意识到。其一，**重试的独立性**：模型带着同一份上下文重试同一提示，往往重犯同一错误，因此重试成功率取 $p$ 偏高，更保守的设定是打折（例如取 $0.7p$）；其二，**校验器自身的两类错误**——模型中的 $c$ 是发现概率，它同时隐含假阴性（漏检，已体现在 $c < 1$）与假阳性（把正确结果拦下重做，浪费预算但不损害正确性）两类，后者在成本敏感的场景需要单独计价。与统计模拟的做法一致，结论的稳健区间应当通过对 $c$、重试次数与相关性做网格扫描得到，而不是只报告单点。

工程含义可以归纳为一条决策规则：在预算固定的前提下，先估计当前工作流的 $k$ 与 $p$，若 $p^k$ 已低于课题组对产出可信性的要求，优先在误差最集中的步骤后加确定性校验（重算关键统计量、核对文件哈希、断言样本量），其次才是把预算投在提升 $p$ 上。这条规则与 7.7.4 的"确定性工作脚本化"、7.13.6 的"查询改写后重检索"是同一原则的三个侧面：**在链路中插入便宜的、可机械执行的检查点，比提升任何单一环节的能力都更划算**。

## 7.2 ReAct开发范式

### 7.2.1 推理与行动交替

ReAct（Reasoning and Acting，arXiv:2210.03629，*ReAct: Synergizing Reasoning and Acting in Language Models*）是当前绝大多数智能体的执行骨架。它在动作之前插入一段显式"思考"：

$$\Pr(\text{thought}_t \mid h_t) \;\to\; \Pr(\text{action}_t \mid h_t, \text{thought}_t) \;\to\; o_t = \mathcal{E}(\text{action}_t)$$

一次迭代产生三元组 $\text{Thought} \to \text{Action} \to \text{Observation}$，随后进入下一轮。与纯粹的"直接给答案"相比，ReAct 有两方面作用：一是把中间推理显式写入上下文，使后续步骤可以条件化于它；二是让外部观测（而非模型的记忆）成为事实来源，缓解幻觉。

### 7.2.2 为什么显式思考有用：一个隐变量视角

把思考变量记为 $z$，最终答案记为 $y$，则直接回答建模的是边际分布 $P(y \mid x) = \sum_z P(y, z \mid x)$，而 ReAct 建模的是联合分布 $P(y, z \mid x)$ 并逐段采样。这与统计学中的**数据增广**（data augmentation）与 EM 算法的思路一致：显式引入隐变量可以把一个难以直接优化的边际似然，转化为一系列条件更易处理的子问题；代价是采样到的 $z$ 未必来自真实的后验，若 $z$ 采样偏离，$y$ 也会偏离。

这也解释了 ReAct 的一个系统性弱点：思考文本越长，后续步骤对早期错误思考的条件依赖越强，误差在轨迹内累积。对确定性子任务（数值计算、文件移动、格式转换），把工作交给脚本而不是让模型"思考着做"，能显著降低这一风险——这条原则在 7.7 会重申为"确定性工作脚本化"。

### 7.2.3 最小循环实现

下面是一个约 40 行的 ReAct 循环骨架，用于说明三个组件如何协作：

```python
# minimal_react.py：最小 ReAct 循环（示意实现，省略真实 API 调用细节）
TOOLS = {
    "run_r_script": lambda path: subprocess.run(["Rscript", path],
                                                capture_output=True, text=True).stdout,
    "read_csv_head": lambda path: open(path).read()[:2000],
}

SYSTEM = """可用工具：{tools}
每轮输出严格一行 JSON：{{"thought": "...", "action": "工具名或 finish", "args": {{}}}}
收到 observation 后继续下一轮；决定结束时 action 取 finish。"""

def react_loop(query, max_steps=8):
    history = [{"role": "user", "content": query}]      # 历史即 h_t
    for step in range(max_steps):
        reply = call_llm(SYSTEM.format(tools=list(TOOLS)), history)  # 采样 thought + action
        call = json.loads(reply)
        history.append({"role": "assistant", "content": reply})
        if call["action"] == "finish":                  # 终止条件
            return call["args"].get("answer")
        try:
            obs = TOOLS[call["action"]](**call["args"]) # 环境返回观测
        except Exception as exc:
            obs = f"ERROR: {exc}"                       # 错误也作为观测，交给模型修正
        history.append({"role": "user", "content": f"observation: {obs}"})
    return None                                         # 超出步数上限，按失败处理
```

三点值得注意：错误以观测形式回灌，而不是直接抛异常终止；步数上限是必需的，否则病态循环不会停机；`history` 的长度决定了上下文开销，长轨迹需要摘要或裁剪策略。

### 7.2.4 常见失败模式

| 失败模式 | 表现 | 缓解 |
|----------|------|------|
| 工具不收敛 | 反复调用同一工具，观测无新信息 | 设置步数上限，要求每步声明"本步新获得什么" |
| 参数幻觉 | 传入不存在的文件路径或字段名 | 参数先经 Schema 校验，路径先做存在性检查 |
| 观测误读 | 把 stderr 当成结果、截断的输出当作完整 | 脚本输出结构化（JSON），不使用终端回显做事实来源 |
| 早停 | 首轮即给出结论，未真正调用工具 | 在 Skill 中写明"未运行脚本不得给出结论"类硬约束 |
| 错误传播 | 第 2 步算错，后续全部基于错值 | 关键中间结果落盘并做可复算校验 |

表中最后一行的"错误传播"正是 7.1.6 中 $p^k$ 衰减的微观机制：第 2 步的错值一旦通过检查，后续每步的条件分布都建立在这个错值之上，即使后续执行全部"正确"，结论也已不可信。缓解的方向因此不是提高后续步骤的质量，而是在错值产生之处就地拦截——这与表中"关键中间结果落盘并做可复算校验"是同一件事。

### 7.2.5 与统计工作的对接

统计分析流水线天然适合 ReAct：读数据（Action）→ 看缺失比例（Observation）→ 判断是否插补（Thought）→ 选择检验（Action）→ 看诊断图（Observation）→ 决定换用稳健方法。研究者平时在 R 控制台里的交互模式，本质上就是这个循环。ReAct 的价值在于：这个循环可以被记录、被重放、被写成 Skill，从而从"某位研究者脑子里的经验"变成"课题组共享的资产"。

## 7.3 主流Agent框架

### 7.3.1 框架全景

下表汇总原稿整理时收录的智能体框架类项目（星标数为素材整理时的记录，随时间波动）：

| 项目名称 | 简介 | GitHub 地址 |
|---------|------|-----------|
| langchain | 最主流的 Agent/LLM 应用开发框架（146K+ star） | github.com/langchain-ai/langchain |
| langgraph | 基于图结构的复杂 Agent 编排框架 | github.com/langchain-ai/langgraph |
| langchainjs | LangChain 的 JS/TS 版本 | github.com/langchain-ai/langchainjs |
| AutoGPT | 自主智能体开山之作（186K star） | github.com/Significant-Gravitas/AutoGPT |
| Auto-GPT-Forge | AutoGPT 官方 Agent 开发工具包 + 基准测试 | github.com/Significant-Gravitas/Auto-GPT-Forge |
| MetaGPT | "AI 软件公司"多智能体协作框架 | github.com/geekan/MetaGPT |
| dify | 开源 LLM 应用开发平台，可视化工作流 + RAG + Agent（154K star） | github.com/langgenius/dify |
| crewAI | 多角色分工协作 Agent 框架 | github.com/crewAIInc/crewAI |
| autogen | 微软多智能体对话框架 | github.com/microsoft/autogen |
| awesome-ai-agents | E2B 维护的 150+ Agent 框架精选清单（20K star） | github.com/e2b-dev/awesome-ai-agents |
| open-interpreter | 让 LLM 在本地执行代码、控制电脑的 Agent | github.com/openinterpreter/open-interpreter |
| babyagi | 任务分解自主循环 Agent 原型 | github.com/yoheinakajima/babyagi |
| ChatDev | 虚拟软件公司多 Agent 协作开发 | github.com/OpenBMB/ChatDev |
| 12-factor-agents | Agent 工程最佳实践指南 | github.com/humanlayer/12-factor-agents |
| E2B | Agent 安全沙箱云运行时 | github.com/e2b-dev/E2B |

### 7.3.2 按场景的选择建议

| 使用场景 | 推荐框架 | 理由 |
|----------|----------|------|
| 快速搭建文档问答、检索增强原型 | LangChain / dify | 组件最全，dify 提供可视化编排 |
| 步骤固定、分支明确的科研流水线 | LangGraph | 图结构便于声明显式分支与并行 |
| 多角色分工（分析师 / 审稿人 / 写作） | CrewAI / AutoGen / MetaGPT | 内置角色与消息传递机制 |
| 本地文件批处理、跑脚本 | open-interpreter / Claude Code | 直接访问本地运行时 |
| 学习 Agent 内部机制 | babyagi / 12-factor-agents | 代码量小，可读性强 |
| 需要隔离不可信代码执行 | E2B | 提供云端沙箱 |

### 7.3.3 两份系统性学习资料

**Hello-Agents《从零开始构建智能体》**（github.com/datawhalechina/hello-agents）由 Datawhale 出品，定位"AI 原生 Agent"，区别于 Dify、Coze、n8n 这类流程驱动的低代码方案。全书五部分 16 章：第一部分讲智能体定义、类型与发展史并巩固 LLM 基础；第二部分动手实现 ReAct 范式、体验低代码平台、掌握 LangGraph，并从零自研一个智能体框架；第三部分覆盖记忆与检索、上下文工程、Agent 训练、多智能体通信协议与性能评估；第四部分是三个综合案例（智能旅行助手、自动化深度研究智能体、赛博小镇）；第五部分为毕业设计。`code` 目录提供全部配套代码，PDF 开源免费。前置要求是基础 Python 与调用 LLM API 的经验，不需要算法或训练背景。

**Agent-Learning-Hub**（github.com/datawhalechina/Agent-Learning-Hub）面向"做出可靠 Agent 而非收藏链接"，全仓库只维护一个 README，把社区分享、官方博客、论文与开源项目整理成可照做的任务清单。路线按阶段推进：先分清 chatbot、workflow、agent、multi-agent 的边界（产出物是一页笔记，说明为何此处需要 agent 而非工作流）；再手写 50–150 行最小 agent loop；接着做工具、RAG 与记忆（产出资料研究助手）；再学透一个现代 agent harness（工具注册、上下文、权限、状态、日志、子任务与反馈）；随后是可复用 skills、browser agent，以及至少 20 条任务的评测表（期望结果 / 实际结果 / 失败分类）；最终交付一个他人可 clone 即跑的项目。该仓库明确不建议把老式的角色扮演式 crew 框架作为主线，并对 Claude Code 给出"官方文档 → 复刻项目 → 架构解析 → 工程对照"的拆解顺序。

### 7.3.4 选型时的三条经验

第一，框架的抽象层级越高，调试成本越大；对以脚本为主的统计流水线，直接写 ReAct 循环往往比引入全套编排框架更省事。第二，框架与模型解耦，但并非与运行时解耦——沙箱、文件系统权限、网络访问都由运行时决定，更换框架不改变这部分风险。第三，评估先于选型：先用 20–30 条本课题组真实任务做基线（记录成功率与失败分类），再决定框架，避免以演示效果替代证据。

## 7.4 Agent技术栈分层

### 7.4.1 分层视图

| 层级 | 代表实现 | 职责 | 统计科研对应物 |
|------|----------|------|----------------|
| 模型层 | Claude、GPT、Gemini、开源权重模型 | 推理、规划、生成结构化调用 | 计算内核 |
| 接口层 | Function Calling、结构化输出、JSON Schema | 把模型意图转成机器可读的调用 | 类型签名 |
| 协议层 | MCP（Model Context Protocol） | 工具与数据源的统一接入标准 | 驱动 / 连接器 |
| 编排层 | LangGraph、CrewAI、AutoGen、Claude Code | 流程控制、状态管理、权限 | 工作流引擎 |
| 能力层 | Skill（SKILL.md + scripts + references） | 封装可复用的领域流程 | 分析方案 / SOP |
| 运行时层 | Docker、E2B、WebAssembly、本地沙箱 | 隔离执行、文件系统与网络 | 计算环境 |
| 产品层 | Claude Code、Codex CLI、Cursor、dify、各类科研平台 | 面向使用者的成品 | 集成开发环境 |

分层的价值在于**替换自由**：协议层标准化之后，能力层（Skill）可以跨产品复用，运行时层可以从本地换成云端沙箱而不改动上层。Agent Skills 之所以能成为开放标准，正是因为它只约定能力层的目录与清单格式，不绑定模型、编排与运行时。

### 7.4.2 分层带来的兼容性问题

分层也引入新的故障面：同一份 Skill 在 Claude Code 与 Codex CLI 下的可用工具集不同，脚本依赖的运行时（Python 版本、R 版本、系统库）也可能不一致。实践中的做法是在 Skill 中显式声明环境需求（Python 版本、依赖包、安装命令），并在脚本开头打印环境信息，与 `sessionInfo()` 在 R 中的角色相同。

### 7.4.3 与统计软件栈的类比

这套分层与统计研究者熟悉的软件栈高度同构：模型层对应数值内核（BLAS、LAPACK），接口层对应函数签名与 S3/S4 方法派发，协议层对应数据库驱动（DBI、ODBC），编排层对应 `targets`/`make`/`drake` 这类流水线管理，能力层对应分析方案（SAP，statistical analysis plan），运行时层对应容器与 renv/conda 环境。**Skill 在统计语境下最接近"分析方案的机器可读版本"**——这也是本章强调 Skill 而非提示词的原因：分析方案需要版本、需要评审、需要复现，提示词不具备这些属性。

## 7.5 MCP与函数调用

### 7.5.1 函数调用（Function Calling）

函数调用是模型选择工具并给出参数的机制：开发者以 JSON Schema 描述函数签名，模型输出符合签名的结构化调用，运行时执行后把结果作为观测回灌。它解决的是"让自然语言意图落到机器可执行的参数上"这一环节。OpenAI Function Calling 的官方文档见 platform.openai.com/docs/guides/function-calling。

下面是一个统计工具的 Schema 声明示例（声明两样本 t 检验的调用契约，供模型生成参数）：

```json
{
  "name": "two_sample_ttest",
  "description": "对两组独立样本执行 t 检验，返回统计量、自由度与 p 值",
  "parameters": {
    "type": "object",
    "properties": {
      "x":       { "type": "array", "items": {"type": "number"}, "description": "第一组观测值" },
      "y":       { "type": "array", "items": {"type": "number"}, "description": "第二组观测值" },
      "equal_var": { "type": "boolean", "default": true, "description": "是否假定方差齐性" },
      "alternative": { "type": "string", "enum": ["two-sided", "less", "greater"], "default": "two-sided" }
    },
    "required": ["x", "y"]
  }
}
```

Schema 的质量直接决定调用成功率：`enum` 约束把自由文本收敛到有限选项，`default` 减少必填参数，`description` 是模型判断"何时用这个工具"的唯一依据。Schema 描述写得含糊，与 Skill 的 `description` 写得含糊后果相同——工具永远不会被选中。

**Toolformer**（arXiv:2302.04761，*Toolformer: Language Models Can Teach Themselves to Use Tools*，Meta AI）给出了另一条路径：模型通过自监督决定何时调用 API、传什么参数、如何融合结果，在少量示例下自行学会工具调用，不需要逐步人工标注。工程实践中，训练式方案与 Schema 声明式方案是互补的：前者提升模型的工具使用直觉，后者提供可校验的契约。

**从"约定"到"保证"：结构化输出只覆盖格式层。** 各家 API 提供的结构化输出（structured output）通过**受限解码**（constrained decoding）实现：解码器按 JSON Schema 编译出的文法，在每一步屏蔽不合法的 token，使输出在**语法上**必然符合 Schema。这一保证是机械的、百分之百的，开源实现如 Outlines（github.com/dottxt-ai/outlines）可复现同样的机制。但 Schema 约束不到语义层：路径是否存在、列名是否属于这份数据集、单位是毫米还是厘米、`equal_var` 该取 true 还是 false，全部依赖模型对上下文的理解。换言之，受限解码把"解析失败"这一类错误完全消除，把全部残余风险集中在语义错误上；后者只能靠 Schema 之外的运行时校验兜底。

下面的模拟把"校验改变错误结构"这一论断量化。设定一次工具调用尝试有四种结局：参数完全正确（0.75）、格式错误（0.15，类型或枚举不合法，Schema 可确定性拦截）、可被运行时发现的语义错误（0.08，文件不存在、列名缺失等）、不可被运行时发现的语义错误（0.02，取错同名列、量纲弄反等）。比较三种策略，每次任务最多尝试 3 次：A 无校验，第一次调用即执行；B 仅 Schema 校验，格式错误被拦截后重试；C Schema 加运行时断言（runtime assertion，`assert os.path.exists(path)`、断言列名与样本量等），可发现的语义错误也触发重试。关注三个量：任务成功率、被察觉的失败、**静默错误**（silent error，错误结果不被察觉地流入下游）。

```python
# param_validation.py：三种参数校验策略下错误结构的蒙特卡洛模拟
import numpy as np

P_OK, P_FMT, P_SEM_VIS, P_SEM_INV = 0.75, 0.15, 0.08, 0.02
MAX_ATTEMPTS, N_SIM, SEED = 3, 20000, 123
OUTCOMES = ["success", "detected", "silent"]


def draw_call(rng):
    u = rng.random()
    if u < P_OK:
        return "ok"
    if u < P_OK + P_FMT:
        return "format"
    if u < P_OK + P_FMT + P_SEM_VIS:
        return "sem_vis"
    return "sem_inv"


def run_task(strategy, rng):
    for i in range(MAX_ATTEMPTS):
        last = i == MAX_ATTEMPTS - 1
        o = draw_call(rng)
        if o == "ok":
            return "success"
        if o == "format":
            # 无校验时格式错误在运行时崩溃，可察觉；有 Schema 时拦截重试
            if strategy != "none" and not last:
                continue
            return "detected"
        if o == "sem_vis":
            if strategy == "schema+runtime" and not last:
                continue                    # 运行时断言发现，重试
            if strategy == "schema":
                return "silent"             # Schema 对语义盲，静默通过
            return "detected"               # 无校验：运行时报错
        return "silent"                     # sem_inv：三种策略都静默
    return "detected"


rng = np.random.default_rng(SEED)
print(f"单次尝试分布：正确 {P_OK} / 格式错误 {P_FMT} / "
      f"可发现语义错误 {P_SEM_VIS} / 不可发现语义错误 {P_SEM_INV}")
print(f"最多尝试 {MAX_ATTEMPTS} 次，n_sim={N_SIM}，seed={SEED}")
print(f"{'策略':<16}{'成功':>8}{'被察觉失败':>12}{'静默错误':>10}"
      f"{'静默占完成比':>13}")
for strategy, label in [("none", "A 无校验"),
                        ("schema", "B 仅Schema"),
                        ("schema+runtime", "C Schema+断言")]:
    res = [run_task(strategy, rng) for _ in range(N_SIM)]
    n = len(res)
    s = {k: res.count(k) / n for k in OUTCOMES}
    share = s["silent"] / (1 - s["detected"])
    print(f"{label:<16}{s['success']:>8.4f}{s['detected']:>12.4f}"
          f"{s['silent']:>10.4f}{share:>13.4f}")
```

输出（Python 3.10.20，numpy 1.26.4；seed=123，n_sim=20000）：

```text
单次尝试分布：正确 0.75 / 格式错误 0.15 / 可发现语义错误 0.08 / 不可发现语义错误 0.02
最多尝试 3 次，n_sim=20000，seed=123
策略                    成功       被察觉失败      静默错误       静默占完成比
A 无校验             0.7557      0.2250    0.0193       0.0250
B 仅Schema         0.8800      0.0031    0.1169       0.1173
C Schema+断言       0.9639      0.0112    0.0249       0.0252
```

（B、C 两行的模拟值与解析式 $P(\text{成功}) = \sum_{i=0}^{2}(0.15)^{i}\times 0.75$ 等闭式结果一致，偏差在蒙特卡洛误差内；A、B、C 对应的解析值分别为 0.75、0.879、0.962。）

解读有三点。第一，仅加 Schema 校验（B）把成功率从 0.75 提到 0.88，却让静默错误从约 2% 升到约 12%：格式错误被拦截后任务"总能跑完"，语义错误不再以崩溃的形式暴露，而是混进顺利完成的结果里。校验在这里把被察觉的失败**转移**成了成功与静默错误，而不是消灭了错误；只报告"任务完成率提升"的评估会掩盖这一转移。第二，加上运行时断言（C）才同时改善两个指标：成功率 0.96，静默错误回落到 2.5%。断言本身是廉价的确定性代码，其可靠性与被调用工具同级，这正是 7.7.4 "确定性工作脚本化"的直接论据。第三，静默错误占完成任务的份额（最后一列）是科研场景最该盯住的指标：被察觉的失败可以重跑，静默的错误结果会进入论文。对统计科研，最小的一组运行时断言应包括：输入文件存在、关键列存在且类型正确、样本量在预期范围内、关键统计量与一条独立计算路径一致（该闭环的具体实现见第 8.9.3 节）。

模拟使用的四个概率是示意参数，实际数值应从本课题组的调用日志估计：对一批真实任务按上述四类结局逐一标注，即得到四个比例的估计值与标准误。比例 0.08 在 $n = 200$ 时的标准误约 0.019，因此几十条日志不足以支撑策略比较。另外，"重试后各次尝试独立"的假设同样偏乐观，模型重试时常重犯同一错误，把重试成功率打折是更保守的设定；方向性结论（校验改变错误结构、断言压低静默错误）不受影响。

### 7.5.2 MCP 协议

MCP（Model Context Protocol）由 Anthropic 于 2024 年 11 月提出并开源，定义模型与数据源、工具之间的统一接口标准。它解决的是"每个数据源都要写一套适配代码"的碎片化问题：按 MCP 实现的服务器可以被任何支持 MCP 的客户端发现与调用，与模型厂商无关。官方规范见 modelcontextprotocol.io，参考实现见 github.com/modelcontextprotocol/servers。

一个 MCP 服务器可以暴露三类能力：

| 能力类型 | 语义 | 统计科研示例 |
|----------|------|--------------|
| Tools | 可被模型调用的动作 | 提交 R 脚本、查询文献库、写入数据库 |
| Resources | 可被读取的只读内容 | 数据集元信息、实验记录、报告模板 |
| Prompts | 预置的提示模板 | 期刊格式要求、审稿意见模板 |

传输层支持 stdio（本地进程，客户端以子进程方式启动服务器）与基于 HTTP 的远程传输两种模式。本地科研场景多用 stdio：服务器与数据在同一台机器上，不经过网络，权限由本地文件系统控制。

### 7.5.3 MCP 配置示例

下面的配置在 Claude Code 的 `settings.json` 中同时挂载文件系统与 Zotero 两个 MCP 服务器（`env` 中的密钥应改用环境变量或密钥管理器注入，不要写死在配置文件里）：

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/project"]
    },
    "zotero": {
      "command": "npx",
      "args": ["-y", "zotero-mcp-server"],
      "env": { "ZOTERO_API_KEY": "${ZOTERO_API_KEY}" }
    }
  }
}
```

配置生效后，模型可直接列出目录、读写文件、检索 Zotero 文献库、批量导入条目并生成引用格式。

### 7.5.4 MCP 的局限

MCP 标准化的是**连接**，不是**正确性**。服务器返回的数据是否可信、工具是否有副作用、并发调用是否会覆盖文件，协议本身不保证。因此：写操作类的 MCP 工具应在客户端侧配置人工确认；长事务（例如批量修改文献条目）应先输出差异预览；对外网数据源的调用要记录时间戳与版本，否则复现时无法追溯。

对统计研究而言，MCP 的真正价值在于**把数据源接入从定制代码变成配置项**：同一个分析脚本可以对接本地 CSV、机构数据库与云存储，只要三者都提供 MCP 接口。这与 DBI 包让 R 代码与具体数据库解耦的思路一致。

## 7.6 Skill范式：定义与核心优势

### 7.6.1 Skill 不是提示词

Agent Skills 是 Anthropic 于 2025 年 10 月发布的开放标准（agentskills.io），其物理形态是"一个文件夹加一个 SKILL.md 清单文件"，把可重复的科研工作流封装为智能体按需加载的技能包。它与提示词工程的区别不在长度，而在契约：

| 维度 | 提示词 | Skill |
|------|--------|-------|
| 形态 | 一段自然语言 | 目录 + 清单 + 脚本 + 参考资料 |
| 内容 | 告诉模型"怎么说" | 规定"怎么做"：步骤、脚本、判定标准 |
| 加载 | 每次会话都要复制粘贴 | 一次安装，按 description 自动路由 |
| 可验证性 | 无 | 有输入契约、输出格式与禁止行为 |
| 版本与共享 | 口头传递 | 目录可入 git，随仓库分发 |
| 确定性工作 | 由模型即兴生成 | 脚本化执行，模型不参与计算 |

Skill 包含四类要素：**输入输出规范**（接受什么参数、产出什么文件）、**执行逻辑**（编号步骤，而非散文体）、**依赖工具链**（脚本、MCP 服务器、命令行工具）、**验证规则**（判定通过与否的阈值、禁止行为清单）。

### 7.6.2 三层架构

Skill 的运行依赖模型的函数调用机制与上下文环境的交互，可拆为三层：

```
┌─────────────────────────────────────────────────────────┐
│                      大语言模型                          │
│     （意图识别 / 策略规划 / 参数提取 / 错误修正）          │
└──────────────────────────┬──────────────────────────────┘
                           │ 输出 JSON Function Call
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Skill 路由与控制层                          │
│   • Dynamic Tool Retrieval（按需动态加载工具）            │
│   • Policy Check & Validation（参数校验与安全控制）       │
│   • MCP (Model Context Protocol) / Standard Schema       │
└──────────────────────────┬──────────────────────────────┘
                           │ 调度与执行
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  底层执行沙箱                            │
│   • Docker / WebAssembly / Python Code Interpreter       │
│   • API 网关 / 数据库连接器 / RStudio / Matlab            │
└─────────────────────────────────────────────────────────┘
```

路由与控制层是 Skill 区别于普通脚本目录的关键：它决定"此刻该加载哪个 Skill"，并在此前完成参数校验与安全检查。

### 7.6.3 三大核心技术范式

**范式一：Schema 规范定义。** 用 JSON Schema 描述 Skill 的名称、功能与输入参数类型，模型生成符合签名的结构化调用（7.5.1）。

**范式二：MCP。** 统一模型与数据源、工具的接口标准，使 Skill 具备跨平台可移植性。GitHub、Notion、Stripe、MongoDB 等公司已官方提供 MCP 服务器。

**范式三：渐进式披露三级加载。** 这是 Skill 在上下文工程上的核心设计，详见 7.8。

### 7.6.4 代表性平台与工具

| 平台 | 时间 | 定位 | 与统计科研的关系 |
|------|------|------|------------------|
| SClaw（国家超算互联网） | 2026 年 3 月上线 | 集成科研 Skill、大模型路由引擎、科学数据库与知识库；开箱即用，客户端升级即可部署 | 技能中心涵盖前沿文献追踪、文档处理、浏览器操控、安全审查；打通飞书、钉钉、企业微信；支持 7×24 小时周期性托管任务（每日动态汇总、每周数据报告、系统巡检） |
| OpenBioMed Skills（清华 AIR × 水木分子） | 2026 年 3 月发布 | 全球首个把生物医药专家决策流程完整编码为可执行代码的 Agent Skill Set | 靶点识别、分子生成、ADMET 预测、临床试验设计被封装为标准 Skill 模块；临床试验设计环节与统计设计理论直接相关 |
| LabClaw（斯坦福 × 普林斯顿） | 开源 | 生物医学科研技能包，含 200 多个技能 | `pubmed-search` 负责检索，`citation-management` 负责引用整理，另有实验数据分析与统计建模类 Skill |
| MindSpore Science（华为） | 2026 年 4 月发布 | 模型类 Skill 自动构建，分钟级把自有模型封装为智能体 Skill | 适合把课题组自研的估计或预测模型封装为可复用工具 |
| MolClaw（上海 AI 实验室） | 2026 年 5 月发布 | 首个长程自主运行、技能可持续扩展的新药筛选智能体 | 基于 SCP（Skill Composition Protocol）实现技能的动态编排与复用 |
| Claude Science（Anthropic） | 2026 年 6 月发布 | 面向科学家的 AI 工作台，定位为"工作台"而非聊天机器人 | 通过 Skill 系统整合文献检索、数据分析、代码执行、论文写作全流程 |

### 7.6.5 技能库范式的源头：Voyager

"技能库自动积累"这一范式来自 **Voyager**（arXiv:2305.16291，*Voyager: An Open-Ended Embodied Agent with Large Language Models*；作者 Guanzhi Wang、Yuqi Xie、Yunfan Jiang、Ajay Mandlekar、Chaowei Xiao、Yuke Zhu、Linxi Fan、Anima Anandkumar，来自 NVIDIA、Caltech 与 UT Austin；代码 github.com/MineDojo/Voyager，MIT 许可；项目主页 voyager.minedojo.org）。

Voyager 由三个组件构成：自动课程、持续增长的技能库、迭代提示机制。技能以**可执行代码**形式存储，可解释、可组合；智能体通过环境反馈、执行错误与自我验证改进程序。论文报告的量化效果为：获得 3.3 倍独特物品、旅行距离 2.3 倍、技术树解锁速度提升 15.3 倍，且技能库可跨世界复用。

官方复现流程如下（需要 Minecraft Java Edition 1.19+ 与 API Key）：

```bash
# 1. 克隆仓库并安装 Python 依赖
git clone https://github.com/MineDojo/Voyager
cd Voyager
pip install -e .

# 2. 安装 Node.js 侧依赖（Mineflayer 环境）
cd voyager/env/mineflayer
npm install -g npx
npm install
cd mineflayer-collectblock
npx tsc
cd ..
npm install
```

```python
# 3. 启动终身学习：技能自动积累到 skill_library/
from voyager import Voyager

azure_login = {
    'client_id': 'YOUR_CLIENT_ID',
    'redirect_url': 'https://127.0.0.1/auth-response',
    'secret_value': '[OPTIONAL] YOUR_SECRET_VALUE',
    'version': 'fabric-loader-0.14.18-1.19',
}
voyager = Voyager(azure_login=azure_login, openai_api_key='YOUR_API_KEY')
voyager.learn()

# 从检查点恢复训练
voyager = Voyager(azure_login=azure_login, openai_api_key='YOUR_API_KEY',
                  ckpt_dir="YOUR_CKPT_DIR", resume=True)

# 加载已学技能库解决新任务
voyager = Voyager(azure_login=azure_login, openai_api_key='YOUR_API_KEY',
                  skill_library_dir="./skill_library/trial1",
                  ckpt_dir="YOUR_CKPT_DIR", resume=False)
sub_goals = voyager.decompose_task(task="Craft a diamond pickaxe")
voyager.inference(sub_goals=sub_goals)
```

Voyager 对科研 Skill 的启示有三点：技能以代码而非提示词存储，因此可以被单元测试；技能库随任务积累，是"程序性记忆"的实证形态；技能跨任务复用时的迁移效果可以量化评估。

### 7.6.6 局限与风险

Skill 不是万能封装。第一，Skill 的质量上限受制于编写者对流程的理解——把错误的方法论固化成 Skill，只会让错误更快传播。第二，Skill 之间会冲突：描述重叠导致路由不确定，同名 Skill 按优先级覆盖（见 7.11.1），调试难度上升。第三，安装第三方 Skill 等于在本机执行未知脚本，属于提示词注入与代码执行的双重风险面，应在安装前阅读 `scripts/` 内容，并在沙箱或受限权限下试运行。第四，Skill 的"验证规则"只是阈值约定，不具备统计保证；把 7.10.1 中 $|\hat\alpha - \alpha| < 0.005$ 这类判定当作正式检验，需要明确其蒙特卡洛误差（在 $n_{\text{sim}} = 10^4$、$\alpha = 0.05$ 时，$\hat\alpha$ 的标准误约 $\sqrt{0.05 \cdot 0.95 / 10^4} \approx 0.0022$，阈值 0.005 约为 2.3 个标准误量级，属粗略判据）。

## 7.7 SKILL.md规范与标准目录结构

### 7.7.1 规范来源

Agent Skills 是开放标准，规范地址为 agentskills.io；GitHub 仓库 github.com/anthropics/skills 内含 `./spec` 完整规范与 `./template` 模板。标准目录结构如下（文件夹名必须为 kebab-case，即小写字母加连字符；`SKILL.md` 文件名区分大小写）：

```
your-skill-name/
├── SKILL.md                # 必需：清单与指令
├── scripts/                # 可选：确定性工作放进脚本
├── references/             # 可选：长文档、规范、示例
└── assets/                 # 可选：模板、样式、字体等
```

### 7.7.2 frontmatter 字段

SKILL.md 顶部为 YAML frontmatter。必填字段只有 `name` 与 `description` 两项，其余为可选：

| 字段 | 必填 | 作用与约定 |
|------|------|------------|
| `name` | 是 | 标识符，kebab-case；用于 `@skill:name` 显式调用 |
| `description` | 是 | 路由依据，需同时回答"做什么"（WHAT）与"何时用"（WHEN） |
| `version` | 否 | 语义化版本，便于团队管理与回滚 |
| `allowed-tools` | 否 | 限定该 Skill 可使用的工具（如 `Bash`, `Read`, `Write`） |
| `license` | 否 | 分发许可 |
| `agent_created` | 否 | 标记由 AI 创建，允许后续自动更新（WorkBuddy 等平台使用） |

`description` 是**唯一的路由规则**：智能体启动时只读取各 Skill 的 name 与 description，靠这段文字判断相关性。写 description 时应嵌入使用者实际会说的短语（如 "weekly report"、"simulation study"、"验证检验"），而不是抽象的功能概括。

### 7.7.3 最小模板

下列模板取自官方 `./template`，可直接复制后改写：

````markdown
---
name: my-skill-name
description: A clear description of what this skill does and when to use it
---
# My Skill Name
[Add your instructions here that Claude will follow when this skill is active]

## Examples
- Example usage 1
- Example usage 2

## Guidelines
- Guideline 1
- Guideline 2
````

### 7.7.4 编写准则

1. **description 优先**：同时给出 WHAT 与 WHEN，并写入真实触发短语。
2. **正文控制在 500 行以内**：超出部分拆到 `references/`，由智能体按需读取。
3. **用编号步骤而非散文**：写明确约束与确切路径，避免"酌情处理"这类模糊指令。
4. **确定性工作脚本化**：数学计算、文件重命名、API 调用一律写成脚本由智能体执行，不让模型即兴生成代码——即兴生成的计算无法复现，也无法被单元测试覆盖。
5. **写明禁止行为**：例如"不得在未运行脚本的情况下给出结论"，硬约束比提示语有效。
6. **声明环境**：Python/R 版本、依赖包与安装命令写入正文（示例见下）。

```markdown
**Environment**
- Python 3.10+
- Packages: numpy, scipy, matplotlib, pysal, geopandas
- Install: pip install -r requirements.txt
```

### 7.7.5 元 Skill：skill-creator

Anthropic 官方提供 `skill-creator` 元技能：描述所需流程，它自动生成规范的 SKILL.md 并做审查迭代，据官方资料可在 15–30 分钟内产出第一个可用 Skill，同时生成 `scripts/` 与 `references/`。用法见 7.11.4。

社区实现的 `reflect` Skill 提供持续学习机制：会话结束后扫描对话记录中使用者对智能体的纠正，提出对 Skill 文件的修改建议并附置信度，人工批准后写入 git。这是**不触碰模型权重**的持续学习，对科研团队的意义在于避免重复纠正同一类错误。

## 7.8 Skill三级加载与渐进式披露

### 7.8.1 三级内容

渐进式披露（progressive disclosure）把 Skill 的内容按代价分为三级：

| 级别 | 内容 | 加载时机 | 单条开销量级 |
|------|------|----------|--------------|
| 第一级 | YAML frontmatter（name + description） | 始终加载进系统提示 | 数十 token |
| 第二级 | SKILL.md 正文 | 判定相关后才加载 | 数百到数千 token |
| 第三级 | `references/`、`scripts/`、`assets/` | 执行过程中按需读取 | 不定，用到才读 |

这一设计使得"拥有上千个 Skill 的智能体"不会因上下文爆炸而失效：任意时刻只有相关 Skill 的全文进入上下文，其余只占 name 与 description 的开销。

### 7.8.2 上下文预算的算术

设窗口上限 200K token，安装 $N = 1000$ 个 Skill，每个 frontmatter 约 50 token，则一级常驻开销约 $5 \times 10^4$ token，占窗口的四分之一。若不做分级、把每个 Skill 的全文（以 2000 token 计）全部常驻，则开销为 $2 \times 10^6$ token，超出窗口一个数量级。分级把"随 $N$ 线性增长的全文开销"替换为"随 $N$ 线性增长的描述开销 + 随当次任务数增长的正文开销"，这是三级加载的全部意义。

需要注意，一级开销虽小但并非免费：$N$ 很大时，description 之间的语义重叠会干扰路由判断（7.13 讨论的检索错误即由此产生）。因此社区实践建议控制常驻 Skill 数量，或按项目分组启用。

### 7.8.3 加载失败的诊断

| 现象 | 可能原因 | 排查 |
|------|----------|------|
| Skill 从未被触发 | description 未覆盖使用者的实际措辞 | 在 description 中加入真实触发短语 |
| Skill 被触发但指令未被执行 | 正文过长被截断，或关键约束写在 `references/` 中未被读取 | 把硬约束放进正文前 100 行 |
| 脚本找不到 | 相对路径基准不确定 | 在正文中给出绝对路径或"相对于 Skill 根目录"的明确说明 |
| 同名 Skill 行为异常 | 优先级覆盖 | 按 7.11.1 的优先级顺序检查各安装位置 |

### 7.8.4 与统计方法的类比

三级加载与统计学中的**两阶段筛选检验**（screening then confirmatory testing）结构相同：第一阶段用廉价指标（description / 边缘统计量）对全部候选做粗筛，控制的是计算成本；第二阶段只对被选中的少数候选投入昂贵资源（全文加载 / 精确检验），控制的是结论质量。两者的失效模式也一致：第一阶段筛错，第二阶段再精确也无济于事——这正是 7.13 要量化的问题。

## 7.9 Skill与CLAUDE.md、MCP、Hooks、Subagents的关系

### 7.9.1 五种机制的分工

| 机制 | 解决什么问题 | 类比 |
|---|---|---|
| CLAUDE.md | 每次会话都要知道的项目事实（构建命令、目录约定） | 宪法 |
| Skill | 按需加载的特定任务流程 | 操作手册 |
| MCP Server | 连接外部服务和可执行工具（数据库、浏览器、文献库） | 手脚 / 接口 |
| Hooks | 确定性副作用（每次保存自动跑测试） | 反射神经 |
| Subagents | 隔离上下文的子任务 | 外包 |

一句话概括分工：**规则放 CLAUDE.md，流程做成 Skill，连外部系统用 MCP，强制检查用 Hooks，隔离子任务用 Subagents**。

### 7.9.2 如何判断内容该放哪里

| 待固化内容 | 归属 | 判断依据 |
|------------|------|----------|
| "本仓库用 `make all` 构建，测试数据放在 `tests/data/`" | CLAUDE.md | 每条会话都需要，与具体任务无关 |
| "跑功效分析时必须先做 10000 次模拟并输出覆盖率" | Skill | 只在做功效分析时需要，步骤明确 |
| "读取实验室 PostgreSQL 数据库" | MCP Server | 需要外部连接与凭据 |
| "每次保存 `.R` 文件自动执行 `lintr` 检查" | Hooks | 由事件触发，无需模型判断 |
| "把 30 篇文献的摘要全部读一遍并交叉比对" | Subagent | 上下文占用大，主流程只需结论 |

判断标准有三条：是否需要模型判断（不需要则考虑 Hooks）、是否每次会话都要（是则 CLAUDE.md）、是否只在特定任务下需要（是则 Skill）。

### 7.9.3 组合使用的一个实例

一个统计课题组的典型配置：CLAUDE.md 写明"数据目录 `data/raw` 只读，`results/` 放产出，随机种子固定为 20260101"；Skill 提供"回归诊断报告"与"功效分析"两个流程；MCP 挂载文件系统与 Zotero；Hooks 在每次写入 `results/*.csv` 后自动运行校验脚本，检查列数与缺失率是否符合约定；长文献调研任务交给 Subagent，只回传结构化摘要。这样配置后，智能体的行为边界由 CLAUDE.md 与 Hooks 硬性约束，流程细节由 Skill 按需提供，数据连接由 MCP 承担。

### 7.9.4 需要注意的边界

CLAUDE.md 不宜写入流程细节，否则每个会话都要为此付出 token，且修改流程会影响所有任务。Skill 不宜写入项目事实（如具体路径），否则同一份 Skill 换到别的项目就失效。Hooks 不宜做需要判断的事（如"检查结论是否合理"），它只适合确定性动作。Subagent 的上下文隔离也意味着它看不到主会话的历史，需要显式传递所需信息。

## 7.10 科研Skill模板

本节给出七个可直接复制的完整模板，其中前四个与统计学直接相关（蒙特卡洛检验验证、实验设计与功效分析、回归诊断、保形预测区间），后三个覆盖出版绘图、R 流水线与文献管理。所有模板遵循 7.7 的规范，脚本均带注释。

| 编号 | 模板 | 触发场景 | 关键脚本 |
|------|------|----------|----------|
| 7.10.1 | monte-carlo-hypothesis-test | 验证检验的第一类错误率、功效与覆盖率 | `scripts/run_simulation.py` |
| 7.10.2 | experimental-design-power | 样本量计算与功效曲线 | `scripts/power_curve.py` |
| 7.10.3 | regression-diagnostics | 回归假设诊断与稳健替代 | `scripts/diagnostics.py` |
| 7.10.4 | conformal-interval | 保形预测区间与覆盖率校验 | `scripts/conformal.py` |
| 7.10.5 | publication-figure | 符合期刊规范的统计图表 | `scripts/make_figure.py` |
| 7.10.6 | r-analysis-pipeline | 标准化 R 分析流程 | `scripts/run_analysis.R` |
| 7.10.7 | lit-review | arXiv 文献追踪与结构化综述 | `scripts/search_arxiv.py` |

### 7.10.1 蒙特卡洛假设检验验证 Skill

统计学中检验方法的理论性质（第一类错误率 $\alpha$、功效 $1-\beta$、置信区间覆盖率）都可以通过大量重复模拟来验证。下列 Skill 把这一验证流程封装为可复用资产。

````markdown
---
name: monte-carlo-hypothesis-test
description: 通过蒙特卡洛模拟自动验证假设检验的 p 值、功效函数与置信区间覆盖率的正确性。适用于 t 检验、卡方检验、方差分析、非参数检验的结果交叉验证。Use when user asks for "验证检验"、"simulation study"、"Monte Carlo"、"交叉验证p值"。
---

# 蒙特卡洛假设检验验证

## 核心逻辑
任何检验方法的名义性质（第一类错误率 α、功效 1-β、置信水平覆盖率）
都可通过大量重复模拟估计。本 skill 自动执行该验证流程，并给出通过/失败判定。

## 执行步骤

### 步骤 1：解析检验场景
读取用户输入：
- 检验方法名称（如 two-sample t-test）
- 原假设 H0 与备择假设 H1
- 名义显著性水平 α（默认 0.05）
- 样本量 n1, n2
- 效应量 effect_size（默认 0.0，即 H0 为真）

### 步骤 2：调用验证脚本
运行 `scripts/run_simulation.py`，参数：
`--test [检验名] --alpha [α] --n1 [样本量] --n2 [对照样本量] --n-sim [模拟次数，默认 10000]`

### 步骤 3：结果对比
脚本输出：
- 模拟第一类错误率 vs 名义 α
- 在 H1 为真时的模拟功效 vs 理论功效
- 置信区间的模拟覆盖率 vs 名义置信水平

### 步骤 4：生成结论报告
判定标准：
- |模拟α − 名义α| < 0.005 → 验证通过
- 功效偏差 < 2% → 验证通过
- 覆盖率落在 [名义水平 − 1%, 名义水平 + 1%] → 验证通过

## 输出格式
| 指标 | 理论值 | 模拟值 | 偏差 | 判定 |
|------|--------|--------|------|------|
| 第一类错误率 | 0.050 | 0.048 | -0.002 | 通过 |
| 功效 | 0.800 | 0.812 | +0.012 | 通过 |
| 覆盖率 | 0.950 | 0.947 | -0.003 | 通过 |

## 禁止行为
- 不得在 n_sim < 1000 时给出"验证通过"结论
- 不得跳过对 H0 为真情形的模拟
- 不得在未运行 scripts/run_simulation.py 的情况下直接给出结论
- 不得把模拟结果表述为精确值，须同时报告 n_sim 与随机种子

## Bundled files
- scripts/run_simulation.py        # 模拟主脚本
- references/report-template.md    # 报告模板
````

配套脚本 `scripts/run_simulation.py`（原始实现，保留并补注）：

```python
#!/usr/bin/env python3
"""蒙特卡洛假设检验自动验证脚本：估计两样本 t 检验的第一类错误率、功效与覆盖率。"""
import numpy as np
from scipy import stats
import json
import sys


def simulate_ttest(alpha=0.05, n1=30, n2=30,
                   effect_size=0.0, n_sim=10000, seed=42):
    """模拟两样本 t 检验的第一类错误率与功效。

    参数
    ----
    alpha        : 名义显著性水平
    n1, n2       : 两组样本量
    effect_size  : 0 表示 H0 为真；非 0 表示第二组均值偏移该量
    n_sim        : 重复次数，越大蒙特卡洛误差越小
    seed         : 随机种子，保证结果可复现
    """
    np.random.seed(seed)
    false_positive = 0
    true_positive = 0
    coverage_count = 0

    for _ in range(n_sim):
        if effect_size == 0:
            # H0 为真：两组来自同一正态分布
            x = np.random.normal(0, 1, n1)
            y = np.random.normal(0, 1, n2)
        else:
            # H1 为真：第二组有偏移
            x = np.random.normal(0, 1, n1)
            y = np.random.normal(effect_size, 1, n2)

        t_stat, p_value = stats.ttest_ind(x, y)

        # 记录第一类错误（仅 H0 为真时计数）
        if effect_size == 0 and p_value < alpha:
            false_positive += 1

        # 记录功效（仅 H1 为真时计数）
        if effect_size != 0 and p_value < alpha:
            true_positive += 1

        # 记录覆盖率：用 t 分布构造 1-alpha 置信区间
        # 注意：np.var 默认 ddof=0，与 ttest_ind 内部的合并方差口径略有差异
        se = np.sqrt(x.var() / n1 + y.var() / n2)
        df = n1 + n2 - 2
        t_crit = stats.t.ppf(1 - alpha / 2, df)
        diff = x.mean() - y.mean()
        ci_lower = diff - t_crit * se
        ci_upper = diff + t_crit * se
        if ci_lower <= effect_size <= ci_upper:
            coverage_count += 1

    results = {
        "simulated_alpha": false_positive / n_sim if effect_size == 0 else None,
        "simulated_power": true_positive / n_sim if effect_size != 0 else None,
        "coverage_rate": coverage_count / n_sim,
        "nominal_alpha": alpha,
        "n_sim": n_sim,
        "seed": seed,
        "environment": {
            "numpy": np.__version__,
            "scipy": stats.__name__,
        },
    }
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="ttest")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n1", type=int, default=30)
    parser.add_argument("--n2", type=int, default=30)
    parser.add_argument("--effect-size", type=float, default=0.0)
    parser.add_argument("--n-sim", type=int, default=10000)
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    results = simulate_ttest(
        alpha=args.alpha, n1=args.n1, n2=args.n2,
        effect_size=args.effect_size, n_sim=args.n_sim,
    )

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
```

本地测试命令（H0 真与 H1 真各跑一次，分别验证第一类错误率与功效）：

```bash
# 测试 H0 为真：估计第一类错误率是否接近名义 α
python scripts/run_simulation.py --alpha 0.05 --n1 30 --n2 30 --n-sim 10000
# 测试 H1 为真：估计功效是否接近理论功效
python scripts/run_simulation.py --alpha 0.05 --n1 30 --n2 30 --effect-size 0.5 --n-sim 10000
```

从零部署该 Skill 的完整流程见 7.15.1。

### 7.10.2 实验设计与功效分析 Skill

功效分析是实验设计阶段的核心环节，也是最容易因参数误设而失效的环节。下面把"给定期望效应量与功效，反解样本量；并绘制功效曲线"封装为 Skill。

````markdown
---
name: experimental-design-power
description: 计算样本量、绘制功效曲线并给出实验设计建议。适用于两样本均值比较、比例比较、单因素方差分析与重复测量设计的功效分析。Use when user asks for "样本量计算"、"power analysis"、"功效不足"、"effect size"、"需要多少样本"。
---

# 实验设计与功效分析

## 核心逻辑
功效是备择假设为真时拒绝原假设的概率
  power = P_{θ ∈ H1}(reject H0) = P_{θ ∈ H1}(p < α)
它由 α、效应量 d、样本量 n 与检验类型共同决定，四者知三求一。
本 skill 只做解析计算与蒙特卡洛校验，不替使用者选择效应量。

## 执行步骤

### 步骤 1：确认设计参数
从用户输入中提取并回述确认：
- 设计类型（两独立样本 / 配对 / 单因素 ANOVA / 比例）
- 名义 α（默认 0.05，双侧）
- 目标功效（默认 0.80）
- 效应量：优先用领域公认的最小有意义效应（MDE）；若使用者无法给出，
  用先导数据估计并明确标注为估计值

### 步骤 2：解析法求样本量
调用 `scripts/power_curve.py --design two-sample --alpha 0.05 --target-power 0.80 --effect-size d`
输出所需每组样本量 n，并向上取整到可执行的整数。

### 步骤 3：蒙特卡洛复核
用同一 n 与 d 跑 5000 次模拟，复核功效是否落在目标值 ±0.02 内。
若偏差超出，说明解析公式的假设（正态性、方差齐性）与模拟设定不符，需报告差异原因。

### 步骤 4：输出功效曲线
绘制 n 从 5 到 5×n_required 的功效曲线，标出目标功效线与当前方案位置。

### 步骤 5：给出设计建议
- 若所需 n 超出可行规模：报告在当前 n 下实际可达的功效，并讨论改用
  配对设计、协变量调整（ANCOVA）或更精确的测量以减小残差方差
- 涉及多重比较时，按比较次数 m 做 Bonferroni 或 BH 修正后重新计算样本量

## 输出格式
| 参数 | 取值 |
|------|------|
| 设计 | 两独立样本 t 检验，双侧 |
| α / 目标功效 | 0.05 / 0.80 |
| 效应量 d | 0.50（含来源说明） |
| 所需每组 n | 64 |
| 蒙特卡洛复核功效 | 0.796（5000 次，种子 20260101） |

## 禁止行为
- 不得在未确认效应量来源的情况下给出样本量数字
- 不得把事后功效（post-hoc power，基于观测效应量回算）作为设计依据
- 不得忽略多重比较修正
- 不得声称"功效 0.80 意味着有 80% 的概率 H1 为真"——这是对错误概率的误读

## Bundled files
- scripts/power_curve.py            # 样本量计算与功效曲线
- references/mde-checklist.md       # 最小有意义效应量确定清单
````

配套脚本（核心部分，依赖 `statsmodels`）：

```python
#!/usr/bin/env python3
"""功效分析：解析法求样本量 + 蒙特卡洛复核 + 功效曲线。"""
import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestIndPower, FTestAnovaPower


def required_n(d, alpha=0.05, power=0.80, ratio=1.0):
    """两独立样本 t 检验：给定 Cohen's d 反解每组样本量（解析解，向上取整）。"""
    analysis = TTestIndPower()
    n = analysis.solve_power(effect_size=d, alpha=alpha, power=power, ratio=ratio)
    return int(np.ceil(n))


def mc_power(d, n, alpha=0.05, n_sim=5000, seed=20260101):
    """蒙特卡洛复核：用模拟估计给定 n 与 d 下的实际功效。"""
    rng = np.random.default_rng(seed)          # 显式 RNG，避免全局种子污染
    reject = 0
    for _ in range(n_sim):
        x = rng.normal(0.0, 1.0, n)
        y = rng.normal(d, 1.0, n)              # 标准化设定下均值差即 d
        _, p = stats.ttest_ind(x, y)
        reject += (p < alpha)
    return reject / n_sim


def power_curve(d, ns, alpha=0.05, n_sim=2000, seed=20260101):
    """在一组候选样本量上计算功效，用于绘图。"""
    return [(n, mc_power(d, n, alpha, n_sim, seed)) for n in ns]
```

### 7.10.3 回归诊断报告 Skill

````markdown
---
name: regression-diagnostics
description: 对线性或广义线性模型执行完整假设诊断并生成报告，必要时给出稳健替代方案。适用于多元回归、ANCOVA、logistic 回归的残差诊断、共线性、异方差与影响点分析。Use when user asks for "回归诊断"、"残差分析"、"异方差"、"VIF"、"影响点"、"cook distance"。
---

# 回归诊断报告

## 核心逻辑
线性模型的高斯-马尔可夫性质与推断有效性依赖一组可诊断的假设：
线性性、误差独立、同方差、正态性、无强影响点、无完全共线性。
本 skill 逐项诊断，输出结论与替代方案，不直接修改研究者的模型设定。

## 执行步骤

### 步骤 1：拟合与基线输出
按使用者给出的公式拟合模型，记录系数、标准误、t 值与样本量。

### 步骤 2：逐项诊断（调用 scripts/diagnostics.py）
| 诊断项 | 方法 | 判定阈值 |
|--------|------|----------|
| 线性性 | 残差 vs 拟合值图 + 成分残差图 | 系统性弯曲需考虑变换或加入多项式项 |
| 异方差 | Breusch-Pagan 检验 | p < 0.05 提示存在异方差 |
| 误差正态性 | Q-Q 图 + Shapiro-Wilk | 样本量大时以图形为主，检验为辅 |
| 影响点 | Cook's D | D > 4/n 的点需逐一核查 |
| 共线性 | VIF | VIF > 10 需处理；> 5 需说明 |
| 独立性 | Durbin-Watson（时间序列）/ 聚类结构检查 | 明显偏离 2 需考虑聚类稳健标准误 |

### 步骤 3：给出替代方案
- 异方差存在：改用 HC3 异方差稳健标准误，或加权最小二乘
- 存在聚类结构：改用聚类稳健标准误，并说明聚类层级
- 强影响点：报告剔除后的系数变化，不默认剔除
- 共线性：报告方差分解比例，讨论是否删变量或改用主成分/岭回归

### 步骤 4：生成报告
按 `references/report-template.md` 输出，含：诊断表、关键图、结论与局限。

## 输出格式
| 诊断项 | 统计量 | p 值 | 结论 | 处理建议 |
|--------|--------|------|------|----------|
| Breusch-Pagan | 12.4 | 0.002 | 存在异方差 | 改用 HC3 稳健标准误 |
| VIF(X1) | 11.7 | — | 共线性偏高 | 考虑中心化或删变量 |

## 禁止行为
- 不得在未报告诊断结果的情况下直接给出最终系数解释
- 不得默认剔除强影响点，剔除须经使用者确认并在报告中并列展示两套结果
- 不得把 p 值大小作为效应量大小的证据
- 不得用逐步回归筛选变量后直接报告原假设检验的 p 值（存在选择性推断问题）

## Bundled files
- scripts/diagnostics.py          # 诊断统计量计算
- references/report-template.md   # 报告模板
- assets/fig-style.json           # 图形样式
````

### 7.10.4 保形预测区间 Skill

保形预测（conformal prediction）给出**有限样本、分布无关**的覆盖率保证，是不确定性量化中与统计学者最相关的现代工具之一。

````markdown
---
name: conformal-interval
description: 构造保形预测区间并校验其边际覆盖率。适用于回归预测的区间估计、分类的预测集合、split conformal 与 CV+ 变体的实现与验证。Use when user asks for "保形预测"、"conformal prediction"、"预测区间覆盖率"、"不确定性量化"、"prediction interval"。
---

# 保形预测区间

## 核心逻辑
Split conformal 的保证是分布无关的边际覆盖：在交换性（exchangeability）假设下，
  P(Y_{n+1} ∈ C(X_{n+1})) ≥ 1 − α
该保证对任意预测模型成立，且是有限样本的（不依赖渐近）。
代价是只保证边际覆盖，不保证条件覆盖——这一点必须在报告中说明。

## 执行步骤

### 步骤 1：划分数据
按 1:1 划分为训练集与校准集（小样本时改用 CV+，见步骤 4）。

### 步骤 2：拟合与计算 conformity score
在训练集拟合任意预测模型 f；在校准集计算分数 s_i = |y_i − f(x_i)|。

### 步骤 3：构造区间
取校准分数的 ⌈(n_cal + 1)(1 − α)⌉ / n_cal 分位数 q̂，
预测区间为 C(x) = [f(x) − q̂, f(x) + q̂]。

### 步骤 4：覆盖率校验（必须执行）
在独立测试集上估计经验覆盖率，要求落在 [1 − α − 0.02, 1 − α + 0.02]。
同步报告区间平均宽度——覆盖率可通过无限放宽区间达成，须与宽度联合报告。
若条件覆盖明显失衡（例如按 x 分组后各组覆盖率差异超过 5 个百分点），
改用 CV+ 或分层保形（Mondrian conformal），并在报告中说明。

### 步骤 5：输出报告
| 指标 | 取值 |
|------|------|
| 名义覆盖 1 − α | 0.90 |
| 测试集经验覆盖 | 0.903（n_test = 2000） |
| 平均区间宽度 | 3.42 |
| 最差分组覆盖 | 0.861（第 3 分位组） |

## 禁止行为
- 不得声称"每个个体的覆盖概率都是 1 − α"（边际覆盖不等于条件覆盖）
- 不得在数据不满足交换性（如强时间依赖）时直接套用，须说明假设违背
- 不得在测试集上调 α 后再报告覆盖率（此时覆盖率已被污染）
- 不得省略区间宽度只报覆盖率

## Bundled files
- scripts/conformal.py            # split conformal 与 CV+ 实现
- references/coverage-notes.md    # 边际覆盖与条件覆盖的说明
````

配套脚本核心部分：

```python
#!/usr/bin/env python3
"""Split conformal 与覆盖率校验。"""
import numpy as np


def split_conformal(f, X_cal, y_cal, alpha=0.1):
    """返回校准分位数 q̂：使区间 [f(x)-q̂, f(x)+q̂] 达到 1-alpha 边际覆盖。"""
    scores = np.abs(y_cal - f.predict(X_cal))            # conformity score
    n_cal = len(scores)
    level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal   # 有限样本修正
    return np.quantile(scores, level, method="higher")


def evaluate(f, qhat, X_test, y_test, alpha=0.1):
    """在独立测试集上评估经验覆盖率与平均宽度。"""
    pred = f.predict(X_test)
    lower, upper = pred - qhat, pred + qhat
    covered = (y_test >= lower) & (y_test <= upper)
    return {
        "nominal": 1 - alpha,
        "empirical_coverage": float(covered.mean()),
        "mean_width": float((upper - lower).mean()),
        "n_test": len(y_test),
    }
```

### 7.10.5 论文出版图 Skill

````markdown
---
name: publication-figure
description: 生成符合期刊规范的统计图表。适用于 matplotlib/plotly 绘图、论文图表格式化、Nature/Science 风格图表。Use when user asks for "论文图"、"publication figure"、"期刊图表"。
---

# 出版级统计图表

## 执行步骤
1. 确认目标期刊与图表类型（箱线图 / 森林图 / 生存曲线 / 热图 / 散点矩阵）
2. 读取 assets/fig-style.json 中的 rcParams（字号、线宽、DPI、色序）
3. 调用 scripts/make_figure.py 出图，矢量与位图各出一份
4. 按 references/journal-specs.md 核对：单位、误差棒定义（SD/SE/CI 必须注明）、
   色盲友好配色、字体嵌入
5. 输出图与可复现的绘图脚本

## 输出要求
- 误差棒须在图注中说明含义（SD、SE 或 95% CI）
- 不使用仅靠颜色区分的分组编码，须叠加形状或线型
- 分辨率不低于 300 dpi，矢量版保存为 PDF/SVG

## 禁止行为
- 不得截断 y 轴以夸大组间差异
- 不得在无样本量说明的情况下绘制箱线图
- 不得省略图注中的统计方法说明

## Bundled files
- scripts/make_figure.py           # 调用 matplotlib，读 assets/fig-style.json
- references/journal-specs.md      # Nature/Science/PNAS 图表规范
- assets/fig-style.json            # rcParams 配置（字号/线宽/DPI）
````

### 7.10.6 R 分析流水线 Skill

````markdown
---
name: r-analysis-pipeline
description: 执行标准化 R 统计分析流程。适用于回归分析、方差分析、混合效应模型、生存分析。Use when user asks for "R分析"、"回归"、"mixed model"、"生存分析"。
---

# R 分析流水线

## 执行步骤
1. 读取数据与变量字典，确认变量类型与缺失编码
2. 描述性统计与缺失模式诊断
3. 按研究问题拟合模型（lm / glm / lme4::lmer / survival::coxph）
4. 诊断（见 regression-diagnostics skill）
5. 输出结果表与图形，写入 results/
6. 记录 sessionInfo() 与 renv.lock，保证可复现

## 输出要求
- 结果表用 broom::tidy 输出，不手工抄录系数
- 随机种子在脚本顶部固定
- 每次运行追加一条运行日志（时间、R 版本、包版本、输入文件哈希）

## 禁止行为
- 不得在未固定种子的情形下报告涉及随机性的结果
- 不得直接覆盖 results/ 下已有产出，须带时间戳另存

## Bundled files
- scripts/run_analysis.R          # Rscript 执行
- references/r-conventions.md     # 项目 R 代码规范
- assets/session-info.txt         # 环境记录模板
````

### 7.10.7 文献管理与自动综述 Skill

本模板取自跨平台通用的 Skill 实现，输入主题即可完成 arXiv 检索、相关性过滤与结构化摘要，适用于课题组的定期文献追踪。

````markdown
---
name: lit-review
description: "Automated literature review on arXiv. Trigger keywords: review papers, literature update, 搜论文, 文献综述, recent papers on"
---

# Skill: Automated Literature Review

## 何时使用
当用户请求以下类型任务时激活：
- "review recent papers on [topic]"
- "literature update about [subject]"
- "搜索最近关于[主题]的论文"
- "帮我跟踪 [领域] 的新论文"

## 执行流程

### 第 1 步：解析用户输入
- 提取研究主题关键词
- 确定时间范围（默认最近 7 天）

### 第 2 步：调用 arXiv API
使用 `scripts/search_arxiv.py`，查询参数：
- search_query: `all:"[topic]"`
- sortBy: submittedDate
- sortOrder: descending
- max_results: 20

### 第 3 步：相关性过滤
- 根据 references/relevance_rules.md 中的关键词权重表打分
- 保留得分 > 0.6 的论文

### 第 4 步：生成结构化输出
按以下模板输出（保存到 literature/ 目录）：

```markdown
# Literature Update: [Topic] — [Date]
## New Papers Found: [N]
### 1. [Paper Title]
- **Authors**: [作者列表，最多 3 人 + "et al."]
- **Date**: YYYY-MM-DD
- **Relevance**: High / Medium / Low
- **Key contribution**: [基于摘要的 1-2 句总结]
- **Link**: [arXiv URL]
```

### 第 5 步：更新索引
将新条目追加到 literature/README.md 的索引列表中

## 注意事项
- 摘要必须忠实于原文 abstract，不得添加推测
- API 失败时重试 2 次，仍失败则报告错误
- 每次最多返回 10 篇高相关论文，避免信息过载

## 使用示例
- "review recent papers on central limit theorem from last week"
- "literature update on AI4Math, only high relevance"
- "搜索最近三天关于 statistical learning theory 的论文"
````

配套脚本 `scripts/search_arxiv.py`：

```python
#!/usr/bin/env python3
"""arXiv 最新论文检索：按主题查询并返回结构化条目。"""
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET


def search_arxiv(topic, max_results=20):
    """调用 arXiv Atom API 检索按提交时间倒序排列的论文。"""
    base_url = "http://export.arxiv.org/api/query?"
    query = (f'search_query=all:"{topic}"'
             f'&sortBy=submittedDate&sortOrder=descending&max_results={max_results}')
    url = base_url + urllib.parse.quote(query)
    with urllib.request.urlopen(url) as response:      # 生产环境需加超时与重试
        xml_data = response.read().decode("utf-8")

    root = ET.fromstring(xml_data)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers = []
    for entry in root.findall("atom:entry", ns):
        papers.append({
            "title": entry.find("atom:title", ns).text.strip(),
            "authors": [a.find("atom:name", ns).text
                        for a in entry.findall("atom:author", ns)],
            "abstract": entry.find("atom:summary", ns).text.strip(),
            "date": entry.find("atom:published", ns).text[:10],
            "link": entry.find("atom:id", ns).text,
        })
    return papers


if __name__ == "__main__":
    import sys
    topic = sys.argv[1] if len(sys.argv) > 1 else "machine learning"
    for p in search_arxiv(topic):
        print(f"[{p['date']}] {p['title']}")
        print(f"  Authors: {', '.join(p['authors'][:3])}")
        print(f"  Link: {p['link']}")
        print()
```

### 7.10.8 文件批量重命名 Skill（简版）

用于文献整理场景，结构为 `file-batch-rename/{SKILL.md, scripts/rename_pdfs.py, references/naming-convention.md}`：

````markdown
---
name: file-batch-rename
description: 批量重命名学术 PDF 文件。适用于文献整理、文件重命名、PDF 管理。Use when user asks for "重命名"、"整理文献"、"batch rename"。
---

# 文件批量重命名

## Steps
1. 扫描指定目录所有 PDF 文件
2. 提取作者-年份-标题信息
3. 按 references/naming-convention.md 的规范重命名
4. 返回重命名前后对照表，并落盘为 rename_log.csv
5. 默认只做预览，经确认后才执行写操作
````

调用方式：在智能体中说明目标目录与命名格式，智能体触发该 Skill 并调用 `scripts/rename_pdfs.py` 完成本地文件操作。

## 7.11 Skill本地部署

### 7.11.1 安装位置与优先级

Skill 可以安装在多个位置，同名 Skill 按下列优先级生效（高优先级覆盖低优先级）：

```
enterprise（组织策略下发）
  > personal（~/.claude/skills/，个人全局）
    > project（.claude/skills/，随 git 共享给全组）
      > nested（子目录级）
        > plugin（插件市场安装）
          > claude.ai synced（云端同步）
```

对课题组的实践建议是：**通用流程放个人级，本课题特有的流程放项目级**。项目级 Skill 随仓库提交，成员 clone 即得，这使 Skill 天然带有版本历史与评审记录。

### 7.11.2 方式一：GitHub CLI 一键安装（推荐）

GitHub CLI 自 2.x 版本起原生支持 `gh skill` 子命令，可在一行命令内完成安装、搜索与管理。该命令自动识别本机已安装的智能体（Claude Code、Codex CLI、Cursor、Gemini CLI、Windsurf、GitHub Copilot）并写入对应配置路径，无需手工判断目录。素材记载该能力于 2026 年 4 月进入 GitHub CLI。

```bash
# 安装 GitHub CLI（macOS 用 Homebrew；Ubuntu/Debian 用 sudo apt install gh）
brew install gh
gh --version

# 语法：gh skill install <repository> [<skill>[@version]]
gh skill install anthropics/skills                   # 官方技能仓库
gh skill install anthropics/skills document-skills    # 仅安装文档类子技能
gh skill install K-Dense-AI/scientific-agent-skills   # 科研专用技能库

# 检索与维护
gh skill search "data analysis"      # 在 GitHub 上搜索可用 skill
gh skill list                        # 列出本机已安装
gh skill update anthropics/skills    # 更新已安装
```

上述命令集是本章唯一推荐的安装入口；手动复制（7.11.3）与插件市场（7.11.4）为补充方式，三者的效果等价，选择其一即可。

### 7.11.3 方式二：手动复制或克隆

适用于离线环境、私有 Skill 或需要 review 后再安装的场合：

```bash
# Claude Code：个人全局（所有项目可用）
mkdir -p ~/.claude/skills/monte-carlo-hypothesis-test
cp -r ./monte-carlo-hypothesis-test/* ~/.claude/skills/

# Claude Code：项目级（随 git 共享给全组）
mkdir -p .claude/skills/monte-carlo-hypothesis-test
cp -r ./monte-carlo-hypothesis-test/* .claude/skills/

# Codex CLI：创建目录后克隆社区技能库（推荐只复制需要的单个技能文件夹，避免加载过多）
mkdir -p ~/.codex/skills
git clone https://github.com/composio-community/awesome-codex-skills.git
```

Windows 下用 PowerShell 创建 Codex 技能目录：

```powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.codex\skills"
```

安装后重启会话即可使用：Skill 会出现在斜杠命令列表中，智能体也可依据 SKILL.md 的 `description` 自动路由。启动时只加载每个技能的名称与描述，任务匹配时再读取全文，因此同时安装数十个技能不会撑爆上下文。

### 7.11.4 方式三：插件市场（Claude Code 内）

```text
/plugin marketplace add anthropics/skills
/plugin install document-skills@anthropic-agent-skills
/plugin install example-skills@anthropic-agent-skills
```

安装后可直接提出"Use the PDF skill to extract the form fields from path/to/some-file.pdf"触发对应技能。`skill-creator` 元技能也通过该市场获取：安装后描述所需流程，即可自动生成 SKILL.md 并做审查迭代。

### 7.11.5 依赖管理：uvx 零污染部署

Skill 常依赖第三方包，直接 `pip install` 会污染宿主环境并引入版本冲突。配合 uv 包管理器，可在 SKILL.md 中指示模型执行 `uvx --from git+... <tool>`，由包管理器创建阅后即焚的临时虚拟环境安装依赖，运行结束即销毁：

```bash
# 安装 uv（Astral 提供的 Python 包与环境管理器）
curl -LsSf https://astral.sh/uv/install.sh | sh

# SKILL.md 中可写入的指令示例（无需预先安装，临时环境自动创建）
uvx --from git+https://github.com/user/repo tool-name --arg value
```

对 R 生态没有等价的 `uvx`，实践中用 `renv::restore()` 固定包版本，或把 R 脚本运行环境容器化（环境锁与容器化的具体做法详见第 8.14.2 节）。

### 7.11.6 平台差异：Codex 与 WorkBuddy

Codex（OpenAI）与 WorkBuddy（腾讯）遵循同一开放标准，因此**一份 Skill 可在两个平台间通用**：把文件夹复制到对应目录即可。两者差异如下：

| 维度 | Codex CLI | WorkBuddy |
|------|-----------|-----------|
| 标准格式 | SKILL.md（开放标准） | SKILL.md（同一开放标准） |
| 全局技能目录 | `~/.codex/skills/` | `~/.workbuddy/skills/` |
| 项目级目录 | `./.codex/skills/`（仅当前项目可用） | 不区分项目，全局生效 |
| 开源技能来源 | awesome-codex-skills（Composio 维护） | 技能市场（内置 20+ 技能）+ 第三方 SKILL.md 包 |
| 创建方式 | 手动写文件（mkdir + 编辑器） | 对话框自然语言创建 |
| 触发方式 | 自然语言匹配 description 中的关键词 | 同上，支持 `@skill:name` 显式调用 |

WorkBuddy 的内置技能市场包含网页搜索、浏览器自动化、办公文档编辑、图表绘制、邮件处理等 20 余种技能包，图形界面中浏览后一键安装即可用自然语言触发。

自然语言创建的流程为：在对话框描述所需能力（例如"上传 CSV 实验数据后，自动检查缺失值并用中位数插补、按正态性选择 t 检验或 Mann-Whitney U 检验、生成描述性统计表与合适图表、整合为 Word 报告"），平台解析意图、生成含 YAML 元数据的 SKILL.md 并保存到 `~/.workbuddy/skills/`，随后可用自然语言触发。平台生成的 frontmatter 含 `agent_created: true` 标记，用于标识该技能可被 AI 自动更新：

```yaml
---
name: my-skill-name
description: "一句话描述，包含触发关键词。系统通过这里的文字判断何时自动激活"
agent_created: true  # 标记为 AI 创建，允许后续自动更新
---
```

导入本地技能包时，在技能界面选择上传并拖拽文件夹即可完成配置，无需额外操作；这意味着在 Codex 中创建的 `lit-review` 技能可直接打包上传，两个平台共用同一份技能定义。

### 7.11.7 验证与排查

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| `gh skill` 命令不存在 | GitHub CLI 版本过旧 | `brew upgrade gh`（macOS）或从 cli.github.com 安装最新版 |
| Skill 未被触发 | description 中的关键词与实际输入不匹配 | 在 description 中写入真实触发短语，同时回答 WHAT 与 WHEN |
| 技能触发但执行错误 | 指令模糊或缺少关键步骤 | 在执行流程中增加明确的步骤编号与输出格式；末尾要求输出执行日志 |
| 本地文件无法读写 | 缺少 filesystem MCP 服务器 | 在 settings.json 中配置 filesystem MCP |
| 执行报错：找不到模块 | Python 依赖缺失 | 用 `uvx --from git+...` 自动创建临时虚拟环境 |
| 多 Skill 同名冲突 | 项目级覆盖个人级 | 按 7.11.1 的优先级检查，必要时关闭暂不使用的技能 |
| Skill 执行慢 | 确定性工作未脚本化 | 把计算、重命名、API 调用全部脚本化，避免模型即兴生成代码 |
| 想强制使用某个 Skill | 自然语言触发不稳定 | 使用显式调用 `@skill:name` 或在对话中提到技能名称 |
| 第三方 Skill 有安全风险 | 可能存在恶意提示词注入 | 优先使用官方推荐 Skill；安装前检查 scripts 内容 |

诊断入口：Claude Code 内置 `/doctor` 命令用于诊断环境；输入斜杠可查看当前已加载的技能列表。在 SKILL.md 执行流程末尾加一句"完成后输出执行日志，说明每一步做了什么"，可显著降低追踪成本。

## 7.12 开源Skill与MCP资源

### 7.12.1 官方与通用

| 仓库 | 显式 URL | 内容 | 许可证 |
|------|---------|------|--------|
| anthropics/skills | github.com/anthropics/skills | 官方标准仓库：创意/技术/企业类示例 + docx/pdf/pptx/xlsx 四大文档技能 | Apache 2.0（文档类 source-available） |
| Agent Skills 规范 | agentskills.io | 开放标准规范文档 | — |
| Claude Code Skills 文档 | code.claude.com/docs/en/skills | 官方使用指南 | — |
| Anthropic 完整构建指南 PDF | resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf | 官方权威教程：规划/设计/测试/分发 | — |

### 7.12.2 科研专用

| 仓库 | 显式 URL | 内容 | 价值 |
|------|---------|------|------|
| K-Dense-AI/scientific-agent-skills | github.com/K-Dense-AI/scientific-agent-skills | "Turn any AI agent into an AI Scientist"——数据科学家、AI 研究人员专用技能库，支持 MCP 方式接入 | 统计学科研团队首选 |
| K-Dense-AI 索引平台 | skillsovermcp.com | 193+ 科研技能索引，每个 SKILL.md 作为 MCP 工具自动加载 | 发现新 skill |
| agenticskills.io | agenticskills.io | 跨平台 skill 目录，支持 Claude Code/Codex/Cursor/Gemini CLI | 193+ 技能，16 类别 |
| skillsovermcp/self-host | github.com/skillsovermcp | 将任意 GitHub 仓库的 skills 自托管为 MCP 服务器 | 内网部署 |
| K-Dense Web | k-dense.ai | K-Dense 科研云平台，提供云端计算与更多 agent | 计算密集型任务 |

### 7.12.3 社区精选

| 仓库 | 显式 URL | 内容 |
|------|---------|------|
| ComposioHQ/awesome-claude-skills | github.com/ComposioHQ/awesome-claude-skills | 覆盖开发、数据分析、商业应用的导航清单 |
| travisvn/awesome-claude-skills | github.com/travisvn/awesome-claude-skills | 精选型，聚焦 Claude Code 工作流 |
| BehiSecc/awesome-claude-skills | github.com/BehiSecc/awesome-claude-skills | 分类清晰：文档处理/开发工具/数据分析 |
| vercel-labs/agent-skills | github.com/vercel-labs/agent-skills | Next.js/React 前端开发 |
| trailofbits/skills | github.com/trailofbits/skills | 安全审计、代码供应链安全 |
| OthmanAdi/planning-with-files | github.com/OthmanAdi/planning-with-files | 项目规划、方案设计 |
| hesreallyhim/awesome-claude-code | github.com/hesreallyhim/awesome-claude-code | Claude Code 工作流、slash-commands、CLAUDE.md 集合 |
| VoltAgent/awesome-claude-code-subagents | github.com/VoltAgent/awesome-claude-code-subagents | 100+ 专用子 agent |
| awesomeclaude.ai | awesomeclaude.ai | 在线目录：204+ Claude Agent Skills，13 类别 |

### 7.12.4 MCP 服务器生态

Skill 解决"怎么做"，MCP 解决"连什么"。下表列出与科研工作流关系最密切的服务器：

| MCP 服务器 | 获取方式 | 科研用途 |
|---------|---------|---------|
| GitHub MCP | github.com/github/github-mcp-server | 代码仓库操作、PR 管理 |
| Notion MCP | notion.so/mcp | 文档知识库读写 |
| 文件系统 MCP | modelcontextprotocol 内置（server-filesystem） | 本地数据读写 |
| Jupyter MCP | github.com/modelcontextprotocol | 代码执行环境 |
| Zotero MCP | zotero-mcp-server（npm 包） | 文献管理 |
| MCP 市场排行榜 | mcpmarket.com | Top 100 MCP 服务器排名 |
| punkpeye/awesome-mcp-servers | github.com/punkpeye/awesome-mcp-servers | 社区最全 MCP 服务器列表 |
| mcpservers.org | mcpservers.org | MCP 服务器目录与评测 |

### 7.12.5 选择第三方 Skill 的三条检查

第一，检查 `scripts/` 目录内容：任何网络请求、删除操作、凭据读取都应在安装前被人工阅读。第二，检查 `description` 是否包含诱导性指令（例如要求读取某个远端 URL 并执行返回内容），这是提示词注入的常见载体。第三，优先使用有明确维护记录与许可证的仓库；许可证不明的 Skill 不应进入课题组的项目级目录。

## 7.13 动态工具检索

### 7.13.1 问题的形式化

当技能库包含数百乃至上千个 Skill 时，把所有 Schema 放进提示词会导致上下文爆炸（7.8.2 已给出算术）。解决方案是把工具选择本身变成一次检索问题。

设工具库 $\mathcal{T} = \{t_1, \ldots, t_N\}$，每个工具 $t_i$ 拥有自然语言描述 $d_i$（SKILL.md 的 description）与结构化签名 $s_i$（JSON Schema）。设文本编码器 $E: \text{text} \mapsto \mathbb{R}^{p}$ 把描述映射为单位向量，则工具的向量表示为 $e_i = E(d_i)$，索引即矩阵 $M = [e_1, \ldots, e_N]^\top \in \mathbb{R}^{N \times p}$。

给定当前子任务的查询 $q$（可以是用户原话，也可以是智能体生成的一句话子任务描述），检索阶段返回

$$S_k(q) \;=\; \underset{A \subset \mathcal{T},\, |A| = k}{\arg\max}\; \sum_{t_i \in A} \cos\big(E(q), E(t_i)\big), \qquad \cos(u, v) = \frac{u^\top v}{\|u\|\,\|v\|}$$

即取与查询余弦相似度最高的 $k$ 个工具。注入阶段把这 $k$ 个工具的 $(d_i, s_i)$ 序列化后追加到上下文 $\mathcal{C}$，模型再从 $\mathcal{C}$ 中选择并生成调用：

$$\text{action} \sim \pi_\theta\big(a \mid \mathcal{C} \oplus \{(d_i, s_i)\}_{t_i \in S_k(q)}\big)$$

这就是**两阶段调度**：先检索（检索器决定候选集），后选择（模型决定用哪个）。它与第 10 章讨论的 Lean 定理检索（Loogle、LeanSearch）以及向量数据库驱动的 RAG 完全同构——三者的差别只在被检索对象分别是工具、定理与文本片段。

### 7.13.2 两阶段流程与误差来源

```
[用户科研需求 Prompt]
        │
        ▼
┌───────────────────────────────┐
│ 阶段一：向量数据库检索          │◄── 存储 1000+ Skill 的 Embedding
│   q → E(q) → cos(E(q), E(tᵢ)) │
└───────────────┬───────────────┘
                │ 召回 Top-K 最相关的 Skill Schema
                ▼
┌───────────────────────────────┐
│ 阶段二：注入当前 Agent 上下文   │──► [LLM 识别并生成结构化调用指令]
└───────────────────────────────┘
```

误差有两个来源。**检索误差**：真实所需工具 $t^*$ 不在 $S_k(q)$ 中，记为事件 $\bar R$，概率 $\varepsilon = P(\bar R)$，对应指标为 $1 - \text{Recall}@k$。**选择误差**：$t^*$ 已在候选集中，但模型选了别的工具（$1 - p_s$）或参数填错（$1 - p_e$）。二者性质不同：前者可以通过改进编码器、扩充描述、增大 $k$ 来降低；后者取决于提示质量与 Schema 清晰度。

### 7.13.3 检索错误率如何传导到任务成功率

设 $p_s = P(\text{选中正确工具} \mid R)$，$p_e = P(\text{参数与执行正确} \mid \text{选中正确})$，$p_f = P(\text{任务成功} \mid \bar R)$。后者不为零是因为智能体还有备选路径：改写查询重新检索、请求使用者澄清、用通用工具（例如直接写一段 Python）兜底。按全概率公式展开：

$$P(\text{成功}) \;=\; (1 - \varepsilon)\, p_s p_e \;+\; \varepsilon\, p_f$$

对 $\varepsilon$ 求导：

$$\frac{\partial P(\text{成功})}{\partial \varepsilon} \;=\; p_f - p_s p_e \;<\; 0$$

因为备选路径的成功率 $p_f$ 远低于正常路径 $p_s p_e$，检索错误率每上升 1 个百分点，端到端成功率约损失 $p_s p_e$ 个百分点。取 $p_s = 0.90$、$p_e = 0.85$、$p_f = 0.10$：

| 检索错误率 $\varepsilon$ | 单步成功率 | 说明 |
|---|---|---|
| 0.00 | 0.765 | 检索器理想 |
| 0.05 | 0.732 | Recall@10 ≈ 0.95 的常见水平 |
| 0.10 | 0.699 | 描述含糊、语义重叠较多 |
| 0.20 | 0.632 | 工具库无治理时的典型区间 |

$\varepsilon$ 从 0.05 升到 0.20，单步成功率下降约 0.10；边际约为每 1 个百分点检索错误换取 0.67 个百分点的端到端损失。改进检索器的收益因此可以直接折算成任务成功率，这使得"优化 description 措辞"这类看似琐碎的工作有了可度量的回报。

若一次任务包含 $L$ 个需要检索工具的步骤，且各步近似独立，则

$$P(\text{整条轨迹成功}) \;\approx\; \big[(1 - \varepsilon) p_s p_e + \varepsilon p_f\big]^{L}$$

取上表的单步值 0.732、$L = 5$，得 0.21。这个数字看起来过于悲观，原因在于独立性与同分布两条假设都不成立：后面的步骤往往复用前面已加载的工具（正相关的成功事件），且困难步骤集中在少数几步。但公式传达的方向性结论是可靠的：**长轨迹对单步可靠性极其敏感**，把 $L$ 从 5 降到 3（例如预先把工具集固定下来），收益远大于把单步可靠性从 0.73 提到 0.78。

### 7.13.4 与多重检验、选择性推断的相似性

上述分解与统计学中的**多重检验**（multiple testing）问题结构相同。把每一步检索看作一次假设检验：

$$H_{0i}:\ t^* \notin S_k(q_i) \quad \text{（该工具与当前子任务无关）}$$

检索错误率 $\varepsilon$ 相当于每次检验的第一类错误率，$L$ 步构成 $L$ 次同时检验。在不做修正的情形下，整条轨迹出现至少一次检索错误的概率（族系错误率 FWER）满足

$$\text{FWER} \;=\; 1 - (1 - \varepsilon)^{L} \;\le\; L\varepsilon$$

右侧正是 Bonferroni 型的一阶上界。若要求 $\text{FWER} \le \delta$，Bonferroni 式修正给出 $\varepsilon \le \delta / L$：轨迹越长，单步检索必须越准。代价是对称于统计检验中的功效损失——把 $\varepsilon$ 压得越低，检索器越保守，越倾向返回那些"描述宽泛、看起来总是对的"的通用工具，等效于降低 Recall 的多样性，与 Bonferroni 牺牲功效换取 FWER 控制完全对应。

若可以容忍少量错选（因为错选后智能体能自行修复），则更贴近 **Benjamini–Hochberg 式的 FDR 控制**：允许错误发现占一个可控比例，用修复机制兜底，换取更高的召回。工程上的对应做法是设定"允许 $m$ 次重检索"的预算，并监控实际错选比例的估计值 $\widehat{\text{FDR}}$。

更深一层的相似性是**选择性推断**（selective inference / post-selection inference）。注入上下文的工具集 $S_k(q)$ 是依赖于查询 $q$ 的数据依赖选择结果，而后续的分析结论又是在这个被选中的工具集上产生的。这等同于在模型选择之后继续做推断：若忽略选择过程，报告的不确定性会系统性偏乐观（winner's curse）。三条实践含义：

1. **评估必须按任务族报告**，而不是挑选几条表现好的任务汇报成功率——后者与只报告显著的 p 值同源。
2. **评估时必须冻结检索器版本与索引快照**，否则 $S_k(\cdot)$ 随时间变化，两次评估结果不可比。
3. **在结论中报告选择过程**，如同论文中报告变量筛选方法；若工具的选取由查询语义决定，则该语义本身是研究设计的一部分。

下表给出对照：

| 动态工具检索 | 多重检验 / 选择性推断 | 共同结构 |
|--------------|----------------------|----------|
| 单个检索步 | 单次假设检验 | 二分类决策，有错误概率 $\varepsilon$ |
| $L$ 步轨迹 | 同时检验 $L$ 个假设 | 错误累积 $1 - (1-\varepsilon)^L$ |
| FWER ≤ δ 要求 | Bonferroni 修正 $\alpha/m$ | 单步阈值需按总数收紧 $\varepsilon \le \delta/L$ |
| 允许可修复的错选 | BH 的 FDR 控制 | 容忍一定比例错误，换取召回 |
| 注入上下文的 $S_k(q)$ | 数据依赖的模型选择 | 事后推断需校正选择效应 |
| Recall@k、首错位置 | FWER、FDR、功效 | 需要区分"错误率"与"漏检率"两类指标 |

**模拟：检索假阳性的规模与 BH 校正的代价。** 上表的对应关系可以用蒙特卡洛模拟直接验证。设定工具库有 $m = 100$ 个候选，其中 $m_1 = 5$ 个与当前子任务真正相关（$m_1 \ll m$ 是大规模技能库的常态）；检索器给真相关工具的得分 $z \sim N(\mu, 1)$、给无关工具的得分 $z \sim N(0, 1)$，$\mu$ 度量描述与查询的语义可分性；取单侧 p 值后比较两种注入规则：不校正（$p < 0.05$ 即注入）与 Benjamini–Hochberg 步进法（step-up procedure，FDR 水平 $q = 0.05$）。

```python
# tool_retrieval_fdr.py：工具检索假阳性与 BH-FDR 控制的模拟
import numpy as np
from scipy import stats

M, M1, ALPHA, Q, N_SIM, SEED = 100, 5, 0.05, 0.05, 5000, 20260101


def bh_select(pvals, q):
    """Benjamini-Hochberg 步进法：返回被判为相关的工具掩码。"""
    order = np.argsort(pvals)
    thresh = q * np.arange(1, len(pvals) + 1) / len(pvals)
    passed = np.where(pvals[order] <= thresh)[0]
    sel = np.zeros(len(pvals), dtype=bool)
    if passed.size:
        sel[order[: passed[-1] + 1]] = True
    return sel


def run_regime(mu):
    """在给定效应量 mu 下，估计两种注入规则的召回与假阳性。"""
    rng = np.random.default_rng(SEED)
    true_mask = np.array([True] * M1 + [False] * (M - M1))
    rows = {"A 不校正": [], "B BH(q=0.05)": []}
    for _ in range(N_SIM):
        z = rng.normal(0.0, 1.0, M)
        z[true_mask] += mu
        pvals = 1.0 - stats.norm.cdf(z)          # 单侧 p 值
        for rule, sel in [("A 不校正", pvals < ALPHA),
                          ("B BH(q=0.05)", bh_select(pvals, Q))]:
            V = int((sel & ~true_mask).sum())    # 假阳性：无关工具被注入
            T = int((sel & true_mask).sum())     # 选真：真相关工具被注入
            rows[rule].append((sel.sum(), T, V))
    print(f"mu={mu}：不校正时至少注入一个无关工具的概率（解析）"
          f"= {1 - (1 - ALPHA) ** (M - M1):.4f}")
    for rule, lst in rows.items():
        arr = np.array(lst, dtype=float)
        fdr = (arr[:, 2] / np.maximum(arr[:, 0], 1)).mean()
        miss = 1 - arr[:, 1].mean() / M1
        print(f"  {rule}: 选中 R={arr[:, 0].mean():.2f}  "
              f"选真={arr[:, 1].mean():.2f}  假阳性 V={arr[:, 2].mean():.2f}  "
              f"经验FDR={fdr:.3f}  真工具漏检率={miss:.3f}")


for mu in [3.0, 4.0]:
    run_regime(mu)
```

实测输出（Python 3.10.20，numpy 1.26.4，scipy 1.15.3；seed=20260101，n_sim=5000）：

```text
mu=3.0：不校正时至少注入一个无关工具的概率（解析）= 0.9923
  A 不校正: 选中 R=9.37  选真=4.55  假阳性 V=4.81  经验FDR=0.490  真工具漏检率=0.089
  B BH(q=0.05): 选中 R=2.76  选真=2.57  假阳性 V=0.19  经验FDR=0.047  真工具漏检率=0.486
mu=4.0：不校正时至少注入一个无关工具的概率（解析）= 0.9923
  A 不校正: 选中 R=9.76  选真=4.95  假阳性 V=4.81  经验FDR=0.468  真工具漏检率=0.010
  B BH(q=0.05): 选中 R=4.65  选真=4.39  假阳性 V=0.27  经验FDR=0.046  真工具漏检率=0.123
```

三个结果值得逐一解读。第一，不校正的后果比直觉严重：95 个无关工具按 0.05 的单步阈值筛选，几乎必然（0.992）至少注入一个无关工具，注入集合中约一半是假阳性（FDR ≈ 0.49）。被注入的无关工具也不是无害的占位符——模型需要在更多候选中做选择，描述之间的语义重叠会推高选错工具与填错参数的概率。第二，BH 校正把 FDR 压回 0.047（理论界为 $\pi_0 q \le 0.05$），代价是 $\mu = 3$ 时近一半真相关工具被漏掉（漏检率 0.486）：校正的功效损失在此显形，与 Bonferroni 一节的定性讨论完全一致。第三，对比 $\mu = 4$ 的两行：不校正的假阳性数没有变化（$V = 4.81$，只由 $\pi_0 \alpha$ 决定），而 BH 的漏检率从 0.486 降到 0.123，FDR 几乎不动。效应量——即 description 与查询的语义可分性——才是决定"校正是否可承受"的量。工程上的对应做法是 7.13.6 的"改写 description（加入反例与触发短语）"：把预算花在提高 $\mu$ 上，比在阈值上做文章更根本，这与统计设计中"提高效应量或降低噪声优于放宽检验水平"的常识同构。

需要说明该类比与主流实现的差异：检索器通常取固定的 top-$k$ 而不是分数阈值，此时不存在显式的检验水平，$k$ 个候选中"真工具不在其中"对应漏检而非假阳性；FDR 类比最贴合的是"按校准分数决定注入哪些工具"的实现（7.13.5 讨论的分数校准正是这一实现的前提）。但无论选择规则为何，"候选集是数据依赖的选择结果"这一事实不变，本节关于选择性推断的三条实践含义（按任务族报告、冻结检索器版本、报告选择过程）不受影响。

### 7.13.5 检索器的评价与校准

检索器本身应当被当作一个待估计的对象来评测，而不是靠直觉调参。构建评测集：收集 $n$ 条子任务查询，人工标注每条的真实工具 $t^*$（允许多标签），然后计算

$$\widehat{\text{Recall}@k} \;=\; \frac{1}{n}\sum_{j=1}^{n} \mathbb{1}\{t^*_j \in S_k(q_j)\}, \qquad \hat\varepsilon = 1 - \widehat{\text{Recall}@k}$$

这是二项比例的估计，标准误 $\sqrt{\hat\varepsilon(1 - \hat\varepsilon)/n}$；$n = 200$、$\hat\varepsilon = 0.05$ 时标准误约 0.015，即 95% 置信区间约为 ±3 个百分点。因此基于几十条查询的"优化有效"结论通常站不住。

还需注意**校准**（calibration）：余弦相似度不是概率。若系统要根据相似度决定是否触发重检索，应使用经过校准的分数（例如在留出集上做 isotonic 回归或 Platt 标定），否则阈值不具备概率含义。这一点与分类模型中的概率校准完全相同。

### 7.13.6 工程缓解手段

| 手段 | 作用 | 代价 |
|------|------|------|
| 增大 $k$ | 直接提升 Recall@k | 上下文占用线性上升，选择误差可能上升 |
| 改写 description（加入反例与触发短语） | 改善向量表示的可分性 | 需人工维护 |
| 层次检索（先选类别再选工具） | 把 $N$ 路检索降为两级小规模检索 | 需要维护类别体系 |
| 允许智能体请求"列出全部工具" | 给检索失败提供兜底 | 一次性上下文开销大 |
| 查询改写后重检索 | 把 $p_f$ 提高到接近正常路径 | 增加一轮延迟 |
| 常用工具常驻上下文 | 对高频工具绕开检索 | 挤占预算，仅适用于少数工具 |
| 冻结索引与检索器版本 | 保证评测可比 | 牺牲最新工具的即时可用性 |

最后一项常被忽略却最关键：检索器是流水线的一部分，其版本应当与分析代码、包版本一起被记录，否则复现时结果不可比。

## 7.14 多智能体协作

### 7.14.1 框架与协作拓扑

| 框架 | 协作模式 | 适用场景 | 局限 |
|------|----------|----------|------|
| MetaGPT | 把软件公司的角色（产品经理、架构师、工程师、QA）与 SOP 编码进提示 | 需求明确的工程化产出 | 角色设定偏软件工程，科研适配需重写 SOP |
| CrewAI | 以角色 + 任务为单位的分层协作 | 分析、写作、审阅这类可切分的流程 | 角色间依赖需显式声明，否则顺序错乱 |
| AutoGen | 智能体间自由对话，支持人类介入 | 需要多轮讨论与人工判断的任务 | 对话轮次不受控，成本与延迟上升 |
| ChatDev | 虚拟软件公司，按阶段流水线协作 | 完整项目生成 | 中间产物质量依赖前一阶段 |
| LangGraph | 以图显式定义多节点与状态 | 步骤固定的科研流水线 | 需要学习状态模式与边条件 |

### 7.14.2 一个统计科研流水线

多智能体在统计课题中的典型分工：

| 角色 | 职责 | 产出 |
|------|------|------|
| 数据工程师 | 读取原始数据、清洗、生成数据字典与缺失报告 | 干净数据集 + 数据字典 |
| 分析员 | 按分析方案拟合模型、跑诊断 | 结果表 + 诊断表 |
| 批评者（Reviewer） | 独立复核：检查分析是否与方案一致、统计量是否被误用 | 问题清单 |
| 撰稿人 | 按期刊格式整理结果 | 结果章节草稿 |
| 复现员 | 从零重跑整条流水线，检验是否得到相同结果 | 复现报告 |

批评者与复现员两个角色在单人工作流中最容易被省略，却是多智能体协作最有价值的部分：它们提供的是**独立的错误检测**，而非额外的计算量。

### 7.14.3 集成视角：误差相关性决定收益

把多个智能体的判断看作一组基学习器，多数投票能否降低错误率取决于误差的相关性。设每个智能体单步错误率为 $e$、两两误差相关系数为 $\rho$，则多数投票的方差中，独立性带来的削减因子约为 $\rho$ 的增函数：$\rho \to 1$ 时集成几乎无收益，$\rho \to 0$ 时错误率随个体数指数下降（bagging 的经典结论）。

在智能体场景中，$\rho$ 通常很高：多个智能体共享同一底层模型、同一提示模板与同一批工具，错误模式高度重合。因此"三个智能体互相投票"远不如"一个智能体写代码、另一个智能体用不同方法独立复算"来得有效——后者的误差来源不同（不同的实现路径），$\rho$ 更低。这与统计实践中"用两种独立方法交叉验证结果"是同一原则，也解释了为什么交叉验证（用不同的数据子集）比重复拟合（用同一批数据）更有信息量。

### 7.14.4 局限

成本随智能体数量线性增长，轨迹长度与 token 消耗同时上升；一致性难以保证，不同智能体对同一约定可能有不同理解，需要把共享事实写进 CLAUDE.md 或共享文件而非各自上下文；责任归属模糊，出错时难以定位是哪个角色的判断有问题；最后，多智能体带来的质量提升往往小于"把单个智能体的工具与验证规则写清楚"带来的提升。对绝大多数统计课题，先做好单智能体加脚本化验证，再考虑引入协作角色。

## 7.15 本地自动化实践

### 7.15.1 从零到可复现：蒙特卡洛验证 Skill 的完整部署

```bash
# 1. 建立 Skill 目录（scripts 放脚本，references 放长文档）
mkdir -p monte-carlo-hypothesis-test/scripts
mkdir -p monte-carlo-hypothesis-test/references

# 2. 编写 SKILL.md（用 7.10.1 模板）与 scripts/run_simulation.py（用 7.10.1 脚本）

# 3. 本地测试脚本（H0 真 / H1 真各一次）
python scripts/run_simulation.py --alpha 0.05 --n1 30 --n2 30 --n-sim 10000
python scripts/run_simulation.py --alpha 0.05 --n1 30 --n2 30 --effect-size 0.5 --n-sim 10000

# 4. 部署：全局安装，或项目级安装（推荐，可随 git 共享）
cp -r monte-carlo-hypothesis-test ~/.claude/skills/
cp -r monte-carlo-hypothesis-test .claude/skills/
```

部署完成后重启会话，课题组成员可直接提出："验证一个 two-sample t-test，n1 = 50，n2 = 50，α = 0.05，运行蒙特卡洛模拟 10000 次，报告第一类错误率与覆盖率。"智能体按 description 路由到该 Skill，读取指令、调用脚本、解析 `results.json`、按模板输出对比表并给出通过与否的结论（含"需增大 n_sim"这一类建议）。

### 7.15.2 四类本地自动化场景

**场景一：文件批量重命名与整理。** 智能体扫描指定目录的 PDF，提取作者—年份—标题信息，按 `references/naming-convention.md` 重命名，并返回重命名前后对照表。默认只做预览，确认后才执行写操作，日志落盘为 `rename_log.csv`。

**场景二：数据处理与统计分析。** 在智能体中直接描述需求：读取 `~/data/clinical.csv`，执行缺失值诊断（区分 MCAR/MAR/MNAR）、两样本 t 检验（实验组 vs 对照组）、生成森林图与箱线图、输出符合期刊规范的报告。智能体调用相关科研技能，在本地 Python 或 R 沙箱中执行代码，读写本地文件并输出结果。

**场景三：文献管理与 Zotero 集成。** 配置 Zotero MCP 服务器（配置示例见 7.5.3）后，智能体可直接检索文献库、批量导入新条目、生成引用格式。

**场景四：终端自动化。** 创建虚拟环境、安装依赖、运行 pytest、修复失败测试这类操作可由智能体的内置终端工具执行；所有命令在执行前显示供人工审批。

### 7.15.3 安全边界

本地自动化把智能体从"生成文本"升级为"改变状态"，因此需要三条硬边界：

1. **权限最小化**：只挂载任务必需的路径，只读数据目录以只读方式挂载。
2. **人工确认**：写文件、删除、网络请求、安装依赖四类操作默认需确认，不开启全自动模式。
3. **可回滚**：批量操作先生成差异预览与备份；涉及原始数据的操作一律不动原始文件，输出写到新目录。

统计研究的特殊性在于原始数据往往不可再得（临床试验、田野调查），任何自动化脚本对原始数据的写操作都应被禁止。

### 7.15.4 课题组的实施路径

第一步，部署官方技能库与科研技能库，验证安装结果（命令见 7.11.2）。第二步，用 7.10 的模板创建第一个与本领域直接相关的 Skill（例如生存分析的验证、贝叶斯后验诊断、Bootstrap 覆盖率验证），替换模板中的场景与脚本。第三步，把 Skill 纳入版本管理，形成课题组共享的技能库：

```
your-repo/
├── .claude/
│   ├── skills/
│   │   ├── monte-carlo-hypothesis-test/
│   │   ├── publication-figure/
│   │   ├── r-analysis-pipeline/
│   │   ├── experimental-design-power/
│   │   └── regression-diagnostics/
│   └── settings.json      # 团队共享配置（含 MCP 服务器）
├── CLAUDE.md              # 项目规则（构建命令 / 目录约定 / 数据只读策略）
└── AGENTS.md              # 跨 Agent 兼容规则
```

Skill 把课题组的隐性方法论（如何做模拟、如何画图、如何写报告）转化为显性可复用资产：写一次，全组长期受益，且跨平台（Claude Code、Cursor、Codex、Gemini CLI）通用。与版本控制类似，其价值不在单次使用，而在把工作方式固化为可评审、可继承的对象。

## 7.16 开源Skill项目清单

以下清单按场景分类，URL 为纯文本可复制形式。星标数为素材整理时的记录，随时间波动。

### 7.16.1 官方仓库（Anthropic）

| # | 项目名称 | 功能介绍 | 项目地址 |
|---|---------|---------|---------|
| 1 | anthropics/skills | 官方 Agent Skills 仓库，含创意/技术/企业类示例 | github.com/anthropics/skills |
| 2 | document-skills | 官方四大文档技能：DOCX/PDF/PPTX/XLSX 生成与编辑 | github.com/anthropics/skills（skills/document-skills 子目录） |
| 3 | example-skills | 官方示例技能，展示 SKILL.md 最佳实践 | github.com/anthropics/skills（skills/example-skills 子目录） |
| 4 | anthropics/claude-cookbooks | 官方 notebook 与示例（RAG、工具调用、Skills、MCP），50.9k 星标 | github.com/anthropics/claude-cookbooks |
| 5 | claude-agent-sdk-python | 官方 Python Agent SDK，7.8k 星标 | github.com/anthropics/claude-agent-sdk-python |
| 6 | claude-agent-sdk-typescript | 官方 TypeScript Agent SDK，1.7k 星标 | github.com/anthropics/claude-agent-sdk-typescript |
| 7 | anthropics/claude-quickstarts | 官方快速入门示例应用，17.4k 星标 | github.com/anthropics/claude-quickstarts |
| 8 | anthropic-sdk-python | Claude Python SDK，3.8k 星标 | github.com/anthropics/anthropic-sdk-python |
| 9 | anthropic-sdk-typescript | Claude TypeScript SDK，2.1k 星标 | github.com/anthropics/anthropic-sdk-typescript |
| 10 | anthropic-sdk-go | Claude Go SDK，1.2k 星标 | github.com/anthropics/anthropic-sdk-go |

### 7.16.2 科研专用

| # | 项目名称 | 功能介绍 | 项目地址 |
|---|---------|---------|---------|
| 11 | K-Dense-AI/scientific-agent-skills | 165+ 已验证科研技能 + 100+ 科学数据库，覆盖生物/化学/医学/药物发现 | github.com/K-Dense-AI/scientific-agent-skills |
| 12 | skillsovermcp.com | 193+ 科研技能索引平台，每个 SKILL.md 作为 MCP 工具自动加载 | skillsovermcp.com |
| 13 | agenticskills.io | 跨平台 Agent Skill 目录，193+ 技能，16 类别，支持 Claude/Codex/Cursor/Gemini | agenticskills.io |
| 14 | skillsovermcp/self-host | 将任意 GitHub 仓库的 skills 自托管为 MCP 服务器 | github.com/skillsovermcp |
| 15 | K-Dense Web | K-Dense 科研云平台，提供云端计算与更多 agent | k-dense.ai |

### 7.16.3 社区聚合仓库

| # | 项目名称 | 功能介绍 | 项目地址 |
|---|---------|---------|---------|
| 16 | ComposioHQ/awesome-claude-skills | 最全面的 Claude Skills 导航清单，74.9k 星标 | github.com/ComposioHQ/awesome-claude-skills |
| 17 | hesreallyhim/awesome-claude-code | Claude Code 工作流、slash-commands、CLAUDE.md 文件集合，51.6k 星标 | github.com/hesreallyhim/awesome-claude-code |
| 18 | VoltAgent/awesome-claude-code-subagents | 100+ 专用子 agent，覆盖全栈开发，24.0k 星标 | github.com/VoltAgent/awesome-claude-code-subagents |
| 19 | travisvn/awesome-claude-skills | 精选 Claude Skills 资源，聚焦 Claude Code 工作流，14.5k 星标 | github.com/travisvn/awesome-claude-skills |
| 20 | BehiSecc/awesome-claude-skills | 分类清晰：文档处理/开发工具/数据分析，9.9k 星标 | github.com/BehiSecc/awesome-claude-skills |
| 21 | langgptai/awesome-claude-prompts | Claude 提示词集合，5.4k 星标 | github.com/langgptai/awesome-claude-prompts |
| 22 | vijaythecoder/awesome-claude-agents | 专用 AI agent 团队，用于构建 feature 与调试，4.4k 星标 | github.com/vijaythecoder/awesome-claude-agents |
| 23 | awesomeclaude.ai | 在线目录：204+ Claude Agent Skills，13 类别 | awesomeclaude.ai |

### 7.16.4 MCP 服务器生态

| # | 项目名称 | 功能介绍 | 项目地址 |
|---|---------|---------|---------|
| 24 | punkpeye/awesome-mcp-servers | 最全 MCP 服务器列表，91.8k 星标 | github.com/punkpeye/awesome-mcp-servers |
| 25 | modelcontextprotocol/servers | MCP 官方参考服务器实现 | github.com/modelcontextprotocol/servers |
| 26 | github/github-mcp-server | GitHub 官方 MCP 服务器，支持仓库/PR/Issue 操作 | github.com/github/github-mcp-server |
| 27 | modelcontextprotocol.io | MCP 协议官方规范与 SDK | modelcontextprotocol.io |
| 28 | mcpservers.org | MCP 服务器目录与评测 | mcpservers.org |
| 29 | mcpmarket.com | MCP 市场：Top 100 MCP 服务器排名 | mcpmarket.com |
| 30 | filesystem-mcp | 文件系统 MCP：AI 安全读写本地文件 | mcpservers.org（搜索 filesystem） |
| 31 | lobehub/lobe-chat | MCP 客户端：7×24 AI 团队调度，333,739+ 技能 | github.com/lobehub/lobe-chat |

### 7.16.5 文档处理技能

| # | 项目名称 | 功能介绍 | 项目地址 |
|---|---------|---------|---------|
| 32 | pptx 技能 | PowerPoint 生成：使用 pptxgenjs 库创建演示文稿 | github.com/anthropics/skills（skills/pptx） |
| 33 | docx 技能 | Word 文档生成：使用 docx-js 库，注意默认 A4 页面 | github.com/anthropics/skills（skills/docx） |
| 34 | xlsx 技能 | Excel 处理：使用 SheetJS 库，支持公式与数据透视表 | github.com/anthropics/skills（skills/xlsx） |
| 35 | pdf 技能 | PDF 解析与生成：提取表单字段、合并拆分 | github.com/anthropics/skills（skills/pdf） |

### 7.16.6 代码与数据分析

| # | 项目名称 | 功能介绍 | 项目地址 |
|---|---------|---------|---------|
| 36 | vercel-labs/agent-skills | Next.js/React 前端开发技能 | github.com/vercel-labs/agent-skills |
| 37 | trailofbits/skills | 安全审计、代码供应链安全技能 | github.com/trailofbits/skills |
| 38 | OthmanAdi/planning-with-files | 项目规划与方案设计技能 | github.com/OthmanAdi/planning-with-files |
| 39 | andrepimenta/claude-code-chat | VS Code 原生聊天界面，支持 MCP，1.1k 星标 | github.com/andrepimenta/claude-code-chat |
| 40 | aaddrick/claude-desktop-debian | Linux 桌面版 Claude，5.3k 星标 | github.com/aaddrick/claude-desktop-debian |

### 7.16.7 自动化与网页抓取

| # | 项目名称 | 功能介绍 | 项目地址 |
|---|---------|---------|---------|
| 41 | firecrawl | 网页抓取基础设施：AI 搜索、读取、操作实时网页 | github.com/firecrawl/firecrawl |
| 42 | MineDojo/Voyager | Skill Library 开创者：Minecraft 中自动积累可执行代码技能库 | github.com/MineDojo/Voyager |
| 43 | ysymyth/ReAct | 推理与行动结合范式：Thought→Action→Observation 循环 | github.com/ysymyth/ReAct |
| 44 | ashwingopalsamy/claude-code-theme | Claude 主题的 VS Code 配色方案 | github.com/ashwingopalsamy/claude-code-theme |

### 7.16.8 开发工具与规范

| # | 项目名称 | 功能介绍 | 项目地址 |
|---|---------|---------|---------|
| 45 | agentskills.io | Agent Skills 开放标准规范 | agentskills.io |
| 46 | code.claude.com/docs | Claude Code Skills 官方文档 | code.claude.com/docs/en/skills |
| 47 | gh-skill 手册 | GitHub CLI gh skill 命令 man page | mankier.com（搜索 gh-skill） |
| 48 | Anthropic Skills 指南 PDF | 官方完整构建指南：规划/设计/测试/分发 | resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf |
| 49 | anthropic-sdk-java | Claude Java SDK，355 星标 | github.com/anthropics/anthropic-sdk-java |
| 50 | anthropic-sdk-ruby | Claude Ruby SDK，361 星标 | github.com/anthropics/anthropic-sdk-ruby |

### 7.16.9 相关生态项目（跨章索引）

素材中另收录一批与 Skill/Agent 生态相邻的项目，主题分别属于文献检索、模型训练、多模态与学习资料，本章仅列出索引，正文详见对应章节：

| # | 项目名称 | 一句话说明 | 地址 | 核实状态 |
|---|---------|-----------|------|----------|
| 1 | Fetch Scholar | 面向研究者的 AI 文献检索平台，支持导出 BibTeX/EndNote/Mendeley，免费额度约每月 50 次 | fetchscholar.com | 候选，需核实（详见第 8 章） |
| 2 | 《动手学大模型》Dive into LLMs | 上海交大开源的大模型编程实践教程，11 个主题，每章配可运行 Notebook，2025 年 6 月上线华为昇腾国产化分支 | github.com/Lordog/dive-into-llms | 已验证，约 38.9k–50.8k 星（详见第 4 章） |
| 3 | 殆知阁 daizhigev20 | 中文古籍纯文本语料库，17,746 部文献，索引约 4.6GB，约 20 亿字，可用于 RAG 知识库构建 | github.com/garychowcmu/daizhigev20，检索站 daizhige.org | 已验证（详见第 2 章） |
| 4 | Echo Loop | 英语听说训练应用，构建"盲听→精听→跟读→复述→间隔复习"闭环，AGPL-3.0 | github.com/echo-loop/Echo-Loop | 用户提供，未独立验证 |
| 5 | Light Skills | 科研/竞赛/创新项目 AI Skill 包，覆盖 Idea 评估到论文写作全流程，MIT 协议 | github.com/Light0305/Light-skills | 用户提供，未独立验证 |
| 6 | English-level-up-tips | 面向中文母语者的英语进阶指南，约 4.8 万星，CC BY-NC 4.0 | github.com/byoungd/English-level-up-tips | 用户提供，广泛知名 |
| 7 | Sivia | 科研绘图 Skill：读论文 PDF 或方法描述→匹配示意图案例库→生成可编辑 PPT/WPS/draw.io 格式，MIT | github.com/exsinger-hub/Sivia | 用户提供，未独立验证 |
| 8 | awesome-llm-apps | 收录 100+ 可直接运行的 LLM 应用、Agent 与 RAG 示例源码 | github.com/Shubhamsaboo/awesome-llm-apps | 用户提供，广泛知名 |
| 9 | TimesFM | Google Research 预训练时序基础模型，主线 TimesFM 3.0 约 3.31 亿参数，权重非商用许可 | github.com/google-research/timesfm | Google 官方项目（详见第 6 章） |
| 10 | MiniT2I | 极简文生图基线：像素空间 MM-JiT + Flow Matching，B/16 版 258M 生成参数 + 341M 文本编码器 | github.com/PeppaKing8/minit2i-jax | 学术前沿（详见第 3 章） |
| 11 | ARIS (Auto-claude-code-research-in-sleep) | 由 83 个可组合 Claude Code Skill 组成的科研自动化框架，跨模型协作串起选题到投稿全流程 | github.com/wanshuiyin/Auto-claude-code-research-in-sleep | 用户提供，未独立验证 |
| 12 | Hello-Agents《从零开始构建智能体》 | Datawhale 出品的中文智能体开源教程，配套自研 HelloAgents 框架 | github.com/datawhalechina/hello-agents | Datawhale 出品，广泛使用（详见 7.3.3） |
| 13 | ZDTaichu5.0-9B | 中科院自动化所紫东太初团队开源的约 9B 参数多模态模型，主打空间推理与工具调用 | github.com/Taichu-AI/ZDTaichu5.0-9B | 中科院官方（详见第 4 章） |
| 14 | MiniMind | 用原生 PyTorch 从零复现大模型全流程，官方称单张 RTX 3090 约 2 小时完成训练，主线 MiniMind-3 为 64M 参数 | github.com/jingyaogong/minimind | 广泛知名，教学用途（详见第 4 章） |
| 15 | Research-Starter-Kit | 南京大学 LAMDA 实验室发起的科研入门指南，覆盖找论文到 Rebuttal 全流程 | github.com/LAMDA-NeSy/Research-Starter-Kit | LAMDA 出品，广泛使用 |
| 16 | Codex Gateway | 官方 Codex app-server 的网页前端与连接网关，支持后台长任务与确认 Diff 后应用 | github.com/asfsfafas/Codex-Gateway | 用户提供，中等置信 |
| 17 | 《大模型基础》Foundations of LLMs | 浙大 ZJU-LLMs 开源教材，第一版六章，提供完整版与分章 PDF、英文版与配套论文清单，月度更新 | github.com/ZJU-LLMs/Foundations-of-LLMs | 约 1.78 万星，配套 B 站课程 50 万+ 观看（详见第 4 章） |
| 18 | Build Your Own X | "从零造轮子"教程索引，30+ 方向，每条标注实现语言，CC0 协议 | github.com/codecrafters-io/build-your-own-x | 约 54.8 万星（学习类仓库最高之一） |
| 19 | ArcReel | 开源 AI 视频生产工作台，由 Claude Agent SDK 驱动多智能体完成分集规划到 FFmpeg 合成，AGPL-3.0，Docker 一行部署（端口 1241） | github.com/ArcReel/ArcReel，官网 arc-reel.com/en | 约 4.1k 星 |

**使用与许可注意事项**

- ArcReel 采用 AGPL-3.0 协议，较 MIT/Apache 严格，用于二次分发或 SaaS 服务需按同协议开源。
- ArcReel 属"工具免费、模型调用付费"模式：Docker 部署后需自行配置各供应商 API Key，生成图像与视频的费用由使用者承担，好处是可随时切换供应商且费用可视化。
- Build Your Own X 是免费教程索引仓库，同名商业平台 codecrafters.io 为独立付费产品，二者不要混淆。
- 《大模型基础》采用月度更新机制，配套论文清单可用于跟踪进展。
- TimesFM 代码为 Apache-2.0，权重为非商用许可，商用前须单独确认。

## 本章小结

- 智能体是在语言模型之外增加规划、记忆、工具与执行四个模块的闭环系统，其轨迹可形式化为序贯决策；评测结果应视为带标准误的估计值，而非确定性结论。
- 多步任务的端到端正确率按 $p^k$ 衰减：单步 0.95 的智能体跑 10 步仅剩约 0.60；在误差集中的步骤后部署"发现概率 $c$、重试一次"的确定性校验，把单步正确率提升到 $p + (1-p)cp$，通常比把预算投在更强的模型上更划算。
- ReAct 范式用 Thought→Action→Observation 的交替循环把外部观测引入事实来源，其失效主要表现为误差沿轨迹累积；确定性子任务应交给脚本而非模型的即兴推理。
- MCP 与函数调用分工明确：函数调用解决"意图到结构化参数"，MCP 解决"工具与数据源的统一接入"；二者标准化的是连接，不保证数据正确性。
- 参数校验改变的是错误的结构而非错误的总量：受限解码只保证格式合法；仅加 Schema 校验会把可察觉的崩溃转移为静默错误，配合运行时断言才能同时提高成功率并压低静默错误占完成任务的份额，后者是科研场景最应盯住的指标。
- Agent Skills 是"文件夹 + SKILL.md"的开放标准，其价值在契约（输入输出、执行步骤、验证规则、禁止行为）而非提示词长度，可版本化、可评审、可跨产品复用。
- 渐进式披露三级加载（frontmatter 常驻、正文按需、资源执行时读取）把上下文开销从"随技能数线性增长的全文"降为"描述开销 + 当次任务正文开销"，是大规模技能库可用的前提。
- 动态工具检索的检索错误率 $\varepsilon$ 以 $p_f - p_s p_e$ 的边际系数传导到端到端成功率，并在 $L$ 步轨迹上以 $1-(1-\varepsilon)^L$ 累积；这与多重检验的族系错误率及数据依赖选择导致的选择性推断问题同构。模拟显示不校正时注入集合中约一半是假阳性，BH 校正把 FDR 压回名义水平的代价是真工具漏检近半，而提高描述与查询的语义可分性（效应量）能在 FDR 不变的前提下大幅降低漏检。
- 多智能体协作的收益取决于各智能体误差的相关性：共享模型与工具时相关性高，投票收益有限；用不同方法独立复算（低相关）才是有效的错误检测手段。
- Skill 在统计科研中最接近"机器可读的分析方案"，其部署应配合数据只读策略、写操作人工确认与产出备份三条边界。

## 延伸资源

- **ReAct（arXiv:2210.03629）**：推理与行动结合的奠基论文，提出 Thought→Action→Observation 循环。arxiv.org/abs/2210.03629，代码 github.com/ysymyth/ReAct
- **Toolformer（arXiv:2302.04761）**：语言模型自监督学习工具调用的早期工作。arxiv.org/abs/2302.04761
- **Voyager（arXiv:2305.16291）**：技能库自动积累范式的源头，技能以可执行代码存储，可跨任务复用。arxiv.org/abs/2305.16291，代码 github.com/MineDojo/Voyager
- **Agent Skills 开放标准（agentskills.io）**：SKILL.md 的规范文档与模板，跨产品通用的基础。agentskills.io
- **anthropics/skills**：官方技能仓库，含文档四大技能与示例技能。github.com/anthropics/skills
- **Claude Code Skills 官方文档**：安装、优先级与调试的权威说明。code.claude.com/docs/en/skills
- **Anthropic 完整构建指南 PDF**：规划、设计、测试、分发四阶段的官方教程。resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf
- **K-Dense-AI/scientific-agent-skills**：面向科研的 165+ 技能库，支持 MCP 接入。github.com/K-Dense-AI/scientific-agent-skills
- **Skills Over MCP（skillsovermcp.com）**：193+ 科研技能索引，每个 SKILL.md 作为 MCP 工具加载。skillsovermcp.com
- **agenticskills.io**：跨平台技能目录，支持 Claude Code/Codex/Cursor/Gemini CLI。agenticskills.io
- **Model Context Protocol（modelcontextprotocol.io）**：协议规范、SDK 与官方参考服务器。modelcontextprotocol.io，github.com/modelcontextprotocol/servers
- **awesome-mcp-servers**：社区最全的 MCP 服务器列表。github.com/punkpeye/awesome-mcp-servers
- **Hello-Agents《从零开始构建智能体》**：Datawhale 出品的中文智能体系统教程，五部分 16 章并附全套代码。github.com/datawhalechina/hello-agents
- **Agent-Learning-Hub**：以"做出可靠 Agent"为目标的分阶段学习路线与评测清单。github.com/datawhalechina/Agent-Learning-Hub
- **12-factor-agents**：智能体工程最佳实践指南，适合作为设计评审的对照清单。github.com/humanlayer/12-factor-agents
- **langgraph**：图结构智能体编排框架，适合步骤固定的科研流水线。github.com/langchain-ai/langgraph
- **gh skill 手册**：GitHub CLI 技能安装命令的用法说明。cli.github.com（搜索 gh skill）
- **Benjamini, Y. & Hochberg, Y. (1995). Controlling the False Discovery Rate: A New and Powerful Approach to Multiple Testing. JRSS-B 57(1), 289–300**：FDR 控制的原始论文，7.13.4 检索校正模拟的统计出处。
- **Taylor, J. & Tibshirani, R. J. (2015). Statistical learning and selective inference. PNAS 112(25), 7629–7634**：选择性推断的入门综述，对应 7.13.4 中数据依赖选择导致的不确定性低估问题。
- **Building Effective Agents（Anthropic 工程博客）**：官方对工作流与智能体的边界、何时用更简单方案的建议，与 7.1.6 与 7.4 的分层互为印证。anthropic.com/engineering/building-effective-agents
- **Outlines（dottxt-ai/outlines）**：开源受限解码库，把 JSON Schema 编译为解码文法，是 7.5.1 中结构化输出格式保证的参考实现。github.com/dottxt-ai/outlines


