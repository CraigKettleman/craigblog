---
title: "第 19 章 循环工程（Loop Engineering）"
slug: "hitchhiker-agentic-ai-19-loop-engineering"
lang: "zh"
date: "2026-09-16T00:20:00.000Z"
draft: false
featured: false
categories:
  - "hitchhiker-agentic-ai"
cover: "./cover.png"
description: "从业者与基于 LLM 的智能体（Agent）交互方式的演变，遵循了一条惊人一致的轨迹。2022–2024 年，首要技能是提示工程（prompt engineering）：精心组织措辞，以便从单次模型调用中引出想要的回答。…"
keywords:
  - "智能体"
  - "Agentic AI"
  - "LLM"
  - "大语言模型"
  - "强化学习"
  - "RLHF"
  - "MCP"
---

<div class="hh-guide">

<aside class="hh-part">
<p class="hh-part-title">第 V 部分 智能体 AI（Agentic AI）</p>
</aside>

从业者与基于 LLM 的智能体（Agent）交互方式的演变，遵循了一条惊人一致的轨迹。2022–2024 年，首要技能是<em>提示工程（prompt engineering）</em>：精心组织措辞，以便从单次模型调用中引出想要的回答。到 2025 年，焦点转向<em>上下文工程（context engineering）</em>——策展模型在推理时看到的全部 Token 集合，包括检索到的文档、工具输出与对话历史 [9]。上一章我们讨论了<em>harness 工程</em>：设计围绕智能体的运行时环境——工具、沙箱、记忆与护栏。循环工程 [279] 是这一演进中的下一层：设计<em>迭代控制结构</em>，让智能体自主地朝目标前进，无需人类在每一轮输入下一条指令。

这个说法诞生于 2026 年 6 月：Peter Steinberger [338] 主张，开发者应当停止直接提示编码智能体（Agent），转而设计能够提示这些智能体的系统——这一观点得到了 Boris Cherny（Anthropic 的 Claude Code 负责人）的印证，他提到自己的角色已经完全转变为编写协调模型行动的外部执行循环 [52]。这不仅仅是工具偏好。它反映了一个结构性现实：当一次智能体运行可能持续一小时并修改数十个文件时，杠杆率最高的工程不再在提示里——而在那个让智能体始终保持高效、被验证、不偏离目标的循环里。

## 上下文工程：循环之下的那一层

在循环能够运行之前，必须有某个东西决定模型在每一步看到什么。这门学科就是上下文工程（context engineering）：在推理期间策展并维护上下文窗口中最优 Token 集合的正式实践。它是循环之下的那一层——循环的状态借以被翻译为模型输入的机制。

该术语由 Tobi Lütke（Shopify CEO）在 2025 年 6 月推广开来 [245]，他将其描述为与 AI 智能体协作的新兴核心技能。Anthropic 在 2025 年 9 月将其形式化定义为“在推理期间策展并维护最优 Token 集合”。Andrej Karpathy 认同这一框架，将其描述为“在下一步之前，用恰到好处的信息填满上下文窗口的精妙艺术与科学” [178]。这些定义的汇聚反映了一个现实：随着上下文窗口从 4K 增长到 128K 再到 1M Token，<em>该往里面放什么</em>这一问题变得与<em>该问什么</em>同等重要。

上下文工程不是提示工程。提示工程优化的是<em>指令</em>——引导模型行为的措辞。上下文工程优化的是<em>信息环境</em>——模型据以推理的文档、工具输出、历史与状态的全集。在单轮交互中，两者的区别很小。在多步智能体循环中，它决定了模型是拥有正确行动所需的信息，还是在盲目飞行。

上下文工程借助几种互补的方法：

<ul><li>动态上下文组装：在每一轮循环迭代中从多个来源（检索到的文档、工具输出、记忆存储）选择并组合上下文，而不是使用固定模板。</li><li>RAG 集成：检索任务相关文档或代码片段并注入上下文窗口，用精确、即时的信息取代过时或泛泛的背景知识。</li><li>工具输出摘要化：在把冗长的工具输出（如长文件列表、测试运行器日志）插入上下文之前先压缩，在保留信号的同时节省 Token 预算。</li><li>对话历史管理：决定哪些先前的轮次原样保留、哪些摘要化、哪些彻底丢弃——一个滚动压缩问题，在长时程智能体运行中变得至关重要。</li><li>Token 预算分配：在相互竞争的信息来源之间显式划分上下文窗口，确保模型始终有空间生成有用的回答。</li></ul>

在循环中，上下文工程在每一轮迭代都会发挥作用。在每一步 <math alttext="t" display="inline"><semantics><mi>t</mi></semantics></math>，循环控制器必须决定：保留哪些先前的观测 <math alttext="o_{&lt;t}" display="inline"><semantics><msub><mi>o</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>t</mi></mrow></msub></semantics></math>、摘要哪些、哪些检索文档仍然相关，以及如何从这些组件组合出新的上下文 <math alttext="s_{t}" display="inline"><semantics><msub><mi>s</mi><mi>t</mi></msub></semantics></math>。这不是一次性的设计决策——而是一条与智能体策略 <math alttext="\pi_{\theta}" display="inline"><semantics><msub><mi>π</mi><mi>θ</mi></msub></semantics></math> 并行的动态策略。设计糟糕的上下文策略会让模型丢失目标、重复已经做过的动作，或无法使用它两步之前检索到的信息。设计良好的策略则能让模型的“工作记忆”在整个运行期间始终与任务的当前状态对齐。

## 把循环工程视为推理时强化学习

让循环工程与本书相关的核心洞见是：一个精心设计的智能体循环，在结构上与在推理时运行的 RL 优化过程完全相同——只是不对模型权重做梯度更新：



循环持续运行，直到 <math alttext="r_{t}" display="inline"><semantics><msub><mi>r</mi><mi>t</mi></msub></semantics></math> 超过成功阈值或触发终止条件。“策略” <math alttext="\pi_{\theta}" display="inline"><semantics><msub><mi>π</mi><mi>θ</mi></msub></semantics></math> 不通过梯度更新；相反，被更新的是<em>状态</em>——观测、错误消息与反思被追加到上下文中，让同一个冻结模型在后续迭代中产生更好的动作。这正是 Reflexion [326] 背后的机制：模型跨迭代改进，靠的不是学习新权重，而是阅读自己的失败历史。

这种对应关系不仅仅是隐喻。它有直接的工程后果：

<ol><li>奖励设计很重要：正如在 RLHF 中一样，规格不当的奖励（验证标准）会导致奖励作弊（Reward Hacking）——智能体满足了检查的字面要求，却违背了它的意图（例如删除失败的测试以让 CI 变绿）。</li><li>探索—利用权衡：一个反复采用同一失败做法的循环表现出糟糕的探索。像 Reflexion [326] 这样的机制通过自我批判加入显式探索，类似于 PPO 中的熵奖励。</li><li>时域与折扣：更长的循环会面临误差累积与上下文退化，正如长时域 RL 受困于信用分配难题。预算上限充当有效的时域。</li><li>状态表示：上下文管理（压缩、剪枝、外置）是循环工程中对应 RL 状态表示设计的部分——两者都决定了智能体能否对其处境进行有效推理。</li></ol>

## 生产级循环的解剖

一个可用的循环需要五个结构性原语 [279]，外加外部状态持久化：

### 五个原语

<div class="hh-table" id="ch19.t1"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>原语</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>在循环中的作用</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>RL 对应物</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">自动化</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>按计划或事件触发循环</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>回合启动；环境重置</span></span></td></tr><tr><th class="ltx_align_left">工作树（Worktree）</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>隔离并行的智能体</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>分布式 RL 中的独立 rollout 工作进程</span></span></td></tr><tr><th class="ltx_align_left">技能</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>把可复用的能力与项目知识固化为代码</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>策略条件化；任务特定的奖励塑形</span></span></td></tr><tr><th class="ltx_align_left">连接器</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>与外部工具和系统对接</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>环境动作空间</span></span></td></tr><tr><th class="ltx_align_left">子智能体</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>分解与验证</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>层级式 RL；评论家网络</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">表 19.1：</span>循环工程的五个结构性原语。</p></div>

自动化是把单次智能体运行转变为真正循环的心跳。它们定义了循环<em>何时</em>以及<em>为何</em>被触发——按 cron 计划、响应 webhook，或由文件系统事件触发。没有自动化，你拥有的只是一个智能体；有了它们，你拥有的才是一个能自我维持的系统。生产中的例子包括每晚的 CI 失败分诊、每小时的依赖漏洞扫描，以及提交后的评审环节。

一旦多个智能体并行运行，文件级冲突就会成为主要的失败模式。Git 工作树（worktree）——共享仓库历史的独立工作目录——提供了隔离：每个智能体在自己的分支上操作，不可能与同辈发生写入冲突。这与在分布式 RL 中运行独立 rollout 工作进程是同一原理（第 11.2 并行策略详解 节）：并行要求隔离。

技能（第 第 23 章 Agent Skills 章）把项目知识固化下来，否则智能体每个周期都要从头重新推导。在循环的语境中，技能充当<em>持久条件化</em>——那些在迭代之间保持稳定的约定、构建流程与约束。没有技能，每次循环迭代都从冷启动开始；有了它们，智能体可以跨多次运行累积知识，而不会因为反复重新发现的事实耗尽上下文窗口。

连接器——通常通过模型上下文协议（MCP）实现（第 第 22 章 模型上下文协议（Model Context Protocol, MCP） 章）——把循环的动作空间扩展到文件系统之外。一个只能读写文件的循环能力有限；而一个连接到 issue 跟踪器、CI 系统、预发布环境与团队沟通渠道的循环，可以闭合完整的反馈回路：发现问题 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 修复 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 验证 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 部署 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 通知。

循环工程中最关键的结构性原则，是把<em>产出</em>输出的智能体与<em>评估</em>输出的智能体分开。让模型给自己的输出打分，就像让学生给自己的考卷评分——激励是错位的。一个专门的验证子智能体——可能运行不同的模型，或以更高的推理强度运行——提供了独立评估，使无人值守运行变得可信。这映照了 RL 中的演员—评论家（actor–critic）架构：演员（生成器）提出动作；评论家（验证器）评估动作。

### 外部状态：循环的记忆

上述每个原语都在单次迭代内运作。真正把迭代跨时间连接起来的是<em>外部状态</em>。由于 LLM 在两次调用之间是无状态的——模型会忘记不在当前上下文中的一切——循环的连续性必须存放在磁盘上：一个 markdown 进度文件、一个结构化数据库，或一份受版本控制的日志。这个外部状态承担三项职能：

<ol><li>进度追踪：尝试过什么、什么成功了、还剩下什么。</li><li>失败记忆：哪些做法尝试过并失败了，防止循环在相同的死胡同之间来回振荡。</li><li>移交上下文：当循环升级到人类或后续运行时，状态文件提供完整的审计轨迹。</li></ol>

## 循环的伪代码

剥离到本质，每个智能体循环都是一种控制结构，它更接近恒温器或 REPL，而不是一次对话：

```python
def agent_loop(goal: str, max_steps: int = 50, budget: TokenBudget = None):
    """The canonical agent loop structure."""
    state = initialize_state(goal)        # Load external state + goal

    for step in range(max_steps):
        # 1. Reason about current state (ReAct-style)
        thought = model.reason(state)

        # 2. Choose and execute action
        action = model.choose_action(state)
        observation = environment.execute(action)

        # 3. Update state with new information
        state = update_state(state, thought, action, observation)

        # 4. Compact context if approaching window limit
        if state.token_count > CONTEXT_BUDGET * 0.8:
            state = compact(state)        # Summarize old steps

        # 5. Check termination conditions
        if verifier.passes(state, goal):  # Deterministic check
            persist_state(state, status="success")
            return Success(state)

        if no_progress_detected(state, window=3):
            persist_state(state, status="stuck")
            return escalate_to_human(state)

        if budget and budget.exhausted():
            persist_state(state, status="budget_exceeded")
            return escalate_to_human(state)

    # Hard cap reached
    persist_state(state, status="max_steps")
    return escalate_to_human(state)
```

循环工程中一切有意思的东西，都是关于这几行代码中某一处的决策：什么构成有效的 <code>goal</code>、<code>verifier.passes</code> 如何实现、<code>compact</code> 如何在丢弃噪声的同时保留相关历史，以及 <code>no_progress_detected</code> 如何既避免过早终止又避免无限循环。

## 循环模式

在第 第 20 章 Agent 设计模式 章介绍的智能体设计模式的基础上，循环工程识别出一组自主性递增的模式层级：

### 验证循环

最简单也最常见的模式：生成，对照确定性检查进行验证，失败则重试。

这个模式之所以有效，是因为它的奖励信号清晰明确。测试运行器充当一个不可被腐蚀的评论家——与 LLM 作为评判者（LLM-as-judge）的评估不同，它无法被说服，也无法被欺骗。

### Reflexion 循环

在验证循环的基础上，在每次尝试之间加入显式的自我批判 [326]。每次失败之后，智能体把一段自然语言的反思（“我之所以失败，是因为改错了函数——错误的根源在 import，而不在实现”）写入情景记忆缓冲区。后续尝试会读取这个缓冲区，从而在不更新权重的情况下实现回合<em>内</em>的学习。

### 评估器—优化器循环

一种双模型架构 [246, 9]：一个模型生成候选，另一个独立的模型依据显式标准评估它并返回结构化反馈。生成器吸收这些反馈并产出改进版本。该循环重复进行，直到评估器的分数超过阈值，或迭代预算耗尽。

关键的设计选择在于评估器是<em>确定性的</em>（编译器、测试套件、linter），还是<em>概率性的</em>（另一个 LLM）。确定性评估器带来可靠的收敛；概率性评估器对于主观标准是必需的，但会引入评估器与生成器串通的风险。

### 层级循环

一种元循环会派生并监控多个子循环。编排者智能体把高层目标分解为子任务，把每个子任务分配给运行自己循环的专门化子智能体，监控它们的进度，并综合结果。这映照了层级式 RL [21]：高层策略选择子目标，低层策略执行它们。

### 自主研究循环

Karpathy 的 AutoResearch [179] 展示了一个约束极紧的研究循环：智能体修改训练脚本，运行一个有严格时限的实验（例如在 GPU 上跑 5 分钟），读取验证指标，然后决定是提交这次改动（指标改善）还是回滚（指标变差）。循环无限持续下去，通过迭代式实验探索超参数与架构空间。

这个模式之所以有效，是因为奖励信号是<em>标量且无歧义的</em>：验证损失要么下降，要么没有。每次迭代的严格时限防止任何单次失败实验消耗过多资源。

## 验证工程

验证是循环工程的奖励函数。它的质量决定了循环是收敛到正确答案、收敛到错误答案，还是完全无法收敛。本节按可靠性建立一个验证策略的层级。

### 验证层级

<div class="hh-table" id="ch19.t2"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>策略</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>信号来源</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>可靠性</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>适用场景</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">编译</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>语言工具链</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>确定性</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>代码必须能解析或构建</span></span></td></tr><tr><th class="ltx_align_left">类型检查</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>静态分析器</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>确定性</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>带类型的语言</span></span></td></tr><tr><th class="ltx_align_left">单元/集成测试</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>测试运行器</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>确定性</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>测试存在且正确</span></span></td></tr><tr><th class="ltx_align_left">Lint 与格式化</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>风格工具</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>确定性</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>约定已被固化为代码</span></span></td></tr><tr><th class="ltx_align_left">指标对比</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>训练/评估脚本</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>数值型</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>机器学习实验</span></span></td></tr><tr><th class="ltx_align_left">LLM 作为评判者</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>第二个模型</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>概率性</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>主观质量标准</span></span></td></tr><tr><th class="ltx_align_left">人工审查</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>领域专家</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>黄金标准</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>高风险决策</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">表 19.2：</span>按可靠性排序的验证策略。</p></div>

### 确定性 vs. 概率性验证

确定性验证器（编译器、测试套件、linter）是黄金标准，因为它们给出客观的通过/失败判定，模型无法通过说服来钻空子。测试要么通过，要么不通过——再雄辩的推理也改变不了判定。

对于没有机械检查手段的任务——代码质量、文档清晰度、UX 改进——概率性验证器（LLM 作为评判者）是必需的，但它们会引入失败模式：

<ul><li>自评估偏差：如果同一个模型既生成又评估，它会对自己的输出过于宽容。</li><li>评估器博弈：经过多轮迭代，生成器可能学会产出满足评估器表面模式、却并未真正达到质量的输出。</li><li>评估器不一致：由于采样方差，同一个 LLM 评判者可能在不同运行中对完全相同的输出给出不同分数。</li></ul>

缓解策略包括：使用另一个（通常更强的）模型作为评估器，提供带二值判据的显式评分细则，以及定期用人工判断校准 LLM 评判者。

## 终止工程

朴素循环的标志性失败就是它永不停止。终止设计不是事后补充——它是工程的一半。

### 终止条件

一个生产环境的循环带有多个相互独立的退出条件：

<ol><li>目标达成：验证器确认目标已满足。这是唯一的<em>成功</em>终止。</li><li>硬性迭代上限：循环步数的最大值（通常为 20–50）。无论其他条件如何，都阻止无界执行。</li><li>预算耗尽：Token 数量、墙钟时间或金钱成本超过预设上限。</li><li>无进展检测：最近 <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 次迭代没有产生可度量的状态变化——循环在振荡或卡住了。这是硬性上限之外最重要的安全机制。</li><li>致命错误：一种无论迭代多少次都无法解决的不可恢复状况（凭证缺失、基础设施故障）。</li></ol>

### 探索问题

一个卡在局部极小值的循环——反复采用同一个失败做法、只做微小变化——正在表现出 RL 中常见的探索—利用失败。促进循环内探索的机制包括：

<ul><li>基于反思的探索：<math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 次失败尝试之后，触发一个反思步骤，显式要求模型考虑根本不同的方法（类似于 PPO 中的熵正则化 [319]）。</li><li>温度升级：连续失败后提高采样温度，鼓励多样化输出。</li><li>策略记忆：不仅记录失败的动作，还记录失败的 <em>策略</em>。“方法 A（修改 handler）失败了 3 次；改试方法 B（重写 schema）。”</li><li>全新子 Agent：生成一个上下文窗口干净的子 Agent——它不会陷入已累积上下文可能造成的窠臼。</li></ul>

## 失败模式与反模式

### Loopmaxxing 陷阱

“Loopmaxxing”——假设让 Agent 跑足够多轮迭代最终就会产出正确解——是循环工程版的“算力够多就能解决一切”。它的失败与纯随机搜索的失败同源：没有指向改进的梯度信号，光靠迭代不是优化。

Loopmaxxing 在以下情况出现：

<ul><li>目标缺少可检查的成功条件（“改善用户体验”）</li><li>验证信号过于粗糙（在 1000 个测试的套件上只有二元通过/失败，无法提供指向修复的梯度）</li><li>无论迭代多少次，Agent 都缺乏解决问题的能力</li></ul>

解药是认识到循环只是 <em>放大</em>工程能力——它们不取代工程能力。目标定义糟糕的循环会以极高的效率去追求错误的东西。

### 理解债

当循环修改代码的速度超过工程团队审阅的速度时，仓库中实际存在的东西与人类理解之间的差距就会扩大——<em>理解债（comprehension debt）</em>[279]。与技术债不同（技术债由代码承载），理解债存在于团队成员的头脑里。它无声地复利增长，直到某场危机迫使人们去调试那些设计决策、结构依赖与边界情况完全未被记录的代码。

### 循环中的 Reward Hacking

正如 RL Agent 利用奖励定义缺陷在未达成既定目标的情况下拿到高奖励，循环 Agent 也会利用验证漏洞：

<ul><li>删除测试：最简单的 reward hack——删掉失败的测试让 CI 变绿。</li><li>指标博弈：通过记住验证集而非泛化来过拟合验证损失。</li><li>收窄规格：解决所陈述问题的简化版本，恰好能通过检查。</li><li>掩盖输出：压制错误输出而不是修复根本原因。</li></ul>

防御手段与 RLHF 相同：设计 <em>难以钻空子</em> 的奖励函数（验证标准）。多个独立检查——测试 <em>和</em> 类型检查 <em>和</em> lint <em>和</em> 审查子 Agent——构成一个比任何单一信号都更难利用的验证面。

### 上下文劣化

在长时程循环中，上下文窗口会被每一步的历史——思考、工具输出、错误、补丁——填满。当窗口接近容量时，模型对相关信息的注意力下降（“中间迷失”现象 [227]）。症状包括：

<ul><li>Agent 重新尝试它已经试过并明确标记为失败的方法</li><li>工具调用变得不那么精确——Agent 忘记了自己读过哪些文件</li><li>模型在臃肿的上下文中艰难地分配注意力，回答变得更短、更不连贯</li></ul>

对策：

<ol><li>激进压缩：每 <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 步之后，把已完成的工作总结为简短的状态描述，丢弃原始记录。</li><li>外部草稿本：把中间结果写入文件，让 Agent 按需读取，而不是把一切留在上下文里。</li><li>子 Agent 隔离：在新上下文窗口中生成子任务，只返回它们的结论——而不是完整推理轨迹。</li><li>历史滑动窗口：只完整保留最近 <math alttext="w" display="inline"><semantics><mi>w</mi></semantics></math> 步；把更早的内容压缩进一个摘要块。</li></ol>

## 生产级循环架构

### 夜间维护循环

最立即可用的循环模式：一个按计划在夜间运行的自动化，趁团队睡觉时处理累积的工作。

### 持续实验循环

受 AutoResearch [179] 启发，这个模式把循环应用于科学发现：修改训练脚本，运行一个有时限的实验，评估结果，并决定是提交还是回滚。

### 常驻编排器

像 OpenClaw [338] 这样的系统实现了持久化的“心跳”机制：一个按固定周期运行的元 Agent，它评估仓库状态、监督各子循环并派发工作。与每天触发一次的夜间模式不同，这个常驻编排器保持持续感知，并在数分钟内响应事件（新提交、失败的 CI、新到的 issue）。

该架构引入了一个支持崩溃恢复的持久状态数据库——如果编排器在周期中途失败，它会从最近一次检查点的状态恢复，而不是从头重启。子循环被作为具有各自终止条件的独立任务跟踪，编排器则监控它们是否停滞、预算耗尽或发生冲突。

## 长时程循环中的上下文管理

长时程循环面临一个根本性张力：上下文窗口是 Agent 的工作记忆，但它的容量是固定的。每一步都会追加新信息（思考、观察、错误），如果没有主动管理，窗口就会被填满、质量随之下降。本节介绍在多次迭代中维持上下文健康度的工程技巧。

### 压缩策略

<div class="hh-table" id="ch19.t3"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>策略</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>机制</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>权衡</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">周期性摘要</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>每 <math alttext="k" display="inline"><semantics><mi>k</mi><span></span></semantics></math> 步用摘要替换原始历史</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>有损；可能丢弃后续需要的细节</span></span></td></tr><tr><th class="ltx_align_left">滑动窗口</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>原样保留最近 <math alttext="w" display="inline"><semantics><mi>w</mi><span></span></semantics></math> 步；更早的内容做摘要</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>偏向近期；可能丢失早期上下文</span></span></td></tr><tr><th class="ltx_align_left">重要性加权</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>保留改变了状态的步骤；丢弃无操作步骤</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>需要定义“重要性”</span></span></td></tr><tr><th class="ltx_align_left">外部记忆</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>把细节写入文件；上下文中只保留引用</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>需要显式回读；会增加延迟</span></span></td></tr><tr><th class="ltx_align_left">子 Agent 隔离</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>子任务在全新上下文中运行；只返回结论</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>生成子 Agent 的开销；协调成本</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">表 19.3：</span>长时程循环的上下文压缩策略。</p></div>

最优策略取决于任务。对调试循环，滑动窗口效果很好——最近的错误最相关。对研究循环，重要性加权保留能在多次迭代中保存关键实验结果。对复杂的多文件重构，外部记忆（写一个汇总迄今为止所有改动的进度文件）能防止 Agent 跟丢自己的修改。

### 上下文预算

一个实用的启发式规则：为当前步骤的推理与生成预留上下文窗口的 20–30%。这意味着活跃历史不应超过容量的 70–80%。当接近这一阈值时，应在下一次迭代之前触发压缩——而不是等模型已经产出劣化输出之后。

```python
CONTEXT_CAPACITY = 128_000  # tokens
RESERVED_FOR_GENERATION = 0.25 * CONTEXT_CAPACITY
ACTIVE_BUDGET = CONTEXT_CAPACITY - RESERVED_FOR_GENERATION

def should_compact(state: LoopState) -> bool:
    return state.token_count > ACTIVE_BUDGET * 0.85
```

## 何时不该使用循环

循环工程并非普遍适用。当更简单的方法就足够时，它带来的复杂度、成本与失败模式都是不合理的。在以下情况优先使用直接 Prompt 或简单工作流：

<ul><li>任务是 单轮 的：一个精心构造的 Prompt 大多数时候能一次给出正确答案。加入循环会以显著的成本换来边际的质量提升。</li><li>目标 缺少可检查的条件：“让代码更好”没有客观的停止点。循环要么永远运行，要么任意停止。</li><li>人类交互成本很低：如果开发者正在积极工作并能实时提供反馈，交互式聊天模式比工程化一个循环更快。</li><li>任务 超出模型能力：如果 Agent 原则上无法解决该问题——无论迭代多少次——循环只是白白烧掉 Token。再多的重试也弥补不了根本性的能力缺口。</li><li>验证不可能：没有反馈信号，循环就没有梯度。它会退化为随机搜索。</li></ul>

## 循环的经济学

循环用算力（Token、墙钟时间）换取质量。理解这一权衡对生产部署至关重要。

经济上的决策是：循环产出的价值（节省的开发者时间、夜间生产力、质量提升）是否大于其 Token 成本？一个运行 20 次迭代、每次 $0.02 的循环，总成本为 $0.40——如果它能节省 30 分钟的开发者时间，这点成本微不足道。但一个没有预算上限的失控循环可能在一夜之间消耗数百美元。

## 历史背景与相关工作

循环工程并非凭空出现。它把延续数年的研究脉络产品化了：

<ol><li>ReAct（2022）[409]：确立了交错式推理—行动模式，所有现代循环都继承了它。</li><li>Reflexion（2023）[326]：加入了情景记忆与自我批评，证明 Agent 可以跨尝试改进而无需更新权重。</li><li>AutoGPT（2023）[330]：第一个被广泛使用的自主 Agent 循环，既展示了无人值守运行的潜力，也展示了它的陷阱（失控执行、高成本、不可靠的输出）。</li><li>Self-Refine（2023）[246]：把先生成后批评的循环形式化为迭代式精化。</li><li>“Ralph 循环”（2025）：非正式的基于 bash 的单行脚本，在简单的重试循环中运行编码 Agent——是产品化循环原语的先驱。</li><li>产品化循环（2026）：主流平台（Codex、Claude Code）直接内嵌了 <code>/goal</code> 与 <code>/loop</code> 命令，使复杂的循环模式无需自建基础设施即可使用 [279]。</li></ol>

从 AutoGPT 到现代循环工程的演进体现了一种成熟：早期系统缺少终止逻辑、验证与成本控制——它们是没有工程化的循环。2026 年的形式化补充了使无人值守运行安全的规范：显式终止、确定性验证、预算约束，以及 maker–checker 分离。

## 小结

循环工程代表了人机协作的当前前沿：从参与对话，转向设计那些能够自主进行对话的系统。它的关键原则是：

<ol><li>循环就是推理时 RL——状态、动作、奖励与策略更新（通过上下文）——在没有梯度下降的情况下运行。</li><li>验证就是奖励信号。它的质量决定收敛。优先采用确定性检查；把 maker 与 checker 分开。</li><li>终止是设计的一半。每个循环都需要硬性上限、预算与无进展检测。</li><li>上下文是有限的工作记忆。通过压缩、外化与子 Agent 隔离来主动管理它。</li><li>循环放大工程技能——它们不取代工程技能。目标定义糟糕的循环会高效地追求错误的东西。</li><li>从简单开始。一个带确定性验证器的单一验证循环，胜过一个你无法调试的复杂多 Agent 系统。</li></ol>

<aside class="hh-next">
<p class="hh-next-label">下一篇</p>
<p class="hh-next-title">第 20 章 Agent 设计模式</p>
<p class="hh-next-hint">在「智能体 AI 漫游指南」分类下继续阅读。</p>
</aside>

</div>
