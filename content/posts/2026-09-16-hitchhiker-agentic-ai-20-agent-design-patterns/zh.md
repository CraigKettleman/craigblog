---
title: "智能体 AI 漫游指南 · 第 20 章 Agent 设计模式"
slug: "hitchhiker-agentic-ai-20-agent-design-patterns"
lang: "zh"
date: "2026-09-16T00:21:00.000Z"
draft: false
featured: false
categories:
  - "hitchhiker-agentic-ai"
cover: "./cover.png"
description: "构建有效的 Agent 不仅需要强大的模型和一组工具。架构——即如何编排 LLM、如何分解任务、控制流如何在各组件间流动——决定了 Agent 是否可靠、可调试且具备成本效益。…"
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

构建有效的 Agent 不仅需要强大的模型和一组工具。<em>架构</em>——即如何编排 LLM、如何分解任务、控制流如何在各组件间流动——决定了 Agent 是否可靠、可调试且具备成本效益。本章介绍从 Anthropic、OpenAI、Google 以及开源社区的生产部署中沉淀出的经典设计模式。

## 工作流模式

这些模式改编自 Anthropic 对 Agent 构建模块的分类 [9]，在<em>预定义</em>的控制流中使用 LLM。由系统（而非模型）决定执行顺序。

### Prompt 链（Prompt Chaining）

最简模式：将复杂任务拆分为固定序列的 LLM 调用，将一个调用的结果作为上下文传入下一个调用。步骤之间的校验门可在错误向下游传播之前及早捕获它们。

<figure id="ch20.f1"><img src="./fig_060_fig60.png" alt="图 20.1：带质量门的 Prompt 链。每一步都是独立的 LLM 调用。质量门可以基于 LLM，也可以基于程序逻辑。" loading="lazy" decoding="async"><figcaption><span class="hh-tag">图 20.1：</span>带质量门的 Prompt 链。每一步都是独立的 LLM 调用。质量门可以基于 LLM，也可以基于程序逻辑。</figcaption></figure>

适用场景：天然顺序化的任务——内容生成、数据转换、多阶段分析。

关键优势：每一步可以使用不同的 Prompt、模型或温度。中间结果可检查、可调试。

### 路由（Routing）

由分类器（LLM 或传统方法）检查输入，并分派到专门的处理器。

<figure id="ch20.f2"><img src="./fig_061_fig61.png" alt="图 20.2：路由模式：输入被分类一次，然后由专门的处理器处理。" loading="lazy" decoding="async"><figcaption><span class="hh-tag">图 20.2：</span>路由模式：输入被分类一次，然后由专门的处理器处理。</figcaption></figure>

适用场景：不同任务类型对应不同的最佳 Prompt、工具或模型。如客服分诊、多模态输入处理。

### 并行化（Parallelization）

多个 LLM 调用并发运行，由一个程序层合并它们的输出。可分为两个子模式：

<ul><li>分段（Sectioning，fan-out）：将输入划分为互不相交的块并独立处理——例如对一个代码库同时运行安全、性能和风格检查。</li><li>投票（Voting，冗余）：用不同随机种子或温度对同一 Prompt 发起 <math alttext="N" display="inline"><semantics><mi>N</mi></semantics></math> 次调用，然后通过多数投票 [368]、奖励模型打分或 LLM-as-judge 选出最佳结果。</li></ul>

### Orchestrator-Workers

在该模式下，由 LLM 自身决定如何拆分工作。一个 Orchestrator 模型分析任务、产出子任务计划、将每个子任务分派给 Worker LLM（可能使用不同的 Prompt 或工具），最终将它们的输出合并为一致的结果。与并行化的关键区别在于：分解逻辑由模型生成，而非硬编码。

<figure id="ch20.f3"><img src="./fig_062_fig62.png" alt="图 20.3：Orchestrator-workers：LLM 决定如何分解任务，并综合各 Worker 的结果。" loading="lazy" decoding="async"><figcaption><span class="hh-tag">图 20.3：</span>Orchestrator-workers：LLM 决定如何分解任务，并综合各 Worker 的结果。</figcaption></figure>

适用场景：开放式问题，子任务的数量和性质无法在设计时穷举——例如「重构这个代码库」需要先理解依赖图，再决定要修改哪些文件。

### Evaluator-Optimizer

一个双模型反馈循环 [246]：生成器产出候选输出，由独立的评估器依据显式标准对其打分。若得分低于阈值，则将评估器的评论追加到生成器的上下文中，循环往复，直到达到质量标准或耗尽重试预算。

<figure id="ch20.f4"><img src="./fig_063_fig63.png" alt="图 20.4：Evaluator-optimizer：无需训练的迭代式精化。" loading="lazy" decoding="async"><figcaption><span class="hh-tag">图 20.4：</span>Evaluator-optimizer：无需训练的迭代式精化。</figcaption></figure>

适用场景：具有明确质量标准的任务——必须通过测试的代码、必须保留语义的翻译、必须符合风格指南的写作。

## 自主 Agent 模式

这些模式把执行流的控制权交给 LLM 本身。

### ReAct（Reason + Act）

最基础的 Agent 模式 [409]。LLM 在思考（内部推理）、行动（工具调用）和观察（处理结果）之间循环交替，直到产出最终答案。

### 规划型 Agent（Planning Agents）

Agent 在执行前生成显式计划，并可在执行过程中修订计划 [364]。

<div class="hh-table" id="ch20.t1"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>策略</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>再规划</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>特征</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">Plan-then-Execute</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>从不</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>简单；对意外结果脆弱</span></span></td></tr><tr><th class="ltx_align_left">Adaptive</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>失败时</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>仅在某步失败时再规划；成本适中</span></span></td></tr><tr><th class="ltx_align_left">Continuous</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>每一步</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>每次观察后完整重估；昂贵但鲁棒</span></span></td></tr><tr><th class="ltx_align_left">Hierarchical</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>子计划完成时</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>高层计划固定；子计划动态生成</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">表 20.1：</span>规划策略对比</p></div>

### 反思与自我批判（Reflection and Self-Critique）

Agent 暂停下来评估自身轨迹并纠正方向：

<ol><li>输出校验：「这样正确吗？有没有遗漏？」</li><li>轨迹回顾：回顾最近 <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 步，找出错误或低效之处。</li><li>策略修订：重新审视整体思路（「我是否在解决正确的问题？」）。</li></ol>

### 工具调用模式（Tool-Use Patterns）

Agent 调用工具的方式会显著影响其可靠性、延迟和成本。已有五种经典模式 [315]：

<div class="hh-table" id="ch20.t2"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>模式</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>描述</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>示例</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">单轮</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>每次 LLM 响应一次工具调用</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>带搜索的简单问答</span></span></td></tr><tr><th class="ltx_align_left">多工具</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>单次响应中发起多个并行工具调用</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>搜索 + 计算 + 格式化</span></span></td></tr><tr><th class="ltx_align_left">顺序</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>工具输出作为下一次工具调用的输入</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>搜索 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo><span></span></semantics></math> 读取 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo><span></span></semantics></math> 抽取</span></span></td></tr><tr><th class="ltx_align_left">嵌套</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>工具调用触发另一个 Agent</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>代码 Agent 调用 test-runner</span></span></td></tr><tr><th class="ltx_align_left">回退</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>首选工具失败；尝试替代方案</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>API <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo><span></span></semantics></math> 抓取 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo><span></span></semantics></math> 缓存</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">表 20.2：</span>工具调用模式</p></div>

最简模式：模型发起一次工具调用、收到结果，然后产出最终答案。足以应对事实查询、单位换算或单次 API 查询。脚手架恰好进行两次 LLM 调用（一次决定使用哪个工具，一次综合结果）。

现代 API（OpenAI、Anthropic）允许模型在单次响应中请求多个工具调用。脚手架并发执行它们，并一并返回所有结果。这大幅降低了那些需要从多个来源获取独立信息的任务的延迟——例如同时获取股价、天气和日程。关键约束是：这些工具必须相互 <em>独立</em>（没有任何工具的输出需要作为另一个工具的输入）。

每个工具的输出作为下一个工具的输入，形成一条数据流水线。模型根据前一个结果决定下一个工具。在研究工作流中很常见：<code>search</code><math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math><code>fetch_page</code><math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math><code>extract_data</code><math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math><code>analyze</code>。脚手架必须跟踪不断增长的上下文，并可能需要摘要中间结果以保持在预算之内。

一次工具调用会唤起一个完全独立的 Agent——它有自己的 Prompt、工具和上下文。父 Agent 将子 Agent 视为黑盒函数。这实现了专门化：研究 Agent 把代码执行委托给编码 Agent，后者可以访问 sandbox 和测试运行器。蜂群（Swarm）模式 [276] 通过专门化 Agent 之间的交接对此做了泛化。

脚手架按优先级顺序尝试工具：如果首选工具失败（超时、限流、API 错误），它会自动回退到替代方案。模型无需感知回退逻辑——脚手架会透明地处理。示例：主搜索 API <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 备用搜索 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 缓存结果 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 告知模型搜索不可用。

## 设计原则

以下原则提炼自 Anthropic 关于构建有效 Agent 的指南 [9]，适用于所有模式：

<ol><li>保持简单。使用能奏效的最简架构。只在被证明必要时才增加复杂性。能解决问题的 Prompt 链，永远优于可能解决问题的多 Agent 系统。</li><li>透明优先于取巧。每一步都应可检查。避免隐藏状态或隐式推理。当 Agent 失败时，你需要理解<em>为什么</em>——不透明的架构会让调试变得不可能。</li><li>提供良好的工具。文档完善、类型明确、报错信息清晰的工具是效力倍增器。描述含糊的工具会被误用；而模式精确、附有使用指引的工具会被正确选择。</li><li>为失败做规划。每次工具调用都可能失败。在脚手架层构建重试逻辑、回退和优雅降级，这样模型就不必去推理基础设施故障。</li><li>使用结构化输出。约束生成（JSON schema、function calling）可以避免解析失败。产出需要正则解析的自由文本的 Agent 是脆弱的；产出经过校验的 JSON 的 Agent 则是鲁棒的。</li><li>用多样化输入进行测试。Agent 的行为比单轮对话更多变。同一个 Prompt 在不同运行中可能产生不同的工具调用序列。要用对抗性方式测试，覆盖边界情况、歧义请求和畸形输入。</li></ol>

## 模式选择指南

选择正确的模式取决于三个因素：(1) 任务结构的可预测程度，(2) 在延迟和成本上你能承受多少次 LLM 调用，(3) 质量是否要求迭代。把下表当作决策矩阵——从最上面（最简单）开始，只有当更简单的模式确实失败时才向下移动。

<div class="hh-table" id="ch20.t3"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>模式</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>复杂度</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>LLM 调用次数</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>最适合</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">Prompt 链</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>低</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="N" display="inline"><semantics><mi>N</mi><span></span></semantics></math>（固定）</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>顺序化任务、内容流水线</span></span></td></tr><tr><th class="ltx_align_left">路由</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>低</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>1 + 1</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>多类型输入、分诊</span></span></td></tr><tr><th class="ltx_align_left">并行化</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>低</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="N" display="inline"><semantics><mi>N</mi><span></span></semantics></math>（并行）</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>独立子任务、投票</span></span></td></tr><tr><th class="ltx_align_left">Orchestrator-workers</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>中</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>可变</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>分解方式未知</span></span></td></tr><tr><th class="ltx_align_left">Evaluator-optimizer</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>中</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>2–10（循环）</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>质量至上的输出</span></span></td></tr><tr><th class="ltx_align_left">ReAct</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>中</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>3–25（循环）</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>通用工具使用、探索</span></span></td></tr><tr><th class="ltx_align_left">规划型 Agent</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>高</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>5–50+</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>长时程、多步骤任务</span></span></td></tr><tr><th class="ltx_align_left">反思</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>高</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>+50% 开销</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>首次尝试常常失败的任务</span></span></td></tr><tr><th class="ltx_align_left">多 Agent</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>高</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>很多</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>复杂领域、专门化</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">表 20.3：</span>何时使用每种 Agent 设计模式</p></div>

模式可以组合：一个规划型 Agent 可以在各个步骤中使用 Prompt 链，在其评审阶段内使用 evaluator-optimizer，并用路由把子任务分派给专家。真正的艺术在于知道何时停止增加层次。

<aside class="hh-next">
<p class="hh-next-label">下一篇</p>
<p class="hh-next-title">智能体 AI 漫游指南 · 第 21 章 Agent 环境与基准</p>
<p class="hh-next-hint">在「智能体 AI 漫游指南」分类下继续阅读。</p>
</aside>

</div>
