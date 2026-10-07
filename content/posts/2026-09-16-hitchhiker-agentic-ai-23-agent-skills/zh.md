---
title: "第 23 章 Agent Skills"
slug: "hitchhiker-agentic-ai-23-agent-skills"
lang: "zh"
date: "2026-09-16T00:24:00.000Z"
draft: false
featured: false
categories:
  - "hitchhiker-agentic-ai"
cover: "./cover.png"
description: "随着 Agent 从单体式的 prompt 加工具系统演化为模块化架构，一个关键的设计问题浮现出来：Agent 的能力应如何被组织、发现与组合？答案越来越收敛到 skill（技能）这一概念——一些离散、可复用的行为单元，可以在不重新训练的前提下被加载、组合与替换。"
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

随着 Agent 从单体式的 prompt 加工具系统演化为模块化架构，一个关键的设计问题浮现出来：<em>Agent 的能力应如何被组织、发现与组合？</em>答案越来越收敛到 skill（技能）这一概念——一些离散、可复用的行为单元，可以在不重新训练的前提下被加载、组合与替换。

这一思路由 Voyager [362] 推广开来：该工作展示了一个在 Minecraft 中运行的 LLM Agent 可以不断积累一个可执行代码 skill 的库，每个 skill 都经过验证并被存储以供后续复用。同样的原则也适用于生产级 Agent：skill 以可组合、可版本化的形式封装领域专长，其可扩展性远超任何单个 prompt 所能容纳。Skill 通常会包装 MCP server（见第 第 22 章 模型上下文协议（Model Context Protocol, MCP） 章）以获得工具访问能力，将 skill 抽象与标准化工具层连接起来。

## 什么是 Skill？

一个 skill 是一个自包含的能力模块，赋予 Agent 在某个特定领域或任务上的专长。与仅暴露单个函数的原始 tool 不同，一个 skill 涵盖：

<ul><li>系统 prompt 增强：注入到 Agent 上下文中的领域特定指令、约束与人设要素。</li><li>Tool 绑定：该 skill 所需的一个或多个 tool（API、MCP server、本地命令）。</li><li>知识：Agent 正确执行该 skill 所需的参考资料、示例或 few-shot 演示。</li><li>工作流逻辑：引导 Agent 完成复杂任务的多步骤流程、决策树或条件流。</li><li>安全护栏：该 skill 特有的安全约束、输出格式要求与校验规则。</li></ul>

## Skill 架构模式

### 静态 Skill 加载

最简单的模式：根据配置在 Agent 初始化时加载 skill。Agent 始终可以访问其全部 skill。

```python
# Pseudocode -- framework-agnostic pattern
agent = Agent(
    model="claude-sonnet-4-20250514",
    skills=["code-review", "documentation", "testing"],
    # Each skill adds prompts, tools, and knowledge to the agent
)
```

优点：简单、可预测、低延迟。<br>

缺点：未被使用的 skill 会浪费上下文窗口；无法扩展到数百个 skill。

### 动态 Skill 发现

Agent 根据当前任务选择激活哪些 skill。一个 skill 路由器（通常是一个轻量分类器或基于 embedding 的匹配器）来判断相关性：

```python
# Pseudocode -- framework-agnostic pattern
relevant_skills = skill_router.match(
    user_request=message,
    available_skills=skill_registry,
    max_skills=3
)
agent.activate(relevant_skills)
```

优点：可扩展到大型 skill 库；上下文效率高。<br>

缺点：路由错误可能漏掉相关 skill；引入额外延迟。

### 层级化 Skill 组合

Skill 可以依赖于其他 skill，从而构成一个有向无环图（DAG）。一个高层 skill（例如“部署应用”）可能会调用一些子 skill（“跑测试”、“构建 Docker 镜像”、“更新 DNS”）：

<ul><li>Skill 显式声明其依赖</li><li>编排器在执行前解析依赖图</li><li>子 skill 可被多个父 skill 共享</li></ul>

## 案例研究：Anthropic 的 Agent 设计

Anthropic 对 Agent 架构的方法 [9] 是关于生产环境中基于 skill 的 Agent 设计最清晰的阐述之一。其设计哲学强调 简单优于复杂、可组合的构建块优于单体框架。（这些模式也会在第 第 20 章 Agent 设计模式 章从编排视角进行讨论。）

### 核心原则

<ol><li>从最简单的方案起步。在尝试更简单的方案（单次 LLM 调用、检索 + 生成）并确认其不足之前，不要急于使用 Agent 化模式。</li><li>Workflow 与 Agent 的区分。Anthropic 将以下两者加以区分：<span class="hh-tag">•</span>Workflows：对 LLM 调用进行预先编排——确定性的控制流，在特定节点包含 LLM 步骤。更可预测、更易调试。<span class="hh-tag">•</span>Agents：由 LLM 动态决定下一步该做什么——工具选择、迭代次数与停止条件都由模型驱动。更灵活，但更难控制。</li><li>以增强型 LLM 作为原子单元。原语从来不是裸模型——它总是与其检索源、可调用工具和持久化记忆捆绑在一起的模型。这一组合单元在实践中就是“配备了 skill 的模型”。</li></ol>

### 构建块模式

Anthropic 总结了五种可组合的 workflow 模式，可作为 skill 模板使用：

<div class="hh-table" id="ch23.t1"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>模式</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>机制</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>适用场景</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left"><span>Prompt Chaining</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>顺序的 LLM 调用，每一步的输出作为下一步的输入。步骤之间设有关卡，校验中间结果。</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>具有清晰分解的多步骤转换任务</span></span></td></tr><tr><th class="ltx_align_left"><span>Routing</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>一个分类器或 LLM 根据任务类型将输入分发给专门的处理者（skill）。</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>需要不同领域专长的多个独立任务类别</span></span></td></tr><tr><th class="ltx_align_left"><span>Parallelization</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>多个 LLM 调用同时进行——可以是分段（拆分任务），也可以是投票（同一任务、汇总结果）。</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>相互独立的子任务；或通过共识获得置信度</span></span></td></tr><tr><th class="ltx_align_left"><span>Orchestrator–Workers</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>一个中心 LLM 将任务拆分为子任务，分派给 worker LLM，再综合结果。</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>子任务难以事先预测的复杂任务</span></span></td></tr><tr><th class="ltx_align_left"><span>Evaluator–Optimizer</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>一个 LLM 生成，另一个 LLM 评估；反复迭代直至达到质量阈值。</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>具有明确质量标准的任务（代码、写作）</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">表 23.1：</span>Anthropic 提出的可组合 Agent 模式。</p></div>

### 增强型 LLM

在 Anthropic 的框架中，基本单元不是裸模型，而是 增强型 LLM：

<div class="hh-equation" id="ch23.ex1"><math alttext="\text{Augmented LLM}=\text{Model}+\text{Retrieval}+\text{Tools}+\text{Memory}" display="block"><semantics><mrow><mtext>Augmented LLM</mtext><mo>=</mo><mrow><mtext>Model</mtext><mo>+</mo><mtext>Retrieval</mtext><mo>+</mo><mtext>Tools</mtext><mo>+</mo><mtext>Memory</mtext></mrow></mrow></semantics></math></div>

这直接对应到 skill 这一概念：每个 skill 都配置了模型在特定任务中可以访问哪些检索源、tool 与记忆存储。skill 边界定义了模型在特定一次调用中 <em>能看到什么、能做什么</em>。

### 实际影响

<ul><li>保持 Agent 循环简单：避免对控制流进行过度工程设计。让模型自行决定。</li><li>投资于 tool 质量：详尽、无歧义的 tool 描述比复杂的路由逻辑更有价值。</li><li>使用结构化输出：强制模型以可解析的格式（JSON、函数调用）输出决策——可减少 skill 执行错误。</li><li>内置恢复能力：Skill 应能优雅地处理错误——以不同参数重试、请求澄清，或升级给人类。</li><li>限制单个 skill 的作用域：试图包办一切的 skill 什么都做不好。窄而定义明确的 skill 比宽泛的 skill 更易于组合。</li></ul>

## Skill 生命周期

<ol><li>发现：系统识别有哪些 skill 可用（注册表、市场、本地定义）。</li><li>选择：根据用户请求匹配并加载相关 skill。</li><li>激活：将 skill 的 prompt、tool 与知识注入 Agent 的上下文。</li><li>执行：Agent 使用该 skill 的能力来完成任务。</li><li>停用：移除 skill 上下文，为后续任务释放上下文窗口空间。</li><li>学习：执行结果可能更新该 skill 的 few-shot 示例，或对路由进行微调。</li></ol>

## Skill 注册表与市场

生产级 skill 系统需要基础设施：

<ul><li>Skill 清单：结构化的描述（名称、能力、所需 tool、输入/输出 schema），支持自动发现与路由。</li><li>版本控制：Skill 会不断演进；Agent 需要固定特定版本以保证可复现性。</li><li>依赖解析：Skill 可能需要特定的 MCP server、API key 或其他 skill。</li><li>权限模型：并非所有 Agent 都应有权访问所有 skill（安全、成本、能力边界）。</li><li>市场：组织可以发布、共享与安装 skill——类似于代码的包管理器。</li></ul>

## Skill vs. 微调

一个自然的问题是：为什么要在运行时注入 skill，而不是对模型进行微调？

<div class="hh-table" id="ch23.t2"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>维度</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Skill（上下文内）</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>微调</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">部署速度</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>即时</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>数小时至数天</span></span></td></tr><tr><th class="ltx_align_left">灵活性</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>运行时替换/组合</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>在训练时固定</span></span></td></tr><tr><th class="ltx_align_left">上下文成本</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>占用上下文窗口</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>运行时代价为零</span></span></td></tr><tr><th class="ltx_align_left">深层行为改变</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>受上下文长度限制</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>深层参数化改变</span></span></td></tr><tr><th class="ltx_align_left">多租户</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>每个用户可用不同的 skill</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>所有用户共用同一模型</span></span></td></tr><tr><th class="ltx_align_left">维护</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>更新文本文件</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>用新数据重新训练</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">表 23.2：</span>用于增添能力的 skill（上下文内）与微调对比。</p></div>

在实践中，这两种方法互为补充：微调提供 <em>基础能力</em>（指令遵循、工具使用格式、推理），而 skill 则在运行时于其上叠加 <em>任务特定的专长</em>。

<aside class="hh-next">
<p class="hh-next-label">下一篇</p>
<p class="hh-next-title">第 24 章 智能体到智能体通信（Agent-to-Agent, A2A）</p>
<p class="hh-next-hint">在「智能体 AI 漫游指南」分类下继续阅读。</p>
</aside>

</div>
