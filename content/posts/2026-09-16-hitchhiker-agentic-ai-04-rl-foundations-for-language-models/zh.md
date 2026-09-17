---
title: "智能体 AI 漫游指南 · 第 4 章 大语言模型的强化学习基础"
slug: "hitchhiker-agentic-ai-04-rl-foundations-for-language-models"
lang: "zh"
date: "2026-09-16T00:05:00.000Z"
draft: false
featured: false
categories:
  - "hitchhiker-agentic-ai"
cover: "./cover.png"
description: "监督微调（Supervised Fine-Tuning, SFT）教模型模仿示例，但模仿存在天花板：模型永远无法超越其训练数据的质量。强化学习突破了这一壁垒。…"
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
<p class="hh-part-title">第 II 部分 面向大语言模型的强化学习方法</p>
</aside>

监督微调（Supervised Fine-Tuning, SFT）教模型模仿示例，但模仿存在天花板：模型永远无法超越其训练数据的质量。强化学习突破了这一壁垒。通过生成新文本、接收奖励（reward）反馈，并朝着获得更高奖励的行为更新，经过 RL 训练的模型能够<em>发现</em>任何人类示范者都未曾写出的策略——产出更有帮助、更准确、且更契合人类偏好的输出 [280]。

这是每一款前沿模型背后的机制：GPT-4 [274]、Claude、Llama-3 [118] 和 DeepSeek-R1 [69] 都在 SFT 之后施加 RL，作为把一个能力强但缺乏导向的模型转化为对齐助手的关键一步。

## 大语言模型 RL 的两大范式

面向语言模型的 RL 方法可大致划分为两类范式，分别适用于不同的目标：

将 RL 应用于大语言模型最初的动机是对齐——让模型变得有帮助、无害且诚实。基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）[280, 442, 57] 利用人类两两比较的判断（「哪个回答更好？」）来训练奖励模型，然后优化策略（policy）以最大化这一学到的奖励。直接偏好优化（DPO）[302] 通过彻底取消奖励模型来简化这一过程，将偏好直接转化为有监督 loss。两种方法都能产出能够遵循指令且尊重安全约束的对齐助手。

近期，RL 不仅被用于对齐，也被用于教授新能力——尤其是推理、数学和代码生成。此处的奖励不再来自人类偏好，而来自可验证的结果：模型是否给出了正确答案？代码是否通过了所有测试？DeepSeek-R1 [69] 证明，配合基于规则的奖励（格式正确性 + 答案准确性）的组相对策略优化（GRPO）能够训练模型在<em>完全不使用人类偏好数据</em>的情况下发展出复杂的思维链（Chain-of-Thought, CoT）推理。这一范式——可验证奖励的强化学习（Reinforcement Learning from Verifiable Rewards, RLVR）——如今已成为构建推理模型和 agentic 系统的主流路线。

## 将文本生成建模为 MDP

让 RL 得以应用于语言模型的关键洞察，是将自回归生成重述为一个马尔可夫决策过程（MDP）：

形式化地，文本生成的 MDP 定义如下：

<ul><li>状态<math alttext="s_{t}=(x,y_{1},\ldots,y_{t-1})" display="inline"><semantics><mrow><msub><mi>s</mi><mi>t</mi></msub><mo>=</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>,</mo><msub><mi>y</mi><mn>1</mn></msub><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msub><mi>y</mi><mrow><mi>t</mi><mo>−</mo><mn>1</mn></mrow></msub><mo stretchy="false">)</mo></mrow></mrow></semantics></math>：Prompt 与到目前为止已生成的所有 token 的拼接。</li><li>动作<math alttext="a_{t}\in\{1,\ldots,|\mathcal{V}|\}" display="inline"><semantics><mrow><msub><mi>a</mi><mi>t</mi></msub><mo>∈</mo><mrow><mo stretchy="false">{</mo><mn>1</mn><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><mrow><mo stretchy="false">|</mo><mi>𝒱</mi><mo stretchy="false">|</mo></mrow><mo stretchy="false">}</mo></mrow></mrow></semantics></math>：从词表（32K–128K 个选项）中选出下一个 token。</li><li>转移<math alttext="P(s_{t+1}|s_{t},a_{t})" display="inline"><semantics><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>s</mi><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub><mo lspace="0em" rspace="0em">|</mo><mrow><msub><mi>s</mi><mi>t</mi></msub><mo>,</mo><msub><mi>a</mi><mi>t</mi></msub></mrow><mo stretchy="false">)</mo></mrow></mrow></semantics></math>：确定性的——只需追加所选 token。环境无随机性。</li><li>奖励<math alttext="r" display="inline"><semantics><mi>r</mi></semantics></math>：通常仅在生成结束时给出（稀疏）。对 RLHF 而言是奖励模型评分；对 RLVR 而言是最终答案的正确性。</li><li>策略<math alttext="\pi_{\theta}(a_{t}|s_{t})" display="inline"><semantics><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>a</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">|</mo><msub><mi>s</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow></semantics></math>：大语言模型的下一 token 概率分布——正是 softmax 输出已经计算出的东西。</li><li>折扣因子<math alttext="\gamma=1.0" display="inline"><semantics><mrow><mi>γ</mi><mo>=</mo><mn>1.0</mn></mrow></semantics></math>：Episode 是有限的（即一条回答），因此无需折扣。</li></ul>

这种映射之所以强大，是因为大语言模型本身<em>就已经是</em>一个策略——其 softmax 输出为每一个状态定义了 <math alttext="\pi_{\theta}(a_{t}|s_{t})" display="inline"><semantics><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>a</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">|</mo><msub><mi>s</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow></semantics></math>。我们无需另行构建一个策略网络；只需调整权重 <math alttext="\theta" display="inline"><semantics><mi>θ</mi></semantics></math>，让模型为能获得更高奖励的 token 序列赋予更高概率。

## RLHF 流水线

经典的 RLHF 流水线 [280] 包含四个阶段：

<ol><li>监督微调（SFT）：在高质量示例上训练 base 模型，得到一个能够遵循指令的策略 <math alttext="\pi_{\text{SFT}}" display="inline"><semantics><msub><mi>π</mi><mtext>SFT</mtext></msub></semantics></math>。</li><li>奖励模型训练：收集人类偏好比较（对同一 prompt 满足 <math alttext="y_{w}\succ y_{l}" display="inline"><semantics><mrow><msub><mi>y</mi><mi>w</mi></msub><mo>≻</mo><msub><mi>y</mi><mi>l</mi></msub></mrow></semantics></math>），并使用 Bradley-Terry 目标训练奖励模型 <math alttext="R_{\phi}(x,y)" display="inline"><semantics><mrow><msub><mi>R</mi><mi>ϕ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math>。</li><li>RL 优化：以奖励模型作为信号，通过 PPO 或 GRPO 优化策略，并对 <math alttext="\pi_{\text{SFT}}" display="inline"><semantics><msub><mi>π</mi><mtext>SFT</mtext></msub></semantics></math> 施加 KL 约束。</li><li>评估与迭代：评估对齐后的模型，收集新的失败案例，并进行迭代。</li></ol>

对于 RLVR（推理/agentic 训练），阶段 1–2 被替换：SFT 模型在推理轨迹上训练，奖励模型被替换为验证器（例如检查数学正确性）。阶段 3 不变——使用 PPO 或 GRPO 针对奖励信号进行优化。

## 本部分路线图

接下来的各章将构建完整的 LLM RL 工具箱：

<ol><li>PPO（第 5 章）——截断的代理目标、用于优势估计的 GAE（广义优势估计）、评论家网络，以及完整的 RLHF 训练循环。GPT-4 和 Claude 背后的主力方法。</li><li>DPO（第 6 章）——通过将偏好转化为对比式监督 loss 来完全绕开 RL。比在线 RL 更简单，但灵活性更低。</li><li>GRPO（第 7 章）——DeepSeek 的无评论家算法，采用组级奖励归一化。DeepSeek-R1 背后的方法，也是推理模型训练的主流选择。</li><li>偏好优化变体（第 8 章）——Online DPO、KTO、Best-of-N，以及方法选择指导。</li><li>奖励建模（第 9 章）——Bradley-Terry 模型、过程奖励与结果奖励的对比、面向 RLVR 的基于规则的奖励，以及多目标组合。</li><li>SFT 最佳实践（第 10 章）——序列打包、聊天模板、数据混合，以及 SFT 质量如何决定 RL 的天花板。</li><li>系统工程（第 11 章）——大规模分布式训练：并行策略、生成与训练解耦，以及支撑数百块 GPU 的基础设施。</li></ol>

<aside class="hh-next">
<p class="hh-next-label">下一篇</p>
<p class="hh-next-title">智能体 AI 漫游指南 · 第 5 章 PPO——近端策略优化</p>
<p class="hh-next-hint">在「智能体 AI 漫游指南」分类下继续阅读。</p>
</aside>

</div>
