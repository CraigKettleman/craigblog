---
title: "智能体 AI 漫游指南 · 第 9 章 奖励模型训练"
slug: "hitchhiker-agentic-ai-09-reward-model-training"
lang: "zh"
date: "2026-09-16T00:10:00.000Z"
draft: false
featured: false
categories:
  - "hitchhiker-agentic-ai"
cover: "./cover.png"
description: "奖励模型是连接人类偏好与 RL 训练信号的桥梁。训练良好的奖励模型对成功的基于人类反馈的强化学习（RLHF）至关重要；训练不佳的奖励模型会导致奖励作弊（reward hacking）与行为失准。本节涵盖奖励模型的理论基础、实用训练技术以及架构选择。"
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

奖励模型是连接人类偏好与 RL 训练信号的桥梁。训练良好的奖励模型对成功的基于人类反馈的强化学习（RLHF）至关重要；训练不佳的奖励模型会导致奖励作弊（reward hacking）与行为失准。本节涵盖奖励模型的理论基础、实用训练技术以及架构选择。

## Bradley-Terry 模型——完整推导

Bradley-Terry 模型 [29] 是成对偏好学习（pairwise preference learning）的标准概率框架。给定 prompt <math alttext="q" display="inline"><semantics><mi>q</mi></semantics></math> 的两个回答 <math alttext="y_{1}" display="inline"><semantics><msub><mi>y</mi><mn>1</mn></msub></semantics></math> 和 <math alttext="y_{2}" display="inline"><semantics><msub><mi>y</mi><mn>2</mn></msub></semantics></math>，该模型假设：

<div class="hh-equation" id="ch9.ex1"><math alttext="P(y_{1}\succ y_{2}\mid q)=\sigma(r(y_{1},q)-r(y_{2},q))=\frac{e^{r(y_{1},q)}}{e^{r(y_{1},q)}+e^{r(y_{2},q)}}," display="block"><semantics><mrow><mrow><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><msub><mi>y</mi><mn>1</mn></msub><mo>≻</mo><msub><mi>y</mi><mn>2</mn></msub></mrow><mo fence="true" lspace="0em" rspace="0em">∣</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mi>σ</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mn>1</mn></msub><mo>,</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mo>−</mo><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mn>2</mn></msub><mo>,</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mfrac><msup><mi>e</mi><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mn>1</mn></msub><mo>,</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow></msup><mrow><msup><mi>e</mi><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mn>1</mn></msub><mo>,</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow></msup><mo>+</mo><msup><mi>e</mi><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mn>2</mn></msub><mo>,</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow></msup></mrow></mfrac></mrow><mo>,</mo></mrow></semantics></math></div>

其中 <math alttext="r:\mathcal{Y}\times\mathcal{Q}\to\mathbb{R}" display="inline"><semantics><mrow><mi>r</mi><mo lspace="0.278em" rspace="0.278em">:</mo><mrow><mrow><mi>𝒴</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mi>𝒬</mi></mrow><mo stretchy="false">→</mo><mi>ℝ</mi></mrow></mrow></semantics></math> 是标量 Reward 函数，<math alttext="\sigma" display="inline"><semantics><mi>σ</mi></semantics></math> 是 sigmoid 函数。

给定由偏好对组成的数据集 <math alttext="\mathcal{D}=\{(q^{(k)},y_{w}^{(k)},y_{l}^{(k)})\}_{k=1}^{N}" display="inline"><semantics><mrow><mi>𝒟</mi><mo>=</mo><msubsup><mrow><mo stretchy="false">{</mo><mrow><mo stretchy="false">(</mo><msup><mi>q</mi><mrow><mo stretchy="false">(</mo><mi>k</mi><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><msubsup><mi>y</mi><mi>w</mi><mrow><mo stretchy="false">(</mo><mi>k</mi><mo stretchy="false">)</mo></mrow></msubsup><mo>,</mo><msubsup><mi>y</mi><mi>l</mi><mrow><mo stretchy="false">(</mo><mi>k</mi><mo stretchy="false">)</mo></mrow></msubsup><mo stretchy="false">)</mo></mrow><mo stretchy="false">}</mo></mrow><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mi>N</mi></msubsup></mrow></semantics></math>，MLE 目标为：

<div class="hh-equation" id="ch9.ex2"><math alttext="\mathcal{L}_{\text{BT}}(\phi)=-\frac{1}{N}\sum_{k=1}^{N}\log\sigma\!\bigl(r_{\phi}(y_{w}^{(k)},q^{(k)})-r_{\phi}(y_{l}^{(k)},q^{(k)})\bigr)," display="block"><semantics><mrow><msub><mi>ℒ</mi><mtext>BT</mtext></msub><mrow><mo stretchy="false">(</mo><mi>ϕ</mi><mo stretchy="false">)</mo></mrow><mo rspace="0em">=</mo><mo lspace="0em">−</mo><mfrac><mn>1</mn><mi>N</mi></mfrac><munderover><mo movablelimits="false">∑</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mi>N</mi></munderover><mi>log</mi><mpadded width="0.437em"><mi>σ</mi></mpadded><mrow><mo maxsize="1.200em" minsize="1.200em">(</mo><msub><mi>r</mi><mi>ϕ</mi></msub><mrow><mo stretchy="false">(</mo><msubsup><mi>y</mi><mi>w</mi><mrow><mo stretchy="false">(</mo><mi>k</mi><mo stretchy="false">)</mo></mrow></msubsup><mo>,</mo><msup><mi>q</mi><mrow><mo stretchy="false">(</mo><mi>k</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mo>−</mo><msub><mi>r</mi><mi>ϕ</mi></msub><mrow><mo stretchy="false">(</mo><msubsup><mi>y</mi><mi>l</mi><mrow><mo stretchy="false">(</mo><mi>k</mi><mo stretchy="false">)</mo></mrow></msubsup><mo>,</mo><msup><mi>q</mi><mrow><mo stretchy="false">(</mo><mi>k</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mo maxsize="1.200em" minsize="1.200em">)</mo></mrow><mo>,</mo></mrow></semantics></math></div>

其中 <math alttext="r_{\phi}" display="inline"><semantics><msub><mi>r</mi><mi>ϕ</mi></msub></semantics></math> 是由 <math alttext="\phi" display="inline"><semantics><mi>ϕ</mi></semantics></math> 参数化的神经网络。这是一个二元交叉熵 Loss，其中「正」类别对应偏好的回答。

一种常见的扩展是引入 margin <math alttext="m" display="inline"><semantics><mi>m</mi></semantics></math>，以确保胜出与落败的 Reward 之间存在最小间隔：

<div class="hh-equation" id="ch9.ex3"><math alttext="\mathcal{L}_{\text{margin}}=-\frac{1}{N}\sum_{k=1}^{N}\log\sigma\!\bigl(r_{\phi}(y_{w}^{(k)},q^{(k)})-r_{\phi}(y_{l}^{(k)},q^{(k)})-m\bigr)." display="block"><semantics><mrow><msub><mi>ℒ</mi><mtext>margin</mtext></msub><mo rspace="0em">=</mo><mo lspace="0em">−</mo><mfrac><mn>1</mn><mi>N</mi></mfrac><munderover><mo movablelimits="false">∑</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mi>N</mi></munderover><mi>log</mi><mpadded width="0.437em"><mi>σ</mi></mpadded><mrow><mo maxsize="1.200em" minsize="1.200em">(</mo><msub><mi>r</mi><mi>ϕ</mi></msub><mrow><mo stretchy="false">(</mo><msubsup><mi>y</mi><mi>w</mi><mrow><mo stretchy="false">(</mo><mi>k</mi><mo stretchy="false">)</mo></mrow></msubsup><mo>,</mo><msup><mi>q</mi><mrow><mo stretchy="false">(</mo><mi>k</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mo>−</mo><msub><mi>r</mi><mi>ϕ</mi></msub><mrow><mo stretchy="false">(</mo><msubsup><mi>y</mi><mi>l</mi><mrow><mo stretchy="false">(</mo><mi>k</mi><mo stretchy="false">)</mo></mrow></msubsup><mo>,</mo><msup><mi>q</mi><mrow><mo stretchy="false">(</mo><mi>k</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mo>−</mo><mi>m</mi><mo maxsize="1.200em" minsize="1.200em">)</mo></mrow><mo lspace="0em">.</mo></mrow></semantics></math></div>

## 奖励模型架构

该架构由以下部分组成：

<ol><li>骨干网络（Backbone）：一个预训练 LLM（例如 Llama、Mistral），将 prompt-response 对编码为一系列隐藏状态。</li><li>池化（Pooling）：提取最后一个 Token 位置的隐藏状态（对于 decoder-only 模型）或 <code>[CLS]</code> Token（对于 encoder 模型）。</li><li>回归头（Regression head）：线性层 <math alttext="W\in\mathbb{R}^{d\times 1}" display="inline"><semantics><mrow><mi>W</mi><mo>∈</mo><msup><mi>ℝ</mi><mrow><mi>d</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mn>1</mn></mrow></msup></mrow></semantics></math>，将池化后的隐藏状态映射为标量 Reward。</li></ol>

## 奖励模型训练技巧

原始奖励模型的输出可能具有任意尺度和偏移。对 Reward 进行中心化（减去均值）可以稳定 RL 训练：

<div class="hh-equation" id="ch9.ex4"><math alttext="r_{\text{centered}}(y,q)=r_{\phi}(y,q)-\mathbb{E}_{y^{\prime}\sim\pi_{\theta}}[r_{\phi}(y^{\prime},q)]." display="block"><semantics><mrow><mrow><mrow><msub><mi>r</mi><mtext>centered</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>y</mi><mo>,</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mrow><msub><mi>r</mi><mi>ϕ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>y</mi><mo>,</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mo>−</mo><mrow><msub><mi>𝔼</mi><mrow><msup><mi>y</mi><mo>′</mo></msup><mo>∼</mo><msub><mi>π</mi><mi>θ</mi></msub></mrow></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">[</mo><mrow><msub><mi>r</mi><mi>ϕ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msup><mi>y</mi><mo>′</mo></msup><mo>,</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mo stretchy="false">]</mo></mrow></mrow></mrow></mrow><mo lspace="0em">.</mo></mrow></semantics></math></div>

在 TRL 中，这通过 <code>center_rewards_coefficient</code> 参数实现，它会向奖励模型 Loss 中添加一个正则项，惩罚均值非零的 Reward。

已知奖励模型会表现出<em>长度偏差</em>：无论质量如何，它们倾向于给较长的回答分配更高的 Reward。可通过以下方式校正：

<ol><li>长度归一化（Length normalisation）：将 Reward 除以回答长度。</li><li>长度受控训练（Length-controlled training）：将长度作为特征，训练模型使其对长度不变。</li><li>校准（Calibration）：事后回归以剔除长度效应。</li></ol>

在 Bradley-Terry Loss 中加入 margin <math alttext="m" display="inline"><semantics><mi>m</mi></semantics></math>，可确保奖励模型对偏好回答与非偏好回答打出具有显著区分度的分数：

<div class="hh-equation" id="ch9.ex5"><math alttext="\mathcal{L}_{\text{margin}}=\max\!\bigl(0,\;m-(r_{w}-r_{l})\bigr)." display="block"><semantics><mrow><mrow><msub><mi>ℒ</mi><mtext>margin</mtext></msub><mo>=</mo><mrow><mpadded width="1.808em"><mi>max</mi></mpadded><mo>⁡</mo><mrow><mo maxsize="1.200em" minsize="1.200em">(</mo><mn>0</mn><mo>,</mo><mrow><mi>m</mi><mo>−</mo><mrow><mo stretchy="false">(</mo><mrow><msub><mi>r</mi><mi>w</mi></msub><mo>−</mo><msub><mi>r</mi><mi>l</mi></msub></mrow><mo stretchy="false">)</mo></mrow></mrow><mo maxsize="1.200em" minsize="1.200em">)</mo></mrow></mrow></mrow><mo lspace="0em">.</mo></mrow></semantics></math></div>

## 过程奖励模型 vs 结果奖励模型

步骤级标注可以通过以下方式自动生成：

<ol><li>Monte Carlo rollout：对每个中间步骤采样多个补全，将达到正确答案的比例作为该步骤的 Reward。</li><li>LLM 作为评判者（LLM-as-judge）：使用强 LLM 评估每一步。</li><li>形式化验证：对于数学/代码，使用验证器检查每一步。</li></ol>

## RLVR 中基于规则的 Reward

基于可验证奖励的强化学习（Reinforcement Learning from Verifiable Rewards, RLVR）使用确定性的、基于规则的奖励函数，而不是学习得到的奖励模型。这大幅减少了奖励作弊（尽管模型仍可能利用格式技巧、边界情况或测试记忆），也是 DeepSeek-R1 [69] 采用的方法。

## 多目标 Reward ——组合策略

当使用多个奖励信号进行训练时，组合策略会显著影响最终策略。

## 基于列表排序的 Reward

Bradley-Terry 模型处理的是<em>成对</em>偏好（<math alttext="y_{w}\succ y_{l}" display="inline"><semantics><mrow><msub><mi>y</mi><mi>w</mi></msub><mo>≻</mo><msub><mi>y</mi><mi>l</mi></msub></mrow></semantics></math>），但许多实际场景需要同时对多个回答进行排序。列表式（listwise）奖励模型从完整排序中学习，能提供更丰富的训练信号并实现更好的校准。

Plackett-Luce（PL）模型 [291] 是 Bradley-Terry 模型向完整排序的标准扩展。给定 <math alttext="K" display="inline"><semantics><mi>K</mi></semantics></math> 个回答 <math alttext="y_{1},\ldots,y_{K}" display="inline"><semantics><mrow><msub><mi>y</mi><mn>1</mn></msub><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msub><mi>y</mi><mi>K</mi></msub></mrow></semantics></math> 及其排序 <math alttext="\pi" display="inline"><semantics><mi>π</mi></semantics></math>（其中 <math alttext="\pi(1)" display="inline"><semantics><mrow><mi>π</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 为最优）：

<aside class="hh-next">
<p class="hh-next-label">下一篇</p>
<p class="hh-next-title">智能体 AI 漫游指南 · 第 10 章 SFT 最佳实践与技巧</p>
<p class="hh-next-hint">在「智能体 AI 漫游指南」分类下继续阅读。</p>
</aside>

</div>
