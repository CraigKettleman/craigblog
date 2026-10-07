---
title: "第 10 章 SFT 最佳实践与技巧"
slug: "hitchhiker-agentic-ai-10-sft-best-practices-and-techniques"
lang: "zh"
date: "2026-09-16T00:11:00.000Z"
draft: false
featured: false
categories:
  - "hitchhiker-agentic-ai"
cover: "./cover.png"
description: "监督微调（Supervised Fine-Tuning, SFT）是 RLHF 流水线的基础。SFT 模型的质量决定了 RL 所能达到的上限：RL 可以精炼和提升某种行为，但无法可靠地引入 SFT 模型中完全不存在的行为。本节涵盖高效 SFT 的关键技术。"
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

监督微调（Supervised Fine-Tuning, SFT）是 RLHF 流水线的基础。SFT 模型的质量决定了 RL 所能达到的上限：RL 可以精炼和提升某种行为，但无法可靠地引入 SFT 模型中完全不存在的行为。本节涵盖高效 SFT 的关键技术。

## 序列打包以提升效率

序列打包将多个短样本拼接为长度为 <code>max_seq_length</code> 的单一序列，并以 EOS Token 分隔。Attention mask 确保不同样本的 Token 之间不会相互 attend：

<ol><li>按长度对样本排序（可选，可提升打包效率）。</li><li>贪心地将样本打包进大小为 <code>max_seq_length</code> 的 bin。</li><li>使用块对角（block-diagonal）Attention mask，防止跨样本 attention。</li><li>只在非 padding Token 上计算 Loss。</li></ol>

## Chat 模板与格式化

ChatML 是使用最广泛的 chat 模板：

```python
# ChatML format
template = """<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
{assistant_message}<|im_end|>"""
```

Llama 3 使用一种带特殊 Token 的不同模板：

```python
# Llama 3 format
template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{assistant_message}<|eot_id|>"""
```

## 仅对补全部分进行 mask

## 多任务 SFT 的数据混合策略

按各数据集的大小比例从中采样：

<div class="hh-equation" id="ch10.ex1"><math alttext="p_{k}=\frac{N_{k}}{\sum_{j=1}^{K}N_{j}}," display="block"><semantics><mrow><mrow><msub><mi>p</mi><mi>k</mi></msub><mo>=</mo><mfrac><msub><mi>N</mi><mi>k</mi></msub><mrow><msubsup><mo>∑</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mi>K</mi></msubsup><msub><mi>N</mi><mi>j</mi></msub></mrow></mfrac></mrow><mo>,</mo></mrow></semantics></math></div>

其中 <math alttext="N_{k}" display="inline"><semantics><msub><mi>N</mi><mi>k</mi></msub></semantics></math> 是数据集 <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 中的样本数量。这是大多数框架的默认做法，当各数据集质量相近时效果良好。

应用温度 <math alttext="T" display="inline"><semantics><mi>T</mi></semantics></math> 来平滑各比例：

<div class="hh-equation" id="ch10.ex2"><math alttext="p_{k}\propto N_{k}^{1/T}." display="block"><semantics><mrow><mrow><msub><mi>p</mi><mi>k</mi></msub><mo>∝</mo><msubsup><mi>N</mi><mi>k</mi><mrow><mn>1</mn><mo>/</mo><mi>T</mi></mrow></msubsup></mrow><mo lspace="0em">.</mo></mrow></semantics></math></div>

<math alttext="T=1" display="inline"><semantics><mrow><mi>T</mi><mo>=</mo><mn>1</mn></mrow></semantics></math>：按比例混合。<math alttext="T\to\infty" display="inline"><semantics><mrow><mi>T</mi><mo stretchy="false">→</mo><mi mathvariant="normal">∞</mi></mrow></semantics></math>：均匀混合。<math alttext="T&lt;1" display="inline"><semantics><mrow><mi>T</mi><mo>&lt;</mo><mn>1</mn></mrow></semantics></math>：过采样大数据集。<math alttext="T&gt;1" display="inline"><semantics><mrow><mi>T</mi><mo>&gt;</mo><mn>1</mn></mrow></semantics></math>：过采样小数据集。

按估计质量对数据集加权（例如参考模型下的困惑度（Perplexity）、人工质量评分）：

<div class="hh-equation" id="ch10.ex3"><math alttext="p_{k}\propto N_{k}\cdot q_{k}," display="block"><semantics><mrow><mrow><msub><mi>p</mi><mi>k</mi></msub><mo>∝</mo><mrow><msub><mi>N</mi><mi>k</mi></msub><mo lspace="0.222em" rspace="0.222em">⋅</mo><msub><mi>q</mi><mi>k</mi></msub></mrow></mrow><mo>,</mo></mrow></semantics></math></div>

其中 <math alttext="q_{k}" display="inline"><semantics><msub><mi>q</mi><mi>k</mi></msub></semantics></math> 是数据集 <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 的质量得分。

## 当 SFT 反而有害——灾难性遗忘与对齐税

当 LLM 依次经历各训练阶段——预训练 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 继续预训练 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> SFT <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> RLHF/DPO——性能退化常常在标准基准上显现。驱动这些退化的是两种 本质不同的现象，把二者混为一谈会导致错误的缓解策略。

### 灾难性遗忘（结构性擦除）

症状：

<ul><li>在微调数据未覆盖的任务上完全失效（例如模型在 chat 数据上 SFT 后忘记了如何做数学）</li><li>语言多样性丧失——模型只按微调分布的狭窄风格生成</li><li>在微调期间未被强化的知识上，事实准确性下降</li><li>仅用英文 SFT 后多语言能力退化</li></ul>

机制成因——Fisher 信息视角：任务 A 的 Fisher 信息矩阵 <math alttext="F" display="inline"><semantics><mi>F</mi></semantics></math> 指出哪些参数对 <math alttext="\mathcal{D}_{A}" display="inline"><semantics><msub><mi>𝒟</mi><mi>A</mi></msub></semantics></math>「重要」：

<div class="hh-equation" id="ch10.e2"><math alttext="F=\mathbb{E}_{x\sim\mathcal{D}_{A}}\!\left[\nabla_{\theta}\log\pi_{\theta}(x)\,\nabla_{\theta}\log\pi_{\theta}(x)^{T}\right]" display="block"><semantics><mrow><mi>F</mi><mo>=</mo><mrow><msub><mi>𝔼</mi><mrow><mi>x</mi><mo>∼</mo><msub><mi>𝒟</mi><mi>A</mi></msub></mrow></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo>[</mo><mrow><msub><mo>∇</mo><mi>θ</mi></msub><mo lspace="0.167em" rspace="0em">​</mo><mi>log</mi><mo lspace="0.167em" rspace="0em">​</mo><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow><mo lspace="0.337em" rspace="0em">​</mo><msub><mo>∇</mo><mi>θ</mi></msub><mo lspace="0.167em" rspace="0em">​</mo><mi>log</mi><mo lspace="0.167em" rspace="0em">​</mo><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><msup><mrow><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow><mi>T</mi></msup></mrow><mo>]</mo></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(10.2)</span></div>

具有高 Fisher 特征值的参数对任务 A 至关重要。在任务 B 上进行无约束的梯度下降会完全忽略这些特征值——<math alttext="\Delta\theta" display="inline"><semantics><mrow><mi mathvariant="normal">Δ</mi><mo lspace="0em" rspace="0em">​</mo><mi>θ</mi></mrow></semantics></math> 沿着 <math alttext="\nabla\mathcal{L}_{B}" display="inline"><semantics><mrow><mo rspace="0.167em">∇</mo><msub><mi>ℒ</mi><mi>B</mi></msub></mrow></semantics></math> 移动，而不论它是否会破坏 <math alttext="\mathcal{L}_{A}" display="inline"><semantics><msub><mi>ℒ</mi><mi>A</mi></msub></semantics></math> 的高 Fisher 方向。

### 对齐税（行为约束）

对齐税是一种 有意为之、预期之内的权衡：模型的原始能力（无约束生成、最大推理带宽）会下降，因为策略被约束为产生安全、格式良好、符合偏好的输出。

机制：在 DPO/PPO 期间，策略 <math alttext="\pi_{\theta}" display="inline"><semantics><msub><mi>π</mi><mi>θ</mi></msub></semantics></math> 因偏离参考 <math alttext="\pi_{\text{ref}}" display="inline"><semantics><msub><mi>π</mi><mtext>ref</mtext></msub></semantics></math> 而通过 KL 散度受到惩罚：

<div class="hh-equation" id="ch10.e3"><math alttext="r_{\text{implicit}}(x,y)=\beta\log\frac{\pi_{\theta}(y|x)}{\pi_{\text{ref}}(y|x)}" display="block"><semantics><mrow><mrow><msub><mi>r</mi><mtext>implicit</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mi>β</mi><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mfrac><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>y</mi><mo lspace="0em" rspace="0em">|</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mi>π</mi><mtext>ref</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>y</mi><mo lspace="0em" rspace="0em">|</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow></mfrac></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(10.3)</span></div>

这条缰绳约束了模型的 输出分布——它无法探索偏离参考过远的高方差推理路径。知识并 未被抹除，只是被 <em>抑制</em>。模型仍然「知道」答案，但其分布被压平，趋向安全、通用的回应。

症状：

<ul><li>过度拒答（对无害提问也回答「我无法帮你处理这个」）</li><li>风格僵硬——大量模糊限定词、过多免责说明、冗长的安全声明</li><li>原始能力基准（MMLU、HumanEval）得分下降，而偏好基准（MT-Bench、AlpacaEval）得分提升</li><li>产生复杂、高熵输出的能力下降（创造性写作、新颖算法）</li></ul>

### 对比分类

<div class="hh-table" id="ch10.t1"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>维度</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>灾难性遗忘</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>对齐税</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left"><span>有意性</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>非有意（优化过程的副产物）</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>预期之内的权衡（为安全性/有用性而有意付出）</span></span></td></tr><tr><th class="ltx_align_left"><span>参数状态</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>先验知识被物理覆盖</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>潜空间分布被约束/截断</span></span></td></tr><tr><th class="ltx_align_left"><span>信息</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>被销毁</span>：权重不再编码该能力</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>被抑制</span>：知识仍然存在，但更难被触发</span></span></td></tr><tr><th class="ltx_align_left"><span>主要发生阶段</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>序贯 SFT、领域继续预训练</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>偏好优化（PPO、DPO、KTO、RLHF）</span></span></td></tr><tr><th class="ltx_align_left"><span>主要症状</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>基线能力完全失效</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>过度拒答、风格僵硬、原始基准得分下降</span></span></td></tr><tr><th class="ltx_align_left"><span>可逆性</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>不从检查点重新训练则不可逆</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>部分可逆：调整 <math alttext="\beta" display="inline"><semantics><mi>β</mi><span></span></semantics></math>、system prompt 或进行微调</span></span></td></tr><tr><th class="ltx_align_left"><span>检测方法</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>预训练评估集上的困惑度飙升</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>困惑度保持稳定，但能力基准上的胜率下降</span></span></td></tr><tr><th class="ltx_align_left"><span>随模型规模的变化</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>不同规模下表现相似</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>较小的模型付出更大的对齐税</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">表 10.1：</span>灾难性遗忘 vs. 对齐税——完整对比。</p></div>

### 缓解策略

针对灾难性遗忘：

<ol><li>数据回放（Data replay）：将 5–10% 的预训练数据混入 SFT 数据集。确保 Gradient 更新不会完全忽略预训练分布。</li><li>弹性权重巩固（Elastic Weight Consolidation, EWC）[186]：添加正则项 <math alttext="\Omega(\theta)=\frac{\lambda}{2}\sum_{i}F_{i}(\theta_{i}-\theta_{i}^{*})^{2}" display="inline"><semantics><mrow><mrow><mi mathvariant="normal">Ω</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mfrac><mi>λ</mi><mn>2</mn></mfrac><mo lspace="0em" rspace="0em">​</mo><mrow><msub><mo>∑</mo><mi>i</mi></msub><mrow><msub><mi>F</mi><mi>i</mi></msub><mo lspace="0em" rspace="0em">​</mo><msup><mrow><mo stretchy="false">(</mo><mrow><msub><mi>θ</mi><mi>i</mi></msub><mo>−</mo><msubsup><mi>θ</mi><mi>i</mi><mo>∗</mo></msubsup></mrow><mo stretchy="false">)</mo></mrow><mn>2</mn></msup></mrow></mrow></mrow></mrow></semantics></math>，惩罚对原始任务具有高 Fisher 信息的参数发生改变。</li><li>LoRA / 参数高效微调：只训练低秩适配器（占参数量的 <math alttext="&lt;1\%" display="inline"><semantics><mrow><mphantom></mphantom><mo>&lt;</mo><mrow><mn>1</mn><mo>%</mo></mrow></mrow></semantics></math>），保持基座权重完全冻结。这可防止预训练知识被 <em>永久销毁</em>——你随时可以移除适配器并恢复原始模型。然而，在适配器处于激活状态时，组合后的系统 <math alttext="(W_{0}+BA)" display="inline"><semantics><mrow><mo stretchy="false">(</mo><mrow><msub><mi>W</mi><mn>0</mn></msub><mo>+</mo><mrow><mi>B</mi><mo lspace="0em" rspace="0em">​</mo><mi>A</mi></mrow></mrow><mo stretchy="false">)</mo></mrow></semantics></math> 仍可能表现出遗忘：适配器可能使模型的有效行为偏离旧技能。LoRA 保护的是检查点，而不是激活的推理行为。</li><li>保守的学习率：使用 <math alttext="1" display="inline"><semantics><mn>1</mn></semantics></math>–<math alttext="5\times 10^{-6}" display="inline"><semantics><mrow><mn>5</mn><mo lspace="0.222em" rspace="0.222em">×</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>6</mn></mrow></msup></mrow></semantics></math>，并只训练少量 epoch（1–3）。学习率越大，遗忘越快。</li><li>渐进式训练：逐步混合各分布，随时间逐渐提高 SFT 数据比例，而不是突然切换。</li></ol>

针对对齐税：

<ol><li>谨慎调节 <math alttext="\beta" display="inline"><semantics><mi>β</mi></semantics></math>：较低的 <math alttext="\beta" display="inline"><semantics><mi>β</mi></semantics></math> 给模型更多自由（减轻对齐税），但可能牺牲安全性。对于大多数设置，最优值为 <math alttext="\beta\in[0.05,0.3]" display="inline"><semantics><mrow><mi>β</mi><mo>∈</mo><mrow><mo stretchy="false">[</mo><mrow><mn>0.05</mn><mo>,</mo><mn>0.3</mn></mrow><mo stretchy="false">]</mo></mrow></mrow></semantics></math>。</li><li>高质量、多样化的 SFT 数据：对齐税的一部分来自 SFT 收窄了输出分布；更广泛、更多样的 SFT 数据可以减轻这一部分。RL 阶段还会通过 KL 正则项 [280] 施加进一步约束。</li><li>条件对齐（Conditional alignment）：训练模型仅当安全标志位激活时才保持对齐。推理时，在基准测试中关闭约束（仅限研究的技术）。</li><li>Constitutional AI / RLAIF：使用模型生成的反馈来构造更细致的偏好数据，在提升对齐的同时保持能力。</li><li>有针对性的 RL 预算：不要用 RL 过度训练。监控能力基准，当对齐税超过可接受阈值（通常为 MMLU 下降 2–5%）时停止。</li></ol>

## 与 RL 的关系——SFT 质量决定 RL 上限

<ol><li>SFT 数据质量：使用高质量、多样化的数据。少量高质量数据优于大量低质量数据。</li><li>SFT 数据覆盖度：确保 SFT 数据覆盖你希望用 RL 提升的任务。如果某个任务不在 SFT 数据中，RL 将很难奏效。</li><li>SFT 训练时长：不要过度训练 SFT 模型。过度训练会降低多样性，使 RL 的探索更加困难。</li><li>预热：在 RL 之前，考虑用任务特定数据做一次短暂的 SFT 预热，即使基座模型已经过指令微调。</li></ol>

<aside class="hh-next">
<p class="hh-next-label">下一篇</p>
<p class="hh-next-title">第 11 章 大规模系统架构与基础设施</p>
<p class="hh-next-hint">在「智能体 AI 漫游指南」分类下继续阅读。</p>
</aside>

</div>
