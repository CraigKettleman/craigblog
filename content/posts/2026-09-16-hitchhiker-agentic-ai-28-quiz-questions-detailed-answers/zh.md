---
title: "第 28 章 测验题与详细解答"
slug: "hitchhiker-agentic-ai-28-quiz-questions-detailed-answers"
lang: "zh"
date: "2026-09-16T00:29:00.000Z"
draft: false
featured: false
categories:
  - "hitchhiker-agentic-ai"
cover: "./cover.png"
description: "本章提供了一整套题目，用于检验和巩固你对本书所讲内容的理解。每道题目都瞄准一个关键概念、算法或系统设计决策——正是这种知识把表面熟悉与真正的专长区分开来。请将这些题目用于自我检验：先尝试自己作答，再阅读详细解答。…"
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
<p class="hh-part-title">第 VI 部分 自测与参考</p>
</aside>

本章提供了一整套题目，用于检验和巩固你对本书所讲内容的理解。每道题目都瞄准一个关键概念、算法或系统设计决策——正是这种知识把表面熟悉与真正的专长区分开来。请将这些题目用于自我检验：先尝试自己作答，再阅读详细解答。题目从基础概念（LLM 架构、强化学习基础）逐步推进到核心算法（PPO、DPO、GRPO），最终覆盖高级系统设计与智能体（Agent）AI 主题。

## 基础题

<details class="hh-quiz" id="q0a">
<summary class="hh-quiz-question">Q0a：在 decoder-only Transformer 中，attention 机制的作用是什么？为什么它是因果的？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>attention 机制允许每个 Token 关注（即对其表示做加权组合）其他 Token 的表示。在 decoder-only Transformer 中，attention 是 因果的（也称为<em>自回归</em>）：Token <math alttext="t" display="inline"><semantics><mi>t</mi></semantics></math> 只能关注 Token <math alttext="1,\ldots,t" display="inline"><semantics><mrow><mn>1</mn><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><mi>t</mi></mrow></semantics></math>，绝不能关注未来 Token <math alttext="t+1,\ldots,T" display="inline"><semantics><mrow><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><mi>T</mi></mrow></semantics></math>。</p><p>为什么是因果的？因为模型从左到右生成文本。推理时，未来 Token 字面意义上还不存在。训练时的因果掩码模拟了这一约束，使模型学会仅用左侧上下文预测每个 Token。在数学上，attention 矩阵被掩盖：</p><div class="hh-equation"><math alttext="\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_{k}}}+M\right)V" display="block"><semantics><mrow><mrow><mtext>Attention</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>Q</mi><mo>,</mo><mi>K</mi><mo>,</mo><mi>V</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mpadded width="3.720em"><mtext>softmax</mtext></mpadded><mo lspace="0em" rspace="0em">​</mo><mrow><mo>(</mo><mrow><mfrac><mrow><mi>Q</mi><mo lspace="0em" rspace="0em">​</mo><msup><mi>K</mi><mo>⊤</mo></msup></mrow><msqrt><msub><mi>d</mi><mi>k</mi></msub></msqrt></mfrac><mo>+</mo><mi>M</mi></mrow><mo>)</mo></mrow><mo lspace="0em" rspace="0em">​</mo><mi>V</mi></mrow></mrow></semantics></math></div><p>其中对 <math alttext="j&gt;i" display="inline"><semantics><mrow><mi>j</mi><mo>&gt;</mo><mi>i</mi></mrow></semantics></math>（未来位置）有 <math alttext="M_{ij}=-\infty" display="inline"><semantics><mrow><msub><mi>M</mi><mrow><mi>i</mi><mo lspace="0em" rspace="0em">​</mo><mi>j</mi></mrow></msub><mo>=</mo><mrow><mo>−</mo><mi mathvariant="normal">∞</mi></mrow></mrow></semantics></math>，强制将这些 attention 权重置零。</p><p>实践含义：这使得推理时的 KV-cache 优化成为可能——由于过去 Token 的 keys 和 values 永远不变，可被缓存复用，将每个新 Token 的生成代价从 <math alttext="O(T^{2})" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msup><mi>T</mi><mn>2</mn></msup><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 降至 <math alttext="O(T)" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>T</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math>。</p><p><em>复习：第 1 章（LLM 架构与优化方法）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q0b">
<summary class="hh-quiz-question">Q0b：解释 FlashAttention。它解决什么问题，又是怎么解决的？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>标准 attention 会计算完整的 <math alttext="T\times T" display="inline"><semantics><mrow><mi>T</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mi>T</mi></mrow></semantics></math> attention 矩阵，需要 <math alttext="O(T^{2})" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msup><mi>T</mi><mn>2</mn></msup><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 显存，并且是 受显存带宽约束的——GPU 大部分时间都花在 HBM（慢、大）与 SRAM（快、小）之间搬运数据，而不是做真正的计算。</p><p>FlashAttention 的洞见：永远不要在 HBM 中物化完整的 attention 矩阵。相反，把计算切分成可放进 SRAM 的分块，用在线 Softmax（Online Softmax）算法 <em>逐块</em>计算 attention，并只把最终输出写入 HBM。</p><p>关键技术：</p><ol><li>分块（Tiling）：将 Q、K、V 拆成大小为 <math alttext="B_{r}\times B_{c}" display="inline"><semantics><mrow><msub><mi>B</mi><mi>r</mi></msub><mo lspace="0.222em" rspace="0.222em">×</mo><msub><mi>B</mi><mi>c</mi></msub></mrow></semantics></math> 的块，可装入 SRAM</li><li>在线 Softmax：维护一个滑动的 max 和 sum，以增量方式计算 softmax，无需访问完整一行</li><li>重算（Recomputation）：在反向传播时，从 Q、K、V 重新计算 attention（开销小），而不存储 <math alttext="T\times T" display="inline"><semantics><mrow><mi>T</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mi>T</mi></mrow></semantics></math> 矩阵（开销大）</li></ol><p>结果：HBM 显存复杂度为 <math alttext="O(T)" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>T</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math>（而非 <math alttext="O(T^{2})" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msup><mi>T</mi><mn>2</mn></msup><mo stretchy="false">)</mo></mrow></mrow></semantics></math>），墙钟（wall-clock）时间加速 2–4<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>，数值输出完全一致（并非近似）。</p><p><em>复习：第 1–2 章（LLM 架构；系统基础）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q0c">
<summary class="hh-quiz-question">Q0c：从高层次看，SFT、RLHF 与 DPO 有什么区别？分别在什么场景下使用？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<ul><li>SFT（监督微调，Supervised Fine-Tuning）：训练模型去模仿高质量示范。Loss：在精选数据上做下一个 Token 预测。教会模型 <em>格式</em>与 <em>风格</em>。</li><li>RLHF：从人类偏好训练一个 reward 模型，然后用 RL（PPO）针对它优化 policy。模型会探索示范数据之外的空间。教会模型 <em>人类偏好什么</em>。</li><li>DPO（直接偏好优化）：跳过 reward 模型。直接在偏好对 <math alttext="(y_{w},y_{l})" display="inline"><semantics><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>w</mi></msub><mo>,</mo><msub><mi>y</mi><mi>l</mi></msub><mo stretchy="false">)</mo></mrow></semantics></math> 上用对比 loss 优化 policy。目标与 RLHF 相同，但流水线更简单。</li></ul><p>典型流水线：先做 SFT（提供良好起点），再做 RLHF 或 DPO（细化偏好）。仅做 SFT 倾向产出冗长、过度模棱两可的回答。RLHF/DPO 让输出更直接、更贴合人类意图。</p><p>各自适用场景：有黄金标准输出时用 SFT。有偏好对但算力有限时用 DPO。需要极致质量且能承担基础设施成本时用 RLHF（PPO）。</p><p><em>复习：第 5、6 与 10 章（PPO；DPO；SFT 最佳实践）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q0d">
<summary class="hh-quiz-question">Q0d：什么是 reward 模型？它如何训练，可能出什么问题？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>reward 模型（Reward Model, RM）是一个神经网络，输入 (prompt, response) 对，输出一个表示质量的标量分数。它在人类偏好数据上训练：给定 <math alttext="(y_{w},y_{l})" display="inline"><semantics><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>w</mi></msub><mo>,</mo><msub><mi>y</mi><mi>l</mi></msub><mo stretchy="false">)</mo></mrow></semantics></math> 对（其中 <math alttext="y_{w}" display="inline"><semantics><msub><mi>y</mi><mi>w</mi></msub></semantics></math> 被偏好），RM 学习赋予 <math alttext="R(y_{w})&gt;R(y_{l})" display="inline"><semantics><mrow><mrow><mi>R</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>w</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>&gt;</mo><mrow><mi>R</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>l</mi></msub><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math>。</p><p>训练：Bradley-Terry loss：<math alttext="\mathcal{L}=-\log\sigma(R(y_{w})-R(y_{l}))" display="inline"><semantics><mrow><mi>ℒ</mi><mo>=</mo><mrow><mo rspace="0.167em">−</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mrow><mi>σ</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mrow><mi>R</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>w</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>−</mo><mrow><mi>R</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>l</mi></msub><mo stretchy="false">)</mo></mrow></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></mrow></semantics></math>。架构：通常与 policy 共用同一个 Transformer，将 LM head 替换为标量投影。</p><p>可能出什么问题：</p><ol><li>Reward hacking：policy 找到了在 RM 上得分很高但实际质量很差的输出（例如过度冗长、重复，或包含 RM 存在偏好的特定短语）</li><li>分布漂移（Distribution Shift）：RM 是在更早的 policy 输出上训练的。随着训练推进，当前 policy 生成的是 RM 无法准确打分的分布外输出</li><li>标签噪声：人类标注者意见不一致、疲劳，或使用不一致的标准。这种噪声会传播到 RM 的预测中</li><li>过度自信：RM 对从未见过的输出赋予极端分数，提供具有误导性的梯度信号</li></ol><p><em>复习：第 9 章（Reward 模型训练）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q0e">
<summary class="hh-quiz-question">Q0e：解释 RL 中的探索-利用（exploration-exploitation）权衡。它在 LLM 训练中如何体现？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>在 RL 中，Agent 必须在以下两者间取得平衡：</p><ul><li>利用（Exploitation）：选择已知能产生高 reward 的动作（贪婪行为）</li><li>探索（Exploration）：尝试可能带来更高 reward（但也可能失败）的新动作</li></ul><p>在 LLM 训练中：policy 就是语言模型。“动作”是 Token 的选择。“利用”指生成与已得高分相似的回答；“探索”指尝试新颖的措辞、结构或推理路径。</p><p>表现形式：</p><ul><li>生成时的温度：温度越高 = 探索越多。GRPO 使用温度 1.0 以在每个 group 内获得多样的样本。</li><li>KL 惩罚：作为一种反探索的刹车——防止 policy 偏离参考模型太远。没有它，policy 可能坍缩到单一高 reward 模板（模式坍缩）。</li><li>GRPO 中的组采样：每个 prompt 生成 <math alttext="G" display="inline"><semantics><mi>G</mi></semantics></math> 个 response，显式探索输出空间，然后强化高于平均的 response。</li></ul><p>张力：探索太少 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 模型陷入局部最优（总是给出同样的安全回答）。探索太多 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 训练不稳定，质量剧烈波动。</p><p><em>复习：第 3 与 7 章（RL 导论；GRPO）。</em></p>
</div>
</details>

## 核心算法题

<details class="hh-quiz" id="q1">
<summary class="hh-quiz-question">Q1：解释 PPO 的裁剪目标。为什么它比朴素 PG 效果更好？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>朴素策略梯度：<math alttext="\nabla J=\mathbb{E}[\nabla\log\pi(a|s)\cdot\hat{A}]" display="inline"><semantics><mrow><mrow><mo rspace="0.167em">∇</mo><mi>J</mi></mrow><mo>=</mo><mrow><mi>𝔼</mi><mo>⁡</mo><mrow><mo stretchy="false">[</mo><mrow><mrow><mrow><mrow><mo rspace="0.167em">∇</mo><mi>log</mi></mrow><mo lspace="0.167em">⁡</mo><mi>π</mi></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>a</mi><mo lspace="0em" rspace="0em">|</mo><mi>s</mi><mo rspace="0.055em" stretchy="false">)</mo></mrow></mrow><mo rspace="0.222em">⋅</mo><mover accent="true"><mi>A</mi><mo>^</mo></mover></mrow><mo stretchy="false">]</mo></mrow></mrow></mrow></semantics></math>。问题：一个幸运/不幸的样本可能产生巨大的梯度 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> policy 跳到糟糕的区域 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 生成垃圾 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 下一个梯度让它更糟 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 无法恢复的“死亡螺旋”。</p><p>PPO 的解法：将概率比 <math alttext="r=\pi_{\text{new}}/\pi_{\text{old}}" display="inline"><semantics><mrow><mi>r</mi><mo>=</mo><mrow><msub><mi>π</mi><mtext>new</mtext></msub><mo>/</mo><msub><mi>π</mi><mtext>old</mtext></msub></mrow></mrow></semantics></math> 裁剪到 <math alttext="[0.8,1.2]" display="inline"><semantics><mrow><mo stretchy="false">[</mo><mn>0.8</mn><mo>,</mo><mn>1.2</mn><mo stretchy="false">]</mo></mrow></semantics></math>。</p><p>机制：对于好动作（<math alttext="\hat{A}&gt;0" display="inline"><semantics><mrow><mover accent="true"><mi>A</mi><mo>^</mo></mover><mo>&gt;</mo><mn>0</mn></mrow></semantics></math>）：目标是 <math alttext="\min(r\hat{A},1.2\hat{A})" display="inline"><semantics><mrow><mi>min</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mi>r</mi><mo lspace="0em" rspace="0em">​</mo><mover accent="true"><mi>A</mi><mo>^</mo></mover></mrow><mo>,</mo><mrow><mn>1.2</mn><mo lspace="0em" rspace="0em">​</mo><mover accent="true"><mi>A</mi><mo>^</mo></mover></mrow><mo stretchy="false">)</mo></mrow></mrow></semantics></math>。一旦 <math alttext="r" display="inline"><semantics><mi>r</mi></semantics></math> 超过 1.2，不再有额外收益——阻止 policy 过度押注于单个样本。对于坏动作（<math alttext="\hat{A}&lt;0" display="inline"><semantics><mrow><mover accent="true"><mi>A</mi><mo>^</mo></mover><mo>&lt;</mo><mn>0</mn></mrow></semantics></math>）：目标是 <math alttext="\min(r\hat{A},0.8\hat{A})" display="inline"><semantics><mrow><mi>min</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mi>r</mi><mo lspace="0em" rspace="0em">​</mo><mover accent="true"><mi>A</mi><mo>^</mo></mover></mrow><mo>,</mo><mrow><mn>0.8</mn><mo lspace="0em" rspace="0em">​</mo><mover accent="true"><mi>A</mi><mo>^</mo></mover></mrow><mo stretchy="false">)</mo></mrow></mrow></semantics></math>。一旦 <math alttext="r" display="inline"><semantics><mi>r</mi></semantics></math> 降到 0.8 以下，惩罚不再增长——防止灾难性遗忘。</p><p>关键洞见：它是 TRPO 的 KL 约束的一阶近似，但无需昂贵的二阶优化。每次更新对 policy 的改变最多为 <math alttext="\pm" display="inline"><semantics><mo>±</mo></semantics></math>20%。</p><p>具体到 LLM：Token 级的比 <math alttext="r_{t}=\pi_{\theta}(y_{t}|y_{&lt;t})/\pi_{\text{old}}(y_{t}|y_{&lt;t})" display="inline"><semantics><mrow><msub><mi>r</mi><mi>t</mi></msub><mo>=</mo><mrow><mrow><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">|</mo><msub><mi>y</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>t</mi></mrow></msub><mo stretchy="false">)</mo></mrow></mrow><mo>/</mo><msub><mi>π</mi><mtext>old</mtext></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">|</mo><msub><mi>y</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>t</mi></mrow></msub><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math> 防止任何单个 Token 的概率变化过于剧烈，从而保持生成的连贯性。</p><p><em>复习：第 5 章（PPO）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q2">
<summary class="hh-quiz-question">Q2：从第一性原理推导 DPO。它做了哪些假设？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>从 RLHF 目标出发：<math alttext="\max_{\pi}\mathbb{E}[r(x,y)]-\beta D_{\text{KL}}[\pi\|\pi_{\text{ref}}]" display="inline"><semantics><mrow><msub><mi>max</mi><mi>π</mi></msub><mi>𝔼</mi><mrow><mo stretchy="false">[</mo><mi>r</mi><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo stretchy="false">)</mo></mrow><mo stretchy="false">]</mo></mrow><mo>−</mo><mi>β</mi><msub><mi>D</mi><mtext>KL</mtext></msub><mrow><mo stretchy="false">[</mo><mi>π</mi><mo lspace="0em" rspace="0.167em">∥</mo><msub><mi>π</mi><mtext>ref</mtext></msub><mo stretchy="false">]</mo></mrow></mrow></semantics></math>。</p><p>第 1 步：写出 KKT 条件。最优 policy 有闭式解：<math alttext="\pi^{*}(y|x)\propto\pi_{\text{ref}}(y|x)\exp(r(x,y)/\beta)" display="inline"><semantics><mrow><mrow><msup><mi>π</mi><mo>∗</mo></msup><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>y</mi><mo lspace="0em" rspace="0em">|</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow><mo>∝</mo><mrow><msub><mi>π</mi><mtext>ref</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>y</mi><mo lspace="0em" rspace="0em">|</mo><mi>x</mi><mo stretchy="false">)</mo></mrow><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>exp</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo stretchy="false">)</mo></mrow></mrow><mo>/</mo><mi>β</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></semantics></math>。</p><p>第 2 步：反解出 reward：<math alttext="r(x,y)=\beta\log(\pi^{*}/\pi_{\text{ref}})+\beta\log Z(x)" display="inline"><semantics><mrow><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mrow><mi>β</mi><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><msup><mi>π</mi><mo>∗</mo></msup><mo>/</mo><msub><mi>π</mi><mtext>ref</mtext></msub></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow><mo>+</mo><mrow><mi>β</mi><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mrow><mi>Z</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></mrow></mrow></semantics></math>。</p><p>第 3 步：代入 Bradley-Terry 模型 <math alttext="P(y_{w}\succ y_{l})=\sigma(r(y_{w})-r(y_{l}))" display="inline"><semantics><mrow><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><msub><mi>y</mi><mi>w</mi></msub><mo>≻</mo><msub><mi>y</mi><mi>l</mi></msub></mrow><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mi>σ</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>w</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>−</mo><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>l</mi></msub><mo stretchy="false">)</mo></mrow></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math>。配分函数 <math alttext="Z(x)" display="inline"><semantics><mrow><mi>Z</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 相互抵消（同一 prompt）。</p><p>第 4 步：用 <math alttext="\pi_{\theta}" display="inline"><semantics><msub><mi>π</mi><mi>θ</mi></msub></semantics></math> 替换 <math alttext="\pi^{*}" display="inline"><semantics><msup><mi>π</mi><mo>∗</mo></msup></semantics></math>（我们正在训练的参数化 policy）：<math alttext="\mathcal{L}=-\mathbb{E}[\log\sigma(\beta\log\frac{\pi_{\theta}(y_{w})}{\pi_{\text{ref}}(y_{w})}-\beta\log\frac{\pi_{\theta}(y_{l})}{\pi_{\text{ref}}(y_{l})})]" display="inline"><semantics><mrow><mi>ℒ</mi><mo>=</mo><mrow><mo>−</mo><mrow><mi>𝔼</mi><mo>⁡</mo><mrow><mo stretchy="false">[</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mrow><mi>σ</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mrow><mi>β</mi><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mfrac><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>w</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mi>π</mi><mtext>ref</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>w</mi></msub><mo stretchy="false">)</mo></mrow></mrow></mfrac></mrow></mrow><mo>−</mo><mrow><mi>β</mi><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mfrac><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>l</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mi>π</mi><mtext>ref</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>l</mi></msub><mo stretchy="false">)</mo></mrow></mrow></mfrac></mrow></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow><mo stretchy="false">]</mo></mrow></mrow></mrow></mrow></semantics></math>。</p><p>假设：</p><ol><li>Bradley-Terry 偏好模型（成对、无平局、可传递）</li><li>最优 policy 可由 <math alttext="\pi_{\theta}" display="inline"><semantics><msub><mi>π</mi><mi>θ</mi></msub></semantics></math> 实现（容量充足）</li><li>偏好数据与训练数据来自同一分布（无分布漂移）</li><li>参考模型固定且合理</li></ol><p>假设被打破时：真实偏好不具有可传递性，训练期间数据发生漂移，标签含噪 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 这正是 Online DPO 与 IPO 存在的原因。</p><p><em>复习：第 6 章（DPO）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q3">
<summary class="hh-quiz-question">Q3：GRPO 与 PPO — 分别在什么情况下选择哪一个？权衡是什么？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>GRPO 的优势：</p><ul><li>不需要价值函数：省下一个模型的内存与复杂度</li><li>更简单：超参数更少，更直观（高于均值 = 好，低于均值 = 坏）</li><li>更适合可验证 reward：数学/代码场景中 <math alttext="r\in\{0,1\}" display="inline"><semantics><mrow><mi>r</mi><mo>∈</mo><mrow><mo stretchy="false">{</mo><mrow><mn>0</mn><mo>,</mo><mn>1</mn></mrow><mo stretchy="false">}</mo></mrow></mrow></semantics></math> 能给出清晰信号</li><li>DeepSeek-R1 证明了仅用二元 reward 就能教会模型涌现推理</li></ul><p>PPO 的优势：</p><ul><li>逐 Token 的信用分配：价值函数为每个 Token 分配 reward，而不只是序列级</li><li>样本效率更高：GAE 用价值预测估计优势，无需生成 <math alttext="G" display="inline"><semantics><mi>G</mi></semantics></math> 个样本</li><li>更适合细腻的 reward：当 reward 是连续的、且在不同 Token 间差异显著时</li><li>更成熟：已在 OpenAI、Anthropic 等经过实战检验</li></ul><p>经验法则：如果 reward 可验证（对/错）<math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> GRPO。如果 reward 细腻（RM 分数）且需要极致质量 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> PPO。如果算力有限 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> GRPO（无需训练 critic）。</p><p>算力对比：GRPO 每个 prompt 生成 <math alttext="G" display="inline"><semantics><mi>G</mi></semantics></math> 个 response（生成量多 8<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>），但跳过了价值函数训练。净效果：总算力相近，但分布不同（更多生成，更少训练）。</p><p><em>复习：第 5 与 7 章（PPO；GRPO）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q4">
<summary class="hh-quiz-question">Q4：GAE 是如何工作的？为 LLM 走一遍具体示例。</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>GAE = <math alttext="n" display="inline"><semantics><mi>n</mi></semantics></math> 步 TD 误差的加权和：<math alttext="\hat{A}_{t}=\sum_{l=0}^{T-t}(\gamma\lambda)^{l}\delta_{t+l}" display="inline"><semantics><mrow><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>t</mi></msub><mo rspace="0.111em">=</mo><mrow><msubsup><mo rspace="0em">∑</mo><mrow><mi>l</mi><mo>=</mo><mn>0</mn></mrow><mrow><mi>T</mi><mo>−</mo><mi>t</mi></mrow></msubsup><mrow><msup><mrow><mo stretchy="false">(</mo><mrow><mi>γ</mi><mo lspace="0em" rspace="0em">​</mo><mi>λ</mi></mrow><mo stretchy="false">)</mo></mrow><mi>l</mi></msup><mo lspace="0em" rspace="0em">​</mo><msub><mi>δ</mi><mrow><mi>t</mi><mo>+</mo><mi>l</mi></mrow></msub></mrow></mrow></mrow></semantics></math>。</p><p>具体示例：response 有 5 个 Token。只在末尾有 reward（<math alttext="r_{5}=0.8" display="inline"><semantics><mrow><msub><mi>r</mi><mn>5</mn></msub><mo>=</mo><mn>0.8</mn></mrow></semantics></math>）。价值预测：<math alttext="V_{1}=0.5,V_{2}=0.55,V_{3}=0.6,V_{4}=0.65,V_{5}=0.7" display="inline"><semantics><mrow><mrow><msub><mi>V</mi><mn>1</mn></msub><mo>=</mo><mn>0.5</mn></mrow><mo>,</mo><mrow><msub><mi>V</mi><mn>2</mn></msub><mo>=</mo><mn>0.55</mn></mrow><mo>,</mo><mrow><msub><mi>V</mi><mn>3</mn></msub><mo>=</mo><mn>0.6</mn></mrow><mo>,</mo><mrow><msub><mi>V</mi><mn>4</mn></msub><mo>=</mo><mn>0.65</mn></mrow><mo>,</mo><mrow><msub><mi>V</mi><mn>5</mn></msub><mo>=</mo><mn>0.7</mn></mrow></mrow></semantics></math>。</p><p>TD 误差（<math alttext="\gamma=1" display="inline"><semantics><mrow><mi>γ</mi><mo>=</mo><mn>1</mn></mrow></semantics></math>）：<math alttext="\delta_{1}=0+V_{2}-V_{1}=0.05" display="inline"><semantics><mrow><msub><mi>δ</mi><mn>1</mn></msub><mo>=</mo><mrow><mrow><mn>0</mn><mo>+</mo><msub><mi>V</mi><mn>2</mn></msub></mrow><mo>−</mo><msub><mi>V</mi><mn>1</mn></msub></mrow><mo>=</mo><mn>0.05</mn></mrow></semantics></math>，<math alttext="\delta_{2}=0+V_{3}-V_{2}=0.05" display="inline"><semantics><mrow><msub><mi>δ</mi><mn>2</mn></msub><mo>=</mo><mrow><mrow><mn>0</mn><mo>+</mo><msub><mi>V</mi><mn>3</mn></msub></mrow><mo>−</mo><msub><mi>V</mi><mn>2</mn></msub></mrow><mo>=</mo><mn>0.05</mn></mrow></semantics></math>，……，<math alttext="\delta_{5}=0.8+0-0.7=0.1" display="inline"><semantics><mrow><msub><mi>δ</mi><mn>5</mn></msub><mo>=</mo><mrow><mrow><mn>0.8</mn><mo>+</mo><mn>0</mn></mrow><mo>−</mo><mn>0.7</mn></mrow><mo>=</mo><mn>0.1</mn></mrow></semantics></math>。</p><p>取 <math alttext="\lambda=0.95" display="inline"><semantics><mrow><mi>λ</mi><mo>=</mo><mn>0.95</mn></mrow></semantics></math> 时：<math alttext="\hat{A}_{5}=0.1" display="inline"><semantics><mrow><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mn>5</mn></msub><mo>=</mo><mn>0.1</mn></mrow></semantics></math>（就是最终的 TD 误差），<math alttext="\hat{A}_{4}=0.05+0.95\times 0.1=0.145" display="inline"><semantics><mrow><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mn>4</mn></msub><mo>=</mo><mrow><mn>0.05</mn><mo>+</mo><mrow><mn>0.95</mn><mo lspace="0.222em" rspace="0.222em">×</mo><mn>0.1</mn></mrow></mrow><mo>=</mo><mn>0.145</mn></mrow></semantics></math>，<math alttext="\hat{A}_{3}=0.05+0.95\times 0.145=0.188" display="inline"><semantics><mrow><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mn>3</mn></msub><mo>=</mo><mrow><mn>0.05</mn><mo>+</mo><mrow><mn>0.95</mn><mo lspace="0.222em" rspace="0.222em">×</mo><mn>0.145</mn></mrow></mrow><mo>=</mo><mn>0.188</mn></mrow></semantics></math>，等等。</p><p>解读：Token 3 得到的优势是 0.188，因为它对一个获得高于预期 reward 的序列有贡献。更早的 Token 通过指数衰减获得信用。</p><p>对 LLM 而言：<math alttext="\gamma=1.0" display="inline"><semantics><mrow><mi>γ</mi><mo>=</mo><mn>1.0</mn></mrow></semantics></math>（所有 Token 都重要，有限时域）。Token <math alttext="t" display="inline"><semantics><mi>t</mi></semantics></math> 处的优势回答了：“考虑到这个 Token 之后发生的内容，这个 Token 的选择比预期更好还是更差？”</p><p><em>复习：第 5 章（PPO）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q5">
<summary class="hh-quiz-question">Q5：如何防止 reward hacking？给出一套分层防御策略。</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>检测信号：RM 分数上升但胜率持平/下降。response 长度单调增长。KL 散度 <math alttext="&gt;" display="inline"><semantics><mo>&gt;</mo></semantics></math> 15。多样性（唯一 n-gram）下降。阅读高 reward 输出能发现漏洞。</p><p>分层防御（按优先级排序）：</p><ol><li>KL 惩罚（主要手段）：自适应控制器将 KL 目标设为 <math alttext="\approx" display="inline"><semantics><mo>≈</mo></semantics></math> 6。如果 KL 上升，<math alttext="\beta" display="inline"><semantics><mi>β</mi></semantics></math> 会自动增大。防止偏离参考模型太远。</li><li>Reward 模型集成（3–5 个模型）：取分数的最小值或均值。单个模型有不同的盲区——能骗过一个模型的漏洞很少能骗过所有模型。</li><li>长度惩罚：<math alttext="r^{\prime}=r-c\cdot\max(0,\text{length}-L_{\text{target}})" display="inline"><semantics><mrow><msup><mi>r</mi><mo>′</mo></msup><mo>=</mo><mrow><mi>r</mi><mo>−</mo><mrow><mi>c</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><mrow><mi>max</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mn>0</mn><mo>,</mo><mrow><mtext>length</mtext><mo>−</mo><msub><mi>L</mi><mtext>target</mtext></msub></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></mrow></semantics></math>。防止“只要生成得更长 = 分数更高”的漏洞。</li><li>定期刷新 RM：每 2000 步，用当前 policy 生成数据，重新标注，加入 RM 训练集。在模型发现漏洞的同时将其封堵。</li><li>基于胜率的停止准则：跟踪相对 SFT baseline 的胜率。如果 RM 分数上升但胜率停滞超过 200 步，就停止训练。模型在钻漏洞，而不是在改进。</li></ol><p>检测后的恢复：回滚到最后一个“干净”的 checkpoint。将 <math alttext="\beta" display="inline"><semantics><mi>β</mi></semantics></math> 增大 2<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>。把发现的漏洞作为负样本加入 RM。</p><p><em>复习：第 9 与 11 章（Reward 模型训练；系统架构）。</em></p>
</div>
</details>

## 系统设计题

<details class="hh-quiz" id="q6">
<summary class="hh-quiz-question">Q6：为训练 70B 模型设计一个 RLHF 系统。逐一走查每个组件。</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>在 72 张 A100-80GB GPU 上的三集群解耦架构：</p><p>集群 1 — 生成（32 张 GPU）：</p><ul><li>8 个 vLLM 实例，每个 TP=4。PagedAttention + 投机解码（1B 草稿模型）。</li><li>连续批处理（Continuous Batching），最多 256 条序列同时进行。为了带宽使用 INT8 权重。</li><li>输出：(prompt, response, 逐 Token log-prob)。吞吐：<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>500 个 response/分钟。</li><li>无状态：只需从共享存储加载最新权重。恢复轻而易举。</li></ul><p>集群 2 — 打分（8 张 GPU）：</p><ul><li>Reward 模型（70B，INT8 = 70GB）装在 4 张 GPU 上（TP=4）。</li><li>参考模型（70B，INT8）装在 4 张 GPU 上（TP=4）。为 KL 计算逐 Token log-prob。</li><li>输出：reward 分数 + 每个 Token 的 KL。轻量、批量推理。</li></ul><p>集群 3 — 训练（32 张 GPU）：</p><ul><li>policy 模型使用 FSDP（ZeRO-3）。Flash Attention 2。梯度检查点（Gradient checkpointing）。</li><li>从 buffer 消费已打分的经验。PPO 更新：在大小为 16 的 mini-batch 上做 4 个 epoch。</li><li>每 50 步把更新后的权重推送到共享存储（异步、后台传输）。</li><li>每 100 步异步 checkpoint（非阻塞写入 NVMe + S3 备份）。</li></ul><p>连接网络：</p><ul><li>生成 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 打分：Ray/Redis 队列（<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>每批 10 MB：Token ID + log-prob）</li><li>打分 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 训练：经验池（Experience Pools，环形，保留最近 500 步）</li><li>训练 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 生成：共享并行 FS（Lustre/GPFS）上的权重存储。每 50 步推送 140GB，异步。</li></ul><p>重叠：当训练处理第 <math alttext="N" display="inline"><semantics><mi>N</mi></semantics></math> 步时，生成已经在为第 <math alttext="N+1" display="inline"><semantics><mrow><mi>N</mi><mo>+</mo><mn>1</mn></mrow></semantics></math> 步产出数据。这隐藏了生成延迟，相比单体架构达到 1.3–1.5<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> 的吞吐。</p><p><em>复习：第 11 章（大规模系统架构与基础设施）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q7">
<summary class="hh-quiz-question">Q7：如何处理生成瓶颈？定量说明各种解决方案。</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>生成占 RLHF 总墙钟时间的 60–70%。根因：自回归解码受显存带宽约束（算术强度 <math alttext="\approx 1" display="inline"><semantics><mrow><mphantom></mphantom><mo>≈</mo><mn>1</mn></mrow></semantics></math> FLOP/byte，而 A100 的 roofline 为 156 FLOP/byte）。</p><p>按影响力排序的解决方案：</p><p>1. 将生成与训练解耦（端到端 1.3–1.5<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>）：在独立硬件上运行生成，与训练重叠。这是最大的单项架构收益。</p><p>2. 使用 PagedAttention 的 vLLM（2–4<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>）：消除由内部碎片造成的 60–80% KV cache 内存浪费。支持大 3–4<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> 的 batch，即更好的带宽利用。</p><p>3. 连续批处理（1.5–2<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>）：不要等最长序列结束。在腾出的槽位立即启动新序列，让 GPU 保持忙碌。</p><p>4. 投机解码（2–3<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>）：1B 草稿模型提出 5 个 Token。70B 模型在一次前向中验证全部 5 个（并行！）。平均接受 3–4 个 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 每次前向 3–4 个 Token，而不是 1 个。</p><p>5. 生成权重使用 INT8/FP8（2<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>）：将每个 Token 的 140GB 权重读取减半。质量损失很小，因为 (a) 我们本来就在带 temperature 采样，(b) 只有生成使用 INT8，训练仍保持 BF16。</p><p>6. CUDA graph + kernel 融合（1.1–1.3<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>）：消除 Python/CUDA 启动开销。将 layernorm+attention+MLP 融合为更少的 kernel。</p><p>组合起来：相对朴素实现 <math alttext="1.5\times 3\times 1.5\times 2.5\times 2\times 1.2=40\times" display="inline"><semantics><mrow><mn>1.5</mn><mo lspace="0.222em" rspace="0.222em">×</mo><mn>3</mn><mo lspace="0.222em" rspace="0.222em">×</mo><mn>1.5</mn><mo lspace="0.222em" rspace="0.222em">×</mo><mn>2.5</mn><mo lspace="0.222em" rspace="0.222em">×</mo><mn>2</mn><mo lspace="0.222em" rspace="0.222em">×</mo><mn>1.2</mn><mo>=</mo><mn>40</mn><mo lspace="0.222em">×</mo></mrow></semantics></math>。实践中，收益递减使总量限制在朴素实现的 <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>10–20<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>。</p><p><em>复习：第 2 与 11 章（系统基础；系统架构）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q8">
<summary class="hh-quiz-question">Q8：解释解耦系统中的权重同步。你能容忍多大的陈旧度（staleness）？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>问题：生成集群使用 policy 权重产出 response。训练集群更新这些权重。它们位于不同的硬件上。如何保持同步？</p><p>为什么完美同步是浪费：70B BF16 的完整同步 = 140GB。在 InfiniBand 400Gb/s（50GB/s）下：每次同步 2.8 秒。如果每一步都同步（每 50–90 秒），会花掉 3–5% 的时间在权重传输上。可以接受，但没有必要。</p><p>陈旧度容忍分析：</p><ul><li>每步 policy 变化：<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>0.1%（以参数变化的均值衡量）</li><li>50 步：<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>5% 累积漂移</li><li>PPO 裁剪范围：可处理最多 20% 的概率比偏差</li><li>经验值：50 步陈旧度 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math><math alttext="&lt;" display="inline"><semantics><mo>&lt;</mo></semantics></math>2% 质量下降（以胜率衡量）</li></ul><p>生产策略：</p><ol><li>每 50 个训练步：将完整 BF16 checkpoint 推送到共享存储（传输 2.8 秒）</li><li>生成集群：在批次之间非阻塞地重新加载权重</li><li>增量压缩（可选）：只发送发生变化的参数（INT8 增量 <math alttext="\approx" display="inline"><semantics><mo>≈</mo></semantics></math> 5GB），作为偏移量应用。带宽减少 10<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>。</li><li>对于超大规模（256+ GPU）：流式同步——在后台持续发送小块数据。平均陈旧度：5–10 步。</li></ol><p>重要细节：生成期间计算的 log-prob 使用的是陈旧权重。PPO 的比 <math alttext="\pi_{\text{new}}/\pi_{\text{old}}" display="inline"><semantics><mrow><msub><mi>π</mi><mtext>new</mtext></msub><mo>/</mo><msub><mi>π</mi><mtext>old</mtext></msub></mrow></semantics></math> 用这些陈旧 log-prob 计算 <math alttext="\pi_{\text{old}}" display="inline"><semantics><msub><mi>π</mi><mtext>old</mtext></msub></semantics></math>。这没有问题，因为 PPO 本就是为离策略修正而设计的。</p><p><em>复习：第 11 章（大规模系统架构与基础设施）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q9">
<summary class="hh-quiz-question">Q9：在 512 张 GPU 上你会如何做容错？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>在 512 张 GPU 上，MTBF 为 4–8 小时。一次 5 天的训练会遭遇 15–30 次故障。</p><p>架构级韧性：</p><ul><li>生成集群 = 无状态。故障实例在 <math alttext="&lt;" display="inline"><semantics><mo>&lt;</mo></semantics></math>60 秒内重启（只需加载权重，无状态）。</li><li>训练集群 = 有状态。需要基于 checkpoint 的恢复。</li><li>打分集群 = 无状态。与生成相同。</li></ul><p>Checkpoint 策略：</p><ul><li>频率：每 50–100 步（5–10 分钟训练）。</li><li>方式：异步（非阻塞）。后台线程在下一步推进的同时写入。使用 FSDP 的分布式保存（每个 rank 并行保存自己的分片）。</li><li>内容：模型权重、优化器状态（Adam m/v）、LR 调度器、RNG 状态、KL 自适应系数、全局步计数器、replay buffer 指针。</li><li>存储：本地 NVMe（快，70B 需 30 秒）+ 异步复制到 S3/共享 FS（持久）。</li><li>保留：保留最近 3 个 checkpoint。自动删除更早的。</li></ul><p>检测与恢复流程：</p><ol><li>NCCL 集合通信超时（60 秒）或心跳丢失（10 秒）<math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 检测到故障。</li><li>通过 NVML 健康检查识别故障节点。</li><li>选项 A（快，<math alttext="&lt;" display="inline"><semantics><mo>&lt;</mo></semantics></math>2 分钟）：Torch Elastic 收缩 world size，重新分配分片，用 <math alttext="N-1" display="inline"><semantics><mrow><mi>N</mi><mo>−</mo><mn>1</mn></mrow></semantics></math> 个节点继续。在后台申请替换节点。</li><li>选项 B（干净，<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>5 分钟）：启动替换节点，重建进程组，加载最近的 checkpoint，恢复。</li><li>经验池已持久化——无需重新生成。</li></ol><p>预防：预筛选压力测试（GEMM、内存、NVLink）。ECC 错误监控（错误激增时抢先迁移）。热备节点（预加载环境）。双轨 InfiniBand 实现网络冗余。</p><p><em>复习：第 11 章（大规模系统架构与基础设施）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q10">
<summary class="hh-quiz-question">Q10：如何从 7B 扩展到 70B 再到 405B？每个规模下有什么变化？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>7B（单台 8 卡节点，数小时）：</p><ul><li>架构：单体（TRL 默认）。所有模型都放在同一批 GPU 上。</li><li>内存：LoRA + INT8 参考模型/RM 轻松装入 8<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>80GB。</li><li>并行：节点内 DP=8 或 FSDP。无网络通信。</li><li>超参数：LR=<math alttext="5\times 10^{-6}" display="inline"><semantics><mrow><mn>5</mn><mo lspace="0.222em" rspace="0.222em">×</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>6</mn></mrow></msup></mrow></semantics></math>，激进的 <math alttext="\beta" display="inline"><semantics><mi>β</mi></semantics></math>=0.02，50K–100K 步。</li><li>时间：每次运行 4–12 小时。迭代快。</li></ul><p>70B（32–64 张 GPU，2–5 天）：</p><ul><li>架构：半解耦。vLLM 生成 + FSDP 训练。</li><li>内存：ZeRO-3 必不可少。梯度检查点。INT8 参考模型/RM。</li><li>并行：节点内 TP=8（生成），节点间 FSDP（训练）。</li><li>超参数：LR=<math alttext="1.5\times 10^{-6}" display="inline"><semantics><mrow><mn>1.5</mn><mo lspace="0.222em" rspace="0.222em">×</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>6</mn></mrow></msup></mrow></semantics></math>，中等的 <math alttext="\beta" display="inline"><semantics><mi>β</mi></semantics></math>=0.05，10K–30K 步。</li><li>容错：异步 checkpoint、监控，但可以手动管理。</li></ul><p>405B（256–512 张 GPU，1–3 周）：</p><ul><li>架构：完全解耦。独立集群。权重存储 + 队列。</li><li>内存：训练使用 ZeRO-3 + TP=8 + PP=2。INT4 生成。</li><li>并行：3D 并行（TP<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>PP<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>DP = 8<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>2<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>16 = 256 张 GPU 训练）。</li><li>超参数：LR=<math alttext="5\times 10^{-7}" display="inline"><semantics><mrow><mn>5</mn><mo lspace="0.222em" rspace="0.222em">×</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>7</mn></mrow></msup></mrow></semantics></math>，非常保守的 <math alttext="\beta" display="inline"><semantics><mi>β</mi></semantics></math>=0.1，2K–5K 步。</li><li>容错：必须。弹性训练、冗余 checkpoint、热备节点。</li><li>关键变化：所需的 RL 训练少得多（模型经过预训练已经非常强大）。但每一步都贵 50<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>，所以不稳定是灾难性的。</li></ul><p>悖论：更大的模型实际上 <em>每一步更容易做 RL 训练</em>（更稳定，loss 地形更平滑）。但不稳定的代价随模型规模放大——在 405B 上一次失败的运行会浪费 $100K+ 的算力。</p><p><em>复习：第 11 章（大规模系统架构与基础设施）。</em></p>
</div>
</details>

## 实战与调试题

<details class="hh-quiz" id="q11">
<summary class="hh-quiz-question">Q11：reward 分数在上升，但模型质量在下降。诊断并修复。</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>典型的 reward hacking / 古德哈特定律。</p><p>诊断流程：</p><ol><li>检查 response 长度：绘制训练过程中的平均长度。单调增长？= 长度漏洞（RM 给更长的 response 打更高分）。</li><li>检查 KL 散度：<math alttext="&gt;" display="inline"><semantics><mo>&gt;</mo></semantics></math>15？= policy 偏离参考模型太远，丢失了能力。</li><li>检查多样性：每个 response 的唯一 trigram 数。在下降？= 模式坍缩（重复同样的高 reward 模式）。</li><li>人工检查：阅读 20 条最高 reward 的 response。它们有什么共同的模式？（例如全部以“这个问题问得好！”开头、全部使用项目符号、过度模棱两可）。</li><li>胜率：在留出的 prompt 上相对 SFT baseline 评估。如果 RM 上升而它持平/下降 = 漏洞确认。</li></ol><p>立即修复：</p><ul><li>回滚到胜率仍在改善的最后一个 checkpoint</li><li>将 <math alttext="\beta" display="inline"><semantics><mi>β</mi></semantics></math> 增大 2–3<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>（更强的 KL 惩罚）</li><li>加入显式长度惩罚：<math alttext="r^{\prime}=r-0.001\cdot\max(0,\text{len}-500)" display="inline"><semantics><mrow><msup><mi>r</mi><mo>′</mo></msup><mo>=</mo><mrow><mi>r</mi><mo>−</mo><mrow><mn>0.001</mn><mo lspace="0.222em" rspace="0.222em">⋅</mo><mrow><mi>max</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mn>0</mn><mo>,</mo><mrow><mtext>len</mtext><mo>−</mo><mn>500</mn></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></mrow></semantics></math></li></ul><p>结构性修复（防止再次发生）：</p><ul><li>RM 集成：在不同数据划分上训练 3–5 个 RM。取最小值或均值。漏洞是模型特有的。</li><li>RM 刷新：每 2000 步，用当前 policy 生成，获取人工标注，重新训练 RM。</li><li>多目标 reward：用相互独立的 RM 组合有用性 + 无害性 + 简洁性。</li><li>基于胜率（而非 RM 分数）的早停。你所优化的指标应当不同于训练信号。</li></ul><p><em>复习：第 9 与 11 章（Reward 模型训练；系统架构）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q12">
<summary class="hh-quiz-question">Q12：如何决定 RL 训练的 prompt 分布？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>Prompt 质量是 RLHF 中最被低估的因素。差的 Prompt = 没有学习信号。</p><p>组成（我的默认配比）：</p><ul><li>40% 真实用户流量（代表实际用例）</li><li>30% 合成（LLM 生成，填补覆盖空白——稀有主题、边角情况）</li><li>20% 课程式（渐进难度——先易，随模型变强提升复杂度）</li><li>10% 对抗式（red-team prompt、越狱尝试、含糊指令）</li></ul><p>关键：Goldilocks 过滤器：</p><ol><li>对每个候选 prompt，用当前模型生成 4–8 个回答。</li><li>用 RM 打分。计算通过率（超过阈值的比例）。</li><li>仅保留通过率在 20–80% 的 prompt：<span class="hh-tag">•</span><math alttext="&lt;" display="inline"><semantics><mo>&lt;</mo></semantics></math>20%：太难。模型几乎总是失败 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 全是负 advantage <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 无有效梯度。<span class="hh-tag">•</span><math alttext="&gt;" display="inline"><semantics><mo>&gt;</mo></semantics></math>80%：太易。模型几乎总是成功 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 全是正 advantage <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 无对比。<span class="hh-tag">•</span>20–80%：完美。成功与失败兼有 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 关于「什么有效」的清晰信号。</li><li>每 500 训练步重新过滤（模型在进步，难度分布也在漂移）。</li></ol><p>主题多样性：确保没有单一主题主导（每个类别 <math alttext="&lt;" display="inline"><semantics><mo>&lt;</mo></semantics></math>10%）。用 Embedding 聚类验证覆盖度。否则模型会过度优化主导主题。</p><p><em>复习：第 7 与 9 章（GRPO；Reward 模型训练）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q13">
<summary class="hh-quiz-question">Q13：RLHF 中 LoRA vs 全量微调。各自在什么情况下使用？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>LoRA（Low-Rank Adaptation，<math alttext="r" display="inline"><semantics><mi>r</mi></semantics></math>=64，<math alttext="\alpha" display="inline"><semantics><mi>α</mi></semantics></math>=16）：</p><ul><li>可训练参数：<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>模型的 0.2%（70B 对应 200M）</li><li>显存节省：无需独立的参考模型（base 模型 = 参考模型）！节省 140GB。</li><li>稳定性：天然更稳定（低秩约束限制了 policy 的漂移幅度）</li><li>速度：每步更快（需要更新的参数更少），但可能需要更多步</li><li>质量天花板：通常达到全量微调质量的 90–95%</li></ul><p>全量微调：</p><ul><li>所有参数都会更新。表达力最强。</li><li>需要独立的参考模型副本（70B 为 140GB）。或者通过非常频繁的 checkpoint 来「锚定」。</li><li>灾难性遗忘的风险更高。需要更低的 LR（<math alttext="3\times" display="inline"><semantics><mrow><mn>3</mn><mo lspace="0.222em">×</mo></mrow></semantics></math> 比 LoRA 低 3×）以及更强的 <math alttext="\beta" display="inline"><semantics><mi>β</mi></semantics></math>。</li><li>更适合：需要大幅分布漂移（新语言、迥异风格），LoRA 触及容量上限的情况。</li></ul><p>我的决策框架：</p><ol><li>先用 LoRA（<math alttext="r" display="inline"><semantics><mi>r</mi></semantics></math>=64）。它便宜 3<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> 且更稳定。</li><li>监控 LoRA 矩阵上的梯度范数。如果持续 <math alttext="&gt;" display="inline"><semantics><mo>&gt;</mo></semantics></math>1.0（相对于参数量很大）：LoRA 已到容量上限。</li><li>仅当胜率停滞且梯度分析提示容量受限时，才切换到全量微调。</li><li>对全量微调：使用 <math alttext="\text{LR}/3" display="inline"><semantics><mrow><mtext>LR</mtext><mo>/</mo><mn>3</mn></mrow></semantics></math>、<math alttext="\beta\times 2" display="inline"><semantics><mrow><mi>β</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mn>2</mn></mrow></semantics></math>、更频繁的 checkpoint，以及基于胜率的早停。</li></ol><p>混合：LoRA 用于对齐/安全（小行为漂移）+ 全量微调用于能力/推理（需要大漂移）。</p><p><em>复习：第 1 与 10 章（LLM 架构；SFT 最佳实践）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q14">
<summary class="hh-quiz-question">Q14：过程奖励模型（Process Reward Model, PRM）vs 结果奖励模型（Outcome Reward Model, ORM）。设计一个 PRM 系统。</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>ORM：只给最终答案打分。「整体回答好不好？」简单，但无法定位推理在<em>何处</em>出错。</p><p>PRM：为每个中间步骤打分。「这个推导的第 3 步对吗？」信息量大得多，但更难训练。</p><p>PRM 对推理任务的优势：</p><ul><li>准确定位推理失败的位置（步骤级信用分配）</li><li>支持树搜索：只展开步骤分数高的分支</li><li>reward hacking 更难：错误步骤 + 侥幸正确答案拿不到高分</li><li>在 MATH 基准上，PRM + best-of-N 比 ORM + best-of-N 高 10–20%</li></ul><p>训练 PRM：</p><ol><li>数据收集（蒙特卡洛方法）：<span class="hh-tag">•</span>对每道题，逐步生成推理轨迹<span class="hh-tag">•</span>在每一步 <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math>，从该点出发把解完整补全 <math alttext="M" display="inline"><semantics><mi>M</mi></semantics></math> 次（<math alttext="M=32" display="inline"><semantics><mrow><mi>M</mi><mo>=</mo><mn>32</mn></mrow></semantics></math>）<span class="hh-tag">•</span>步骤分数 = 抵达正确答案的补全比例<span class="hh-tag">•</span>分数明显下降的步骤 = 「出错」的步骤</li><li>标注：如果某一步的补全率 <math alttext="&gt;" display="inline"><semantics><mo>&gt;</mo></semantics></math> 50%，则标为「正确」；<math alttext="&lt;" display="inline"><semantics><mo>&lt;</mo></semantics></math> 20% 的标为「错误」。</li><li>模型：与 base 模型同架构 + 每个 Token 位置上的分类头。用步骤标签做二元交叉熵训练。</li><li>推理：为每步打分。若任一步骤的分数 <math alttext="&lt;" display="inline"><semantics><mo>&lt;</mo></semantics></math>0.3，则将该轨迹标记为有缺陷。</li></ol><p>在 RLHF 中使用 PRM：PRM 给出的逐 Token reward 直接喂入 GAE。每个 Token 都获得即时反馈，而非仅在序列末端。这极大改善了长推理链上的信用分配。</p><p><em>复习：第 9 与 13 章（Reward 模型训练；大型推理模型的 RL）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q15">
<summary class="hh-quiz-question">Q15：如何评估 RL 是否真的改进了模型？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>多面向评估（单一指标无法捕捉「质量」）：</p><p>1. 胜率（最重要、最可靠）：</p><ul><li>500+ 条多样 prompt。用 LLM judge（GPT-4 或 Claude）在盲测 A/B 对比中相对 SFT baseline 选出胜者。</li><li>目标：<math alttext="&gt;" display="inline"><semantics><mo>&gt;</mo></semantics></math>55% 胜率 = 有意义的改进。<math alttext="&gt;" display="inline"><semantics><mo>&gt;</mo></semantics></math>65% = 强改进。</li><li>使用位置去偏（交换 A/B 顺序并取平均）。报告置信区间。</li></ul><p>2. 能力基准（回归检测）：</p><ul><li>MMLU（知识）、HumanEval（代码）、MATH（推理）、MT-Bench（多轮）。</li><li>任何 <math alttext="&gt;" display="inline"><semantics><mo>&gt;</mo></semantics></math>2% 的下降 = 值得警惕的对齐税。排查哪些类别出现了退化。</li></ul><p>3. 分类别评估：</p><ul><li>安全：有害 prompt 上的拒答率（应上升）</li><li>真实性：TruthfulQA 分数（应上升或持平）</li><li>有用性：指令遵循基准上的任务完成率</li></ul><p>4. 分布指标：</p><ul><li>回答长度分布（不应出现剧烈偏移）</li><li>词汇多样性（每条回答的 unique token 数）</li><li>格式合规性（若针对特定格式训练过）</li></ul><p>5. 人工评估（黄金标准，成本高昂）：</p><ul><li>每个样本由 3+ 名熟练标注员做盲测 A/B。标注员间一致性 <math alttext="&gt;" display="inline"><semantics><mo>&gt;</mo></semantics></math> 70%。</li><li>仅用于最终模型选择，不在训练过程中使用（太慢/太贵）。</li></ul><p>危险信号：RM 分数上升 + 胜率持平 = reward hacking，而非真正的改进。胜率上升 + 基准下降 = 对齐税过高，降低 RL 强度。</p><p><em>复习：第 14 章（LLM 评估）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q16">
<summary class="hh-quiz-question">Q16：端到端描述 reward 模型的训练流水线。</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>阶段 1 —— 数据生成：</p><ul><li>收集 50K–100K 条多样 prompt（真实流量 + 合成）</li><li>对每个 prompt，以不同 temperature（0.3、0.7、1.0）并从多个模型生成 4–8 个回答（多样性是关键——如果所有回答都很相似，偏好就没有信息量）</li><li>合计：200K–800K 个候选回答</li></ul><p>阶段 2 —— 偏好收集：</p><ul><li>方案 A（昂贵，质量最佳）：由人工标注员比较成对回答。每对 3 名标注员。成本：每次比较 $2–5。</li><li>方案 B（便宜，与人类的一致率 85–90%）：LLM judge（GPT-4/Claude）。便宜 10<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>。适合规模化。</li><li>格式：(prompt, chosen response, rejected response)。标注员意见不一致（<math alttext="&lt;" display="inline"><semantics><mo>&lt;</mo></semantics></math>70% 一致性）的样本对被丢弃。</li><li>最终数据集：100K–500K 个样本对。</li></ul><p>阶段 3 —— 训练：</p><ul><li>架构：与 base LLM 相同 + 标量头（每个序列一个回归输出）。</li><li>损失：<math alttext="\mathcal{L}=-\mathbb{E}[\log\sigma(r(x,y_{w})-r(x,y_{l}))]" display="inline"><semantics><mrow><mi>ℒ</mi><mo>=</mo><mrow><mo>−</mo><mrow><mi>𝔼</mi><mo>⁡</mo><mrow><mo stretchy="false">[</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mrow><mi>σ</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>,</mo><msub><mi>y</mi><mi>w</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>−</mo><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>,</mo><msub><mi>y</mi><mi>l</mi></msub><mo stretchy="false">)</mo></mrow></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow><mo stretchy="false">]</mo></mrow></mrow></mrow></mrow></semantics></math>（Bradley-Terry）。</li><li>训练：只训 1 个 epoch！ RM 过拟合极快。验证准确率 68–75% 就算好（更高往往意味着过拟合到标注伪影）。</li><li>技巧：把 reward 居中到 0 附近（减去滑动均值）。检查长度偏差（若长度与分数的相关性 <math alttext="&gt;" display="inline"><semantics><mo>&gt;</mo></semantics></math> 0.3，则在训练中加入长度惩罚）。</li></ul><p>阶段 4 —— 验证：</p><ul><li>留出的偏好样本对：准确率应在 68–75%。</li><li>在新数据上与人类的一致率：<math alttext="&gt;" display="inline"><semantics><mo>&gt;</mo></semantics></math> 80%。</li><li>长度偏差检查：回答长度与 RM 分数之间的相关性应 <math alttext="&lt;" display="inline"><semantics><mo>&lt;</mo></semantics></math> 0.2。</li><li>一致性检查：同一 prompt 下，改写后的回答应得到相近的分数。</li></ul><p><em>复习：第 9 章（Reward 模型训练）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q17">
<summary class="hh-quiz-question">Q17：KL 散度爆炸时会发生什么？根因与修复。</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>KL 衡量的是什么：当前 policy 与参考模型之间的平均对数比：<math alttext="D_{\text{KL}}=\mathbb{E}_{y\sim\pi_{\theta}}[\log(\pi_{\theta}(y|x)/\pi_{\text{ref}}(y|x))]" display="inline"><semantics><mrow><msub><mi>D</mi><mtext>KL</mtext></msub><mo>=</mo><mrow><msub><mi>𝔼</mi><mrow><mi>y</mi><mo>∼</mo><msub><mi>π</mi><mi>θ</mi></msub></mrow></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">[</mo><mrow><mi>log</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mrow><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>y</mi><mo lspace="0em" rspace="0em">|</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow><mo>/</mo><msub><mi>π</mi><mtext>ref</mtext></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>y</mi><mo lspace="0em" rspace="0em">|</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow><mo stretchy="false">]</mo></mrow></mrow></mrow></semantics></math>。KL=0 表示与参考模型完全相同。KL=10 表示 policy 在其偏好的输出上多放了 10 nats 的概率。</p><p>健康范围：训练期间为 3–10。缓慢上升没问题。突然飙升 = 有问题。</p><p>KL 爆炸的根因：</p><ol><li>学习率过高：policy 迈出一大步，偏离参考模型。修复：把 LR 降低 2–5<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>。</li><li>Reward hacking：找到了一个远离参考行为却能拿到高 reward 的漏洞。修复：提高 <math alttext="\beta" display="inline"><semantics><mi>β</mi></semantics></math>，加入 RM 集成。</li><li>Mode collapse：policy 集中到某一个回答模板上。KL 在该模板处很高，其他处都很低。修复：提高熵奖励，提高 temperature。</li><li>坏 batch：某个运气不好的 batch 带着极端 advantage 把 policy 推偏了。修复：梯度裁剪，减小 mini-batch 大小。</li><li>价值函数发散：错误的 advantage 估计导致错误的更新。修复：降低价值函数的 LR，或切换到 GRPO（无价值函数）。</li></ol><p>恢复流程：</p><ol><li>检测：KL <math alttext="&gt;" display="inline"><semantics><mo>&gt;</mo></semantics></math> 15 持续 50+ 步，或 KL 在一步内跳升 <math alttext="&gt;" display="inline"><semantics><mo>&gt;</mo></semantics></math>5。</li><li>立即：加载最后一个干净的 checkpoint（KL <math alttext="&lt;" display="inline"><semantics><mo>&lt;</mo></semantics></math> 10）。</li><li>调整：把 LR 降低 50%。把 <math alttext="\beta" display="inline"><semantics><mi>β</mi></semantics></math> 提高 2<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>。把 cliprange 降到 0.1。</li><li>恢复训练：前 200 步密切监控。</li></ol><p><em>复习：第 5 与 7 章（PPO；GRPO）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q18">
<summary class="hh-quiz-question">Q18：比较单体式与解耦式 RLHF 架构。各自在什么情况下合适？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>单体式（TRL 默认：单进程，所有模型放在同一批 GPU 上）：</p><ul><li>优点：代码简单。没有分布式系统的复杂性。调试容易。</li><li>缺点：GPU 有 60% 的时间闲置（生成时算力闲置，训练时带宽闲置）。超过 <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>16 张 GPU 后无法高效扩展。所有模型争抢同一块显存。</li><li>适用场景：模型 <math alttext="\leq" display="inline"><semantics><mo>≤</mo></semantics></math> 13B，单节点，研究/原型开发。</li></ul><p>半解耦（vLLM 生成 + FSDP 训练，同一集群）：</p><ul><li>优点：利用率更好（生成与训练可以部分重叠）。可扩展到 64 张 GPU。</li><li>缺点：仍然共享硬件，无法独立优化。比单体式更复杂。</li><li>适用场景：13B–70B，2–8 个节点，生产环境实验。</li></ul><p>完全解耦（由队列连接的独立集群）：</p><ul><li>优点：每个集群针对自身负载优化。生成可以独立于训练扩展。生成集群是无状态的（容错轻而易举）。可扩展到数百张 GPU。</li><li>缺点：分布式系统的复杂性。权重过时。队列管理。网络开销。</li><li>适用场景：<math alttext="\geq" display="inline"><semantics><mo>≥</mo></semantics></math> 70B 生产训练。需要规模、容错和高利用率。</li></ul><p>关键洞见：生成受带宽约束，训练受算力约束。同一套硬件无法同时为两者优化。解耦让生成节点可以拥有：更大的显存带宽、INT8 权重、大 batch。训练节点则拥有：完整 BF16 精度、Flash Attention、FSDP 分片。</p><p><em>复习：第 11 章（系统架构与大规模基础设施）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q19">
<summary class="hh-quiz-question">Q19：如何为 RL 训练搭建课程学习？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>课程 = 难度逐步递增，让模型循序渐进地学习。</p><p>为什么重要：如果把最难的 prompt 直接丢给一个弱模型，它得到的全是负 reward <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 没有学习信号（一切都同样糟糕）。如果从容易的开始，模型会先发展出基本能力，再在此基础上进阶。</p><p>实现方式：</p><ol><li>难度评分：按当前模型的通过率对每个 prompt 评级（来自 Goldilocks 过滤）。易 = <math alttext="&gt;" display="inline"><semantics><mo>&gt;</mo></semantics></math>80% 通过率，中 = 30–80%，难 = <math alttext="&lt;" display="inline"><semantics><mo>&lt;</mo></semantics></math>30%。</li><li>排期：第 0–1000 步：70% 易、20% 中、10% 难。第 1000–5000 步：30% 易、50% 中、20% 难。第 5000+ 步：10% 易、40% 中、50% 难。</li><li>动态调整：每 500 步重新评估难度分布。模型已「掌握」的 prompt（通过率 <math alttext="&gt;" display="inline"><semantics><mo>&gt;</mo></semantics></math> 95%）会被淘汰。同时引入新的、更难的 prompt。</li><li>具体到 GRPO：课程学习确保每个组里始终混合着成功与失败。没有课程学习，难 prompt 会给出全零的组（无用）。</li></ol><p>证据：DeepSeek-R1 使用了隐式课程学习——从简单的数学/代码题开始，模型发展出基本推理能力，随后在没有任何显式排期的情况下解决了逐步变难的问题。</p><p><em>复习：第 7 与 12 章（GRPO；LLM 智能体训练）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q20">
<summary class="hh-quiz-question">Q20：你有 64 张 A100-80GB GPU 的预算，需要 RL 训练一个 70B 模型。设计分配方案。</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>8 个节点 <math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> 8 张 GPU = 共 64 张。需要在生成、打分和训练之间划分。</p><p>我的分配方案：</p><ul><li>生成：24 张 GPU（3 个节点）。6 个 vLLM 实例，TP=4。INT8 权重 = 70GB/模型，为 KV cache 留出空间。连续批处理，batch <math alttext="\approx" display="inline"><semantics><mo>≈</mo></semantics></math> 合计 128。</li><li>打分：8 张 GPU（1 个节点）。同一节点上放 RM（INT8，TP=4）+ 参考模型（INT8，TP=4）。或者共用 4 张 GPU，以 TP=4 在 RM 与参考模型之间轮流使用。</li><li>训练：32 张 GPU（4 个节点）。FSDP 横跨全部 32 张。每张 GPU 持有 <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>70B/32 = 2.2GB 参数 + 优化器分片。激活值有充足的余量。为保险起见使用梯度检查点。</li></ul><p>预期吞吐：</p><ul><li>生成：6 个实例 <math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math><math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>80 个回答/分钟 = 480 个回答/分钟</li><li>训练：batch 为 128，每 <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>15 秒一步（仅训练，不等待生成）</li><li>重叠：在训练一步 <math alttext="N" display="inline"><semantics><mi>N</mi></semantics></math>（15 秒）期间，生成会产出 <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>120 个回答，供第 <math alttext="N+1" display="inline"><semantics><mrow><mi>N</mi><mo>+</mo><mn>1</mn></mrow></semantics></math> 步使用。完美的流水线。</li></ul><p>瓶颈分析：生成 128 个回答需要 <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>45 秒。训练需要 <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>15 秒。打分需要 <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>5 秒。生成是瓶颈。可以把 8 张 GPU 从训练挪到生成（40 生成、24 训练）以求平衡，但那样训练又成了瓶颈。当前分配接近最优。</p><p>显存紧张时的替代方案：把打分移到训练节点上（分时复用：生成时打分，训练时训练）。省下 8 张 GPU。流水线略差，但可行。</p><p><em>复习：第 2 与 11 章（系统基础；系统架构）。</em></p>
</div>
</details>

## GRPO 变体与高级 RL 题

<details class="hh-quiz" id="q21">
<summary class="hh-quiz-question">Q21：什么是 DAPO，它相对标准 GRPO 有哪些改进？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>DAPO（Dynamic Adaptive Policy Optimization）引入了 5 项关键修改：</p><p>1. Clip-Higher（非对称裁剪）：标准 PPO/GRPO 在两个方向上以 <math alttext="\epsilon=0.2" display="inline"><semantics><mrow><mi>ϵ</mi><mo>=</mo><mn>0.2</mn></mrow></semantics></math> 等量裁剪。DAPO 使用 <math alttext="\epsilon_{\text{low}}=0.2" display="inline"><semantics><mrow><msub><mi>ϵ</mi><mtext>low</mtext></msub><mo>=</mo><mn>0.2</mn></mrow></semantics></math>，却用 <math alttext="\epsilon_{\text{high}}=0.28" display="inline"><semantics><mrow><msub><mi>ϵ</mi><mtext>high</mtext></msub><mo>=</mo><mn>0.28</mn></mrow></semantics></math>。这允许模型更激进地 <em>提高</em> 好动作的概率，同时仍限制它对坏动作的压制幅度。直觉：探索需要比利用更大的空间。</p><p>2. 过长过滤（Overlong Filtering）：如果一条回答触及最大长度上限（被截断，没有 EOS token），它会被完全从损失中屏蔽掉。理由：被截断的回答不含自然的停止信号——在它们上面训练会教模型认为「说到一半就停」是可以接受的。</p><p>3. Token 级损失：损失按所有序列的 Token 总数归一化，而不是按序列数量。这防止较长的序列主导梯度。</p><p>4. 软过长惩罚（Soft Overlong Punishment）：不再做二元截断过滤，而是在回答接近最大长度时施加渐进惩罚。<math alttext="r_{\text{soft}}=-c\cdot\max(0,\text{len}-L_{\text{soft}})/(L_{\text{max}}-L_{\text{soft}})" display="inline"><semantics><mrow><msub><mi>r</mi><mtext>soft</mtext></msub><mo rspace="0em">=</mo><mo lspace="0em">−</mo><mi>c</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><mi>max</mi><mrow><mo stretchy="false">(</mo><mn>0</mn><mo>,</mo><mtext>len</mtext><mo>−</mo><msub><mi>L</mi><mtext>soft</mtext></msub><mo stretchy="false">)</mo></mrow><mo>/</mo><mrow><mo stretchy="false">(</mo><msub><mi>L</mi><mtext>max</mtext></msub><mo>−</mo><msub><mi>L</mi><mtext>soft</mtext></msub><mo stretchy="false">)</mo></mrow></mrow></semantics></math>。</p><p>5. 动态采样（Dynamic Sampling）：训练期间对 prompt 重新采样，确保每个 batch 都混合着成功/失败（TRL 中尚未实现）。</p><p>何时使用：需要最大化探索和长补全（32K+ token）的大规模推理 RL。非对称裁剪尤其有价值。</p><p><em>复习：第 7 与 8 章（GRPO；偏好优化变体）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q22">
<summary class="hh-quiz-question">Q22：解释 vLLM 的训练-推理不一致问题。它为什么会发生，TIS/MIS 又如何修复它？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>问题：当用 vLLM 做生成、用训练框架（DeepSpeed/FSDP）做更新时，同一个模型、同样的权重却会产出 <em>不同</em> 的 Token 概率。原因如下：</p><ul><li>数值 kernel 不同（vLLM 使用为吞吐优化的自定义 CUDA kernel）</li><li>attention 实现不同（训练中为 Flash Attention，vLLM 中为 PagedAttention）</li><li>精度处理不同（vLLM 用 FP8/INT8，训练用 BF16）</li><li>批处理差异影响层归一化的数值</li></ul><p>这会悄无声息地破坏 PPO 的 on-policy 假设：我们用来自 vLLM 的 <math alttext="\pi_{\text{old}}" display="inline"><semantics><msub><mi>π</mi><mtext>old</mtext></msub></semantics></math> 和来自训练框架的 <math alttext="\pi_{\theta}" display="inline"><semantics><msub><mi>π</mi><mi>θ</mi></msub></semantics></math> 来计算比率 <math alttext="\pi_{\theta}/\pi_{\text{old}}" display="inline"><semantics><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo>/</mo><msub><mi>π</mi><mtext>old</mtext></msub></mrow></semantics></math>。这个比率从第零步起就是错的！</p><p>TIS（Truncated Importance Sampling，截断重要性采样）：通过乘以 <math alttext="\min(\pi_{\text{train}}/\pi_{\text{inference}},C)" display="inline"><semantics><mrow><mi>min</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><msub><mi>π</mi><mtext>train</mtext></msub><mo>/</mo><msub><mi>π</mi><mtext>inference</mtext></msub></mrow><mo>,</mo><mi>C</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 来校正梯度。带上限 <math alttext="C" display="inline"><semantics><mi>C</mi></semantics></math> 的 <math alttext="\min" display="inline"><semantics><mi>min</mi></semantics></math> 可防止极端校正破坏训练稳定性。典型值 <math alttext="C=2.0" display="inline"><semantics><mrow><mi>C</mi><mo>=</mo><mn>2.0</mn></mrow></semantics></math>。</p><p>MIS（Masked Importance Sampling，掩码重要性采样）：更激进——凡是 <math alttext="\pi_{\text{train}}/\pi_{\text{inference}}&gt;C" display="inline"><semantics><mrow><mrow><msub><mi>π</mi><mtext>train</mtext></msub><mo>/</mo><msub><mi>π</mi><mtext>inference</mtext></msub></mrow><mo>&gt;</mo><mi>C</mi></mrow></semantics></math> 的 Token 直接丢弃。对梯度零贡献。防止任何估计不准的 Token 影响更新。</p><p>序列级 vs Token 级：序列级 IS 在理论上正确（无偏）；Token 级 IS 有偏但方差更低。实践中，带截断的序列级表现最好。</p><p><em>复习：第 7 与 11 章（GRPO；系统架构）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q23">
<summary class="hh-quiz-question">Q23：GSPO vs GRPO——根本区别是什么，什么时候重要？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>GRPO：逐 <em>Token</em> 计算重要性比率：<math alttext="w_{i,t}=\pi_{\theta}(o_{i,t}|q,o_{i,&lt;t})/\pi_{\text{old}}(o_{i,t}|q,o_{i,&lt;t})" display="inline"><semantics><mrow><msub><mi>w</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub><mo>=</mo><mrow><mrow><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>o</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub><mo lspace="0em" rspace="0em">|</mo><mrow><mi>q</mi><mo>,</mo><msub><mi>o</mi><mrow><mrow><mi>i</mi><mo>,</mo><mphantom></mphantom></mrow><mo>&lt;</mo><mi>t</mi></mrow></msub></mrow><mo stretchy="false">)</mo></mrow></mrow><mo>/</mo><msub><mi>π</mi><mtext>old</mtext></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>o</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub><mo lspace="0em" rspace="0em">|</mo><mrow><mi>q</mi><mo>,</mo><msub><mi>o</mi><mrow><mrow><mi>i</mi><mo>,</mo><mphantom></mphantom></mrow><mo>&lt;</mo><mi>t</mi></mrow></msub></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math>，然后对每个 Token 独立裁剪。</p><p>GSPO：在 <em>序列级</em> 计算重要性比率：<math alttext="s_{i}(\theta)=(\pi_{\theta}(o_{i}|q)/\pi_{\text{old}}(o_{i}|q))^{1/|o_{i}|}" display="inline"><semantics><mrow><mrow><msub><mi>s</mi><mi>i</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><msup><mrow><mo stretchy="false">(</mo><mrow><mrow><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>o</mi><mi>i</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mo>/</mo><msub><mi>π</mi><mtext>old</mtext></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>o</mi><mi>i</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mo stretchy="false">)</mo></mrow><mrow><mn>1</mn><mo>/</mo><mrow><mo stretchy="false">|</mo><msub><mi>o</mi><mi>i</mi></msub><mo stretchy="false">|</mo></mrow></mrow></msup></mrow></semantics></math>——即 Token 概率的几何均值。对这个单一的序列级比率进行裁剪。</p><p>为什么重要：GRPO 的逐 Token 裁剪把每个 Token 当作独立的，但在语言中它们高度相关。序列早期一处微小的逐 Token 变化会随众多 Token 指数级累积。GSPO 通过考察完整的序列概率来捕捉这一点。</p><p>长度归一化：<math alttext="1/|o_{i}|" display="inline"><semantics><mrow><mn>1</mn><mo>/</mo><mrow><mo stretchy="false">|</mo><msub><mi>o</mi><mi>i</mi></msub><mo stretchy="false">|</mo></mrow></mrow></semantics></math> 次幂确保在不同长度的序列之间进行公平比较。没有它，较长的序列总会得到更低的概率比率。</p><p>何时使用 GSPO：当训练走向 off-policy 时（<code>steps_per_generation &gt; 1</code> 或 <code>num_iterations &gt; 1</code>）。如果完全 on-policy（比率 <math alttext="\approx 1" display="inline"><semantics><mrow><mphantom></mphantom><mo>≈</mo><mn>1</mn></mrow></semantics></math>），GRPO 与 GSPO 等价。</p><p><em>复习：第 7 与 8 章（GRPO；偏好优化变体）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q24">
<summary class="hh-quiz-question">Q24：论文「It Takes Two」表明 G=2 就能匹敌 G=16。这怎么可能？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>关键洞见在于，GRPO 的有效性并非来自准确的 advantage 估计（那需要很大的 <math alttext="G" display="inline"><semantics><mi>G</mi></semantics></math>），而是来自一个 隐式对比目标。</p><p>当 <math alttext="G=2" display="inline"><semantics><mrow><mi>G</mi><mo>=</mo><mn>2</mn></mrow></semantics></math>、reward 为二值（一个正确、一个错误）时：<math alttext="\hat{A}_{\text{correct}}=+1" display="inline"><semantics><mrow><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mtext>correct</mtext></msub><mo>=</mo><mrow><mo>+</mo><mn>1</mn></mrow></mrow></semantics></math>，<math alttext="\hat{A}_{\text{wrong}}=-1" display="inline"><semantics><mrow><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mtext>wrong</mtext></msub><mo>=</mo><mrow><mo>−</mo><mn>1</mn></mrow></mrow></semantics></math>（归一化后）。损失变为：提高正确回答的概率，降低错误回答的概率。这本质上就是一个 DPO 式的对比损失！</p><p>为什么更大的 <math alttext="G" display="inline"><semantics><mi>G</mi></semantics></math> 帮助不大：归一化后的 advantage <math alttext="\hat{A}_{i}=(r_{i}-\mu)/\sigma" display="inline"><semantics><mrow><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>i</mi></msub><mo>=</mo><mrow><mrow><mo stretchy="false">(</mo><mrow><msub><mi>r</mi><mi>i</mi></msub><mo>−</mo><mi>μ</mi></mrow><mo stretchy="false">)</mo></mrow><mo>/</mo><mi>σ</mi></mrow></mrow></semantics></math> 已经在好与坏之间造出了对比。更多样本能给出更好的 <math alttext="\mu" display="inline"><semantics><mi>μ</mi></semantics></math> 估计，但梯度方向由最好与最差之间的 <em>对比</em> 主导，而非均值准确率。</p><p>算力节省：<math alttext="G=2" display="inline"><semantics><mrow><mi>G</mi><mo>=</mo><mn>2</mn></mrow></semantics></math> 意味着生成算力比 <math alttext="G=16" display="inline"><semantics><mrow><mi>G</mi><mo>=</mo><mn>16</mn></mrow></semantics></math> 少 8<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>。由于生成占训练时间的 60%，这相当于训练快 <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>4<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>。</p><p>注意事项：通过率在 30–70% 时效果最好。如果通过率非常低（<math alttext="&lt;" display="inline"><semantics><mo>&lt;</mo></semantics></math>10%），<math alttext="G=2" display="inline"><semantics><mrow><mi>G</mi><mo>=</mo><mn>2</mn></mrow></semantics></math> 常常给出两个失败样本（无信号）。对难题需要更大的 <math alttext="G" display="inline"><semantics><mi>G</mi></semantics></math>。</p><p><em>复习：第 7 章（GRPO）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q25">
<summary class="hh-quiz-question">Q25：什么是 SAPO，它的软门控与硬裁剪有何不同？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>标准 PPO/GRPO 使用硬裁剪：<math alttext="\text{clip}(r,1-\epsilon,1+\epsilon)" display="inline"><semantics><mrow><mtext>clip</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>r</mi><mo>,</mo><mrow><mn>1</mn><mo>−</mo><mi>ϵ</mi></mrow><mo>,</mo><mrow><mn>1</mn><mo>+</mo><mi>ϵ</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></semantics></math>。在边界处，梯度会突然降到零。这造出了一个「死区」，模型在其中得不到任何学习信号。</p><p>SAPO 用平滑的 sigmoid 门控取代它：随着比率偏离 1，梯度被逐步衰减，绝不会突然归零。它使用非对称 temperature：</p><ul><li><math alttext="\tau_{+}=1.0" display="inline"><semantics><mrow><msub><mi>τ</mi><mo>+</mo></msub><mo>=</mo><mn>1.0</mn></mrow></semantics></math> 用于正 advantage（标准衰减）</li><li><math alttext="\tau_{-}=1.05" display="inline"><semantics><mrow><msub><mi>τ</mi><mo>−</mo></msub><mo>=</mo><mn>1.05</mn></mrow></semantics></math> 用于负 advantage（为抑制而略更激进的衰减）</li></ul><p>好处：（1）梯度地形中没有「悬崖」。（2）略微落在裁剪范围之外的 Token 仍有贡献（被衰减，而非归零）。（3）优化轨迹更稳定。（4）序列连贯——考虑完整的序列上下文。</p><p>取舍：信任域比硬裁剪略宽松，因此需要仔细调 temperature。但总体上对超参数选择更稳健。</p><p><em>复习：第 7 与 8 章（GRPO；偏好优化变体）。</em></p>
</div>
</details>

## DPO 扩展题

<details class="hh-quiz" id="q26">
<summary class="hh-quiz-question">Q26：比较 f-DPO 的散度选择。什么时候用前向 KL、JS 或反向 KL？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>标准 DPO 隐式地使用反向 KL（<math alttext="D_{\text{KL}}[\pi_{\theta}\|\pi_{\text{ref}}]" display="inline"><semantics><mrow><msub><mi>D</mi><mtext>KL</mtext></msub><mrow><mo stretchy="false">[</mo><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0.167em">∥</mo><msub><mi>π</mi><mtext>ref</mtext></msub><mo stretchy="false">]</mo></mrow></mrow></semantics></math>）：</p><ul><li>反向 KL（默认）：模式寻求（mode-seeking）。policy 把概率集中到参考模型概率高的地方。避免生成参考模型不会生成的文本。适合安全场景（保守）。</li><li>前向 KL：覆盖全域（mass-covering）。policy 试图覆盖参考模型的所有模式，甚至低概率的模式。有利于多样性，但可能生成低质量输出。</li><li>Jensen-Shannon：前向与反向之间的对称折中。在模式覆盖与模式寻求之间取得平衡。通用对齐场景往往最优。</li><li>Alpha 散度（<math alttext="\alpha=0.5" display="inline"><semantics><mrow><mi>α</mi><mo>=</mo><mn>0.5</mn></mrow></semantics></math>）：在前向（<math alttext="\alpha=0" display="inline"><semantics><mrow><mi>α</mi><mo>=</mo><mn>0</mn></mrow></semantics></math>）与反向（<math alttext="\alpha=1" display="inline"><semantics><mrow><mi>α</mi><mo>=</mo><mn>1</mn></mrow></semantics></math>）之间插值。可调。</li></ul><p>实践建议：从反向 KL 开始（标准 DPO）。如果模型过于保守（不愿尝试有创意的解法），改用 JS 散度。如果多样性至关重要（创意写作、头脑风暴），试试前向 KL。</p><p><em>复习：第 6 与 8 章（DPO；偏好优化变体）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q27">
<summary class="hh-quiz-question">Q27：你的 DPO 偏好数据有 15% 的标签噪声。你会怎么做？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>按复杂程度递增有三种方案：</p><p>1. Robust DPO（噪声率已知时最佳）：解析地对损失去偏：<math alttext="\mathcal{L}_{\text{robust}}=\frac{(1-\varepsilon)\mathcal{L}_{\text{DPO}}(y_{w},y_{l})-\varepsilon\mathcal{L}_{\text{DPO}}(y_{l},y_{w})}{1-2\varepsilon}" display="inline"><semantics><mrow><msub><mi>ℒ</mi><mtext>robust</mtext></msub><mo>=</mo><mfrac><mrow><mrow><mrow><mo stretchy="false">(</mo><mrow><mn>1</mn><mo>−</mo><mi>ε</mi></mrow><mo stretchy="false">)</mo></mrow><mo lspace="0em" rspace="0em">​</mo><msub><mi>ℒ</mi><mtext>DPO</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>w</mi></msub><mo>,</mo><msub><mi>y</mi><mi>l</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>−</mo><mrow><mi>ε</mi><mo lspace="0em" rspace="0em">​</mo><msub><mi>ℒ</mi><mtext>DPO</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>l</mi></msub><mo>,</mo><msub><mi>y</mi><mi>w</mi></msub><mo stretchy="false">)</mo></mrow></mrow></mrow><mrow><mn>1</mn><mo>−</mo><mrow><mn>2</mn><mo lspace="0em" rspace="0em">​</mo><mi>ε</mi></mrow></mrow></mfrac></mrow></semantics></math>。设 <math alttext="\varepsilon=0.15" display="inline"><semantics><mrow><mi>ε</mi><mo>=</mo><mn>0.15</mn></mrow></semantics></math>。这可以证明在期望意义上恢复干净的 DPO 目标。TRL：<code>loss_type="robust", label_smoothing=0.15</code>。</p><p>2. IPO（噪声率未知时最佳）：带目标间隔的平方损失。被错误标注的样本对影响有界（平方损失不会发散）。对任意噪声模式都更稳健，且无需知道 <math alttext="\varepsilon" display="inline"><semantics><mi>ε</mi></semantics></math>。TRL：<code>loss_type="ipo"</code>。</p><p>3. TR-DPO（分布漂移时最佳）：训练期间通过 EMA 更新参考模型。即使早期数据有噪声，不断演进的参考模型也能帮助模型自我纠正。TRL：<code>sync_ref_model=True, ref_model_mixup_alpha=0.6</code>。</p><p>数据侧修复：（1）过滤掉标注员间一致性 <math alttext="&lt;" display="inline"><semantics><mo>&lt;</mo></semantics></math>70% 的样本对。（2）用 RM 给样本对打分；丢弃 RM 与标签不一致的样本对。（3）主动学习：重新标注最不确定的样本对。</p><p><em>复习：第 6 与 8 章（DPO；偏好优化变体）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q28">
<summary class="hh-quiz-question">Q28：什么是 SimPO？为什么「无参考模型」是优势？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>SimPO 用回答的平均对数概率作为隐式 reward 信号：<math alttext="r(x,y)=\frac{1}{|y|}\sum_{t}\log\pi_{\theta}(y_{t}|x,y_{&lt;t})" display="inline"><semantics><mrow><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mfrac><mn>1</mn><mrow><mo stretchy="false">|</mo><mi>y</mi><mo stretchy="false">|</mo></mrow></mfrac><mo lspace="0em" rspace="0em">​</mo><mrow><msub><mo>∑</mo><mi>t</mi></msub><mrow><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>π</mi><mi>θ</mi></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">|</mo><mrow><mi>x</mi><mo>,</mo><msub><mi>y</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>t</mi></mrow></msub></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></mrow></semantics></math>——无需参考模型。</p><p>loss 中加入目标 margin <math alttext="\gamma" display="inline"><semantics><mi>γ</mi></semantics></math>：chosen 回答的平均 log-prob 应至少比 rejected 高 <math alttext="\gamma" display="inline"><semantics><mi>γ</mi></semantics></math>。</p><p>为什么「无参考」重要：</p><ol><li>显存：无参考模型 = 70B 模型节省 70–140GB。可在同样硬件上训练更大的模型。</li><li>简洁性：无需管理/加载/服务第二份模型副本。</li><li>无陈旧参考：DPO 的参考随着训练推进越来越不相关。SimPO 没有这个问题。</li><li>内置长度归一化：<math alttext="1/|y|" display="inline"><semantics><mrow><mn>1</mn><mo>/</mo><mrow><mo stretchy="false">|</mo><mi>y</mi><mo stretchy="false">|</mo></mrow></mrow></semantics></math> 天然防止长度偏差（DPO 需显式处理）。</li></ol><p>权衡：没有参考锚点，模型有更多自由去塌缩或漂移。<math alttext="\gamma" display="inline"><semantics><mi>γ</mi></semantics></math> margin 和长度归一化部分缓解了这一点，但在激进训练时 SimPO 可能不如 DPO 稳定。</p><p><em>复习：第 8 章（偏好优化变体）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q29">
<summary class="hh-quiz-question">Q29：解释 Iterative RPO。为何在推理任务中把 DPO 与 NLL loss 结合？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>用于推理的标准 DPO 有一种微妙的失效模式：它学会<em>判别</em>（给正确轨迹更高的隐式 reward），但不一定学会<em>生成</em>它们。</p><p>为什么：DPO 的梯度推高 chosen 的概率、压低 rejected 的概率。但 chosen 回答可能与模型自身会生成的东西差异巨大，以至于提升其概率并不能教会模型产出类似的推理模式。</p><p>RPO 的修复：在 chosen 回答上添加负对数似然（NLL/SFT）loss：<math alttext="\mathcal{L}=\mathcal{L}_{\text{DPO}}+\alpha\cdot\mathcal{L}_{\text{NLL}}(y_{w})" display="inline"><semantics><mrow><mi>ℒ</mi><mo>=</mo><mrow><msub><mi>ℒ</mi><mtext>DPO</mtext></msub><mo>+</mo><mrow><mrow><mi>α</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><msub><mi>ℒ</mi><mtext>NLL</mtext></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>w</mi></msub><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></semantics></math>。</p><p>NLL 项显式训练模型逐步生成获胜回答。DPO 项确保模型同时学会避免落败回答。结合起来：模型同时学到「如何正确推理」（NLL）与「该避免什么」（DPO）。</p><p>迭代：生成回答 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 检查正确性 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 构造对 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 用 RPO 训练 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 重复。每次迭代，模型更擅长生成正确推理，为下一轮提供更高质量的训练数据。</p><p>TRL：<code>loss_type=["sigmoid", "sft"], loss_weights=[1.0, 1.0]</code></p><p><em>复习：第 8 与 13 章（偏好优化变体；大型推理模型的 RL）。</em></p>
</div>
</details>

## GPU 架构与硬件题

<details class="hh-quiz" id="q30">
<summary class="hh-quiz-question">Q30：解释 GPU 内存层次结构。它对 LLM 推理为何重要？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>从最快到最慢：</p><ol><li>寄存器（Registers）：线程私有，<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>256 KB/SM。即时访问（0 周期延迟）。</li><li>SRAM（共享内存）：每 SM 私有，<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>192–228 KB/SM（A100：164 KB 可配置）。带宽：<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>19 TB/s（聚合）。延迟：<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>20 周期。</li><li>L2 缓存：跨整张 GPU 共享，40–60 MB（H100：50 MB）。带宽：<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>5 TB/s。延迟：<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>200 周期。</li><li>高带宽内存（HBM）：GPU 主显存，80 GB（A100）。带宽：2–3.35 TB/s。延迟：<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>400 周期。</li><li>CPU DRAM：通过 PCIe，512 GB+。带宽：32–64 GB/s。延迟：<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>10K 周期。</li></ol><p>对 LLM 为何重要：自回归生成在每个 Token 都要读取完整模型权重（70B 约 <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>140 GB）。以 2 TB/s 的 HBM 带宽，仅流式读权重就要 70ms。实际计算（一次矩阵-向量乘法）只要 0.5ms。GPU 有 99% 的时间在等数据。</p><p>Flash Attention 利用了这一点：通过把中间结果（QK 分数、softmax）保留在 SRAM（19 TB/s）而非写到 HBM（2 TB/s），消除了 attention 的 90% 内存流量。计算量未变，但 HBM 的读写降低 10<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>。</p><p><em>复习：第 2 章（LLM 的系统基础）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q31">
<summary class="hh-quiz-question">Q31：Flash Attention 如何工作？什么是 online softmax 技巧？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>问题：标准 attention 在 HBM 中物化 <math alttext="n\times n" display="inline"><semantics><mrow><mi>n</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mi>n</mi></mrow></semantics></math> 的 attention 矩阵。<math alttext="n=8192" display="inline"><semantics><mrow><mi>n</mi><mo>=</mo><mn>8192</mn></mrow></semantics></math>：每个 head <math alttext="8192^{2}\times 2=134" display="inline"><semantics><mrow><mrow><msup><mn>8192</mn><mn>2</mn></msup><mo lspace="0.222em" rspace="0.222em">×</mo><mn>2</mn></mrow><mo>=</mo><mn>134</mn></mrow></semantics></math> MB，32 头每层 4.3 GB。必须写到 HBM 再读回做 softmax 与乘法——3 次完整的 HBM 往返。</p><p>Flash Attention 方案：从不存储完整的 <math alttext="n\times n" display="inline"><semantics><mrow><mi>n</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mi>n</mi></mrow></semantics></math> 矩阵。以可放进 SRAM 的分块（tile）处理。</p><p>算法：</p><ol><li>将 <math alttext="Q" display="inline"><semantics><mi>Q</mi></semantics></math> 拆为 <math alttext="B_{r}" display="inline"><semantics><msub><mi>B</mi><mi>r</mi></msub></semantics></math> 行的块，<math alttext="K/V" display="inline"><semantics><mrow><mi>K</mi><mo>/</mo><mi>V</mi></mrow></semantics></math> 拆为 <math alttext="B_{c}" display="inline"><semantics><msub><mi>B</mi><mi>c</mi></msub></semantics></math> 行的块。</li><li>对每个 <math alttext="Q" display="inline"><semantics><mi>Q</mi></semantics></math> 块：遍历所有 <math alttext="K" display="inline"><semantics><mi>K</mi></semantics></math> 块，计算部分 attention 分数。</li><li>online softmax 技巧：维护滑动 max <math alttext="m" display="inline"><semantics><mi>m</mi></semantics></math> 与滑动 sum <math alttext="\ell" display="inline"><semantics><mi mathvariant="normal">ℓ</mi></semantics></math> 以做 softmax 归一。处理新的 <math alttext="K" display="inline"><semantics><mi>K</mi></semantics></math> 块时更新：<math alttext="m_{\text{new}}=\max(m_{\text{old}},\max(\text{scores}))" display="inline"><semantics><mrow><msub><mi>m</mi><mtext>new</mtext></msub><mo>=</mo><mrow><mi>max</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>m</mi><mtext>old</mtext></msub><mo>,</mo><mrow><mi>max</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mtext>scores</mtext><mo stretchy="false">)</mo></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math>，将上一累加器按 <math alttext="e^{m_{\text{old}}-m_{\text{new}}}" display="inline"><semantics><msup><mi>e</mi><mrow><msub><mi>m</mi><mtext>old</mtext></msub><mo>−</mo><msub><mi>m</mi><mtext>new</mtext></msub></mrow></msup></semantics></math> 重新缩放，再加上新的贡献。</li><li>输出以增量方式累加——从不需要完整的 <math alttext="n\times n" display="inline"><semantics><mrow><mi>n</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mi>n</mi></mrow></semantics></math> 矩阵。</li></ol><p>关键洞见：softmax 本来是全局操作（对所有元素取 <math alttext="\max" display="inline"><semantics><mi>max</mi></semantics></math> 和 <math alttext="\sum" display="inline"><semantics><mo>∑</mo></semantics></math>）。online 技巧将其分解为带修正因子的局部更新。数学上精确——并非近似。</p><p>结果：显存 <math alttext="O(n)" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>n</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 而非 <math alttext="O(n^{2})" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msup><mi>n</mi><mn>2</mn></msup><mo stretchy="false">)</mo></mrow></mrow></semantics></math>。速度提升 2–4<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>（更少 HBM 访问，更多时间在 SRAM）。</p><p>Flash Attention 2：在 warp 之间做更好的工作划分，将非矩阵乘的 FLOPs 减少 2<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>。</p><p>Flash Attention 3（H100/Hopper）：使用张量内存加速器（Tensor Memory Accelerator, TMA）做异步加载、warp 专门化（生产者/消费者 warp）、支持 FP8。</p><p><em>复习：第 1 与 2 章（LLM 架构；系统基础）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q32">
<summary class="hh-quiz-question">Q32：解释 PagedAttention。它如何解决键值缓存（KV cache）问题？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>问题：生成过程中，每条序列都需要键值缓存（存储所有先前 Token 的 K 与 V 张量）。对 70B 模型：每个 Token 需要 <math alttext="2\times n_{\text{layers}}\times d_{\text{model}}\times 2" display="inline"><semantics><mrow><mn>2</mn><mo lspace="0.222em" rspace="0.222em">×</mo><msub><mi>n</mi><mtext>layers</mtext></msub><mo lspace="0.222em" rspace="0.222em">×</mo><msub><mi>d</mi><mtext>model</mtext></msub><mo lspace="0.222em" rspace="0.222em">×</mo><mn>2</mn></mrow></semantics></math> 字节 = <math alttext="2\times 80\times 8192\times 2\approx 2.5" display="inline"><semantics><mrow><mrow><mn>2</mn><mo lspace="0.222em" rspace="0.222em">×</mo><mn>80</mn><mo lspace="0.222em" rspace="0.222em">×</mo><mn>8192</mn><mo lspace="0.222em" rspace="0.222em">×</mo><mn>2</mn></mrow><mo>≈</mo><mn>2.5</mn></mrow></semantics></math> MB。2048 Token 的序列：<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>5 GB 的键值缓存。</p><p>传统分配方式：为每条活跃序列预分配 max_sequence_length。若 max=2048 但平均为 500，就浪费了已分配内存的 75%。若有 50 条并发序列，就是数百 GB 的浪费。</p><p>PagedAttention：受操作系统虚拟内存启发：</p><ol><li>键值缓存被拆分为固定大小的 <em>块（block）</em>（页），每块保存 16 个 Token 的 KV。</li><li>一张 <em>block table</em>（类似页表）把逻辑 Token 位置映射到物理内存块。</li><li>块随着序列增长按需分配。没有预分配浪费。</li><li>序列结束后，释放的块立即归还到池中。</li></ol><p>额外好处：</p><ul><li>前缀共享：共享同一 system prompt 的多条序列共用键值缓存块（写时复制）。对聊天应用节省 30–50% 内存。</li><li>抢占：可以把低优先级序列的块「换出」到 CPU，为更高优先级的请求释放 GPU 显存。</li><li>近乎零碎片：内部碎片仅限最后一个块（<math alttext="&lt;" display="inline"><semantics><mo>&lt;</mo></semantics></math>16 个 Token）。外部碎片被消除（任何空闲块都可用在任何位置）。</li></ul><p>结果：同样的内存下可多容纳 3–5<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> 条并发序列 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 服务吞吐提升 3–5<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>。</p><p><em>复习：第 2 章（LLM 的系统基础）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q33">
<summary class="hh-quiz-question">Q33：比较 NVLink 与 InfiniBand。在 RLHF 训练中各自何时使用？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>NVLink（节点内，GPU 到 GPU）：</p><ul><li>带宽：600 GB/s（A100），900 GB/s（H100）——双向总和</li><li>延迟：<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>1 <math alttext="\mu" display="inline"><semantics><mi>μ</mi></semantics></math>s</li><li>范围：单个物理节点内（8 张 GPU 通过 NVSwitch 互联）</li><li>使用场景：张量并行（Tensor Parallelism）（TP=8）。每层的矩阵乘被拆分到多张 GPU 上，每层之后都需要 AllReduce。需要超高带宽 + 低延迟。</li></ul><p>InfiniBand NDR（节点间，node-to-node）：</p><ul><li>带宽：每端口 400 Gb/s = 50 GB/s。使用 8 个端口（GPUDirect RDMA）：每节点聚合 400 GB/s。</li><li>延迟：<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>1–5 <math alttext="\mu" display="inline"><semantics><mi>μ</mi></semantics></math>s（RDMA）</li><li>范围：集群中各节点之间。需要交换机（fat-tree 拓扑）。</li><li>使用场景：数据并行（Data Parallelism）/ FSDP 的梯度同步。梯度的 AllReduce 每个训练步只发生一次（而非每层一次），因此对延迟的容忍度更高。</li></ul><p>具体到 RLHF：</p><ul><li><em>生成</em>：节点内通过 NVLink 做 TP=8。跨节点的多个 vLLM 实例之间不通信（易并行）。</li><li><em>训练</em>：节点内通过 NVLink 做 TP=8 + 节点间通过 InfiniBand 做 FSDP。梯度在完整反向传播后同步。</li><li><em>权重同步</em>：训练 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 生成使用 InfiniBand（传输 140 GB，异步，50 GB/s 下约需 <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>3s）。</li></ul><p><em>复习：第 2 与 11 章（系统基础；系统架构）。</em></p>
</div>
</details>

## 优化与训练题

<details class="hh-quiz" id="q34">
<summary class="hh-quiz-question">Q34：解释 Adam 与 AdamW 的区别。这一差异对 LLM 为何重要？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>带 L2 正则化的 Adam：<math alttext="\theta_{t+1}=\theta_{t}-\alpha\cdot(\hat{m}_{t}/(\sqrt{\hat{v}_{t}}+\epsilon)+\lambda\theta_{t})" display="inline"><semantics><mrow><msub><mi>θ</mi><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>=</mo><mrow><msub><mi>θ</mi><mi>t</mi></msub><mo>−</mo><mrow><mi>α</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><mrow><mo stretchy="false">(</mo><mrow><mrow><msub><mover accent="true"><mi>m</mi><mo>^</mo></mover><mi>t</mi></msub><mo>/</mo><mrow><mo stretchy="false">(</mo><mrow><msqrt><msub><mover accent="true"><mi>v</mi><mo>^</mo></mover><mi>t</mi></msub></msqrt><mo>+</mo><mi>ϵ</mi></mrow><mo stretchy="false">)</mo></mrow></mrow><mo>+</mo><mrow><mi>λ</mi><mo lspace="0em" rspace="0em">​</mo><msub><mi>θ</mi><mi>t</mi></msub></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></semantics></math>。权重衰减项 <math alttext="\lambda\theta_{t}" display="inline"><semantics><mrow><mi>λ</mi><mo lspace="0em" rspace="0em">​</mo><msub><mi>θ</mi><mi>t</mi></msub></mrow></semantics></math> 位于自适应缩放的<em>内部</em>。梯度较大的参数（<math alttext="v_{t}" display="inline"><semantics><msub><mi>v</mi><mi>t</mi></msub></semantics></math> 较大）获得<em>更少</em>的权重衰减（除以 <math alttext="\sqrt{v_{t}}" display="inline"><semantics><msqrt><msub><mi>v</mi><mi>t</mi></msub></msqrt></semantics></math>）。这不是真正的权重衰减——它依赖于尺度。</p><p>AdamW（解耦权重衰减）：<math alttext="\theta_{t+1}=(1-\alpha\lambda)\theta_{t}-\alpha\cdot\hat{m}_{t}/(\sqrt{\hat{v}_{t}}+\epsilon)" display="inline"><semantics><mrow><msub><mi>θ</mi><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>=</mo><mrow><mrow><mrow><mo stretchy="false">(</mo><mrow><mn>1</mn><mo>−</mo><mrow><mi>α</mi><mo lspace="0em" rspace="0em">​</mo><mi>λ</mi></mrow></mrow><mo stretchy="false">)</mo></mrow><mo lspace="0em" rspace="0em">​</mo><msub><mi>θ</mi><mi>t</mi></msub></mrow><mo>−</mo><mrow><mrow><mi>α</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><msub><mover accent="true"><mi>m</mi><mo>^</mo></mover><mi>t</mi></msub></mrow><mo>/</mo><mrow><mo stretchy="false">(</mo><mrow><msqrt><msub><mover accent="true"><mi>v</mi><mo>^</mo></mover><mi>t</mi></msub></msqrt><mo>+</mo><mi>ϵ</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></semantics></math>。权重衰减在自适应更新<em>之外</em>、<em>之前</em>施加。无论梯度历史如何，每个参数都获得相同比例的衰减。</p><p>对 LLM 为何重要：</p><ol><li>LLM 的参数横跨多个数量级（embedding 层 vs attention vs FFN）。Adam 的耦合 L2 实际上对小梯度参数的惩罚比对大梯度参数更重——这是错误的行为。</li><li>解耦的 WD 在所有层上提供统一的正则化，防止某些层无限增长而另一些层过度收缩。</li><li>经验上：在长预训练任务中，与相同有效正则化强度的 Adam+L2 相比，AdamW 的 perplexity 好 2–5%。</li></ol><p>具体到 RL：通常用 <math alttext="\lambda=0" display="inline"><semantics><mrow><mi>λ</mi><mo>=</mo><mn>0</mn></mrow></semantics></math>（无权重衰减）。KL 惩罚提供了正则化。但对 SFT：用 AdamW 配合 <math alttext="\lambda=0.01" display="inline"><semantics><mrow><mi>λ</mi><mo>=</mo><mn>0.01</mn></mrow></semantics></math>–<math alttext="0.1" display="inline"><semantics><mn>0.1</mn></semantics></math> 是标准做法。</p><p><em>复习：第 1 章（LLM 架构与优化方法）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q35">
<summary class="hh-quiz-question">Q35：为什么学习率预热是必要的？没有它会怎样？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>问题：Adam 的二阶矩估计 <math alttext="v_{t}=\beta_{2}v_{t-1}+(1-\beta_{2})g_{t}^{2}" display="inline"><semantics><mrow><msub><mi>v</mi><mi>t</mi></msub><mo>=</mo><mrow><mrow><msub><mi>β</mi><mn>2</mn></msub><mo lspace="0em" rspace="0em">​</mo><msub><mi>v</mi><mrow><mi>t</mi><mo>−</mo><mn>1</mn></mrow></msub></mrow><mo>+</mo><mrow><mrow><mo stretchy="false">(</mo><mrow><mn>1</mn><mo>−</mo><msub><mi>β</mi><mn>2</mn></msub></mrow><mo stretchy="false">)</mo></mrow><mo lspace="0em" rspace="0em">​</mo><msubsup><mi>g</mi><mi>t</mi><mn>2</mn></msubsup></mrow></mrow></mrow></semantics></math> 从 <math alttext="v_{0}=0" display="inline"><semantics><mrow><msub><mi>v</mi><mn>0</mn></msub><mo>=</mo><mn>0</mn></mrow></semantics></math> 开始。偏差校正 <math alttext="\hat{v}_{t}=v_{t}/(1-\beta_{2}^{t})" display="inline"><semantics><mrow><msub><mover accent="true"><mi>v</mi><mo>^</mo></mover><mi>t</mi></msub><mo>=</mo><mrow><msub><mi>v</mi><mi>t</mi></msub><mo>/</mo><mrow><mo stretchy="false">(</mo><mrow><mn>1</mn><mo>−</mo><msubsup><mi>β</mi><mn>2</mn><mi>t</mi></msubsup></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math> 在数学上做了补偿，但实践中：</p><ul><li>最初几步：<math alttext="v_{t}" display="inline"><semantics><msub><mi>v</mi><mi>t</mi></msub></semantics></math> 基于 1–5 个梯度样本。对真实方差的估计极不准确。</li><li>若某参数在初始阶段恰好得到很小的梯度，<math alttext="v_{t}" display="inline"><semantics><msub><mi>v</mi><mi>t</mi></msub></semantics></math> 就极小 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 有效 LR 巨大 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 灾难性的更新。</li><li>偏差校正会放大早期更新：在第 1 步，<math alttext="\hat{v}_{1}=v_{1}/(1-0.999)=1000\cdot v_{1}" display="inline"><semantics><mrow><msub><mover accent="true"><mi>v</mi><mo>^</mo></mover><mn>1</mn></msub><mo>=</mo><mrow><msub><mi>v</mi><mn>1</mn></msub><mo>/</mo><mrow><mo stretchy="false">(</mo><mrow><mn>1</mn><mo>−</mo><mn>0.999</mn></mrow><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mn>1000</mn><mo lspace="0.222em" rspace="0.222em">⋅</mo><msub><mi>v</mi><mn>1</mn></msub></mrow></mrow></semantics></math>。</li></ul><p>不做预热：最初的 10–100 步常出现梯度尖峰，永久性地损害模型。在优化器稳定之前，早期表示就被打乱了。</p><p>预热的修复：从 LR <math alttext="\approx 0" display="inline"><semantics><mrow><mphantom></mphantom><mo>≈</mo><mn>0</mn></mrow></semantics></math> 开始，在 <math alttext="W" display="inline"><semantics><mi>W</mi></semantics></math> 步内线性增至目标值（通常为训练的 3–10%）。等到 LR 达到满值时，<math alttext="v_{t}" display="inline"><semantics><msub><mi>v</mi><mi>t</mi></msub></semantics></math> 已累积了足够多的样本而变得准确。</p><p>典型设置：</p><ul><li>预训练：2000 步预热（<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>200K 步的 1%）</li><li>SFT：100 步预热（<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>2000 步的 5%）</li><li>RL（PPO/GRPO）：20–50 步预热（较短，模型已因 SFT 而稳定）</li></ul><p><em>复习：第 1 与 10 章（LLM 架构；SFT 最佳实践）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q36">
<summary class="hh-quiz-question">Q36：比较各种学习率调度。对 RL 微调你会选哪一种？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>余弦衰减：<math alttext="\eta_{t}=\eta_{\text{min}}+\frac{1}{2}(\eta_{\text{max}}-\eta_{\text{min}})(1+\cos(\pi t/T))" display="inline"><semantics><mrow><msub><mi>η</mi><mi>t</mi></msub><mo>=</mo><mrow><msub><mi>η</mi><mtext>min</mtext></msub><mo>+</mo><mrow><mfrac><mn>1</mn><mn>2</mn></mfrac><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mrow><msub><mi>η</mi><mtext>max</mtext></msub><mo>−</mo><msub><mi>η</mi><mtext>min</mtext></msub></mrow><mo stretchy="false">)</mo></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mrow><mn>1</mn><mo>+</mo><mrow><mi>cos</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mrow><mi>π</mi><mo lspace="0em" rspace="0em">​</mo><mi>t</mi></mrow><mo>/</mo><mi>T</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></semantics></math>。预训练与 SFT 的标准做法。平滑衰减，大部分时间处于中等 LR。</p><p>线性衰减：<math alttext="\eta_{t}=\eta_{\text{max}}(1-t/T)" display="inline"><semantics><mrow><msub><mi>η</mi><mi>t</mi></msub><mo>=</mo><mrow><msub><mi>η</mi><mtext>max</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mrow><mn>1</mn><mo>−</mo><mrow><mi>t</mi><mo>/</mo><mi>T</mi></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math>。更简单，短程训练下结果与余弦相近。</p><p>WSD（Warmup-Stable-Decay）：预热 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 80% 阶段保持恒定 LR <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 最后 20% 快速衰减。预训练的新标准。「稳定」阶段带来一致的学习；最后的衰减榨出残余收益。</p><p>恒定调度：不衰减。预热后保持 <math alttext="\eta_{t}=\eta_{\text{max}}" display="inline"><semantics><mrow><msub><mi>η</mi><mi>t</mi></msub><mo>=</mo><msub><mi>η</mi><mtext>max</mtext></msub></mrow></semantics></math>。</p><p>对 RL 微调（PPO/GRPO），我会选择：恒定调度 + 短预热。理由：</p><ol><li>RL 训练长度高度不可预测（依据胜率而非 epoch 决定何时停止）。</li><li>余弦/线性衰减假设你事先知道总步数。</li><li>LR 本就已经很低（<math alttext="10^{-6}" display="inline"><semantics><msup><mn>10</mn><mrow><mo>−</mo><mn>6</mn></mrow></msup></semantics></math>），进一步衰减会让更新几乎不可见。</li><li>PPO 的自适应 KL 控制器已经在调节有效步长。</li><li>若必须衰减：在宽裕的预算上使用线性衰减，指标进入平台期就提前停止。</li></ol><p><em>复习：第 1 与 5 章（LLM 架构；PPO）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q37">
<summary class="hh-quiz-question">Q37：为什么梯度裁剪对 RL 训练至关重要，而对 SFT 不那么重要？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>SFT：监督 loss 平滑且行为良好。梯度范数在各 batch 间保持一致（通常为 0.1–1.0）。在 1.0 处裁剪很少被触发——它只是一张安全网。</p><p>RL（PPO/GRPO）：梯度范数高度多变，原因如下：</p><ol><li>奖励方差：一个 batch 可能全是高 reward 的回答，下一个可能全都很低。advantage <math alttext="\hat{A}" display="inline"><semantics><mover accent="true"><mi>A</mi><mo>^</mo></mover></semantics></math> 剧烈波动。</li><li>比值爆炸：若某个稀有 Token 的概率变化很大，<math alttext="r_{t}=\pi_{\text{new}}/\pi_{\text{old}}" display="inline"><semantics><mrow><msub><mi>r</mi><mi>t</mi></msub><mo>=</mo><mrow><msub><mi>π</mi><mtext>new</mtext></msub><mo>/</mo><msub><mi>π</mi><mtext>old</mtext></msub></mrow></mrow></semantics></math> 可能非常大 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 在裁剪起作用前就产生很大的梯度。</li><li>稀疏奖励：在二值 reward 的 GRPO 中，有些 prompt 的结果全对（advantage <math alttext="\approx 0" display="inline"><semantics><mrow><mphantom></mphantom><mo>≈</mo><mn>0</mn></mrow></semantics></math>），随后一个难题 prompt 突然给出极端 advantage。</li><li>KL 项：当策略发生偏离时，KL 惩罚的梯度可能突增。</li></ol><p>不做裁剪：一个坏 batch 就能产生 100<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> 于正常量级的梯度 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 一步摧毁模型。无法恢复（对全部预训练的灾难性遗忘）。</p><p>典型设置：<code>max_grad_norm=1.0</code>。有些在 RL 训练早期用 0.5 以求更安全。范数是跨所有参数全局计算的（而非逐层）。</p><p>监控：如果裁剪在超过 20% 的步上被触发，你的 LR 可能太高，或 batch size 太小。</p><p><em>复习：第 5 与 7 章（PPO；GRPO）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q38">
<summary class="hh-quiz-question">Q38：训练中 BF16 与 FP16 的对比。这一选择何时重要？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>FP16：1 位符号 + 5 位指数 + 10 位尾数。范围：<math alttext="\pm 65504" display="inline"><semantics><mrow><mo>±</mo><mn>65504</mn></mrow></semantics></math>。精度：<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>3.3 个十进制有效位。</p><p>BF16：1 位符号 + 8 位指数 + 7 位尾数。范围：<math alttext="\pm 3.4\times 10^{38}" display="inline"><semantics><mrow><mo>±</mo><mn>3.4</mn><mo lspace="0.222em" rspace="0.222em">×</mo><mn>10</mn><msup><mrow></mrow><mn>38</mn></msup></mrow></semantics></math>（与 FP32 相同！）。精度：<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>2.4 个十进制有效位。</p><p>为什么 BF16 对 LLM 更优：</p><ol><li>无需 loss scaling：FP16 的范围小（<math alttext="\pm" display="inline"><semantics><mo>±</mo></semantics></math>65K），意味着梯度与激活经常上溢/下溢。需要动态 loss scaling（把 loss 乘 1024，再对梯度除回来）。BF16 拥有 FP32 的范围——上溢基本不可能。</li><li>更简单的代码：没有 loss scaler，无需检查 inf/nan，无需动态调整 scaling。</li><li>对 RL 至关重要：RL 的梯度比 SFT 更嘈杂、更尖锐。FP16 的 loss scaling 经常失败（选错 scale，导致 NaN）。BF16「就是能用」。</li></ol><p>何时 FP16 可能更好：如果你需要最高精度（某些科学计算任务），并且能管理好 loss scaling。FP16 多 3 位尾数 = 结果略更精确。</p><p>FP32 主权重：即便前向/反向用 BF16，也要用 FP32 累积梯度更新，防止舍入误差在成千上万个小步中累积。这是所有 LLM 训练的标准做法。</p><p><em>复习：第 2 章（LLM 的系统基础）。</em></p>
</div>
</details>

## Reward 模型与 SFT 题

<details class="hh-quiz" id="q39">
<summary class="hh-quiz-question">Q39：推导 Bradley-Terry 奖励模型 loss。它有哪些局限？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>Bradley-Terry 模型：给定两个回答，较好的那个（<math alttext="y_{w}" display="inline"><semantics><msub><mi>y</mi><mi>w</mi></msub></semantics></math>）被偏好的概率：<math alttext="P(y_{w}\succ y_{l}|x)=\sigma(r(x,y_{w})-r(x,y_{l}))" display="inline"><semantics><mrow><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><msub><mi>y</mi><mi>w</mi></msub><mo>≻</mo><msub><mi>y</mi><mi>l</mi></msub></mrow><mo lspace="0em" rspace="0em">|</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mi>σ</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>,</mo><msub><mi>y</mi><mi>w</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>−</mo><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>,</mo><msub><mi>y</mi><mi>l</mi></msub><mo stretchy="false">)</mo></mrow></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math>，其中 <math alttext="\sigma" display="inline"><semantics><mi>σ</mi></semantics></math> 是 sigmoid。</p><p>MLE 推导：给定 <math alttext="N" display="inline"><semantics><mi>N</mi></semantics></math> 个偏好对，最大化似然：<math alttext="\prod_{i}P(y_{w}^{i}\succ y_{l}^{i})" display="inline"><semantics><mrow><msub><mo>∏</mo><mi>i</mi></msub><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><msubsup><mi>y</mi><mi>w</mi><mi>i</mi></msubsup><mo>≻</mo><msubsup><mi>y</mi><mi>l</mi><mi>i</mi></msubsup></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math>。取负对数：<math alttext="\mathcal{L}=-\sum_{i}\log\sigma(r(x_{i},y_{w}^{i})-r(x_{i},y_{l}^{i}))" display="inline"><semantics><mrow><mi>ℒ</mi><mo rspace="0em">=</mo><mo lspace="0em" rspace="0.055em">−</mo><msub><mo>∑</mo><mi>i</mi></msub><mi>log</mi><mi>σ</mi><mrow><mo stretchy="false">(</mo><mi>r</mi><mrow><mo stretchy="false">(</mo><msub><mi>x</mi><mi>i</mi></msub><mo>,</mo><msubsup><mi>y</mi><mi>w</mi><mi>i</mi></msubsup><mo stretchy="false">)</mo></mrow><mo>−</mo><mi>r</mi><mrow><mo stretchy="false">(</mo><msub><mi>x</mi><mi>i</mi></msub><mo>,</mo><msubsup><mi>y</mi><mi>l</mi><mi>i</mi></msubsup><mo stretchy="false">)</mo></mrow><mo stretchy="false">)</mo></mrow></mrow></semantics></math>。</p><p>局限：</p><ol><li>无法处理平局：BT 无法建模「同样好」——它强制给出严格偏好。</li><li>传递性：假设若 A<math alttext="&gt;" display="inline"><semantics><mo>&gt;</mo></semantics></math>B 且 B<math alttext="&gt;" display="inline"><semantics><mo>&gt;</mo></semantics></math>C，则 A<math alttext="&gt;" display="inline"><semantics><mo>&gt;</mo></semantics></math>C。人类并不具备传递性。</li><li>与上下文无关：无论当时有哪些备选，reward 都相同。</li><li>标量坍缩：把所有质量维度压缩成一个数。一个回答可能安全但无帮助——RM 必须做权衡。</li><li>长度偏置：更长的回答得到更高分（信息更多 = 更可能包含标注者想要的内容）。必须显式去相关。</li></ol><p>缓解措施：margin loss（要求最小间隔 <math alttext="\delta" display="inline"><semantics><mi>δ</mi></semantics></math>）、reward 中心化（减去滑动均值）、训练时的长度惩罚、多头 RM（对 helpfulness/safety/accuracy 分别打分）。</p><p><em>复习：第 9 章（奖励模型训练）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q40">
<summary class="hh-quiz-question">Q40：SFT 中的序列打包是什么，为何重要？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>问题：训练样本长度不一。标准 batching 会把所有样本填充到 max_length。若 max=4096 但平均为 500，就有 88% 的算力浪费在 padding Token 上（它们贡献零梯度）。</p><p>打包方案：把多个短样本拼接成一条 max_length 序列，用 EOS token 分隔。同时训练所有样本。</p><p>示例：与其把 4 条序列填充到 4096（16K token，14K padding），不如打包成 1 条 4096 的序列，让 4 个样本首尾相接（4096 个真实 token，0 padding）。效率高 4<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>。</p><p>关键细节——块对角 attention mask：没有特殊处理时，样本 2 会 attend 到样本 1 的 token（交叉污染）。必须使用块对角 attention mask，限制每个样本只 attend 自身的 token。</p><p>在 TRL 中：<code>SFTConfig(packing=True, max_seq_length=4096)</code>。会自动处理 mask。</p><p>注意事项：（1）较长的样本仍需各自独占 batch 条目（不能在序列中间切开）。（2）position embedding 的实现略有复杂度（每个样本需重置）。（3）有人认为打包改变了有效 batch size（每步样本更多）——需相应调整 LR。</p><p><em>复习：第 10 章（SFT 最佳实践与技术）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q41">
<summary class="hh-quiz-question">Q41：解释 SFT 的 completion-only masking。若不用它会怎样？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>在 chat 格式的 SFT 数据中：<code>[system] + [user message] + [assistant response]</code>。标准 NLL loss 会在包括 system prompt 与 user message 在内的所有 token 上计算 loss。</p><p>不做掩码的问题：模型浪费容量去学习预测 user 的消息（推理时根本不需要生成它）。更糟的是：若训练数据里的 user 消息五花八门，模型会对「现在轮到谁」感到困惑。</p><p>completion-only masking：把 prompt 中所有 token（system + user）的 loss 权重设为 0。只在 assistant 回答的 token 上计算 loss。</p><p>TRL：<code>DataCollatorForCompletionOnlyLM(response_template="&lt;|assistant|&gt;")</code></p><p>影响：在指令遵循 benchmark 上通常好 5–15%。收敛更快（梯度信号集中在有用的 token 上）。计算成本不变。</p><p>微妙之处：必须把回答模板 token 计入 loss（教会模型开始作答），但要排除它之前的一切。</p><p><em>复习：第 10 章（SFT 最佳实践与技术）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q42">
<summary class="hh-quiz-question">Q42：SFT 质量如何影响 RL 的上限？什么是 pass@k 诊断？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>上限定理（非正式）：RL 只能强化模型已经能以不可忽略概率产生的行为。如果 SFT 模型生成正确解的概率为 0%，RL 永远找不到它。</p><p>为什么：GRPO/PPO 从当前策略采样并强化好的样本。若分布中不存在好样本，就没有可强化的东西。RL 的探索受限于基座策略的支撑集。</p><p>pass@k 诊断：每个 prompt 生成 <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 个回答，检查是否<em>有任何一个</em>正确：</p><ul><li>pass@1：模型的典型表现（贪婪/低 temperature）。</li><li>pass@8：GRPO 在 <math alttext="G=8" display="inline"><semantics><mrow><mi>G</mi><mo>=</mo><mn>8</mn></mrow></semantics></math> 下能达到的上界。</li><li>pass@64：激进 Best-of-N 的上界。</li><li>pass@256：RL 提升的近似上限。</li></ul><p>解读：</p><ul><li>pass@1=20%、pass@64=80%：很好！RL 有 4<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> 的提升空间。预期收益显著。</li><li>pass@1=20%、pass@64=25%：几乎没有提升空间。RL 帮助不大。需要先做更好的 SFT。</li><li>pass@1=5%、pass@64=60%：模型<em>能够</em>解出但很少做到。这是 RL 的完美场景（强化那些稀有的成功）。</li></ul><p>规则：若 pass@64 <math alttext="&lt;" display="inline"><semantics><mo>&lt;</mo></semantics></math> 1.5<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> pass@1，就先投入更好的 SFT 数据，再开始 RL。</p><p><em>复习：第 7 与 10 章（GRPO；SFT 最佳实践）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q43">
<summary class="hh-quiz-question">Q43：为聊天模型设计一个多目标 reward 系统。你如何平衡 helpfulness 与 safety？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>架构：为每个目标使用独立的 reward 模型：</p><ul><li><math alttext="r_{\text{helpful}}" display="inline"><semantics><msub><mi>r</mi><mtext>helpful</mtext></msub></semantics></math>：在 helpfulness 偏好上训练（质量、准确性、完整性）</li><li><math alttext="r_{\text{safe}}" display="inline"><semantics><msub><mi>r</mi><mtext>safe</mtext></msub></semantics></math>：在 safety 偏好上训练（拒答、无害、无幻觉）</li><li><math alttext="r_{\text{format}}" display="inline"><semantics><msub><mi>r</mi><mtext>format</mtext></msub></semantics></math>：基于规则（遵循指令、格式规范、长度适当）</li></ul><p>组合策略：</p><ol><li>加权求和（最简单）：<math alttext="r=w_{1}r_{\text{helpful}}+w_{2}r_{\text{safe}}+w_{3}r_{\text{format}}" display="inline"><semantics><mrow><mi>r</mi><mo>=</mo><mrow><mrow><msub><mi>w</mi><mn>1</mn></msub><mo lspace="0em" rspace="0em">​</mo><msub><mi>r</mi><mtext>helpful</mtext></msub></mrow><mo>+</mo><mrow><msub><mi>w</mi><mn>2</mn></msub><mo lspace="0em" rspace="0em">​</mo><msub><mi>r</mi><mtext>safe</mtext></msub></mrow><mo>+</mo><mrow><msub><mi>w</mi><mn>3</mn></msub><mo lspace="0em" rspace="0em">​</mo><msub><mi>r</mi><mtext>format</mtext></msub></mrow></mrow></mrow></semantics></math>。问题：safety 可能被 helpfulness 压过。</li><li>约束法（更安全）：最大化 <math alttext="r_{\text{helpful}}" display="inline"><semantics><msub><mi>r</mi><mtext>helpful</mtext></msub></semantics></math>，约束条件为 <math alttext="r_{\text{safe}}&gt;\tau" display="inline"><semantics><mrow><msub><mi>r</mi><mtext>safe</mtext></msub><mo>&gt;</mo><mi>τ</mi></mrow></semantics></math>。通过以下方式实现：<math alttext="r=r_{\text{helpful}}-\lambda\cdot\max(0,\tau-r_{\text{safe}})" display="inline"><semantics><mrow><mi>r</mi><mo>=</mo><mrow><msub><mi>r</mi><mtext>helpful</mtext></msub><mo>−</mo><mrow><mi>λ</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><mrow><mi>max</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mn>0</mn><mo>,</mo><mrow><mi>τ</mi><mo>−</mo><msub><mi>r</mi><mtext>safe</mtext></msub></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></mrow></semantics></math>，其中 <math alttext="\lambda" display="inline"><semantics><mi>λ</mi></semantics></math> 很大。</li><li>GDPO 归一化（最适合 GRPO）：在组内对每个 reward 独立归一化，然后组合：<math alttext="\hat{A}=w_{1}\hat{A}_{\text{helpful}}+w_{2}\hat{A}_{\text{safe}}" display="inline"><semantics><mrow><mover accent="true"><mi>A</mi><mo>^</mo></mover><mo>=</mo><mrow><mrow><msub><mi>w</mi><mn>1</mn></msub><mo lspace="0em" rspace="0em">​</mo><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mtext>helpful</mtext></msub></mrow><mo>+</mo><mrow><msub><mi>w</mi><mn>2</mn></msub><mo lspace="0em" rspace="0em">​</mo><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mtext>safe</mtext></msub></mrow></mrow></mrow></semantics></math>。防止某个 reward 因尺度差异而主导。</li><li>字典序：safety 是硬约束（必须通过），然后再优化 helpfulness。分阶段训练：先做 safety 对齐，再做 helpfulness。</li></ol><p>实用权重：从 <math alttext="w_{\text{safe}}=2.0,w_{\text{helpful}}=1.0,w_{\text{format}}=0.5" display="inline"><semantics><mrow><mrow><msub><mi>w</mi><mtext>safe</mtext></msub><mo>=</mo><mn>2.0</mn></mrow><mo>,</mo><mrow><msub><mi>w</mi><mtext>helpful</mtext></msub><mo>=</mo><mn>1.0</mn></mrow><mo>,</mo><mrow><msub><mi>w</mi><mtext>format</mtext></msub><mo>=</mo><mn>0.5</mn></mrow></mrow></semantics></math> 开始。safety 的权重是 2<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>，因为它的失效模式（有害内容）比 helpfulness 失效（回答平庸）糟糕得多。</p><p><em>复习：第 9 与 12 章（奖励模型训练；LLM 智能体训练）。</em></p>
</div>
</details>

## 系统架构扩展题

<details class="hh-quiz" id="q44">
<summary class="hh-quiz-question">Q44：投机解码如何工作？它在 RLHF 中何时有用？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>问题：大模型每次前向只生成一个 token（70B 约 <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>70ms）。慢。</p><p>投机解码：</p><ol><li>草稿：小模型（1–7B）快速生成 <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 个候选 token（<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>5ms 完成全部 <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math>）。</li><li>验证：大模型做一次前向，并行给全部 <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 个 token 打分。接受满足 <math alttext="p_{\text{large}}(t_{i})\geq p_{\text{draft}}(t_{i})" display="inline"><semantics><mrow><mrow><msub><mi>p</mi><mtext>large</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>t</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>≥</mo><mrow><msub><mi>p</mi><mtext>draft</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>t</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math> 的 token（总是接受），其余的按概率接受。</li><li>结果：平均每次验证步骤接受 3–4 个 token。加速比：2–3<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>。</li></ol><p>关键性质：输出分布与单独从大模型采样<em>完全相同</em>。没有质量损失。draft 模型只影响速度，不影响输出。</p><p>具体到 RLHF：生成占 60% 的算力。生成上 2–3<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> 的加速 = 端到端 1.5–2<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> 的加速。结合 vLLM + INT8：生成从瓶颈变为与训练持平。</p><p>局限：（1）draft 模型必须共享 tokenizer。（2）在高温下效果较差（draft 模型不够准确）。（3）draft 模型需要额外的 GPU 显存。（4）超过 <math alttext="k=5" display="inline"><semantics><mrow><mi>k</mi><mo>=</mo><mn>5</mn></mrow></semantics></math> 后收益递减（接受率下降）。</p><p><em>复习：第 2 章与第 11 章（系统基础；系统架构）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q45">
<summary class="hh-quiz-question">Q45：解释 roofline 模型。如何判断一个 kernel 是计算受限还是内存受限？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>roofline 模型把可达到的性能（FLOPS）画成算术强度的函数（每字节内存流量对应的 FLOPS）。</p><p>两种区域：</p><ul><li>内存受限（拐点左侧）：性能受限于向计算单元喂数据的速度。实际 FLOPS = 带宽 <math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> 算术强度。GPU 利用率 <math alttext="&lt;" display="inline"><semantics><mo>&lt;</mo></semantics></math> 100%。</li><li>计算受限（拐点右侧）：性能受限于峰值 FLOPS。内存足够快。GPU 达到最大利用率。</li></ul><p>拐点：峰值 FLOPS / 峰值带宽。对 A100：312 TF / 2 TB/s = 156 FLOP/byte。</p><p>LLM 操作：</p><ul><li>自回归生成（batch=1）：读取 140GB 权重，做 140G FLOPs = 1 FLOP/byte。<em>极度</em>内存受限（比拐点低 156<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>）。GPU 利用率仅 0.6%。</li><li>训练前向（batch=128，seq=2048）：算术强度 <math alttext="\approx 200" display="inline"><semantics><mrow><mphantom></mphantom><mo>≈</mo><mn>200</mn></mrow></semantics></math>+ FLOP/byte。计算受限。接近峰值利用率。</li><li>Attention（长序列）：<math alttext="O(n^{2}d)" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><msup><mi>n</mi><mn>2</mn></msup><mo lspace="0em" rspace="0em">​</mo><mi>d</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></semantics></math> FLOPs / <math alttext="O(n^{2}+nd)" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><msup><mi>n</mi><mn>2</mn></msup><mo>+</mo><mrow><mi>n</mi><mo lspace="0em" rspace="0em">​</mo><mi>d</mi></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 字节。<math alttext="n" display="inline"><semantics><mi>n</mi></semantics></math> 长时：计算受限。<math alttext="n" display="inline"><semantics><mi>n</mi></semantics></math> 短时：内存受限。FlashAttention 无论如何都让其留在 SRAM。</li></ul><p>实用法：若 kernel 内存受限，减少内存流量（量化、缓存、分块）。若计算受限，减少 FLOPs（剪枝、蒸馏、降低精度）。</p><p><em>复习：第 2 章（LLM 的系统基础）。</em></p>
</div>
</details>

<details class="hh-quiz" id="q46">
<summary class="hh-quiz-question">Q46：连续批处理（Continuous Batching）如何工作？为何它对 RLHF 生成至关重要？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>静态批处理：启动 <math alttext="B" display="inline"><semantics><mi>B</mi></semantics></math> 条序列。等所有序列结束。若一条生成 500 Token、另一条只生成 50 Token，那 50 Token 序列的 GPU 槽位会闲置 450 Token 的时间。</p><p>连续批处理（迭代级调度）：每个生成步后检查哪些序列已完成。立即在腾出的槽位插入新序列。GPU 槽位从不闲置。</p><p>为何对 RLHF 至关重要：</p><ol><li>RLHF 生成多样输出（高 temperature）。长度方差巨大——有些回答 50 Token，有些 2000+。</li><li>无连续批处理：平均利用率 <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>40–50%（等最慢的序列）。</li><li>有连续批处理：利用率 <math alttext="&gt;" display="inline"><semantics><mo>&gt;</mo></semantics></math>90%。吞吐高 2–3<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>。</li><li>RLHF 需要大 batch（每步 128+ 回答）。用静态批处理生成 128 个回答需要 max_tokens <math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> 128 个串行步。连续批处理摊销了这一开销。</li></ol><p>实现：vLLM 的调度器在每个解码步后检查。抢占：若有新的高优先级请求到达且显存已满，可将低优先级序列的 KV cache 换出到 CPU，稍后恢复。</p><p><em>复习：第 2 章与第 11 章（系统基础；系统架构）。</em></p>
</div>
</details>

## Transformer 架构问题

<details class="hh-quiz" id="ch28-s11-p1">
<summary class="hh-quiz-question">Q：为什么在现代 LLM 中 RoPE 的主导地位胜过可学习的绝对位置嵌入？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>RoPE 通过旋转矩阵将<em>相对</em>位置直接编码到 Q/K 点积中。关键优势：</p><ol><li>Attention 分数只依赖于相对距离 <math alttext="i-j" display="inline"><semantics><mrow><mi>i</mi><mo>−</mo><mi>j</mi></mrow></semantics></math>，而不依赖绝对位置——这能更好地泛化到未见过的序列长度。</li><li>可通过频率缩放（NTK-aware、YaRN）扩展到超出训练长度，无需重新训练。</li><li>无需额外参数（旋转由位置索引确定性地给出）。</li><li>可学习的绝对嵌入固定到训练长度，无法外推——在 4K 上下文训练的模型在 8K 时会失败。</li></ol><p><em>复习：第 1 章（LLM 架构与优化方法）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s11-p2">
<summary class="hh-quiz-question">Q：解释 SwiGLU，以及它为何在现代 Transformer 中取代了 ReLU。</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>SwiGLU：<math alttext="\text{FFN}(x)=W_{2}(\text{Swish}(W_{1}x)\odot W_{3}x)" display="inline"><semantics><mrow><mrow><mtext>FFN</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><msub><mi>W</mi><mn>2</mn></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mrow><mrow><mrow><mtext>Swish</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mrow><msub><mi>W</mi><mn>1</mn></msub><mo lspace="0em" rspace="0em">​</mo><mi>x</mi></mrow><mo rspace="0.055em" stretchy="false">)</mo></mrow></mrow><mo rspace="0.222em">⊙</mo><msub><mi>W</mi><mn>3</mn></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mi>x</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math>，其中 <math alttext="\text{Swish}(x)=x\cdot\sigma(x)" display="inline"><semantics><mrow><mrow><mtext>Swish</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mrow><mi>x</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><mi>σ</mi></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math>。</p><p>为什么它更好：</p><ul><li>该<em>门控</em>机制（<math alttext="\odot W_{3}x" display="inline"><semantics><mrow><mphantom></mphantom><mo lspace="0.222em" rspace="0.222em">⊙</mo><mrow><msub><mi>W</mi><mn>3</mn></msub><mo lspace="0em" rspace="0em">​</mo><mi>x</mi></mrow></mrow></semantics></math>）让网络可以有选择地抑制或放大维度——比逐点 ReLU 更具表达力。</li><li>Swish 是平滑的（不存在 ReLU 零梯度区那样的死神经元）。</li><li>经验上：在相同 FLOP 预算下，语言建模基准上提升 1–2%。</li><li>权衡：需要 3 个权重矩阵而非 2 个（通过将隐藏维度从 <math alttext="4d" display="inline"><semantics><mrow><mn>4</mn><mo lspace="0em" rspace="0em">​</mo><mi>d</mi></mrow></semantics></math> 降到 <math alttext="8d/3" display="inline"><semantics><mrow><mrow><mn>8</mn><mo lspace="0em" rspace="0em">​</mo><mi>d</mi></mrow><mo>/</mo><mn>3</mn></mrow></semantics></math> 来解决）。</li></ul><p><em>复习：第 1 章（LLM 架构与优化方法）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s11-p3">
<summary class="hh-quiz-question">Q：什么是分组查询注意力（Grouped Query Attention, GQA），Llama-3 为何采用它？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>标准 MHA：<math alttext="H" display="inline"><semantics><mi>H</mi></semantics></math> 个 query 头、<math alttext="H" display="inline"><semantics><mi>H</mi></semantics></math> 个 key 头、<math alttext="H" display="inline"><semantics><mi>H</mi></semantics></math> 个 value 头。GQA：<math alttext="H" display="inline"><semantics><mi>H</mi></semantics></math> 个 query 头但只有 <math alttext="G&lt;H" display="inline"><semantics><mrow><mi>G</mi><mo>&lt;</mo><mi>H</mi></mrow></semantics></math> 个 key/value 头（在 query 组间共享）。</p><p>Llama-3 70B：64 个 query 头、8 个 KV 头（每个 KV 头被 8 个 query 头共享）。</p><p>优势：</p><ul><li>KV cache 大小缩小 <math alttext="H/G=8\times" display="inline"><semantics><mrow><mi>H</mi><mo>/</mo><mi>G</mi><mo>=</mo><mn>8</mn><mo lspace="0.222em">×</mo></mrow></semantics></math>——对推理至关重要（长序列下 KV cache 是主导的显存开销）。</li><li>质量损失极小（基准上 <math alttext="&lt;" display="inline"><semantics><mo>&lt;</mo></semantics></math>0.5%），因为 KV 模式在不同头之间高度相关。</li><li>推理吞吐与 KV cache 的减少成比例增长（更多序列能装进显存 = 更大的 batch size）。</li></ul><p><em>复习：第 1 章（LLM 架构与优化方法）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s11-p4">
<summary class="hh-quiz-question">Q：为何 decoder-only 架构在 LLM 中胜过了 encoder-decoder？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<ol><li>统一目标：预训练 = 微调 = 推理，都使用下一个 Token 预测。不存在架构不匹配。</li><li>参数效率：所有参数都对生成有贡献。在 encoder-decoder 中，纯生成任务下 encoder 参数被“浪费”。</li><li>扩展更简单：一个模型、一个损失函数、一套超参数需要调。</li><li>KV cache 效率：decoder-only 只有一份 KV cache；encoder-decoder 有两份（encoder + decoder 的交叉注意力）。</li><li>涌现的少样本：decoder-only 天然支持上下文学习（把示例前置到 Prompt 中）。</li></ol><p>对于固定输入长度的 seq2seq 任务（如翻译），encoder-decoder 仍然占优，但这部分在 LLM 用例中占比正在缩小。</p><p><em>复习：第 1 章（LLM 架构与优化方法）。</em></p>
</div>
</details>

## FlashAttention 问题

<details class="hh-quiz" id="ch28-s12-p1">
<summary class="hh-quiz-question">Q：FlashAttention 计算的结果与标准 attention 相同，但快 2–4<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>。如果 FLOPs 数相同，这怎么可能？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>FlashAttention 之所以更快，是因为它减少了 <em>HBM 显存流量</em>，而非 FLOPs。标准 attention 在 HBM（慢）中物化 <math alttext="n\times n" display="inline"><semantics><mrow><mi>n</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mi>n</mi></mrow></semantics></math> 的 attention 矩阵，再读回来做 Softmax，再读一次做 <math alttext="PV" display="inline"><semantics><mrow><mi>P</mi><mo lspace="0em" rspace="0em">​</mo><mi>V</mi></mrow></semantics></math> 乘法——总共在 <math alttext="O(n^{2})" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msup><mi>n</mi><mn>2</mn></msup><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 数据上有 4 次 HBM 往返。</p><p>FlashAttention 对计算进行分块，使 <math alttext="n\times n" display="inline"><semantics><mrow><mi>n</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mi>n</mi></mrow></semantics></math> 矩阵完全在 SRAM（快，19 TB/s）中计算并消费，从不写入 HBM（2 TB/s）。“在线 Softmax”技巧通过维护滚动统计量来实现这一点。</p><p>结果：HBM 流量从 <math alttext="O(n^{2}d)" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><msup><mi>n</mi><mn>2</mn></msup><mo lspace="0em" rspace="0em">​</mo><mi>d</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 降到 <math alttext="O(n^{2}d/M)" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mrow><msup><mi>n</mi><mn>2</mn></msup><mo lspace="0em" rspace="0em">​</mo><mi>d</mi></mrow><mo>/</mo><mi>M</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></semantics></math>，其中 <math alttext="M" display="inline"><semantics><mi>M</mi></semantics></math> 为 SRAM 大小。相同 FLOPs，显存流量减少 10–50<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>，<math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 实际墙钟加速 2–4<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>。</p><p><em>复习：第 1 章与第 2 章（LLM 架构；系统基础）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s12-p2">
<summary class="hh-quiz-question">Q：为什么 FlashAttention 对 FFN 层没有帮助？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>FFN 层是<em>计算受限</em>的，而非内存受限。它们的算术强度（大 batch GEMM 下为 <math alttext="I\approx 300" display="inline"><semantics><mrow><mi>I</mi><mo>≈</mo><mn>300</mn></mrow></semantics></math> FLOP/byte）已经高于 roofline 的峰值拐点（A100 上为 156 FLOP/byte）。</p><p>FlashAttention 对 attention 有帮助，是因为 attention 深度内存受限（<math alttext="I\approx 1" display="inline"><semantics><mrow><mi>I</mi><mo>≈</mo><mn>1</mn></mrow></semantics></math>–<math alttext="60" display="inline"><semantics><mn>60</mn></semantics></math> FLOP/byte）。通过把数据留在 SRAM 中，它消除了内存瓶颈。</p><p>对 FFN 而言：瓶颈已经是张量核心（Tensor Core）（而非内存带宽），因此减少显存流量无济于事。相反，FFN 受益于量化（缩小权重体积 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 提高算术强度）以及更大的 batch size。</p><p><em>复习：第 1 章与第 2 章（LLM 架构；系统基础）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s12-p3">
<summary class="hh-quiz-question">Q：解释 online softmax 技巧，以及它为何对 FlashAttention 至关重要。</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>标准 softmax 在计算任何输出之前需要全局最大值 <math alttext="m=\max_{j}x_{j}" display="inline"><semantics><mrow><mi>m</mi><mo>=</mo><mrow><msub><mi>max</mi><mi>j</mi></msub><mo lspace="0.167em">⁡</mo><msub><mi>x</mi><mi>j</mi></msub></mrow></mrow></semantics></math>——这要求先看到全部 <math alttext="n" display="inline"><semantics><mi>n</mi></semantics></math> 个 attention 分数，从而被迫物化完整的 <math alttext="n\times n" display="inline"><semantics><mrow><mi>n</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mi>n</mi></mrow></semantics></math> 矩阵。</p><p>online softmax 技巧按块顺序处理，维护一个滚动的 <math alttext="(m,\ell,O)" display="inline"><semantics><mrow><mo stretchy="false">(</mo><mi>m</mi><mo>,</mo><mi mathvariant="normal">ℓ</mi><mo>,</mo><mi>O</mi><mo stretchy="false">)</mo></mrow></semantics></math> 状态：</p><ol><li>处理新块 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math>更新滚动最大值：<math alttext="m_{\text{new}}=\max(m_{\text{old}},\max(s_{\text{new}}))" display="inline"><semantics><mrow><msub><mi>m</mi><mtext>new</mtext></msub><mo>=</mo><mrow><mi>max</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>m</mi><mtext>old</mtext></msub><mo>,</mo><mrow><mi>max</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>s</mi><mtext>new</mtext></msub><mo stretchy="false">)</mo></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math></li><li>重新缩放旧的求和：<math alttext="\ell_{\text{new}}=e^{m_{\text{old}}-m_{\text{new}}}\cdot\ell_{\text{old}}+\text{new terms}" display="inline"><semantics><mrow><msub><mi mathvariant="normal">ℓ</mi><mtext>new</mtext></msub><mo>=</mo><mrow><mrow><msup><mi>e</mi><mrow><msub><mi>m</mi><mtext>old</mtext></msub><mo>−</mo><msub><mi>m</mi><mtext>new</mtext></msub></mrow></msup><mo lspace="0.222em" rspace="0.222em">⋅</mo><msub><mi mathvariant="normal">ℓ</mi><mtext>old</mtext></msub></mrow><mo>+</mo><mtext>new terms</mtext></mrow></mrow></semantics></math></li><li>重新缩放输出：<math alttext="O_{\text{new}}=\text{rescaled}(O_{\text{old}})+\text{new contribution}" display="inline"><semantics><mrow><msub><mi>O</mi><mtext>new</mtext></msub><mo>=</mo><mrow><mrow><mtext>rescaled</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>O</mi><mtext>old</mtext></msub><mo stretchy="false">)</mo></mrow></mrow><mo>+</mo><mtext>new contribution</mtext></mrow></mrow></semantics></math></li></ol><p>这在数学上精确——没有近似。它使逐块处理成为可能，每个块都能装进 SRAM，从不需要把完整的 <math alttext="n\times n" display="inline"><semantics><mrow><mi>n</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mi>n</mi></mrow></semantics></math> 矩阵放进内存。</p><p><em>复习：第 1 章与第 2 章（LLM 架构；系统基础）。</em></p>
</div>
</details>

## LoRA 与 PEFT 问题

<details class="hh-quiz" id="ch28-s13-p1">
<summary class="hh-quiz-question">Q：LoRA 为何有效？有什么理论洞见能解释低秩更新？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>Aghajanyan 等人 [3] 表明，微调运行在极低的<em>内在维度</em>上——微调任务的有效参数空间远小于模型的总参数量。一个 175B 模型对某个给定任务的内在维度可能只有 <math alttext="&lt;" display="inline"><semantics><mo>&lt;</mo></semantics></math>10,000。</p><p>LoRA 直接利用了这一点：通过把更新约束到秩 <math alttext="r" display="inline"><semantics><mi>r</mi></semantics></math>（<math alttext="W^{\prime}=W+BA" display="inline"><semantics><mrow><msup><mi>W</mi><mo>′</mo></msup><mo>=</mo><mrow><mi>W</mi><mo>+</mo><mrow><mi>B</mi><mo lspace="0em" rspace="0em">​</mo><mi>A</mi></mrow></mrow></mrow></semantics></math>、<math alttext="B\in\mathbb{R}^{d\times r}" display="inline"><semantics><mrow><mi>B</mi><mo>∈</mo><msup><mi>ℝ</mi><mrow><mi>d</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mi>r</mi></mrow></msup></mrow></semantics></math>），它把学习限制在每个权重矩阵的 <math alttext="r" display="inline"><semantics><mi>r</mi></semantics></math> 维子空间内。由于真实的任务子空间是低维的，这样做几乎不损失什么，同时把可训练参数减少 100–1000<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>。</p><p>直觉：微调不改变模型“知道”什么（满秩的 <math alttext="W" display="inline"><semantics><mi>W</mi></semantics></math> 保持冻结）；它只调整<em>如何</em>把已有知识组合起来用于新任务——一个低秩扰动。</p><p><em>复习：第 1 章（LLM 架构与优化方法）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s13-p2">
<summary class="hh-quiz-question">Q：对 70B 模型比较 QLoRA、完整 LoRA 与全量微调。各自应在何时选用？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>决策树：（1）若任务需要深度改变知识 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 全量微调。（2）若适配新的风格/格式 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> LoRA。（3）若显存受限或需快速迭代 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> QLoRA。（4）若关心秩：从 <math alttext="r=16" display="inline"><semantics><mrow><mi>r</mi><mo>=</mo><mn>16</mn></mrow></semantics></math> 开始；若训练损失在高于全量微调的水平上停滞，则增秩。</p><p><em>复习：第 1 章与第 10 章（LLM 架构；监督微调最佳实践）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s13-p3">
<summary class="hh-quiz-question">Q：什么是 DoRA，它为何优于标准 LoRA？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>DoRA（Weight-Decomposed Low-Rank Adaptation）把 <math alttext="W" display="inline"><semantics><mi>W</mi></semantics></math> 分解为幅度 <math alttext="\|W\|" display="inline"><semantics><mrow><mo stretchy="false">‖</mo><mi>W</mi><mo stretchy="false">‖</mo></mrow></semantics></math> 与方向 <math alttext="W/\|W\|" display="inline"><semantics><mrow><mi>W</mi><mo>/</mo><mrow><mo stretchy="false">‖</mo><mi>W</mi><mo stretchy="false">‖</mo></mrow></mrow></semantics></math>，然后只对方向分量应用 LoRA：</p><div class="hh-equation"><math alttext="W^{\prime}=m\odot\frac{W+BA}{\|W+BA\|}" display="block"><semantics><mrow><msup><mi>W</mi><mo>′</mo></msup><mo>=</mo><mrow><mi>m</mi><mo lspace="0.222em" rspace="0.222em">⊙</mo><mfrac><mrow><mi>W</mi><mo>+</mo><mrow><mi>B</mi><mo lspace="0em" rspace="0em">​</mo><mi>A</mi></mrow></mrow><mrow><mo stretchy="false">‖</mo><mrow><mi>W</mi><mo>+</mo><mrow><mi>B</mi><mo lspace="0em" rspace="0em">​</mo><mi>A</mi></mrow></mrow><mo stretchy="false">‖</mo></mrow></mfrac></mrow></mrow></semantics></math></div><p>其中 <math alttext="m" display="inline"><semantics><mi>m</mi></semantics></math>（幅度）也可训练，但只是每个输出神经元上的一个简单标量。</p><p>为何有帮助：全量微调会自然地独立更新幅度与方向。标准 LoRA 把二者耦合在一起（低秩更新以一种受限的方式同时改变两者）。DoRA 将它们解耦，使 LoRA 获得与全量微调相同的“自由度”结构。结果：在推理时无额外计算量（合并适配器）的情况下，推理任务上提升 1–3%。</p><p><em>复习：第 1 章（LLM 架构与优化方法）。</em></p>
</div>
</details>

## 模型压缩问题

<details class="hh-quiz" id="ch28-s14-p1">
<summary class="hh-quiz-question">Q：解释 AWQ。为什么保护 1% 的权重就能保住 99% 的质量？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>AWQ（Activation-Aware Weight Quantization）观察到权重重要性高度不均匀：与大幅激活值相乘的权重对输出的贡献不成比例地大。</p><p>关键洞见：<math alttext="\|W\cdot X\|" display="inline"><semantics><mrow><mo stretchy="false">‖</mo><mrow><mi>W</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><mi>X</mi></mrow><mo stretchy="false">‖</mo></mrow></semantics></math> 同时取决于 <math alttext="W" display="inline"><semantics><mi>W</mi></semantics></math> 与 <math alttext="X" display="inline"><semantics><mi>X</mi></semantics></math>。一个小权重乘以大幅激活值，比一个大权重乘以接近零的激活值更重要。</p><p>AWQ 找出前 1% 的“显著”通道（在校准数据上激活幅度始终很大的那些），并通过缩放来保护它们：量化前把显著通道乘以因子 <math alttext="s&gt;1" display="inline"><semantics><mrow><mi>s</mi><mo>&gt;</mo><mn>1</mn></mrow></semantics></math>（然后在激活中除以 <math alttext="s" display="inline"><semantics><mi>s</mi></semantics></math>）。这降低了重要通道的相对量化误差。</p><p>结果：在 70B 模型上实现 4-bit 量化，质量损失 <math alttext="&lt;" display="inline"><semantics><mo>&lt;</mo></semantics></math>1%，因为 99% 的非显著权重可以承受激进的量化。</p><p><em>复习：第 1 章与第 2 章（LLM 架构；系统基础）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s14-p2">
<summary class="hh-quiz-question">Q：什么时候该用 FP8、4-bit 量化或 BF16？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<ul><li>BF16：训练（RLHF 中的策略模型），精度重要时使用。任何被梯度更新的模型的默认选择。</li><li>FP8（E4M3）：配合 Transformer Engine 在 H100 上训练（2<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> 吞吐，<math alttext="&lt;" display="inline"><semantics><mo>&lt;</mo></semantics></math>0.5% 质量损失）。也用于在 H100 上需要最大吞吐量的推理。</li><li>INT8/FP8 推理：RLHF 中的冻结模型（参考模型、奖励模型）——不被训练，因此降低精度是安全的。</li><li>4-bit（AWQ/GPTQ）：规模化推理服务。部署时显存/质量的最佳权衡。也用于 QLoRA 的基座模型。</li><li>2-bit：实验性；显存极度受限的边缘部署。质量损失 5–10%。</li></ul><p>规则：推理时尽可能激进地量化，训练时保持 BF16（H100 上可用 FP8）。</p><p><em>复习：第 2 章（LLM 的系统基础）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s14-p3">
<summary class="hh-quiz-question">Q：解释 NVIDIA 2:4 结构化稀疏。加速比和约束是什么？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>2:4 稀疏意味着：每 4 个连续元素中，恰好有 2 个必须为零。这在权重层面强制执行。</p><p>硬件支持：A100/H100 的张量核心拥有专用的 2:4 稀疏 GEMM 指令，会跳过零元素，恰好实现 2<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> 吞吐，且没有软件开销。</p><p>约束：你必须在这种特定模式下实现<em>恰好</em> 50% 的稀疏度。不能是 30% 或 70% 的稀疏度；也不能是任意的稀疏模式。剪枝必须遵守 4 元素分组结构。</p><p>如何实现：训练后（或微调期间），对每组 4 个权重，把绝对值最小的 2 个置零。然后微调数百步以恢复质量。质量损失：对大型模型（70B+）通常为 <math alttext="&lt;" display="inline"><semantics><mo>&lt;</mo></semantics></math>1%。</p><p><em>复习：第 2 章（LLM 的系统基础）。</em></p>
</div>
</details>

## 专家混合（Mixture of Experts）问题

<details class="hh-quiz" id="ch28-s15-p1">
<summary class="hh-quiz-question">Q：Mixtral 8x7B 总参数 47B，但每个 Token 只有 13B 处于激活状态。解释这如何工作，以及它为何高效。</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>Mixtral 把每个 FFN 层替换为 8 个并行的专家 FFN（每个在 FFN 部分约 <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>7B 参数）。一个路由网络为每个 Token 选择 Top-2 专家。</p><p>为何总计 47B：Attention 层是共享的（不复制）= <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>5B。FFN 专家：8 <math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math><math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>5.25B = 42B。总计：<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>47B。</p><p>为何激活 13B：每个 Token 只有 2 个专家被激活。激活参数 = attention（<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>5B）+ 2 个 FFN 专家（<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>2 <math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> 5.25B）<math alttext="\approx" display="inline"><semantics><mo>≈</mo></semantics></math> 13B。</p><p>为何高效：计算成本随<em>激活</em>参数（13B）扩展，与 13B 的稠密模型相当。但容量（存储的知识）随<em>总</em>参数（47B）扩展，与远更大的模型相当。结果：Mixtral 以 13B 的计算成本达到 Llama-2 70B 的质量。</p><p>显存成本：仍然需要把全部 47B 参数放在显存中（所有专家都要加载），因此显存 = 47B 模型，但计算 = 13B 模型。</p><p><em>复习：第 1 章（LLM 架构与优化方法）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s15-p2">
<summary class="hh-quiz-question">Q：MoE 中的负载均衡问题是什么，如何解决？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>若无约束，路由器倾向于把大多数 Token 送给 1–2 个“热门”专家（富者愈富的动态）。这会导致：</p><ul><li>容量浪费：8 个专家中有 6 个未被使用，模型实际上退化为 2 个专家的规模。</li><li>计算不均衡：若每个专家位于不同的 GPU 上，热门专家成为瓶颈，而其他专家闲置。</li></ul><p>解决方案：辅助负载均衡损失：<math alttext="\mathcal{L}_{\text{bal}}=\alpha\cdot N\sum_{i=1}^{N}f_{i}\cdot p_{i}" display="inline"><semantics><mrow><mi>ℒ</mi><msub><mrow></mrow><mtext>bal</mtext></msub><mo>=</mo><mi>α</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><mi>N</mi><mo>∑</mo><msub><mrow></mrow><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow></msub><msup><mrow></mrow><mi>N</mi></msup><mi>f</mi><msub><mrow></mrow><mi>i</mi></msub><mo lspace="0.222em" rspace="0.222em">⋅</mo><mi>p</mi><msub><mrow></mrow><mi>i</mi></msub></mrow></semantics></math>，其中 <math alttext="f_{i}" display="inline"><semantics><msub><mi>f</mi><mi>i</mi></msub></semantics></math> = 路由到专家 <math alttext="i" display="inline"><semantics><mi>i</mi></semantics></math> 的 Token 比例，<math alttext="p_{i}" display="inline"><semantics><msub><mi>p</mi><mi>i</mi></msub></semantics></math> = 专家 <math alttext="i" display="inline"><semantics><mi>i</mi></semantics></math> 的平均路由概率。这会惩罚不均匀的分布。</p><p>替代方案：专家容量因子——对每个 batch 中每个专家的最大 Token 数设硬上限。溢出的 Token 会被丢弃或重新路由。</p><p>典型的 <math alttext="\alpha" display="inline"><semantics><mi>α</mi></semantics></math>：0.01–0.1（小到不损害主损失，大到能防止坍缩）。</p><p><em>复习：第 1 章（LLM 架构与优化方法）。</em></p>
</div>
</details>

## 训练中的多样性问题

<details class="hh-quiz" id="ch28-s16-p1">
<summary class="hh-quiz-question">Q：如果 GRPO 组内全部 N 个回答完全相同，会发生什么？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>若全部 <math alttext="N" display="inline"><semantics><mi>N</mi></semantics></math> 个回答完全相同：所有奖励 <math alttext="r_{i}" display="inline"><semantics><msub><mi>r</mi><mi>i</mi></msub></semantics></math> 都相等，因此 <math alttext="\sigma_{G}=0" display="inline"><semantics><mrow><msub><mi>σ</mi><mi>G</mi></msub><mo>=</mo><mn>0</mn></mrow></semantics></math> 与优势 <math alttext="\hat{A}_{i}=(r_{i}-\mu_{G})/\sigma_{G}" display="inline"><semantics><mrow><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>i</mi></msub><mo>=</mo><mrow><mrow><mo stretchy="false">(</mo><mrow><msub><mi>r</mi><mi>i</mi></msub><mo>−</mo><msub><mi>μ</mi><mi>G</mi></msub></mrow><mo stretchy="false">)</mo></mrow><mo>/</mo><msub><mi>σ</mi><mi>G</mi></msub></mrow></mrow></semantics></math> 无定义（除以零）。实践中，实现会为所有项设置 <math alttext="\hat{A}_{i}=0" display="inline"><semantics><mrow><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>i</mi></msub><mo>=</mo><mn>0</mn></mrow></semantics></math>，意味着 零学习信号——这一步被浪费了。</p><p>预防：</p><ol><li>Temperature：生成时使用 <math alttext="\tau=0.7" display="inline"><semantics><mrow><mi>τ</mi><mo>=</mo><mn>0.7</mn></mrow></semantics></math>–<math alttext="1.0" display="inline"><semantics><mn>1.0</mn></semantics></math>（不要贪心）。</li><li>较大的 <math alttext="N" display="inline"><semantics><mi>N</mi></semantics></math>：<math alttext="N=8" display="inline"><semantics><mrow><mi>N</mi><mo>=</mo><mn>8</mn></mrow></semantics></math>–<math alttext="16" display="inline"><semantics><mn>16</mn></semantics></math> 提高出现多样回答的概率。</li><li>重复拒绝：DAPO 的做法——拒绝重复的回答并重新采样。</li><li>频率惩罚：生成时惩罚重复出现的 n-gram。</li><li>监控：跟踪每组的唯一回答比例。若 <math alttext="&lt;" display="inline"><semantics><mo>&lt;</mo></semantics></math>50%，则提高 temperature。</li></ol><p><em>复习：第 7 章（GRPO）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s16-p2">
<summary class="hh-quiz-question">Q：解释 RLHF 中的多样性—质量权衡。如何检测模式坍缩？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>权衡：高多样性（高熵/temperature）= 回答多样但可能随机/低质量。低多样性 = 一致但重复、被奖励作弊的回答。</p><p>检测模式坍缩（这些都应在训练中监控）：</p><ol><li>回答熵：计算每个 Token 的熵 <math alttext="H=-\sum p_{i}\log p_{i}" display="inline"><semantics><mrow><mi>H</mi><mo rspace="0em">=</mo><mo lspace="0em" rspace="0.055em">−</mo><mo>∑</mo><mi>p</mi><msub><mrow></mrow><mi>i</mi></msub><mi>log</mi><mi>p</mi><msub><mrow></mrow><mi>i</mi></msub></mrow></semantics></math>。若迅速下降 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 坍缩。</li><li>唯一 n-gram 比例：对同一 Prompt 的回答中唯一 4-gram 的占比。健康值：<math alttext="&gt;" display="inline"><semantics><mo>&gt;</mo></semantics></math>0.6。</li><li>奖励分布宽度：若 <math alttext="\sigma(\text{rewards})" display="inline"><semantics><mrow><mi>σ</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mtext>rewards</mtext><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 收缩到接近零 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 所有回答质量相同 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 很可能完全相同。</li><li>KL 散度：若 <math alttext="D_{\text{KL}}[\pi_{\theta}\|\pi_{\text{ref}}]" display="inline"><semantics><mrow><msub><mi>D</mi><mtext>KL</mtext></msub><mrow><mo stretchy="false">[</mo><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0.167em">∥</mo><msub><mi>π</mi><mtext>ref</mtext></msub><mo stretchy="false">]</mo></mrow></mrow></semantics></math> 迅速增长，说明策略正在远离参考 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 通常走向一个狭窄的模式。</li><li>长度直方图：若所有回答收敛到相同长度 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 模板化行为。</li></ol><p>修复：增大 KL 系数 <math alttext="\beta" display="inline"><semantics><mi>β</mi></semantics></math>，增大熵奖励，提高采样 temperature，或回滚到更早的 checkpoint。</p><p><em>复习：第 7 章与第 9 章（GRPO；奖励模型训练）。</em></p>
</div>
</details>

## 推测解码（Speculative Decoding）问题

<details class="hh-quiz" id="ch28-s17-p1">
<summary class="hh-quiz-question">Q：投机解码声称“无质量损失”。用不同的方式生成 Token 怎么可能产生相同的输出分布？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>接受/拒绝机制保证了分布等价性：</p><p>对每个草稿 Token <math alttext="\hat{x}" display="inline"><semantics><mover accent="true"><mi>x</mi><mo>^</mo></mover></semantics></math>，其草稿概率为 <math alttext="q(\hat{x})" display="inline"><semantics><mrow><mi>q</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mover accent="true"><mi>x</mi><mo>^</mo></mover><mo stretchy="false">)</mo></mrow></mrow></semantics></math>、目标概率为 <math alttext="p(\hat{x})" display="inline"><semantics><mrow><mi>p</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mover accent="true"><mi>x</mi><mo>^</mo></mover><mo stretchy="false">)</mo></mrow></mrow></semantics></math>：</p><ul><li>以概率 <math alttext="\min(1,p(\hat{x})/q(\hat{x}))" display="inline"><semantics><mrow><mi>min</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mn>1</mn><mo>,</mo><mrow><mrow><mi>p</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mover accent="true"><mi>x</mi><mo>^</mo></mover><mo stretchy="false">)</mo></mrow></mrow><mo>/</mo><mrow><mi>q</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mover accent="true"><mi>x</mi><mo>^</mo></mover><mo stretchy="false">)</mo></mrow></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 接受</li><li>被拒绝时：从 <em>残差分布</em><math alttext="\propto\max(0,p(x)-q(x))" display="inline"><semantics><mrow><mphantom></mphantom><mo>∝</mo><mrow><mi>max</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mn>0</mn><mo>,</mo><mrow><mrow><mi>p</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow><mo>−</mo><mrow><mi>q</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math> 中采样</li></ul><p>这在数学上等价于直接从 <math alttext="p" display="inline"><semantics><mi>p</mi></semantics></math>（目标分布）采样。证明梗概：输出 Token <math alttext="x" display="inline"><semantics><mi>x</mi></semantics></math> 的概率为 <math alttext="q(x)\cdot\min(1,p(x)/q(x))+P(\text{reject})\cdot\frac{\max(0,p(x)-q(x))}{\sum_{y}\max(0,p(y)-q(y))}=p(x)" display="inline"><semantics><mrow><mrow><mrow><mrow><mi>q</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo rspace="0.055em" stretchy="false">)</mo></mrow></mrow><mo rspace="0.222em">⋅</mo><mrow><mi>min</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mn>1</mn><mo>,</mo><mrow><mrow><mi>p</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow><mo>/</mo><mrow><mi>q</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow><mo>+</mo><mrow><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mtext>reject</mtext><mo rspace="0.055em" stretchy="false">)</mo></mrow></mrow><mo rspace="0.222em">⋅</mo><mfrac><mrow><mi>max</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mn>0</mn><mo>,</mo><mrow><mrow><mi>p</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow><mo>−</mo><mrow><mi>q</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow><mrow><mstyle displaystyle="false"><msub><mo maxsize="0.700em" minsize="0.700em" stretchy="true">∑</mo><mi>y</mi></msub></mstyle><mrow><mi>max</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mn>0</mn><mo>,</mo><mrow><mrow><mi>p</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>y</mi><mo stretchy="false">)</mo></mrow></mrow><mo>−</mo><mrow><mi>q</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>y</mi><mo stretchy="false">)</mo></mrow></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></mfrac></mrow></mrow><mo>=</mo><mrow><mi>p</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math>。</p><p>加速来自于摊销：当草稿质量好（接受率高）时，一次目标前向就能确认多个 Token。该保证与草稿质量无关——草稿差只会带来更低的加速（更多拒绝），而不会降低质量。</p><p><em>复习：第 2 章（LLM 的系统基础）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s17-p2">
<summary class="hh-quiz-question">Q：在投机解码中比较 Medusa 与 Eagle。各自应在何时选用？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>Medusa：向目标模型添加 <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 个并行预测头。每个头独立地预测位置 <math alttext="t+i" display="inline"><semantics><mrow><mi>t</mi><mo>+</mo><mi>i</mi></mrow></semantics></math> 上的 Token。<em>优点</em>：无需单独的模型，显存开销 <math alttext="&lt;" display="inline"><semantics><mo>&lt;</mo></semantics></math>1%。<em>缺点</em>：各头独立预测——无法把位置 <math alttext="t+2" display="inline"><semantics><mrow><mi>t</mi><mo>+</mo><mn>2</mn></mrow></semantics></math> 建立在位置 <math alttext="t+1" display="inline"><semantics><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></semantics></math> 上已预测的内容之上。接受率：60–80%。</p><p>Eagle：在目标模型的隐藏状态上接一个轻量级自回归解码器。草稿 Token 以自回归方式生成（每个都以前一个为条件）。<em>优点</em>：捕捉 Token 间的依赖关系 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 85–95% 的接受率。<em>缺点</em>：显存略多（小型解码器），且草稿生成是串行的。</p><p>选择 Medusa 的情形：显存极其紧张；集成简单；适度加速即可（2–2.5<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>）。</p><p>选择 Eagle 的情形：需要最大加速（3–4<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>）；能承受额外的小模型；延迟关键的单流生成。</p><p>选择 N-gram 的情形：输出重复（代码、结构化数据）；零成本；无需训练。</p><p><em>复习：第 2 章（LLM 的系统基础）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s17-p3">
<summary class="hh-quiz-question">Q：为什么投机解码在大 batch size 下没有帮助？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>在大 batch size（<math alttext="\geq" display="inline"><semantics><mo>≥</mo></semantics></math>64）下，自回归生成已经是<em>计算高效</em>的：权重读取成本被许多序列摊销。算术强度接近 roofline 的峰值拐点。</p><p>投机解码引入了额外开销：</p><ol><li>草稿生成成本（即使是小模型，在大 batch 下也不是免费的）</li><li>验证前向每序列要处理<math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 个额外的 Token（batch <math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math><math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> Token）</li><li>草稿模型或 Medusa 头的显存占用</li><li>被拒绝的 Token 浪费计算</li></ol><p>在 batch=1（延迟受限、内存受限）时：投机把每步 1 个 Token 变成每步 3–4 个 Token——收益巨大。</p><p>在 batch=128（已经计算高效）时：投机带来的额外 Token 对吞吐几乎没有帮助，因为 GPU 已接近饱和。开销甚至可能<em>降低</em>吞吐。</p><p>规则：投机解码针对延迟（小 batch）；批处理针对吞吐（大 batch）。不要把两者结合。</p><p><em>复习：第 2 章（LLM 的系统基础）。</em></p>
</div>
</details>

## Agent 强化学习问题

<details class="hh-quiz" id="ch28-s18-p1">
<summary class="hh-quiz-question">Q：为什么标准 RLHF（单轮 PPO/DPO）对多步智能体（Agent）失效？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>标准 RLHF 优化单轮质量：给定一个 Prompt，产出一个好的回答。多步智能体面临根本不同的挑战：</p><ol><li>信用分配：在 50 步的轨迹中，是哪一步导致了失败？单轮奖励把信用均匀分配给整个回答；多步需要<em>逐步</em>的信用。</li><li>稀疏奖励：只有轨迹结束时才有成功/失败信号。PPO 的广义优势估计（GAE）假设存在中间奖励；没有它们，优势估计会很嘈杂。</li><li>动作空间：动作是结构化的 tool 调用（JSON），而不只是 Token 序列。模型必须同时学习语法、语义与策略。</li><li>非平稳性：环境随每个动作而改变（tool 输出会修改状态）。每一步都有不同的“Prompt”，不像单轮那样输入固定。</li><li>探索：智能体必须发现新颖的 tool 使用策略，而不仅仅是改写文本。</li></ol><p>解决方案：轨迹级 GRPO（对完整轨迹排序）、过程奖励模型（PRM）（逐步反馈），或对成功轨迹做过滤后的 SFT。</p><p><em>复习：第 12 章（LLM 智能体训练）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s18-p2">
<summary class="hh-quiz-question">Q：解释 GRPO 如何适配智能体训练。与单轮 GRPO 的关键区别是什么？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>单轮 GRPO：对一个 Prompt 生成 <math alttext="N" display="inline"><semantics><mi>N</mi></semantics></math> 个回答，按奖励排序，计算优势。</p><p>智能体 GRPO 的区别：</p><ol><li>生成单位：完整的<em>轨迹</em>（10–100 步），而非单个回答。每条轨迹是组内的一个“样本”。</li><li>奖励（Reward）：终止式（任务成功/失败）或轨迹级（各步奖励之和），而非逐 Token。</li><li>掩码（Masking）：只在智能体的输出（推理 + tool 调用）上计算策略损失。把 tool <em>输出</em>（环境响应）从梯度计算中屏蔽掉。</li><li>组大小：通常更小（<math alttext="N=4" display="inline"><semantics><mrow><mi>N</mi><mo>=</mo><mn>4</mn></mrow></semantics></math>–8），因为轨迹开销很大（每条轨迹需要多次前向传播）。</li><li>KL 惩罚：逐步施加，以防止在每个决策点偏离 SFT 策略。</li><li>长度归一化（Length normalization）：按智能体的 <em>动作</em>数量（而非 Token 数量）归一化，以免惩罚充分的推理。</li></ol><p><em>复习：第 7 与 12 章（GRPO；LLM 智能体训练）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s18-p3">
<summary class="hh-quiz-question">Q：比较用于智能体的 STaR、Reflexion 与 ReAct。</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>STaR（Self-Taught Reasoner，自学推理者）：生成推理链 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 按正确性过滤 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 在正确的链上微调。<em>适用场景</em>：你拥有可验证任务（数学、代码），并希望在不使用 RL 的情况下从基础模型自举出推理能力。</p><p>Reflexion：失败后生成言语化反馈（“哪里出错了？”）<math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 带着上下文中的反思重试。不更新权重。<em>适用场景</em>：推理时改进；可用于训练的算力有限；能够自我诊断的任务。</p><p>ReAct：在结构化循环中交替进行推理（思考）与行动（tool 使用）。<em>适用场景</em>：多步 tool 使用任务；你需要透明度（推理轨迹可解释）；智能体必须自行决定是思考还是行动。</p><p>关键差异：</p><p><em>复习：第 12 与 18 章（LLM 智能体训练；智能体设计模式）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s18-p4">
<summary class="hh-quiz-question">Q：为什么研究型智能体更偏好 GRPO 而非 PPO？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>对于轨迹长度为 20–100 步的研究型智能体：</p><p>PPO 需要价值模型：<math alttext="V(s_{t})" display="inline"><semantics><mrow><mi>V</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>s</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 必须从当前状态预测期望总奖励。对于研究任务（此处状态 = 128K Token 的上下文，包含论文、代码与结果），训练准确的价值函数极其困难——“读完 3 篇论文并写了部分代码”的价值很难预测。</p><p>GRPO 完全避开价值估计：它为每个研究问题生成 <math alttext="N" display="inline"><semantics><mi>N</mi></semantics></math> 条完整轨迹，并用组内排序作为优势。无需预测中间价值——只需比较结果。</p><p>其他原因：</p><ul><li>研究质量大致是二元的（报告好 vs. 报告差）——排序是自然的。</li><li>轨迹长且昂贵；GRPO 的 <math alttext="N=4" display="inline"><semantics><mrow><mi>N</mi><mo>=</mo><mn>4</mn></mrow></semantics></math> 尚可承受；PPO 则需要大量 rollout 才能得到稳定的价值估计。</li><li>终止奖励稀疏；在稀疏奖励下，GAE（广义优势估计）给出的逐步优势同样是嘈杂的。</li></ul><p><em>复习：第 7 与 12 章（GRPO；LLM 智能体训练）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s18-p5">
<summary class="hh-quiz-question">Q：为编码智能体设计一个奖励函数。存在哪些奖励作弊风险？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>奖励设计：</p><div class="hh-equation"><math alttext="R=0.5\cdot R_{\text{tests}}+0.2\cdot R_{\text{quality}}+0.2\cdot R_{\text{efficiency}}+0.1\cdot R_{\text{safety}}" display="block"><semantics><mrow><mi>R</mi><mo>=</mo><mrow><mrow><mn>0.5</mn><mo lspace="0.222em" rspace="0.222em">⋅</mo><msub><mi>R</mi><mtext>tests</mtext></msub></mrow><mo>+</mo><mrow><mn>0.2</mn><mo lspace="0.222em" rspace="0.222em">⋅</mo><msub><mi>R</mi><mtext>quality</mtext></msub></mrow><mo>+</mo><mrow><mn>0.2</mn><mo lspace="0.222em" rspace="0.222em">⋅</mo><msub><mi>R</mi><mtext>efficiency</mtext></msub></mrow><mo>+</mo><mrow><mn>0.1</mn><mo lspace="0.222em" rspace="0.222em">⋅</mo><msub><mi>R</mi><mtext>safety</mtext></msub></mrow></mrow></mrow></semantics></math></div><ul><li><math alttext="R_{\text{tests}}" display="inline"><semantics><msub><mi>R</mi><mtext>tests</mtext></msub></semantics></math>：单元测试通过比例（0–1）。真值可验证。</li><li><math alttext="R_{\text{quality}}" display="inline"><semantics><msub><mi>R</mi><mtext>quality</mtext></msub></semantics></math>：由 LLM 评判代码风格、文档与可维护性。</li><li><math alttext="R_{\text{efficiency}}" display="inline"><semantics><msub><mi>R</mi><mtext>efficiency</mtext></msub></semantics></math>：<math alttext="\max(0,1-\text{steps}/30)" display="inline"><semantics><mrow><mi>max</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mn>0</mn><mo>,</mo><mrow><mn>1</mn><mo>−</mo><mrow><mtext>steps</mtext><mo>/</mo><mn>30</mn></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow></semantics></math> —— 快速完成的奖励。</li><li><math alttext="R_{\text{safety}}" display="inline"><semantics><msub><mi>R</mi><mtext>safety</mtext></msub></semantics></math>：不允许危险操作（rm -rf、访问 sandbox 之外的网络）。</li></ul><p>奖励作弊风险：</p><ol><li>硬编码输出：智能体学会直接打印期望的测试输出，而不去真正计算。<em>修复</em>：随机化测试输入；在留出用例上测试。</li><li>删除测试：智能体修改或删除失败的测试。<em>修复</em>：把测试在 sandbox 中设为只读。</li><li>平凡解：智能体写出能通过测试但不泛化的最小代码。<em>修复</em>：大型、多样化的测试集；基于属性的测试。</li><li>效率作弊：智能体为最大化效率奖励而跳过推理步骤。<em>修复</em>：先满足最低质量阈值，效率奖励才生效。</li></ol><p><em>复习：第 9、12 与 19 章（Reward 模型训练；LLM 智能体训练；智能体环境）。</em></p>
</div>
</details>

## 列表式 Reward 与高级 RM 问题

<details class="hh-quiz" id="ch28-s19-p1">
<summary class="hh-quiz-question">Q：解释 Plackett-Luce 模型。它如何推广 Bradley-Terry？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>Bradley-Terry 对<em>成对</em>偏好建模：<math alttext="P(y_{1}\succ y_{2})=\sigma(r(y_{1})-r(y_{2}))" display="inline"><semantics><mrow><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><msub><mi>y</mi><mn>1</mn></msub><mo>≻</mo><msub><mi>y</mi><mn>2</mn></msub></mrow><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mi>σ</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mn>1</mn></msub><mo stretchy="false">)</mo></mrow></mrow><mo>−</mo><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mn>2</mn></msub><mo stretchy="false">)</mo></mrow></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math>。</p><p>Plackett-Luce 把 <math alttext="K" display="inline"><semantics><mi>K</mi></semantics></math> 个项目的<em>完整排序</em>建模为顺序选择：</p><div class="hh-equation"><math alttext="P(\pi)=\prod_{i=1}^{K}\frac{e^{r(y_{\pi(i)})}}{\sum_{j=i}^{K}e^{r(y_{\pi(j)})}}" display="block"><semantics><mrow><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>π</mi><mo stretchy="false">)</mo></mrow></mrow><mo rspace="0.111em">=</mo><mrow><munderover><mo movablelimits="false">∏</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>K</mi></munderover><mfrac><msup><mi>e</mi><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mrow><mi>π</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></mrow></msub><mo stretchy="false">)</mo></mrow></mrow></msup><mrow><msubsup><mo>∑</mo><mrow><mi>j</mi><mo>=</mo><mi>i</mi></mrow><mi>K</mi></msubsup><msup><mi>e</mi><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mrow><mi>π</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></mrow></msub><mo stretchy="false">)</mo></mrow></mrow></msup></mrow></mfrac></mrow></mrow></semantics></math></div><p>解释：依次挑选剩余项中最好的。位置 1 = 对全部 <math alttext="K" display="inline"><semantics><mi>K</mi></semantics></math> 做 Softmax；位置 2 = 对剩余的 <math alttext="K-1" display="inline"><semantics><mrow><mi>K</mi><mo>−</mo><mn>1</mn></mrow></semantics></math> 做 Softmax；以此类推。</p><p>推广关系：当 <math alttext="K=2" display="inline"><semantics><mrow><mi>K</mi><mo>=</mo><mn>2</mn></mrow></semantics></math> 时，PL 恰好退化为 BT：<math alttext="P(y_{1}\succ y_{2})=\frac{e^{r(y_{1})}}{e^{r(y_{1})}+e^{r(y_{2})}}=\sigma(r(y_{1})-r(y_{2}))" display="inline"><semantics><mrow><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><msub><mi>y</mi><mn>1</mn></msub><mo>≻</mo><msub><mi>y</mi><mn>2</mn></msub></mrow><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mfrac><msup><mi>e</mi><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mn>1</mn></msub><mo stretchy="false">)</mo></mrow></mrow></msup><mrow><msup><mi>e</mi><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mn>1</mn></msub><mo stretchy="false">)</mo></mrow></mrow></msup><mo>+</mo><msup><mi>e</mi><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mn>2</mn></msub><mo stretchy="false">)</mo></mrow></mrow></msup></mrow></mfrac><mo>=</mo><mrow><mi>σ</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mn>1</mn></msub><mo stretchy="false">)</mo></mrow></mrow><mo>−</mo><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mn>2</mn></msub><mo stretchy="false">)</mo></mrow></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math>。</p><p>优势：<math alttext="K=8" display="inline"><semantics><mrow><mi>K</mi><mo>=</mo><mn>8</mn></mrow></semantics></math> 个项目的排序提供 <math alttext="\binom{8}{2}=28" display="inline"><semantics><mrow><mrow><mo>(</mo><mfrac linethickness="0pt"><mn>8</mn><mn>2</mn></mfrac><mo>)</mo></mrow><mo>=</mo><mn>28</mn></mrow></semantics></math> 个隐式成对比较，外加相对差距信息——比单个成对样本丰富得多的训练信号。</p><p><em>复习：第 9 章（Reward 模型训练）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s19-p2">
<summary class="hh-quiz-question">Q：什么是过程奖励模型（Process Reward Model, PRM），它在什么情况下优于结果奖励模型（Outcome Reward Model, ORM）？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>ORM：仅对最终输出打分。<math alttext="r(x,y_{\text{final}})" display="inline"><semantics><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>,</mo><msub><mi>y</mi><mtext>final</mtext></msub><mo stretchy="false">)</mo></mrow></mrow></semantics></math> = 整个回答一个标量。</p><p>PRM：对推理的每一个<em>步骤</em>打分。<math alttext="r(x,y_{\text{step }t})" display="inline"><semantics><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>,</mo><msub><mi>y</mi><mrow><mtext>step </mtext><mo lspace="0em" rspace="0em">​</mo><mi>t</mi></mrow></msub><mo stretchy="false">)</mo></mrow></mrow></semantics></math> = 每一中间步一个标量。</p><p>PRM 更好的场景：</p><ol><li>长推理链：10+ 步的数学题。ORM 无法判断哪一步出错；PRM 提供逐步的信用分配。</li><li>搜索/验证：PRM 支持树搜索（对推理步做束搜索（beam search），剪掉步奖励低的分支）。</li><li>训练信号密度：PRM 每条轨迹给出 <math alttext="T" display="inline"><semantics><mi>T</mi></semantics></math> 个奖励（每步一个），而 ORM 只有一个奖励 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 更低方差的优势估计。</li></ol><p>ORM 更好的场景：任务短（单轮）；步骤边界不清晰；每步标注的成本过高。</p><p>PRM 标注：可通过“Math-Shepherd”方法自动化：对每一步，从该点多次补全整个解。如果从步骤 <math alttext="t" display="inline"><semantics><mi>t</mi></semantics></math> 出发的补全成功而从步骤 <math alttext="t+1" display="inline"><semantics><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></semantics></math> 出发的补全失败，那么步骤 <math alttext="t+1" display="inline"><semantics><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></semantics></math> 很可能是错的。</p><p><em>复习：第 9 与 13 章（Reward 模型训练；大型推理模型的 RL）。</em></p>
</div>
</details>

## 大型推理模型的 RL 问题

<details class="hh-quiz" id="ch28-s20-p1">
<summary class="hh-quiz-question">Q：DeepSeek-R1 在长推理链上训练，却为何不使用过程奖励模型？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>DeepSeek-R1 仅使用基于结果的奖励（准确率 + 格式），原因如下：</p><ol><li>可验证任务：数学和代码有确定的真值答案。即便是长链，二元的准确率奖励也能提供足够的信号。</li><li>PRM 失效模式：步骤级奖励模型自身会引入奖励作弊——模型可以学到产出“在 PRM 看来正确”但实际并不正确的步骤。</li><li>GRPO 的组归一化：通过为每个 Prompt 采样 <math alttext="G" display="inline"><semantics><mi>G</mi></semantics></math> 个补全并在组内归一化优势，GRPO 即使没有每步奖励，也能自然提供关于哪些推理 <em>策略</em> 有效的相对信号。</li><li>涌现的自我纠正：在仅有结果奖励时，模型学会在链内自我纠正（“顿悟时刻”），如果每步奖励对推理过程微观管理，这种行为将无法涌现。</li></ol><p>关键洞见：任务领域的可验证性使 PRM 变得不必要——对于主观任务（创意写作），仅有结果奖励可能不够。</p><p><em>复习：第 13 章（大型推理模型的 RL）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s20-p2">
<summary class="hh-quiz-question">Q：解释测试时计算扩展律（Test-Time Compute Scaling Law）及其对模型部署的启示</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>测试时计算扩展律表述为：</p><div class="hh-equation"><math alttext="\text{Accuracy}(C_{\text{train}},C_{\text{test}})\approx f(\alpha\log C_{\text{train}}+\beta\log C_{\text{test}})" display="block"><semantics><mrow><mrow><mtext>Accuracy</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>C</mi><mtext>train</mtext></msub><mo>,</mo><msub><mi>C</mi><mtext>test</mtext></msub><mo stretchy="false">)</mo></mrow></mrow><mo>≈</mo><mrow><mi>f</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mrow><mi>α</mi><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>C</mi><mtext>train</mtext></msub></mrow></mrow><mo>+</mo><mrow><mi>β</mi><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>C</mi><mtext>test</mtext></msub></mrow></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math></div><p>启示：</p><ol><li>算力等价：在推理任务上，使用 64<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> 推理 Token 的 7B 模型可以与使用 1<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> Token 的 70B 模型相匹敌。</li><li>自适应分配：简单问题用短链（便宜）；困难问题用长链（贵）。平均成本低于始终使用大模型。</li><li>部署灵活性：不再部署一个大模型，而是部署一个较小的推理模型，并按查询难度对推理算力进行扩展。</li><li>收益递减：对数关系意味着测试时计算翻倍带来的准确率提升递减——训练与推理算力之间存在一个最优分配。</li></ol><p>“过度思考”失败模式：非常长的链可能因错误累积和 Attention 稀释而<em>降低</em>准确率。最优链长取决于问题难度。</p><p><em>复习：第 13 章（大型推理模型的 RL）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s20-p3">
<summary class="hh-quiz-question">Q：MCTS（蒙特卡洛树搜索，Monte Carlo Tree Search）如何应用于 LLM 推理？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>用于推理的 MCTS 把每个部分解视为一个树节点：</p><p>每次迭代四个阶段：</p><ol><li>选择：用 UCB 从根开始导航：<math alttext="\text{UCB}(s)=Q(s)+c\sqrt{\frac{\ln N(\text{parent})}{N(s)}}" display="inline"><semantics><mrow><mrow><mtext>UCB</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>s</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mrow><mi>Q</mi><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>s</mi><mo stretchy="false">)</mo></mrow></mrow><mo>+</mo><mrow><mi>c</mi><mo lspace="0em" rspace="0em">​</mo><msqrt><mfrac><mrow><mi>ln</mi><mo lspace="0.167em">⁡</mo><mrow><mi>N</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mtext>parent</mtext><mo stretchy="false">)</mo></mrow></mrow></mrow><mrow><mi>N</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>s</mi><mo stretchy="false">)</mo></mrow></mrow></mfrac></msqrt></mrow></mrow></mrow></semantics></math></li><li>扩展：从 LLM 生成新的推理步（子节点）</li><li>模拟：从新节点完成解（rollout）</li><li>反向传播：根据最终正确性更新路径上的 Q 值</li></ol><p>与博弈 MCTS 的关键差异：</p><ul><li>分支因子：推理具有巨大的分支因子（任何下一句都可能）。实际实现使用 LLM 的 top-k 输出来限制分支。</li><li>价值函数：训练好的 PRM 估计部分解的质量，替代随机 rollout。</li><li>步骤粒度：每个“步”可能是一句话、一个方程或一次逻辑推断——粒度选择很关键。</li></ul><p>使用案例：AlphaProof（数学奥林匹克），以及据推测 OpenAI o1/o3 的隐藏推理。</p><p><em>复习：第 13 章（大型推理模型的 RL）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s20-p4">
<summary class="hh-quiz-question">Q：比较通过蒸馏 vs 直接 RL 来打造小型推理模型</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>蒸馏（DeepSeek-R1-Distill 路线）：</p><ul><li>从大模型（R1-671B）生成推理链</li><li>在这些链上对小模型做 SFT</li><li>结果：小模型模仿大模型的推理 <em>格式</em></li><li>优点：便宜（仅 SFT）。缺点：可能学到表层模式而非真正推理。</li></ul><p>直接对小模型做 RL：</p><ul><li>用 GRPO/PPO 在可验证奖励上训练小模型</li><li>模型发现自己的推理策略</li><li>优点：真实能力。缺点：算力消耗大得多；对于非常小的模型可能不收敛。</li></ul><p>经验发现：R1-Distill-7B（蒸馏）在大多数基准上优于 direct-RL-7B。大模型给出的推理链提供了如此强的监督，以至于仅靠 SFT 就极具竞争力。不过，蒸馏模型对真正新颖的问题类型的泛化能力较弱。</p><p>最佳实践：先做蒸馏（便宜的基础方案），再视情况在蒸馏模型上跑 RL 以获取进一步提升（Qwen 采用的“distill + RL”组合）。</p><p><em>复习：第 13 章（大型推理模型的 RL）。</em></p>
</div>
</details>

## LLM 评估问题

<details class="hh-quiz" id="ch28-s21-p1">
<summary class="hh-quiz-question">Q：推导 ELO 评分更新规则，并解释 Chatbot Arena 为何使用它</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>ELO 推导：</p><p>选手 A 对 B 的期望得分：<math alttext="E_{A}=\frac{1}{1+10^{(R_{B}-R_{A})/400}}" display="inline"><semantics><mrow><msub><mi>E</mi><mi>A</mi></msub><mo>=</mo><mfrac><mn>1</mn><mrow><mn>1</mn><mo>+</mo><msup><mn>10</mn><mrow><mrow><mo stretchy="false">(</mo><mrow><msub><mi>R</mi><mi>B</mi></msub><mo>−</mo><msub><mi>R</mi><mi>A</mi></msub></mrow><mo stretchy="false">)</mo></mrow><mo>/</mo><mn>400</mn></mrow></msup></mrow></mfrac></mrow></semantics></math>（logistic 模型）。</p><p>一场实际得分为 <math alttext="S_{A}\in\{0,0.5,1\}" display="inline"><semantics><mrow><msub><mi>S</mi><mi>A</mi></msub><mo>∈</mo><mrow><mo stretchy="false">{</mo><mrow><mn>0</mn><mo>,</mo><mn>0.5</mn><mo>,</mo><mn>1</mn></mrow><mo stretchy="false">}</mo></mrow></mrow></semantics></math> 的比赛之后：<math alttext="R_{A}^{\prime}=R_{A}+K(S_{A}-E_{A})" display="inline"><semantics><mrow><msubsup><mi>R</mi><mi>A</mi><mo>′</mo></msubsup><mo>=</mo><mrow><msub><mi>R</mi><mi>A</mi></msub><mo>+</mo><mrow><mi>K</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><msub><mi>S</mi><mi>A</mi></msub><mo>−</mo><msub><mi>E</mi><mi>A</mi></msub></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></semantics></math></p><p><math alttext="K" display="inline"><semantics><mi>K</mi></semantics></math> 因子控制更新幅度（<math alttext="K" display="inline"><semantics><mi>K</mi></semantics></math> 越大 = 对近期结果越敏感）。</p><p>为什么 Chatbot Arena 使用 ELO：</p><ol><li>传递性：如果模型 A 胜 B、B 胜 C，ELO 预测 A 胜 C。这从成对比较中得到一个全序。</li><li>在线更新：无需重新评估所有配对即可加入新模型。每次新的比较都会增量更新评分。</li><li>置信度：经过 <math alttext="N" display="inline"><semantics><mi>N</mi></semantics></math> 次比较后，评分不确定性以 <math alttext="O(1/\sqrt{N})" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mn>1</mn><mo>/</mo><msqrt><mi>N</mi></msqrt></mrow><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 的速率收缩。标准误：<math alttext="\text{SE}\approx\frac{400}{\sqrt{N}}" display="inline"><semantics><mrow><mtext>SE</mtext><mo>≈</mo><mfrac><mn>400</mn><msqrt><mi>N</mi></msqrt></mfrac></mrow></semantics></math>。</li><li>捕获人类偏好：真实用户无需阐明评判标准即可提供诚实的偏好。聚合起来即可揭示真实的模型质量。</li></ol><p>Chatbot Arena 的具体做法：使用 Bradley-Terry 极大似然估计（MLE，在收敛时等价于 ELO），并配以自举置信区间。风格受控的 ELO 去除了长度/格式偏置。</p><p><em>复习：第 14 章（LLM 评估）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s21-p2">
<summary class="hh-quiz-question">Q：代码生成中的 pass@k 指标是什么？为什么无偏估计量很重要？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>pass@k = <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 个生成样本中至少有一个通过全部测试用例的概率。</p><p>朴素（有偏）估计量：生成 <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 个样本，检查是否有通过的。问题：方差高、开销大（每个问题都需要多次试验）。</p><p>无偏估计量（Chen et al., 2021）：生成 <math alttext="n\geq k" display="inline"><semantics><mrow><mi>n</mi><mo>≥</mo><mi>k</mi></mrow></semantics></math> 个样本，统计其中通过的个数 <math alttext="c" display="inline"><semantics><mi>c</mi></semantics></math>：</p><div class="hh-equation"><math alttext="\text{pass@}k=1-\frac{\binom{n-c}{k}}{\binom{n}{k}}" display="block"><semantics><mrow><mrow><mtext>pass@</mtext><mo lspace="0em" rspace="0em">​</mo><mi>k</mi></mrow><mo>=</mo><mrow><mn>1</mn><mo>−</mo><mfrac><mrow><mo>(</mo><mfrac linethickness="0pt"><mrow><mi>n</mi><mo>−</mo><mi>c</mi></mrow><mi>k</mi></mfrac><mo>)</mo></mrow><mrow><mo>(</mo><mfrac linethickness="0pt"><mi>n</mi><mi>k</mi></mfrac><mo>)</mo></mrow></mfrac></mrow></mrow></semantics></math></div><p>为什么无偏很重要：</p><ol><li>只需生成 <math alttext="n" display="inline"><semantics><mi>n</mi></semantics></math> 个样本一次（例如 <math alttext="n=200" display="inline"><semantics><mrow><mi>n</mi><mo>=</mo><mn>200</mn></mrow></semantics></math>），即可从同一批样本计算 pass@1、pass@10、pass@100</li><li>无需把整个评估重复 <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 次</li><li>统计上精确（组合论证：不含正确样本的 <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 子集占比）</li><li>通过对数空间实现数值稳定的计算：<math alttext="\text{pass@}k=1-\exp\left(\sum_{i=0}^{k-1}\log(n-c-i)-\log(n-i)\right)" display="inline"><semantics><mrow><mrow><mtext>pass@</mtext><mo lspace="0em" rspace="0em">​</mo><mi>k</mi></mrow><mo>=</mo><mrow><mn>1</mn><mo>−</mo><mrow><mi>exp</mi><mo>⁡</mo><mrow><mo>(</mo><mrow><mrow><msubsup><mo lspace="0em">∑</mo><mrow><mi>i</mi><mo>=</mo><mn>0</mn></mrow><mrow><mi>k</mi><mo>−</mo><mn>1</mn></mrow></msubsup><mrow><mi>log</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mi>n</mi><mo>−</mo><mi>c</mi><mo>−</mo><mi>i</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow><mo>−</mo><mrow><mi>log</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mi>n</mi><mo>−</mo><mi>i</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow><mo>)</mo></mrow></mrow></mrow></mrow></semantics></math></li></ol><p>直觉：如果 200 个样本中有 50 个通过（<math alttext="c=50" display="inline"><semantics><mrow><mi>c</mi><mo>=</mo><mn>50</mn></mrow></semantics></math>，<math alttext="n=200" display="inline"><semantics><mrow><mi>n</mi><mo>=</mo><mn>200</mn></mrow></semantics></math>），则 pass@1 <math alttext="\approx 0.25" display="inline"><semantics><mrow><mphantom></mphantom><mo>≈</mo><mn>0.25</mn></mrow></semantics></math>，pass@10 <math alttext="\approx 0.94" display="inline"><semantics><mrow><mphantom></mphantom><mo>≈</mo><mn>0.94</mn></mrow></semantics></math>。该估计量统计的是：大小为 <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 的抽样中有多大比例会包含至少一个成功样本。</p><p><em>复习：第 14 与 19 章（LLM 评估；智能体环境）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s21-p3">
<summary class="hh-quiz-question">Q：如何检测并缓解基准污染？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>污染：训练数据中包含基准测试样本（或其近似改写），从而虚高分数。</p><p>检测方法：</p><ol><li>N-gram 重叠：检查训练数据中是否包含与测试样本完全相同或近乎相同的匹配。8-gram 重叠且覆盖率 <math alttext="&gt;" display="inline"><semantics><mo>&gt;</mo></semantics></math>80% 即可疑。</li><li>金丝雀串（Canary strings）：在测试集中插入唯一标识符，检查模型能否复现它们。</li><li>改写版基准：创建语义等价但文本不同的基准版本。准确率大幅下降说明存在记忆。</li><li>时间分析：模型在训练截止前与截止后测试样本上的表现。在旧样本上异常高的表现说明存在污染。</li><li>成员推断（Membership Inference）：用统计检验判断特定样本是否在训练数据中。</li></ol><p>缓解：</p><ul><li>动态基准：定期生成新的测试样本（LiveCodeBench、Chatbot Arena）</li><li>私有测试集：对测试样本保密（LMSYS）</li><li>训练期间去污染：从训练数据中移除检测到的重叠部分</li><li>报告污染分析：在基准分数之外一并披露重叠指标</li></ul><p><em>复习：第 14 章（LLM 评估）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s21-p4">
<summary class="hh-quiz-question">Q：解释 LLM-as-Judge 中的位置偏置以及如何缓解它</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>位置偏置：当用 LLM 评判两个回答（A vs B）时，模型会系统性地偏好处于特定位置（通常是第一个或最后一个）的回答，而不管其质量如何。</p><p>经验幅度：GPT-4 表现出 10–15% 的位置偏置；Claude 为 5–10%。更小的模型偏置更大。</p><p>缓解策略：</p><ol><li>位置交换：对每一对评判两次（A-B 与 B-A）。最终决定 = 多数。若不一致，则标记为“平局”。这消除了系统性的位置偏置，但成本翻倍。</li><li>多评委小组：使用 3 个以上不同的模型作为评委。多数投票可降低单个模型的偏置。</li><li>参考引导式：提供评分细则或参考回答。评委依据评分细则独立地为每个回答打分，再比较分数（完全消除成对比较）。</li><li>校准式 Prompt：加入明确指令：“呈现顺序是随机的，不应影响你的判断。”</li></ol><p>其他偏置：冗长偏置（偏好更长的回答）、自我增强偏置（模型偏好自己的输出）、权威偏置（顺从引用来源的回答）。</p><p><em>复习：第 14 章（LLM 评估）。</em></p>
</div>
</details>

## Agent 记忆问题

<details class="hh-quiz" id="ch28-s22-p1">
<summary class="hh-quiz-question">Q：比较四种智能体记忆类型，并说明各自何时关键</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>关键洞见：这四者并非彼此独立——它们相互作用。情景记忆为语义记忆提供输入（从具体经历泛化出事实）。程序性记忆由情景反馈不断精炼（学习哪些 tool 序列有效）。工作记忆则统筹对其它各类记忆的检索。</p><p>MemGPT 类比：工作记忆 = 热（在上下文中），情景/语义记忆 = 温（向量库），程序性记忆 = 冷（归档的策略）。智能体自行决定何时把信息换入/换出。</p><p><em>复习：第 16 章（智能体记忆系统）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s22-p2">
<summary class="hh-quiz-question">Q：记忆检索中的时间衰减如何工作，为什么它很重要？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>时间衰减在检索时降低较旧记忆的权重：</p><div class="hh-equation"><math alttext="\text{score}(m)=\alpha\cdot\text{similarity}(q,m)+(1-\alpha)\cdot\text{recency}(m)" display="block"><semantics><mrow><mrow><mtext>score</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>m</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mrow><mrow><mi>α</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><mtext>similarity</mtext></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>q</mi><mo>,</mo><mi>m</mi><mo stretchy="false">)</mo></mrow></mrow><mo>+</mo><mrow><mrow><mrow><mo stretchy="false">(</mo><mrow><mn>1</mn><mo>−</mo><mi>α</mi></mrow><mo rspace="0.055em" stretchy="false">)</mo></mrow><mo rspace="0.222em">⋅</mo><mtext>recency</mtext></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>m</mi><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></semantics></math></div><p>其中 <math alttext="\text{recency}(m)=e^{-\lambda\cdot\Delta t}" display="inline"><semantics><mrow><mtext>recency</mtext><mrow><mo stretchy="false">(</mo><mi>m</mi><mo stretchy="false">)</mo></mrow><mo>=</mo><msup><mi>e</mi><mrow><mo>−</mo><mi>λ</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><mi mathvariant="normal">Δ</mi><mi>t</mi></mrow></msup></mrow></semantics></math>，<math alttext="\Delta t" display="inline"><semantics><mrow><mi mathvariant="normal">Δ</mi><mo lspace="0em" rspace="0em">​</mo><mi>t</mi></mrow></semantics></math> = 距上次访问的时间。</p><p>为什么重要：</p><ol><li>相关性衰减：用户偏好会变化。6 个月前的偏好可能已经过时。</li><li>矛盾消解：当新旧信息冲突时，近因偏置会自然地偏向当前为真者。</li><li>检索效率：没有衰减，记忆会无界增长，检索也会返回越来越不相关的陈年条目。</li><li>认知合理性：人类也会遗忘——近期事件更易提取。这与间隔效应相呼应。</li></ol><p>基于访问的刷新：当一条记忆被检索并使用后，其时间戳会更新（类似 LRU 缓存）。频繁访问的记忆无论创建于何时都保持“新鲜”。</p><p>衰减率调优：<math alttext="\lambda" display="inline"><semantics><mi>λ</mi></semantics></math> 取决于领域。客服：高衰减（偏好变化快）。法律/医疗：低衰减（事实持久）。可通过 RL 学习。</p><p><em>复习：第 16 章（智能体记忆系统）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s22-p3">
<summary class="hh-quiz-question">Q：如何用 RL 训练记忆操作？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>记忆操作（write/read/update/delete）可以作为智能体的马尔可夫决策过程（MDP）中的动作：</p><p>形式化：</p><ul><li>状态：当前上下文 + 记忆状态</li><li>动作：标准动作 + <code>memory_write(key/value)</code>、<code>memory_read(query)</code>、<code>memory_delete(key)</code></li><li>奖励：任务成功（记忆是否起作用？）+ 记忆效率惩罚（读取越少越好）</li></ul><p>RL 学到什么：</p><ol><li>存储什么：重要信息（API key/用户偏好）vs 临时细节</li><li>何时检索：回答领域问题之前 vs 一般闲聊期间</li><li>压缩策略：何时摘要旧记忆，何时原样保留</li><li>遗忘：何时旧信息已过时、应被移除</li></ol><p>训练信号：反事实——“如果智能体没有存储/检索这条记忆，它还会成功吗？”通过轨迹比较来实现：记忆使用得当的轨迹获得更高的奖励。</p><p>挑战：延迟奖励——现在存储的信息可能要到 100 步之后才起作用。需要长时序的信用分配（高 <math alttext="\lambda" display="inline"><semantics><mi>λ</mi></semantics></math> 的 GAE）。</p><p><em>复习：第 12 与 16 章（LLM 智能体训练；智能体记忆系统）。</em></p>
</div>
</details>

## Agent 编排问题

<details class="hh-quiz" id="ch28-s23-p1">
<summary class="hh-quiz-question">Q：解释上下文预算问题，以及如何用动态分配来解决它</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>问题：智能体有 <math alttext="L" display="inline"><semantics><mi>L</mi></semantics></math> 个 Token 的上下文窗口，但需要为以下内容留出空间：</p><div class="hh-equation"><math alttext="C=S+M+T+H+R\leq L" display="block"><semantics><mrow><mi>C</mi><mo>=</mo><mrow><mi>S</mi><mo>+</mo><mi>M</mi><mo>+</mo><mi>T</mi><mo>+</mo><mi>H</mi><mo>+</mo><mi>R</mi></mrow><mo>≤</mo><mi>L</mi></mrow></semantics></math></div><p>其中 <math alttext="S" display="inline"><semantics><mi>S</mi></semantics></math> = 系统 Prompt，<math alttext="M" display="inline"><semantics><mi>M</mi></semantics></math> = 记忆/检索到的上下文，<math alttext="T" display="inline"><semantics><mi>T</mi></semantics></math> = tool 描述，<math alttext="H" display="inline"><semantics><mi>H</mi></semantics></math> = 对话历史，<math alttext="R" display="inline"><semantics><mi>R</mi></semantics></math> = 为回答预留。</p><p>随着对话增长，<math alttext="H" display="inline"><semantics><mi>H</mi></semantics></math> 不断增加，把其它组件挤出去。</p><p>动态分配策略：</p><ol><li>固定下限：<math alttext="S_{\min}" display="inline"><semantics><msub><mi>S</mi><mi>min</mi></msub></semantics></math>、<math alttext="R_{\min}" display="inline"><semantics><msub><mi>R</mi><mi>min</mi></msub></semantics></math> 是不可让步的</li><li>自适应历史：当 <math alttext="H&gt;H_{\max}" display="inline"><semantics><mrow><mi>H</mi><mo>&gt;</mo><msub><mi>H</mi><mi>max</mi></msub></mrow></semantics></math> 时摘要旧轮次。保留最近 <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 轮为原文，其余摘要处理。</li><li>按需提供 tool：只包含与当前查询相关的 tool 描述（而不是全部 50 个 tool）。用分类器或嵌入相似度选出前 <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 个 tool。</li><li>惰性记忆：只在需要时（分析完查询之后）检索记忆，而不是预先加载。</li></ol><p>溢出处理：当压缩之后总量仍超过 <math alttext="L" display="inline"><semantics><mi>L</mi></semantics></math> 时：</p><ul><li>丢弃最不重要的 tool 描述</li><li>更激进地把历史摘要为每轮一句话</li><li>减少记忆槽位</li><li>若仍然超出：截断并向用户给出警告</li></ul><p>飞行前检查：务必在调用 LLM 之前统计 Token 数。绝不要在推理时才发现溢出。</p><p><em>复习：第 17 章（Agent Harness —— 上下文管理与编排）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s23-p2">
<summary class="hh-quiz-question">Q：比较 ReAct 与 Plan-and-Execute 两种编排模式</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>ReAct（Reason + Act）：</p><ul><li>循环：思考 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 行动 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 观察 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 思考 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> ……</li><li>每一步都基于此前所有观察来决定下一个动作</li><li>优点：自适应——可以根据 tool 输出改变方向</li><li>缺点：短视——没有预先规划；可能陷入循环；每次 LLM 调用都要看到完整历史（昂贵）</li></ul><p>Plan-and-Execute：</p><ul><li>阶段 1：生成完整计划（步骤列表）</li><li>阶段 2：按序执行各步骤（执行器更简单；可使用更便宜的模型）</li><li>阶段 3：执行失败时重新规划</li><li>优点：高效（规划一次比每一步都推理更便宜）；独立的步骤可以并行</li><li>缺点：计划脆弱——早期步骤失败可能使整个计划失效。重新规划会增加延迟。</li></ul><p>何时用哪种：</p><ul><li>ReAct：探索性任务；未知环境；每一步的结果决定下一步的任务</li><li>Plan-and-Execute：定义明确的任务；已知的 tool 集合；可并行的子任务；对成本敏感的部署</li><li>混合模式：先做高层规划，再在每个计划步骤内用 ReAct（LangGraph 推荐的模式）</li></ul><p><em>复习：第 17 与 18 章（Agent Harness；智能体设计模式）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s23-p3">
<summary class="hh-quiz-question">Q：如何检测并防止智能体执行中的无限循环？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>当智能体重复同一动作却期待不同结果时，就会陷入无限循环。</p><p>检测方法：</p><ol><li>最大迭代次数防护：硬性上限（例如 25 步）。简单，但在真正耗时的长任务上会丢失已完成的工作。</li><li>动作哈希窗口：对最近 <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 个（动作/观察）对做哈希。若当前哈希与最近 <math alttext="w" display="inline"><semantics><mi>w</mi></semantics></math> 步中的某个哈希匹配，则判定为检测到循环。</li><li>语义相似度：对近期动作做嵌入。若相邻动作之间的余弦相似度超过阈值（<math alttext="&gt;" display="inline"><semantics><mo>&gt;</mo></semantics></math>0.95），则很可能卡住了。</li><li>进度监控：定义任务特定的进度指标。若 <math alttext="N" display="inline"><semantics><mi>N</mi></semantics></math> 步内没有进展，则进行干预。</li></ol><p>恢复策略：</p><ol><li>注入提示：添加系统消息：“你似乎在重复动作。试试换一种方法。”</li><li>强制不同动作：在下一步的动作空间中屏蔽掉被重复的动作。</li><li>上报：带着部分结果返回给用户并请求指导。</li><li>回溯：重置到循环开始之前的检查点，并尝试替代路径。</li></ol><p>最佳实践：把最大迭代次数（安全网）+ 基于哈希的检测（提前干预）+ 优雅上报（维护用户信任）组合起来。</p><p><em>复习：第 17 与 18 章（Agent Harness；智能体设计模式）。</em></p>
</div>
</details>

## MCP 协议问题

<details class="hh-quiz" id="ch28-s24-p1">
<summary class="hh-quiz-question">Q：解释 MCP 的 N+M 架构，以及它对智能体生态为何重要</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>N<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>M 问题：没有 MCP 时，<math alttext="N" display="inline"><semantics><mi>N</mi></semantics></math> 个智能体框架各自都要实现与 <math alttext="M" display="inline"><semantics><mi>M</mi></semantics></math> 个 tool 的集成 = 总共 <math alttext="N\times M" display="inline"><semantics><mrow><mi>N</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mi>M</mi></mrow></semantics></math> 个集成。新增一个 tool 需要 <math alttext="N" display="inline"><semantics><mi>N</mi></semantics></math> 个实现。</p><p>MCP 的 N+M 方案：标准化接口。每个 Agent 实现一个 MCP client（共 <math alttext="N" display="inline"><semantics><mi>N</mi></semantics></math> 个）。每个 tool 实现一个 MCP server（共 <math alttext="M" display="inline"><semantics><mi>M</mi></semantics></math> 个）。总集成数 = <math alttext="N+M" display="inline"><semantics><mrow><mi>N</mi><mo>+</mo><mi>M</mi></mrow></semantics></math>。</p><p>具体例子：5 个 Agent 框架（LangChain/AutoGen/CrewAI/Claude/自定义）<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> 20 个 tool（GitHub/Slack/DB/filesystem/…）= 不使用 MCP 时需要 100 个集成。使用 MCP 时：5 个 client + 20 个 server = 25 个实现。</p><p>为什么重要：</p><ol><li>Tool 复用：只需构建一次 tool server，任何兼容 MCP 的 Agent 都能使用</li><li>Agent 可移植性：从 Claude 切换到自定义 Agent 时无需重写 tool 集成</li><li>生态增长：添加新 tool 的门槛更低，激励社区构建更多 tool</li><li>可组合性：运行时可将多个 server 动态连接到一个 Agent</li></ol><p>类比：USB 标准化了外设连接。USB 之前：每个设备都有专有接口。USB 之后：一个端口适配所有设备。MCP 对 agent-tool 连接做了同样的事。</p><p><em>复习：第 20 章（Model Context Protocol）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s24-p2">
<summary class="hh-quiz-question">Q：MCP 的四个核心原语是什么，各自在什么情况下使用？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>关键区别：</p><ul><li>Tools 与 Resources：Tools 具有<em>副作用</em>（创建/修改/删除）。Resources 是<em>只读的</em>。这一区别对安全性很重要——Agent 可以自由读取 resources，但调用 tools 必须获得批准。</li><li>Sampling 反转了方向：通常由 client（Agent）调用 server（tool）。而在 Sampling 中，server 会请求 client 的 LLM 帮忙。用例：代码分析 server 需要 LLM 解释一段代码。</li><li>Prompts 是元数据（可复用模板），而非执行。它们帮助 Agent 构造更好的 tool 调用。</li></ul><p><em>复习：第 20 章（Model Context Protocol）。</em></p>
</div>
</details>

## Agent 通信（A2A）问题

<details class="hh-quiz" id="ch28-s25-p1">
<summary class="hh-quiz-question">Q：Google 的 A2A 协议与 MCP 有何不同，什么时候需要两者同时使用？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>核心区别：</p><ul><li>MCP：Agent <math alttext="\leftrightarrow" display="inline"><semantics><mo stretchy="false">↔</mo></semantics></math> Tool（具有既定 schema 的结构化函数调用）</li><li>A2A：Agent <math alttext="\leftrightarrow" display="inline"><semantics><mo stretchy="false">↔</mo></semantics></math> Agent（不透明的任务委派——你不知道对方 Agent 如何工作）</li></ul><p>A2A 关键概念：</p><ul><li>Agent Cards：描述 Agent 能力的 JSON（类似简历）。发现机制。</li><li>不透明执行：请求者看不到受托方的内部推理。只是发送任务并获取结果。</li><li>任务生命周期：submitted <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> working <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> completed/failed（通过 SSE 提供流式更新）</li></ul><p>何时两者都需要：</p><ol><li>编排 Agent 用 A2A 将“研究这个话题”委派给一个研究 Agent</li><li>研究 Agent 用 MCP 调用网页搜索、文件读取和数据库 tool</li><li>结果经 A2A 流回编排 Agent</li></ol><p>架构：A2A 位于<em>Agent 间</em>层；MCP 位于<em>Agent–tool</em>层。完整系统两者并用：A2A 用于 Agent 间协调，MCP 用于每个 Agent 的 tool 访问。</p><p><em>复习：第 20 章与第 22 章（MCP；Agent-to-Agent 通信）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s25-p2">
<summary class="hh-quiz-question">Q：什么是合同网协议（Contract Net Protocol），它如何应用于 LLM Agent？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>合同网协议（Contract Net Protocol，CNP）是来自分布式 AI 的任务分配机制：</p><p>步骤：</p><ol><li>宣告：管理者向所有可用 Agent 广播任务描述</li><li>投标：Agent 评估自身能力并提交投标（置信度；预估成本；预估时间）</li><li>授标：管理者按标准（能力/成本/可用性）选出最佳投标</li><li>执行：中标 Agent 执行任务</li><li>回报：Agent 将结果报告给管理者</li></ol><p>对 LLM Agent 而言：</p><ul><li>投标 = 自我评估：每个 Agent LLM 评估“我能把这个任务做好吗？”并给出置信度分数。这需要校准过的自我认知。</li><li>专业化涌现：代码 Agent 在代码任务上出高价；研究 Agent 在研究任务上出高价。无需中心路由逻辑。</li><li>负载均衡：若一个 Agent 忙（预估时间长），其他 Agent 胜出。</li><li>失败处理：若中标 Agent 失败，则向剩余 Agent 重新宣告（自动故障切换）。</li></ul><p>LLM 的局限：LLM 常常高估自身能力（幻觉置信度）。投标应纳入历史记录（相似任务的历史成功率），而不仅是自报置信度。</p><p><em>复习：第 22 章与第 23 章（A2A；多 Agent 系统）。</em></p>
</div>
</details>

## 多 Agent 系统问题

<details class="hh-quiz" id="ch28-s26-p1">
<summary class="hh-quiz-question">Q：比较 LLM 多 Agent 系统的中心化 vs 去中心化架构</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>中心化（Supervisor）：</p><ul><li>一个编排 LLM 将任务路由给专家 worker</li><li>控制流清晰；易于调试（检查 supervisor 决策）</li><li>单点故障；supervisor 成为 token 瓶颈</li><li>最适合：定义清晰的工作流；小型 Agent 团队（3–5 个 Agent）</li></ul><p>去中心化（点对点）：</p><ul><li>Agent 直接通信；无中央协调者</li><li>弹性（无单点故障）；可水平扩展</li><li>难以调试（涌现行为）；可能出现冲突和死锁</li><li>无结构时通信按 <math alttext="O(n^{2})" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msup><mi>n</mi><mn>2</mn></msup><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 扩展</li><li>最适合：弹性系统；大量 Agent 群体；期望涌现行为的创造性任务</li></ul><p>混合（层次化）：带子管理者的树形结构。结合优点：组内本地自治 + 顶层全局协调。通信按 <math alttext="O(n\log n)" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mi>n</mi><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mi>n</mi></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 扩展。</p><p>决策框架：需要可预测性和可审计性时用中心化。需要弹性和创造性时用去中心化。大型（<math alttext="&gt;" display="inline"><semantics><mo>&gt;</mo></semantics></math>10 个 Agent）系统用层次化。</p><p><em>复习：第 23 章（多 Agent 系统）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s26-p2">
<summary class="hh-quiz-question">Q：什么是 CTDE，为什么它对训练多 Agent LLM 系统重要？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>CTDE = 中心化训练；去中心化执行（Centralized Training; Decentralized Execution）。</p><p>问题：多 Agent RL 中，每个 Agent 的环境都是非平稳的（其他 Agent 同时在改变各自的 policy）。这使独立训练不稳定。</p><p>CTDE 方案：</p><ul><li>训练时：一个中心化 critic 可访问所有 Agent 的观察和动作：<math alttext="V(s_{1},s_{2},\ldots,s_{n},a_{1},a_{2},\ldots,a_{n})" display="inline"><semantics><mrow><mi>V</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>s</mi><mn>1</mn></msub><mo>,</mo><msub><mi>s</mi><mn>2</mn></msub><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msub><mi>s</mi><mi>n</mi></msub><mo>,</mo><msub><mi>a</mi><mn>1</mn></msub><mo>,</mo><msub><mi>a</mi><mn>2</mn></msub><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msub><mi>a</mi><mi>n</mi></msub><mo stretchy="false">)</mo></mrow></mrow></semantics></math>。这通过把非平稳性从 value 函数中剔除来稳定训练。</li><li>执行时：每个 Agent 仅基于自身观察行动：<math alttext="a_{i}=\pi_{i}(o_{i})" display="inline"><semantics><mrow><msub><mi>a</mi><mi>i</mi></msub><mo>=</mo><mrow><msub><mi>π</mi><mi>i</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>o</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math>。推理时没有通信开销。</li></ul><p>对 LLM Agent 而言：中心化 critic 可以是一个 reward 模型，评估所有 Agent 的<em>联合</em>输出（例如，Agent 团队是否产出了一个正确的软件系统？），而每个 Agent 通过反事实功劳分配被训练以最大化其对团队 reward 的贡献。</p><p>实际挑战：完整 CTDE 要求所有 Agent 在共享状态下同时训练——对 LLM 而言代价昂贵。近似方法：分轮训练 Agent（冻结其余 Agent 训练一个），或使用具有周期同步的种群训练。</p><p><em>复习：第 23 章（多 Agent 系统）。</em></p>
</div>
</details>

## Agent 开发框架问题

<details class="hh-quiz" id="ch28-s27-p1">
<summary class="hh-quiz-question">Q：比较 LangGraph、AutoGen、CrewAI 用于构建多 Agent 系统</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>选 LangGraph 当：需要细粒度控制；复杂条件流；带持久化和人机协同的生产部署。</p><p>选 AutoGen 当：多 Agent 对话的快速原型；代码执行 Agent；研究实验。</p><p>选 CrewAI 当：简单的基于角色的团队；顺序任务执行；快速演示；代码量最小。</p><p>都不选（自定义）当：需要最大性能/控制；不希望被框架锁定；或有非标准的编排模式。</p><p><em>复习：第 24 章（Agent 开发框架）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s27-p2">
<summary class="hh-quiz-question">Q：如何在生产中测试和评估一个 Agent 系统？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>Agent 测试遵循一个 测试金字塔：</p><p>第 1 层——单元测试（快；多）：</p><ul><li>独立测试单个 tool（mock LLM；验证 tool 逻辑）</li><li>测试 prompt 模板（给定上下文；验证正确的 prompt 构造）</li><li>测试解析器（给定 LLM 输出；验证正确提取）</li></ul><p>第 2 层——集成测试（中等速度）：</p><ul><li>用确定性输入测试完整的 Agent 循环</li><li>“黄金轨迹”测试：已知良好的执行轨迹，必须能复现</li><li>tool 链测试：验证多 tool 序列能端到端工作</li></ul><p>第 3 层——行为测试（慢；少）：</p><ul><li>Agent 是否遵守安全约束？（对抗性输入）</li><li>它是否在适当时候请求澄清？</li><li>它是否保持在 token/成本预算之内？</li></ul><p>生产评估：</p><ul><li>A/B 测试：将 5% 的流量路由到新 Agent 版本</li><li>影子模式：让新 Agent 与旧 Agent 并行运行，比较输出但不对外服务</li><li>LLM 作为评判者：自动为 Agent 回复打质量分</li><li>用户满意度：点赞/点踩；任务完成率；解决时长</li></ul><p>关键指标：任务成功率（Task Success Rate，TSR）——Agent 在无人干预下正确完成的任务比例。</p><p><em>复习：第 14 章与第 24 章（LLM 评估；Agent 开发框架）。</em></p>
</div>
</details>

## Agent 环境问题

<details class="hh-quiz" id="ch28-s28-p1">
<summary class="hh-quiz-question">Q：为一个网页浏览 Agent 环境设计 reward 函数</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>对于 WebArena 风格的任务（例如“找到 12 月 15 日从 NYC 到 SF 最便宜的航班”）：</p><p>稀疏 reward（简单但难以学习）：</p><div class="hh-equation"><math alttext="r=\begin{cases}1&amp;\text{if final page/state matches ground truth}\\
0&amp;\text{otherwise}\end{cases}" display="block"><semantics><mrow><mi>r</mi><mo>=</mo><mrow><mo>{</mo><mtable columnspacing="5pt" displaystyle="true" rowspacing="0pt"><mtr><mtd columnalign="left"><mn>1</mn></mtd><mtd columnalign="left"><mtext>if final page/state matches ground truth</mtext></mtd></mtr><mtr><mtd columnalign="left"><mn>0</mn></mtd><mtd columnalign="left"><mtext>otherwise</mtext></mtd></mtr></mtable></mrow></mrow></semantics></math></div><p>稠密 reward（更适合训练；更难设计）：</p><ol><li>进度奖励：<math alttext="+0.1" display="inline"><semantics><mrow><mo>+</mo><mn>0.1</mn></mrow></semantics></math> 每出现一个使 Agent 更接近目标的页面（通过对与目标状态的文本相似度衡量）</li><li>效率惩罚：<math alttext="-0.01" display="inline"><semantics><mrow><mo>−</mo><mn>0.01</mn></mrow></semantics></math> 每个动作（鼓励更短的轨迹）</li><li>里程碑奖励：<math alttext="+0.3" display="inline"><semantics><mrow><mo>+</mo><mn>0.3</mn></mrow></semantics></math> 达到中间目标（例如导航到航班搜索页面）</li><li>无效动作惩罚：<math alttext="-0.05" display="inline"><semantics><mrow><mo>−</mo><mn>0.05</mn></mrow></semantics></math> 产生错误的动作（404；表单校验失败）</li></ol><p>基于势函数的塑形（保持最优策略）：</p><div class="hh-equation"><math alttext="r_{\text{shaped}}(s,a,s^{\prime})=r(s,a,s^{\prime})+\gamma\Phi(s^{\prime})-\Phi(s)" display="block"><semantics><mrow><mrow><msub><mi>r</mi><mtext>shaped</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>s</mi><mo>,</mo><mi>a</mi><mo>,</mo><msup><mi>s</mi><mo>′</mo></msup><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mrow><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>s</mi><mo>,</mo><mi>a</mi><mo>,</mo><msup><mi>s</mi><mo>′</mo></msup><mo stretchy="false">)</mo></mrow></mrow><mo>+</mo><mrow><mi>γ</mi><mo lspace="0em" rspace="0em">​</mo><mi mathvariant="normal">Φ</mi><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msup><mi>s</mi><mo>′</mo></msup><mo stretchy="false">)</mo></mrow></mrow></mrow><mo>−</mo><mrow><mi mathvariant="normal">Φ</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>s</mi><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></semantics></math></div><p>其中 <math alttext="\Phi(s)=-\text{min\_steps\_to\_goal}(s)" display="inline"><semantics><mrow><mrow><mi mathvariant="normal">Φ</mi><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>s</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mo>−</mo><mrow><mtext>min_steps_to_goal</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>s</mi><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></semantics></math>（由启发式或学习到的 value 函数估计）。</p><p>挑战：部分可观测性（无法总能判断自己是否更接近目标）；随机环境（页面内容会变化）；奖励作弊（Agent 找到满足奖励但不符合用户意图的捷径）。</p><p><em>复习：第 12 章与第 19 章（LLM Agent 训练；Agent 环境）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s28-p2">
<summary class="hh-quiz-question">Q：是什么让 SWE-bench 成为一个特别有挑战性的 Agent 基准？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>SWE-bench 在来自流行 Python 仓库的真实 GitHub issue 上测试 Agent：</p><p>为什么难：</p><ol><li>仓库级上下文：Agent 必须理解 10 万行以上的代码库。它们放不进上下文窗口——必须探索、搜索和导航。</li><li>需求不明确的任务：issue 由人类撰写，带有隐含上下文。Agent 必须推断真正需要什么。</li><li>多文件编辑：解决方案往往跨越多个文件，且依赖关系层层级联。</li><li>测试验证：必须同时通过现有测试和验证该修复的新测试。</li><li>没有手把手引导：不同于 HumanEval（单个函数），SWE-bench 要求完整的软件工程工作流：阅读 issue <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 探索代码 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 定位 bug <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 实现修复 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 验证。</li></ol><p>最新水平（2024–2025）：最好的 Agent 能解决 SWE-bench Verified（精选子集）的 <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>50%。完整 SWE-bench：<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>30%。</p><p>训练的关键洞见：SWE-bench 暴露了“编码能力”（写出正确的函数）与“软件工程能力”（理解系统；在代码库中导航；做最小改动）之间的差距。在 SWE-bench 风格的环境上进行 RL 训练，教会 Agent 的不仅是代码生成，还有探索与规划策略。</p><p><em>复习：第 19 章（Agent 环境与基准）。</em></p>
</div>
</details>

## Agent UI 框架问题

<details class="hh-quiz" id="ch28-s29-p1">
<summary class="hh-quiz-question">Q：比较面向 Agent 的对话式与画布式 UI 范式</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>对话式（ChatGPT；Claude 默认）：</p><ul><li>线性消息流：user <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> assistant <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> user <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> …</li><li>优点：UX 熟悉；适合探索和问答；易于实现</li><li>缺点：生成产物（代码/文档）淹没在对话中。难以针对某个具体产物迭代。长对话中上下文会丢失。</li></ul><p>画布/产物式（Claude Artifacts；ChatGPT Canvas；Cursor）：</p><ul><li>侧边面板显示生成内容；对话面板用于指令</li><li>Agent 可以创建、编辑并迭代持久化产物</li><li>优点：产物独立于对话持久存在。用户可直接编辑。有版本历史。</li><li>缺点：UI 更复杂；需要检测产物类型；更难实现向两个面板同时流式输出。</li></ul><p>何时用哪种：</p><ul><li>对话式：头脑风暴；问答；快速任务；移动端界面</li><li>画布式：代码生成；文档撰写；数据分析——任何带有需要迭代的持久化输出的任务</li><li>混合（大多数现代 UI）：默认对话；检测到代码/文档/可视化输出时自动升级到画布</li></ul><p>对 Agent 训练而言：UI 范式会影响 reward 信号。画布式 UI 提供显式的编辑反馈（用户修改产物），可用于在线学习。</p><p><em>复习：第 25 章（Agent UI 框架）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s29-p2">
<summary class="hh-quiz-question">Q：如何为人机协同的 Agent 系统设计审批门（approval gate）？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>审批门在关键点暂停 Agent 执行，以供人类审查。</p><p>三层模型：</p><ol><li>自动批准（无门）：安全且可逆的动作。读操作；搜索；计算。</li><li>通知（软门）：可能产生影响但可恢复。发送邮件；创建草稿；修改文件。Agent 会继续执行，但用户会收到通知并可撤销。</li><li>阻止（硬门）：不可逆或高风险。删除数据；转账；发布内容；执行带副作用的代码。Agent 必须等待明确批准。</li></ol><p>设计原则：</p><ul><li>尽量减少打断：门太多 = 用户会放弃这个 Agent。三层模型让大多数动作顺畅执行，同时拦截危险动作。</li><li>展示上下文：在审批门显示：什么动作；为什么（Agent 的推理）；将发生什么变化；如何撤销。</li><li>批量批准：如果 Agent 需要 5 次文件写入，就把它们一起呈现，而不是逐个呈现。</li><li>超时处理：如果用户在 <math alttext="T" display="inline"><semantics><mi>T</mi></semantics></math> 分钟内未响应，则重试通知，或以安全默认值继续，或优雅中止。</li><li>从审批中学习：跟踪批准/拒绝的模式。如果用户总是批准某类动作，可以考虑将其自动升级。</li></ul><p>实现：tool 注解（MCP 的 <code>destructiveHint</code> 和 <code>readOnlyHint</code>）驱动自动门的分配。自定义规则可基于上下文覆盖。</p><p><em>复习：第 17 章与第 25 章（Agent Harness；Agent UI 框架）。</em></p>
</div>
</details>

## RAG 与 Agent 化 RAG 问题

<details class="hh-quiz" id="ch28-s30-p1">
<summary class="hh-quiz-question">Q：解释互逆排名融合（Reciprocal Rank Fusion，RRF），以及它为什么适用于混合检索</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>RRF 结合多个检索系统的排名，无需分数校准：</p><div class="hh-equation"><math alttext="\text{RRF}(d)=\sum_{r\in R}\frac{1}{k+r(d)}" display="block"><semantics><mrow><mrow><mtext>RRF</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>d</mi><mo stretchy="false">)</mo></mrow></mrow><mo rspace="0.111em">=</mo><mrow><munder><mo movablelimits="false">∑</mo><mrow><mi>r</mi><mo>∈</mo><mi>R</mi></mrow></munder><mfrac><mn>1</mn><mrow><mi>k</mi><mo>+</mo><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>d</mi><mo stretchy="false">)</mo></mrow></mrow></mrow></mfrac></mrow></mrow></semantics></math></div><p>其中 <math alttext="r(d)" display="inline"><semantics><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>d</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 是文档 <math alttext="d" display="inline"><semantics><mi>d</mi></semantics></math> 在检索器 <math alttext="r" display="inline"><semantics><mi>r</mi></semantics></math> 中的排名，<math alttext="k=60" display="inline"><semantics><mrow><mi>k</mi><mo>=</mo><mn>60</mn></mrow></semantics></math> 是一个常数，用于防止高排名文档占据主导。</p><p>为什么有效：</p><ol><li>无需分数归一化：BM25 分数无上界；稠密相似度位于 <math alttext="[-1,1]" display="inline"><semantics><mrow><mo stretchy="false">[</mo><mrow><mrow><mo>−</mo><mn>1</mn></mrow><mo>,</mo><mn>1</mn></mrow><mo stretchy="false">]</mo></mrow></semantics></math> 区间。RRF 只使用排名，使它们可直接比较。</li><li>对异常值稳健：某个检索器给出异常高的分数也不会占据主导，因为 <math alttext="1/(k+1)\approx 0.016" display="inline"><semantics><mrow><mrow><mn>1</mn><mo>/</mo><mrow><mo stretchy="false">(</mo><mrow><mi>k</mi><mo>+</mo><mn>1</mn></mrow><mo stretchy="false">)</mo></mrow></mrow><mo>≈</mo><mn>0.016</mn></mrow></semantics></math> 即使对排名第 1 的文档也是如此。</li><li>信号互补：BM25 捕捉精确关键词匹配；稠密检索捕捉语义相似性。被两者都排在前列的文档会获得提升。</li></ol><p>示例：文档 <math alttext="d" display="inline"><semantics><mi>d</mi></semantics></math> 在 BM25 中排名第 3，在稠密检索中排名第 7。RRF 分数为 <math alttext="=1/(60+3)+1/(60+7)=0.0159+0.0149=0.0308" display="inline"><semantics><mrow><mphantom></mphantom><mo>=</mo><mrow><mrow><mn>1</mn><mo>/</mo><mrow><mo stretchy="false">(</mo><mrow><mn>60</mn><mo>+</mo><mn>3</mn></mrow><mo stretchy="false">)</mo></mrow></mrow><mo>+</mo><mrow><mn>1</mn><mo>/</mo><mrow><mo stretchy="false">(</mo><mrow><mn>60</mn><mo>+</mo><mn>7</mn></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow><mo>=</mo><mrow><mn>0.0159</mn><mo>+</mo><mn>0.0149</mn></mrow><mo>=</mo><mn>0.0308</mn></mrow></semantics></math>。一个在其中一个检索器中排名第 1、在另一个中排名第 100 的文档得到 <math alttext="1/61+1/160=0.0226" display="inline"><semantics><mrow><mrow><mrow><mn>1</mn><mo>/</mo><mn>61</mn></mrow><mo>+</mo><mrow><mn>1</mn><mo>/</mo><mn>160</mn></mrow></mrow><mo>=</mo><mn>0.0226</mn></mrow></semantics></math>——尽管有 top-1 排名，得分却更低。</p><p>实践中：混合检索（BM25 + 稠密 + RRF）在 85% 以上的基准上都优于单独使用其中任一种。</p><p><em>复习：第 15 章（检索增强生成）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s30-p2">
<summary class="hh-quiz-question">Q：什么是 Agentic RAG，它与标准 RAG 有何不同？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>标准 RAG 遵循固定流水线：query <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> retrieve <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> generate。它没有能力：</p><ul><li>决定是否需要检索</li><li>评估检索到的文档是否充分</li><li>在检索失败时重新构造查询</li><li>合并多个检索步骤的信息</li></ul><p>Agentic RAG 将检索视为 Agent MDP 中的一个<em>动作</em>：</p><ul><li>是否检索的决策：Agent 评估自己是否已经知道答案（对训练数据中的事实性问题跳过检索）</li><li>查询规划：将复杂问题分解为子查询（“X 发生在哪一年？”+“当时谁当总统？”）</li><li>自我评估：检索后评估相关性。若不充分，则重构查询或尝试不同来源。</li><li>多跳推理：检索 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 推理 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 识别知识缺口 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 再次检索</li><li>来源路由：将查询路由到合适的知识库（时事用网页；公司信息用内部文档；编程用代码搜索）</li></ul><p>关键架构差异：标准 RAG = 确定性流水线。Agentic RAG = 带条件转移的状态机（LangGraph 模式，含 retrieve/grade/rewrite/generate 节点）。</p><p>权衡：Agentic RAG 在复杂查询上更准确，但会增加延迟（路由/评分需要多次 LLM 调用）。简单的事实性查找用标准 RAG；多跳或含糊的查询用 Agentic RAG。</p><p><em>复习：第 15 章与第 17 章（RAG；Agent Harness）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s30-p3">
<summary class="hh-quiz-question">Q：比较 Self-RAG 与 CRAG 这两种提升检索质量的方法</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>Self-RAG（Asai 等，2023）：</p><ul><li>训练特殊的 <em>reflection token</em>，并将其加入 LLM 词表</li><li>推理时，模型会输出 [Retrieve]、[IsRel]、[IsSup]、[IsUse] 之类的 token</li><li>模型决定<em>何时</em>检索（并非每个查询都需要检索）</li><li>检索后，模型自我评分：检索到的段落是否相关？我的答案是否由它推导而来？</li><li>训练：在用 GPT-4 生成的 reflection 标签增强的数据上做 SFT</li><li>优点：单个模型处理一切。缺点：需要自定义训练。</li></ul><p>CRAG（Corrective RAG，Yan 等，2024）：</p><ul><li>使用一个轻量的 <em>检索评估器</em>（独立模型）为检索到的文档打分</li><li>基于置信度的三种动作：<code>Correct</code>（按原样使用）、<code>Ambiguous</code>（用网页搜索补充）、<code>Incorrect</code>（丢弃；回退到网页）</li><li>增加一个 <em>知识精炼</em>步骤：只从检索到的文档中提取相关句子</li><li>优点：可与任何冻结的 LLM 配合使用。缺点：多一个用于评估的模型；增加延迟。</li></ul><p>关键区别：Self-RAG 将检索决策内嵌到 LLM 本身（需要训练）。CRAG 是一种流水线方法，可包装任意 LLM（无需训练）。Self-RAG 更优雅；CRAG 对使用现有模型的生产环境更实用。</p><p><em>复习：第 15 章（检索增强生成）。</em></p>
</div>
</details>

<details class="hh-quiz" id="ch28-s30-p4">
<summary class="hh-quiz-question">Q：什么是“迷失在中间”问题，如何缓解？</summary>
<div class="hh-quiz-answer">
<p class="hh-quiz-answer-label">解答</p>
<p>问题：当检索到的上下文很长（段落很多）时，LLM 会不成比例地关注上下文<em>开头</em>和<em>末尾</em>的信息，而忽略中间的信息。如果答案位于 10 个段落中的第 5 个，模型可能会漏掉它。</p><p>经验证据：Liu 等（2023）表明，在检索 20 个文档时，当相关文档位于第 5–15 位而不是第 1–3 位时，准确率会下降 15–20%。</p><p>缓解策略：</p><ol><li>重排并截断：使用交叉编码器（cross-encoder）重排，然后只保留最相关的前 3 个段落（更少 = 更少的迷失在中间）。</li><li>策略性排序：把最相关的段落放在上下文的开头和末尾，低相关的放在中间。</li><li>上下文压缩：插入前将每个段落摘要为 1–2 句。文本更少 = 位置偏置更少。</li><li>Map-reduce：独立处理每个段落（map），再合并答案（reduce）。完全消除位置效应。</li><li>引用提示：要求模型引用它使用了哪个段落。这会迫使它关注所有段落。</li><li>减小分块大小：更小的分块意味着覆盖答案所需的总分块数更少。</li></ol><p>最佳实践：先检索很多（20+），重排到前 3–5，再按相关性排序（最好的在前）。对大多数用例而言，这完全绕开了该问题。</p><p><em>复习：第 15 章（检索增强生成）。</em></p>
</div>
</details>

<aside class="hh-next">
<p class="hh-next-label">下一篇</p>
<p class="hh-next-title">第 29 章 速查手册</p>
<p class="hh-next-hint">在「智能体 AI 漫游指南」分类下继续阅读。</p>
</aside>

</div>
