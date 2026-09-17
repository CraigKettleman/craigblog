---
title: "智能体 AI 漫游指南 · 第 5 章 PPO——近端策略优化"
slug: "hitchhiker-agentic-ai-05-ppo-proximal-policy-optimization"
lang: "zh"
date: "2026-09-16T00:06:00.000Z"
draft: false
featured: false
categories:
  - "hitchhiker-agentic-ai"
cover: "./cover.png"
description: "问题：朴素的策略梯度更新对步长没有任何约束。仅仅一次倒霉的 batch 就可能把策略推入一个生成垃圾文本的区域 → 垃圾文本获得低奖励 → 下一次梯度让情况更糟 → 不可挽回地崩溃。"
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

## 动机与历史

问题：朴素的策略梯度更新对步长没有任何约束。仅仅一次倒霉的 batch 就可能把策略推入一个生成垃圾文本的区域 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 垃圾文本获得低奖励 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 下一次梯度让情况更糟 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 不可挽回地崩溃。

解决方案的演进：

<ol><li>TRPO[317]（2015）：对新旧策略之间的 KL 散度施加约束。效果完美，但需要昂贵的二阶优化（Fisher 信息矩阵、共轭梯度）。</li><li>PPO（2017）[319]：用一个简单的一阶 clipped 目标实现类似的稳定性。实现复杂度低了 10<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>，效果几乎一样，且能轻松扩展到分布式训练。</li></ol>

## Clipped 目标

PPO 的核心创新是一个 clipped surrogate 目标：它能阻止破坏性的大幅策略更新，同时保持实现简单。

<div class="hh-equation" id="ch5.e1"><math alttext="\boxed{L^{\text{CLIP}}(\theta)=\mathbb{E}_{t}\left[\min\left(r_{t}(\theta)\hat{A}_{t},\;\text{clip}(r_{t}(\theta),1{-}\epsilon,1{+}\epsilon)\hat{A}_{t}\right)\right]}" display="block"><semantics><menclose notation="box"><mrow><mrow><msup><mi>L</mi><mtext>CLIP</mtext></msup><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><msub><mi>𝔼</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo>[</mo><mrow><mi>min</mi><mo>⁡</mo><mrow><mo>(</mo><mrow><msub><mi>r</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow><mo lspace="0em" rspace="0em">​</mo><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>t</mi></msub></mrow><mo rspace="0.447em">,</mo><mrow><mtext>clip</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mrow><msub><mi>r</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>,</mo><mrow><mn>1</mn><mo>−</mo><mi>ϵ</mi></mrow><mo>,</mo><mrow><mn>1</mn><mo>+</mo><mi>ϵ</mi></mrow><mo stretchy="false">)</mo></mrow><mo lspace="0em" rspace="0em">​</mo><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>t</mi></msub></mrow><mo>)</mo></mrow></mrow><mo>]</mo></mrow></mrow></mrow></menclose></semantics></math><span class="hh-equation-number">(5.1)</span></div>

其中 <math alttext="r_{t}(\theta)=\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{\text{old}}}(a_{t}|s_{t})}" display="inline"><semantics><mrow><mrow><msub><mi>r</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mfrac><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>a</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">|</mo><msub><mi>s</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mi>π</mi><msub><mi>θ</mi><mtext>old</mtext></msub></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>a</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">|</mo><msub><mi>s</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow></mfrac></mrow></semantics></math> 是概率比。

## 完整的 PPO Loss

<div class="hh-equation" id="ch5.e2"><math alttext="L=L^{\text{CLIP}}-c_{1}\underbrace{(V_{\theta}(s_{t})-V^{\text{target}}_{t})^{2}}_{\text{value loss}}+c_{2}\underbrace{H[\pi_{\theta}(\cdot|s_{t})]}_{\text{entropy bonus}}" display="block"><semantics><mrow><mi>L</mi><mo>=</mo><mrow><mrow><msup><mi>L</mi><mtext>CLIP</mtext></msup><mo>−</mo><mrow><msub><mi>c</mi><mn>1</mn></msub><mo lspace="0em" rspace="0em">​</mo><munder><munder accentunder="true"><msup><mrow><mo stretchy="false">(</mo><mrow><mrow><msub><mi>V</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>s</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>−</mo><msubsup><mi>V</mi><mi>t</mi><mtext>target</mtext></msubsup></mrow><mo stretchy="false">)</mo></mrow><mn>2</mn></msup><mo stretchy="true">⏟</mo></munder><mtext>value loss</mtext></munder></mrow></mrow><mo>+</mo><mrow><msub><mi>c</mi><mn>2</mn></msub><mo lspace="0em" rspace="0em">​</mo><munder><munder accentunder="true"><mrow><mi>H</mi><mo stretchy="false">[</mo><mi>π</mi><msub><mrow></mrow><mi>θ</mi></msub><mo stretchy="false">(</mo><mo lspace="0em" rspace="0em">⋅</mo><mo fence="false" rspace="0.167em" stretchy="false">|</mo><mi>s</mi><msub><mrow></mrow><mi>t</mi></msub><mo stretchy="false">)</mo><mo stretchy="false">]</mo></mrow><mo stretchy="true">⏟</mo></munder><mtext>entropy bonus</mtext></munder></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(5.2)</span></div>

<ul><li>价值损失（<math alttext="c_{1}=0.1" display="inline"><semantics><mrow><msub><mi>c</mi><mn>1</mn></msub><mo>=</mo><mn>0.1</mn></mrow></semantics></math>）：训练评论家（critic）去预测回报。同样被 clip 以保持稳定。</li><li>熵奖励（<math alttext="c_{2}=0.01" display="inline"><semantics><mrow><msub><mi>c</mi><mn>2</mn></msub><mo>=</mo><mn>0.01</mn></mrow></semantics></math>）：防止过早收敛到确定性策略，对探索至关重要。</li></ul>

## PPO Gradient 与更新规则的推导

本节追溯从 RL 目标到 PPO 更新规则的数学路径，说明 clipped surrogate <em>为何</em>有效。

### 第 1 步：RL 目标

目标是在策略下最大化期望累计奖励：

<div class="hh-equation" id="ch5.e3"><math alttext="J(\theta)=\mathbb{E}_{\tau\sim\pi_{\theta}}\left[\sum_{t=0}^{T}r_{t}\right]" display="block"><semantics><mrow><mrow><mi>J</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><msub><mi>𝔼</mi><mrow><mi>τ</mi><mo>∼</mo><msub><mi>π</mi><mi>θ</mi></msub></mrow></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo>[</mo><mrow><munderover><mo lspace="0em" movablelimits="false">∑</mo><mrow><mi>t</mi><mo>=</mo><mn>0</mn></mrow><mi>T</mi></munderover><msub><mi>r</mi><mi>t</mi></msub></mrow><mo>]</mo></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(5.3)</span></div>

### 第 2 步：Policy Gradient 定理

<math alttext="J(\theta)" display="inline"><semantics><mrow><mi>J</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 关于策略参数的梯度：

<div class="hh-equation" id="ch5.e4"><math alttext="\boxed{\nabla_{\theta}J(\theta)=\mathbb{E}_{\pi_{\theta}}\left[\sum_{t=0}^{T}\nabla_{\theta}\log\pi_{\theta}(a_{t}|s_{t})\cdot\hat{A}_{t}\right]}" display="block"><semantics><menclose notation="box"><mrow><mrow><mrow><msub><mo>∇</mo><mi>θ</mi></msub><mi>J</mi></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><msub><mi>𝔼</mi><msub><mi>π</mi><mi>θ</mi></msub></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo>[</mo><mrow><mrow><munderover><mo lspace="0em" movablelimits="false">∑</mo><mrow><mi>t</mi><mo>=</mo><mn>0</mn></mrow><mi>T</mi></munderover><mrow><msub><mo>∇</mo><mi>θ</mi></msub><mo lspace="0.167em" rspace="0em">​</mo><mi>log</mi><mo lspace="0.167em" rspace="0em">​</mo><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>a</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">|</mo><msub><mi>s</mi><mi>t</mi></msub><mo rspace="0.055em" stretchy="false">)</mo></mrow></mrow></mrow><mo rspace="0.222em">⋅</mo><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>t</mi></msub></mrow><mo>]</mo></mrow></mrow></mrow></menclose></semantics></math><span class="hh-equation-number">(5.4)</span></div>

其中 <math alttext="\hat{A}_{t}" display="inline"><semantics><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>t</mi></msub></semantics></math> 是优势函数（即在状态 <math alttext="s_{t}" display="inline"><semantics><msub><mi>s</mi><mi>t</mi></msub></semantics></math> 下动作 <math alttext="a_{t}" display="inline"><semantics><msub><mi>a</mi><mi>t</mi></msub></semantics></math> 相对平均动作的好坏程度）。用优势替代完整回报是为了降低方差。

### 第 3 步：离策略数据的重要性采样

PPO 使用 <math alttext="\pi_{\theta_{\text{old}}}" display="inline"><semantics><msub><mi>π</mi><msub><mi>θ</mi><mtext>old</mtext></msub></msub></semantics></math> 收集数据，却更新 <math alttext="\pi_{\theta}" display="inline"><semantics><msub><mi>π</mi><mi>θ</mi></msub></semantics></math>。为修正这种分布失配，需应用重要性采样：

<div class="hh-equation" id="ch5.e5"><math alttext="\nabla_{\theta}J(\theta)=\mathbb{E}_{\pi_{\theta_{\text{old}}}}\left[\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{\text{old}}}(a_{t}|s_{t})}\nabla_{\theta}\log\pi_{\theta}(a_{t}|s_{t})\cdot\hat{A}_{t}\right]" display="block"><semantics><mrow><mrow><mrow><msub><mo>∇</mo><mi>θ</mi></msub><mi>J</mi></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><msub><mi>𝔼</mi><msub><mi>π</mi><msub><mi>θ</mi><mtext>old</mtext></msub></msub></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo>[</mo><mrow><mrow><mfrac><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>a</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">|</mo><msub><mi>s</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mi>π</mi><msub><mi>θ</mi><mtext>old</mtext></msub></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>a</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">|</mo><msub><mi>s</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow></mfrac><mo lspace="0.167em" rspace="0em">​</mo><msub><mo>∇</mo><mi>θ</mi></msub><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>π</mi><mi>θ</mi></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>a</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">|</mo><msub><mi>s</mi><mi>t</mi></msub><mo rspace="0.055em" stretchy="false">)</mo></mrow></mrow><mo rspace="0.222em">⋅</mo><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>t</mi></msub></mrow><mo>]</mo></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(5.5)</span></div>

定义概率比 <math alttext="r_{t}(\theta)=\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{\text{old}}}(a_{t}|s_{t})}" display="inline"><semantics><mrow><mrow><msub><mi>r</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mfrac><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>a</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">|</mo><msub><mi>s</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mi>π</mi><msub><mi>θ</mi><mtext>old</mtext></msub></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>a</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">|</mo><msub><mi>s</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow></mfrac></mrow></semantics></math>。利用恒等式 <math alttext="\nabla_{\theta}\log f=\frac{\nabla_{\theta}f}{f}" display="inline"><semantics><mrow><mrow><msub><mo>∇</mo><mi>θ</mi></msub><mo lspace="0.167em" rspace="0em">​</mo><mi>log</mi><mo lspace="0.167em" rspace="0em">​</mo><mi>f</mi></mrow><mo>=</mo><mfrac><mrow><msub><mo>∇</mo><mi>θ</mi></msub><mi>f</mi></mrow><mi>f</mi></mfrac></mrow></semantics></math>，可得：

<div class="hh-equation" id="ch5.e6"><math alttext="\nabla_{\theta}J(\theta)=\mathbb{E}_{\pi_{\theta_{\text{old}}}}\left[\nabla_{\theta}\,r_{t}(\theta)\cdot\hat{A}_{t}\right]" display="block"><semantics><mrow><mrow><mrow><msub><mo>∇</mo><mi>θ</mi></msub><mi>J</mi></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><msub><mi>𝔼</mi><msub><mi>π</mi><msub><mi>θ</mi><mtext>old</mtext></msub></msub></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo>[</mo><mrow><mrow><mrow><msub><mo rspace="0.167em">∇</mo><mi>θ</mi></msub><msub><mi>r</mi><mi>t</mi></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo rspace="0.055em" stretchy="false">)</mo></mrow></mrow><mo rspace="0.222em">⋅</mo><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>t</mi></msub></mrow><mo>]</mo></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(5.6)</span></div>

这意味着最大化 surrogate 目标：

<div class="hh-equation" id="ch5.e7"><math alttext="L^{\text{CPI}}(\theta)=\mathbb{E}_{t}\left[r_{t}(\theta)\cdot\hat{A}_{t}\right]" display="block"><semantics><mrow><mrow><msup><mi>L</mi><mtext>CPI</mtext></msup><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><msub><mi>𝔼</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo>[</mo><mrow><mrow><msub><mi>r</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo rspace="0.055em" stretchy="false">)</mo></mrow></mrow><mo rspace="0.222em">⋅</mo><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>t</mi></msub></mrow><mo>]</mo></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(5.7)</span></div>

### 第 4 步：无约束 Surrogate 的问题

<math alttext="L^{\text{CPI}}" display="inline"><semantics><msup><mi>L</mi><mtext>CPI</mtext></msup></semantics></math> 是一个合法的目标，但若无约束，单次梯度步就可能让 <math alttext="r_{t}(\theta)" display="inline"><semantics><mrow><msub><mi>r</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 远离 1.0，导致：

<ul><li>重要性权重变得极端 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 方差升高</li><li>策略进入未经检验的区域 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 奖励模型给出不可靠的分数</li><li>灾难性崩溃：策略生成垃圾，且无法恢复</li></ul>

TRPO 方案：约束 <math alttext="D_{\text{KL}}(\pi_{\theta_{\text{old}}}\|\pi_{\theta})\leq\delta" display="inline"><semantics><mrow><msub><mi>D</mi><mtext>KL</mtext></msub><mrow><mo stretchy="false">(</mo><msub><mi>π</mi><msub><mi>θ</mi><mtext>old</mtext></msub></msub><mo lspace="0em" rspace="0.167em">∥</mo><msub><mi>π</mi><mi>θ</mi></msub><mo stretchy="false">)</mo></mrow><mo>≤</mo><mi>δ</mi></mrow></semantics></math>。需要二阶方法（成本高）。

### 第 5 步：PPO 的 Clipped Surrogate（一阶近似）

PPO 用一个 clipped 目标取代硬性 KL 约束，仅使用一阶梯度即可达到类似行为：

<div class="hh-equation" id="ch5.e8"><math alttext="\boxed{L^{\text{CLIP}}(\theta)=\mathbb{E}_{t}\left[\min\!\left(r_{t}(\theta)\hat{A}_{t},\;\text{clip}(r_{t}(\theta),1{-}\epsilon,1{+}\epsilon)\hat{A}_{t}\right)\right]}" display="block"><semantics><menclose notation="box"><mrow><mrow><msup><mi>L</mi><mtext>CLIP</mtext></msup><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><msub><mi>𝔼</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo>[</mo><mrow><mpadded width="1.653em"><mi>min</mi></mpadded><mo>⁡</mo><mrow><mo>(</mo><mrow><msub><mi>r</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow><mo lspace="0em" rspace="0em">​</mo><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>t</mi></msub></mrow><mo rspace="0.447em">,</mo><mrow><mtext>clip</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mrow><msub><mi>r</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>,</mo><mrow><mn>1</mn><mo>−</mo><mi>ϵ</mi></mrow><mo>,</mo><mrow><mn>1</mn><mo>+</mo><mi>ϵ</mi></mrow><mo stretchy="false">)</mo></mrow><mo lspace="0em" rspace="0em">​</mo><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>t</mi></msub></mrow><mo>)</mo></mrow></mrow><mo>]</mo></mrow></mrow></mrow></menclose></semantics></math><span class="hh-equation-number">(5.8)</span></div>

梯度的推导：

令 <math alttext="L_{t}=\min(r_{t}\hat{A}_{t},\;\bar{r}_{t}\hat{A}_{t})" display="inline"><semantics><mrow><msub><mi>L</mi><mi>t</mi></msub><mo>=</mo><mrow><mi>min</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><msub><mi>r</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">​</mo><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>t</mi></msub></mrow><mo rspace="0.447em">,</mo><mrow><msub><mover accent="true"><mi>r</mi><mo>¯</mo></mover><mi>t</mi></msub><mo lspace="0em" rspace="0em">​</mo><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>t</mi></msub></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math>，其中 <math alttext="\bar{r}_{t}=\text{clip}(r_{t},1{-}\epsilon,1{+}\epsilon)" display="inline"><semantics><mrow><msub><mover accent="true"><mi>r</mi><mo>¯</mo></mover><mi>t</mi></msub><mo>=</mo><mrow><mtext>clip</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>r</mi><mi>t</mi></msub><mo>,</mo><mrow><mn>1</mn><mo>−</mo><mi>ϵ</mi></mrow><mo>,</mo><mrow><mn>1</mn><mo>+</mo><mi>ϵ</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math>。

<div class="hh-equation" id="ch5.e9"><math alttext="\nabla_{\theta}L_{t}=\begin{cases}\nabla_{\theta}r_{t}(\theta)\cdot\hat{A}_{t}&amp;\text{if }r_{t}\hat{A}_{t}&lt;\bar{r}_{t}\hat{A}_{t}\text{ (unclipped term is smaller)}\\
0&amp;\text{if }r_{t}\hat{A}_{t}\geq\bar{r}_{t}\hat{A}_{t}\text{ (clipped term is smaller, gradient = 0)}\end{cases}" display="block"><semantics><mrow><mrow><msub><mo>∇</mo><mi>θ</mi></msub><msub><mi>L</mi><mi>t</mi></msub></mrow><mo>=</mo><mrow><mo>{</mo><mtable columnspacing="5pt" displaystyle="true" rowspacing="0pt"><mtr><mtd columnalign="left"><mrow><mrow><mrow><msub><mo>∇</mo><mi>θ</mi></msub><msub><mi>r</mi><mi>t</mi></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo rspace="0.055em" stretchy="false">)</mo></mrow></mrow><mo rspace="0.222em">⋅</mo><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>t</mi></msub></mrow></mtd><mtd columnalign="left"><mrow><mrow><mtext>if </mtext><mo lspace="0em" rspace="0em">​</mo><msub><mi>r</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">​</mo><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>t</mi></msub></mrow><mo>&lt;</mo><mrow><msub><mover accent="true"><mi>r</mi><mo>¯</mo></mover><mi>t</mi></msub><mo lspace="0em" rspace="0em">​</mo><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>t</mi></msub><mo lspace="0em" rspace="0em">​</mo><mtext> (unclipped term is smaller)</mtext></mrow></mrow></mtd></mtr><mtr><mtd columnalign="left"><mn>0</mn></mtd><mtd columnalign="left"><mrow><mrow><mtext>if </mtext><mo lspace="0em" rspace="0em">​</mo><msub><mi>r</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">​</mo><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>t</mi></msub></mrow><mo>≥</mo><mrow><msub><mover accent="true"><mi>r</mi><mo>¯</mo></mover><mi>t</mi></msub><mo lspace="0em" rspace="0em">​</mo><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>t</mi></msub><mo lspace="0em" rspace="0em">​</mo><mtext> (clipped term is smaller, gradient = 0)</mtext></mrow></mrow></mtd></mtr></mtable></mrow></mrow></semantics></math><span class="hh-equation-number">(5.9)</span></div>

展开各条件：

<ul><li>当 <math alttext="\hat{A}_{t}&gt;0" display="inline"><semantics><mrow><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>t</mi></msub><mo>&gt;</mo><mn>0</mn></mrow></semantics></math> 且 <math alttext="r_{t}&lt;1+\epsilon" display="inline"><semantics><mrow><msub><mi>r</mi><mi>t</mi></msub><mo>&lt;</mo><mrow><mn>1</mn><mo>+</mo><mi>ϵ</mi></mrow></mrow></semantics></math>：梯度正常传播——策略被鼓励提升 <math alttext="\pi_{\theta}(a_{t}|s_{t})" display="inline"><semantics><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>a</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">|</mo><msub><mi>s</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow></semantics></math>。</li><li>当 <math alttext="\hat{A}_{t}&gt;0" display="inline"><semantics><mrow><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>t</mi></msub><mo>&gt;</mo><mn>0</mn></mrow></semantics></math> 且 <math alttext="r_{t}\geq 1+\epsilon" display="inline"><semantics><mrow><msub><mi>r</mi><mi>t</mi></msub><mo>≥</mo><mrow><mn>1</mn><mo>+</mo><mi>ϵ</mi></mrow></mrow></semantics></math>：梯度为 零——策略提升已足够，停止继续推。</li><li>当 <math alttext="\hat{A}_{t}&lt;0" display="inline"><semantics><mrow><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>t</mi></msub><mo>&lt;</mo><mn>0</mn></mrow></semantics></math> 且 <math alttext="r_{t}&gt;1-\epsilon" display="inline"><semantics><mrow><msub><mi>r</mi><mi>t</mi></msub><mo>&gt;</mo><mrow><mn>1</mn><mo>−</mo><mi>ϵ</mi></mrow></mrow></semantics></math>：梯度正常传播——策略被鼓励降低 <math alttext="\pi_{\theta}(a_{t}|s_{t})" display="inline"><semantics><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>a</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">|</mo><msub><mi>s</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow></semantics></math>。</li><li>当 <math alttext="\hat{A}_{t}&lt;0" display="inline"><semantics><mrow><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>t</mi></msub><mo>&lt;</mo><mn>0</mn></mrow></semantics></math> 且 <math alttext="r_{t}\leq 1-\epsilon" display="inline"><semantics><mrow><msub><mi>r</mi><mi>t</mi></msub><mo>≤</mo><mrow><mn>1</mn><mo>−</mo><mi>ϵ</mi></mrow></mrow></semantics></math>：梯度为 零——策略降低已足够，停止继续推。</li></ul>

### 第 6 步：完整的 PPO 更新规则

将 clipped 策略损失、价值损失与熵奖励组合起来：

<div class="hh-equation" id="ch5.e10"><math alttext="\boxed{\theta_{k+1}=\theta_{k}+\alpha\cdot\nabla_{\theta}\left[L^{\text{CLIP}}(\theta)-c_{1}L^{\text{VF}}(\theta)+c_{2}H[\pi_{\theta}]\right]}" display="block"><semantics><menclose notation="box"><mrow><msub><mi>θ</mi><mrow><mi>k</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>=</mo><mrow><msub><mi>θ</mi><mi>k</mi></msub><mo>+</mo><mrow><mi>α</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><mrow><msub><mo>∇</mo><mi>θ</mi></msub><mrow><mo>[</mo><mrow><mrow><mrow><msup><mi>L</mi><mtext>CLIP</mtext></msup><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>−</mo><mrow><msub><mi>c</mi><mn>1</mn></msub><mo lspace="0em" rspace="0em">​</mo><msup><mi>L</mi><mtext>VF</mtext></msup><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></mrow></mrow><mo>+</mo><mrow><msub><mi>c</mi><mn>2</mn></msub><mo lspace="0em" rspace="0em">​</mo><mi>H</mi><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">[</mo><msub><mi>π</mi><mi>θ</mi></msub><mo stretchy="false">]</mo></mrow></mrow></mrow><mo>]</mo></mrow></mrow></mrow></mrow></mrow></menclose></semantics></math><span class="hh-equation-number">(5.10)</span></div>

其中：



## Rollout Buffer 与 Rollout

在 PPO 中，数据管理依赖于一个专门的短期存储系统，称为 Rollout Buffer。与把经验无限期保存在回放缓冲区中的 off-policy 算法（DQN）不同，PPO 需要一个临时结构来满足其 on-policy 的数学约束。

### 什么是 Rollout？

一次 rollout（轨迹）是指智能体在环境中运行其当前策略所生成的一串交互：

<ul><li>该过程：智能体观察状态、选择动作、接收奖励，并转移到下一个状态。它会重复固定数量的步数，直至回合结束。</li><li>在 LLM/RLHF 中：一次 rollout 就是从一个数据集中取出一条 prompt，让语言模型逐 token 生成完整序列，直到命中文本结束标记。每个 token 就是一次「step」。</li></ul>

### Rollout Buffer

Rollout Buffer 临时保存 rollout 阶段收集到的全部数据。对每个生成的 token / 每一步，它记录：

<div class="hh-equation" id="ch5.e13"><math alttext="\boxed{\mathcal{B}=\left\{\left(s_{t},\;a_{t},\;\log\pi_{\theta_{\text{old}}}(a_{t}|s_{t}),\;r_{t},\;V(s_{t})\right)\right\}_{t=1}^{T}}" display="block"><semantics><menclose notation="box"><mrow><mi>ℬ</mi><mo>=</mo><msubsup><mrow><mo>{</mo><mrow><mo>(</mo><msub><mi>s</mi><mi>t</mi></msub><mo>,</mo><msub><mi>a</mi><mi>t</mi></msub><mo>,</mo><mrow><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>π</mi><msub><mi>θ</mi><mtext>old</mtext></msub></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>a</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">|</mo><msub><mi>s</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>,</mo><msub><mi>r</mi><mi>t</mi></msub><mo>,</mo><mrow><mi>V</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>s</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>)</mo></mrow><mo>}</mo></mrow><mrow><mi>t</mi><mo>=</mo><mn>1</mn></mrow><mi>T</mi></msubsup></mrow></menclose></semantics></math><span class="hh-equation-number">(5.13)</span></div>

<ul><li><math alttext="s_{t},a_{t},r_{t}" display="inline"><semantics><mrow><msub><mi>s</mi><mi>t</mi></msub><mo>,</mo><msub><mi>a</mi><mi>t</mi></msub><mo>,</mo><msub><mi>r</mi><mi>t</mi></msub></mrow></semantics></math>：第 <math alttext="t" display="inline"><semantics><mi>t</mi></semantics></math> 步的状态、所采取的动作与奖励。</li><li><math alttext="\log\pi_{\theta_{\text{old}}}(a_{t}|s_{t})" display="inline"><semantics><mrow><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>π</mi><msub><mi>θ</mi><mtext>old</mtext></msub></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>a</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">|</mo><msub><mi>s</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow></semantics></math>：在生成该动作的那份策略下采取该动作的对数概率（计算概率比时需要）。</li><li><math alttext="V(s_{t})" display="inline"><semantics><mrow><mi>V</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>s</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow></semantics></math>：价值函数给出的基线预测（计算 GAE 优势时需要）。</li></ul>

### Rollout Buffer 的生命周期

该缓冲区按照严格的三阶段周期循环运转：

<ol><li>收集：当前策略与环境交互，用新鲜轨迹填满缓冲区（对于 batch=128、max_tokens=512 的 70B 模型：每次 rollout 最多 65K 条 token 级转移）。</li><li>训练：跨轨迹计算 GAE 优势。用 clipped 目标做 <math alttext="K" display="inline"><semantics><mi>K</mi></semantics></math> 个 epoch（通常 3–10）的 mini-batch 梯度下降，以更新策略权重。</li><li>清空：整个缓冲区会被彻底清空。因为 PPO 是 on-policy 的，旧策略生成的数据无法安全地用于下一个更新周期——比率 <math alttext="r_{t}(\theta)" display="inline"><semantics><mrow><msub><mi>r</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 会变陈旧，clipping 的保证也会失效。</li></ol>

## 用于 RLHF 的 PPO：完整循环

<figure id="ch5.s6.fig1"><img src="./fig_025_fig25.png" alt="" loading="lazy" decoding="async"></figure>

## 细节机制：Logits 与 Policy 更新

<figure id="ch5.f1"><img src="./fig_026_fig26.png" alt="图 5.1：PPO 端到端流程：从 prompt batch 出发，经由生成、奖励打分、KL 计算、优势估计，直至 clipped 策略更新。反馈回路展示了更新后的策略被用于下一轮生成。" loading="lazy" decoding="async"><figcaption><span class="hh-tag">图 5.1：</span>PPO 端到端流程：从 prompt batch 出发，经由生成、奖励打分、KL 计算、优势估计，直至 clipped 策略更新。反馈回路展示了更新后的策略被用于下一轮生成。</figcaption></figure>

PPO 在内存中维护两个不同的参数状态，它们共享相同的神经网络拓扑，但在优化过程中持有不同的权重值：

### 阶段 1：Rollout（数据收集）

在数据收集期间，智能体与环境交互 <math alttext="T" display="inline"><semantics><mi>T</mi></semantics></math> 步。在每一时间步 <math alttext="t" display="inline"><semantics><mi>t</mi></semantics></math>：

<ol><li>环境给出当前状态/观测 <math alttext="s_{t}" display="inline"><semantics><msub><mi>s</mi><mi>t</mi></msub></semantics></math>（对 LLM 而言：prompt 加上目前已生成的 token）。</li><li>状态 <math alttext="s_{t}" display="inline"><semantics><msub><mi>s</mi><mi>t</mi></msub></semantics></math> 被送入当前网络快照（<math alttext="\theta_{\text{old}}" display="inline"><semantics><msub><mi>θ</mi><mtext>old</mtext></msub></semantics></math>）。</li><li>网络输出原始未归一化的值——logits<math alttext="z_{\text{old}}" display="inline"><semantics><msub><mi>z</mi><mtext>old</mtext></msub></semantics></math>——一个大小为 <math alttext="|V|" display="inline"><semantics><mrow><mo stretchy="false">|</mo><mi>V</mi><mo stretchy="false">|</mo></mrow></semantics></math> 的向量（词表规模 32K–128K）。</li><li>概率通过 Softmax 计算：<math alttext="\boxed{P(a\mid s_{t})=\text{Softmax}(z_{\text{old}})=\frac{\exp(z_{\text{old},a})}{\sum_{j=1}^{|V|}\exp(z_{\text{old},j})}}" display="block"><semantics><menclose notation="box"><mrow><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>a</mi><mo fence="true" lspace="0em" rspace="0em">∣</mo><msub><mi>s</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mtext>Softmax</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>z</mi><mtext>old</mtext></msub><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mfrac><mrow><mi>exp</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>z</mi><mrow><mtext>old</mtext><mo>,</mo><mi>a</mi></mrow></msub><mo stretchy="false">)</mo></mrow></mrow><mrow><msubsup><mo>∑</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mrow><mo stretchy="false">|</mo><mi>V</mi><mo stretchy="false">|</mo></mrow></msubsup><mrow><mi>exp</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>z</mi><mrow><mtext>old</mtext><mo>,</mo><mi>j</mi></mrow></msub><mo stretchy="false">)</mo></mrow></mrow></mrow></mfrac></mrow></menclose></semantics></math><span class="hh-tag">(5.14)</span></li><li>从 <math alttext="P(a\mid s_{t})" display="inline"><semantics><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>a</mi><mo fence="true" lspace="0em" rspace="0em">∣</mo><msub><mi>s</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 中采样得到动作 <math alttext="a_{t}" display="inline"><semantics><msub><mi>a</mi><mi>t</mi></msub></semantics></math>（下一个 token），并把转移元组 <math alttext="\langle s_{t},a_{t},r_{t},s_{t+1}\rangle" display="inline"><semantics><mrow><mo stretchy="false">⟨</mo><mrow><msub><mi>s</mi><mi>t</mi></msub><mo>,</mo><msub><mi>a</mi><mi>t</mi></msub><mo>,</mo><msub><mi>r</mi><mi>t</mi></msub><mo>,</mo><msub><mi>s</mi><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub></mrow><mo stretchy="false">⟩</mo></mrow></semantics></math> 连同 <math alttext="\log\pi_{\theta_{\text{old}}}(a_{t}\mid s_{t})" display="inline"><semantics><mrow><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>π</mi><msub><mi>θ</mi><mtext>old</mtext></msub></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>a</mi><mi>t</mi></msub><mo fence="true" lspace="0em" rspace="0em">∣</mo><msub><mi>s</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 一起存入 rollout buffer。</li></ol>

### 阶段 2：优化循环（Mini-Batch 更新）

一旦 rollout buffer 填满，PPO 就会在 mini-batch 上跑 <math alttext="K" display="inline"><semantics><mi>K</mi></semantics></math> 个 epoch（通常 3–10）。每次梯度步都会用存储的状态 <math alttext="s_{t}" display="inline"><semantics><msub><mi>s</mi><mi>t</mi></msub></semantics></math> 为两个策略生成 logits：

旧策略评估（冻结）：

<div class="hh-equation" id="ch5.e15"><math alttext="z_{\text{old}}=f(s_{t};\theta_{\text{old}})\quad\longrightarrow\quad\log\pi_{\theta_{\text{old}}}(a_{t}\mid s_{t})=\text{LogSoftmax}(z_{\text{old}})[a_{t}]" display="block"><semantics><mrow><mrow><msub><mi>z</mi><mtext>old</mtext></msub><mo>=</mo><mrow><mi>f</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>s</mi><mi>t</mi></msub><mo>,</mo><msub><mi>θ</mi><mtext>old</mtext></msub><mo stretchy="false">)</mo></mrow></mrow></mrow><mspace width="1em"></mspace><mo stretchy="false">⟶</mo><mspace width="1.167em"></mspace><mrow><mrow><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>π</mi><msub><mi>θ</mi><mtext>old</mtext></msub></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>a</mi><mi>t</mi></msub><mo fence="true" lspace="0em" rspace="0em">∣</mo><msub><mi>s</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mtext>LogSoftmax</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>z</mi><mtext>old</mtext></msub><mo stretchy="false">)</mo></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">[</mo><msub><mi>a</mi><mi>t</mi></msub><mo stretchy="false">]</mo></mrow></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(5.15)</span></div>

<em>实现捷径：直接复用 rollout 时存下的标量，而不重新计算。</em>

在线策略评估（更新中）：

<div class="hh-equation" id="ch5.e16"><math alttext="z_{\text{new}}=f(s_{t};\theta)\quad\longrightarrow\quad\log\pi_{\theta}(a_{t}\mid s_{t})=\text{LogSoftmax}(z_{\text{new}})[a_{t}]" display="block"><semantics><mrow><mrow><msub><mi>z</mi><mtext>new</mtext></msub><mo>=</mo><mrow><mi>f</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>s</mi><mi>t</mi></msub><mo>,</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></mrow></mrow><mspace width="1em"></mspace><mo stretchy="false">⟶</mo><mspace width="1.167em"></mspace><mrow><mrow><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>π</mi><mi>θ</mi></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>a</mi><mi>t</mi></msub><mo fence="true" lspace="0em" rspace="0em">∣</mo><msub><mi>s</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mtext>LogSoftmax</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>z</mi><mtext>new</mtext></msub><mo stretchy="false">)</mo></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">[</mo><msub><mi>a</mi><mi>t</mi></msub><mo stretchy="false">]</mo></mrow></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(5.16)</span></div>

由于 <math alttext="\theta" display="inline"><semantics><mi>θ</mi></semantics></math> 在每个 mini-batch 梯度步之后都会更新，<math alttext="z_{\text{new}}" display="inline"><semantics><msub><mi>z</mi><mtext>new</mtext></msub></semantics></math> 在整个优化循环中持续变化，而 <math alttext="z_{\text{old}}" display="inline"><semantics><msub><mi>z</mi><mtext>old</mtext></msub></semantics></math> 始终保持完全静止。

### 从 Logits 到概率比

PPO 的核心比率衡量的是：同一动作在新策略下比在旧策略下更可能还是更不可能：

<div class="hh-equation" id="ch5.e17"><math alttext="\boxed{r_{t}(\theta)=\frac{\pi_{\theta}(a_{t}\mid s_{t})}{\pi_{\theta_{\text{old}}}(a_{t}\mid s_{t})}}" display="block"><semantics><menclose notation="box"><mrow><mrow><msub><mi>r</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mfrac><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>a</mi><mi>t</mi></msub><mo fence="true" lspace="0em" rspace="0em">∣</mo><msub><mi>s</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mi>π</mi><msub><mi>θ</mi><mtext>old</mtext></msub></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>a</mi><mi>t</mi></msub><mo fence="true" lspace="0em" rspace="0em">∣</mo><msub><mi>s</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow></mfrac></mrow></menclose></semantics></math><span class="hh-equation-number">(5.17)</span></div>

为避免直接相除原始概率造成灾难性的数值下溢/上溢，计算在 对数空间中进行：



对比值取差值的指数即可还原出比率：

<div class="hh-equation" id="ch5.e20"><math alttext="\boxed{r_{t}(\theta)=\exp\!\left(\log\pi_{\theta}(a_{t}\mid s_{t})-\log\pi_{\theta_{\text{old}}}(a_{t}\mid s_{t})\right)}" display="block"><semantics><menclose notation="box"><mrow><mrow><msub><mi>r</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mpadded width="1.370em"><mi>exp</mi></mpadded><mo>⁡</mo><mrow><mo>(</mo><mrow><mrow><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>π</mi><mi>θ</mi></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>a</mi><mi>t</mi></msub><mo fence="true" lspace="0em" rspace="0em">∣</mo><msub><mi>s</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>−</mo><mrow><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>π</mi><msub><mi>θ</mi><mtext>old</mtext></msub></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>a</mi><mi>t</mi></msub><mo fence="true" lspace="0em" rspace="0em">∣</mo><msub><mi>s</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow></mrow><mo>)</mo></mrow></mrow></mrow></menclose></semantics></math><span class="hh-equation-number">(5.20)</span></div>

该比率被注入 PPO 的 clipping 目标：

<div class="hh-equation" id="ch5.e21"><math alttext="\boxed{\mathcal{L}^{\text{CLIP}}(\theta)=\hat{\mathbb{E}}_{t}\left[\min\!\left(r_{t}(\theta)\hat{A}_{t},\;\text{clip}(r_{t}(\theta),\,1{-}\epsilon,\,1{+}\epsilon)\,\hat{A}_{t}\right)\right]}" display="block"><semantics><menclose notation="box"><mrow><mrow><msup><mi>ℒ</mi><mtext>CLIP</mtext></msup><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><msub><mover accent="true"><mi>𝔼</mi><mo>^</mo></mover><mi>t</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo>[</mo><mrow><mpadded width="1.653em"><mi>min</mi></mpadded><mo>⁡</mo><mrow><mo>(</mo><mrow><msub><mi>r</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow><mo lspace="0em" rspace="0em">​</mo><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>t</mi></msub></mrow><mo rspace="0.447em">,</mo><mrow><mtext>clip</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mrow><msub><mi>r</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>,</mo><mrow><mn> 1</mn><mo>−</mo><mi>ϵ</mi></mrow><mo>,</mo><mrow><mn> 1</mn><mo>+</mo><mi>ϵ</mi></mrow><mo stretchy="false">)</mo></mrow><mo lspace="0.170em" rspace="0em">​</mo><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>t</mi></msub></mrow><mo>)</mo></mrow></mrow><mo>]</mo></mrow></mrow></mrow></menclose></semantics></math><span class="hh-equation-number">(5.21)</span></div>

### PPO 权重生命周期

<div class="hh-table" id="ch5.t1"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>阶段</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>在线 <math alttext="\theta" display="inline"><semantics><mi>θ</mi><span></span></semantics></math></span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>旧 <math alttext="\theta_{\text{old}}" display="inline"><semantics><msub><mi>θ</mi><mtext>old</mtext></msub><span></span></semantics></math></span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>比率 <math alttext="r_{t}(\theta)" display="inline"><semantics><mrow><msub><mi>r</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></mrow><span></span></semantics></math></span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">1. Rollout 开始</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>活跃副本</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>同一个活跃副本</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>始终为 <math alttext="1.0" display="inline"><semantics><mn>1.0</mn><span></span></semantics></math>（按同一性）</span></span></td></tr><tr><th class="ltx_align_left">2. Batch 第 1 步</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>计算梯度</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>冻结</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="1.0" display="inline"><semantics><mn>1.0</mn><span></span></semantics></math>（初始步骤）</span></span></td></tr><tr><th class="ltx_align_left">3. Batch 第 <math alttext="N" display="inline"><semantics><mi>N</mi><span></span></semantics></math> 步</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>修改中（<math alttext="\theta\neq\theta_{\text{old}}" display="inline"><semantics><mrow><mi>θ</mi><mo>≠</mo><msub><mi>θ</mi><mtext>old</mtext></msub></mrow><span></span></semantics></math>）</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>冻结</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>偏离 <math alttext="1.0" display="inline"><semantics><mn>1.0</mn><span></span></semantics></math>（例如 <math alttext="1.06" display="inline"><semantics><mn>1.06</mn><span></span></semantics></math>、<math alttext="0.94" display="inline"><semantics><mn>0.94</mn><span></span></semantics></math>）</span></span></td></tr><tr><th class="ltx_align_left">4. Clipping 生效</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>被 <math alttext="\epsilon" display="inline"><semantics><mi>ϵ</mi><span></span></semantics></math> 约束</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>冻结</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>被困在边界（<math alttext="1\pm\epsilon" display="inline"><semantics><mrow><mn>1</mn><mo>±</mo><mi>ϵ</mi></mrow><span></span></semantics></math>）</span></span></td></tr><tr><th class="ltx_align_left">5. 优化结束</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>高度优化</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>被丢弃</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>N/A</span></span></td></tr><tr><th class="ltx_align_left">6. 下一个周期</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="\theta\rightarrow\theta_{\text{old}}" display="inline"><semantics><mrow><mi>θ</mi><mo stretchy="false">→</mo><msub><mi>θ</mi><mtext>old</mtext></msub></mrow><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>接收新的 <math alttext="\theta" display="inline"><semantics><mi>θ</mi><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>重置回 <math alttext="1.0" display="inline"><semantics><mn>1.0</mn><span></span></semantics></math></span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">表 5.1：</span>在 PPO 训练各阶段中 <math alttext="\theta" display="inline"><semantics><mi>θ</mi></semantics></math> 与 <math alttext="\theta_{\text{old}}" display="inline"><semantics><msub><mi>θ</mi><mtext>old</mtext></msub></semantics></math> 的演化。</p></div>

### 连续 Action 空间扩展

对于连续动作空间（对 LLM 并不典型，但对机器人 RL 很重要），网络输出的不是离散 logits，而是分布参数：

<ul><li>预测的均值向量 <math alttext="\mu" display="inline"><semantics><mi>μ</mi></semantics></math></li><li>预测的标准差向量 <math alttext="\sigma" display="inline"><semantics><mi>σ</mi></semantics></math></li></ul>

对数概率通过高斯对数 PDF 计算：

<div class="hh-equation" id="ch5.e22"><math alttext="\boxed{\log\pi(a_{t}\mid s_{t})=-\frac{1}{2}\left(\frac{a_{t}-\mu}{\sigma}\right)^{\!2}-\log(\sigma)-\frac{1}{2}\log(2\pi)}" display="block"><semantics><menclose notation="box"><mrow><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mrow><mi>π</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>a</mi><mi>t</mi></msub><mo fence="true" lspace="0em" rspace="0em">∣</mo><msub><mi>s</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow></mrow><mo>=</mo><mrow><mrow><mo>−</mo><mrow><mfrac><mn>1</mn><mn>2</mn></mfrac><mo lspace="0em" rspace="0em">​</mo><msup><mrow><mo>(</mo><mfrac><mrow><msub><mi>a</mi><mi>t</mi></msub><mo>−</mo><mi>μ</mi></mrow><mi>σ</mi></mfrac><mo>)</mo></mrow><mn>2</mn></msup></mrow></mrow><mo>−</mo><mrow><mi>log</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>σ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>−</mo><mrow><mfrac><mn>1</mn><mn>2</mn></mfrac><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mn>2</mn><mo lspace="0em" rspace="0em">​</mo><mi>π</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></mrow></menclose></semantics></math><span class="hh-equation-number">(5.22)</span></div>

随后比率 <math alttext="r_{t}(\theta)=\exp(\log\pi_{\theta}-\log\pi_{\theta_{\text{old}}})" display="inline"><semantics><mrow><mrow><msub><mi>r</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mi>exp</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>π</mi><mi>θ</mi></msub></mrow><mo>−</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>π</mi><msub><mi>θ</mi><mtext>old</mtext></msub></msub></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math> 的计算方式完全相同，并送入同一个 clipping 目标。

## TRL 实现

HuggingFace TRL 库 [376] 为 LLM 的所有主流 RL 方法提供了生产级实现。

```python
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from peft import LoraConfig

# Model setup
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16, device_map="auto",
    peft_config=LoraConfig(r=64, lora_alpha=16, target_modules=["q_proj","v_proj","k_proj","o_proj"])
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# PPO config with all critical hyperparameters
ppo_config = PPOConfig(
    learning_rate=1.5e-6,        # Low LR for stability
    batch_size=128,              # Prompts per step
    mini_batch_size=16,          # Gradient accumulation unit
    ppo_epochs=4,                # Epochs per batch (reuse data)
    gamma=1.0,                   # No discounting (single turn)
    lam=0.95,                    # GAE lambda
    cliprange=0.2,               # PPO epsilon
    cliprange_value=0.2,         # Value function clipping
    vf_coef=0.1,                 # Value loss coefficient
    init_kl_coef=0.05,           # Initial KL penalty
    target_kl=6.0,               # Adaptive KL target
    whiten_rewards=True,         # Normalize advantages
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,
)

ppo_trainer = PPOTrainer(config=ppo_config, model=model, tokenizer=tokenizer,
    dataset=prompt_dataset, data_collator=collator)

# Training loop
for batch in ppo_trainer.dataloader:
    # 1. Generate responses
    query_tensors = batch["input_ids"]
    response_tensors = ppo_trainer.generate(
        query_tensors, max_new_tokens=512, temperature=0.7, top_p=0.9, do_sample=True
    )
    # 2. Score with reward model
    texts = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
    rewards = [torch.tensor(reward_model.score(q, r)) for q, r in zip(batch["query"], texts)]
    # 3. PPO update (handles KL, GAE, clipping internally)
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    # Monitor: stats["ppo/mean_scores"], stats["ppo/policy/approx_kl"]
```

## 关键超参

参数典型取值设置错误的后果<code>cliprange</code>0.2过低：不学习。过高：不稳定。<code>init_kl_coef</code>0.01–0.1过低：奖励作弊。过高：卡在 SFT 阶段。<code>target_kl</code>4–8自适应控制器的目标值。越低越保守。<code>ppo_epochs</code>4过多：过拟合到 batch。过少：浪费生成算力。<code>learning_rate</code><math alttext="1{-}5\times 10^{-6}" display="inline"><semantics><mrow><mn>1</mn><mo>−</mo><mrow><mn>5</mn><mo lspace="0.222em" rspace="0.222em">×</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>6</mn></mrow></msup></mrow></mrow></semantics></math>过高：灾难性遗忘。<code>batch_size</code>64–256越大：梯度越平滑，生成算力消耗越多。<code>temperature</code>0.7–1.0越低：探索更少。越高：优势噪声更大。

<aside class="hh-next">
<p class="hh-next-label">下一篇</p>
<p class="hh-next-title">智能体 AI 漫游指南 · 第 6 章 DPO——直接偏好优化</p>
<p class="hh-next-hint">在「智能体 AI 漫游指南」分类下继续阅读。</p>
</aside>

</div>
