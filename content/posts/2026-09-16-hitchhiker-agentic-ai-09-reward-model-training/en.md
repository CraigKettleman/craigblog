---
title: "The Hitchhiker's Guide to Agentic AI · Chapter 9 Reward Model Training"
slug: "hitchhiker-agentic-ai-09-reward-model-training"
lang: "en"
date: "2026-09-16T00:10:00.000Z"
draft: false
featured: false
categories:
  - "hitchhiker-agentic-ai"
cover: "./cover.png"
description: "Reward models are the bridge between human preferences and the RL training signal.…"
keywords:
  - "agentic AI"
  - "LLM"
  - "reinforcement learning"
  - "RLHF"
  - "MCP"
  - "A2A"
  - "agents"
---

<div class="hh-guide">

<aside class="hh-part">
<p class="hh-part-title">Part II RL Methods for LLMs</p>
</aside>

Reward models are the bridge between human preferences and the RL training signal. A well-trained reward model is essential for successful RLHF; a poorly trained one leads to reward hacking and misaligned behaviour. This section covers the theoretical foundations, practical training techniques, and architectural choices for reward models.

## Bradley-Terry Model – Full Derivation

The Bradley-Terry model [29] is the standard probabilistic framework for pairwise preference learning. Given two responses <math alttext="y_{1}" display="inline"><semantics><msub><mi>y</mi><mn>1</mn></msub></semantics></math> and <math alttext="y_{2}" display="inline"><semantics><msub><mi>y</mi><mn>2</mn></msub></semantics></math> to a prompt <math alttext="q" display="inline"><semantics><mi>q</mi></semantics></math>, the model assumes:

<div class="hh-equation" id="ch9.ex1"><math alttext="P(y_{1}\succ y_{2}\mid q)=\sigma(r(y_{1},q)-r(y_{2},q))=\frac{e^{r(y_{1},q)}}{e^{r(y_{1},q)}+e^{r(y_{2},q)}}," display="block"><semantics><mrow><mrow><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><msub><mi>y</mi><mn>1</mn></msub><mo>≻</mo><msub><mi>y</mi><mn>2</mn></msub></mrow><mo fence="true" lspace="0em" rspace="0em">∣</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mi>σ</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mn>1</mn></msub><mo>,</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mo>−</mo><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mn>2</mn></msub><mo>,</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mfrac><msup><mi>e</mi><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mn>1</mn></msub><mo>,</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow></msup><mrow><msup><mi>e</mi><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mn>1</mn></msub><mo>,</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow></msup><mo>+</mo><msup><mi>e</mi><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mn>2</mn></msub><mo>,</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow></msup></mrow></mfrac></mrow><mo>,</mo></mrow></semantics></math></div>

where <math alttext="r:\mathcal{Y}\times\mathcal{Q}\to\mathbb{R}" display="inline"><semantics><mrow><mi>r</mi><mo lspace="0.278em" rspace="0.278em">:</mo><mrow><mrow><mi>𝒴</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mi>𝒬</mi></mrow><mo stretchy="false">→</mo><mi>ℝ</mi></mrow></mrow></semantics></math> is the scalar reward function and <math alttext="\sigma" display="inline"><semantics><mi>σ</mi></semantics></math> is the sigmoid function.

Given a dataset <math alttext="\mathcal{D}=\{(q^{(k)},y_{w}^{(k)},y_{l}^{(k)})\}_{k=1}^{N}" display="inline"><semantics><mrow><mi>𝒟</mi><mo>=</mo><msubsup><mrow><mo stretchy="false">{</mo><mrow><mo stretchy="false">(</mo><msup><mi>q</mi><mrow><mo stretchy="false">(</mo><mi>k</mi><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><msubsup><mi>y</mi><mi>w</mi><mrow><mo stretchy="false">(</mo><mi>k</mi><mo stretchy="false">)</mo></mrow></msubsup><mo>,</mo><msubsup><mi>y</mi><mi>l</mi><mrow><mo stretchy="false">(</mo><mi>k</mi><mo stretchy="false">)</mo></mrow></msubsup><mo stretchy="false">)</mo></mrow><mo stretchy="false">}</mo></mrow><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mi>N</mi></msubsup></mrow></semantics></math> of preference pairs, the MLE objective is:

<div class="hh-equation" id="ch9.ex2"><math alttext="\mathcal{L}_{\text{BT}}(\phi)=-\frac{1}{N}\sum_{k=1}^{N}\log\sigma\!\bigl(r_{\phi}(y_{w}^{(k)},q^{(k)})-r_{\phi}(y_{l}^{(k)},q^{(k)})\bigr)," display="block"><semantics><mrow><msub><mi>ℒ</mi><mtext>BT</mtext></msub><mrow><mo stretchy="false">(</mo><mi>ϕ</mi><mo stretchy="false">)</mo></mrow><mo rspace="0em">=</mo><mo lspace="0em">−</mo><mfrac><mn>1</mn><mi>N</mi></mfrac><munderover><mo movablelimits="false">∑</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mi>N</mi></munderover><mi>log</mi><mpadded width="0.437em"><mi>σ</mi></mpadded><mrow><mo maxsize="1.200em" minsize="1.200em">(</mo><msub><mi>r</mi><mi>ϕ</mi></msub><mrow><mo stretchy="false">(</mo><msubsup><mi>y</mi><mi>w</mi><mrow><mo stretchy="false">(</mo><mi>k</mi><mo stretchy="false">)</mo></mrow></msubsup><mo>,</mo><msup><mi>q</mi><mrow><mo stretchy="false">(</mo><mi>k</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mo>−</mo><msub><mi>r</mi><mi>ϕ</mi></msub><mrow><mo stretchy="false">(</mo><msubsup><mi>y</mi><mi>l</mi><mrow><mo stretchy="false">(</mo><mi>k</mi><mo stretchy="false">)</mo></mrow></msubsup><mo>,</mo><msup><mi>q</mi><mrow><mo stretchy="false">(</mo><mi>k</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mo maxsize="1.200em" minsize="1.200em">)</mo></mrow><mo>,</mo></mrow></semantics></math></div>

where <math alttext="r_{\phi}" display="inline"><semantics><msub><mi>r</mi><mi>ϕ</mi></msub></semantics></math> is a neural network parameterised by <math alttext="\phi" display="inline"><semantics><mi>ϕ</mi></semantics></math>. This is a binary cross-entropy loss where the “positive” class is the preferred response.

A common extension adds a margin <math alttext="m" display="inline"><semantics><mi>m</mi></semantics></math> to ensure a minimum gap between winning and losing rewards:

<div class="hh-equation" id="ch9.ex3"><math alttext="\mathcal{L}_{\text{margin}}=-\frac{1}{N}\sum_{k=1}^{N}\log\sigma\!\bigl(r_{\phi}(y_{w}^{(k)},q^{(k)})-r_{\phi}(y_{l}^{(k)},q^{(k)})-m\bigr)." display="block"><semantics><mrow><msub><mi>ℒ</mi><mtext>margin</mtext></msub><mo rspace="0em">=</mo><mo lspace="0em">−</mo><mfrac><mn>1</mn><mi>N</mi></mfrac><munderover><mo movablelimits="false">∑</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mi>N</mi></munderover><mi>log</mi><mpadded width="0.437em"><mi>σ</mi></mpadded><mrow><mo maxsize="1.200em" minsize="1.200em">(</mo><msub><mi>r</mi><mi>ϕ</mi></msub><mrow><mo stretchy="false">(</mo><msubsup><mi>y</mi><mi>w</mi><mrow><mo stretchy="false">(</mo><mi>k</mi><mo stretchy="false">)</mo></mrow></msubsup><mo>,</mo><msup><mi>q</mi><mrow><mo stretchy="false">(</mo><mi>k</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mo>−</mo><msub><mi>r</mi><mi>ϕ</mi></msub><mrow><mo stretchy="false">(</mo><msubsup><mi>y</mi><mi>l</mi><mrow><mo stretchy="false">(</mo><mi>k</mi><mo stretchy="false">)</mo></mrow></msubsup><mo>,</mo><msup><mi>q</mi><mrow><mo stretchy="false">(</mo><mi>k</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mo>−</mo><mi>m</mi><mo maxsize="1.200em" minsize="1.200em">)</mo></mrow><mo lspace="0em">.</mo></mrow></semantics></math></div>

## Reward Model Architectures

The architecture is:

<ol><li>Backbone: a pretrained LLM (e.g., Llama, Mistral) that encodes the prompt-response pair into a sequence of hidden states.</li><li>Pooling: extract the hidden state at the last token position (for decoder-only models) or the <code>[CLS]</code> token (for encoder models).</li><li>Regression head: a linear layer <math alttext="W\in\mathbb{R}^{d\times 1}" display="inline"><semantics><mrow><mi>W</mi><mo>∈</mo><msup><mi>ℝ</mi><mrow><mi>d</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mn>1</mn></mrow></msup></mrow></semantics></math> that maps the pooled hidden state to a scalar reward.</li></ol>

## Reward Model Training Tricks

Raw reward model outputs can have arbitrary scale and offset. Centering the rewards (subtracting the mean) stabilises RL training:

<div class="hh-equation" id="ch9.ex4"><math alttext="r_{\text{centered}}(y,q)=r_{\phi}(y,q)-\mathbb{E}_{y^{\prime}\sim\pi_{\theta}}[r_{\phi}(y^{\prime},q)]." display="block"><semantics><mrow><mrow><mrow><msub><mi>r</mi><mtext>centered</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>y</mi><mo>,</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mrow><msub><mi>r</mi><mi>ϕ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>y</mi><mo>,</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mo>−</mo><mrow><msub><mi>𝔼</mi><mrow><msup><mi>y</mi><mo>′</mo></msup><mo>∼</mo><msub><mi>π</mi><mi>θ</mi></msub></mrow></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">[</mo><mrow><msub><mi>r</mi><mi>ϕ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msup><mi>y</mi><mo>′</mo></msup><mo>,</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mo stretchy="false">]</mo></mrow></mrow></mrow></mrow><mo lspace="0em">.</mo></mrow></semantics></math></div>

In TRL, this is implemented via the <code>center_rewards_coefficient</code> parameter, which adds a regularisation term to the reward model loss that penalises non-zero mean rewards.

Reward models are known to exhibit <em>length bias</em>: they tend to assign higher rewards to longer responses, regardless of quality. This can be corrected by:

<ol><li>Length normalisation: divide the reward by the response length.</li><li>Length-controlled training: include length as a feature and train the model to be length-invariant.</li><li>Calibration: post-hoc regression to remove the length effect.</li></ol>

Adding a margin <math alttext="m" display="inline"><semantics><mi>m</mi></semantics></math> to the Bradley-Terry loss ensures the reward model assigns meaningfully different scores to preferred and dispreferred responses:

<div class="hh-equation" id="ch9.ex5"><math alttext="\mathcal{L}_{\text{margin}}=\max\!\bigl(0,\;m-(r_{w}-r_{l})\bigr)." display="block"><semantics><mrow><mrow><msub><mi>ℒ</mi><mtext>margin</mtext></msub><mo>=</mo><mrow><mpadded width="1.808em"><mi>max</mi></mpadded><mo>⁡</mo><mrow><mo maxsize="1.200em" minsize="1.200em">(</mo><mn>0</mn><mo>,</mo><mrow><mi>m</mi><mo>−</mo><mrow><mo stretchy="false">(</mo><mrow><msub><mi>r</mi><mi>w</mi></msub><mo>−</mo><msub><mi>r</mi><mi>l</mi></msub></mrow><mo stretchy="false">)</mo></mrow></mrow><mo maxsize="1.200em" minsize="1.200em">)</mo></mrow></mrow></mrow><mo lspace="0em">.</mo></mrow></semantics></math></div>

## Process Reward Models vs Outcome Reward Models

Step-level annotations can be generated automatically using:

<ol><li>Monte Carlo rollouts: for each intermediate step, sample multiple completions and use the fraction that reach the correct answer as the step reward.</li><li>LLM-as-judge: use a strong LLM to evaluate each step.</li><li>Formal verification: for math/code, use a verifier to check each step.</li></ol>

## Rule-Based Rewards for RLVR

Reinforcement Learning from Verifiable Rewards (RLVR) uses deterministic, rule-based reward functions instead of learned reward models. This substantially reduces reward hacking (though models can still exploit format tricks, edge cases, or test memorization) and is the approach used in DeepSeek-R1 [69].

## Multi-Objective Rewards – Combination Strategies

When training with multiple reward signals, the combination strategy significantly affects the final policy.

## Listwise Rank-Based Rewards

While the Bradley-Terry model handles <em>pairwise</em> preferences (<math alttext="y_{w}\succ y_{l}" display="inline"><semantics><mrow><msub><mi>y</mi><mi>w</mi></msub><mo>≻</mo><msub><mi>y</mi><mi>l</mi></msub></mrow></semantics></math>), many practical scenarios involve ranking multiple responses simultaneously. Listwise reward models learn from complete orderings, providing richer training signal and enabling better calibration.

The Plackett-Luce (PL) model [291] is the standard extension of Bradley-Terry to full rankings. Given <math alttext="K" display="inline"><semantics><mi>K</mi></semantics></math> responses <math alttext="y_{1},\ldots,y_{K}" display="inline"><semantics><mrow><msub><mi>y</mi><mn>1</mn></msub><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msub><mi>y</mi><mi>K</mi></msub></mrow></semantics></math> with ranking <math alttext="\pi" display="inline"><semantics><mi>π</mi></semantics></math> (where <math alttext="\pi(1)" display="inline"><semantics><mrow><mi>π</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></mrow></semantics></math> is the best):

<aside class="hh-next">
<p class="hh-next-label">下一篇</p>
<p class="hh-next-title">The Hitchhiker's Guide to Agentic AI · Chapter 10 SFT Best Practices and Techniques</p>
<p class="hh-next-hint">在「智能体 AI 漫游指南」分类下继续阅读。</p>
</aside>

</div>
