---
title: "Chapter 4 RL Foundations for Language Models"
slug: "hitchhiker-agentic-ai-04-rl-foundations-for-language-models"
lang: "en"
date: "2026-09-16T00:05:00.000Z"
draft: false
featured: false
categories:
  - "hitchhiker-agentic-ai"
cover: "./cover.png"
description: "Supervised fine-tuning (SFT) teaches a model to imitate demonstrations, but imitation has a ceiling: the model can never exceed the quality…"
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

Supervised fine-tuning (SFT) teaches a model to imitate demonstrations, but imitation has a ceiling: the model can never exceed the quality of its training data. Reinforcement learning breaks this barrier. By generating novel text, receiving reward feedback, and updating toward higher-reward behaviours, an RL-trained model can <em>discover</em> strategies that no human demonstrator wrote—producing outputs that are more helpful, more accurate, and better aligned with human preferences [280].

This is the mechanism behind every frontier model: GPT-4 [274], Claude, Llama-3 [118], and DeepSeek-R1 [69] all apply RL after SFT as the critical step that transforms a capable but unsteered model into an aligned assistant.

## Two Paradigms for RL in LLMs

RL methods for language models fall into two broad paradigms, each suited to different goals:

The original motivation for applying RL to LLMs was alignment—making models helpful, harmless, and honest. Reinforcement Learning from Human Feedback (RLHF)[280, 442, 57] trains a reward model from pairwise human judgments (“which response is better?”) and then optimizes the policy to maximize that learned reward. DPO[302] simplifies this by eliminating the reward model entirely, converting preferences directly into a supervised loss. Both approaches produce aligned assistants that follow instructions and respect safety constraints.

More recently, RL has been used not just for alignment but for teaching new capabilities—particularly reasoning, mathematics, and code generation. Here the reward comes not from human preferences but from verifiable outcomes: did the model produce the correct answer? Did the code pass all tests? DeepSeek-R1 [69] demonstrated that GRPO with rule-based rewards (format correctness + answer accuracy) can train models to develop sophisticated chain-of-thought reasoning <em>without any human preference data</em>. This paradigm—RL from Verifiable Rewards (RLVR)—is now the dominant approach for building reasoning models and agentic systems.

## Text Generation as an MDP

The key insight that makes RL applicable to language models is recasting autoregressive generation as a Markov Decision Process:

Formally, the MDP for text generation is:

<ul><li>State<math alttext="s_{t}=(x,y_{1},\ldots,y_{t-1})" display="inline"><semantics><mrow><msub><mi>s</mi><mi>t</mi></msub><mo>=</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>,</mo><msub><mi>y</mi><mn>1</mn></msub><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msub><mi>y</mi><mrow><mi>t</mi><mo>−</mo><mn>1</mn></mrow></msub><mo stretchy="false">)</mo></mrow></mrow></semantics></math>: the prompt concatenated with all tokens generated so far.</li><li>Action<math alttext="a_{t}\in\{1,\ldots,|\mathcal{V}|\}" display="inline"><semantics><mrow><msub><mi>a</mi><mi>t</mi></msub><mo>∈</mo><mrow><mo stretchy="false">{</mo><mn>1</mn><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><mrow><mo stretchy="false">|</mo><mi>𝒱</mi><mo stretchy="false">|</mo></mrow><mo stretchy="false">}</mo></mrow></mrow></semantics></math>: choosing the next token from the vocabulary (32K–128K options).</li><li>Transition<math alttext="P(s_{t+1}|s_{t},a_{t})" display="inline"><semantics><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>s</mi><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub><mo lspace="0em" rspace="0em">|</mo><mrow><msub><mi>s</mi><mi>t</mi></msub><mo>,</mo><msub><mi>a</mi><mi>t</mi></msub></mrow><mo stretchy="false">)</mo></mrow></mrow></semantics></math>: deterministic—just append the chosen token. No environment stochasticity.</li><li>Reward<math alttext="r" display="inline"><semantics><mi>r</mi></semantics></math>: typically given only at the end of generation (sparse). For RLHF: the reward model score. For RLVR: correctness of the final answer.</li><li>Policy<math alttext="\pi_{\theta}(a_{t}|s_{t})" display="inline"><semantics><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>a</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">|</mo><msub><mi>s</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow></semantics></math>: the LLM’s next-token probability distribution—exactly what the softmax output already computes.</li><li>Discount<math alttext="\gamma=1.0" display="inline"><semantics><mrow><mi>γ</mi><mo>=</mo><mn>1.0</mn></mrow></semantics></math>: episodes are finite (one response), so no discounting needed.</li></ul>

This mapping is powerful because the LLM <em>already is</em> a policy—its softmax output defines <math alttext="\pi_{\theta}(a_{t}|s_{t})" display="inline"><semantics><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>a</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">|</mo><msub><mi>s</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow></semantics></math> for every state. We don’t need to build a separate policy network; we just need to adjust the weights <math alttext="\theta" display="inline"><semantics><mi>θ</mi></semantics></math> so the model assigns higher probability to token sequences that earn higher reward.

## The RLHF Pipeline

The classic RLHF pipeline [280] consists of four stages:

<ol><li>Supervised Fine-Tuning (SFT): Train a base model on high-quality demonstrations to produce a policy <math alttext="\pi_{\text{SFT}}" display="inline"><semantics><msub><mi>π</mi><mtext>SFT</mtext></msub></semantics></math> that can follow instructions.</li><li>Reward Model Training: Collect human preference comparisons (<math alttext="y_{w}\succ y_{l}" display="inline"><semantics><mrow><msub><mi>y</mi><mi>w</mi></msub><mo>≻</mo><msub><mi>y</mi><mi>l</mi></msub></mrow></semantics></math> for the same prompt) and train a reward model <math alttext="R_{\phi}(x,y)" display="inline"><semantics><mrow><msub><mi>R</mi><mi>ϕ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math> using the Bradley-Terry objective.</li><li>RL Optimization: Use the reward model as a signal to optimize the policy via PPO or GRPO, subject to a KL constraint against <math alttext="\pi_{\text{SFT}}" display="inline"><semantics><msub><mi>π</mi><mtext>SFT</mtext></msub></semantics></math>.</li><li>Evaluation and Iteration: Evaluate the aligned model, collect new failure cases, and iterate.</li></ol>

For RLVR (reasoning/agentic training), stages 1–2 are replaced: the SFT model is trained on reasoning traces, and the reward model is replaced by a verifier (e.g., checking mathematical correctness). Stage 3 remains the same—PPO or GRPO optimization against the reward signal.

## Roadmap of This Part

The chapters ahead build the complete RL-for-LLMs toolkit:

<ol><li>PPO (Chapter 5) — The clipped surrogate objective, GAE for advantage estimation, the critic network, and the full RLHF training loop. The workhorse behind GPT-4 and Claude.</li><li>DPO (Chapter 6) — Bypassing RL entirely by converting preferences into a contrastive supervised loss. Simpler but less flexible than online RL.</li><li>GRPO (Chapter 7) — DeepSeek’s critic-free algorithm that uses group-level reward normalization. The method behind DeepSeek-R1 and the dominant choice for reasoning model training.</li><li>Preference optimization variants (Chapter 8) — Online DPO, KTO, Best-of-N, and guidance on method selection.</li><li>Reward modeling (Chapter 9) — Bradley-Terry models, process vs. outcome rewards, rule-based rewards for RLVR, and multi-objective combinations.</li><li>SFT best practices (Chapter 10) — Sequence packing, chat templates, data mixing, and how SFT quality determines the RL ceiling.</li><li>Systems engineering (Chapter 11) — Distributed training at scale: parallelism strategies, generation–training decoupling, and infrastructure for hundreds of GPUs.</li></ol>

<aside class="hh-next">
<p class="hh-next-label">下一篇</p>
<p class="hh-next-title">Chapter 5 PPO — Proximal Policy Optimization</p>
<p class="hh-next-hint">在「智能体 AI 漫游指南」分类下继续阅读。</p>
</aside>

</div>
