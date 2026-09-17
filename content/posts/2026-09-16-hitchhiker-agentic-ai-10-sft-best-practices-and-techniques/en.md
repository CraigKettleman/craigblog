---
title: "The Hitchhiker's Guide to Agentic AI · Chapter 10 SFT Best Practices and Techniques"
slug: "hitchhiker-agentic-ai-10-sft-best-practices-and-techniques"
lang: "en"
date: "2026-09-16T00:11:00.000Z"
draft: false
featured: false
categories:
  - "hitchhiker-agentic-ai"
cover: "./cover.png"
description: "Supervised Fine-Tuning (SFT) is the foundation of the RLHF pipeline.…"
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

Supervised Fine-Tuning (SFT) is the foundation of the RLHF pipeline. The quality of the SFT model determines the ceiling of what RL can achieve: RL can refine and improve a behaviour, but it cannot reliably introduce a behaviour that is entirely absent from the SFT model. This section covers the key techniques for effective SFT.

## Sequence Packing for Efficiency

Sequence packing concatenates multiple short examples into a single sequence of length <code>max_seq_length</code>, separated by EOS tokens. The attention mask ensures that tokens from different examples do not attend to each other:

<ol><li>Sort examples by length (optional, improves packing efficiency).</li><li>Greedily pack examples into bins of size <code>max_seq_length</code>.</li><li>Use a block-diagonal attention mask to prevent cross-example attention.</li><li>Compute loss only on non-padding tokens.</li></ol>

## Chat Templates and Formatting

ChatML is the most widely used chat template:

```python
# ChatML format
template = """<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
{assistant_message}<|im_end|>"""
```

Llama 3 uses a different template with special tokens:

```python
# Llama 3 format
template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{assistant_message}<|eot_id|>"""
```

## Completion-Only Masking

## Data Mixing Strategies for Multi-Task SFT

Sample from each dataset proportionally to its size:

<div class="hh-equation" id="ch10.ex1"><math alttext="p_{k}=\frac{N_{k}}{\sum_{j=1}^{K}N_{j}}," display="block"><semantics><mrow><mrow><msub><mi>p</mi><mi>k</mi></msub><mo>=</mo><mfrac><msub><mi>N</mi><mi>k</mi></msub><mrow><msubsup><mo>∑</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mi>K</mi></msubsup><msub><mi>N</mi><mi>j</mi></msub></mrow></mfrac></mrow><mo>,</mo></mrow></semantics></math></div>

where <math alttext="N_{k}" display="inline"><semantics><msub><mi>N</mi><mi>k</mi></msub></semantics></math> is the number of examples in dataset <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math>. This is the default in most frameworks and works well when datasets are of similar quality.

Apply a temperature <math alttext="T" display="inline"><semantics><mi>T</mi></semantics></math> to smooth the proportions:

<div class="hh-equation" id="ch10.ex2"><math alttext="p_{k}\propto N_{k}^{1/T}." display="block"><semantics><mrow><mrow><msub><mi>p</mi><mi>k</mi></msub><mo>∝</mo><msubsup><mi>N</mi><mi>k</mi><mrow><mn>1</mn><mo>/</mo><mi>T</mi></mrow></msubsup></mrow><mo lspace="0em">.</mo></mrow></semantics></math></div>

<math alttext="T=1" display="inline"><semantics><mrow><mi>T</mi><mo>=</mo><mn>1</mn></mrow></semantics></math>: proportional mixing. <math alttext="T\to\infty" display="inline"><semantics><mrow><mi>T</mi><mo stretchy="false">→</mo><mi mathvariant="normal">∞</mi></mrow></semantics></math>: uniform mixing. <math alttext="T&lt;1" display="inline"><semantics><mrow><mi>T</mi><mo>&lt;</mo><mn>1</mn></mrow></semantics></math>: over-samples large datasets. <math alttext="T&gt;1" display="inline"><semantics><mrow><mi>T</mi><mo>&gt;</mo><mn>1</mn></mrow></semantics></math>: over-samples small datasets.

Weight datasets by estimated quality (e.g., perplexity under a reference model, human quality ratings):

<div class="hh-equation" id="ch10.ex3"><math alttext="p_{k}\propto N_{k}\cdot q_{k}," display="block"><semantics><mrow><mrow><msub><mi>p</mi><mi>k</mi></msub><mo>∝</mo><mrow><msub><mi>N</mi><mi>k</mi></msub><mo lspace="0.222em" rspace="0.222em">⋅</mo><msub><mi>q</mi><mi>k</mi></msub></mrow></mrow><mo>,</mo></mrow></semantics></math></div>

where <math alttext="q_{k}" display="inline"><semantics><msub><mi>q</mi><mi>k</mi></msub></semantics></math> is the quality score for dataset <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math>.

## When SFT Hurts – Catastrophic Forgetting and Alignment Tax

As LLMs transition through sequential training phases — pre-training <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> continued pre-training <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> SFT <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> RLHF/DPO — performance degradation frequently manifests on standard benchmarks. Two fundamentally distinct phenomena drive these regressions, and confusing them leads to wrong mitigation strategies.

### Catastrophic Forgetting (Structural Erasure)

Symptoms:

<ul><li>Complete breakdown on tasks not in fine-tuning data (e.g., model forgets how to do math after SFT on chat data)</li><li>Loss of language diversity — model only generates in the narrow style of fine-tuning distribution</li><li>Reduced factual accuracy on knowledge not reinforced during fine-tuning</li><li>Degraded multilingual ability after English-only SFT</li></ul>

Mechanistic cause — Fisher Information perspective: The Fisher Information Matrix <math alttext="F" display="inline"><semantics><mi>F</mi></semantics></math> of Task A identifies which parameters are “important” for <math alttext="\mathcal{D}_{A}" display="inline"><semantics><msub><mi>𝒟</mi><mi>A</mi></msub></semantics></math>:

<div class="hh-equation" id="ch10.e2"><math alttext="F=\mathbb{E}_{x\sim\mathcal{D}_{A}}\!\left[\nabla_{\theta}\log\pi_{\theta}(x)\,\nabla_{\theta}\log\pi_{\theta}(x)^{T}\right]" display="block"><semantics><mrow><mi>F</mi><mo>=</mo><mrow><msub><mi>𝔼</mi><mrow><mi>x</mi><mo>∼</mo><msub><mi>𝒟</mi><mi>A</mi></msub></mrow></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo>[</mo><mrow><msub><mo>∇</mo><mi>θ</mi></msub><mo lspace="0.167em" rspace="0em">​</mo><mi>log</mi><mo lspace="0.167em" rspace="0em">​</mo><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow><mo lspace="0.337em" rspace="0em">​</mo><msub><mo>∇</mo><mi>θ</mi></msub><mo lspace="0.167em" rspace="0em">​</mo><mi>log</mi><mo lspace="0.167em" rspace="0em">​</mo><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><msup><mrow><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow><mi>T</mi></msup></mrow><mo>]</mo></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(10.2)</span></div>

Parameters with high Fisher eigenvalues are critical for Task A. Unconstrained gradient descent on Task B ignores these eigenvalues entirely — <math alttext="\Delta\theta" display="inline"><semantics><mrow><mi mathvariant="normal">Δ</mi><mo lspace="0em" rspace="0em">​</mo><mi>θ</mi></mrow></semantics></math> points along <math alttext="\nabla\mathcal{L}_{B}" display="inline"><semantics><mrow><mo rspace="0.167em">∇</mo><msub><mi>ℒ</mi><mi>B</mi></msub></mrow></semantics></math> regardless of whether it destroys high-Fisher directions for <math alttext="\mathcal{L}_{A}" display="inline"><semantics><msub><mi>ℒ</mi><mi>A</mi></msub></semantics></math>.

### Alignment Tax (Behavioral Constraint)

The alignment tax is a deliberate, expected trade-off: the model’s raw capability (unconstrained generation, maximal reasoning bandwidth) decreases because the policy is constrained to produce safe, well-formatted, preference-aligned outputs.

Mechanism: During DPO/PPO, the policy <math alttext="\pi_{\theta}" display="inline"><semantics><msub><mi>π</mi><mi>θ</mi></msub></semantics></math> is penalized for deviating from the reference <math alttext="\pi_{\text{ref}}" display="inline"><semantics><msub><mi>π</mi><mtext>ref</mtext></msub></semantics></math> via KL divergence:

<div class="hh-equation" id="ch10.e3"><math alttext="r_{\text{implicit}}(x,y)=\beta\log\frac{\pi_{\theta}(y|x)}{\pi_{\text{ref}}(y|x)}" display="block"><semantics><mrow><mrow><msub><mi>r</mi><mtext>implicit</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mi>β</mi><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mfrac><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>y</mi><mo lspace="0em" rspace="0em">|</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mi>π</mi><mtext>ref</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>y</mi><mo lspace="0em" rspace="0em">|</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow></mfrac></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(10.3)</span></div>

This leash constrains the model’s output distribution — it cannot explore high-variance reasoning paths that deviate too far from the reference. The knowledge is not erased; it’s <em>suppressed</em>. The model still “knows” the answer but its distribution is flattened toward safe, generic responses.

Symptoms:

<ul><li>Over-refusal (“I can’t help with that” for benign queries)</li><li>Stylistic stiffness — hedge words, excessive caveats, verbose safety disclaimers</li><li>Lower scores on raw capability benchmarks (MMLU, HumanEval) while improving on preference benchmarks (MT-Bench, AlpacaEval)</li><li>Reduced ability to produce complex, high-entropy outputs (creative writing, novel algorithms)</li></ul>

### Comparative Taxonomy

<div class="hh-table" id="ch10.t1"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Dimension</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Catastrophic Forgetting</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Alignment Tax</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left"><span>Intentionality</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Unintentional (optimization artifact)</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Expected trade-off (incurred deliberately for safety/helpfulness)</span></span></td></tr><tr><th class="ltx_align_left"><span>Parameter state</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Prior knowledge physically overwritten</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Latent distributions constrained/truncated</span></span></td></tr><tr><th class="ltx_align_left"><span>Information</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Destroyed</span>: weights no longer encode the capability</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Suppressed</span>: knowledge exists but is harder to trigger</span></span></td></tr><tr><th class="ltx_align_left"><span>Dominant phase</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Sequential SFT, domain continued pre-training</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Preference optimization (PPO, DPO, KTO, RLHF)</span></span></td></tr><tr><th class="ltx_align_left"><span>Primary symptom</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Complete breakdown of baseline capabilities</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Over-refusal, stylistic stiffness, lower raw benchmark scores</span></span></td></tr><tr><th class="ltx_align_left"><span>Reversibility</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Irreversible without retraining from checkpoint</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Partially reversible: adjust <math alttext="\beta" display="inline"><semantics><mi>β</mi><span></span></semantics></math>, system prompt, or fine-tune</span></span></td></tr><tr><th class="ltx_align_left"><span>Detection</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Perplexity on pre-training eval set spikes</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Perplexity stable but win-rate on capability benchmarks drops</span></span></td></tr><tr><th class="ltx_align_left"><span>Scales with model size</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Similar across scales</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Smaller models pay a larger alignment tax</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 10.1:</span>Catastrophic Forgetting vs. Alignment Tax — complete comparison.</p></div>

### Mitigation Strategies

For Catastrophic Forgetting:

<ol><li>Data replay: Mix 5–10% of pre-training data into SFT dataset. Ensures gradient updates don’t completely neglect pre-training distribution.</li><li>Elastic Weight Consolidation (EWC)[186]: Add regularization <math alttext="\Omega(\theta)=\frac{\lambda}{2}\sum_{i}F_{i}(\theta_{i}-\theta_{i}^{*})^{2}" display="inline"><semantics><mrow><mrow><mi mathvariant="normal">Ω</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mfrac><mi>λ</mi><mn>2</mn></mfrac><mo lspace="0em" rspace="0em">​</mo><mrow><msub><mo>∑</mo><mi>i</mi></msub><mrow><msub><mi>F</mi><mi>i</mi></msub><mo lspace="0em" rspace="0em">​</mo><msup><mrow><mo stretchy="false">(</mo><mrow><msub><mi>θ</mi><mi>i</mi></msub><mo>−</mo><msubsup><mi>θ</mi><mi>i</mi><mo>∗</mo></msubsup></mrow><mo stretchy="false">)</mo></mrow><mn>2</mn></msup></mrow></mrow></mrow></mrow></semantics></math> that penalizes changes to parameters with high Fisher information for the original task.</li><li>LoRA / Parameter-efficient fine-tuning: Train only low-rank adapters (<math alttext="&lt;1\%" display="inline"><semantics><mrow><mphantom></mphantom><mo>&lt;</mo><mrow><mn>1</mn><mo>%</mo></mrow></mrow></semantics></math> of parameters), leaving base weights completely frozen. This prevents <em>permanent destruction</em> of pre-trained knowledge — you can always remove the adapter and recover the original model. However, while the adapter is active, the combined system <math alttext="(W_{0}+BA)" display="inline"><semantics><mrow><mo stretchy="false">(</mo><mrow><msub><mi>W</mi><mn>0</mn></msub><mo>+</mo><mrow><mi>B</mi><mo lspace="0em" rspace="0em">​</mo><mi>A</mi></mrow></mrow><mo stretchy="false">)</mo></mrow></semantics></math> can still exhibit forgetting: the adapter may shift the model’s effective behavior away from old skills. LoRA protects the checkpoint, not the active inference behavior.</li><li>Conservative learning rate: Use <math alttext="1" display="inline"><semantics><mn>1</mn></semantics></math>–<math alttext="5\times 10^{-6}" display="inline"><semantics><mrow><mn>5</mn><mo lspace="0.222em" rspace="0.222em">×</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>6</mn></mrow></msup></mrow></semantics></math> with few epochs (1–3). Larger rates accelerate forgetting.</li><li>Progressive training: Mix distributions gradually, increasing SFT data proportion over time rather than switching abruptly.</li></ol>

For Alignment Tax:

<ol><li>Tune <math alttext="\beta" display="inline"><semantics><mi>β</mi></semantics></math> carefully: Lower <math alttext="\beta" display="inline"><semantics><mi>β</mi></semantics></math> gives the model more freedom (reduces the tax) but may sacrifice safety. Optimal <math alttext="\beta\in[0.05,0.3]" display="inline"><semantics><mrow><mi>β</mi><mo>∈</mo><mrow><mo stretchy="false">[</mo><mrow><mn>0.05</mn><mo>,</mo><mn>0.3</mn></mrow><mo stretchy="false">]</mo></mrow></mrow></semantics></math> for most settings.</li><li>High-quality, diverse SFT data: Part of the alignment tax comes from SFT narrowing the output distribution; broader, more diverse SFT data reduces this component. The RL phase adds further constraint via KL regularization [280].</li><li>Conditional alignment: Train the model to be aligned only when a safety flag is active. At inference, disable constraints for benchmarking (research-only technique).</li><li>Constitutional AI / RLAIF: Use model-generated feedback to create more nuanced preference data that preserves capability while improving alignment.</li><li>Targeted RL budget: Don’t over-train with RL. Monitor capability benchmarks and stop when the tax exceeds acceptable thresholds (typically 2–5% MMLU regression).</li></ol>

## Connection to RL – SFT Quality Determines RL Ceiling

<ol><li>SFT data quality: use high-quality, diverse data. A small amount of high-quality data is better than a large amount of low-quality data.</li><li>SFT data coverage: ensure the SFT data covers the tasks you want to improve with RL. If a task is not in the SFT data, RL will struggle.</li><li>SFT training duration: do not over-train the SFT model. Over-training reduces diversity and makes RL exploration harder.</li><li>Warm-up: consider a short SFT warm-up on task-specific data before RL, even if the base model is already instruction-tuned.</li></ol>

<aside class="hh-next">
<p class="hh-next-label">下一篇</p>
<p class="hh-next-title">The Hitchhiker's Guide to Agentic AI · Chapter 11 System Architecture & Infrastructure at Scale</p>
<p class="hh-next-hint">在「智能体 AI 漫游指南」分类下继续阅读。</p>
</aside>

</div>
