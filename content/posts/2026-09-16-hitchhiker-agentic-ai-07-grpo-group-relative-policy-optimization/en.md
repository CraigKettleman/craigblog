---
title: "Chapter 7 GRPO — Group Relative Policy Optimization"
slug: "hitchhiker-agentic-ai-07-grpo-group-relative-policy-optimization"
lang: "en"
date: "2026-09-16T00:08:00.000Z"
draft: false
featured: false
categories:
  - "hitchhiker-agentic-ai"
cover: "./cover.png"
description: "Group Relative Policy Optimization (GRPO) [323] is a reinforcement learning algorithm designed specifically for language models that elimina…"
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

Group Relative Policy Optimization (GRPO) [323] is a reinforcement learning algorithm designed specifically for language models that eliminates the need for a separate value network (critic). Introduced by DeepSeek as part of their DeepSeekMath work and later scaled to DeepSeek-R1 [69], GRPO has rapidly become the dominant RL method for LLM training—adopted by most open-source alignment frameworks (TRL, OpenRLHF, veRL) as the default algorithm.

The core idea is deceptively simple: instead of training a neural network to predict expected reward (the critic in PPO), GRPO <em>estimates</em> it empirically by generating multiple responses to the same prompt and using the group’s reward statistics as a baseline. This removes an entire model from memory, halves the engineering complexity, and—surprisingly—often outperforms PPO because empirical baselines are more accurate than a poorly-trained value function.

GRPO is particularly effective for:

<ul><li>Reasoning tasks with verifiable rewards (math, code) where binary correctness provides a clean signal.</li><li>Large models (70B+) where the memory savings from removing the critic are critical.</li><li>Multi-turn and agentic settings where value estimation across tool calls is intractable.</li></ul>

This chapter covers GRPO’s motivation, algorithm, key variants (Dr. GRPO, DAPO, 2-GRPO, GDPO), and practical implementation with TRL.

## Motivation

PPO’s value model (critic) has three major problems for language:

<ol><li>Memory: The value head shares the policy backbone (140GB for 70B). Doubles memory if separate.</li><li>Accuracy: Predicting expected reward for a partial sequence is extremely hard. The value function is often wrong <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> wrong advantages <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> wrong gradient direction.</li><li>Training: Value head needs many samples to converge. During early RL, it gives noisy predictions that destabilize policy learning.</li></ol>

GRPO’s key insight[323]: Instead of learning <math alttext="V(s)" display="inline"><semantics><mrow><mi>V</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>s</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math>, <em>estimate</em> it empirically from a group of samples. Generate <math alttext="G" display="inline"><semantics><mi>G</mi></semantics></math> responses to the same prompt, compute their rewards, and use the group statistics as the baseline.

## Algorithm

<ol><li>For each prompt <math alttext="x" display="inline"><semantics><mi>x</mi></semantics></math>, sample <math alttext="G" display="inline"><semantics><mi>G</mi></semantics></math> completions: <math alttext="\{y_{1},\ldots,y_{G}\}\sim\pi_{\theta}(\cdot|x)" display="inline"><semantics><mrow><mrow><mo stretchy="false">{</mo><msub><mi>y</mi><mn>1</mn></msub><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msub><mi>y</mi><mi>G</mi></msub><mo stretchy="false">}</mo></mrow><mo>∼</mo><msub><mi>π</mi><mi>θ</mi></msub><mrow><mo stretchy="false">(</mo><mo lspace="0em" rspace="0em">⋅</mo><mo fence="false" rspace="0.167em" stretchy="false">|</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math></li><li>Score each: <math alttext="r_{i}=R(x,y_{i})" display="inline"><semantics><mrow><msub><mi>r</mi><mi>i</mi></msub><mo>=</mo><mrow><mi>R</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>,</mo><msub><mi>y</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math></li><li>Normalize within group: <math alttext="\hat{A}_{i}=\frac{r_{i}-\mu_{G}}{\sigma_{G}}" display="inline"><semantics><mrow><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>i</mi></msub><mo>=</mo><mfrac><mrow><msub><mi>r</mi><mi>i</mi></msub><mo>−</mo><msub><mi>μ</mi><mi>G</mi></msub></mrow><msub><mi>σ</mi><mi>G</mi></msub></mfrac></mrow></semantics></math> where <math alttext="\mu_{G}=\frac{1}{G}\sum_{j}r_{j}" display="inline"><semantics><mrow><msub><mi>μ</mi><mi>G</mi></msub><mo>=</mo><mrow><mfrac><mn>1</mn><mi>G</mi></mfrac><mo lspace="0em" rspace="0em">​</mo><mrow><msub><mo>∑</mo><mi>j</mi></msub><msub><mi>r</mi><mi>j</mi></msub></mrow></mrow></mrow></semantics></math>, <math alttext="\sigma_{G}=\text{std}(\{r_{j}\})" display="inline"><semantics><mrow><msub><mi>σ</mi><mi>G</mi></msub><mo>=</mo><mrow><mtext>std</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mrow><mo stretchy="false">{</mo><msub><mi>r</mi><mi>j</mi></msub><mo stretchy="false">}</mo></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math></li><li>Apply PPO-style clipped update using these advantages</li></ol>

<div class="hh-equation" id="ch7.e1"><math alttext="\boxed{\hat{A}_{i}=\frac{r_{i}-\mu_{G}}{\sigma_{G}},\qquad L=\mathbb{E}\left[\min\left(r_{t}(\theta)\hat{A}_{i},\;\text{clip}(r_{t}(\theta),1{\pm}\epsilon)\hat{A}_{i}\right)\right]-\beta D_{\text{KL}}[\pi_{\theta}\|\pi_{\text{ref}}]}" display="block"><semantics><menclose notation="box"><mrow><mover accent="true"><mi>A</mi><mo>^</mo></mover><msub><mrow></mrow><mi>i</mi></msub><mo>=</mo><mfrac><mrow><msub><mi>r</mi><mi>i</mi></msub><mo>−</mo><msub><mi>μ</mi><mi>G</mi></msub></mrow><msub><mi>σ</mi><mi>G</mi></msub></mfrac><mo rspace="2.167em">,</mo><mi>L</mi><mo>=</mo><mi>𝔼</mi><mo>[</mo><mi>min</mi><mo>(</mo><mi>r</mi><msub><mrow></mrow><mi>t</mi></msub><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo><mover accent="true"><mi>A</mi><mo>^</mo></mover><msub><mrow></mrow><mi>i</mi></msub><mo rspace="0.447em">,</mo><mtext>clip</mtext><mo stretchy="false">(</mo><mi>r</mi><msub><mrow></mrow><mi>t</mi></msub><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo><mo>,</mo><mn>1</mn><mo>±</mo><mi>ϵ</mi><mo stretchy="false">)</mo><mover accent="true"><mi>A</mi><mo>^</mo></mover><msub><mrow></mrow><mi>i</mi></msub><mo>)</mo><mo>]</mo><mo>−</mo><mi>β</mi><mi>D</mi><msub><mrow></mrow><mtext>KL</mtext></msub><mo stretchy="false">[</mo><mi>π</mi><msub><mrow></mrow><mi>θ</mi></msub><mo lspace="0em" rspace="0.167em">∥</mo><mi>π</mi><msub><mrow></mrow><mtext>ref</mtext></msub><mo stretchy="false">]</mo></mrow></menclose></semantics></math><span class="hh-equation-number">(7.1)</span></div>

<figure id="ch7.f1"><img src="./fig_029_fig29.png" alt="Figure 7.1: GRPO in action: G=5G{=}5 responses are sampled for a single math prompt. Three are correct (r=1r{=}1), two a" loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 7.1:</span>GRPO in action: <math alttext="G{=}5" display="inline"><semantics><mrow><mi>G</mi><mo>=</mo><mn>5</mn></mrow></semantics></math> responses are sampled for a single math prompt. Three are correct (<math alttext="r{=}1" display="inline"><semantics><mrow><mi>r</mi><mo>=</mo><mn>1</mn></mrow></semantics></math>), two are wrong (<math alttext="r{=}0" display="inline"><semantics><mrow><mi>r</mi><mo>=</mo><mn>0</mn></mrow></semantics></math>). The group mean <math alttext="\mu_{G}{=}0.6" display="inline"><semantics><mrow><msub><mi>μ</mi><mi>G</mi></msub><mo>=</mo><mn>0.6</mn></mrow></semantics></math> acts as the baseline; correct responses receive positive advantage (reinforced), wrong ones receive negative advantage (suppressed).</figcaption></figure>

## TRL Implementation

The following shows a minimal working example using HuggingFace TRL.

```python
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

grpo_config = GRPOConfig(
    output_dir="./grpo_output",
    num_generations=8,           # G = group size
    temperature=1.0,             # High temp for diversity within group
    max_completion_length=2048,  # Max response length
    beta=0.04,                   # KL penalty coefficient
    learning_rate=1e-6,
    per_device_train_batch_size=2,  # Prompts per device (x8 gens = 16 responses)
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    bf16=True,
    gradient_checkpointing=True,
    max_grad_norm=0.5,
    logging_steps=10,
    # vLLM generation for speed (critical for GRPO due to 8x generation)
    use_vllm=True,
    vllm_gpu_memory_utilization=0.7,
)

# Reward function: binary correctness for math
def reward_fn(completions, prompts, **kwargs):
    """Return list of floats: 1.0 if correct, 0.0 if wrong."""
    rewards = []
    for completion, prompt in zip(completions, prompts):
        answer = extract_answer(completion)
        expected = get_ground_truth(prompt)
        rewards.append(1.0 if answer == expected else 0.0)
    return rewards

# Can combine multiple reward functions!
def format_reward_fn(completions, **kwargs):
    """Bonus for using proper LaTeX formatting."""
    return [0.5 if "\\boxed{" in c else 0.0 for c in completions]

trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    reward_funcs=[reward_fn, format_reward_fn],  # Multi-objective!
    train_dataset=math_dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

## Group Size Analysis

<math alttext="G" display="inline"><semantics><mi>G</mi></semantics></math>Signal QualityComputeWhen to Use2Very noisy (coin flip)LowNever recommended — too noisy for stable learning4ModerateModerateQuick experiments, easy tasks (pass rate <math alttext="&gt;" display="inline"><semantics><mo>&gt;</mo></semantics></math> 50%)8Good (standard)HighDefault. Good balance for most tasks16ExcellentVery highHard tasks (pass rate <math alttext="&lt;" display="inline"><semantics><mo>&lt;</mo></semantics></math> 20%), need many attempts to get positives32Near-perfectExtremeOnly if you have massive compute and very hard task

## GRPO Variants and Extensions

### Diversity in GRPO Groups

<div class="hh-table" id="ch7.t1"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Method</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>How It Promotes Diversity</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">Entropy bonus</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Add <math alttext="\alpha H(\pi_{\theta})" display="inline"><semantics><mrow><mi>α</mi><mo lspace="0em" rspace="0em">​</mo><mi>H</mi><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>π</mi><mi>θ</mi></msub><mo stretchy="false">)</mo></mrow></mrow><span></span></semantics></math> to the reward. Directly penalizes low-entropy (deterministic) policies.</span></span></td></tr><tr><th class="ltx_align_left">KL penalty</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="-\beta D_{\text{KL}}[\pi_{\theta}\|\pi_{\text{ref}}]" display="inline"><semantics><mrow><mo>−</mo><mi>β</mi><msub><mi>D</mi><mtext>KL</mtext></msub><mrow><mo stretchy="false">[</mo><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0.167em">∥</mo><msub><mi>π</mi><mtext>ref</mtext></msub><mo stretchy="false">]</mo></mrow></mrow><span></span></semantics></math> prevents collapse toward a single mode.</span></span></td></tr><tr><th class="ltx_align_left">Rejection sampling</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Generate many candidates, keep top-<math alttext="k" display="inline"><semantics><mi>k</mi><span></span></semantics></math> by reward. Naturally selects for diverse high-quality responses.</span></span></td></tr><tr><th class="ltx_align_left">Best-of-N</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>At inference: generate <math alttext="N" display="inline"><semantics><mi>N</mi><span></span></semantics></math> responses, score all, return the best. Diversity comes from sampling.</span></span></td></tr><tr><th class="ltx_align_left">DPO with diverse pairs</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Train on pairs where chosen/rejected differ in <em>approach</em>, not just quality.</span></span></td></tr><tr><th class="ltx_align_left">Multi-reward</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Use multiple reward models (safety, helpfulness, code quality). Prevents collapsing to one dimension.</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 7.1:</span>Diversity-promoting methods for RL training.</p></div>

Post-training alignment (RLHF, DPO) often reduces output diversity due to <em>typicality bias</em>: human annotators systematically prefer familiar, “typical” text over novel alternatives. This mode collapse is a data-level phenomenon, not purely algorithmic.

Verbalized Sampling (VS) [422] is a training-free prompting strategy that circumvents this collapse by asking the model to explicitly verbalize a probability distribution over multiple responses in a single generation.

```python
# Verbalized Sampling: prompt model to output distribution
def verbalized_sample(model, tokenizer, task, n=5):
    prompt = (
        f"{task}\n\n"
        f"Generate {n} different responses and assign a probability "
        f"to each (probabilities should sum to 1.0). "
        f"Format: [response] (probability: X.XX)"
    )
    output = model.generate(
        tokenizer(prompt, return_tensors="pt").input_ids,
        max_new_tokens=1024,
        temperature=0.7,
        do_sample=True,
    )
    # Parse responses and probabilities from output
    responses, probs = parse_verbalized_distribution(
        tokenizer.decode(output[0])
    )
    # Sample from the verbalized distribution
    import random
    chosen = random.choices(responses, weights=probs, k=1)[0]
    return chosen
```

Before diving into the extensions, let us briefly recap the base GRPO algorithm established in the previous sections. The core mechanism—sampling a group of completions, normalizing their rewards, and applying a clipped policy gradient—is elegant in its simplicity. However, practitioners quickly discovered specific failure modes: pretraining bias diluting gradients (Dr. GRPO), symmetric clipping limiting exploration (DAPO), wasteful large group sizes (2-GRPO), and reward-scale imbalance in multi-objective settings (GDPO). The following sections address each of these in turn.

### DAPO – Dynamic Adaptive Policy Optimization

Standard PPO/GRPO clips the importance ratio symmetrically at <math alttext="[1-\epsilon,1+\epsilon]" display="inline"><semantics><mrow><mo stretchy="false">[</mo><mrow><mrow><mn>1</mn><mo>−</mo><mi>ϵ</mi></mrow><mo>,</mo><mrow><mn>1</mn><mo>+</mo><mi>ϵ</mi></mrow></mrow><mo stretchy="false">]</mo></mrow></semantics></math>. DAPO replaces this with an asymmetric band:

<div class="hh-equation" id="ch7.ex3"><math alttext="\boxed{\mathrm{clip}_{\text{DAPO}}(\rho,A)=\begin{cases}\mathrm{clip}(\rho,\,1-\epsilon,\,1+\epsilon_{\text{high}})&amp;\text{if }A&gt;0\\
\mathrm{clip}(\rho,\,1-\epsilon,\,1+\epsilon)&amp;\text{if }A\leq 0\end{cases}}" display="block"><semantics><menclose notation="box"><mrow><mrow><msub><mi>clip</mi><mtext>DAPO</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>ρ</mi><mo>,</mo><mi>A</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mo>{</mo><mtable columnspacing="5pt" displaystyle="true" rowspacing="0pt"><mtr><mtd columnalign="left"><mrow><mi>clip</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>ρ</mi><mo>,</mo><mrow><mn> 1</mn><mo>−</mo><mi>ϵ</mi></mrow><mo>,</mo><mrow><mn> 1</mn><mo>+</mo><msub><mi>ϵ</mi><mtext>high</mtext></msub></mrow><mo stretchy="false">)</mo></mrow></mrow></mtd><mtd columnalign="left"><mrow><mrow><mtext>if </mtext><mo lspace="0em" rspace="0em">​</mo><mi>A</mi></mrow><mo>&gt;</mo><mn>0</mn></mrow></mtd></mtr><mtr><mtd columnalign="left"><mrow><mi>clip</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>ρ</mi><mo>,</mo><mrow><mn> 1</mn><mo>−</mo><mi>ϵ</mi></mrow><mo>,</mo><mrow><mn> 1</mn><mo>+</mo><mi>ϵ</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></mtd><mtd columnalign="left"><mrow><mrow><mtext>if </mtext><mo lspace="0em" rspace="0em">​</mo><mi>A</mi></mrow><mo>≤</mo><mn>0</mn></mrow></mtd></mtr></mtable></mrow></mrow></menclose></semantics></math></div>

where <math alttext="\epsilon_{\text{high}}&gt;\epsilon" display="inline"><semantics><mrow><msub><mi>ϵ</mi><mtext>high</mtext></msub><mo>&gt;</mo><mi>ϵ</mi></mrow></semantics></math> (typical values: <math alttext="\epsilon=0.2" display="inline"><semantics><mrow><mi>ϵ</mi><mo>=</mo><mn>0.2</mn></mrow></semantics></math>, <math alttext="\epsilon_{\text{high}}=0.28" display="inline"><semantics><mrow><msub><mi>ϵ</mi><mtext>high</mtext></msub><mo>=</mo><mn>0.28</mn></mrow></semantics></math>). When the advantage is positive the policy is allowed to move further toward the good token; when the advantage is negative the usual conservative clipping applies to avoid over-suppression.

Base GRPO divides the loss by the <em>number of sequences</em><math alttext="G" display="inline"><semantics><mi>G</mi></semantics></math>. DAPO divides by the <em>total number of tokens</em> across all sequences:

<div class="hh-equation" id="ch7.ex4"><math alttext="\mathcal{L}_{\text{token}}=-\frac{1}{\sum_{i=1}^{G}|o_{i}|}\sum_{i=1}^{G}\sum_{t=1}^{|o_{i}|}\min\!\bigl(\rho_{i,t}\hat{A}_{i},\;\mathrm{clip}_{\text{DAPO}}(\rho_{i,t},\hat{A}_{i})\,\hat{A}_{i}\bigr)." display="block"><semantics><mrow><msub><mi>ℒ</mi><mtext>token</mtext></msub><mo rspace="0em">=</mo><mo lspace="0em">−</mo><mfrac><mn>1</mn><mrow><msubsup><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>G</mi></msubsup><mrow><mo lspace="0em" stretchy="false">|</mo><msub><mi>o</mi><mi>i</mi></msub><mo stretchy="false">|</mo></mrow></mrow></mfrac><munderover><mo movablelimits="false" rspace="0em">∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>G</mi></munderover><munderover><mo movablelimits="false">∑</mo><mrow><mi>t</mi><mo>=</mo><mn>1</mn></mrow><mrow><mo stretchy="false">|</mo><msub><mi>o</mi><mi>i</mi></msub><mo stretchy="false">|</mo></mrow></munderover><mpadded width="1.653em"><mi>min</mi></mpadded><mrow><mo maxsize="1.200em" minsize="1.200em">(</mo><msub><mi>ρ</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>i</mi></msub><mo rspace="0.447em">,</mo><msub><mi>clip</mi><mtext>DAPO</mtext></msub><mrow><mo stretchy="false">(</mo><msub><mi>ρ</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub><mo>,</mo><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>i</mi></msub><mo rspace="0.170em" stretchy="false">)</mo></mrow><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>i</mi></msub><mo maxsize="1.200em" minsize="1.200em">)</mo></mrow><mo lspace="0em">.</mo></mrow></semantics></math></div>

This prevents long completions from dominating the gradient signal simply because they contain more tokens.

When a completion is truncated (no EOS token within the maximum length budget), it provides <em>misleading</em> signal: the model is penalised for tokens that were generated correctly but happened to appear before the truncation boundary. DAPO masks these completions entirely:

<div class="hh-equation" id="ch7.ex5"><math alttext="m_{i}=\mathbf{1}[\text{EOS}\in o_{i}],\qquad\mathcal{L}_{\text{filtered}}=-\frac{\sum_{i=1}^{G}m_{i}\sum_{t}(\cdots)}{\sum_{i=1}^{G}m_{i}|o_{i}|}." display="block"><semantics><mrow><msub><mi>m</mi><mi>i</mi></msub><mo>=</mo><mn>𝟏</mn><mrow><mo stretchy="false">[</mo><mtext>EOS</mtext><mo>∈</mo><msub><mi>o</mi><mi>i</mi></msub><mo stretchy="false">]</mo></mrow><mo rspace="2.167em">,</mo><msub><mi>ℒ</mi><mtext>filtered</mtext></msub><mo rspace="0em">=</mo><mo lspace="0em">−</mo><mfrac><mrow><msubsup><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>G</mi></msubsup><mrow><msub><mi>m</mi><mi>i</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><msub><mo rspace="0em">∑</mo><mi>t</mi></msub><mrow><mo stretchy="false">(</mo><mo lspace="0em" rspace="0em">⋯</mo><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow><mrow><msubsup><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>G</mi></msubsup><mrow><msub><mi>m</mi><mi>i</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">|</mo><msub><mi>o</mi><mi>i</mi></msub><mo stretchy="false">|</mo></mrow></mrow></mrow></mfrac><mo lspace="0em">.</mo></mrow></semantics></math></div>

Rather than a hard mask, a softer variant applies a length penalty that grows smoothly as completions approach the maximum length <math alttext="L_{\max}" display="inline"><semantics><msub><mi>L</mi><mi>max</mi></msub></semantics></math>:

<div class="hh-equation" id="ch7.ex6"><math alttext="r_{i}\leftarrow r_{i}-\lambda\cdot\max\!\left(0,\,\frac{|o_{i}|-L_{\text{cache}}}{L_{\max}-L_{\text{cache}}}\right)," display="block"><semantics><mrow><mrow><msub><mi>r</mi><mi>i</mi></msub><mo stretchy="false">←</mo><mrow><msub><mi>r</mi><mi>i</mi></msub><mo>−</mo><mrow><mi>λ</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><mrow><mpadded width="1.808em"><mi>max</mi></mpadded><mo>⁡</mo><mrow><mo>(</mo><mn>0</mn><mo rspace="0.337em">,</mo><mfrac><mrow><mrow><mo stretchy="false">|</mo><msub><mi>o</mi><mi>i</mi></msub><mo stretchy="false">|</mo></mrow><mo>−</mo><msub><mi>L</mi><mtext>cache</mtext></msub></mrow><mrow><msub><mi>L</mi><mi>max</mi></msub><mo>−</mo><msub><mi>L</mi><mtext>cache</mtext></msub></mrow></mfrac><mo>)</mo></mrow></mrow></mrow></mrow></mrow><mo>,</mo></mrow></semantics></math></div>

where <math alttext="L_{\text{cache}}" display="inline"><semantics><msub><mi>L</mi><mtext>cache</mtext></msub></semantics></math> is a “safe” length threshold.

DAPO re-samples prompts whose entire group of completions receives the same reward (all correct or all incorrect), because such groups contribute zero gradient after normalisation. This keeps the effective batch size stable throughout training.

### GSPO – Group Sequence Policy Optimization

GSPO [51] defines a <em>sequence-level</em> importance weight as the geometric mean of per-token ratios, which equals the <math alttext="|o_{i}|" display="inline"><semantics><mrow><mo stretchy="false">|</mo><msub><mi>o</mi><mi>i</mi></msub><mo stretchy="false">|</mo></mrow></semantics></math>-th root of the full sequence probability ratio:

<div class="hh-equation" id="ch7.ex7"><math alttext="\boxed{s_{i}(\theta)=\left(\frac{\pi_{\theta}(o_{i}\mid q)}{\pi_{\text{old}}(o_{i}\mid q)}\right)^{1/|o_{i}|}=\exp\!\left(\frac{1}{|o_{i}|}\sum_{t=1}^{|o_{i}|}\log\frac{\pi_{\theta}(o_{i,t}|q,o_{i,&lt;t})}{\pi_{\text{old}}(o_{i,t}|q,o_{i,&lt;t})}\right).}" display="block"><semantics><menclose notation="box"><mrow><mrow><mrow><msub><mi>s</mi><mi>i</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><msup><mrow><mo>(</mo><mfrac><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>o</mi><mi>i</mi></msub><mo fence="true" lspace="0em" rspace="0em">∣</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mi>π</mi><mtext>old</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>o</mi><mi>i</mi></msub><mo fence="true" lspace="0em" rspace="0em">∣</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow></mfrac><mo>)</mo></mrow><mrow><mn>1</mn><mo>/</mo><mrow><mo stretchy="false">|</mo><msub><mi>o</mi><mi>i</mi></msub><mo stretchy="false">|</mo></mrow></mrow></msup><mo>=</mo><mrow><mpadded width="1.370em"><mi>exp</mi></mpadded><mo>⁡</mo><mrow><mo>(</mo><mrow><mfrac><mn>1</mn><mrow><mo stretchy="false">|</mo><msub><mi>o</mi><mi>i</mi></msub><mo stretchy="false">|</mo></mrow></mfrac><mo lspace="0em" rspace="0em">​</mo><mrow><munderover><mo movablelimits="false">∑</mo><mrow><mi>t</mi><mo>=</mo><mn>1</mn></mrow><mrow><mo stretchy="false">|</mo><msub><mi>o</mi><mi>i</mi></msub><mo stretchy="false">|</mo></mrow></munderover><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mfrac><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>o</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub><mo lspace="0em" rspace="0em">|</mo><mrow><mi>q</mi><mo>,</mo><msub><mi>o</mi><mrow><mrow><mi>i</mi><mo>,</mo><mphantom></mphantom></mrow><mo>&lt;</mo><mi>t</mi></mrow></msub></mrow><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mi>π</mi><mtext>old</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>o</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub><mo lspace="0em" rspace="0em">|</mo><mrow><mi>q</mi><mo>,</mo><msub><mi>o</mi><mrow><mrow><mi>i</mi><mo>,</mo><mphantom></mphantom></mrow><mo>&lt;</mo><mi>t</mi></mrow></msub></mrow><mo stretchy="false">)</mo></mrow></mrow></mfrac></mrow></mrow></mrow><mo>)</mo></mrow></mrow></mrow><mo lspace="0em">.</mo></mrow></menclose></semantics></math></div>

This is the <em>length-normalised</em> sequence probability ratio. The GSPO loss clips this single scalar per sequence:

<div class="hh-equation" id="ch7.ex8"><math alttext="\mathcal{L}_{\text{GSPO}}=-\frac{1}{G}\sum_{i=1}^{G}\min\!\Bigl(s_{i}(\theta)\,\hat{A}_{i},\;\mathrm{clip}(s_{i}(\theta),1{-}\epsilon,1{+}\epsilon)\,\hat{A}_{i}\Bigr)." display="block"><semantics><mrow><msub><mi>ℒ</mi><mtext>GSPO</mtext></msub><mo rspace="0em">=</mo><mo lspace="0em">−</mo><mfrac><mn>1</mn><mi>G</mi></mfrac><munderover><mo movablelimits="false">∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>G</mi></munderover><mpadded width="1.653em"><mi>min</mi></mpadded><mrow><mo maxsize="1.600em" minsize="1.600em">(</mo><msub><mi>s</mi><mi>i</mi></msub><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo rspace="0.170em" stretchy="false">)</mo></mrow><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>i</mi></msub><mo rspace="0.447em">,</mo><mi>clip</mi><mrow><mo stretchy="false">(</mo><msub><mi>s</mi><mi>i</mi></msub><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow><mo>,</mo><mn>1</mn><mo>−</mo><mi>ϵ</mi><mo>,</mo><mn>1</mn><mo>+</mo><mi>ϵ</mi><mo rspace="0.170em" stretchy="false">)</mo></mrow><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>i</mi></msub><mo maxsize="1.600em" minsize="1.600em">)</mo></mrow><mo lspace="0em">.</mo></mrow></semantics></math></div>

### Dr. GRPO – Debiased Reward GRPO

Dr. GRPO modifies the per-token gradient weight to account for the token’s marginal contribution to the reward signal. Tokens that the model already assigns high probability to (regardless of the reward) are down-weighted:

<div class="hh-equation" id="ch7.ex9"><math alttext="w_{i,t}=\hat{A}_{i}\cdot\bigl(1-\pi_{\text{ref}}(o_{i,t}|q,o_{i,&lt;t})\bigr)," display="block"><semantics><mrow><mrow><msub><mi>w</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub><mo>=</mo><mrow><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>i</mi></msub><mo lspace="0.222em" rspace="0.222em">⋅</mo><mrow><mo maxsize="1.200em" minsize="1.200em">(</mo><mrow><mn>1</mn><mo>−</mo><mrow><msub><mi>π</mi><mtext>ref</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>o</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub><mo lspace="0em" rspace="0em">|</mo><mrow><mi>q</mi><mo>,</mo><msub><mi>o</mi><mrow><mrow><mi>i</mi><mo>,</mo><mphantom></mphantom></mrow><mo>&lt;</mo><mi>t</mi></mrow></msub></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow><mo maxsize="1.200em" minsize="1.200em">)</mo></mrow></mrow></mrow><mo>,</mo></mrow></semantics></math></div>

where <math alttext="\pi_{\text{ref}}" display="inline"><semantics><msub><mi>π</mi><mtext>ref</mtext></msub></semantics></math> is the reference (pretrained) model. This is a form of <em>token efficiency</em>: the gradient is concentrated on tokens where the policy genuinely needs to change.

### 2-GRPO – Minimal Two-Rollout GRPO

The key insight is that GRPO’s effectiveness does <em>not</em> primarily come from accurate advantage estimation (which requires large <math alttext="G" display="inline"><semantics><mi>G</mi></semantics></math>). Instead, it comes from an implicit <em>contrastive objective</em> that is structurally similar to DPO:

<div class="hh-equation" id="ch7.ex10"><math alttext="\mathcal{L}_{\text{2-GRPO}}\approx-\mathbb{E}_{(o^{+},o^{-})\sim\pi_{\theta}}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_{\theta}(o^{+}|q)}{\pi_{\text{old}}(o^{+}|q)}-\beta\log\frac{\pi_{\theta}(o^{-}|q)}{\pi_{\text{old}}(o^{-}|q)}\right)\right]," display="block"><semantics><mrow><mrow><msub><mi>ℒ</mi><mtext>2-GRPO</mtext></msub><mo>≈</mo><mrow><mo>−</mo><mrow><msub><mi>𝔼</mi><mrow><mrow><mo stretchy="false">(</mo><msup><mi>o</mi><mo>+</mo></msup><mo>,</mo><msup><mi>o</mi><mo>−</mo></msup><mo stretchy="false">)</mo></mrow><mo>∼</mo><msub><mi>π</mi><mi>θ</mi></msub></mrow></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo>[</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mrow><mpadded width="0.437em"><mi>σ</mi></mpadded><mo>⁡</mo><mrow><mo>(</mo><mrow><mrow><mi>β</mi><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mfrac><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msup><mi>o</mi><mo>+</mo></msup><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mi>π</mi><mtext>old</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msup><mi>o</mi><mo>+</mo></msup><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow></mfrac></mrow></mrow><mo>−</mo><mrow><mi>β</mi><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mfrac><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msup><mi>o</mi><mo>−</mo></msup><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mi>π</mi><mtext>old</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msup><mi>o</mi><mo>−</mo></msup><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow></mfrac></mrow></mrow></mrow><mo>)</mo></mrow></mrow></mrow><mo>]</mo></mrow></mrow></mrow></mrow><mo>,</mo></mrow></semantics></math></div>

where <math alttext="o^{+}" display="inline"><semantics><msup><mi>o</mi><mo>+</mo></msup></semantics></math> is the higher-reward completion and <math alttext="o^{-}" display="inline"><semantics><msup><mi>o</mi><mo>−</mo></msup></semantics></math> the lower-reward one. With <math alttext="G=2" display="inline"><semantics><mrow><mi>G</mi><mo>=</mo><mn>2</mn></mrow></semantics></math>, this contrastive structure is explicit. With <math alttext="G=16" display="inline"><semantics><mrow><mi>G</mi><mo>=</mo><mn>16</mn></mrow></semantics></math>, the same signal is present but diluted by redundant pairs.

### SAPO – Soft Adaptive Policy Optimization

SAPO replaces the <math alttext="\min(\rho A,\mathrm{clip}(\rho,\cdot)\,A)" display="inline"><semantics><mrow><mi>min</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mi>ρ</mi><mo lspace="0em" rspace="0em">​</mo><mi>A</mi></mrow><mo>,</mo><mrow><mrow><mi>clip</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>ρ</mi><mo rspace="0em">,</mo><mo lspace="0em" rspace="0em">⋅</mo><mo stretchy="false">)</mo></mrow></mrow><mo lspace="0.170em" rspace="0em">​</mo><mi>A</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></semantics></math> objective with a smooth surrogate:

<div class="hh-equation" id="ch7.ex11"><math alttext="\boxed{\mathcal{L}_{\text{SAPO}}(\rho,A)=\begin{cases}-A\cdot\sigma\!\left(\dfrac{\rho-1}{\tau_{+}}\right)\cdot\rho&amp;\text{if }A&gt;0\\[8.0pt]
-A\cdot\sigma\!\left(\dfrac{1-\rho}{\tau_{-}}\right)\cdot\rho&amp;\text{if }A\leq 0\end{cases}}" display="block"><semantics><menclose notation="box"><mrow><mrow><msub><mi>ℒ</mi><mtext>SAPO</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>ρ</mi><mo>,</mo><mi>A</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mo>{</mo><mtable columnspacing="5pt" displaystyle="true" rowspacing="0pt"><mtr><mtd columnalign="left"><mrow><mo>−</mo><mi>A</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><mpadded width="0.437em"><mi>σ</mi></mpadded><mo>(</mo><mfrac><mrow><mi>ρ</mi><mo>−</mo><mn>1</mn></mrow><msub><mi>τ</mi><mo>+</mo></msub></mfrac><mo rspace="0.055em">)</mo><mo rspace="0.222em">⋅</mo><mi>ρ</mi></mrow></mtd><mtd columnalign="left"><mrow><mrow><mtext>if </mtext><mo lspace="0em" rspace="0em">​</mo><mi>A</mi></mrow><mo>&gt;</mo><mn>0</mn></mrow></mtd></mtr><mtr><mtd columnalign="left"><mrow><mo>−</mo><mi>A</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><mpadded width="0.437em"><mi>σ</mi></mpadded><mo>(</mo><mfrac><mrow><mn>1</mn><mo>−</mo><mi>ρ</mi></mrow><msub><mi>τ</mi><mo>−</mo></msub></mfrac><mo rspace="0.055em">)</mo><mo rspace="0.222em">⋅</mo><mi>ρ</mi></mrow></mtd><mtd columnalign="left"><mrow><mrow><mtext>if </mtext><mo lspace="0em" rspace="0em">​</mo><mi>A</mi></mrow><mo>≤</mo><mn>0</mn></mrow></mtd></mtr></mtable></mrow></mrow></menclose></semantics></math></div>

where <math alttext="\sigma" display="inline"><semantics><mi>σ</mi></semantics></math> is the sigmoid function and <math alttext="\tau_{+},\tau_{-}" display="inline"><semantics><mrow><msub><mi>τ</mi><mo>+</mo></msub><mo>,</mo><msub><mi>τ</mi><mo>−</mo></msub></mrow></semantics></math> are asymmetric temperature parameters. A higher temperature produces a softer gate (more exploration); a lower temperature approaches hard clipping.

### TIS and MIS – Truncated and Masked Importance Sampling

TIS corrects the bias by multiplying the gradient by a truncated correction factor:

<div class="hh-equation" id="ch7.ex12"><math alttext="\boxed{w_{\text{TIS}}(o_{i})=\min\!\left(C,\;\frac{\pi_{\text{train}}(o_{i}|q)}{\pi_{\text{vllm}}(o_{i}|q)}\right),}" display="block"><semantics><menclose notation="box"><mrow><mrow><mrow><msub><mi>w</mi><mtext>TIS</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>o</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mpadded width="1.653em"><mi>min</mi></mpadded><mo>⁡</mo><mrow><mo>(</mo><mi>C</mi><mo rspace="0.447em">,</mo><mfrac><mrow><msub><mi>π</mi><mtext>train</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>o</mi><mi>i</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mi>π</mi><mtext>vllm</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>o</mi><mi>i</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow></mfrac><mo>)</mo></mrow></mrow></mrow><mo>,</mo></mrow></menclose></semantics></math></div>

where <math alttext="\pi_{\text{train}}" display="inline"><semantics><msub><mi>π</mi><mtext>train</mtext></msub></semantics></math> is the probability from the training forward pass and <math alttext="\pi_{\text{vllm}}" display="inline"><semantics><msub><mi>π</mi><mtext>vllm</mtext></msub></semantics></math> is the probability reported by vLLM. The truncation at <math alttext="C" display="inline"><semantics><mi>C</mi></semantics></math> prevents extreme corrections from destabilising training.

MIS takes a harder approach: it zeros out the gradient for any sequence where the correction ratio exceeds a threshold <math alttext="C" display="inline"><semantics><mi>C</mi></semantics></math>:

<div class="hh-equation" id="ch7.ex13"><math alttext="w_{\text{MIS}}(o_{i})=\mathbf{1}\!\left[\frac{\pi_{\text{train}}(o_{i}|q)}{\pi_{\text{vllm}}(o_{i}|q)}\leq C\right]." display="block"><semantics><mrow><msub><mi>w</mi><mtext>MIS</mtext></msub><mrow><mo stretchy="false">(</mo><msub><mi>o</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow><mo rspace="0.108em">=</mo><mrow><mo>[</mo><mfrac><mrow><msub><mi>π</mi><mtext>train</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>o</mi><mi>i</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mi>π</mi><mtext>vllm</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>o</mi><mi>i</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow></mfrac><mo>≤</mo><mi>C</mi><mo>]</mo></mrow><mo lspace="0em">.</mo></mrow></semantics></math></div>

This is more conservative but avoids the risk of large (even truncated) correction weights.

Both TIS and MIS can be applied at the token level or the sequence level:

<ul><li>Sequence-level: compute the ratio as the geometric mean over all tokens (as in GSPO). Theoretically correct but higher variance.</li><li>Token-level: compute a separate ratio for each token. Biased (the product of per-token corrections is not the sequence correction) but lower variance.</li></ul>

### VESPO – Variational Sequence-Level Soft Policy Optimization

VESPO derives a weighting function <math alttext="W(\tau)" display="inline"><semantics><mrow><mi>W</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>τ</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math> for each trajectory <math alttext="\tau" display="inline"><semantics><mi>τ</mi></semantics></math> from the variational objective. The final gradient weight takes the form:

<div class="hh-equation" id="ch7.ex14"><math alttext="\boxed{g(\tau)=W(\tau)^{k}\cdot\exp\!\bigl(\lambda(1-W(\tau))\bigr),}" display="block"><semantics><menclose notation="box"><mrow><mrow><mrow><mi>g</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>τ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mrow><mi>W</mi><mo lspace="0em" rspace="0em">​</mo><msup><mrow><mo stretchy="false">(</mo><mi>τ</mi><mo rspace="0.055em" stretchy="false">)</mo></mrow><mi>k</mi></msup></mrow><mo rspace="0.222em">⋅</mo><mrow><mpadded width="1.370em"><mi>exp</mi></mpadded><mo>⁡</mo><mrow><mo maxsize="1.200em" minsize="1.200em">(</mo><mrow><mi>λ</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mn>1</mn><mo>−</mo><mrow><mi>W</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>τ</mi><mo stretchy="false">)</mo></mrow></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow><mo maxsize="1.200em" minsize="1.200em">)</mo></mrow></mrow></mrow></mrow><mo>,</mo></mrow></menclose></semantics></math></div>

where <math alttext="W(\tau)=\pi_{\theta}(\tau)/\pi_{\text{old}}(\tau)" display="inline"><semantics><mrow><mrow><mi>W</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>τ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mrow><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>τ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>/</mo><msub><mi>π</mi><mtext>old</mtext></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>τ</mi><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math> is the sequence-level importance weight, <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> controls the sharpness of the weighting, and <math alttext="\lambda" display="inline"><semantics><mi>λ</mi></semantics></math> controls the exponential decay for stale (low-weight) trajectories. This kernel:

<ul><li>Is smooth everywhere (no discontinuous gradient at clip boundaries).</li><li>Naturally down-weights stale trajectories (<math alttext="W\ll 1" display="inline"><semantics><mrow><mi>W</mi><mo>≪</mo><mn>1</mn></mrow></semantics></math>) via the exponential term.</li><li>Is asymmetric: high-weight trajectories (<math alttext="W&gt;1" display="inline"><semantics><mrow><mi>W</mi><mo>&gt;</mo><mn>1</mn></mrow></semantics></math>) are treated differently from low-weight ones.</li></ul>

### DPPO – Direct Policy Divergence Policy Optimization

DPPO computes the trust region constraint directly using either Total Variation (TV) or KL divergence between the old and new policy distributions:

<div class="hh-equation" id="ch7.ex15"><math alttext="\mathcal{L}_{\text{DPPO}}=-\mathbb{E}\!\left[\hat{A}\cdot\pi_{\theta}(o|q)\cdot\mathbf{1}[D(\pi_{\theta}\|\pi_{\text{old}})\leq\delta]\right]," display="block"><semantics><mrow><msub><mi>ℒ</mi><mtext>DPPO</mtext></msub><mo rspace="0em">=</mo><mo lspace="0em">−</mo><mpadded width="0.497em"><mi>𝔼</mi></mpadded><mrow><mo>[</mo><mover accent="true"><mi>A</mi><mo>^</mo></mover><mo lspace="0.222em" rspace="0.222em">⋅</mo><msub><mi>π</mi><mi>θ</mi></msub><mrow><mo stretchy="false">(</mo><mi>o</mi><mo fence="false" rspace="0.167em" stretchy="false">|</mo><mi>q</mi><mo rspace="0.055em" stretchy="false">)</mo></mrow><mo rspace="0.222em">⋅</mo><mn>𝟏</mn><mrow><mo stretchy="false">[</mo><mi>D</mi><mrow><mo stretchy="false">(</mo><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0.167em">∥</mo><msub><mi>π</mi><mtext>old</mtext></msub><mo stretchy="false">)</mo></mrow><mo>≤</mo><mi>δ</mi><mo stretchy="false">]</mo></mrow><mo>]</mo></mrow><mo>,</mo></mrow></semantics></math></div>

where <math alttext="D" display="inline"><semantics><mi>D</mi></semantics></math> is the chosen divergence measure. In practice, DPPO approximates this with token-level binary or top-<math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> masks:

<ul><li>binary_tv: mask tokens where <math alttext="|\pi_{\theta}-\pi_{\text{old}}|&gt;\delta" display="inline"><semantics><mrow><mrow><mo stretchy="false">|</mo><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo>−</mo><msub><mi>π</mi><mtext>old</mtext></msub></mrow><mo stretchy="false">|</mo></mrow><mo>&gt;</mo><mi>δ</mi></mrow></semantics></math>.</li><li>binary_kl: mask tokens where <math alttext="\pi_{\theta}\log(\pi_{\theta}/\pi_{\text{old}})&gt;\delta" display="inline"><semantics><mrow><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo>/</mo><msub><mi>π</mi><mtext>old</mtext></msub></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow><mo>&gt;</mo><mi>δ</mi></mrow></semantics></math>.</li><li>topk_tv: keep only the top-<math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> tokens by TV contribution.</li><li>topk_kl: keep only the top-<math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> tokens by KL contribution.</li></ul>

### ScaleRL and CISPO

Standard GRPO normalises rewards within a group of <math alttext="G" display="inline"><semantics><mi>G</mi></semantics></math> completions for a single prompt. CISPO normalises rewards across the <em>entire batch</em>:

<div class="hh-equation" id="ch7.ex16"><math alttext="\hat{A}_{i}=\frac{r_{i}-\mu_{\text{batch}}}{\sigma_{\text{batch}}+\epsilon}," display="block"><semantics><mrow><mrow><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>i</mi></msub><mo>=</mo><mfrac><mrow><msub><mi>r</mi><mi>i</mi></msub><mo>−</mo><msub><mi>μ</mi><mtext>batch</mtext></msub></mrow><mrow><msub><mi>σ</mi><mtext>batch</mtext></msub><mo>+</mo><mi>ϵ</mi></mrow></mfrac></mrow><mo>,</mo></mrow></semantics></math></div>

where <math alttext="\mu_{\text{batch}}" display="inline"><semantics><msub><mi>μ</mi><mtext>batch</mtext></msub></semantics></math> and <math alttext="\sigma_{\text{batch}}" display="inline"><semantics><msub><mi>σ</mi><mtext>batch</mtext></msub></semantics></math> are computed over all rewards in the current training batch. This provides a more stable baseline and prevents any single prompt from dominating the gradient.

CISPO combines batch-level scaling with DAPO’s token-level loss aggregation and asymmetric clipping:

<div class="hh-equation" id="ch7.ex17"><math alttext="\mathcal{L}_{\text{CISPO}}=-\frac{1}{\sum_{i,t}m_{i,t}}\sum_{i=1}^{G}\sum_{t=1}^{|o_{i}|}m_{i,t}\cdot\min\!\bigl(\rho_{i,t}\hat{A}_{i},\;\mathrm{clip}_{\text{DAPO}}(\rho_{i,t},\hat{A}_{i})\,\hat{A}_{i}\bigr)," display="block"><semantics><mrow><msub><mi>ℒ</mi><mtext>CISPO</mtext></msub><mo rspace="0em">=</mo><mo lspace="0em">−</mo><mfrac><mn>1</mn><mrow><msub><mo>∑</mo><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub><msub><mi>m</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub></mrow></mfrac><munderover><mo movablelimits="false" rspace="0em">∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>G</mi></munderover><munderover><mo movablelimits="false">∑</mo><mrow><mi>t</mi><mo>=</mo><mn>1</mn></mrow><mrow><mo stretchy="false">|</mo><msub><mi>o</mi><mi>i</mi></msub><mo stretchy="false">|</mo></mrow></munderover><msub><mi>m</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub><mo lspace="0.222em" rspace="0.222em">⋅</mo><mpadded width="1.653em"><mi>min</mi></mpadded><mrow><mo maxsize="1.200em" minsize="1.200em">(</mo><msub><mi>ρ</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>i</mi></msub><mo rspace="0.447em">,</mo><msub><mi>clip</mi><mtext>DAPO</mtext></msub><mrow><mo stretchy="false">(</mo><msub><mi>ρ</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub><mo>,</mo><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>i</mi></msub><mo rspace="0.170em" stretchy="false">)</mo></mrow><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>i</mi></msub><mo maxsize="1.200em" minsize="1.200em">)</mo></mrow><mo>,</mo></mrow></semantics></math></div>

where <math alttext="m_{i,t}" display="inline"><semantics><msub><mi>m</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub></semantics></math> is the overlong-filtering mask.

### GDPO – Group Reward-Decoupled Policy Optimization

The core mechanism normalises each reward <em>independently</em> before aggregating:

<div class="hh-equation" id="ch7.ex18"><math alttext="\boxed{\hat{A}_{n}^{(i)}=\frac{r_{n}^{(i)}-\mu_{n}}{\sigma_{n}+\epsilon},\qquad\hat{A}^{(i)}=\sum_{n=1}^{N}w_{n}\hat{A}_{n}^{(i)},}" display="block"><semantics><menclose notation="box"><mrow><mover accent="true"><mi>A</mi><mo>^</mo></mover><msub><mrow></mrow><mi>n</mi></msub><msup><mrow></mrow><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>=</mo><mfrac><mrow><msubsup><mi>r</mi><mi>n</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msubsup><mo>−</mo><msub><mi>μ</mi><mi>n</mi></msub></mrow><mrow><msub><mi>σ</mi><mi>n</mi></msub><mo>+</mo><mi>ϵ</mi></mrow></mfrac><mo rspace="2.167em">,</mo><mover accent="true"><mi>A</mi><mo>^</mo></mover><msup><mrow></mrow><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo rspace="0.111em">=</mo><mo movablelimits="false">∑</mo><msub><mrow></mrow><mrow><mi>n</mi><mo>=</mo><mn>1</mn></mrow></msub><msup><mrow></mrow><mi>N</mi></msup><mi>w</mi><msub><mrow></mrow><mi>n</mi></msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><msub><mrow></mrow><mi>n</mi></msub><msup><mrow></mrow><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>,</mo></mrow></menclose></semantics></math></div>

where <math alttext="r_{n}^{(i)}" display="inline"><semantics><msubsup><mi>r</mi><mi>n</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msubsup></semantics></math> is the <math alttext="n" display="inline"><semantics><mi>n</mi></semantics></math>-th reward for completion <math alttext="i" display="inline"><semantics><mi>i</mi></semantics></math>, <math alttext="\mu_{n}" display="inline"><semantics><msub><mi>μ</mi><mi>n</mi></msub></semantics></math> and <math alttext="\sigma_{n}" display="inline"><semantics><msub><mi>σ</mi><mi>n</mi></msub></semantics></math> are the mean and standard deviation of reward <math alttext="n" display="inline"><semantics><mi>n</mi></semantics></math> within the group, and <math alttext="w_{n}" display="inline"><semantics><msub><mi>w</mi><mi>n</mi></msub></semantics></math> are user-specified weights.

### GOPO – Group Ordinal Policy Optimization

GOPO [54] starts from a simple observation: reward models are trained with pairwise comparisons (“is A better than B?”), so only the rank order of their outputs is trustworthy—the raw numeric scores carry no inherent meaning. Yet GRPO feeds those raw magnitudes directly into the advantage calculation. For tasks with non-verifiable rewards—summarization, open-ended chat, instruction following—this mismatch introduces noise, because a gap of 0.6 reward points might reflect genuine quality in one region of the output space and mean nothing in another.

Key Insight: Discard reward magnitudes entirely. Use only the ordinal ranking of rewards within a group.

Algorithm: Given a group of <math alttext="N" display="inline"><semantics><mi>N</mi></semantics></math> responses <math alttext="\{o_{1},\ldots,o_{N}\}" display="inline"><semantics><mrow><mo stretchy="false">{</mo><msub><mi>o</mi><mn>1</mn></msub><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msub><mi>o</mi><mi>N</mi></msub><mo stretchy="false">}</mo></mrow></semantics></math> with rewards <math alttext="\{r_{1},\ldots,r_{N}\}" display="inline"><semantics><mrow><mo stretchy="false">{</mo><msub><mi>r</mi><mn>1</mn></msub><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msub><mi>r</mi><mi>N</mi></msub><mo stretchy="false">}</mo></mrow></semantics></math>:

<ol><li>Rank responses by reward: assign rank <math alttext="\text{rank}(o_{i})\in\{1,\ldots,N\}" display="inline"><semantics><mrow><mrow><mtext>rank</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>o</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>∈</mo><mrow><mo stretchy="false">{</mo><mn>1</mn><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><mi>N</mi><mo stretchy="false">}</mo></mrow></mrow></semantics></math> (1 = worst, <math alttext="N" display="inline"><semantics><mi>N</mi></semantics></math> = best).</li><li>Replace raw advantages with rank-based scores:<math alttext="\boxed{\hat{A}_{i}^{\text{GOPO}}=f\!\left(\frac{\text{rank}(o_{i})}{N}\right)}" display="block"><semantics><menclose notation="box"><mrow><msubsup><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>i</mi><mtext>GOPO</mtext></msubsup><mo>=</mo><mrow><mpadded width="0.427em"><mi>f</mi></mpadded><mo>⁡</mo><mrow><mo>(</mo><mfrac><mrow><mtext>rank</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>o</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mi>N</mi></mfrac><mo>)</mo></mrow></mrow></mrow></menclose></semantics></math><span class="hh-tag">(7.2)</span>where <math alttext="f" display="inline"><semantics><mi>f</mi></semantics></math> is a monotonic transformation (e.g., linear mapping to <math alttext="[-1,1]" display="inline"><semantics><mrow><mo stretchy="false">[</mo><mrow><mrow><mo>−</mo><mn>1</mn></mrow><mo>,</mo><mn>1</mn></mrow><mo stretchy="false">]</mo></mrow></semantics></math> or quantile normalization).</li><li>Apply PPO-style clipped objective using rank-based advantages.</li></ol>

Comparison with GRPO:

AspectGRPOGOPOAdvantage signal<math alttext="\hat{A}_{i}=(r_{i}-\mu)/\sigma" display="inline"><semantics><mrow><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>i</mi></msub><mo>=</mo><mrow><mrow><mo stretchy="false">(</mo><mrow><msub><mi>r</mi><mi>i</mi></msub><mo>−</mo><mi>μ</mi></mrow><mo stretchy="false">)</mo></mrow><mo>/</mo><mi>σ</mi></mrow></mrow></semantics></math> (uses magnitudes)<math alttext="\hat{A}_{i}=f(\text{rank}_{i}/N)" display="inline"><semantics><mrow><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>i</mi></msub><mo>=</mo><mrow><mi>f</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><msub><mtext>rank</mtext><mi>i</mi></msub><mo>/</mo><mi>N</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math> (uses ordinal rank only)Sensitivity to reward scaleHigh — miscalibrated RM scores distort advantagesNone — invariant to monotonic reward transformationsBest forVerifiable rewards (binary, well-calibrated)Non-verifiable rewards (RM-based, noisy magnitudes)

Empirical gains (over GRPO on non-verifiable tasks):

<ul><li>Reward curves (both training and held-out) sit above GRPO throughout optimization</li><li>Win-rates judged by a separate LLM evaluator improve at most training checkpoints</li><li>Convergence is markedly faster—matching GRPO’s final quality with fewer gradient steps</li><li>The advantage grows as the reward model becomes noisier or more poorly calibrated</li></ul>

<aside class="hh-next">
<p class="hh-next-label">下一篇</p>
<p class="hh-next-title">Chapter 8 Preference Optimization Variants</p>
<p class="hh-next-hint">在「智能体 AI 漫游指南」分类下继续阅读。</p>
</aside>

</div>
