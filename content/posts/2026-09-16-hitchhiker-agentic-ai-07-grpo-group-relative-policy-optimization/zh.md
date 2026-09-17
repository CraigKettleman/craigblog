---
title: "智能体 AI 漫游指南 · 第 7 章 GRPO ——组相对策略优化"
slug: "hitchhiker-agentic-ai-07-grpo-group-relative-policy-optimization"
lang: "zh"
date: "2026-09-16T00:08:00.000Z"
draft: false
featured: false
categories:
  - "hitchhiker-agentic-ai"
cover: "./cover.png"
description: "组相对策略优化（Group Relative Policy Optimization, GRPO）[323] 是一种专为语言模型设计的强化学习算法，它消除了对独立评论家（critic）网络的需求。…"
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

组相对策略优化（Group Relative Policy Optimization, GRPO）[323] 是一种专为语言模型设计的强化学习算法，它消除了对独立评论家（critic）网络的需求。GRPO 由 DeepSeek 在 DeepSeekMath 工作中提出，随后在 DeepSeek-R1 [69] 中被扩展到更大规模，已经迅速成为 LLM 训练中的主流 RL 方法——被大多数开源对齐框架（TRL、OpenRLHF、veRL）作为默认算法采用。

其核心思想看似简单却出人意料地有效：与其训练一个神经网络去预测期望奖励（reward）——即 PPO 中的评论家（critic）——GRPO 通过对同一 prompt 生成多条响应、利用该组的奖励统计量作为 baseline，从经验上<em>估计</em>这一基线。这从显存中去掉了一整个模型，将工程复杂度减半，而且——令人意外的是——往往优于 PPO，因为经验基线比训练不充分的价值函数（value function）更准确。

GRPO 在下列场景中尤其有效：

<ul><li>推理任务：具有可验证的奖励（reward）（数学、代码），二值正确性提供干净信号。</li><li>大模型（70B 及以上）：去掉评论家带来的显存节省至关重要。</li><li>多轮对话（Multi-Turn）与 Agent 场景：跨工具调用（Tool Calling）的价值估计本就难以处理。</li></ul>

本章涵盖 GRPO 的动机、算法、关键变体（Dr. GRPO、DAPO、2-GRPO、GDPO）以及基于 TRL 的实战实现。

## 动机

PPO 的价值模型（评论家）在语言任务上存在三大问题：

<ol><li>显存：价值头（value head）与策略共享主干（70B 模型占 140GB）。若分离则显存翻倍。</li><li>精度：对部分序列预测期望奖励极其困难。价值函数经常出错 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> advantage 出错 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 梯度方向出错。</li><li>训练：价值头需要大量样本才能收敛。RL 早期它给出嘈杂的预测，破坏策略学习的稳定性。</li></ol>

GRPO 的关键洞见[323]：与其学习 <math alttext="V(s)" display="inline"><semantics><mrow><mi>V</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>s</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math>，不如从一组样本中经验地<em>估计</em>它。对同一 prompt 生成 <math alttext="G" display="inline"><semantics><mi>G</mi></semantics></math> 条响应，计算它们的奖励，并用组统计量作为 baseline。

## 算法

<ol><li>对每个 prompt <math alttext="x" display="inline"><semantics><mi>x</mi></semantics></math>，采样 <math alttext="G" display="inline"><semantics><mi>G</mi></semantics></math> 条补全（completion）：<math alttext="\{y_{1},\ldots,y_{G}\}\sim\pi_{\theta}(\cdot|x)" display="inline"><semantics><mrow><mrow><mo stretchy="false">{</mo><msub><mi>y</mi><mn>1</mn></msub><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msub><mi>y</mi><mi>G</mi></msub><mo stretchy="false">}</mo></mrow><mo>∼</mo><msub><mi>π</mi><mi>θ</mi></msub><mrow><mo stretchy="false">(</mo><mo lspace="0em" rspace="0em">⋅</mo><mo fence="false" rspace="0.167em" stretchy="false">|</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math></li><li>对每条打分：<math alttext="r_{i}=R(x,y_{i})" display="inline"><semantics><mrow><msub><mi>r</mi><mi>i</mi></msub><mo>=</mo><mrow><mi>R</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>,</mo><msub><mi>y</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math></li><li>组内归一化：<math alttext="\hat{A}_{i}=\frac{r_{i}-\mu_{G}}{\sigma_{G}}" display="inline"><semantics><mrow><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>i</mi></msub><mo>=</mo><mfrac><mrow><msub><mi>r</mi><mi>i</mi></msub><mo>−</mo><msub><mi>μ</mi><mi>G</mi></msub></mrow><msub><mi>σ</mi><mi>G</mi></msub></mfrac></mrow></semantics></math>，其中 <math alttext="\mu_{G}=\frac{1}{G}\sum_{j}r_{j}" display="inline"><semantics><mrow><msub><mi>μ</mi><mi>G</mi></msub><mo>=</mo><mrow><mfrac><mn>1</mn><mi>G</mi></mfrac><mo lspace="0em" rspace="0em">​</mo><mrow><msub><mo>∑</mo><mi>j</mi></msub><msub><mi>r</mi><mi>j</mi></msub></mrow></mrow></mrow></semantics></math>，<math alttext="\sigma_{G}=\text{std}(\{r_{j}\})" display="inline"><semantics><mrow><msub><mi>σ</mi><mi>G</mi></msub><mo>=</mo><mrow><mtext>std</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mrow><mo stretchy="false">{</mo><msub><mi>r</mi><mi>j</mi></msub><mo stretchy="false">}</mo></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math></li><li>用这些 advantage 应用 PPO 风格的 clipped 更新</li></ol>

<div class="hh-equation" id="ch7.e1"><math alttext="\boxed{\hat{A}_{i}=\frac{r_{i}-\mu_{G}}{\sigma_{G}},\qquad L=\mathbb{E}\left[\min\left(r_{t}(\theta)\hat{A}_{i},\;\text{clip}(r_{t}(\theta),1{\pm}\epsilon)\hat{A}_{i}\right)\right]-\beta D_{\text{KL}}[\pi_{\theta}\|\pi_{\text{ref}}]}" display="block"><semantics><menclose notation="box"><mrow><mover accent="true"><mi>A</mi><mo>^</mo></mover><msub><mrow></mrow><mi>i</mi></msub><mo>=</mo><mfrac><mrow><msub><mi>r</mi><mi>i</mi></msub><mo>−</mo><msub><mi>μ</mi><mi>G</mi></msub></mrow><msub><mi>σ</mi><mi>G</mi></msub></mfrac><mo rspace="2.167em">,</mo><mi>L</mi><mo>=</mo><mi>𝔼</mi><mo>[</mo><mi>min</mi><mo>(</mo><mi>r</mi><msub><mrow></mrow><mi>t</mi></msub><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo><mover accent="true"><mi>A</mi><mo>^</mo></mover><msub><mrow></mrow><mi>i</mi></msub><mo rspace="0.447em">,</mo><mtext>clip</mtext><mo stretchy="false">(</mo><mi>r</mi><msub><mrow></mrow><mi>t</mi></msub><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo><mo>,</mo><mn>1</mn><mo>±</mo><mi>ϵ</mi><mo stretchy="false">)</mo><mover accent="true"><mi>A</mi><mo>^</mo></mover><msub><mrow></mrow><mi>i</mi></msub><mo>)</mo><mo>]</mo><mo>−</mo><mi>β</mi><mi>D</mi><msub><mrow></mrow><mtext>KL</mtext></msub><mo stretchy="false">[</mo><mi>π</mi><msub><mrow></mrow><mi>θ</mi></msub><mo lspace="0em" rspace="0.167em">∥</mo><mi>π</mi><msub><mrow></mrow><mtext>ref</mtext></msub><mo stretchy="false">]</mo></mrow></menclose></semantics></math><span class="hh-equation-number">(7.1)</span></div>

<figure id="ch7.f1"><img src="./fig_029_fig29.png" alt="图 7.1：GRPO 实战：对一个数学 prompt 采样 G=5G{=}5 条响应。三条正确（r=1r{=}1），两条错误（r=0r{=}0）。组均值 μG=0.6\mu_{G}{=}0.6 充当 baseline；正确响应获得正 adv" loading="lazy" decoding="async"><figcaption><span class="hh-tag">图 7.1：</span>GRPO 实战：对一个数学 prompt 采样 <math alttext="G{=}5" display="inline"><semantics><mrow><mi>G</mi><mo>=</mo><mn>5</mn></mrow></semantics></math> 条响应。三条正确（<math alttext="r{=}1" display="inline"><semantics><mrow><mi>r</mi><mo>=</mo><mn>1</mn></mrow></semantics></math>），两条错误（<math alttext="r{=}0" display="inline"><semantics><mrow><mi>r</mi><mo>=</mo><mn>0</mn></mrow></semantics></math>）。组均值 <math alttext="\mu_{G}{=}0.6" display="inline"><semantics><mrow><msub><mi>μ</mi><mi>G</mi></msub><mo>=</mo><mn>0.6</mn></mrow></semantics></math> 充当 baseline；正确响应获得正 advantage（强化），错误响应获得负 advantage（抑制）。</figcaption></figure>

## TRL 实现

下面给出一个使用 HuggingFace TRL 的最小可运行示例。

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

## 组大小分析

<math alttext="G" display="inline"><semantics><mi>G</mi></semantics></math>信号质量计算开销何时使用2非常嘈杂（如同抛硬币）低不建议使用——过于嘈杂，难以稳定学习4中等中等快速实验、简单任务（通过率 <math alttext="&gt;" display="inline"><semantics><mo>&gt;</mo></semantics></math> 50%）8良好（标准）高默认选择。对大多数任务而言平衡性良好16优秀很高困难任务（通过率 <math alttext="&lt;" display="inline"><semantics><mo>&lt;</mo></semantics></math> 20%），需要多次尝试才能得到正样本32近乎完美极高仅当你拥有海量算力且任务极难时才使用

## GRPO 变体与扩展

### GRPO 组中的多样性

<div class="hh-table" id="ch7.t1"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>方法</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>它如何促进多样性</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">熵奖励</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>在奖励中加入 <math alttext="\alpha H(\pi_{\theta})" display="inline"><semantics><mrow><mi>α</mi><mo lspace="0em" rspace="0em">​</mo><mi>H</mi><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>π</mi><mi>θ</mi></msub><mo stretchy="false">)</mo></mrow></mrow><span></span></semantics></math>。直接惩罚低熵（确定性）的策略。</span></span></td></tr><tr><th class="ltx_align_left">KL 惩罚</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="-\beta D_{\text{KL}}[\pi_{\theta}\|\pi_{\text{ref}}]" display="inline"><semantics><mrow><mo>−</mo><mi>β</mi><msub><mi>D</mi><mtext>KL</mtext></msub><mrow><mo stretchy="false">[</mo><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0.167em">∥</mo><msub><mi>π</mi><mtext>ref</mtext></msub><mo stretchy="false">]</mo></mrow></mrow><span></span></semantics></math> 防止坍塌到单一模式。</span></span></td></tr><tr><th class="ltx_align_left">拒绝采样</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>生成大量候选，按奖励保留前 <math alttext="k" display="inline"><semantics><mi>k</mi><span></span></semantics></math> 个。自然会筛选出多样且高质量的响应。</span></span></td></tr><tr><th class="ltx_align_left">Best-of-N</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>在推理时：生成 <math alttext="N" display="inline"><semantics><mi>N</mi><span></span></semantics></math> 条响应，为全部打分，返回最优者。多样性来自采样。</span></span></td></tr><tr><th class="ltx_align_left">使用多样化配对的直接偏好优化（DPO）</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>在 chosen/rejected 于<em>方法</em>上有别（而非仅在质量上有别）的配对上训练。</span></span></td></tr><tr><th class="ltx_align_left">多奖励</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>使用多个奖励模型（安全性、有用性、代码质量）。防止坍塌到单一维度。</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">表 7.1：</span>RL 训练中促进多样性的方法。</p></div>

后训练对齐（RLHF、DPO）常常因<em>典型性偏差</em>而降低输出多样性：人类标注者系统性地偏好熟悉、「典型」的文本，而非新颖的替代方案。这种模式坍塌是数据层面的现象，并不纯粹是算法问题。

言语化采样（Verbalized Sampling, VS）[422] 是一种无需训练的 prompting 策略，它要求模型在单次生成中显式说出一个概率分布，从而绕开这种坍塌。

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

在深入各项扩展之前，先简要回顾前文建立的基础 GRPO 算法。其核心机制——采样一组补全、对其奖励做归一化、再施加 clipped 的策略梯度——十分简洁而优雅。然而，实践者很快发现了若干具体的失效模式：预训练偏差稀释梯度（Dr. GRPO）、对称裁剪限制探索（DAPO）、组规模过大造成浪费（2-GRPO），以及多目标场景下的奖励尺度失衡（GDPO）。以下各节将逐一讨论这些问题。

### DAPO ——Dynamic Adaptive Policy Optimization

标准 PPO/GRPO 在 <math alttext="[1-\epsilon,1+\epsilon]" display="inline"><semantics><mrow><mo stretchy="false">[</mo><mrow><mrow><mn>1</mn><mo>−</mo><mi>ϵ</mi></mrow><mo>,</mo><mrow><mn>1</mn><mo>+</mo><mi>ϵ</mi></mrow></mrow><mo stretchy="false">]</mo></mrow></semantics></math> 处对重要性比值（importance ratio）做对称裁剪。DAPO 将其替换为非对称的区间：

<div class="hh-equation" id="ch7.ex3"><math alttext="\boxed{\mathrm{clip}_{\text{DAPO}}(\rho,A)=\begin{cases}\mathrm{clip}(\rho,\,1-\epsilon,\,1+\epsilon_{\text{high}})&amp;\text{if }A&gt;0\\
\mathrm{clip}(\rho,\,1-\epsilon,\,1+\epsilon)&amp;\text{if }A\leq 0\end{cases}}" display="block"><semantics><menclose notation="box"><mrow><mrow><msub><mi>clip</mi><mtext>DAPO</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>ρ</mi><mo>,</mo><mi>A</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mo>{</mo><mtable columnspacing="5pt" displaystyle="true" rowspacing="0pt"><mtr><mtd columnalign="left"><mrow><mi>clip</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>ρ</mi><mo>,</mo><mrow><mn> 1</mn><mo>−</mo><mi>ϵ</mi></mrow><mo>,</mo><mrow><mn> 1</mn><mo>+</mo><msub><mi>ϵ</mi><mtext>high</mtext></msub></mrow><mo stretchy="false">)</mo></mrow></mrow></mtd><mtd columnalign="left"><mrow><mrow><mtext>if </mtext><mo lspace="0em" rspace="0em">​</mo><mi>A</mi></mrow><mo>&gt;</mo><mn>0</mn></mrow></mtd></mtr><mtr><mtd columnalign="left"><mrow><mi>clip</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>ρ</mi><mo>,</mo><mrow><mn> 1</mn><mo>−</mo><mi>ϵ</mi></mrow><mo>,</mo><mrow><mn> 1</mn><mo>+</mo><mi>ϵ</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></mtd><mtd columnalign="left"><mrow><mrow><mtext>if </mtext><mo lspace="0em" rspace="0em">​</mo><mi>A</mi></mrow><mo>≤</mo><mn>0</mn></mrow></mtd></mtr></mtable></mrow></mrow></menclose></semantics></math></div>

其中 <math alttext="\epsilon_{\text{high}}&gt;\epsilon" display="inline"><semantics><mrow><msub><mi>ϵ</mi><mtext>high</mtext></msub><mo>&gt;</mo><mi>ϵ</mi></mrow></semantics></math>（典型取值：<math alttext="\epsilon=0.2" display="inline"><semantics><mrow><mi>ϵ</mi><mo>=</mo><mn>0.2</mn></mrow></semantics></math>、<math alttext="\epsilon_{\text{high}}=0.28" display="inline"><semantics><mrow><msub><mi>ϵ</mi><mtext>high</mtext></msub><mo>=</mo><mn>0.28</mn></mrow></semantics></math>）。当 advantage 为正时，策略被允许更进一步朝好的 token 移动；当 advantage 为负时，则适用通常的保守裁剪，以避免过度抑制。

基础 GRPO 将损失除以<em>序列数量</em><math alttext="G" display="inline"><semantics><mi>G</mi></semantics></math>。DAPO 则除以所有序列的<em>token 总数</em>：

<div class="hh-equation" id="ch7.ex4"><math alttext="\mathcal{L}_{\text{token}}=-\frac{1}{\sum_{i=1}^{G}|o_{i}|}\sum_{i=1}^{G}\sum_{t=1}^{|o_{i}|}\min\!\bigl(\rho_{i,t}\hat{A}_{i},\;\mathrm{clip}_{\text{DAPO}}(\rho_{i,t},\hat{A}_{i})\,\hat{A}_{i}\bigr)." display="block"><semantics><mrow><msub><mi>ℒ</mi><mtext>token</mtext></msub><mo rspace="0em">=</mo><mo lspace="0em">−</mo><mfrac><mn>1</mn><mrow><msubsup><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>G</mi></msubsup><mrow><mo lspace="0em" stretchy="false">|</mo><msub><mi>o</mi><mi>i</mi></msub><mo stretchy="false">|</mo></mrow></mrow></mfrac><munderover><mo movablelimits="false" rspace="0em">∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>G</mi></munderover><munderover><mo movablelimits="false">∑</mo><mrow><mi>t</mi><mo>=</mo><mn>1</mn></mrow><mrow><mo stretchy="false">|</mo><msub><mi>o</mi><mi>i</mi></msub><mo stretchy="false">|</mo></mrow></munderover><mpadded width="1.653em"><mi>min</mi></mpadded><mrow><mo maxsize="1.200em" minsize="1.200em">(</mo><msub><mi>ρ</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>i</mi></msub><mo rspace="0.447em">,</mo><msub><mi>clip</mi><mtext>DAPO</mtext></msub><mrow><mo stretchy="false">(</mo><msub><mi>ρ</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub><mo>,</mo><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>i</mi></msub><mo rspace="0.170em" stretchy="false">)</mo></mrow><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>i</mi></msub><mo maxsize="1.200em" minsize="1.200em">)</mo></mrow><mo lspace="0em">.</mo></mrow></semantics></math></div>

这样可以防止长补全仅仅因为包含更多 token 就在梯度信号中占据主导。

当一条补全被截断（在最大长度预算内没有出现 EOS token）时，它提供的是<em>误导性</em>信号：模型会因为那些生成正确、却恰好位于截断边界之前的 token 而受到惩罚。DAPO 会将这些补全整体屏蔽掉：

<div class="hh-equation" id="ch7.ex5"><math alttext="m_{i}=\mathbf{1}[\text{EOS}\in o_{i}],\qquad\mathcal{L}_{\text{filtered}}=-\frac{\sum_{i=1}^{G}m_{i}\sum_{t}(\cdots)}{\sum_{i=1}^{G}m_{i}|o_{i}|}." display="block"><semantics><mrow><msub><mi>m</mi><mi>i</mi></msub><mo>=</mo><mn>𝟏</mn><mrow><mo stretchy="false">[</mo><mtext>EOS</mtext><mo>∈</mo><msub><mi>o</mi><mi>i</mi></msub><mo stretchy="false">]</mo></mrow><mo rspace="2.167em">,</mo><msub><mi>ℒ</mi><mtext>filtered</mtext></msub><mo rspace="0em">=</mo><mo lspace="0em">−</mo><mfrac><mrow><msubsup><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>G</mi></msubsup><mrow><msub><mi>m</mi><mi>i</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><msub><mo rspace="0em">∑</mo><mi>t</mi></msub><mrow><mo stretchy="false">(</mo><mo lspace="0em" rspace="0em">⋯</mo><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow><mrow><msubsup><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>G</mi></msubsup><mrow><msub><mi>m</mi><mi>i</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">|</mo><msub><mi>o</mi><mi>i</mi></msub><mo stretchy="false">|</mo></mrow></mrow></mrow></mfrac><mo lspace="0em">.</mo></mrow></semantics></math></div>

与硬性屏蔽不同，一种更温和的变体施加长度惩罚，它随着补全接近最大长度 <math alttext="L_{\max}" display="inline"><semantics><msub><mi>L</mi><mi>max</mi></msub></semantics></math> 而平滑增大：

<div class="hh-equation" id="ch7.ex6"><math alttext="r_{i}\leftarrow r_{i}-\lambda\cdot\max\!\left(0,\,\frac{|o_{i}|-L_{\text{cache}}}{L_{\max}-L_{\text{cache}}}\right)," display="block"><semantics><mrow><mrow><msub><mi>r</mi><mi>i</mi></msub><mo stretchy="false">←</mo><mrow><msub><mi>r</mi><mi>i</mi></msub><mo>−</mo><mrow><mi>λ</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><mrow><mpadded width="1.808em"><mi>max</mi></mpadded><mo>⁡</mo><mrow><mo>(</mo><mn>0</mn><mo rspace="0.337em">,</mo><mfrac><mrow><mrow><mo stretchy="false">|</mo><msub><mi>o</mi><mi>i</mi></msub><mo stretchy="false">|</mo></mrow><mo>−</mo><msub><mi>L</mi><mtext>cache</mtext></msub></mrow><mrow><msub><mi>L</mi><mi>max</mi></msub><mo>−</mo><msub><mi>L</mi><mtext>cache</mtext></msub></mrow></mfrac><mo>)</mo></mrow></mrow></mrow></mrow></mrow><mo>,</mo></mrow></semantics></math></div>

其中 <math alttext="L_{\text{cache}}" display="inline"><semantics><msub><mi>L</mi><mtext>cache</mtext></msub></semantics></math> 是「安全」长度阈值。

DAPO 会重新采样那些整组补全都获得相同奖励（全部正确或全部错误）的 prompt，因为这类组在归一化之后贡献的梯度为零。这样可以在整个训练过程中保持有效 batch size 稳定。

### GSPO ——Group Sequence Policy Optimization

GSPO [51] 将<em>序列级</em>重要性权重定义为逐 token 比值的几何平均，它等于完整序列概率比的 <math alttext="|o_{i}|" display="inline"><semantics><mrow><mo stretchy="false">|</mo><msub><mi>o</mi><mi>i</mi></msub><mo stretchy="false">|</mo></mrow></semantics></math> 次方根：

<div class="hh-equation" id="ch7.ex7"><math alttext="\boxed{s_{i}(\theta)=\left(\frac{\pi_{\theta}(o_{i}\mid q)}{\pi_{\text{old}}(o_{i}\mid q)}\right)^{1/|o_{i}|}=\exp\!\left(\frac{1}{|o_{i}|}\sum_{t=1}^{|o_{i}|}\log\frac{\pi_{\theta}(o_{i,t}|q,o_{i,&lt;t})}{\pi_{\text{old}}(o_{i,t}|q,o_{i,&lt;t})}\right).}" display="block"><semantics><menclose notation="box"><mrow><mrow><mrow><msub><mi>s</mi><mi>i</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><msup><mrow><mo>(</mo><mfrac><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>o</mi><mi>i</mi></msub><mo fence="true" lspace="0em" rspace="0em">∣</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mi>π</mi><mtext>old</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>o</mi><mi>i</mi></msub><mo fence="true" lspace="0em" rspace="0em">∣</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow></mfrac><mo>)</mo></mrow><mrow><mn>1</mn><mo>/</mo><mrow><mo stretchy="false">|</mo><msub><mi>o</mi><mi>i</mi></msub><mo stretchy="false">|</mo></mrow></mrow></msup><mo>=</mo><mrow><mpadded width="1.370em"><mi>exp</mi></mpadded><mo>⁡</mo><mrow><mo>(</mo><mrow><mfrac><mn>1</mn><mrow><mo stretchy="false">|</mo><msub><mi>o</mi><mi>i</mi></msub><mo stretchy="false">|</mo></mrow></mfrac><mo lspace="0em" rspace="0em">​</mo><mrow><munderover><mo movablelimits="false">∑</mo><mrow><mi>t</mi><mo>=</mo><mn>1</mn></mrow><mrow><mo stretchy="false">|</mo><msub><mi>o</mi><mi>i</mi></msub><mo stretchy="false">|</mo></mrow></munderover><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mfrac><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>o</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub><mo lspace="0em" rspace="0em">|</mo><mrow><mi>q</mi><mo>,</mo><msub><mi>o</mi><mrow><mrow><mi>i</mi><mo>,</mo><mphantom></mphantom></mrow><mo>&lt;</mo><mi>t</mi></mrow></msub></mrow><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mi>π</mi><mtext>old</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>o</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub><mo lspace="0em" rspace="0em">|</mo><mrow><mi>q</mi><mo>,</mo><msub><mi>o</mi><mrow><mrow><mi>i</mi><mo>,</mo><mphantom></mphantom></mrow><mo>&lt;</mo><mi>t</mi></mrow></msub></mrow><mo stretchy="false">)</mo></mrow></mrow></mfrac></mrow></mrow></mrow><mo>)</mo></mrow></mrow></mrow><mo lspace="0em">.</mo></mrow></menclose></semantics></math></div>

这是经过<em>长度归一化</em>的序列概率比。GSPO 的 loss 对每条序列只裁剪这一个标量：

<div class="hh-equation" id="ch7.ex8"><math alttext="\mathcal{L}_{\text{GSPO}}=-\frac{1}{G}\sum_{i=1}^{G}\min\!\Bigl(s_{i}(\theta)\,\hat{A}_{i},\;\mathrm{clip}(s_{i}(\theta),1{-}\epsilon,1{+}\epsilon)\,\hat{A}_{i}\Bigr)." display="block"><semantics><mrow><msub><mi>ℒ</mi><mtext>GSPO</mtext></msub><mo rspace="0em">=</mo><mo lspace="0em">−</mo><mfrac><mn>1</mn><mi>G</mi></mfrac><munderover><mo movablelimits="false">∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>G</mi></munderover><mpadded width="1.653em"><mi>min</mi></mpadded><mrow><mo maxsize="1.600em" minsize="1.600em">(</mo><msub><mi>s</mi><mi>i</mi></msub><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo rspace="0.170em" stretchy="false">)</mo></mrow><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>i</mi></msub><mo rspace="0.447em">,</mo><mi>clip</mi><mrow><mo stretchy="false">(</mo><msub><mi>s</mi><mi>i</mi></msub><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow><mo>,</mo><mn>1</mn><mo>−</mo><mi>ϵ</mi><mo>,</mo><mn>1</mn><mo>+</mo><mi>ϵ</mi><mo rspace="0.170em" stretchy="false">)</mo></mrow><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>i</mi></msub><mo maxsize="1.600em" minsize="1.600em">)</mo></mrow><mo lspace="0em">.</mo></mrow></semantics></math></div>

### Dr. GRPO ——Debiased Reward GRPO

Dr. GRPO 修改逐 token 的梯度权重，以考虑该 token 对奖励信号的边际贡献。模型已经赋予高概率的 token（无论奖励如何）会被降权：

<div class="hh-equation" id="ch7.ex9"><math alttext="w_{i,t}=\hat{A}_{i}\cdot\bigl(1-\pi_{\text{ref}}(o_{i,t}|q,o_{i,&lt;t})\bigr)," display="block"><semantics><mrow><mrow><msub><mi>w</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub><mo>=</mo><mrow><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>i</mi></msub><mo lspace="0.222em" rspace="0.222em">⋅</mo><mrow><mo maxsize="1.200em" minsize="1.200em">(</mo><mrow><mn>1</mn><mo>−</mo><mrow><msub><mi>π</mi><mtext>ref</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>o</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub><mo lspace="0em" rspace="0em">|</mo><mrow><mi>q</mi><mo>,</mo><msub><mi>o</mi><mrow><mrow><mi>i</mi><mo>,</mo><mphantom></mphantom></mrow><mo>&lt;</mo><mi>t</mi></mrow></msub></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow><mo maxsize="1.200em" minsize="1.200em">)</mo></mrow></mrow></mrow><mo>,</mo></mrow></semantics></math></div>

其中 <math alttext="\pi_{\text{ref}}" display="inline"><semantics><msub><mi>π</mi><mtext>ref</mtext></msub></semantics></math> 是参考（预训练）模型。这是一种<em>token 效率</em>：梯度集中在策略真正需要改变的 token 上。

### 2-GRPO ——最简的两次 Rollout GRPO

关键洞见在于：GRPO 的有效性<em>并非</em>主要来自精确的 advantage 估计（后者需要很大的 <math alttext="G" display="inline"><semantics><mi>G</mi></semantics></math>）。相反，它来自一个隐式的<em>对比目标</em>，其结构与 DPO 相似：

<div class="hh-equation" id="ch7.ex10"><math alttext="\mathcal{L}_{\text{2-GRPO}}\approx-\mathbb{E}_{(o^{+},o^{-})\sim\pi_{\theta}}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_{\theta}(o^{+}|q)}{\pi_{\text{old}}(o^{+}|q)}-\beta\log\frac{\pi_{\theta}(o^{-}|q)}{\pi_{\text{old}}(o^{-}|q)}\right)\right]," display="block"><semantics><mrow><mrow><msub><mi>ℒ</mi><mtext>2-GRPO</mtext></msub><mo>≈</mo><mrow><mo>−</mo><mrow><msub><mi>𝔼</mi><mrow><mrow><mo stretchy="false">(</mo><msup><mi>o</mi><mo>+</mo></msup><mo>,</mo><msup><mi>o</mi><mo>−</mo></msup><mo stretchy="false">)</mo></mrow><mo>∼</mo><msub><mi>π</mi><mi>θ</mi></msub></mrow></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo>[</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mrow><mpadded width="0.437em"><mi>σ</mi></mpadded><mo>⁡</mo><mrow><mo>(</mo><mrow><mrow><mi>β</mi><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mfrac><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msup><mi>o</mi><mo>+</mo></msup><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mi>π</mi><mtext>old</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msup><mi>o</mi><mo>+</mo></msup><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow></mfrac></mrow></mrow><mo>−</mo><mrow><mi>β</mi><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mfrac><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msup><mi>o</mi><mo>−</mo></msup><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mi>π</mi><mtext>old</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msup><mi>o</mi><mo>−</mo></msup><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow></mfrac></mrow></mrow></mrow><mo>)</mo></mrow></mrow></mrow><mo>]</mo></mrow></mrow></mrow></mrow><mo>,</mo></mrow></semantics></math></div>

其中 <math alttext="o^{+}" display="inline"><semantics><msup><mi>o</mi><mo>+</mo></msup></semantics></math> 是奖励更高的补全，<math alttext="o^{-}" display="inline"><semantics><msup><mi>o</mi><mo>−</mo></msup></semantics></math> 是奖励更低的补全。当 <math alttext="G=2" display="inline"><semantics><mrow><mi>G</mi><mo>=</mo><mn>2</mn></mrow></semantics></math> 时，这种对比结构是显式的。当 <math alttext="G=16" display="inline"><semantics><mrow><mi>G</mi><mo>=</mo><mn>16</mn></mrow></semantics></math> 时，同样的信号仍然存在，但被冗余的配对稀释了。

### SAPO ——Soft Adaptive Policy Optimization

SAPO 用一个平滑的代理目标取代 <math alttext="\min(\rho A,\mathrm{clip}(\rho,\cdot)\,A)" display="inline"><semantics><mrow><mi>min</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mi>ρ</mi><mo lspace="0em" rspace="0em">​</mo><mi>A</mi></mrow><mo>,</mo><mrow><mrow><mi>clip</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>ρ</mi><mo rspace="0em">,</mo><mo lspace="0em" rspace="0em">⋅</mo><mo stretchy="false">)</mo></mrow></mrow><mo lspace="0.170em" rspace="0em">​</mo><mi>A</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 目标：

<div class="hh-equation" id="ch7.ex11"><math alttext="\boxed{\mathcal{L}_{\text{SAPO}}(\rho,A)=\begin{cases}-A\cdot\sigma\!\left(\dfrac{\rho-1}{\tau_{+}}\right)\cdot\rho&amp;\text{if }A&gt;0\\[8.0pt]
-A\cdot\sigma\!\left(\dfrac{1-\rho}{\tau_{-}}\right)\cdot\rho&amp;\text{if }A\leq 0\end{cases}}" display="block"><semantics><menclose notation="box"><mrow><mrow><msub><mi>ℒ</mi><mtext>SAPO</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>ρ</mi><mo>,</mo><mi>A</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mo>{</mo><mtable columnspacing="5pt" displaystyle="true" rowspacing="0pt"><mtr><mtd columnalign="left"><mrow><mo>−</mo><mi>A</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><mpadded width="0.437em"><mi>σ</mi></mpadded><mo>(</mo><mfrac><mrow><mi>ρ</mi><mo>−</mo><mn>1</mn></mrow><msub><mi>τ</mi><mo>+</mo></msub></mfrac><mo rspace="0.055em">)</mo><mo rspace="0.222em">⋅</mo><mi>ρ</mi></mrow></mtd><mtd columnalign="left"><mrow><mrow><mtext>if </mtext><mo lspace="0em" rspace="0em">​</mo><mi>A</mi></mrow><mo>&gt;</mo><mn>0</mn></mrow></mtd></mtr><mtr><mtd columnalign="left"><mrow><mo>−</mo><mi>A</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><mpadded width="0.437em"><mi>σ</mi></mpadded><mo>(</mo><mfrac><mrow><mn>1</mn><mo>−</mo><mi>ρ</mi></mrow><msub><mi>τ</mi><mo>−</mo></msub></mfrac><mo rspace="0.055em">)</mo><mo rspace="0.222em">⋅</mo><mi>ρ</mi></mrow></mtd><mtd columnalign="left"><mrow><mrow><mtext>if </mtext><mo lspace="0em" rspace="0em">​</mo><mi>A</mi></mrow><mo>≤</mo><mn>0</mn></mrow></mtd></mtr></mtable></mrow></mrow></menclose></semantics></math></div>

其中 <math alttext="\sigma" display="inline"><semantics><mi>σ</mi></semantics></math> 是 sigmoid 函数，<math alttext="\tau_{+},\tau_{-}" display="inline"><semantics><mrow><msub><mi>τ</mi><mo>+</mo></msub><mo>,</mo><msub><mi>τ</mi><mo>−</mo></msub></mrow></semantics></math> 是非对称的 temperature 参数。temperature 越高，门控越柔和（探索更多）；temperature 越低，则越接近硬裁剪。

### TIS 与 MIS ——Truncated 与 Masked Importance Sampling

TIS 通过将梯度乘以一个截断后的修正因子来修正偏差：

<div class="hh-equation" id="ch7.ex12"><math alttext="\boxed{w_{\text{TIS}}(o_{i})=\min\!\left(C,\;\frac{\pi_{\text{train}}(o_{i}|q)}{\pi_{\text{vllm}}(o_{i}|q)}\right),}" display="block"><semantics><menclose notation="box"><mrow><mrow><mrow><msub><mi>w</mi><mtext>TIS</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>o</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mpadded width="1.653em"><mi>min</mi></mpadded><mo>⁡</mo><mrow><mo>(</mo><mi>C</mi><mo rspace="0.447em">,</mo><mfrac><mrow><msub><mi>π</mi><mtext>train</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>o</mi><mi>i</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mi>π</mi><mtext>vllm</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>o</mi><mi>i</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow></mfrac><mo>)</mo></mrow></mrow></mrow><mo>,</mo></mrow></menclose></semantics></math></div>

其中 <math alttext="\pi_{\text{train}}" display="inline"><semantics><msub><mi>π</mi><mtext>train</mtext></msub></semantics></math> 是训练前向传播得到的概率，<math alttext="\pi_{\text{vllm}}" display="inline"><semantics><msub><mi>π</mi><mtext>vllm</mtext></msub></semantics></math> 是 vLLM 报告的概率。在 <math alttext="C" display="inline"><semantics><mi>C</mi></semantics></math> 处的截断可防止极端修正破坏训练稳定性。

MIS 采取更激进的做法：对任何修正比超过阈值 <math alttext="C" display="inline"><semantics><mi>C</mi></semantics></math> 的序列，将其梯度置零：

<div class="hh-equation" id="ch7.ex13"><math alttext="w_{\text{MIS}}(o_{i})=\mathbf{1}\!\left[\frac{\pi_{\text{train}}(o_{i}|q)}{\pi_{\text{vllm}}(o_{i}|q)}\leq C\right]." display="block"><semantics><mrow><msub><mi>w</mi><mtext>MIS</mtext></msub><mrow><mo stretchy="false">(</mo><msub><mi>o</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow><mo rspace="0.108em">=</mo><mrow><mo>[</mo><mfrac><mrow><msub><mi>π</mi><mtext>train</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>o</mi><mi>i</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mi>π</mi><mtext>vllm</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>o</mi><mi>i</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow></mfrac><mo>≤</mo><mi>C</mi><mo>]</mo></mrow><mo lspace="0em">.</mo></mrow></semantics></math></div>

这种做法更保守，但避免了使用较大（即使是截断后的）修正权重的风险。

TIS 与 MIS 既可以应用在 token 级，也可以应用在序列级：

<ul><li>序列级：把比值计算为所有 token 上的几何平均（如 GSPO 那样）。理论上正确，但方差更高。</li><li>token 级：为每个 token 分别计算比值。有偏（逐 token 修正的乘积并不等于序列修正），但方差更低。</li></ul>

### VESPO ——Variational Sequence-Level Soft Policy Optimization

VESPO 从变分目标出发，为每条轨迹 <math alttext="\tau" display="inline"><semantics><mi>τ</mi></semantics></math> 推导出加权函数 <math alttext="W(\tau)" display="inline"><semantics><mrow><mi>W</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>τ</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math>。最终的梯度权重形如：

<div class="hh-equation" id="ch7.ex14"><math alttext="\boxed{g(\tau)=W(\tau)^{k}\cdot\exp\!\bigl(\lambda(1-W(\tau))\bigr),}" display="block"><semantics><menclose notation="box"><mrow><mrow><mrow><mi>g</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>τ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mrow><mi>W</mi><mo lspace="0em" rspace="0em">​</mo><msup><mrow><mo stretchy="false">(</mo><mi>τ</mi><mo rspace="0.055em" stretchy="false">)</mo></mrow><mi>k</mi></msup></mrow><mo rspace="0.222em">⋅</mo><mrow><mpadded width="1.370em"><mi>exp</mi></mpadded><mo>⁡</mo><mrow><mo maxsize="1.200em" minsize="1.200em">(</mo><mrow><mi>λ</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mn>1</mn><mo>−</mo><mrow><mi>W</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>τ</mi><mo stretchy="false">)</mo></mrow></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow><mo maxsize="1.200em" minsize="1.200em">)</mo></mrow></mrow></mrow></mrow><mo>,</mo></mrow></menclose></semantics></math></div>

其中 <math alttext="W(\tau)=\pi_{\theta}(\tau)/\pi_{\text{old}}(\tau)" display="inline"><semantics><mrow><mrow><mi>W</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>τ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mrow><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>τ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>/</mo><msub><mi>π</mi><mtext>old</mtext></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>τ</mi><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math> 是序列级重要性权重，<math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 控制加权的锐度，<math alttext="\lambda" display="inline"><semantics><mi>λ</mi></semantics></math> 控制对陈旧（低权重）轨迹的指数衰减。该核：

<ul><li>在各处都平滑（裁剪边界处没有不连续的梯度）。</li><li>通过指数项自然地降低陈旧轨迹（<math alttext="W\ll 1" display="inline"><semantics><mrow><mi>W</mi><mo>≪</mo><mn>1</mn></mrow></semantics></math>）的权重。</li><li>是非对称的：高权重轨迹（<math alttext="W&gt;1" display="inline"><semantics><mrow><mi>W</mi><mo>&gt;</mo><mn>1</mn></mrow></semantics></math>）与低权重轨迹的处理方式不同。</li></ul>

### DPPO ——Direct Policy Divergence Policy Optimization

DPPO 直接用新旧策略分布之间的全变差（Total Variation, TV）或 KL 散度来计算信赖域约束：

<div class="hh-equation" id="ch7.ex15"><math alttext="\mathcal{L}_{\text{DPPO}}=-\mathbb{E}\!\left[\hat{A}\cdot\pi_{\theta}(o|q)\cdot\mathbf{1}[D(\pi_{\theta}\|\pi_{\text{old}})\leq\delta]\right]," display="block"><semantics><mrow><msub><mi>ℒ</mi><mtext>DPPO</mtext></msub><mo rspace="0em">=</mo><mo lspace="0em">−</mo><mpadded width="0.497em"><mi>𝔼</mi></mpadded><mrow><mo>[</mo><mover accent="true"><mi>A</mi><mo>^</mo></mover><mo lspace="0.222em" rspace="0.222em">⋅</mo><msub><mi>π</mi><mi>θ</mi></msub><mrow><mo stretchy="false">(</mo><mi>o</mi><mo fence="false" rspace="0.167em" stretchy="false">|</mo><mi>q</mi><mo rspace="0.055em" stretchy="false">)</mo></mrow><mo rspace="0.222em">⋅</mo><mn>𝟏</mn><mrow><mo stretchy="false">[</mo><mi>D</mi><mrow><mo stretchy="false">(</mo><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0.167em">∥</mo><msub><mi>π</mi><mtext>old</mtext></msub><mo stretchy="false">)</mo></mrow><mo>≤</mo><mi>δ</mi><mo stretchy="false">]</mo></mrow><mo>]</mo></mrow><mo>,</mo></mrow></semantics></math></div>

其中 <math alttext="D" display="inline"><semantics><mi>D</mi></semantics></math> 是所选用的散度度量。实践中，DPPO 用 token 级的二值掩码或 top-<math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 掩码来近似它：

<ul><li>binary_tv：屏蔽满足 <math alttext="|\pi_{\theta}-\pi_{\text{old}}|&gt;\delta" display="inline"><semantics><mrow><mrow><mo stretchy="false">|</mo><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo>−</mo><msub><mi>π</mi><mtext>old</mtext></msub></mrow><mo stretchy="false">|</mo></mrow><mo>&gt;</mo><mi>δ</mi></mrow></semantics></math> 的 token。</li><li>binary_kl：屏蔽满足 <math alttext="\pi_{\theta}\log(\pi_{\theta}/\pi_{\text{old}})&gt;\delta" display="inline"><semantics><mrow><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo>/</mo><msub><mi>π</mi><mtext>old</mtext></msub></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow><mo>&gt;</mo><mi>δ</mi></mrow></semantics></math> 的 token。</li><li>topk_tv：仅保留按 TV 贡献排序的前 <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 个 token。</li><li>topk_kl：仅保留按 KL 贡献排序的前 <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 个 token。</li></ul>

### ScaleRL 和 CISPO

标准 GRPO 在单个 prompt 的 <math alttext="G" display="inline"><semantics><mi>G</mi></semantics></math> 条补全构成的组内对奖励做归一化。CISPO 则在<em>整个 batch</em>范围内对奖励做归一化：

<div class="hh-equation" id="ch7.ex16"><math alttext="\hat{A}_{i}=\frac{r_{i}-\mu_{\text{batch}}}{\sigma_{\text{batch}}+\epsilon}," display="block"><semantics><mrow><mrow><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>i</mi></msub><mo>=</mo><mfrac><mrow><msub><mi>r</mi><mi>i</mi></msub><mo>−</mo><msub><mi>μ</mi><mtext>batch</mtext></msub></mrow><mrow><msub><mi>σ</mi><mtext>batch</mtext></msub><mo>+</mo><mi>ϵ</mi></mrow></mfrac></mrow><mo>,</mo></mrow></semantics></math></div>

其中 <math alttext="\mu_{\text{batch}}" display="inline"><semantics><msub><mi>μ</mi><mtext>batch</mtext></msub></semantics></math> 和 <math alttext="\sigma_{\text{batch}}" display="inline"><semantics><msub><mi>σ</mi><mtext>batch</mtext></msub></semantics></math> 在当前训练 batch 的所有奖励（reward）上计算。这提供了更稳定的 baseline，并防止任何单一 prompt 主导 gradient。

CISPO 结合了 batch 级缩放、DAPO 的 token 级 loss 聚合以及非对称 clipping：

<div class="hh-equation" id="ch7.ex17"><math alttext="\mathcal{L}_{\text{CISPO}}=-\frac{1}{\sum_{i,t}m_{i,t}}\sum_{i=1}^{G}\sum_{t=1}^{|o_{i}|}m_{i,t}\cdot\min\!\bigl(\rho_{i,t}\hat{A}_{i},\;\mathrm{clip}_{\text{DAPO}}(\rho_{i,t},\hat{A}_{i})\,\hat{A}_{i}\bigr)," display="block"><semantics><mrow><msub><mi>ℒ</mi><mtext>CISPO</mtext></msub><mo rspace="0em">=</mo><mo lspace="0em">−</mo><mfrac><mn>1</mn><mrow><msub><mo>∑</mo><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub><msub><mi>m</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub></mrow></mfrac><munderover><mo movablelimits="false" rspace="0em">∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>G</mi></munderover><munderover><mo movablelimits="false">∑</mo><mrow><mi>t</mi><mo>=</mo><mn>1</mn></mrow><mrow><mo stretchy="false">|</mo><msub><mi>o</mi><mi>i</mi></msub><mo stretchy="false">|</mo></mrow></munderover><msub><mi>m</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub><mo lspace="0.222em" rspace="0.222em">⋅</mo><mpadded width="1.653em"><mi>min</mi></mpadded><mrow><mo maxsize="1.200em" minsize="1.200em">(</mo><msub><mi>ρ</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>i</mi></msub><mo rspace="0.447em">,</mo><msub><mi>clip</mi><mtext>DAPO</mtext></msub><mrow><mo stretchy="false">(</mo><msub><mi>ρ</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub><mo>,</mo><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>i</mi></msub><mo rspace="0.170em" stretchy="false">)</mo></mrow><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>i</mi></msub><mo maxsize="1.200em" minsize="1.200em">)</mo></mrow><mo>,</mo></mrow></semantics></math></div>

其中 <math alttext="m_{i,t}" display="inline"><semantics><msub><mi>m</mi><mrow><mi>i</mi><mo>,</mo><mi>t</mi></mrow></msub></semantics></math> 是过长过滤 mask。

### GDPO ——Group Reward-Decoupled Policy Optimization

其核心机制是在聚合前<em>独立</em>归一化每项奖励：

<div class="hh-equation" id="ch7.ex18"><math alttext="\boxed{\hat{A}_{n}^{(i)}=\frac{r_{n}^{(i)}-\mu_{n}}{\sigma_{n}+\epsilon},\qquad\hat{A}^{(i)}=\sum_{n=1}^{N}w_{n}\hat{A}_{n}^{(i)},}" display="block"><semantics><menclose notation="box"><mrow><mover accent="true"><mi>A</mi><mo>^</mo></mover><msub><mrow></mrow><mi>n</mi></msub><msup><mrow></mrow><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>=</mo><mfrac><mrow><msubsup><mi>r</mi><mi>n</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msubsup><mo>−</mo><msub><mi>μ</mi><mi>n</mi></msub></mrow><mrow><msub><mi>σ</mi><mi>n</mi></msub><mo>+</mo><mi>ϵ</mi></mrow></mfrac><mo rspace="2.167em">,</mo><mover accent="true"><mi>A</mi><mo>^</mo></mover><msup><mrow></mrow><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo rspace="0.111em">=</mo><mo movablelimits="false">∑</mo><msub><mrow></mrow><mrow><mi>n</mi><mo>=</mo><mn>1</mn></mrow></msub><msup><mrow></mrow><mi>N</mi></msup><mi>w</mi><msub><mrow></mrow><mi>n</mi></msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><msub><mrow></mrow><mi>n</mi></msub><msup><mrow></mrow><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>,</mo></mrow></menclose></semantics></math></div>

其中 <math alttext="r_{n}^{(i)}" display="inline"><semantics><msubsup><mi>r</mi><mi>n</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msubsup></semantics></math> 是补全（completion）<math alttext="i" display="inline"><semantics><mi>i</mi></semantics></math> 的第 <math alttext="n" display="inline"><semantics><mi>n</mi></semantics></math> 项奖励，<math alttext="\mu_{n}" display="inline"><semantics><msub><mi>μ</mi><mi>n</mi></msub></semantics></math> 和 <math alttext="\sigma_{n}" display="inline"><semantics><msub><mi>σ</mi><mi>n</mi></msub></semantics></math> 是组内奖励 <math alttext="n" display="inline"><semantics><mi>n</mi></semantics></math> 的均值与标准差，<math alttext="w_{n}" display="inline"><semantics><msub><mi>w</mi><mi>n</mi></msub></semantics></math> 是用户指定的权重。

### GOPO ——Group Ordinal Policy Optimization

GOPO [54] 始于一个简单观察：奖励模型是用成对比较（“A 是否优于 B？”）训练的，因此只有其输出的排序是可信的——原始数值分数本身并没有内在含义。然而 GRPO 却把这些原始幅度直接喂入优势计算。对于奖励不可验证的任务——摘要、开放式对话、指令遵循——这种不匹配会引入噪声：在输出空间的某个区域，0.6 个奖励分的差距可能反映真实的质量差异，而在另一个区域却毫无意义。

关键洞见：完全丢弃奖励的幅度信息。只使用组内奖励的序数排名。

算法：给定一组 <math alttext="N" display="inline"><semantics><mi>N</mi></semantics></math> 个响应 <math alttext="\{o_{1},\ldots,o_{N}\}" display="inline"><semantics><mrow><mo stretchy="false">{</mo><msub><mi>o</mi><mn>1</mn></msub><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msub><mi>o</mi><mi>N</mi></msub><mo stretchy="false">}</mo></mrow></semantics></math>，其奖励为 <math alttext="\{r_{1},\ldots,r_{N}\}" display="inline"><semantics><mrow><mo stretchy="false">{</mo><msub><mi>r</mi><mn>1</mn></msub><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msub><mi>r</mi><mi>N</mi></msub><mo stretchy="false">}</mo></mrow></semantics></math>：

<ol><li>按奖励对响应排名：赋予排名 <math alttext="\text{rank}(o_{i})\in\{1,\ldots,N\}" display="inline"><semantics><mrow><mrow><mtext>rank</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>o</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>∈</mo><mrow><mo stretchy="false">{</mo><mn>1</mn><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><mi>N</mi><mo stretchy="false">}</mo></mrow></mrow></semantics></math>（1 = 最差，<math alttext="N" display="inline"><semantics><mi>N</mi></semantics></math> = 最好）。</li><li>将原始优势替换为基于排名的分数：<math alttext="\boxed{\hat{A}_{i}^{\text{GOPO}}=f\!\left(\frac{\text{rank}(o_{i})}{N}\right)}" display="block"><semantics><menclose notation="box"><mrow><msubsup><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>i</mi><mtext>GOPO</mtext></msubsup><mo>=</mo><mrow><mpadded width="0.427em"><mi>f</mi></mpadded><mo>⁡</mo><mrow><mo>(</mo><mfrac><mrow><mtext>rank</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>o</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mi>N</mi></mfrac><mo>)</mo></mrow></mrow></mrow></menclose></semantics></math><span class="hh-tag">(7.2)</span>其中 <math alttext="f" display="inline"><semantics><mi>f</mi></semantics></math> 是单调变换（例如线性映射到 <math alttext="[-1,1]" display="inline"><semantics><mrow><mo stretchy="false">[</mo><mrow><mrow><mo>−</mo><mn>1</mn></mrow><mo>,</mo><mn>1</mn></mrow><mo stretchy="false">]</mo></mrow></semantics></math>，或分位数归一化）。</li><li>使用基于排名的优势，应用 PPO 风格的 clipped 目标函数。</li></ol>

与 GRPO 的对比：

方面GRPOGOPO优势信号<math alttext="\hat{A}_{i}=(r_{i}-\mu)/\sigma" display="inline"><semantics><mrow><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>i</mi></msub><mo>=</mo><mrow><mrow><mo stretchy="false">(</mo><mrow><msub><mi>r</mi><mi>i</mi></msub><mo>−</mo><mi>μ</mi></mrow><mo stretchy="false">)</mo></mrow><mo>/</mo><mi>σ</mi></mrow></mrow></semantics></math>（使用幅度）<math alttext="\hat{A}_{i}=f(\text{rank}_{i}/N)" display="inline"><semantics><mrow><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>i</mi></msub><mo>=</mo><mrow><mi>f</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><msub><mtext>rank</mtext><mi>i</mi></msub><mo>/</mo><mi>N</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math>（仅使用序数排名）对奖励量级的敏感度高——标定不当的 RM 分数会扭曲优势无——对单调奖励变换保持不变最适合的场景可验证的奖励（二值、标定良好）不可验证的奖励（基于 RM、量级含噪）

经验收益（在不可验证任务上相对 GRPO）：

<ul><li>奖励曲线（训练与 held-out）在整个优化过程中都高于 GRPO</li><li>由独立 LLM 评估器判定的胜率在大多数训练 checkpoint 上都有提升</li><li>收敛明显更快——用更少的 gradient step 达到 GRPO 的最终质量</li><li>当奖励模型噪声更大或标定更差时，这一优势会更加明显</li></ul>

<aside class="hh-next">
<p class="hh-next-label">下一篇</p>
<p class="hh-next-title">智能体 AI 漫游指南 · 第 8 章 偏好优化变体</p>
<p class="hh-next-hint">在「智能体 AI 漫游指南」分类下继续阅读。</p>
</aside>

</div>
