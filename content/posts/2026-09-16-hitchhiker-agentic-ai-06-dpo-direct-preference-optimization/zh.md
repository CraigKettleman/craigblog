---
title: "智能体 AI 漫游指南 · 第 6 章 DPO——直接偏好优化"
slug: "hitchhiker-agentic-ai-06-dpo-direct-preference-optimization"
lang: "zh"
date: "2026-09-16T00:07:00.000Z"
draft: false
featured: false
categories:
  - "hitchhiker-agentic-ai"
cover: "./cover.png"
description: "PPO 需要在显存中维持 4 个模型（policy、reference、reward 模型、value head）、复杂的 RL 基础设施，并且以不稳定著称。DPO [302] 提出的问题是：我们能否跳过 RL，直接从偏好中学习？"
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

## 动机

PPO 需要在显存中维持 4 个模型（policy、reference、reward 模型、value head）、复杂的 RL 基础设施，并且以不稳定著称。DPO [302] 提出的问题是：<em>我们能否跳过 RL，直接从偏好中学习？</em>

关键洞察：在 RLHF 目标（reward 最大化 + KL 惩罚）下的最优 policy 拥有解析解。我们可以推导出一个有监督 loss，隐式优化同一个目标。

## 数学推导

第 1 步：RLHF 目标：<math alttext="\max_{\pi}\mathbb{E}_{x,y\sim\pi}[r(x,y)]-\beta D_{\text{KL}}[\pi\|\pi_{\text{ref}}]" display="inline"><semantics><mrow><msub><mi>max</mi><mi>π</mi></msub><msub><mi>𝔼</mi><mrow><mrow><mi>x</mi><mo>,</mo><mi>y</mi></mrow><mo>∼</mo><mi>π</mi></mrow></msub><mrow><mo stretchy="false">[</mo><mi>r</mi><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo stretchy="false">)</mo></mrow><mo stretchy="false">]</mo></mrow><mo>−</mo><mi>β</mi><msub><mi>D</mi><mtext>KL</mtext></msub><mrow><mo stretchy="false">[</mo><mi>π</mi><mo lspace="0em" rspace="0.167em">∥</mo><msub><mi>π</mi><mtext>ref</mtext></msub><mo stretchy="false">]</mo></mrow></mrow></semantics></math>

第 2 步：最优解为：<math alttext="\pi^{*}(y|x)=\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{r(x,y)}{\beta}\right)" display="inline"><semantics><mrow><mrow><msup><mi>π</mi><mo>∗</mo></msup><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>y</mi><mo lspace="0em" rspace="0em">|</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mfrac><mn>1</mn><mrow><mi>Z</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow></mfrac><mo lspace="0em" rspace="0em">​</mo><msub><mi>π</mi><mtext>ref</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>y</mi><mo lspace="0em" rspace="0em">|</mo><mi>x</mi><mo stretchy="false">)</mo></mrow><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>exp</mi><mo>⁡</mo><mrow><mo>(</mo><mfrac><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo stretchy="false">)</mo></mrow></mrow><mi>β</mi></mfrac><mo>)</mo></mrow></mrow></mrow></mrow></semantics></math>

第 3 步：重排式子，用 policy 表达 reward：<math alttext="r(x,y)=\beta\log\frac{\pi^{*}(y|x)}{\pi_{\text{ref}}(y|x)}+\beta\log Z(x)" display="inline"><semantics><mrow><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mrow><mi>β</mi><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mfrac><mrow><msup><mi>π</mi><mo>∗</mo></msup><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>y</mi><mo lspace="0em" rspace="0em">|</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mi>π</mi><mtext>ref</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>y</mi><mo lspace="0em" rspace="0em">|</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow></mfrac></mrow></mrow><mo>+</mo><mrow><mi>β</mi><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mrow><mi>Z</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></mrow></mrow></semantics></math>

第 4 步：代入 Bradley-Terry 偏好模型 <math alttext="P(y_{w}\succ y_{l})=\sigma(r(y_{w})-r(y_{l}))" display="inline"><semantics><mrow><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><msub><mi>y</mi><mi>w</mi></msub><mo>≻</mo><msub><mi>y</mi><mi>l</mi></msub></mrow><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mi>σ</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>w</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>−</mo><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>l</mi></msub><mo stretchy="false">)</mo></mrow></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math>。<math alttext="Z(x)" display="inline"><semantics><mrow><mi>Z</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 项相互抵消！

<div class="hh-equation" id="ch6.e1"><math alttext="\boxed{\mathcal{L}_{\text{DPO}}(\theta)=-\mathbb{E}_{(x,y_{w},y_{l})}\left[\log\sigma\left(\beta\log\frac{\pi_{\theta}(y_{w}|x)}{\pi_{\text{ref}}(y_{w}|x)}-\beta\log\frac{\pi_{\theta}(y_{l}|x)}{\pi_{\text{ref}}(y_{l}|x)}\right)\right]}" display="block"><semantics><menclose notation="box"><mrow><mrow><msub><mi>ℒ</mi><mtext>DPO</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mo>−</mo><mrow><msub><mi>𝔼</mi><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>,</mo><msub><mi>y</mi><mi>w</mi></msub><mo>,</mo><msub><mi>y</mi><mi>l</mi></msub><mo stretchy="false">)</mo></mrow></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo>[</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mrow><mi>σ</mi><mo>⁡</mo><mrow><mo>(</mo><mrow><mrow><mi>β</mi><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mfrac><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>w</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mi>π</mi><mtext>ref</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>w</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow></mfrac></mrow></mrow><mo>−</mo><mrow><mi>β</mi><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mfrac><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>l</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mi>π</mi><mtext>ref</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>l</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow></mfrac></mrow></mrow></mrow><mo>)</mo></mrow></mrow></mrow><mo>]</mo></mrow></mrow></mrow></mrow></menclose></semantics></math><span class="hh-equation-number">(6.1)</span></div>

## Gradient 分析

DPO 的 gradient 可分解为：

<div class="hh-equation" id="ch6.e2"><math alttext="\nabla_{\theta}\mathcal{L}=-\beta\cdot\underbrace{\sigma(-\hat{r}_{w}+\hat{r}_{l})}_{\text{weight: higher when model is wrong}}\cdot\left[\nabla_{\theta}\log\pi_{\theta}(y_{w}|x)-\nabla_{\theta}\log\pi_{\theta}(y_{l}|x)\right]" display="block"><semantics><mrow><msub><mo>∇</mo><mi>θ</mi></msub><mi>ℒ</mi><mo rspace="0em">=</mo><mo lspace="0em">−</mo><mi>β</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><munder><munder accentunder="true"><mrow><mi>σ</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mrow><mo>−</mo><msub><mover accent="true"><mi>r</mi><mo>^</mo></mover><mi>w</mi></msub></mrow><mo>+</mo><msub><mover accent="true"><mi>r</mi><mo>^</mo></mover><mi>l</mi></msub></mrow><mo rspace="0.055em" stretchy="false">)</mo></mrow></mrow><mo stretchy="true">⏟</mo></munder><mtext>weight: higher when model is wrong</mtext></munder><mo rspace="0.222em">⋅</mo><mrow><mo>[</mo><msub><mo rspace="0.167em">∇</mo><mi>θ</mi></msub><mi>log</mi><msub><mi>π</mi><mi>θ</mi></msub><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>w</mi></msub><mo fence="false" rspace="0.167em" stretchy="false">|</mo><mi>x</mi><mo stretchy="false">)</mo></mrow><mo>−</mo><msub><mo rspace="0.167em">∇</mo><mi>θ</mi></msub><mi>log</mi><msub><mi>π</mi><mi>θ</mi></msub><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>l</mi></msub><mo fence="false" rspace="0.167em" stretchy="false">|</mo><mi>x</mi><mo stretchy="false">)</mo></mrow><mo>]</mo></mrow></mrow></semantics></math><span class="hh-equation-number">(6.2)</span></div>

解读：gradient 会提升「被选」项的概率，降低「被拒」项的概率。当模型当前更偏好错误答案时，权重最大——它把学习重点放在「易混淆」的成对样本上。

## TRL 实现

下面给出一个使用 HuggingFace TRL 的最小可运行示例。

```python
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Dataset format: {"prompt": str, "chosen": str, "rejected": str}
dataset = load_dataset("argilla/ultrafeedback-binarized-preferences")

lora_config = LoraConfig(r=64, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])

dpo_config = DPOConfig(
    output_dir="./dpo_output",
    beta=0.1,                    # KL regularization strength
    learning_rate=5e-7,          # Very low LR for stability
    loss_type="sigmoid",         # Standard DPO loss
    max_length=2048,             # Max sequence length
    max_prompt_length=1024,      # Truncation for prompts
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # Effective batch = 16
    gradient_checkpointing=True,
    bf16=True,
    num_train_epochs=1,          # DPO overfits fast - 1 epoch!
    warmup_ratio=0.1,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=500,
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,             # With LoRA, ref = base model (no copy needed!)
    args=dpo_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    peft_config=lora_config,
)
trainer.train()
# Key metrics to monitor: train/rewards/chosen, train/rewards/rejected, train/rewards/margins
```

## DPO 的完整机制

本节给出 DPO 完整的计算细节——训练过程中在 token 层面究竟发生了什么。

### 序列级对数概率

DPO 中的关键量是给定 prompt <math alttext="x" display="inline"><semantics><mi>x</mi></semantics></math> 时整条序列<math alttext="y=(y_{1},y_{2},\ldots,y_{T})" display="inline"><semantics><mrow><mi>y</mi><mo>=</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mn>1</mn></msub><mo>,</mo><msub><mi>y</mi><mn>2</mn></msub><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msub><mi>y</mi><mi>T</mi></msub><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 的对数概率。它按各 token 对数概率之和计算：

<div class="hh-equation" id="ch6.e3"><math alttext="\boxed{\log\pi_{\theta}(y|x)=\sum_{t=1}^{T}\log\pi_{\theta}(y_{t}\mid x,y_{&lt;t})}" display="block"><semantics><menclose notation="box"><mrow><mrow><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>π</mi><mi>θ</mi></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>y</mi><mo lspace="0em" rspace="0em">|</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow><mo rspace="0.111em">=</mo><mrow><munderover><mo movablelimits="false">∑</mo><mrow><mi>t</mi><mo>=</mo><mn>1</mn></mrow><mi>T</mi></munderover><mrow><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>π</mi><mi>θ</mi></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>t</mi></msub><mo fence="true" lspace="0em" rspace="0em">∣</mo><mrow><mi>x</mi><mo>,</mo><msub><mi>y</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>t</mi></mrow></msub></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></menclose></semantics></math><span class="hh-equation-number">(6.3)</span></div>

每一项 <math alttext="\log\pi_{\theta}(y_{t}|x,y_{&lt;t})" display="inline"><semantics><mrow><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>π</mi><mi>θ</mi></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">|</mo><mrow><mi>x</mi><mo>,</mo><msub><mi>y</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>t</mi></mrow></msub></mrow><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 是位置 <math alttext="t" display="inline"><semantics><mi>t</mi></semantics></math> 处针对序列中<em>实际</em> token <math alttext="y_{t}" display="inline"><semantics><msub><mi>y</mi><mi>t</mi></msub></semantics></math> 的 log-softmax 输出。这与标准语言建模所用的交叉熵 loss 完全相同——但这里我们做的是求和，而不是取平均。

关键细节：gradient 会流经 <math alttext="y_{w}" display="inline"><semantics><msub><mi>y</mi><mi>w</mi></msub></semantics></math> 和 <math alttext="y_{l}" display="inline"><semantics><msub><mi>y</mi><mi>l</mi></msub></semantics></math> 中的每一个 token 位置。中间 token 没有被 mask——每个 token 都对序列级对数概率有贡献。

### DPO Loss 的分解

从 loss 出发：

<div class="hh-equation" id="ch6.e4"><math alttext="\mathcal{L}_{\text{DPO}}(\theta)=-\mathbb{E}_{(x,y_{w},y_{l})\sim\mathcal{D}}\!\left[\log\sigma\!\left(\beta\cdot h_{\theta}(x,y_{w},y_{l})\right)\right]" display="block"><semantics><mrow><mrow><msub><mi>ℒ</mi><mtext>DPO</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mo>−</mo><mrow><msub><mi>𝔼</mi><mrow><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>,</mo><msub><mi>y</mi><mi>w</mi></msub><mo>,</mo><msub><mi>y</mi><mi>l</mi></msub><mo stretchy="false">)</mo></mrow><mo>∼</mo><mi>𝒟</mi></mrow></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo>[</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mrow><mpadded width="0.437em"><mi>σ</mi></mpadded><mo>⁡</mo><mrow><mo>(</mo><mrow><mrow><mi>β</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><msub><mi>h</mi><mi>θ</mi></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>,</mo><msub><mi>y</mi><mi>w</mi></msub><mo>,</mo><msub><mi>y</mi><mi>l</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>)</mo></mrow></mrow></mrow><mo>]</mo></mrow></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(6.4)</span></div>

其中「隐式 reward margin」<math alttext="h_{\theta}" display="inline"><semantics><msub><mi>h</mi><mi>θ</mi></msub></semantics></math> 为：

<div class="hh-equation" id="ch6.e5"><math alttext="h_{\theta}(x,y_{w},y_{l})=\underbrace{\log\frac{\pi_{\theta}(y_{w}|x)}{\pi_{\text{ref}}(y_{w}|x)}}_{\text{chosen reward proxy}}-\underbrace{\log\frac{\pi_{\theta}(y_{l}|x)}{\pi_{\text{ref}}(y_{l}|x)}}_{\text{rejected reward proxy}}" display="block"><semantics><mrow><mrow><msub><mi>h</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>,</mo><msub><mi>y</mi><mi>w</mi></msub><mo>,</mo><msub><mi>y</mi><mi>l</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><munder><munder accentunder="true"><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mfrac><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>w</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mi>π</mi><mtext>ref</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>w</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow></mfrac></mrow><mo stretchy="true">⏟</mo></munder><mtext>chosen reward proxy</mtext></munder><mo>−</mo><munder><munder accentunder="true"><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mfrac><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>l</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mi>π</mi><mtext>ref</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>l</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow></mfrac></mrow><mo stretchy="true">⏟</mo></munder><mtext>rejected reward proxy</mtext></munder></mrow></mrow></semantics></math><span class="hh-equation-number">(6.5)</span></div>

展开到 token 级别：

<div class="hh-equation" id="ch6.e6"><math alttext="\boxed{h_{\theta}=\sum_{t=1}^{|y_{w}|}\!\left[\log\pi_{\theta}(y_{w}^{t}|x,y_{w}^{&lt;t})-\log\pi_{\text{ref}}(y_{w}^{t}|x,y_{w}^{&lt;t})\right]-\sum_{t=1}^{|y_{l}|}\!\left[\log\pi_{\theta}(y_{l}^{t}|x,y_{l}^{&lt;t})-\log\pi_{\text{ref}}(y_{l}^{t}|x,y_{l}^{&lt;t})\right]}" display="block"><semantics><menclose notation="box"><mrow><msub><mi>h</mi><mi>θ</mi></msub><mo rspace="0.111em">=</mo><mrow><mrow><munderover><mo movablelimits="false" rspace="0em">∑</mo><mrow><mi>t</mi><mo>=</mo><mn>1</mn></mrow><mrow><mo stretchy="false">|</mo><msub><mi>y</mi><mi>w</mi></msub><mo stretchy="false">|</mo></mrow></munderover><mrow><mo>[</mo><mrow><mrow><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>π</mi><mi>θ</mi></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msubsup><mi>y</mi><mi>w</mi><mi>t</mi></msubsup><mo lspace="0em" rspace="0em">|</mo><mrow><mi>x</mi><mo>,</mo><msubsup><mi>y</mi><mi>w</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>t</mi></mrow></msubsup></mrow><mo stretchy="false">)</mo></mrow></mrow><mo>−</mo><mrow><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>π</mi><mtext>ref</mtext></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msubsup><mi>y</mi><mi>w</mi><mi>t</mi></msubsup><mo lspace="0em" rspace="0em">|</mo><mrow><mi>x</mi><mo>,</mo><msubsup><mi>y</mi><mi>w</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>t</mi></mrow></msubsup></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow><mo>]</mo></mrow></mrow><mo rspace="0.055em">−</mo><mrow><munderover><mo movablelimits="false" rspace="0em">∑</mo><mrow><mi>t</mi><mo>=</mo><mn>1</mn></mrow><mrow><mo stretchy="false">|</mo><msub><mi>y</mi><mi>l</mi></msub><mo stretchy="false">|</mo></mrow></munderover><mrow><mo>[</mo><mrow><mrow><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>π</mi><mi>θ</mi></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msubsup><mi>y</mi><mi>l</mi><mi>t</mi></msubsup><mo lspace="0em" rspace="0em">|</mo><mrow><mi>x</mi><mo>,</mo><msubsup><mi>y</mi><mi>l</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>t</mi></mrow></msubsup></mrow><mo stretchy="false">)</mo></mrow></mrow><mo>−</mo><mrow><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>π</mi><mtext>ref</mtext></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msubsup><mi>y</mi><mi>l</mi><mi>t</mi></msubsup><mo lspace="0em" rspace="0em">|</mo><mrow><mi>x</mi><mo>,</mo><msubsup><mi>y</mi><mi>l</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>t</mi></mrow></msubsup></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow><mo>]</mo></mrow></mrow></mrow></mrow></menclose></semantics></math><span class="hh-equation-number">(6.6)</span></div>

### Forward Pass：逐步详解

对于一条训练样本 <math alttext="(x,y_{w},y_{l})" display="inline"><semantics><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>,</mo><msub><mi>y</mi><mi>w</mi></msub><mo>,</mo><msub><mi>y</mi><mi>l</mi></msub><mo stretchy="false">)</mo></mrow></semantics></math>：

<ol><li>拼接：构造两条序列：<math alttext="[x;y_{w}]" display="inline"><semantics><mrow><mo stretchy="false">[</mo><mi>x</mi><mo>;</mo><msub><mi>y</mi><mi>w</mi></msub><mo stretchy="false">]</mo></mrow></semantics></math> 和 <math alttext="[x;y_{l}]" display="inline"><semantics><mrow><mo stretchy="false">[</mo><mi>x</mi><mo>;</mo><msub><mi>y</mi><mi>l</mi></msub><mo stretchy="false">]</mo></mrow></semantics></math>。在 batch 内补齐到相同长度。</li><li>前向传播（policy <math alttext="\pi_{\theta}" display="inline"><semantics><msub><mi>π</mi><mi>θ</mi></msub></semantics></math>）：让两条序列都通过模型。收集每个回答位置上的 logits。</li><li>提取对数概率：在回答的每个位置 <math alttext="t" display="inline"><semantics><mi>t</mi></semantics></math>，取 <math alttext="\log\text{softmax}(\text{logits}_{t})[y_{t}]" display="inline"><semantics><mrow><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mtext>softmax</mtext></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mtext>logits</mtext><mi>t</mi></msub><mo stretchy="false">)</mo></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">[</mo><msub><mi>y</mi><mi>t</mi></msub><mo stretchy="false">]</mo></mrow></mrow></semantics></math>——即实际 token 的对数概率。</li><li>对 token 求和：logp_chosen<math alttext="\displaystyle=\sum_{t\in\text{response positions}}\log\pi_{\theta}(y_{w}^{t}|x,y_{w}^{&lt;t})" display="inline"><semantics><mrow><mphantom></mphantom><mo>=</mo><mrow><mstyle displaystyle="true"><munder><mo movablelimits="false">∑</mo><mrow><mi>t</mi><mo>∈</mo><mtext>response positions</mtext></mrow></munder></mstyle><mrow><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>π</mi><mi>θ</mi></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msubsup><mi>y</mi><mi>w</mi><mi>t</mi></msubsup><mo lspace="0em" rspace="0em">|</mo><mrow><mi>x</mi><mo>,</mo><msubsup><mi>y</mi><mi>w</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>t</mi></mrow></msubsup></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></semantics></math><span class="hh-tag">(6.7)</span>logp_rejected<math alttext="\displaystyle=\sum_{t\in\text{response positions}}\log\pi_{\theta}(y_{l}^{t}|x,y_{l}^{&lt;t})" display="inline"><semantics><mrow><mphantom></mphantom><mo>=</mo><mrow><mstyle displaystyle="true"><munder><mo movablelimits="false">∑</mo><mrow><mi>t</mi><mo>∈</mo><mtext>response positions</mtext></mrow></munder></mstyle><mrow><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>π</mi><mi>θ</mi></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msubsup><mi>y</mi><mi>l</mi><mi>t</mi></msubsup><mo lspace="0em" rspace="0em">|</mo><mrow><mi>x</mi><mo>,</mo><msubsup><mi>y</mi><mi>l</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>t</mi></mrow></msubsup></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></semantics></math><span class="hh-tag">(6.8)</span></li><li>减去 reference（预先计算，或来自第二次前向传播）：ratio_w<math alttext="\displaystyle=\text{logp\_chosen}-\text{ref\_logp\_chosen}" display="inline"><semantics><mrow><mphantom></mphantom><mo>=</mo><mrow><mtext>logp_chosen</mtext><mo>−</mo><mtext>ref_logp_chosen</mtext></mrow></mrow></semantics></math><span class="hh-tag">(6.9)</span>ratio_l<math alttext="\displaystyle=\text{logp\_rejected}-\text{ref\_logp\_rejected}" display="inline"><semantics><mrow><mphantom></mphantom><mo>=</mo><mrow><mtext>logp_rejected</mtext><mo>−</mo><mtext>ref_logp_rejected</mtext></mrow></mrow></semantics></math><span class="hh-tag">(6.10)</span></li><li>计算 loss：<math alttext="\mathcal{L}=-\log\sigma(\beta\cdot(\text{ratio\_w}-\text{ratio\_l}))" display="inline"><semantics><mrow><mi>ℒ</mi><mo>=</mo><mrow><mo rspace="0.167em">−</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mrow><mi>σ</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mi>β</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><mrow><mo stretchy="false">(</mo><mrow><mtext>ratio_w</mtext><mo>−</mo><mtext>ratio_l</mtext></mrow><mo stretchy="false">)</mo></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></mrow></semantics></math></li><li>反向传播：gradient 经由第 5 步 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 第 4 步 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 第 3 步 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 第 2 步回传，以更新 <math alttext="\theta" display="inline"><semantics><mi>θ</mi></semantics></math>。</li></ol>

### Token 级 Gradient 分析

每个 token 都能得到 gradient 吗？能。被选序列中位置 <math alttext="t" display="inline"><semantics><mi>t</mi></semantics></math> 处 logits 的 gradient 为：

<div class="hh-equation" id="ch6.e11"><math alttext="\frac{\partial\mathcal{L}}{\partial\text{logits}_{t}^{(w)}}=-\underbrace{\sigma(-\beta\cdot h_{\theta})}_{\text{scaling factor}}\cdot\beta\cdot\frac{\partial\log\pi_{\theta}(y_{w}^{t}|\cdot)}{\partial\text{logits}_{t}^{(w)}}" display="block"><semantics><mrow><mfrac><mrow><mo>∂</mo><mi>ℒ</mi></mrow><mrow><mo>∂</mo><msubsup><mtext>logits</mtext><mi>t</mi><mrow><mo stretchy="false">(</mo><mi>w</mi><mo stretchy="false">)</mo></mrow></msubsup></mrow></mfrac><mo rspace="0em">=</mo><mo lspace="0em">−</mo><munder accentunder="true"><mrow><mi>σ</mi><mo stretchy="false">(</mo><mo lspace="0em">−</mo><mi>β</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><mi>h</mi><msub><mrow></mrow><mi>θ</mi></msub><mo rspace="0.055em" stretchy="false">)</mo></mrow><mo stretchy="true">⏟</mo></munder><msub><mrow></mrow><mtext>scaling factor</mtext></msub><mo rspace="0.222em">⋅</mo><mi>β</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><mfrac><mrow><mo rspace="0.167em">∂</mo><mi>log</mi><mi>π</mi><msub><mrow></mrow><mi>θ</mi></msub><mo stretchy="false">(</mo><mi>y</mi><msub><mrow></mrow><mi>w</mi></msub><msup><mrow></mrow><mi>t</mi></msup><mo fence="false" stretchy="false">|</mo><mo lspace="0em" rspace="0em">⋅</mo><mo stretchy="false">)</mo></mrow><mrow><mo>∂</mo><msubsup><mtext>logits</mtext><mi>t</mi><mrow><mo stretchy="false">(</mo><mi>w</mi><mo stretchy="false">)</mo></mrow></msubsup></mrow></mfrac></mrow></semantics></math><span class="hh-equation-number">(6.11)</span></div>

关键洞察：缩放因子 <math alttext="\sigma(-\beta\cdot h_{\theta})" display="inline"><semantics><mrow><mi>σ</mi><mrow><mo stretchy="false">(</mo><mo lspace="0em">−</mo><mi>β</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><msub><mi>h</mi><mi>θ</mi></msub><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 在两条序列中为所有 token 共享。它起到自适应学习率的作用：

<ul><li>当 <math alttext="h_{\theta}" display="inline"><semantics><msub><mi>h</mi><mi>θ</mi></msub></semantics></math> 很小（模型无法区分被选与被拒）时：缩放为 <math alttext="\approx 0.5" display="inline"><semantics><mrow><mphantom></mphantom><mo>≈</mo><mn>0.5</mn></mrow></semantics></math>——gradient 强，学习激进。</li><li>当 <math alttext="h_{\theta}" display="inline"><semantics><msub><mi>h</mi><mi>θ</mi></msub></semantics></math> 很大（模型已经偏好被选）时：缩放为 <math alttext="\approx 0" display="inline"><semantics><mrow><mphantom></mphantom><mo>≈</mo><mn>0</mn></mrow></semantics></math>——gradient 可忽略，不要过拟合。</li></ul>

对被选 token 的影响：概率被<em>提升</em>（对数概率被推高）。<br>

对被拒 token 的影响：概率被<em>降低</em>（对数概率被压低）。<br>

相对于 reference：只有与 <math alttext="\pi_{\text{ref}}" display="inline"><semantics><msub><mi>π</mi><mtext>ref</mtext></msub></semantics></math> 的<em>差值</em>才重要。如果模型已经给被选回答分配了高概率（与 reference 一致），gradient 就很小。

### 逐 Token vs. 序列级：长度归一化

一个微妙的问题：更长的序列天然具有更低的对数概率（求和的项更多，每项 <math alttext="\leq 0" display="inline"><semantics><mrow><mphantom></mphantom><mo>≤</mo><mn>0</mn></mrow></semantics></math>）。如果 <math alttext="|y_{w}|\gg|y_{l}|" display="inline"><semantics><mrow><mrow><mo stretchy="false">|</mo><msub><mi>y</mi><mi>w</mi></msub><mo stretchy="false">|</mo></mrow><mo>≫</mo><mrow><mo stretchy="false">|</mo><msub><mi>y</mi><mi>l</mi></msub><mo stretchy="false">|</mo></mrow></mrow></semantics></math>，loss 可能偏向于偏好更短的回答。

解决方案：

<ul><li>长度归一化 DPO：用 <math alttext="\frac{1}{|y|}\sum_{t}\log\pi_{\theta}(y_{t}|\cdot)" display="inline"><semantics><mrow><mfrac><mn>1</mn><mrow><mo stretchy="false">|</mo><mi>y</mi><mo stretchy="false">|</mo></mrow></mfrac><msub><mo>∑</mo><mi>t</mi></msub><mi>log</mi><msub><mi>π</mi><mi>θ</mi></msub><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>t</mi></msub><mo fence="false" stretchy="false">|</mo><mo lspace="0em" rspace="0em">⋅</mo><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 替换 <math alttext="\log\pi_{\theta}(y|x)" display="inline"><semantics><mrow><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>π</mi><mi>θ</mi></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>y</mi><mo lspace="0em" rspace="0em">|</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math>。部分实现采用了这一做法（SimPO 即采用它）。</li><li>标准 DPO：使用原始求和（不做归一化）。这会<em>隐式地</em>惩罚冗长——模型必须为被选回答中的每个 token 赋予高概率。</li><li>实际影响：在基准测试上，长度归一化 DPO 减少了长度博弈，但可能损害指令遵循质量。标准（未归一化）版本在生产中更常见。</li></ul>

### 标签掩码：哪些 token 会收到 Gradient

### 伪代码：DPO 训练步

### 常见陷阱

## DPO 变体及其各自的失败场景

## β\beta 选择指南

<math alttext="\beta" display="inline"><semantics><mi>β</mi></semantics></math>档位适用场景0.01非常激进仅当数据极其干净且你需要较大的分布偏移时0.05激进数据质量好，希望在 SFT 基础上有明显提升0.1标准默认起点。质量与稳定性之间平衡良好0.2保守数据有噪声，或模型已接近期望行为0.5非常保守安全性微调，且不能破坏模型既有能力

## DPO 的 Batch Size 配置与扩展

与在单序列 token 预测上运行的标准 SFT 不同，DPO 利用成对 loss，将被偏好序列与不被偏好序列进行比较。这从根本上改变了显存占用与优化稳定性。

### 全局 Batch Size 目标

跨多种 DPO 实现的经验证据确立了一个最优的全局 batch size 区间：

<div class="hh-equation" id="ch6.e12"><math alttext="\boxed{B_{\text{global}}\in[32,128]}" display="block"><semantics><menclose notation="box"><mrow><msub><mi>B</mi><mtext>global</mtext></msub><mo>∈</mo><mrow><mo stretchy="false">[</mo><mn>32,128</mn><mo stretchy="false">]</mo></mrow></mrow></menclose></semantics></math><span class="hh-equation-number">(6.12)</span></div>

<ul><li><math alttext="B_{\text{global}}&lt;32" display="inline"><semantics><mrow><msub><mi>B</mi><mtext>global</mtext></msub><mo>&lt;</mo><mn>32</mn></mrow></semantics></math>：隐式 reward 估计中的 gradient 噪声严重 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> policy 在对齐目标之间（有用性 vs. 安全性）破坏性地来回振荡。</li><li><math alttext="B_{\text{global}}&gt;128" display="inline"><semantics><mrow><msub><mi>B</mi><mtext>global</mtext></msub><mo>&gt;</mo><mn>128</mn></mrow></semantics></math>：收敛速度收益递减；分布式计算中的通信开销很高。</li></ul>

### 数学分解

因为 DPO 会同时加载两份模型副本（活动 policy <math alttext="\pi_{\theta}" display="inline"><semantics><msub><mi>π</mi><mi>θ</mi></msub></semantics></math> + 冻结的 reference <math alttext="\pi_{\text{ref}}" display="inline"><semantics><msub><mi>π</mi><mtext>ref</mtext></msub></semantics></math>），每条序列的显存占用翻倍。全局 batch size 可分解为：

<div class="hh-equation" id="ch6.e13"><math alttext="\boxed{B_{\text{global}}=B_{\text{micro}}\times N_{\text{GPUs}}\times K_{\text{accum}}}" display="block"><semantics><menclose notation="box"><mrow><msub><mi>B</mi><mtext>global</mtext></msub><mo>=</mo><mrow><msub><mi>B</mi><mtext>micro</mtext></msub><mo lspace="0.222em" rspace="0.222em">×</mo><msub><mi>N</mi><mtext>GPUs</mtext></msub><mo lspace="0.222em" rspace="0.222em">×</mo><msub><mi>K</mi><mtext>accum</mtext></msub></mrow></mrow></menclose></semantics></math><span class="hh-equation-number">(6.13)</span></div>

<ul><li><math alttext="B_{\text{micro}}" display="inline"><semantics><msub><mi>B</mi><mtext>micro</mtext></msub></semantics></math>：单设备微 batch 大小（每次前向传播处理的偏好成对样本数）。</li><li><math alttext="N_{\text{GPUs}}" display="inline"><semantics><msub><mi>N</mi><mtext>GPUs</mtext></msub></semantics></math>：并行数据处理设备数量。</li><li><math alttext="K_{\text{accum}}" display="inline"><semantics><msub><mi>K</mi><mtext>accum</mtext></msub></semantics></math>：权重更新前的 gradient 累积步数。</li></ul>

成对倍增因子：单个 DPO 数据实例包含一个 prompt（<math alttext="x" display="inline"><semantics><mi>x</mi></semantics></math>）、一个被选回答（<math alttext="y_{w}" display="inline"><semantics><msub><mi>y</mi><mi>w</mi></msub></semantics></math>）和一个被拒回答（<math alttext="y_{l}" display="inline"><semantics><msub><mi>y</mi><mi>l</mi></msub></semantics></math>）。每个微 batch 实际产生的张量负载为：

<div class="hh-equation" id="ch6.e14"><math alttext="T_{\text{sequences}}=2\times B_{\text{micro}}" display="block"><semantics><mrow><msub><mi>T</mi><mtext>sequences</mtext></msub><mo>=</mo><mrow><mn>2</mn><mo lspace="0.222em" rspace="0.222em">×</mo><msub><mi>B</mi><mtext>micro</mtext></msub></mrow></mrow></semantics></math><span class="hh-equation-number">(6.14)</span></div>

对于 <math alttext="&gt;" display="inline"><semantics><mo>&gt;</mo></semantics></math>7B 参数规模的模型，在 80GB GPU 上以 4096–8192 token 的上下文长度运行时，物理上限被严格约束在 <math alttext="B_{\text{micro}}\in[1,2]" display="inline"><semantics><mrow><msub><mi>B</mi><mtext>micro</mtext></msub><mo>∈</mo><mrow><mo stretchy="false">[</mo><mrow><mn>1</mn><mo>,</mo><mn>2</mn></mrow><mo stretchy="false">]</mo></mrow></mrow></semantics></math>。

### 分布式扩展配置

<div class="hh-table" id="ch6.t1"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>配置</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>单 GPU</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>8-GPU 节点</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left"><math alttext="B_{\text{global}}" display="inline"><semantics><msub><mi>B</mi><mtext>global</mtext></msub><span></span></semantics></math></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>64</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>64</span></span></td></tr><tr><th class="ltx_align_left"><math alttext="B_{\text{micro}}" display="inline"><semantics><msub><mi>B</mi><mtext>micro</mtext></msub><span></span></semantics></math></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>2（4 条序列）</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>2（4 条序列）</span></span></td></tr><tr><th class="ltx_align_left"><math alttext="N_{\text{GPUs}}" display="inline"><semantics><msub><mi>N</mi><mtext>GPUs</mtext></msub><span></span></semantics></math></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>1</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>8</span></span></td></tr><tr><th class="ltx_align_left"><math alttext="K_{\text{accum}}" display="inline"><semantics><msub><mi>K</mi><mtext>accum</mtext></msub><span></span></semantics></math></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>32 步</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>4 步</span></span></td></tr><tr><th class="ltx_align_left">吞吐量</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>串行/慢</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>高并行吞吐</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">表 6.1：</span>DPO 训练的分布式扩展配置（<math alttext="B_{\text{global}}=64" display="inline"><semantics><mrow><msub><mi>B</mi><mtext>global</mtext></msub><mo>=</mo><mn>64</mn></mrow></semantics></math> 目标）。</p></div>

### 显存优化：预计算 Reference 对数概率

DPO loss 为：

<div class="hh-equation" id="ch6.e15"><math alttext="\mathcal{L}_{\text{DPO}}(\theta)=-\mathbb{E}_{(x,y_{w},y_{l})}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_{\theta}(y_{w}|x)}{\pi_{\text{ref}}(y_{w}|x)}-\beta\log\frac{\pi_{\theta}(y_{l}|x)}{\pi_{\text{ref}}(y_{l}|x)}\right)\right]" display="block"><semantics><mrow><mrow><msub><mi>ℒ</mi><mtext>DPO</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mo>−</mo><mrow><msub><mi>𝔼</mi><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>,</mo><msub><mi>y</mi><mi>w</mi></msub><mo>,</mo><msub><mi>y</mi><mi>l</mi></msub><mo stretchy="false">)</mo></mrow></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo>[</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mrow><mpadded width="0.437em"><mi>σ</mi></mpadded><mo>⁡</mo><mrow><mo>(</mo><mrow><mrow><mi>β</mi><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mfrac><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>w</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mi>π</mi><mtext>ref</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>w</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow></mfrac></mrow></mrow><mo>−</mo><mrow><mi>β</mi><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mfrac><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>l</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mi>π</mi><mtext>ref</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>l</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow></mfrac></mrow></mrow></mrow><mo>)</mo></mrow></mrow></mrow><mo>]</mo></mrow></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(6.15)</span></div>

因为 <math alttext="\pi_{\text{ref}}" display="inline"><semantics><msub><mi>π</mi><mtext>ref</mtext></msub></semantics></math> 在整个训练过程中完全静态，其输出可以预先计算：

## DPO 扩展与变体

直接偏好优化（Direct Preference Optimization，DPO）通过推导 reward 函数与最优 policy 之间的解析映射，将 RLHF 重新表述为一个监督学习问题。标准 DPO loss 为：

<div class="hh-equation" id="ch6.ex1"><math alttext="\mathcal{L}_{\text{DPO}}(\theta)=-\mathbb{E}_{(q,y_{w},y_{l})}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_{\theta}(y_{w}|q)}{\pi_{\text{ref}}(y_{w}|q)}-\beta\log\frac{\pi_{\theta}(y_{l}|q)}{\pi_{\text{ref}}(y_{l}|q)}\right)\right]," display="block"><semantics><mrow><mrow><mrow><msub><mi>ℒ</mi><mtext>DPO</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mo>−</mo><mrow><msub><mi>𝔼</mi><mrow><mo stretchy="false">(</mo><mi>q</mi><mo>,</mo><msub><mi>y</mi><mi>w</mi></msub><mo>,</mo><msub><mi>y</mi><mi>l</mi></msub><mo stretchy="false">)</mo></mrow></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo>[</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mrow><mpadded width="0.437em"><mi>σ</mi></mpadded><mo>⁡</mo><mrow><mo>(</mo><mrow><mrow><mi>β</mi><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mfrac><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>w</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mi>π</mi><mtext>ref</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>w</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow></mfrac></mrow></mrow><mo>−</mo><mrow><mi>β</mi><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mfrac><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>l</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mi>π</mi><mtext>ref</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>l</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow></mfrac></mrow></mrow></mrow><mo>)</mo></mrow></mrow></mrow><mo>]</mo></mrow></mrow></mrow></mrow><mo>,</mo></mrow></semantics></math></div>

其中 <math alttext="y_{w}" display="inline"><semantics><msub><mi>y</mi><mi>w</mi></msub></semantics></math> 是被偏好（胜出）的回答，<math alttext="y_{l}" display="inline"><semantics><msub><mi>y</mi><mi>l</mi></msub></semantics></math> 是不被偏好（落败）的回答，<math alttext="\beta" display="inline"><semantics><mi>β</mi></semantics></math> 控制 KL 惩罚的强度。以下小节介绍最重要的扩展与变体。

### f-DPO——广义 f-Divergence DPO

f-DPO loss 用 f-散度生成函数的导数替换对数比：

<div class="hh-equation" id="ch6.ex2"><math alttext="\mathcal{L}_{f\text{-DPO}}=-\mathbb{E}\!\left[f^{\prime}\!\left(\frac{\pi_{\theta}(y_{w}|q)}{\pi_{\text{ref}}(y_{w}|q)}\right)-f^{\prime}\!\left(\frac{\pi_{\theta}(y_{l}|q)}{\pi_{\text{ref}}(y_{l}|q)}\right)\right]," display="block"><semantics><mrow><mrow><msub><mi>ℒ</mi><mrow><mi>f</mi><mo lspace="0em" rspace="0em">​</mo><mtext>-DPO</mtext></mrow></msub><mo>=</mo><mrow><mo>−</mo><mrow><mpadded width="0.497em"><mi>𝔼</mi></mpadded><mo>⁡</mo><mrow><mo>[</mo><mrow><mrow><msup><mi>f</mi><mo>′</mo></msup><mo lspace="0em" rspace="0em">​</mo><mrow><mo>(</mo><mfrac><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>w</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mi>π</mi><mtext>ref</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>w</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow></mfrac><mo>)</mo></mrow></mrow><mo>−</mo><mrow><msup><mi>f</mi><mo>′</mo></msup><mo lspace="0em" rspace="0em">​</mo><mrow><mo>(</mo><mfrac><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>l</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mi>π</mi><mtext>ref</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>l</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow></mfrac><mo>)</mo></mrow></mrow></mrow><mo>]</mo></mrow></mrow></mrow></mrow><mo>,</mo></mrow></semantics></math></div>

其中 <math alttext="f^{\prime}" display="inline"><semantics><msup><mi>f</mi><mo>′</mo></msup></semantics></math> 是 f-散度生成函数的导数。

### Robust DPO

假设每个标签以概率 <math alttext="\epsilon" display="inline"><semantics><mi>ϵ</mi></semantics></math>（噪声率）被翻转。去偏后的 loss 为：

<div class="hh-equation" id="ch6.ex3"><math alttext="\boxed{\mathcal{L}_{\text{robust}}=\frac{(1-\epsilon)\,\mathcal{L}_{\text{DPO}}(y_{w},y_{l})-\epsilon\,\mathcal{L}_{\text{DPO}}(y_{l},y_{w})}{1-2\epsilon},}" display="block"><semantics><menclose notation="box"><mrow><mrow><msub><mi>ℒ</mi><mtext>robust</mtext></msub><mo>=</mo><mfrac><mrow><mrow><mrow><mo stretchy="false">(</mo><mrow><mn>1</mn><mo>−</mo><mi>ϵ</mi></mrow><mo stretchy="false">)</mo></mrow><mo lspace="0.170em" rspace="0em">​</mo><msub><mi>ℒ</mi><mtext>DPO</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>w</mi></msub><mo>,</mo><msub><mi>y</mi><mi>l</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>−</mo><mrow><mi>ϵ</mi><mo lspace="0.170em" rspace="0em">​</mo><msub><mi>ℒ</mi><mtext>DPO</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>l</mi></msub><mo>,</mo><msub><mi>y</mi><mi>w</mi></msub><mo stretchy="false">)</mo></mrow></mrow></mrow><mrow><mn>1</mn><mo>−</mo><mrow><mn>2</mn><mo lspace="0em" rspace="0em">​</mo><mi>ϵ</mi></mrow></mrow></mfrac></mrow><mo>,</mo></mrow></menclose></semantics></math></div>

其中 <math alttext="\mathcal{L}_{\text{DPO}}(y_{w},y_{l})" display="inline"><semantics><mrow><msub><mi>ℒ</mi><mtext>DPO</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>w</mi></msub><mo>,</mo><msub><mi>y</mi><mi>l</mi></msub><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 是把 <math alttext="y_{w}" display="inline"><semantics><msub><mi>y</mi><mi>w</mi></msub></semantics></math> 视为被偏好回答的标准 DPO loss，<math alttext="\mathcal{L}_{\text{DPO}}(y_{l},y_{w})" display="inline"><semantics><mrow><msub><mi>ℒ</mi><mtext>DPO</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>l</mi></msub><mo>,</mo><msub><mi>y</mi><mi>w</mi></msub><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 是标签翻转后的 loss。这一修正去除了标签噪声引入的偏差。

### TR-DPO——信任域 DPO

TR-DPO 使用指数移动平均（EMA）更新 reference 模型：

<div class="hh-equation" id="ch6.ex4"><math alttext="\pi_{\text{ref}}^{(t+1)}\leftarrow\alpha\cdot\pi_{\theta}^{(t)}+(1-\alpha)\cdot\pi_{\text{ref}}^{(t)}," display="block"><semantics><mrow><mrow><msubsup><mi>π</mi><mtext>ref</mtext><mrow><mo stretchy="false">(</mo><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow><mo stretchy="false">)</mo></mrow></msubsup><mo stretchy="false">←</mo><mrow><mrow><mi>α</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><msubsup><mi>π</mi><mi>θ</mi><mrow><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo></mrow></msubsup></mrow><mo>+</mo><mrow><mrow><mo stretchy="false">(</mo><mrow><mn>1</mn><mo>−</mo><mi>α</mi></mrow><mo rspace="0.055em" stretchy="false">)</mo></mrow><mo rspace="0.222em">⋅</mo><msubsup><mi>π</mi><mtext>ref</mtext><mrow><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo></mrow></msubsup></mrow></mrow></mrow><mo>,</mo></mrow></semantics></math></div>

其中 <math alttext="\alpha\in(0,1)" display="inline"><semantics><mrow><mi>α</mi><mo>∈</mo><mrow><mo stretchy="false">(</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 是混合系数。每 <math alttext="T_{\text{sync}}" display="inline"><semantics><msub><mi>T</mi><mtext>sync</mtext></msub></semantics></math> 个 gradient 步执行一次。

### EXO——精确优化

EXO 最小化模型分布与目标（reward 最优）分布之间的反向 KL：

<div class="hh-equation" id="ch6.ex5"><math alttext="\mathcal{L}_{\text{EXO}}=\mathbb{E}_{y\sim\pi_{\theta}}\!\left[\log\frac{\pi_{\theta}(y|q)}{p^{*}(y|q)}\right]," display="block"><semantics><mrow><mrow><msub><mi>ℒ</mi><mtext>EXO</mtext></msub><mo>=</mo><mrow><msub><mi>𝔼</mi><mrow><mi>y</mi><mo>∼</mo><msub><mi>π</mi><mi>θ</mi></msub></mrow></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo>[</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mfrac><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>y</mi><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mrow><msup><mi>p</mi><mo>∗</mo></msup><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>y</mi><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow></mfrac></mrow><mo>]</mo></mrow></mrow></mrow><mo>,</mo></mrow></semantics></math></div>

其中 <math alttext="p^{*}(y|q)\propto\pi_{\text{ref}}(y|q)\exp(r(y,q)/\beta)" display="inline"><semantics><mrow><mrow><msup><mi>p</mi><mo>∗</mo></msup><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>y</mi><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mo>∝</mo><mrow><msub><mi>π</mi><mtext>ref</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>y</mi><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>exp</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mrow><mi>r</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>y</mi><mo>,</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mo>/</mo><mi>β</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></semantics></math> 是最优 policy。实践中，EXO 利用可用的偏好成对样本来近似它：

<div class="hh-equation" id="ch6.ex6"><math alttext="\mathcal{L}_{\text{EXO}}\approx-\mathbb{E}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_{\text{ref}}(y_{w}|q)}{\pi_{\theta}(y_{w}|q)}-\beta\log\frac{\pi_{\text{ref}}(y_{l}|q)}{\pi_{\theta}(y_{l}|q)}\right)\right]." display="block"><semantics><mrow><mrow><msub><mi>ℒ</mi><mtext>EXO</mtext></msub><mo>≈</mo><mrow><mo>−</mo><mrow><mpadded width="0.497em"><mi>𝔼</mi></mpadded><mo>⁡</mo><mrow><mo>[</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mrow><mpadded width="0.437em"><mi>σ</mi></mpadded><mo>⁡</mo><mrow><mo>(</mo><mrow><mrow><mi>β</mi><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mfrac><mrow><msub><mi>π</mi><mtext>ref</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>w</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>w</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow></mfrac></mrow></mrow><mo>−</mo><mrow><mi>β</mi><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mfrac><mrow><msub><mi>π</mi><mtext>ref</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>l</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>l</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow></mfrac></mrow></mrow></mrow><mo>)</mo></mrow></mrow></mrow><mo>]</mo></mrow></mrow></mrow></mrow><mo lspace="0em">.</mo></mrow></semantics></math></div>

注意与 DPO 相比，<math alttext="\pi_{\theta}" display="inline"><semantics><msub><mi>π</mi><mi>θ</mi></msub></semantics></math> 与 <math alttext="\pi_{\text{ref}}" display="inline"><semantics><msub><mi>π</mi><mtext>ref</mtext></msub></semantics></math> 的角色是<em>对调</em>的。

### NCA——噪声对比对齐

NCA 把对齐重新表述为噪声对比估计。loss 包含三项：

<div class="hh-equation" id="ch6.ex7"><math alttext="\boxed{\mathcal{L}_{\text{NCA}}=-\log\sigma(r_{w})-\tfrac{1}{2}\log\sigma(-r_{w})-\tfrac{1}{2}\log\sigma(-r_{l}),}" display="block"><semantics><menclose notation="box"><mrow><mrow><msub><mi>ℒ</mi><mtext>NCA</mtext></msub><mo>=</mo><mrow><mrow><mo rspace="0.167em">−</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mrow><mi>σ</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>r</mi><mi>w</mi></msub><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow><mo>−</mo><mrow><mstyle displaystyle="false"><mfrac><mn>1</mn><mn>2</mn></mfrac></mstyle><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mrow><mi>σ</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mo>−</mo><msub><mi>r</mi><mi>w</mi></msub></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow><mo>−</mo><mrow><mstyle displaystyle="false"><mfrac><mn>1</mn><mn>2</mn></mfrac></mstyle><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mrow><mi>σ</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mo>−</mo><msub><mi>r</mi><mi>l</mi></msub></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></mrow></mrow><mo>,</mo></mrow></menclose></semantics></math></div>

其中 <math alttext="r_{y}=\beta\log(\pi_{\theta}(y|q)/\pi_{\text{ref}}(y|q))" display="inline"><semantics><mrow><msub><mi>r</mi><mi>y</mi></msub><mo>=</mo><mrow><mi>β</mi><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mrow><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>y</mi><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mo>/</mo><msub><mi>π</mi><mtext>ref</mtext></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>y</mi><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></semantics></math> 是隐式 reward。第一项鼓励 <math alttext="y_{w}" display="inline"><semantics><msub><mi>y</mi><mi>w</mi></msub></semantics></math> 获得高 reward；第二、三项则惩罚 <math alttext="y_{w}" display="inline"><semantics><msub><mi>y</mi><mi>w</mi></msub></semantics></math> 和 <math alttext="y_{l}" display="inline"><semantics><msub><mi>y</mi><mi>l</mi></msub></semantics></math> 获得高 reward（防止崩塌）。

### SLiC-HF——序列似然校准

SLiC-HF loss 为：

<div class="hh-equation" id="ch6.ex8"><math alttext="\mathcal{L}_{\text{SLiC}}=\max\!\left(0,\;\delta-\beta\log\frac{\pi_{\theta}(y_{w}|q)}{\pi_{\text{ref}}(y_{w}|q)}+\beta\log\frac{\pi_{\theta}(y_{l}|q)}{\pi_{\text{ref}}(y_{l}|q)}\right)," display="block"><semantics><mrow><mrow><msub><mi>ℒ</mi><mtext>SLiC</mtext></msub><mo>=</mo><mrow><mpadded width="1.808em"><mi>max</mi></mpadded><mo>⁡</mo><mrow><mo>(</mo><mn>0</mn><mo>,</mo><mrow><mrow><mi>δ</mi><mo>−</mo><mrow><mi>β</mi><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mfrac><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>w</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mi>π</mi><mtext>ref</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>w</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow></mfrac></mrow></mrow></mrow><mo>+</mo><mrow><mi>β</mi><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mfrac><mrow><msub><mi>π</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>l</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mi>π</mi><mtext>ref</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>l</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow></mfrac></mrow></mrow></mrow><mo>)</mo></mrow></mrow></mrow><mo>,</mo></mrow></semantics></math></div>

其中 <math alttext="\delta" display="inline"><semantics><mi>δ</mi></semantics></math> 是 margin 阈值。当模型在胜出与落败回答之间已经给出 <math alttext="\delta" display="inline"><semantics><mi>δ</mi></semantics></math> 的 margin 时，loss 为零。

### Iterative RPO——推理偏好优化

RPO loss 结合了 DPO 与 SFT：

<div class="hh-equation" id="ch6.ex9"><math alttext="\mathcal{L}_{\text{RPO}}=\lambda_{1}\mathcal{L}_{\text{DPO}}(y_{w},y_{l})+\lambda_{2}\mathcal{L}_{\text{NLL}}(y_{w})," display="block"><semantics><mrow><mrow><msub><mi>ℒ</mi><mtext>RPO</mtext></msub><mo>=</mo><mrow><mrow><msub><mi>λ</mi><mn>1</mn></msub><mo lspace="0em" rspace="0em">​</mo><msub><mi>ℒ</mi><mtext>DPO</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>w</mi></msub><mo>,</mo><msub><mi>y</mi><mi>l</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>+</mo><mrow><msub><mi>λ</mi><mn>2</mn></msub><mo lspace="0em" rspace="0em">​</mo><msub><mi>ℒ</mi><mtext>NLL</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>w</mi></msub><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow><mo>,</mo></mrow></semantics></math></div>

其中 <math alttext="\mathcal{L}_{\text{NLL}}(y_{w})=-\log\pi_{\theta}(y_{w}|q)" display="inline"><semantics><mrow><mrow><msub><mi>ℒ</mi><mtext>NLL</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>w</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mo rspace="0.167em">−</mo><mrow><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>π</mi><mi>θ</mi></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>w</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></semantics></math> 是作用于胜出回答的标准语言建模 loss。

### SimPO——简化版偏好优化

SimPO 把隐式 reward 定义为：

<div class="hh-equation" id="ch6.ex10"><math alttext="r_{\text{SimPO}}(y|q)=\frac{\beta}{|y|}\log\pi_{\theta}(y|q)," display="block"><semantics><mrow><mrow><mrow><msub><mi>r</mi><mtext>SimPO</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>y</mi><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mfrac><mi>β</mi><mrow><mo stretchy="false">|</mo><mi>y</mi><mo stretchy="false">|</mo></mrow></mfrac><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>π</mi><mi>θ</mi></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>y</mi><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow></mrow><mo>,</mo></mrow></semantics></math></div>

loss 定义为：

<div class="hh-equation" id="ch6.ex11"><math alttext="\boxed{\mathcal{L}_{\text{SimPO}}=-\mathbb{E}\!\left[\log\sigma\!\left(\frac{\beta}{|y_{w}|}\log\pi_{\theta}(y_{w}|q)-\frac{\beta}{|y_{l}|}\log\pi_{\theta}(y_{l}|q)-\gamma\right)\right],}" display="block"><semantics><menclose notation="box"><mrow><mrow><msub><mi>ℒ</mi><mtext>SimPO</mtext></msub><mo>=</mo><mrow><mo>−</mo><mrow><mpadded width="0.497em"><mi>𝔼</mi></mpadded><mo>⁡</mo><mrow><mo>[</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mrow><mpadded width="0.437em"><mi>σ</mi></mpadded><mo>⁡</mo><mrow><mo>(</mo><mrow><mrow><mfrac><mi>β</mi><mrow><mo stretchy="false">|</mo><msub><mi>y</mi><mi>w</mi></msub><mo stretchy="false">|</mo></mrow></mfrac><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>π</mi><mi>θ</mi></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>w</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mo>−</mo><mrow><mfrac><mi>β</mi><mrow><mo stretchy="false">|</mo><msub><mi>y</mi><mi>l</mi></msub><mo stretchy="false">|</mo></mrow></mfrac><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>π</mi><mi>θ</mi></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>l</mi></msub><mo lspace="0em" rspace="0em">|</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mo>−</mo><mi>γ</mi></mrow><mo>)</mo></mrow></mrow></mrow><mo>]</mo></mrow></mrow></mrow></mrow><mo>,</mo></mrow></menclose></semantics></math></div>

其中 <math alttext="\gamma&gt;0" display="inline"><semantics><mrow><mi>γ</mi><mo>&gt;</mo><mn>0</mn></mrow></semantics></math> 是目标 reward margin，它确保胜出回答的 reward 至少比落败回答高出 <math alttext="\gamma" display="inline"><semantics><mi>γ</mi></semantics></math>。

<aside class="hh-next">
<p class="hh-next-label">下一篇</p>
<p class="hh-next-title">智能体 AI 漫游指南 · 第 7 章 GRPO ——组相对策略优化</p>
<p class="hh-next-hint">在「智能体 AI 漫游指南」分类下继续阅读。</p>
</aside>

</div>
