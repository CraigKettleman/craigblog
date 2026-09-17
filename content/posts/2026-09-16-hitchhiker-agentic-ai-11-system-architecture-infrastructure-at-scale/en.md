---
title: "The Hitchhiker's Guide to Agentic AI · Chapter 11 System Architecture & Infrastructure at Scale"
slug: "hitchhiker-agentic-ai-11-system-architecture-infrastructure-at-scale"
lang: "en"
date: "2026-09-16T00:12:00.000Z"
draft: false
featured: false
categories:
  - "hitchhiker-agentic-ai"
cover: "./cover.png"
description: "Training LLMs with reinforcement learning from human feedback is as much a systems engineering challenge as it is an algorithmic one.…"
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

Training LLMs with reinforcement learning from human feedback is as much a systems engineering challenge as it is an algorithmic one. Unlike standard supervised fine-tuning—which involves a single model, a single forward-backward pass, and well-understood scaling—RLHF requires <em>multiple models</em> (policy, reference, reward model, value head) to be loaded simultaneously, coordinated through a complex rollout-scoring-training loop, and distributed across dozens to hundreds of GPUs. This chapter covers the systems-level details that make large-scale RLHF training possible: memory budgeting, parallelism strategies (Data, Tensor, Pipeline, Sequence, and their combinations), the generation bottleneck, decoupled architectures, weight synchronization, fault tolerance, and production monitoring.

## The 4-Model Memory Challenge

<figure id="ch11.f1"><img src="./fig_031_fig31.png" alt="Figure 11.1: 70B PPO memory budget: the four models required for RLHF and their memory footprints. Total: 1470–1560GB. M" loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 11.1:</span>70B PPO memory budget: the four models required for RLHF and their memory footprints. Total: 1470–1560GB. Minimum 19–20 A100-80GB (naive). With ZeRO-3: fits in 8 nodes.</figcaption></figure>

## Parallelism Strategies in Detail

Training large language models requires distributing computation across many GPUs. There are fundamentally different <em>axes</em> along which to parallelize, each with distinct trade-offs. This section provides detailed coverage of each strategy with mathematical formulations, diagrams, and practical guidance.

<figure id="ch11.f2"><img src="./fig_032_fig32.png" alt="Figure 11.2: Overview of the four parallelism strategies. Production systems typically combine 2–3 of these simultaneous" loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 11.2:</span>Overview of the four parallelism strategies. Production systems typically combine 2–3 of these simultaneously.</figcaption></figure>

### Data Parallelism (DP) and Distributed Data Parallelism (DDP)

Data Parallelism is the simplest and most common form of distributed training [210]. Each GPU holds a <em>complete copy</em> of the model, processes a different mini-batch, and synchronizes gradients.

A single-process approach where one “master” GPU scatters input, gathers outputs, and broadcasts gradients. Limited by GIL and PCIe bandwidth to the master GPU.

Multi-process: each GPU runs its own process. Gradients are synchronized via ring-AllReduce [321] in the background while backward computation continues.

<figure id="ch11.f3"><img src="./fig_033_ddp.png" alt="Figure 11.3: DDP: each GPU holds a full model replica and processes a different batch. Gradients are averaged via ring A" loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 11.3:</span>DDP: each GPU holds a full model replica and processes a different batch. Gradients are averaged via ring AllReduce, overlapped with backward computation.</figcaption></figure>

Key properties of DDP:

<ul><li>Memory: Each GPU stores full model + optimizer + gradients. For 70B BF16: <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>560 GB/GPU—impossible without memory optimizations.</li><li>Communication: One AllReduce of gradient tensor per step. Size = model parameters <math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> 2 bytes (BF16). Ring AllReduce cost: <math alttext="2\cdot\frac{N-1}{N}\cdot M" display="inline"><semantics><mrow><mn>2</mn><mo lspace="0.222em" rspace="0.222em">⋅</mo><mfrac><mrow><mi>N</mi><mo>−</mo><mn>1</mn></mrow><mi>N</mi></mfrac><mo lspace="0.222em" rspace="0.222em">⋅</mo><mi>M</mi></mrow></semantics></math> bytes transferred per GPU.</li><li>Scaling: Near-linear up to <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>64 GPUs. Beyond that, communication starts to dominate.</li><li>Gradient bucketing: DDP groups parameters into buckets (default 25 MB) and starts AllReduce as soon as a bucket’s gradients are ready—overlapping communication with backward computation.</li></ul>

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="nccl")  # NCCL for GPU communication
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank],
            gradient_as_bucket_view=True,    # Memory optimization
            static_graph=True)               # Enable comm optimizations
```

### Tensor Parallelism (TP)

Tensor Parallelism (Megatron-LM style [327]) splits <em>individual weight matrices</em> across GPUs. Each GPU computes a partial result, and an AllReduce combines them.

The weight matrix <math alttext="W\in\mathbb{R}^{d\times h}" display="inline"><semantics><mrow><mi>W</mi><mo>∈</mo><msup><mi>ℝ</mi><mrow><mi>d</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mi>h</mi></mrow></msup></mrow></semantics></math> is split column-wise across <math alttext="T" display="inline"><semantics><mi>T</mi></semantics></math> GPUs:

<div class="hh-equation" id="ch11.e1"><math alttext="W=[W_{0}\;|\;W_{1}\;|\;\cdots\;|\;W_{T-1}],\quad W_{i}\in\mathbb{R}^{d\times h/T}" display="block"><semantics><mrow><mrow><mi>W</mi><mo>=</mo><mrow><mo stretchy="false">[</mo><mrow><msub><mi>W</mi><mn>0</mn></msub><mo fence="false" rspace="0.447em" stretchy="false">|</mo><mrow><msub><mi>W</mi><mn>1</mn></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">|</mo><mo>⋯</mo><mo stretchy="false">|</mo></mrow><mo lspace="0.280em" rspace="0em">​</mo><msub><mi>W</mi><mrow><mi>T</mi><mo>−</mo><mn>1</mn></mrow></msub></mrow></mrow><mo stretchy="false">]</mo></mrow></mrow><mo rspace="1.167em">,</mo><mrow><msub><mi>W</mi><mi>i</mi></msub><mo>∈</mo><msup><mi>ℝ</mi><mrow><mrow><mi>d</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mi>h</mi></mrow><mo>/</mo><mi>T</mi></mrow></msup></mrow></mrow></semantics></math><span class="hh-equation-number">(11.1)</span></div>

Each GPU <math alttext="i" display="inline"><semantics><mi>i</mi></semantics></math> computes <math alttext="Y_{i}=XW_{i}" display="inline"><semantics><mrow><msub><mi>Y</mi><mi>i</mi></msub><mo>=</mo><mrow><mi>X</mi><mo lspace="0em" rspace="0em">​</mo><msub><mi>W</mi><mi>i</mi></msub></mrow></mrow></semantics></math> independently (no communication). The output is split along the hidden dimension.

The weight matrix is split row-wise: <math alttext="W=[W_{0};W_{1};\ldots;W_{T-1}]" display="inline"><semantics><mrow><mi>W</mi><mo>=</mo><mrow><mo stretchy="false">[</mo><mrow><msub><mi>W</mi><mn>0</mn></msub><mo>;</mo><msub><mi>W</mi><mn>1</mn></msub><mo>;</mo><mi mathvariant="normal">…</mi><mo>;</mo><msub><mi>W</mi><mrow><mi>T</mi><mo>−</mo><mn>1</mn></mrow></msub></mrow><mo stretchy="false">]</mo></mrow></mrow></semantics></math> where <math alttext="W_{i}\in\mathbb{R}^{d/T\times h}" display="inline"><semantics><mrow><msub><mi>W</mi><mi>i</mi></msub><mo>∈</mo><msup><mi>ℝ</mi><mrow><mrow><mi>d</mi><mo>/</mo><mi>T</mi></mrow><mo lspace="0.222em" rspace="0.222em">×</mo><mi>h</mi></mrow></msup></mrow></semantics></math>. Input <math alttext="X" display="inline"><semantics><mi>X</mi></semantics></math> must also be split. Each GPU computes a partial sum, then an AllReduce produces the final output.

<figure id="ch11.f4"><img src="./fig_034_tp-column.png" alt="Figure 11.4: Column-parallel linear layer (TP=2). The weight is split column-wise; each GPU computes X​WiXW_{i} independ" loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 11.4:</span>Column-parallel linear layer (TP=2). The weight is split column-wise; each GPU computes <math alttext="XW_{i}" display="inline"><semantics><mrow><mi>X</mi><mo lspace="0em" rspace="0em">​</mo><msub><mi>W</mi><mi>i</mi></msub></mrow></semantics></math> independently. The MLP pairs this with a row-parallel layer to avoid redundant AllReduce.</figcaption></figure>

In a Transformer layer, Megatron-LM applies TP as follows:

<ol><li>MLP: Column-parallel for the first linear (<math alttext="h\to 4h" display="inline"><semantics><mrow><mi>h</mi><mo stretchy="false">→</mo><mrow><mn>4</mn><mo lspace="0em" rspace="0em">​</mo><mi>h</mi></mrow></mrow></semantics></math>), row-parallel for the second (<math alttext="4h\to h" display="inline"><semantics><mrow><mrow><mn>4</mn><mo lspace="0em" rspace="0em">​</mo><mi>h</mi></mrow><mo stretchy="false">→</mo><mi>h</mi></mrow></semantics></math>). One AllReduce after the row-parallel layer.</li><li>Attention: <math alttext="Q" display="inline"><semantics><mi>Q</mi></semantics></math>, <math alttext="K" display="inline"><semantics><mi>K</mi></semantics></math>, <math alttext="V" display="inline"><semantics><mi>V</mi></semantics></math> projections are column-parallel (split heads across GPUs). Output projection is row-parallel. One AllReduce after output projection.</li><li>Total: 2 AllReduce per transformer layer (one for attention, one for MLP).</li></ol>

<figure id="ch11.f5"><img src="./fig_035_tp-transformer.png" alt="Figure 11.5: Tensor Parallel communication pattern in one Transformer block. Two AllReduce operations (marked in red) ar" loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 11.5:</span>Tensor Parallel communication pattern in one Transformer block. Two AllReduce operations (marked in red) are required per layer—one after attention, one after MLP.</figcaption></figure>

### Sequence Parallelism (SP)

Sequence Parallelism [191] addresses a memory bottleneck that Tensor Parallelism alone cannot solve: the activation memory in LayerNorm and Dropout layers.

With TP, weight memory is split across GPUs. But LayerNorm and Dropout operate on the <em>full</em> hidden dimension and are replicated on every GPU. Their activations (needed for backward pass) consume memory proportional to <math alttext="b\times s\times d" display="inline"><semantics><mrow><mi>b</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mi>s</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mi>d</mi></mrow></semantics></math>—the same on every GPU, unreduced by TP.

Split the <em>sequence dimension</em> for operations that don’t require cross-GPU communication (LayerNorm, Dropout, residual connections). Each GPU processes a <math alttext="s/T" display="inline"><semantics><mrow><mi>s</mi><mo>/</mo><mi>T</mi></mrow></semantics></math> slice of the sequence for these operations, then gathers the full sequence only where needed (attention, linear layers).

<figure id="ch11.f6"><img src="./fig_036_seq-parallel.png" alt="Figure 11.6: Sequence Parallelism reduces activation memory for LayerNorm/Dropout by splitting along the sequence dimens" loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 11.6:</span>Sequence Parallelism reduces activation memory for LayerNorm/Dropout by splitting along the sequence dimension. Communication (AllGather/ReduceScatter) replaces the AllReduce used in standard TP—same total bytes transferred, but memory is saved.</figcaption></figure>

Memory savings from SP (70B model, TP=8, batch=4, seq=2048):

<div class="hh-equation" id="ch11.e2"><math alttext="\text{Activation savings}=(T-1)\times b\times s\times d\times n_{\text{layers}}\times 2\text{ bytes}=7\times 4\times 2048\times 8192\times 80\times 2\approx\textbf{59 GB/GPU}" display="block"><semantics><mrow><mtext>Activation savings</mtext><mo>=</mo><mrow><mrow><mo stretchy="false">(</mo><mrow><mi>T</mi><mo>−</mo><mn>1</mn></mrow><mo rspace="0.055em" stretchy="false">)</mo></mrow><mo rspace="0.222em">×</mo><mi>b</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mi>s</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mi>d</mi><mo lspace="0.222em" rspace="0.222em">×</mo><msub><mi>n</mi><mtext>layers</mtext></msub><mo lspace="0.222em" rspace="0.222em">×</mo><mrow><mn>2</mn><mo lspace="0em" rspace="0em">​</mo><mtext> bytes</mtext></mrow></mrow><mo>=</mo><mrow><mn>7</mn><mo lspace="0.222em" rspace="0.222em">×</mo><mn>4</mn><mo lspace="0.222em" rspace="0.222em">×</mo><mn>2048</mn><mo lspace="0.222em" rspace="0.222em">×</mo><mn>8192</mn><mo lspace="0.222em" rspace="0.222em">×</mo><mn>80</mn><mo lspace="0.222em" rspace="0.222em">×</mo><mn>2</mn></mrow><mo>≈</mo><mtext>59 GB/GPU</mtext></mrow></semantics></math><span class="hh-equation-number">(11.2)</span></div>

### Pipeline Parallelism (PP)

Pipeline Parallelism splits the model <em>vertically</em> by layers, assigning consecutive groups of layers to different devices (stages). Activations flow forward through stages; gradients flow backward.

Naive pipeline execution creates “bubbles”—idle time while a stage waits for input from the previous stage or gradients from the next:

<figure id="ch11.f7"><img src="./fig_037_pipeline-bubble.png" alt="Figure 11.7: Pipeline bubble comparison. Left: naive pipeline with one micro-batch has 75% idle time. Right: GPipe with " loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 11.7:</span>Pipeline bubble comparison. Left: naive pipeline with one micro-batch has 75% idle time. Right: GPipe with <math alttext="M=4" display="inline"><semantics><mrow><mi>M</mi><mo>=</mo><mn>4</mn></mrow></semantics></math> micro-batches reduces bubbles significantly. With <math alttext="M\gg P" display="inline"><semantics><mrow><mi>M</mi><mo>≫</mo><mi>P</mi></mrow></semantics></math>, bubble fraction approaches zero.</figcaption></figure>

For <math alttext="P" display="inline"><semantics><mi>P</mi></semantics></math> pipeline stages and <math alttext="M" display="inline"><semantics><mi>M</mi></semantics></math> micro-batches per step:

<div class="hh-equation" id="ch11.e3"><math alttext="\text{Bubble fraction}=\frac{P-1}{P+M-1}\approx\frac{P-1}{M}\quad\text{(when }M\gg P\text{)}" display="block"><semantics><mrow><mrow><mtext>Bubble fraction</mtext><mo>=</mo><mfrac><mrow><mi>P</mi><mo>−</mo><mn>1</mn></mrow><mrow><mrow><mi>P</mi><mo>+</mo><mi>M</mi></mrow><mo>−</mo><mn>1</mn></mrow></mfrac><mo>≈</mo><mfrac><mrow><mi>P</mi><mo>−</mo><mn>1</mn></mrow><mi>M</mi></mfrac></mrow><mspace width="1em"></mspace><mrow><mrow><mtext>(when </mtext><mo lspace="0em" rspace="0em">​</mo><mi>M</mi></mrow><mo>≫</mo><mrow><mi>P</mi><mo lspace="0em" rspace="0em">​</mo><mtext>)</mtext></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(11.3)</span></div>

To keep bubble overhead <math alttext="&lt;" display="inline"><semantics><mo>&lt;</mo></semantics></math>10%, you need <math alttext="M\geq 10\cdot(P-1)" display="inline"><semantics><mrow><mi>M</mi><mo>≥</mo><mrow><mn>10</mn><mo lspace="0.222em" rspace="0.222em">⋅</mo><mrow><mo stretchy="false">(</mo><mrow><mi>P</mi><mo>−</mo><mn>1</mn></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math>. For PP=4: at least 30 micro-batches.

<div class="hh-table" id="ch11.t1"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Schedule</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Bubble</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Memory</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Characteristics</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">GPipe</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="\frac{P-1}{M+P-1}" display="inline"><semantics><mfrac><mrow><mi>P</mi><mo>−</mo><mn>1</mn></mrow><mrow><mrow><mi>M</mi><mo>+</mo><mi>P</mi></mrow><mo>−</mo><mn>1</mn></mrow></mfrac><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="M\times" display="inline"><semantics><mrow><mi>M</mi><mo lspace="0.222em">×</mo></mrow><span></span></semantics></math> activations</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Simple; all-forward then all-backward <span>[152]</span></span></span></td></tr><tr><th class="ltx_align_left">1F1B</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="\frac{P-1}{M+P-1}" display="inline"><semantics><mfrac><mrow><mi>P</mi><mo>−</mo><mn>1</mn></mrow><mrow><mrow><mi>M</mi><mo>+</mo><mi>P</mi></mrow><mo>−</mo><mn>1</mn></mrow></mfrac><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="P\times" display="inline"><semantics><mrow><mi>P</mi><mo lspace="0.222em">×</mo></mrow><span></span></semantics></math> activations</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Interleaved; steady-state memory bounded <span>[263]</span></span></span></td></tr><tr><th class="ltx_align_left">Interleaved 1F1B</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="\frac{P-1}{M\cdot V+P-1}" display="inline"><semantics><mfrac><mrow><mi>P</mi><mo>−</mo><mn>1</mn></mrow><mrow><mrow><mrow><mi>M</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><mi>V</mi></mrow><mo>+</mo><mi>P</mi></mrow><mo>−</mo><mn>1</mn></mrow></mfrac><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="P\times" display="inline"><semantics><mrow><mi>P</mi><mo lspace="0.222em">×</mo></mrow><span></span></semantics></math> activations</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Virtual stages (<math alttext="V" display="inline"><semantics><mi>V</mi><span></span></semantics></math>); further reduces bubble <span>[264]</span></span></span></td></tr><tr><th class="ltx_align_left">Zero-Bubble (ZB-H1)</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="\approx 0" display="inline"><semantics><mrow><mphantom></mphantom><mo>≈</mo><mn>0</mn></mrow><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="P\times" display="inline"><semantics><mrow><mi>P</mi><mo lspace="0.222em">×</mo></mrow><span></span></semantics></math> activations</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Splits backward into B and W phases <span>[296]</span></span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 11.1:</span>Pipeline scheduling strategies</p></div>

Unlike TP (AllReduce), PP only requires point-to-point communication of activations between adjacent stages:

<div class="hh-equation" id="ch11.e4"><math alttext="\text{Data per transfer}=b_{\text{micro}}\times s\times d\times 2\text{ bytes (BF16)}" display="block"><semantics><mrow><mtext>Data per transfer</mtext><mo>=</mo><mrow><msub><mi>b</mi><mtext>micro</mtext></msub><mo lspace="0.222em" rspace="0.222em">×</mo><mi>s</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mi>d</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mrow><mn>2</mn><mo lspace="0em" rspace="0em">​</mo><mtext> bytes (BF16)</mtext></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(11.4)</span></div>

For micro-batch=4, seq=2048, <math alttext="d" display="inline"><semantics><mi>d</mi></semantics></math>=8192: <math alttext="4\times 2048\times 8192\times 2=128" display="inline"><semantics><mrow><mrow><mn>4</mn><mo lspace="0.222em" rspace="0.222em">×</mo><mn>2048</mn><mo lspace="0.222em" rspace="0.222em">×</mo><mn>8192</mn><mo lspace="0.222em" rspace="0.222em">×</mo><mn>2</mn></mrow><mo>=</mo><mn>128</mn></mrow></semantics></math> MB per transfer. At InfiniBand 50 GB/s: 2.6 ms per transfer—small relative to compute per stage.

Not all layers have equal compute:

<ul><li>Embedding layer: Very cheap (lookup table).</li><li>Transformer blocks: Uniform compute.</li><li>Final LM head: Moderate (large matrix multiply for vocabulary projection).</li></ul>

Assign more transformer layers to middle stages and fewer to the first/last stages to balance compute.

### Fully Sharded Data Parallelism (FSDP / ZeRO-3)

FSDP [431] (PyTorch) and ZeRO-3 [304] (DeepSpeed) address the memory duplication inherent in DDP: instead of every GPU holding a full copy of parameters, gradients, and optimizer states, each GPU owns only a <math alttext="1/N" display="inline"><semantics><mrow><mn>1</mn><mo>/</mo><mi>N</mi></mrow></semantics></math> slice and reconstructs the full tensor on-the-fly when needed.

<figure id="ch11.f8"><img src="./fig_038_fsdp.png" alt="Figure 11.8: FSDP shards all model state across GPUs. Each GPU owns 1/N1/N of parameters, optimizer states, and gradient" loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 11.8:</span>FSDP shards all model state across GPUs. Each GPU owns <math alttext="1/N" display="inline"><semantics><mrow><mn>1</mn><mo>/</mo><mi>N</mi></mrow></semantics></math> of parameters, optimizer states, and gradients. Full parameters are reconstructed on-demand via AllGather before each layer’s computation.</figcaption></figure>

FSDP execution flow per layer:

<ol><li>Forward: AllGather parameters <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> compute <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> discard non-owned shards.</li><li>Backward: AllGather parameters (again) <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> compute gradients <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> ReduceScatter gradients (each GPU gets its gradient shard) <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> discard non-owned parameter shards.</li><li>Optimizer step: Each GPU updates only its owned shard using its gradient shard and optimizer states.</li></ol>

<div class="hh-table" id="ch11.t2"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Strategy</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Sharded</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Memory/GPU</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Communication</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">DDP (no sharding)</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Nothing</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>1120 GB <math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>AllReduce (gradients only)</span></span></td></tr><tr><th class="ltx_align_left">ZeRO-1</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Optimizer states</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>385 GB <math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>AllReduce (gradients)</span></span></td></tr><tr><th class="ltx_align_left">ZeRO-2</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Optimizer + gradients</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>368 GB <math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>AllReduce (gradients)</span></span></td></tr><tr><th class="ltx_align_left">ZeRO-3 / FSDP</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Everything</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>140 GB</span> ✓</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>AllGather + ReduceScatter (per layer)</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 11.2:</span>Memory comparison: DDP vs FSDP/ZeRO stages (70B model, 8 GPUs). Baseline: BF16 params (140 GB) + BF16 grads (140 GB) + FP32 master+m+v (840 GB) = 1120 GB per GPU.</p></div>

```python
from functools import partial
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, BackwardPrefetch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# Wrap model with FSDP
auto_wrap = partial(transformer_auto_wrap_policy,
                    transformer_layer_cls={LlamaDecoderLayer})
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
    mixed_precision=mp_policy,
    auto_wrap_policy=auto_wrap,  # Wrap each transformer layer
    use_orig_params=True,        # Required for torch.compile compatibility
    limit_all_gathers=True,      # Bound peak memory (1 AllGather in flight at a time)
    forward_prefetch=True,       # Prefetch next layer's params during current layer
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # Prefetch during backward
)
```

### 3D Parallelism: Combining Strategies

Production systems at scale (70B+) combine TP, PP, and DP/FSDP simultaneously:

<figure id="ch11.f9"><img src="./fig_039_3d-parallel.png" alt="Figure 11.9: 3D parallelism layout for 16 GPUs: TP=4 (within each box, using NVLink), PP=2 (orange arrows, stages), DP=2" loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 11.9:</span>3D parallelism layout for 16 GPUs: TP=4 (within each box, using NVLink), PP=2 (orange arrows, stages), DP=2 (red arrows, gradient sync). Each dimension exploits a different level of the communication hierarchy.</figcaption></figure>

<div class="hh-table" id="ch11.t3"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Strategy</span></th><th class="ltx_align_left"><span>Splits</span></th><th class="ltx_align_left"><span>Communication</span></th><th class="ltx_align_left"><span>Scaling Limit</span></th><th class="ltx_align_left"><span>Overhead</span></th><th class="ltx_align_left"><span>When to Use</span></th></tr></thead><tbody><tr><td class="ltx_align_left"><span>DP/DDP</span></td><td class="ltx_align_left"><span>Batch</span></td><td class="ltx_align_left"><span>AllReduce (grads)</span></td><td class="ltx_align_left"><math alttext="\sim" display="inline"><semantics><mo mathsize="0.800em">∼</mo><span></span></semantics></math><span>64 GPUs</span></td><td class="ltx_align_left"><span>5–10%</span></td><td class="ltx_align_left"><span>Model fits on 1 GPU</span></td></tr><tr><td class="ltx_align_left"><span>FSDP</span></td><td class="ltx_align_left"><span>Params+Opt+Grad</span></td><td class="ltx_align_left"><span>AllGather+RS</span></td><td class="ltx_align_left"><span>100s of GPUs</span></td><td class="ltx_align_left"><span>10–20%</span></td><td class="ltx_align_left"><span>Default for <math alttext="&gt;" display="inline"><semantics><mo>&gt;</mo><span></span></semantics></math>13B</span></td></tr><tr><td class="ltx_align_left"><span>TP</span></td><td class="ltx_align_left"><span>Weight matrices</span></td><td class="ltx_align_left"><span>AllReduce (2/layer)</span></td><td class="ltx_align_left"><span>8 GPUs (1 node)</span></td><td class="ltx_align_left"><span>12–18%</span></td><td class="ltx_align_left"><span>Large model inference+train</span></td></tr><tr><td class="ltx_align_left"><span>SP</span></td><td class="ltx_align_left"><span>Activations (seq)</span></td><td class="ltx_align_left"><span>Reuses TP comms</span></td><td class="ltx_align_left"><span>Same as TP</span></td><td class="ltx_align_left"><math alttext="\approx" display="inline"><semantics><mo mathsize="0.800em">≈</mo><span></span></semantics></math><span>0% extra</span></td><td class="ltx_align_left"><span>Always with TP</span></td></tr><tr><td class="ltx_align_left"><span>PP</span></td><td class="ltx_align_left"><span>Layers (stages)</span></td><td class="ltx_align_left"><span>Point-to-point</span></td><td class="ltx_align_left"><math alttext="\sim" display="inline"><semantics><mo mathsize="0.800em">∼</mo><span></span></semantics></math><span>16 stages</span></td><td class="ltx_align_left"><span>15–30%</span></td><td class="ltx_align_left"><span>100B+ models only</span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 11.3:</span>Parallelism strategy comparison summary</p></div>

### Decoupled DiLoCo: Training Across Datacenters

Conventional distributed training assumes high-bandwidth interconnects—100+ Gbps InfiniBand within a single datacenter. This requirement makes it impossible to train across geographically distributed datacenters using standard internet links. Decoupled DiLoCo[113] (Google DeepMind, April 2026) breaks this assumption, enabling large-scale training across regions with commodity bandwidth.

Decoupled DiLoCo extends this by further decoupling the outer synchronization: workers do not need to synchronize simultaneously. Instead, updates are asynchronous across regions, with a coordinator applying outer updates as they arrive. This tolerates the variable latency of wide-area networks.

## The Generation Bottleneck: Quantitative Analysis

<div class="hh-table" id="ch11.t4"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Config</span></th><th class="ltx_align_left"><span>Batch</span></th><th class="ltx_align_left"><span>Time/batch</span></th><th class="ltx_align_left"><span>Tok/s/GPU</span></th><th class="ltx_align_left"><span>Notes</span></th></tr></thead><tbody><tr><td class="ltx_align_left"><span>TP=1, batch=1</span></td><td class="ltx_align_left"><span>1</span></td><td class="ltx_align_left"><span>36s</span></td><td class="ltx_align_left"><span>14</span></td><td class="ltx_align_left"><span>Baseline, worst case</span></td></tr><tr><td class="ltx_align_left"><span>TP=4, batch=1</span></td><td class="ltx_align_left"><span>1</span></td><td class="ltx_align_left"><span>9s</span></td><td class="ltx_align_left"><span>57</span></td><td class="ltx_align_left"><span>Linear TP scaling for gen</span></td></tr><tr><td class="ltx_align_left"><span>TP=4, batch=32</span></td><td class="ltx_align_left"><span>32</span></td><td class="ltx_align_left"><span>15s</span></td><td class="ltx_align_left"><span>1092</span></td><td class="ltx_align_left"><span>Near-optimal batching</span></td></tr><tr><td class="ltx_align_left"><span>TP=4, batch=128, vLLM</span></td><td class="ltx_align_left"><span>128</span></td><td class="ltx_align_left"><span>45s</span></td><td class="ltx_align_left"><span>1456</span></td><td class="ltx_align_left"><span>Continuous batching</span></td></tr><tr><td class="ltx_align_left"><span>TP=4, batch=128, INT8</span></td><td class="ltx_align_left"><span>128</span></td><td class="ltx_align_left"><span>25s</span></td><td class="ltx_align_left"><span>2621</span></td><td class="ltx_align_left"><span>2<math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math> bandwidth savings</span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 11.4:</span>Generation throughput for 70B model (512 tokens, various configurations)</p></div>

Optimization stack (cumulative speedup):

<ol><li>vLLM + PagedAttention[196] (2–4<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>): Eliminates KV cache fragmentation, enables larger batches</li><li>Continuous batching[412] (1.5–2<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>): Don’t wait for longest sequence; start new ones as others finish</li><li>Speculative decoding[206] (2–3<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>): Small draft model proposes 5 tokens, large model verifies in one forward pass. Accept 3–4 on average.</li><li>INT8/FP8 weights for gen (2<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>): Halve bandwidth needs. Quality loss is minimal since we’re sampling (not computing exact logits for training)</li><li>CUDA graphs (1.1–1.3<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>): Eliminate kernel launch overhead for fixed-shape operations</li><li>Prefix caching (1.5<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> for shared-prefix prompts): Don’t recompute system prompt KV cache</li></ol>

```python
# Production vLLM generation setup
from vllm import LLM, SamplingParams

engine = LLM(
    model="./policy_checkpoint",
    tensor_parallel_size=4,           # TP=4 per instance
    gpu_memory_utilization=0.92,      # Leave headroom for KV cache
    max_num_batched_tokens=16384,     # Max tokens in flight
    max_num_seqs=256,                 # Max concurrent sequences
    dtype="bfloat16",
    enable_prefix_caching=True,       # Cache system prompt KV
    speculative_model="./draft_1B",   # Speculative decoding
    num_speculative_tokens=5,
    block_size=16,                    # PagedAttention block size
    swap_space=4,                     # GB swap space for preemption
)

# Generate responses for RLHF batch
sampling_params = SamplingParams(
    temperature=0.7, top_p=0.9, max_tokens=512,
    logprobs=1,  # Need log-probs for PPO ratio calculation
)
outputs = engine.generate(prompts, sampling_params)
# Extract: responses, log_probs for each token (needed for PPO/GRPO)
```

## Decoupled Architecture: Production Design

Production RLHF systems such as DeepSpeed-Chat [410] and OpenRLHF [148] use a decoupled architecture that separates generation, scoring, and training into independently scalable clusters.

<figure id="ch11.f10"><img src="./fig_040_fig40.png" alt="Figure 11.10: Decoupled RLHF architecture. Each cluster optimized for its workload. Scored rollouts accumulate in the ex" loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 11.10:</span>Decoupled RLHF architecture. Each cluster optimized for its workload. Scored rollouts accumulate in the experience buffer before being consumed by training.</figcaption></figure>

## Weight Synchronization Strategies

StrategyStalenessBandwidthQuality ImpactSynchronous (every step)0 steps140 GB/stepPerfect but too slowPeriodic (every 50 steps)25 avg2.8 GB/step amortized<math alttext="&lt;" display="inline"><semantics><mo mathsize="0.900em">&lt;</mo></semantics></math>2% quality lossDelta compression (INT8)25 avg0.4 GB/step<math alttext="&lt;" display="inline"><semantics><mo mathsize="0.900em">&lt;</mo></semantics></math>3% quality lossAsync streaming5–10 steps14 GB/step (background)<math alttext="&lt;" display="inline"><semantics><mo mathsize="0.900em">&lt;</mo></semantics></math>1% quality loss

## Memory Optimization Techniques

ZeRO StageWhat Gets ShardedMemory/GPU (70B, 8 GPUs)None (Data Parallel)Nothing (full replica)560GB per GPU (impossible)ZeRO-1Optimizer states only175GBZeRO-2Optimizer states + Gradients105GBZeRO-3 (FSDP)Optimizer + Gradients + Parameters70GB (fits in A100-80GB!)

Additional techniques:

<ul><li>Gradient checkpointing[49]: Don’t store all activations; recompute during backward pass. Saves <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>60% activation memory, costs <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>33% extra compute. Selective: only checkpoint attention layers (memory-heavy), keep FFN activations (compute-heavy to recompute).</li><li>Mixed precision[255]: Forward in BF16 (2 bytes/param), optimizer states in FP32 (4 bytes each for m,v). Master weights in FP32 for accumulation.</li><li>CPU offloading (ZeRO-Infinity [305]): Move optimizer states to CPU RAM. 50% memory savings but 2–3<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> slower (PCIe 64GB/s bottleneck).</li><li>Activation offloading: Move activations to CPU during forward, bring back for backward. Only when memory is truly critical.</li><li>Flash Attention[64, 65]: O(<math alttext="n" display="inline"><semantics><mi>n</mi></semantics></math>) memory instead of O(<math alttext="n^{2}" display="inline"><semantics><msup><mi>n</mi><mn>2</mn></msup></semantics></math>) for attention. 2–4<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> faster + massive memory savings for long sequences.</li></ul>

### Flash Attention’s Impact on RLHF

```python
# DeepSpeed ZeRO-3 configuration for 70B RLHF training
ds_config = {
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,                    # Overlap communication with compute
        "contiguous_gradients": True,            # Better memory layout
        "reduce_scatter": True,                  # More efficient than allreduce
        "reduce_bucket_size": 5e7,               # 50M params per bucket
        "prefetch_bucket_size": 5e7,             # Prefetch next bucket
        "param_persistence_threshold": 1e5,      # Keep small params on all GPUs
        "offload_optimizer": {"device": "cpu", "pin_memory": True},  # CPU offload
        "sub_group_size": 1e9,                   # Reduce fragmentation
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "train_micro_batch_size_per_gpu": 2,
    "wall_clock_breakdown": True,
}
```

## Fault Tolerance at Scale

Production fault tolerance stack:

<ol><li>Detection: NCCL timeout (60s), GPU heartbeat (10s), NVML health monitoring, ECC error counting.</li><li>Checkpointing: Async every 50–100 steps. Non-blocking (background thread). Save: model weights, optimizer states (Adam m/v), scheduler state, RNG states, KL coefficient, replay buffer. Keep last 3 checkpoints. Time: <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>30s for 70B (parallel write to NVMe).</li><li>Recovery: (a) Generation cluster = stateless, just restart and load latest weights. (b) Training cluster: load checkpoint, rebuild NCCL process group excluding failed node, redistribute FSDP shards, resume from last checkpoint.</li><li>Elastic training: Torch Elastic / Kubernetes auto-scaling. Replace failed node within minutes. Training continues with <math alttext="N-1" display="inline"><semantics><mrow><mi>N</mi><mo>−</mo><mn>1</mn></mrow></semantics></math> GPUs temporarily.</li><li>Prevention: GPU health pre-screening (run GEMM stress test before starting). Hot spares on standby. Redundant network paths (dual-rail InfiniBand).</li></ol>

## End-to-End Latency Breakdown

<figure id="ch11.f11"><img src="./fig_041_fig41.png" alt="Figure 11.11: Without overlap (monolithic). With decoupled: gen overlaps with training, effective 1.4×\times speedup." loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 11.11:</span>Without overlap (monolithic). With decoupled: gen overlaps with training, effective 1.4<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> speedup.</figcaption></figure>

PhaseTime (70B)Bound ByOptimizationGeneration (128<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>512 tok)30–45sMemory bandwidthvLLM, spec decoding, INT8Reward scoring5–8sCompute (batch forward)INT8 RM, batch=128Reference log-probs4–6sCompute (batch forward)INT8 ref, or LoRA (free)PPO update (4 epochs)8–12sCompute (backprop)FSDP, Flash AttentionWeight sync0–3sNetwork (async)Delta compression, asyncTotal (monolithic)50–75sTotal (decoupled, overlapped)35–50sGen overlaps with prev training

## Monitoring and Observability

## Network Topology and Communication Patterns

Efficient distributed training requires understanding the hierarchical communication fabric that connects GPUs. Modern clusters use a two-tier architecture: ultra-fast intra-node links and slower but scalable inter-node networks.

### Intra-Node: NVLink and NVSwitch

<div class="hh-table" id="ch11.t5"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Generation</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>BW per link</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Links/GPU</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Total BW</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Platform</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">NVLink 3.0</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>50 GB/s</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>12</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>600 GB/s</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>A100 (DGX A100)</span></span></td></tr><tr><th class="ltx_align_left">NVLink 4.0</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>50 GB/s</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>18</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>900 GB/s</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>H100 (DGX H100)</span></span></td></tr><tr><th class="ltx_align_left">NVLink 5.0</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>100 GB/s</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>18</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>1800 GB/s</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>B200 (DGX B200)</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 11.5:</span>NVLink generations and their impact on LLM training</p></div>

Within a single node (typically 8 GPUs), NVSwitch provides full-bisection bandwidth between all GPU pairs. This means any GPU can communicate with any other at full NVLink speed simultaneously—critical for Tensor Parallelism where every layer requires an AllReduce across all 8 GPUs.

### Inter-Node: InfiniBand and RoCE

For FSDP/ZeRO-3 AllGather and ReduceScatter operations across nodes, the inter-node network dominates.

<div class="hh-table" id="ch11.t6"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Technology</span></th><th class="ltx_align_left"><span>Bandwidth</span></th><th class="ltx_align_left"><span>Latency</span></th><th class="ltx_align_left"><span>Notes</span></th></tr></thead><tbody><tr><td class="ltx_align_left"><span>InfiniBand NDR</span></td><td class="ltx_align_left"><span>400 Gb/s (50 GB/s)</span></td><td class="ltx_align_left"><span>1–2 <math alttext="\mu" display="inline"><semantics><mi>μ</mi><span></span></semantics></math>s</span></td><td class="ltx_align_left"><span>Gold standard, RDMA, lossless</span></td></tr><tr><td class="ltx_align_left"><span>InfiniBand NDR (dual-rail)</span></td><td class="ltx_align_left"><span>800 Gb/s (100 GB/s)</span></td><td class="ltx_align_left"><span>1–2 <math alttext="\mu" display="inline"><semantics><mi>μ</mi><span></span></semantics></math>s</span></td><td class="ltx_align_left"><span>Used in H100 clusters</span></td></tr><tr><td class="ltx_align_left"><span>RoCE v2</span></td><td class="ltx_align_left"><span>100–400 Gb/s</span></td><td class="ltx_align_left"><span>2–5 <math alttext="\mu" display="inline"><semantics><mi>μ</mi><span></span></semantics></math>s</span></td><td class="ltx_align_left"><span>Cheaper, needs PFC/ECN tuning</span></td></tr><tr><td class="ltx_align_left"><span>Ethernet (TCP)</span></td><td class="ltx_align_left"><span>100–400 Gb/s</span></td><td class="ltx_align_left"><span>10–50 <math alttext="\mu" display="inline"><semantics><mi>μ</mi><span></span></semantics></math>s</span></td><td class="ltx_align_left"><span>Not suitable for <math alttext="&gt;" display="inline"><semantics><mo>&gt;</mo><span></span></semantics></math>16 GPU training</span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 11.6:</span>Inter-node networking options for LLM training clusters</p></div>

### Communication Primitives and Their Costs

Understanding when each collective is used helps diagnose bottlenecks:

<div class="hh-table" id="ch11.t7"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Collective</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Data Moved</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Used By</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>When</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">AllReduce</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="2\cdot\frac{N-1}{N}\cdot M" display="inline"><semantics><mrow><mn>2</mn><mo lspace="0.222em" rspace="0.222em">⋅</mo><mfrac><mrow><mi>N</mi><mo>−</mo><mn>1</mn></mrow><mi>N</mi></mfrac><mo lspace="0.222em" rspace="0.222em">⋅</mo><mi>M</mi></mrow><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>TP, DP</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Sum gradients or activations across GPUs</span></span></td></tr><tr><th class="ltx_align_left">AllGather</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="\frac{N-1}{N}\cdot M" display="inline"><semantics><mrow><mfrac><mrow><mi>N</mi><mo>−</mo><mn>1</mn></mrow><mi>N</mi></mfrac><mo lspace="0.222em" rspace="0.222em">⋅</mo><mi>M</mi></mrow><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>FSDP forward</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Reconstruct full parameter tensor before matmul</span></span></td></tr><tr><th class="ltx_align_left">ReduceScatter</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="\frac{N-1}{N}\cdot M" display="inline"><semantics><mrow><mfrac><mrow><mi>N</mi><mo>−</mo><mn>1</mn></mrow><mi>N</mi></mfrac><mo lspace="0.222em" rspace="0.222em">⋅</mo><mi>M</mi></mrow><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>FSDP backward</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Distribute gradient shards after backprop</span></span></td></tr><tr><th class="ltx_align_left">Broadcast</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="M" display="inline"><semantics><mi>M</mi><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>PP</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Send activations to next pipeline stage</span></span></td></tr><tr><th class="ltx_align_left">Send/Recv</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="M" display="inline"><semantics><mi>M</mi><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>PP</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Point-to-point between adjacent stages</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 11.7:</span>NCCL collective operations in distributed LLM training</p></div>

where <math alttext="M" display="inline"><semantics><mi>M</mi></semantics></math> is the message size (bytes) and <math alttext="N" display="inline"><semantics><mi>N</mi></semantics></math> is the number of participants.

### Network Topology Design

Production clusters use fat-tree or rail-optimized topologies:

<ul><li>Fat-tree: Full bisection bandwidth at every level. Any node can communicate with any other at full speed. Expensive (many switches) but maximally flexible.</li><li>Rail-optimized: GPU <math alttext="i" display="inline"><semantics><mi>i</mi></semantics></math> on every node connects to the same leaf switch (“rail <math alttext="i" display="inline"><semantics><mi>i</mi></semantics></math>”). AllReduce within a rail is cheap; cross-rail traffic is expensive. Used by Meta’s RSC and Google’s TPU pods.</li><li>3D torus / Dragonfly: Used in HPC clusters (Frontier, Aurora). Topology-aware job placement is critical.</li></ul>

## Training Throughput and Model FLOPs Utilization

### Measuring Training Efficiency: MFU

Model FLOPs Utilization (MFU)[55] is the standard metric for training efficiency:

<div class="hh-equation" id="ch11.e5"><math alttext="\text{MFU}=\frac{\text{Observed throughput (tokens/sec)}\times\text{FLOPs per token}}{\text{Peak hardware FLOPS}}" display="block"><semantics><mrow><mtext>MFU</mtext><mo>=</mo><mfrac><mrow><mtext>Observed throughput (tokens/sec)</mtext><mo lspace="0.222em" rspace="0.222em">×</mo><mtext>FLOPs per token</mtext></mrow><mtext>Peak hardware FLOPS</mtext></mfrac></mrow></semantics></math><span class="hh-equation-number">(11.5)</span></div>

For a transformer with <math alttext="P" display="inline"><semantics><mi>P</mi></semantics></math> parameters, <math alttext="s" display="inline"><semantics><mi>s</mi></semantics></math> sequence length, and <math alttext="b" display="inline"><semantics><mi>b</mi></semantics></math> batch size:

<div class="hh-equation" id="ch11.e6"><math alttext="\text{FLOPs per token}\approx 6P+12\cdot n_{\text{layers}}\cdot d_{\text{model}}\cdot s" display="block"><semantics><mrow><mtext>FLOPs per token</mtext><mo>≈</mo><mrow><mrow><mn>6</mn><mo lspace="0em" rspace="0em">​</mo><mi>P</mi></mrow><mo>+</mo><mrow><mn>12</mn><mo lspace="0.222em" rspace="0.222em">⋅</mo><msub><mi>n</mi><mtext>layers</mtext></msub><mo lspace="0.222em" rspace="0.222em">⋅</mo><msub><mi>d</mi><mtext>model</mtext></msub><mo lspace="0.222em" rspace="0.222em">⋅</mo><mi>s</mi></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(11.6)</span></div>

The factor of 6 comes from: 2 (multiply-add) <math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> 3 (forward + backward, where backward <math alttext="\approx 2\times" display="inline"><semantics><mrow><mo>≈</mo><mn>2</mn><mo lspace="0.222em">×</mo></mrow></semantics></math> forward). The second term accounts for attention’s <math alttext="O(s^{2})" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msup><mi>s</mi><mn>2</mn></msup><mo stretchy="false">)</mo></mrow></mrow></semantics></math> cost.

<div class="hh-table" id="ch11.t8"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Model</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Hardware</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>MFU</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Tokens/sec/GPU</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Configuration</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">LLaMA-7B</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>8<math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math>A100</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>57%</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>3,200</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>FSDP, FlashAttn, BF16</span></span></td></tr><tr><th class="ltx_align_left">LLaMA-13B</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>16<math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math>A100</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>52%</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>1,750</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>FSDP, FlashAttn, BF16</span></span></td></tr><tr><th class="ltx_align_left">LLaMA-70B</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>64<math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math>A100</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>45%</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>380</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>FSDP+TP=8, FlashAttn</span></span></td></tr><tr><th class="ltx_align_left">GPT-4 (est.)</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>10,000+ H100</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>40–50%</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>—</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>3D parallelism</span></span></td></tr><tr><th class="ltx_align_left">PaLM-540B</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>6144 TPUv4</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>46%</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>—</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>DP+TP+PP</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 11.8:</span>MFU benchmarks across scales and hardware</p></div>

### Compute-Optimal Batch Sizing

The effective batch size interacts with hardware utilization in non-obvious ways:

<div class="hh-equation" id="ch11.e7"><math alttext="\text{Effective batch size}=\text{micro\_batch}\times\text{grad\_accum}\times\text{DP degree}" display="block"><semantics><mrow><mtext>Effective batch size</mtext><mo>=</mo><mrow><mtext>micro_batch</mtext><mo lspace="0.222em" rspace="0.222em">×</mo><mtext>grad_accum</mtext><mo lspace="0.222em" rspace="0.222em">×</mo><mtext>DP degree</mtext></mrow></mrow></semantics></math><span class="hh-equation-number">(11.7)</span></div>

<ul><li>Too small: GPU underutilized (low arithmetic intensity), communication dominates.</li><li>Too large: Diminishing learning per token (critical batch size exceeded), wastes compute.</li><li>Sweet spot: The <em>critical batch size</em><math alttext="B_{\text{crit}}" display="inline"><semantics><msub><mi>B</mi><mtext>crit</mtext></msub></semantics></math> where gradient noise equals gradient signal. For LLMs, <math alttext="B_{\text{crit}}\sim 1" display="inline"><semantics><mrow><msub><mi>B</mi><mtext>crit</mtext></msub><mo>∼</mo><mn>1</mn></mrow></semantics></math>–<math alttext="4" display="inline"><semantics><mn>4</mn></semantics></math>M tokens [250].</li></ul>

For RLHF specifically, the batch contains <em>rollouts</em> (not just tokens):

<div class="hh-equation" id="ch11.e8"><math alttext="\text{RLHF batch}=N_{\text{prompts}}\times K_{\text{generations}}\times L_{\text{avg response length}}" display="block"><semantics><mrow><mtext>RLHF batch</mtext><mo>=</mo><mrow><msub><mi>N</mi><mtext>prompts</mtext></msub><mo lspace="0.222em" rspace="0.222em">×</mo><msub><mi>K</mi><mtext>generations</mtext></msub><mo lspace="0.222em" rspace="0.222em">×</mo><msub><mi>L</mi><mtext>avg response length</mtext></msub></mrow></mrow></semantics></math><span class="hh-equation-number">(11.8)</span></div>

Typical production values: <math alttext="N=128" display="inline"><semantics><mrow><mi>N</mi><mo>=</mo><mn>128</mn></mrow></semantics></math> prompts, <math alttext="K=1" display="inline"><semantics><mrow><mi>K</mi><mo>=</mo><mn>1</mn></mrow></semantics></math>–<math alttext="4" display="inline"><semantics><mn>4</mn></semantics></math> generations, <math alttext="L=256" display="inline"><semantics><mrow><mi>L</mi><mo>=</mo><mn>256</mn></mrow></semantics></math>–<math alttext="512" display="inline"><semantics><mn>512</mn></semantics></math> tokens <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 32K–256K tokens per step.

### Profiling and Bottleneck Diagnosis

Key profiling tools and what they reveal:

ToolCapturesBest For<code>torch.profiler</code>Kernel timing, memoryFinding slow ops, memory leaksNVIDIA Nsight SystemsFull GPU timelineVisualizing overlap, gaps between kernels<code>nccl_debug=INFO</code>Collective sizes/timesDiagnosing communication bottlenecks<code>torch.cuda.memory_stats</code>Allocation patternsFinding fragmentation, peak usageDeepSpeed Flops ProfilerPer-layer FLOPsIdentifying load imbalance<code>py-spy</code> / <code>scalene</code>CPU profilingData loading, tokenization bottlenecks

## Cost Analysis and Cloud Deployment

Understanding the economics of RLHF training is essential for planning.

### Hardware Cost Comparison

<div class="hh-table" id="ch11.t9"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>GPU</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>On-Demand/hr</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Spot/hr</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Memory</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Use Case</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">A100 80GB</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>$2.50–3.50</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>$1.00–1.50</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>80 GB HBM2e</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Budget training, gen cluster</span></span></td></tr><tr><th class="ltx_align_left">H100 80GB</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>$4.00–6.00</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>$2.00–3.00</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>80 GB HBM3</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Production training</span></span></td></tr><tr><th class="ltx_align_left">H200 141GB</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>$6.00–8.00</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>—</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>141 GB HBM3e</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Large context, fewer-GPU configs</span></span></td></tr><tr><th class="ltx_align_left">MI300X 192GB</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>$3.50–5.00</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>$1.50–2.50</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>192 GB HBM3</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Cost-effective alternative</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 11.9:</span>Approximate cloud GPU costs for RLHF training (2024–2025 pricing)</p></div>

### RLHF Training Cost Estimation

<div class="hh-equation" id="ch11.e9"><math alttext="\text{Cost}=\frac{N_{\text{steps}}\times T_{\text{step}}}{3600}\times N_{\text{GPUs}}\times C_{\text{GPU/hr}}" display="block"><semantics><mrow><mtext>Cost</mtext><mo>=</mo><mrow><mfrac><mrow><msub><mi>N</mi><mtext>steps</mtext></msub><mo lspace="0.222em" rspace="0.222em">×</mo><msub><mi>T</mi><mtext>step</mtext></msub></mrow><mn>3600</mn></mfrac><mo lspace="0.222em" rspace="0.222em">×</mo><msub><mi>N</mi><mtext>GPUs</mtext></msub><mo lspace="0.222em" rspace="0.222em">×</mo><msub><mi>C</mi><mtext>GPU/hr</mtext></msub></mrow></mrow></semantics></math><span class="hh-equation-number">(11.9)</span></div>

### Cost Optimization Strategies

<ul><li>Spot/preemptible instances: 50–70% savings. Requires robust checkpointing (save every 5 minutes).</li><li>Right-sizing: Don’t use H100 for generation (memory-bound); A100 achieves similar tokens/$ for inference.</li><li>Quantized inference: INT8/FP8 for generation and scoring halves GPU count for those clusters.</li><li>Progressive training: Start with 8B proxy model for reward engineering/debugging (<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>$200), then scale to 70B.</li><li>LoRA for reference-free: Eliminates reference model entirely (50% memory reduction).</li><li>Shorter sequences first: Curriculum from 256<math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math>512<math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math>1024 token generations saves 40% compute.</li></ul>

## Distributed Checkpointing

At scale, naive checkpointing becomes a bottleneck. A 70B model with optimizer state requires saving <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>840 GB per checkpoint (FP32 master weights + Adam m + v).

### Checkpointing Strategies

<div class="hh-table" id="ch11.t10"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Strategy</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Save Time (70B)</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Storage/ckpt</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Characteristics</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">Synchronous (all ranks)</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>30–60s (blocking)</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>420 GB</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Simple, stalls training</span></span></td></tr><tr><th class="ltx_align_left">Async (background copy)</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="&lt;" display="inline"><semantics><mo>&lt;</mo><span></span></semantics></math>1s (non-blocking)</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>420 GB</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Overlaps with next step</span></span></td></tr><tr><th class="ltx_align_left">Incremental (delta)</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="&lt;" display="inline"><semantics><mo>&lt;</mo><span></span></semantics></math>1s</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>5–20 GB</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Only save changed params</span></span></td></tr><tr><th class="ltx_align_left">Sharded (FSDP native)</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>5–10s</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>420 GB sharded</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Each rank saves its shard</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 11.10:</span>Checkpointing approaches for large-scale RLHF</p></div>

### Production Checkpointing with torch.distributed.checkpoint

```python
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, StateDictOptions

# Save: each rank writes its shard in parallel
state_dict = {"model": get_state_dict(model, options=StateDictOptions(full_state_dict=False))}
dcp.save(
    state_dict=state_dict,
    storage_writer=dcp.FileSystemWriter("/mnt/checkpoints/step_5000"),
    planner=dcp.DefaultSavePlanner(),  # Handles FSDP sharding automatically
)

# Async save: non-blocking, runs in background thread
future = dcp.async_save(
    state_dict=state_dict,
    storage_writer=dcp.FileSystemWriter("/mnt/checkpoints/step_5000"),
)
# Training continues immediately; future.result() blocks only if needed
```

## Hardware Selection Guide

Choosing the right hardware depends on model size, budget, and training phase.

<div class="hh-table" id="ch11.t11"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Model Size</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Training Phase</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Recommended</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Configuration</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left"><math alttext="\leq" display="inline"><semantics><mo>≤</mo><span></span></semantics></math>7B</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>SFT + RLHF</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>1–2<math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math> A100</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Single node, no parallelism needed</span></span></td></tr><tr><th class="ltx_align_left">7–13B</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>SFT + RLHF</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>4–8<math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math> A100</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>FSDP, optional TP=2 for gen</span></span></td></tr><tr><th class="ltx_align_left">13–34B</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>SFT + RLHF</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>8–16<math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math> A100/H100</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>FSDP + TP=4 for gen</span></span></td></tr><tr><th class="ltx_align_left">70B</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>RLHF (full)</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>32–64<math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math> A100/H100</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Decoupled, FSDP + TP=8</span></span></td></tr><tr><th class="ltx_align_left">70B</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>RLHF (LoRA)</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>8–16<math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math> A100/H100</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>No ref model, LoRA adapters</span></span></td></tr><tr><th class="ltx_align_left"><math alttext="&gt;" display="inline"><semantics><mo>&gt;</mo><span></span></semantics></math>100B</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>RLHF</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>128+<math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math> H100</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>3D parallelism (TP+PP+DP)</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 11.11:</span>Hardware recommendations by model scale and training phase</p></div>

## Optimizer Configuration for RL Training

RL training (PPO, GRPO, DPO) imposes unique demands on the optimizer compared to pretraining or SFT. The loss landscape is non-stationary (the policy changes what data is generated), gradients are noisier (reward signal variance), and training is more prone to catastrophic forgetting or reward hacking. This section consolidates RL-specific optimizer guidance, using AdamW [239] as the default optimizer.

### Why RL Requires Different Optimizer Settings

### Recommended Hyperparameters by RL Method

<div class="hh-table" id="ch11.t12"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Method</span></th><th class="ltx_align_left"><span>Optimizer</span></th><th class="ltx_align_left"><span>LR</span></th><th class="ltx_align_left"><span>WD</span></th><th class="ltx_align_left"><span>Warmup</span></th><th class="ltx_align_left"><span>Schedule</span></th></tr></thead><tbody><tr><th class="ltx_align_left"><span>DPO</span></th><th class="ltx_align_left"><span>AdamW</span></th><th class="ltx_align_left"><math alttext="5\text{e-}7" display="inline"><semantics><mrow><mn mathsize="0.800em">5</mn><mo lspace="0em" rspace="0em">​</mo><mtext mathsize="0.800em">e-</mtext><mo lspace="0em" rspace="0em">​</mo><mn mathsize="0.800em">7</mn></mrow><span></span></semantics></math></th><td class="ltx_align_left"><span>0.0</span></td><td class="ltx_align_left"><span>50 steps</span></td><td class="ltx_align_left"><span>Constant or Linear</span></td></tr><tr><th class="ltx_align_left"><span>PPO (policy)</span></th><th class="ltx_align_left"><span>AdamW</span></th><th class="ltx_align_left"><math alttext="1\text{e-}6" display="inline"><semantics><mrow><mn mathsize="0.800em">1</mn><mo lspace="0em" rspace="0em">​</mo><mtext mathsize="0.800em">e-</mtext><mo lspace="0em" rspace="0em">​</mo><mn mathsize="0.800em">6</mn></mrow><span></span></semantics></math></th><td class="ltx_align_left"><span>0.0</span></td><td class="ltx_align_left"><span>20 steps</span></td><td class="ltx_align_left"><span>Constant</span></td></tr><tr><th class="ltx_align_left"><span>PPO (critic)</span></th><th class="ltx_align_left"><span>AdamW</span></th><th class="ltx_align_left"><math alttext="1\text{e-}6" display="inline"><semantics><mrow><mn mathsize="0.800em">1</mn><mo lspace="0em" rspace="0em">​</mo><mtext mathsize="0.800em">e-</mtext><mo lspace="0em" rspace="0em">​</mo><mn mathsize="0.800em">6</mn></mrow><span></span></semantics></math></th><td class="ltx_align_left"><span>0.0</span></td><td class="ltx_align_left"><span>20 steps</span></td><td class="ltx_align_left"><span>Constant</span></td></tr><tr><th class="ltx_align_left"><span>GRPO</span></th><th class="ltx_align_left"><span>AdamW</span></th><th class="ltx_align_left"><math alttext="1\text{e-}6" display="inline"><semantics><mrow><mn mathsize="0.800em">1</mn><mo lspace="0em" rspace="0em">​</mo><mtext mathsize="0.800em">e-</mtext><mo lspace="0em" rspace="0em">​</mo><mn mathsize="0.800em">6</mn></mrow><span></span></semantics></math></th><td class="ltx_align_left"><span>0.0</span></td><td class="ltx_align_left"><span>20 steps</span></td><td class="ltx_align_left"><span>Constant</span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 11.12:</span>Optimizer settings for RL training phases. All use <math alttext="\beta_{1}=0.9" display="inline"><semantics><mrow><msub><mi>β</mi><mn>1</mn></msub><mo>=</mo><mn>0.9</mn></mrow></semantics></math>, <math alttext="\beta_{2}=0.95" display="inline"><semantics><mrow><msub><mi>β</mi><mn>2</mn></msub><mo>=</mo><mn>0.95</mn></mrow></semantics></math>, <math alttext="\epsilon=10^{-8}" display="inline"><semantics><mrow><mi>ϵ</mi><mo>=</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>8</mn></mrow></msup></mrow></semantics></math>, <code>max_grad_norm</code>=1.0, BF16.</p></div>

### Beta-2 = 0.95 for RL: Faster Adaptation

The default Adam <math alttext="\beta_{2}=0.999" display="inline"><semantics><mrow><msub><mi>β</mi><mn>2</mn></msub><mo>=</mo><mn>0.999</mn></mrow></semantics></math> gives a very long memory for the second moment (<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>1000-step effective window). In RL training, the loss landscape changes rapidly as the policy evolves—the gradient variance from 1000 steps ago is irrelevant. Using <math alttext="\beta_{2}=0.95" display="inline"><semantics><mrow><msub><mi>β</mi><mn>2</mn></msub><mo>=</mo><mn>0.95</mn></mrow></semantics></math> shortens the window to <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>20 steps, making the adaptive learning rate respond quickly to changing gradient statistics.

### Mixed Precision for RL: FP32 Master Weights Are Critical

RL training is particularly sensitive to numerical precision:

<ul><li>Gradients are noisier—small updates must accumulate accurately over many steps</li><li>Learning rates are very small (<math alttext="10^{-6}" display="inline"><semantics><msup><mn>10</mn><mrow><mo>−</mo><mn>6</mn></mrow></msup></semantics></math>–<math alttext="10^{-7}" display="inline"><semantics><msup><mn>10</mn><mrow><mo>−</mo><mn>7</mn></mrow></msup></semantics></math>), making <math alttext="\Delta\theta\ll\theta" display="inline"><semantics><mrow><mrow><mi mathvariant="normal">Δ</mi><mo lspace="0em" rspace="0em">​</mo><mi>θ</mi></mrow><mo>≪</mo><mi>θ</mi></mrow></semantics></math></li><li>BF16 mantissa (7 bits <math alttext="\approx" display="inline"><semantics><mo>≈</mo></semantics></math> 0.8% relative precision) cannot represent updates of magnitude <math alttext="10^{-6}" display="inline"><semantics><msup><mn>10</mn><mrow><mo>−</mo><mn>6</mn></mrow></msup></semantics></math> relative to weights of magnitude <math alttext="10^{0}" display="inline"><semantics><msup><mn>10</mn><mn>0</mn></msup></semantics></math></li></ul>

Always use FP32 master weights for RL training. BF16-only training (no FP32 copy) reliably causes reward collapse in PPO/GRPO after 100–500 steps.

### Gradient Clipping is Critical for RL

In PPO and GRPO, the reward signal can be highly variable, especially early in training. A single bad batch can produce gradients with norm <math alttext="&gt;100" display="inline"><semantics><mrow><mphantom></mphantom><mo>&gt;</mo><mn>100</mn></mrow></semantics></math>, which would completely destroy the model weights. <code>max_grad_norm=1.0</code> is the standard setting. For SFT, clipping is less critical but still recommended.

### Diagnosing RL Training Instability

### HuggingFace TRL Configuration for RL

The TRL library [376] provides production-ready implementations of PPO, DPO, and other RL methods for LLMs.

```
from trl import PPOConfig, PPOTrainer, DPOConfig, DPOTrainer

# --- PPO Configuration ---
ppo_config = PPOConfig(
    # Optimizer (AdamW with RL-specific settings)
    learning_rate=1e-6,           # 10-100x smaller than SFT

    # PPO-specific
    ppo_epochs=4,                 # mini-batch updates per rollout
    mini_batch_size=16,
    batch_size=64,                # rollout batch size

    # Gradient control
    max_grad_norm=1.0,

    # KL penalty (replaces weight decay as regularizer)
    init_kl_coef=0.2,            # initial KL penalty coefficient
    adap_kl_ctrl=True,           # adaptive KL targeting
    target_kl=6.0,               # target KL divergence

    # Mixed precision
    bf16=True,                   # BF16 compute, FP32 master weights
)

ppo_trainer = PPOTrainer(
    model=model,
    ref_model=ref_model,
    config=ppo_config,
    tokenizer=tokenizer,
    dataset=dataset,
)

# --- DPO Configuration ---
dpo_config = DPOConfig(
    output_dir="./dpo_output",

    # Optimizer
    learning_rate=5e-7,           # even smaller than PPO
    optim="adamw_torch",
    adam_beta1=0.9,
    adam_beta2=0.95,              # shorter memory for RL
    weight_decay=0.0,            # no WD -- KL provides regularization

    # Schedule
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=50,

    # Gradient control
    max_grad_norm=1.0,

    # DPO-specific
    beta=0.1,                    # KL constraint strength
    loss_type="sigmoid",         # standard DPO loss

    # Mixed precision
    bf16=True,

    # Training
    num_train_epochs=1,          # DPO typically 1 epoch
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
dpo_trainer.train()
```

<span class="hh-tag">Listing 6:</span>Complete PPO and DPO optimizer configuration using TRL.

### MoE Considerations for RL Training

Miles [295] is PyTorch’s official stack for large-scale LLM RL, connecting SGLang (for high-throughput rollout generation) with Megatron-LM (for distributed training updates). It supports both disaggregated execution (separate GPU pools for rollout and training, maximizing utilization) and colocated execution (fitting both on the same nodes for smaller-scale experiments). Miles represents the standardization of what was previously ad-hoc infrastructure—analogous to how PyTorch itself standardized deep learning training a decade ago.

<aside class="hh-next">
<p class="hh-next-label">下一篇</p>
<p class="hh-next-title">The Hitchhiker's Guide to Agentic AI · Chapter 12 LLM Agentic Training</p>
<p class="hh-next-hint">在「智能体 AI 漫游指南」分类下继续阅读。</p>
</aside>

</div>
