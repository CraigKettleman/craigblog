---
title: "智能体 AI 漫游指南 · 第 2 章 面向 LLM 的系统基础"
slug: "hitchhiker-agentic-ai-02-systems-foundations-for-llms"
lang: "zh"
date: "2026-09-16T00:03:00.000Z"
draft: false
featured: false
categories:
  - "hitchhiker-agentic-ai"
cover: "./cover.png"
description: "现代大型语言模型几乎完全依赖 GPU（图形处理器，Graphics Processing Unit）进行训练和服务。理解 GPU 架构对于在并行策略、内存管理、内核优化和基础设施规模等方面做出明智决策至关重要。本节针对 LLM 工作负载，系统地介绍 GPU 硬件相关知识。"
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
<p class="hh-part-title">第 I 部分 基础</p>
</aside>

## GPU 架构——从硅片到 LLM 训练

现代大型语言模型几乎完全依赖 GPU（图形处理器，Graphics Processing Unit）进行训练和服务。理解 GPU 架构对于在并行策略、内存管理、内核优化和基础设施规模等方面做出明智决策至关重要。本节针对 LLM 工作负载，系统地介绍 GPU 硬件相关知识。

### 为什么深度学习要用 GPU？

GPU 和 CPU 代表了根本不同的硬件设计哲学。理解这种差异能解释为何 LLM 训练在 GPU 上能快 100–1000<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>。

### NVIDIA GPU 微架构代际演进

NVIDIA 已经发布了一系列 GPU 架构，每一代都为深度学习带来关键创新：

<div class="hh-table" id="ch2.t1"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>架构</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>年份</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>旗舰</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>深度学习关键创新</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">Pascal</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>2016</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>P100</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>首款 HBM GPU；FP16 支持；NVLink 1</span></span></td></tr><tr><th class="ltx_align_left">Volta</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>2017</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>V100</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Tensor Core</span>（第一代）；混合精度训练</span></span></td></tr><tr><th class="ltx_align_left">Turing</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>2018</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>T4</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>INT8 推理；RT core（非 ML 用途）</span></span></td></tr><tr><th class="ltx_align_left">Ampere</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>2020</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>A100</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>BF16 Tensor Core；TF32；第 3 代 NVLink；MIG</span></span></td></tr><tr><th class="ltx_align_left">Hopper</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>2022</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>H100</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>FP8 Tensor Core；TMA；Transformer Engine；NVLink 4</span></span></td></tr><tr><th class="ltx_align_left">Blackwell</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>2024</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>B200</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>第 2 代 Transformer Engine；NVLink 5（1.8 TB/s）；FP4</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">表 2.1：</span>面向深度学习的 NVIDIA GPU 微架构时间线。</p></div>

### LLM 训练与推理常用 GPU

<div class="hh-table" id="ch2.t2"><table class="ltx_tabular ltx_align_middle"><tbody><tr><td class="ltx_align_left"><span>GPU</span></td><td class="ltx_align_left"><span>架构</span></td><td class="ltx_align_left"><span>HBM</span></td><td class="ltx_align_left"><span>BF16 TF</span></td><td class="ltx_align_left"><span>HBM 带宽</span></td><td class="ltx_align_left"><span>NVLink</span></td><td class="ltx_align_left"><span>LLM 角色</span></td></tr><tr><td class="ltx_align_left"><span>V100-32GB</span></td><td class="ltx_align_left"><span>Volta</span></td><td class="ltx_align_left"><span>32 GB</span></td><td class="ltx_align_left"><span>125 TF*</span></td><td class="ltx_align_left"><span>900 GB/s</span></td><td class="ltx_align_left"><span>300 GB/s</span></td><td class="ltx_align_left"><span>旧型号；小模型微调</span></td></tr><tr><td class="ltx_align_left"><span>A100-40GB</span></td><td class="ltx_align_left"><span>Ampere</span></td><td class="ltx_align_left"><span>40 GB</span></td><td class="ltx_align_left"><span>312 TF</span></td><td class="ltx_align_left"><span>1.5 TB/s</span></td><td class="ltx_align_left"><span>600 GB/s</span></td><td class="ltx_align_left"><span>经济型训练/推理</span></td></tr><tr><td class="ltx_align_left"><span>A100-80GB</span></td><td class="ltx_align_left"><span>Ampere</span></td><td class="ltx_align_left"><span>80 GB</span></td><td class="ltx_align_left"><span>312 TF</span></td><td class="ltx_align_left"><span>2.0 TB/s</span></td><td class="ltx_align_left"><span>600 GB/s</span></td><td class="ltx_align_left"><span>标准 RLHF（70B 模型用 8–64 卡）</span></td></tr><tr><td class="ltx_align_left"><span>H100 SXM</span></td><td class="ltx_align_left"><span>Hopper</span></td><td class="ltx_align_left"><span>80 GB</span></td><td class="ltx_align_left"><span>990 TF</span></td><td class="ltx_align_left"><span>3.35 TB/s</span></td><td class="ltx_align_left"><span>900 GB/s</span></td><td class="ltx_align_left"><span>训练快 3<math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math></span></td></tr><tr><td class="ltx_align_left"><span>H200 SXM</span></td><td class="ltx_align_left"><span>Hopper</span></td><td class="ltx_align_left"><span>141 GB</span></td><td class="ltx_align_left"><span>990 TF</span></td><td class="ltx_align_left"><span>4.8 TB/s</span></td><td class="ltx_align_left"><span>900 GB/s</span></td><td class="ltx_align_left"><span>用更少 GPU 装下 70B 策略 + 参考模型</span></td></tr><tr><td class="ltx_align_left"><span>B200 SXM</span></td><td class="ltx_align_left"><span>Blackwell</span></td><td class="ltx_align_left"><span>192 GB</span></td><td class="ltx_align_left"><span>2250 TF</span></td><td class="ltx_align_left"><span>8.0 TB/s</span></td><td class="ltx_align_left"><span>1800 GB/s</span></td><td class="ltx_align_left"><span>新一代；比 H100 快 2<math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math></span></td></tr><tr><td class="ltx_align_left" colspan="7"><em>AMD 和 Google 替代方案：</em></td></tr><tr><td class="ltx_align_left"><span>MI300X</span></td><td class="ltx_align_left"><span>CDNA3</span></td><td class="ltx_align_left"><span>192 GB</span></td><td class="ltx_align_left"><span>1300 TF</span></td><td class="ltx_align_left"><span>5.3 TB/s</span></td><td class="ltx_align_left"><span>N/A</span></td><td class="ltx_align_left"><span>显存最大；ROCm</span></td></tr><tr><td class="ltx_align_left"><span>TPU v5e</span></td><td class="ltx_align_left"><span>Google</span></td><td class="ltx_align_left"><span>16 GB</span></td><td class="ltx_align_left"><span>197 TF</span></td><td class="ltx_align_left"><span>1.6 TB/s</span></td><td class="ltx_align_left"><span>ICI 1.6 TB/s</span></td><td class="ltx_align_left"><span>仅云端；JAX/XLA</span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">表 2.2：</span>与 LLM 工作负载相关的 GPU 规格。所有带宽数字均为双向。</p></div>

### GPU 内部架构——流式多处理器（SM）

GPU 由一组 流式多处理器（Streaming Multiprocessor，SM）阵列组成，每个 SM 都是独立的处理器，拥有自己的寄存器文件、共享内存和执行单元。理解 SM 是理解 GPU 性能的关键。

<figure id="ch2.f1"><img src="./fig_016_fig16.png" alt="图 2.1：左：A100 单个流式多处理器（SM）的内部结构——64 个 FP32 CUDA core、4 个 Tensor Core、4 个 warp 调度器、256 KB 寄存器文件以及 192 KB 共享内存/L1 缓存。右：完整 A" loading="lazy" decoding="async"><figcaption><span class="hh-tag">图 2.1：</span>左：A100 单个流式多处理器（SM）的内部结构——64 个 FP32 CUDA core、4 个 Tensor Core、4 个 warp 调度器、256 KB 寄存器文件以及 192 KB 共享内存/L1 缓存。右：完整 A100 芯片包含 108 个 SM，共享 40 MB L2 缓存和 80 GB HBM2e。左侧边栏的带宽标注显示了从寄存器到 HBM 的急剧下降。</figcaption></figure>

### 各代际 GPU 芯片的扩展

NVIDIA GPU 架构的演进显示出计算密度、片上内存和面向深度学习的专用单元在持续扩展：

<div class="hh-table" id="ch2.t3"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>架构</span></th><th class="ltx_align_left"><span>SM 数</span></th><th class="ltx_align_left"><span>TC/SM</span></th><th class="ltx_align_left"><span>SRAM/SM</span></th><th class="ltx_align_left"><span>L2</span></th><th class="ltx_align_left"><span>关键变化</span></th></tr></thead><tbody><tr><td class="ltx_align_left">Volta（V100）</td><td class="ltx_align_left">80</td><td class="ltx_align_left">8</td><td class="ltx_align_left">128 KB</td><td class="ltx_align_left">6 MB</td><td class="ltx_align_left">首次引入 Tensor Core</td></tr><tr><td class="ltx_align_left">Ampere（A100）</td><td class="ltx_align_left">108</td><td class="ltx_align_left">4</td><td class="ltx_align_left">192 KB</td><td class="ltx_align_left">40 MB</td><td class="ltx_align_left">BF16/TF32；更大的 L2</td></tr><tr><td class="ltx_align_left">Hopper（H100）</td><td class="ltx_align_left">132</td><td class="ltx_align_left">4</td><td class="ltx_align_left">256 KB</td><td class="ltx_align_left">50 MB</td><td class="ltx_align_left">TMA；FP8；Thread Block Cluster</td></tr><tr><td class="ltx_align_left">Blackwell（B200）</td><td class="ltx_align_left">148</td><td class="ltx_align_left">4</td><td class="ltx_align_left">256 KB</td><td class="ltx_align_left">128 MB</td><td class="ltx_align_left">2<math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math> 个 die；FP4；TMEM；NVLink 5</td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">表 2.3：</span>各代 NVIDIA 架构的 SM 级扩展。</p></div>

### GPU 内存层级与带宽

现代 GPU 训练和推理性能几乎完全取决于<em>你对跨内存层级的数据移动管理得好不好</em>。理解这一层级并非可选项——它是后续各节讨论的每一项优化技术的基础。

每个 CUDA 线程都可访问一个私有寄存器文件。寄存器是芯片上最快的存储——读写在一个时钟周期内完成，无需仲裁。A100 每个 SM 有 65,536 个 32 位寄存器。将寄存器溢出到本地内存（L1/L2）是重大的性能隐患。

A100 上每个 SM 拥有 192 KB 的 L1/共享内存组合池（H100 为 256 KB），其中 A100 最多可将 164 KB 配置为共享内存。共享内存由程序员显式管理（在较新的 CUDA 版本中也可由编译器管理）。例如，Flash Attention 完全建立在这样一个洞见之上：attention 的分块计算可以装入 SRAM。

A100 上 40 MB 的 L2 由全部 108 个 SM 共享。它充当 SRAM 与 HBM 之间的暂存区。对于具有良好空间局部性的工作负载（例如在一个 batch 内被反复访问的权重矩阵），L2 命中率可以大幅降低实际 HBM 流量。

HBM 是直接安装在 GPU 封装上的堆叠 DRAM，通过宽中介层连接。A100 SXM 拥有 80 GB HBM2e，带宽 2 TB/s。H100 SXM5 拥有 80 GB HBM3，带宽 3.35 TB/s。它是模型权重、KV cache、激活值和优化器状态的主要工作内存。

GPU HBM 与 CPU DRAM 之间的数据传输经由 PCIe 总线。PCIe Gen4 <math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>16 每个方向提供 <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>32 GB/s（双向 64 GB/s）；Gen5 在此基础上翻倍。与 HBM（单向）相比，这是 <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>60<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> 的带宽缩减。CPU offloading（ZeRO-Infinity、DeepSpeed）会利用这条链路，但必须谨慎使用，以免成为瓶颈。

NVMe SSD（例如 Samsung 990 Pro）的顺序读取速度可达 <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>7 GB/s。ZeRO-Infinity 可以将优化器状态卸载到 NVMe，但只有当计算与 IO 之比非常高时（大批次、慢速训练步）才可行。

### 算术强度与 Roofline 模型

<figure id="ch2.f2"><img src="./fig_017_fig17.png" alt="图 2.2：A100 BF16 的 Roofline 模型。attention 深处于内存密集区间；大型 GEMM（FFN 层）则是计算密集的。" loading="lazy" decoding="async"><figcaption><span class="hh-tag">图 2.2：</span>A100 BF16 的 Roofline 模型。attention 深处于内存密集区间；大型 GEMM（FFN 层）则是计算密集的。</figcaption></figure>

### Attention 是内存密集的；FFN 是计算密集的

### Tensor Cores

<ul><li>支持的精度：FP64、TF32、BF16、FP16、INT8、FP8（H100 及以后）。</li><li>累加：内部始终以 FP32 累加，即使输入是 BF16。这可避免点积过程中出现灾难性抵消。</li><li>要求：当矩阵维度是 8（BF16）或 16（FP8）的倍数时，Tensor Core 效率最高。填充到这些倍数通常是值得的。</li><li>WGMMA（H100）：Hopper 引入了 warpgroup 级别的 MMA 指令，它们作用于更大的分块（64<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>256<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>16），并且可以与 TMA（Tensor Memory Accelerator）数据搬运形成流水线。</li></ul>

### 通信架构——NVLink、InfiniBand 与 PCIe

分布式 LLM 训练与推理需要在 GPU、节点和存储之间移动海量数据。通信网络往往是大规模训练的瓶颈。

PCIe 用于：

<ul><li>CPU <math alttext="\leftrightarrow" display="inline"><semantics><mo stretchy="false">↔</mo></semantics></math> GPU 数据传输（模型加载、CPU offloading）</li><li>NVLink 不可用时的跨节点 GPU 通信（少见，且非常慢）</li><li>NVMe 存储访问（经由 CPU）</li></ul>

NVLink 是同一节点内 GPU 之间的点对点互联。每条链路都是双向的。H100 SXM5 拥有 18 条 NVLink 4 链路，每条提供 50 GB/s 双向带宽，合计 900 GB/s。

在 DGX H100 系统中，全部 8 个 GPU 通过 NVSwitch 连接——这是一款专用交换芯片，可提供<em>完全二分带宽（full bisection bandwidth）</em>。这意味着任意 GPU 都能以完整的 NVLink 速度同时与任意其他 GPU 通信，而不只是环形拓扑中的邻居。

对于节点（服务器）之间的通信，InfiniBand 提供高带宽、低延迟的网络，并支持直接访问 GPU 内存。

大型 GPU 集群使用胖树网络拓扑。一个采用 <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 端口交换机的三级胖树可支持 <math alttext="k^{3}/4" display="inline"><semantics><mrow><msup><mi>k</mi><mn>3</mn></msup><mo>/</mo><mn>4</mn></mrow></semantics></math> 个节点，并具有完全二分带宽。对于具有 <math alttext="k=64" display="inline"><semantics><mrow><mi>k</mi><mo>=</mo><mn>64</mn></mrow></semantics></math> 个端口的 400Gb/s NDR 交换机：<math alttext="64^{3}/4=65{,}536" display="inline"><semantics><mrow><mrow><msup><mn>64</mn><mn>3</mn></msup><mo>/</mo><mn>4</mn></mrow><mo>=</mo><mn>65,536</mn></mrow></semantics></math> 个节点。

在实践中，集群使用<em>轨道优化（rail-optimized）</em>拓扑，其中节点内的每个 GPU 连接到不同的机架顶端（top-of-rack）交换机。这可确保 AllReduce 操作（涉及所有 GPU）同时使用全部网络链路，从而最大化带宽。

分布式训练依赖集合通信原语。原语的选择决定了带宽需求和扩展行为。

下图展示了一个典型的两节点 GPU 集群拓扑，其中同时包含节点内（NVLink）和节点间（InfiniBand）通信路径。

<figure id="ch2.f3"><img src="./fig_018_fig18.png" alt="图 2.3：两节点 8 GPU 拓扑。节点内：通过 NVSwitch 的 NVLink 4（总计 900 GB/s）。节点间：通过机架顶端（top-of-rack）交换机的 InfiniBand NDR 400Gb/s。每个节点有 8 个 " loading="lazy" decoding="async"><figcaption><span class="hh-tag">图 2.3：</span>两节点 8 GPU 拓扑。节点内：通过 NVSwitch 的 NVLink 4（总计 900 GB/s）。节点间：通过机架顶端（top-of-rack）交换机的 InfiniBand NDR 400Gb/s。每个节点有 8 个 IB NIC（每个 GPU 一个），用于轨道优化的 AllReduce。</figcaption></figure>

## vLLM——分页注意力与高吞吐推理

vLLM [196] 提出了 PagedAttention，它借用操作系统用于 RAM 的分页抽象，并将其应用于 GPU 的 KV cache。在 LLM 推理过程中，<em>KV cache</em>——为所有先前 token 存储的 key 和 value 张量——是主要的内存消耗者。高效地管理它是高吞吐推理的核心挑战。

### KV cache 碎片化问题

### 分页注意力——为 KV cache 引入虚拟内存

PagedAttention（Kwon 等，2023）借用了操作系统的<em>分页</em>抽象。KV cache 不再为每条序列分配一个连续块，而是被切分为固定大小的页（page）（块），并由一张类似于 CPU 页表的间接表把每条序列的逻辑 token 位置转换为分散的物理 GPU 内存地址。

### PagedAttention 的收益

内部碎片至多为每条序列一个部分填充块（最后一个块）。当块大小为 16 时，最坏情况下的浪费为每条序列 15 个 token——可忽略不计。外部碎片被消除，因为块是固定大小且可互换的。

块随序列增长按需分配。无需预先知道最终序列长度。这对生成任务至关重要，因为输出长度未知。

多条共享同一前缀（例如同一系统提示）的序列可以共享该前缀的 <em>相同物理块</em>。块表只需让多条序列指向相同的物理块。当某条序列需要写入共享块（与前缀分叉）时，会触发写时复制。

当 GPU 显存耗尽时，vLLM 可以通过把某条序列的 KV 块换出到 CPU DRAM（或者直接丢弃，之后重新计算）来 <em>抢占</em> 该序列。这之所以可行，是因为块是自包含且非连续的——若是连续分配，则需要复制整个缓冲区。

### 连续批处理

传统的批处理（「静态批处理」）会等到一批中 <em>所有</em> 序列都完成后才开始新批次。如果一条序列生成 500 个 token、另一条生成 10 个，那么 GPU 在短序列上有 490 步是空闲的。这极其浪费。

### vLLM 中的投机解码

投机解码（Speculative Decoding）使用一个小型 <em>草稿模型</em>（例如 1B 参数）快速提出 <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 个候选 token，再由大型 <em>目标模型</em>在一次前向传播中验证。从第一个被拒绝处之前的所有 token 都会被接受（预期接受数：每次验证 3–5 个 token）。这为延迟敏感的单序列生成带来 2–3<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> 提速，且无任何质量损失。

vLLM 将投机解码与 PagedAttention 集成：

<ul><li>草稿 token 被分配投机性 KV 块</li><li>拒绝时，投机块被释放（在分页分配下成本极低）</li><li>接受时，投机块被提升为主序列的一部分</li><li>块表更新是 <math alttext="O(k)" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>k</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math>——只需更新少量表项</li></ul>

### 具体显存节约——大规模 70B 模型

### vLLM：端到端系统

vLLM 将 PagedAttention 封装在一整套服务栈中：连续批处理（Continuous Batching）、前缀缓存（Prefix Caching）、投机解码（Speculative Decoding）以及张量并行的模型分片协同工作，最大化每美元 GPU 的吞吐。

### 架构总览

<figure id="ch2.f4"><img src="./fig_019_fig19.png" alt="图 2.4：vLLM 架构：请求自上而下流动。Scheduler 管理准入与抢占，Block Manager 负责虚拟到物理 KV cache 映射（像操作系统页表一样），Model Executor 在 GPU HBM 中从预分配的块池读" loading="lazy" decoding="async"><figcaption><span class="hh-tag">图 2.4：</span>vLLM 架构：请求自上而下流动。Scheduler 管理准入与抢占，Block Manager 负责虚拟到物理 KV cache 映射（像操作系统页表一样），Model Executor 在 GPU HBM 中从预分配的块池读取数据并执行批量推理。</figcaption></figure>

### 核心组件

<ul><li>API Server：接收 OpenAI 兼容的请求（completions、chat）。对输入进行 tokenize，并创建「序列组（sequence groups）」（用于 beam search 或多样本）。</li><li>Scheduler：vLLM 的大脑。维护三个队列：<span class="hh-tag">–</span><code>waiting</code>：尚未开始的新请求（等待 prefill）<span class="hh-tag">–</span><code>running</code>：正在生成 token（decode 阶段）<span class="hh-tag">–</span><code>swapped</code>：被抢占、KV cache 已卸载到 CPU 的请求每次迭代，调度器根据可用 GPU 内存块决定运行哪些请求。</li><li>Block Manager：为 KV cache 实现虚拟内存抽象。将逻辑块（按序列）映射到物理块（GPU 内存池中）。负责处理：<span class="hh-tag">–</span>分配（生成新 token <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 需要新块）<span class="hh-tag">–</span>写时复制（Copy-on-Write）（用于 beam search：多 beam 共享前缀块，仅在分叉时复制）<span class="hh-tag">–</span>Swap（抢占/恢复时在 GPU <math alttext="\leftrightarrow" display="inline"><semantics><mo stretchy="false">↔</mo></semantics></math> CPU 之间迁移）<span class="hh-tag">–</span>前缀缓存（Prefix Caching）（当 prompt 共享相同前缀时复用缓存的块）</li><li>Model Executor：运行实际的 LLM 前向传播。管理跨 GPU 的张量并行，调度从分页 KV cache 块读取的 attention kernel。</li><li>KV Cache Pool：预分配的 GPU 内存，划分为固定大小的块（默认：每块 16 个 token <math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> num_heads <math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> head_dim <math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> 2 字节）。运行时无动态分配 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 零碎片。</li></ul>

### 请求生命周期（端到端流程）

<ol><li>到达：客户端发送 prompt。API server 对其 tokenize，创建一个 <code>SequenceGroup</code>，并放入 <code>waiting</code> 队列。</li><li>调度：每一步，调度器执行：<span class="hh-tag">(a)</span>检查是否有 <code>swapped</code> 序列可以恢复（有足够的空闲块）。<span class="hh-tag">(b)</span>检查是否有 <code>waiting</code> 序列可以开始 prefill（有足够的块容纳完整 prompt）。<span class="hh-tag">(c)</span>在 <code>running</code> 序列之间分配剩余块预算（如果当前块已满，每条序列每步需要 1 个新块）。<span class="hh-tag">(d)</span>若超出预算：抢占优先级最低的运行中序列（将 KV 换出到 CPU 或稍后重新计算）。</li><li>Prefill（请求的第一次迭代）：整个 prompt 在一次前向传播中被处理。为所有 prompt token 计算 KV cache 并存入已分配的块。这是计算受限的（大批量 token）。</li><li>Decode（后续迭代）：每一步每条序列生成一个新 token。所有运行中的序列被一起批处理（连续批处理）。这是内存受限的（读取完整 KV cache，生成 1 个 token）。</li><li>块分配：每个 decode 步之后，如果某条序列的最后一个块已满，Block Manager 会分配一个新的物理块，并将其映射到下一个逻辑块。</li><li>完成：当某条序列遇到 EOS 或达到最大长度时，它会被从 <code>running</code> 中移除。其物理块立即被释放 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 可供其他序列使用。响应以流式返回客户端。</li></ol>

### 前缀缓存（自动 Prompt 缓存）

当多个请求共享同一前缀（系统提示、few-shot 示例）时：

<ol><li>对每个逻辑块的 token 内容做哈希。</li><li>新请求到达时，检查缓存中是否已有任何前缀块。</li><li>若命中：跳过这些 token 的 prefill，直接复用物理 KV 块。首词元延迟（Time-to-first-token）大幅下降。</li><li>驱逐：LRU 策略。只有在内存压力需要时才释放缓存的块。</li></ol>

影响：对于带长系统提示（所有用户共享 2K+ token）的聊天应用，前缀缓存可将 TTFT 降低 60–80%。

### vLLM 中的引导（受约束）解码

vLLM 通过可插拔后端原生支持约束解码（第 1.12.11 受限解码（Constrained Decoding，结构化生成 Structured Generation） 节），从而在服务时以极小的性能开销提供 <em>有保证的</em> 结构化输出。

OpenAI 兼容的 API 通过 <code>guided_*</code> 参数或 <code>response_format</code> 字段接受约束：

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1")

# --- JSON Schema constraint ---
response = client.chat.completions.create(
    model="meta-llama/Llama-3-70B-Instruct",
    messages=[{"role": "user",
               "content": "Extract: name, age, city from: "
                          "'John is 30 and lives in NYC'"}],
    extra_body={
        "guided_json": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "city": {"type": "string"}
            },
            "required": ["name", "age", "city"]
        }
    }
)
# Output is guaranteed valid JSON matching the schema

# --- Regex constraint ---
response = client.completions.create(
    model="meta-llama/Llama-3-70B-Instruct",
    prompt="Generate an IPv4 address: ",
    extra_body={
        "guided_regex": r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    }
)

# --- Choice constraint ---
response = client.completions.create(
    model="meta-llama/Llama-3-70B-Instruct",
    prompt="Sentiment: ",
    extra_body={"guided_choice": ["positive", "negative", "neutral"]}
)
```

vLLM 把掩码计算委托给后端引擎：

<ul><li>XGrammar（自 v0.7 起为默认）：下推自动机引擎，支持 JSON schema、正则表达式和任意 EBNF 文法。由于高效的 C++ 内核，对复杂 schema 速度最快。</li><li>Outlines[378]：基于 FSM；支持 JSON 和正则。在 XGrammar 不可用时作为回退。</li></ul>

掩码在模型前向传播产生 logits <em>之后</em>、采样 <em>之前</em>被应用——实际中每步增加 <math alttext="&lt;" display="inline"><semantics><mo>&lt;</mo></semantics></math>1 ms，因为 FSM/PDA 状态转移和预计算的索引查找都是 <math alttext="O(1)" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></mrow></semantics></math>。

由于约束只对 logits 做掩码（不重新计算 attention 或 FFN），吞吐损失可以忽略不计（基准测试中 <math alttext="&lt;" display="inline"><semantics><mo>&lt;</mo></semantics></math>2%）。主要成本是把 schema <em>编译</em>为 FSM/PDA 索引，依 schema 复杂度需要 0.5–5 s。vLLM 会在请求之间缓存已编译的 schema，因此每个唯一 schema 只需支付一次该成本。

<div class="hh-table" id="ch2.t4"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>指标</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>vLLM</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>HF Generate</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>原因</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">吞吐（tok/s）</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>2,500–4,000</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>300–600</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>连续批处理 + PagedAttention</span></span></td></tr><tr><th class="ltx_align_left">显存利用率</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>90–95%</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>50–60%</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>零碎片，动态块分配</span></span></td></tr><tr><th class="ltx_align_left">最大并发序列数</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>200–500</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>16–32</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>分页 KV 消除了按序列预留</span></span></td></tr><tr><th class="ltx_align_left">首词元延迟</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>100–300ms</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>500–2000ms</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>对重复的系统提示使用前缀缓存</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">表 2.4：</span>vLLM 与其他方案的性能对比（70B 模型，A100 <math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> 4，TP=4）。</p></div>

Dynamo [270] 是 NVIDIA 的开源编排层，位于各个推理引擎（SGLang、TensorRT-LLM、vLLM）<em>之上</em>，并将它们转变为一个协调的多节点服务系统。关键能力包括分离式服务（把 prefill 和 decode 阶段分散到专门的节点池）、基于 KV cache 局部性的智能请求路由、多级 KV 缓存（GPU <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> CPU <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> SSD）以及自动扩缩容。它弥合了单 GPU 推理优化（由 vLLM 覆盖）与数据中心规模的生产服务之间的差距。

<aside class="hh-next">
<p class="hh-next-label">下一篇</p>
<p class="hh-next-title">智能体 AI 漫游指南 · 第 3 章 强化学习导论</p>
<p class="hh-next-hint">在「智能体 AI 漫游指南」分类下继续阅读。</p>
</aside>

</div>
