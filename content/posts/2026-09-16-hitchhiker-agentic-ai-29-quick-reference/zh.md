---
title: "智能体 AI 漫游指南 · 第 29 章 速查手册"
slug: "hitchhiker-agentic-ai-29-quick-reference"
lang: "zh"
date: "2026-09-16T00:30:00.000Z"
draft: false
featured: false
categories:
  - "hitchhiker-agentic-ai"
cover: "./cover.png"
description: "本章汇总了关键公式、架构规格、API 参考以及失效模式诊断，便于开发与调试时快速查阅。"
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

本章汇总了关键公式、架构规格、API 参考以及失效模式诊断，便于开发与调试时快速查阅。

## 核心 RL 与对齐公式



## Transformer 与架构公式



## 解码方法

方法公式 / 规则关键参数Greedy<math alttext="y_{t}=\arg\max_{v}P(v|y_{&lt;t})" display="inline"><semantics><mrow><msub><mi>y</mi><mi>t</mi></msub><mo>=</mo><mrow><mi>arg</mi><mo lspace="0.167em">⁡</mo><mrow><msub><mi>max</mi><mi>v</mi></msub><mo lspace="0.167em">⁡</mo><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>v</mi><mo lspace="0em" rspace="0em">|</mo><msub><mi>y</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>t</mi></mrow></msub><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></mrow></semantics></math>—Beam search按联合概率保留 top-<math alttext="B" display="inline"><semantics><mi>B</mi></semantics></math> 个部分序列<math alttext="B=4" display="inline"><semantics><mrow><mi>B</mi><mo>=</mo><mn>4</mn></mrow></semantics></math>–<math alttext="8" display="inline"><semantics><mn>8</mn></semantics></math>Temperature<math alttext="P^{\prime}(v)=\text{softmax}(\text{logit}_{v}/T)" display="inline"><semantics><mrow><mrow><msup><mi>P</mi><mo>′</mo></msup><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>v</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mtext>softmax</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mrow><msub><mtext>logit</mtext><mi>v</mi></msub><mo>/</mo><mi>T</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math><math alttext="T\in[0.1,1.5]" display="inline"><semantics><mrow><mi>T</mi><mo>∈</mo><mrow><mo stretchy="false">[</mo><mrow><mn>0.1</mn><mo>,</mo><mn>1.5</mn></mrow><mo stretchy="false">]</mo></mrow></mrow></semantics></math>Top-<math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math>仅保留 top-<math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 个 logit，其余置零后重新归一化<math alttext="k=40" display="inline"><semantics><mrow><mi>k</mi><mo>=</mo><mn>40</mn></mrow></semantics></math>–<math alttext="100" display="inline"><semantics><mn>100</mn></semantics></math>Top-<math alttext="p" display="inline"><semantics><mi>p</mi></semantics></math> (nucleus)保留最小集合 <math alttext="V^{\prime}" display="inline"><semantics><msup><mi>V</mi><mo>′</mo></msup></semantics></math>，使得 <math alttext="\sum_{v\in V^{\prime}}P(v)\geq p" display="inline"><semantics><mrow><mrow><msub><mo>∑</mo><mrow><mi>v</mi><mo>∈</mo><msup><mi>V</mi><mo>′</mo></msup></mrow></msub><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>v</mi><mo stretchy="false">)</mo></mrow></mrow></mrow><mo>≥</mo><mi>p</mi></mrow></semantics></math><math alttext="p=0.9" display="inline"><semantics><mrow><mi>p</mi><mo>=</mo><mn>0.9</mn></mrow></semantics></math>–<math alttext="0.95" display="inline"><semantics><mn>0.95</mn></semantics></math>Min-<math alttext="p" display="inline"><semantics><mi>p</mi></semantics></math>保留满足 <math alttext="P(v)\geq p_{\text{min}}\cdot P(v_{\text{max}})" display="inline"><semantics><mrow><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>v</mi><mo stretchy="false">)</mo></mrow></mrow><mo>≥</mo><mrow><msub><mi>p</mi><mtext>min</mtext></msub><mo lspace="0.222em" rspace="0.222em">⋅</mo><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>v</mi><mtext>max</mtext></msub><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></semantics></math> 的 token<math alttext="p_{\text{min}}=0.05" display="inline"><semantics><mrow><msub><mi>p</mi><mtext>min</mtext></msub><mo>=</mo><mn>0.05</mn></mrow></semantics></math>–<math alttext="0.1" display="inline"><semantics><mn>0.1</mn></semantics></math>Repetition penalty若 <math alttext="v" display="inline"><semantics><mi>v</mi></semantics></math> 已出现过，则 <math alttext="\text{logit}_{v}\leftarrow\text{logit}_{v}/\theta" display="inline"><semantics><mrow><msub><mtext>logit</mtext><mi>v</mi></msub><mo stretchy="false">←</mo><mrow><msub><mtext>logit</mtext><mi>v</mi></msub><mo>/</mo><mi>θ</mi></mrow></mrow></semantics></math><math alttext="\theta=1.1" display="inline"><semantics><mrow><mi>θ</mi><mo>=</mo><mn>1.1</mn></mrow></semantics></math>–<math alttext="1.3" display="inline"><semantics><mn>1.3</mn></semantics></math>

## 系统与并行

公式取值 (70B, BF16)说明模型显存<math alttext="2P" display="inline"><semantics><mrow><mn>2</mn><mo lspace="0em" rspace="0em">​</mo><mi>P</mi></mrow></semantics></math> 字节<math alttext="140" display="inline"><semantics><mn>140</mn></semantics></math> GB（仅权重）Adam 优化器<math alttext="2P\times 4" display="inline"><semantics><mrow><mrow><mn>2</mn><mo lspace="0em" rspace="0em">​</mo><mi>P</mi></mrow><mo lspace="0.222em" rspace="0.222em">×</mo><mn>4</mn></mrow></semantics></math> 字节 (m + v)<math alttext="280" display="inline"><semantics><mn>280</mn></semantics></math> GB完整训练占用<math alttext="\sim 8P" display="inline"><semantics><mrow><mphantom></mphantom><mo>∼</mo><mrow><mn>8</mn><mo lspace="0em" rspace="0em">​</mo><mi>P</mi></mrow></mrow></semantics></math> 字节<math alttext="560" display="inline"><semantics><mn>560</mn></semantics></math> GB（权重 + 优化器 + 梯度）FSDP 每 GPU 显存<math alttext="8P/N_{\text{GPUs}}" display="inline"><semantics><mrow><mrow><mn>8</mn><mo lspace="0em" rspace="0em">​</mo><mi>P</mi></mrow><mo>/</mo><msub><mi>N</mi><mtext>GPUs</mtext></msub></mrow></semantics></math>8 卡时为 <math alttext="70" display="inline"><semantics><mn>70</mn></semantics></math> GB生成算术强度<math alttext="2P/2P=1" display="inline"><semantics><mrow><mrow><mrow><mrow><mn>2</mn><mo lspace="0em" rspace="0em">​</mo><mi>P</mi></mrow><mo>/</mo><mn>2</mn></mrow><mo lspace="0em" rspace="0em">​</mo><mi>P</mi></mrow><mo>=</mo><mn>1</mn></mrow></semantics></math> FLOP/byte严重内存瓶颈Token 速率（生成）HBM_BW <math alttext="/(2P)" display="inline"><semantics><mrow><mphantom></mphantom><mo>/</mo><mrow><mo stretchy="false">(</mo><mrow><mn>2</mn><mo lspace="0em" rspace="0em">​</mo><mi>P</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></semantics></math><math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>14 tok/s (A100, batch=1)TP AllReduce / 层<math alttext="2\times 2\cdot\frac{T-1}{T}\cdot bsd" display="inline"><semantics><mrow><mrow><mn>2</mn><mo lspace="0.222em" rspace="0.222em">×</mo><mn>2</mn></mrow><mo lspace="0.222em" rspace="0.222em">⋅</mo><mfrac><mrow><mi>T</mi><mo>−</mo><mn>1</mn></mrow><mi>T</mi></mfrac><mo lspace="0.222em" rspace="0.222em">⋅</mo><mrow><mi>b</mi><mo lspace="0em" rspace="0em">​</mo><mi>s</mi><mo lspace="0em" rspace="0em">​</mo><mi>d</mi></mrow></mrow></semantics></math> 字节<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>188 MB (70B, TP=8)PP 气泡占比<math alttext="(P-1)/(P+M-1)" display="inline"><semantics><mrow><mrow><mo stretchy="false">(</mo><mrow><mi>P</mi><mo>−</mo><mn>1</mn></mrow><mo stretchy="false">)</mo></mrow><mo>/</mo><mrow><mo stretchy="false">(</mo><mrow><mrow><mi>P</mi><mo>+</mo><mi>M</mi></mrow><mo>−</mo><mn>1</mn></mrow><mo stretchy="false">)</mo></mrow></mrow></semantics></math><math alttext="P" display="inline"><semantics><mi>P</mi></semantics></math> = 阶段数，<math alttext="M" display="inline"><semantics><mi>M</mi></semantics></math> = 微批数MFUobserved_toks <math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> 6<math alttext="P" display="inline"><semantics><mi>P</mi></semantics></math> / peak_FLOPS目标：<math alttext="&gt;40\%" display="inline"><semantics><mrow><mphantom></mphantom><mo>&gt;</mo><mrow><mn>40</mn><mo>%</mo></mrow></mrow></semantics></math>

## GPU 硬件规格

GPU显存带宽 (HBM)BF16 TFLOPSNVLink备注A100-80GB80 GB HBM2e2.0 TB/s312600 GB/s主力型号，广泛可用H100-80GB80 GB HBM33.35 TB/s989900 GB/s当前一代，支持 FP8H200-141GB141 GB HBM3e4.8 TB/s989900 GB/s大上下文 / 更少 GPUB200192 GB HBM3e8.0 TB/s22501800 GB/s下一代（2025）

## 超参数范围

参数典型范围默认值说明<math alttext="\beta" display="inline"><semantics><mi>β</mi></semantics></math> (DPO/KTO)0.05–0.50.1越大越保守<math alttext="\epsilon" display="inline"><semantics><mi>ϵ</mi></semantics></math> (PPO clip)0.1–0.30.2越大更新越激进<math alttext="\gamma" display="inline"><semantics><mi>γ</mi></semantics></math> (GAE 折扣)0.99–1.01.0情景式任务使用 1.0<math alttext="\lambda" display="inline"><semantics><mi>λ</mi></semantics></math> (GAE)0.9–0.990.95越小偏差越大、方差越小KL 系数 (<math alttext="\beta_{\text{KL}}" display="inline"><semantics><msub><mi>β</mi><mtext>KL</mtext></msub></semantics></math>)0.01–0.20.05自适应目标 KL <math alttext="\approx" display="inline"><semantics><mo>≈</mo></semantics></math> 5–8学习率 (RLHF)1e-7 – 5e-65e-7远低于预训练学习率 (SFT)1e-5 – 5e-52e-5标准微调范围LoRA 秩 <math alttext="r" display="inline"><semantics><mi>r</mi></semantics></math>8–12816–64<math alttext="r" display="inline"><semantics><mi>r</mi></semantics></math> 越大，容量越大、显存越多LoRA alpha <math alttext="\alpha" display="inline"><semantics><mi>α</mi></semantics></math><math alttext="r" display="inline"><semantics><mi>r</mi></semantics></math> – <math alttext="2r" display="inline"><semantics><mrow><mn>2</mn><mo lspace="0em" rspace="0em">​</mo><mi>r</mi></mrow></semantics></math><math alttext="2r" display="inline"><semantics><mrow><mn>2</mn><mo lspace="0em" rspace="0em">​</mo><mi>r</mi></mrow></semantics></math>缩放因子；<math alttext="\alpha/r" display="inline"><semantics><mrow><mi>α</mi><mo>/</mo><mi>r</mi></mrow></semantics></math> 为有效尺度温度（生成）0.6–1.00.7越低候选越同质生成数量 <math alttext="K" display="inline"><semantics><mi>K</mi></semantics></math>4–644–16用于 GRPO / Online DPO / Best-of-N梯度裁剪范数0.5–2.01.0防止梯度爆炸

## TRL API 速查

Trainer方法关键配置数据格式<code>SFTTrainer</code>监督微调（SFT）<code>packing, max_seq_length</code>prompt + completion<code>RewardTrainer</code>奖励模型<code>center_rewards_coefficient</code>prompt + chosen + rejected<code>PPOTrainer</code>PPO<code>init_kl_coef, target_kl, cliprange</code>prompts (online gen)<code>DPOTrainer</code>DPO/IPO<code>beta, loss_type="sigmoid"/"ipo"</code>prompt + chosen + rejected<code>GRPOTrainer</code>GRPO<code>num_generations, beta, use_vllm</code>prompts + reward_fn<code>OnlineDPOTrainer</code>Online DPO<code>num_generations, reward_model_path</code>prompts (online gen)<code>KTOTrainer</code>KTO<code>desirable_weight, undesirable_weight</code>prompt + completion + label<code>ORPOTrainer</code>ORPO<code>beta</code>prompt + chosen + rejected<code>Best-of-N (manual)</code>Best-of-N<code>n_samples</code>prompts (inference)

## RAG 流水线公式



## 智能体设计模式

模式结构适用场景ReAct思考 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 行动 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 观察 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 循环通用工具使用型智能体Plan-and-Execute计划 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 执行步骤 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 修订长时程、结构化任务主管（Supervisor）路由器 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 专家智能体多领域、子任务边界清晰蜂群（Swarm，交接）智能体转移控制权 + 上下文客户服务、升级流程层级式（Hierarchical）委派式智能体树复杂任务分解Human-in-the-loop智能体 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 审批门控 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 继续高风险、不可逆操作

## 智能体通信协议

协议范围传输方式关键概念MCP（模型上下文协议）工具集成stdio / HTTP+SSE服务端暴露工具；客户端发现并调用A2A智能体到智能体HTTP + JSON-RPC任务具有生命周期（submitted<math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math>working<math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math>done）OpenAI Function Calling工具使用API 载荷<code>tools[]</code> 数组中的 JSON schema

## 上下文窗口预算

<div class="hh-equation" id="ch29.e22"><math alttext="C\geq\underbrace{S}_{\text{system}}+\underbrace{M}_{\text{memory/RAG}}+\underbrace{T}_{\text{tool defs}}+\underbrace{H}_{\text{history}}+\underbrace{R}_{\text{reserved output}}" display="block"><semantics><mrow><mi>C</mi><mo>≥</mo><mrow><munder><munder accentunder="true"><mi>S</mi><mo stretchy="true">⏟</mo></munder><mtext>system</mtext></munder><mo>+</mo><munder><munder accentunder="true"><mi>M</mi><mo stretchy="true">⏟</mo></munder><mtext>memory/RAG</mtext></munder><mo>+</mo><munder><munder accentunder="true"><mi>T</mi><mo stretchy="true">⏟</mo></munder><mtext>tool defs</mtext></munder><mo>+</mo><munder><munder accentunder="true"><mi>H</mi><mo stretchy="true">⏟</mo></munder><mtext>history</mtext></munder><mo>+</mo><munder><munder accentunder="true"><mi>R</mi><mo stretchy="true">⏟</mo></munder><mtext>reserved output</mtext></munder></mrow></mrow></semantics></math><span class="hh-equation-number">(29.22)</span></div>

经验法则（针对 128K 上下文）：

<ul><li>系统提示：1–4K tokens（固定）</li><li>工具定义：2–8K（随工具数量增长）</li><li>RAG 上下文：4–16K（top-<math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 个分块）</li><li>历史记录：无界增长 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 摘要/截断</li><li>预留输出：2–8K</li></ul>

## 常见失效模式与修复

症状可能原因修复奖励上升，质量下降奖励作弊RM 集成、长度惩罚、提高 <math alttext="\beta" display="inline"><semantics><mi>β</mi></semantics></math>KL 爆炸（<math alttext="&gt;" display="inline"><semantics><mo>&gt;</mo></semantics></math>15）学习率过高或模式崩溃降低学习率、回滚检查点熵崩溃过早收敛提高熵系数、提高温度训练损失 NaN梯度爆炸降低学习率、增大梯度裁剪、检查数据训练 5K 步后无提升prompt 分布不佳Goldilocks 过滤（通过率 20–80%）基准回退对齐税减少 RL 预算、使用 LoRA、混入 SFT 数据长度单调增长RM 中的长度利用长度惩罚、用长度控制重新训练 RM生成期间 OOMKV cache 溢出减小 batch、增大 TP、PagedAttention智能体无限循环缺少最大迭代次数护栏设置 <code>max_iterations</code>，添加循环检测工具调用解析失败输出格式不一致Few-shot 示例、受约束解码RAG 返回无关文档Embedding / 分块不佳重排序器、混合检索、更小的分块多智能体死锁循环依赖强制 DAG、为每个智能体设置超时

## 方法选择决策树

<ol><li>有成对的偏好数据（chosen + rejected）？<span class="hh-tag">•</span>标签有噪声 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math>IPO<span class="hh-tag">•</span>显存受限，尚未做 SFT <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math>ORPO<span class="hh-tag">•</span>数据干净、算力有限 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math>DPO<span class="hh-tag">•</span>DPO 出现平台期，需要探索 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math>Online DPO</li><li>只有二元反馈（赞/踩）？<math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math>KTO</li><li>有可验证的奖励（数学/代码）？<math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math>GRPO</li><li>需要最高质量，不计成本？<math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math>PPO</li><li>想要免训练的提升？<math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math>Best-of-N</li></ol>

## 评估指标

指标范围衡量内容困惑度（Perplexity）<math alttext="[1,\infty)" display="inline"><semantics><mrow><mo stretchy="false">[</mo><mn>1</mn><mo>,</mo><mi mathvariant="normal">∞</mi><mo stretchy="false">)</mo></mrow></semantics></math>模型的惊讶程度；越低表示语言建模越好胜率（相对于基线）<math alttext="[0,1]" display="inline"><semantics><mrow><mo stretchy="false">[</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></semantics></math>被评判器/人类偏好的输出占比BLEU<math alttext="[0,1]" display="inline"><semantics><mrow><mo stretchy="false">[</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></semantics></math>与参考译文的 <math alttext="n" display="inline"><semantics><mi>n</mi></semantics></math>-gram 重叠（偏重精确率）ROUGE-L<math alttext="[0,1]" display="inline"><semantics><mrow><mo stretchy="false">[</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></semantics></math>与参考译文的最长公共子序列Pass@<math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math><math alttext="[0,1]" display="inline"><semantics><mrow><mo stretchy="false">[</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></semantics></math><math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 个代码样本中 <math alttext="\geq" display="inline"><semantics><mo>≥</mo></semantics></math>1 个通过测试的概率MMLU / GPQA<math alttext="[0,1]" display="inline"><semantics><mrow><mo stretchy="false">[</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></semantics></math>知识与推理基准上的多选准确率HumanEval<math alttext="[0,1]" display="inline"><semantics><mrow><mo stretchy="false">[</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></semantics></math>生成代码的功能正确性忠实性（RAG）<math alttext="[0,1]" display="inline"><semantics><mrow><mo stretchy="false">[</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></semantics></math>被检索上下文支撑的论断占比上下文相关性<math alttext="[0,1]" display="inline"><semantics><mrow><mo stretchy="false">[</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></semantics></math>检索到的内容中与查询相关的占比答案相关性<math alttext="[0,1]" display="inline"><semantics><mrow><mo stretchy="false">[</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></semantics></math>答案切合问题的程度

## 推理与测试时扩展

方法计算开销机制思维链（CoT）1.5–3<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> tokens在 prompt 中写「Think step by step」自洽性（Self-Consistency）<math alttext="N\times" display="inline"><semantics><mrow><mi>N</mi><mo lspace="0.222em">×</mo></mrow></semantics></math> 次生成采样 <math alttext="N" display="inline"><semantics><mi>N</mi></semantics></math> 条 CoT 路径，对最终答案多数投票思维树（ToT）<math alttext="B\times D\times" display="inline"><semantics><mrow><mi>B</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mi>D</mi><mo lspace="0.222em">×</mo></mrow></semantics></math> 次生成在推理树上做 BFS/DFS；评估各分支Best-of-<math alttext="N" display="inline"><semantics><mi>N</mi></semantics></math><math alttext="N\times" display="inline"><semantics><mrow><mi>N</mi><mo lspace="0.222em">×</mo></mrow></semantics></math> 次生成采样 <math alttext="N" display="inline"><semantics><mi>N</mi></semantics></math> 个，用 RM 打分，取最高分束搜索（用于推理）<math alttext="B\times" display="inline"><semantics><mrow><mi>B</mi><mo lspace="0.222em">×</mo></mrow></semantics></math> 次生成保留 top-<math alttext="B" display="inline"><semantics><mi>B</mi></semantics></math> 条部分推理链预算强制可变动态为更难的问题分配更多 token验证（ORM/PRM）<math alttext="N\times" display="inline"><semantics><mrow><mi>N</mi><mo lspace="0.222em">×</mo></mrow></semantics></math> 次生成 + 打分生成 <math alttext="N" display="inline"><semantics><mi>N</mi></semantics></math> 个解，按结果/过程 RM 排序

## 记忆系统类型

类型存储使用场景工作记忆上下文窗口当前对话、即时的工具结果情景记忆（Episodic memory）向量存储过往交互、用户偏好、会话历史语义记忆知识图谱 / Embedding事实、概念、领域知识程序性记忆技能库 / 代码操作步骤、习得的工作流

## MCP 速查

原语方向是否有副作用？用途工具客户端 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 服务端是执行操作（创建、修改、删除）资源客户端 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 服务端否（只读）读取数据（文件、数据库记录、配置）Prompts客户端 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 服务端否常见任务的可复用模板采样服务端 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 客户端否服务端请求客户端进行 LLM 生成

传输方式：<code>stdio</code>（本地子进程）或 <code>HTTP+SSE</code>（远程、可流式）。<br>

能力发现：客户端在连接初始化时调用 <code>tools/list</code>、<code>resources/list</code>、<code>prompts/list</code>。<br>

工具注解：<code>readOnlyHint</code>、<code>destructiveHint</code>、<code>idempotentHint</code>、<code>openWorldHint</code>。

## A2A 协议速查

概念描述Agent Card位于 <code>/.well-known/agent.json</code> 的 JSON——名称、技能、支持的内容类型任务工作单元：<code>id</code>、<code>status</code>、<code>artifacts</code>。生命周期：submitted <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> working <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> completed/failed消息任务内的通信单元（role：user/agent，parts：text/file/data）产物智能体产生的输出（结构化数据、文件、生成内容）推送通知面向长时运行任务的基于 Webhook 的更新（通过 <code>tasks/pushNotification/set</code>）

关键端点：<code>tasks/send</code>（创建/更新）、<code>tasks/get</code>（轮询状态）、<code>tasks/sendSubscribe</code>（SSE 流）。

## 智能体框架对比

框架编排多智能体适用场景LangGraph显式状态图条件路由生产环境：持久化、HITL、精细控制OpenAI Agents SDK声明式交接基于交接简洁性：护栏、追踪、快速上手AutoGen (AG2)对话驱动GroupChat原型开发：代码执行、研究CrewAI基于角色的团队顺序/并行低代码：快速演示、简单流水线Google ADK会话 + 事件原生 A2A企业级：产物管理、多模态

## 智能体 RL 公式



## 智能体安全检查清单

威胁层次缓解措施Prompt 注入（直接）输入输入校验、指令层级、分隔符Prompt 注入（间接）工具输出将工具输出视为不可信；不要执行检索文档中的指令工具误用执行最小权限；<code>destructiveHint</code> 门控；沙箱化数据外泄输出输出过滤；将工具访问限制在允许的域名内过度自主架构最大迭代次数；成本预算；人工审批门控混淆代理多智能体验证任务来源；基于能力的访问控制

## 智能体评估指标

指标公式 / 定义目标任务成功率 (TSR)正确完成数 / 总任务数<math alttext="&gt;85\%" display="inline"><semantics><mrow><mphantom></mphantom><mo>&gt;</mo><mrow><mn>85</mn><mo>%</mo></mrow></mrow></semantics></math>（生产环境）完成所需步数每个成功任务的平均智能体动作数越低越高效单任务成本总 token 数 <math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> 每 token 价格取决于预算延迟 (TTFC)从请求到首个有用输出的时间交互场景 <math alttext="&lt;5" display="inline"><semantics><mrow><mphantom></mphantom><mo>&lt;</mo><mn>5</mn></mrow></semantics></math>s工具调用准确率正确的工具选择数 / 总调用数<math alttext="&gt;90\%" display="inline"><semantics><mrow><mphantom></mphantom><mo>&gt;</mo><mrow><mn>90</mn><mo>%</mo></mrow></mrow></semantics></math>恢复率成功重试数 / 初始失败数<math alttext="&gt;60\%" display="inline"><semantics><mrow><mphantom></mphantom><mo>&gt;</mo><mrow><mn>60</mn><mo>%</mo></mrow></mrow></semantics></math>人工升级率需要人工的任务数 / 总任务数<math alttext="&lt;15\%" display="inline"><semantics><mrow><mphantom></mphantom><mo>&lt;</mo><mrow><mn>15</mn><mo>%</mo></mrow></mrow></semantics></math>

## 关键智能体基准

基准领域指标SOTA (2025)SWE-bench Verified软件工程已解决问题占比<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>70%WebArena网页浏览任务成功率<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>40%OSWorld桌面计算机操作任务成功率<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>25%GAIA通用 AI 助手精确匹配准确率<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>75% (L1)Tau-bench工具使用可靠性通过率（5 次试验）<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>65%HumanEval / MBPP代码生成Pass@1<math alttext="&gt;95\%" display="inline"><semantics><mrow><mphantom></mphantom><mo>&gt;</mo><mrow><mn>95</mn><mo>%</mo></mrow></mrow></semantics></math>

<aside class="hh-next">
<p class="hh-next-label">下一篇</p>
<p class="hh-next-title">智能体 AI 漫游指南 · 第 30 章 总结与未来方向</p>
<p class="hh-next-hint">在「智能体 AI 漫游指南」分类下继续阅读。</p>
</aside>

</div>
