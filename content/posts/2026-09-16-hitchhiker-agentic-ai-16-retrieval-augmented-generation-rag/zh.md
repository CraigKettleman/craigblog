---
title: "第 16 章 检索增强生成（RAG）"
slug: "hitchhiker-agentic-ai-16-retrieval-augmented-generation-rag"
lang: "zh"
date: "2026-09-16T00:17:00.000Z"
draft: false
featured: false
categories:
  - "hitchhiker-agentic-ai"
cover: "./cover.png"
description: "检索增强生成（Retrieval-Augmented Generation, RAG）[209] 已成为在生产环境中部署大型语言模型时最具实际影响力的技术之一。…"
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
<p class="hh-part-title">第 V 部分 智能体 AI（Agentic AI）</p>
</aside>

检索增强生成（Retrieval-Augmented Generation, RAG）[209] 已成为在生产环境中部署大型语言模型时最具实际影响力的技术之一。RAG 不再仅依赖训练时编码进模型权重中的知识，而是为 LLM 配备一个动态、可更新的外部记忆——使其能够在广泛的知识密集型任务中给出准确、有据可查且可验证的响应。

## 动机与问题陈述

### 参数化知识 vs. 非参数化知识

我们可以将两种知识来源之间的区别形式化。令 <math alttext="\mathcal{M}_{\theta}" display="inline"><semantics><msub><mi>ℳ</mi><mi>θ</mi></msub></semantics></math> 表示参数为 <math alttext="\theta" display="inline"><semantics><mi>θ</mi></semantics></math> 的语言模型，<math alttext="\mathcal{D}=\{d_{1},d_{2},\ldots,d_{N}\}" display="inline"><semantics><mrow><mi>𝒟</mi><mo>=</mo><mrow><mo stretchy="false">{</mo><mrow><msub><mi>d</mi><mn>1</mn></msub><mo>,</mo><msub><mi>d</mi><mn>2</mn></msub><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msub><mi>d</mi><mi>N</mi></msub></mrow><mo stretchy="false">}</mo></mrow></mrow></semantics></math> 为外部文档语料库。在每种范式下，给定查询 <math alttext="q" display="inline"><semantics><mi>q</mi></semantics></math> 生成回答 <math alttext="a" display="inline"><semantics><mi>a</mi></semantics></math> 的概率为：



其中 <math alttext="P_{\text{ret}}(d\mid q,\mathcal{D})" display="inline"><semantics><mrow><msub><mi>P</mi><mtext>ret</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>d</mi><mo fence="true" lspace="0em" rspace="0em">∣</mo><mrow><mi>q</mi><mo>,</mo><mi>𝒟</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 是关于文档的检索分布。RAG 对检索到的证据进行边缘化，使生成过程接地于非参数化知识。

### 何时选择 RAG vs. 微调 vs. 长上下文

<div class="hh-table" id="ch16.t1"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>标准</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>RAG</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>微调</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>长上下文</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>RAG + 微调</span></span></span></th></tr></thead><tbody><tr><td class="ltx_align_left">知识频繁更新</td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>✓</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>✓</span></span></td></tr><tr><td class="ltx_align_left">需要引用 / 接地</td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>✓</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>✓</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>✓</span></span></td></tr><tr><td class="ltx_align_left">专有大型语料库</td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>✓</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>✓</span></span></td></tr><tr><td class="ltx_align_left">适配风格 / 格式</td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>✓</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>✓</span></span></td></tr><tr><td class="ltx_align_left">教授新推理技能</td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>✓</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>✓</span></span></td></tr><tr><td class="ltx_align_left">语料库可放入上下文窗口</td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>✓</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math></span></span></td></tr><tr><td class="ltx_align_left">要求低延迟</td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>✓</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math></span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">表 16.1：</span>决策指南：RAG vs. 微调 vs. 长上下文</p></div>

## 核心 RAG 架构

一个标准的 RAG 系统由两个阶段构成：处理和存储文档的离线索引流水线，以及为查询提供服务的在线检索—生成流水线。

### 完整流水线图

<figure id="ch16.f1"><img src="./fig_050_rag_arch.png" alt="图 16.1：端到端 RAG 架构。离线流水线（蓝色）对文档进行一次性索引；在线流水线（绿色/橙色）在推理时为每个查询提供服务。" loading="lazy" decoding="async"><figcaption><span class="hh-tag">图 16.1：</span>端到端 RAG 架构。离线流水线（蓝色）对文档进行一次性索引；在线流水线（绿色/橙色）在推理时为每个查询提供服务。</figcaption></figure>

### 索引流水线

文档以异构格式抵达（PDF、HTML、Markdown、DOCX、代码）。加载器提取干净的文本，并保留元数据（来源 URL、页码、章节标题、时间戳），这些元数据会与 Embedding 一并存储，用于过滤和引用。

长文档必须切分成可放入 Embedding 模型上下文窗口（通常为 512 个 Token）并保持语义连贯的块。分块策略是 RAG 系统设计中影响最大的决策之一（参见第 16.4 分块策略 节）。

每个块 <math alttext="c_{i}" display="inline"><semantics><msub><mi>c</mi><mi>i</mi></msub></semantics></math> 通过 Embedding 模型 <math alttext="f_{\phi}" display="inline"><semantics><msub><mi>f</mi><mi>ϕ</mi></msub></semantics></math> 编码为稠密向量 <math alttext="\mathbf{e}_{i}=f_{\phi}(c_{i})\in\mathbb{R}^{d}" display="inline"><semantics><mrow><msub><mi>𝐞</mi><mi>i</mi></msub><mo>=</mo><mrow><msub><mi>f</mi><mi>ϕ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>c</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>∈</mo><msup><mi>ℝ</mi><mi>d</mi></msup></mrow></semantics></math>。这些向量与原始文本和元数据一同存储到向量数据库中。

### 检索

给定查询 <math alttext="q" display="inline"><semantics><mi>q</mi></semantics></math>，检索步骤将其编码为 <math alttext="\mathbf{q}=f_{\phi}(q)" display="inline"><semantics><mrow><mi>𝐪</mi><mo>=</mo><mrow><msub><mi>f</mi><mi>ϕ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math>，并通过余弦相似度找出最相似的 <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 个块：

<div class="hh-equation" id="ch16.e3"><math alttext="\text{sim}(\mathbf{q},\mathbf{e}_{i})=\frac{\mathbf{q}\cdot\mathbf{e}_{i}}{\|\mathbf{q}\|\,\|\mathbf{e}_{i}\|}" display="block"><semantics><mrow><mrow><mtext>sim</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>𝐪</mi><mo>,</mo><msub><mi>𝐞</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mfrac><mrow><mi>𝐪</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><msub><mi>𝐞</mi><mi>i</mi></msub></mrow><mrow><mrow><mo stretchy="false">‖</mo><mi>𝐪</mi><mo stretchy="false">‖</mo></mrow><mo lspace="0.170em" rspace="0em">​</mo><mrow><mo stretchy="false">‖</mo><msub><mi>𝐞</mi><mi>i</mi></msub><mo stretchy="false">‖</mo></mrow></mrow></mfrac></mrow></semantics></math><span class="hh-equation-number">(16.3)</span></div>

返回 top-<math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 块 <math alttext="\mathcal{C}_{k}=\{c_{(1)},\ldots,c_{(k)}\}" display="inline"><semantics><mrow><msub><mi>𝒞</mi><mi>k</mi></msub><mo>=</mo><mrow><mo stretchy="false">{</mo><msub><mi>c</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msub><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msub><mi>c</mi><mrow><mo stretchy="false">(</mo><mi>k</mi><mo stretchy="false">)</mo></mrow></msub><mo stretchy="false">}</mo></mrow></mrow></semantics></math> 作为上下文。

### 生成

检索到的块被注入到一个 Prompt 模板中：

```
SYSTEM_PROMPT = """You are a helpful assistant. Answer the question using ONLY
the provided context. If the context does not contain enough information,
say so explicitly. Cite your sources using [Doc N] notation."""

def build_rag_prompt(query: str, chunks: list[dict]) -> str:
    context_str = "\n\n".join(
        f"[Doc {i+1}] (Source: {c['source']}, Page: {c.get('page','N/A')})\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    return f"""{SYSTEM_PROMPT}

Context:
{context_str}

Question: {query}

Answer:"""
```

<span class="hh-tag">代码清单 8：</span>标准 RAG Prompt 模板

## 检索方法

### 稀疏检索：BM25 与 TF-IDF

稀疏检索方法将文档和查询表示为词汇表上的高维稀疏向量。给定查询 <math alttext="q" display="inline"><semantics><mi>q</mi></semantics></math>（含词项 <math alttext="t_{1},\ldots,t_{n}" display="inline"><semantics><mrow><msub><mi>t</mi><mn>1</mn></msub><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msub><mi>t</mi><mi>n</mi></msub></mrow></semantics></math>）时，针对文档 <math alttext="d" display="inline"><semantics><mi>d</mi></semantics></math> 的经典 BM25 评分函数 [309] 为：

<div class="hh-equation" id="ch16.e4"><math alttext="\text{BM25}(d,q)=\sum_{i=1}^{n}\text{IDF}(t_{i})\cdot\frac{f(t_{i},d)\cdot(k_{1}+1)}{f(t_{i},d)+k_{1}\cdot\left(1-b+b\cdot\frac{|d|}{\text{avgdl}}\right)}" display="block"><semantics><mrow><mrow><mtext>BM25</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>d</mi><mo>,</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mo rspace="0.111em">=</mo><mrow><mrow><munderover><mo movablelimits="false">∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><mrow><mtext>IDF</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>t</mi><mi>i</mi></msub><mo rspace="0.055em" stretchy="false">)</mo></mrow></mrow></mrow><mo rspace="0.222em">⋅</mo><mfrac><mrow><mrow><mi>f</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>t</mi><mi>i</mi></msub><mo>,</mo><mi>d</mi><mo rspace="0.055em" stretchy="false">)</mo></mrow></mrow><mo rspace="0.222em">⋅</mo><mrow><mo stretchy="false">(</mo><mrow><msub><mi>k</mi><mn>1</mn></msub><mo>+</mo><mn>1</mn></mrow><mo stretchy="false">)</mo></mrow></mrow><mrow><mrow><mi>f</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>t</mi><mi>i</mi></msub><mo>,</mo><mi>d</mi><mo stretchy="false">)</mo></mrow></mrow><mo>+</mo><mrow><msub><mi>k</mi><mn>1</mn></msub><mo lspace="0.222em" rspace="0.222em">⋅</mo><mrow><mo>(</mo><mrow><mrow><mn>1</mn><mo>−</mo><mi>b</mi></mrow><mo>+</mo><mrow><mi>b</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><mfrac><mrow><mo stretchy="false">|</mo><mi>d</mi><mo stretchy="false">|</mo></mrow><mtext>avgdl</mtext></mfrac></mrow></mrow><mo>)</mo></mrow></mrow></mrow></mfrac></mrow></mrow></semantics></math><span class="hh-equation-number">(16.4)</span></div>

其中 <math alttext="f(t_{i},d)" display="inline"><semantics><mrow><mi>f</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>t</mi><mi>i</mi></msub><mo>,</mo><mi>d</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 为词频，<math alttext="|d|" display="inline"><semantics><mrow><mo stretchy="false">|</mo><mi>d</mi><mo stretchy="false">|</mo></mrow></semantics></math> 为文档长度，avgdl 为平均文档长度，<math alttext="k_{1}\in[1.2,2.0]" display="inline"><semantics><mrow><msub><mi>k</mi><mn>1</mn></msub><mo>∈</mo><mrow><mo stretchy="false">[</mo><mrow><mn>1.2</mn><mo>,</mo><mn>2.0</mn></mrow><mo stretchy="false">]</mo></mrow></mrow></semantics></math>、<math alttext="b=0.75" display="inline"><semantics><mrow><mi>b</mi><mo>=</mo><mn>0.75</mn></mrow></semantics></math> 为可调参数。

### 稠密检索：DPR

稠密段落检索（Dense Passage Retrieval, DPR）[180] 使用两个独立的 BERT 编码器——一个<em>查询编码器</em><math alttext="E_{Q}" display="inline"><semantics><msub><mi>E</mi><mi>Q</mi></msub></semantics></math> 和一个<em>段落编码器</em><math alttext="E_{P}" display="inline"><semantics><msub><mi>E</mi><mi>P</mi></msub></semantics></math>——通过对比损失训练，使相关的查询—段落对在 Embedding 空间中彼此靠近。

<div class="hh-equation" id="ch16.e5"><math alttext="\text{sim}(q,p)=E_{Q}(q)^{\top}E_{P}(p)" display="block"><semantics><mrow><mrow><mtext>sim</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>q</mi><mo>,</mo><mi>p</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><msub><mi>E</mi><mi>Q</mi></msub><mo lspace="0em" rspace="0em">​</mo><msup><mrow><mo stretchy="false">(</mo><mi>q</mi><mo stretchy="false">)</mo></mrow><mo>⊤</mo></msup><mo lspace="0em" rspace="0em">​</mo><msub><mi>E</mi><mi>P</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>p</mi><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(16.5)</span></div>

给定一批 <math alttext="B" display="inline"><semantics><mi>B</mi></semantics></math> 个查询—段落对 <math alttext="\{(q_{i},p_{i}^{+})\}_{i=1}^{B}" display="inline"><semantics><msubsup><mrow><mo stretchy="false">{</mo><mrow><mo stretchy="false">(</mo><msub><mi>q</mi><mi>i</mi></msub><mo>,</mo><msubsup><mi>p</mi><mi>i</mi><mo>+</mo></msubsup><mo stretchy="false">)</mo></mrow><mo stretchy="false">}</mo></mrow><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>B</mi></msubsup></semantics></math>，对比损失将批内所有其他段落都视为负样本：

<div class="hh-equation" id="ch16.e6"><math alttext="\mathcal{L}_{\text{DPR}}=-\frac{1}{B}\sum_{i=1}^{B}\log\frac{\exp\!\left(E_{Q}(q_{i})^{\top}E_{P}(p_{i}^{+})/\tau\right)}{\sum_{j=1}^{B}\exp\!\left(E_{Q}(q_{i})^{\top}E_{P}(p_{j})/\tau\right)}" display="block"><semantics><mrow><mi>ℒ</mi><msub><mrow></mrow><mtext>DPR</mtext></msub><mo rspace="0em">=</mo><mo lspace="0em">−</mo><mfrac><mn>1</mn><mi>B</mi></mfrac><mo movablelimits="false">∑</mo><msub><mrow></mrow><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow></msub><msup><mrow></mrow><mi>B</mi></msup><mi>log</mi><mfrac><mrow><mpadded width="1.370em"><mi>exp</mi></mpadded><mo>⁡</mo><mrow><mo>(</mo><mrow><mrow><msub><mi>E</mi><mi>Q</mi></msub><mo lspace="0em" rspace="0em">​</mo><msup><mrow><mo stretchy="false">(</mo><msub><mi>q</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow><mo>⊤</mo></msup><mo lspace="0em" rspace="0em">​</mo><msub><mi>E</mi><mi>P</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msubsup><mi>p</mi><mi>i</mi><mo>+</mo></msubsup><mo stretchy="false">)</mo></mrow></mrow><mo>/</mo><mi>τ</mi></mrow><mo>)</mo></mrow></mrow><mrow><msubsup><mo>∑</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mi>B</mi></msubsup><mrow><mpadded width="1.370em"><mi>exp</mi></mpadded><mo>⁡</mo><mrow><mo>(</mo><mrow><mrow><msub><mi>E</mi><mi>Q</mi></msub><mo lspace="0em" rspace="0em">​</mo><msup><mrow><mo stretchy="false">(</mo><msub><mi>q</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow><mo>⊤</mo></msup><mo lspace="0em" rspace="0em">​</mo><msub><mi>E</mi><mi>P</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>p</mi><mi>j</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>/</mo><mi>τ</mi></mrow><mo>)</mo></mrow></mrow></mrow></mfrac></mrow></semantics></math><span class="hh-equation-number">(16.6)</span></div>

其中 <math alttext="\tau" display="inline"><semantics><mi>τ</mi></semantics></math> 是 temperature 超参数。难负样本（字面相似但语义无关的段落）对于训练强检索器至关重要。

在大规模场景下，对数百万个 Embedding 做穷举搜索是不可行的。FAISS [171]（Facebook AI Similarity Search）提供高效的近似最近邻（ANN）搜索，具体方式包括：

<ul><li>IVF（倒排文件索引）：将向量聚类到 Voronoi 单元中；只搜索邻近的单元</li><li>HNSW（分层可导航小世界）[248]：基于图的索引，支持 <math alttext="O(\log N)" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mi>N</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 搜索</li><li>PQ（乘积量化）：压缩向量以减少内存占用</li></ul>

### 基于倒数排名融合的混合检索

混合检索结合稀疏分数与稠密分数。一种简单的线性组合为：

<div class="hh-equation" id="ch16.e7"><math alttext="s_{\text{hybrid}}(d,q)=\alpha\cdot s_{\text{dense}}(d,q)+(1-\alpha)\cdot s_{\text{sparse}}(d,q)" display="block"><semantics><mrow><mrow><msub><mi>s</mi><mtext>hybrid</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>d</mi><mo>,</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mrow><mrow><mi>α</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><msub><mi>s</mi><mtext>dense</mtext></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>d</mi><mo>,</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mo>+</mo><mrow><mrow><mrow><mo stretchy="false">(</mo><mrow><mn>1</mn><mo>−</mo><mi>α</mi></mrow><mo rspace="0.055em" stretchy="false">)</mo></mrow><mo rspace="0.222em">⋅</mo><msub><mi>s</mi><mtext>sparse</mtext></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>d</mi><mo>,</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(16.7)</span></div>

然而，来自不同系统的分数无法直接比较。互逆排名融合（Reciprocal Rank Fusion, RRF）[62] 通过基于排名而非分数来规避这一问题：

<div class="hh-equation" id="ch16.e8"><math alttext="\text{RRF}(d)=\sum_{r\in\mathcal{R}}\frac{1}{k+\text{rank}_{r}(d)}" display="block"><semantics><mrow><mrow><mtext>RRF</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>d</mi><mo stretchy="false">)</mo></mrow></mrow><mo rspace="0.111em">=</mo><mrow><munder><mo movablelimits="false">∑</mo><mrow><mi>r</mi><mo>∈</mo><mi>ℛ</mi></mrow></munder><mfrac><mn>1</mn><mrow><mi>k</mi><mo>+</mo><mrow><msub><mtext>rank</mtext><mi>r</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>d</mi><mo stretchy="false">)</mo></mrow></mrow></mrow></mfrac></mrow></mrow></semantics></math><span class="hh-equation-number">(16.8)</span></div>

其中 <math alttext="\mathcal{R}" display="inline"><semantics><mi>ℛ</mi></semantics></math> 是排序列表的集合（例如 BM25 排名与稠密排名），<math alttext="\text{rank}_{r}(d)" display="inline"><semantics><mrow><msub><mtext>rank</mtext><mi>r</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>d</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 是文档 <math alttext="d" display="inline"><semantics><mi>d</mi></semantics></math> 在列表 <math alttext="r" display="inline"><semantics><mi>r</mi></semantics></math> 中的排名，<math alttext="k=60" display="inline"><semantics><mrow><mi>k</mi><mo>=</mo><mn>60</mn></mrow></semantics></math> 是平滑常数，用于降低排名极高的文档所带来的影响。

### 学习式稀疏检索：SPLADE 与 SPLADEv2

SPLADE（Sparse Lexical and Expansion Model）[95] 使用预训练的掩码语言模型（MLM，例如 BERT/DistilBERT）为每个文档或查询生成一个覆盖<em>整个词汇表</em>的稀疏向量。关键洞见在于：MLM 头已经知道文本中每个位置与哪些词在语义上相关——SPLADE 把这一知识重新用作词项重要性权重。

给定输入文本 <math alttext="x=[x_{1},\ldots,x_{n}]" display="inline"><semantics><mrow><mi>x</mi><mo>=</mo><mrow><mo stretchy="false">[</mo><mrow><msub><mi>x</mi><mn>1</mn></msub><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msub><mi>x</mi><mi>n</mi></msub></mrow><mo stretchy="false">]</mo></mrow></mrow></semantics></math>：

<ol><li>经过 Transformer 编码器，通过 MLM 头获得上下文表示 <math alttext="\mathbf{H}\in\mathbb{R}^{n\times|\mathcal{V}|}" display="inline"><semantics><mrow><mi>𝐇</mi><mo>∈</mo><msup><mi>ℝ</mi><mrow><mi>n</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mrow><mo stretchy="false">|</mo><mi>𝒱</mi><mo stretchy="false">|</mo></mrow></mrow></msup></mrow></semantics></math></li><li>在所有位置上进行聚合，并施加饱和激活：</li></ol>

<div class="hh-equation" id="ch16.e9"><math alttext="w_{t}(x)=\log\!\left(1+\text{ReLU}\!\left(\max_{i\in[1,n]}\mathbf{H}_{i}[t]\right)\right)" display="block"><semantics><mrow><mrow><msub><mi>w</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mpadded width="1.146em"><mi>log</mi></mpadded><mo>⁡</mo><mrow><mo>(</mo><mrow><mn>1</mn><mo>+</mo><mrow><mpadded width="2.535em"><mtext>ReLU</mtext></mpadded><mo lspace="0em" rspace="0em">​</mo><mrow><mo>(</mo><mrow><mrow><munder><mi>max</mi><mrow><mi>i</mi><mo>∈</mo><mrow><mo stretchy="false">[</mo><mrow><mn>1</mn><mo>,</mo><mi>n</mi></mrow><mo stretchy="false">]</mo></mrow></mrow></munder><mo lspace="0.167em">⁡</mo><msub><mi>𝐇</mi><mi>i</mi></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">[</mo><mi>t</mi><mo stretchy="false">]</mo></mrow></mrow><mo>)</mo></mrow></mrow></mrow><mo>)</mo></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(16.9)</span></div>

其中 <math alttext="\mathbf{H}_{i}[t]" display="inline"><semantics><mrow><msub><mi>𝐇</mi><mi>i</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">[</mo><mi>t</mi><mo stretchy="false">]</mo></mrow></mrow></semantics></math> 是输入位置 <math alttext="i" display="inline"><semantics><mi>i</mi></semantics></math> 处词汇表 Token <math alttext="t" display="inline"><semantics><mi>t</mi></semantics></math> 的 MLM logit。

<ul><li><math alttext="\log(1+\cdot)" display="inline"><semantics><mrow><mi>log</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo rspace="0em">+</mo><mo lspace="0em" rspace="0em">⋅</mo><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 饱和可防止任何单个词项占据主导（类似 BM25 中的 TF 饱和）</li><li>ReLU 保证了稀疏性——大多数词汇表词项的权重为零</li><li>跨位置的 <math alttext="\max" display="inline"><semantics><mi>max</mi></semantics></math> 池化会为每个词项捕捉来自文本任意位置的最强信号</li><li>扩展：即使<em>未出现</em>在原始文本中的 Token 也能获得非零权重（例如，一篇关于「neural networks」的文档可能获得「deep learning」「AI」「backpropagation」的权重）</li></ul>

查询和文档分别映射为稀疏向量 <math alttext="\mathbf{w}^{q},\mathbf{w}^{d}\in\mathbb{R}^{|\mathcal{V}|}" display="inline"><semantics><mrow><msup><mi>𝐰</mi><mi>q</mi></msup><mo>,</mo><mrow><msup><mi>𝐰</mi><mi>d</mi></msup><mo>∈</mo><msup><mi>ℝ</mi><mrow><mo stretchy="false">|</mo><mi>𝒱</mi><mo stretchy="false">|</mo></mrow></msup></mrow></mrow></semantics></math>。相关性得分就是简单的点积：

<div class="hh-equation" id="ch16.e10"><math alttext="s(q,d)=\sum_{t\in\mathcal{V}}w_{t}^{q}\cdot w_{t}^{d}" display="block"><semantics><mrow><mrow><mi>s</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>q</mi><mo>,</mo><mi>d</mi><mo stretchy="false">)</mo></mrow></mrow><mo rspace="0.111em">=</mo><mrow><mrow><munder><mo movablelimits="false">∑</mo><mrow><mi>t</mi><mo>∈</mo><mi>𝒱</mi></mrow></munder><msubsup><mi>w</mi><mi>t</mi><mi>q</mi></msubsup></mrow><mo lspace="0.222em" rspace="0.222em">⋅</mo><msubsup><mi>w</mi><mi>t</mi><mi>d</mi></msubsup></mrow></mrow></semantics></math><span class="hh-equation-number">(16.10)</span></div>

由于两个向量都是稀疏的（在 30K 的词汇表中通常只有 20–200 个非零项），这可以用标准的倒排索引（Lucene、Anserini）高效计算——查询时无需 GPU。

SPLADE 使用对比学习（批内负样本 + 难负样本）训练，并额外加入两个正则项：

<div class="hh-equation" id="ch16.e11"><math alttext="\mathcal{L}=\mathcal{L}_{\text{contrastive}}+\lambda_{q}\|\mathbf{w}^{q}\|_{1}+\lambda_{d}\|\mathbf{w}^{d}\|_{1}" display="block"><semantics><mrow><mi>ℒ</mi><mo>=</mo><mrow><msub><mi>ℒ</mi><mtext>contrastive</mtext></msub><mo>+</mo><mrow><msub><mi>λ</mi><mi>q</mi></msub><mo lspace="0em" rspace="0em">​</mo><msub><mrow><mo stretchy="false">‖</mo><msup><mi>𝐰</mi><mi>q</mi></msup><mo stretchy="false">‖</mo></mrow><mn>1</mn></msub></mrow><mo>+</mo><mrow><msub><mi>λ</mi><mi>d</mi></msub><mo lspace="0em" rspace="0em">​</mo><msub><mrow><mo stretchy="false">‖</mo><msup><mi>𝐰</mi><mi>d</mi></msup><mo stretchy="false">‖</mo></mrow><mn>1</mn></msub></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(16.11)</span></div>

对查询和文档表示的 <math alttext="L_{1}" display="inline"><semantics><msub><mi>L</mi><mn>1</mn></msub></semantics></math> 惩罚促进稀疏性——没有它们，模型会学到稠密表示，从而违背初衷。

SPLADEv2 [94] 引入了若干改进，显著提升了效率与效果：

<ol><li>从交叉编码器蒸馏：SPLADEv2 不再仅使用二元相关性标签训练，而是采用交叉编码器教师（例如 MonoT5 [269]）提供软相关性分数。这带来了更丰富的训练信号：<math alttext="\mathcal{L}_{\text{distill}}=\text{KL}\!\left(\sigma(s_{\text{student}})\,\|\,\sigma(s_{\text{teacher}})\right)" display="block"><semantics><mrow><msub><mi>ℒ</mi><mtext>distill</mtext></msub><mo>=</mo><mpadded width="1.431em"><mtext>KL</mtext></mpadded><mrow><mo>(</mo><mi>σ</mi><mrow><mo stretchy="false">(</mo><msub><mi>s</mi><mtext>student</mtext></msub><mo stretchy="false">)</mo></mrow><mo lspace="0.170em" rspace="0.337em">∥</mo><mi>σ</mi><mrow><mo stretchy="false">(</mo><msub><mi>s</mi><mtext>teacher</mtext></msub><mo stretchy="false">)</mo></mrow><mo>)</mo></mrow></mrow></semantics></math><span class="hh-tag">(16.12)</span></li><li>查询与文档使用独立编码器：SPLADEv2 对查询和文档采用不同的稀疏度目标。查询被鼓励<em>更加稀疏</em>（查找更快），而文档可以稍稠密一些（离线预先计算）：<math alttext="\lambda_{q}&gt;\lambda_{d}\quad\text{(e.g., }\lambda_{q}=3\times 10^{-4},\;\lambda_{d}=1\times 10^{-4}\text{)}" display="block"><semantics><mrow><mrow><msub><mi>λ</mi><mi>q</mi></msub><mo>&gt;</mo><msub><mi>λ</mi><mi>d</mi></msub></mrow><mspace width="1em"></mspace><mrow><mrow><mtext>(e.g., </mtext><mo lspace="0em" rspace="0em">​</mo><msub><mi>λ</mi><mi>q</mi></msub></mrow><mo>=</mo><mrow><mn>3</mn><mo lspace="0.222em" rspace="0.222em">×</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>4</mn></mrow></msup></mrow></mrow><mo rspace="0.447em">,</mo><mrow><msub><mi>λ</mi><mi>d</mi></msub><mo>=</mo><mrow><mrow><mn>1</mn><mo lspace="0.222em" rspace="0.222em">×</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>4</mn></mrow></msup></mrow><mo lspace="0em" rspace="0em">​</mo><mtext>)</mtext></mrow></mrow></mrow></semantics></math><span class="hh-tag">(16.13)</span></li><li>FLOPs 正则化：SPLADEv2 不再使用简单的 <math alttext="L_{1}" display="inline"><semantics><msub><mi>L</mi><mn>1</mn></msub></semantics></math>，而是引入感知 FLOPs 的正则项，直接惩罚预期的检索开销：<math alttext="\mathcal{L}_{\text{FLOPS}}=\sum_{t\in\mathcal{V}}\left(\overline{a}_{t}^{q}\right)^{2}+\sum_{t\in\mathcal{V}}\left(\overline{a}_{t}^{d}\right)^{2}" display="block"><semantics><mrow><msub><mi>ℒ</mi><mtext>FLOPS</mtext></msub><mo rspace="0.111em">=</mo><mrow><mrow><munder><mo movablelimits="false" rspace="0em">∑</mo><mrow><mi>t</mi><mo>∈</mo><mi>𝒱</mi></mrow></munder><msup><mrow><mo>(</mo><msubsup><mover accent="true"><mi>a</mi><mo stretchy="true">¯</mo></mover><mi>t</mi><mi>q</mi></msubsup><mo>)</mo></mrow><mn>2</mn></msup></mrow><mo rspace="0.055em">+</mo><mrow><munder><mo movablelimits="false" rspace="0em">∑</mo><mrow><mi>t</mi><mo>∈</mo><mi>𝒱</mi></mrow></munder><msup><mrow><mo>(</mo><msubsup><mover accent="true"><mi>a</mi><mo stretchy="true">¯</mo></mover><mi>t</mi><mi>d</mi></msubsup><mo>)</mo></mrow><mn>2</mn></msup></mrow></mrow></mrow></semantics></math><span class="hh-tag">(16.14)</span>其中 <math alttext="\overline{a}_{t}" display="inline"><semantics><msub><mover accent="true"><mi>a</mi><mo stretchy="true">¯</mo></mover><mi>t</mi></msub></semantics></math> 是词项 <math alttext="t" display="inline"><semantics><mi>t</mi></semantics></math> 在整个 batch 上的平均激活值。这会惩罚在大量文档中都非零的词项（倒排列表越长 = 检索越慢）。</li><li>高效骨干网络：使用 DistilBERT（66M 参数）而非 BERT-base（110M 参数），在质量损失极小的前提下将编码时间减半。</li></ol>

### ColBERT：晚期交互

ColBERT [183] 将查询和文档编码为 Token 级 Embedding 的<em>集合</em>，并使用 <em>MaxSim</em> 算子进行打分：

<div class="hh-equation" id="ch16.e15"><math alttext="s(q,d)=\sum_{i\in|\mathbf{q}|}\max_{j\in|\mathbf{d}|}\mathbf{q}_{i}^{\top}\mathbf{d}_{j}" display="block"><semantics><mrow><mrow><mi>s</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>q</mi><mo>,</mo><mi>d</mi><mo stretchy="false">)</mo></mrow></mrow><mo rspace="0.111em">=</mo><mrow><munder><mo movablelimits="false">∑</mo><mrow><mi>i</mi><mo>∈</mo><mrow><mo stretchy="false">|</mo><mi>𝐪</mi><mo stretchy="false">|</mo></mrow></mrow></munder><mrow><mrow><munder><mi>max</mi><mrow><mi>j</mi><mo>∈</mo><mrow><mo stretchy="false">|</mo><mi>𝐝</mi><mo stretchy="false">|</mo></mrow></mrow></munder><mo lspace="0.167em">⁡</mo><msubsup><mi>𝐪</mi><mi>i</mi><mo>⊤</mo></msubsup></mrow><mo lspace="0em" rspace="0em">​</mo><msub><mi>𝐝</mi><mi>j</mi></msub></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(16.15)</span></div>

这种后期交互（late interaction）机制比单向量双编码器更具表现力，同时因为文档 Embedding 是离线预先计算的，它又远比交叉编码器快。

查询编码器 <math alttext="E_{Q}" display="inline"><semantics><msub><mi>E</mi><mi>Q</mi></msub></semantics></math> 和文档编码器 <math alttext="E_{D}" display="inline"><semantics><msub><mi>E</mi><mi>D</mi></msub></semantics></math> 都是基于 BERT 的模型，产生<em>逐 Token</em>的 Embedding（而不是单个 [CLS] 向量）。每个 Token 的 Embedding 通过一个线性层投影到更低的维度（通常为 128）：



ColBERT 使用针对正、负段落的成对 softmax 交叉熵损失进行训练。给定查询 <math alttext="q" display="inline"><semantics><mi>q</mi></semantics></math>、正段落 <math alttext="d^{+}" display="inline"><semantics><msup><mi>d</mi><mo>+</mo></msup></semantics></math> 和一组负段落 <math alttext="\{d^{-}_{1},\ldots,d^{-}_{N}\}" display="inline"><semantics><mrow><mo stretchy="false">{</mo><msubsup><mi>d</mi><mn>1</mn><mo>−</mo></msubsup><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msubsup><mi>d</mi><mi>N</mi><mo>−</mo></msubsup><mo stretchy="false">}</mo></mrow></semantics></math>：

<div class="hh-equation" id="ch16.e18"><math alttext="\mathcal{L}_{\text{ColBERT}}=-\log\frac{\exp(s(q,d^{+}))}{\exp(s(q,d^{+}))+\sum_{k=1}^{N}\exp(s(q,d^{-}_{k}))}" display="block"><semantics><mrow><msub><mi>ℒ</mi><mtext>ColBERT</mtext></msub><mo>=</mo><mrow><mo rspace="0.167em">−</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mfrac><mrow><mi>exp</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mi>s</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>q</mi><mo>,</mo><msup><mi>d</mi><mo>+</mo></msup><mo stretchy="false">)</mo></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow><mrow><mrow><mi>exp</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mi>s</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>q</mi><mo>,</mo><msup><mi>d</mi><mo>+</mo></msup><mo stretchy="false">)</mo></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow><mo rspace="0.055em">+</mo><mrow><msubsup><mo>∑</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mi>N</mi></msubsup><mrow><mi>exp</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mi>s</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>q</mi><mo>,</mo><msubsup><mi>d</mi><mi>k</mi><mo>−</mo></msubsup><mo stretchy="false">)</mo></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></mfrac></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(16.18)</span></div>

其中 <math alttext="s(q,d)" display="inline"><semantics><mrow><mi>s</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>q</mi><mo>,</mo><mi>d</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 是式 16.15 中的 MaxSim 分数。负样本来自：

<ul><li>批内负样本：同一训练 batch 中的其他段落（免费且数量充足）</li><li>难负样本：由 BM25 检索出的、字面相似但语义无关的段落（对质量影响最大）</li><li>蒸馏负样本（ColBERTv2 [313]）：使用交叉编码器教师挖掘最难的负样本，并将其分数蒸馏进 ColBERT</li></ul>

在索引阶段，所有文档 Token 的 Embedding 都会被预先计算并存储（ColBERTv2 中可选地通过残差量化进行压缩）。在查询阶段，只有查询 Token 需要实时编码，MaxSim 针对已存储的文档 Embedding 计算。这种分离带来了：

<ul><li>离线文档编码：编码一次，服务大量查询</li><li>PLAID 索引[313]：对文档 Embedding 聚类，用质心做初步候选检索，然后只对候选计算精确的 MaxSim——将延迟降低 5–10<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math></li><li>索引大小：每个文档 <math alttext="|d|\times 128" display="inline"><semantics><mrow><mrow><mo stretchy="false">|</mo><mi>d</mi><mo rspace="0.055em" stretchy="false">|</mo></mrow><mo rspace="0.222em">×</mo><mn>128</mn></mrow></semantics></math> 个浮点数（比单向量方法更大，但通过量化可压缩到每维 <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>2 字节）</li></ul>

### 检索方法对比

<div class="hh-table" id="ch16.t2"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>方法</span></th><th class="ltx_align_left"><span>延迟</span></th><th class="ltx_align_left"><span>准确率</span></th><th class="ltx_align_left"><span>索引大小</span></th><th class="ltx_align_left"><span>GPU</span></th><th class="ltx_align_left"><span>最适合</span></th></tr></thead><tbody><tr><td class="ltx_align_left"><span>TF-IDF <span>[336]</span></span></td><td class="ltx_align_left"><span>极低</span></td><td class="ltx_align_left"><span>低</span></td><td class="ltx_align_left"><span>小</span></td><td class="ltx_align_left"><span>否</span></td><td class="ltx_align_left"><span>基线、精确匹配</span></td></tr><tr><td class="ltx_align_left"><span>BM25 <span>[309]</span></span></td><td class="ltx_align_left"><span>极低</span></td><td class="ltx_align_left"><span>中</span></td><td class="ltx_align_left"><span>小</span></td><td class="ltx_align_left"><span>否</span></td><td class="ltx_align_left"><span>关键词搜索、罕见词项</span></td></tr><tr><td class="ltx_align_left"><span>DPR / 双编码器 <span>[180]</span></span></td><td class="ltx_align_left"><span>低</span></td><td class="ltx_align_left"><span>高</span></td><td class="ltx_align_left"><span>大</span></td><td class="ltx_align_left"><span>是</span></td><td class="ltx_align_left"><span>语义相似度</span></td></tr><tr><td class="ltx_align_left"><span>SPLADE <span>[95]</span></span></td><td class="ltx_align_left"><span>低</span></td><td class="ltx_align_left"><span>高</span></td><td class="ltx_align_left"><span>中</span></td><td class="ltx_align_left"><span>是</span></td><td class="ltx_align_left"><span>准确率与速度兼得</span></td></tr><tr><td class="ltx_align_left"><span>ColBERT <span>[183]</span></span></td><td class="ltx_align_left"><span>中</span></td><td class="ltx_align_left"><span>很高</span></td><td class="ltx_align_left"><span>很大</span></td><td class="ltx_align_left"><span>是</span></td><td class="ltx_align_left"><span>高准确率检索</span></td></tr><tr><td class="ltx_align_left"><span>交叉编码器 <span>[268]</span></span></td><td class="ltx_align_left"><span>高</span></td><td class="ltx_align_left"><span>最高</span></td><td class="ltx_align_left"><span>不适用</span></td><td class="ltx_align_left"><span>是</span></td><td class="ltx_align_left"><span>对 top-<math alttext="k" display="inline"><semantics><mi>k</mi><span></span></semantics></math> 重排</span></td></tr><tr><td class="ltx_align_left"><span>混合（RRF）<span>[62]</span></span></td><td class="ltx_align_left"><span>低</span></td><td class="ltx_align_left"><span>很高</span></td><td class="ltx_align_left"><span>大</span></td><td class="ltx_align_left"><span>是</span></td><td class="ltx_align_left"><span>生产系统</span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">表 16.2：</span>各检索方法在关键维度上的对比</p></div>

## 分块策略

分块是将文档切分为若干片段的过程，这些片段（1）足够小，可以放入 Embedding 模型的上下文窗口，（2）语义连贯，并且（3）在单独被检索出来时包含足够有用的上下文。

### 带重叠的固定大小分块

最简单的策略：每 <math alttext="W" display="inline"><semantics><mi>W</mi></semantics></math> 个 Token 切分一次，相邻块之间重叠 <math alttext="O" display="inline"><semantics><mi>O</mi></semantics></math> 个 Token。

```
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,       # tokens per chunk
    chunk_overlap=64,     # overlap to preserve context across boundaries
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(documents)
```

<span class="hh-tag">代码清单 9：</span>带重叠的固定大小分块

重叠公式：对于长度为 <math alttext="L" display="inline"><semantics><mi>L</mi></semantics></math> 个 Token 的文档，块的数量为：

<div class="hh-equation" id="ch16.e19"><math alttext="N_{\text{chunks}}=\left\lceil\frac{L-O}{W-O}\right\rceil" display="block"><semantics><mrow><msub><mi>N</mi><mtext>chunks</mtext></msub><mo>=</mo><mrow><mo>⌈</mo><mfrac><mrow><mi>L</mi><mo>−</mo><mi>O</mi></mrow><mrow><mi>W</mi><mo>−</mo><mi>O</mi></mrow></mfrac><mo>⌉</mo></mrow></mrow></semantics></math><span class="hh-equation-number">(16.19)</span></div>

### 语义分块

语义分块不在固定间隔处切分，而是在通过测量相邻句子之间的 Embedding 相似度检测出的<em>主题边界</em>处切分：

```
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

chunker = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",  # or "standard_deviation"
    breakpoint_threshold_amount=95,          # split at top 5% dissimilarity
)
chunks = chunker.split_documents(documents)
```

<span class="hh-tag">代码清单 10：</span>基于 Embedding 相似度的语义分块

### 基于文档结构的分块

对于结构化文档（Markdown、HTML、代码），在自然边界处切分：

<ul><li>Markdown：在 <code>##</code> 标题处切分，保留章节上下文</li><li>HTML：在 <code>⟨section⟩</code>、<code>⟨article⟩</code>、<code>⟨p⟩</code> 标签处切分</li><li>代码：在函数/类定义处切分，并在每个块中保留 import</li><li>表格：整张表作为单个块保留；绝不在行中间切分</li></ul>

### 父—子分块

一种将检索粒度与生成上下文解耦的强大模式：

<ol><li>索引小的子块（例如 128 个 Token）以进行精确检索</li><li>返回大的父块（例如 512 个 Token）给 LLM 以获得更丰富的上下文</li></ol>

```
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter  = RecursiveCharacterTextSplitter(chunk_size=400)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=InMemoryStore(),
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
retriever.add_documents(documents)
```

<span class="hh-tag">代码清单 11：</span>使用 LangChain 的父子分块

### 块大小的经验指南

<div class="hh-table" id="ch16.t3"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>用例</span></th><th class="ltx_align_left"><span>建议块大小</span></th><th class="ltx_align_left"><span>重叠</span></th></tr></thead><tbody><tr><td class="ltx_align_left">事实型问答（精确事实）</td><td class="ltx_align_left">128–256 个 Token</td><td class="ltx_align_left">20–32 个 Token</td></tr><tr><td class="ltx_align_left">摘要 / 综合</td><td class="ltx_align_left">512–1024 个 Token</td><td class="ltx_align_left">64–128 个 Token</td></tr><tr><td class="ltx_align_left">代码检索</td><td class="ltx_align_left">完整函数</td><td class="ltx_align_left">无</td></tr><tr><td class="ltx_align_left">法律 / 监管文档</td><td class="ltx_align_left">段落级</td><td class="ltx_align_left">1 个句子</td></tr><tr><td class="ltx_align_left">对话 / 聊天</td><td class="ltx_align_left">256–512 个 Token</td><td class="ltx_align_left">32–64 个 Token</td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">表 16.3：</span>按用例划分的块大小建议</p></div>

## 高级 RAG 模式

### 查询变换

原始用户查询往往含糊、过短，或者与文档用语匹配不佳。查询变换技术在搜索步骤之前改进检索。

与其直接对查询做 Embedding，不如生成一个<em>假设性答案</em>，再对它做 Embedding：

<div class="hh-equation" id="ch16.e20"><math alttext="\hat{d}=\text{LLM}(q),\quad\mathbf{e}_{\text{query}}=f_{\phi}(\hat{d})" display="block"><semantics><mrow><mrow><mover accent="true"><mi>d</mi><mo>^</mo></mover><mo>=</mo><mrow><mtext>LLM</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow></mrow><mo rspace="1.167em">,</mo><mrow><msub><mi>𝐞</mi><mtext>query</mtext></msub><mo>=</mo><mrow><msub><mi>f</mi><mi>ϕ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mover accent="true"><mi>d</mi><mo>^</mo></mover><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(16.20)</span></div>

其直觉是：假设性答案与真实文档处于相同的语言语域，从而缩小查询与文档之间的分布差距。

对于具体问题，先生成一个更通用的「后退」问题，对两者都做检索，再合并上下文。例如：「乙醇在 2 atm 下的沸点是多少？」<math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 后退：「哪些因素会影响液体的沸点？」

生成 <math alttext="M" display="inline"><semantics><mi>M</mi></semantics></math> 个多样化的查询改写，对每个都做检索，再取结果的并集：

```
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=ChatOpenAI(temperature=0.7),
    include_original=True,   # also retrieve for original query
)
# Internally generates 3 query variants, retrieves for each, deduplicates
docs = retriever.get_relevant_documents(query)
```

<span class="hh-tag">代码清单 12：</span>多查询检索

### 重排序

在初步检索出 top-<math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 候选之后，<em>交叉编码器</em>重排器会联合地为每个查询—文档对打分（同时关注两者），以更高的延迟为代价给出准确得多的相关性分数：

<div class="hh-equation" id="ch16.e21"><math alttext="s_{\text{cross}}(q,d)=\text{CrossEncoder}([q;d])" display="block"><semantics><mrow><mrow><msub><mi>s</mi><mtext>cross</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>q</mi><mo>,</mo><mi>d</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mtext>CrossEncoder</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mrow><mo stretchy="false">[</mo><mrow><mi>q</mi><mo>;</mo><mi>d</mi></mrow><mo stretchy="false">]</mo></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(16.21)</span></div>

交叉编码器无法用于第一阶段检索（没有预先计算的文档 Embedding），但非常适合对一小批候选（通常为 <math alttext="k=20" display="inline"><semantics><mrow><mi>k</mi><mo>=</mo><mn>20</mn></mrow></semantics></math>–<math alttext="100" display="inline"><semantics><mn>100</mn></semantics></math> 个）进行重排。

```
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-large")

def rerank(query: str, docs: list[str], top_n: int = 5) -> list[str]:
    pairs = [(query, doc) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), reverse=True)
    return [doc for _, doc in ranked[:top_n]]
```

<span class="hh-tag">代码清单 13：</span>使用 BGE 的交叉编码器重排

### 上下文压缩

检索到的块往往在相关段落周围夹杂着无关句子。上下文压缩使用 LLM 只抽取相关的部分：

```
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever()
)
compressed_docs = compression_retriever.get_relevant_documents(query)
```

<span class="hh-tag">代码清单 14：</span>基于 LLM 的上下文压缩

### Self-RAG

Self-RAG [13] 训练单个模型来（1）决定<em>是否</em>要检索，（2）在有或没有检索的情况下生成，（3）使用特殊的反思 Token <em>批判</em>自己的输出：

<ul><li><code>[Retrieve]</code>：模型是否应该检索更多段落？</li><li><code>[IsRel]</code>：检索到的段落是否与查询相关？</li><li><code>[IsSup]</code>：生成的陈述是否由检索到的段落推出？</li><li><code>[IsUse]</code>：整体回答是否有用？</li></ul>

模型经过端到端训练，在生成回答的同时预测这些 Token，从而实现对检索与自评的细粒度控制。

### CRAG：纠正式 RAG

CRAG [401] 增加了一个<em>检索评估器</em>，对检索到的文档进行评级并触发纠正动作：

<ol><li>检索 top-<math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 文档</li><li>为每个文档评级：正确 / 模糊 / 错误</li><li>如果所有文档都不正确或模糊 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 回退到网页搜索</li><li>如果部分文档正确 <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 使用知识精炼（剔除无关句子）</li><li>基于精炼后的上下文生成回答</li></ol>

### 自适应 RAG

自适应 RAG [162] 根据预测的复杂度将查询路由到不同的检索策略：

<ul><li>不检索：模型仅凭参数即可回答的简单事实性查询</li><li>单步 RAG：针对中等难度查询的标准「先检索后生成」</li><li>多步 RAG：针对复杂多跳问题的迭代检索</li></ul>

一个在查询复杂度标签上训练的轻量级分类器会为每个到来的查询选择路由。

### Graph RAG

微软的 Graph RAG [82] 从文档语料库构建<em>知识图谱</em>，并使用社区检测生成层级式摘要：

<ol><li>实体抽取：LLM 从每个块中抽取实体与关系</li><li>图构建：构建一个图 <math alttext="G=(V,E)" display="inline"><semantics><mrow><mi>G</mi><mo>=</mo><mrow><mo stretchy="false">(</mo><mi>V</mi><mo>,</mo><mi>E</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math>，其中节点是实体、边是关系</li><li>社区检测：应用 Leiden 算法在多个分辨率上发现社区</li><li>社区摘要：LLM 为每个社区生成一段摘要</li><li>查询：对于全局查询，在社区摘要上做 map-reduce；对于局部查询，使用标准向量搜索</li></ol>

### RAG-Fusion

RAG-Fusion [300] 从原始查询生成多个搜索查询，对每个都做检索，并使用 RRF（式 16.8）融合各个排序列表：

```
def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    """Fuse multiple ranked document lists using RRF."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)

def rag_fusion(query: str, retriever, llm, n_queries: int = 4) -> str:
    # Step 1: Generate query variants
    variants = generate_query_variants(query, llm, n=n_queries)
    # Step 2: Retrieve for each variant
    all_ranked = [retriever.retrieve(q) for q in [query] + variants]
    # Step 3: Fuse with RRF
    fused_docs = reciprocal_rank_fusion(all_ranked)
    # Step 4: Generate answer
    return generate_answer(query, fused_docs[:5], llm)
```

<span class="hh-tag">代码清单 15：</span>使用 RRF 的 RAG-Fusion

## 高效 RAG 解码：REFRAG

RAG 的一个实际瓶颈是<em>解码延迟</em>：拼接进 LLM 上下文的检索段落往往很长却只有稀疏的相关性，抬高了首 Token 时间（TTFT）和 KV cache 内存占用。REFRAG [221] 观察到，由于检索段落各自来源独立（重排时经过多样性筛选或去重），它们的 attention 模式呈<em>块对角</em>——大多数跨段落的 attention 接近零。这种稀疏性意味着解码过程中对 RAG 上下文的大部分计算都是不必要的。

REFRAG 通过一个三阶段解码策略来利用这一结构：

<ol><li>压缩：用紧凑的摘要（例如每个段落块的均值池化 key/value）替换检索段落的完整 KV 表示，大幅降低内存占用。</li><li>感知：在每个解码步，对压缩后的表示施加轻量 attention，以确定哪些段落块与当前 Token 相关。</li><li>扩展：只为被选中的块重建完整的 KV 条目，在稀疏的活跃集合上执行精确 attention。</li></ol>

在基于 LLaMA 的模型上，REFRAG 实现了最高 <math alttext="30.85\times" display="inline"><semantics><mrow><mn>30.85</mn><mo lspace="0.222em">×</mo></mrow></semantics></math> 的 TTFT 加速（相比此前的稀疏 attention 基线提升 <math alttext="3.75\times" display="inline"><semantics><mrow><mn>3.75</mn><mo lspace="0.222em">×</mo></mrow></semantics></math>），且困惑度没有任何损失。在固定内存预算下，它还把有效上下文长度扩展了 <math alttext="16\times" display="inline"><semantics><mrow><mn>16</mn><mo lspace="0.222em">×</mo></mrow></semantics></math>。这些收益在 RAG、多轮对话和长文档摘要任务中均成立。

## 智能体 RAG

### 动机：静态 RAG 的局限

标准 RAG 遵循固定的「先检索后生成」模式。它在以下情况会失效：

<ul><li>多跳问题：「2023 年收购了 OpenAI 主要竞争对手的那家公司是谁创立的？」需要串联多次检索</li><li>含糊查询：正确的检索策略取决于检索到了什么</li><li>异构来源：不同的子问题需要不同的知识库</li><li>迭代式精炼：初步检索可能揭示出需要换一个查询</li></ul>

### 智能体 RAG 架构

<figure id="ch16.f2"><img src="./fig_051_agentic_rag.png" alt="图 16.2：智能体 RAG 的控制流。智能体迭代地进行规划、检索、评估充分性，并在返回答案前自检接地性。" loading="lazy" decoding="async"><figcaption><span class="hh-tag">图 16.2：</span>智能体 RAG 的控制流。智能体迭代地进行规划、检索、评估充分性，并在返回答案前自检接地性。</figcaption></figure>

### 多源路由

智能体 RAG 系统可以把子查询路由到专门的知识源。核心洞见在于：不同的问题类型需要不同的检索后端——没有任何单一索引能在所有方面都表现出色。

设想一个为金融分析师服务的助手处理四个查询：

<ul><li>“我们公司的带薪休假（PTO）政策是什么？” <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math>向量数据库（内部文档）</li><li>“美联储昨天宣布了什么？” <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math>网页搜索（实时）</li><li>“按地区展示 Q3 营收” <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math>SQL 数据库（结构化数据）</li><li>“我们的鉴权中间件如何校验 Token？” <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math>代码索引（代码库）</li></ul>

“从单一索引扁平化检索”的做法要么错过答案，要么返回无关段落。路由在检索开始前，为合适的子问题选择<em>合适的工具</em>。

三种主要方法，按复杂度递增：

<ol><li>基于规则的路由。通过关键词触发（例如，SQL 关键字 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 数据库，URL 模式 <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 网页）。快速且可解释，但对歧义查询非常脆弱。</li><li>基于分类器的路由。由一个轻量模型（如微调过的 BERT 分类器，或在查询 Embedding 上的逻辑回归）预测最佳来源。延迟低（<math alttext="&lt;" display="inline"><semantics><mo>&lt;</mo></semantics></math>10 毫秒），可基于路由日志训练，但需要带标签数据。</li><li>基于 LLM 的路由。LLM 本身在一次结构化输出调用中决定来源（见下面的代码清单）。最为灵活——可以处理新颖的查询类型并解释其推理——但会增加一次 LLM 调用的延迟。</li></ol>

<ul><li>回退链：若主要来源返回的结果置信度较低，则尝试次优来源。</li><li>并行扇出：对于歧义查询，同时从多个来源检索，并通过互逆排名融合（表 表 16.2）合并结果。</li><li>成本感知：网页搜索和 API 调用可能存在金钱开销或速率限制；路由器应将其纳入考量。</li><li>可观测性：记录每个路由决策及其推理过程——对调试和再训练至关重要。</li></ul>

```
from enum import Enum
from pydantic import BaseModel

class KnowledgeSource(str, Enum):
    VECTOR_DB   = "vector_db"    # internal documents
    WEB_SEARCH  = "web_search"   # real-time web
    SQL_DB      = "sql_db"       # structured data
    CODE_INDEX  = "code_index"   # codebase
    API         = "api"          # external APIs

class RouteDecision(BaseModel):
    source: KnowledgeSource
    refined_query: str
    reasoning: str

def route_query(query: str, llm) -> RouteDecision:
    """Use LLM to decide which knowledge source to query."""
    prompt = f"""Given the query: "{query}"

Decide which knowledge source to use:
- vector_db: for internal documents, policies, past reports
- web_search: for current events, recent information
- sql_db: for numerical data, statistics, structured records
- code_index: for code examples, API documentation
- api: for real-time data (weather, stock prices, etc.)

Return a JSON with: source, refined_query, reasoning."""

    return llm.with_structured_output(RouteDecision).invoke(prompt)
```

<span class="hh-tag">代码清单 16：</span>多源智能体 RAG 路由器

### 完整的智能体 RAG 实现

前面几节介绍了各个独立组件——路由、检索、评估。一个完整的智能体 RAG 系统将它们编排为一个<em>有状态节点的图</em>，其中控制流取决于中间结果。下面的实现使用 LangGraph 将四个节点串成一个循环：

<ol><li>计划（Plan）：将用户查询分解为子查询（每个信息需求对应一个）。</li><li>检索（Retrieve）：将每个子查询路由到合适的来源并取回文档。</li><li>评估（Evaluate）：判断累积的上下文是否足以回答原始查询。</li><li>生成（Generate）：综合检索到的文档生成带引用的最终答案。</li></ol>

关键设计模式是<em>条件循环</em>：评估之后，智能体要么进入生成阶段（如果上下文足够，或迭代预算耗尽），要么带着精炼后的子查询回到检索阶段。这与一个在信息收集动作上运行的 RL 智能体所遵循的“感知—行动—评估”循环如出一辙。

```
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import operator

class AgentState(TypedDict):
    query: str
    sub_queries: list[str]
    retrieved_docs: Annotated[list[dict], operator.add]
    context_sufficient: bool
    answer: str
    iterations: int
    max_iterations: int

def plan_node(state: AgentState) -> AgentState:
    """Decompose query into sub-queries."""
    sub_queries = decompose_query(state["query"])
    return {**state, "sub_queries": sub_queries, "iterations": 0}

def retrieve_node(state: AgentState) -> AgentState:
    """Retrieve documents for current sub-queries."""
    new_docs = []
    for sq in state["sub_queries"]:
        source = route_query(sq)
        docs = retrieve_from_source(sq, source)
        new_docs.extend(docs)
    return {**state, "retrieved_docs": new_docs,
            "iterations": state["iterations"] + 1}

def evaluate_node(state: AgentState) -> AgentState:
    """Evaluate whether retrieved context is sufficient."""
    sufficient = evaluate_context_sufficiency(
        query=state["query"],
        docs=state["retrieved_docs"]
    )
    return {**state, "context_sufficient": sufficient}

def generate_node(state: AgentState) -> AgentState:
    """Generate answer from retrieved context."""
    answer = generate_with_citations(
        query=state["query"],
        docs=state["retrieved_docs"]
    )
    return {**state, "answer": answer}

def should_retrieve(state: AgentState) -> str:
    if state["context_sufficient"]:
        return "generate"
    if state["iterations"] >= state["max_iterations"]:
        return "generate"  # give up and generate with what we have
    return "retrieve"

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("plan",     plan_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("evaluate", evaluate_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("plan")
workflow.add_edge("plan",     "retrieve")
workflow.add_edge("retrieve", "evaluate")
workflow.add_conditional_edges("evaluate", should_retrieve,
    {"retrieve": "retrieve", "generate": "generate"})
workflow.add_edge("generate", END)

agent = workflow.compile()

# Run
result = agent.invoke({
    "query": "What were the main causes of the 2023 banking crisis?",
    "max_iterations": 3,
    "retrieved_docs": [],
    "iterations": 0,
})
```

<span class="hh-tag">代码清单 17：</span>基于 LangGraph 的智能体 RAG

### 工具增强的 RAG

智能体 RAG 可以将检索与计算工具结合起来：

```
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool

@tool
def search_documents(query: str) -> str:
    """Search internal document knowledge base."""
    docs = vectorstore.similarity_search(query, k=5)
    return "\n\n".join(d.page_content for d in docs)

@tool
def query_database(sql: str) -> str:
    """Execute SQL query on the analytics database."""
    return db.run(sql)

@tool
def web_search(query: str) -> str:
    """Search the web for current information."""
    return tavily_client.search(query)

@tool
def execute_python(code: str) -> str:
    """Execute Python code for calculations."""
    return python_repl.run(code)

tools = [search_documents, query_database, web_search, execute_python]
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

<span class="hh-tag">代码清单 18：</span>结合 SQL 与检索的工具增强 RAG

### Search-R1：RL 训练的智能体 RAG

上述智能体 RAG 方法依赖 <em>Prompt 工程</em>式编排——智能体的搜索行为由指令控制，而不是通过训练习得。Search-R1[170] 采取了根本不同的思路：它通过强化学习训练 LLM，使其把<em>学会何时搜索、搜索什么、搜索多少次</em>作为推理过程的一部分。

Search-R1 扩展了 DeepSeek-R1 [69] 推理框架，把搜索引擎查询视为 RL 训练循环中的动作。在思维链生成过程中，模型可以发出特殊 token <code>&lt;search&gt;query&lt;/search&gt;</code>，触发从搜索引擎的实时检索。检索结果被注回推理上下文，模型继续生成。

模型生成与搜索动作交织的推理轨迹：

<div class="hh-equation" id="ch16.ex2"><math alttext="\underbrace{\text{think}_{1}}_{\text{reasoning}}\to\underbrace{\texttt{\textlangle search\textrangle}q_{1}\texttt{\textlangle/search\textrangle}}_{\text{action}}\to\underbrace{[\text{results}_{1}]}_{\text{observation}}\to\text{think}_{2}\to\texttt{\textlangle search\textrangle}q_{2}\texttt{\textlangle/search\textrangle}\to\cdots\to\text{answer}" display="block"><semantics><mrow><munder><munder accentunder="true"><msub><mtext>think</mtext><mn>1</mn></msub><mo stretchy="true">⏟</mo></munder><mtext>reasoning</mtext></munder><mo stretchy="false">→</mo><munder><munder accentunder="true"><mrow><mtext>⟨search⟩</mtext><mo lspace="0em" rspace="0em">​</mo><msub><mi>q</mi><mn>1</mn></msub><mo lspace="0em" rspace="0em">​</mo><mtext>⟨/search⟩</mtext></mrow><mo stretchy="true">⏟</mo></munder><mtext>action</mtext></munder><mo stretchy="false">→</mo><munder><munder accentunder="true"><mrow><mo stretchy="false">[</mo><msub><mtext>results</mtext><mn>1</mn></msub><mo stretchy="false">]</mo></mrow><mo stretchy="true">⏟</mo></munder><mtext>observation</mtext></munder><mo stretchy="false">→</mo><msub><mtext>think</mtext><mn>2</mn></msub><mo stretchy="false">→</mo><mrow><mtext>⟨search⟩</mtext><mo lspace="0em" rspace="0em">​</mo><msub><mi>q</mi><mn>2</mn></msub><mo lspace="0em" rspace="0em">​</mo><mtext>⟨/search⟩</mtext></mrow><mo rspace="0.1389em" stretchy="false">→</mo><mo lspace="0.1389em" rspace="0.1389em">⋯</mo><mo lspace="0.1389em" stretchy="false">→</mo><mtext>answer</mtext></mrow></semantics></math></div>

整条轨迹（推理 + 搜索 + 最终答案）由终局奖励打分：即最终答案相对于标准答案标签（ground-truth label）的正确性。

Search-R1 使用 GRPO（组相对策略优化）：

<ol><li>为每个问题采样 <math alttext="N" display="inline"><semantics><mi>N</mi></semantics></math> 条轨迹，每条轨迹可能包含 0–5 次搜索调用</li><li>实时执行搜索——环境返回真实的搜索引擎结果</li><li>对终局答案的正确性打分（与标准答案比对，使用精确匹配或 F1）</li><li>计算组相对优势：<math alttext="\hat{A}_{i}=(R_{i}-\mu_{G})/\sigma_{G}" display="inline"><semantics><mrow><msub><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>i</mi></msub><mo>=</mo><mrow><mrow><mo stretchy="false">(</mo><mrow><msub><mi>R</mi><mi>i</mi></msub><mo>−</mo><msub><mi>μ</mi><mi>G</mi></msub></mrow><mo stretchy="false">)</mo></mrow><mo>/</mo><msub><mi>σ</mi><mi>G</mi></msub></mrow></mrow></semantics></math></li><li>更新策略：使用 GRPO 的裁剪目标——强化搜索有效的轨迹</li></ol>

模型学会：

<ul><li>在不确定时搜索——避免为已掌握的知识进行不必要的搜索</li><li>构造有效的查询——学会能返回相关结果的查询措辞</li><li>多次搜索——基于初始结果迭代精炼查询</li><li>整合检索到的上下文——用搜索结果支撑或修正自己的推理</li></ul>

<div class="hh-table" id="ch16.t4"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>维度</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>基于 Prompt 的智能体 RAG</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Search-R1</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">搜索决策</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Prompt/启发式规则</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>通过 RL 学习得到</span></span></td></tr><tr><th class="ltx_align_left">查询构造</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>由 Prompt 引导（“改写查询”）</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>端到端训练</span></span></td></tr><tr><th class="ltx_align_left">搜索次数</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>固定，或在推理时由 LLM 决定</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>学习得到的最优次数</span></span></td></tr><tr><th class="ltx_align_left">训练信号</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>无（模型冻结）</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>正确性奖励</span></span></td></tr><tr><th class="ltx_align_left">搜索集成方式</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>追加到上下文</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>交织在思维链中</span></span></td></tr><tr><th class="ltx_align_left">失败恢复</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>重试启发式规则</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>学习得到的退避/重构策略</span></span></td></tr><tr><th class="ltx_align_left">推理时的额外开销</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>框架开销（LangGraph）</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>模型原生行为</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">表 16.4：</span>Search-R1（RL 训练）与基于 Prompt 的智能体 RAG 对比。</p></div>

在开放域问答基准（NQ [195]、TriviaQA [173]、HotpotQA [406]）上，使用 7B 模型的 Search-R1 优于：

<ul><li>标准 RAG（单次检索），准确率高出 15–20%</li><li>Prompt 引导的智能体 RAG（ReAct 风格），准确率高出 8–12%</li><li>接近使用标准 RAG 的更大模型（70B）的性能</li></ul>

关键洞见：学会何时搜索以及如何搜索，比拥有一个知道更多的更大模型更有价值。一个善于搜索的小模型胜过一个不搜索的大模型。

## 评测

评估一个 RAG 系统比孤立地评估检索或生成更难，因为错误可能源自流水线的<em>任何阶段</em>——而且它们会叠加。完美的生成器无法弥补无关的检索结果；而如果生成器产生幻觉或忽略上下文，再完美的检索器也是白费。

因此，有效的 RAG 评估在三个层面上展开：

<ol><li>检索质量：检索器是否找出了正确的段落？（召回率、精确率、MRR、NDCG）</li><li>生成质量：答案是否正确、是否忠实于检索到的上下文、是否完整？（正确性、忠实性、答案相关性）</li><li>端到端质量：整个系统是否令用户满意？（人类偏好、任务成功率、经延迟调整的效用）</li></ol>

一种常见的失败模式是只优化一个层面——例如，用较大的 <math alttext="K" display="inline"><semantics><mi>K</mi></semantics></math> 来最大化 Recall@<math alttext="K" display="inline"><semantics><mi>K</mi></semantics></math>，会让上下文塞满边缘相关的段落，反而<em>降低</em>生成质量。下面的指标同时覆盖检索与生成，使实践者能够诊断哪一阶段是瓶颈。

### 检索指标

令 <math alttext="\mathcal{R}_{k}" display="inline"><semantics><msub><mi>ℛ</mi><mi>k</mi></msub></semantics></math> 为排名前 <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 的检索文档集合，<math alttext="\mathcal{R}^{*}" display="inline"><semantics><msup><mi>ℛ</mi><mo>∗</mo></msup></semantics></math> 为相关文档集合。

<div class="hh-equation" id="ch16.e22"><math alttext="\text{Recall@}K=\frac{|\mathcal{R}_{K}\cap\mathcal{R}^{*}|}{|\mathcal{R}^{*}|}" display="block"><semantics><mrow><mrow><mtext>Recall@</mtext><mo lspace="0em" rspace="0em">​</mo><mi>K</mi></mrow><mo>=</mo><mfrac><mrow><mo stretchy="false">|</mo><mrow><msub><mi>ℛ</mi><mi>K</mi></msub><mo>∩</mo><msup><mi>ℛ</mi><mo>∗</mo></msup></mrow><mo stretchy="false">|</mo></mrow><mrow><mo stretchy="false">|</mo><msup><mi>ℛ</mi><mo>∗</mo></msup><mo stretchy="false">|</mo></mrow></mfrac></mrow></semantics></math><span class="hh-equation-number">(16.22)</span></div>

<div class="hh-equation" id="ch16.e23"><math alttext="\text{Precision@}K=\frac{|\mathcal{R}_{K}\cap\mathcal{R}^{*}|}{K}" display="block"><semantics><mrow><mrow><mtext>Precision@</mtext><mo lspace="0em" rspace="0em">​</mo><mi>K</mi></mrow><mo>=</mo><mfrac><mrow><mo stretchy="false">|</mo><mrow><msub><mi>ℛ</mi><mi>K</mi></msub><mo>∩</mo><msup><mi>ℛ</mi><mo>∗</mo></msup></mrow><mo stretchy="false">|</mo></mrow><mi>K</mi></mfrac></mrow></semantics></math><span class="hh-equation-number">(16.23)</span></div>

<div class="hh-equation" id="ch16.e24"><math alttext="\text{MRR}=\frac{1}{|Q|}\sum_{i=1}^{|Q|}\frac{1}{\text{rank}_{i}}" display="block"><semantics><mrow><mtext>MRR</mtext><mo>=</mo><mrow><mfrac><mn>1</mn><mrow><mo stretchy="false">|</mo><mi>Q</mi><mo stretchy="false">|</mo></mrow></mfrac><mo lspace="0em" rspace="0em">​</mo><mrow><munderover><mo movablelimits="false">∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mrow><mo stretchy="false">|</mo><mi>Q</mi><mo stretchy="false">|</mo></mrow></munderover><mfrac><mn>1</mn><msub><mtext>rank</mtext><mi>i</mi></msub></mfrac></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(16.24)</span></div>

其中 <math alttext="\text{rank}_{i}" display="inline"><semantics><msub><mtext>rank</mtext><mi>i</mi></msub></semantics></math> 是查询 <math alttext="i" display="inline"><semantics><mi>i</mi></semantics></math> 的第一个相关文档的排名。

<div class="hh-equation" id="ch16.e25"><math alttext="\text{NDCG@}K=\frac{\text{DCG@}K}{\text{IDCG@}K},\quad\text{DCG@}K=\sum_{i=1}^{K}\frac{\text{rel}_{i}}{\log_{2}(i+1)}" display="block"><semantics><mrow><mrow><mrow><mtext>NDCG@</mtext><mo lspace="0em" rspace="0em">​</mo><mi>K</mi></mrow><mo>=</mo><mfrac><mrow><mtext>DCG@</mtext><mo lspace="0em" rspace="0em">​</mo><mi>K</mi></mrow><mrow><mtext>IDCG@</mtext><mo lspace="0em" rspace="0em">​</mo><mi>K</mi></mrow></mfrac></mrow><mo rspace="1.167em">,</mo><mrow><mrow><mtext>DCG@</mtext><mo lspace="0em" rspace="0em">​</mo><mi>K</mi></mrow><mo rspace="0.111em">=</mo><mrow><munderover><mo movablelimits="false">∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>K</mi></munderover><mfrac><msub><mtext>rel</mtext><mi>i</mi></msub><mrow><msub><mi>log</mi><mn>2</mn></msub><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mi>i</mi><mo>+</mo><mn>1</mn></mrow><mo stretchy="false">)</mo></mrow></mrow></mfrac></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(16.25)</span></div>

其中 <math alttext="\text{rel}_{i}\in\{0,1,2,\ldots\}" display="inline"><semantics><mrow><msub><mtext>rel</mtext><mi>i</mi></msub><mo>∈</mo><mrow><mo stretchy="false">{</mo><mrow><mn>0</mn><mo>,</mo><mn>1</mn><mo>,</mo><mn>2</mn><mo>,</mo><mi mathvariant="normal">…</mi></mrow><mo stretchy="false">}</mo></mrow></mrow></semantics></math> 是第 <math alttext="i" display="inline"><semantics><mi>i</mi></semantics></math> 个结果的分级相关性，IDCG 是理想的（完美的）DCG。

### 生成指标

衡量生成的答案是否<em>有据可依地扎根</em>于检索到的上下文——即答案中的每一项断言都能归因到某篇检索文档。由 LLM 评判器评估：

<div class="hh-equation" id="ch16.e26"><math alttext="\text{Faithfulness}=\frac{\text{\# claims supported by context}}{\text{\# total claims in answer}}" display="block"><semantics><mrow><mtext>Faithfulness</mtext><mo>=</mo><mfrac><mtext># claims supported by context</mtext><mtext># total claims in answer</mtext></mfrac></mrow></semantics></math><span class="hh-equation-number">(16.26)</span></div>

衡量答案是否回应了问题。做法是由答案反向生成问题，并度量其与原始查询的相似度：

<div class="hh-equation" id="ch16.e27"><math alttext="\text{AnswerRelevance}=\frac{1}{N}\sum_{i=1}^{N}\cos\!\left(E(q),E(\hat{q}_{i})\right)" display="block"><semantics><mrow><mtext>AnswerRelevance</mtext><mo>=</mo><mrow><mfrac><mn>1</mn><mi>N</mi></mfrac><mo lspace="0em" rspace="0em">​</mo><mrow><munderover><mo movablelimits="false">∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>N</mi></munderover><mrow><mpadded width="1.216em"><mi>cos</mi></mpadded><mo>⁡</mo><mrow><mo>(</mo><mrow><mi>E</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mo>,</mo><mrow><mi>E</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mover accent="true"><mi>q</mi><mo>^</mo></mover><mi>i</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>)</mo></mrow></mrow></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(16.27)</span></div>

其中 <math alttext="\hat{q}_{i}" display="inline"><semantics><msub><mover accent="true"><mi>q</mi><mo>^</mo></mover><mi>i</mi></msub></semantics></math> 是由答案生成的问题。



### RAGAs 框架

RAGAs（Retrieval Augmented Generation Assessment）[85] 提供了使用 LLM 评判器的无参考评估框架：

```
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)
from datasets import Dataset

eval_dataset = Dataset.from_dict({
    "question":  questions,
    "answer":    generated_answers,
    "contexts":  retrieved_contexts,   # list of lists
    "ground_truth": reference_answers,
})

results = evaluate(
    dataset=eval_dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    ],
)
print(results.to_pandas())
```

<span class="hh-tag">代码清单 19：</span>RAGAs 评估（v0.1 API；v0.2+ 使用 <code>user_input</code>、<code>response</code>、<code>retrieved_contexts</code>、<code>reference</code>）

### 常见失败模式

## 生产考量

### Embedding 模型选型

Embedding 模型是 RAG 系统中影响最大的单项组件选择——它决定了检索的质量上限。该领域发展迅速；表 表 16.5 汇总了当前在成本—质量谱系上的可选方案。

<div class="hh-table" id="ch16.t5"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>模型</span></th><th class="ltx_align_left"><span>维度</span></th><th class="ltx_align_left"><span>最大 Token 数</span></th><th class="ltx_align_left"><span>MTEB 均值</span></th><th class="ltx_align_left"><span>访问方式</span></th><th class="ltx_align_left"><span>备注</span></th></tr></thead><tbody><tr><td class="ltx_align_left"><em>基于 API（托管）</em></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td class="ltx_align_left"><span>Voyage <span>voyage-4-large</span></span></td><td class="ltx_align_left"><span>1024*</span></td><td class="ltx_align_left"><span>32K</span></td><td class="ltx_align_left"><span>—</span></td><td class="ltx_align_left"><span>API</span></td><td class="ltx_align_left"><span>最佳检索质量</span></td></tr><tr><td class="ltx_align_left"><span>OpenAI <span>text-embedding-3-large</span></span></td><td class="ltx_align_left"><span>3072</span></td><td class="ltx_align_left"><span>8191</span></td><td class="ltx_align_left"><span>64.6</span></td><td class="ltx_align_left"><span>API</span></td><td class="ltx_align_left"><span>套娃式维度</span></td></tr><tr><td class="ltx_align_left"><span>Cohere <span>embed-english-v3.0</span></span></td><td class="ltx_align_left"><span>1024</span></td><td class="ltx_align_left"><span>512</span></td><td class="ltx_align_left"><span>64.5</span></td><td class="ltx_align_left"><span>API</span></td><td class="ltx_align_left"><span>支持 int8/二进制</span></td></tr><tr><td class="ltx_align_left"><span>Google <span>text-embedding-005</span></span></td><td class="ltx_align_left"><span>768</span></td><td class="ltx_align_left"><span>2048</span></td><td class="ltx_align_left"><span>—</span></td><td class="ltx_align_left"><span>API</span></td><td class="ltx_align_left"><span>Vertex AI 集成</span></td></tr><tr><td class="ltx_align_left"><em>开放权重（自托管）</em></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td class="ltx_align_left"><span>nvidia/NV-Embed-v2<span><span>[203]</span></span></span></td><td class="ltx_align_left"><span>4096</span></td><td class="ltx_align_left"><span>32K</span></td><td class="ltx_align_left"><span>72.3</span></td><td class="ltx_align_left"><span>免费</span></td><td class="ltx_align_left"><span>#1 MTEB（2024 年 9 月）</span></td></tr><tr><td class="ltx_align_left"><span>Alibaba-NLP/gte-Qwen2-7B<span><span>[214]</span></span></span></td><td class="ltx_align_left"><span>3584</span></td><td class="ltx_align_left"><span>32K</span></td><td class="ltx_align_left"><span>70.2</span></td><td class="ltx_align_left"><span>免费</span></td><td class="ltx_align_left"><span>Apache-2.0，多语言</span></td></tr><tr><td class="ltx_align_left"><span>BAAI/bge-m3<span><span>[45]</span></span></span></td><td class="ltx_align_left"><span>1024</span></td><td class="ltx_align_left"><span>8192</span></td><td class="ltx_align_left"><span>65.0</span></td><td class="ltx_align_left"><span>免费</span></td><td class="ltx_align_left"><span>稠密 + 稀疏 + 多向量</span></td></tr><tr><td class="ltx_align_left"><span>jinaai/jina-embeddings-v3</span></td><td class="ltx_align_left"><span>1024</span></td><td class="ltx_align_left"><span>8192</span></td><td class="ltx_align_left"><span>66.0</span></td><td class="ltx_align_left"><span>免费</span></td><td class="ltx_align_left"><span>多语言，LoRA 适配器</span></td></tr><tr><td class="ltx_align_left"><span>BAAI/bge-large-en-v1.5<span><span>[395]</span></span></span></td><td class="ltx_align_left"><span>1024</span></td><td class="ltx_align_left"><span>512</span></td><td class="ltx_align_left"><span>64.2</span></td><td class="ltx_align_left"><span>免费</span></td><td class="ltx_align_left"><span>成熟，支持完善</span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">表 16.5：</span>用于生产级 RAG 的 Embedding 模型（截至 2026 年）。MTEB 分数是检索、分类、聚类与 STS 任务上的总体平均值。</p></div>

<ul><li>领域匹配：专用模型（例如用于代码的 <code>voyage-code-3</code>、用于金融的 <code>voyage-finance-2</code>）在领域任务上可以比通用模型高出 5–15%。</li><li>上下文长度：具有 32K Token 上下文的模型（Voyage-4、NV-Embed-v2）可以嵌入整篇文档而无需分块，从而简化流水线。</li><li>套娃式维度嵌入（Matryoshka embeddings）：支持灵活输出维度（256–4096）的模型让你可以在服务时用质量换取存储/延迟，而无需重新编码。</li><li>量化支持：模型层面的 int8 或二进制量化（Cohere、Voyage）可将索引大小减少 4–32<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>，而召回损失极小。</li><li>多语言：对于非英语或跨语言 RAG，优先选择明确以多语言训练的模型（BGE-M3、Jina-v3、Voyage-4）。</li></ul>

### 向量数据库对比

<div class="hh-table" id="ch16.t6"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>数据库</span></th><th class="ltx_align_left"><span>托管方式</span></th><th class="ltx_align_left"><span>规模</span></th><th class="ltx_align_left"><span>过滤</span></th><th class="ltx_align_left"><span>混合检索</span></th><th class="ltx_align_left"><span>适用场景</span></th></tr></thead><tbody><tr><td class="ltx_align_left"><span>FAISS1</span></td><td class="ltx_align_left"><span>自托管</span></td><td class="ltx_align_left"><span>数十亿</span></td><td class="ltx_align_left"><span>有限</span></td><td class="ltx_align_left"><span>否</span></td><td class="ltx_align_left"><span>研究、离线</span></td></tr><tr><td class="ltx_align_left"><span>Pinecone2</span></td><td class="ltx_align_left"><span>托管</span></td><td class="ltx_align_left"><span>数十亿</span></td><td class="ltx_align_left"><span>是</span></td><td class="ltx_align_left"><span>是</span></td><td class="ltx_align_left"><span>Serverless，易于搭建</span></td></tr><tr><td class="ltx_align_left"><span>Weaviate3</span></td><td class="ltx_align_left"><span>两者皆可</span></td><td class="ltx_align_left"><span>数十亿</span></td><td class="ltx_align_left"><span>是</span></td><td class="ltx_align_left"><span>是</span></td><td class="ltx_align_left"><span>GraphQL，多模态</span></td></tr><tr><td class="ltx_align_left"><span>Chroma4</span></td><td class="ltx_align_left"><span>自托管</span></td><td class="ltx_align_left"><span>百万级</span></td><td class="ltx_align_left"><span>是</span></td><td class="ltx_align_left"><span>否</span></td><td class="ltx_align_left"><span>本地开发、原型验证</span></td></tr><tr><td class="ltx_align_left"><span>Qdrant5</span></td><td class="ltx_align_left"><span>两者皆可</span></td><td class="ltx_align_left"><span>数十亿</span></td><td class="ltx_align_left"><span>是</span></td><td class="ltx_align_left"><span>是</span></td><td class="ltx_align_left"><span>高性能</span></td></tr><tr><td class="ltx_align_left"><span>Milvus6</span></td><td class="ltx_align_left"><span>两者皆可</span></td><td class="ltx_align_left"><span>数十亿</span></td><td class="ltx_align_left"><span>是</span></td><td class="ltx_align_left"><span>是</span></td><td class="ltx_align_left"><span>企业级，GPU 加速</span></td></tr><tr><td class="ltx_align_left"><span>pgvector7</span></td><td class="ltx_align_left"><span>自托管</span></td><td class="ltx_align_left"><span>百万级</span></td><td class="ltx_align_left"><span>是</span></td><td class="ltx_align_left"><span>是</span></td><td class="ltx_align_left"><span>现有 Postgres 用户</span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">表 16.6：</span>面向生产级 RAG 系统的向量数据库对比</p></div>

5<a href="https://qdrant.tech" target="_blank" rel="noopener noreferrer">https://qdrant.tech</a> 6<a href="https://milvus.io" target="_blank" rel="noopener noreferrer">https://milvus.io</a> 7<a href="https://github.com/pgvector/pgvector" target="_blank" rel="noopener noreferrer">https://github.com/pgvector/pgvector</a>

### 延迟优化

<ol><li>预过滤：在 ANN 搜索之前使用元数据过滤（日期范围、类别、来源）来缩小搜索空间</li><li>近似最近邻（Approximate NN）：使用 HNSW 或 IVF 索引替代精确搜索；以 <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>1% 的召回损失换取 <math alttext="10\times" display="inline"><semantics><mrow><mn>10</mn><mo lspace="0.222em">×</mo></mrow></semantics></math> 的加速</li><li>Embedding 缓存：缓存频繁重复查询的嵌入</li><li>异步检索：并行从多个来源检索</li><li>流式生成：在检索完成的同时流式输出 LLM 结果</li><li>量化：对嵌入使用 int8 或二进制量化，以降低内存占用并提高吞吐量</li></ol>

上文的技术（3）和（4）可以自然组合：先缓存查询嵌入，再把检索请求并发扇出到多个后端。在多源 RAG 系统（第 16.7 智能体 RAG 节）中，用户查询可能需要来自向量数据库、关键词索引和网页 API 的结果。串行检索会累加延迟；并行检索只需付出<em>最慢</em>来源的代价。代码清单 使用 Python 的 <code>asyncio</code> 演示了这一模式——<code>lru_cache</code> 装饰器确保重复查询完全跳过嵌入模型，而 <code>asyncio.gather</code> 则同时派发所有来源查询。

```
import asyncio
from functools import lru_cache

@lru_cache(maxsize=1024)
def get_cached_embedding(text: str) -> list[float]:
    return embedding_model.embed_query(text)

async def parallel_retrieve(
    query: str,
    sources: list[str],
    k: int = 5
) -> list[dict]:
    """Retrieve from multiple sources in parallel."""
    tasks = [
        asyncio.create_task(retrieve_from_source_async(query, src, k))
        for src in sources
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    # Flatten and deduplicate
    all_docs = []
    for r in results:
        if not isinstance(r, Exception):
            all_docs.extend(r)
    return deduplicate_by_content(all_docs)
```

<span class="hh-tag">代码清单 20：</span>面向低延迟的异步并行检索

### 增量索引与版本管理

在生产环境中，文档语料库从不是静态的——政策会被修订，新报告每天入库，过时内容必须移除。全量重建索引（重新分块、重新嵌入、重新上传）代价高昂且会导致服务中断。增量索引通过在文档级别应用变更来解决这一问题。

<ul><li>Upsert：当文档被创建或更新时，删除该 <code>doc_id</code> 的所有现有分块，再对新内容重新分块、嵌入并插入。这保证不会残留过期片段。</li><li>删除/过期：按文档 ID 移除分块（显式删除），或按 TTL 移除（对新闻、市场数据等时效性来源自动回收垃圾）。</li><li>版本追踪：在分块元数据中存储 <code>version</code> 和 <code>indexed_at</code> 时间戳。这使得回滚（从来源恢复先前版本）与可审计性（“模型看到的是哪个版本？”）成为可能。</li></ul>

<ul><li>Embedding 模型漂移：如果升级 Embedding 模型，新旧向量互不兼容。解决方案：(a) 为每个模型版本维护独立索引，并在后台迁移；或 (b) 使用兼容套娃式维度的模型，其维度截断可保持兼容性。</li><li>分块边界变化：更改分块策略会使所有现有分块失效。版本元数据让你能够识别并有选择地重建受影响文档的索引。</li><li>最终一致性：在分布式向量数据库中，新写入的向量可能无法立即被搜索到。设计流水线时要能容忍短暂的索引延迟（通常为几秒到几分钟）。</li></ul>

代码清单 21 展示了一个最小的 <code>RAGIndexManager</code> 类，封装了 upsert 与过期逻辑，适合包装任何支持元数据过滤的向量库。

```
class RAGIndexManager:
    def __init__(self, vectorstore, metadata_store, chunker, embedder):
        self.vs = vectorstore
        self.meta = metadata_store
        self.chunker = chunker
        self.embedder = embedder

    def upsert_document(self, doc_id: str, content: str,
                        metadata: dict) -> None:
        """Add or update a document, replacing old chunks."""
        # Delete existing chunks for this document
        self.vs.delete(filter={"doc_id": doc_id})
        # Chunk new version (vectorstore embeds internally)
        chunks = self.chunker.split_text(content)
        self.vs.add_texts(
            texts=chunks,
            metadatas=[{**metadata, "doc_id": doc_id,
                        "version": metadata.get("version", 1),
                        "indexed_at": datetime.utcnow().isoformat()}
                       for _ in chunks],
        )

    def expire_old_documents(self, ttl_days: int = 365) -> int:
        """Remove documents older than TTL."""
        cutoff = (datetime.utcnow() - timedelta(days=ttl_days)).isoformat()
        return self.vs.delete(filter={"indexed_at": {"$lt": cutoff}})
```

<span class="hh-tag">代码清单 21：</span>带版本管理的增量索引更新

## RAG 与微调的协同

### 何时将 RAG 与微调结合

微调与 RAG 解决的是互补的短板：

<ul><li>仅微调：模型学会风格与格式，但可能编造事实</li><li>仅 RAG：模型能够获取事实，但可能不知道如何最优地使用它们</li><li>两者结合：微调模型，使其<em>善用检索到的上下文</em>——引用来源、承认不确定性，并忽略无关上下文</li></ul>

### RAFT：检索增强微调（Retrieval-Augmented Fine-Tuning）

RAFT [425] 训练模型在混合了相关文档与<em>干扰</em>文档的情况下回答问题，教会模型识别并只使用相关上下文：

<ol><li>对每个训练样本 <math alttext="(q,a,d^{*})" display="inline"><semantics><mrow><mo stretchy="false">(</mo><mi>q</mi><mo>,</mo><mi>a</mi><mo>,</mo><msup><mi>d</mi><mo>∗</mo></msup><mo stretchy="false">)</mo></mrow></semantics></math>，采样 <math alttext="k-1" display="inline"><semantics><mrow><mi>k</mi><mo>−</mo><mn>1</mn></mrow></semantics></math> 篇干扰文档 <math alttext="\{d_{i}^{-}\}" display="inline"><semantics><mrow><mo stretchy="false">{</mo><msubsup><mi>d</mi><mi>i</mi><mo>−</mo></msubsup><mo stretchy="false">}</mo></mrow></semantics></math></li><li>微调输入：<code>[q, <math alttext="d^{*}" display="inline"><semantics><msup><mi>d</mi><mo>∗</mo></msup></semantics></math>, <math alttext="d_{1}^{-}" display="inline"><semantics><msubsup><mi>d</mi><mn>1</mn><mo>−</mo></msubsup></semantics></math>, …, <math alttext="d_{k-1}^{-}" display="inline"><semantics><msubsup><mi>d</mi><mrow><mi>k</mi><mo>−</mo><mn>1</mn></mrow><mo>−</mo></msubsup></semantics></math>]</code><math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math><code>[chain-of-thought + a]</code></li><li>思维链明确引用 <math alttext="d^{*}" display="inline"><semantics><msup><mi>d</mi><mo>∗</mo></msup></semantics></math> 的内容，教会模型让答案有据可依</li></ol>

<div class="hh-equation" id="ch16.e30"><math alttext="\mathcal{L}_{\text{RAFT}}=-\mathbb{E}_{(q,a,d^{*},\{d_{i}^{-}\})}\left[\log P_{\theta}\!\left(\text{CoT}(d^{*})\oplus a\;\middle|\;q,d^{*},\{d_{i}^{-}\}\right)\right]" display="block"><semantics><mrow><msub><mi>ℒ</mi><mtext>RAFT</mtext></msub><mo rspace="0em">=</mo><mo lspace="0em">−</mo><msub><mi>𝔼</mi><mrow><mo stretchy="false">(</mo><mi>q</mi><mo>,</mo><mi>a</mi><mo>,</mo><msup><mi>d</mi><mo>∗</mo></msup><mo>,</mo><mrow><mo stretchy="false">{</mo><msubsup><mi>d</mi><mi>i</mi><mo>−</mo></msubsup><mo stretchy="false">}</mo></mrow><mo stretchy="false">)</mo></mrow></msub><mrow><mo>[</mo><mi>log</mi><msub><mi>P</mi><mi>θ</mi></msub><mrow><mo>(</mo><mtext>CoT</mtext><mrow><mo stretchy="false">(</mo><msup><mi>d</mi><mo>∗</mo></msup><mo stretchy="false">)</mo></mrow><mo>⊕</mo><mi>a</mi><mo stretchy="true">|</mo><mi>q</mi><mo>,</mo><msup><mi>d</mi><mo>∗</mo></msup><mo>,</mo><mrow><mo stretchy="false">{</mo><msubsup><mi>d</mi><mi>i</mi><mo>−</mo></msubsup><mo stretchy="false">}</mo></mrow><mo>)</mo></mrow><mo>]</mo></mrow></mrow></semantics></math><span class="hh-equation-number">(16.30)</span></div>

### 检索器—生成器联合训练

为了获得最佳性能，检索器与生成器可以联合训练。REALM [128] 和 RAG [209] 论文提出端到端训练，让梯度流经检索步骤：

<div class="hh-equation" id="ch16.e31"><math alttext="\nabla_{\theta}\mathcal{L}=\nabla_{\theta}\left[-\log\sum_{d\in\mathcal{D}}P_{\theta}(a\mid q,d)\cdot P_{\phi}(d\mid q)\right]" display="block"><semantics><mrow><msub><mo>∇</mo><mi>θ</mi></msub><mi>ℒ</mi><mo>=</mo><msub><mo>∇</mo><mi>θ</mi></msub><mrow><mo>[</mo><mo lspace="0em">−</mo><mi>log</mi><munder><mo movablelimits="false">∑</mo><mrow><mi>d</mi><mo>∈</mo><mi>𝒟</mi></mrow></munder><msub><mi>P</mi><mi>θ</mi></msub><mrow><mo stretchy="false">(</mo><mi>a</mi><mo lspace="0em" rspace="0.167em">∣</mo><mi>q</mi><mo>,</mo><mi>d</mi><mo rspace="0.055em" stretchy="false">)</mo></mrow><mo rspace="0.222em">⋅</mo><msub><mi>P</mi><mi>ϕ</mi></msub><mrow><mo stretchy="false">(</mo><mi>d</mi><mo lspace="0em" rspace="0.167em">∣</mo><mi>q</mi><mo stretchy="false">)</mo></mrow><mo>]</mo></mrow></mrow></semantics></math><span class="hh-equation-number">(16.31)</span></div>

检索器参数 <math alttext="\phi" display="inline"><semantics><mi>ϕ</mi></semantics></math> 使用 REINFORCE 估计器更新，也可以把 <math alttext="P_{\phi}(d\mid q)" display="inline"><semantics><mrow><msub><mi>P</mi><mi>ϕ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>d</mi><mo fence="true" lspace="0em" rspace="0em">∣</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 视为对文档的可微注意力。

## 各种 RAG 方法的综合对比

<div class="hh-table" id="ch16.t7"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>方法</span></th><th class="ltx_align_left"><span>准确率</span></th><th class="ltx_align_left"><span>延迟</span></th><th class="ltx_align_left"><span>复杂度</span></th><th class="ltx_align_left"><span>成本</span></th><th class="ltx_align_left"><span>适用场景</span></th></tr></thead><tbody><tr><td class="ltx_align_left"><span>朴素 RAG <span>[209]</span></span></td><td class="ltx_align_left"><span>中</span></td><td class="ltx_align_left"><span>低</span></td><td class="ltx_align_left"><span>低</span></td><td class="ltx_align_left"><span>低</span></td><td class="ltx_align_left"><span>原型验证、简单问答</span></td></tr><tr><td class="ltx_align_left"><span>RAG + 重排序 <span>[268]</span></span></td><td class="ltx_align_left"><span>高</span></td><td class="ltx_align_left"><span>中</span></td><td class="ltx_align_left"><span>中</span></td><td class="ltx_align_left"><span>中</span></td><td class="ltx_align_left"><span>生产级问答系统</span></td></tr><tr><td class="ltx_align_left"><span>HyDE <span>[106]</span></span></td><td class="ltx_align_left"><span>高</span></td><td class="ltx_align_left"><span>中</span></td><td class="ltx_align_left"><span>低</span></td><td class="ltx_align_left"><span>中</span></td><td class="ltx_align_left"><span>语义不匹配的领域</span></td></tr><tr><td class="ltx_align_left"><span>Multi-Query RAG</span></td><td class="ltx_align_left"><span>高</span></td><td class="ltx_align_left"><span>中</span></td><td class="ltx_align_left"><span>中</span></td><td class="ltx_align_left"><span>中</span></td><td class="ltx_align_left"><span>歧义查询</span></td></tr><tr><td class="ltx_align_left"><span>RAG-Fusion <span>[300]</span></span></td><td class="ltx_align_left"><span>高</span></td><td class="ltx_align_left"><span>中</span></td><td class="ltx_align_left"><span>中</span></td><td class="ltx_align_left"><span>中</span></td><td class="ltx_align_left"><span>多样的查询类型</span></td></tr><tr><td class="ltx_align_left"><span>Self-RAG <span>[13]</span></span></td><td class="ltx_align_left"><span>高</span></td><td class="ltx_align_left"><span>中</span></td><td class="ltx_align_left"><span>高</span></td><td class="ltx_align_left"><span>中</span></td><td class="ltx_align_left"><span>选择性检索</span></td></tr><tr><td class="ltx_align_left"><span>CRAG <span>[401]</span></span></td><td class="ltx_align_left"><span>高</span></td><td class="ltx_align_left"><span>中</span></td><td class="ltx_align_left"><span>高</span></td><td class="ltx_align_left"><span>高</span></td><td class="ltx_align_left"><span>不可靠的语料库</span></td></tr><tr><td class="ltx_align_left"><span>Adaptive RAG <span>[162]</span></span></td><td class="ltx_align_left"><span>高</span></td><td class="ltx_align_left"><span>低–高</span></td><td class="ltx_align_left"><span>高</span></td><td class="ltx_align_left"><span>中</span></td><td class="ltx_align_left"><span>混合的查询复杂度</span></td></tr><tr><td class="ltx_align_left"><span>Graph RAG <span>[82]</span></span></td><td class="ltx_align_left"><span>极高</span></td><td class="ltx_align_left"><span>高</span></td><td class="ltx_align_left"><span>极高</span></td><td class="ltx_align_left"><span>高</span></td><td class="ltx_align_left"><span>全局综合类查询</span></td></tr><tr><td class="ltx_align_left"><span>Agentic RAG</span></td><td class="ltx_align_left"><span>极高</span></td><td class="ltx_align_left"><span>高</span></td><td class="ltx_align_left"><span>极高</span></td><td class="ltx_align_left"><span>高</span></td><td class="ltx_align_left"><span>多跳推理</span></td></tr><tr><td class="ltx_align_left"><span>RAFT <span>[425]</span></span></td><td class="ltx_align_left"><span>极高</span></td><td class="ltx_align_left"><span>低</span></td><td class="ltx_align_left"><span>极高</span></td><td class="ltx_align_left"><span>极高</span></td><td class="ltx_align_left"><span>领域专用部署</span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">表 16.7：</span>各类 RAG 方法在关键维度上的对比</p></div>

<aside class="hh-next">
<p class="hh-next-label">下一篇</p>
<p class="hh-next-title">第 17 章 智能体记忆系统</p>
<p class="hh-next-hint">在「智能体 AI 漫游指南」分类下继续阅读。</p>
</aside>

</div>
