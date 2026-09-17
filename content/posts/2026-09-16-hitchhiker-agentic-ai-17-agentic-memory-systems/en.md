---
title: "The Hitchhiker's Guide to Agentic AI · Chapter 17 Agentic Memory Systems"
slug: "hitchhiker-agentic-ai-17-agentic-memory-systems"
lang: "en"
date: "2026-09-16T00:18:00.000Z"
draft: false
featured: false
categories:
  - "hitchhiker-agentic-ai"
cover: "./cover.png"
description: "Large language models are, at their core, stateless function approximators: given a prompt x, they produce a distribution over continuations…"
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
<p class="hh-part-title">Part V Agentic AI</p>
</aside>

## Motivation: Why Agents Need Memory

Large language models are, at their core, stateless function approximators: given a prompt <math alttext="x" display="inline"><semantics><mi>x</mi></semantics></math>, they produce a distribution over continuations <math alttext="p_{\theta}(y\mid x)" display="inline"><semantics><mrow><msub><mi>p</mi><mi>θ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>y</mi><mo fence="true" lspace="0em" rspace="0em">∣</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math>. Every inference call begins from scratch. The <em>context window</em>—the finite sequence of tokens the model can attend to—is the only information available at generation time. For short, self-contained tasks this is sufficient. For long-horizon agentic tasks it is a fundamental bottleneck.

Three distinct failure modes arise when agents lack persistent memory:

<ol><li>Catastrophic forgetting of context. Once an event scrolls out of the context window it is irrecoverably lost. The agent cannot refer back to a decision made 10,000 tokens ago.</li><li>Inability to learn from experience. Without episodic storage, every episode is the agent’s first. Successful strategies cannot be reused; mistakes are repeated.</li><li>Lack of personalization. User preferences, domain facts, and relationship history must be re-established in every session, degrading user experience and efficiency.</li></ol>

Formally, we model an agent as a tuple <math alttext="\mathcal{A}=(\pi_{\theta},\mathcal{M},\mathcal{R},\mathcal{W})" display="inline"><semantics><mrow><mi>𝒜</mi><mo>=</mo><mrow><mo stretchy="false">(</mo><msub><mi>π</mi><mi>θ</mi></msub><mo>,</mo><mi>ℳ</mi><mo>,</mo><mi>ℛ</mi><mo>,</mo><mi>𝒲</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math> where <math alttext="\pi_{\theta}" display="inline"><semantics><msub><mi>π</mi><mi>θ</mi></msub></semantics></math> is the policy (the LLM), <math alttext="\mathcal{M}" display="inline"><semantics><mi>ℳ</mi></semantics></math> is the memory store, <math alttext="\mathcal{R}:\mathcal{Q}\times\mathcal{M}\to\mathcal{D}" display="inline"><semantics><mrow><mi>ℛ</mi><mo lspace="0.278em" rspace="0.278em">:</mo><mrow><mrow><mi>𝒬</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mi>ℳ</mi></mrow><mo stretchy="false">→</mo><mi>𝒟</mi></mrow></mrow></semantics></math> is a retrieval function mapping queries to retrieved documents, and <math alttext="\mathcal{W}:\mathcal{M}\times\mathcal{E}\to\mathcal{M}" display="inline"><semantics><mrow><mi>𝒲</mi><mo lspace="0.278em" rspace="0.278em">:</mo><mrow><mrow><mi>ℳ</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mi>ℰ</mi></mrow><mo stretchy="false">→</mo><mi>ℳ</mi></mrow></mrow></semantics></math> is a write function updating memory with new experiences <math alttext="\mathcal{E}" display="inline"><semantics><mi>ℰ</mi></semantics></math>. At each step <math alttext="t" display="inline"><semantics><mi>t</mi></semantics></math> the agent observes <math alttext="o_{t}" display="inline"><semantics><msub><mi>o</mi><mi>t</mi></msub></semantics></math>, retrieves relevant context <math alttext="c_{t}=\mathcal{R}(o_{t},\mathcal{M})" display="inline"><semantics><mrow><msub><mi>c</mi><mi>t</mi></msub><mo>=</mo><mrow><mi>ℛ</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>o</mi><mi>t</mi></msub><mo>,</mo><mi>ℳ</mi><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math>, and acts:

<div class="hh-equation" id="ch17.ex1"><math alttext="a_{t}\sim\pi_{\theta}\!\left(\cdot\;\middle|\;[s_{t};\,c_{t};\,h_{t}]\right)," display="block"><semantics><mrow><msub><mi>a</mi><mi>t</mi></msub><mo>∼</mo><msub><mi>π</mi><mi>θ</mi></msub><mrow><mo>(</mo><mo lspace="0em" rspace="0.224em">⋅</mo><mo stretchy="true">|</mo><mrow><mo stretchy="false">[</mo><msub><mi>s</mi><mi>t</mi></msub><mo rspace="0.337em">;</mo><msub><mi>c</mi><mi>t</mi></msub><mo rspace="0.337em">;</mo><msub><mi>h</mi><mi>t</mi></msub><mo stretchy="false">]</mo></mrow><mo>)</mo></mrow><mo>,</mo></mrow></semantics></math></div>

where <math alttext="s_{t}" display="inline"><semantics><msub><mi>s</mi><mi>t</mi></msub></semantics></math> is the current system prompt, <math alttext="c_{t}" display="inline"><semantics><msub><mi>c</mi><mi>t</mi></msub></semantics></math> is retrieved memory, and <math alttext="h_{t}" display="inline"><semantics><msub><mi>h</mi><mi>t</mi></msub></semantics></math> is the recent in-context history. After acting, the agent may write new information: <math alttext="\mathcal{M}\leftarrow\mathcal{W}(\mathcal{M},\,(o_{t},a_{t},r_{t}))" display="inline"><semantics><mrow><mi>ℳ</mi><mo stretchy="false">←</mo><mrow><mi>𝒲</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>ℳ</mi><mo>,</mo><mrow><mo stretchy="false">(</mo><msub><mi>o</mi><mi>t</mi></msub><mo>,</mo><msub><mi>a</mi><mi>t</mi></msub><mo>,</mo><msub><mi>r</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math>.

## Taxonomy of Memory Types

<figure id="ch17.f1"><img src="./fig_052_memory-taxonomy.png" alt="Figure 17.1: Four-way taxonomy of agentic memory systems, mirroring cognitive science distinctions. Each memory type has" loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 17.1:</span>Four-way taxonomy of agentic memory systems, mirroring cognitive science distinctions. Each memory type has distinct access patterns, update frequencies, and retrieval mechanisms.</figcaption></figure>

### Working Memory (Short-Term)

Working memory is the agent’s <em>active workspace</em>: the information currently being manipulated. In LLM agents it corresponds to:

<ul><li>Scratchpads. Intermediate reasoning steps written to a dedicated buffer before producing a final answer (e.g. chain-of-thought [374], scratchpad [272]).</li><li>Chain-of-thought buffers. The sequence of reasoning tokens <math alttext="z_{1},z_{2},\ldots,z_{k}" display="inline"><semantics><mrow><msub><mi>z</mi><mn>1</mn></msub><mo>,</mo><msub><mi>z</mi><mn>2</mn></msub><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msub><mi>z</mi><mi>k</mi></msub></mrow></semantics></math> generated before the answer token <math alttext="a" display="inline"><semantics><mi>a</mi></semantics></math>, modeled as <math alttext="p(a\mid x)=\sum_{z}p(a\mid x,z)\,p(z\mid x)" display="inline"><semantics><mrow><mrow><mi>p</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>a</mi><mo fence="true" lspace="0em" rspace="0em">∣</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow><mo rspace="0.111em">=</mo><mrow><msub><mo>∑</mo><mi>z</mi></msub><mrow><mrow><mi>p</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>a</mi><mo fence="true" lspace="0em" rspace="0em">∣</mo><mrow><mi>x</mi><mo>,</mo><mi>z</mi></mrow><mo stretchy="false">)</mo></mrow></mrow><mo lspace="0.170em" rspace="0em">​</mo><mi>p</mi><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>z</mi><mo fence="true" lspace="0em" rspace="0em">∣</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></semantics></math>.</li><li>Conversation context. The recent turn history <math alttext="[(u_{1},a_{1}),\ldots,(u_{t},a_{t})]" display="inline"><semantics><mrow><mo stretchy="false">[</mo><mrow><mrow><mo stretchy="false">(</mo><msub><mi>u</mi><mn>1</mn></msub><mo>,</mo><msub><mi>a</mi><mn>1</mn></msub><mo stretchy="false">)</mo></mrow><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><mrow><mo stretchy="false">(</mo><msub><mi>u</mi><mi>t</mi></msub><mo>,</mo><msub><mi>a</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo stretchy="false">]</mo></mrow></semantics></math> kept in the context window.</li></ul>

Working memory is <em>fast</em> (zero retrieval latency—it is already in context), <em>volatile</em> (lost when the context is cleared), and <em>capacity-limited</em> (bounded by <math alttext="L" display="inline"><semantics><mi>L</mi></semantics></math>).

### Episodic Memory (Experience-Based)

Episodic memory stores <em>specific past events</em> indexed by context and time. For agents:

<ul><li>Past interactions. Full or summarized records of prior conversations, task attempts, and their outcomes.</li><li>Successful trajectories. High-reward action sequences that can be retrieved as few-shot exemplars for similar future tasks.</li><li>Failure cases. Documented mistakes with root-cause annotations, enabling the agent to avoid repeating errors.</li><li>Retrieval-augmented episodic recall. Given a new task <math alttext="q" display="inline"><semantics><mi>q</mi></semantics></math>, retrieve the <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> most similar past episodes <math alttext="\{e_{i}\}_{i=1}^{k}" display="inline"><semantics><msubsup><mrow><mo stretchy="false">{</mo><msub><mi>e</mi><mi>i</mi></msub><mo stretchy="false">}</mo></mrow><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>k</mi></msubsup></semantics></math> and include them in context.</li></ul>

Episodic memory is typically implemented as a vector store (Section 17.3.1 RAG-Based Memory) with embeddings over episode summaries.

### Semantic Memory (World Knowledge)

Semantic memory encodes <em>general facts and concepts</em> decoupled from specific episodes:

<ul><li>Factual knowledge. Entities, attributes, and relationships (e.g. “Paris is the capital of France”).</li><li>Domain concepts. Definitions, taxonomies, and ontologies relevant to the agent’s task domain.</li><li>Knowledge graphs. Structured representations <math alttext="\mathcal{G}=(\mathcal{V},\mathcal{E})" display="inline"><semantics><mrow><mi>𝒢</mi><mo>=</mo><mrow><mo stretchy="false">(</mo><mi>𝒱</mi><mo>,</mo><mi>ℰ</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math> where nodes <math alttext="v\in\mathcal{V}" display="inline"><semantics><mrow><mi>v</mi><mo>∈</mo><mi>𝒱</mi></mrow></semantics></math> are entities and edges <math alttext="e\in\mathcal{E}" display="inline"><semantics><mrow><mi>e</mi><mo>∈</mo><mi>ℰ</mi></mrow></semantics></math> are typed relations.</li></ul>

Unlike episodic memory, semantic memory is <em>context-independent</em>: the fact that water boils at <math alttext="100^{\circ}" display="inline"><semantics><msup><mn>100</mn><mo>∘</mo></msup></semantics></math>C is true regardless of when or where it was learned.

### Procedural Memory (Skills)

Procedural memory encodes <em>how to do things</em>—skills and action patterns that have been automatized:

<ul><li>Learned tool-use patterns. Which API to call for which task, how to format inputs, how to handle errors.</li><li>Action sequences. Multi-step procedures (e.g. “to deploy code: run tests <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> build image <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> push <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> update manifest”).</li><li>Policies as memory. The model weights <math alttext="\theta" display="inline"><semantics><mi>θ</mi></semantics></math> themselves encode procedural knowledge; fine-tuning on successful trajectories is a form of procedural memory consolidation.</li></ul>

## Memory Architectures

### RAG-Based Memory

Retrieval-Augmented Generation (RAG) [209] is the dominant paradigm for external memory in LLM agents. The memory store <math alttext="\mathcal{M}" display="inline"><semantics><mi>ℳ</mi></semantics></math> is a collection of documents <math alttext="\{d_{i}\}_{i=1}^{N}" display="inline"><semantics><msubsup><mrow><mo stretchy="false">{</mo><msub><mi>d</mi><mi>i</mi></msub><mo stretchy="false">}</mo></mrow><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>N</mi></msubsup></semantics></math>; retrieval maps a query <math alttext="q" display="inline"><semantics><mi>q</mi></semantics></math> to a ranked subset.

Each document <math alttext="d_{i}" display="inline"><semantics><msub><mi>d</mi><mi>i</mi></msub></semantics></math> is encoded by an embedding model <math alttext="\phi" display="inline"><semantics><mi>ϕ</mi></semantics></math>: <math alttext="\mathbf{v}_{i}=\phi(d_{i})\in\mathbb{R}^{D}" display="inline"><semantics><mrow><msub><mi>𝐯</mi><mi>i</mi></msub><mo>=</mo><mrow><mi>ϕ</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>d</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>∈</mo><msup><mi>ℝ</mi><mi>D</mi></msup></mrow></semantics></math>. Queries are similarly encoded: <math alttext="\mathbf{q}=\phi(q)" display="inline"><semantics><mrow><mi>𝐪</mi><mo>=</mo><mrow><mi>ϕ</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math>. Retrieval returns the top-<math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> documents by similarity:

<div class="hh-equation" id="ch17.ex2"><math alttext="\text{Retrieve}(q,\mathcal{M},k)=\underset{S\subseteq[N],\,|S|=k}{\arg\max}\sum_{i\in S}\text{sim}(\mathbf{q},\mathbf{v}_{i})," display="block"><semantics><mrow><mrow><mrow><mtext>Retrieve</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>q</mi><mo>,</mo><mi>ℳ</mi><mo>,</mo><mi>k</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><munder accentunder="true"><mrow><mi>arg</mi><mo lspace="0.167em">⁡</mo><mi>max</mi></mrow><mrow><mrow><mi mathsize="0.700em">S</mi><mo mathsize="0.700em">⊆</mo><mrow><mo maxsize="0.700em" minsize="0.700em">[</mo><mi mathsize="0.700em">N</mi><mo maxsize="0.700em" minsize="0.700em">]</mo></mrow></mrow><mo mathsize="0.700em" rspace="0.337em">,</mo><mrow><mrow><mo maxsize="0.700em" minsize="0.700em" stretchy="true">|</mo><mi mathsize="0.700em">S</mi><mo maxsize="0.700em" minsize="0.700em" stretchy="true">|</mo></mrow><mo mathsize="0.700em">=</mo><mi mathsize="0.700em">k</mi></mrow></mrow></munder><mo lspace="0em" rspace="0em">​</mo><mrow><munder><mo movablelimits="false">∑</mo><mrow><mi>i</mi><mo>∈</mo><mi>S</mi></mrow></munder><mrow><mtext>sim</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>𝐪</mi><mo>,</mo><msub><mi>𝐯</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></mrow><mo>,</mo></mrow></semantics></math></div>

where <math alttext="\text{sim}(\cdot,\cdot)" display="inline"><semantics><mrow><mtext>sim</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mo lspace="0em" rspace="0em">⋅</mo><mo rspace="0em">,</mo><mo lspace="0em" rspace="0em">⋅</mo><mo stretchy="false">)</mo></mrow></mrow></semantics></math> is typically cosine similarity. Approximate nearest-neighbor (ANN) indices (FAISS [171], HNSW [248], ScaNN [126]) make this tractable for <math alttext="N\sim 10^{7}" display="inline"><semantics><mrow><mi>N</mi><mo>∼</mo><msup><mn>10</mn><mn>7</mn></msup></mrow></semantics></math>.

<ul><li>Dense retrieval. Both query and documents are encoded by neural encoders (e.g. DPR [180], <code>text-embedding-3-large</code>). Captures semantic similarity but requires GPU inference.</li><li>Sparse retrieval. BM25 or TF-IDF over token overlap. Fast, interpretable, strong for exact keyword matches.</li><li>Hybrid retrieval. Combine dense and sparse scores via reciprocal rank fusion (RRF):<math alttext="\text{RRF}(d,k)=\sum_{r\in\text{rankers}}\frac{1}{k+\text{rank}_{r}(d)}," display="block"><semantics><mrow><mrow><mrow><mtext>RRF</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>d</mi><mo>,</mo><mi>k</mi><mo stretchy="false">)</mo></mrow></mrow><mo rspace="0.111em">=</mo><mrow><munder><mo movablelimits="false">∑</mo><mrow><mi>r</mi><mo>∈</mo><mtext>rankers</mtext></mrow></munder><mfrac><mn>1</mn><mrow><mi>k</mi><mo>+</mo><mrow><msub><mtext>rank</mtext><mi>r</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>d</mi><mo stretchy="false">)</mo></mrow></mrow></mrow></mfrac></mrow></mrow><mo>,</mo></mrow></semantics></math>where <math alttext="k=60" display="inline"><semantics><mrow><mi>k</mi><mo>=</mo><mn>60</mn></mrow></semantics></math> is a smoothing constant. Hybrid consistently outperforms either alone [44].</li></ul>

A cross-encoder re-ranker <math alttext="f_{\psi}(q,d)\in[0,1]" display="inline"><semantics><mrow><mrow><msub><mi>f</mi><mi>ψ</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>q</mi><mo>,</mo><mi>d</mi><mo stretchy="false">)</mo></mrow></mrow><mo>∈</mo><mrow><mo stretchy="false">[</mo><mrow><mn>0</mn><mo>,</mo><mn>1</mn></mrow><mo stretchy="false">]</mo></mrow></mrow></semantics></math> scores each retrieved document jointly with the query, providing higher accuracy at the cost of <math alttext="O(k)" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>k</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math> forward passes. The pipeline is: retrieve <math alttext="k^{\prime}\gg k" display="inline"><semantics><mrow><msup><mi>k</mi><mo>′</mo></msup><mo>≫</mo><mi>k</mi></mrow></semantics></math> candidates with ANN, re-rank with cross-encoder, return top <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math>.

### Summarization-Based Memory

When verbatim storage is too expensive or noisy, <em>summarization</em> compresses information before storage.

At each step <math alttext="t" display="inline"><semantics><mi>t</mi></semantics></math>, the agent maintains a running summary <math alttext="S_{t}" display="inline"><semantics><msub><mi>S</mi><mi>t</mi></msub></semantics></math>. When new information <math alttext="e_{t}" display="inline"><semantics><msub><mi>e</mi><mi>t</mi></msub></semantics></math> arrives:

<div class="hh-equation" id="ch17.ex4"><math alttext="S_{t+1}=\text{LLM}\!\left(\texttt{``Summarize: [}S_{t}\texttt{] + [}e_{t}\texttt{]''}\right)." display="block"><semantics><mrow><msub><mi>S</mi><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>=</mo><mpadded width="2.270em"><mtext>LLM</mtext></mpadded><mrow><mo>(</mo><mtext>‘‘Summarize: [</mtext><msub><mi>S</mi><mi>t</mi></msub><mtext>] + [</mtext><msub><mi>e</mi><mi>t</mi></msub><mtext>]’’</mtext><mo>)</mo></mrow><mo lspace="0em">.</mo></mrow></semantics></math></div>

This keeps memory size <math alttext="O(1)" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></mrow></semantics></math> but risks losing detail.

Organize memory in levels <math alttext="L_{0}\supset L_{1}\supset\cdots\supset L_{K}" display="inline"><semantics><mrow><msub><mi>L</mi><mn>0</mn></msub><mo>⊃</mo><msub><mi>L</mi><mn>1</mn></msub><mo rspace="0.1389em">⊃</mo><mo lspace="0.1389em" rspace="0.1389em">⋯</mo><mo lspace="0.1389em">⊃</mo><msub><mi>L</mi><mi>K</mi></msub></mrow></semantics></math> where <math alttext="L_{0}" display="inline"><semantics><msub><mi>L</mi><mn>0</mn></msub></semantics></math> is verbatim and each <math alttext="L_{i+1}" display="inline"><semantics><msub><mi>L</mi><mrow><mi>i</mi><mo>+</mo><mn>1</mn></mrow></msub></semantics></math> is a summary of <math alttext="L_{i}" display="inline"><semantics><msub><mi>L</mi><mi>i</mi></msub></semantics></math>. Retrieval first checks <math alttext="L_{K}" display="inline"><semantics><msub><mi>L</mi><mi>K</mi></msub></semantics></math> (most compressed, fastest) and drills down as needed. This mirrors the <em>progressive summarization</em> technique of Forte [96].

<ul><li>Store verbatim: precise facts, code snippets, numerical results, user quotes.</li><li>Summarize: narrative context, reasoning chains, redundant observations.</li><li>Discard: noise, failed tool calls with no informational content.</li></ul>

### Graph-Based Memory

A knowledge graph <math alttext="\mathcal{G}=(\mathcal{V},\mathcal{E},\mathcal{R})" display="inline"><semantics><mrow><mi>𝒢</mi><mo>=</mo><mrow><mo stretchy="false">(</mo><mi>𝒱</mi><mo>,</mo><mi>ℰ</mi><mo>,</mo><mi>ℛ</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math> stores facts as triples <math alttext="(h,r,t)" display="inline"><semantics><mrow><mo stretchy="false">(</mo><mi>h</mi><mo>,</mo><mi>r</mi><mo>,</mo><mi>t</mi><mo stretchy="false">)</mo></mrow></semantics></math> where <math alttext="h,t\in\mathcal{V}" display="inline"><semantics><mrow><mrow><mi>h</mi><mo>,</mo><mi>t</mi></mrow><mo>∈</mo><mi>𝒱</mi></mrow></semantics></math> are entities and <math alttext="r\in\mathcal{R}" display="inline"><semantics><mrow><mi>r</mi><mo>∈</mo><mi>ℛ</mi></mrow></semantics></math> is a relation. Agents can query via SPARQL [134], Cypher [97], or natural-language-to-graph translation.

New observations are parsed by an extraction model <math alttext="\text{IE}:\text{text}\to\{(h_{i},r_{i},t_{i})\}" display="inline"><semantics><mrow><mtext>IE</mtext><mo lspace="0.278em" rspace="0.278em">:</mo><mrow><mtext>text</mtext><mo stretchy="false">→</mo><mrow><mo stretchy="false">{</mo><mrow><mo stretchy="false">(</mo><msub><mi>h</mi><mi>i</mi></msub><mo>,</mo><msub><mi>r</mi><mi>i</mi></msub><mo>,</mo><msub><mi>t</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow><mo stretchy="false">}</mo></mrow></mrow></mrow></semantics></math> and merged into <math alttext="\mathcal{G}" display="inline"><semantics><mi>𝒢</mi></semantics></math>. Coreference resolution and entity linking ensure consistency.

GraphRAG [82] augments RAG with graph traversal: given a query, retrieve seed entities, then expand via <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math>-hop neighborhood traversal to surface related facts not directly matched by embedding similarity. This is particularly powerful for multi-hop reasoning:

<div class="hh-equation" id="ch17.ex5"><math alttext="\text{GraphRetrieve}(q,\mathcal{G},k)=\bigcup_{v\in\text{seeds}(q)}\mathcal{N}_{k}(v,\mathcal{G})," display="block"><semantics><mrow><mrow><mrow><mtext>GraphRetrieve</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>q</mi><mo>,</mo><mi>𝒢</mi><mo>,</mo><mi>k</mi><mo stretchy="false">)</mo></mrow></mrow><mo rspace="0.111em">=</mo><mrow><munder><mo movablelimits="false">⋃</mo><mrow><mi>v</mi><mo>∈</mo><mrow><mtext>seeds</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow></mrow></munder><mrow><msub><mi>𝒩</mi><mi>k</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>v</mi><mo>,</mo><mi>𝒢</mi><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow><mo>,</mo></mrow></semantics></math></div>

where <math alttext="\mathcal{N}_{k}(v,\mathcal{G})" display="inline"><semantics><mrow><msub><mi>𝒩</mi><mi>k</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>v</mi><mo>,</mo><mi>𝒢</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math> is the <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math>-hop neighborhood of <math alttext="v" display="inline"><semantics><mi>v</mi></semantics></math>.

Facts have validity intervals: <math alttext="(h,r,t,[t_{\text{start}},t_{\text{end}}])" display="inline"><semantics><mrow><mo stretchy="false">(</mo><mi>h</mi><mo>,</mo><mi>r</mi><mo>,</mo><mi>t</mi><mo>,</mo><mrow><mo stretchy="false">[</mo><mrow><msub><mi>t</mi><mtext>start</mtext></msub><mo>,</mo><msub><mi>t</mi><mtext>end</mtext></msub></mrow><mo stretchy="false">]</mo></mrow><mo stretchy="false">)</mo></mrow></semantics></math>. Temporal KGs [197] enable queries like “Who was the CEO of OpenAI in 2023?” without conflating past and present states.

### Key-Value Memory Networks

Differentiable memory networks [377, 343] represent memory as a set of key-value pairs <math alttext="\{(\mathbf{k}_{i},\mathbf{v}_{i})\}_{i=1}^{M}" display="inline"><semantics><msubsup><mrow><mo stretchy="false">{</mo><mrow><mo stretchy="false">(</mo><msub><mi>𝐤</mi><mi>i</mi></msub><mo>,</mo><msub><mi>𝐯</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow><mo stretchy="false">}</mo></mrow><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>M</mi></msubsup></semantics></math> with soft attention-based retrieval:

<div class="hh-equation" id="ch17.ex6"><math alttext="\alpha_{i}=\text{softmax}\!\left(\frac{\mathbf{q}^{\top}\mathbf{k}_{i}}{\sqrt{D}}\right),\qquad\mathbf{c}=\sum_{i=1}^{M}\alpha_{i}\mathbf{v}_{i}." display="block"><semantics><mrow><mrow><mrow><msub><mi>α</mi><mi>i</mi></msub><mo>=</mo><mrow><mpadded width="3.720em"><mtext>softmax</mtext></mpadded><mo lspace="0em" rspace="0em">​</mo><mrow><mo>(</mo><mfrac><mrow><msup><mi>𝐪</mi><mo>⊤</mo></msup><mo lspace="0em" rspace="0em">​</mo><msub><mi>𝐤</mi><mi>i</mi></msub></mrow><msqrt><mi>D</mi></msqrt></mfrac><mo>)</mo></mrow></mrow></mrow><mo rspace="2.167em">,</mo><mrow><mi>𝐜</mi><mo rspace="0.111em">=</mo><mrow><munderover><mo movablelimits="false">∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>M</mi></munderover><mrow><msub><mi>α</mi><mi>i</mi></msub><mo lspace="0em" rspace="0em">​</mo><msub><mi>𝐯</mi><mi>i</mi></msub></mrow></mrow></mrow></mrow><mo lspace="0em">.</mo></mrow></semantics></math></div>

The retrieved context <math alttext="\mathbf{c}" display="inline"><semantics><mi>𝐜</mi></semantics></math> is a differentiable function of the query, enabling end-to-end training. Modern transformer attention is a special case of this mechanism. For agentic use, memory slots can be updated via gradient descent or via explicit write operations.

### MemGPT and Virtual Context Management

MemGPT [281] introduces a <em>virtual context</em> abstraction analogous to virtual memory in operating systems. Memory is organized in tiers:

The agent decides <em>which</em> memory to promote to hot context (page-in) and <em>which</em> to evict (page-out) based on:

<ul><li>Recency: recently accessed items are more likely to be needed.</li><li>Relevance: items with high similarity to the current query.</li><li>Importance: items tagged as high-importance during write.</li></ul>

In MemGPT, the LLM itself issues memory management function calls (<code>memory_search</code>, <code>memory_insert</code>, <code>memory_delete</code>) as part of its action space. This makes memory management a <em>learned behavior</em> rather than a hard-coded policy—a natural target for RL training (Section 17.7 Training Memory Systems with Reinforcement Learning).

## Memory Operations

### Write: Committing to Memory

Not every observation should be stored. The write decision is a filtering problem:

<div class="hh-equation" id="ch17.ex7"><math alttext="\text{Write}(e)=\mathbf{1}\!\left[\text{importance}(e)&gt;\tau\right]," display="block"><semantics><mrow><mtext>Write</mtext><mrow><mo stretchy="false">(</mo><mi>e</mi><mo stretchy="false">)</mo></mrow><mo rspace="0.108em">=</mo><mrow><mo>[</mo><mtext>importance</mtext><mrow><mo stretchy="false">(</mo><mi>e</mi><mo stretchy="false">)</mo></mrow><mo>&gt;</mo><mi>τ</mi><mo>]</mo></mrow><mo>,</mo></mrow></semantics></math></div>

where <math alttext="\tau" display="inline"><semantics><mi>τ</mi></semantics></math> is a threshold and <math alttext="\text{importance}(e)" display="inline"><semantics><mrow><mtext>importance</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>e</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math> can be:

<ul><li>Surprise:<math alttext="-\log p_{\theta}(e\mid\text{context})" display="inline"><semantics><mrow><mo rspace="0.167em">−</mo><mrow><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>p</mi><mi>θ</mi></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>e</mi><mo fence="true" lspace="0em" rspace="0em">∣</mo><mtext>context</mtext><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math>—unexpected events are more informative.</li><li>Reward signal: events associated with high <math alttext="|r_{t}|" display="inline"><semantics><mrow><mo stretchy="false">|</mo><msub><mi>r</mi><mi>t</mi></msub><mo stretchy="false">|</mo></mrow></semantics></math> (positive or negative) are worth remembering.</li><li>LLM self-assessment: prompt the model to rate importance on a 1–10 scale.</li></ul>

Before writing a new fact <math alttext="f_{\text{new}}" display="inline"><semantics><msub><mi>f</mi><mtext>new</mtext></msub></semantics></math>, check for conflicts with existing memory:

<div class="hh-equation" id="ch17.ex8"><math alttext="\text{Conflict}(f_{\text{new}},\mathcal{M})=\exists\,f\in\mathcal{M}:\text{Contradicts}(f_{\text{new}},f)." display="block"><semantics><mrow><mrow><mrow><mtext>Conflict</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>f</mi><mtext>new</mtext></msub><mo>,</mo><mi>ℳ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mo rspace="0.337em">∃</mo><mi>f</mi></mrow><mo>∈</mo><mi>ℳ</mi><mo lspace="0.278em" rspace="0.278em">:</mo><mrow><mtext>Contradicts</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>f</mi><mtext>new</mtext></msub><mo>,</mo><mi>f</mi><mo stretchy="false">)</mo></mrow></mrow></mrow><mo lspace="0em">.</mo></mrow></semantics></math></div>

Contradiction detection can be implemented via NLI models or by prompting the LLM. On conflict, the agent must decide: overwrite, keep both with timestamps, or flag for human review.

Beyond <em>what</em> to store, the <em>how</em> matters greatly. Memory entries range from atomic facts to verbose transcripts, with distinct trade-offs:

<div class="hh-table" id="ch17.t1"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Format</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Pros</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Cons</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left"><span>Atomic facts</span></th><td class="ltx_align_top"></td><td class="ltx_align_top"></td></tr><tr><th class="ltx_align_left">“User prefers Python.”</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Precise retrieval; composable; easy deduplication and contradiction detection</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Loses context; extraction errors; brittle for nuanced information</span></span></td></tr><tr><th class="ltx_align_left"><span>Structured notes</span></th><td class="ltx_align_top"></td><td class="ltx_align_top"></td></tr><tr><th class="ltx_align_left">(A-MEM <span>[400]</span>)</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Rich metadata (tags, links); supports graph traversal; balances precision and context</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Higher write cost; schema design required</span></span></td></tr><tr><th class="ltx_align_left"><span>Summarized episodes</span></th><td class="ltx_align_top"></td><td class="ltx_align_top"></td></tr><tr><th class="ltx_align_left">(MemGPT <span>[281]</span>)</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Preserves narrative coherence; compact; good for multi-turn reasoning</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Summarization lossy; hard to update partially</span></span></td></tr><tr><th class="ltx_align_left"><span>Verbatim transcripts</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Lossless; no extraction errors; supports exact quotation</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Large storage; noisy retrieval; expensive to scan</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 17.1:</span>Memory granularity trade-offs.</p></div>

In practice, production systems often combine granularities [53]: extract atomic facts for precise recall, maintain summarized episodes for narrative context, and archive verbatim transcripts in cold storage for auditability. The Generative Agents architecture [285] stores observations as atomic “memory objects” with natural-language descriptions, importance scores, and timestamps—enabling both precise retrieval and temporal reasoning.

<ul><li>Match granularity to query type. If users ask factoid questions (“What’s my API key?”), atomic facts win. If they ask contextual questions (“Why did we decide to use Redis?”), episode summaries are needed.</li><li>Store at the finest grain you can afford, then build coarser views on top. It is easy to summarize atomic facts; it is impossible to recover atoms from a lossy summary.</li><li>Include provenance. Every memory entry should link back to its source (conversation turn, document, tool output) so the agent can verify and the user can audit.</li></ul>

### Read / Retrieve

The retrieval query <math alttext="q" display="inline"><semantics><mi>q</mi></semantics></math> need not be the raw observation. Better strategies:

<ul><li>HyDE (Hypothetical Document Embeddings) [106]: generate a hypothetical answer, embed it, and use that embedding as the query.</li><li>Query expansion: generate multiple paraphrases of the query and take the union of retrieved results.</li><li>Step-back prompting: abstract the specific query to a more general question before retrieval.</li></ul>

Older memories may be less relevant. A time-weighted score:

<div class="hh-equation" id="ch17.ex9"><math alttext="\text{score}(d,q,t)=\lambda\cdot\text{sim}(\mathbf{q},\mathbf{v}_{d})+(1-\lambda)\cdot\exp\!\left(-\frac{t-t_{d}}{\tau_{\text{decay}}}\right)," display="block"><semantics><mrow><mrow><mrow><mtext>score</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>d</mi><mo>,</mo><mi>q</mi><mo>,</mo><mi>t</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mrow><mrow><mi>λ</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><mtext>sim</mtext></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>𝐪</mi><mo>,</mo><msub><mi>𝐯</mi><mi>d</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>+</mo><mrow><mrow><mo stretchy="false">(</mo><mrow><mn>1</mn><mo>−</mo><mi>λ</mi></mrow><mo rspace="0.055em" stretchy="false">)</mo></mrow><mo rspace="0.222em">⋅</mo><mrow><mpadded width="1.370em"><mi>exp</mi></mpadded><mo>⁡</mo><mrow><mo>(</mo><mrow><mo>−</mo><mfrac><mrow><mi>t</mi><mo>−</mo><msub><mi>t</mi><mi>d</mi></msub></mrow><msub><mi>τ</mi><mtext>decay</mtext></msub></mfrac></mrow><mo>)</mo></mrow></mrow></mrow></mrow></mrow><mo>,</mo></mrow></semantics></math></div>

where <math alttext="t_{d}" display="inline"><semantics><msub><mi>t</mi><mi>d</mi></msub></semantics></math> is the memory’s creation time and <math alttext="\tau_{\text{decay}}" display="inline"><semantics><msub><mi>τ</mi><mtext>decay</mtext></msub></semantics></math> controls the decay rate. The Generative Agents paper [285] uses a similar recency-weighted retrieval.

### Update: Conflict Resolution and Consolidation

Memory consolidation merges related memories to reduce redundancy and surface higher-level patterns:

<div class="hh-equation" id="ch17.ex10"><math alttext="\mathcal{M}^{\prime}=\text{Consolidate}(\mathcal{M})=\text{Cluster}(\mathcal{M})\cup\text{Summarize}(\text{Cluster}(\mathcal{M}))." display="block"><semantics><mrow><mrow><msup><mi>ℳ</mi><mo>′</mo></msup><mo>=</mo><mrow><mtext>Consolidate</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>ℳ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mrow><mtext>Cluster</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>ℳ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>∪</mo><mrow><mtext>Summarize</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mrow><mtext>Cluster</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>ℳ</mi><mo stretchy="false">)</mo></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow><mo lspace="0em">.</mo></mrow></semantics></math></div>

Biological memory forgets; so should artificial memory. Strategies:

<ul><li>LRU eviction: remove least-recently-used entries when capacity is exceeded.</li><li>Importance-weighted forgetting:<math alttext="p(\text{forget}\,|\,d)\propto\exp(-\text{importance}(d))" display="inline"><semantics><mrow><mrow><mi>p</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mtext>forget</mtext><mo lspace="0.170em" rspace="0.170em">|</mo><mi>d</mi><mo stretchy="false">)</mo></mrow></mrow><mo>∝</mo><mrow><mi>exp</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mo>−</mo><mrow><mtext>importance</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>d</mi><mo stretchy="false">)</mo></mrow></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math>.</li><li>Spaced repetition: memories accessed repeatedly are retained longer, following the exponential forgetting curve [81].</li></ul>

### Reflect: Meta-Cognitive Operations

Reflection [285, 326] is a higher-order memory operation: the agent reads its own memory and generates <em>insights</em>:

<div class="hh-equation" id="ch17.ex11"><math alttext="\text{Reflect}(\mathcal{M})\to\{i_{1},i_{2},\ldots\}\subset\mathcal{M}_{\text{semantic}}," display="block"><semantics><mrow><mrow><mrow><mtext>Reflect</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>ℳ</mi><mo stretchy="false">)</mo></mrow></mrow><mo stretchy="false">→</mo><mrow><mo stretchy="false">{</mo><msub><mi>i</mi><mn>1</mn></msub><mo>,</mo><msub><mi>i</mi><mn>2</mn></msub><mo>,</mo><mi mathvariant="normal">…</mi><mo stretchy="false">}</mo></mrow><mo>⊂</mo><msub><mi>ℳ</mi><mtext>semantic</mtext></msub></mrow><mo>,</mo></mrow></semantics></math></div>

where each insight <math alttext="i_{j}" display="inline"><semantics><msub><mi>i</mi><mi>j</mi></msub></semantics></math> is a higher-level abstraction derived from multiple episodic memories.

Reflection <em>reads</em> from episodic memory but <em>writes</em> to semantic memory. The resulting insights are context-independent generalizations (“always check for empty inputs”), not episode-specific records—hence they belong in semantic memory <math alttext="\mathcal{M}_{\text{semantic}}" display="inline"><semantics><msub><mi>ℳ</mi><mtext>semantic</mtext></msub></semantics></math>. However, during the reflection process itself, the intermediate reasoning (retrieved episodes + synthesis prompt + generated insight) occupies <em>working memory</em> (the context window). In short:

<ul><li>Input: episodic memory (specific past events)</li><li>Computation: working memory (active reasoning in context)</li><li>Output: semantic memory (durable, generalized insight)</li></ul>

This mirrors biological memory consolidation, where episodic experiences are gradually transformed into semantic knowledge during sleep and reflection.

## Memory for Multi-Turn Conversations

### User Modeling and Preference Tracking

A persistent user model <math alttext="\mathcal{U}" display="inline"><semantics><mi>𝒰</mi></semantics></math> stores:

<ul><li>Explicit preferences: stated likes/dislikes, communication style preferences.</li><li>Implicit preferences: inferred from behavior (e.g. user always asks for code in Python, prefers concise answers).</li><li>Expertise level: domain knowledge inferred from vocabulary and question complexity.</li><li>Goals and context: ongoing projects, current tasks, organizational role.</li></ul>

The user model is updated after each interaction:

<div class="hh-equation" id="ch17.ex12"><math alttext="\mathcal{U}_{t+1}=\text{Update}(\mathcal{U}_{t},\,(u_{t},a_{t},\text{feedback}_{t}))." display="block"><semantics><mrow><mrow><msub><mi>𝒰</mi><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>=</mo><mrow><mtext>Update</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>𝒰</mi><mi>t</mi></msub><mo>,</mo><mrow><mo stretchy="false">(</mo><msub><mi>u</mi><mi>t</mi></msub><mo>,</mo><msub><mi>a</mi><mi>t</mi></msub><mo>,</mo><msub><mtext>feedback</mtext><mi>t</mi></msub><mo stretchy="false">)</mo></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow><mo lspace="0em">.</mo></mrow></semantics></math></div>

### Session Continuity

Without memory, each conversation starts cold. With session memory:

<ol><li>At session start, retrieve the user model <math alttext="\mathcal{U}" display="inline"><semantics><mi>𝒰</mi></semantics></math> and recent session summaries.</li><li>Inject a personalized system prompt: “You are helping Alice, a senior ML engineer working on a distributed training project. Last session you helped debug a gradient synchronization issue.”</li><li>At session end, summarize the session and update <math alttext="\mathcal{U}" display="inline"><semantics><mi>𝒰</mi></semantics></math>.</li></ol>

### Personalization Through Memory

Personalization improves both <em>efficiency</em> (fewer clarifying questions) and <em>quality</em> (responses calibrated to user expertise). Key techniques:

<ul><li>Adaptive verbosity: adjust response length based on user’s historical engagement.</li><li>Domain priming: prepend relevant domain context from semantic memory.</li><li>Proactive recall: surface relevant past interactions without being asked (“You asked about this topic last month; here’s what we found then”).</li></ul>

## Memory for Multi-Agent Systems

When multiple agents collaborate on a shared task, memory becomes a <em>coordination mechanism</em>—not just a personal knowledge store. A planning agent that decomposes a task must communicate sub-goals to executor agents; a critic agent must access the same context as the agent it evaluates; a research team of agents must avoid duplicating work. Without shared memory, agents must communicate everything through direct messages, creating bandwidth bottlenecks and losing information when conversations scroll out of context. Shared memory solves this by providing a persistent, queryable substrate that all agents can read from and write to—turning implicit coordination (“I hope the other agent remembers”) into explicit state (“the answer is on the blackboard”).

### Shared Memory Pools

In multi-agent systems, agents may share a common memory store <math alttext="\mathcal{M}_{\text{shared}}" display="inline"><semantics><msub><mi>ℳ</mi><mtext>shared</mtext></msub></semantics></math> alongside private stores <math alttext="\mathcal{M}_{i}" display="inline"><semantics><msub><mi>ℳ</mi><mi>i</mi></msub></semantics></math>:

<div class="hh-equation" id="ch17.ex13"><math alttext="\text{context}_{i}(t)=\mathcal{R}(\mathcal{M}_{i},q_{i})\cup\mathcal{R}(\mathcal{M}_{\text{shared}},q_{i})." display="block"><semantics><mrow><mrow><mrow><msub><mtext>context</mtext><mi>i</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mrow><mi>ℛ</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>ℳ</mi><mi>i</mi></msub><mo>,</mo><msub><mi>q</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>∪</mo><mrow><mi>ℛ</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>ℳ</mi><mtext>shared</mtext></msub><mo>,</mo><msub><mi>q</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow><mo lspace="0em">.</mo></mrow></semantics></math></div>

Shared memory enables <em>implicit coordination</em>: agent <math alttext="A" display="inline"><semantics><mi>A</mi></semantics></math> writes a finding; agent <math alttext="B" display="inline"><semantics><mi>B</mi></semantics></math> retrieves it without explicit communication.

### Blackboard Architecture

The <em>blackboard</em> pattern [135] is a classic multi-agent coordination mechanism:

Each agent reads from and writes to the blackboard. A <em>controller</em> monitors the blackboard and activates agents when their preconditions are met. This decouples agents: they communicate through shared state rather than direct messaging.

### Consensus and Conflict in Shared Knowledge

When multiple agents write to shared memory, conflicts arise. Resolution strategies:

<ul><li>Last-write-wins: simple but loses information.</li><li>Versioned memory: maintain a history of all writes; agents can query any version.</li><li>Voting / consensus: require <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math>-of-<math alttext="n" display="inline"><semantics><mi>n</mi></semantics></math> agents to agree before a fact is committed.</li><li>Confidence-weighted merging:<math alttext="f_{\text{merged}}=\sum_{i}w_{i}f_{i}" display="inline"><semantics><mrow><msub><mi>f</mi><mtext>merged</mtext></msub><mo rspace="0.111em">=</mo><mrow><msub><mo>∑</mo><mi>i</mi></msub><mrow><msub><mi>w</mi><mi>i</mi></msub><mo lspace="0em" rspace="0em">​</mo><msub><mi>f</mi><mi>i</mi></msub></mrow></mrow></mrow></semantics></math> where <math alttext="w_{i}" display="inline"><semantics><msub><mi>w</mi><mi>i</mi></msub></semantics></math> is agent <math alttext="i" display="inline"><semantics><mi>i</mi></semantics></math>’s confidence.</li><li>Designated authority: assign ownership of memory regions to specific agents.</li></ul>

## Training Memory Systems with Reinforcement Learning

### Reward Signals for Memory Operations

Memory operations (read, write, update, reflect) can be treated as actions in the RL framework. The challenge is designing reward signals that incentivize <em>useful</em> memory behavior:

<ul><li>Task reward propagation. If a memory retrieval leads to a correct answer, credit the retrieval action. Sparse but unambiguous.</li><li>Retrieval precision reward.<math alttext="r_{\text{retrieve}}=\text{Relevance}(d_{\text{retrieved}},\text{task})" display="inline"><semantics><mrow><msub><mi>r</mi><mtext>retrieve</mtext></msub><mo>=</mo><mrow><mtext>Relevance</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>d</mi><mtext>retrieved</mtext></msub><mo>,</mo><mtext>task</mtext><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math>, estimated by a learned relevance model.</li><li>Memory efficiency reward. Penalize unnecessary writes: <math alttext="r_{\text{write}}=-\lambda\cdot\mathbf{1}[\text{write}]" display="inline"><semantics><mrow><msub><mi>r</mi><mtext>write</mtext></msub><mo rspace="0em">=</mo><mo lspace="0em">−</mo><mi>λ</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><mn>𝟏</mn><mrow><mo stretchy="false">[</mo><mtext>write</mtext><mo stretchy="false">]</mo></mrow></mrow></semantics></math>, encouraging selective storage.</li><li>Consistency reward. Reward memory states that are internally consistent (no contradictions).</li></ul>

The combined reward for a memory operation <math alttext="m_{t}" display="inline"><semantics><msub><mi>m</mi><mi>t</mi></msub></semantics></math> at step <math alttext="t" display="inline"><semantics><mi>t</mi></semantics></math>:

<div class="hh-equation" id="ch17.ex14"><math alttext="r_{t}^{\text{mem}}=r_{t}^{\text{task}}+\alpha\cdot r_{t}^{\text{retrieve}}+\beta\cdot r_{t}^{\text{write}}+\gamma\cdot r_{t}^{\text{consistency}}." display="block"><semantics><mrow><mrow><msubsup><mi>r</mi><mi>t</mi><mtext>mem</mtext></msubsup><mo>=</mo><mrow><msubsup><mi>r</mi><mi>t</mi><mtext>task</mtext></msubsup><mo>+</mo><mrow><mi>α</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><msubsup><mi>r</mi><mi>t</mi><mtext>retrieve</mtext></msubsup></mrow><mo>+</mo><mrow><mi>β</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><msubsup><mi>r</mi><mi>t</mi><mtext>write</mtext></msubsup></mrow><mo>+</mo><mrow><mi>γ</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><msubsup><mi>r</mi><mi>t</mi><mtext>consistency</mtext></msubsup></mrow></mrow></mrow><mo lspace="0em">.</mo></mrow></semantics></math></div>

### Learning What to Remember

The <em>what-to-remember</em> problem is a meta-learning challenge: the agent must learn a write policy <math alttext="\pi_{\text{write}}(e)" display="inline"><semantics><mrow><msub><mi>π</mi><mtext>write</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>e</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math> that maximizes future task performance. This is difficult because:

<ol><li>The value of a memory is only revealed in the future (delayed reward).</li><li>The space of possible future queries is unknown at write time.</li><li>Memories interact: the value of storing <math alttext="e" display="inline"><semantics><mi>e</mi></semantics></math> depends on what else is in <math alttext="\mathcal{M}" display="inline"><semantics><mi>ℳ</mi></semantics></math>.</li></ol>

Approaches:

<ul><li>Hindsight relabeling[8]. After a successful episode, retroactively label the memories that were retrieved as “important” and train the write policy to store similar items.</li><li>Meta-RL[79]. Train the write policy across a distribution of tasks; the policy learns to store information that generalizes across tasks.</li><li>Curiosity-driven storage[286]. Store observations that are surprising (high prediction error), as these are likely to be informative.</li></ul>

### Memory-Augmented Policy Optimization

The idea of jointly optimizing a policy and its memory system dates to differentiable memory networks [119] and was extended to retrieval-augmented LLMs by REALM [128]. The full policy gradient objective for a memory-augmented agent:

<div class="hh-equation" id="ch17.ex15"><math alttext="\mathcal{L}(\theta,\phi)=\mathbb{E}_{\tau\sim\pi_{\theta}}\!\left[\sum_{t=0}^{T}\gamma^{t}r_{t}\right]-\lambda\cdot\mathcal{L}_{\text{mem}}(\phi)," display="block"><semantics><mrow><mrow><mrow><mi>ℒ</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo>,</mo><mi>ϕ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mrow><msub><mi>𝔼</mi><mrow><mi>τ</mi><mo>∼</mo><msub><mi>π</mi><mi>θ</mi></msub></mrow></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo>[</mo><mrow><munderover><mo lspace="0em" movablelimits="false">∑</mo><mrow><mi>t</mi><mo>=</mo><mn>0</mn></mrow><mi>T</mi></munderover><mrow><msup><mi>γ</mi><mi>t</mi></msup><mo lspace="0em" rspace="0em">​</mo><msub><mi>r</mi><mi>t</mi></msub></mrow></mrow><mo>]</mo></mrow></mrow><mo>−</mo><mrow><mrow><mi>λ</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><msub><mi>ℒ</mi><mtext>mem</mtext></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>ϕ</mi><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow><mo>,</mo></mrow></semantics></math></div>

where <math alttext="\theta" display="inline"><semantics><mi>θ</mi></semantics></math> are the LLM parameters, <math alttext="\phi" display="inline"><semantics><mi>ϕ</mi></semantics></math> are the memory system parameters (e.g. retrieval model weights), and <math alttext="\mathcal{L}_{\text{mem}}" display="inline"><semantics><msub><mi>ℒ</mi><mtext>mem</mtext></msub></semantics></math> is a regularization term on memory complexity.

## Comparison of Memory Approaches

<div class="hh-table" id="ch17.t2"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Architecture</span></th><th class="ltx_align_left"><span>Capacity</span></th><th class="ltx_align_left"><span>Retrieval</span></th><th class="ltx_align_left"><span>Update Cost</span></th><th class="ltx_align_left"><span>Trainable</span></th><th class="ltx_align_left"><span>Best For</span></th></tr></thead><tbody><tr><td class="ltx_align_left">In-context (working)</td><td class="ltx_align_left"><math alttext="O(L)" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>L</mi><mo stretchy="false">)</mo></mrow></mrow><span></span></semantics></math> tokens</td><td class="ltx_align_left">0 ms</td><td class="ltx_align_left">Free</td><td class="ltx_align_left">Via fine-tuning</td><td class="ltx_align_left">Short tasks, active reasoning</td></tr><tr><td class="ltx_align_left">Dense RAG <span>[209]</span></td><td class="ltx_align_left"><math alttext="O(10^{7})" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msup><mn>10</mn><mn>7</mn></msup><mo stretchy="false">)</mo></mrow></mrow><span></span></semantics></math> docs</td><td class="ltx_align_left">10–50 ms</td><td class="ltx_align_left"><math alttext="O(1)" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></mrow><span></span></semantics></math> embed</td><td class="ltx_align_left">Encoder only</td><td class="ltx_align_left">Semantic search, QA</td></tr><tr><td class="ltx_align_left">Sparse (BM25) <span>[309]</span></td><td class="ltx_align_left"><math alttext="O(10^{8})" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msup><mn>10</mn><mn>8</mn></msup><mo stretchy="false">)</mo></mrow></mrow><span></span></semantics></math> docs</td><td class="ltx_align_left">1–5 ms</td><td class="ltx_align_left"><math alttext="O(|d|)" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mo stretchy="false">|</mo><mi>d</mi><mo stretchy="false">|</mo></mrow><mo stretchy="false">)</mo></mrow></mrow><span></span></semantics></math> index</td><td class="ltx_align_left">No</td><td class="ltx_align_left">Keyword search, legal/medical</td></tr><tr><td class="ltx_align_left">Hybrid RAG <span>[44]</span></td><td class="ltx_align_left"><math alttext="O(10^{7})" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msup><mn>10</mn><mn>7</mn></msup><mo stretchy="false">)</mo></mrow></mrow><span></span></semantics></math> docs</td><td class="ltx_align_left">15–60 ms</td><td class="ltx_align_left"><math alttext="O(1)" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></mrow><span></span></semantics></math> embed</td><td class="ltx_align_left">Encoder only</td><td class="ltx_align_left">General-purpose retrieval</td></tr><tr><td class="ltx_align_left">Summarization</td><td class="ltx_align_left">Unlimited</td><td class="ltx_align_left">0 ms (in-ctx)</td><td class="ltx_align_left"><math alttext="O(|e|)" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mo stretchy="false">|</mo><mi>e</mi><mo stretchy="false">|</mo></mrow><mo stretchy="false">)</mo></mrow></mrow><span></span></semantics></math> LLM call</td><td class="ltx_align_left">Via fine-tuning</td><td class="ltx_align_left">Long conversations, narratives</td></tr><tr><td class="ltx_align_left">Knowledge Graph <span>[197]</span></td><td class="ltx_align_left"><math alttext="O(10^{9})" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msup><mn>10</mn><mn>9</mn></msup><mo stretchy="false">)</mo></mrow></mrow><span></span></semantics></math> triples</td><td class="ltx_align_left">5–100 ms</td><td class="ltx_align_left"><math alttext="O(1)" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></mrow><span></span></semantics></math> insert</td><td class="ltx_align_left">Embedding layer</td><td class="ltx_align_left">Structured facts, multi-hop</td></tr><tr><td class="ltx_align_left">KV Memory Net <span>[343]</span></td><td class="ltx_align_left"><math alttext="O(M)" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>M</mi><mo stretchy="false">)</mo></mrow></mrow><span></span></semantics></math> slots</td><td class="ltx_align_left"><math alttext="O(M)" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>M</mi><mo stretchy="false">)</mo></mrow></mrow><span></span></semantics></math> attn</td><td class="ltx_align_left">Gradient step</td><td class="ltx_align_left">Fully</td><td class="ltx_align_left">End-to-end differentiable tasks</td></tr><tr><td class="ltx_align_left">MemGPT tiered <span>[281]</span></td><td class="ltx_align_left">Unlimited</td><td class="ltx_align_left">0–100 ms</td><td class="ltx_align_left">Mixed</td><td class="ltx_align_left">Via RL</td><td class="ltx_align_left">Long-horizon agents, assistants</td></tr><tr><td class="ltx_align_left">Graph RAG <span>[82]</span></td><td class="ltx_align_left"><math alttext="O(10^{7})" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msup><mn>10</mn><mn>7</mn></msup><mo stretchy="false">)</mo></mrow></mrow><span></span></semantics></math> nodes</td><td class="ltx_align_left">20–200 ms</td><td class="ltx_align_left"><math alttext="O(1)" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></mrow><span></span></semantics></math> insert</td><td class="ltx_align_left">Encoder only</td><td class="ltx_align_left">Complex reasoning, communities</td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 17.2:</span>Comparison of agentic memory architectures across key dimensions.</p></div>

## Evaluating Memory Systems

Evaluating agentic memory is challenging because the quality of memory operations is only revealed <em>indirectly</em>—through downstream task performance over long horizons. A memory system that achieves perfect recall of stored facts can still fail if it retrieves irrelevant context or overwhelms the LLM’s context window.

### Evaluation Dimensions

LongMemEval [384] identifies five core capabilities that a long-term memory system must demonstrate:

<ol><li>Information extraction. Can the system identify and store salient facts from conversational turns? Measured by fact recall: what fraction of ground-truth facts are recoverable from memory?</li><li>Multi-session reasoning. Can the system synthesize information scattered across multiple past sessions? E.g., “Based on our conversations last week and yesterday, what changed in the project scope?”</li><li>Temporal reasoning. Can the system correctly answer time-dependent queries? E.g., “What did I say was my priority <em>before</em> the reorg?” requires distinguishing temporal states.</li><li>Knowledge updates. When facts change (user moves cities, preferences shift), does memory reflect the latest state while preserving history?</li><li>Abstention. When the system has no relevant memory, does it correctly say “I don’t know” rather than hallucinate a plausible but fabricated recollection?</li></ol>

### Benchmarks

<div class="hh-table" id="ch17.t3"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Benchmark</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Venue</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Scale</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Focus</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">LongMemEval <span>[384]</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>ICLR 2025</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>500 questions, scalable histories</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Five memory abilities; multi-session chat</span></span></td></tr><tr><th class="ltx_align_left">LOCOMO <span>[247]</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>EMNLP 2024</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Multi-session dialogues</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Single-hop, temporal, multi-hop, open-domain QA over conversations</span></span></td></tr><tr><th class="ltx_align_left">InfiniteBench <span>[428]</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>ACL 2024</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>100K+ token contexts</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Long-context recall, not memory-specific but tests limits</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 17.3:</span>Benchmarks for evaluating agentic memory systems.</p></div>

### Metrics

<ul><li>Memory Recall: <math alttext="\frac{\text{\# ground-truth facts retrievable from memory}}{\text{\# total ground-truth facts}}" display="inline"><semantics><mfrac><mtext># ground-truth facts retrievable from memory</mtext><mtext># total ground-truth facts</mtext></mfrac></semantics></math>. Measures completeness of storage.</li><li>Memory Precision: <math alttext="\frac{\text{\# relevant items in top-}k\text{ retrieval}}{k}" display="inline"><semantics><mfrac><mrow><mtext># relevant items in top-</mtext><mo lspace="0em" rspace="0em">​</mo><mi>k</mi><mo lspace="0em" rspace="0em">​</mo><mtext> retrieval</mtext></mrow><mi>k</mi></mfrac></semantics></math>. Measures noise in retrieval.</li><li>Latency: time from query to retrieved context (p50 and p95).</li><li>Token efficiency: total tokens injected into context per query. Lower is better—unnecessary context degrades LLM accuracy and increases cost.</li></ul>

<ul><li>Answer accuracy: correctness of the final response conditioned on memory (EM, F1, or LLM-as-judge).</li><li>Faithfulness: does the response accurately reflect what memory contains, without fabrication?</li><li>Personalization quality: user satisfaction, measured via preference ratings or A/B tests between memory-augmented and memoryless systems.</li><li>Contradiction rate: how often the system produces responses inconsistent with previously stated facts.</li></ul>

<ul><li>Write selectivity: fraction of turns that trigger a memory write. Too high <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> noise; too low <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> gaps.</li><li>Staleness: how often outdated facts are retrieved despite an update existing.</li><li>Storage growth rate: tokens stored per interaction hour. Unbounded growth is unsustainable.</li></ul>

## Implementation Patterns

### Vector Store Memory with Embeddings

The most common memory pattern stores entries as embedding vectors alongside metadata (timestamps, importance scores, tags). Retrieval combines cosine similarity with temporal decay, so recent and important memories surface first. Duplicate detection and LRU eviction keep the store bounded.

```
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json

@dataclass
class MemoryEntry:
    """A single memory entry with metadata."""
    content: str
    embedding: np.ndarray
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 0.5
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    tags: list[str] = field(default_factory=list)
    source: str = "agent"

class VectorMemoryStore:
    """
    Hybrid dense+sparse memory store with temporal decay.
    Supports importance-weighted retrieval and LRU eviction.
    """

    def __init__(
        self,
        embed_fn,           # callable: str -> np.ndarray
        max_entries: int = 10_000,
        decay_rate: float = 0.01,   # per hour
        recency_weight: float = 0.3,
    ):
        self.embed_fn = embed_fn
        self.max_entries = max_entries
        self.decay_rate = decay_rate
        self.recency_weight = recency_weight
        self.entries: list[MemoryEntry] = []

    # -- Write --------------------------------------------------------------

    def write(
        self,
        content: str,
        importance: float = 0.5,
        tags: list[str] | None = None,
        check_duplicates: bool = True,
    ) -> MemoryEntry:
        """Commit a new memory, evicting if at capacity."""
        if check_duplicates and self._is_duplicate(content):
            return None  # Skip near-duplicate entries

        embedding = self.embed_fn(content)
        entry = MemoryEntry(
            content=content,
            embedding=embedding,
            importance=importance,
            tags=tags or [],
        )

        if len(self.entries) >= self.max_entries:
            self._evict()

        self.entries.append(entry)
        return entry

    def _is_duplicate(self, content: str, threshold: float = 0.95) -> bool:
        """Check if a near-duplicate already exists."""
        if not self.entries:
            return False
        emb = self.embed_fn(content)
        sims = self._cosine_similarities(emb)
        return float(np.max(sims)) > threshold

    def _evict(self):
        """Remove the least important + least recent entry."""
        now = datetime.now()
        scores = []
        for e in self.entries:
            age_hours = (now - e.timestamp).total_seconds() / 3600
            recency = np.exp(-self.decay_rate * age_hours)
            score = e.importance * (1 - self.recency_weight) \
                  + recency * self.recency_weight
            scores.append(score)
        worst_idx = int(np.argmin(scores))
        self.entries.pop(worst_idx)

    # -- Retrieve -----------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int = 5,
        recency_boost: bool = True,
    ) -> list[MemoryEntry]:
        """
        Hybrid retrieval: dense similarity + temporal recency.
        Returns top-k entries sorted by combined score.
        """
        if not self.entries:
            return []

        q_emb = self.embed_fn(query)
        dense_scores = self._cosine_similarities(q_emb)

        now = datetime.now()
        combined = []
        for i, (entry, d_score) in enumerate(
            zip(self.entries, dense_scores)
        ):
            if recency_boost:
                age_h = (now - entry.timestamp).total_seconds() / 3600
                recency = np.exp(-self.decay_rate * age_h)
                score = (1 - self.recency_weight) * d_score \
                      + self.recency_weight * recency
            else:
                score = d_score
            combined.append((score, i))

        combined.sort(reverse=True)
        top_k = [self.entries[i] for _, i in combined[:k]]

        # Update access metadata
        for entry in top_k:
            entry.access_count += 1
            entry.last_accessed = now

        return top_k

    def _cosine_similarities(self, query_emb: np.ndarray) -> np.ndarray:
        """Vectorized cosine similarity against all stored embeddings."""
        matrix = np.stack([e.embedding for e in self.entries])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix_norm = matrix / (norms + 1e-8)
        q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        return matrix_norm @ q_norm

    # -- Reflect ------------------------------------------------------------

    def reflect(self, llm_fn, k: int = 10) -> list[str]:
        """
        Meta-cognitive reflection: retrieve recent memories,
        synthesize higher-level insights, and store them back.
        """
        if len(self.entries) < 3:
            return []

        # Retrieve recent high-importance memories
        recent = sorted(
            self.entries, key=lambda e: e.timestamp, reverse=True
        )[:k]
        context = "\n".join(f"- {e.content}" for e in recent)

        # Ask LLM to generate insights
        prompt = (
            "Given these recent memories, extract 2-3 high-level "
            "insights or patterns:\n" + context
        )
        raw_insights = llm_fn(prompt)

        # Store each insight as a high-importance memory
        insights = []
        for line in raw_insights.strip().split("\n"):
            line = line.strip().lstrip("-*").strip()
            if len(line) > 20:
                self.write(
                    f"[INSIGHT] {line}",
                    importance=0.9,
                    check_duplicates=True,
                )
                insights.append(line)
        return insights

    def get_stats(self) -> dict:
        """Return memory statistics for monitoring."""
        return {
            "total_entries": len(self.entries),
            "avg_importance": float(
                np.mean([e.importance for e in self.entries])
            ) if self.entries else 0.0,
            "oldest_entry": min(
                (e.timestamp for e in self.entries), default=None
            ),
        }
```

<span class="hh-tag">Listing 22:</span>Vector store memory with embeddings, importance scoring, and hybrid retrieval.

### Hierarchical Memory Manager

Inspired by MemGPT [281], this pattern organises memory into three tiers: <em>hot</em> (in-context, immediate access), <em>warm</em> (vector store, fast retrieval), and <em>cold</em> (archival, unlimited capacity). Entries are automatically promoted or demoted based on access frequency and importance—analogous to CPU cache hierarchies.

```
from enum import Enum
from collections import OrderedDict

class MemoryTier(Enum):
    HOT  = "hot"    # In-context: immediate access
    WARM = "warm"   # Vector store: fast retrieval
    COLD = "cold"   # Archival: slow but unlimited

class HierarchicalMemoryManager:
    """
    Three-tier memory manager inspired by MemGPT.
    Hot tier is an LRU cache; warm is a vector store;
    cold is append-only archival storage.
    """

    def __init__(
        self,
        vector_store: VectorMemoryStore,
        hot_capacity: int = 20,     # max entries in hot tier
        warm_capacity: int = 5_000,
        llm_summarize_fn=None,      # callable for summarization
    ):
        self.vector_store = vector_store
        self.hot_capacity = hot_capacity
        self.warm_capacity = warm_capacity
        self.summarize = llm_summarize_fn

        # Hot tier: ordered dict for LRU semantics
        self.hot: OrderedDict[str, MemoryEntry] = OrderedDict()
        # Cold tier: append-only list (would be a DB in production)
        self.cold: list[MemoryEntry] = []

    # -- Page-in: promote warm -> hot ---------------------------------------

    def page_in(self, query: str, k: int = 3) -> list[MemoryEntry]:
        """
        Retrieve from warm store and promote to hot tier.
        Evicts least-recently-used hot entries if needed.
        """
        candidates = self.vector_store.retrieve(query, k=k)
        promoted = []
        for entry in candidates:
            key = entry.content[:64]  # use prefix as key
            if key not in self.hot:
                if len(self.hot) >= self.hot_capacity:
                    self._evict_hot()
                self.hot[key] = entry
                self.hot.move_to_end(key)
            promoted.append(entry)
        return promoted

    def _evict_hot(self):
        """Evict LRU entry from hot tier back to warm."""
        # OrderedDict: first item is LRU
        key, entry = self.hot.popitem(last=False)
        # Re-insert into warm store (already there, just update access)
        # In a real system, we'd update the warm store's metadata

    # -- Write with tier assignment ------------------------------------------

    def write(
        self,
        content: str,
        importance: float = 0.5,
        tier: MemoryTier = MemoryTier.WARM,
    ) -> MemoryEntry:
        """Write to the appropriate tier."""
        if tier == MemoryTier.HOT:
            entry = MemoryEntry(
                content=content,
                embedding=self.vector_store.embed_fn(content),
                importance=importance,
            )
            key = content[:64]
            if len(self.hot) >= self.hot_capacity:
                self._evict_hot()
            self.hot[key] = entry
            return entry

        elif tier == MemoryTier.WARM:
            return self.vector_store.write(content, importance=importance)

        else:  # COLD
            entry = MemoryEntry(
                content=content,
                embedding=np.array([]),  # no embedding for cold
                importance=importance,
            )
            self.cold.append(entry)
            return entry

    # -- Summarize and compress ---------------------------------------------

    def compress_hot_to_warm(self) -> Optional[str]:
        """
        Summarize hot tier contents and write summary to warm.
        Called when hot tier is full and new important content arrives.
        """
        if not self.hot or not self.summarize:
            return None

        hot_contents = "\n".join(
            f"- {e.content}" for e in self.hot.values()
        )
        summary = self.summarize(
            f"Summarize these memory entries concisely:\n{hot_contents}"
        )
        self.vector_store.write(summary, importance=0.7)
        return summary

    # -- Unified retrieval --------------------------------------------------

    def retrieve(self, query: str, k: int = 5) -> list[MemoryEntry]:
        """
        Retrieve from all tiers, prioritizing hot.
        Returns up to k entries sorted by relevance.
        """
        results = []

        # 1. Check hot tier (exact + semantic)
        q_emb = self.vector_store.embed_fn(query)
        for entry in self.hot.values():
            if entry.embedding.size > 0:
                sim = float(
                    np.dot(q_emb, entry.embedding)
                    / (np.linalg.norm(q_emb) * np.linalg.norm(entry.embedding) + 1e-8)
                )
                if sim > 0.7:
                    results.append((sim + 1.0, entry))  # +1 hot bonus

        # 2. Retrieve from warm store
        warm_results = self.vector_store.retrieve(query, k=k)
        for entry in warm_results:
            results.append((0.5, entry))

        # 3. Deduplicate and sort
        seen = set()
        final = []
        for score, entry in sorted(results, reverse=True):
            key = entry.content[:64]
            if key not in seen:
                seen.add(key)
                final.append(entry)
            if len(final) >= k:
                break

        return final

    def get_hot_context(self) -> str:
        """Return hot tier as a formatted context string."""
        if not self.hot:
            return ""
        lines = ["[Memory Context]"]
        for entry in list(self.hot.values())[-10:]:  # last 10
            lines.append(f"  - {entry.content}")
        return "\n".join(lines)
```

<span class="hh-tag">Listing 23:</span>Hierarchical memory manager implementing hot/warm/cold tiers with automatic promotion and demotion.

### Memory-Augmented Agent Loop

This pattern, introduced by MemGPT [281] and formalized in the CoALA framework [344], wires the memory system into the agent’s reasoning loop via a <em>read–act–reflect–write</em> cycle: before responding, the agent retrieves relevant memories; after responding, it decides what to store. Special tokens in the LLM output trigger memory operations, giving the model self-directed control over its own persistence.

```
import re
from typing import Any

class MemoryAugmentedAgent:
    """
    An LLM agent with a full read-act-reflect-write memory cycle.
    Implements the MemGPT-style self-directed memory management.
    """

    SYSTEM_PROMPT = """You are a memory-augmented AI assistant.
You have access to persistent memory across conversations.
At each turn you may issue memory commands:
  [MEMORY_SEARCH: <query>]  - retrieve relevant memories
  [MEMORY_WRITE: <content>] - store important information
  [MEMORY_REFLECT]          - synthesize insights from memory

Always think step by step. Use memory to avoid repeating mistakes
and to personalize your responses."""

    def __init__(
        self,
        llm_fn,                         # callable: messages -> str
        memory_manager: HierarchicalMemoryManager,
        importance_threshold: float = 0.6,
        max_memory_tokens: int = 1500,
    ):
        self.llm = llm_fn
        self.memory = memory_manager
        self.importance_threshold = importance_threshold
        self.max_memory_tokens = max_memory_tokens
        self.conversation_history: list[dict] = []

    # -- Main agent step ----------------------------------------------------

    def step(self, user_message: str) -> str:
        """
        Full agent step:
        1. Retrieve relevant memories
        2. Construct augmented prompt
        3. Generate response (possibly with memory commands)
        4. Execute memory commands
        5. Reflect and consolidate
        6. Return response to user
        """

        # Step 1: Retrieve relevant memories
        memories = self.memory.retrieve(user_message, k=5)
        memory_context = self._format_memories(memories)

        # Step 2: Construct augmented prompt
        messages = self._build_messages(user_message, memory_context)

        # Step 3: Generate response
        raw_response = self.llm(messages)

        # Step 4: Execute any memory commands in the response
        clean_response, memory_ops = self._parse_memory_commands(
            raw_response
        )
        self._execute_memory_ops(memory_ops, user_message, clean_response)

        # Step 5: Auto-write important information
        self._auto_write(user_message, clean_response)

        # Step 6: Update conversation history
        self.conversation_history.append(
            {"role": "user", "content": user_message}
        )
        self.conversation_history.append(
            {"role": "assistant", "content": clean_response}
        )

        return clean_response

    # -- Memory retrieval and formatting -----------------------------------

    def _format_memories(self, memories: list[MemoryEntry]) -> str:
        if not memories:
            return ""
        lines = ["Relevant memories:"]
        for i, m in enumerate(memories, 1):
            age = (datetime.now() - m.timestamp).days
            lines.append(
                f"  [{i}] (importance={m.importance:.1f}, "
                f"{age}d ago) {m.content}"
            )
        return "\n".join(lines)

    def _build_messages(
        self, user_message: str, memory_context: str
    ) -> list[dict]:
        system = self.SYSTEM_PROMPT
        if memory_context:
            system += f"\n\n{memory_context}"
        system += f"\n\n{self.memory.get_hot_context()}"

        messages = [{"role": "system", "content": system}]
        # Include recent conversation history (last 6 turns)
        messages.extend(self.conversation_history[-6:])
        messages.append({"role": "user", "content": user_message})
        return messages

    # -- Memory command parsing ---------------------------------------------

    def _parse_memory_commands(
        self, response: str
    ) -> tuple[str, list[dict]]:
        """Extract and remove memory commands from response."""
        ops = []
        patterns = {
            "search":  r"\[MEMORY_SEARCH:\s*(.+?)\]",
            "write":   r"\[MEMORY_WRITE:\s*(.+?)\]",
            "reflect": r"\[MEMORY_REFLECT\]",
        }
        clean = response
        for op_type, pattern in patterns.items():
            for match in re.finditer(pattern, response, re.DOTALL):
                content = match.group(1) if op_type != "reflect" else None
                ops.append({"type": op_type, "content": content})
                clean = clean.replace(match.group(0), "").strip()
        return clean, ops

    def _execute_memory_ops(
        self,
        ops: list[dict],
        user_msg: str,
        response: str,
    ):
        """Execute memory commands issued by the LLM."""
        for op in ops:
            if op["type"] == "search":
                results = self.memory.retrieve(op["content"], k=3)
                # Page results into hot tier for immediate use
                self.memory.page_in(op["content"], k=3)

            elif op["type"] == "write":
                self.memory.write(
                    op["content"],
                    importance=0.8,  # explicitly written = important
                    tier=MemoryTier.WARM,
                )

            elif op["type"] == "reflect":
                self._reflect()

    # -- Auto-write heuristic -----------------------------------------------

    def _auto_write(self, user_msg: str, response: str):
        """
        Automatically store important information without explicit command.
        Uses a simple heuristic: write if response contains facts,
        decisions, or user preferences.
        """
        importance_keywords = [
            "remember", "important", "note that", "you prefer",
            "your name is", "decided to", "the answer is",
            "key insight", "learned that",
        ]
        combined = (user_msg + " " + response).lower()
        if any(kw in combined for kw in importance_keywords):
            summary = f"User: {user_msg[:100]} | Agent: {response[:200]}"
            self.memory.write(
                summary,
                importance=self.importance_threshold,
                tier=MemoryTier.WARM,
            )

    # -- Reflection --------------------------------------------------------

    def _reflect(self):
        """
        Meta-cognitive reflection: synthesize insights from recent memory.
        Stores high-level insights back into semantic memory.
        """
        recent = self.memory.retrieve("recent important events", k=10)
        if len(recent) < 3:
            return  # Not enough to reflect on

        recent_text = "\n".join(f"- {m.content}" for m in recent)
        insight_prompt = [
            {"role": "system", "content": "You extract high-level insights."},
            {"role": "user", "content":
                f"Based on these memories, what are 2-3 key insights?\n"
                f"{recent_text}\nRespond with bullet points only."},
        ]
        insights = self.llm(insight_prompt)
        # Store each insight as a high-importance semantic memory
        for line in insights.split("\n"):
            line = line.strip().lstrip("*-").strip()
            if len(line) > 20:
                self.memory.write(
                    f"[INSIGHT] {line}",
                    importance=0.9,
                    tier=MemoryTier.WARM,
                )
```

<span class="hh-tag">Listing 24:</span>Complete memory-augmented agent loop with read-act-reflect-write cycle.

## Recent Advances in Agentic Memory

The memory systems described above established the foundational patterns. Several recent works push the boundaries further:

### CoALA: Cognitive Architectures for Language Agents

Sumers et al. [344] propose <em>Cognitive Architectures for Language Agents</em> (CoALA), a unifying framework that organizes the growing zoo of LLM agents using principles from cognitive science and symbolic AI. CoALA decomposes a language agent into:

<ul><li>Modular memory: working memory (the context window), episodic memory (past experiences), semantic memory (world knowledge), and procedural memory (action schemas)—mirroring our taxonomy in Section 17.2 Taxonomy of Memory Types.</li><li>Structured action space: internal actions (reasoning, retrieval, memory writes) and external actions (tool use, environment interaction).</li><li>Decision cycle: a generalized sense–plan–act loop with explicit retrieval and write steps.</li></ul>

CoALA’s contribution is less a new system than a <em>design language</em>: it provides a systematic way to analyze existing agents and identify missing capabilities, making it a useful reference architecture for practitioners.

### Mem0: Production-Scale Memory Layer

Mem0 [53] addresses the gap between research memory systems and production deployment. Key ideas:

<ul><li>Automatic extraction: Rather than relying on the LLM to explicitly issue memory-write commands, Mem0 automatically extracts salient facts from conversation turns and consolidates them into a persistent store.</li><li>Graph-based memory: Beyond flat vector stores, Mem0 maintains a <em>relational graph</em> over extracted entities and facts, enabling multi-hop memory queries (“What did the user say about topic X in the context of project Y?”).</li><li>Memory compression: Redundant or superseded facts are automatically merged, keeping the memory store compact and current.</li></ul>

On the LOCOMO benchmark, Mem0 achieves 26% relative improvement over OpenAI’s baseline memory, with 91% lower p95 latency and <math alttext="&gt;" display="inline"><semantics><mo>&gt;</mo></semantics></math>90% token cost reduction compared to full-context approaches.

### Sleep-Time Compute: Offline Memory Processing

Lin et al. [219] introduce <em>sleep-time compute</em>, a paradigm where agents process and consolidate memory <em>between</em> user interactions rather than only at query time. The analogy is to biological sleep, during which the brain consolidates memories and pre-computes useful associations.

During idle periods (“sleep”), the agent:

<ol><li>Anticipates likely future queries given the current context.</li><li>Pre-computes reasoning chains, summaries, and structured representations.</li><li>Stores these pre-computed artifacts so that test-time inference can retrieve and reuse them.</li></ol>

Sleep-time compute reduces the test-time compute needed to achieve equivalent accuracy by <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math><math alttext="5\times" display="inline"><semantics><mrow><mn>5</mn><mo lspace="0.222em">×</mo></mrow></semantics></math> on reasoning benchmarks. When amortized across multiple related queries about the same context, average cost per query drops by <math alttext="2.5\times" display="inline"><semantics><mrow><mn>2.5</mn><mo lspace="0.222em">×</mo></mrow></semantics></math>. The approach is most effective when user queries are <em>predictable</em>—i.e., when the context strongly constrains what questions will be asked.

### A-MEM: Zettelkasten-Inspired Agentic Memory

A-MEM [400] introduces a memory system that borrows from the <em>Zettelkasten</em> method—a note-taking system based on densely interconnected atomic notes—to enable dynamic, self-organizing memory for LLM agents.

<ul><li>Structured notes. Each memory entry is not a raw text chunk but a <em>note</em> with multiple structured attributes: a contextual description, keywords, tags, and explicit links to related notes. This metadata enables richer retrieval than embedding similarity alone.</li><li>Dynamic linking. When a new memory is added, the system analyzes existing memories to identify semantically meaningful connections and establishes bidirectional links. The result is a <em>knowledge network</em> rather than a flat list.</li><li>Memory evolution. Critically, adding a new note can <em>trigger updates</em> to existing notes—refining their contextual representations and attributes as the agent’s understanding deepens. This makes memory a living structure that improves over time, not a static archive.</li><li>Agent-driven organization. Unlike fixed-schema memory systems, A-MEM lets the LLM itself decide how to organize, link, and update memories—making the organizational structure adaptive to the task domain.</li></ul>

Across six foundation models on multi-session reasoning tasks, A-MEM consistently outperforms flat vector stores, summarization-based memory, and graph-database approaches, demonstrating that <em>how</em> memories are organized matters as much as <em>what</em> is stored.

## Proactive Memory Architectures

A fundamental challenge in long-horizon agentic tasks is behavioral state decay: after dozens of tool calls, the executor agent loses track of its original goals, constraints, and accumulated context. Standard memory systems address this reactively—the agent queries memory when it needs information. Proactive memory architectures[253] (Meta AI, July 2026, “Remember When It Matters”) take a different approach: a dedicated memory agent <em>monitors</em> the executor and injects constraints <em>before</em> the executor drifts.

## Summary

Agentic memory systems are a foundational component of capable AI agents, addressing the fundamental limitation of finite context windows. We have surveyed:

<ul><li>A four-way taxonomy (working, episodic, semantic, procedural) that mirrors cognitive science and reflects distinct engineering requirements.</li><li>Five architectural families: RAG-based, summarization-based, graph-based, key-value networks, and tiered virtual context (MemGPT).</li><li>Four core operations: write (with importance scoring and contradiction detection), read/retrieve (with temporal decay and query expansion), update (with conflict resolution and consolidation), and reflect (meta-cognitive insight generation).</li><li>Multi-turn and multi-agent extensions: user modeling, session continuity, shared memory pools, and blackboard architectures.</li><li>RL training of memory systems: reward signals for memory operations, learning what to remember, and memory-augmented policy optimization.</li><li>Proactive memory: decoupled memory agents that monitor executor behavior and inject reminders before drift compounds—a shift from reactive retrieval to anticipatory intervention.</li></ul>

The field is rapidly evolving. Key open challenges include: (1) <em>memory grounding</em>—ensuring retrieved memories are faithfully incorporated rather than ignored or hallucinated over; (2) <em>scalable consistency</em>—maintaining coherent shared memory in large multi-agent systems; (3) <em>privacy-preserving memory</em>—enabling personalization without compromising user data; and (4) <em>proactive injection policies</em>—learning when and what to remind the executor without overwhelming its context. As context windows grow, the boundary between in-context and external memory will shift, but the fundamental need for <em>selective, structured, retrievable</em> information storage will remain.

<aside class="hh-next">
<p class="hh-next-label">下一篇</p>
<p class="hh-next-title">The Hitchhiker's Guide to Agentic AI · Chapter 18 Agent Harness – Context Management and…</p>
<p class="hh-next-hint">在「智能体 AI 漫游指南」分类下继续阅读。</p>
</aside>

</div>
