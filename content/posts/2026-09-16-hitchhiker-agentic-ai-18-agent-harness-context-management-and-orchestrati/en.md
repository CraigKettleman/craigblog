---
title: "Chapter 18 Agent Harness – Context Management and…"
slug: "hitchhiker-agentic-ai-18-agent-harness-context-management-and-orchestrati"
lang: "en"
date: "2026-09-16T00:19:00.000Z"
draft: false
featured: false
categories:
  - "hitchhiker-agentic-ai"
cover: "./cover.png"
description: "Modern LLM-based agents do not operate in isolation.…"
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

Modern LLM-based agents do not operate in isolation. Between the raw language model and the real-world tasks it must accomplish lies a layer of infrastructure that manages memory, routes tool calls, tracks state, and enforces safety constraints. This infrastructure is called the agent harness. Understanding how to design and implement a robust harness is as important as understanding the model itself—a poorly designed harness can nullify the capabilities of even the most powerful LLM, while a well-designed one can dramatically amplify what a modest model can achieve.

This section covers the full stack of agent harness design: context window management, prompt architecture, tool integration, orchestration patterns, state management, error handling, and production concerns. We conclude with a framework comparison and a complete implementation example.

## What Is an Agent Harness?

The harness enforces a clean separation of concerns:

<ul><li>Reasoning – delegated entirely to the LLM; the harness does not second-guess model outputs.</li><li>Execution – the harness dispatches tool calls, manages I/O, and enforces sandboxing.</li><li>Memory – the harness maintains short-term (context window), working (scratchpad), and long-term (vector store / database) memory.</li><li>Communication – the harness handles message routing between agents, users, and external services.</li><li>Observability – the harness instruments every step for logging, tracing, and debugging.</li></ul>

<figure id="ch18.f1"><img src="./fig_053_harness-arch.png" alt="Figure 18.1: High-level architecture of an agent harness. The LLM handles only reasoning; all execution, memory, routing" loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 18.1:</span>High-level architecture of an agent harness. The LLM handles only reasoning; all execution, memory, routing, and observability are managed by the harness.</figcaption></figure>

## Context Window Management

The context window is the agent’s working memory. Every token in the window costs money and latency; every token <em>not</em> in the window is invisible to the model. Managing this finite resource is one of the most consequential engineering decisions in agent design.

### The Context Budget Problem

Let <math alttext="C" display="inline"><semantics><mi>C</mi></semantics></math> be the maximum context length (in tokens) supported by the model. The context is partitioned into several competing components:

<div class="hh-equation" id="ch18.e1"><math alttext="C\geq\underbrace{S}_{\text{system prompt}}+\underbrace{M}_{\text{memory/RAG}}+\underbrace{T}_{\text{tool defs}}+\underbrace{H}_{\text{history}}+\underbrace{R}_{\text{reserved output}}" display="block"><semantics><mrow><mi>C</mi><mo>≥</mo><mrow><munder><munder accentunder="true"><mi>S</mi><mo stretchy="true">⏟</mo></munder><mtext>system prompt</mtext></munder><mo>+</mo><munder><munder accentunder="true"><mi>M</mi><mo stretchy="true">⏟</mo></munder><mtext>memory/RAG</mtext></munder><mo>+</mo><munder><munder accentunder="true"><mi>T</mi><mo stretchy="true">⏟</mo></munder><mtext>tool defs</mtext></munder><mo>+</mo><munder><munder accentunder="true"><mi>H</mi><mo stretchy="true">⏟</mo></munder><mtext>history</mtext></munder><mo>+</mo><munder><munder accentunder="true"><mi>R</mi><mo stretchy="true">⏟</mo></munder><mtext>reserved output</mtext></munder></mrow></mrow></semantics></math><span class="hh-equation-number">(18.1)</span></div>

As a conversation grows, <math alttext="H" display="inline"><semantics><mi>H</mi></semantics></math> expands without bound while <math alttext="C" display="inline"><semantics><mi>C</mi></semantics></math> remains fixed. Tool outputs can be large (e.g., a web page, a code execution result), causing sudden spikes in <math alttext="T+H" display="inline"><semantics><mrow><mi>T</mi><mo>+</mo><mi>H</mi></mrow></semantics></math>. The harness must continuously enforce Equation 18.1.

### Context Allocation Strategies

Assign hard token limits to each component:



Fixed allocation is simple and predictable but wastes capacity when some components are small.

Solve a constrained optimization at each turn:

<div class="hh-equation" id="ch18.e3"><math alttext="\max_{S,M,T,H,R}\;\text{Utility}(S,M,T,H,R)\quad\text{s.t.}\quad S+M+T+H+R\leq C" display="block"><semantics><mrow><mrow><mrow><munder><mi>max</mi><mrow><mi>S</mi><mo>,</mo><mi>M</mi><mo>,</mo><mi>T</mi><mo>,</mo><mi>H</mi><mo>,</mo><mi>R</mi></mrow></munder><mo lspace="0.447em">⁡</mo><mtext>Utility</mtext></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>S</mi><mo>,</mo><mi>M</mi><mo>,</mo><mi>T</mi><mo>,</mo><mi>H</mi><mo>,</mo><mi>R</mi><mo stretchy="false">)</mo></mrow></mrow><mspace width="1em"></mspace><mtext>s.t.</mtext><mspace width="1em"></mspace><mrow><mrow><mi>S</mi><mo>+</mo><mi>M</mi><mo>+</mo><mi>T</mi><mo>+</mo><mi>H</mi><mo>+</mo><mi>R</mi></mrow><mo>≤</mo><mi>C</mi></mrow></mrow></semantics></math><span class="hh-equation-number">(18.3)</span></div>

where Utility is a task-specific scoring function (e.g., weighted sum of relevance scores). In practice, dynamic allocation is approximated greedily: fill the highest-priority components first, compress or truncate lower-priority ones.

### Context Compression

When <math alttext="H" display="inline"><semantics><mi>H</mi></semantics></math> exceeds its budget, the harness must compress history without losing critical information.

Replace the oldest <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> turns with an LLM-generated summary [281]:

<div class="hh-equation" id="ch18.e4"><math alttext="H^{\prime}=\text{Summarize}(H_{1:k})\;\|\;H_{k+1:n}" display="block"><semantics><mrow><msup><mi>H</mi><mo>′</mo></msup><mo>=</mo><mtext>Summarize</mtext><mrow><mo stretchy="false">(</mo><msub><mi>H</mi><mrow><mn>1</mn><mo lspace="0.278em" rspace="0.278em">:</mo><mi>k</mi></mrow></msub><mo stretchy="false">)</mo></mrow><mo rspace="0.447em">∥</mo><msub><mi>H</mi><mrow><mrow><mi>k</mi><mo>+</mo><mn>1</mn></mrow><mo lspace="0.278em" rspace="0.278em">:</mo><mi>n</mi></mrow></msub></mrow></semantics></math><span class="hh-equation-number">(18.4)</span></div>

The summary is typically 5–10<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> shorter than the original. A dedicated “summarizer” model (smaller and cheaper) can be used for this step.

Score each message by relevance to the current query <math alttext="q" display="inline"><semantics><mi>q</mi></semantics></math>:

<div class="hh-equation" id="ch18.e5"><math alttext="\text{score}(m_{i})=\text{sim}(e(m_{i}),\,e(q))+\lambda\cdot\text{recency}(i)" display="block"><semantics><mrow><mrow><mtext>score</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>m</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mrow><mtext>sim</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mrow><mi>e</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>m</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>,</mo><mrow><mi>e</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow><mo>+</mo><mrow><mrow><mi>λ</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><mtext>recency</mtext></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(18.5)</span></div>

where <math alttext="e(\cdot)" display="inline"><semantics><mrow><mi>e</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mo lspace="0em" rspace="0em">⋅</mo><mo stretchy="false">)</mo></mrow></mrow></semantics></math> is an embedding function and <math alttext="\text{recency}(i)=i/n" display="inline"><semantics><mrow><mrow><mtext>recency</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mi>i</mi><mo>/</mo><mi>n</mi></mrow></mrow></semantics></math>. Retain the top-<math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> messages by score.

Assign importance weights <math alttext="w_{i}" display="inline"><semantics><msub><mi>w</mi><mi>i</mi></msub></semantics></math> to each turn (e.g., turns containing tool results or user corrections get higher weight). Truncate lowest-weight turns first:

<div class="hh-equation" id="ch18.e6"><math alttext="\min_{S\subseteq[n]}\sum_{i\notin S}w_{i}\quad\text{s.t.}\quad\sum_{i\in S}|m_{i}|\leq B_{H}" display="block"><semantics><mrow><mrow><mi>min</mi><mo>⁡</mo><mrow><mmultiscripts><munder><mo movablelimits="false">∑</mo><mrow><mi>i</mi><mo>∉</mo><mi>S</mi></mrow></munder><mprescripts></mprescripts><mrow><mi>S</mi><mo>⊆</mo><mrow><mo stretchy="false">[</mo><mi>n</mi><mo stretchy="false">]</mo></mrow></mrow><mrow></mrow></mmultiscripts><mo>⁡</mo><msub><mi>w</mi><mi>i</mi></msub></mrow></mrow><mspace width="1em"></mspace><mtext>s.t.</mtext><mspace width="1em"></mspace><mrow><mrow><munder><mo movablelimits="false" rspace="0em">∑</mo><mrow><mi>i</mi><mo>∈</mo><mi>S</mi></mrow></munder><mrow><mo stretchy="false">|</mo><msub><mi>m</mi><mi>i</mi></msub><mo stretchy="false">|</mo></mrow></mrow><mo>≤</mo><msub><mi>B</mi><mi>H</mi></msub></mrow></mrow></semantics></math><span class="hh-equation-number">(18.6)</span></div>

This is a variant of the 0/1 knapsack problem, solvable greedily by sorting on <math alttext="w_{i}/|m_{i}|" display="inline"><semantics><mrow><msub><mi>w</mi><mi>i</mi></msub><mo>/</mo><mrow><mo stretchy="false">|</mo><msub><mi>m</mi><mi>i</mi></msub><mo stretchy="false">|</mo></mrow></mrow></semantics></math>.

### Sliding Window Approaches

<ul><li>FIFO (First-In, First-Out): Drop the oldest messages when the window fills. Simple but loses early context (e.g., original task description).</li><li>Importance-Ranked Retention: Keep the system prompt and first user message pinned; apply importance scoring to the rest.</li><li>Hierarchical Summarization: Maintain a multi-level summary pyramid—recent turns verbatim, older turns as paragraph summaries, oldest turns as a single abstract.</li></ul>

<figure id="ch18.f2"><img src="./fig_054_sliding-window.png" alt="Figure 18.2: Three sliding-window strategies. Red = pinned, gray = dropped, blue = retained verbatim, yellow = summarize" loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 18.2:</span>Three sliding-window strategies. Red = pinned, gray = dropped, blue = retained verbatim, yellow = summarized, green = new message.</figcaption></figure>

### Recursive Context Decomposition

The strategies above—summarization, selective retention, sliding windows—all accept a fundamental constraint: <em>everything must fit in one context window</em>. A more radical approach rejects this constraint entirely: let the model recursively call itself (or a sub-model) on partitions of the context, aggregating results across calls [420].

Context rot—the empirical degradation of model accuracy as context length grows—means that even models with large context windows (128k+) perform worse on long inputs. By keeping each individual call short and focused, recursive decomposition avoids this degradation entirely. Zhang et al. [420] demonstrated that a recursive GPT-5-mini <em>outperforms</em> non-recursive GPT-5 on difficult long-context benchmarks, while being cheaper per query.

A practical RLM harness provides the model with a REPL environment containing the context as a variable. The model can:

<ol><li>Inspect the context programmatically (regex, slicing, length checks).</li><li>Partition it into manageable chunks based on structure or relevance.</li><li>Sub-query by spawning recursive LLM calls over each chunk.</li><li>Aggregate sub-results into a final answer.</li></ol>

This pattern generalizes beyond summarization: recursive search (find a needle across millions of tokens), recursive analysis (audit a large codebase), and recursive extraction (parse a corpus of documents) all follow the same decompose–recurse–aggregate structure.

<figure id="ch18.f3"><img src="./fig_055_rlm.png" alt="Figure 18.3: Recursive Language Model (RLM). The root model partitions the context into chunks, spawns sub-LLM calls at " loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 18.3:</span>Recursive Language Model (RLM). The root model partitions the context into chunks, spawns sub-LLM calls at depth 1, which may recurse further (depth 2). Results flow back up (dashed green arrows) and are aggregated into a final answer. No single call processes the full context.</figcaption></figure>

### Token Counting and Budget Monitoring

Token counting should use the model’s <em>exact</em> tokenizer (e.g., <code>tiktoken</code> for OpenAI models, <code>transformers</code> tokenizer for open-source models). Rule-of-thumb approximations (“4 chars per token”) can be off by 20–40% for code, JSON, or non-English text.

## Prompt Architecture

The prompt is the primary interface between the harness and the model. A well-structured prompt is modular, composable, and version-controlled.

### System Prompt Design

A production system prompt typically contains four sections:

<ol><li>Persona: Who the agent is, its name, role, and communication style.</li><li>Capabilities: What the agent can do (tools available, knowledge cutoff, supported languages).</li><li>Constraints: What the agent must <em>not</em> do (safety rules, scope limits, confidentiality).</li><li>Output Format: Expected response structure (JSON schema, markdown, step-by-step reasoning).</li></ol>

### Dynamic Prompt Assembly

Rather than a single monolithic string, production harnesses assemble prompts from components at runtime:

<div class="hh-equation" id="ch18.e8"><math alttext="\text{Prompt}=\text{Concat}\bigl(\text{SystemBlock},\;\text{MemoryBlock},\;\text{ToolBlock},\;\text{HistoryBlock},\;\text{QueryBlock}\bigr)" display="block"><semantics><mrow><mtext>Prompt</mtext><mo>=</mo><mrow><mtext>Concat</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo maxsize="1.200em" minsize="1.200em">(</mo><mtext>SystemBlock</mtext><mo>,</mo><mtext>MemoryBlock</mtext><mo>,</mo><mtext>ToolBlock</mtext><mo>,</mo><mtext>HistoryBlock</mtext><mo>,</mo><mtext>QueryBlock</mtext><mo maxsize="1.200em" minsize="1.200em">)</mo></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(18.8)</span></div>

Each block is independently versioned, tested, and can be swapped without touching others. A prompt registry stores named templates with semantic versioning (e.g., <code>system/v2.3.1</code>).

### Few-Shot Management

Few-shot examples improve reliability but consume tokens. The harness should [224]:

<ul><li>Select relevant examples using embedding similarity to the current query.</li><li>Rotate examples to avoid overfitting to a fixed set.</li><li>Budget examples within the <math alttext="M" display="inline"><semantics><mi>M</mi></semantics></math> allocation (Equation 18.2).</li><li>Cache embeddings of the example library to avoid recomputation.</li></ul>

Formally, few-shot selection is a constrained optimization—maximizing total relevance subject to a token budget:

<div class="hh-equation" id="ch18.e9"><math alttext="\text{examples}^{*}=\underset{E\subseteq\mathcal{E},\;|E|\leq k}{\arg\max}\sum_{e\in E}\text{sim}(e(e_{\text{input}}),\,e(q))\quad\text{s.t.}\quad\sum_{e\in E}|e|\leq B_{M}" display="block"><semantics><mrow><mrow><msup><mtext>examples</mtext><mo>∗</mo></msup><mo>=</mo><mrow><munder accentunder="true"><mrow><mi>arg</mi><mo lspace="0.167em">⁡</mo><mi>max</mi></mrow><mrow><mrow><mi mathsize="0.700em">E</mi><mo mathsize="0.700em">⊆</mo><mi mathsize="0.700em">ℰ</mi></mrow><mo mathsize="0.700em" rspace="0.447em">,</mo><mrow><mrow><mo maxsize="0.700em" minsize="0.700em" stretchy="true">|</mo><mi mathsize="0.700em">E</mi><mo maxsize="0.700em" minsize="0.700em" stretchy="true">|</mo></mrow><mo mathsize="0.700em">≤</mo><mi mathsize="0.700em">k</mi></mrow></mrow></munder><mo lspace="0em" rspace="0em">​</mo><mrow><munder><mo movablelimits="false">∑</mo><mrow><mi>e</mi><mo>∈</mo><mi>E</mi></mrow></munder><mrow><mtext>sim</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mrow><mi>e</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>e</mi><mtext>input</mtext></msub><mo stretchy="false">)</mo></mrow></mrow><mo>,</mo><mrow><mi>e</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></mrow><mspace width="1em"></mspace><mtext>s.t.</mtext><mspace width="1em"></mspace><mrow><mrow><munder><mo movablelimits="false" rspace="0em">∑</mo><mrow><mi>e</mi><mo>∈</mo><mi>E</mi></mrow></munder><mrow><mo stretchy="false">|</mo><mi>e</mi><mo stretchy="false">|</mo></mrow></mrow><mo>≤</mo><msub><mi>B</mi><mi>M</mi></msub></mrow></mrow></semantics></math><span class="hh-equation-number">(18.9)</span></div>

### Tool Descriptions

Tool descriptions are part of the prompt and directly affect tool selection quality. A well-designed tool signature has five components:

<ol><li>Name: Use a verb–noun pattern (<code>search_web</code>, <code>read_file</code>, <code>send_email</code>). Avoid generic names like <code>do_action</code> or ambiguous ones like <code>process</code>.</li><li>Description: One to two sentences explaining <em>what</em> the tool does, <em>when</em> to use it, and <em>when not</em> to use it. This is the primary signal the model uses for selection.</li><li>Input parameters: Each parameter needs a type, a human-readable description, and whether it is required or optional (with a sensible default).</li><li>Output specification: Document the return format—structured JSON, plain text, or error codes—so the model can parse results correctly.</li><li>Constraints: Rate limits, maximum input size, required permissions, or side effects (e.g., “This tool sends a real email—use only after user confirmation”).</li></ol>

Additional best practices for tool descriptions in the prompt:

<ul><li>Be specific: “Search the web for current information” is better than “Search”.</li><li>Include when to use: “Use this when the user asks about events after your knowledge cutoff.”</li><li>Include when NOT to use: Reduces false positives.</li><li>Exclude irrelevant tools: Dynamically include only tools relevant to the current task to save tokens and reduce confusion.</li><li>Optimize descriptions: A/B test descriptions; small wording changes can shift tool selection accuracy by 10–20%.</li></ul>

## Tool Integration and Execution

Tool use is a defining capability of modern LLM agents [315]. The harness manages tool definitions, selection, execution, and output processing.

### Tool Definition Schemas

Different providers use different schemas for tool definitions:

Anthropic uses a similar JSON schema but with an <code>input_schema</code> key instead of <code>parameters</code>, and tools are passed in a top-level <code>tools</code> array:

MCP (Section 18.4.5 Model Context Protocol (MCP)) provides a standardized protocol for tool discovery and invocation across providers, decoupling tool definitions from any single API format.

### Tool Selection and Routing

The model selects tools based on its understanding of tool descriptions and the current task. The harness can influence this:

<ul><li>Auto tool use: The model decides whether and which tool to call.</li><li>Forced tool use: The harness specifies <code>tool\_choice: {type: "function", function: {name: "X"}}</code> to force a specific tool (useful for structured extraction).</li><li>Parallel tool calls: Modern APIs allow the model to request multiple tool calls in a single turn, which the harness executes concurrently.</li></ul>

When an agent has access to hundreds or thousands of tools, including all definitions in the prompt is infeasible (token cost) and counterproductive (selection confusion). Two key approaches address this:

<ul><li>Retrieval-augmented tool selection: At each turn, retrieve only the top-<math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> most relevant tools using embedding similarity between the user query and tool descriptions. This mirrors RAG for documents—only contextually relevant tools are injected into the prompt. Gorilla[287] demonstrated that combining retrieval with retriever-aware training (RAT) enables LLMs to accurately select from thousands of overlapping APIs, adapting to version changes at test time.</li><li>Fine-tuned tool selection:ToolLLM[299] trains models on a large corpus of tool-use trajectories (16,000+ APIs) using a depth-first search-based decision tree (DFSDT) to generate solution paths. The resulting model learns generalizable tool selection strategies that transfer to unseen APIs, achieving significantly better accuracy than prompt-only approaches.</li></ul>

In practice, production harnesses combine these strategies: a retrieval layer pre-filters the tool set, the prompt includes the filtered tools, and the model’s native function-calling capability handles final selection.

### Tool Output Processing

Raw tool outputs are rarely ready for direct insertion into the context:

<ol><li>Parse and validate: Check that the output matches the expected schema.</li><li>Truncate large outputs: Web pages, code outputs, and database results can be enormous. Apply summarization or chunking before inserting into context.</li><li>Error normalization: Convert provider-specific errors into a standard format the model can reason about.</li><li>Retry logic: On transient failures (network timeout, rate limit), retry with exponential backoff before reporting failure to the model.</li></ol>

### Sandboxing and Safety

Tool execution is a major attack surface. The harness must enforce:

<ul><li>Execution isolation: Run code tools in containers (Docker, gVisor) or VMs with no network access by default.</li><li>Permission models: Declare required permissions per tool (read-only filesystem, network access, etc.) and enforce them at the OS level.</li><li>Resource limits: CPU time, memory, and wall-clock timeouts prevent runaway executions.</li><li>Input sanitization: Validate and sanitize all model-generated tool arguments before execution (prevent prompt injection via tool outputs).</li><li>Audit logging: Log every tool call with arguments, outputs, and timestamps for post-hoc review.</li></ul>

### Model Context Protocol (MCP)

The Model Context Protocol (MCP) [10] is an open standard for connecting LLM applications to external tools and data sources. It decouples tool <em>providers</em> from tool <em>consumers</em>. We cover MCP in depth in Chapter Chapter 22 Model Context Protocol (MCP); here we summarize the key ideas relevant to harness design.

MCP uses a client-server model:

<ul><li>MCP Server: Exposes tools, resources, and prompts over a standardized protocol. Can be a local process or a remote service.</li><li>MCP Client: The agent harness connects to one or more MCP servers, discovers available tools, and routes tool calls.</li><li>Transport Layers: Supports <code>stdio</code> (local subprocess), HTTP+SSE (remote), and WebSocket transports.</li></ul>

At startup, the harness calls <code>tools/list</code> on each connected MCP server to discover available tools and their schemas. This enables dynamic tool registration—new tools become available without redeploying the harness.

<ol><li>Model outputs a tool call (e.g., <code>mcp_server_name::tool_name(args)</code>).</li><li>Harness routes the call to the appropriate MCP server via <code>tools/call</code>.</li><li>MCP server executes the tool and returns a structured result.</li><li>Harness inserts the result into the context as a <code>tool</code> message.</li></ol>

<figure id="ch18.f4"><img src="./fig_056_mcp-arch.png" alt="Figure 18.4: MCP architecture. The harness acts as an MCP client, routing tool calls to specialized MCP servers over sta" loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 18.4:</span>MCP architecture. The harness acts as an MCP client, routing tool calls to specialized MCP servers over standardized transports.</figcaption></figure>

## Orchestration Patterns

Orchestration defines <em>how</em> the agent decides what to do next. Different patterns suit different task structures.

### ReAct Loop (Reason + Act)

The ReAct pattern [409] interleaves reasoning (“Thought”) with action (“Act”) and observation (“Observe”) in a tight loop:

<div class="hh-equation" id="ch18.e10"><math alttext="\text{Thought}_{t}\to\text{Action}_{t}\to\text{Observation}_{t}\to\text{Thought}_{t+1}\to\cdots" display="block"><semantics><mrow><msub><mtext>Thought</mtext><mi>t</mi></msub><mo stretchy="false">→</mo><msub><mtext>Action</mtext><mi>t</mi></msub><mo stretchy="false">→</mo><msub><mtext>Observation</mtext><mi>t</mi></msub><mo stretchy="false">→</mo><msub><mtext>Thought</mtext><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub><mo rspace="0.1389em" stretchy="false">→</mo><mo lspace="0.1389em">⋯</mo></mrow></semantics></math><span class="hh-equation-number">(18.10)</span></div>

<figure id="ch18.f5"><img src="./fig_057_react-loop.png" alt="Figure 18.5: ReAct loop: the agent alternates between reasoning and acting until a termination condition is met." loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 18.5:</span>ReAct loop: the agent alternates between reasoning and acting until a termination condition is met.</figcaption></figure>

<ul><li>The “Thought” step is typically a scratchpad—a chain-of-thought reasoning trace [374] that is <em>not</em> shown to the user.</li><li>The harness parses the model’s output to extract the action (tool name + arguments).</li><li>A max iterations guard prevents infinite loops.</li><li>The loop terminates when the model outputs a “Final Answer” action or a stop token.</li></ul>

### Plan-and-Execute

Rather than deciding one step at a time, the agent first generates a complete plan, then executes each step [364]:

<ol><li>Planning phase: Given the task, generate a structured plan (list of subtasks with dependencies).</li><li>Execution phase: Execute each subtask, potentially using a different (cheaper) model.</li><li>Plan revision: If a step fails or produces unexpected results, re-plan from the current state.</li></ol>

<div class="hh-equation" id="ch18.e11"><math alttext="\text{Plan}=\text{Planner}(q),\quad\text{Result}=\prod_{i=1}^{|\text{Plan}|}\text{Executor}(\text{Plan}[i],\,\text{context}_{i})" display="block"><semantics><mrow><mrow><mtext>Plan</mtext><mo>=</mo><mrow><mtext>Planner</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></mrow></mrow><mo rspace="1.167em">,</mo><mrow><mtext>Result</mtext><mo rspace="0.111em">=</mo><mrow><munderover><mo movablelimits="false">∏</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mrow><mo stretchy="false">|</mo><mtext>Plan</mtext><mo stretchy="false">|</mo></mrow></munderover><mrow><mtext>Executor</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mrow><mtext>Plan</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">[</mo><mi>i</mi><mo stretchy="false">]</mo></mrow></mrow><mo>,</mo><msub><mtext>context</mtext><mi>i</mi></msub><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(18.11)</span></div>

Plan-and-execute is more efficient for long-horizon tasks (fewer LLM calls) but less adaptive to unexpected observations.

### Multi-Agent Orchestration

Complex tasks benefit from multiple specialized agents working together. Four canonical patterns:

A central “supervisor” LLM receives the user request, decomposes it, and routes subtasks to specialist agents. Results are aggregated by the supervisor.

<figure id="ch18.f6"><img src="./fig_058_supervisor.png" alt="Figure 18.6: Supervisor pattern: one orchestrator routes to specialist agents." loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 18.6:</span>Supervisor pattern: one orchestrator routes to specialist agents.</figcaption></figure>

Agents communicate directly without a central coordinator. Each agent can invoke any other agent as a tool. Flexible but harder to debug and prone to circular dependencies.

A tree structure where high-level agents delegate to mid-level agents, which delegate to leaf agents. Enables recursive task decomposition. Used in systems like AutoGen’s nested chat.

Popularized by OpenAI’s Swarm library [276], this pattern uses handoffs: an agent can transfer control to another agent along with the full conversation context. Key concepts:

<ul><li>Agents have instructions and tools.</li><li>Handoffs are special tools that transfer control.</li><li>Context variables are shared state passed between agents.</li><li>The active agent changes dynamically based on task needs.</li></ul>

### Human-in-the-Loop

Production agents must know when to pause and ask for human input:

<ul><li>Approval gates: Before irreversible actions (sending emails, deleting files, making purchases), require explicit human confirmation.</li><li>Escalation criteria: Escalate when confidence is below a threshold, when the task is outside defined scope, or when a safety rule is triggered.</li><li>Feedback integration: Human corrections are inserted into the context and can update the agent’s plan.</li><li>Async approval: For long-running tasks, the agent can pause, notify the human via email/Slack, and resume when approved.</li></ul>

### Workflow Graphs

For complex, structured workflows, the orchestration logic is expressed as a directed acyclic graph (DAG) or state machine:

<ul><li>LangGraph[155]: Extends LangChain with a graph-based execution model. Nodes are agent steps; edges are conditional transitions. Supports cycles (for ReAct loops) and parallel branches.</li><li>AutoGen[385]: Microsoft’s framework for multi-agent conversation graphs. Supports nested chats, group chats, and human-in-the-loop patterns.</li><li>State machines: Explicit states (e.g., <code>PLANNING</code>, <code>EXECUTING</code>, <code>WAITING_FOR_HUMAN</code>, <code>DONE</code>) with defined transitions. Easier to reason about and test than implicit loop logic.</li></ul>

<div class="hh-equation" id="ch18.e13"><math alttext="G=(V,E,\sigma_{0}),\quad v\in V:\text{agent step},\quad e\in E:\text{conditional transition},\quad\sigma_{0}:\text{initial state}" display="block"><semantics><mrow><mrow><mi>G</mi><mo>=</mo><mrow><mo stretchy="false">(</mo><mi>V</mi><mo>,</mo><mi>E</mi><mo>,</mo><msub><mi>σ</mi><mn>0</mn></msub><mo stretchy="false">)</mo></mrow></mrow><mo rspace="1.167em">,</mo><mrow><mi>v</mi><mo>∈</mo><mi>V</mi><mo lspace="0.278em" rspace="0.278em">:</mo><mtext>agent step</mtext></mrow><mo rspace="1.167em">,</mo><mrow><mi>e</mi><mo>∈</mo><mi>E</mi><mo lspace="0.278em" rspace="0.278em">:</mo><mtext>conditional transition</mtext></mrow><mo rspace="1.167em">,</mo><mrow><msub><mi>σ</mi><mn>0</mn></msub><mo lspace="0.278em" rspace="0.278em">:</mo><mtext>initial state</mtext></mrow></mrow></semantics></math><span class="hh-equation-number">(18.13)</span></div>

<figure id="ch18.f7"><img src="./fig_059_workflow-graph.png" alt="Figure 18.7: Example workflow graph for a human-in-the-loop agent. States and conditional transitions are explicit, maki" loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 18.7:</span>Example workflow graph for a human-in-the-loop agent. States and conditional transitions are explicit, making the control flow auditable.</figcaption></figure>

## State Management

Agents are inherently stateful. The harness must manage multiple layers of state:

### Conversation State

The message history is the primary state artifact. Each message has:

<ul><li>Role:<code>system</code>, <code>user</code>, <code>assistant</code>, <code>tool</code>.</li><li>Content: Text, tool call, or tool result.</li><li>Metadata: Timestamp, token count, importance score, compression status.</li></ul>

### Task State

For long-running tasks, the harness tracks:

<ul><li>Progress: Which subtasks are complete, in-progress, or pending.</li><li>Checkpoints: Serialized state snapshots that allow resumption after failure.</li><li>Rollback: The ability to undo the last <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> actions if a mistake is detected.</li></ul>

### Agent State

The agent’s internal state includes:

<ul><li>Current plan: The sequence of steps the agent intends to take.</li><li>Pending actions: Tool calls that have been issued but not yet returned.</li><li>Beliefs: Facts the agent has established (e.g., “the user’s timezone is UTC+9”).</li></ul>

### Persistent State

For cross-session continuity [281, 362]:

<ul><li>User profiles: Preferences, past interactions, learned facts about the user.</li><li>Long-term memory: Vector database of past conversations, searchable by semantic similarity.</li><li>Task history: Completed tasks with outcomes, used for few-shot retrieval.</li></ul>

## Error Handling and Recovery

Agents operate in adversarial, unpredictable environments. Robust error handling is non-negotiable.

### Retry Strategies

<ul><li>Exponential backoff: For transient failures (rate limits, network errors), retry after <math alttext="\min(2^{k}\cdot t_{0}+\epsilon,t_{\max})" display="inline"><semantics><mrow><mi>min</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mrow><msup><mn>2</mn><mi>k</mi></msup><mo lspace="0.222em" rspace="0.222em">⋅</mo><msub><mi>t</mi><mn>0</mn></msub></mrow><mo>+</mo><mi>ϵ</mi></mrow><mo>,</mo><msub><mi>t</mi><mi>max</mi></msub><mo stretchy="false">)</mo></mrow></mrow></semantics></math> seconds, where <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> is the retry count and <math alttext="\epsilon" display="inline"><semantics><mi>ϵ</mi></semantics></math> is random jitter.</li><li>Fallback models: If the primary model is unavailable or returns an error, fall back to a secondary model (potentially less capable but available).</li><li>Graceful degradation: If a tool is unavailable, inform the model and let it attempt the task without that tool.</li></ul>

The backoff delay for the <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math>-th retry is:

<div class="hh-equation" id="ch18.e14"><math alttext="t_{k}=\min\!\left(2^{k}\cdot t_{0}+\mathcal{U}(0,t_{0}),\;t_{\max}\right),\quad k=0,1,2,\ldots" display="block"><semantics><mrow><msub><mi>t</mi><mi>k</mi></msub><mo>=</mo><mpadded width="1.653em"><mi>min</mi></mpadded><mrow><mo>(</mo><msup><mn>2</mn><mi>k</mi></msup><mo lspace="0.222em" rspace="0.222em">⋅</mo><msub><mi>t</mi><mn>0</mn></msub><mo>+</mo><mi>𝒰</mi><mrow><mo stretchy="false">(</mo><mn>0</mn><mo>,</mo><msub><mi>t</mi><mn>0</mn></msub><mo stretchy="false">)</mo></mrow><mo rspace="0.447em">,</mo><msub><mi>t</mi><mi>max</mi></msub><mo>)</mo></mrow><mo rspace="1.167em">,</mo><mi>k</mi><mo>=</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo>,</mo><mn>2</mn><mo>,</mo><mi mathvariant="normal">…</mi></mrow></semantics></math><span class="hh-equation-number">(18.14)</span></div>

### Loop Detection

Agents can get stuck in infinite loops—repeatedly calling the same tool with the same arguments, or oscillating between two states. Detection and self-correction strategies [326]:

<ul><li>Max iteration guard: Hard limit on the number of steps per task (e.g., 50 steps).</li><li>Action deduplication: Hash each (tool, args) pair; if the same call appears <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> times, break the loop.</li><li>Progress detection: If the agent’s state has not changed in <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> steps, trigger a “stuck” handler.</li></ul>

Formally, a loop is detected when the same action hash appears within a sliding window of size <math alttext="W" display="inline"><semantics><mi>W</mi></semantics></math>:

<div class="hh-equation" id="ch18.e15"><math alttext="\text{loop\_detected}\iff\exists\,i&lt;j\leq t:\text{hash}(\text{action}_{i})=\text{hash}(\text{action}_{j})\;\land\;j-i\leq W" display="block"><semantics><mrow><mtext>loop_detected</mtext><mo stretchy="false">⇔</mo><mrow><mrow><mo rspace="0.337em">∃</mo><mi>i</mi></mrow><mo>&lt;</mo><mi>j</mi><mo>≤</mo><mi>t</mi></mrow><mo lspace="0.278em" rspace="0.278em">:</mo><mrow><mrow><mtext>hash</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mtext>action</mtext><mi>i</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mrow><mrow><mtext>hash</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mtext>action</mtext><mi>j</mi></msub><mo rspace="0.280em" stretchy="false">)</mo></mrow></mrow><mo rspace="0.502em">∧</mo><mi>j</mi></mrow><mo>−</mo><mi>i</mi></mrow><mo>≤</mo><mi>W</mi></mrow></mrow></semantics></math><span class="hh-equation-number">(18.15)</span></div>

### Graceful Failure

When the agent cannot complete a task:

<ol><li>Explain what was accomplished (partial results).</li><li>Explain why the task could not be completed.</li><li>Suggest recovery actions (e.g., “Please provide your API key to enable web search”).</li><li>Preserve state so the task can be resumed.</li></ol>

### Observability

## Scaling and Production Concerns

### Latency Optimization

<ul><li>Parallel tool calls: Execute independent tool calls concurrently using <code>asyncio</code> or thread pools. Can reduce multi-tool latency by <math alttext="N\times" display="inline"><semantics><mrow><mi>N</mi><mo lspace="0.222em">×</mo></mrow></semantics></math> for <math alttext="N" display="inline"><semantics><mi>N</mi></semantics></math> parallel calls.</li><li>Streaming: Use streaming APIs to begin processing the model’s response before it is complete. Reduces time-to-first-token for the user.</li><li>Prompt caching: Many providers (Anthropic, OpenAI) offer prompt caching for repeated prefixes (e.g., system prompt + tool definitions). Can reduce latency and cost by 50–90% for the cached portion.</li><li>Speculative execution: Begin executing the most likely next tool call before the model has finished generating, and cancel if the prediction was wrong.</li></ul>

### Cost Management

<ul><li>Token budgets: Enforce per-task and per-user token budgets. Alert when approaching limits.</li><li>Model routing: Use a cheap, fast model (e.g., GPT-4o-mini, Claude Haiku) for simple steps (tool selection, formatting) and an expensive model (GPT-4o, Claude Opus) only for complex reasoning [46].</li><li>Caching: Cache deterministic tool outputs (e.g., database lookups, static web pages) to avoid redundant API calls.</li></ul>

The total cost of an agent task with <math alttext="T" display="inline"><semantics><mi>T</mi></semantics></math> LLM steps and <math alttext="K" display="inline"><semantics><mi>K</mi></semantics></math> tool calls is:

<div class="hh-equation" id="ch18.e16"><math alttext="\text{Cost}_{\text{task}}=\sum_{i=1}^{T}\underbrace{p_{\text{in}}\cdot n_{\text{in},i}+p_{\text{out}}\cdot n_{\text{out},i}}_{\text{LLM cost}}+\sum_{j=1}^{K}\underbrace{c_{j}}_{\text{tool cost}}" display="block"><semantics><mrow><msub><mtext>Cost</mtext><mtext>task</mtext></msub><mo rspace="0.111em">=</mo><mrow><mrow><munderover><mo movablelimits="false">∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>T</mi></munderover><munder><munder accentunder="true"><mrow><mrow><msub><mi>p</mi><mtext>in</mtext></msub><mo lspace="0.222em" rspace="0.222em">⋅</mo><msub><mi>n</mi><mrow><mtext>in</mtext><mo>,</mo><mi>i</mi></mrow></msub></mrow><mo>+</mo><mrow><msub><mi>p</mi><mtext>out</mtext></msub><mo lspace="0.222em" rspace="0.222em">⋅</mo><msub><mi>n</mi><mrow><mtext>out</mtext><mo>,</mo><mi>i</mi></mrow></msub></mrow></mrow><mo stretchy="true">⏟</mo></munder><mtext>LLM cost</mtext></munder></mrow><mo rspace="0.055em">+</mo><mrow><munderover><mo movablelimits="false">∑</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mi>K</mi></munderover><munder><munder accentunder="true"><msub><mi>c</mi><mi>j</mi></msub><mo stretchy="true">⏟</mo></munder><mtext>tool cost</mtext></munder></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(18.16)</span></div>

where <math alttext="p_{\text{in}},p_{\text{out}}" display="inline"><semantics><mrow><msub><mi>p</mi><mtext>in</mtext></msub><mo>,</mo><msub><mi>p</mi><mtext>out</mtext></msub></mrow></semantics></math> are per-token prices, <math alttext="n_{\text{in},i},n_{\text{out},i}" display="inline"><semantics><mrow><msub><mi>n</mi><mrow><mtext>in</mtext><mo>,</mo><mi>i</mi></mrow></msub><mo>,</mo><msub><mi>n</mi><mrow><mtext>out</mtext><mo>,</mo><mi>i</mi></mrow></msub></mrow></semantics></math> are input/output token counts for step <math alttext="i" display="inline"><semantics><mi>i</mi></semantics></math>, and <math alttext="c_{j}" display="inline"><semantics><msub><mi>c</mi><mi>j</mi></msub></semantics></math> is the cost of tool call <math alttext="j" display="inline"><semantics><mi>j</mi></semantics></math>.

### Rate Limiting and Queuing

When running many agents concurrently:

<ul><li>Token bucket rate limiter: Enforce per-minute token limits across all agents sharing an API key.</li><li>Priority queues: High-priority tasks (interactive user requests) preempt low-priority tasks (batch processing).</li><li>Backpressure: When the queue is full, reject new tasks with a <code>503 Service Unavailable</code> rather than silently queuing indefinitely.</li></ul>

### Evaluation in Production

<ul><li>A/B testing: Route a fraction of traffic to a new agent version and compare success rates, cost, and latency.</li><li>Canary deployments: Gradually increase traffic to a new version while monitoring for regressions.</li><li>Shadow mode: Run a new agent in parallel with the production agent, compare outputs, but only serve the production output to users.</li><li>LLM-as-judge: Use a separate LLM to evaluate agent outputs on dimensions like helpfulness, accuracy, and safety [433].</li></ul>

## Framework Comparison

<div class="hh-table" id="ch18.t1"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Framework</span></th><th class="ltx_align_left"><span>Flex.</span></th><th class="ltx_align_left"><span>Complex.</span></th><th class="ltx_align_left"><span>Prod.</span></th><th class="ltx_align_left"><span>Multi-Agent</span></th><th class="ltx_align_left"><span>Best For</span></th></tr></thead><tbody><tr><td class="ltx_align_left"><span>LangChain</span></td><td class="ltx_align_left"><span>H</span></td><td class="ltx_align_left"><span>H</span></td><td class="ltx_align_left"><span>M</span></td><td class="ltx_align_left"><span>M</span></td><td class="ltx_align_left"><span>Rapid prototyping, chains</span></td></tr><tr><td class="ltx_align_left"><span>LangGraph</span></td><td class="ltx_align_left"><span>H</span></td><td class="ltx_align_left"><span>H</span></td><td class="ltx_align_left"><span>H</span></td><td class="ltx_align_left"><span>H</span></td><td class="ltx_align_left"><span>Complex stateful workflows</span></td></tr><tr><td class="ltx_align_left"><span>AutoGen</span></td><td class="ltx_align_left"><span>M</span></td><td class="ltx_align_left"><span>M</span></td><td class="ltx_align_left"><span>M</span></td><td class="ltx_align_left"><span>H</span></td><td class="ltx_align_left"><span>Multi-agent conversations</span></td></tr><tr><td class="ltx_align_left"><span>CrewAI</span></td><td class="ltx_align_left"><span>M</span></td><td class="ltx_align_left"><span>L</span></td><td class="ltx_align_left"><span>M</span></td><td class="ltx_align_left"><span>H</span></td><td class="ltx_align_left"><span>Role-based teams</span></td></tr><tr><td class="ltx_align_left"><span>OAI Assistants</span></td><td class="ltx_align_left"><span>L</span></td><td class="ltx_align_left"><span>L</span></td><td class="ltx_align_left"><span>H</span></td><td class="ltx_align_left"><span>L</span></td><td class="ltx_align_left"><span>Simple hosted agents</span></td></tr><tr><td class="ltx_align_left"><span>OpenAI Swarm</span></td><td class="ltx_align_left"><span>M</span></td><td class="ltx_align_left"><span>L</span></td><td class="ltx_align_left"><span>L</span></td><td class="ltx_align_left"><span>H</span></td><td class="ltx_align_left"><span>Handoff patterns</span></td></tr><tr><td class="ltx_align_left"><span>Custom</span></td><td class="ltx_align_left"><span>H</span></td><td class="ltx_align_left"><span>H</span></td><td class="ltx_align_left"><span>H</span></td><td class="ltx_align_left"><span>H</span></td><td class="ltx_align_left"><span>Full control, no lock-in</span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 18.1:</span>Comparison of major agent orchestration frameworks.</p></div>

Legend: H = High, M = Medium, L = Low. Flex. = Flexibility, Complex. = Complexity, Prod. = Production-readiness.

<ul><li>LangChain[41]1 provides a rich ecosystem of integrations but has a steep learning curve and abstractions that can obscure what is actually happening.</li><li>LangGraph[155]2 adds explicit graph-based control flow to LangChain, making complex multi-step agents much more manageable.</li><li>AutoGen[385]3 excels at multi-agent conversations and nested chats, with good support for human-in-the-loop patterns.</li><li>CrewAI[261]4 offers a high-level, role-based abstraction (“crew of agents”) that is easy to get started with but less flexible for custom patterns.</li><li>OpenAI Assistants API5 is fully managed (no infrastructure to run) but offers limited customization and vendor lock-in.</li><li>OpenAI Swarm[276]6 is a lightweight, educational framework demonstrating the handoff pattern; not production-ready.</li><li>Custom harness offers maximum control and is the right choice for production systems with specific requirements, but requires significant engineering investment.</li></ul>

## Implementation: Production Agent Harness

The following is a complete, production-quality agent harness implementation demonstrating context management, tool integration, the ReAct orchestration loop, and error handling.

```
"""
production_harness.py -- A production-quality agent harness.
Demonstrates: context management, tool integration,
ReAct loop, error handling, and observability.
"""

from __future__ import annotations
import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import tiktoken
from openai import AsyncOpenAI

# -- Logging / Observability ----------------------------------
logger = logging.getLogger("agent_harness")

# -- Data Models ----------------------------------------------

class Role(str, Enum):
    SYSTEM    = "system"
    USER      = "user"
    ASSISTANT = "assistant"
    TOOL      = "tool"

@dataclass
class Message:
    role:        Role
    content:     str
    tool_calls:  Optional[list[dict]] = None
    tool_call_id: Optional[str]       = None
    metadata:    dict                 = field(default_factory=dict)

    def to_api_dict(self) -> dict:
        d: dict = {"role": self.role.value,
                   "content": self.content or None}
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d

@dataclass
class ToolDefinition:
    name:        str
    description: str
    parameters:  dict
    handler:     Callable
    requires_approval: bool = False

    def to_api_dict(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name":        self.name,
                "description": self.description,
                "parameters":  self.parameters,
            }
        }

# -- Context Manager ------------------------------------------

class ContextManager:
    """
    Manages the context window with budget enforcement,
    compression, and token counting.
    """
    BUDGET_FRACTIONS = {
        "system":   0.10,
        "memory":   0.20,
        "tools":    0.10,
        "history":  0.50,
        "reserved": 0.10,
    }

    def __init__(self, model: str, max_tokens: int):
        self.model      = model
        self.max_tokens = max_tokens
        self.enc        = tiktoken.encoding_for_model(model)
        self.history:   list[Message] = []
        self.system_msg: Optional[Message] = None

    def count_tokens(self, text: str) -> int:
        return len(self.enc.encode(text))

    def count_message_tokens(self, msg: Message) -> int:
        # OpenAI overhead: 4 tokens per message + role
        return self.count_tokens(msg.content or "") + 4

    def total_history_tokens(self) -> int:
        return sum(self.count_message_tokens(m)
                   for m in self.history)

    def history_budget(self) -> int:
        return int(self.max_tokens
                   * self.BUDGET_FRACTIONS["history"])

    def add_message(self, msg: Message) -> None:
        self.history.append(msg)
        self._enforce_budget()

    def _enforce_budget(self) -> None:
        budget = self.history_budget()
        while (self.total_history_tokens() > budget
               and len(self.history) > 2):
            # Drop oldest non-pinned message (index 1).
            # If it has tool_calls, also drop the tool results
            # that follow it to keep the conversation valid.
            dropped = self.history.pop(1)
            if dropped.tool_calls:
                while (len(self.history) > 1
                       and self.history[1].role == Role.TOOL):
                    self.history.pop(1)
        logger.debug(
            "Context: %d/%d tokens used",
            self.total_history_tokens(), budget
        )

    def preflight_check(self, tool_tokens: int) -> bool:
        """Returns True if we are within budget."""
        sys_tokens = (self.count_message_tokens(self.system_msg)
                      if self.system_msg else 0)
        total = (sys_tokens
                 + tool_tokens
                 + self.total_history_tokens())
        reserved = int(self.max_tokens
                       * self.BUDGET_FRACTIONS["reserved"])
        ok = total <= (self.max_tokens - reserved)
        if not ok:
            logger.warning(
                "Context overflow: %d > %d",
                total, self.max_tokens - reserved
            )
        return ok

    def build_messages(self) -> list[dict]:
        msgs = []
        if self.system_msg:
            msgs.append(self.system_msg.to_api_dict())
        msgs.extend(m.to_api_dict() for m in self.history)
        return msgs

# -- Tool Executor --------------------------------------------

class ToolExecutor:
    """
    Executes tool calls with sandboxing, retry logic,
    and output truncation.
    """
    MAX_OUTPUT_TOKENS = 2000
    MAX_RETRIES       = 3

    def __init__(self, tools: list[ToolDefinition],
                 approval_callback: Optional[Callable] = None,
                 encoding: str = "cl100k_base"):
        self.tools    = {t.name: t for t in tools}
        self.approval = approval_callback
        self.enc      = tiktoken.get_encoding(encoding)

    async def execute(self, tool_name: str,
                      args: dict) -> str:
        tool = self.tools.get(tool_name)
        if not tool:
            return f"Error: unknown tool '{tool_name}'"

        # Human-in-the-loop approval gate
        if tool.requires_approval and self.approval:
            approved = await self.approval(tool_name, args)
            if not approved:
                return "Action rejected by human reviewer."

        for attempt in range(self.MAX_RETRIES):
            try:
                result = await asyncio.wait_for(
                    self._call(tool, args), timeout=30.0
                )
                return self._truncate(result)
            except asyncio.TimeoutError:
                logger.warning("Tool %s timed out (attempt %d)",
                               tool_name, attempt + 1)
                if attempt == self.MAX_RETRIES - 1:
                    return f"Error: tool '{tool_name}' timed out"
                await asyncio.sleep(2 ** attempt)  # backoff
            except Exception as exc:
                logger.error("Tool %s error: %s", tool_name, exc)
                if attempt == self.MAX_RETRIES - 1:
                    return f"Error: {exc}"
                await asyncio.sleep(2 ** attempt)
        return "Error: max retries exceeded"

    async def _call(self, tool: ToolDefinition,
                    args: dict) -> str:
        if asyncio.iscoroutinefunction(tool.handler):
            result = await tool.handler(**args)
        else:
            result = await asyncio.get_running_loop().run_in_executor(
                None, lambda: tool.handler(**args)
            )
        return str(result)

    def _truncate(self, text: str) -> str:
        tokens = self.enc.encode(text)
        if len(tokens) <= self.MAX_OUTPUT_TOKENS:
            return text
        truncated = self.enc.decode(
            tokens[:self.MAX_OUTPUT_TOKENS]
        )
        return truncated + "\n[... output truncated ...]"

# -- Loop Detector --------------------------------------------

class LoopDetector:
    """Detects repeated actions within a sliding window."""
    def __init__(self, window: int = 5, max_repeats: int = 2):
        self.window      = window
        self.max_repeats = max_repeats
        self.action_hashes: list[str] = []

    def record(self, tool_name: str, args: dict) -> bool:
        """Returns True if a loop is detected."""
        h = hashlib.md5(
            f"{tool_name}:{json.dumps(args, sort_keys=True)}"
            .encode()
        ).hexdigest()
        self.action_hashes.append(h)
        recent = self.action_hashes[-self.window:]
        if recent.count(h) >= self.max_repeats:
            logger.warning("Loop detected: %s called %d times",
                           tool_name, recent.count(h))
            return True
        return False

# -- Agent Harness --------------------------------------------

class AgentHarness:
    """
    Production agent harness implementing the ReAct loop
    with full context management, tool integration,
    error handling, and observability.
    """
    MAX_ITERATIONS = 50

    def __init__(
        self,
        model:        str,
        system_prompt: str,
        tools:        list[ToolDefinition],
        max_tokens:   int = 128_000,
        approval_cb:  Optional[Callable] = None,
        client:       Optional[AsyncOpenAI] = None,
    ):
        self.model   = model
        self.client  = client or AsyncOpenAI()
        self.ctx_mgr = ContextManager(model, max_tokens)
        self.executor = ToolExecutor(tools, approval_cb)
        self.loop_det = LoopDetector()
        self.tools    = tools

        # Set system message
        sys_msg = Message(Role.SYSTEM, system_prompt)
        self.ctx_mgr.system_msg = sys_msg

    async def run(self, user_input: str) -> str:
        """
        Execute the ReAct loop for a user request.
        Returns the final response string.
        """
        run_id   = hashlib.md5(
            f"{time.time()}:{user_input}".encode()
        ).hexdigest()[:8]
        start_ts = time.monotonic()
        logger.info("[%s] Starting run: %s", run_id,
                    user_input[:80])

        # Add user message to context
        self.ctx_mgr.add_message(
            Message(Role.USER, user_input)
        )

        tool_defs = [t.to_api_dict() for t in self.tools]
        tool_tokens = sum(
            self.ctx_mgr.count_tokens(json.dumps(t))
            for t in tool_defs
        )

        for iteration in range(self.MAX_ITERATIONS):
            # Pre-flight context check
            if not self.ctx_mgr.preflight_check(tool_tokens):
                logger.error("[%s] Context overflow at iter %d",
                             run_id, iteration)
                return ("I've run out of context space. "
                        "Please start a new conversation.")

            # -- LLM Call ----------------------------------
            messages = self.ctx_mgr.build_messages()
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tool_defs if self.tools else None,
                    tool_choice="auto",
                    temperature=0.0,
                )
            except Exception as exc:
                logger.error("[%s] LLM call failed: %s",
                             run_id, exc)
                return f"I encountered an error: {exc}"

            choice  = response.choices[0]
            msg     = choice.message
            finish  = choice.finish_reason

            # Store assistant message
            assistant_msg = Message(
                role=Role.ASSISTANT,
                content=msg.content or "",
                tool_calls=([tc.model_dump()
                             for tc in msg.tool_calls]
                            if msg.tool_calls else None),
            )
            self.ctx_mgr.add_message(assistant_msg)

            # -- Terminal condition -------------------------
            if finish == "stop" or not msg.tool_calls:
                elapsed = time.monotonic() - start_ts
                logger.info(
                    "[%s] Done in %d iters, %.2fs",
                    run_id, iteration + 1, elapsed
                )
                return msg.content or "Task complete."

            # -- Tool Execution -----------------------------
            tool_results = await self._execute_tool_calls(
                msg.tool_calls, run_id
            )

            # Check for loops
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                if self.loop_det.record(tc.function.name, args):
                    return ("I seem to be stuck in a loop. "
                            "Please clarify your request.")

            # Add tool results to context
            for tool_call_id, result in tool_results.items():
                self.ctx_mgr.add_message(Message(
                    role=Role.TOOL,
                    content=result,
                    tool_call_id=tool_call_id,
                ))

        # Max iterations reached
        logger.warning("[%s] Max iterations reached", run_id)
        return ("I reached the maximum number of steps "
                "without completing the task. "
                "Here is what I found so far: "
                + (msg.content or ""))

    async def _execute_tool_calls(
        self,
        tool_calls: list,
        run_id: str,
    ) -> dict[str, str]:
        """Execute tool calls in parallel."""
        tasks = {}
        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            logger.info("[%s] Tool call: %s(%s)",
                        run_id, name, args)
            tasks[tc.id] = self.executor.execute(name, args)

        results = await asyncio.gather(
            *tasks.values(), return_exceptions=True
        )
        output = {}
        for tool_id, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                output[tool_id] = f"Error: {result}"
            else:
                output[tool_id] = result
        return output

# -- Example Usage --------------------------------------------

async def main():
    # Define tools
    async def search_web(query: str,
                         num_results: int = 5) -> str:
        # In production: call a real search API
        return f"[Search results for '{query}': ...]"

    async def run_python(code: str) -> str:
        # In production: execute in a sandbox container
        return f"[Execution result of code: ...]"

    tools = [
        ToolDefinition(
            name="search_web",
            description=(
                "Search the web for current information. "
                "Use when the user asks about recent events "
                "or facts beyond your knowledge cutoff."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "default": 5
                    },
                },
                "required": ["query"],
            },
            handler=search_web,
        ),
        ToolDefinition(
            name="run_python",
            description=(
                "Execute Python code in a sandbox. "
                "Use for calculations, data processing, "
                "or generating visualizations."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    },
                },
                "required": ["code"],
            },
            handler=run_python,
            requires_approval=True,  # Requires human sign-off
        ),
    ]

    harness = AgentHarness(
        model="gpt-4o",
        system_prompt=(
            "You are a helpful research assistant. "
            "Think step by step before acting. "
            "Always cite your sources."
        ),
        tools=tools,
        max_tokens=128_000,
    )

    response = await harness.run(
        "What were the key AI research breakthroughs "
        "in the first half of 2025?"
    )
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

<span class="hh-tag">Listing 25:</span>Production Agent Harness – Core Implementation

### Summary

The agent harness is the engineering foundation that transforms a language model into a capable, reliable agent. The key takeaways from this section are:

<ul><li>Context is a finite, precious resource. Enforce budgets explicitly, count tokens with the model’s exact tokenizer, and compress history proactively.</li><li>Prompts are code. Version-control them, test them, and assemble them modularly from components.</li><li>Tools are the agent’s actuators. Define them precisely, sandbox their execution, and handle their outputs defensively.</li><li>Orchestration patterns are not one-size-fits-all. ReAct for exploratory tasks, Plan-and-Execute for structured tasks, multi-agent for complex decomposable tasks.</li><li>State management is a first-class concern. Design state schemas upfront; retrofitting them is painful.</li><li>Errors are inevitable; graceful recovery is a feature. Implement retry logic, loop detection, and informative failure messages.</li><li>Observability is not optional. You cannot debug what you cannot see. Instrument everything from day one.</li><li>Production concerns compound. Latency, cost, rate limits, and evaluation all interact. Address them systematically, not as afterthoughts.</li></ul>

<aside class="hh-next">
<p class="hh-next-label">下一篇</p>
<p class="hh-next-title">Chapter 19 Loop Engineering</p>
<p class="hh-next-hint">在「智能体 AI 漫游指南」分类下继续阅读。</p>
</aside>

</div>
