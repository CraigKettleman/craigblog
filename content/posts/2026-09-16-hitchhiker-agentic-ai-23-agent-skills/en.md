---
title: "The Hitchhiker's Guide to Agentic AI · Chapter 23 Agent Skills"
slug: "hitchhiker-agentic-ai-23-agent-skills"
lang: "en"
date: "2026-09-16T00:24:00.000Z"
draft: false
featured: false
categories:
  - "hitchhiker-agentic-ai"
cover: "./cover.png"
description: "As agents evolve from monolithic prompt-and-tool systems into modular architectures, a key design question emerges: how should an agent’s ca…"
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

As agents evolve from monolithic prompt-and-tool systems into modular architectures, a key design question emerges: <em>how should an agent’s capabilities be organized, discovered, and composed?</em> The answer increasingly converges on the concept of skills — discrete, reusable units of behaviour that can be loaded, combined, and swapped without retraining.

The idea was popularized by Voyager [362], which demonstrated that an LLM agent in Minecraft could accumulate a growing library of executable code skills, each verified and stored for later reuse. The same principle applies to production agents: skills encapsulate domain expertise in a composable, versionable format that scales beyond what any single prompt can hold. Skills frequently wrap MCP servers (Chapter Chapter 22 Model Context Protocol (MCP)) for tool access, connecting the skill abstraction to the standardized tool layer.

## What Is a Skill?

A skill is a self-contained capability module that gives an agent expertise in a specific domain or task. Unlike a raw tool (which exposes a single function), a skill encompasses:

<ul><li>System prompt augmentation: Domain-specific instructions, constraints, and persona elements injected into the agent’s context.</li><li>Tool bindings: One or more tools the skill requires (APIs, MCP servers, local commands).</li><li>Knowledge: Reference material, examples, or few-shot demonstrations the agent needs to execute the skill correctly.</li><li>Workflow logic: Multi-step procedures, decision trees, or conditional flows that guide the agent through complex tasks.</li><li>Guardrails: Skill-specific safety constraints, output format requirements, and validation rules.</li></ul>

## Skill Architecture Patterns

### Static Skill Loading

The simplest pattern: skills are loaded at agent initialization based on configuration. The agent always has access to all its skills.

```python
# Pseudocode -- framework-agnostic pattern
agent = Agent(
    model="claude-sonnet-4-20250514",
    skills=["code-review", "documentation", "testing"],
    # Each skill adds prompts, tools, and knowledge to the agent
)
```

Pros: Simple, predictable, low latency. <br>

Cons: Context window waste when skills are unused; doesn’t scale to hundreds of skills.

### Dynamic Skill Discovery

The agent selects which skills to activate based on the current task. A skill router (often a lightweight classifier or embedding-based matcher) determines relevance:

```python
# Pseudocode -- framework-agnostic pattern
relevant_skills = skill_router.match(
    user_request=message,
    available_skills=skill_registry,
    max_skills=3
)
agent.activate(relevant_skills)
```

Pros: Scales to large skill libraries; context-efficient. <br>

Cons: Routing errors can miss relevant skills; adds latency.

### Hierarchical Skill Composition

Skills can depend on other skills, forming a DAG. A high-level skill (e.g., “Deploy Application”) may invoke sub-skills (“Run Tests”, “Build Docker Image”, “Update DNS”):

<ul><li>Skills declare dependencies explicitly</li><li>The orchestrator resolves the dependency graph before execution</li><li>Sub-skills can be shared across multiple parent skills</li></ul>

## Case Study: Anthropic’s Agent Design

Anthropic’s approach to agent architecture [9] provides one of the clearest articulations of skill-based agent design in production. Their philosophy emphasizes simplicity over complexity and composable building blocks over monolithic frameworks. (These patterns are also covered from an orchestration perspective in Chapter Chapter 20 Agent Design Patterns.)

### Core Principles

<ol><li>Start with the simplest solution. Don’t reach for agentic patterns until simpler approaches (single LLM call, retrieval + generation) have been tried and found insufficient.</li><li>Workflows vs. agents. Anthropic distinguishes between:<span class="hh-tag">•</span>Workflows: Predefined orchestration of LLM calls — deterministic control flow with LLM steps at specific nodes. More predictable, easier to debug.<span class="hh-tag">•</span>Agents: The LLM dynamically decides what to do next — tool selection, iteration count, and stopping criteria are all model-driven. More flexible, harder to control.</li><li>Augmented LLM as the atomic unit. The primitive is never a bare model—it is always a model bundled with its retrieval sources, callable tools, and persistent memory. This composite unit is, in practice, a skill-equipped model.</li></ol>

### Building Block Patterns

Anthropic identifies five composable workflow patterns that function as skill templates:

<div class="hh-table" id="ch23.t1"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Pattern</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Mechanism</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>When to Use</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left"><span>Prompt Chaining</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Sequential LLM calls where each step’s output feeds the next. Gates between steps validate intermediate results.</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Multi-step transformations with clear decomposition</span></span></td></tr><tr><th class="ltx_align_left"><span>Routing</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>A classifier or LLM directs input to a specialized handler (skill) based on task type.</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Distinct task categories requiring different expertise</span></span></td></tr><tr><th class="ltx_align_left"><span>Parallelization</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Multiple LLM calls run simultaneously — either sectioning (split task) or voting (same task, aggregate).</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Independent subtasks; or confidence via consensus</span></span></td></tr><tr><th class="ltx_align_left"><span>Orchestrator–Workers</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>A central LLM breaks the task into subtasks, delegates to worker LLMs, then synthesizes results.</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Complex tasks where subtasks aren’t predictable in advance</span></span></td></tr><tr><th class="ltx_align_left"><span>Evaluator–Optimizer</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>One LLM generates, another evaluates; iterate until quality threshold is met.</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Tasks with clear quality criteria (code, writing)</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 23.1:</span>Anthropic’s composable agent patterns.</p></div>

### The Augmented LLM

In Anthropic’s framing, the fundamental unit is not the bare model but the augmented LLM:

<div class="hh-equation" id="ch23.ex1"><math alttext="\text{Augmented LLM}=\text{Model}+\text{Retrieval}+\text{Tools}+\text{Memory}" display="block"><semantics><mrow><mtext>Augmented LLM</mtext><mo>=</mo><mrow><mtext>Model</mtext><mo>+</mo><mtext>Retrieval</mtext><mo>+</mo><mtext>Tools</mtext><mo>+</mo><mtext>Memory</mtext></mrow></mrow></semantics></math></div>

This maps directly to the skill concept: each skill configures which retrieval sources, tools, and memory stores the model has access to for a specific task. The skill boundary defines what the model <em>can see and do</em> within a particular invocation.

### Practical Implications

<ul><li>Keep agent loops simple: Avoid over-engineering the control flow. Let the model decide.</li><li>Invest in tool quality: Detailed, unambiguous tool descriptions are more valuable than complex routing logic.</li><li>Use structured outputs: Force the model to output decisions in parseable formats (JSON, function calls) — reduces skill execution errors.</li><li>Build in recovery: Skills should handle errors gracefully — retry with different parameters, ask for clarification, or escalate to a human.</li><li>Limit scope per skill: A skill that tries to do everything will do nothing well. Narrow, well-defined skills compose better than broad ones.</li></ul>

## Skill Lifecycle

<ol><li>Discovery: The system identifies which skills are available (registry, marketplace, local definitions).</li><li>Selection: Based on the user request, relevant skills are matched and loaded.</li><li>Activation: Skill prompts, tools, and knowledge are injected into the agent’s context.</li><li>Execution: The agent uses the skill’s capabilities to accomplish the task.</li><li>Deactivation: Skill context is removed to free context window space for subsequent tasks.</li><li>Learning: Execution results may update the skill’s few-shot examples or fine-tune routing.</li></ol>

## Skill Registries and Marketplaces

Production skill systems require infrastructure:

<ul><li>Skill manifest: A structured description (name, capabilities, required tools, input/output schema) enabling automatic discovery and routing.</li><li>Version control: Skills evolve; agents need to pin specific versions for reproducibility.</li><li>Dependency resolution: Skills may require specific MCP servers, API keys, or other skills.</li><li>Permission model: Not all agents should have access to all skills (security, cost, capability boundaries).</li><li>Marketplace: Organizations can publish, share, and install skills — analogous to package managers for code.</li></ul>

## Skills vs. Fine-Tuning

A natural question: why use runtime skill injection instead of fine-tuning the model?

<div class="hh-table" id="ch23.t2"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Dimension</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Skills (In-Context)</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Fine-Tuning</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">Deployment speed</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Instant</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Hours–days</span></span></td></tr><tr><th class="ltx_align_left">Flexibility</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Swap/combine at runtime</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Fixed at training time</span></span></td></tr><tr><th class="ltx_align_left">Context cost</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Uses context window</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Zero runtime cost</span></span></td></tr><tr><th class="ltx_align_left">Deep behavior change</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Limited by context length</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Deep parametric change</span></span></td></tr><tr><th class="ltx_align_left">Multi-tenant</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Different skills per user</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Same model for all</span></span></td></tr><tr><th class="ltx_align_left">Maintenance</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Update text files</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Retrain on new data</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 23.2:</span>Skills (in-context) vs. fine-tuning for adding capabilities.</p></div>

In practice, the two approaches are complementary: fine-tuning provides <em>base capabilities</em> (instruction following, tool use format, reasoning), while skills provide <em>task-specific expertise</em> layered on top at runtime.

<aside class="hh-next">
<p class="hh-next-label">下一篇</p>
<p class="hh-next-title">The Hitchhiker's Guide to Agentic AI · Chapter 24 Agent-to-Agent Communication (A2A)</p>
<p class="hh-next-hint">在「智能体 AI 漫游指南」分类下继续阅读。</p>
</aside>

</div>
