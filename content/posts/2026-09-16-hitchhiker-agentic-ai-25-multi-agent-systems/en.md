---
title: "The Hitchhiker's Guide to Agentic AI · Chapter 25 Multi-Agent Systems"
slug: "hitchhiker-agentic-ai-25-multi-agent-systems"
lang: "en"
date: "2026-09-16T00:26:00.000Z"
draft: false
featured: false
categories:
  - "hitchhiker-agentic-ai"
cover: "./cover.png"
description: "The history of artificial intelligence is, in many ways, a history of scale.…"
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

## Motivation: Why Multiple Agents?

The history of artificial intelligence is, in many ways, a history of scale. Early AI systems were monolithic: a single program, a single knowledge base, a single inference engine. As problems grew more complex, researchers discovered that no single agent—however capable—could efficiently handle every aspect of a rich, open-ended task. This insight, long established in distributed AI and multi-agent systems (MAS) research [375, 382], has found renewed urgency in the era of large language models.

Four fundamental motivations drive the shift from monolithic agents to <em>agent societies</em>:

Different sub-tasks benefit from different capabilities, prompting strategies, and even different base models. A code-generation agent can be fine-tuned on programming corpora; a fact-checking agent can be grounded with retrieval tools; a creative-writing agent can be prompted for stylistic diversity. Forcing a single agent to excel at all of these simultaneously is both inefficient and often impossible.

Many real-world tasks decompose into independent sub-tasks that can be executed concurrently. A research pipeline that requires literature review, data analysis, and report writing can run all three in parallel, dramatically reducing wall-clock time. Sequential single-agent processing is a bottleneck that multi-agent parallelism eliminates.

A single agent is a single point of failure. If it hallucinates, gets stuck in a loop, or produces a subtly wrong answer, there is no check. Multi-agent systems introduce redundancy: a second agent can verify, critique, or independently re-derive results. Adversarial agents can probe for weaknesses before outputs are trusted.

Perhaps most intriguingly, agent collectives can exhibit capabilities that no individual agent possesses. Through debate, negotiation, and iterative refinement, multi-agent systems can arrive at solutions that transcend what any single agent could produce alone—a computational analog to the emergent intelligence of social organisms.

The transition from monolithic agents to agent societies mirrors a broader pattern in complex systems: as the problem space grows, distributed, modular architectures consistently outperform centralized, monolithic ones. The question is no longer <em>whether</em> to use multiple agents, but <em>how</em> to organize them.

## Multi-Agent Architectures

The topology of a multi-agent system—how agents are connected and how authority flows among them—is the most consequential architectural decision. Four canonical patterns have emerged, each with distinct trade-offs.

### Centralized (Supervisor/Manager) Architecture

In a centralized architecture, a single <em>orchestrator</em> agent (variously called supervisor, manager, or planner) holds global state, decomposes tasks, delegates sub-tasks to worker agents, and aggregates their results. The topology is a hub-and-spoke: all communication flows through the central node.

<figure id="ch25.f1"><img src="./fig_069_centralized-arch.png" alt="Figure 25.1: Centralized (Supervisor) architecture. The manager delegates tasks to specialized workers and aggregates th" loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 25.1:</span>Centralized (Supervisor) architecture. The manager delegates tasks to specialized workers and aggregates their outputs. All communication flows through the central hub.</figcaption></figure>

The manager’s responsibilities include:

<ul><li>Task routing: deciding which worker is best suited for each sub-task</li><li>Context management: providing each worker with the relevant subset of global context</li><li>Result aggregation: synthesizing worker outputs into a coherent whole</li><li>Error handling: detecting worker failures and re-routing or retrying</li></ul>

### Decentralized (Peer-to-Peer) Architecture

In a decentralized architecture, agents interact directly with one another without a central coordinator. The topology is a mesh: any agent can communicate with any other. Coordination emerges from local interactions rather than global planning.

<figure id="ch25.f2"><img src="./fig_070_decentralized-arch.png" alt="Figure 25.2: Decentralized (peer-to-peer) architecture. Agents communicate directly; coordination emerges from local int" loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 25.2:</span>Decentralized (peer-to-peer) architecture. Agents communicate directly; coordination emerges from local interactions.</figcaption></figure>

Emergent coordination in peer-to-peer systems arises through mechanisms such as:

<ul><li>Negotiation: agents bid for tasks or resources</li><li>Stigmergy: agents modify shared state that others observe (see Section 25.3.6 Stigmergy: Indirect Communication Through Environment)</li><li>Gossip protocols: agents propagate information through the network</li><li>Local consensus: small groups of agents reach agreement without global coordination</li></ul>

### Hierarchical Architecture

Hierarchical architectures generalize the centralized pattern into a tree structure with multiple levels of management. A top-level orchestrator delegates to domain-specific sub-managers, who in turn delegate to specialized workers. This mirrors the organizational structure of large enterprises.

<figure id="ch25.f3"><img src="./fig_071_hierarchical-arch.png" alt="Figure 25.3: Hierarchical architecture. A top-level orchestrator delegates to domain sub-managers, who delegate to speci" loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 25.3:</span>Hierarchical architecture. A top-level orchestrator delegates to domain sub-managers, who delegate to specialized workers. Dashed arrow shows an escalation path.</figcaption></figure>

Key features of hierarchical systems:

<ul><li>Delegation chains: authority and context flow down the tree; results flow up</li><li>Escalation paths: workers can escalate unresolvable issues to their manager</li><li>Domain isolation: sub-managers maintain domain-specific context, reducing the cognitive load on the top-level orchestrator</li><li>Scope limitation: each agent only needs to know about its immediate superiors and subordinates</li></ul>

The enterprise analogy is apt: a CEO (top orchestrator) sets strategy; VPs (sub-managers) translate strategy into domain plans; individual contributors (workers) execute. The hierarchy enables scale while preserving accountability.

### Swarm Architecture

Swarm architectures, inspired by biological systems (ant colonies, bird flocking), consist of many loosely coupled agents that follow simple local rules, producing complex global behavior without any central coordinator or global state.

OpenAI’s Swarm framework [276] (now superseded by the OpenAI Agents SDK, but its conceptual primitives remain influential) operationalizes this with two primitives:

<ul><li>Routines: sequences of instructions an agent follows to complete a sub-task</li><li>Handoffs: an agent transferring control (and relevant context) to another agent</li></ul>

## Coordination Mechanisms

How agents coordinate—how they share information, divide work, and resolve conflicts—is as important as the topology. Six canonical coordination mechanisms apply to LLM-based multi-agent systems.

### Shared State (Global Blackboard)

The blackboard architecture[135] provides a shared data structure that all agents can read from and write to. In LLM systems, this is typically implemented as a shared dictionary, database, or structured document.

```
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

@dataclass
class BlackboardEntry:
    value: Any
    author: str
    timestamp: float
    confidence: float = 1.0

class Blackboard:
    """Thread-safe shared state for multi-agent coordination."""

    def __init__(self):
        self._data: Dict[str, BlackboardEntry] = {}
        self._lock = threading.RLock()
        self._subscribers: Dict[str, List[Callable]] = {}

    def write(self, key: str, value: Any, author: str,
              confidence: float = 1.0) -> bool:
        """Write to blackboard; higher-confidence entries win conflicts."""
        with self._lock:
            existing = self._data.get(key)
            if existing and existing.confidence > confidence:
                return False  # Conflict: existing entry wins
            import time
            self._data[key] = BlackboardEntry(
                value=value, author=author,
                timestamp=time.time(), confidence=confidence
            )
            self._notify(key, value)
            return True

    def read(self, key: str) -> Any:
        with self._lock:
            entry = self._data.get(key)
            return entry.value if entry else None

    def subscribe(self, key: str, callback: Callable):
        """Agents subscribe to changes on specific keys."""
        self._subscribers.setdefault(key, []).append(callback)

    def _notify(self, key: str, value: Any):
        for cb in self._subscribers.get(key, []):
            cb(key, value)
```

<span class="hh-tag">Listing 33:</span>Shared blackboard with conflict resolution

### Message Passing

Message passing is the most natural coordination mechanism for LLM agents: agents communicate by sending structured text messages to one another. Key design decisions include:

<ul><li>Message format: structured (JSON schema) vs. natural language vs. hybrid</li><li>Routing: direct (agent-to-agent) vs. broadcast vs. topic-based pub/sub</li><li>Conversation threads: maintaining context across multi-turn exchanges</li><li>Acknowledgment: whether senders require confirmation of receipt/processing</li></ul>

### Planning and Decomposition

A manager agent decomposes a high-level task into a directed acyclic graph (DAG) of sub-tasks, assigns each to an appropriate worker, and tracks dependencies. This is the multi-agent analog of classical hierarchical task network (HTN) planning.

```
from dataclasses import dataclass, field
from typing import List, Optional
import asyncio

@dataclass
class Task:
    id: str
    description: str
    assigned_to: str
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"   # pending | running | done | failed
    result: Optional[str] = None

class TaskDAG:
    def __init__(self):
        self.tasks: dict[str, Task] = {}

    def add_task(self, task: Task):
        self.tasks[task.id] = task

    def ready_tasks(self) -> List[Task]:
        """Return tasks whose dependencies are all completed."""
        return [
            t for t in self.tasks.values()
            if t.status == "pending"
            and all(self.tasks[d].status == "done"
                    for d in t.dependencies)
        ]

    async def execute(self, agent_pool: dict):
        while any(t.status != "done" for t in self.tasks.values()):
            ready = self.ready_tasks()
            if not ready:
                await asyncio.sleep(0.1)
                continue
            # Execute ready tasks in parallel
            await asyncio.gather(*[
                self._run_task(t, agent_pool[t.assigned_to])
                for t in ready
            ])

    async def _run_task(self, task: Task, agent):
        task.status = "running"
        try:
            task.result = await agent.execute(task.description)
            task.status = "done"
        except Exception as e:
            task.status = "failed"
            raise
```

<span class="hh-tag">Listing 34:</span>Task DAG decomposition

### Voting and Consensus

When multiple agents produce conflicting outputs, voting mechanisms aggregate their responses into a single decision. Common schemes include:

<ul><li>Majority voting: the most common answer wins; effective for factual questions</li><li>Weighted voting: agents with higher track records or confidence scores receive more weight</li><li>Debate-based resolution: agents argue for their positions; a judge agent decides</li><li>Delphi method: iterative rounds where agents revise their answers after seeing others’ reasoning</li></ul>

Formally, given <math alttext="n" display="inline"><semantics><mi>n</mi></semantics></math> agents producing outputs <math alttext="\{o_{1},\ldots,o_{n}\}" display="inline"><semantics><mrow><mo stretchy="false">{</mo><msub><mi>o</mi><mn>1</mn></msub><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msub><mi>o</mi><mi>n</mi></msub><mo stretchy="false">}</mo></mrow></semantics></math> with weights <math alttext="\{w_{1},\ldots,w_{n}\}" display="inline"><semantics><mrow><mo stretchy="false">{</mo><msub><mi>w</mi><mn>1</mn></msub><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msub><mi>w</mi><mi>n</mi></msub><mo stretchy="false">}</mo></mrow></semantics></math>, the weighted consensus is:

<div class="hh-equation" id="ch25.e1"><math alttext="o^{*}=\arg\max_{o}\sum_{i=1}^{n}w_{i}\cdot\mathbf{1}[o_{i}=o]" display="block"><semantics><mrow><msup><mi>o</mi><mo>∗</mo></msup><mo>=</mo><mi>arg</mi><munder><mi>max</mi><mi>o</mi></munder><munderover><mo movablelimits="false">∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><msub><mi>w</mi><mi>i</mi></msub><mo lspace="0.222em" rspace="0.222em">⋅</mo><mn>𝟏</mn><mrow><mo stretchy="false">[</mo><msub><mi>o</mi><mi>i</mi></msub><mo>=</mo><mi>o</mi><mo stretchy="false">]</mo></mrow></mrow></semantics></math><span class="hh-equation-number">(25.1)</span></div>

For continuous outputs (e.g., probability estimates), weighted averaging applies:

<div class="hh-equation" id="ch25.e2"><math alttext="\hat{p}=\frac{\sum_{i=1}^{n}w_{i}\cdot p_{i}}{\sum_{i=1}^{n}w_{i}}" display="block"><semantics><mrow><mover accent="true"><mi>p</mi><mo>^</mo></mover><mo>=</mo><mfrac><mrow><mrow><msubsup><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></msubsup><msub><mi>w</mi><mi>i</mi></msub></mrow><mo lspace="0.222em" rspace="0.222em">⋅</mo><msub><mi>p</mi><mi>i</mi></msub></mrow><mrow><msubsup><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></msubsup><msub><mi>w</mi><mi>i</mi></msub></mrow></mfrac></mrow></semantics></math><span class="hh-equation-number">(25.2)</span></div>

### Market-Based Coordination

Market mechanisms allocate tasks and resources through auctions and bidding. The Contract Net Protocol [334], one of the oldest multi-agent coordination mechanisms, is a task auction:

<ol><li>A <em>manager</em> broadcasts a task announcement with requirements</li><li><em>Contractor</em> agents submit bids (capability declarations + cost estimates)</li><li>The manager awards the contract to the best bidder</li><li>The winning contractor executes and reports results</li></ol>

In LLM systems, bids can be expressed in natural language (“I can complete this in 3 steps with high confidence”) or structured formats. Market mechanisms are particularly effective for resource-constrained settings where API costs must be minimized.

### Stigmergy: Indirect Communication Through Environment

Stigmergy[117] replaces explicit agent-to-agent messaging with a simpler mechanism: each agent modifies the shared environment as a side effect of its work, and other agents react to those modifications rather than to direct signals. The classic illustration is a foraging ant depositing pheromone on its return path; subsequent ants amplify successful routes without any ant “talking” to another.

In LLM multi-agent systems, stigmergy manifests as:

<ul><li>Shared documents: agents write to a shared document; others read and build upon it</li><li>Code repositories: one agent commits code; another reads and extends it</li><li>Annotation layers: agents annotate shared artifacts (highlight errors, add comments)</li><li>Task queues: agents add and consume tasks from a shared queue</li></ul>

Stigmergy enables coordination without explicit communication overhead—agents simply observe the state of the shared environment and act accordingly.

## Communication Protocols

Effective multi-agent systems require well-defined communication protocols: agreed-upon formats, semantics, and patterns for agent-to-agent messages. (For the standardized inter-agent protocol, see Chapter Chapter 24 Agent-to-Agent Communication (A2A).)

### Structured Message Formats

Messages between LLM agents should be structured to enable reliable parsing and routing. A minimal message schema:

```
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any
from datetime import datetime, timezone
import uuid

PerformativeType = Literal[
    "inform",    # Share information
    "request",   # Request an action
    "propose",   # Propose a course of action
    "accept",    # Accept a proposal
    "reject",    # Reject a proposal
    "query",     # Ask a question
    "confirm",   # Confirm receipt/completion
    "failure",   # Report a failure
]

class AgentMessage(BaseModel):
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str          # Groups related messages
    sender: str                   # Agent identifier
    receiver: str                 # Target agent (or "broadcast")
    performative: PerformativeType
    content: str                  # Natural language content
    metadata: Dict[str, Any] = {} # Structured payload
    reply_to: Optional[str] = None  # message_id being replied to
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_llm_prompt(self) -> str:
        """Render message as a prompt fragment for the receiving agent."""
        return (
            f"[MESSAGE from {self.sender}]\n"
            f"Type: {self.performative}\n"
            f"Content: {self.content}\n"
            + (f"Metadata: {self.metadata}\n" if self.metadata else "")
        )
```

<span class="hh-tag">Listing 35:</span>Agent message schema

### Performative Types (FIPA-ACL Inspired)

Drawing from the FIPA Agent Communication Language [93], modernized for LLM agents:

<div class="hh-table" id="ch25.t1"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Performative</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Semantics</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Example Use</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left"><span>inform</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Sender believes <math alttext="\phi" display="inline"><semantics><mi>ϕ</mi><span></span></semantics></math> is true</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Share research findings</span></span></td></tr><tr><th class="ltx_align_left"><span>request</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Sender wants receiver to do <math alttext="\alpha" display="inline"><semantics><mi>α</mi><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Delegate a sub-task</span></span></td></tr><tr><th class="ltx_align_left"><span>propose</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Sender proposes plan <math alttext="\pi" display="inline"><semantics><mi>π</mi><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Suggest an approach</span></span></td></tr><tr><th class="ltx_align_left"><span>accept</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Receiver agrees to proposal</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Confirm task assignment</span></span></td></tr><tr><th class="ltx_align_left"><span>reject</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Receiver declines proposal</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Refuse incompatible task</span></span></td></tr><tr><th class="ltx_align_left"><span>query</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Sender wants to know <math alttext="\phi" display="inline"><semantics><mi>ϕ</mi><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Ask for clarification</span></span></td></tr><tr><th class="ltx_align_left"><span>confirm</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Sender confirms <math alttext="\phi" display="inline"><semantics><mi>ϕ</mi><span></span></semantics></math> occurred</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Acknowledge completion</span></span></td></tr><tr><th class="ltx_align_left"><span>failure</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Sender failed to achieve <math alttext="\alpha" display="inline"><semantics><mi>α</mi><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Report error</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 25.1:</span>FIPA-ACL-inspired performative types for LLM agent messages.</p></div>

### Context Sharing Strategies

A critical challenge in multi-agent communication is context management: how much history does each agent need? Three strategies:

<ul><li>Full history: pass the entire conversation history to each agent. Maximally informative but expensive; context windows fill quickly.</li><li>Summary: a summarizer agent condenses prior exchanges into a compact summary. Efficient but lossy; important details may be dropped.</li><li>Relevant excerpt: retrieve only the most relevant prior messages using semantic search. Balances cost and informativeness; requires a retrieval mechanism.</li></ul>

## Role Design and Specialization

The design of agent roles—their capabilities, personas, and responsibilities—is as much an art as a science. Well-designed roles enable specialization; poorly designed roles create confusion and redundancy.

### Defining Agent Roles

Common roles in LLM multi-agent systems:

<div class="hh-table" id="ch25.t2"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Role</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Primary Capability</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Typical Tools</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">Researcher</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Information gathering, synthesis</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Web search, RAG, databases</span></span></td></tr><tr><th class="ltx_align_left">Planner</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Task decomposition, scheduling</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>None (reasoning only)</span></span></td></tr><tr><th class="ltx_align_left">Coder</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Code generation, debugging</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Code interpreter, linter</span></span></td></tr><tr><th class="ltx_align_left">Reviewer</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Quality assessment, critique</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>None (reasoning only)</span></span></td></tr><tr><th class="ltx_align_left">Tester</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Test generation, execution</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Test runner, coverage tools</span></span></td></tr><tr><th class="ltx_align_left">Writer</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Prose generation, editing</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Grammar checker, style guide</span></span></td></tr><tr><th class="ltx_align_left">Critic</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Adversarial evaluation</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>None (reasoning only)</span></span></td></tr><tr><th class="ltx_align_left">Orchestrator</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Coordination, delegation</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>All agent interfaces</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 25.2:</span>Common agent roles in LLM multi-agent systems.</p></div>

### Capability-Based vs. Role-Based Assignment

Two philosophies for task assignment:

<ul><li>Role-based: tasks are assigned based on predefined role labels. Simple and predictable; may be suboptimal when a task spans multiple roles.</li><li>Capability-based: tasks are assigned based on a dynamic assessment of each agent’s capabilities relative to the task requirements. More flexible; requires a capability registry and matching mechanism.</li></ul>

### Dynamic Role Reassignment

In long-running systems, static role assignments become suboptimal. Dynamic reassignment allows agents to take on new roles based on:

<ul><li>Current workload (load balancing)</li><li>Demonstrated performance on recent tasks</li><li>Changing task requirements</li><li>Agent failures requiring coverage</li></ul>

### Persona Design for Diversity of Thought

A subtle but powerful technique: give agents distinct personas that encourage diverse perspectives. Rather than five identical “assistant” agents, design:

<ul><li>An <em>optimist</em> who emphasizes opportunities</li><li>A <em>skeptic</em> who challenges assumptions</li><li>A <em>pragmatist</em> who focuses on implementation</li><li>A <em>visionary</em> who thinks long-term</li><li>A <em>devil’s advocate</em> who argues the opposite position</li></ul>

This diversity of thought, inspired by techniques like Six Thinking Hats [28], reduces groupthink and produces more robust collective reasoning.

## Multi-Agent Patterns for LLMs

Beyond architectural topologies, several interaction patterns have proven particularly effective for LLM-based multi-agent systems. (These complement the single-agent design patterns in Chapter Chapter 20 Agent Design Patterns.)

### Debate Pattern

Multiple agents argue for different positions; a judge agent evaluates the arguments and decides. Debate has been shown to improve factual accuracy and reduce hallucinations [78].

```
async def debate_round(question: str, agents: list, judge: Agent,
                       rounds: int = 2) -> str:
    """Run a multi-agent debate and return the judge's verdict."""
    positions = {a.name: await a.generate_position(question)
                 for a in agents}

    for round_num in range(rounds):
        # Each agent sees others' positions and can rebut
        rebuttals = {}
        for agent in agents:
            others = {k: v for k, v in positions.items()
                      if k != agent.name}
            rebuttals[agent.name] = await agent.rebut(
                question, positions[agent.name], others
            )
        positions = rebuttals

    # Judge evaluates all final positions
    verdict = await judge.evaluate(question, positions)
    return verdict
```

<span class="hh-tag">Listing 36:</span>Debate pattern implementation

### Reflection Pattern

One agent generates an output; a second agent critiques it; the first agent revises based on the critique. This implements a generate-critique-revise loop that iteratively improves quality.

```
async def reflection_loop(task: str, generator: Agent,
                          critic: Agent, max_rounds: int = 3) -> str:
    draft = await generator.generate(task)

    for _ in range(max_rounds):
        critique = await critic.critique(task, draft)
        if critique.is_satisfactory:
            break
        draft = await generator.revise(task, draft, critique.feedback)

    return draft
```

<span class="hh-tag">Listing 37:</span>Reflection pattern

### Division of Labor Pattern

The task is decomposed into independent sub-tasks executed in parallel. Results are aggregated by a synthesis agent. This pattern maximizes throughput for embarrassingly parallel tasks.

### Pipeline Pattern

Agents form a sequential processing chain: each agent transforms the output of the previous agent. Analogous to Unix pipes. Effective for tasks with clear sequential dependencies (e.g., research <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> outline <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> draft <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> edit <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> format).

### Ensemble Pattern

Multiple agents independently solve the same problem; a selection mechanism picks the best answer (best-of-<math alttext="N" display="inline"><semantics><mi>N</mi></semantics></math>) or aggregates answers (mixture-of-experts style). Improves reliability at the cost of compute.

<div class="hh-equation" id="ch25.e3"><math alttext="o^{*}=\arg\max_{o\in\{o_{1},\ldots,o_{N}\}}\text{score}(o,\text{task})" display="block"><semantics><mrow><msup><mi>o</mi><mo>∗</mo></msup><mo>=</mo><mrow><mrow><mi>arg</mi><mo lspace="0.167em">⁡</mo><munder><mi>max</mi><mrow><mi>o</mi><mo>∈</mo><mrow><mo stretchy="false">{</mo><msub><mi>o</mi><mn>1</mn></msub><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msub><mi>o</mi><mi>N</mi></msub><mo stretchy="false">}</mo></mrow></mrow></munder></mrow><mo lspace="0.167em" rspace="0em">​</mo><mtext>score</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>o</mi><mo>,</mo><mtext>task</mtext><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(25.3)</span></div>

where score can be a reward model, a judge LLM, or a verifier.

### Teacher-Student Pattern

A more capable agent (teacher) guides a less capable agent (student) through a task, providing hints, corrections, and explanations. This pattern enables knowledge distillation at inference time and can be used to fine-tune the student agent.

### Red Team Pattern

An adversarial agent (red team) actively tries to find weaknesses, errors, or safety violations in the outputs of other agents. The red team agent is prompted to be maximally critical and creative in its attacks. This pattern is essential for safety-critical applications.

## Training Multi-Agent Systems with Reinforcement Learning

Training multi-agent systems with RL introduces challenges that go beyond single-agent RL. The fundamental difficulty is that each agent’s environment includes other learning agents, making the environment non-stationary from any single agent’s perspective.

### Mathematical Formulation

A multi-agent system is formalized as a Markov Game (also called a stochastic game) [324]:

<div class="hh-equation" id="ch25.e4"><math alttext="\mathcal{G}=\langle\mathcal{N},\mathcal{S},\{\mathcal{A}^{i}\}_{i\in\mathcal{N}},\mathcal{T},\{R^{i}\}_{i\in\mathcal{N}},\gamma\rangle" display="block"><semantics><mrow><mi>𝒢</mi><mo>=</mo><mrow><mo stretchy="false">⟨</mo><mrow><mi>𝒩</mi><mo>,</mo><mi>𝒮</mi><mo>,</mo><msub><mrow><mo stretchy="false">{</mo><msup><mi>𝒜</mi><mi>i</mi></msup><mo stretchy="false">}</mo></mrow><mrow><mi>i</mi><mo>∈</mo><mi>𝒩</mi></mrow></msub><mo>,</mo><mi>𝒯</mi><mo>,</mo><msub><mrow><mo stretchy="false">{</mo><msup><mi>R</mi><mi>i</mi></msup><mo stretchy="false">}</mo></mrow><mrow><mi>i</mi><mo>∈</mo><mi>𝒩</mi></mrow></msub><mo>,</mo><mi>γ</mi></mrow><mo stretchy="false">⟩</mo></mrow></mrow></semantics></math><span class="hh-equation-number">(25.4)</span></div>

where <math alttext="\mathcal{N}=\{1,\ldots,n\}" display="inline"><semantics><mrow><mi>𝒩</mi><mo>=</mo><mrow><mo stretchy="false">{</mo><mrow><mn>1</mn><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><mi>n</mi></mrow><mo stretchy="false">}</mo></mrow></mrow></semantics></math> is the set of agents, <math alttext="\mathcal{S}" display="inline"><semantics><mi>𝒮</mi></semantics></math> is the shared state space, <math alttext="\mathcal{A}^{i}" display="inline"><semantics><msup><mi>𝒜</mi><mi>i</mi></msup></semantics></math> is agent <math alttext="i" display="inline"><semantics><mi>i</mi></semantics></math>’s action space, <math alttext="\mathcal{T}:\mathcal{S}\times\mathcal{A}^{1}\times\cdots\times\mathcal{A}^{n}\to\Delta(\mathcal{S})" display="inline"><semantics><mrow><mi>𝒯</mi><mo lspace="0.278em" rspace="0.278em">:</mo><mi>𝒮</mi><mo lspace="0.222em" rspace="0.222em">×</mo><msup><mi>𝒜</mi><mn>1</mn></msup><mo lspace="0.222em">×</mo><mo lspace="0.055em" rspace="0.055em">⋯</mo><mo rspace="0.222em">×</mo><msup><mi>𝒜</mi><mi>n</mi></msup><mo stretchy="false">→</mo><mi mathvariant="normal">Δ</mi><mrow><mo stretchy="false">(</mo><mi>𝒮</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math> is the transition function, <math alttext="R^{i}:\mathcal{S}\times\mathcal{A}^{1}\times\cdots\times\mathcal{A}^{n}\to\mathbb{R}" display="inline"><semantics><mrow><mi>R</mi><msup><mrow></mrow><mi>i</mi></msup><mo lspace="0.278em" rspace="0.278em">:</mo><mi>𝒮</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mi>𝒜</mi><msup><mrow></mrow><mn>1</mn></msup><mo lspace="0.222em">×</mo><mo lspace="0.055em" rspace="0.055em">⋯</mo><mo rspace="0.222em">×</mo><mi>𝒜</mi><msup><mrow></mrow><mi>n</mi></msup><mo stretchy="false">→</mo><mi>ℝ</mi></mrow></semantics></math> is agent <math alttext="i" display="inline"><semantics><mi>i</mi></semantics></math>’s reward function, and <math alttext="\gamma" display="inline"><semantics><mi>γ</mi></semantics></math> is the discount factor.

Each agent <math alttext="i" display="inline"><semantics><mi>i</mi></semantics></math> seeks to maximize its expected discounted return:

<div class="hh-equation" id="ch25.e5"><math alttext="J^{i}(\pi^{1},\ldots,\pi^{n})=\mathbb{E}_{\pi^{1},\ldots,\pi^{n}}\left[\sum_{t=0}^{\infty}\gamma^{t}R^{i}(s_{t},a_{t}^{1},\ldots,a_{t}^{n})\right]" display="block"><semantics><mrow><mrow><msup><mi>J</mi><mi>i</mi></msup><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msup><mi>π</mi><mn>1</mn></msup><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msup><mi>π</mi><mi>n</mi></msup><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><msub><mi>𝔼</mi><mrow><msup><mi>π</mi><mn>1</mn></msup><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msup><mi>π</mi><mi>n</mi></msup></mrow></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo>[</mo><mrow><munderover><mo lspace="0em" movablelimits="false">∑</mo><mrow><mi>t</mi><mo>=</mo><mn>0</mn></mrow><mi mathvariant="normal">∞</mi></munderover><mrow><msup><mi>γ</mi><mi>t</mi></msup><mo lspace="0em" rspace="0em">​</mo><msup><mi>R</mi><mi>i</mi></msup><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>s</mi><mi>t</mi></msub><mo>,</mo><msubsup><mi>a</mi><mi>t</mi><mn>1</mn></msubsup><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msubsup><mi>a</mi><mi>t</mi><mi>n</mi></msubsup><mo stretchy="false">)</mo></mrow></mrow></mrow><mo>]</mo></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(25.5)</span></div>

### Independent Learning

The simplest approach: each agent <math alttext="i" display="inline"><semantics><mi>i</mi></semantics></math> treats other agents as part of its environment and optimizes its own policy <math alttext="\pi^{i}" display="inline"><semantics><msup><mi>π</mi><mi>i</mi></msup></semantics></math> independently using standard single-agent RL (e.g., PPO, REINFORCE).

<div class="hh-equation" id="ch25.e6"><math alttext="\nabla_{\theta^{i}}J^{i}\approx\mathbb{E}\left[\nabla_{\theta^{i}}\log\pi^{i}(a^{i}_{t}|o^{i}_{t})\cdot\hat{A}^{i}_{t}\right]" display="block"><semantics><mrow><mrow><msub><mo>∇</mo><msup><mi>θ</mi><mi>i</mi></msup></msub><msup><mi>J</mi><mi>i</mi></msup></mrow><mo>≈</mo><mrow><mi>𝔼</mi><mo>⁡</mo><mrow><mo>[</mo><mrow><mrow><msub><mo>∇</mo><msup><mi>θ</mi><mi>i</mi></msup></msub><mo lspace="0.167em" rspace="0em">​</mo><mi>log</mi><mo lspace="0.167em" rspace="0em">​</mo><msup><mi>π</mi><mi>i</mi></msup><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msubsup><mi>a</mi><mi>t</mi><mi>i</mi></msubsup><mo lspace="0em" rspace="0em">|</mo><msubsup><mi>o</mi><mi>t</mi><mi>i</mi></msubsup><mo rspace="0.055em" stretchy="false">)</mo></mrow></mrow><mo rspace="0.222em">⋅</mo><msubsup><mover accent="true"><mi>A</mi><mo>^</mo></mover><mi>t</mi><mi>i</mi></msubsup></mrow><mo>]</mo></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(25.6)</span></div>

### Centralized Training, Decentralized Execution (CTDE)

CTDE [240, 306] is the dominant paradigm for cooperative multi-agent RL. During training, a centralized critic has access to the global state <math alttext="s" display="inline"><semantics><mi>s</mi></semantics></math> and all agents’ actions <math alttext="\mathbf{a}=(a^{1},\ldots,a^{n})" display="inline"><semantics><mrow><mi>𝐚</mi><mo>=</mo><mrow><mo stretchy="false">(</mo><msup><mi>a</mi><mn>1</mn></msup><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msup><mi>a</mi><mi>n</mi></msup><mo stretchy="false">)</mo></mrow></mrow></semantics></math>. During execution, each agent acts using only its local observation <math alttext="o^{i}" display="inline"><semantics><msup><mi>o</mi><mi>i</mi></msup></semantics></math>.

The centralized critic for agent <math alttext="i" display="inline"><semantics><mi>i</mi></semantics></math>:

<div class="hh-equation" id="ch25.e7"><math alttext="Q^{i}_{\phi}(s,\mathbf{a})=Q^{i}_{\phi}(s,a^{1},\ldots,a^{n})" display="block"><semantics><mrow><mrow><msubsup><mi>Q</mi><mi>ϕ</mi><mi>i</mi></msubsup><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>s</mi><mo>,</mo><mi>𝐚</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><msubsup><mi>Q</mi><mi>ϕ</mi><mi>i</mi></msubsup><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>s</mi><mo>,</mo><msup><mi>a</mi><mn>1</mn></msup><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msup><mi>a</mi><mi>n</mi></msup><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(25.7)</span></div>

The decentralized actor for agent <math alttext="i" display="inline"><semantics><mi>i</mi></semantics></math>:

<div class="hh-equation" id="ch25.e8"><math alttext="\pi^{i}_{\theta^{i}}(a^{i}|o^{i})" display="block"><semantics><mrow><msubsup><mi>π</mi><msup><mi>θ</mi><mi>i</mi></msup><mi>i</mi></msubsup><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msup><mi>a</mi><mi>i</mi></msup><mo lspace="0em" rspace="0em">|</mo><msup><mi>o</mi><mi>i</mi></msup><mo stretchy="false">)</mo></mrow></mrow></semantics></math><span class="hh-equation-number">(25.8)</span></div>

The policy gradient with centralized critic:

<div class="hh-equation" id="ch25.e9"><math alttext="\nabla_{\theta^{i}}J^{i}=\mathbb{E}\left[\nabla_{\theta^{i}}\log\pi^{i}(a^{i}|o^{i})\cdot Q^{i}_{\phi}(s,\mathbf{a})\right]" display="block"><semantics><mrow><mrow><msub><mo>∇</mo><msup><mi>θ</mi><mi>i</mi></msup></msub><msup><mi>J</mi><mi>i</mi></msup></mrow><mo>=</mo><mrow><mi>𝔼</mi><mo>⁡</mo><mrow><mo>[</mo><mrow><mrow><mrow><msub><mo>∇</mo><msup><mi>θ</mi><mi>i</mi></msup></msub><mo lspace="0.167em" rspace="0em">​</mo><mi>log</mi><mo lspace="0.167em" rspace="0em">​</mo><msup><mi>π</mi><mi>i</mi></msup><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msup><mi>a</mi><mi>i</mi></msup><mo lspace="0em" rspace="0em">|</mo><msup><mi>o</mi><mi>i</mi></msup><mo rspace="0.055em" stretchy="false">)</mo></mrow></mrow><mo rspace="0.222em">⋅</mo><msubsup><mi>Q</mi><mi>ϕ</mi><mi>i</mi></msubsup></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>s</mi><mo>,</mo><mi>𝐚</mi><mo stretchy="false">)</mo></mrow></mrow><mo>]</mo></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(25.9)</span></div>

CTDE resolves non-stationarity during training (the centralized critic sees the full joint state) while preserving decentralized execution (no communication required at inference time).

### Communication Learning

Rather than using fixed communication protocols, agents can learn what to communicate. In differentiable communication frameworks [342, 66], agents produce continuous communication vectors <math alttext="m^{i}_{t}" display="inline"><semantics><msubsup><mi>m</mi><mi>t</mi><mi>i</mi></msubsup></semantics></math> that are passed to other agents:

<div class="hh-equation" id="ch25.e10"><math alttext="a^{i}_{t},m^{i}_{t}=\pi^{i}_{\theta^{i}}(o^{i}_{t},\{m^{j}_{t-1}\}_{j\neq i})" display="block"><semantics><mrow><msubsup><mi>a</mi><mi>t</mi><mi>i</mi></msubsup><mo>,</mo><mrow><msubsup><mi>m</mi><mi>t</mi><mi>i</mi></msubsup><mo>=</mo><mrow><msubsup><mi>π</mi><msup><mi>θ</mi><mi>i</mi></msup><mi>i</mi></msubsup><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msubsup><mi>o</mi><mi>t</mi><mi>i</mi></msubsup><mo>,</mo><msub><mrow><mo stretchy="false">{</mo><msubsup><mi>m</mi><mrow><mi>t</mi><mo>−</mo><mn>1</mn></mrow><mi>j</mi></msubsup><mo stretchy="false">}</mo></mrow><mrow><mi>j</mi><mo>≠</mo><mi>i</mi></mrow></msub><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(25.10)</span></div>

The communication vectors are optimized end-to-end via backpropagation through the joint reward signal. For LLM agents, this is approximated by training agents to produce structured natural language messages that maximize task performance.

### Emergent Communication

When agents are trained from scratch with only a reward signal (no predefined language), they can develop emergent communication protocols[200]: shared symbol systems that encode task-relevant information. While fascinating scientifically, emergent communication in LLM systems is typically undesirable—we want agents to communicate in human-interpretable language.

### Self-Play

In competitive or mixed-motive settings, self-play[332] trains agents by having them compete against copies of themselves. This generates an automatic curriculum: as the agent improves, its opponent (a previous version of itself) becomes harder to beat.

For LLM agents, self-play is used in:

<ul><li>Red team vs. blue team training</li><li>Debate training (agents argue against each other)</li><li>Negotiation training (agents negotiate with each other)</li></ul>

### Population-Based Training

Population-Based Training (PBT)[158] maintains a diverse population of agents with different policies, hyperparameters, and specializations. Agents are periodically evaluated; underperforming agents are replaced by mutated copies of high-performing agents.

For multi-agent LLM systems, PBT enables:

<ul><li>Automatic discovery of effective role specializations</li><li>Robustness to individual agent failures (diverse population)</li><li>Avoidance of local optima through population diversity</li></ul>

### Social Welfare and Nash Equilibrium

In multi-agent settings, the notion of optimality is more complex than in single-agent settings. Two key solution concepts:

Nash Equilibrium: a joint policy <math alttext="(\pi^{1*},\ldots,\pi^{n*})" display="inline"><semantics><mrow><mo stretchy="false">(</mo><msup><mi>π</mi><mrow><mn>1</mn><mo lspace="0.222em">∗</mo></mrow></msup><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msup><mi>π</mi><mrow><mi>n</mi><mo lspace="0.222em">∗</mo></mrow></msup><mo stretchy="false">)</mo></mrow></semantics></math> such that no agent can improve its expected return by unilaterally deviating:

<div class="hh-equation" id="ch25.e11"><math alttext="J^{i}(\pi^{i*},\pi^{-i*})\geq J^{i}(\pi^{i},\pi^{-i*})\quad\forall i,\forall\pi^{i}" display="block"><semantics><mrow><msup><mi>J</mi><mi>i</mi></msup><mrow><mo stretchy="false">(</mo><msup><mi>π</mi><mrow><mi>i</mi><mo lspace="0.222em">∗</mo></mrow></msup><mo>,</mo><msup><mi>π</mi><mrow><mo>−</mo><mi>i</mi><mo lspace="0.222em">∗</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mo>≥</mo><msup><mi>J</mi><mi>i</mi></msup><mrow><mo stretchy="false">(</mo><msup><mi>π</mi><mi>i</mi></msup><mo>,</mo><msup><mi>π</mi><mrow><mo>−</mo><mi>i</mi><mo lspace="0.222em">∗</mo></mrow></msup><mo rspace="0.167em" stretchy="false">)</mo></mrow><mo rspace="0.167em">∀</mo><mi>i</mi><mo>,</mo><mo rspace="0.167em">∀</mo><msup><mi>π</mi><mi>i</mi></msup></mrow></semantics></math><span class="hh-equation-number">(25.11)</span></div>

where <math alttext="\pi^{-i}" display="inline"><semantics><msup><mi>π</mi><mrow><mo>−</mo><mi>i</mi></mrow></msup></semantics></math> denotes the joint policy of all agents except <math alttext="i" display="inline"><semantics><mi>i</mi></semantics></math>.

Social Welfare Maximization: optimize the sum of all agents’ returns:

<div class="hh-equation" id="ch25.e12"><math alttext="\max_{\pi^{1},\ldots,\pi^{n}}\sum_{i=1}^{n}J^{i}(\pi^{1},\ldots,\pi^{n})" display="block"><semantics><mrow><mi>max</mi><mo>⁡</mo><mrow><mrow><mmultiscripts><munderover><mo movablelimits="false">∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><mprescripts></mprescripts><mrow><msup><mi>π</mi><mn>1</mn></msup><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msup><mi>π</mi><mi>n</mi></msup></mrow><mrow></mrow></mmultiscripts><mo>⁡</mo><msup><mi>J</mi><mi>i</mi></msup></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msup><mi>π</mi><mn>1</mn></msup><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msup><mi>π</mi><mi>n</mi></msup><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(25.12)</span></div>

In fully cooperative settings (all agents share the same reward), social welfare maximization is the appropriate objective. In competitive settings, Nash equilibrium is the relevant solution concept. Most real-world multi-agent LLM systems are mixed-motive: agents have partially aligned, partially conflicting objectives.

## Challenges and Solutions

### Coordination Overhead

Every inter-agent message consumes tokens—and therefore time and money. In a naive implementation, agents communicate constantly, even when unnecessary.

Quantifying communication cost: if a message costs <math alttext="c" display="inline"><semantics><mi>c</mi></semantics></math> tokens and the receiving agent’s task has value <math alttext="v" display="inline"><semantics><mi>v</mi></semantics></math>, communicate only if the expected improvement in task value <math alttext="\Delta v&gt;c\cdot\text{cost\_per\_token}" display="inline"><semantics><mrow><mrow><mi mathvariant="normal">Δ</mi><mo lspace="0em" rspace="0em">​</mo><mi>v</mi></mrow><mo>&gt;</mo><mrow><mi>c</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><mtext>cost_per_token</mtext></mrow></mrow></semantics></math>.

### Redundancy vs. Efficiency

Multiple agents may independently solve the same sub-problem, wasting compute. Solutions:

<ul><li>Duplicate detection: before starting a task, check the blackboard for existing results</li><li>Result caching: store completed sub-task results with semantic keys for retrieval</li><li>Task locking: mark tasks as “in progress” to prevent duplicate execution</li></ul>

### Attribution

When a multi-agent system succeeds or fails, which agent is responsible? Attribution is critical for:

<ul><li>RL reward assignment (credit assignment problem)</li><li>Debugging and improvement</li><li>Trust calibration (which agents to rely on)</li></ul>

The counterfactual credit assignment approach estimates each agent’s contribution by asking: “How much would the outcome have changed if this agent had acted differently?”

<div class="hh-equation" id="ch25.e13"><math alttext="\text{credit}^{i}=J(\pi^{1},\ldots,\pi^{n})-J(\pi^{1},\ldots,\pi^{i}_{\text{default}},\ldots,\pi^{n})" display="block"><semantics><mrow><msup><mtext>credit</mtext><mi>i</mi></msup><mo>=</mo><mrow><mrow><mi>J</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msup><mi>π</mi><mn>1</mn></msup><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msup><mi>π</mi><mi>n</mi></msup><mo stretchy="false">)</mo></mrow></mrow><mo>−</mo><mrow><mi>J</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msup><mi>π</mi><mn>1</mn></msup><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msubsup><mi>π</mi><mtext>default</mtext><mi>i</mi></msubsup><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msup><mi>π</mi><mi>n</mi></msup><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(25.13)</span></div>

### Scalability

Naive message passing scales as <math alttext="O(n^{2})" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msup><mi>n</mi><mn>2</mn></msup><mo stretchy="false">)</mo></mrow></mrow></semantics></math> with the number of agents. Solutions:

<ul><li>Hierarchical communication: agents communicate only within their subtree</li><li>Topic-based pub/sub: agents subscribe only to relevant message topics</li><li>Sparse communication graphs: only connect agents that need to interact</li><li>Asynchronous communication: agents don’t block waiting for responses</li></ul>

### Emergent Behavior and Safety

Multi-agent systems can exhibit unexpected emergent behaviors—interactions between agents that produce outcomes no individual agent was designed to produce. This is both a feature (emergent capabilities) and a risk (emergent failures).

### Evaluation

Evaluating multi-agent systems requires metrics at multiple levels:

<div class="hh-table" id="ch25.t3"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Level</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Metric</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Example</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">System</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Task completion rate</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>% of tasks completed correctly</span></span></td></tr><tr><th class="ltx_align_left">System</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>End-to-end latency</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Time from task to final output</span></span></td></tr><tr><th class="ltx_align_left">System</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Total token cost</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Tokens consumed across all agents</span></span></td></tr><tr><th class="ltx_align_left">Agent</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Individual accuracy</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Per-agent task success rate</span></span></td></tr><tr><th class="ltx_align_left">Agent</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Communication efficiency</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Useful messages / total messages</span></span></td></tr><tr><th class="ltx_align_left">Agent</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Contribution score</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Counterfactual credit (Eq. <span></span>)</span></span></td></tr><tr><th class="ltx_align_left">Emergent</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Coordination quality</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Degree of task overlap / gaps</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 25.3:</span>Multi-level evaluation metrics for multi-agent systems.</p></div>

## Real-World Multi-Agent Applications

### Software Development Team

A multi-agent software development team mirrors a real engineering organization:

```python
from dataclasses import dataclass
from typing import Optional
import asyncio

@dataclass
class SoftwareTeamState:
    requirements: str
    architecture: Optional[str] = None
    code: Optional[str] = None
    tests: Optional[str] = None
    review_feedback: Optional[str] = None
    final_code: Optional[str] = None
    approved: bool = False

class SoftwareDevelopmentTeam:
    """
    Multi-agent software team:
      Architect -> Coder -> Tester -> Reviewer -> (iterate or ship)
    """

    def __init__(self, llm_factory):
        self.architect = llm_factory(
            system_prompt="""You are a software architect. Given requirements,
            produce a clear technical design: components, interfaces, data
            structures, and implementation plan."""
        )
        self.coder = llm_factory(
            system_prompt="""You are an expert software engineer. Given a
            technical design, write clean, well-documented, production-ready
            code. Follow best practices for the language."""
        )
        self.tester = llm_factory(
            system_prompt="""You are a QA engineer. Given code, write
            comprehensive tests: unit tests, edge cases, integration tests.
            Identify potential bugs and failure modes."""
        )
        self.reviewer = llm_factory(
            system_prompt="""You are a senior code reviewer. Evaluate code
            for correctness, security, performance, and maintainability.
            Provide specific, actionable feedback. Approve only if excellent."""
        )

    async def build(self, requirements: str,
                    max_iterations: int = 3) -> SoftwareTeamState:
        state = SoftwareTeamState(requirements=requirements)

        # Phase 1: Architecture
        state.architecture = await self.architect.invoke(
            f"Requirements:\n{requirements}\n\nProduce technical design."
        )

        for iteration in range(max_iterations):
            # Phase 2: Implementation
            prompt = (f"Design:\n{state.architecture}\n\n"
                      + (f"Previous feedback:\n{state.review_feedback}\n\n"
                         if state.review_feedback else "")
                      + "Write the implementation.")
            state.code = await self.coder.invoke(prompt)

            # Phase 3: Testing
            state.tests = await self.tester.invoke(
                f"Code:\n{state.code}\n\nWrite comprehensive tests."
            )

            # Phase 4: Review
            review = await self.reviewer.invoke(
                f"Code:\n{state.code}\n\nTests:\n{state.tests}\n\n"
                "Review this code. End with APPROVED or NEEDS_REVISION."
            )

            if "APPROVED" in review:
                state.final_code = state.code
                state.approved = True
                break
            else:
                state.review_feedback = review

        return state

    async def run(self, requirements: str) -> str:
        state = await self.build(requirements)
        if state.approved:
            return f"# Final Implementation\n\n{state.final_code}"
        else:
            return f"# Best Attempt (not approved)\n\n{state.code}"
```

### Research Team

A research team agent society mirrors academic collaboration:

<ul><li>Literature Reviewer: searches and synthesizes existing work</li><li>Hypothesis Generator: proposes novel research directions</li><li>Experimentalist: designs and runs experiments (via code execution)</li><li>Statistician: analyzes results and assesses significance</li><li>Writer: synthesizes findings into a coherent report</li></ul>

### Customer Service System

A tiered customer service system:

<ul><li>Router: classifies incoming requests and routes to specialists</li><li>Billing Specialist: handles payment and account issues</li><li>Technical Specialist: resolves product/service issues</li><li>Escalation Agent: handles complex cases requiring human judgment</li></ul>

### Creative Team

A creative production pipeline:

<ul><li>Brainstormer: generates diverse ideas without self-censorship</li><li>Drafter: develops the most promising ideas into full drafts</li><li>Editor: refines drafts for clarity, style, and coherence</li><li>Critic: provides adversarial feedback to strengthen the work</li></ul>

## Architecture Comparison

<div class="hh-table" id="ch25.t4"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Architecture</span></th><th class="ltx_align_left"><span>Scalability</span></th><th class="ltx_align_left"><span>Debug</span></th><th class="ltx_align_left"><span>Coord. Cost</span></th><th class="ltx_align_left"><span>Fault Tol.</span></th><th class="ltx_align_left"><span>Best For</span></th></tr></thead><tbody><tr><td class="ltx_align_left">Centralized (Supervisor)</td><td class="ltx_align_left">M</td><td class="ltx_align_left">H</td><td class="ltx_align_left">L</td><td class="ltx_align_left">L</td><td class="ltx_align_left">Simple pipelines; clear task decomposition; small teams</td></tr><tr><td class="ltx_align_left">Decentralized (P2P)</td><td class="ltx_align_left">H</td><td class="ltx_align_left">L</td><td class="ltx_align_left">H</td><td class="ltx_align_left">H</td><td class="ltx_align_left">Dynamic environments; resilience-critical; large-scale</td></tr><tr><td class="ltx_align_left">Hierarchical</td><td class="ltx_align_left">H</td><td class="ltx_align_left">M</td><td class="ltx_align_left">M</td><td class="ltx_align_left">M</td><td class="ltx_align_left">Enterprise workflows; complex multi-domain tasks</td></tr><tr><td class="ltx_align_left">Swarm</td><td class="ltx_align_left">H</td><td class="ltx_align_left">L</td><td class="ltx_align_left">L</td><td class="ltx_align_left">H</td><td class="ltx_align_left">Customer service routing; simple handoff chains</td></tr><tr><td class="ltx_align_left">Pipeline</td><td class="ltx_align_left">M</td><td class="ltx_align_left">H</td><td class="ltx_align_left">L</td><td class="ltx_align_left">L</td><td class="ltx_align_left">Sequential processing; clear stage dependencies</td></tr><tr><td class="ltx_align_left">Ensemble</td><td class="ltx_align_left">L</td><td class="ltx_align_left">H</td><td class="ltx_align_left">H</td><td class="ltx_align_left">H</td><td class="ltx_align_left">High-stakes decisions; reliability over efficiency</td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 25.4:</span>Multi-agent architecture patterns compared across key dimensions. Ratings: High / Medium / Low.</p></div>

## Self-Evolving Agents: BDI Meets LLMs

A May 2026 line of research [427] represents the first serious attempt to combine Belief-Desire-Intention (BDI) architectures—the classical framework for rational agents in multi-agent systems—with LLMs. Unlike static agents that execute fixed tool-call sequences, BDI-LLM agents autonomously modify their own goals and rewrite their underlying code based on environmental feedback:

<ul><li>Beliefs: The LLM maintains and updates a structured world model (beliefs about file states, API responses, user intent)</li><li>Desires: High-level goals that can be dynamically reprioritized based on observed outcomes</li><li>Intentions: Concrete plans (code + tool sequences) that the agent can rewrite when a plan fails</li></ul>

## Summary

Multi-agent systems represent a fundamental shift in how we deploy LLMs: from isolated assistants to collaborative societies of specialized agents. The key insights from this section:

The field of multi-agent LLM systems is evolving rapidly. The patterns and techniques described here represent the current state of the art, but new architectures, coordination mechanisms, and training algorithms are emerging continuously. The foundational principles—specialization, coordination, emergent behavior, and the tension between efficiency and robustness—will remain relevant regardless of how the specific implementations evolve.

<aside class="hh-next">
<p class="hh-next-label">下一篇</p>
<p class="hh-next-title">The Hitchhiker's Guide to Agentic AI · Chapter 26 Agent Development Frameworks</p>
<p class="hh-next-hint">在「智能体 AI 漫游指南」分类下继续阅读。</p>
</aside>

</div>
