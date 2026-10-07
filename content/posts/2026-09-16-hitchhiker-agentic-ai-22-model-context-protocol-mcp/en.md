---
title: "Chapter 22 Model Context Protocol (MCP)"
slug: "hitchhiker-agentic-ai-22-model-context-protocol-mcp"
lang: "en"
date: "2026-09-16T00:23:00.000Z"
draft: false
featured: false
categories:
  - "hitchhiker-agentic-ai"
cover: "./cover.png"
description: "The rise of tool-augmented language models has created a fragmentation problem: every agent framework, every LLM provider, and every enterpr…"
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

The rise of tool-augmented language models has created a fragmentation problem: every agent framework, every LLM provider, and every enterprise deployment invents its own mechanism for connecting models to external tools and data sources. The Model Context Protocol (MCP)[10], introduced by Anthropic in late 2024, is an open standard designed to solve this problem once and for all—providing a universal, vendor-neutral interface between AI applications and the tools they need.

## Motivation: The Tool Integration Problem

Consider the combinatorial explosion facing any organization that wants to connect AI agents to its infrastructure. Suppose there are <math alttext="N" display="inline"><semantics><mi>N</mi></semantics></math> distinct agent frameworks (LangChain, AutoGen, CrewAI, custom agents, …) and <math alttext="M" display="inline"><semantics><mi>M</mi></semantics></math> distinct tool providers (GitHub, Slack, PostgreSQL, Jira, …). Without a standard protocol, each combination requires a bespoke integration:

<div class="hh-equation" id="ch22.e1"><math alttext="\text{Integrations without standard}=N\times M" display="block"><semantics><mrow><mtext>Integrations without standard</mtext><mo>=</mo><mrow><mi>N</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mi>M</mi></mrow></mrow></semantics></math><span class="hh-equation-number">(22.1)</span></div>

With a universal protocol, each side only needs to implement the protocol once:

<div class="hh-equation" id="ch22.e2"><math alttext="\text{Integrations with standard}=N+M" display="block"><semantics><mrow><mtext>Integrations with standard</mtext><mo>=</mo><mrow><mi>N</mi><mo>+</mo><mi>M</mi></mrow></mrow></semantics></math><span class="hh-equation-number">(22.2)</span></div>

For <math alttext="N=20" display="inline"><semantics><mrow><mi>N</mi><mo>=</mo><mn>20</mn></mrow></semantics></math> agent frameworks and <math alttext="M=50" display="inline"><semantics><mrow><mi>M</mi><mo>=</mo><mn>50</mn></mrow></semantics></math> tool providers, this reduces the integration burden from 1,000 custom connectors to just 70 protocol implementations—a 14<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> reduction. This is precisely the insight behind protocols like USB (universal device connectivity), HTTP (universal web communication), and LSP (Language Server Protocol for IDE tooling). MCP applies the same philosophy to AI tool use.

The analogy to the Language Server Protocol (LSP)1 is particularly apt. Before LSP, every IDE had to implement language support (autocomplete, go-to-definition, error highlighting) for every programming language separately. After LSP, language servers and editors only need to speak a common protocol. MCP does for AI tool use what LSP did for developer tooling.

<figure id="ch22.f1"><img src="./fig_066_mcp-flow.png" alt="Figure 22.1: How MCP works: a single user request flows through the Host, LLM, and MCP Server. The LLM decides which too" loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 22.1:</span>How MCP works: a single user request flows through the Host, LLM, and MCP Server. The LLM decides which tool to call (step 3); the Host routes the call to the appropriate server via JSON-RPC (step 4); the result flows back through the LLM for natural-language formatting (steps 5–7). The user never sees the protocol machinery.</figcaption></figure>

## Architecture Overview

MCP follows a client-server architecture with three distinct roles, connected by a well-defined protocol layer.

### The Three-Role Model

MCP Host

The LLM application that the end user interacts with directly. Examples include Claude Desktop, a VS Code extension, a custom chatbot, or an autonomous agent. The host is responsible for managing the overall user experience, deciding which MCP servers to connect to, and enforcing security policies. The host contains one or more MCP clients.

MCP Client

A protocol-level component embedded within the host application. Each client maintains a <em>stateful, one-to-one connection</em> with a single MCP server. The client handles protocol negotiation, message serialization, and the lifecycle of the connection. A single host may run multiple clients simultaneously, each connected to a different server.

MCP Server

A lightweight process or service that exposes capabilities (tools, resources, prompts) to clients. Servers are typically thin wrappers around existing APIs, databases, or system interfaces. They are designed to be simple to implement—the complexity of the protocol is handled by the client/host layer.

### Transport Layers

MCP is transport-agnostic at the protocol level, but defines two standard transport mechanisms:

stdio (Standard I/O)

The client spawns the server as a child process and communicates via standard input/output streams. This is the simplest and most common transport for local tools. It provides strong isolation (the server runs in a separate process) and requires no network configuration. Ideal for filesystem access, local code execution, and developer tools.

Streamable HTTP

The server runs as an HTTP service. The client sends JSON-RPC requests via HTTP POST; the server may respond with a single JSON response or upgrade to a Server-Sent Events (SSE) stream for incremental results. This transport supports remote servers, enables server-side push notifications, and works through standard web infrastructure (proxies, load balancers, firewalls). Suitable for cloud-hosted tools and enterprise deployments. (This replaced the earlier HTTP+SSE-only transport in the 2025-03-26 protocol revision.)

### Protocol Lifecycle

Every MCP connection follows a four-phase lifecycle:

<ol><li>Initialization: The client sends an <code>initialize</code> request containing its protocol version and supported capabilities. The server responds with its own version and capabilities. This establishes the feature set available for the session.</li><li>Capability Negotiation: Both sides declare what they support (e.g., whether the server offers tools, resources, or prompts; whether the client supports sampling). Capabilities not declared by both sides are not used.</li><li>Operation: The main phase. The client sends requests (tool calls, resource reads, prompt fetches) and the server responds. The server may also send notifications (e.g., resource change events) without being asked.</li><li>Shutdown: Either side can initiate a graceful shutdown. The client sends a <code>shutdown</code> notification; the server cleans up resources and terminates.</li></ol>

### Stateful Sessions vs. Stateless Requests

A key design decision in MCP is that connections are stateful sessions, not stateless HTTP requests. This matters for several reasons:

<ul><li>Efficiency: Capability negotiation happens once at connection time, not on every request.</li><li>Context: Servers can maintain session state (e.g., an open database transaction, a checked-out file lock).</li><li>Subscriptions: Servers can push notifications to clients when resources change.</li><li>Long-running operations: Progress reporting is natural in a stateful session.</li></ul>

The tradeoff is that stateful sessions require connection management (reconnection logic, session recovery) that stateless APIs avoid.

### Full Architecture Diagram

Figure 图 22.2 illustrates the full MCP stack, from the user interface down to external services.

<figure id="ch22.f2"><img src="./fig_067_mcp-architecture.png" alt="Figure 22.2: Full MCP architecture stack. The Host manages one or more Clients, each maintaining a stateful session with" loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 22.2:</span>Full MCP architecture stack. The Host manages one or more Clients, each maintaining a stateful session with an MCP Server over a transport layer (stdio or Streamable HTTP). All client–server communication uses JSON-RPC 2.0. Servers wrap external services and expose them as standardized Tools, Resources, and Prompts.</figcaption></figure>

## Core Primitives

MCP defines four core primitives that servers can expose to clients. Each primitive has a distinct purpose, direction of control, and use case.

### Tools

Tools are the most important primitive—they are function-like operations that the server exposes for the LLM to invoke. A tool has:

<ul><li>A name (unique identifier within the server)</li><li>A description (natural language explanation for the LLM)</li><li>An inputSchema (JSON Schema defining the parameters)</li><li>An optional outputSchema (JSON Schema for the return value)</li></ul>

Tools represent <em>actions with side effects</em>: creating files, sending messages, executing code, querying databases. The LLM decides when and how to call tools; the server executes them.

### Resources

Resources are data that the server can provide to the client. Unlike tools (which are invoked by the LLM), resources are typically <em>read by the host application</em> to populate the LLM’s context window. Resources have URIs (e.g., <code>file:///home/user/notes.txt</code>, <code>db://customers/42</code>) and can be static or dynamic.

Resources support subscriptions: the client can subscribe to a resource URI and receive notifications when the underlying data changes. This enables reactive agents that respond to real-world events.

### Prompts

Prompts are reusable prompt templates that the server offers. They allow server authors to encode domain expertise into structured prompts that the host can present to users or inject into conversations. For example, a GitHub MCP server might offer a “code review” prompt template that takes a PR number as input and generates a structured review request.

### Sampling

Sampling is the most unusual primitive—it runs in the <em>reverse direction</em>. Instead of the client asking the server to do something, the <em>server asks the client to perform LLM inference</em>. This reverse flow allows tool servers to incorporate model-driven reasoning steps (e.g., summarizing retrieved data before returning it) without needing their own LLM deployment. The host retains full control over whether to honor sampling requests, maintaining the security boundary.

## Protocol Specification

MCP is built on JSON-RPC 2.0[121], a lightweight remote procedure call protocol that uses JSON for message encoding. This choice provides a well-understood, language-agnostic foundation with broad library support.

### JSON-RPC 2.0 Message Format

There are three message types in JSON-RPC 2.0:

Request (client <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> server, expects a response):

```python
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "tools/call",
  "params": {
    "name": "read_file",
    "arguments": { "path": "/home/user/notes.txt" }
  }
}
```

Response (server <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> client, in reply to a request):

```python
{
  "jsonrpc": "2.0",
  "id": 42,
  "result": {
    "content": [
      { "type": "text", "text": "Meeting notes: ..." }
    ],
    "isError": false
  }
}
```

Notification (either direction, no response expected):

```python
{
  "jsonrpc": "2.0",
  "method": "notifications/resources/updated",
  "params": { "uri": "file:///home/user/notes.txt" }
}
```

### Capability Negotiation Handshake

The initialization handshake establishes what both sides can do:

```python
// Client sends:
{
  "jsonrpc": "2.0", "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "sampling": {},          // client supports sampling requests
      "roots": { "listChanged": true }
    },
    "clientInfo": { "name": "MyAgent", "version": "1.0.0" }
  }
}

// Server responds:
{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "tools": { "listChanged": true },   // server has tools
      "resources": { "subscribe": true }, // server supports subscriptions
      "prompts": {}
    },
    "serverInfo": { "name": "filesystem", "version": "0.6.2" }
  }
}
```

### Error Handling

JSON-RPC errors follow a standard format with numeric error codes. MCP defines additional codes beyond the JSON-RPC standard:

```python
{
  "jsonrpc": "2.0", "id": 42,
  "error": {
    "code": -32602,          // Invalid params (JSON-RPC standard)
    "message": "Invalid file path: path must be absolute",
    "data": { "path": "relative/path.txt" }
  }
}
```

### Progress Reporting

For long-running operations, MCP supports progress notifications. The client includes a <code>progressToken</code> in the request; the server sends periodic <code>notifications/progress</code> messages:

```python
// Request with progress token
{
  "jsonrpc": "2.0", "id": 10,
  "method": "tools/call",
  "params": {
    "name": "index_codebase",
    "arguments": { "path": "/repo" },
    "_meta": { "progressToken": "index-op-1" }
  }
}

// Server sends progress notifications (no id = notification)
{
  "jsonrpc": "2.0",
  "method": "notifications/progress",
  "params": {
    "progressToken": "index-op-1",
    "progress": 45,
    "total": 100,
    "message": "Indexed 450/1000 files..."
  }
}
```

## Tool Definition and Discovery

Tools are the heart of MCP. Getting tool definitions right is critical because the LLM uses the name and description to decide <em>which tool to call and when</em>.

### Tool Schema Format

A complete tool definition:

```python
{
  "name": "search_codebase",
  "description": "Search for a pattern across all files in the repository.
    Returns matching file paths and line numbers. Use this when you need
    to find where a function is defined, where a variable is used, or
    where a specific string appears. Supports regex patterns.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "pattern": {
        "type": "string",
        "description": "Regex pattern to search for"
      },
      "path": {
        "type": "string",
        "description": "Directory to search in (default: repo root)",
        "default": "."
      },
      "case_sensitive": {
        "type": "boolean",
        "description": "Whether the search is case-sensitive",
        "default": false
      }
    },
    "required": ["pattern"]
  }
}
```

### Dynamic Tool Registration

Servers can add, remove, or modify tools during a session by sending a <code>notifications/tools/list_changed</code> notification. The client then re-fetches the tool list with a <code>tools/list</code> request. This enables:

<ul><li>Context-sensitive tools: A code editor server might expose different tools depending on the currently open file type.</li><li>Permission-gated tools: Tools that become available only after the user grants specific permissions.</li><li>Dynamic plugin systems: Tools loaded from external registries at runtime.</li></ul>

### Tool Annotations

MCP introduced tool annotations—metadata hints that help hosts make better decisions about tool execution (added in the 2025-03-26 protocol revision):

```python
{
  "name": "delete_file",
  "description": "Permanently delete a file from the filesystem.",
  "inputSchema": { ... },
  "annotations": {
    "readOnlyHint": false,      // This tool modifies state
    "destructiveHint": true,    // Changes are irreversible
    "idempotentHint": false,    // Calling twice has different effects
    "openWorldHint": false      // Does not interact with external services
  }
}
```

<code>readOnlyHint</code>

If <code>true</code>, the tool only reads data and has no side effects. Hosts may auto-approve read-only tools without user confirmation.

<code>destructiveHint</code>

If <code>true</code>, the tool performs irreversible actions. Hosts should require explicit user confirmation.

<code>idempotentHint</code>

If <code>true</code>, calling the tool multiple times with the same arguments has the same effect as calling it once. Safe to retry on failure.

<code>openWorldHint</code>

If <code>true</code>, the tool interacts with external services beyond the server’s direct control (e.g., sending an email, posting to social media).

## Security Model

MCP operates across multiple trust boundaries. Understanding these boundaries is essential for safe deployment.

### Trust Hierarchy

Host (highest trust)

The host application is trusted by the user. It enforces security policies, manages user consent, and controls which servers the client connects to. The host is the ultimate arbiter of what actions are permitted.

Client (trusted by host)

The client implements the protocol faithfully and enforces the host’s policies. It validates server responses and sanitizes data before passing it to the LLM.

Server (conditionally trusted)

Servers are trusted to implement their declared capabilities honestly, but the host should not blindly trust server-provided data. A compromised or malicious server could attempt prompt injection attacks by embedding instructions in resource content.

External Services (untrusted)

Services that MCP servers interact with (web APIs, databases, file systems) are untrusted from the protocol’s perspective. Servers must validate and sanitize all external data.

### User Consent

MCP mandates that users must explicitly consent to tool execution, especially for tools with side effects. The host is responsible for:

<ul><li>Presenting clear descriptions of what a tool will do before execution</li><li>Distinguishing between read-only and destructive operations (using annotations)</li><li>Providing audit logs of all tool calls made on the user’s behalf</li><li>Allowing users to revoke permissions at any time</li></ul>

### Input Validation and Sanitization

Servers must validate all inputs against their declared JSON Schema before execution. Common vulnerabilities to guard against:

<ul><li>Path traversal: <code>../../etc/passwd</code> in file path arguments</li><li>SQL injection: Unsanitized strings in database query tools</li><li>Command injection: Shell metacharacters in code execution tools</li><li>SSRF: URLs pointing to internal network resources in HTTP tools</li></ul>

### Credential Management

MCP servers frequently need credentials to access external services. Best practices:

<ul><li>OAuth 2.0: For user-delegated access to third-party services (GitHub, Google, Slack). The server handles the OAuth flow; the host stores tokens securely.</li><li>Environment variables: API keys should be injected via environment variables, not hardcoded or passed through the protocol.</li><li>Secrets managers: Production deployments should use dedicated secrets management (AWS Secrets Manager, HashiCorp Vault) rather than environment variables.</li><li>Minimal permissions: Servers should request only the permissions they need (read-only database access, not admin credentials).</li></ul>

### Sandboxing Strategies

For servers that execute arbitrary code or access sensitive resources:

<ul><li>Process isolation: Run each server in a separate process with restricted OS permissions (seccomp, AppArmor, SELinux).</li><li>Container isolation: Deploy servers in Docker containers with minimal capabilities and no network access to internal services.</li><li>Read-only filesystems: Mount filesystems read-only unless write access is explicitly required.</li><li>Network policies: Use firewall rules to restrict which external services a server can reach.</li></ul>

## Implementation Patterns

### Building an MCP Server in Python

The official Python SDK provides <code>FastMCP</code>, a high-level framework that handles protocol negotiation, serialization, and transport automatically. Below is a complete note-taking MCP server:

```
#!/usr/bin/env python3
"""
A simple MCP server exposing note-taking tools and resources.
Install: pip install "mcp[cli]"
Run:     mcp run notes_server.py        (stdio)
         mcp run notes_server.py --transport streamable-http  (HTTP)
"""
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# -- Server setup --------------------------------------------------------------
mcp = FastMCP("notes-server")
NOTES_DIR = Path.home() / ".notes"
NOTES_DIR.mkdir(exist_ok=True)

# -- Tools (LLM-invoked actions) -----------------------------------------------
@mcp.tool()
def create_note(title: str, content: str, tags: list[str] | None = None) -> str:
    """Create a new text note with a given title and content.

    Use this when the user wants to save information for later.
    Returns the path where the note was saved.
    """
    tags = tags or []
    safe_title = "".join(
        c if c.isalnum() or c in " -_" else "_" for c in title
    ).strip()
    note_path = NOTES_DIR / f"{safe_title}.md"

    frontmatter = f"---\ntitle: {title}\ntags: {tags}\n---\n\n"
    note_path.write_text(frontmatter + content, encoding="utf-8")
    return f"Note saved to {note_path}"

@mcp.tool()
def search_notes(query: str) -> str:
    """Search notes by keyword. Searches both titles and content.

    Returns a list of matching note titles and snippets.
    Use this before creating a note to check if one already exists.
    """
    query_lower = query.lower()
    results = []

    for note_file in NOTES_DIR.glob("*.md"):
        text = note_file.read_text(encoding="utf-8")
        if query_lower in text.lower():
            idx = text.lower().find(query_lower)
            snippet = text[max(0, idx - 50):idx + 100].replace("\n", " ")
            results.append(f"- **{note_file.stem}**: ...{snippet}...")

    return "\n".join(results) if results else f"No notes found matching '{query}'"

# -- Resources (context data for the LLM) -------------------------------------
@mcp.resource("notes://{title}")
def get_note(title: str) -> str:
    """Read a note by title."""
    note_path = NOTES_DIR / f"{title}.md"
    if not note_path.exists():
        raise ValueError(f"Note not found: {title}")
    return note_path.read_text(encoding="utf-8")

# -- Entry point ----------------------------------------------------------------
if __name__ == "__main__":
    mcp.run()  # defaults to stdio transport
```

<span class="hh-tag">Listing 27:</span>Complete MCP Server: Note-Taking Tool (FastMCP)

Key differences from older low-level APIs:

<ul><li>Declarative tools: The <code>@mcp.tool()</code> decorator infers the JSON Schema from Python type hints and the docstring—no manual <code>inputSchema</code> needed.</li><li>Automatic transport: <code>mcp.run()</code> handles stdio or Streamable HTTP based on how the server is launched.</li><li>Resources as functions: <code>@mcp.resource("uri-template")</code> exposes data with URI-based routing.</li></ul>

### Building an MCP Client

A minimal client that connects to the notes server and calls a tool:

```
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # Connect to the notes server via stdio
    server_params = StdioServerParameters(
        command="python",
        args=["notes_server.py"],
        env=None  # inherit environment
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Phase 1: Initialize
            await session.initialize()

            # Phase 2: Discover available tools
            tools_result = await session.list_tools()
            print("Available tools:")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description[:60]}...")

            # Phase 3: Call a tool
            result = await session.call_tool(
                "create_note",
                arguments={
                    "title": "MCP Architecture Notes",
                    "content": "MCP uses JSON-RPC 2.0 over stdio or HTTP+SSE.",
                    "tags": ["mcp", "architecture"]
                }
            )
            print(f"\nTool result: {result.content[0].text}")

            # Phase 4: List resources
            resources = await session.list_resources()
            print(f"\nAvailable resources: {len(resources.resources)}")

asyncio.run(main())
```

<span class="hh-tag">Listing 28:</span>MCP Client: Connecting and Calling Tools

### Connecting to Multiple Servers Simultaneously

A host application typically manages multiple server connections. The pattern uses a connection pool:

```
import asyncio
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPHost:
    """Manages connections to multiple MCP servers."""

    def __init__(self):
        self.sessions: dict[str, ClientSession] = {}
        self.tool_registry: dict[str, tuple[str, object]] = {}
        self._exit_stack = AsyncExitStack()

    async def connect(self, name: str, params: StdioServerParameters):
        """Connect to a named MCP server and register its tools."""
        read, write = await self._exit_stack.enter_async_context(
            stdio_client(params)
        )
        session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await session.initialize()
        self.sessions[name] = session

        # Register all tools from this server
        tools = await session.list_tools()
        for tool in tools.tools:
            self.tool_registry[tool.name] = (name, tool)
            print(f"Registered tool '{tool.name}' from server '{name}'")

    async def call_tool(self, tool_name: str, arguments: dict):
        """Route a tool call to the appropriate server."""
        if tool_name not in self.tool_registry:
            raise ValueError(f"Unknown tool: {tool_name}")

        server_name, _ = self.tool_registry[tool_name]
        session = self.sessions[server_name]
        return await session.call_tool(tool_name, arguments)

    async def get_all_tools(self) -> list:
        """Return all tools across all connected servers."""
        return [tool for _, tool in self.tool_registry.values()]

    async def close(self):
        await self._exit_stack.aclose()

async def main():
    host = MCPHost()

    # Connect to multiple servers concurrently
    await asyncio.gather(
        host.connect("filesystem", StdioServerParameters(
            command="npx", args=["-y", "@modelcontextprotocol/server-filesystem",
                                  "/home/user"]
        )),
        host.connect("github", StdioServerParameters(
            command="npx", args=["-y", "@modelcontextprotocol/server-github"]
        )),
        host.connect("notes", StdioServerParameters(
            command="python", args=["notes_server.py"]
        )),
    )

    # All tools available through a single interface
    all_tools = await host.get_all_tools()
    print(f"Total tools available: {len(all_tools)}")

    await host.close()

asyncio.run(main())
```

<span class="hh-tag">Listing 29:</span>Multi-Server MCP Host Pattern

### Error Recovery and Reconnection

Production MCP clients must handle server crashes and network interruptions:

```
import asyncio
import logging
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

async def resilient_tool_call(
    params: StdioServerParameters,
    tool_name: str,
    arguments: dict,
    max_retries: int = 3,
    backoff_base: float = 1.0
):
    """Call a tool with automatic reconnection on failure."""
    for attempt in range(max_retries):
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await session.call_tool(tool_name, arguments)

        except (ConnectionError, TimeoutError, OSError) as e:
            if attempt == max_retries - 1:
                raise
            wait_time = backoff_base * (2 ** attempt)
            logger.warning(
                f"Tool call failed (attempt {attempt+1}/{max_retries}): {e}. "
                f"Retrying in {wait_time:.1f}s..."
            )
            await asyncio.sleep(wait_time)
```

<span class="hh-tag">Listing 30:</span>Resilient MCP Connection with Retry Logic

## The MCP Ecosystem

Since its release, MCP has attracted a rapidly growing ecosystem of servers, clients, and tooling.2

### Popular MCP Servers

### MCP in Production Applications

MCP has been adopted by several major AI development tools:

Claude Desktop

Anthropic’s desktop application3 was the first major MCP host. Users configure servers in a JSON config file; Claude can then use tools from all connected servers in any conversation.

Cursor

The AI-powered code editor4 supports MCP servers, allowing developers to connect their development tools (databases, issue trackers, documentation systems) directly to the coding assistant.

VS Code (GitHub Copilot)

Microsoft added MCP support5 to GitHub Copilot in VS Code, enabling the coding assistant to access project-specific tools and data sources.

Custom Agents

The open-source community has built MCP support into frameworks like LangChain6, LlamaIndex7, and AutoGen8, enabling any agent built on these frameworks to use MCP servers.

### Server Registries and Discovery

The MCP ecosystem is developing infrastructure for server discovery:

<ul><li>MCP Registry9: An official curated list of verified MCP servers maintained by Anthropic.</li><li>npm: Many JavaScript/TypeScript MCP servers are published as npm packages under the <code>@modelcontextprotocol</code> scope.</li><li>PyPI: Python servers are published as pip packages (e.g., <code>pip install mcp-server-sqlite</code>).</li><li>GitHub: The <code>modelcontextprotocol/servers</code>10 repository maintains a reference collection of official servers.</li><li>Python SDK documentation11: Full API reference and examples for building servers and clients.</li></ul>

## MCP vs. Alternatives

### When to Use MCP vs. Custom Integration

Use MCP when:

<ul><li>You want your tools to work with multiple LLM providers or agent frameworks</li><li>You are building tools that others will use (open-source or enterprise distribution)</li><li>You need stateful sessions, resource subscriptions, or server-push capabilities</li><li>You want to leverage the existing ecosystem of MCP servers</li></ul>

Use custom integration when:

<ul><li>You have a single, tightly-coupled LLM provider and no plans to switch</li><li>You need extremely low latency and cannot afford the protocol overhead</li><li>Your tool interface is so unusual that MCP primitives do not map well</li><li>You are in early prototyping and want to minimize dependencies</li></ul>

### Migration Paths

Migrating from OpenAI function calling to MCP is straightforward: the JSON Schema format for tool parameters is identical. The main changes are:

<ol><li>Wrap tool implementations in an MCP server (using the Python or TypeScript SDK)</li><li>Replace direct API calls with <code>session.call_tool()</code> in the client</li><li>Add capability negotiation and lifecycle management</li></ol>

LangChain tools can be wrapped in MCP servers using the <code>langchain-mcp-adapters</code> package, which provides automatic conversion between LangChain’s <code>BaseTool</code> interface and MCP tool definitions.

## MCP for Agent Training

Beyond deployment, MCP has significant implications for <em>training</em> tool-using agents. This section explores how MCP can serve as infrastructure for reinforcement learning and supervised fine-tuning of LLMs.

### MCP Servers as RL Environment Interfaces

In reinforcement learning for LLMs (see Section Chapter 3 Introduction to Reinforcement Learning), the agent must interact with an environment to receive rewards. MCP servers provide a natural, standardized interface for this:

<ul><li>Action space: The set of available tools defines the agent’s action space. MCP’s <code>tools/list</code> endpoint provides a structured, machine-readable action space that can be dynamically updated.</li><li>Observation space: MCP resources provide structured observations. A coding environment might expose the current file contents, test results, and error messages as resources.</li><li>Reward signals: Tool call results can encode reward signals. A test-running tool might return <code>{"passed": 8, "failed": 2, "reward": 0.8}</code> alongside the test output.</li><li>Environment reset: A <code>reset_environment</code> tool can restore the environment to its initial state between episodes.</li></ul>

### Standardized Action Spaces via MCP

One challenge in training tool-using agents is that different environments have different action spaces, making it difficult to transfer learned policies. MCP provides a universal action space abstraction:

<div class="hh-equation" id="ch22.e3"><math alttext="\mathcal{A}_{\text{MCP}}=\bigcup_{s\in\mathcal{S}}\text{Tools}(s)" display="block"><semantics><mrow><msub><mi>𝒜</mi><mtext>MCP</mtext></msub><mo rspace="0.111em">=</mo><mrow><munder><mo movablelimits="false">⋃</mo><mrow><mi>s</mi><mo>∈</mo><mi>𝒮</mi></mrow></munder><mrow><mtext>Tools</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>s</mi><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(22.3)</span></div>

where <math alttext="\mathcal{S}" display="inline"><semantics><mi>𝒮</mi></semantics></math> is the set of connected MCP servers and <math alttext="\text{Tools}(s)" display="inline"><semantics><mrow><mtext>Tools</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>s</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math> is the tool set of server <math alttext="s" display="inline"><semantics><mi>s</mi></semantics></math>. The agent learns a policy <math alttext="\pi(a\mid o,\mathcal{A}_{\text{MCP}})" display="inline"><semantics><mrow><mi>π</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>a</mi><mo fence="true" lspace="0em" rspace="0em">∣</mo><mrow><mi>o</mi><mo>,</mo><msub><mi>𝒜</mi><mtext>MCP</mtext></msub></mrow><mo stretchy="false">)</mo></mrow></mrow></semantics></math> that conditions on the available action set, enabling zero-shot generalization to new tool sets.

The JSON Schema format for tool parameters provides a structured action representation that the LLM can parse and generate reliably. This is more tractable than free-form API documentation and enables systematic exploration of the action space during training.

### Recording Tool-Use Trajectories for SFT

MCP’s structured protocol makes it easy to record high-quality tool-use trajectories for supervised fine-tuning:

```
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any
from mcp import ClientSession

@dataclass
class ToolCallRecord:
    timestamp: float
    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    duration_ms: float
    is_error: bool

@dataclass
class Trajectory:
    task_description: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    final_answer: str = ""
    success: bool = False
    total_reward: float = 0.0

class RecordingMCPClient:
    """Wraps an MCP session to record all tool calls for SFT data."""

    def __init__(self, session: ClientSession, trajectory: Trajectory):
        self.session = session
        self.trajectory = trajectory

    async def call_tool(self, name: str, arguments: dict) -> Any:
        start = time.monotonic()
        result = await self.session.call_tool(name, arguments)
        duration = (time.monotonic() - start) * 1000

        self.trajectory.tool_calls.append(ToolCallRecord(
            timestamp=time.time(),
            tool_name=name,
            arguments=arguments,
            result={"content": [c.text for c in result.content
                                 if hasattr(c, "text")]},
            duration_ms=duration,
            is_error=result.isError
        ))
        return result

    def save_trajectory(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self.trajectory), f, indent=2)
```

<span class="hh-tag">Listing 31:</span>Trajectory Recording Middleware for SFT Data Collection

Recorded trajectories can be converted to instruction-following training examples:

```
def trajectory_to_sft_example(traj: Trajectory) -> dict:
    """Convert a recorded MCP trajectory to a chat-format SFT example."""
    messages = [
        {"role": "system", "content": (
            "You are a helpful assistant with access to tools. "
            "Use tools to complete tasks step by step."
        )},
        {"role": "user", "content": traj.task_description}
    ]

    for i, call in enumerate(traj.tool_calls):
        call_id = f"call_{i:04d}"
        # Assistant decides to call a tool
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": call.tool_name,
                    "arguments": json.dumps(call.arguments)
                }
            }]
        })
        # Tool returns a result
        messages.append({
            "role": "tool",
            "content": json.dumps(call.result),
            "tool_call_id": call_id,
        })

    # Final answer
    messages.append({
        "role": "assistant",
        "content": traj.final_answer
    })

    return {
        "messages": messages,
        "metadata": {
            "success": traj.success,
            "reward": traj.total_reward,
            "num_tool_calls": len(traj.tool_calls)
        }
    }
```

<span class="hh-tag">Listing 32:</span>Converting MCP Trajectories to SFT Training Examples

## Summary

The Model Context Protocol represents a significant step toward standardizing how AI agents interact with the world. By reducing the <math alttext="N\times M" display="inline"><semantics><mrow><mi>N</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mi>M</mi></mrow></semantics></math> integration problem to <math alttext="N+M" display="inline"><semantics><mrow><mi>N</mi><mo>+</mo><mi>M</mi></mrow></semantics></math>, MCP lowers the barrier to building capable, tool-augmented AI systems. Its key design decisions—JSON-RPC 2.0 as the wire format, stateful sessions, four core primitives (tools, resources, prompts, sampling), and a clear security model—reflect hard-won lessons from the LSP and USB ecosystems.

For practitioners building RL-trained agents, MCP offers a particularly compelling value proposition: a standardized, extensible interface for defining action spaces, collecting training trajectories, and deploying trained agents across diverse environments. As the ecosystem matures and benchmark suites emerge, MCP may become the de facto substrate for tool-using agent research—the gymnasium of the LLM era.

<aside class="hh-next">
<p class="hh-next-label">下一篇</p>
<p class="hh-next-title">Chapter 23 Agent Skills</p>
<p class="hh-next-hint">在「智能体 AI 漫游指南」分类下继续阅读。</p>
</aside>

</div>
