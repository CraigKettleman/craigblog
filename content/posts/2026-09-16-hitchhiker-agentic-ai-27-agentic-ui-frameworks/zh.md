---
title: "智能体 AI 漫游指南 · 第 27 章 Agentic UI 框架"
slug: "hitchhiker-agentic-ai-27-agentic-ui-frameworks"
lang: "zh"
date: "2026-09-16T00:28:00.000Z"
draft: false
featured: false
categories:
  - "hitchhiker-agentic-ai"
cover: "./cover.png"
description: "随着大语言模型从被动的文本生成器演进为能够规划、调用工具并进行多步推理的主动 Agent，人类与之交互的界面也必须同步演化。传统聊天界面——为单轮或短上下文对话而设计——无法满足 Agent 化工作流的要求：长时间运行的任务、分支决策树、并行的工具调用，以及对有效人类监督的需求。…"
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

随着大语言模型从被动的文本生成器演进为能够规划、调用工具并进行多步推理的主动 Agent，人类与之交互的界面也必须同步演化。传统聊天界面——为单轮或短上下文对话而设计——无法满足 Agent 化工作流的要求：长时间运行的任务、分支决策树、并行的工具调用，以及对有效人类监督的需求。本节考察 <em>Agent 化 UI 框架（agentic UI frameworks）</em>的图景：使丰富、透明、可信赖的人机协作成为可能的设计范式、组件库与实现模式。

## 动机：超越聊天框

聊天界面与 Agent 化界面之间的差距，犹如自动售货机与熟练协作者之间的差距。当一个 Agent 执行 20 步研究任务、浏览网页、撰写并运行代码、综合成一份报告时，用户需要解答的问题是简单的文本响应所无法提供的：

<ul><li>Agent 此刻在做什么？长时间运行的任务需要进度反馈；沉默会滋生不信任。</li><li>Agent 为何做出这个决策？推理过程的透明性使用户能够尽早发现错误。</li><li>使用了哪些工具，输入了什么？工具来源（tool provenance）对核实事实性论断与审计行为至关重要。</li><li>我应该在哪里介入？Agent 必须呈现那些需要人类判断的决策点，同时避免用每一项微决策淹没用户。</li><li>我能撤销吗？不可逆操作（发送邮件、修改文件、执行代码）需要明确的确认与回滚路径。</li></ul>

因此，Agent 化 UI 的设计位于人机交互（Human-Computer Interaction, HCI）、可解释 AI（Explainable AI, XAI）与软件工程的交汇处。核心设计目标包括：

<ol><li>透明性：让 Agent 的内部状态对用户清晰可读。</li><li>可控性：提供有意义的介入节点，而不要求持续监督。</li><li>信任校准：帮助用户建立关于 Agent 能力与局限的准确心智模型。</li><li>效率：最小化认知负担；在合适的时间呈现合适的信息。</li><li>可恢复性：让错误的发现与撤销成本低廉。</li></ol>

## Agent 的 UI 范式

没有任何单一的 UI 范式能适应所有 Agent 化用例。合适的界面取决于任务时长、所需的人类介入程度、输出类型以及用户的专业水平。其谱系从完全对话式的聊天界面，跨越到几乎无需人类交互的完全自主仪表盘。

### 基于聊天的界面

聊天范式——消息气泡、文本输入框和滚动历史——仍然是与 LLM 交互最为熟悉的入口。其优势在于学习曲线平缓与自然语言的灵活性。在 Agent 化场景下，聊天界面通常会扩展以下能力：

<ul><li>流式响应：Token 在生成时即逐步出现，提供即时反馈并降低感知延迟。通过服务器推送事件（Server-Sent Events, SSE）或 WebSockets 实现。</li><li>内联工具指示器：消息流中的小徽章或可展开区段，用于显示何时调用了工具（例如“<code>[Searched the web for: climate change 2024]</code>”）。</li><li>输入指示与状态消息：“Agent is thinking…”、“Running Python code…”、“Fetching results…”等提示在延迟间隙保持用户的知情感。</li><li>消息线程化：对于多轮 Agent 任务，可折叠的子线程能容纳中间步骤，而不会让主对话变得杂乱。</li></ul>

### 画布与产物式界面

画布范式（canvas paradigm）由 Claude Artifacts1 与 ChatGPT Canvas2 推广，引入了 <em>分屏</em>布局：左侧承载对话，右侧（“canvas” 或 “artifact panel”）将生成的内容——代码、文档、图表、电子表格——以可实时编辑的产物（artifact）形式呈现。

关键特征：

<ul><li>持久化产物：生成的内容跨轮次持久存在，可以通过自然语言指令迭代优化（“把图表改成蓝色”、“给这个函数加上错误处理”）。</li><li>原位编辑：用户可以直接编辑产物，Agent 也能观察并响应这些编辑。</li><li>版本历史：产物维护修订历史，支持回滚到任意先前状态。</li><li>多产物工作区：进阶实现支持多个并存的产物（例如一个代码文件、其测试套件以及一个文档页面）。</li></ul>

画布范式特别适合 <em>共创</em>任务：写作、编程、数据分析与设计——这些场景的产出是文档或产物，而非对话式的回答。

### 工作流可视化

对于执行结构化计划（步骤序列或步骤图）的 Agent，工作流可视化 UI 使计划显式化并可追踪。该范式常见于：

<ul><li>Agent 化流水线（LangGraph、AutoGen、CrewAI）：将 Agent 的执行图渲染为有向无环图（Directed Acyclic Graph, DAG）或流程图，节点代表步骤，边代表数据流或控制流。</li><li>任务分解视图：将 Agent 的高层计划呈现为清单或甘特图式的时间线，每个子任务可展开以显示其自身的步骤。</li><li>实时进度追踪：节点在执行时改变颜色或显示加载图标；完成的节点展示输出；失败的节点展示错误细节。</li></ul>

LangGraph Studio3 是该范式的典范，它为 LangGraph Agent 提供基于图的调试器与可视化工具。用户可以检查每个节点的状态、回放执行过程，并注入修改后的状态以测试备选路径。

### 仪表盘与监控界面

对于长时间运行或生产环境中的 Agent，仪表盘 UI 提供运维视角：

<ul><li>实时状态：哪些 Agent 正在运行、空闲或失败；当前任务与步骤。</li><li>资源指标：Token 消耗、API 调用计数、延迟直方图、成本估算。</li><li>队列管理：待处理任务、优先级排序、限流状态。</li><li>告警与异常检测：异常行为（过多的重试、成本激增、反复失败）以通知形式呈现。</li><li>历史分析：任务完成率、平均时长、随时间变化的错误频率。</li></ul>

仪表盘 UI 通常使用 Grafana4、自定义 React 仪表盘或 Streamlit 等工具构建，面向的是 <em>运维人员</em>而非最终用户。

### 协作型界面

协作式 UI 将 Agent 视为共享工作区——文档、代码库或设计画布——中与人类协作者并肩的对等贡献者。关键特征包括：

<ul><li>在线状态指示：Agent 在共享工作区中以具名光标或头像的形式出现。</li><li>变更归属：Agent 所做的编辑与人类编辑在视觉上加以区分（例如用颜色标出的 diff）。</li><li>内联建议：Agent 以修订标记或评论的形式提出修改，人类可以接受、拒绝或修改。</li><li>冲突解决：当 Agent 与人类同时编辑同一区域时，UI 会呈现冲突并促成解决。</li></ul>

该范式正在 Cursor5（结合 AI 的协作式代码编辑）、Notion AI6 以及集成 Gemini 的 Google Docs7 等工具中兴起。

### 带检查点的自主模式

在自主性谱系的另一端，一些 Agent 基本独立运行——浏览网页、编写代码、执行命令——只在预设的 <em>检查点</em>处才现身并要求人类批准。该范式用于：

<ul><li>计算机操作类 Agent（Anthropic Computer Use8、OpenAI Operator9）：Agent 控制浏览器或桌面；UI 显示实时屏幕画面，并在不可逆操作前暂停等待批准。</li><li>带门禁的自动化流水线：CI/CD 式工作流，Agent 完成一个阶段后等待人类“合并”再继续。</li><li>定时 Agent：按计划运行并异步汇报结果的 Agent，配有基于通知的 UI，用于审阅输出并批准后续操作。</li></ul>

## Agent 的关键 UI 组件

无论采用哪种总体范式，Agent 化 UI 都共享一组反复出现的组件。本节梳理其中最重要的部分，并给出各自的设计指引。

### 思考与推理展示

现代 LLM，尤其是那些以思维链或扩展思考方式训练的模型（例如 OpenAI o1/o3、带扩展思考的 Anthropic Claude），在给出最终响应之前会生成大量内部推理。将这一推理呈现出来是一把双刃剑：它提升了透明性，却也可能用冗长的内心独白淹没用户。

最佳实践：

<ul><li>可折叠的推理块：显示一句摘要（“思考了 12 秒”），并附展开开关供想要查看细节的用户使用。</li><li>渐进式披露：默认只显示最终结论；推理按需提供。</li><li>结构化推理：如果模型产出结构化的思考内容（假设、证据、结论），应以视觉层级加以呈现，而非堆成一整面文字墙。</li><li>推理与响应的区分：在视觉上清晰区分内部推理（其中可能包含错误或错误的起点）与最终响应。</li></ul>

### 工具使用可视化

工具调用是 Agent 与世界交互的主要机制。将其可视化对建立信任与调试至关重要。

工具可视化的设计模式：

<ul><li>内联工具卡片：消息流中的紧凑卡片，显示工具名称、一行输入摘要和状态（运行中/成功/错误）。可展开查看完整细节。</li><li>工具时间线：横向时间线，展示一轮中所有工具调用及其耗时，便于识别瓶颈。</li><li>输入/输出 diff：对于会修改状态的工具（例如文件编辑），展示前后对比的 diff。</li><li>工具图标与品牌标识：为常见工具（网页搜索、代码执行、文件系统、API）提供可识别的图标，以支持快速扫视。</li><li>错误高亮：失败的工具调用以红色显示，并附上错误消息与任何重试尝试。</li></ul>

### 进度指示器

多步 Agent 任务需要丰富的进度反馈：

<ul><li>步骤级进度：以编号列表列出计划步骤，每完成一步打勾。对于动态计划，步骤可随 Agent 的调整而增删。</li><li>Token 流式指示器：生成期间显示闪烁光标或动画省略号；为高级用户提供每秒 Token 数计数器。</li><li>预计完成时间：在可行的情况下，基于任务复杂度与历史表现给出 ETA。以适当的不确定性呈现（“大约 2–5 分钟”）。</li><li>子任务嵌套：对于层级化任务，采用树形进度视图，子任务可展开。</li><li>取消：一个清晰可见的“停止”按钮，能够优雅地中止 Agent 并总结目前已完成的工作。</li></ul>

### 审批门控

审批门禁是人在回路控制的主要机制。它们的设计必须做到 <em>信息充分</em>（给用户足够的上下文以做出良好决策），同时不至于 <em>令人疲劳</em>（为每一个琐碎操作都要求批准）。

审批门禁的 UI 元素：

<ul><li>操作摘要：用平实语言描述 Agent 想做什么（“向 john@example.com 发送一封带附件的报告邮件”）。</li><li>风险指示器：以视觉信号表示操作的可逆性（绿色 = 容易撤销，黄色 = 难以撤销，红色 = 不可逆）。</li><li>批准 / 拒绝 / 修改：三选一界面；“修改”会在批准前打开一个操作参数编辑器。</li><li>上下文面板：可展开区段，说明 Agent 为何想采取此操作（相关推理、先前步骤）。</li><li>超时行为：清楚说明用户不作响应时会发生什么（Agent 暂停，而非继续）。</li></ul>

### 上下文展示

Agent 会维护影响其行为的内部状态——记忆、启用的工具、检索到的文档、对话历史。让这一状态可见有助于用户理解并预测 Agent 的行为。

<ul><li>记忆面板：显示 Agent 当前对用户、任务及先前交互“记住”了什么。用户可编辑。</li><li>启用工具列表：Agent 当前可用的工具有哪些，并提供启用/禁用开关。</li><li>检索到的上下文：当前位于 Agent 上下文窗口中的文档或数据块，并附来源引用。</li><li>Token 预算指示器：上下文窗口已被消耗多少，帮助用户判断何时该开启新会话。</li></ul>

### 错误与恢复 UI

Agent 会失败——工具返回错误、模型产生幻觉、计划变得不可行。UI 必须优雅地处理失败：

<ul><li>错误卡片：内联展示失败，包含错误类型、消息以及 Agent 的解读。</li><li>重试控件：手动重试按钮，可选地调整参数。</li><li>替代方案：当主要方案失败时，Agent 提出替代方案；UI 将其呈现为可选项。</li><li>部分结果：如果多步任务中途失败，UI 展示已完成的步骤及其输出，保留部分价值。</li><li>升级路径：当 Agent 无法继续时，提供通往人工支持或手动完成的明确路径。</li></ul>

### 置信度指示器

LLM 是带有校准（或校准不良）不确定性的概率系统。呈现置信度有助于用户判断何时该信任、何时该核实：

<ul><li>言语对冲展示：高亮“我不确定”或“你可能需要核实”之类的措辞，以引起对低置信度论断的注意。</li><li>来源质量指示器：对于检索到的信息，显示来源的时效性、权威性与相关性得分。</li><li>显式不确定性询问：一个“你有多确定？”按钮，促使 Agent 自我评估并解释其不确定性。</li><li>核实建议：对于高风险输出，Agent 主动建议核实步骤（“我建议独立复核这一计算”）。</li></ul>

## 框架与库

不断壮大的框架生态正在加速 Agent 化 UI 的开发。我们按主要语言与用例梳理其中采用最广的一些。

### Vercel AI SDK

Vercel AI SDK [358] 是一个 TypeScript/JavaScript 库，用于在 React、Next.js、Svelte 和 Vue 中构建流式 AI 界面。它是生产级 Web Agent UI 中使用最广的框架。

核心抽象：

<ul><li><code>useChat</code>：一个 React hook，用于管理聊天会话，支持流式传输、消息历史与加载状态。</li><li><code>useCompletion</code>：用于单轮文本补全并支持流式传输的 hook。</li><li><code>useObject</code>：流式传输结构化 JSON 对象，支持复杂输出的渐进式渲染。</li><li><code>streamText</code> / <code>streamObject</code>：在 HTTP 上流式传输 LLM 响应的服务端函数。</li></ul>

生成式 UI（AI SDK RSC）：Vercel AI SDK 最具特色的能力，是通过 React Server Components（RSC）对 <em>生成式 UI</em>的支持。LLM 不必返回文本，而是可以调用工具，其运行结果会渲染为任意 React 组件——一个天气小组件、一张股票走势图、一个预订表单——并直接流式传输进 UI。这一点将在第 27.5 生成式 UI 节进一步讨论。

### Chainlit

Chainlit [40] 是一个 Python 框架，用于以最少的样板代码构建可用于生产的 Agent UI。它在 LangChain 与 LlamaIndex 生态中尤为流行。

关键特性：

<ul><li>步骤可视化：Chainlit 原生地将 LangChain 与 LlamaIndex 的执行步骤渲染为可折叠的树，展示每次链式调用、检索与工具调用。</li><li>多模态支持：开箱即用的文件上传、图像显示、音频播放与 PDF 渲染。</li><li>认证与会话：内置用户认证、持久化对话历史与多用户支持。</li><li>自定义元素：可以注册 React 组件并从 Python 渲染，从而实现丰富的自定义可视化。</li><li>反馈收集：内置的点赞/点踩反馈，可附评论，并存储到数据库。</li></ul>

```
import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

agent = create_react_agent(
    ChatOpenAI(model="gpt-4o"), tools=[search]
)

@cl.on_message
async def on_message(message: cl.Message):
    # Chainlit automatically renders each step as a collapsible UI element
    # when using the callback handler
    async with cl.Step(name="Agent", type="run") as step:
        step.input = message.content
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": message.content}]},
            config={"callbacks": [cl.LangchainCallbackHandler()]}
        )
        output = result["messages"][-1].content
        step.output = output

    await cl.Message(content=output).send()
```

<span class="hh-tag">代码清单 66：</span>带步骤可视化的最小 Chainlit Agent

### Gradio

Gradio [1] 是一个 Python 库，用于快速构建机器学习演示与 Agent 界面。它的 <code>gr.ChatInterface</code> 与 <code>gr.Blocks</code> API 让用户以极少的代码快速原型化对话式 Agent。

面向 Agent UI 的优势：

<ul><li>零配置部署：通过 Hugging Face Spaces 一行代码即可分享。</li><li>自定义组件：Gradio Custom Components 系统允许构建能与 Python 后端无缝集成的 React 组件。</li><li>多模态输入：以极简配置支持文件上传、图像、音频、视频与摄像头输入。</li><li>流式传输：原生支持基于生成器的流式响应。</li></ul>

局限：Gradio 的布局系统不如完整的 React 框架灵活，其状态管理以会话为作用域，这使得复杂的多 Agent 协调颇具挑战。

### Streamlit

Streamlit [156] 是一个用于数据应用的 Python 框架，已被广泛用于 Agent 仪表盘与监控 UI。它的响应式执行模型——每次交互都重跑整个脚本——简单，但对复杂的 Agent 化工作流可能构成限制。

Agent 化用例：

<ul><li>Agent 仪表盘：使用 <code>st.metric</code>、<code>st.dataframe</code> 与 <code>st.status</code> 实现实时指标、任务队列与状态显示。</li><li>会话状态：<code>st.session_state</code> 在重跑之间保持 Agent 状态，从而支持多轮对话。</li><li>流式传输：<code>st.write_stream</code> 渐进式渲染生成器输出。</li><li>片段（Fragments）：<code>@st.fragment</code> 装饰器支持部分重跑，提升实时更新仪表盘的性能。</li></ul>

### OpenAI Assistants Playground

OpenAI Assistants Playground 可作为 Agent 化 UI 设计的参考实现。它展示了：

<ul><li>基于线程的对话管理与持久化历史。</li><li>文件附件与检索的可视化。</li><li>代码解释器执行及输出展示（stdout、图像、文件）。</li><li>函数调用展示，可检视输入/输出。</li><li>运行步骤可视化，展示模型调用与工具调用的序列。</li></ul>

虽然它不是用于构建自定义 UI 的框架，但 Playground 的设计模式被广泛效仿。

### LangGraph Studio

LangGraph Studio [154] 是一个桌面应用，为 LangGraph Agent 提供可视化 IDE。它是目前可用的最完善的工具使用与工作流可视化环境。

特性：

<ul><li>图可视化：以交互方式渲染 Agent 的状态机，节点代表 Agent 步骤，边代表状态转移。</li><li>状态检视：在执行过程中的任意时刻，都可以将完整的 Agent 状态（所有变量、记忆、工具结果）作为结构化 JSON 进行检视。</li><li>时间旅行调试：回放任意先前的执行步骤，修改状态，并从该点重新运行。</li><li>人在回路集成：可以在任意节点上设置断点；执行暂停并等待人类输入后再继续。</li><li>多 Agent 支持：可视化主管-子 Agent 层级结构以及 Agent 间的消息传递。</li></ul>

### 框架对比

表 表 27.1 总结了上述框架的关键特征。

<div class="hh-table" id="ch27.t1"><table class="ltx_tabular ltx_align_middle"><tbody><tr><td class="ltx_align_left"><span>框架</span></td><td class="ltx_align_left"><span>语言</span></td><td class="ltx_align_center"><span>流式传输</span></td><td class="ltx_align_center"><span>工具可视化</span></td><td class="ltx_align_center"><span>多 Agent</span></td><td class="ltx_align_center"><span>生成式 UI</span></td><td class="ltx_align_center"><span>生产可用</span></td></tr><tr><td class="ltx_align_left"><span>Vercel AI SDK</span></td><td class="ltx_align_left"><span>TypeScript</span></td><td class="ltx_align_center"><span>✓</span></td><td class="ltx_align_center"><span>部分</span></td><td class="ltx_align_center"><span>部分</span></td><td class="ltx_align_center"><span>✓</span></td><td class="ltx_align_center"><span>✓</span></td></tr><tr><td class="ltx_align_left"><span>Chainlit</span></td><td class="ltx_align_left"><span>Python</span></td><td class="ltx_align_center"><span>✓</span></td><td class="ltx_align_center"><span>✓</span></td><td class="ltx_align_center"><span>部分</span></td><td class="ltx_align_center"><span>部分</span></td><td class="ltx_align_center"><span>✓</span></td></tr><tr><td class="ltx_align_left"><span>Gradio</span></td><td class="ltx_align_left"><span>Python</span></td><td class="ltx_align_center"><span>✓</span></td><td class="ltx_align_center"><math alttext="\circ" display="inline"><semantics><mo mathsize="0.800em">∘</mo><span></span></semantics></math></td><td class="ltx_align_center"><math alttext="\times" display="inline"><semantics><mo mathsize="0.800em">×</mo><span></span></semantics></math></td><td class="ltx_align_center"><math alttext="\circ" display="inline"><semantics><mo mathsize="0.800em">∘</mo><span></span></semantics></math></td><td class="ltx_align_center"><span>✓</span></td></tr><tr><td class="ltx_align_left"><span>Streamlit</span></td><td class="ltx_align_left"><span>Python</span></td><td class="ltx_align_center"><span>✓</span></td><td class="ltx_align_center"><math alttext="\circ" display="inline"><semantics><mo mathsize="0.800em">∘</mo><span></span></semantics></math></td><td class="ltx_align_center"><math alttext="\times" display="inline"><semantics><mo mathsize="0.800em">×</mo><span></span></semantics></math></td><td class="ltx_align_center"><math alttext="\times" display="inline"><semantics><mo mathsize="0.800em">×</mo><span></span></semantics></math></td><td class="ltx_align_center"><span>✓</span></td></tr><tr><td class="ltx_align_left"><span>OAI Playground</span></td><td class="ltx_align_left"><span>N/A（托管服务）</span></td><td class="ltx_align_center"><span>✓</span></td><td class="ltx_align_center"><span>✓</span></td><td class="ltx_align_center"><math alttext="\times" display="inline"><semantics><mo mathsize="0.800em">×</mo><span></span></semantics></math></td><td class="ltx_align_center"><math alttext="\times" display="inline"><semantics><mo mathsize="0.800em">×</mo><span></span></semantics></math></td><td class="ltx_align_center"><math alttext="\times" display="inline"><semantics><mo mathsize="0.800em">×</mo><span></span></semantics></math></td></tr><tr><td class="ltx_align_left"><span>LangGraph Studio</span></td><td class="ltx_align_left"><span>Python/TS</span></td><td class="ltx_align_center"><span>✓</span></td><td class="ltx_align_center"><span>✓</span></td><td class="ltx_align_center"><span>✓</span></td><td class="ltx_align_center"><math alttext="\times" display="inline"><semantics><mo mathsize="0.800em">×</mo><span></span></semantics></math></td><td class="ltx_align_center"><span>部分</span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">表 27.1：</span>Agent 化 UI 框架对比。</p></div>

## 生成式 UI

生成式 UI 代表了 LLM 与界面之间关系的一次根本性转变。不再是开发者预先规定所有可能的 UI 状态，而是由模型根据当前语境动态选择并参数化合适的 UI 组件。

### 用 React Server Components 构建动态界面

Vercel AI SDK 的 RSC（React Server Components10）集成是生成式 UI 最成熟的实现。其架构运作如下：

<ol><li>用户向 Next.js11 服务器 action 发送一条消息。</li><li>服务器带着一组工具调用 LLM，每个工具都与一个 React 组件相关联。</li><li>当 LLM 调用某个工具（例如 <code>show_weather</code>）时，服务器用工具的输出作为 props 渲染对应的 React 组件。</li><li>渲染出的组件作为 React Server Component 流式传输到客户端，内联出现在聊天中。</li></ol>

```
// app/actions.tsx (Server Action)
import { streamUI } from 'ai/rsc';
import { openai } from '@ai-sdk/openai';
import { WeatherCard } from '@/components/WeatherCard';
import { StockChart } from '@/components/StockChart';

export async function chat(userMessage: string) {
  const result = await streamUI({
    model: openai('gpt-4o'),
    messages: [{ role: 'user', content: userMessage }],
    tools: {
      show_weather: {
        description: 'Display current weather for a location',
        parameters: z.object({
          location: z.string(),
          unit: z.enum(['celsius', 'fahrenheit']),
        }),
        // Tool result rendered as a React component
        generate: async ({ location, unit }) => {
          const data = await fetchWeather(location, unit);
          return <WeatherCard data={data} />;
        },
      },
      show_stock: {
        description: 'Display stock price chart',
        parameters: z.object({ ticker: z.string() }),
        generate: async ({ ticker }) => {
          const data = await fetchStockData(ticker);
          return <StockChart ticker={ticker} data={data} />;
        },
      },
    },
  });
  return result.value;
}
```

<span class="hh-tag">代码清单 67：</span>使用 Vercel AI SDK RSC 的生成式 UI（TypeScript）

### 基于内容类型的自适应界面

生成式 UI 使界面能够适应所呈现内容的性质：

<ul><li>表格数据<math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 可排序、可筛选的数据表，并带导出选项。</li><li>地理数据<math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 带标记与图层的地图。</li><li>时间序列<math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 可缩放、带注释的折线图。</li><li>代码<math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 带运行按钮的语法高亮编辑器。</li><li>文档<math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 带注释工具的格式化文档查看器。</li><li>表单/结构化输入<math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 动态生成的表单字段。</li></ul>

模型充当 <em>UI 编排器（UI orchestrator）</em>，为每一条信息选择最合适的呈现方式。这降低了开发者预先设想所有可能的输出类型并提前构建对应组件的需要。

## 流式与实时模式

流式传输（Streaming）是 Agent 化 UI 的根基：它把体验从「等待一个结果」转变为「观看 Agent 工作」。本节讨论关键的流式传输模式及其实现考量。

### Token 流

Token 流式传输在 Token 生成时逐步交付 LLM 输出，而不是等待完整响应。通常使用两种传输机制：

<ul><li>服务器推送事件（Server-Sent Events，SSE）12：从服务器到客户端的单向 HTTP 流。每个事件携带一段 Token。SSE 简单，可运行在标准 HTTP/1.1 之上，浏览器会自动重连。它是 LLM 流式 API 的主流机制（OpenAI、Anthropic、Google 都使用 SSE）。</li><li>WebSockets：双向持久连接。实现起来更复杂，但对于客户端需要在流中途发送数据的交互式流式传输场景（例如中断 Agent、在生成过程中提供反馈）而言是必需的。</li></ul>

```
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
import json

app = FastAPI()
client = AsyncOpenAI()

async def token_stream(prompt: str):
    """Generator that yields SSE-formatted token chunks."""
    stream = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            # SSE format: "data: <json>\n\n"
            yield f"data: {json.dumps({'token': delta.content})}\n\n"
        elif chunk.choices[0].finish_reason:
            yield f"data: {json.dumps({'done': True})}\n\n"

@app.get("/stream")
async def stream_endpoint(prompt: str):
    return StreamingResponse(
        token_stream(prompt),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

<span class="hh-tag">代码清单 68：</span>使用 FastAPI 的 SSE Token 流式传输

### 工具调用流

现代 LLM API 支持流式工具调用：工具名称与参数被逐步流式传输，使 UI 能够在工具尚未被真正调用之前就显示「Agent 正在以查询 'climate change 2024' 调用 <code>search_web</code>……」。这需要解析不完整的 JSON，可以借助流式 JSON 解析器完成。

工具调用流式传输的模式：

<ul><li>渐进式参数展示：在参数流入时即展示工具参数，即使调用尚未完成。</li><li>并行工具调用指示器：当模型同时调用多个工具时，将它们全部显示为待处理，并在结果到达时逐个更新。</li><li>工具结果流式传输：某些工具（例如代码执行、网页抓取）本身就能流式输出结果；将这些结果逐步传递到 UI。</li></ul>

### 多 Agent 流

在多 Agent 系统中，多个 Agent 可能同时生成输出。UI 必须处理并行的流：

<ul><li>带 Agent 标签的流：每条流都标注了所属 Agent 的身份；UI 在各自独立的通道或面板中渲染它们。</li><li>流合并：在 supervisor–subagent 模式中，supervisor 的流可能与 subagent 的流交错；UI 必须维持一致的顺序。</li><li>反压（Backpressure）：如果 UI 的渲染速度跟不上流的到达速度（例如多个 Agent 同时生成），就必须有反压机制来防止缓冲区溢出。策略包括：丢弃中间 Token（只显示最新的）、合并为批量更新，或暂停较慢的流。</li></ul>

### 乐观 UI 更新

乐观 UI 更新通过在服务器确认之前就立即把用户操作反映到 UI 中，提升感知到的响应速度：

<ul><li>当用户发送一条消息时，该消息会在请求仍在进行期间立即（乐观地）出现在聊天历史中。</li><li>当审批门控被接受时，UI 会立即把该操作显示为「已批准」，并开始显示 Agent 的后续步骤，即使服务器尚未处理完该批准。</li><li>如果服务器返回错误，乐观更新会被回滚，并显示错误状态。</li></ul>

### 背压处理

在高吞吐的 Agent 化场景中，数据到达速率可能超过 UI 的渲染能力。管理反压的策略：

<ul><li>Token 批处理：将 Token 缓冲 50–100ms 后批量渲染，而不是逐个渲染，从而降低 DOM 更新频率。</li><li>虚拟滚动：对于很长的输出，只渲染内容中可见的部分，丢弃屏幕外的 DOM 节点。</li><li>限流更新：对于指标与状态显示，无论数据到达速率如何，都以固定频率更新（例如 10 Hz）。</li><li>渐进式细节：在高吞吐时段显示摘要视图；完整细节按需提供。</li></ul>

## 人机协同 UI 设计

人在回路（Human-in-the-loop，HITL）交互是 Agent 化 UI 中最具影响的设计挑战之一。目标是在维持有意义的人类监督的同时，不产生一个抵消自动化效率收益的瓶颈。

### 何时打断 Agent

并非所有 Agent 操作都需要人工审查。一个有原则的中断策略需要考虑：

<ul><li>可逆性：不可逆操作（删除文件、发送邮件、进行购买）总是需要审批。可逆操作（读取文件、搜索网页）通常不需要。</li><li>作用域：影响外部系统或他人的操作比纯本地操作需要更严格的审查。</li><li>置信度：当 Agent 对自身解读用户意图的置信度较低时，应请求澄清而不是继续执行。</li><li>成本：高成本操作（大规模 API 调用、昂贵计算）需要审批。</li><li>新颖性：Agent 在此上下文中此前未执行过的操作比常规操作需要更严格的审查。</li></ul>

### 分层审批工作流

分级审批策略在监督与效率之间取得平衡：

各层级之间的阈值可以由用户配置（「发送邮件前总是先询问」），也可以从用户行为中学习（如果用户总是批准网页搜索，未来就自动批准它们）。

### 反馈机制

除审批门控之外，Agent 化 UI 还应提供丰富的反馈机制，帮助 Agent 随时间不断改进：

<ul><li>点赞/点踩：对响应的简单二值反馈，会被存储并用于 RLHF 微调或偏好学习。</li><li>行内修正：用户可以直接编辑 Agent 的输出；原始输出与修正后输出之间的差异即是一个训练信号。</li><li>偏好选择：当 Agent 提供多个选项时，用户的选择即是一个偏好信号。</li><li>显式指令：「不要再这样做」「X 之前总是先询问」「优先选择方法 Y 而非 Z」——这些自然语言指令会更新 Agent 的行为策略。</li><li>带理由的评分：评分时可选附带的自由文本说明，比二值反馈提供更丰富的信号。</li></ul>

### 通过 UI 交互教会 Agent

最成熟的 HITL UI 把每一次交互都视为一次教学机会：

<ul><li>示范：用户手动执行一项任务；Agent 观察并学习其偏好的做法。</li><li>带泛化的修正：当用户修正某个 Agent 操作时，UI 会询问「我应该始终换一种做法吗？」以将该修正泛化。</li><li>偏好引出：定期提示用户比较两种 Agent 行为并指出更偏好哪一种。</li><li>行为画像：UI 维护一份可见的「偏好」画像，用户可查看并编辑，使 Agent 习得的行为透明且可控。</li></ul>

## 无障碍与信任

信任不是一项功能——它是一个始终如预期行事、能清楚地解释自身、并能从失败中优雅恢复的系统所涌现出的性质。Agent 化 UI 的设计必须把信任作为一等关注点。

### 解释 Agent 的决策

Agent 化 UI 中的可解释性不止于展示思维链（chain-of-thought）。它要求：

<ul><li>决策理由：对于影响重大的决策，Agent 不仅应解释它决定了<em>什么</em>，还应解释<em>为什么</em>——考虑了哪些因素、否决了哪些备选方案、以及做了哪些假设。</li><li>来源归属：论断应链接到其来源；检索到的文档应可引用。</li><li>反事实解释：「如果你说的是 X 而不是 Y，我会做 Z」——帮助用户理解 Agent 的决策边界。</li><li>不确定性量化：明确说明置信度，以及驱动不确定性的因素。</li></ul>

### 展示置信度水平

置信度指示器必须经过校准且有意义：

<ul><li>语言化置信度：对大多数用户而言，自然语言表达（「我相当有信心」「这点我不太确定」）比数值概率更易解读。</li><li>视觉化置信度：颜色编码（绿/黄/红）、图标变体或字重可在不增加文字的情况下编码置信度。</li><li>按论断的置信度：对于含多项论断的响应，逐项的置信度指示（例如内联脚注）比单一的响应级评分更具信息量。</li></ul>

### 撤销与回滚能力

每一项重要的 Agent 操作在技术可行时都应可撤销：

<ul><li>带撤销的操作日志：所有 Agent 操作的时序日志，每个可逆操作旁附「撤销」按钮。</li><li>基于快照的回滚：对于有状态任务（例如代码编辑、文档写作），定期快照支持回滚到任意先前状态。</li><li>Dry-run 模式：在执行计划之前，Agent 可模拟运行并展示预测的状态变化，允许用户在任何真实操作发生前批准或修改。</li><li>优雅降级：当撤销不可行时（例如邮件已发出），UI 应清楚说明，并提供最佳可行的替代方案（例如发送跟进邮件）。</li></ul>

### UI 中的审计轨迹

对于企业与受监管的使用场景，审计轨迹（audit trails）必不可少：

<ul><li>不可变操作日志：每一次 Agent 操作、工具调用与人类批准都会被记录，并附带时间戳、用户身份与完整参数。</li><li>可导出的历史：审计轨迹可导出为 JSON、CSV 或 PDF，用于合规报告。</li><li>Diff 视图：对于文档或代码修改，审计轨迹包含前后对比 diff。</li><li>会话回放：可逐步回放整个 Agent 会话，用于调试或合规审查。</li></ul>

### 管理用户期望

失准的期望是用户不信任的主要来源。Agent 化 UI 应主动管理期望：

<ul><li>能力披露：清晰、易获取的文档，说明 Agent 能做什么、不能做什么。</li><li>局限承认：当 Agent 遇到超出其能力的任务时，应明确说明，而不是默默尝试并失败。</li><li>不确定性沟通：主动沟通不确定性，而非等待用户去发现错误。</li><li>一致的人格：一致的 Agent 身份与沟通风格能建立熟悉感与可预测性。</li></ul>

## 实现示例：全栈 Agent 化 UI

现在给出一个具体的实现示例，将流式传输、工具可视化与审批门控组合于一个 Python/React 技术栈中。Backend 使用 FastAPI 配合 LangGraph；Frontend 使用 React，借鉴 Vercel AI SDK 的模式并适配自定义 Backend。

### Backend：FastAPI + LangGraph 实现流式与审批门控

```
# backend/main.py
import asyncio
import json
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

app = FastAPI()

# -- Tool definitions ----------------------------------------------------------

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Search results for '{query}': [simulated results]"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email. REQUIRES HUMAN APPROVAL."""
    return f"Email sent to {to} with subject '{subject}'"

@tool
def read_file(path: str) -> str:
    """Read a file from the filesystem."""
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found: {path}"

# Tools requiring approval (Tier 3)
APPROVAL_REQUIRED_TOOLS = {"send_email"}

# -- Approval gate store (in-memory; use Redis in production) ------------------

approval_store: dict[str, asyncio.Event] = {}
approval_results: dict[str, dict] = {}

# -- LLM setup -----------------------------------------------------------------

llm = ChatOpenAI(model="gpt-4o", streaming=True)
tools = [web_search, send_email, read_file]
llm_with_tools = llm.bind_tools(tools)

def should_request_approval(tool_name: str) -> bool:
    return tool_name in APPROVAL_REQUIRED_TOOLS

# -- Streaming endpoint --------------------------------------------------------

async def agent_stream(
    session_id: str,
    user_message: str,
) -> AsyncGenerator[str, None]:
    """Stream agent events as SSE."""

    def sse(event_type: str, data: dict) -> str:
        return f"data: {json.dumps({'type': event_type, **data})}\n\n"

    yield sse("status", {"message": "Agent starting..."})

    # Simulate multi-step agent execution
    steps = [
        ("thinking", {"content": "Analyzing the request..."}),
        ("tool_call", {
            "tool": "web_search",
            "input": {"query": user_message},
            "tier": 1,  # Auto-approve
        }),
        ("tool_result", {
            "tool": "web_search",
            "output": f"Results for: {user_message}",
            "duration_ms": 342,
        }),
    ]

    for event_type, data in steps:
        await asyncio.sleep(0.5)  # Simulate processing time
        yield sse(event_type, data)

    # Simulate a Tier 3 action requiring approval
    approval_id = f"{session_id}_email_001"
    approval_event = asyncio.Event()
    approval_store[approval_id] = approval_event

    yield sse("approval_required", {
        "approval_id": approval_id,
        "tool": "send_email",
        "tier": 3,
        "risk": "irreversible",
        "action_summary": "Send summary email to user@example.com",
        "parameters": {
            "to": "user@example.com",
            "subject": f"Research results: {user_message}",
            "body": "Here are the findings...",
        },
    })

    # Wait for human approval (timeout after 5 minutes)
    try:
        await asyncio.wait_for(approval_event.wait(), timeout=300)
        result = approval_results.get(approval_id, {})

        if result.get("approved"):
            yield sse("tool_call", {
                "tool": "send_email",
                "input": result.get("parameters", {}),
                "tier": 3,
                "approved_by": "human",
            })
            await asyncio.sleep(0.3)
            yield sse("tool_result", {
                "tool": "send_email",
                "output": "Email sent successfully",
                "duration_ms": 128,
            })
        else:
            yield sse("action_rejected", {
                "tool": "send_email",
                "reason": result.get("reason", "User rejected"),
            })
    except asyncio.TimeoutError:
        yield sse("approval_timeout", {
            "approval_id": approval_id,
            "message": "Approval timed out; action skipped",
        })

    # Final response
    yield sse("token", {"content": "I've completed the research. "})
    yield sse("token", {"content": "Here's a summary of what I found..."})
    yield sse("done", {"total_tokens": 847, "duration_ms": 2341})

@app.get("/chat/stream")
async def chat_stream(session_id: str, message: str):
    return StreamingResponse(
        agent_stream(session_id, message),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

class ApprovalRequest(BaseModel):
    approval_id: str
    approved: bool
    parameters: dict | None = None
    reason: str | None = None

@app.post("/chat/approve")
async def handle_approval(req: ApprovalRequest):
    if req.approval_id not in approval_store:
        raise HTTPException(status_code=404, detail="Approval not found")
    approval_results[req.approval_id] = {
        "approved": req.approved,
        "parameters": req.parameters,
        "reason": req.reason,
    }
    approval_store[req.approval_id].set()
    return {"status": "ok"}
```

<span class="hh-tag">代码清单 69：</span>带流式传输与审批门控的 FastAPI 后端

### Frontend：React 实现流式与工具可视化

```
// frontend/AgentChat.tsx
import { useState, useEffect, useRef } from 'react';

// -- Types ---------------------------------------------------------------------

type AgentEvent =
  | { type: 'status'; message: string }
  | { type: 'thinking'; content: string }
  | { type: 'token'; content: string }
  | { type: 'tool_call'; tool: string; input: object; tier: number }
  | { type: 'tool_result'; tool: string; output: string; duration_ms: number }
  | { type: 'approval_required'; approval_id: string; tool: string;
      tier: number; risk: string; action_summary: string; parameters: object }
  | { type: 'action_rejected'; tool: string; reason: string }
  | { type: 'done'; total_tokens: number; duration_ms: number };

// -- Tool Card Component -------------------------------------------------------

function ToolCard({ event }: { event: AgentEvent & { type: 'tool_call' } }) {
  const [expanded, setExpanded] = useState(false);
  const tierColors = { 1: '#22c55e', 2: '#f59e0b', 3: '#ef4444' };
  const color = tierColors[event.tier as keyof typeof tierColors] || '#6b7280';

  return (
    <div style={{ border: `1px solid ${color}`, borderRadius: 8, padding: 8,
                  margin: '4px 0', fontSize: 13 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ color, fontWeight: 600 }}>[gear] {event.tool}</span>
        <span style={{ color: '#6b7280', fontSize: 11 }}>
          Tier {event.tier} | {event.tier === 1 ? 'Auto' : 'Approved'}
        </span>
        <button onClick={() => setExpanded(!expanded)}
                style={{ marginLeft: 'auto', fontSize: 11 }}>
          {expanded ? 'Hide' : 'Details'}
        </button>
      </div>
      {expanded && (
        <pre style={{ marginTop: 8, fontSize: 11, background: '#f3f4f6',
                      padding: 8, borderRadius: 4, overflow: 'auto' }}>
          {JSON.stringify(event.input, null, 2)}
        </pre>
      )}
    </div>
  );
}

// -- Approval Gate Component ---------------------------------------------------

function ApprovalGate({
  event,
  onDecision,
}: {
  event: AgentEvent & { type: 'approval_required' };
  onDecision: (approved: boolean, params?: object) => void;
}) {
  const riskColors = { reversible: '#22c55e', 'hard-to-undo': '#f59e0b',
                       irreversible: '#ef4444' };
  const riskColor = riskColors[event.risk as keyof typeof riskColors] || '#6b7280';

  return (
    <div style={{ border: `2px solid ${riskColor}`, borderRadius: 8,
                  padding: 16, margin: '8px 0', background: '#fef9f0' }}>
      <div style={{ fontWeight: 700, color: riskColor, marginBottom: 8 }}>
        [!] Approval Required: {event.tool}
      </div>
      <div style={{ marginBottom: 8 }}>{event.action_summary}</div>
      <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 12 }}>
        Risk level: <span style={{ color: riskColor }}>{event.risk}</span>
      </div>
      <div style={{ display: 'flex', gap: 8 }}>
        <button
          onClick={() => onDecision(true, event.parameters)}
          style={{ background: '#22c55e', color: 'white', border: 'none',
                   borderRadius: 6, padding: '8px 16px', cursor: 'pointer' }}>
          [ok] Approve
        </button>
        <button
          onClick={() => onDecision(false)}
          style={{ background: '#ef4444', color: 'white', border: 'none',
                   borderRadius: 6, padding: '8px 16px', cursor: 'pointer' }}>
          [x] Reject
        </button>
      </div>
    </div>
  );
}

// -- Main Chat Component -------------------------------------------------------

export function AgentChat() {
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [response, setResponse] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [input, setInput] = useState('');
  const sessionId = useRef(`session_${Date.now()}`);

  const sendMessage = async () => {
    if (!input.trim() || isStreaming) return;
    setEvents([]);
    setResponse('');
    setIsStreaming(true);

    const url = `/chat/stream?session_id=${sessionId.current}`
              + `&message=${encodeURIComponent(input)}`;
    const es = new EventSource(url);

    es.onmessage = (e) => {
      const event: AgentEvent = JSON.parse(e.data);
      if (event.type === 'token') {
        setResponse(prev => prev + event.content);
      } else if (event.type === 'done') {
        setIsStreaming(false);
        es.close();
      } else {
        setEvents(prev => [...prev, event]);
      }
    };

    es.onerror = () => { setIsStreaming(false); es.close(); };
    setInput('');
  };

  const handleApproval = async (
    approvalId: string,
    approved: boolean,
    parameters?: object,
  ) => {
    await fetch('/chat/approve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ approval_id: approvalId, approved, parameters }),
    });
  };

  return (
    <div style={{ maxWidth: 800, margin: '0 auto', padding: 16 }}>
      <div style={{ minHeight: 400, border: '1px solid #e5e7eb',
                    borderRadius: 8, padding: 16, marginBottom: 16 }}>
        {events.map((event, i) => {
          if (event.type === 'tool_call')
            return <ToolCard key={i} event={event} />;
          if (event.type === 'approval_required')
            return (
              <ApprovalGate key={i} event={event}
                onDecision={(approved, params) =>
                  handleApproval(event.approval_id, approved, params)} />
            );
          if (event.type === 'status' || event.type === 'thinking')
            return (
              <div key={i} style={{ color: '#6b7280', fontSize: 12,
                                    fontStyle: 'italic', margin: '4px 0' }}>
                {event.type === 'thinking' ? event.content : event.message}
              </div>
            );
          return null;
        })}
        {response && (
          <div style={{ marginTop: 8, lineHeight: 1.6 }}>
            {response}
            {isStreaming && <span className="cursor-blink">|</span>}
          </div>
        )}
      </div>
      <div style={{ display: 'flex', gap: 8 }}>
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && sendMessage()}
          placeholder="Ask the agent..."
          style={{ flex: 1, padding: '8px 12px', borderRadius: 6,
                   border: '1px solid #d1d5db', fontSize: 14 }}
        />
        <button onClick={sendMessage} disabled={isStreaming}
                style={{ padding: '8px 16px', background: '#3b82f6',
                         color: 'white', border: 'none', borderRadius: 6,
                         cursor: isStreaming ? 'not-allowed' : 'pointer' }}>
          {isStreaming ? 'Running...' : 'Send'}
        </button>
      </div>
    </div>
  );
}
```

<span class="hh-tag">代码清单 70：</span>带流式工具可视化与审批门控的 React 前端

## 小结

Agent 化 UI 框架代表着人机交互的新前沿，要求从第一性原理出发重新思考界面设计。本节的关键洞见如下：

<ol><li>范式选择很重要：合适的 UI 范式（对话、画布、工作流、仪表盘、协作、自主）取决于任务结构、所需的人类参与程度以及输出类型。大多数生产系统会组合多种范式。</li><li>透明性不可妥协：用户无法信任他们看不见的东西。思考展示、工具可视化与上下文面板不是可选功能——它们是可信 Agent 化系统的根基。</li><li>流式传输是基线：用户期望实时看到 Agent 工作。Token 流式传输、工具调用流式传输与多 Agent 流式传输是必备能力。</li><li>审批门控必须分级：扁平化的审批策略（全部批准或全部不批准）在实践中是行不通的。对安全操作自动批准、对危险操作设门控的分级策略，能在不制造瓶颈的情况下维持监督。</li><li>生成式 UI 是前沿：LLM 不仅能生成文本，还能生成 UI 组件——图表、表单、地图、小组件——这使得界面能够适应内容，而不是把内容强行塞进固定模板。</li><li>信任来自一致性与可恢复性：对于建立用户信任而言，撤销能力、审计轨迹与经过校准的置信度指示器与原始能力同样重要。</li></ol>

本节所述的框架与模式——Vercel AI SDK、Chainlit、Gradio、Streamlit、LangGraph Studio——提供了构建模块。对实践者而言，挑战在于以用户的具体需求与所在领域的特定风险为指引，审慎地将它们组合起来。

<aside class="hh-next">
<p class="hh-next-label">下一篇</p>
<p class="hh-next-title">智能体 AI 漫游指南 · 第 28 章 测验题与详细解答</p>
<p class="hh-next-hint">在「智能体 AI 漫游指南」分类下继续阅读。</p>
</aside>

</div>
