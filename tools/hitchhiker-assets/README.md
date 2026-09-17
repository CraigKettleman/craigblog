# 《智能体 AI 漫游指南》正文转换器

把 arXiv HTML 版《The Hitchhiker's Guide to Agentic AI》转换成博客文章
（`content/posts/` 下的 Markdown + 图片），**每章一篇**。

来源：

- 英文原版：<https://arxiv.org/html/2606.24937v2>（CC BY-SA 4.0）
- 中文译本：`hitchhiker-agentic-ai-zh/index.html`（本地已翻译单页）

## 产出

| 文章 | 内容 |
| --- | --- |
| `…-0-preface` | 前言：免责声明、关于作者、前言、引言、缩略语表（5 段合成一篇） |
| `…-01-…` ～ `…-30-…` | 第 1–30 章，每章独立一篇 |

共 31 篇 × 2 语言。末篇（第 30 章）附全书 443 条参考文献；第 28 章为
108 道自测题。发布日期从 2026-09-15 起每篇 +1 天，保证列表与 RSS 的
顺序与书的章节顺序一致。

slug 形如 `hitchhiker-agentic-ai-05-ppo-proximal-policy-optimization`：
中英双语共用同一 slug（中文标题没有 ASCII 字母，会退化成 `chapter-N`，
所以中文运行需要 `--titles` 指向英文单页取标题）。

## 用法

```bash
# 英文（先跑，产出正式 slug 与图片）
node convert.mjs --lang en \
  --html   /path/to/arxiv-2606.24937v2.html \
  --assets ../../hitchhiker-agentic-ai-zh/assets/figures \
  --out    ../../content/posts

# 中文（--titles 只为取英文标题生成一致的 slug）
node convert.mjs --lang zh \
  --html   ../../hitchhiker-agentic-ai-zh/index.html \
  --titles /path/to/arxiv-2606.24937v2.html \
  --assets ../../hitchhiker-agentic-ai-zh/assets/figures \
  --out    ../../content/posts
```

两次运行互不覆盖：每篇只重写自己语言的 `.md`，目录与图片保留。

## 标题图

`generate-covers.mjs` 为每篇生成一张统一的「印刷卡片」封面（1200×800 PNG）：

```bash
node generate-covers.mjs --out content/posts
```

- 纸色底 + 橙色网点 + 章号大字 + 章名 + 该章主题图形（纯几何绘制）
- 31 个主题图形，坐标固定在画布右侧，与左侧标题区互不重叠
- 字体：拉丁用 JetBrains Mono（仓库内已有），中文按篇取 Noto Sans SC 子集
- **resvg 不做逐字形回退**，混排必须按字体拆成多个 `<text>`；
  断行只在空格、CJK↔拉丁交界、全角括号前后发生
- 转换器会写 `cover: "./cover.png"` 到 frontmatter

## 结构

| 文件 | 职责 |
| --- | --- |
| `convert.mjs` | 入口：逐章分组、frontmatter、图片落盘、Markdown 组装 |
| `section-map.mjs` | 系列起始日期、前言章节 id、slug 生成 |
| `lib/html.mjs` | 极简 HTML 解析器（只服务 LaTeXML 输出）与遍历/序列化工具 |
| `lib/labels.mjs` | 章节树抽取，交叉引用/文献编号映射 |
| `lib/serialize.mjs` | DOM → Markdown/HTML 的块级与行内改写 |

## 关键取舍

- **正文逐字保留**：只做结构层面的等价改写，不重写文本。
- **公式保留原生 MathML**：浏览器直接渲染，不做图片化。
- **代码清单取原始源码**：LaTeXML 在清单里附了 `data:` 下载链接，
  解码即为逐字源码，比按行拼装的行内文本更可靠。
- **交叉引用降级为纯文本**：切成 31 篇后跨文章锚点全部失效，
  文献引用统一渲染为 `[N]`，正文引用补全为标题文本。
- **自测题重构为折叠块**：源文档把题干与解答包在一张 SVG 图里，
  这里拆回 `<details>`，读者先自测再展开解析。
- **单章成篇时标题层级上提一级**：章节标题已作文章标题，
  节标题（h3/h4）整体上提为 h2/h3，否则目录会缺父级。
- **标题编号统一走站点徽标**：转换时剔掉源文档标题里的「1.1」「1.2.1」
  等编号，改由站点给 h2/h3 自动编号（`.post-content h2::before`）。
  同时清理两类源数据噪声：粘在标题尾部的 PDF 页眉
  （「…主题H. Roitman — …」）与混入标题正文的图内数字。
- **标题长度裁剪**：velite 限制 title ≤ 99 字符，超长按词边界截断加省略号。
