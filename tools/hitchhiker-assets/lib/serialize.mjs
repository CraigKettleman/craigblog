/**
 * LaTeXML DOM -> Markdown / 站内 HTML 的序列化。
 *
 * 原则：
 * - 正文段落化为 Markdown 段落，内部行内标记转成等价 HTML（数学、上下标、
 *   代码、强调），因为站点正文本身以 HTML 注入，行内 HTML 比 Markdown 更可控。
 * - 独立公式、图、表、代码清单保持块级 HTML（MathML / <table> / <pre>），
 *   由站点 CSS 统一排版。
 * - 所有跨文章的 <a href="#..."> 引用降级为纯文本，避免死链。
 */

import {
  children,
  classes,
  clone,
  decodeEntities,
  escapeAttr,
  escapeText,
  findAll,
  findFirst,
  hasAnyClass,
  hasClass,
  isElement,
  normText,
  serializeHtml,
  textOf,
} from "./html.mjs";

const SKIP_CLASSES = new Set([
  "ltx_page_navbar",
  "ltx_page_footer",
  "ltx_page_logo",
  "ltx_role_affiliationmark",
  "ltx_note_outer", // 脚注单独处理
]);

/**
 * 序列化一个章节的块级子节点。
 * @param {object} element
 * @param {object} ctx
 * @param {{quiz?:boolean}} [opts]
 */
export function serializeBlocks(element, ctx, opts = {}) {
  const blocks = [];
  walkBlocks(element, ctx, blocks, opts);
  return blocks;
}

const CONTAINER_CLASSES = [
  "ltx_section",
  "ltx_subsection",
  "ltx_subsubsection",
  "ltx_paragraph",
  "ltx_para",
  "ltx_chapter",
  "ltx_abstract",
  "ltx_acknowledgement",
];

function walkBlocks(node, ctx, out, opts = {}) {
  for (const child of node.children) {
    if (!isElement(child)) continue;
    emitBlock(child, ctx, out, 0, opts);
  }
}

function emitBlock(node, ctx, out, depth = 0, opts = {}) {
  const cls = classes(node);

  if (SKIP_CLASSES.has(node.name) || cls.some((c) => SKIP_CLASSES.has(c))) return;
  if (cls.includes("ltx_title")) return; // 标题单独处理
  if (cls.includes("ltx_pagination") || cls.includes("ltx_role_newpage")) return;
  if (cls.includes("ltx_note")) return; // 脚注已并入正文流

  // 测验题：题干 + 折叠解答。
  // 只有叶子段落才是单道题（节容器也含 minipage，不能当作一道题）。
  if (opts.quiz && isQuizLeaf(node)) {
    const quiz = serializeQuizItem(node, ctx);
    if (quiz) {
      out.push(quiz);
      return;
    }
  }

  // 段落容器里直接挂了块级代码清单（div.ltx_para > div.ltx_listing）：
  // 若当作容器直接下发，清单会被行内化压成一整段，这里先把它拆出来。
  if (cls.includes("ltx_para")) {
    const listings = children(node).filter((n) => classes(n).includes("ltx_listing"));
    if (listings.length > 0) {
      for (const child of node.children) {
        if (!isElement(child)) continue;
        if (classes(child).includes("ltx_listing")) emitBlock(child, ctx, out, depth, opts);
      }
      return;
    }
  }

  // 结构性容器：下发一层
  if (CONTAINER_CLASSES.some((c) => cls.includes(c))) {
    emitSectionHeading(node, ctx, out, opts);
    walkBlocks(node, ctx, out, opts);
    return;
  }

  if (cls.includes("ltx_p")) {
    const html = serializeInline(node, ctx).trim();
    if (html) out.push({ kind: "paragraph", html });
    return;
  }

  if (node.name === "dl" && cls.includes("ltx_description")) {
    // 缩略语表的词条都是短缩写；「1.3 版新增内容」这类也用 dl 排版但词条是整句，
    // 按词条长度把它们分开，后者降级为普通列表。
    if (isGlossaryList(node)) {
      out.push({ kind: "html", html: serializeDescriptionList(node, ctx) });
    } else {
      out.push({ kind: "html", html: serializeDescriptionListAsList(node, ctx) });
    }
    return;
  }

  if (cls.includes("ltx_itemize") || cls.includes("ltx_enumerate")) {
    out.push({ kind: "list", html: serializeList(node, ctx) });
    return;
  }

  if (cls.includes("ltx_quote")) {
    const inner = serializeInline(node, ctx).trim();
    if (inner) out.push({ kind: "quote", html: `<blockquote>${inner}</blockquote>` });
    return;
  }

  if (cls.includes("ltx_listing") || cls.includes("ltx_lstlisting")) {
    const listing = serializeListing(node, ctx);
    if (listing) out.push(listing);
    return;
  }

  if (node.name === "figure") {
    // 浮动清单（figure.ltx_float.ltx_lstlisting）：内容在子 div.ltx_listing 里
    if (cls.includes("ltx_lstlisting")) {
      const listing = findFirst(node, (n) => classes(n).includes("ltx_listing"));
      if (listing) emitBlock(listing, ctx, out, depth, opts);
      return;
    }
    if (cls.includes("ltx_table")) out.push({ kind: "table", html: serializeTable(node, ctx) });
    else out.push({ kind: "figure", html: serializeFigure(node, ctx) });
    return;
  }

  if (node.name === "table" && cls.includes("ltx_equation")) {
    out.push({ kind: "equation", html: serializeEquation(node, ctx) });
    return;
  }

  if (node.name === "table" && cls.includes("ltx_eqn_table")) {
    out.push({ kind: "equation", html: serializeEquation(node, ctx) });
    return;
  }

  if (cls.includes("ltx_bibliography")) return; // 参考文献由 renderReferences 处理

  if (node.name === "section" || node.name === "div") {
    walkBlocks(node, ctx, out, opts);
    return;
  }

  // 兜底：当作行内段落处理
  const html = serializeInline(node, ctx).trim();
  if (html) out.push({ kind: "paragraph", html });
}

/** 节标题（h4/h5）转成文章内的 h3/h4。 */
/**
 * 节标题（ltx_section / ltx_subsection）转成文章内标题。
 * 单章成篇时章节标题已经做了文章标题，节标题整体上提一级，
 * 否则正文会从 h3 开始、在目录里缺父级（headingOffset = -1）。
 */
function emitSectionHeading(node, ctx, out, opts = {}) {
  const cls = classes(node);
  const base = cls.includes("ltx_section") ? 3 : cls.includes("ltx_subsection") ? 4 : null;
  if (base === null) return;
  const level = Math.min(6, Math.max(2, base + (opts.headingOffset ?? 0)));
  const titleEl = findFirst(node, (n) => hasClass(n, "ltx_title"));
  if (!titleEl) return;
  const title = headingTitle(titleEl);
  if (!title) return;
  out.push({ kind: "heading", level, text: title, id: (node.attrs.id ?? "").toLowerCase() });
}

/**
 * 取章节小标题的正文。
 * 源文档把「1.1」这类编号放在独立的 <span class="ltx_tag"> 里，
 * 站点的 h2/h3 徽标会自己编号，因此这里剔掉源编号避免重复。
 * 另外部分标题的文本节点尾部粘着页眉（「…标题H. Roitman — 智能体 AI
 * 漫游指南：从基础到系统」），一并截掉。
 */
function headingTitle(titleEl) {
  const hasTag = findFirst(titleEl, (n) => hasClass(n, "ltx_tag")) !== null;
  const parts = titleEl.children.filter(
    (child) => !(isElement(child) && classes(child).includes("ltx_tag")),
  );
  let text = parts
    .map((child) => (isElement(child) ? normText(child) : child.text))
    .join(" ")
    .replace(/\s+/g, " ")
    .trim();
  text = stripRunningHeader(text);
  // 源编号已由 ltx_tag 提供时，标题文本开头若还粘着数字（如「11.1」小节
  // 的文本是「4 模型显存挑战」，那个 4 来自图内文字），这个数字是噪声，剔掉。
  // 没有 ltx_tag 的标题（「2026 年的全景」「1.3 版的新内容」）则原样保留。
  if (hasTag) text = text.replace(/^\d+(?:\.\d+)*\s+(?=\S)/, "").trim();
  if (text) return text;
  // 少数标题结构异常时回退到「去掉开头的数字编号」
  return stripRunningHeader(normText(titleEl))
    .replace(/^\d+(?:\.\d+)*\s*/, "")
    .trim();
}

/** 去掉标题尾部误入的页眉（如「…主题H. Roitman — …」）。 */
function stripRunningHeader(text) {
  const idx = text.indexOf("H. Roitman");
  return (idx > 0 ? text.slice(0, idx) : text).trim();
}

// ---------------------------------------------------------------------------
// 行内序列化
// ---------------------------------------------------------------------------

/** 把节点内容序列化为行内 HTML。 */
export function serializeInline(node, ctx) {
  return node.children
    .map((child) => inlineToString(child, ctx))
    .join("")
    .replace(/[ \t]*\n[ \t]*/g, " ")
    .replace(/ {2,}/g, " ");
}

function inlineToString(node, ctx) {
  if (!isElement(node)) return escapeText(node.text);
  const cls = classes(node);

  if (SKIP_CLASSES.has(node.name) || cls.some((c) => SKIP_CLASSES.has(c))) {
    if (cls.includes("ltx_note_outer")) return serializeFootnote(node, ctx);
    return "";
  }
  if (node.name === "annotation") return ""; // MathML 的 TeX 注解不输出

  if (node.name === "math") return serializeMath(node);
  if (node.name === "svg") return ""; // 题图 SVG 缩略图不搬运
  if (node.name === "img") return "";

  if (node.name === "a") return serializeAnchor(node, ctx);

  // 文献引用：源文档里是 <cite>[<a href="#bib.bibN">274</a>]</cite>。
  // 参考文献集中在最后一篇，编号就是序号，降级为纯文本避免死链。
  if (node.name === "cite") {
    const num = normText(node).replace(/^\[|\]$/g, "").trim();
    return num ? `[${escapeText(num)}]` : "";
  }

  if (node.name === "em" || node.name === "i") {
    return `<em>${serializeInline(node, ctx)}</em>`;
  }
  if (node.name === "strong" || node.name === "b") {
    return `<strong>${serializeInline(node, ctx)}</strong>`;
  }

  if (node.name === "sup") return `<sup>${serializeInline(node, ctx)}</sup>`;
  if (node.name === "sub") return `<sub>${serializeInline(node, ctx)}</sub>`;

  if (cls.includes("ltx_font_typewriter") || node.name === "code" || node.name === "tt") {
    return `<code>${serializeInline(node, ctx)}</code>`;
  }

  if (node.name === "span" && cls.includes("ltx_note")) {
    // 行内脚注标记：输出为上标编号，内容由 serializeFootnote 附加
    return "";
  }

  if (node.name === "span" && cls.includes("ltx_note_mark")) return "";

  // 编号标签（如「图 1：」「表 2.1：」）保持原文
  if (cls.some((c) => c.startsWith("ltx_tag_"))) {
    const text = normText(node);
    return text ? `<span class="hh-tag">${escapeText(text)}</span>` : "";
  }

  if (node.name === "br") return "<br>";

  return serializeInline(node, ctx);
}

/**
 * MathML 原样输出。
 * 必须去掉 <annotation>（TeX 源码）：velite 的 markdown 管线在遇到
 * `\mathbf{h}_{i}...` 这类内容的注解时会误判并截断，导致公式后面
 * 漏出 `{i}-\bm{\mu})` 这样的源码（保留 alttext 已足够承载无障碍信息）。
 */
function serializeMath(node) {
  const copy = clone(node);
  stripStyles(copy);
  dropAnnotations(copy);
  return serializeHtml(copy);
}

/** 递归移除 MathML 的 TeX 注解节点。 */
function dropAnnotations(node) {
  if (!isElement(node)) return;
  node.children = node.children.filter(
    (child) => !(isElement(child) && child.name === "annotation"),
  );
  for (const child of node.children) dropAnnotations(child);
}

function stripStyles(node) {
  if (!isElement(node)) return;
  delete node.attrs.style;
  delete node.attrs.mathcolor;
  delete node.attrs.class;
  delete node.attrs.id;
  delete node.attrs.intent;
  for (const child of node.children) stripStyles(child);
}

/** 链接：站内引用降级为纯文本，外链与 data: 保留/丢弃。 */
function serializeAnchor(node, ctx) {
  const href = node.attrs.href ?? "";
  const cls = classes(node);

  if (href.startsWith("data:")) return "";

  // 外部链接（包括参考文献里的 arxiv/DOI 链接）保留为可点链接。
  // 这些 <a> 也带 ltx_ref class，必须先于下面的引用降级分支处理。
  if (/^https?:/i.test(href) || href.startsWith("mailto:")) {
    const label = serializeInline(node, ctx) || href;
    return `<a href="${escapeAttr(href)}" target="_blank" rel="noopener noreferrer">${label}</a>`;
  }

  // 站内引用（#...）：切分成多篇后锚点会跨文章失效，降级为标题文本。
  if (cls.includes("ltx_ref") || href.startsWith("#")) {
    if (href.startsWith("#")) {
      const target = href.slice(1);
      const label = ctx.labelMap.get(target);
      const heading = ctx.headingMap.get(target);
      if (label) return escapeText(label);
      if (heading) return escapeText(heading);
    }
    return escapeText(normText(node));
  }

  const text = serializeInline(node, ctx);
  if (!href || !text.trim()) return text;
  return `<a href="${escapeAttr(href)}" target="_blank" rel="noopener noreferrer">${text}</a>`;
}

/** 脚注：以括号形式内联，避免依赖悬浮层。 */
function serializeFootnote(node, ctx) {
  const content = findFirst(node, (n) => hasClass(n, "ltx_note_content"));
  if (!content) return "";
  const inner = serializeInline(content, ctx)
    .replace(/<[^>]+>/g, (m) => (m.startsWith("<code") || m.startsWith("</code") ? m : ""))
    .trim();
  return inner ? `（${inner}）` : "";
}

// ---------------------------------------------------------------------------
// 块级：列表 / 代码 / 图 / 表 / 公式
// ---------------------------------------------------------------------------

/**
 * 词条是否像缩略语：短、无句尾标点，且释义明显长于词条。
 * 「1.3 版新增内容」的 dl 同样用 dt/dd，但词条是「第 N 章——主题：」这类短语。
 */
function isGlossaryList(node) {
  const pairs = [];
  let term = "";
  for (const child of children(node)) {
    if (child.name === "dt") term = normText(child);
    else if (child.name === "dd") pairs.push([term, normText(child)]);
  }
  if (pairs.length === 0) return false;
  const looksLikeTerm = pairs.filter(([t]) => t.length <= 16 && !/[:：]$/.test(t));
  return looksLikeTerm.length / pairs.length >= 0.8;
}

/** 非缩略语的定义列表：转成「**词条**：释义」的紧凑列表。 */
function serializeDescriptionListAsList(node, ctx) {
  const items = [];
  let term = "";
  for (const child of children(node)) {
    const cls = classes(child);
    if (child.name === "dt" && cls.includes("ltx_item")) {
      term = serializeInline(child, ctx).replace(/<[^>]+>/g, "").trim();
      continue;
    }
    if (child.name !== "dd") continue;
    const def = serializeInline(child, ctx).trim();
    if (!term && !def) continue;
    items.push(`<li><strong>${escapeText(term)}</strong> ${def}</li>`);
    term = "";
  }
  return `<ul>${items.join("")}</ul>`;
}

/**
 * 定义列表（缩略语表）：dt 是词条，dd 是释义。
 * 源文档的 dl 是两栏排版，这里转成词条 + 释义的紧凑块。
 */
function serializeDescriptionList(node, ctx) {
  const out = [];
  let term = "";
  for (const child of children(node)) {
    const cls = classes(child);
    if (child.name === "dt" && cls.includes("ltx_item")) {
      term = serializeInline(child, ctx).replace(/<[^>]+>/g, "").trim();
      continue;
    }
    if (child.name !== "dd") continue;
    const def = serializeInline(child, ctx).trim();
    if (!term && !def) continue;
    out.push(
      `<div class="hh-glossary-item"><dt>${escapeText(term)}</dt><dd>${def}</dd></div>`,
    );
    term = "";
  }
  return `<dl class="hh-glossary">${out.join("")}</dl>`;
}

function serializeList(node, ctx) {
  const ordered = classes(node).includes("ltx_enumerate");
  const items = children(node).filter((c) => classes(c).includes("ltx_item"));
  const rendered = items.map((item) => {
    const body = item.children
      .map((c) => {
        if (!isElement(c)) return escapeText(c.text);
        if (classes(c).includes("ltx_tag")) return "";
        if (classes(c).includes("ltx_para")) return serializeInline(c, ctx);
        return inlineToString(c, ctx);
      })
      .join("")
      .trim();
    return `<li>${body}</li>`;
  });
  return `<${ordered ? "ol" : "ul"}>${rendered.join("")}</${ordered ? "ol" : "ul"}>`;
}

/** 代码清单：优先取 LaTeXML 附带的原始源码（data: 下载链接）。 */
function serializeListing(node, ctx) {
  const langClass = classes(node).find((c) => c.startsWith("ltx_lst_language_"));
  const lang = langClass ? langClass.replace("ltx_lst_language_", "").toLowerCase() : "";

  let code = "";
  const dataLink = findFirst(node, (n) => n.name === "a" && (n.attrs.href ?? "").startsWith("data:"));
  if (dataLink) {
    const raw = dataLink.attrs.href;
    const comma = raw.indexOf(",");
    const meta = raw.slice(5, comma);
    const payload = raw.slice(comma + 1);
    try {
      code = meta.includes("base64")
        ? Buffer.from(payload, "base64").toString("utf8")
        : decodeURIComponent(payload);
    } catch {
      code = "";
    }
  }

  if (!code) {
    code = children(node)
      .filter((c) => classes(c).includes("ltx_listingline"))
      .map((line) => textOf(line).replace(/\u00a0/g, " "))
      .join("\n");
  }

  code = code.replace(/\s+$/, "");
  if (!code.trim()) return null;

  const captionEl = findFirst(node, (n) => hasClass(n, "ltx_caption"));
  const caption = captionEl ? serializeInline(captionEl, ctx).trim() : "";
  return { kind: "code", lang, code, caption };
}

function serializeFigure(node, ctx) {
  const img = findFirst(node, (n) => n.name === "img");
  const captionEl = findFirst(node, (n) => hasClass(n, "ltx_caption"));
  const caption = captionEl ? serializeInline(captionEl, ctx).trim() : "";
  const id = node.attrs.id ?? "";

  // 少数图是纯 SVG 绘制（无 <img>）：保留图注，不输出空图。
  if (!img && !caption) return "";
  let imgHtml = "";
  if (img) {
    const src = ctx.registerAsset(img.attrs.src ?? "");
    // alt 由图注承担，避免读屏重复朗读整段说明
    const alt = normText(captionEl).replace(/<[^>]+>/g, "").trim();
    imgHtml = `<img src="${escapeAttr(src)}" alt="${escapeAttr(alt.slice(0, 120))}" loading="lazy" decoding="async">`;
  }

  return [
    `<figure${id ? ` id="${escapeAttr(id.toLowerCase())}"` : ""}>`,
    imgHtml,
    caption ? `<figcaption>${caption}</figcaption>` : "",
    "</figure>",
  ].join("");
}

/** 表格：去掉 LaTeXML 的边框 class 内联样式，保留结构。 */
function serializeTable(node, ctx) {
  const captionEl = findFirst(node, (n) => hasClass(n, "ltx_caption"));
  const caption = captionEl ? serializeInline(captionEl, ctx).trim() : "";
  const table = findFirst(node, (n) => n.name === "table");
  const id = node.attrs.id ?? "";

  if (!table) return "";

  const clean = clone(table);
  cleanupTable(clean);

  return [
    `<div class="hh-table"${id ? ` id="${escapeAttr(id.toLowerCase())}"` : ""}>`,
    serializeHtml(clean),
    caption ? `<p class="hh-table-caption">${caption}</p>` : "",
    "</div>",
  ].join("");
}

function cleanupTable(node) {
  if (!isElement(node)) return;
  delete node.attrs.id;
  delete node.attrs.style;
  delete node.attrs.mathcolor;
  delete node.attrs.intent;
  node.attrs.class = classes(node)
    .filter((c) => c === "ltx_tabular" || c.startsWith("ltx_align_"))
    .join(" ");
  if (!node.attrs.class) delete node.attrs.class;

  // 表格单元格里会有文献/公式锚点：这些引用降到其他文章或已重排，
  // 保留 href 会留下死链，这里去掉链接外壳只留文本。
  if (node.name === "a") {
    const href = node.attrs.href ?? "";
    if (href.startsWith("#") || href.startsWith("data:")) {
      node.name = "span";
      node.attrs = {};
    }
  } else if (node.name === "cite") {
    const text = textOf(node).replace(/^\[|\]$/g, "").trim();
    node.name = "span";
    node.attrs = {};
    node.children = [{ type: "text", text: text ? `[${text}]` : "", parent: node }];
  } else if (node.name === "annotation") {
    // velite 的 markdown 管线会误判含 `{...}` 的 TeX 注解并截断，
    // 表格里的公式同样需要去掉注解（alttext 已保留源码）。
    node.name = "span";
    node.attrs = {};
    node.children = [];
  }

  for (const child of node.children) cleanupTable(child);
}

/** 独立公式：保留 MathML 与编号。 */
function serializeEquation(node, ctx) {
  const math = findFirst(node, (n) => n.name === "math" && n.attrs.display === "block");
  const tag = findFirst(node, (n) => hasClass(n, "ltx_tag_equation"));
  const number = tag ? normText(tag) : "";
  const id = node.attrs.id ?? "";
  if (!math) return "";
  const body = serializeMath(math);
  return [
    `<div class="hh-equation"${id ? ` id="${escapeAttr(id.toLowerCase())}"` : ""}>`,
    body,
    number ? `<span class="hh-equation-number">${escapeText(number)}</span>` : "",
    "</div>",
  ].join("");
}

// ---------------------------------------------------------------------------
// 测验题
// ---------------------------------------------------------------------------

/**
 * 单道测验题在源文档里是一个叶子块：恰好两个 minipage（题干 + 解答）。
 * 节容器（ltx_section）也含有这些 minipage，但它的父链上不会再套 ltx_para，
 * 而题目段落本身就是最内层的 ltx_para，因此用父链排除容器。
 */
function isQuizLeaf(node) {
  const cls = classes(node);
  if (!cls.includes("ltx_para") && !cls.includes("ltx_p")) return false;
  if (ancestorHasClass(node, "ltx_para")) return false;
  return findAll(node, (n) => classes(n).includes("ltx_minipage")).length === 2;
}

function ancestorHasClass(node, name) {
  let parent = node.parent;
  while (parent) {
    if (isElement(parent) && classes(parent).includes(name)) return true;
    parent = parent.parent;
  }
  return false;
}

/**
 * 测验题在源文档里被 LaTeXML 包成一张 SVG 图（foreignObject 内嵌题干与解答）。
 * 这里把它拆回「题干 + 解答」两块：题干作为标题，解答折叠进 <details>，
 * 方便读者先自测再看解析。
 */
function serializeQuizItem(node, ctx) {
  const minipages = findAll(node, (n) => classes(n).includes("ltx_minipage"));
  if (minipages.length < 2) return null;

  const [questionMp, answerMp] = minipages;
  const question = quizBlocks(questionMp, ctx);
  const answer = quizBlocks(answerMp, ctx, { dropAnswerLead: true });
  if (!question.length || !answer.length) return null;

  return {
    kind: "quiz",
    id: quizId(question.join(""), node),
    question: question.join(""),
    answer: answer.join(""),
  };
}

/** 把 minipage 内的块级 span（ltx_p / ltx_itemize / ltx_equation）序列化为 HTML 片段。 */
function quizBlocks(minipage, ctx, opts = {}) {
  const out = [];
  const body = findFirst(minipage, (n) => classes(n).includes("ltx_p")) ?? minipage;
  const container = body.parent ?? minipage;

  for (const child of children(container)) {
    const cls = classes(child);
    if (cls.includes("ltx_itemize") || cls.includes("ltx_enumerate")) {
      out.push(serializeList(child, ctx));
      continue;
    }
    if (cls.includes("ltx_equation")) {
      const html = serializeSpanEquation(child, ctx);
      if (html) out.push(html);
      continue;
    }
    if (cls.includes("ltx_p")) {
      let html = serializeInline(child, ctx).trim();
      // 解答首段只有一个引导词（Answer: / 答：），站点已另起「解答」标签，去掉重复
      if (opts.dropAnswerLead) {
        html = html.replace(/^(?:<[^>]+>)*(?:Answer|答)[:：](?:<\/[^>]+>)*\s*/, "");
      }
      if (html) out.push(`<p>${html}</p>`);
      continue;
    }
  }

  // minipage 本身只有一个 ltx_p（题干）时，直接取它
  if (!out.length) {
    const html = serializeInline(minipage, ctx).trim();
    if (html) out.push(`<p>${html}</p>`);
  }
  return out;
}

/** 测验区里的独立公式是 <span class="ltx_equation">，结构同表格版。 */
function serializeSpanEquation(node, ctx) {
  const math = findFirst(node, (n) => n.name === "math" && n.attrs.display === "block");
  const tag = findFirst(node, (n) => hasClass(n, "ltx_tag_equation"));
  const number = tag ? normText(tag) : "";
  if (!math) return "";
  return (
    `<div class="hh-equation">${serializeMath(math)}` +
    (number ? `<span class="hh-equation-number">${escapeText(number)}</span>` : "") +
    `</div>`
  );
}

/** 题号：取题干文本的「Q…」前缀，回退到元素 id。 */
function quizId(questionHtml, node) {
  const plain = questionHtml.replace(/<[^>]+>/g, "").trim();
  const m = /^(Q[\dA-Za-z]+)\s*[:：]/.exec(plain);
  if (m) return m[1].toLowerCase();
  return (node.attrs.id ?? "quiz").toLowerCase().replace(/[.\s]+/g, "-");
}

// ---------------------------------------------------------------------------
// 参考文献
// ---------------------------------------------------------------------------

/** 参考文献按 LaTeXML 的结构重排为紧凑列表。 */
export function serializeReferences(bibliography, ctx) {
  const items = findAll(bibliography, (n) => classes(n).includes("ltx_bibitem"));
  const out = [];
  for (const item of items) {
    const tag = findFirst(item, (n) => hasClass(n, "ltx_tag_bibitem"));
    const number = tag ? normText(tag) : "";
    const blocks = children(item).filter((c) => classes(c).includes("ltx_bibblock"));
    const parts = [];
    for (const block of blocks) {
      const cls = classes(block);
      if (cls.includes("ltx_bib_cited")) continue;
      const text = serializeInline(block, ctx).replace(/\s+/g, " ").trim();
      if (!text) continue;
      parts.push(text);
    }
    if (!parts.length) continue;
    out.push(
      `<li id="${escapeAttr((item.attrs.id ?? "").toLowerCase())}">` +
        `<span class="hh-bib-number">${escapeText(number)}</span>` +
        `<span class="hh-bib-body">${parts.join(" ")}</span>` +
        `</li>`,
    );
  }
  return `<ol class="hh-bibliography">${out.join("")}</ol>`;
}
