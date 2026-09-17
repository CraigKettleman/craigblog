/**
 * 文档结构与交叉引用重建。
 *
 * 源文档里 `\ref` 生成 `<a class="ltx_ref" href="#Ch7.S3">…</a>`，锚文本常是
 * LaTeXML 注入的编号。切分成多篇文章后跨文章锚点全部失效，因此建立三类映射：
 *   1. 章节/节 id -> 标题（把「第 7 章」补成「第 7 章 PPO——近端策略优化」）
 *   2. 图表/公式 id -> 编号（Figure 3 -> 图 3）
 *   3. 文献 key -> 序号（bib.bib12 -> [12]）
 */

import {
  children,
  classes,
  findFirst,
  findAll,
  hasClass,
  normText,
  parseHtml,
} from "./html.mjs";

/** 扫描整篇文档，抽出章节树与分部。 */
export function parseDocument(html) {
  const root = parseHtml(html);
  const article = findFirst(root, (n) => n.name === "article") ?? root;

  const parts = [];
  const chapters = [];

  // 源文档里章节嵌在分部（section.ltx_part）内部，需要递归扫描。
  const visit = (node, currentPart) => {
    for (const child of children(node)) {
      const cls = classes(child);
      if (cls.includes("ltx_part")) {
        const titleEl = findFirst(child, (n) => hasClass(n, "ltx_title"));
        const id = child.attrs.id ?? "";
        const part = {
          id,
          number: romanToNumber(id.replace(/^Pt/, "")),
          title: titleEl ? stripTag(titleEl).title : normText(child),
          label: titleEl ? stripTag(titleEl).label : "",
          element: child,
          chapters: [],
        };
        parts.push(part);
        visit(child, part);
        continue;
      }
      if (cls.includes("ltx_chapter")) {
        const titleEl = findFirst(child, (n) => hasClass(n, "ltx_title"));
        const id = child.attrs.id ?? "";
        const heading = titleEl ? stripTag(titleEl) : { label: "", title: "" };
        const chapter = {
          id,
          element: child,
          label: heading.label,
          title: heading.title || normText(child),
          number: chapterNumber(id),
          part: currentPart,
          sections: [],
        };
        currentPart?.chapters.push(chapter);
        chapters.push(chapter);
        continue;
      }
    }
  };

  visit(article, null);

  for (const chapter of chapters) {
    for (const node of findAll(
      chapter.element,
      (n) => classes(n).includes("ltx_section") || classes(n).includes("ltx_subsection"),
    )) {
      const titleEl = findFirst(node, (n) => hasClass(n, "ltx_title"));
      const heading = titleEl ? stripTag(titleEl) : { label: "", title: "" };
      chapter.sections.push({
        id: node.attrs.id,
        label: heading.label,
        title: heading.title,
        element: node,
      });
    }
  }

  return { root, article, chapters, parts };
}

/** 去掉标题里的编号标签，返回 { label, title }。 */
function stripTag(titleEl) {
  const tagEl = findFirst(titleEl, (n) => hasClass(n, "ltx_tag"));
  const label = tagEl ? normText(tagEl) : "";
  let full = normText(titleEl);
  if (label && full.startsWith(label)) full = full.slice(label.length).trim();
  return { label, title: full };
}

function chapterNumber(id) {
  const m = /^Ch(\d+)$/.exec(id);
  return m ? Number(m[1]) : null;
}

function romanToNumber(roman) {
  const table = { I: 1, V: 5, X: 10, L: 50, C: 100 };
  let total = 0;
  const s = roman.toUpperCase();
  for (let i = 0; i < s.length; i++) {
    const cur = table[s[i]] ?? 0;
    const next = table[s[i + 1]] ?? 0;
    total += cur < next ? -cur : cur;
  }
  return total || null;
}

/** 章节/节 id -> 可读标题。 */
export function buildHeadingMap(doc) {
  const map = new Map();
  for (const chapter of doc.chapters) {
    map.set(chapter.id, join(chapter.label, chapter.title));
    for (const section of chapter.sections) {
      if (section.title) map.set(section.id, join(section.label, section.title));
    }
  }
  return map;
}

/** 图表/表格/公式 id -> 编号文本（如「图 3」）。 */
export function buildLabelMap(doc) {
  const map = new Map();
  const kinds = [
    ["ltx_tag_figure", "图"],
    ["ltx_tag_table", "表"],
    ["ltx_tag_equation", "式"],
  ];
  // 只取节点自身的标签，避免容器（章/节）继承子元素的编号。
  for (const node of findAll(doc.article, (n) => Boolean(n.attrs.id))) {
    const tag = directTag(node);
    if (!tag) continue;
    for (const [cls, prefix] of kinds) {
      if (!hasClass(tag, cls)) continue;
      const m = /([\w.]+)\s*[:：]?$/.exec(normText(tag));
      if (m) map.set(node.attrs.id, `${prefix} ${m[1]}`);
    }
  }
  return map;
}

/** 广度上找最近的编号标签：直接子节点，或 caption/标题容器内的第一层标签。 */
function directTag(node) {
  const isTag = (n) => classes(n).some((c) => c.startsWith("ltx_tag_"));
  for (const child of children(node)) {
    if (isTag(child)) return child;
  }
  for (const child of children(node)) {
    const cls = classes(child);
    if (!cls.includes("ltx_caption") && !cls.includes("ltx_title")) continue;
    for (const inner of children(child)) if (isTag(inner)) return inner;
  }
  return null;
}

/** 文献 key -> [序号]。 */
export function buildBibMap(doc) {
  const map = new Map();
  for (const node of findAll(doc.article, (n) =>
    classes(n).includes("ltx_bibitem"),
  )) {
    const tag = findFirst(node, (n) => hasClass(n, "ltx_tag_bibitem"));
    const m = tag ? /\[([^\]]+)\]/.exec(normText(tag)) : null;
    if (node.attrs.id && m) map.set(node.attrs.id, `[${m[1]}]`);
  }
  return map;
}

function join(label, title) {
  if (!label) return title;
  return title ? `${label} ${title}` : label;
}
