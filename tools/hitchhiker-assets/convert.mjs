/**
 * 《智能体 AI 漫游指南》正文转换器。
 *
 * 输入：arXiv HTML 版单页（英文原版 / 中文译本），LaTeXML 生成，
 * 结构与章节 id 一一对应（Ch1..Ch30、Chx1..Chx5、Pt1..Pt6）。
 * 输出：每章一篇博客 Markdown（含 frontmatter），图片落盘到对应文章目录。
 * 前置内容（免责声明/作者/前言/引言/缩略语表）合成首篇。
 *
 * 用法：
 *   node convert.mjs --lang zh \
 *     --html  <中文单页.html> --assets <中文 assets 目录> \
 *     --out   content/posts
 */

import { copyFile, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { basename, join } from "node:path";

import {
  classes,
  findAll,
  findFirst,
  hasClass,
  normReadableText,
  normText,
} from "./lib/html.mjs";
import {
  serializeBlocks,
  serializeInline,
  serializeReferences,
} from "./lib/serialize.mjs";
import {
  buildBibMap,
  buildHeadingMap,
  buildLabelMap,
  parseDocument,
} from "./lib/labels.mjs";
import {
  dateForIndex,
  PRELIM_IDS,
  PRELIM_SLUG,
  slugifyChapter,
} from "./section-map.mjs";

const KEYWORDS_ZH = [
  "智能体",
  "Agentic AI",
  "LLM",
  "大语言模型",
  "强化学习",
  "RLHF",
  "MCP",
];

const KEYWORDS_EN = [
  "agentic AI",
  "LLM",
  "reinforcement learning",
  "RLHF",
  "MCP",
  "A2A",
  "agents",
];

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

const lang = arg("lang", "zh");
const sourceHtml = arg("html", "");
const outRoot = arg("out", "content/posts");
const assetsDir = arg("assets", "");
/**
 * 可选的英文版单页：仅用于取章节标题生成 slug。
 * 中文标题没有 ASCII 字母，slug 会退化成 chapter-N；有英文标题时
 * 中英双语共用同一 slug，保证 hreflang 与语言切换能正确配对。
 */
const titlesHtml = arg("titles", "");

if (!sourceHtml || !assetsDir) {
  console.error(
    "usage: node convert.mjs --lang zh --html <path> --assets <dir> [--out content/posts]",
  );
  process.exit(1);
}

const doc = parseDocument(await readFile(sourceHtml, "utf8"));

/** 章号 -> 英文标题，用于生成跨语言一致的 slug。 */
const slugTitles = new Map();
if (titlesHtml) {
  const titlesDoc = parseDocument(await readFile(titlesHtml, "utf8"));
  for (const c of titlesDoc.chapters) {
    if (c.number) slugTitles.set(c.number, c.title);
  }
}
const bibMap = buildBibMap(doc);
const headingMap = buildHeadingMap(doc);
const labelMap = buildLabelMap(doc);

console.log(
  `[${lang}] 章节 ${doc.chapters.length} / 分部 ${doc.parts.length} / 文献 ${bibMap.size} / 标题 ${headingMap.size} / 标签 ${labelMap.size}`,
);

// ---------------------------------------------------------------------------
// 分组：每章一篇；前置内容合成首篇
// ---------------------------------------------------------------------------

/**
 * @typedef {{ slug:string, date:string, chapters:object[], title:{en:string,zh:string},
 *             description:{en:string,zh:string}, part:object|null, isPrelim:boolean,
 *             number:number|null }} Group
 */

/** @type {Group[]} */
const groups = [];

/** 取章节标题（已剥离编号标签）。 */
function chapterTitle(chapter) {
  return chapter.title.trim();
}

/** 取章节首段作为简介；超长则截断到句末。 */
function chapterIntro(chapter, limit = 140) {
  const kids = chapter.element.children.filter((c) => c.type === "element");
  for (const child of kids) {
    if (classes(child).includes("ltx_title")) continue;
    const p =
      (classes(child).includes("ltx_p") ? child : null) ??
      findFirst(child, (n) => classes(n).includes("ltx_p"));
    if (!p) continue;
    const text = normReadableText(p);
    if (!text) continue;
    if (text.length <= limit) return text;
    const cut = text.slice(0, limit);
    const stop = Math.max(cut.lastIndexOf("\u3002"), cut.lastIndexOf(". "));
    return `${(stop > 40 ? cut.slice(0, stop + 1) : cut).trim()}\u2026`;
  }
  return "";
}

/** 章号标签：「第 5 章」/「Chapter 5」。 */
function shortLabel(chapter, targetLang) {
  if (!chapter.number) return "";
  return targetLang === "zh" ? `\u7b2c ${chapter.number} \u7ae0` : `Chapter ${chapter.number}`;
}

// 首篇：前置内容（免责声明 / 关于作者 / 前言 / 引言 / 缩略语表）
const prelimChapters = doc.chapters.filter((c) => PRELIM_IDS.includes(c.id));
if (prelimChapters.length === 0) throw new Error("no prelim chapters found");
groups.push({
  slug: PRELIM_SLUG,
  date: dateForIndex(0),
  chapters: prelimChapters,
  number: null,
  isPrelim: true,
  part: null,
  title: {
    zh: "\u667a\u80fd\u4f53 AI \u6f2b\u6e38\u6307\u5357 \u00b7 \u524d\u8a00",
    en: "The Hitchhiker's Guide to Agentic AI \u00b7 Preface",
  },
  description: {
    zh: "\u300aThe Hitchhiker's Guide to Agentic AI\u300b\u4e2d\u8bd1\u672c\u524d\u8a00\uff1a\u514d\u8d23\u58f0\u660e\u3001\u4f5c\u8005\u4ecb\u7ecd\u3001\u5f15\u8a00\u4e0e\u7f29\u7565\u8bed\u8868\u3002",
    en: "Preface to The Hitchhiker's Guide to Agentic AI: disclaimer, about the author, introduction and glossary of acronyms.",
  },
});

// 30 章逐章成篇
const numberedChapters = doc.chapters
  .filter((c) => c.number !== null)
  .sort((a, b) => a.number - b.number);

/** velite 对 title 有 99 字符上限，超长时按词边界截断章名。 */
function fitTitle(prefix, label, title, limit = 99) {
  const full = `${prefix} · ${label} ${title}`;
  if (full.length <= limit) return full;
  const room = limit - `${prefix} · ${label} `.length - 1;
  if (room < 12) return `${prefix} · ${label}`;
  const cut = title.slice(0, room);
  const lastSpace = cut.lastIndexOf(" ");
  return `${prefix} · ${label} ${(lastSpace > 8 ? cut.slice(0, lastSpace) : cut).trim()}…`;
}

for (const [index, chapter] of numberedChapters.entries()) {
  const title = chapterTitle(chapter);
  const label = shortLabel(chapter, "en");
  const labelZh = shortLabel(chapter, "zh");
  groups.push({
    slug: `hitchhiker-agentic-ai-${String(chapter.number).padStart(2, "0")}-${slugifyChapter(slugTitles.get(chapter.number) ?? title, chapter.number)}`,
    date: dateForIndex(index + 1),
    chapters: [chapter],
    number: chapter.number,
    isPrelim: false,
    part: chapter.part ?? null,
    title: {
      zh: fitTitle("智能体 AI 漫游指南", labelZh, title),
      en: fitTitle("The Hitchhiker's Guide to Agentic AI", label, title),
    },
    description: {
      zh: chapterIntro(chapter) || `${labelZh}：${title}`,
      en: chapterIntro(chapter) || `${label}: ${title}`,
    },
  });
}

const ordered = groups;

// ---------------------------------------------------------------------------
// 输出
// ---------------------------------------------------------------------------

/** 后续文章链接（每篇结尾指向下一篇，形成阅读链）。 */
const nextOf = new Map();
for (let i = 0; i < ordered.length - 1; i++) {
  nextOf.set(ordered[i].slug, ordered[i + 1]);
}

/** 末篇附全书参考文献。 */
const lastSlug = ordered[ordered.length - 1].slug;

for (const group of ordered) {

  const assets = new Map(); // 目标文件名 -> 源绝对路径
  const ctx = {
    lang,
    bibMap,
    headingMap,
    labelMap,
    assetsDir,
    registerAsset(src) {
      const file = basename(src);
      if (!assets.has(file)) assets.set(file, join(assetsDir, file));
      return `./${file}`;
    },
  };

  const blocks = [];

  // 篇首标记：这批文章的标题编号统一走站点的 h2/h3 徽标（源文档的
  // 「1.1」等编号已在转换时剔掉），用这个容器类承载排版微调。
  blocks.push({ kind: "html", html: `<div class="hh-guide">` });

  // 分部说明：单章成篇时只标出所属部分（文章标题已写明章名，不再罗列）。
  if (group.part && group.number) {
    blocks.push({ kind: "partCard", part: group.part, chapters: [] });
  }

  for (const chapter of group.chapters) {
    // 单章成篇时章节标题即文章标题，正文内不再重复一遍 h2；
    // 首篇（前置内容）仍保留各段小标题。
    if (group.chapters.length > 1) {
      blocks.push({
        kind: "heading",
        level: 2,
        text: chapter.label ? `${chapter.label} ${chapter.title}` : chapter.title,
        id: chapter.id.toLowerCase(),
      });
    }
    // 第 28 章是自测题，题干与解答被 LaTeXML 包在 SVG 图里，需专门重建
    blocks.push(
      ...serializeBlocks(chapter.element, ctx, {
        quiz: chapter.id === "Ch28",
        // 单章成篇：章节标题已作文章标题，节标题上提一级（h3 -> h2）
        headingOffset: group.chapters.length === 1 ? -1 : 0,
      }),
    );
  }

  // 参考文献（源文档只有一份，附在末篇）
  if (group.slug === lastSlug) {
    const bib = findAll(doc.article, (n) => classes(n).includes("ltx_bibliography"))[0];
    if (bib) {
      const bibTitle = findFirst(bib, (n) => hasClass(n, "ltx_title"));
      blocks.push({
        kind: "heading",
        level: 2,
        text: bibTitle ? normText(bibTitle) : "\u53c2\u8003\u6587\u732e",
        id: "references",
      });
      blocks.push({ kind: "html", html: serializeReferences(bib, ctx) });
    }
  }

  const next = nextOf.get(group.slug);
  if (next) {
    blocks.push({ kind: "next", slug: next.slug, title: next.title[lang] });
  }
  blocks.push({ kind: "html", html: `</div>` });

  const markdown = renderBlocks(blocks);
  const dir = join(outRoot, `${group.date}-${group.slug}`);

  // 双语由两次独立运行产出（zh / en），不能整目录删除，否则会带走另一语言的文件。
  // 只清理本语言旧的正文，保留目录与另一语言版本。
  await mkdir(dir, { recursive: true });
  await rm(join(dir, `${lang}.md`), { force: true });
  for (const [file, from] of assets) {
    await copyFile(from, join(dir, file));
  }

  const frontMatter = buildFrontMatter(group, lang);
  await writeFile(
    join(dir, `${lang}.md`),
    `${frontMatter}\n\n${markdown.trim()}\n`,
    "utf8",
  );

  console.log(
    `[${lang}] ${group.slug}: ${group.chapters.length} \u7ae0 / ${(markdown.length / 1000).toFixed(0)}k \u5b57\u7b26 / ${assets.size} \u56fe`,
  );
}

// ---------------------------------------------------------------------------

function renderBlocks(blocks) {
  const out = [];
  for (const block of blocks) {
    switch (block.kind) {
      case "heading":
        out.push(`${"#".repeat(block.level)} ${block.text}\n`);
        break;
      case "paragraph":
      case "html":
        out.push(`${block.html}\n`);
        break;
      case "list":
      case "quote":
      case "figure":
      case "equation":
      case "table":
        out.push(block.html, "");
        break;
      case "code":
        out.push(`\`\`\`${block.lang ?? ""}\n${block.code}\n\`\`\`\n`);
        if (block.caption) out.push(`${block.caption}\n`);
        break;
      case "quiz":
        out.push(
          `<details class="hh-quiz" id="${block.id}">`,
          `<summary class="hh-quiz-question">${block.question.replace(/^<p>|<\/p>$/g, "")}</summary>`,
          `<div class="hh-quiz-answer">`,
          `<p class="hh-quiz-answer-label">解答</p>`,
          block.answer,
          `</div>`,
          `</details>`,
          "",
        );
        break;
      case "partCard": {
        const chapterList = block.chapters
          .map((c) => (c.label ? `${c.label} ${c.title}` : c.title))
          .join("、");
        out.push(
          `<aside class="hh-part">`,
          `<p class="hh-part-title">${block.part.label} ${block.part.title}</p>`,
          ...(chapterList
            ? [`<p class="hh-part-scope">本篇覆盖：${chapterList}</p>`]
            : []),
          `</aside>`,
          "",
        );
        break;
      }
      case "next":
        out.push(
          `<aside class="hh-next">`,
          `<p class="hh-next-label">下一篇</p>`,
          `<p class="hh-next-title">${block.title}</p>`,
          `<p class="hh-next-hint">在「智能体 AI 漫游指南」分类下继续阅读。</p>`,
          `</aside>`,
          "",
        );
        break;
      default:
        throw new Error(`unknown block kind ${block.kind}`);
    }
  }
  return out.join("\n");
}

function buildFrontMatter(group, lang) {
  const keywords = lang === "zh" ? KEYWORDS_ZH : KEYWORDS_EN;
  const lines = [
    "---",
    `title: ${JSON.stringify(group.title[lang])}`,
    `slug: ${JSON.stringify(group.slug)}`,
    `lang: ${JSON.stringify(lang)}`,
    `date: ${JSON.stringify(`${group.date}T00:00:00.000Z`)}`,
    "draft: false",
    "featured: false",
    "categories:",
    `  - ${JSON.stringify("hitchhiker-agentic-ai")}`,
    // 标题图由 generate-covers.mjs 生成到文章目录（cover.png）
    `cover: ${JSON.stringify("./cover.png")}`,
    `description: ${JSON.stringify(group.description[lang])}`,
    "keywords:",
    ...keywords.map((k) => `  - ${JSON.stringify(k)}`),
    "---",
  ];
  return lines.join("\n");
}
