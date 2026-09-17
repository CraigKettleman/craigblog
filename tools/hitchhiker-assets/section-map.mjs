/**
 * 《智能体 AI 漫游指南》分章映射。
 *
 * 源书 30 章 + 前置部分（免责声明 / 关于作者 / 前言 / 引言 / 缩略语表）。
 * 每章独立成一篇博客文章：slug 由章号与标题生成，发布日期按阅读顺序串成
 * 一天一篇的时间线，保证列表页与 RSS 的先后次序与书的顺序一致。
 *
 * 「前言」把 5 段前置内容合成一篇，作为整个系列的入口。
 */

/** 系列起始日期；每章 +1 天，形成连续时间线。 */
export const SERIES_START = "2026-09-15";

/** 前置内容合成一篇时的 slug。 */
export const PRELIM_SLUG = "hitchhiker-agentic-ai-0-preface";

/** 属于前置部分（不参与章号编号）的源章节 id。 */
export const PRELIM_IDS = ["Chx1", "Chx2", "Chx3", "Chx4", "Chx5"];

/**
 * 章标题里需要替换/清理的片段，用于生成 slug 与短标题。
 * 源标题形如「LLM Architecture and Optimization Methods」，
 * 中文形如「LLM 架构与优化方法」。
 */

/** 按顺序给第 n 章（1-based）分配日期。 */
export function dateForIndex(index) {
  const base = new Date(`${SERIES_START}T00:00:00.000Z`);
  base.setUTCDate(base.getUTCDate() + index);
  return base.toISOString().slice(0, 10);
}

/**
 * 由章标题生成 slug 片段：
 * 保留 ASCII 字母数字，其余转连字符；中文标题回退为 chapter-N。
 */
export function slugifyChapter(title, number) {
  const ascii = title
    .normalize("NFKD")
    .replace(/[^\x00-\x7F]/g, " ")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
  if (ascii.length >= 3) return ascii.slice(0, 48);
  return `chapter-${number}`;
}
