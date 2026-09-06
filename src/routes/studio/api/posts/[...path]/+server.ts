import { error, json, type RequestHandler } from "@sveltejs/kit";
import { env } from "$env/dynamic/private";
import { readFile, writeFile, rm } from "node:fs/promises";
import { join } from "node:path";
import { parseFrontMatter, serializeFrontMatter } from "$lib/frontmatter";

const CONTENT = "content";
const LANGS = ["en", "zh"] as const;
type Lang = (typeof LANGS)[number];

function guard() {
  if (env.LOCAL_ADMIN !== "1") throw error(404);
}

function isLang(v: unknown): v is Lang {
  return v === "en" || v === "zh";
}

function safePath(p: string): string {
  if (!p || p.includes("..") || p.includes("\0")) throw error(400, "bad path");
  return p;
}

/** 读取某语言的 frontmatter + 正文；文件不存在时返回空结构。 */
async function readLangFile(dir: string, lang: Lang) {
  const file = join(CONTENT, dir, `${lang}.md`);
  try {
    const raw = await readFile(file, "utf8");
    const { data, body } = parseFrontMatter(raw);
    return { exists: true, data, body };
  } catch {
    return { exists: false, data: {} as Record<string, unknown>, body: "" };
  }
}

export const prerender = false;

/**
 * GET — 返回双语全量原文（正文 raw markdown + 全部 frontmatter 字段）。
 * path 为文章目录（如 posts/2026-01-01-hello）。
 */
export const GET: RequestHandler = async ({ params }) => {
  guard();
  const dir = safePath(params.path ?? "");

  const en = await readLangFile(dir, "en");
  const zh = await readLangFile(dir, "zh");
  if (!en.exists && !zh.exists) error(404, "post not found");

  const pick = (key: string, source: { data: Record<string, unknown> }) =>
    source.data[key] as string | undefined;

  return json({
    path: dir,
    section: "posts",
    slug: String(en.data.slug ?? zh.data.slug ?? ""),
    date: String(en.data.date ?? zh.data.date ?? ""),
    draft: Boolean(en.data.draft ?? zh.data.draft ?? false),
    featured: Boolean(en.data.featured ?? zh.data.featured ?? false),
    categories: Array.isArray(en.data.categories ?? zh.data.categories)
      ? ((en.data.categories ?? zh.data.categories) as string[])
      : [],
    keywords: Array.isArray(en.data.keywords ?? zh.data.keywords)
      ? ((en.data.keywords ?? zh.data.keywords) as string[])
      : undefined,
    title: {
      en: en.exists ? pick("title", en) ?? "" : undefined,
      zh: zh.exists ? pick("title", zh) ?? "" : undefined,
    },
    description: {
      en: en.exists ? pick("description", en) : undefined,
      zh: zh.exists ? pick("description", zh) : undefined,
    },
    body: { en: en.body, zh: zh.body },
  });
};

interface PutBody {
  /** 共享元数据（省略 = 不变） */
  slug?: string;
  date?: string;
  draft?: boolean;
  featured?: boolean;
  categories?: string[];
  keywords?: string[];
  /** 语言级字段；省略某语言 = 该语言不变；空串 title/description/body = 清空该项 */
  title?: Partial<Record<Lang, string>>;
  description?: Partial<Record<Lang, string | null>>;
  body?: Partial<Record<Lang, string>>;
}

/**
 * PUT — 双语无损合并保存：
 * - 请求中未涉及的字段/语言一律保持原文件内容；
 * - 只写请求涉及的字段；某语言文件不存在且请求未提供该语言内容时不创建；
 * - 这样在 /en 页面编辑绝不会碰坏 zh.md。
 */
export const PUT: RequestHandler = async ({ params, request }) => {
  guard();
  const dir = safePath(params.path ?? "");
  const body = (await request.json()) as PutBody;

  for (const lang of LANGS) {
    const current = await readLangFile(dir, lang);
    const wantsUpdate =
      body.title?.[lang] !== undefined ||
      body.description?.[lang] !== undefined ||
      body.body?.[lang] !== undefined;
    if (!current.exists && !wantsUpdate) continue; // 该语言不存在且本次不涉及 → 跳过
    if (!current.exists && body.slug === undefined && body.date === undefined) {
      // 该语言文件尚不存在，无法靠“共享元数据”单独创建
      if (!wantsUpdate) continue;
    }

    const data: Record<string, unknown> = { ...current.data };

    // 共享元数据写进 en 文件即可（zh 没有独立 slug/date 的概念时也补一份保持一致）
    if (body.slug !== undefined) data.slug = body.slug;
    if (body.date !== undefined) data.date = body.date;
    if (body.draft !== undefined) data.draft = body.draft;
    if (body.featured !== undefined) data.featured = body.featured;
    if (body.categories !== undefined) data.categories = body.categories;
    if (body.keywords !== undefined) data.keywords = body.keywords;

    if (body.title?.[lang] !== undefined) {
      data.title = body.title[lang];
    } else if (!data.title) {
      // en/zh 双份内容共用一个 slug；标题缺失时用另一语言的标题兜底
      const other: Lang = lang === "en" ? "zh" : "en";
      const otherData = (await readLangFile(dir, other)).data;
      if (typeof otherData.title === "string" && otherData.title) {
        data.title = otherData.title;
      }
    }
    if (body.description?.[lang] !== undefined) {
      const v = body.description[lang];
      if (v === null || v === "") delete data.description;
      else data.description = v;
    }
    if (data.lang === undefined) data.lang = lang; // lang 写死为文件语言

    let content = body.body?.[lang];
    if (content === undefined) content = current.body; // 未提供 → 保持原正文
    // 空串保护：正文显式给空串 = 清空，但文件里必须有 title/slug 等必需字段才能被 velite 接受
    const serialized = serializeFrontMatter(data, content);
    await writeFile(join(CONTENT, dir, `${lang}.md`), serialized, "utf8");
  }

  return json({ ok: true, path: dir });
};

export const DELETE: RequestHandler = async ({ params }) => {
  guard();
  const dir = safePath(params.path ?? "");
  await rm(join(CONTENT, dir), { recursive: true, force: true });
  return json({ ok: true });
};
