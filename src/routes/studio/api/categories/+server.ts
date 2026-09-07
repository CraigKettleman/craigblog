import { error, json, type RequestHandler } from "@sveltejs/kit";
import { env } from "$env/dynamic/private";
import { readdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { categories } from "#velite";
import { parseFrontMatter, serializeFrontMatter } from "$lib/frontmatter";
import {
  countCategoryUsage,
  readCategoryFile,
  writeCategoryFile,
  type CategoryEntry,
  type CategorySection,
} from "$lib/server/categories-file";

const CONTENT = "content";
const SECTIONS: CategorySection[] = ["posts", "projects"];

function guard() {
  if (env.LOCAL_ADMIN !== "1") throw error(404);
}

function parseSection(value: unknown): CategorySection {
  if (value === "posts" || value === "projects") return value;
  throw error(400, "invalid section");
}

/** 分类 slug 约束：字母数字下划线连字符与中文，禁空格/斜杠等破坏路径的字符。 */
function validSlug(slug: string): boolean {
  return /^[\w\u4e00-\u9fff-]+$/.test(slug);
}

function slugifyName(name: string): string {
  return (
    name
      .toLowerCase()
      .replace(/[^a-z0-9\u4e00-\u9fff]+/g, "-")
      .replace(/^-|-$/g, "")
      .slice(0, 40) || "category"
  );
}

export const prerender = false;

export const GET: RequestHandler = ({ url }) => {
  guard();
  const section = url.searchParams.get("section");
  return json(
    categories
      .filter((c) => !section || c.section === section)
      .map((c) => ({
        slug: c.slug,
        section: c.section,
        name: c.name,
        description: c.description,
        count: c.count,
      })),
  );
};

/** 新建分类：slug 缺省时由名称生成；双语名必须至少给一种语言，另一种以同一文本兜底。 */
export const POST: RequestHandler = async ({ request }) => {
  guard();
  const body = (await request.json()) as {
    section?: string;
    slug?: string;
    name?: { en?: string; zh?: string };
    description?: { en?: string; zh?: string };
  };
  const section = parseSection(body.section);
  const nameEn = (body.name?.en ?? "").trim();
  const nameZh = (body.name?.zh ?? "").trim();
  const fallback = nameEn || nameZh;
  if (!fallback) throw error(400, "分类名称不能为空");

  const entries = await readCategoryFile(section);
  const slug = (body.slug ?? "").trim() || fallback.toLowerCase().replace(/\s+/g, "-");
  if (!validSlug(slug)) throw error(400, "slug 含非法字符");
  if (entries.some((e) => e.slug === slug)) throw error(409, `分类 ${slug} 已存在`);

  const entry: CategoryEntry = {
    slug,
    name: { en: nameEn || fallback, zh: nameZh || fallback },
  };
  if (body.description?.en || body.description?.zh) {
    entry.description = { en: body.description.en ?? "", zh: body.description.zh ?? "" };
  }
  entries.push(entry);
  await writeCategoryFile(section, entries);
  return json({ ok: true, slug }, { status: 201 });
};

/**
 * 修改分类：名称/描述部分更新；newSlug 更名时同步改写所有引用该分类的文章
 * frontmatter（分类在文章里按 slug 存储），保持数据一致。
 */
export const PUT: RequestHandler = async ({ request }) => {
  guard();
  const body = (await request.json()) as {
    section?: string;
    slug?: string;
    newSlug?: string;
    name?: { en?: string; zh?: string };
    description?: { en?: string; zh?: string; clear?: boolean };
  };
  const section = parseSection(body.section);
  if (!body.slug) throw error(400, "slug required");

  const entries = await readCategoryFile(section);
  const index = entries.findIndex((e) => e.slug === body.slug);
  if (index === -1) throw error(404, "分类不存在");
  const entry = entries[index];

  const finalSlug = (body.newSlug ?? "").trim() || body.slug;
  if (!validSlug(finalSlug)) throw error(400, "slug 含非法字符");
  if (
    finalSlug !== entry.slug &&
    entries.some((e) => e.slug === finalSlug)
  ) {
    throw error(409, `分类 ${finalSlug} 已存在`);
  }

  if (body.name?.en !== undefined) entry.name.en = body.name.en.trim();
  if (body.name?.zh !== undefined) entry.name.zh = body.name.zh.trim();
  if (body.description?.clear) {
    delete entry.description;
  } else if (body.description) {
    entry.description ??= { en: "", zh: "" };
    if (body.description.en !== undefined) entry.description.en = body.description.en.trim();
    if (body.description.zh !== undefined) entry.description.zh = body.description.zh.trim();
  }
  entry.slug = finalSlug;
  await writeCategoryFile(section, entries);

  // slug 更名：改写所有引用旧 slug 的文章 frontmatter（分类以 slug 关联）
  if (finalSlug !== body.slug) {
    const dir = join(CONTENT, section);
    let names: string[] = [];
    try {
      names = await readdir(dir);
    } catch {
      names = [];
    }
    for (const name of names) {
      for (const lang of ["en", "zh"] as const) {
        const file = join(dir, name, `${lang}.md`);
        try {
          const raw = await readFile(file, "utf8");
          const { data, body: mdBody } = parseFrontMatter(raw);
          const cats = Array.isArray(data.categories) ? data.categories.map(String) : [];
          if (!cats.includes(body.slug)) continue;
          const updated = cats.map((c) => (c === body.slug ? finalSlug : c));
          const fm = { ...data, categories: updated };
          await writeFile(file, serializeFrontMatter(fm, mdBody), "utf8");
        } catch {
          /* 该语言文件不存在，跳过 */
        }
      }
    }
  }

  return json({ ok: true, slug: finalSlug });
};

/** 删除分类：仍被文章引用时拒绝（409），避免 velite 校验失败导致整站内容管线报错。 */
export const DELETE: RequestHandler = async ({ request }) => {
  guard();
  const body = (await request.json()) as { section?: string; slug?: string };
  const section = parseSection(body.section);
  if (!body.slug) throw error(400, "slug required");

  const usage = await countCategoryUsage(section, body.slug);
  const total = usage.en + usage.zh;
  if (total > 0) {
    throw error(409, `该分类仍被 ${total} 篇文章使用，请先在文章中移除引用`);
  }

  const entries = await readCategoryFile(section);
  const next = entries.filter((e) => e.slug !== body.slug);
  if (next.length === entries.length) throw error(404, "分类不存在");
  await writeCategoryFile(section, next);
  return json({ ok: true });
};
