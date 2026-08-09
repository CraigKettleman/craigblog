import { error, json, type RequestHandler } from "@sveltejs/kit";
import { env } from "$env/dynamic/private";
import { readFile, writeFile, rm } from "node:fs/promises";
import { join } from "node:path";

const CONTENT = "content";

function guard() {
  if (env.LOCAL_ADMIN !== "1") throw error(404);
}

function parseFrontMatter(raw: string): { fm: Record<string, unknown>; body: string } {
  const match = raw.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/);
  if (!match) return { fm: {}, body: raw };
  const fm: Record<string, unknown> = {};
  for (const line of match[1].split("\n")) {
    const kv = line.match(/^([a-zA-Z_]+):\s*(.*)$/);
    if (kv) {
      const key = kv[1];
      let val: unknown = kv[2].trim();
      if (typeof val === "string" && val.startsWith("[") && val.endsWith("]")) {
        val = val.slice(1, -1).split(",").map((s) => s.trim().replace(/^['"]|['"]$/g, "")).filter(Boolean);
      }
      if (val === "true") val = true;
      if (val === "false") val = false;
      fm[key] = val;
    }
  }
  return { fm, body: match[2] };
}

function stringifyCat(categories: string[]): string {
  if (categories.length === 0) return "[]";
  return `[${categories.join(", ")}]`;
}

export const prerender = false;

export const GET: RequestHandler = async ({ params }) => {
  guard();
  const p = params.path ?? "";
  if (!p || p.includes("..")) throw error(400, "bad path");

  const enRaw = await readFile(join(CONTENT, p, "en.md"), "utf8");
  const zhRaw = await readFile(join(CONTENT, p, "zh.md"), "utf8");
  const en = parseFrontMatter(enRaw);
  const zh = parseFrontMatter(zhRaw);
  const section = p.startsWith("life-posts/") ? "life" : "posts";

  return json({
    path: p,
    section,
    slug: en.fm.slug || "",
    date: en.fm.date || "",
    draft: Boolean(en.fm.draft),
    featured: Boolean(en.fm.featured),
    categories: Array.isArray(en.fm.categories) ? en.fm.categories : [],
    title: { en: en.fm.title || "", zh: zh.fm.title || "" },
    content: { en: en.body, zh: zh.body },
  });
};

export const PUT: RequestHandler = async ({ params, request }) => {
  guard();
  const p = params.path ?? "";
  if (!p || p.includes("..")) throw error(400, "bad path");

  const body = (await request.json()) as {
    slug: string;
    date: string;
    draft: boolean;
    featured: boolean;
    categories: string[];
    title: { en: string; zh: string };
    content: { en: string; zh: string };
  };

  const categories = stringifyCat(body.categories);

  for (const lang of ["en", "zh"] as const) {
    const title = body.title[lang] || body.title.en;
    const content = body.content[lang] || "";
    const fm = [
      "---",
      `title: ${title}`,
      `slug: ${body.slug}`,
      `lang: ${lang}`,
      `date: ${body.date}`,
      `draft: ${body.draft}`,
      `featured: ${body.featured}`,
      `categories: ${categories}`,
      "---",
      "",
      content,
    ].join("\n");
    await writeFile(join(CONTENT, p, `${lang}.md`), fm, "utf8");
  }

  return json({ ok: true, path: p });
};

export const DELETE: RequestHandler = async ({ params }) => {
  guard();
  const p = params.path ?? "";
  if (!p || p.includes("..")) throw error(400, "bad path");
  await rm(join(CONTENT, p), { recursive: true, force: true });
  return json({ ok: true });
};
