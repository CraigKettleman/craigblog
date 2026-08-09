import { error, json, type RequestHandler } from "@sveltejs/kit";
import { env } from "$env/dynamic/private";
import { readdir, readFile, mkdir, writeFile, rm } from "node:fs/promises";
import { join, basename } from "node:path";

const CONTENT = "content";

function guard() {
  if (env.LOCAL_ADMIN !== "1") throw error(404);
}

interface PostMeta {
  path: string;
  section: string;
  slug: string;
  date: string;
  draft: boolean;
  featured: boolean;
  categories: string[];
  title: { en: string; zh: string };
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
      // Parse arrays: [a, b, c]
      if (typeof val === "string" && val.startsWith("[") && val.endsWith("]")) {
        val = val.slice(1, -1).split(",").map((s) => s.trim().replace(/^['"]|['"]$/g, "")).filter(Boolean);
      }
      // Parse booleans
      if (val === "true") val = true;
      if (val === "false") val = false;
      fm[key] = val;
    }
  }
  return { fm, body: match[2] };
}

async function scanPosts(): Promise<PostMeta[]> {
  const result: PostMeta[] = [];
  for (const section of ["posts", "life-posts"]) {
    const dir = join(CONTENT, section);
    let entries: string[];
    try {
      entries = await readdir(dir);
    } catch {
      continue;
    }
    for (const entry of entries) {
      const postDir = join(dir, entry);
      // Skip non-directories
      const enFile = join(postDir, "en.md");
      let raw: string;
      try {
        raw = await readFile(enFile, "utf8");
      } catch {
        continue;
      }
      const { fm } = parseFrontMatter(raw);
      const slug = (fm.slug as string) || entry;
      const date = (fm.date as string) || "";
      result.push({
        path: `${section}/${entry}`,
        section: section === "life-posts" ? "life" : "posts",
        slug,
        date: typeof date === "string" ? date : "",
        draft: Boolean(fm.draft),
        featured: Boolean(fm.featured),
        categories: Array.isArray(fm.categories) ? fm.categories as string[] : [],
        title: { en: (fm.title as string) || slug, zh: "" },
      });
    }
  }
  result.sort((a, b) => (a.date < b.date ? 1 : -1));
  return result;
}

export const prerender = false;

export const GET: RequestHandler = async () => {
  guard();
  return json(await scanPosts());
};

export const POST: RequestHandler = async ({ request }) => {
  guard();
  const body = (await request.json()) as {
    section: string;
    slug: string;
    date: string;
    draft: boolean;
    featured: boolean;
    categories: string[];
    title: { en: string; zh: string };
    content: { en: string; zh: string };
  };

  const section = body.section === "life-posts" ? "life-posts" : "posts";
  const slug = body.slug || body.title.en.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  const dirName = `${body.date}-${slug}`;
  const postDir = join(CONTENT, section, dirName);

  await mkdir(postDir, { recursive: true });

  const categories = JSON.stringify(body.categories);

  for (const lang of ["en", "zh"] as const) {
    const title = body.title[lang] || body.title.en;
    const content = body.content[lang] || "";
    const fm = [
      "---",
      `title: ${title}`,
      `slug: ${slug}`,
      `lang: ${lang}`,
      `date: ${body.date}`,
      `draft: ${body.draft}`,
      `featured: ${body.featured}`,
      `categories: ${categories}`,
      "---",
      "",
      content,
    ].join("\n");
    await writeFile(join(postDir, `${lang}.md`), fm, "utf8");
  }

  return json({ path: `${section}/${dirName}` }, { status: 201 });
};

/** Delete a post directory and its contents. */
export const DELETE: RequestHandler = async ({ request }) => {
  guard();
  const { path } = (await request.json()) as { path: string };
  if (!path || path.includes("..")) throw error(400, "bad path");
  const full = join(CONTENT, path);
  await rm(full, { recursive: true, force: true });
  return json({ ok: true });
};
