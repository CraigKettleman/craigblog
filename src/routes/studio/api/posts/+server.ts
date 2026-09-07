import { error, json, type RequestHandler } from "@sveltejs/kit";
import { env } from "$env/dynamic/private";
import { readdir, readFile, mkdir, writeFile, rm } from "node:fs/promises";
import { join } from "node:path";
import { parseFrontMatter, serializeFrontMatter } from "$lib/frontmatter";

const CONTENT = "content";
const SECTIONS = ["posts", "projects"] as const;
type Section = (typeof SECTIONS)[number];

function guard() {
  if (env.LOCAL_ADMIN !== "1") throw error(404);
}

function parseSection(value: unknown): Section {
  return value === "projects" ? "projects" : "posts";
}

interface PostMeta {
  path: string;
  section: Section;
  slug: string;
  date: string;
  draft: boolean;
  featured: boolean;
  categories: string[];
  title: { en: string; zh: string };
}

async function scanPosts(): Promise<PostMeta[]> {
  const result: PostMeta[] = [];
  for (const section of SECTIONS) {
    const dir = join(CONTENT, section);
    let entries: string[];
    try {
      entries = await readdir(dir);
    } catch {
      continue;
    }
    for (const entry of entries) {
      const postDir = join(dir, entry);
      // 以任一语言文件存在为准（单语言投稿也可见）
      let raw: string | null = null;
      for (const lang of ["en", "zh"] as const) {
        try {
          raw = await readFile(join(postDir, `${lang}.md`), "utf8");
          break;
        } catch {
          /* 尝试下一语言 */
        }
      }
      if (raw === null) continue;
      const { data } = parseFrontMatter(raw);
      const slug = (data.slug as string) || entry;
      const date = (data.date as string) || "";
      result.push({
        path: `${section}/${entry}`,
        section,
        slug,
        date: typeof date === "string" ? date : "",
        draft: Boolean(data.draft),
        featured: Boolean(data.featured),
        categories: Array.isArray(data.categories) ? (data.categories as string[]) : [],
        title: { en: (data.title as string) || slug, zh: "" },
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
    section?: string;
    slug: string;
    date: string;
    draft: boolean;
    featured: boolean;
    categories: string[];
    title: { en: string; zh: string };
    content: { en: string; zh: string };
  };

  const section = parseSection(body.section);
  const slug = body.slug || body.title.en.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  const dirName = `${body.date}-${slug}`;
  const postDir = join(CONTENT, section, dirName);

  await mkdir(postDir, { recursive: true });

  for (const lang of ["en", "zh"] as const) {
    const title = body.title[lang] || body.title.en;
    const content = body.content[lang] || "";

    const fmData: Record<string, unknown> = {
      title: title,
      slug: slug,
      lang: lang,
      date: body.date,
      draft: body.draft,
      featured: body.featured,
      categories: body.categories,
    };

    const fileContent = serializeFrontMatter(fmData, content);
    await writeFile(join(postDir, `${lang}.md`), fileContent, "utf8");
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
