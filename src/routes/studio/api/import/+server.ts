import { error, json, type RequestHandler } from "@sveltejs/kit";
import { env } from "$env/dynamic/private";
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { parseFrontMatter, serializeFrontMatter } from "$lib/frontmatter";

const CONTENT = "content";

function guard() {
  if (env.LOCAL_ADMIN !== "1") throw error(404);
}

export const prerender = false;

interface ImportBody {
  section: "posts";
  lang: "en" | "zh";
  markdown: string;
  title?: string;
  slug?: string;
  date?: string;
  draft?: boolean;
  featured?: boolean;
  categories?: string[];
  description?: string;
  keywords?: string[];
  cover?: string;
  video?: string;
  updated?: string;
  wechatLink?: string;
}

function slugify(input: string): string {
  return input
    .toLowerCase()
    .replace(/[^a-z0-9\u4e00-\u9fff]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 80) || "untitled";
}

export const POST: RequestHandler = async ({ request }) => {
  guard();
  const body = (await request.json()) as ImportBody;
  if (!body.markdown) throw error(400, "markdown required");
  if (body.section !== "posts") {
    throw error(400, "invalid section");
  }
  if (body.lang !== "en" && body.lang !== "zh") {
    throw error(400, "invalid lang");
  }

  const { data: parsed, body: parsedBody } = parseFrontMatter(body.markdown);

  const title =
    body.title ||
    (typeof parsed.title === "string" ? parsed.title : "") ||
    "untitled";
  const slug =
    body.slug ||
    (typeof parsed.slug === "string" ? parsed.slug : "") ||
    slugify(title);
  const date =
    body.date ||
    (typeof parsed.date === "string" ? parsed.date.slice(0, 10) : "") ||
    new Date().toISOString().slice(0, 10);
  const categories =
    body.categories ??
    (Array.isArray(parsed.categories) ? parsed.categories.map(String) : []);
  const draft =
    body.draft ??
    (typeof parsed.draft === "boolean" ? parsed.draft : true);
  const featured =
    body.featured ??
    (typeof parsed.featured === "boolean" ? parsed.featured : false);
  const description =
    body.description ??
    (typeof parsed.description === "string" ? parsed.description : undefined);
  const keywords =
    body.keywords ??
    (Array.isArray(parsed.keywords)
      ? parsed.keywords.map(String)
      : undefined);
  const cover =
    body.cover ??
    (typeof parsed.cover === "string" ? parsed.cover : undefined);
  const video =
    body.video ??
    (typeof parsed.video === "string" ? parsed.video : undefined);
  const updated =
    body.updated ??
    (typeof parsed.updated === "string" ? parsed.updated : undefined);
  const wechatLink =
    body.wechatLink ??
    (typeof parsed.wechatLink === "string" ? parsed.wechatLink : undefined);

  const sectionDir = "posts";
  const dirName = `${date}-${slug}`;
  const postDir = join(CONTENT, sectionDir, dirName);

  const fmData: Record<string, unknown> = {
    title,
    slug,
    lang: body.lang,
    date,
    draft,
    featured,
    categories,
  };
  if (description) fmData.description = description;
  if (keywords && keywords.length > 0) fmData.keywords = keywords;
  if (cover) fmData.cover = cover;
  if (video) fmData.video = video;
  if (updated) fmData.updated = updated;
  if (wechatLink) fmData.wechatLink = wechatLink;

  await mkdir(postDir, { recursive: true });
  const content = serializeFrontMatter(fmData, parsedBody);
  await writeFile(join(postDir, `${body.lang}.md`), content, "utf8");

  return json({ path: `${sectionDir}/${dirName}` }, { status: 201 });
};
