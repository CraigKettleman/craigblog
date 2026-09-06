import { error } from "@sveltejs/kit";
import { postsOf, type Section } from "$lib/content";
import { generateRssFeed } from "$lib/feed";
import { getDictionary, type Language } from "$lib/dictionaries";
import { siteMeta } from "$lib/site";
import type { RequestHandler } from "./$types";

// Feeds keep their extensionless URLs (/feed, /feed/zh/tech, ...), so they
// are served by the Node server instead of being prerendered — a static file
// without an extension would lose its XML content type.
export const prerender = false;

const FEED_DESCRIPTIONS: Record<Language, Record<string, string>> = {
  en: {
    all: "All posts from Craig — tech articles and essays.",
    tech: "Tech articles and development insights from Craig.",
  },
  zh: {
    all: "Craig 的全部文章。",
    tech: "Craig 的文章。",
  },
};

function parsePath(
  path: string,
): { lang: Language; section?: Section } | undefined {
  const segments = path.split("/").filter(Boolean);
  const lang: Language = segments[0] === "zh" ? "zh" : "en";
  if (segments[0] === "zh") segments.shift();

  if (segments.length === 0) return { lang };
  if (segments.length > 1) return undefined;
  if (segments[0] === "tech") return { lang, section: "posts" };
  return undefined;
}

export const GET: RequestHandler = async ({ params, url }) => {
  const parsed = parsePath(params.path);
  if (!parsed) error(404, "Unknown feed");

  const { lang, section } = parsed;
  const dictionary = getDictionary(lang);
  const site = siteMeta(lang);
  const kind = section === "posts" ? "tech" : "all";
  const suffix =
    kind === "all" ? "" : ` - ${dictionary.labels[section as Section]}`;

  const xml = generateRssFeed({
    title: `${site.websiteName}${suffix}`,
    description: FEED_DESCRIPTIONS[lang][kind],
    link: url.pathname,
    language: lang,
    items: postsOf(lang, section).map((post) => ({
      title: post.title,
      description: post.description,
      link: post.permalink,
      date: post.date,
      categories: post.categories,
      author: "craigmail.ai@gmail.com (Craig)",
    })),
  });

  return new Response(xml, {
    headers: {
      "Content-Type": "application/xml; charset=utf-8",
      "Cache-Control": "public, max-age=3600, s-maxage=3600",
    },
  });
};
