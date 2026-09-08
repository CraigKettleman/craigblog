import { error } from "@sveltejs/kit";
import { env } from "$env/dynamic/private";
import type { PageServerLoad } from "./$types";
import { categories, posts } from "$lib/content";

export const prerender = false;

/** studio 数据看板：与 submit 同源的非本地化后台路由，用 ?lang= 透传界面语言 */
export const load: PageServerLoad = ({ request, url }) => {
  if (env.LOCAL_ADMIN !== "1") throw error(404);
  const q = url.searchParams.get("lang");
  let lang: "zh" | "en" | undefined;
  if (q === "zh" || q === "en") {
    lang = q;
  } else {
    const accept = (request.headers.get("accept-language") ?? "").toLowerCase();
    lang = accept.startsWith("zh") ? "zh" : "en";
  }
  // 只下发轻量字段（不含正文），热力 / TOP 榜在客户端聚合
  const litePosts = posts.map((p) => ({
    title: p.title,
    slug: p.slug,
    lang: p.lang,
    section: p.section,
    date: p.date.slice(0, 10),
    categories: p.categories,
    permalink: p.permalink,
    draft: p.draft,
  }));
  const liteCategories = categories.map((c) => ({
    slug: c.slug,
    section: c.section,
    name: c.name,
    count: c.count,
  }));
  return { lang, posts: litePosts, categories: liteCategories, generatedAt: new Date().toISOString() };
};
