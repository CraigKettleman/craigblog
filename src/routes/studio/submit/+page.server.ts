import { error } from "@sveltejs/kit";
import { env } from "$env/dynamic/private";
import type { PageServerLoad } from "./$types";

export const prerender = false;

export const load: PageServerLoad = ({ request, url }) => {
  if (env.LOCAL_ADMIN !== "1") throw error(404);
  // 非本地化路由：把 ?lang= 透传给根布局，用于打印机外壳的界面语言
  const q = url.searchParams.get("lang");
  return { lang: q === "zh" || q === "en" ? q : undefined };
};
