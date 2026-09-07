import { error, redirect } from "@sveltejs/kit";
import { env } from "$env/dynamic/private";
import type { PageServerLoad } from "./$types";

export const prerender = false;

/**
 * 传统双栏后台已移除：后台即公开站本身（镜像可编辑模式）。
 * /studio 仅作为入口重定向到语言首页，按 accept-language 选择语言。
 */
export const load: PageServerLoad = ({ request }) => {
  if (env.LOCAL_ADMIN !== "1") throw error(404);
  const wantsZh = (request.headers.get("accept-language") ?? "")
    .toLowerCase()
    .startsWith("zh");
  redirect(302, wantsZh ? "/zh" : "/en");
};
