import { redirect } from "@sveltejs/kit";
import { getDictionary, type Language } from "$lib/dictionaries";
import type { PageServerLoad } from "./$types";

export const load: PageServerLoad = ({ params }) => {
  const lang = params.lang as Language;

  // 文章列表展示在「分享」页；直接访问 /{lang}/posts 时重定向过去。
  // 文章详情（/{lang}/posts/{slug}）与分类页走同一路由组，不受影响。
  redirect(301, getDictionary(lang).urls.share);
};
