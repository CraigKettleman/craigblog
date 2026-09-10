import { error } from "@sveltejs/kit";
import QRCode from "qrcode";
import {
  categoriesOf,
  findPost,
  posts,
  postTranslations,
  type Post,
  type Section,
} from "$lib/content";
import type { Language } from "$lib/dictionaries";
import { extractToc } from "$lib/toc";
import type { EntryGenerator, PageServerLoad } from "./$types";

export const entries: EntryGenerator = () =>
  posts.map((post) => ({
    lang: post.lang,
    section: post.section,
    slug: post.slug,
  }));

/**
 * 文章规格铭牌数据：
 * - 编号：同栏目内按日期升序的去重文章序号（双语同 slug 共用一个编号）
 * - 字数：中文按汉字 + 英文单词计，英文按空白分词计
 * - 时长：中文约 400 字/分钟，英文约 200 词/分钟，向上取整
 */
function postSpecs(post: Post) {
  const chronological = posts
    .filter((p) => p.section === post.section)
    .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
  const seen = new Set<string>();
  const orderedPaths: string[] = [];
  for (const p of chronological) {
    if (!seen.has(p.path)) {
      seen.add(p.path);
      orderedPaths.push(p.path);
    }
  }
  const postNumber = String(orderedPaths.indexOf(post.path) + 1).padStart(
    3,
    "0",
  );

  // 剧除标签与代码块后统计正文字数
  const text = post.content
    .replace(/<pre[\s\S]*?<\/pre>/g, " ")
    .replace(/<[^>]+>/g, " ");
  const words =
    post.lang === "zh"
      ? (text.match(/[\u4e00-\u9fff]/g)?.length ?? 0) +
        (text.match(/[a-zA-Z]+/g)?.length ?? 0)
      : text.trim().split(/\s+/).filter(Boolean).length;
  const readingMinutes = Math.max(1, Math.round(words / (post.lang === "zh" ? 400 : 200)));

  return {
    postNumber,
    wordCount: words.toLocaleString("en-US"),
    readingMinutes,
  };
}

export const load: PageServerLoad = async ({ params }) => {
  const lang = params.lang as Language;
  const section = params.section as Section;

  const post = findPost(lang, section, params.slug);
  if (!post) error(404, "Post not found");

  const translations = postTranslations(post)
    .filter((other) => other.lang !== lang)
    .map((other) => ({ lang: other.lang, permalink: other.permalink }));

  const categories = categoriesOf(section)
    .filter((category) => post.categories.includes(category.slug))
    .map((category) => ({
      slug: category.slug,
      name: category.name,
      permalink: category.permalink,
    }));

  // Pre-render the WeChat QR code as inline SVG at build time.
  const wechatQrSvg = post.wechatLink
    ? await QRCode.toString(post.wechatLink, {
        type: "svg",
        errorCorrectionLevel: "M",
        margin: 0,
        color: { dark: "#2C2824", light: "#ffffff" },
      })
    : undefined;

  return {
    section,
    post,
    translations,
    categories,
    wechatQrSvg,
    toc: extractToc(post.content),
    ...postSpecs(post),
  };
};
