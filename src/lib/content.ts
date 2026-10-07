import { categories as allCategories, posts as allPosts } from "#velite";
import type { Language } from "$lib/dictionaries";

export const sections = ["posts", "projects"] as const;

export type Section = (typeof sections)[number];

type Localized = Record<Language, string>;

export interface Post {
  title: string;
  slug: string;
  lang: Language;
  section: Section;
  date: string;
  updated?: string;
  cover?: { src: string; width: number; height: number };
  video?: string;
  description?: string;
  keywords?: string[];
  draft: boolean;
  featured: boolean;
  categories: string[];
  wechatLink?: string;
  excerpt: string;
  content: string;
  permalink: string;
  path: string;
}

export interface Category {
  slug: string;
  section: Section;
  name: Localized;
  description?: Localized;
  count: Record<Language, number>;
  permalink: Localized;
}

export function isSection(value: string): value is Section {
  return (sections as readonly string[]).includes(value);
}

/**
 * 草稿只在 dev 构建里可见。
 * 用构建模式 MODE 而非 import.meta.env.DEV 判断：Vite 的 DEV 取自进程环境变量
 * NODE_ENV，构建时该变量若被外部置成 development，生产构建会把草稿当成已发布
 * 内容一起 prerender 出去（草稿页面与 sitemap 都会带上）。MODE 只跟 --mode 走。
 */
const includeDrafts = import.meta.env.MODE !== "production";

/** All published posts, newest first. */
export const posts: Post[] = (allPosts as unknown as Post[])
  .filter((post) => includeDrafts || !post.draft)
  .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());

export const categories: Category[] = allCategories as unknown as Category[];

/** The lightweight shape sent to list pages (no rendered content). */
export type PostListItem = Pick<
  Post,
  | "title"
  | "slug"
  | "lang"
  | "section"
  | "date"
  | "description"
  | "cover"
  | "categories"
  | "permalink"
  | "draft"
>;

export function toListItem(post: Post): PostListItem {
  return {
    title: post.title,
    slug: post.slug,
    lang: post.lang,
    section: post.section,
    date: post.date,
    description: post.description,
    cover: post.cover,
    categories: post.categories,
    permalink: post.permalink,
    draft: post.draft,
  };
}

export function postsOf(
  lang: Language,
  section?: Section,
  category?: string,
): Post[] {
  return posts.filter(
    (post) =>
      post.lang === lang &&
      (!section || post.section === section) &&
      (!category || post.categories.includes(category)),
  );
}

export function findPost(
  lang: Language,
  section: Section,
  slug: string,
): Post | undefined {
  return posts.find(
    (post) =>
      post.lang === lang && post.section === section && post.slug === slug,
  );
}

/** The same post in other languages, for hreflang alternates. */
export function postTranslations(post: Post): Post[] {
  return posts.filter(
    (other) => other.section === post.section && other.slug === post.slug,
  );
}

export function categoriesOf(section: Section): Category[] {
  return categories.filter((category) => category.section === section);
}

export function findCategory(
  section: Section,
  slug: string,
): Category | undefined {
  return categories.find(
    (category) => category.section === section && category.slug === slug,
  );
}

/**
 * 系列内的「下一篇」。
 *
 * 《智能体 AI 漫游指南》按章节拆成多篇，属于同一分类且同语言。
 * 站点的发布日期按阅读顺序逐日递增，因此按日期升序取当前篇的下一章即可；
 * 末篇返回 undefined。仅对注册了该分类的文章生效，其余文章不受影响。
 */
export function nextInSeries(post: Post): Post | undefined {
  const SERIES_CATEGORIES = ["hitchhiker-agentic-ai"];
  if (!post.categories.some((slug) => SERIES_CATEGORIES.includes(slug))) {
    return undefined;
  }
  const siblings = posts
    .filter(
      (other) =>
        other.lang === post.lang &&
        other.section === post.section &&
        other.categories.some((slug) => SERIES_CATEGORIES.includes(slug)),
    )
    .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
  const index = siblings.findIndex((other) => other.path === post.path);
  return index >= 0 ? siblings[index + 1] : undefined;
}
