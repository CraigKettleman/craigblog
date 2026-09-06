import rehypePrettyCode from "rehype-pretty-code";
import { defineCollection, defineConfig, s } from "velite";

/**
 * 单一内容管线：
 *   content/posts/** -> section "posts"（对外称「分享」），served under /{lang}/posts
 * 分类在 content/categories/posts.yml 注册。
 */

const lang = s.enum(["en", "zh"]);

const localized = s.object({
  en: s.string().max(20),
  zh: s.string().max(20),
});

const localizedDescription = s
  .object({
    en: s.string().max(100),
    zh: s.string().max(100),
  })
  .optional();

const count = s
  .object({
    en: s.number(),
    zh: s.number(),
  })
  .default({ en: 0, zh: 0 });

/** 可指定长度上限的双语字符串。 */
const localizedText = (max: number) =>
  s.object({
    en: s.string().max(max),
    zh: s.string().max(max),
  });

function sectionOfPath(path: string): "posts" {
  return "posts";
}

const categories = defineCollection({
  name: "Category",
  pattern: "categories/*.yml",
  schema: s
    .object({
      slug: s.string(),
      name: localized,
      description: localizedDescription,
      count,
      path: s.path(),
    })
    .transform(({ path, ...data }) => {
      const section = sectionOfPath(path);
      return {
        ...data,
        section,
        permalink: {
          en: `/en/${section}/categories/${data.slug}`,
          zh: `/zh/${section}/categories/${data.slug}`,
        },
      };
    }),
});

const site = defineCollection({
  name: "Site",
  pattern: "site.yml",
  single: true,
  schema: s.object({
    websiteName: localizedText(30),
    motto: localizedText(300),
    mottos: s.object({
      en: s.array(s.string().max(300)).default([]),
      zh: s.array(s.string().max(300)).default([]),
    }),
    brandName: localizedText(30),
    brandTagline: localizedText(60),
    // 「更多」页个人简介（markdown 文本，双语）
    about: s
      .object({
        en: s.string().max(50000),
        zh: s.string().max(50000),
      })
      .default({ en: "", zh: "" }),
  }),
});

const posts = defineCollection({
  name: "Post",
  pattern: ["posts/**/*.md"],
  schema: s
    .object({
      title: s.string().max(99),
      slug: s.string(),
      lang,
      date: s.isodate(),
      updated: s.isodate().optional(),
      cover: s.image().optional(),
      video: s.file().optional(),
      description: s.string().max(999).optional(),
      keywords: s.array(s.string()).optional(),
      draft: s.boolean().default(false),
      featured: s.boolean().default(false),
      categories: s.array(s.string()),
      wechatLink: s.string().optional(),
      excerpt: s.excerpt(),
      content: s.markdown(),
      path: s.path(),
    })
    .transform(({ path, ...data }) => {
      const section = sectionOfPath(path);
      // s.path() 是无扩展名文件路径（如 posts/2026-01-01-x/en），
      // 剥掉语言后缀得到文章目录路径，供后台按目录读写双语文件。
      const dirPath = path.replace(/\/[a-z]{2}$/, "");
      return {
        ...data,
        path: dirPath,
        section,
        permalink: `/${data.lang}/${section}/${data.slug}`,
      };
    }),
});

export default defineConfig({
  root: "content",
  output: {
    data: ".velite",
    assets: "static/blog",
    base: "/blog/",
    name: "[name]-[hash:6].[ext]",
    clean: true,
  },
  collections: { site, categories, posts },
  markdown: { rehypePlugins: [rehypePrettyCode] },
  prepare: ({ categories, posts }) => {
    const unknownCategories = posts
      .flatMap((post) =>
        post.categories.map((slug) => ({ section: post.section, slug })),
      )
      .filter(
        ({ section, slug }) =>
          !categories.some((c) => c.section === section && c.slug === slug),
      );

    if (unknownCategories.length > 0) {
      console.error(
        "Unknown categories found:",
        unknownCategories
          .map(({ section, slug }) => `${section}/${slug}`)
          .join(", "),
      );
      return false;
    }

    for (const category of categories) {
      category.count = {
        en: 0,
        zh: 0,
      };
      for (const post of posts) {
        if (
          post.section === category.section &&
          post.categories.includes(category.slug)
        ) {
          category.count[post.lang] += 1;
        }
      }
    }
  },
});
