<script lang="ts">
  import { page } from "$app/state";
  import { displayDate } from "$lib/date";
  import { getDictionary, languageLabels } from "$lib/dictionaries";
  import { getEditMode } from "$lib/admin-state.svelte";
  import { saveAndNavigate, saveJson, savePost } from "$lib/admin-save";
  import {
    generateBlogPostingJsonLd,
    generateBreadcrumbJsonLd,
  } from "$lib/json-ld";
  import { categoriesOf, type Section } from "$lib/content";
  import Icon from "$lib/components/Icon.svelte";
  import PostAdvertising from "$lib/components/PostAdvertising.svelte";
  import PostContent from "$lib/components/PostContent.svelte";
  import Comments from "$lib/components/Comments.svelte";
  import PrintedDivider from "$lib/components/PrintedDivider.svelte";
  import PrintedLabel from "$lib/components/PrintedLabel.svelte";
  import PrintedSection from "$lib/components/PrintedSection.svelte";
  import Seo from "$lib/components/Seo.svelte";
  import EditableText from "$lib/components/admin/EditableText.svelte";
  import EditableMarkdown from "$lib/components/admin/EditableMarkdown.svelte";

  let { data } = $props();

  let lang = $derived(data.lang);
  let post = $derived(data.post);
  let dictionary = $derived(getDictionary(lang));
  let baseUrl = $derived(dictionary.meta.baseUrl);
  let postUrl = $derived(`${baseUrl}${post.permalink}`);
  let admin = $derived(!!page.data.admin);
  let editing = $derived(admin && getEditMode());

  // 分类标签源：全部可选分类（编辑态原位切换归属），来自 content/categories/*.yml。
  let allCategories = $derived(categoriesOf(data.section as Section));

  async function saveTitle(v: string) {
    await savePost(post.path, { title: { [lang]: v } });
  }

  async function saveDate(v: string) {
    if (!v) return; // 清空日期会破坏 frontmatter 的 isodate 约束
    await savePost(post.path, { date: v });
  }

  async function saveDescription(v: string) {
    // 空串 = 删除可选的 description 字段（API 约定）
    await savePost(post.path, { description: { [lang]: v } });
  }

  /** slug 变更会改变文章 URL：保存后跳转新地址（velite 重建会整页刷新，由 saveAndNavigate 汇合）。 */
  async function saveSlug(v: string) {
    if (!v.trim()) return;
    await saveAndNavigate(`/${lang}/${data.section}/${v}`, () =>
      saveJson(
        `/studio/api/posts/${post.path}`,
        "PUT",
        { slug: v },
        { invalidate: false },
      ),
    );
  }

  async function toggleDraft() {
    await savePost(post.path, { draft: !post.draft });
  }

  async function toggleFeatured() {
    await savePost(post.path, { featured: !post.featured });
  }

  async function toggleCategory(slug: string) {
    const set = new Set(post.categories);
    if (set.has(slug)) set.delete(slug);
    else set.add(slug);
    await savePost(post.path, { categories: [...set] });
  }

  async function deletePost() {
    if (!confirm(`确定删除文章「${post.title}」？目录及其双语文件将被移除。`)) return;
    await saveAndNavigate(dictionary.urls.share, () =>
      saveJson(`/studio/api/posts/${post.path}`, "DELETE", null, {
        invalidate: false,
      }),
    );
  }

  /** 从 studio API 拉取 raw Markdown 正文（页面数据里只有渲染后的 HTML）。 */
  async function loadPostRaw(): Promise<string> {
    const res = await fetch(`/studio/api/posts/${post.path}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const raw = (await res.json()) as { body: Record<string, string> };
    return raw.body[lang] ?? "";
  }

  /**
   * 创建缺失的另一语言文件（同 slug 成对）：元数据沿用当前文章，正文为空
   * 由随后在该语言页面原位书写。velite 重建会整页刷新，跳转交由 saveAndNavigate。
   */
  async function createTranslation() {
    const other = lang === "en" ? "zh" : "en";
    await saveAndNavigate(`/${other}/${data.section}/${post.slug}`, () =>
      saveJson(
        `/studio/api/posts/${post.path}`,
        "PUT",
        {
          slug: post.slug,
          date: post.date,
          draft: post.draft,
          featured: post.featured,
          categories: post.categories,
          title: { [other]: post.title },
          body: { [other]: "" },
        },
        { invalidate: false },
      ),
    );
  }

  async function saveBody(rawMarkdown: string) {
    await savePost(post.path, { body: { [lang]: rawMarkdown } });
  }
</script>

<Seo
  {lang}
  title={post.title}
  description={post.description}
  keywords={post.keywords}
  path={post.permalink}
  ogType="article"
  article={{
    publishedTime: post.date,
    modifiedTime: post.updated,
    tags: post.categories,
  }}
  alternates={Object.fromEntries(
    [[lang, post.permalink], ...data.translations.map((t) => [t.lang, t.permalink])],
  )}
  jsonLd={[
    generateBlogPostingJsonLd({
      title: post.title,
      description: post.description,
      url: postUrl,
      datePublished: post.date,
      dateModified: post.updated,
      image: post.cover?.src ? `${baseUrl}${post.cover.src}` : undefined,
      categories: post.categories,
      lang,
    }),
    generateBreadcrumbJsonLd({
      items: [
        {
          name: dictionary.labels.home,
          url: `${baseUrl}${dictionary.urls.home}`,
        },
        {
          name: dictionary.labels[data.section],
          url: `${baseUrl}${dictionary.urls[data.section]}`,
        },
        { name: post.title, url: postUrl },
      ],
    }),
  ]}
/>

<div>
  <!-- Post header -->
  <PrintedSection>
    {#if editing}
      <!-- 元数据工具行：slug / 草稿 / 精选 / 删除，全部原位生效 -->
      <div
        class="mb-3 flex flex-wrap items-center gap-x-2.5 gap-y-1.5 rounded-sm border border-dashed border-printer-accent/40 bg-printer-accent/5 dark:bg-printer-accent-dark/10 px-2.5 py-1.5"
      >
        <span
          class="flex h-[23px] items-center font-mono text-[9px] uppercase leading-none tracking-wider text-printer-ink-light dark:text-printer-ink-dark/50"
        >
          slug
        </span>
        <div class="flex h-[23px] w-56 items-center">
          <EditableText
            editing
            value={post.slug}
            onsave={saveSlug}
            inputClass="font-mono text-[11px] leading-none"
          />
        </div>
        <button type="button" onclick={toggleDraft} title="切换草稿状态">
          <PrintedLabel variant={post.draft ? "accent" : "muted"}>
            {post.draft ? "草稿 DRAFT" : "已发布 LIVE"}
          </PrintedLabel>
        </button>
        <button type="button" onclick={toggleFeatured} title="切换精选状态">
          <PrintedLabel variant={post.featured ? "accent" : "muted"}>
            {post.featured ? "精选 FEATURED" : "普通"}
          </PrintedLabel>
        </button>
        <button
          type="button"
          onclick={deletePost}
          class="inline-flex h-[23px] items-center rounded-sm border border-red-500/30 px-2 font-mono text-[10px] leading-none uppercase tracking-wider text-red-500 hover:bg-red-500/10 dark:text-red-400"
        >
          删除
        </button>
      </div>
    {/if}

    <div class="flex flex-wrap gap-1.5 mb-3">
      {#if editing}
        {#each allCategories as category (category.slug)}
          <button type="button" onclick={() => toggleCategory(category.slug)}>
            <PrintedLabel variant={post.categories.includes(category.slug) ? "accent" : "muted"}>
              {category.name[lang]}
            </PrintedLabel>
          </button>
        {/each}
      {:else}
        {#each data.categories as category (category.slug)}
          <a href={category.permalink[lang]}>
            <PrintedLabel variant="accent">{category.name[lang]}</PrintedLabel>
          </a>
        {/each}
      {/if}
    </div>
    <h1
      class="font-serif text-2xl font-bold text-printer-ink dark:text-printer-ink-dark leading-tight"
    >
      <EditableText
        editing={editing}
        value={post.title}
        onsave={saveTitle}
        maxlength={99}
      />
    </h1>
    <div class="flex items-center gap-3 mt-2 flex-wrap">
      <span
        class="font-mono text-[10px] text-printer-ink-light dark:text-printer-ink-dark/40 tabular-nums"
      >
        <EditableText
          editing={editing}
          type="date"
          value={post.date.slice(0, 10)}
          display={displayDate(post.date, lang)}
          onsave={saveDate}
          inputClass="font-mono text-[10px] text-printer-ink dark:text-printer-ink-dark/60"
        />
      </span>
      {#if data.translations.length > 0}
        <div class="flex items-center gap-2">
          <span
            class="font-mono text-[10px] text-printer-ink-light dark:text-printer-ink-dark/40"
          >
            |
          </span>
          {#each data.translations as translation (translation.lang)}
            <a
              class="font-mono text-[10px] text-printer-accent dark:text-printer-accent-dark hover:underline"
              href={translation.permalink}
            >
              {languageLabels[translation.lang]}
            </a>
          {/each}
        </div>
      {:else if editing}
        <!-- 双语文章由同 slug 的 en.md/zh.md 组成；缺另一语言时可就地创建 -->
        <button
          type="button"
          onclick={createTranslation}
          class="rounded-sm border border-dashed border-printer-accent/50 px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider text-printer-accent hover:bg-printer-accent/10 dark:border-printer-accent-dark/50 dark:text-printer-accent-dark"
        >
          ＋ 创建{lang === "en" ? "中文版" : "English 版"}
        </button>
      {/if}
    </div>
    {#if post.description || editing}
      <p
        class="font-serif text-xs text-printer-ink-light dark:text-printer-ink-dark/50 mt-2 leading-relaxed"
      >
        <EditableText
          editing={editing}
          value={post.description ?? ""}
          onsave={saveDescription}
          maxlength={999}
          placeholder="描述（可选）"
        />
      </p>
    {/if}
  </PrintedSection>

  <PrintedDivider style="solid" />

  <!-- Post content -->
  <EditableMarkdown
    editing={editing}
    sourceKey={post.path}
    loadRaw={loadPostRaw}
    onsave={saveBody}
  >
    <PostContent html={post.content} />
  </EditableMarkdown>

  <!-- WeChat QR Code -->
  {#if data.wechatQrSvg}
    <PrintedDivider style="dashed" />
    <div class="flex items-center gap-4 py-1">
      <div class="shrink-0 w-20 h-20 bg-white p-1.5">
        <!-- eslint-disable-next-line svelte/no-at-html-tags -- build-time QR SVG -->
        {@html data.wechatQrSvg}
      </div>
      <div>
        <p class="font-mono text-[11px] text-printer-ink dark:text-printer-ink-dark">
          WeChat
        </p>
        <p
          class="font-mono text-[10px] text-printer-ink-light dark:text-printer-ink-dark/40 mt-0.5"
        >
          {dictionary.labels.wechatScanHint}
        </p>
      </div>
    </div>
  {/if}

  <PrintedDivider style="dashed" />

  <!-- Footer actions -->
  <div class="flex items-center gap-4">
    <span
      class="font-mono text-[10px] text-printer-ink-light dark:text-printer-ink-dark/40 uppercase tracking-wider"
    >
      {dictionary.labels.shareTo}
    </span>
    <a
      href={dictionary.urls.shareToX(post.title, post.permalink)}
      target="_blank"
      rel="noopener"
      class="text-printer-ink-light dark:text-printer-ink-dark/40 hover:text-printer-ink dark:hover:text-printer-ink-dark transition-colors"
      aria-label="Share to X"
    >
      <Icon name="x" class="w-4 h-4" />
    </a>
  </div>

  <div class="mt-6">
    <a
      href={dictionary.urls[data.section]}
      class="inline-flex items-center gap-1 font-mono text-[11px] tracking-wider text-printer-accent dark:text-printer-accent-dark hover:underline"
    >
      {dictionary.labels.backToSection[data.section]}
    </a>
  </div>

  <Comments {lang} thread={post.permalink} />

  <PrintedDivider style="dotted" />

  <!-- Advertising -->
  <PrintedSection>
    <PostAdvertising advertisements={dictionary.postAdvertisements} />
  </PrintedSection>
</div>
