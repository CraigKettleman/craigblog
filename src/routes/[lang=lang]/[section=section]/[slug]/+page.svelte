<script lang="ts">
  import { browser } from "$app/environment";
  import { invalidateAll } from "$app/navigation";
  import { page } from "$app/state";
  import { displayDate } from "$lib/date";
  import { getDictionary, languageLabels } from "$lib/dictionaries";
  import { getEditMode } from "$lib/admin-state.svelte";
  import {
    generateBlogPostingJsonLd,
    generateBreadcrumbJsonLd,
  } from "$lib/json-ld";
  import Icon from "$lib/components/Icon.svelte";
  import PostAdvertising from "$lib/components/PostAdvertising.svelte";
  import PostContent from "$lib/components/PostContent.svelte";
  import Comments from "$lib/components/Comments.svelte";
  import PrintedDivider from "$lib/components/PrintedDivider.svelte";
  import PrintedLabel from "$lib/components/PrintedLabel.svelte";
  import PrintedSection from "$lib/components/PrintedSection.svelte";
  import Seo from "$lib/components/Seo.svelte";
  import PostEditor from "$lib/components/admin/PostEditor.svelte";

  let { data } = $props();

  let lang = $derived(data.lang);
  let post = $derived(data.post);
  let dictionary = $derived(getDictionary(lang));
  let baseUrl = $derived(dictionary.meta.baseUrl);
  let postUrl = $derived(`${baseUrl}${post.permalink}`);
  let admin = $derived(!!page.data.admin);
  let editing = $derived(admin && getEditMode());

  // 编辑器抽屉
  let editOpen = $state(false);

  function openEditor() {
    editOpen = true;
  }

  async function onSaved() {
    // 等 velite watch 重建（实测约 50ms）后重跑 load，拿到新渲染数据
    await new Promise((r) => setTimeout(r, 400));
    await invalidateAll();
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
      <div class="mb-3 flex justify-end">
        <button
          onclick={openEditor}
          class="rounded-sm border border-printer-accent/50 px-2.5 py-1 font-mono text-[10px] uppercase tracking-wider text-printer-accent hover:bg-printer-accent/10 dark:border-printer-accent-dark/50 dark:text-printer-accent-dark dark:hover:bg-printer-accent-dark/10"
        >
          ✎ 编辑文章
        </button>
      </div>
    {/if}

    <div class="flex flex-wrap gap-1.5 mb-3">
      {#each data.categories as category (category.slug)}
        <a href={category.permalink[lang]}>
          <PrintedLabel variant="accent">{category.name[lang]}</PrintedLabel>
        </a>
      {/each}
    </div>
    <h1
      class="font-serif text-2xl font-bold text-printer-ink dark:text-printer-ink-dark leading-tight"
    >
      {post.title}
    </h1>
    <div class="flex items-center gap-3 mt-2 flex-wrap">
      <span
        class="font-mono text-[10px] text-printer-ink-light dark:text-printer-ink-dark/40 tabular-nums"
      >
        {displayDate(post.date, lang)}
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
      {/if}
    </div>
    {#if post.description}
      <p
        class="font-serif text-xs text-printer-ink-light dark:text-printer-ink-dark/50 mt-2 leading-relaxed"
      >
        {post.description}
      </p>
    {/if}
  </PrintedSection>

  <PrintedDivider style="solid" />

  <!-- Post content -->
  <PostContent html={post.content} />

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

{#if editing && editOpen && browser}
  <PostEditor
    {lang}
    postPath={post.path}
    onclose={() => (editOpen = false)}
    onSaved={onSaved}
  />
{/if}
