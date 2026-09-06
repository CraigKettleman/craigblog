<script lang="ts">
  import { page } from "$app/state";
  import { getDictionary, type Language } from "$lib/dictionaries";
  import { siteMeta } from "$lib/site";
  import Icon from "$lib/components/Icon.svelte";
  import PostList from "$lib/components/PostList.svelte";
  import PrintedDivider from "$lib/components/PrintedDivider.svelte";
  import PrintedLabel from "$lib/components/PrintedLabel.svelte";
  import PrintedPageTitle from "$lib/components/PrintedPageTitle.svelte";
  import PrintedSection from "$lib/components/PrintedSection.svelte";
  import Seo from "$lib/components/Seo.svelte";

  let { data } = $props();

  let lang = $derived(page.params.lang as Language);
  let dictionary = $derived(getDictionary(lang));
  let site = $derived(siteMeta(lang));
  let feedHref = $derived(lang === "zh" ? "/feed/zh" : "/feed");
</script>

<Seo
  {lang}
  title="{dictionary.labels.share} - {site.websiteName}"
  description={dictionary.labels.share}
  path={dictionary.urls.share}
/>

<div>
  <!-- Header -->
  <PrintedSection>
    <PrintedPageTitle icon="send">{dictionary.labels.share}</PrintedPageTitle>
    <p
      class="font-serif text-xs text-printer-ink-light dark:text-printer-ink-dark/50"
    >
      {lang === "zh"
        ? "在这里分享我收集的链接、发现与想法。"
        : "Links, finds and notes I want to share."}
    </p>
  </PrintedSection>

  <PrintedDivider style="solid" />

  <!-- Categories as label strips -->
  <PrintedSection label={dictionary.labels.categories} labelIcon="tag">
    <div class="flex flex-wrap gap-1.5 mb-2">
      {#each data.categories as category (category.slug)}
        <a href={category.permalink[lang]}>
          <PrintedLabel variant="default">
            {category.name[lang]}
            <span class="opacity-50">({category.count[lang]})</span>
          </PrintedLabel>
        </a>
      {/each}
    </div>
  </PrintedSection>

  <PrintedDivider style="dashed" />

  <!-- Post list -->
  <PostList posts={data.posts} {lang} />

  <PrintedDivider style="dashed" />

  <!-- Follow -->
  <PrintedSection label={lang === "zh" ? "关注我" : "Follow"} labelIcon="user">
    <div class="flex flex-wrap gap-2">
      {#each dictionary.contacts as contact (contact.link)}
        <a
          href={contact.link}
          target="_blank"
          rel="noopener"
          class="inline-flex items-center gap-1.5 font-mono text-[10px] tracking-wider uppercase px-2.5 py-1 rounded-sm border border-printer-ink/8 dark:border-printer-ink-dark/8 text-printer-ink-light dark:text-printer-ink-dark/50 hover:text-printer-accent dark:hover:text-printer-accent-dark hover:border-printer-accent/20 dark:hover:border-printer-accent-dark/20 transition-colors"
        >
          <Icon name={contact.icon} class="w-3 h-3" />
          {contact.label}
        </a>
      {/each}
    </div>
  </PrintedSection>

  <PrintedDivider style="dashed" />

  <!-- RSS -->
  <PrintedSection label={lang === "zh" ? "订阅" : "Subscribe"} labelIcon="pulse">
    <a
      href={feedHref}
      class="inline-flex items-center gap-1.5 font-mono text-[10px] tracking-wider uppercase px-2.5 py-1 rounded-sm border border-printer-ink/8 dark:border-printer-ink-dark/8 text-printer-ink-light dark:text-printer-ink-dark/50 hover:text-printer-accent dark:hover:text-printer-accent-dark hover:border-printer-accent/20 dark:hover:border-printer-accent-dark/20 transition-colors"
    >
      RSS
    </a>
    <p
      class="font-serif text-xs text-printer-ink-light dark:text-printer-ink-dark/50 mt-2"
    >
      {lang === "zh"
        ? "用 RSS 阅读器订阅我的更新。"
        : "Subscribe to my updates with an RSS reader."}
    </p>
  </PrintedSection>
</div>
