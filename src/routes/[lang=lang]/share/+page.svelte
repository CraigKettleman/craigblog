<script lang="ts">
  import { page } from "$app/state";
  import { getDictionary, type Language } from "$lib/dictionaries";
  import Icon from "$lib/components/Icon.svelte";
  import PrintedDivider from "$lib/components/PrintedDivider.svelte";
  import PrintedPageTitle from "$lib/components/PrintedPageTitle.svelte";
  import PrintedSection from "$lib/components/PrintedSection.svelte";
  import Seo from "$lib/components/Seo.svelte";

  let lang = $derived(page.params.lang as Language);
  let dictionary = $derived(getDictionary(lang));
  let feedHref = $derived(lang === "zh" ? "/feed/zh" : "/feed");
</script>

<Seo
  {lang}
  title="{dictionary.labels.share} - {dictionary.meta.websiteName}"
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

  <PrintedDivider style="dashed" />

  <!-- Footer -->
  <div
    class="font-mono text-[9px] text-printer-ink-light dark:text-printer-ink-dark/30 tracking-wider uppercase text-center py-4"
  >
    {lang === "zh" ? "分享内容筹备中～" : "More shares coming soon~"}
  </div>
</div>
