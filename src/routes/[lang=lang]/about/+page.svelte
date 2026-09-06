<script lang="ts">
  import { browser } from "$app/environment";
  import { invalidateAll } from "$app/navigation";
  import { page } from "$app/state";
  import { getDictionary, type Language } from "$lib/dictionaries";
  import { getEditMode } from "$lib/admin-state.svelte";
  import { siteMeta } from "$lib/site";
  import PostContent from "$lib/components/PostContent.svelte";
  import PrintedDivider from "$lib/components/PrintedDivider.svelte";
  import PrintedPageTitle from "$lib/components/PrintedPageTitle.svelte";
  import PrintedSection from "$lib/components/PrintedSection.svelte";
  import Comments from "$lib/components/Comments.svelte";
  import Seo from "$lib/components/Seo.svelte";
  import SiteSettingsEditor from "$lib/components/admin/SiteSettingsEditor.svelte";

  let lang = $derived(page.params.lang as Language);
  let dictionary = $derived(getDictionary(lang));
  let site = $derived(siteMeta(lang));
  let subtitle = $derived(dictionary.labels.aboutSubtitle.trim());
  let admin = $derived(!!page.data.admin);
  let editing = $derived(admin && getEditMode());

  // 个人简介以 site.yml 为准（可在「可编辑模式」下修改），无则回退到字典默认值。
  let aboutSource = $derived(site.about.trim());

  // Convert the simple markdown-ish about text to HTML.
  let aboutHtml = $derived(
    (aboutSource || dictionary.aboutContent)
      .trim()
      .replace(/### (.+)/g, "<h3>$1</h3>")
      .replace(/## (.+)/g, "<h2>$1</h2>")
      .replace(/# (.+)/g, "<h1>$1</h1>")
      .replace(
        /\[([^\]]+)\]\(([^)]+)\)/g,
        '<a href="$2" target="_blank" rel="noopener">$1</a>',
      )
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\n\n/g, "</p><p>")
      .replace(/^/, "<p>")
      .replace(/$/, "</p>")
      .replace(/<p><h([123])>/g, "<h$1>")
      .replace(/<\/h([123])><\/p>/g, "</h$1>")
      .replace(/<p><\/p>/g, ""),
  );

  let editOpen = $state(false);

  async function onSaved() {
    editOpen = false;
    await new Promise((r) => setTimeout(r, 400));
    await invalidateAll();
  }
</script>

<Seo
  {lang}
  title="{dictionary.labels.aboutTitle} - {site.websiteName}"
  description={subtitle || dictionary.labels.aboutTitle}
  path={dictionary.urls.about}
/>

<div>
  <!-- Header -->
  <PrintedSection>
    {#if editing}
      <div class="mb-3 flex justify-end">
        <button
          onclick={() => (editOpen = true)}
          class="rounded-sm border border-printer-accent/50 px-2.5 py-1 font-mono text-[10px] uppercase tracking-wider text-printer-accent hover:bg-printer-accent/10 dark:border-printer-accent-dark/50 dark:text-printer-accent-dark dark:hover:bg-printer-accent-dark/10"
        >
          ✎ 编辑个人简介
        </button>
      </div>
    {/if}

    <PrintedPageTitle icon="user">{dictionary.labels.aboutTitle}</PrintedPageTitle>
    {#if subtitle}
      <p class="font-serif text-xs text-printer-ink-light dark:text-printer-ink-dark/50">
        {subtitle}
      </p>
    {/if}
  </PrintedSection>

  <PrintedDivider style="solid" />

  <!-- About content -->
  <PostContent html={aboutHtml} />

  <PrintedDivider style="dashed" />

  <Comments {lang} thread={`/${lang}/about`} />
</div>

{#if editing && editOpen && browser}
  <SiteSettingsEditor {lang} onclose={() => (editOpen = false)} onSaved={onSaved} />
{/if}
