<script lang="ts">
  import { page } from "$app/state";
  import { getDictionary, type Language } from "$lib/dictionaries";
  import { getEditMode } from "$lib/admin-state.svelte";
  import { saveSite } from "$lib/admin-save";
  import { siteMeta } from "$lib/site";
  import PostContent from "$lib/components/PostContent.svelte";
  import PrintedDivider from "$lib/components/PrintedDivider.svelte";
  import PrintedPageTitle from "$lib/components/PrintedPageTitle.svelte";
  import PrintedSection from "$lib/components/PrintedSection.svelte";
  import Comments from "$lib/components/Comments.svelte";
  import Seo from "$lib/components/Seo.svelte";
  import EditableMarkdown from "$lib/components/admin/EditableMarkdown.svelte";

  let lang = $derived(page.params.lang as Language);
  let dictionary = $derived(getDictionary(lang));
  let site = $derived(siteMeta(lang));
  let subtitle = $derived(dictionary.labels.aboutSubtitle.trim());
  let admin = $derived(!!page.data.admin);
  let editing = $derived(admin && getEditMode());

  // 个人简介以 site.yml 为准（可在「可编辑模式」下修改），无则回退到字典默认值。
  let aboutSource = $derived(site.about.trim() || dictionary.aboutContent);
  const aboutForEdit = $derived(site.about.trim());

  // Convert the simple markdown-ish about text to HTML.
  let aboutHtml = $derived(
    aboutSource
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

  async function saveAbout(raw: string) {
    await saveSite({ about: { [lang]: raw } });
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
    <PrintedPageTitle icon="user">{dictionary.labels.aboutTitle}</PrintedPageTitle>
    {#if subtitle}
      <p class="font-serif text-xs text-printer-ink-light dark:text-printer-ink-dark/50">
        {subtitle}
      </p>
    {/if}
  </PrintedSection>

  <PrintedDivider style="solid" />

  <!-- About content -->
  <EditableMarkdown
    editing={editing}
    sourceKey="about"
    loadRaw={() => Promise.resolve(aboutForEdit)}
    onsave={saveAbout}
    label="简介 · MARKDOWN"
  >
    <PostContent html={aboutHtml} />
  </EditableMarkdown>

  <PrintedDivider style="dashed" />

  <Comments {lang} thread={`/${lang}/about`} />
</div>
