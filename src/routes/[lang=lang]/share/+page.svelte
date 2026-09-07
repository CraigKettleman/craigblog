<script lang="ts">
  import { page } from "$app/state";
  import { getDictionary, type Language } from "$lib/dictionaries";
  import { getEditMode } from "$lib/admin-state.svelte";
  import { saveSite } from "$lib/admin-save";
  import { siteMeta } from "$lib/site";
  import Icon from "$lib/components/Icon.svelte";
  import PostList from "$lib/components/PostList.svelte";
  import PrintedDivider from "$lib/components/PrintedDivider.svelte";
  import PrintedLabel from "$lib/components/PrintedLabel.svelte";
  import PrintedPageTitle from "$lib/components/PrintedPageTitle.svelte";
  import PrintedSection from "$lib/components/PrintedSection.svelte";
  import Seo from "$lib/components/Seo.svelte";
  import CategoryAdmin from "$lib/components/admin/CategoryAdmin.svelte";
  import EditableText from "$lib/components/admin/EditableText.svelte";

  let { data } = $props();

  let lang = $derived(page.params.lang as Language);
  let dictionary = $derived(getDictionary(lang));
  let site = $derived(siteMeta(lang));
  let feedHref = $derived(lang === "zh" ? "/feed/zh" : "/feed");
  let admin = $derived(!!page.data.admin);
  let editing = $derived(admin && getEditMode());

  // 分享页文案：site.yml 可编辑字段，缺失时回退到页面原有硬编码文案
  let subtitle = $derived(site.shareSubtitle || (lang === "zh"
    ? "在这里分享我收集的链接、发现与想法。"
    : "Links, finds and notes I want to share."));
  let subscribeHint = $derived(site.shareSubscribeHint || (lang === "zh"
    ? "用 RSS 阅读器订阅我的更新。"
    : "Subscribe to my updates with an RSS reader."));

  async function saveShareField(key: "shareSubtitle" | "shareSubscribeHint", v: string) {
    await saveSite({ [key]: { [lang]: v } });
  }
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
      <EditableText
        editing={editing}
        value={subtitle}
        onsave={(v) => saveShareField("shareSubtitle", v)}
        maxlength={200}
      />
    </p>
  </PrintedSection>

  <PrintedDivider style="solid" />

  <!-- Categories as label strips -->
  <PrintedSection label={dictionary.labels.categories} labelIcon="tag">
    {#if editing}
      <CategoryAdmin {lang} section="posts" categories={data.categories} />
    {:else}
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
    {/if}
  </PrintedSection>

  <PrintedDivider style="dashed" />

  <!-- Post list -->
  {#if editing}
    <div class="mb-3 flex justify-end">
      <a
        href="/studio/submit"
        class="rounded-sm border border-printer-accent/50 bg-printer-accent/5 dark:bg-printer-accent-dark/10 px-2.5 py-1 font-mono text-[10px] uppercase tracking-wider text-printer-accent hover:bg-printer-accent/10 dark:border-printer-accent-dark/50 dark:text-printer-accent-dark dark:hover:bg-printer-accent-dark/20"
      >
        ＋ 新文章
      </a>
    </div>
  {/if}
  <PostList posts={data.posts} {lang} editable={editing} />

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
      <EditableText
        editing={editing}
        value={subscribeHint}
        onsave={(v) => saveShareField("shareSubscribeHint", v)}
        maxlength={200}
      />
    </p>
  </PrintedSection>
</div>
