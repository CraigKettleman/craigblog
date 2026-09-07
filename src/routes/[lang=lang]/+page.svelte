<script lang="ts">
  import { onMount } from "svelte";
  import { page } from "$app/state";
  import { getDictionary, type Language } from "$lib/dictionaries";
  import { siteMeta } from "$lib/site";
  import { getEditMode } from "$lib/admin-state.svelte";
  import { saveSite } from "$lib/admin-save";
  import { generateWebSiteJsonLd } from "$lib/json-ld";
  import Icon from "$lib/components/Icon.svelte";
  import PostList from "$lib/components/PostList.svelte";
  import PrintedDivider from "$lib/components/PrintedDivider.svelte";
  import PrintedLabel from "$lib/components/PrintedLabel.svelte";
  import PrintedSection from "$lib/components/PrintedSection.svelte";
  import Seo from "$lib/components/Seo.svelte";
  import SocialHoverCard from "$lib/components/SocialHoverCard.svelte";
  import EditableText from "$lib/components/admin/EditableText.svelte";

  let { data } = $props();

  let lang = $derived(data.lang);
  let dictionary = $derived(getDictionary(lang));
  let site = $derived(siteMeta(lang));
  let admin = $derived(!!page.data.admin);
  let editing = $derived(admin && getEditMode());

  // 编辑态：座右铭以完整列表呈现（展示态随机展示一条），就地增删改。
  let mottoDrafts = $state<string[]>([]);
  $effect(() => {
    const list = site.mottos.length > 0 ? site.mottos : site.motto ? [site.motto] : [];
    mottoDrafts = [...list];
  });

  async function saveWebsiteName(v: string) {
    await saveSite({ websiteName: { [lang]: v } });
  }

  function addMotto() {
    mottoDrafts = [...mottoDrafts, ""];
  }

  function removeMotto(index: number) {
    const next = mottoDrafts.filter((_, i) => i !== index);
    mottoDrafts = next;
    saveSite({ mottos: { [lang]: next.filter((s) => s.trim() !== "") } }).catch(() => {});
  }

  async function saveMotto(index: number, v: string) {
    const next = [...mottoDrafts];
    next[index] = v;
    mottoDrafts = next;
    await saveSite({ mottos: { [lang]: next.filter((s) => s.trim() !== "") } });
  }

  // Prerendered pages show the first motto; rotate randomly per visit.
  let mottoIndex = $state(0);
  onMount(() => {
    mottoIndex = Math.floor(Math.random() * site.mottos.length);
  });
  let motto = $derived(site.mottos[mottoIndex] ?? site.motto);

  let primaryWorks = $derived(
    dictionary.works.filter((work) => work.primary).slice(0, 4),
  );

  const printedOn = new Date().toISOString().split("T")[0];
</script>

<Seo
  {lang}
  title={site.websiteName}
  description={site.motto}
  path={dictionary.urls.home}
  feeds
  jsonLd={[
    generateWebSiteJsonLd({
      name: site.websiteName,
      alternateName: "Craig",
      url: `${dictionary.meta.baseUrl}${dictionary.urls.home}`,
      description: site.motto,
    }),
  ]}
/>

<div>
  <!-- Profile header - printed label style -->
  <PrintedSection>
    <div class="flex items-start gap-4 mb-2">
      <div class="flex-1">
        <h1
          class="font-serif text-2xl font-bold tracking-tight text-printer-ink dark:text-printer-ink-dark"
        >
          <EditableText editing={editing} value={site.websiteName} onsave={saveWebsiteName} />
        </h1>
        {#if editing}
          <div class="mt-1.5 space-y-1">
            {#each mottoDrafts as m, i (i)}
              <div class="flex items-baseline gap-1.5">
                <p
                  class="flex-1 font-serif text-xs sm:text-[13px] text-printer-ink-light dark:text-printer-ink-dark/60 leading-relaxed"
                >
                  <EditableText editing value={m} onsave={(v) => saveMotto(i, v)} />
                </p>
                <button
                  type="button"
                  onclick={() => removeMotto(i)}
                  class="font-mono text-[11px] text-printer-ink-light/50 hover:text-red-500 dark:text-printer-ink-dark/40"
                  aria-label="删除这条座右铭"
                >
                  ×
                </button>
              </div>
            {/each}
            <button
              type="button"
              onclick={addMotto}
              class="font-mono text-[10px] tracking-wider text-printer-accent dark:text-printer-accent-dark hover:underline"
            >
              + 添加一条
            </button>
          </div>
        {:else}
          <p
            class="font-serif text-xs sm:text-[13px] text-printer-ink-light dark:text-printer-ink-dark/60 mt-1 leading-relaxed"
          >
            {motto}
          </p>
        {/if}
      </div>
    </div>

    <!-- Contact strip -->
    <div class="flex flex-wrap gap-2 mt-4">
      {#each dictionary.contacts as contact (contact.link)}
        {@const kind =
          contact.icon === "x"
            ? ("x" as const)
            : contact.icon === "github"
              ? ("github" as const)
              : contact.icon === "mail"
                ? ("email" as const)
                : undefined}
        {@const linkClass =
          "inline-flex items-center gap-1.5 font-mono text-[10px] tracking-wider uppercase px-2.5 py-1 rounded-sm border border-printer-ink/8 dark:border-printer-ink-dark/8 text-printer-ink-light dark:text-printer-ink-dark/50 hover:text-printer-accent dark:hover:text-printer-accent-dark hover:border-printer-accent/20 dark:hover:border-printer-accent-dark/20 transition-colors"}
        {#if kind}
          <SocialHoverCard
            {kind}
            href={contact.link}
            {lang}
            {dictionary}
            align="left"
            side="bottom"
            class={linkClass}
          >
            <Icon name={contact.icon} class="w-3 h-3" />
            {contact.label}
          </SocialHoverCard>
        {:else}
          <a
            href={contact.link}
            target="_blank"
            rel="noopener"
            class={linkClass}
          >
            <Icon name={contact.icon} class="w-3 h-3" />
            {contact.label}
          </a>
        {/if}
      {/each}
    </div>
  </PrintedSection>

  <PrintedDivider style="dotted" />

  <!-- Works section -->
  <PrintedSection label={dictionary.labels.projects} labelIcon="apps">
    <div class="grid grid-cols-2 gap-2">
      {#each primaryWorks as work (work.name)}
        <a
          href={work.link}
          target="_blank"
          rel="noopener"
          class="group flex items-center gap-3 py-3 -mx-2 px-2 rounded-md hover:bg-printer-ink/3 dark:hover:bg-printer-ink-dark/3 transition-colors"
        >
          {#if work.image}
            <img
              class="h-10 w-10 rounded-lg border border-printer-ink/10 dark:border-printer-ink-dark/10 shrink-0"
              src={work.image}
              alt={dictionary.labels.icon(work.name)}
              width="40"
              height="40"
              loading="lazy"
            />
          {:else}
            <div
              class="h-10 w-10 rounded-lg bg-printer-accent/10 dark:bg-printer-accent-dark/10 flex items-center justify-center font-mono text-lg font-bold text-printer-accent dark:text-printer-accent-dark shrink-0"
            >
              {work.name[0]}
            </div>
          {/if}
          <div class="min-w-0 flex-1">
            <div
              class="font-mono text-sm font-medium text-printer-ink dark:text-printer-ink-dark group-hover:text-printer-accent dark:group-hover:text-printer-accent-dark transition-colors"
            >
              {work.name}
            </div>
            <div
              class="font-mono text-[10px] text-printer-ink-light dark:text-printer-ink-dark/40 mt-0.5 line-clamp-1"
            >
              {work.summary}
            </div>
          </div>
          <span
            class="font-mono text-[10px] text-printer-ink-light dark:text-printer-ink-dark/30 group-hover:text-printer-accent dark:group-hover:text-printer-accent-dark transition-colors shrink-0"
          >
            →
          </span>
        </a>
      {/each}
    </div>
    <a
      href={dictionary.urls.projects}
      class="inline-flex items-center gap-1 font-mono text-[11px] tracking-wider text-printer-accent dark:text-printer-accent-dark mt-3 hover:underline"
    >
      ◦ {dictionary.labels.viewAll} →
    </a>
  </PrintedSection>

  <PrintedDivider style="dashed" />

  <!-- Latest Tech Posts -->
  <PrintedSection label={dictionary.labels.latestTech} labelIcon="window">
    <PostList posts={data.latestTech} {lang} compact editable={editing} />
    <a
      href={dictionary.urls.posts}
      class="inline-flex items-center gap-1 font-mono text-[11px] tracking-wider text-printer-accent dark:text-printer-accent-dark mt-3 hover:underline"
    >
      ▸ {dictionary.labels.viewAll} →
    </a>
  </PrintedSection>

  <!-- Footer stamp -->
  <div
    class="mt-8 pt-4 border-t border-dotted border-printer-ink/10 dark:border-printer-ink-dark/10"
  >
    <div class="flex items-center justify-between">
      <div
        class="font-mono text-[9px] text-printer-ink-light dark:text-printer-ink-dark/30 tracking-wider uppercase"
      >
        {dictionary.labels.printedOn}
        {printedOn}
      </div>
      <PrintedLabel variant="muted">hhy.homes</PrintedLabel>
    </div>
  </div>
</div>
