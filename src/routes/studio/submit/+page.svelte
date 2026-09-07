<script lang="ts">
  import { browser } from "$app/environment";
  import { page } from "$app/state";
  import { getDictionary, isLanguage } from "$lib/dictionaries";
  import { parseFrontMatter } from "$lib/frontmatter";
  import { saveAndNavigate } from "$lib/admin-save";
  import PrintedDivider from "$lib/components/PrintedDivider.svelte";
  import PrintedLabel from "$lib/components/PrintedLabel.svelte";
  import PrintedPageTitle from "$lib/components/PrintedPageTitle.svelte";
  import PrintedSection from "$lib/components/PrintedSection.svelte";

  interface CategoryMeta {
    slug: string;
    section: "posts" | "projects";
    name: { en: string; zh: string };
    description?: { en: string; zh: string };
  }

  type ContentLang = "en" | "zh";
  const LANGS: ContentLang[] = ["en", "zh"];

  /** 界面文案随外壳语言（?lang=）分离，中文/英文各一套。 */
  const T = {
    zh: {
      title: "投稿",
      subtitle: "书写或导入 Markdown，发布到所选栏目。",
      titleField: "标题 · TITLE",
      titlePlaceholder: "标题",
      descField: "描述 · DESCRIPTION（可选）",
      importField: "导入 · .MD 文件（自动解析 frontmatter）",
      bodyField: "正文 · MARKDOWN",
      bodyPlaceholder: "在此书写，或从上方导入 .md 文件",
      imported: (name: string) => `已导入 ${name}`,
      metaLabel: "元数据 · META",
      sectionLabel: "栏目 · SECTION",
      share: "分享",
      projects: "项目",
      slugField: "Slug（留空则自动从标题生成）",
      dateField: "日期 · DATE",
      draftLabel: "草稿（不发布到公共版）",
      featuredLabel: "精选",
      categoriesLabel: "分类（可多选）",
      noCategories: "该栏目暂无分类",
      publish: "✓ 发布",
      publishing: "发布中...",
      statusWriting: "正在发布...",
      statusDone: "✓ 发布成功，正在打开新文章…",
      statusNeed: "请先书写或导入至少一个语言的正文。",
      statusFail: "发布失败",
    },
    en: {
      title: "Submit",
      subtitle: "Write or import Markdown, publish to a section.",
      titleField: "TITLE",
      titlePlaceholder: "Title",
      descField: "DESCRIPTION (optional)",
      importField: "IMPORT · .MD FILE (parses frontmatter)",
      bodyField: "BODY · MARKDOWN",
      bodyPlaceholder: "Write here, or import a .md file above",
      imported: (name: string) => `Imported ${name}`,
      metaLabel: "META",
      sectionLabel: "SECTION",
      share: "SHARE",
      projects: "PROJECTS",
      slugField: "SLUG (auto-generated from title if empty)",
      dateField: "DATE",
      draftLabel: "DRAFT (hidden from public)",
      featuredLabel: "FEATURED",
      categoriesLabel: "CATEGORIES (multi-select)",
      noCategories: "No categories in this section yet",
      publish: "✓ PUBLISH",
      publishing: "PUBLISHING...",
      statusWriting: "Publishing...",
      statusDone: "✓ Published. Opening the new post…",
      statusNeed: "Write or import at least one language's body first.",
      statusFail: "Publish failed",
    },
  } as const;

  interface LangDraft {
    title: string;
    description: string;
    /** raw Markdown 正文（可来自 .md 文件导入或直接书写） */
    text: string;
    fileName: string;
  }

  // 界面语言由根布局根据 ?lang= 推断（submit/+page.server.ts 透传）
  let lang = $derived.by(() => {
    const fromData = (page.data as { lang?: unknown }).lang;
    return typeof fromData === "string" && isLanguage(fromData) ? fromData : "en";
  });
  let dictionary = $derived(getDictionary(lang));
  let t = $derived(T[lang]);

  let tab = $state<ContentLang>("zh");
  let drafts = $state<Record<ContentLang, LangDraft>>({
    en: { title: "", description: "", text: "", fileName: "" },
    zh: { title: "", description: "", text: "", fileName: "" },
  });

  // 共享元数据（两种语言一份）；栏目由入口链接指定（如项目页「＋ 新文章」），
  // 在挂载后从 URL 读取，避免 SSR 与客户端初始值不一致导致水合错位
  let section = $state<"posts" | "projects">("posts");
  let slug = $state("");

  $effect(() => {
    if (
      browser &&
      new URLSearchParams(window.location.search).get("section") === "projects"
    ) {
      section = "projects";
    }
  });
  let date = $state(new Date().toISOString().slice(0, 10));
  let draft = $state(true);
  let featured = $state(false);
  let categories = $state<string[]>([]);
  let keywords = $state<string[]>([]);

  let allCategories = $state<CategoryMeta[]>([]);
  let submitting = $state(false);
  let status = $state<{ kind: "info" | "ok" | "error"; text: string } | null>(null);

  $effect(() => {
    if (browser) loadCategories();
  });

  async function loadCategories() {
    try {
      const res = await fetch("/studio/api/categories");
      if (res.ok) allCategories = (await res.json()) as CategoryMeta[];
    } catch {
      /* 分类加载失败时仅影响 chip 列表，输入框仍可发布 */
    }
  }

  const sectionCategories = $derived(
    allCategories.filter((c) => c.section === section),
  );

  function onFileInput(e: Event, which: ContentLang) {
    const input = e.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const text = String(reader.result ?? "");
      drafts[which].fileName = file.name;
      drafts[which].text = text;
      applyFrontmatter(text, which);
    };
    reader.readAsText(file);
  }

  /** 从导入的 Markdown frontmatter 反填表单（已有输入不覆盖）。 */
  function applyFrontmatter(text: string, which: ContentLang) {
    const { data } = parseFrontMatter(text);
    if (typeof data.title === "string") drafts[which].title = data.title;
    if (typeof data.description === "string" && !drafts[which].description) {
      drafts[which].description = data.description;
    }
    if (!slug && typeof data.slug === "string") slug = data.slug;
    if (!date.match(/^\d{4}-/) && typeof data.date === "string") {
      date = data.date.slice(0, 10);
    }
    if (Array.isArray(data.categories) && categories.length === 0) {
      categories = data.categories.map(String);
    }
    if (Array.isArray(data.keywords) && keywords.length === 0) {
      keywords = data.keywords.map(String);
    }
  }

  function toggleCategory(categorySlug: string) {
    const set = new Set(categories);
    if (set.has(categorySlug)) set.delete(categorySlug);
    else set.add(categorySlug);
    categories = [...set];
  }

  function deriveSlug(): string {
    if (slug) return slug;
    const base = (drafts.en.title || drafts.zh.title || "untitled")
      .toLowerCase()
      .replace(/[^a-z0-9\u4e00-\u9fff]+/g, "-")
      .replace(/^-|-$/g, "");
    return base || "untitled";
  }

  async function submit() {
    if (submitting) return;
    const langs = LANGS.filter((l) => drafts[l].text.trim() !== "");
    if (langs.length === 0) {
      status = { kind: "info", text: t.statusNeed };
      return;
    }
    submitting = true;
    status = { kind: "info", text: t.statusWriting };
    try {
      const finalSlug = deriveSlug();
      // velite 重建会整页刷新并杀掉后续跳转，统一交由 saveAndNavigate 登记/汇合
      await saveAndNavigate(
        `/${langs.includes(tab) ? tab : langs[0]}/${section}/${finalSlug}`,
        async () => {
          for (const l of langs) {
            const other: ContentLang = l === "en" ? "zh" : "en";
            const res = await fetch("/studio/api/import", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                section,
                lang: l,
                markdown: drafts[l].text,
                title: drafts[l].title || drafts[other].title,
                slug: finalSlug,
                date,
                draft,
                featured,
                categories,
                description: drafts[l].description || undefined,
                keywords: keywords.length > 0 ? keywords : undefined,
              }),
            });
            if (!res.ok) {
              const err = await res.text();
              throw new Error(`${l}: ${err || res.statusText}`);
            }
          }
        },
      );
      status = { kind: "ok", text: t.statusDone };
    } catch (e) {
      status = {
        kind: "error",
        text: `${t.statusFail}: ${e instanceof Error ? e.message : e}`,
      };
    } finally {
      submitting = false;
    }
  }

  const printedOn = new Date().toISOString().split("T")[0];
</script>

<div>
  <!-- Header -->
  <PrintedSection>
    <PrintedPageTitle icon="pen">{t.title}</PrintedPageTitle>
    <p class="font-serif text-xs text-printer-ink-light dark:text-printer-ink-dark/50">
      {t.subtitle}
    </p>
  </PrintedSection>

  <PrintedDivider style="solid" />

  <!-- Language tabs -->
  <div class="mb-4 flex items-center gap-1">
    {#each LANGS as l (l)}
      <button
        type="button"
        onclick={() => (tab = l)}
        class={[
          "px-3 py-1 font-mono text-[10px] uppercase tracking-widest rounded-sm",
          tab === l
            ? "bg-printer-ink text-printer-paper dark:bg-printer-ink-dark dark:text-printer-paper-dark"
            : "border border-printer-ink/20 text-printer-ink-light hover:border-printer-accent dark:border-printer-ink-dark/20 dark:text-printer-ink-dark/60",
        ].join(" ")}
      >
        {l === "en" ? "English" : "中文"}
      </button>
    {/each}
    {#if drafts[tab].fileName}
      <span class="ml-2 font-mono text-[9px] uppercase tracking-wider text-printer-accent">
        {t.imported(drafts[tab].fileName)}
      </span>
    {/if}
  </div>

  <!-- Per-language content -->
  <div class="mb-2 space-y-3">
    <label class="block">
      <span class="mb-0.5 block font-mono text-[9px] uppercase tracking-wider text-printer-ink-light/60 dark:text-printer-ink-dark/50">
        {t.titleField}
      </span>
      <input
        type="text"
        bind:value={drafts[tab].title}
        maxlength={99}
        placeholder={t.titlePlaceholder}
        class="w-full rounded-sm border border-printer-ink/20 bg-transparent px-2.5 py-1.5 text-[13px] focus:border-printer-accent focus:outline-none dark:border-printer-ink-dark/20 dark:text-printer-ink-dark"
      />
    </label>
    <label class="block">
      <span class="mb-0.5 block font-mono text-[9px] uppercase tracking-wider text-printer-ink-light/60 dark:text-printer-ink-dark/50">
        {t.descField}
      </span>
      <input
        type="text"
        bind:value={drafts[tab].description}
        maxlength={999}
        class="w-full rounded-sm border border-printer-ink/20 bg-transparent px-2.5 py-1.5 text-[12px] focus:border-printer-accent focus:outline-none dark:border-printer-ink-dark/20 dark:text-printer-ink-dark"
      />
    </label>
    <div>
      <span class="mb-0.5 block font-mono text-[9px] uppercase tracking-wider text-printer-ink-light/60 dark:text-printer-ink-dark/50">
        {t.importField}
      </span>
      <input
        type="file"
        accept=".md,.markdown,text/markdown"
        onchange={(e) => onFileInput(e, tab)}
        class="w-full text-[11px] text-printer-ink dark:text-printer-ink-dark file:mr-2 file:py-1 file:px-2 file:rounded file:border file:border-printer-ink/20 dark:file:border-printer-ink-dark/20 file:bg-transparent file:font-mono file:text-[10px] file:uppercase file:tracking-wider file:text-printer-ink dark:file:text-printer-ink-dark"
      />
    </div>
    <label class="block">
      <span class="mb-0.5 block font-mono text-[9px] uppercase tracking-wider text-printer-ink-light/60 dark:text-printer-ink-dark/50">
        {t.bodyField}
      </span>
      <textarea
        bind:value={drafts[tab].text}
        rows="16"
        spellcheck="false"
        placeholder={t.bodyPlaceholder}
        class="w-full resize-y rounded-sm border border-printer-ink/20 bg-transparent px-2.5 py-2 font-mono text-[12px] leading-relaxed focus:border-printer-accent focus:outline-none dark:border-printer-ink-dark/20 dark:text-printer-ink-dark"
      ></textarea>
    </label>
  </div>

  <PrintedDivider style="dashed" />

  <!-- Shared meta -->
  <PrintedSection label={t.metaLabel} labelIcon="archive">
    <!-- 栏目 → 分类 两级级联：先选栏目，分类 chips 随之切换 -->
    <div class="mb-4">
      <span
        class="mb-1 block font-mono text-[9px] uppercase tracking-wider text-printer-ink-light/60 dark:text-printer-ink-dark/50"
      >
        {t.sectionLabel}
      </span>
      <div class="flex flex-wrap items-center gap-1.5">
        {#each ([
          "posts",
          "projects",
        ] as const) as s (s)}
          <button type="button" onclick={() => (section = s)}>
            <PrintedLabel
              variant={section === s ? "accent" : "muted"}
              icon={s === "posts" ? "send" : "apps"}
            >
              {s === "posts" ? t.share : t.projects}
            </PrintedLabel>
          </button>
        {/each}
      </div>
    </div>

    <div class="grid grid-cols-1 gap-3 sm:grid-cols-2 mb-3">
      <label class="block">
        <span class="mb-0.5 block font-mono text-[9px] uppercase tracking-wider text-printer-ink-light/60 dark:text-printer-ink-dark/50">
          {t.slugField}
        </span>
        <input
          type="text"
          bind:value={slug}
          placeholder={deriveSlug()}
          class="w-full rounded-sm border border-printer-ink/20 bg-transparent px-2.5 py-1.5 font-mono text-[12px] focus:border-printer-accent focus:outline-none dark:border-printer-ink-dark/20 dark:text-printer-ink-dark"
        />
      </label>
      <label class="block">
        <span class="mb-0.5 block font-mono text-[9px] uppercase tracking-wider text-printer-ink-light/60 dark:text-printer-ink-dark/50">
          {t.dateField}
        </span>
        <input
          type="date"
          bind:value={date}
          class="w-full rounded-sm border border-printer-ink/20 bg-transparent px-2.5 py-1.5 font-mono text-[12px] focus:border-printer-accent focus:outline-none dark:border-printer-ink-dark/20 dark:text-printer-ink-dark"
        />
      </label>
    </div>

    <div class="flex flex-wrap items-center gap-x-5 gap-y-2 mb-3">
      <label class="flex items-center gap-1.5">
        <input type="checkbox" bind:checked={draft} class="rounded-sm" />
        <span class="font-mono text-[11px] tracking-wider">{t.draftLabel}</span>
      </label>
      <label class="flex items-center gap-1.5">
        <input type="checkbox" bind:checked={featured} class="rounded-sm" />
        <span class="font-mono text-[11px] tracking-wider">{t.featuredLabel}</span>
      </label>
    </div>

    <div>
      <span class="mb-1 block font-mono text-[9px] uppercase tracking-wider text-printer-ink-light/60 dark:text-printer-ink-dark/50">
        {t.categoriesLabel}
      </span>
      <div class="flex flex-wrap gap-1.5">
        {#each sectionCategories as cat (cat.slug)}
          <button
            type="button"
            onclick={() => toggleCategory(cat.slug)}
            class={[
              "font-mono text-[10px] tracking-wider px-2 py-0.5 rounded border transition",
              categories.includes(cat.slug)
                ? "bg-printer-accent text-printer-paper dark:bg-printer-accent-dark dark:text-printer-paper-dark border-transparent"
                : "border-printer-ink/20 text-printer-ink-light hover:border-printer-accent dark:border-printer-ink-dark/20 dark:text-printer-ink-dark/60",
            ].join(" ")}
          >
            {cat.name[tab]}
          </button>
        {/each}
        {#if sectionCategories.length === 0}
          <span class="font-mono text-[10px] text-printer-ink-light/50">{t.noCategories}</span>
        {/if}
      </div>
    </div>
  </PrintedSection>

  <PrintedDivider style="dashed" />

  <!-- Publish -->
  <div class="flex flex-wrap items-center gap-3">
    <button
      type="button"
      onclick={submit}
      disabled={submitting}
      class="rounded-sm border border-printer-accent dark:border-printer-accent-dark bg-printer-accent dark:bg-printer-accent-dark px-4 py-1.5 font-mono text-[11px] uppercase tracking-wider text-printer-paper dark:text-printer-paper-dark hover:opacity-90 disabled:opacity-50"
    >
      {submitting ? t.publishing : t.publish}
    </button>
    {#if status}
      <span
        class={[
          "font-mono text-[10px]",
          status.kind === "ok"
            ? "text-printer-accent dark:text-printer-accent-dark"
            : status.kind === "error"
              ? "text-red-600 dark:text-red-400"
              : "text-printer-ink-light dark:text-printer-ink-dark/60",
        ].join(" ")}
      >
        {status.text}
      </span>
    {/if}
  </div>

  <!-- Footer stamp -->
  <div class="mt-8 pt-4 border-t border-dotted border-printer-ink/10 dark:border-printer-ink-dark/10">
    <div class="flex items-center justify-between">
      <div class="font-mono text-[9px] text-printer-ink-light dark:text-printer-ink-dark/30 tracking-wider uppercase">
        {dictionary.labels.printedOn} {printedOn}
      </div>
      <PrintedLabel variant="muted">hhy.homes</PrintedLabel>
    </div>
  </div>
</div>
