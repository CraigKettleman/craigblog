<script lang="ts">
  import { browser } from "$app/environment";
  import { parseFrontMatter } from "$lib/frontmatter";
  import AdminDrawer from "./admin/AdminDrawer.svelte";
  import Icon from "./Icon.svelte";
  import type { Language } from "$lib/dictionaries";

  interface CategoryMeta {
    slug: string;
    section: "posts";
    name: { en: string; zh: string };
    description?: { en: string; zh: string };
  }

  let {
    lang,
    onclose,
  }: {
    lang: Language;
    onclose: () => void;
  } = $props();

  let enText = $state("");
  let zhText = $state("");
  let enFileName = $state("");
  let zhFileName = $state("");

  let titleEn = $state("");
  let titleZh = $state("");
  let slug = $state("");
  let date = $state(new Date().toISOString().slice(0, 10));
  let draft = $state(true);
  let featured = $state(false);
  let descriptionEn = $state("");
  let descriptionZh = $state("");
  let categories = $state<string[]>([]);
  let keywords = $state<string[]>([]);

  let section = $state<"posts">("posts");

  let allCategories = $state<CategoryMeta[]>([]);
  let submitting = $state(false);
  let status = $state("");

  $effect(() => {
    if (browser) loadCategories();
  });

  async function loadCategories() {
    try {
      const res = await fetch("/studio/api/categories");
      if (res.ok) allCategories = (await res.json()) as CategoryMeta[];
    } catch {
      /* ignore */
    }
  }

  const sectionCategories = $derived(allCategories.filter((c) => c.section === "posts"));

  function onFileInput(e: Event, which: "en" | "zh") {
    const input = e.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    if (which === "en") enFileName = file.name;
    else zhFileName = file.name;
    const reader = new FileReader();
    reader.onload = () => {
      const text = String(reader.result ?? "");
      if (which === "en") {
        enText = text;
        applyFrontmatter(text, "en");
      } else {
        zhText = text;
        applyFrontmatter(text, "zh");
      }
    };
    reader.readAsText(file);
  }

  function applyFrontmatter(text: string, which: "en" | "zh") {
    const { data } = parseFrontMatter(text);
    if (which === "en") {
      if (typeof data.title === "string") titleEn = data.title;
      if (typeof data.description === "string") descriptionEn = data.description;
    } else {
      if (typeof data.title === "string") titleZh = data.title;
      if (typeof data.description === "string") descriptionZh = data.description;
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

  function toggleCategory(slug: string) {
    const set = new Set(categories);
    if (set.has(slug)) set.delete(slug);
    else set.add(slug);
    categories = [...set];
  }

  function deriveSlug(): string {
    if (slug) return slug;
    const base = (titleEn || titleZh || "untitled")
      .toLowerCase()
      .replace(/[^a-z0-9\u4e00-\u9fff]+/g, "-")
      .replace(/^-|-$/g, "");
    return base || "untitled";
  }

  async function submit() {
    if (submitting) return;
    if (!enText && !zhText) {
      status = "请至少选择一个 MD 文件。";
      return;
    }
    submitting = true;
    status = "正在发布...";
    try {
      const finalSlug = deriveSlug();
      const items: Array<{
        lang: "en" | "zh";
        text: string;
        title: string;
        description: string;
      }> = [];
      if (enText) {
        items.push({
          lang: "en",
          text: enText,
          title: titleEn || titleZh,
          description: descriptionEn,
        });
      }
      if (zhText) {
        items.push({
          lang: "zh",
          text: zhText,
          title: titleZh || titleEn,
          description: descriptionZh,
        });
      }
      for (const item of items) {
        const res = await fetch("/studio/api/import", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            section,
            lang: item.lang,
            markdown: item.text,
            title: item.title,
            slug: finalSlug,
            date,
            draft,
            featured,
            categories,
            description: item.description || undefined,
            keywords: keywords.length ? keywords : undefined,
          }),
        });
        if (!res.ok) {
          const err = await res.text();
          throw new Error(`${item.lang}: ${err || res.statusText}`);
        }
      }
      status = "✓ 发布成功。";
      setTimeout(onclose, 1200);
    } catch (e) {
      status = `发布失败：${e instanceof Error ? e.message : e}`;
    } finally {
      submitting = false;
    }
  }
</script>

<AdminDrawer title="投稿" width="max-w-2xl" {onclose}>
  {#snippet footer()}
    <button
      type="button"
      onclick={submit}
      disabled={submitting || (!enText && !zhText)}
      class="bg-orange-500 hover:bg-orange-600 px-4 py-1.5 font-mono text-[11px] uppercase tracking-wider text-white disabled:opacity-50"
    >
      {submitting ? "发布中..." : "发布"}
    </button>
    <button
      type="button"
      onclick={onclose}
      class="font-mono text-[11px] uppercase tracking-wider text-printer-ink-light hover:text-printer-ink dark:text-printer-ink-dark/50 dark:hover:text-printer-ink-dark"
    >
      取消
    </button>
    {#if status}
      <span
        class={[
          "font-mono text-[10px]",
          status.startsWith("✓")
            ? "text-printer-accent dark:text-printer-accent-dark"
            : "text-red-600 dark:text-red-400",
        ].join(" ")}
      >
        {status}
      </span>
    {/if}
  {/snippet}

  <!-- 打印纸样式头带 -->
  <div class="mb-4 flex items-center gap-2 rounded-sm border border-orange-500/30 bg-orange-500/5 dark:bg-orange-500/10 px-3 py-2">
    <Icon name="upload" class="w-4 h-4 text-orange-500" />
    <div>
      <div class="font-mono text-[11px] font-bold tracking-wider uppercase text-orange-600 dark:text-orange-400">
        投稿
      </div>
      <div class="font-serif text-[10px] text-printer-ink-light dark:text-printer-ink-dark/50">
        {lang === "zh" ? "选择 Markdown 文件发布到「分享」" : "Publish a Markdown file to Share"}
      </div>
    </div>
  </div>

    <!-- 文件 -->
    <div class="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-4">
      <label class="block">
        <span class="font-mono text-[10px] tracking-wider uppercase text-printer-ink-light dark:text-printer-ink-dark/50">
          English MD
        </span>
        <input
          type="file"
          accept=".md,.markdown,text/markdown"
          onchange={(e) => onFileInput(e, "en")}
          class="mt-1 w-full text-[11px] file:mr-2 file:py-1 file:px-2 file:rounded file:border file:border-printer-ink/20 file:bg-transparent file:font-mono file:text-[10px] file:uppercase file:tracking-wider"
        />
        {#if enFileName}
          <span class="font-mono text-[9px] text-printer-accent">{enFileName}</span>
        {/if}
      </label>
      <label class="block">
        <span class="font-mono text-[10px] tracking-wider uppercase text-printer-ink-light dark:text-printer-ink-dark/50">
          中文 MD
        </span>
        <input
          type="file"
          accept=".md,.markdown,text/markdown"
          onchange={(e) => onFileInput(e, "zh")}
          class="mt-1 w-full text-[11px] file:mr-2 file:py-1 file:px-2 file:rounded file:border file:border-printer-ink/20 file:bg-transparent file:font-mono file:text-[10px] file:uppercase file:tracking-wider"
        />
        {#if zhFileName}
          <span class="font-mono text-[9px] text-printer-accent">{zhFileName}</span>
        {/if}
      </label>
    </div>

    <!-- 分区（固定「分享」，不再可选） -->
    <div class="mb-4">
      <span class="font-mono text-[10px] tracking-wider uppercase text-printer-ink-light dark:text-printer-ink-dark/50">
        分区
      </span>
      <div class="mt-1 inline-flex items-center gap-1.5 rounded-sm border border-printer-accent/40 bg-printer-accent/5 px-2.5 py-1 font-mono text-[11px] text-printer-accent dark:border-printer-accent-dark/40 dark:bg-printer-accent-dark/10 dark:text-printer-accent-dark">
        <Icon name="send" class="w-3 h-3" />
        {lang === "zh" ? "分享" : "Share"}
      </div>
    </div>

    <!-- 分类 -->
    <div class="mb-4">
      <span class="font-mono text-[10px] tracking-wider uppercase text-printer-ink-light dark:text-printer-ink-dark/50">
        分类（可多选）
      </span>
      <div class="mt-1 flex flex-wrap gap-1.5">
        {#each sectionCategories as cat (cat.slug)}
          <button
            type="button"
            onclick={() => toggleCategory(cat.slug)}
            class={[
              "font-mono text-[10px] tracking-wider px-2 py-0.5 rounded border transition",
              categories.includes(cat.slug)
                ? "bg-printer-accent text-printer-paper dark:bg-printer-accent-dark dark:text-printer-paper-dark border-transparent"
                : "border-printer-ink/20 dark:border-printer-ink-dark/20 hover:border-printer-accent dark:hover:border-printer-accent-dark",
            ].join(" ")}
          >
            {cat.name[lang]}
          </button>
        {/each}
        {#if sectionCategories.length === 0}
          <span class="font-mono text-[10px] text-printer-ink-light/50">暂无分类</span>
        {/if}
      </div>
    </div>

    <!-- 元信息 -->
    <div class="grid grid-cols-2 gap-3 mb-4">
      <label class="block">
        <span class="font-mono text-[9px] tracking-wider uppercase text-printer-ink-light/60">
          Slug（留空则自动从标题生成）
        </span>
        <input
          bind:value={slug}
          placeholder={deriveSlug()}
          class="mt-1 w-full border border-printer-ink/20 dark:border-printer-ink-dark/20 bg-transparent rounded px-2 py-1 text-[12px] font-mono"
        />
      </label>
      <label class="block">
        <span class="font-mono text-[9px] tracking-wider uppercase text-printer-ink-light/60">
          日期
        </span>
        <input
          type="date"
          bind:value={date}
          class="mt-1 w-full border border-printer-ink/20 dark:border-printer-ink-dark/20 bg-transparent rounded px-2 py-1 text-[12px]"
        />
      </label>
      <label class="flex items-center gap-1.5">
        <input type="checkbox" bind:checked={draft} class="rounded" />
        <span class="font-mono text-[11px] tracking-wider">草稿</span>
      </label>
      <label class="flex items-center gap-1.5">
        <input type="checkbox" bind:checked={featured} class="rounded" />
        <span class="font-mono text-[11px] tracking-wider">精选</span>
      </label>
    </div>

    <!-- 标题 -->
    <div class="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-4">
      <label class="block">
        <span class="font-mono text-[9px] tracking-wider uppercase text-printer-ink-light/60">
          英文标题
        </span>
        <input
          bind:value={titleEn}
          maxlength="99"
          class="mt-1 w-full border border-printer-ink/20 dark:border-printer-ink-dark/20 bg-transparent rounded px-2 py-1 text-[12px]"
        />
      </label>
      <label class="block">
        <span class="font-mono text-[9px] tracking-wider uppercase text-printer-ink-light/60">
          中文标题
        </span>
        <input
          bind:value={titleZh}
          maxlength="99"
          class="mt-1 w-full border border-printer-ink/20 dark:border-printer-ink-dark/20 bg-transparent rounded px-2 py-1 text-[12px]"
        />
      </label>
    </div>
  </AdminDrawer>
