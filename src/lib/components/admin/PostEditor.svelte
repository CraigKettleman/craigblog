<script lang="ts">
  import { browser } from "$app/environment";
  import AdminDrawer from "./AdminDrawer.svelte";
  import type { Language } from "$lib/dictionaries";

  interface CategoryMeta {
    slug: string;
    section: "posts";
    name: { en: string; zh: string };
  }

  interface PostRaw {
    path: string;
    section: "posts";
    slug: string;
    date: string;
    draft: boolean;
    featured: boolean;
    categories: string[];
    title: { en?: string; zh?: string };
    description: { en?: string; zh?: string };
    body: { en: string; zh: string };
  }

  /** 某语言的一套内容草稿。 */
  interface LangDraft {
    title: string;
    description: string;
    body: string;
  }

  let {
    lang,
    postPath,
    onclose,
    onSaved,
  }: {
    lang: Language;
    /** 文章目录路径，如 posts/2026-01-01-hello */
    postPath: string;
    onclose: () => void;
    /** 保存成功后回调（父页面负责刷新渲染结果） */
    onSaved?: () => void;
  } = $props();

  let loading = $state(true);
  let loadError = $state("");
  let post = $state<PostRaw | null>(null);
  let allCategories = $state<CategoryMeta[]>([]);
  let saving = $state(false);
  let status = $state("");

  // 共享元数据（两份语言共用）
  let editSlug = $state("");
  let editDate = $state("");
  let editDraft = $state(false);
  let editFeatured = $state(false);
  let editCategories = $state<string[]>([]);

  // 每个语言一套草稿；正文编辑面板跟随当前 tab
  // svelte-ignore state_referenced_locally -- tab 初始值即页面语言，之后仅由用户切换
  let tab = $state<Language>(lang);
  let drafts = $state<Record<Language, LangDraft>>({
    en: { title: "", description: "", body: "" },
    zh: { title: "", description: "", body: "" },
  });

  let editTitle = $derived(drafts[tab].title);
  let editDescription = $derived(drafts[tab].description);
  let editBody = $derived(drafts[tab].body);

  function setTitle(v: string) {
    drafts[tab].title = v;
  }
  function setDescription(v: string) {
    drafts[tab].description = v;
  }
  function setBody(v: string) {
    drafts[tab].body = v;
  }

  function switchTab(next: Language) {
    tab = next;
  }

  $effect(() => {
    if (browser) load();
  });

  async function load() {
    try {
      const [postRes, catRes] = await Promise.all([
        fetch(`/studio/api/posts/${postPath}`),
        fetch("/studio/api/categories"),
      ]);
      if (!postRes.ok) throw new Error(`加载失败 ${postRes.status}`);
      const data = (await postRes.json()) as PostRaw;
      post = data;
      drafts = {
        en: {
          title: data.title.en ?? "",
          description: data.description.en ?? "",
          body: data.body.en ?? "",
        },
        zh: {
          title: data.title.zh ?? "",
          description: data.description.zh ?? "",
          body: data.body.zh ?? "",
        },
      };
      editSlug = data.slug;
      editDate = data.date;
      editDraft = data.draft;
      editFeatured = data.featured;
      editCategories = [...data.categories];
      if (catRes.ok) allCategories = (await catRes.json()) as CategoryMeta[];
      loading = false;
    } catch (e) {
      loadError = e instanceof Error ? e.message : String(e);
      loading = false;
    }
  }

  function toggleCategory(slug: string) {
    const set = new Set(editCategories);
    if (set.has(slug)) set.delete(slug);
    else set.add(slug);
    editCategories = [...set];
  }

  const sectionCategories = $derived(
    allCategories.filter((c) => c.section === post?.section),
  );

  function validate(): string | null {
    if (!editSlug.trim()) return "slug 不能为空";
    if (!editDate.trim()) return "日期不能为空";
    return null;
  }

  async function save() {
    if (!post || saving) return;
    const problem = validate();
    if (problem) {
      status = problem;
      return;
    }
    saving = true;
    status = "保存中...";
    try {
      // 双语全量回传：另一语言是未编辑的原样草稿，保证不被清空
      const res = await fetch(`/studio/api/posts/${post.path}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          slug: editSlug,
          date: editDate,
          draft: editDraft,
          featured: editFeatured,
          categories: editCategories,
          title: { en: drafts.en.title, zh: drafts.zh.title },
          description: {
            en: drafts.en.description,
            zh: drafts.zh.description,
          },
          body: { en: drafts.en.body, zh: drafts.zh.body },
        }),
      });
      if (!res.ok) {
        const errText = await res.text();
        throw new Error(errText || `HTTP ${res.status}`);
      }
      status = "✓ 已保存";
      onSaved?.();
    } catch (e) {
      status = `保存失败：${e instanceof Error ? e.message : e}`;
    } finally {
      saving = false;
    }
  }
</script>

<AdminDrawer
  title={post ? `编辑文章 · ${post.slug}` : "编辑文章"}
  {onclose}
>
  {#snippet footer()}
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
    <button
      type="button"
      onclick={save}
      disabled={saving || loading || !!loadError}
      class="ml-auto bg-printer-accent px-4 py-1.5 font-mono text-[11px] uppercase tracking-wider text-printer-paper hover:opacity-90 disabled:opacity-50 dark:bg-printer-accent-dark dark:text-printer-paper-dark"
    >
      {saving ? "保存中..." : "保存"}
    </button>
  {/snippet}

  {#if loading}
    <p class="font-mono text-[10px] uppercase tracking-wider text-printer-ink-light/60">
      加载中...
    </p>
  {:else if loadError}
    <p class="font-mono text-[10px] text-red-600 dark:text-red-400">
      加载失败：{loadError}
    </p>
  {:else if post}
    <!-- 语言 tab -->
    <div class="mb-4 flex items-center gap-1">
      {#each ["en", "zh"] as l (l)}
        <button
          type="button"
          onclick={() => switchTab(l as Language)}
          class={[
            "px-3 py-1 font-mono text-[10px] uppercase tracking-widest",
            tab === l
              ? "bg-printer-ink text-printer-paper dark:bg-printer-ink-dark dark:text-printer-paper-dark"
              : "border border-printer-ink/20 text-printer-ink-light hover:border-printer-accent dark:border-printer-ink-dark/20 dark:text-printer-ink-dark/60",
          ].join(" ")}
        >
          {l === "en" ? "English" : "中文"}
        </button>
      {/each}
      <span
        class="ml-2 font-mono text-[9px] uppercase tracking-wider text-printer-ink-light/50"
      >
        posts
        {#if editDraft} · 草稿{/if}
      </span>
    </div>

    <!-- 当前语言内容 -->
    <div class="mb-5 space-y-3">
      <label class="block">
        <span
          class="mb-0.5 block font-mono text-[9px] uppercase tracking-wider text-printer-ink-light/60"
        >
          标题（{tab === "en" ? "Title" : "标题"}）
        </span>
        <input
          type="text"
          value={editTitle}
          oninput={(e) => setTitle((e.target as HTMLInputElement).value)}
          maxlength="99"
          class="w-full rounded-sm border border-printer-ink/20 bg-transparent px-2.5 py-1.5 text-[13px] focus:border-printer-accent focus:outline-none dark:border-printer-ink-dark/20 dark:text-printer-ink-dark"
        />
      </label>
      <label class="block">
        <span
          class="mb-0.5 block font-mono text-[9px] uppercase tracking-wider text-printer-ink-light/60"
        >
          描述（可选）
        </span>
        <input
          type="text"
          value={editDescription}
          oninput={(e) => setDescription((e.target as HTMLInputElement).value)}
          maxlength="999"
          class="w-full rounded-sm border border-printer-ink/20 bg-transparent px-2.5 py-1.5 text-[12px] focus:border-printer-accent focus:outline-none dark:border-printer-ink-dark/20 dark:text-printer-ink-dark"
        />
      </label>
      <label class="block">
        <span
          class="mb-0.5 block font-mono text-[9px] uppercase tracking-wider text-printer-ink-light/60"
        >
          正文 · Markdown
        </span>
        <textarea
          value={editBody}
          oninput={(e) => setBody((e.target as HTMLTextAreaElement).value)}
          rows="18"
          spellcheck="false"
          class="w-full resize-y rounded-sm border border-printer-ink/20 bg-transparent px-2.5 py-2 font-mono text-[12px] leading-relaxed focus:border-printer-accent focus:outline-none dark:border-printer-ink-dark/20 dark:text-printer-ink-dark"
        ></textarea>
        <p
          class="mt-1 font-mono text-[9px] tracking-wider text-printer-ink-light/50"
        >
          保存后页面会刷新，直接呈现渲染效果。
        </p>
      </label>
    </div>

    <!-- 共享元数据 -->
    <div
      class="mb-2 border-t border-dashed border-printer-ink/10 pt-4 dark:border-printer-ink-dark/10"
    >
      <p
        class="mb-2 font-mono text-[9px] uppercase tracking-wider text-printer-ink-light/50"
      >
        元数据（两种语言共用）
      </p>
      <div class="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <label class="block">
          <span
            class="mb-0.5 block font-mono text-[9px] uppercase tracking-wider text-printer-ink-light/60"
          >
            Slug
          </span>
          <input
            type="text"
            bind:value={editSlug}
            class="w-full rounded-sm border border-printer-ink/20 bg-transparent px-2.5 py-1.5 font-mono text-[12px] focus:border-printer-accent focus:outline-none dark:border-printer-ink-dark/20 dark:text-printer-ink-dark"
          />
        </label>
        <label class="block">
          <span
            class="mb-0.5 block font-mono text-[9px] uppercase tracking-wider text-printer-ink-light/60"
          >
            日期
          </span>
          <input
            type="date"
            bind:value={editDate}
            class="w-full rounded-sm border border-printer-ink/20 bg-transparent px-2.5 py-1.5 font-mono text-[12px] focus:border-printer-accent focus:outline-none dark:border-printer-ink-dark/20 dark:text-printer-ink-dark"
          />
        </label>
      </div>
      <div class="mt-3 flex flex-wrap items-center gap-x-5 gap-y-2">
        <label class="flex items-center gap-1.5">
          <input type="checkbox" bind:checked={editDraft} class="rounded-sm" />
          <span class="font-mono text-[11px] tracking-wider">
            草稿（不发布到公共版）
          </span>
        </label>
        <label class="flex items-center gap-1.5">
          <input type="checkbox" bind:checked={editFeatured} class="rounded-sm" />
          <span class="font-mono text-[11px] tracking-wider">精选</span>
        </label>
      </div>
      <div class="mt-3">
        <span
          class="mb-1 block font-mono text-[9px] uppercase tracking-wider text-printer-ink-light/60"
        >
          分类
        </span>
        <div class="flex flex-wrap gap-1.5">
          {#each sectionCategories as cat (cat.slug)}
            <button
              type="button"
              onclick={() => toggleCategory(cat.slug)}
              class={[
                "font-mono text-[10px] tracking-wider px-2 py-0.5 rounded border transition",
                editCategories.includes(cat.slug)
                  ? "bg-printer-accent text-printer-paper border-transparent dark:bg-printer-accent-dark dark:text-printer-paper-dark"
                  : "border-printer-ink/20 text-printer-ink-light hover:border-printer-accent dark:border-printer-ink-dark/20 dark:text-printer-ink-dark/60",
              ].join(" ")}
            >
              {cat.name[lang]}
            </button>
          {/each}
          {#if sectionCategories.length === 0}
            <span class="font-mono text-[10px] text-printer-ink-light/50">
              暂无分类
            </span>
          {/if}
        </div>
      </div>
    </div>
  {/if}
</AdminDrawer>
