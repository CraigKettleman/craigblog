<script lang="ts">
  import { browser } from "$app/environment";
  import AdminDrawer from "./AdminDrawer.svelte";
  import type { Language } from "$lib/dictionaries";

  interface SiteData {
    websiteName: { en: string; zh: string };
    motto: { en: string; zh: string };
    mottos: { en: string[]; zh: string[] };
    brandName: { en: string; zh: string };
    brandTagline: { en: string; zh: string };
    about: { en: string; zh: string };
  }

  let {
    lang,
    onclose,
    onSaved,
  }: {
    lang: Language;
    onclose: () => void;
    onSaved?: () => void;
  } = $props();

  let loading = $state(true);
  let loadError = $state("");
  let saving = $state(false);
  let status = $state("");

  // svelte-ignore state_referenced_locally -- tab 初始值即页面语言，之后仅由用户切换
  let tab = $state<Language>(lang);
  let data = $state<SiteData>({
    websiteName: { en: "", zh: "" },
    motto: { en: "", zh: "" },
    mottos: { en: [], zh: [] },
    brandName: { en: "", zh: "" },
    brandTagline: { en: "", zh: "" },
    about: { en: "", zh: "" },
  });

  // 当前 tab 编辑字段（motots 用多行文本，每行一条）
  let editWebsiteName = $state("");
  let editMotto = $state("");
  let editBrandName = $state("");
  let editBrandTagline = $state("");
  let editMottosText = $state("");
  let editAbout = $state("");

  function applyTab() {
    editWebsiteName = data.websiteName[tab] ?? "";
    editMotto = data.motto[tab] ?? "";
    editBrandName = data.brandName[tab] ?? "";
    editBrandTagline = data.brandTagline[tab] ?? "";
    editMottosText = (data.mottos[tab] ?? []).join("\n");
    editAbout = data.about[tab] ?? "";
  }

  function switchTab(next: Language) {
    // 先存回当前 tab
    persistTab();
    tab = next;
    applyTab();
  }

  function persistTab() {
    data.websiteName[tab] = editWebsiteName;
    data.motto[tab] = editMotto;
    data.brandName[tab] = editBrandName;
    data.brandTagline[tab] = editBrandTagline;
    data.mottos[tab] = editMottosText
      .split("\n")
      .map((s) => s.trim())
      .filter(Boolean);
    data.about[tab] = editAbout;
  }

  $effect(() => {
    if (browser) load();
  });

  async function load() {
    try {
      const res = await fetch("/studio/api/site");
      if (!res.ok) throw new Error(`加载失败 ${res.status}`);
      data = (await res.json()) as SiteData;
      tab = lang;
      applyTab();
      loading = false;
    } catch (e) {
      loadError = e instanceof Error ? e.message : String(e);
      loading = false;
    }
  }

  async function save() {
    if (saving) return;
    persistTab();
    saving = true;
    status = "保存中...";
    try {
      const res = await fetch("/studio/api/site", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
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

<AdminDrawer title="站点资料" width="max-w-2xl" {onclose}>
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
  {:else}
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
    </div>

    <div class="space-y-3">
      <label class="block">
        <span
          class="mb-0.5 block font-mono text-[9px] uppercase tracking-wider text-printer-ink-light/60"
        >
          名称（页头品牌名）
        </span>
        <input
          type="text"
          bind:value={editWebsiteName}
          class="w-full rounded-sm border border-printer-ink/20 bg-transparent px-2.5 py-1.5 text-[13px] focus:border-printer-accent focus:outline-none dark:border-printer-ink-dark/20 dark:text-printer-ink-dark"
        />
      </label>
      <label class="block">
        <span
          class="mb-0.5 block font-mono text-[9px] uppercase tracking-wider text-printer-ink-light/60"
        >
          座右铭
        </span>
        <input
          type="text"
          bind:value={editMotto}
          class="w-full rounded-sm border border-printer-ink/20 bg-transparent px-2.5 py-1.5 text-[13px] focus:border-printer-accent focus:outline-none dark:border-printer-ink-dark/20 dark:text-printer-ink-dark"
        />
      </label>
      <label class="block">
        <span
          class="mb-0.5 block font-mono text-[9px] uppercase tracking-wider text-printer-ink-light/60"
        >
          座右铭列表（每行一条，页面随机展示）
        </span>
        <textarea
          bind:value={editMottosText}
          rows="3"
          class="w-full resize-y rounded-sm border border-printer-ink/20 bg-transparent px-2.5 py-2 font-mono text-[12px] leading-relaxed focus:border-printer-accent focus:outline-none dark:border-printer-ink-dark/20 dark:text-printer-ink-dark"
        ></textarea>
      </label>
      <label class="block">
        <span
          class="mb-0.5 block font-mono text-[9px] uppercase tracking-wider text-printer-ink-light/60"
        >
          品牌副标题（页头 tagline）
        </span>
        <input
          type="text"
          bind:value={editBrandTagline}
          class="w-full rounded-sm border border-printer-ink/20 bg-transparent px-2.5 py-1.5 text-[13px] focus:border-printer-accent focus:outline-none dark:border-printer-ink-dark/20 dark:text-printer-ink-dark"
        />
      </label>
      <label class="block">
        <span
          class="mb-0.5 block font-mono text-[9px] uppercase tracking-wider text-printer-ink-light/60"
        >
          品牌名（一般与名称一致）
        </span>
        <input
          type="text"
          bind:value={editBrandName}
          class="w-full rounded-sm border border-printer-ink/20 bg-transparent px-2.5 py-1.5 text-[13px] focus:border-printer-accent focus:outline-none dark:border-printer-ink-dark/20 dark:text-printer-ink-dark"
        />
      </label>
      <label class="block">
        <span
          class="mb-0.5 block font-mono text-[9px] uppercase tracking-wider text-printer-ink-light/60"
        >
          个人简介（Markdown，显示在「更多」页）
        </span>
        <textarea
          bind:value={editAbout}
          rows="8"
          class="w-full resize-y rounded-sm border border-printer-ink/20 bg-transparent px-2.5 py-2 font-mono text-[12px] leading-relaxed focus:border-printer-accent focus:outline-none dark:border-printer-ink-dark/20 dark:text-printer-ink-dark"
        ></textarea>
      </label>
    </div>
  {/if}
</AdminDrawer>
