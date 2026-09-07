<script lang="ts">
  import type { Snippet } from "svelte";

  /**
   * Markdown 正文「原地编辑」组件：
   * - 非编辑态渲染公共版内容（children，如 <PostContent>）；
   * - 编辑态在原位展开等宽 textarea 编辑 raw Markdown。
   *   服务端数据只含渲染后的 HTML，raw 按需经 loadRaw 从 studio API 拉取。
   */
  let {
    editing,
    sourceKey,
    loadRaw,
    onsave,
    children,
    label = "正文 · MARKDOWN",
  }: {
    editing: boolean;
    /** 数据源标识（如文章目录路径），变更后重新拉取 raw。 */
    sourceKey: string;
    /** 拉取 raw Markdown 正文。 */
    loadRaw: () => Promise<string>;
    /** 保存回调；失败抛错（草稿保留在输入框）。 */
    onsave: (raw: string) => void | Promise<void>;
    children: Snippet;
    label?: string;
  } = $props();

  let loadedKey = $state<string | null>(null);
  let rawLoaded = $state("");
  let draft = $state("");
  let loading = $state(false);
  let loadError = $state("");
  let saving = $state(false);
  let saveError = $state("");

  $effect(() => {
    if (!editing) return;
    if (loadedKey === sourceKey || loading) return;
    loading = true;
    loadError = "";
    loadRaw()
      .then((raw) => {
        rawLoaded = raw;
        draft = raw;
        loadedKey = sourceKey;
      })
      .catch((e) => {
        loadError = e instanceof Error ? e.message : String(e);
      })
      .finally(() => {
        loading = false;
      });
  });

  async function save() {
    if (saving) return;
    saving = true;
    saveError = "";
    try {
      await onsave(draft);
    } catch (e) {
      saveError = e instanceof Error ? e.message : String(e);
    } finally {
      saving = false;
    }
  }
</script>

{#if editing}
  <div class="mb-8">
    <div class="mb-1 flex items-center gap-2">
      <span
        class="font-mono text-[10px] tracking-[0.3em] uppercase text-printer-accent dark:text-printer-accent-dark bg-printer-accent/5 dark:bg-printer-accent-dark/10 px-2 py-[3px] rounded-sm"
      >
        {label}
      </span>
      <div class="flex-1 h-px bg-printer-ink/5 dark:bg-printer-ink-dark/5"></div>
      {#if saving}
        <span class="font-mono text-[10px] text-printer-ink-light">保存中…</span>
      {:else if saveError}
        <span class="font-mono text-[10px] text-red-600 dark:text-red-400">
          保存失败：{saveError}
        </span>
      {/if}
    </div>

    {#if loading}
      <p class="font-mono text-[10px] uppercase tracking-wider text-printer-ink-light/60">
        加载正文…
      </p>
    {:else if loadError}
      <p class="font-mono text-[10px] text-red-600 dark:text-red-400">
        加载失败：{loadError}
      </p>
    {:else}
      <textarea
        bind:value={draft}
        rows="20"
        spellcheck="false"
        class="w-full resize-y rounded-sm border border-printer-accent/30 bg-printer-paper dark:bg-printer-paper-dark px-3 py-2.5 font-mono text-[12px] leading-relaxed text-printer-ink dark:text-printer-ink-dark focus:border-printer-accent dark:focus:border-printer-accent-dark focus:outline-none"
      ></textarea>
      <div class="mt-2 flex items-center gap-2">
        <button
          type="button"
          onclick={save}
          disabled={saving}
          class="rounded-sm border border-printer-accent dark:border-printer-accent-dark bg-printer-accent dark:bg-printer-accent-dark px-3 py-1 font-mono text-[10px] tracking-wider uppercase text-printer-paper dark:text-printer-paper-dark hover:opacity-90 disabled:opacity-50"
        >
          {saving ? "保存中..." : "✓ 保存"}
        </button>
        <button
          type="button"
          onclick={() => (draft = rawLoaded)}
          disabled={saving}
          class="rounded-sm border border-printer-ink/15 dark:border-printer-ink-dark/20 px-3 py-1 font-mono text-[10px] tracking-wider uppercase text-printer-ink-light hover:border-printer-accent dark:text-printer-ink-dark/60 disabled:opacity-50"
        >
          还原
        </button>
      </div>
    {/if}
  </div>
{:else}
  {@render children()}
{/if}
