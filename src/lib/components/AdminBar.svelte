<script lang="ts">
  import { browser } from "$app/environment";
  import { getEditMode, setEditMode } from "$lib/admin-state.svelte";

  let deploying = $state(false);
  let deployLog = $state("");
  let showLog = $state(false);
  let done = $state(false);

  async function publish() {
    if (!browser || deploying) return;
    deploying = true;
    done = false;
    deployLog = "正在发布...\n";
    showLog = true;
    try {
      const res = await fetch("/studio/api/deploy", { method: "POST" });
      if (!res.body) {
        deployLog += "无响应流。\n";
        deploying = false;
        return;
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      while (true) {
        const { done: finished, value } = await reader.read();
        if (finished) break;
        deployLog += decoder.decode(value, { stream: true });
      }
      deployLog += "\n✓ 发布完成。";
      done = true;
    } catch (e) {
      deployLog += `\n错误：${e}`;
    }
    deploying = false;
  }
</script>

{#if browser}
  <div
    class="fixed bottom-4 left-1/2 -translate-x-1/2 z-50 flex items-center gap-2 rounded-full border border-printer-ink/15 dark:border-printer-ink-dark/20 bg-printer-paper/95 dark:bg-printer-paper-dark/95 backdrop-blur shadow-lg px-3 py-1.5 text-[11px]"
  >
    <button
      onclick={() => setEditMode(!getEditMode())}
      title={getEditMode() ? "退出编辑，回到镜像预览" : "开启编辑模式"}
      class={[
        "font-mono tracking-wider uppercase px-2 py-0.5 rounded transition",
        getEditMode()
          ? "bg-printer-accent text-printer-paper dark:bg-printer-accent-dark dark:text-printer-paper-dark"
          : "hover:bg-printer-ink/10 dark:hover:bg-printer-ink-dark/10",
      ].join(" ")}
    >
      {getEditMode() ? "完成" : "✎ 编辑"}
    </button>
    <div class="w-px h-4 bg-printer-ink/15 dark:bg-printer-ink-dark/20"></div>
    <button
      onclick={publish}
      disabled={deploying}
      class="font-mono tracking-wider uppercase px-2 py-0.5 rounded text-printer-accent dark:text-printer-accent-dark hover:bg-printer-accent/10 dark:hover:bg-printer-accent-dark/10 disabled:opacity-50 transition"
    >
      {deploying ? "发布中" : "发布"}
    </button>
  </div>

  {#if showLog}
    <div
      class="fixed bottom-16 left-1/2 -translate-x-1/2 z-50 w-[min(620px,92vw)] max-h-64 overflow-auto rounded border border-printer-ink/15 dark:border-printer-ink-dark/20 bg-printer-paper dark:bg-printer-paper-dark shadow-xl p-3"
    >
      <div class="flex justify-between items-center mb-2">
        <span class="font-mono text-[10px] tracking-wider uppercase">
          {done ? "✓ 发布完成" : deploying ? "发布中..." : "发布日志"}
        </span>
        <button
          onclick={() => { showLog = false; }}
          class="font-mono text-[11px] text-printer-ink-light hover:text-printer-ink"
          aria-label="关闭"
        >
          ×
        </button>
      </div>
      <pre class="font-mono text-[10px] leading-relaxed whitespace-pre-wrap break-words text-printer-ink dark:text-printer-ink-dark">{deployLog}</pre>
    </div>
  {/if}
{/if}
