<script lang="ts">
  import type { Snippet } from "svelte";

  let {
    title,
    width = "max-w-4xl",
    onclose,
    children,
    footer,
  }: {
    title: string;
    /** 面板宽度类，默认 max-w-4xl。 */
    width?: string;
    onclose: () => void;
    children: Snippet;
    footer?: Snippet;
  } = $props();
</script>

<!-- 遮罩（真按钮，避免 a11y 警告） -->
<svelte:window
  onkeydown={(e) => {
    if (e.key === "Escape") onclose();
  }}
/>
<div class="fixed inset-0 z-[80]">
  <button
    type="button"
    aria-label="关闭"
    onclick={onclose}
    class="absolute inset-0 h-full w-full cursor-default bg-black/40"
  ></button>
  <div class="pointer-events-none absolute inset-0 flex items-center justify-center p-3 sm:p-6">
    <div
      role="dialog"
      aria-modal="true"
      aria-label={title}
      class="pointer-events-auto flex max-h-[92vh] w-full flex-col rounded border border-printer-ink/15 bg-printer-paper shadow-2xl dark:border-printer-ink-dark/20 dark:bg-printer-paper-dark {width}"
    >
      <div
        class="flex items-center justify-between border-b border-dashed border-printer-ink/15 px-5 py-3 dark:border-printer-ink-dark/20"
      >
        <h2 class="font-mono text-[11px] uppercase tracking-[0.25em]">
          {title}
        </h2>
        <button
          type="button"
          onclick={onclose}
          aria-label="关闭"
          class="font-mono text-sm leading-none text-printer-ink-light hover:text-printer-ink dark:text-printer-ink-dark/50 dark:hover:text-printer-ink-dark"
        >
          ×
        </button>
      </div>
      <div class="flex-1 overflow-y-auto px-5 py-4">
        {@render children()}
      </div>
      {#if footer}
        <div
          class="flex items-center gap-3 border-t border-dashed border-printer-ink/15 px-5 py-3 dark:border-printer-ink-dark/20"
        >
          {@render footer()}
        </div>
      {/if}
    </div>
  </div>
</div>
