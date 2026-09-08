<script lang="ts">
  /**
   * 横向条形图：TOP 榜 / 分类计数通用。
   * 条身用墨色，榜首用品牌橙；数值右对齐 mono。
   */
  export type BarRow = { label: string; value: number; hint?: string };

  let {
    rows,
    format = (n: number) => String(n),
    accentFirst = true,
  }: {
    rows: BarRow[];
    format?: (n: number) => string;
    accentFirst?: boolean;
  } = $props();

  let max = $derived(Math.max(1, ...rows.map((r) => r.value)));
  // 数据签名：变化时重挂载，重播生长动画
  let sig = $derived(rows.map((r) => `${r.label}:${r.value}`).join("|"));
</script>

{#key sig}
  <ul class="space-y-3">
    {#each rows as row, i (row.label + i)}
    <li>
      <div class="flex items-baseline justify-between gap-2">
        <span class="min-w-0 flex-1 truncate font-mono text-[10px] tracking-wider text-printer-ink dark:text-printer-ink-dark">
          {row.label}
          {#if row.hint}
            <span class="text-printer-ink-light/70 dark:text-printer-ink-dark/30">· {row.hint}</span>
          {/if}
        </span>
        <span class="shrink-0 font-mono text-[10px] font-bold text-printer-ink dark:text-printer-ink-dark">
          {format(row.value)}
        </span>
      </div>
      <div class="mt-1.5 h-2 w-full bg-printer-ink/8 dark:bg-printer-ink-dark/8">
        <div
          class={[
            "bc-grow h-full transition-[width] duration-700",
            i === 0 && accentFirst
              ? "bg-printer-accent dark:bg-printer-accent-dark"
              : "bg-printer-ink/55 dark:bg-printer-ink-dark/55",
          ].join(" ")}
          style:width={`${((row.value / max) * 100).toFixed(1)}%`}
          style:animation-delay={`${i * 70}ms`}
        ></div>
      </div>
    </li>
    {/each}
  </ul>
{/key}

<style>
  @keyframes bc-grow {
    from {
      transform: scaleX(0);
    }
  }
  .bc-grow {
    transform-origin: left center;
    animation: bc-grow 700ms cubic-bezier(0.22, 1, 0.36, 1) both;
  }
  @media (prefers-reduced-motion: reduce) {
    .bc-grow {
      animation: none;
    }
  }
</style>
