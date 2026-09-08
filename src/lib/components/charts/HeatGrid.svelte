<script lang="ts">
  /**
   * 发布热力图：GitHub 式周历点阵，数据为真实文章日期。
   * 低档用墨色、高档用品牌橙，带月份刻度与悬停读数。
   */
  export type HeatCell = { date: string; count: number; level: 0 | 1 | 2 | 3 | 4 };

  let {
    cells,
    weeks = 26,
    lang,
  }: {
    cells: HeatCell[];
    weeks?: number;
    lang: "zh" | "en";
  } = $props();

  let hover = $state<HeatCell | null>(null);

  const LEVEL_OPACITY = [0.08, 0.3, 0.55, 0.78, 1];

  /** 每列起始月份变化处显示月份刻度 */
  let monthTicks = $derived.by(() => {
    const ticks: { col: number; label: string }[] = [];
    let lastMonth = -1;
    for (let col = 0; col < weeks; col++) {
      const cell = cells[col * 7];
      if (!cell) continue;
      const m = new Date(`${cell.date}T00:00:00`).getMonth();
      if (m !== lastMonth) {
        ticks.push({ col, label: String(m + 1) });
        lastMonth = m;
      }
    }
    return ticks;
  });

  function fmtDate(date: string): string {
    const d = new Date(`${date}T00:00:00`);
    return new Intl.DateTimeFormat(lang === "zh" ? "zh-CN" : "en-US", { dateStyle: "medium" }).format(d);
  }
</script>

<div>
  <div class="relative ml-6">
    <div class="absolute -top-3.5 left-0 right-0 flex font-mono text-[8px] tracking-widest text-printer-ink-light dark:text-printer-ink-dark/40">
      {#each monthTicks as tick (tick.col)}
        <span class="absolute" style:left={`${(tick.col / weeks) * 100}%`}>{tick.label}</span>
      {/each}
    </div>
    <div class="grid grid-flow-col grid-rows-7 gap-[3px]" style:grid-auto-columns="minmax(0, 1fr)">
      {#each cells as cell, i (cell.date)}
        <button
          type="button"
          title={`${cell.date} · ${cell.count}`}
          onmouseenter={() => (hover = cell)}
          onmouseleave={() => (hover = null)}
          class={[
            "hg-in w-full aspect-square cursor-default",
            cell.level <= 2 ? "bg-printer-ink dark:bg-printer-ink-dark" : "bg-printer-accent dark:bg-printer-accent-dark",
          ].join(" ")}
          style:opacity={LEVEL_OPACITY[cell.level]}
          style:animation-delay={`${Math.floor(i / 7) * 16}ms`}
        ></button>
      {/each}
    </div>
  </div>
  <div class="mt-2 ml-6 flex items-center justify-between font-mono text-[9px] tracking-wider uppercase text-printer-ink-light dark:text-printer-ink-dark/50">
    <span>{hover ? `${fmtDate(hover.date)} · ${hover.count}` : "—"}</span>
    <span class="inline-flex items-center gap-1">
      {lang === "zh" ? "少" : "LESS"}
      {#each [0, 1, 2, 3, 4] as lv (lv)}
        <span
          class="inline-block h-2.5 w-2.5 {lv <= 2 ? "bg-printer-ink dark:bg-printer-ink-dark" : "bg-printer-accent dark:bg-printer-accent-dark"}"
          style:opacity={LEVEL_OPACITY[lv]}
        ></span>
      {/each}
      {lang === "zh" ? "多" : "MORE"}
    </span>
  </div>
</div>

<style>
  @keyframes hg-in {
    from {
      transform: scale(0);
    }
  }
  .hg-in {
    animation: hg-in 380ms cubic-bezier(0.22, 1, 0.36, 1) both;
  }
  @media (prefers-reduced-motion: reduce) {
    .hg-in {
      animation: none;
    }
  }
</style>
