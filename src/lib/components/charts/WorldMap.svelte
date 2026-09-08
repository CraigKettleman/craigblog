<script lang="ts">
  /**
   * 世界地图（真实 Natural Earth 海岸线，等距圆柱投影）。
   * 按国家 ISO-2 着色（分位数分级），悬停显示国名 + 访问量。
   */
  import { COUNTRIES, MAP_HEIGHT, MAP_WIDTH } from "$lib/assets/world-map";

  let {
    values,
    lang,
    format = (n: number) => String(n),
    emptyHint = "",
  }: {
    /** ISO-2 → 访问量 */
    values: Map<string, number>;
    lang: "zh" | "en";
    format?: (n: number) => string;
    emptyHint?: string;
  } = $props();

  let hover = $state<{ a2: string; name: string; pv: number; x: number; y: number } | null>(null);

  let max = $derived.by(() => {
    let m = 0;
    for (const v of values.values()) m = Math.max(m, v);
    return m;
  });

  /** 分位数分级：0 无数据，1-4 递增墨色/橙色 */
  function levelOf(pv: number): 0 | 1 | 2 | 3 | 4 {
    if (pv <= 0 || max <= 0) return 0;
    const r = pv / max;
    if (r >= 0.66) return 4;
    if (r >= 0.33) return 3;
    if (r >= 0.12) return 2;
    return 1;
  }

  const FILL: Record<number, { cls: string; opacity: number }> = {
    0: { cls: "fill-printer-ink dark:fill-printer-ink-dark", opacity: 0.07 },
    1: { cls: "fill-printer-ink dark:fill-printer-ink-dark", opacity: 0.28 },
    2: { cls: "fill-printer-ink dark:fill-printer-ink-dark", opacity: 0.5 },
    3: { cls: "fill-printer-accent dark:fill-printer-accent-dark", opacity: 0.55 },
    4: { cls: "fill-printer-accent dark:fill-printer-accent-dark", opacity: 0.95 },
  };

  function onMoveSvg(e: MouseEvent) {
    const target = e.target as SVGPathElement;
    const a2 = target.getAttribute?.("data-a2");
    if (a2 === null || a2 === undefined) {
      hover = null;
      return;
    }
    const c = COUNTRIES.find((x) => (x.a2 || x.en) === a2);
    if (!c) {
      hover = null;
      return;
    }
    const wrap = (e.currentTarget as SVGSVGElement).closest("[data-map-wrap]") as HTMLElement;
    const rect = wrap.getBoundingClientRect();
    hover = {
      a2: c.a2,
      name: lang === "zh" ? c.zh : c.en,
      pv: values.get(c.a2) ?? 0,
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
  }
</script>

<div data-map-wrap class="relative">
  <svg
    viewBox="0 0 {MAP_WIDTH} {MAP_HEIGHT}"
    class="w-full h-auto"
    role="img"
    onmousemove={onMoveSvg}
    onmouseleave={() => (hover = null)}
  >
    <!-- 经纬网：赤道 + 回归线，打印图纸感 -->
    {#each [0, 23.5, -23.5] as lat (lat)}
      {@const y = ((85 - lat) / 170) * MAP_HEIGHT}
      <line
        x1="0"
        y1={y}
        x2={MAP_WIDTH}
        y2={y}
        stroke="currentColor"
        stroke-width="0.6"
        stroke-dasharray={lat === 0 ? "4 4" : "1.5 5"}
        class="text-printer-ink/12 dark:text-printer-ink-dark/12"
      />
    {/each}
    {#each COUNTRIES as c, i (c.a2 || c.en)}
      {@const lv = levelOf(values.get(c.a2) ?? 0)}
      <path
        d={c.d}
        data-a2={c.a2 || c.en}
        class="{FILL[lv].cls} wm-in"
        style:animation-delay={`${Math.round((c.cx / MAP_WIDTH) * 700)}ms`}
        fill-opacity={hover && hover.a2 === c.a2 ? Math.min(1, FILL[lv].opacity + 0.25) : FILL[lv].opacity}
        stroke="currentColor"
        stroke-width="0.4"
        class:cursor-pointer={lv > 0}
      />
    {/each}
  </svg>

  {#if hover}
    <div
      class="pointer-events-none absolute z-10 border border-printer-ink/15 dark:border-printer-ink-dark/15 bg-printer-paper dark:bg-printer-paper-dark thermal-texture px-2 py-1 shadow-md"
      style:left="{hover.x + 10}px"
      style:top="{hover.y + 8}px"
    >
      <div class="font-mono text-[10px] font-bold tracking-wider text-printer-ink dark:text-printer-ink-dark">
        {hover.name} {hover.a2 ? `(${hover.a2})` : ""}
      </div>
      <div class="font-mono text-[9px] text-printer-ink-light dark:text-printer-ink-dark/50">
        {hover.pv > 0 ? format(hover.pv) : emptyHint || "0"}
      </div>
    </div>
  {/if}

  <!-- 图例 -->
  <div class="mt-1 flex items-center gap-1.5 font-mono text-[8px] tracking-widest uppercase text-printer-ink-light dark:text-printer-ink-dark/40">
    <span>0</span>
    {#each [1, 2, 3, 4] as lv (lv)}
      <span class="h-2 w-4 {FILL[lv].cls}" style:opacity={FILL[lv].opacity}></span>
    {/each}
    <span>≤ {format(max)}</span>
  </div>
</div>

<style>
  @keyframes wm-in {
    from {
      opacity: 0;
    }
  }
  /* 从左到右的“打印头扫过”式显现 */
  .wm-in {
    animation: wm-in 420ms ease-out both;
  }
  @media (prefers-reduced-motion: reduce) {
    .wm-in {
      animation: none;
    }
  }
</style>
