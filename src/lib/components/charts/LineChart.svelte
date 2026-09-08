<script lang="ts">
  /**
   * 折线图：多序列 + 面积填充 + 悬停十字线读数。
   * 风格：热敏打印 —— 虚线网格、mono 刻度、无圆角。
   */
  export type LineSeries = {
    name: string;
    /** accent = 品牌橙实线；ink = 墨色虚线 */
    tone: "accent" | "ink";
    values: number[];
  };

  let {
    series,
    labels,
    format = (n: number) => String(n),
    height = 220,
  }: {
    series: LineSeries[];
    labels: string[];
    format?: (n: number) => string;
    height?: number;
  } = $props();

  const W = 640;
  const PAD = { l: 44, r: 10, t: 12, b: 20 };

  let hover = $state<number | null>(null);

  let n = $derived(labels.length);
  let max = $derived(Math.max(1, ...series.flatMap((s) => s.values)));
  let niceMax = $derived(niceCeil(max));

  function niceCeil(v: number): number {
    if (v <= 10) return 10;
    const mag = 10 ** Math.floor(Math.log10(v));
    return Math.ceil((v * 1.12) / mag) * mag;
  }

  function tx(i: number): number {
    return PAD.l + (i / Math.max(1, n - 1)) * (W - PAD.l - PAD.r);
  }
  function ty(v: number): number {
    return PAD.t + (1 - v / niceMax) * (height - PAD.t - PAD.b);
  }

  let paths = $derived(
    series.map((s) => ({
      ...s,
      d: s.values.map((v, i) => `${i === 0 ? "M" : "L"}${tx(i).toFixed(1)} ${ty(v).toFixed(1)}`).join(" "),
      area:
        s.values.length > 1
          ? `${s.values.map((v, i) => `${i === 0 ? "M" : "L"}${tx(i).toFixed(1)} ${ty(v).toFixed(1)}`).join(" ")} L${tx(s.values.length - 1).toFixed(1)} ${(height - PAD.b).toFixed(1)} L${tx(0).toFixed(1)} ${(height - PAD.b).toFixed(1)} Z`
          : "",
    })),
  );

  let xTicks = $derived.by(() => {
    if (n === 0) return [] as { x: number; label: string }[];
    const idx = [...new Set([0, 0.25, 0.5, 0.75, 1].map((f) => Math.min(n - 1, Math.round(f * (n - 1)))))] as number[];
    return idx.map((i) => ({ x: tx(i), label: labels[i] }));
  });

  // 数据签名：变化时重挂载路径，重播描线动画
  let sig = $derived(series.map((s) => s.values.join(".")).join("|"));

  function onMove(e: MouseEvent) {
    const svg = e.currentTarget as SVGSVGElement;
    const rect = svg.getBoundingClientRect();
    const fx = ((e.clientX - rect.left) / rect.width) * W;
    const ratio = (fx - PAD.l) / Math.max(1, W - PAD.l - PAD.r);
    const i = Math.round(ratio * (n - 1));
    hover = i >= 0 && i < n ? i : null;
  }
</script>

<div class="relative">
  <svg
    viewBox="0 0 {W} {height}"
    class="w-full h-auto select-none"
    role="img"
    onmousemove={onMove}
    onmouseleave={() => (hover = null)}
  >
    {#each [0.25, 0.5, 0.75, 1] as f (f)}
      {@const y = PAD.t + (1 - f) * (height - PAD.t - PAD.b)}
      <line x1={PAD.l} y1={y} x2={W - PAD.r} y2={y} stroke="currentColor" stroke-width="1" stroke-dasharray="3 4" class="text-printer-ink/15 dark:text-printer-ink-dark/15" />
      <text x={PAD.l - 6} y={y + 3} text-anchor="end" font-size="9" class="fill-printer-ink-light dark:fill-printer-ink-dark/40 font-mono">
        {format(Math.round(niceMax * f))}
      </text>
    {/each}
    <line x1={PAD.l} y1={height - PAD.b} x2={W - PAD.r} y2={height - PAD.b} stroke="currentColor" stroke-width="1" class="text-printer-ink/30 dark:text-printer-ink-dark/30" />
    {#key sig}
      {#each paths as s (s.name)}
        {#if s.tone === "accent" && s.area}
          <path d={s.area} class="lc-area fill-printer-accent/12 dark:fill-printer-accent-dark/12" />
        {/if}
        <path
          d={s.d}
          fill="none"
          pathLength={s.tone === "accent" ? 1 : undefined}
          stroke-width={s.tone === "accent" ? 2 : 1.25}
          stroke-dasharray={s.tone === "accent" ? undefined : "5 3"}
          class={s.tone === "accent"
            ? "lc-draw stroke-printer-accent dark:stroke-printer-accent-dark"
            : "lc-fade stroke-printer-ink-light dark:stroke-printer-ink-dark/50"}
        />
      {/each}
    {/key}
    {#if hover !== null}
      <line x1={tx(hover)} y1={PAD.t} x2={tx(hover)} y2={height - PAD.b} stroke="currentColor" stroke-width="1" stroke-dasharray="2 3" class="text-printer-ink/40 dark:text-printer-ink-dark/40" />
      {#each paths as s (s.name)}
        <circle cx={tx(hover)} cy={ty(s.values[hover] ?? 0)} r="3" class={s.tone === "accent" ? "fill-printer-accent dark:fill-printer-accent-dark" : "fill-printer-ink-light dark:fill-printer-ink-dark"} />
      {/each}
    {/if}
    {#each xTicks as tick (tick.label)}
      <text x={tick.x} y={height - 6} text-anchor="middle" font-size="9" class="fill-printer-ink-light dark:fill-printer-ink-dark/40 font-mono">
        {tick.label}
      </text>
    {/each}
  </svg>

  {#if hover !== null}
    <div
      class="pointer-events-none absolute top-1 z-10 -translate-x-1/2 border border-printer-ink/15 dark:border-printer-ink-dark/15 bg-printer-paper dark:bg-printer-paper-dark thermal-texture px-2 py-1 shadow-md"
      style:left={`${(tx(hover) / W) * 100}%`}
    >
      <div class="font-mono text-[9px] tracking-widest uppercase text-printer-ink-light dark:text-printer-ink-dark/50">
        {labels[hover]}
      </div>
      {#each series as s (s.name)}
        <div class="flex items-center gap-1.5 font-mono text-[10px] text-printer-ink dark:text-printer-ink-dark">
          <span class={s.tone === "accent" ? "inline-block h-1.5 w-1.5 bg-printer-accent dark:bg-printer-accent-dark" : "inline-block h-1.5 w-1.5 bg-printer-ink-light dark:bg-printer-ink-dark"}></span>
          {s.name} {format(s.values[hover] ?? 0)}
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  @keyframes lc-dash {
    from {
      stroke-dashoffset: 1;
    }
    to {
      stroke-dashoffset: 0;
    }
  }
  @keyframes lc-soft {
    from {
      opacity: 0;
    }
  }
  .lc-draw {
    stroke-dasharray: 1;
    animation: lc-dash 1000ms cubic-bezier(0.22, 1, 0.36, 1) both;
  }
  .lc-area {
    animation: lc-soft 800ms 400ms both;
  }
  .lc-fade {
    animation: lc-soft 900ms 250ms both;
  }
  @media (prefers-reduced-motion: reduce) {
    .lc-draw,
    .lc-area,
    .lc-fade {
      animation: none;
      stroke-dashoffset: 0;
    }
  }
</style>
