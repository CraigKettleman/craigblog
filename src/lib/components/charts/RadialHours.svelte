<script lang="ts">
  /**
   * 24 小时径向钟：极坐标条形，一眼看出访问高峰时段。
   * 0 点在正上方，顺时针 24 格；峰值格用品牌橙。
   * 悬停走 svg 事件委托（data-h），避免给静态图元挂交互 handler。
   */
  import CountUp from "./CountUp.svelte";

  let {
    hours,
    format = (n: number) => String(n),
    size = 190,
  }: {
    hours: number[];
    format?: (n: number) => string;
    size?: number;
  } = $props();

  let hover = $state<number | null>(null);

  let max = $derived(Math.max(1, ...hours));
  let peak = $derived(hours.indexOf(max));

  // size 由调用方固定传入；集中派生几何常量，避免逐行 ignore
  // 内外半径留出刻度环空间，保证 00/06/12/18 四个刻度不被 viewBox 裁切
  const geo = $derived({ cx: size / 2, cy: size / 2, r0: size * 0.2, rMax: size * 0.4, label: size * 0.4 + 10 });

  function barFor(h: number) {
    const angle = (h / 24) * Math.PI * 2 - Math.PI / 2;
    const len = geo.r0 + (hours[h] / max) * (geo.rMax - geo.r0);
    return {
      x1: geo.cx + Math.cos(angle) * geo.r0,
      y1: geo.cy + Math.sin(angle) * geo.r0,
      x2: geo.cx + Math.cos(angle) * len,
      y2: geo.cy + Math.sin(angle) * len,
    };
  }

  function labelPos(h: number) {
    const angle = (h / 24) * Math.PI * 2 - Math.PI / 2;
    return {
      x: geo.cx + Math.cos(angle) * geo.label,
      y: geo.cy + Math.sin(angle) * geo.label + 3,
    };
  }

  function onMoveSvg(e: MouseEvent) {
    const raw = (e.target as SVGElement).getAttribute?.("data-h");
    hover = raw === null || raw === undefined ? null : Number(raw);
  }
</script>

<div class="relative mx-auto" style:width="{size}px" style:height="{size}px">
  <svg
    viewBox="0 0 {size} {size}"
    class="w-full h-full"
    role="img"
    onmousemove={onMoveSvg}
    onmouseleave={() => (hover = null)}
  >
    <circle cx={geo.cx} cy={geo.cy} r={geo.r0} fill="none" stroke="currentColor" stroke-width="1" stroke-dasharray="2 3" class="text-printer-ink/20 dark:text-printer-ink-dark/20" />
    {#each hours as _, h (h)}
      {@const b = barFor(h)}
      <line
        data-h={h}
        pathLength={1}
        style:animation-delay={`${h * 18}ms`}
        x1={b.x1}
        y1={b.y1}
        x2={b.x2}
        y2={b.y2}
        stroke-width={4}
        class={[
          "rh-draw",
          h === peak
            ? "stroke-printer-accent dark:stroke-printer-accent-dark"
            : "stroke-printer-ink/60 dark:stroke-printer-ink-dark/60",
        ].join(" ")}
        stroke-opacity={hover === null || hover === h ? 1 : 0.35}
      />
    {/each}
    {#each [0, 6, 12, 18] as h (h)}
      {@const p = labelPos(h)}
      <text x={p.x} y={p.y} text-anchor="middle" font-size="8" class="fill-printer-ink-light dark:fill-printer-ink-dark/40 font-mono">
        {String(h).padStart(2, "0")}
      </text>
    {/each}
    <text x={geo.cx} y={geo.cy - 2} text-anchor="middle" font-size="15" font-weight="700" class="fill-printer-ink dark:fill-printer-ink-dark font-mono">
      {#if hover !== null}
        {format(hours[hover])}
      {:else}
        <CountUp value={max} {format} />
      {/if}
    </text>
    <text x={geo.cx} y={geo.cy + 10} text-anchor="middle" font-size="7.5" class="fill-printer-ink-light dark:fill-printer-ink-dark/50 font-mono">
      {hover !== null ? `${String(hover).padStart(2, "0")}:00` : `PEAK ${String(peak).padStart(2, "0")}:00`}
    </text>
  </svg>
</div>

<style>
  @keyframes rh-dash {
    from {
      stroke-dashoffset: 1;
    }
    to {
      stroke-dashoffset: 0;
    }
  }
  .rh-draw {
    stroke-dasharray: 1;
    animation: rh-dash 600ms cubic-bezier(0.22, 1, 0.36, 1) both;
  }
  @media (prefers-reduced-motion: reduce) {
    .rh-draw {
      animation: none;
      stroke-dashoffset: 0;
    }
  }
</style>
