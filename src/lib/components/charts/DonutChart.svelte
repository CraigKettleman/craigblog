<script lang="ts">
  /**
   * 饼图 / 环图：真实占比，mono 图例，悬停高亮扇区。
   * 配色只用打印机墨色系 + 品牌橙，靠透明度分层，保持热敏纸质感。
   */
  export type Slice = { label: string; value: number };

  let {
    slices,
    variant = "donut",
    centerLabel = "",
    centerValue = "",
    format = (n: number) => String(n),
    size = 168,
    layout = "row",
  }: {
    slices: Slice[];
    variant?: "donut" | "pie";
    centerLabel?: string;
    centerValue?: string;
    format?: (n: number) => string;
    size?: number;
    /** row = 图在左例在右；col = 图在上例在下（窄栏用） */
    layout?: "row" | "col";
  } = $props();

  const TONES: { fill: string; opacity: number }[] = [
    { fill: "var(--color-printer-accent)", opacity: 1 },
    { fill: "var(--color-printer-ink)", opacity: 0.82 },
    { fill: "var(--color-printer-ink-light)", opacity: 0.75 },
    { fill: "var(--color-printer-accent)", opacity: 0.45 },
    { fill: "var(--color-printer-ink)", opacity: 0.4 },
    { fill: "var(--color-printer-ink-light)", opacity: 0.35 },
    { fill: "var(--color-printer-accent)", opacity: 0.22 },
    { fill: "var(--color-printer-ink)", opacity: 0.16 },
  ];

  let hover = $state<number | null>(null);

  function onMoveSvg(e: MouseEvent) {
    const raw = (e.target as SVGElement).getAttribute?.("data-i");
    hover = raw === null || raw === undefined ? null : Number(raw);
  }

  let total = $derived(slices.reduce((a, s) => a + s.value, 0));
  let arcs = $derived.by(() => {
    if (total <= 0) return [] as { d: string; fill: string; opacity: number; slice: Slice; share: number; full: boolean }[];
    const cx = size / 2;
    const cy = size / 2;
    const r1 = size / 2 - 2;
    const r0 = variant === "donut" ? r1 * 0.58 : 0;
    let angle = -Math.PI / 2;
    return slices.map((slice, i) => {
      const share = slice.value / total;
      const a0 = angle;
      const a1 = angle + share * Math.PI * 2;
      angle = a1;
      const tone = TONES[i % TONES.length];
      if (share >= 0.9999) {
        return { d: "", fill: tone.fill, opacity: tone.opacity, slice, share, full: true };
      }
      const large = a1 - a0 > Math.PI ? 1 : 0;
      const pt = (r: number, a: number) => `${(cx + r * Math.cos(a)).toFixed(2)} ${(cy + r * Math.sin(a)).toFixed(2)}`;
      const d =
        r0 > 0
          ? `M${pt(r1, a0)}A${r1} ${r1} 0 ${large} 1 ${pt(r1, a1)}L${pt(r0, a1)}A${r0} ${r0} 0 ${large} 0 ${pt(r0, a0)}Z`
          : `M${cx} ${cy}L${pt(r1, a0)}A${r1} ${r1} 0 ${large} 1 ${pt(r1, a1)}Z`;
      return { d, fill: tone.fill, opacity: tone.opacity, slice, share, full: false };
    });
  });
</script>

<div class={layout === "col" ? "flex flex-col items-center gap-3" : "flex items-center gap-4"}>
  <div class="relative shrink-0" style:width="{size}px" style:height="{size}px">
    <svg viewBox="0 0 {size} {size}" class="w-full h-full" role="img" onmousemove={onMoveSvg} onmouseleave={() => (hover = null)}>
      {#each arcs as arc, i (arc.slice.label)}
        {#if arc.full}
          <circle
            data-i={i}
            class="dc-slice"
            style:transform-origin="{size / 2}px {size / 2}px"
            style:animation-delay={`${i * 90}ms`}
            cx={size / 2}
            cy={size / 2}
            r={(size / 2 - 2 + (variant === "donut" ? (size / 2 - 2) * 0.58 : 0)) / 2}
            fill="none"
            stroke={arc.fill}
            stroke-opacity={hover === null || hover === i ? arc.opacity : arc.opacity * 0.4}
            stroke-width={(size / 2 - 2) * (variant === "donut" ? 0.42 : 1)}
          />
        {:else}
          <path
            d={arc.d}
            data-i={i}
            class="dc-slice"
            style:transform-origin="{size / 2}px {size / 2}px"
            style:animation-delay={`${i * 90}ms`}
            fill={arc.fill}
            fill-opacity={hover === null || hover === i ? arc.opacity : arc.opacity * 0.35}
          />
        {/if}
      {/each}
    </svg>
    {#if variant === "donut"}
      <div class="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
        <span class="font-mono text-lg font-bold leading-none text-printer-ink dark:text-printer-ink-dark">
          {hover !== null ? format(slices[hover].value) : centerValue}
        </span>
        <span class="mt-1 max-w-[70%] truncate font-mono text-[8px] tracking-[0.2em] uppercase text-printer-ink-light dark:text-printer-ink-dark/50">
          {hover !== null ? slices[hover].label : centerLabel}
        </span>
      </div>
    {/if}
  </div>

  <ul class={layout === "col" ? "w-full space-y-1" : "min-w-0 flex-1 space-y-1"}>
    {#each slices as slice, i (slice.label)}
      <li
        class="dc-rise flex items-center gap-2 cursor-default"
        class:opacity-50={hover !== null && hover !== i}
        style:animation-delay={`${200 + i * 60}ms`}
      >
        <span
          class="h-2 w-2 shrink-0"
          style:background={TONES[i % TONES.length].fill}
          style:opacity={TONES[i % TONES.length].opacity}
        ></span>
        <span class="min-w-0 flex-1 truncate font-mono text-[10px] tracking-wider text-printer-ink dark:text-printer-ink-dark">
          {slice.label}
        </span>
        <span class="shrink-0 font-mono text-[10px] text-printer-ink-light dark:text-printer-ink-dark/50">
          {format(slice.value)} · {total > 0 ? ((slice.value / total) * 100).toFixed(0) : 0}%
        </span>
      </li>
    {/each}
  </ul>
</div>

<style>
  @keyframes dc-pop {
    from {
      opacity: 0;
      transform: scale(0.7) rotate(-6deg);
    }
  }
  @keyframes dc-rise {
    from {
      opacity: 0;
      transform: translateY(5px);
    }
  }
  .dc-slice {
    animation: dc-pop 520ms cubic-bezier(0.34, 1.56, 0.64, 1) both;
  }
  .dc-rise {
    animation: dc-rise 450ms cubic-bezier(0.22, 1, 0.36, 1) both;
  }
  @media (prefers-reduced-motion: reduce) {
    .dc-slice,
    .dc-rise {
      animation: none;
    }
  }
</style>
