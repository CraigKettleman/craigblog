<script lang="ts">
  /** 数字滚动动效：值变化时 ease-out 计数过渡 */
  let {
    value,
    format = (n: number) => String(n),
    duration = 650,
  }: {
    value: number;
    format?: (n: number) => string;
    duration?: number;
  } = $props();

  let shown = $state(0);
  // 用普通变量记录上次目标值，避免把 shown 读进 effect 造成自触发循环
  let last = 0;

  $effect(() => {
    const from = last;
    const to = value;
    last = to;
    if (from === to) {
      shown = to;
      return;
    }
    const t0 = performance.now();
    let raf = requestAnimationFrame(function step(t: number) {
      const k = Math.min(1, (t - t0) / duration);
      const eased = 1 - (1 - k) ** 3;
      shown = Math.round(from + (to - from) * eased);
      if (k < 1) raf = requestAnimationFrame(step);
    });
    return () => cancelAnimationFrame(raf);
  });
</script>

<span>{format(shown)}</span>
