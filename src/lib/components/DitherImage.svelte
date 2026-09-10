<script lang="ts">
  /**
   * 按 CaliCastle/cali.so 的 components/dither-veil.tsx 移植（无音效）：
   * - mode="dither"：整幅 Bayer 有序抖动画，hover 淡出显影成照片（列表缩略图）
   * - mode="collage"：入场时 voronoi 色块按莫尔斯节律闪烁一次（reduced-motion 跳过），
   *   之后点击在「照片 ⇄ 整幅抖动画」之间做 Bayer 溶解切换（16 阈值步进、可打断）
   * 墨色/纸色由 --dither-ink / --dither-paper 提供（见 app.css）。
   */
  let {
    src,
    alt = "",
    mode = "dither",
  }: { src: string; alt?: string; mode?: "dither" | "collage" } = $props();

  let canvas: HTMLCanvasElement | undefined = $state();
  let img: HTMLImageElement | undefined = $state();

  const PIXEL = 2.5; // 每个 dither 格的 CSS 像素边长
  const ASCII_CELL = 7; // ascii 字符格边长
  // 亮度 → 字符的密度梯度（源站同款字符表）
  const RAMP = " -li+tcsea";
  const BAYER_ORDER = [
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
  ];
  const BAYER = BAYER_ORDER.map((row) => row.map((v) => (v + 0.5) / 16));

  const SEEDS = 33;
  // 13 个 voronoi 色块，交替两种印刷风格（dither / ascii）
  const PATCH_STYLES: Array<1 | 2> = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1];
  // 入场闪烁节律：· — · · –（先一次全覆盖，再逐渐收束）
  const DOT = 70;
  const DASH = 160;
  const LONG = 260;
  const BEAT_GAP = 60;
  const BEATS: Array<[number, number[]]> = [
    [DOT, [0, 1, 2]],
    [LONG, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]],
    [DOT, [8, 9, 10, 11]],
    [DOT, [0, 2, 4, 6]],
    [DASH, [3, 5, 7, 9]],
  ];
  const DISSOLVE_STEP_MS = 38;

  function mulberry(seed: number): () => number {
    return () => {
      seed |= 0;
      seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function hashOf(s: string): number {
    let h = 0;
    for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
    return Math.abs(h);
  }

  $effect(() => {
    if (!canvas || !img) return;
    const canvasEl = canvas;
    const imgEl = img;
    const maybeCtx = canvasEl.getContext("2d");
    if (!maybeCtx) return;
    const ctx = maybeCtx;

    let cancelled = false;
    let retries = 0;
    const timers: ReturnType<typeof setTimeout>[] = [];
    let ro: ResizeObserver | undefined;
    let themeObserver: MutationObserver | undefined;

    function colors() {
      const cs = getComputedStyle(canvasEl);
      return {
        paper: cs.getPropertyValue("--dither-paper").trim() || "#fffef9",
        ink: cs.getPropertyValue("--dither-ink").trim() || "#2c2824",
      };
    }

    // 降采样到 cols×rows 并取亮度，按 5%~95% 分位归一（源站同款）
    function levels(cols: number, rows: number) {
      const off = document.createElement("canvas");
      off.width = cols;
      off.height = rows;
      const octx = off.getContext("2d", { willReadFrequently: true });
      if (!octx) return null;
      octx.drawImage(imgEl, 0, 0, cols, rows);
      let data: Uint8ClampedArray;
      try {
        data = octx.getImageData(0, 0, cols, rows).data;
      } catch {
        return null; // 跨域像素污染画布：放弃 dither，露出原图兜底
      }
      let sampled = false;
      for (let i = 3; i < data.length; i += 4) {
        if (data[i] > 0) {
          sampled = true;
          break;
        }
      }
      if (!sampled) return null; // 像素尚未解码：交由重试
      const lums = new Float32Array(cols * rows);
      for (let i = 0; i < lums.length; i++) {
        const j = i * 4;
        lums[i] = (0.2126 * data[j] + 0.7152 * data[j + 1] + 0.0722 * data[j + 2]) / 255;
      }
      const sorted = Float32Array.from(lums).sort();
      const lo = sorted[Math.floor(sorted.length * 0.05)];
      const hi = sorted[Math.floor(sorted.length * 0.95)];
      const range = Math.max(0.05, hi - lo);
      return { lums, norm: (v: number) => Math.min(1, Math.max(0, (v - lo) / range)) };
    }

    function sizeCanvas(rect: DOMRect) {
      const dpr = window.devicePixelRatio || 1;
      canvasEl.width = Math.round(rect.width * dpr);
      canvasEl.height = Math.round(rect.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, rect.width, rect.height);
    }

    function drawDitherFull(rect: DOMRect): boolean {
      const cols = Math.max(1, Math.round(rect.width / PIXEL));
      const rows = Math.max(1, Math.round(rect.height / PIXEL));
      const sample = levels(cols, rows);
      if (!sample) return false;
      const { paper, ink } = colors();
      const cw = rect.width / cols;
      const ch = rect.height / rows;
      ctx.fillStyle = paper;
      ctx.fillRect(0, 0, rect.width, rect.height);
      ctx.fillStyle = ink;
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const lum = sample.norm(sample.lums[r * cols + c]);
          if (1 - lum > BAYER[r % 4][c % 4]) ctx.fillRect(c * cw, r * ch, cw, ch);
        }
      }
      return true;
    }

    // ——— collage 入场闪烁 ———
    interface Grid {
      cols: number;
      rows: number;
      cell: number;
      assign: Int16Array; // 每格所属色块 id，-1 = 照片
      sample: NonNullable<ReturnType<typeof levels>>;
    }

    function buildGrid(
      rect: DOMRect,
      cell: number,
      patchOf: (x: number, y: number) => number,
    ): Grid | null {
      const cols = Math.max(1, Math.round(rect.width / cell));
      const rows = Math.max(1, Math.round(rect.height / cell));
      const sample = levels(cols, rows);
      if (!sample) return null;
      const assign = new Int16Array(cols * rows);
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < cols; c++)
          assign[r * cols + c] = patchOf((c + 0.5) * cell, (r + 0.5) * cell);
      return { cols, rows, cell, assign, sample };
    }

    function drawPatches(rect: DOMRect, dither: Grid, ascii: Grid, visible: boolean[]) {
      const { paper, ink } = colors();
      ctx.clearRect(0, 0, rect.width, rect.height);
      for (let r = 0; r < dither.rows; r++) {
        for (let c = 0; c < dither.cols; c++) {
          const id = dither.assign[r * dither.cols + c];
          if (id < 0 || !visible[id] || PATCH_STYLES[id] !== 1) continue;
          const x = c * dither.cell;
          const y = r * dither.cell;
          ctx.fillStyle = paper;
          ctx.fillRect(x, y, dither.cell + 0.5, dither.cell + 0.5);
          const lum = dither.sample.norm(dither.sample.lums[r * dither.cols + c]);
          if (1 - lum > BAYER[r % 4][c % 4]) {
            ctx.fillStyle = ink;
            ctx.fillRect(x, y, dither.cell, dither.cell);
          }
        }
      }
      ctx.font = `${ASCII_CELL + 1}px ui-monospace, monospace`;
      ctx.textBaseline = "top";
      for (let r = 0; r < ascii.rows; r++) {
        for (let c = 0; c < ascii.cols; c++) {
          const id = ascii.assign[r * ascii.cols + c];
          if (id < 0 || !visible[id] || PATCH_STYLES[id] !== 2) continue;
          const x = c * ascii.cell;
          const y = r * ascii.cell;
          ctx.fillStyle = paper;
          ctx.fillRect(x, y, ascii.cell + 0.5, ascii.cell + 0.5);
          const lum = ascii.sample.norm(ascii.sample.lums[r * ascii.cols + c]);
          const chr = RAMP[Math.min(RAMP.length - 1, Math.round((1 - lum) * (RAMP.length - 1)))];
          if (chr !== " ") {
            ctx.fillStyle = ink;
            ctx.fillText(chr, x, y);
          }
        }
      }
    }

    let prepared: {
      rect: DOMRect;
      dither: Grid;
      ascii: Grid;
      full: { cols: number; rows: number; sample: NonNullable<ReturnType<typeof levels>> };
    } | null = null;

    function prepare(rect: DOMRect) {
      const rand = mulberry(hashOf(src));
      const seeds: Array<{ x: number; y: number }> = [];
      for (let i = 0; i < SEEDS; i++) seeds.push({ x: rand() * rect.width, y: rand() * rect.height });
      const patchSeeds: number[] = [];
      while (patchSeeds.length < PATCH_STYLES.length) {
        const pick = Math.floor(rand() * SEEDS);
        if (!patchSeeds.includes(pick)) patchSeeds.push(pick);
      }
      const patchOf = (x: number, y: number): number => {
        let best = 0;
        let bestD = Infinity;
        for (let i = 0; i < seeds.length; i++) {
          const dx = x - seeds[i].x;
          const dy = y - seeds[i].y;
          const d = dx * dx + dy * dy;
          if (d < bestD) {
            bestD = d;
            best = i;
          }
        }
        return patchSeeds.indexOf(best);
      };
      const dither = buildGrid(rect, PIXEL, patchOf);
      const ascii = buildGrid(rect, ASCII_CELL, patchOf);
      const cols = Math.max(1, Math.round(rect.width / PIXEL));
      const rows = Math.max(1, Math.round(rect.height / PIXEL));
      const fullSample = levels(cols, rows);
      prepared =
        dither && ascii && fullSample
          ? { rect, dither, ascii, full: { cols, rows, sample: fullSample } }
          : null;
    }

    let playing = false;
    function play() {
      if (playing || !prepared) return;
      if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
      playing = true;
      const { rect, dither, ascii } = prepared;
      let t = 0;
      for (const [dur, patches] of BEATS) {
        const visible = PATCH_STYLES.map((_, i) => patches.includes(i));
        const at = t;
        timers.push(setTimeout(() => drawPatches(rect, dither, ascii, visible), at));
        timers.push(setTimeout(() => ctx.clearRect(0, 0, rect.width, rect.height), at + dur));
        t += dur + BEAT_GAP;
      }
      timers.push(
        setTimeout(() => {
          playing = false;
        }, t),
      );
    }

    // ——— Bayer 溶解：点击在照片 ⇄ 抖动画之间切换（可打断） ———
    let level = 0;
    let target = 0;
    let stepTimer: ReturnType<typeof setTimeout> | null = null;

    function dissolveStep(order: number, toArt: boolean) {
      if (!prepared) return;
      const { rect, full } = prepared;
      const { cols, rows, sample } = full;
      const { paper, ink } = colors();
      const cw = rect.width / cols;
      const ch = rect.height / rows;
      for (let r = 0; r < rows; r++) {
        const rowOrder = BAYER_ORDER[r % 4];
        for (let c = 0; c < cols; c++) {
          if (rowOrder[c % 4] !== order) continue;
          const x = c * cw;
          const y = r * ch;
          if (!toArt) {
            ctx.clearRect(x, y, cw + 0.5, ch + 0.5);
            continue;
          }
          const lum = sample.norm(sample.lums[r * cols + c]);
          ctx.fillStyle = 1 - lum > BAYER[r % 4][c % 4] ? ink : paper;
          ctx.fillRect(x, y, cw + 0.5, ch + 0.5);
        }
      }
    }

    function tick() {
      stepTimer = null;
      if (level === target) return;
      if (level < target) {
        dissolveStep(level, true);
        level++;
      } else {
        dissolveStep(level - 1, false);
        level--;
      }
      if (level !== target) {
        stepTimer = setTimeout(tick, DISSOLVE_STEP_MS);
      } else if (level === 0 && prepared) {
        // 逐格 clearRect 会在小数格边界留下抗锯齿残影 —— 整幅清除一次归净
        ctx.clearRect(0, 0, prepared.rect.width, prepared.rect.height);
      }
    }

    function onToggle() {
      if (playing || !prepared) return;
      target = target === 16 ? 0 : 16;
      if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
        const { rect } = prepared;
        if (stepTimer) clearTimeout(stepTimer);
        stepTimer = null;
        if (target === 16) drawDitherFull(rect);
        else ctx.clearRect(0, 0, rect.width, rect.height);
        level = target;
        return;
      }
      if (!stepTimer) tick(); // 首步立即落地，点击即有反馈
    }

    function render(): boolean {
      const rect = canvasEl.getBoundingClientRect();
      if (rect.width < 4) return true;
      if (mode === "dither") {
        sizeCanvas(rect);
        return drawDitherFull(rect);
      }
      const first = !prepared;
      sizeCanvas(rect);
      prepare(rect);
      if (!prepared) return false;
      if (first) play();
      else if (target === 16) {
        drawDitherFull(rect);
        level = 16;
      } else {
        level = 0;
      }
      return true;
    }

    function attemptRender() {
      if (cancelled) return;
      if (!render() && retries < 4) {
        retries += 1;
        timers.push(setTimeout(attemptRender, 150 * retries));
      }
    }

    function startRender() {
      attemptRender();
      if (typeof imgEl.decode === "function") imgEl.decode().then(attemptRender, () => {});
    }

    if (imgEl.complete && imgEl.naturalWidth > 0) startRender();
    else imgEl.addEventListener("load", startRender, { once: true });

    ro = new ResizeObserver(() => {
      if (imgEl.complete && imgEl.naturalWidth > 0) attemptRender();
    });
    ro.observe(canvasEl);

    // 主题切换时重取 --dither-* 颜色重绘
    themeObserver = new MutationObserver(attemptRender);
    themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    });

    // collage：点击封面切换照片 ⇄ 抖动画
    let host: HTMLElement | undefined;
    if (mode === "collage") {
      host = canvasEl.closest<HTMLElement>(".post-cover-frame") ?? undefined;
      if (host) {
        host.addEventListener("click", onToggle);
        host.style.cursor = "pointer";
      }
    }

    return () => {
      cancelled = true;
      imgEl.removeEventListener("load", startRender);
      ro?.disconnect();
      themeObserver?.disconnect();
      if (host) host.removeEventListener("click", onToggle);
      if (stepTimer) clearTimeout(stepTimer);
      for (const t of timers) clearTimeout(t);
    };
  });
</script>

{#if mode === "collage"}
  <img bind:this={img} {src} {alt} class="post-cover-img" />
  <canvas bind:this={canvas} class="post-cover-veil" aria-hidden="true"></canvas>
{:else}
  <img bind:this={img} {src} {alt} loading="lazy" class="post-thumb-layer" />
  <canvas
    bind:this={canvas}
    class="post-thumb-layer post-thumb-veil"
    aria-hidden="true"
  ></canvas>
{/if}
