<script lang="ts">
  /**
   * Desktop-only light switch: a pull cord on the right edge of the screen
   * toggles dark mode via a Braun/TE-style pendant lamp hanging on the left —
   * a matte plastic drum housing with an orange trim ring and a frosted
   * diffuser globe, matching the printer-shell design language.
   *
   * Pulling the cord in light mode blacks out the whole screen, then the
   * bulb sputters awake revealing the dark theme. Pulling again makes the
   * bulb flicker and die, and daylight fades back in.
   *
   * The bulb / room-shading layers are shown purely via CSS (`dark` class +
   * `lg` breakpoint), so they are correct on first paint with no JS.
   */
  let {
    isDark,
    ontoggle,
    lang = "en",
  }: {
    isDark: boolean;
    /** Called mid-transition, while the screen is fully black. */
    ontoggle: (dark: boolean) => void;
    lang?: string;
  } = $props();

  let busy = $state(false);
  let pulled = $state(false);
  let blackoutElement: HTMLDivElement | undefined = $state();

  // Easter egg: clicking the lit bulb chips it. Each tap drives a fresh spider
  // crack into the glass (jagged, branching, drawn in ~a tenth of a second),
  // and the third tap shatters the globe for real — a white pop, then glass
  // chunks burst out and rain down the screen.
  let cracks = $state(0);
  let broken = $derived(cracks >= 3);

  let crackPaths = $state<string[]>([]);

  let globeEl: SVGCircleElement | undefined = $state();
  let bulbSwayEl: HTMLDivElement | undefined = $state();
  let shardLayer: HTMLDivElement | undefined = $state();

  let title = $derived(
    isDark
      ? lang === "zh"
        ? "拉绳关灯"
        : "Pull the cord to turn the light off"
      : lang === "zh"
        ? "拉绳开灯"
        : "Pull the cord to turn the light on",
  );

  // Blackout opacity over time — an old bulb sputtering awake…
  const FLICKER_ON: Keyframe[] = [
    { opacity: 1, offset: 0 },
    { opacity: 1, offset: 0.08 },
    { opacity: 0.45, offset: 0.14 },
    { opacity: 0.9, offset: 0.22 },
    { opacity: 0.15, offset: 0.32 },
    { opacity: 0.75, offset: 0.4 },
    { opacity: 0.85, offset: 0.47 },
    { opacity: 0.08, offset: 0.56 },
    { opacity: 0.5, offset: 0.66 },
    { opacity: 0, offset: 0.76 },
    { opacity: 0.18, offset: 0.86 },
    { opacity: 0, offset: 1 },
  ];

  function wait(ms: number) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  async function run(
    element: Element,
    keyframes: Keyframe[],
    options: KeyframeAnimationOptions,
  ) {
    const animation = element.animate(keyframes, { fill: "forwards", ...options });
    try {
      await animation.finished;
    } catch {
      // Cancelled mid-flight (e.g. rapid re-toggle) — the caller re-syncs.
    }
  }

  async function pull() {
    if (busy || !blackoutElement) return;
    busy = true;
    const toDark = !isDark;
    blackoutElement.getAnimations().forEach((animation) => animation.cancel());

    // Tug the cord down.
    pulled = true;
    await wait(170);

    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      ontoggle(toDark);
      if (toDark && cracks >= 3) cracks = 0; // a fresh bulb when night mode returns after breaking
      resetShards();
      pulled = false;
      busy = false;
      return;
    }

    if (toDark) {
      // Lights out — the room goes pitch black…
      await run(blackoutElement, [{ opacity: 0 }, { opacity: 1 }], {
        duration: 150,
        easing: "ease-in",
      });
      ontoggle(true); // swap the theme under cover of darkness
      if (cracks >= 3) cracks = 0; // fresh bulb when night mode returns after breaking
      resetShards();
      pulled = false; // cord springs back up
      await wait(140); // a beat of total darkness
      // …then the old bulb sputters awake.
      await run(blackoutElement, FLICKER_ON, { duration: 780, easing: "linear" });
    } else {
      // Daylight returns instantly, then the bulb clicks off and fades away
      // (the CSS transitions on the bulb/shading layers do the "lamp dying").
      ontoggle(false);
      await wait(120);
      pulled = false;
      await wait(650); // keep the switch busy until the bulb has been hoisted away
    }

    busy = false;
  }

  function onBulbClick() {
    if (!isDark || broken) return;
    cracks += 1;
    if (cracks === 3) {
      shatter();
      return;
    }
    // Each tap lands somewhere new on the glass and webs out from there.
    const [ix, iy] = impactPoint(50, 128, 23);
    crackPaths = [...crackPaths, ...crackBatch(ix, iy, 23)];
  }

  // ------------------------------------------------------------------
  // Procedural glass cracks — jagged, branching webs from a tapped point.
  // ------------------------------------------------------------------
  function rnd(a: number, b: number) {
    return a + Math.random() * (b - a);
  }

  function impactPoint(cx: number, cy: number, r: number): [number, number] {
    const a = rnd(0, Math.PI * 2);
    const d = rnd(2, r * 0.38);
    return [cx + Math.cos(a) * d, cy + Math.sin(a) * d];
  }

  function crackLine(ix: number, iy: number, angle: number, length: number): string {
    const segs = 5 + Math.floor(Math.random() * 4);
    const dx = Math.cos(angle);
    const dy = Math.sin(angle);
    const pts: string[] = [`M${ix.toFixed(1)} ${iy.toFixed(1)}`];
    let px = ix;
    let py = iy;
    for (let i = 1; i <= segs; i++) {
      const t = i / segs;
      px += dx * (length / segs);
      py += dy * (length / segs);
      const wob = (Math.random() - 0.5) * (1.5 + t * 9); // wander widens outward
      pts.push(`${(px - dy * wob).toFixed(1)} ${(py + dx * wob).toFixed(1)}`);
    }
    return pts.join(" L");
  }

  function crackBatch(ix: number, iy: number, r: number): string[] {
    const out: string[] = [];
    // Dense little crater right at the tap point.
    const crater = 4 + Math.floor(Math.random() * 3);
    for (let i = 0; i < crater; i++) {
      out.push(crackLine(ix, iy, rnd(0, Math.PI * 2), rnd(2.5, 7)));
    }
    // Main radial cracks, some reaching the rim.
    const mains = 5 + Math.floor(Math.random() * 2);
    for (let i = 0; i < mains; i++) {
      const a = (i / mains) * Math.PI * 2 + rnd(-0.3, 0.3);
      const len = rnd(r * 0.5, r * 0.82);
      out.push(crackLine(ix, iy, a, len));
      if (Math.random() < 0.55) {
        const frac = rnd(0.35, 0.72);
        const bx = ix + Math.cos(a) * len * frac;
        const by = iy + Math.sin(a) * len * frac;
        const ba = a + rnd(0.5, 1.15) * (Math.random() < 0.5 ? 1 : -1);
        out.push(crackLine(bx, by, ba, rnd(r * 0.2, r * 0.45)));
      }
    }
    return out;
  }

  // ------------------------------------------------------------------
  // Shatter — screen-space glass that really rains down.
  // ------------------------------------------------------------------
  type Shard = {
    el: HTMLDivElement;
    x: number;
    y: number;
    vx: number;
    vy: number;
    rot: number;
    vrot: number;
    size: number;
    resting: number;
    born: number;
  };

  const SHARD_POLYS = [
    "50% 0, 96% 28%, 72% 100%, 14% 78%",
    "30% 0, 100% 22%, 78% 100%, 6% 82%",
    "48% 0, 100% 38%, 62% 100%, 0 60%",
    "22% 0, 88% 0, 100% 72%, 40% 100%",
    "50% 4%, 92% 44%, 60% 100%, 8% 72%",
    "12% 0, 100% 30%, 70% 96%, 0 60%",
    "52% 0, 100% 56%, 44% 100%, 0 36%",
  ];

  let shards: Shard[] = [];
  let shardRaf = 0;
  let shardLastT = 0;

  function shatter() {
    if (!globeEl || !shardLayer) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    const rect = globeEl.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;

    // A violent jitter for a beat — the flash hides the pop.
    bulbSwayEl?.classList.add("boom");
    window.setTimeout(() => bulbSwayEl?.classList.remove("boom"), 330);

    // White pop at the break point.
    const flash = document.createElement("div");
    flash.className = "light-flash";
    flash.style.left = `${cx - 34}px`;
    flash.style.top = `${cy - 34}px`;
    shardLayer.appendChild(flash);
    flash.animate(
      [
        { transform: "scale(0.4)", opacity: 0.95 },
        { transform: "scale(1.7)", opacity: 0 },
      ],
      { duration: 340, easing: "cubic-bezier(0.1, 0.8, 0.3, 1)", fill: "forwards" },
    );
    window.setTimeout(() => flash.remove(), 380);

    // Glass chunks burst outward, then gravity takes them to the floor.
    for (let i = 0; i < 24; i++) {
      const size = 7 + Math.random() * 11;
      const el = document.createElement("div");
      el.className = "light-shard";
      el.style.width = `${size}px`;
      el.style.height = `${(size * (0.8 + Math.random() * 0.6)).toFixed(1)}px`;
      el.style.clipPath = `polygon(${SHARD_POLYS[i % SHARD_POLYS.length]})`;
      shardLayer.appendChild(el);
      const ang = Math.random() * Math.PI * 2;
      const speed = 60 + Math.random() * 210;
      shards.push({
        el,
        x: cx - size / 2,
        y: cy - size / 2,
        vx: Math.cos(ang) * speed,
        vy: Math.sin(ang) * speed * 0.7 - 60 - Math.random() * 130,
        rot: Math.random() * 360,
        vrot: (Math.random() - 0.5) * 720,
        size,
        resting: 0,
        born: performance.now(),
      });
    }

    if (!shardRaf) {
      shardLastT = 0;
      shardRaf = requestAnimationFrame(shardStep);
    }
  }

  function shardStep(now: number) {
    const dt = shardLastT ? Math.min(0.04, (now - shardLastT) / 1000) : 0.016;
    shardLastT = now;
    const GRAVITY = 1500;
    const FLOOR = window.innerHeight - 8;

    shards = shards.filter((s) => s.el.isConnected);
    for (const s of shards) {
      s.vy += GRAVITY * dt;
      s.vx *= Math.exp(-0.3 * dt);
      s.x += s.vx * dt;
      s.y += s.vy * dt;
      s.rot += s.vrot * dt;

      if (s.y + s.size >= FLOOR && s.vy > 0) {
        s.y = FLOOR - s.size;
        s.vy = -s.vy * (0.32 + Math.random() * 0.12);
        s.vx *= 0.68;
        s.vrot *= 0.6;
      }

      if (s.y + s.size >= FLOOR - 1 && Math.abs(s.vy) < 14) {
        s.resting += dt;
      } else {
        s.resting = 0;
      }

      let opacity = 1;
      if (s.resting > 0.85) {
        const f = Math.min(1, (s.resting - 0.85) / 0.65);
        opacity = 1 - f;
        if (f >= 1) {
          s.el.remove();
          continue;
        }
      } else if (performance.now() - s.born > 4600) {
        s.el.remove();
        continue;
      }

      s.el.style.transform = `translate(${s.x.toFixed(1)}px, ${s.y.toFixed(1)}px) rotate(${s.rot.toFixed(1)}deg)`;
      s.el.style.opacity = opacity.toFixed(2);
    }
    shards = shards.filter((s) => s.el.isConnected);

    if (shards.length > 0) {
      shardRaf = requestAnimationFrame(shardStep);
    } else {
      shardRaf = 0;
    }
  }

  // A fresh bulb next time: sweep up leftover shards/flash and old cracks.
  function resetShards() {
    shardLayer
      ?.querySelectorAll(".light-shard, .light-flash")
      .forEach((el) => el.remove());
    shards = [];
    crackPaths = [];
    if (shardRaf) {
      cancelAnimationFrame(shardRaf);
      shardRaf = 0;
    }
  }
</script>

<!-- Room shading — the further from the bulb, the darker the page -->
<div class="lamp-shade hidden lg:block fixed inset-0 z-[90] pointer-events-none" aria-hidden="true"></div>
<div
  class="lamp-warmth hidden lg:block fixed inset-0 z-[90] pointer-events-none"
  class:broken={broken}
  aria-hidden="true"
></div>

<!-- The bulb, hanging left of the content column -->
<div
  class="bulb-root hidden lg:block fixed top-0 left-[max(8px,calc(50%-596px))] z-[95] pointer-events-none"
  class:broken={broken}
  aria-hidden="true"
>
  <div class="bulb-sway relative" bind:this={bulbSwayEl}>
    <div class="bulb-glow absolute"></div>
    <svg width="100" height="190" viewBox="0 0 100 190" fill="none" class="bulb-svg relative block">
      <defs>
        <radialGradient id="light-switch-diffuser" cx="0.5" cy="0.42" r="0.75">
          <stop offset="0" stop-color="#fff8dd" />
          <stop offset="0.5" stop-color="#ffeaae" />
          <stop offset="0.85" stop-color="#ffcd78" />
          <stop offset="1" stop-color="#ffb45a" />
        </radialGradient>
        <!-- horizontal sheen that makes the flat drum read as a cylinder -->
        <linearGradient id="light-switch-cyl" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stop-color="#ffffff" stop-opacity="0.32" />
          <stop offset="0.3" stop-color="#ffffff" stop-opacity="0.06" />
          <stop offset="0.6" stop-color="#ffffff" stop-opacity="0" />
          <stop offset="1" stop-color="#000000" stop-opacity="0.16" />
        </linearGradient>
      </defs>
      <!-- straight rubber pendant cable -->
      <path d="M50 -2 V58" class="bulb-wire" />
      <!-- strain relief where the cable enters the housing -->
      <rect x="46" y="55" width="8" height="14" rx="4" class="bulb-cap" />
      <!-- frosted diffuser globe -->
      <circle
        bind:this={globeEl}
        cx="50"
        cy="128"
        r="23"
        fill="url(#light-switch-diffuser)"
        class="bulb-glass"
      />
      <!-- dead-diffuser overlay, shown the instant the lamp switches off -->
      <circle cx="50" cy="128" r="23" class="bulb-glass-dead" />
      <!-- soft highlight on the globe -->
      <path d="M35.5 122 a17.5 17.5 0 0 1 7.5 -10.5" class="bulb-glint" />
      <!-- cracks that accumulate as the lit bulb is clicked in dark mode —
           procedural spider webs, each drawn in as it lands -->
      {#each crackPaths as d, i (i)}
        <path pathLength="1" d={d} class="bulb-crack" />
      {/each}
      <!-- empty socket, revealed once the globe is gone -->
      <ellipse cx="50" cy="128" rx="22" ry="13" class="bulb-socket" />
      <!-- svelte-ignore a11y_click_events_have_key_events -->
      <!-- svelte-ignore a11y_no_static_element_interactions -->
      <!-- invisible hit target: clickable only while the bulb is lit -->
      <circle
        cx="50"
        cy="128"
        r="23"
        fill="transparent"
        style:pointer-events={isDark && !broken ? "auto" : "none"}
        style:cursor={isDark && !broken ? "pointer" : "default"}
        onclick={onBulbClick}
      />
      <!-- matte plastic housing drum -->
      <path d="M29 104 v-29 a7 7 0 0 1 7 -7 h28 a7 7 0 0 1 7 7 v29 z" class="bulb-shade" />
      <path
        d="M29 104 v-29 a7 7 0 0 1 7 -7 h28 a7 7 0 0 1 7 7 v29 z"
        fill="url(#light-switch-cyl)"
        class="bulb-shade-sheen"
      />
      <!-- vent slots -->
      <path d="M40 79 H60 M40 84.5 H60 M40 90 H60" class="bulb-vent" />
      <!-- accent trim ring where the globe meets the housing -->
      <rect x="29" y="104" width="42" height="4" rx="2" class="bulb-ring" />
    </svg>
  </div>
</div>

<!-- The pull cord, hanging right of the content column -->
<div class="hidden lg:block fixed -top-12 right-[calc(50%-532px)] z-[95] cord-sway">
  <button
    type="button"
    role="switch"
    aria-checked={isDark}
    aria-label={title}
    {title}
    onclick={pull}
    class="cord-group block cursor-pointer focus:outline-none"
    class:pulled
  >
    <svg width="28" height="270" viewBox="0 0 28 270" fill="none" class="block">
      <defs>
        <linearGradient id="light-switch-handle-sheen" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stop-color="#ffffff" stop-opacity="0.35" />
          <stop offset="0.35" stop-color="#ffffff" stop-opacity="0.06" />
          <stop offset="0.65" stop-color="#ffffff" stop-opacity="0" />
          <stop offset="1" stop-color="#000000" stop-opacity="0.18" />
        </linearGradient>
        <clipPath id="light-switch-handle-clip">
          <rect x="6.5" y="202" width="15" height="32" rx="7.5" />
        </clipPath>
      </defs>
      <!-- straight rubber cable -->
      <line x1="14" y1="0" x2="14" y2="196" class="cord-rope" />
      <!-- ferrule where the cable enters the handle -->
      <rect x="11" y="193" width="6" height="10" rx="3" class="cord-ferrule" />
      <!-- matte plastic pull handle -->
      <rect x="6.5" y="202" width="15" height="32" rx="7.5" class="cord-handle" />
      <g clip-path="url(#light-switch-handle-clip)">
        <rect x="6.5" y="207.5" width="15" height="3.5" class="cord-accent" />
        <rect
          x="6.5"
          y="202"
          width="15"
          height="32"
          fill="url(#light-switch-handle-sheen)"
          class="cord-sheen"
        />
        <path d="M10 220 H18 M10 224.5 H18 M10 229 H18" class="cord-grip" />
      </g>
    </svg>
  </button>
</div>

<!-- Shatter layer — screen-space glass chunks fall over the whole viewport -->
<div bind:this={shardLayer} class="shatter-layer" aria-hidden="true"></div>

<!-- Blackout curtain the flicker animations run on -->
<div
  bind:this={blackoutElement}
  class="hidden lg:block fixed inset-0 z-[100] bg-black opacity-0 pointer-events-none"
  aria-hidden="true"
></div>

<style>
  /* ---------------- room lighting ---------------- */

  /* The bulb hangs at max(58px, 50% - 546px), 135px — gradients are centred there.
     Both layers appear instantly when dark mode arrives (hidden behind the
     blackout curtain anyway) but fade away gently when the lamp is switched
     off in daylight. */
  .lamp-shade,
  .lamp-warmth {
    opacity: 0;
    transition: opacity 420ms ease;
  }
  :global(.dark) .lamp-shade,
  :global(.dark) .lamp-warmth {
    opacity: 1;
    transition: none;
  }

  /* Light falls off with distance from the bulb */
  .lamp-shade {
    background: radial-gradient(
      140% 140% at max(58px, 50% - 546px) 135px,
      rgba(0, 0, 0, 0) 0%,
      rgba(0, 0, 0, 0.05) 22%,
      rgba(0, 0, 0, 0.16) 45%,
      rgba(0, 0, 0, 0.3) 72%,
      rgba(0, 0, 0, 0.4) 100%
    );
  }

  /* Warm incandescent spill close to the bulb */
  .lamp-warmth {
    background: radial-gradient(
      620px circle at max(58px, 50% - 546px) 135px,
      rgba(255, 190, 110, 0.14),
      rgba(255, 185, 105, 0.05) 45%,
      transparent 72%
    );
    mix-blend-mode: screen;
  }

  /* ---------------- bulb ---------------- */

  /* Lowered down from above the screen when the lamp arrives; when switched
     off it dies first (grey glass) and is then hoisted back up out of view. */
  .bulb-root {
    visibility: hidden;
    transform: translateY(-240px);
    transition:
      transform 540ms cubic-bezier(0.5, 0, 0.85, 0.6) 180ms,
      visibility 0s linear 720ms;
  }
  :global(.dark) .bulb-root {
    visibility: visible;
    transform: translateY(0);
    transition: transform 680ms cubic-bezier(0.22, 1.18, 0.36, 1) 80ms;
  }

  .bulb-sway {
    transform-origin: 50px 0;
    animation: light-switch-sway 9s ease-in-out infinite;
  }

  .bulb-svg {
    filter: none;
    transition: filter 130ms ease-out;
  }
  :global(.dark) .bulb-svg {
    filter: drop-shadow(0 0 18px rgba(255, 190, 100, 0.4));
  }

  .bulb-glow {
    left: -8px;
    top: 70px;
    width: 116px;
    height: 116px;
    border-radius: 9999px;
    background: radial-gradient(
      closest-side,
      rgba(255, 216, 130, 0.55),
      rgba(255, 190, 100, 0.2) 55%,
      transparent 100%
    );
    filter: blur(6px);
    opacity: 0;
    transition: opacity 130ms ease-out;
  }
  :global(.dark) .bulb-glow {
    opacity: 1;
    animation: light-switch-shimmer 4s ease-in-out infinite;
  }

  .bulb-wire {
    stroke: rgba(60, 56, 50, 0.45);
    stroke-width: 3;
    stroke-linecap: round;
  }
  :global(.dark) .bulb-wire {
    stroke: rgba(205, 212, 232, 0.3);
  }
  .bulb-cap {
    fill: #33363f;
  }
  :global(.dark) .bulb-cap {
    fill: #3c4150;
  }
  /* Cream Braun-plastic drum in daylight, charcoal plastic at night */
  .bulb-shade {
    fill: #ece5d4;
    stroke: rgba(0, 0, 0, 0.14);
    stroke-width: 1;
  }
  :global(.dark) .bulb-shade {
    fill: #272b36;
    stroke: rgba(255, 255, 255, 0.1);
  }
  :global(.dark) .bulb-shade-sheen {
    opacity: 0.55;
  }
  .bulb-vent {
    stroke: rgba(0, 0, 0, 0.16);
    stroke-width: 2;
    stroke-linecap: round;
  }
  :global(.dark) .bulb-vent {
    stroke: rgba(0, 0, 0, 0.5);
  }
  .bulb-ring {
    fill: var(--color-printer-accent);
  }
  :global(.dark) .bulb-ring {
    fill: var(--color-printer-accent-dark);
  }
  .bulb-glass {
    stroke: rgba(255, 226, 170, 0.55);
    stroke-width: 0.8;
  }
  /* Frosted off-white overlay revealed the instant the lamp dies */
  .bulb-glass-dead {
    fill: #e9e5db;
    stroke: rgba(0, 0, 0, 0.12);
    stroke-width: 0.8;
    opacity: 0.95;
    transition: opacity 130ms ease-out;
  }
  :global(.dark) .bulb-glass-dead {
    opacity: 0;
  }
  .bulb-glint {
    stroke: rgba(255, 255, 255, 0.6);
    stroke-width: 1.8;
    stroke-linecap: round;
  }

  /* ---------------- pull cord ---------------- */

  .cord-sway {
    transform-origin: top center;
    animation: light-switch-sway 8s ease-in-out -2s infinite;
  }

  .cord-group {
    /* springy release */
    transition: transform 600ms cubic-bezier(0.18, 1.6, 0.4, 1);
  }
  .cord-group.pulled {
    transform: translateY(46px);
    /* taut, quick pull */
    transition: transform 170ms cubic-bezier(0.55, 0, 0.9, 0.7);
  }
  .cord-group:not(.pulled):hover {
    transform: translateY(4px);
  }

  .cord-rope {
    stroke: rgba(60, 56, 50, 0.5);
    stroke-width: 2.5;
  }
  :global(.dark) .cord-rope {
    stroke: rgba(205, 212, 232, 0.3);
  }
  .cord-ferrule {
    fill: #33363f;
  }
  :global(.dark) .cord-ferrule {
    fill: #3c4150;
  }
  /* Capsule pull handle in the same Braun plastic as the lamp housing */
  .cord-handle {
    fill: #ece5d4;
    stroke: rgba(0, 0, 0, 0.16);
    stroke-width: 1;
  }
  :global(.dark) .cord-handle {
    fill: #272b36;
    stroke: rgba(255, 255, 255, 0.12);
  }
  .cord-accent {
    fill: var(--color-printer-accent);
  }
  :global(.dark) .cord-accent {
    fill: var(--color-printer-accent-dark);
  }
  :global(.dark) .cord-sheen {
    opacity: 0.55;
  }
  .cord-grip {
    stroke: rgba(0, 0, 0, 0.18);
    stroke-width: 1.6;
    stroke-linecap: round;
  }
  :global(.dark) .cord-grip {
    stroke: rgba(0, 0, 0, 0.5);
  }
  .cord-group:focus-visible .cord-handle {
    stroke: var(--color-printer-accent);
    stroke-width: 2;
  }

  @keyframes light-switch-sway {
    0%,
    100% {
      transform: rotate(-0.9deg);
    }
    50% {
      transform: rotate(0.9deg);
    }
  }

  @keyframes light-switch-shimmer {
    0%,
    100% {
      opacity: 1;
    }
    50% {
      opacity: 0.82;
    }
  }

  /* ---------------- cracked-bulb easter egg ---------------- */

  /* Procedural spider cracks — drawn in over a beat via pathLength dash math.
     `pathLength="1"` is set in the markup so 1 dash = the whole path. */
  .bulb-crack {
    fill: none;
    stroke: rgba(115, 100, 75, 0.8);
    stroke-width: 1.4;
    stroke-linecap: round;
    stroke-linejoin: round;
    stroke-dasharray: 1;
    stroke-dashoffset: 1;
    animation: bulb-crack-draw 0.16s ease-out forwards;
  }
  :global(.dark) .bulb-crack {
    stroke: rgba(240, 225, 195, 0.8);
  }
  :global(.dark) .bulb-crack {
    filter: drop-shadow(0 0 1.5px rgba(255, 235, 200, 0.4));
  }

  @keyframes bulb-crack-draw {
    from {
      stroke-dashoffset: 1;
    }
    to {
      stroke-dashoffset: 0;
    }
  }

  /* One violent jitter the instant the globe shatters — takes over the idle
     sway for its brief duration. The `.boom` class is added from JS, hence
     the global selector. */
  :global(.bulb-sway.boom) {
    animation: bulb-boom 300ms cubic-bezier(0.36, 0.07, 0.19, 0.97);
  }

  @keyframes bulb-boom {
    0%,
    100% {
      transform: rotate(0deg);
    }
    12% {
      transform: rotate(-10deg) translateX(-2px);
    }
    28% {
      transform: rotate(6deg) translateX(1px);
    }
    44% {
      transform: rotate(-5deg);
    }
    58% {
      transform: rotate(3deg);
    }
    72% {
      transform: rotate(-1.5deg);
    }
    86% {
      transform: rotate(0.6deg);
    }
  }

  /* The empty socket left behind — dark opening where the globe sat */
  .bulb-socket {
    opacity: 0;
    fill: rgba(40, 38, 44, 0.5);
    transition: opacity 200ms ease;
  }
  :global(.dark) .bulb-socket {
    fill: rgba(10, 12, 18, 0.8);
  }

  /* Once shattered the lamp sheds no light: no halo, no warm spill, and the
     globe is gone — shards flew off, leaving an empty socket. Re-entering
     night mode on the cord restores it. */
  .bulb-root.broken .bulb-glow {
    opacity: 0 !important;
    animation: none !important;
  }
  :global(.dark) .bulb-root.broken .bulb-svg {
    filter: none;
  }
  .bulb-root.broken .bulb-glass,
  .bulb-root.broken .bulb-glass-dead,
  .bulb-root.broken .bulb-glint,
  .bulb-root.broken .bulb-crack {
    opacity: 0 !important;
  }
  .bulb-root.broken .bulb-socket {
    opacity: 1;
  }

  /* ---------------- shatter layer ---------------- */

  .shatter-layer {
    position: fixed;
    inset: 0;
    z-index: 96;
    pointer-events: none;
    overflow: hidden;
  }

  /* Glass chunks — amber frosted chips. Positions come entirely from the JS
     physics; these are created dynamically, hence the global selectors. */
  :global(.light-shard) {
    position: absolute;
    left: 0;
    top: 0;
    will-change: transform, opacity;
    transform-origin: center;
    background: linear-gradient(135deg, #fff3d6 0%, #ffd98c 40%, #ffb95e 100%);
    box-shadow:
      inset 0 0 0 1px rgba(255, 220, 160, 0.5),
      0 2px 8px rgba(0, 0, 0, 0.35);
  }
  :global(.dark .light-shard) {
    box-shadow:
      inset 0 0 0 1px rgba(255, 235, 190, 0.65),
      0 2px 10px rgba(0, 0, 0, 0.5);
  }

  /* White pop at the instant the globe breaks */
  :global(.light-flash) {
    position: absolute;
    width: 68px;
    height: 68px;
    border-radius: 50%;
    background: radial-gradient(
      circle,
      rgba(255, 250, 225, 0.98) 0%,
      rgba(255, 226, 150, 0.55) 45%,
      transparent 72%
    );
    mix-blend-mode: screen;
    will-change: transform, opacity;
  }
  .lamp-warmth.broken {
    opacity: 0 !important;
    transition: none;
  }

  @media (prefers-reduced-motion: reduce) {
    .bulb-sway,
    .cord-sway,
    .bulb-glow {
      animation: none !important;
    }
    .bulb-root {
      transition: none !important;
    }
    /* Cracks still show, just without the draw-in sweep. */
    .bulb-crack {
      animation: none !important;
      stroke-dashoffset: 0;
    }
  }
</style>
