<script lang="ts">
  import type { TocItem } from "$lib/toc";

  let {
    items,
    title,
    lang,
    backHref,
  }: { items: TocItem[]; title: string; lang: string; backHref: string } =
    $props();

  /**
   * 纸面容器 .printer-paper-wrap 带 clip-path，会裁剪包括 fixed 在内的所有
   * 后代；目录要显示在纸面外的页面背景上，因此挂载后移动到 body 下。
   */
  function portal(node: HTMLElement) {
    document.body.appendChild(node);
    return {
      destroy() {
        node.remove();
      },
    };
  }

  interface RailLandmark {
    kind: "landmark";
    id: string | undefined; // undefined = 标题（回顶部）
    variant: "title" | "heading";
    label: string;
  }
  interface RailTick {
    kind: "tick";
  }
  type RailNode = RailLandmark | RailTick;

  // 与 cali.so 相同的节律：每个地标之间恰好隔 3 条安静刻度（TICKS_BETWEEN_LANDMARKS）
  const TICKS_BETWEEN_LANDMARKS = 3;
  const rail: RailNode[] = $derived.by(() => [
    { kind: "landmark", id: undefined, variant: "title", label: title },
    ...items.flatMap((item) => [
      ...Array.from({ length: TICKS_BETWEEN_LANDMARKS }, () => ({
        kind: "tick" as const,
      })),
      { kind: "landmark" as const, id: item.id, variant: "heading" as const, label: item.text },
    ]),
  ]);

  // 当前滚动位置对应的地标 id（undefined = 顶部/标题，与源站一致）
  let activeId = $state<string | null>(null);
  let backToTopVisible = $state(false);

  $effect(() => {
    if (items.length === 0) return;
    const targets = items
      .map((item) => document.getElementById(item.id))
      .filter((el) => el !== null);
    if (targets.length === 0) return;

    const TARGET_OFFSET = 100;
    let raf = 0;

    const update = () => {
      raf = 0;
      const doc = document.documentElement;
      let current: string | null = null;
      for (const el of targets) {
        if (el.getBoundingClientRect().top <= TARGET_OFFSET + 1) current = el.id;
        else break;
      }
      if (window.scrollY <= 1) current = null;
      if (window.innerHeight + window.scrollY >= doc.scrollHeight - 2) {
        current = targets[targets.length - 1].id;
      }
      backToTopVisible = window.scrollY >= window.innerHeight * 0.75;
      activeId = current;
    };
    const schedule = () => {
      if (!raf) raf = requestAnimationFrame(update);
    };

    update();
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    return () => {
      window.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", schedule);
      if (raf) cancelAnimationFrame(raf);
    };
  });

  /** 与源站一致：地标跳转为瞬时定位（无平滑），标题回顶部 */
  function visit(node: RailLandmark, event: MouseEvent) {
    event.preventDefault();
    if (!node.id) {
      window.scrollTo({ top: 0 });
      history.replaceState(null, "", window.location.pathname);
      activeId = null;
      return;
    }
    const target = document.getElementById(node.id);
    if (!target) return;
    window.scrollTo({
      top: window.scrollY + target.getBoundingClientRect().top - 100,
    });
    history.replaceState(null, "", `#${node.id}`);
    activeId = node.id;
  }

  function backToTop() {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
</script>

{#if items.length > 0}
  <!-- 固定在纸面外的页面左侧（参考 cali.so），仅超宽屏显示；portal 至 body 以逃出纸面 clip-path -->
  <nav use:portal class="post-toc" aria-label="文章地图">
    <div class="post-toc-utilities">
      <a href={backHref} class="post-toc-utility">
        <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
          <g
            fill="none"
            stroke="currentColor"
            stroke-width="1.5"
            stroke-linecap="round"
            stroke-linejoin="round"
          >
            <path
              d="m1.25,5.25h7c1.381,0,2.5,1.119,2.5,2.5h0c0,1.381-1.119,2.5-2.5,2.5h-1.25"
            />
            <polyline points="4.25 8.5 1 5.25 4.25 2" />
          </g>
        </svg>
        <span>{lang === "zh" ? "写作" : "Writing"}</span>
      </a>
    </div>
    <div class="post-toc-nodes">
      {#each rail as node, index (index)}
        <div
          class="post-toc-node"
          style:--enter-delay="{Math.round(Math.abs(index - (rail.length - 1) / 2) * 12)}ms"
        >
          {#if node.kind === "tick"}
            <span class="post-toc-tick" aria-hidden="true"></span>
          {:else}
            <a
              href={node.id ? `#${node.id}` : "/"}
              data-variant={node.variant}
              aria-current={activeId === node.id || (activeId === null && !node.id) ? "location" : undefined}
              aria-label={node.label}
              title={node.label}
              onclick={(event) => visit(node, event)}
            >
              <span class="post-toc-tick" aria-hidden="true"></span>
              <span class="post-toc-label">{node.label}</span>
            </a>
          {/if}
        </div>
      {/each}
    </div>
    <div class="post-toc-utilities">
      <button
        type="button"
        class="post-toc-utility post-toc-back-to-top"
        aria-hidden={!backToTopVisible}
        tabindex={backToTopVisible ? 0 : -1}
        data-visible={backToTopVisible || undefined}
        onclick={backToTop}
      >
        <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
          <g
            fill="none"
            stroke="currentColor"
            stroke-width="1.5"
            stroke-linecap="round"
            stroke-linejoin="round"
          >
            <path d="M11.25 6C11.25 3.1005 8.89949 0.75 6 0.75C3.1005 0.75 0.75 3.10051 0.75 6" />
            <path d="M6 11.25L6 3.75" />
            <path d="M3.75 6L6 3.75L8.25 6" />
          </g>
        </svg>
        <span>{lang === "zh" ? "顶部" : "Top"}</span>
      </button>
    </div>
  </nav>
{/if}
