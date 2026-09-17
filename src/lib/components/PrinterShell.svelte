<script lang="ts">
  import { afterNavigate, beforeNavigate, goto } from "$app/navigation";
  import { page } from "$app/state";
  import { onMount } from "svelte";
  import type { Snippet } from "svelte";
  import type { Dictionary, Language } from "$lib/dictionaries";
  import { isLanguage } from "$lib/dictionaries";
  import { siteMeta } from "$lib/site";
  import { getEditMode } from "$lib/admin-state.svelte";
  import { saveSite } from "$lib/admin-save";
  import PrinterSnail from "./PrinterSnail.svelte";
  import SocialHoverCard from "./SocialHoverCard.svelte";
  import RotaryDial from "./RotaryDial.svelte";
  import LightSwitch from "./LightSwitch.svelte";
  import Stickers from "./Stickers.svelte";
  import Icon from "./Icon.svelte";
  import EditableText from "./admin/EditableText.svelte";

  type ColorMode = "system" | "light" | "dark";

  let {
    lang,
    dictionary,
    admin = false,
    children,
  }: {
    lang: Language;
    dictionary: Dictionary;
    admin?: boolean;
    children: Snippet;
  } = $props();

  let site = $derived(siteMeta(lang));
  let editing = $derived(admin && getEditMode());

  async function saveBrandName(v: string) {
    await saveSite({ brandName: { [lang]: v } });
  }

  async function saveBrandTagline(v: string) {
    await saveSite({ brandTagline: { [lang]: v } });
  }

  let navItems = $derived([
    { label: dictionary.labels.home, href: dictionary.urls.home },
    { label: dictionary.labels.share, href: dictionary.urls.share },
    { label: dictionary.labels.projects, href: dictionary.urls.projects },
    { label: dictionary.labels.about, href: dictionary.urls.about },
    // 数据看板与主页平行：仅本地后台（LOCAL_ADMIN dev）在「更多」右侧多一枚按钮，
    // 生产构建 admin 恒为 false，公网导航不出现。
    ...(admin ? [{ label: dictionary.labels.studio, href: `/studio/dashboard?lang=${lang}` }] : []),
  ]);

  // ------------------------------------------------------------------
  // Navigation active/pending state
  // ------------------------------------------------------------------
  let pendingNavHref = $state<string | null>(null);

  function isActive(href: string): boolean {
    const pathname = page.url.pathname;
    const base = href.split("?")[0];
    if (base === dictionary.urls.home) return pathname === base;
    // 技术文章挂在 /{lang}/posts，但归属「分享」页。
    if (
      base === dictionary.urls.share &&
      pathname.startsWith(`/${lang}/posts`)
    ) {
      return true;
    }
    return pathname.startsWith(base);
  }

  function onNavPress(href: string) {
    pendingNavHref = isActive(href) ? null : href;
  }

  // ------------------------------------------------------------------
  // Color mode — the inline script in app.html applies the stored mode
  // before paint; this just handles user toggling afterwards.
  // ------------------------------------------------------------------
  let colorMode = $state<ColorMode>("system");
  let systemDark = $state(false);
  let isDark = $derived(
    colorMode === "dark" || (colorMode === "system" && systemDark),
  );

  onMount(() => {
    const stored = document.documentElement.dataset.colorMode;
    if (stored === "light" || stored === "dark" || stored === "system") {
      colorMode = stored;
    }

    const media = window.matchMedia("(prefers-color-scheme: dark)");
    systemDark = media.matches;
    const onMediaChange = () => {
      systemDark = media.matches;
      if (colorMode === "system") {
        document.documentElement.classList.toggle("dark", media.matches);
      }
    };
    media.addEventListener("change", onMediaChange);
    return () => media.removeEventListener("change", onMediaChange);
  });

  function setColorMode(mode: ColorMode) {
    colorMode = mode;
    try {
      localStorage.setItem("color-mode", mode);
    } catch {}
    document.documentElement.dataset.colorMode = mode;
    const dark =
      mode === "dark" ||
      (mode === "system" &&
        window.matchMedia("(prefers-color-scheme: dark)").matches);
    document.documentElement.classList.toggle("dark", dark);
  }

  // ------------------------------------------------------------------
  // Language switch — let the dial animate before navigating.
  // ------------------------------------------------------------------
  const LANGUAGE_DIAL_ANIMATION_MS = 220;
  // svelte-ignore state_referenced_locally -- initial value; kept in sync by the effect below
  let displayLang = $state<string>(lang);
  let langSwitchTimer: ReturnType<typeof setTimeout> | undefined;

  $effect(() => {
    displayLang = lang;
  });

  function switchToLanguage(newLang: string) {
    if (newLang === displayLang) return;
    const first = page.url.pathname.split("/")[1];
    let newPath: string;
    if (isLanguage(first)) {
      const rest = page.url.pathname.split("/").slice(2);
      newPath = `/${newLang}${rest.length ? `/${rest.join("/")}` : ""}`;
    } else {
      // 非本地化路由（如 /studio/submit）：保持路径不变，用 ?lang= 切换界面语言，
      // 直接拼 /{lang}/... 会拼出不存在的 404 地址
      const url = new URL(page.url.href);
      url.searchParams.set("lang", newLang);
      newPath = url.pathname + url.search;
    }
    displayLang = newLang;
    pendingNavHref = null;
    clearTimeout(langSwitchTimer);
    langSwitchTimer = setTimeout(() => goto(newPath), LANGUAGE_DIAL_ANIMATION_MS);
  }

  // ------------------------------------------------------------------
  // 走纸动画：打开页面时整张纸从出纸口滑出（不是滑一小段），切换页面时
  // 先反向把旧页吸回打印机，再让新页滑出。单程固定 0.5s，走纸速度 = 纸长 / 0.5s
  // （纸越长走得越快），匀速走完。
  // 纸张「藏在出纸口内侧」的状态由 app.html 预置的 .paper-preload 承担，
  // 动画与它同帧交接，首屏不会先闪出一帧完整页面。
  // ------------------------------------------------------------------
  const FEED_MS = 500;
  const PAPER_PRELOAD_CLASS = "paper-preload";

  let paperElement: HTMLDivElement;
  let sheetAnimation: Animation | undefined;
  // 走纸流程接管了一次导航后置为 true：它重发的那次导航要放行，不能再拦一次
  let retracting = false;

  function setPaperHidden(hidden: boolean) {
    document.documentElement.classList.toggle(PAPER_PRELOAD_CLASS, hidden);
  }

  function prefersReducedMotion(): boolean {
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }

  /** 纸张当前的纵向偏移，换算成纸高的百分比（正在跑的动画也算进去） */
  function paperOffsetPct(el: HTMLElement): number {
    const transform = getComputedStyle(el).transform;
    if (!transform || transform === "none" || !el.offsetHeight) return 0;
    return (new DOMMatrixReadOnly(transform).f / el.offsetHeight) * 100;
  }

  /**
   * 跑一段走纸。返回 false 表示中途被新的一段走纸取消，收尾交给那一段。
   * keepHidden 为真（吸入结束）时交回预置隐藏状态，纸张继续留在出纸口内侧。
   */
  async function runSheetFeed(
    startPct: number,
    endPct: number,
    keepHidden: boolean,
  ): Promise<boolean> {
    sheetAnimation?.cancel(); // 上一段没跑完就让位，由这一段接着当前位置走
    paperElement.style.willChange = "transform";
    const animation = paperElement.animate(
      [
        { transform: `translateY(${startPct}%)` },
        { transform: `translateY(${endPct}%)` },
      ],
      { duration: FEED_MS, easing: "linear", fill: "both" },
    );
    sheetAnimation = animation;
    try {
      await animation.finished;
    } catch {
      // 被新的一段走纸取消（见上方说明），收尾交给它
      return false;
    }
    if (keepHidden) setPaperHidden(true);
    animation.cancel(); // 撤掉 fill：transform 常驻会让纸内的 fixed 元素改换定位基准
    paperElement.style.willChange = "";
    sheetAnimation = undefined;
    return true;
  }

  /** 滑出：从当前位置把纸完整吐出来，页脚撕边最后离开出纸口 */
  function feedOutPaper(): Promise<boolean> {
    // 偏移要在撤掉预置隐藏之前读：类一撤 transform 就变回 none
    const startPct = paperOffsetPct(paperElement);
    setPaperHidden(false); // 起始帧已经盖住纸面，预置隐藏可以撤了
    return runSheetFeed(startPct, 0, false);
  }

  /** 吸入：反向走纸，把纸整张收回打印机 */
  function retractPaper(): Promise<boolean> {
    // 从当前位置出发，所以上一段没跑完也能接上
    return runSheetFeed(paperOffsetPct(paperElement), -100, true);
  }

  /** 吸入旧页，然后重发这次导航，由 afterNavigate 接着把新页滑出来 */
  async function retractThenNavigate(target: URL) {
    retracting = true;
    try {
      // 中途被打断说明有别的链接接管了导航，这次就不重发了
      if (!(await retractPaper())) return;
      await goto(target.href);
    } catch (error) {
      // 导航没能完成时把纸放回来，否则整页会一直藏在出纸口后面
      setPaperHidden(false);
      console.error("页面切换失败，已恢复纸张显示", error);
    } finally {
      retracting = false;
    }
  }

  beforeNavigate((navigation) => {
    if (retracting || !navigation.to || navigation.willUnload) return;
    if (navigation.type !== "link" && navigation.type !== "goto") return;
    if (!paperElement || prefersReducedMotion()) return;

    // 同一条路由只换 query/hash（后台筛选、页内锚点）不动画，保持即时响应
    const target = navigation.to.url;
    if (target.pathname === page.url.pathname) return;

    navigation.cancel();
    void retractThenNavigate(target);
  });

  afterNavigate((navigation) => {
    pendingNavHref = null;
    // 这些情况不走纸：前进/后退（保留浏览器原生的即时切换与滚动位置恢复）、
    // 纸张还完整露在外面（说明这次导航没经过吸入，例如点了当前页的链接，
    // 再走一次滑出只会让纸凭空弹回出纸口里）
    if (
      navigation.type === "popstate" ||
      !paperElement ||
      prefersReducedMotion() ||
      paperOffsetPct(paperElement) > -0.5
    ) {
      setPaperHidden(false);
      return;
    }
    void feedOutPaper();
  });

  // Keep the indicator light's pulse in sync with wall-clock time so it
  // doesn't visually reset on client-side navigations.
  let indicatorDelay = $state("0ms");
  onMount(() => {
    indicatorDelay = `-${Date.now() % 2000}ms`;
  });

  const currentYear = new Date().getFullYear();
</script>

<!-- Desktop-only pull-cord light switch (bulb top-left, cord top-right) -->
<LightSwitch
  {isDark}
  lang={displayLang}
  ontoggle={(dark) => setColorMode(dark ? "dark" : "light")}
/>

<div class="min-h-screen page-grid flex flex-col items-center px-3 py-6 sm:py-10">
  <!-- Printer Body -->
  <div class="printer-stack w-full max-w-4xl relative">
    <!-- Snail crawling along the very top edge of the printer shell -->
    <PrinterSnail />

    <!-- Header housing — brand plate & nav only. The paper slit is a separate sticky
         element below (see .printer-slit-bar), so the shell can scroll out of view
         while the slit stays pinned to the top of the viewport. -->
    <div
      class="printer-header-border dark:border dark:border-white/[0.06] border-b-0 dark:border-b-0 rounded-t-[2.5rem] overflow-hidden relative z-10"
    >
      <!-- Dark mode ambient glow — soft top light spill -->
      <div
        class="hidden dark:block absolute inset-0 pointer-events-none"
        aria-hidden="true"
      >
        <div
          class="absolute -top-[40%] left-1/2 -translate-x-1/2 w-[120%] h-[80%] bg-[radial-gradient(ellipse_at_center,rgba(100,120,255,0.07)_0%,rgba(80,100,220,0.03)_40%,transparent_70%)]"
        ></div>
      </div>

      <!-- Draggable shell stickers -->
      <Stickers />

      <!-- Top part - Brand & Nav -->
      <div
        class="bg-printer-body dark:bg-printer-body-dark px-6 pt-6 pb-5 sm:px-10 sm:pt-10 relative"
      >
        <!-- Brand plate -->
        <div class="relative flex items-start justify-between mb-8">
          <div class="flex items-center gap-4">
            <div class="relative">
              <div
                class="absolute -inset-2 rounded-full bg-black/5 dark:bg-white/[0.08] shadow-inner"
              ></div>
              <img
                class="h-8 w-8 rounded-full ring-1 ring-black/10 dark:ring-white/[0.15] shadow-sm dark:shadow-[0_0_12px_rgba(100,120,255,0.1)] relative z-10"
                src="/static/avatar.png"
                alt="Craig"
                width="32"
                height="32"
              />
            </div>
            <div>
              <div
                class="font-mono text-sm font-bold tracking-[0.25em] text-printer-ink dark:text-printer-ink-dark uppercase"
              >
                <EditableText
                  editing={editing}
                  value={site.brandName}
                  onsave={saveBrandName}
                  maxlength={30}
                />
              </div>
              <div
                class="font-mono text-[9px] tracking-[0.1em] text-printer-ink-light dark:text-printer-ink-dark/40 uppercase mt-0.5"
              >
                <EditableText
                  editing={editing}
                  value={site.brandTagline}
                  onsave={saveBrandTagline}
                  maxlength={60}
                />
              </div>
            </div>
          </div>

          <div class="flex items-start gap-6">
            <div class="flex flex-col items-center gap-1.5">
              <div
                class="relative w-3.5 h-3.5 rounded-full bg-black/10 dark:bg-black/40 flex items-center justify-center"
              >
                <div
                  class="w-2.5 h-2.5 rounded-full bg-green-500/90 shadow-[0_0_8px_rgba(34,197,94,0.6),inset_0_-1px_2px_rgba(0,0,0,0.3)] animate-[pulse_2s_infinite]"
                  style:animation-delay={indicatorDelay}
                ></div>
                <div
                  class="absolute inset-0 rounded-full border border-black/10 dark:border-white/5 shadow-inner pointer-events-none"
                ></div>
              </div>
              <span
                class="font-mono text-[8px] text-printer-ink-light dark:text-printer-ink-dark/40 uppercase tracking-widest leading-none"
              >
                ON
              </span>
            </div>
          </div>
        </div>

        <!-- Navigation row -->
        <div
          class="relative mt-2 flex flex-col sm:flex-row items-stretch sm:items-center gap-1.5 sm:gap-2"
        >
          <nav class="relative flex items-center gap-2 sm:gap-2.5 flex-1 w-full py-1.5">
            {#each navItems as item (item.href)}
              <a
                href={item.href}
                onclick={() => onNavPress(item.href)}
                class={[
                  "printer-btn whitespace-nowrap",
                  (pendingNavHref
                    ? pendingNavHref === item.href
                    : isActive(item.href)) && "active",
                ]}
              >
                <span class="leading-none">{item.label}</span>
              </a>
            {/each}
          </nav>
          <div class="sm:hidden h-[1px] bg-black/10 dark:bg-white/10"></div>
          <div class="flex items-center justify-end gap-5 shrink-0 py-1">
            {#if admin}
              <a
                href={`/studio/submit?lang=${lang}`}
                class="printer-btn whitespace-nowrap !bg-printer-accent/80 hover:!bg-printer-accent dark:!bg-printer-accent-dark/80 dark:hover:!bg-printer-accent-dark !text-white border-transparent flex items-center gap-1.5"
                title={lang === "zh" ? "投稿" : "Submit"}
              >
                <Icon name="pen" class="w-3.5 h-3.5" />
                <span class="leading-none">{lang === "zh" ? "投稿" : "Submit"}</span>
              </a>
            {/if}
            <RotaryDial
              options={[
                { value: "en", label: "EN" },
                { value: "zh", label: "中" },
              ]}
              value={displayLang}
              onchange={switchToLanguage}
              title={displayLang === "en" ? "切换到中文" : "Switch to English"}
            />
            <!-- Mobile/tablet: rotary dial. Desktop: replaced by the pull-cord light switch. -->
            <div class="lg:hidden">
              <RotaryDial
                options={[
                  { value: "system", icon: "computer" },
                  { value: "light", icon: "sun" },
                  { value: "dark", icon: "moon" },
                ]}
                value={colorMode}
                onchange={(mode) => setColorMode(mode as ColorMode)}
                labelLayout="inline"
                title={colorMode === "system"
                  ? "System"
                  : colorMode === "light"
                    ? "Light"
                    : "Dark"}
              />
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Paper feed slit — a sticky element of its own: it travels up with the shell until
         it reaches the top of the viewport, then stays pinned there while the shell scrolls
         out of view and the paper slides up behind it, as if the printer were pulling the
         page back in. It carries the shell band above the slit plus the slit's top edge, and
         nothing else: the paper comes out of the slit and lies over the slit's lower part,
         so the rest of the slit and the lip below it are painted under the paper
         (see .printer-slit-lip). z-30 keeps this strip above the paper (z-20). -->
    <div
      class="printer-slit-bar bg-printer-shell dark:bg-printer-shell-dark h-2 flex items-center justify-center z-30 dark:border dark:border-white/[0.06] border-t-0 dark:border-t-0 border-b-0 dark:border-b-0"
    >
      <!-- Inset shadows to give depth to the slit area -->
      <div
        class="absolute inset-0 shadow-[inset_0_2px_4px_rgba(0,0,0,0.08)] dark:shadow-[inset_0_3px_6px_rgba(0,0,0,0.5)] pointer-events-none"
      ></div>
      <div
        class="absolute inset-y-0 left-0 w-8 bg-gradient-to-r from-black/[0.05] to-transparent dark:from-black/[0.2]"
      ></div>
      <div
        class="absolute inset-y-0 right-0 w-8 bg-gradient-to-l from-black/[0.05] to-transparent dark:from-black/[0.2]"
      ></div>
      <!-- Upper edge of the paper slit — the paper covers the rest of the slit below this line -->
      <div
        class="absolute bottom-0 left-2 right-2 sm:left-8 sm:right-8 h-px bg-black/60 dark:bg-black/90 rounded-t-[1px] shadow-[inset_0_1px_3px_rgba(0,0,0,0.8)] dark:shadow-[inset_0_1px_4px_rgba(0,0,0,0.9)]"
      ></div>
      <!-- Printer head blocking light — rides with the slit, the paper slides behind it -->
      <div class="paper-top-occlusion" aria-hidden="true"></div>
    </div>

    <!-- Lower part of the paper slit + the shell lip below it: the paper leaves the slit and
         lies over both, so this strip must sit underneath the paper (no z-index here — the
         paper is z-20) while still pinning itself to the slit (it follows the bar in flow,
         so its sticky threshold is simply the bar's height). Its top edge continues the
         slit's upper edge, so the slit reads as one 6px line wherever the paper doesn't
         cover it. -->
    <div
      class="printer-slit-lip bg-printer-shell dark:bg-printer-shell-dark h-3 rounded-b-sm dark:border dark:border-white/[0.06] border-t-0 dark:border-t-0"
    >
      <div
        class="absolute inset-y-0 left-0 w-8 bg-gradient-to-r from-black/[0.05] to-transparent dark:from-black/[0.2]"
      ></div>
      <div
        class="absolute inset-y-0 right-0 w-8 bg-gradient-to-l from-black/[0.05] to-transparent dark:from-black/[0.2]"
      ></div>
      <!-- Lower part of the paper slit, hidden behind the paper in the middle -->
      <div
        class="absolute top-0 left-2 right-2 sm:left-8 sm:right-8 h-[5px] bg-black/60 dark:bg-black/90 rounded-b-[1px] shadow-[inset_0_1px_3px_rgba(0,0,0,0.8)] dark:shadow-[inset_0_1px_4px_rgba(0,0,0,0.9)]"
      ></div>
    </div>

    <!-- Cast shadow outside the printer shell bottom edge -->
    <div class="relative h-0 pointer-events-none" aria-hidden="true">
      <div class="printer-shell-bottom-shadow"></div>
    </div>

    <!-- Printed paper output area — clipped so paper slides in from the slit -->
    <div
      class="printer-paper-wrap relative mx-3 sm:mx-10 -mt-[12px] z-20"
      style:clip-path="inset(0 -20px -56px -20px)"
    >
      <div class="printer-paper-sheet" bind:this={paperElement}>
        <div
          class="printer-paper-area bg-printer-paper dark:bg-printer-paper-dark dark:border dark:border-white/[0.04] thermal-texture min-h-[60vh] shadow-[0_4px_12px_rgba(0,0,0,0.15),0_1px_2px_rgba(0,0,0,0.1)] dark:shadow-[0_4px_20px_rgba(0,0,0,0.4),0_1px_3px_rgba(0,0,0,0.3)] relative z-0 flex flex-col overflow-hidden"
        >
          <div
            class="absolute -top-1 left-0 right-0 h-1 bg-printer-paper dark:bg-printer-paper-dark"
          ></div>

          <!-- Perforation marks -->
          <div
            class="absolute left-0 top-0 bottom-0 w-4 flex flex-col items-center justify-start gap-6 pt-4 opacity-20 pointer-events-none"
          >
            {#each Array.from({ length: 60 }), i (i)}
              <div
                class="w-1.5 h-1.5 rounded-full border border-printer-ink/30 dark:border-printer-ink-dark/30 shrink-0"
              ></div>
            {/each}
          </div>
          <div
            class="absolute right-0 top-0 bottom-0 w-4 flex flex-col items-center justify-start gap-6 pt-4 opacity-20 pointer-events-none"
          >
            {#each Array.from({ length: 60 }), i (i)}
              <div
                class="w-1.5 h-1.5 rounded-full border border-printer-ink/30 dark:border-printer-ink-dark/30 shrink-0"
              ></div>
            {/each}
          </div>

          <div class="printer-content-area flex-1 px-6 sm:px-10 py-8 relative z-10">
            {@render children()}
          </div>

          <div
            class="px-6 sm:px-10 py-6 mt-4 border-t border-dashed border-printer-ink/10 dark:border-printer-ink-dark/10 relative z-10"
          >
            <div
              class="flex flex-col sm:flex-row items-center justify-between gap-4 text-printer-ink-light dark:text-printer-ink-dark/40"
            >
              <div
                class="font-mono text-[10px] tracking-widest uppercase order-2 sm:order-1"
              >
                © {currentYear} Craig
              </div>
              <div
                class="font-mono text-[10px] tracking-widest uppercase flex items-center gap-4 order-1 sm:order-2"
              >
                <SocialHoverCard
                  kind="github"
                  href="https://github.com/CraigKEttleman"
                  {lang}
                  {dictionary}
                  class="hover:text-printer-accent transition-colors"
                >
                  GitHub
                </SocialHoverCard>
                <SocialHoverCard
                  kind="x"
                  href="https://x.com/CraigKettlmgc8"
                  {lang}
                  {dictionary}
                  class="hover:text-printer-accent transition-colors"
                >
                  X
                </SocialHoverCard>
                <SocialHoverCard
                  kind="email"
                  href="mailto:craigmail.ai@gmail.com"
                  {lang}
                  {dictionary}
                  align="right"
                  class="hover:text-printer-accent transition-colors"
                >
                  Email
                </SocialHoverCard>
              </div>
            </div>
          </div>
        </div>
        <div class="paper-edge-bottom h-0"></div>
      </div>
    </div>
  </div>
</div>
