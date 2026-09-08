<script lang="ts">
  import { browser } from "$app/environment";
  import { getDictionary, isLanguage } from "$lib/dictionaries";
  import type { Summary } from "$lib/analytics-types";
  import { buildHeatmap, formatNum, toCSV } from "$lib/dashboard-data";
  import { COUNTRIES } from "$lib/assets/world-map";
  import PrintedDivider from "$lib/components/PrintedDivider.svelte";
  import PrintedLabel from "$lib/components/PrintedLabel.svelte";
  import PrintedPageTitle from "$lib/components/PrintedPageTitle.svelte";
  import PrintedSection from "$lib/components/PrintedSection.svelte";
  import RotaryDial from "$lib/components/RotaryDial.svelte";
  import LineChart from "$lib/components/charts/LineChart.svelte";
  import DonutChart from "$lib/components/charts/DonutChart.svelte";
  import BarChart from "$lib/components/charts/BarChart.svelte";
  import RadialHours from "$lib/components/charts/RadialHours.svelte";
  import WorldMap from "$lib/components/charts/WorldMap.svelte";
  import HeatGrid from "$lib/components/charts/HeatGrid.svelte";
  import CountUp from "$lib/components/charts/CountUp.svelte";

  let { data } = $props();

  let lang = $derived.by(() => {
    const l = (data as { lang?: unknown }).lang;
    return typeof l === "string" && isLanguage(l) ? l : "en";
  });
  let dictionary = $derived(getDictionary(lang));

  const T = {
    zh: {
      title: "数据看板",
      subtitle: "与主页平行的后台报表：访问、地区、内容，一张纸上看完。",
      source: "线上 hhy.homes",
      range: "范围",
      refresh: "⟳ 刷新",
      export: "↓ CSV",
      real: "REAL",
      kpiPv: "总访问 PV",
      kpiUv: "独立访客 UV",
      kpiCountries: "覆盖国家",
      kpiPages: "触达页面",
      today: "今日",
      avgDay: "日均",
      peak: "峰值",
      trend: "访问趋势 · TRAFFIC",
      pv: "PV",
      uv: "UV",
      regions: "访问地区 · WORLD",
      mapHint: "Natural Earth 真实海岸线 · 按国家着色 · 悬停看明细",
      regionRank: "国家排行 TOP 8",
      devices: "设备 · DEVICES",
      refs: "来源 · SOURCES",
      langs: "语言 · LANGS",
      visitors: "访客构成 · VISITORS",
      vNew: "新访客",
      vReturning: "回访客",
      hours: "24 小时 · HOURS",
      top: "热门页面 · TOP PAGES",
      heat: "发布热力 · PUBLISH",
      heatHint: "真实文章日期聚合 · 近 26 周",
      cats: "分类计数 · CATEGORIES",
      system: "系统 · SYSTEM",
      sysGen: "报表生成",
      sysTotal: "累计",
      sysSource: "数据源",
      sysEnv: "环境",
      sysEnvValue: "LOCAL_ADMIN DEV",
      emptyTitle: "线上暂无访问记录",
      emptyProd: "埋点随本次部署上线，数据会随真实访问逐渐累积。",
      errTitle: "数据源不可用",
      errNO_KEY: "缺少 COMMENTS_ADMIN_KEY：在本地根目录建 .env 写入该密钥（值见部署时打印，或服务器 /var/www/hhy.homes/.env）。",
      errNETWORK: "无法连接线上服务（网络或服务器不可达）。",
      errHTTP: "线上接口返回错误。",
      errRetry: "重试",
      loading: "读取中…",
      unknown: "未知",
      deviceDesktop: "桌面",
      deviceMobile: "移动",
      deviceTablet: "平板",
      refDirect: "直接访问",
      refSearch: "搜索引擎",
      refSocial: "社交",
      refInternal: "站内",
      refReferral: "外链",
      printedOn: "Printed on",
    },
    en: {
      title: "DASHBOARD",
      subtitle: "The studio sheet beside the public site: traffic, regions, content — one page.",
      source: "PROD hhy.homes",
      range: "RANGE",
      refresh: "⟳ REFRESH",
      export: "↓ CSV",
      real: "REAL",
      kpiPv: "PAGE VIEWS",
      kpiUv: "VISITORS",
      kpiCountries: "COUNTRIES",
      kpiPages: "PAGES HIT",
      today: "TODAY",
      avgDay: "AVG/DAY",
      peak: "PEAK",
      trend: "TRAFFIC · TREND",
      pv: "PV",
      uv: "UV",
      regions: "WORLD · REGIONS",
      mapHint: "Natural Earth coastlines · shaded by country · hover for detail",
      regionRank: "TOP 8 COUNTRIES",
      devices: "DEVICES",
      refs: "SOURCES",
      langs: "LANGS",
      visitors: "VISITORS · MIX",
      vNew: "NEW",
      vReturning: "RETURNING",
      hours: "24 HOURS",
      top: "TOP PAGES",
      heat: "PUBLISH · HEATMAP",
      heatHint: "Real post dates · last 26 weeks",
      cats: "CATEGORIES",
      system: "SYSTEM",
      sysGen: "GENERATED",
      sysTotal: "ALL-TIME",
      sysSource: "SOURCE",
      sysEnv: "ENV",
      sysEnvValue: "LOCAL_ADMIN DEV",
      emptyTitle: "NO PRODUCTION TRAFFIC YET",
      emptyProd: "The beacon ships with this deploy and fills up as real visits arrive.",
      errTitle: "SOURCE UNAVAILABLE",
      errNO_KEY: "COMMENTS_ADMIN_KEY missing: create a local .env with that key (printed at deploy time, or see server /var/www/hhy.homes/.env).",
      errNETWORK: "Cannot reach the production server.",
      errHTTP: "Production endpoint returned an error.",
      errRetry: "RETRY",
      loading: "LOADING…",
      unknown: "Unknown",
      deviceDesktop: "Desktop",
      deviceMobile: "Mobile",
      deviceTablet: "Tablet",
      refDirect: "Direct",
      refSearch: "Search",
      refSocial: "Social",
      refInternal: "Internal",
      refReferral: "Referral",
      printedOn: "Printed on",
    },
  } as const;
  let t = $derived(T[lang]);

  // ------------------------------------------------------------------
  // 数据加载：只读线上（hhy.homes），手动刷新
  // ------------------------------------------------------------------
  let range = $state<"7" | "30" | "90">("30");
  let summary = $state<Summary | null>(null);
  let loadError = $state<string | null>(null);
  let loading = $state(false);
  let reloadKey = $state(0);

  let livePosts = $derived((data.posts ?? []).filter((p: { draft?: boolean }) => !p.draft));

  $effect(() => {
    if (!browser) return;
    void load(range, reloadKey, lang);
  });

  async function load(days: "7" | "30" | "90", key: number, uiLang: "zh" | "en") {
    void key;
    loading = true;
    loadError = null;
    try {
      const res = await fetch(`/studio/api/analytics?source=prod&days=${days}`);
      const body = (await res.json()) as { ok: boolean; summary?: Summary; error?: string };
      if (!res.ok || !body.ok) {
        summary = null;
        loadError = body.error ?? `HTTP_${res.status}`;
      } else {
        summary = body.summary ?? null;
        loadError = null;
      }
    } catch {
      summary = null;
      loadError = "NETWORK";
    } finally {
      loading = false;
    }
    void uiLang;
  }

  function reload() {
    reloadKey += 1;
  }

  // ------------------------------------------------------------------
  // 派生视图数据
  // ------------------------------------------------------------------
  let countryName = $derived.by(() => {
    const m = new Map<string, { zh: string; en: string }>();
    for (const c of COUNTRIES) m.set(c.a2, { zh: c.zh, en: c.en });
    return m;
  });
  function cname(code: string): string {
    const hit = countryName.get(code);
    return hit ? (lang === "zh" ? hit.zh : hit.en) : code || t.unknown;
  }

  let countryValues = $derived.by(() => {
    const m = new Map<string, number>();
    for (const c of summary?.countries ?? []) m.set(c.code, c.pv);
    return m;
  });

  const DEVICE_KEY = {
    desktop: "deviceDesktop",
    mobile: "deviceMobile",
    tablet: "deviceTablet",
  } as const;
  let deviceSlices = $derived(
    (summary?.devices ?? []).map((d) => ({
      label: d.label in DEVICE_KEY ? t[DEVICE_KEY[d.label as keyof typeof DEVICE_KEY]] : d.label,
      value: d.value,
    })),
  );
  const REF_KEY = {
    direct: "refDirect",
    search: "refSearch",
    social: "refSocial",
    internal: "refInternal",
    referral: "refReferral",
  } as const;
  let refSlices = $derived(
    (summary?.refs ?? []).map((d) => ({
      label: d.label in REF_KEY ? t[REF_KEY[d.label as keyof typeof REF_KEY]] : d.label,
      value: d.value,
    })),
  );
  let langSlices = $derived(
    (summary?.langs ?? []).map((d) => ({ label: d.label === "zh" ? "中文" : "English", value: d.value })),
  );
  let visitorSlices = $derived([
    { label: t.vNew, value: summary?.visitors.new ?? 0 },
    { label: t.vReturning, value: summary?.visitors.returning ?? 0 },
  ]);

  let countryBars = $derived(
    (summary?.countries ?? []).slice(0, 8).map((c) => ({ label: cname(c.code), value: c.pv, hint: c.code })),
  );

  let pageBars = $derived.by(() => {
    const posts = livePosts as { title: string; permalink: string }[];
    return (summary?.pages ?? []).slice(0, 8).map((p) => {
      const hit = posts.find((x) => x.permalink === p.path);
      return { label: hit ? hit.title : p.path, value: p.pv, hint: p.path };
    });
  });

  // 发布热力 / 分类计数：永远来自真实内容
  let heatDates = $derived.by(() => {
    const seen = new Set<string>();
    const out: string[] = [];
    for (const p of livePosts as { slug: string; date: string }[]) {
      const key = `${p.slug}::${p.date.slice(0, 10)}`;
      if (!seen.has(key)) {
        seen.add(key);
        out.push(p.date.slice(0, 10));
      }
    }
    return out;
  });
  let heat = $derived(buildHeatmap(heatDates, 26));
  let catBars = $derived(
    ((data.categories ?? []) as { name: { zh: string; en: string }; count: { zh: number; en: number } }[])
      .map((c) => ({ label: c.name[lang], value: c.count[lang] }))
      .sort((a, b) => b.value - a.value),
  );

  let todayPv = $derived(summary?.daily.at(-1)?.pv ?? 0);
  let todayUv = $derived(summary?.daily.at(-1)?.uv ?? 0);
  let avgPv = $derived(summary ? Math.round(summary.totals.pv / summary.days) : 0);
  let peakDay = $derived.by(() => {
    const list = summary?.daily ?? [];
    if (list.length === 0) return null;
    return list.reduce((a, b) => (b.pv > a.pv ? b : a));
  });

  function exportCSV() {
    if (!browser || !summary) return;
    const blob = new Blob([toCSV(summary.daily)], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `dashboard-prod-${range}d.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const printedOn = new Date().toISOString().split("T")[0];
</script>

<div>
  <!-- 报表头 -->
  <PrintedSection>
    <div class="flex flex-wrap items-start justify-between gap-3">
      <div>
        <PrintedPageTitle icon="pulse">{t.title}</PrintedPageTitle>
        <p class="mt-1 max-w-xl font-serif text-xs text-printer-ink-light dark:text-printer-ink-dark/50 leading-relaxed">
          {t.subtitle}
        </p>
      </div>
      <div class="flex flex-col items-end gap-1.5">
        <PrintedLabel icon="pulse">{t.real}</PrintedLabel>
        <span class="font-mono text-[9px] tracking-widest uppercase text-printer-ink-light/70 dark:text-printer-ink-dark/40">
          {t.printedOn} {printedOn}
        </span>
      </div>
    </div>

    <!-- 动作行：仅刷新 + 下载（外壳同款实体键）；时间旋钮（同语言切换旋钮）靠右 -->
    <div class="mt-6 flex flex-wrap items-center gap-3">
      <button type="button" class="printer-btn" onclick={reload}>
        {t.refresh}
      </button>
      <button type="button" class="printer-btn" onclick={exportCSV} disabled={!summary}>
        {t.export}
      </button>
      <div class="ml-auto flex items-center gap-2 pr-1">
        <span class="font-mono text-[9px] tracking-widest uppercase text-printer-ink-light/60 dark:text-printer-ink-dark/50">
          {t.range}
        </span>
        <RotaryDial
          options={[
            { value: "7", label: "7D" },
            { value: "30", label: "30D" },
            { value: "90", label: "90D" },
          ]}
          value={range}
          onchange={(v) => (range = v as "7" | "30" | "90")}
          title={t.range}
        />
      </div>
    </div>
  </PrintedSection>

  <PrintedDivider style="solid" />

  {#if loading}
    <div class="py-16 text-center font-mono text-[11px] tracking-[0.3em] uppercase text-printer-ink-light dark:text-printer-ink-dark/50">
      {t.loading}
    </div>
  {:else if loadError}
    <!-- 数据源不可用：如实报错，不伪造 -->
    <div class="border border-dashed border-printer-ink/20 dark:border-printer-ink-dark/20 px-5 py-8 text-center">
      <div class="font-mono text-[11px] tracking-[0.3em] uppercase text-printer-accent dark:text-printer-accent-dark">
        {t.errTitle} · {loadError}
      </div>
      <p class="mx-auto mt-3 max-w-md font-serif text-xs leading-relaxed text-printer-ink-light dark:text-printer-ink-dark/60">
        {loadError === "NO_KEY" ? t.errNO_KEY : loadError === "NETWORK" ? t.errNETWORK : t.errHTTP}
      </p>
      <div class="mt-4 flex justify-center">
        <button type="button" class="printer-btn" onclick={reload}>{t.errRetry}</button>
      </div>
    </div>
  {:else if !summary || summary.totals.pv === 0}
    <!-- 真实数据为空：诚实空态 -->
    <div class="border border-dashed border-printer-ink/20 dark:border-printer-ink-dark/20 px-5 py-8 text-center">
      <div class="font-mono text-[11px] tracking-[0.3em] uppercase text-printer-ink-light dark:text-printer-ink-dark/50">
        {t.emptyTitle}
      </div>
      <p class="mx-auto mt-3 max-w-md font-serif text-xs leading-relaxed text-printer-ink-light dark:text-printer-ink-dark/60">
        {t.emptyProd}
      </p>
      <div class="mt-4 flex justify-center">
        <button type="button" class="printer-btn" onclick={reload}>{t.errRetry}</button>
      </div>
    </div>
  {:else}
    <!-- KPI -->
    <PrintedSection>
      <div class="grid grid-cols-2 gap-4 lg:grid-cols-4">
        {#each [
          { label: t.kpiPv, value: summary.totals.pv, foot: `${t.avgDay} ${formatNum(avgPv, lang)}` },
          { label: t.kpiUv, value: summary.totals.uv, foot: `${t.today} ${formatNum(todayUv, lang)}` },
          { label: t.kpiCountries, value: summary.totals.countries, foot: peakDay ? `${t.peak} ${peakDay.label}` : "—" },
          { label: t.kpiPages, value: summary.totals.pages, foot: `${range}D` },
        ] as kpi, kpiI (kpi.label)}
          <div
            class="rise-in border border-printer-ink/10 dark:border-printer-ink-dark/10 bg-printer-paper dark:bg-printer-paper-dark px-4 py-4"
            style:animation-delay={`${kpiI * 70}ms`}
          >
            <div class="font-mono text-[9px] tracking-[0.25em] uppercase text-printer-ink-light dark:text-printer-ink-dark/50">
              {kpi.label}
            </div>
            <div class="mt-2 font-mono text-2xl font-bold leading-none text-printer-ink dark:text-printer-ink-dark">
              <CountUp value={kpi.value} format={(n) => formatNum(n, lang)} />
            </div>
            <div class="mt-2 font-mono text-[9px] tracking-wider uppercase text-printer-ink-light/70 dark:text-printer-ink-dark/40">
              {kpi.foot}
            </div>
          </div>
        {/each}
      </div>
    </PrintedSection>

    <PrintedDivider style="dashed" />

    <!-- 折线：PV / UV -->
    <PrintedSection label={t.trend} labelIcon="pulse">
      <LineChart
        series={[
          { name: t.pv, tone: "accent", values: summary.daily.map((d) => d.pv) },
          { name: t.uv, tone: "ink", values: summary.daily.map((d) => d.uv) },
        ]}
        labels={summary.daily.map((d) => d.label)}
        format={(n) => formatNum(n, lang)}
      />
    </PrintedSection>

    <PrintedDivider style="dashed" />

    <!-- 世界地图 + 国家排行（上下排列） -->
    <PrintedSection label={t.regions} labelIcon="send">
      <p class="mb-3 font-mono text-[9px] tracking-widest uppercase text-printer-ink-light/60 dark:text-printer-ink-dark/40">
        {t.mapHint}
      </p>
      <div class="space-y-6">
        <div class="border border-printer-ink/10 dark:border-printer-ink-dark/10 bg-printer-paper dark:bg-printer-paper-dark p-4">
          <WorldMap values={countryValues} {lang} format={(n) => formatNum(n, lang)} />
        </div>
        <div class="border border-printer-ink/10 dark:border-printer-ink-dark/10 bg-printer-paper dark:bg-printer-paper-dark px-5 py-5">
          <div class="mb-4 font-mono text-[9px] tracking-[0.25em] uppercase text-printer-ink-light dark:text-printer-ink-dark/50">
            {t.regionRank}
          </div>
          <div class="grid grid-cols-1 gap-x-12 gap-y-4 sm:grid-cols-2">
            <BarChart rows={countryBars.slice(0, 4)} format={(n) => formatNum(n, lang)} />
            <BarChart rows={countryBars.slice(4)} format={(n) => formatNum(n, lang)} accentFirst={false} />
          </div>
        </div>
      </div>
    </PrintedSection>

    <PrintedDivider style="dashed" />

    <!-- 四枚饼/环图 -->
    <div class="grid grid-cols-1 gap-8 sm:grid-cols-2 lg:grid-cols-4">
      <PrintedSection label={t.devices} labelIcon="computer">
        <DonutChart slices={deviceSlices} variant="pie" layout="col" format={(n) => formatNum(n, lang)} size={116} />
      </PrintedSection>
      <PrintedSection label={t.refs} labelIcon="send">
        <DonutChart slices={refSlices} variant="donut" layout="col" centerValue={formatNum(summary.totals.pv, lang)} centerLabel={t.pv} format={(n) => formatNum(n, lang)} size={116} />
      </PrintedSection>
      <PrintedSection label={t.langs} labelIcon="tag">
        <DonutChart slices={langSlices} variant="donut" layout="col" centerValue={formatNum(summary.totals.uv, lang)} centerLabel={t.uv} format={(n) => formatNum(n, lang)} size={116} />
      </PrintedSection>
      <PrintedSection label={t.visitors} labelIcon="user">
        <DonutChart slices={visitorSlices} variant="donut" layout="col" centerValue={formatNum(summary.totals.uv, lang)} centerLabel={t.uv} format={(n) => formatNum(n, lang)} size={116} />
      </PrintedSection>
    </div>

    <PrintedDivider style="dashed" />

    <!-- 24h 径向钟 + 热门页面 -->
    <div class="grid grid-cols-1 gap-8 lg:grid-cols-5">
      <PrintedSection label={t.hours} labelIcon="pulse" class="lg:col-span-2">
        <RadialHours hours={summary.hours} format={(n) => formatNum(n, lang)} />
      </PrintedSection>
      <PrintedSection label={t.top} labelIcon="star" class="lg:col-span-3">
        <BarChart rows={pageBars} format={(n) => formatNum(n, lang)} />
      </PrintedSection>
    </div>

    <PrintedDivider style="dashed" />

    <!-- 发布热力 + 分类计数（上下排列，真实内容） -->
    <div class="space-y-6">
      <PrintedSection label={t.heat} labelIcon="apps" class="mb-0">
        <p class="mb-3 font-mono text-[9px] tracking-widest uppercase text-printer-ink-light/60 dark:text-printer-ink-dark/40">
          {t.heatHint} · {heat.total}
        </p>
        <HeatGrid cells={heat.cells} {lang} />
      </PrintedSection>
      <PrintedSection label={t.cats} labelIcon="tag" class="mb-0">
        <div class="grid grid-cols-1 gap-x-12 gap-y-4 sm:grid-cols-2">
          <BarChart rows={catBars.slice(0, Math.ceil(catBars.length / 2))} format={(n) => formatNum(n, lang)} accentFirst={false} />
          <BarChart rows={catBars.slice(Math.ceil(catBars.length / 2))} format={(n) => formatNum(n, lang)} accentFirst={false} />
        </div>
      </PrintedSection>
    </div>

    <PrintedDivider style="dashed" />

    <!-- 系统状态 -->
    <PrintedSection label={t.system} labelIcon="tools">
      <div class="grid grid-cols-1 gap-x-6 gap-y-2 font-mono text-[11px] tracking-wider sm:grid-cols-2">
        {#each [
          [t.sysGen, `${summary.generatedAt.slice(0, 19).replace("T", " ")}Z`],
          [t.sysTotal, `${formatNum(summary.allTime.pv, lang)} PV · ${formatNum(summary.allTime.uv, lang)} UV`],
          [t.sysSource, t.source],
          [t.sysEnv, t.sysEnvValue],
        ] as row (row[0])}
          <div class="flex justify-between gap-3 border-b border-dotted border-printer-ink/10 dark:border-printer-ink-dark/10 py-1.5">
            <span class="uppercase text-printer-ink-light dark:text-printer-ink-dark/50">{row[0]}</span>
            <span class="text-printer-ink dark:text-printer-ink-dark">{row[1]}</span>
          </div>
        {/each}
      </div>
    </PrintedSection>
  {/if}

  <!-- 页脚戳 -->
  <div class="mt-10 border-t border-dotted border-printer-ink/10 dark:border-printer-ink-dark/10 pt-4">
    <div class="flex items-center justify-between">
      <div class="font-mono text-[9px] tracking-wider uppercase text-printer-ink-light dark:text-printer-ink-dark/30">
        {t.printedOn} {printedOn}
      </div>
      <PrintedLabel variant="muted">hhy.homes</PrintedLabel>
    </div>
  </div>
</div>

<style>
  @keyframes rise {
    from {
      opacity: 0;
      transform: translateY(6px);
    }
  }
  .rise-in {
    animation: rise 480ms cubic-bezier(0.22, 1, 0.36, 1) both;
  }
  @media (prefers-reduced-motion: reduce) {
    .rise-in {
      animation: none;
    }
  }
</style>
