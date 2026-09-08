<script lang="ts">
  import "../app.css";
  import { page } from "$app/state";
  import {
    getDictionary,
    isLanguage,
    defaultLanguage,
  } from "$lib/dictionaries";
  import PrinterShell from "$lib/components/PrinterShell.svelte";
  import AdminBar from "$lib/components/AdminBar.svelte";
  import { trackView } from "$lib/analytics-client";

  let { children } = $props();

  // 语言推断：本地化路由取路径首段；非本地化路由由页面 server load 透传的
  // ?lang= 决定（预渲染页面没有查询串，不能直接读 url.searchParams）
  let lang = $derived.by(() => {
    const param = page.params.lang ?? page.url.pathname.split("/")[1];
    if (param && isLanguage(param)) return param;
    const fromData = (page.data as { lang?: unknown }).lang;
    return typeof fromData === "string" && isLanguage(fromData) ? fromData : defaultLanguage;
  });
  let dictionary = $derived(getDictionary(lang));
  let admin = $derived(!!page.data.admin);

  $effect(() => {
    document.documentElement.lang = lang;
  });

  // 匿名访问信标：首载与 SPA 跳转各计一次真实 PV
  $effect(() => {
    trackView(page.url.pathname, lang);
  });
</script>

<PrinterShell {lang} {dictionary} {admin}>
  {@render children()}
</PrinterShell>

{#if admin}
  <AdminBar />
{/if}
