<script lang="ts">
  import "../app.css";
  import { page } from "$app/state";
  import { getDictionary, isLanguage, defaultLanguage } from "$lib/dictionaries";
  import PrinterShell from "$lib/components/PrinterShell.svelte";
  import AdminBar from "$lib/components/AdminBar.svelte";

  let { children } = $props();

  let lang = $derived.by(() => {
    const param = page.params.lang ?? page.url.pathname.split("/")[1];
    return param && isLanguage(param) ? param : defaultLanguage;
  });
  let dictionary = $derived(getDictionary(lang));
  let admin = $derived(!!page.data.admin);

  $effect(() => {
    document.documentElement.lang = lang;
  });
</script>

<PrinterShell {lang} {dictionary} {admin}>
  {@render children()}
</PrinterShell>

{#if admin}
  <AdminBar />
{/if}
