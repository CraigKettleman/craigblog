<script lang="ts">
  import { onMount } from "svelte";
  import type { Dictionary } from "$lib/dictionaries";

  let {
    advertisements,
  }: {
    advertisements: Dictionary["postAdvertisements"];
  } = $props();

  // Prerendered pages always show the first ad; rotate randomly per visit.
  let index = $state(0);
  onMount(() => {
    index = Math.floor(Math.random() * advertisements.length);
  });

  let advertisement = $derived(advertisements[index]);
</script>

{#if advertisement}
  <a
    class="flex gap-4 py-3 px-3 -mx-1 rounded-md hover:bg-printer-ink/3 dark:hover:bg-printer-ink-dark/3 transition-colors border border-dashed border-printer-ink/10 dark:border-printer-ink-dark/10"
    href={advertisement.link}
    target="_blank"
    rel="noopener"
  >
  <img
    class="rounded-lg w-12 h-12 border border-printer-ink/10 dark:border-printer-ink-dark/10 shrink-0"
    src={advertisement.icon}
    alt={advertisement.title}
    width="48"
    height="48"
    loading="lazy"
  />
  <div class="flex flex-col gap-1 min-w-0">
    <div
      class="font-mono text-[10px] text-printer-ink-light dark:text-printer-ink-dark/40 uppercase tracking-wider"
    >
      {advertisement.title}
    </div>
    <div
      class="font-mono text-xs text-printer-ink/70 dark:text-printer-ink-dark/70 leading-relaxed"
    >
      {advertisement.description}
    </div>
  </div>
  </a>
{/if}
