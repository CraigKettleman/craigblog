<script lang="ts">
  import type { Section } from "$lib/content";
  import type { Language } from "$lib/dictionaries";
  import { saveJson } from "$lib/admin-save";
  import PrintedLabel from "$lib/components/PrintedLabel.svelte";
  import EditableText from "$lib/components/admin/EditableText.svelte";

  /**
   * 编辑态的分类管理条：原位改名、删除（服务端拒绝被引用的分类）、行内新建。
   * 供分享页与项目页共用；非编辑态由调用方自行渲染只读 chips。
   */
  let {
    lang,
    section,
    categories,
  }: {
    lang: Language;
    section: Section;
    /** 当前栏目的分类（slug + 双语名）。 */
    categories: Array<{ slug: string; name: { en: string; zh: string } }>;
  } = $props();

  let adding = $state(false);
  let newName = $state("");

  function autofocus(el: HTMLInputElement) {
    el.focus();
  }

  async function rename(slug: string, v: string) {
    if (!v.trim()) return;
    await saveJson("/studio/api/categories", "PUT", {
      section,
      slug,
      name: { [lang]: v },
    });
  }

  async function remove(slug: string) {
    if (!confirm(`删除分类「${slug}」？`)) return;
    await saveJson("/studio/api/categories", "DELETE", { section, slug });
  }

  function onNewKeydown(e: KeyboardEvent) {
    if (e.key === "Enter") {
      e.preventDefault();
      (e.currentTarget as HTMLInputElement).blur();
    } else if (e.key === "Escape") {
      adding = false;
    }
  }

  async function commitNew() {
    if (!adding) return;
    adding = false;
    const name = newName.trim();
    if (!name) return;
    // 两种语言先以同一文本兜底，可在分类页再分别修订
    await saveJson("/studio/api/categories", "POST", {
      section,
      name: { en: name, zh: name },
    });
  }
</script>

<div class="flex flex-wrap items-center gap-x-1.5 gap-y-1.5 mb-2">
  {#each categories as cat (cat.slug)}
    <div class="flex items-center gap-1">
      <PrintedLabel variant="default">
        <EditableText
          editing
          value={cat.name[lang]}
          onsave={(v) => rename(cat.slug, v)}
          maxlength={20}
          inputClass="font-mono text-[10px] w-24 uppercase tracking-widest"
        />
      </PrintedLabel>
      <button
        type="button"
        onclick={() => remove(cat.slug)}
        aria-label="删除分类 {cat.name[lang]}"
        class="font-mono text-[11px] leading-none text-printer-ink-light/50 hover:text-red-500 dark:text-printer-ink-dark/40"
      >
        ×
      </button>
    </div>
  {/each}

  {#if adding}
    <input
      use:autofocus
      bind:value={newName}
      onkeydown={onNewKeydown}
      onblur={commitNew}
      maxlength={20}
      placeholder="新分类名"
      class="inline-flex items-center h-[23px] rounded-sm border border-dashed border-printer-accent/60 bg-transparent px-2 font-mono text-[10px] uppercase tracking-widest text-printer-ink dark:text-printer-ink-dark outline-none focus:border-printer-accent dark:border-printer-accent-dark/60"
    />
  {:else}
    <button
      type="button"
      onclick={() => {
        adding = true;
        newName = "";
      }}
      class="font-mono text-[10px] tracking-wider text-printer-accent dark:text-printer-accent-dark hover:underline"
    >
      ＋ 新建分类
    </button>
  {/if}
</div>
