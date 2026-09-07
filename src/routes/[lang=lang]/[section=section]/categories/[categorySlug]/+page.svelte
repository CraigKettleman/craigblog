<script lang="ts">
  import { page } from "$app/state";
  import { getDictionary } from "$lib/dictionaries";
  import { getEditMode } from "$lib/admin-state.svelte";
  import { saveAndNavigate, saveJson } from "$lib/admin-save";
  import { siteMeta } from "$lib/site";
  import PostList from "$lib/components/PostList.svelte";
  import PrintedDivider from "$lib/components/PrintedDivider.svelte";
  import PrintedPageTitle from "$lib/components/PrintedPageTitle.svelte";
  import PrintedSection from "$lib/components/PrintedSection.svelte";
  import Seo from "$lib/components/Seo.svelte";
  import EditableText from "$lib/components/admin/EditableText.svelte";

  let { data } = $props();

  let lang = $derived(data.lang);
  let dictionary = $derived(getDictionary(lang));
  let site = $derived(siteMeta(lang));
  let category = $derived(data.category);
  let admin = $derived(!!page.data.admin);
  let editing = $derived(admin && getEditMode());

  async function saveName(v: string) {
    if (!v.trim()) return;
    await saveJson("/studio/api/categories", "PUT", {
      section: data.section,
      slug: category.slug,
      name: { [lang]: v },
    });
  }

  async function saveDescription(v: string) {
    // 空串 = 清除可选描述（API 约定 clear）
    await saveJson("/studio/api/categories", "PUT", {
      section: data.section,
      slug: category.slug,
      description: v.trim() ? { [lang]: v } : { clear: true },
    });
  }

  /** slug 更名会改变分类页 URL：保存后跳转新地址（由 saveAndNavigate 汇合整页刷新）。 */
  async function saveSlug(v: string) {
    if (!v.trim() || v === category.slug) return;
    await saveAndNavigate(`/${lang}/${data.section}/categories/${v}`, () =>
      saveJson(
        "/studio/api/categories",
        "PUT",
        { section: data.section, slug: category.slug, newSlug: v },
        { invalidate: false },
      ),
    );
  }

  async function removeCategory() {
    if (!confirm(`删除分类「${category.name[lang]}」？`)) return;
    await saveAndNavigate(dictionary.urls[data.section], () =>
      saveJson(
        "/studio/api/categories",
        "DELETE",
        { section: data.section, slug: category.slug },
        { invalidate: false },
      ),
    );
  }
</script>

<Seo
  {lang}
  title="{category.name[lang]} - {site.websiteName}"
  description={category.description?.[lang]}
  path={category.permalink[lang]}
/>

<div>
  <!-- Header -->
  <PrintedSection>
    {#if editing}
      <!-- 元数据工具行：slug / 删除 -->
      <div
        class="mb-3 flex flex-wrap items-center gap-x-3 gap-y-1.5 rounded-sm border border-dashed border-printer-accent/40 bg-printer-accent/5 dark:bg-printer-accent-dark/10 px-2.5 py-1.5"
      >
        <span
          class="font-mono text-[9px] uppercase tracking-wider text-printer-ink-light dark:text-printer-ink-dark/50"
        >
          slug
        </span>
        <div class="w-56">
          <EditableText
            editing
            value={category.slug}
            onsave={saveSlug}
            inputClass="font-mono text-[11px]"
          />
        </div>
        <button
          type="button"
          onclick={removeCategory}
          class="rounded-sm border border-red-500/30 px-2 py-[3px] font-mono text-[10px] leading-none uppercase tracking-wider text-red-500 hover:bg-red-500/10 dark:text-red-400"
        >
          删除
        </button>
      </div>
    {/if}

    <PrintedPageTitle icon="tag">
      <EditableText
        editing={editing}
        value={category.name[lang]}
        onsave={saveName}
        maxlength={20}
      />
    </PrintedPageTitle>
    {#if category.description?.[lang] || editing}
      <p class="font-serif text-xs text-printer-ink-light dark:text-printer-ink-dark/50">
        <EditableText
          editing={editing}
          value={category.description?.[lang] ?? ""}
          onsave={saveDescription}
          maxlength={100}
          placeholder="分类描述（可选）"
        />
      </p>
    {/if}
    <p
      class="font-serif text-[10px] text-printer-ink-light dark:text-printer-ink-dark/40 mt-1"
    >
      {dictionary.labels.entries(data.posts.length)}
    </p>
  </PrintedSection>

  <!-- Back link -->
  <div class="mb-4">
    <a
      href={dictionary.urls[data.section]}
      class="inline-flex items-center gap-1 font-mono text-[11px] tracking-wider text-printer-accent dark:text-printer-accent-dark hover:underline"
    >
      {dictionary.labels.allSectionPosts[data.section]}
    </a>
  </div>

  <PrintedDivider style="dashed" />

  <!-- Post list -->
  <PostList posts={data.posts} {lang} />
</div>
