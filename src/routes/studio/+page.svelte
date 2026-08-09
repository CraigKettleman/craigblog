<script lang="ts">
  import { browser } from "$app/environment";

  interface PostMeta {
    path: string;
    section: string;
    slug: string;
    title: { en: string; zh: string };
    date: string;
    draft: boolean;
    featured: boolean;
    categories: string[];
  }

  interface PostContent {
    en: string;
    zh: string;
  }

  let posts = $state<PostMeta[]>([]);
  let selectedPath = $state<string | null>(null);
  let loading = $state(false);
  let saving = $state(false);
  let deploying = $state(false);
  let deployLog = $state("");

  // Editor state
  let editSlug = $state("");
  let editDate = $state("");
  let editDraft = $state(false);
  let editFeatured = $state(false);
  let editCategories = $state("");
  let editTitleEn = $state("");
  let editTitleZh = $state("");
  let editBodyEn = $state("");
  let editBodyZh = $state("");
  let editSection = $state<"posts" | "life-posts">("posts");
  let isNew = $state(false);

  async function loadPosts() {
    if (!browser) return;
    loading = true;
    try {
      const res = await fetch("/studio/api/posts");
      posts = (await res.json()) as PostMeta[];
    } catch { /* */ }
    loading = false;
  }

  async function selectPost(path: string) {
    if (!browser) return;
    isNew = false;
    selectedPath = path;
    loading = true;
    try {
      const res = await fetch(`/studio/api/posts/${path}`);
      const data = (await res.json()) as PostMeta & { content: PostContent };
      editSlug = data.slug;
      editDate = data.date.slice(0, 10);
      editDraft = data.draft;
      editFeatured = data.featured;
      editCategories = data.categories.join(", ");
      editTitleEn = data.title.en;
      editTitleZh = data.title.zh;
      editBodyEn = data.content.en;
      editBodyZh = data.content.zh;
      editSection = data.section === "life" ? "life-posts" : "posts";
    } catch { /* */ }
    loading = false;
  }

  function newPost() {
    isNew = true;
    selectedPath = null;
    const today = new Date().toISOString().slice(0, 10);
    editSlug = "";
    editDate = today;
    editDraft = true;
    editFeatured = false;
    editCategories = "";
    editTitleEn = "";
    editTitleZh = "";
    editBodyEn = "";
    editBodyZh = "";
    editSection = "posts";
  }

  async function savePost() {
    if (!browser || saving) return;
    saving = true;
    try {
      const body = {
        section: editSection,
        slug: editSlug || editTitleEn.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, ""),
        date: editDate,
        draft: editDraft,
        featured: editFeatured,
        categories: editCategories.split(",").map((s) => s.trim()).filter(Boolean),
        title: { en: editTitleEn, zh: editTitleZh },
        content: { en: editBodyEn, zh: editBodyZh },
      };

      const url = isNew
        ? "/studio/api/posts"
        : `/studio/api/posts/${selectedPath}`;
      const method = isNew ? "POST" : "PUT";

      const res = await fetch(url, {
        method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) { alert("Save failed"); return; }

      const result = (await res.json()) as { path: string };
      await loadPosts();
      selectPost(result.path);
    } finally {
      saving = false;
    }
  }

  async function deletePost() {
    if (!browser || !selectedPath || isNew) return;
    if (!confirm(`Delete ${selectedPath}?`)) return;
    try {
      await fetch(`/studio/api/posts/${selectedPath}`, { method: "DELETE" });
      selectedPath = null;
      await loadPosts();
    } catch { /* */ }
  }

  async function deploy() {
    if (!browser || deploying) return;
    deploying = true;
    deployLog = "Starting deploy...\n";
    try {
      const res = await fetch("/studio/api/deploy", { method: "POST" });
      const reader = res.body?.getReader();
      if (!reader) { deployLog += "No response stream.\n"; deploying = false; return; }
      const decoder = new TextDecoder();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        deployLog += decoder.decode(value, { stream: true });
      }
      deployLog += "\nDone.";
    } catch (e) {
      deployLog += `\nError: ${e}`;
    }
    deploying = false;
  }

  $effect(() => {
    if (browser) loadPosts();
  });
</script>

<div class="flex h-screen bg-printer-paper dark:bg-printer-paper-dark text-printer-ink dark:text-printer-ink-dark">
  <!-- Sidebar -->
  <aside class="w-64 border-r border-printer-ink/15 dark:border-printer-ink-dark/15 overflow-y-auto shrink-0">
    <div class="p-3 border-b border-printer-ink/10 dark:border-printer-ink-dark/10">
      <h1 class="font-mono text-[11px] tracking-[0.2em] uppercase">Studio</h1>
      <button
        type="button"
        onclick={newPost}
        class="mt-2 w-full rounded-sm border border-printer-accent dark:border-printer-accent-dark bg-printer-accent dark:bg-printer-accent-dark px-3 py-1 font-mono text-[10px] tracking-wider uppercase text-printer-paper dark:text-printer-paper-dark hover:opacity-90"
      >
        + New Post
      </button>
    </div>

    {#if loading && posts.length === 0}
      <p class="p-3 font-mono text-[10px] text-printer-ink-light/50">Loading...</p>
    {:else}
      {#each ["posts", "life-posts"] as section}
        {@const sectionPosts = posts.filter((p) => p.section === (section === "life-posts" ? "life" : "posts"))}
        {#if sectionPosts.length > 0}
          <div class="px-3 py-1 font-mono text-[9px] tracking-wider uppercase text-printer-ink-light/50">
            {section}
          </div>
          {#each sectionPosts as post (post.path)}
            <button
              type="button"
              onclick={() => selectPost(post.path)}
              class="w-full text-left px-3 py-1.5 hover:bg-printer-ink/5 dark:hover:bg-printer-ink-dark/5 transition-colors{selectedPath === post.path ? ' bg-printer-ink/10 dark:bg-printer-ink-dark/10' : ''}"
            >
              <div class="text-[12px] leading-tight truncate">{post.title.en || post.slug}</div>
              <div class="font-mono text-[9px] text-printer-ink-light/50">
                {post.date.slice(0, 10)}
                {#if post.draft}<span class="ml-1 text-printer-accent">Draft</span>{/if}
              </div>
            </button>
          {/each}
        {/if}
      {/each}
    {/if}
  </aside>

  <!-- Main -->
  <main class="flex-1 overflow-y-auto">
    {#if selectedPath || isNew}
      <div class="max-w-3xl mx-auto p-6">
        <!-- Top bar -->
        <div class="flex items-center gap-2 mb-6">
          <h2 class="font-mono text-[11px] tracking-[0.2em] uppercase">
            {isNew ? "New Post" : editTitleEn || editSlug}
          </h2>
          <div class="ml-auto flex gap-2">
            <button
              type="button"
              onclick={deploy}
              disabled={deploying}
              class="rounded-sm border border-printer-ink/15 dark:border-printer-ink-dark/15 px-3 py-1 font-mono text-[10px] tracking-wider uppercase hover:border-printer-accent disabled:opacity-50"
            >
              {deploying ? "Deploying..." : "Deploy"}
            </button>
            <button
              type="button"
              onclick={savePost}
              disabled={saving}
              class="rounded-sm border border-printer-accent dark:border-printer-accent-dark bg-printer-accent dark:bg-printer-accent-dark px-4 py-1 font-mono text-[10px] tracking-wider uppercase text-printer-paper dark:text-printer-paper-dark hover:opacity-90 disabled:opacity-50"
            >
              {saving ? "Saving..." : "Save"}
            </button>
            {#if !isNew}
              <button
                type="button"
                onclick={deletePost}
                class="rounded-sm border border-red-500/30 px-3 py-1 font-mono text-[10px] tracking-wider uppercase text-red-500 hover:bg-red-500/10"
              >
                Delete
              </button>
            {/if}
          </div>
        </div>

        <!-- Fields -->
        <div class="grid grid-cols-2 gap-3 mb-4">
          <label class="block">
            <span class="font-mono text-[9px] tracking-wider text-printer-ink-light/50">Section</span>
            <select
              bind:value={editSection}
              class="w-full mt-0.5 rounded-sm border border-printer-ink/20 dark:border-printer-ink-dark/20 bg-transparent px-2 py-1 text-[12px]"
            >
              <option value="posts">posts</option>
              <option value="life-posts">life-posts</option>
            </select>
          </label>
          <label class="block">
            <span class="font-mono text-[9px] tracking-wider text-printer-ink-light/50">Date</span>
            <input
              type="date"
              bind:value={editDate}
              class="w-full mt-0.5 rounded-sm border border-printer-ink/20 dark:border-printer-ink-dark/20 bg-transparent px-2 py-1 text-[12px]"
            />
          </label>
          <label class="block">
            <span class="font-mono text-[9px] tracking-wider text-printer-ink-light/50">Slug</span>
            <input
              type="text"
              bind:value={editSlug}
              class="w-full mt-0.5 rounded-sm border border-printer-ink/20 dark:border-printer-ink-dark/20 bg-transparent px-2 py-1 text-[12px] font-mono"
            />
          </label>
          <label class="block">
            <span class="font-mono text-[9px] tracking-wider text-printer-ink-light/50">Categories (comma-separated)</span>
            <input
              type="text"
              bind:value={editCategories}
              class="w-full mt-0.5 rounded-sm border border-printer-ink/20 dark:border-printer-ink-dark/20 bg-transparent px-2 py-1 text-[12px]"
            />
          </label>
        </div>

        <div class="flex gap-4 mb-4">
          <label class="flex items-center gap-1">
            <input type="checkbox" bind:checked={editDraft} class="rounded-sm" />
            <span class="font-mono text-[10px] tracking-wider">Draft</span>
          </label>
          <label class="flex items-center gap-1">
            <input type="checkbox" bind:checked={editFeatured} class="rounded-sm" />
            <span class="font-mono text-[10px] tracking-wider">Featured</span>
          </label>
        </div>

        <!-- EN editor -->
        <div class="mb-4">
          <div class="font-mono text-[10px] tracking-wider text-printer-accent mb-1">English</div>
          <label class="block mb-2">
            <span class="font-mono text-[9px] tracking-wider text-printer-ink-light/50">Title</span>
            <input
              type="text"
              bind:value={editTitleEn}
              maxlength="99"
              class="w-full mt-0.5 rounded-sm border border-printer-ink/20 dark:border-printer-ink-dark/20 bg-transparent px-2 py-1 text-[13px]"
            />
          </label>
          <label class="block">
            <span class="font-mono text-[9px] tracking-wider text-printer-ink-light/50">Body (Markdown)</span>
            <textarea
              bind:value={editBodyEn}
              rows="16"
              class="w-full mt-0.5 rounded-sm border border-printer-ink/20 dark:border-printer-ink-dark/20 bg-transparent px-2 py-1.5 text-[13px] leading-relaxed font-mono resize-y"
            ></textarea>
          </label>
        </div>

        <!-- ZH editor -->
        <div class="mb-6">
          <div class="font-mono text-[10px] tracking-wider text-printer-accent mb-1">中文</div>
          <label class="block mb-2">
            <span class="font-mono text-[9px] tracking-wider text-printer-ink-light/50">标题</span>
            <input
              type="text"
              bind:value={editTitleZh}
              maxlength="99"
              class="w-full mt-0.5 rounded-sm border border-printer-ink/20 dark:border-printer-ink-dark/20 bg-transparent px-2 py-1 text-[13px]"
            />
          </label>
          <label class="block">
            <span class="font-mono text-[9px] tracking-wider text-printer-ink-light/50">正文 (Markdown)</span>
            <textarea
              bind:value={editBodyZh}
              rows="16"
              class="w-full mt-0.5 rounded-sm border border-printer-ink/20 dark:border-printer-ink-dark/20 bg-transparent px-2 py-1.5 text-[13px] leading-relaxed font-mono resize-y"
            ></textarea>
          </label>
        </div>

        <!-- Deploy log -->
        {#if deployLog}
          <pre
            class="rounded-sm border border-printer-ink/15 dark:border-printer-ink-dark/15 bg-printer-paper dark:bg-printer-paper-dark px-3 py-2 font-mono text-[10px] leading-relaxed whitespace-pre-wrap overflow-x-auto"
          >{deployLog}</pre>
        {/if}
      </div>
    {:else}
      <div class="flex items-center justify-center h-full">
        <p class="font-mono text-[11px] tracking-wider text-printer-ink-light/40">
          Select a post or create a new one.
        </p>
      </div>
    {/if}
  </main>
</div>
