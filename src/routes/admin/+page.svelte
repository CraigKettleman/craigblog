<script lang="ts">
  import { browser } from "$app/environment";
  import PrintedDivider from "$lib/components/PrintedDivider.svelte";

  export const prerender = false;

  interface AdminComment {
    id: number;
    thread: string;
    parentId: number | null;
    content: string;
    status: "published" | "hidden";
    createdAt: string;
    ipHash: string;
    ua?: string;
    author: {
      type: "guest" | "github";
      name: string;
      website?: string;
      login?: string;
      avatarUrl?: string | null;
      isOwner: boolean;
    };
  }

  const KEY_STORE = "hhy.admin.key";

  let key = $state<string | null>(null);
  let keyInput = $state("");
  let authed = $state(false);
  let loading = $state(false);
  let errorMsg = $state<string | null>(null);
  let all = $state<AdminComment[]>([]);

  let editingId = $state<number | null>(null);
  let editContent = $state("");
  let busyId = $state<number | null>(null);

  let groups = $derived.by(() => {
    const map = new Map<string, AdminComment[]>();
    for (const comment of all) {
      const arr = map.get(comment.thread) ?? [];
      arr.push(comment);
      map.set(comment.thread, arr);
    }
    return [...map.entries()];
  });

  function readStoredKey() {
    if (!browser) return;
    const stored = sessionStorage.getItem(KEY_STORE);
    if (stored) {
      key = stored;
      loadAll();
    }
  }

  $effect(() => {
    readStoredKey();
  });

  async function api(path: string, init?: RequestInit): Promise<Response> {
    const headers = new Headers(init?.headers);
    if (key) headers.set("Authorization", `Bearer ${key}`);
    if (init?.body) headers.set("Content-Type", "application/json");
    return fetch(path, { ...init, headers });
  }

  async function loadAll() {
    if (!key) return;
    loading = true;
    errorMsg = null;
    try {
      const res = await api("/api/comments/admin");
      if (res.status === 401) {
        authed = false;
        errorMsg = "密钥不正确。";
        return;
      }
      if (!res.ok) throw new Error();
      const data = (await res.json()) as { total: number; comments: AdminComment[] };
      all = data.comments;
      authed = true;
    } catch {
      errorMsg = "加载失败，请重试。";
    } finally {
      loading = false;
    }
  }

  function submitKey() {
    key = keyInput.trim();
    if (!key) return;
    sessionStorage.setItem(KEY_STORE, key);
    loadAll();
  }

  function signOut() {
    sessionStorage.removeItem(KEY_STORE);
    key = null;
    keyInput = "";
    authed = false;
    all = [];
  }

  async function remove(id: number) {
    if (!confirm("删除这条评论（连同其回复）？")) return;
    busyId = id;
    try {
      const res = await api(`/api/comments/admin/${id}`, { method: "DELETE" });
      if (!res.ok) throw new Error();
      all = all.filter((c) => c.id !== id && c.parentId !== id);
    } catch {
      alert("删除失败。");
    } finally {
      busyId = null;
    }
  }

  async function toggleStatus(comment: AdminComment) {
    busyId = comment.id;
    try {
      const next = comment.status === "published" ? "hidden" : "published";
      const res = await api(`/api/comments/admin/${comment.id}`, {
        method: "PATCH",
        body: JSON.stringify({ status: next }),
      });
      if (!res.ok) throw new Error();
      all = all.map((c) => (c.id === comment.id ? { ...c, status: next } : c));
    } catch {
      alert("操作失败。");
    } finally {
      busyId = null;
    }
  }

  function startEdit(comment: AdminComment) {
    editingId = comment.id;
    editContent = comment.content;
  }

  function cancelEdit() {
    editingId = null;
    editContent = "";
  }

  async function saveEdit(id: number) {
    const content = editContent.trim();
    if (!content) {
      alert("内容不能为空。");
      return;
    }
    busyId = id;
    try {
      const res = await api(`/api/comments/admin/${id}`, {
        method: "PATCH",
        body: JSON.stringify({ content }),
      });
      if (!res.ok) throw new Error();
      all = all.map((c) => (c.id === id ? { ...c, content } : c));
      editingId = null;
    } catch {
      alert("保存失败。");
    } finally {
      busyId = null;
    }
  }

  function escapeHtml(s: string): string {
    return s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatTime(iso: string): string {
    const d = new Date(iso);
    return `${d.toLocaleDateString("zh-CN")} ${d.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })}`;
  }
</script>

<div>
  <div class="flex items-center gap-2">
    <span
      class="font-mono text-[10px] tracking-[0.25em] uppercase text-printer-ink-light dark:text-printer-ink-dark/50"
    >
      ✎ 评论后台
    </span>
    {#if authed}
      <span
        class="font-mono text-[10px] tracking-[0.25em] uppercase text-printer-ink-light/70 dark:text-printer-ink-dark/30"
      >
        · {all.length} 条评论
      </span>
    {/if}
  </div>

  <PrintedDivider style="dashed" />

  {#if !authed}
    <form
      class="mt-6 max-w-sm"
      onsubmit={(e) => {
        e.preventDefault();
        submitKey();
      }}
    >
      <label class="block">
        <span
          class="block font-mono text-[10px] tracking-wider uppercase text-printer-ink-light dark:text-printer-ink-dark/40 mb-1"
        >
          管理密钥
        </span>
        <input
          type="password"
          bind:value={keyInput}
          placeholder="COMMENTS_ADMIN_KEY"
          class="w-full rounded-sm border border-printer-ink/20 dark:border-printer-ink-dark/20 bg-transparent px-2 py-1.5 text-[13px] text-printer-ink dark:text-printer-ink-dark focus:outline-none focus:border-printer-accent dark:focus:border-printer-accent-dark"
        />
      </label>
      {#if errorMsg}
        <p class="mt-2 font-mono text-[10px] tracking-wider text-red-600 dark:text-red-400">
          {errorMsg}
        </p>
      {/if}
      <button
        type="submit"
        class="mt-3 rounded-sm border border-printer-accent dark:border-printer-accent-dark bg-printer-accent dark:bg-printer-accent-dark px-4 py-1.5 font-mono text-[11px] tracking-wider uppercase text-printer-paper dark:text-printer-paper-dark hover:opacity-90 transition-opacity"
      >
        进入后台
      </button>
      <p
        class="mt-4 font-mono text-[10px] tracking-wider text-printer-ink-light dark:text-printer-ink-dark/40"
      >
        密钥在部署时生成并打印一次，存在服务器 /var/www/hhy.homes/.env 的 COMMENTS_ADMIN_KEY。
      </p>
    </form>
  {:else}
    <div class="mt-6 flex items-center justify-between">
      <p
        class="font-mono text-[10px] tracking-wider uppercase text-printer-ink-light dark:text-printer-ink-dark/40"
      >
        {#if loading}加载中…{/if}
      </p>
      <button
        type="button"
        onclick={signOut}
        class="font-mono text-[10px] tracking-wider uppercase text-printer-accent dark:text-printer-accent-dark hover:underline"
      >
        退出
      </button>
    </div>

    {#if groups.length === 0}
      <div
        class="mt-4 rounded-sm border border-dashed border-printer-ink/15 dark:border-printer-ink-dark/15 bg-printer-paper dark:bg-printer-paper-dark px-4 py-6 text-center"
      >
        <p
          class="font-mono text-[10px] tracking-wider uppercase text-printer-ink-light dark:text-printer-ink-dark/40"
        >
          还没有评论。
        </p>
      </div>
    {:else}
      {#each groups as [thread, comments]}
        <div class="mt-6">
          <div
            class="flex items-center gap-2 border-b border-printer-ink/15 dark:border-printer-ink-dark/15 pb-1"
          >
            <span
              class="font-mono text-[10px] tracking-[0.25em] uppercase text-printer-accent dark:text-printer-accent-dark"
            >
              {thread}
            </span>
            <span
              class="font-mono text-[10px] text-printer-ink-light/70 dark:text-printer-ink-dark/30"
            >
              {comments.length}
            </span>
          </div>
          <ul class="mt-2 space-y-2">
            {#each comments as comment (comment.id)}
              <li
                class="rounded-sm border border-printer-ink/10 dark:border-printer-ink-dark/10 bg-printer-paper dark:bg-printer-paper-dark px-3 py-2"
              >
                <div class="flex items-center gap-2">
                  {#if comment.author.avatarUrl}
                    <img
                      src={comment.author.avatarUrl}
                      alt=""
                      class="w-5 h-5 rounded-full object-cover"
                    />
                  {:else}
                    <span
                      class="w-5 h-5 inline-flex items-center justify-center rounded-full bg-printer-accent dark:bg-printer-accent-dark text-printer-paper dark:text-printer-paper-dark font-mono text-[9px]"
                    >
                      {(comment.author.name.trim()[0] ?? "?").toUpperCase()}
                    </span>
                  {/if}
                  <span class="text-[12px] font-medium text-printer-ink dark:text-printer-ink-dark">
                    {comment.author.name}
                  </span>
                  {#if comment.author.isOwner}
                    <span
                      class="rounded-sm border border-printer-accent/50 dark:border-printer-accent-dark/50 px-1 font-mono text-[9px] tracking-wider uppercase text-printer-accent dark:text-printer-accent-dark"
                    >
                      博主
                    </span>
                  {/if}
                  <span
                    class="font-mono text-[9px] text-printer-ink-light/60 dark:text-printer-ink-dark/30"
                  >
                    {comment.status === "hidden" ? "已隐藏" : ""}
                  </span>
                  <span
                    class="ml-auto font-mono text-[9px] text-printer-ink-light/60 dark:text-printer-ink-dark/30"
                  >
                    {formatTime(comment.createdAt)} · ip:{comment.ipHash.slice(0, 8)}
                  </span>
                </div>

                {#if editingId === comment.id}
                  <textarea
                    bind:value={editContent}
                    rows="3"
                    class="mt-2 w-full rounded-sm border border-printer-ink/20 dark:border-printer-ink-dark/20 bg-transparent px-2 py-1.5 text-[12px] leading-relaxed text-printer-ink dark:text-printer-ink-dark focus:outline-none focus:border-printer-accent dark:focus:border-printer-accent-dark"
                  ></textarea>
                {:else}
                  <p
                    class="mt-1 text-[12px] leading-relaxed text-printer-ink dark:text-printer-ink-dark whitespace-pre-wrap break-words"
                  >
                    {@html escapeHtml(comment.content)}
                  </p>
                {/if}

                <div class="mt-2 flex items-center gap-3">
                  {#if editingId === comment.id}
                    <button
                      type="button"
                      onclick={() => saveEdit(comment.id)}
                      disabled={busyId === comment.id}
                      class="font-mono text-[10px] tracking-wider uppercase text-printer-accent dark:text-printer-accent-dark hover:underline disabled:opacity-50"
                    >
                      保存
                    </button>
                    <button
                      type="button"
                      onclick={cancelEdit}
                      class="font-mono text-[10px] tracking-wider uppercase text-printer-ink-light dark:text-printer-ink-dark/40 hover:underline"
                    >
                      取消
                    </button>
                  {:else}
                    <button
                      type="button"
                      onclick={() => startEdit(comment)}
                      class="font-mono text-[10px] tracking-wider uppercase text-printer-ink-light dark:text-printer-ink-dark/40 hover:underline"
                    >
                      编辑
                    </button>
                    <button
                      type="button"
                      onclick={() => toggleStatus(comment)}
                      disabled={busyId === comment.id}
                      class="font-mono text-[10px] tracking-wider uppercase text-printer-ink-light dark:text-printer-ink-dark/40 hover:underline disabled:opacity-50"
                    >
                      {comment.status === "published" ? "隐藏" : "恢复"}
                    </button>
                    <button
                      type="button"
                      onclick={() => remove(comment.id)}
                      disabled={busyId === comment.id}
                      class="font-mono text-[10px] tracking-wider uppercase text-red-600 dark:text-red-400 hover:underline disabled:opacity-50"
                    >
                      删除
                    </button>
                  {/if}
                </div>
              </li>
            {/each}
          </ul>
        </div>
      {/each}
    {/if}
  {/if}
</div>
