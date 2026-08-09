<script lang="ts">
  import { browser } from "$app/environment";
  import { getDictionary, type Language } from "$lib/dictionaries";
  import PrintedDivider from "$lib/components/PrintedDivider.svelte";
  import LoginModal from "$lib/components/LoginModal.svelte";

  let { lang, thread }: { lang: Language; thread: string } = $props();

  let dictionary = $derived(getDictionary(lang));
  let t = $derived(dictionary.comments);

  interface PublicComment {
    id: number;
    parentId: number | null;
    parentName?: string;
    content: string;
    createdAt: string;
    author: {
      type: string;
      name: string;
      website?: string;
      login?: string;
      avatarUrl?: string | null;
      isOwner: boolean;
    };
    replies: PublicComment[];
    upvotes: number;
    downvotes: number;
    userVote?: "up" | "down" | null;
  }

  interface SessionUser {
    provider: string;
    login: string;
    name: string;
    avatarUrl: string | null;
  }

  let comments = $state<PublicComment[]>([]);
  let total = $state(0);
  let loading = $state(true);
  let loadFailed = $state(false);
  let githubEnabled = $state(false);
  let accountEnabled = $state(false);

  let token = $state<string | null>(null);
  let user = $state<SessionUser | null>(null);

  let content = $state("");
  let honeypot = $state("");
  let replyTo = $state<{ id: number; name: string } | null>(null);
  let submitting = $state(false);
  let formError = $state<string | null>(null);

  let sort = $state<"newest" | "hot">("newest");
  let showLoginModal = $state(false);
  let expandedThreads = $state<Set<number>>(new Set());
  let deleting = $state<Set<number>>(new Set());
  let voting = $state<Set<number>>(new Set());
  let placeholder = $state("");

  // ---- localStorage helpers ----

  function readSession() {
    if (!browser) return;
    token = localStorage.getItem("hhy.comments.token");
    const raw = localStorage.getItem("hhy.comments.user");
    if (raw) {
      try {
        user = JSON.parse(raw) as SessionUser;
      } catch {
        user = null;
      }
    } else {
      user = null;
    }
  }

  function loadDeleteTokens(): Record<string, string> {
    if (!browser) return {};
    try {
      return JSON.parse(localStorage.getItem("hhy.comments.deleteTokens") ?? "{}") as Record<string, string>;
    } catch {
      return {};
    }
  }

  function saveDeleteToken(key: string, dt: string) {
    if (!browser) return;
    const map = loadDeleteTokens();
    map[key] = dt;
    localStorage.setItem("hhy.comments.deleteTokens", JSON.stringify(map));
  }

  function getDeleteToken(commentId: number): string | null {
    const map = loadDeleteTokens();
    return map[`${thread}/${commentId}`] ?? null;
  }

  // ---- data ----

  async function loadComments() {
    if (!browser) return;
    loading = true;
    loadFailed = false;
    try {
      const headers: Record<string, string> = {};
      if (token) headers.Authorization = `Bearer ${token}`;
      const res = await fetch(`/api/comments/${thread}?sort=${sort}`, { headers });
      if (!res.ok) throw new Error();
      const data = (await res.json()) as { count: number; comments: PublicComment[] };
      comments = data.comments;
      total = data.count;
    } catch {
      loadFailed = true;
    } finally {
      loading = false;
    }
  }

  async function loadConfig() {
    if (!browser) return;
    try {
      const res = await fetch("/api/comments/config");
      const data = (await res.json()) as { githubEnabled: boolean; accountEnabled: boolean };
      githubEnabled = data.githubEnabled;
      accountEnabled = data.accountEnabled;
    } catch {
      githubEnabled = false;
      accountEnabled = false;
    }
  }

  $effect(() => {
    if (!browser) return;
    readSession();
    loadConfig();
    // eslint-disable-next-line @typescript-eslint/no-unused-expressions
    placeholder = t.contentPlaceholders[Math.floor(Math.random() * t.contentPlaceholders.length)] ?? t.content;
  });

  $effect(() => {
    if (!browser) return;
    loadComments();
  });

  function setSort(next: "newest" | "hot") {
    if (sort === next) return;
    sort = next;
    loadComments();
  }

  // ---- auth ----

  function onauthed(u: SessionUser, tk: string) {
    user = u;
    token = tk;
  }

  function signOut() {
    if (!browser) return;
    localStorage.removeItem("hhy.comments.token");
    localStorage.removeItem("hhy.comments.user");
    token = null;
    user = null;
  }

  // ---- actions ----

  async function handleDelete(comment: PublicComment) {
    if (!browser) return;
    if (deleting.has(comment.id)) return;
    if (!confirm(t.deleteConfirm)) return;

    deleting.add(comment.id);
    deleting = new Set(deleting);
    try {
      const body: Record<string, unknown> = { id: comment.id };
      const dt = getDeleteToken(comment.id);
      if (dt) body.deleteToken = dt;
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      if (token) headers.Authorization = `Bearer ${token}`;

      const res = await fetch(`/api/comments/${thread}`, {
        method: "DELETE",
        headers,
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error();
      await loadComments();
    } catch {
      // silent
    } finally {
      deleting.delete(comment.id);
      deleting = new Set(deleting);
    }
  }

  async function handleVote(comment: PublicComment, action: "up" | "down") {
    if (!browser) return;
    if (!user) { showLoginModal = true; return; }
    if (voting.has(comment.id)) return;
    voting.add(comment.id);
    voting = new Set(voting);
    try {
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      if (token) headers.Authorization = `Bearer ${token}`;
      const res = await fetch(`/api/comments/${thread}`, {
        method: "PATCH",
        headers,
        body: JSON.stringify({ commentId: comment.id, action }),
      });
      if (res.ok) {
        const data = (await res.json()) as { upvotes: number; downvotes: number; userVote: "up" | "down" | null };
        comment.upvotes = data.upvotes;
        comment.downvotes = data.downvotes;
        comment.userVote = data.userVote;
      }
    } catch {
      // silent
    } finally {
      voting.delete(comment.id);
      voting = new Set(voting);
    }
  }

  function toggleExpand(commentId: number) {
    const next = new Set(expandedThreads);
    if (next.has(commentId)) {
      next.delete(commentId);
    } else {
      next.add(commentId);
    }
    expandedThreads = next;
  }

  // ---- rendering helpers ----

  function escapeHtml(s: string): string {
    return s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function linkify(text: string): string {
    const escaped = escapeHtml(text);
    return escaped.replace(
      /(https?:\/\/[^\s<]+)/g,
      '<a href="$1" target="_blank" rel="nofollow noopener" class="text-printer-accent dark:text-printer-accent-dark underline">$1</a>',
    );
  }

  function formatTime(iso: string): string {
    const date = new Date(iso);
    const diff = Date.now() - date.getTime();
    const minute = 60_000;
    const hour = 3_600_000;
    const day = 86_400_000;
    if (diff < minute) return lang === "zh" ? "刚刚" : "just now";
    if (diff < hour) {
      const m = Math.floor(diff / minute);
      return lang === "zh" ? `${m} 分钟前` : `${m} min ago`;
    }
    if (diff < day) {
      const h = Math.floor(diff / hour);
      return lang === "zh" ? `${h} 小时前` : `${h} hours ago`;
    }
    return date.toLocaleDateString(lang === "zh" ? "zh-CN" : "en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  }

  function fullTime(iso: string): string {
    return new Date(iso).toLocaleString(lang === "zh" ? "zh-CN" : "en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  }

  function initialOf(name: string): string {
    return (name.trim()[0] ?? "?").toUpperCase();
  }

  function parentName(c: PublicComment): string {
    return c.parentName ?? "";
  }

  function canDelete(comment: PublicComment): boolean {
    return !!getDeleteToken(comment.id);
  }

  // ---- submit ----

  async function submit() {
    if (submitting) return;
    formError = null;
    const body = content.trim();

    if (body.length < 2) {
      formError = t.contentRequired;
      return;
    }
    if (body.length > 2000) {
      formError = t.contentTooLong;
      return;
    }

    submitting = true;
    try {
      const payload: Record<string, unknown> = {
        content: body,
        parentId: replyTo?.id ?? null,
        email_confirm: honeypot,
      };
      if (!user) {
        payload.name = t.anonymous;
      }
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      if (token) headers.Authorization = `Bearer ${token}`;

      const res = await fetch(`/api/comments/${thread}`, {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
      });
      if (res.status === 429) {
        formError = t.rateLimited;
        return;
      }
      if (!res.ok) {
        formError = t.serverError;
        return;
      }
      const data = (await res.json()) as { id: number; deleteToken?: string };
      if (data.deleteToken) {
        saveDeleteToken(`${thread}/${data.id}`, data.deleteToken);
      }
      content = "";
      replyTo = null;
      await loadComments();
    } catch {
      formError = t.serverError;
    } finally {
      submitting = false;
    }
  }
</script>

<section>
  <PrintedDivider style="dashed" />

  <div class="mt-6">
    <!-- Header row: title + sort toggle -->
    <div class="flex items-baseline gap-2 mb-4">
      <span
        class="font-mono text-[10px] tracking-[0.25em] uppercase text-printer-ink-light dark:text-printer-ink-dark/50"
      >
        ✎ {t.title}
      </span>
      {#if !loading}
        <span
          class="font-mono text-[10px] tracking-[0.25em] uppercase text-printer-ink-light/70 dark:text-printer-ink-dark/30"
        >
          · {t.count(total)}
        </span>
      {/if}
      <div class="ml-auto flex items-center gap-1.5">
        <button
          type="button"
          onclick={() => setSort("newest")}
          class="font-mono text-[10px] tracking-wider uppercase hover:text-printer-accent dark:hover:text-printer-accent-dark transition-colors {sort === 'newest' ? 'text-printer-accent dark:text-printer-accent-dark' : 'text-printer-ink-light/60 dark:text-printer-ink-dark/40'}"
        >
          {t.sortNewest}
        </button>
        <span class="text-printer-ink-light/30 dark:text-printer-ink-dark/20">|</span>
        <button
          type="button"
          onclick={() => setSort("hot")}
          class="font-mono text-[10px] tracking-wider uppercase hover:text-printer-accent dark:hover:text-printer-accent-dark transition-colors {sort === 'hot' ? 'text-printer-accent dark:text-printer-accent-dark' : 'text-printer-ink-light/60 dark:text-printer-ink-dark/40'}"
        >
          {t.sortHot}
        </button>
      </div>
    </div>

    <!-- Comment form: new comments at top; replies render inline under the target comment -->
    {#if replyTo}
      <div
        class="mb-6 flex items-center gap-2 rounded-sm border border-dashed border-printer-accent/50 dark:border-printer-accent-dark/50 bg-printer-paper dark:bg-printer-paper-dark px-3 py-2"
      >
        <span
          class="font-mono text-[10px] tracking-wider uppercase text-printer-ink-light dark:text-printer-ink-dark/40"
        >
          {t.replyingTo(replyTo.name)} ↓
        </span>
        <button
          type="button"
          onclick={() => (replyTo = null)}
          class="ml-auto font-mono text-[10px] tracking-wider uppercase text-printer-accent dark:text-printer-accent-dark hover:underline"
        >
          {t.cancel}
        </button>
      </div>
    {:else}
      <form class="mb-6" onsubmit={(e) => { e.preventDefault(); submit(); }}>
        <div
          class="rounded-sm border border-printer-ink/15 dark:border-printer-ink-dark/15 bg-printer-paper dark:bg-printer-paper-dark px-3 py-3"
        >
          {#if !user}
            <p
              class="mb-2 font-mono text-[10px] tracking-wider text-printer-ink-light dark:text-printer-ink-dark/40"
            >
              {t.anonymousHint}
            </p>
          {/if}

          <!-- Honeypot -->
          <label class="hidden" aria-hidden="true">
            <input tabindex="-1" autocomplete="off" bind:value={honeypot} name="email_confirm" />
          </label>

          <textarea
            bind:value={content}
            rows="3"
            maxlength="2000"
            placeholder={placeholder}
            class="w-full rounded-sm border border-printer-ink/20 dark:border-printer-ink-dark/20 bg-transparent px-2 py-1.5 text-[13px] leading-relaxed text-printer-ink dark:text-printer-ink-dark focus:outline-none focus:border-printer-accent dark:focus:border-printer-accent-dark resize-y"
          ></textarea>

          {#if formError}
            <p
              class="mt-2 font-mono text-[10px] tracking-wider text-red-600 dark:text-red-400"
            >
              {formError}
            </p>
          {/if}

          <div class="mt-3 flex items-center justify-end gap-2">
            {#if user}
              <span class="mr-auto text-[11px] text-printer-ink-light dark:text-printer-ink-dark/40">
                {t.postingAs(user.login)}
                <button
                  type="button"
                  onclick={signOut}
                  class="ml-1 font-mono text-[10px] tracking-wider text-printer-accent dark:text-printer-accent-dark hover:underline"
                >
                  ({t.signOut})
                </button>
              </span>
            {:else}
              <button
                type="button"
                onclick={() => (showLoginModal = true)}
                class="mr-auto rounded-sm border border-printer-ink/15 dark:border-printer-ink-dark/15 bg-printer-paper dark:bg-printer-paper-dark px-3 py-1 font-mono text-[10px] tracking-wider uppercase text-printer-ink dark:text-printer-ink-dark hover:border-printer-accent dark:hover:border-printer-accent-dark transition-colors"
              >
                {t.signIn}
              </button>
            {/if}

            <button
              type="submit"
              disabled={submitting}
              class="rounded-sm border border-printer-accent dark:border-printer-accent-dark bg-printer-accent dark:bg-printer-accent-dark px-4 py-1.5 font-mono text-[11px] tracking-wider uppercase text-printer-paper dark:text-printer-paper-dark hover:opacity-90 disabled:opacity-50 transition-opacity"
            >
              {submitting ? t.submitting : t.submit}
            </button>
          </div>
        </div>
      </form>
    {/if}

    <!-- Comment list -->
    {#if loading}
      <div
        class="rounded-sm border border-printer-ink/15 dark:border-printer-ink-dark/15 bg-printer-paper dark:bg-printer-paper-dark px-4 py-6 text-center"
      >
        <p
          class="font-mono text-[10px] tracking-wider uppercase text-printer-ink-light dark:text-printer-ink-dark/40"
        >
          {t.loading}
        </p>
      </div>
    {:else if loadFailed}
      <div
        class="rounded-sm border border-dashed border-printer-ink/15 dark:border-printer-ink-dark/15 bg-printer-paper dark:bg-printer-paper-dark px-4 py-6 text-center"
      >
        <p
          class="font-mono text-[10px] tracking-wider uppercase text-printer-ink-light dark:text-printer-ink-dark/40"
        >
          {t.error}
        </p>
      </div>
    {:else if comments.length === 0}
      <div
        class="rounded-sm border border-dashed border-printer-ink/15 dark:border-printer-ink-dark/15 bg-printer-paper dark:bg-printer-paper-dark px-4 py-6 text-center"
      >
        <p
          class="font-mono text-[10px] tracking-wider uppercase text-printer-ink-light dark:text-printer-ink-dark/40"
        >
          {t.empty}
        </p>
      </div>
    {:else}
      <ul class="space-y-4">
        {#each comments as comment (comment.id)}
          <li>
            <div
              class="rounded-sm border border-printer-ink/15 dark:border-printer-ink-dark/15 bg-printer-paper dark:bg-printer-paper-dark px-3 py-3"
            >
              {@render CommentMeta({ comment, topLevel: true })}
              <div
                class="mt-2 text-[13px] leading-relaxed text-printer-ink dark:text-printer-ink-dark whitespace-pre-wrap break-words"
              >
                {@html linkify(comment.content)}
              </div>
              <!-- Action buttons -->
              <div class="mt-2 flex items-center gap-3">
                <button
                  type="button"
                  onclick={() => handleVote(comment, "up")}
                  disabled={voting.has(comment.id)}
                  title={user ? "" : t.loginToVote}
                  class="font-mono text-[10px] tracking-wider uppercase flex items-center gap-1 transition-colors disabled:opacity-50 {comment.userVote === 'up' ? 'text-printer-accent dark:text-printer-accent-dark' : 'text-printer-ink-light dark:text-printer-ink-dark/40 hover:text-printer-accent dark:hover:text-printer-accent-dark'}"
                >
                  ▲ {comment.upvotes > 0 ? comment.upvotes : ""} {t.upvote}
                </button>
                <button
                  type="button"
                  onclick={() => handleVote(comment, "down")}
                  disabled={voting.has(comment.id)}
                  title={user ? "" : t.loginToVote}
                  class="font-mono text-[10px] tracking-wider uppercase flex items-center gap-1 transition-colors disabled:opacity-50 {comment.userVote === 'down' ? 'text-printer-accent dark:text-printer-accent-dark' : 'text-printer-ink-light dark:text-printer-ink-dark/40 hover:text-printer-accent dark:hover:text-printer-accent-dark'}"
                >
                  ▼ {comment.downvotes > 0 ? comment.downvotes : ""} {t.downvote}
                </button>
                <button
                  type="button"
                  onclick={() => (replyTo = { id: comment.id, name: comment.author.name })}
                  class="font-mono text-[10px] tracking-wider uppercase text-printer-accent dark:text-printer-accent-dark hover:underline"
                >
                  {t.reply}
                </button>
                {#if canDelete(comment)}
                  <button
                    type="button"
                    onclick={() => handleDelete(comment)}
                    disabled={deleting.has(comment.id)}
                    class="font-mono text-[10px] tracking-wider uppercase text-red-500/70 hover:text-red-500 disabled:opacity-50"
                  >
                    {t.delete}
                  </button>
                {/if}
              </div>
            </div>

            {#if replyTo?.id === comment.id}
              {@render ReplyInline()}
            {/if}

            <!-- Replies: collapsed by default -->
            {#if comment.replies.length > 0}
              {#if expandedThreads.has(comment.id)}
                <ul class="mt-2 ml-3 border-l-2 border-printer-ink/10 dark:border-printer-ink-dark/10 pl-3 space-y-2">
                  {#each comment.replies as reply (reply.id)}
                    <li>
                      <div
                        class="rounded-sm border border-printer-ink/10 dark:border-printer-ink-dark/10 bg-printer-paper dark:bg-printer-paper-dark px-3 py-2"
                      >
                        {@render CommentMeta({ comment: reply, topLevel: false })}
                        <p
                          class="mt-1 font-mono text-[10px] tracking-wider text-printer-ink-light dark:text-printer-ink-dark/40"
                        >
                          {t.replyLine(reply.author.name, parentName(reply))}
                        </p>
                        <div
                          class="mt-1 text-[13px] leading-relaxed text-printer-ink dark:text-printer-ink-dark whitespace-pre-wrap break-words"
                        >
                          {@html linkify(reply.content)}
                        </div>
                        <div class="mt-1 flex items-center gap-3">
                          <button
                            type="button"
                            onclick={() => handleVote(reply, "up")}
                            disabled={voting.has(reply.id)}
                            title={user ? "" : t.loginToVote}
                            class="font-mono text-[10px] tracking-wider uppercase flex items-center gap-1 transition-colors disabled:opacity-50 {reply.userVote === 'up' ? 'text-printer-accent dark:text-printer-accent-dark' : 'text-printer-ink-light dark:text-printer-ink-dark/40 hover:text-printer-accent dark:hover:text-printer-accent-dark'}"
                          >
                            ▲ {reply.upvotes > 0 ? reply.upvotes : ""} {t.upvote}
                          </button>
                          <button
                            type="button"
                            onclick={() => handleVote(reply, "down")}
                            disabled={voting.has(reply.id)}
                            title={user ? "" : t.loginToVote}
                            class="font-mono text-[10px] tracking-wider uppercase flex items-center gap-1 transition-colors disabled:opacity-50 {reply.userVote === 'down' ? 'text-printer-accent dark:text-printer-accent-dark' : 'text-printer-ink-light dark:text-printer-ink-dark/40 hover:text-printer-accent dark:hover:text-printer-accent-dark'}"
                          >
                            ▼ {reply.downvotes > 0 ? reply.downvotes : ""} {t.downvote}
                          </button>
                          <button
                            type="button"
                            onclick={() => (replyTo = { id: reply.id, name: reply.author.name })}
                            class="font-mono text-[10px] tracking-wider uppercase text-printer-accent dark:text-printer-accent-dark hover:underline"
                          >
                            {t.reply}
                          </button>
                          {#if canDelete(reply)}
                            <button
                              type="button"
                              onclick={() => handleDelete(reply)}
                              disabled={deleting.has(reply.id)}
                              class="font-mono text-[10px] tracking-wider uppercase text-red-500/70 hover:text-red-500 disabled:opacity-50"
                            >
                              {t.delete}
                            </button>
                          {/if}
                        </div>
                        {#if replyTo?.id === reply.id}
                          {@render ReplyInline()}
                        {/if}
                      </div>
                    </li>
                  {/each}
                </ul>
                <button
                  type="button"
                  onclick={() => toggleExpand(comment.id)}
                  class="mt-1 ml-3 font-mono text-[10px] tracking-wider text-printer-ink-light dark:text-printer-ink-dark/40 hover:text-printer-accent dark:hover:text-printer-accent-dark"
                >
                  {t.collapseReplies}
                </button>
              {:else}
                <button
                  type="button"
                  onclick={() => toggleExpand(comment.id)}
                  class="mt-1 ml-3 font-mono text-[10px] tracking-wider text-printer-accent dark:text-printer-accent-dark hover:underline"
                >
                  {t.expandReplies(comment.replies.length)}
                </button>
              {/if}
            {/if}
          </li>
        {/each}
      </ul>
    {/if}
  </div>
</section>

<!-- Login modal -->
<LoginModal {lang} show={showLoginModal} onclose={() => (showLoginModal = false)} {onauthed} />

<!-- Meta snippet for a single comment header -->
{#snippet CommentMeta({ comment, topLevel }: { comment: PublicComment; topLevel: boolean })}
  <div class="flex items-center gap-2">
    {@render Avatar({ name: comment.author.name, avatarUrl: comment.author.avatarUrl, size: "sm" })}
    {#if comment.author.website}
      <a
        href={comment.author.website}
        target="_blank"
        rel="nofollow noopener"
        class="text-[12px] font-medium text-printer-ink dark:text-printer-ink-dark hover:underline"
      >
        {comment.author.name}
      </a>
    {:else}
      <span class="text-[12px] font-medium text-printer-ink dark:text-printer-ink-dark">
        {comment.author.name}
      </span>
    {/if}
    {#if comment.author.type === "github"}
      <span
        class="rounded-sm border border-printer-accent/50 dark:border-printer-accent-dark/50 px-1 font-mono text-[9px] tracking-wider uppercase text-printer-accent dark:text-printer-accent-dark"
      >
        {t.developer}
      </span>
    {:else if comment.author.isOwner}
      <span
        class="rounded-sm border border-printer-accent/50 dark:border-printer-accent-dark/50 px-1 font-mono text-[9px] tracking-wider uppercase text-printer-accent dark:text-printer-accent-dark"
      >
        {t.owner}
      </span>
    {:else if comment.author.type === "account"}
      <span
        class="rounded-sm border border-printer-ink/30 dark:border-printer-ink-dark/30 px-1 font-mono text-[9px] tracking-wider uppercase text-printer-ink-light dark:text-printer-ink-dark/50"
      >
        {t.user}
      </span>
    {:else}
      <span
        class="rounded-sm border border-printer-ink/20 dark:border-printer-ink-dark/20 px-1 font-mono text-[9px] tracking-wider uppercase text-printer-ink-light/60 dark:text-printer-ink-dark/40"
      >
        {t.visitor}
      </span>
    {/if}
    <span
      class="ml-auto font-mono text-[10px] text-printer-ink-light/70 dark:text-printer-ink-dark/30"
      title={fullTime(comment.createdAt)}
    >
      {formatTime(comment.createdAt)}
    </span>
  </div>
{/snippet}

<!-- Avatar snippet -->
{#snippet Avatar({ name, avatarUrl, size = "sm" }: { name: string; avatarUrl?: string | null; size?: "sm" | "md" })}
  {#if avatarUrl}
    <img
      src={avatarUrl}
      alt=""
      class="{size === 'sm' ? 'w-6 h-6' : 'w-8 h-8'} rounded-full object-cover"
    />
  {:else}
    <span
      class="{size === 'sm' ? 'w-6 h-6 text-[11px]' : 'w-8 h-8 text-[14px]'} inline-flex items-center justify-center rounded-full bg-printer-accent dark:bg-printer-accent-dark text-printer-paper dark:text-printer-paper-dark font-mono"
    >
      {initialOf(name)}
    </span>
  {/if}
{/snippet}

<!-- Inline reply form: rendered under the comment being replied to -->
{#snippet ReplyInline()}
  <form class="mt-2" onsubmit={(e) => { e.preventDefault(); submit(); }}>
    <div
      class="rounded-sm border border-printer-accent/40 dark:border-printer-accent-dark/40 bg-printer-paper dark:bg-printer-paper-dark px-2 py-2"
    >
      <textarea
        bind:value={content}
        rows="2"
        maxlength="2000"
        placeholder={placeholder}
        class="w-full rounded-sm border border-printer-ink/20 dark:border-printer-ink-dark/20 bg-transparent px-2 py-1.5 text-[13px] leading-relaxed text-printer-ink dark:text-printer-ink-dark focus:outline-none focus:border-printer-accent dark:focus:border-printer-accent-dark resize-y"
      ></textarea>

      {#if formError}
        <p class="mt-1 font-mono text-[10px] tracking-wider text-red-600 dark:text-red-400">
          {formError}
        </p>
      {/if}

      <div class="mt-2 flex items-center justify-end gap-2">
        {#if !user}
          <span
            class="mr-auto font-mono text-[9px] tracking-wider text-printer-ink-light/60 dark:text-printer-ink-dark/30"
          >
            {t.anonymousHint}
          </span>
        {/if}
        <button
          type="button"
          onclick={() => (replyTo = null)}
          class="font-mono text-[10px] tracking-wider uppercase text-printer-ink-light dark:text-printer-ink-dark/40 hover:text-printer-ink dark:hover:text-printer-ink-dark"
        >
          {t.cancel}
        </button>
        <button
          type="submit"
          disabled={submitting}
          class="rounded-sm border border-printer-accent dark:border-printer-accent-dark bg-printer-accent dark:bg-printer-accent-dark px-3 py-1 font-mono text-[10px] tracking-wider uppercase text-printer-paper dark:text-printer-paper-dark hover:opacity-90 disabled:opacity-50 transition-opacity"
        >
          {submitting ? t.submitting : t.submit}
        </button>
      </div>
    </div>
  </form>
{/snippet}
