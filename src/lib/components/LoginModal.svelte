<script lang="ts">
  import type { Language } from "$lib/dictionaries";
  import { getDictionary } from "$lib/dictionaries";

  let { lang, show = false, onclose, onauthed }: {
    lang: Language;
    show: boolean;
    onclose: () => void;
    onauthed: (user: { provider: string; login: string; name: string; avatarUrl: string | null }, token: string) => void;
  } = $props();

  let dictionary = $derived(getDictionary(lang));
  let t = $derived(dictionary.comments);

  let mode = $state<"login" | "register">("login");
  let email = $state("");
  let username = $state("");
  let password = $state("");
  let code = $state("");
  let loading = $state(false);
  let error = $state<string | null>(null);
  let sendingCode = $state(false);
  let codeSentMsg = $state(false);
  let countdown = $state(0);
  let devCode = $state<string | null>(null);
  let timer: ReturnType<typeof setInterval> | undefined;

  function startCountdown(seconds: number) {
    countdown = seconds;
    if (timer) clearInterval(timer);
    timer = setInterval(() => {
      countdown -= 1;
      if (countdown <= 0 && timer) {
        clearInterval(timer);
        timer = undefined;
      }
    }, 1000);
  }

  $effect(() => {
    return () => {
      if (timer) clearInterval(timer);
    };
  });

  function reset() {
    mode = "login";
    email = "";
    username = "";
    password = "";
    code = "";
    error = null;
    codeSentMsg = false;
    devCode = null;
    countdown = 0;
  }

  $effect(() => {
    if (show) reset();
  });

  async function sendCode() {
    if (sendingCode || countdown > 0) return;
    if (!email.includes("@")) { error = t.authBadInput; return; }
    sendingCode = true;
    error = null;
    try {
      const res = await fetch("/api/auth/send-code", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email }),
      });
      if (res.status === 429) {
        const data = await res.json() as { waitMs?: number };
        startCountdown(Math.ceil((data.waitMs ?? 60_000) / 1000));
        return;
      }
      if (!res.ok) { error = t.authBadInput; return; }
      const data = await res.json() as { devCode?: string };
      codeSentMsg = true;
      devCode = data.devCode ?? null;
      startCountdown(60);
    } catch {
      error = t.serverError;
    } finally {
      sendingCode = false;
    }
  }

  async function handleSubmit(e: Event) {
    e.preventDefault();
    if (loading) return;
    error = null;
    loading = true;

    try {
      if (mode === "register") {
        const res = await fetch("/api/auth/register", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, username, password, code }),
        });
        if (res.status === 429) { error = t.rateLimited; return; }
        if (res.status === 409) {
          const data = await res.json() as { error: string };
          if (data.error === "bad_code") error = t.badCode;
          else if (data.error === "email_taken" || data.error === "username_taken") error = t.accountExists;
          else error = t.serverError;
          return;
        }
        if (res.status === 400) {
          const data = await res.json() as { message?: string };
          error =
            data.message === "bad_email" ? t.invalidEmail :
            data.message === "bad_username" ? t.invalidUsername :
            data.message === "bad_password" ? t.invalidPassword :
            data.message === "code_required" ? t.codeRequired :
            t.authBadInput;
          return;
        }
        if (!res.ok) { error = t.serverError; return; }
        // After register, auto-login
      }

      // Login
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      if (res.status === 429) { error = t.rateLimited; return; }
      if (res.status === 401) {
        error = t.authBadInput;
        return;
      }
      if (!res.ok) { error = t.serverError; return; }

      const data = await res.json() as {
        token: string;
        user: { provider: string; login: string; name: string; avatarUrl: string | null };
      };

      localStorage.setItem("hhy.comments.token", data.token);
      localStorage.setItem("hhy.comments.user", JSON.stringify(data.user));
      onauthed(data.user, data.token);
      onclose();
    } catch {
      error = t.serverError;
    } finally {
      loading = false;
    }
  }

  function githubLogin() {
    const next = encodeURIComponent(window.location.pathname + window.location.search);
    window.location.href = `/auth/github/login?next=${next}`;
  }
</script>

{#if show}
  <div
    class="fixed inset-0 z-50 flex items-center justify-center"
    role="dialog"
    aria-modal="true"
  >
    <!-- Backdrop -->
    <button
      type="button"
      class="absolute inset-0 bg-printer-ink/40 dark:bg-black/60"
      onclick={onclose}
      aria-label="Close"
    ></button>

    <!-- Modal card -->
    <div
      class="relative w-full max-w-sm mx-4 rounded-sm border border-printer-ink/20 dark:border-printer-ink-dark/20 bg-printer-paper dark:bg-printer-paper-dark px-5 py-5 shadow-lg"
    >
      <h2
        class="font-mono text-[11px] tracking-[0.25em] uppercase text-printer-ink dark:text-printer-ink-dark mb-4"
      >
        ✎ {mode === "login" ? t.signIn : t.signUp}
      </h2>

      <form onsubmit={handleSubmit}>
        <label class="block mb-3">
          <span
            class="block font-mono text-[10px] tracking-wider text-printer-ink-light dark:text-printer-ink-dark/40 mb-1"
          >
            {t.email}
          </span>
          <input
            type="email"
            bind:value={email}
            required
            autocomplete="email"
            class="w-full rounded-sm border border-printer-ink/20 dark:border-printer-ink-dark/20 bg-transparent px-2 py-1.5 text-[13px] text-printer-ink dark:text-printer-ink-dark focus:outline-none focus:border-printer-accent dark:focus:border-printer-accent-dark"
          />
        </label>

        {#if mode === "register"}
          <label class="block mb-3">
            <span
              class="block font-mono text-[10px] tracking-wider text-printer-ink-light dark:text-printer-ink-dark/40 mb-1"
            >
              {t.username}
            </span>
            <input
              type="text"
              bind:value={username}
              required
              minlength="2"
              maxlength="24"
              autocomplete="username"
              class="w-full rounded-sm border border-printer-ink/20 dark:border-printer-ink-dark/20 bg-transparent px-2 py-1.5 text-[13px] text-printer-ink dark:text-printer-ink-dark focus:outline-none focus:border-printer-accent dark:focus:border-printer-accent-dark"
            />
          </label>

          <label class="block mb-3">
            <span
              class="block font-mono text-[10px] tracking-wider text-printer-ink-light dark:text-printer-ink-dark/40 mb-1"
            >
              {t.verificationCode}
            </span>
            <div class="flex gap-2">
              <input
                type="text"
                bind:value={code}
                required
                maxlength="6"
                inputmode="numeric"
                autocomplete="one-time-code"
                class="flex-1 rounded-sm border border-printer-ink/20 dark:border-printer-ink-dark/20 bg-transparent px-2 py-1.5 text-[13px] font-mono text-printer-ink dark:text-printer-ink-dark focus:outline-none focus:border-printer-accent dark:focus:border-printer-accent-dark"
              />
              <button
                type="button"
                onclick={sendCode}
                disabled={sendingCode || countdown > 0}
                class="shrink-0 rounded-sm border border-printer-accent dark:border-printer-accent-dark px-2 py-1 font-mono text-[10px] tracking-wider text-printer-accent dark:text-printer-accent-dark hover:bg-printer-accent/10 disabled:opacity-50"
              >
                {sendingCode ? "…" : countdown > 0 ? t.resendIn(countdown) : t.sendCode}
              </button>
            </div>
            {#if codeSentMsg}
              <span class="mt-1 block font-mono text-[9px] tracking-wider text-printer-accent">
                {t.codeSent}
              </span>
            {/if}
            {#if devCode}
              <span class="mt-1 block font-mono text-[9px] tracking-wider text-printer-ink-light/60 dark:text-printer-ink-dark/40">
                {t.verificationCode}: {devCode}
              </span>
            {/if}
          </label>
        {/if}

        <label class="block mb-3">
          <span
            class="block font-mono text-[10px] tracking-wider text-printer-ink-light dark:text-printer-ink-dark/40 mb-1"
          >
            {t.password}
          </span>
          <input
            type="password"
            bind:value={password}
            required
            minlength="6"
            maxlength="128"
            autocomplete={mode === "login" ? "current-password" : "new-password"}
            class="w-full rounded-sm border border-printer-ink/20 dark:border-printer-ink-dark/20 bg-transparent px-2 py-1.5 text-[13px] text-printer-ink dark:text-printer-ink-dark focus:outline-none focus:border-printer-accent dark:focus:border-printer-accent-dark"
          />
        </label>

        {#if error}
          <p
            class="mb-3 font-mono text-[10px] tracking-wider text-red-600 dark:text-red-400"
          >
            {error}
          </p>
        {/if}

        <button
          type="submit"
          disabled={loading}
          class="w-full rounded-sm border border-printer-accent dark:border-printer-accent-dark bg-printer-accent dark:bg-printer-accent-dark px-4 py-1.5 font-mono text-[11px] tracking-wider uppercase text-printer-paper dark:text-printer-paper-dark hover:opacity-90 disabled:opacity-50 transition-opacity"
        >
          {loading ? "…" : mode === "login" ? t.signIn : t.signUp}
        </button>
      </form>

      <div class="mt-3 text-center">
        <button
          type="button"
          onclick={() => { mode = mode === "login" ? "register" : "login"; error = null; }}
          class="font-mono text-[10px] tracking-wider text-printer-accent dark:text-printer-accent-dark hover:underline"
        >
          {mode === "login" ? t.noAccount : t.haveAccount}
        </button>
      </div>

      <div class="mt-3 flex items-center gap-2">
        <div
          class="flex-1 border-t border-printer-ink/10 dark:border-printer-ink-dark/10"
        ></div>
        <span
          class="font-mono text-[9px] tracking-wider text-printer-ink-light/50 dark:text-printer-ink-dark/30 uppercase"
        >
          {t.orContinue}
        </span>
        <div
          class="flex-1 border-t border-printer-ink/10 dark:border-printer-ink-dark/10"
        ></div>
      </div>

      <button
        type="button"
        onclick={githubLogin}
        class="mt-3 w-full inline-flex items-center justify-center gap-2 rounded-sm border border-printer-ink/15 dark:border-printer-ink-dark/15 bg-printer-paper dark:bg-printer-paper-dark px-3 py-1.5 font-mono text-[11px] tracking-wider uppercase text-printer-ink dark:text-printer-ink-dark hover:border-printer-accent dark:hover:border-printer-accent-dark transition-colors"
      >
        <svg viewBox="0 0 16 16" class="w-4 h-4 fill-current" aria-hidden="true">
          <path
            d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"
          />
        </svg>
        {t.signInGithub}
      </button>

      <div class="mt-3 text-center">
        <button
          type="button"
          onclick={onclose}
          class="font-mono text-[10px] tracking-wider text-printer-ink-light dark:text-printer-ink-dark/40 hover:text-printer-ink dark:hover:text-printer-ink-dark"
        >
          {t.continueAnonymous}
        </button>
      </div>
    </div>
  </div>
{/if}
