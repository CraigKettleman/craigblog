import { error, type RequestHandler } from "@sveltejs/kit";
import { env } from "$env/dynamic/private";
import { sign } from "$lib/server/session";

export const prerender = false;

interface GitHubUser {
  login?: string;
  name?: string | null;
  avatar_url?: string | null;
}

export const GET: RequestHandler = async ({ url, cookies }) => {
  const clientId = env.GITHUB_CLIENT_ID;
  const clientSecret = env.GITHUB_CLIENT_SECRET;
  if (!clientId || !clientSecret) throw error(503, "GitHub login not configured");

  const code = url.searchParams.get("code");
  const state = url.searchParams.get("state");
  const expected = cookies.get("gh_state");
  const next = cookies.get("gh_next") ?? "/";
  cookies.delete("gh_state", { path: "/" });
  cookies.delete("gh_next", { path: "/" });
  if (!code || !state || !expected || state !== expected) {
    throw error(400, "state mismatch");
  }

  const tokenRes = await fetch("https://github.com/login/oauth/access_token", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({
      client_id: clientId,
      client_secret: clientSecret,
      code,
    }),
  });
  const tokenData = (await tokenRes.json()) as { access_token?: string };
  if (!tokenData.access_token) throw error(502, "token exchange failed");

  const userRes = await fetch("https://api.github.com/user", {
    headers: {
      Authorization: `Bearer ${tokenData.access_token}`,
      Accept: "application/vnd.github+json",
      "User-Agent": "hhy-homes",
    },
  });
  if (!userRes.ok) throw error(502, "failed to load GitHub user");
  const user = (await userRes.json()) as GitHubUser;
  if (!user.login) throw error(502, "failed to load GitHub user");

  const token = sign({
    provider: "github",
    login: user.login,
    name: user.name || user.login,
    avatarUrl: user.avatar_url ?? null,
  });

  // Hand the session back to the browser (the site is fully client-rendered),
  // then bounce to where the reader came from.
  const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Signing in…</title></head>
<body><script>
try {
  localStorage.setItem("hhy.comments.token", ${JSON.stringify(token)});
  localStorage.setItem("hhy.comments.user", ${JSON.stringify(
    JSON.stringify({ provider: "github", login: user.login, name: user.name || user.login, avatarUrl: user.avatar_url ?? null }),
  )});
} catch (e) {}
location.replace(${JSON.stringify(next)});
</script></body></html>`;

  return new Response(html, {
    headers: {
      "Content-Type": "text/html; charset=utf-8",
      "Cache-Control": "no-store",
    },
  });
};
