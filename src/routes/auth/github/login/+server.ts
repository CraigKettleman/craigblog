import { randomBytes } from "node:crypto";
import { error, redirect, type RequestHandler } from "@sveltejs/kit";
import { env } from "$env/dynamic/private";

export const prerender = false;

export const GET: RequestHandler = ({ url, cookies }) => {
  const clientId = env.GITHUB_CLIENT_ID;
  if (!clientId) throw error(503, "GitHub login not configured");

  // Only allow same-site return paths.
  const next = url.searchParams.get("next") ?? "/";
  const safeNext = next.startsWith("/") && !next.startsWith("//") ? next : "/";

  const nonce = randomBytes(16).toString("hex");
  const cookieOpts = {
    path: "/",
    httpOnly: true,
    sameSite: "lax" as const,
    secure: url.protocol === "https:",
    maxAge: 600,
  };
  cookies.set("gh_state", nonce, cookieOpts);
  cookies.set("gh_next", safeNext, cookieOpts);

  const authorize = new URL("https://github.com/login/oauth/authorize");
  authorize.searchParams.set("client_id", clientId);
  authorize.searchParams.set("redirect_uri", `${env.SITE_ORIGIN || url.origin}/auth/github/callback`);
  authorize.searchParams.set("scope", "read:user");
  authorize.searchParams.set("state", nonce);

  throw redirect(302, authorize.toString());
};
