import { json, type RequestHandler } from "@sveltejs/kit";
import { env } from "$env/dynamic/private";

export const prerender = false;

export const GET: RequestHandler = () => {
  const githubEnabled = Boolean(env.GITHUB_CLIENT_ID && env.GITHUB_CLIENT_SECRET);
  return json({ githubEnabled, accountEnabled: true });
};
