import { error, json, type RequestHandler } from "@sveltejs/kit";
import {
  EMAIL_RE,
  MAX_EMAIL,
  MAX_PASSWORD,
  MAX_USERNAME,
  MIN_PASSWORD,
  registerAccount,
  USERNAME_RE,
} from "$lib/server/accounts";
import { clientIp, consumeAuthSlot, hashIp } from "$lib/server/rate-limit";

export const prerender = false;

export const POST: RequestHandler = async ({ request }) => {
  const ip = clientIp(request.headers.get("x-forwarded-for"));
  if (!consumeAuthSlot(hashIp(ip))) {
    throw error(429, "too many attempts");
  }

  const body = (await request.json().catch(() => null)) as Record<string, unknown> | null;
  if (!body || typeof body !== "object") throw error(400, "bad_request");

  const email = typeof body.email === "string" ? body.email.trim() : "";
  const username = typeof body.username === "string" ? body.username.trim() : "";
  const password = typeof body.password === "string" ? body.password : "";
  const code = typeof body.code === "string" ? body.code.trim() : "";

  if (!EMAIL_RE.test(email) || email.length > MAX_EMAIL) throw error(400, "bad_email");
  if (!USERNAME_RE.test(username)) throw error(400, "bad_username");
  if (password.length < MIN_PASSWORD || password.length > MAX_PASSWORD) {
    throw error(400, "bad_password");
  }
  if (!code) throw error(400, "code_required");

  const result = await registerAccount({ email, username, password, code });
  if (!result.ok) {
    return json({ error: result.reason }, { status: 409 });
  }

  return json({ ok: true, user: result.user }, { status: 201 });
};
