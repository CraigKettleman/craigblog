import { error, json, type RequestHandler } from "@sveltejs/kit";
import { loginAccount } from "$lib/server/accounts";
import { clientIp, consumeAuthSlot, hashIp } from "$lib/server/rate-limit";
import { sign } from "$lib/server/session";

export const prerender = false;

export const POST: RequestHandler = async ({ request }) => {
  const ip = clientIp(request.headers.get("x-forwarded-for"));
  if (!consumeAuthSlot(hashIp(ip))) {
    throw error(429, "rate limited");
  }

  const body = (await request.json().catch(() => null)) as Record<string, unknown> | null;
  if (!body || typeof body !== "object") throw error(400, "bad json");

  const email = typeof body.email === "string" ? body.email.trim() : "";
  const password = typeof body.password === "string" ? body.password : "";

  if (!email || !password) throw error(400, "bad request");

  const user = await loginAccount({ email, password });
  if (!user) {
    return json({ error: "bad_credentials" }, { status: 401 });
  }

  const token = sign({
    provider: "account",
    login: user.username,
    name: user.username,
    avatarUrl: null,
  });

  return json({
    token,
    user: {
      provider: "account",
      login: user.username,
      name: user.username,
      avatarUrl: null,
    },
  });
};
