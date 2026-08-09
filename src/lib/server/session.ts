import { createHmac, timingSafeEqual } from "node:crypto";
import { env } from "$env/dynamic/private";

export interface SessionUser {
  provider: "github" | "account";
  login: string;
  name: string;
  avatarUrl: string | null;
}

interface SessionPayload extends SessionUser {
  exp: number;
}

const LIFETIME_MS = 30 * 24 * 60 * 60 * 1000;

function secret(): string {
  return env.SESSION_SECRET ?? "";
}

function b64url(text: string): string {
  return Buffer.from(text, "utf8").toString("base64url");
}

function signPayload(payload: string): string {
  return createHmac("sha256", secret()).update(payload).digest("base64url");
}

/** Sign a short-lived, tamper-evident session token. */
export function sign(user: SessionUser): string {
  const payload = b64url(
    JSON.stringify({ ...user, exp: Date.now() + LIFETIME_MS }),
  );
  return `${payload}.${signPayload(payload)}`;
}

/** Verify a session token; returns the user or null when invalid/expired. */
export function verify(raw: string | null | undefined): SessionUser | null {
  if (!raw) return null;
  const token = raw.replace(/^Bearer\s+/i, "");
  if (!token) return null;
  if (!secret()) return null;
  const [payload, sig] = token.split(".");
  if (!payload || !sig) return null;
  const expected = Buffer.from(signPayload(payload));
  const got = Buffer.from(sig);
  if (got.length !== expected.length || !timingSafeEqual(got, expected)) {
    return null;
  }
  try {
    const data = JSON.parse(Buffer.from(payload, "base64url").toString("utf8")) as SessionPayload;
    if (typeof data.exp !== "number" || data.exp < Date.now()) return null;
    if (typeof data.login !== "string" || !data.login) return null;
    if (data.provider !== "github" && data.provider !== "account") return null;
    return {
      provider: data.provider,
      login: data.login,
      name: typeof data.name === "string" && data.name ? data.name : data.login,
      avatarUrl: typeof data.avatarUrl === "string" ? data.avatarUrl : null,
    };
  } catch {
    return null;
  }
}

/** Constant-time check of a `Bearer <key>` admin credential. */
export function isAdminKey(authHeader: string | null | undefined): boolean {
  if (!authHeader) return false;
  const token = authHeader.replace(/^Bearer\s+/i, "");
  const expected = env.COMMENTS_ADMIN_KEY ?? "";
  if (!token || !expected) return false;
  const a = Buffer.from(token);
  const b = Buffer.from(expected);
  return a.length === b.length && timingSafeEqual(a, b);
}
