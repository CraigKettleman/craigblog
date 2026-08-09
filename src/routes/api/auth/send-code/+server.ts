import { error, json, type RequestHandler } from "@sveltejs/kit";
import { EMAIL_RE, issueCode } from "$lib/server/accounts";
import { smtpConfigured, sendEmail, verificationMail } from "$lib/server/mailer";
import { clientIp, consumeAuthSlot, hashIp } from "$lib/server/rate-limit";

export const prerender = false;

/** Generate a 6-digit verification code and email it. 60s resend cooldown. */
export const POST: RequestHandler = async ({ request }) => {
  const ip = clientIp(request.headers.get("x-forwarded-for"));
  if (!consumeAuthSlot(hashIp(ip))) throw error(429, "rate limited");

  const body = (await request.json().catch(() => null)) as Record<string, unknown> | null;
  if (!body || typeof body.email !== "string") throw error(400, "bad json");

  const email = body.email.trim();
  if (!EMAIL_RE.test(email)) throw error(400, "bad email");

  const issued = issueCode(email);
  if (!issued.ok) {
    return json({ error: "too_frequent", waitMs: issued.waitMs }, { status: 429 });
  }
  const code = issued.code!;

  if (smtpConfigured()) {
    try {
      await sendEmail(email, verificationMail(code).subject, verificationMail(code).text);
      return json({ ok: true });
    } catch {
      // Email failed to send; still keep the code so a retry can reuse it? No —
      // a fresh code is safer. Return an error so the user can retry.
      return json({ error: "send_failed" }, { status: 502 });
    }
  }

  // No SMTP configured yet — dev fallback: return the code so the flow still
  // works locally. The frontend shows it in a "test mode" hint.
  return json({ ok: true, devCode: code });
};
