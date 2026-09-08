import { error, json } from "@sveltejs/kit";
import { env } from "$env/dynamic/private";
import type { RequestHandler } from "./$types";
import {
  classifyDevice,
  classifyReferrer,
  recordView,
  summarize,
} from "$lib/server/analytics";
import { clientIp } from "$lib/server/rate-limit";
import { isAdminKey } from "$lib/server/session";

export const prerender = false;

/**
 * 访问统计埋点。
 * POST：匿名信标（sendBeacon / fetch keepalive），只收路径/语言/来源类/国家码，
 *       服务端自行判定设备类与 ipHash，不接收也不存储原始 IP。
 * GET ：管理密钥（COMMENTS_ADMIN_KEY）鉴权，返回聚合摘要，供本地 studio 只读代理线上数据。
 */
export const POST: RequestHandler = async ({ request, getClientAddress }) => {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    throw error(400);
  }
  const b = (body ?? {}) as Record<string, unknown>;
  const p = typeof b.p === "string" && b.p.startsWith("/") ? b.p : null;
  if (!p) throw error(400);
  const l = b.l === "zh" ? "zh" : "en";
  const rRaw = typeof b.r === "string" ? b.r : "";
  const r = (["direct", "search", "social", "internal", "referral"] as const).includes(
    rRaw as never,
  )
    ? (rRaw as "direct")
    : "direct";
  const c = typeof b.c === "string" ? b.c.toUpperCase().slice(0, 2) : "";

  const ua = request.headers.get("user-agent") ?? "";
  const origin = env.SITE_ORIGIN ?? "https://hhy.homes";
  void recordView({
    path: p,
    lang: l,
    device: classifyDevice(ua),
    ref: r,
    country: c,
    ip: clientIp(request.headers.get("x-forwarded-for"), getClientAddress()),
  }).catch(() => {
    /* 统计写入失败不影响访客体验，也不伪装成功：事件丢弃即可 */
  });
  return new Response(null, { status: 204 });
};

export const GET: RequestHandler = async ({ url, request }) => {
  const auth = request.headers.get("authorization");
  if (!isAdminKey(auth)) throw error(401);
  const daysRaw = Number(url.searchParams.get("days") ?? 30);
  const days = [7, 30, 90].includes(daysRaw) ? daysRaw : 30;
  return json(await summarize(days));
};
