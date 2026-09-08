import { error, json } from "@sveltejs/kit";
import { env } from "$env/dynamic/private";
import type { RequestHandler } from "./$types";
import { summarize } from "$lib/server/analytics";
import type { Summary } from "$lib/analytics-types";

export const prerender = false;

/**
 * studio 看板的数据源代理（仅 LOCAL_ADMIN dev 可达）：
 * - source=local：读本机 data/analytics.json（本地浏览产生的真实事件）
 * - source=prod ：只读代理线上 https://hhy.homes/api/analytics（沿用 COMMENTS_ADMIN_KEY），
 *                 与评论后台「本地代理线上」同一模式；失败时如实返回错误码，不伪造数据。
 */
export const GET: RequestHandler = async ({ url }) => {
  if (env.LOCAL_ADMIN !== "1") throw error(404);
  const source = url.searchParams.get("source") === "prod" ? "prod" : "local";
  const daysRaw = Number(url.searchParams.get("days") ?? 30);
  const days = [7, 30, 90].includes(daysRaw) ? daysRaw : 30;

  if (source === "local") {
    const summary = await summarize(days);
    return json({ ok: true, source, summary });
  }

  const key = env.COMMENTS_ADMIN_KEY;
  if (!key) {
    return json({ ok: false, source, error: "NO_KEY" });
  }
  // 线上地址可用 env 覆盖（本地联调用，默认生产域名）
  const origin = env.ANALYTICS_ORIGIN ?? "https://hhy.homes";
  try {
    const res = await fetch(`${origin}/api/analytics?days=${days}`, {
      headers: { Authorization: `Bearer ${key}` },
      signal: AbortSignal.timeout(8000),
    });
    if (!res.ok) return json({ ok: false, source, error: `HTTP_${res.status}` });
    const summary = (await res.json()) as Summary;
    return json({ ok: true, source, summary });
  } catch {
    return json({ ok: false, source, error: "NETWORK" });
  }
};
