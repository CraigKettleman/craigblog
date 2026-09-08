import { browser } from "$app/environment";

/**
 * 匿名访问信标（真实数据来源）。
 *
 * 隐私约定：
 * - 不发送原始 IP；服务端只存 ipHash 前 16 位用于 UV 去重。
 * - 国家码由浏览器向公共地理接口自解析（7 天本地缓存），解析失败记为空。
 * - 尊重 Do-Not-Track；/studio、/admin、/api 等后台路径不计入。
 */

const GEO_CACHE_KEY = "hhy.geo";
const GEO_TTL_MS = 7 * 24 * 3600 * 1000;

const lastSent = new Map<string, number>();

function classifyReferrer(): string {
  const ref = document.referrer;
  if (!ref) return "direct";
  let host = "";
  try {
    host = new URL(ref).hostname.toLowerCase();
  } catch {
    return "direct";
  }
  if (host === location.hostname || host.endsWith(`.${location.hostname}`)) return "internal";
  if (/(^|\.)google\.|^bing\.|^duckduckgo\.|^baidu\.|^sogou\.|^so\.com|^yandex\./.test(host)) return "search";
  if (/(^|\.)twitter\.|^x\.com|(^|\.)facebook\.|(^|\.)reddit\.|news\.ycombinator\.|(^|\.)weibo\.|(^|\.)zhihu\.|(^|\.)v2ex\.|(^|\.)t\.co/.test(host)) return "social";
  return "referral";
}

async function resolveCountry(): Promise<string> {
  try {
    const raw = localStorage.getItem(GEO_CACHE_KEY);
    if (raw) {
      const cached = JSON.parse(raw) as { c?: string; exp?: number };
      if (cached && typeof cached.c === "string" && (cached.exp ?? 0) > Date.now()) {
        return cached.c;
      }
    }
  } catch {
    /* 缓存不可读时直接重新解析 */
  }
  let code = "";
  try {
    const res = await fetch("https://api.country.is/", { signal: AbortSignal.timeout(4000) });
    if (res.ok) {
      const j = (await res.json()) as { country?: string };
      code = typeof j.country === "string" ? j.country.toUpperCase() : "";
    }
  } catch {
    /* 主接口失败走备用 */
  }
  if (!code) {
    try {
      const res = await fetch("https://ipapi.co/json/", { signal: AbortSignal.timeout(4000) });
      if (res.ok) {
        const j = (await res.json()) as { country_code?: string };
        code = typeof j.country_code === "string" ? j.country_code.toUpperCase() : "";
      }
    } catch {
      code = "";
    }
  }
  try {
    localStorage.setItem(GEO_CACHE_KEY, JSON.stringify({ c: code, exp: Date.now() + GEO_TTL_MS }));
  } catch {
    /* 存储不可用时本次仍可用 */
  }
  return code;
}

/** 页面浏览信标：首次加载与 SPA 跳转各计一次（同一路径 5 秒内去重） */
export function trackView(pathname: string, lang: string): void {
  if (!browser) return;
  if (navigator.doNotTrack === "1") return;
  if (/^\/(studio|admin|api)(\/|$)/.test(pathname)) return;
  const prev = lastSent.get(pathname) ?? 0;
  if (Date.now() - prev < 5000) return;
  lastSent.set(pathname, Date.now());

  const payload = { p: pathname, l: lang === "zh" ? "zh" : "en", r: classifyReferrer(), c: "" };

  void (async () => {
    payload.c = await resolveCountry();
    const body = JSON.stringify(payload);
    try {
      if (navigator.sendBeacon) {
        navigator.sendBeacon("/api/analytics", new Blob([body], { type: "application/json" }));
      } else {
        await fetch("/api/analytics", { method: "POST", body, keepalive: true });
      }
    } catch {
      /* 信标失败静默：统计是旁路功能，不能影响页面 */
    }
  })();
}
