import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { dev } from "$app/environment";
import { env } from "$env/dynamic/private";
import { hashIp } from "./rate-limit";
import type { DailyPoint, Summary } from "$lib/analytics-types";

/**
 * 自建访问统计（真实数据管线）。
 *
 * 设计原则：
 * - 只记匿名事件：时间、路径、语言、设备类、来源类、国家码（ISO-2）、ipHash 前 16 位。
 *   不落原始 IP、不落 UA 原文、不落任何可识别个人的字段。
 * - 存储沿用评论库的 JSON 文件模式（ANALYTICS_FILE，默认服务器 data/ 目录），
 *   滚动保留 120 天、上限 10 万条，写入走串行队列 + 临时文件原子替换。
 * - 汇总在读时聚合，dashboard 按需取 7/30/90 天窗口。
 */

export type DeviceKind = "desktop" | "mobile" | "tablet" | "bot";
export type RefKind = "direct" | "search" | "social" | "internal" | "referral";

export interface ViewEvent {
  /** epoch ms */
  t: number;
  /** 站内路径 */
  p: string;
  l: "en" | "zh";
  d: DeviceKind;
  r: RefKind;
  /** ISO-2 国家码，空串表示未知 */
  c: string;
  /** ipHash 前 16 位，仅用于 UV 去重 */
  h: string;
}

interface AnalyticsDb {
  v: 1;
  events: ViewEvent[];
}

const RETENTION_MS = 120 * 24 * 3600 * 1000;
const MAX_EVENTS = 100_000;
/** 同一访客同一页面 30 分钟内只计一次 PV，防刷新灌水 */
const DEDUPE_MS = 30 * 60 * 1000;

function dataFile(): string {
  return (
    env.ANALYTICS_FILE ??
    (dev
      ? `${process.cwd()}/data/analytics.json`
      : "/var/www/hhy.homes/data/analytics.json")
  );
}

let cache: { file: string; db: AnalyticsDb } | null = null;

async function load(): Promise<AnalyticsDb> {
  const file = dataFile();
  if (cache && cache.file === file) return cache.db;
  try {
    const raw = await readFile(file, "utf8");
    const parsed = JSON.parse(raw) as AnalyticsDb;
    cache = { file, db: { v: 1, events: Array.isArray(parsed.events) ? parsed.events : [] } };
  } catch {
    cache = { file, db: { v: 1, events: [] } };
  }
  return cache.db;
}

let queue: Promise<void> = Promise.resolve();
function enqueue<T>(fn: () => Promise<T>): Promise<T> {
  const run = queue.then(fn, fn);
  queue = run.then(
    () => undefined,
    () => undefined,
  );
  return run;
}

async function save(): Promise<void> {
  const db = await load();
  const file = dataFile();
  await mkdir(dirname(file), { recursive: true });
  const tmp = `${file}.tmp`;
  await writeFile(tmp, JSON.stringify(db), "utf8");
  await rename(tmp, file);
}

export function classifyDevice(ua: string): DeviceKind {
  const s = ua.toLowerCase();
  if (/bot|crawl|spider|slurp|headless|curl|wget|python-requests|go-http/.test(s)) return "bot";
  if (/ipad|tablet|playbook|kindle/.test(s)) return "tablet";
  if (/mobi|android|iphone|ipod|windows phone/.test(s)) return "mobile";
  return "desktop";
}

export function classifyReferrer(referrer: string, origin: string): RefKind {
  if (!referrer) return "direct";
  let host = "";
  try {
    host = new URL(referrer).hostname.toLowerCase();
  } catch {
    return "direct";
  }
  if (!host) return "direct";
  let originHost = "";
  try {
    originHost = new URL(origin).hostname.toLowerCase();
  } catch {
    originHost = "";
  }
  if (originHost && (host === originHost || host.endsWith(`.${originHost}`))) return "internal";
  if (/(^|\.)google\.|^bing\.|^duckduckgo\.|^baidu\.|^sogou\.|^so\.com|^yandex\./.test(host)) return "search";
  if (/(^|\.)twitter\.|^x\.com|(^|\.)facebook\.|(^|\.)reddit\.|news\.ycombinator\.|(^|\.)weibo\.|(^|\.)zhihu\.|(^|\.)v2ex\.|(^|\.)t\.co/.test(host)) return "social";
  return "referral";
}

/** 记录一次真实访问；机器人直接丢弃（只统计真人）。返回是否入库。 */
export async function recordView(input: {
  path: string;
  lang: "en" | "zh";
  device: DeviceKind;
  ref: RefKind;
  country: string;
  ip: string;
}): Promise<boolean> {
  if (input.device === "bot") return false;
  const h = hashIp(input.ip).slice(0, 16);
  return enqueue(async () => {
    const db = await load();
    const now = Date.now();
    const dup = db.events.some(
      (e) => e.h === h && e.p === input.path && now - e.t < DEDUPE_MS,
    );
    if (dup) return false;
    db.events.push({
      t: now,
      p: input.path.slice(0, 200),
      l: input.lang,
      d: input.device,
      r: input.ref,
      c: /^[A-Z]{2}$/.test(input.country) ? input.country : "",
      h,
    });
    const cutoff = now - RETENTION_MS;
    if (db.events[0] && db.events[0].t < cutoff) {
      db.events = db.events.filter((e) => e.t >= cutoff);
    }
    if (db.events.length > MAX_EVENTS) {
      db.events = db.events.slice(-MAX_EVENTS);
    }
    await save();
    return true;
  });
}

function dayKey(t: number): string {
  const d = new Date(t);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

/** 聚合最近 days 天的真实事件 */
export async function summarize(days: number): Promise<Summary> {
  const db = await load();
  const now = Date.now();
  const from = now - days * 24 * 3600 * 1000;
  const events = db.events.filter((e) => e.t >= from);

  const dailyMap = new Map<string, { pv: number; uv: Set<string> }>();
  const hours = new Array<number>(24).fill(0);
  const countryMap = new Map<string, { pv: number; uv: Set<string> }>();
  const deviceMap = new Map<DeviceKind, number>();
  const refMap = new Map<RefKind, number>();
  const langMap = new Map<"en" | "zh", number>();
  const pageMap = new Map<string, { pv: number; uv: Set<string> }>();
  const uvAll = new Set<string>();
  const visitorDays = new Map<string, Set<string>>();

  for (const e of events) {
    const dk = dayKey(e.t);
    const day = dailyMap.get(dk) ?? { pv: 0, uv: new Set<string>() };
    day.pv += 1;
    day.uv.add(e.h);
    dailyMap.set(dk, day);

    const vd = visitorDays.get(e.h) ?? new Set<string>();
    vd.add(dk);
    visitorDays.set(e.h, vd);

    hours[new Date(e.t).getHours()] += 1;

    if (e.c) {
      const c = countryMap.get(e.c) ?? { pv: 0, uv: new Set<string>() };
      c.pv += 1;
      c.uv.add(e.h);
      countryMap.set(e.c, c);
    }
    deviceMap.set(e.d, (deviceMap.get(e.d) ?? 0) + 1);
    refMap.set(e.r, (refMap.get(e.r) ?? 0) + 1);
    langMap.set(e.l, (langMap.get(e.l) ?? 0) + 1);
    const pg = pageMap.get(e.p) ?? { pv: 0, uv: new Set<string>() };
    pg.pv += 1;
    pg.uv.add(e.h);
    pageMap.set(e.p, pg);
    uvAll.add(e.h);
  }

  const daily: DailyPoint[] = [];
  const end = new Date();
  end.setHours(0, 0, 0, 0);
  for (let i = days - 1; i >= 0; i--) {
    const d = new Date(end);
    d.setDate(d.getDate() - i);
    const key = dayKey(d.getTime());
    const v = dailyMap.get(key);
    daily.push({
      date: key,
      label: `${d.getMonth() + 1}-${d.getDate()}`,
      pv: v?.pv ?? 0,
      uv: v?.uv.size ?? 0,
    });
  }

  const toCounts = <T,>(m: Map<T, number>) =>
    [...m.entries()].map(([label, value]) => ({ label, value })).sort((a, b) => b.value - a.value);

  // 回访客：窗口期内出现在 ≥2 个不同日
  let returning = 0;
  for (const days of visitorDays.values()) {
    if (days.size >= 2) returning += 1;
  }

  // 存留期内累计（不受窗口限制）
  const allTimeUv = new Set<string>();
  for (const e of db.events) allTimeUv.add(e.h);

  return {
    generatedAt: new Date().toISOString(),
    days,
    totals: {
      pv: events.length,
      uv: uvAll.size,
      countries: countryMap.size,
      pages: pageMap.size,
    },
    allTime: { pv: db.events.length, uv: allTimeUv.size },
    visitors: { new: uvAll.size - returning, returning },
    daily,
    hours,
    countries: [...countryMap.entries()]
      .map(([code, v]) => ({ code, pv: v.pv, uv: v.uv.size }))
      .sort((a, b) => b.pv - a.pv),
    devices: toCounts(deviceMap),
    refs: toCounts(refMap),
    langs: toCounts(langMap),
    pages: [...pageMap.entries()]
      .map(([path, v]) => ({ path, pv: v.pv, uv: v.uv.size }))
      .sort((a, b) => b.pv - a.pv),
  };
}
