/**
 * studio 数据看板的纯函数工具（无 Node 依赖，前后端可共用）。
 *
 * 访问统计已接入真实管线（埋点 → JSON 落盘 → /studio/api/analytics 线上代理），
 * 这里只保留发布热力聚合与导出工具。
 */
import type { DailyPoint } from "$lib/analytics-types";

/** 发布热力单元格 */
export interface HeatCell {
  date: string;
  count: number;
  level: 0 | 1 | 2 | 3 | 4;
}

/** 数字本地化 */
export function formatNum(n: number, lang: "zh" | "en"): string {
  return new Intl.NumberFormat(lang === "zh" ? "zh-CN" : "en-US").format(n);
}

/** 趋势导出 CSV */
export function toCSV(daily: DailyPoint[]): string {
  return ["date,pv,uv", ...daily.map((d) => `${d.date},${d.pv},${d.uv}`)].join("\n");
}

/** 发布热力：按日统计文章数，映射 0-4 档 */
export function buildHeatmap(postDates: string[], weeks = 26, end?: Date): { cells: HeatCell[]; total: number } {
  const counts = new Map<string, number>();
  for (const d of postDates) {
    const key = d.slice(0, 10);
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  const endDate = end ?? new Date();
  endDate.setHours(0, 0, 0, 0);
  // 对齐到周六结尾，保证 7 列完整
  const tail = 6 - endDate.getDay();
  endDate.setDate(endDate.getDate() + tail);
  const size = weeks * 7;
  const cells: HeatCell[] = [];
  for (let i = size - 1; i >= 0; i--) {
    const d = new Date(endDate);
    d.setDate(d.getDate() - i);
    const key = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
    const count = counts.get(key) ?? 0;
    const level: HeatCell["level"] = count === 0 ? 0 : count === 1 ? 1 : count === 2 ? 2 : count <= 4 ? 3 : 4;
    cells.push({ date: key, count, level });
  }
  return { cells, total: postDates.length };
}
