/** 访问统计的共享类型（前后端通用，不含 Node 依赖） */

export interface DailyPoint {
  date: string;
  label: string;
  pv: number;
  uv: number;
}

export interface Summary {
  generatedAt: string;
  days: number;
  totals: { pv: number; uv: number; countries: number; pages: number };
  /** 存留期内（非窗口期）的累计量 */
  allTime: { pv: number; uv: number };
  /** 窗口期内：访问天数 ≥2 记为回访客 */
  visitors: { new: number; returning: number };
  daily: DailyPoint[];
  /** 0-23 点访问分布 */
  hours: number[];
  countries: { code: string; pv: number; uv: number }[];
  devices: { label: string; value: number }[];
  refs: { label: string; value: number }[];
  langs: { label: string; value: number }[];
  pages: { path: string; pv: number; uv: number }[];
}
