/**
 * 《智能体 AI 漫游指南》标题图生成器。
 *
 * 为 31 篇文章各生成一张风格统一的「印刷卡片」封面：
 * 纸色底 + 橙色网点纹理 + 章号大字 + 中文章名 + 该章主题图形。
 *
 * 图形用几何线条直接绘制（SVG path），风格与站点的打印机美学一致；
 * 渲染用 @resvg/resvg-js（纯静态，无浏览器依赖）。中文字形按需从
 * Google Fonts 取子集（与 src/lib/og/image.ts 同款做法）。
 *
 * 用法：
 *   node generate-covers.mjs --out ../../content/posts
 */

import { mkdtemp, readdir, readFile, writeFile } from "node:fs/promises";
import { rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { Resvg } from "@resvg/resvg-js";

/** 仓库根目录：本脚本位于 tools/hitchhiker-assets/，所有相对路径都以仓库根为基准。 */
const REPO_ROOT = join(dirname(fileURLToPath(import.meta.url)), "../..");
const OUT_ROOT = resolve(REPO_ROOT, arg("out", "content/posts"));
const MONO_FONT_PATH = resolve(REPO_ROOT, "src/lib/og/fonts/jetbrains-mono-700.ttf");

/** 封面尺寸：与列表缩略图（偏扁）和拍立得相框都能兼容的 3:2。 */
const WIDTH = 1200;
const HEIGHT = 800;

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

// ---------------------------------------------------------------------------
// 站点配色（取自 src/app.css 的 --color-printer-*）
// ---------------------------------------------------------------------------
const INK = "#2c2824";
const INK_LIGHT = "#8a8078";
const PAPER = "#fffef9";
const PAPER_EDGE = "#f0ede6";
const ACCENT = "#ff6b35";

// ---------------------------------------------------------------------------
// 各章主题图形：1200×800 画布上，右侧 ~420×420 区域内的纯几何绘制。
// 每个函数返回 SVG 片段（不含 <svg> 外壳）。
// ---------------------------------------------------------------------------

const TAU = Math.PI * 2;

/** 圆形排布的点，用于表示 Transformer 的注意力/序列。 */
function dotRing(cx, cy, radius, count, size, color = INK) {
  const dots = [];
  for (let i = 0; i < count; i++) {
    const angle = (i / count) * TAU - Math.PI / 2;
    const x = cx + Math.cos(angle) * radius;
    const y = cy + Math.sin(angle) * radius;
    dots.push(
      `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${size}" fill="${color}"/>`,
    );
  }
  return dots.join("");
}

/** 注意力连线：从底部一点连向一排点，透明度递减。 */
function attentionFan(cx, cy, width, count, color = ACCENT) {
  const lines = [];
  for (let i = 0; i < count; i++) {
    const x = cx - width / 2 + (width * i) / (count - 1);
    const alpha = (0.15 + 0.85 * (i / (count - 1))).toFixed(2);
    lines.push(
      `<line x1="${cx}" y1="${cy + 120}" x2="${x.toFixed(1)}" y2="${cy - 90}" stroke="${color}" stroke-width="1.5" stroke-opacity="${alpha}"/>`,
    );
  }
  return lines.join("");
}

/** 堆叠的层（Transformer 层 / 记忆层）。 */
function stackLayers(x, y, w, h, count, gap = 14) {
  const parts = [];
  for (let i = 0; i < count; i++) {
    const oy = y + i * (h + gap);
    parts.push(
      `<rect x="${x}" y="${oy}" width="${w}" height="${h}" rx="3" fill="none" stroke="${INK}" stroke-width="1.5" stroke-opacity="${0.25 + 0.14 * i}"/>`,
    );
  }
  return parts.join("");
}

/** 网格（GPU 阵列 / 并行 / 批处理）。 */
function grid(x, y, cols, rows, cell, gap = 8, color = INK) {
  const parts = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const fill = (r + c) % 3 === 0 ? ACCENT : color;
      const opacity = (r + c) % 3 === 0 ? 0.9 : 0.22;
      parts.push(
        `<rect x="${x + c * (cell + gap)}" y="${y + r * (cell + gap)}" width="${cell}" height="${cell}" rx="2" fill="${fill}" fill-opacity="${opacity}"/>`,
      );
    }
  }
  return parts.join("");
}

/** 流程箭头。 */
function arrow(x1, y1, x2, y2, color = ACCENT, width = 2) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const head = 12;
  const hx1 = x2 - head * Math.cos(angle - Math.PI / 7);
  const hy1 = y2 - head * Math.sin(angle - Math.PI / 7);
  const hx2 = x2 - head * Math.cos(angle + Math.PI / 7);
  const hy2 = y2 - head * Math.sin(angle + Math.PI / 7);
  return (
    `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}"/>` +
    `<polygon points="${x2},${y2} ${hx1.toFixed(1)},${hy1.toFixed(1)} ${hx2.toFixed(1)},${hy2.toFixed(1)}" fill="${color}"/>`
  );
}

/** 折线（曲线/轨迹）。 */
function polyline(points, color = INK, width = 2, opacity = 1) {
  const d = points.map(([x, y], i) => `${i ? "L" : "M"}${x},${y}`).join(" ");
  return `<path d="${d}" fill="none" stroke="${color}" stroke-width="${width}" stroke-opacity="${opacity}" stroke-linejoin="round" stroke-linecap="round"/>`;
}

/**
 * 每章一个主题图形。坐标基于 1200×800 画布，
 * 统一落在右侧 (x: 700–1080, y: 220–620) 区域内，避免与左侧文字重叠。
 */
const MOTIFS = {
  // 免责声明：一枚带「印章 / 验詁」意味的圆形图章 + 盾形底
  disclaimer: () => `
    <path d="M900,290 L1000,330 L1000,450 Q1000,530 900,580 Q800,530 800,450 L800,330 Z" fill="${ACCENT}" fill-opacity="0.10" stroke="${ACCENT}" stroke-width="2.5"/>
    <path d="M856,436 L888,468 L950,398" fill="none" stroke="${ACCENT}" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>
    ${dotRing(900, 430, 148, 24, 4)}
  `,

  // 前言：一本打开的书 + 书签
  preface: () => `
    <path d="M760,340 Q830,320 900,340 L900,540 Q830,520 760,540 Z" fill="none" stroke="${INK}" stroke-width="2"/>
    <path d="M1040,340 Q970,320 900,340 L900,540 Q970,520 1040,540 Z" fill="none" stroke="${INK}" stroke-width="2"/>
    <line x1="900" y1="340" x2="900" y2="540" stroke="${INK}" stroke-width="2" stroke-opacity="0.5"/>
    <rect x="880" y="300" width="18" height="60" rx="2" fill="${ACCENT}"/>
    ${[380, 420, 460, 500].map((y) => `<line x1="782" y1="${y}" x2="868" y2="${y}" stroke="${INK}" stroke-width="1.5" stroke-opacity="0.25"/>`).join("")}
    ${[380, 420, 460, 500].map((y) => `<line x1="932" y1="${y}" x2="1018" y2="${y}" stroke="${INK}" stroke-width="1.5" stroke-opacity="0.25"/>`).join("")}
  `,

  // 第 1 章 LLM 架构：注意力扇形 + 多层堆叠
  architecture: () => `
    ${stackLayers(760, 300, 300, 34, 4, 16)}
    ${dotRing(910, 560, 90, 10, 6)}
    <circle cx="${910}" cy="560" r="52" fill="${ACCENT}" fill-opacity="0.9"/>
    ${attentionFan(910, 560, 260, 9)}
  `,

  // 第 2 章 系统基础：GPU 网格 + 显存分层
  systems: () => `
    ${grid(770, 280, 4, 3, 62, 16)}
    <line x1="760" y1="560" x2="1060" y2="560" stroke="${INK}" stroke-width="2"/>
    ${[0, 1, 2].map((i) => `<rect x="${780 + i * 100}" y="${580 + i * 24}" width="${260 - i * 100}" height="16" rx="2" fill="${ACCENT}" fill-opacity="${0.85 - i * 0.25}"/>`).join("")}
  `,

  // 第 3 章 强化学习导论：状态-动作-奖励循环
  rlIntro: () => `
    <circle cx="910" cy="440" r="130" fill="none" stroke="${INK}" stroke-width="2" stroke-opacity="0.3" stroke-dasharray="6 8"/>
    <circle cx="910" cy="310" r="26" fill="${INK}"/>
    <circle cx="1040" cy="505" r="26" fill="${ACCENT}"/>
    <circle cx="780" cy="505" r="26" fill="${INK}" fill-opacity="0.55"/>
    ${arrow(930, 322, 1020, 486)}
    ${arrow(1016, 520, 806, 520)}
    ${arrow(792, 486, 896, 328)}
  `,

  // 第 4 章 RL 基础：策略梯度上升
  rlFoundations: () => `
    ${polyline([[770, 560], [820, 500], [870, 520], [920, 430], [970, 450], [1020, 340]], ACCENT, 4)}
    ${polyline([[770, 600], [860, 580], [950, 590], [1050, 560]], INK, 2, 0.3)}
    <circle cx="1020" cy="340" r="9" fill="${ACCENT}"/>
    ${[0, 1, 2, 3, 4].map((i) => `<line x1="${790 + i * 60}" y1="640" x2="${790 + i * 60}" y2="620" stroke="${INK}" stroke-width="1.5" stroke-opacity="0.2"/>`).join("")}
  `,

  // 第 5 章 PPO：裁剪区间 [0.8, 1.2]
  ppo: () => `
    <line x1="770" y1="500" x2="1060" y2="500" stroke="${INK}" stroke-width="2"/>
    <line x1="770" y1="500" x2="770" y2="300" stroke="${INK}" stroke-width="2"/>
    ${polyline([[790, 560], [850, 440], [900, 340], [940, 320], [990, 320], [1040, 320]], INK, 3, 0.35)}
    <rect x="880" y="300" width="150" height="200" fill="${ACCENT}" fill-opacity="0.14"/>
    <line x1="880" y1="290" x2="880" y2="510" stroke="${ACCENT}" stroke-width="2" stroke-dasharray="5 5"/>
    <line x1="1030" y1="290" x2="1030" y2="510" stroke="${ACCENT}" stroke-width="2" stroke-dasharray="5 5"/>
    ${polyline([[790, 560], [860, 450], [900, 380], [1030, 380]], ACCENT, 4)}
  `,

  // 第 6 章 DPO：两条回答的偏好对比
  dpo: () => `
    <rect x="770" y="300" width="120" height="200" rx="4" fill="none" stroke="${INK}" stroke-width="2" stroke-opacity="0.3"/>
    <rect x="960" y="300" width="120" height="200" rx="4" fill="none" stroke="${ACCENT}" stroke-width="2"/>
    ${[330, 370, 410, 450].map((y) => `<line x1="790" y1="${y}" x2="870" y2="${y}" stroke="${INK}" stroke-width="1.5" stroke-opacity="0.2"/>`).join("")}
    ${[330, 370, 410, 450].map((y) => `<line x1="980" y1="${y}" x2="1060" y2="${y}" stroke="${ACCENT}" stroke-width="1.5" stroke-opacity="0.5"/>`).join("")}
    <path d="M900,400 l14,16 l26,-30" fill="none" stroke="${ACCENT}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
    <text x="1060" y="290" font-family="JetBrains Mono, Noto Sans SC" font-size="26" fill="${ACCENT}" text-anchor="end">+1</text>
    <text x="870" y="290" font-family="JetBrains Mono, Noto Sans SC" font-size="26" fill="${INK_LIGHT}" text-anchor="end">0</text>
  `,

  // 第 7 章 GRPO：一组样本 + 组内相对优势
  grpo: () => `
    ${[0, 1, 2, 3, 4].map((i) => {
      const h = [80, 150, 110, 200, 130][i];
      const y = 540 - h;
      const isHigh = h >= 150;
      return `<rect x="${790 + i * 56}" y="${y}" width="36" height="${h}" rx="3" fill="${isHigh ? ACCENT : INK}" fill-opacity="${isHigh ? 0.9 : 0.25}"/>`;
    }).join("")}
    <line x1="770" y1="540" x2="1070" y2="540" stroke="${INK}" stroke-width="2"/>
    <line x1="770" y1="420" x2="1070" y2="420" stroke="${ACCENT}" stroke-width="1.5" stroke-dasharray="6 6"/>
    <text x="1070" y="410" font-family="JetBrains Mono, Noto Sans SC" font-size="20" fill="${ACCENT}" text-anchor="end">mean</text>
  `,

  // 第 8 章 偏好优化变体：多条目标曲线
  variants: () => `
    ${polyline([[780, 560], [860, 470], [940, 430], [1040, 400]], ACCENT, 3)}
    ${polyline([[780, 580], [860, 520], [940, 490], [1040, 470]], INK, 2, 0.45)}
    ${polyline([[780, 600], [860, 570], [940, 550], [1040, 540]], INK, 2, 0.28)}
    ${polyline([[780, 620], [860, 605], [940, 600], [1040, 598]], INK, 2, 0.18)}
    ${[0, 1, 2, 3].map((i) => `<circle cx="${780 + i * 87}" cy="${[560, 580, 600, 620][i]}" r="6" fill="${[ACCENT, INK, INK, INK][i]}" fill-opacity="${[1, 0.45, 0.28, 0.18][i]}"/>`).join("")}
  `,

  // 第 9 章 奖励模型：打分刻度
  rewardModel: () => `
    <rect x="770" y="330" width="290" height="180" rx="6" fill="none" stroke="${INK}" stroke-width="2" stroke-opacity="0.3"/>
    ${[0, 1, 2, 3, 4, 5, 6, 7, 8].map((i) => `<line x1="${790 + i * 33}" y1="420" x2="${790 + i * 33}" y2="${[420 - 60, 420 - 20][i % 2]} " stroke="${INK}" stroke-width="1.5" stroke-opacity="0.28"/>`).join("")}
    <rect x="770" y="440" width="180" height="70" rx="6" fill="${ACCENT}" fill-opacity="0.85"/>
    <text x="860" y="487" font-family="JetBrains Mono, Noto Sans SC" font-size="30" fill="${PAPER}" text-anchor="middle">0.87</text>
    ${arrow(1060, 460, 1110, 460)}
  `,

  // 第 10 章 SFT：数据 → 模型
  sft: () => `
    ${[0, 1, 2].map((i) => `<rect x="${770 + i * 16}" y="${330 + i * 14}" width="150" height="90" rx="4" fill="${PAPER_EDGE}" stroke="${INK}" stroke-width="1.5" stroke-opacity="${0.5 - i * 0.12}"/>`).join("")}
    ${arrow(950, 390, 1000, 390)}
    <rect x="1010" y="330" width="90" height="150" rx="6" fill="none" stroke="${ACCENT}" stroke-width="2.5"/>
    ${[0, 1, 2].map((i) => `<circle cx="${1055}" cy="${365 + i * 40}" r="8" fill="${ACCENT}" fill-opacity="${0.9 - i * 0.28}"/>`).join("")}
  `,

  // 第 11 章 大规模系统：集群 + 并行切分
  infrastructure: () => `
    ${grid(770, 290, 5, 3, 44, 14)}
    ${[0, 1, 2].map((i) => `<rect x="${770 + i * 100}" y="${540}" width="${290 - i * 100}" height="26" rx="3" fill="${INK}" fill-opacity="${0.5 - i * 0.15}"/>`).join("")}
    ${arrow(910, 520, 910, 556, ACCENT)}
  `,

  // 第 12 章 智能体训练：轨迹回放
  agenticTraining: () => `
    ${polyline([[780, 560], [830, 500], [880, 520], [930, 440], [980, 460], [1030, 360], [1060, 300]], ACCENT, 3)}
    ${[0, 1, 2, 3, 4].map((i) => {
      const pts = [[780, 560], [830, 500], [880, 520], [930, 440], [980, 460]][i];
      return `<circle cx="${pts[0]}" cy="${pts[1]}" r="8" fill="${PAPER}" stroke="${ACCENT}" stroke-width="2.5"/>`;
    }).join("")}
    <rect x="770" y="600" width="290" height="46" rx="4" fill="none" stroke="${INK}" stroke-width="1.5" stroke-opacity="0.25"/>
    <rect x="770" y="600" width="190" height="46" rx="4" fill="${ACCENT}" fill-opacity="0.8"/>
  `,

  // 第 13 章 推理模型：思维树搜索
  reasoning: () => {
    const nodes = [
      [910, 300], [830, 400], [990, 400], [790, 500],
      [870, 500], [950, 500], [1030, 500],
    ];
    const edges = [[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [2, 6]];
    return `
      ${edges.map(([a, b]) => `<line x1="${nodes[a][0]}" y1="${nodes[a][1]}" x2="${nodes[b][0]}" y2="${nodes[b][1]}" stroke="${INK}" stroke-width="1.5" stroke-opacity="0.28"/>`).join("")}
      ${nodes.map(([x, y], i) => `<circle cx="${x}" cy="${y}" r="${i === 0 ? 20 : 13}" fill="${i === 6 ? ACCENT : INK}" fill-opacity="${i === 6 ? 1 : 0.35}"/>`).join("")}
      ${polyline([[790, 500], [870, 500], [950, 500], [1030, 500]], ACCENT, 3, 0.9)}
    `;
  },

  // 第 14 章 评估：基准刻度
  evaluation: () => `
    <rect x="770" y="320" width="290" height="200" rx="6" fill="none" stroke="${INK}" stroke-width="2" stroke-opacity="0.3"/>
    ${[0, 1, 2, 3].map((i) => `<rect x="${790}" y="${350 + i * 45}" width="${[240, 200, 160, 110][i]}" height="24" rx="3" fill="${i === 0 ? ACCENT : INK}" fill-opacity="${i === 0 ? 0.9 : 0.22}"/>`).join("")}
    <text x="1060" y="560" font-family="JetBrains Mono, Noto Sans SC" font-size="24" fill="${INK_LIGHT}" text-anchor="end">pass@k</text>
  `,

  // 第 15 章 智能体导论：感知-推理-行动循环
  agenticIntro: () => `
    <circle cx="910" cy="450" r="120" fill="none" stroke="${ACCENT}" stroke-width="3" stroke-dasharray="10 8"/>
    <circle cx="910" cy="450" r="46" fill="${INK}"/>
    ${dotRing(910, 450, 120, 3, 14, ACCENT)}
    <line x1="910" y1="330" x2="910" y2="330" stroke="${INK}"/>
  `,

  // 第 16 章 RAG：文档 → 检索 → 生成
  rag: () => `
    ${[0, 1, 2].map((i) => `<rect x="770" y="${310 + i * 70}" width="110" height="56" rx="4" fill="none" stroke="${INK}" stroke-width="1.5" stroke-opacity="${0.5 - i * 0.12}"/>`).join("")}
    ${[0, 1, 2].map((i) => `<line x1="790" y1="${332 + i * 70}" x2="860" y2="${332 + i * 70}" stroke="${INK}" stroke-width="1.5" stroke-opacity="0.25"/>`).join("")}
    ${arrow(890, 420, 950, 420)}
    ${grid(960, 340, 2, 2, 40, 12)}
    ${arrow(1000, 440, 1000, 490)}
    <rect x="950" y="500" width="90" height="70" rx="6" fill="${ACCENT}" fill-opacity="0.85"/>
  `,

  // 第 17 章 记忆系统：分层记忆
  memory: () => `
    ${stackLayers(770, 290, 290, 52, 4, 18)}
    <rect x="850" y="600" width="130" height="40" rx="20" fill="${ACCENT}" fill-opacity="0.85"/>
    ${arrow(915, 560, 915, 596)}
  `,

  // 第 18 章 Agent Harness：编排调度
  harness: () => `
    <rect x="880" y="290" width="120" height="60" rx="6" fill="${ACCENT}" fill-opacity="0.9"/>
    ${[0, 1, 2].map((i) => `<rect x="${770 + i * 110}" y="470" width="90" height="60" rx="6" fill="none" stroke="${INK}" stroke-width="2" stroke-opacity="0.4"/>`).join("")}
    ${[0, 1, 2].map((i) => arrow(940, 356, 815 + i * 110 + 45, 464).replace(/stroke-opacity="1"/g, "")).join("")}
    <line x1="770" y1="600" x2="1060" y2="600" stroke="${INK}" stroke-width="1.5" stroke-opacity="0.2" stroke-dasharray="4 6"/>
  `,

  // 第 19 章 循环工程：收敛螺旋
  loopEngineering: () => {
    const pts = [];
    for (let i = 0; i <= 60; i++) {
      const t = (i / 60) * Math.PI * 3.4;
      const r = 150 - (i / 60) * 110;
      pts.push([910 + Math.cos(t) * r, 450 + Math.sin(t) * r]);
    }
    return `
      ${polyline(pts, INK, 2, 0.35)}
      ${arrow(pts[56][0], pts[56][1], pts[60][0], pts[60][1], ACCENT, 2.5)}
      <circle cx="910" cy="450" r="14" fill="${ACCENT}"/>
    `;
  },

  // 第 20 章 设计模式：模式拼块
  patterns: () => `
    ${grid(780, 300, 3, 3, 76, 18)}
    <rect x="780" y="300" width="76" height="76" rx="4" fill="${ACCENT}" fill-opacity="0.9"/>
    <rect x="780" y="488" width="76" height="76" rx="4" fill="${ACCENT}" fill-opacity="0.55"/>
  `,

  // 第 21 章 环境与基准：环境网格 + 目标
  environments: () => `
    ${grid(780, 320, 5, 4, 46, 14, INK)}
    <circle cx="990" cy="430" r="34" fill="none" stroke="${ACCENT}" stroke-width="3"/>
    <circle cx="990" cy="430" r="12" fill="${ACCENT}"/>
    <path d="M900,520 q40,-40 80,-60" fill="none" stroke="${ACCENT}" stroke-width="2.5" stroke-dasharray="6 6"/>
  `,

  // 第 22 章 MCP：N×M 连接标准化
  mcp: () => `
    ${[0, 1, 2].map((i) => `<circle cx="790" cy="${350 + i * 90}" r="22" fill="${INK}" fill-opacity="0.45"/>`).join("")}
    ${[0, 1, 2].map((i) => `<rect x="1010" y="${330 + i * 90}" width="60" height="44" rx="6" fill="none" stroke="${INK}" stroke-width="2" stroke-opacity="0.45"/>`).join("")}
    <rect x="880" y="300" width="100" height="240" rx="8" fill="${ACCENT}" fill-opacity="0.16" stroke="${ACCENT}" stroke-width="2.5"/>
    ${[0, 1, 2].map((i) => `<line x1="812" y1="${350 + i * 90}" x2="880" y2="${420}" stroke="${INK}" stroke-width="1.5" stroke-opacity="0.3"/>`).join("")}
    ${[0, 1, 2].map((i) => `<line x1="980" y1="420" x2="1010" y2="${352 + i * 90}" stroke="${INK}" stroke-width="1.5" stroke-opacity="0.3"/>`).join("")}
    <text x="930" y="460" font-family="JetBrains Mono, Noto Sans SC" font-size="30" fill="${ACCENT}" font-weight="bold" text-anchor="middle">N×M</text>
  `,

  // 第 23 章 Agent Skills：能力积木
  skills: () => `
    ${[0, 1, 2, 3].map((i) => `<rect x="${770 + i * 76}" y="300" width="60" height="60" rx="6" fill="${INK}" fill-opacity="${0.18 + i * 0.1}"/>`).join("")}
    ${[0, 1, 2, 3, 4].map((i) => `<rect x="${770 + i * 60}" y="420" width="48" height="48" rx="6" fill="${ACCENT}" fill-opacity="${0.85 - i * 0.13}"/>`).join("")}
    ${arrow(910, 380, 910, 412)}
    <rect x="900" y="520" width="120" height="120" rx="10" fill="${ACCENT}" fill-opacity="0.9"/>
    <path d="M940,580 l16,20 l28,-36" fill="none" stroke="${PAPER}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
  `,

  // 第 24 章 A2A：智能体对等通信
  a2a: () => `
    <circle cx="800" cy="440" r="46" fill="${INK}" fill-opacity="0.5"/>
    <circle cx="1020" cy="440" r="46" fill="${ACCENT}"/>
    ${arrow(852, 420, 968, 420, ACCENT, 2.5)}
    ${arrow(968, 462, 852, 462, INK, 2)}
    <text x="910" y="380" font-family="JetBrains Mono, Noto Sans SC" font-size="22" fill="${INK_LIGHT}" text-anchor="middle">A2A</text>
  `,

  // 第 25 章 多智能体：三种拓扑
  multiAgent: () => `
    <circle cx="790" cy="350" r="20" fill="${ACCENT}"/>
    ${[[900, 320], [900, 420], [1010, 350]].map(([x, y]) => `<circle cx="${x}" cy="${y}" r="18" fill="${INK}" fill-opacity="0.4"/>`).join("")}
    ${[[790, 350, 900, 320], [790, 350, 900, 420], [900, 320, 1010, 350], [900, 420, 1010, 350]].map(([x1, y1, x2, y2]) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${INK}" stroke-width="1.5" stroke-opacity="0.3"/>`).join("")}
    ${[[800, 500], [860, 560], [920, 500], [980, 560], [1040, 500]].map(([x, y]) => `<circle cx="${x}" cy="${y}" r="16" fill="${INK}" fill-opacity="0.3"/>`).join("")}
  `,

  // 第 26 章 开发框架：图执行
  frameworks: () => `
    <circle cx="790" cy="360" r="22" fill="${ACCENT}"/>
    <circle cx="910" cy="320" r="20" fill="${INK}" fill-opacity="0.4"/>
    <circle cx="910" cy="470" r="20" fill="${INK}" fill-opacity="0.4"/>
    <circle cx="1030" cy="400" r="26" fill="${ACCENT}" fill-opacity="0.85"/>
    ${[[790, 360, 910, 320], [790, 360, 910, 470], [910, 320, 1030, 400], [910, 470, 1030, 400]].map(([x1, y1, x2, y2]) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${INK}" stroke-width="2" stroke-opacity="0.3"/>`).join("")}
    <rect x="770" y="560" width="290" height="50" rx="6" fill="none" stroke="${INK}" stroke-width="1.5" stroke-opacity="0.22"/>
  `,

  // 第 27 章 Agentic UI：流式界面
  ui: () => `
    <rect x="770" y="300" width="290" height="200" rx="10" fill="none" stroke="${INK}" stroke-width="2" stroke-opacity="0.35"/>
    ${[0, 1, 2].map((i) => `<rect x="796" y="${410 + i * 0}" width="${[200, 150, 90][i]}" height="18" rx="9" fill="${i === 2 ? ACCENT : INK}" fill-opacity="${i === 2 ? 0.9 : 0.2}"/>`.replace(/y="\d+"/, `y="${335 + i * 46}"`)).join("")}
    <rect x="770" y="530" width="290" height="56" rx="28" fill="${ACCENT}" fill-opacity="0.14" stroke="${ACCENT}" stroke-width="2"/>
    ${[0, 1, 2].map((i) => `<circle cx="${820 + i * 34}" cy="558" r="8" fill="${ACCENT}" fill-opacity="${0.9 - i * 0.3}"/>`).join("")}
  `,

  // 第 28 章 自测题：问号 + 选项
  quiz: () => `
    <circle cx="910" cy="400" r="120" fill="none" stroke="${ACCENT}" stroke-width="3" stroke-opacity="0.25"/>
    <text x="910" y="460" font-family="JetBrains Mono, Noto Sans SC" font-size="150" fill="${ACCENT}" font-weight="bold" text-anchor="middle">?</text>
    ${[0, 1, 2].map((i) => `<rect x="${770 + i * 105}" y="570" width="80" height="44" rx="6" fill="${i === 0 ? ACCENT : INK}" fill-opacity="${i === 0 ? 0.9 : 0.2}"/>`).join("")}
  `,

  // 第 29 章 速查手册：格栅化条目
  quickref: () => `
    ${grid(780, 300, 3, 4, 60, 20)}
    <rect x="780" y="300" width="60" height="60" rx="4" fill="${ACCENT}" fill-opacity="0.9"/>
    <rect x="940" y="460" width="60" height="60" rx="4" fill="${ACCENT}" fill-opacity="0.6"/>
  `,

  // 第 30 章 总结与展望：路径 + 分岔
  conclusion: () => `
    ${polyline([[770, 520], [860, 480], [950, 440], [1040, 380]], ACCENT, 4)}
    ${[0, 1, 2].map((i) => `<circle cx="${770 + i * 135}" cy="${520 - i * 70}" r="14" fill="${PAPER}" stroke="${ACCENT}" stroke-width="3"/>`).join("")}
    ${[0, 1, 2].map((i) => `<line x1="${770 + i * 135}" y1="${520 - i * 70}" x2="${770 + i * 135}" y2="${560}" stroke="${INK}" stroke-width="1.5" stroke-opacity="0.18"/>`).join("")}
    <circle cx="1100" cy="340" r="20" fill="${ACCENT}" fill-opacity="0.4" stroke="${ACCENT}" stroke-width="2"/>
  `,
};

// ---------------------------------------------------------------------------
// 章 → 图形 的映射
// ---------------------------------------------------------------------------
const MOTIF_BY_NUMBER = {
  1: "architecture",
  2: "systems",
  3: "rlIntro",
  4: "rlFoundations",
  5: "ppo",
  6: "dpo",
  7: "grpo",
  8: "variants",
  9: "rewardModel",
  10: "sft",
  11: "infrastructure",
  12: "agenticTraining",
  13: "reasoning",
  14: "evaluation",
  15: "agenticIntro",
  16: "rag",
  17: "memory",
  18: "harness",
  19: "loopEngineering",
  20: "patterns",
  21: "environments",
  22: "mcp",
  23: "skills",
  24: "a2a",
  25: "multiAgent",
  26: "frameworks",
  27: "ui",
  28: "quiz",
  29: "quickref",
  30: "conclusion",
};

// ---------------------------------------------------------------------------
// 纸面纹理：细网格 + 橙色网点抖动块（呼应站点的热敏打印质感）
// ---------------------------------------------------------------------------
function paperTexture(seed) {
  const parts = [];
  // 细网格
  parts.push(
    `<defs><pattern id="grid" width="24" height="24" patternUnits="userSpaceOnUse">` +
      `<path d="M24 0 L0 0 0 24" fill="none" stroke="${INK}" stroke-width="0.5" stroke-opacity="0.05"/>` +
      `</pattern></defs>`,
  );
  parts.push(`<rect width="${WIDTH}" height="${HEIGHT}" fill="url(#grid)"/>`);

  // 右下角橙色网点（Bayer 感，越靠边越稀疏）
  let h = seed * 2654435761;
  const rnd = () => {
    h = (h * 1103515245 + 12345) & 0x7fffffff;
    return h / 0x7fffffff;
  };
  for (let i = 0; i < 420; i++) {
    const x = rng(rnd, 640, WIDTH - 20);
    const y = rng(rnd, 40, HEIGHT - 40);
    // 越靠右下越密集
    const bias = (x - 640) / (WIDTH - 640) + (y - 40) / (HEIGHT - 80);
    if (rnd() > bias / 2.4) continue;
    parts.push(
      `<rect x="${x.toFixed(0)}" y="${y.toFixed(0)}" width="3" height="3" fill="${ACCENT}" fill-opacity="0.5"/>`,
    );
  }
  return parts.join("");
}

function rng(rnd, min, max) {
  return min + rnd() * (max - min);
}

// ---------------------------------------------------------------------------
// 字形
// ---------------------------------------------------------------------------
async function loadCjkSubset(text) {
  const chars = Array.from(new Set(Array.from(text)))
    .filter((ch) => ch.charCodeAt(0) > 127)
    .sort()
    .join("");
  if (!chars) return null;
  try {
    const css = await fetch(
      `https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@700&text=${encodeURIComponent(chars)}`,
      { headers: { "User-Agent": "curl/8" } },
    ).then((r) => r.text());
    const url = css.match(/src:\s*url\(([^)]+)\)/)?.[1];
    if (!url) return null;
    return Buffer.from(await fetch(url).then((r) => r.arrayBuffer()));
  } catch (error) {
    console.warn("[covers] CJK 子集加载失败:", error);
    return null;
  }
}

// ---------------------------------------------------------------------------
// 组装
// ---------------------------------------------------------------------------
/** 标题排版区域：左侧栏，与右侧图形区（x >= 720）互不重叠。 */
const TEXT_LEFT = 96;
const TEXT_MAX_WIDTH = 560; // 96 + 560 = 656，留 64px 与图形区隔离

/**
 * 按实际字宽折行（拉丁字符约占 0.6em，CJK 占 1em），
 * 不依赖字体度量，够用且稳定。
 */
function wrapTitle(title, fontSize, maxWidth) {
  const widthOfText = (text) =>
    Array.from(text).reduce((sum, ch) => sum + charWidth(ch, fontSize), 0);

  // 先把标题切成「断行片段」。断点位于：
  //   - 空格处（空格保留在片段末尾，避免拉丁词之间丢空格）
  //   - CJK 与拉丁/数字的交界处（中文排版惯例）
  //   - 全角括号前后（「（Loop Engineering）」可整体移动，不留孤零零的右括号）
  const OPENERS = "（([【《「‘“";
  const CLOSERS = "）)]】》」’”";
  const tokens = [];
  let buffer = "";

  const flush = () => {
    if (buffer) tokens.push(buffer);
    buffer = "";
  };

  let prev = "";
  for (const ch of title) {
    const isSpace = /\s/.test(ch);
    const curr = isCjkChar(ch);
    const prevCjk = prev ? isCjkChar(prev) : null;

    // 空白：并入当前片段后断点
    if (isSpace) {
      buffer += ch;
      flush();
      prev = ch;
      continue;
    }
    // 语言交界处断点（拉丁 ↔ CJK），但全角括号两侧不额外断
    const isBracket = OPENERS.includes(ch) || CLOSERS.includes(ch);
    const prevBracket = prev ? OPENERS.includes(prev) || CLOSERS.includes(prev) : false;
    const scriptBoundary =
      buffer && prev && prevCjk !== null && curr !== prevCjk && !isSpace;

    if (scriptBoundary && !isBracket && !prevBracket) {
      flush();
    }
    // 左括号前断开：让「（Loop Engineering）」自成一段
    if (OPENERS.includes(ch) && buffer && !isSpace) {
      flush();
    }
    buffer += ch;
    prev = ch;
  }
  flush();

  // 逐个片段装箱；放不下就换行
  const lines = [];
  let current = "";
  for (const token of tokens) {
    const candidate = current + token;
    if (widthOfText(candidate) <= maxWidth || !current) {
      // 单段本身就超宽时，退化为按字符硬断
      if (!current && widthOfText(token) > maxWidth) {
        let chunk = "";
        for (const ch of token) {
          if (chunk && widthOfText(chunk + ch) > maxWidth) {
            lines.push(chunk);
            chunk = ch;
          } else {
            chunk += ch;
          }
        }
        current = chunk;
        continue;
      }
      current = candidate;
      continue;
    }
    lines.push(current.trimEnd());
    current = token.replace(/^\s+/, "");
  }
  if (current.trim()) lines.push(current.trimEnd());
  return lines.filter(Boolean);
}

/** 判断是否为 CJK 字符（用于断行与字体选择）。 */
function isCjkChar(ch) {
  const code = ch.codePointAt(0);
  return (
    (code >= 0x3000 && code <= 0x303f) ||
    (code >= 0x3400 && code <= 0x4dbf) ||
    (code >= 0x4e00 && code <= 0x9fff) ||
    (code >= 0xf900 && code <= 0xfaff) ||
    (code >= 0xff00 && code <= 0xffef)
  );
}

/** 按实际字形宽度估算字符宽度（拉丁约占 0.6em，CJK 与全角占 1em）。 */
function charWidth(ch, fontSize) {
  if (isCjkChar(ch)) return fontSize;
  if (/[\s]/.test(ch)) return fontSize * 0.3;
  return fontSize * 0.6;
}

/**
 * 把一段文字按字体切成若干 <text> 片段。
 * resvg 不做逐字形回退，混排必须显式拆开，否则缺字的片段会渲染成豆腐块。
 */
function renderMixedText(text, opts) {
  const { x, y, fontSize, weight, fill, latinFamily, cjkFamily } = opts;
  const runs = [];
  let current = "";
  let currentIsCjk = null;

  for (const ch of text) {
    const cjk = isCjkChar(ch);
    if (currentIsCjk === null || cjk === currentIsCjk) {
      current += ch;
      currentIsCjk = cjk;
    } else {
      runs.push([current, currentIsCjk]);
      current = ch;
      currentIsCjk = cjk;
    }
  }
  if (current) runs.push([current, currentIsCjk]);

  let cursor = x;
  const parts = [];
  for (const [runText, isCjkRun] of runs) {
    const family = isCjkRun ? cjkFamily : latinFamily;
    parts.push(
      `  <text x="${cursor.toFixed(1)}" y="${y}" font-family="${family}" font-size="${fontSize}" font-weight="${weight}" fill="${fill}">${escapeXml(runText)}</text>`,
    );
    for (const ch of runText) cursor += charWidth(ch, fontSize);
  }
  return parts.join("\n");
}

async function renderCover({ chapterNumber, label, title, motif, cjkFont, monoFont }) {
  const motifSvg = MOTIFS[motif] ? MOTIFS[motif]() : "";

  // 标题按可用宽度折行：从大字号往下试，取第一个「行数放得下且不硬断词」的字号。
  const MAX_LINES = 3;
  let fontSize = 74;
  let lines = wrapTitle(title, fontSize, TEXT_MAX_WIDTH);
  for (const size of [70, 64, 58, 52, 46, 40]) {
    if (lines.length <= MAX_LINES) break;
    fontSize = size;
    lines = wrapTitle(title, fontSize, TEXT_MAX_WIDTH);
  }
  if (lines.length > MAX_LINES) lines = lines.slice(0, MAX_LINES);

  const lineHeight = fontSize * 1.22;
  const blockTop = 430;
  const startY = blockTop + (lines.length - 1) * lineHeight;

  // resvg 不做逐字形回退：必须按字体切成若干 <text> 片段，
  // 拉丁/数字交给 JetBrains Mono，中日韩交给 Noto Sans SC。
  const titleText = lines
    .map((line, i) => {
      const y = (startY + i * lineHeight - (lines.length - 1) * lineHeight).toFixed(0);
      return renderMixedText(line, {
        x: TEXT_LEFT,
        y,
        fontSize,
        weight: 700,
        fill: INK,
        latinFamily: "JetBrains Mono",
        cjkFamily: "Noto Sans SC",
      });
    })
    .join("\n");

  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${WIDTH}" height="${HEIGHT}" viewBox="0 0 ${WIDTH} ${HEIGHT}">
  <rect width="${WIDTH}" height="${HEIGHT}" fill="${PAPER}"/>
  ${paperTexture(chapterNumber ?? 0)}

  <!-- 左侧橙色竖条：与站点纸张的「分节标记」呼应 -->
  <rect x="0" y="0" width="14" height="${HEIGHT}" fill="${ACCENT}"/>

  <!-- 章号大字 -->
  <text x="${TEXT_LEFT}" y="268" font-family="JetBrains Mono" font-size="156" font-weight="bold" fill="${ACCENT}" fill-opacity="0.9">${label}</text>
  <line x1="${TEXT_LEFT}" y1="312" x2="${TEXT_LEFT + 150}" y2="312" stroke="${ACCENT}" stroke-width="4"/>

  <!-- 章名（自动折行，字号随行数递减） -->
${titleText}

  <!-- 主题图形（右侧固定区域） -->
  <g>${motifSvg}</g>

  <!-- 页脚铭牌 -->
  <line x1="${TEXT_LEFT}" y1="${HEIGHT - 92}" x2="${WIDTH - TEXT_LEFT}" y2="${HEIGHT - 92}" stroke="${INK}" stroke-width="1" stroke-opacity="0.14"/>
  <text x="${TEXT_LEFT}" y="${HEIGHT - 56}" font-family="JetBrains Mono" font-size="22" fill="${INK_LIGHT}" letter-spacing="2">THE HITCHHIKER'S GUIDE TO AGENTIC AI</text>
  ${renderMixedText("智能体 AI 漫游指南", { x: WIDTH - TEXT_LEFT - 176, y: HEIGHT - 56, fontSize: 22, weight: 400, fill: INK_LIGHT, latinFamily: "JetBrains Mono", cjkFamily: "Noto Sans SC" })}
</svg>`;

  const resvg = new Resvg(svg, {
    fitTo: { mode: "width", value: WIDTH },
    font: {
      fontFiles: [monoFont, ...(cjkFont ? [cjkFont] : [])],
      loadSystemFonts: false,
      defaultFontFamily: "JetBrains Mono",
    },
  });
  return resvg.render().asPng();
}

function escapeXml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ---------------------------------------------------------------------------
// 主流程：扫描 content/posts 里本系列的文章，逐篇生成 cover.png
// ---------------------------------------------------------------------------
const TEMP_DIR = await mkdtemp(join(tmpdir(), "hh-covers-"));
process.on("exit", () => {
  rmSync(TEMP_DIR, { recursive: true, force: true });
});

/** resvg 的 fontFiles 只接受磁盘路径，把内存字体落到临时文件后返回路径。 */
function saveTempFont(name, buffer) {
  const path = join(TEMP_DIR, name);
  writeFileSync(path, buffer);
  return path;
}

async function main() {
  const monoFont = saveTempFont(
    "mono-700.ttf",
    await readFile(MONO_FONT_PATH),
  );

  const dirs = (await readdir(OUT_ROOT)).filter((d) =>
    d.includes("hitchhiker-agentic-ai"),
  );
  if (dirs.length === 0) throw new Error(`no series posts under ${OUT_ROOT}`);

  let done = 0;
  for (const dir of dirs) {
    const full = join(OUT_ROOT, dir);
    const zh = await readFile(join(full, "zh.md"), "utf8").catch(() => null);
    if (!zh) continue;

    const front = zh.slice(0, zh.indexOf("\n---", 4));
    const title = /title: "(.*)"/.exec(front)?.[1] ?? dir;

    // 章号：从 slug 取（hitchhiker-agentic-ai-05-…）；前言与免责声明没有号
    const numMatch = /hitchhiker-agentic-ai-(\d+)-/.exec(dir);
    const chapterNumber = numMatch ? Number(numMatch[1]) : null;
    const isDisclaimer = dir.includes("00-disclaimer");

    // 章名：去掉「智能体 AI 漫游指南 · 第 N 章 」/「· 前言」/「· 免责声明」前缀
    const shortTitle = title
      .replace(/^智能体 AI 漫游指南\s*·\s*/, "")
      .replace(/^第\s*\d+\s*章\s*/, "")
      .trim();

    // 字体子集：只取本篇实际用到的汉字
    const subsetText = `${title}${shortTitle}智能体 漫游指南`;
    const cjk = await loadCjkSubset(subsetText);
    const cjkFont = cjk ? saveTempFont(`cjk-${dir}.ttf`, cjk) : null;

    const png = await renderCover({
      chapterNumber,
      label: isDisclaimer ? "00" : chapterNumber ? String(chapterNumber).padStart(2, "0") : "00",
      title: shortTitle,
      motif: isDisclaimer
        ? "disclaimer"
        : chapterNumber
          ? MOTIF_BY_NUMBER[chapterNumber]
          : "preface",
      cjkFont,
      monoFont,
    });

    await writeFile(join(full, "cover.png"), png);
    done += 1;
    console.log(`[cover] ${dir} -> ${shortTitle}`);
  }

  console.log(`[cover] 完成 ${done} 张`);
}

await main();
