/**
 * 生成免责声明一文的「来源与许可」流程图（1200×800）。
 *
 * 与 generate-covers.mjs 同一套视觉语言：纸色底、橙色网点、JetBrains Mono
 * 拉丁字形 + Noto Sans SC 中文子集。resvg 不做逐字形字体回退，因此每个
 * 文本片段都按字体拆成独立的 <text> 元素。
 *
 * 用法：node generate-disclaimer-figure.mjs
 */
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { Resvg } from "@resvg/resvg-js";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(HERE, "../..");
const OUT = resolve(
  REPO_ROOT,
  "content/posts/2026-09-16-hitchhiker-agentic-ai-00-disclaimer/fig_disclaimer.png",
);

const W = 1200;
const H = 800;
const INK = "#2c2824";
const PAPER = "#fffef9";
const ACCENT = "#ff6b35";
const MUTED = "#8a8378";

const MONO_FONT_PATH = resolve(
  REPO_ROOT,
  "src/lib/og/fonts/jetbrains-mono-700.ttf",
);

const TEMP_DIR = mkdtempSync(join(tmpdir(), "hh-disclaimer-"));
process.on("exit", () => rmSync(TEMP_DIR, { recursive: true, force: true }));

// ── 文本排版 ────────────────────────────────────────────────────────────

function isCjkChar(ch) {
  const c = ch.codePointAt(0);
  return (
    (c >= 0x3000 && c <= 0x303f) ||
    (c >= 0x3400 && c <= 0x4dbf) ||
    (c >= 0x4e00 && c <= 0x9fff) ||
    (c >= 0xf900 && c <= 0xfaff) ||
    (c >= 0xff00 && c <= 0xffef)
  );
}

/**
 * 把混合文本拆成「同字体连续片段」，每段一个 <text>。
 * resvg 不逐字形回退，所以中英必须分开渲染。
 */
function renderMixedText(text, { x, y, size, fill, weight = 700, anchor = "start" }) {
  const runs = [];
  for (const ch of text) {
    const cjk = isCjkChar(ch);
    const last = runs[runs.length - 1];
    if (last && last.cjk === cjk) last.text += ch;
    else runs.push({ cjk, text: ch });
  }

  // 先按各自宽度算出每段起点：resvg 的 text-anchor 不支持整行居中，
  // 需手动分配 x。
  const widthOf = (t, cjk) =>
    Array.from(t).reduce((sum, ch) => {
      if (/[\s]/.test(ch)) return sum + size * 0.3;
      return sum + (cjk ? size : size * 0.6);
    }, 0);

  const total = runs.reduce((s, r) => s + widthOf(r.text, r.cjk), 0);
  let cursor = anchor === "middle" ? x - total / 2 : anchor === "end" ? x - total : x;

  return runs
    .map((r) => {
      const start = cursor;
      cursor += widthOf(r.text, r.cjk);
      const family = r.cjk ? "Noto Sans SC" : "JetBrains Mono";
      return `<text x="${start.toFixed(1)}" y="${y}" font-family="${family}" font-size="${size}" font-weight="${weight}" fill="${fill}">${escapeXml(r.text)}</text>`;
    })
    .join("");
}

function escapeXml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

/** 按可用宽度折行（空格 / CJK↔拉丁交界处断）。 */
function wrapMixed(text, size, maxWidth) {
  const tokens = [];
  let buf = "";
  let bufCjk = null;
  for (const ch of text) {
    if (/\s/.test(ch)) {
      if (buf) tokens.push({ text: buf, cjk: bufCjk });
      buf = "";
      bufCjk = null;
      continue;
    }
    const cjk = isCjkChar(ch);
    if (buf && bufCjk !== null && cjk !== bufCjk) {
      tokens.push({ text: buf, cjk: bufCjk });
      buf = "";
    }
    buf += ch;
    bufCjk = cjk;
  }
  if (buf) tokens.push({ text: buf, cjk: bufCjk });

  const widthOf = (t, cjk) =>
    Array.from(t).reduce(
      (s, ch) => s + (/[\s]/.test(ch) ? size * 0.3 : cjk ? size : size * 0.6),
      0,
    );

  const lines = [];
  let cur = "";
  let curCjk = null;
  for (const tk of tokens) {
    const joined = cur ? cur + tk.text : tk.text;
    if (widthOf(joined, curCjk ?? tk.cjk) <= maxWidth || !cur) {
      cur = joined;
      curCjk = curCjk ?? tk.cjk;
      continue;
    }
    lines.push({ text: cur, cjk: curCjk });
    cur = tk.text;
    curCjk = tk.cjk;
  }
  if (cur) lines.push({ text: cur, cjk: curCjk });
  return lines;
}

/** 带引线的文本块：第一行用主色小标，下面跟正文。 */
function textBlock(lines, { x, y, size, fill, anchor = "middle", lineGap = 1.35 }) {
  return lines
    .map((line, i) =>
      renderMixedText(line.text, {
        x,
        y: y + i * size * lineGap,
        size,
        fill,
        weight: 700,
        anchor,
      }),
    )
    .join("");
}

// ── 图形 ────────────────────────────────────────────────────────────────

function bayerDots(x0, y0, cols, rows, cell, color, alpha) {
  const Bayer = [
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
  ];
  let out = "";
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = Bayer[r % 4][c % 4] / 16;
      if (v < 0.45) continue;
      out += `<circle cx="${x0 + c * cell}" cy="${y0 + r * cell}" r="${(cell * 0.3 * (0.6 + v)).toFixed(2)}" fill="${color}" fill-opacity="${(alpha * v).toFixed(2)}"/>`;
    }
  }
  return out;
}

/** 左侧「来源」卡片。 */
function sourceCard(x, y, label, lines) {
  const w = 300;
  const h = 176;
  return `
    <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="#ffffff" fill-opacity="0.85" stroke="${INK}" stroke-opacity="0.25" stroke-width="2"/>
    <rect x="${x}" y="${y}" width="${w}" height="5" rx="2" fill="${ACCENT}"/>
    ${textBlock([{ text: label }], { x: x + w / 2, y: y + 56, size: 30, fill: INK })}
    ${lines
      .map(
        (l, i) =>
          renderMixedText(l, {
            x: x + w / 2,
            y: y + 104 + i * 34,
            size: 21,
            fill: MUTED,
            weight: 400,
            anchor: "middle",
          }),
      )
      .join("")}
  `;
}

/** 中间「重排版」卡片（强调色描边）。 */
function processCard(x, y, lines) {
  const w = 300;
  const h = 176;
  return `
    <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="${ACCENT}" fill-opacity="0.07" stroke="${ACCENT}" stroke-width="2.5"/>
    <rect x="${x}" y="${y}" width="${w}" height="5" rx="2" fill="${ACCENT}"/>
    ${lines
      .map(
        (l, i) =>
          renderMixedText(l, {
            x: x + w / 2,
            y: y + 74 + i * 40,
            size: 26,
            fill: INK,
            weight: 700,
            anchor: "middle",
          }),
      )
      .join("")}
  `;
}

/** 右侧「发布」卡片。 */
function publishCard(x, y) {
  const w = 300;
  const h = 176;
  return `
    <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="#ffffff" fill-opacity="0.85" stroke="${INK}" stroke-opacity="0.25" stroke-width="2"/>
    <rect x="${x}" y="${y}" width="${w}" height="5" rx="2" fill="${ACCENT}"/>
    ${textBlock([{ text: "本站发布" }], { x: x + w / 2, y: y + 58, size: 30, fill: INK })}
    <rect x="${x + 58}" y="${y + 96}" width="184" height="38" rx="19" fill="${ACCENT}" fill-opacity="0.12" stroke="${ACCENT}" stroke-width="1.5"/>
    ${renderMixedText("CC BY-SA 4.0", {
      x: x + w / 2,
      y: y + 122,
      size: 19,
      fill: ACCENT,
      weight: 700,
      anchor: "middle",
    })}
  `;
}

/** 两卡片之间的箭头。 */
function arrow(x1, x2, y) {
  return `
    <line x1="${x1}" y1="${y}" x2="${x2 - 12}" y2="${y}" stroke="${INK}" stroke-opacity="0.45" stroke-width="2.5"/>
    <path d="M${x2 - 16},${y - 9} L${x2 - 2},${y} L${x2 - 16},${y + 9} Z" fill="${INK}" fill-opacity="0.45"/>
  `;
}

async function loadCjkSubset(text) {
  const url = `https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;700&text=${encodeURIComponent(text)}`;
  const css = await fetch(url, { headers: { "User-Agent": "curl/8" } }).then((r) => {
    if (!r.ok) throw new Error(`Google Fonts ${r.status}`);
    return r.text();
  });
  const m = css.match(/src:\s*url\(([^)]+)\)/);
  if (!m) throw new Error("未在 Google Fonts 响应中找到字体地址");
  const buf = await fetch(m[1]).then((r) => {
    if (!r.ok) throw new Error(`字体下载失败 ${r.status}`);
    return r.arrayBuffer();
  });
  return Buffer.from(buf);
}

async function main() {
  const monoFont = join(TEMP_DIR, "mono.ttf");
  writeFileSync(monoFont, await readFile(MONO_FONT_PATH));

  const caption =
    "来源与许可：从原始来源，经过中英双语重排版与优化，最终以 CC BY-SA 4.0 许可在本站发布。";
  // 子集必须覆盖画面上出现的每一个汉字，漏一个字就会渲染成豆腐块
  const allText = `来源说明原始重排版双语优化本站发布许可译仓库智能体漫游指南${caption}`;
  const cjkPath = join(TEMP_DIR, "cjk.ttf");
  writeFileSync(cjkPath, await loadCjkSubset(allText));

  const cardY = 232;
  const cardW = 300;
  const gap = 90;
  const x1 = 90;
  const x2 = x1 + cardW + gap;
  const x3 = x2 + cardW + gap;

  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
  <rect width="${W}" height="${H}" fill="${PAPER}"/>
  <rect x="0" y="0" width="14" height="${H}" fill="${ACCENT}"/>
  ${bayerDots(1010, 60, 20, 26, 9, ACCENT, 0.5)}
  ${bayerDots(60, 560, 18, 20, 9, ACCENT, 0.32)}

  ${renderMixedText("来源与许可", { x: 90, y: 118, size: 52, fill: INK, anchor: "start" })}
  <line x1="90" y1="148" x2="300" y2="148" stroke="${ACCENT}" stroke-width="5"/>
  ${renderMixedText("Sources & Licensing", { x: 90, y: 186, size: 24, fill: MUTED, weight: 400, anchor: "start" })}

  ${sourceCard(x1, cardY, "原始来源", ["arXiv:2606.24937v2", "GitHub 中译仓库"])}
  ${processCard(x2, cardY, ["中英双语", "重排版与优化"])}
  ${publishCard(x3, cardY)}

  ${arrow(x1 + cardW + 16, x1 + cardW + gap - 16, cardY + 88)}
  ${arrow(x2 + cardW + 16, x2 + cardW + gap - 16, cardY + 88)}

  <line x1="90" y1="530" x2="1110" y2="530" stroke="${INK}" stroke-opacity="0.15" stroke-width="1.5"/>

  ${wrapMixed(caption, 22, 1020)
    .map((l, i) =>
      renderMixedText(l.text, {
        x: 90,
        y: 590 + i * 40,
        size: 22,
        fill: MUTED,
        weight: 400,
        anchor: "start",
      }),
    )
    .join("")}

  ${renderMixedText("THE HITCHHIKER'S GUIDE TO AGENTIC AI", { x: 90, y: 750, size: 20, fill: MUTED, weight: 400, anchor: "start" })}
  ${renderMixedText("智能体 AI 漫游指南", { x: 1110, y: 750, size: 20, fill: MUTED, weight: 400, anchor: "end" })}
</svg>`;

  const png = new Resvg(svg, {
    font: { fontFiles: [monoFont, cjkPath], loadSystemFonts: false },
  })
    .render()
    .asPng();

  writeFileSync(OUT, png);
  console.log(`[figure] ${OUT}`);
}

await main();
