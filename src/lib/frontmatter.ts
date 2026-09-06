/**
 * 共享的 frontmatter 解析/序列化工具（客户端与服务端通用，无 Node 依赖）。
 *
 * 本项目的文章 frontmatter 是扁平 `key: value` 结构（title/slug/lang/date/
 * draft/featured/categories，以及可选的 cover/video/description/keywords/
 * updated/wechatLink）。这里只处理这一子集，保证读写无损往返。
 */

export interface FrontMatterResult {
  data: Record<string, unknown>;
  body: string;
}

/** 解析 `---\n...\n---\n\nbody` 形式的扁平 frontmatter。 */
export function parseFrontMatter(raw: string): FrontMatterResult {
  const m = raw.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/);
  if (!m) return { data: {}, body: raw };

  const data: Record<string, unknown> = {};
  const lines = m[1].split(/\r?\n/);

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.trim() === "") continue;

    // 块状数组：`key:` 之后跟随若干 `- item` 行。
    const blockKey = line.match(/^([A-Za-z_][A-Za-z0-9_-]*):\s*$/);
    if (blockKey) {
      const items: unknown[] = [];
      while (i + 1 < lines.length && /^\s*-\s+/.test(lines[i + 1])) {
        items.push(parseScalar(lines[i + 1].replace(/^\s*-\s+/, "").trim()));
        i++;
      }
      data[blockKey[1]] = items;
      continue;
    }

    const kv = line.match(/^([A-Za-z_][A-Za-z0-9_-]*):\s*(.*)$/);
    if (kv) data[kv[1]] = parseScalar(kv[2].trim());
  }

  return { data, body: m[2] };
}

function parseScalar(raw: string): unknown {
  if (raw === "true") return true;
  if (raw === "false") return false;
  if (raw === "") return "";
  if (raw.startsWith("[") && raw.endsWith("]")) {
    const inner = raw.slice(1, -1).trim();
    if (inner === "") return [];
    return splitList(inner).map(parseScalar);
  }
  if (raw.startsWith('"') && raw.endsWith('"')) {
    try {
      return JSON.parse(raw);
    } catch {
      return raw.slice(1, -1);
    }
  }
  if (raw.startsWith("'") && raw.endsWith("'")) return raw.slice(1, -1);
  if (/^-?\d+(\.\d+)?$/.test(raw)) return Number(raw);
  return raw;
}

/** 按逗号切分，忽略引号内的逗号。 */
function splitList(inner: string): string[] {
  const out: string[] = [];
  let current = "";
  let quote: string | null = null;
  for (const ch of inner) {
    if (quote) {
      current += ch;
      if (ch === quote) quote = null;
    } else if (ch === '"' || ch === "'") {
      quote = ch;
      current += ch;
    } else if (ch === ",") {
      out.push(current.trim());
      current = "";
    } else {
      current += ch;
    }
  }
  if (current.trim() !== "") out.push(current.trim());
  return out;
}

/** 将值序列化为 frontmatter 标量。字符串统一用 JSON 双引号，避免歧义。 */
export function stringifyScalar(value: unknown): string {
  if (value === null || value === undefined) return '""';
  if (typeof value === "string") return JSON.stringify(value);
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map(stringifyScalar).join(", ")}]`;
  }
  return JSON.stringify(String(value));
}

/** 序列化扁平 frontmatter（不含 velite 注入的 path/excerpt/content）。 */
export function serializeFrontMatter(
  data: Record<string, unknown>,
  body: string,
): string {
  const lines: string[] = ["---"];
  for (const [key, value] of Object.entries(data)) {
    if (value === undefined) continue;
    lines.push(`${key}: ${stringifyScalar(value)}`);
  }
  lines.push("---", "", body.trimEnd());
  return `${lines.join("\n")}\n`;
}
