import { error, json, type RequestHandler } from "@sveltejs/kit";
import { env } from "$env/dynamic/private";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import { stringifyScalar } from "$lib/frontmatter";

const SITE_FILE = "content/site.yml";

function guard() {
  if (env.LOCAL_ADMIN !== "1") throw error(404);
}

export const prerender = false;

interface Bilingual<T> {
  en: T;
  zh: T;
}

interface SiteData {
  websiteName: Bilingual<string>;
  motto: Bilingual<string>;
  mottos: Bilingual<string[]>;
  brandName: Bilingual<string>;
  brandTagline: Bilingual<string>;
  about: Bilingual<string>;
}

function parseSiteYml(raw: string): SiteData {
  const out: SiteData = {
    websiteName: { en: "", zh: "" },
    motto: { en: "", zh: "" },
    mottos: { en: [], zh: [] },
    brandName: { en: "", zh: "" },
    brandTagline: { en: "", zh: "" },
    about: { en: "", zh: "" },
  };
  const lines = raw.split(/\r?\n/);
  let topKey = "" as keyof SiteData;
  let lang: "en" | "zh" | null = null;
  for (const line of lines) {
    if (line.trim() === "" || line.trim().startsWith("#")) continue;
    const top = line.match(/^([a-zA-Z]+):\s*(.*)$/);
    if (top) {
      topKey = top[1] as keyof SiteData;
      lang = null;
      continue;
    }
    const langLine = line.match(/^\s{2}(en|zh):\s*(.*)$/);
    if (langLine && topKey) {
      lang = langLine[1] as "en" | "zh";
      const inline = langLine[2].trim();
      if (inline.startsWith("[")) {
        // flow array
        const arr = parseFlowArray(inline);
        (out[topKey] as Bilingual<string[]>)[lang] = arr;
      } else if (inline) {
        (out[topKey] as Bilingual<string>)[lang] = parseScalarValue(inline);
      } else {
        (out[topKey] as Bilingual<string[]>)[lang] = [];
      }
      continue;
    }
    const item = line.match(/^\s{4}-\s+(.*)$/);
    if (item && topKey && lang) {
      const arr = (out[topKey] as Bilingual<string[]>)[lang];
      if (Array.isArray(arr)) arr.push(parseScalarValue(item[1].trim()));
    }
  }
  return out;
}

function parseFlowArray(raw: string): string[] {
  const inner = raw.slice(1, -1).trim();
  if (!inner) return [];
  const parts: string[] = [];
  let cur = "";
  let quote: string | null = null;
  for (const ch of inner) {
    if (quote) {
      cur += ch;
      if (ch === quote) quote = null;
    } else if (ch === '"' || ch === "'") {
      quote = ch;
      cur += ch;
    } else if (ch === ",") {
      parts.push(cur.trim());
      cur = "";
    } else {
      cur += ch;
    }
  }
  if (cur.trim() !== "") parts.push(cur.trim());
  return parts.map(parseScalarValue);
}

function parseScalarValue(raw: string): string {
  if (
    (raw.startsWith('"') && raw.endsWith('"')) ||
    (raw.startsWith("'") && raw.endsWith("'"))
  ) {
    if (raw.startsWith('"')) {
      try {
        return JSON.parse(raw);
      } catch {
        return raw.slice(1, -1);
      }
    }
    return raw.slice(1, -1);
  }
  return raw;
}

function serializeSiteYml(data: SiteData): string {
  const biScalar = (key: string, v: Bilingual<string>) =>
    `${key}:\n  en: ${stringifyScalar(v.en)}\n  zh: ${stringifyScalar(v.zh)}\n`;
  const biArray = (key: string, v: Bilingual<string[]>) => {
    const items = (arr: string[]) =>
      arr.length === 0
        ? " []"
        : "\n" + arr.map((s) => `    - ${stringifyScalar(s)}`).join("\n");
    return `${key}:\n  en:${items(v.en)}\n  zh:${items(v.zh)}\n`;
  };
  return [
    "# 站点元信息（双语）。在本地「可编辑模式」下修改并保存后，重新构建即可生效。\n",
    biScalar("websiteName", data.websiteName),
    biScalar("motto", data.motto),
    biArray("mottos", data.mottos),
    biScalar("brandName", data.brandName),
    biScalar("brandTagline", data.brandTagline),
    biScalar("about", data.about),
  ].join("");
}

export const GET: RequestHandler = async () => {
  guard();
  try {
    const raw = await readFile(SITE_FILE, "utf8");
    return json(parseSiteYml(raw));
  } catch {
    throw error(500, "Failed to read site.yml");
  }
};

export const PUT: RequestHandler = async ({ request }) => {
  guard();
  const body = (await request.json()) as Partial<SiteData>;
  // 合并现有值，避免缺字段时覆盖成空。
  let existing: SiteData;
  try {
    existing = parseSiteYml(await readFile(SITE_FILE, "utf8"));
  } catch {
    existing = {
      websiteName: { en: "Craig", zh: "Craig" },
      motto: { en: "", zh: "" },
      mottos: { en: [], zh: [] },
      brandName: { en: "Craig", zh: "Craig" },
      brandTagline: { en: "Personal Website", zh: "个人主页" },
      about: { en: "", zh: "" },
    };
  }
  const merged: SiteData = {
    websiteName: { ...existing.websiteName, ...(body.websiteName ?? {}) },
    motto: { ...existing.motto, ...(body.motto ?? {}) },
    mottos: {
      en: body.mottos?.en ?? existing.mottos.en,
      zh: body.mottos?.zh ?? existing.mottos.zh,
    },
    brandName: { ...existing.brandName, ...(body.brandName ?? {}) },
    brandTagline: { ...existing.brandTagline, ...(body.brandTagline ?? {}) },
    about: { ...existing.about, ...(body.about ?? {}) },
  };
  const content = serializeSiteYml(merged);
  await mkdir(dirname(SITE_FILE), { recursive: true });
  await writeFile(SITE_FILE, content, "utf8");
  return json({ ok: true });
};
