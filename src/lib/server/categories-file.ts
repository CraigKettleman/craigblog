import { readFile, readdir, writeFile, mkdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import { parseFrontMatter } from "$lib/frontmatter";

/**
 * content/categories/<section>.yml 的读写（栏目分类注册表）。
 * 文件 schema 固定：`- slug` + 双语 name + 可选双语 description 的列表，
 * 这里按该 schema 做针对性的解析/序列化，与 site.yml 的处理思路一致。
 */

export type CategorySection = "posts" | "projects";

export interface CategoryEntry {
  slug: string;
  name: { en: string; zh: string };
  description?: { en: string; zh: string };
}

const CONTENT = "content";

export function categoryFilePath(section: CategorySection): string {
  return join(CONTENT, "categories", `${section}.yml`);
}

/** 去掉 YAML 标量外围引号（JSON 双引号或单引号）。 */
function parseYamlScalar(raw: string): string {
  if (raw.startsWith('"') && raw.endsWith('"')) {
    try {
      return JSON.parse(raw) as string;
    } catch {
      return raw.slice(1, -1);
    }
  }
  if (raw.startsWith("'") && raw.endsWith("'")) return raw.slice(1, -1);
  return raw;
}

/** 输出 YAML 标量：简单值保持朴素写法（与现有文件风格一致），复杂值 JSON 引号化。 */
export function yamlScalar(value: string): string {
  if (value === "") return '""';
  const plain =
    /^[\w\u4e00-\u9fff][\w\u4e00-\u9fff\s.,()（）·—…：:、，。！？!?\-'"]*$/.test(
      value,
    ) && !/[#&*!|>%@`{}]/.test(value);
  return plain ? value : JSON.stringify(value);
}

/** 解析分类 yml（列表项：slug + name{en,zh} + 可选 description{en,zh}）。 */
export function parseCategoryYml(raw: string): CategoryEntry[] {
  const entries: CategoryEntry[] = [];
  let current: CategoryEntry | null = null;
  let field: "name" | "description" | null = null;

  for (const line of raw.split(/\r?\n/)) {
    if (line.trim() === "" || line.trim().startsWith("#")) continue;

    const item = line.match(/^-\s+slug:\s*(.*)$/);
    if (item) {
      current = { slug: parseYamlScalar(item[1].trim()), name: { en: "", zh: "" } };
      entries.push(current);
      field = null;
      continue;
    }
    const sectionField = line.match(/^\s{2}(name|description):\s*$/);
    if (sectionField && current) {
      field = sectionField[1] as "name" | "description";
      continue;
    }
    const langLine = line.match(/^\s{4}(en|zh):\s*(.*)$/);
    if (langLine && current && field) {
      const lang = langLine[1] as "en" | "zh";
      const value = parseYamlScalar(langLine[2].trim());
      if (field === "name") {
        current.name[lang] = value;
      } else {
        current.description ??= { en: "", zh: "" };
        current.description[lang] = value;
      }
    }
  }
  return entries;
}

/** 序列化分类列表为 yml（条目间空行分隔，与现有文件风格一致）。 */
export function serializeCategoryYml(entries: CategoryEntry[]): string {
  const blocks = entries.map((entry) => {
    const lines = [
      `- slug: ${yamlScalar(entry.slug)}`,
      "  name:",
      `    en: ${yamlScalar(entry.name.en ?? "")}`,
      `    zh: ${yamlScalar(entry.name.zh ?? "")}`,
    ];
    const desc = entry.description;
    if (desc && ((desc.en ?? "") || (desc.zh ?? ""))) {
      lines.push(
        "  description:",
        `    en: ${yamlScalar(desc.en ?? "")}`,
        `    zh: ${yamlScalar(desc.zh ?? "")}`,
      );
    }
    return lines.join("\n");
  });
  return blocks.join("\n\n") + "\n";
}

export async function readCategoryFile(
  section: CategorySection,
): Promise<CategoryEntry[]> {
  try {
    const raw = await readFile(categoryFilePath(section), "utf8");
    return parseCategoryYml(raw);
  } catch {
    return [];
  }
}

export async function writeCategoryFile(
  section: CategorySection,
  entries: CategoryEntry[],
): Promise<void> {
  const file = categoryFilePath(section);
  await mkdir(dirname(file), { recursive: true });
  await writeFile(file, serializeCategoryYml(entries), "utf8");
}

/** 某栏目分类被文章引用的条数（按语言），用于删除守卫与 UI 提示。 */
export async function countCategoryUsage(
  section: CategorySection,
  slug: string,
): Promise<{ en: number; zh: number }> {
  const counts = { en: 0, zh: 0 };
  let names: string[];
  try {
    names = await readdir(join(CONTENT, section));
  } catch {
    return counts;
  }
  for (const name of names) {
    for (const lang of ["en", "zh"] as const) {
      try {
        const raw = await readFile(join(CONTENT, section, name, `${lang}.md`), "utf8");
        const { data } = parseFrontMatter(raw);
        const cats = Array.isArray(data.categories) ? data.categories.map(String) : [];
        if (cats.includes(slug)) counts[lang] += 1;
      } catch {
        /* 该语言文件不存在，跳过 */
      }
    }
  }
  return counts;
}
