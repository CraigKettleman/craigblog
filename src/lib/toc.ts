/** 文章目录条目（由渲染后的 HTML 抽取）。 */
export interface TocItem {
  id: string;
  text: string;
  depth: 2 | 3;
}

/**
 * 从渲染后的 HTML 中抽取 h2/h3 生成目录。
 * 标题 id 由 velite 管线里的 rehype-slug 注入；
 * 先剔除 <pre> 代码块，避免代码里的伪标题被误收。
 */
export function extractToc(html: string): TocItem[] {
  const items: TocItem[] = [];
  const cleaned = html.replace(/<pre[\s\S]*?<\/pre>/g, "");
  const headingRe = /<h([23])\b[^>]*\bid="([^"]+)"[^>]*>([\s\S]*?)<\/h\1>/g;
  for (const match of cleaned.matchAll(headingRe)) {
    const depth = Number(match[1]) as 2 | 3;
    const text = match[3]
      .replace(/<[^>]+>/g, "")
      .replace(/&amp;/g, "&")
      .replace(/&lt;/g, "<")
      .replace(/&gt;/g, ">")
      .replace(/&quot;/g, '"')
      .trim();
    if (text) items.push({ id: match[2], text, depth });
  }
  return items;
}
