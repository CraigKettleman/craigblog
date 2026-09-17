/**
 * 极简 HTML 解析器：只服务 LaTeXML 产出（格式良好、无裸 `<`、无脚本注入），
 * 不追求 HTML5 容错。输出节点树，保留属性与文本原样，供结构改写使用。
 */

const VOID_ELEMENTS = new Set([
  "area",
  "base",
  "br",
  "col",
  "embed",
  "hr",
  "img",
  "input",
  "link",
  "meta",
  "param",
  "source",
  "track",
  "wbr",
]);

/** @typedef {{ type:'element', name:string, attrs:Record<string,string>, children:Node[], parent:any }} ElementNode */
/** @typedef {{ type:'text', text:string, parent:any }} TextNode */
/** @typedef {ElementNode|TextNode} Node */

const ENTITIES = {
  amp: "&",
  lt: "<",
  gt: ">",
  quot: '"',
  apos: "'",
  nbsp: "\u00a0",
  mdash: "—",
  ndash: "–",
  hellip: "…",
  times: "×",
  rarr: "→",
  larr: "←",
  harr: "↔",
  sim: "∼",
  le: "≤",
  ge: "≥",
  ne: "≠",
  deg: "°",
  prime: "′",
  ldquo: "“",
  rdquo: "”",
  lsquo: "‘",
  rsquo: "’",
};

export function decodeEntities(text) {
  return text.replace(/&(#x?[0-9a-fA-F]+|[a-zA-Z][a-zA-Z0-9]*);/g, (m, body) => {
    if (body[0] === "#") {
      const code =
        body[1] === "x" || body[1] === "X"
          ? Number.parseInt(body.slice(2), 16)
          : Number.parseInt(body.slice(1), 10);
      if (Number.isFinite(code)) {
        try {
          return String.fromCodePoint(code);
        } catch {
          return m;
        }
      }
      return m;
    }
    return ENTITIES[body] ?? m;
  });
}

export function escapeText(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

export function escapeAttr(text) {
  return escapeText(text).replace(/"/g, "&quot;");
}

function parseAttrs(source) {
  const attrs = {};
  const re = /([:@a-zA-Z_][-.:\w]*)\s*(?:=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+)))?/g;
  let m;
  while ((m = re.exec(source))) {
    const raw = m[2] ?? m[3] ?? m[4] ?? "";
    attrs[m[1]] = decodeEntities(raw);
  }
  return attrs;
}

/** 解析完整文档，返回根节点。 */
export function parseHtml(html) {
  const root = { type: "element", name: "#root", attrs: {}, children: [], parent: null };
  const stack = [root];
  let i = 0;

  const push = (node) => {
    const parent = stack[stack.length - 1];
    node.parent = parent;
    parent.children.push(node);
  };

  while (i < html.length) {
    const lt = html.indexOf("<", i);
    if (lt < 0) {
      push({ type: "text", text: decodeEntities(html.slice(i)), parent: null });
      break;
    }
    if (lt > i) {
      const raw = html.slice(i, lt);
      if (raw.trim() !== "" || /[^\s]/.test(raw)) {
        push({ type: "text", text: decodeEntities(raw), parent: null });
      }
    }
    // 注释 / DOCTYPE / CDATA
    if (html.startsWith("<!--", lt)) {
      const end = html.indexOf("-->", lt);
      i = end < 0 ? html.length : end + 3;
      continue;
    }
    if (html.startsWith("<!", lt) || html.startsWith("<?", lt)) {
      const end = html.indexOf(">", lt);
      i = end < 0 ? html.length : end + 1;
      continue;
    }
    if (html.startsWith("</", lt)) {
      const end = html.indexOf(">", lt);
      const name = html.slice(lt + 2, end).trim().toLowerCase();
      for (let depth = stack.length - 1; depth > 0; depth--) {
        if (stack[depth].name === name) {
          stack.length = depth;
          break;
        }
      }
      i = end + 1;
      continue;
    }
    // 开始标签
    const end = findTagEnd(html, lt);
    const inner = html.slice(lt + 1, end);
    const selfClosing = inner.endsWith("/");
    const body = selfClosing ? inner.slice(0, -1) : inner;
    const nameMatch = /^([a-zA-Z][-\w:]*)/.exec(body);
    if (!nameMatch) {
      i = end + 1;
      continue;
    }
    const name = nameMatch[1].toLowerCase();
    const node = {
      type: "element",
      name,
      attrs: parseAttrs(body.slice(nameMatch[1].length)),
      children: [],
      parent: null,
    };
    push(node);
    if (!selfClosing && !VOID_ELEMENTS.has(name)) stack.push(node);
    i = end + 1;
  }

  return root.children.find((n) => n.type === "element") ?? root;
}

/** 找到与 `<` 配对的 `>`，跳过引号内的内容。 */
function findTagEnd(html, start) {
  let quote = null;
  for (let i = start + 1; i < html.length; i++) {
    const ch = html[i];
    if (quote) {
      if (ch === quote) quote = null;
      continue;
    }
    if (ch === '"' || ch === "'") {
      quote = ch;
      continue;
    }
    if (ch === ">") return i;
  }
  return html.length;
}

// ---------------------------------------------------------------------------
// 遍历辅助
// ---------------------------------------------------------------------------

export function isElement(node) {
  return !!node && node.type === "element";
}

export function classes(node) {
  return isElement(node) ? (node.attrs.class ?? "").split(/\s+/).filter(Boolean) : [];
}

export function hasClass(node, name) {
  return classes(node).includes(name);
}

export function hasAnyClass(node, names) {
  const set = classes(node);
  return names.some((n) => set.includes(n));
}

export function children(node) {
  return node.children.filter(isElement);
}

export function walk(node, visit) {
  for (const child of node.children) {
    if (!isElement(child)) continue;
    if (visit(child) === false) continue;
    walk(child, visit);
  }
}

export function findFirst(node, predicate) {
  for (const child of node.children) {
    if (!isElement(child)) continue;
    if (predicate(child)) return child;
    const found = findFirst(child, predicate);
    if (found) return found;
  }
  return null;
}

export function findAll(node, predicate, out = []) {
  for (const child of node.children) {
    if (!isElement(child)) continue;
    if (predicate(child)) out.push(child);
    findAll(child, predicate, out);
  }
  return out;
}

/** 收集元素下的所有文本（不含子元素自身的标签），保留原始空白。 */
export function textOf(node) {
  if (!node) return "";
  if (node.type === "text") return node.text;
  return node.children.map(textOf).join("");
}

/**
 * 面向阅读的纯文本：跳过 MathML 的 TeX 注解（<annotation>），
 * 只取公式的呈现文本（如「→」），避免简介里混进 \rightarrow 这类源码。
 */
export function readableText(node) {
  if (!node) return "";
  if (node.type === "text") return node.text;
  if (node.name === "annotation") return "";
  return node.children.map(readableText).join("");
}

/** 跳过 TeX 注解并归一化空白后的文本。 */
export function normReadableText(node) {
  return readableText(node).replace(/\s+/g, " ").trim();
}

/** 归一化空白后的文本。 */
export function normText(node) {
  return textOf(node).replace(/\s+/g, " ").trim();
}

/** 克隆节点（复制属性与文本，父子指针重建）。 */
export function clone(node) {
  if (node.type === "text") return { type: "text", text: node.text, parent: null };
  return {
    type: "element",
    name: node.name,
    attrs: { ...node.attrs },
    children: node.children.map(clone),
    parent: null,
  };
}

/** 输出节点为 HTML 字符串（自闭合 void 元素）。 */
export function serializeHtml(node) {
  if (node.type === "text") return escapeText(node.text);
  const attrs = Object.entries(node.attrs)
    .filter(([, v]) => v !== undefined && v !== null)
    .map(([k, v]) => (v === "" ? ` ${k}` : ` ${k}="${escapeAttr(v)}"`))
    .join("");
  if (VOID_ELEMENTS.has(node.name)) return `<${node.name}${attrs}>`;
  return `<${node.name}${attrs}>${node.children.map(serializeHtml).join("")}</${node.name}>`;
}
