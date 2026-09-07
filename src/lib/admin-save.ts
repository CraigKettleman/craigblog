/**
 * 「可编辑模式」统一保存链路：
 *   fetch 写 content/ 源文件 → 等 velite watch 重建 → invalidateAll 刷新页面渲染。
 * 失败时把原因写进全局保存状态（AdminBar 呈现）并抛出，由调用方决定是否保留草稿值。
 */
import { goto, invalidateAll } from "$app/navigation";
import {
  AFTER_PUBLISH_KEY,
  beginSave,
  endSave,
} from "./admin-state.svelte";

/** velite watch 增量重建的经验等待时间。 */
const VELITE_REBUILD_MS = 400;

/** 发起 JSON 请求并完成「等待重建 + 刷新」流程；invalidate=false 适用于保存后跳转新地址的场景。 */
export async function saveJson(
  url: string,
  method: "PUT" | "POST" | "DELETE",
  body: unknown,
  options: { invalidate?: boolean } = {},
): Promise<Response> {
  beginSave();
  try {
    const res = await fetch(url, {
      method,
      headers: { "Content-Type": "application/json" },
      body: body === null ? undefined : JSON.stringify(body),
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      // SvelteKit error(…) 响应体是 {"message":"…"}，提取纯文本便于直接展示
      let message = text;
      try {
        const parsed = JSON.parse(text) as { message?: unknown };
        if (parsed && typeof parsed.message === "string") message = parsed.message;
      } catch {
        /* 非 JSON 则原样使用 */
      }
      throw new Error(message || `HTTP ${res.status}`);
    }
    // 等 velite watch 重建后再重跑 load，页面直接呈现新内容
    await new Promise((r) => setTimeout(r, VELITE_REBUILD_MS));
    if (options.invalidate !== false) await invalidateAll();
    endSave();
    return res;
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    endSave(message);
    throw e;
  }
}

/** 文章目录上的 PUT（双语无损合并），最常用的保存入口。 */
export function savePost(
  path: string,
  patch: Record<string, unknown>,
): Promise<Response> {
  return saveJson(`/studio/api/posts/${path}`, "PUT", patch);
}

/** site.yml 的 PUT（按语言字段合并）。 */
export function saveSite(patch: Record<string, unknown>): Promise<Response> {
  return saveJson("/studio/api/site", "PUT", patch);
}

/**
 * 保存后跳转到（新）地址：slug 更名、删除文章、投稿发布都会改变当前 URL。
 * velite 重建通常会触发 Vite 整页刷新并杀掉当前 JS 上下文，因此先把目标登记到
 * sessionStorage（刷新后的页面由 AdminBar 挂载时读取并跳转）；若本上下文存活，
 * 则等重建完成后自行跳转并清除登记。保存失败（无重建、无刷新）时清除登记并抛错。
 */
export async function saveAndNavigate(
  target: string,
  doSave: () => Promise<unknown>,
): Promise<void> {
  beginSave();
  try {
    sessionStorage.setItem(AFTER_PUBLISH_KEY, target);
  } catch {
    /* 存储不可用时退化为仅直接跳转 */
  }
  try {
    await doSave();
    // 等 velite watch 重建，避免新地址在重建完成前被打开而 404
    await new Promise((r) => setTimeout(r, VELITE_REBUILD_MS));
    try {
      sessionStorage.removeItem(AFTER_PUBLISH_KEY);
    } catch {
      /* 同上 */
    }
    await goto(target);
    endSave();
  } catch (e) {
    try {
      sessionStorage.removeItem(AFTER_PUBLISH_KEY);
    } catch {
      /* 同上 */
    }
    const message = e instanceof Error ? e.message : String(e);
    endSave(message);
    throw e;
  }
}
