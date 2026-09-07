/**
 * 编辑模式开关与全局保存状态（仅本地 admin 环境有意义）。
 * 默认关闭：进入 `pnpm admin` 先看到与公共版完全一致的镜像页面；
 * 开启后各页面内容原地变为可编辑，投稿入口跳转 /studio/submit 纸页面。
 *
 * 编辑模式持久化在 sessionStorage：velite 重建后 Vite 会整页刷新（page reload），
 * 不持久化的话每次保存都会把用户踢回镜像预览。
 */
const STORAGE_KEY = "hhy.admin.editMode";
/** 发布成功后的跳转交接：velite 重建会整页刷新，由刷新后的页面读取并跳转。 */
export const AFTER_PUBLISH_KEY = "hhy.admin.afterPublish";

const state = $state({
  editModeOn: false, // SSR 初始为 false；浏览器端由 initEditMode() 从 sessionStorage 恢复
  /** 保存状态机：idle → saving → saved / error，由 admin-save.ts 驱动。 */
  phase: "idle" as "idle" | "saving" | "saved" | "error",
  message: "",
});

export function getEditMode(): boolean {
  return state.editModeOn;
}

/** 浏览器端恢复编辑模式（整页刷新后保持编辑态）；仅在 onMount 等客户端时机调用。 */
export function initEditMode(): void {
  try {
    state.editModeOn = sessionStorage.getItem(STORAGE_KEY) === "1";
  } catch {
    /* sessionStorage 不可用时保持默认关闭 */
  }
}

/** 读取并清除发布后的跳转目标（一次性）；由 AdminBar 挂载时调用。 */
export function takeAfterPublish(): string | null {
  try {
    const target = sessionStorage.getItem(AFTER_PUBLISH_KEY);
    if (target) sessionStorage.removeItem(AFTER_PUBLISH_KEY);
    return target;
  } catch {
    return null;
  }
}

export function setEditMode(on: boolean): void {
  state.editModeOn = on;
  try {
    sessionStorage.setItem(STORAGE_KEY, on ? "1" : "0");
  } catch {
    /* 同上 */
  }
}

/** AdminBar 等处呈现的全局保存状态（只读快照）。 */
export function saveStatus(): { phase: "idle" | "saving" | "saved" | "error"; message: string } {
  return { phase: state.phase, message: state.message };
}

export function beginSave(): void {
  state.phase = "saving";
  state.message = "";
}

export function endSave(error?: string): void {
  if (error) {
    state.phase = "error";
    state.message = error;
  } else {
    state.phase = "saved";
    state.message = "✓ 已保存";
  }
}
