/**
 * 编辑模式开关（仅本地 admin 环境有意义）。
 * 默认关闭：进入 `pnpm admin` 先看到与公共版完全一致的镜像页面；
 * 开启后才会出现编辑手柄、投稿按钮等编辑能力。
 */
const state = $state({ editModeOn: false });

export function getEditMode(): boolean {
  return state.editModeOn;
}

export function setEditMode(on: boolean): void {
  state.editModeOn = on;
}
