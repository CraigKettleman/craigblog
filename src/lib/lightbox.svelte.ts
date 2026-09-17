/**
 * 正文图片灯箱的全局状态。
 *
 * 与 admin-state 同款写法：模块级 $state，组件读取、action 写入。
 * 正文容器上绑 `use:lightbox` 委托点击，页面里渲染一个 <ImageLightbox />。
 */
import type { Action } from "svelte/action";

const state = $state({
  src: null as string | null,
  alt: "",
});

/** 打开前的 body overflow，关闭时还原。 */
let previousOverflow = "";

export function lightboxState() {
  return state;
}

export function closeLightbox(): void {
  state.src = null;
  state.alt = "";
  document.body.style.overflow = previousOverflow;
}

export function isLightboxOpen(): boolean {
  return state.src !== null;
}

/**
 * 正文容器上的点击委托：任意正文插图都可打开灯箱。
 * 封面/缩略图等其他图片有各自交互，这里跳过。
 */
export const lightbox: Action<HTMLElement> = (node) => {
  const onClick = (event: MouseEvent) => {
    const img = (event.target as HTMLElement | null)?.closest("img");
    if (!img || !(img instanceof HTMLImageElement)) return;
    if (!node.contains(img)) return;
    if (img.closest(".post-cover-frame, .post-thumb")) return;
    event.preventDefault();
    previousOverflow = document.body.style.overflow;
    state.src = img.currentSrc || img.src;
    state.alt = img.alt ?? "";
    document.body.style.overflow = "hidden";
  };
  node.addEventListener("click", onClick);
  return {
    destroy() {
      node.removeEventListener("click", onClick);
    },
  };
};
