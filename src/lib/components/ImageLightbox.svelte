<script lang="ts">
  /**
   * 正文图片灯箱的浮层。
   *
   * 状态与点击委托在 $lib/lightbox.svelte.ts；页面只需在正文旁渲染本组件，
   * 并给正文容器绑 `use:lightbox`。Esc / 点击空白 / 关闭按钮均可关闭。
   *
   * 背景用 <button> 承担「点击关闭」，图片本身不再挂事件，
   * 这样既无 a11y 告警，也不会因点图片而误关。
   */
  import { closeLightbox, lightboxState } from "$lib/lightbox.svelte";

  let state = $derived(lightboxState());

  function onKeydown(event: KeyboardEvent) {
    if (event.key === "Escape" && state.src) closeLightbox();
  }
</script>

<svelte:window onkeydown={onKeydown} />

{#if state.src}
  <div class="post-lightbox" role="dialog" aria-modal="true" aria-label={state.alt || "图片预览"}>
    <button
      type="button"
      class="post-lightbox-backdrop"
      aria-label="关闭"
      onclick={closeLightbox}
    ></button>
    <img src={state.src} alt={state.alt} />
    <button
      type="button"
      class="post-lightbox-close"
      aria-label="关闭"
      onclick={closeLightbox}
    >
      ×
    </button>
  </div>
{/if}
