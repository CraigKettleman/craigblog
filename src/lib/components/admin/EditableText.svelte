<script lang="ts">
  /**
   * 单行「原地编辑」组件：
   * - 非编辑态输出与公共版一致的纯文本（排版样式由父元素继承）；
   * - 编辑态原位变为透明输入框（仅保留虚线下划线提示可编辑），
   *   Enter / 失焦保存，Esc 还原；值未变化时不触发保存。
   */
  let {
    editing,
    value,
    display,
    onsave,
    class: cls = "",
    inputClass = "",
    type = "text",
    placeholder = "",
    maxlength,
  }: {
    editing: boolean;
    value: string;
    /** 非编辑态展示文本（如格式化日期），默认展示 value 本身。 */
    display?: string;
    /** 保存回调；失败可抛错（输入框保留用户输入，错误经全局状态呈现）。 */
    onsave: (v: string) => void | Promise<void>;
    class?: string;
    /** 编辑态附加样式（如日期/Slug 的 mono 字体与固定宽度）。 */
    inputClass?: string;
    type?: string;
    placeholder?: string;
    maxlength?: number;
  } = $props();

  function commit(e: Event) {
    const input = e.currentTarget as HTMLInputElement;
    if (input.value === value) return;
    onsave(input.value);
  }

  function onkeydown(e: KeyboardEvent) {
    const input = e.currentTarget as HTMLInputElement;
    if (e.key === "Enter") {
      e.preventDefault();
      input.blur();
    } else if (e.key === "Escape") {
      input.value = value;
      input.blur();
    }
  }
</script>

{#if editing}
  <input
    {type}
    {value}
    {placeholder}
    maxlength={maxlength}
    onblur={commit}
    onkeydown={onkeydown}
    class="w-full min-w-0 -mb-px rounded-none border-b border-dashed border-printer-accent/60 bg-transparent px-0.5 outline-none focus:border-printer-accent {inputClass}"
  />
{:else}
  <span class={cls}>{display ?? value}</span>
{/if}
