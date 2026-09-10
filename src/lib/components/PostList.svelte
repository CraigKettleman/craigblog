<script lang="ts">
  import type { PostListItem } from "$lib/content";
  import { displayDate } from "$lib/date";
  import DitherImage from "$lib/components/DitherImage.svelte";

  let {
    posts,
    lang,
    compact = false,
    editable = false,
  }: {
    posts: PostListItem[];
    lang: string;
    /** Tighter rows without separators (used on the home page). */
    compact?: boolean;
    /** 可编辑模式：为草稿条目显示标记。 */
    editable?: boolean;
  } = $props();

  // 入场动效：delay 从列表中心向两端扩散（复刻 cali.so 的 enter-swing 节奏）
  const enterDelay = (index: number) =>
    `${120 + Math.round(Math.abs(index - (posts.length - 1) / 2) * 50)}ms`;
</script>

<div class={["flex flex-col", compact && "gap-3"]}>
  {#each posts as post, index (post.permalink)}
    <div class="post-row-enter" style:--enter-delay={enterDelay(index)}>
      <a href={post.permalink} class="group block">
        <div
          class={[
            "-mx-2 px-2 rounded-md hover:bg-printer-ink/3 dark:hover:bg-printer-ink-dark/3 transition-colors",
            compact ? "py-2" : "py-3",
          ]}
        >
          <div class="flex items-center gap-3">
            <!-- 左：印刷纸堆缩略图（无封面时为网点纸块占位） -->
            <div class="post-thumb shrink-0" aria-hidden="true">
              <div class="post-thumb-sheet post-thumb-sheet-back"></div>
              <div class="post-thumb-sheet post-thumb-sheet-front"></div>
              {#if post.cover}
                <!-- 源站同款：canvas 拜耳抖动像素画，hover 显影成真图 -->
                <DitherImage src={post.cover.src} />
              {:else}
                <div class="post-thumb-blank"></div>
              {/if}
            </div>

            <h3
              class="min-w-0 flex-1 truncate font-serif text-base text-printer-ink dark:text-printer-ink-dark group-hover:text-printer-accent dark:group-hover:text-printer-accent-dark transition-colors leading-snug"
            >
              {post.title}
            </h3>

            <!-- 点线引导线（窄屏隐藏，靠 truncate 收尾） -->
            <div class="post-leader hidden md:block" aria-hidden="true"></div>

            <span
              class="shrink-0 font-mono text-[10px] text-printer-ink-light dark:text-printer-ink-dark/40 tabular-nums"
            >
              {displayDate(post.date, lang)}
              {#if editable && post.draft}
                <span class="text-printer-accent dark:text-printer-accent-dark">· 草稿</span>
              {/if}
            </span>
          </div>
          {#if post.description && !compact}
            <p
              class="pl-[4.75rem] font-serif text-xs text-printer-ink-light dark:text-printer-ink-dark/40 mt-1 line-clamp-1"
            >
              {post.description}
            </p>
          {/if}
        </div>
      </a>
      {#if !compact && index < posts.length - 1}
        <div
          class="border-b border-dotted border-printer-ink/5 dark:border-printer-ink-dark/5"
        ></div>
      {/if}
    </div>
  {/each}
</div>
