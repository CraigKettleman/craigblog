import { rename, rm } from "node:fs/promises";
import { sveltekit } from "@sveltejs/kit/vite";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig, type PluginOption } from "vite";

/**
 * Runs velite to turn `content/` into JSON under `.velite` (watching in dev).
 * The generated data is imported through the `#velite` alias.
 */
const VELITE_STARTED = Symbol.for("hhy.velite-started");

/**
 * 把 staging 产物原子换入正式目录。rename 在同一文件系统上是原子操作，
 * 并发读者（本机 dev server 的 SSR）只会看到旧目录或新目录，不会读到
 * 清空/半写的中间态。assets 同步换入，保证图片引用不脱节。
 */
async function swapStagingOutput() {
  for (const [staging, target] of [
    [".velite-next", ".velite"],
    ["static/blog-next", "static/blog"],
  ] as const) {
    const backup = `${target}.prev`;
    await rm(backup, { recursive: true, force: true });
    await rename(target, backup);
    try {
      await rename(staging, target);
    } catch (err) {
      // 换入失败时回滚，保留旧数据可继续服务
      await rename(backup, target);
      throw err;
    }
    await rm(backup, { recursive: true, force: true });
  }
}

function velite(): PluginOption {
  return {
    name: "velite",
    async configResolved(config) {
      // SvelteKit builds multiple environments with fresh plugin instances;
      // guard globally so velite only runs once per process.
      const globals = globalThis as Record<symbol, boolean>;
      if (globals[VELITE_STARTED]) return;
      globals[VELITE_STARTED] = true;

      const { build } = await import("velite");
      if (config.command === "build") {
        // 生产构建（含 deploy）：先在临时目录生成，完成后原子换入，
        // 避免清空 .velite 的窗口撞上并发运行的 dev server。
        process.env.VELITE_STAGING = "1";
        await build({ clean: true });
        await swapStagingOutput();
      } else {
        // dev 启动时的构建发生在服务开始前，无并发读者，直接清空重建。
        await build({ watch: true, clean: true });
      }
    },
  };
}

export default defineConfig({
  plugins: [velite(), tailwindcss(), sveltekit()],
  ssr: {
    // adapter-node externalizes production deps by default; the deploy only
    // ships `build/`, so bundle nodemailer into the SSR output instead.
    noExternal: ["nodemailer"],
  },
});
