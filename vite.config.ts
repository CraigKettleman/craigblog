import { rename, rm, symlink, readlink, lstat } from "node:fs/promises";
import path from "node:path";
import { sveltekit } from "@sveltejs/kit/vite";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig, type PluginOption } from "vite";

/**
 * Runs velite to turn `content/` into JSON under `.velite` (watching in dev).
 * The generated data is imported through the `#velite` alias.
 */
const VELITE_STARTED = Symbol.for("hhy.velite-started");

/**
 * 把 staging 产物原子换入正式目录。
 *
 * 直接 rename 目录做不到原子换入：目标目录必须先让位（.velite -> .velite.prev），
 * 在两次 rename 之间 .velite 是不存在的，并发读者（本机 dev server 的 SSR）恰好
 * 在这一瞬间解析 #velite 就会撞上「file does not exist」。
 *
 * 改为符号链接：数据放在 A/B 两个真实目录里轮流使用，.velite 是指向当前那个的
 * 软链接。换入时先把新数据放进本轮不用的那个目录，再建一个指向它的临时软链接，
 * 用 rename 原子覆盖到正式链接上。读者只会看到旧目标或新目标，不存在「链接缺失」
 * 的中间态，也不会读到 Velite 正在清理的那个目录。
 */
const DATA_LINK = ".velite";
const ASSET_LINK = "static/blog";

/** 求出某个链接下一轮应该写入的槽位（当前没在用的那个）。 */
async function inactiveSlot(link: string): Promise<string> {
  const [slotA, slotB] = [`${link}-a`, `${link}-b`];
  const stat = await lstat(link).catch(() => null);
  if (stat?.isSymbolicLink()) {
    // 链接里存的是相对自身目录的路径，归一化后再比对
    const target = path.resolve(path.dirname(link), await readlink(link));
    if (target === path.resolve(slotA)) return slotB;
  }
  return slotA;
}

/**
 * 把某个槽位切为当前数据目录。rename 覆盖软链接是单次原子操作，
 * 并发读者（本机 dev server 的 SSR）只会看到旧目标或新目标，
 * 不存在「链接缺失」的中间态。
 *
 * slot 是相对仓库根的路径（如 static/blog-b）或绝对路径，而软链接里的目标要相对
 * 链接所在目录解析，所以这里先算成相对 link 目录的路径。
 */
async function flipLink(link: string, slot: string) {
  const stat = await lstat(link).catch(() => null);
  // 旧版布局里 link 是真目录：它装的是上一份数据，新数据已在 slot 里，整开即可
  if (stat && !stat.isSymbolicLink()) {
    await rm(link, { recursive: true, force: true });
  }
  const dir = path.dirname(link);
  const target = path.relative(dir, slot) || path.basename(slot);
  const temp = `${link}.swap`;
  await rm(temp, { force: true });
  await symlink(target, temp);
  await rename(temp, link);
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
        // 生产构建（含 deploy）：先写进本轮不用的槽位，构建成功后原子换入，
        // 并发运行的 dev server 全程读到的都是完整的旧/新数据。
        const dataSlot = await inactiveSlot(DATA_LINK);
        const assetSlot = await inactiveSlot(ASSET_LINK);
        process.env.VELITE_DATA_SLOT = dataSlot;
        process.env.VELITE_ASSET_SLOT = assetSlot;
        await build({ clean: true });
        await flipLink(DATA_LINK, dataSlot);
        await flipLink(ASSET_LINK, assetSlot);
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
