import { env } from "$env/dynamic/private";
import type { LayoutServerLoad } from "./$types";

/**
 * 把 `LOCAL_ADMIN=1` 标志透传给客户端。
 * 仅在本地 `pnpm admin` 时为 true；生产构建（pnpm build）时为 false，
 * 因此预渲染产物中不会包含任何编辑界面（公网绝不暴露）。
 */
export const load: LayoutServerLoad = () => ({
  admin: env.LOCAL_ADMIN === "1",
});
