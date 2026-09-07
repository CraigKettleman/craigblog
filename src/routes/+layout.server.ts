import { env } from "$env/dynamic/private";
import { dev } from "$app/environment";
import type { LayoutServerLoad } from "./$types";

/**
 * 把 `LOCAL_ADMIN=1` 标志透传给客户端。
 *
 * 仅在本地 dev（pnpm admin）时为 true。dev 在构建时同样可能带着 LOCAL_ADMIN
 * （如经 dev server 触发的 deploy.sh 里的 pnpm build 继承了该变量），
 * 因此这里额外用 `dev` 双保险：任何生产构建/运行时恒为 false，
 * 预渲染产物绝不包含编辑界面（公网绝不暴露）。
 */
export const load: LayoutServerLoad = () => ({
  admin: dev && env.LOCAL_ADMIN === "1",
});
