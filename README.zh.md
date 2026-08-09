## 我的主页

[English](./README.md)

这里是我个人主页的源码。

## 技术栈

- [Svelte](https://svelte.dev) / [SvelteKit](https://svelte.dev/docs/kit)
- [TailwindCSS](https://tailwindcss.com)
- [Velite](https://velite.js.org)
- VPS 上的 [Node.js](https://nodejs.org)（通过 `@sveltejs/adapter-node`）

整个站点在构建期预渲染。一个轻量 Node 服务负责无语言前缀 URL 的
`Accept-Language` 重定向、无扩展名的 RSS 订阅源，以及 `/api/social` 统计接口。
部署由 `deploy/deploy.sh` 自动化（rsync → systemd → nginx + Let's Encrypt）。

## 内容

两个博客共用同一套内容管道（Velite）：

- `content/posts/**` —— 技术文章，对应 `/{lang}/posts`
- `content/life-posts/**` —— 生活文章，对应 `/{lang}/life`
- `content/categories/{posts,life}.yml` —— 两个板块各自的分类

## 开发与部署

- `pnpm dev`：启动本地开发服务器。
- `pnpm check`：类型检查。
- `pnpm build`：构建站点（包含预渲染页面与 OG 图片）。
- `pnpm preview`：构建并在本地运行 Node 产物（`node build/index.js`）。
- `pnpm deploy`：构建并部署到 VPS（`deploy/deploy.sh`：rsync → systemd → nginx + Let's Encrypt）。
