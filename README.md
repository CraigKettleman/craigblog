## My Homepage

[中文](./README.zh.md)

This is the source code of my personal homepage.

## Stack

- [Svelte](https://svelte.dev) / [SvelteKit](https://svelte.dev/docs/kit)
- [TailwindCSS](https://tailwindcss.com)
- [Velite](https://velite.js.org)
- [Node.js](https://nodejs.org) on a VPS via `@sveltejs/adapter-node`

The whole site is prerendered at build time. A small Node server handles the
`Accept-Language` redirect for locale-less URLs, the extensionless RSS feeds,
and the `/api/social` stats endpoint. Deployment is automated by
`deploy/deploy.sh` (rsync → systemd → nginx + Let's Encrypt).

## Content

Both blogs share one content pipeline (Velite):

- `content/posts/**` — tech posts, served under `/{lang}/posts`
- `content/life-posts/**` — life posts, served under `/{lang}/life`
- `content/categories/{posts,life}.yml` — categories for each section

## Develop & Deploy

- `pnpm dev`: Start the local dev server.
- `pnpm check`: Type-check the project.
- `pnpm build`: Build the site (including prerendered pages and OG images).
- `pnpm preview`: Build and run the Node build locally (`node build/index.js`).
- `pnpm deploy`: Build and deploy to the VPS via `deploy/deploy.sh`
  (rsync → systemd → nginx + Let's Encrypt).
