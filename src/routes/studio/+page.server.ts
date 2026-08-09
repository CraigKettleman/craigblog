import { error } from "@sveltejs/kit";
import { env } from "$env/dynamic/private";
import type { PageServerLoad } from "./$types";

export const prerender = false;

export const load: PageServerLoad = () => {
  if (env.LOCAL_ADMIN !== "1") throw error(404);
};
