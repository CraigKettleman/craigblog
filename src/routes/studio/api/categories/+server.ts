import { error, json, type RequestHandler } from "@sveltejs/kit";
import { env } from "$env/dynamic/private";
import { categories } from "#velite";

function guard() {
  if (env.LOCAL_ADMIN !== "1") throw error(404);
}

export const prerender = false;

export const GET: RequestHandler = () => {
  guard();
  return json(
    categories.map((c) => ({
      slug: c.slug,
      section: c.section,
      name: c.name,
      description: c.description,
    })),
  );
};
