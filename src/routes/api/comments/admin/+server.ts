import { error, json, type RequestHandler } from "@sveltejs/kit";
import { listAllComments } from "$lib/server/comments";
import { isAdminKey } from "$lib/server/session";

export const prerender = false;

export const GET: RequestHandler = async ({ request }) => {
  if (!isAdminKey(request.headers.get("authorization"))) {
    throw error(401, "unauthorized");
  }
  const comments = await listAllComments();
  return json({ total: comments.length, comments });
};
