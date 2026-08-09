import { error, json, type RequestHandler } from "@sveltejs/kit";
import { removeComment, updateComment } from "$lib/server/comments";
import { isAdminKey } from "$lib/server/session";

export const prerender = false;

function requireAdmin(request: Request): void {
  if (!isAdminKey(request.headers.get("authorization"))) {
    throw error(401, "unauthorized");
  }
}

export const DELETE: RequestHandler = async ({ request, params }) => {
  requireAdmin(request);
  const id = Number(params.id);
  if (!Number.isInteger(id) || id <= 0) throw error(400, "bad id");
  const removed = await removeComment(id);
  if (!removed) throw error(404, "not found");
  return json({ ok: true });
};

export const PATCH: RequestHandler = async ({ request, params }) => {
  requireAdmin(request);
  const id = Number(params.id);
  if (!Number.isInteger(id) || id <= 0) throw error(400, "bad id");

  const body = (await request.json().catch(() => null)) as Record<string, unknown> | null;
  if (!body || typeof body !== "object") throw error(400, "bad json");

  const patch: { status?: "published" | "hidden"; content?: string } = {};
  if (body.status === "published" || body.status === "hidden") patch.status = body.status;
  if (typeof body.content === "string") patch.content = body.content;
  if (!patch.status && !patch.content) throw error(400, "nothing to update");

  const comment = await updateComment(id, patch);
  if (!comment) throw error(404, "not found");
  return json({ ok: true, comment });
};
