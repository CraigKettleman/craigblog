import { error, json, type RequestHandler } from "@sveltejs/kit";
import {
  cleanText,
  createComment,
  listThread,
  makeAuthor,
  MAX_CONTENT,
  MAX_THREAD,
  MIN_CONTENT,
  removeComment,
  verifyDeleteToken,
  voteComment,
  type SortOrder,
} from "$lib/server/comments";
import { clientIp, consumeCommentSlot, hashIp } from "$lib/server/rate-limit";
import { isAdminKey, verify } from "$lib/server/session";

export const prerender = false;

const HONEYPOT_KEY = "email_confirm";

// Guests can't pick an identity — they always post as the uniform "momo".
const GUEST_NAME = "momo";

export const GET: RequestHandler = async ({ params, url, request }) => {
  const thread = params.thread ?? "";
  if (!thread || thread.length > MAX_THREAD) throw error(400, "bad thread");
  const sort = (url.searchParams.get("sort") ?? "newest") as SortOrder;
  if (sort !== "newest" && sort !== "hot") throw error(400, "bad sort");
  const session = verify(request.headers.get("authorization"));
  return json(await listThread(thread, sort, session?.login));
};

export const POST: RequestHandler = async ({ params, request }) => {
  const thread = params.thread ?? "";
  if (!thread || thread.length > MAX_THREAD) throw error(400, "bad thread");

  const body = (await request.json().catch(() => null)) as Record<string, unknown> | null;
  if (!body || typeof body !== "object") throw error(400, "bad json");

  if (body[HONEYPOT_KEY]) {
    return json({ ok: true }, { status: 201 });
  }

  const content = typeof body.content === "string" ? cleanText(body.content, MAX_CONTENT) : "";
  if (content.length < MIN_CONTENT) throw error(400, "content too short");
  if (content.length > MAX_CONTENT) throw error(400, "content too long");

  let parentId: number | null = null;
  if (body.parentId != null) {
    const parsed = Number(body.parentId);
    if (!Number.isInteger(parsed) || parsed <= 0) throw error(400, "bad parent");
    parentId = parsed;
  }

  const session = verify(request.headers.get("authorization"));

  let author;
  if (session) {
    author = makeAuthor({
      type: session.provider,
      name: session.name || session.login,
      login: session.login,
      avatarUrl: session.avatarUrl,
    });
  } else {
    author = makeAuthor({ type: "guest", name: GUEST_NAME });
  }

  const ip = clientIp(request.headers.get("x-forwarded-for"));
  if (!consumeCommentSlot(hashIp(ip), !!session)) {
    throw error(429, "rate limited");
  }

  const { comment, deleteToken } = await createComment({
    thread,
    parentId,
    content,
    author,
    ip,
    ua: request.headers.get("user-agent") ?? undefined,
  });

  return json(
    {
      id: comment.id,
      parentId: comment.parentId,
      content: comment.content,
      createdAt: comment.createdAt,
      author: comment.author,
      upvotes: comment.upvotes,
      replies: [],
      deleteToken,
    },
    { status: 201 },
  );
};

export const DELETE: RequestHandler = async ({ params, request }) => {
  const thread = params.thread ?? "";
  if (!thread || thread.length > MAX_THREAD) throw error(400, "bad thread");

  const body = (await request.json().catch(() => null)) as Record<string, unknown> | null;
  if (!body || typeof body.id !== "number") throw error(400, "bad request");

  const id = body.id as number;

  // Admin key can delete anything
  if (isAdminKey(request.headers.get("authorization"))) {
    const ok = await removeComment(id);
    if (!ok) throw error(404, "not found");
    return json({ ok: true });
  }

  // Delete token from the comment author
  const deleteToken = typeof body.deleteToken === "string" ? body.deleteToken : "";
  if (deleteToken && (await verifyDeleteToken(id, deleteToken))) {
    const ok = await removeComment(id);
    if (!ok) throw error(404, "not found");
    return json({ ok: true });
  }

  throw error(403, "forbidden");
};

/** Vote (up/down toggle) on a comment. Requires a signed-in session. */
export const PATCH: RequestHandler = async ({ request }) => {
  const body = (await request.json().catch(() => null)) as Record<string, unknown> | null;
  if (!body || typeof body.commentId !== "number") throw error(400, "bad request");
  const action = body.action;
  if (action !== "up" && action !== "down") throw error(400, "bad action");

  const session = verify(request.headers.get("authorization"));
  if (!session) throw error(401, "sign in required");

  const result = await voteComment(body.commentId as number, session.login, action);
  if (result === null) throw error(404, "not found");

  return json(result);
};
