import { createHash, randomBytes } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { env } from "$env/dynamic/private";
import { hashIp } from "./rate-limit";

export const MAX_NAME = 50;
export const MAX_CONTENT = 2000;
export const MAX_WEBSITE = 200;
export const MAX_THREAD = 200;
export const MIN_CONTENT = 2;

export type CommentAuthorType = "guest" | "github" | "account";

export interface CommentAuthor {
  type: CommentAuthorType;
  name: string;
  website?: string;
  login?: string;
  avatarUrl?: string | null;
  isOwner: boolean;
}

export interface CommentRecord {
  id: number;
  thread: string;
  parentId: number | null;
  content: string;
  status: "published" | "hidden";
  createdAt: string;
  ipHash: string;
  ua?: string;
  author: CommentAuthor;
  deleteTokenHash?: string;
  upvotes: number;
  upvoters: string[];
  downvotes: number;
  downvoters: string[];
}

export interface PublicComment {
  id: number;
  parentId: number | null;
  parentName?: string;
  content: string;
  createdAt: string;
  author: CommentAuthor;
  replies: PublicComment[];
  upvotes: number;
  downvotes: number;
  /** "up" | "down" | null — the current viewer's vote, when authed. */
  userVote?: "up" | "down" | null;
}

interface CommentDb {
  version: number;
  nextId: number;
  comments: CommentRecord[];
}

function dataFile(): string {
  return env.COMMENTS_FILE || "/var/www/hhy.homes/data/comments.json";
}

const OWNER_LOGIN = (env.OWNER_GITHUB_LOGIN ?? "CraigKEttleman").toLowerCase();

export function isOwnerLogin(login: string | undefined): boolean {
  return !!login && login.toLowerCase() === OWNER_LOGIN;
}

export function makeAuthor(input: {
  type: CommentAuthorType;
  name: string;
  website?: string;
  login?: string;
  avatarUrl?: string | null;
}): CommentAuthor {
  const author: CommentAuthor = {
    type: input.type,
    name: input.name,
    isOwner: (input.type === "github" || input.type === "account") && isOwnerLogin(input.login),
  };
  if (input.website) author.website = input.website;
  if (input.login) author.login = input.login;
  if (input.avatarUrl) author.avatarUrl = input.avatarUrl;
  return author;
}

export function cleanText(input: string, max: number): string {
  return input.trim().slice(0, max);
}

export function isValidWebsite(url: string): boolean {
  try {
    const parsed = new URL(url);
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch {
    return false;
  }
}

let db: CommentDb | null = null;
let writeQueue: Promise<void> = Promise.resolve();

async function load(): Promise<CommentDb> {
  if (db) return db;
  const file = dataFile();
  try {
    const raw = await readFile(file, "utf8");
    const parsed = JSON.parse(raw) as CommentDb;
    db = {
      version: 1,
      nextId: typeof parsed.nextId === "number" ? parsed.nextId : 1,
      comments: Array.isArray(parsed.comments)
        ? parsed.comments.map((c) => {
            const old = c as any;
            const upvoters: string[] = Array.isArray(old.upvoters)
              ? old.upvoters
              : Array.isArray(old.upvoteIps)
                ? old.upvoteIps
                : [];
            return {
              ...c,
              upvotes: typeof old.upvotes === "number" ? old.upvotes : 0,
              upvoters,
              downvotes: typeof old.downvotes === "number" ? old.downvotes : 0,
              downvoters: Array.isArray(old.downvoters) ? old.downvoters : [],
            };
          })
        : [],
    };
  } catch {
    db = { version: 1, nextId: 1, comments: [] };
  }
  return db;
}

function enqueue(fn: () => Promise<void>): Promise<void> {
  const run = writeQueue.then(fn, fn);
  writeQueue = run.catch(() => {});
  return run;
}

async function save(): Promise<void> {
  if (!db) return;
  const file = dataFile();
  const tmp = `${file}.tmp`;
  await mkdir(dirname(file), { recursive: true });
  await writeFile(tmp, JSON.stringify(db, null, 2), "utf8");
  await rename(tmp, file);
}

function toPublic(record: CommentRecord): Omit<PublicComment, "replies" | "userVote"> {
  return {
    id: record.id,
    parentId: record.parentId,
    content: record.content,
    createdAt: record.createdAt,
    author: record.author,
    upvotes: record.upvotes,
    downvotes: record.downvotes,
  };
}

/** Resolve a viewer's vote for a record, given their login (lowercased). */
function voteOf(record: CommentRecord, login?: string): "up" | "down" | null {
  if (!login) return null;
  const key = login.toLowerCase();
  if (record.upvoters.some((v) => v.toLowerCase() === key)) return "up";
  if (record.downvoters.some((v) => v.toLowerCase() === key)) return "down";
  return null;
}

export type SortOrder = "newest" | "hot";

export async function listThread(
  thread: string,
  sort: SortOrder = "newest",
  login?: string,
): Promise<{ count: number; comments: PublicComment[] }> {
  const data = await load();
  const items = data.comments
    .filter((c) => c.thread === thread && c.status === "published")
    .sort((a, b) => (a.createdAt < b.createdAt ? -1 : 1));

  const byId = new Map(items.map((c) => [c.id, c]));

  // Build reply counts per top-level comment
  const replyCount = new Map<number, number>();
  for (const c of items) {
    if (c.parentId == null) continue;
    let root = c;
    for (let guard = 0; guard < 10; guard++) {
      const parent = byId.get(root.parentId ?? -1);
      if (!parent) break;
      root = parent;
      if (parent.parentId == null) break;
    }
    replyCount.set(root.id, (replyCount.get(root.id) ?? 0) + 1);
  }

  const repliesByRoot = new Map<number, PublicComment[]>();
  for (const c of items) {
    if (c.parentId == null) continue;
    let root = c;
    for (let guard = 0; guard < 10; guard++) {
      const parent = byId.get(root.parentId ?? -1);
      if (!parent) break;
      root = parent;
      if (parent.parentId == null) break;
    }
    const bucket = repliesByRoot.get(root.id) ?? [];
    bucket.push({
      ...toPublic(c),
      parentName: byId.get(c.parentId)?.author.name,
      userVote: voteOf(c, login),
      replies: [],
    });
    repliesByRoot.set(root.id, bucket);
  }

  let topLevel = items
    .filter((c) => c.parentId == null)
    .map((c) => ({
      ...toPublic(c),
      userVote: voteOf(c, login),
      replies: repliesByRoot.get(c.id) ?? [],
    }));

  if (sort === "hot") {
    topLevel.sort((a, b) => {
      const ra = replyCount.get(a.id) ?? 0;
      const rb = replyCount.get(b.id) ?? 0;
      if (ra !== rb) return rb - ra;
      return a.createdAt < b.createdAt ? 1 : -1;
    });
  }

  return { count: items.length, comments: topLevel };
}

export interface NewCommentInput {
  thread: string;
  parentId: number | null;
  content: string;
  author: CommentAuthor;
  ip: string;
  ua?: string;
}

export async function createComment(
  input: NewCommentInput,
): Promise<{ comment: CommentRecord; deleteToken: string }> {
  const deleteToken = randomBytes(32).toString("base64url");
  const deleteTokenHash = createHash("sha256").update(deleteToken).digest("hex");

  let created: CommentRecord | null = null;
  await enqueue(async () => {
    const data = await load();
    const comment: CommentRecord = {
      id: data.nextId,
      thread: input.thread,
      parentId: input.parentId,
      content: input.content,
      status: "published",
      createdAt: new Date().toISOString(),
      ipHash: hashIp(input.ip),
      ua: input.ua ? input.ua.slice(0, 200) : undefined,
      author: input.author,
      deleteTokenHash,
      upvotes: 0,
      upvoters: [],
      downvotes: 0,
      downvoters: [],
    };
    data.nextId += 1;
    data.comments.push(comment);
    await save();
    created = comment;
  });
  if (!created) throw new Error("failed to create comment");
  return { comment: created, deleteToken };
}

/** Verify a delete token for a specific comment. */
export async function verifyDeleteToken(id: number, token: string): Promise<boolean> {
  const data = await load();
  const comment = data.comments.find((c) => c.id === id);
  if (!comment?.deleteTokenHash) return false;
  const hash = createHash("sha256").update(token).digest("hex");
  return hash === comment.deleteTokenHash;
}

/** Delete a comment and any replies to it. */
export async function removeComment(id: number): Promise<boolean> {
  let removed = false;
  await enqueue(async () => {
    const data = await load();
    const before = data.comments.length;
    data.comments = data.comments.filter((c) => c.id !== id && c.parentId !== id);
    removed = data.comments.length !== before;
    if (removed) await save();
  });
  return removed;
}

export async function updateComment(
  id: number,
  patch: { status?: "published" | "hidden"; content?: string },
): Promise<CommentRecord | null> {
  let updated: CommentRecord | null = null;
  await enqueue(async () => {
    const data = await load();
    const comment = data.comments.find((c) => c.id === id);
    if (!comment) return;
    if (patch.status) comment.status = patch.status;
    if (typeof patch.content === "string") {
      comment.content = cleanText(patch.content, MAX_CONTENT);
    }
    await save();
    updated = comment;
  });
  return updated;
}

export interface VoteResult {
  upvotes: number;
  downvotes: number;
  userVote: "up" | "down" | null;
}

/**
 * Cast or retract a vote for a signed-in user. Same action toggles off; the
 * opposite action moves the vote. Returns the new counts, or null if the
 * comment doesn't exist.
 */
export async function voteComment(
  id: number,
  login: string,
  action: "up" | "down",
): Promise<VoteResult | null> {
  const key = login.toLowerCase();
  let result: VoteResult | null = null;
  await enqueue(async () => {
    const data = await load();
    const comment = data.comments.find((c) => c.id === id);
    if (!comment) return;

    const hadUp = comment.upvoters.some((v) => v.toLowerCase() === key);
    const hadDown = comment.downvoters.some((v) => v.toLowerCase() === key);

    if (action === "up") {
      // Toggle off if already upvoted
      if (hadUp) {
        comment.upvoters = comment.upvoters.filter((v) => v.toLowerCase() !== key);
        comment.upvotes -= 1;
      } else {
        if (hadDown) {
          comment.downvoters = comment.downvoters.filter((v) => v.toLowerCase() !== key);
          comment.downvotes -= 1;
        }
        comment.upvoters.push(login);
        comment.upvotes += 1;
      }
    } else {
      if (hadDown) {
        comment.downvoters = comment.downvoters.filter((v) => v.toLowerCase() !== key);
        comment.downvotes -= 1;
      } else {
        if (hadUp) {
          comment.upvoters = comment.upvoters.filter((v) => v.toLowerCase() !== key);
          comment.upvotes -= 1;
        }
        comment.downvoters.push(login);
        comment.downvotes += 1;
      }
    }

    await save();
    result = {
      upvotes: comment.upvotes,
      downvotes: comment.downvotes,
      userVote: voteOf(comment, login),
    };
  });
  return result;
}

/** All comments, newest first, including hidden ones (admin view). */
export async function listAllComments(): Promise<CommentRecord[]> {
  const data = await load();
  return [...data.comments].sort((a, b) => (a.createdAt < b.createdAt ? 1 : -1));
}
