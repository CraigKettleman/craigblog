import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { randomBytes, randomInt, scrypt, timingSafeEqual } from "node:crypto";
import { env } from "$env/dynamic/private";

export const MAX_EMAIL = 120;
export const MAX_USERNAME = 24;
export const MIN_PASSWORD = 6;
export const MAX_PASSWORD = 128;

export const USERNAME_RE = /^[\p{L}\p{N}_-]{2,24}$/u;
export const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export interface UserRecord {
  id: number;
  email: string;
  username: string;
  passwordHash: string;
  createdAt: string;
}

interface UserDb {
  version: number;
  nextId: number;
  users: UserRecord[];
}

// ---- Verification codes (in-memory, 5-min TTL) ----

interface CodeEntry { code: string; expires: number; attempts: number; sentAt: number; }

const codes = new Map<string, CodeEntry>();

export const CODE_TTL_MS = 5 * 60_000;
export const CODE_RESEND_MS = 60_000;

export interface CodeIssue {
  ok: boolean;
  code?: string;
  /** Remaining cooldown in ms when ok is false. */
  waitMs?: number;
}

/** Generate a 6-digit code for an email, honoring a 60s resend cooldown. */
export function issueCode(email: string): CodeIssue {
  const key = email.trim().toLowerCase();
  const existing = codes.get(key);
  if (existing) {
    const wait = existing.sentAt + CODE_RESEND_MS - Date.now();
    if (wait > 0) return { ok: false, waitMs: wait };
  }
  const code = String(randomInt(100000, 999999));
  codes.set(key, { code, expires: Date.now() + CODE_TTL_MS, attempts: 0, sentAt: Date.now() });
  return { ok: true, code };
}

/** Verify a code. Max 5 attempts. Returns true if valid. */
export function verifyCode(email: string, code: string): boolean {
  const key = email.trim().toLowerCase();
  const entry = codes.get(key);
  if (!entry) return false;
  if (Date.now() > entry.expires) { codes.delete(key); return false; }
  entry.attempts += 1;
  if (entry.attempts > 5) { codes.delete(key); return false; }
  if (entry.code !== code) return false;
  codes.delete(key);
  return true;
}

// ---- User storage ----

function usersFile(): string {
  return env.USERS_FILE || "/var/www/hhy.homes/data/users.json";
}

let db: UserDb | null = null;
let writeQueue: Promise<void> = Promise.resolve();

async function load(): Promise<UserDb> {
  if (db) return db;
  const file = usersFile();
  try {
    const raw = await readFile(file, "utf8");
    const parsed = JSON.parse(raw) as UserDb;
    db = {
      version: 1,
      nextId: typeof parsed.nextId === "number" ? parsed.nextId : 1,
      users: Array.isArray(parsed.users) ? parsed.users : [],
    };
  } catch {
    db = { version: 1, nextId: 1, users: [] };
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
  const file = usersFile();
  const tmp = `${file}.tmp`;
  await mkdir(dirname(file), { recursive: true });
  await writeFile(tmp, JSON.stringify(db, null, 2), "utf8");
  await rename(tmp, file);
}

async function hashPassword(password: string): Promise<string> {
  const salt = randomBytes(16);
  const hash = await new Promise<Buffer>((resolve, reject) => {
    scrypt(password, salt, 64, (err, key) => (err ? reject(err) : resolve(key)));
  });
  return `${salt.toString("hex")}:${hash.toString("hex")}`;
}

async function verifyPassword(password: string, stored: string): Promise<boolean> {
  const [saltHex, hashHex] = stored.split(":");
  if (!saltHex || !hashHex) return false;
  const hash = await new Promise<Buffer>((resolve, reject) => {
    scrypt(password, Buffer.from(saltHex, "hex"), 64, (err, key) =>
      err ? reject(err) : resolve(key),
    );
  });
  const expected = Buffer.from(hashHex, "hex");
  return hash.length === expected.length && timingSafeEqual(hash, expected);
}

export type RegisterResult =
  | { ok: true; user: { id: number; email: string; username: string } }
  | { ok: false; reason: "email_taken" | "username_taken" | "bad_code" };

export async function registerAccount(input: {
  email: string;
  username: string;
  password: string;
  code: string;
}): Promise<RegisterResult> {
  const email = input.email.trim().toLowerCase();
  const username = input.username.trim();
  if (!EMAIL_RE.test(email) || email.length > MAX_EMAIL) throw new Error("bad email");
  if (!USERNAME_RE.test(username)) throw new Error("bad username");
  if (input.password.length < MIN_PASSWORD || input.password.length > MAX_PASSWORD) {
    throw new Error("bad password");
  }

  if (!verifyCode(email, input.code)) {
    return { ok: false, reason: "bad_code" };
  }

  let result: RegisterResult = { ok: false, reason: "email_taken" };
  await enqueue(async () => {
    const data = await load();
    if (data.users.some((u) => u.email === email)) return;
    if (data.users.some((u) => u.username.toLowerCase() === username.toLowerCase())) {
      result = { ok: false, reason: "username_taken" };
      return;
    }
    const user: UserRecord = {
      id: data.nextId,
      email,
      username,
      passwordHash: await hashPassword(input.password),
      createdAt: new Date().toISOString(),
    };
    data.nextId += 1;
    data.users.push(user);
    await save();
    result = { ok: true, user: { id: user.id, email: user.email, username: user.username } };
  });
  return result;
}

/** Verify credentials; returns the user on success, null on mismatch. */
export async function loginAccount(input: {
  email: string;
  password: string;
}): Promise<UserRecord | null> {
  const email = input.email.trim().toLowerCase();
  const data = await load();
  const user = data.users.find((u) => u.email === email);
  if (!user) return null;
  if (!(await verifyPassword(input.password, user.passwordHash))) return null;
  return user;
}
