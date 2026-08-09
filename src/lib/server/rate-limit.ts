import { createHash } from "node:crypto";

export function hashIp(ip: string): string {
  return createHash("sha256").update(ip).digest("hex");
}

/** Client IP from X-Forwarded-For (nginx sets it), falling back to the socket. */
export function clientIp(forwarded: string | null, remote?: string): string {
  if (forwarded) {
    const first = forwarded.split(",")[0]?.trim();
    if (first) return first;
  }
  return remote ?? "unknown";
}

const WINDOW_MS = 60 * 1000;
const MAX_GUEST = 5;
const MAX_GITHUB = 50;
const MAX_AUTH = 10;

// ipHash -> timestamps within the current window
const commentHits = new Map<string, number[]>();
const authHits = new Map<string, number[]>();

function hitsFor(map: Map<string, number[]>, ipHash: string): number[] {
  const now = Date.now();
  const kept = (map.get(ipHash) ?? []).filter((t) => now - t < WINDOW_MS);
  map.set(ipHash, kept);
  return kept;
}

/**
 * Atomically check the sliding window and record a slot. Returns false when the
 * ipHash has already hit the cap. GitHub-signed commenters get a wider cap.
 */
export function consumeCommentSlot(ipHash: string, isGithub: boolean): boolean {
  const max = isGithub ? MAX_GITHUB : MAX_GUEST;
  const arr = hitsFor(commentHits, ipHash);
  if (arr.length >= max) return false;
  arr.push(Date.now());
  return true;
}

/** Auth attempts (register/login) share a tighter window per IP. */
export function consumeAuthSlot(ipHash: string): boolean {
  const arr = hitsFor(authHits, ipHash);
  if (arr.length >= MAX_AUTH) return false;
  arr.push(Date.now());
  return true;
}
