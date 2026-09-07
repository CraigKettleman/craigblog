import { error, json, type RequestHandler } from "@sveltejs/kit";
import { env } from "$env/dynamic/private";
import { spawn } from "node:child_process";
import { join } from "node:path";
import { readFile, writeFile, rm } from "node:fs/promises";

export const prerender = false;

/**
 * 部署日志落在临时文件：deploy.sh 内的 pnpm build 会触发 velite 重建，
 * 进而导致 dev 页面整页刷新、断开日志流；把日志持久到文件后，
 * 刷新后的页面可通过 GET 重新接上看进度（dev server 进程本身存活，部署不受影响）。
 */
const LOG_FILE = "/tmp/hhy-homes-deploy.log";
const PID_FILE = "/tmp/hhy-homes-deploy.pid";

function guard() {
  if (env.LOCAL_ADMIN !== "1") throw error(404);
}

export const POST: RequestHandler = () => {
  guard();

  const cwd = join(import.meta.dirname, "..", "..", "..", "..", "..");

  // 客户端断开（dev 页面刷新）后 child 仍在运行；enqueue 前必须检查，
  // 否则向已关闭的 controller 推送会抛未捕获异常并杀死 dev server。
  let closed = false;

  const stream = new ReadableStream({
    start(controller) {
      const push = (text: string) => {
        if (closed) return;
        try {
          controller.enqueue(new TextEncoder().encode(text));
        } catch {
          closed = true;
        }
      };
      const finish = () => {
        if (closed) return;
        closed = true;
        try {
          controller.close();
        } catch {
          /* 已被框架关闭 */
        }
      };

      // tee 把日志同时写入临时文件，供页面刷新后 GET 重新接上
      const child = spawn(
        "bash",
        ["-c", "bash deploy/deploy.sh 2>&1 | tee /tmp/hhy-homes-deploy.log"],
        {
          cwd,
          env: { ...process.env, SSH_KEY: process.env.HOME + "/.ssh/hhy_homes" },
        },
      );
      if (child.pid) {
        rm(PID_FILE, { force: true })
          .then(() => writeFile(PID_FILE, String(child.pid)))
          .catch(() => {});
      }

      child.stdout.on("data", (chunk: Buffer) => push(chunk.toString()));
      child.stderr.on("data", (chunk: Buffer) => push(chunk.toString()));

      child.on("close", () => {
        rm(PID_FILE, { force: true }).catch(() => {});
        finish();
      });

      child.on("error", (err) => {
        push(`\nError: ${err.message}\n`);
        rm(PID_FILE, { force: true }).catch(() => {});
        finish();
      });
    },
    cancel() {
      // 客户端断开：进程继续在后台完成部署，仅停止向断开的连接推送
    },
  });

  return new Response(stream, {
    headers: { "Content-Type": "text/plain; charset=utf-8" },
  });
};

/** 部署状态查询：页面刷新后由 AdminBar 轮询接续（running = pid 存活）。 */
export const GET: RequestHandler = async () => {
  guard();
  try {
    const pid = Number(await readFile(PID_FILE, "utf8"));
    let running = false;
    try {
      process.kill(pid, 0); // 探活：不发送信号
      running = true;
    } catch {
      running = false;
    }
    const log = await readFile(LOG_FILE, "utf8").catch(() => "");
    return json({ running, log: log.slice(-6000) });
  } catch {
    return json({ running: false, log: "" });
  }
};
