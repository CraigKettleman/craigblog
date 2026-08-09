import { error, type RequestHandler } from "@sveltejs/kit";
import { env } from "$env/dynamic/private";
import { spawn } from "node:child_process";
import { join } from "node:path";

export const prerender = false;

export const POST: RequestHandler = () => {
  if (env.LOCAL_ADMIN !== "1") throw error(404);

  const cwd = join(import.meta.dirname, "..", "..", "..", "..", "..");

  const stream = new ReadableStream({
    start(controller) {
      const child = spawn("bash", ["deploy/deploy.sh"], {
        cwd,
        env: { ...process.env, SSH_KEY: process.env.HOME + "/.ssh/hhy_homes" },
      });

      child.stdout.on("data", (chunk: Buffer) => {
        controller.enqueue(new TextEncoder().encode(chunk.toString()));
      });

      child.stderr.on("data", (chunk: Buffer) => {
        controller.enqueue(new TextEncoder().encode(chunk.toString()));
      });

      child.on("close", () => {
        controller.close();
      });

      child.on("error", (err) => {
        controller.enqueue(new TextEncoder().encode(`\nError: ${err.message}\n`));
        controller.close();
      });
    },
  });

  return new Response(stream, {
    headers: { "Content-Type": "text/plain; charset=utf-8" },
  });
};
