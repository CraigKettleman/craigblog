import nodemailer from "nodemailer";
import { env } from "$env/dynamic/private";

/** Whether SMTP credentials are present so real emails can be sent. */
export function smtpConfigured(): boolean {
  return Boolean(env.SMTP_HOST && env.SMTP_USER && env.SMTP_PASS);
}

/** Send a plain-text email. Throws on failure. */
export async function sendEmail(to: string, subject: string, text: string): Promise<void> {
  const host = env.SMTP_HOST;
  const user = env.SMTP_USER;
  const pass = env.SMTP_PASS;
  if (!host || !user || !pass) throw new Error("SMTP not configured");

  const transporter = nodemailer.createTransport({
    host,
    port: Number(env.SMTP_PORT || 465),
    secure: (env.SMTP_PORT || "465") === "465",
    auth: { user, pass },
  });

  await transporter.sendMail({
    from: env.SMTP_FROM || user,
    to,
    subject,
    text,
  });
}

/** Compose the verification-code email body. */
export function verificationMail(code: string): { subject: string; text: string } {
  return {
    subject: "hhy.homes 注册验证码",
    text: `你的注册验证码是：${code}\n\n验证码 5 分钟内有效。\n如果不是你本人操作，请忽略此邮件。`,
  };
}
