import { site as siteData } from "#velite";
import type { Language } from "$lib/dictionaries";

export interface SiteMeta {
  websiteName: string;
  motto: string;
  mottos: string[];
  brandName: string;
  brandTagline: string;
  about: string;
  shareSubtitle: string;
  shareSubscribeHint: string;
}

/** site.yml 缺失或字段不完整时的兜底值，避免页面渲染失败。 */
const FALLBACKS: Record<Language, SiteMeta> = {
  en: {
    websiteName: "Craig",
    motto: "Attention is all i need.",
    mottos: ["Attention is all i need."],
    brandName: "Craig",
    brandTagline: "Personal Website",
    about: "This is the personal space of [Craig](https://hhy.homes).\n\nUpdated irregularly. Content is being prepared — check back later.",
    shareSubtitle: "",
    shareSubscribeHint: "",
  },
  zh: {
    websiteName: "Craig",
    motto: "Attention is all i need.",
    mottos: ["Attention is all i need."],
    brandName: "Craig",
    brandTagline: "个人主页",
    about: "这里是 [Craig](https://hhy.homes) 的个人空间。\n\n不定期更新。内容筹备中，敬请期待。",
    shareSubtitle: "",
    shareSubscribeHint: "",
  },
};

type SiteData = {
  websiteName: Record<Language, string>;
  motto: Record<Language, string>;
  mottos: Record<Language, string[]>;
  brandName: Record<Language, string>;
  brandTagline: Record<Language, string>;
  about: Record<Language, string>;
  shareSubtitle?: Record<Language, string>;
  shareSubscribeHint?: Record<Language, string>;
};

/** 读取某语言的站点元信息（名称/座右铭等），缺失时回退到默认值。 */
export function siteMeta(lang: Language): SiteMeta {
  const data = siteData as unknown as SiteData | undefined;
  const fallback = FALLBACKS[lang];
  return {
    websiteName: data?.websiteName?.[lang] ?? fallback.websiteName,
    motto: data?.motto?.[lang] ?? fallback.motto,
    mottos: (() => {
      const m = data?.mottos?.[lang];
      return m && m.length > 0 ? m : fallback.mottos;
    })(),
    brandName: data?.brandName?.[lang] ?? fallback.brandName,
    brandTagline: data?.brandTagline?.[lang] ?? fallback.brandTagline,
    about: (data?.about?.[lang] || "").trim() || fallback.about,
    // 分享页文案：site.yml 未配置时返回空串，由页面回退到字典默认值
    shareSubtitle: (data?.shareSubtitle?.[lang] ?? "").trim(),
    shareSubscribeHint: (data?.shareSubscribeHint?.[lang] ?? "").trim(),
  };
}
