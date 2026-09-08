import type { Dictionary } from "./en";

const dictionary: Dictionary = {
  meta: {
    baseUrl: "https://hhy.homes",
    fillKeywords(keywords?: string[]): string[] {
      return [
        "Craig",
        "craig",
        "个人主页",
        "个人网站",
        "个人博客",
        "学生",
        "Bento",
        ...(keywords ?? []),
      ];
    },
  },
  urls: {
    home: "/zh",
    share: "/zh/share",
    projects: "/zh/projects",
    // 技术文章列表展示在「分享」页；posts 保留用于返回链接与文章详情。
    posts: "/zh/share",
    about: "/zh/about",

    shareToX(title: string, postLink: string) {
      return `https://twitter.com/share?text=${encodeURIComponent(
        `我正在看「${title}」 @CraigKettlmgc8`,
      )}&url=${encodeURIComponent(`https://hhy.homes${postLink}`)}`;
    },
  },
  labels: {
    home: "主页",
    share: "分享",
    projects: "项目",
    posts: "分享",
    about: "更多",
    studio: "看板",
    latestTech: "最新",
    myProjects: "我的项目",
    recommended: "推荐",
    activity: "活动",
    categories: "分类",
    featured: "精选",
    archive: "归档",
    viewAll: "更多",
    shareTo: "分享到：",
    backToSection: {
      posts: "← 返回分享",
      projects: "← 返回项目",
    },
    allSectionPosts: {
      posts: "← 全部分享文章",
      projects: "← 全部项目文章",
    },
    notFoundStatus: "纸空了",
    notFoundTitle: "托盘已空",
    notFoundSubtitle: "请正确放入纸张以打印内容。",
    notFoundButton: "← 打印主页",
    notFoundError: "ERR 404 · PAPER_NOT_FOUND",
    printedOn: "打印于",
    reading: "最近阅读",
    films: "最近观影",
    music: "最近聆听",
    aboutTitle: "更多",
    aboutSubtitle: "",
    wechatScanHint: "微信扫码阅读原文",
    entries(count: number) {
      return `${count} 条`;
    },
    icon(label: string) {
      return `${label}的图标`;
    },
  },
  tools: [],
  works: [],
  contacts: [
    {
      label: "X (Twitter)",
      name: "@CraigKettlmgc8",
      link: "https://x.com/CraigKettlmgc8",
      icon: "x",
    },
    {
      label: "GitHub",
      name: "@CraigKEttleman",
      link: "https://github.com/CraigKEttleman",
      icon: "github",
    },
    {
      label: "邮箱",
      name: "craigmail.ai@gmail.com",
      link: "mailto:craigmail.ai@gmail.com",
      icon: "mail",
    },
  ],
  social: {
    followers: "关注者",
    following: "关注中",
    posts: "帖子",
    repos: "仓库",
    contributions: "年贡献",
    recentActivity: "近期活动",
    since(year: string) {
      return `${year} 年至今`;
    },
    emailTo: "收件人",
    emailHint: "邮件会直达我的收件箱。",
  },
  comments: {
    title: "评论",
    count(n: number) {
      return `${n} 条评论`;
    },
    empty: "还没有评论，来抢沙发。",
    loading: "加载评论…",
    error: "评论加载失败，请刷新重试。",
    content: "评论内容",
    submit: "发布",
    submitting: "发布中…",
    reply: "回复",
    cancel: "取消",
    replyingTo(name: string) {
      return `回复 ${name}`;
    },
    replyLine(from: string, to: string) {
      return `${from} 回复 ${to}：`;
    },
    signIn: "登录",
    signUp: "注册",
    signInGithub: "使用 GitHub 登录",
    signOut: "退出登录",
    postingAs(name: string) {
      return `以 ${name} 发布`;
    },
    owner: "博主",
    developer: "开发者",
    user: "用户",
    visitor: "访客",
    anonymous: "momo",
    anonymousHint: "未登录将以匿名身份发表评论。",
    contentRequired: "请填写评论内容。",
    contentTooLong: "评论过长（最多 2000 字）。",
    rateLimited: "评论太频繁，请稍后再试。",
    serverError: "出了点问题，请重试。",
    delete: "删除",
    deleteConfirm: "确定删除此评论？",
    expandReplies(n: number) {
      return `共 ${n} 条回复 →`;
    },
    collapseReplies: "收起回复",
    sortNewest: "最新",
    sortHot: "最热",
    upvote: "赞",
    downvote: "踩",
    loginToVote: "登录后可以点赞",
    sendCode: "发送验证码",
    codeSent: "验证码已发送",
    resendIn(seconds: number) {
      return `${seconds}s 后重发`;
    },
    verificationCode: "验证码",
    codeRequired: "请输入验证码。",
    badCode: "验证码错误或已过期。",
    modalTitle: "登录",
    email: "邮箱",
    username: "用户名",
    password: "密码",
    haveAccount: "已有账号？去登录",
    noAccount: "没有账号？去注册",
    authBadInput: "请检查输入信息。",
    invalidEmail: "请输入正确的邮箱地址。",
    invalidUsername: "用户名需为 2–24 位字母、数字、下划线或连字符。",
    invalidPassword: "密码需为 6–128 位字符。",
    accountExists: "该邮箱或用户名已被注册。",
    continueAnonymous: "匿名参与",
    orContinue: "或",
    contentPlaceholders: [
      "天青色等烟雨，评论在等你～",
      "万水千山总是情，留下评论行不行？",
      "轻轻敲碎沉睡的心灵，让我看看你的点评",
      "宫廷玉液酒，评论走一走",
      "下面我简单喵两句",
      "尊重是打动人心的入场券",
      "评论千万条，就差你一条～",
      "这里是评论区，不是无人区 :-（",
      "只是一直等你而已，才不是想被评论呢～",
    ],
  },
  postAdvertisements: [],
  archive: {
    reading: [],
    films: [],
    music: [],
  },
  aboutContent: `
这里是 [Craig](https://hhy.homes) 的个人空间。

不定期更新。内容筹备中，敬请期待。
  `,
};

export default dictionary;
