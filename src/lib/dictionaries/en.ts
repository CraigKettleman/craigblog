import type { IconName } from "$lib/icons";

interface Contact {
  label: string;
  name: string;
  link: string;
  icon: IconName;
}

interface Work {
  name: string;
  summary: string;
  image?: string;
  link: string;
  primary: boolean;
}

interface Tool {
  name: string;
  summary: string;
  link: string;
  icon: string;
  rating: number;
  platform: string;
  pricing: string;
}

interface PostAdvertisement {
  title: string;
  description: string;
  icon: string;
  link: string;
}

interface ArchiveItem {
  title: string;
  summary: string;
}

// Placeholder contacts — replace with your real profiles.
const contacts: Contact[] = [
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
    label: "Email",
    name: "craigmail.ai@gmail.com",
    link: "mailto:craigmail.ai@gmail.com",
    icon: "mail",
  },
];

const dictionary = {
  meta: {
    baseUrl: "https://hhy.homes",
    fillKeywords(keywords?: string[]): string[] {
      return [
        "Craig",
        "craig",
        "Personal Website",
        "Portfolio",
        "Personal Blog",
        "Personal Website",
        "Student",
        "Bento",
        ...(keywords ?? []),
      ];
    },
  },
  urls: {
    home: "/en",
    share: "/en/share",
    projects: "/en/projects",
    // 技术文章列表展示在「分享」页；posts 保留用于返回链接与文章详情。
    posts: "/en/share",
    about: "/en/about",

    shareToX(title: string, postLink: string) {
      return `https://twitter.com/share?text=${encodeURIComponent(
        `I am reading ${title.toLocaleUpperCase()} @CraigKettlmgc8`,
      )}&url=${encodeURIComponent(`https://hhy.homes${postLink}`)}`;
    },
  },
  labels: {
    home: "Home",
    share: "Share",
    projects: "Project",
    posts: "Share",
    about: "More",
    studio: "Stats",
    latestTech: "Latest",
    myProjects: "My Projects",
    recommended: "Recommended",
    activity: "Activity",
    categories: "Categories",
    featured: "Featured",
    archive: "Archive",
    viewAll: "View All",
    shareTo: "Share to: ",
    backToSection: {
      posts: "← BACK TO SHARE",
      projects: "← BACK TO PROJECTS",
    },
    allSectionPosts: {
      posts: "← ALL SHARE POSTS",
      projects: "← ALL PROJECT POSTS",
    },
    notFoundStatus: "Paper Tray Empty",
    notFoundTitle: "Out of Paper",
    notFoundSubtitle: "Please insert paper correctly to print content.",
    notFoundButton: "← Print Home",
    notFoundError: "ERR 404 · PAPER_NOT_FOUND",
    printedOn: "Printed on",
    reading: "Recent Reading",
    films: "Recent Films",
    music: "Recent Listening",
    aboutTitle: "More",
    aboutSubtitle: "",
    wechatScanHint: "Scan to read on WeChat",
    entries(count: number) {
      return `${count} ${count === 1 ? "entry" : "entries"}`;
    },
    icon(label: string) {
      return `Icon for ${label}`;
    },
  },
  tools: [] as Tool[],
  works: [] as Work[],
  contacts,
  social: {
    followers: "Followers",
    following: "Following",
    posts: "Posts",
    repos: "Repos",
    contributions: "Contribs/yr",
    recentActivity: "Recent activity",
    since(year: string) {
      return `since ${year}`;
    },
    emailTo: "To",
    emailHint: "Mail lands straight in my inbox.",
  },
  comments: {
    title: "Comments",
    count(n: number) {
      return n === 1 ? "1 comment" : `${n} comments`;
    },
    empty: "Be the first to comment.",
    loading: "Loading comments…",
    error: "Couldn't load comments. Please refresh.",
    content: "Comment",
    submit: "Post",
    submitting: "Posting…",
    reply: "Reply",
    cancel: "Cancel",
    replyingTo(name: string) {
      return `Replying to ${name}`;
    },
    replyLine(from: string, to: string) {
      return `${from} replied to ${to}:`;
    },
    signIn: "Sign in",
    signUp: "Sign up",
    signInGithub: "Sign in with GitHub",
    signOut: "Sign out",
    postingAs(name: string) {
      return `Posting as ${name}`;
    },
    owner: "Owner",
    developer: "Developer",
    user: "User",
    visitor: "Guest",
    anonymous: "momo",
    anonymousHint: "Not signed in? Your comment will be posted anonymously.",
    contentRequired: "Please write a comment.",
    contentTooLong: "Comment is too long (max 2000 characters).",
    rateLimited: "You're commenting too fast — try again in a few seconds.",
    serverError: "Something went wrong. Please try again.",
    delete: "Delete",
    deleteConfirm: "Delete this comment?",
    expandReplies(n: number) {
      return `${n} ${n === 1 ? "reply" : "replies"} →`;
    },
    collapseReplies: "Collapse replies",
    sortNewest: "Newest",
    sortHot: "Hottest",
    upvote: "Like",
    downvote: "Dislike",
    loginToVote: "Sign in to vote",
    sendCode: "Send Code",
    codeSent: "Code sent",
    resendIn(seconds: number) {
      return `Resend in ${seconds}s`;
    },
    verificationCode: "Verification Code",
    codeRequired: "Please enter the verification code.",
    badCode: "Invalid or expired verification code.",
    modalTitle: "Sign in",
    email: "Email",
    username: "Username",
    password: "Password",
    haveAccount: "Already have an account? Sign in",
    noAccount: "Don't have an account? Sign up",
    authBadInput: "Please check your input.",
    invalidEmail: "Please enter a valid email address.",
    invalidUsername: "Username must be 2–24 letters, digits, _ or -.",
    invalidPassword: "Password must be 6–128 characters.",
    accountExists: "Email or username already taken.",
    continueAnonymous: "Continue anonymously",
    orContinue: "or",
    contentPlaceholders: [
      "The sky is blue, the comments await you~",
      "Over mountains and rivers, care to leave a comment?",
      "Gently tap on the window of my soul, let me see your thoughts.",
      "A fine vintage of words — care to share a sip?",
      "Allow me to purr a few words.",
      "Respect is the ticket to anyone's heart.",
      "So many comments could be here — missing just yours~",
      "This is the comment section, not the void :-(",
      "I was just waiting for you... not that I wanted comments or anything~",
    ],
  },
  postAdvertisements: [] as PostAdvertisement[],
  archive: {
    reading: [] as ArchiveItem[],
    films: [] as ArchiveItem[],
    music: [] as ArchiveItem[],
  },
  aboutContent: `
This is the personal space of [Craig](https://hhy.homes).

Updated irregularly. Content is being prepared — check back later.
  `,
};

export default dictionary;

export type Dictionary = typeof dictionary;
