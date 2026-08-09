export interface ResumeContact {
  kind: "website" | "email" | "github";
  label: string;
  value: string;
  href: string;
}

export interface ResumePreferenceGroup {
  kind: "cities" | "industries" | "roles";
  label: string;
  items: string[];
}

export interface ResumeProject {
  name: string;
  note?: string;
  href?: string;
  description: string;
  stack: string[];
}

export interface ResumeExperience {
  role: string;
  organization: string;
  location?: string;
  period: string;
  summary: string;
  projects: ResumeProject[];
}

export interface ResumeSkillGroup {
  label: string;
  items: string[];
}

export interface ResumeEducation {
  school: string;
  degree: string;
  location: string;
  period: string;
}

export interface ResumeContent {
  pageTitle: string;
  name: string;
  summary: string;
  imageDescription: string;
  highlights: string[];
  sectionLabels: {
    preferences: string;
    experience: string;
    skills: string;
    education: string;
  };
  contacts: ResumeContact[];
  preferenceGroups: ResumePreferenceGroup[];
  experiences: ResumeExperience[];
  skillGroups: ResumeSkillGroup[];
  education: ResumeEducation;
}

// Placeholder resume — replace with your own details.
export const resumeContent: Record<"zh" | "en", ResumeContent> = {
  zh: {
    pageTitle: "简历",
    name: "Craig",
    summary: "简历内容待填写，敬请期待。",
    imageDescription: "个人简历，内容筹备中。",
    highlights: [],
    sectionLabels: {
      preferences: "求职偏好",
      experience: "工作经历",
      skills: "技能",
      education: "教育",
    },
    contacts: [
      {
        kind: "website",
        label: "网站",
        value: "hhy.homes",
        href: "https://hhy.homes",
      },
      {
        kind: "email",
        label: "邮箱",
        value: "craigmail.ai@gmail.com",
        href: "mailto:craigmail.ai@gmail.com",
      },
      {
        kind: "github",
        label: "GitHub",
        value: "github.com/CraigKEttleman",
        href: "https://github.com/CraigKEttleman",
      },
    ],
    preferenceGroups: [],
    experiences: [],
    skillGroups: [],
    education: {
      school: "待填写",
      degree: "待填写",
      location: "",
      period: "",
    },
  },
  en: {
    pageTitle: "Resume",
    name: "Craig",
    summary: "Resume content to be filled in.",
    imageDescription: "A personal resume, under construction.",
    highlights: [],
    sectionLabels: {
      preferences: "Preferences",
      experience: "Experience",
      skills: "Skills",
      education: "Education",
    },
    contacts: [
      {
        kind: "website",
        label: "Website",
        value: "hhy.homes",
        href: "https://hhy.homes",
      },
      {
        kind: "email",
        label: "Email",
        value: "craigmail.ai@gmail.com",
        href: "mailto:craigmail.ai@gmail.com",
      },
      {
        kind: "github",
        label: "GitHub",
        value: "github.com/CraigKEttleman",
        href: "https://github.com/CraigKEttleman",
      },
    ],
    preferenceGroups: [],
    experiences: [],
    skillGroups: [],
    education: {
      school: "TBD",
      degree: "TBD",
      location: "",
      period: "",
    },
  },
};
