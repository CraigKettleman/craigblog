<script lang="ts">
  import { page } from "$app/state";
  import { getDictionary, type Language } from "$lib/dictionaries";
  import { resumeContent } from "$lib/resume";
  import type { IconName } from "$lib/icons";
  import Icon from "$lib/components/Icon.svelte";
  import PrintedDivider from "$lib/components/PrintedDivider.svelte";
  import PrintedLabel from "$lib/components/PrintedLabel.svelte";
  import PrintedPageTitle from "$lib/components/PrintedPageTitle.svelte";
  import PrintedSection from "$lib/components/PrintedSection.svelte";
  import Seo from "$lib/components/Seo.svelte";

  let lang = $derived(page.params.lang as Language);
  let dictionary = $derived(getDictionary(lang));
  let content = $derived(resumeContent[lang]);

  const contactIcons: Record<string, IconName> = {
    website: "window",
    email: "mail",
    github: "github",
  };

  const printedOn = new Date().toISOString().split("T")[0];
</script>

<Seo
  {lang}
  title="{content.pageTitle} - {dictionary.meta.websiteName}"
  description={content.summary}
  keywords={["resume", "cv"]}
  path="/{lang}/resume"
  noindex
/>

<div>
  <PrintedSection>
    <PrintedPageTitle icon="user">{content.pageTitle}</PrintedPageTitle>
    <div class="flex flex-col gap-4">
      <div>
        <h2
          class="font-serif text-2xl font-bold tracking-tight text-printer-ink dark:text-printer-ink-dark"
        >
          {content.name}
        </h2>
        <p
          class="mt-2 max-w-2xl font-serif text-xs leading-relaxed text-printer-ink-light dark:text-printer-ink-dark/55"
        >
          {content.summary}
        </p>
      </div>

      <div class="flex flex-wrap gap-2">
        {#each content.contacts as contact (contact.label)}
          {@const external = contact.href.startsWith("https://")}
          <a
            href={contact.href}
            target={external ? "_blank" : undefined}
            rel={external ? "noopener" : undefined}
            class="inline-flex items-center gap-1.5 rounded-sm border border-printer-ink/8 px-2.5 py-1 font-mono text-[10px] tracking-wider text-printer-ink-light transition-colors hover:border-printer-accent/20 hover:text-printer-accent dark:border-printer-ink-dark/8 dark:text-printer-ink-dark/50 dark:hover:border-printer-accent-dark/20 dark:hover:text-printer-accent-dark"
          >
            <Icon name={contactIcons[contact.kind]} class="h-3 w-3 shrink-0" />
            <span class="uppercase">{contact.label}</span>
            <span
              class="normal-case tracking-normal text-printer-ink/70 dark:text-printer-ink-dark/70"
            >
              {contact.value}
            </span>
          </a>
        {/each}
      </div>
    </div>
  </PrintedSection>

  <PrintedDivider style="solid" />

  <PrintedSection label={content.sectionLabels.preferences} labelIcon="tag">
    <div class="grid gap-4 sm:grid-cols-3">
      {#each content.preferenceGroups as group (group.label)}
        <div
          class="rounded-md border border-printer-ink/8 bg-printer-ink/3 p-3 dark:border-printer-ink-dark/8 dark:bg-printer-ink-dark/3"
        >
          <div
            class="font-mono text-[10px] uppercase tracking-[0.25em] text-printer-ink-light dark:text-printer-ink-dark/45"
          >
            {group.label}
          </div>
          <div class="mt-3 flex flex-wrap gap-1.5">
            {#each group.items as item (item)}
              <PrintedLabel variant="default">{item}</PrintedLabel>
            {/each}
          </div>
        </div>
      {/each}
    </div>
  </PrintedSection>

  <PrintedDivider style="dashed" />

  <PrintedSection label={content.sectionLabels.experience} labelIcon="apps">
    <div class="flex flex-col gap-8">
      {#each content.experiences as experience (`${experience.role}-${experience.period}`)}
        <div>
          <div class="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <h3 class="font-serif text-lg text-printer-ink dark:text-printer-ink-dark">
                {experience.role}
              </h3>
              <p
                class="mt-1 font-mono text-[10px] uppercase tracking-[0.25em] text-printer-ink-light dark:text-printer-ink-dark/45"
              >
                {experience.organization}{experience.location
                  ? ` · ${experience.location}`
                  : ""}
              </p>
            </div>
            <PrintedLabel variant="muted">{experience.period}</PrintedLabel>
          </div>

          <p
            class="mt-3 max-w-2xl font-serif text-xs leading-relaxed text-printer-ink-light dark:text-printer-ink-dark/55"
          >
            {experience.summary}
          </p>

          <div class="mt-5 flex flex-col">
            {#each experience.projects as project, index (project.name)}
              <div>
                <div class="rounded-md px-0 py-3">
                  <div class="flex flex-wrap items-center gap-2">
                    {#if project.href}
                      <a
                        href={project.href}
                        target="_blank"
                        rel="noopener"
                        class="font-mono text-sm font-medium text-printer-ink transition-colors hover:text-printer-accent dark:text-printer-ink-dark dark:hover:text-printer-accent-dark"
                      >
                        {project.name}
                      </a>
                    {:else}
                      <span
                        class="font-mono text-sm font-medium text-printer-ink dark:text-printer-ink-dark"
                      >
                        {project.name}
                      </span>
                    {/if}
                    {#if project.note}
                      <span
                        class="font-mono text-[10px] text-printer-ink-light dark:text-printer-ink-dark/40"
                      >
                        {project.note}
                      </span>
                    {/if}
                  </div>

                  <p
                    class="mt-2 font-serif text-xs leading-relaxed text-printer-ink-light dark:text-printer-ink-dark/55"
                  >
                    {project.description}
                  </p>

                  <div class="mt-3 flex flex-wrap gap-1.5">
                    {#each project.stack as item (item)}
                      <PrintedLabel variant="default">{item}</PrintedLabel>
                    {/each}
                  </div>
                </div>

                {#if index < experience.projects.length - 1}
                  <div
                    class="border-b border-dotted border-printer-ink/5 dark:border-printer-ink-dark/5"
                  ></div>
                {/if}
              </div>
            {/each}
          </div>
        </div>
      {/each}
    </div>
  </PrintedSection>

  <PrintedDivider style="dashed" />

  <PrintedSection label={content.sectionLabels.skills} labelIcon="send">
    <div class="grid gap-4 sm:grid-cols-3">
      {#each content.skillGroups as group (group.label)}
        <div
          class="rounded-md border border-printer-ink/8 bg-printer-ink/3 p-3 dark:border-printer-ink-dark/8 dark:bg-printer-ink-dark/3"
        >
          <div
            class="font-mono text-[10px] uppercase tracking-[0.25em] text-printer-ink-light dark:text-printer-ink-dark/45"
          >
            {group.label}
          </div>
          <div class="mt-3 flex flex-wrap gap-1.5">
            {#each group.items as item (item)}
              <PrintedLabel variant="default">{item}</PrintedLabel>
            {/each}
          </div>
        </div>
      {/each}
    </div>
  </PrintedSection>

  <PrintedDivider style="dashed" />

  <PrintedSection label={content.sectionLabels.education} labelIcon="book">
    <div
      class="flex flex-col gap-3 rounded-md border border-printer-ink/8 bg-printer-ink/3 p-4 dark:border-printer-ink-dark/8 dark:bg-printer-ink-dark/3 sm:flex-row sm:items-start sm:justify-between"
    >
      <div>
        <h3 class="font-serif text-lg text-printer-ink dark:text-printer-ink-dark">
          {content.education.school}
        </h3>
        <p class="mt-1 font-serif text-xs text-printer-ink-light dark:text-printer-ink-dark/55">
          {content.education.degree}
        </p>
      </div>
      <div class="flex flex-col gap-2 sm:items-end">
        <PrintedLabel variant="muted">{content.education.period}</PrintedLabel>
        <div
          class="font-mono text-[10px] uppercase tracking-[0.25em] text-printer-ink-light dark:text-printer-ink-dark/45"
        >
          {content.education.location}
        </div>
      </div>
    </div>
  </PrintedSection>

  <div class="mt-8 border-t border-dotted border-printer-ink/10 pt-4 dark:border-printer-ink-dark/10">
    <div class="flex items-center justify-between gap-3">
      <div
        class="font-mono text-[9px] uppercase tracking-wider text-printer-ink-light dark:text-printer-ink-dark/30"
      >
        {dictionary.labels.printedOn}
        {printedOn}
      </div>
      <PrintedLabel variant="muted">resume</PrintedLabel>
    </div>
  </div>
</div>
