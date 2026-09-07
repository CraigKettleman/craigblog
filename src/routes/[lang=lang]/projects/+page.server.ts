import { categoriesOf, postsOf, toListItem } from "$lib/content";
import type { Language } from "$lib/dictionaries";
import type { PageServerLoad } from "./$types";

export const load: PageServerLoad = ({ params }) => {
  const lang = params.lang as Language;

  return {
    projectPosts: postsOf(lang, "projects").map(toListItem),
    categories: categoriesOf("projects").map((category) => ({
      slug: category.slug,
      name: category.name,
    })),
  };
};
