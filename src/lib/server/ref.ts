import { findSection } from '#lib/ref.js';
import type { RefPage } from '#lib/types.js';

type Frontmatter = Omit<RefPage, 'section' | 'slug' | 'aliases' | 'tags'> &
	Partial<Pick<RefPage, 'aliases' | 'tags'>>;

// Metadata only, so listing pages never pull the page components into a bundle.
const modules = import.meta.glob<Frontmatter>('/src/ref/*/*.md', {
	eager: true,
	import: 'metadata'
});

/** All published reference pages, sorted by `order` then title. */
export function getRefPages(): RefPage[] {
	return Object.entries(modules)
		.map(([path, metadata]) => {
			const [section, file] = path.split('/').slice(-2);
			if (!findSection(section)) throw new Error(`${path}: unknown reference section '${section}'`);
			return {
				aliases: [],
				tags: [],
				...metadata,
				section,
				slug: file.replace(/\.md$/, '')
			} satisfies RefPage;
		})
		.filter((page) => page.published)
		.sort((a, b) => a.order - b.order || a.title.localeCompare(b.title));
}
