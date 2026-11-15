import { sections } from '#lib/ref.js';
import { getRefPages } from '#lib/server/ref.js';

export function load() {
	const pages = getRefPages();
	return {
		groups: sections.map((section) => ({
			...section,
			pages: pages.filter((page) => page.section === section.id)
		}))
	};
}
