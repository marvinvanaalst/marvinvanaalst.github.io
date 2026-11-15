import { findSection, sections } from '#lib/ref.js';
import { getRefPages } from '#lib/server/ref.js';
import { error } from '@sveltejs/kit';

export function entries() {
	return sections.map((section) => ({ section: section.id }));
}

export function load({ params }) {
	const section = findSection(params.section);
	if (!section) error(404, `Unknown section ${params.section}`);
	return { section, pages: getRefPages().filter((page) => page.section === section.id) };
}
