import { findSection } from '#lib/ref.js';
import { getRefPages } from '#lib/server/ref.js';
import { error } from '@sveltejs/kit';

export function entries() {
	return getRefPages().map(({ section, slug }) => ({ section, slug }));
}

export function load({ params }) {
	const section = findSection(params.section);
	const siblings = getRefPages().filter((page) => page.section === params.section);
	const index = siblings.findIndex((page) => page.slug === params.slug);
	if (!section || index === -1) error(404, `Could not find ${params.section}/${params.slug}`);
	return {
		section,
		prev: siblings[index - 1] ?? null,
		next: siblings[index + 1] ?? null
	};
}
