import type { Heading } from '#lib/mdpages/remark-plugins.js';
import type { RefPage } from '#lib/types.js';
import type { Component } from 'svelte';

type RefModule = {
	default: Component;
	metadata: Omit<RefPage, 'section' | 'slug'> & { headings: Heading[] };
};

export async function load({ params, data }) {
	const page: RefModule = await import(`../../../../ref/${params.section}/${params.slug}.md`);
	return { ...data, content: page.default, meta: page.metadata };
}
