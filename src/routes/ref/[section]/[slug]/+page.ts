import type { PluginData } from 'mdsvex';
import type { RefPage } from '#lib/types.js';
import type { Component } from 'svelte';

type RefModule = {
	default: Component;
	metadata: Omit<RefPage, 'section' | 'slug'>;
	plugin_data?: PluginData;
};

export async function load({ params, data }) {
	const page: RefModule = await import(`../../../../ref/${params.section}/${params.slug}.md`);
	const headings = page.plugin_data?.toc?.headings ?? [];
	return {
		...data,
		content: page.default,
		meta: page.metadata,
		headings: headings.map(({ depth, id, text }) => ({ depth, id, text }))
	};
}
