import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';
import { mdsvex } from 'mdsvex';
import { toc } from '@mdsvex/plugin-toc';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [
		mdsvex({
			extensions: ['.md'],
			heading_ids: true,
			math: { renderer: 'katex' },
			// h2/h3 as plugin_data.toc.headings, for the /ref table of contents
			plugins: [toc()],
			templates: {
				default: './src/lib/mdpages/Layout.svelte',
				tutorials: './src/lib/mdpages/Layout.svelte',
				ref: './src/lib/mdpages/RefLayout.svelte'
			},
			// 0.x picked a named layout by folder name; templates need it explicitly.
			select_template: (id) =>
				id.includes('/tutorials/') ? 'tutorials' : id.includes('/ref/') ? 'ref' : undefined,
			smart_punctuation: true
		}),
		sveltekit({
			extensions: ['.svelte', '.md'],
			preprocess: [vitePreprocess()],
			adapter: adapter({
				pages: 'build',
				assets: 'build',
				precompress: false,
				strict: true,
				fallback: '404.html'
			})
		})
	]
});
