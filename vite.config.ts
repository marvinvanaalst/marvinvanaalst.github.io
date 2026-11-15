import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';
import { escapeSvelte, mdsvex, type MdsvexOptions } from 'mdsvex';
import remarkMath from 'remark-math';
import { createHighlighter } from 'shiki';
import { remarkHeadings, remarkKatex } from './src/lib/mdpages/remark-plugins.ts';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const layouts_dir = join(__dirname, './src/lib/mdpages');

const highlighter = await createHighlighter({
	themes: ['poimandres'],
	langs: [
		'javascript',
		'typescript',
		'python',
		'rust',
		'c',
		'cpp',
		'julia',
		'fortran-free-form',
		'toml'
	]
});

const mdsvexOptions: MdsvexOptions = {
	extensions: ['.md'],
	remarkPlugins: [remarkMath, remarkHeadings, remarkKatex],
	highlight: {
		highlighter: (code, lang = 'text') => {
			const html = escapeSvelte(
				highlighter.codeToHtml(code, { lang: lang ?? 'text', theme: 'poimandres' })
			);
			return `{@html \`${html}\` }`;
		}
	},
	layout: {
		_: join(layouts_dir, 'Layout.svelte'),
		tutorials: join(layouts_dir, 'Layout.svelte'),
		ref: join(layouts_dir, 'RefLayout.svelte')
	}
};

export default defineConfig({
	plugins: [
		sveltekit({
			extensions: ['.svelte', '.md'],
			preprocess: [vitePreprocess(), mdsvex(mdsvexOptions)],
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
