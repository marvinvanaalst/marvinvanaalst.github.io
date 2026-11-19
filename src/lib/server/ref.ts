import { findSection } from '#lib/ref.js';
import type { RefPage } from '#lib/types.js';
import { parse } from 'yaml';

// Raw source, not the compiled module: compiling all pages just to read their
// frontmatter made the first /ref request in dev take ~10 s.
const sources = import.meta.glob<string>('/src/ref/*/*.md', {
	eager: true,
	query: '?raw',
	import: 'default'
});

function readFrontmatter(path: string, source: string): Record<string, unknown> {
	const match = /^---\r?\n([\s\S]*?)\r?\n---/.exec(source);
	if (!match) throw new Error(`${path}: missing frontmatter`);
	return parse(match[1]) ?? {};
}

function field<T>(path: string, fm: Record<string, unknown>, key: string, type: string): T {
	if (typeof fm[key] !== type) throw new Error(`${path}: frontmatter '${key}' must be a ${type}`);
	return fm[key] as T;
}

function stringList(path: string, fm: Record<string, unknown>, key: string): string[] {
	const value = fm[key] ?? [];
	if (!Array.isArray(value)) throw new Error(`${path}: frontmatter '${key}' must be a list`);
	return value.map(String);
}

/** All published reference pages, sorted by `order` then title. */
export function getRefPages(): RefPage[] {
	return Object.entries(sources)
		.map(([path, source]) => {
			const [section, file] = path.split('/').slice(-2);
			if (!findSection(section)) throw new Error(`${path}: unknown reference section '${section}'`);
			const fm = readFrontmatter(path, source);
			return {
				section,
				slug: file.replace(/\.md$/, ''),
				title: field<string>(path, fm, 'title', 'string'),
				description: field<string>(path, fm, 'description', 'string'),
				aliases: stringList(path, fm, 'aliases'),
				tags: stringList(path, fm, 'tags'),
				order: field<number>(path, fm, 'order', 'number'),
				updated: String(fm.updated ?? ''),
				published: field<boolean>(path, fm, 'published', 'boolean')
			} satisfies RefPage;
		})
		.filter((page) => page.published)
		.sort((a, b) => a.order - b.order || a.title.localeCompare(b.title));
}
