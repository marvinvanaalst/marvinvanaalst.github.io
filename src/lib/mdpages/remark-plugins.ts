import GithubSlugger from 'github-slugger';
import katex from 'katex';

// mdsvex 0.12 runs on the pre-micromark unified 9 stack, so these work on plain
// mdast objects instead of the current @types/mdast (v4) node types.
type Node = {
	type: string;
	value?: string;
	depth?: number;
	children?: Node[];
	data?: { hProperties?: Record<string, unknown> };
};

type VFile = { data: { fm?: Record<string, unknown> } };

export type Heading = { depth: 2 | 3; text: string; id: string };

function walk(node: Node, fn: (node: Node, index: number, parent: Node) => void) {
	node.children?.forEach((child, index) => {
		fn(child, index, node);
		walk(child, fn);
	});
}

function toText(node: Node): string {
	if (node.value !== undefined) return node.value;
	return (node.children ?? []).map(toText).join('');
}

/**
 * Renders remark-math's `math` / `inlineMath` nodes with KaTeX at build time.
 * Braces are escaped because the MathML annotation keeps the raw TeX, which
 * Svelte would otherwise parse as `{expressions}`. Bad TeX fails the build.
 */
export function remarkKatex() {
	return (tree: Node) => {
		walk(tree, (node, index, parent) => {
			if (node.type !== 'math' && node.type !== 'inlineMath') return;
			const html = katex.renderToString(node.value ?? '', {
				displayMode: node.type === 'math',
				throwOnError: true
			});
			parent.children![index] = {
				type: 'html',
				value: html.replaceAll('{', '&#123;').replaceAll('}', '&#125;')
			};
		});
	};
}

/** Gives every heading a GitHub-style id and exposes h2/h3 as `metadata.headings`. */
export function remarkHeadings() {
	return (tree: Node, file: VFile) => {
		const slugger = new GithubSlugger();
		const headings: Heading[] = [];
		walk(tree, (node) => {
			if (node.type !== 'heading') return;
			const text = toText(node);
			const id = slugger.slug(text);
			node.data = { ...node.data, hProperties: { ...node.data?.hProperties, id } };
			if (node.depth === 2 || node.depth === 3) headings.push({ depth: node.depth, text, id });
		});
		file.data.fm = { ...file.data.fm, headings };
	};
}
