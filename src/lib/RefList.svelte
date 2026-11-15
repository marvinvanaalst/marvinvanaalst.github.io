<script lang="ts">
	import { resolve } from '$app/paths';
	import type { RefPage } from './types';

	let { pages }: { pages: RefPage[] } = $props();
</script>

<ul>
	{#each pages as page (page.section + '/' + page.slug)}
		<li>
			<a href={resolve('/ref/[section]/[slug]', { section: page.section, slug: page.slug })}
				>{page.title} <span aria-hidden="true">→</span></a
			>
			<p class="muted">{page.description}</p>
			{#if page.aliases.length}
				<p class="aliases">{page.aliases.join(' · ')}</p>
			{/if}
		</li>
	{/each}
</ul>

<style>
	ul {
		list-style: none;
	}

	li {
		padding: 14px 0;
		border-top: 1px solid var(--line);
	}

	a {
		color: var(--text);
		font-weight: 550;
	}

	a span {
		color: var(--accent);
	}

	p {
		font-size: 14px;
		margin-top: 4px;
	}

	.aliases {
		font: 11px/1.6 var(--font-mono);
		color: var(--muted);
		letter-spacing: 0.02em;
	}
</style>
