<script lang="ts">
	import { resolve } from '$app/paths';
	import PageHeader from '#lib/PageHeader.svelte';
	import * as config from '#lib/config.js';
	import { formatDate } from '#lib/utils.js';

	let { data } = $props();
</script>

<svelte:head>
	<title>{data.meta.title} — Reference — {config.title}</title>
	<meta name="description" content={data.meta.description} />
</svelte:head>

<nav class="eyebrow crumbs" aria-label="Breadcrumb">
	<a href={resolve('/ref')}>Reference</a>
	<span class="muted">/</span>
	<a href={resolve('/ref/[section]', { section: data.section.id })}>{data.section.title}</a>
</nav>

<PageHeader title={data.meta.title} subtitle={data.meta.description} />

<div class="layout">
	{#if data.meta.headings.length}
		<aside>
			<nav class="toc" aria-label="On this page">
				<span class="eyebrow muted">On this page</span>
				<ul>
					{#each data.meta.headings as heading (heading.id)}
						<li class:sub={heading.depth === 3}><a href="#{heading.id}">{heading.text}</a></li>
					{/each}
				</ul>
			</nav>
		</aside>
	{/if}

	<article>
		<data.content />

		<p class="eyebrow muted updated">Updated {formatDate(data.meta.updated)}</p>

		<nav class="pager" aria-label="Section pages">
			{#if data.prev}
				<a
					href={resolve('/ref/[section]/[slug]', {
						section: data.prev.section,
						slug: data.prev.slug
					})}><span class="eyebrow muted">← Previous</span>{data.prev.title}</a
				>
			{/if}
			{#if data.next}
				<a
					class="next"
					href={resolve('/ref/[section]/[slug]', {
						section: data.next.section,
						slug: data.next.slug
					})}><span class="eyebrow muted">Next →</span>{data.next.title}</a
				>
			{/if}
		</nav>
	</article>
</div>

<style>
	.crumbs {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
		padding-top: 40px;
		margin-bottom: -40px;
	}

	.layout {
		display: grid;
		grid-template-columns: minmax(0, 1fr) 220px;
		gap: 64px;
	}

	article {
		grid-column: 1;
		grid-row: 1;
		min-width: 0;
	}

	aside {
		grid-column: 2;
	}

	.toc {
		position: sticky;
		top: 24px;
		max-height: calc(100dvh - 48px);
		overflow-y: auto;
		font-size: 13px;
	}

	.toc ul {
		list-style: none;
		margin-top: 10px;
		border-left: 1px solid var(--line);
	}

	.toc li {
		padding: 3px 0 3px 12px;
	}

	.toc li.sub {
		padding-left: 24px;
		font-size: 12px;
	}

	.toc a {
		color: var(--muted);
	}

	.toc a:hover {
		color: var(--accent);
	}

	.updated {
		margin-top: 24px;
	}

	.pager {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 20px;
		margin-top: 24px;
		padding-top: 22px;
		border-top: 1px solid var(--line);
	}

	.pager a {
		display: flex;
		flex-direction: column;
		gap: 4px;
		color: var(--text);
	}

	.pager .next {
		grid-column: 2;
		text-align: right;
	}

	@media (max-width: 1050px) {
		.layout {
			grid-template-columns: minmax(0, 1fr);
			gap: 0;
		}

		aside {
			grid-column: 1;
			grid-row: 1;
			margin-bottom: 32px;
		}

		.toc {
			position: static;
			max-height: none;
		}

		article {
			grid-row: 2;
		}
	}
</style>
