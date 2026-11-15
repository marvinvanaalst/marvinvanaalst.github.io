<script lang="ts">
	import { resolve } from '$app/paths';
	import PageHeader from '#lib/PageHeader.svelte';
	import RefList from '#lib/RefList.svelte';
	import SectionHeading from '#lib/SectionHeading.svelte';
	import * as config from '#lib/config.js';

	let { data } = $props();

	let query = $state('');

	const terms = $derived(query.toLowerCase().split(/\s+/).filter(Boolean));

	const groups = $derived(
		data.groups
			.map((group) => ({
				...group,
				pages: group.pages.filter((page) => {
					const haystack = [
						page.title,
						page.description,
						group.title,
						...page.aliases,
						...page.tags
					]
						.join(' ')
						.toLowerCase();
					return terms.every((term) => haystack.includes(term));
				})
			}))
			.filter((group) => terms.length === 0 || group.pages.length > 0)
	);
</script>

<svelte:head>
	<title>Reference — {config.title}</title>
</svelte:head>

<PageHeader
	title="Reference"
	subtitle="Quick refreshers on numerical methods and the maths around them, sorted by what you are trying to do."
/>

<label class="filter">
	<span class="eyebrow muted">Filter</span>
	<input
		type="search"
		bind:value={query}
		placeholder="e.g. GMRES, stiff, functor"
		autocomplete="off"
		spellcheck="false"
	/>
</label>

{#each groups as group, i (group.id)}
	<section class="ref-section">
		<SectionHeading number={String(i + 1).padStart(2, '0')}>
			<a href={resolve('/ref/[section]', { section: group.id })}>{group.title}</a>
		</SectionHeading>
		<p class="muted blurb">{group.blurb}</p>
		{#if group.pages.length}
			<RefList pages={group.pages} />
		{:else}
			<p class="muted empty">Nothing written yet.</p>
		{/if}
	</section>
{:else}
	<p class="muted">No page matches “{query}”.</p>
{/each}

<style>
	.filter {
		display: flex;
		align-items: center;
		gap: 16px;
		margin-bottom: 40px;
	}

	input {
		flex: 1;
		max-width: 420px;
		padding: 8px 12px;
		border: 1px solid var(--line);
		border-radius: 3px;
		background: var(--panel);
		color: var(--text);
		font: 14px var(--font-sans);
	}

	input:focus-visible {
		outline: 2px solid var(--accent);
		outline-offset: 2px;
	}

	.ref-section {
		margin-bottom: 32px;
	}

	.ref-section :global(.section-heading h2 a) {
		color: var(--text);
		font-size: inherit;
		white-space: normal;
	}

	.blurb {
		font-size: 14px;
		margin: -10px 0 12px;
	}

	.empty {
		font-size: 13px;
		font-style: italic;
	}
</style>
