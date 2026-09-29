<script lang="ts">
	import PageHeader from '$lib/PageHeader.svelte';
	import Row from '$lib/Row.svelte';
	import json from '$lib/publications.json';
</script>

<svelte:head><title>Research — Marvin van Aalst</title></svelte:head>

<PageHeader title="Research" subtitle="Advancing human knowledge ever so slightly." />

<p class="note muted">
	An up-to-date collection of my scientific work can always be found on <a
		href="https://orcid.org/0000-0002-7434-0249">ORCID</a
	>
	and
	<a href="https://scholar.google.com/citations?user=LRUnKtIAAAAJ&hl=en&oi=ao">Google Scholar</a>.
</p>

<section>
	{#each json as { title, date, doi, authors } (doi)}
		<Row eyebrow={date.slice(0, 4)} {title} href={doi}>{authors.join(', ')}</Row>
	{/each}
</section>

<style>
	.note {
		margin-bottom: 32px;
		font-size: 14px;
	}
</style>
