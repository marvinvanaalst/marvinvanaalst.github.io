<script lang="ts">
	import BlogCard from '$lib/Card.svelte';
	import json from '$lib/publications.json';
</script>

<hgroup>
	<h1>Papers</h1>
	<p>Advancing human knowledge ever so slightly.</p>
</hgroup>

<p>
	An up-to-date collection of my scientific work can always be found on <a
		href="https://orcid.org/0000-0002-7434-0249">ORCID</a
	>
	and
	<a href="https://scholar.google.com/citations?user=LRUnKtIAAAAJ&hl=en&oi=ao">Google Scholar</a>.
</p>
<p>Still, I wanted to toy around with this 😉</p>

{#each json as { title, date, doi, authors } (doi)}
	<BlogCard {title}>
		<p><a href={doi} rel="external">doi</a>, published {date}.</p>
		<p>Authored by {authors.join(', ')}</p>
	</BlogCard>
{/each}
