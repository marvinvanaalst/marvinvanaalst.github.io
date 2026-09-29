<script lang="ts">
	import { resolve } from '$app/paths';
	import PageHeader from '$lib/PageHeader.svelte';
	import Row from '$lib/Row.svelte';
	import * as config from '$lib/config';
	import { formatDate } from '$lib/utils';

	let { data } = $props();
</script>

<svelte:head>
	<title>Notes — {config.title}</title>
</svelte:head>

<PageHeader title="Notes" subtitle="Probably mostly random ramblings." />

<section>
	{#each data.posts as post (post.slug)}
		<Row
			eyebrow={formatDate(post.date)}
			title={post.title}
			href={resolve('/blog/[slug]', { slug: post.slug })}
			external={false}>{post.description}</Row
		>
	{/each}
</section>
