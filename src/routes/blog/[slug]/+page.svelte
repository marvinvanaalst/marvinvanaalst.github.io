<script lang="ts">
	import { formatDate } from '$lib/utils';

	let { data } = $props();
</script>

<svelte:head>
	<title>{data.meta.title}</title>
	<meta property="og:type" content="article" />
	<meta property="og:title" content={data.meta.title} />
</svelte:head>

<hgroup>
	<h1>{data.meta.title}</h1>
	<p>Published at {formatDate(data.meta.date)}</p>
</hgroup>

<data.content />

<!-- Categories. Since I don't have any use for them right now,
    let's not use them
-->
<!-- {#each data.meta.categories as category}
	<span>&num;{category}</span>
{/each} -->
