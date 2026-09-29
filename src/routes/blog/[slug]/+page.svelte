<script lang="ts">
	import PageHeader from '$lib/PageHeader.svelte';
	import { formatDate } from '$lib/utils';

	let { data } = $props();
</script>

<svelte:head>
	<title>{data.meta.title}</title>
	<meta property="og:type" content="article" />
	<meta property="og:title" content={data.meta.title} />
</svelte:head>

<PageHeader title={data.meta.title} subtitle="Published {formatDate(data.meta.date)}" />

<data.content />
