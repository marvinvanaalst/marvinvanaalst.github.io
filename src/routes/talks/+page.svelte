<script lang="ts">
	import { resolve } from '$app/paths';
	import PageHeader from '$lib/PageHeader.svelte';
	import Row from '$lib/Row.svelte';
	import SectionHeading from '$lib/SectionHeading.svelte';
	import { talks, teaching } from '$lib/talks';
</script>

<svelte:head><title>Talks & teaching — Marvin van Aalst</title></svelte:head>

<PageHeader title="Talks & teaching" subtitle="Posters, talks, workshops and courses." />

<section>
	<SectionHeading>Talks and poster presentations</SectionHeading>
	{#each talks as talk (talk.date + talk.title + talk.venue)}
		<Row
			eyebrow={talk.date}
			title={talk.title}
			href={talk.slug ? resolve('/talks/[slug]', { slug: talk.slug }) : undefined}
			external={false}>{talk.venue}</Row
		>
	{/each}
</section>

<section class="teaching">
	<SectionHeading>Teaching</SectionHeading>
	{#each teaching as course (course.date + course.title)}
		<Row eyebrow={course.date} title={course.title} />
	{/each}
</section>

<style>
	.teaching {
		margin-top: 48px;
	}
</style>
