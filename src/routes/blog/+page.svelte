<script lang="ts">
	import { resolve } from '$app/paths';
	import BlogCard from '$lib/Card.svelte';
	import * as config from '$lib/config';

	let { data } = $props();
</script>

<svelte:head>
	<title>{config.title}</title>
</svelte:head>

<hgroup>
	<h1>Blog</h1>
	<p>Probably mostly random ramblings.</p>
</hgroup>

{#each data.posts as post (post.slug)}
	<BlogCard title={post.title}>
		<p class="description">{post.description}</p>
		<a href={resolve('/blog/[slug]', { slug: post.slug })}>Read on</a>
	</BlogCard>
{/each}
