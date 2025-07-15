<script lang="ts">
	import BlogCard from '$lib/BlogCard.svelte';
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

{#each data.posts as post}
	<BlogCard title={post.title}>
		<p class="description">{post.description}</p>
		<a href="/blog/{post.slug}">Read on</a>
	</BlogCard>
{/each}
