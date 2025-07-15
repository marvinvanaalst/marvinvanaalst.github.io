<script lang="ts">
	import type { Snippet } from 'svelte';
	let {
		children,
		title
	}: {
		children: Snippet;
		title: string;
	} = $props();
</script>

<div class="card">
	<h2>{title}</h2>
	{@render children()}
</div>

<style>
	.card {
		background-color: var(--pico-muted-border-color);
		padding: 2rem 1rem;
		margin: 2rem 1rem;
		box-shadow: 5px 5px 15px 5px var(--pico-background-color);
	}
</style>
