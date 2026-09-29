<script lang="ts">
	import { resolve } from '$app/paths';
	import * as config from '$lib/config';

	let { params } = $props();
</script>

<svelte:head>
	<title>{config.title} — Talks</title>
</svelte:head>

<p><a href={resolve('/talks')}>← Talks</a></p>

<div class="deck">
	<iframe src="/decks/{params.slug}/index.html" title="Talk slides" loading="lazy"></iframe>
</div>

<style>
	.deck {
		width: 100%;
		aspect-ratio: 16 / 9;
	}

	iframe {
		width: 100%;
		height: 100%;
		border: none;
	}
</style>
