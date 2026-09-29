<script lang="ts">
	import { resolve } from '$app/paths';
	import * as config from '$lib/config';

	let { params } = $props();
</script>

<svelte:head>
	<title>{config.title} — Talks</title>
</svelte:head>

<p class="back"><a href={resolve('/talks')}>← Talks & teaching</a></p>

<div class="deck">
	<iframe src="/decks/{params.slug}/index.html" title="Talk slides" loading="lazy"></iframe>
</div>

<style>
	.back {
		padding: 32px 0 20px;
		font-size: 13px;
	}

	.deck {
		width: 100%;
		aspect-ratio: 16 / 9;
		border: 1px solid var(--line);
	}

	iframe {
		width: 100%;
		height: 100%;
		border: none;
	}
</style>
