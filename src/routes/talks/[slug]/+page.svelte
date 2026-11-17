<script lang="ts">
	import { resolve } from '$app/paths';
	import * as config from '#lib/config.js';
	import Deck from '#lib/Deck.svelte';

	let { params } = $props();
</script>

<svelte:head>
	<title>{config.title} — Talks</title>
</svelte:head>

<p class="back"><a href={resolve('/talks')}>← Talks & teaching</a></p>

<Deck slug={params.slug} />

<style>
	.back {
		padding: 32px 0 20px;
		font-size: 13px;
	}
</style>
