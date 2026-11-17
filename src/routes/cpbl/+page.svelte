<script lang="ts">
	import { asset } from '$app/paths';
	import Deck from '#lib/Deck.svelte';
	import PageHeader from '#lib/PageHeader.svelte';
</script>

<svelte:head><title>What I built at CPBL — Marvin van Aalst</title></svelte:head>

<PageHeader
	title="What I built at CPBL"
	subtitle="Software, science and infrastructure from my time at the Computational and Plant Biology Lab, RWTH Aachen."
/>

<Deck slug="cpbl-achievements" title="What I built at CPBL slides" />

<p class="fullscreen">
	<a href={asset('decks/cpbl-achievements/index.html')} target="_blank" rel="noopener"
		>Open slides full screen ↗</a
	>
</p>

<style>
	.fullscreen {
		padding: 16px 0 48px;
		font-size: 13px;
	}
</style>
