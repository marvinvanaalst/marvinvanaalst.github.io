<script lang="ts">
	let { slug, title = 'Talk slides' }: { slug: string; title?: string } = $props();
</script>

<div class="deck">
	<iframe src="/decks/{slug}/index.html" {title} loading="lazy"></iframe>
</div>

<style>
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
