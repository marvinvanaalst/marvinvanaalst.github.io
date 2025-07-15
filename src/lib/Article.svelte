<script lang="ts">
	let { children } = $props();
</script>

<article>
	<div>{@render children()}</div>
</article>

<style>
	article {
		width: 100%;
		padding: 2rem;
		background-color: var(--pico-background-color);
	}

	@media screen and (min-width: 800px) {
		article {
			min-height: 512px;
			background-color: var(--pico-card-background-color);
		}
	}
</style>
