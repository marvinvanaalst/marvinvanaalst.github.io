<script lang="ts">
	let { children } = $props();
</script>

<article>
	<div>{@render children()}</div>
</article>

<style>
	article {
		width: 100%;
		max-width: 70ch;
		padding: 2rem 0;
		background-color: var(--pico-background-color);
	}

	@media screen and (min-width: 1024px) {
		article {
			padding: 2rem;
			min-height: 512px;
			background-color: var(--pico-card-background-color);
		}
	}
</style>
