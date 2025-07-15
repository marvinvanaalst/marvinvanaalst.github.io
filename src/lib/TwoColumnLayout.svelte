<script lang="ts">
	let { children } = $props();
</script>

<div>
	{@render children()}
</div>

<style>
	div {
		display: flex;
		flex-direction: column;
	}

	@media screen and (min-width: 1024px) {
		div {
			flex-direction: row;
			justify-content: center;
		}
	}
</style>
