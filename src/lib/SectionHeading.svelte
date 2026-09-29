<script lang="ts">
	import type { Snippet } from 'svelte';

	let { number, children, action }: { number?: string; children: Snippet; action?: Snippet } =
		$props();
</script>

<div class="section-heading">
	<h2>
		{#if number}<span class="section-number">{number} /</span>{/if}
		{@render children()}
	</h2>
	{#if action}{@render action()}{/if}
</div>

<style>
	.section-heading {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		gap: 20px;
		padding: 22px 0;
		border-top: 1px solid var(--text);
	}

	.section-heading :global(a) {
		font-size: 12px;
		white-space: nowrap;
	}

	.section-number {
		color: var(--muted);
		font: 11px var(--font-mono);
		margin-right: 20px;
		vertical-align: middle;
	}
</style>
