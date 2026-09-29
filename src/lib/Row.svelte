<script lang="ts">
	import type { Snippet } from 'svelte';

	let {
		eyebrow,
		title,
		href,
		children
	}: { eyebrow: string; title: string; href: string; children?: Snippet } = $props();
</script>

<div class="row">
	<span class="eyebrow muted">{eyebrow}</span>
	<div>
		<!-- Rows link to external URLs (DOIs) or to already-resolved routes. -->
		<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -->
		<a {href}>{title} <span>↗</span></a>
		{#if children}<p class="muted">{@render children()}</p>{/if}
	</div>
</div>

<style>
	.row {
		display: grid;
		grid-template-columns: 180px 1fr;
		gap: 20px;
		border-top: 1px solid var(--line);
		padding: 22px 0;
	}

	a {
		color: var(--text);
		max-width: 65ch;
	}

	a span {
		color: var(--accent);
	}

	p {
		font-size: 14px;
		margin-top: 6px;
	}

	@media (max-width: 760px) {
		.row {
			grid-template-columns: 1fr;
			gap: 6px;
		}
	}
</style>
