<!--
  @component

  Dev Documentation
  -----------------
  - mdsvex layout for src/ref/**: the shared prose styles plus KaTeX's stylesheet.
  - Math is pre-rendered at build time by mdsvex (katex renderer); only the CSS + fonts ship.
-->

<script lang="ts">
	import 'katex/dist/katex.min.css';
	import Layout from './Layout.svelte';

	let { children } = $props();
</script>

<div class="ref">
	<Layout>{@render children()}</Layout>
</div>

<style>
	.ref :global(.katex) {
		font-size: 1.08em;
	}

	.ref :global(.katex-display) {
		overflow-x: auto;
		overflow-y: hidden;
		padding: 2px 0;
	}

	.ref :global(td) {
		vertical-align: top;
	}
</style>
