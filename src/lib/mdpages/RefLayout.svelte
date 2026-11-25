<!--
  @component

  Dev Documentation
  -----------------
  - mdsvex layout for src/ref/**: the shared prose layout (which also carries KaTeX's
    stylesheet) plus ref-specific table styles.
-->

<script lang="ts">
	import Layout from './Layout.svelte';

	let { children } = $props();
</script>

<div class="ref">
	<Layout>{@render children()}</Layout>
</div>

<style>
	.ref :global(td) {
		vertical-align: top;
	}
</style>
