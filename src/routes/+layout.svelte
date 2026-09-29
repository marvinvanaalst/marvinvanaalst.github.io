<script lang="ts">
	import { page } from '$app/state';
	import * as config from '$lib/config';
	import '../app.css';

	import Footer from '$lib/Footer.svelte';
	import Header from '$lib/Header.svelte';

	let { children } = $props();
</script>

<!-- SEO -->
<svelte:head>
	<meta name="description" content={config.description} />
	<meta property="og:title" content={config.title} />
	<meta property="og:description" content={config.description} />
	<meta property="og:url" content={config.url} />
	<meta property="og:type" content="website" />
	<meta name="twitter:card" content="summary" />
</svelte:head>

{#if page.url.pathname.startsWith('/designs')}
	{@render children()}
{:else}
	<div class="shell">
		<Header />
		<main id="main-content">{@render children()}</main>
		<Footer />
	</div>
{/if}

<style>
	.shell {
		max-width: 1320px;
		margin: auto;
		padding: 0 48px;
	}

	@media (max-width: 1050px) {
		.shell {
			padding: 0 30px;
		}
	}

	@media (max-width: 760px) {
		.shell {
			padding: 0 22px;
		}
	}
</style>
