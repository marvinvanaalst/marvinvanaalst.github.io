<script lang="ts">
	import * as config from '$lib/config';
	import '../app.css';

	import { resolve } from '$app/paths';
	import Article from '$lib/Article.svelte';
	import Navbar from '$lib/Navbar.svelte';
	import Sidebar from '$lib/Sidebar.svelte';
	import TwoColumnLayout from '$lib/TwoColumnLayout.svelte';

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

<Navbar>
	<li><a href={resolve('/')}>Home</a></li>
	<li><a href={resolve('/papers')}>Papers</a></li>
	<li><a href={resolve('/talks')}>Talks</a></li>
	<li><a href={resolve('/software')}>Software</a></li>
	<li><a href={resolve('/blog')}>Blog</a></li>
</Navbar>
<TwoColumnLayout>
	<Sidebar />
	<Article>{@render children()}</Article>
</TwoColumnLayout>
