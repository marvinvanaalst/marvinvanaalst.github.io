<script lang="ts">
	import * as config from '$lib/config';
	import '../app.css';

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
	<li><a href="/">Home</a></li>
	<li><a href="/papers">Papers</a></li>
	<li><a href="/talks">Talks</a></li>
	<li><a href="/software">Software</a></li>
	<li><a href="/blog">Blog</a></li>
</Navbar>
<TwoColumnLayout>
	<Sidebar />
	<Article>{@render children()}</Article>
</TwoColumnLayout>
