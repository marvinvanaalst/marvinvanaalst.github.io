<script lang="ts">
	import '../app.css';

	import Article from '$lib/Article.svelte';
	import Navbar from '$lib/Navbar.svelte';
	import Sidebar from '$lib/Sidebar.svelte';
	import TwoColumnLayout from '$lib/TwoColumnLayout.svelte';

	let { children } = $props();
	export const prerender = true;
</script>

<Navbar>
	<li><a href="/">Home</a></li>
	<li><a href="/papers">Papers</a></li>
	<!-- <li><a href="/talks">Talks</a></li> -->
	<li><a href="/software">Software</a></li>
	<li><a href="/blog">Blog</a></li>
</Navbar>
<TwoColumnLayout>
	<Sidebar />
	<Article>{@render children()}</Article>
</TwoColumnLayout>
