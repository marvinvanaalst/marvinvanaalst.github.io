<script lang="ts">
	import { resolve } from '$app/paths';
	import PageHeader from '#lib/PageHeader.svelte';
	import RefList from '#lib/RefList.svelte';
	import * as config from '#lib/config.js';

	let { data } = $props();
</script>

<svelte:head>
	<title>{data.section.title} — Reference — {config.title}</title>
</svelte:head>

<nav class="eyebrow crumbs" aria-label="Breadcrumb">
	<a href={resolve('/ref')}>Reference</a>
</nav>

<PageHeader title={data.section.title} subtitle={data.section.blurb} />

{#if data.pages.length}
	<RefList pages={data.pages} />
{:else}
	<p class="muted">Nothing written yet.</p>
{/if}

<style>
	.crumbs {
		padding-top: 40px;
		margin-bottom: -40px;
	}
</style>
