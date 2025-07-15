<script lang="ts">
	import { faGithub, faGitlab } from '@fortawesome/free-brands-svg-icons';
	import { faNewspaper } from '@fortawesome/free-solid-svg-icons';
	import type { Snippet } from 'svelte';
	import Fa from 'svelte-fa';
	let {
		children,
		title,
		github = null,
		gitlab = null,
		doi = null
	}: {
		children: Snippet;
		title: string;
		github?: string | null;
		gitlab?: string | null;
		doi?: string | null;
	} = $props();
</script>

<div class="card">
	<h2>{title}</h2>
	{@render children()}
	{#if github != null}
		<a href={github}><Fa icon={faGithub} /> Check me out at Github</a>
	{:else if gitlab != null}
		<a href={gitlab}><Fa icon={faGitlab} /> Check me out at GitLab</a>
	{/if}
	{#if doi != null}
		<a href="https://doi.org/{doi}"><Fa icon={faNewspaper} />Publication</a>
	{/if}
</div>

<style>
	.card {
		background-color: var(--pico-muted-border-color);
		padding: 2rem 1rem;
		margin: 2rem 1rem;
		box-shadow: 5px 5px 15px 5px var(--pico-background-color);
	}
</style>
