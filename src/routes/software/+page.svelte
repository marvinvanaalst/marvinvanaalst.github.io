<script lang="ts">
	import PageHeader from '$lib/PageHeader.svelte';
	import ProjectBlock from '$lib/ProjectBlock.svelte';
	import { software } from '$lib/software';
</script>

<svelte:head><title>Software — Marvin van Aalst</title></svelte:head>

<PageHeader title="Software" subtitle="Things I built or contributed to with ❤️" />

<section class="list">
	{#each software as project, i (project.name)}
		<ProjectBlock {project} index={i} text={project.description} />
	{/each}
</section>

<style>
	.list {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 0 48px;
	}

	.list :global(article) {
		border-top: 1px solid var(--line);
		padding-top: 24px;
	}

	@media (max-width: 760px) {
		.list {
			grid-template-columns: 1fr;
		}
	}
</style>
