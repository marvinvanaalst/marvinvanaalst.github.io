<script lang="ts">
	import type { SoftwareProject } from './types';

	let { project, index, text }: { project: SoftwareProject; index: number; text: string } =
		$props();
</script>

<article>
	<div class="top">
		<span class="eyebrow muted">{String(index + 1).padStart(2, '0')} / {project.tag}</span>
		<span class="language">{project.lang}</span>
	</div>
	<h3>{project.name}</h3>
	<p>{text}</p>
	<div class="links">
		<!-- External repository and DOI URLs, not application routes. -->
		<!-- eslint-disable svelte/no-navigation-without-resolve -->
		<a href={project.repo}>Repository ↗</a>
		{#if project.doi}<a href="https://doi.org/{project.doi}">Paper ↗</a>{/if}
		<!-- eslint-enable svelte/no-navigation-without-resolve -->
	</div>
</article>

<style>
	article {
		display: flex;
		flex-direction: column;
		padding: 8px 0 34px;
	}

	.top {
		display: flex;
		justify-content: space-between;
		flex-wrap: wrap;
		gap: 10px;
		align-items: baseline;
		margin-bottom: 18px;
	}

	.top .eyebrow {
		font-size: 9px;
	}

	.language {
		font: 10px var(--font-mono);
		color: var(--muted);
	}

	p {
		font-size: 14px;
		color: var(--muted);
		margin-top: 12px;
	}

	.links {
		display: flex;
		gap: 22px;
		padding-top: 22px;
		margin-top: auto;
		font-size: 12px;
	}
</style>
