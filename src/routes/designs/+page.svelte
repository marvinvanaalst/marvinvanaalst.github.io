<script lang="ts">
	import { resolve } from '$app/paths';
	import '$lib/designs.css';
	const concepts = [
		{
			id: 'precision',
			number: '01',
			name: 'Precision',
			description:
				'A clear technical index. Strong typography, violet ink, and a structured view of the work.',
			detail: 'Direct · structured · open'
		},
		{
			id: 'workbench',
			number: '02',
			name: 'Workbench',
			description:
				'A software-first workspace. Compact navigation, a persistent profile, and an engineering rhythm.',
			detail: 'Technical · compact · focused'
		},
		{
			id: 'fieldnotes',
			number: '03',
			name: 'Fieldnotes',
			description:
				'A thoughtful scientific notebook. Warm paper, annotated mathematics, and room to read.',
			detail: 'Personal · considered · spacious'
		}
	];
</script>

<svelte:head><title>Three homepage directions — Marvin van Aalst</title></svelte:head>
<div class="design-preview chooser">
	<div class="chooser-inner">
		<a class="eyebrow" href={resolve('/')}>← Current homepage</a>
		<header>
			<p class="eyebrow">Marvin van Aalst / Design studies</p>
			<h1>One focus.<br />Three perspectives.</h1>
			<p class="intro">
				Research software engineering, with scientific depth and a human voice. Choose a direction
				to explore the full homepage.
			</p>
		</header>
		<div class="concept-grid">
			{#each concepts as concept (concept.id)}
				<a class="concept" href={resolve('/designs/[variant]', { variant: concept.id })}>
					<div class="miniature {concept.id}" aria-hidden="true">
						<span class="mini-nav">M / vA <span>Software · Research</span></span>
						<div class="mini-body">
							<i></i>
							<div>
								<b>Science,<br />implemented.</b><span class="mini-line"></span><span
									class="mini-line short"
								></span>
								<div class="mini-projects"><span></span><span></span><span></span></div>
							</div>
						</div>
					</div>
					<p class="eyebrow">{concept.number} / {concept.detail}</p>
					<h2>{concept.name} <span>↗</span></h2>
					<p>{concept.description}</p>
					<span class="concept-open">Explore direction →</span>
				</a>
			{/each}
		</div>
		<p class="chooser-note">
			Each direction includes mobile layouts and a light / dark switch. The project links lead to
			the existing software and research.
		</p>
	</div>
</div>
