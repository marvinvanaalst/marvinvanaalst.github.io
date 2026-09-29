<script lang="ts">
	import { asset, resolve } from '$app/paths';
	import ProjectBlock from '$lib/ProjectBlock.svelte';
	import Row from '$lib/Row.svelte';
	import SectionHeading from '$lib/SectionHeading.svelte';
	import publications from '$lib/publications.json';
	import { software } from '$lib/software';

	const featured = software.filter((project) => project.featured);
	const recent = [...publications].sort((a, b) => b.date.localeCompare(a.date)).slice(0, 2);
</script>

<section class="hero" aria-labelledby="hero-title">
	<div class="hero-copy">
		<img
			class="portrait"
			src={asset('/profile.jpg')}
			alt="Marvin van Aalst"
			width="104"
			height="104"
		/>
		<p class="eyebrow lead">Research software engineer / RWTH Aachen</p>
		<p class="hello">Hello there 👋</p>
		<h1 id="hero-title">Scientific ideas.<br /><span>Working software.</span></h1>
		<p class="intro">
			I’m Marvin. I build scientific software for universal differential equations, connecting
			mechanistic models with machine learning.
		</p>
		<p class="secondary">
			My research background is in biological modeling and photosynthesis. I like making complex
			models easier to work with.
		</p>
		<div class="links">
			<a href="#software">Explore my software <span>↓</span></a>
			<a href="https://github.com/marvinvanaalst/">Find me on GitHub ↗</a>
		</div>
	</div>
	<figure class="model">
		<p class="eyebrow">Universal differential equations</p>
		<div class="equation" aria-label="du over dt equals f of u and p and t plus U theta of u and t">
			<span>du<span class="fraction-rule"></span>dt</span><span>=</span><span class="known"
				>f(u, p, t)</span
			><span>+</span><span class="learned">U<sub>θ</sub>(u, t)</span>
		</div>
		<div class="equation-labels">
			<span>Known dynamics</span><span>Learned component</span>
		</div>
		<svg
			viewBox="0 0 320 105"
			role="img"
			aria-label="Schematic of a mechanistic model extended by a learned component"
			><path class="axis" d="M20 10V85H310" /><path
				class="curve baseline"
				d="M20 74C55 74 65 25 110 28S170 65 205 43S260 22 310 20"
			/><path class="curve" d="M20 74C55 74 65 12 110 16S170 80 205 43S260 12 310 15" /></svg
		>
		<figcaption>
			A model structure for combining what we know with what we learn. <span
				>Illustrative schematic</span
			>
		</figcaption>
	</figure>
</section>

<section id="software">
	<SectionHeading number="01">
		Selected software
		{#snippet action()}<a href={resolve('/software')}>All software ↗</a>{/snippet}
	</SectionHeading>
	<div class="project-list">
		{#each featured as project, i (project.name)}
			<ProjectBlock {project} index={i} text={project.summary} />
		{/each}
	</div>
</section>

<section id="research" class="research">
	<SectionHeading number="02">
		Research behind the tools
		{#snippet action()}<a href={resolve('/papers')}>All papers ↗</a>{/snippet}
	</SectionHeading>
	{#each recent as paper (paper.doi)}
		<Row eyebrow={paper.date.slice(0, 4)} title={paper.title} href={paper.doi} />
	{/each}
</section>

<section class="about">
	<div>
		<p class="eyebrow">03 / A little background</p>
		<h2>Biology brought me here.<br />Building keeps me curious.</h2>
	</div>
	<div>
		<p>
			I studied Biology and Quantitative Biology at Heinrich Heine University Düsseldorf, where I
			completed my PhD on computational analysis and optimisation of photosynthetic carbon fixation.
		</p>
		<p>
			Today, I work as a postdoc and research software engineer in the Computational Life Science
			lab at RWTH Aachen.
		</p>
		<a href={resolve('/talks')}>Talks & teaching ↗</a>
	</div>
</section>

<section class="identity" aria-label="About me">
	<img src={asset('/profile.jpg')} alt="" width="75" height="75" />
	<p class="muted">RWTH Aachen University<br />Computational Life Science</p>
	<div class="profile-links">
		<a href="https://github.com/marvinvanaalst/">GitHub ↗</a>
		<a href="https://orcid.org/0000-0002-7434-0249/">ORCID ↗</a>
		<a href="https://www.cpbl.rwth-aachen.de/">Lab profile ↗</a>
	</div>
</section>

<style>
	section {
		scroll-margin-top: 25px;
	}

	.hero {
		display: grid;
		grid-template-columns: 1.4fr 1fr;
		gap: 70px;
		padding: 76px 0 72px;
		align-items: center;
	}

	.portrait {
		display: block;
		width: 104px;
		height: 104px;
		margin-bottom: 26px;
		border-radius: 50%;
		object-fit: cover;
		border: 3px solid var(--paper);
		box-shadow: 0 0 0 2px var(--accent);
	}

	.lead {
		padding-left: 17px;
		border-left: 3px solid var(--accent);
	}

	.hello {
		margin: 30px 0 13px;
		font-size: 15px;
	}

	h1 span {
		color: var(--accent);
	}

	.intro {
		font-size: 19px;
		line-height: 1.65;
		margin-top: 27px;
	}

	.secondary {
		color: var(--muted);
		font-size: 15px;
		margin-top: 14px;
	}

	.links {
		display: flex;
		gap: 26px;
		margin-top: 28px;
		font-size: 13px;
	}

	.links a:first-child {
		font-weight: 600;
	}

	.model {
		padding: 28px;
		border-left: 1px solid var(--line);
	}

	.equation {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 8px;
		font:
			24px/1.4 Georgia,
			serif;
		margin-top: 30px;
		white-space: nowrap;
	}

	.equation > span:first-child {
		display: flex;
		flex-direction: column;
		text-align: center;
		font-style: italic;
	}

	.fraction-rule {
		width: 100%;
		height: 1px;
		background: currentColor;
	}

	.learned {
		color: var(--accent);
	}

	.equation-labels {
		display: flex;
		justify-content: space-between;
		gap: 10px;
		margin: 16px 0 0 65px;
		color: var(--muted);
		font: 10px/1.5 var(--font-mono);
	}

	svg {
		display: block;
		width: 100%;
		margin-top: 35px;
	}

	.axis {
		stroke: var(--line);
		fill: none;
	}

	.curve {
		stroke: var(--accent);
		stroke-width: 2;
		fill: none;
	}

	.baseline {
		stroke: var(--muted);
		stroke-width: 1.3;
		stroke-dasharray: 4 5;
	}

	figcaption {
		margin-top: 18px;
		color: var(--muted);
		font-size: 11px;
		line-height: 1.6;
	}

	figcaption span {
		display: block;
		font: 9px/1.5 var(--font-mono);
		margin-top: 8px;
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}

	.project-list {
		display: grid;
		grid-template-columns: repeat(3, minmax(0, 1fr));
		gap: 32px;
	}

	.research {
		padding-top: 25px;
	}

	.about {
		margin-top: 40px;
		padding: 35px 0 65px;
		border-top: 1px solid var(--line);
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 60px;
	}

	.about h2 {
		margin-top: 18px;
	}

	.about p:not(.eyebrow) {
		font-size: 14px;
		color: var(--muted);
		margin-bottom: 15px;
	}

	.about a {
		font-size: 13px;
	}

	.identity {
		display: flex;
		align-items: center;
		gap: 24px;
		border-top: 1px solid var(--line);
		padding: 25px 0 35px;
	}

	.identity img {
		width: 75px;
		height: 75px;
		border-radius: 50%;
		object-fit: cover;
	}

	.identity p {
		font-size: 14px;
	}

	.profile-links {
		display: flex;
		gap: 20px;
		margin-left: auto;
		font-size: 12px;
	}

	@media (max-width: 1050px) {
		.hero {
			gap: 30px;
		}

		.equation {
			font-size: 20px;
		}

		.model {
			padding: 20px;
		}

		.project-list {
			gap: 20px;
		}
	}

	@media (max-width: 760px) {
		.hero {
			grid-template-columns: 1fr;
			padding: 40px 0;
			gap: 35px;
		}

		.portrait {
			width: 84px;
			height: 84px;
			margin-bottom: 20px;
		}

		h1 {
			font-size: clamp(39px, 8.5vw, 58px);
		}

		.intro {
			font-size: 17px;
		}

		.links {
			flex-wrap: wrap;
			gap: 14px 25px;
		}

		.model {
			max-width: 500px;
			width: 100%;
			border-left: 0;
			border-top: 1px solid var(--line);
		}

		.equation {
			width: 100%;
			gap: 20px;
		}

		.project-list,
		.about {
			grid-template-columns: 1fr;
		}

		.about {
			gap: 30px;
		}

		.identity {
			flex-wrap: wrap;
		}

		.profile-links {
			margin-left: 0;
			width: 100%;
		}
	}
</style>
