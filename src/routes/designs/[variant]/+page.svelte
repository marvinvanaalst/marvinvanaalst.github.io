<script lang="ts">
	import { resolve, asset } from '$app/paths';
	import '$lib/designs.css';
	let { data } = $props();
	let dark = $state(false);
	const names: Record<string, string> = {
		precision: 'Precision',
		workbench: 'Workbench',
		fieldnotes: 'Fieldnotes'
	};
	const projects = [
		{
			name: 'MxlPy',
			tag: 'Mechanistic learning',
			text: 'Connect mechanistic modeling with machine learning to build explainable, data-informed models.',
			repo: 'https://github.com/Computational-Biology-Aachen/MxlPy',
			doi: 'https://doi.org/10.1101/2025.05.06.652335',
			lang: 'Python'
		},
		{
			name: 'modelbase',
			tag: 'Dynamic modeling',
			text: 'Build and analyze dynamic mathematical models of biological systems, from reactions to metabolic networks.',
			repo: 'https://gitlab.com/qtb-hhu/modelbase-software',
			doi: 'https://doi.org/10.1186/s12859-021-04122-7',
			lang: 'Python'
		},
		{
			name: 'dismo',
			tag: 'Spatial models',
			text: 'Bring internal dynamics and transport processes together in discrete spatial models based on differential equations.',
			repo: 'https://gitlab.com/qtb-hhu/dismo',
			doi: 'https://doi.org/10.1101/2023.10.17.562679',
			lang: 'Python'
		}
	];
</script>

<svelte:head><title>{names[data.variant]} — Dr. Marvin van Aalst</title></svelte:head>
<div class="design-preview {data.variant}" class:dark>
	<div class="review-bar">
		<a href={resolve('/designs')}>← All directions</a>
		<nav aria-label="Design variants">
			{#each Object.entries(names) as [id, name] (id)}<a
					aria-current={data.variant === id ? 'page' : undefined}
					href={resolve('/designs/[variant]', { variant: id })}>{name}</a
				>{/each}
		</nav>
		<button
			onclick={() => (dark = !dark)}
			aria-label={dark ? 'Switch to light theme' : 'Switch to dark theme'}
			>{dark ? 'Light ◐' : 'Dark ◑'}</button
		>
	</div>
	<div class="site-shell">
		<header class="site-header">
			<a class="wordmark" href={resolve('/designs/[variant]', { variant: data.variant })}
				>mv<span>/</span>a<span class="wordmark-name">Marvin van Aalst</span></a
			>
			<nav aria-label="Main navigation">
				<a href="#software">Software</a><a href="#research">Research</a><a href={resolve('/talks')}
					>Talks & teaching</a
				><a href={resolve('/blog')}>Notes</a>
			</nav>
		</header>
		<div class="page-grid">
			<aside class="identity">
				<img src={asset('/profile.jpg')} alt="Marvin van Aalst" width="150" height="150" />
				<div>
					<p class="eyebrow">Dr. Marvin van Aalst</p>
					<p class="identity-role">Research software<br />engineer</p>
					<p class="muted">RWTH Aachen University<br />Computational Life Science</p>
				</div>
				<div class="profile-links">
					<a href="https://github.com/marvinvanaalst/">GitHub ↗</a><a
						href="https://orcid.org/0000-0002-7434-0249/">ORCID ↗</a
					><a href="https://www.cpbl.rwth-aachen.de/">Lab profile ↗</a>
				</div>
				<p class="side-note">
					Biological questions.<br />Mathematical models.<br />Software that connects them.
				</p>
			</aside>
			<main id="main-content">
				<section class="hero" aria-labelledby="hero-title">
					<div class="hero-copy">
						<p class="eyebrow">Research software engineer / RWTH Aachen</p>
						<p class="hello">Hello there 👋</p>
						<h1 id="hero-title">
							{#if data.variant === 'precision'}Scientific ideas.<br /><span>Working software.</span
								>{:else if data.variant === 'workbench'}Building the tools<br />to model life.{:else}Where
								models<br />meet learning.{/if}
						</h1>
						<p class="intro">
							I’m Marvin. I build scientific software for universal differential equations,
							connecting mechanistic models with machine learning.
						</p>
						<p class="hero-secondary">
							My research background is in biological modeling and photosynthesis. I like making
							complex models easier to work with.
						</p>
						<div class="hero-links">
							<a href="#software">Explore my software <span>↓</span></a><a
								href="https://github.com/marvinvanaalst/">Find me on GitHub ↗</a
							>
						</div>
					</div>
					<figure class="model">
						<p class="eyebrow">Universal differential equations</p>
						<div
							class="equation"
							aria-label="du over dt equals f of u and p and t plus U theta of u and t"
						>
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
							/><path
								class="curve"
								d="M20 74C55 74 65 12 110 16S170 80 205 43S260 12 310 15"
							/></svg
						>
						<figcaption>
							A model structure for combining what we know with what we learn. <span
								>Illustrative schematic</span
							>
						</figcaption>
					</figure>
				</section>
				<section id="software" class="software-section">
					<div class="section-heading">
						<h2><span class="section-number">01 /</span> Selected software</h2>
						<a href={resolve('/software')}>All software ↗</a>
					</div>
					<div class="project-list">
						{#each projects as project, i (project.name)}<article class="project">
								<div class="project-top">
									<span class="eyebrow">0{i + 1} / {project.tag}</span><span class="language"
										>{project.lang}</span
									>
								</div>
								<h3>{project.name}</h3>
								<p>{project.text}</p>
								<div class="project-links">
									<!-- External repository and DOI URLs, not application routes. -->
									<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -->
									<a href={project.repo}>Repository ↗</a><a href={project.doi}>Paper ↗</a>
								</div>
							</article>{/each}
					</div>
				</section>
				<section id="research" class="research-section">
					<div class="section-heading">
						<h2><span class="section-number">02 /</span> Research behind the tools</h2>
						<a href={resolve('/papers')}>All papers ↗</a>
					</div>
					<div class="research-row">
						<span class="eyebrow">2025 / Software</span><a
							href="https://doi.org/10.1101/2025.05.06.652335"
							>MxlPy — Python Package for Mechanistic Learning in Life Science <span>↗</span></a
						>
					</div>
					<div class="research-row">
						<span class="eyebrow">2025 / Biology</span><a
							href="https://doi.org/10.1126/sciadv.adt9287"
							>Alternatives to photorespiration: A system-level analysis reveals mechanisms of
							enhanced plant productivity <span>↗</span></a
						>
					</div>
				</section>
				<section class="about-section">
					<div>
						<p class="eyebrow">03 / A little background</p>
						<h2>Biology brought me here.<br />Building keeps me curious.</h2>
					</div>
					<div>
						<p>
							I studied Biology and Quantitative Biology at Heinrich Heine University Düsseldorf,
							where I completed my PhD on computational analysis and optimisation of photosynthetic
							carbon fixation.
						</p>
						<p>
							Today, I work as a postdoc and research software engineer in the Computational Life
							Science lab at RWTH Aachen.
						</p>
						<a href={resolve('/talks')}>Talks & teaching ↗</a>
					</div>
				</section>
			</main>
		</div>
		<footer class="site-footer">
			<span>Dr. Marvin van Aalst <span class="muted">/ Built with care & ❤️</span></span>
			<div>
				<a href="https://github.com/marvinvanaalst/">GitHub ↗</a><a
					href="https://gitlab.com/marvin.vanaalst/">GitLab ↗</a
				><a href="https://orcid.org/0000-0002-7434-0249/">ORCID ↗</a>
			</div>
		</footer>
	</div>
</div>
