# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

- **Academic peers and hiring committees**: PIs, search committees and potential collaborators sizing Marvin up as a researcher. They scan to judge fit: position and affiliation, research focus, papers, talks, teaching.
- **Industry / research-software-engineering roles**: recruiters and engineering teams looking at Marvin as a research software engineer. They want evidence of shipped, maintained software and engineering depth.

Both groups arrive with limited time, often from a CV, a paper, ORCID or GitHub. A good visit ends with them clear on who Marvin is, what he builds and researches, and how to follow up.

Secondary: students and self-learners who land on a tutorial. They're served, but the site isn't organized around them.

## Product Purpose

The personal academic site of Dr. Marvin van Aalst, a postdoc and research software engineer in the Computational Life Science lab at RWTH Aachen. It presents him as a researcher in mathematical/theoretical biology (plant metabolism, photosynthesis, metabolic modeling) and as a builder of scientific software. Success means a peer or recruiter can quickly verify his profile and find the proof: papers, software, talks, teaching.

## Positioning

A scientist who builds the tools as well as using them. His software (MxlPy, modelbase, moped, dismo, COBREXA.jl contributions, and others) is published, cited and used, and it sits next to his first-author research on photosynthesis and metabolism. The site's claim is that both halves are real and connected.

## Operating Context

- Sections: Home (intro plus a short CV), Papers, Software, Talks & Teaching, Blog.
- Visitors often arrive from external profiles (ORCID, Google Scholar, GitHub, GitLab, RWTH lab page) and follow links back out to them.
- Tutorials (the blog) are an occasional extra: worked mxlpy modelling examples, not a growing centerpiece.

## Capabilities and Constraints

- **Stack**: SvelteKit with `adapter-static`, deployed to GitHub Pages. It must remain a static build.
- Blog posts are mdsvex Markdown (`src/posts/*.md`) with Shiki highlighting. Papers render from `src/lib/publications.json`.
- Enforced budgets: client JS chunks ≤ 500 KB (`scripts/check-chunk-budget.mjs`), committed raster images ≤ 300 KB and ≤ 2000 px on the longest edge (`scripts/check-image-budget.mjs`).
- Pico CSS is incidental, not a commitment; future work may replace it.
- Currently has a light/dark theme switcher.

## Brand Commitments

- Name: "Dr. Marvin van Aalst". Affiliation: RWTH Aachen University.
- Voice: informal, warm, first person, with the occasional emoji ("Hello there 👋", "built with ❤️", "Advancing human knowledge ever so slightly"). This voice is intentional and must stay.

## Evidence on Hand

- Profile photo: `static/profile.jpg`.
- Publication list with DOIs and authors: `src/lib/publications.json`.
- Software list with repo and DOI links: `src/routes/software/+page.svelte`.
- Talks (2018–2024) and teaching (2018–2026): `src/routes/talks/+page.svelte`.
- Six tutorials with generated figures: `src/posts/`, `static/tutorials/`.
- Education and thesis history: `src/routes/+page.svelte`.
- None of the following exist, and none may be fabricated: testimonials, usage or download numbers, citation counts, awards, logos of employers or partners.

## Product Principles

1. **Proof over claims.** Every statement about research or software should link to its evidence: DOI, repository, venue.
2. **Scannable in a minute.** A busy reviewer should grasp role, focus and output without having to read paragraphs.
3. **Researcher and engineer, together.** Present science and software as one connected body of work, not two separate résumés.
4. **Human voice.** Keep it personal and friendly, not a corporate portfolio.
5. **Light and fast.** Static, within budget, quick on any connection.

## Accessibility & Inclusion

The user confirmed there is an accessibility target but hasn't named the standard. *Open decision:* WCAG 2.2 AA is the likely baseline and needs confirming. The site must work in both light and dark themes.
