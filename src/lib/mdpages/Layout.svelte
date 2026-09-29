<!--
  @component

  Dev Documentation
  -----------------
  - mdsvex passes the frontmatter (title, date, ...) as props; only the content is rendered here.
  - shiki highlighter returns {@html ...} raw HTML, so code blocks are styled via :global(.shiki).
-->

<script lang="ts">
	let { children } = $props();
</script>

<div class="prose">
	{@render children()}
</div>

<style>
	.prose {
		max-width: 72ch;
		padding-bottom: 32px;
		line-height: 1.75;
	}

	.prose :global(> * + *) {
		margin-top: 1.1em;
	}

	.prose :global(h1),
	.prose :global(h2),
	.prose :global(h3),
	.prose :global(h4) {
		line-height: 1.25;
		margin-top: 2em;
	}

	.prose :global(> :first-child) {
		margin-top: 0;
	}

	.prose :global(h1) {
		font-size: 30px;
		letter-spacing: -0.04em;
	}

	.prose :global(h2) {
		font-size: 24px;
		padding-top: 22px;
		border-top: 1px solid var(--line);
	}

	.prose :global(h3) {
		font-size: 19px;
	}

	.prose :global(h4) {
		font-size: 16px;
	}

	.prose :global(p),
	.prose :global(li) {
		max-width: none;
	}

	.prose :global(ul),
	.prose :global(ol) {
		padding-left: 1.4em;
	}

	.prose :global(li + li) {
		margin-top: 0.3em;
	}

	.prose :global(a) {
		text-decoration: underline;
		text-underline-offset: 3px;
	}

	.prose :global(:not(pre) > code) {
		background: var(--panel);
		border: 1px solid var(--line);
		border-radius: 3px;
		padding: 1px 5px;
		font: 0.88em var(--font-mono);
		hyphens: none;
	}

	.prose :global(.shiki) {
		padding: 1rem;
		border-radius: 3px;
		overflow-x: auto;
		font: 13px/1.6 var(--font-mono);
	}

	.prose :global(.shiki code) {
		font: inherit;
	}

	.prose :global(table) {
		display: block;
		overflow-x: auto;
		border-collapse: collapse;
		width: 100%;
		text-align: left;
		font-size: 14px;
	}

	.prose :global(th),
	.prose :global(td) {
		padding: 8px 12px;
		border-bottom: 1px solid var(--line);
	}

	.prose :global(th) {
		font: 11px var(--font-mono);
		text-transform: uppercase;
		letter-spacing: 0.075em;
		color: var(--muted);
	}

	.prose :global(hr) {
		border: 0;
		border-top: 1px solid var(--line);
		margin: 2em 0;
	}

	.prose :global(blockquote) {
		margin-left: 0;
		padding-left: 17px;
		border-left: 3px solid var(--accent);
		color: var(--muted);
	}

	.prose :global(img) {
		max-width: 100%;
		height: auto;
	}
</style>
