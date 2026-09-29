# Personal github.io website

Built with [svelte](https://svelte.dev/) and [picocss](https://picocss.com/).

```bash
curl -fsSL https://bun.sh/install | bash
bun run dev
```

## Adding a talk

Talks are written in Marp Markdown in `0-uni-meta/talks/<slug>/`. To publish one at
`/talks/<slug>` here:

```bash
cd /path/to/0-uni-meta/talks/<slug>
npx @marp-team/marp-cli <slug>.md -o index.html --html

mkdir -p /path/to/marvinvanaalst.github.io/static/decks/<slug>
cp index.html /path/to/marvinvanaalst.github.io/static/decks/<slug>/
cp -r assets /path/to/marvinvanaalst.github.io/static/decks/<slug>/
```

Then add an entry to `src/routes/talks/+page.svelte`, linking to
`resolve('/talks/[slug]', { slug: '<slug>' })`.

`static/decks/<slug>/` is a vendored, self-contained export — its images are exempt
from `scripts/check-image-budget.mjs` and `scripts/check-duplicate-assets.mjs`.
Never commit the rendered `index.html` back into `0-uni-meta`.
