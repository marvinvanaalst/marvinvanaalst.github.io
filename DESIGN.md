---
name: Marvin van Aalst
description: Personal academic site of a researcher who also builds the tools: quiet, friendly, notebook-plain.
colors:
  violet-ink: "#8352c5"
  violet-ink-hover: "#6935b3"
  violet-ink-dark: "#b290d9"
  violet-ink-dark-hover: "#c9afe4"
  violet-fill: "#7540bf"
  orchid-gradient-start: "#d78cd2"
  lilac-gradient-mid: "#b999ee"
  periwinkle-gradient-mid: "#9da5e9"
  cornflower-gradient-end: "#8fa9f0"
  paper: "#ffffff"
  graphite-text: "#373c44"
  graphite-heading: "#2d3138"
  slate-muted: "#646b79"
  fog-rule: "#e7eaf0"
  night-paper: "#13171f"
  night-panel: "#181c25"
  moonlight-text: "#c2c7d0"
  moonlight-heading: "#f0f1f3"
  night-rule: "#202632"
  teal-drift: "#006165"
typography:
  display:
    fontFamily: "system-ui, \"Segoe UI\", Roboto, Oxygen, Ubuntu, Cantarell, Helvetica, Arial, \"Helvetica Neue\", sans-serif"
    fontSize: "2rem"
    fontWeight: 700
    lineHeight: 1.125
  headline:
    fontFamily: "system-ui, \"Segoe UI\", Roboto, Oxygen, Ubuntu, Cantarell, Helvetica, Arial, \"Helvetica Neue\", sans-serif"
    fontSize: "1.75rem"
    fontWeight: 700
    lineHeight: 1.15
  title:
    fontFamily: "system-ui, \"Segoe UI\", Roboto, Oxygen, Ubuntu, Cantarell, Helvetica, Arial, \"Helvetica Neue\", sans-serif"
    fontSize: "1.5rem"
    fontWeight: 700
    lineHeight: 1.175
  body:
    fontFamily: "system-ui, \"Segoe UI\", Roboto, Oxygen, Ubuntu, Cantarell, Helvetica, Arial, \"Helvetica Neue\", sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "ui-monospace, SFMono-Regular, \"SF Mono\", Menlo, Consolas, \"Liberation Mono\", monospace"
    fontSize: "0.875em"
    fontWeight: 400
    lineHeight: 1.5
rounded:
  control: "0.25rem"
  chip: "10px"
  portrait: "50%"
spacing:
  base: "1rem"
  panel: "2rem"
  nav-y: "1rem"
  nav-x: "0.5rem"
components:
  link:
    textColor: "{colors.violet-ink}"
  link-hover:
    textColor: "{colors.violet-ink-hover}"
  article-panel:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.graphite-text}"
    padding: "2rem"
    width: "70ch"
  card:
    backgroundColor: "{colors.fog-rule}"
    textColor: "{colors.graphite-text}"
    padding: "2rem 1rem"
  card-dark:
    backgroundColor: "{colors.night-rule}"
    textColor: "{colors.moonlight-text}"
    padding: "2rem 1rem"
  profile-portrait:
    rounded: "{rounded.portrait}"
    size: "150px"
  code-chip:
    backgroundColor: "{colors.teal-drift}"
    textColor: "{colors.paper}"
    rounded: "{rounded.chip}"
    padding: "0 6px"
  nav-bar:
    height: "4rem"
    padding: "1rem 0.5rem"
---

# Design System: Marvin van Aalst

## Overview

**Creative North Star: "The Lab Notebook"**

The site reads like a working scientist's notebook: plain paper, system type, one ink color, and a friendly scribble in the margin. Structure is almost entirely inherited from Pico CSS (violet theme). The site adds very little on top of it, and that restraint suits the job. A peer or recruiter should get facts, links and dates without any visual ceremony.

The mood is calm and friendly rather than neutral. Warmth doesn't come from decoration. It comes from a few small, deliberate moments: the soft orchid-to-cornflower gradient on the name on the home page, emoji in the copy, the round portrait, the emoji theme toggle. Everything else stays quiet so those moments read as personal and not as branding.

Density is low. One narrow reading column (70ch panel, 60ch paragraphs) sits next to a slim identity sidebar. Light and dark themes are both first-class. Pico's `prefers-color-scheme` handling is overridden by a manual toggle that stores the choice in localStorage.

**Key Characteristics:**
- One violet ink for links and interactive elements; neutrals do everything else.
- System font stack throughout; no webfonts.
- Flat surfaces separated by tone, not by shadows or borders.
- A single signature flourish: the gradient on the name heading.
- Narrow measure: 60ch paragraphs inside a 70ch panel.
- Light and dark parity, switchable by the visitor.

## Colors

A cool grey neutral base with a single violet ink. Pico's violet theme provides both, and the name gradient adds a pastel accent that appears once.

### Primary
- **Violet Ink** (light) / **Soft Violet Ink** (dark): link and interactive color. Hover deepens in light mode (Deep Violet Ink) and lightens in dark mode. Pico applies it with a half-alpha underline on links.
- **Violet Fill**: solid backgrounds on Pico buttons in both themes. Rare on this site, since there are almost no filled buttons.

### Tertiary
- **Orchid → Lilac → Periwinkle → Cornflower** (four-stop left-to-right gradient): clipped to the text of the name `h1` on the home page only. It is a signature, not a palette. It never appears as a fill, border or background.

### Neutral
- **Paper** / **Night Paper**: page background in light and dark.
- **Night Panel**: the raised reading panel in dark mode. In light mode the panel is Paper, so it doesn't separate from the page.
- **Graphite Text** / **Moonlight Text**: body copy.
- **Graphite Heading** / **Moonlight Heading**: `h1`, slightly stronger than body text.
- **Slate Muted**: `hgroup` subtitles and secondary text.
- **Fog Rule** / **Night Rule**: Pico's muted border color. It's reused here as the card fill on wide screens.

### Drift
- **Teal Drift**: defined as `--primary` in `src/app.css`. It only reaches the mdsvex blog layout (inline code chips and Markdown links in `src/lib/Link.svelte`). It is not part of the system: fold these uses into Violet Ink when that code is next touched.

### Named Rules
**The One Ink Rule.** Violet is the only interactive color. If something is violet it is clickable, and if it is clickable it is violet (or inherits Pico's link treatment).

**The Signature-Once Rule.** The pastel gradient belongs to the name heading. Don't reuse it on section headings, buttons or dividers. Its rarity is what makes it personal.

## Typography

**Display Font:** system-ui (with Pico's full sans-serif fallback stack)
**Body Font:** same stack
**Label/Mono Font:** ui-monospace (with SF Mono, Menlo, Consolas fallbacks), used for code

**Character:** Native, unbranded and fast to load. It's the typography of a well-kept document rather than a publication, and hierarchy comes from size and weight alone.

### Hierarchy
- **Display** (700, 2rem, 1.125): page titles such as the name, "Papers", "Software". Usually inside an `hgroup` with a muted one-line subtitle.
- **Headline** (700, 1.75rem, 1.15): section headings ("Quick CV", "Teaching") and card titles.
- **Title** (700, 1.5rem, 1.175): subsections inside tutorials.
- **Body** (400, 1rem, 1.5): paragraphs capped at 60ch. The root size scales with the viewport (100% → 131.25% from 576px to 1536px), so rem values grow on large screens.
- **Label / Code** (400, 0.875em, mono): inline code and Shiki blocks (poimandres theme).

### Named Rules
**The Short Line Rule.** Paragraphs never exceed 60ch. The narrow measure is what makes long academic text scannable.

**The Subtitle-with-a-Wink Rule.** Page titles pair a plain noun ("Papers") with a small muted `hgroup` line in the first-person voice ("Advancing human knowledge ever so slightly."). The title stays plain, and the personality goes in the subtitle.

## Layout

The body is a Pico `.container`, centered, with a max width that steps up at each breakpoint: 510 / 700 / 950 / 1200 / 1450px at 576 / 768 / 1024 / 1280 / 1536px.

- **Nav bar:** right-aligned links at least 4rem tall. At ≤800px they collapse into a `details` dropdown behind a bars icon, with the theme toggle kept outside it.
- **Two-column shell (≥1024px):** a sidebar (at least 15rem wide) holds the portrait and profile links in a column, next to a centered article panel (up to 70ch, 2rem padding, at least 512px tall).
- **576–1023px:** the sidebar becomes a horizontal band above the article: portrait on the left, profile links in a two-column grid.
- **<576px:** the sidebar is hidden, so the portrait and profile links (ORCID, GitHub, GitLab, RWTH) disappear on phones. This is a known gap, not a principle.
- **Rhythm:** Pico's 1rem spacing unit throughout. Cards add 2rem vertical margin between items.

## Elevation & Depth

The system is flat. Depth is shown by tone: on wide screens the reading panel is one step lighter (dark mode) and cards are filled with the muted border tone.

`Card.svelte` declares `box-shadow: 5px 5px 15px 5px var(--pico-background-color)`. Because the shadow color equals the page color, it's invisible in light mode and only faintly darkens the surroundings in dark mode. Treat that as residue, not vocabulary.

### Named Rules
**The Tone-Not-Shadow Rule.** Separate surfaces by tone only: page, then panel, then card fill. Don't add visible drop shadows.

## Shapes

The shapes are rectilinear and soft-edged by default: Pico's 0.25rem radius on controls, and square corners on cards and the article panel. The one strong shape is the **circular portrait**: 150px, fully round, with a 4px ring in the current text color so it inverts with the theme. Inline code chips in tutorials use a pill-ish 10px radius.

## Components

### Navigation
- **Style:** Pico `nav`, text links in Violet Ink with no background, right-aligned.
- **States:** Pico hover darkens or lightens the ink and underlines. There is no active-page indicator.
- **Mobile (≤800px):** a hamburger `details.dropdown` with a right-to-left list. The theme toggle stays visible.

### Theme Toggle
- **Character:** a playful, bare emoji button: 🌓 (auto), 🌙 (switch to dark), ☀️ (switch to light).
- **Style:** no background, border or shadow. Nav-element padding. Hover takes the violet hover color and underlines.
- **Note:** the button removes the focus outline (`outline: none`) and has no visible replacement. Keyboard focus is invisible, which violates the accessibility target.

### Cards / Containers
- **Used by:** papers, software, blog index.
- **Corner Style:** square.
- **Background:** transparent on narrow screens (content simply stacks). At ≥1024px it is filled with Fog Rule / Night Rule.
- **Internal Padding:** 2rem 1rem (wide), 2rem 0 (narrow).
- **Content pattern:** a Headline-sized title, one or two paragraphs, then plain text links with a leading Font Awesome icon ("Check me out at Github", "Publication", "doi", "Read on").

### Profile Portrait (signature)
A 150px circle with a 4px ring in the text color, centered at the top of the sidebar. If the image fails to load, it falls back to a same-size filled disc reading "Image not found".

### Sidebar Link List
Font Awesome brand icons (building, ORCID, GitHub, GitLab) plus labels in Violet Ink. The list is a single column on wide screens and a two-column grid on medium screens.

### Tutorial Prose (mdsvex layout)
Markdown elements are mapped to custom components in `src/lib/mdpages` and `src/lib/text`. Paragraphs are justified with automatic hyphenation. Code blocks use Shiki's poimandres theme with 1rem padding. Several of these components reference variables that are never defined (`--dark`, `--light`, `--secondary`, `--text-ypad`, `--font-family`), so they fall back to inherited values. The inline code chip is white on 80%-alpha Teal Drift.

## Do's and Don'ts

### Do:
- **Do** use Violet Ink (via Pico's link and primary variables) for every interactive element, in both themes.
- **Do** pair each page title with a muted `hgroup` subtitle in the first-person voice.
- **Do** keep paragraphs at or below 60ch and the reading panel at or below 70ch.
- **Do** separate surfaces with the Paper → Panel → Rule tonal steps.
- **Do** check every change in both light and dark themes, and with the manual toggle overriding the system setting.
- **Do** lead external links with a small Font Awesome icon when they point to a known destination (GitHub, GitLab, ORCID, DOI).

### Don't:
- **Don't** introduce a second accent color. Teal is drift to remove, not a precedent.
- **Don't** reuse the name gradient beyond the name heading.
- **Don't** add webfonts. The system stack is part of the notebook plainness and keeps the site fast.
- **Don't** add visible drop shadows or decorative borders to cards.
- **Don't** remove focus outlines without a visible replacement.
