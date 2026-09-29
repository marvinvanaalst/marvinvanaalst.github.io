<script lang="ts">
	import { onMount } from 'svelte';

	type Theme = 'auto' | 'light' | 'dark';

	let theme = $state<Theme>('auto');
	let systemDark = $state(false);

	const isDark = $derived(theme === 'auto' ? systemDark : theme === 'dark');
	const icon = $derived(theme === 'auto' ? '◐' : isDark ? '☀' : '☾');
	const title = $derived(
		theme === 'auto'
			? 'Following system theme (click to override)'
			: isDark
				? 'Switch to light theme'
				: 'Switch to dark theme'
	);

	onMount(() => {
		const query = window.matchMedia('(prefers-color-scheme: dark)');
		systemDark = query.matches;
		const onChange = () => (systemDark = query.matches);
		query.addEventListener('change', onChange);

		const saved = localStorage.getItem('theme');
		theme = saved === 'light' || saved === 'dark' ? saved : 'auto';

		return () => query.removeEventListener('change', onChange);
	});

	function toggle() {
		theme = isDark ? 'light' : 'dark';
		document.documentElement.setAttribute('data-theme', theme);
		localStorage.setItem('theme', theme);
	}
</script>

<button onclick={toggle} {title} aria-label={title}>{icon}</button>

<style>
	button {
		margin: 0;
		border: 1px solid var(--line);
		border-radius: 3px;
		background: transparent;
		color: var(--text);
		padding: 3px 9px;
		font: inherit;
		font-size: 13px;
		cursor: pointer;
	}

	button:hover {
		border-color: var(--accent);
		color: var(--accent);
	}
</style>
