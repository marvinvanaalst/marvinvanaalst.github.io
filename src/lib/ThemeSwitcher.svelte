<script lang="ts">
	import { onMount } from 'svelte';

	// Two-state toggle (modern-web-guidance dark-mode-toggle): "follow the
	// system" or "the opposite of the system", stored literally. Toggling back
	// to what the system shows removes the override instead of pinning it.
	type Scheme = 'light' | 'dark';
	const KEY = 'color-scheme';

	let stored = $state<Scheme | null>(null);
	let systemDark = $state(false);

	const isDark = $derived(stored ? stored === 'dark' : systemDark);
	const label = $derived(isDark ? 'Light' : 'Dark');
	const title = $derived(isDark ? 'Switch to light theme' : 'Switch to dark theme');

	function readStored(): Scheme | null {
		try {
			const value = localStorage.getItem(KEY);
			return value === 'light' || value === 'dark' ? value : null;
		} catch {
			return null;
		}
	}

	function apply(value: Scheme | null) {
		stored = value;
		const meta = document.querySelector<HTMLMetaElement>('meta[name="color-scheme"]');
		if (meta) meta.content = value ?? 'light dark';
	}

	onMount(() => {
		const query = window.matchMedia('(prefers-color-scheme: dark)');
		systemDark = query.matches;
		const onSystemChange = () => (systemDark = query.matches);
		query.addEventListener('change', onSystemChange);

		stored = readStored();
		const onStorage = (event: StorageEvent) => {
			if (event.key === KEY) apply(readStored());
		};
		window.addEventListener('storage', onStorage);

		return () => {
			query.removeEventListener('change', onSystemChange);
			window.removeEventListener('storage', onStorage);
		};
	});

	function toggle() {
		const target: Scheme = isDark ? 'light' : 'dark';
		const system: Scheme = systemDark ? 'dark' : 'light';
		const next = target === system ? null : target;
		apply(next);
		try {
			if (next) localStorage.setItem(KEY, next);
			else localStorage.removeItem(KEY);
		} catch {
			// Not persisted, but still applied for this page view.
		}
	}
</script>

<button type="button" onclick={toggle} {title} aria-label={title}>{label}</button>

<style>
	button {
		margin: 0;
		border: 1px solid var(--line);
		border-radius: 3px;
		background: transparent;
		color: var(--text);
		padding: 3px 9px;
		font: 12px/1.5 var(--font-mono);
		cursor: pointer;
	}

	button:hover {
		border-color: var(--accent);
		color: var(--accent);
	}
</style>
