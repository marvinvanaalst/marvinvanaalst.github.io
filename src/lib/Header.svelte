<script lang="ts">
	import { resolve } from '$app/paths';
	import ThemeSwitcher from './ThemeSwitcher.svelte';
</script>

<header>
	<a class="wordmark" href={resolve('/')} aria-label="Marvin van Aalst, home">mv<span>/</span>a</a>
	<nav aria-label="Main navigation">
		<a href={resolve('/software')}>Software</a>
		<a href={resolve('/papers')}>Research</a>
		<a href={resolve('/talks')}>Talks & teaching</a>
		<a href={resolve('/blog')}>Notes</a>
		<ThemeSwitcher />
	</nav>
</header>

<style>
	header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 32px 0;
		border-bottom: 1px solid var(--line);
	}

	.wordmark {
		color: var(--text);
		font: 600 25px/1 var(--font-mono);
		letter-spacing: -0.07em;
	}

	.wordmark span {
		color: var(--accent);
	}

	nav {
		display: flex;
		align-items: center;
		gap: 25px;
		font-size: 13px;
	}

	nav a {
		color: var(--text);
	}

	@media (max-width: 760px) {
		header {
			flex-direction: column;
			align-items: flex-start;
			gap: 22px;
			padding: 25px 0;
		}

		nav {
			flex-wrap: wrap;
			gap: 14px 20px;
			font-size: 12px;
		}
	}
</style>
