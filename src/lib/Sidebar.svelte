<script lang="ts">
	import Fa from 'svelte-fa';

	// import RoundImagePlaceholder from '$lib/RoundImagePlaceholder.svelte';
	import { faGithub, faGitlab, faOrcid } from '@fortawesome/free-brands-svg-icons';
	import { faBuilding } from '@fortawesome/free-solid-svg-icons';
	import RoundImage from './RoundImage.svelte';
</script>

<aside aria-label="Sidebar">
	<!-- <RoundImagePlaceholder /> -->

	<RoundImage path="profile.jpg" />
	<ul>
		<li>
			<a href="https://www.cpbl.rwth-aachen.de/">
				<Fa icon={faBuilding} />
				<span>RWTH Aachen</span>
			</a>
		</li>
		<li>
			<a href="https://orcid.org/0000-0002-7434-0249/">
				<Fa icon={faOrcid} />
				<span>ORCID</span>
			</a>
		</li>
		<li>
			<a href="https://github.com/marvinvanaalst/">
				<Fa icon={faGithub} />
				<span>Github</span>
			</a>
		</li>
		<li>
			<a href="https://gitlab.com/marvin.vanaalst/">
				<Fa icon={faGitlab} />
				<span>Gitlab</span>
			</a>
		</li>
	</ul>
</aside>

<style>
	aside {
		display: none;
	}

	@media screen and (min-width: 800px) {
		aside {
			padding-top: var(--pico-block-spacing-vertical);
			display: flex;
			flex-direction: column;
			align-items: center;
			min-width: 15rem;
		}
	}
</style>
