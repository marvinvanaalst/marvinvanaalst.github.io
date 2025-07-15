<script lang="ts">
	let {
		path
	}: {
		path: string;
	} = $props();
</script>

<img src={path} alt="profile" />

<style>
	img {
		width: 150px;
		height: 150px;
		border-radius: 50%;
		margin-bottom: 1rem;
		border: 4px solid var(--pico-color);
	}
</style>
