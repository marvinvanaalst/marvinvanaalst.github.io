<script lang="ts">
	let { title, subtitle }: { title: string; subtitle?: string } = $props();
</script>

<div class="page-header">
	<h1>{title}</h1>
	{#if subtitle}<p class="muted">{subtitle}</p>{/if}
</div>

<style>
	.page-header {
		padding: 64px 0 40px;
	}

	h1 {
		font-size: clamp(36px, 4vw, 52px);
	}

	p {
		margin-top: 14px;
		font-size: 17px;
	}

	@media (max-width: 760px) {
		.page-header {
			padding: 40px 0 28px;
		}
	}
</style>
