export type Categories = 'sveltekit' | 'svelte';

export type Post = {
	title: string;
	slug: string;
	description: string;
	date: string;
	categories: Categories[];
	published: boolean;
};

export type Publication = {
	title: string;
	date: string;
	doi: string;
	authors: string[];
};

export type SoftwareProject = {
	name: string;
	tag: string;
	lang: string;
	summary: string;
	description: string;
	repo: string;
	doi?: string;
	featured?: boolean;
};
