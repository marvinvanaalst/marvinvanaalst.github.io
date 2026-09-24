import type { PageLoad } from './$types';
import { error } from '@sveltejs/kit';

export const prerender = true;
export const entries = () => [
	{ variant: 'precision' },
	{ variant: 'workbench' },
	{ variant: 'fieldnotes' }
];
export const load: PageLoad = ({ params }) => {
	if (!['precision', 'workbench', 'fieldnotes'].includes(params.variant)) error(404);
	return { variant: params.variant };
};
