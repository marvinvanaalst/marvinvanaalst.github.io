export type Talk = {
	date: string;
	title: string;
	venue: string;
	slug?: string;
};

export type Teaching = {
	date: string;
	title: string;
};

export const talks: Talk[] = [
	{
		date: '2026-09',
		title: 'Making sense of time series with universal differential equations',
		venue: 'ISGSB (International Study Group for Systems Biology), Ljubljana, Slovenia',
		slug: '2026-09-isgsb'
	},
	{
		date: '2026-05',
		title: 'Universal differential equations',
		venue: 'CCLS Brews and Breakthroughs, Aachen, Germany'
	},
	{
		date: '2026-05',
		title: 'Chill and warm sugars — or how I learned to find the fluxes',
		venue: 'Internal lab talk'
	},
	{
		date: '2026-04',
		title: 'Neural differential equations, take two',
		venue: 'CPBL tool talk'
	},
	{
		date: '2026-01',
		title: 'Mechanistic learning in photosynthetic organisms',
		venue: 'CCLS Symposium, Aachen, Germany'
	},
	{
		date: '2025-11',
		title: 'Neural & universal differential equations',
		venue: 'CPBL tool talk'
	},
	{
		date: '2024',
		title: 'modelbase — constructing modular, reproducible models',
		venue: 'Internal (recorded walkthrough)'
	},
	{
		date: '2024-09',
		title: 'Automatic kinetic model creation using mxlpy',
		venue: 'GCB (German Conference on Bioinformatics), Bielefeld, Germany'
	},
	{
		date: '2024-07',
		title: 'Secondary carbon-fixation improves photorespiration',
		venue: 'ECMTB (European Conference on Mathematical and Theoretical Biology), Toledo, Spain'
	},
	{
		date: '2024-06',
		title: 'Secondary carbon-fixation improves photorespiration',
		venue: 'EPS2 (European Congress for Photosynthesis Research), Padova, Italy'
	},
	{
		date: '2022-09',
		title: 'Optimality principles of leaf venation patterns',
		venue: 'ISGSB (International Study Group for Systems Biology), Innsbruck, Austria'
	},
	{
		date: '2021-09',
		title:
			'How to build and analyse mathematical models of biological systems using Python & modelbase (workshop)',
		venue: 'GCB (German Conference on Bioinformatics), Halle, Germany'
	},
	{
		date: '2021-06',
		title: 'Optimality Principles in leaf venation patterns',
		venue: 'Crops in silico, virtual'
	},
	{
		date: '2019-02',
		title: 'Metabolic Productivity of Photosynthetic glandular trichomes',
		venue: 'MBP (Molecular Biology of Plants), Dabringhausen, Germany'
	},
	{
		date: '2018-09',
		title: 'Optimality principles of leaf venation patterns',
		venue: 'ISGSB (International Study Group for Systems Biology), Tromsø, Norway'
	}
];

export const teaching: Teaching[] = [
	{ date: '2026-03', title: 'JII Open Hackathon Nigeria' },
	{ date: '2025-ws', title: 'Machine learning in natural sciences (16.02) at RWTH Aachen' },
	{ date: '2025-ws', title: 'Societal challenges datathon (42.17) at RWTH Aachen' },
	{ date: '2025-ss', title: 'Interdisciplinary Data Science (16.17) at RWTH Aachen' },
	{ date: '2024-ws', title: 'Machine learning in natural sciences (16.02) at RWTH Aachen' },
	{ date: '2023-ws', title: 'QBio202 - Deterministic processes in Biology at HHU Düsseldorf' },
	{ date: '2023-08', title: 'Embu summer school, Kenya' },
	{ date: '2022-09', title: 'Watamu summer school, Kenya' },
	{ date: '2019-ss', title: 'M4455 - Mathematical modelling at HHU Düsseldorf' },
	{ date: '2019-ws', title: 'BIQ940 - Mathematical modelling at HHU Düsseldorf' },
	{ date: '2018-ws', title: 'BIQ940 - Mathematical modelling at HHU Düsseldorf' }
];
