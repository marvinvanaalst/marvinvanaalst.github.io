import type { SoftwareProject } from './types';

export const software: SoftwareProject[] = [
	{
		name: 'MxlPy',
		tag: 'Mechanistic learning',
		lang: 'Python',
		summary:
			'Connect mechanistic modeling with machine learning to build explainable, data-informed models.',
		description:
			'MxlPy (pronounced "em axe el pie") is a Python package for mechanistic learning (Mxl) - the combination of mechanistic modeling and machine learning to deliver explainable, data-informed solutions.',
		repo: 'https://github.com/Computational-Biology-Aachen/MxlPy',
		doi: '10.1101/2025.05.06.652335',
		featured: true
	},
	{
		name: 'modelbase',
		tag: 'Dynamic modeling',
		lang: 'Python',
		summary:
			'Build and analyze dynamic mathematical models of biological systems, from reactions to metabolic networks.',
		description:
			'modelbase is a python package to help you build and analyze dynamic mathematical models of biological systems. It has originally been designed for the simulation of metabolic systems, but can be used for virtually any processes, in which some substances get converted into others.',
		repo: 'https://gitlab.com/qtb-hhu/modelbase-software',
		doi: '10.1186/s12859-021-04122-7',
		featured: true
	},
	{
		name: 'MxlBricks',
		tag: 'Reusable building blocks',
		lang: 'Python',
		summary:
			'Build mechanistic learning models quickly from re-usable reaction bricks on top of MxlPy.',
		description:
			'mxlbricks is a library built on top of MxlPy to enable quick building of mechanistic learning models by using re-usable reaction bricks.',
		repo: 'https://github.com/Computational-Biology-Aachen/mxl-bricks',
		featured: true
	},
	{
		name: 'pySBML',
		tag: 'SBML',
		lang: 'Python',
		summary: 'Take SBML models and make them simpler.',
		description: 'pySBML takes SBML models and makes them simpler ❤️',
		repo: 'https://github.com/Computational-Biology-Aachen/pysbml'
	},
	{
		name: 'absorpig',
		tag: 'Spectra',
		lang: 'Python',
		summary: 'Extract the pigment composition of measured absorption spectra.',
		description:
			'Extract pigment composition of measured absorption spectra of photosynthetic organisms.',
		repo: 'https://github.com/Computational-Biology-Aachen/absorpig'
	},
	{
		name: 'moped',
		tag: 'Metabolic models',
		lang: 'Python',
		summary: 'Reproducible construction, curation and analysis of metabolic models.',
		description:
			'moped serves as an integrative hub for reproducible construction, modification, curation and analysis of metabolic models. moped supports draft reconstruction of models directly from genome/proteome sequences and pathway/genome databases utilizing GPR annotations, providing a completely reproducible model construction and curation process',
		repo: 'https://gitlab.com/qtb-hhu/moped',
		doi: '10.3390/metabo12040275'
	},
	{
		name: 'dismo',
		tag: 'Spatial models',
		lang: 'Python',
		summary:
			'Bring internal dynamics and transport processes together in discrete spatial models based on differential equations.',
		description:
			'dismo is a Python package for building and analysing discrete spatial models based on ordinary differential equations. Its primary purpose is to allow arbitrarily complex internal and transport processes to easily be mapped over multiple different regular grids. For this it features one, two and three-dimensional layouts, with standard and non-standard (e.g. hexagonal or triangular) grids.',
		repo: 'https://gitlab.com/qtb-hhu/dismo',
		doi: '10.1101/2023.10.17.562679'
	},
	{
		name: 'cycparser (python)',
		tag: 'Databases',
		lang: 'Python',
		summary: 'Parse *cyc database flatfiles such as MetaCyc and BioCyc.',
		description: 'Library to parse *cyc database flatfiles, such as MetaCyc and BioCyc.',
		repo: 'https://gitlab.com/qtb-hhu/cycparser-py'
	},
	{
		name: 'cycparser (rust)',
		tag: 'Databases',
		lang: 'Rust',
		summary: 'Rust implementation of the cycparser Python package.',
		description: 'Rust implementation of the cycparser Python package.',
		repo: 'https://gitlab.com/qtb-hhu/cycparser-rs'
	},
	{
		name: 'COBREXA.jl',
		tag: 'Constraint-based analysis',
		lang: 'Julia',
		summary: 'Scalable, high-performance analysis of very large-scale metabolic models.',
		description:
			'Cobrexa is a Julia package for scalable, high-performance constraint-based reconstruction and analysis of very large-scale biological models. Its primary purpose is to facilitate the integration of modern high performance computing environments with the processing and analysis of large-scale metabolic models of challenging complexity. We report the architecture of the package, and demonstrate how the design promotes analysis scalability on several use-cases with multi-organism community models.',
		repo: 'https://github.com/COBREXA/COBREXA.jl'
	}
];
