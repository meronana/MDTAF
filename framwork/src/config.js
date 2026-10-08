// Site copy. Once the paper title, authors and links are final, only edit this file.
export const SITE = {
  title: 'MDTAF',
  subtitle:
    'A transferability assessment framework for cross-experimental-domain transfer, from in vitro assays to human pharmacokinetics',
  authors: 'Yesugen',
  affiliation: 'Yesugen',
  links: [
    { label: 'Paper', href: '#', soon: true },
    { label: 'Code', href: 'https://github.com/meronana/MDTAF', soon: false },
    { label: 'Try it', href: '#try', soon: false },
  ],
};

export const ENDPOINTS = {
  fu: { label: 'f<sub>u</sub>', text: 'fu', name: 'Fraction unbound' },
  CL: { label: 'CL', text: 'CL', name: 'Clearance' },
  t12: { label: 't<sub>½</sub>', text: 't½', name: 'Half-life' },
};

// Domain list for the picker. endpoints = endpoints that have real data in that domain
// (animal PK: only CL from Animal_PK_data.csv is used in the experiments)
export const DOMAINS = {
  source: [
    { id: 'invitro', name: 'In vitro ADME', note: 'ChEMBL · TDC · Biogen', endpoints: ['fu', 'CL', 't12'] },
    { id: 'animal', name: 'Animal in vivo (rat + dog + monkey)', note: 'Lombardo 2013 / PKSmart', endpoints: ['CL'] },
  ],
  target: [
    { id: 'human', name: 'Human in vivo PK', note: 'PKSmart', endpoints: ['fu', 'CL', 't12'] },
    { id: 'rat', name: 'Rat in vivo', note: 'Lombardo 2013', endpoints: ['CL'] },
    { id: 'dog', name: 'Dog in vivo', note: 'Lombardo 2013', endpoints: ['CL'] },
    { id: 'monkey', name: 'Monkey in vivo', note: 'Lombardo 2013', endpoints: ['CL'] },
  ],
};

// Interpretation bands for label correspondence (1-NN ρ). Empirical thresholds for display only
export const RHO_BANDS = [
  { min: 0.5, label: 'High', text: 'Source labels track target labels closely. Source-label pretraining is likely to help.' },
  { min: 0.3, label: 'Moderate', text: 'Partial agreement. Expect small, uncertain gains; validate on held-out target data.' },
  { min: -1, label: 'Low', text: 'Source labels carry little information about the target. Input alignment alone will not fix this.' },
];

// One-to-one with benchmark/phase_0X_*.ipynb
export const PHASES = [
  { n: 1, group: 'Data', title: 'Input validation', body: 'Standardise SMILES, deduplicate, and confirm zero source–target overlap by InChIKey.' },
  { n: 2, group: 'Input space', title: 'Chemical space', body: 'ECFP4 Tanimoto similarity, Bemis–Murcko scaffolds and descriptor distributions between domains.' },
  { n: 3, group: 'Input space', title: 'Representation gap', body: 'MMD, CORAL and sliced-Wasserstein distance in the frozen Graphormer embedding space.' },
  { n: 4, group: 'Label space', title: 'Label correspondence', body: 'Spearman ρ between each target compound and its nearest source neighbour (1-NN, Tanimoto ≥ 0.4).', key: true },
  { n: 5, group: 'Input space', title: 'Domain discriminability', body: 'How easily a classifier tells source from target (ROC-AUC; 0.5 = indistinguishable).' },
  { n: 6, group: 'Data', title: 'Dataset compatibility', body: 'Sample size, balance and data quality checks for whether adaptation is feasible at all.' },
  { n: 7, group: 'Decision', title: 'Transferability profile', body: 'Combine all dimensions into a per-endpoint profile and a recommendation for or against transfer.' },
  { n: 8, group: 'Validation', title: 'DA benchmark', body: 'MMD, CORAL, DANN, CDAN and importance weighting against a target-only baseline under scaffold CV.' },
  { n: 9, group: 'Validation', title: 'Prediction validation', body: 'Check whether the profile predicted which endpoints actually benefit from source data.' },
];

// Data download page. The licence must be settled before public release
export const DATA_PAGE = {
  // The curated data are committed to the repository; the page links to them on GitHub
  repo: { name: 'meronana/MDTAF', branch: 'main' },
  license: 'License to be confirmed before public release',
  sources: [
    { name: 'ChEMBL', role: 'in vitro source (fu, CLint, t½)', note: 'ChEMBL data are released under CC BY-SA 3.0' },
    { name: 'Therapeutics Data Commons (TDC)', role: 'in vitro source (PPBR, hepatocyte / microsome CL, half-life)', note: 'see each TDC dataset for its original licence' },
    { name: 'Biogen ADME (Fang et al., 2023)', role: 'in vitro source (HLM CLint)', note: 'see the original publication' },
    { name: 'PKSmart', role: 'human in vivo target (fu, CL, t½)', note: 'see the original publication' },
  ],
  pipeline: [
    ['Collect', 'Pull in vitro ADME records from ChEMBL, TDC and Biogen; human PK targets from PKSmart.'],
    ['Canonicalise', 'RDKit canonical SMILES; drop unparsable structures.'],
    ['Deduplicate', 'Merge duplicate measurements per compound with assay-aware rules (smart deduplication).'],
    ['Transform', 'log10 for all labels; source z-scored over the whole set, target z-scored per CV fold.'],
    ['Split', 'Bemis–Murcko scaffold 5-fold CV on the target; fold test compounds removed from the source.'],
  ],
  // Column descriptions (only shown for names that match the manifest's columns)
  columns: {
    smiles: 'RDKit canonical SMILES',
    value: 'Label after transformation (source: log10 → z-score; target: log10)',
    ik: 'First block of the InChIKey (connectivity), used for overlap checks',
    relation: 'Qualifier reported with the measurement (=, <, >)',
    units: 'Original unit of the measurement',
    assay_id: 'ChEMBL assay ID',
    doc_id: 'ChEMBL document ID',
    desc: 'Assay description from the original source',
    activity_id: 'ChEMBL activity ID',
    domain: 'Experimental domain tag from curation (in_vitro / unknown)',
    species: 'Species of the assay system',
  },
};
