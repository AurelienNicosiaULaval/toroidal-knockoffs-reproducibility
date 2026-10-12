# Toroidal knockoffs: code and simulation records

Code and retained numerical outputs supporting **Consistency and shared estimation error in toroidal knockoffs**, by Aurélien Nicosia. This repository is reproducibility material for a manuscript prepared for *Statistical Papers*. It does not indicate journal submission or acceptance.

## Reproduce the numerical checks

Python is the executed reference (Python 3.13). From a fresh clone:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-sp08.txt
.venv/bin/python run_project.py check
.venv/bin/python run_project.py rebuild
.venv/bin/python run_project.py replay
```

`check` verifies the manifest and all 24,000 individual JSON records, runs the reference tests, recomputes the SP06 selection summaries and paired differences, separately reaggregates the 7,000 v5/v6 datasets, and checks nine saved SP14 outputs, including three QR fits from small raw designs. `rebuild` additionally generates the figure from the audited SP06 summaries. Neither command reruns the simulation campaigns. `replay` regenerates one recorded SP06 seed and compares scores, p-values and selections.

Every execution writes a new timestamped folder and receipt under `results/minimal_runs/`. The individual records are grouped byte-for-byte in `individual_records.tar.gz`; commands safely unpack and verify them without replacing changed records. Experimental versions and adverse outcomes remain distinct. The manuscript and supplementary proofs are supplied to the journal separately; this public runner reconstructs numerical tables and figures, not their PDFs.

## Content and scope

- `src/`, `tests/`, `scripts/`: Python reference implementation, frozen producers and independent numerical checks.
- `configs/`, `docs/`: configurations, seeds and scientific protocols.
- `results/`: preserved aggregates, execution receipts, source snapshots and selected raw numerical inputs.
- `compact_scores/`: losslessly stacked SP06 p-values and knockoff scores, with provenance.
- `individual_records.tar.gz`: 24,000 retained individual records, including 8,000 matched reanalyses and the 7,000 historical v5/v6 datasets.
- `provenance/`: original file hashes and the inventory of omitted large arrays; `MANIFEST_SHA256.json` hashes the files distributed here.

The checks recompute 173,000 SP06 selection decisions and 36,000 v5/v6 decisions. Three small SP12 datasets permit independent QR checks; all nine SP14 coefficient/conditional-law outputs are retained. Six larger SP12 designs, SP09 raw covariates, SP11 raw Gaussian noise and SP06 pilot derivative/covariance arrays are omitted. Their producers, seeds and original hashes are retained. The compact checks do not refit the omitted inputs or verify every original covariance array. Historical receipts retain their original scope; they are not claims that their full experiments were rerun for this release.

The full 1.374 GB parent archive remains with the author. This approximately 52 MB package is for reaggregation, selected numerical checks and seed replay. All included empirical inputs are simulated; no participant-level dataset or third-party paper is distributed.

## Regenerate omitted calculations

In a new working copy, the original producers under `scripts/statistical_papers/` can regenerate diagnostic inputs: `sp09_loading_order_check.py`, `sp11_gaussian_risk_check.py`, and `sp12_growing_check.py`. For SP06, use fresh output directories with `sp06_selection.py --config configs/statistical_papers/sp06_1_selection.json`, followed by the frozen SP06.2 and SP06.4 configurations. `sp06_harmonics_hc.py` performs the matched SP06.3/SP06.4-HC reanalyses. Platform-dependent floating point differences can prevent bitwise replay; investigate them without overwriting retained records.

## Version and attribution

Scientific version: SP14. Public package: `v1.0.0`. The release tag identifies the version cited by the manuscript. `CITATION.cff` identifies this software/data package; no DOI has been assigned.

The model-X, latent-redraw, torus graph, Metro and robustness frameworks remain attributed to their original authors in the associated manuscript. The project-specific Metro implementation follows the published sequential recursion of Bates, Candès, Janson and Wang (2021); no upstream knockpy source is redistributed here.

Copyright remains with the respective rightsholders. Public access does not assign authorship or grant an additional blanket software/data license. No third-party software is vendored; installed dependencies retain their own licenses.
