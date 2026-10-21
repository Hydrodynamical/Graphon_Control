# Reproducing the existing numerical appendix

The original saved results are bundled in `data/appendix/` (about 17 MB).
`appendix_manifest.json` selects the five runs and historical figures by path
and SHA-256. Reproduction uses saved arrays only: no training, checkpoint
loading, or integration of the dynamics.

| Manifest key | Original run | Appendix result |
| --- | --- | --- |
| `bc` | `e2_eps1_T10` | bounded-confidence table and two figures |
| `k1` | `e1_k1_T5` | depth-comparison table, k=1 |
| `k2` | `e1_k2_T12` | depth-comparison table and noiseless reference, k=2 |
| `k3` | `e1_k3_T20` | headline Kuramoto table and two figures, k=3 |
| `noise` | `e1_k2_T12_sig03` | noisy Kuramoto ensemble panel |

Each run includes its unchanged `config.json`, `metrics.json`, and
`trajectories.npz`. Checkpoints and training histories are unnecessary for
this workflow and are omitted. The reference folder contains only the five
numerical figure PDFs; manuscript sources are not included.

All runs record N=64, dt=0.05, seed 0 and training jitter 0.1. Deterministic
evaluations use the exact initial profiles stated in the appendix. BC and the
k=1/2 Kuramoto runs use 2,000 Adam updates; k=3 uses 3,000. The archived results
match the rounded entries in the three appendix tables.

## Regenerate from saved arrays

Install the package from the repository root as described in
[DEVELOPMENT.md](DEVELOPMENT.md), then run:

```sh
python reproduce_appendix.py \
  --manifest appendix_manifest.json \
  --output out/appendix-regenerated
```

The command checks input hashes before creating a new output directory and
writes five figure PDFs plus:

- `tables.csv`: recomputed statistics for both paper arms in all four deterministic runs.
- `verification.json`: 131 checks against archived metrics and diagnostic arrays, plus objective rounding checks.
- `noise_summary.json`: retained ensemble statistics and their limitations.
- `provenance.json`: input and output hashes for this export.

Diagnostics and objectives are recomputed by arithmetic on archived state and
kernel arrays. Small differences from stored metrics arise when float64
reductions replace the original float32 reductions. The tests also compare the
other rounded table entries with the bundled results. Existing output
directories are refused; choose a new name for another export. Generated
outputs under `out/` are ignored by Git.

## Recover the exact historical PDFs

```sh
./collect_paper_figures.sh appendix_manifest.json
```

This copies the selected historical PDFs into `out/appendix-reference/`.
The reference PDFs match the manuscript assets byte-for-byte. The original
BC run's kernel PDF differed from its figure mirror only in one creation-date
byte; the bundle retains the matching mirror. Regenerated figures match the
reference renders at 100 dpi in the tested environment. PDF creation dates and
rendering-library versions can change file bytes; use the reference copies
when identical PDF files are required.

The standalone combined-panel exporter is also available:

```sh
python export_appendix_panel.py \
  --run data/appendix/runs/e1_k3_T20 \
  --output out/appendix-panel-review/kur_rw_panel.pdf
```

## Archive limitation

The noisy run retains ensemble means and 10–90% quantiles, not the 32 individual
paths. Its panel and terminal mean r=0.974 are reproducible from those aggregates.
The statement that every individual path has terminal winding zero cannot be
independently checked from the retained files. Missing paths are not synthesized.
