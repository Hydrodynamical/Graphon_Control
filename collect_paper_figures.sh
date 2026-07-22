#!/bin/sh
# Collect the figures used by main5.tex into the flat folder figures/paper/
# (the folder to drag into Overleaf). Re-run after regenerating any experiment
# figures; the experiment mirror dirs (figures/e1_*, figures/e2_*) stay the
# archival source of truth.
set -e
cd "$(dirname "$0")"
mkdir -p figures/paper

# schematic / example figures (names unchanged)
for f in x0-histogram w1 w2 graphon-N graphon-2N; do
  cp "figures/$f.pdf" "figures/paper/$f.pdf"
done

# bounded-confidence experiment (subsec:exp-bc)
cp figures/e2_eps1_T10/opinion_traj.pdf     figures/paper/bc_opinion_traj.pdf
cp figures/e2_eps1_T10/state_evolution.pdf  figures/paper/bc_state_evolution.pdf
cp figures/e2_eps1_T10/kernel_strip.pdf     figures/paper/bc_kernel_strip.pdf

# Kuramoto experiment (subsec:exp-kuramoto)
cp figures/e1_k3_T20/r_panels.pdf           figures/paper/kur_r_panels.pdf
cp figures/e1_k3_T20/state_evolution.pdf    figures/paper/kur_state_evolution.pdf
cp figures/e1_k3_T20/kernel_strip.pdf       figures/paper/kur_kernel_strip.pdf
cp figures/e1_k3_T20/winding.pdf            figures/paper/kur_winding.pdf
cp figures/e1_k2_T12_sigma0p3/r_noise_panel.pdf figures/paper/kur_noise_panel.pdf

echo "figures/paper/ refreshed:"
ls figures/paper/
