# Manuscript-ready data bundle

This directory is generated from frozen experiment artifacts by
`experiments/prepare_manuscript_data.py`. It does not rerun or modify any policy.

- `fig5_*`: confirmatory forest-plot data.
- `fig6_*`: optional-BDA ECDF, stratification, and utility decomposition.
- `fig7_*`: allocator stability, cycle strata, and workload.
- `fig8_*`: scale, utility, runtime, rounds, and message trade-offs.
- `fig9_*`: battlefield-structure and reachability heatmaps.
- `table5_*`: main lifecycle and information-action results.
- `table6_*`: exact-reference and robustness summaries.
- `table7_d5_*`: registered raw/warped x retain/rebuild dynamic factorial results.
- `relative_gain_summary.csv`: ratio-of-means relative improvements with paired
  or shared-seed-block bootstrap confidence intervals.
- `communication_accounting.csv`: separate public-event, screening, and CBBA
  communication. Screening is a reproducible dense full-fleet upper bound, not
  a measured network trace; CBBA packet counts come from runtime diagnostics
  where available.

The statistical unit is the independent scenario. Method records are paired
repeated evaluations of the same scenario and are not additional samples.
