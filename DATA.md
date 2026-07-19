# Data Contents and Scope

## Included Data

`results/manuscript_data/` contains the compact data products used to report the study results:

- paired and stratified utility contrasts;
- ratio-of-means relative improvements and bootstrap confidence intervals;
- BDA-use distributions and utility decomposition;
- allocator convergence, cycle, workload, message, and timing summaries;
- scalability and allocation-pressure results;
- battlefield-structure and reachability sensitivity matrices;
- centralized exact-reference gaps;
- the D5 raw/warped bidding factorial results;
- communication-accounting summaries;
- `inventory.json` and `validation_report.json`.

`results/dynamic_mainline/` contains the frozen D2--D5 scenario manifests and the matching execution metadata required by the manifest-driven runners.

## Excluded Data

The raw per-method scenario records and event logs occupy approximately 1.1 GB and are excluded from this repository to keep the public package lightweight. They can be regenerated from the included source code and frozen manifests. The repository also excludes all manuscript PDFs, LaTeX sources, figures, reviewer materials, caches, and binary tools.

## Statistical Unit

The independent scenario is the statistical unit. Evaluations of multiple methods on the same scenario are paired repeated measurements and are not counted as independent samples. Shared-seed-block bootstrap procedures preserve common-random-number dependence in the battlefield and reachability sensitivity experiments.

## Integrity Check

`results/manuscript_data/validation_report.json` records automated consistency checks for the compact data package. Its expected top-level status is `PASS`.
