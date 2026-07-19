# Dynamic Lifecycle D2 Manifest Design

## Status

This document freezes the proposed confirmation design only. It does not authorize D2 implementation or execution.

## Experimental matrix

- Eight unchanged cells: N∈{2,3}, M∈{3,5}, resource tier∈{tight,loose}.
- Equal cell weight: 1/8.
- 512 new scenarios per cell, 4096 scenarios total.
- Scenario indices 1000–1511 under a new `d2-generator-v1` namespace; no D0/D1 ID overlap.
- Ten unchanged methods, producing 40,960 expected terminal records.
- Canonical execution uses 22 workers. A frozen 16-scenario subset is replayed with workers 1 and 2.

## Sample-size basis

The design uses the eight D1 scenario-level paired P−B1m variances but does not use the D1 effect mean to choose cells or sample size. The sum of cell variances is 0.15403609. A factor of two inflates this pilot variance to protect against uncertainty from only 20 D1 pairs per cell.

For the equal-cell estimator, the approximate standard error at 512 scenarios per cell is:

\[
\sqrt{\frac{2\sum_c s_c^2}{8^2\times512}}=0.003066.
\]

The normal approximation requires 506 scenarios per cell for 90% power that the 95% CI lower bound exceeds zero when the true effect is 0.01 under the inflated variance; the design rounds this to 512.

The joint rule also requires the point estimate to be at least 0.01. If the true effect is exactly 0.01, the probability that an unbiased estimate exceeds 0.01 remains approximately 0.5 regardless of sample size. The design therefore does not claim 90% joint success probability at the boundary.

## Frozen analysis

- Primary: equal-cell paired P−B1m normalized-utility difference.
- Success: mean≥0.01, percentile bootstrap 95% CI lower bound>0, complete matrix, zero Gates.
- Bootstrap: 10,000 within-cell paired resamples, Type-7 linear quantiles, independent D2 namespace.
- Holm secondary family: P−B2, P−B3, P−B4, P−B5(4), P−B6.
- Sensitivity: P−B5(2), P−B5(8).
- P−CEX is reported separately.

Any missing, duplicate, extra, failed, non-finite, Gate, or replay-mismatched record makes confirmation `FAILED/INCOMPLETE`. Complete-case confirmation, automatic retry, seed replacement, and cell removal are forbidden.
