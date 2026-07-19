# D5 allocator factorial ablation

## Purpose

D5 isolates two coordination choices while keeping the belief update, dynamic task manager,
mode screening, event trigger, resource constraints, commit-next execution, utility model,
and round cap unchanged.

| Variant | Broadcast bid | Update protocol |
|---|---|---|
| V00 | raw marginal | retain-and-release |
| V01 | raw marginal | full reconstruction |
| V10 | prefix-warped marginal | retain-and-release |
| V11 | prefix-warped marginal | full reconstruction |

## Registered domains

- Allocation pressure: six frozen D3 pressure conditions, 128 fresh seeds per condition.
- Scale: seven frozen D3 scale cells, 64 fresh seeds per cell.
- Four variants are evaluated with common random numbers in every scenario.
- Total: 1,216 scenarios and 4,864 method records.

The D5 generator namespaces and seed ranges are disjoint from D2--D4. No algorithm,
utility, physical parameter, round cap, condition, or seed may be changed after formal
execution is authorized.

## Outcomes

Primary outcomes are allocator cycle/round-cap incidence, legal commit rate,
allocation-stall rate, and final normalized realized utility. Secondary outcomes are
winner conflicts, rounds and message packets per successful commit, total planning time,
warping activations, and raw-prefix increases. A raw allocation objective from a failed
epoch is diagnostic only and is not treated as an executable allocation value.

Factorial effects are computed within registered strata:

\[
\Delta_W=\frac{(Y_{10}-Y_{00})+(Y_{11}-Y_{01})}{2},\quad
\Delta_F=\frac{(Y_{01}-Y_{00})+(Y_{11}-Y_{10})}{2},
\]

\[
\Delta_{WF}=Y_{11}-Y_{10}-Y_{01}+Y_{00}.
\]

Confidence intervals use paired bootstrap resampling of scenarios within stratum;
planning epochs are not independent samples.

## Frozen interpretation branches

- V01 approximately equals V11: full reconstruction is the dominant mechanism.
- V10 approximately equals V11: bid warping is the dominant mechanism.
- Only V11 is stable: the mechanisms interact synergistically.
- All variants are similar: allocator-kernel claims are reduced and the task lifecycle
  remains the main contribution.

Results are reported without effect-driven tuning or condition removal.
