# D2 Diagnostic and D3 External-Validation Design

## Status and separation from D2

The current P algorithm, its D2 parameters, and the D2 confirmation data are
frozen. D2 remains the only primary confirmation experiment. The analyses and
experiments below may explain or externally probe P, but may not retroactively
change the D2 verdict or motivate tuning on D2 seeds.

There are two independent work streams:

1. Post-hoc explanatory analysis of immutable D2 artifacts, plus a trace-only
   deterministic replay whose terminal records and events must match D2 exactly.
2. Registered D3 external validation under entirely new scenario IDs and random
   namespaces. D3 contains scale, continuous-belief, model-mismatch,
   utility-preference, small-scale CBBA-isolation, and medium-scale
   allocation-pressure suites. CEX is restricted to the small isolation suite
   because its exact epoch search is not a scalable baseline.

## Frozen objective semantics

The implemented realized utility is

\[
U=V_{\mathrm{destroyed}}-C_{\mathrm{service}}
  -C_{\mathrm{distance}}-C_{\mathrm{ammo}},
\]

where destruction rewards and service costs are time-discounted. Mission time is
also a hard horizon constraint. Makespan is reported as an outcome and is not an
additional soft penalty.

## D2 explanatory diagnostics

All method summaries report utility components, makespan, belief score, action
counts, handoffs, replans, and CBBA rounds. Every P-minus-baseline paired
difference reports mean, median, 5/25/75/95 percentiles, win/tie/loss, ECDF data,
and the exact utility decomposition

\[
\Delta U=\Delta V-\Delta C_s-\Delta C_d-\Delta C_m.
\]

The BDA analysis compares paired P and B4 outcomes and stratifies scenarios by P's
BDA count, resource tier, fixed prior-probability bins, and attacks previously
made against the same target. A trace replay reconstructs belief immediately
before and after each BDA, entropy change, and the next same-target action. These
are conditional mechanism diagnostics, not per-event causal estimates.

The periodic analysis compares B5(2), B5(4), and B5(8): completion-to-grid wait,
action and replan counts, CBBA rounds, resource consumption, selected path score,
and no-commit decisions. This distinguishes beneficial waiting from unnecessary
delay without assuming either mechanism in advance.

## D3-A: scaling

Seven methods are retained: P, B1m, B4, B5(4), B6, SCBBA, and DVCBBA. Seven scale cells separate
swarm scaling at constant M/N=2.5, workload scaling at fixed N=6, and one joint
stress case:

- (N,M)=(4,10),(6,15),(8,20);
- (N,M)=(6,10),(6,20),(6,30);
- (N,M)=(8,30).

There are 96 new scenarios per cell. To avoid confounding scale with increasing
spatial density, target and UAV positions are uniform in a square with half-width
\(6\sqrt{M/5}\). Resources follow frozen load-aware rules:

\[
A_i=\lceil1.2M/N\rceil,
\quad T_{\max}=\lceil12\sqrt{M/5}+8M/N\rceil,
\quad R_i=1.25T_{\max}.
\]

Scale, mismatch, and preference suites retain the four D2 belief archetypes so
that each changes only its stated factor. Only the continuous-belief suite changes
the prior distribution.

Results include normalized utility, initially alive value neutralized, invalid
attack rate, action counts, planning CPU time, replan count, and CBBA rounds.

## D3-B: continuous joint beliefs

The original eight resource/size cells receive 128 new scenarios each. Each
target belief is uniform on the four-state simplex,
\(b\sim\mathrm{Dirichlet}(1,1,1,1)\), implemented deterministically as normalized
\(-\log U_k\). Ground truth is then drawn from that belief, preserving calibration
while removing the four-archetype restriction.

## D3-C: model mismatch

The planner always uses the frozen nominal D2 likelihoods and attack probabilities.
The environment uses one of nine one-factor-at-a-time conditions: nominal; sensor
quality at -20%, -10%, +10%, +20%; or attack probability at -20%, -10%, +10%,
+20%. There are 64 new paired scenarios for every condition and each of the eight
original cells.

For positive sensor quality q,

\[
O^{true}=(1-q)O^0+qI.
\]

For negative quality q,

\[
O^{true}=(1-|q|)O^0+|q|U,
\]

where every column of U is (0.5,0.5). This preserves column stochasticity. Attack
miscalibration uses \(p^{true}=\operatorname{clip}(p^0(1+\delta),0,1)\). Initial
states and semantic uniform draws are shared across mismatch conditions.
Condition labels appear in scenario IDs but are excluded from semantic random keys.

## D3-D: utility preference sensitivity

Three profiles change planning and evaluation weights together while leaving
physics, action durations, and hard constraints unchanged:

- value priority: service, distance, and ammo cost multipliers all 0.5;
- balanced: all multipliers 1;
- resource saving: service 1.5, distance 3, and ammo 4.

Every profile uses 64 new scenarios in each original cell with cross-profile CRN.
This is preference sensitivity, not model-mismatch robustness.

## D3-E: CBBA isolation

The original eight small cells receive 64 new scenarios each. Four methods are
compared: P, SCBBA, DVCBBA, and CEX. SCBBA is the literature-facing static
one-shot vanilla CBBA baseline. DVCBBA shares P's belief update, task manager,
mode screening, event clock, resource constraints, fixed task set within an
epoch, and commit-next execution; only the allocator changes from
Johnson-warped full reconstruction to vanilla raw marginal bidding with bundle
retention and first-lost-task suffix release.

Every allocator epoch reports convergence status, cycle or round-cap status,
winner conflicts, rounds, message packets and scalar entries, allocation
objective, fixed-screened-task exact gap, and all-mode CEX gap. Final utility is
reported separately. A DVCBBA cycle or timeout is a measured baseline outcome,
not an infrastructure failure; missing rows, exceptions, non-finite values, and
simulator gates remain failures.

## D3-F: allocation pressure

To test whether the small \(N\leq3,M\leq5\) domain compresses allocator
differences, a fixed \(N=6,M=15\) cell receives 64 new scenarios in each of six
pre-frozen conditions: scaled reference, shared high-value demand, two compact
target clusters, tight resources, long routes, and combined stress. Methods are
P, SCBBA, DVCBBA, and B6. Exact search is excluded at this scale.

The primary mechanism variable is positive-pair density: the number of positive
agent-task bids divided by the available agent-target pairs across allocator
epochs. It is reported together with conflicts, convergence, rounds, messages,
allocation objective, path/resource outcomes, and final utility. Conditions may
not be removed, reweighted, or retuned after effects are observed. This suite
tests an applicability hypothesis; it is not a license to optimize P against the
new seeds.

Implementation must introduce separate planner and environment model objects. The
environment object controls generated observations and physical attack outcomes;
the frozen nominal planner object controls Bayesian updates, attack prediction,
and bids. This separation is an experiment interface, not a change to P's control
logic.

## Analysis and timing

D3 is registered external validation, not a second opportunity to optimize P.
All registered cells, conditions, and profiles are reported. Paired within-stratum
bootstrap confidence intervals use 10,000 Type-7 replicates. D2 remains the only
primary pass/fail effect claim.

Planning process CPU time is captured per call and summed per episode. A separate
serial scale benchmark uses the first and last scenario in each scale cell, one
warm-up and three measured repetitions, and reports median and p95. Timing does
not enter utility or completeness verdicts.

Any missing, duplicate, extra, failed, non-finite, Gate, or replay-mismatched
record makes the corresponding D3 execution incomplete. Automatic retry, seed
replacement, condition removal, and effect-driven tuning are forbidden.

Before authorization, a disjoint effect-blind structural smoke uses no formal D3
scenario ID. It checks all transforms, scale cells, continuous-belief validity,
CRN contracts, action-region coverage, and runtime feasibility. No utility contrast
may be computed. If it fails, the design must be revised under a new manifest
version before any formal D3 execution.
