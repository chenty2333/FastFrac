# Methods and measurement protocol

Details behind the CPU experiments in `fastfrac_unified_results/`. The summary of all
measurements is in `fastfrac_final_unified_data.md`.

## Backend and timing contract

- Every deterministic selector uses `unified_kernel.cpp` through
  `unified_backend.py`. Scalar resistance evaluation, projected inverse-norm
  evaluation, full shifted vectors, and selected-edge update vectors are all
  compiled. The same maximum OpenMP thread count is used throughout. BLAS is
  restricted to one thread.
- Independent incidence right-hand sides are batched for CSR memory reuse.
  A single online right-hand side does not create eight independent solves;
  equal thread caps do not imply identical attainable parallelism for different
  algorithms. Logical SpMV and padded executed-column counts are retained
  separately.
- Each method and each repetition builds a fresh initial scalar resistance
  table. The main performance suite uses one common boundary: Phase 1 ends when
  the candidate-by-shift resistance/error table is complete. Phase 2 includes
  scores, table copies, heap setup, all selected rounds, rank-one updates,
  residual checks, and additional rechecks. Reported total time is Init plus
  Online. No offline table is reused between independently timed repetitions.
- Compilation, graph generation, file loading/writing, and independent
  reference evaluations are outside individual selector timings. Their costs
  remain part of the experiment's overall wall time where recorded. Means and
  standard deviations use five measured runs and sample SD (`ddof=1`).
- `Dynamic-Logdet` is the same CELF controller on the single unshifted system
  (`shifts=[0]`, `weights=[1]`, relative tolerance 10^-7), not a separately
  optimized implementation. A regularized logdet weight is
  `1/(1+alpha*lambda_L)`, which remains finite at zero Laplacian eigenvalue.
- BA-50k Naive executes all 20 rounds in every repetition. BA-100k Naive alone
  is labelled as a measured first-round projection. Projected times are never
  called completed-run measurements or used to invent five-run variance.

The scientific tables' selector timings also use five fresh complete runs.
The independently audited heat, marginal errors, structural metrics, and true
objective evaluations are excluded from those selector times.

### Recorded execution interruption and resume

The supplied measurements include a transport interruption during the
BA-100k stage. Completed per-repetition checkpoints were retained, and only
the missing repetitions and five Naive first-round measurements were resumed.
An interrupted partial trial contributes no timing, solver count, or selection
to the final summaries. The five completed timings therefore need not come
from one uninterrupted process session.

An empty BA-100k graph-cache file was regenerated with the recorded NetworkX
and NumPy generators and fixed seeds. For CELF, Inc, and Dynamic-Logdet, all
2,000 initialization iteration counts and the selected sequence matched the
already completed traces. The C++ numerical kernels and thread settings were
unchanged; graph-cache writes were made atomic. The source hashes, interruption
scope, and trace-match results are recorded in
`provenance/execution_interruption.json`. The artifact preserves
this provenance record and the completed observations, not the incomplete
trial as a measured run.

## Meaning of the marginal-oracle ablation

The oracle-ablation and main-performance tables are separately timed experiments.
The ablation controller includes per-oracle trace/recording wrappers consistently
across its routes; its Small-BA IBP runtime therefore need not equal the main
performance controller's Small-BA CELF runtime. All recorded measurements are
retained rather than substituting one controller's times for another.

`IBP-Scalar`, `Raw-Loewner-Proj`, and `Raw-Loewner-FullVec` use the same shifted
quadrature nodes, matrix representation, recurrence backend, stopping
tolerance, and thread cap. Raw-Proj obtains the inverse-square quadratic form
from a differentiated shifted LDL recurrence, equivalent in exact arithmetic
to a norm in the Lanczos tridiagonal space. Raw-FullVec explicitly maintains
and reconstructs the shifted n-dimensional solutions and checks true
residuals. A runtime difference between these two routes is not attributed to
a mathematical inability to project the raw formula.

CRN-SLQ uses its compiled C++/OpenMP kernel with the same compiler flags and
thread cap. One fixed Rademacher probe family is shared across candidate
trials, graph states, and paired exhaustive/lazy runs. The 32 probes are a prefix
of the 128-probe family; each probe uses 15 Lanczos steps. Repeating the fixed
workload five times measures timing variation, **not variation across random
probe seeds**. Incidence RHS counts and probe RHS counts are different units;
logical SpMV counts are additionally reported.

The paired CRN-SLQ-CELF versus CRN-SLQ-Greedy audit distinguishes wrong lazy
decisions under a shared estimator from error relative to the true fractional
trace. Cached values falling below the true fractional marginal alone do not
prove that CELF pruned incorrectly under its own fixed-probe objective. All
observed selections and oracle traces are retained, including observations
that do not support an anticipated failure or speedup.

## Scientific controls and definitions

### Quadrature audit

The q sweep is `{4,8,12,16,24,36,54}`. It covers Small-BA at alpha = 1, s = .5 and the
locked Multiscale-SBM at alpha = 50, s = .1 and s = .5. A fresh full Exact-Eig greedy is
computed in each case. For every state on its sequence, every remaining
candidate is audited against dense spectral values. Path-wide maximum and
mean relative error are reported alongside the initial-state maximum. The
analytic one-sided bound and ordered/set agreement are separate quantities.
Very high q values reach a floating-point dense-reference floor; the analytic
exact-arithmetic bound is not a bound on that reference's rounding error.

### External heat utility and full scale scan

The original SBM and the prespecified heterogeneous graph are reused without
a seed/parameter search. Heat utility is

`Delta H(tau) = tr(exp(-tau*L0)) - tr(exp(-tau*LS))`

at tau = .05, .2, 1, evaluated from complete dense Laplacian eigenspectra after
selection. It is independent of the fractional objective used to select edges.
Both positive and negative changes relative to Dynamic-Logdet are reported.

The heterogeneous scan contains every combination of
`s={.05,.10,.25,.50,.70,.85,.95}` and `alpha={1,10,50,200}`. It reports all 28
points, both weighted and unweighted endpoint-degree sums, and both local
structure metrics. A q = 54 selection check accompanies every timed q = 12 result.
The low-four-mode derivative diagnostic additionally covers both graphs over
the same grid. The lowest four modes refer to `A=I+alpha*L`, including its
constant mode; the latter has zero incidence-vector contribution.

All structure statistics use the unchanged base graph and all k selected
edges. `N2` is `|B2(u) intersection B2(v) minus {u,v}|`, where B2 contains vertices
at graph distance at most 2, ignoring weights. `E_loc` counts undirected base
edges induced by `N(u) union N(v) union {u,v}`. Block 0 is fixed as the lowest
conductance module in the heterogeneous graph; blocks 2/3 are the highest
conductance modules. Allocation labels are fixed before selection.

The historical multiscale `Degree` baseline ranks the sum of **weighted**
endpoint degrees. The additional `Degree-Unweighted` row exposes the effect
of this choice. Random controls use 10 fixed-seed uniform k-subsets; raw per-draw
gains are retained.

### Exact OPT and adaptive verification

The 20 locked n = 25, M = 15, k = 4 instances reuse their previously exhaustively verified
OPT values (27,300 total subsets), after graph and candidate identity checks.
All method selections and exact eigenvalue gains are recomputed. The OPT file's
SHA256 and this reuse are explicitly recorded; the historical enumeration is
not relabelled as newly timed enumeration.

Adaptive and Uncertified Inc use identical initial estimates and selected-edge
vector kernels. Uncertified chooses cached point maxima and performs no extra
verification. Ordered sequence agreement, unordered set agreement, and true
gain loss are all distinct reported metrics.

Every adaptive rejection records the selected prefix, candidate, competitor,
residual interval, and independent high-order marginal comparison. A
structural tie requires an explicit exact current-adjacency automorphism;
merely observing a gap below 10^-14 is insufficient. All 21 ca-GrQc rechecks have
such certificates. The final graph from its tighter-solve reference is exactly
permutation-similar to the observed augmented graph. Both the literal tighter
reference matches and the nominal-tolerance reference fields are retained;
neither is silently replaced after seeing the outcome. These are floating-point
residual monitors with explicit padding, not directed-rounding certificates
for every arithmetic operation.

## Inputs

The JSON inputs include original topology, candidate ordering, seeds, and
historical metadata. Only their fixed inputs, saved Exact-Eig reference where
identified, and explicitly identified exact OPT values are reused. New timing
tables read only the new result files. The original ordinary-graph objective
quality controls at s = .2, .5, .8 also remain in these input records as
supplementary comparisons; their historical solver timings are never mixed
into a unified-backend performance ratio.

The SNAP files are original downloaded source files. The measured simple
undirected view retains all endpoint vertices, removes self-loops, and
deduplicates edges; disconnected components are retained. Raw-source hashes
and preprocessing counts are recorded. Dataset source pages are:

- https://snap.stanford.edu/data/email-Eu-core.html
- https://snap.stanford.edu/data/ca-GrQc.html
- https://snap.stanford.edu/data/ca-CondMat.html

Generated BA and SBM-4k edge/candidate arrays are stored in
`fastfrac_unified_results/graphs/`. The graph generators and fixed seeds can
regenerate them. Preserve the full `fastfrac_unified_results/` tree, including
per-repetition performance JSON, scientific audit JSON, marginal oracle traces,
and manifests. Prior failed or interrupted legacy experiments are not part of
this repository's evidence.
