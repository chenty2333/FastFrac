# FastFrac

**Fast edge recommendation for fractional spectral traces on graphs.**

This repository contains the implementation, experiment drivers, and raw
results for FastFrac.

Given a graph and a budget of `k` edges from a candidate pool, which edges
improve the graph most? Effective-resistance, network-coherence, and
log-determinant criteria each weight the Laplacian modes in one fixed way,
although an edge that helps short-time diffusion may matter less for
long-time relaxation. FastFrac maximizes the gain of the positive fractional
spectral trace

```math
F_s(S) = \mathrm{Tr}\big((I+\alpha L_S)^{s}\big), \qquad s \in (0,1),
```

where `L_S` is the Laplacian after adding edge set `S`. Smaller `s` gives
relatively more weight to low-eigenvalue modes, and the gain divided by `s`
tends to the log-determinant gain as `s → 0`. The cost is that each greedy
round needs matrix-function marginals for every candidate.

## Method

- **Scalar marginals.** Integration by parts writes each edge's rank-one
  marginal as a positive integral of shifted effective-resistance scalars.
- **Certified surrogate.** A fixed positive-weight Gauss–Jacobi rule gives a
  monotone submodular surrogate with a one-sided relative-error bound.
- **Reusable index.** The `M × q` initial resistance scalars (`M` candidates,
  `q` quadrature nodes) are reused for filtered candidate sets and smaller
  budgets while the graph, `s`, and `α` stay fixed.
- **Guaranteed selection.** Certified lazy greedy (CELF) and rank-one
  incremental maintenance (Inc) achieve a `(1 − 1/e − ε)` approximation
  under prescribed numerical error controls.

On ten heterogeneous multiscale graphs, selection with `s = 0.85` gives 6.79%
more heat-trace reduction than log-determinant selection at diffusion time
`τ = 0.2` and 10.14% less at `τ = 1`. On small instances both online paths
reach a mean 0.9993 of the exhaustive optimum. On eight SNAP networks (up to
4.0M vertices, 34.7M edges), CELF is an estimated 15–17× faster than naive
quadrature greedy; com-LiveJournal takes 2.4 hours on CPU versus 36.7 hours
projected for the baseline.

## Repository layout

```
fastfrac_cpu_benchmark.py              quadrature rule, Naive, CELF, Inc
unified_kernel.cpp, unified_backend.py compiled shifted-Lanczos solver shared by all methods
fastfrac_final_unified_experiments.py  driver: synthetic and medium SNAP experiments
fastfrac_large_snap.py                 driver: million-vertex SNAP runs
build_fastfrac_unified_report.py       rebuilds fastfrac_final_unified_data.md
other *.py                             experiment groups used by the drivers (flat; modules import each other)
fastfrac_unified_results/              synthetic and medium SNAP results: per-run JSON, audits, traces
fastfrac_large_results/                million-vertex SNAP results and log
fastfrac_cpu_results/,
fastfrac_revision_results/             fixed inputs: generated graphs, candidate pools, exhaustive optima
fastfrac_realworld_data/               SNAP download script and SHA-256 sums
figure_sources/                        scripts that draw the paper's figures
provenance/                            record of one interrupted and resumed run
fastfrac_final_unified_data.md         summary tables of every measurement
METHODS.md                             timing contract, baselines, audit definitions
paper/FastFrac.pdf                     the paper
```

## Quick start

CPU only. Requires Linux, Python 3.12, and a C++17 compiler with OpenMP; the
drivers compile the C++ kernels on first use. Paper timings used an AMD EPYC
9V74 (8 threads) for the main experiments and an Intel Xeon Platinum 8481C
(18 threads) for the large SNAP runs; com-LiveJournal needs about 27 GB of
memory.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
sh fastfrac_realworld_data/fetch.sh                                    # SNAP inputs, verified by SHA-256

python fastfrac_final_unified_experiments.py --threads 8 --repeats 5   # all synthetic + medium SNAP experiments
python fastfrac_large_snap.py --graphs com-DBLP --threads 18           # one large SNAP network
python build_fastfrac_unified_report.py                                # rebuild fastfrac_final_unified_data.md
```

`--groups routes science main multiscale curves scaling` runs a subset of the
experiments; the full suite takes many hours. [`METHODS.md`](METHODS.md)
describes what is timed and how.

## Companion project

[POLAR](https://github.com/chenty2333/POLAR) studies the negative-power trace
`Tr((L + σI)^{-s})`, which is not submodular and needs a different marginal
oracle.
