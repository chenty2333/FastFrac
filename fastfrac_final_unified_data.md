# FastFrac: unified CPU results

All new performance rows use the same compiled C++17/OpenMP backend with a maximum of 8 worker threads and one BLAS thread. Each algorithm starts from the same base graph and candidate bytes in five fresh timing repetitions; the tables retain the completed repetition counts. Standard deviations are sample SDs (ddof=1). Compilation, graph generation, file I/O, and independent accuracy evaluations are excluded from individual optimization times and recorded separately where applicable. Every reported speedup is computed from these new measurements.

For the performance runs, Phase 1 computes the initial candidate-by-shift resistance/error table. Phase 2 includes scoring, table copies, heap setup, every accepted edge, cache maintenance, true-residual checks, and adaptive rechecks. Each method and repetition pays for its own freshly measured initialization. Dynamic-Logdet is CELF on the single unshifted system (t=0), with the same tolerance and backend. One incidence RHS means one shared traversal for all shifts. SLQ probe-RHS counts have a different meaning and are labelled separately.

The scientific sweeps use the same numerical kernels but retain their existing controller timing boundary, which can include initial scoring in the initialization timer. Compare **total times only** across those scientific sweeps. All Phase-1/Phase-2 comparisons here and in the paper use the strict scalar-table boundary in the `performance/` measurements.

BA-50k Naive below executes all 20 rounds in every repetition. BA-100k Naive measures its first round five times and projects the full-run time; its projected total is explicitly separate from completed measurements.

## 1. Unified main performance: five repetitions

| Dataset | n | m | M | k | Method | Runs | Init (s), mean ± SD | Online (s), mean ± SD | Total (s), mean ± SD | Online RHS | Total RHS | Mean h/RHS | Krylov SpMV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Small-BA | 400 | 1,191 | 200 | 10 | CELF | 5 | 0.002475 ± 0.000233 | 0.005673 ± 0.000549 | 0.008148 ± 0.000532 | 11 | 211 | 30.863 | 6512 |
| Small-BA | 400 | 1,191 | 200 | 10 | Inc | 5 | 0.002576 ± 0.000400 | 0.008239 ± 0.000519 | 0.010815 ± 0.000910 | 10 | 210 | 30.862 | 6481 |
| Small-BA | 400 | 1,191 | 200 | 10 | Dynamic-Logdet | 5 | 0.002616 ± 0.000297 | 0.006101 ± 0.000429 | 0.008717 ± 0.000655 | 11 | 211 | 31.090 | 6560 |
| Small-BA | 400 | 1,191 | 200 | 10 | Naive | 5 | 0.002577 ± 0.000235 | 0.025803 ± 0.003951 | 0.028380 ± 0.004143 | 1755 | 1955 | 30.776 | 60168 |
| Medium-WS | 2,500 | 10,000 | 1000 | 20 | CELF | 5 | 0.057469 ± 0.006306 | 0.035571 ± 0.000605 | 0.093040 ± 0.006826 | 22 | 1022 | 24.826 | 25372 |
| Medium-WS | 2,500 | 10,000 | 1000 | 20 | Inc | 5 | 0.059510 ± 0.002398 | 0.056761 ± 0.003180 | 0.116271 ± 0.003883 | 20 | 1020 | 24.825 | 25322 |
| Medium-WS | 2,500 | 10,000 | 1000 | 20 | Dynamic-Logdet | 5 | 0.063581 ± 0.004859 | 0.036324 ± 0.000610 | 0.099904 ± 0.004720 | 22 | 1022 | 25.002 | 25552 |
| Medium-WS | 2,500 | 10,000 | 1000 | 20 | Naive | 5 | 0.060735 ± 0.000572 | 1.121950 ± 0.035083 | 1.182685 ± 0.034826 | 18810 | 19810 | 24.815 | 491579 |
| Large-PowerCluster | 10,000 | 39,976 | 3000 | 30 | CELF | 5 | 1.897575 ± 0.143228 | 0.577750 ± 0.024859 | 2.475325 ± 0.154125 | 31 | 3031 | 69.696 | 211249 |
| Large-PowerCluster | 10,000 | 39,976 | 3000 | 30 | Inc | 5 | 2.006533 ± 0.286359 | 0.809268 ± 0.034517 | 2.815801 ± 0.313107 | 30 | 3030 | 69.697 | 211182 |
| Large-PowerCluster | 10,000 | 39,976 | 3000 | 30 | Dynamic-Logdet | 5 | 2.050627 ± 0.183101 | 0.622710 ± 0.031821 | 2.673337 ± 0.205041 | 32 | 3032 | 71.572 | 217007 |
| Large-PowerCluster | 10,000 | 39,976 | 3000 | 30 | Naive | 5 | 2.233277 ± 0.403580 | 57.494934 ± 3.596950 | 59.728211 ± 3.636266 | 86565 | 89565 | 69.675 | 6240468 |
| email-Eu-core | 1,005 | 16,064 | 2000 | 20 | CELF | 5 | 0.397701 ± 0.039859 | 0.220937 ± 0.009022 | 0.618638 ± 0.045514 | 32 | 2032 | 73.359 | 149066 |
| email-Eu-core | 1,005 | 16,064 | 2000 | 20 | Inc | 5 | 0.398406 ± 0.027752 | 0.169428 ± 0.005106 | 0.567834 ± 0.025241 | 20 | 2020 | 73.317 | 148101 |
| email-Eu-core | 1,005 | 16,064 | 2000 | 20 | Dynamic-Logdet | 5 | 0.399679 ± 0.054195 | 0.375976 ± 0.016899 | 0.775655 ± 0.058985 | 52 | 2052 | 75.910 | 155768 |
| email-Eu-core | 1,005 | 16,064 | 2000 | 20 | Naive | 5 | 0.398439 ± 0.046054 | 7.781771 ± 0.855543 | 8.180210 ± 0.885313 | 37810 | 39810 | 77.203 | 3073461 |
| ca-GrQc | 5,242 | 14,484 | 2000 | 20 | CELF | 5 | 0.527213 ± 0.031764 | 0.212185 ± 0.006998 | 0.739398 ± 0.025147 | 44 | 2044 | 55.251 | 112933 |
| ca-GrQc | 5,242 | 14,484 | 2000 | 20 | Inc | 5 | 0.532153 ± 0.023906 | 0.287190 ± 0.010478 | 0.819343 ± 0.020640 | 41 | 2041 | 55.325 | 112918 |
| ca-GrQc | 5,242 | 14,484 | 2000 | 20 | Dynamic-Logdet | 5 | 0.494031 ± 0.047094 | 0.217977 ± 0.008850 | 0.712008 ± 0.040432 | 44 | 2044 | 56.905 | 116314 |
| ca-GrQc | 5,242 | 14,484 | 2000 | 20 | Naive | 5 | 0.505194 ± 0.034256 | 9.519924 ± 0.563860 | 10.025118 ± 0.548231 | 37810 | 39810 | 55.855 | 2223571 |
| ca-CondMat | 23,133 | 93,439 | 2000 | 20 | CELF | 5 | 5.945081 ± 1.072215 | 1.061589 ± 0.039289 | 7.006671 ± 1.106617 | 22 | 2022 | 84.983 | 171835 |
| ca-CondMat | 23,133 | 93,439 | 2000 | 20 | Inc | 5 | 5.587642 ± 1.136732 | 1.361866 ± 0.051423 | 6.949508 ± 1.145264 | 23 | 2023 | 84.942 | 171837 |
| ca-CondMat | 23,133 | 93,439 | 2000 | 20 | Dynamic-Logdet | 5 | 6.076991 ± 0.836476 | 1.065813 ± 0.035036 | 7.142804 ± 0.869667 | 23 | 2023 | 88.854 | 179752 |
| ca-CondMat | 23,133 | 93,439 | 2000 | 20 | Naive | 5 | 4.917813 ± 0.881174 | 99.047736 ± 7.623340 | 103.965548 ± 8.123176 | 37810 | 39810 | 85.368 | 3398482 |
| BA-50k | 50,000 | 149,991 | 2000 | 20 | CELF | 5 | 18.139624 ± 1.262302 | 1.997086 ± 0.042673 | 20.136709 ± 1.226080 | 19 | 2019 | 86.371 | 174383 |
| BA-50k | 50,000 | 149,991 | 2000 | 20 | Inc | 5 | 19.634734 ± 2.268870 | 2.832869 ± 0.056391 | 22.467603 ± 2.276472 | 20 | 2020 | 86.370 | 174467 |
| BA-50k | 50,000 | 149,991 | 2000 | 20 | Dynamic-Logdet | 5 | 17.225437 ± 2.679200 | 2.107880 ± 0.072555 | 19.333317 ± 2.742785 | 19 | 2019 | 89.521 | 180742 |
| BA-50k | 50,000 | 149,991 | 2000 | 20 | Naive | 5 | 18.998300 ± 5.149300 | 325.818957 ± 24.711500 | 344.817257 ± 29.086626 | 37810 | 39810 | 86.383 | 3438901 |
| BA-100k | 100,000 | 299,991 | 2000 | 20 | CELF | 5 | 69.338192 ± 3.014652 | 4.930187 ± 0.176946 | 74.268379 ± 2.912950 | 19 | 2019 | 105.792 | 213594 |
| BA-100k | 100,000 | 299,991 | 2000 | 20 | Inc | 5 | 66.825811 ± 2.982562 | 7.571470 ± 0.306833 | 74.397281 ± 3.265670 | 20 | 2020 | 105.791 | 213697 |
| BA-100k | 100,000 | 299,991 | 2000 | 20 | Dynamic-Logdet | 5 | 68.565608 ± 1.751576 | 5.128553 ± 0.103880 | 73.694161 ± 1.840424 | 19 | 2019 | 110.868 | 223842 |

| Dataset | Method | Total speedup vs Naive | Online speedup vs Naive | CELF/Inc identical ordered sequence |
| --- | --- | --- | --- | --- |
| Small-BA | CELF | 3.483 | 4.548 | True |
| Small-BA | Inc | 2.624 | 3.132 | True |
| Small-BA | Dynamic-Logdet | 3.256 | 4.229 | True |
| Medium-WS | CELF | 12.712 | 31.541 | True |
| Medium-WS | Inc | 10.172 | 19.766 | True |
| Medium-WS | Dynamic-Logdet | 11.838 | 30.888 | True |
| Large-PowerCluster | CELF | 24.129 | 99.515 | True |
| Large-PowerCluster | Inc | 21.212 | 71.046 | True |
| Large-PowerCluster | Dynamic-Logdet | 22.342 | 92.330 | True |
| email-Eu-core | CELF | 13.223 | 35.222 | True |
| email-Eu-core | Inc | 14.406 | 45.930 | True |
| email-Eu-core | Dynamic-Logdet | 10.546 | 20.698 | True |
| ca-GrQc | CELF | 13.558 | 44.866 | True |
| ca-GrQc | Inc | 12.236 | 33.149 | True |
| ca-GrQc | Dynamic-Logdet | 14.080 | 43.674 | True |
| ca-CondMat | CELF | 14.838 | 93.301 | True |
| ca-CondMat | Inc | 14.960 | 72.729 | True |
| ca-CondMat | Dynamic-Logdet | 14.555 | 92.932 | True |
| BA-50k | CELF | 17.124 | 163.147 | True |
| BA-50k | Inc | 15.347 | 115.014 | True |
| BA-50k | Dynamic-Logdet | 17.835 | 154.572 | True |

BA-100k Naive: the 5 measured first-round times are [64.42946300399853, 59.6718228539994, 66.41210919699915, 61.430776671000785, 62.90327044699916] seconds; their mean ± SD is 62.969488 ± 2.609108 s. The 20-round projection is 1253.407667 ± 51.934297 s under constant per-candidate cost. These projected totals are distinct from completed greedy runs.

## 2. Graph-size, candidate-pool, and budget curves

### 2.1 Six-size BA scaling: complete CELF and Inc runs

| Dataset | n | m | M | k | Method | Runs | Init (s), mean ± SD | Online (s), mean ± SD | Total (s), mean ± SD | Online RHS | Total RHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BA-1k | 1000 | 2991 | 2000 | 20 | CELF | 5 | 0.056356 ± 0.002764 | 0.024618 ± 0.000751 | 0.080973 ± 0.003452 | 25 | 2025 |
| BA-1k | 1000 | 2991 | 2000 | 20 | Inc | 5 | 0.056834 ± 0.001083 | 0.043568 ± 0.001413 | 0.100403 ± 0.002347 | 20 | 2020 |
| BA-5k | 5000 | 14991 | 2000 | 20 | CELF | 5 | 0.484008 ± 0.040334 | 0.130522 ± 0.008063 | 0.614530 ± 0.040654 | 22 | 2022 |
| BA-5k | 5000 | 14991 | 2000 | 20 | Inc | 5 | 0.499875 ± 0.069924 | 0.192620 ± 0.023652 | 0.692495 ± 0.088984 | 20 | 2020 |
| BA-10k | 10000 | 29991 | 2000 | 20 | CELF | 5 | 1.135971 ± 0.148839 | 0.320436 ± 0.038041 | 1.456407 ± 0.167582 | 22 | 2022 |
| BA-10k | 10000 | 29991 | 2000 | 20 | Inc | 5 | 1.128326 ± 0.069269 | 0.412213 ± 0.016515 | 1.540538 ± 0.061367 | 20 | 2020 |
| BA-25k | 25000 | 74991 | 2000 | 20 | CELF | 5 | 5.280053 ± 0.697343 | 0.840515 ± 0.033782 | 6.120569 ± 0.694092 | 19 | 2019 |
| BA-25k | 25000 | 74991 | 2000 | 20 | Inc | 5 | 4.955047 ± 0.430856 | 1.316120 ± 0.158010 | 6.271167 ± 0.551510 | 20 | 2020 |
| BA-50k | 50000 | 149991 | 2000 | 20 | CELF | 5 | 18.139624 ± 1.262302 | 1.997086 ± 0.042673 | 20.136709 ± 1.226080 | 19 | 2019 |
| BA-50k | 50000 | 149991 | 2000 | 20 | Inc | 5 | 19.634734 ± 2.268870 | 2.832869 ± 0.056391 | 22.467603 ± 2.276472 | 20 | 2020 |
| BA-100k | 100000 | 299991 | 2000 | 20 | CELF | 5 | 69.338192 ± 3.014652 | 4.930187 ± 0.176946 | 74.268379 ± 2.912950 | 19 | 2019 |
| BA-100k | 100000 | 299991 | 2000 | 20 | Inc | 5 | 66.825811 ± 2.982562 | 7.571470 ± 0.306833 | 74.397281 ± 3.265670 | 20 | 2020 |

### 2.2 Candidate-pool and budget sweeps

| Configuration | M | k | Method | Total (s), mean ± SD | Online (s), mean ± SD | RHS |
| --- | --- | --- | --- | --- | --- | --- |
| budget-10 | 1000 | 10 | CELF | 0.076449 ± 0.001920 | 0.017137 ± 0.000249 | 1009 |
| budget-10 | 1000 | 10 | Inc | 0.084455 ± 0.007110 | 0.028874 ± 0.000942 | 1010 |
| budget-10 | 1000 | 10 | Naive | 0.548888 ± 0.049962 | 0.490500 ± 0.043881 | 9955 |
| budget-20 | 1000 | 20 | CELF | 0.094730 ± 0.008162 | 0.037974 ± 0.002310 | 1022 |
| budget-20 | 1000 | 20 | Inc | 0.112849 ± 0.011846 | 0.058576 ± 0.004460 | 1020 |
| budget-20 | 1000 | 20 | Naive | 1.251545 ± 0.057882 | 1.194781 ± 0.057256 | 19810 |
| budget-40 | 1000 | 40 | CELF | 0.152199 ± 0.019650 | 0.094443 ± 0.012637 | 1056 |
| budget-40 | 1000 | 40 | Inc | 0.171453 ± 0.003448 | 0.110326 ± 0.003913 | 1040 |
| budget-40 | 1000 | 40 | Naive | 2.488970 ± 0.117354 | 2.427838 ± 0.115163 | 39220 |
| budget-5 | 1000 | 5 | CELF | 0.066133 ± 0.003189 | 0.009068 ± 0.001418 | 1004 |
| budget-5 | 1000 | 5 | Inc | 0.079030 ± 0.006892 | 0.017405 ± 0.002765 | 1005 |
| budget-5 | 1000 | 5 | Naive | 0.299344 ± 0.042284 | 0.243088 ± 0.038579 | 4990 |
| budget-60 | 1000 | 60 | CELF | 0.206675 ± 0.009400 | 0.150640 ± 0.004979 | 1093 |
| budget-60 | 1000 | 60 | Inc | 0.226645 ± 0.008710 | 0.169710 ± 0.006158 | 1060 |
| budget-60 | 1000 | 60 | Naive | 3.569274 ± 0.224281 | 3.509942 ± 0.224667 | 58230 |
| pool-1000 | 1000 | 20 | CELF | 0.099229 ± 0.008018 | 0.037757 ± 0.002179 | 1022 |
| pool-1000 | 1000 | 20 | Inc | 0.121370 ± 0.007595 | 0.059247 ± 0.003431 | 1020 |
| pool-1000 | 1000 | 20 | Naive | 1.289000 ± 0.070432 | 1.229100 ± 0.070815 | 19810 |
| pool-2000 | 2000 | 20 | CELF | 0.155129 ± 0.004855 | 0.038651 ± 0.002409 | 2023 |
| pool-2000 | 2000 | 20 | Inc | 0.196435 ± 0.043810 | 0.085480 ± 0.043112 | 2020 |
| pool-2000 | 2000 | 20 | Naive | 2.337795 ± 0.087911 | 2.206266 ± 0.056780 | 39810 |
| pool-250 | 250 | 20 | CELF | 0.058699 ± 0.007837 | 0.037615 ± 0.005510 | 270 |
| pool-250 | 250 | 20 | Inc | 0.069069 ± 0.006835 | 0.054461 ± 0.008542 | 270 |
| pool-250 | 250 | 20 | Naive | 0.323970 ± 0.054124 | 0.309065 ± 0.052686 | 4810 |
| pool-4000 | 4000 | 20 | CELF | 0.297022 ± 0.024412 | 0.063015 ± 0.009670 | 4033 |
| pool-4000 | 4000 | 20 | Inc | 0.326577 ± 0.026078 | 0.086157 ± 0.007480 | 4020 |
| pool-4000 | 4000 | 20 | Naive | 4.605099 ± 0.269364 | 4.364120 ± 0.257613 | 79810 |
| pool-500 | 500 | 20 | CELF | 0.068885 ± 0.008277 | 0.038090 ± 0.002786 | 522 |
| pool-500 | 500 | 20 | Inc | 0.088078 ± 0.009236 | 0.057617 ± 0.005048 | 520 |
| pool-500 | 500 | 20 | Naive | 0.658104 ± 0.048346 | 0.621119 ± 0.045380 | 9810 |

### 2.3 Iteration counts and sparse-product costs

| n | m | dmax | Mean initialization h | Initialization mean (s) | Single CSR SpMV (µs; one thread) | Effective cost/logical SpMV (µs; eight-thread run) | Total logical SpMV | Total padded SpMV equivalents |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 2991 | 106.0 | 37.9925 | 0.056356 | 5.2227 | 0.741669 | 76933 | 78916 |
| 5000 | 14991 | 218.0 | 53.3280 | 0.484008 | 23.8172 | 4.538030 | 107822 | 110142 |
| 10000 | 29991 | 313.0 | 62.9165 | 1.135971 | 48.7840 | 9.027605 | 127198 | 130517 |
| 25000 | 74991 | 446.0 | 75.7455 | 5.280053 | 185.0066 | 34.853908 | 152898 | 156495 |
| 50000 | 149991 | 764.0 | 86.3845 | 18.139624 | 418.0599 | 104.993510 | 174383 | 178478 |
| 100000 | 299991 | 1034.0 | 105.8120 | 69.338192 | 905.5717 | 327.648055 | 213594 | 219874 |

Log-log OLS exponents across the 6 available BA sizes: initialization time = 1.537008; average Krylov iterations = 0.218254; effective cost per logical SpMV = 1.318754. Their additive identity follows from T_init = M × hbar × effective_cost, with fixed M. The last factor includes recurrence arithmetic, batching, memory behavior, and parallel overhead. This decomposition measures both iteration growth and effective per-iteration cost.

A CSR SpMV performs exactly nnz(A) multiply-add pairs plus O(n) work, giving O(n+m) arithmetic complexity. Its wall time also reflects cache and memory regimes. The unpreconditioned worst-case CG estimate is O(sqrt(kappa) log(1/tol)). Combining the BA maximum-degree scale dmax = O(n^(1/2)) with kappa ≤ 1 + 2αdmax gives an O(n^(1/4)) iteration upper-bound factor at fixed α. For fixed q and s with the balanced shift scale τ = √Λ, the smallest node has t_min = c√Λ for fixed c > 0, giving κ = (Λ+c√Λ)/(1+c√Λ). The asymptotic shifted-CG bound is then O(Λ^(1/4) log(1/tol)); combined with that BA degree scale, it gives an n^(1/8) upper-bound factor. These worst-case bounds are distinct from an empirical n^0.4 iteration law. The table separately records one-thread CSR microbenchmark times and the effective cost derived from eight-thread initialization runs.

## 3. Multiscale competition with the identical compiled backend

| Dataset | n | m | M | k | s | Method | Init (s) | Online (s) | Total (s) | Online RHS | Total RHS | Per-round online RHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Heterogeneous-Multiscale | 400 | 1167 | 150 | 15 | 0.1 | CELF | 0.014386 ± 0.001772 | 0.057608 ± 0.003438 | 0.071995 ± 0.003410 | 32 | 182 | [0, 2, 2, 2, 1, 3, 3, 4, 4, 1, 4, 1, 1, 2, 2] |
| Heterogeneous-Multiscale | 400 | 1167 | 150 | 15 | 0.1 | Inc | 0.014059 ± 0.001387 | 0.039706 ± 0.002279 | 0.053765 ± 0.002994 | 15 | 165 | [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] |
| Heterogeneous-Multiscale | 400 | 1167 | 150 | 15 | 0.1 | Dynamic-Logdet | 0.013718 ± 0.001208 | 0.110387 ± 0.007475 | 0.124105 ± 0.007321 | 62 | 212 | [0, 2, 3, 2, 3, 5, 5, 3, 4, 1, 4, 2, 1, 20, 7] |
| Heterogeneous-Multiscale | 400 | 1167 | 150 | 15 | 0.5 | CELF | 0.015021 ± 0.000739 | 0.046972 ± 0.007517 | 0.061993 ± 0.007707 | 25 | 175 | [0, 1, 2, 1, 1, 3, 1, 1, 1, 4, 1, 2, 3, 3, 1] |
| Heterogeneous-Multiscale | 400 | 1167 | 150 | 15 | 0.5 | Inc | 0.014476 ± 0.001065 | 0.037753 ± 0.000663 | 0.052228 ± 0.000739 | 15 | 165 | [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] |
| Heterogeneous-Multiscale | 400 | 1167 | 150 | 15 | 0.5 | Dynamic-Logdet | 0.016533 ± 0.003136 | 0.107771 ± 0.005897 | 0.124303 ± 0.005923 | 62 | 212 | [0, 2, 3, 2, 3, 5, 5, 3, 4, 1, 4, 2, 1, 20, 7] |
| Heterogeneous-Multiscale | 400 | 1167 | 150 | 15 | 0.85 | CELF | 0.014198 ± 0.000746 | 0.033220 ± 0.001777 | 0.047419 ± 0.002104 | 19 | 169 | [0, 1, 1, 2, 1, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1] |
| Heterogeneous-Multiscale | 400 | 1167 | 150 | 15 | 0.85 | Inc | 0.014593 ± 0.001559 | 0.035800 ± 0.000600 | 0.050393 ± 0.001703 | 15 | 165 | [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] |
| Heterogeneous-Multiscale | 400 | 1167 | 150 | 15 | 0.85 | Dynamic-Logdet | 0.015608 ± 0.001050 | 0.101882 ± 0.001350 | 0.117491 ± 0.002221 | 62 | 212 | [0, 2, 3, 2, 3, 5, 5, 3, 4, 1, 4, 2, 1, 20, 7] |
| Multiscale-SBM-4k | 4000 | 59772 | 600 | 20 | 0.1 | CELF | 0.210148 ± 0.009066 | 2.611808 ± 0.105414 | 2.821957 ± 0.110393 | 244 | 844 | [0, 1, 2, 4, 6, 1, 8, 10, 8, 7, 15, 18, 13, 5, 30, 32, 12, 17, 30, 25] |
| Multiscale-SBM-4k | 4000 | 59772 | 600 | 20 | 0.1 | Inc | 0.218471 ± 0.008306 | 0.276261 ± 0.015552 | 0.494732 ± 0.020234 | 20 | 620 | [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] |
| Multiscale-SBM-4k | 4000 | 59772 | 600 | 20 | 0.1 | Dynamic-Logdet | 0.200337 ± 0.013337 | 5.283457 ± 0.157894 | 5.483794 ± 0.149816 | 508 | 1108 | [0, 2, 2, 7, 9, 11, 6, 44, 28, 68, 56, 34, 46, 38, 48, 21, 13, 28, 20, 27] |
| Multiscale-SBM | 400 | 5990 | 150 | 15 | 0.1 | CELF | 0.005668 ± 0.000379 | 0.373541 ± 0.027106 | 0.379209 ± 0.027037 | 437 | 587 | [0, 4, 72, 49, 37, 6, 68, 12, 58, 12, 39, 8, 57, 13, 2] |
| Multiscale-SBM | 400 | 5990 | 150 | 15 | 0.1 | Inc | 0.005079 ± 0.000708 | 0.020263 ± 0.000450 | 0.025342 ± 0.000712 | 15 | 165 | [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] |
| Multiscale-SBM | 400 | 5990 | 150 | 15 | 0.1 | Dynamic-Logdet | 0.005164 ± 0.000601 | 0.413944 ± 0.009575 | 0.419108 ± 0.009070 | 489 | 639 | [0, 4, 72, 49, 39, 9, 68, 7, 64, 5, 44, 9, 61, 20, 38] |
| Multiscale-SBM | 400 | 5990 | 150 | 15 | 0.5 | CELF | 0.005128 ± 0.000438 | 0.169306 ± 0.009324 | 0.174435 ± 0.009701 | 195 | 345 | [0, 4, 8, 17, 3, 56, 2, 15, 8, 8, 17, 3, 17, 30, 7] |
| Multiscale-SBM | 400 | 5990 | 150 | 15 | 0.5 | Inc | 0.005063 ± 0.001055 | 0.020931 ± 0.002173 | 0.025994 ± 0.002808 | 15 | 165 | [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] |
| Multiscale-SBM | 400 | 5990 | 150 | 15 | 0.5 | Dynamic-Logdet | 0.004977 ± 0.000761 | 0.422682 ± 0.022801 | 0.427659 ± 0.022583 | 489 | 639 | [0, 4, 72, 49, 39, 9, 68, 7, 64, 5, 44, 9, 61, 20, 38] |
| Multiscale-SBM | 400 | 5990 | 150 | 15 | 0.85 | CELF | 0.005042 ± 0.000608 | 0.047795 ± 0.004399 | 0.052838 ± 0.004860 | 52 | 202 | [0, 2, 3, 2, 2, 10, 3, 2, 3, 3, 3, 7, 6, 4, 2] |
| Multiscale-SBM | 400 | 5990 | 150 | 15 | 0.85 | Inc | 0.005243 ± 0.000469 | 0.020010 ± 0.001402 | 0.025253 ± 0.001791 | 15 | 165 | [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] |
| Multiscale-SBM | 400 | 5990 | 150 | 15 | 0.85 | Dynamic-Logdet | 0.004593 ± 0.000491 | 0.411303 ± 0.012243 | 0.415896 ± 0.012270 | 489 | 639 | [0, 4, 72, 49, 39, 9, 68, 7, 64, 5, 44, 9, 61, 20, 38] |

All ratios in the following table divide CELF by Inc. A time ratio above one is an Inc speedup; RHS ratios measure solve-count reduction independently of time. Time ratios use the corresponding five-run means.

| Dataset | n | s | Inc total speedup | Inc online speedup | Online RHS reduction (CELF/Inc) | Total RHS reduction (CELF/Inc) | Identical ordered sequence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Heterogeneous-Multiscale | 400 | 0.1 | 1.339047 | 1.450859 | 2.133333 | 1.103030 | True |
| Heterogeneous-Multiscale | 400 | 0.5 | 1.186958 | 1.244187 | 1.666667 | 1.060606 | True |
| Heterogeneous-Multiscale | 400 | 0.85 | 0.940974 | 0.927942 | 1.266667 | 1.024242 | True |
| Multiscale-SBM-4k | 4000 | 0.1 | 5.704012 | 9.454130 | 12.200000 | 1.361290 | True |
| Multiscale-SBM | 400 | 0.1 | 14.963801 | 18.434444 | 29.133333 | 3.557576 | True |
| Multiscale-SBM | 400 | 0.5 | 6.710586 | 8.088690 | 13.000000 | 2.090909 | True |
| Multiscale-SBM | 400 | 0.85 | 2.092323 | 2.388545 | 3.466667 | 1.224242 | True |

### Large-PowerCluster: raw online calls per accepted round

Each array is recorded after the initial scalar table has been computed and covers rounds 1 through k. The array from the first repetition is shown, together with whether all recorded repetitions have exactly the same array. Its sum is the online RHS count.

| Method | Runs | Rounds | Sum of online RHS | Same array in every run | Raw per-round online RHS |
| --- | --- | --- | --- | --- | --- |
| CELF | 5 | 30 | 31 | True | `[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1]` |
| Inc | 5 | 30 | 30 | True | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]` |
| Dynamic-Logdet | 5 | 30 | 32 | True | `[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1]` |
| Naive | 5 | 30 | 86565 | True | `[0, 2999, 2998, 2997, 2996, 2995, 2994, 2993, 2992, 2991, 2990, 2989, 2988, 2987, 2986, 2985, 2984, 2983, 2982, 2981, 2980, 2979, 2978, 2977, 2976, 2975, 2974, 2973, 2972, 2971]` |

## 4. Same-backend marginal evaluator ablation

| Dataset | Route | Runs | Seconds, mean ± SD | Incidence RHS | Probe RHS | Krylov SpMV | Residual SpMV | Maximum relative marginal error | True G | Set match with reference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Small-BA | FastFrac-CELF (IBP-Scalar) | 5 | 0.012019 ± 0.000968 | 211 | — | 6512 | 0 | 1.376928e-10 | 4.651877009 | 10/10 |
| Small-BA | Raw-Loewner-Proj | 5 | 0.011349 ± 0.000979 | 211 | — | 6512 | 0 | 3.483141e-09 | 4.651877009 | 10/10 |
| Small-BA | Raw-Loewner-FullVec | 5 | 0.023250 ± 0.004036 | 211 | — | 6512 | 2532 | 3.483152e-09 | 4.651877009 | 10/10 |
| Small-BA | CRN-SLQ-Greedy-p32 | 5 | 0.375355 ± 0.033805 | — | 62592 | 938880 | 0 | 4.304477e-01 | 4.429166810 | 1/10 |
| Small-BA | CRN-SLQ-Greedy-p128 | 5 | 1.422657 ± 0.051179 | — | 250368 | 3755520 | 0 | 2.263703e-01 | 4.586301761 | 4/10 |
| Small-BA | CRN-SLQ-CELF-p128 | 5 | 0.216505 ± 0.009737 | — | 27008 | 405120 | 0 | 2.263703e-01 | 4.586301761 | 4/10 |
| Medium-WS | FastFrac-CELF (IBP-Scalar) | 5 | 0.092781 ± 0.000597 | 1022 | — | 25372 | 0 | 4.456385e-11 | 7.257383514 | 20/20 |
| Medium-WS | Raw-Loewner-Proj | 5 | 0.092411 ± 0.001986 | 1022 | — | 25372 | 0 | 1.063122e-09 | 7.257383514 | 20/20 |
| Medium-WS | Raw-Loewner-FullVec | 5 | 0.355098 ± 0.016400 | 1022 | — | 25372 | 12264 | 1.063120e-09 | 7.257383514 | 20/20 |
| Medium-WS | CRN-SLQ-Greedy-p32 | 5 | 17.965565 ± 1.141227 | — | 633952 | 9509280 | 0 | 6.386924e-01 | 6.880657756 | 0/20 |
| Medium-WS | CRN-SLQ-Greedy-p128 | 5 | 74.865446 ± 3.397164 | — | 2535808 | 38037120 | 0 | 2.560566e-01 | 6.999686510 | 2/20 |
| Medium-WS | CRN-SLQ-CELF-p128 | 5 | 4.063926 ± 0.152025 | — | 130688 | 1960320 | 0 | 2.560566e-01 | 6.999686510 | 2/20 |

Raw-Proj evaluates the inverse-square quadratic form in the Lanczos projection via a differentiated LDL recurrence. Raw-FullVec explicitly forms every shifted n-vector through equivalent short direction recurrences and validates its true residual. CRN probes are fixed across every candidate trial and round; p32 is a prefix of p128.

### Actual lazy selections under identical probes

This comparison holds the sampled objective fixed. A strict wrong lazy decision requires divergence from exhaustive greedy under the same probes and current selected set. True gain loss is exhaustive minus lazy; it is separate from either method’s error relative to exact fractional greedy.

| Dataset | Set overlap, lazy vs exhaustive | Ordered matches | First divergent round | Strict wrong pruning observed | True gain loss, exhaustive minus lazy |
| --- | --- | --- | --- | --- | --- |
| Small-BA | 10/10 | 10/10 | None | False | 0 |
| Medium-WS | 20/20 | 20/20 | None | False | 0 |

### Cached-point upper-bound diagnostics

The first diagnostic compares an actual CELF cache key with the **current marginal of the same fixed-probe SLQ objective**, at rounds where the lazy and exhaustive paths have the same prefix. A violation exceeds 1e-10 × max(1, abs(current marginal)). The second compares that cache key with the **independently evaluated true fractional-trace marginal**, using the recorded relative threshold of 1e-8. The targets and thresholds differ, so these rates are not interchangeable. Maximum marginal relative error in the route table above is a third quantity: point-estimation error against the true marginal. A cache-bound violation need not change a selected winner.

| Dataset | Comparison target | Rounds evaluated | Cached keys checked | Keys below current marginal | Violation rate | Maximum positive absolute excess |
| --- | --- | --- | --- | --- | --- | --- |
| Small-BA | Same fixed-probe SLQ objective | 10 | 1755 | 657 | 37.435897% | 0.000104828496 |
| Small-BA | True fractional-trace marginal | 10 | 1755 | 787 | 44.843305% | not recorded |
| Medium-WS | Same fixed-probe SLQ objective | 20 | 18810 | 6957 | 36.985646% | 0.000371847178 |
| Medium-WS | True fractional-trace marginal | 20 | 18810 | 9609 | 51.084530% | not recorded |

## External utility and complete spectral-scale scan

The original SBM and the prespecified heterogeneous four-block graph are reused without seed or parameter search. Added edges have unit weight. All degree and neighborhood diagnostics refer to the unchanged base graph and include every selected edge. Block 0 is the heterogeneous graph's lowest-weight module; blocks 2 and 3 are its highest-weight modules. These fixed labels describe the allocation rather than an optimized post-hoc partition.

For the regularized matrix A = I + αL, the log-determinant derivative weight is 1/(1+αλ), which is bounded at λ = 0. Its low-frequency preference therefore does not imply a singular divergence or guarantee that it overallocates a particular cut. Heat-trace reductions below are independently evaluated from complete dense Laplacian eigenspectra at the declared horizons.

### Multiscale-SBM: independent heat-trace and allocation outcomes, α = 50, k = 15

| Method | ΔH(.05) | ΔH(.2) | ΔH(1) | vs LD (.05) | vs LD (.2) | vs LD (1) | Block0-cut edges | Core2/3 edges | N2 | E_loc | Overlap LD /15 | Total s (5-run mean ± SD) | RHS | Krylov SpMV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dynamic-Logdet | 0.470919 | 0.123519 | 0.262582 | +0.000% | +0.000% | +0.000% | 8 | 2 | 0.133 | 203.733 | 15 | 0.417340 ± 0.022856 | 639 | 15792 |
| FastFrac(s=0.10) | 0.475648 | 0.126250 | 0.261376 | +1.004% | +2.211% | -0.459% | 8 | 2 | 0.133 | 199.267 | 14 | 0.391390 ± 0.036746 | 587 | 14420 |
| FastFrac(s=0.30) | 0.480711 | 0.128438 | 0.258853 | +2.079% | +3.983% | -1.420% | 8 | 3 | 0.200 | 193.533 | 10 | 0.325614 ± 0.016147 | 517 | 12403 |
| FastFrac(s=0.50) | 0.484616 | 0.130043 | 0.255609 | +2.909% | +5.282% | -2.655% | 9 | 3 | 0.200 | 191.133 | 9 | 0.165970 ± 0.001875 | 345 | 7842 |
| FastFrac(s=0.70) | 0.485035 | 0.129893 | 0.255320 | +2.998% | +5.160% | -2.765% | 9 | 3 | 0.200 | 190.467 | 9 | 0.093388 ± 0.003176 | 253 | 5370 |
| FastFrac(s=0.85) | 0.485035 | 0.129893 | 0.255320 | +2.998% | +5.160% | -2.765% | 9 | 3 | 0.200 | 190.467 | 9 | 0.052144 ± 0.002000 | 202 | 4050 |

The block-0/core categories for the uniform SBM use the same fixed labels but do not represent a weight-defined core-periphery hierarchy. Counts are reported for topology comparability.

### Heterogeneous-Multiscale: independent heat-trace and allocation outcomes, α = 50, k = 15

| Method | ΔH(.05) | ΔH(.2) | ΔH(1) | vs LD (.05) | vs LD (.2) | vs LD (1) | Block0-cut edges | Core2/3 edges | N2 | E_loc | Overlap LD /15 | Total s (5-run mean ± SD) | RHS | Krylov SpMV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dynamic-Logdet | 1.242774 | 3.345170 | 5.155263 | +0.000% | +0.000% | +0.000% | 14 | 0 | 2.400 | 8.667 | 15 | 0.125662 ± 0.006802 | 212 | 37514 |
| FastFrac(s=0.10) | 1.242774 | 3.345170 | 5.155263 | +0.000% | +0.000% | +0.000% | 14 | 0 | 2.400 | 8.667 | 15 | 0.072400 ± 0.001336 | 182 | 30758 |
| FastFrac(s=0.30) | 1.242774 | 3.345170 | 5.155263 | +0.000% | +0.000% | +0.000% | 14 | 0 | 2.400 | 8.667 | 15 | 0.065646 ± 0.003455 | 178 | 29023 |
| FastFrac(s=0.50) | 1.260749 | 3.421553 | 5.017860 | +1.446% | +2.283% | -2.665% | 13 | 0 | 4.133 | 9.667 | 14 | 0.058174 ± 0.000171 | 175 | 27651 |
| FastFrac(s=0.70) | 1.295364 | 3.528287 | 4.629322 | +4.232% | +5.474% | -10.202% | 11 | 0 | 9.467 | 13.800 | 12 | 0.047794 ± 0.000392 | 170 | 25930 |
| FastFrac(s=0.85) | 1.295364 | 3.528287 | 4.629322 | +4.232% | +5.474% | -10.202% | 11 | 0 | 9.467 | 13.800 | 12 | 0.046268 ± 0.001273 | 169 | 25184 |


### Complete heterogeneous scan (all 7 s values × 4 α values)

q = 12 and tol = 10⁻⁷ for the timed method. The q = 54 validation selection is recomputed separately at tol = 10⁻¹⁰; a different sequence is explicitly shown rather than hidden. True gains use complete dense eigenspectra.

| α | s | dΣ unweighted | dΣ weighted | N2 | E_loc | Cross /15 | FastFrac G | Dynamic-LD G | vs LD | Overlap /15 | q12=q54 sequence | Total s mean ± SD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.05 | 8.200 | 4.839 | 7.333 | 12.067 | 12 | 0.501658 | 0.501606 | +0.0104% | 14 | yes | 0.021213 ± 0.000093 |
| 1 | 0.10 | 8.200 | 4.839 | 7.333 | 12.067 | 12 | 1.056299 | 1.055138 | +0.1100% | 14 | yes | 0.022171 ± 0.002523 |
| 1 | 0.25 | 9.133 | 4.061 | 9.467 | 13.800 | 11 | 3.100340 | 3.081918 | +0.5978% | 13 | yes | 0.019220 ± 0.000499 |
| 1 | 0.50 | 8.800 | 3.983 | 9.733 | 13.267 | 11 | 8.199189 | 8.109865 | +1.1014% | 13 | yes | 0.019207 ± 0.000114 |
| 1 | 0.70 | 10.000 | 3.806 | 11.533 | 15.467 | 10 | 14.513593 | 14.354800 | +1.1062% | 12 | yes | 0.019673 ± 0.000065 |
| 1 | 0.85 | 10.000 | 3.806 | 11.533 | 15.467 | 10 | 21.146659 | 20.986580 | +0.7628% | 12 | yes | 0.019456 ± 0.000137 |
| 1 | 0.95 | 11.000 | 3.675 | 13.400 | 17.800 | 9 | 26.761672 | 26.679817 | +0.3068% | 11 | yes | 0.019484 ± 0.000360 |
| 10 | 0.05 | 6.733 | 6.636 | 2.400 | 8.667 | 14 | 0.960085 | 0.960085 | +0.0000% | 15 | yes | 0.062279 ± 0.030572 |
| 10 | 0.10 | 6.733 | 6.636 | 2.400 | 8.667 | 14 | 2.179444 | 2.179444 | +0.0000% | 15 | yes | 0.048566 ± 0.004185 |
| 10 | 0.25 | 6.733 | 6.636 | 2.400 | 8.667 | 14 | 8.059162 | 8.059162 | +0.0000% | 15 | yes | 0.045766 ± 0.003090 |
| 10 | 0.50 | 7.333 | 5.833 | 4.133 | 9.667 | 13 | 32.316886 | 32.180442 | +0.4240% | 14 | yes | 0.043869 ± 0.001811 |
| 10 | 0.70 | 9.133 | 4.061 | 9.467 | 13.800 | 11 | 82.116719 | 81.247530 | +1.0698% | 12 | yes | 0.037150 ± 0.000800 |
| 10 | 0.85 | 9.133 | 4.061 | 9.467 | 13.800 | 11 | 158.469449 | 157.019763 | +0.9233% | 12 | yes | 0.036007 ± 0.001687 |
| 10 | 0.95 | 8.800 | 3.983 | 9.733 | 13.267 | 11 | 242.884031 | 241.909774 | +0.4027% | 12 | yes | 0.037816 ± 0.000531 |
| 50 | 0.05 | 6.733 | 6.636 | 2.400 | 8.667 | 14 | 1.198599 | 1.198599 | +0.0000% | 15 | yes | 0.084450 ± 0.000585 |
| 50 | 0.10 | 6.733 | 6.636 | 2.400 | 8.667 | 14 | 2.897989 | 2.897989 | +0.0000% | 15 | yes | 0.072578 ± 0.000431 |
| 50 | 0.25 | 6.733 | 6.636 | 2.400 | 8.667 | 14 | 13.095245 | 13.095245 | +0.0000% | 15 | yes | 0.067153 ± 0.001734 |
| 50 | 0.50 | 7.333 | 5.833 | 4.133 | 9.667 | 13 | 75.132819 | 74.920415 | +0.2835% | 14 | yes | 0.068564 ± 0.019371 |
| 50 | 0.70 | 9.133 | 4.061 | 9.467 | 13.800 | 11 | 257.823011 | 255.586584 | +0.8750% | 12 | yes | 0.051234 ± 0.003448 |
| 50 | 0.85 | 9.133 | 4.061 | 9.467 | 13.800 | 11 | 627.216266 | 621.887591 | +0.8569% | 12 | yes | 0.046824 ± 0.001323 |
| 50 | 0.95 | 9.133 | 4.061 | 9.467 | 13.800 | 11 | 1123.178404 | 1118.855958 | +0.3863% | 12 | yes | 0.051176 ± 0.000679 |
| 200 | 0.05 | 6.133 | 7.342 | 0.000 | 7.600 | 15 | 1.421313 | 1.419256 | +0.1449% | 14 | yes | 0.273668 ± 0.025746 |
| 200 | 0.10 | 6.133 | 7.342 | 0.000 | 7.600 | 15 | 3.603091 | 3.588844 | +0.3970% | 14 | yes | 0.233220 ± 0.029690 |
| 200 | 0.25 | 6.733 | 6.636 | 2.400 | 8.667 | 14 | 19.319617 | 18.983971 | +1.7681% | 13 | yes | 0.105165 ± 0.004498 |
| 200 | 0.50 | 7.333 | 5.833 | 4.133 | 9.667 | 13 | 152.255458 | 148.414939 | +2.5877% | 12 | yes | 0.076894 ± 0.002922 |
| 200 | 0.70 | 9.133 | 4.061 | 9.467 | 13.800 | 11 | 683.451422 | 665.433707 | +2.7077% | 10 | yes | 0.058273 ± 0.001312 |
| 200 | 0.85 | 9.133 | 4.061 | 9.467 | 13.800 | 11 | 2041.416621 | 2002.428020 | +1.9471% | 10 | yes | 0.054393 ± 0.003850 |
| 200 | 0.95 | 9.133 | 4.061 | 9.467 | 13.800 | 11 | 4194.007212 | 4161.446226 | +0.7824% | 10 | yes | 0.071806 ± 0.028010 |

### Dynamic-Logdet runtime and structure on each heterogeneous α

| α | Init s mean ± SD | Online s mean ± SD | Total s mean ± SD | Total RHS | Online RHS | Krylov SpMV | dΣ unweighted | dΣ weighted | N2 | E_loc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.013851 ± 0.012958 | 0.017896 ± 0.001378 | 0.031747 ± 0.012897 | 172 | 22 | 10031 | 7.333 | 5.833 | 4.133 | 9.667 |
| 10 | 0.012199 ± 0.000457 | 0.033585 ± 0.001210 | 0.045784 ± 0.001229 | 173 | 23 | 21663 | 6.733 | 6.636 | 2.400 | 8.667 |
| 50 | 0.015747 ± 0.001174 | 0.103312 ± 0.000838 | 0.119059 ± 0.001639 | 212 | 62 | 37514 | 6.733 | 6.636 | 2.400 | 8.667 |
| 200 | 0.021288 ± 0.004554 | 0.332992 ± 0.021941 | 0.354280 ± 0.021856 | 316 | 166 | 67931 | 6.067 | 7.677 | 0.000 | 7.467 |

### Complete low-four-mode derivative share scan

| α | s | SBM P_low crossing | Heterogeneous P_low crossing |
| --- | --- | --- | --- |
| 1 | 0.05 | 17.621678% | 6.168906% |
| 1 | 0.10 | 15.417138% | 5.658186% |
| 1 | 0.25 | 10.129595% | 4.331265% |
| 1 | 0.50 | 4.808967% | 2.715957% |
| 1 | 0.70 | 2.586793% | 1.839808% |
| 1 | 0.85 | 1.611844% | 1.361271% |
| 1 | 0.95 | 1.172951% | 1.108975% |
| 10 | 0.05 | 64.676898% | 27.361912% |
| 10 | 0.10 | 58.228015% | 24.355406% |
| 10 | 0.25 | 38.064699% | 16.243527% |
| 10 | 0.50 | 13.538630% | 7.058654% |
| 10 | 0.70 | 4.975131% | 3.319335% |
| 10 | 0.85 | 2.248666% | 1.835209% |
| 10 | 0.95 | 1.311634% | 1.225824% |
| 50 | 0.05 | 89.277351% | 55.691794% |
| 50 | 0.10 | 85.407415% | 50.795865% |
| 50 | 0.25 | 67.017961% | 35.268847% |
| 50 | 0.50 | 25.797185% | 13.734594% |
| 50 | 0.70 | 7.791362% | 5.124753% |
| 50 | 0.85 | 2.839438% | 2.293861% |
| 50 | 0.95 | 1.419014% | 1.321436% |
| 200 | 0.05 | 96.768848% | 77.707986% |
| 200 | 0.10 | 95.164207% | 73.128623% |
| 200 | 0.25 | 84.804579% | 55.515957% |
| 200 | 0.50 | 40.545224% | 22.316537% |
| 200 | 0.70 | 11.237381% | 7.212222% |
| 200 | 0.85 | 3.453604% | 2.743133% |
| 200 | 0.95 | 1.516424% | 1.403911% |

P_low is the mean across initial crossing candidates of the fraction of the directional derivative contributed by the four smallest eigenmodes of A. It is not a four-mode approximation to the finite-edge marginal. The exact per-edge definitions and selected indices are retained in science_scan_downstream.json.

### Static, degree and random controls on both multiscale graphs

| Dataset | Method | G(.10) | G(.50) | G(.85) | dΣ unweighted | dΣ weighted | N2 | E_loc | Selection s mean ± SD | RHS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Multiscale-SBM | Static-EffRes | 0.768899511 | 25.543526848 | 447.370043966 | 44.200 | 44.200 | 0.067 | 188.933 | 0.005793 ± 0.000315 | 150 |
| Multiscale-SBM | Degree | 0.794467922 | 25.804966487 | 448.333779548 | 43.533 | 43.533 | 0.067 | 187.400 | 0.000938 ± 0.000174 | 0 |
| Multiscale-SBM | Degree-Unweighted | 0.794467922 | 25.804966487 | 448.333779548 | 43.533 | 43.533 | 0.067 | 187.400 | 0.000290 ± 0.000097 | 0 |
| Multiscale-SBM | Random | 0.555198015 | 21.547547325 | 428.120923122 | 60.300 | 60.300 | 51.073 | 405.007 | 0.000123 ± 0.000100 | 0 |
| Heterogeneous-Multiscale | Static-EffRes | 2.759401921 | 72.087604215 | 613.649492724 | 6.133 | 7.169 | 0.000 | 7.533 | 0.018896 ± 0.000192 | 150 |
| Heterogeneous-Multiscale | Degree | 2.368739818 | 70.209374189 | 618.815058073 | 13.467 | 3.225 | 21.133 | 22.667 | 0.000533 ± 0.000050 | 0 |
| Heterogeneous-Multiscale | Degree-Unweighted | 2.672074403 | 69.905824528 | 605.880113718 | 6.000 | 8.566 | 0.000 | 7.400 | 0.000233 ± 0.000021 | 0 |
| Heterogeneous-Multiscale | Random | 1.310807872 | 41.751091976 | 507.666951724 | 20.773 | 47.832 | 17.507 | 36.927 | 0.000089 ± 0.000056 | 0 |

Degree preserves the locked multiscale benchmark definition: ascending weighted endpoint-degree sum. Degree-Unweighted is separately reported to expose the effect of weights. Random averages10 independently seeded uniform subsets within each repetition; selected-set generation time excludes the independent dense evaluation, consistently with all other selection timings. Full gains for all7s and heat-horizon values are retained in JSON.

## Quadrature audit against full Exact-Eig greedy at α = 1 and50

For each dataset/s, a fresh full dense-eigenvalue greedy is run for all k rounds. Quadrature errors are audited over every remaining candidate at every state on that exact sequence. The initial-candidate maximum is given separately so that an initial-only error is not confused with a path-wide maximum. The Gaussian surrogate is evaluated from dense spectral resolvents for this quadrature-only audit; timed greedy runs use the common compiled backend.

| Dataset | α | s | q | Analytic ηq | Max rel. error (path) | Mean rel. error (path) | Max rel. error (initial) | Ordered sequence match | Set match | Total s mean ± SD | Audited marginals |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Small-BA | 1 | 0.50 | 4 | 3.5339e-02 | 3.4709e-04 | 6.8554e-05 | 3.4709e-04 | 100.0% | 100.0% | 0.010776 ± 0.004318 | 1955 |
| Small-BA | 1 | 0.50 | 8 | 3.2355e-04 | 2.1059e-07 | 1.6388e-08 | 2.1056e-07 | 100.0% | 100.0% | 0.007336 ± 0.000241 | 1955 |
| Small-BA | 1 | 0.50 | 12 | 2.9106e-06 | 1.3810e-10 | 6.1408e-12 | 1.3770e-10 | 100.0% | 100.0% | 0.007760 ± 0.000523 | 1955 |
| Small-BA | 1 | 0.50 | 16 | 2.6180e-08 | 5.8644e-13 | 1.1855e-13 | 2.8829e-13 | 100.0% | 100.0% | 0.007319 ± 0.000380 | 1955 |
| Small-BA | 1 | 0.50 | 24 | 2.1179e-12 | 5.8602e-13 | 1.1768e-13 | 2.9082e-13 | 100.0% | 100.0% | 0.007469 ± 0.000208 | 1955 |
| Small-BA | 1 | 0.50 | 36 | 1.5411e-18 | 5.8602e-13 | 1.1767e-13 | 2.9082e-13 | 100.0% | 100.0% | 0.007412 ± 0.000655 | 1955 |
| Small-BA | 1 | 0.50 | 54 | 9.5660e-28 | 5.8580e-13 | 1.1763e-13 | 2.9082e-13 | 100.0% | 100.0% | 0.007492 ± 0.000159 | 1955 |
| Multiscale-SBM | 50 | 0.10 | 4 | 4.7798e-01 | 1.0389e-01 | 2.6920e-02 | 1.0389e-01 | 66.7% | 80.0% | 0.416977 ± 0.034779 | 2145 |
| Multiscale-SBM | 50 | 0.10 | 8 | 9.8622e-02 | 9.5692e-03 | 1.4772e-03 | 9.5692e-03 | 100.0% | 100.0% | 0.408216 ± 0.063846 | 2145 |
| Multiscale-SBM | 50 | 0.10 | 12 | 1.6584e-02 | 1.0759e-03 | 1.1965e-04 | 1.0759e-03 | 100.0% | 100.0% | 0.458972 ± 0.063624 | 2145 |
| Multiscale-SBM | 50 | 0.10 | 16 | 2.6903e-03 | 1.3232e-04 | 1.2649e-05 | 1.3232e-04 | 100.0% | 100.0% | 0.397245 ± 0.010250 | 2145 |
| Multiscale-SBM | 50 | 0.10 | 24 | 6.9911e-05 | 2.2872e-06 | 2.0174e-07 | 2.2872e-06 | 100.0% | 100.0% | 0.392359 ± 0.014875 | 2145 |
| Multiscale-SBM | 50 | 0.10 | 36 | 2.9229e-07 | 6.2054e-09 | 5.5998e-10 | 6.2054e-09 | 100.0% | 100.0% | 0.373774 ± 0.005240 | 2145 |
| Multiscale-SBM | 50 | 0.10 | 54 | 7.9010e-11 | 8.8122e-11 | 2.5700e-11 | 8.8122e-11 | 100.0% | 100.0% | 0.405388 ± 0.014335 | 2145 |
| Multiscale-SBM | 50 | 0.50 | 4 | 4.7798e-01 | 8.1960e-02 | 5.7049e-02 | 8.1959e-02 | 46.7% | 100.0% | 0.124362 ± 0.005086 | 2145 |
| Multiscale-SBM | 50 | 0.50 | 8 | 9.8622e-02 | 6.8241e-03 | 2.3202e-03 | 6.8241e-03 | 100.0% | 100.0% | 0.160049 ± 0.001788 | 2145 |
| Multiscale-SBM | 50 | 0.50 | 12 | 1.6584e-02 | 7.7490e-04 | 1.2992e-04 | 7.7490e-04 | 100.0% | 100.0% | 0.164388 ± 0.001308 | 2145 |
| Multiscale-SBM | 50 | 0.50 | 16 | 2.6903e-03 | 9.7199e-05 | 1.0646e-05 | 9.7199e-05 | 100.0% | 100.0% | 0.173091 ± 0.007734 | 2145 |
| Multiscale-SBM | 50 | 0.50 | 24 | 6.9911e-05 | 1.7209e-06 | 1.4532e-07 | 1.7209e-06 | 100.0% | 100.0% | 0.176873 ± 0.003889 | 2145 |
| Multiscale-SBM | 50 | 0.50 | 36 | 2.9229e-07 | 4.7388e-09 | 3.8321e-10 | 4.7388e-09 | 100.0% | 100.0% | 0.172138 ± 0.004538 | 2145 |
| Multiscale-SBM | 50 | 0.50 | 54 | 7.9010e-11 | 4.9272e-12 | 1.5478e-12 | 3.9447e-12 | 100.0% | 100.0% | 0.173711 ± 0.004403 | 2145 |

| Dataset | s | Exact-Eig greedy seconds | Dense eigensolves |
| --- | --- | --- | --- |
| Small-BA | 0.50 | 11.970108 | 1965 |
| Multiscale-SBM | 0.10 | 14.601720 | 2160 |
| Multiscale-SBM | 0.50 | 13.940430 | 2160 |

## Twenty-instance global-optimum audit with all requested baselines

The20 locked n = 25, M = 15, k = 4 instances and their previously exhaustively verified OPT values are reused after graph/candidate identity checks. Those OPT values cover27,300 subsets; all method selections and exact eigenvalue gains below are freshly recomputed. Random denotes the mean of10 fixed-seed subsets in each instance.

| Method | Min G/OPT | Mean G/OPT | Std across20 instances | Max G/OPT |
| --- | --- | --- | --- | --- |
| FastFrac-CELF | 0.990183179 | 0.999280704 | 0.002371973 | 1.000000000 |
| FastFrac-Inc | 0.990183179 | 0.999280704 | 0.002371973 | 1.000000000 |
| Dynamic-Logdet | 0.990183179 | 0.999125224 | 0.002379910 | 1.000000000 |
| Static-EffRes | 0.969111755 | 0.991023587 | 0.008254548 | 1.000000000 |
| Degree | 0.962104360 | 0.990295029 | 0.010283279 | 1.000000000 |
| Random | 0.871457477 | 0.902561585 | 0.016262625 | 0.934758754 |

The worst-case submodular reference is 1 − 1/e = 0.632120559. Finite-instance ratios do not establish objective curvature or replace the approximation proof.

## Adaptive residual verification versus disabled verification

Adaptive and Uncertified use identical compiled initialization and vector kernels at each stated coarse tolerance. Uncertified always chooses the cached point maximum and never retries. Audit eigenvalues, independent high-order marginal evaluations and graph-automorphism checks occur outside timed algorithm runs. A candidate pair is called a structural tie only when an explicit current weighted-adjacency automorphism maps one edge to the other. A gap below10⁻¹⁴ without that certificate is listed separately as a numerical tie.

| Dataset | tol | Mode | R | Certified R_tie | Unproved numerical ties | R_res | Final cache max rel. error | Ordered match | Set match | True G | Reference G − G | Loss % | Total s mean ± SD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Small-BA | 1e-01 | Adaptive | 34 | 0 | 0 | 34 | 1.960201e-02 | 100.0% | 100.0% | 4.651877009 | -1.154632e-14 | -0.000000% | 0.013675 ± 0.000187 |
| Small-BA | 1e-01 | Uncertified | 0 | 0 | 0 | 0 | 1.964234e-02 | 80.0% | 100.0% | 4.651877009 | -1.154632e-14 | -0.000000% | 0.005210 ± 0.000703 |
| Small-BA | 1e-02 | Adaptive | 0 | 0 | 0 | 0 | 3.981990e-04 | 100.0% | 100.0% | 4.651877009 | -1.154632e-14 | -0.000000% | 0.007266 ± 0.000521 |
| Small-BA | 1e-02 | Uncertified | 0 | 0 | 0 | 0 | 3.981990e-04 | 100.0% | 100.0% | 4.651877009 | -1.154632e-14 | -0.000000% | 0.006179 ± 0.000282 |
| Small-BA | 1e-04 | Adaptive | 0 | 0 | 0 | 0 | 2.834766e-06 | 100.0% | 100.0% | 4.651877009 | -1.154632e-14 | -0.000000% | 0.018486 ± 0.021975 |
| Small-BA | 1e-04 | Uncertified | 0 | 0 | 0 | 0 | 2.834766e-06 | 100.0% | 100.0% | 4.651877009 | -1.154632e-14 | -0.000000% | 0.008226 ± 0.000664 |
| Small-BA | 1e-07 | Adaptive | 0 | 0 | 0 | 0 | 1.782684e-09 | 100.0% | 100.0% | 4.651877009 | -1.154632e-14 | -0.000000% | 0.011631 ± 0.000377 |
| Small-BA | 1e-07 | Uncertified | 0 | 0 | 0 | 0 | 1.782684e-09 | 100.0% | 100.0% | 4.651877009 | -1.154632e-14 | -0.000000% | 0.010722 ± 0.000173 |
| ca-GrQc | 1e-07 | Adaptive | 21 | 21 | 0 | 0 | 2.515575e-09 | 75.0% | 95.0% | 12.395819917 | +1.776357e-15 | +0.000000% | 0.780236 ± 0.023744 |
| ca-GrQc | 1e-07 | Uncertified | 0 | 0 | 0 | 0 | 2.515574e-09 | 75.0% | 95.0% | 12.395819917 | +1.776357e-15 | +0.000000% | 0.551314 ± 0.037955 |

The reference in the preceding table is dense spectral q12 surrogate greedy for Small-BA and floating-point tol10⁻¹² CELF for ca-GrQc. The latter is a tighter solve, not an exact symbolic tie-break oracle. The separately measured nominal tol10⁻⁷ CELF comparison is retained below; the tighter-reference numbers above are not replaced.

| Dataset | Mode | Nominal ordered match | Nominal set match | Tighter-solve ordered match | Tighter-solve set match | Final spectral equivalence to tighter reference |
| --- | --- | --- | --- | --- | --- | --- |
| ca-GrQc | Adaptive | 100.0% | 100.0% | 75.0% | 95.0% | exact permutation similarity |
| ca-GrQc | Uncertified | 100.0% | 100.0% | 75.0% | 95.0% | exact permutation similarity |

Final ca-GrQc spectral-equality certificate: [[3108, 4491], [4347, 5079], [4491, 3108], [5079, 4347]]. This vertex permutation maps the observed final augmented matrix to the tighter-reference final matrix with exactly zero sparse difference entries. All represented matrix entries are integers; therefore every spectral trace is identical despite different literal tied edge representatives.

The numerical comparison tolerance used by the implementation is2×10⁻¹¹ at the smallest solve tolerance; it is explicitly distinct from the10⁻¹⁴ diagnostic tie threshold. Ordered matches and unordered top-k matches are reported separately. Complete events include the current selected prefix, competing candidates, observed intervals, independently evaluated margins, and any exact automorphism certificate.


## Reproduction
```bash
python fastfrac_final_unified_experiments.py --threads 8 --repeats 5
python unified_routes_postprocess.py
python build_fastfrac_unified_report.py
```
The JSON files retain selected edge indices, individual repetition times, exact graph and candidate data, per-RHS iterations, per-round counts, independent reference values, and equality certificates.
