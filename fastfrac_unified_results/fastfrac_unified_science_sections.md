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

