"""Supplement groups 4 and 5; called by the bounded revision orchestrator.

No experiment is executed on import.  ``deadline`` is an absolute
``time.perf_counter()`` timestamp.  Incomplete runs report TLE and never
substitute an extrapolation for an actual objective value or optimum.
"""
from pathlib import Path
from time import perf_counter
from itertools import combinations
import hashlib
import json
import math

import numpy as np
from scipy import sparse
from scipy.linalg import eigvalsh
from threadpoolctl import threadpool_limits
import fastfrac_cpu_benchmark as ff

ROOT = Path(__file__).resolve().parent
S_VALUES = (0.10, 0.50, 0.85)
SEED = 20261031


class BudgetExpired(RuntimeError):
    pass


def _guard(deadline):
    if perf_counter() >= deadline:
        raise BudgetExpired('absolute wall-clock budget expired')


def _write(output_dir, filename, value):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ff.write_json(output_dir / filename, value)


def _matrix(G, alpha):
    L = ff.nx.laplacian_matrix(G, nodelist=range(len(G)), weight='weight').astype(float).tocsr()
    return (sparse.eye(len(G), format='csr') + alpha * L).tocsr()


def _original_sbm():
    path = ROOT / 'fastfrac_cpu_results/fastfrac_cpu_supplement.json'
    locked = json.loads(path.read_text())['multiscale']
    G = ff.nx.Graph()
    G.add_nodes_from(range(locked['metadata']['n']))
    G.add_weighted_edges_from(locked['base_edges'])
    C = np.asarray(locked['candidates'], dtype=np.int64)
    labels = np.asarray(locked['communities'], dtype=np.int64)
    meta = dict(locked['metadata'])
    meta['source'] = 'Exact reconstruction of the locked multiscale benchmark JSON.'
    meta['source_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return G, C, labels, meta


def _heterogeneous_sbm():
    """One prespecified heterogeneous graph; no seed/parameter search.

    Four 100-node preferential-attachment blocks have distinct edge-weight
    scales.  Three weak bridges connect them.  The candidate pool covers both
    peripheral and core endpoints, so selection is not restricted to one class.
    """
    n_block = 100
    scales = (0.2, 0.7, 2.0, 6.0)
    seed = SEED + 4701
    rng = np.random.default_rng(seed + 1)
    G = ff.nx.Graph()
    G.add_nodes_from(range(4 * n_block))
    for block, scale in enumerate(scales):
        H = ff.nx.barabasi_albert_graph(n_block, 3, seed=seed + 10 * block)
        for u, v in sorted(H.edges()):
            weight = float(scale * np.exp(rng.normal(0.0, 0.35)))
            G.add_edge(block * n_block + u, block * n_block + v, weight=weight)
    weak = ((0, 100, 0.002), (100, 200, 0.02), (200, 300, 0.2))
    G.add_weighted_edges_from(weak)
    labels = np.repeat(np.arange(4), n_block)
    weighted_degree = np.asarray([G.degree(i, weight='weight') for i in range(len(G))])
    low, high = [], []
    for block in range(4):
        ids = np.arange(block * n_block, (block + 1) * n_block)
        order = ids[np.lexsort((ids, weighted_degree[ids]))]
        low.append(order[:25])
        high.append(order[-25:])
    candidate_rng = np.random.default_rng(seed + 101)
    candidate = set()
    for idx, (a, b) in enumerate(combinations(range(4), 2)):
        quota = 13 if idx < 3 else 12
        pools = [sorted((int(u), int(v)) for u in vertices[a] for v in vertices[b]
                        if not G.has_edge(int(u), int(v))) for vertices in (low, high)]
        quotas = (quota // 2, quota - quota // 2)
        for pool, size in zip(pools, quotas):
            for i in candidate_rng.choice(len(pool), size, replace=False):
                candidate.add(pool[int(i)])
    for block in range(4):
        ids = sorted(map(int, high[block]))
        pool = [(u, v) for u, v in combinations(ids, 2) if not G.has_edge(u, v)]
        for i in candidate_rng.choice(len(pool), 19 if block < 3 else 18, replace=False):
            candidate.add(pool[int(i)])
    C = np.asarray(sorted(candidate), dtype=np.int64)
    assert len(C) == 150
    return G, C, labels, dict(name='Heterogeneous-Multiscale', n=len(G),
        m=G.number_of_edges(), M=len(C), k=15, alpha=50.0, q=12, tol=1e-7,
        seed=seed, candidate_seed=seed + 101,
        generator='4 BA(100,3) blocks; independent lognormal edge multipliers, sigma=0.35',
        intra_weight_scales=list(scales), weak_bridges=[list(x) for x in weak],
        candidate_definition='75 cross-block candidates, split between bottom/top weighted-degree quartiles; 75 intrablock top-quartile nonedges',
        connected_components=ff.nx.number_connected_components(G),
        candidate_sha256=hashlib.sha256(C.astype('<i8').tobytes()).hexdigest(),
        graph_parameter_search=False)


def _structural_metrics(G, C, selected, labels):
    neighbors = [set(G[i]) for i in range(len(G))]
    balls2 = []
    for i, one in enumerate(neighbors):
        ball = set(one)
        for v in one:
            ball.update(neighbors[v])
        ball.discard(i)
        balls2.append(ball)
    degree = np.asarray([G.degree(i) for i in range(len(G))], dtype=float)
    wdegree = np.asarray([G.degree(i, weight='weight') for i in range(len(G))], dtype=float)
    dsum, wdsum, common2, egoedges, reach2 = [], [], [], [], []
    for index in selected:
        u, v = map(int, C[index])
        dsum.append(float(degree[u] + degree[v]))
        wdsum.append(float(wdegree[u] + wdegree[v]))
        common2.append(len((balls2[u] & balls2[v]) - {u, v}))
        ego = neighbors[u] | neighbors[v] | {u, v}
        egoedges.append(sum(len(neighbors[x] & ego) for x in ego) // 2)
        reach2.append(len(balls2[u]) + len(balls2[v]))
    return dict(mean_endpoint_degree_sum=float(np.mean(dsum)),
        mean_endpoint_weighted_degree_sum=float(np.mean(wdsum)),
        mean_shared_two_hop_neighbors=float(np.mean(common2)),
        mean_one_hop_ego_union_induced_edges=float(np.mean(egoedges)),
        mean_endpoint_two_hop_reach_sum=float(np.mean(reach2)),
        selected_cross_block_edges=int(sum(labels[C[i, 0]] != labels[C[i, 1]] for i in selected)),
        per_edge=dict(endpoint_degree_sum=dsum, endpoint_weighted_degree_sum=wdsum,
                      shared_two_hop_neighbors=common2, one_hop_ego_union_induced_edges=egoedges))


def _stable_gain(base, changed, s):
    return float(np.sum(base ** s * np.expm1(s * np.log1p((changed - base) / base))))


def run_structure(deadline, output_dir=None):
    """Dynamic-logdet controls plus prespecified structural diagnostics."""
    start = perf_counter()
    result = dict(status='running', rows=[], structure_rows=[], workloads=[],
        definitions={
            'degree': 'unweighted degree in the unchanged base graph; weighted degree reported separately',
            'shared_two_hop_neighbors': '|B_2(u) intersection B_2(v) minus {u,v}|, where B_2 contains other vertices at graph distance at most 2, ignoring weights',
            'one_hop_ego_union_induced_edges': 'number of undirected base edges induced by N(u) union N(v) union {u,v}; candidate edge not added',
            'metric_reference_graph': 'all structural metrics use the unchanged base graph',
            'gain': 'independent full dense eigenvalues of I + alpha L before and after the complete selected set',
            'overlap': 'intersection size with the dynamically selected log-determinant edge set',
            'interpretation': 'No monotone structural trend or positive advantage over dynamic logdet is assumed.'})
    with threadpool_limits(limits=1):
        try:
            for factory in (_original_sbm, _heterogeneous_sbm):
                _guard(deadline)
                G, C, labels, meta = factory()
                alpha = float(meta['alpha']); k = int(meta['k'])
                A0 = _matrix(G, alpha)
                weighted_degree = np.asarray([G.degree(i, weight='weight') for i in range(len(G))])
                upper = float(1 + 2 * alpha * (weighted_degree.max() + k))
                workload = dict(metadata={**meta, 'upper': upper}, methods={}, status='running',
                    base_edges=[(int(u), int(v), float(d.get('weight', 1))) for u, v, d in sorted(G.edges(data=True))],
                    candidates=C.tolist(), communities=labels.tolist())
                result['workloads'].append(workload)
                _guard(deadline)
                logdet = ff.run_inc(A0, C, k, np.array([0.0]), np.array([1.0]), 1e-9, alpha=alpha)
                workload['methods']['Dynamic-Logdet'] = logdet
                for s in S_VALUES:
                    _guard(deadline)
                    shifts, weights, eta = ff.rule(s, 12, upper)
                    run = ff.run_celf(A0, C, k, shifts, weights, 1e-7, alpha=alpha)
                    run['eta_q12'] = eta
                    workload['methods'][f'FastFrac(s={s:.2f})'] = run
                _guard(deadline)
                base = eigvalsh(A0.toarray(), check_finite=False)
                spectra = {}
                for name, run in workload['methods'].items():
                    _guard(deadline)
                    selected = run['selected']
                    key = tuple(sorted(selected))
                    if key not in spectra:
                        A = A0.toarray()
                        for i in selected:
                            u, v = C[i]
                            A[u, u] += alpha; A[v, v] += alpha
                            A[u, v] -= alpha; A[v, u] -= alpha
                        spectra[key] = eigvalsh(A, check_finite=False, overwrite_a=True)
                    run['true_gains_by_s'] = {str(s): _stable_gain(base, spectra[key], s) for s in S_VALUES}
                    metrics = _structural_metrics(G, C, selected, labels)
                    run['structure'] = metrics
                    result['structure_rows'].append(dict(dataset=meta['name'], method=name,
                        **{key: value for key, value in metrics.items() if key != 'per_edge'}))
                selected_log = set(logdet['selected'])
                for s in S_VALUES:
                    run = workload['methods'][f'FastFrac(s={s:.2f})']
                    ours = run['true_gains_by_s'][str(s)]
                    other = logdet['true_gains_by_s'][str(s)]
                    result['rows'].append(dict(dataset=meta['name'], s=s,
                        fastfrac_gain=ours, dynamic_logdet_gain=other,
                        fastfrac_gain_over_dynamic_logdet_pct=100 * (ours / other - 1),
                        overlap_with_logdet=len(set(run['selected']) & selected_log), k=k,
                        fastfrac_selected=run['selected'], dynamic_logdet_selected=logdet['selected']))
                workload['status'] = 'complete'
                _write(output_dir, 'structure_dynamic_logdet.json', result)
                print(f'REVISION structure {meta["name"]}: complete', flush=True)
            result['status'] = 'complete'
        except (BudgetExpired, TimeoutError) as e:
            result['status'] = 'TLE'
            result['stop_reason'] = str(e)
    result['seconds'] = perf_counter() - start
    _write(output_dir, 'structure_dynamic_logdet.json', result)
    return result


def _small_instance(index):
    seed = SEED + 50000 + index
    rng = np.random.default_rng(seed)
    G = ff.nx.from_prufer_sequence(rng.integers(0, 25, size=23).tolist())
    for u, v in combinations(range(25), 2):
        if not G.has_edge(u, v) and rng.random() < 0.08:
            G.add_edge(u, v)
    nonedges = np.asarray(sorted(ff.nx.non_edges(G)), dtype=np.int64)
    C = nonedges[rng.choice(len(nonedges), 15, replace=False)]
    C = C[np.lexsort((C[:, 1], C[:, 0]))]
    return G, _matrix(G, 1.0), C, seed


def run_opt(deadline, output_dir=None):
    """Enumerate every 4-subset of 15 candidates on each completed instance."""
    start = perf_counter()
    result = dict(status='running', instances=[], requested_instances=20,
        n=25, M=15, k=4, s=0.5, alpha=1.0, q=12, tol=1e-7,
        subsets_per_instance=math.comb(15, 4), exhaustive_total_requested=20 * math.comb(15, 4),
        theoretical_reference=1 - math.exp(-1),
        graph_generator='uniform random labeled tree (Pruefer sequence) plus independent nonedges with probability 0.08',
        exhaustive_solver='numpy.linalg.eigvalsh on batches of 128 real symmetric 25x25 matrices',
        interpretation='Finite-instance evidence against the exact optimum; not a proof of the worst-case approximation theorem.')
    combos = np.asarray(list(combinations(range(15), 4)), dtype=np.int64)
    with threadpool_limits(limits=1):
        try:
            for instance in range(20):
                _guard(deadline)
                begin = perf_counter()
                G, A0, C, seed = _small_instance(instance)
                A = A0.toarray()
                lam0 = np.linalg.eigvalsh(A)
                updates = np.zeros((15, 25, 25))
                for i, (u, v) in enumerate(C):
                    updates[i, u, u] = updates[i, v, v] = 1
                    updates[i, u, v] = updates[i, v, u] = -1
                optimum = -math.inf; best = None; tested = 0
                t_enum = perf_counter()
                for first in range(0, len(combos), 128):
                    _guard(deadline)
                    batch = combos[first:first + 128]
                    matrices = A[None, :, :] + updates[batch].sum(axis=1)
                    lam = np.linalg.eigvalsh(matrices)
                    gains = np.sum(np.sqrt(lam0)[None, :] *
                        np.expm1(0.5 * np.log1p((lam - lam0[None, :]) / lam0[None, :])), axis=1)
                    where = int(np.argmax(gains))
                    if gains[where] > optimum:
                        optimum = float(gains[where]); best = batch[where].tolist()
                    tested += len(batch)
                enum_seconds = perf_counter() - t_enum
                assert tested == 1365
                upper = float(1 + 2 * (max(dict(G.degree()).values()) + 4))
                shifts, weights, eta = ff.rule(0.5, 12, upper)
                _guard(deadline)
                celf = ff.run_celf(A0, C, 4, shifts, weights, 1e-7)
                _guard(deadline)
                inc = ff.run_inc(A0, C, 4, shifts, weights, 1e-7)
                row = dict(instance=instance + 1, seed=seed, m=G.number_of_edges(),
                    optimum=optimum, optimum_selected=best, subsets_enumerated=tested,
                    enumeration_seconds=enum_seconds, eta_q12=eta,
                    base_edges=sorted([list(e) for e in G.edges()]), candidates=C.tolist(),
                    same_sequence=celf['selected'] == inc['selected'])
                for name, run in (('celf', celf), ('inc', inc)):
                    lam = np.linalg.eigvalsh(A + updates[run['selected']].sum(axis=0))
                    gain = _stable_gain(lam0, lam, 0.5)
                    row[name] = dict(selected=run['selected'], gain=gain,
                        ratio=gain / optimum, seconds=run['seconds'], rhs_calls=run['calls'])
                row['seconds'] = perf_counter() - begin
                result['instances'].append(row)
                _write(output_dir, 'global_optimum.json', result)
            result['status'] = 'complete'
        except (BudgetExpired, TimeoutError) as e:
            result['status'] = 'TLE'
            result['stop_reason'] = str(e)
    result['completed_instances'] = len(result['instances'])
    result['subsets_enumerated_on_completed_instances'] = 1365 * len(result['instances'])
    result['summary'] = {}
    for name in ('celf', 'inc'):
        ratios = [row[name]['ratio'] for row in result['instances']]
        if ratios:
            result['summary'][name] = dict(min_ratio=float(np.min(ratios)),
                mean_ratio=float(np.mean(ratios)), max_ratio=float(np.max(ratios)),
                std_ratio=float(np.std(ratios, ddof=1)) if len(ratios) > 1 else 0.0,
                instances=len(ratios), below_one_minus_one_over_e=sum(x < 1 - math.exp(-1) for x in ratios))
    result['seconds'] = perf_counter() - start
    _write(output_dir, 'global_optimum.json', result)
    return result
