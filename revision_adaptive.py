"""Residual-interval adaptive FastFrac-Inc supplement (experiment 6).

The coarse tolerance is used for initialization and first selected-edge solves.
Every rejected/refined heap top is an additional online RHS call R.  Reference
spectra only audit results; they never select or certify an edge for the method.
All timing and R values are observed, not prescribed by the desired narrative.
"""
from pathlib import Path
from time import perf_counter
from dataclasses import asdict
import heapq
import json

import numpy as np
from scipy import sparse
from scipy.linalg import eigh, eigvalsh
from threadpoolctl import threadpool_limits

import fastfrac_cpu_benchmark as ff


class DeadlineReached(RuntimeError):
    pass


def _check(deadline):
    if perf_counter() >= deadline:
        raise DeadlineReached("experiment deadline reached")


def _small_ba():
    path = Path(__file__).resolve().parent / 'fastfrac_cpu_results/Small-BA.json'
    locked = json.loads(path.read_text())
    edges = np.asarray(locked['base_edges'], dtype=np.int64)
    C = np.asarray(locked['candidates'], dtype=np.int64)
    n = int(locked['metadata']['n'])
    u, v = edges.T
    data = np.tile([1., 1., -1., -1.], (len(edges), 1)).ravel()
    row = np.column_stack((u, v, u, v)).ravel()
    col = np.column_stack((u, v, v, u)).ravel()
    L = sparse.coo_matrix((data, (row, col)), shape=(n, n)).tocsr()
    A0 = (sparse.eye(n, format='csr') + L).tocsr()
    return A0, C, locked


def _exact_resistances(A, C, shifts):
    lam, V = eigh(A.toarray(), check_finite=False, overwrite_a=True)
    difference = V[C[:, 0]] - V[C[:, 1]]
    R = (difference * difference) @ (1. / (lam[:, None] + shifts))
    return R, lam


def _maxerr(point, true, alive=None):
    if alive is not None:
        point, true = point[alive], true[alive]
    if not len(point):
        return 0.
    return float(np.max(np.abs(point - true) / true))


def _exact_surrogate_reference(A0, C, k, shifts, weights, deadline):
    A = A0.copy()
    chosen = []
    alive = np.ones(len(C), dtype=bool)
    t0 = perf_counter()
    for r in range(k):
        _check(deadline)
        R, _ = _exact_resistances(A, C, shifts)
        values = ff.scores(R, weights)
        values[~alive] = -np.inf
        i = int(np.argmax(values))
        chosen.append(i)
        alive[i] = False
        A = ff.add_edge(A, C[i])
    return chosen, perf_counter() - t0


def _adaptive_one(A0, C, k, shifts, weights, upper, tol, reference,
                  lam0, deadline):
    t0 = perf_counter()
    st = ff.Stats()
    A = A0.copy()
    chosen = []
    alive = np.ones(len(C), dtype=bool)
    audit_seconds = 0.
    _check(deadline)
    R, lo, hi = ff.scalar_ms_batch(A, C, shifts, tol, st)
    init_seconds = perf_counter() - t0
    initial_calls = st.calls
    audit_start = perf_counter()
    exact_R, _ = _exact_resistances(A, C, shifts)
    initial_error = _maxerr(R, exact_R)
    audit_seconds += perf_counter() - audit_start
    rejected = 0
    rounds = []
    max_interval_violation = 0.
    max_before, max_after = 0., 0.
    stopped = None
    try:
        for r in range(k):
            _check(deadline)
            before = _maxerr(R, exact_R, alive)
            max_before = max(max_before, before)
            upper_scores = ff.scores(hi, weights)
            heap = [(-float(upper_scores[i]), int(i))
                    for i in np.flatnonzero(alive)]
            heapq.heapify(heap)
            round_calls0 = st.calls
            round_refines0 = rejected
            localtol = tol
            while True:
                _check(deadline)
                _, i = heapq.heappop(heap)
                d, dl, du, X, rr = ff.vector_ms(A, C[i], shifts, localtol, st)
                # Fresh variational values, including the denominator used
                # in the subsequent Sherman--Morrison correction.
                R[i], lo[i], hi[i] = d, dl, du
                actual = float(ff.scores(d, weights))
                lower = float(ff.scores(dl, weights))
                upper_i = float(ff.scores(du, weights))
                competitor = -heap[0][0] if heap else -np.inf
                if lower > competitor:
                    acceptance = 'separated intervals'
                    break
                # Same explicit comparison tolerance as the original code.
                if localtol <= 1e-12 and actual >= competitor - 2e-11 * max(actual, 1.):
                    acceptance = 'declared 2e-11 numerical tie tolerance'
                    break
                heapq.heappush(heap, (-upper_i, i))
                rejected += 1
                if upper_i >= competitor:
                    localtol = max(localtol * .01, 1e-12)
                else:
                    localtol = tol

            after = _maxerr(R, exact_R, alive)
            max_after = max(max_after, after)
            violation = np.maximum(lo - exact_R, exact_R - hi)
            max_interval_violation = max(max_interval_violation,
                float(np.max(np.maximum(violation[alive], 0.) / exact_R[alive])))
            chosen.append(i)
            cross = X[C[:, 0]] - X[C[:, 1]]
            R -= cross ** 2 / (1. + d)
            delta = np.sqrt(2.) * rr / (1. + shifts)
            correction_lo = np.maximum(np.abs(cross) - delta, 0.) ** 2 / (1. + du)
            correction_hi = (np.abs(cross) + delta) ** 2 / (1. + dl)
            lo = np.maximum(0., lo - correction_hi)
            hi = np.maximum(lo, hi - correction_lo)
            alive[i] = False
            A = ff.add_edge(A, C[i])
            audit_start = perf_counter()
            exact_R, lam = _exact_resistances(A, C, shifts)
            post_update = _maxerr(R, exact_R, alive)
            audit_seconds += perf_counter() - audit_start
            rounds.append(dict(round=r+1, selected=i,
                online_rhs_calls=st.calls-round_calls0,
                extra_rechecks=rejected-round_refines0,
                max_cache_relerr_before_rechecks=before,
                max_cache_relerr_after_rechecks=after,
                max_cache_relerr_after_update=post_update,
                selected_vector_actual_relative_residual=float(np.max(rr)/np.sqrt(2.)),
                selected_vector_final_requested_tolerance=float(localtol),
                acceptance=acceptance))
    except (DeadlineReached, TimeoutError) as e:
        stopped = str(e)

    final_error = _maxerr(R, exact_R, alive)
    final_interval_error = float(np.max(np.maximum(
        np.maximum(lo[alive] - exact_R[alive], exact_R[alive] - hi[alive]), 0.) /
        exact_R[alive])) if np.any(alive) else 0.
    max_interval_violation = max(max_interval_violation, final_interval_error)
    # Lemma 3's sufficient (not necessary) prescribed budget at nu=0.01.
    nu = .01
    a = float(1. + shifts.min())
    beta = 1. + 2. * k / a
    epsilon0_budget = nu / (2. * beta)
    zeta_budget = min(1., a / 2., nu * a*a / (16.*k*(upper+shifts.min())))
    selected_residual = max((r['selected_vector_actual_relative_residual'] for r in rounds), default=0.)
    matching = sum(i == j for i, j in zip(chosen, reference))
    total_seconds = perf_counter() - t0
    complete = len(chosen) == k
    result = dict(dataset='Small-BA', tol=tol, status='complete' if complete else 'TLE',
        completed_rounds=len(chosen), k=k, n=A0.shape[0], M=len(C), q=len(shifts),
        init_seconds=init_seconds, algorithm_seconds=total_seconds-audit_seconds,
        wall_seconds_including_audits=total_seconds, independent_audit_seconds=audit_seconds,
        init_rhs=initial_calls, online_rhs=st.calls-initial_calls,
        R=st.calls-initial_calls-len(chosen), rejected_or_refined_heap_tops=rejected,
        selected=chosen, reference_selected=reference,
        positional_match_count=matching,
        positional_match_percent=100.*matching/k,
        set_overlap_count=len(set(chosen)&set(reference)),
        set_overlap_percent=100.*len(set(chosen)&set(reference))/k,
        initial_cache_max_relerr=initial_error,
        final_cache_max_relerr=final_error,
        max_cache_relerr_before_rechecks=max_before,
        max_cache_relerr_after_rechecks=max_after,
        max_true_resistance_outside_interval_relative=max_interval_violation,
        actual_max_relative_residual_all_vector_solves=st.max_true_residual,
        actual_max_selected_vector_residual=selected_residual,
        lemma3_target_nu=nu, lemma3_initial_relerr_budget=epsilon0_budget,
        lemma3_selected_residual_budget=zeta_budget,
        lemma3_prescribed_sufficient_budget_met=bool(
            initial_error <= epsilon0_budget and selected_residual <= zeta_budget),
        per_round=rounds, stats=asdict(st))
    if complete:
        result['true_gain'] = float(np.sum(lam0**.5*np.expm1(.5*np.log1p((lam-lam0)/lam0))))
    else:
        result['true_gain'] = None
        result['stop_reason'] = stopped
    return result


def run_adaptive(deadline, output_dir):
    """Run with an absolute perf_counter deadline and write experiment6.json."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    result = dict(experiment=6, name='Adaptive residual-interval maintenance',
        rows=[], status='pending',
        protocol={'coarse_tolerance_applies_to':'initialization and first vector solve per popped candidate',
            'adaptive_refinement':'tol *= 0.01, floor 1e-12; freshly computed variational denominator',
            'R_definition':'online multi-shift RHS calls minus completed greedy rounds',
            'reference':'independent dense spectral evaluation of every q=12 surrogate marginal',
            'reference_usage':'audit only; never used to choose or certify algorithm edges',
            'rounding':'double-precision residual intervals with the existing 5e-12 relative padding; not directed rounding',
            'initialization':'all M scalar shifted resistances, with the same requested tolerance',
            'final_cache_error_domain':'remaining unselected candidates at all q shifts',
            'sequence_match':'position-wise exact edge index agreement, separately from set overlap',
            'timing':'algorithm_seconds excludes independently measured dense-spectrum audit time',
            'target_guarantee_nu':0.01,
            'seed_policy':'locked Small-BA graph and candidate pool; no seed search'})
    with threadpool_limits(limits=1):
        try:
            _check(deadline)
            A0, C, locked = _small_ba()
            k = int(locked['metadata']['k'])
            upper = float(locked['metadata']['upper'])
            shifts, weights, eta = ff.rule(.5, 12, upper)
            result['quadrature_eta_bound'] = eta
            lam0 = eigvalsh(A0.toarray(), check_finite=False)
            reference, seconds = _exact_surrogate_reference(A0, C, k, shifts, weights, deadline)
            result['reference_seconds'] = seconds
            result['reference_selected'] = reference
            locked_ref = locked.get('runs', {}).get('Exact-Eig', {}).get('selected')
            result['matches_locked_exact_eig_sequence'] = reference == locked_ref if locked_ref is not None else None
            ff.write_json(out/'experiment6_adaptive.json', result)
            for tol in (1e-1, 1e-2, 1e-4, 1e-7):
                _check(deadline)
                row = _adaptive_one(A0, C, k, shifts, weights, upper, tol,
                                    reference, lam0, deadline)
                result['rows'].append(row)
                ff.write_json(out/'experiment6_adaptive.json', result)
                if row['status'] != 'complete':
                    break
            result['status'] = 'complete' if len(result['rows']) == 4 and all(
                x['status'] == 'complete' for x in result['rows']) else 'TLE'
        except (DeadlineReached, TimeoutError) as e:
            result['status'] = 'TLE'
            result['stop_reason'] = str(e)
    result['wall_seconds'] = perf_counter() - start
    ff.write_json(out/'experiment6_adaptive.json', result)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=8.)
    parser.add_argument('--output', default='fastfrac_revision_results')
    args = parser.parse_args()
    answer = run_adaptive(perf_counter()+args.seconds, args.output)
    print(json.dumps({k:v for k,v in answer.items() if k != 'rows'}, indent=2))
    for row in answer['rows']:
        print(json.dumps({k:row[k] for k in ('tol','status','R','final_cache_max_relerr',
                         'positional_match_percent','algorithm_seconds')}))
