"""Generate the complete unified Markdown report exclusively from saved runs."""
from pathlib import Path
import json
import numpy as np
from unified_science import render_science_markdown,add_nominal_reference_audit
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'fastfrac_unified_results'

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'
def ms(d):return f'{d["mean"]:.6f} ± {d["std"]:.6f}'
def load(p):return json.loads(p.read_text())
def main():
    text=['# FastFrac: unified CPU results\n',
      'All new performance rows use the same compiled C++17/OpenMP backend with a maximum of 8 worker threads and one BLAS thread. Each algorithm starts from the same base graph and candidate bytes in five fresh timing repetitions; the tables retain the completed repetition counts. Standard deviations are sample SDs (ddof=1). Compilation, graph generation, file I/O, and independent accuracy evaluations are excluded from individual optimization times and recorded separately where applicable. Every reported speedup is computed from these new measurements.\n',
      'For the performance runs, Phase 1 computes the initial candidate-by-shift resistance/error table. Phase 2 includes scoring, table copies, heap setup, every accepted edge, cache maintenance, true-residual checks, and adaptive rechecks. Each method and repetition pays for its own freshly measured initialization. Dynamic-Logdet is CELF on the single unshifted system (t=0), with the same tolerance and backend. One incidence RHS means one shared traversal for all shifts. SLQ probe-RHS counts have a different meaning and are labelled separately.\n',
      'The scientific sweeps use the same numerical kernels but retain their existing controller timing boundary, which can include initial scoring in the initialization timer. Compare **total times only** across those scientific sweeps. All Phase-1/Phase-2 comparisons here and in the paper use the strict scalar-table boundary in the `performance/` measurements.\n',
      'BA-50k Naive below executes all 20 rounds in every repetition. BA-100k Naive measures its first round five times and projects the full-run time; its projected total is explicitly separate from completed measurements.\n']
    names=['Small-BA','Medium-WS','Large-PowerCluster','email-Eu-core','ca-GrQc','ca-CondMat','BA-50k','BA-100k']
    text.append('## 1. Unified main performance: five repetitions\n');rows=[]
    for name in names:
        p=OUT/'performance'/f'{name}.json'
        if not p.exists():continue
        d=load(p);meta=d['metadata']
        for method,summary in d.get('summary',{}).items():
            rows.append([name,f'{meta["n"]:,}',f'{meta["m"]:,}',meta['M'],meta['k'],method,
             summary['repetitions'],ms(summary['init_seconds']),ms(summary['online_seconds']),ms(summary['seconds']),
             int(summary['online_calls']['mean']),int(summary['calls']['mean']),
             f'{summary["mean_iterations"]["mean"]:.3f}',f'{summary["krylov_spmv"]["mean"]:.0f}'])
    text.append(table(['Dataset','n','m','M','k','Method','Runs','Init (s), mean ± SD','Online (s), mean ± SD','Total (s), mean ± SD','Online RHS','Total RHS','Mean h/RHS','Krylov SpMV'],rows))
    rows=[]
    for name in names:
        p=OUT/'performance'/f'{name}.json'
        if not p.exists():continue
        d=load(p);ss=d.get('summary',{})
        if 'Naive' in ss:
            for m in ('CELF','Inc','Dynamic-Logdet'):
                if m not in ss:continue
                rows.append([name,m,f'{ss["Naive"]["seconds"]["mean"]/ss[m]["seconds"]["mean"]:.3f}',
                  f'{ss["Naive"]["online_seconds"]["mean"]/ss[m]["online_seconds"]["mean"]:.3f}',
                  d.get('celf_inc_identical_sequence','—')])
    text.append(table(['Dataset','Method','Total speedup vs Naive','Online speedup vs Naive','CELF/Inc identical ordered sequence'],rows))
    p=OUT/'performance'/'BA-100k.json'
    if p.exists() and 'Naive_one_round' in (d:=load(p)):
        a=d['Naive_one_round'];v=np.array([r['seconds'] for r in a]);pr=np.array([r['projected_total_seconds'] for r in a])
        text.append(f'BA-100k Naive: the {len(a)} measured first-round times are {v.tolist()} seconds; their mean ± SD is {v.mean():.6f} ± {v.std(ddof=1):.6f} s. The 20-round projection is {pr.mean():.6f} ± {pr.std(ddof=1):.6f} s under constant per-candidate cost. These projected totals are distinct from completed greedy runs.\n')
    text.append('## 2. Graph-size, candidate-pool, and budget curves\n')
    text.append('### 2.1 Six-size BA scaling: complete CELF and Inc runs\n')
    rows=[]
    for n in (1000,5000,10000,25000,50000,100000):
        p=OUT/'performance'/f'BA-{n//1000}k.json'
        if not p.exists():continue
        d=load(p);meta=d['metadata']
        for method in ('CELF','Inc'):
            ss=d.get('summary',{}).get(method)
            if ss is None:continue
            rows.append([meta['name'],meta['n'],meta['m'],meta['M'],meta['k'],method,
                ss['repetitions'],ms(ss['init_seconds']),ms(ss['online_seconds']),ms(ss['seconds']),
                int(ss['online_calls']['mean']),int(ss['calls']['mean'])])
    text.append(table(['Dataset','n','m','M','k','Method','Runs','Init (s), mean ± SD',
        'Online (s), mean ± SD','Total (s), mean ± SD','Online RHS','Total RHS'],rows))
    text.append('### 2.2 Candidate-pool and budget sweeps\n')
    rows=[]
    for p in sorted((OUT/'curves').glob('*.json')):
        d=load(p);meta=d['metadata']
        for m,ss in d.get('summary',{}).items():
            rows.append([meta['name'],meta['M'],meta['k'],m,ms(ss['seconds']),ms(ss['online_seconds']),int(ss['calls']['mean'])])
    text.append(table(['Configuration','M','k','Method','Total (s), mean ± SD','Online (s), mean ± SD','RHS'],rows))
    text.append('### 2.3 Iteration counts and sparse-product costs\n')
    rows=[];fits=[]
    for n in (1000,5000,10000,25000,50000,100000):
        p=OUT/'performance'/f'BA-{n//1000}k.json'
        if not p.exists():continue
        d=load(p);ss=d.get('summary',{}).get('CELF')
        if ss is None:continue
        init_iters=np.array([r['init_counters']['logical_spmv']/d['metadata']['M'] for r in d['runs']['CELF']])
        sp=d.get('single_spmv_microbenchmark',[]);spmean=float(np.mean([r['seconds_per_spmv'] for r in sp])) if sp else None
        init=ss['init_seconds']['mean'];h=init_iters.mean();cost=init/(d['metadata']['M']*h)
        fits.append((n,init,h,cost))
        rows.append([n,d['metadata']['m'],d['metadata']['max_degree'],f'{h:.4f}',
           f'{init:.6f}',f'{1e6*spmean:.4f}' if spmean is not None else 'pending',f'{1e6*cost:.6f}',
           int(ss['krylov_spmv']['mean']),int(ss['executed_padded_spmv']['mean'])])
    text.append(table(['n','m','dmax','Mean initialization h','Initialization mean (s)',
        'Single CSR SpMV (µs; one thread)','Effective cost/logical SpMV (µs; eight-thread run)',
        'Total logical SpMV','Total padded SpMV equivalents'],rows))
    if len(fits)>2:
        x=np.log(np.array(fits)[:,0]);v=np.array(fits)
        betas=[float(np.polyfit(x,np.log(v[:,i]),1)[0]) for i in (1,2,3)]
        text.append(f'Log-log OLS exponents across the {len(fits)} available BA sizes: initialization time = {betas[0]:.6f}; average Krylov iterations = {betas[1]:.6f}; effective cost per logical SpMV = {betas[2]:.6f}. Their additive identity follows from T_init = M × hbar × effective_cost, with fixed M. The last factor includes recurrence arithmetic, batching, memory behavior, and parallel overhead. This decomposition measures both iteration growth and effective per-iteration cost.\n')
    text.append('A CSR SpMV performs exactly nnz(A) multiply-add pairs plus O(n) work, giving O(n+m) arithmetic complexity. Its wall time also reflects cache and memory regimes. The unpreconditioned worst-case CG estimate is O(sqrt(kappa) log(1/tol)). Combining the BA maximum-degree scale dmax = O(n^(1/2)) with kappa ≤ 1 + 2αdmax gives an O(n^(1/4)) iteration upper-bound factor at fixed α. For fixed q and s with the balanced shift scale τ = √Λ, the smallest node has t_min = c√Λ for fixed c > 0, giving κ = (Λ+c√Λ)/(1+c√Λ). The asymptotic shifted-CG bound is then O(Λ^(1/4) log(1/tol)); combined with that BA degree scale, it gives an n^(1/8) upper-bound factor. These worst-case bounds are distinct from an empirical n^0.4 iteration law. The table separately records one-thread CSR microbenchmark times and the effective cost derived from eight-thread initialization runs.\n')
    text.append('## 3. Multiscale competition with the identical compiled backend\n');rows=[];competition=[]
    for p in sorted(set((OUT/'performance').glob('*Multiscale*.json'))|set((OUT/'performance').glob('Multi-Scale*.json'))):
        d=load(p);m=d['metadata']
        for method,ss in d.get('summary',{}).items():
            r=d['runs'][method][0];rows.append([m['name'],m['n'],m['m'],m['M'],m['k'],m['s'],method,
               ms(ss['init_seconds']),ms(ss['online_seconds']),ms(ss['seconds']),int(ss['online_calls']['mean']),int(ss['calls']['mean']),r['online_rhs_per_round']])
        ss=d.get('summary',{})
        if 'CELF' in ss and 'Inc' in ss:
            c,i=ss['CELF'],ss['Inc']
            competition.append([m['name'],m['n'],m['s'],
                f'{c["seconds"]["mean"]/i["seconds"]["mean"]:.6f}',
                f'{c["online_seconds"]["mean"]/i["online_seconds"]["mean"]:.6f}',
                f'{c["online_calls"]["mean"]/i["online_calls"]["mean"]:.6f}',
                f'{c["calls"]["mean"]/i["calls"]["mean"]:.6f}',
                d.get('celf_inc_identical_sequence','—')])
    text.append(table(['Dataset','n','m','M','k','s','Method','Init (s)','Online (s)','Total (s)',
        'Online RHS','Total RHS','Per-round online RHS'],rows))
    text.append('All ratios in the following table divide CELF by Inc. A time ratio above one is an Inc speedup; RHS ratios measure solve-count reduction independently of time. Time ratios use the corresponding five-run means.\n')
    text.append(table(['Dataset','n','s','Inc total speedup','Inc online speedup',
        'Online RHS reduction (CELF/Inc)','Total RHS reduction (CELF/Inc)',
        'Identical ordered sequence'],competition))
    p=OUT/'performance'/'Large-PowerCluster.json'
    if p.exists():
        d=load(p);trace_rows=[]
        for method in ('CELF','Inc','Dynamic-Logdet','Naive'):
            runs=d.get('runs',{}).get(method,[])
            if not runs:continue
            counts=runs[0]['online_rhs_per_round']
            trace_rows.append([method,len(runs),len(counts),sum(counts),
                all(r['online_rhs_per_round']==counts for r in runs),
                '`'+json.dumps(counts)+'`'])
        text.append('### Large-PowerCluster: raw online calls per accepted round\n')
        text.append('Each array is recorded after the initial scalar table has been computed and covers rounds 1 through k. The array from the first repetition is shown, together with whether all recorded repetitions have exactly the same array. Its sum is the online RHS count.\n')
        text.append(table(['Method','Runs','Rounds','Sum of online RHS','Same array in every run',
            'Raw per-round online RHS'],trace_rows))
    p=OUT/'evaluation_routes_unified.json'
    if p.exists():
        data=load(p);text.append('## 4. Same-backend marginal evaluator ablation\n');rows=[];lazy_rows=[];key_rows=[]
        for d in data['workloads']:
            for r in d['routes']:
                a=r.get('audit',{});v=r['runs'][0]
                rows.append([d['dataset'],r['name'],len(r['runs']),f'{r.get("seconds_mean",0):.6f} ± {r.get("seconds_std",0):.6f}',
                   v.get('calls','—'),v.get('probe_rhs','—'),v['krylov_spmv'],v.get('residual_spmv',0),
                   f'{a.get("max_marginal_relative_error",float("nan")):.6e}',f'{a.get("true_gain",float("nan")):.9f}',
                   f'{a.get("topk_set_overlap_count",0)}/{d["k"]}'])
            if 'slq_lazy_vs_same_probe_exhaustive' in d:
                x=d['slq_lazy_vs_same_probe_exhaustive']
                lazy_rows.append([d['dataset'],f'{x["set_overlap"]}/{d["k"]}',
                    f'{x["sequence_match_count"]}/{d["k"]}',x['first_divergent_round'],
                    x['strict_wrong_pruning_observed'],
                    f'{x["true_gain_loss_exhaustive_minus_lazy"]:.12g}'])
            ka=d.get('sampled_objective_lazy_key_audit')
            if ka is not None:
                key_rows.append([d['dataset'],'Same fixed-probe SLQ objective',
                    ka['audited_common_prefix_rounds'],ka['cached_keys_checked'],ka['undercoverage_count'],
                    f'{100*ka["undercoverage_rate"]:.6f}%',
                    f'{ka["maximum_positive_absolute_excess"]:.9g}'])
            lazy=next((r for r in d['routes'] if r['name']=='CRN-SLQ-CELF-p128'),None)
            if lazy is not None and 'audit' in lazy:
                a=lazy['audit']
                key_rows.append([d['dataset'],'True fractional-trace marginal',d['k'],
                    a['cache_points_checked'],a['cache_point_undercoverage_count'],
                    f'{100*a["cache_point_undercoverage_rate"]:.6f}%','not recorded'])
        text.append(table(['Dataset','Route','Runs','Seconds, mean ± SD','Incidence RHS','Probe RHS',
            'Krylov SpMV','Residual SpMV','Maximum relative marginal error','True G','Set match with reference'],rows))
        text.append('Raw-Proj evaluates the inverse-square quadratic form in the Lanczos projection via a differentiated LDL recurrence. Raw-FullVec explicitly forms every shifted n-vector through equivalent short direction recurrences and validates its true residual. CRN probes are fixed across every candidate trial and round; p32 is a prefix of p128.\n')
        text.append('### Actual lazy selections under identical probes\n')
        text.append('This comparison holds the sampled objective fixed. A strict wrong lazy decision requires divergence from exhaustive greedy under the same probes and current selected set. True gain loss is exhaustive minus lazy; it is separate from either method’s error relative to exact fractional greedy.\n')
        text.append(table(['Dataset','Set overlap, lazy vs exhaustive','Ordered matches',
            'First divergent round','Strict wrong pruning observed','True gain loss, exhaustive minus lazy'],lazy_rows))
        text.append('### Cached-point upper-bound diagnostics\n')
        text.append('The first diagnostic compares an actual CELF cache key with the **current marginal of the same fixed-probe SLQ objective**, at rounds where the lazy and exhaustive paths have the same prefix. A violation exceeds 1e-10 × max(1, abs(current marginal)). The second compares that cache key with the **independently evaluated true fractional-trace marginal**, using the recorded relative threshold of 1e-8. The targets and thresholds differ, so these rates are not interchangeable. Maximum marginal relative error in the route table above is a third quantity: point-estimation error against the true marginal. A cache-bound violation need not change a selected winner.\n')
        text.append(table(['Dataset','Comparison target','Rounds evaluated','Cached keys checked',
            'Keys below current marginal','Violation rate','Maximum positive absolute excess'],key_rows))
    add_nominal_reference_audit(OUT)
    render_science_markdown(OUT)
    p=OUT/'fastfrac_unified_science_sections.md'
    if p.exists():text.append(p.read_text())
    text.append('## Reproduction\n```bash\npython fastfrac_final_unified_experiments.py --threads 8 --repeats 5\npython unified_routes_postprocess.py\npython build_fastfrac_unified_report.py\n```\nThe JSON files retain selected edge indices, individual repetition times, exact graph and candidate data, per-RHS iterations, per-round counts, independent reference values, and equality certificates.\n')
    (ROOT/'fastfrac_final_unified_data.md').write_text('\n'.join(text))

if __name__=='__main__':main()
