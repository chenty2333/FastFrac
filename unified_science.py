"""Scientific controls for the unified FastFrac experiment.

No benchmark executes on import.  This module expects unified_backend.install()
to have selected the same compiled kernels used by the main performance suite.
Every reported observation is computed; no requested trend is imposed.
"""
from pathlib import Path
from time import perf_counter
from itertools import combinations, permutations
from dataclasses import asdict
import heapq
import hashlib
import json
import gzip
import math
import numpy as np
from scipy import sparse
from scipy.linalg import eigh, eigvalsh
from threadpoolctl import threadpool_limits
import fastfrac_cpu_benchmark as ff
from revision_structure_opt import (_original_sbm, _heterogeneous_sbm, _matrix,
    _structural_metrics, _small_instance, _stable_gain)
from revision_adaptive import _small_ba, _exact_resistances, _maxerr

ROOT = Path(__file__).resolve().parent
S_SCAN = (.05, .10, .25, .50, .70, .85, .95)
ALPHA_SCAN = (1., 10., 50., 200.)
S_DOWNSTREAM = (.10, .30, .50, .70, .85)


def _save(out, filename, value):
    Path(out).mkdir(parents=True, exist_ok=True)
    ff.write_json(Path(out) / filename, value)


def _summary(runs):
    ans = dict(repetitions=len(runs), selected=runs[0]['selected'],
        stable_sequence=all(r['selected'] == runs[0]['selected'] for r in runs), runs=runs)
    for key in ('seconds', 'init_seconds', 'online_seconds', 'calls', 'init_calls',
                'online_calls', 'krylov_spmv', 'residual_spmv', 'R'):
        vals = np.asarray([r[key] for r in runs if key in r], float)
        if len(vals):
            ans[key] = dict(mean=float(vals.mean()),
                std=float(vals.std(ddof=1)) if len(vals)>1 else 0.,
                min=float(vals.min()), max=float(vals.max()))
    return ans


def timed(call, repetitions=5):
    runs=[]
    for repeat in range(repetitions):
        row=call()
        row['online_seconds']=row['seconds']-row['init_seconds']
        row['mean_krylov_iterations_per_rhs']=row['krylov_spmv']/max(1,row['calls'])
        if 'rejected_or_refined_heap_tops' in row:
            row['R']=row['rejected_or_refined_heap_tops']
        row['repeat']=repeat+1
        runs.append(row)
    return _summary(runs)


def _upper(G, alpha, k):
    return float(1+2*alpha*(max(dict(G.degree(weight='weight')).values())+k))


def _lam_selected(A0,C,selection,alpha):
    B=A0.toarray()
    for i in selection:
        u,v=C[i]
        B[u,u]+=alpha; B[v,v]+=alpha; B[u,v]-=alpha; B[v,u]-=alpha
    return eigvalsh(B,check_finite=False,overwrite_a=True)


def _allocation(C,selected,labels):
    counts={f'{a}-{b}':0 for a in range(4) for b in range(a,4)}
    for i in selected:
        a,b=sorted(map(int,labels[C[i]]));counts[f'{a}-{b}']+=1
    cross=np.asarray([v for key,v in counts.items() if key[0]!=key[-1]],float)
    positive=cross[cross>0]
    p=positive/positive.sum() if len(positive) else positive
    return dict(block_pair_counts=counts,
        weakest_block0_cut_edges=int(sum(labels[C[i,0]]!=labels[C[i,1]] and
            0 in labels[C[i]] for i in selected)),
        core_blocks2_3_internal_edges=int(sum(np.all(labels[C[i]]>=2) for i in selected)),
        strongest_cut_fraction=float(cross.max()/len(selected)),
        cross_cut_entropy=float(-np.sum(p*np.log(p))) if len(p) else 0.)


def run_scan_downstream(output_dir,repeats=5):
    start=perf_counter()
    ans=dict(status='running',scan=[],downstream=[],modal_scan=[],methods=[],baselines=[],
        protocol={
          'timings':'five fresh complete runs; no shared or cached Phase-1 between repetitions',
          'scan':'locked heterogeneous graph; all 7 s x 4 alpha combinations, q=12',
          'structure':'unchanged base graph; all k selected edges, including any intrablock edges',
          'heat':'exact full dense eigenspectra; Delta H(tau)=tr(exp(-tau L0))-tr(exp(-tau LS))',
          'allocation':'block0 is the lowest-weight module; blocks2,3 are highest-weight cores; categories fixed before running',
          'no_search':'locked graphs reused; no graph, seed or parameter search to produce a favorable trend',
          'modal_fraction':'arithmetic mean over initial crossing candidates of sum_{i=1}^4 lambda_i^(s-1)(b^T v_i)^2 / sum_i lambda_i^(s-1)(b^T v_i)^2',
          'regularization':'A=I+alpha L: logdet derivative weights 1/(1+alpha lambda_L) are bounded at lambda_L=0',
          'evidence':'a spectral objective does not guarantee dominance at any prescribed heat horizon'})
    factories=[_original_sbm,_heterogeneous_sbm]
    for factory in factories:
        G,C,labels,meta=factory(); n=len(G);k=15
        meta=dict(meta)
        if factory is _original_sbm:meta['name']='Multiscale-SBM'
        L=ff.nx.laplacian_matrix(G,nodelist=range(n),weight='weight').astype(float).toarray()
        lamL,V=eigh(L,check_finite=False)
        lamL=np.maximum(lamL,0.)
        comp=(V[C[:,0]]-V[C[:,1]])**2
        cross=labels[C[:,0]]!=labels[C[:,1]]
        ans['methods'].append(dict(dataset=meta['name'],metadata=meta,
            base_edges=[(int(u),int(v),float(d.get('weight',1.))) for u,v,d in G.edges(data=True)],
            candidates=C.tolist(),communities=labels.tolist()))
        # Modal curves for both original and heterogeneous graph, all requested points.
        for alpha in ALPHA_SCAN:
            lam=1+alpha*lamL
            for s in S_SCAN:
                w=lam**(s-1); fractions=(comp[:,:4]@w[:4])/(comp@w)
                ans['modal_scan'].append(dict(dataset=meta['name'],alpha=alpha,s=s,
                    P_low_cross=float(np.mean(fractions[cross])),
                    P_low_intra=float(np.mean(fractions[~cross]))))
        if factory is _heterogeneous_sbm:
            for alpha in ALPHA_SCAN:
                A0=_matrix(G,alpha);upper=_upper(G,alpha,k);lam0=1+alpha*lamL
                ld=timed(lambda:ff.run_celf(A0,C,k,np.array([0.]),np.array([1.]),1e-7,alpha=alpha),repeats)
                ldlam=_lam_selected(A0,C,ld['selected'],alpha)
                ans['scan'].append(dict(dataset=meta['name'],alpha=alpha,s=None,
                    method='Dynamic-Logdet',timing=ld,structure=_structural_metrics(G,C,ld['selected'],labels),
                    allocation=_allocation(C,ld['selected'],labels),
                    gains_by_s={str(s):_stable_gain(lam0,ldlam,s) for s in S_SCAN}))
                for s in S_SCAN:
                    shifts,w,eta=ff.rule(s,12,upper)
                    run=timed(lambda:ff.run_celf(A0,C,k,shifts,w,1e-7,alpha=alpha),repeats)
                    lam=_lam_selected(A0,C,run['selected'],alpha)
                    tsh,tw,_=ff.rule(s,54,upper)
                    check=ff.run_celf(A0,C,k,tsh,tw,1e-10,alpha=alpha)
                    ours=_stable_gain(lam0,lam,s);other=_stable_gain(lam0,ldlam,s)
                    ans['scan'].append(dict(dataset=meta['name'],alpha=alpha,s=s,
                        method='FastFrac-CELF',q=12,eta=eta,timing=run,
                        true_gain=ours,dynamic_logdet_gain=other,improvement_logdet_pct=100*(ours/other-1),
                        overlap_logdet=len(set(run['selected'])&set(ld['selected'])),
                        q54_same_sequence=check['selected']==run['selected'],q54_selected=check['selected'],
                        structure=_structural_metrics(G,C,run['selected'],labels),
                        allocation=_allocation(C,run['selected'],labels)))
                _save(output_dir,'science_scan_downstream.json',ans)
                print(f'UNIFIED SCIENCE scan {meta["name"]} alpha={alpha:g}',flush=True)
        # Independent downstream utility at alpha=50, including s=0.3 and0.7.
        alpha=50.;A0=_matrix(G,alpha);upper=_upper(G,alpha,k);lam0=1+alpha*lamL
        ld=timed(lambda:ff.run_celf(A0,C,k,np.array([0.]),np.array([1.]),1e-7,alpha=alpha),repeats)
        selection_runs=[('Dynamic-Logdet',None,ld)]
        for s in S_DOWNSTREAM:
            sh,w,eta=ff.rule(s,12,upper)
            rr=timed(lambda:ff.run_celf(A0,C,k,sh,w,1e-7,alpha=alpha),repeats)
            selection_runs.append(('FastFrac',s,rr))
        for name,s,run in selection_runs:
            la=_lam_selected(A0,C,run['selected'],alpha);lams=np.maximum((la-1)/alpha,0.)
            heat={str(tau):float(np.sum(np.exp(-tau*lamL)*(-np.expm1(-tau*(lams-lamL)))))
                for tau in (.05,.2,1.)}
            ans['downstream'].append(dict(dataset=meta['name'],method=name,s=s,alpha=alpha,
                timing=run,heat_trace_reduction=heat,
                true_gains={str(t):_stable_gain(lam0,la,t) for t in S_DOWNSTREAM},
                overlap_logdet=len(set(run['selected'])&set(ld['selected'])),
                structure=_structural_metrics(G,C,run['selected'],labels),
                allocation=_allocation(C,run['selected'],labels)))
        # Classical controls are selected once per fresh repetition; evaluation
        # of their true objectives is independent and excluded from timings.
        def heuristic(kind):
            t0=perf_counter();st=ff.Stats()
            if kind=='Static-EffRes':
                rr,_,_=ff.scalar_ms_batch(A0,C,np.array([0.]),1e-7,st)
                selection=np.lexsort((np.arange(len(C)),-rr[:,0]))[:k].tolist()
                sets=[selection]
            elif kind in ('Degree','Degree-Unweighted'):
                degree_weight='weight' if kind=='Degree' else None
                deg=np.asarray([G.degree(u,weight=degree_weight)+G.degree(v,weight=degree_weight) for u,v in C])
                selection=np.lexsort((np.arange(len(C)),deg))[:k].tolist();sets=[selection]
            else:
                rng=np.random.default_rng(20261101+int(meta.get('seed',0)))
                sets=[rng.choice(len(C),k,replace=False).tolist() for _ in range(10)];selection=sets[0]
            elapsed=perf_counter()-t0
            return dict(selected=selection,selected_sets=sets,seconds=elapsed,
                init_seconds=elapsed,init_calls=st.calls,online_calls=0,**asdict(st))
        for name in ('Static-EffRes','Degree','Degree-Unweighted','Random'):
            run=timed(lambda:heuristic(name),repeats)
            sets=run['runs'][0]['selected_sets'];gains={str(s):[] for s in S_SCAN};heat={str(t):[] for t in (.05,.2,1.)}
            metrics=[]
            for selected in sets:
                la=_lam_selected(A0,C,selected,alpha);ls=np.maximum((la-1)/alpha,0.)
                for s in S_SCAN:gains[str(s)].append(_stable_gain(lam0,la,s))
                for t in (.05,.2,1.):heat[str(t)].append(float(np.sum(np.exp(-t*lamL)*(-np.expm1(-t*(ls-lamL))))))
                metrics.append(_structural_metrics(G,C,selected,labels))
            ans['baselines'].append(dict(dataset=meta['name'],method=name,alpha=alpha,timing=run,
                selected_sets=sets,gains_by_s={s:float(np.mean(v)) for s,v in gains.items()},
                gain_trials=gains,heat_trace_reduction={t:float(np.mean(v)) for t,v in heat.items()},
                structure={key:float(np.mean([m[key] for m in metrics])) for key in metrics[0] if key!='per_edge'}))
        _save(output_dir,'science_scan_downstream.json',ans)
        print(f'UNIFIED SCIENCE downstream {meta["name"]}',flush=True)
    ans['seconds']=perf_counter()-start;ans['status']='complete'
    _save(output_dir,'science_scan_downstream.json',ans)
    return ans


def _exact_greedy(A0,C,k,s,alpha):
    A=A0.toarray();chosen=[];alive=np.ones(len(C),bool);grid=[];start=perf_counter()
    for r in range(k):
        lam,V=eigh(A,check_finite=False)
        ids=np.flatnonzero(alive);true=[]
        for i in ids:
            B=A.copy();u,v=C[i]
            B[u,u]+=alpha;B[v,v]+=alpha;B[u,v]-=alpha;B[v,u]-=alpha
            trial=eigvalsh(B,check_finite=False,overwrite_a=True)
            true.append(_stable_gain(lam,trial,s))
        vals=np.asarray(true);winner=int(ids[np.argmax(vals)])
        grid.append(dict(round=r+1,ids=ids,truth=vals,lam=lam,
            comp=(V[C[ids,0]]-V[C[ids,1]])**2))
        chosen.append(winner);alive[winner]=False;u,v=C[winner]
        A[u,u]+=alpha;A[v,v]+=alpha;A[u,v]-=alpha;A[v,u]-=alpha
    return chosen,grid,perf_counter()-start


def run_quadrature(output_dir,repeats=5):
    start=perf_counter()
    ans=dict(status='running',
        protocol='For each dataset/s, recompute full Exact-Eig greedy; evaluate all remaining candidates at each of its k states. Quadrature-only error uses independent dense eigenspectra for r. Timings are5 fresh compiled CELF runs.',
        rows=[],reference=[])
    G,C,labels,meta=_original_sbm();meta=dict(meta);meta['name']='Multiscale-SBM'
    smallA,smallC,smalllocked=_small_ba()
    problems=[('Small-BA',smallA,smallC,1.,smalllocked['metadata']['upper'],10,(.5,)),
        (meta['name'],_matrix(G,50.),C,50.,_upper(G,50.,15),15,(.1,.5))]
    for name,A0,C,alpha,upper,k,svals in problems:
        for s in svals:
            selected,grid,seconds=_exact_greedy(A0,C,k,s,alpha)
            ans['reference'].append(dict(dataset=name,s=s,selected=selected,seconds=seconds,
                eigensolves=sum(len(g['ids'])+1 for g in grid)))
            for q in (4,8,12,16,24,36,54):
                sh,w,eta=ff.rule(s,q,upper);errors=[];initial=None
                for r,g in enumerate(grid):
                    R=g['comp']@(1/(g['lam'][:,None]+sh[None,:]))
                    pred=ff.scores(R,w,alpha);err=np.abs(pred-g['truth'])/g['truth']
                    if r==0:initial=err.copy()
                    errors.extend(err)
                run=timed(lambda:ff.run_celf(A0,C,k,sh,w,1e-7,alpha=alpha),repeats)
                ans['rows'].append(dict(dataset=name,alpha=alpha,upper=upper,k=k,s=s,q=q,eta=eta,
                    max_relative_error=float(np.max(errors)),mean_relative_error=float(np.mean(errors)),
                    initial_max_relative_error=float(initial.max()),initial_mean_relative_error=float(initial.mean()),
                    marginal_samples=len(errors),timing=run,
                    sequence_match_percent=100*np.mean(np.asarray(run['selected'])==np.asarray(selected)),
                    set_match_percent=100*len(set(run['selected'])&set(selected))/k,
                    selected=run['selected'],exact_selected=selected))
                _save(output_dir,'science_quadrature.json',ans)
            print(f'UNIFIED SCIENCE q audit {name} s={s}, exact seconds={seconds:.3f}',flush=True)
    ans['seconds']=perf_counter()-start;ans['status']='complete'
    _save(output_dir,'science_quadrature.json',ans)
    return ans


def run_opt_controls(output_dir,repeats=5):
    start=perf_counter();path=ROOT/'fastfrac_revision_results/global_optimum.json'
    old=json.loads(path.read_text());assert old['status']=='complete' and len(old['instances'])==20
    ans=dict(status='running',instances=[],reference=1-math.exp(-1),
        optimum_provenance=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            independently_previously_enumerated_subsets=27300,
            reuse='Exact OPT numbers are reused after validating graph and candidate identities; new method selections and gains are recomputed.'),
        random_policy='10 independent fixed-seed uniform k-subsets per instance; reported per-instance mean, with individual ratios retained')
    for z in range(20):
        G,A,C,seed=_small_instance(z);locked=old['instances'][z]
        assert seed==locked['seed'] and C.tolist()==locked['candidates']
        assert set(map(tuple,locked['base_edges']))==set(map(tuple,sorted([list(e) for e in G.edges()])))
        lam0=eigvalsh(A.toarray(),check_finite=False);upper=_upper(G,1.,4);sh,w,_=ff.rule(.5,12,upper)
        runs={
            'FastFrac-CELF':timed(lambda:ff.run_celf(A,C,4,sh,w,1e-7),repeats),
            'FastFrac-Inc':timed(lambda:ff.run_inc(A,C,4,sh,w,1e-7),repeats),
            'Dynamic-Logdet':timed(lambda:ff.run_celf(A,C,4,np.array([0.]),np.array([1.]),1e-7),repeats)}
        r,_,_=ff.scalar_ms_batch(A,C,np.array([0.]),1e-10,ff.Stats())
        static=np.lexsort((np.arange(len(C)),-r[:,0]))[:4].tolist()
        deg=np.array([G.degree(u)+G.degree(v) for u,v in C])
        degree=np.lexsort((np.arange(len(C)),deg))[:4].tolist()
        selections={name:rr['selected'] for name,rr in runs.items()}
        selections.update({'Static-EffRes':static,'Degree':degree})
        row=dict(instance=z+1,seed=seed,optimum=locked['optimum'],methods={})
        for name,selection in selections.items():
            lam=_lam_selected(A,C,selection,1.);gain=_stable_gain(lam0,lam,.5)
            row['methods'][name]=dict(selected=selection,gain=gain,ratio=gain/locked['optimum'])
            if name in runs:row['methods'][name]['timing']=runs[name]
        rng=np.random.default_rng(seed+901);ratios=[];rand=[]
        for _ in range(10):
            sel=rng.choice(15,4,replace=False).tolist();gain=_stable_gain(lam0,_lam_selected(A,C,sel,1.),.5)
            ratios.append(gain/locked['optimum']);rand.append(sel)
        row['methods']['Random']=dict(ratio=float(np.mean(ratios)),trial_ratios=ratios,selections=rand)
        ans['instances'].append(row);_save(output_dir,'science_opt_controls.json',ans)
    ans['summary']={}
    for name in ans['instances'][0]['methods']:
        arr=np.array([r['methods'][name]['ratio'] for r in ans['instances']])
        ans['summary'][name]=dict(min=float(arr.min()),mean=float(arr.mean()),max=float(arr.max()),std=float(arr.std(ddof=1)))
    ans['seconds']=perf_counter()-start;ans['status']='complete'
    _save(output_dir,'science_opt_controls.json',ans)
    print('UNIFIED SCIENCE OPT controls complete',flush=True)
    return ans


def inc_audit_run(A0,C,k,shifts,weights,tol,adaptive=True,alpha=1.):
    """Same interval recurrence for adaptive and uncertified ranking; record events."""
    start=perf_counter();A=A0.copy();st=ff.Stats();alive=np.ones(len(C),bool)
    R,lo,hi=ff.scalar_ms_batch(A,C,shifts,tol,st);init_s=perf_counter()-start;init_c=st.calls
    chosen=[];gains=[];events=[];rounds=[]
    for rnd in range(k):
        call0=st.calls;localtol=tol
        if adaptive:
            upper=ff.scores(hi,weights,alpha)
            heap=[(-float(upper[i]),int(i)) for i in np.flatnonzero(alive)];heapq.heapify(heap)
            while True:
                _,i=heapq.heappop(heap)
                d,dl,du,X,rr=ff.vector_ms(A,C[i],shifts,localtol,st)
                R[i],lo[i],hi[i]=d,dl,du
                actual=float(ff.scores(d,weights,alpha));lower=float(ff.scores(dl,weights,alpha));upper_i=float(ff.scores(du,weights,alpha))
                competitor=-heap[0][0] if heap else -np.inf
                if lower>competitor:break
                if localtol<=1e-12 and actual>=competitor-2e-11*max(actual,1.):break
                events.append(dict(round=rnd+1,prefix=chosen.copy(),candidate=i,
                    competitor=int(heap[0][1]),tol=localtol,point=actual,
                    lower=lower,upper=upper_i,competitor_cached_upper=competitor))
                heapq.heappush(heap,(-upper_i,i))
                localtol=max(localtol*.01,1e-12) if upper_i>=competitor else tol
        else:
            values=ff.scores(R,weights,alpha);values[~alive]=-np.inf;i=int(np.argmax(values))
            d,dl,du,X,rr=ff.vector_ms(A,C[i],shifts,tol,st)
            R[i],lo[i],hi[i]=d,dl,du
            actual=float(ff.scores(d,weights,alpha))
        chosen.append(i);gains.append(actual)
        cross=X[C[:,0]]-X[C[:,1]]
        R-=alpha*cross**2/(1+alpha*d)
        delta=np.sqrt(2.)*rr/(1+shifts)
        correction_lo=alpha*np.maximum(np.abs(cross)-delta,0.)**2/(1+alpha*du)
        correction_hi=alpha*(np.abs(cross)+delta)**2/(1+alpha*dl)
        lo=np.maximum(0,lo-correction_hi);hi=np.maximum(lo,hi-correction_lo)
        alive[i]=False;A=ff.add_edge(A,C[i],alpha)
        rounds.append(dict(round=rnd+1,selected=i,online_rhs=st.calls-call0,
            max_actual_vector_relative_residual=float(rr.max()/np.sqrt(2.))))
    elapsed=perf_counter()-start
    return dict(selected=chosen,seconds=elapsed,init_seconds=init_s,online_seconds=elapsed-init_s,
        init_calls=init_c,online_calls=st.calls-init_c,**asdict(st),
        R=st.calls-init_c-k,adaptive=adaptive,tol=tol,marginal_sum=sum(gains),
        events=events,rounds=rounds,_final_R=R,_alive=alive,_final_A=A)


def _short_automorphism(A,edge,other):
    """Exact finite certificate: permute the <=4 endpoints and fix all others."""
    source=set(map(int,edge));target=set(map(int,other));ids=sorted(source|target)
    if source==target:return dict(mapping=[],kind='same edge')
    neighborhoods={x:dict(zip(A.indices[A.indptr[x]:A.indptr[x+1]],A.data[A.indptr[x]:A.indptr[x+1]])) for x in ids}
    for image in permutations(ids):
        p=dict(zip(ids,image))
        if {p[x] for x in source}!=target:continue
        good=True
        for x in ids:
            mapped={p.get(int(y),int(y)):float(w) for y,w in neighborhoods[x].items()}
            if mapped!=neighborhoods[p[x]]:good=False;break
        if good:return dict(mapping=[[x,p[x]] for x in ids if x!=p[x]],
            kind='exact current weighted-adjacency automorphism fixing all other vertices')
    return None


def _small_component_automorphism(A,edge,other):
    """Optional exact certificate swapping isomorphic small connected components.

    This bounded search supplements endpoint transpositions without attempting
    an unbounded general graph-isomorphism search on the giant component.
    """
    _,component=sparse.csgraph.connected_components(A,directed=False)
    members={int(c):np.flatnonzero(component==c) for c in set(component[list(edge)+list(other)])}
    for target in (tuple(map(int,other)),tuple(map(int,other[::-1]))):
        mapping={};failed=False
        for x,y in zip(map(int,edge),target):
            if mapping.get(x,x)==y:continue
            if x in mapping or y in mapping:failed=True;break
            cx,cy=int(component[x]),int(component[y])
            if cx==cy:
                # A same-component endpoint transposition can coexist with a
                # disconnected-component swap; the complete permutation is
                # verified against the entire affected adjacency below.
                mapping[x]=y;mapping[y]=x
                continue
            source=members[cx];dest=members[cy]
            if len(source)!=len(dest) or len(source)>40:failed=True;break
            G=ff.nx.from_scipy_sparse_array(A[source][:,source])
            H=ff.nx.from_scipy_sparse_array(A[dest][:,dest])
            nxroot=int(np.flatnonzero(source==x)[0]);nyroot=int(np.flatnonzero(dest==y)[0])
            ff.nx.set_node_attributes(G,{i:i==nxroot for i in G},'root')
            ff.nx.set_node_attributes(H,{i:i==nyroot for i in H},'root')
            matcher=ff.nx.algorithms.isomorphism.GraphMatcher(G,H,
                node_match=ff.nx.algorithms.isomorphism.categorical_node_match('root',False),
                edge_match=ff.nx.algorithms.isomorphism.numerical_edge_match('weight',0.,rtol=0.,atol=0.))
            try:local=next(matcher.isomorphisms_iter())
            except StopIteration:failed=True;break
            for a,b in local.items():
                u,v=int(source[a]),int(dest[b]);mapping[u]=v;mapping[v]=u
        if failed or {mapping.get(int(x),int(x)) for x in edge}!=set(map(int,other)):continue
        good=True
        for x in mapping:
            ids=A.indices[A.indptr[x]:A.indptr[x+1]];ws=A.data[A.indptr[x]:A.indptr[x+1]]
            mapped={mapping.get(int(y),int(y)):float(w) for y,w in zip(ids,ws)}
            xx=mapping[x]
            true=dict(zip(A.indices[A.indptr[xx]:A.indptr[xx+1]],A.data[A.indptr[xx]:A.indptr[xx+1]]))
            if mapped!=true:good=False;break
        if good:return dict(mapping=[[x,mapping[x]] for x in sorted(mapping)],
            kind='exact current weighted-adjacency automorphism swapping rooted isomorphic components of at most40 vertices')
    return None


def _classify_events(A0,C,events,shifts,weights,alpha,upper):
    classified=[];states={};audit_st=ff.Stats();start=perf_counter()
    # High-order independent true-objective marginal comparison, beyond q12.
    tsh,tw,eta=ff.rule(.5,96,upper)
    for event in events:
        prefix=tuple(event['prefix'])
        if prefix not in states:
            A=A0.copy()
            for i in prefix:A=ff.add_edge(A,C[i],alpha)
            states[prefix]=A
        A=states[prefix];ids=[event['candidate'],event['competitor']]
        r,_,_=ff.scalar_ms_batch(A,C[ids],tsh,1e-12,audit_st)
        val=ff.scores(r,tw,alpha);gap=float(abs(val[0]-val[1]))
        cert=_short_automorphism(A,C[ids[0]],C[ids[1]])
        if cert is None and gap<1e-12:
            cert=_small_component_automorphism(A,C[ids[0]],C[ids[1]])
        if cert is not None:kind='certified structural tie'
        elif gap<1e-14:kind='numerical tie; structural equivalence unproved'
        else:kind='unequal true margins / residual or cached interval overlap'
        classified.append({**event,'independent_q96_marginals':val.tolist(),
            'absolute_true_marginal_gap':gap,'classification':kind,'automorphism_certificate':cert})
    return classified,dict(seconds=perf_counter()-start,stats=asdict(audit_st),q=96,
        eta=eta,R_tie=sum(r['classification']=='certified structural tie' for r in classified),
        R_numtie=sum(r['classification'].startswith('numerical tie') for r in classified),
        R_res=sum(r['classification'].startswith('unequal true') for r in classified))


def _load_grqc():
    raw=np.loadtxt(gzip.open(ROOT/'fastfrac_realworld_data/ca-GrQc.txt.gz'),dtype=np.int64)
    ids,inv=np.unique(raw,return_inverse=True);ed=np.sort(inv.reshape(-1,2),axis=1)
    ed=np.unique(ed[ed[:,0]!=ed[:,1]],axis=0);n=len(ids)
    G=ff.nx.Graph();G.add_nodes_from(range(n));G.add_edges_from(ed.tolist())
    existing=set(map(tuple,ed));rng=np.random.default_rng(20267032);candidate=set()
    while len(candidate)<2000:
        u,v=sorted(map(int,rng.integers(n,size=2)))
        if u!=v and (u,v) not in existing:candidate.add((u,v))
    C=np.asarray(sorted(candidate),np.int64);C=np.ascontiguousarray(C[rng.permutation(len(C))])
    return _matrix(G,1.),C,dict(name='ca-GrQc',n=n,m=len(ed),k=20,upper=_upper(G,1.,20))


def run_adaptive_controls(output_dir,repeats=5):
    start=perf_counter();ans=dict(status='running',rows=[],protocol={
        'R':'online vector solves minus k; every rejected/refined popped candidate is logged',
        'uncertified':'same compiled initializer, coarse selected-vector tolerance and Sherman-Morrison recurrence; choose cached point maximum, no rechecks',
        'structural_tie':'verified current weighted-adjacency automorphism mapping candidate to competitor: endpoint permutation or exchange of rooted isomorphic components of at most40 vertices',
        'numerical_tie':'independent q96/tol1e-12 difference <1e-14 without an automorphism certificate is reported separately, never called structural',
        'reference':'high-accuracy q12/tol1e-12 compiled CELF; Small-BA additionally uses dense spectral greedy; tie-safe declared numerical tolerance',
        'timing':'classification, exact gain and cache audits excluded from all algorithm timings'})
    A0,C,locked=_small_ba();smallmeta={**locked['metadata'],'name':'Small-BA'}
    for A0,C,meta,tols in [(A0,C,smallmeta,(.1,.01,1e-4,1e-7)),(*_load_grqc(),(1e-7,))]:
        k=meta['k'];sh,w,eta=ff.rule(.5,12,meta['upper'])
        reference=ff.run_celf(A0,C,k,sh,w,1e-12)
        if meta['name']=='Small-BA':
            AA=A0.copy();truth=[];alive=np.ones(len(C),bool)
            for _ in range(k):
                er,_=_exact_resistances(AA,C,sh);val=ff.scores(er,w);val[~alive]=-np.inf;i=int(np.argmax(val))
                truth.append(i);alive[i]=False;AA=ff.add_edge(AA,C[i])
            dense_reference=truth
            lam0=eigvalsh(A0.toarray(),check_finite=False)
            reference_gain=_stable_gain(lam0,_lam_selected(A0,C,truth,1.),.5)
        else:
            truth=reference['selected'];dense_reference=None
            reference_gain=ff.reference_gain(A0,C,truth,meta['upper'],s_values=(.5,),eta_target=1e-12)['by_s']['0.5']['gain']
        for tol in tols:
            for adaptive in (True,False):
                rows=[];first=None
                for repeat in range(repeats):
                    run=inc_audit_run(A0,C,k,sh,w,tol,adaptive)
                    extras={key:run.pop(key) for key in ('_final_R','_alive','_final_A')}
                    if first is None:first=(run.copy(),extras)
                    rows.append(run)
                run,extras=first;summary=_summary(rows)
                if meta['name']=='Small-BA':
                    true_R,lam=_exact_resistances(extras['_final_A'],C,sh)
                    cache_error=_maxerr(extras['_final_R'],true_R,extras['_alive'])
                    gain=_stable_gain(lam0,lam,.5)
                else:
                    true_R,_,_=ff.scalar_ms_batch(extras['_final_A'],C,sh,1e-12,ff.Stats())
                    cache_error=_maxerr(extras['_final_R'],true_R,extras['_alive'])
                    gain=ff.reference_gain(A0,C,run['selected'],meta['upper'],s_values=(.5,),eta_target=1e-12)['by_s']['0.5']['gain']
                events,classinfo=_classify_events(A0,C,run['events'],sh,w,1.,meta['upper'])
                ans['rows'].append(dict(dataset=meta['name'],tol=tol,adaptive=adaptive,timing=summary,
                    selected=run['selected'],reference_selected=truth,
                    dense_reference_selected=dense_reference,
                    positional_match_percent=100*np.mean(np.asarray(run['selected'])==np.asarray(truth)),
                    set_match_percent=100*len(set(run['selected'])&set(truth))/k,
                    true_gain=gain,reference_gain=reference_gain,
                    gain_loss=reference_gain-gain,gain_loss_percent=100*(1-gain/reference_gain),
                    final_cache_max_relative_error=cache_error,classified_events=events,classification=classinfo))
                _save(output_dir,'science_adaptive_controls.json',ans)
                print(f'UNIFIED SCIENCE adaptive {meta["name"]} tol={tol:g} enabled={adaptive} R={run["R"]}',flush=True)
    ans['seconds']=perf_counter()-start;ans['status']='complete'
    _save(output_dir,'science_adaptive_controls.json',ans)
    return ans


def run_science(output_dir,repeats=5,groups=('scan','quadrature','opt','adaptive')):
    start=perf_counter();results={}
    calls=dict(scan=run_scan_downstream,quadrature=run_quadrature,opt=run_opt_controls,adaptive=run_adaptive_controls)
    with threadpool_limits(limits=1):
        for name in groups:
            t=perf_counter();results[name]=calls[name](output_dir,repeats)
            print(f'UNIFIED SCIENCE group {name} {perf_counter()-t:.3f}s',flush=True)
    return dict(seconds=perf_counter()-start,groups=results)


def add_nominal_reference_audit(output_dir):
    """Read-only scientific postprocessing of completed, already timed runs.

    Preserve all tighter-reference literal matches.  Add a separate nominal
    reference and, where an existing event certificate suffices, exact final
    permutation similarity of the two augmented matrices.
    """
    out=Path(output_dir);path=out/'science_adaptive_controls.json'
    if not path.exists():return None
    data=json.loads(path.read_text())
    matrices={}
    for row in data['rows']:
        dense=row.get('dense_reference_selected') is not None
        row['tighter_solve_reference']=dict(tolerance=None if dense else 1e-12,
            selected=row['reference_selected'],
            positional_match_percent=row['positional_match_percent'],
            set_match_percent=row['set_match_percent'],
            description='Dense eigenspectral q12 surrogate greedy at every state.' if dense else
                'High-accuracy floating-point CELF, not an exact symbolic tie-break reference.')
        pp=out/'performance'/f'{row["dataset"]}.json'
        if pp.exists():
            performance=json.loads(pp.read_text());runs=performance.get('runs',{}).get('CELF',[])
            if runs:
                selected=runs[0]['selected'];k=len(selected)
                row['nominal_reference']=dict(tolerance=1e-7,source=str(pp),selected=selected,
                    positional_match_percent=100*np.mean(np.asarray(row['selected'])==np.asarray(selected)),
                    set_match_percent=100*len(set(row['selected'])&set(selected))/k,
                    reference_run='first freshly timed common-backend CELF repetition')
        if row['dataset']!='ca-GrQc':continue
        if 'ca-GrQc' not in matrices:matrices['ca-GrQc']=_load_grqc()[:2]
        A0,C=matrices['ca-GrQc'];A=A0.copy();B=A0.copy()
        for i in row['selected']:A=ff.add_edge(A,C[i])
        for i in row['reference_selected']:B=ff.add_edge(B,C[i])
        # A disabled run has no own recheck log; use recorded certificates
        # from the corresponding adaptive run only after verifying equality.
        donor=next(r for r in data['rows'] if r['dataset']==row['dataset'] and r['adaptive'])
        certificates=[e['automorphism_certificate'] for e in donor['classified_events'] if e['automorphism_certificate']]
        for cert in certificates:
            perm=np.arange(A.shape[0])
            for a,b in cert['mapping']:perm[a]=b
            inv=np.argsort(perm);diff=(A[inv][:,inv]-B).tocsr();diff.eliminate_zeros()
            if diff.nnz==0:
                row['final_spectral_equivalence_certificate']=dict(
                    permutation=cert['mapping'],matrix_difference_nnz=0,
                    statement='The two final augmented A matrices are exactly permutation-similar (all represented entries are integers). Every spectral trace, including fractional powers and heat traces, is identical.',
                    tied_representative_equivalence_percent=100.)
                break
    ff.write_json(path,data)
    return data


def render_science_markdown(output_dir):
    """Render complete observed scientific tables; no numbers are hard-coded."""
    out=Path(output_dir);sections=[]
    def table(headers,rows):
        return '\n'.join(['| '+' | '.join(headers)+' |',
            '| '+' | '.join(['---']*len(headers))+' |']+
            ['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'
    def ms(summary,key='seconds'):
        d=summary[key];return f'{d["mean"]:.6f} ± {d["std"]:.6f}'
    def number(v,d=6):return f'{v:.{d}f}'
    scan_path=out/'science_scan_downstream.json'
    if scan_path.exists():
        x=json.loads(scan_path.read_text())
        sections.append('## External utility and complete spectral-scale scan\n')
        sections.append('The original SBM and the prespecified heterogeneous four-block graph are reused without seed or parameter search. Added edges have unit weight. All degree and neighborhood diagnostics refer to the unchanged base graph and include every selected edge. Block 0 is the heterogeneous graph\'s lowest-weight module; blocks 2 and 3 are its highest-weight modules. These fixed labels describe the allocation rather than an optimized post-hoc partition.\n')
        sections.append('For the regularized matrix A = I + αL, the log-determinant derivative weight is 1/(1+αλ), which is bounded at λ = 0. Its low-frequency preference therefore does not imply a singular divergence or guarantee that it overallocates a particular cut. Heat-trace reductions below are independently evaluated from complete dense Laplacian eigenspectra at the declared horizons.\n')
        for name in ('Multiscale-SBM','Heterogeneous-Multiscale'):
            records=[r for r in x['downstream'] if r['dataset']==name]
            if not records:continue
            baseline=next(r for r in records if r['method']=='Dynamic-Logdet')
            sections.append(f'### {name}: independent heat-trace and allocation outcomes, α = 50, k = 15\n')
            rows=[]
            for r in records:
                name2=r['method'] if r['s'] is None else f'FastFrac(s={r["s"]:.2f})'
                h=r['heat_trace_reduction'];b=baseline['heat_trace_reduction'];a=r['allocation'];st=r['structure'];tm=r['timing']
                rows.append([name2,*[number(h[str(t)]) for t in (.05,.2,1.)],
                    *[f'{100*(h[str(t)]/b[str(t)]-1):+.3f}%' for t in (.05,.2,1.)],
                    a['weakest_block0_cut_edges'],a['core_blocks2_3_internal_edges'],
                    number(st['mean_shared_two_hop_neighbors'],3),number(st['mean_one_hop_ego_union_induced_edges'],3),
                    r['overlap_logdet'],ms(tm),int(tm['calls']['mean']),int(tm['krylov_spmv']['mean'])])
            sections.append(table(['Method','ΔH(.05)','ΔH(.2)','ΔH(1)','vs LD (.05)','vs LD (.2)','vs LD (1)',
                'Block0-cut edges','Core2/3 edges','N2','E_loc','Overlap LD /15','Total s (5-run mean ± SD)','RHS','Krylov SpMV'],rows))
            sections.append('The block-0/core categories for the uniform SBM use the same fixed labels but do not represent a weight-defined core-periphery hierarchy. Counts are reported for topology comparability.\n' if name=='Multiscale-SBM' else '')
        sections.append('### Complete heterogeneous scan (all 7 s values × 4 α values)\n')
        sections.append('q = 12 and tol = 10⁻⁷ for the timed method. The q = 54 validation selection is recomputed separately at tol = 10⁻¹⁰; a different sequence is explicitly shown rather than hidden. True gains use complete dense eigenspectra.\n')
        rows=[]
        for r in x['scan']:
            if r['s'] is None:continue
            st=r['structure'];rows.append([f'{r["alpha"]:g}',f'{r["s"]:.2f}',
                number(st['mean_endpoint_degree_sum'],3),number(st['mean_endpoint_weighted_degree_sum'],3),
                number(st['mean_shared_two_hop_neighbors'],3),number(st['mean_one_hop_ego_union_induced_edges'],3),
                st['selected_cross_block_edges'],number(r['true_gain']),number(r['dynamic_logdet_gain']),
                f'{r["improvement_logdet_pct"]:+.4f}%',r['overlap_logdet'],
                'yes' if r['q54_same_sequence'] else 'NO',ms(r['timing'])])
        sections.append(table(['α','s','dΣ unweighted','dΣ weighted','N2','E_loc','Cross /15','FastFrac G','Dynamic-LD G',
            'vs LD','Overlap /15','q12=q54 sequence','Total s mean ± SD'],rows))
        rows=[]
        for r in x['scan']:
            if r['s'] is not None:continue
            st=r['structure'];tm=r['timing'];rows.append([f'{r["alpha"]:g}',
                ms(tm,'init_seconds'),ms(tm,'online_seconds'),ms(tm),
                int(tm['calls']['mean']),int(tm['online_calls']['mean']),int(tm['krylov_spmv']['mean']),
                number(st['mean_endpoint_degree_sum'],3),number(st['mean_endpoint_weighted_degree_sum'],3),
                number(st['mean_shared_two_hop_neighbors'],3),number(st['mean_one_hop_ego_union_induced_edges'],3)])
        sections.append('### Dynamic-Logdet runtime and structure on each heterogeneous α\n')
        sections.append(table(['α','Init s mean ± SD','Online s mean ± SD','Total s mean ± SD','Total RHS','Online RHS',
            'Krylov SpMV','dΣ unweighted','dΣ weighted','N2','E_loc'],rows))
        sections.append('### Complete low-four-mode derivative share scan\n')
        rows=[]
        for alpha in ALPHA_SCAN:
            for s in S_SCAN:
                rr=[r for r in x['modal_scan'] if r['alpha']==alpha and r['s']==s]
                d={r['dataset']:r for r in rr}
                rows.append([f'{alpha:g}',f'{s:.2f}',
                    *[f'{100*d[n]["P_low_cross"]:.6f}%' for n in ('Multiscale-SBM','Heterogeneous-Multiscale')]])
        sections.append(table(['α','s','SBM P_low crossing','Heterogeneous P_low crossing'],rows))
        sections.append('P_low is the mean across initial crossing candidates of the fraction of the directional derivative contributed by the four smallest eigenmodes of A. It is not a four-mode approximation to the finite-edge marginal. The exact per-edge definitions and selected indices are retained in science_scan_downstream.json.\n')
        sections.append('### Static, degree and random controls on both multiscale graphs\n')
        rows=[]
        for r in x.get('baselines',[]):
            st=r['structure'];tm=r['timing'];rows.append([r['dataset'],r['method'],
                *[f'{r["gains_by_s"][str(s)]:.9f}' for s in (.1,.5,.85)],
                number(st['mean_endpoint_degree_sum'],3),number(st['mean_endpoint_weighted_degree_sum'],3),
                number(st['mean_shared_two_hop_neighbors'],3),number(st['mean_one_hop_ego_union_induced_edges'],3),
                ms(tm),int(tm['calls']['mean'])])
        sections.append(table(['Dataset','Method','G(.10)','G(.50)','G(.85)','dΣ unweighted','dΣ weighted',
            'N2','E_loc','Selection s mean ± SD','RHS'],rows))
        sections.append('Degree preserves the locked multiscale benchmark definition: ascending weighted endpoint-degree sum. Degree-Unweighted is separately reported to expose the effect of weights. Random averages10 independently seeded uniform subsets within each repetition; selected-set generation time excludes the independent dense evaluation, consistently with all other selection timings. Full gains for all7s and heat-horizon values are retained in JSON.\n')
    q_path=out/'science_quadrature.json'
    if q_path.exists():
        x=json.loads(q_path.read_text());sections.append('## Quadrature audit against full Exact-Eig greedy at α = 1 and50\n')
        sections.append('For each dataset/s, a fresh full dense-eigenvalue greedy is run for all k rounds. Quadrature errors are audited over every remaining candidate at every state on that exact sequence. The initial-candidate maximum is given separately so that an initial-only error is not confused with a path-wide maximum. The Gaussian surrogate is evaluated from dense spectral resolvents for this quadrature-only audit; timed greedy runs use the common compiled backend.\n')
        rows=[]
        for r in x['rows']:
            rows.append([r['dataset'],f'{r["alpha"]:g}',f'{r["s"]:.2f}',r['q'],f'{r["eta"]:.4e}',f'{r["max_relative_error"]:.4e}',
                f'{r["mean_relative_error"]:.4e}',f'{r["initial_max_relative_error"]:.4e}',
                f'{r["sequence_match_percent"]:.1f}%',f'{r["set_match_percent"]:.1f}%',ms(r['timing']),r['marginal_samples']])
        sections.append(table(['Dataset','α','s','q','Analytic ηq','Max rel. error (path)','Mean rel. error (path)','Max rel. error (initial)',
            'Ordered sequence match','Set match','Total s mean ± SD','Audited marginals'],rows))
        sections.append(table(['Dataset','s','Exact-Eig greedy seconds','Dense eigensolves'],
            [[r['dataset'],f'{r["s"]:.2f}',f'{r["seconds"]:.6f}',r['eigensolves']] for r in x['reference']]))
    opt_path=out/'science_opt_controls.json'
    if opt_path.exists():
        x=json.loads(opt_path.read_text());sections.append('## Twenty-instance global-optimum audit with all requested baselines\n')
        sections.append('The20 locked n = 25, M = 15, k = 4 instances and their previously exhaustively verified OPT values are reused after graph/candidate identity checks. Those OPT values cover27,300 subsets; all method selections and exact eigenvalue gains below are freshly recomputed. Random denotes the mean of10 fixed-seed subsets in each instance.\n')
        sections.append(table(['Method','Min G/OPT','Mean G/OPT','Std across20 instances','Max G/OPT'],
            [[name,f'{r["min"]:.9f}',f'{r["mean"]:.9f}',f'{r["std"]:.9f}',f'{r["max"]:.9f}'] for name,r in x['summary'].items()]))
        sections.append(f'The worst-case submodular reference is 1 − 1/e = {x["reference"]:.9f}. Finite-instance ratios do not establish objective curvature or replace the approximation proof.\n')
    ad_path=out/'science_adaptive_controls.json'
    if ad_path.exists():
        x=json.loads(ad_path.read_text());sections.append('## Adaptive residual verification versus disabled verification\n')
        sections.append('Adaptive and Uncertified use identical compiled initialization and vector kernels at each stated coarse tolerance. Uncertified always chooses the cached point maximum and never retries. Audit eigenvalues, independent high-order marginal evaluations and graph-automorphism checks occur outside timed algorithm runs. A candidate pair is called a structural tie only when an explicit current weighted-adjacency automorphism maps one edge to the other. A gap below10⁻¹⁴ without that certificate is listed separately as a numerical tie.\n')
        rows=[]
        for r in x['rows']:
            c=r['classification'];t=r['timing'];rows.append([r['dataset'],f'{r["tol"]:.0e}',
                'Adaptive' if r['adaptive'] else 'Uncertified',int(t['R']['mean']),c['R_tie'],c['R_numtie'],c['R_res'],
                f'{r["final_cache_max_relative_error"]:.6e}',f'{r["positional_match_percent"]:.1f}%',
                f'{r["set_match_percent"]:.1f}%',f'{r["true_gain"]:.9f}',f'{r["gain_loss"]:+.6e}',
                f'{r["gain_loss_percent"]:+.6f}%',ms(t)])
        sections.append(table(['Dataset','tol','Mode','R','Certified R_tie','Unproved numerical ties','R_res',
            'Final cache max rel. error','Ordered match','Set match','True G','Reference G − G','Loss %','Total s mean ± SD'],rows))
        sections.append('The reference in the preceding table is dense spectral q12 surrogate greedy for Small-BA and floating-point tol10⁻¹² CELF for ca-GrQc. The latter is a tighter solve, not an exact symbolic tie-break oracle. The separately measured nominal tol10⁻⁷ CELF comparison is retained below; the tighter-reference numbers above are not replaced.\n')
        nominal=[]
        for r in x['rows']:
            if r['dataset']!='ca-GrQc' or 'nominal_reference' not in r:continue
            nr=r['nominal_reference'];cert=r.get('final_spectral_equivalence_certificate')
            nominal.append([r['dataset'],'Adaptive' if r['adaptive'] else 'Uncertified',
                f'{nr["positional_match_percent"]:.1f}%',f'{nr["set_match_percent"]:.1f}%',
                f'{r["positional_match_percent"]:.1f}%',f'{r["set_match_percent"]:.1f}%',
                'exact permutation similarity' if cert else 'not established'])
        if nominal:
            sections.append(table(['Dataset','Mode','Nominal ordered match','Nominal set match',
                'Tighter-solve ordered match','Tighter-solve set match','Final spectral equivalence to tighter reference'],nominal))
        certificates=[r.get('final_spectral_equivalence_certificate') for r in x['rows'] if r['dataset']=='ca-GrQc']
        cert=next((c for c in certificates if c),None)
        if cert:
            sections.append('Final ca-GrQc spectral-equality certificate: '+json.dumps(cert['permutation'])+
                '. This vertex permutation maps the observed final augmented matrix to the tighter-reference final matrix with exactly zero sparse difference entries. All represented matrix entries are integers; therefore every spectral trace is identical despite different literal tied edge representatives.\n')
        sections.append('The numerical comparison tolerance used by the implementation is2×10⁻¹¹ at the smallest solve tolerance; it is explicitly distinct from the10⁻¹⁴ diagnostic tie threshold. Ordered matches and unordered top-k matches are reported separately. Complete events include the current selected prefix, competing candidates, observed intervals, independently evaluated margins, and any exact automorphism certificate.\n')
    target=out/'fastfrac_unified_science_sections.md';target.write_text('\n'.join(sections)+'\n')
    return target


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='fastfrac_unified_results')
    parser.add_argument('--repeats',type=int,default=5);parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--groups',nargs='+',default=['scan','quadrature','opt','adaptive'])
    args=parser.parse_args()
    import unified_backend
    unified_backend.install(args.threads)
    result=run_science(args.output,args.repeats,args.groups)
    render_science_markdown(args.output)
    print(json.dumps(dict(seconds=result['seconds'],groups=list(result['groups'])),indent=2))
