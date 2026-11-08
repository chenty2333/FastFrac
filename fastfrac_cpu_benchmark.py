"""Reproducible CPU benchmark; no dense matrices in medium/large optimization.

Dependencies: numpy scipy networkx==3.5 threadpoolctl. Run:
  python fastfrac_cpu_benchmark.py --output fastfrac_cpu_results
All timed methods run serially with one BLAS thread. A multi-shift call means
one candidate RHS, even when independent RHSs are batched in a sparse SpMM.
Scalar Lanczos uses short recurrences. Vector rechecks/reference evaluation
use full reorthogonalization and explicit true residuals. Floating intervals
are numerical error monitors, not directed-rounding machine certificates.
Greedy quadrature and reference-evaluation functions accept alpha=1.0; callers
using other alpha values must supply A0=I+alpha*L0 and a matching spectral bound.
The original benchmark generator, Exact-Eig loop, and sensitivity suite use alpha=1.
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import sys
from pathlib import Path
if Path('/tmp/fastfrac_bench_deps').exists():
    sys.path.insert(0, '/tmp/fastfrac_bench_deps')
import argparse
from collections import Counter
from dataclasses import dataclass, asdict
import hashlib
import heapq
import json
import platform
from time import perf_counter
import numpy as np
import scipy
from scipy import sparse
from scipy.linalg import eigh, eigvalsh
from scipy.special import roots_jacobi
from threadpoolctl import threadpool_limits
import networkx as nx

SEED = 20261031
S_VALUES = (0.2, 0.5, 0.8)
SPECS = [('Small-BA', 400, 200, 10), ('Medium-WS', 2500, 1000, 20),
         ('Large-PowerCluster', 10000, 3000, 30)]


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


def rule(s, q, upper):
    # alpha+beta=-1 triggers an unused 0/0 branch inside some SciPy versions.
    with np.errstate(invalid='ignore', divide='ignore'):
        x, w = roots_jacobi(q, -s, s-1.0)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(w)) and np.all(w>0)):
        raise ArithmeticError('Invalid Gauss-Jacobi nodes or weights')
    tau = np.sqrt(upper)
    shifts = tau*(1+x)/(1-x)
    weights = 2*s*np.sin(np.pi*s)/np.pi*tau**s*w/(1-x)
    rho = (upper**0.25+1)/(upper**0.25-1)
    z = np.exp(-2*q*np.log(rho))
    return shifts, weights, float(4*z/(1+z)**2)


def make_graph(index):
    name, n, M, k = SPECS[index]
    seed = SEED + 1000*index
    if index == 0:
        G = nx.barabasi_albert_graph(n, 3, seed=seed)
        generator = 'barabasi_albert_graph(n=400,m=3)'
    elif index == 1:
        G = nx.connected_watts_strogatz_graph(n, 8, 0.1, seed=seed)
        generator = 'connected_watts_strogatz_graph(n=2500,k=8,p=0.1)'
    else:
        G = nx.powerlaw_cluster_graph(n, 4, 0.45, seed=seed)
        generator = 'powerlaw_cluster_graph(n=10000,m=4,p=0.45)'
    assert nx.is_connected(G)
    edges = np.array(sorted(tuple(sorted(e)) for e in G.edges()), dtype=np.int64)
    degree = np.array([G.degree(i) for i in range(n)], dtype=np.int64)
    communities = sorted(nx.community.louvain_communities(G, seed=seed+17), key=min)
    groups = [np.array(sorted(x), dtype=np.int64) for x in communities if len(x)>1]
    label = np.empty(n, dtype=np.int64)
    for g, vertices in enumerate(communities):
        label[list(vertices)] = g
    bins = np.array_split(np.lexsort((np.arange(n), degree)), 4)
    neighbors = [np.array(sorted(G[i]), dtype=np.int64) for i in range(n)]
    rng = np.random.default_rng(seed+101)
    candidate, categories = set(), {}
    # 50% uniform nonedges; 25% two-hop; 25% degree-stratified intercommunity.
    quotas = [('uniform', M//2), ('two_hop', M//4), ('cross_degree', M-M//2-M//4)]
    for mode, target in quotas:
        accepted = attempts = 0
        while accepted < target:
            attempts += 1
            if mode == 'two_hop' and attempts < 1000000:
                center = int(rng.integers(n))
                if len(neighbors[center]) < 2:
                    continue
                u, v = map(int, rng.choice(neighbors[center], 2, replace=False))
            elif mode == 'cross_degree' and attempts < 1000000:
                a, b = rng.integers(4, size=2)
                u, v = int(rng.choice(bins[a])), int(rng.choice(bins[b]))
                if label[u] == label[v]:
                    continue
            else:
                u, v = map(int, rng.integers(n, size=2))
            e = (min(u,v), max(u,v))
            if u == v or G.has_edge(u,v) or e in candidate:
                continue
            candidate.add(e); categories[e] = mode; accepted += 1
    C = np.array(sorted(candidate), dtype=np.int64)
    L = nx.laplacian_matrix(G, nodelist=range(n)).astype(float).tocsr()
    A = (sparse.eye(n, format='csr')+L).tocsr()
    metadata = dict(name=name, n=n, m=len(edges), M=M, k=k, seed=seed,
                    candidate_seed=seed+101, generator=generator,
                    community_method='Louvain, seed=graph_seed+17',
                    communities=len(communities), max_degree=int(degree.max()),
                    mean_clustering=float(nx.average_clustering(G)),
                    candidate_mix=dict(Counter(categories.values())),
                    candidate_intercommunity=int(np.sum(label[C[:,0]] != label[C[:,1]])),
                    candidate_with_common_neighbor=sum(bool(set(G[u]) & set(G[v])) for u,v in C),
                    degree_rank_bin_ranges=[[int(degree[x].min()),int(degree[x].max())] for x in bins],
                    candidate_sha256=hashlib.sha256(C.astype('<i8').tobytes()).hexdigest(),
                    upper=float(1+2*(degree.max()+k)))
    return A, C, degree, edges, label, metadata


def add_edge(A, e, alpha=1.0):
    u, v = map(int,e)
    E = sparse.coo_matrix(([alpha,alpha,-alpha,-alpha],([u,v,u,v],[u,v,v,u])),shape=A.shape)
    return (A+E).tocsr()


@dataclass
class Stats:
    calls: int = 0
    scalar_calls: int = 0
    vector_calls: int = 0
    krylov_spmv: int = 0
    sparse_kernel_calls: int = 0
    residual_spmv: int = 0
    fallback_calls: int = 0
    max_residual_proxy: float = 0.0
    max_true_residual: float = 0.0


def intervals(d, residual, shifts):
    # Explicit allowance for double-precision arithmetic, not formal interval math.
    pad = 5e-12*np.maximum(np.abs(d), 1e-300)
    low = np.maximum(0, d-pad)
    high = d+residual**2/(1+shifts)+pad
    return low, high


def scalar_ms_batch(A, edges, shifts, tol, stats, batch_size=48):
    """Independent three-term Lanczos columns, with converged columns removed."""
    n, count, q = A.shape[0], len(edges), len(shifts)
    out = np.empty((count,q)); residual_out = np.empty_like(out)
    stats.calls += count; stats.scalar_calls += count
    root2 = np.sqrt(2.)
    for first in range(0,count,batch_size):
        block = edges[first:first+batch_size]
        active = np.arange(len(block))
        V = np.zeros((n,len(block)))
        V[block[:,0],active] = 1/root2
        V[block[:,1],active] = -1/root2
        previous = np.zeros_like(V)
        beta = np.zeros(len(block))
        pivot = forward = None
        d = np.zeros((len(block),q))
        for step in range(min(4*n,800)):
            Z = A@V
            stats.krylov_spmv += len(active); stats.sparse_kernel_calls += 1
            Z -= previous*beta[None,:]
            diagonal = np.einsum('ij,ij->j',V,Z)
            Z -= V*diagonal[None,:]
            next_beta = np.sqrt(np.einsum('ij,ij->j',Z,Z))
            if step == 0:
                pivot = diagonal[:,None]+shifts
                forward = np.full_like(pivot,root2)
            else:
                multiplier = beta[:,None]/pivot
                pivot = diagonal[:,None]+shifts-beta[:,None]*multiplier
                forward = -multiplier*forward
            if np.any(pivot <= 0) or not np.all(np.isfinite(pivot)):
                raise ArithmeticError('Nonpositive shifted LDL pivot')
            d += forward*forward/pivot
            residual = next_beta[:,None]*np.abs(forward/pivot)
            done = np.max(residual,axis=1) <= root2*tol
            done |= next_beta < 1e-14
            if np.any(done):
                out[first+active[done]] = d[done]
                residual_out[first+active[done]] = residual[done]
                stats.max_residual_proxy = max(stats.max_residual_proxy,
                    float(np.max(residual[done])/root2))
            keep = ~done
            if not np.any(keep):
                break
            if np.all(keep):
                previous = V
                Z /= next_beta[None,:]
                V = Z
                beta = next_beta
            else:
                active = active[keep]
                previous = np.ascontiguousarray(V[:,keep])
                V = np.ascontiguousarray(Z[:,keep])
                V /= next_beta[keep][None,:]
                beta = next_beta[keep]
                pivot, forward, d = pivot[keep], forward[keep], d[keep]
        else:
            for index in active:
                stats.fallback_calls += 1
                dv, lo, hi, X, rr = vector_ms(A, block[index], shifts,tol,stats)
                out[first+index]=dv; residual_out[first+index]=rr
    lo, hi = intervals(out,residual_out,shifts)
    return out,lo,hi


def vector_ms(A, edge, shifts, tol, stats):
    """Shared Lanczos with full reorthogonalization and true residual validation."""
    stats.calls += 1; stats.vector_calls += 1
    n,q = A.shape[0],len(shifts)
    maxiter = min(n,600)
    Q = np.empty((n,maxiter), order='F')
    b = np.zeros(n);b[edge[0]]=1;b[edge[1]]=-1
    current = b/np.sqrt(2.); previous=np.zeros(n); beta=0.
    pivots, forwards, off = [],[],[]
    for step in range(maxiter):
        Q[:,step]=current
        z=A@current-beta*previous
        stats.krylov_spmv += 1;stats.sparse_kernel_calls += 1
        diagonal=float(current@z);z-=diagonal*current
        V=Q[:,:step+1]
        for _ in range(2):
            z-=V@(V.T@z)
        next_beta=float(np.linalg.norm(z))
        if not pivots:
            pivot=diagonal+shifts;forward=np.full(q,np.sqrt(2.))
        else:
            multiplier=beta/pivots[-1]
            pivot=diagonal+shifts-beta*multiplier
            forward=-multiplier*forwards[-1]
            off.append(beta)
        if np.any(pivot<=0):raise ArithmeticError('Nonpositive vector LDL pivot')
        pivots.append(pivot);forwards.append(forward)
        proxy=next_beta*np.abs(forward/pivot)
        if np.max(proxy) <= np.sqrt(2.)*tol or next_beta<1e-14 or step==maxiter-1:
            h=len(pivots);Y=np.empty((h,q));Y[-1]=forwards[-1]/pivots[-1]
            for i in range(h-2,-1,-1):
                Y[i]=forwards[i]/pivots[i]-off[i]/pivots[i]*Y[i+1]
            X=Q[:,:h]@Y
            residual=b[:,None]-(A@X+X*shifts)
            stats.residual_spmv+=q;stats.sparse_kernel_calls+=1
            rr=np.linalg.norm(residual,axis=0)
            if np.max(rr) <= np.sqrt(2.)*tol*1.1:
                stats.max_true_residual=max(stats.max_true_residual,float(np.max(rr)/np.sqrt(2.)))
                bx=X[edge[0]]-X[edge[1]]
                dvar=bx+np.einsum('ij,ij->j',X,residual)
                low,high=intervals(dvar,rr,shifts)
                return np.maximum(0,dvar),low,high,X,rr
        if next_beta<1e-14:break
        previous,current,beta=current,z/next_beta,next_beta
    raise ArithmeticError(f'Vector solve failed true residual target {tol}, max={np.max(rr)}')


def scores(R, weights, alpha=1.0):
    return np.log1p(alpha*np.maximum(0,R))@weights


def run_naive(A0,C,k,shifts,weights,tol,verbose=True,alpha=1.0):
    A=A0.copy();st=Stats();alive=np.ones(len(C),bool);chosen=[];gain=[]
    start=perf_counter();init_seconds=None;init_calls=0
    for r in range(k):
        ids=np.flatnonzero(alive)
        R,lo,hi=scalar_ms_batch(A,C[ids],shifts,tol,st)
        sc=scores(R,weights,alpha);p=int(np.argmax(sc));i=int(ids[p])
        if r==0:init_seconds=perf_counter()-start;init_calls=st.calls
        chosen.append(i);gain.append(float(sc[p]));alive[i]=False;A=add_edge(A,C[i],alpha)
        if verbose and (r==0 or (r+1)%5==0 or r==k-1):
            print(f'  Naive round {r+1}/{k}, elapsed={perf_counter()-start:.3f}s',flush=True)
    return finish('Naive-Quad-Greedy',chosen,gain,start,init_seconds,init_calls,st)


def run_celf(A0,C,k,shifts,weights,tol,alpha=1.0):
    A=A0.copy();st=Stats();chosen=[];gain=[];start=perf_counter()
    R,lo,hi=scalar_ms_batch(A,C,shifts,tol,st)
    point,low,high=scores(R,weights,alpha),scores(lo,weights,alpha),scores(hi,weights,alpha)
    init_seconds=perf_counter()-start;init_calls=st.calls
    heap=[(-float(high[i]),i,0,float(low[i]),float(point[i]),tol) for i in range(len(C))]
    heapq.heapify(heap);rechecks=0
    while len(chosen)<k:
        neg,i,stamp,lower,value,localtol=heapq.heappop(heap)
        if stamp!=len(chosen):
            R,l,u=scalar_ms_batch(A,C[i:i+1],shifts,tol,st,batch_size=1)
            value=float(scores(R,weights,alpha)[0]);lower=float(scores(l,weights,alpha)[0])
            upper=min(-neg,float(scores(u,weights,alpha)[0]));stamp=len(chosen);localtol=tol
        else:upper=-neg
        competitor=-heap[0][0] if heap else -np.inf
        if lower>competitor:
            chosen.append(i);gain.append(value);A=add_edge(A,C[i],alpha)
        elif upper<=competitor:
            heapq.heappush(heap,(-upper,i,stamp,lower,value,localtol))
        else:
            # Rare overlap: reconstruct true residual rather than trusting the short recurrence.
            localtol=max(localtol*.01,1e-12)
            dv,l,u,_,_=vector_ms(A,C[i],shifts,localtol,st);rechecks+=1
            value=float(scores(dv,weights,alpha));lower=float(scores(l,weights,alpha));upper=float(scores(u,weights,alpha))
            if localtol==1e-12 and upper-lower<2e-11*max(value,1.):
                # Lexicographic tie at a declared numerical comparison tolerance.
                if value>=competitor-2e-11*max(value,1.):
                    chosen.append(i);gain.append(value);A=add_edge(A,C[i],alpha);continue
            heapq.heappush(heap,(-upper,i,stamp,lower,value,localtol))
    ans=finish('FastFrac-CELF',chosen,gain,start,init_seconds,init_calls,st)
    ans['extra_true_residual_rechecks']=rechecks
    return ans


def run_inc(A0,C,k,shifts,weights,tol,alpha=1.0):
    A=A0.copy();st=Stats();chosen=[];gain=[];alive=np.ones(len(C),bool)
    start=perf_counter();R,lo,hi=scalar_ms_batch(A,C,shifts,tol,st)
    init_seconds=perf_counter()-start;init_calls=st.calls;rejected=0
    for r in range(k):
        upper=scores(hi,weights,alpha)
        heap=[(-float(upper[i]),int(i)) for i in np.flatnonzero(alive)];heapq.heapify(heap)
        localtol=tol
        while True:
            _,i=heapq.heappop(heap)
            dv,dl,du,X,rr=vector_ms(A,C[i],shifts,localtol,st)
            R[i],lo[i],hi[i]=dv,dl,du
            actual=float(scores(dv,weights,alpha));lower=float(scores(dl,weights,alpha));upperi=float(scores(du,weights,alpha))
            competitor=-heap[0][0] if heap else -np.inf
            if lower>competitor:
                break
            if localtol<=1e-12 and actual>=competitor-2e-11*max(actual,1.):
                break
            heapq.heappush(heap,(-upperi,i));rejected+=1
            if upperi>=competitor:localtol=max(localtol*.01,1e-12)
            else:localtol=tol
        chosen.append(i);gain.append(actual)
        cross=X[C[:,0]]-X[C[:,1]]
        # Fresh variational denominator and correction interval propagation.
        R-=alpha*cross**2/(1+alpha*dv)
        delta=np.sqrt(2.)*rr/(1+shifts)
        correction_low=alpha*np.maximum(np.abs(cross)-delta,0)**2/(1+alpha*du)
        correction_high=alpha*(np.abs(cross)+delta)**2/(1+alpha*dl)
        lo=np.maximum(0,lo-correction_high)
        hi=np.maximum(lo,hi-correction_low)
        alive[i]=False;A=add_edge(A,C[i],alpha)
    ans=finish('FastFrac-Inc',chosen,gain,start,init_seconds,init_calls,st)
    ans['rejected_or_refined_heap_tops']=rejected
    ans['max_cache_interval_relative_width']=float(np.max((hi-lo)/np.maximum(R,1e-300)))
    return ans


def finish(name,chosen,gain,start,init_seconds,init_calls,st):
    return dict(name=name,selected=list(map(int,chosen)),marginal_sum=float(sum(gain)),
                gains=gain,seconds=perf_counter()-start,init_seconds=init_seconds,
                init_calls=init_calls,online_calls=st.calls-init_calls,**asdict(st))


def run_exact(A0,C,k,s):
    A=A0.toarray();alive=np.ones(len(C),bool);chosen=[];gain=[];trace_grid=[]
    start=perf_counter();lam=eigvalsh(A,check_finite=False);base_trace=float(np.sum(lam**s));calls=1
    for r in range(k):
        ids=np.flatnonzero(alive);row=[]
        best=-np.inf;winner=None;winner_trace=None
        for i in ids:
            u,v=C[i];trial=A.copy()
            trial[u,u]+=1;trial[v,v]+=1;trial[u,v]-=1;trial[v,u]-=1
            tr=float(np.sum(eigvalsh(trial,check_finite=False,overwrite_a=True)**s));calls+=1
            value=tr-base_trace;row.append(value)
            if value>best:best=value;winner=int(i);winner_trace=tr
        trace_grid.append(dict(ids=ids.tolist(),marginals=row))
        chosen.append(winner);gain.append(best);alive[winner]=False
        u,v=C[winner];A[u,u]+=1;A[v,v]+=1;A[u,v]-=1;A[v,u]-=1;base_trace=winner_trace
        if r==0 or (r+1)%5==0:print(f'  Exact-Eig round {r+1}/{k}, elapsed={perf_counter()-start:.3f}s',flush=True)
    return dict(name='Exact-Eig',selected=chosen,gains=gain,marginal_sum=sum(gain),
                seconds=perf_counter()-start,calls=0,eigendecompositions=calls,
                krylov_spmv=0,residual_spmv=0,trace_grid=trace_grid)


def eigen_gain(lam0,A,C,selection,s,alpha=1.0):
    B=A.copy()
    for i in selection:B=add_edge(B,C[i],alpha)
    lam=eigvalsh(B.toarray(),check_finite=False)
    return float(np.sum(lam0**s*np.expm1(s*np.log1p((lam-lam0)/lam0))))


def reference_gain(A0,C,selection,upper,s_values=S_VALUES,eta_target=1e-11,alpha=1.0):
    """Positive path integral, higher q/tighter solves, shared across all s."""
    rho=(upper**.25+1)/(upper**.25-1)
    qref=max(24,int(np.ceil(np.log(4/eta_target)/(2*np.log(rho)))))
    rules=[rule(s,qref,upper) for s in s_values]
    shifts=np.concatenate([r[0] for r in rules]);stats=Stats();A=A0.copy()
    value=np.zeros(len(s_values));low=np.zeros_like(value);high=np.zeros_like(value)
    start=perf_counter()
    for i in sorted(selection):
        d,l,u,_,_=vector_ms(A,C[i],shifts,1e-11,stats)
        for a,(_,w,eta) in enumerate(rules):
            sl=slice(a*qref,(a+1)*qref)
            value[a]+=float(scores(d[sl],w,alpha));low[a]+=float(scores(l[sl],w,alpha));high[a]+=float(scores(u[sl],w,alpha))
        A=add_edge(A,C[i],alpha)
    return dict(method='high-precision path quadrature with true residual checks',q=qref,
                tolerance=1e-11,seconds=perf_counter()-start,stats=asdict(stats),
                by_s={str(s):dict(gain=float(value[i]),lower=float(low[i]),
                         upper=float(high[i]/(1-rules[i][2])),quadrature_eta=rules[i][2],
                         relative_interval_width=float((high[i]/(1-rules[i][2])-low[i])/value[i]))
                      for i,s in enumerate(s_values)})


def sensitivity(A0,C,k,upper,exact,tol):
    # Fix the entire exact greedy trajectory for all q comparisons.
    states=[];A=A0.copy()
    for r,i in enumerate(exact['selected']):
        lam,V=eigh(A.toarray(),check_finite=False)
        ids=np.array(exact['trace_grid'][r]['ids'])
        comp=(V[C[ids,0]]-V[C[ids,1]])**2
        states.append((A.copy(),ids,lam,comp,np.array(exact['trace_grid'][r]['marginals'])))
        A=add_edge(A,C[i])
    output=[]
    for q in (4,8,12,16,24):
        shifts,weights,eta=rule(.5,q,upper)
        result=run_celf(A0,C,k,shifts,weights,tol)
        total_errors=[];quad_errors=[];st=Stats();audit_start=perf_counter()
        for A,ids,lam,comp,true in states:
            R,_,_=scalar_ms_batch(A,C[ids],shifts,tol,st)
            approx=scores(R,weights)
            spectral=scores(comp@(1/(lam[:,None]+shifts)),weights)
            total_errors.extend((np.abs(approx-true)/true).tolist())
            quad_errors.extend((np.abs(spectral-true)/true).tolist())
        output.append(dict(q=q,seconds=result['seconds'],calls=result['calls'],
                           selected=result['selected'],max_marginal_relative_error=max(total_errors),
                           mean_marginal_relative_error=float(np.mean(total_errors)),
                           max_quadrature_only_relative_error=max(quad_errors),
                           tested_marginals=len(total_errors),theoretical_eta=eta,
                           audit_seconds=perf_counter()-audit_start))
        print(f'  sensitivity q={q}: max_rel={max(total_errors):.3e}, time={result["seconds"]:.4f}s',flush=True)
    return output


def benchmark_graph(index,output,only_pilot=False):
    A0,C,degree,edges,labels,meta=make_graph(index)
    name=meta['name'];k=meta['k'];upper=meta['upper']
    data=dict(metadata=meta,environment=dict(python=platform.python_version(),numpy=np.__version__,
                scipy=scipy.__version__,networkx=nx.__version__,blas_threads=1,batch_size=48,
                timed_repetitions=1),parameters=dict(alpha=1.,s=.5,q=12,tol=1e-7),
                base_edges=edges.tolist(),candidates=C.tolist(),communities=labels.tolist(),runs={},quality={})
    path=Path(output)/f'{name}.json';Path(output).mkdir(exist_ok=True)
    print(f'{name}: {meta}',flush=True)
    shifts,weights,eta=rule(.5,12,upper);data['parameters']['eta_q12']=eta
    # One common untimed warm-up (no reusable cache).
    stats=Stats();R,l,u=scalar_ms_batch(A0,C[:24],shifts,1e-7,stats)
    vd,_,_,_,_=vector_ms(A0,C[0],shifts,1e-9,stats)
    error=float(np.max(np.abs(R[0]-vd)/vd))
    assert error<1e-7,(name,error)
    data['short_recurrence_pilot_relative_error']=error
    if only_pilot:
        start=perf_counter();st=Stats();scalar_ms_batch(A0,C[:96],shifts,1e-7,st)
        print('PILOT',name,'96 RHS seconds',perf_counter()-start,'counts',asdict(st),flush=True)
        return
    # Each method independently initializes. Runs are serial; no CPU benchmarks overlap.
    for fn in (run_naive,run_celf,run_inc):
        result=fn(A0,C,k,shifts,weights,1e-7)
        data['runs'][result['name']]=result
        write_json(path,data)
        print('TIMED',name,result['name'],result['seconds'],'calls',result['calls'],flush=True)
    if index==0:
        exact=run_exact(A0,C,k,.5);data['runs']['Exact-Eig']=exact;write_json(path,data)
        data['sensitivity']=sensitivity(A0,C,k,upper,exact,1e-7);write_json(path,data)
    for s in (.2,.8):
        sh,w,_=rule(s,12,upper)
        data['quality'][str(s)]={}
        for fn in (run_celf,run_inc):
            result=fn(A0,C,k,sh,w,1e-7)
            data['quality'][str(s)][result['name']]=result
            write_json(path,data)
            print('QUALITY',name,s,result['name'],result['seconds'],'calls',result['calls'],flush=True)
    data['quality']['0.5']={key:data['runs'][key] for key in ('FastFrac-CELF','FastFrac-Inc')}
    # Requested static regularized-effective-resistance top-k.
    st=Stats();tstart=perf_counter();R,_,_=scalar_ms_batch(A0,C,np.array([0.]),1e-9,st)
    static=np.lexsort((np.arange(len(C)),-R[:,0]))[:k].tolist()
    degscore=degree[C[:,0]]+degree[C[:,1]]
    deficit=np.lexsort((np.arange(len(C)),degscore))[:k].tolist()
    rng=np.random.default_rng(meta['seed']+909)
    random=[rng.choice(len(C),k,replace=False).tolist() for _ in range(10)]
    data['baselines']=dict(static_effective_resistance=static,degree_deficit=deficit,
                           random=random,random_seed=meta['seed']+909,
                           effres_seconds=perf_counter()-tstart,effres_stats=asdict(st))
    # A stronger control separates fractional-objective effects from static-vs-dynamic effects.
    dynamic=run_inc(A0,C,k,np.array([0.]),np.array([1.]),1e-7)
    data['baselines']['dynamic_logdet_greedy']=dynamic
    write_json(path,data)
    selections={}
    def add_selection(label,sel):
        key=','.join(map(str,sorted(sel)))
        selections.setdefault(key,dict(selected=sorted(sel),labels=[]))['labels'].append(label)
    for label,result in data['runs'].items():add_selection(label,result['selected'])
    for s,methods in data['quality'].items():
        for label,result in methods.items():add_selection(f'{label}@s={s}',result['selected'])
    add_selection('Static-EffRes',static);add_selection('Degree-Deficit',deficit)
    add_selection('Dynamic-Logdet',dynamic['selected'])
    for i,sel in enumerate(random):add_selection(f'Random-{i}',sel)
    for row in data.get('sensitivity',[]):add_selection(f'q={row["q"]}',row['selected'])
    audit_start=perf_counter();data['reference_evaluations']={}
    lam0=eigvalsh(A0.toarray(),check_finite=False) if index==0 else None
    for a,(key,obj) in enumerate(selections.items()):
        if index==0:
            obj['evaluation']=dict(method='dense eigenvalues',by_s={str(s):dict(
                gain=eigen_gain(lam0,A0,C,obj['selected'],s)) for s in S_VALUES})
        else:
            obj['evaluation']=reference_gain(A0,C,obj['selected'],upper)
        data['reference_evaluations'][key]=obj
        write_json(path,data)
        if a%5==0 or a+1==len(selections):
            print(f'  {name} reference sets {a+1}/{len(selections)}, elapsed={perf_counter()-audit_start:.2f}s',flush=True)
    data['audit_seconds']=perf_counter()-audit_start
    data['complete']=True
    write_json(path,data)
    print('COMPLETE',name,'audit_seconds',data['audit_seconds'],flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='fastfrac_cpu_results')
    parser.add_argument('--dataset',choices=['all','small','medium','large'],default='all')
    parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args()
    indices=range(3) if args.dataset=='all' else [{'small':0,'medium':1,'large':2}[args.dataset]]
    with threadpool_limits(limits=1):
        for index in indices:benchmark_graph(index,args.output,args.pilot)


if __name__=='__main__':
    main()
