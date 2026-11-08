"""Five-repeat, single-backend CPU benchmarks; every completed row is measured.

No old timing enters these outputs. Only BA-100k Naive is explicitly a measured
one-round projection. Fresh initialization is paid separately by every method
and repetition. CPU benchmark jobs are serialized by the driver.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
from time import perf_counter
from dataclasses import asdict
import argparse, gzip, hashlib, json, platform, subprocess
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
import fastfrac_cpu_benchmark as ff
import unified_backend as ub
from fastfrac_revision_experiments import locked, matrix, uniform_candidates
from revision_structure_opt import _original_sbm, _heterogeneous_sbm, _matrix

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'fastfrac_unified_results'
METHODS=('CELF','Inc','Dynamic-Logdet','Naive')

def save(path,value):
    ff.write_json(path,value)

def summary(rows):
    keys=('init_seconds','online_seconds','seconds','init_calls','online_calls','calls',
          'krylov_spmv','residual_spmv','mean_iterations','executed_padded_spmv')
    ans={'repetitions':len(rows)}
    for key in keys:
        x=np.array([r[key] for r in rows],float)
        ans[key]={'mean':float(x.mean()),'std':float(x.std(ddof=1)) if len(x)>1 else None,
                  'min':float(x.min()),'max':float(x.max())}
    ans['all_same_sequence']=all(r['selected']==rows[0]['selected'] for r in rows)
    ans['selected']=rows[0]['selected']
    return ans

def run_one(method,A,C,k,shifts,weights,alpha=1.,tol=1e-7):
    # One common phase boundary: materialize the scalar resistance/error table.
    ub.reset_counters(); st=ff.Stats(); start=perf_counter()
    index=ff.scalar_ms_batch(A,C,shifts,tol,st)
    init=perf_counter()-start; init_stats=asdict(st)
    init_counter=ub.get_counters(); saved=ff.scalar_ms_batch; used=False
    original_stats,original_add=ff.Stats,ff.add_edge
    holders=[];per_round=[];previous=[len(C)]
    def make_stats():
        v=original_stats();holders.append(v);return v
    def from_index(B,edges,ts,tolerance,stats,batch_size=48):
        nonlocal used
        if not used:
            used=True
            assert np.array_equal(edges,C) and np.array_equal(ts,shifts)
            for key,value in init_stats.items():setattr(stats,key,value)
            return tuple(x.copy() for x in index)
        return saved(B,edges,ts,tolerance,stats,batch_size)
    def traced_add(B,e,alpha=1.):
        ans=original_add(B,e,alpha)
        current=holders[0].calls;per_round.append(current-previous[0]);previous[0]=current
        return ans
    ff.scalar_ms_batch=from_index;ff.Stats=make_stats;ff.add_edge=traced_add
    t0=perf_counter()
    try:
        if method=='Naive':r=ff.run_naive(A,C,k,shifts,weights,tol,verbose=False,alpha=alpha)
        elif method=='Inc':r=ff.run_inc(A,C,k,shifts,weights,tol,alpha=alpha)
        else:r=ff.run_celf(A,C,k,shifts,weights,tol,alpha=alpha)
    finally:
        ff.scalar_ms_batch=saved;ff.Stats=original_stats;ff.add_edge=original_add
    online=perf_counter()-t0
    counters=ub.get_counters()
    r.update(init_seconds=init,online_seconds=online,seconds=init+online,
             phase_boundary='scalar resistance/error table complete; all score/heap/selection work online',
             online_rhs_per_round=per_round,mean_iterations=float(np.mean(counters['iterations'])),
             executed_padded_spmv=counters['executed_padded_spmv'],
             counters=counters,init_counters=init_counter,method=method,status='measured_complete')
    assert sum(per_round)==r['online_calls']
    return r

def real_graph(name,offset):
    path=ROOT/'fastfrac_realworld_data'/f'{name}.txt.gz'
    raw=np.loadtxt(gzip.open(path),dtype=np.int64)
    ids,inv=np.unique(raw,return_inverse=True);edges=np.sort(inv.reshape(-1,2),axis=1)
    edges=np.unique(edges[edges[:,0]!=edges[:,1]],axis=0);n=len(ids)
    C=uniform_candidates(n,edges,2000,20261031+6000+offset)
    A=matrix(n,edges);k=20
    return A,C,dict(name=name,n=n,m=len(edges),M=len(C),k=k,
       upper=float(1+2*((A.diagonal()-1).max()+k)),alpha=1.,s=.5,
       candidate_seed=20261031+6000+offset,source=f'https://snap.stanford.edu/data/{name}.html',
       source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
       preprocessing='retain all endpoint vertices; undirect; deduplicate; remove self-loops')

def ba_graph(n):
    cache=OUT/'graphs'/f'BA-{n}.npz';cache.parent.mkdir(parents=True,exist_ok=True)
    if cache.exists() and cache.stat().st_size>0:
        d=np.load(cache);edges=d['edges'];C=d['C']
    else:
        G=ff.nx.barabasi_albert_graph(n,3,seed=20261031+n)
        edges=np.array(sorted(tuple(sorted(e)) for e in G.edges()),dtype=np.int64)
        C=uniform_candidates(n,edges,2000,20261031+n+101)
        temporary=cache.with_name(cache.stem+'.tmp.npz')
        np.savez_compressed(temporary,edges=edges,C=C)
        temporary.replace(cache)
    A=matrix(n,edges);k=20
    return A,C,dict(name=f'BA-{n//1000}k',n=n,m=len(edges),M=len(C),k=k,
       upper=float(1+2*((A.diagonal()-1).max()+k)),alpha=1.,s=.5,
       max_degree=float((A.diagonal()-1).max()),generator='Barabasi-Albert(n,3)',
       seed=20261031+n,candidate_seed=20261031+n+101)

def sbm4k():
    n=4000;seed=20261101;G=ff.nx.Graph();G.add_nodes_from(range(n))
    for b in range(4):
        H=ff.nx.fast_gnp_random_graph(1000,.03,seed=seed+b)
        G.add_edges_from((1000*b+u,1000*b+v) for u,v in H.edges())
    G.add_weighted_edges_from([(0,1000,.02),(1000,2000,.02),(2000,3000,.02)])
    rng=np.random.default_rng(seed+101);degree=np.array([G.degree(i) for i in range(n)])
    labels=np.repeat(np.arange(4),1000);low=[];high=[]
    for b in range(4):
        ids=np.arange(b*1000,(b+1)*1000);order=ids[np.lexsort((ids,degree[ids]))]
        low.append(order[:250]);high.append(order[-250:])
    pairs=set()
    while len(pairs)<300:
        a,b=rng.choice(4,2,replace=False);u=int(rng.choice(low[a]));v=int(rng.choice(low[b]))
        e=tuple(sorted((u,v)))
        if not G.has_edge(*e):pairs.add(e)
    while len(pairs)<600:
        b=int(rng.integers(4));u,v=map(int,rng.choice(high[b],2,replace=False));e=tuple(sorted((u,v)))
        if not G.has_edge(*e):pairs.add(e)
    C=np.array(sorted(pairs),dtype=np.int64);A=_matrix(G,50.)
    np.savez_compressed(OUT/'graphs'/'SBM-4k.npz',edges=np.array([(u,v,d.get('weight',1.)) for u,v,d in G.edges(data=True)]),C=C)
    return A,C,dict(name='Multiscale-SBM-4k',n=n,m=G.number_of_edges(),M=600,k=20,
       upper=float(1+2*50*(max(dict(G.degree(weight='weight')).values())+20)),alpha=50.,s=.1,
       seed=seed,candidate_seed=seed+101)

def work(name,A,C,meta,repeats=5,methods=METHODS,path=None):
    path=path or OUT/'performance'/f'{name}.json';path.parent.mkdir(parents=True,exist_ok=True)
    result=json.loads(path.read_text()) if path.exists() else dict(metadata=meta,runs={},status='running')
    shifts,w,eta=ff.rule(meta.get('s',.5),12,meta['upper'])
    result['quadrature_bound']=eta
    for rep in range(repeats):
        # Deterministic rotation prevents always giving one algorithm cold caches.
        order=list(methods[rep%len(methods):])+list(methods[:rep%len(methods)])
        for method in order:
            rows=result['runs'].setdefault(method,[])
            if len(rows)>rep:continue
            ts,ws=(np.array([0.]),np.array([1.])) if method=='Dynamic-Logdet' else (shifts,w)
            r=run_one(method,A,C,meta['k'],ts,ws,meta.get('alpha',1.))
            r['repeat']=rep+1;rows.append(r);save(path,result)
            print(f'UNIFIED {name} {method} {rep+1}/{repeats} total={r["seconds"]:.6f}s online={r["online_seconds"]:.6f}s RHS={r["calls"]}',flush=True)
    result['summary']={m:summary(r) for m,r in result['runs'].items()}
    if 'CELF' in result['runs'] and 'Inc' in result['runs']:
        result['celf_inc_identical_sequence']=all(a['selected']==b['selected'] for a,b in zip(result['runs']['CELF'],result['runs']['Inc']))
    if 'Naive' in result['runs']:
        result['naive_celf_identical_sequence']=result['runs']['Naive'][0]['selected']==result['runs'].get('CELF',result['runs']['Naive'])[0]['selected']
    result['status']='complete';save(path,result)
    return result

def main_workloads(repeats=5):
    for name in ('Small-BA','Medium-WS','Large-PowerCluster'):
        A,C,meta=locked(name);meta.update(alpha=1.,s=.5)
        work(name,A,C,meta,repeats)
    for offset,name in enumerate(('email-Eu-core','ca-GrQc','ca-CondMat')):
        A,C,meta=real_graph(name,offset);work(name,A,C,meta,repeats)

def scaling(repeats=5):
    results=[]
    for n in (1000,5000,10000,25000,50000,100000):
        A,C,meta=ba_graph(n)
        methods=METHODS if n<=50000 else ('CELF','Inc','Dynamic-Logdet')
        r=work(meta['name'],A,C,meta,repeats,methods)
        timing=[ub.measure_spmv(A,repeats=150,threads=1) for _ in range(5)]
        r['single_spmv_microbenchmark']=timing
        if n==100000:
            projection=[]
            for rep in range(repeats):
                ts,w,_=ff.rule(.5,12,meta['upper']);a=run_one('Naive',A,C,1,ts,w)
                a['status']='one_round_measured_full_run_projected'
                a['projected_total_seconds']=a['seconds']*(20*2000-190)/2000
                a['projection_formula']='T1*(k*M-k*(k-1)/2)/M; not a completed greedy run'
                projection.append(a)
            r['Naive_one_round']=projection
        save(OUT/'performance'/f'{meta["name"]}.json',r);results.append(r)
    save(OUT/'scaling.json',{'workloads':results})

def curves(repeats=5):
    A,C,meta=locked('Medium-WS')
    edges=np.array(json.loads((ROOT/'fastfrac_cpu_results/Medium-WS.json').read_text())['base_edges'])
    pool=uniform_candidates(2500,edges,4000,20261031+7000)
    upper=float(1+2*((A.diagonal()-1).max()+60))
    rows=[]
    for M in (250,500,1000,2000,4000):
        m=dict(meta,name=f'pool-{M}',M=M,k=20,upper=upper,alpha=1.,s=.5)
        rows.append(work(m['name'],A,pool[:M],m,repeats,('CELF','Inc','Naive'),OUT/'curves'/f'pool-{M}.json'))
    for k in (5,10,20,40,60):
        m=dict(meta,name=f'budget-{k}',M=1000,k=k,upper=upper,alpha=1.,s=.5)
        rows.append(work(m['name'],A,pool[:1000],m,repeats,('CELF','Inc','Naive'),OUT/'curves'/f'budget-{k}.json'))
    save(OUT/'curves.json',rows)

def multiscale(repeats=5):
    for factory in (_original_sbm,_heterogeneous_sbm):
        G,C,labels,meta=factory();A=_matrix(G,50.)
        if meta['name']=='Multi-Scale-SBM':meta['name']='Multiscale-SBM'
        for s in (.1,.5,.85):
            m=dict(meta,s=s,alpha=50.,upper=float(1+100*(max(dict(G.degree(weight='weight')).values())+15)))
            work(f'{meta["name"]}-s{s}',A,C,m,repeats,('CELF','Inc','Dynamic-Logdet'))
    A,C,meta=sbm4k();work(meta['name'],A,C,meta,repeats,('CELF','Inc','Dynamic-Logdet'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--group',choices=['main','scaling','curves','multiscale','all'],default='all')
    p.add_argument('--repeats',type=int,default=5);p.add_argument('--threads',type=int,default=8)
    a=p.parse_args();OUT.mkdir(exist_ok=True);(OUT/'graphs').mkdir(exist_ok=True)
    ub.install(a.threads)
    with threadpool_limits(limits=1):
        # OpenMP is explicitly configured inside each native entry point.
        for group,fn in [('main',main_workloads),('multiscale',multiscale),('curves',curves),('scaling',scaling)]:
            if a.group in (group,'all'):fn(a.repeats)

if __name__=='__main__':main()
