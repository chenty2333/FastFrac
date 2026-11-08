"""Million-vertex SNAP timing runs on the same kernel and defaults as Exp-2.

Same preprocessing and M=2000, k=20, s=0.5, alpha=1, q=12, tol=1e-7 as the
smaller SNAP workloads; Naive is not run because each round repeats a full
initialization. Results resume by repetition, so --repeats can be raised later.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
from time import perf_counter
import argparse, gzip, hashlib, json, platform, resource, subprocess, traceback
import numpy as np
from threadpoolctl import threadpool_limits
import unified_backend as ub
import unified_performance as up
from fastfrac_revision_experiments import matrix

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'fastfrac_realworld_data'
OUT=ROOT/'fastfrac_large_results'
GRAPHS={'com-Youtube':'bigdata/communities/com-youtube.ungraph.txt.gz',
        'roadNet-CA':'roadNet-CA.txt.gz',
        'as-Skitter':'as-skitter.txt.gz',
        'com-LiveJournal':'bigdata/communities/com-lj.ungraph.txt.gz',
        'com-DBLP':'bigdata/communities/com-dblp.ungraph.txt.gz',
        'com-Amazon':'bigdata/communities/com-amazon.ungraph.txt.gz',
        'roadNet-PA':'roadNet-PA.txt.gz',
        'web-Google':'web-Google.txt.gz'}
SEED=20261031+6100

def candidates(n,edges,M,seed):
    """uniform_candidates semantics, with a sorted key array instead of a tuple set."""
    keys=np.sort(edges[:,0]*n+edges[:,1]);rng=np.random.default_rng(seed);C=set()
    while len(C)<M:
        u,v=sorted(map(int,rng.integers(n,size=2)))
        i=np.searchsorted(keys,u*n+v)
        if u!=v and not (i<len(keys) and keys[i]==u*n+v):C.add((u,v))
    c=np.asarray(sorted(C),dtype=np.int64)
    return np.ascontiguousarray(c[rng.permutation(M)])

def load(name,offset):
    rel=GRAPHS[name];path=DATA/Path(rel).name;t=perf_counter()
    raw=np.loadtxt(gzip.open(path),dtype=np.int64)
    ids,inv=np.unique(raw,return_inverse=True);edges=np.sort(inv.reshape(-1,2),axis=1)
    edges=np.unique(edges[edges[:,0]!=edges[:,1]],axis=0);n=len(ids)
    C=candidates(n,edges,2000,SEED+offset);A=matrix(n,edges);k=20
    return A,C,dict(name=name,n=n,m=len(edges),M=len(C),k=k,
       upper=float(1+2*((A.diagonal()-1).max()+k)),max_degree=float((A.diagonal()-1).max()),
       alpha=1.,s=.5,candidate_seed=SEED+offset,source=f'https://snap.stanford.edu/data/{rel}',
       source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
       preprocessing='retain all endpoint vertices; undirect; deduplicate; remove self-loops',
       load_seconds=perf_counter()-t,machine=platform.processor(),threads=ub.THREADS,
       cpu=subprocess.run("lscpu | grep 'Model name'",shell=True,capture_output=True,text=True).stdout.strip())

def main():
    p=argparse.ArgumentParser();p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--threads',type=int,default=8);p.add_argument('--graphs',nargs='+',default=list(GRAPHS))
    p.add_argument('--methods',nargs='+',default=['CELF','Inc','Dynamic-Logdet'])
    a=p.parse_args();OUT.mkdir(exist_ok=True);ub.install(a.threads)
    with threadpool_limits(limits=1):
        for name in a.graphs:
            try:
                A,C,meta=load(name,list(GRAPHS).index(name))
                print(f'LOADED {name} n={meta["n"]} m={meta["m"]} dmax={meta["max_degree"]:.0f} {meta["load_seconds"]:.0f}s',flush=True)
                up.work(name,A,C,meta,a.repeats,tuple(a.methods),OUT/f'{name}.json')
                print(f'DONE {name} peak_rss_gb={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.1f}',flush=True)
            except Exception:
                print(f'FAILED {name}\n{traceback.format_exc()}',flush=True)

if __name__=='__main__':main()
