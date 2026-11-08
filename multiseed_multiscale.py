"""Ten-seed replication of the multiscale selection experiment.

Seeds are fixed in advance and every seed is reported; there is no selection.
The heterogeneous generator is the same code as revision_structure_opt with its
seed exposed. The original SBM script is not in the package, so its generator is
rebuilt from the recorded metadata (blocks, weak bridges, candidate definition).
"""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
from itertools import combinations
from pathlib import Path
import json
import numpy as np
from scipy.linalg import eigvalsh
from threadpoolctl import threadpool_limits
import fastfrac_cpu_benchmark as ff
import unified_backend as backend
from revision_structure_opt import _matrix,_structural_metrics,_stable_gain

ROOT=Path(__file__).resolve().parent
SEEDS=range(10);S=(.10,.50,.85);TAUS=(.05,.2,1.);ALPHA=50.;K=15;Q=12

def sbm(seed):
    H=ff.nx.stochastic_block_model([100]*4,(np.eye(4)*.3).tolist(),seed=seed)
    G=ff.nx.Graph();G.add_nodes_from(range(400))
    G.add_weighted_edges_from((int(u),int(v),1.) for u,v in H.edges())
    G.add_weighted_edges_from([(0,100,.02),(200,300,.02)])
    labels=np.repeat(np.arange(4),100);rng=np.random.default_rng(seed+101)
    deg=np.asarray([G.degree(i) for i in range(400)]);low,high=[],[]
    for b in range(4):
        ids=np.arange(b*100,(b+1)*100);o=ids[np.lexsort((ids,deg[ids]))];low.append(o[:25]);high.append(o[-25:])
    A=sorted((int(u),int(v)) for a,c in combinations(range(4),2) for u in low[a] for v in low[c] if not G.has_edge(int(u),int(v)))
    B=sorted((int(u),int(v)) for b in range(4) for u,v in combinations(sorted(map(int,high[b])),2) if not G.has_edge(u,v))
    pick=[A[i] for i in rng.choice(len(A),75,replace=False)]+[B[i] for i in rng.choice(len(B),75,replace=False)]
    return G,np.asarray(sorted(pick),dtype=np.int64),labels

def hetero(seed):
    rng=np.random.default_rng(seed+1);G=ff.nx.Graph();G.add_nodes_from(range(400))
    for b,scale in enumerate((.2,.7,2.,6.)):
        H=ff.nx.barabasi_albert_graph(100,3,seed=seed+10*b)
        for u,v in sorted(H.edges()):
            G.add_edge(b*100+u,b*100+v,weight=float(scale*np.exp(rng.normal(0.,.35))))
    G.add_weighted_edges_from(((0,100,.002),(100,200,.02),(200,300,.2)))
    labels=np.repeat(np.arange(4),100)
    wd=np.asarray([G.degree(i,weight='weight') for i in range(400)]);low,high=[],[]
    for b in range(4):
        ids=np.arange(b*100,(b+1)*100);o=ids[np.lexsort((ids,wd[ids]))];low.append(o[:25]);high.append(o[-25:])
    crng=np.random.default_rng(seed+101);cand=set()
    for idx,(a,c) in enumerate(combinations(range(4),2)):
        quota=13 if idx<3 else 12
        pools=[sorted((int(u),int(v)) for u in vs[a] for v in vs[c] if not G.has_edge(int(u),int(v))) for vs in (low,high)]
        for pool,size in zip(pools,(quota//2,quota-quota//2)):
            for i in crng.choice(len(pool),size,replace=False):cand.add(pool[int(i)])
    for b in range(4):
        ids=sorted(map(int,high[b]));pool=[(u,v) for u,v in combinations(ids,2) if not G.has_edge(u,v)]
        for i in crng.choice(len(pool),19 if b<3 else 18,replace=False):cand.add(pool[int(i)])
    return G,np.asarray(sorted(cand),dtype=np.int64),labels

def evaluate(G,A0,C,sel,labels,lam0,lamL):
    B=A0.toarray()
    for i in sel:
        u,v=C[i];B[u,u]+=ALPHA;B[v,v]+=ALPHA;B[u,v]-=ALPHA;B[v,u]-=ALPHA
    la=eigvalsh(B,check_finite=False);ls=np.maximum((la-1)/ALPHA,0.)
    m=_structural_metrics(G,C,sel,labels)
    return dict(selected=list(map(int,sel)),gain={str(s):_stable_gain(lam0,la,s) for s in S},
        heat={str(t):float(np.sum(np.exp(-t*lamL)*(-np.expm1(-t*(ls-lamL))))) for t in TAUS},
        structure={k:v for k,v in m.items() if k!='per_edge'},
        cross=int(sum(labels[C[i,0]]!=labels[C[i,1]] for i in sel)))

def run_instance(G,C,labels):
    A0=_matrix(G,ALPHA);n=len(G);L=ff.nx.laplacian_matrix(G,nodelist=range(n),weight='weight').astype(float).toarray()
    lamL=np.maximum(eigvalsh(L,check_finite=False),0.);lam0=1+ALPHA*lamL
    upper=float(1+2*ALPHA*(max(dict(G.degree(weight='weight')).values())+K));out=dict(m=G.number_of_edges(),methods={})
    ev=lambda sel:evaluate(G,A0,C,sel,labels,lam0,lamL)
    ld=ff.run_celf(A0,C,K,np.array([0.]),np.array([1.]),1e-7,alpha=ALPHA);out['methods']['Logdet']=ev(ld['selected'])
    st=ff.Stats();rr,_,_=ff.scalar_ms_batch(A0,C,np.array([0.]),1e-7,st)
    out['methods']['Static']=ev(np.lexsort((np.arange(len(C)),-rr[:,0]))[:K].tolist())
    deg=np.asarray([G.degree(u,weight='weight')+G.degree(v,weight='weight') for u,v in C])
    out['methods']['Degree']=ev(np.lexsort((np.arange(len(C)),deg))[:K].tolist())
    for s in S:
        sh,w,eta=ff.rule(s,Q,upper)
        c=ff.run_celf(A0,C,K,sh,w,1e-7,alpha=ALPHA);i=ff.run_inc(A0,C,K,sh,w,1e-7,alpha=ALPHA)
        tsh,tw,_=ff.rule(s,54,upper);chk=ff.run_celf(A0,C,K,tsh,tw,1e-10,alpha=ALPHA)
        row=ev(c['selected']);row.update(eta=eta,overlap_logdet=len(set(c['selected'])&set(ld['selected'])),
            celf_seconds=c['seconds'],inc_seconds=i['seconds'],celf_calls=c['calls'],inc_calls=i['calls'],
            init_calls=c['init_calls'],same_celf_inc=c['selected']==i['selected'],same_q54=chk['selected']==c['selected'],same_set_q54=set(chk['selected'])==set(c['selected']),q54_gain=ev(chk['selected'])['gain'][str(s)])
        out['methods'][f's{s}']=row
    return out

def main():
    backend.install(8);out=dict(alpha=ALPHA,k=K,q=Q,seeds=dict(sbm=[20270001+i for i in SEEDS],hetero=[20271001+i for i in SEEDS]),sbm=[],hetero=[])
    G,C,_=hetero(20261031+4701)
    ref=json.loads((ROOT/'fastfrac_unified_results/science_scan_downstream.json').read_text())['methods'][1]
    out['hetero_generator_reproduces_recorded_graph']=(C.tolist()==ref['candidates'])
    print('hetero generator check',out['hetero_generator_reproduces_recorded_graph'],flush=True)
    with threadpool_limits(limits=1,user_api='blas'):
        for name,fn in (('sbm',sbm),('hetero',hetero)):
            for seed in out['seeds'][name]:
                G,C,labels=fn(seed);r=run_instance(G,C,labels);r['seed']=seed;out[name].append(r)
                print(name,seed,'done',flush=True)
                ff.write_json(ROOT/'fastfrac_unified_results/multiseed_multiscale.json',out)
if __name__=='__main__':main()
