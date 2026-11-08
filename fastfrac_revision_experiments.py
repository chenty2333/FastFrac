"""Pure-CPU, hard-budget revision experiments. Does not modify the paper.

Usage: python fastfrac_revision_experiments.py --seconds 90 --threads 8
All deadlines, completed/partial states, and measured/extrapolated times are
recorded. Paired CELF/Inc measurements share one freshly timed offline index;
each reported total is that measured index time plus its measured online time.
"""
import time
PROCESS_START=time.perf_counter()
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse,json,gzip,hashlib,subprocess,heapq,platform,signal
from dataclasses import asdict
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
import fastfrac_cpu_benchmark as ff
import revision_backend as backend
from revision_structure_opt import run_structure,run_opt,_original_sbm,_matrix
from revision_adaptive import run_adaptive

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'fastfrac_revision_results'

def write(name,data):
    ff.write_json(OUT/name,data)

def matrix(n,edges,weights=None):
    e=np.asarray(edges,dtype=np.int64);u,v=e.T
    w=np.ones(len(e)) if weights is None else np.asarray(weights)
    row=np.column_stack((u,v,u,v)).ravel();col=np.column_stack((u,v,v,u)).ravel()
    val=np.column_stack((w,w,-w,-w)).ravel()
    return (sparse.eye(n,format='csr')+sparse.coo_matrix((val,(row,col)),shape=(n,n)).tocsr()).tocsr()

def locked(name):
    d=json.loads((ROOT/'fastfrac_cpu_results'/f'{name}.json').read_text())
    return matrix(d['metadata']['n'],d['base_edges']),np.asarray(d['candidates'],dtype=np.int64),d['metadata']

def uniform_candidates(n,edges,M,seed):
    existing=set(map(tuple,np.sort(np.asarray(edges),axis=1)))
    rng=np.random.default_rng(seed);C=set()
    while len(C)<M:
        u,v=sorted(map(int,rng.integers(n,size=2)))
        if u!=v and (u,v) not in existing:C.add((u,v))
    # Random fixed order makes candidate pools nested without degree/ID bias.
    c=np.asarray(sorted(C),dtype=np.int64)
    return np.ascontiguousarray(c[rng.permutation(M)])

def snapshot(A,C,shifts,tol=1e-7):
    st=ff.Stats();start=time.perf_counter()
    R,l,u=ff.scalar_ms_batch(A,C,shifts,tol,st)
    return dict(R=R,lo=l,hi=u,stats=asdict(st),seconds=time.perf_counter()-start)

def from_index(method,A,C,k,shifts,weights,index,alpha=1.):
    original=ff.scalar_ms_batch;used=False
    def read_index(B,edges,ts,tol,st,batch_size=48):
        nonlocal used
        if not used:
            used=True
            assert np.array_equal(edges,C) and np.array_equal(ts,shifts)
            for key,value in index['stats'].items():setattr(st,key,value)
            return index['R'].copy(),index['lo'].copy(),index['hi'].copy()
        return original(B,edges,ts,tol,st,batch_size)
    ff.scalar_ms_batch=read_index
    try:r=backend.traced(method,A,C,k,shifts,weights,1e-7,alpha)
    finally:ff.scalar_ms_batch=original
    online=r['seconds'];r['online_seconds']=online;r['init_seconds']=index['seconds']
    r['seconds']=index['seconds']+online
    r['timing_scope']='measured shared offline initialization + measured online query including table copy and heap setup'
    r['shared_initialization']=True
    return r

def pair(A,C,k,upper,s=.5,alpha=1.,order=('CELF','Inc')):
    shifts,weights,eta=ff.rule(s,12,upper)
    index=snapshot(A,C,shifts)
    runs={}
    for method in order:runs[method]=from_index(method,A,C,k,shifts,weights,index,alpha)
    return dict(status='complete',n=A.shape[0],M=len(C),k=k,s=s,alpha=alpha,q=12,
                upper=upper,quadrature_bound=eta,init_seconds=index['seconds'],
                same_ordered_sequence=runs['CELF']['selected']==runs['Inc']['selected'],runs=runs)

def group1_and_2(deadline):
    backend.DEADLINE=deadline
    result=dict(repetitions=[],real_world=[],scaling=[],pool_curve=[],budget_curve=[],traces=[],status='complete')
    def checkpoint():write('performance_and_traces.json',result)
    try:
        # Five paired fresh indices, alternating query order; no concurrent timings.
        for name in ('Small-BA','Medium-WS','Large-PowerCluster'):
            A,C,meta=locked(name);rows=[]
            item={'name':name,'metadata':meta,'repetitions':rows,'status':'running'}
            result['repetitions'].append(item)
            for r in range(5):
                if time.perf_counter()>deadline:raise TimeoutError('performance deadline')
                ans=pair(A,C,meta['k'],meta['upper'],order=('CELF','Inc') if r%2==0 else ('Inc','CELF'))
                rows.append(ans)
                print(f'REP {name} {r+1}/5 {ans["runs"]["CELF"]["seconds"]:.4f}/{ans["runs"]["Inc"]["seconds"]:.4f}',flush=True)
                if name=='Large-PowerCluster' and r==0:
                    result['traces'].append({'name':name,'s':.5,'k':meta['k'],
                        'CELF':ans['runs']['CELF']['online_rhs_per_round'],
                        'Inc':ans['runs']['Inc']['online_rhs_per_round']})
                checkpoint()
            item['status']='complete'
            item['summary']={}
            for method in ('CELF','Inc'):
                values=np.array([r['runs'][method]['seconds'] for r in rows])
                on=np.array([r['runs'][method]['online_seconds'] for r in rows])
                item['summary'][method]={'mean':float(values.mean()),'std':float(values.std(ddof=1)),
                     'online_mean':float(on.mean()),'online_std':float(on.std(ddof=1))}
        # Original multiscale graph, exact saved topology/candidates.
        G,C,labels,meta=_original_sbm();A=_matrix(G,50.)
        for s in (.1,.5):
            ans=pair(A,C,15,meta['upper'],s,50.)
            result['traces'].append({'name':'Multiscale-SBM','s':s,'k':15,
                'CELF':ans['runs']['CELF']['online_rhs_per_round'],
                'Inc':ans['runs']['Inc']['online_rhs_per_round'],'runs':ans['runs']})
            checkpoint()
        # Real graphs retain every vertex, including small components.
        for offset,name in enumerate(('email-Eu-core','ca-GrQc')):
            raw=np.loadtxt(gzip.open(ROOT/'fastfrac_realworld_data'/f'{name}.txt.gz'),dtype=np.int64)
            ids,inv=np.unique(raw,return_inverse=True);ed=np.sort(inv.reshape(-1,2),axis=1)
            ed=np.unique(ed[ed[:,0]!=ed[:,1]],axis=0);n=len(ids)
            A=matrix(n,ed);C=uniform_candidates(n,ed,2000,20261031+6000+offset)
            upper=float(1+2*((A.diagonal()-1).max()+20))
            ans=pair(A,C,20,upper);ans.update(name=name,m=len(ed),
                 source=f'https://snap.stanford.edu/data/{name}.html',
                 raw_records=len(raw),raw_self_loops=int((raw[:,0]==raw[:,1]).sum()),
                 preprocessing='all endpoint vertices; undirected; remove loops and duplicate edges',
                 candidate_seed=20261031+6000+offset,
                 dataset_sha256=hashlib.sha256((ROOT/'fastfrac_realworld_data'/f'{name}.txt.gz').read_bytes()).hexdigest())
            result['real_world'].append(ans);checkpoint()
            print(f'REAL {name} {ans["runs"]["CELF"]["seconds"]:.3f}/{ans["runs"]["Inc"]["seconds"]:.3f}',flush=True)
        # Requested M and k curves use a fixed q-rule valid up to K=60.
        A,C0,meta=locked('Medium-WS');edges=np.asarray(json.loads((ROOT/'fastfrac_cpu_results/Medium-WS.json').read_text())['base_edges'])
        C=uniform_candidates(2500,edges,4000,20261031+7000)
        upper=float(1+2*((A.diagonal()-1).max()+60))
        for M in (250,500,1000,2000,4000):
            ans=pair(A,C[:M],20,upper);result['pool_curve'].append(ans);checkpoint()
        shifts,w,eta=ff.rule(.5,12,upper);index=snapshot(A,C[:1000],shifts)
        for k in (5,10,20,40,60):
            rows={m:from_index(m,A,C[:1000],k,shifts,w,index) for m in ('CELF','Inc')}
            result['budget_curve'].append({'M':1000,'k':k,'n':2500,'runs':rows,'same_ordered_sequence':rows['CELF']['selected']==rows['Inc']['selected']});checkpoint()
        # Fresh synthetic BA family, never labeled a real-world network.
        for n in (1000,5000,10000,25000,50000):
            if time.perf_counter()>deadline:raise TimeoutError('scaling deadline')
            G=ff.nx.barabasi_albert_graph(n,3,seed=20261031+n)
            ed=np.array(sorted(tuple(sorted(e)) for e in G.edges()),dtype=np.int64)
            A=matrix(n,ed);C=uniform_candidates(n,ed,2000,20261031+n+101)
            upper=float(1+2*((A.diagonal()-1).max()+20))
            shifts,w,eta=ff.rule(.5,12,upper);index=snapshot(A,C,shifts)
            entry={'n':n,'m':len(ed),'M':2000,'k':20,'q':12,'s':.5,'status':'running',
                 'generator':'synthetic Barabasi-Albert(n,3)','seed':20261031+n,'candidate_seed':20261031+n+101,'runs':{}}
            result['scaling'].append(entry);checkpoint()
            # The first Naive round is exactly this evaluated full index plus
            # measured score selection/update. Full-loop time is extrapolated.
            if n>=25000:
                start=time.perf_counter();winner=int(np.argmax(ff.scores(index['R'],w)))
                _=ff.add_edge(A,C[winner]);choice_s=time.perf_counter()-start
                first_round=index['seconds']+choice_s
                entry['naive']={'status':'TLE/full loop not run under budget','measured_first_round_seconds':first_round,
                    'measured_first_round_rhs':2000,'extrapolated_total_seconds':first_round*(20*2000-20*19/2)/2000,
                    'extrapolation':'T1*(kM-k(k-1)/2)/M; assumes unchanged per-candidate solve cost'}
            for method in ('CELF','Inc'):
                entry['runs'][method]=from_index(method,A,C,20,shifts,w,index);checkpoint()
            entry['same_ordered_sequence']=entry['runs']['CELF']['selected']==entry['runs']['Inc']['selected']
            entry['status']='complete';checkpoint()
            print(f'SCALE {n} {entry["runs"]["CELF"]["seconds"]:.3f}/{entry["runs"]["Inc"]["seconds"]:.3f}',flush=True)
    except TimeoutError as error:
        result['status']='TLE';result['error']=str(error);checkpoint()
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seconds',type=float,default=90);parser.add_argument('--threads',type=int,default=8)
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    start=PROCESS_START;deadline=start+args.seconds-1
    def hard_deadline(signum, frame):
        raise TimeoutError('global wall-clock deadline, including Python heap loops')
    signal.signal(signal.SIGALRM,hard_deadline)
    signal.setitimer(signal.ITIMER_REAL,max(.01,deadline-time.perf_counter()))
    record={'requested_seconds':args.seconds,'cpu_threads':args.threads,'blas_threads':1,
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':ff.scipy.__version__,
                           'networkx':ff.nx.__version__,'platform':platform.platform()},
            'timing_contract':'paired measurements: one freshly timed offline index reused by the two methods; each total adds the same measured initialization to independently measured online query time; five indices for five repetitions',
            'celf_tie_fix':'equal interval upper bounds enter refinement; only strict upper<competitor is reinserted without refinement',
            'hard_deadline':'SIGALRM plus native-kernel monotonic deadlines',
            'status':'running','groups':{}}
    write('manifest.json',record)
    compile_start=time.perf_counter()
    for src,lib in [('revision_scalar_kernel.cpp','librevision_scalar.so'),('revision_slq_kernel.cpp','librevision_slq.so')]:
        subprocess.run(['g++','-O3','-march=native','-std=c++17','-fopenmp','-shared','-fPIC',str(ROOT/src),'-o',str(ROOT/lib)],check=True,timeout=20)
    record['compilation_seconds']=time.perf_counter()-compile_start
    backend.install(args.threads);backend.DEADLINE=deadline
    with threadpool_limits(limits=1):
        # Independent finite-probe validation of the SLQ tridiagonal kernel.
        from revision_eval_routes import slq_trace_family
        rng=np.random.default_rng(914);X=rng.normal(size=(8,8));B=X@X.T+np.eye(8)
        Z=rng.choice(np.array([-1.,1.]),size=(4,8));vals,cts,status=slq_trace_family(sparse.csr_matrix(B),np.array([[-1,-1]],dtype=np.int64),Z,8,.5,deadline,args.threads)
        eigen,V=np.linalg.eigh(B);truth=float(np.mean(np.sum((Z@V)**2*np.sqrt(eigen),axis=1)))
        record['slq_kernel_finite_probe_validation_relative_error']=float(abs(vals[0]-truth)/truth)
        assert status==0 and record['slq_kernel_finite_probe_validation_relative_error']<1e-10
        A,C,meta=locked('Small-BA');t,w,_=ff.rule(.5,12,meta['upper'])
        r1=backend.ORIGINAL_SCALAR(A,C[:16],t,1e-7,ff.Stats())[0]
        r2=ff.scalar_ms_batch(A,C[:16],t,1e-7,ff.Stats())[0]
        discrepancy=float(np.max(np.abs(r1-r2)/r1));record['backend_validation_max_relative_error']=discrepancy
        if discrepancy>1e-9:raise AssertionError('native kernel disagrees with original')
        print('START VALIDATED',discrepancy,'elapsed',time.perf_counter()-start,flush=True)
        for name,call,budget in [('structure',run_structure,8),('global_optimum',run_opt,6),('adaptive',run_adaptive,7)]:
            t0=time.perf_counter();stage_deadline=min(deadline,t0+budget);backend.DEADLINE=stage_deadline
            record['groups'][name]=call(stage_deadline,OUT)
            print('DONE',name,time.perf_counter()-t0,'elapsed',time.perf_counter()-start,flush=True)
            write('manifest.json',record)
        # Reserve 25 seconds for alternative-oracle direct comparisons.
        backend.DEADLINE=deadline
        record['groups']['performance']=group1_and_2(max(time.perf_counter(),deadline-25))
        print('DONE PERFORMANCE elapsed',time.perf_counter()-start,flush=True)
        from revision_eval_routes import run_routes
        backend.DEADLINE=deadline
        record['groups']['routes']=run_routes(deadline,OUT)
    record['compute_seconds']=time.perf_counter()-start
    record['status']='finished within deadline' if record['compute_seconds']<=args.seconds else 'deadline overrun'
    write('manifest.json',record)
    signal.setitimer(signal.ITIMER_REAL,0)
    print('FINISHED',record['compute_seconds'],record['status'],flush=True)

if __name__=='__main__':
    try:main()
    except TimeoutError as error:
        signal.setitimer(signal.ITIMER_REAL,0)
        path=OUT/'manifest.json'
        data=json.loads(path.read_text()) if path.exists() else {}
        data.update(status='TLE: global wall-clock deadline',error=str(error),
                    compute_seconds=time.perf_counter()-PROCESS_START)
        write('manifest.json',data)
        print('GLOBAL TLE',data['compute_seconds'],str(error),flush=True)
