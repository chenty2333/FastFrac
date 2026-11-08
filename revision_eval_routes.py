"""Fresh CPU comparisons of the four requested marginal-evaluation routes.

No manuscript files or locked measurements are modified. run_routes takes an
absolute perf_counter deadline. Unfinished greedy runs are marked TLE and never
receive a final k-edge gain. SLQ uses one fixed Rademacher probe matrix across
all candidate sets and all rounds; this is CRN-SLQ, not Hutch++.
"""
from pathlib import Path
from time import perf_counter
from dataclasses import asdict
import ctypes
import heapq
import json
import os
import numpy as np
from scipy import sparse
from scipy.linalg import eigh, eigvalsh
from threadpoolctl import threadpool_limits
import fastfrac_cpu_benchmark as ff


class BudgetExpired(RuntimeError):
    pass


def _check(deadline):
    if perf_counter() >= deadline:
        raise BudgetExpired()


def _load_small():
    # Reuse graph/candidate bytes, not any previous timing or result.
    saved = Path(__file__).with_name('fastfrac_cpu_results')/'Small-BA.json'
    if saved.exists():
        obj=json.loads(saved.read_text())
        edges=np.asarray(obj['base_edges'],dtype=np.int64)
        C=np.asarray(obj['candidates'],dtype=np.int64)
        n=400
        u,v=edges.T
        rr=np.concatenate([u,v,u,v]);cc=np.concatenate([u,v,v,u])
        vv=np.concatenate([np.ones(len(u)),np.ones(len(u)),-np.ones(len(u)),-np.ones(len(u))])
        A=(sparse.eye(n,format='csr')+sparse.coo_matrix((vv,(rr,cc)),shape=(n,n))).tocsr()
        return A,C,obj['metadata']
    A,C,_,_,_,meta=ff.make_graph(0)
    return A,C,meta


def _fast_route(A0,C,k,shifts,weights,tol,deadline):
    """Instrument the existing FastFrac-CELF, including its interval keys."""
    original_scalar,original_vector,original_add=ff.scalar_ms_batch,ff.vector_ms,ff.add_edge
    edgeid={tuple(map(int,e)):i for i,e in enumerate(C)}
    selected=[];records=[];seen_stats=[]
    def scalar(A,edges,shifts,tol,stats,*args,**kwargs):
        _check(deadline)
        R,lo,hi=original_scalar(A,edges,shifts,tol,stats,*args,**kwargs)
        seen_stats[:]=[stats]
        ids=np.array([edgeid[tuple(map(int,e))] for e in edges],dtype=int)
        records.append(dict(round=len(selected),ids=ids,values=ff.scores(R,weights),upper=ff.scores(hi,weights)))
        return R,lo,hi
    def vector(A,edge,shifts,tol,stats,*args,**kwargs):
        _check(deadline)
        out=original_vector(A,edge,shifts,tol,stats,*args,**kwargs)
        seen_stats[:]=[stats]
        records.append(dict(round=len(selected),ids=np.array([edgeid[tuple(map(int,edge))]]),
                            values=np.array([ff.scores(out[0],weights)]),upper=np.array([ff.scores(out[2],weights)])))
        return out
    def add(A,edge,*args,**kwargs):
        result=original_add(A,edge,*args,**kwargs)
        selected.append(edgeid[tuple(map(int,edge))])
        return result
    ff.scalar_ms_batch,ff.vector_ms,ff.add_edge=scalar,vector,add
    start=perf_counter()
    try:
        result=ff.run_celf(A0,C,k,shifts,weights,tol)
        result['status']='complete'
    except (BudgetExpired, TimeoutError):
        result=dict(name='FastFrac-CELF',status='TLE',seconds=perf_counter()-start,
                    selected=selected,**(asdict(seen_stats[0]) if seen_stats else {}))
    finally:
        ff.scalar_ms_batch,ff.vector_ms,ff.add_edge=original_scalar,original_vector,original_add
    return result,records


def _raw_route(A0,C,k,shifts,weights,tol,deadline,s=.5):
    """Direct raw Loewner integral, with q explicit n-vectors per call.

    On the SAME Jacobi nodes as the IBP oracle, the raw summand is
    (W_j/s)*t_j*||x_j||^2/(1+b^T x_j), for alpha=1.
    Unlike the shifted-logdet objective, this discretization is not asserted
    to preserve submodularity; its classical lazy ordering is tested below.
    """
    A=A0.copy();st=ff.Stats();selected=[];gains=[];records=[];heap=[]
    start=perf_counter();init_seconds=None
    def evaluate(i):
        _check(deadline)
        d,lo,hi,X,rr=ff.vector_ms(A,C[i],shifts,tol,st)
        val=float(np.dot(weights*shifts/s,np.sum(X*X,axis=0)/(1+d)))
        records.append(dict(round=len(selected),ids=np.array([i]),values=np.array([val]),upper=np.array([val])))
        return val
    try:
        for i in range(len(C)):
            val=evaluate(i);heap.append((-val,i,0))
        heapq.heapify(heap);init_seconds=perf_counter()-start
        while len(selected)<k:
            _check(deadline)
            neg,i,stamp=heapq.heappop(heap)
            if stamp!=len(selected):
                val=evaluate(i);heapq.heappush(heap,(-val,i,len(selected)))
                continue
            selected.append(i);gains.append(-neg);A=ff.add_edge(A,C[i])
        status='complete'
    except (BudgetExpired, TimeoutError):status='TLE'
    result=dict(name='Raw-Loewner-CELF',status=status,selected=selected,gains=gains,
                seconds=perf_counter()-start,init_seconds=init_seconds,**asdict(st))
    return result,records


def _load_slq_library():
    filename=os.environ.get('FASTFRAC_SLQ_LIBRARY',str(Path(__file__).with_name('librevision_slq.so')))
    library=ctypes.CDLL(filename)
    fn=library.slq_family
    I64=np.ctypeslib.ndpointer(np.int64,flags='C_CONTIGUOUS')
    I32=np.ctypeslib.ndpointer(np.int32,flags='C_CONTIGUOUS')
    F64=np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS')
    fn.argtypes=[ctypes.c_int,I64,I32,F64,F64,ctypes.c_int,ctypes.c_int,
                 ctypes.c_double,I64,ctypes.c_int,F64,I64,ctypes.c_double,ctypes.c_int]
    fn.restype=ctypes.c_int
    return library,fn


def slq_trace_family(A,edges,probes,h,s,deadline,threads=8):
    """Public narrow helper used by lightweight kernel-validation code."""
    _,fn=_load_slq_library()
    A=A.tocsr();ptr=np.ascontiguousarray(A.indptr,dtype=np.int64)
    col=np.ascontiguousarray(A.indices,dtype=np.int32)
    val=np.ascontiguousarray(A.data,dtype=np.float64)
    edges=np.ascontiguousarray(edges,dtype=np.int64)
    probes=np.ascontiguousarray(probes,dtype=np.float64)
    out=np.empty(len(edges));counts=np.zeros(3,dtype=np.int64)
    status=fn(A.shape[0],ptr,col,val,probes,len(probes),h,s,edges,len(edges),
              out,counts,max(0.,deadline-perf_counter()),threads)
    return out,counts,status


def _slq_route(A0,C,k,p,h,s,deadline,threads):
    rng=np.random.default_rng(20261031+739)
    # p=32 is a prefix of p=128, enabling an additional paired comparison.
    probes=rng.choice(np.array([-1.,1.]),size=(p,A0.shape[0]))
    A=A0.copy();selected=[];gains=[];records=[];alive=np.ones(len(C),dtype=bool)
    counts=np.zeros(3,dtype=np.int64);base=None;start=perf_counter();status='complete'
    for round_id in range(k):
        if perf_counter()>=deadline:status='TLE';break
        ids=np.flatnonzero(alive)
        family=C[ids] if base is not None else np.vstack([[-1,-1],C[ids]])
        values,ct,st=slq_trace_family(A,family,probes,h,s,deadline,threads)
        counts+=ct
        if base is None:
            base=float(values[0]);values=values[1:]
        valid=np.isfinite(values)&np.isfinite(base)
        records.append(dict(round=round_id,ids=ids[valid],values=values[valid]-base,upper=values[valid]-base))
        if st or not valid.all():status='TLE';break
        marg=values-base;winner=int(np.argmax(marg));i=int(ids[winner])
        selected.append(i);gains.append(float(marg[winner]));alive[i]=False
        base=float(values[winner]);A=ff.add_edge(A,C[i])
    result=dict(name=f'CRN-SLQ-Greedy-p{p}',status=status,selected=selected,gains=gains,
                seconds=perf_counter()-start,probes=p,lanczos_steps=h,probe_seed=20261031+739,
                trace_evaluations=int(counts[0]),probe_rhs=int(counts[1]),
                krylov_spmv=int(counts[2]),calls=None,shared_crn=True,
                algorithm='fixed-probe stochastic Lanczos quadrature; not Hutch++')
    return result,records


def _exact_route(A0,C,k,s,deadline):
    A=A0.toarray();selected=[];gains=[];records=[];alive=np.ones(len(C),bool)
    start=perf_counter();calls=0;status='complete'
    try:
        _check(deadline);base=float(np.sum(eigvalsh(A,check_finite=False)**s));calls+=1
        for r in range(k):
            ids=np.flatnonzero(alive);values=[];traces=[]
            for i in ids:
                _check(deadline)
                u,v=C[i];B=A.copy()
                B[u,u]+=1;B[v,v]+=1;B[u,v]-=1;B[v,u]-=1
                tr=float(np.sum(eigvalsh(B,check_finite=False,overwrite_a=True)**s));calls+=1
                values.append(tr-base);traces.append(tr)
            values=np.asarray(values)
            records.append(dict(round=r,ids=ids,values=values,upper=values.copy()))
            winner=int(np.argmax(values));i=int(ids[winner]);selected.append(i);gains.append(float(values[winner]))
            base=traces[winner];alive[i]=False;u,v=C[i]
            A[u,u]+=1;A[v,v]+=1;A[u,v]-=1;A[v,u]-=1
    except (BudgetExpired, TimeoutError):status='TLE'
    return dict(name='Exact-Eig',status=status,selected=selected,gains=gains,
                seconds=perf_counter()-start,eigendecompositions=calls,calls=0),records


def _audit(result,records,A0,C,k,shref,wref,refcache,deadline):
    """Independent spectral-resolvent audit, with exact dense spot checks.

    Cache checks use all still-alive edges at START of each round, not only
    candidates refreshed by CELF. For exhaustive SLQ they are shadow-CELF
    checks: SLQ itself does not run lazy pruning.
    """
    start=perf_counter();errors=[];bounds={};previous_values={};checked=violations=0
    increases=increase_checks=0;done_rounds=0;selected=result['selected'];audit_complete=True
    grouped={}
    for rec in records:grouped.setdefault(rec['round'],[]).append(rec)
    try:
        for r in sorted(grouped):
            _check(deadline);state=tuple(selected[:r])
            if state not in refcache:
                A=A0.copy()
                for i in state:A=ff.add_edge(A,C[i])
                lam,V=eigh(A.toarray(),check_finite=False)
                component=(V[C[:,0]]-V[C[:,1]])**2
                rr=component@(1/(lam[:,None]+shref[None,:]))
                refcache[state]={'values':ff.scores(rr,wref),'lam':lam}
            truth=refcache[state]['values']
            live=set(range(len(C)))-set(state)
            if r>0:
                for i in live:
                    if i in bounds:
                        checked+=1
                        violations+=int(truth[i]>bounds[i]+1e-8*max(truth[i],1e-12))
            for rec in grouped[r]:
                for i,val,upper in zip(rec['ids'],rec['values'],rec['upper']):
                    i=int(i);val=float(val);upper=float(upper)
                    errors.append(abs(val-truth[i])/truth[i])
                    if i in previous_values and previous_values[i][0]<r:
                        increase_checks+=1
                        increases+=int(val>previous_values[i][1]+1e-8*max(abs(val),1e-12))
                    previous_values[i]=(r,val)
                    bounds[i]=upper
            done_rounds+=1
        _check(deadline)
        if () not in refcache:
            refcache[()]={'lam':eigvalsh(A0.toarray(),check_finite=False)}
        gain=ff.eigen_gain(refcache[()]['lam'],A0,C,selected,.5)
        result['partial_gain']=gain
        result['true_gain']=gain if len(selected)==k else None
    except (BudgetExpired, TimeoutError):
        audit_complete=False;result['true_gain']=None;result['partial_gain']=None
    result.update(audit_seconds=perf_counter()-start,audit_complete=audit_complete,
                  audited_rounds=done_rounds,audited_marginals=len(errors),
                  max_marginal_relative_error=max(errors) if errors else None,
                  mean_marginal_relative_error=float(np.mean(errors)) if errors else None,
                  stale_upper_bounds_checked=checked,stale_upper_bound_violations=violations,
                  stale_upper_bound_violation_rate=violations/checked if checked else None,
                  reevaluations_checked=increase_checks,stale_score_increases=increases,
                  stale_score_increase_rate=increases/increase_checks if increase_checks else None,
                  completed_rounds=len(selected),target_rounds=k)


def run_routes(deadline,output_dir):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    start=perf_counter();A,C,meta=_load_small();k=10;s=.5
    shifts,weights,eta=ff.rule(s,12,meta['upper'])
    shref,wref,etaref=ff.rule(s,64,meta['upper'])
    threads=int(os.environ.get('FASTFRAC_CPU_THREADS','8'))
    data=dict(workload='Small-BA',n=400,M=len(C),k=k,s=s,alpha=1.,q=12,tol=1e-7,
              metadata=meta,rows=[],audit_reference=dict(q=64,eta=etaref,
              method='Dense eigendecomposition of each current matrix plus 64-point spectral-resolvent quadrature; final G from dense eigenvalues.'),
              bound_violation_definition='At the start of rounds 2..k, exact-reference current gain exceeds the latest cached key by more than 1e-8 times the current gain. All unselected candidates with a cached key are checked. SLQ uses shadow CELF checks, not actual lazy selection.',
              timing_definition='Fresh algorithm wall times exclude separately reported independent auditing; all execution is included in the global wall budget.',
              library_threads=threads)
    refcache={}
    functions=[lambda dl:_fast_route(A,C,k,shifts,weights,1e-7,dl),
               lambda dl:_raw_route(A,C,k,shifts,weights,1e-7,dl),
               lambda dl:_slq_route(A,C,k,32,15,s,dl,threads),
               lambda dl:_slq_route(A,C,k,128,15,s,dl,threads),
               lambda dl:_exact_route(A,C,k,s,dl)]
    names=['FastFrac-CELF','Raw-Loewner-CELF','CRN-SLQ-Greedy-p32','CRN-SLQ-Greedy-p128','Exact-Eig']
    with threadpool_limits(limits=1,user_api='blas'):
        for index,(name,fn) in enumerate(zip(names,functions)):
            remaining=deadline-perf_counter()
            if remaining<=.05:
                data['rows'].append(dict(name=name,status='not_started_budget',true_gain=None,completed_rounds=0))
                continue
            # Preserve a fresh Exact-Eig measurement opportunity. If the module
            # budget is smaller, incomplete runs remain explicitly labeled.
            route_deadline=deadline
            if index in (2,3) and remaining>13:
                route_deadline=deadline-12
            try:
                result,records=fn(route_deadline)
            except (OSError,ArithmeticError) as exc:
                result=dict(name=name,status='error',error=str(exc),selected=[],true_gain=None)
                records=[]
            _audit(result,records,A,C,k,shref,wref,refcache,deadline)
            data['rows'].append(result)
            trace=[{key:(value.tolist() if isinstance(value,np.ndarray) else value) for key,value in rec.items()} for rec in records]
            (out/(name.replace('/','_')+'_oracle_trace.json')).write_text(json.dumps(trace,separators=(',',':')))
            data['elapsed_seconds']=perf_counter()-start
            (out/'evaluation_routes.json').write_text(json.dumps(data,indent=2,allow_nan=False))
            print('ROUTE',name,result.get('status'),result.get('seconds'),result.get('completed_rounds'),flush=True)
    # A small independent spot check validates the spectral-resolvent reference
    # against literal trial eigendecompositions, if the hard budget permits it.
    differences=[]
    if () in refcache and 'values' in refcache[()]:
        lam0=refcache[()]['lam']
        for i in np.linspace(0,len(C)-1,16,dtype=int):
            if perf_counter()>=deadline:break
            exact=ff.eigen_gain(lam0,A,C,[int(i)],s)
            differences.append(abs(refcache[()]['values'][i]-exact)/exact)
    data['audit_reference']['dense_eig_spot_checks']=len(differences)
    data['audit_reference']['max_relative_spot_discrepancy']=max(differences) if differences else None
    data['elapsed_seconds']=perf_counter()-start
    (out/'evaluation_routes.json').write_text(json.dumps(data,indent=2,allow_nan=False))
    return data


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--seconds',type=float,default=30)
    parser.add_argument('--output',default='fastfrac_revision_results/routes')
    args=parser.parse_args();run_routes(perf_counter()+args.seconds,args.output)
