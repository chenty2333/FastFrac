"""Unified C++/OpenMP evaluator ablations; no numerical backend mixing.

All optimizers use identical graph/candidate bytes, OpenMP thread cap, BLAS=1,
q=12 and tol=1e-7. Each measured repetition starts from A0. Audits are untimed.
Fixed CRN probes are shared across candidates and rounds, and between greedy
and CELF; differences between those two directly test lazy pruning validity.
"""
from pathlib import Path
from time import perf_counter
from dataclasses import asdict
import ctypes, heapq, json, subprocess
import numpy as np
from scipy import sparse
from scipy.linalg import eigvalsh, eigh
from threadpoolctl import threadpool_limits
import fastfrac_cpu_benchmark as ff

ROOT=Path(__file__).resolve().parent
SEED=20261031+739
THREADS=8
LIB=None

def compile_slq():
    global LIB
    source=ROOT/'unified_slq_kernel.cpp';lib=ROOT/'libunified_slq.so'
    if not lib.exists() or lib.stat().st_mtime<source.stat().st_mtime:
        subprocess.run(['g++','-O3','-march=native','-std=c++17','-fopenmp','-shared','-fPIC',str(source),'-o',str(lib)],check=True)
    LIB=ctypes.CDLL(str(lib));fn=LIB.unified_slq_family
    p=ctypes.c_void_p
    fn.argtypes=[ctypes.c_int,p,p,p,p,ctypes.c_int,ctypes.c_int,ctypes.c_double,p,ctypes.c_int,p,p,ctypes.c_int,ctypes.c_int]
    fn.restype=ctypes.c_int

def slq_family(A,edges,probes,h=15,s=.5):
    if LIB is None:compile_slq()
    A=A.tocsr();ptr=np.ascontiguousarray(A.indptr,dtype=np.int64);col=np.ascontiguousarray(A.indices,dtype=np.int32)
    values=np.ascontiguousarray(A.data,dtype=np.float64);edges=np.ascontiguousarray(edges,dtype=np.int64);probes=np.ascontiguousarray(probes,dtype=np.float64)
    out=np.empty(len(edges));counts=np.zeros(4,dtype=np.int64);p=lambda a:ctypes.c_void_p(a.ctypes.data)
    status=LIB.unified_slq_family(A.shape[0],p(ptr),p(col),p(values),p(probes),len(probes),h,s,p(edges),len(edges),p(out),p(counts),THREADS,8)
    if status or not np.all(np.isfinite(out)):raise ArithmeticError('SLQ tridiagonal evaluation failed')
    return out,counts

def validate_slq():
    rng=np.random.default_rng(3914);X=rng.normal(size=(8,8));A=X@X.T+np.eye(8)
    Z=rng.choice(np.array([-1.,1.]),size=(32,8));vals,ct=slq_family(sparse.csr_matrix(A),np.array([[-1,-1],[0,1]],dtype=np.int64),Z,h=8)
    truth=[]
    for edge in (None,(0,1)):
        B=A.copy()
        if edge is not None:
            b=np.zeros(8);b[0]=1;b[1]=-1;B+=np.outer(b,b)
        lam,V=eigh(B);truth.append(float(np.mean(np.sum((Z@V)**2*np.sqrt(lam),axis=1))))
    err=float(np.max(np.abs(vals-truth)/truth));assert err<1e-11,(vals,truth)
    return {'finite_probe_exact_eigendecomposition_relative_error':err,'probe_rhs':int(ct[1]),'spmv':int(ct[2])}

def load_workload(name):
    d=json.loads((ROOT/'fastfrac_cpu_results'/f'{name}.json').read_text());e=np.asarray(d['base_edges'],dtype=np.int64);C=np.asarray(d['candidates'],dtype=np.int64);n=d['metadata']['n'];u,v=e.T
    row=np.column_stack((u,v,u,v)).ravel();col=np.column_stack((u,v,v,u)).ravel();val=np.tile([1.,1.,-1.,-1.],(len(e),1)).ravel()
    A=(sparse.eye(n,format='csr')+sparse.coo_matrix((val,(row,col)),shape=(n,n))).tocsr()
    return A,C,d

def ibp_route(A,C,k,t,w,tol):
    original_scalar,original_vector,original_add=ff.scalar_ms_batch,ff.vector_ms,ff.add_edge
    edgeid={tuple(map(int,e)):i for i,e in enumerate(C)};records=[];selected=[]
    def scalar(B,edges,shifts,localtol,stats,*args,**kwargs):
        R,lo,hi=original_scalar(B,edges,shifts,localtol,stats,*args,**kwargs)
        records.append({'round':len(selected),'ids':np.array([edgeid[tuple(map(int,e))] for e in edges]),'values':ff.scores(R,w),'upper':ff.scores(hi,w)})
        return R,lo,hi
    def vector(B,edge,shifts,localtol,stats,*args,**kwargs):
        ans=original_vector(B,edge,shifts,localtol,stats,*args,**kwargs)
        records.append({'round':len(selected),'ids':np.array([edgeid[tuple(map(int,edge))]]),'values':np.array([ff.scores(ans[0],w)]),'upper':np.array([ff.scores(ans[2],w)])})
        return ans
    def add(B,edge,*args,**kwargs):
        ans=original_add(B,edge,*args,**kwargs);selected.append(edgeid[tuple(map(int,edge))]);return ans
    ff.scalar_ms_batch,ff.vector_ms,ff.add_edge=scalar,vector,add
    start=perf_counter()
    try:result=ff.run_celf(A,C,k,t,w,tol)
    finally:ff.scalar_ms_batch,ff.vector_ms,ff.add_edge=original_scalar,original_vector,original_add
    result['seconds']=perf_counter()-start;result['name']='FastFrac-CELF (IBP-Scalar)'
    return result,records

def raw_route(A0,C,k,t,w,tol,mode,s=.5):
    import unified_backend as backend
    start=perf_counter();A=A0.copy();st=ff.Stats();chosen=[];gains=[];records=[]
    def evaluate(ids):
        R,n2,rr=backend.raw_ms_batch(A,C[ids],t,tol,st,mode=mode)
        values=(n2/(1.+R))@(w*t/s)
        records.append({'round':len(chosen),'ids':np.asarray(ids),'values':values.copy(),'upper':values.copy()})
        return values
    values=evaluate(np.arange(len(C)));initial=perf_counter()-start;initcalls=st.calls
    heap=[(-float(value),i,0) for i,value in enumerate(values)];heapq.heapify(heap)
    while len(chosen)<k:
        neg,i,stamp=heapq.heappop(heap)
        if stamp!=len(chosen):
            val=float(evaluate(np.array([i]))[0]);heapq.heappush(heap,(-val,i,len(chosen)));continue
        chosen.append(i);gains.append(-neg);A=ff.add_edge(A,C[i])
    return dict(name=f'Raw-Loewner-{ "Proj" if mode=="proj" else "FullVec"}',selected=chosen,gains=gains,seconds=perf_counter()-start,init_seconds=initial,init_calls=initcalls,online_calls=st.calls-initcalls,**asdict(st)),records

def slq_route(A0,C,k,p,lazy=False,h=15,s=.5):
    start=perf_counter();rng=np.random.default_rng(SEED);Z=rng.choice(np.array([-1.,1.]),size=(p,A0.shape[0]))
    A=A0.copy();chosen=[];gains=[];records=[];alive=np.ones(len(C),bool);counts=np.zeros(4,dtype=np.int64)
    val,ct=slq_family(A,np.vstack([[-1,-1],C]),Z,h,s);counts+=ct;base=float(val[0]);marg=val[1:]-base
    records.append({'round':0,'ids':np.arange(len(C)),'values':marg.copy(),'upper':marg.copy()})
    initseconds=perf_counter()-start;initcounts=counts.copy()
    if lazy:
        heap=[(-float(g),i,0,float(val[i+1])) for i,g in enumerate(marg)];heapq.heapify(heap)
        while len(chosen)<k:
            neg,i,stamp,trialtrace=heapq.heappop(heap)
            if stamp!=len(chosen):
                v,ct=slq_family(A,C[i:i+1],Z,h,s);counts+=ct;gain=float(v[0]-base)
                records.append({'round':len(chosen),'ids':np.array([i]),'values':np.array([gain]),'upper':np.array([gain])})
                heapq.heappush(heap,(-gain,i,len(chosen),float(v[0])));continue
            chosen.append(i);gains.append(-neg);alive[i]=False;base=trialtrace;A=ff.add_edge(A,C[i])
    else:
        for r in range(k):
            ids=np.flatnonzero(alive)
            if r:
                val,ct=slq_family(A,C[ids],Z,h,s);counts+=ct;marg=val-base
                records.append({'round':r,'ids':ids.copy(),'values':marg.copy(),'upper':marg.copy()})
                currenttraces=val
            else:currenttraces=val[1:]
            loc=int(np.argmax(marg));i=int(ids[loc]);chosen.append(i);gains.append(float(marg[loc]));alive[i]=False;base=float(currenttraces[loc]);A=ff.add_edge(A,C[i])
    return dict(name=f'CRN-SLQ-{ "CELF" if lazy else "Greedy" }-p{p}',selected=chosen,gains=gains,seconds=perf_counter()-start,init_seconds=initseconds,
                trace_evaluations=int(counts[0]),probe_rhs=int(counts[1]),krylov_spmv=int(counts[2]),padded_spmv_equivalents=int(counts[3]),
                init_probe_rhs=int(initcounts[1]),online_probe_rhs=int(counts[1]-initcounts[1]),
                probe_seed=SEED,probes=p,lanczos_steps=h,common_random_numbers=True),records

def _serial_records(records):
    return [{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in rec.items()} for rec in records]

def _audit(A0,C,meta,result,records,cache,s=.5):
    """All evaluated marginals, plus latest cached values on unselected edges.

    For n<=400 use dense spectra at each visited state. For n=2500 use a
    separately computed q>=64,tol=1e-11 reference; final gain is a positive
    path sum with explicit variational/analytic interval, never trace subtraction.
    """
    start=perf_counter();upper=meta['upper'];rho=(upper**.25+1)/(upper**.25-1);qr=max(64,int(np.ceil(np.log(4e13)/(2*np.log(rho)))))
    tr,wr,eta=ff.rule(s,qr,upper);errors=[];bounds={};checked=violated=0;selected=result['selected'];states={}
    for rec in records:states.setdefault(rec['round'],[]).append(rec)
    reference_widths=[]
    for r,recs in states.items():
        state=tuple(selected[:r])
        if state not in cache:
            A=A0.copy()
            for i in state:A=ff.add_edge(A,C[i])
            if A.shape[0]<=400:
                lam,V=eigh(A.toarray(),check_finite=False);comp=(V[C[:,0]]-V[C[:,1]])**2
                values=ff.scores(comp@(1/(lam[:,None]+tr[None,:])),wr);width=None
            else:
                R,lo,hi=ff.scalar_ms_batch(A,C,tr,1e-11,ff.Stats());values=ff.scores(R,wr)
                lower=ff.scores(lo,wr);higher=ff.scores(hi,wr)/(1-eta);width=float(np.max((higher-lower)/values))
            cache[state]={'values':values,'reference_relative_interval_width':width}
        truth=cache[state]['values']
        if cache[state]['reference_relative_interval_width'] is not None:reference_widths.append(cache[state]['reference_relative_interval_width'])
        if r:
            for i in set(range(len(C)))-set(state):
                if i in bounds:
                    checked+=1;violated+=bool(truth[i]>bounds[i]+1e-8*max(truth[i],1e-12))
        for rec in recs:
            ids=np.asarray(rec['ids']);v=np.asarray(rec['values']);u=np.asarray(rec['upper'])
            errors.extend((np.abs(v-truth[ids])/truth[ids]).tolist())
            bounds.update({int(i):float(x) for i,x in zip(ids,u)})
    if A0.shape[0]<=400:
        lam0=eigvalsh(A0.toarray(),check_finite=False);gain=ff.eigen_gain(lam0,A0,C,selected,s)
        interval=None;truthmethod='dense eigvalsh trace difference, float64'
    else:
        audit=ff.reference_gain(A0,C,selected,upper,s_values=(s,),eta_target=1e-12)
        interval=audit['by_s'][str(s)];gain=interval['gain'];truthmethod='independent high-order path quadrature with true-residual interval'
    result.update(true_gain=float(gain),true_gain_method=truthmethod,true_gain_interval=interval,
                  max_marginal_relative_error=float(max(errors)),mean_marginal_relative_error=float(np.mean(errors)),
                  audited_marginals=len(errors),audit_q=qr,audit_eta=eta,
                  max_reference_relative_interval_width=max(reference_widths) if reference_widths else None,
                  cache_points_checked=checked,cache_point_undercoverage_count=violated,
                  cache_point_undercoverage_rate=violated/checked if checked else None,
                  audit_seconds=perf_counter()-start)

def _lazy_first_error(greedy,lazy,greedy_records):
    for r,(g,l) in enumerate(zip(greedy['selected'],lazy['selected'])):
        if g!=l:
            rec=next(x for x in greedy_records if x['round']==r)
            values=dict(zip(map(int,rec['ids']),map(float,rec['values'])))
            return {'common_prefix_rounds':r,'first_divergent_round':r+1,'exhaustive_winner':g,'lazy_winner':l,
                    'exhaustive_winner_estimated_marginal':values[g],
                    'lazy_winner_estimated_marginal':values[l],
                    'estimated_marginal_regret_at_first_wrong_lazy_choice':values[g]-values[l],
                    'strict_wrong_pruning_observed':bool(values[g]>values[l]+1e-10)}
    return {'common_prefix_rounds':len(greedy['selected']),'first_divergent_round':None,'strict_wrong_pruning_observed':False}

def run_routes(output_dir,repeats=5,threads=8,workloads=('Small-BA','Medium-WS')):
    global THREADS
    import unified_backend as backend
    THREADS=int(threads);backend.install(threads);compile_slq()
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);start=perf_counter()
    report={'backend':'C++17 -O3 -march=native -fopenmp','openmp_threads':threads,'blas_threads':1,'repetitions':repeats,
            'timing_contract':'independent end-to-end runs from A0; graph generation, compilation and all reference audits excluded; no concurrent timed benchmarks',
            'SLQ_policy':'fixed Rademacher matrix seed=20261031+739, p32 prefix of p128, h=15; same probes shared by exhaustive and lazy variants',
            'count_policy':'RHS for IBP/Raw means one incidence vector; SLQ probe RHS means one probe for one trace evaluation; these are not interchangeable',
            'validation':validate_slq(),'workloads':[]}
    def checkpoint():ff.write_json(out/'evaluation_routes_unified.json',report)
    with threadpool_limits(limits=1,user_api='blas'):
        for name in workloads:
            A,C,locked=load_workload(name);meta=locked['metadata'];k=meta['k'];t,w,eta=ff.rule(.5,12,meta['upper'])
            data={'dataset':name,'n':A.shape[0],'M':len(C),'k':k,'s':.5,'alpha':1.,'q':12,'tol':1e-7,'metadata':meta,'routes':[]}
            report['workloads'].append(data);checkpoint()
            functions=[('FastFrac-CELF (IBP-Scalar)',lambda:ibp_route(A,C,k,t,w,1e-7)),
                       ('Raw-Loewner-Proj',lambda:raw_route(A,C,k,t,w,1e-7,'proj')),
                       ('Raw-Loewner-FullVec',lambda:raw_route(A,C,k,t,w,1e-7,'full')),
                       ('CRN-SLQ-Greedy-p32',lambda:slq_route(A,C,k,32)),
                       ('CRN-SLQ-Greedy-p128',lambda:slq_route(A,C,k,128)),
                       ('CRN-SLQ-CELF-p128',lambda:slq_route(A,C,k,128,lazy=True))]
            allrecords={};byname={}
            for routename,fn in functions:
                row={'name':routename,'runs':[]};data['routes'].append(row)
                for rep in range(repeats):
                    backend.reset_counters();result,records=fn();result['backend_counters']=backend.get_counters();row['runs'].append(result)
                    if rep==0:allrecords[routename]=records
                    print(f'UNIFIED ROUTE {name} {routename} {rep+1}/{repeats}: {result["seconds"]:.6f} s',flush=True);checkpoint()
                times=np.array([r['seconds'] for r in row['runs']]);row['seconds_mean']=float(times.mean());row['seconds_std']=float(times.std(ddof=1)) if repeats>1 else 0.
                row['same_selection_all_repetitions']=all(x['selected']==row['runs'][0]['selected'] for x in row['runs'])
                byname[routename]=row;checkpoint()
            # Independent audits run after ALL timings for this workload.
            cache={};sh,wref,_=ff.rule(.5,64,meta['upper'])
            reference=ff.run_celf(A,C,k,sh,wref,1e-11)
            data['reference_selection']=reference['selected'];data['reference_selection_method']='q64, tol1e-11 deterministic greedy; dense spectral marginal auditing on Small-BA'
            if name=='Small-BA' and 'Exact-Eig' in locked['runs']:
                data['locked_exact_eig_selection']=locked['runs']['Exact-Eig']['selected']
                data['reference_matches_locked_exact_eig']=reference['selected']==locked['runs']['Exact-Eig']['selected']
            for row in data['routes']:
                representative=row['runs'][0];_audit(A,C,meta,representative,allrecords[row['name']],cache)
                row['audit']={key:value for key,value in representative.items() if key not in ('backend_counters',)}
                row['audit']['topk_set_overlap_count']=len(set(representative['selected'])&set(reference['selected']))
                row['audit']['sequence_match_count']=sum(a==b for a,b in zip(representative['selected'],reference['selected']))
                row['audit']['exact_same_ordered_sequence']=representative['selected']==reference['selected']
                ff.write_json(out/f'{name}_{row["name"].replace(" ","_").replace("(","").replace(")","")}_oracle_trace.json',_serial_records(allrecords[row['name']]))
                checkpoint()
            g=byname['CRN-SLQ-Greedy-p128']['audit'];l=byname['CRN-SLQ-CELF-p128']['audit']
            data['slq_lazy_vs_same_probe_exhaustive']=_lazy_first_error(g,l,allrecords['CRN-SLQ-Greedy-p128'])
            data['slq_lazy_vs_same_probe_exhaustive'].update(set_overlap=len(set(g['selected'])&set(l['selected'])),
                  sequence_match_count=sum(a==b for a,b in zip(g['selected'],l['selected'])),
                  true_gain_exhaustive=g['true_gain'],true_gain_lazy=l['true_gain'],true_gain_loss_exhaustive_minus_lazy=g['true_gain']-l['true_gain'])
            checkpoint()
    report['elapsed_seconds']=perf_counter()-start;report['status']='complete';checkpoint();return report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',default='fastfrac_unified_results');p.add_argument('--repeats',type=int,default=5);p.add_argument('--threads',type=int,default=8);p.add_argument('--workloads',nargs='+',default=['Small-BA','Medium-WS']);a=p.parse_args()
    run_routes(a.output,a.repeats,a.threads,a.workloads)
