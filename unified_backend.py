"""Unified CPU backend: identical CSR/Lanczos/LDL kernel for all deterministic routes.

Scalar and raw-projected modes use scalar short recurrences. Full-vector mode
uses the same recurrence but retains q n-vector directions/solutions and checks
true residuals. Logical SpMV counts exclude already-converged padded columns;
executed_padded_spmv records those additional arithmetic lanes explicitly.
"""
from pathlib import Path
import ctypes
import inspect
import subprocess
from time import perf_counter
import numpy as np
import fastfrac_cpu_benchmark as ff

THREADS=8
DEADLINE=0.
LIB=None
MAXITER=2400
BLOCKSIZE=16
_SOURCE_CELF=inspect.getsource(ff.run_celf)
_COUNTERS={}

def reset_counters():
    global _COUNTERS
    _COUNTERS=dict(rhs=0,logical_spmv=0,executed_padded_spmv=0,residual_spmv=0,
                   iterations=[],modes=[],calls=[])

def get_counters():
    return {k:list(v) if isinstance(v,list) else v for k,v in _COUNTERS.items()}

reset_counters()

def compile_library():
    root=Path(__file__).resolve().parent
    subprocess.run(['g++','-O3','-march=native','-std=c++17','-fopenmp','-shared','-fPIC',
                    str(root/'unified_kernel.cpp'),'-o',str(root/'librevision_unified.so')],check=True)

def install(threads=8):
    global THREADS,LIB
    THREADS=int(threads)
    root=Path(__file__).resolve().parent
    if not (root/'librevision_unified.so').exists():compile_library()
    LIB=ctypes.CDLL(str(root/'librevision_unified.so'))
    p=ctypes.c_void_p;i=ctypes.c_int;d=ctypes.c_double
    LIB.unified_batch.argtypes=[i,p,p,p,i,p,i,p,d,i,i,i,i,d,p,p,p,p,p,p,p]
    LIB.unified_batch.restype=i
    LIB.unified_spmv_benchmark.argtypes=[i,p,p,p,i,i,p]
    LIB.unified_spmv_benchmark.restype=d
    ff.scalar_ms_batch=scalar_ms_batch
    ff.vector_ms=vector_ms
    source=_SOURCE_CELF.replace('elif upper<=competitor:','elif upper<competitor:')
    exec(compile(source,'<unified strict-tie CELF>','exec'),ff.__dict__)

def _batch(A,edges,shifts,tol,stats,mode,batch_size=None):
    if LIB is None:install(THREADS)
    if DEADLINE and perf_counter()>=DEADLINE:raise TimeoutError('unified backend deadline')
    A=A.tocsr();row=np.ascontiguousarray(A.indptr,dtype=np.int32)
    col=np.ascontiguousarray(A.indices,dtype=np.int32);data=np.ascontiguousarray(A.data,dtype=np.float64)
    edges=np.ascontiguousarray(edges,dtype=np.int64).reshape(-1,2)
    shifts=np.ascontiguousarray(shifts,dtype=np.float64);M=len(edges);q=len(shifts)
    B=min(BLOCKSIZE,max(1,M)) if batch_size is None else min(BLOCKSIZE,max(1,int(batch_size)))
    blocks=(M+B-1)//B
    out=np.zeros((M,q));norm=np.zeros_like(out);res=np.zeros_like(out)
    its=np.zeros(M,dtype=np.int32);padded=np.zeros(blocks,dtype=np.int64);checks=np.zeros(blocks,dtype=np.int64)
    X=np.zeros((A.shape[0],q)) if mode==3 else np.empty(0)
    p=lambda a:ctypes.c_void_p(a.ctypes.data)
    code=LIB.unified_batch(A.shape[0],p(row),p(col),p(data),M,p(edges),q,p(shifts),float(tol),THREADS,B,
                           mode,min(max(4*A.shape[0],100),MAXITER),DEADLINE,p(out),p(norm),p(res),p(its),p(padded),p(checks),p(X))
    stats.calls+=M
    if mode>=2:stats.vector_calls+=M
    else:stats.scalar_calls+=M
    stats.krylov_spmv+=int(its.sum());stats.residual_spmv+=int(checks.sum())
    stats.sparse_kernel_calls+=int(sum(max(its[a:a+B],default=0) for a in range(0,M,B)))+int(checks.sum()/max(q,1))
    _COUNTERS['rhs']+=M;_COUNTERS['logical_spmv']+=int(its.sum())
    _COUNTERS['executed_padded_spmv']+=int(padded.sum());_COUNTERS['residual_spmv']+=int(checks.sum())
    _COUNTERS['iterations'].extend(its.tolist());_COUNTERS['modes'].append(mode)
    _COUNTERS['calls'].append(dict(rhs=M,mode=mode,logical_spmv=int(its.sum()),
                       padded_spmv=int(padded.sum()),residual_spmv=int(checks.sum())))
    if code==2:raise TimeoutError('unified backend deadline')
    if code:raise ArithmeticError(f'unified Lanczos code={code}, max_iterations={its.max(initial=0)}, tol={tol}')
    if res.size:
        if mode>=2:stats.max_true_residual=max(stats.max_true_residual,float(res.max()/np.sqrt(2.)))
        else:stats.max_residual_proxy=max(stats.max_residual_proxy,float(res.max()/np.sqrt(2.)))
    return out,norm,res,X

def scalar_ms_batch(A,edges,shifts,tol,stats,batch_size=48):
    r,_,rr,_=_batch(A,edges,shifts,tol,stats,0,batch_size)
    lo,hi=ff.intervals(r,rr,shifts)
    return r,lo,hi

def raw_ms_batch(A,edges,shifts,tol,stats,mode='proj',batch_size=48):
    """Return (r, ||x||², residual), using identical quadrature-independent solves."""
    code={'proj':1,'full':2}[mode]
    r,norm,rr,_=_batch(A,edges,shifts,tol,stats,code,batch_size)
    return r,norm,rr

def vector_ms(A,edge,shifts,tol,stats):
    r,_,rr,X=_batch(A,np.asarray(edge).reshape(1,2),shifts,tol,stats,3,1)
    lo,hi=ff.intervals(r[0],rr[0],shifts)
    return r[0],lo,hi,X,rr[0]

def measure_spmv(A,repeats=100,threads=None):
    """Single-vector CSR timing, 3 warmups excluded; counts exactly nnz multiplies/call."""
    if LIB is None:install(THREADS)
    A=A.tocsr();row=np.ascontiguousarray(A.indptr,dtype=np.int32)
    col=np.ascontiguousarray(A.indices,dtype=np.int32);data=np.ascontiguousarray(A.data,dtype=np.float64)
    check=np.zeros(1);p=lambda a:ctypes.c_void_p(a.ctypes.data)
    elapsed=LIB.unified_spmv_benchmark(A.shape[0],p(row),p(col),p(data),int(repeats),
                                      THREADS if threads is None else int(threads),p(check))
    return dict(repetitions=int(repeats),seconds=float(elapsed),seconds_per_spmv=float(elapsed/repeats),
                nnz=int(A.nnz),multiply_add_pairs_per_spmv=int(A.nnz),checksum=float(check[0]))

