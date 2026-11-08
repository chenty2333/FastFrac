"""CPU/OpenMP batching of unchanged independent shifted Lanczos RHS solves."""
from pathlib import Path
import ctypes
import inspect
from time import perf_counter
import numpy as np
import fastfrac_cpu_benchmark as ff

THREADS = 8
DEADLINE = 0.
LIB = None
ORIGINAL_SCALAR = ff.scalar_ms_batch
ORIGINAL_CELF_SOURCE = inspect.getsource(ff.run_celf)

def install(threads=8):
    global LIB, THREADS
    THREADS=int(threads)
    LIB=ctypes.CDLL(str(Path(__file__).with_name('librevision_scalar.so')))
    p=ctypes.c_void_p
    LIB.scalar_batch.argtypes=[ctypes.c_int,p,p,p,ctypes.c_int,p,ctypes.c_int,p,
                              ctypes.c_double,ctypes.c_int,ctypes.c_int,ctypes.c_double,p,p,p]
    LIB.scalar_batch.restype=ctypes.c_int
    ff.scalar_ms_batch=scalar_ms_batch
    # An equal upper bound must be refined, not indefinitely reinserted with
    # the same heap key. Keep the locked historical source file unchanged.
    source=ORIGINAL_CELF_SOURCE.replace('elif upper<=competitor:',
                                        'elif upper<competitor:')
    exec(compile(source, '<revision CELF strict-tie fix>', 'exec'), ff.__dict__)

def scalar_ms_batch(A,edges,shifts,tol,stats,batch_size=48):
    if DEADLINE and perf_counter()>=DEADLINE:raise TimeoutError('native scalar deadline')
    A=A.tocsr(); rows=np.asarray(A.indptr,dtype=np.int32);cols=np.asarray(A.indices,dtype=np.int32)
    data=np.asarray(A.data,dtype=np.float64);edges=np.ascontiguousarray(edges,dtype=np.int64)
    shifts=np.ascontiguousarray(shifts,dtype=np.float64);M=len(edges);q=len(shifts)
    out=np.zeros((M,q));rr=np.zeros_like(out);it=np.zeros(M,dtype=np.int32)
    p=lambda a:ctypes.c_void_p(a.ctypes.data)
    code=LIB.scalar_batch(A.shape[0],p(rows),p(cols),p(data),M,p(edges),q,p(shifts),
                          tol,THREADS,min(16,max(1,M)),DEADLINE,p(out),p(rr),p(it))
    stats.calls+=M;stats.scalar_calls+=M;stats.krylov_spmv+=int(it.sum())
    stats.sparse_kernel_calls+=int(sum(max(it[i:i+16],default=0) for i in range(0,M,16)))
    if code==2:raise TimeoutError('native scalar deadline')
    if code:raise ArithmeticError(f'native scalar Lanczos code={code}')
    if rr.size:stats.max_residual_proxy=max(stats.max_residual_proxy,float(rr.max()/np.sqrt(2.)))
    lo,hi=ff.intervals(out,rr,shifts)
    return out,lo,hi

def traced(method,A,C,k,shifts,weights,tol=1e-7,alpha=1.):
    """Trace accepted rounds without modifying candidate evaluation logic."""
    old_stats,old_add=ff.Stats,ff.add_edge
    holders=[];counts=[];previous=[len(C)]
    def make_stats():
        x=old_stats();holders.append(x);return x
    def add(A,e,alpha=1.):
        ans=old_add(A,e,alpha)
        current=holders[0].calls
        counts.append(current-previous[0]);previous[0]=current
        return ans
    ff.Stats=make_stats;ff.add_edge=add
    try:
        call={'CELF':ff.run_celf,'Inc':ff.run_inc,'Naive':ff.run_naive}[method]
        if method=='Naive':result=call(A,C,k,shifts,weights,tol,verbose=False,alpha=alpha)
        else:result=call(A,C,k,shifts,weights,tol,alpha=alpha)
        result['online_rhs_per_round']=counts
        assert sum(counts)==result['online_calls']
        return result
    finally:ff.Stats=old_stats;ff.add_edge=old_add
