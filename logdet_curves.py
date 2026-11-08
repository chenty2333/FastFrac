"""Candidate-pool and budget sweeps with Dynamic-Logdet included.

All four methods are rerun together on one machine so the curves are comparable.
Same pool, seeds and spectral bound as unified_performance.curves().
"""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
import json,platform,subprocess
import numpy as np
from threadpoolctl import threadpool_limits
import unified_backend as ub
import unified_performance as up
from unified_performance import locked,work,uniform_candidates,ROOT
OUT=ROOT/'fastfrac_unified_results/curves_logdet'
METHODS=('CELF','Inc','Naive','Dynamic-Logdet')
def main():
    ub.install(8);OUT.mkdir(parents=True,exist_ok=True);up.OUT=ROOT/'fastfrac_unified_results'
    A,C,meta=locked('Medium-WS')
    edges=np.array(json.loads((ROOT/'fastfrac_cpu_results/Medium-WS.json').read_text())['base_edges'])
    pool=uniform_candidates(2500,edges,4000,20261031+7000)
    upper=float(1+2*((A.diagonal()-1).max()+60))
    with threadpool_limits(limits=1,user_api='blas'):
        for M in (250,500,1000,2000,4000):
            m=dict(meta,name=f'pool-{M}',M=M,k=20,upper=upper,alpha=1.,s=.5)
            work(m['name'],A,pool[:M],m,5,METHODS,OUT/f'pool-{M}.json')
        for k in (5,10,20,40,60):
            m=dict(meta,name=f'budget-{k}',M=1000,k=k,upper=upper,alpha=1.,s=.5)
            work(m['name'],A,pool[:1000],m,5,METHODS,OUT/f'budget-{k}.json')
    (OUT/'machine.txt').write_text(subprocess.check_output(['lscpu'],text=True)+platform.platform())
if __name__=='__main__':main()
