"""Reproduce the full unified CPU suite. No wall-clock truncation of results.

python fastfrac_final_unified_experiments.py --threads 8 --repeats 5
BA-50k Naive executes every round in every repetition. BA-100k Naive alone
is a separately labelled first-round projection. Benchmarks never overlap.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
from time import perf_counter
import argparse,fcntl,hashlib,json,platform,subprocess
import numpy as np
import scipy
from threadpoolctl import threadpool_limits
import fastfrac_cpu_benchmark as ff
import unified_backend as backend
import unified_performance as performance
from unified_routes import run_routes,compile_slq
from unified_science import run_science,render_science_markdown

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--threads',type=int,default=8)
    p.add_argument('--repeats',type=int,default=5);p.add_argument('--output',default='fastfrac_unified_results')
    p.add_argument('--groups',nargs='+',default=['routes','science','main','multiscale','curves','scaling'])
    a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    performance.OUT=out;(out/'graphs').mkdir(exist_ok=True)
    with open(out/'cpu_benchmark.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        t0=perf_counter();backend.compile_library();backend.install(a.threads);compile_slq()
        manifest={'status':'running','threads_max':a.threads,'blas_threads':1,'repetitions':a.repeats,
          'backend':'C++17, -O3 -march=native -fopenmp; all deterministic paths share unified_kernel.cpp',
          'parallelism':'independent RHS blocks; a single online RHS does not manufacture eight independent solves',
          'timing_contract':'fresh scalar table for each method/repetition; Phase2 includes heap, copies, all rounds, rankone updates, true residual checks and rechecks',
          'real_graph_policy':'all endpoint vertices, undirected simple graph, loops removed; disconnected components retained',
          'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'networkx':ff.nx.__version__,
          'platform':platform.platform(),'compiler':subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
          'cpu':subprocess.check_output(['lscpu'],text=True),'groups':{},'compilation_seconds':perf_counter()-t0}
        ff.write_json(out/'unified_manifest.json',manifest)
        with threadpool_limits(limits=1,user_api='blas'):
            for group in a.groups:
                t=perf_counter()
                if group=='routes':run_routes(out,a.repeats,a.threads)
                elif group=='science':run_science(out,a.repeats);render_science_markdown(out)
                elif group=='main':performance.main_workloads(a.repeats)
                elif group=='multiscale':performance.multiscale(a.repeats)
                elif group=='curves':performance.curves(a.repeats)
                elif group=='scaling':performance.scaling(a.repeats)
                else:raise ValueError(group)
                manifest['groups'][group]={'status':'complete','seconds':perf_counter()-t}
                ff.write_json(out/'unified_manifest.json',manifest)
        manifest['status']='complete';manifest['elapsed_seconds']=perf_counter()-t0
        manifest['source_sha256']={path.name:hashlib.sha256(path.read_bytes()).hexdigest()
            for path in list(ROOT.glob('unified_*.py'))+list(ROOT.glob('unified_*.cpp'))}
        ff.write_json(out/'unified_manifest.json',manifest)

if __name__=='__main__':main()
