import json,sys,numpy as np
d=json.load(open(sys.argv[1]));m=json.load(open(sys.argv[2]))
me=[x for x in d['methods'] if x['dataset']=='Heterogeneous-Multiscale'][0]
n=me['metadata']['n'];cand=me['candidates']
L0=np.zeros((n,n))
for u,v,w in me['base_edges']: L0[u,u]+=w;L0[v,v]+=w;L0[u,v]-=w;L0[v,u]-=w
ds={(r['s'],r['method']):r for r in d['downstream'] if r['dataset']=='Heterogeneous-Multiscale'}
def LS(sel,wt=1.):
    M=L0.copy()
    for e in sel:
        u,v=cand[e];M[u,u]+=wt;M[v,v]+=wt;M[u,v]-=wt;M[v,u]-=wt
    return M
e0=np.linalg.eigvalsh(L0)
A=ds[(0.1,'FastFrac')];B=ds[(0.85,'FastFrac')]
eA=np.linalg.eigvalsh(LS(A['timing']['selected']));eB=np.linalg.eigvalsh(LS(B['timing']['selected']))
H=lambda e,t:np.exp(-t*e).sum()
for t,k in((.05,'0.05'),(.2,'0.2'),(1,'1.0')):
    print(t,H(e0,t)-H(eA,t),A['heat_trace_reduction'][k],H(e0,t)-H(eB,t),B['heat_trace_reduction'][k])
ts=np.logspace(-3,2,1001)
rel=np.array([100*((H(e0,t)-H(eB,t))/(H(e0,t)-H(eA,t))-1) for t in ts])
absA=np.array([H(e0,t)-H(eA,t) for t in ts]);absB=np.array([H(e0,t)-H(eB,t) for t in ts])
cr=[ts[i] for i in range(len(ts)-1) if rel[i]*rel[i+1]<0];print('cross',cr)
for t in(.001,.01,.05,.1,.2,.3,.5,.7,1,2,5,10,30,100):
    i=np.argmin(abs(ts-t));print(t,round(rel[i],2),round(absA[i],4),round(absB[i],4))
print('min',ts[rel.argmin()],rel.min(),'max',ts[rel.argmax()],rel.max())
np.save('heatcurve.npy',np.c_[ts,rel,absA,absB])
pts=[(100*(g['methods']['s0.85']['heat']['0.2']/g['methods']['Logdet']['heat']['0.2']-1),100*(g['methods']['s0.85']['heat']['1.0']/g['methods']['Logdet']['heat']['1.0']-1)) for g in m['hetero']]
print(np.round(pts,1).tolist())
print(m.get('hetero_generator_reproduces_recorded_graph'))
print(A['allocation']['block_pair_counts'],B['allocation']['block_pair_counts'])
