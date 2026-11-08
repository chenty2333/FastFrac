import json,sys,numpy as np
d=json.load(open(sys.argv[1]));m=json.load(open(sys.argv[2]))
me=[x for x in d['methods'] if x['dataset']=='Heterogeneous-Multiscale'][0]
ds={(r['s'],r['method']):r for r in d['downstream'] if r['dataset']=='Heterogeneous-Multiscale'}
A=ds[(0.1,'FastFrac')];B=ds[(0.85,'FastFrac')];LG=ds[(None,'Dynamic-Logdet')]
assert set(A['timing']['selected'])==set(LG['timing']['selected'])
ca=A['allocation']['block_pair_counts'];cb=B['allocation']['block_pair_counts']
cand=me['candidates'];C=me['communities']
SA,SB=set(A['timing']['selected']),set(B['timing']['selected'])
def pc(S): 
    o={}
    for e in S:
        u,v=cand[e];k=tuple(sorted((C[u],C[v])));o[k]=o.get(k,0)+1
    return o
sh,xa,xb=pc(SA&SB),pc(SA-SB),pc(SB-SA);print(sh,xa,xb,file=sys.stderr)
L=[];w=L.append
w(r'\documentclass{article}\usepackage{ffdesign}\pagestyle{empty}\begin{document}\begin{ffcrop}%')
w(r'\begin{tikzpicture}[x=1pt,y=1pt,font=\ffseven,text=ffink]')
w(r'\path[use as bounding box] (0,-142) rectangle (252,44);')
scales=me['metadata']['intra_weight_scales'];br=me['metadata']['weak_bridges']
cx=[42,102,162,228];R=9;tones=[10,24,44,72]
PW=.42
# arcs
def arc(i,j,n,sign,col):
    x1,x2=cx[i],cx[j];h=7+19*((x2-x1)/186)**.9
    yb=sign*(R-1)
    w(r'\draw[%s,line width=%.2fpt] (%.1f,%.1f) .. controls (%.1f,%.1f) and (%.1f,%.1f) .. (%.1f,%.1f);'%(col,PW*n,x1+2,yb,x1+(x2-x1)*.2,sign*(R+h*1.33),x2-(x2-x1)*.2,sign*(R+h*1.33),x2,yb))
    return (x1+x2)/2,.25*yb+.75*sign*(R+h*1.33)
def loop(i,n,sign,col,sz):
    x=cx[i];c=.707
    w(r'\draw[%s,line width=%.2fpt] (%.2f,%.2f) circle (%.2fpt);'%(col,PW*n,x-(R+sz*.75)*c,sign*(R+sz*.75)*c,sz))
gray='ffink!32'
labels=[]
for sign,xs,col,t in((1,xa,'sLow','sLow'),(-1,xb,'sHigh','sHigh!75!black')):
    for (i,j),n in sorted(sh.items()):
        if i!=j:
            a,b=arc(i,j,n,sign,gray)
            labels.append(r'\node[fill=white,inner sep=.5pt,font=\ffsix,text=ffmid] at (%.1f,%.1f) {%d};'%(a,b,n))
    for (i,j),n in sorted(xs.items()):
        if i!=j:
            a,b=arc(i,j,n,sign,col)
            labels.append(r'\node[fill=white,inner sep=.5pt,font=\ffseven\bfseries,text=%s] at (%.1f,%.1f) {%d};'%(t,a,b,n))
for l in labels: w(l)
# intrablock loops: 1 shared on both; 3 exclusive at s=.85
loop(0,xb[(0,0)],-1,'sHigh',6.5)
loop(0,sh[(0,0)],1,gray,3.2);loop(0,sh[(0,0)],-1,gray,3.2)
w(r'\node[inner sep=.5pt,font=\ffseven\bfseries,text=sHigh!75!black] at (19,%.1f) {%d};'%(-R-1,xb[(0,0)]))
# bridges and discs
for u,v,wt in br:
    i,j=C[u],C[v]
    w(r'\draw[ffmid,line width=.4pt,dash pattern=on 1.2pt off 1pt] (%.1f,0)--(%.1f,0);'%(cx[i]+R,cx[j]-R))
    w(r'\node[fill=white,inner sep=.4pt,font=\ffsix,text=ffmid] at (%.1f,0) {%s};'%((cx[i]+cx[j])/2,str(wt).lstrip('0')))
for i,sc in enumerate(scales):
    w(r'\fill[ffink!%d] (%.1f,0) circle (%dpt);'%(tones[i],cx[i],R))
    w(r'\node[inner sep=0pt,font=\ffsix,text=%s] at (%.1f,0) {$\times$%s};'%('white' if tones[i]>40 else 'ffink',cx[i],str(sc).rstrip('0').rstrip('.')))
w(r'\node[anchor=north west,inner sep=0pt,align=left,text=sLow] at (0,44) {\ffhead $s$ = 0.10\\[-.5pt]\ffsix\mdseries same set as log-det};')
w(r'\node[anchor=south west,inner sep=0pt,align=left,text=sHigh!75!black] at (150,-30) {\ffhead $s$ = 0.85\\[-.5pt]\ffsix\mdseries 12 shared, 3 swapped};')
w(r'\node[anchor=north east,inner sep=0pt,align=right,text=ffmid,font=\ffsix] at (252,-38) {edge counts real; discs schematic};')
# ---- curve ----
T=np.load('heatcurve.npy');T=T[(T[:,0]>=.01)&(T[:,0]<=10)]
XL,XR,YT,YB=30,246,-58,-122
vt,vb=12,-20
X=lambda t:XL+(np.log10(t)+2)/3*(XR-XL);Y=lambda v:YB+(v-vb)/(vt-vb)*(YT-YB)
w(r'\draw[ffrule,line width=.4pt] (0,-50)--(252,-50);')
for v in(-20,-10,10):
    w(r'\draw[ffhair,line width=.4pt] (%.1f,%.2f)--(%.1f,%.2f);'%(XL,Y(v),XR,Y(v)))
for v in(-20,-10,0,10):
    w(r'\node[anchor=east,inner sep=2pt,font=\ffsix,text=ffmid] at (%.1f,%.2f) {%s};'%(XL,Y(v),('%+d\\%%'%v) if v else '0'))
sub=T[::6]
pos=sub[sub[:,1]>=0];neg=sub[sub[:,1]<0]
# exact crossing by linear interp on the fine grid
i=np.where(T[:-1,1]*T[1:,1]<0)[0][0];tc=T[i,0];print('cross',tc,file=sys.stderr)
w(r'\fill[sHigh!28] (%.2f,%.2f) -- %s -- (%.2f,%.2f) -- cycle;'%(X(pos[0,0]),Y(0),' -- '.join('(%.2f,%.2f)'%(X(a),Y(b)) for a,b in pos[:,:2]),X(tc),Y(0)))
w(r'\fill[sLow!20] (%.2f,%.2f) -- %s -- (%.2f,%.2f) -- cycle;'%(X(tc),Y(0),' -- '.join('(%.2f,%.2f)'%(X(a),Y(b)) for a,b in neg[:,:2]),X(neg[-1,0]),Y(0)))
w(r'\draw[ffink,line width=.5pt] (%.1f,%.2f)--(%.1f,%.2f);'%(XL,Y(0),XR,Y(0)))
w(r'\draw[ffink,line width=1pt,line join=round] %s;'%' -- '.join('(%.2f,%.2f)'%(X(a),Y(b)) for a,b in sub[:,:2]))
for tau in('0.05','0.2','1.0'):
    t=float(tau)
    for g in m['hetero']:
        v=100*(g['methods']['s0.85']['heat'][tau]/g['methods']['Logdet']['heat'][tau]-1)
        w(r'\draw[ffink,line width=.4pt] (%.2f,%.2f)--(%.2f,%.2f);'%(X(t)-2.4,Y(v),X(t)+2.4,Y(v)))
    r=100*(B['heat_trace_reduction'][tau]/A['heat_trace_reduction'][tau]-1)
    w(r'\fill[white,draw=ffink,line width=.5pt] (%.2f,%.2f) ++(0,1.8) -- ++(1.8,-1.8) -- ++(-1.8,-1.8) -- ++(-1.8,1.8) -- cycle;'%(X(t),Y(r)))
for t,l in((.01,'0.01'),(.05,'.05'),(.2,'.2'),(1,'1'),(10,'10')):
    w(r'\draw[ffink,line width=.4pt] (%.2f,%.1f)--(%.2f,%.1f);'%(X(t),YB,X(t),YB-2))
    w(r'\node[anchor=north,inner sep=1pt,font=\ffsix] at (%.2f,%.1f) {%s};'%(X(t),YB-2,l))
w(r'\draw[ffink,line width=.4pt] (%.1f,%.1f)--(%.1f,%.1f);'%(XL,YB,XR,YB))
w(r'\node[anchor=north,inner sep=1pt,font=\ffsix,text=ffmid] at (%.1f,%.1f) {diffusion time $\tau$};'%(X(3.2),YB-2))
w(r'\node[anchor=south,inner sep=1pt,font=\ffsix,text=ffink] at (%.2f,%.2f) {$\tau$ = %.2f};'%(X(tc)+4,Y(0)+.5,tc))
w(r'\node[anchor=south west,inner sep=1pt,font=\ffsix,text=sHigh!75!black] at (%.1f,%.1f) {$s$ = 0.85 removes more heat};'%(XL+1,Y(10)-1))
w(r'\node[anchor=north east,inner sep=1pt,font=\ffsix,text=sLow] at (%.1f,%.1f) {log-det removes more};'%(XR,Y(0)-14))
w(r'\node[anchor=south west,inner sep=0pt,font=\ffsix,text=ffmid] at (0,-142) {$\Delta\mathcal{H}(\tau)$ vs.\ log-det. Line, $\diamond$: graph above, exact spectra. Dashes: 10 new graphs};')
w(r'\end{tikzpicture}\end{ffcrop}\end{document}')
open('p4_opening.tex','w').write('\n'.join(L))
