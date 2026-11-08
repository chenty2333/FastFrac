import json,re,numpy as np,sys
R='fastfrac_unified_results/'; G=sys.argv[1]
perf=lambda n:json.load(open(R+f'performance/{n}.json'))
def sig(x,n=3):
    if x==0:return '0'
    from math import floor,log10
    d=max(0,n-1-floor(log10(abs(x))));return f'{x:.{d}f}' if d>0 else f'{round(x,n-1-floor(log10(abs(x)))):.0f}'
com=lambda x:f'{int(round(x)):,}'.replace(',','{,}')
# ---- verify opening-figure claims
sc=json.load(open(R+'science_scan_downstream.json'))
m=sc['methods'][1];C=np.array(m['candidates']);lab=np.array(m['communities'])
D={(d['method'],d['s']):d for d in sc['downstream'] if d['dataset']==m['metadata']['name']} if 0 else None
het=[d for d in sc['downstream'] if d['dataset']=='Heterogeneous-Multiscale']
ld=[d for d in het if d['method']=='Dynamic-Logdet'][0]['timing']['selected'];s85=[d for d in het if d['s']==.85][0]['timing']['selected']
def pairs(sel):
    from collections import Counter
    return Counter(tuple(sorted(map(int,lab[C[i]]))) for i in sel)
print('logdet',pairs(ld));print('s85',pairs(s85));print('overlap',len(set(ld)&set(s85)))
print('swapped out',pairs(set(ld)-set(s85)),'in',pairs(set(s85)-set(ld)))
# ---- Table II
rows=[('Small-BA','Small-BA'),('Medium-WS','Medium-WS'),('Large-PowerCluster','Large-PowerCluster'),('email-Eu-core','email-Eu-core'),('ca-GrQc','ca-GrQc'),('ca-CondMat','ca-CondMat'),('BA-50k','BA-50k'),('BA-100k','BA-100k'),('SBM-4k','Multiscale-SBM-4k')]
out=[];cvs=[]
for name,f in rows:
    d=perf(f);s=d['summary'];me=d['metadata']
    meth=['Naive','CELF','Inc','Dynamic-Logdet']
    init=np.mean([s[x]['init_seconds']['mean'] for x in meth if x in s])
    cell=[];rhs=[]
    for x in meth:
        if x in s and not (x=='Naive' and name=='BA-100k'):
            cell.append(sig(s[x]['online_seconds']['mean']));rhs.append(com(s[x]['online_calls']['mean']))
            cvs.append(s[x]['online_seconds']['std']/s[x]['online_seconds']['mean'])
        else: cell.append('--');rhs.append('--')
    mm=me.get('m',0)
    out.append(f"{name} & {com(me['n'])} & {com(mm)} & {com(me['M'])} & {me['k']} & {sig(init)} & "+' & '.join(cell)+' & '+' & '.join(rhs)+r'\\')
open(G+'/tab2rows.tex','w').write('\n'.join(out));print('online CV max/median',max(cvs),np.median(cvs))
print(open(G+'/tab2rows.tex').read())
# ---- scaling
def ba(meth,key='seconds'):
    o=[]
    for n,f in ((1000,'BA-1k'),(5000,'BA-5k'),(10000,'BA-10k'),(25000,'BA-25k'),(50000,'BA-50k'),(100000,'BA-100k')):
        s=perf(f)['summary']
        if meth=='Naive' and n==100000:continue
        o.append(f'({n},{s[meth][key]["mean"]:.5g})')
    return ' '.join(o)
def cur(pref,xs,meth):
    return ' '.join(f"({x},{json.load(open(R+f'curves_logdet/{pref}-{x}.json'))['summary'][meth]['seconds']['mean']:.5g})" for x in xs)
St=dict(Naive='ffnaive',CELF='ffcelf',Inc='ffinc');St['Dynamic-Logdet']='fflog'
f=[r"""\begin{figure*}[t]
\centering
\ref{leg:scal}\\[1pt]
\begin{tikzpicture}
\begin{groupplot}[group style={group size=3 by 1,horizontal sep=44pt},ffd,xmode=log,ymode=log,scale only axis,width=128pt,height=86pt]
\nextgroupplot[title={(a) Graph size (BA, $M{=}2000$, $k{=}20$)},xtick={1000,10000,100000},xlabel={Vertices $n$},ylabel={Time (s)},ymin=.02,ymax=1000,
legend columns=4,legend to name=leg:scal,legend style={font=\ffseven,draw=none,fill=none,column sep=9pt},legend cell align=left]"""]
for m_,l in (('Naive','Naive'),('CELF','CELF'),('Inc','Inc'),('Dynamic-Logdet','Logdet')):
    f.append(rf"\addplot[{St[m_]}] coordinates {{{ba(m_)}}};"+"\n"+rf"\addlegendentry{{{l}}}")
f.append(rf"\addplot[ffcelf,line width=.5pt,dash pattern=on .8pt off 1.4pt,mark=none,forget plot] coordinates {{{ba('CELF','online_seconds')}}} node[anchor=north,font=\ffsix,text=ffink,pos=.62,yshift=-4pt] {{CELF online only}};")
f.append(r"\nextgroupplot[title={(b) Candidate pool ($n{=}2500$, $k{=}20$)},xtick={250,500,1000,2000,4000},xticklabels={250,500,1k,2k,4k},xlabel={Candidates $M$},ylabel={Total time (s)},ymin=.02,ymax=4]")
for m_ in St:f.append(rf"\addplot[{St[m_]}] coordinates {{{cur('pool',(250,500,1000,2000,4000),m_)}}};")
f.append(r"\nextgroupplot[title={(c) Budget ($n{=}2500$, $M{=}1000$)},xtick={5,10,20,40,60},xticklabels={5,10,20,40,60},xlabel={Budget $k$},ymin=.02,ymax=4]")
for m_ in St:f.append(rf"\addplot[{St[m_]}] coordinates {{{cur('budget',(5,10,20,40,60),m_)}}};")
f.append(r"""\end{groupplot}
\end{tikzpicture}
\caption{Scaling with graph size, candidate pool and budget. Means of five runs. Lazy selection separates CELF, Inc and Logdet from Naive, and the three stay close to each other; initialization dominates at large $n$. Panel (a): EPYC; (b, c): Intel (see Section~\ref{sec:protocol}).}
\label{fig:scalability}
\end{figure*}""")
open(G+'/fig_scal.tex','w').write('\n'.join(f))
# ---- bridge competition
def cum(f,meth,xs=None):
    a=perf(f)['runs'][meth][0]['online_rhs_per_round'];c=np.cumsum(a)
    return ' '.join(f'({i+1},{int(v)})' for i,v in enumerate(c) if v>0),int(c[-1])
g=[r"""\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 1,horizontal sep=22pt},ffd,ymode=log,scale only axis,width=90pt,height=74pt,xlabel={Greedy round},ymin=.8,ymax=700]"""]
for t,f_,ticks,yl in ((r'(a) SBM-400','Multiscale-SBM-s0.1','{1,5,10,15}',r',ylabel={Cumulative online RHS}'),(r'(b) SBM-4k','Multiscale-SBM-4k','{1,5,10,15,20}','')):
    g.append(rf"\nextgroupplot[title={{{t}}},xtick={ticks}{yl}]")
    for me,lb,st,col,sh in (('Dynamic-Logdet','Logdet','fflog','sLow','2pt'),('CELF','CELF','ffcelf','ffink','-3pt'),('Inc','Inc','ffinc','ffinc','0pt')):
        co,e=cum(f_,me)
        g.append(rf"\addplot[{st}] coordinates {{{co}}} node[anchor=west,pos=1,font=\ffsix,text={col},xshift=.5pt,yshift={sh}] {{{e}}};")
g.append(r"""\end{groupplot}
\end{tikzpicture}
\caption{Cumulative online RHS under bridge competition at $s=0.10$ (initialization excluded; final counts at line ends). CELF and Logdet repeatedly refresh comparable candidates, while Inc stays at $k$. Zero-count first rounds are not drawn.}
\label{fig:thrashing}
\end{figure}""")
open(G+'/fig_thrash.tex','w').write('\n'.join(g))
# ---- multiscale figure
ms=json.load(open(R+'multiseed_multiscale.json'))
modal=[d for d in sc['modal_scan'] if d['dataset']=='Multiscale-SBM']
h=[r"""\begin{figure*}[t]
\centering
\begin{tikzpicture}
\begin{axis}[ffd,name=a,scale only axis,width=118pt,height=88pt,xmin=.03,xmax=.97,ymin=0,ymax=100,xtick={.05,.25,.5,.75,.95},xticklabels={0.05,0.25,0.5,0.75,0.95},xlabel={Exponent $s$},ylabel={Low-mode share $P_{\rm low}$ (\%)},title={(a) Spectral concentration (SBM)}]"""]
for al,sh,x,y in ((1,35,.08,6),(10,55,.08,42),(50,78,.1,70),(200,100,.3,88)):
    co=' '.join(f"({d['s']},{100*d['P_low_cross']:.4g})" for d in modal if d['alpha']==al)
    h.append(rf"\addplot[color=ffink!{sh},line width=.9pt,mark=*,mark size=1pt] coordinates {{{co}}};")
    v5=[d['P_low_cross']*100 for d in modal if d['alpha']==al and d['s']==.5][0];ly={1:34,10:49,50:64,200:79}[al]
    h.append(rf"\draw[ffrule,line width=.35pt] (axis cs:.5,{v5:.3g})--(axis cs:.62,{ly});")
    h.append(rf"\node[anchor=west,font=\ffsix,text=ffink,inner sep=1pt] at (axis cs:.62,{ly}) {{$\alpha{{=}}{al}$}};")
h.append(r"\shade[left color=sLow,middle color=sMid,right color=sHigh] (axis cs:.1,104) rectangle (axis cs:.85,106);")
h.append(r"\end{axis}")
order=[('Logdet','Logdet','sLow'),('s0.5','0.50','sMid'),('s0.85','0.85','sHigh')]
def facet(name,at,key,title,ymax,ytick,yl=''):
    o=[rf"\begin{{axis}}[ffd,name={name},at={{({at})}},anchor=west,scale only axis,width=62pt,height=88pt,xmin=.5,xmax=3.5,xtick={{1,2,3}},xticklabels={{Logdet,0.50,0.85}},ymin=0,ymax={ymax},ytick={ytick},title={{{title}}},xlabel={{Selection ($s$)}}{yl}]"]
    for i,(k,l,c) in enumerate(order):
        v=[r['methods'][k]['structure'][key] for r in ms['hetero']]
        for y in v:o.append(rf"\draw[{c},line width=.45pt,opacity=.75] (axis cs:{i+1-.22},{y:.4g})--(axis cs:{i+1+.22},{y:.4g});")
        o.append(rf"\node[diamond,fill=white,draw=ffink,line width=.5pt,inner sep=1.1pt] at (axis cs:{i+1},{np.mean(v):.4g}) {{}};")
    o.append(r"\end{axis}");return o
h+=facet('b','a.east',"mean_shared_two_hop_neighbors",r"(b) Common neighbors",16,'{0,4,8,12,16}')
h[-len(h)+0:]=h[-len(h)+0:]
h=[x.replace('at={(a.east)}','at={(a.east)},xshift=38pt') for x in h]
h+=[x.replace('at={(b.east)}','at={(b.east)},xshift=34pt') for x in facet('c','b.east',"mean_endpoint_weighted_degree_sum",r"(c) Weighted degree",9,'{0,3,6,9}')]
pt=lambda fam,k:' '.join(f"({100*(r['methods'][k]['heat']['1.0']/r['methods']['Logdet']['heat']['1.0']-1):.3g},{100*(r['methods'][k]['heat']['0.2']/r['methods']['Logdet']['heat']['0.2']-1):.3g})" for r in ms[fam])
h.append(r"""\begin{axis}[ffd,name=d,at={(c.east)},xshift=40pt,anchor=west,scale only axis,width=118pt,height=88pt,xmin=-18,xmax=2,ymin=-3,ymax=12,xlabel={Change at $\tau=1$ (\%)},ylabel={Change at $\tau=0.2$ (\%)},title={(d) Ten new graphs vs.\ Logdet},xtick={-15,-10,-5,0},ytick={0,4,8,12},ymajorgrids=false]
\draw[ffink,line width=.4pt] (axis cs:-18,0)--(axis cs:2,0);\draw[ffink,line width=.4pt] (axis cs:0,-3)--(axis cs:0,12);""")
for fam,mk,sz in (('sbm','*',1.5),('hetero','triangle*',2.1)):
    for k,c in (('s0.5','sMid'),('s0.85','sHigh')):
        h.append(rf"\addplot[only marks,mark={mk},mark size={sz}pt,color={c},mark options={{fill={c},draw=white,line width=.2pt}}] coordinates {{{pt(fam,k)}}};")
h.append(r"""\node[anchor=south west,font=\ffsix,inner sep=1pt] at (axis cs:-17.8,-2.9) {\textcolor{sMid}{$\bullet$\,$s{=}0.50$}\ \ \textcolor{sHigh}{$\bullet$\,$s{=}0.85$}\ \ \textcolor{ffmid}{$\bullet$\,SBM\ \,$\blacktriangle$\,hetero.}};
\end{axis}
\end{tikzpicture}
\caption{Exponent effects. (a) Mean share of the four lowest modes in the derivative over initial inter-cluster SBM candidates; the bar marks the color scale for $s$ used throughout. (b, c) Selected-edge structure on ten heterogeneous graphs: each dash is one graph, each diamond the mean. (d) Per-graph heat-trace change relative to Logdet; all 40 points (SBM and heterogeneous, $s=0.50,0.85$) fall in the upper-left quadrant.}
\label{fig:multiscale}
\end{figure*}""")
open(G+'/fig_ms.tex','w').write('\n'.join(h))
n_ok=0;tot=0
for fam in('sbm','hetero'):
  for k in('s0.5','s0.85'):
    for r in ms[fam]:
      a=r['methods'][k]['heat'];b=r['methods']['Logdet']['heat'];tot+=1;n_ok+= (a['0.2']>b['0.2'] and a['1.0']<b['1.0'])
print('upper-left',n_ok,tot);print('xr',[ (min(float(x.split(',')[0][1:]) for x in pt(f,k).split()),max(float(x.split(',')[1][:-1]) for x in pt(f,k).split())) for f in('sbm','hetero') for k in('s0.5','s0.85')])
