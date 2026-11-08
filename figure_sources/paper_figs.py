"""All FastFrac figures, from recorded results. Usage: python3 paper_figs.py   (needs Chrome and Ghostscript)"""
import sys, json, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOTP = HERE.parent
sys.argv = ['x', str(ROOTP), str(ROOTP / 'figures')]
exec((HERE / 'svg_kit.py').read_text())
sc = json.load(open(R / 'science_scan_downstream.json')); ms = json.load(open(R / 'multiseed_multiscale.json'))
me = [x for x in sc['methods'] if x['dataset'] == 'Heterogeneous-Multiscale'][0]
ds = {(r['s'], r['method']): r for r in sc['downstream'] if r['dataset'] == 'Heterogeneous-Multiscale'}
perf = lambda n: json.load(open(R / f'performance/{n}.json'))
edges, cand, C = me['base_edges'], me['candidates'], me['communities']
SA = set(ds[(.1, 'FastFrac')]['timing']['selected']); SB = set(ds[(.85, 'FastFrac')]['timing']['selected'])
assert SA == set(ds[(None, 'Dynamic-Logdet')]['timing']['selected'])
WHITE = '#FFFFFF'; DORANGE = mix(ORANGE, INK, .2)

def lap(sel=()):
    L = np.zeros((400, 400))
    for u, v, w in edges: L[u, u] += w; L[v, v] += w; L[u, v] -= w; L[v, u] -= w
    for e in sel:
        u, v = cand[e]; L[u, u] += 1; L[v, v] += 1; L[u, v] -= 1; L[v, u] -= 1
    return L

def fig_horizon():
    W = 252; f = Fig('fig_horizon', W, 100)
    e0, eA, eB = (np.linalg.eigvalsh(lap(S)) for S in ((), SA, SB))
    H = lambda e, t: np.exp(-t * e).sum()
    ts = np.logspace(-2, 1, 400); rel = np.array([100 * ((H(e0, t) - H(eB, t)) / (H(e0, t) - H(eA, t)) - 1) for t in ts])
    ax = Axes(f, 27, 5, W - 33, 72, (.01, 10), (-21, 12), xlog=True)
    ax.yticks([-20, -10, 10], ['−20%', '−10%', '+10%']); f.text(ax.x0 - 3, ax.Y(0) + 2.3, '0', 6.8, INK, 'end')
    i = np.where(rel[:-1] * rel[1:] < 0)[0][0]; tc = ts[i] + (ts[i + 1] - ts[i]) * rel[i] / (rel[i] - rel[i + 1])
    pp = [(ax.X(t), ax.Y(v)) for t, v in zip(ts, rel)]; z = ax.Y(0)
    f.area([(ax.X(.01), z)] + [p for t, p in zip(ts, pp) if t < tc] + [(ax.X(tc), z)], mix(WHITE, ORANGE, .42))
    f.area([(ax.X(tc), z)] + [p for t, p in zip(ts, pp) if t > tc] + [(ax.X(10), z)], mix(WHITE, BLUE, .3))
    f.line(ax.x0, z, ax.x0 + ax.w, z, INK, .45); f.poly(pp, INK, 1.)
    f.text(ax.X(.0108), ax.Y(9.3), '*s* = 0.85', 7, DORANGE, 'start', True); f.text(ax.X(.0108), ax.Y(5), 'removes more', 6.6, DORANGE)
    f.text(ax.X(10) - 2, ax.Y(-3.8), 'Logdet removes more', 6.7, BLUE, 'end')
    for tau in ('0.05', '0.2', '1.0'):
        x = ax.X(float(tau))
        v0 = 100 * (ds[(.85, 'FastFrac')]['heat_trace_reduction'][tau] / ds[(.1, 'FastFrac')]['heat_trace_reduction'][tau] - 1)
        vs = [100 * (g['methods']['s0.85']['heat'][tau] / g['methods']['Logdet']['heat'][tau] - 1) for g in ms['hetero']]
        f.line(x, ax.Y(min(vs)), x, ax.Y(max(vs)), INK, .55)
        for v in (min(vs), max(vs)): f.line(x - 1.5, ax.Y(v), x + 1.5, ax.Y(v), INK, .55)
        f.dot(x, ax.Y(v0), 1.6, WHITE, INK, .7)
    ax.xticks([.01, .05, .2, 1, 10], ['0.01', '0.05', '0.2', '1', '10'], 'Diffusion time *τ*')
    f.line(ax.X(tc), z, ax.X(tc), z + 6.4, INK, .4); f.text(ax.X(tc) - 2, z + 7.6, f'*τ* = {tc:.2f}', 6.8, INK, 'end')
    f.h = 96; f.save(); print('horizon cross', tc)

def fig_framework():
    W = 516; f = Fig('fig_framework', W, 128)
    TINT = lambda c: mix(WHITE, BLUE, .1 + .55 * c)
    f.text(0, 7, 'One objective becomes', 8.2, INK, 'start', True); f.text(0, 16, 'one table of scalars', 8.2, INK, 'start', True)
    def eq(y, tag, m):
        f.text(0, y - 8.5, tag, 6.4, MID)
        f.raw(f'<text x="0" y="{y}" font-family="{SERIF}" font-size="8.8" fill="{INK}">{m}</text>')
    I = lambda t: f'<tspan font-style="italic">{t}</tspan>'
    sub = lambda t: f'<tspan baseline-shift="-2" font-size="6" font-style="italic">{t}</tspan>'
    sup = lambda t: f'<tspan baseline-shift="3" font-size="6" font-style="italic">{t}</tspan>'
    eq(42, 'Objective', f'Tr(({I("I")} + {I("α")}{I("L")}{sub("S")})' + f'{sup("s")})')
    eq(70, 'Lemma 1: rank-one marginal as an integral', f'{I("s c")}{sub("s")}&#160;<tspan font-size="11" font-family="\'STIX Two Math\'">∫</tspan> {I("t")}{sup("s−1")}&#160;log(1 + {I("α r")}{sub("e")}({I("t")})) {I("dt")}')
    eq(98, 'Theorem 3: submodular for every *q* ≥ 1', f'{I("g")}{sub("q")}&#160;= Σ{sub("j")}&#160;{I("w")}{sub("j")}&#160;log(1 + {I("α r")}{sub("e")}({I("t")}{sub("j")}))')
    gx, cw, Q = 176, 12.5, 12; gy, ch, Mr = 40, 5.4, 13; gw = cw * Q
    g = lambda u: (u + .02) ** .5 * math.exp(-2.8 * u) * 2.3
    cy = 34; h = 26; us = np.linspace(0, 1, 100); pts = [(gx + gw * u, cy - h * g(u)) for u in us]
    f.area([(gx, cy)] + pts + [(gx + gw, cy)], mix(WHITE, BLUE, .1)); f.poly(pts, BLUE, .9)
    for j in range(Q):
        u = (j + .5) / Q; x = gx + gw * u; y = cy - h * g(u)
        f.line(x, y, x, gy + .5, BLUE, .5); f.dot(x, y, 1.5, BLUE, WHITE, .3)
    for r_ in range(Mr):
        for c in range(Q): f.rect(gx + c * cw, gy + r_ * ch, cw - .7, ch - .7, TINT(1 - c / Q))
    f.text(gx - 3, gy + 4.6, '*M*', 7, INK, 'end')
    f.text(gx + gw, 7, 'one column per quadrature node *t*_{j}', 6.6, BLUE, 'end')
    f.text(gx, gy + Mr * ch + 7.5, 'Offline index: ℐ[*e*, *j*] = *r*_{e}(*t*_{j}; ∅)', 7, INK, 'start', True)
    f.text(gx, gy + Mr * ch + 15.5, '*M* RHS once, *Mq* stored scalars; rebuild if *G*_{0}, *C*, *α*, *s* or the rule changes', 6.4, MID)
    rx = 362; sw, sh = 4, 3.0; sQ, sM = 12, 13
    def mini(y, rows, cc, title, l1, l2, tc):
        for r_ in range(sM):
            for c in range(sQ):
                hit = r_ in rows if rows is not None else True
                f.rect(rx + c * sw, y + r_ * sh, sw - .4, sh - .4, cc if hit else mix(WHITE, BLUE, .1))
        tx = rx + sQ * sw + 8
        f.text(tx, y + 7, title, 8, tc, 'start', True); f.text(tx, y + 16.5, l1, 6.5, INK); f.text(tx, y + 24.5, l2, 6.5, MID)
    mini(8, {0, 1}, INK, 'CELF', 'refresh stale heap tops', '*L* RHS, *M* ≤ *L* ≤ *N*_{k}', INK)
    mini(68, None, mix(WHITE, INC, .78), 'Inc', 'update every row', '*O*(*Mq*) sweep, *k* + *R* RHS', INC)
    f.text(rx, 56.5, 'Rows touched in one round', 6.2, MID)
    f.text(rx, 116.5, 'from one selected-edge solve', 6.2, MID)
    def arw(x1, y1, x2, c=INK):
        f.line(x1, y1, x2 - 3, y1, c, .55); f.area([(x2, y1), (x2 - 3.6, y1 - 1.6), (x2 - 3.6, y1 + 1.6)], c)
    arw(128, 94, gx - 2); arw(gx + gw + 2, 44, rx - 2); arw(gx + gw + 2, 88, rx - 2)
    f.save()

def fig_weights():
    W = 252; f = Fig('fig_weights', W, 118)
    al = me['metadata']['alpha']; lamA = 1 + al * np.maximum(np.linalg.eigvalsh(lap()), 0)
    ax = Axes(f, 27, 12, 92, 76, (1, lamA.max() * 1.2), (9e-5, 1.3), True, True)
    ax.yticks([1, 1e-2, 1e-4], ['1', '10^{−2}', '10^{−4}'])
    xs = np.logspace(0, math.log10(lamA.max()), 60)
    for s_, c, t, tx in ((0, BLUE, '*s*→0', 5), (.5, TEAL, '*s* = 0.5', 12), (.85, ORANGE, '*s* = 0.85', 20)):
        f.poly([(ax.X(x), ax.Y(x ** (s_ - 1))) for x in xs], c, 1.05)
    xl = lamA.max() / 2
    f.text(ax.X(70), ax.Y(70 ** -1) + 7.5, '*s*→0', 6.6, BLUE, 'end')
    f.text(ax.X(4000), ax.Y(4000 ** -.5) + 8.5, '*s* = 0.5', 6.6, TEAL, 'end')
    f.text(ax.X(4000), ax.Y(4000 ** -.15) + 8, '*s* = 0.85', 6.6, ORANGE, 'end')
    yr = ax.y0 + ax.h
    for v in lamA[::2]: f.line(ax.X(v), yr + .5, ax.X(v), yr + 3.5, mix(WHITE, INK, .5), .22)
    f.line(ax.x0, yr, ax.x0 + ax.w, yr, INK, .45)
    for e_ in (0, 2, 4):
        v = 10 ** e_; f.line(ax.X(v), yr + 3.5, ax.X(v), yr + 5.5, INK, .4); f.text(ax.X(v), yr + 12, '1' if e_ == 0 else f'10^{{{e_}}}', 6.8, INK, 'middle')
    f.text(ax.x0 + ax.w / 2, yr + 21, 'Eigenvalue *λ* of *A*', 7, INK, 'middle')
    f.text(0, 7, '(a) Weight *λ*^{s−1}', 7.6, INK, 'start', True)
    modal = [d for d in sc['modal_scan'] if d['dataset'] == 'Multiscale-SBM']
    a = Axes(f, 150, 12, 96, 76, (.03, 1.0), (0, 100))
    a.yticks([0, 25, 50, 75, 100])
    for al_, t in ((1, .35), (10, .55), (50, .75), (200, 1.)):
        rows = sorted((d['s'], 100 * d['P_low_cross']) for d in modal if d['alpha'] == al_)
        P = [(a.X(x), a.Y(y)) for x, y in rows]; c = mix(WHITE, INK, t); f.poly(P, c, .95)
    for al_ in (200, 50, 10, 1):
        rows = sorted((d['s'], 100 * d['P_low_cross']) for d in modal if d['alpha'] == al_)
        v = float(np.interp(.35, [p for p, _ in rows], [q for _, q in rows]))
        f.rect(a.X(.35) - 10.5, a.Y(v) - 3.8, 21, 7.2, WHITE); f.text(a.X(.35), a.Y(v) + 2.2, f'*α* = {al_}', 6.5, INK if al_ >= 50 else MID, 'middle')
    a.xticks([.05, .25, .5, .75, .95], ['.05', '.25', '.5', '.75', '.95'], 'Exponent *s*')
    f.text(a.x0 - 16, 7, '(b) Low-mode share (%)', 7.6, INK, 'start', True)
    f.save()

def fig_incround():
    """Exact-objective lazy greedy on the recorded SBM-400 instance; asserts equality with the recorded run."""
    m0 = sc['methods'][0]; n = 400; al = 50.; s = .1; Cd = m0['candidates']; Mc = len(Cd)
    Lb = np.zeros((n, n))
    for u, v, w in m0['base_edges']: Lb[u, u] += w; Lb[v, v] += w; Lb[u, v] -= w; Lb[v, u] -= w
    A = np.eye(n) + al * Lb
    rec = perf('Multiscale-SBM-s0.1')['runs']['CELF'][0]
    F_ = lambda M: (np.linalg.eigvalsh(M) ** s).sum()
    def add(M, e):
        u, v = Cd[e]; B = M.copy(); B[u, u] += al; B[v, v] += al; B[u, v] -= al; B[v, u] -= al; return B
    F = F_(A); key = np.array([F_(add(A, e)) - F for e in range(Mc)]); stamp = np.zeros(Mc, int); Sel = []; cnt = []; snap = {}
    for r_ in range(15):
        F = F_(A); old = key.copy(); ref = []
        while True:
            e = max((i for i in range(Mc) if i not in Sel), key=lambda i: (key[i], -i))
            if stamp[e] == r_: break
            key[e] = F_(add(A, e)) - F; stamp[e] = r_; ref.append(e)
        snap[r_] = (old.copy(), key.copy(), list(ref), e, list(Sel)); Sel.append(e); cnt.append(len(ref)); A = add(A, e)
    assert cnt == rec['online_rhs_per_round'] and Sel == rec['selected'], (cnt, Sel)
    W = 252; f = Fig('fig_incround', W, 112)
    for pi, (r_, ttl) in enumerate(((1, '(a) Round 2: 4 refreshes'), (2, '(b) Round 3: 72 refreshes'))):
        old, new, ref, win, prior = snap[r_]
        idx = [i for i in np.argsort(-old) if i not in prior][:73]
        wv = new[win]; x0 = 27 + pi * 116
        ax = Axes(f, x0, 12, 104, 66, (0, 75), (-30, 30))
        ax.yticks([-30, -15, 15, 30], ['−30', '−15', '+15', '+30'] if pi == 0 else ['', '', '', ''])
        if pi == 0: f.text(x0 - 3, ax.Y(0) + 2.2, '0', 6.8, INK, 'end'); f.text(x0 - 21, 12 + 33, 'Bound relative to winner (%)', 6.6, INK, 'middle', rot=-90)
        clip = lambda v: float(np.clip(v, -30, 30))
        for j, i in enumerate(idx):
            x = ax.X(j + .5); a_ = 100 * (old[i] / wv - 1); b_ = 100 * (new[i] / wv - 1)
            if i in ref:
                f.line(x, ax.Y(clip(a_)), x, ax.Y(clip(b_)), mix(WHITE, INK, .4), .35)
                f.dot(x, ax.Y(clip(b_)), .85, INK)
            else:
                f.dot(x, ax.Y(clip(a_)), .75, WHITE, MID, .3)
        f.line(x0, ax.Y(0), x0 + 104, ax.Y(0), INK, .5)
        ax.xticks([0, 25, 50, 75], label='Candidates by stale bound')
        f.text(x0, 7, ttl, 7.6, INK, 'start', True)
        f.text(x0 + 104, ax.Y(-30) - 2, 'Inc: 1 RHS, all 150 exact', 6.5, INC, 'end', True)
        print('round', r_ + 1, 'refresh', len(ref), 'range', min(100 * (old[i] / wv - 1) for i in idx), max(100 * (old[i] / wv - 1) for i in idx), 'newmin', min(100 * (new[i] / wv - 1) for i in ref) if ref else None)
    f.h = 104; f.save(); return cnt

def fig_quad():
    W = 252; f = Fig('fig_quad', W, 100)
    ax = Axes(f, 28, 8, 158, 70, (3, 55), (1e-14, 3), ylog=True)
    ax.yticks([1, 1e-4, 1e-8, 1e-12], ['1', '10^{−4}', '10^{−8}', '10^{−12}'])
    f.text(1, 5, 'Relative marginal error', 7, INK)
    xs = [4, 8, 12, 16, 24, 36, 54]
    bSBM = [0.4779773238, 0.098621667371, 0.016583857162, 0.0026903426739, 6.991066123e-05, 2.9228587457e-07, 7.9009959796e-11]
    bBA = [0.035339313871, 0.00032354978599, 2.9106262873e-06, 2.6179585718e-08, 2.1179425106e-12]
    S1 = [0.10389189034, 0.0095692353844, 0.0010759273689, 0.00013231752984, 2.2871942959e-06, 6.2054143133e-09, 8.812211477e-11]
    S5 = [0.081960276454, 0.0068241364835, 0.00077489606403, 9.7198889938e-05, 1.720940902e-06, 4.7387824416e-09, 4.9272272972e-12]
    BA = [0.00034708862845, 2.1059497436e-07, 1.3809993186e-10, 5.8643751218e-13, 5.8601515891e-13, 5.8601515891e-13, 5.8580398228e-13]
    pts = lambda v: [(ax.X(x), ax.Y(y)) for x, y in zip(xs, v)]
    f.poly(pts(bSBM), NAIVE, .6, '2.4 1.5'); f.poly(pts(bBA), NAIVE, .6, '2.4 1.5')
    for v, c, dd in ((S1, BLUE, None), (S5, TEAL, None), (BA, TEAL, '3 1.6')):
        f.poly(pts(v), c, .95, dd)
        for x, y in pts(v): f.dot(x, y, 1.2, c, WHITE, .25)
    ax.xticks(xs, label='Quadrature order *q*')
    lx = ax.X(54) + 4
    f.text(lx, ax.Y(8.8e-11) - 1.5, 'SBM, *s* = 0.10', 6.8, BLUE); f.text(lx, ax.Y(4.9e-12) + 1.5, 'SBM, *s* = 0.50', 6.8, TEAL); f.text(lx, ax.Y(5.9e-13) + 6, 'Small-BA, *s* = 0.50', 6.8, TEAL)
    f.text(ax.X(19), ax.Y(1.5e-2), '*η*_{q}, SBM', 6.8, MID); f.text(ax.X(24.5), ax.Y(4e-11), '*η*_{q}, Small-BA', 6.8, MID)
    f.save()

def fig_scaling():
    W = 252; f = Fig('fig_scaling', W, 112)
    M = [('Naive', NAIVE, .95, None), ('CELF', INK, .95, None), ('Dynamic-Logdet', BLUE, .85, None), ('Inc', INC, .95, '3 1.6')]
    NS = ((1000, 'BA-1k'), (5000, 'BA-5k'), (10000, 'BA-10k'), (25000, 'BA-25k'), (50000, 'BA-50k'), (100000, 'BA-100k'))
    ser = {m: [(n, perf(k)['summary'][m]['seconds']['mean']) for n, k in NS if not (m == 'Naive' and n == 100000)] for m, *_ in M}
    onl = [(n, perf(k)['summary']['CELF']['online_seconds']['mean']) for n, k in NS]
    a = Axes(f, 27, 6, 170, 80, (800, 125000), (.02, 1000), True, True)
    a.yticks([.1, 1, 10, 100, 1000], ['0.1', '1', '10', '10^{2}', '10^{3}'], 'Total time (s)')
    for m, c, w, d in M:
        P = [(a.X(x), a.Y(y)) for x, y in ser[m]]; f.poly(P, c, w, d)
        for x, y in P: f.dot(x, y, 1.2, c, WHITE, .22)
    f.poly([(a.X(p), a.Y(q)) for p, q in onl], INK, .55, '.9 1.4')
    a.xticks([1000, 10000, 100000], ['10^{3}', '10^{4}', '10^{5}'], 'Vertices *n* (BA, *M* = 2000, *k* = 20)')
    f.text(a.X(ser['Naive'][-1][0]) - 2, a.Y(ser['Naive'][-1][1]) - 3.5, 'Naive', 6.8, MID, 'end')
    lx = a.X(100000) + 3; y = a.Y(ser['CELF'][-1][1])
    f.text(lx, y - 1, 'CELF, Inc,', 6.8, INK); f.text(lx, y + 6.5, 'Logdet', 6.8, INK)
    y = a.Y(onl[-1][1]); f.text(lx, y + 2.3, 'CELF online', 6.8, MID)
    f.save()

def fig_rounds():
    W = 252; f = Fig('fig_rounds', W, 164)
    rows = (('Large-PowerCluster', 'Large-PowerCluster', 'ordinary graph, *M* = 3000'), ('Multiscale-SBM-s0.1', 'SBM-400', 'bridge competition, *M* = 150'), ('Multiscale-SBM-4k', 'SBM-4k', 'bridge competition, *M* = 600'))
    y = 4; ymax = 80; hh = 22; x0 = 22; pw = 150
    for key, nm, sub_ in rows:
        d = perf(key); ce = d['runs']['CELF'][0]['online_rhs_per_round']; ic = d['runs']['Inc'][0]['online_rhs_per_round']; s = d['summary']; K = len(ce)
        f.text(0, y + 6, nm, 7.2, INK, 'start', True); f.text(W, y + 6, sub_, 6.3, MID, 'end')
        base = y + 11 + hh; st = pw / 30; bw = st * .78
        for v in (40, 80): f.line(x0, base - hh * v / ymax, x0 + pw, base - hh * v / ymax, HAIR, .4)
        for i, (a_, b_) in enumerate(zip(ce, ic)):
            x = x0 + i * st
            if a_: f.rect(x, base - hh * a_ / ymax, bw, max(hh * a_ / ymax, .55), INK)
            f.rect(x, base + .9, bw, max(hh * b_ / ymax, .9), INC)
        f.line(x0, base + .45, x0 + pw, base + .45, RULE, .3)
        for v in (0, 40, 80): f.text(x0 - 3, base - hh * v / ymax + 2.2, str(v), 6.2, MID, 'end')
        tc, ti = s['CELF']['online_seconds']['mean'], s['Inc']['online_seconds']['mean']; tx = x0 + pw + 8
        f.text(tx, base - 17, f'CELF {sum(ce)} RHS', 6.5, INK); f.text(tx, base - 10, f'{tc:.3g} s online', 6.3, MID)
        f.text(tx, base - 1, f'Inc {sum(ic)} RHS', 6.5, INC); f.text(tx, base + 6, f'{ti:.3g} s online', 6.3, MID)
        y = base + 11
    f.h = y - 3; f.save()

def fig_struct():
    W = 252; f = Fig('fig_struct', W, 106)
    order = [('Logdet', 'Logdet', BLUE), ('s0.5', '0.50', TEAL), ('s0.85', '0.85', ORANGE)]
    def slope(x0, key, title, ymax, ticks):
        ax = Axes(f, x0, 12, 88, 66, (.72, 3.28), (0, ymax)); ax.yticks(ticks)
        vals = [[r['methods'][k]['structure'][key] for k, *_ in order] for r in ms['hetero']]
        for v in vals: f.poly([(ax.X(i + 1), ax.Y(y)) for i, y in enumerate(v)], mix(WHITE, INK, .22), .4)
        for i, (_, _, c) in enumerate(order):
            for v in vals: f.dot(ax.X(i + 1), ax.Y(v[i]), 1.05, c, WHITE, .2)
        mean = np.mean(vals, axis=0); f.poly([(ax.X(i + 1), ax.Y(y)) for i, y in enumerate(mean)], INK, 1.05)
        for i, y in enumerate(mean): f.dot(ax.X(i + 1), ax.Y(y), 1.75, WHITE, INK, .8)
        ax.xticks([1, 2, 3], ['Logdet', '0.50', '0.85'], 'Selection (*s*)'); f.text(x0 - 14, 7, title, 7.6, INK, 'start', True)
    slope(22, 'mean_shared_two_hop_neighbors', '(a) Common neighbors', 16, [0, 4, 8, 12, 16])
    slope(150, 'mean_endpoint_weighted_degree_sum', '(b) Weighted degree', 9, [0, 3, 6, 9])
    f.h = 100; f.save()

def fig_perf():
    W = 516; rows = [('Small-BA', 'Small-BA'), ('Medium-WS', 'Medium-WS'), ('Large-PowerCluster', 'Large-PowerCluster'), ('email-Eu-core', 'email-Eu-core'), ('ca-GrQc', 'ca-GrQc'), ('ca-CondMat', 'ca-CondMat'), ('BA-50k', 'BA-50k'), ('BA-100k', 'BA-100k'), ('SBM-4k', 'Multiscale-SBM-4k')]
    f = Fig('fig_perf', W, 130)
    meth = [('Naive', NAIVE, 'd'), ('CELF', INK, 'c'), ('Inc', INC, 's'), ('Dynamic-Logdet', BLUE, 't')]
    top = 22; rh = 10.4; a = Axes(f, 176, top, 136, rh * 10, (.002, 1000), (0, 1), True); b = Axes(f, 330, top, 92, rh * 10, (5, 100000), (0, 1), True)
    f.text(0, 8, 'Conditions', 7.4, INK, 'start', True); f.text(a.x0, 8, 'Online time (s), lower is better', 7.4, INK, 'start', True); f.text(b.x0, 8, 'Online RHS, lower is better', 7.4, INK, 'start', True); f.text(W, 8, 'Total time (s)', 7.4, INK, 'end', True)
    cx = dict(n=88, m=116, M=138, k=149, ti=170)
    for x, t in ((88, '*n*'), (116, '*m*'), (138, '*M*'), (149, '*k*')): f.text(x, 17, t, 6.6, MID, 'end')
    f.text(0, 17, 'Dataset', 6.6, MID)
    tot = {'CELF': 462, 'Inc': 488, 'Dynamic-Logdet': 516}
    for k_, x in tot.items(): f.text(x, 17, {'Dynamic-Logdet': 'Logdet'}.get(k_, k_), 6.4, {'CELF': INK, 'Inc': INC, 'Dynamic-Logdet': BLUE}[k_], 'end')
    def mark(x, y, c, k):
        if k == 'd': f.raw(f'<path d="M{x:.2f} {y-2:.2f}l2 2l-2 2l-2 -2z" fill="{c}"/>')
        elif k == 't': f.raw(f'<path d="M{x:.2f} {y-3.2:.2f}L{x+3:.2f} {y+2.2:.2f}L{x-3:.2f} {y+2.2:.2f}Z" fill="none" stroke="{c}" stroke-width=".65"/>')
        elif k == 's': f.rect(x - 1.6, y - 1.6, 3.2, 3.2, c, WHITE, .2)
        else: f.dot(x, y, 1.75, c, WHITE, .2)
    T = ((a, [.01, .1, 1, 10, 100], ['0.01', '0.1', '1', '10', '100']), (b, [10, 100, 1000, 10000], ['10', '100', '1k', '10k']))
    for ax, ticks, _ in T:
        for t in ticks: f.line(ax.X(t), top - 1, ax.X(t), top + 9 * rh, HAIR, .5)
    y = top; out = {}
    M_ = {'SBM-4k': 59772}
    for i, (nm, key) in enumerate(rows):
        d = perf(key); s = d['summary']; m_ = d['metadata']
        if nm == 'SBM-4k': f.rect(0, y - .5, W, rh - .2, '#F5EBF0')
        if i in (3, 6): f.line(0, y - .5, W, y - .5, RULE, .35)
        yc = y + rh / 2 - .4
        f.text(0, yc + 2.3, nm + ('†' if nm == 'SBM-4k' else ''), 6.8, INK, 'start', nm == 'SBM-4k')
        mm = m_.get('m', M_.get(nm))
        for x_, val in ((88, m_['n']), (116, mm), (138, m_['M']), (149, m_['k'])): f.text(x_, yc + 2.3, f'{val:,}', 6.5, INK, 'end')
        vt = {x: s[x]['online_seconds']['mean'] for x, *_ in meth if x in s and not (x == 'Naive' and nm == 'BA-100k')}
        vr = {x: s[x]['online_calls']['mean'] for x in vt}
        for ax, v in ((a, vt), (b, vr)):
            xs = [ax.X(x) for x in v.values()]; f.line(min(xs), yc, max(xs), yc, mix(WHITE, INK, .3), .5)
            for x, c, k in meth:
                if x in v: mark(ax.X(v[x]), yc, c, k)
        if 'Naive' in vt: f.text(a.X(vt['Naive']) + 4, yc + 2.3, f"{vt['Naive'] / vt['CELF']:.0f}×", 6.3, MID)
        if nm == 'SBM-4k': f.text(a.X(vt['Inc']) - 4, yc + 2.3, f"Inc {vt['CELF'] / vt['Inc']:.1f}× online", 6.3, INC, 'end')
        tt = {x: s[x]['seconds']['mean'] for x in tot}; best = min(tt, key=tt.get); out[nm] = tt
        for k_, x_ in tot.items(): f.text(x_, yc + 2.3, f'{tt[k_]:.3g}', 6.5, INK, 'end', k_ == best)
        y += rh
    yb = top + 9 * rh + 1
    for ax, ticks, labs in T:
        f.line(ax.x0, yb, ax.x0 + ax.w, yb, INK, .45)
        for t, l in zip(ticks, labs): f.line(ax.X(t), yb, ax.X(t), yb + 2, INK, .4); f.text(ax.X(t), yb + 8.6, l, 6.7, INK, 'middle')
    kx = 0; ky = yb + 9
    for (x, c, k), t in zip(meth, ('Naive', 'CELF', 'Inc', 'Logdet')):
        mark(kx + 2, ky - 2.3, c, k); f.text(kx + 7, ky, t, 6.8, c); kx += 30
    f.text(kx - 1, ky, '×: online Naive/CELF', 6.3, MID)
    f.h = ky + 3; f.save(); print({k: {a_: round(b_, 3) for a_, b_ in v.items()} for k, v in out.items() if k in ('SBM-4k', 'Large-PowerCluster', 'BA-100k')})

def fig_replication():
    W = 252; S = (('s0.1', '0.10', .1), ('s0.5', '0.50', .5), ('s0.85', '0.85', .85)); T = ('0.05', '0.2', '1.0')
    f = Fig('tab_replication', W, 128); cols = [96, 148, 200]; cw = 50; rh = 15.5
    f.text(cols[0], 6.5, '*τ* = 0.05', 6.8, INK); f.text(cols[1], 6.5, '*τ* = 0.2', 6.8, INK); f.text(cols[2], 6.5, '*τ* = 1', 6.8, INK)
    f.text(14, 6.5, '*s*', 6.8, MID, 'end'); f.text(44, 6.5, 'Swap', 6.8, MID, 'end'); f.text(90, 6.5, 'Gain', 6.8, MID, 'end')
    y = 9
    for fam, nm in (('sbm', 'SBM'), ('hetero', 'Heterogeneous')):
        f.text(0, y + 6, nm, 6.8, INK, 'start', True); y += 8.5
        for k, lab, sv in S:
            rs = ms[fam]
            ov = np.array([100 * (1 - r['methods'][k]['overlap_logdet'] / 15) for r in rs])
            gn = np.array([100 * (r['methods'][k]['gain'][str(sv)] / r['methods']['Logdet']['gain'][str(sv)] - 1) for r in rs])
            f.text(14, y + 7, lab, 6.8, INK, 'end'); f.text(44, y + 7, f'{ov.mean():.0f}%', 6.8, MID, 'end'); f.text(90, y + 7, f'+{gn.mean():.2f}', 6.8, MID, 'end')
            for x, t in zip(cols, T):
                d = np.array([100 * (r['methods'][k]['heat'][t] / r['methods']['Logdet']['heat'][t] - 1) for r in rs])
                m_, sd = d.mean(), d.std(ddof=1); wins = int((d > 0).sum())
                sig = abs(m_) >= sd; t_ = min(abs(m_) / 8, 1) ** .7
                bg = mix(WHITE, ORANGE if m_ > 0 else BLUE, .1 + .55 * t_) if sig else '#F4F5F6'
                f.rect(x, y, cw - 1.2, rh - 1.2, bg)
                f.text(x + 3, y + 6.5, ('+' if m_ > 0 else '−') + f'{abs(m_):.2f}', 7, INK, 'start', sig)
                f.text(x + 28, y + 6.5, f'±{sd:.2f}', 5.8, INK)
                for i in range(10): f.dot(x + 3.2 + i * 2.7, y + 11, .85, INK if i < wins else 'none', INK, .3)
            y += rh
        y += 3
    f.h = y - 1; f.save()

cnt = fig_incround(); print('celf per round', cnt)
for fn in (fig_horizon, fig_framework, fig_weights, fig_quad, fig_scaling, fig_rounds, fig_perf, fig_replication): fn()
cur = lambda p, x, m: json.load(open(R / f'curves_logdet/{p}-{x}.json'))['summary'][m]['seconds']['mean']
for p, xs in (('pool', (250, 4000)), ('budget', (5, 60))):
    print(p, {m: [round(cur(p, x, m), 3) for x in xs] for m in ('Naive', 'CELF', 'Inc', 'Dynamic-Logdet')})
