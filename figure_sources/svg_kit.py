"""FastFrac figures authored as SVG from the recorded result files. No value is typed by hand
except text copied from the paper. Usage: figs.py ROOT OUTDIR"""
import json, math, re, subprocess, sys, html, os
from pathlib import Path
import numpy as np

ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2]).resolve(); OUT.mkdir(exist_ok=True, parents=True)
R = ROOT / 'fastfrac_unified_results'
INK, MID, RULE, HAIR = '#1B2430', '#5F6873', '#C9CED4', '#E9EBEE'
BLUE, TEAL, ORANGE, INC, NAIVE = '#1D4B8F', '#1F8A8C', '#DB7A12', '#B8255F', '#8A929C'
SANS = "'Nimbus Sans'"; SERIF = "'STIX Two Text'"
# Figures whose files in figures/ were hand-refined after generation (2026-11-05 revision:
# rescaled fonts, ellipsis rows in fig_framework, Inc markers in fig_rounds, x/10 counts in
# tab_replication, ...). These edits are not in the generator code, so save() never overwrites
# them: regenerated versions go to OUT/_regenerated/ for comparison. Set
# FASTFRAC_OVERWRITE_FINAL=1 to overwrite deliberately.
FINAL_FIGS = {'fig_framework', 'fig_weights', 'fig_incround', 'fig_rounds', 'fig_perf', 'tab_replication'}


def mix(a, b, t):
    pa = [int(a[i:i + 2], 16) for i in (1, 3, 5)]; pb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return '#%02X%02X%02X' % tuple(round(x + (y - x) * t) for x, y in zip(pa, pb))


TOK = re.compile(r'(\*[^*]+\*|_\{[^}]*\}|\^\{[^}]*\})')


def spans(s, size):
    """*x* math italic, _{x} subscript, ^{x} superscript (inner * allowed via plain italics)."""
    out, shift = [], 0.0
    for part in TOK.split(s):
        if not part: continue
        if part.startswith('*'):
            out.append(f'<tspan font-family="{SERIF}" font-style="italic" font-size="{size * 1.1:.2f}" dy="{-shift:.2f}">{html.escape(part[1:-1])}</tspan>'); shift = 0
        elif part[0] in '_^':
            d = size * (.28 if part[0] == '_' else -.36)
            out.append(f'<tspan font-family="{SERIF}" font-style="italic" font-size="{size * .78:.2f}" dy="{d - shift:.2f}">{html.escape(part[2:-1])}</tspan>'); shift = d
        else:
            out.append(f'<tspan dy="{-shift:.2f}">{html.escape(part)}</tspan>'); shift = 0
    return ''.join(out)


class Fig:
    def __init__(s, name, w, h): s.name, s.w, s.h, s.b = name, w, h, []
    def raw(s, x): s.b.append(x)
    def line(s, x1, y1, x2, y2, c=INK, w=.4, dash=None, cap='butt'):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        s.raw(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{c}" stroke-width="{w}" stroke-linecap="{cap}"{d}/>')
    def path(s, d, c=INK, w=.8, fill='none', dash=None):
        e = f' stroke-dasharray="{dash}"' if dash else ''
        s.raw(f'<path d="{d}" fill="{fill}" stroke="{c}" stroke-width="{w}" stroke-linejoin="round" stroke-linecap="round"{e}/>')
    def poly(s, pts, c=INK, w=1., dash=None):
        s.path('M' + ' L'.join(f'{x:.2f} {y:.2f}' for x, y in pts), c, w, dash=dash)
    def area(s, pts, fill): s.raw('<path d="M' + ' L'.join(f'{x:.2f} {y:.2f}' for x, y in pts) + f'Z" fill="{fill}"/>')
    def dot(s, x, y, r=1.3, fill=INK, stroke='none', sw=0):
        s.raw(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    def rect(s, x, y, w, h, fill='none', c='none', sw=.4, r=0):
        s.raw(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{r}" fill="{fill}" stroke="{c}" stroke-width="{sw}"/>')
    def text(s, x, y, t, size=7, c=INK, a='start', bold=False, rot=None):
        tr = f' transform="rotate({rot} {x:.2f} {y:.2f})"' if rot is not None else ''
        wt = ' font-weight="700"' if bold else ''
        s.raw(f'<text x="{x:.2f}" y="{y:.2f}" font-family="{SANS}" font-size="{size}" fill="{c}" text-anchor="{a}"{wt}{tr}>{spans(t, size)}</text>')
    def tri(s, x, y, r, fill, stroke='white', sw=.25):
        s.raw(f'<path d="M{x:.2f} {y - r:.2f}L{x + r * .95:.2f} {y + r * .72:.2f}L{x - r * .95:.2f} {y + r * .72:.2f}Z" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    def save(s):
        body = '\n'.join(s.b)
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{s.w}pt" height="{s.h}pt" viewBox="0 0 {s.w} {s.h}">{body}</svg>'
        out = OUT
        if s.name in FINAL_FIGS and (OUT / f'{s.name}.pdf').exists() and os.environ.get('FASTFRAC_OVERWRITE_FINAL') != '1':
            out = OUT / '_regenerated'; out.mkdir(exist_ok=True)
            print(f'[svg_kit] {s.name} is hand-finalized; kept {OUT / s.name}.pdf, wrote regenerated copy to {out}/')
        (out / f'{s.name}.svg').write_text(svg)
        page = f'<!doctype html><meta charset="utf-8"><style>@page{{size:{s.w}pt {s.h}pt;margin:0}}html,body{{margin:0;padding:0}}svg{{display:block}}</style>{svg}'
        h = out / f'{s.name}.html'; h.write_text(page)
        subprocess.run(['google-chrome', '--headless', '--no-sandbox', '--disable-gpu', '--no-pdf-header-footer',
                        f'--user-data-dir=/tmp/fastfrac-chrome-profile', f'--print-to-pdf={out}/{s.name}.pdf', f'file://{h}'], check=True, capture_output=True)
        raw = out / f'{s.name}.pdf'; tmp = out / f'{s.name}.tmp.pdf'
        subprocess.run(['/usr/bin/gs', '-q', '-dNOPAUSE', '-dBATCH', '-sDEVICE=pdfwrite', '-dNoOutputFonts', '-dCompatibilityLevel=1.5', f'-sOutputFile={tmp}', str(raw)], check=True, capture_output=True); tmp.replace(raw)


class Axes:
    def __init__(s, f, x, y, w, h, xlim, ylim, xlog=False, ylog=False):
        s.f, s.x0, s.y0, s.w, s.h, s.xlim, s.ylim, s.xlog, s.ylog = f, x, y, w, h, xlim, ylim, xlog, ylog
    def X(s, v):
        a, b = s.xlim
        return s.x0 + s.w * ((math.log10(v) - math.log10(a)) / (math.log10(b) - math.log10(a)) if s.xlog else (v - a) / (b - a))
    def Y(s, v):
        a, b = s.ylim
        return s.y0 + s.h - s.h * ((math.log10(v) - math.log10(a)) / (math.log10(b) - math.log10(a)) if s.ylog else (v - a) / (b - a))
    def xticks(s, vals, labels=None, label=None, axis=True):
        f = s.f; yb = s.y0 + s.h
        if axis: f.line(s.x0, yb, s.x0 + s.w, yb, INK, .45)
        for i, v in enumerate(vals):
            f.line(s.X(v), yb, s.X(v), yb + 2, INK, .4)
            f.text(s.X(v), yb + 8.6, labels[i] if labels else str(v), 6.8, INK, 'middle')
        if label: f.text(s.x0 + s.w / 2, yb + 17.5, label, 7, INK, 'middle')
    def yticks(s, vals, labels=None, label=None, grid=True):
        f = s.f
        for i, v in enumerate(vals):
            if grid: f.line(s.x0, s.Y(v), s.x0 + s.w, s.Y(v), HAIR, .5)
            f.text(s.x0 - 3, s.Y(v) + 2.3, labels[i] if labels else str(v), 6.8, INK, 'end')
        if label: f.text(s.x0 - 21, s.y0 + s.h / 2, label, 7, INK, 'middle', rot=-90)
    def title(s, t, sub=None):
        s.f.text(s.x0 - (0), s.y0 - 6, t, 7.6, INK, 'start', True)


def pow10(e):
    return '1' if e == 0 else f'10^{{{e}}}'.replace('-', '−')


