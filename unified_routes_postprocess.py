"""Pure JSON postprocessing: distinguish true-trace error from lazy-key validity."""
from pathlib import Path
import json
import fastfrac_cpu_benchmark as ff

def sampled_key_audit(output_dir):
    out=Path(output_dir);path=out/'evaluation_routes_unified.json';data=json.loads(path.read_text())
    for d in data['workloads']:
        name=d['dataset'];by={x['name']:x for x in d['routes']}
        if 'audit' not in by.get('CRN-SLQ-CELF-p128',{}):continue
        g=by['CRN-SLQ-Greedy-p128']['audit'];l=by['CRN-SLQ-CELF-p128']['audit']
        grecs=json.loads((out/f'{name}_CRN-SLQ-Greedy-p128_oracle_trace.json').read_text())
        lrecs=json.loads((out/f'{name}_CRN-SLQ-CELF-p128_oracle_trace.json').read_text())
        gby={r['round']:dict(zip(r['ids'],r['values'])) for r in grecs};lby={}
        for rec in lrecs:lby.setdefault(rec['round'],[]).append(rec)
        cached={};checked=violations=0;largest=0.;rounds=0
        for r in range(d['k']):
            if g['selected'][:r]!=l['selected'][:r]:break
            current=gby[r]
            if r:
                for i in set(range(d['M']))-set(l['selected'][:r]):
                    if i in cached:
                        checked+=1;excess=current[i]-cached[i];largest=max(largest,excess)
                        violations+=int(excess>1e-10*max(1.,abs(current[i])))
            for rec in lby.get(r,[]):cached.update(zip(rec['ids'],rec['values']))
            rounds+=1
        d['sampled_objective_lazy_key_audit']={
            'definition':'At identical-prefix rounds, compare current exhaustive CRN-SLQ-p128 marginal with the latest point key cached by actual CRN-SLQ-CELF-p128; tolerance1e-10*max(1,abs(current)).',
            'audited_common_prefix_rounds':rounds,'cached_keys_checked':checked,'undercoverage_count':violations,
            'undercoverage_rate':violations/checked if checked else None,'maximum_positive_absolute_excess':largest}
    ff.write_json(path,data);return data
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',default='fastfrac_unified_results');a=p.parse_args();sampled_key_audit(a.output)
