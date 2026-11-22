#!/usr/bin/env python3
"""Read-only standard-library verification of the 8 archived adapted baseline result tables.
It verifies METRICS-ONLY; it does not train the baseline models.
"""
from pathlib import Path
import csv, math, sys

ROOT=Path(__file__).resolve().parents[1]
CL=['CORROSION','EQUIPMENT_FAILURE','EXTERNAL_DAMAGE','MATERIAL_DEFECT','OPERATIONAL','OTHER']

def read(rel):
    with (ROOT/rel).open(encoding='utf-8',newline='') as f: return list(csv.DictReader(f))

def metrics(y,p):
    correct=sum(a==b for a,b in zip(y,p)); n=len(y)
    assert n==len(p) and n>0
    f1=[]; support=[]
    for lab in CL:
        tp=sum(a==lab and b==lab for a,b in zip(y,p))
        fp=sum(a!=lab and b==lab for a,b in zip(y,p))
        fn=sum(a==lab and b!=lab for a,b in zip(y,p))
        f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0)
        support.append(tp+fn)
    return correct/n, sum(f1)/6, sum(a*b for a,b in zip(f1,support))/n,correct

def main():
    man={r['row_id']: r['label'] for r in read('models_and_configs/geoad_materials/data/data_row_manifest_development_only.csv') if r['split']=='dev_confirm'}
    assert len(man)==6417
    table={r['arm']:r for r in read('results/rq1_diagnosis_vs_baselines/t3_baseline_metrics.csv')}
    assert len(table)==8, f'Expected eight baseline summaries, got {len(table)}'
    for name,summary in table.items():
        rows=read(f'results/per_row_predictions/baselines_8/{name}.csv')
        assert len(rows)==6417
        ids=[r['row_id'] for r in rows]
        assert len(set(ids))==6417 and set(ids)==set(man), f'{name}: row IDs mismatch'
        assert all(r['y_true_label']==man[r['row_id']] for r in rows), f'{name}: true labels disagree'
        pred=[CL[int(r['pred'])] for r in rows]
        a,m,w,c=metrics([man[k] for k in ids],pred)
        expected=[float(summary[k]) for k in ('accuracy','macro_f1','weighted_f1')]
        got=[a,m,w]
        assert all(abs(t-v)<1e-10 for t,v in zip(expected,got)), f'{name}: expected={expected}, got={got}'
        assert c==int(summary['n_correct']),f'{name}: correct count discrepancy'
        print(f'PASS [METRICS-ONLY] {name}: N=6417, ACC={a*100:.4f}%, macro-F1={m*100:.4f}%')
    print('PASS [METRICS-ONLY] eight adapted baselines 8/8; no new training claimed')
if __name__=='__main__':
    try: main()
    except Exception as e:
        print(f'FAIL {type(e).__name__}: {e}',file=sys.stderr);sys.exit(1)
