#!/usr/bin/env python3
"""Read-only offline checks; validates archived metrics without ML libraries."""
import csv,sys,collections
from pathlib import Path
R=Path(__file__).resolve().parents[1]
CL=['CORROSION','EQUIPMENT_FAILURE','EXTERNAL_DAMAGE','MATERIAL_DEFECT','OPERATIONAL','OTHER']

def read(p):
 with (R/p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def scores(truth,pred):
 assert len(truth)==len(pred)>0
 acc=100*sum(a==b for a,b in zip(truth,pred))/len(truth)
 f=[]
 for lab in CL:
  tp=sum(a==lab and b==lab for a,b in zip(truth,pred))
  fp=sum(a!=lab and b==lab for a,b in zip(truth,pred))
  fn=sum(a==lab and b!=lab for a,b in zip(truth,pred))
  f.append(2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0.0)
 return round(acc,2),round(100*sum(f)/len(f),2)

def chk(name,cond,ex=''):
 print(('PASS' if cond else 'FAIL')+' '+name+(' '+ex if ex else ''))
 if not cond:raise AssertionError(name)

if __name__=='__main__':
 man=read('models_and_configs/geoad_materials/data/data_row_manifest_development_only.csv')
 truth={r['row_id']:r['label'] for r in man if r['split']=='dev_confirm'}
 pred=read('results/per_row_predictions/geoad/P_no_proto.csv')
 chk('dev-confirm count',len(truth)==len(pred)==6417)
 yt=[truth[r['row_id']] for r in pred]
 yp=[r['pred_label'] for r in pred]
 acc,f1=scores(yt,yp)
 chk('classifier Acc/Macro-F1', (acc,f1)==(71.59,58.63),f'{acc}% / {f1}%')
 p=read('results/per_row_predictions/PREDICTIONS_582.csv')
 ev={r['row_id']:r for r in read('results/per_row_predictions/EVALUATION_582.csv')}
 chk('replay rows match evaluator',len(p)==len(ev)==582 and {r['row_id'] for r in p}==set(ev))
 for year,k in [(2025,12),(2026,2)]:
  subset=[r for r in p if int(r['window'])==year]
  truth=[ev[r['row_id']]['label'] for r in subset]
  keep=[r['KEEP'] for r in subset]
  full=[r['P23_FEEDBACK_MEMORY'] for r in subset]
  ks=sum(a!=b for a,b in zip(keep,full)); fixes=sum(a!=b and b==y for a,b,y in zip(keep,full,truth)); harms=sum(a!=b and a==y for a,b,y in zip(keep,full,truth))
  chk(f'{year} Full changed/fixes/harms', (ks,fixes,harms)==(k,k,0),f'{ks}/{fixes}/{harms}')
  if year==2025:
   ak,mk=scores(truth,keep);af,mf=scores(truth,full)
   chk('2025 KEEP Acc/Macro-F1',(ak,mk)==(61.81,35.64))
   chk('2025 Full Acc/Macro-F1',(af,mf)==(64.02,39.27))
  for name,c,f,acc_,mf_ in ((('B0_CONFIDENCE_EQUAL_K',5,3,62.18,36.17) if year==2025 else ('B0_CONFIDENCE_EQUAL_K',1,1,55.,28.24)), (('B1_MARGIN_EQUAL_K',9,2,63.10,38.28) if year==2025 else ('B1_MARGIN_EQUAL_K',2,0,60.,39.87))):
   pp=[r[name] for r in subset]
   changed=sum(a!=b for a,b in zip(keep,pp)); corrected=sum(a!=b and b==y for a,b,y in zip(keep,pp,truth)); harmed=sum(a!=b and a==y for a,b,y in zip(keep,pp,truth))
   a,m=scores(truth,pp)
   chk(f'{year} {name}',(changed,corrected,harmed,a,m)==(k,c,f,acc_,mf_),f'changed={changed},fix={corrected},harm={harmed},Acc={a},Macro-F1={m}')
 print('ALL ARCHIVED METRIC CHECKS PASS (not neural inference)')
