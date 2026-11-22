#!/usr/bin/env python3
"""GeoAgent P23: One-shot, retrospective prequential feedback simulation.

Only trusted, SHA-pinned P22 models are loaded. No classifier retraining;
decisions never access labels for their own evaluation window.
Online deployment validity requires --mode strict with authentic event/release dates.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import time
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

# Release fix (2026-11-21): the sibling modules are imported by bare name, which only
# worked when the script was launched as `python agent/feedback/<script>.py`; running it
# as a module (`python -m agent.feedback.p23_feedback_replay`) failed with
# ModuleNotFoundError. Purely an import-path fix; no computation changes.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p22_frozen_reference as F

CLASSES = tuple(F.CLASSES)
PROB_COLS = list(F.PROB_COLS)
MODEL_SHA = {
    'P22_head_raw.joblib': '30f007a415366b583a4505e888a7717111e1e86714e5c7614dc3c59ff101d462',
    'P22_head_prob.joblib': 'c5f85c71b566fed450278ebadd6d3500bbb6348d7d1515b9839a3de4185a92d3',
    'P22_action_value.joblib': 'f0f5cc4fe7b78d2b91518d5ad84c56f75c63d6da6831b0601c6d49d7bf563804',
}
PRIOR_STRENGTH=20.0
PRIOR_WEIGHT=0.50
BUDGET=0.10
KNOWN_EVAL_SHA='69ec2669006367dde0434c3dc3e303474fc53b18af9a036e7ec4aaafbf185d86'


def json_default(v):
    if isinstance(v,np.generic): return v.item()
    if isinstance(v,np.ndarray): return v.tolist()
    raise TypeError(f'Not JSON serializable: {type(v).__name__}')

def sha(path: Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def check_data(d:pd.DataFrame)->None:
    req=['row_id','accident_group','incident_year','label',*PROB_COLS,*F.NUM,*F.CAT]
    missing=[c for c in req if c not in d.columns]
    if missing: raise ValueError('MISSING_COLUMNS: '+','.join(missing))
    if d.row_id.isna().any() or d.row_id.astype(str).duplicated().any():
        raise ValueError('BAD_ROW_IDS')
    if d.accident_group.isna().any() or d.accident_group.astype(str).str.strip().eq('').any():
        raise ValueError('BAD_GROUP_IDS')
    if not d.label.isin(CLASSES).all():raise ValueError('BAD_LABELS')
    years=pd.to_numeric(d.incident_year,errors='coerce')
    if years.isna().any() or (years%1!=0).any():raise ValueError('BAD_YEARS')
    arr=d[PROB_COLS].to_numpy(float)
    if not np.isfinite(arr).all() or (arr<0).any() or not np.allclose(arr.sum(1),1,atol=1e-5):
        raise ValueError('BAD_PROBABILITIES')
    g_year=d.assign(year=years.astype(int)).groupby('accident_group').year.nunique()
    if (g_year>1).any():raise ValueError('CROSS_YEAR_GROUP: same accident in multiple windows')


def release_audit(d:pd.DataFrame,years:list[int],mode:str)->dict:
    if mode=='retrospective':
        return {'mode':'retrospective','actual_label_release_verified':False,
                'warning':'SIMULATED batch feedback after each annual window; NOT verified real-time label availability'}
    if 'event_date' not in d or 'label_available_at' not in d:
        raise ValueError('STRICT_REQUIRES_event_date_AND_label_available_at')
    event=pd.to_datetime(d.event_date,utc=True,errors='coerce')
    rel=pd.to_datetime(d.label_available_at,utc=True,errors='coerce')
    if event.isna().any() or rel.isna().any():raise ValueError('BAD_RELEASE_DATES')
    for early,later in zip(years[:-1],years[1:]):
        early_mask=d.incident_year.astype(int)==early
        later_mask=d.incident_year.astype(int)==later
        if rel[early_mask].max()>event[later_mask].min():
            raise ValueError(f'LABEL_NOT_AVAILABLE_BEFORE_NEXT_WINDOW:{early}->{later}')
    return {'mode':'strict','actual_label_release_verified':True}


def feedback_mask(d:pd.DataFrame,available_years:list[int])->np.ndarray:
    return d.incident_year.astype(int).isin(available_years).to_numpy()


def action_profile(actions:pd.DataFrame,delta:np.ndarray):
    """Only previously revealed outcomes enter this shrinkage memory."""
    state={}
    for a,v in zip(actions.itertuples(index=False),delta):
        key=(int(a.base_class),int(a.dest_class))
        n,s=state.get(key,(0,0.0))
        state[key]=(n+1,s+float(v))
    return state


def score_profile(actions:pd.DataFrame,state:dict)->np.ndarray:
    out=[]
    for a in actions.itertuples(index=False):
        n,s=state.get((int(a.base_class),int(a.dest_class)),(0,0.0))
        out.append(s/(n+PRIOR_STRENGTH))
    return np.asarray(out)


def select(actions:pd.DataFrame,score:np.ndarray,base:np.ndarray,ids:np.ndarray)->np.ndarray:
    chosen,_=F.value_policy(actions,score,base,ids,budget=BUDGET)
    return chosen


def metrics(y,base,pred):
    z=F.metrics(y,base,pred)
    z['mean_delta_accuracy_pp']=float(100*(pred==y).mean()-100*(base==y).mean())
    return z


def paired_group_ci(groups,y,a,b,seed=2301,nboot=500):
    _,gid=np.unique(groups.astype(str),return_inverse=True)
    n=int(gid.max()+1)
    diff=(a==y).astype(float)-(b==y).astype(float)
    totals=np.bincount(gid,weights=diff,minlength=n)
    counts=np.bincount(gid,minlength=n)
    rng=np.random.default_rng(seed)
    boot=[]
    for _ in range(nboot):
        ix=rng.integers(0,n,size=n)
        boot.append(100*totals[ix].sum()/counts[ix].sum())
    return [float(x) for x in np.quantile(boot,[.025,.975])]


def get_aux(d,models):
    p=d[PROB_COLS].to_numpy(float)
    base=p.argmax(axis=1)
    raw=models['P22_head_raw.joblib'].predict_proba(F.head_features(d,p,True))
    prob=models['P22_head_prob.joblib'].predict_proba(F.head_features(d,p,False))
    if raw.shape != p.shape or prob.shape != p.shape: raise ValueError('MODEL_CLASS_ORDER_OR_DIM_MISMATCH')
    # P22 class order is fixed, no silent remapping.
    return p,base,raw,prob


def ordinary_features(p,qraw,qprob):
    return np.concatenate([p,qraw,qprob],axis=1)


def fit_ordinary(x,y):
    """One fixed fit per feedback update, same observed feedback as Agent1."""
    sc=StandardScaler().fit(x)
    m=LogisticRegression(C=.1,max_iter=400,solver='lbfgs')
    m.fit(sc.transform(x),y)
    return (sc,m)


def ordinary_policy(x,model,base,ids):
    if model is None: return base.copy()
    sc,m=model
    q=m.predict_proba(sc.transform(x))
    classes=m.classes_.astype(int)
    best_idx=np.argmax(q,axis=1)
    dest=classes[best_idx]
    gain=q[np.arange(len(dest)),best_idx] - np.array([
        q[i,np.flatnonzero(classes==b)[0]] if b in classes else 0.0
        for i,b in enumerate(base)])
    gain=np.asarray(gain,dtype=float).reshape(-1)
    eligible=np.flatnonzero((dest!=base)&(gain>0))
    order=eligible[np.lexsort((ids[eligible].astype(str),-gain[eligible]))]
    order=order[:math.ceil(BUDGET*len(base))]
    out=base.copy();out[order]=dest[order]
    return out


def evaluate_one(d_train,d_eval,models,iteration):
    # IMPORTANT: labels of d_eval are NEVER passed into inference or fit.
    start=time.perf_counter()
    pt,bt,qrt,qpt=get_aux(d_train,models)
    pe,be,qre,qpe=get_aux(d_eval,models)
    yt=np.array([CLASSES.index(k) for k in d_train.label])
    # local row indices for past labeled feedback only
    at=F.create_actions(pt,qrt,qpt,bt,np.arange(len(d_train)))
    de=F.delta_labels(at,yt,bt)
    value_model=F.fit_value_model(at,de)
    prior=action_profile(at,de)
    ae=F.create_actions(pe,qre,qpe,be,np.arange(len(d_eval)))
    raw_score=value_model.predict(ae[F.FEATURES_NUM+F.FEATURES_CAT])
    profile_score=score_profile(ae,prior)
    full_score=(1-PRIOR_WEIGHT)*raw_score+PRIOR_WEIGHT*profile_score
    ids=d_eval.row_id.astype(str).to_numpy()
    preds={
      'KEEP':be.copy(),
      'AGENT1_FEEDBACK_MEMORY':select(ae,full_score,be,ids),
      'ABLATION_NO_ERROR_MEMORY':select(ae,raw_score,be,ids),
      'STATIC_P22_CVG':None,
      'P21_FIXED_RAW':F.fixed_q_rule(qre,be,ids),
    }
    frozen_value=models['P22_action_value.joblib'].predict(ae[F.FEATURES_NUM+F.FEATURES_CAT])
    preds['STATIC_P22_CVG']=select(ae,frozen_value,be,ids)
    k=int(np.count_nonzero(preds['AGENT1_FEEDBACK_MEMORY']!=be))
    preds['B0_CONFIDENCE_EQUAL_K']=F.equal_k_rule(pe,qre,qpe,be,ids,k,'raw_conf')
    preds['B1_MARGIN_EQUAL_K']=F.equal_k_rule(pe,qre,qpe,be,ids,k,'raw_margin')
    if len(np.unique(yt))>=2:
        ordinary=fit_ordinary(ordinary_features(pt,qrt,qpt),yt)
    else:
        ordinary=None
    preds['ORDINARY_FEEDBACK_LOGREG']=ordinary_policy(ordinary_features(pe,qre,qpe),ordinary,be,ids)
    # Only after all inference policies fixed, unlock current-window labels for metrics.
    ye=np.array([CLASSES.index(k) for k in d_eval.label])
    g=d_eval.accident_group.astype(str).to_numpy()
    stats={key:metrics(ye,be,pr) for key,pr in preds.items()}
    cints={key:paired_group_ci(g,ye,pr,be,seed=2301+iteration) for key,pr in preds.items() if key!='KEEP'}
    comparisons={name:{'delta_net_correct':int(stats['AGENT1_FEEDBACK_MEMORY']['net_correct']-stats[name]['net_correct']),
                      'group_bootstrap_95_ci_accuracy_pp':paired_group_ci(g,ye,preds['AGENT1_FEEDBACK_MEMORY'],preds[name],seed=2301+iteration)}
                 for name in ['ABLATION_NO_ERROR_MEMORY','B0_CONFIDENCE_EQUAL_K','B1_MARGIN_EQUAL_K','ORDINARY_FEEDBACK_LOGREG','STATIC_P22_CVG']}
    out={
        'relative_comparisons':comparisons,
        'train_feedback_n':len(d_train),'eval_n':len(d_eval),'eval_groups':int(d_eval.accident_group.nunique()),
        'class_support':{c:int((ye==i).sum()) for i,c in enumerate(CLASSES)},
        'decision_count_full':k,'learned_gain_fit_ridge':True,
        'nonzero_training_feedback':{'positive':int((de==1).sum()),'negative':int((de==-1).sum()),'neutral':int((de==0).sum())},
        'model_elapsed_seconds':round(time.perf_counter()-start,3),
        'stats':stats,'group_bootstrap_95_ci_accuracy_pp':cints,
        'class_support_warning':any((ye==i).sum()<30 for i in range(6)),
    }
    return out,preds,be


def run(args):
    dst=Path(args.out)
    if dst.exists(): raise FileExistsError(f'OUTPUT_ALREADY_EXISTS: {dst}')
    inp=Path(args.input)
    digest=sha(inp)
    expected=args.expected_input_sha256 or KNOWN_EVAL_SHA
    if digest!=expected:raise ValueError(f'INPUT_HASH_MISMATCH expected={expected} actual={digest}')
    model_dir=Path(args.models_dir)
    for filename,expected_hash in MODEL_SHA.items():
        path=model_dir/filename
        if sha(path)!=expected_hash:raise ValueError('MODEL_HASH_MISMATCH:'+filename)
    d=pd.read_csv(inp,dtype={'row_id':str,'accident_group':str},low_memory=False)
    check_data(d)
    years=[int(v) for v in sorted(d.incident_year.astype(int).unique())]
    if len(years)<3:raise ValueError('LESS_THAN_THREE_DISTINCT_YEARS: no two-update experiment possible')
    if len(years)>3:raise ValueError('MORE_THAN_THREE_YEARS: pre-register exact three yearly windows first')
    audit=release_audit(d,years,args.mode)
    # Validation complete; no output until all preconditions pass.
    models={n:joblib.load(model_dir/n) for n in MODEL_SHA}  # trusted local, hash-pinned
    if float(models['P22_action_value.joblib'].named_steps['ridge'].alpha)!=50.0:
        raise ValueError('FROZEN_P22_ALPHA_MISMATCH')
    d=d.sort_values(['incident_year','accident_group','row_id'],kind='stable').reset_index(drop=True)
    windows={yr:d[d.incident_year.astype(int)==yr].reset_index(drop=True) for yr in years}
    for yr in years:
        if len(windows[yr])<25:raise ValueError(f'TOO_SMALL_WINDOW_{yr}: {len(windows[yr])}')
    reports={}
    pool_predictions=[]
    for it in (1,2):
        prior_years=years[:it]; target_year=years[it]
        train=d[d.incident_year.astype(int).isin(prior_years)].reset_index(drop=True)
        result,predictions,base=evaluate_one(train,windows[target_year],models,it)
        pool_predictions.append((windows[target_year],predictions,base))
        result['feedback_years']=prior_years
        result['evaluated_year']=target_year
        reports[str(target_year)]=result
    # Pool only *evaluated* windows; no W1 in post-update score.
    combined={}
    for method in reports[str(years[1])]['stats']:
        n=sum(reports[str(y)]['eval_n'] for y in years[1:])
        delta=sum(reports[str(y)]['stats'][method]['net_correct'] for y in years[1:])
        combined[method]={'net_correct':int(delta),'total_evaluated_n':int(n),'delta_accuracy_pp':float(100*delta/n),
                          'changed':sum(reports[str(y)]['stats'][method]['changed'] for y in years[1:])}
    # Pre-set decision logic, without parameter tuning.
    all_y=np.concatenate([np.array([CLASSES.index(k) for k in df.label]) for df,_,_ in pool_predictions])
    all_g=np.concatenate([df.accident_group.astype(str).to_numpy() for df,_,_ in pool_predictions])
    full_all=np.concatenate([pp['AGENT1_FEEDBACK_MEMORY'] for _,pp,_ in pool_predictions])
    pooled_cis={}
    for name in pool_predictions[0][1]:
        other=np.concatenate([pp[name] for _,pp,_ in pool_predictions])
        pooled_cis[name]=paired_group_ci(all_g,all_y,full_all,other,seed=2303)
    full=combined['AGENT1_FEEDBACK_MEMORY']; keep=combined['KEEP']
    stronger=max(combined['B0_CONFIDENCE_EQUAL_K']['net_correct'],
                 combined['B1_MARGIN_EQUAL_K']['net_correct'],
                 combined['ORDINARY_FEEDBACK_LOGREG']['net_correct'])
    best_vs_simple=full['net_correct']>stronger
    both_positive=all(reports[str(y)]['stats']['AGENT1_FEEDBACK_MEMORY']['net_correct']>0 for y in years[1:])
    ablation_gain=full['net_correct']-combined['ABLATION_NO_ERROR_MEMORY']['net_correct']
    max_drops=[]
    for yr in years[1:]:
        s=reports[str(yr)]['stats']['AGENT1_FEEDBACK_MEMORY']['recall_delta_pp']
        max_drops += [min(s.values())]
    class_support_weak=any(reports[str(y)]['class_support_warning'] for y in years[1:])
    positive_signal=(full['net_correct']>0 and best_vs_simple and both_positive and ablation_gain>0 and all(x>=-3 for x in max_drops))
    if positive_signal and class_support_weak:
        status='EXPLORATORY_GAIN_CLASS_SAFETY_UNRESOLVED'
    else:
        status='EXPLORATORY_POSITIVE' if positive_signal else 'CONSERVATIVE_ZERO_HARM_PROFILE'
    result={'status':status,'scope':'RETROSPECTIVE_HISTORICAL_REPLAY',
            'protocol_scope':'Retrospective annual-window historical replay on frozen predictions; online deployment requires strict release-date audit.',
            'input_sha256':digest,'model_sha256':MODEL_SHA,'protocol':audit,
            'window_years':years,'pooled_CI_of_full_vs_each_strategy_accuracy_pp':pooled_cis,'settings':{'budget':BUDGET,'pair_prior_strength':PRIOR_STRENGTH,'pair_prior_weight':PRIOR_WEIGHT,
              'ridge_alpha':50.0,'ordinary_logreg_C':.1,'bootstrap_replicates':500},
            'windows':reports,'pooled_scores':combined,'decision':{
              'net_positive':full['net_correct']>0,'better_than_all_simple':best_vs_simple,
              'positive_in_both_windows':both_positive,'memory_ablation_net_gain':int(ablation_gain),
              'largest_class_recall_drop_pp':float(min(max_drops)),
              'class_support_insufficient_for_strict_g3':class_support_weak,
              'full_vs_KEEP_CI_excludes_zero':pooled_cis['KEEP'][0]>0,
              'one_shot_stopping_rule':'FIXED_PROTOCOL_NO_RETUNING',
            }}
    dst.mkdir(parents=True,exist_ok=False)
    (dst/'P23_RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=json_default)+'\n')
    try:
        import sklearn
        sk_version=sklearn.__version__
    except Exception:
        sk_version='unknown'
    provenance={'created_at_utc':datetime.now(timezone.utc).isoformat(),
                'input_path':str(inp.resolve()),'input_sha256':digest,
                'frozen_models_dir':str(model_dir.resolve()),'model_sha256':MODEL_SHA,
                'python':sys.version.split()[0],'scikit_learn':sk_version,
                'platform':platform.platform(),'command':sys.argv,
                'evaluation_status':'RETROSPECTIVE_2024PLUS_HISTORICAL_REPLAY',
                'model_updates':2,'preserve_original_files':True}
    (dst/'P23_PROVENANCE.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2,default=json_default)+'\n')
    report=f"""# P23 Historical Feedback Replay Summary\n\n- Status: **{status}**\n- Input SHA-256: `{digest}`\n- Protocol scope: **Retrospective historical replay** (not a prospective online deployment).\n- Feedback windows: {years[0]} -> {years[1]} -> {years[2]}\n- Full policy pooled net gain: **{full['net_correct']}** / {full['total_evaluated_n']} (delta Accuracy **+{full['delta_accuracy_pp']:.4f} pp**, {full['changed']} accepted revisions)\n- Max simple-baseline pooled net gain: **{stronger}**\n- Class-pair memory ablation net delta: **{ablation_gain}** (inspect alongside intervention counts and harmful revisions)\n- Positive net gain in both windows: {both_positive} | Max class recall drop: {min(max_drops):.3f} pp\n- Protocol note: {audit.get('warning', 'Strict release-date audit passed.')}\n"""
    (dst/'P23_DECISION.md').write_text(report)
    print(report)
    return result


def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--input',required=True,help='Trusted 2024+ evaluation CSV from last recovery bridge')
    pa.add_argument('--models-dir',required=True,help='Frozen P22 three joblib weights')
    pa.add_argument('--out',required=True,help='NEW output directory; never overwrite')
    pa.add_argument('--expected-input-sha256',default=None,help='only if a different previously authorized input is explicitly locked')
    pa.add_argument('--mode',choices=['retrospective','strict'],default='retrospective')
    args=pa.parse_args();run(args)

if __name__=='__main__':main()
