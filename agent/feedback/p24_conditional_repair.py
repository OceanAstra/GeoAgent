#!/usr/bin/env python3
"""P24: feedback-conditioned six-class repair proposal, one-shot retrospective study.

Fresh candidate source: for a frozen GeOAD prediction b, propose one of all
other classes using conditional similarity among *previously revealed* labels.
The comparator is identical except it uses no sample-specific similarity.
This is exploratory replay, not a claim of prospective online validity.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# Release fix (2026-11-21): see the same note in p23_feedback_replay.py. Import-path only.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import p22_frozen_reference as F
import p23_feedback_replay as P

CLASSES = tuple(P.CLASSES)
MODEL_SHA = dict(P.MODEL_SHA)
EXPECTED_INPUT_SHA = '64b9937b7e949e6656aa2532243846c214f0a35c3a32bb9d6c8ef59cce0ff27b'
K_NEIGHBORS=25
PRIOR_STRENGTH=8.0
BUDGET=0.10
SEED=2401


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()


def category_pair_counts(base,pred):
    changed=np.flatnonzero(base!=pred)
    return dict(sorted(Counter(f'{CLASSES[int(base[i])]}->{CLASSES[int(pred[i])]}' for i in changed).items()))


def observe_features(p,qraw,qprob):
    """Only inference-time probabilities; NO labels, timestamps or source identifiers."""
    blocks=[np.asarray(a,dtype=float) for a in (p,qraw,qprob)]
    if any(x.ndim!=2 or x.shape[1]!=6 for x in blocks):
        raise ValueError('EXPECTED_THREE_SIX_CLASS_PROBABILITY_MATRICES')
    n=blocks[0].shape[0]
    if any(x.shape[0]!=n for x in blocks):raise ValueError('UNALIGNED_FEATURE_ROWS')
    x=np.hstack(blocks)
    if not np.isfinite(x).all():raise ValueError('NONFINITE_FEATURES')
    return x


class ConditionalRepairMemory:
    """One fixed local evidence mechanism, trained only on prior feedback.

    Local estimates P(Y | frozen-class, similarity-of-predictions). A global
    base-class prior is the *same-space* ablation. Predictive value for
    base->destination is P(Y=dest)-P(Y=base). All 6 labels are reachable.
    """
    def __init__(self,k=K_NEIGHBORS,prior_strength=PRIOR_STRENGTH):
        self.k=int(k);self.prior_strength=float(prior_strength)
        if self.k<1 or self.prior_strength<=0:raise ValueError('BAD_FIXED_PARAMS')

    def fit(self,x,base,y):
        x=np.asarray(x,float);base=np.asarray(base,int);y=np.asarray(y,int)
        if x.ndim!=2 or x.shape[1]!=18:raise ValueError('BAD_FEATURE_SHAPE')
        if len(x)!=len(base) or len(base)!=len(y):raise ValueError('BAD_LENGTH')
        if not np.isin(base,range(6)).all() or not np.isin(y,range(6)).all():
            raise ValueError('BAD_CLASSES')
        self.mu=x.mean(0);self.scale=np.maximum(x.std(0),0.05)
        self.banks={};self.priors={};self.support={}
        z=(x-self.mu)/self.scale
        for b in range(6):
            mask=base==b; ys=y[mask]
            self.banks[b]=(z[mask],ys)
            self.support[b]=int(mask.sum())
            counts=np.bincount(ys,minlength=6).astype(float)
            # Add symmetric half-count only to avoid unseen-class zero mass.
            self.priors[b]=(counts+0.5)/(counts.sum()+3.0)
        return self

    def posterior(self,x,base,conditioned=True):
        x=np.asarray(x,float);base=np.asarray(base,int)
        if len(x)!=len(base):raise ValueError('BAD_QUERY_LENGTH')
        z=(x-self.mu)/self.scale
        post=np.zeros((len(x),6))
        for j,b in enumerate(base):
            b=int(b)
            if b not in self.banks:raise ValueError('INVALID_BASE_CLASS')
            prior=self.priors[b]
            bank,ys=self.banks[b]
            if not conditioned or len(ys)==0:
                post[j]=prior;continue
            squared=((bank-z[j])**2).sum(1)
            take=np.argsort(squared,kind='stable')[:min(self.k,len(ys))]
            dist=np.sqrt(squared[take])
            bw=max(float(np.median(dist)),1.0)
            w=np.exp(-0.5*(dist/bw)**2)
            local=np.bincount(ys[take],weights=w,minlength=6)
            post[j]=(local+self.prior_strength*prior)/(w.sum()+self.prior_strength)
        return post


def choose(base,posterior,row_id,budget=BUDGET):
    """KEEP unless estimated gain is positive; max one action per row."""
    base=np.asarray(base,int);row_id=np.asarray(row_id,dtype=str)
    posterior=np.asarray(posterior,float)
    if posterior.shape!=(len(base),6):raise ValueError('BAD_POSTERIOR_SHAPE')
    if not np.isfinite(posterior).all():raise ValueError('BAD_POSTERIOR')
    gain=posterior-posterior[np.arange(len(base)),base][:,None]
    gain[np.arange(len(base)),base]=-np.inf
    dest=np.argmax(gain,axis=1)
    score=gain[np.arange(len(base)),dest]
    eligible=np.flatnonzero(score>0)
    order=eligible[np.lexsort((row_id[eligible],-score[eligible]))]
    chosen=order[:int(math.ceil(budget*len(base)))]
    pred=base.copy();pred[chosen]=dest[chosen]
    return pred,{'eligible':int(len(eligible)),'changed':int(len(chosen)),
                 'score_min_selected':float(score[chosen].min()) if len(chosen) else None}


def paired_ci(groups,y,first,second,seed=SEED,replications=600):
    return P.paired_group_ci(np.asarray(groups),np.asarray(y),
                             np.asarray(first),np.asarray(second),seed=seed,nboot=replications)


def prospective_change(prev_train,next_train,eval_df,models):
    """Counterfactual: only historical feedback changes; eval labels untouched."""
    pe,be,qre,qpe=P.get_aux(eval_df,models)
    xe=observe_features(pe,qre,qpe)
    prev=get_fitted_memory(prev_train,models)
    nxt=get_fitted_memory(next_train,models)
    ids=eval_df.row_id.astype(str).to_numpy()
    pred1,_=choose(be,prev.posterior(xe,be),ids)
    pred2,_=choose(be,nxt.posterior(xe,be),ids)
    return pred1,pred2


def get_fitted_memory(train,models):
    p,b,qr,qp=P.get_aux(train,models)
    x=observe_features(p,qr,qp)
    y=np.array([CLASSES.index(c) for c in train.label],dtype=int)
    return ConditionalRepairMemory().fit(x,b,y)


def evaluate(train,ev,models,iteration):
    # All fitting and decisions happen before *ever* reading ev.label.
    model=get_fitted_memory(train,models)
    p,b,qr,qp=P.get_aux(ev,models)
    x=observe_features(p,qr,qp)
    ids=ev.row_id.astype(str).to_numpy()
    conditioned,cond_audit=choose(b,model.posterior(x,b,True),ids)
    global_ablation,global_audit=choose(b,model.posterior(x,b,False),ids)
    # P23 creates its own fixed alternatives from exactly the same old feedback.
    ev_without_labels=ev.copy()
    ev_without_labels['label']=CLASSES[0]  # blocked from reference inference/fit
    _,p23_preds,_=P.evaluate_one(train,ev_without_labels,models,iteration)
    reference={
        'KEEP':b,
        'P24_CONDITIONAL':conditioned,
        'ABLATION_GLOBAL_PRIOR':global_ablation,
        'P23_FEEDBACK_MEMORY':p23_preds['AGENT1_FEEDBACK_MEMORY'],
        'P23_NO_MEMORY':p23_preds['ABLATION_NO_ERROR_MEMORY'],
        'ORDINARY_FEEDBACK_LOGREG':p23_preds['ORDINARY_FEEDBACK_LOGREG'],
        'STATIC_P22_CVG':p23_preds['STATIC_P22_CVG'],
    }
    k=int(np.count_nonzero(conditioned!=b))
    reference['B0_CONFIDENCE_EQUAL_K']=F.equal_k_rule(p,qr,qp,b,ids,k,'raw_conf')
    reference['B1_MARGIN_EQUAL_K']=F.equal_k_rule(p,qr,qp,b,ids,k,'raw_margin')
    # Unseal evaluation labels only after predictions for all methods are fixed.
    y=np.array([CLASSES.index(c) for c in ev.label],dtype=int)
    group=ev.accident_group.astype(str).to_numpy()
    stats={name:F.metrics(y,b,pred) for name,pred in reference.items()}
    ci_vs_keep={name:paired_ci(group,y,pred,b,seed=SEED+iteration)
                for name,pred in reference.items() if name!='KEEP'}
    ci_vs_abl=paired_ci(group,y,conditioned,global_ablation,seed=SEED+iteration)
    ci_vs_p23=paired_ci(group,y,conditioned,reference['P23_FEEDBACK_MEMORY'],seed=SEED+iteration)
    return {'feedback_n':len(train),'eval_n':len(ev),'actual_groups':int(len(set(group))),
            'class_support':dict(zip(CLASSES,np.bincount(y,minlength=6).tolist())),
            'conditional_selection':cond_audit,'global_selection':global_audit,
            'stats':stats,'ci_95_vs_keep_pp':ci_vs_keep,
            'ci_95_vs_global_pp':ci_vs_abl,'ci_95_vs_p23_pp':ci_vs_p23,
            'pair_counts':{name:category_pair_counts(b,pred) for name,pred in reference.items()},
            'weak_class_support':bool(np.bincount(y,minlength=6).min()<30),
            'claim_tier':'RETROSPECTIVE_EXPLORATORY_NOT_INDEPENDENT'},reference,b,y


def run(args):
    inp=Path(args.input);out=Path(args.out);models_dir=Path(args.models_dir)
    if out.exists():raise FileExistsError(f'OUTPUT_EXISTS {out}')
    expected=args.expected_input_sha256 or EXPECTED_INPUT_SHA
    if file_sha(inp)!=expected:raise ValueError('INPUT_SHA256_MISMATCH')
    for name,digest in MODEL_SHA.items():
        if file_sha(models_dir/name)!=digest:raise ValueError('FROZEN_MODEL_SHA_MISMATCH:'+name)
    d=pd.read_csv(inp,dtype={'row_id':str,'accident_group':str},low_memory=False)
    P.check_data(d)
    years=sorted(d.incident_year.astype(int).unique().tolist())
    if years != [2024,2025,2026]:raise ValueError('EXPECTED_LOCKED_2024_2025_2026_WINDOWS')
    # P23 data have NOT been independently tested for true label release times.
    if args.mode!='retrospective':raise ValueError('ONLY_RETROSPECTIVE_MODE_ALLOWED_WITH_CURRENT_EVIDENCE')
    if len(d)!=1160 or (d.incident_year.astype(int)==2026).sum()!=40:
        raise ValueError('ROW_COUNT_OR_WINDOW_IDENTITY_MISMATCH')
    all_groups=[set(d.loc[d.incident_year==year,'accident_group'].astype(str)) for year in years]
    for i in range(3):
        for j in range(i+1,3):
            if all_groups[i]&all_groups[j]:raise ValueError('CROSS_WINDOW_GROUP_COLLISION')
    models={name:joblib.load(models_dir/name) for name in MODEL_SHA}
    d=d.sort_values(['incident_year','row_id'],kind='stable').reset_index(drop=True)
    windows={year:d[d.incident_year==year].reset_index(drop=True) for year in years}
    reports={};preds={};ys={};bases={}
    for step,target in enumerate([2025,2026],start=1):
        feedback=d[d.incident_year<target].reset_index(drop=True)
        reports[target],preds[target],bases[target],ys[target]=evaluate(feedback,windows[target],models,step)
        reports[target]['feedback_window']=[y for y in years if y<target]
        reports[target]['eval_window']=target
    # Does the additional 2025 feedback actually change 2026 predictions?
    pred_before,pred_after=prospective_change(windows[2024],d[d.incident_year<2026],windows[2026],models)
    second_change=int(np.count_nonzero(pred_before!=pred_after))
    second_benefit=int(((pred_after==ys[2026]).sum()-(pred_before==ys[2026]).sum()))
    pooled={}
    for name in preds[2025]:
        base=np.concatenate([bases[2025],bases[2026]])
        pred=np.concatenate([preds[2025][name],preds[2026][name]])
        y=np.concatenate([ys[2025],ys[2026]])
        pooled[name]=F.metrics(y,base,pred)
    yall=np.concatenate([ys[2025],ys[2026]])
    groups=np.concatenate([windows[y].accident_group.astype(str).to_numpy() for y in [2025,2026]])
    full=np.concatenate([preds[y]['P24_CONDITIONAL'] for y in [2025,2026]])
    global_pred=np.concatenate([preds[y]['ABLATION_GLOBAL_PRIOR'] for y in [2025,2026]])
    prior=np.concatenate([preds[y]['P23_FEEDBACK_MEMORY'] for y in [2025,2026]])
    cis={'P24_vs_KEEP':paired_ci(groups,yall,full,np.concatenate([bases[2025],bases[2026]])),
         'P24_vs_GLOBAL':paired_ci(groups,yall,full,global_pred),
         'P24_vs_P23':paired_ci(groups,yall,full,prior)}
    # Deterministic status based on pre-registered replay thresholds.
    status='CONSERVATIVE_CONDITIONAL_PROFILE'
    if (pooled['P24_CONDITIONAL']['net_correct']>0 and
        pooled['P24_CONDITIONAL']['net_correct']>pooled['ABLATION_GLOBAL_PRIOR']['net_correct'] and
        pooled['P24_CONDITIONAL']['net_correct']>pooled['P23_FEEDBACK_MEMORY']['net_correct'] and
        second_change>0 and second_benefit>0 and
        cis['P24_vs_GLOBAL'][0]>0):
        status='EXPLORATORY_CONDITIONAL_POSITIVE'
    result={'status':status,'created_at':datetime.now(timezone.utc).isoformat(),
        'dataset_identity':{'input_sha256':expected,'rows':len(d),'windows':{str(y):len(windows[y]) for y in years}},
        'design':{'method':'signed-conditional-prototype-memory','k_neighbours':K_NEIGHBORS,
                  'prior_strength':PRIOR_STRENGTH,'budget':BUDGET,'scope':'RETROSPECTIVE_HISTORICAL_REPLAY',
                  'protocol_notes':['Evaluated on the frozen 2024+ historical replay panel',
                  'Retrospective annual-window protocol without verified label_available_at timestamps',
                  '2026 window (n=40) is descriptive',
                  'Report-number grouping used for paired cluster bootstrap']},
        'year_results':{str(year):reports[year] for year in [2025,2026]},
        'pooled':pooled,'pooled_paired_ci_95_pp':cis,
        'second_feedback_effect':{'num_decisions_changed':second_change,
                                 'net_correct_difference_in_2026':second_benefit,
                                 'note':'2026 window (n=40); descriptive historical replay summary'}}
    out.mkdir(parents=True,exist_ok=False)
    (out/'P24_RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=P.json_default),encoding='utf8')
    report=(f'# P24 Conditional Repair Replay Summary\n\n- Status: **{status}**\n'
           f'- P24 conditional pooled net gain: **{pooled["P24_CONDITIONAL"]["net_correct"]}**; '
           f'global-prior net: **{pooled["ABLATION_GLOBAL_PRIOR"]["net_correct"]}**; '
           f'P23 feedback-memory net: **{pooled["P23_FEEDBACK_MEMORY"]["net_correct"]}**.\n'
           f'- Second feedback update changed **{second_change}** decisions in the 2026 window (net impact: {second_benefit}).\n'
           '- Protocol note: retrospective exploratory replay on frozen predictions; online deployment requires prospective validation.\n')
    (out/'P24_DECISION.md').write_text(report,encoding='utf8')
    print(report)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True)
    ap.add_argument('--models-dir',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--mode',default='retrospective',choices=['retrospective'])
    ap.add_argument('--expected-input-sha256',default=EXPECTED_INPUT_SHA)
    run(ap.parse_args())

if __name__=='__main__': main()
