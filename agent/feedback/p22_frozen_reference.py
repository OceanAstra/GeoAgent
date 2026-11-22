#!/usr/bin/env python3
"""P22 one-shot, source-grounded, development-only candidate-action value gate.

A: fit P21-style six-class heads; B: fit action utility; C: fixed evaluation.
No held-out/sealed data, no re-training frozen GeOAD or Agent1 core.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CLASSES = ('CORROSION','EQUIPMENT_FAILURE','EXTERNAL_DAMAGE',
           'MATERIAL_DEFECT','OPERATIONAL','OTHER')
PROB_COLS = [f'prob_{c}' for c in CLASSES]
OLD_CAND_COLS = [f'candidate_pred_{c}' for c in
    ('P8_R3','PBR_MILD','PBR_FULL','P_ONLY_MILD','CONTROL')]
NUM = ('PIPE_DIAMETER','PIPE_WALL_THICKNESS','PIPE_SMYS','ACCIDENT_PSIG',
       'MOP_PSIG','INTERNAL_INSPECTION_IND','PIPE_AGE_AT_ACCIDENT',
       'MANUFACTURE_AGE_AT_ACCIDENT','PRESSURE_RATIO')
CAT = ('PIPE_SPECIFICATION','PIPE_SEAM_TYPE','PIPE_COATING_TYPE','PIPE_TYPE')
EXPECTED_SHA = 'e82c49428e70f437729ce0dd78603c7c17a6107f4bf981f1348b70049caf38e0'
SEED_NAMESPACE = 'ZHL_AGENT1_P22_20261114_LOCKED_V1'
FORBIDDEN = ('sealed','holdout','heldout','do_not_open','1065')
FEATURES_NUM = ([f'p_{i}' for i in range(6)] +
                [f'qraw_{i}' for i in range(6)] +
                [f'qprob_{i}' for i in range(6)] +
                ['p_base','p_dest','qraw_base','qraw_dest','qprob_base',
                 'qprob_dest','p_entropy','p_max','p_gap','action_rank'])
FEATURES_CAT = ['base_class','dest_class','source']


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def validate_input(path, alignment):
    """Match P21 verified artifact. Reject any unauthorized replacement."""
    for item in (path,alignment):
        if any(x in str(Path(item).resolve()).lower() for x in FORBIDDEN):
            raise ValueError('SEALED_OR_HOLDOUT_PATH_REFUSED')
    if sha256(path)!=EXPECTED_SHA: raise ValueError('INPUT_SHA_MISMATCH')
    receipt=json.loads(Path(alignment).read_text(encoding='utf8'))
    if receipt.get('output_sha256')!=EXPECTED_SHA or receipt.get('n_rows')!=6417:
        raise ValueError('ALIGNMENT_RECEIPT_MISMATCH')
    d=pd.read_csv(path,dtype={'row_id':str},low_memory=False)
    need = ['row_id','split','label','accident_group']+PROB_COLS+list(NUM)+list(CAT)+OLD_CAND_COLS
    missing=[c for c in need if c not in d]
    if missing: raise ValueError(f'MISSING_COLUMNS:{missing}')
    if len(d)!=6417 or d.row_id.isna().any() or d.row_id.duplicated().any():
        raise ValueError('ROW_ID_INVALID')
    if d.split.isna().any() or set(d.split.astype(str))!={'dev_confirm'}:
        raise ValueError('DEV_CONFIRM_ONLY')
    if d.accident_group.isna().any(): raise ValueError('MISSING_GROUP')
    y=pd.to_numeric(d.label,errors='raise').to_numpy(dtype=int)
    if not np.isin(y,np.arange(6)).all(): raise ValueError('LABEL_SPACE_INVALID')
    p=d[PROB_COLS].to_numpy(dtype=float)
    if not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.allclose(p.sum(axis=1),1,atol=1e-5):
        raise ValueError('INVALID_PROBS')
    base=p.argmax(axis=1)
    if abs(np.mean(base==y)-0.7159108617734143)>1e-7:
        raise ValueError('FROZEN_BASELINE_MISMATCH')
    old=d[OLD_CAND_COLS].to_numpy(dtype=float)
    if not np.isfinite(old).all() or not np.isin(old,np.arange(6)).all():
        raise ValueError('INVALID_OLD_CANDIDATES')
    if int(((base!=y)&(old==y[:,None]).any(axis=1)).sum())!=163:
        raise ValueError('HISTORICAL_CANDIDATE_MISMATCH')
    return d,y,p,base


def assign_split(groups):
    """Group-hash fixed before reading labels. 60/20/20 A/B/C."""
    out=[]
    for g in groups:
        digest=hashlib.sha256((SEED_NAMESPACE+':'+str(g)).encode('utf8')).digest()
        u=int.from_bytes(digest[:8],'big')/2**64
        out.append('A' if u<.60 else ('B' if u<.80 else 'C'))
    return np.asarray(out)


def verify_splits(split, groups, y, min_class=12):
    for stage in 'ABC':
        yi=y[split==stage]
        if len(yi)==0 or min(np.bincount(yi,minlength=6))<min_class:
            raise ValueError(f'INSUFFICIENT_PER_CLASS_{stage}')
    fam={stage:set(np.asarray(groups,dtype=str)[split==stage]) for stage in 'ABC'}
    if (fam['A']&fam['B']) or (fam['A']&fam['C']) or (fam['B']&fam['C']):
        raise ValueError('GROUP_LEAKAGE')


def head_features(df,p,raw):
    x=pd.DataFrame(np.log(np.clip(p,1e-8,1.0)),columns=[f'LOGP_{c}' for c in CLASSES],index=df.index)
    if raw:
        for c in NUM: x[c]=pd.to_numeric(df[c],errors='coerce')
        for c in CAT: x[c]=df[c].fillna('__MISSING__').astype(str)
    return x


def make_head(raw):
    num=[f'LOGP_{c}' for c in CLASSES]+(list(NUM) if raw else [])
    blocks=[('num',Pipeline([('impute',SimpleImputer(strategy='median')),
                             ('scale',StandardScaler())]),num)]
    if raw:
        blocks.append(('cat',OneHotEncoder(handle_unknown='ignore'),list(CAT)))
    return Pipeline([('pre',ColumnTransformer(blocks)),
        ('model',LogisticRegression(C=0.1,solver='lbfgs',max_iter=1200))])


def fit_heads(d,y,p,split):
    """Only A labels are visible to candidate training."""
    heads={};probs={}
    for name,raw in [('raw',True),('prob',False)]:
        x=head_features(d,p,raw)
        model=make_head(raw)
        model.fit(x.loc[split=='A'],y[split=='A'])
        if not np.array_equal(model.named_steps['model'].classes_,np.arange(6)):
            raise ValueError('HEAD_TRAIN_CLASS_MISSING')
        heads[name]=model
        probs[name]=model.predict_proba(x)
    return heads,probs


def create_actions(p,qraw,qprob,base,rows):
    """Propose raw top-2 non-base and prob-only top-1 non-base, dedup by dest.
       All inputs are prediction-time features; no y argument by design.
    """
    eps=1e-12
    out=[]
    for i in rows:
        bi=int(base[i]); seen=set()
        order_raw=np.argsort(-qraw[i],kind='stable')
        order_prob=np.argsort(-qprob[i],kind='stable')
        selections=( [('raw_top1',int(c),1) for c in order_raw if c!=bi][:1]
            + [('raw_top2',int(c),2) for c in order_raw if c!=bi][1:2]
            + [('prob_top1',int(c),1) for c in order_prob if c!=bi][:1] )
        for source,dest,rank in selections:
            if dest in seen: continue
            seen.add(dest)
            pi=p[i]; qr=qraw[i]; qp=qprob[i]
            rec={'idx':int(i),'dest':dest,'source':source,'base_class':str(bi),
                 'dest_class':str(dest),'action_rank':float(rank),
                 'p_base':float(pi[bi]),'p_dest':float(pi[dest]),
                 'qraw_base':float(qr[bi]),'qraw_dest':float(qr[dest]),
                 'qprob_base':float(qp[bi]),'qprob_dest':float(qp[dest]),
                 'p_entropy':float(-(pi*np.log(np.clip(pi,eps,1))).sum()),
                 'p_max':float(pi.max()),
                 'p_gap':float(np.sort(pi)[-1]-np.sort(pi)[-2])}
            for j in range(6):
                rec[f'p_{j}']=float(pi[j]);rec[f'qraw_{j}']=float(qr[j]);rec[f'qprob_{j}']=float(qp[j])
            out.append(rec)
    actions=pd.DataFrame.from_records(out)
    if actions.empty: raise ValueError('NO_ACTIONS_GENERATED')
    if (actions.dest.to_numpy()==base[actions.idx.to_numpy()]).any():
        raise ValueError('ACTION_EQUALS_KEEP')
    if actions.duplicated(['idx','dest']).any():raise ValueError('ACTION_DEDUP_FAIL')
    return actions


def delta_labels(actions,y,base):
    i=actions.idx.to_numpy(dtype=int)
    dest=actions.dest.to_numpy(dtype=int)
    return (dest==y[i]).astype(int)-(base[i]==y[i]).astype(int)


def fit_value_model(actions,delta):
    if len(actions)!=len(delta):raise ValueError('ACTION_LABEL_LENGTH_MISMATCH')
    if not np.isin(delta,[-1,0,1]).all():raise ValueError('DELTA_INVALID')
    processor=ColumnTransformer([
        ('num',Pipeline([('impute',SimpleImputer(strategy='median')),
                         ('scale',StandardScaler())]),FEATURES_NUM),
        ('cat',OneHotEncoder(handle_unknown='ignore'),FEATURES_CAT)])
    model=Pipeline([('pre',processor),('ridge',Ridge(alpha=50.0))])
    model.fit(actions[FEATURES_NUM+FEATURES_CAT],delta)
    return model


def value_policy(actions,score,base,row_ids,budget=0.10,allowed_sources=None):
    """Keep unless learned E[delta] >0. No y used. At most one action/row."""
    if len(actions)!=len(score):raise ValueError('SCORES_BAD_LENGTH')
    if not np.isfinite(score).all():raise ValueError('SCORES_NOT_FINITE')
    a=actions.copy();a['gain']=score
    a=a[a.gain>0]
    if allowed_sources is not None:a=a[a.source.isin(allowed_sources)]
    a['row_id_sort']=np.asarray(row_ids,dtype=str)[a.idx.to_numpy(dtype=int)]
    a=a.sort_values(['gain','row_id_sort','dest'],ascending=[False,True,True])
    a=a.drop_duplicates('idx',keep='first')
    cap=int(np.ceil(budget*len(base)))
    a=a.iloc[:cap]
    pred=base.copy()
    pred[a.idx.to_numpy(dtype=int)]=a.dest.to_numpy(dtype=int)
    return pred,a


def fixed_q_rule(q,base,row_ids,margin_min=0.05,budget=0.10):
    alt=q.argmax(axis=1)
    mar=q[np.arange(len(q)),alt]-q[np.arange(len(q)),base]
    ok=np.flatnonzero((alt!=base)&(mar>margin_min))
    ids=np.asarray(row_ids,dtype=str)
    order=ok[np.lexsort((ids[ok],-mar[ok]))]
    order=order[:int(np.ceil(budget*len(base)))]
    pred=base.copy();pred[order]=alt[order]
    return pred


def equal_k_rule(p,qraw,qprob,base,row_ids,k,rank_mode):
    """Pre-fixed non-learning controls with identical K, no true label."""
    n=len(base)
    if k<=0:return base.copy()
    if k>n:raise ValueError('K_GT_ROWS')
    q=qprob if rank_mode=='prob_conf' else qraw
    alt=np.argsort(-q,axis=1,kind='stable')
    dest=np.array([next(int(c) for c in alt[i] if c!=base[i]) for i in range(n)])
    if rank_mode in ('raw_conf','prob_conf'):
        rank_score=1.0-p.max(axis=1)
    elif rank_mode=='raw_margin':
        rank_score=q[np.arange(n),dest]-q[np.arange(n),base]
    else: raise ValueError('UNKNOWN_BASELINE')
    ids=np.asarray(row_ids,dtype=str)
    ix=np.lexsort((ids,-rank_score))[:k]
    pred=base.copy();pred[ix]=dest[ix]
    return pred


def metrics(y,base,pred):
    recalls0=recall_score(y,base,labels=np.arange(6),average=None,zero_division=0)
    recalls1=recall_score(y,pred,labels=np.arange(6),average=None,zero_division=0)
    return {'changed':int((base!=pred).sum()),
       'fixes':int(((base!=y)&(pred==y)).sum()),
       'breaks':int(((base==y)&(pred!=y)).sum()),
       'wrong_to_wrong':int(((base!=y)&(pred!=y)&(base!=pred)).sum()),
       'net_correct':int((pred==y).sum()-(base==y).sum()),
       'accuracy':float(accuracy_score(y,pred)),
       'macro_f1':float(f1_score(y,pred,average='macro',labels=np.arange(6),zero_division=0)),
       'class_support':{c:int((y==j).sum()) for j,c in enumerate(CLASSES)},
       'recall_delta_pp':{c:float(100*(a-b)) for c,a,b in zip(CLASSES,recalls1,recalls0)}}


def paired_bootstrap(y,pred_a,pred_b,groups,iterations=500):
    _,code=np.unique(np.asarray(groups,dtype=str),return_inverse=True)
    ng=int(code.max()+1)
    corr=(pred_a==y).astype(int)-(pred_b==y).astype(int)
    gain=np.bincount(code,weights=corr,minlength=ng)
    count=np.bincount(code,minlength=ng)
    rng=np.random.default_rng(22)
    samples=[]
    for _ in range(iterations):
        pick=rng.integers(0,ng,size=ng)
        samples.append(float(gain[pick].sum()/count[pick].sum()))
    return [float(v) for v in np.quantile(samples,[0.025,0.975])]


def run(d,y,p,base,strict=True):
    groups=d.accident_group.astype(str).to_numpy();sp=assign_split(groups)
    verify_splits(sp,groups,y,min_class=12 if strict else 1)
    heads,q=fit_heads(d,y,p,sp)
    ib=np.flatnonzero(sp=='B');ic=np.flatnonzero(sp=='C')
    act_b=create_actions(p,q['raw'],q['prob'],base,ib)
    act_c=create_actions(p,q['raw'],q['prob'],base,ic)
    train_delta=delta_labels(act_b,y,base)
    value=fit_value_model(act_b,train_delta)
    pred_gain=value.predict(act_c[FEATURES_NUM+FEATURES_CAT])
    local_actions=act_c.copy();local_actions['idx']=np.searchsorted(ic,act_c.idx.to_numpy())
    # This is valid since ic and act_c.idx are both sorted global indices; verify exact mapping.
    if not np.array_equal(ic[local_actions.idx.to_numpy()],act_c.idx.to_numpy()):
        raise ValueError('LOCAL_INDEX_MAPPING_ERROR')
    bc=base[ic];yc=y[ic];pc=p[ic];ids=d.row_id.to_numpy(dtype=str)[ic]
    qrc=q['raw'][ic];qpc=q['prob'][ic]
    gated,selected=value_policy(local_actions,pred_gain,bc,ids)
    gated_raw,_=value_policy(local_actions,pred_gain,bc,ids,allowed_sources=['raw_top1'])
    K=int((gated!=bc).sum())
    policies={
        'KEEP':bc.copy(),
        'P22_VALUE_GATE_POOL':gated,
        'P22_GATE_RAW_TOP1_ONLY':gated_raw,
        'P21_FIXED_RAW_RULE_REBUILT':fixed_q_rule(qrc,bc,ids),
        'PROB_ONLY_FIXED_RULE_REBUILT':fixed_q_rule(qpc,bc,ids),
        'B0_CONFIDENCE_EQUAL_K':equal_k_rule(pc,qrc,qpc,bc,ids,K,'raw_conf'),
        'B1_QRAW_MARGIN_EQUAL_K':equal_k_rule(pc,qrc,qpc,bc,ids,K,'raw_margin'),
        'B2_QPROB_CONF_EQUAL_K':equal_k_rule(pc,qrc,qpc,bc,ids,K,'prob_conf')}
    results={k:metrics(yc,bc,v) for k,v in policies.items()}
    # Oracle: ONLY posthoc evaluation and never a runnable selection strategy.
    repairable=(bc!=yc)&np.isin(np.arange(len(ic)),local_actions[local_actions.dest.to_numpy()==yc[local_actions.idx.to_numpy()]].idx)
    result={
      'claim_tier':'REUSED_DEV_EXPLORATORY_ONLY',
      'label_space':list(CLASSES),
      'split_sizes':{a:int((sp==a).sum()) for a in 'ABC'},
      'split_class_counts':{a:{CLASSES[j]:int((y[sp==a]==j).sum()) for j in range(6)} for a in 'ABC'},
      'n_groups':{a:int(len(set(groups[sp==a]))) for a in 'ABC'},
      'b_action_counts':{'positive':int((train_delta==1).sum()),'negative':int((train_delta==-1).sum()),'neutral':int((train_delta==0).sum())},
      'c_candidate_rows':int(len(act_c)),
      'c_posthoc_repairable_errors':int(repairable.sum()),
      'c_base_errors':int((bc!=yc).sum()),
      'actions_selected':int(K),
      'policies':results,
      'ci_exploratory_accuracy_gain_vs_keep':paired_bootstrap(yc,gated,bc,groups[ic]),
      'ci_exploratory_accuracy_gain_vs_confK':paired_bootstrap(yc,gated,policies['B0_CONFIDENCE_EQUAL_K'],groups[ic]),
      'ci_exploratory_accuracy_gain_vs_qmarginK':paired_bootstrap(yc,gated,policies['B1_QRAW_MARGIN_EQUAL_K'],groups[ic]),
      'criteria_descriptive':{
         'net_vs_KEEP_positive':bool(results['P22_VALUE_GATE_POOL']['net_correct']>0),
         'better_than_both_equal_k':bool(results['P22_VALUE_GATE_POOL']['net_correct']>
           max(results['B0_CONFIDENCE_EQUAL_K']['net_correct'],results['B1_QRAW_MARGIN_EQUAL_K']['net_correct'])),
         'macro_f1_not_below_KEEP':bool(results['P22_VALUE_GATE_POOL']['macro_f1']>=results['KEEP']['macro_f1']),
         'no_class_recall_drop_below_minus3pp':bool(all(v>=-3 for v in results['P22_VALUE_GATE_POOL']['recall_delta_pp'].values())),
         'zero_change':bool(K==0)},
      'warnings':['Development-only. This dev_confirm was reused by earlier P1-P21 designs, not an independent test.',
                  'A/B/C separation prevents within-run target leakage but not historical adaptive overfitting.',
                  'Old five candidates excluded from primary due to unproven training independence from B/C labels.',
                  'The action learner is standard expected-utility regression, NOT a standalone novel published algorithm.',
                  '3pp recall limit is descriptive, not formal G3 or statistical guarantee.']}
    return result,policies,ic,selected,heads,value


def main():
    ap=argparse.ArgumentParser()
    # No absolute server defaults: this standalone entry is not used by the P23
    # replay (which passes its own --input/--models-dir) and must not reach into
    # a machine-specific directory.
    ap.add_argument('--input',default=None,help='P20 candidate input CSV (required for this standalone entry)')
    ap.add_argument('--alignment',default=None,help='P20 alignment JSON (required for this standalone entry)')
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    if not args.input or not args.alignment:
        ap.error('--input and --alignment are required (no server-side defaults)')
    out=Path(args.out).resolve()
    if out.exists():raise FileExistsError('OUTPUT_MUST_BE_NEW')
    if any(x in str(out).lower() for x in FORBIDDEN):raise ValueError('FORBIDDEN_OUTPUT')
    d,y,p,base=validate_input(args.input,args.alignment)
    result,predictions,ic,selected,heads,value=run(d,y,p,base)
    out.mkdir(parents=True)
    (out/'REPORT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    prediction=pd.DataFrame({'row_id':d.row_id.to_numpy()[ic],
        'label_DEV_ONLY':y[ic],'base_pred':base[ic]})
    for k,v in predictions.items():prediction[k]=v
    prediction.to_csv(out/'PREDICTIONS_C_DEV_ONLY.csv',index=False)
    # Model artifacts stay on server. Do not send to third-party services.
    import joblib
    for name,head in heads.items():joblib.dump(head,out/f'P22_head_{name}.joblib')
    joblib.dump(value,out/'P22_action_value.joblib')
    prov={'input_path':str(Path(args.input).resolve()),'input_sha256':EXPECTED_SHA,
          'alignment_sha256':sha256(args.alignment),
          'report_sha256':sha256(out/'REPORT.json'),
          'predictions_sha256':sha256(out/'PREDICTIONS_C_DEV_ONLY.csv'),
          'model_sha256':{f.name:sha256(f) for f in out.glob('*.joblib')},
          'read_only_originals':True,'sealed_accessed':False,'new_training':'candidate_A_and_value_B_only'}
    (out/'PROVENANCE.json').write_text(json.dumps(prov,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    main_policy=result['policies']['P22_VALUE_GATE_POOL']
    decision_lines=['# P22 One-Shot Action-Value Reference — Development Summary', '',
       '> Development exploration only; not an independent blind test.', '',
       f"A/B/C split sizes: {result['split_sizes']}",
       f"Main policy accepted revisions: {main_policy['changed']}",
       f"Corrected / harmed / net: {main_policy['fixes']}/{main_policy['breaks']}/{main_policy['net_correct']}",
       f"Accuracy: {main_policy['accuracy']:.6f} | Macro-F1: {main_policy['macro_f1']:.6f}",
       f"KEEP net correct: {result['policies']['KEEP']['net_correct']}",
       f"Equal-K confidence net correct: {result['policies']['B0_CONFIDENCE_EQUAL_K']['net_correct']}",
       f"Equal-K margin net correct: {result['policies']['B1_QRAW_MARGIN_EQUAL_K']['net_correct']}",
       f"Descriptive criteria: {result['criteria_descriptive']}", '',
       'Development results serve as frozen reference artifacts for subsequent replay evaluation without post-hoc retuning.',
       'If zero revisions are accepted, report conservative abstention with no accuracy gain.',
       'Full metrics are recorded in REPORT.json; model artifacts are saved in this output directory.']
    (out/'DECISION.md').write_text('\n'.join(decision_lines)+'\n',encoding='utf8')
    print(json.dumps({'out':str(out),'split_sizes':result['split_sizes'],
        'main_policy':result['policies']['P22_VALUE_GATE_POOL'],
        'criteria':result['criteria_descriptive']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
