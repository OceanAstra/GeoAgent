import unittest
import numpy as np
import pandas as pd
import p23_feedback_replay as P

class FakeHead:
    def __init__(self,shift):self.shift=shift
    def predict_proba(self,x):
        # head_features starts with six LOGP columns
        a=np.exp(x.iloc[:,:6].to_numpy(float))
        out=(1-self.shift)*a+self.shift*np.roll(a,1,axis=1)
        return out/out.sum(axis=1,keepdims=True)

class FakeValue:
    def predict(self,x):
        return .02*(x.qraw_dest.to_numpy(float)-x.qraw_base.to_numpy(float))


def synthetic(n_per_year=70):
    rng=np.random.default_rng(2201)
    n=3*n_per_year
    p=rng.dirichlet(np.ones(6)*1.8,size=n)
    rows={
      'row_id':[f'R{i}' for i in range(n)],
      'accident_group':[f'G{i}' for i in range(n)],
      'incident_year':np.repeat([2024,2025,2026],n_per_year),
      'label':[P.CLASSES[i] for i in rng.integers(0,6,size=n)]}
    for j,c in enumerate(P.PROB_COLS):rows[c]=p[:,j]
    for c in P.F.NUM:rows[c]=rng.normal(10,2,size=n)
    for c in P.F.CAT:rows[c]=rng.choice(['A','B','C'],size=n)
    return pd.DataFrame(rows)

class P23Tests(unittest.TestCase):
    def test_check_data_valid(self):P.check_data(synthetic())
    def test_bad_probabilities_fail(self):
        d=synthetic();d.loc[0,P.PROB_COLS[0]]=3
        with self.assertRaisesRegex(ValueError,'BAD_PROBABILITIES'):P.check_data(d)
    def test_cross_year_group_fail(self):
        d=synthetic();d.loc[0,'accident_group']=d.iloc[-1].accident_group
        with self.assertRaisesRegex(ValueError,'CROSS_YEAR_GROUP'):P.check_data(d)
    def test_missing_release_date_strict_fail(self):
        with self.assertRaisesRegex(ValueError,'STRICT_REQUIRES'):
            P.release_audit(synthetic(),[2024,2025,2026],'strict')
    def test_prior_only_previous_labels(self):
        d=synthetic();train=d.iloc[:70].copy();ev=d.iloc[70:140].copy()
        fake={'P22_head_raw.joblib':FakeHead(.34),'P22_head_prob.joblib':FakeHead(.17),
              'P22_action_value.joblib':FakeValue()}
        _,preds1,base1=P.evaluate_one(train,ev,fake,1)
        ev2=ev.copy();ev2['label']=ev2['label'].iloc[::-1].to_numpy()
        _,preds2,base2=P.evaluate_one(train,ev2,fake,1)
        self.assertTrue(np.array_equal(base1,base2))
        for k in preds1:self.assertTrue(np.array_equal(preds1[k],preds2[k]),k)
    def test_future_year_labels_cannot_change_history(self):
        d=synthetic();past=d.iloc[:70].copy()
        one=P.feedback_mask(d,[2024]);self.assertEqual(one.sum(),70)
        altered=d.copy();altered.loc[altered.incident_year==2026,'label']=P.CLASSES[0]
        self.assertTrue(np.array_equal(one,P.feedback_mask(altered,[2024])))
    def test_two_feedback_updates_execute(self):
        d=synthetic(n_per_year=60)
        fake={'P22_head_raw.joblib':FakeHead(.34),'P22_head_prob.joblib':FakeHead(.17),
              'P22_action_value.joblib':FakeValue()}
        w1=d[d.incident_year==2024].reset_index(drop=True)
        w2=d[d.incident_year==2025].reset_index(drop=True)
        w3=d[d.incident_year==2026].reset_index(drop=True)
        result1,_,_=P.evaluate_one(w1,w2,fake,1)
        prior=d[d.incident_year.isin([2024,2025])].reset_index(drop=True)
        result2,_,_=P.evaluate_one(prior,w3,fake,2)
        self.assertEqual(result1['train_feedback_n'],60)
        self.assertEqual(result2['train_feedback_n'],120)
        self.assertIn('AGENT1_FEEDBACK_MEMORY',result1['stats'])
        self.assertIn('ORDINARY_FEEDBACK_LOGREG',result2['stats'])

    def test_prior_smoothing(self):
        a=pd.DataFrame({'base_class':['0','0','0'],'dest_class':['1','1','2']})
        states={(0,1):(2,2.0),(0,2):(1,-1.0)}
        val=P.score_profile(a,states)
        self.assertTrue(val[0]>0 and val[2]<0)
        self.assertLess(val[0],.10)
    def test_bootstrap_identity(self):
        y=np.array([0,1,1,0]);g=np.array(['a','a','b','b'])
        self.assertEqual(P.paired_group_ci(g,y,y,y),[0.,0.])

if __name__=='__main__':unittest.main()
