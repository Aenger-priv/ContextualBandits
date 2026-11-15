"""Independent checks against analytic references using the mixed-context code."""
import unittest
from dataclasses import replace
import numpy as np
from contextual_bandits.policies import Policy, RidgeState, dr_outcomes, softmax, METHODS
from contextual_bandits.simulator import generate
from contextual_bandits.evaluate import run, cold_metrics

class PolicyContracts(unittest.TestCase):
 def test_twelve_dimensional_ridge_window_and_discount(self):
  env=generate('recurring',91,0,365)
  for window,gamma in [(None,1.),(7,1.),(None,.9)]:
   state=RidgeState(5,12,window,gamma)
   history=[]
   for t in range(30):
    state.clock(t)
    for a in range(5):
     A=np.eye(12);b=np.zeros(12)
     for j,arm,x,y in history:
      if arm==a and (window is None or j>t-window):
       weight=gamma**(t-j)
       A+=weight*np.outer(x,x);b+=weight*y*x
     np.testing.assert_allclose(state.A[a],A,atol=1e-12)
     np.testing.assert_allclose(state.b[a],b,atol=1e-12)
     np.testing.assert_allclose(state.predict(env.x[t])[a],env.x[t]@np.linalg.solve(A,b),atol=1e-12)
    a=t%4;x=env.x[t];y=env.outcomes[t,a]
    state.add(t,x,[a],[y]);history.append((t,a,x,y))
 def test_dr_expectation_and_predictable_nuisance(self):
  probs=np.array([.1,.2,.3,.15,.25]);means=np.array([1,-.2,.7,.1,.4]);q=np.array([9,4,-1,2,0])
  np.testing.assert_allclose(sum(probs[a]*dr_outcomes(q,a,means[a],probs[a]) for a in range(5)),means,atol=1e-14)
  x=generate('stationary',91,0,365).x[0]
  p=Policy('dr_softmax',5,12,np.random.default_rng(2))
  a,prop=p.select(0,x,np.array([1,1,1,1,0],bool));self.assertEqual(prop,.25)
  p.observe(a,1.)
  np.testing.assert_allclose(p.state.b[a],4*x)
  np.testing.assert_allclose(p.nuisance.b[a],x)
  np.testing.assert_array_equal(p.state.A[4],np.eye(12))
  self.assertGreaterEqual(softmax(np.array([1e6,0,-1e6,2,3]),.001).min(),.01)
 def test_all_partial_policies_ignore_unchosen_and_future_outcomes(self):
  env=generate('birth_contextual',91,0,365)
  for m in METHODS:
   if m=='full_information':continue
   summary,a=run(env,m,{},92,0,'birth_contextual')
   changed=np.full_like(env.outcomes,1e8)
   changed[np.arange(365),a['actions']]=env.outcomes[np.arange(365),a['actions']]
   _,b=run(replace(env,outcomes=changed),m,{},92,0,'birth_contextual')
   np.testing.assert_array_equal(a['actions'],b['actions'])
   changed=env.outcomes.copy();changed[280:]=1e8
   _,b=run(replace(env,outcomes=changed),m,{},92,0,'birth_contextual')
   np.testing.assert_array_equal(a['actions'][:281],b['actions'][:281])
   self.assertTrue(np.all(env.available[np.arange(365),a['actions']]))
   self.assertAlmostEqual(summary['oracle_total'],a['expected'].sum()+a['regret'].sum())
 def test_ts_probability_against_gaussian_formula(self):
  from scipy.stats import norm
  actions=[]
  for seed in range(4000):
   p=Policy('ts',2,1,np.random.default_rng(seed),{'v':.25});p.state.b[1,0]=.3
   actions.append(p.select(0,np.ones(1),np.ones(2,bool))[0])
  self.assertAlmostEqual(np.mean(actions),norm.cdf(.3/np.sqrt(.5)),delta=.025)
 def test_cold_start_right_boundary(self):
  env=generate('birth_inferior',91,0,365);a=np.zeros(365,int);r=np.ones(365)
  m=cold_metrics(env,a,r)[0];self.assertEqual(m['delay'],100);self.assertFalse(m['discovered'])
  a[180]=4;self.assertEqual(cold_metrics(env,a,r)[0]['delay'],0)
  a[180]=0;a[279]=4;self.assertEqual(cold_metrics(env,a,r)[0]['delay'],99)
  a[279]=0;a[280]=4;self.assertFalse(cold_metrics(env,a,r)[0]['discovered'])
if __name__=='__main__':unittest.main()
