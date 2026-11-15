import unittest
import numpy as np
from contextual_bandits.evaluate import primary_contrasts
from contextual_bandits.simulator import generate,SCENARIOS
from contextual_bandits.policies import METHODS
class AnalysisContracts(unittest.TestCase):
 def test_contrasts_preserve_pairing_and_world_unit(self):
  rows=[]
  for w in range(3):
   for s in SCENARIOS:
    for m in METHODS:
     value=100*w+SCENARIOS.index(s)
     value+=dict(ucb=2,sw_ucb=5,discount_ucb=1,ts=4,softmax=7,dr_softmax=8,
                 sw_softmax=3,sw_dr_softmax=2,epsilon=6,context_free=2).get(m,0)
     if m=='ts' and s=='abrupt_high':value+=.5
     rows.append(dict(world=w,scenario=s,method=m,expected_mean=value))
  names,a=primary_contrasts(rows,'ucb')
  self.assertEqual(a.shape,(3,7))
  np.testing.assert_allclose(a,np.tile([2,3,-1,.5,1,-1,4],(3,1)))
  self.assertEqual(names[3],'ts_noise_interaction')
 def test_quality_controls_match_their_interpretation(self):
  for s in ('birth_strong','birth_inferior','birth_tie','birth_contextual','no_context'):
   e=generate(s,991,0,365);b=e.births[0][0]
   if s=='no_context':
    np.testing.assert_allclose(e.mu[:,:4],np.broadcast_to(e.mu[0,:4],(365,4)))
   elif s=='birth_inferior':np.testing.assert_allclose(e.mu[b:,4]-e.mu[b:,:4].min(axis=1),-.1)
   elif s=='birth_contextual':
    expected=np.array([.1 if c['currency_pair'] in ('EUR/USD','USD/JPY') else -.2 for c in e.contexts[b:]])
    np.testing.assert_allclose(e.mu[b:,4]-e.mu[b:,:4].max(axis=1),expected)
   else:np.testing.assert_allclose(e.mu[b:,4]-e.mu[b:,:4].max(axis=1),.1 if s=='birth_strong' else .001)
