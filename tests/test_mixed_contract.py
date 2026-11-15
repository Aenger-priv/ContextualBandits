import json, unittest
from pathlib import Path
import numpy as np
from contextual_bandits.simulator import generate,encode,from_records,FEATURE_NAMES,SCENARIOS
from contextual_bandits.interface import ContextPolicy
class MixedContract(unittest.TestCase):
 def test_original_records_accepted(self):
  records=json.loads(Path('fx_trading_dataset.json').read_text())
  w=from_records(records,rewards_are_means=True)
  self.assertEqual(w.x.shape,(18250,12));self.assertEqual(w.births,((9000,4),))
  self.assertEqual(w.available[0].sum(),4);self.assertEqual(w.available[-1].sum(),5)
 def test_encoder_independent_reference(self):
  c={'currency_pair':'GBP/USD','volatility':.096,'size':390,'time_of_day':'pre-close','date':'2023-01-01'}
  expected=[1,0,1,0,0,0,0,0,0,1,.192,.195]
  np.testing.assert_allclose(encode(c),expected)
  d=dict(c,date='2024-09-01');np.testing.assert_array_equal(encode(c),encode(d))
  with self.assertRaises(ValueError):encode(dict(c,currency_pair='UNDECLARED'))
 def test_schema_roundtrip_and_four_to_five(self):
  for scenario in SCENARIOS:
   w=generate(scenario,1234,0,horizon=365)
   r=w.records();z=from_records(json.loads(json.dumps(r)),w.mu)
   self.assertEqual(w.digest(),z.digest());self.assertEqual(w.births,((180,4),))
   self.assertEqual(set(r[179]['rewards']),{'0','1','2','3'})
   self.assertEqual(set(r[180]['rewards']),{'0','1','2','3','4'})
   self.assertEqual(set(r[0]),{'context','rewards'})
 def test_noise_pairing_and_fresh_worlds(self):
  a=generate('abrupt_low',1234,0,365);b=generate('abrupt_high',1234,0,365)
  np.testing.assert_array_equal(a.x,b.x);np.testing.assert_array_equal(a.mu,b.mu)
  np.testing.assert_allclose(b.outcomes-b.mu,6*(a.outcomes-a.mu),atol=1e-14)
  self.assertNotEqual(a.digest(),generate('abrupt_low',1234,1,365).digest())
 def test_online_boundary_and_oracle_semantics(self):
  w=generate('stationary',1234,0,365)
  p=ContextPolicy('ucb')
  a,prob=p.select(w.contexts[0],[0,1,2,3]);self.assertLess(a,4)
  p.observe(a,float(w.outcomes[0,a]));self.assertEqual(p.clock,1)
  with self.assertRaises(ValueError):from_records(w.records())
if __name__=='__main__':unittest.main()
