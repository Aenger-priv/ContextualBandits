import unittest,tempfile,json
from pathlib import Path
import numpy as np
import pandas as pd
from contextual_bandits.workbench import make_world,dataset_path,run_experiment,world_dates,Experiment,save_experiment,load_experiment,plot_world,plot_choices
from contextual_bandits.evaluate import run
class NotebookContracts(unittest.TestCase):
 def setUp(self):
  self.datasets=tempfile.TemporaryDirectory()
  self.addCleanup(self.datasets.cleanup)
 def test_generation_saves_original_json_without_overwriting_edits(self):
  w=make_world('recurring',80001,0,365,dataset_parent=self.datasets.name)
  p=dataset_path(w,self.datasets.name)
  self.assertEqual(json.loads(p.read_text()),w.records())
  self.assertIn('\n  {\n',p.read_text())
  rows=json.loads(p.read_text())
  self.assertEqual(set(rows[179]['rewards']),{'0','1','2','3'})
  self.assertEqual(set(rows[180]['rewards']),{'0','1','2','3','4'})
  np.testing.assert_array_equal(json.loads(p.with_name('expected_rewards.json').read_text()),w.mu)
  other=make_world('recurring',80001,0,365,dataset_parent=self.datasets.name)
  self.assertEqual(dataset_path(other,self.datasets.name),p)
  p.write_text('user analysis edits')
  with self.assertRaises(FileExistsError):
   make_world('recurring',80001,0,365,dataset_parent=self.datasets.name)
  self.assertEqual(p.read_text(),'user analysis edits')
 def test_monte_carlo_saves_each_world_before_policy_evaluation(self):
  from unittest.mock import patch
  def check_saved(world,*args):
   self.assertEqual(json.loads(dataset_path(world,self.datasets.name).read_text()),world.records())
   return run(world,*args)
  with patch('contextual_bandits.workbench.run',side_effect=check_saved):
   exp=run_experiment(['recurring'],['ucb'],{'ucb':{}},{'recurring':0},
     repeats=2,horizon=365,progress=False,dataset_parent=self.datasets.name)
  paths=[Path(d['path']) for d in exp.metadata['datasets']]
  self.assertEqual(len(set(paths)),2)
  self.assertNotEqual(json.loads(paths[0].read_text()),json.loads(paths[1].read_text()))
 def test_saved_raw_contexts_and_curves_roundtrip(self):
  w=make_world('recurring',80001,0,365,start_date='2024-02-01',dataset_parent=self.datasets.name)
  s,e=run(w,'ucb',{},80001,0,'recurring')
  exp=Experiment({(0,'recurring'):w},{(0,'recurring','ucb'):e},pd.DataFrame([dict(world=0,scenario='recurring',method='ucb',**s)]),{'repeats':1,'scenarios':['recurring'],'methods':['ucb']})
  fig=plot_choices(exp,'recurring')
  import matplotlib.dates as md
  self.assertAlmostEqual(fig.axes[0].get_xlim()[1],md.date2num(world_dates(w)[-1]+pd.Timedelta(days=1)))
  import matplotlib.pyplot as plt
  plt.close(fig)
  with tempfile.TemporaryDirectory() as d:
   path=save_experiment(exp,d);other=load_experiment(path)
   self.assertEqual(w.contexts,other.worlds[0,'recurring'].contexts)
   self.assertEqual(w.shifts,other.worlds[0,'recurring'].shifts)
   self.assertEqual(w.digest(),other.worlds[0,'recurring'].digest())
   np.testing.assert_array_equal(e['actions'],other.events[0,'recurring','ucb']['actions'])
   raw=next(path.glob('*.json.gz'));raw.write_bytes(b'broken')
   with self.assertRaises(ValueError):load_experiment(path)
 def test_dates_are_not_relabelled(self):
  w=make_world('recurring',80001,0,365,start_date='2024-02-01',dataset_parent=self.datasets.name)
  self.assertEqual(str(world_dates(w)[0].date()),'2024-02-01')
  with self.assertRaises(ValueError):world_dates(w,start='2023-01-01')
  fig=plot_world(w,window=5)
  import matplotlib.pyplot as plt
  plt.close(fig)
