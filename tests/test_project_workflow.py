import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from contextual_bandits import study
from contextual_bandits.workbench import default_settings

class ProjectWorkflow(unittest.TestCase):
 def test_dynamic_study_horizon_and_readable_world_exports(self):
  with tempfile.TemporaryDirectory() as d:
   base=Path(d);results=base/'results';results.mkdir()
   config=default_settings()
   with patch.object(study,'BASE',base),patch.object(study,'RESULTS',results),patch.object(study,'HORIZON',365):
    # The stage's changed horizon must not fall back to a definition-time default.
    with patch.object(study,'ProcessPoolExecutor') as pool:
     pool.return_value.__enter__.return_value.map.return_value=[]
     study.batch('batch',0,config['configs'],config['fixed'],771)
     manifest=json.loads((results/'batch/manifest.json').read_text())
     self.assertEqual(manifest['config']['horizon'],365)
    (results/'smoke').mkdir()
    rows=study.world_job(('smoke',0,config['configs'],config['fixed'],771,365))
    self.assertEqual(len(rows),14*14)
    raw=results/'smoke/world-0000-recurring-dataset.json'
    records=json.loads(raw.read_text());self.assertEqual(len(records),365)
    self.assertEqual(set(records[179]['rewards']),{'0','1','2','3'})
    self.assertEqual(set(records[180]['rewards']),{'0','1','2','3','4'})
    self.assertNotIn('research/',json.dumps(manifest))
