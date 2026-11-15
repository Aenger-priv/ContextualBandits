"""Clean-process stages: development, validation, pilot, freeze, confirmation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import shutil
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import importlib.metadata as md
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import numpy as np
from scipy.stats import t as student
from .simulator import SCENARIOS,NAMESPACE,generate
from .policies import METHODS
from .evaluate import run,primary_contrasts

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/study'
CONF=BASE/'configs'
RESULTS=BASE/'results'
HORIZON=18250
CANDIDATES={
 'ucb':[{'alpha':a} for a in (.1,.5,1.)],
 'sw_ucb':[{'alpha':a,'window':w} for a in (.1,.5,1.) for w in (100,300,600)],
 'discount_ucb':[{'alpha':a,'gamma':g} for a in (.1,.5,1.) for g in (.98,.995,.999)],
 'ts':[{'v':v} for v in (.0025,.025,.25)],
 'epsilon':[{'epsilon':e} for e in (.02,.10,.25)],
 'softmax':[{'tau':v} for v in (.03,.10,.30)],
}

def dump(path,data,exclusive=True):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x' if exclusive else 'w') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def source_hashes():
    paths=sorted((ROOT/'contextual_bandits').glob('*.py'))+[ROOT/'pyproject.toml',ROOT/'uv.lock']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}

def git_revision():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()

def metadata(stage,config):
    return {'stage':stage,'created_utc':datetime.now(timezone.utc).isoformat(),'git_commit':git_revision(),
            'git_status':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
            'source_hashes':source_hashes(),'config':config,'python':sys.version,'platform':platform.platform(),
            'versions':{p:md.version(p) for p in ['numpy','scipy','pandas','matplotlib']},
            'workers':4,'thread_environment':{k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']},
            'original_inputs':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'fx_trading_dataset.json',ROOT/'volatility_series.csv']}}

def load_rows(stage):return json.loads((RESULTS/stage/'summary.json').read_text())

def complete_configs(chosen):
    c={m:{} for m in METHODS};c.update(chosen)
    c['context_free']=dict(c['epsilon'])
    for m in ('sw_softmax','dr_softmax','sw_dr_softmax'):
        c[m]=dict(c['softmax'])
        if m.startswith('sw_'):c[m]['window']=300
    return c

def development_job(args):
    scenario,method,ci,c,w=args
    env=generate(scenario,NAMESPACE['development'],w,HORIZON)
    summary,_=run(env,method,c,NAMESPACE['development'],w,scenario)
    return dict(world=w,scenario=scenario,method=method,candidate=ci,config=c,**summary)

def development():
    out=RESULTS/'development';out.mkdir(exist_ok=True)
    dump(out/'manifest.json',metadata('development',{'candidates':CANDIDATES,'worlds':4,'horizon':HORIZON}))
    start=time.perf_counter();fixed={}
    for s in SCENARIOS:
        means=np.mean([generate(s,NAMESPACE['development'],w,HORIZON).mu[:,:4].mean(axis=0) for w in range(4)],axis=0)
        fixed[s]=int(np.argmax(means))
    jobs=[(s,m,i,c,w) for s in SCENARIOS for m,cs in CANDIDATES.items() for i,c in enumerate(cs) for w in range(4)]
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for row in pool.map(development_job,jobs):
            rows.append(row)
            if len(rows)%56==0:print('development',len(rows),'/',len(jobs),flush=True)
    scores={};chosen={}
    for m,cs in CANDIDATES.items():
        score=[float(np.mean([r['expected_mean'] for r in rows if r['method']==m and r['candidate']==i])) for i in range(len(cs))]
        scores[m]=score;chosen[m]=cs[int(np.argmax(score))]
    dump(out/'summary.json',rows)
    dump(out/'timing.json',{'seconds':time.perf_counter()-start})
    dump(CONF/'development-selection.json',{'configs':complete_configs(chosen),'fixed':fixed,'candidate_scores':scores})

def world_job(args):
    stage,w,configs,fixed,namespace,horizon=args
    out=RESULTS/stage;wp=out/f'world-{w:04d}.json'
    if wp.exists():
        checkpoint=json.loads(wp.read_text())
        for name,h in checkpoint['files'].items():
            if sha(out/name)!=h:raise ValueError('checkpoint hash mismatch')
        return checkpoint['rows']
    rows=[];files={}
    try:
        for scenario in SCENARIOS:
            env=generate(scenario,namespace,w,horizon)
            ep=out/f'world-{w:04d}-{scenario}.npz'
            with ep.open('xb') as f:np.savez_compressed(f,x=env.x,mu=env.mu,outcomes=env.outcomes,available=env.available)
            files[ep.name]=sha(ep)
            raw=out/f'world-{w:04d}-{scenario}-dataset.json'
            with raw.open('x') as f:json.dump(env.records(),f,indent=2);f.write('\n')
            files[raw.name]=sha(raw)
            diagnostic=world_diagnostics(env)
            for method in METHODS:
                summary,events=run(env,method,configs[method],namespace,w,scenario,fixed[scenario])
                path=out/f'world-{w:04d}-{scenario}-{method}.npz'
                with path.open('xb') as f:np.savez_compressed(f,**events)
                files[path.name]=sha(path)
                rows.append(dict(world=w,scenario=scenario,method=method,environment_sha256=env.digest(),diagnostic=diagnostic,**summary))
        dump(wp,{'rows':rows,'files':files});return rows
    except Exception:
        dump(out/f'failure-{w:04d}.json',{'traceback':traceback.format_exc(),'rows':rows});raise

def batch(stage,n,configs,fixed,namespace,horizon=None):
    if horizon is None:horizon=HORIZON
    out=RESULTS/stage;out.mkdir(exist_ok=True)
    cfg={'n_worlds':n,'configs':configs,'fixed':fixed,'namespace':namespace,'horizon':horizon,'scenarios':list(SCENARIOS)}
    mp=out/'manifest.json'
    if mp.exists():
        old=json.loads(mp.read_text())
        if old['source_hashes']!=source_hashes() or old['config']!=cfg:raise ValueError('resume source/config mismatch')
    else:dump(mp,metadata(stage,cfg))
    start=time.perf_counter();rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs=[(stage,w,configs,fixed,namespace,horizon) for w in range(n)]
        for block in pool.map(world_job,jobs):
            rows.extend(block);print(stage,'world',len(rows)//(len(SCENARIOS)*len(METHODS)),'/',n,'seconds',round(time.perf_counter()-start),flush=True)
    if not (out/'summary.json').exists():dump(out/'summary.json',rows)
    if not (out/'timing.json').exists():dump(out/'timing.json',{'seconds':time.perf_counter()-start,'n_worlds':n,'files_bytes':sum(p.stat().st_size for p in out.iterdir()),'peak_rss_bytes':peak_rss()})
    return rows


def peak_rss():
    import resource
    v=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v if sys.platform=='darwin' else v*1024


def world_diagnostics(env):
    opt=np.argmax(np.where(env.available,env.mu,-np.inf),axis=1)
    ranked=np.sort(np.where(env.available,env.mu,-np.inf),axis=1)
    return {'mean_gap':float(np.mean(ranked[:,-1]-ranked[:,-2])),
            'optimal_arm_shares':[float(np.mean(opt==a)) for a in range(env.mu.shape[1])],
            'optimal_arm_turnover':float(np.mean(opt[1:]!=opt[:-1])),
            'oracle_mean':float(np.max(np.where(env.available,env.mu,-np.inf),axis=1).mean()),
            'best_initial_fixed_hindsight':float(env.mu[:,:np.sum(env.available[0])].mean(axis=0).max())}


def validation():
    selection=json.loads((CONF/'development-selection.json').read_text())
    rows=batch('validation',8,selection['configs'],selection['fixed'],NAMESPACE['validation'])
    eligible=[m for m in METHODS if m not in ('uniform','fixed','full_information')]
    scores={m:float(np.mean([r['expected_mean'] for r in rows if r['method']==m])) for m in eligible}
    selected=sorted(scores,key=lambda m:(-scores[m],m))[0]
    dump(CONF/'validation-selection.json',{'selected':selected,'scores':scores})


def pilot():
    selection=json.loads((CONF/'development-selection.json').read_text())
    selected=json.loads((CONF/'validation-selection.json').read_text())['selected']
    rows=batch('pilot',16,selection['configs'],selection['fixed'],NAMESPACE['pilot'])
    names,values=primary_contrasts(rows,selected)
    sd=values.std(axis=0,ddof=1);crit=float(student.ppf(1-.05/14,15))
    requirements=np.ceil((crit*1.2*sd/.005)**2).astype(int)
    n=8*math.ceil(max(32,int(requirements.max()))/8)
    cost=json.loads((RESULTS/'pilot/timing.json').read_text())
    plan={'contrast_names':names,'pilot_sd':sd.tolist(),'critical_value':crit,'sd_inflation':1.2,
          'half_width_target':.005,'requirements':requirements.tolist(),'n_confirmation':n,
          'projected_seconds':cost['seconds']/16*n,'projected_bytes':cost['files_bytes']/16*n,
          'pilot_peak_rss_bytes':cost['peak_rss_bytes']}
    used=sum(json.loads((RESULTS/s/'timing.json').read_text())['seconds'] for s in ['development','validation','pilot'])
    remaining=7200-used-600
    storage=shutil.disk_usage(BASE).free-3*1024**3
    cap=8*int(min(remaining/(cost['seconds']/16),storage/(cost['files_bytes']/16))//8)
    plan.update(required_n=n,remaining_seconds=remaining,budget_cap=cap)
    plan['n_confirmation']=min(n,cap)
    plan['projected_seconds']=cost['seconds']/16*plan['n_confirmation']
    plan['projected_bytes']=cost['files_bytes']/16*plan['n_confirmation']
    dump(CONF/'precision-plan.json',plan)
    print(json.dumps(plan,indent=2))


def freeze():
    selection=json.loads((CONF/'development-selection.json').read_text())
    v=json.loads((CONF/'validation-selection.json').read_text())
    plan=json.loads((CONF/'precision-plan.json').read_text())
    if plan['projected_seconds']>plan['remaining_seconds'] or plan['n_confirmation']<32:raise ValueError('budget decision required')
    # Ensure scientific sources, tests, protocol, and lock have a committed review point.
    paths=list(source_hashes())
    status=subprocess.check_output(['git','status','--porcelain','--',*paths],cwd=ROOT,text=True)
    if status.strip():raise ValueError('commit scientific sources/protocol/lock before freeze')
    f={'version':1,'git_commit':git_revision(),'source_hashes':source_hashes(),
       'configs':selection['configs'],'fixed':selection['fixed'],'selected':v['selected'],
       'precision':plan,'namespace':NAMESPACE['confirmation'],'n_worlds':plan['n_confirmation'],
       'world_ids':list(range(plan['n_confirmation'])),'scenarios':list(SCENARIOS),'horizon':HORIZON,
       'created_utc':datetime.now(timezone.utc).isoformat(),
       'prior_namespaces':{k:v for k,v in NAMESPACE.items() if k!='confirmation'}}
    dump(CONF/'freeze.json',f)
    print('Frozen',f['selected'],'N =',f['n_worlds'],'projected seconds',round(plan['projected_seconds']))


def confirmation():
    f=json.loads((CONF/'freeze.json').read_text())
    if f['source_hashes']!=source_hashes():raise ValueError('scientific source differs from freeze')
    batch('confirmation',f['n_worlds'],f['configs'],f['fixed'],f['namespace'],f['horizon'])


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['development','validation','pilot','freeze','confirmation'])
    p.add_argument('--output',type=Path,default=ROOT/'runs/study',help='Directory shared by all stages of this study')
    p.add_argument('--horizon',type=int,default=18250,help='Trades per world; positive multiple of 365')
    args=p.parse_args()
    global BASE,CONF,RESULTS,HORIZON
    if args.horizon<365 or args.horizon%365:p.error('horizon must be a positive multiple of 365')
    BASE=args.output.resolve();BASE.mkdir(parents=True,exist_ok=True)
    CONF=BASE/'configs';RESULTS=BASE/'results';HORIZON=args.horizon
    CONF.mkdir(parents=True,exist_ok=True);RESULTS.mkdir(parents=True,exist_ok=True)
    for stage in ['development','validation','pilot']:
        previous=RESULTS/stage/'manifest.json'
        if previous.exists() and json.loads(previous.read_text())['config'].get('horizon',HORIZON)!=HORIZON:
            p.error('use the same horizon across stages, or start a new study directory')
    globals()[args.stage]()

if __name__=='__main__':main()
