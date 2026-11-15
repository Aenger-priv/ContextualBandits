"""Mixed-context synthetic worlds using the original context/rewards record contract."""
from dataclasses import dataclass
from datetime import date,timedelta
import hashlib
import numpy as np

PAIRS=('EUR/USD','GBP/USD','USD/JPY','AUD/USD','USD/CAD')
TIMES=('morning','midday','afternoon','pre-close')
SCENARIOS=('stationary','abrupt_low','abrupt_high','drift','recurring','nonlinear',
           'near_tie','no_context','imbalanced','stress','birth_strong','birth_inferior','birth_tie','birth_contextual')
NAMESPACE={'development':61101,'validation':62202,'pilot':63303,'confirmation':64404}
FEATURE_NAMES=('intercept',)+tuple('pair:'+s for s in PAIRS)+tuple('time:'+s for s in TIMES)+('volatility/0.5','size/2000')

def rng_for(namespace,world,name):
    key=int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],'little')
    return np.random.default_rng(np.random.SeedSequence([namespace,world,key]))

def encode(context):
    """Fixed vocabulary/scales; never fit on evaluation contexts. Date is metadata."""
    required={'currency_pair','volatility','size','time_of_day','date'}
    if set(context)!=required:raise ValueError('context must contain exactly the five original fields')
    if context['currency_pair'] not in PAIRS or context['time_of_day'] not in TIMES:
        raise ValueError('unknown category: declare the vocabulary before evaluation')
    date.fromisoformat(context['date'])
    v=float(context['volatility']);s=float(context['size'])
    if not np.isfinite([v,s]).all() or v<0 or s<=0:raise ValueError('invalid numeric context')
    return np.array([1.,*[float(context['currency_pair']==p) for p in PAIRS],
                     *[float(context['time_of_day']==t) for t in TIMES],v/.5,s/2000])

@dataclass(frozen=True)
class World:
    x:np.ndarray
    mu:np.ndarray
    outcomes:np.ndarray
    available:np.ndarray
    births:tuple
    shifts:tuple
    contexts:tuple=()
    def digest(self):
        h=hashlib.sha256()
        for a in (self.x,self.mu,self.outcomes,self.available):h.update(a.tobytes())
        return h.hexdigest()
    def records(self):
        return [{'context':dict(c),'rewards':{str(a):float(self.outcomes[t,a])
                 for a in np.flatnonzero(self.available[t])}} for t,c in enumerate(self.contexts)]

def from_records(records,expected_rewards=None,*,rewards_are_means=False):
    """Read original JSON records; explicit oracle semantics avoid noisy hindsight regret."""
    if not records:raise ValueError('empty dataset')
    contexts=tuple(dict(r['context']) for r in records);x=np.stack([encode(c) for c in contexts])
    if any(a['date']>b['date'] for a,b in zip(contexts,contexts[1:])):raise ValueError('records must be chronological')
    available=np.zeros((len(records),5),bool);outcomes=np.zeros((len(records),5))
    for t,r in enumerate(records):
        if set(r)!= {'context','rewards'}:raise ValueError('record must contain context and rewards')
        keys={int(a) for a in r['rewards']}
        if keys not in ({0,1,2,3},{0,1,2,3,4}):raise ValueError('expected four initial options then a fifth')
        for a,y in r['rewards'].items():
            a=int(a);outcomes[t,a]=float(y);available[t,a]=True
    if available[0,4] or np.any(np.diff(available[:,4].astype(int))<0):raise ValueError('invalid fifth-option arrival')
    if not np.isfinite(outcomes).all():raise ValueError('nonfinite reward')
    if expected_rewards is None:
        if not rewards_are_means:raise ValueError('expected-reward oracle requires a sidecar or explicit deterministic-mean semantics')
        mu=outcomes.copy()
    else:
        mu=np.asarray(expected_rewards,dtype=float)
        if mu.shape!=outcomes.shape or not np.isfinite(mu).all():raise ValueError('invalid mean sidecar')
    first=np.flatnonzero(available[:,4]);births=((int(first[0]),4),) if len(first) else ()
    return World(x,mu,outcomes,available,births,(),contexts)

def generate(scenario,namespace,world,horizon=18250,*,start_date='2023-01-01',days=365,
             new_arm_day=180,regime_change_days=None):
    if scenario not in SCENARIOS:raise ValueError(scenario)
    if horizon%days or horizon<days or not 0<new_arm_day<days:raise ValueError('horizon must be a positive multiple of days; arrival inside range')
    per_day=horizon//days
    family='abrupt' if scenario in ('abrupt_low','abrupt_high') else scenario
    rng=rng_for(namespace,world,'environment:'+family)
    weights=np.array([.70,.10,.10,.05,.05]) if scenario=='imbalanced' else np.array([.30,.20,.20,.10,.20])
    pair_ids=rng.choice(5,horizon,p=weights)
    day_ids=np.repeat(np.arange(days),per_day)
    basevol=np.array([.08,.12,.10,.15,.11]);vol=np.empty((days,5));vol[0]=basevol
    for day in range(1,days):vol[day]=np.maximum(.015,.94*vol[day-1]+.06*basevol+rng.normal(0,.008,5))
    sizes=np.maximum(10,np.rint(rng.normal(np.array([1000,500,750,300,250])[pair_ids],
                                           np.array([200,100,150,50,50])[pair_ids]))).astype(int)
    tid=np.array([rng.choice(4,p=[.3,.3,.25,.15] if p<2 else [.15,.25,.3,.3] if p<4 else [.25]*4) for p in pair_ids])
    start=date.fromisoformat(start_date)
    contexts=tuple({'currency_pair':PAIRS[p],'volatility':round(float(vol[d,p]),4),
                    'size':int(size),'time_of_day':TIMES[t],'date':(start+timedelta(days=int(d))).isoformat()}
                   for p,d,size,t in zip(pair_ids,day_ids,sizes,tid))
    x=np.stack([encode(c) for c in contexts])
    # Reward coefficients are scenario-independent in meaning; all four predictors act.
    theta=np.zeros((4,len(FEATURE_NAMES)));theta[:,0]=[.44,.43,.42,.41]
    theta[:,1:6]=np.array([[.10,.02,-.03,.04,-.05],[-.04,.10,.03,-.02,.06],
                          [.01,-.02,.10,.06,-.03],[.02,.03,-.04,.10,.07]])
    theta[:,6:10]=np.array([[.08,.02,-.03,-.06],[-.03,.07,.03,-.02],[-.04,-.01,.05,.09],[.06,-.04,.02,.03]])
    theta[:,10]=[.12,-.20,.30,-.08];theta[:,11]=[-.10,.12,.06,-.06]
    theta+=rng.normal(0,.008,theta.shape)
    if scenario=='no_context':theta[:,1:]=0
    if scenario=='near_tie':theta[:,1:]*=.025;theta[:,0]=.45+np.arange(4)*.001
    base=x@theta.T;other=base[:,[1,0,3,2]];mu=base.copy();shifts=()
    default=[90,240] if scenario=='recurring' else [180]
    changes=default if regime_change_days is None else list(regime_change_days)
    if scenario in ('abrupt_low','abrupt_high','recurring'):
        if changes!=sorted(set(changes)) or any(not 0<d<days for d in changes):raise ValueError('invalid change days')
        if scenario.startswith('abrupt') and len(changes)!=1:raise ValueError('abrupt needs one change')
        phase=np.searchsorted(changes,day_ids,side='right')%2
        mu[phase==1]=other[phase==1];shifts=tuple(int(d*per_day) for d in changes)
    if scenario=='drift':
        w=day_ids/(days-1);mu=(1-w[:,None])*base+w[:,None]*other
    if scenario=='nonlinear':
        v=x[:,10];s=x[:,11]
        mu+=.15*np.column_stack((np.sin(8*v),-np.sin(8*v),s*v,-s*v))
    birth=new_arm_day*per_day
    # Exactly four to five in every scenario; new option's quality is explicit.
    best=mu.max(axis=1)
    fifth=mu.mean(axis=1)+.015
    if scenario=='birth_strong':fifth=best+.10
    if scenario=='birth_inferior':fifth=mu.min(axis=1)-.10
    if scenario=='birth_tie':fifth=best+.001
    if scenario=='birth_contextual':fifth=best+np.where(np.isin(pair_ids,[0,2]),.10,-.20)
    mu=np.column_stack((mu,fifth));available=np.ones(mu.shape,bool);available[:birth,4]=False
    sd=.30 if scenario in ('abrupt_high','stress') else .05
    if scenario=='stress':noise=sd*(rng.standard_t(3,(horizon,1))+rng.standard_t(3,mu.shape))/np.sqrt(6)
    else:noise=sd*rng.normal(size=mu.shape)
    mu=np.where(available,mu,0.);outcomes=np.where(available,mu+noise,0.)
    return World(x,mu,outcomes,available,((birth,4),),shifts,contexts)
