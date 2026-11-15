"""Partial-feedback policies with explicit global memory clocks and local RNGs."""
from collections import deque
import numpy as np

METHODS=('uniform','fixed','context_free','greedy','epsilon','ucb','sw_ucb',
         'discount_ucb','ts','softmax','sw_softmax','dr_softmax','sw_dr_softmax',
         'full_information')


def softmax(scores, temperature, mixture=.05):
    if temperature<=0 or not 0<=mixture<=1:raise ValueError('invalid exploration')
    e=np.exp((scores-np.max(scores))/temperature)
    return (1-mixture)*e/e.sum()+mixture/len(e)


def dr_outcomes(prediction, arm, reward, propensity):
    q=np.asarray(prediction,dtype=float).copy()
    if not 0<propensity<=1:raise ValueError('chosen action lacks support')
    q[arm]+=(reward-q[arm])/propensity
    return q


class RidgeState:
    def __init__(self,k,d,window=None,gamma=1.):
        self.k,self.d,self.window,self.gamma=k,d,window,gamma
        self.A=np.tile(np.eye(d),(k,1,1));self.b=np.zeros((k,d))
        self.history=deque();self.last_clock=-1

    def clock(self,t):
        if t!=self.last_clock+1:raise ValueError('clock must advance exactly once')
        if self.gamma!=1 and t>0:
            self.A=self.gamma*self.A+(1-self.gamma)*np.eye(self.d)
            self.b*=self.gamma
        if self.window is not None:
            while self.history and self.history[0][0]<=t-self.window:
                _,arms,x,y=self.history.popleft()
                self.A[arms]-=np.outer(x,x);self.b[arms]-=y[:,None]*x
        self.last_clock=t

    def add(self,t,x,arms,ys):
        arms=np.asarray(arms,dtype=int);ys=np.asarray(ys,dtype=float)
        if t!=self.last_clock or len(arms)!=len(ys):raise ValueError('invalid update')
        self.A[arms]+=np.outer(x,x);self.b[arms]+=ys[:,None]*x
        if self.window is not None:self.history.append((t,arms.copy(),x.copy(),ys.copy()))

    def predict(self,x):
        theta=np.linalg.solve(self.A,self.b[...,None])[...,0]
        return theta@x

    def variance(self,x):
        rhs=np.broadcast_to(x,(self.k,self.d))[...,None]
        return np.maximum(np.linalg.solve(self.A,rhs)[...,0]@x,0)


class Policy:
    """select sees current x/availability only; observe sees just chosen reward."""
    def __init__(self,method,k,d,rng,config=None,fixed_arm=0):
        if method not in METHODS:raise ValueError(method)
        self.method,self.rng,self.config,self.fixed_arm=method,rng,config or {},fixed_arm
        self.context_free=method=='context_free'
        dim=1 if self.context_free else d
        window=self.config.get('window',300) if method.startswith('sw_') else None
        gamma=self.config.get('gamma',.995) if method=='discount_ucb' else 1.
        self.state=RidgeState(k,dim,window,gamma)
        self.nuisance=RidgeState(k,dim,window) if 'dr_' in method else None
        self.max_inverse_propensity=0.;self.max_pseudo=0.
        self.pending=None

    def select(self,t,x,available):
        if self.pending is not None:raise ValueError('previous feedback outstanding')
        x=np.array([1.]) if self.context_free else x.copy()
        ids=np.flatnonzero(available)
        if not len(ids):raise ValueError('no eligible action')
        self.state.clock(t)
        if self.nuisance:self.nuisance.clock(t)
        q=self.state.predict(x)[ids]
        p=np.zeros(self.state.k)
        m=self.method
        if m=='uniform':p[ids]=1/len(ids)
        elif m=='fixed':
            if self.fixed_arm not in ids:raise ValueError('fixed incumbent unavailable')
            p[self.fixed_arm]=1
        elif m in ('context_free','epsilon'):
            e=self.config.get('epsilon',.10);p[ids]=e/len(ids);p[ids[np.argmax(q)]]+=1-e
        elif 'softmax' in m:p[ids]=softmax(q,self.config.get('tau',.10))
        elif 'ucb' in m:
            values=q+self.config.get('alpha',.5)*np.sqrt(self.state.variance(x)[ids])
            p[ids[np.argmax(values)]]=1
        elif m=='ts':
            values=q+self.rng.normal(size=len(ids))*np.sqrt(self.config.get('v',.025)*self.state.variance(x)[ids])
            p[ids[np.argmax(values)]]=1
        else:p[ids[np.argmax(q)]]=1
        # TS p is the conditional point mass after internal sampling, not its marginal propensity.
        # No importance-weighted method uses TS p.
        arm=int(self.rng.choice(self.state.k,p=p)) if np.count_nonzero(p)>1 else int(np.argmax(p))
        nuisance=self.nuisance.predict(x) if self.nuisance else None
        self.pending=(t,x,ids,arm,p[arm],nuisance)
        return arm,float(p[arm])

    def observe(self,arm,reward):
        if self.pending is None:raise ValueError('select must precede feedback')
        t,x,ids,chosen,p,q=self.pending
        if arm!=chosen or not np.isfinite(reward):raise ValueError('invalid feedback')
        if self.nuisance is not None:
            # q was frozen before seeing reward or fitting either model on this event.
            targets=dr_outcomes(q,arm,reward,p)
            self.state.add(t,x,ids,targets[ids])
            self.nuisance.add(t,x,[arm],[reward])
            self.max_inverse_propensity=max(self.max_inverse_propensity,1/p)
            self.max_pseudo=max(self.max_pseudo,float(np.abs(targets[ids]).max()))
        else:self.state.add(t,x,[arm],[reward])
        self.pending=None

    def observe_full(self,outcomes):
        if self.method!='full_information':raise ValueError('extra feedback forbidden')
        t,x,ids,chosen,_,_=self.pending
        self.state.add(t,x,ids,np.asarray(outcomes)[ids]);self.pending=None
