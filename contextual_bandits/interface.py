"""Context-dictionary boundary for online selected-feedback operation."""
import numpy as np
from .policies import Policy
from .simulator import encode,FEATURE_NAMES,rng_for
class ContextPolicy:
    def __init__(self,method,config=None,seed=1):
        self.policy=Policy(method,5,len(FEATURE_NAMES),rng_for(seed,0,'online:'+method),config)
        self.clock=0
    def select(self,context,available_options):
        ids=list(available_options)
        if len(set(ids))!=len(ids) or not ids or any(a not in range(5) for a in ids):raise ValueError('invalid options')
        mask=np.zeros(5,bool);mask[ids]=True
        return self.policy.select(self.clock,encode(context),mask)
    def observe(self,chosen_option,reward):
        self.policy.observe(chosen_option,reward);self.clock+=1
