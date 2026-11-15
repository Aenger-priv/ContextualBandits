"""Evaluator owns oracle/counterfactuals; learners receive a scalar selected reward."""
import numpy as np
from .policies import Policy
from .simulator import rng_for


def cold_metrics(world,actions,regret):
    result=[]
    for birth,arm in world.births:
        end=min(len(actions),birth+100)
        a=actions[birth:end];n=len(a)
        optimal=np.argmax(np.where(world.available[birth:end],world.mu[birth:end],-np.inf),axis=1)
        is_best=optimal==arm;chosen=a==arm;found=np.flatnonzero(chosen)
        delay=int(found[0]) if len(found) else n
        correct=chosen==is_best
        recovery=n
        for j in range(0,n-19):
            if correct[j:j+20].mean()>=.8:recovery=j+19;break
        result.append({'birth':birth,'arm':arm,'window':n,'delay':delay,
                       'discovered':bool(len(found)),'by20':bool(len(found) and delay<20),
                       'regret':float(regret[birth:end].sum()),
                       'appropriate_fraction':float(correct.mean()),
                       'best_opportunities':int(is_best.sum()),
                       'chosen_when_best':int(np.sum(chosen&is_best)),
                       'inferior_opportunities':int((~is_best).sum()),
                       'chosen_when_inferior':int(np.sum(chosen&~is_best)),
                       'appropriate_block_delay':int(recovery),
                       'appropriate_block_found':bool(recovery<n)})
    return result


def run(world,method,config,namespace,world_id,scenario,fixed_arm=0):
    p=Policy(method,world.mu.shape[1],world.x.shape[1],
             rng_for(namespace,world_id,'policy:'+scenario+':'+method),config,fixed_arm)
    n=len(world.x);actions=np.empty(n,dtype=np.int16);prop=np.empty(n)
    for t in range(n):
        arm,prob=p.select(t,world.x[t],world.available[t])
        actions[t]=arm;prop[t]=prob
        if method=='full_information':p.observe_full(world.outcomes[t].copy())
        else:p.observe(arm,float(world.outcomes[t,arm]))
    expected=world.mu[np.arange(n),actions];reward=world.outcomes[np.arange(n),actions]
    oracle=np.max(np.where(world.available,world.mu,-np.inf),axis=1)
    regret=oracle-expected
    summary={'expected_mean':float(expected.mean()),'realized_mean':float(reward.mean()),
             'regret':float(regret.sum()),'oracle_total':float(oracle.sum()),
             'max_inverse_propensity':p.max_inverse_propensity,'max_pseudo':p.max_pseudo,
             'shift_regret':float(np.mean([regret[s:s+100].sum() for s in world.shifts])) if world.shifts else None,
             'cold':cold_metrics(world,actions,regret)}
    return summary,dict(actions=actions,propensity=prop,rewards=reward,
                        expected=expected,regret=regret)


def primary_contrasts(rows,selected):
    # rows are one or more complete independent world blocks; no decision resampling.
    methods={(r['world'],r['scenario'],r['method']):r['expected_mean'] for r in rows}
    ids=sorted({r['world'] for r in rows});scenarios=sorted({r['scenario'] for r in rows})
    names=['selected_vs_fixed','window_vs_ucb','discount_vs_ucb','ts_noise_interaction',
           'dr_vs_softmax','sw_dr_vs_sw_softmax','context_vs_context_free']
    values=[]
    for w in ids:
        def diff(a,b,ss=scenarios):return np.mean([methods[w,s,a]-methods[w,s,b] for s in ss])
        shifting=['abrupt_low','drift','recurring']
        values.append([diff(selected,'fixed'),diff('sw_ucb','ucb',shifting),
                       diff('discount_ucb','ucb',shifting),
                       diff('ts','ucb',['abrupt_high'])-diff('ts','ucb',['abrupt_low']),
                       diff('dr_softmax','softmax'),diff('sw_dr_softmax','sw_softmax'),
                       diff('epsilon','context_free')])
    return names,np.asarray(values)
