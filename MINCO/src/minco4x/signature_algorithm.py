"""FLOW-CVaR signature orchestration: Markov patient flow + network surge MILP."""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from src.decision_math.flow_cvar import as_dict, summarize_flow_cvar
from src.decision_math.stochastic_milp_oracle import StochasticCapacityInstance, solve_small_stochastic_milp

FLOW_STATES=('H1','H2','EXIT')
DEFAULT_TRANSITION=np.array([
    [0.80,0.10,0.10],
    [0.14,0.70,0.16],
    [0.00,0.00,1.00],
],dtype=float)
DEFAULT_STATE=np.array([15.0,7.0,0.0])
DEFAULT_ARRIVALS=np.array([2.02,0.60,0.0])

def build_markov_flow_forecast(state=DEFAULT_STATE, transition=DEFAULT_TRANSITION, arrivals=DEFAULT_ARRIVALS)->dict:
    state=np.asarray(state,dtype=float); transition=np.asarray(transition,dtype=float); arrivals=np.asarray(arrivals,dtype=float)
    if transition.shape!=(3,3) or not np.allclose(transition.sum(axis=1),1.0):
        raise ValueError('patient-flow transition matrix must be 3x3 and row-stochastic')
    if state.shape!=(3,) or arrivals.shape!=(3,) or np.any(state<0) or np.any(arrivals<0):
        raise ValueError('patient-flow state/arrivals must be non-negative 3-vectors')
    propagated=state@transition
    next_state=propagated+arrivals
    return {
        'states':FLOW_STATES,
        'current_state':state.tolist(),
        'transition_matrix':transition.tolist(),
        'propagated_state':propagated.tolist(),
        'new_arrivals':arrivals.tolist(),
        'next_state':next_state.tolist(),
        'forecast_hospital_census':{'H1':float(next_state[0]),'H2':float(next_state[1])},
        'row_sums':transition.sum(axis=1).tolist(),
    }

def make_network_flow_instance(*,cvar_alpha=.80,cvar_weight=1.50,severe_demand=22.0)->tuple[StochasticCapacityInstance,dict]:
    flow=build_markov_flow_forecast()
    h1=float(flow['forecast_hospital_census']['H1']); h2=float(flow['forecast_hospital_census']['H2'])
    inst=StochasticCapacityInstance(
        hospitals=('H1','H2'),periods=(0,),scenario_probabilities=np.array([.85,.15]),
        demand=np.array([[[h1],[h2]],[[float(severe_demand)],[10.0]]]),
        base_beds=np.array([12.,10.]),safe_fraction=np.array([.90,.90]),
        surge_beds=np.array([4.,3.]),surge_cost=np.array([18.,13.]),
        flex_beds_per_block=np.array([2.,2.]),flex_cost=np.array([7.,7.]),max_flex_blocks=np.array([2,2]),
        elective_deferral_limit=np.array([[3.],[3.]]),
        transfer_capacity=np.array([[[0.],[4.]],[[4.],[0.]]]),
        transfer_cost=.8,defer_cost=1.8,unsafe_cost=8.,boarding_cost=20.,
        cvar_alpha=float(cvar_alpha),cvar_weight=float(cvar_weight),
    )
    return inst,flow

def solve_flow_cvar_network(*,cvar_alpha=.80,cvar_weight=1.50,severe_demand=22.0)->dict:
    inst,flow=make_network_flow_instance(cvar_alpha=cvar_alpha,cvar_weight=cvar_weight,severe_demand=severe_demand)
    sol=solve_small_stochastic_milp(inst); dec=summarize_flow_cvar(inst,sol)
    no_tail_inst=replace(inst,cvar_weight=0.0); no_tail_sol=solve_small_stochastic_milp(no_tail_inst); no_tail=summarize_flow_cvar(no_tail_inst,no_tail_sol)
    actions=[]
    if sol.status=='OPTIMAL':
        for hi,h in enumerate(inst.hospitals):
            if int(sol.surge[hi,0])>0: actions.append({'action':'ACTIVATE_SURGE_BEDS','hospital':h,'units':int(inst.surge_beds[hi])})
            if int(sol.flex_blocks[hi,0])>0: actions.append({'action':'ACTIVATE_FLEX_STAFF','hospital':h,'blocks':int(sol.flex_blocks[hi,0])})
    scenario_actions=[]
    for wi,label in enumerate(('reference','rare_surge')):
        for hi,h in enumerate(inst.hospitals):
            deferred=int(sol.defer[wi,hi,0]);
            if deferred: scenario_actions.append({'scenario':label,'action':'DIVERT_OR_DEFER','hospital':h,'units':deferred})
        for i,origin in enumerate(inst.hospitals):
            for j,dest in enumerate(inst.hospitals):
                qty=int(sol.transfer[wi,i,j,0])
                if qty: scenario_actions.append({'scenario':label,'action':'TRANSFER','from':origin,'to':dest,'units':qty})
    return {
        'flow_forecast':flow,
        'decision':as_dict(dec),'no_cvar_baseline':as_dict(no_tail),
        'first_stage_actions':actions,'scenario_actions':scenario_actions,
        'scenario_probabilities':inst.scenario_probabilities.tolist(),
        'scenario_demand':inst.demand.tolist(),
        'raw_transfer_total':int(sol.transfer.sum()),'raw_deferral_total':int(sol.defer.sum()),
        'evidence_class':'synthetic algorithmic validation',
    }
