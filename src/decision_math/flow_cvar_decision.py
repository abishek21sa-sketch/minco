from __future__ import annotations
import hashlib,json
import numpy as np
from src.minco4x.signature_algorithm import solve_flow_cvar_network

def build_flow_cvar_decision(*,cvar_alpha:float=.80,cvar_weight:float=1.50,severe_demand:float=22.0)->dict:
    result=solve_flow_cvar_network(cvar_alpha=cvar_alpha,cvar_weight=cvar_weight,severe_demand=severe_demand)
    d=result['decision']; base=result['no_cvar_baseline']; flow=result['flow_forecast']
    checks={
      'optimizer_optimal':d['status']=='OPTIMAL','solution_feasible':bool(d['feasible']),
      'markov_transition_row_stochastic':bool(np.allclose(flow['row_sums'],1.0)),
      'network_has_two_hospitals':len(flow['forecast_hospital_census'])==2,
      'tail_risk_not_worse_than_no_cvar':d['cvar_loss']<=base['cvar_loss']+1e-9,
      'clinical_claim_is_bounded':'not a clinical outcome guarantee' in d['bounded_claim'],
    }
    authorized=all(checks.values())
    params={'cvar_alpha':float(cvar_alpha),'cvar_weight':float(cvar_weight),'severe_demand':float(severe_demand)}
    fp={'parameters':params,'decision':d,'flow':flow,'checks':checks,'actions':result['first_stage_actions']}
    did='FLOW-'+hashlib.sha256(json.dumps(fp,sort_keys=True,default=str).encode()).hexdigest()[:16].upper()
    return {'decision_id':did,'algorithm':'FLOW-CVaR','gate':'AUTHORIZED' if authorized else 'BLOCKED','human_review_required':True,
      'parameters':params,'checks':checks,'decision':d,'no_cvar_baseline':base,
      'flow_forecast':flow,'actions':result['first_stage_actions'],'scenario_actions':result['scenario_actions'],
      'scenario_probabilities':result['scenario_probabilities'],'scenario_demand':result['scenario_demand'],
      'operator_note':'AUTHORIZED means the modeled network capacity decision passed formulation/evidence checks. It does not authorize autonomous clinical, staffing, diversion, or transfer actions.',
      'evidence_class':result['evidence_class']}
