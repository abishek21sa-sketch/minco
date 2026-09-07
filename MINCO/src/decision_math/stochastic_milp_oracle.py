"""Independent small-instance stochastic MILP oracle using SciPy/HiGHS.

MINCO's primary Phase-1 optimizer is Julia/JuMP/Gurobi.  This module exists as
an independent validation oracle for small benchmark instances so correctness
is not inferred from one solver implementation alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


@dataclass(frozen=True)
class StochasticCapacityInstance:
    hospitals: tuple[str, ...]
    periods: tuple[int, ...]
    scenario_probabilities: np.ndarray  # [w]
    demand: np.ndarray  # [w,h,t]
    base_beds: np.ndarray  # [h]
    safe_fraction: np.ndarray  # [h]
    surge_beds: np.ndarray  # [h]
    surge_cost: np.ndarray  # [h]
    flex_beds_per_block: np.ndarray  # [h]
    flex_cost: np.ndarray  # [h]
    max_flex_blocks: np.ndarray  # [h]
    elective_deferral_limit: np.ndarray  # [h,t]
    transfer_capacity: np.ndarray  # [h,h,t]
    transfer_cost: float = 0.8
    defer_cost: float = 1.8
    unsafe_cost: float = 8.0
    boarding_cost: float = 20.0
    cvar_alpha: float = 0.95
    cvar_weight: float = 0.35

    def validate(self) -> None:
        h, t, w = len(self.hospitals), len(self.periods), len(self.scenario_probabilities)
        if self.demand.shape != (w, h, t):
            raise ValueError("demand shape must be [scenario,hospital,period]")
        if self.elective_deferral_limit.shape != (h, t):
            raise ValueError("elective_deferral_limit shape mismatch")
        if self.transfer_capacity.shape != (h, h, t):
            raise ValueError("transfer_capacity shape mismatch")
        if not np.isclose(self.scenario_probabilities.sum(), 1.0):
            raise ValueError("scenario probabilities must sum to one")
        if np.any(self.scenario_probabilities <= 0):
            raise ValueError("scenario probabilities must be positive")
        for arr in (
            self.demand, self.base_beds, self.safe_fraction, self.surge_beds,
            self.surge_cost, self.flex_beds_per_block, self.flex_cost,
            self.max_flex_blocks, self.elective_deferral_limit, self.transfer_capacity,
        ):
            if np.any(np.asarray(arr) < 0):
                raise ValueError("Capacity MILP inputs must be nonnegative")
        if np.any((self.safe_fraction <= 0) | (self.safe_fraction > 1)):
            raise ValueError("safe_fraction must be in (0,1]")
        if not 0 < self.cvar_alpha < 1 or self.cvar_weight < 0:
            raise ValueError("Invalid CVaR parameters")


@dataclass(frozen=True)
class OracleSolution:
    status: str
    objective: float
    surge: np.ndarray
    flex_blocks: np.ndarray
    defer: np.ndarray
    transfer: np.ndarray
    unsafe: np.ndarray
    boarding: np.ndarray
    scenario_loss: np.ndarray
    eta: float
    xi: np.ndarray
    raw_status: int
    message: str


class _Index:
    def __init__(self, instance: StochasticCapacityInstance):
        h, t, w = len(instance.hospitals), len(instance.periods), len(instance.scenario_probabilities)
        self.cursor = 0
        self.y = np.arange(self.cursor, self.cursor + h*t).reshape(h,t); self.cursor += h*t
        self.f = np.arange(self.cursor, self.cursor + h*t).reshape(h,t); self.cursor += h*t
        self.d = np.arange(self.cursor, self.cursor + w*h*t).reshape(w,h,t); self.cursor += w*h*t
        self.x = np.arange(self.cursor, self.cursor + w*h*h*t).reshape(w,h,h,t); self.cursor += w*h*h*t
        self.u = np.arange(self.cursor, self.cursor + w*h*t).reshape(w,h,t); self.cursor += w*h*t
        self.b = np.arange(self.cursor, self.cursor + w*h*t).reshape(w,h,t); self.cursor += w*h*t
        self.eta = self.cursor; self.cursor += 1
        self.xi = np.arange(self.cursor, self.cursor + w); self.cursor += w
        self.n = self.cursor


def _loss_coefficients(inst: StochasticCapacityInstance, idx: _Index, scenario: int) -> np.ndarray:
    coeff = np.zeros(idx.n, dtype=float)
    coeff[idx.d[scenario].ravel()] = inst.defer_cost
    coeff[idx.x[scenario].ravel()] = inst.transfer_cost
    coeff[idx.u[scenario].ravel()] = inst.unsafe_cost
    coeff[idx.b[scenario].ravel()] = inst.boarding_cost
    # self-transfers are fixed to zero, so their cost coefficient is irrelevant.
    return coeff


def solve_small_stochastic_milp(inst: StochasticCapacityInstance, *, fixed_surge: np.ndarray | None = None, fixed_flex: np.ndarray | None = None) -> OracleSolution:
    inst.validate()
    idx = _Index(inst)
    h, t, w = len(inst.hospitals), len(inst.periods), len(inst.scenario_probabilities)

    c = np.zeros(idx.n, dtype=float)
    for hi in range(h):
        for ti in range(t):
            c[idx.y[hi,ti]] = inst.surge_cost[hi]
            c[idx.f[hi,ti]] = inst.flex_cost[hi]
    for wi, probability in enumerate(inst.scenario_probabilities):
        c += probability * _loss_coefficients(inst, idx, wi)
    c[idx.eta] += inst.cvar_weight
    for wi, probability in enumerate(inst.scenario_probabilities):
        c[idx.xi[wi]] += inst.cvar_weight * probability / (1.0 - inst.cvar_alpha)

    lower = np.zeros(idx.n, dtype=float)
    upper = np.full(idx.n, np.inf, dtype=float)
    upper[idx.y.ravel()] = 1.0
    for hi in range(h):
        upper[idx.f[hi].ravel()] = inst.max_flex_blocks[hi]
    if fixed_surge is not None:
        fs = np.asarray(fixed_surge, dtype=float)
        if fs.shape != (h,t):
            raise ValueError("fixed_surge shape mismatch")
        lower[idx.y.ravel()] = fs.ravel(); upper[idx.y.ravel()] = fs.ravel()
    if fixed_flex is not None:
        ff = np.asarray(fixed_flex, dtype=float)
        if ff.shape != (h,t):
            raise ValueError("fixed_flex shape mismatch")
        lower[idx.f.ravel()] = ff.ravel(); upper[idx.f.ravel()] = ff.ravel()
    for wi in range(w):
        upper[idx.d[wi].ravel()] = np.tile(inst.elective_deferral_limit, 1).ravel()
        for hi in range(h):
            for hj in range(h):
                for ti in range(t):
                    upper[idx.x[wi,hi,hj,ti]] = 0.0 if hi == hj else inst.transfer_capacity[hi,hj,ti]

    integrality = np.zeros(idx.n, dtype=int)
    integrality[idx.y.ravel()] = 1
    integrality[idx.f.ravel()] = 1
    integrality[idx.d.ravel()] = 1
    integrality[idx.x.ravel()] = 1

    rows: list[dict[int,float]] = []
    lows: list[float] = []
    highs: list[float] = []

    def add(row: dict[int,float], low: float=-np.inf, high: float=np.inf) -> None:
        rows.append(row); lows.append(low); highs.append(high)

    for wi in range(w):
        for hi in range(h):
            for ti in range(t):
                # treated = demand - defer + incoming - outgoing
                # boarding >= treated - physical capacity
                row_b: dict[int,float] = {idx.b[wi,hi,ti]: -1.0, idx.d[wi,hi,ti]: -1.0,
                                           idx.y[hi,ti]: -inst.surge_beds[hi], idx.f[hi,ti]: -inst.flex_beds_per_block[hi]}
                for hj in range(h):
                    if hj != hi:
                        row_b[idx.x[wi,hj,hi,ti]] = row_b.get(idx.x[wi,hj,hi,ti],0.0) + 1.0
                        row_b[idx.x[wi,hi,hj,ti]] = row_b.get(idx.x[wi,hi,hj,ti],0.0) - 1.0
                # demand + linear terms <= base capacity
                add(row_b, high=float(inst.base_beds[hi] - inst.demand[wi,hi,ti]))

                # unsafe >= treated - safe_fraction * physical capacity
                sf = float(inst.safe_fraction[hi])
                row_u: dict[int,float] = {idx.u[wi,hi,ti]: -1.0, idx.d[wi,hi,ti]: -1.0,
                                           idx.y[hi,ti]: -sf*inst.surge_beds[hi], idx.f[hi,ti]: -sf*inst.flex_beds_per_block[hi]}
                for hj in range(h):
                    if hj != hi:
                        row_u[idx.x[wi,hj,hi,ti]] = row_u.get(idx.x[wi,hj,hi,ti],0.0) + 1.0
                        row_u[idx.x[wi,hi,hj,ti]] = row_u.get(idx.x[wi,hi,hj,ti],0.0) - 1.0
                add(row_u, high=float(sf*inst.base_beds[hi] - inst.demand[wi,hi,ti]))

    # CVaR excess: xi_w >= loss_w - eta -> loss - eta - xi <= 0
    loss_coeffs = []
    for wi in range(w):
        lc = _loss_coefficients(inst, idx, wi)
        loss_coeffs.append(lc)
        nz = {int(j): float(v) for j,v in enumerate(lc) if abs(v) > 0}
        nz[idx.eta] = -1.0
        nz[idx.xi[wi]] = -1.0
        add(nz, high=0.0)

    a = lil_matrix((len(rows), idx.n), dtype=float)
    for ri, mapping in enumerate(rows):
        for col, value in mapping.items():
            a[ri,col] = value
    constraints = LinearConstraint(a.tocsr(), np.asarray(lows), np.asarray(highs))
    result = milp(c=c, integrality=integrality, bounds=Bounds(lower,upper), constraints=constraints,
                  options={"time_limit": 30.0})
    status_map = {0:"OPTIMAL", 1:"LIMIT_REACHED", 2:"INFEASIBLE", 3:"UNBOUNDED", 4:"OTHER"}
    status = status_map.get(int(result.status), "OTHER")
    if result.x is None:
        empty = np.empty((0,))
        return OracleSolution(status, float("inf"), empty, empty, empty, empty, empty, empty, empty,
                              float("nan"), empty, int(result.status), str(result.message))
    xvec = result.x
    scenario_loss = np.array([float(lc @ xvec) for lc in loss_coeffs])
    return OracleSolution(
        status=status,
        objective=float(result.fun),
        surge=np.rint(xvec[idx.y]).astype(int),
        flex_blocks=np.rint(xvec[idx.f]).astype(int),
        defer=np.rint(xvec[idx.d]).astype(int),
        transfer=np.rint(xvec[idx.x]).astype(int),
        unsafe=xvec[idx.u].copy(),
        boarding=xvec[idx.b].copy(),
        scenario_loss=scenario_loss,
        eta=float(xvec[idx.eta]),
        xi=xvec[idx.xi].copy(),
        raw_status=int(result.status),
        message=str(result.message),
    )



def validate_stochastic_solution(inst: StochasticCapacityInstance, solution: OracleSolution, *, tolerance: float = 1e-6) -> dict[str, float | bool]:
    """Independently recompute all principal capacity/recourse constraints."""
    inst.validate()
    if solution.status != "OPTIMAL":
        return {"feasible": False, "status_optimal": False}
    w_count, h_count, t_count = inst.demand.shape
    max_deferral_excess = 0.0
    max_transfer_excess = 0.0
    max_boarding_shortfall = 0.0
    max_unsafe_shortfall = 0.0
    max_cvar_excess_shortfall = 0.0
    max_integrality_error = 0.0
    for wi in range(w_count):
        for hi in range(h_count):
            for ti in range(t_count):
                max_deferral_excess = max(max_deferral_excess, float(solution.defer[wi,hi,ti] - inst.elective_deferral_limit[hi,ti]))
                capacity = float(inst.base_beds[hi] + inst.surge_beds[hi]*solution.surge[hi,ti] + inst.flex_beds_per_block[hi]*solution.flex_blocks[hi,ti])
                incoming = float(sum(solution.transfer[wi,hj,hi,ti] for hj in range(h_count) if hj != hi))
                outgoing = float(sum(solution.transfer[wi,hi,hj,ti] for hj in range(h_count) if hj != hi))
                treated = float(inst.demand[wi,hi,ti] - solution.defer[wi,hi,ti] + incoming - outgoing)
                max_boarding_shortfall = max(max_boarding_shortfall, treated - capacity - float(solution.boarding[wi,hi,ti]))
                max_unsafe_shortfall = max(max_unsafe_shortfall, treated - float(inst.safe_fraction[hi])*capacity - float(solution.unsafe[wi,hi,ti]))
                for hj in range(h_count):
                    value = float(solution.transfer[wi,hi,hj,ti])
                    allowed = 0.0 if hi == hj else float(inst.transfer_capacity[hi,hj,ti])
                    max_transfer_excess = max(max_transfer_excess, value - allowed)
    max_cvar_excess_shortfall = float(np.max(solution.scenario_loss - solution.eta - solution.xi))
    for arr in (solution.surge, solution.flex_blocks, solution.defer, solution.transfer):
        max_integrality_error = max(max_integrality_error, float(np.max(np.abs(arr - np.rint(arr)))))
    max_violation = max(0.0, max_deferral_excess, max_transfer_excess, max_boarding_shortfall, max_unsafe_shortfall, max_cvar_excess_shortfall, max_integrality_error)
    return {
        "feasible": bool(max_violation <= tolerance),
        "status_optimal": True,
        "max_violation": float(max_violation),
        "max_deferral_excess": float(max(0.0,max_deferral_excess)),
        "max_transfer_excess": float(max(0.0,max_transfer_excess)),
        "max_boarding_shortfall": float(max(0.0,max_boarding_shortfall)),
        "max_unsafe_shortfall": float(max(0.0,max_unsafe_shortfall)),
        "max_cvar_excess_shortfall": float(max(0.0,max_cvar_excess_shortfall)),
        "max_integrality_error": float(max_integrality_error),
    }

def brute_force_single_hospital_first_stage(inst: StochasticCapacityInstance) -> dict[str, Any]:
    """Independent enumeration oracle for one-hospital, one-period instances.

    Recourse has no transfers. For each first-stage choice, optimal deferral,
    unsafe slack, and boarding can be computed by enumerating deferral on an
    integer grid. This is deliberately slow and tiny; it is a correctness oracle.
    """
    inst.validate()
    if len(inst.hospitals) != 1 or len(inst.periods) != 1:
        raise ValueError("Enumeration oracle supports exactly one hospital and one period")
    best: dict[str, Any] | None = None
    max_def = int(round(float(inst.elective_deferral_limit[0,0])))
    for surge, flex in product([0,1], range(int(inst.max_flex_blocks[0])+1)):
        first_cost = surge*inst.surge_cost[0] + flex*inst.flex_cost[0]
        physical = inst.base_beds[0] + surge*inst.surge_beds[0] + flex*inst.flex_beds_per_block[0]
        safe = inst.safe_fraction[0]*physical
        scenario_best_losses=[]
        for wi in range(len(inst.scenario_probabilities)):
            demand=float(inst.demand[wi,0,0])
            best_loss=float("inf")
            for defer in range(max_def+1):
                treated=max(0.0,demand-defer)
                unsafe=max(0.0,treated-safe)
                boarding=max(0.0,treated-physical)
                loss=inst.defer_cost*defer+inst.unsafe_cost*unsafe+inst.boarding_cost*boarding
                best_loss=min(best_loss,loss)
            scenario_best_losses.append(best_loss)
        losses=np.asarray(scenario_best_losses)
        expected=float(np.dot(inst.scenario_probabilities,losses))
        candidates=np.unique(losses)
        cvar=min(
            eta + float(np.dot(inst.scenario_probabilities, np.maximum(losses-eta,0.0)))/(1-inst.cvar_alpha)
            for eta in candidates
        )
        objective=float(first_cost+expected+inst.cvar_weight*cvar)
        item={"objective":objective,"surge":surge,"flex":flex,"scenario_loss":losses,"cvar":cvar}
        if best is None or objective < best["objective"]-1e-9:
            best=item
    assert best is not None
    return best


def make_tiny_oracle_instance() -> StochasticCapacityInstance:
    return StochasticCapacityInstance(
        hospitals=("H1",), periods=(0,),
        scenario_probabilities=np.array([0.5,0.5]),
        demand=np.array([[[11.0]], [[15.0]]]),
        base_beds=np.array([10.0]), safe_fraction=np.array([0.9]),
        surge_beds=np.array([4.0]), surge_cost=np.array([8.0]),
        flex_beds_per_block=np.array([2.0]), flex_cost=np.array([5.0]), max_flex_blocks=np.array([2]),
        elective_deferral_limit=np.array([[3.0]]),
        transfer_capacity=np.zeros((1,1,1)),
        cvar_alpha=0.8, cvar_weight=0.4,
    )
