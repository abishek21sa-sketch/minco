# ADR-0002 — Separate Transfer Topology from Transfer Capacity

**Status:** Accepted — Finalization Phase 1

## Context

The original `transfer_lanes.allowed` field is binary. Scenario code could scale it using `transfer_capacity_multiplier`. Because allowed lanes are later filtered with `allowed == 1`, any multiplier below one unintentionally removed every enabled lane rather than reducing capacity.

## Decision

- Keep `allowed` strictly binary and use it only for network topology.
- Add `transfer_capacity` as a nonnegative lane/day capacity parameter.
- Scale only `transfer_capacity` in transfer-degradation scenarios.
- Constrain `x[i,j,t] <= transfer_capacity[i,j]` in the Gurobi network model.
- In bundled synthetic networks, initialize each directed lane capacity at 20% of the source hospital's nominal ICU capacity per day. This is a transparent synthetic planning assumption, not a measured transport limit.

## Consequences

Historical optimization/scenario results produced before this change are retained as legacy research evidence but are not current V1 benchmark claims. Current results must be regenerated under the explicit transfer-capacity model.
