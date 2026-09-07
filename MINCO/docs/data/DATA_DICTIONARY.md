# MINCO Reference Data Dictionary

The bundled Meridian Health Network is a **synthetic reference case** used to verify MINCO workflows. It is not a real hospital system and does not contain patient data.

## `data/base_instance/hospitals.csv`

| Field | Meaning |
|---|---|
| `hospital_id` | Stable synthetic facility identifier |
| `hospital_name` | Human-readable synthetic name |
| `hospital_type` | Synthetic facility role/category |

## `capacities.csv`

Unique key: (`hospital_id`, `resource`). `base_capacity` is nominal available capacity for ICU or Ward resources.

## `surge_caps.csv`

Unique key: (`hospital_id`, `resource`). `max_surge` is the maximum modeled temporary capacity activation.

## `safe_thresholds.csv`

Unique key: (`hospital_id`, `resource`). `safe_utilization` is a modeled planning threshold in (0, 1]. It is not a clinical standard.

## `transfer_lanes.csv`

Unique key: (`from_hospital`, `to_hospital`). `allowed` is a binary topology switch. `transfer_capacity` is the modeled maximum ICU transfer load per planning day on that enabled directed lane. In the bundled synthetic networks it is set to 20% of the source hospital's nominal ICU capacity per lane/day. `transfer_cost` and `transfer_time` are synthetic planning parameters. Transfer capacity is an operational modeling assumption, not a measured hospital transport limit.

## `costs.csv`

Unique key: `cost_name`. Contains objective-function weights for the optimization experiments.

## `elective_bounds.csv`

Unique key: (`day`, `hospital_id`, `cohort`). Provides minimum and maximum modeled elective arrivals that may be accepted.

## `arrivals.csv`

Unique key: (`day`, `hospital_id`, `cohort`). Contains expected synthetic arrivals by planning day.

## `data/transitions/cohort_*.csv`

Unique key: (`from_state`, `to_state`) within each cohort. Each file is a 5 × 5 row-stochastic transition matrix over ED, ICU, Ward, Discharged, and Dead.
