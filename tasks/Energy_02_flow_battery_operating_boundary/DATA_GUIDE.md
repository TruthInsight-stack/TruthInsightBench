# Data Guide: Rate, Efficiency, Power, and Cycling Boundaries in a Flow-Electrochemical System

- Task ID: `Energy_02_flow_battery_operating_boundary`
- T0: `2022-01-01`
- Payload mode: `bundled`
- Complete source-data boundary: `true`

## Scientific scope

All frozen charge-like and discharge-like curves, rate-dependent efficiency and power summaries, and the complete long-cycle capacity sequence in the task boundary.

## Independent analysis unit

One complete rate curve or the complete ordered cycling experiment; state-of-charge points and adjacent cycles are repeated observations.

## Variables and file groups

- `rate_dependent_voltage_curves.tsv`: `scan_role`, `anonymous_rate_setting`, `state_of_charge_percent`, `cell_voltage_V`
- `rate_tradeoff_summaries.tsv`: `summary_role`, `anonymous_rate_value`, `response_value`, `response_unit_role`
- `long_cycle_capacity.tsv`: `cycle_index`, `capacity_like_value`

## Missingness

No curator-imposed row deletion, random subsampling, or imputation is applied within the frozen task boundary.

## Warnings

- Anonymous rate labels preserve ordering but not source-device identity; interpretations must stay within the supplied system.
- Short-rate curves and long-cycle data probe different failure modes and should not be treated as duplicate evidence.

## Distribution boundary

The archived target article and source-data workbook are CC BY; the task bundles every numerical point in the frozen rate, summary, and cycling series.
