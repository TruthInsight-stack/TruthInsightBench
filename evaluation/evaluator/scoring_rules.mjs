/** Pure scoring rules shared by the semantic and deterministic closure stages. */

const LEVELS = new Set([0, 0.5, 1]);

export function dataIdentityLevel(facts) {
  if (!facts.input_integrity || !facts.source_paths_auditable) return 0;
  return facts.finding_traceability_passed ? 1 : 0.5;
}

export function targetMatchLevel(discoveryId, dimensions) {
  if (discoveryId === null || discoveryId === undefined || discoveryId === "") return 0;
  const values = [
    dimensions.object_match,
    dimensions.scope_match,
    dimensions.direction_match,
    dimensions.core_quantity_match,
  ];
  if (!values.every((value) => typeof value === "boolean")) {
    throw new Error("target-match dimensions must be boolean");
  }
  if (values.every(Boolean)) return 1;
  if (dimensions.object_match && (dimensions.direction_match || dimensions.core_quantity_match)) return 0.5;
  return 0;
}

export function separatedValidationExecutionLevel(facts) {
  const provenanceLevel = Number(facts.provenance_level);
  if (!LEVELS.has(provenanceLevel) || provenanceLevel === 0 || !facts.split_result_present) return 0;
  if (
    !facts.participant_receipt_valid
    || !facts.time_order_valid
    || !facts.artifact_and_recompute_verified
    || !facts.exit_codes_successful
    || !facts.core_result_present
    || facts.direction_critical_failure
  ) return 0;

  const coverage = Number(facts.required_step_coverage);
  if (!Number.isFinite(coverage) || coverage < 0 || coverage > 1) return 0;
  if (coverage === 1) return 1;
  if (coverage >= 0.5 && facts.missing_steps_are_noncritical) return 0.5;
  return 0;
}
