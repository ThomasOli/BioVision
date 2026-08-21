export type ObbInferenceThresholdPlanInput = {
  hasTrainedArtifact: boolean;
  sessionSettingsCustomized: boolean;
  detectionPreset?: string;
  conf: number;
  nmsIou: number;
};

export type ObbInferenceThresholdPlan = {
  source: "artifact" | "explicit";
  detectionPreset: string;
  conf?: number;
  nmsIou?: number;
};

/**
 * Keep validation-calibrated thresholds as the default for trained artifacts.
 *
 * The renderer always carries normalized recommendation values, even when the
 * user has not changed them. Treating those values as explicit would mask the
 * confidence/NMS pair pinned to the immutable model artifact. The internal
 * `custom` preset preserves the calibrated pair without applying another
 * preset-specific threshold transform. A user-customized session (or an
 * explicitly requested custom preset) remains an intentional override.
 */
export function resolveObbInferenceThresholdPlan(
  input: ObbInferenceThresholdPlanInput
): ObbInferenceThresholdPlan {
  const requestedPreset = String(input.detectionPreset || "balanced").trim().toLowerCase();
  const explicitOverride =
    !input.hasTrainedArtifact ||
    input.sessionSettingsCustomized ||
    requestedPreset === "custom";

  if (!explicitOverride) {
    return {
      source: "artifact",
      detectionPreset: "custom",
    };
  }

  return {
    source: "explicit",
    detectionPreset: requestedPreset || "balanced",
    conf: input.conf,
    nmsIou: input.nmsIou,
  };
}
