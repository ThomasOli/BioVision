import assert from "node:assert/strict";
import test from "node:test";

import { resolveObbInferenceThresholdPlan } from "./obbInferenceOptions";

test("trained artifacts use their validation-calibrated thresholds by default", () => {
  const plan = resolveObbInferenceThresholdPlan({
    hasTrainedArtifact: true,
    sessionSettingsCustomized: false,
    detectionPreset: "balanced",
    conf: 0.3,
    nmsIou: 0.3,
  });

  assert.deepEqual(plan, {
    source: "artifact",
    detectionPreset: "custom",
  });
});

test("user-customized thresholds remain an explicit artifact override", () => {
  const plan = resolveObbInferenceThresholdPlan({
    hasTrainedArtifact: true,
    sessionSettingsCustomized: true,
    detectionPreset: "precision",
    conf: 0.61,
    nmsIou: 0.42,
  });

  assert.deepEqual(plan, {
    source: "explicit",
    detectionPreset: "precision",
    conf: 0.61,
    nmsIou: 0.42,
  });
});

test("zero-shot detection retains normalized session thresholds", () => {
  const plan = resolveObbInferenceThresholdPlan({
    hasTrainedArtifact: false,
    sessionSettingsCustomized: false,
    detectionPreset: "recall",
    conf: 0.2,
    nmsIou: 0.72,
  });

  assert.equal(plan.source, "explicit");
  assert.equal(plan.conf, 0.2);
  assert.equal(plan.nmsIou, 0.72);
  assert.equal(plan.detectionPreset, "recall");
});
