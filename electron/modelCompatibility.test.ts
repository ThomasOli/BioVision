import assert from "node:assert/strict";
import test from "node:test";

import { compareModelSchemaContract } from "./modelCompatibility";

const sessionLandmarks = [
  { index: 1, name: "Head", required: true },
  { index: 2, name: "Tail", required: false },
];

test("v2 schema fingerprint mismatch is blocking", () => {
  const issues = compareModelSchemaContract({
    sessionSemanticFingerprint: "v2-session0000001",
    sessionLandmarks,
    modelSemanticFingerprint: "v2-model000000001",
    modelSemanticVersion: 2,
    modelLandmarks: sessionLandmarks,
  });
  assert.ok(issues.some((issue) =>
    issue.code === "schema_semantic_fingerprint_mismatch" && issue.severity === "error"
  ));
});

test("landmark IDs, names and required semantics are exact contracts", () => {
  const nameIssues = compareModelSchemaContract({
    sessionSemanticFingerprint: "v2-same",
    sessionLandmarks,
    modelSemanticFingerprint: "v2-same",
    modelSemanticVersion: 2,
    modelLandmarks: [
      { index: 1, name: "Snout", required: true },
      { index: 2, name: "Tail", required: false },
    ],
  });
  assert.ok(nameIssues.some((issue) => issue.code === "landmark_name_contract_mismatch"));

  const requiredIssues = compareModelSchemaContract({
    sessionSemanticFingerprint: "v2-same",
    sessionLandmarks,
    modelSemanticFingerprint: "v2-same",
    modelSemanticVersion: 2,
    modelLandmarks: sessionLandmarks.map((entry) => ({ ...entry, required: true })),
  });
  assert.ok(requiredIssues.some((issue) => issue.code === "landmark_required_contract_mismatch"));
});

test("legacy model IDs can be validated while missing names remain visible", () => {
  const issues = compareModelSchemaContract({
    sessionLandmarks,
    modelLandmarkIds: [1, 2],
  });
  assert.equal(issues.some((issue) => issue.severity === "error"), false);
  assert.ok(issues.some((issue) => issue.code === "legacy_model_landmark_names_unverified"));
});
