import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  resolveTrainedObbDetector,
  resolveZeroShotDetector,
  validateObbPromotionCandidate,
} from "./detectorProvenance";

const policy = {
  mode: "directional",
  targetOrientation: "left",
  headCategories: ["head"],
  tailCategories: ["tail"],
  anteriorAnchorIds: [1],
  posteriorAnchorIds: [2],
  bilateralPairs: [],
  bilateralClassAxis: "vertical_obb",
  obbLevelingMode: "on",
};

function fixture(options?: {
  contract?: boolean;
  aliasBytes?: string;
  configAliasBytes?: string;
}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "biovision-obb-contract-"));
  const artifactPath = path.join(root, "artifact.pt");
  const aliasPath = path.join(root, "session_obb_detector.pt");
  const configPath = path.join(root, "obb_config.json");
  const configAliasPath = path.join(root, "session_obb_detector_config.json");
  const manifestPath = path.join(root, "manifest.json");
  const registryPath = path.join(root, "obb_registry.json");
  fs.writeFileSync(artifactPath, "artifact-bytes");
  fs.writeFileSync(aliasPath, options?.aliasBytes ?? "artifact-bytes");
  fs.writeFileSync(configPath, "config-bytes");
  fs.writeFileSync(configAliasPath, options?.configAliasBytes ?? "config-bytes");
  const artifactSha256 = createHash("sha256")
    .update(fs.readFileSync(artifactPath))
    .digest("hex");
  const configSha256 = createHash("sha256")
    .update(fs.readFileSync(configPath))
    .digest("hex");
  fs.writeFileSync(manifestPath, JSON.stringify({
    modelId: "obb:run-1",
    runId: "run-1",
    artifact: { path: artifactPath, sha256: artifactSha256 },
    config: {
      path: configPath,
      relativePath: "obb_config.json",
      sha256: configSha256,
    },
    ...(options?.contract === false ? {} : {
      schemaSemanticFingerprint: "v2-fish",
      schemaSemanticVersion: 2,
      orientationContract: policy,
    }),
  }));
  fs.writeFileSync(registryPath, JSON.stringify({
    version: 2,
    models: [{
      modelId: "obb:run-1",
      runId: "run-1",
      name: "Fish OBB",
      path: artifactPath,
      configPath,
      config: {
        path: configPath,
        relativePath: "obb_config.json",
        sha256: configSha256,
      },
      manifestPath,
      status: "active",
      ...(options?.contract === false ? {} : {
        artifactSha256,
        schemaSemanticFingerprint: "v2-fish",
        schemaSemanticVersion: 2,
        orientationContract: policy,
      }),
    }],
  }));
  return { root, aliasPath, configPath, configAliasPath, registryPath };
}

test("maps exact active alias bytes to immutable OBB provenance and contract", (t) => {
  const files = fixture();
  t.after(() => fs.rmSync(files.root, { recursive: true, force: true }));
  const resolved = resolveTrainedObbDetector({
    aliasPath: files.aliasPath,
    registryPath: files.registryPath,
    sessionSemanticFingerprint: "v2-fish",
    sessionSemanticVersion: 2,
    sessionOrientationContract: policy,
  });

  assert.equal(resolved.blocking, false);
  assert.equal(resolved.provenance.modelId, "obb:run-1");
  assert.match(resolved.provenance.artifactSha256 || "", /^[a-f0-9]{64}$/);
});

test("blocks an alias that does not map to the active immutable artifact", (t) => {
  const files = fixture({ aliasBytes: "different-alias-bytes" });
  t.after(() => fs.rmSync(files.root, { recursive: true, force: true }));
  const resolved = resolveTrainedObbDetector({
    aliasPath: files.aliasPath,
    registryPath: files.registryPath,
    sessionSemanticFingerprint: "v2-fish",
    sessionSemanticVersion: 2,
    sessionOrientationContract: policy,
  });

  assert.equal(resolved.blocking, true);
  assert.ok(resolved.issues.some((issue) => issue.code === "obb_active_alias_registry_mismatch"));
});

test("blocks an active config alias that does not match immutable config bytes", (t) => {
  const files = fixture({ configAliasBytes: "different-config-bytes" });
  t.after(() => fs.rmSync(files.root, { recursive: true, force: true }));
  const resolved = resolveTrainedObbDetector({
    aliasPath: files.aliasPath,
    registryPath: files.registryPath,
    sessionSemanticFingerprint: "v2-fish",
    sessionSemanticVersion: 2,
    sessionOrientationContract: policy,
  });

  assert.equal(resolved.blocking, true);
  assert.ok(resolved.issues.some(
    (issue) => issue.code === "obb_active_config_alias_mismatch"
  ));
});

test("blocks legacy OBB records whose schema/orientation contract is missing", (t) => {
  const files = fixture({ contract: false });
  t.after(() => fs.rmSync(files.root, { recursive: true, force: true }));
  const resolved = resolveTrainedObbDetector({
    aliasPath: files.aliasPath,
    registryPath: files.registryPath,
    sessionSemanticFingerprint: "v2-fish",
    sessionSemanticVersion: 2,
    sessionOrientationContract: policy,
  });

  assert.ok(resolved.issues.some((issue) => issue.code === "legacy_obb_schema_contract_missing"));
});

test("blocks semantic and orientation-only OBB contract mismatches", (t) => {
  const files = fixture();
  t.after(() => fs.rmSync(files.root, { recursive: true, force: true }));
  const resolved = resolveTrainedObbDetector({
    aliasPath: files.aliasPath,
    registryPath: files.registryPath,
    sessionSemanticFingerprint: "v2-other-schema",
    sessionSemanticVersion: 2,
    sessionOrientationContract: { ...policy, targetOrientation: "right" },
  });

  assert.ok(resolved.issues.some(
    (issue) => issue.code === "obb_schema_semantic_fingerprint_mismatch"
  ));
  assert.ok(resolved.issues.some(
    (issue) => issue.code === "obb_orientation_contract_mismatch"
  ));
});

test("resolves relative manifests and blocks registry/manifest contradictions", (t) => {
  const files = fixture();
  t.after(() => fs.rmSync(files.root, { recursive: true, force: true }));
  const registry = JSON.parse(fs.readFileSync(files.registryPath, "utf8"));
  registry.models[0].manifestPath = "manifest.json";
  registry.models[0].schemaSemanticFingerprint = "v2-registry-conflict";
  registry.models[0].artifactSha256 = "0".repeat(64);
  registry.models[0].modelId = "obb:registry-conflict";
  fs.writeFileSync(files.registryPath, JSON.stringify(registry));

  const resolved = resolveTrainedObbDetector({
    aliasPath: files.aliasPath,
    registryPath: files.registryPath,
    sessionSemanticFingerprint: "v2-fish",
    sessionSemanticVersion: 2,
    sessionOrientationContract: policy,
  });

  assert.ok(resolved.issues.some(
    (issue) => issue.code === "obb_registry_manifest_schema_mismatch"
  ));
  assert.ok(resolved.issues.some(
    (issue) => issue.code === "obb_registry_manifest_artifact_hash_mismatch"
  ));
  assert.ok(resolved.issues.some(
    (issue) => issue.code === "obb_registry_manifest_model_id_mismatch"
  ));
});

test("manual OBB promotion validates candidate bytes and schema before alias mutation", (t) => {
  const files = fixture();
  t.after(() => fs.rmSync(files.root, { recursive: true, force: true }));
  const registry = JSON.parse(fs.readFileSync(files.registryPath, "utf8"));
  registry.models[0].status = "candidate";
  fs.writeFileSync(files.registryPath, JSON.stringify(registry));
  const resolved = validateObbPromotionCandidate({
    registryPath: files.registryPath,
    modelIdentifier: "obb:run-1",
    sessionSemanticFingerprint: "v2-fish",
    sessionSemanticVersion: 2,
    sessionOrientationContract: policy,
  });
  assert.equal(resolved.blocking, false);

  fs.writeFileSync(registry.models[0].path, "tampered");
  const tampered = validateObbPromotionCandidate({
    registryPath: files.registryPath,
    modelIdentifier: "obb:run-1",
    sessionSemanticFingerprint: "v2-fish",
    sessionSemanticVersion: 2,
    sessionOrientationContract: policy,
  });
  assert.ok(tampered.issues.some((issue) => issue.code === "obb_artifact_hash_mismatch"));
});

test("manual OBB promotion validates config descriptor path, hash and bytes", (t) => {
  const files = fixture();
  t.after(() => fs.rmSync(files.root, { recursive: true, force: true }));
  const registry = JSON.parse(fs.readFileSync(files.registryPath, "utf8"));
  registry.models[0].status = "candidate";
  registry.models[0].config.path = path.join(files.root, "wrong-config.json");
  fs.writeFileSync(files.registryPath, JSON.stringify(registry));

  const wrongPath = validateObbPromotionCandidate({
    registryPath: files.registryPath,
    modelIdentifier: "obb:run-1",
    sessionSemanticFingerprint: "v2-fish",
    sessionSemanticVersion: 2,
    sessionOrientationContract: policy,
  });
  assert.ok(wrongPath.issues.some(
    (issue) => issue.code === "obb_registry_manifest_config_path_mismatch"
  ));

  registry.models[0].config.path = files.configPath;
  fs.writeFileSync(files.registryPath, JSON.stringify(registry));
  fs.writeFileSync(files.configPath, "tampered-config");
  const tampered = validateObbPromotionCandidate({
    registryPath: files.registryPath,
    modelIdentifier: "obb:run-1",
    sessionSemanticFingerprint: "v2-fish",
    sessionSemanticVersion: 2,
    sessionOrientationContract: policy,
  });
  assert.ok(tampered.issues.some((issue) => issue.code === "obb_config_hash_mismatch"));
});

test("zero-shot detector identity is explicit and hashes the exact local weights", (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "biovision-zero-shot-"));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const weights = path.join(root, "yolov8s-worldv2.pt");
  fs.writeFileSync(weights, "world-weights");
  const provenance = resolveZeroShotDetector(weights, "yolo_world+sam2");
  assert.equal(provenance.modelId, "zero-shot:ultralytics:yolov8s-worldv2.pt");
  assert.match(provenance.artifactSha256 || "", /^[a-f0-9]{64}$/);
  assert.equal(provenance.kind, "zero_shot");
});
