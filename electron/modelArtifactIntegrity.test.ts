import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { validateLandmarkPromotionArtifact } from "./modelArtifactIntegrity";

const ID_MAPPING_FORMAT = "biovision.landmark-id-mapping.v1";

function fixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "biovision-landmark-integrity-"));
  const artifactPath = path.join(root, "model.dat");
  const idMappingPath = path.join(root, "id_mapping.json");
  const manifestPath = path.join(root, "manifest.json");
  fs.writeFileSync(artifactPath, "landmark-model");
  fs.writeFileSync(idMappingPath, JSON.stringify({ original_ids: [1, 2] }));
  const sha256 = createHash("sha256").update(fs.readFileSync(artifactPath)).digest("hex");
  const idMappingSha256 = createHash("sha256")
    .update(fs.readFileSync(idMappingPath))
    .digest("hex");
  const idMappingDescriptor = {
    format: ID_MAPPING_FORMAT,
    path: idMappingPath,
    relativePath: "id_mapping.json",
    sha256: idMappingSha256,
  };
  const manifest = {
    modelId: "dlib:run-1",
    runId: "run-1",
    artifact: { path: artifactPath, sha256 },
    sidecars: { idMapping: { ...idMappingDescriptor } },
  };
  fs.writeFileSync(manifestPath, JSON.stringify(manifest));
  const entry = {
    modelId: "dlib:run-1",
    runId: "run-1",
    predictorType: "dlib",
    path: artifactPath,
    artifact: { path: artifactPath, sha256 },
    runManifestPath: manifestPath,
    sidecars: { idMapping: { ...idMappingDescriptor } },
  };
  return { root, artifactPath, idMappingPath, manifestPath, manifest, entry };
}

test("manual landmark promotion requires registry, manifest and actual hashes to agree", (t) => {
  const files = fixture();
  t.after(() => fs.rmSync(files.root, { recursive: true, force: true }));
  const { entry } = files;
  assert.deepEqual(validateLandmarkPromotionArtifact(entry), []);

  entry.artifact.sha256 = "0".repeat(64);
  assert.ok(validateLandmarkPromotionArtifact(entry).some(
    (issue) => issue.code === "landmark_artifact_hash_mismatch"
  ));
});

test("manual landmark promotion requires the mandatory ID mapping in both contracts", (t) => {
  const files = fixture();
  t.after(() => fs.rmSync(files.root, { recursive: true, force: true }));

  delete (files.manifest.sidecars as Record<string, unknown>).idMapping;
  fs.writeFileSync(files.manifestPath, JSON.stringify(files.manifest));
  const issues = validateLandmarkPromotionArtifact(files.entry);
  assert.ok(issues.some((issue) => issue.code === "landmark_sidecar_set_mismatch"));
  assert.ok(issues.some((issue) => issue.code === "landmark_id_mapping_sidecar_missing"));
});

test("manual landmark promotion rejects registry-only sidecars", (t) => {
  const files = fixture();
  t.after(() => fs.rmSync(files.root, { recursive: true, force: true }));
  (files.entry.sidecars as Record<string, Record<string, string>>).extra = {
    path: files.idMappingPath,
    relativePath: "id_mapping.json",
    sha256: files.entry.sidecars.idMapping.sha256,
  };

  assert.ok(validateLandmarkPromotionArtifact(files.entry).some(
    (issue) => issue.code === "landmark_sidecar_set_mismatch"
  ));
});

test("manual landmark promotion verifies ID mapping format, path and bytes", (t) => {
  const files = fixture();
  t.after(() => fs.rmSync(files.root, { recursive: true, force: true }));

  files.entry.sidecars.idMapping.format = "unsupported";
  files.entry.sidecars.idMapping.relativePath = "different.json";
  fs.writeFileSync(files.idMappingPath, "tampered");
  const issues = validateLandmarkPromotionArtifact(files.entry);
  assert.ok(issues.some((issue) => issue.code === "landmark_id_mapping_format_unsupported"));
  assert.ok(issues.some((issue) => issue.code === "landmark_id_mapping_path_mismatch"));
  assert.ok(issues.some((issue) => issue.code === "landmark_sidecar_integrity_mismatch"));
});
