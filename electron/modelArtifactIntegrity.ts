import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";

export type ArtifactIntegrityIssue = {
  code: string;
  message: string;
};

const LANDMARK_ID_MAPPING_FORMAT = "biovision.landmark-id-mapping.v1";
const LANDMARK_ID_MAPPING_FILENAME = "id_mapping.json";

function readJson(filePath: string): any | null {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch {
    return null;
  }
}

function fileSha256(filePath: string): string {
  return createHash("sha256").update(fs.readFileSync(filePath)).digest("hex");
}

function resolvePath(basePath: string, value: unknown): string | undefined {
  const raw = String(value || "").trim();
  if (!raw) return undefined;
  return path.isAbsolute(raw) ? raw : path.resolve(path.dirname(basePath), raw);
}

function normalizedSha(value: unknown): string {
  return String(value || "").trim().toLowerCase();
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function normalizedRelativePath(value: unknown): string {
  return String(value || "").trim().replace(/\\/g, "/");
}

function resolveDescriptorPath(
  manifestPath: string,
  descriptor: Record<string, unknown>
): string | undefined {
  const explicitPath = resolvePath(manifestPath, descriptor.path);
  const relativePath = normalizedRelativePath(descriptor.relativePath);
  const relativeResolved = relativePath
    ? path.resolve(path.dirname(manifestPath), ...relativePath.split("/"))
    : undefined;
  if (
    explicitPath &&
    relativeResolved &&
    path.resolve(explicitPath) !== path.resolve(relativeResolved)
  ) {
    return undefined;
  }
  return explicitPath || relativeResolved;
}

/** Verify an immutable landmark registry record before changing runtime aliases. */
export function validateLandmarkPromotionArtifact(
  entry: Record<string, any>
): ArtifactIntegrityIssue[] {
  const issues: ArtifactIntegrityIssue[] = [];
  const artifactPath = String(entry.path || "").trim();
  const manifestPath = String(entry.runManifestPath || "").trim();
  if (!artifactPath || !fs.existsSync(artifactPath)) {
    return [{ code: "landmark_artifact_missing", message: "Immutable landmark artifact is missing." }];
  }
  if (!manifestPath || !fs.existsSync(manifestPath)) {
    return [{ code: "landmark_manifest_missing", message: "Immutable landmark manifest is missing." }];
  }
  const manifest = readJson(manifestPath);
  if (!manifest || typeof manifest !== "object") {
    return [{ code: "landmark_manifest_invalid", message: "Immutable landmark manifest is invalid." }];
  }

  const actualArtifactSha = fileSha256(artifactPath);
  const registryArtifactSha = normalizedSha(entry.artifact?.sha256);
  const manifestArtifactSha = normalizedSha(manifest.artifact?.sha256);
  if (!registryArtifactSha || !manifestArtifactSha) {
    issues.push({
      code: "landmark_artifact_hash_missing",
      message: "Registry or manifest artifact SHA-256 is missing.",
    });
  }
  if (
    (registryArtifactSha && registryArtifactSha !== actualArtifactSha) ||
    (manifestArtifactSha && manifestArtifactSha !== actualArtifactSha) ||
    (registryArtifactSha && manifestArtifactSha && registryArtifactSha !== manifestArtifactSha)
  ) {
    issues.push({
      code: "landmark_artifact_hash_mismatch",
      message: "Registry, manifest, and immutable landmark artifact hashes do not agree.",
    });
  }

  const manifestArtifactPath = resolvePath(manifestPath, manifest.artifact?.path);
  if (
    manifestArtifactPath &&
    path.resolve(manifestArtifactPath) !== path.resolve(artifactPath)
  ) {
    issues.push({
      code: "landmark_artifact_path_mismatch",
      message: "Registry and manifest landmark artifact paths do not agree.",
    });
  }
  if (!entry.modelId || !manifest.modelId || entry.modelId !== manifest.modelId) {
    issues.push({
      code: "landmark_model_id_mismatch",
      message: "Registry and manifest immutable model IDs do not agree.",
    });
  }
  if (entry.runId && manifest.runId && entry.runId !== manifest.runId) {
    issues.push({
      code: "landmark_run_id_mismatch",
      message: "Registry and manifest training run IDs do not agree.",
    });
  }

  const configPath = String(entry.configPath || "").trim();
  if (entry.predictorType === "cnn" || configPath || manifest.config) {
    const manifestConfigPath = resolvePath(manifestPath, manifest.config?.path);
    if (!configPath || !fs.existsSync(configPath) || !manifestConfigPath) {
      issues.push({
        code: "landmark_config_missing",
        message: "Immutable landmark configuration is missing.",
      });
    } else {
      const actualConfigSha = fileSha256(configPath);
      const registryConfigSha = normalizedSha(entry.config?.sha256);
      const manifestConfigSha = normalizedSha(manifest.config?.sha256);
      if (
        !registryConfigSha ||
        !manifestConfigSha ||
        registryConfigSha !== actualConfigSha ||
        manifestConfigSha !== actualConfigSha ||
        path.resolve(manifestConfigPath) !== path.resolve(configPath)
      ) {
        issues.push({
          code: "landmark_config_integrity_mismatch",
          message: "Registry, manifest, and immutable landmark configuration do not agree.",
        });
      }
    }
  }

  const manifestSidecars = isRecord(manifest.sidecars)
    ? manifest.sidecars
    : {};
  const registrySidecars = isRecord(entry.sidecars)
    ? entry.sidecars
    : {};

  const manifestSidecarNames = Object.keys(manifestSidecars).sort();
  const registrySidecarNames = Object.keys(registrySidecars).sort();
  if (JSON.stringify(manifestSidecarNames) !== JSON.stringify(registrySidecarNames)) {
    issues.push({
      code: "landmark_sidecar_set_mismatch",
      message: "Registry and manifest immutable landmark sidecar sets do not agree.",
    });
  }

  const manifestIdMapping = manifestSidecars.idMapping;
  const registryIdMapping = registrySidecars.idMapping;
  if (!isRecord(manifestIdMapping) || !isRecord(registryIdMapping)) {
    issues.push({
      code: "landmark_id_mapping_sidecar_missing",
      message: "The mandatory immutable landmark ID mapping sidecar is missing.",
    });
  } else {
    if (
      manifestIdMapping.format !== LANDMARK_ID_MAPPING_FORMAT ||
      registryIdMapping.format !== LANDMARK_ID_MAPPING_FORMAT
    ) {
      issues.push({
        code: "landmark_id_mapping_format_unsupported",
        message: "The immutable landmark ID mapping sidecar format is missing or unsupported.",
      });
    }
    const manifestRelativePath = normalizedRelativePath(manifestIdMapping.relativePath);
    const registryRelativePath = normalizedRelativePath(registryIdMapping.relativePath);
    const manifestExplicitPath = resolvePath(manifestPath, manifestIdMapping.path);
    const registryExplicitPath = resolvePath(manifestPath, registryIdMapping.path);
    const manifestIdMappingPath = resolveDescriptorPath(manifestPath, manifestIdMapping);
    const registryIdMappingPath = resolveDescriptorPath(manifestPath, registryIdMapping);
    if (
      manifestRelativePath !== LANDMARK_ID_MAPPING_FILENAME ||
      registryRelativePath !== LANDMARK_ID_MAPPING_FILENAME ||
      !manifestExplicitPath ||
      !registryExplicitPath ||
      !manifestIdMappingPath ||
      !registryIdMappingPath ||
      path.resolve(manifestIdMappingPath) !== path.resolve(registryIdMappingPath)
    ) {
      issues.push({
        code: "landmark_id_mapping_path_mismatch",
        message: "Registry and manifest immutable landmark ID mapping paths do not agree.",
      });
    }
  }

  for (const name of [...new Set([...manifestSidecarNames, ...registrySidecarNames])]) {
    const manifestDescriptor = manifestSidecars[name];
    const registryDescriptor = registrySidecars[name];
    if (!isRecord(manifestDescriptor) || !isRecord(registryDescriptor)) {
      continue;
    }
    const manifestSidecarPath = resolveDescriptorPath(manifestPath, manifestDescriptor);
    const registrySidecarPath = resolveDescriptorPath(manifestPath, registryDescriptor);
    const manifestSha = normalizedSha(manifestDescriptor.sha256);
    const registrySha = normalizedSha(registryDescriptor.sha256);
    if (
      !manifestSidecarPath ||
      !registrySidecarPath ||
      path.resolve(manifestSidecarPath) !== path.resolve(registrySidecarPath) ||
      !fs.existsSync(manifestSidecarPath) ||
      !manifestSha ||
      !registrySha ||
      fileSha256(manifestSidecarPath) !== manifestSha ||
      manifestSha !== registrySha
    ) {
      issues.push({
        code: "landmark_sidecar_integrity_mismatch",
        message: `Immutable landmark sidecar "${name}" failed integrity verification.`,
      });
    }
  }
  return issues;
}
