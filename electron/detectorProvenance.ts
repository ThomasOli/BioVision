import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";

export type DetectionModelProvenance = {
  modelId: string;
  artifactSha256: string | null;
  configSha256?: string | null;
  displayName: string;
  kind: "trained_obb" | "zero_shot";
};

export type ObbDetectorContractIssue = {
  code: string;
  severity: "error" | "warning";
  message: string;
};

export type ObbDetectorResolution = {
  provenance: DetectionModelProvenance;
  compatible: boolean;
  blocking: boolean;
  issues: ObbDetectorContractIssue[];
  registryEntry?: Record<string, unknown>;
  configPath?: string;
  schemaSemanticFingerprint?: string;
  schemaSemanticVersion?: number;
  orientationContract?: Record<string, unknown>;
};

/**
 * Landmark training is downstream of the active OBB detector.  An alias file
 * by itself is not enough to unlock that stage: it must map to one immutable,
 * active registry artifact and satisfy the current schema/orientation
 * contract without warnings or blocking integrity errors.
 */
export function isVerifiedTrainedObbDetector(
  resolution: ObbDetectorResolution | null | undefined
): boolean {
  return Boolean(
    resolution &&
      resolution.provenance.kind === "trained_obb" &&
      resolution.provenance.artifactSha256 &&
      resolution.compatible &&
      !resolution.blocking &&
      resolution.issues.length === 0
  );
}

const normalizeStringList = (value: unknown): string[] => {
  if (!Array.isArray(value)) return [];
  return [...new Set(
    value
      .map((item) => String(item ?? "").trim().toLowerCase())
      .filter(Boolean)
  )].sort();
};

const normalizeIntegerList = (value: unknown): number[] => {
  if (!Array.isArray(value)) return [];
  return [...new Set(
    value
      .map(Number)
      .filter(Number.isFinite)
      .map((item) => Math.round(item))
  )].sort((left, right) => left - right);
};

/** Normalize the orientation-only part of the immutable schema contract. */
export function normalizeObbOrientationContract(
  value: unknown
): Record<string, unknown> | undefined {
  if (!value || typeof value !== "object") return undefined;
  const raw = value as Record<string, unknown>;
  const mode = String(raw.mode ?? "").trim().toLowerCase();
  if (!new Set(["directional", "bilateral", "axial", "invariant"]).has(mode)) {
    return undefined;
  }

  const bilateralPairs = Array.isArray(raw.bilateralPairs)
    ? raw.bilateralPairs
        .filter((pair): pair is unknown[] => Array.isArray(pair) && pair.length === 2)
        .map((pair) => pair.map(Number))
        .filter((pair) => pair.every(Number.isFinite))
        .map(([left, right]) => (
          left <= right
            ? [Math.round(left), Math.round(right)]
            : [Math.round(right), Math.round(left)]
        ))
    : [];
  const uniquePairs = new Map(
    bilateralPairs.map((pair) => [`${pair[0]}:${pair[1]}`, pair])
  );

  return {
    mode,
    ...(mode === "directional"
      ? { targetOrientation: raw.targetOrientation === "right" ? "right" : "left" }
      : {}),
    headCategories: normalizeStringList(raw.headCategories),
    tailCategories: normalizeStringList(raw.tailCategories),
    anteriorAnchorIds: normalizeIntegerList(raw.anteriorAnchorIds),
    posteriorAnchorIds: normalizeIntegerList(raw.posteriorAnchorIds),
    bilateralPairs: [...uniquePairs.values()].sort(
      (left, right) => left[0] - right[0] || left[1] - right[1]
    ),
    bilateralClassAxis: String(raw.bilateralClassAxis || "vertical_obb")
      .trim()
      .toLowerCase(),
    obbLevelingMode:
      String(raw.obbLevelingMode || "on").trim().toLowerCase() === "off"
        ? "off"
        : "on",
  };
}

function readJson(filePath: string): any | null {
  try {
    if (!fs.existsSync(filePath)) return null;
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch {
    return null;
  }
}

function sha256File(filePath: string): string {
  return createHash("sha256").update(fs.readFileSync(filePath)).digest("hex");
}

function resolveEntryPath(registryPath: string, rawPath: unknown): string | undefined {
  const value = String(rawPath || "").trim();
  if (!value) return undefined;
  return path.isAbsolute(value) ? value : path.resolve(path.dirname(registryPath), value);
}

type EntryContract = {
  schemaSemanticFingerprint?: string;
  schemaSemanticVersion?: number;
  orientationContract?: Record<string, unknown>;
  registryArtifactSha256?: string;
  manifestArtifactSha256?: string;
  registryConfigSha256?: string;
  manifestConfigSha256?: string;
  actualConfigSha256?: string;
  configPath?: string;
  issues: ObbDetectorContractIssue[];
};

function readEntryContract(
  registryPath: string,
  entry: Record<string, any>
): EntryContract {
  const issues: ObbDetectorContractIssue[] = [];
  const manifestPath = resolveEntryPath(registryPath, entry.manifestPath);
  const manifest = manifestPath ? readJson(manifestPath) : null;
  if (!manifestPath || !manifest) {
    issues.push({
      code: "obb_artifact_manifest_missing",
      severity: "error",
      message: "The active OBB registry record has no readable immutable artifact manifest.",
    });
  }
  const lineageSchema = manifest?.lineage?.schema;
  const registryFingerprint = String(entry.schemaSemanticFingerprint || "").trim();
  const directManifestFingerprint = String(
    manifest?.schemaSemanticFingerprint || ""
  ).trim();
  const lineageFingerprint = String(lineageSchema?.semanticFingerprint || "").trim();
  const manifestFingerprint = directManifestFingerprint || lineageFingerprint;
  if (
    directManifestFingerprint &&
    lineageFingerprint &&
    directManifestFingerprint !== lineageFingerprint
  ) {
    issues.push({
      code: "obb_manifest_lineage_schema_mismatch",
      severity: "error",
      message: "The OBB artifact manifest contradicts its embedded schema lineage.",
    });
  }
  if (
    registryFingerprint &&
    manifestFingerprint &&
    registryFingerprint !== manifestFingerprint
  ) {
    issues.push({
      code: "obb_registry_manifest_schema_mismatch",
      severity: "error",
      message: "The OBB registry and artifact manifest disagree on schema identity.",
    });
  }

  const registryVersion = Number(entry.schemaSemanticVersion);
  const directManifestVersion = Number(manifest?.schemaSemanticVersion);
  const lineageVersion = Number(lineageSchema?.semanticVersion);
  const manifestVersion = Number.isFinite(directManifestVersion)
    ? directManifestVersion
    : lineageVersion;
  if (
    Number.isFinite(directManifestVersion) &&
    Number.isFinite(lineageVersion) &&
    directManifestVersion !== lineageVersion
  ) {
    issues.push({
      code: "obb_manifest_lineage_version_mismatch",
      severity: "error",
      message: "The OBB artifact manifest contradicts its embedded schema version.",
    });
  }
  if (
    Number.isFinite(registryVersion) &&
    Number.isFinite(manifestVersion) &&
    registryVersion !== manifestVersion
  ) {
    issues.push({
      code: "obb_registry_manifest_version_mismatch",
      severity: "error",
      message: "The OBB registry and artifact manifest disagree on schema version.",
    });
  }

  const registryOrientation = normalizeObbOrientationContract(
    entry.orientationContract
  );
  const directManifestOrientation = normalizeObbOrientationContract(
    manifest?.orientationContract
  );
  const lineageOrientation = normalizeObbOrientationContract(
    lineageSchema?.semantics?.orientationPolicy
  );
  const manifestOrientation = directManifestOrientation || lineageOrientation;
  if (
    directManifestOrientation &&
    lineageOrientation &&
    JSON.stringify(directManifestOrientation) !== JSON.stringify(lineageOrientation)
  ) {
    issues.push({
      code: "obb_manifest_lineage_orientation_mismatch",
      severity: "error",
      message: "The OBB artifact manifest contradicts its embedded orientation lineage.",
    });
  }
  if (
    registryOrientation &&
    manifestOrientation &&
    JSON.stringify(registryOrientation) !== JSON.stringify(manifestOrientation)
  ) {
    issues.push({
      code: "obb_registry_manifest_orientation_mismatch",
      severity: "error",
      message: "The OBB registry and artifact manifest disagree on orientation semantics.",
    });
  }

  const registryArtifactSha256 = String(
    entry.artifactSha256 ?? entry.artifact?.sha256 ?? ""
  ).trim().toLowerCase();
  const manifestArtifactSha256 = String(
    manifest?.artifact?.sha256 ?? ""
  ).trim().toLowerCase();
  if (
    registryArtifactSha256 &&
    manifestArtifactSha256 &&
    registryArtifactSha256 !== manifestArtifactSha256
  ) {
    issues.push({
      code: "obb_registry_manifest_artifact_hash_mismatch",
      severity: "error",
      message: "The OBB registry and artifact manifest disagree on artifact hash.",
    });
  }

  const manifestModelId = String(manifest?.modelId || "").trim();
  const registryModelId = String(entry.modelId || "").trim();
  if (manifest && !manifestModelId) {
    issues.push({
      code: "obb_manifest_model_id_missing",
      severity: "error",
      message: "The OBB artifact manifest has no immutable model ID.",
    });
  } else if (manifestModelId && registryModelId && manifestModelId !== registryModelId) {
    issues.push({
      code: "obb_registry_manifest_model_id_mismatch",
      severity: "error",
      message: "The OBB registry and artifact manifest disagree on model ID.",
    });
  }
  const manifestRunId = String(manifest?.runId || "").trim();
  const registryRunId = String(entry.runId || "").trim();
  if (manifestRunId && registryRunId && manifestRunId !== registryRunId) {
    issues.push({
      code: "obb_registry_manifest_run_id_mismatch",
      severity: "error",
      message: "The OBB registry and artifact manifest disagree on training run ID.",
    });
  }

  const registryArtifactPath = resolveEntryPath(registryPath, entry.path);
  const manifestArtifactRaw = String(manifest?.artifact?.path || "").trim();
  const manifestArtifactPath = manifestArtifactRaw
    ? path.isAbsolute(manifestArtifactRaw)
      ? manifestArtifactRaw
      : path.resolve(path.dirname(manifestPath || registryPath), manifestArtifactRaw)
    : undefined;
  if (
    registryArtifactPath &&
    manifestArtifactPath &&
    path.resolve(registryArtifactPath) !== path.resolve(manifestArtifactPath)
  ) {
    issues.push({
      code: "obb_registry_manifest_artifact_path_mismatch",
      severity: "error",
      message: "The OBB registry and artifact manifest disagree on artifact path.",
    });
  }

  const registryConfigDescriptor = entry.config;
  const manifestConfigDescriptor = manifest?.config;
  let registryConfigSha256: string | undefined;
  let manifestConfigSha256: string | undefined;
  let actualConfigSha256: string | undefined;
  let configPath: string | undefined;
  if (
    !registryConfigDescriptor ||
    typeof registryConfigDescriptor !== "object" ||
    Array.isArray(registryConfigDescriptor) ||
    !manifestConfigDescriptor ||
    typeof manifestConfigDescriptor !== "object" ||
    Array.isArray(manifestConfigDescriptor)
  ) {
    issues.push({
      code: "obb_config_descriptor_missing",
      severity: "error",
      message: "The OBB config descriptor is missing from its registry or artifact manifest.",
    });
  } else {
    registryConfigSha256 = String(registryConfigDescriptor.sha256 || "")
      .trim()
      .toLowerCase();
    manifestConfigSha256 = String(manifestConfigDescriptor.sha256 || "")
      .trim()
      .toLowerCase();
    const registryConfigPath = resolveEntryPath(
      registryPath,
      registryConfigDescriptor.path
    );
    const manifestConfigPath = resolveEntryPath(
      manifestPath || registryPath,
      manifestConfigDescriptor.path
    );
    const registryConfigCompatibilityPath = resolveEntryPath(registryPath, entry.configPath);
    const registryRelativePath = String(registryConfigDescriptor.relativePath || "")
      .trim()
      .replace(/\\/g, "/");
    const manifestRelativePath = String(manifestConfigDescriptor.relativePath || "")
      .trim()
      .replace(/\\/g, "/");
    const expectedConfigPath = manifestPath
      ? path.resolve(path.dirname(manifestPath), "obb_config.json")
      : undefined;
    configPath = registryConfigPath || manifestConfigPath;
    if (!registryConfigPath || !manifestConfigPath) {
      issues.push({
        code: "obb_config_path_missing",
        severity: "error",
        message: "The OBB config path is missing from its registry or artifact manifest.",
      });
    } else if (
      path.resolve(registryConfigPath) !== path.resolve(manifestConfigPath) ||
      registryRelativePath !== "obb_config.json" ||
      manifestRelativePath !== "obb_config.json" ||
      (expectedConfigPath && path.resolve(registryConfigPath) !== expectedConfigPath) ||
      (registryConfigCompatibilityPath &&
        path.resolve(registryConfigCompatibilityPath) !== path.resolve(registryConfigPath))
    ) {
      issues.push({
        code: "obb_registry_manifest_config_path_mismatch",
        severity: "error",
        message: "The OBB registry and artifact manifest disagree on config path.",
      });
    }
    if (!registryConfigSha256 || !manifestConfigSha256) {
      issues.push({
        code: "obb_config_hash_missing",
        severity: "error",
        message: "The OBB config SHA-256 is missing from its registry or artifact manifest.",
      });
    } else if (registryConfigSha256 !== manifestConfigSha256) {
      issues.push({
        code: "obb_registry_manifest_config_hash_mismatch",
        severity: "error",
        message: "The OBB registry and artifact manifest disagree on config hash.",
      });
    }
    if (!configPath || !fs.existsSync(configPath)) {
      issues.push({
        code: "obb_config_artifact_missing",
        severity: "error",
        message: "The immutable OBB config artifact is missing.",
      });
    } else {
      actualConfigSha256 = sha256File(configPath);
      if (
        (registryConfigSha256 && registryConfigSha256 !== actualConfigSha256) ||
        (manifestConfigSha256 && manifestConfigSha256 !== actualConfigSha256)
      ) {
        issues.push({
          code: "obb_config_hash_mismatch",
          severity: "error",
          message: "The immutable OBB config failed SHA-256 verification.",
        });
      }
    }
  }

  const fingerprint = manifestFingerprint || registryFingerprint;
  const version = Number.isFinite(manifestVersion) ? manifestVersion : registryVersion;
  const orientationContract = manifestOrientation || registryOrientation;
  return {
    ...(fingerprint ? { schemaSemanticFingerprint: fingerprint } : {}),
    ...(Number.isFinite(version) ? { schemaSemanticVersion: version } : {}),
    ...(orientationContract ? { orientationContract } : {}),
    ...(registryArtifactSha256 ? { registryArtifactSha256 } : {}),
    ...(manifestArtifactSha256 ? { manifestArtifactSha256 } : {}),
    ...(registryConfigSha256 ? { registryConfigSha256 } : {}),
    ...(manifestConfigSha256 ? { manifestConfigSha256 } : {}),
    ...(actualConfigSha256 ? { actualConfigSha256 } : {}),
    ...(configPath ? { configPath } : {}),
    issues,
  };
}

function appendSessionContractIssues(
  issues: ObbDetectorContractIssue[],
  contract: EntryContract,
  args: {
    sessionSemanticFingerprint?: unknown;
    sessionSemanticVersion?: unknown;
    sessionOrientationContract?: unknown;
  }
): void {
  const sessionFingerprint = String(args.sessionSemanticFingerprint || "").trim();
  const sessionVersion = Number(args.sessionSemanticVersion);
  if (
    !contract.schemaSemanticFingerprint ||
    !Number.isFinite(Number(contract.schemaSemanticVersion)) ||
    !contract.orientationContract
  ) {
    issues.push({
      code: "legacy_obb_schema_contract_missing",
      severity: "error",
      message:
        "The active OBB detector predates immutable schema/orientation metadata and cannot be verified safely.",
    });
    return;
  }

  if (
    contract.schemaSemanticVersion !== 2 ||
    !contract.schemaSemanticFingerprint.startsWith("v2-")
  ) {
    issues.push({
      code: "obb_schema_semantic_version_unsupported",
      severity: "error",
      message: "The active OBB detector uses an unsupported schema semantic version.",
    });
  }
  if (!sessionFingerprint || sessionVersion !== 2) {
    issues.push({
      code: "session_schema_semantic_contract_missing",
      severity: "error",
      message: "The active session has no supported v2 schema semantic contract.",
    });
  } else if (contract.schemaSemanticFingerprint !== sessionFingerprint) {
    issues.push({
      code: "obb_schema_semantic_fingerprint_mismatch",
      severity: "error",
      message:
        "The active OBB detector was trained for different landmark/orientation semantics.",
    });
  }

  const sessionOrientation = normalizeObbOrientationContract(
    args.sessionOrientationContract
  );
  if (!sessionOrientation) {
    issues.push({
      code: "session_orientation_contract_missing",
      severity: "error",
      message: "The active session orientation contract cannot be verified.",
    });
  } else if (
    JSON.stringify(contract.orientationContract) !== JSON.stringify(sessionOrientation)
  ) {
    issues.push({
      code: "obb_orientation_contract_mismatch",
      severity: "error",
      message: "The active OBB detector orientation contract differs from the session.",
    });
  }
}

export function resolveTrainedObbDetector(args: {
  aliasPath: string;
  configAliasPath?: string;
  registryPath: string;
  sessionSemanticFingerprint?: unknown;
  sessionSemanticVersion?: unknown;
  sessionOrientationContract?: unknown;
}): ObbDetectorResolution {
  const aliasSha256 = sha256File(args.aliasPath);
  const registry = readJson(args.registryPath);
  const models = Array.isArray(registry?.models) ? registry.models : [];
  const activeEntries = models.filter(
    (entry: any) => entry && typeof entry === "object" && entry.status === "active"
  );
  const matchingEntries = activeEntries.filter((entry: any) => {
    const artifactPath = resolveEntryPath(args.registryPath, entry.path);
    if (!artifactPath || !fs.existsSync(artifactPath)) return false;
    try {
      return sha256File(artifactPath) === aliasSha256;
    } catch {
      return false;
    }
  });
  const matchedEntry = matchingEntries[0] as Record<string, any> | undefined;
  const issues: ObbDetectorContractIssue[] = [];

  if (!matchedEntry) {
    issues.push({
      code: "obb_active_alias_registry_mismatch",
      severity: "error",
      message:
        "The session OBB alias bytes do not map to an active immutable registry record.",
    });
  } else if (matchingEntries.length > 1) {
    issues.push({
      code: "obb_multiple_active_registry_matches",
      severity: "error",
      message: "More than one active OBB registry record maps to the detector alias.",
    });
  }

  const modelId = String(matchedEntry?.modelId || "").trim();
  if (matchedEntry && !modelId) {
    issues.push({
      code: "obb_active_model_id_missing",
      severity: "error",
      message: "The active OBB registry record has no immutable model ID.",
    });
  }

  const contract: EntryContract = matchedEntry
    ? readEntryContract(args.registryPath, matchedEntry)
    : { issues: [] };
  issues.push(...contract.issues);
  if (
    (contract.registryArtifactSha256 && contract.registryArtifactSha256 !== aliasSha256) ||
    (contract.manifestArtifactSha256 && contract.manifestArtifactSha256 !== aliasSha256)
  ) {
    issues.push({
      code: "obb_artifact_hash_mismatch",
      severity: "error",
      message: "The active OBB artifact hash does not match the detector bytes in use.",
    });
  }
  const configAliasPath = args.configAliasPath || path.join(
    path.dirname(args.aliasPath),
    `${path.basename(args.aliasPath, path.extname(args.aliasPath))}_config.json`
  );
  let configAliasSha256: string | null = null;
  if (!fs.existsSync(configAliasPath)) {
    issues.push({
      code: "obb_active_config_alias_missing",
      severity: "error",
      message: "The active OBB config alias is missing.",
    });
  } else {
    configAliasSha256 = sha256File(configAliasPath);
    if (
      (contract.registryConfigSha256 &&
        contract.registryConfigSha256 !== configAliasSha256) ||
      (contract.manifestConfigSha256 &&
        contract.manifestConfigSha256 !== configAliasSha256) ||
      (contract.actualConfigSha256 &&
        contract.actualConfigSha256 !== configAliasSha256)
    ) {
      issues.push({
        code: "obb_active_config_alias_mismatch",
        severity: "error",
        message: "The active OBB config alias does not match its immutable registry record.",
      });
    }
  }

  appendSessionContractIssues(issues, contract, args);

  const blocking = issues.some((issue) => issue.severity === "error");
  return {
    provenance: {
      modelId: modelId || `obb:sha256:${aliasSha256}`,
      artifactSha256: aliasSha256,
      configSha256: configAliasSha256,
      displayName: String(matchedEntry?.name || "Session OBB Detector"),
      kind: "trained_obb",
    },
    compatible: !blocking && issues.length === 0,
    blocking,
    issues,
    ...(matchedEntry ? { registryEntry: matchedEntry } : {}),
    ...(contract.configPath ? { configPath: contract.configPath } : {}),
    ...(contract.schemaSemanticFingerprint
      ? { schemaSemanticFingerprint: contract.schemaSemanticFingerprint }
      : {}),
    ...(contract.schemaSemanticVersion != null
      ? { schemaSemanticVersion: contract.schemaSemanticVersion }
      : {}),
    ...(contract.orientationContract
      ? { orientationContract: contract.orientationContract }
      : {}),
  };
}

export function validateObbPromotionCandidate(args: {
  registryPath: string;
  modelIdentifier: string;
  sessionSemanticFingerprint?: unknown;
  sessionSemanticVersion?: unknown;
  sessionOrientationContract?: unknown;
}): ObbDetectorResolution {
  const registry = readJson(args.registryPath);
  const models = Array.isArray(registry?.models) ? registry.models : [];
  const entry = models.find(
    (candidate: any) =>
      candidate?.modelId === args.modelIdentifier ||
      candidate?.path === args.modelIdentifier
  ) as Record<string, any> | undefined;
  const issues: ObbDetectorContractIssue[] = [];
  if (!entry) {
    issues.push({
      code: "obb_promotion_candidate_missing",
      severity: "error",
      message: "The requested immutable OBB registry record does not exist.",
    });
  }

  const artifactPath = entry
    ? resolveEntryPath(args.registryPath, entry.path)
    : undefined;
  let actualArtifactSha256: string | null = null;
  if (!artifactPath || !fs.existsSync(artifactPath)) {
    issues.push({
      code: "obb_promotion_artifact_missing",
      severity: "error",
      message: "The requested immutable OBB artifact is missing.",
    });
  } else {
    actualArtifactSha256 = sha256File(artifactPath);
  }

  const contract: EntryContract = entry
    ? readEntryContract(args.registryPath, entry)
    : { issues: [] };
  issues.push(...contract.issues);
  if (
    actualArtifactSha256 &&
    ((contract.registryArtifactSha256 &&
      contract.registryArtifactSha256 !== actualArtifactSha256) ||
      (contract.manifestArtifactSha256 &&
        contract.manifestArtifactSha256 !== actualArtifactSha256))
  ) {
    issues.push({
      code: "obb_artifact_hash_mismatch",
      severity: "error",
      message: "The requested OBB artifact failed immutable hash verification.",
    });
  }
  if (entry && !String(entry.modelId || "").trim()) {
    issues.push({
      code: "obb_active_model_id_missing",
      severity: "error",
      message: "The requested OBB registry record has no immutable model ID.",
    });
  }
  appendSessionContractIssues(issues, contract, args);
  const blocking = issues.some((issue) => issue.severity === "error");
  return {
    provenance: {
      modelId: String(entry?.modelId || args.modelIdentifier || "obb:unverified"),
      artifactSha256: actualArtifactSha256,
      configSha256: contract.actualConfigSha256 || null,
      displayName: String(entry?.name || "Session OBB Detector"),
      kind: "trained_obb",
    },
    compatible: !blocking && issues.length === 0,
    blocking,
    issues,
    ...(entry ? { registryEntry: entry } : {}),
    ...(contract.configPath ? { configPath: contract.configPath } : {}),
    ...(contract.schemaSemanticFingerprint
      ? { schemaSemanticFingerprint: contract.schemaSemanticFingerprint }
      : {}),
    ...(contract.schemaSemanticVersion != null
      ? { schemaSemanticVersion: contract.schemaSemanticVersion }
      : {}),
    ...(contract.orientationContract
      ? { orientationContract: contract.orientationContract }
      : {}),
  };
}

export function resolveZeroShotDetector(
  artifactPath: string,
  detectionMethod = "yolo_world"
): DetectionModelProvenance {
  const normalizedMethod = String(detectionMethod || "yolo_world")
    .split("+")[0]
    .trim()
    .toLowerCase();
  const isYoloWorld = normalizedMethod === "yolo_world";
  let artifactSha256: string | null = null;
  try {
    if (fs.existsSync(artifactPath)) artifactSha256 = sha256File(artifactPath);
  } catch {
    artifactSha256 = null;
  }
  return {
    modelId: isYoloWorld
      ? "zero-shot:ultralytics:yolov8s-worldv2.pt"
      : `zero-shot:${normalizedMethod || "unknown"}`,
    artifactSha256,
    displayName: isYoloWorld
      ? "Ultralytics YOLO-World v2 (zero-shot)"
      : `${normalizedMethod || "Unknown"} zero-shot detector`,
    kind: "zero_shot",
  };
}
