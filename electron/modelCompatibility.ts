export type SchemaContractIssue = {
  code: string;
  severity: "error" | "warning";
  message: string;
};

type LandmarkContractEntry = {
  index: number;
  name?: string;
  required?: boolean;
};

function normalizeName(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  const normalized = value.trim().toLowerCase();
  return normalized || undefined;
}

function normalizeLandmarkContract(value: unknown): LandmarkContractEntry[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((entry: any, position) => {
      const index = Number.isFinite(Number(entry?.index))
        ? Math.max(1, Math.round(Number(entry.index)))
        : position + 1;
      return {
        index,
        ...(normalizeName(entry?.name) ? { name: normalizeName(entry.name) } : {}),
        ...(typeof entry?.required === "boolean" ? { required: entry.required } : {}),
      };
    })
    .sort((left, right) => left.index - right.index);
}

function normalizeLandmarkIds(value: unknown): LandmarkContractEntry[] {
  if (!Array.isArray(value)) return [];
  return [...new Set(
    value
      .map(Number)
      .filter(Number.isFinite)
      .map((entry) => Math.max(1, Math.round(entry)))
  )]
    .sort((left, right) => left - right)
    .map((index) => ({ index }));
}

/** Compare the immutable training-time schema contract with the active session. */
export function compareModelSchemaContract(args: {
  sessionSemanticFingerprint?: unknown;
  sessionLandmarks: unknown;
  modelSemanticFingerprint?: unknown;
  modelSemanticVersion?: unknown;
  modelLandmarks?: unknown;
  modelLandmarkIds?: unknown;
}): SchemaContractIssue[] {
  const issues: SchemaContractIssue[] = [];
  const sessionFingerprint = String(args.sessionSemanticFingerprint || "").trim();
  const modelFingerprint = String(args.modelSemanticFingerprint || "").trim();
  const modelSemanticVersion = Number(args.modelSemanticVersion);

  if (modelFingerprint) {
    if (modelSemanticVersion !== 2 || !modelFingerprint.startsWith("v2-")) {
      issues.push({
        code: "model_schema_semantic_version_unsupported",
        severity: "error",
        message: "Model schema semantics are not in the supported v2 format.",
      });
    } else if (!sessionFingerprint) {
      issues.push({
        code: "session_schema_semantic_fingerprint_missing",
        severity: "error",
        message: "The active session has no semantic schema fingerprint to validate this model.",
      });
    } else if (modelFingerprint !== sessionFingerprint) {
      issues.push({
        code: "schema_semantic_fingerprint_mismatch",
        severity: "error",
        message: "Model and session schema semantics differ (landmarks, required flags, or orientation policy).",
      });
    }
  } else {
    issues.push({
      code: "legacy_model_schema_fingerprint_missing",
      severity: "warning",
      message: "Legacy model has no v2 schema fingerprint; validating its recoverable landmark contract instead.",
    });
  }

  const sessionContract = normalizeLandmarkContract(args.sessionLandmarks);
  const explicitModelContract = normalizeLandmarkContract(args.modelLandmarks);
  const modelContract = explicitModelContract.length > 0
    ? explicitModelContract
    : normalizeLandmarkIds(args.modelLandmarkIds);
  if (sessionContract.length === 0 || modelContract.length === 0) {
    issues.push({
      code: "model_landmark_contract_missing",
      severity: "error",
      message: "Model landmark IDs/names cannot be verified against the active session.",
    });
    return issues;
  }

  const sessionIds = sessionContract.map((entry) => entry.index);
  const modelIds = modelContract.map((entry) => entry.index);
  if (JSON.stringify(sessionIds) !== JSON.stringify(modelIds)) {
    issues.push({
      code: "landmark_id_contract_mismatch",
      severity: "error",
      message: `Model landmark IDs [${modelIds.join(", ")}] do not match session IDs [${sessionIds.join(", ")}].`,
    });
    return issues;
  }

  const modelHasNames = modelContract.every((entry) => Boolean(entry.name));
  if (modelHasNames) {
    const sessionNames = sessionContract.map((entry) => entry.name || "");
    const modelNames = modelContract.map((entry) => entry.name || "");
    if (JSON.stringify(sessionNames) !== JSON.stringify(modelNames)) {
      issues.push({
        code: "landmark_name_contract_mismatch",
        severity: "error",
        message: "Model landmark names do not match the active session for the same IDs.",
      });
    }
  } else {
    issues.push({
      code: "legacy_model_landmark_names_unverified",
      severity: "warning",
      message: "Legacy model landmark names are unavailable; only landmark IDs could be validated.",
    });
  }

  const modelHasRequiredFlags = modelContract.every(
    (entry) => typeof entry.required === "boolean"
  );
  if (modelHasRequiredFlags) {
    const sessionRequired = sessionContract.map((entry) => entry.required !== false);
    const modelRequired = modelContract.map((entry) => entry.required !== false);
    if (JSON.stringify(sessionRequired) !== JSON.stringify(modelRequired)) {
      issues.push({
        code: "landmark_required_contract_mismatch",
        severity: "error",
        message: "Model required/optional landmark semantics do not match the active session.",
      });
    }
  }

  return issues;
}
