import type { LandmarkDefinition, OrientationPolicy } from "@/types/Image";

export const SCHEMA_SEMANTIC_VERSION = 2 as const;

const normalizeText = (value: unknown): string =>
  String(value ?? "").trim().toLowerCase();

const normalizeIntegerList = (values: unknown): number[] => {
  if (!Array.isArray(values)) return [];
  return [...new Set(
    values
      .map((value) => Number(value))
      .filter((value) => Number.isFinite(value))
      .map((value) => Math.round(value))
  )].sort((left, right) => left - right);
};

const normalizeTextList = (values: unknown): string[] => {
  if (!Array.isArray(values)) return [];
  return [...new Set(values.map(normalizeText).filter(Boolean))].sort();
};

const normalizeBilateralPairs = (values: unknown): [number, number][] => {
  if (!Array.isArray(values)) return [];
  const normalized = values
    .filter((pair): pair is unknown[] => Array.isArray(pair) && pair.length === 2)
    .map((pair) => pair.map((value) => Math.round(Number(value))))
    .filter((pair) => pair.every((value) => Number.isFinite(value)))
    .map(([left, right]) => (
      left <= right ? [left, right] as [number, number] : [right, left] as [number, number]
    ));
  const unique = new Map(normalized.map((pair) => [`${pair[0]}:${pair[1]}`, pair]));
  return [...unique.values()].sort((left, right) => left[0] - right[0] || left[1] - right[1]);
};

const fnv1a64 = (input: string): string => {
  let hash = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  for (let index = 0; index < input.length; index += 1) {
    hash ^= BigInt(input.charCodeAt(index));
    hash = BigInt.asUintN(64, hash * prime);
  }
  return hash.toString(16).padStart(16, "0");
};

/** Legacy landmark-only identity retained for existing session routing. */
export function computeLegacySchemaFingerprint(
  landmarkTemplate: LandmarkDefinition[]
): string {
  const normalized = (landmarkTemplate || []).map((landmark, position) => ({
    index: Number.isFinite(Number(landmark?.index))
      ? Math.max(1, Number(landmark.index))
      : position + 1,
    name: normalizeText(landmark?.name),
    category: normalizeText(landmark?.category),
  }));

  let hash = 2166136261;
  const input = JSON.stringify(normalized);
  for (let index = 0; index < input.length; index += 1) {
    hash ^= input.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

/**
 * Canonical training semantics. Description text is intentionally excluded;
 * landmark identity/category and every orientation behavior field are included.
 */
export function canonicalizeSchemaSemantics(
  landmarkTemplate: LandmarkDefinition[],
  orientationPolicy: OrientationPolicy
): string {
  const landmarks = (landmarkTemplate || [])
    .map((landmark, position) => ({
      index: Number.isFinite(Number(landmark?.index))
        ? Math.max(1, Math.round(Number(landmark.index)))
        : position + 1,
      name: normalizeText(landmark?.name),
      category: normalizeText(landmark?.category),
      required: landmark?.required !== false,
    }))
    .sort((left, right) => (
      left.index - right.index ||
      left.name.localeCompare(right.name) ||
      left.category.localeCompare(right.category)
    ));
  const mode = orientationPolicy?.mode;
  const policy = {
    mode,
    targetOrientation:
      orientationPolicy?.targetOrientation === "right" ? "right" :
      orientationPolicy?.targetOrientation === "left" ? "left" : null,
    headCategories: normalizeTextList(orientationPolicy?.headCategories),
    tailCategories: normalizeTextList(orientationPolicy?.tailCategories),
    anteriorAnchorIds: normalizeIntegerList(orientationPolicy?.anteriorAnchorIds),
    posteriorAnchorIds: normalizeIntegerList(orientationPolicy?.posteriorAnchorIds),
    bilateralPairs: normalizeBilateralPairs(orientationPolicy?.bilateralPairs),
    bilateralClassAxis: orientationPolicy?.bilateralClassAxis ?? null,
    obbLevelingMode: orientationPolicy?.obbLevelingMode ?? null,
  };
  return JSON.stringify({
    version: SCHEMA_SEMANTIC_VERSION,
    landmarks,
    orientationPolicy: policy,
  });
}

export function computeSchemaSemanticFingerprint(
  landmarkTemplate: LandmarkDefinition[],
  orientationPolicy: OrientationPolicy
): string {
  return `v${SCHEMA_SEMANTIC_VERSION}-${fnv1a64(
    canonicalizeSchemaSemantics(landmarkTemplate, orientationPolicy)
  )}`;
}
