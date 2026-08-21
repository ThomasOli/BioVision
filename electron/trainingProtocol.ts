export type ModelLifecycleStatus = "active" | "candidate" | "deprecated";

export type ModelPromotionDetails = {
  promoted?: boolean;
  reason?: string;
  metric?: string | null;
  candidateScore?: number | null;
  baselineScore?: number | null;
  baselineModelId?: string | null;
  [key: string]: unknown;
};

export type LandmarkTrainingPublication = {
  modelId?: string;
  artifactTag?: string;
  modelStatus?: ModelLifecycleStatus;
  promotion?: ModelPromotionDetails;
};

function readLastMarker(output: string, marker: string): string | undefined {
  const prefix = `${marker} `;
  const lines = String(output || "").split(/\r?\n/);
  for (let index = lines.length - 1; index >= 0; index -= 1) {
    const line = lines[index].trim();
    if (line.startsWith(prefix)) {
      const value = line.slice(prefix.length).trim();
      if (value) return value;
    }
  }
  return undefined;
}

/** Parse the stable stdout publication markers emitted by landmark trainers. */
export function parseLandmarkTrainingPublication(
  output: string
): LandmarkTrainingPublication {
  const modelId = readLastMarker(output, "MODEL_ID");
  const artifactTag = readLastMarker(output, "MODEL_TAG");
  const rawStatus = readLastMarker(output, "MODEL_STATUS");
  const modelStatus =
    rawStatus === "active" || rawStatus === "candidate" || rawStatus === "deprecated"
      ? rawStatus
      : undefined;
  const rawPromotion = readLastMarker(output, "PROMOTION_JSON");
  let promotion: ModelPromotionDetails | undefined;
  if (rawPromotion) {
    try {
      const parsed = JSON.parse(rawPromotion);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        promotion = parsed as ModelPromotionDetails;
      }
    } catch {
      // Older or interrupted trainers may emit no usable structured record.
    }
  }
  return {
    ...(modelId ? { modelId } : {}),
    ...(artifactTag ? { artifactTag } : {}),
    ...(modelStatus ? { modelStatus } : {}),
    ...(promotion ? { promotion } : {}),
  };
}
