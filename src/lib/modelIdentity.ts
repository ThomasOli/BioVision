import type { TrainedModel } from "@/types/Image";

/** Legacy key reader retained so persisted pre-registry preferences still resolve. */
export const getLegacyModelKey = (model: TrainedModel): string =>
  `${model.name}::${model.predictorType ?? "dlib"}`;

/** Immutable model IDs are preferred; legacy scans continue to use name/type. */
export const getModelKey = (model: TrainedModel): string =>
  String(model.modelId || "").trim() || getLegacyModelKey(model);

/** Runtime lookup must use the immutable artifact tag, never a mutable label. */
export const getModelArtifactTag = (model: TrainedModel): string =>
  String(model.artifactTag || "").trim() || model.name;

export const modelMatchesKey = (model: TrainedModel, key: string): boolean =>
  getModelKey(model) === key || getLegacyModelKey(model) === key;
