import type {
  AnnotatedImage,
  ObbDatasetProfile,
  RepresentativeImageDimensions,
  ObbDetectionSettings,
  ObbImageSize,
  ObbModelTier,
  ObbTrainingSettings,
} from "@/types/Image";

type HardwareLike = {
  device?: "cpu" | "mps" | "cuda" | null;
  ramGb?: number | null;
  gpuMemoryGb?: number | null;
};

type ImageDimensionLike = {
  width: number;
  height: number;
};

type ImageProfileLike = RepresentativeImageDimensions | undefined;

export const DEFAULT_OBB_TRAINING_SETTINGS: ObbTrainingSettings = {
  iou: 0.3,
  cls: 1.5,
  box: 5.0,
};

export const DEFAULT_OBB_DETECTION_SETTINGS: ObbDetectionSettings = {
  detectionPreset: "balanced",
  conf: 0.3,
  nmsIou: 0.3,
  maxObjects: 20,
  imgsz: 640,
};

const clampNumber = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

const normalizeImageSize = (value: number | undefined, fallback: ObbImageSize): ObbImageSize => {
  if (value === 960) return 960;
  if (value === 1280) return 1280;
  if (value === 640) return 640;
  return fallback;
};

const normalizeModelTier = (value: ObbModelTier | undefined, fallback?: ObbModelTier): ObbModelTier | undefined => {
  if (value === "nano" || value === "small" || value === "medium" || value === "large") {
    return value;
  }
  return fallback;
};

const sortAscending = (values: number[]) => [...values].sort((a, b) => a - b);

const median = (values: number[]): number | undefined => {
  if (values.length === 0) return undefined;
  const sorted = sortAscending(values);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) return sorted[mid];
  return Math.round((sorted[mid - 1] + sorted[mid]) / 2);
};

const percentile = (values: number[], fraction: number): number | undefined => {
  if (values.length === 0) return undefined;
  const sorted = sortAscending(values);
  const index = Math.min(
    sorted.length - 1,
    Math.max(0, Math.ceil(clampNumber(fraction, 0, 1) * sorted.length) - 1)
  );
  return sorted[index];
};

const polygonArea = (corners: [number, number][]): number => {
  if (corners.length < 3) return 0;
  let twiceArea = 0;
  for (let index = 0; index < corners.length; index += 1) {
    const [x1, y1] = corners[index];
    const [x2, y2] = corners[(index + 1) % corners.length];
    twiceArea += x1 * y2 - x2 * y1;
  }
  return Math.abs(twiceArea) / 2;
};

const getBoxGeometry = (box: AnnotatedImage["boxes"][number]): {
  shortSide: number;
  area: number;
} => {
  if (Array.isArray(box.obbCorners) && box.obbCorners.length === 4) {
    const edgeLengths = box.obbCorners.map(([x1, y1], index) => {
      const [x2, y2] = box.obbCorners![(index + 1) % box.obbCorners!.length];
      return Math.hypot(x2 - x1, y2 - y1);
    });
    return {
      shortSide: Math.min(...edgeLengths),
      area: polygonArea(box.obbCorners),
    };
  }
  return {
    shortSide: Math.min(Math.abs(box.width), Math.abs(box.height)),
    area: Math.abs(box.width * box.height),
  };
};

const boxAabbIou = (
  left: AnnotatedImage["boxes"][number],
  right: AnnotatedImage["boxes"][number]
): number => {
  const intersectionWidth = Math.max(
    0,
    Math.min(left.left + left.width, right.left + right.width) - Math.max(left.left, right.left)
  );
  const intersectionHeight = Math.max(
    0,
    Math.min(left.top + left.height, right.top + right.height) - Math.max(left.top, right.top)
  );
  const intersection = intersectionWidth * intersectionHeight;
  const union = Math.abs(left.width * left.height) + Math.abs(right.width * right.height) - intersection;
  return union > 0 ? intersection / union : 0;
};

/** Build recommendation inputs from annotations already loaded in the workspace. */
export function summarizeObbDatasetProfile(
  images: AnnotatedImage[],
  imageProfile?: ImageProfileLike
): ObbDatasetProfile | undefined {
  const validImages = (images || []).filter((image) => Array.isArray(image?.boxes));
  if (validImages.length === 0) return undefined;

  const finalized = validImages.filter((image) => image.isFinalized);
  const annotated = validImages.filter((image) => image.boxes.length > 0 || image.hasBoxes);
  const scopedImages = finalized.length > 0 ? finalized : annotated;
  if (scopedImages.length === 0) return undefined;

  const objectsPerImage = scopedImages.map((image) => image.boxes.length);
  const boxes = scopedImages.flatMap((image) => image.boxes);
  const geometries = boxes
    .map(getBoxGeometry)
    .filter((geometry) => geometry.shortSide > 0 && geometry.area > 0);
  const imageArea = Math.max(0, Number(imageProfile?.width) * Number(imageProfile?.height));
  const classCounts: Record<string, number> = {};
  for (const box of boxes) {
    const classKey = Number.isFinite(Number(box.class_id))
      ? `class:${Math.round(Number(box.class_id))}`
      : String(box.className || "unclassified").trim().toLowerCase() || "unclassified";
    classCounts[classKey] = (classCounts[classKey] || 0) + 1;
  }
  const nonzeroClassCounts = Object.values(classCounts).filter((count) => count > 0);
  const classImbalanceRatio = nonzeroClassCounts.length > 1
    ? Math.max(...nonzeroClassCounts) / Math.min(...nonzeroClassCounts)
    : undefined;
  const rotatedCount = boxes.filter((box) => {
    if (!Number.isFinite(Number(box.angle))) return false;
    const normalized = Math.abs(Number(box.angle)) % 180;
    return normalized > 1 && normalized < 179;
  }).length;
  let overlappingObjects = 0;
  for (const image of scopedImages) {
    const overlappingIndices = new Set<number>();
    for (let leftIndex = 0; leftIndex < image.boxes.length; leftIndex += 1) {
      for (let rightIndex = leftIndex + 1; rightIndex < image.boxes.length; rightIndex += 1) {
        if (boxAabbIou(image.boxes[leftIndex], image.boxes[rightIndex]) >= 0.1) {
          overlappingIndices.add(leftIndex);
          overlappingIndices.add(rightIndex);
        }
      }
    }
    overlappingObjects += overlappingIndices.size;
  }

  const shortSides = geometries.map((geometry) => geometry.shortSide);
  const areaFractions = imageArea > 0
    ? geometries.map((geometry) => geometry.area / imageArea)
    : [];

  return {
    sampledImageCount: validImages.length,
    annotatedImageCount: scopedImages.filter((image) => image.boxes.length > 0).length,
    objectCount: boxes.length,
    medianObjectsPerImage: median(objectsPerImage),
    p90ObjectsPerImage: percentile(objectsPerImage, 0.9),
    medianObjectShortSidePx: median(shortSides),
    p10ObjectShortSidePx: percentile(shortSides, 0.1),
    ...(areaFractions.length > 0
      ? { medianObjectAreaFraction: percentile(areaFractions, 0.5) }
      : {}),
    classCounts,
    ...(classImbalanceRatio !== undefined ? { classImbalanceRatio } : {}),
    rotatedObjectFraction: boxes.length > 0 ? rotatedCount / boxes.length : 0,
    overlapProxyFraction: boxes.length > 0 ? overlappingObjects / boxes.length : 0,
  };
}

function describeDatasetSignals(
  datasetProfile: ObbDatasetProfile | undefined,
  imageInfo: ReturnType<typeof describeImageProfile>
): {
  crowdedScenes: boolean;
  smallObjects: boolean;
  classImbalanced: boolean;
  overlapHeavy: boolean;
  recommendedObjectCap: number;
  note: string;
} {
  const medianObjects = Number(datasetProfile?.medianObjectsPerImage) || 0;
  const p90Objects = Number(datasetProfile?.p90ObjectsPerImage) || 0;
  const representativeShortSide = Math.min(
    imageInfo.representativeWidth || Number.POSITIVE_INFINITY,
    imageInfo.representativeHeight || Number.POSITIVE_INFINITY
  );
  const p10ShortSide = Number(datasetProfile?.p10ObjectShortSidePx) || 0;
  const shortSideFraction = Number.isFinite(representativeShortSide) && representativeShortSide > 0
    ? p10ShortSide / representativeShortSide
    : 0;
  const medianAreaFraction = Number(datasetProfile?.medianObjectAreaFraction) || 0;
  const crowdedScenes = medianObjects >= 4 || p90Objects >= 8;
  const smallObjects =
    (shortSideFraction > 0 && shortSideFraction <= 0.04) ||
    (medianAreaFraction > 0 && medianAreaFraction <= 0.01);
  const classImbalanced = Number(datasetProfile?.classImbalanceRatio) >= 4;
  const overlapHeavy = Number(datasetProfile?.overlapProxyFraction) >= 0.2;
  const recommendedObjectCap = Math.round(clampNumber(Math.ceil(p90Objects * 1.5), 20, 100));
  const notes = [
    ...(crowdedScenes ? ["crowded scenes"] : []),
    ...(smallObjects ? ["small objects"] : []),
    ...(classImbalanced ? ["class imbalance"] : []),
    ...(overlapHeavy ? ["overlap/occlusion proxy"] : []),
  ];
  return {
    crowdedScenes,
    smallObjects,
    classImbalanced,
    overlapHeavy,
    recommendedObjectCap,
    note: notes.length > 0 ? ` Annotation profile: ${notes.join(", ")}.` : "",
  };
}

function describeImageProfile(imageProfile?: ImageProfileLike): {
  representativeWidth: number;
  representativeHeight: number;
  megapixels: number;
  highResolution: boolean;
  ultraHighResolution: boolean;
} {
  const representativeWidth = Math.max(0, Math.round(Number(imageProfile?.width) || 0));
  const representativeHeight = Math.max(0, Math.round(Number(imageProfile?.height) || 0));
  const megapixels = (representativeWidth * representativeHeight) / 1_000_000;
  const highResolution =
    representativeWidth >= 1280 ||
    representativeHeight >= 960 ||
    megapixels >= 1.0;
  const ultraHighResolution =
    representativeWidth >= 1920 ||
    representativeHeight >= 1280 ||
    megapixels >= 2.0;

  return {
    representativeWidth,
    representativeHeight,
    megapixels,
    highResolution,
    ultraHighResolution,
  };
}

export function summarizeRepresentativeImageDimensions(
  dimensions: ImageDimensionLike[]
): RepresentativeImageDimensions | undefined {
  const normalized = dimensions.filter(
    (entry) =>
      Number.isFinite(Number(entry?.width)) &&
      Number.isFinite(Number(entry?.height)) &&
      Number(entry.width) > 0 &&
      Number(entry.height) > 0
  );
  if (normalized.length === 0) return undefined;

  const widths = normalized.map((entry) => Math.round(Number(entry.width)));
  const heights = normalized.map((entry) => Math.round(Number(entry.height)));
  const width = median(widths) ?? widths[0];
  const height = median(heights) ?? heights[0];

  return {
    width,
    height,
    sampleCount: normalized.length,
    megapixels: Number(((width * height) / 1_000_000).toFixed(2)),
  };
}

export function areObbTrainingSettingsEqual(
  a?: ObbTrainingSettings,
  b?: ObbTrainingSettings
): boolean {
  const left = normalizeObbTrainingSettings(a);
  const right = normalizeObbTrainingSettings(b);
  return JSON.stringify(left) === JSON.stringify(right);
}

export function areObbDetectionSettingsEqual(
  a?: ObbDetectionSettings,
  b?: ObbDetectionSettings
): boolean {
  const left = normalizeObbDetectionSettings(a);
  const right = normalizeObbDetectionSettings(b);
  return JSON.stringify(left) === JSON.stringify(right);
}

export function normalizeObbTrainingSettings(
  settings?: ObbTrainingSettings,
  fallback: ObbTrainingSettings = DEFAULT_OBB_TRAINING_SETTINGS
): ObbTrainingSettings {
  return {
    ...fallback,
    ...(normalizeModelTier(settings?.modelTier, fallback.modelTier)
      ? { modelTier: normalizeModelTier(settings?.modelTier, fallback.modelTier) }
      : {}),
    ...(settings?.imgsz !== undefined
      ? { imgsz: normalizeImageSize(settings.imgsz, fallback.imgsz ?? 640) }
      : {}),
    ...(Number.isFinite(Number(settings?.epochs))
      ? { epochs: Math.round(clampNumber(Number(settings?.epochs), 1, 500)) }
      : {}),
    ...(Number.isFinite(Number(settings?.batch))
      ? { batch: Math.round(clampNumber(Number(settings?.batch), 1, 128)) }
      : {}),
    iou: clampNumber(Number(settings?.iou ?? fallback.iou ?? 0.3), 0.05, 0.95),
    cls: clampNumber(Number(settings?.cls ?? fallback.cls ?? 1.5), 0.1, 10),
    box: clampNumber(Number(settings?.box ?? fallback.box ?? 5.0), 0.1, 20),
  };
}

export function normalizeObbDetectionSettings(
  settings?: ObbDetectionSettings,
  fallback: ObbDetectionSettings = DEFAULT_OBB_DETECTION_SETTINGS
): ObbDetectionSettings {
  const preset = settings?.detectionPreset ?? fallback.detectionPreset ?? "balanced";
  return {
    detectionPreset:
      preset === "precision" ||
      preset === "recall" ||
      preset === "single_object" ||
      preset === "custom"
        ? preset
        : "balanced",
    conf: clampNumber(Number(settings?.conf ?? fallback.conf ?? 0.3), 0.01, 0.99),
    nmsIou: clampNumber(Number(settings?.nmsIou ?? fallback.nmsIou ?? 0.3), 0.05, 0.95),
    maxObjects: Math.round(clampNumber(Number(settings?.maxObjects ?? fallback.maxObjects ?? 20), 1, 250)),
    imgsz: normalizeImageSize(settings?.imgsz, fallback.imgsz ?? 640),
  };
}

export function getRecommendedObbTrainingSettings(
  imageCount: number,
  hardware: HardwareLike,
  imageProfile?: ImageProfileLike,
  datasetProfile?: ObbDatasetProfile
): { settings: ObbTrainingSettings; summary: string } {
  const device = hardware.device ?? "cpu";
  const gpuMemoryGb = hardware.gpuMemoryGb ?? 0;
  const largeDataset = imageCount >= 400;
  const moderateDataset = imageCount >= 200;
  const highEnd = device === "cuda" && gpuMemoryGb >= 8;
  const veryHighEnd = device === "cuda" && gpuMemoryGb >= 12;
  const imageInfo = describeImageProfile(imageProfile);
  const datasetSignals = describeDatasetSignals(datasetProfile, imageInfo);
  const highResolution = imageInfo.highResolution;
  const ultraHighResolution = imageInfo.ultraHighResolution;
  const needsFineDetail =
    ultraHighResolution || datasetSignals.smallObjects || datasetSignals.overlapHeavy;
  const lowResolution =
    imageInfo.representativeWidth > 0 &&
    imageInfo.representativeHeight > 0 &&
    imageInfo.representativeWidth <= 800 &&
    imageInfo.representativeHeight <= 600 &&
    imageInfo.megapixels <= 0.5;
  const imageDescriptor =
    imageInfo.representativeWidth > 0 && imageInfo.representativeHeight > 0
      ? `${imageInfo.representativeWidth}x${imageInfo.representativeHeight}`
      : null;

  if (device === "cpu") {
    return {
      settings: normalizeObbTrainingSettings({
        modelTier: "nano",
        imgsz: needsFineDetail ? 960 : 640,
        epochs: 30,
        batch: needsFineDetail ? 4 : 6,
      }),
      summary: imageDescriptor
        ? `CPU-only + ${imageDescriptor}: start with the lightest viable OBB settings to keep training stable.${datasetSignals.note}`
        : `CPU-only: start with the lightest viable OBB settings.${datasetSignals.note}`,
    };
  }

  if (needsFineDetail && largeDataset && veryHighEnd) {
    return {
      settings: normalizeObbTrainingSettings({
        modelTier: "medium",
        imgsz: 960,
        epochs: 100,
        batch: 8,
      }),
      summary: imageDescriptor
        ? `Large fine-detail dataset (${imageDescriptor}) on a 12+ GB GPU: medium at 960 is justified, but keep the starting point conservative.${datasetSignals.note}`
        : `Large fine-detail dataset on a 12+ GB GPU: medium at 960 is the minimum stronger starting point.${datasetSignals.note}`,
    };
  }

  if (highResolution || datasetSignals.smallObjects) {
    return {
      settings: normalizeObbTrainingSettings({
        modelTier: "small",
        imgsz: 960,
        epochs: largeDataset ? 100 : 75,
        batch: highEnd ? 8 : 6,
      }),
      summary: imageDescriptor
        ? `Fine-detail ${imageDescriptor} imagery: start with a small model at 960 before considering a larger tier.${datasetSignals.note}`
        : `Fine-detail imagery: start with a small model at 960 before scaling up.${datasetSignals.note}`,
    };
  }

  if (moderateDataset || !lowResolution) {
    return {
      settings: normalizeObbTrainingSettings({
        modelTier: "small",
        imgsz: 640,
        epochs: moderateDataset ? 75 : 50,
        batch: highEnd ? 10 : 8,
      }),
      summary: imageDescriptor
        ? `Moderate ${imageDescriptor} imagery: start with a small model at 640 for faster iteration.${datasetSignals.note}`
        : `Moderate detector dataset: start with a small model at 640 for faster iteration.${datasetSignals.note}`,
    };
  }

  return {
    settings: normalizeObbTrainingSettings({
      modelTier: "small",
      imgsz: 640,
      epochs: 50,
      batch: highEnd ? 10 : 8,
    }),
    summary: imageDescriptor
      ? `Low-resolution ${imageDescriptor} imagery: start with the minimum sufficient OBB setup for faster startup and iteration.${datasetSignals.note}`
      : `Start with the minimum sufficient OBB setup for faster startup and iteration.${datasetSignals.note}`,
  };
}

export function getRecommendedObbDetectionSettings(
  imageCount: number,
  imageProfile?: ImageProfileLike,
  datasetProfile?: ObbDatasetProfile
): { settings: ObbDetectionSettings; summary: string } {
  // Kept in the signature for API compatibility. Dataset volume is a training
  // signal; it must not be mistaken for per-image object density at inference.
  void imageCount;
  const imageInfo = describeImageProfile(imageProfile);
  const datasetSignals = describeDatasetSignals(datasetProfile, imageInfo);
  const highResolution = imageInfo.highResolution;
  const ultraHighResolution = imageInfo.ultraHighResolution;
  const lowResolution =
    imageInfo.representativeWidth > 0 &&
    imageInfo.representativeHeight > 0 &&
    imageInfo.representativeWidth <= 800 &&
    imageInfo.representativeHeight <= 600 &&
    imageInfo.megapixels <= 0.5;
  const imageDescriptor =
    imageInfo.representativeWidth > 0 && imageInfo.representativeHeight > 0
      ? `${imageInfo.representativeWidth}x${imageInfo.representativeHeight}`
      : null;

  if (ultraHighResolution || datasetSignals.smallObjects) {
    return {
      settings: normalizeObbDetectionSettings({
        detectionPreset: datasetSignals.crowdedScenes ? "recall" : "custom",
        conf: 0.25,
        nmsIou: 0.35,
        maxObjects: datasetSignals.recommendedObjectCap,
        imgsz: 960,
      }),
      summary: imageDescriptor
        ? `Fine-detail ${imageDescriptor} imagery: start at 960; use Recall only when the annotation profile shows crowded scenes.${datasetSignals.note}`
        : `Fine-detail imagery: start at 960; use Recall only when the annotation profile shows crowded scenes.${datasetSignals.note}`,
    };
  }

  if (datasetSignals.crowdedScenes) {
    return {
      settings: normalizeObbDetectionSettings({
        detectionPreset: "recall",
        conf: 0.25,
        nmsIou: 0.3,
        maxObjects: datasetSignals.recommendedObjectCap,
        imgsz: highResolution ? 960 : 640,
      }),
      summary: imageDescriptor
        ? `Crowded ${imageDescriptor} scenes: use Recall with an object cap derived from annotations.${datasetSignals.note}`
        : `Crowded scenes: use Recall with an object cap derived from annotations.${datasetSignals.note}`,
    };
  }

  if (highResolution || !lowResolution) {
    return {
      settings: normalizeObbDetectionSettings({
        detectionPreset: highResolution ? "custom" : "balanced",
        conf: highResolution ? 0.28 : 0.3,
        nmsIou: 0.3,
        maxObjects: datasetSignals.recommendedObjectCap,
        imgsz: highResolution ? 960 : 640,
      }),
      summary: imageDescriptor
        ? `${imageDescriptor} imagery: start conservatively and use Recall only when validated misses justify it.${datasetSignals.note}`
        : `Start conservatively and use Recall only when validated misses justify it.${datasetSignals.note}`,
      };
  }

  return {
    settings: normalizeObbDetectionSettings({
      detectionPreset: "balanced",
      conf: 0.3,
      nmsIou: 0.3,
      maxObjects: 20,
      imgsz: 640,
    }),
    summary: `Balanced works well for sparse scenes; switch to Recall or Custom when validated objects are missed.${datasetSignals.note}`,
  };
}
