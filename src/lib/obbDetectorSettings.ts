import type {
  RepresentativeImageDimensions,
  ObbDetectionSettings,
  ObbImageSize,
  ObbModelTier,
  ObbTrainingSettings,
} from "@/types/Image";

type HardwareLike = {
  device?: "cpu" | "mps" | "cuda" | null;
  ramGb?: number | null;
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
  imageProfile?: ImageProfileLike
): { settings: ObbTrainingSettings; summary: string } {
  const device = hardware.device ?? "cpu";
  const ramGb = hardware.ramGb ?? 0;
  const denseDataset = imageCount >= 400;
  const moderateDataset = imageCount >= 200;
  const highEnd = device === "cuda" && ramGb >= 16;
  const veryHighEnd = device === "cuda" && ramGb >= 24;
  const imageInfo = describeImageProfile(imageProfile);
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

  if (device === "cpu") {
    return {
      settings: normalizeObbTrainingSettings({
        modelTier: "nano",
        imgsz: ultraHighResolution ? 960 : 640,
        epochs: 30,
        batch: ultraHighResolution ? 4 : 6,
      }),
      summary: imageDescriptor
        ? `CPU-only + ${imageDescriptor}: start with the lightest viable OBB settings to keep training stable.`
        : "CPU-only: start with the lightest viable OBB settings.",
    };
  }

  if (ultraHighResolution && denseDataset && veryHighEnd) {
    return {
      settings: normalizeObbTrainingSettings({
        modelTier: "medium",
        imgsz: 960,
        epochs: 100,
        batch: 8,
      }),
      summary: imageDescriptor
        ? `Very dense ${imageDescriptor} imagery on a strong GPU: medium at 960 is justified, but keep the starting point conservative.`
        : "Very dense high-resolution imagery on a strong GPU: medium at 960 is the minimum stronger starting point.",
    };
  }

  if (highResolution) {
    return {
      settings: normalizeObbTrainingSettings({
        modelTier: "small",
        imgsz: 960,
        epochs: denseDataset ? 100 : 75,
        batch: highEnd ? 8 : 6,
      }),
      summary: imageDescriptor
        ? `Dense ${imageDescriptor} imagery: start with a small model at 960 before considering a larger tier.`
        : "Higher-resolution imagery: start with a small model at 960 before scaling up.",
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
        ? `Moderate ${imageDescriptor} imagery: start with a small model at 640 for faster iteration.`
        : "Moderate detector dataset: start with a small model at 640 for faster iteration.",
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
      ? `Low-resolution ${imageDescriptor} imagery: start with the minimum sufficient OBB setup for faster startup and iteration.`
      : "Start with the minimum sufficient OBB setup for faster startup and iteration.",
  };
}

export function getRecommendedObbDetectionSettings(
  imageCount: number,
  imageProfile?: ImageProfileLike
): { settings: ObbDetectionSettings; summary: string } {
  const denseDataset = imageCount >= 400;
  const moderateDataset = imageCount >= 200;
  const imageInfo = describeImageProfile(imageProfile);
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

  if (denseDataset && ultraHighResolution) {
    return {
      settings: normalizeObbDetectionSettings({
        detectionPreset: "custom",
        conf: 0.25,
        nmsIou: 0.35,
        maxObjects: 36,
        imgsz: 960,
      }),
      summary: imageDescriptor
        ? `Dense ${imageDescriptor} imagery: start with Custom at 960 and a higher object cap before scaling up.`
        : "Dense high-resolution imagery: start with Custom at 960 before scaling up.",
    };
  }

  if (highResolution) {
    return {
      settings: normalizeObbDetectionSettings({
        detectionPreset: denseDataset ? "recall" : "custom",
        conf: denseDataset ? 0.25 : 0.28,
        nmsIou: 0.3,
        maxObjects: denseDataset ? 30 : 24,
        imgsz: 960,
      }),
      summary: imageDescriptor
        ? `Higher-resolution ${imageDescriptor} imagery: prefer Recall or Custom at 960 when objects are crowded or small.`
        : "Higher-resolution imagery: prefer Recall or Custom at 960 when objects are crowded or small.",
    };
  }

  if (moderateDataset || !lowResolution) {
    return {
      settings: normalizeObbDetectionSettings({
        detectionPreset: moderateDataset ? "recall" : "balanced",
        conf: moderateDataset ? 0.25 : 0.3,
        nmsIou: 0.3,
        maxObjects: moderateDataset ? 24 : 20,
        imgsz: 640,
      }),
      summary: imageDescriptor
        ? `Moderate ${imageDescriptor} imagery: start with 640 and use Recall only when you need more coverage.`
        : "Moderate imagery: start with 640 and use Recall only when you need more coverage.",
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
    summary: "Balanced works well for sparse scenes; switch to Recall or Custom when objects are missed.",
  };
}
