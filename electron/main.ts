import { app, BrowserWindow, ipcMain, dialog, nativeImage, protocol } from "electron";
import fs from "fs";
import * as path from "path";
import * as os from "os";
import { spawn, ChildProcess } from "child_process";
import { createInterface, Interface as ReadlineInterface } from "readline";

const contextMenu = require("electron-context-menu");

let mainWindow: BrowserWindow | null;
const userDataDir = app.getPath("userData");
const defaultProjectRoot = path.join(userDataDir, "training-model");
const configPath = path.join(userDataDir, "biovision-config.json");

function loadProjectRoot() {
  try {
    const raw = fs.readFileSync(configPath, "utf-8");
    const parsed = JSON.parse(raw);
    if (parsed.projectRoot && typeof parsed.projectRoot === "string") {
      return parsed.projectRoot;
    }
  } catch (e) {
    // ignore and fall back to default
  }
  return defaultProjectRoot;
}

function persistProjectRoot(root: string) {
  fs.mkdirSync(path.dirname(configPath), { recursive: true });
  fs.writeFileSync(configPath, JSON.stringify({ projectRoot: root }, null, 2));
}

let projectRoot = loadProjectRoot();
fs.mkdirSync(projectRoot, { recursive: true });
const finalizedSegmentSignatureCache = new Map<string, string>();
type SegmentQueueState = "idle" | "queued" | "running" | "saved" | "skipped" | "failed";
type SegmentQueueJob = {
  queueKey: string;
  sessionKey: string;
  speciesId: string;
  filename: string;
  sessionDir: string;
  imagePath?: string;
  acceptedBoxes: FinalizedAcceptedBox[];
  signature: string;
};
type SegmentQueueStatusEntry = {
  state: SegmentQueueState;
  signature?: string;
  updatedAt: string;
  reason?: string;
  expectedCount?: number;
  savedCount?: number;
};
const segmentSaveQueues = new Map<string, SegmentQueueJob[]>();
const segmentQueueRunningSessions = new Set<string>();
const segmentQueueStatusByImage = new Map<string, SegmentQueueStatusEntry>();
let sam2CompatibilityCache:
  | {
      ok: boolean;
      error?: string;
      checkedAtMs: number;
    }
  | null = null;

const template = [
  { label: "Minimize", click: () => mainWindow?.minimize() },
  { label: "Maximize", click: () => mainWindow?.maximize() },
  { type: "separator" },
  { label: "Copy", click: () => mainWindow?.webContents.copy() },
  { label: "Paste", click: () => mainWindow?.webContents.paste() },
  { label: "Delete", click: () => mainWindow?.webContents.delete() },
  { type: "separator" },
  // {
  //   label: 'Save Image',
  //   visible: (params: { mediaType?: string }) => params.mediaType === 'image',
  //   click: (menuItem: any, browserWindow: BrowserWindow, event: any) => {
  //     const imageURL = event.srcURL;

  //     if (imageURL) {
  //       download(browserWindow, imageURL, {
  //         saveAs: true,
  //         directory: app.getPath('downloads'),
  //       })
  //         .then(dl => {
  //           if (dl && !dl.getSavePath()) {
  //             // The user canceled the download
  //             console.log('Download canceled by the user.');
  //           } else if (dl) {
  //             console.log('Download completed:', dl.getSavePath());
  //           }
  //         })
  //         .catch(error => {
  //           console.error('Download error:', error);
  //         });
  //     }
  //   },
  // },
  { type: "separator" },
  { label: "Quit", click: () => app.quit() },
];

contextMenu({ prepend: () => template });

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      contextIsolation: true, // Ã¢â€ Â REQUIRED
      nodeIntegration: false,
      preload: path.join(__dirname, "preload.js"),
    },
  });

  // Load the Vite application URL or build output
  const VITE_DEV_SERVER_URL = process.env.VITE_DEV_SERVER_URL;

  if (VITE_DEV_SERVER_URL) {
    mainWindow.loadURL(VITE_DEV_SERVER_URL);
  } else {
    mainWindow.loadFile(path.join(__dirname, "index.html"));
  }

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

app.on("ready", () => {
  // Register a safe local-file protocol so the renderer can load session images
  // directly from disk paths without base64 encoding through IPC.
  protocol.registerFileProtocol("localfile", (request, callback) => {
    // URL format: localfile:///C:/path/to/file  (triple-slash absolute path)
    const rawPath = decodeURIComponent(request.url.replace(/^localfile:\/\//, ""));
    // On Windows the pathname begins with a leading slash before the drive letter; strip it.
    const filePath =
      process.platform === "win32" && /^\/[A-Za-z]:/.test(rawPath)
        ? rawPath.slice(1)
        : rawPath;
    callback({ path: filePath });
  });
  createWindow();
});

function getPythonPath(): string {
  // Windows: venv\Scripts\python.exe
  const venvWin = path.join(__dirname, "..", "venv", "Scripts", "python.exe");
  if (fs.existsSync(venvWin)) return venvWin;

  // Unix/macOS: venv/bin/python
  const venvUnix = path.join(__dirname, "..", "venv", "bin", "python");
  if (fs.existsSync(venvUnix)) return venvUnix;

  // Fall back to system Python
  return "python";
}

function runPython(args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    const pyPath = getPythonPath();
    const proc = spawn(pyPath, args);

    let out = "";
    let err = "";

    proc.stdout.on("data", (d) => (out += d.toString()));
    proc.stderr.on("data", (d) => (err += d.toString()));

    proc.on("close", (code) => {
      if (code !== 0) {
        return reject(new Error(err || `Python exited with code ${code}`));
      }
      resolve(out.trim());
    });
  });
}

function runPythonWithProgress(
  args: string[],
  onProgress?: (percent: number, stage: string, details?: Record<string, unknown>) => void
): Promise<string> {
  return new Promise((resolve, reject) => {
    const pyPath = getPythonPath();
    const proc = spawn(pyPath, args);
    let out = "";
    let err = "";

    proc.stdout.on("data", (d) => (out += d.toString()));
    proc.stderr.on("data", (d) => {
      const chunk = d.toString();
      err += chunk;
      if (onProgress) {
        for (const line of chunk.split("\n")) {
          const trimmed = line.trim();
          if (!trimmed) continue;

          const jsonMatch = trimmed.match(/^PROGRESS_JSON\s+(.+)$/);
          if (jsonMatch) {
            try {
              const payload = JSON.parse(jsonMatch[1]) as Record<string, unknown>;
              const rawPercent = Number(payload.percent);
              const percent = Number.isFinite(rawPercent) ? rawPercent : 0;
              const stage =
                typeof payload.message === "string" && payload.message.trim().length > 0
                  ? payload.message.trim()
                  : (typeof payload.stage === "string" ? payload.stage : "progress");
              onProgress(percent, stage, payload);
              continue;
            } catch {
              // fall through to legacy parser
            }
          }

          const m = trimmed.match(/^PROGRESS\s+(\d+)\s+(.+)$/);
          if (m) onProgress(parseInt(m[1], 10), m[2].trim());
        }
      }
    });

    proc.on("close", (code) => {
      if (code !== 0) {
        return reject(new Error(err || `Python exited with code ${code}`));
      }
      resolve(out.trim());
    });
  });
}

// ---------------------------------------------------------------------------
// Hardware capability probe Ã¢â‚¬â€ called once at app startup from React
// ---------------------------------------------------------------------------
ipcMain.handle("system:probe-hardware", async () => {
  try {
    const script = path.join(__dirname, "../backend/hardware_probe.py");
    const out = await runPython([script]);
    const parsed = JSON.parse(out.trim());
    return {
      device: parsed.device ?? "cpu",
      gpuName: parsed.gpu_name ?? null,
      ramGb: parsed.ram_gb ?? null,
    };
  } catch (err) {
    console.warn("Hardware probe failed:", err);
    // Safe fallback Ã¢â‚¬â€ treat as CPU-only
    return { device: "cpu", gpuName: null, ramGb: null };
  }
});

ipcMain.handle("ml:get-project-root", async () => {
  return { projectRoot };
});

ipcMain.handle("ml:select-project-root", async () => {
  const result = await dialog.showOpenDialog({
    properties: ["openDirectory", "createDirectory"],
  });

  if (result.canceled || result.filePaths.length === 0) {
    return { canceled: true };
  }

  const selectedPath = result.filePaths[0];
  projectRoot = selectedPath;
  fs.mkdirSync(projectRoot, { recursive: true });
  persistProjectRoot(projectRoot);

  return { canceled: false, projectRoot };
});

const DATASET_IMAGE_EXTS = /\.(jpg|jpeg|png|gif|bmp|webp|tiff|tif)$/i;
const MODEL_NAME_RE = /^[a-zA-Z0-9._-]+$/;

function sanitizeSpeciesId(speciesId?: string): string {
  return String(speciesId || "")
    .trim()
    .replace(/[^a-zA-Z0-9_-]/g, "_") || "default";
}

function getSessionsRoot(): string {
  return path.resolve(projectRoot, "sessions");
}

function getEffectiveRoot(speciesId?: string): string {
  return speciesId ? getSessionDir(speciesId) : path.resolve(projectRoot);
}

function ensureTrainingLayout(root: string): void {
  for (const sub of ["images", "labels", "xml", "models", "debug", "corrected_images"]) {
    fs.mkdirSync(path.join(root, sub), { recursive: true });
  }
}

function isFiniteNumber(value: unknown): boolean {
  return Number.isFinite(Number(value));
}

type OrientationMode = "directional" | "bilateral" | "axial" | "invariant";
type PcaLevelingMode = "off" | "on" | "auto";

type NormalizedOrientationPolicy = {
  mode: OrientationMode;
  targetOrientation?: "left" | "right";
  headCategories: string[];
  tailCategories: string[];
  bilateralPairs: [number, number][];
  pcaLevelingMode: PcaLevelingMode;
};

type ModelTrainingProfile = {
  modelName: string;
  predictorType: "dlib" | "cnn";
  orientationMode: OrientationMode;
  orientationPolicy: NormalizedOrientationPolicy;
  canonicalTrainingEnabled: boolean;
  trainedWithSam2Segments: boolean;
  canonicalMaskSource: "none" | "segments" | "rough_otsu" | "mixed" | "unknown" | "obb_geometry";
  canonicalMaskStats: {
    total: number;
    segments: number;
    roughOtsu: number;
    unknown: number;
  };
  pcaLevelingMode: PcaLevelingMode;
  targetOrientation?: "left" | "right";
  headCategories: string[];
  tailCategories: string[];
  bilateralPairs: [number, number][];
  metadataSources: string[];
};

type ModelCompatibilityIssue = {
  code: string;
  severity: "error" | "warning";
  message: string;
};

type ModelCompatibilityResult = {
  ok: boolean;
  compatible: boolean;
  blocking: boolean;
  requiresOverride: boolean;
  issues: ModelCompatibilityIssue[];
  sessionPolicy?: NormalizedOrientationPolicy;
  modelProfile?: ModelTrainingProfile;
  runtime?: {
    sam2Ready: boolean;
    sam2Required: boolean;
    requirementSource?: string;
    trainedMaskSource?: "none" | "segments" | "rough_otsu" | "mixed" | "unknown" | "obb_geometry";
    checkedAt: string;
    error?: string;
  };
  error?: string;
  obbDetectorReady?: boolean;
  obbDetectorPath?: string;
};

const COMPAT_ORIENTATION_MODES = new Set<OrientationMode>([
  "directional",
  "bilateral",
  "axial",
  "invariant",
]);
const COMPAT_PCA_MODES = new Set<PcaLevelingMode>(["off", "on", "auto"]);
const SAM2_COMPATIBILITY_CACHE_TTL_MS = 2 * 60 * 1000;

function safeReadJson(filePath: string): any | null {
  try {
    if (!fs.existsSync(filePath)) return null;
    return JSON.parse(fs.readFileSync(filePath, "utf-8"));
  } catch {
    return null;
  }
}

function summarizeCanonicalMaskUsageFromCropMetadata(
  effectiveRoot: string,
  modelName: string
): {
  source: "none" | "segments" | "rough_otsu" | "mixed" | "unknown" | "obb_geometry";
  total: number;
  segments: number;
  roughOtsu: number;
  unknown: number;
} {
  const cropMetaPath = path.join(effectiveRoot, "debug", `crop_metadata_${modelName}.json`);
  const raw = safeReadJson(cropMetaPath);
  if (!Array.isArray(raw) || raw.length === 0) {
    return {
      source: "unknown",
      total: 0,
      segments: 0,
      roughOtsu: 0,
      unknown: 0,
    };
  }

  let total = 0;
  let segments = 0;
  let roughOtsu = 0;
  let obbGeometry = 0;
  let unknown = 0;

  raw.forEach((entry) => {
    if (!entry || typeof entry !== "object") return;
    const canonicalEnabled = Boolean((entry as any).canonical_training_enabled);
    const canonicalMeta = (entry as any).canonicalization;
    if (!canonicalEnabled && (!canonicalMeta || typeof canonicalMeta !== "object")) {
      return;
    }
    total += 1;
    const rawMaskSource = String((entry as any).canonical_mask_source || "")
      .trim()
      .toLowerCase();
    const rawCanonicalizationSource = String((entry as any).canonicalization_source || "")
      .trim()
      .toLowerCase();
    const source =
      rawMaskSource === "obb"
        ? "obb_geometry"
        : rawMaskSource || rawCanonicalizationSource;
    if (source === "segments") {
      segments += 1;
    } else if (source === "rough_otsu") {
      roughOtsu += 1;
    } else if (source === "obb_geometry" || source === "none") {
      obbGeometry += 1;
    } else {
      unknown += 1;
    }
  });

  let resolved: "none" | "segments" | "rough_otsu" | "mixed" | "unknown" | "obb_geometry" = "unknown";
  if (total === 0) {
    resolved = "none";
  } else if (segments > 0 && roughOtsu > 0) {
    resolved = "mixed";
  } else if (segments > 0) {
    resolved = "segments";
  } else if (roughOtsu > 0) {
    resolved = "rough_otsu";
  } else if (obbGeometry > 0 && unknown === 0) {
    resolved = "obb_geometry";
  }

  return {
    source: resolved,
    total,
    segments,
    roughOtsu,
    unknown,
  };
}

function normalizeCategoryList(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  const out = new Set<string>();
  value.forEach((item) => {
    const normalized = String(item || "").trim().toLowerCase();
    if (normalized) out.add(normalized);
  });
  return [...out];
}

function normalizeOrientationMode(value: unknown): OrientationMode {
  const mode = String(value || "").trim().toLowerCase();
  if (COMPAT_ORIENTATION_MODES.has(mode as OrientationMode)) {
    return mode as OrientationMode;
  }
  return "invariant";
}

function normalizePcaLevelingMode(value: unknown): PcaLevelingMode {
  const mode = String(value || "").trim().toLowerCase();
  if (COMPAT_PCA_MODES.has(mode as PcaLevelingMode)) {
    return mode as PcaLevelingMode;
  }
  return "off";
}

function inferOrientationPolicyFromTemplate(landmarkTemplate: unknown): NormalizedOrientationPolicy {
  const categories = new Set<string>();
  if (Array.isArray(landmarkTemplate)) {
    landmarkTemplate.forEach((lm: any) => {
      const cat = String(lm?.category || "").trim().toLowerCase();
      if (cat) categories.add(cat);
    });
  }
  const hasHead = categories.has("head");
  const inferredTail = ["tail", "caudal-fin"].filter((cat) => categories.has(cat));
  const tailCategories = inferredTail.length > 0 ? inferredTail : ["tail", "caudal-fin"];
  if (hasHead || inferredTail.length > 0) {
    return {
      mode: "directional",
      targetOrientation: "left",
      headCategories: hasHead ? ["head"] : [],
      tailCategories,
      bilateralPairs: [],
      pcaLevelingMode: "auto",
    };
  }
  return {
    mode: "invariant",
    headCategories: [],
    tailCategories: [],
    bilateralPairs: [],
    pcaLevelingMode: "off",
  };
}

function normalizeOrientationPolicy(
  rawPolicy: unknown,
  landmarkTemplate: unknown
): NormalizedOrientationPolicy {
  const inferred = inferOrientationPolicyFromTemplate(landmarkTemplate);
  const raw = rawPolicy && typeof rawPolicy === "object" ? (rawPolicy as Record<string, unknown>) : {};

  const mode = normalizeOrientationMode(raw.mode ?? inferred.mode);
  const pcaLevelingMode = normalizePcaLevelingMode(raw.pcaLevelingMode ?? inferred.pcaLevelingMode);
  const rawTarget = String(raw.targetOrientation || inferred.targetOrientation || "left").trim().toLowerCase();
  const targetOrientation = rawTarget === "right" ? "right" : "left";

  const headFallback = inferred.headCategories.length > 0 ? inferred.headCategories : ["head"];
  const tailFallback =
    inferred.tailCategories.length > 0 ? inferred.tailCategories : ["tail", "caudal-fin"];
  const headCategories = normalizeCategoryList(raw.headCategories);
  const tailCategories = normalizeCategoryList(raw.tailCategories);

  const normalizedPairs: [number, number][] = [];
  if (Array.isArray(raw.bilateralPairs)) {
    const seen = new Set<string>();
    raw.bilateralPairs.forEach((pair: any) => {
      if (!Array.isArray(pair) || pair.length !== 2) return;
      const a = Number(pair[0]);
      const b = Number(pair[1]);
      if (!Number.isFinite(a) || !Number.isFinite(b) || a === b) return;
      const key = `${Math.min(a, b)}:${Math.max(a, b)}`;
      if (seen.has(key)) return;
      seen.add(key);
      normalizedPairs.push([Math.round(a), Math.round(b)]);
    });
  }

  const policy: NormalizedOrientationPolicy = {
    mode,
    pcaLevelingMode,
    headCategories: mode === "directional" ? (headCategories.length > 0 ? headCategories : headFallback) : [],
    tailCategories: mode === "directional" ? (tailCategories.length > 0 ? tailCategories : tailFallback) : [],
    bilateralPairs: mode === "bilateral" ? normalizedPairs : [],
  };
  if (mode === "directional") {
    policy.targetOrientation = targetOrientation;
  }
  return policy;
}

function loadSessionOrientationPolicyForCompatibility(
  speciesId: string
): { policy: NormalizedOrientationPolicy; landmarkTemplate: any[] } {
  const sessionPath = path.join(getEffectiveRoot(speciesId), "session.json");
  const sessionRaw = safeReadJson(sessionPath) || {};
  const template = Array.isArray(sessionRaw.landmarkTemplate) ? sessionRaw.landmarkTemplate : [];
  return {
    policy: normalizeOrientationPolicy(sessionRaw.orientationPolicy, template),
    landmarkTemplate: template,
  };
}

function loadModelTrainingProfileForCompatibility(
  speciesId: string,
  modelName: string,
  predictorType: "dlib" | "cnn"
): ModelTrainingProfile {
  const effectiveRoot = getEffectiveRoot(speciesId);
  const metadataSources: string[] = [];
  const fallbackSession = loadSessionOrientationPolicyForCompatibility(speciesId);

  let rawTrainingConfig: Record<string, any> = {};
  if (predictorType === "dlib") {
    const idMappingPath = path.join(effectiveRoot, "debug", `id_mapping_${modelName}.json`);
    const idMappingRaw = safeReadJson(idMappingPath);
    if (idMappingRaw && typeof idMappingRaw === "object") {
      rawTrainingConfig = (idMappingRaw.training_config || {}) as Record<string, any>;
      metadataSources.push(idMappingPath);
    }
  } else {
    const cnnTrainParamsPath = path.join(effectiveRoot, "debug", `training_params_${modelName}_cnn.json`);
    const cnnTrainParamsRaw = safeReadJson(cnnTrainParamsPath);
    if (cnnTrainParamsRaw && typeof cnnTrainParamsRaw === "object") {
      rawTrainingConfig = {
        orientation_mode: cnnTrainParamsRaw.orientation_mode,
        orientation_policy: cnnTrainParamsRaw.orientation_policy,
        canonical_training_enabled: cnnTrainParamsRaw.canonical_training_enabled,
        pca_leveling_mode: cnnTrainParamsRaw.pca_leveling_mode,
        target_orientation: cnnTrainParamsRaw.target_orientation,
      };
      metadataSources.push(cnnTrainParamsPath);
    }

    const cnnConfigPath = path.join(effectiveRoot, "models", `cnn_${modelName}_config.json`);
    const cnnConfigRaw = safeReadJson(cnnConfigPath);
    if (cnnConfigRaw && typeof cnnConfigRaw === "object") {
      metadataSources.push(cnnConfigPath);
    }
  }

  const trainingPolicy = normalizeOrientationPolicy(
    rawTrainingConfig.orientation_policy,
    fallbackSession.landmarkTemplate
  );
  const orientationMode = normalizeOrientationMode(
    rawTrainingConfig.orientation_mode ?? trainingPolicy.mode
  );
  const pcaLevelingMode = normalizePcaLevelingMode(
    rawTrainingConfig.pca_leveling_mode ?? trainingPolicy.pcaLevelingMode
  );
  const rawTarget = String(
    rawTrainingConfig.target_orientation ??
      trainingPolicy.targetOrientation ??
      fallbackSession.policy.targetOrientation ??
      "left"
  )
    .trim()
    .toLowerCase();
  const targetOrientation: "left" | "right" = rawTarget === "right" ? "right" : "left";
  const canonicalTrainingEnabled =
    typeof rawTrainingConfig.canonical_training_enabled === "boolean"
      ? rawTrainingConfig.canonical_training_enabled
      : orientationMode !== "invariant" && pcaLevelingMode !== "off";
  const maskUsage = summarizeCanonicalMaskUsageFromCropMetadata(effectiveRoot, modelName);
  const trainedWithSam2Segments =
    Boolean(canonicalTrainingEnabled) && maskUsage.segments > 0;

  return {
    modelName,
    predictorType,
    orientationMode,
    orientationPolicy: {
      ...trainingPolicy,
      mode: orientationMode,
      pcaLevelingMode,
      ...(orientationMode === "directional" ? { targetOrientation } : {}),
    },
    canonicalTrainingEnabled: Boolean(canonicalTrainingEnabled),
    trainedWithSam2Segments,
    canonicalMaskSource: maskUsage.source,
    canonicalMaskStats: {
      total: maskUsage.total,
      segments: maskUsage.segments,
      roughOtsu: maskUsage.roughOtsu,
      unknown: maskUsage.unknown,
    },
    pcaLevelingMode,
    targetOrientation: orientationMode === "directional" ? targetOrientation : undefined,
    headCategories:
      orientationMode === "directional"
        ? normalizeCategoryList(trainingPolicy.headCategories)
        : [],
    tailCategories:
      orientationMode === "directional"
        ? normalizeCategoryList(trainingPolicy.tailCategories)
        : [],
    bilateralPairs:
      orientationMode === "bilateral"
        ? (trainingPolicy.bilateralPairs || []).map((pair) => [pair[0], pair[1]] as [number, number])
        : [],
    metadataSources,
  };
}

function hasCategoryOverlap(left: string[], right: string[]): boolean {
  if (left.length === 0 || right.length === 0) return false;
  const rightSet = new Set(right);
  return left.some((item) => rightSet.has(item));
}

function formatCompatibilityErrorSummary(issues: ModelCompatibilityIssue[]): string {
  const blocking = issues
    .filter((issue) => issue.severity === "error")
    .map((issue) => issue.message);
  if (blocking.length === 0) return "No blocking compatibility issues.";
  return blocking.join(" ");
}

async function resolveSam2CompatibilityReadiness(force = false): Promise<{ ok: boolean; error?: string }> {
  const now = Date.now();
  if (
    !force &&
    sam2CompatibilityCache &&
    now - sam2CompatibilityCache.checkedAtMs <= SAM2_COMPATIBILITY_CACHE_TTL_MS
  ) {
    return { ok: sam2CompatibilityCache.ok, error: sam2CompatibilityCache.error };
  }

  const ready = await ensureSam2Ready();
  sam2CompatibilityCache = {
    ok: ready.ok,
    error: ready.error,
    checkedAtMs: now,
  };
  return { ok: ready.ok, error: ready.error };
}

async function evaluateModelCompatibility(args: {
  speciesId?: string;
  modelName: string;
  predictorType: "dlib" | "cnn";
  includeRuntime?: boolean;
}): Promise<ModelCompatibilityResult> {
  if (!args.speciesId) {
    return {
      ok: true,
      compatible: true,
      blocking: false,
      requiresOverride: false,
      issues: [],
    };
  }

  try {
    const session = loadSessionOrientationPolicyForCompatibility(args.speciesId);
    const profile = loadModelTrainingProfileForCompatibility(
      args.speciesId,
      args.modelName,
      args.predictorType
    );
    const issues: ModelCompatibilityIssue[] = [];

    if (profile.orientationMode !== session.policy.mode) {
      issues.push({
        code: "orientation_mode_mismatch",
        severity: "error",
        message: `Model trained for "${profile.orientationMode}" orientation schema but session is "${session.policy.mode}".`,
      });
    }

    if (profile.orientationMode === "directional" && session.policy.mode === "directional") {
      const modelTarget = profile.targetOrientation || "left";
      const sessionTarget = session.policy.targetOrientation || "left";
      if (modelTarget !== sessionTarget) {
        issues.push({
          code: "target_orientation_mismatch",
          severity: "error",
          message: `Model canonical target is "${modelTarget}" but session target is "${sessionTarget}".`,
        });
      }

      if (profile.headCategories.length > 0 && session.policy.headCategories.length > 0) {
        if (!hasCategoryOverlap(profile.headCategories, session.policy.headCategories)) {
          issues.push({
            code: "head_category_mismatch",
            severity: "warning",
            message: "Model and session head categories do not overlap; orientation hints may degrade.",
          });
        }
      }

      if (profile.tailCategories.length > 0 && session.policy.tailCategories.length > 0) {
        if (!hasCategoryOverlap(profile.tailCategories, session.policy.tailCategories)) {
          issues.push({
            code: "tail_category_mismatch",
            severity: "error",
            message: "Model and session tail categories do not overlap (for fish include caudal-fin as tail).",
          });
        }
      }
    }

    if (profile.orientationMode === "bilateral" && session.policy.mode === "bilateral") {
      if (profile.bilateralPairs.length > 0 && session.policy.bilateralPairs.length === 0) {
        issues.push({
          code: "bilateral_pairs_missing",
          severity: "error",
          message: "Model was trained with bilateral pair swaps but session has no bilateral pairs configured.",
        });
      }
    }

    if (profile.canonicalTrainingEnabled) {
      if (session.policy.mode === "invariant") {
        issues.push({
          code: "canonical_vs_invariant_mismatch",
          severity: "error",
          message: "Model expects canonicalized orientation flow but session is configured as invariant.",
        });
      }
      if (session.policy.pcaLevelingMode === "off") {
        issues.push({
          code: "pca_disabled_mismatch",
          severity: "error",
          message: "Model was trained with PCA/canonicalization enabled but session PCA leveling is OFF.",
        });
      }
      if (profile.canonicalMaskSource === "unknown") {
        issues.push({
          code: "canonical_mask_source_unknown",
          severity: "warning",
          message:
            "Training mask source metadata is missing/unknown; SAM2 parity requirements cannot be fully verified.",
        });
      }
    } else if (session.policy.mode !== "invariant" && session.policy.pcaLevelingMode === "on") {
      issues.push({
        code: "canonicalization_training_gap",
        severity: "warning",
        message: "Session requests forced PCA leveling, but model metadata indicates non-canonical training.",
      });
    }

    let runtime: ModelCompatibilityResult["runtime"] | undefined;
    const sam2Required = Boolean(profile.trainedWithSam2Segments);
    const shouldCheckSam2Runtime =
      Boolean(args.includeRuntime) &&
      (sam2Required || profile.canonicalTrainingEnabled);
    if (shouldCheckSam2Runtime) {
      const sam2 = await resolveSam2CompatibilityReadiness(false);
      runtime = {
        sam2Ready: sam2.ok,
        sam2Required,
        requirementSource: sam2Required ? "trained_with_sam2_segments" : "advisory_only",
        trainedMaskSource: profile.canonicalMaskSource,
        checkedAt: new Date().toISOString(),
        ...(sam2.error ? { error: sam2.error } : {}),
      };
      if (sam2Required && !sam2.ok) {
        issues.push({
          code: "sam2_training_mismatch",
          severity: "error",
          message:
            "Model was trained with SAM2-derived segment masks, but SAM2 is unavailable at inference. This is blocked by default to preserve train/inference parity. Override only if you accept reduced orientation accuracy.",
        });
      } else if (!sam2.ok && profile.canonicalTrainingEnabled) {
        issues.push({
          code: "sam2_unavailable_masks_disabled",
          severity: "warning",
          message:
            "SAM2 is unavailable; mask refinement will be disabled at inference.",
        });
      }
    }

    // Check for session OBB detector
    const effectiveRoot = getEffectiveRoot(args.speciesId);
    const obbDetectorPath = path.join(effectiveRoot, "models", "session_obb_detector.pt");
    const obbDetectorExists = fs.existsSync(obbDetectorPath);

    const blocking = issues.some((issue) => issue.severity === "error");
    return {
      ok: true,
      compatible: !blocking && issues.length === 0,
      blocking,
      requiresOverride: blocking,
      issues,
      sessionPolicy: session.policy,
      modelProfile: profile,
      runtime,
      obbDetectorReady: obbDetectorExists,
      obbDetectorPath: obbDetectorExists ? obbDetectorPath : undefined,
    };
  } catch (error: any) {
    return {
      ok: false,
      compatible: false,
      blocking: true,
      requiresOverride: true,
      issues: [
        {
          code: "compatibility_check_failed",
          severity: "error",
          message: error?.message || "Compatibility check failed.",
        },
      ],
      error: error?.message || "Compatibility check failed.",
    };
  }
}

type FinalizedAcceptedLandmark = {
  id: number;
  x: number;
  y: number;
  isSkipped?: boolean;
};

type FinalizedAcceptedBox = {
  left: number;
  top: number;
  width: number;
  height: number;
  orientation_override?: "left" | "right" | "uncertain";
  obbCorners?: [number, number][];
  angle?: number;
  class_id?: number;
  landmarks?: FinalizedAcceptedLandmark[];
};

function normalizeFinalizedAcceptedBoxes(
  rawBoxes: Array<{
    left: number;
    top: number;
    width: number;
    height: number;
    orientation_override?: "left" | "right" | "uncertain";
    obbCorners?: [number, number][];
    angle?: number;
    class_id?: number;
    landmarks?: Array<{ id: number; x: number; y: number; isSkipped?: boolean }>;
  }>
): FinalizedAcceptedBox[] {
  const accepted: FinalizedAcceptedBox[] = [];
  (rawBoxes || []).forEach((raw, idx) => {
    const left = Math.round(Number(raw?.left));
    const top = Math.round(Number(raw?.top));
    const width = Math.round(Number(raw?.width));
    const height = Math.round(Number(raw?.height));
    if (!isFiniteNumber(left) || !isFiniteNumber(top) || width <= 0 || height <= 0) {
      return;
    }
    const box: FinalizedAcceptedBox = { left, top, width, height };
    if (
      raw?.orientation_override === "left" ||
      raw?.orientation_override === "right" ||
      raw?.orientation_override === "uncertain"
    ) {
      box.orientation_override = raw.orientation_override;
    }
    if (Array.isArray(raw?.obbCorners) && raw.obbCorners.length === 4) {
      box.obbCorners = raw.obbCorners.map((point: any) => [
        Number(point?.[0]) || 0,
        Number(point?.[1]) || 0,
      ]) as [number, number][];
    }
    if (isFiniteNumber(raw?.angle)) {
      box.angle = Number(raw.angle);
    }
    if (isFiniteNumber(raw?.class_id)) {
      box.class_id = Math.round(Number(raw.class_id));
    }
    if (Array.isArray(raw?.landmarks) && raw.landmarks.length > 0) {
      try {
        const normalizedLandmarks = normalizeLandmarks(
          raw.landmarks as any[],
          `acceptedBoxes[${idx}]`
        );
        if (normalizedLandmarks.length > 0) {
          box.landmarks = normalizedLandmarks.map((lm) => ({
            id: Number(lm.id),
            x: Number(lm.x),
            y: Number(lm.y),
            ...(lm.isSkipped ? { isSkipped: true } : {}),
          }));
        }
      } catch (_) {
        // Keep box geometry even if a subset of landmarks are malformed.
      }
    }
    accepted.push(box);
  });

  accepted.sort(
    (a, b) =>
      a.left - b.left ||
      a.top - b.top ||
      a.width - b.width ||
      a.height - b.height
  );
  return accepted;
}

function buildAcceptedBoxesSignature(boxes: Array<{ left: number; top: number; width: number; height: number }>): string {
  const reduced = (boxes || [])
    .map((b) => ({
      left: Math.round(Number(b.left) || 0),
      top: Math.round(Number(b.top) || 0),
      width: Math.round(Number(b.width) || 0),
      height: Math.round(Number(b.height) || 0),
    }))
    .filter((b) => b.width > 0 && b.height > 0)
    .sort(
      (a, b) =>
        a.left - b.left ||
        a.top - b.top ||
        a.width - b.width ||
        a.height - b.height
    );
  return JSON.stringify(reduced);
}

function buildSegmentQueueImageKey(speciesId: string, filename: string): string {
  return `${String(speciesId || "").trim()}::${path.basename(String(filename || "").trim()).toLowerCase()}`;
}

function setSegmentQueueStatus(
  speciesId: string,
  filename: string,
  state: SegmentQueueState,
  signature?: string,
  reason?: string,
  counts?: { expectedCount?: number; savedCount?: number }
): void {
  const key = buildSegmentQueueImageKey(speciesId, filename);
  const entry: SegmentQueueStatusEntry = {
    state,
    updatedAt: new Date().toISOString(),
    ...(signature ? { signature } : {}),
    ...(reason ? { reason } : {}),
    ...(counts?.expectedCount != null ? { expectedCount: counts.expectedCount } : {}),
    ...(counts?.savedCount != null ? { savedCount: counts.savedCount } : {}),
  };
  segmentQueueStatusByImage.set(key, entry);
  mainWindow?.webContents.send("session:segment-save-status", {
    speciesId,
    filename,
    ...entry,
  });
}

function getSegmentQueueStatus(speciesId: string, filename: string): SegmentQueueStatusEntry {
  return (
    segmentQueueStatusByImage.get(buildSegmentQueueImageKey(speciesId, filename)) ?? {
      state: "idle",
      updatedAt: new Date().toISOString(),
    }
  );
}

function getImageDimensions(imagePath: string): { width: number; height: number } | null {
  try {
    const image = nativeImage.createFromPath(imagePath);
    const size = image.getSize();
    if (size.width > 0 && size.height > 0) {
      return { width: size.width, height: size.height };
    }
  } catch (_) {
    // ignore and fallback
  }
  return null;
}

function normalizeLandmarks(
  rawLandmarks: any[],
  context: string
): Array<{ id: number; x: number; y: number; isSkipped?: boolean }> {
  const normalized: Array<{ id: number; x: number; y: number; isSkipped?: boolean }> = [];
  rawLandmarks.forEach((landmark: any, landmarkIndex: number) => {
    const skipped = Boolean(landmark?.isSkipped);
    const rawId = landmark?.id;
    const id = isFiniteNumber(rawId) ? Number(rawId) : landmarkIndex;
    const x = Number(landmark?.x);
    const y = Number(landmark?.y);

    if (!isFiniteNumber(id)) {
      throw new Error(`${context} landmark ${landmarkIndex} has invalid id.`);
    }
    if (!skipped && (!isFiniteNumber(x) || !isFiniteNumber(y))) {
      throw new Error(`${context} landmark ${landmarkIndex} has invalid x/y.`);
    }

    normalized.push({
      id,
      x: skipped ? -1 : x,
      y: skipped ? -1 : y,
      ...(skipped ? { isSkipped: true } : {}),
    });
  });
  return normalized;
}

type ImportedLandmark = { id: number; x: number; y: number; isSkipped?: boolean };
type ImportedObbCorners = [number, number][];

type ImportedRawBox = {
  left?: number;
  top?: number;
  width?: number;
  height?: number;
  obbCorners?: ImportedObbCorners;
  angle?: number;
  class_id?: number;
  landmarks?: any[];
};

type ImportedNormalizedBox = {
  left: number;
  top: number;
  width: number;
  height: number;
  landmarks: ImportedLandmark[];
  maskOutline?: [number, number][];
  obbCorners?: ImportedObbCorners;
  angle?: number;
  class_id?: number;
  geometryOrigin?: "source" | "derived";
};

type ImportGeometryConfig = {
  axisMode: "auto" | "manual_anchors";
  anchorLandmarkIds?: {
    anteriorId: number;
    posteriorId: number;
  };
  paddingMode: "tight" | "asymmetric";
  paddingProfile?: {
    forwardPct: number;
    backwardPct: number;
    topPct: number;
    bottomPct: number;
  };
};

type NormalizedImportGeometryConfig = {
  axisMode: "auto" | "manual_anchors";
  anchorLandmarkIds?: {
    anteriorId: number;
    posteriorId: number;
  };
  paddingMode: "tight" | "asymmetric";
  paddingProfile: {
    forward: number;
    backward: number;
    top: number;
    bottom: number;
  };
};

type ImportGeometrySummary = {
  sourceObbPreserved: number;
  manualAnchorDerived: number;
  autoDerived: number;
  fallbackBoxes: number;
  usedAsymmetricPadding: boolean;
};

function getValidImportedLandmarks(landmarks: ImportedLandmark[]): Array<{ id: number; x: number; y: number }> {
  return landmarks
    .filter((lm) => !lm.isSkipped && isFiniteNumber(lm.x) && isFiniteNumber(lm.y))
    .map((lm) => ({ id: Number(lm.id), x: Number(lm.x), y: Number(lm.y) }));
}

function buildImportedObbCorners(
  cx: number,
  cy: number,
  width: number,
  height: number,
  angleDeg: number
): ImportedObbCorners {
  const r = angleDeg * (Math.PI / 180);
  const cos = Math.cos(r);
  const sin = Math.sin(r);
  const hw = width / 2;
  const hh = height / 2;
  return [
    [cx + cos * (-hw) - sin * (-hh), cy + sin * (-hw) + cos * (-hh)],
    [cx + cos * hw - sin * (-hh), cy + sin * hw + cos * (-hh)],
    [cx + cos * hw - sin * hh, cy + sin * hw + cos * hh],
    [cx + cos * (-hw) - sin * hh, cy + sin * (-hw) + cos * hh],
  ] as ImportedObbCorners;
}

function clampImportedObbCorners(
  corners: ImportedObbCorners,
  imageDims: { width: number; height: number } | null
): ImportedObbCorners {
  if (!imageDims) return corners;
  return corners.map(([x, y]) => ([
    Math.max(0, Math.min(Number(imageDims.width) - 1, Number(x))),
    Math.max(0, Math.min(Number(imageDims.height) - 1, Number(y))),
  ])) as ImportedObbCorners;
}

function normalizeImportedObbCorners(
  rawCorners: unknown,
  imageDims: { width: number; height: number } | null
): ImportedObbCorners | undefined {
  if (!Array.isArray(rawCorners) || rawCorners.length !== 4) return undefined;
  const corners: ImportedObbCorners = [];
  rawCorners.forEach((point) => {
    if (!Array.isArray(point) || point.length < 2) return;
    const x = Number(point[0]);
    const y = Number(point[1]);
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    corners.push([x, y]);
  });
  if (corners.length !== 4) return undefined;
  return clampImportedObbCorners(corners, imageDims);
}

function importedCornersToAabb(
  corners: ImportedObbCorners,
  imageDims: { width: number; height: number } | null
): { left: number; top: number; width: number; height: number } {
  const xs = corners.map((c) => Number(c[0]));
  const ys = corners.map((c) => Number(c[1]));
  let left = Math.min(...xs);
  let top = Math.min(...ys);
  let right = Math.max(...xs);
  let bottom = Math.max(...ys);
  if (imageDims) {
    left = Math.max(0, Math.min(left, imageDims.width - 1));
    top = Math.max(0, Math.min(top, imageDims.height - 1));
    right = Math.max(left + 1, Math.min(right, imageDims.width));
    bottom = Math.max(top + 1, Math.min(bottom, imageDims.height));
  }
  const width = right - left;
  const height = bottom - top;
  if (!(width > 1 && height > 1)) {
    throw new Error("derived bounding box is too small/invalid.");
  }
  return { left, top, width, height };
}

function getImportedObbDimensions(corners: ImportedObbCorners): { width: number; height: number } {
  const edgeWidthA = Math.hypot(
    Number(corners[1][0]) - Number(corners[0][0]),
    Number(corners[1][1]) - Number(corners[0][1])
  );
  const edgeWidthB = Math.hypot(
    Number(corners[2][0]) - Number(corners[3][0]),
    Number(corners[2][1]) - Number(corners[3][1])
  );
  const edgeHeightA = Math.hypot(
    Number(corners[2][0]) - Number(corners[1][0]),
    Number(corners[2][1]) - Number(corners[1][1])
  );
  const edgeHeightB = Math.hypot(
    Number(corners[3][0]) - Number(corners[0][0]),
    Number(corners[3][1]) - Number(corners[0][1])
  );
  return {
    width: Math.max(2, (edgeWidthA + edgeWidthB) / 2),
    height: Math.max(2, (edgeHeightA + edgeHeightB) / 2),
  };
}

function buildImportSam2PromptBox(
  box: ImportedNormalizedBox,
  imageDims: { width: number; height: number } | null,
  geometryConfig?: ImportGeometryConfig
): [number, number, number, number] {
  if (box.geometryOrigin !== "source") {
    return [
      Math.round(box.left),
      Math.round(box.top),
      Math.round(box.left + box.width),
      Math.round(box.top + box.height),
    ];
  }

  const resolvedGeometry = normalizeImportGeometryConfig(geometryConfig);
  if (box.obbCorners && box.obbCorners.length === 4 && isFiniteNumber(box.angle)) {
    const { width, height } = getImportedObbDimensions(box.obbCorners);
    const padForward = width * resolvedGeometry.paddingProfile.forward;
    const padBackward = width * resolvedGeometry.paddingProfile.backward;
    const padTop = height * resolvedGeometry.paddingProfile.top;
    const padBottom = height * resolvedGeometry.paddingProfile.bottom;
    const centerX = box.obbCorners.reduce((sum, point) => sum + Number(point[0]), 0) / 4;
    const centerY = box.obbCorners.reduce((sum, point) => sum + Number(point[1]), 0) / 4;
    const angle = Number(box.angle);
    const radians = angle * (Math.PI / 180);
    const ux = Math.cos(radians);
    const uy = Math.sin(radians);
    const vx = -Math.sin(radians);
    const vy = Math.cos(radians);
    const expandedCenterX = centerX + ux * ((padBackward - padForward) / 2) + vx * ((padBottom - padTop) / 2);
    const expandedCenterY = centerY + uy * ((padBackward - padForward) / 2) + vy * ((padBottom - padTop) / 2);
    const expandedCorners = clampImportedObbCorners(
      buildImportedObbCorners(
        expandedCenterX,
        expandedCenterY,
        width + padForward + padBackward,
        height + padTop + padBottom,
        angle
      ),
      imageDims
    );
    const expandedAabb = importedCornersToAabb(expandedCorners, imageDims);
    return [
      Math.round(expandedAabb.left),
      Math.round(expandedAabb.top),
      Math.round(expandedAabb.left + expandedAabb.width),
      Math.round(expandedAabb.top + expandedAabb.height),
    ];
  }

  const padLeft = box.width * resolvedGeometry.paddingProfile.forward;
  const padRight = box.width * resolvedGeometry.paddingProfile.backward;
  const padTop = box.height * resolvedGeometry.paddingProfile.top;
  const padBottom = box.height * resolvedGeometry.paddingProfile.bottom;
  const left = Math.max(0, box.left - padLeft);
  const top = Math.max(0, box.top - padTop);
  const right = imageDims ? Math.min(imageDims.width, box.left + box.width + padRight) : box.left + box.width + padRight;
  const bottom = imageDims ? Math.min(imageDims.height, box.top + box.height + padBottom) : box.top + box.height + padBottom;
  return [Math.round(left), Math.round(top), Math.round(right), Math.round(bottom)];
}

function resolveTemplateLandmarkId(landmarkTemplate: any[], categories: string[]): number | null {
  if (!Array.isArray(landmarkTemplate) || !Array.isArray(categories) || categories.length === 0) {
    return null;
  }
  const normalizedCategories = new Set(
    categories.map((value) => String(value || "").trim().toLowerCase()).filter(Boolean)
  );
  const matches = landmarkTemplate
    .map((lm: any) => ({
      index: Number(lm?.index),
      category: String(lm?.category || "").trim().toLowerCase(),
    }))
    .filter((lm) => Number.isFinite(lm.index) && normalizedCategories.has(lm.category))
    .map((lm) => Math.round(lm.index));
  return matches.length > 0 ? Math.min(...matches) : null;
}

function resolveHeadTailLandmarkIdsForImport(
  orientationPolicy: NormalizedOrientationPolicy,
  landmarkTemplate: any[]
): { headId: number | null; tailId: number | null } {
  return {
    headId: resolveTemplateLandmarkId(landmarkTemplate, orientationPolicy.headCategories),
    tailId: resolveTemplateLandmarkId(landmarkTemplate, orientationPolicy.tailCategories),
  };
}

function findImportedLandmarkById(landmarks: ImportedLandmark[], id: number | null): ImportedLandmark | null {
  if (id == null) return null;
  return landmarks.find((lm) => !lm.isSkipped && Number(lm.id) === Number(id)) ?? null;
}

function normalizeImportedAngle360(angleDeg: number): number {
  let angle = Number(angleDeg) % 360;
  if (angle < 0) angle += 360;
  return angle;
}

function normalizeImportedAngleSigned(angleDeg: number): number {
  let angle = normalizeImportedAngle360(angleDeg);
  if (angle > 180) angle -= 360;
  if (angle <= -180) angle += 360;
  if (Math.abs(angle + 180) <= 1e-6) return 180;
  return angle;
}

function importedAngularDistanceDeg(a: number, b: number): number {
  return Math.abs((((Number(a) - Number(b)) + 180) % 360 + 360) % 360 - 180);
}

function snapDerivedImportedAngle(angleDeg: number, mode: OrientationMode, toleranceDeg = 5.0): number {
  if (mode === "invariant") return 0;
  const angle360 = normalizeImportedAngle360(angleDeg);
  const targets =
    mode === "directional"
      ? [0, 180]
      : mode === "bilateral"
      ? [90, 270]
      : [0, 90, 180, 270];
  let bestTarget = angle360;
  let bestDistance = Number.POSITIVE_INFINITY;
  targets.forEach((target) => {
    const distance = importedAngularDistanceDeg(angle360, target);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestTarget = target;
    }
  });
  if (bestDistance <= toleranceDeg) {
    return normalizeImportedAngleSigned(bestTarget);
  }
  return normalizeImportedAngleSigned(angle360);
}

function getImportedQuartileBucketSize(count: number): number {
  return Math.max(1, Math.ceil(Number(count) * 0.25));
}

function getImportedCentroid(points: Array<{ x: number; y: number }>): { x: number; y: number } {
  const count = Math.max(1, points.length);
  return {
    x: points.reduce((sum, point) => sum + Number(point.x), 0) / count,
    y: points.reduce((sum, point) => sum + Number(point.y), 0) / count,
  };
}

function deriveImportedBiCentroidAxis(
  valid: Array<{ id: number; x: number; y: number }>,
  axis: "horizontal" | "vertical"
): { ux: number; uy: number; separation: number } {
  if (valid.length < 2) {
    throw new Error("need at least two landmarks for bi-centroid axis");
  }
  const key = axis === "horizontal" ? "x" : "y";
  const sorted = [...valid].sort((a, b) => Number(a[key]) - Number(b[key]));
  const bucketSize = getImportedQuartileBucketSize(sorted.length);
  const start = sorted.slice(0, bucketSize);
  const end = sorted.slice(sorted.length - bucketSize);
  const startCentroid = getImportedCentroid(start);
  const endCentroid = getImportedCentroid(end);
  const dx = Number(endCentroid.x) - Number(startCentroid.x);
  const dy = Number(endCentroid.y) - Number(startCentroid.y);
  const separation = Math.hypot(dx, dy);
  if (separation <= 1e-6) {
    throw new Error("bi-centroid buckets collapse to the same centroid");
  }
  return { ux: dx / separation, uy: dy / separation, separation };
}

function normalizeImportGeometryConfig(raw?: ImportGeometryConfig | null): NormalizedImportGeometryConfig {
  const axisMode = raw?.axisMode === "manual_anchors" ? "manual_anchors" : "auto";
  const paddingMode = raw?.paddingMode === "asymmetric" ? "asymmetric" : "tight";
  const anteriorId = raw?.anchorLandmarkIds?.anteriorId;
  const posteriorId = raw?.anchorLandmarkIds?.posteriorId;
  const anchorLandmarkIds =
    axisMode === "manual_anchors" &&
    isFiniteNumber(anteriorId) &&
    isFiniteNumber(posteriorId) &&
    Number(anteriorId) !== Number(posteriorId)
      ? {
          anteriorId: Math.round(Number(anteriorId)),
          posteriorId: Math.round(Number(posteriorId)),
        }
      : undefined;

  const pct = (value: unknown) => Math.max(0, Number(value) || 0) / 100;
  const paddingProfile = {
    forward: paddingMode === "asymmetric" ? pct(raw?.paddingProfile?.forwardPct) : 0.1,
    backward: paddingMode === "asymmetric" ? pct(raw?.paddingProfile?.backwardPct) : 0.1,
    top: paddingMode === "asymmetric" ? pct(raw?.paddingProfile?.topPct) : 0.1,
    bottom: paddingMode === "asymmetric" ? pct(raw?.paddingProfile?.bottomPct) : 0.1,
  };

  return {
    axisMode,
    ...(anchorLandmarkIds ? { anchorLandmarkIds } : {}),
    paddingMode,
    paddingProfile,
  };
}

function getImportedPerpendicularAxis(ux: number, uy: number): { vx: number; vy: number } {
  const candidateA = { vx: -uy, vy: ux };
  const candidateB = { vx: uy, vy: -ux };
  return candidateA.vy <= candidateB.vy ? candidateA : candidateB;
}

function createImportGeometrySummary(): ImportGeometrySummary {
  return {
    sourceObbPreserved: 0,
    manualAnchorDerived: 0,
    autoDerived: 0,
    fallbackBoxes: 0,
    usedAsymmetricPadding: false,
  };
}

function summarizeImportGeometry(summary: ImportGeometrySummary, unmatchedCount = 0): string[] {
  const warnings: string[] = [];
  if (summary.manualAnchorDerived > 0) {
    warnings.push(`Derived OBBs from manual anchors for ${summary.manualAnchorDerived} box${summary.manualAnchorDerived === 1 ? "" : "es"}.`);
  } else if (summary.autoDerived > 0) {
    warnings.push(`Derived OBBs automatically for ${summary.autoDerived} box${summary.autoDerived === 1 ? "" : "es"}.`);
  }
  if (summary.usedAsymmetricPadding) {
    warnings.push("Applied custom specimen padding during import.");
  }
  if (summary.fallbackBoxes > 0) {
    warnings.push(`${summary.fallbackBoxes} box${summary.fallbackBoxes === 1 ? "" : "es"} used a simple fallback because the selected anchors or landmark geometry were insufficient.`);
  }
  if (unmatchedCount > 0) {
    warnings.push(`${unmatchedCount} image${unmatchedCount === 1 ? "" : "s"} had no matching annotation.`);
  }
  return warnings;
}

function deriveImportedClassId(
  landmarks: ImportedLandmark[],
  orientationPolicy: NormalizedOrientationPolicy,
  landmarkTemplate: any[]
): number | undefined {
  if (orientationPolicy.mode === "invariant") return 0;
  const { headId, tailId } = resolveHeadTailLandmarkIdsForImport(orientationPolicy, landmarkTemplate);
  const head = findImportedLandmarkById(landmarks, headId);
  const tail = findImportedLandmarkById(landmarks, tailId);
  if (!head || !tail) return undefined;
  if (orientationPolicy.mode === "axial") {
    return Number(head.y) < Number(tail.y) ? 0 : 1;
  }
  return Number(head.x) < Number(tail.x) ? 0 : 1;
}

function deriveImportedObbFromLandmarks(
  landmarks: ImportedLandmark[],
  imageDims: { width: number; height: number } | null,
  orientationPolicy: NormalizedOrientationPolicy,
  landmarkTemplate: any[],
  geometryConfig?: ImportGeometryConfig
): {
  left: number;
  top: number;
  width: number;
  height: number;
  obbCorners: ImportedObbCorners;
  angle: number;
  class_id?: number;
  derivation:
    | "head_tail"
    | "directional_bi_centroid"
    | "bilateral_bi_centroid"
    | "axial_axis_snapped"
    | "invariant_axis_aligned"
    | "fallback";
} {
  const resolvedGeometry = normalizeImportGeometryConfig(geometryConfig);
  const valid = getValidImportedLandmarks(landmarks);
  if (valid.length === 0) {
    throw new Error("cannot derive OBB from landmarks: no valid non-skipped landmarks.");
  }

  const uniquePoints = new Set(valid.map((lm) => `${Math.round(lm.x * 1000)}:${Math.round(lm.y * 1000)}`));
  if (uniquePoints.size < 2) {
    const aabb = deriveBoxFromLandmarks(landmarks, imageDims);
    const cx = aabb.left + aabb.width / 2;
    const cy = aabb.top + aabb.height / 2;
    const obbCorners = clampImportedObbCorners(
      buildImportedObbCorners(cx, cy, aabb.width, aabb.height, 0),
      imageDims
    );
    return {
      ...importedCornersToAabb(obbCorners, imageDims),
      obbCorners,
      angle: 0,
      class_id: deriveImportedClassId(landmarks, orientationPolicy, landmarkTemplate),
      derivation: "fallback",
    };
  }

  const manualAnteriorId =
    resolvedGeometry.axisMode === "manual_anchors" ? resolvedGeometry.anchorLandmarkIds?.anteriorId ?? null : null;
  const manualPosteriorId =
    resolvedGeometry.axisMode === "manual_anchors" ? resolvedGeometry.anchorLandmarkIds?.posteriorId ?? null : null;
  const { headId, tailId } =
    resolvedGeometry.axisMode === "manual_anchors"
      ? { headId: manualAnteriorId, tailId: manualPosteriorId }
      : resolveHeadTailLandmarkIdsForImport(orientationPolicy, landmarkTemplate);
  const head = findImportedLandmarkById(landmarks, headId);
  const tail = findImportedLandmarkById(landmarks, tailId);
  let ux = 1;
  let uy = 0;
  let derivation:
    | "head_tail"
    | "directional_bi_centroid"
    | "bilateral_bi_centroid"
    | "axial_axis_snapped"
    | "invariant_axis_aligned"
    | "fallback" = "fallback";

  if (orientationPolicy.mode === "invariant") {
    const aabb = deriveBoxFromLandmarks(landmarks, imageDims);
    const cx = aabb.left + aabb.width / 2;
    const cy = aabb.top + aabb.height / 2;
    const obbCorners = clampImportedObbCorners(
      buildImportedObbCorners(cx, cy, aabb.width, aabb.height, 0),
      imageDims
    );
    return {
      ...importedCornersToAabb(obbCorners, imageDims),
      obbCorners,
      angle: 0,
      class_id: deriveImportedClassId(landmarks, orientationPolicy, landmarkTemplate),
      derivation: "invariant_axis_aligned",
    };
  }

  if (head && tail) {
    const dx = Number(tail.x) - Number(head.x);
    const dy = Number(tail.y) - Number(head.y);
    const norm = Math.hypot(dx, dy);
    if (norm > 1e-6) {
      ux = dx / norm;
      uy = dy / norm;
      derivation = "head_tail";
    }
  }

  if (
    resolvedGeometry.axisMode === "manual_anchors" &&
    derivation !== "head_tail"
  ) {
    const aabb = deriveBoxFromLandmarks(landmarks, imageDims);
    const cx = aabb.left + aabb.width / 2;
    const cy = aabb.top + aabb.height / 2;
    const obbCorners = clampImportedObbCorners(
      buildImportedObbCorners(cx, cy, aabb.width, aabb.height, 0),
      imageDims
    );
    return {
      ...importedCornersToAabb(obbCorners, imageDims),
      obbCorners,
      angle: 0,
      class_id: deriveImportedClassId(landmarks, orientationPolicy, landmarkTemplate),
      derivation: "fallback",
    };
  }

  if (derivation !== "head_tail") {
    if (orientationPolicy.mode === "directional") {
      try {
        const axis = deriveImportedBiCentroidAxis(valid, "horizontal");
        ux = axis.ux;
        uy = axis.uy;
        derivation = "directional_bi_centroid";
      } catch {
        derivation = "fallback";
      }
    } else if (orientationPolicy.mode === "bilateral") {
      try {
        const axis = deriveImportedBiCentroidAxis(valid, "vertical");
        ux = axis.ux;
        uy = axis.uy;
        derivation = "bilateral_bi_centroid";
      } catch {
        derivation = "fallback";
      }
    } else {
      const candidates: Array<{ ux: number; uy: number; separation: number }> = [];
      try {
        candidates.push(deriveImportedBiCentroidAxis(valid, "horizontal"));
      } catch {}
      try {
        candidates.push(deriveImportedBiCentroidAxis(valid, "vertical"));
      } catch {}
      if (candidates.length > 0) {
        const axis = candidates.reduce((best, current) =>
          current.separation > best.separation ? current : best
        );
        ux = axis.ux;
        uy = axis.uy;
        derivation = "axial_axis_snapped";
      } else {
        derivation = "fallback";
      }
    }
  }

  const { vx, vy } = getImportedPerpendicularAxis(ux, uy);
  const uValues = valid.map((lm) => lm.x * ux + lm.y * uy);
  const vValues = valid.map((lm) => lm.x * vx + lm.y * vy);
  const minU = Math.min(...uValues);
  const maxU = Math.max(...uValues);
  const minV = Math.min(...vValues);
  const maxV = Math.max(...vValues);
  const rawWidth = Math.max(2, maxU - minU);
  const rawHeight = Math.max(2, maxV - minV);
  const padForward = Math.max(4, rawWidth * resolvedGeometry.paddingProfile.forward);
  const padBackward = Math.max(4, rawWidth * resolvedGeometry.paddingProfile.backward);
  const padTop = Math.max(4, rawHeight * resolvedGeometry.paddingProfile.top);
  const padBottom = Math.max(4, rawHeight * resolvedGeometry.paddingProfile.bottom);
  const minUWithPad = minU - padForward;
  const maxUWithPad = maxU + padBackward;
  const minVWithPad = minV - padTop;
  const maxVWithPad = maxV + padBottom;
  const centerU = (minUWithPad + maxUWithPad) / 2;
  const centerV = (minVWithPad + maxVWithPad) / 2;
  const centerX = centerU * ux + centerV * vx;
  const centerY = centerU * uy + centerV * vy;
  const width = maxUWithPad - minUWithPad;
  const height = maxVWithPad - minVWithPad;
  const angle = snapDerivedImportedAngle(Math.atan2(uy, ux) * 180 / Math.PI, orientationPolicy.mode);
  const obbCorners = clampImportedObbCorners(
    buildImportedObbCorners(centerX, centerY, width, height, angle),
    imageDims
  );
  return {
    ...importedCornersToAabb(obbCorners, imageDims),
    obbCorners,
    angle,
    class_id: deriveImportedClassId(landmarks, orientationPolicy, landmarkTemplate),
    derivation,
  };
}

function normalizeImportedBoxGeometry(
  rawBox: ImportedRawBox,
  normalizedLandmarks: ImportedLandmark[],
  imageDims: { width: number; height: number } | null,
  context: string,
  orientationPolicy: NormalizedOrientationPolicy,
  landmarkTemplate: any[],
  warnings: string[],
  geometryConfig?: ImportGeometryConfig,
  summary?: ImportGeometrySummary
): ImportedNormalizedBox {
  const sourceObbCorners = normalizeImportedObbCorners(rawBox?.obbCorners, imageDims);
  if (sourceObbCorners) {
    if (summary) {
      summary.sourceObbPreserved += 1;
    }
    const aabb = importedCornersToAabb(sourceObbCorners, imageDims);
    const angle = isFiniteNumber(rawBox?.angle)
      ? Number(rawBox?.angle)
      : Math.atan2(
          Number(sourceObbCorners[1][1]) - Number(sourceObbCorners[0][1]),
          Number(sourceObbCorners[1][0]) - Number(sourceObbCorners[0][0])
        ) * 180 / Math.PI;
    const derivedClassId = deriveImportedClassId(normalizedLandmarks, orientationPolicy, landmarkTemplate);
    const classId = isFiniteNumber(rawBox?.class_id)
      ? Math.round(Number(rawBox?.class_id))
      : derivedClassId;
    return {
      ...aabb,
      landmarks: normalizedLandmarks,
      obbCorners: sourceObbCorners,
      angle,
      geometryOrigin: "source",
      ...(classId != null ? { class_id: classId } : {}),
    };
  }

  const derived = deriveImportedObbFromLandmarks(
    normalizedLandmarks,
    imageDims,
    orientationPolicy,
    landmarkTemplate,
    geometryConfig
  );
  const resolvedGeometry = normalizeImportGeometryConfig(geometryConfig);
  if (summary) {
    if (resolvedGeometry.paddingMode === "asymmetric") {
      summary.usedAsymmetricPadding = true;
    }
    if (derived.derivation === "fallback") {
      summary.fallbackBoxes += 1;
    } else if (resolvedGeometry.axisMode === "manual_anchors") {
      summary.manualAnchorDerived += 1;
    } else {
      summary.autoDerived += 1;
    }
  }
  warnings.push(
    derived.derivation === "fallback"
      ? `${context}: ${
          resolvedGeometry.axisMode === "manual_anchors"
            ? "manual anchor landmarks were missing or degenerate; saved a fallback OBB."
            : "landmarks were insufficient for stable orientation; saved a fallback OBB."
        }`
      : `${context}: derived OBB from ${
          resolvedGeometry.axisMode === "manual_anchors" ? "manual anchors" : "landmarks"
        }${resolvedGeometry.paddingMode === "asymmetric" ? " with custom padding" : ""}.`
  );
  return {
    left: derived.left,
    top: derived.top,
    width: derived.width,
    height: derived.height,
    landmarks: normalizedLandmarks,
    obbCorners: derived.obbCorners,
    angle: derived.angle,
    geometryOrigin: "derived",
    ...(derived.class_id != null ? { class_id: derived.class_id } : {}),
  };
}

function deriveBoxFromLandmarks(
  landmarks: Array<{ id: number; x: number; y: number; isSkipped?: boolean }>,
  imageDims: { width: number; height: number } | null
): { left: number; top: number; width: number; height: number } {
  const valid = landmarks.filter((lm) => !lm.isSkipped && isFiniteNumber(lm.x) && isFiniteNumber(lm.y));
  if (valid.length === 0) {
    throw new Error("cannot derive box from landmarks: no valid non-skipped landmarks.");
  }

  let minX = Math.min(...valid.map((lm) => lm.x));
  let minY = Math.min(...valid.map((lm) => lm.y));
  let maxX = Math.max(...valid.map((lm) => lm.x));
  let maxY = Math.max(...valid.map((lm) => lm.y));

  // Add a small margin around landmark extent.
  const rawW = Math.max(2, maxX - minX);
  const rawH = Math.max(2, maxY - minY);
  const padX = Math.max(4, rawW * 0.1);
  const padY = Math.max(4, rawH * 0.1);
  minX -= padX;
  minY -= padY;
  maxX += padX;
  maxY += padY;

  if (imageDims) {
    minX = Math.max(0, Math.min(minX, imageDims.width - 1));
    minY = Math.max(0, Math.min(minY, imageDims.height - 1));
    maxX = Math.max(1, Math.min(maxX, imageDims.width));
    maxY = Math.max(1, Math.min(maxY, imageDims.height));
  }

  const width = maxX - minX;
  const height = maxY - minY;
  if (!(width > 1 && height > 1)) {
    throw new Error("derived bounding box is too small/invalid.");
  }

  return { left: minX, top: minY, width, height };
}

function boxHasAuthoritativeObb(rawBox: any): boolean {
  return (
    Array.isArray(rawBox?.obbCorners) &&
    rawBox.obbCorners.length === 4 &&
    rawBox.obbCorners.every(
      (point: any) =>
        Array.isArray(point) &&
        point.length >= 2 &&
        Number.isFinite(Number(point[0])) &&
        Number.isFinite(Number(point[1]))
    ) &&
    Number.isFinite(Number(rawBox?.angle))
  );
}

function sessionContainsLegacyNonObbLabels(sessionDir: string): boolean {
  const labelsDir = path.join(sessionDir, "labels");
  if (!fs.existsSync(labelsDir)) return false;

  for (const labelName of fs.readdirSync(labelsDir).filter((name) => name.endsWith(".json"))) {
    const raw = safeReadJson(path.join(labelsDir, labelName));
    if (!raw || typeof raw !== "object") continue;

    const boxLists = [
      Array.isArray((raw as any).boxes) ? (raw as any).boxes : [],
      Array.isArray((raw as any)?.finalizedDetection?.acceptedBoxes)
        ? (raw as any).finalizedDetection.acceptedBoxes
        : [],
    ];
    for (const list of boxLists) {
      for (const box of list) {
        const hasLandmarks = Array.isArray(box?.landmarks) && box.landmarks.length > 0;
        if (hasLandmarks && !boxHasAuthoritativeObb(box)) {
          return true;
        }
      }
    }
  }

  return false;
}

// Ã¢â€â‚¬Ã¢â€â‚¬ Annotation file parsers Ã¢â€â‚¬Ã¢â€â‚¬

type AnnotationEntry = {
  boxes: ImportedRawBox[];
};

const MATCHABLE_IMAGE_EXTS = new Set([
  ".jpg",
  ".jpeg",
  ".png",
  ".gif",
  ".bmp",
  ".webp",
  ".tiff",
  ".tif",
]);

function pushUniqueKey(out: string[], key: string): void {
  const normalized = key.trim().toLowerCase();
  if (!normalized) return;
  if (!out.includes(normalized)) out.push(normalized);
}

/**
 * Build matchable keys from an annotation file path token.
 * Examples:
 * - "in/example.png" -> ["example.png", "example"]
 * - "xxx/something.png.yy" -> ["something.png.yy", "something.png", "something"]
 * - "in/example.jpeg.in" -> ["example.jpeg.in", "example.jpeg", "example"]
 */
function buildMatchableKeys(fileToken: string): string[] {
  const out: string[] = [];
  const raw = String(fileToken ?? "").trim();
  if (!raw) return out;

  // Normalize separators so basename extraction works across OS-exported paths.
  const normalizedPath = raw.replace(/\\/g, "/");
  const base = path.posix.basename(normalizedPath);
  if (!base) return out;

  pushUniqueKey(out, base);

  let candidate = base;
  let parsed = path.parse(candidate);
  pushUniqueKey(out, parsed.name);

  // Strip trailing non-image extensions until known image extension appears.
  while (parsed.ext) {
    if (MATCHABLE_IMAGE_EXTS.has(parsed.ext.toLowerCase())) {
      break;
    }
    candidate = parsed.name;
    parsed = path.parse(candidate);
    pushUniqueKey(out, candidate);
    pushUniqueKey(out, parsed.name);
  }

  return out;
}

function parseXmlAttrs(str: string): Record<string, string> {
  const attrs: Record<string, string> = {};
  // Handle quoted values (single or double quotes) Ã¢â‚¬â€ paths may contain slashes
  const re = /(\w+)=(?:'([^']*)'|"([^"]*)")/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(str)) !== null) {
    attrs[m[1]] = m[2] ?? m[3] ?? "";
  }
  return attrs;
}

/**
 * Parse an imglab-format dlib XML annotation file.
 * Returns a Map keyed by image basename Ã¢â€ â€™ { box, landmarks[] }.
 */
function parseImglabXml(filePath: string): Map<string, AnnotationEntry> {
  const content = fs.readFileSync(filePath, "utf-8");
  const result = new Map<string, AnnotationEntry>();

  // Match each <image ...> block
  const imageRe = /<image\s([^>]+)>([\s\S]*?)<\/image>/g;
  let imgMatch: RegExpExecArray | null;

  while ((imgMatch = imageRe.exec(content)) !== null) {
    const imgAttrs = parseXmlAttrs(imgMatch[1]);
    const fileAttr = imgAttrs["file"];
    if (!fileAttr) continue;
    const keys = buildMatchableKeys(fileAttr);
    if (keys.length === 0) continue;
    const body = imgMatch[2];

    const boxes: ImportedRawBox[] = [];
    const boxRe = /<box\s([^>]+)>([\s\S]*?)<\/box>/g;
    let boxMatch: RegExpExecArray | null;
    while ((boxMatch = boxRe.exec(body)) !== null) {
      const boxAttrs = parseXmlAttrs(boxMatch[1]);
      const landmarks: Array<{ id: number; x: number; y: number }> = [];
      const partRe = /<part\s([^>]+?)\/>/g;
      let partMatch: RegExpExecArray | null;
      while ((partMatch = partRe.exec(boxMatch[2])) !== null) {
        const pa = parseXmlAttrs(partMatch[1]);
        if (pa["name"] !== undefined && pa["x"] !== undefined && pa["y"] !== undefined) {
          landmarks.push({
            id: parseInt(pa["name"], 10),
            x: parseFloat(pa["x"]),
            y: parseFloat(pa["y"]),
          });
        }
      }
      landmarks.sort((a, b) => a.id - b.id);
      if (landmarks.length === 0) continue;
      boxes.push({
        left: parseFloat(boxAttrs["left"] ?? "0"),
        top: parseFloat(boxAttrs["top"] ?? "0"),
        width: parseFloat(boxAttrs["width"] ?? "0"),
        height: parseFloat(boxAttrs["height"] ?? "0"),
        landmarks,
      });
    }
    if (boxes.length === 0) continue;
    const entry = { boxes };
    for (const key of keys) {
      result.set(key, entry);
    }
  }

  return result;
}

/**
 * Parse a BioVision JSON annotation file.
 * Accepts a single object { imageFilename, boxes[] } or an array of such objects.
 * Returns a Map keyed by image filename Ã¢â€ â€™ { box, landmarks[] }.
 */
function parseBioVisionJson(filePath: string): Map<string, AnnotationEntry> {
  const raw = JSON.parse(fs.readFileSync(filePath, "utf-8"));
  const records: any[] = Array.isArray(raw) ? raw : [raw];
  const result = new Map<string, AnnotationEntry>();

  for (const record of records) {
    const imageFilename: string = record?.imageFilename;
    if (!imageFilename) continue;

    const rawBoxes: any[] = Array.isArray(record?.boxes) ? record.boxes : [];
    const topLevelLandmarks: any[] = Array.isArray(record?.landmarks)
      ? record.landmarks
      : Array.isArray(record?.annotations)
      ? record.annotations
      : [];
    const boxes: ImportedRawBox[] = [];
    if (rawBoxes.length > 0) {
      rawBoxes.forEach((box: any) => {
        if (!Array.isArray(box?.landmarks) || box.landmarks.length === 0) return;
        boxes.push({
          left: Number(box?.left ?? 0),
          top: Number(box?.top ?? 0),
          width: Number(box?.width ?? 0),
          height: Number(box?.height ?? 0),
          obbCorners: Array.isArray(box?.obbCorners) ? box.obbCorners : undefined,
          angle: isFiniteNumber(box?.angle) ? Number(box?.angle) : undefined,
          class_id: isFiniteNumber(box?.class_id) ? Math.round(Number(box?.class_id)) : undefined,
          landmarks: box.landmarks,
        });
      });
    } else if (topLevelLandmarks.length > 0) {
      boxes.push({ landmarks: topLevelLandmarks });
    }
    if (boxes.length === 0) continue;

    const entry = { boxes };
    const keys = buildMatchableKeys(imageFilename);
    if (keys.length === 0) continue;
    for (const key of keys) {
      result.set(key, entry);
    }
  }

  return result;
}

interface PreAnnotatedRecord {
  imageSourcePath: string;
  imageFilename: string;
  normalizedLabel: {
    imageFilename: string;
    boxes: ImportedNormalizedBox[];
  };
}

function collectPreAnnotatedRecords(
  datasetDir: string,
  speciesId?: string,
  geometryConfig?: ImportGeometryConfig
): { records: PreAnnotatedRecord[]; warnings: string[]; detailedWarnings: string[]; summary: ImportGeometrySummary } {
  const warnings: string[] = [];
  const detailedWarnings: string[] = [];
  const summary = createImportGeometrySummary();
  const importContext = speciesId
    ? loadSessionOrientationPolicyForCompatibility(speciesId)
    : {
        policy: normalizeOrientationPolicy(undefined, []),
        landmarkTemplate: [],
      };
  const labelDirCandidates = [
    path.join(datasetDir, "labels"),
    path.join(datasetDir, "Labels"),
    path.join(datasetDir, "annotations"),
    datasetDir,
  ];

  const labelsDir = labelDirCandidates.find((candidate) => {
    if (!fs.existsSync(candidate)) return false;
    if (!fs.statSync(candidate).isDirectory()) return false;
    return fs.readdirSync(candidate).some((file) => file.toLowerCase().endsWith(".json"));
  });

  if (!labelsDir) {
    throw new Error("No label JSON files found. Expected a labels/ folder or JSON files in the selected directory.");
  }

  const labelFiles = fs
    .readdirSync(labelsDir)
    .filter((file) => file.toLowerCase().endsWith(".json"))
    .map((file) => path.join(labelsDir, file));

  if (labelFiles.length === 0) {
    throw new Error("No label JSON files found in the selected dataset.");
  }

  const records: PreAnnotatedRecord[] = [];
  const idSets: Array<Set<number>> = [];

  for (const labelPath of labelFiles) {
    let parsed: any;
    try {
      parsed = JSON.parse(fs.readFileSync(labelPath, "utf-8"));
    } catch (e: any) {
      throw new Error(`Invalid JSON in ${path.basename(labelPath)}: ${e.message}`);
    }

    const imageFilenameRaw = parsed?.imageFilename;
    if (typeof imageFilenameRaw !== "string" || imageFilenameRaw.trim().length === 0) {
      throw new Error(`${path.basename(labelPath)} is missing a valid imageFilename.`);
    }
    const imageFilename = path.basename(imageFilenameRaw.trim());
    if (!DATASET_IMAGE_EXTS.test(imageFilename)) {
      detailedWarnings.push(`${path.basename(labelPath)} references a non-standard image extension: ${imageFilename}`);
    }

    const imageCandidates = [
      path.join(datasetDir, "images", imageFilename),
      path.join(datasetDir, "Images", imageFilename),
      path.join(datasetDir, imageFilename),
      path.join(path.dirname(labelPath), imageFilename),
    ];
    const imageSourcePath = imageCandidates.find((candidate) => {
      return fs.existsSync(candidate) && fs.statSync(candidate).isFile();
    });
    if (!imageSourcePath) {
      throw new Error(
        `${path.basename(labelPath)} references image '${imageFilename}', but it was not found in dataset/images or dataset root.`
      );
    }

    const imageDims = getImageDimensions(imageSourcePath);
    if (!imageDims) {
      detailedWarnings.push(`${path.basename(labelPath)}: could not read image dimensions; derived boxes won't be clamped.`);
    }

    const boxes = Array.isArray(parsed?.boxes) ? parsed.boxes : [];
    const hasBoxes = boxes.length > 0;
    const topLevelLandmarks = Array.isArray(parsed?.landmarks)
      ? parsed.landmarks
      : Array.isArray(parsed?.annotations)
      ? parsed.annotations
      : [];
    const canDeriveSingleBox = topLevelLandmarks.length > 0;
    if (!hasBoxes && !canDeriveSingleBox) {
      throw new Error(`${path.basename(labelPath)} has no boxes and no top-level landmarks/annotations to derive a box.`);
    }

    const rawBoxes: ImportedRawBox[] = hasBoxes
      ? boxes.map((box: any) => ({
          left: box?.left,
          top: box?.top,
          width: box?.width,
          height: box?.height,
          obbCorners: box?.obbCorners,
          angle: box?.angle,
          class_id: box?.class_id,
          landmarks: Array.isArray(box?.landmarks) ? box.landmarks : [],
        }))
      : [{ landmarks: topLevelLandmarks }];
    const normalizedBoxes: PreAnnotatedRecord["normalizedLabel"]["boxes"] = [];
    const landmarkIds = new Set<number>();
    let validLandmarkCount = 0;
    rawBoxes.forEach((box: ImportedRawBox, boxIndex: number) => {
      const landmarks = Array.isArray(box?.landmarks) ? box.landmarks : [];
      if (landmarks.length === 0) {
        throw new Error(`${path.basename(labelPath)} box ${boxIndex} has no landmarks.`);
      }

      const normalizedLandmarks = normalizeLandmarks(
        landmarks,
        `${path.basename(labelPath)} box ${boxIndex}`
      );
      normalizedLandmarks.forEach((lm) => {
        if (!lm.isSkipped) {
          landmarkIds.add(lm.id);
          validLandmarkCount += 1;
        }
      });
      normalizedBoxes.push(
        normalizeImportedBoxGeometry(
          box,
          normalizedLandmarks,
          imageDims,
          `${path.basename(labelPath)} box ${boxIndex}`,
          importContext.policy,
          importContext.landmarkTemplate,
          detailedWarnings,
          geometryConfig,
          summary
        )
      );
    });
    if (!hasBoxes) {
      detailedWarnings.push(`${path.basename(labelPath)}: no boxes found; derived 1 OBB from landmarks.`);
    }

    if (validLandmarkCount === 0) {
      throw new Error(`${path.basename(labelPath)} has no valid (non-skipped) landmarks.`);
    }

    if (landmarkIds.size > 0) {
      idSets.push(landmarkIds);
    }

    records.push({
      imageSourcePath,
      imageFilename,
      normalizedLabel: {
        imageFilename,
        boxes: normalizedBoxes,
      },
    });
  }

  if (records.length === 0) {
    throw new Error("No valid annotations found in selected dataset.");
  }

  if (idSets.length > 1) {
    const common = new Set<number>(idSets[0]);
    for (const ids of idSets.slice(1)) {
      for (const id of [...common]) {
        if (!ids.has(id)) common.delete(id);
      }
    }
    if (common.size === 0) {
      detailedWarnings.push("No common landmark IDs across imported samples. prepare_dataset may fail.");
    } else {
      const uniqueAll = new Set<number>();
      idSets.forEach((ids) => ids.forEach((id) => uniqueAll.add(id)));
      if (common.size < uniqueAll.size) {
        detailedWarnings.push(
          `Landmark IDs are inconsistent across files. Common IDs retained for training: ${[...common]
            .sort((a, b) => a - b)
            .join(", ")}`
        );
      }
    }
  }

  warnings.push(...summarizeImportGeometry(summary));
  return { records, warnings, detailedWarnings, summary };
}

function summarizeValidationErrors(validation: any): string {
  if (!validation) return "Unknown validation error.";
  const errors = Array.isArray(validation.errors) ? validation.errors : [];
  if (errors.length === 0) return "Unknown validation error.";
  if (errors.length === 1) return errors[0];
  return `${errors[0]} (+${errors.length - 1} more)`;
}

function summarizeLabelDataset(effectiveRoot: string): {
  labelFiles: number;
  trainableImages: number;
  landmarkStatus: "ok" | "warning";
  landmarkMessage: string;
  warnings: string[];
} {
  const warnings: string[] = [];
  const labelsDir = path.join(effectiveRoot, "labels");
  if (!fs.existsSync(labelsDir)) {
    return {
      labelFiles: 0,
      trainableImages: 0,
      landmarkStatus: "warning",
      landmarkMessage: "No labels found",
      warnings: ["labels directory does not exist"],
    };
  }

  const jsonPaths = fs
    .readdirSync(labelsDir)
    .filter((name) => name.toLowerCase().endsWith(".json"))
    .map((name) => path.join(labelsDir, name));

  const idSets: Array<Set<number>> = [];
  let trainableImages = 0;

  for (const labelPath of jsonPaths) {
    try {
      const parsed = JSON.parse(fs.readFileSync(labelPath, "utf-8"));
      const boxes = Array.isArray(parsed?.boxes) ? parsed.boxes : [];
      const ids = new Set<number>();

      for (const box of boxes) {
        const landmarks = Array.isArray(box?.landmarks) ? box.landmarks : [];
        for (const lm of landmarks) {
          if (lm?.isSkipped) continue;
          const id = Number(lm?.id);
          const x = Number(lm?.x);
          const y = Number(lm?.y);
          if (Number.isFinite(id) && Number.isFinite(x) && Number.isFinite(y) && x >= 0 && y >= 0) {
            ids.add(id);
          }
        }
      }

      if (ids.size > 0) {
        idSets.push(ids);
        trainableImages += 1;
      }
    } catch {
      warnings.push(`invalid JSON skipped: ${path.basename(labelPath)}`);
    }
  }

  if (idSets.length === 0) {
    return {
      labelFiles: jsonPaths.length,
      trainableImages,
      landmarkStatus: "warning",
      landmarkMessage: "No valid landmark sets",
      warnings,
    };
  }

  let landmarkStatus: "ok" | "warning" = "ok";
  let landmarkMessage = "consistent";
  if (idSets.length > 1) {
    const common = new Set<number>(idSets[0]);
    for (const ids of idSets.slice(1)) {
      for (const id of [...common]) {
        if (!ids.has(id)) common.delete(id);
      }
    }

    const uniqueAll = new Set<number>();
    idSets.forEach((ids) => ids.forEach((id) => uniqueAll.add(id)));
    if (common.size === 0) {
      landmarkStatus = "warning";
      landmarkMessage = "no common IDs across samples";
    } else if (common.size < uniqueAll.size) {
      landmarkStatus = "warning";
      landmarkMessage = `partial overlap; common IDs: ${[...common].sort((a, b) => a - b).join(", ")}`;
    }
  }

  return {
    labelFiles: jsonPaths.length,
    trainableImages,
    landmarkStatus,
    landmarkMessage,
    warnings,
  };
}

interface TrainOptions {
  testSplit?: number;  // Fraction for test set (default 0.2)
  seed?: number;       // Random seed for reproducibility
  customOptions?: Record<string, number | boolean>;  // Custom training parameters
  speciesId?: string;  // Session-scoped training
  useImportedXml?: boolean; // Train directly from existing xml/train_{tag}.xml
  predictorType?: "dlib" | "cnn"; // Predictor backend (default: "dlib")
  cnnVariant?: string; // CNN backbone id (e.g. simplebaseline, mobilenet_v3_large)
}

const FALLBACK_CNN_VARIANTS = [
  {
    id: "simplebaseline",
    label: "SimpleBaseline (ResNet-50)",
    description: "Balanced accuracy/speed baseline for most datasets.",
    selectable: true,
    recommended: true,
    reason: null as string | null,
  },
  {
    id: "mobilenet_v3_large",
    label: "MobileNetV3 Large",
    description: "Fastest option; useful on CPU or lower-memory systems.",
    selectable: true,
    recommended: false,
    reason: null as string | null,
  },
  {
    id: "efficientnet_b0",
    label: "EfficientNet-B0",
    description: "Compact backbone with strong generalization on medium datasets.",
    selectable: true,
    recommended: false,
    reason: null as string | null,
  },
  {
    id: "hrnet_w32",
    label: "HRNet-W32",
    description: "Highest-capacity option; best with stronger GPU resources.",
    selectable: false,
    recommended: false,
    reason: "Capability probe unavailable. Use simplebaseline by default.",
  },
];

async function resolveCnnVariantCapabilities(): Promise<{
  ok: boolean;
  torchAvailable: boolean;
  torchvisionAvailable: boolean;
  device: string;
  gpuName: string | null;
  gpuMemoryGb: number | null;
  defaultVariant: string;
  variants: Array<{
    id: string;
    label: string;
    description: string;
    selectable: boolean;
    recommended?: boolean;
    reason?: string | null;
  }>;
  warning?: string;
}> {
  try {
    const out = await runPython([path.join(__dirname, "../backend/inference/list_cnn_variants.py")]);
    const parsed = JSON.parse(out || "{}");
    if (!parsed || !Array.isArray(parsed.variants)) {
      throw new Error("Invalid CNN variant response payload.");
    }
    return {
      ok: true,
      torchAvailable: !!parsed.torch_available,
      torchvisionAvailable: !!parsed.torchvision_available,
      device: String(parsed.device || "cpu"),
      gpuName: parsed.gpu_name ?? null,
      gpuMemoryGb: parsed.gpu_memory_gb ?? null,
      defaultVariant: String(parsed.default_variant || "simplebaseline"),
      variants: parsed.variants,
    };
  } catch (e: any) {
    return {
      ok: true,
      torchAvailable: false,
      torchvisionAvailable: false,
      device: "cpu",
      gpuName: null,
      gpuMemoryGb: null,
      defaultVariant: "simplebaseline",
      variants: FALLBACK_CNN_VARIANTS,
      warning: e?.message || "Capability probe failed; using safe defaults.",
    };
  }
}

ipcMain.handle("ml:get-cnn-variants", async () => {
  return resolveCnnVariantCapabilities();
});

ipcMain.handle(
  "ml:training-preflight",
  async (
    _event,
    args: {
      speciesId?: string;
      modelName: string;
      useImportedXml?: boolean;
      workspaceImages?: number;
      importedImagesHint?: number;
    }
  ) => {
    try {
      const modelName = (args?.modelName || "").trim();
      if (!MODEL_NAME_RE.test(modelName)) {
        return { ok: false, error: "Enter a valid model name for preflight." };
      }

      const effectiveRoot = getEffectiveRoot(args?.speciesId);
      ensureTrainingLayout(effectiveRoot);
      const useImportedXml = !!args?.useImportedXml;

      if (useImportedXml) {
        const xmlValidator = path.join(__dirname, "../backend/data/validate_dlib_xml.py");
        const trainXml = path.join(effectiveRoot, "xml", `train_${modelName}.xml`);
        if (!fs.existsSync(trainXml)) {
          return { ok: false, error: `train_${modelName}.xml not found.` };
        }

        const trainValidation = JSON.parse(await runPython([xmlValidator, trainXml]));
        if (!trainValidation.ok) {
          return {
            ok: false,
            error: `Train XML validation failed: ${summarizeValidationErrors(trainValidation)}`,
          };
        }

        const testXml = path.join(effectiveRoot, "xml", `test_${modelName}.xml`);
        let testValidation: any = null;
        if (fs.existsSync(testXml)) {
          testValidation = JSON.parse(await runPython([xmlValidator, testXml]));
          if (!testValidation.ok) {
            return {
              ok: false,
              error: `Test XML validation failed: ${summarizeValidationErrors(testValidation)}`,
            };
          }
        }

        return {
          ok: true,
          useImportedXml: true,
          trainXmlImages: trainValidation.num_images,
          testXmlImages: testValidation ? testValidation.num_images : 0,
          landmarkStatus: "ok",
          landmarkMessage: "validated",
          warnings: [],
        };
      }

      const summary = summarizeLabelDataset(effectiveRoot);
      const workspaceImages = args?.workspaceImages ?? 0;
      const importedImages = args?.importedImagesHint ?? 0;

      return {
        ok: true,
        useImportedXml: false,
        workspaceImages,
        importedImages,
        totalTrainableImages: summary.trainableImages,
        landmarkStatus: summary.landmarkStatus,
        landmarkMessage: summary.landmarkMessage,
        warnings: summary.warnings,
      };
    } catch (e: any) {
      console.error("Training preflight failed:", e);
      return { ok: false, error: e.message };
    }
  }
);

ipcMain.handle("ml:train", async (_event, modelName: string, options?: TrainOptions) => {
  const predictorTypeRaw = (options as any)?.predictorType ?? "dlib";
  if (predictorTypeRaw !== "dlib" && predictorTypeRaw !== "cnn") {
    return { ok: false, error: "Unsupported predictor type. Use dlib or cnn." };
  }
  const predictorType: "dlib" | "cnn" = predictorTypeRaw;
  let lastTrainProgressPercent = 0;
  const emitTrainProgress = (
    percent: number,
    stage: string,
    message: string,
    details?: Record<string, unknown>
  ) => {
    const rawPct = Math.max(0, Math.min(100, Math.round(percent)));
    // Keep train progress monotonic across nested evaluation/parity subloops
    // (some backend phases restart local progress counters).
    const pct = Math.max(lastTrainProgressPercent, rawPct);
    lastTrainProgressPercent = pct;
    const detailPayload =
      details && typeof details === "object"
        ? {
            ...details,
            raw_percent: rawPct,
            monotonic_percent: pct,
          }
        : details;
    mainWindow?.webContents.send("ml:train-progress", {
      percent: pct,
      stage,
      message,
      predictorType,
      modelName,
      details: detailPayload,
    });
  };
  const resolveProgressStage = (fallback: string, details?: Record<string, unknown>) => {
    const raw = details?.["stage"];
    if (typeof raw === "string" && raw.trim().length > 0) {
      return raw.trim();
    }
    return fallback;
  };
  const resolveProgressMessage = (fallback: string, details?: Record<string, unknown>) => {
    const raw = details?.["message"];
    if (typeof raw === "string" && raw.trim().length > 0) {
      return raw.trim();
    }
    return fallback;
  };

  try {
    if (!MODEL_NAME_RE.test(modelName)) {
      throw new Error("Invalid model name. Use letters, numbers, dot, underscore, or hyphen.");
    }

    const testSplit = options?.testSplit ?? 0.2;
    const seed = options?.seed ?? 42;
    const effectiveRoot = getEffectiveRoot(options?.speciesId);
    ensureTrainingLayout(effectiveRoot);
    emitTrainProgress(3, "preflight", "Validating training inputs...");

    if (options?.useImportedXml) {
      const xmlValidator = path.join(__dirname, "../backend/data/validate_dlib_xml.py");
      const trainXml = path.join(effectiveRoot, "xml", `train_${modelName}.xml`);
      if (!fs.existsSync(trainXml)) {
        throw new Error(`train_${modelName}.xml not found. Import a dlib train XML file first.`);
      }

      const trainValidation = JSON.parse(await runPython([xmlValidator, trainXml]));
      if (!trainValidation.ok) {
        throw new Error(`Train XML validation failed: ${summarizeValidationErrors(trainValidation)}`);
      }

      const testXml = path.join(effectiveRoot, "xml", `test_${modelName}.xml`);
      if (fs.existsSync(testXml)) {
        const testValidation = JSON.parse(await runPython([xmlValidator, testXml]));
        if (!testValidation.ok) {
          throw new Error(`Test XML validation failed: ${summarizeValidationErrors(testValidation)}`);
        }
      }
      emitTrainProgress(20, "preflight", "Imported XML validated.");
    } else {
      // Prepare dataset with train/test split
      emitTrainProgress(12, "prepare_dataset", "Preparing dataset...");
      await runPythonWithProgress([
        path.join(__dirname, "../backend/data/prepare_dataset.py"),
        effectiveRoot,
        modelName,
        testSplit.toString(),
        seed.toString(),
      ], (pct, stage, details) => {
        const scaled = 12 + Math.round((Math.max(0, Math.min(100, pct)) / 100) * 18);
        const uiStage = resolveProgressStage("prepare_dataset", details);
        const uiMessage = resolveProgressMessage(stage, details);
        emitTrainProgress(scaled, uiStage, uiMessage, details);
      });
      emitTrainProgress(30, "prepare_dataset", "Dataset preparation complete.");
    }

    // Run dataset audit (non-blocking: surface warnings but don't abort)
    let auditReport: Record<string, unknown> | null = null;
    emitTrainProgress(35, "evaluation", "Auditing dataset...");
    try {
      const auditScript = path.join(__dirname, "../backend/data/audit_dataset.py");
      if (fs.existsSync(auditScript)) {
        const auditOut = await runPython([
          auditScript,
          "--project-root", effectiveRoot,
          "--tag", modelName,
        ]).catch(() => null);
        if (auditOut) {
          const debugDir = path.join(effectiveRoot, "debug");
          const auditPath = path.join(debugDir, `audit_${modelName}.json`);
          if (fs.existsSync(auditPath)) {
            try { auditReport = JSON.parse(fs.readFileSync(auditPath, "utf-8")); } catch (_) {}
          }
        }
      }
    } catch (_) {}

    // Train with selected predictor type
    let out: string;
    if (predictorType === "cnn") {
      const cnnCapabilities = await resolveCnnVariantCapabilities();
      const selectableVariants = (cnnCapabilities.variants || []).filter(
        (v) => v && v.selectable
      );
      if (selectableVariants.length === 0) {
        const reasonBits: string[] = [];
        if (!cnnCapabilities.torchAvailable) {
          reasonBits.push("PyTorch is not available");
        }
        if (!cnnCapabilities.torchvisionAvailable) {
          reasonBits.push("torchvision is not available");
        }
        if (cnnCapabilities.warning) {
          reasonBits.push(cnnCapabilities.warning);
        }
        const suffix =
          reasonBits.length > 0 ? ` (${reasonBits.join("; ")})` : "";
        throw new Error(
          `CNN training is unavailable on this system${suffix}.`
        );
      }

      const cnnVariantRaw = (options as any)?.cnnVariant;
      const requestedVariant =
        typeof cnnVariantRaw === "string" && cnnVariantRaw.trim().length > 0
          ? cnnVariantRaw.trim()
          : "";
      let cnnVariant = requestedVariant;
      if (requestedVariant) {
        const requested = cnnCapabilities.variants.find(
          (v) => v.id === requestedVariant
        );
        if (requested && !requested.selectable) {
          throw new Error(
            `CNN variant "${requestedVariant}" is not available on this system${
              requested.reason ? `: ${requested.reason}` : "."
            }`
          );
        }
      }
      if (!cnnVariant || !selectableVariants.some((v) => v.id === cnnVariant)) {
        const fallbackVariant =
          selectableVariants.find(
            (v) => v.id === cnnCapabilities.defaultVariant
          )?.id ?? selectableVariants[0].id;
        cnnVariant = fallbackVariant;
      }
      const epochsRaw = Number(options?.customOptions?.epochs);
      const lrRaw = Number((options as any)?.customOptions?.lr);
      const batchRaw = Number((options as any)?.customOptions?.batch ?? (options as any)?.customOptions?.batch_size);
      const cnnArgs = [
        path.join(__dirname, "../backend/training/train_cnn_model.py"),
        effectiveRoot,
        modelName,
        "--model-variant", cnnVariant,
      ];
      // Pass resolved device so Python honours hardware routing + AMP selection
      if (cnnCapabilities.device && ["cpu", "mps", "cuda"].includes(cnnCapabilities.device)) {
        cnnArgs.push("--device", cnnCapabilities.device);
      }
      emitTrainProgress(
        40,
        "preflight",
        `CNN system gate: device=${cnnCapabilities.device}, variant=${cnnVariant}.`,
        {
          stage: "preflight",
          device: cnnCapabilities.device,
          gpu_name: cnnCapabilities.gpuName ?? undefined,
          gpu_memory_gb: cnnCapabilities.gpuMemoryGb ?? undefined,
          model_variant: cnnVariant,
        }
      );
      if (Number.isFinite(epochsRaw) && epochsRaw > 0) {
        cnnArgs.push("--epochs", String(Math.round(epochsRaw)));
      }
      if (Number.isFinite(lrRaw) && lrRaw > 0) {
        cnnArgs.push("--lr", String(lrRaw));
      }
      if (Number.isFinite(batchRaw) && batchRaw > 0) {
        cnnArgs.push("--batch-size", String(Math.round(batchRaw)));
      }
      emitTrainProgress(42, "training", "Training CNN model...");
      out = await runPythonWithProgress(cnnArgs, (pct, stage, details) => {
        const scaled = 42 + Math.round((Math.max(0, Math.min(100, pct)) / 100) * 50);
        const uiStage = resolveProgressStage("training", details);
        const uiMessage = resolveProgressMessage(stage, details);
        emitTrainProgress(scaled, uiStage, uiMessage, details);
      });
    } else {
      const trainArgs = [
        path.join(__dirname, "../backend/training/train_shape_model.py"),
        effectiveRoot,
        modelName,
      ];
      if (options?.customOptions) {
        trainArgs.push(JSON.stringify(options.customOptions));
      }
      emitTrainProgress(42, "training", "Training dlib shape predictor...");
      out = await runPythonWithProgress(trainArgs, (pct, stage, details) => {
        const scaled = 42 + Math.round((Math.max(0, Math.min(100, pct)) / 100) * 50);
        const uiStage = resolveProgressStage("training", details);
        const uiMessage = resolveProgressMessage(stage, details);
        emitTrainProgress(scaled, uiStage, uiMessage, details);
      });
    }

    emitTrainProgress(95, "evaluation", "Evaluating trained model...");

    // Parse output for train/test errors (mean and median)
    const trainErrorMatch       = out.match(/TRAIN_ERROR\s+([\d.]+)/);
    const testErrorMatch        = out.match(/TEST_ERROR\s+([\d.]+)/);
    const trainMedianErrorMatch = out.match(/TRAIN_MEDIAN_ERROR\s+([\d.e+-]+)/);
    const testMedianErrorMatch  = out.match(/TEST_MEDIAN_ERROR\s+([\d.e+-]+)/);
    const modelPathMatch = out.match(/MODEL_PATH\s+(.+)/);

    syncLandmarkModelRegistry(effectiveRoot, {
      setActive: { name: modelName, predictorType },
    });

    emitTrainProgress(100, "done", "Training complete.");

    return {
      ok: true,
      output: out,
      trainError:       trainErrorMatch       ? parseFloat(trainErrorMatch[1])       : null,
      testError:        testErrorMatch        ? parseFloat(testErrorMatch[1])        : null,
      trainMedianError: trainMedianErrorMatch ? parseFloat(trainMedianErrorMatch[1]) : null,
      testMedianError:  testMedianErrorMatch  ? parseFloat(testMedianErrorMatch[1])  : null,
      modelPath:        modelPathMatch        ? modelPathMatch[1].trim()             : null,
      auditReport:      auditReport ?? undefined,
    };
  } catch (e: any) {
    emitTrainProgress(100, "error", `Training failed: ${e.message}`);
    console.error("Training failed:", e);
    return { ok: false, error: e.message };
  }
});

ipcMain.handle("ml:import-preannotated-dataset", async (_event, args?: { speciesId?: string; geometryConfig?: ImportGeometryConfig }) => {
  try {
    const picker = await dialog.showOpenDialog({
      properties: ["openDirectory"],
      title: "Select pre-annotated dataset folder",
    });
    if (picker.canceled || picker.filePaths.length === 0) {
      return { ok: false, canceled: true, warnings: [] };
    }

    const sourceDir = picker.filePaths[0];
    const { records, warnings, detailedWarnings, summary } = collectPreAnnotatedRecords(
      sourceDir,
      args?.speciesId,
      args?.geometryConfig
    );
    if (detailedWarnings.length > 0) {
      console.warn("[ml:import-preannotated-dataset] import details:", detailedWarnings.slice(0, 50));
    }

    const effectiveRoot = getEffectiveRoot(args?.speciesId);
    ensureTrainingLayout(effectiveRoot);
    const imagesDir = path.join(effectiveRoot, "images");
    const labelsDir = path.join(effectiveRoot, "labels");

    let importedImages = 0;
    let importedLabels = 0;
    let overwrittenImages = 0;
    let overwrittenLabels = 0;
    const seenImages = new Set<string>();

    for (const record of records) {
      const imageName = path.basename(record.imageFilename);
      const imageDest = path.join(imagesDir, imageName);
      const labelName = imageName.replace(/\.[^.]+$/, ".json");
      const labelDest = path.join(labelsDir, labelName);

      if (fs.existsSync(imageDest)) overwrittenImages += 1;
      if (!seenImages.has(imageName)) {
        fs.copyFileSync(record.imageSourcePath, imageDest);
        importedImages += 1;
        seenImages.add(imageName);
      }

      if (fs.existsSync(labelDest)) overwrittenLabels += 1;
      fs.writeFileSync(
        labelDest,
        JSON.stringify(
          {
            ...record.normalizedLabel,
            boxes: record.normalizedLabel.boxes.map((box, index) => {
              const { geometryOrigin, ...persistedBox } = box;
              return { id: index, ...persistedBox };
            }),
          },
          null,
          2
        )
      );
      importedLabels += 1;
    }

    return {
      ok: true,
      canceled: false,
      sourceDir,
      importedImages,
      importedLabels,
      overwrittenImages,
      overwrittenLabels,
      warnings,
      importSummary: summary,
    };
  } catch (e: any) {
    console.error("Import pre-annotated dataset failed:", e);
    return { ok: false, canceled: false, error: e.message, warnings: [] };
  }
});

ipcMain.handle("ml:import-dlib-xml", async (_event, args: { modelName: string; speciesId?: string }) => {
  try {
    const modelName = args?.modelName?.trim();
    if (!modelName || !MODEL_NAME_RE.test(modelName)) {
      return {
        ok: false,
        canceled: false,
        warnings: [],
        error: "Model name is required before importing dlib XML.",
      };
    }

    const picker = await dialog.showOpenDialog({
      properties: ["openFile", "multiSelections"],
      filters: [{ name: "XML", extensions: ["xml"] }],
      title: "Select dlib XML files (train and optional test)",
    });
    if (picker.canceled || picker.filePaths.length === 0) {
      return { ok: false, canceled: true, warnings: [] };
    }

    const xmlFiles = picker.filePaths.filter((file) => file.toLowerCase().endsWith(".xml"));
    if (xmlFiles.length === 0) {
      return { ok: false, canceled: false, warnings: [], error: "No XML files selected." };
    }

    const warnings: string[] = [];
    const namedTrain = xmlFiles.find((file) => /train/i.test(path.basename(file)));
    const trainXml = namedTrain ?? xmlFiles[0];
    const remaining = xmlFiles.filter((file) => file !== trainXml);
    const testXml = remaining.find((file) => /test/i.test(path.basename(file))) ?? remaining[0];

    if (!namedTrain && xmlFiles.length > 1) {
      warnings.push(`Train XML inferred from first file: ${path.basename(trainXml)}`);
    }
    if (xmlFiles.length > 2) {
      warnings.push("More than two XML files selected; only one train and one test file were imported.");
    }

    const effectiveRoot = getEffectiveRoot(args.speciesId);
    ensureTrainingLayout(effectiveRoot);
    const xmlDir = path.join(effectiveRoot, "xml");
    const xmlValidator = path.join(__dirname, "../backend/data/validate_dlib_xml.py");

    const trainDest = path.join(xmlDir, `train_${modelName}.xml`);
    const trainValidation = JSON.parse(await runPython([xmlValidator, trainXml, trainDest]));
    if (!trainValidation.ok) {
      return {
        ok: false,
        canceled: false,
        warnings,
        error: `Train XML validation failed: ${summarizeValidationErrors(trainValidation)}`,
      };
    }

    let testDest: string | undefined;
    let testValidation: any = null;
    if (testXml) {
      testDest = path.join(xmlDir, `test_${modelName}.xml`);
      testValidation = JSON.parse(await runPython([xmlValidator, testXml, testDest]));
      if (!testValidation.ok) {
        return {
          ok: false,
          canceled: false,
          warnings,
          error: `Test XML validation failed: ${summarizeValidationErrors(testValidation)}`,
        };
      }
    }

    return {
      ok: true,
      canceled: false,
      warnings,
      trainXmlPath: trainDest,
      testXmlPath: testDest,
      trainStats: {
        num_images: trainValidation.num_images,
        num_boxes: trainValidation.num_boxes,
        num_parts: trainValidation.num_parts,
      },
      testStats: testValidation
        ? {
            num_images: testValidation.num_images,
            num_boxes: testValidation.num_boxes,
            num_parts: testValidation.num_parts,
          }
        : null,
    };
  } catch (e: any) {
    console.error("Import dlib XML failed:", e);
    return { ok: false, canceled: false, warnings: [], error: e.message };
  }
});

// Run detailed model testing
ipcMain.handle("ml:test-model", async (_event, args: string | { modelName: string; speciesId?: string }) => {
  try {
    // Accept either legacy string arg or new object with speciesId
    const modelName = typeof args === "string" ? args : args.modelName;
    const speciesId = typeof args === "object" ? args.speciesId : undefined;
    const effectiveRoot = getEffectiveRoot(speciesId);

    const out = await runPython([
      path.join(__dirname, "../backend/inference/shape_tester.py"),
      effectiveRoot,
      modelName,
    ]);

    // Parse the JSON summary from output
    const jsonStart = out.lastIndexOf("{");
    if (jsonStart >= 0) {
      const jsonStr = out.substring(jsonStart);
      const results = JSON.parse(jsonStr);
      return { ok: true, results };
    }

    return { ok: true, output: out };
  } catch (e: any) {
    console.error("Model testing failed:", e);
    return { ok: false, error: e.message };
  }
});


ipcMain.handle("ml:save-labels", async (_event, images) => {
  const imagesDir = path.join(projectRoot, "images");
  const labelsDir = path.join(projectRoot, "labels");

  fs.mkdirSync(imagesDir, { recursive: true });
  fs.mkdirSync(labelsDir, { recursive: true });

  for (const img of images) {
    const destImagePath = path.join(imagesDir, img.filename);
    fs.copyFileSync(img.path, destImagePath);

    // Export boxes with nested landmarks
    const boxes = (img.boxes || []).map((box: any) => ({
      left: box.left,
      top: box.top,
      width: box.width,
      height: box.height,
      landmarks: (box.landmarks || []).map((lm: any) => ({
        x: lm.x,
        y: lm.y,
        id: lm.id,
      })),
    }));

    fs.writeFileSync(
      path.join(labelsDir, img.filename.replace(/\.\w+$/, ".json")),
      JSON.stringify({
        imageFilename: img.filename,
        boxes: boxes,
      }, null, 2)
    );
  }

  return { ok: true };
});

interface PredictOptions {
  multiSpecimen?: boolean;
  predictorType?: "dlib" | "cnn";
  allowIncompatible?: boolean;
  boxes?: Array<{
    left: number;
    top: number;
    width: number;
    height: number;
    right?: number;
    bottom?: number;
    obbCorners?: [number, number][];
    angle?: number;
    class_id?: number;
    orientation_hint?: {
      orientation?: "left" | "right";
      confidence?: number;
      source?: string;
      head_point?: [number, number];
      tail_point?: [number, number];
    };
  }>;
}

interface PredictBatchItem {
  batchIndex: number;
  imagePath: string;
  filename?: string;
  boxes: NonNullable<PredictOptions["boxes"]>;
}

interface PredictBatchArgs {
  speciesId?: string;
  modelName: string;
  predictorType?: "dlib" | "cnn";
  allowIncompatible?: boolean;
  items: PredictBatchItem[];
}

async function runPredictionRequest(args: {
  imagePath: string;
  tag: string;
  speciesId?: string;
  options?: PredictOptions;
  onProgress?: (percent: number, stage: string, details?: Record<string, unknown>) => void;
}): Promise<any> {
  let tempFile: string | null = null;
  let tempBoxesFile: string | null = null;
  try {
    const modelRoot = getEffectiveRoot(args.speciesId);
    let effectivePath = args.imagePath;
    if (/[^\x00-\x7F]/.test(args.imagePath)) {
      const ext = path.extname(args.imagePath);
      tempFile = path.join(app.getPath("temp"), `bv_infer_${Date.now()}${ext}`);
      fs.copyFileSync(args.imagePath, tempFile);
      effectivePath = tempFile;
    }

    const cnnModelPath = path.join(modelRoot, "models", `cnn_${args.tag}.pth`);
    const dlibModelPath = path.join(modelRoot, "models", `predictor_${args.tag}.dat`);
    const requestedPredictor = args.options?.predictorType;
    let predictorType: "dlib" | "cnn";
    if (requestedPredictor === "cnn") {
      if (!fs.existsSync(cnnModelPath)) {
        throw new Error(`CNN model not found for "${args.tag}".`);
      }
      predictorType = "cnn";
    } else if (requestedPredictor === "dlib") {
      if (!fs.existsSync(dlibModelPath)) {
        throw new Error(`dlib model not found for "${args.tag}".`);
      }
      predictorType = "dlib";
    } else {
      predictorType = fs.existsSync(cnnModelPath) ? "cnn" : "dlib";
    }

    const compatibility = await evaluateModelCompatibility({
      speciesId: args.speciesId,
      modelName: args.tag,
      predictorType,
      includeRuntime: true,
    });
    if (!compatibility.ok) {
      throw new Error(compatibility.error || "Model/session compatibility check failed.");
    }
    if (compatibility.blocking && !args.options?.allowIncompatible) {
      throw new Error(
        `Inference blocked by compatibility checks. ${formatCompatibilityErrorSummary(
          compatibility.issues
        )} Use override to continue.`
      );
    }

    const obbDetectorPath = compatibility.obbDetectorPath;
    const hasProvidedBoxes = Array.isArray(args.options?.boxes) && args.options!.boxes.length > 0;
    if (!hasProvidedBoxes && !obbDetectorPath) {
      throw new Error(
        "OBB detector required for inference when no oriented boxes are provided. Train the session OBB detector first."
      );
    }

    const pythonArgs = [
      path.join(__dirname, "../backend/inference/predict.py"),
      modelRoot,
      args.tag,
      effectivePath,
      "--predictor-type",
      predictorType,
    ];
    if (args.options?.multiSpecimen) {
      pythonArgs.push("--multi");
    }
    if (obbDetectorPath) {
      pythonArgs.push("--yolo-model", obbDetectorPath);
    }
    const boxesForInference =
      Array.isArray(args.options?.boxes) && args.options!.boxes.length > 0 ? args.options!.boxes : undefined;

    if (Array.isArray(boxesForInference) && boxesForInference.length > 0) {
      tempBoxesFile = path.join(
        app.getPath("temp"),
        `bv_boxes_${Date.now()}_${Math.random().toString(16).slice(2)}.json`
      );
      fs.writeFileSync(tempBoxesFile, JSON.stringify(boxesForInference));
      pythonArgs.push("--boxes-json", tempBoxesFile);
    }

    const out = await runPythonWithProgress(pythonArgs, args.onProgress);
    return JSON.parse(out);
  } finally {
    if (tempFile) {
      try {
        fs.unlinkSync(tempFile);
      } catch {}
    }
    if (tempBoxesFile) {
      try {
        fs.unlinkSync(tempBoxesFile);
      } catch {}
    }
  }
}

ipcMain.handle("ml:predict", async (_event, imagePath: string, tag: string, speciesId?: string, options?: PredictOptions) => {
  try {
    const data = await runPredictionRequest({
      imagePath,
      tag,
      speciesId,
      options,
      onProgress: (percent, stage) => {
        mainWindow?.webContents.send("ml:predict-progress", { percent, stage });
      },
    });
    return { ok: true, data };
  } catch (e: any) {
    console.error("Prediction failed:", e);
    return { ok: false, error: e.message };
  }
});

ipcMain.handle("ml:predict-batch", async (_event, args: PredictBatchArgs) => {
  try {
    if (!args?.modelName) {
      throw new Error("modelName is required.");
    }
    if (!Array.isArray(args.items) || args.items.length === 0) {
      return { ok: true, results: [] };
    }

    const speciesId = args.speciesId;
    const modelRoot = getEffectiveRoot(speciesId);
    const requestedPredictor = args.predictorType === "cnn" ? "cnn" : "dlib";
    const total = args.items.length;

    mainWindow?.webContents.send("ml:predict-progress", {
      percent: 4,
      stage: "checking_compatibility",
      currentIndex: 0,
      total,
    });

    const compatibility = await evaluateModelCompatibility({
      speciesId,
      modelName: args.modelName,
      predictorType: requestedPredictor,
      includeRuntime: true,
    });
    if (!compatibility.ok) {
      throw new Error(compatibility.error || "Model/session compatibility check failed.");
    }
    if (compatibility.blocking && !args.allowIncompatible) {
      throw new Error(
        `Inference blocked by compatibility checks. ${formatCompatibilityErrorSummary(
          compatibility.issues
        )} Use override to continue.`
      );
    }

    mainWindow?.webContents.send("ml:predict-progress", {
      percent: 8,
      stage: "preparing_worker",
      currentIndex: 0,
      total,
    });

    const results: Array<{
      batchIndex: number;
      imagePath: string;
      filename?: string;
      ok: boolean;
      data?: any;
      error?: string;
    }> = [];

    for (let idx = 0; idx < args.items.length; idx++) {
      const item = args.items[idx];
      const batchIndex = Number(item.batchIndex);
      const currentIndex = idx + 1;
      try {
        mainWindow?.webContents.send("ml:predict-progress", {
          percent: 10 + Math.round(((currentIndex - 1) / total) * 80),
          stage: idx === 0 ? "loading_model" : "predicting",
          currentIndex,
          total,
          imagePath: item.imagePath,
        });

        let data: any;
        try {
          data = await landmarkInferenceWorker.predict(
            {
              project_root: modelRoot,
              tag: args.modelName,
              predictor_type: requestedPredictor,
              image_path: item.imagePath,
              boxes: item.boxes,
            },
            (progress) => {
              const base = (currentIndex - 1) / total;
              const span = 1 / total;
              const overall = 10 + Math.round((base + (Math.max(0, Math.min(100, progress.percent)) / 100) * span) * 80);
              mainWindow?.webContents.send("ml:predict-progress", {
                percent: overall,
                stage: progress.stage,
                currentIndex,
                total,
                imagePath: item.imagePath,
              });
            }
          );
        } catch (workerError) {
          console.warn("Landmark worker predict failed, falling back to one-shot path:", workerError);
          data = await runPredictionRequest({
            imagePath: item.imagePath,
            tag: args.modelName,
            speciesId,
            options: {
              multiSpecimen: true,
              predictorType: requestedPredictor,
              allowIncompatible: true,
              boxes: item.boxes,
            },
            onProgress: (percent, stage) => {
              const base = (currentIndex - 1) / total;
              const span = 1 / total;
              const overall = 10 + Math.round((base + (Math.max(0, Math.min(100, percent)) / 100) * span) * 80);
              mainWindow?.webContents.send("ml:predict-progress", {
                percent: overall,
                stage,
                currentIndex,
                total,
                imagePath: item.imagePath,
              });
            },
          });
        }

        results.push({
          batchIndex,
          imagePath: item.imagePath,
          filename: item.filename,
          ok: true,
          data,
        });
      } catch (error: any) {
        results.push({
          batchIndex,
          imagePath: item.imagePath,
          filename: item.filename,
          ok: false,
          error: error?.message || "Landmark inference failed",
        });
      }
    }

    mainWindow?.webContents.send("ml:predict-progress", {
      percent: 100,
      stage: "done",
      currentIndex: total,
      total,
    });
    return { ok: true, results };
  } catch (e: any) {
    console.error("Batch prediction failed:", e);
    return { ok: false, error: e.message };
  }
});



ipcMain.handle("select-image-folder", async () => {
  const result = await dialog.showOpenDialog({
    properties: ["openDirectory"],
  });

  if (result.canceled || result.filePaths.length === 0) {
    return { canceled: true };
  }

  const folderPath = result.filePaths[0];

  const imageFiles = fs
    .readdirSync(folderPath)
    .filter((f) => /\.(jpg|jpeg|png|gif|bmp|webp)$/i.test(f));

  // Read file contents and return as base64 for conversion to File objects
  const images = imageFiles.map((filename) => {
    const filePath = path.join(folderPath, filename);
    const data = fs.readFileSync(filePath);
    const ext = path.extname(filename).toLowerCase();
    const mimeTypes: Record<string, string> = {
      '.jpg': 'image/jpeg',
      '.jpeg': 'image/jpeg',
      '.png': 'image/png',
      '.gif': 'image/gif',
      '.bmp': 'image/bmp',
      '.webp': 'image/webp',
    };
    return {
      filename,
      path: filePath,
      data: data.toString('base64'),
      mimeType: mimeTypes[ext] || 'image/jpeg',
    };
  });

  return {
    canceled: false,
    images,
  };
});

// Select individual image files for inference
ipcMain.handle("select-images", async () => {
  const result = await dialog.showOpenDialog({
    properties: ["openFile", "multiSelections"],
    filters: [{ name: "Images", extensions: ["jpg", "jpeg", "png", "bmp", "tiff", "tif"] }],
  });

  if (result.canceled || result.filePaths.length === 0) {
    return { canceled: true };
  }

  const mimeTypes: Record<string, string> = {
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.bmp': 'image/bmp',
    '.tiff': 'image/tiff',
    '.tif': 'image/tiff',
  };

  const files = result.filePaths.map((filePath) => {
    const data = fs.readFileSync(filePath);
    const ext = path.extname(filePath).toLowerCase();
    return {
      path: filePath,
      name: path.basename(filePath),
      data: data.toString('base64'),
      mimeType: mimeTypes[ext] || 'image/jpeg',
    };
  });

  return { canceled: false, files };
});

// Open a folder-picker dialog and return only the selected path (no file reading)
ipcMain.handle("select-folder-path", async () => {
  const result = await dialog.showOpenDialog({ properties: ["openDirectory"] });
  if (result.canceled || !result.filePaths.length) return { canceled: true };
  return { canceled: false, folderPath: result.filePaths[0] };
});

// Open a file-picker dialog filtered to annotation formats (.xml, .json)
ipcMain.handle("select-annotation-file", async () => {
  const result = await dialog.showOpenDialog({
    properties: ["openFile"],
    filters: [{ name: "Annotation files", extensions: ["xml", "json"] }],
  });
  if (result.canceled || !result.filePaths.length) return { canceled: true };
  return { canceled: false, filePath: result.filePaths[0] };
});

// Load a folder of images and an annotation file (XML or JSON) into a session,
// returning images as base64 with pre-populated BoundingBox arrays so the
// frontend can display them in the carousel with landmarks already drawn.
ipcMain.handle(
  "ml:load-annotated-folder",
  async (
    _event,
    args: {
      imageFolderPath: string;
      annotationFilePath: string;
      speciesId: string;
      geometryConfig?: ImportGeometryConfig;
      useSam2BoxDerivation?: boolean;
    }
  ) => {
    try {
      const { imageFolderPath, annotationFilePath, speciesId } = args;
      const importContext = loadSessionOrientationPolicyForCompatibility(speciesId);
      const supplementalWarnings: string[] = [];
      const useSam2BoxDerivation = Boolean(args.useSam2BoxDerivation);
      let canUseSam2BoxDerivation = false;

      if (useSam2BoxDerivation) {
        const sam2Ready = await ensureSam2Ready();
        if (sam2Ready.ok) {
          canUseSam2BoxDerivation = true;
        } else {
          supplementalWarnings.push(
            `SAM2 box derivation was requested but is unavailable: ${sam2Ready.error || "SAM2 is not ready."} Imported boxes were kept as provided.`
          );
        }
      }

      // 1. Parse annotation file based on extension
      const annExt = path.extname(annotationFilePath).toLowerCase();
      let annotationMap: Map<string, AnnotationEntry>;

      if (annExt === ".xml") {
        annotationMap = parseImglabXml(annotationFilePath);
      } else if (annExt === ".json") {
        annotationMap = parseBioVisionJson(annotationFilePath);
      } else {
        return { ok: false, error: `Unsupported annotation format: ${annExt}` };
      }

      // 2. List image files in the folder
      const imageFiles = fs
        .readdirSync(imageFolderPath)
        .filter((f) => IMAGE_EXTS.test(f));

      if (imageFiles.length === 0) {
        return { ok: false, error: "No images found in the selected folder." };
      }

      // 3. Ensure session directories exist
      const sessionDir = getSessionDir(speciesId);
      const imagesDir = path.join(sessionDir, "images");
      const labelsDir = path.join(sessionDir, "labels");
      fs.mkdirSync(imagesDir, { recursive: true });
      fs.mkdirSync(labelsDir, { recursive: true });

      const images: Array<{
        filename: string;
        mimeType: string;
        diskPath: string;
        boxes: any[];
      }> = [];
      const unmatched: string[] = [];
      const detailedWarnings: string[] = [];
      const summary = createImportGeometrySummary();

      for (const filename of imageFiles) {
        const srcPath = path.join(imageFolderPath, filename);
        const diskPath = path.join(imagesDir, filename);
        const imgExt = path.extname(filename).toLowerCase();
        const mimeType = MIME_TYPES[imgExt] || "image/jpeg";

        // Copy image into session (no base64 read Ã¢â‚¬â€ renderer uses localfile:// URLs)
        fs.copyFileSync(srcPath, diskPath);

        const annotationKeys = buildMatchableKeys(filename);
        const annotation = annotationKeys
          .map((key) => annotationMap.get(key))
          .find((entry) => Boolean(entry));

        let boxes: any[] = [];
        if (annotation) {
          const imageDims = getImageDimensions(diskPath) ?? getImageDimensions(srcPath);
          let normalizedBoxes = annotation.boxes
            .map((rawBox, index) => {
              const context = `${filename} box ${index}`;
              const normalizedLandmarks = normalizeLandmarks(rawBox.landmarks ?? [], context);
              return normalizeImportedBoxGeometry(
                rawBox,
                normalizedLandmarks,
                imageDims,
                context,
                importContext.policy,
                importContext.landmarkTemplate,
                detailedWarnings,
                args.geometryConfig,
                summary
              );
            })
            .filter((box): box is ImportedNormalizedBox => Boolean(box));

          if (canUseSam2BoxDerivation && normalizedBoxes.length > 0) {
            const sam2Boxes: ImportedNormalizedBox[] = [];
            for (let index = 0; index < normalizedBoxes.length; index += 1) {
              const normalizedBox = normalizedBoxes[index];
              try {
                const refinedGeometry = await deriveImportedBoxGeometryWithSam2(
                  diskPath,
                  normalizedBox,
                  imageDims,
                  args.geometryConfig
                );
                if (refinedGeometry) {
                  sam2Boxes.push({
                    ...normalizedBox,
                    ...refinedGeometry,
                  });
                } else {
                  detailedWarnings.push(`${filename} box ${index}: SAM2 did not return usable geometry; kept imported box.`);
                  sam2Boxes.push(normalizedBox);
                }
              } catch (sam2Error: any) {
                detailedWarnings.push(
                  `${filename} box ${index}: SAM2 derivation failed (${sam2Error?.message || "unknown error"}); kept imported box.`
                );
                sam2Boxes.push(normalizedBox);
              }
            }
            normalizedBoxes = sam2Boxes;
          }

          boxes = normalizedBoxes.map((box, index) => ({
            id: index,
            left: box.left,
            top: box.top,
            width: box.width,
            height: box.height,
            landmarks: box.landmarks,
            source: "manual" as const,
            ...(box.maskOutline ? { maskOutline: box.maskOutline } : {}),
            ...(box.obbCorners ? { obbCorners: box.obbCorners } : {}),
            ...(Number.isFinite(box.angle) ? { angle: box.angle } : {}),
            ...(Number.isFinite(box.class_id) ? { class_id: box.class_id } : {}),
          }));

          // Persist labels JSON to session so training can read them later
          const labelPath = path.join(
            labelsDir,
            `${path.parse(filename).name}.json`
          );
          fs.writeFileSync(
            labelPath,
            JSON.stringify({ imageFilename: filename, boxes }, null, 2)
          );
        } else {
          unmatched.push(filename);
        }

        images.push({ filename, mimeType, diskPath, boxes });
      }

      // Update session.json image count (non-critical)
      try {
        const sessionJsonPath = path.join(sessionDir, "session.json");
        if (fs.existsSync(sessionJsonPath)) {
          const meta = JSON.parse(fs.readFileSync(sessionJsonPath, "utf-8"));
          meta.imageCount = fs
            .readdirSync(imagesDir)
            .filter((f) => IMAGE_EXTS.test(f)).length;
          meta.lastModified = new Date().toISOString();
          fs.writeFileSync(sessionJsonPath, JSON.stringify(meta, null, 2));
        }
      } catch (_) {
        // non-critical
      }

      const warnings = [...summarizeImportGeometry(summary, unmatched.length), ...supplementalWarnings];
      if (detailedWarnings.length > 0) {
        console.warn("[ml:load-annotated-folder] import details:", detailedWarnings.slice(0, 50));
      }


      return { ok: true, images, unmatched, warnings, importSummary: summary };
    } catch (e: any) {
      console.error("ml:load-annotated-folder failed:", e);
      return { ok: false, error: e.message };
    }
  }
);

// List trained models in the session (or global) models directory
type LandmarkModelStatus = "active" | "deprecated";

type LandmarkModelRegistryEntry = {
  key: string;
  name: string;
  predictorType: "dlib" | "cnn";
  path: string;
  createdAt: string;
  status: LandmarkModelStatus;
};

type LandmarkModelRegistryPayload = {
  version: 1;
  updatedAt: string;
  models: LandmarkModelRegistryEntry[];
};

function getCnnConfigPath(modelsDir: string, name: string): string {
  return path.join(modelsDir, `cnn_${name}_config.json`);
}

function getCnnModelCompatibility(
  effectiveRoot: string,
  name: string
): { compatible: boolean; reason?: string } {
  const modelsDir = path.join(effectiveRoot, "models");
  const configPath = getCnnConfigPath(modelsDir, name);
  if (!fs.existsSync(configPath)) {
    return {
      compatible: false,
      reason: "This CNN model predates the heatmap-head format and must be retrained.",
    };
  }
  try {
    const parsed = JSON.parse(fs.readFileSync(configPath, "utf-8"));
    const headType = String(parsed?.cnn_head_type || "").trim().toLowerCase();
    const hasRequiredKeys =
      parsed &&
      Number.isFinite(Number(parsed.cnn_deconv_layers)) &&
      Number.isFinite(Number(parsed.cnn_deconv_filters)) &&
      Number.isFinite(Number(parsed.cnn_softargmax_beta));
    if (headType !== "heatmap_deconv" || !hasRequiredKeys) {
      return {
        compatible: false,
        reason: "This CNN model predates the heatmap-head format and must be retrained.",
      };
    }
    return { compatible: true };
  } catch {
    return {
      compatible: false,
      reason: "This CNN model predates the heatmap-head format and must be retrained.",
    };
  }
}

function getModelRegistryPath(effectiveRoot: string): string {
  return path.join(effectiveRoot, "models", "model_registry.json");
}

function buildLandmarkModelKey(name: string, predictorType: "dlib" | "cnn"): string {
  return `${name}::${predictorType}`;
}

function scanLandmarkModels(effectiveRoot: string): Array<{
  key: string;
  name: string;
  path: string;
  predictorType: "dlib" | "cnn";
  createdAt: string;
  size: number;
}> {
  const modelsDir = path.join(effectiveRoot, "models");
  if (!fs.existsSync(modelsDir)) {
    fs.mkdirSync(modelsDir, { recursive: true });
    return [];
  }

  const files = fs.readdirSync(modelsDir);
  const dlibModels = files
    .filter((f) => f.endsWith(".dat") && f.startsWith("predictor_"))
    .map((file) => {
      const filePath = path.join(modelsDir, file);
      const stats = fs.statSync(filePath);
      const name = file.replace(/^predictor_/, "").replace(/\.dat$/, "");
      return {
        key: buildLandmarkModelKey(name, "dlib"),
        name,
        path: filePath,
        predictorType: "dlib" as const,
        createdAt: stats.birthtime.toISOString(),
        size: stats.size,
      };
    });

  const cnnModels = files
    .filter((f) => f.endsWith(".pth") && f.startsWith("cnn_"))
    .map((file) => {
      const filePath = path.join(modelsDir, file);
      const stats = fs.statSync(filePath);
      const name = file.replace(/^cnn_/, "").replace(/\.pth$/, "");
      return {
        key: buildLandmarkModelKey(name, "cnn"),
        name,
        path: filePath,
        predictorType: "cnn" as const,
        createdAt: stats.birthtime.toISOString(),
        size: stats.size,
      };
    });

  return [...dlibModels, ...cnnModels];
}

function readLandmarkModelRegistry(effectiveRoot: string): LandmarkModelRegistryPayload {
  const registryPath = getModelRegistryPath(effectiveRoot);
  if (!fs.existsSync(registryPath)) {
    return { version: 1, updatedAt: new Date().toISOString(), models: [] };
  }
  try {
    const parsed = JSON.parse(fs.readFileSync(registryPath, "utf-8"));
    return {
      version: 1,
      updatedAt: String(parsed?.updatedAt || new Date().toISOString()),
      models: Array.isArray(parsed?.models) ? parsed.models : [],
    };
  } catch {
    return { version: 1, updatedAt: new Date().toISOString(), models: [] };
  }
}

function writeLandmarkModelRegistry(effectiveRoot: string, payload: LandmarkModelRegistryPayload): void {
  const registryPath = getModelRegistryPath(effectiveRoot);
  fs.mkdirSync(path.dirname(registryPath), { recursive: true });
  fs.writeFileSync(registryPath, JSON.stringify(payload, null, 2), "utf-8");
}

function syncLandmarkModelRegistry(
  effectiveRoot: string,
  options?: { setActive?: { name: string; predictorType: "dlib" | "cnn" } | null }
): LandmarkModelRegistryPayload {
  const scanned = scanLandmarkModels(effectiveRoot);
  const existing = readLandmarkModelRegistry(effectiveRoot);
  const existingByKey = new Map(existing.models.map((entry) => [entry.key, entry]));
  const preferredActiveKey = options?.setActive
    ? buildLandmarkModelKey(options.setActive.name, options.setActive.predictorType)
    : null;

  const nextModels: LandmarkModelRegistryEntry[] = scanned.map((model) => {
    const existingEntry = existingByKey.get(model.key);
    return {
      key: model.key,
      name: model.name,
      predictorType: model.predictorType,
      path: model.path,
      createdAt: model.createdAt,
      status: existingEntry?.status ?? "deprecated",
    };
  });

  (["dlib", "cnn"] as const).forEach((predictorType) => {
    const sameType = nextModels
      .filter((entry) => entry.predictorType === predictorType)
      .sort((a, b) => Date.parse(b.createdAt) - Date.parse(a.createdAt));
    if (sameType.length === 0) return;

    let activeKey = preferredActiveKey;
    if (!activeKey || !sameType.some((entry) => entry.key === activeKey)) {
      const existingActive = sameType.find((entry) => entry.status === "active");
      activeKey = existingActive?.key ?? sameType[0].key;
    }

    sameType.forEach((entry) => {
      entry.status = entry.key === activeKey ? "active" : "deprecated";
    });
  });

  const payload: LandmarkModelRegistryPayload = {
    version: 1,
    updatedAt: new Date().toISOString(),
    models: nextModels,
  };
  writeLandmarkModelRegistry(effectiveRoot, payload);
  return payload;
}

function resolveModelStatus(
  registry: LandmarkModelRegistryPayload,
  name: string,
  predictorType: "dlib" | "cnn"
): LandmarkModelStatus {
  return (
    registry.models.find((entry) => entry.key === buildLandmarkModelKey(name, predictorType))?.status ??
    "active"
  );
}

ipcMain.handle("ml:list-models", async (_event, input?: string | {
  speciesId?: string;
  activeOnly?: boolean;
  includeDeprecated?: boolean;
}) => {
  try {
    const speciesId = typeof input === "string" ? input : input?.speciesId;
    const activeOnly = typeof input === "object" && input?.activeOnly === true;
    const includeDeprecated =
      typeof input === "object" ? input.includeDeprecated !== false : true;
    const effectiveRoot = getEffectiveRoot(speciesId);
    const modelsDir = path.join(effectiveRoot, "models");

    if (!fs.existsSync(modelsDir)) {
      fs.mkdirSync(modelsDir, { recursive: true });
      return { ok: true, models: [] };
    }

    const registry = syncLandmarkModelRegistry(effectiveRoot);
    const scannedLandmarkModels = scanLandmarkModels(effectiveRoot).map((model) => ({
      ...model,
      createdAt: new Date(model.createdAt),
      status: resolveModelStatus(registry, model.name, model.predictorType),
      ...(model.predictorType === "cnn"
        ? getCnnModelCompatibility(effectiveRoot, model.name)
        : { compatible: true as const }),
    }));

    const files = fs.readdirSync(modelsDir);
    const yoloPoseModels = files
      .filter((f) => {
        const lower = f.toLowerCase();
        return (
          (lower.endsWith(".pt") && lower.startsWith("pose_")) ||
          (lower.endsWith(".pt") && lower.startsWith("yolo_pose_"))
        );
      })
      .map((file) => {
        const filePath = path.join(modelsDir, file);
        const stats = fs.statSync(filePath);
        const tag = file
          .replace(/^pose_/, "")
          .replace(/^yolo_pose_/, "")
          .replace(/\.pt$/, "");
        return {
          name: tag,
          path: filePath,
          size: stats.size,
          createdAt: stats.birthtime,
          predictorType: "yolo_pose" as const,
          status: "active" as const,
        };
      });

    const landmarkModels = scannedLandmarkModels.filter((model) => {
      if (activeOnly && model.predictorType === "cnn" && model.compatible === false) return false;
      if (!includeDeprecated && model.status === "deprecated") return false;
      if (activeOnly && model.status !== "active") return false;
      return true;
    });

    const poseModels = activeOnly
      ? yoloPoseModels
      : yoloPoseModels;

    const models = [...landmarkModels, ...poseModels]
      .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());

    return { ok: true, models };
  } catch (e: any) {
    console.error("Failed to list models:", e);
    return { ok: false, error: e.message };
  }
});

// Delete a trained model
ipcMain.handle(
  "ml:delete-model",
  async (
    _event,
    modelName: string,
    speciesId?: string,
    predictorType?: "dlib" | "cnn" | "yolo_pose"
  ) => {
  try {
    const modelsDir = path.join(getEffectiveRoot(speciesId), "models");
    const dlibPath = path.join(modelsDir, `predictor_${modelName}.dat`);
    const cnnPath = path.join(modelsDir, `cnn_${modelName}.pth`);
    const cnnConfigPath = path.join(modelsDir, `cnn_${modelName}_config.json`);
    const posePath = path.join(modelsDir, `pose_${modelName}.pt`);
    const yoloPosePath = path.join(modelsDir, `yolo_pose_${modelName}.pt`);
    const yoloAliasPath = path.join(modelsDir, `yolo_${modelName}.pt`);

    const dlibExists = fs.existsSync(dlibPath);
    const cnnExists = fs.existsSync(cnnPath);
    const poseExists = fs.existsSync(posePath);
    const yoloPoseExists = fs.existsSync(yoloPosePath);
    const yoloAliasExists = fs.existsSync(yoloAliasPath);

    if (predictorType === "dlib") {
      if (!dlibExists) return { ok: false, error: "dlib model not found" };
      fs.unlinkSync(dlibPath);
      syncLandmarkModelRegistry(getEffectiveRoot(speciesId));
      return { ok: true };
    }
    if (predictorType === "cnn") {
      if (!cnnExists) return { ok: false, error: "CNN model not found" };
      fs.unlinkSync(cnnPath);
      if (fs.existsSync(cnnConfigPath)) fs.unlinkSync(cnnConfigPath);
      syncLandmarkModelRegistry(getEffectiveRoot(speciesId));
      return { ok: true };
    }
    if (predictorType === "yolo_pose") {
      let removed = false;
      if (poseExists) {
        fs.unlinkSync(posePath);
        removed = true;
      }
      if (yoloPoseExists) {
        fs.unlinkSync(yoloPosePath);
        removed = true;
      }
      if (yoloAliasExists) {
        fs.unlinkSync(yoloAliasPath);
        removed = true;
      }
      // Remove versioned checkpoints for this class/tag.
      fs.readdirSync(modelsDir)
        .filter((f) => {
          const lower = f.toLowerCase();
          const prefix = `yolo_${modelName.toLowerCase()}_v`;
          return lower.startsWith(prefix) && lower.endsWith(".pt");
        })
        .forEach((f) => {
          fs.unlinkSync(path.join(modelsDir, f));
          removed = true;
        });
      if (!removed) return { ok: false, error: "YOLO-pose model not found" };
      return { ok: true };
    }

    // Backward-compatible behavior: if predictor type is not specified,
    // delete both variants for this model tag.
    if (!dlibExists && !cnnExists && !poseExists && !yoloPoseExists && !yoloAliasExists) return { ok: false, error: "Model not found" };
    if (dlibExists) fs.unlinkSync(dlibPath);
    if (cnnExists) {
      fs.unlinkSync(cnnPath);
      if (fs.existsSync(cnnConfigPath)) fs.unlinkSync(cnnConfigPath);
    }
    if (poseExists) fs.unlinkSync(posePath);
    if (yoloPoseExists) fs.unlinkSync(yoloPosePath);
    if (yoloAliasExists) fs.unlinkSync(yoloAliasPath);
    syncLandmarkModelRegistry(getEffectiveRoot(speciesId));
    return { ok: true };
  } catch (e: any) {
    console.error("Failed to delete model:", e);
    return { ok: false, error: e.message };
  }
});

// Rename a trained model
ipcMain.handle(
  "ml:rename-model",
  async (
    _event,
    oldName: string,
    newName: string,
    speciesId?: string,
    predictorType?: "dlib" | "cnn" | "yolo_pose"
  ) => {
  try {
    const modelsDir = path.join(getEffectiveRoot(speciesId), "models");
    const effectiveRoot = getEffectiveRoot(speciesId);
    const priorRegistry = syncLandmarkModelRegistry(effectiveRoot);
    const priorStatus = predictorType === "dlib" || predictorType === "cnn"
      ? priorRegistry.models.find((entry) => entry.key === buildLandmarkModelKey(oldName, predictorType))?.status
      : undefined;
    const oldDlib = path.join(modelsDir, `predictor_${oldName}.dat`);
    const newDlib = path.join(modelsDir, `predictor_${newName}.dat`);
    const oldCnn = path.join(modelsDir, `cnn_${oldName}.pth`);
    const newCnn = path.join(modelsDir, `cnn_${newName}.pth`);
    const oldCnnCfg = path.join(modelsDir, `cnn_${oldName}_config.json`);
    const newCnnCfg = path.join(modelsDir, `cnn_${newName}_config.json`);
    const oldPose = path.join(modelsDir, `pose_${oldName}.pt`);
    const newPose = path.join(modelsDir, `pose_${newName}.pt`);
    const oldYoloPose = path.join(modelsDir, `yolo_pose_${oldName}.pt`);
    const newYoloPose = path.join(modelsDir, `yolo_pose_${newName}.pt`);

    const isDlib = fs.existsSync(oldDlib);
    const isCnn = fs.existsSync(oldCnn);
    const isPose = fs.existsSync(oldPose) || fs.existsSync(oldYoloPose);

    if (predictorType === "dlib") {
      if (!isDlib) return { ok: false, error: "dlib model not found" };
      if (fs.existsSync(newDlib)) return { ok: false, error: "A dlib model with that name already exists" };
      fs.renameSync(oldDlib, newDlib);
      syncLandmarkModelRegistry(effectiveRoot, {
        setActive: priorStatus === "active" ? { name: newName, predictorType: "dlib" } : null,
      });
      return { ok: true };
    }
    if (predictorType === "cnn") {
      if (!isCnn) return { ok: false, error: "CNN model not found" };
      if (fs.existsSync(newCnn)) return { ok: false, error: "A CNN model with that name already exists" };
      fs.renameSync(oldCnn, newCnn);
      if (fs.existsSync(oldCnnCfg)) fs.renameSync(oldCnnCfg, newCnnCfg);
      syncLandmarkModelRegistry(effectiveRoot, {
        setActive: priorStatus === "active" ? { name: newName, predictorType: "cnn" } : null,
      });
      return { ok: true };
    }
    if (predictorType === "yolo_pose") {
      if (!isPose) return { ok: false, error: "YOLO-pose model not found" };
      if (fs.existsSync(newPose) || fs.existsSync(newYoloPose)) {
        return { ok: false, error: "A YOLO-pose model with that name already exists" };
      }
      if (fs.existsSync(oldPose)) fs.renameSync(oldPose, newPose);
      if (fs.existsSync(oldYoloPose)) fs.renameSync(oldYoloPose, newYoloPose);
      return { ok: true };
    }

    if (!isDlib && !isCnn && !isPose) return { ok: false, error: "Model not found" };
    if (isDlib && fs.existsSync(newDlib)) return { ok: false, error: "A model with that name already exists" };
    if (isCnn && fs.existsSync(newCnn)) return { ok: false, error: "A model with that name already exists" };

    if (isDlib) fs.renameSync(oldDlib, newDlib);
    if (isCnn) {
      fs.renameSync(oldCnn, newCnn);
      if (fs.existsSync(oldCnnCfg)) fs.renameSync(oldCnnCfg, newCnnCfg);
    }
    if (fs.existsSync(oldPose)) fs.renameSync(oldPose, newPose);
    if (fs.existsSync(oldYoloPose)) fs.renameSync(oldYoloPose, newYoloPose);
    syncLandmarkModelRegistry(effectiveRoot, {
      setActive:
        priorStatus === "active" && (predictorType === "dlib" || predictorType === "cnn")
          ? { name: newName, predictorType }
          : null,
    });
    return { ok: true };
  } catch (e: any) {
    console.error("Failed to rename model:", e);
    return { ok: false, error: e.message };
  }
});

// Get info about a specific model
ipcMain.handle("ml:get-model-info", async (_event, modelName: string, speciesId?: string) => {
  try {
    const modelsDir = path.join(getEffectiveRoot(speciesId), "models");
    const dlibPath = path.join(modelsDir, `predictor_${modelName}.dat`);
    const cnnPath = path.join(modelsDir, `cnn_${modelName}.pth`);
    const posePath = path.join(modelsDir, `pose_${modelName}.pt`);
    const yoloPosePath = path.join(modelsDir, `yolo_pose_${modelName}.pt`);

    if (fs.existsSync(dlibPath)) {
      const stats = fs.statSync(dlibPath);
      return {
        ok: true,
        model: { name: modelName, path: dlibPath, size: stats.size, createdAt: stats.birthtime, predictorType: "dlib" },
      };
    }
    if (fs.existsSync(cnnPath)) {
      const stats = fs.statSync(cnnPath);
      return {
        ok: true,
        model: { name: modelName, path: cnnPath, size: stats.size, createdAt: stats.birthtime, predictorType: "cnn" },
      };
    }
    if (fs.existsSync(posePath)) {
      const stats = fs.statSync(posePath);
      return {
        ok: true,
        model: { name: modelName, path: posePath, size: stats.size, createdAt: stats.birthtime, predictorType: "yolo_pose" },
      };
    }
    if (fs.existsSync(yoloPosePath)) {
      const stats = fs.statSync(yoloPosePath);
      return {
        ok: true,
        model: { name: modelName, path: yoloPosePath, size: stats.size, createdAt: stats.birthtime, predictorType: "yolo_pose" },
      };
    }
    return { ok: false, error: "Model not found" };
  } catch (e: any) {
    console.error("Failed to get model info:", e);
    return { ok: false, error: e.message };
  }
});

ipcMain.handle(
  "ml:check-model-compatibility",
  async (
    _event,
    args: {
      speciesId?: string;
      modelName: string;
      predictorType?: "dlib" | "cnn";
      includeRuntime?: boolean;
    }
  ) => {
    try {
      if (!args?.modelName) {
        return {
          ok: false,
          compatible: false,
          blocking: true,
          requiresOverride: true,
          issues: [
            {
              code: "missing_model_name",
              severity: "error",
              message: "modelName is required.",
            },
          ],
          error: "modelName is required.",
        } as ModelCompatibilityResult;
      }
      const predictorType = args.predictorType === "cnn" ? "cnn" : "dlib";
      return await evaluateModelCompatibility({
        speciesId: args.speciesId,
        modelName: args.modelName,
        predictorType,
        includeRuntime: args.includeRuntime ?? true,
      });
    } catch (e: any) {
      return {
        ok: false,
        compatible: false,
        blocking: true,
        requiresOverride: true,
        issues: [
          {
            code: "compatibility_check_failed",
            severity: "error",
            message: e?.message || "Compatibility check failed.",
          },
        ],
        error: e?.message || "Compatibility check failed.",
      } as ModelCompatibilityResult;
    }
  }
);

interface DetectionOptions {
  speciesId?: string;
}

ipcMain.handle("ml:detect-specimens", async (_event, imagePath: string, options?: DetectionOptions) => {
  try {
    const speciesId = options?.speciesId;
    const effectiveRoot = getEffectiveRoot(speciesId);

    const sessionObbPath = path.join(effectiveRoot, "models", "session_obb_detector.pt");
    if (fs.existsSync(sessionObbPath)) {
      // Route all trained-session detection through the shared OBB detector script.
      const pythonArgs = [
        path.join(__dirname, "../backend/detection/detect_specimen.py"),
        imagePath,
        "--multi",
        "--yolo-model",
        sessionObbPath,
      ];

      const out = await runPython(pythonArgs);
      const data = JSON.parse(out.trim());
      if (!data) {
        return { ok: false, error: "No detection result", boxes: [] };
      }
      if (data.ok === true && Array.isArray(data.boxes)) {
        return data;
      }
      return { ok: false, error: "Unexpected detection format", boxes: [] };
    }

    if (!superAnnotator.isRunning) {
      await superAnnotator.send({ cmd: "init" });
    }

    const sessionJsonPath = path.join(effectiveRoot, "session.json");
    const sessionMeta = safeReadJson(sessionJsonPath) || {};
    const classPrompt =
      String(sessionMeta?.name || "").trim() ||
      String(speciesId || "").trim().replace(/[-_]+/g, " ") ||
      "specimen";

    const result = await superAnnotator.send({
      cmd: "annotate",
      image_path: imagePath,
      class_name: classPrompt,
      options: {
        conf_threshold: 0.3,
        sam_enabled: false,
        max_objects: 20,
        detection_mode: "auto",
        detection_preset: "balanced",
        orientation_policy: sessionMeta?.orientationPolicy ?? undefined,
      },
    });

    if (result?.status === "result" && Array.isArray(result.objects)) {
      const boxes = result.objects.map((obj: any) => ({
        ...obj.box,
        ...(obj.obb?.corners ? { obbCorners: obj.obb.corners } : {}),
        ...(typeof obj.obb?.angle === "number" ? { angle: obj.obb.angle } : {}),
      }));
      return {
        ok: true,
        boxes,
        num_detections: boxes.length,
        detection_method: result.detection_method ?? "yolo_world",
        fallback: result.detection_method !== "yolo_obb",
      };
    }

    return { ok: false, error: result?.error || "Zero-shot detection failed", boxes: [] };
  } catch (e: any) {
    console.error("Specimen detection failed:", e);
    return { ok: false, error: e.message, boxes: [] };
  }
});

// Check OBB detector runtime availability
ipcMain.handle("ml:check-yolo", async () => {
  try {
    const out = await runPython([
      path.join(__dirname, "../backend/detection/detect_specimen.py"),
      "--check",
    ]);

    return JSON.parse(out.trim());
  } catch (e: any) {
    return { available: false, primary_method: "yolo_obb", error: e.message };
  }
});

// Ã¢â€â‚¬Ã¢â€â‚¬ Session management IPC handlers Ã¢â€â‚¬Ã¢â€â‚¬

function getSessionDir(speciesId: string): string {
  return path.join(getSessionsRoot(), sanitizeSpeciesId(speciesId));
}

function getFinalizedImagesPath(sessionDir: string): string {
  return path.join(sessionDir, "finalized_images.json");
}

function readFinalizedList(sessionDir: string): string[] {
  const finalizedListPath = getFinalizedImagesPath(sessionDir);
  if (!fs.existsSync(finalizedListPath)) return [];
  try {
    const parsed = JSON.parse(fs.readFileSync(finalizedListPath, "utf-8"));
    if (!Array.isArray(parsed)) return [];
    return parsed
      .map((v) => (typeof v === "string" ? v.trim() : ""))
      .filter((v) => v.length > 0);
  } catch {
    return [];
  }
}

function writeFinalizedList(sessionDir: string, names: string[]): void {
  const finalizedListPath = getFinalizedImagesPath(sessionDir);
  const seen = new Set<string>();
  const deduped: string[] = [];
  (names || []).forEach((name) => {
    const safe = String(name || "").trim();
    if (!safe) return;
    const key = path.basename(safe).toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    deduped.push(safe);
  });
  fs.writeFileSync(finalizedListPath, JSON.stringify(deduped));
}

function removeFinalizedFromLabel(labelPath: string): { hadFinalizedDetection: boolean } {
  if (!fs.existsSync(labelPath)) {
    return { hadFinalizedDetection: false };
  }
  try {
    const payload = JSON.parse(fs.readFileSync(labelPath, "utf-8"));
    if (!payload || typeof payload !== "object") {
      return { hadFinalizedDetection: false };
    }
    const hadFinalizedDetection = Boolean(
      payload.finalizedDetection && typeof payload.finalizedDetection === "object"
    );
    if (hadFinalizedDetection) {
      delete payload.finalizedDetection;
      fs.writeFileSync(labelPath, JSON.stringify(payload, null, 2));
    }
    return { hadFinalizedDetection };
  } catch {
    return { hadFinalizedDetection: false };
  }
}

function normalizeComparablePath(value?: string): string {
  const raw = String(value || "").trim();
  if (!raw) return "";
  try {
    return path.resolve(raw).replace(/\\/g, "/").toLowerCase();
  } catch {
    return raw.replace(/\\/g, "/").toLowerCase();
  }
}

function deleteSegmentsForImage(
  sessionDir: string,
  filename: string,
  imagePath?: string
): { removed: number } {
  const segmentsDir = path.join(sessionDir, "segments");
  if (!fs.existsSync(segmentsDir)) return { removed: 0 };

  const targetFilename = path.basename(String(filename || "")).toLowerCase();
  const targetPath = normalizeComparablePath(imagePath);
  const segmentFiles = fs.readdirSync(segmentsDir);
  let removed = 0;

  for (const file of segmentFiles) {
    if (!file.endsWith("_meta.json")) continue;
    const metaPath = path.join(segmentsDir, file);
    let sourceImage = "";
    try {
      const meta = JSON.parse(fs.readFileSync(metaPath, "utf-8"));
      sourceImage =
        typeof meta?.source_image === "string" ? String(meta.source_image) : "";
    } catch {
      continue;
    }

    const sourceFilename = path.basename(sourceImage).toLowerCase();
    const sourcePath = normalizeComparablePath(sourceImage);
    const matchesFilename = sourceFilename && sourceFilename === targetFilename;
    const matchesPath = Boolean(targetPath && sourcePath && sourcePath === targetPath);
    if (!matchesFilename && !matchesPath) continue;

    const stem = file.replace(/_meta\.json$/i, "");
    for (const segFile of segmentFiles) {
      if (segFile === file || segFile.startsWith(`${stem}_`)) {
        try {
          fs.unlinkSync(path.join(segmentsDir, segFile));
        } catch {
          // best effort cleanup
        }
      }
    }
    removed += 1;
  }

  return { removed };
}

function clearFinalizedSegmentCache(speciesId: string, filename: string): void {
  const safeFilename = path.basename(String(filename || "")).toLowerCase();
  const prefix = `${speciesId}::`;
  for (const key of Array.from(finalizedSegmentSignatureCache.keys())) {
    if (!key.startsWith(prefix)) continue;
    const keyFilename = path.basename(key.slice(prefix.length)).toLowerCase();
    if (keyFilename === safeFilename) {
      finalizedSegmentSignatureCache.delete(key);
    }
  }
}

function cancelQueuedSegmentSave(speciesId: string, filename: string): void {
  const safeFilename = path.basename(String(filename || "")).toLowerCase();
  for (const [sessionKey, queue] of segmentSaveQueues.entries()) {
    const nextQueue = queue.filter(
      (job) => !(
        job.speciesId === speciesId &&
        path.basename(job.filename).toLowerCase() === safeFilename
      )
    );
    if (nextQueue.length === 0) {
      segmentSaveQueues.delete(sessionKey);
    } else if (nextQueue.length !== queue.length) {
      segmentSaveQueues.set(sessionKey, nextQueue);
    }
  }
  setSegmentQueueStatus(speciesId, filename, "idle");
}

function cancelAllQueuedSegmentSavesForSpecies(speciesId: string): void {
  for (const [sessionKey, queue] of segmentSaveQueues.entries()) {
    const nextQueue = queue.filter((job) => job.speciesId !== speciesId);
    if (nextQueue.length === 0) {
      segmentSaveQueues.delete(sessionKey);
    } else if (nextQueue.length !== queue.length) {
      segmentSaveQueues.set(sessionKey, nextQueue);
    }
  }
  for (const key of Array.from(segmentQueueStatusByImage.keys())) {
    if (key.startsWith(`${speciesId}::`)) {
      segmentQueueStatusByImage.delete(key);
    }
  }
  for (const key of Array.from(finalizedSegmentSignatureCache.keys())) {
    if (key.startsWith(`${speciesId}::`)) {
      finalizedSegmentSignatureCache.delete(key);
    }
  }
}

async function processSegmentSaveQueue(sessionKey: string): Promise<void> {
  if (segmentQueueRunningSessions.has(sessionKey)) return;
  segmentQueueRunningSessions.add(sessionKey);
  try {
    while (true) {
      const queue = segmentSaveQueues.get(sessionKey) || [];
      const job = queue[0];
      if (!job) {
        segmentSaveQueues.delete(sessionKey);
        break;
      }

      const latestSignature = finalizedSegmentSignatureCache.get(job.queueKey);
      if (latestSignature !== job.signature) {
        queue.shift();
        if (queue.length === 0) segmentSaveQueues.delete(sessionKey);
        else segmentSaveQueues.set(sessionKey, queue);
        continue;
      }

      if (!job.imagePath || job.acceptedBoxes.length === 0) {
        finalizedSegmentSignatureCache.delete(job.queueKey);
        setSegmentQueueStatus(
          job.speciesId,
          job.filename,
          "skipped",
          job.signature,
          "no_boxes_or_image",
          { expectedCount: job.acceptedBoxes.length, savedCount: 0 }
        );
        queue.shift();
        if (queue.length === 0) segmentSaveQueues.delete(sessionKey);
        else segmentSaveQueues.set(sessionKey, queue);
        continue;
      }

      // Auto-reinit if the idle timer killed the process between annotation and finalize.
      // save_segments_for_boxes has its own fallback chain (cache → SAM2 → rectangle),
      // so we don't gate on sam2_ready here — let Python decide.
      const iterativeSam2Eligible = await checkSam2MinimumRequirements();
      if (!iterativeSam2Eligible.ok) {
        finalizedSegmentSignatureCache.delete(job.queueKey);
        setSegmentQueueStatus(
          job.speciesId,
          job.filename,
          "failed",
          job.signature,
          iterativeSam2Eligible.error || "sam2_min_requirements_not_met",
          { expectedCount: job.acceptedBoxes.length, savedCount: 0 }
        );
        queue.shift();
        if (queue.length === 0) segmentSaveQueues.delete(sessionKey);
        else segmentSaveQueues.set(sessionKey, queue);
        continue;
      }
      if (iterativeSam2Eligible.ok && !superAnnotator.initCompleted) {
        try {
          const initRes = await superAnnotator.send({ cmd: "init" });
          if (initRes?.status !== "error") superAnnotator.initCompleted = true;
          else { break; }
        } catch {
          break;  // process truly unreachable
        }
      }

      try {
        // Verify process is alive; log SAM2 state but do not gate on sam2_ready.
        // The Python save_segments_for_boxes function handles: cached masks (no SAM2
        // needed), fresh SAM2 inference, and rectangle fallback — in that order.
        await superAnnotator.send({ cmd: "check" });
      } catch (error: any) {
        finalizedSegmentSignatureCache.delete(job.queueKey);
        setSegmentQueueStatus(
          job.speciesId,
          job.filename,
          "failed",
          job.signature,
          error?.message || "sam2_check_failed",
          { expectedCount: job.acceptedBoxes.length, savedCount: 0 }
        );
        queue.shift();
        if (queue.length === 0) segmentSaveQueues.delete(sessionKey);
        else segmentSaveQueues.set(sessionKey, queue);
        continue;
      }

      setSegmentQueueStatus(
        job.speciesId,
        job.filename,
        "running",
        job.signature,
        undefined,
        { expectedCount: job.acceptedBoxes.length, savedCount: 0 }
      );
      try {
        const xyxyBoxes = job.acceptedBoxes.map((b) => [
          b.left,
          b.top,
          b.left + b.width,
          b.top + b.height,
        ]);
        const saveResult = await superAnnotator.send({
          cmd: "save_segments_for_boxes",
          image_path: job.imagePath,
          boxes: xyxyBoxes,
          session_dir: job.sessionDir,
          iterative: iterativeSam2Eligible.ok,
          expand_ratio: 0.10,
        });

        const currentSignature = finalizedSegmentSignatureCache.get(job.queueKey);
        if (saveResult?.status === "error") {
          throw new Error(saveResult.error || "segment_save_failed");
        }
        const savedCount = Number(saveResult?.saved ?? 0);
        const expectedCount = job.acceptedBoxes.length;
        if (savedCount < expectedCount && currentSignature === job.signature) {
          finalizedSegmentSignatureCache.delete(job.queueKey);
          setSegmentQueueStatus(
            job.speciesId,
            job.filename,
            "failed",
            job.signature,
            `partial_segment_save:${savedCount}/${expectedCount}`,
            { expectedCount, savedCount }
          );
        } else if (currentSignature === job.signature) {
          persistFinalizedDetection(
            job.sessionDir,
            job.speciesId,
            job.filename,
            job.acceptedBoxes,
            job.signature
          );
          setSegmentQueueStatus(
            job.speciesId,
            job.filename,
            "saved",
            job.signature,
            undefined,
            { expectedCount, savedCount }
          );
        } else {
          setSegmentQueueStatus(job.speciesId, job.filename, "idle");
        }
      } catch (error: any) {
        const currentSignature = finalizedSegmentSignatureCache.get(job.queueKey);
        if (currentSignature === job.signature) {
          finalizedSegmentSignatureCache.delete(job.queueKey);
          setSegmentQueueStatus(
            job.speciesId,
            job.filename,
            "failed",
            job.signature,
            error?.message || "segment_save_failed",
            { expectedCount: job.acceptedBoxes.length, savedCount: 0 }
          );
        } else {
          setSegmentQueueStatus(job.speciesId, job.filename, "idle");
        }
      } finally {
        queue.shift();
        if (queue.length === 0) segmentSaveQueues.delete(sessionKey);
        else segmentSaveQueues.set(sessionKey, queue);
      }
    }
  } finally {
    segmentQueueRunningSessions.delete(sessionKey);
  }
}

function enqueueSegmentSave(job: SegmentQueueJob): { queued: boolean; state: "queued" | "skipped" } {
  if (!job.imagePath || job.acceptedBoxes.length === 0) {
    setSegmentQueueStatus(
      job.speciesId,
      job.filename,
      "skipped",
      job.signature,
      "no_boxes_or_image",
      { expectedCount: job.acceptedBoxes.length, savedCount: 0 }
    );
    return { queued: false, state: "skipped" };
  }

  const queue = segmentSaveQueues.get(job.sessionKey) || [];
  const duplicate = queue.some((queued) => queued.queueKey === job.queueKey && queued.signature === job.signature);
  if (!duplicate) {
    const nextQueue = queue.filter((queued) => queued.queueKey !== job.queueKey);
    nextQueue.push(job);
    segmentSaveQueues.set(job.sessionKey, nextQueue);
  }
  setSegmentQueueStatus(
    job.speciesId,
    job.filename,
    "queued",
    job.signature,
    undefined,
    { expectedCount: job.acceptedBoxes.length, savedCount: 0 }
  );
  void processSegmentSaveQueue(job.sessionKey);
  return { queued: true, state: "queued" };
}

function kickAllSegmentSaveQueues(): void {
  for (const sessionKey of segmentSaveQueues.keys()) {
    void processSegmentSaveQueue(sessionKey);
  }
}

function unfinalizeImageInSession(
  speciesId: string,
  filename: string,
  imagePath?: string
): {
  ok: boolean;
  filename: string;
  removedFromList?: boolean;
  removedSegments?: number;
  hadFinalizedDetection?: boolean;
  error?: string;
} {
  const sessionDir = getSessionDir(speciesId);
  const safeFilename = path.basename(String(filename || "").trim());
  if (!safeFilename) {
    return { ok: false, filename: "", error: "Invalid filename" };
  }

  const safeLower = safeFilename.toLowerCase();
  const finalizedNames = readFinalizedList(sessionDir);
  const retainedNames = finalizedNames.filter(
    (entry) => path.basename(String(entry || "")).toLowerCase() !== safeLower
  );
  const removedFromList = retainedNames.length !== finalizedNames.length;
  if (removedFromList) {
    writeFinalizedList(sessionDir, retainedNames);
  }

  const labelPath = path.join(
    sessionDir,
    "labels",
    safeFilename.replace(/\.\w+$/, ".json")
  );
  const { hadFinalizedDetection } = removeFinalizedFromLabel(labelPath);
  const { removed } = deleteSegmentsForImage(sessionDir, safeFilename, imagePath);

  clearFinalizedSegmentCache(speciesId, safeFilename);
  cancelQueuedSegmentSave(speciesId, safeFilename);

  return {
    ok: true,
    filename: safeFilename,
    removedFromList,
    removedSegments: removed,
    hadFinalizedDetection,
  };
}

const INFERENCE_REVIEW_DRAFTS_FILE = "inference_review_drafts.json";
const RETRAIN_QUEUE_FILE = "retrain_queue.json";
const INFERENCE_SESSIONS_DIR = "inference_sessions";
const INFERENCE_SESSION_MANIFEST_FILE = "manifest.json";
const INFERENCE_SESSION_INDEX_FILE = "session_index.json";
const CANONICAL_INFERENCE_SESSION_ID = "default";

type InferenceDraftSpecimen = {
  box: {
    left: number;
    top: number;
    width: number;
    height: number;
    confidence?: number;
    class_id?: number;
    class_name?: string;
    obbCorners?: [number, number][];
    angle?: number;
    orientation_override?: "left" | "right" | "uncertain";
    orientation_hint?: {
      orientation?: "left" | "right";
      confidence?: number;
      source?: string;
      head_point?: [number, number];
      tail_point?: [number, number];
    };
  };
  landmarks: { id: number; x: number; y: number }[];
};

type InferenceReviewDraftItem = {
  key: string;
  imagePath: string;
  filename: string;
  specimens: InferenceDraftSpecimen[];
  edited: boolean;
  saved: boolean;
  reviewComplete?: boolean;
  committedAt?: string | null;
  landmarkModelKey?: string;
  landmarkPredictorType?: "dlib" | "cnn" | "yolo_pose";
  boxSignature?: string;
  inferenceSignature?: string;
  updatedAt: string;
};

type InferenceReviewDraftsPayload = {
  version: 1;
  updatedAt: string;
  items: Record<string, InferenceReviewDraftItem>;
};

type RetrainQueueItem = {
  key: string;
  speciesId: string;
  inferenceSessionId?: string;
  landmarkModelKey?: string;
  landmarkModelName?: string;
  landmarkPredictorType?: "dlib" | "cnn" | "yolo_pose";
  detectionModelKey?: string;
  detectionModelName?: string;
  filename: string;
  imagePath?: string;
  source: string;
  boxesCount: number;
  landmarksCount: number;
  queuedAt: string;
  updatedAt: string;
};

type RetrainQueuePayload = {
  version: 1;
  updatedAt: string;
  items: Record<string, RetrainQueueItem>;
};

type InferenceSessionManifest = {
  version: 1;
  sessionId: string;
  speciesId: string;
  displayName?: string;
  models: {
    landmark: {
      key: string;
      name?: string;
      predictorType?: "dlib" | "cnn" | "yolo_pose";
    };
    detection: {
      key: string;
      name?: string;
    };
  };
  preferences?: {
    lastUsedLandmarkModelKey?: string;
    lastUsedPredictorType?: "dlib" | "cnn" | "yolo_pose";
    detectionModelKey?: string;
    detectionModelName?: string;
  };
  createdAt: string;
  updatedAt: string;
};

type InferenceSessionIndex = {
  version: 1;
  canonicalSessionId: string;
  migratedFromSessionId?: string;
  updatedAt: string;
};

function sanitizeInferenceSessionId(raw: string): string {
  return String(raw || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 120);
}

function buildInferenceSessionId(
  speciesId: string,
  landmarkModelKey: string,
  detectionModelKey?: string
): string {
  const schema = sanitizeInferenceSessionId(speciesId) || "schema";
  const landmark = sanitizeInferenceSessionId(landmarkModelKey) || "landmark_default";
  const detection = sanitizeInferenceSessionId(detectionModelKey || "session_detection_default");
  return `${schema}__lm_${landmark}__det_${detection}`;
}

function getInferenceSessionDir(speciesId: string, inferenceSessionId: string): string {
  return path.join(
    getSessionDir(speciesId),
    INFERENCE_SESSIONS_DIR,
    sanitizeInferenceSessionId(inferenceSessionId)
  );
}

function getInferenceSessionsRoot(speciesId: string): string {
  return path.join(getSessionDir(speciesId), INFERENCE_SESSIONS_DIR);
}

function getInferenceSessionManifestPath(speciesId: string, inferenceSessionId: string): string {
  return path.join(
    getInferenceSessionDir(speciesId, inferenceSessionId),
    INFERENCE_SESSION_MANIFEST_FILE
  );
}

function getInferenceSessionIndexPath(speciesId: string): string {
  return path.join(getInferenceSessionsRoot(speciesId), INFERENCE_SESSION_INDEX_FILE);
}

function readInferenceSessionManifest(
  speciesId: string,
  inferenceSessionId: string
): InferenceSessionManifest | null {
  const manifestPath = getInferenceSessionManifestPath(speciesId, inferenceSessionId);
  if (!fs.existsSync(manifestPath)) return null;
  try {
    const parsed = JSON.parse(fs.readFileSync(manifestPath, "utf-8"));
    if (!parsed || typeof parsed !== "object") return null;
    return parsed as InferenceSessionManifest;
  } catch {
    return null;
  }
}

function readInferenceSessionIndex(speciesId: string): InferenceSessionIndex | null {
  const indexPath = getInferenceSessionIndexPath(speciesId);
  if (!fs.existsSync(indexPath)) return null;
  try {
    const parsed = JSON.parse(fs.readFileSync(indexPath, "utf-8"));
    if (!parsed || typeof parsed !== "object") return null;
    const canonicalSessionId = sanitizeInferenceSessionId(parsed.canonicalSessionId);
    if (!canonicalSessionId) return null;
    return {
      version: 1,
      canonicalSessionId,
      migratedFromSessionId: parsed.migratedFromSessionId
        ? sanitizeInferenceSessionId(parsed.migratedFromSessionId)
        : undefined,
      updatedAt: String(parsed.updatedAt || new Date().toISOString()),
    };
  } catch {
    return null;
  }
}

function writeInferenceSessionIndex(speciesId: string, payload: InferenceSessionIndex): void {
  const sessionsRoot = getInferenceSessionsRoot(speciesId);
  fs.mkdirSync(sessionsRoot, { recursive: true });
  fs.writeFileSync(getInferenceSessionIndexPath(speciesId), JSON.stringify(payload, null, 2));
}

function copyFileIfMissing(sourcePath: string, targetPath: string): void {
  if (!fs.existsSync(sourcePath) || fs.existsSync(targetPath)) return;
  fs.mkdirSync(path.dirname(targetPath), { recursive: true });
  try {
    fs.copyFileSync(sourcePath, targetPath);
  } catch {
    // non-fatal migration helper
  }
}

function getInferenceReviewDraftsPath(speciesId: string, inferenceSessionId?: string): string {
  if (inferenceSessionId) {
    return path.join(getInferenceSessionDir(speciesId, inferenceSessionId), INFERENCE_REVIEW_DRAFTS_FILE);
  }
  return path.join(getSessionDir(speciesId), INFERENCE_REVIEW_DRAFTS_FILE);
}

function getRetrainQueuePath(speciesId: string, inferenceSessionId?: string): string {
  if (inferenceSessionId) {
    return path.join(getInferenceSessionDir(speciesId, inferenceSessionId), RETRAIN_QUEUE_FILE);
  }
  return path.join(getSessionDir(speciesId), RETRAIN_QUEUE_FILE);
}

function ensureInferenceSessionManifest(args: {
  speciesId: string;
  inferenceSessionId: string;
  displayName?: string;
  landmarkModelKey?: string;
  landmarkModelName?: string;
  landmarkPredictorType?: "dlib" | "cnn" | "yolo_pose";
  detectionModelKey?: string;
  detectionModelName?: string;
  preferences?: {
    lastUsedLandmarkModelKey?: string;
    lastUsedPredictorType?: "dlib" | "cnn" | "yolo_pose";
    detectionModelKey?: string;
    detectionModelName?: string;
  };
}): InferenceSessionManifest {
  const sessionDir = getInferenceSessionDir(args.speciesId, args.inferenceSessionId);
  fs.mkdirSync(sessionDir, { recursive: true });
  const manifestPath = getInferenceSessionManifestPath(args.speciesId, args.inferenceSessionId);
  const now = new Date().toISOString();
  let existing: InferenceSessionManifest | null = null;
  if (fs.existsSync(manifestPath)) {
    try {
      existing = JSON.parse(fs.readFileSync(manifestPath, "utf-8"));
    } catch {
      existing = null;
    }
  }

  const manifest: InferenceSessionManifest = {
    version: 1,
    sessionId: args.inferenceSessionId,
    speciesId: args.speciesId,
    displayName:
      (typeof args.displayName === "string" && args.displayName.trim()) ||
      existing?.displayName ||
      args.inferenceSessionId,
    models: {
      landmark: {
        key: args.landmarkModelKey || existing?.models?.landmark?.key || "unknown_landmark",
        name: args.landmarkModelName || existing?.models?.landmark?.name,
        predictorType:
          args.landmarkPredictorType ||
          existing?.models?.landmark?.predictorType,
      },
      detection: {
        key: args.detectionModelKey || existing?.models?.detection?.key || "session_detection_default",
        name: args.detectionModelName || existing?.models?.detection?.name,
      },
    },
    preferences: {
      lastUsedLandmarkModelKey:
        args.preferences?.lastUsedLandmarkModelKey ??
        existing?.preferences?.lastUsedLandmarkModelKey,
      lastUsedPredictorType:
        args.preferences?.lastUsedPredictorType ??
        existing?.preferences?.lastUsedPredictorType,
      detectionModelKey:
        args.preferences?.detectionModelKey ??
        existing?.preferences?.detectionModelKey ??
        args.detectionModelKey,
      detectionModelName:
        args.preferences?.detectionModelName ??
        existing?.preferences?.detectionModelName ??
        args.detectionModelName,
    },
    createdAt: existing?.createdAt || now,
    updatedAt: now,
  };
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  return manifest;
}

function resolveCanonicalInferenceSession(
  speciesId: string,
  options?: { createIfMissing?: boolean }
): { inferenceSessionId: string | null; manifest: InferenceSessionManifest | null; migratedFrom?: string } {
  const sessionsRoot = getInferenceSessionsRoot(speciesId);
  const canonicalSessionId = CANONICAL_INFERENCE_SESSION_ID;
  const canonicalManifest = readInferenceSessionManifest(speciesId, canonicalSessionId);
  if (canonicalManifest) {
    const existingIndex = readInferenceSessionIndex(speciesId);
    if (!existingIndex || existingIndex.canonicalSessionId !== canonicalSessionId) {
      writeInferenceSessionIndex(speciesId, {
        version: 1,
        canonicalSessionId,
        updatedAt: new Date().toISOString(),
      });
    }
    return { inferenceSessionId: canonicalSessionId, manifest: canonicalManifest };
  }

  fs.mkdirSync(sessionsRoot, { recursive: true });
  const index = readInferenceSessionIndex(speciesId);
  if (index?.canonicalSessionId && index.canonicalSessionId !== canonicalSessionId) {
    const legacyByIndex = readInferenceSessionManifest(speciesId, index.canonicalSessionId);
    if (legacyByIndex) {
      const sourceDir = getInferenceSessionDir(speciesId, index.canonicalSessionId);
      const targetDir = getInferenceSessionDir(speciesId, canonicalSessionId);
      fs.mkdirSync(targetDir, { recursive: true });
      copyFileIfMissing(
        path.join(sourceDir, INFERENCE_REVIEW_DRAFTS_FILE),
        path.join(targetDir, INFERENCE_REVIEW_DRAFTS_FILE)
      );
      copyFileIfMissing(
        path.join(sourceDir, RETRAIN_QUEUE_FILE),
        path.join(targetDir, RETRAIN_QUEUE_FILE)
      );
      copyFileIfMissing(
        path.join(sourceDir, "image_paths.json"),
        path.join(targetDir, "image_paths.json")
      );
      const migrated = ensureInferenceSessionManifest({
        speciesId,
        inferenceSessionId: canonicalSessionId,
        displayName: legacyByIndex.displayName || legacyByIndex.sessionId,
        landmarkModelKey: legacyByIndex.models?.landmark?.key,
        landmarkModelName: legacyByIndex.models?.landmark?.name,
        landmarkPredictorType: legacyByIndex.models?.landmark?.predictorType,
        detectionModelKey: legacyByIndex.models?.detection?.key,
        detectionModelName: legacyByIndex.models?.detection?.name,
        preferences: legacyByIndex.preferences,
      });
      writeInferenceSessionIndex(speciesId, {
        version: 1,
        canonicalSessionId,
        migratedFromSessionId: index.canonicalSessionId,
        updatedAt: new Date().toISOString(),
      });
      return {
        inferenceSessionId: canonicalSessionId,
        manifest: migrated,
        migratedFrom: index.canonicalSessionId,
      };
    }
  }

  const legacyCandidates = fs
    .readdirSync(sessionsRoot, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => sanitizeInferenceSessionId(entry.name))
    .filter((id) => id && id !== canonicalSessionId)
    .map((id) => {
      const manifest = readInferenceSessionManifest(speciesId, id);
      if (!manifest) return null;
      const dir = getInferenceSessionDir(speciesId, id);
      const updatedMs = Number.isFinite(Date.parse(manifest.updatedAt))
        ? Date.parse(manifest.updatedAt)
        : fs.statSync(dir).mtimeMs;
      return { id, manifest, dir, updatedMs };
    })
    .filter((entry): entry is { id: string; manifest: InferenceSessionManifest; dir: string; updatedMs: number } => Boolean(entry))
    .sort((a, b) => b.updatedMs - a.updatedMs);

  if (legacyCandidates.length > 0) {
    const pick = legacyCandidates[0];
    const targetDir = getInferenceSessionDir(speciesId, canonicalSessionId);
    fs.mkdirSync(targetDir, { recursive: true });
    copyFileIfMissing(
      path.join(pick.dir, INFERENCE_REVIEW_DRAFTS_FILE),
      path.join(targetDir, INFERENCE_REVIEW_DRAFTS_FILE)
    );
    copyFileIfMissing(
      path.join(pick.dir, RETRAIN_QUEUE_FILE),
      path.join(targetDir, RETRAIN_QUEUE_FILE)
    );
    copyFileIfMissing(
      path.join(pick.dir, "image_paths.json"),
      path.join(targetDir, "image_paths.json")
    );
    const migrated = ensureInferenceSessionManifest({
      speciesId,
      inferenceSessionId: canonicalSessionId,
      displayName: pick.manifest.displayName || pick.manifest.sessionId,
      landmarkModelKey: pick.manifest.models?.landmark?.key,
      landmarkModelName: pick.manifest.models?.landmark?.name,
      landmarkPredictorType: pick.manifest.models?.landmark?.predictorType,
      detectionModelKey: pick.manifest.models?.detection?.key,
      detectionModelName: pick.manifest.models?.detection?.name,
      preferences: pick.manifest.preferences,
    });
    writeInferenceSessionIndex(speciesId, {
      version: 1,
      canonicalSessionId,
      migratedFromSessionId: pick.id,
      updatedAt: new Date().toISOString(),
    });
    return {
      inferenceSessionId: canonicalSessionId,
      manifest: migrated,
      migratedFrom: pick.id,
    };
  }

  if (!options?.createIfMissing) {
    return { inferenceSessionId: null, manifest: null };
  }

  const created = ensureInferenceSessionManifest({
    speciesId,
    inferenceSessionId: canonicalSessionId,
    displayName: "Inference Session",
    detectionModelKey: "session_detection_default",
    detectionModelName: "Session Detection Model",
  });
  writeInferenceSessionIndex(speciesId, {
    version: 1,
    canonicalSessionId,
    updatedAt: new Date().toISOString(),
  });
  return { inferenceSessionId: canonicalSessionId, manifest: created };
}

function migrateLegacyInferenceArtifactsToSession(speciesId: string, inferenceSessionId: string): void {
  const legacyDraftPath = getInferenceReviewDraftsPath(speciesId);
  const legacyQueuePath = getRetrainQueuePath(speciesId);
  const sessionDraftPath = getInferenceReviewDraftsPath(speciesId, inferenceSessionId);
  const sessionQueuePath = getRetrainQueuePath(speciesId, inferenceSessionId);
  const sessionDir = getInferenceSessionDir(speciesId, inferenceSessionId);
  fs.mkdirSync(sessionDir, { recursive: true });

  if (!fs.existsSync(sessionDraftPath) && fs.existsSync(legacyDraftPath)) {
    try {
      fs.copyFileSync(legacyDraftPath, sessionDraftPath);
    } catch (_) {}
  }
  if (!fs.existsSync(sessionQueuePath) && fs.existsSync(legacyQueuePath)) {
    try {
      fs.copyFileSync(legacyQueuePath, sessionQueuePath);
    } catch (_) {}
  }
}

function normalizeDraftPath(value: string): string {
  return (value || "").replace(/\\/g, "/").toLowerCase();
}

function buildInferenceReviewDraftKey(imagePath: string, filename?: string): string {
  const safeFilename = (filename || path.basename(imagePath || "") || "").toLowerCase();
  const normalizedPath = normalizeDraftPath(path.resolve(imagePath || filename || safeFilename));
  return `${safeFilename}::${normalizedPath}`;
}

function readInferenceReviewDrafts(speciesId: string, inferenceSessionId?: string): InferenceReviewDraftsPayload {
  const draftPath = getInferenceReviewDraftsPath(speciesId, inferenceSessionId);
  if (!fs.existsSync(draftPath)) {
    return { version: 1, updatedAt: new Date().toISOString(), items: {} };
  }
  try {
    const parsed = JSON.parse(fs.readFileSync(draftPath, "utf-8"));
    const items = parsed?.items && typeof parsed.items === "object" ? parsed.items : {};
    return {
      version: 1,
      updatedAt: String(parsed?.updatedAt || new Date().toISOString()),
      items,
    };
  } catch {
    return { version: 1, updatedAt: new Date().toISOString(), items: {} };
  }
}

function writeInferenceReviewDrafts(
  speciesId: string,
  payload: InferenceReviewDraftsPayload,
  inferenceSessionId?: string
): void {
  const sessionDir = inferenceSessionId
    ? getInferenceSessionDir(speciesId, inferenceSessionId)
    : getSessionDir(speciesId);
  fs.mkdirSync(sessionDir, { recursive: true });
  const draftPath = getInferenceReviewDraftsPath(speciesId, inferenceSessionId);
  fs.writeFileSync(draftPath, JSON.stringify(payload, null, 2));
}

function buildRetrainQueueKey(filename: string): string {
  return (filename || "").trim().toLowerCase();
}

function readRetrainQueue(speciesId: string, inferenceSessionId?: string): RetrainQueuePayload {
  const queuePath = getRetrainQueuePath(speciesId, inferenceSessionId);
  if (!fs.existsSync(queuePath)) {
    return { version: 1, updatedAt: new Date().toISOString(), items: {} };
  }
  try {
    const parsed = JSON.parse(fs.readFileSync(queuePath, "utf-8"));
    const items = parsed?.items && typeof parsed.items === "object" ? parsed.items : {};
    return {
      version: 1,
      updatedAt: String(parsed?.updatedAt || new Date().toISOString()),
      items,
    };
  } catch {
    return { version: 1, updatedAt: new Date().toISOString(), items: {} };
  }
}

function writeRetrainQueue(
  speciesId: string,
  payload: RetrainQueuePayload,
  inferenceSessionId?: string
): void {
  const sessionDir = inferenceSessionId
    ? getInferenceSessionDir(speciesId, inferenceSessionId)
    : getSessionDir(speciesId);
  fs.mkdirSync(sessionDir, { recursive: true });
  const queuePath = getRetrainQueuePath(speciesId, inferenceSessionId);
  fs.writeFileSync(queuePath, JSON.stringify(payload, null, 2));
}

function sanitizeDraftSpecimens(specimens: unknown): InferenceDraftSpecimen[] {
  if (!Array.isArray(specimens)) return [];
  return specimens
    .filter((s: any) => s?.box && Number(s.box.width) > 0 && Number(s.box.height) > 0)
    .map((s: any) => ({
      box: {
        left: Math.round(Number(s.box.left) || 0),
        top: Math.round(Number(s.box.top) || 0),
        width: Math.round(Number(s.box.width) || 0),
        height: Math.round(Number(s.box.height) || 0),
        confidence: Number.isFinite(Number(s.box.confidence))
          ? Number(s.box.confidence)
          : undefined,
        class_id: Number.isFinite(Number(s.box.class_id))
          ? Number(s.box.class_id)
          : undefined,
        class_name:
          typeof s.box.class_name === "string" && s.box.class_name.trim()
            ? s.box.class_name.trim()
            : undefined,
        orientation_override:
          s.box.orientation_override === "left" ||
          s.box.orientation_override === "right" ||
          s.box.orientation_override === "uncertain"
            ? s.box.orientation_override
            : undefined,
        obbCorners: Array.isArray(s.box.obbCorners) && s.box.obbCorners.length === 4
          ? (s.box.obbCorners as any[]).map((p: any) =>
              Array.isArray(p) && p.length >= 2
                ? [Number(p[0]), Number(p[1])] as [number, number]
                : [0, 0] as [number, number]
            )
          : undefined,
        angle: typeof s.box.angle === "number" && Number.isFinite(s.box.angle)
          ? s.box.angle
          : undefined,
        orientation_hint: s.box.orientation_hint
          ? {
              orientation:
                s.box.orientation_hint.orientation === "left" || s.box.orientation_hint.orientation === "right"
                  ? s.box.orientation_hint.orientation
                  : undefined,
              confidence: Number.isFinite(Number(s.box.orientation_hint.confidence))
                ? Number(s.box.orientation_hint.confidence)
                : undefined,
              source:
                typeof s.box.orientation_hint.source === "string" && s.box.orientation_hint.source.trim()
                  ? s.box.orientation_hint.source.trim()
                  : undefined,
              head_point:
                Array.isArray(s.box.orientation_hint.head_point) && s.box.orientation_hint.head_point.length === 2
                  ? [Number(s.box.orientation_hint.head_point[0]), Number(s.box.orientation_hint.head_point[1])]
                  : undefined,
              tail_point:
                Array.isArray(s.box.orientation_hint.tail_point) && s.box.orientation_hint.tail_point.length === 2
                  ? [Number(s.box.orientation_hint.tail_point[0]), Number(s.box.orientation_hint.tail_point[1])]
                  : undefined,
            }
          : undefined,
      },
      landmarks: Array.isArray(s.landmarks)
        ? s.landmarks.map((lm: any) => ({
            id: Number(lm?.id) || 0,
            x: Math.round(Number(lm?.x) || 0),
            y: Math.round(Number(lm?.y) || 0),
          }))
        : [],
    }));
}

const IMAGE_EXTS = /\.(jpg|jpeg|png|gif|bmp|webp)$/i;

const MIME_TYPES: Record<string, string> = {
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
  ".gif": "image/gif",
  ".bmp": "image/bmp",
  ".webp": "image/webp",
};

const ORIENTATION_MODES = new Set(["directional", "bilateral", "axial", "invariant"]);

function hasConfiguredOrientationPolicy(meta: any): boolean {
  if (!meta || typeof meta !== "object") return false;
  if (!meta.orientationPolicy || typeof meta.orientationPolicy !== "object") return false;
  const mode = String(meta.orientationPolicy.mode || "").trim().toLowerCase();
  if (!ORIENTATION_MODES.has(mode)) return false;
  return Boolean(meta.orientationPolicyConfigured);
}

ipcMain.handle(
  "session:create",
  async (
    _event,
    args: {
      speciesId: string;
      name: string;
      landmarkTemplate: any[];
      orientationPolicy?: {
        mode?: "directional" | "bilateral" | "axial" | "invariant";
        targetOrientation?: "left" | "right";
        headCategories?: string[];
        tailCategories?: string[];
        bilateralPairs?: [number, number][];
        pcaLevelingMode?: "off" | "on" | "auto";
      };
    }
  ) => {
    try {
      const sessionDir = getSessionDir(args.speciesId);
      for (const sub of ["images", "labels", "models", "xml", "corrected_images", "debug"]) {
        fs.mkdirSync(path.join(sessionDir, sub), { recursive: true });
      }
      fs.writeFileSync(
        path.join(sessionDir, "session.json"),
        JSON.stringify(
          {
            speciesId: args.speciesId,
            name: args.name,
            landmarkTemplate: args.landmarkTemplate,
            orientationPolicy: args.orientationPolicy || undefined,
            orientationPolicyConfigured: Boolean(args.orientationPolicy),
            orientationPolicyConfiguredAt: args.orientationPolicy
              ? new Date().toISOString()
              : undefined,
            augmentationPolicy: { gravity_aligned: true },
            createdAt: new Date().toISOString(),
            lastModified: new Date().toISOString(),
            imageCount: 0,
          },
          null,
          2
        )
      );
      return { ok: true };
    } catch (e: any) {
      console.error("session:create failed:", e);
      return { ok: false, error: e.message };
    }
  }
);

ipcMain.handle(
  "session:save-image",
  async (
    _event,
    args: {
      speciesId: string;
      imageData: string;
      filename: string;
      mimeType: string;
    }
  ) => {
    try {
      const sessionDir = getSessionDir(args.speciesId);
      const imagesDir = path.join(sessionDir, "images");
      fs.mkdirSync(imagesDir, { recursive: true });

      const destPath = path.join(imagesDir, args.filename);
      const buffer = Buffer.from(args.imageData, "base64");
      fs.writeFileSync(destPath, buffer);

      // Update or create session.json imageCount
      const sessionJsonPath = path.join(sessionDir, "session.json");
      try {
        const imageFiles = fs
          .readdirSync(imagesDir)
          .filter((f) => IMAGE_EXTS.test(f));
        if (fs.existsSync(sessionJsonPath)) {
          const meta = JSON.parse(fs.readFileSync(sessionJsonPath, "utf-8"));
          meta.imageCount = imageFiles.length;
          meta.lastModified = new Date().toISOString();
          fs.writeFileSync(sessionJsonPath, JSON.stringify(meta, null, 2));
        } else {
          // Fallback: create session.json if it was never created
          fs.writeFileSync(
            sessionJsonPath,
            JSON.stringify(
              {
                speciesId: args.speciesId,
                name: args.speciesId,
                landmarkTemplate: [],
                createdAt: new Date().toISOString(),
                lastModified: new Date().toISOString(),
                imageCount: imageFiles.length,
              },
              null,
              2
            )
          );
        }
      } catch (_) {
        // non-critical
      }

      return { ok: true, diskPath: destPath };
    } catch (e: any) {
      console.error("session:save-image failed:", e);
      return { ok: false, error: e.message, diskPath: "" };
    }
  }
);

ipcMain.handle(
  "session:save-annotations",
  async (
    _event,
    args: {
      speciesId: string;
      filename: string;
      boxes: any[];
    }
  ) => {
    try {
      const sessionDir = getSessionDir(args.speciesId);
      const imagesDir = path.join(sessionDir, "images");
      const labelsDir = path.join(sessionDir, "labels");
      fs.mkdirSync(imagesDir, { recursive: true });
      fs.mkdirSync(labelsDir, { recursive: true });

      const basename = args.filename.replace(/\.\w+$/, ".json");
      const labelPath = path.join(labelsDir, basename);
      const boxes = (args.boxes || []).map((box: any) => {
        const b: any = {
          id: box.id,
          left: box.left,
          top: box.top,
          width: box.width,
          height: box.height,
          confidence: box.confidence,
          source: box.source,
          landmarks: (box.landmarks || []).map((lm: any) => ({
            x: lm.x,
            y: lm.y,
            id: lm.id,
            isSkipped: lm.isSkipped,
            label: lm.label,
          })),
        };
        // Preserve OBB geometry fields for training pipeline
        if (Array.isArray(box.obbCorners) && box.obbCorners.length === 4) {
          b.obbCorners = box.obbCorners;
        }
        if (box.angle != null) b.angle = box.angle;
        if (box.class_id != null) b.class_id = box.class_id;
        return b;
      });

      let rejectedDetections: any[] = [];
      let finalizedDetection: any = undefined;
      if (fs.existsSync(labelPath)) {
        try {
          const previous = JSON.parse(fs.readFileSync(labelPath, "utf-8"));
          if (Array.isArray(previous?.rejectedDetections)) {
            rejectedDetections = previous.rejectedDetections;
          }
          if (previous?.finalizedDetection && typeof previous.finalizedDetection === "object") {
            finalizedDetection = previous.finalizedDetection;
          }
        } catch (_) {
          // ignore malformed previous JSON
        }
      }

      const payload: any = {
        imageFilename: args.filename,
        speciesId: args.speciesId,
        boxes,
        rejectedDetections,
      };
      // Preserve finalized detection snapshot across autosaves.
      if (finalizedDetection) {
        payload.finalizedDetection = finalizedDetection;
      }

      fs.writeFileSync(labelPath, JSON.stringify(payload, null, 2));

      // Update lastModified
      const sessionJsonPath = path.join(sessionDir, "session.json");
      if (fs.existsSync(sessionJsonPath)) {
        try {
          const meta = JSON.parse(fs.readFileSync(sessionJsonPath, "utf-8"));
          meta.lastModified = new Date().toISOString();
          fs.writeFileSync(sessionJsonPath, JSON.stringify(meta, null, 2));
        } catch (_) {
          // non-critical
        }
      }

      return { ok: true };
    } catch (e: any) {
      console.error("session:save-annotations failed:", e);
      return { ok: false, error: e.message };
    }
  }
);

ipcMain.handle(
  "session:finalize-accepted-boxes",
  async (
    _event,
    args: {
      speciesId: string;
      filename: string;
      boxes: {
        left: number;
        top: number;
        width: number;
        height: number;
        landmarks?: { id: number; x: number; y: number; isSkipped?: boolean }[];
      }[];
      imagePath?: string;
    }
  ) => {
    try {
      const sessionDir = getSessionDir(args.speciesId);
      const sessionImagePath = path.join(sessionDir, "images", args.filename);
      const resolvedImagePath = fs.existsSync(sessionImagePath) ? sessionImagePath : args.imagePath;
      const sam2Ready = await ensureSam2Ready();
      if (!sam2Ready.ok) {
        return { ok: false, error: sam2Ready.error || "SAM2 is required to finalize segments." };
      }

      const acceptedBoxes = normalizeFinalizedAcceptedBoxes(args.boxes || []);

      const cacheKey = `${args.speciesId}::${args.filename}`;
      const signature = buildAcceptedBoxesSignature(acceptedBoxes);
      const cacheHit = finalizedSegmentSignatureCache.get(cacheKey) === signature;
      finalizedSegmentSignatureCache.set(cacheKey, signature);

      let segmentQueueState: "queued" | "skipped" = "skipped";
      let segmentSaveQueued = false;
      if (!cacheHit) {
        const queued = enqueueSegmentSave({
          queueKey: cacheKey,
          sessionKey: sessionDir,
          speciesId: args.speciesId,
          filename: args.filename,
          sessionDir,
          imagePath: resolvedImagePath,
          acceptedBoxes,
          signature,
        });
        segmentQueueState = queued.state;
        segmentSaveQueued = queued.queued;
      } else {
        const currentStatus = getSegmentQueueStatus(args.speciesId, args.filename);
        segmentQueueState = currentStatus.state === "queued" ? "queued" : "skipped";
      }

      return {
        ok: true,
        finalized: false,
        queued: segmentSaveQueued,
        acceptedCount: acceptedBoxes.length,
        signature,
        skipped: cacheHit,
        segmentSaveQueued,
        segmentQueueState,
      };
    } catch (e: any) {
      console.error("session:finalize-accepted-boxes failed:", e);
      return { ok: false, error: e.message };
    }
  }
);

ipcMain.handle(
  "session:unfinalize-image",
  async (
    _event,
    args: {
      speciesId: string;
      filename: string;
      imagePath?: string;
    }
  ) => {
    try {
      const speciesId = String(args?.speciesId || "").trim();
      const filename = String(args?.filename || "").trim();
      if (!speciesId) {
        return { ok: false, error: "Invalid speciesId" };
      }
      if (!filename) {
        return { ok: false, error: "Invalid filename" };
      }

      const result = unfinalizeImageInSession(speciesId, filename, args?.imagePath);
      if (!result.ok) {
        return { ok: false, error: result.error || "Failed to unfinalize image." };
      }

      return {
        ok: true,
        unfinalized: true,
        filename: result.filename,
        removedFromList: Boolean(result.removedFromList),
        removedSegments: Number(result.removedSegments || 0),
        hadFinalizedDetection: Boolean(result.hadFinalizedDetection),
      };
    } catch (e: any) {
      console.error("session:unfinalize-image failed:", e);
      return { ok: false, error: e.message };
    }
  }
);

ipcMain.handle(
  "session:unfinalize-images",
  async (
    _event,
    args: {
      speciesId: string;
      filenames?: string[];
    }
  ) => {
    try {
      const speciesId = String(args?.speciesId || "").trim();
      if (!speciesId) {
        return { ok: false, error: "Invalid speciesId", requested: 0, succeeded: 0, failed: 0, removedSegmentsTotal: 0 };
      }

      const sessionDir = getSessionDir(speciesId);
      const explicitNames = Array.isArray(args?.filenames) ? args.filenames : [];
      const targetNames =
        explicitNames.length > 0
          ? explicitNames
              .map((name) => path.basename(String(name || "").trim()))
              .filter((name) => Boolean(name))
          : readFinalizedList(sessionDir)
              .map((name) => path.basename(String(name || "").trim()))
              .filter((name) => Boolean(name));

      const dedupedNames: string[] = [];
      const seen = new Set<string>();
      targetNames.forEach((name) => {
        const key = name.toLowerCase();
        if (seen.has(key)) return;
        seen.add(key);
        dedupedNames.push(name);
      });

      let succeeded = 0;
      let failed = 0;
      let removedSegmentsTotal = 0;
      const errors: Array<{ filename: string; error: string }> = [];

      for (const filename of dedupedNames) {
        const result = unfinalizeImageInSession(speciesId, filename);
        if (result.ok) {
          succeeded += 1;
          removedSegmentsTotal += Number(result.removedSegments || 0);
        } else {
          failed += 1;
          errors.push({ filename, error: result.error || "Failed to unfinalize image." });
        }
      }

      return {
        ok: failed === 0,
        requested: dedupedNames.length,
        succeeded,
        failed,
        removedSegmentsTotal,
        ...(errors.length > 0 ? { errors } : {}),
      };
    } catch (e: any) {
      console.error("session:unfinalize-images failed:", e);
      return { ok: false, error: e.message, requested: 0, succeeded: 0, failed: 0, removedSegmentsTotal: 0 };
    }
  }
);

ipcMain.handle(
  "session:add-rejected-detection",
  async (
    _event,
    args: {
      speciesId: string;
      filename: string;
      rejectedDetection: {
        left: number;
        top: number;
        width: number;
        height: number;
        confidence?: number;
        className?: string;
        detectionMethod?: string;
        rejectedAt?: string;
      };
    }
  ) => {
    try {
      const sessionDir = getSessionDir(args.speciesId);
      const labelsDir = path.join(sessionDir, "labels");
      fs.mkdirSync(labelsDir, { recursive: true });

      const basename = args.filename.replace(/\.\w+$/, ".json");
      const labelPath = path.join(labelsDir, basename);

      let payload: any = {
        imageFilename: args.filename,
        speciesId: args.speciesId,
        boxes: [],
        rejectedDetections: [],
      };
      if (fs.existsSync(labelPath)) {
        try {
          payload = JSON.parse(fs.readFileSync(labelPath, "utf-8"));
        } catch (_) {
          // Keep default payload if existing file is malformed.
        }
      }

      payload.imageFilename = payload.imageFilename || args.filename;
      payload.speciesId = payload.speciesId || args.speciesId;
      payload.boxes = Array.isArray(payload.boxes) ? payload.boxes : [];
      payload.rejectedDetections = Array.isArray(payload.rejectedDetections) ? payload.rejectedDetections : [];
      payload.rejectedDetections.push({
        ...args.rejectedDetection,
        rejectedAt: args.rejectedDetection?.rejectedAt || new Date().toISOString(),
      });

      fs.writeFileSync(labelPath, JSON.stringify(payload, null, 2));
      return { ok: true };
    } catch (e: any) {
      console.error("session:add-rejected-detection failed:", e);
      return { ok: false, error: e.message };
    }
  }
);

ipcMain.handle(
  "session:load",
  async (_event, args: { speciesId: string }) => {
    try {
      const sessionDir = getSessionDir(args.speciesId);
      const imagesDir = path.join(sessionDir, "images");
      const labelsDir = path.join(sessionDir, "labels");
      const warnings = sessionContainsLegacyNonObbLabels(sessionDir)
        ? ["This session contains legacy boxes without OBB geometry. Re-import the annotations to normalize OBB labels."]
        : [];

      // Load session metadata from session.json
      let meta: any = null;
      const sessionJsonPath = path.join(sessionDir, "session.json");
      if (fs.existsSync(sessionJsonPath)) {
        try {
          meta = JSON.parse(fs.readFileSync(sessionJsonPath, "utf-8"));
          if (meta && typeof meta === "object") {
            meta.orientationPolicyConfigured = hasConfiguredOrientationPolicy(meta);
          }
        } catch (_) {
          // skip bad session.json
        }
      }
      const effectiveRoot = getEffectiveRoot(args.speciesId);
      const obbDetectorPath = path.join(effectiveRoot, "models", "session_obb_detector.pt");
      meta = meta && typeof meta === "object" ? meta : {};
      meta.obbDetectorReady = fs.existsSync(obbDetectorPath);

      if (!fs.existsSync(imagesDir)) {
        return { ok: true, images: [], meta, warnings };
      }

      // Load finalized filenames
      const finalizedSet = new Set(
        readFinalizedList(sessionDir).map((name) =>
          path.basename(String(name || "")).toLowerCase()
        )
      );

      const imageFiles = fs
        .readdirSync(imagesDir)
        .filter((f) => IMAGE_EXTS.test(f));

      const images = imageFiles.map((filename) => {
        const filePath = path.join(imagesDir, filename);
        const ext = path.extname(filename).toLowerCase();
        let isFinalizedFromLabel = false;
        let hasBoxes = false;
        let boxes: any[] = [];
        const labelPath = path.join(labelsDir, filename.replace(/\.\w+$/, ".json"));
        if (fs.existsSync(labelPath)) {
          try {
            const labelData = JSON.parse(fs.readFileSync(labelPath, "utf-8"));
            isFinalizedFromLabel = Boolean(labelData?.finalizedDetection?.isFinalized);
            if (Array.isArray(labelData?.boxes)) {
              boxes = normalizeSessionStoredBoxes(labelData.boxes);
              hasBoxes = boxes.some((b: any) => Number(b?.width) > 0 && Number(b?.height) > 0);
            }
          } catch (_) {
            // skip bad label files
          }
        }

        return {
          filename,
          diskPath: filePath,
          mimeType: MIME_TYPES[ext] || "image/jpeg",
          hasBoxes,
          boxes,
          finalized: finalizedSet.has(filename.toLowerCase()) || isFinalizedFromLabel,
        };
      });

      return { ok: true, images, meta, warnings };
    } catch (e: any) {
      console.error("session:load failed:", e);
      return { ok: false, error: e.message, images: [] };
    }
  }
);

ipcMain.handle(
  "session:load-annotation",
  async (_event, args: { speciesId: string; filename: string }) => {
    try {
      const sessionDir = getSessionDir(args.speciesId);
      const labelsDir = path.join(sessionDir, "labels");
      const warnings = sessionContainsLegacyNonObbLabels(sessionDir)
        ? ["This session contains legacy boxes without OBB geometry. Re-import the annotations to normalize OBB labels."]
        : [];
      const safeFilename = path.basename(String(args.filename || "").trim());
      if (!safeFilename) {
        return { ok: false, error: "Invalid filename", boxes: [] };
      }
      const labelPath = path.join(labelsDir, safeFilename.replace(/\.\w+$/, ".json"));

      if (!fs.existsSync(labelPath)) {
        return { ok: true, boxes: [], finalized: false, warnings };
      }

      const labelData = JSON.parse(fs.readFileSync(labelPath, "utf-8"));
      return {
        ok: true,
        boxes: normalizeSessionStoredBoxes(labelData?.boxes),
        finalized: Boolean(labelData?.finalizedDetection?.isFinalized),
        warnings,
      };
    } catch (e: any) {
      console.error("session:load-annotation failed:", e);
      return { ok: false, error: e.message, boxes: [] };
    }
  }
);

ipcMain.handle(
  "session:get-segment-save-status",
  async (_event, args: { speciesId: string; filename: string }) => {
    try {
      return { ok: true, status: getSegmentQueueStatus(args.speciesId, args.filename) };
    } catch (e: any) {
      return { ok: false, error: e.message };
    }
  }
);

ipcMain.handle("session:list", async () => {
  try {
    const sessionsDir = path.join(projectRoot, "sessions");
    if (!fs.existsSync(sessionsDir)) {
      return { ok: true, sessions: [] };
    }

    const entries = fs
      .readdirSync(sessionsDir, { withFileTypes: true })
      .filter((d) => d.isDirectory());

    const sessions = [];
    for (const entry of entries) {
      const sessionJsonPath = path.join(
        sessionsDir,
        entry.name,
        "session.json"
      );
      if (!fs.existsSync(sessionJsonPath)) continue;
      try {
        const meta = JSON.parse(fs.readFileSync(sessionJsonPath, "utf-8"));
        sessions.push({
          speciesId: meta.speciesId || entry.name,
          name: meta.name || entry.name,
          imageCount: meta.imageCount || 0,
          lastModified: meta.lastModified || meta.createdAt || "",
          landmarkCount: (meta.landmarkTemplate || []).length,
          orientationPolicy: meta.orientationPolicy || undefined,
          orientationPolicyConfigured: hasConfiguredOrientationPolicy(meta),
        });
      } catch (_) {
        // skip bad session.json
      }
    }

    // Sort by lastModified descending
    sessions.sort(
      (a, b) =>
        new Date(b.lastModified).getTime() -
        new Date(a.lastModified).getTime()
    );

    return { ok: true, sessions };
  } catch (e: any) {
    console.error("session:list failed:", e);
    return { ok: false, error: e.message, sessions: [] };
  }
});

ipcMain.handle(
  "session:delete-all-images",
  async (_event, args: { speciesId: string }) => {
    try {
      const sessionDir = getSessionDir(args.speciesId);
      const imagesDir = path.join(sessionDir, "images");
      const labelsDir = path.join(sessionDir, "labels");
      cancelAllQueuedSegmentSavesForSpecies(args.speciesId);

      // Delete all image files
      if (fs.existsSync(imagesDir)) {
        for (const file of fs.readdirSync(imagesDir)) {
          try { fs.unlinkSync(path.join(imagesDir, file)); } catch {}
        }
      }
      // Delete all label files
      if (fs.existsSync(labelsDir)) {
        for (const file of fs.readdirSync(labelsDir)) {
          try { fs.unlinkSync(path.join(labelsDir, file)); } catch {}
        }
      }
      // Delete persisted inference-review drafts (legacy root + canonical session)
      try {
        const draftPath = getInferenceReviewDraftsPath(args.speciesId);
        if (fs.existsSync(draftPath)) {
          fs.unlinkSync(draftPath);
        }
      } catch (_) {
        // non-critical
      }
      try {
        const sessionDir = getInferenceSessionDir(args.speciesId, CANONICAL_INFERENCE_SESSION_ID);
        const sessionDraftPath = path.join(sessionDir, INFERENCE_REVIEW_DRAFTS_FILE);
        if (fs.existsSync(sessionDraftPath)) fs.unlinkSync(sessionDraftPath);
        const pathsFile = path.join(sessionDir, "image_paths.json");
        if (fs.existsSync(pathsFile)) fs.writeFileSync(pathsFile, JSON.stringify({ imagePaths: [] }, null, 2));
      } catch (_) {
        // non-critical
      }
      // Delete persisted retrain queue
      try {
        const queuePath = getRetrainQueuePath(args.speciesId);
        if (fs.existsSync(queuePath)) {
          fs.unlinkSync(queuePath);
        }
      } catch (_) {
        // non-critical
      }

      // Update session.json imageCount to 0
      const sessionJsonPath = path.join(sessionDir, "session.json");
      if (fs.existsSync(sessionJsonPath)) {
        try {
          const meta = JSON.parse(fs.readFileSync(sessionJsonPath, "utf-8"));
          meta.imageCount = 0;
          meta.lastModified = new Date().toISOString();
          fs.writeFileSync(sessionJsonPath, JSON.stringify(meta, null, 2));
        } catch (_) {}
      }

      return { ok: true };
    } catch (e: any) {
      console.error("session:delete-all-images failed:", e);
      return { ok: false, error: e.message };
    }
  }
);

function inferImageDimensionsForDetection(
  imagePath: string,
  boxes: { left: number; top: number; width: number; height: number }[]
): { width: number; height: number } {
  try {
    const size = nativeImage.createFromPath(imagePath).getSize();
    const width = Math.max(1, Math.round(Number(size?.width) || 0));
    const height = Math.max(1, Math.round(Number(size?.height) || 0));
    if (width > 1 && height > 1) return { width, height };
  } catch {
    // fallback to box extent
  }
  const width = Math.max(
    1,
    boxes.reduce((max, b) => Math.max(max, Math.round((b.left || 0) + (b.width || 0))), 0)
  );
  const height = Math.max(
    1,
    boxes.reduce((max, b) => Math.max(max, Math.round((b.top || 0) + (b.height || 0))), 0)
  );
  return { width, height };
}

function persistInferenceCorrectionToSession(args: {
  speciesId: string;
  imagePath: string;
  box?: { left: number; top: number; width: number; height: number };
  landmarks?: { id: number; x: number; y: number }[];
  specimens?: {
    box: {
      left: number;
      top: number;
      width: number;
      height: number;
      orientation_override?: "left" | "right" | "uncertain";
      obbCorners?: [number, number][];
      angle?: number;
      class_id?: number;
    };
    landmarks: { id: number; x: number; y: number }[];
  }[];
  rejectedDetections?: {
    left: number;
    top: number;
    width: number;
    height: number;
    confidence?: number;
    className?: string;
    detectionMethod?: string;
  }[];
  allowEmpty?: boolean;
  filename?: string;
}): { ok: boolean; savedPath?: string; error?: string; imageName?: string } {
  try {
    const sessionDir = getSessionDir(args.speciesId);
    const imagesDir = path.join(sessionDir, "images");
    const labelsDir = path.join(sessionDir, "labels");
    fs.mkdirSync(imagesDir, { recursive: true });
    fs.mkdirSync(labelsDir, { recursive: true });

    let imgName = args.filename ?? path.basename(args.imagePath);
    let imgDest = path.join(imagesDir, imgName);

    if (fs.existsSync(args.imagePath)) {
      if (!fs.existsSync(imgDest)) {
        fs.copyFileSync(args.imagePath, imgDest);
      } else {
        // Check if the existing file is different from the source
        const srcStat = fs.statSync(args.imagePath);
        const dstStat = fs.statSync(imgDest);
        if (srcStat.size !== dstStat.size || srcStat.mtimeMs !== dstStat.mtimeMs) {
          // Different file Ã¢â‚¬â€ disambiguate with a short hash of the source path
          const hash = require("crypto").createHash("md5").update(args.imagePath).digest("hex").slice(0, 6);
          const ext = path.extname(imgName);
          const base = path.basename(imgName, ext);
          imgName = `${base}_${hash}${ext}`;
          imgDest = path.join(imagesDir, imgName);
          if (!fs.existsSync(imgDest)) {
            fs.copyFileSync(args.imagePath, imgDest);
          }
        }
      }
    }

    const lblDest = path.join(labelsDir, imgName.replace(/\.\w+$/, ".json"));

    let boxesPayload: any[] = [];
    if (Array.isArray(args.specimens) && args.specimens.length > 0) {
      boxesPayload = args.specimens
        .filter((s) => s?.box && s.box.width > 0 && s.box.height > 0)
        .map((s) => {
          const entry: any = {
            left: Math.round(s.box.left),
            top: Math.round(s.box.top),
            width: Math.round(s.box.width),
            height: Math.round(s.box.height),
            ...(s.box.orientation_override === "left" ||
            s.box.orientation_override === "right" ||
            s.box.orientation_override === "uncertain"
              ? { orientation_override: s.box.orientation_override }
              : {}),
            landmarks: (s.landmarks || []).map((lm) => ({
              id: Number(lm.id),
              x: Number(lm.x),
              y: Number(lm.y),
              isSkipped: false,
            })),
          };
          if (Array.isArray(s.box.obbCorners) && s.box.obbCorners.length === 4) entry.obbCorners = s.box.obbCorners;
          if (s.box.angle != null) entry.angle = s.box.angle;
          if (s.box.class_id != null) entry.class_id = s.box.class_id;
          return entry;
        });
    } else if (args.box && Array.isArray(args.landmarks)) {
      boxesPayload = [
        {
          left: Math.round(args.box.left),
          top: Math.round(args.box.top),
          width: Math.round(args.box.width),
          height: Math.round(args.box.height),
          landmarks: args.landmarks.map((lm) => ({
            id: Number(lm.id),
            x: Number(lm.x),
            y: Number(lm.y),
            isSkipped: false,
          })),
        },
      ];
    }

    if (boxesPayload.length === 0 && !args.allowEmpty) {
      return { ok: false, error: "No valid corrected specimens to save." };
    }

    let existingRejectedDetections: any[] = [];
    if (fs.existsSync(lblDest)) {
      try {
        const prev = JSON.parse(fs.readFileSync(lblDest, "utf-8"));
        if (Array.isArray(prev?.rejectedDetections)) {
          existingRejectedDetections = prev.rejectedDetections;
        }
      } catch {
        // ignore malformed existing file
      }
    }
    const incomingRejectedDetections = Array.isArray(args.rejectedDetections)
      ? args.rejectedDetections
          .filter((d) => d && Number(d.width) > 0 && Number(d.height) > 0)
          .map((d) => ({
            left: Math.round(Number(d.left) || 0),
            top: Math.round(Number(d.top) || 0),
            width: Math.round(Number(d.width) || 0),
            height: Math.round(Number(d.height) || 0),
            ...(Number.isFinite(Number(d.confidence))
              ? { confidence: Number(d.confidence) }
              : {}),
            ...(d.className ? { className: String(d.className) } : {}),
            ...(d.detectionMethod ? { detectionMethod: String(d.detectionMethod) } : {}),
            rejectedAt: new Date().toISOString(),
          }))
      : [];
    const mergedRejectedMap = new Map<string, any>();
    [...existingRejectedDetections, ...incomingRejectedDetections].forEach((d) => {
      const left = Math.round(Number(d?.left) || 0);
      const top = Math.round(Number(d?.top) || 0);
      const width = Math.round(Number(d?.width) || 0);
      const height = Math.round(Number(d?.height) || 0);
      if (width <= 0 || height <= 0) return;
      const key = `${left}:${top}:${width}:${height}`;
      if (!mergedRejectedMap.has(key)) {
        mergedRejectedMap.set(key, { ...d, left, top, width, height });
      }
    });
    const mergedRejectedDetections = Array.from(mergedRejectedMap.values());

    const acceptedBoxes = normalizeFinalizedAcceptedBoxes(
      boxesPayload.map((b) => ({
        left: b.left,
        top: b.top,
        width: b.width,
        height: b.height,
        orientation_override:
          b.orientation_override === "left" ||
          b.orientation_override === "right" ||
          b.orientation_override === "uncertain"
            ? b.orientation_override
            : undefined,
        ...(Array.isArray(b.obbCorners) && b.obbCorners.length === 4
          ? { obbCorners: b.obbCorners }
          : {}),
        ...(b.angle != null ? { angle: b.angle } : {}),
        ...(b.class_id != null ? { class_id: b.class_id } : {}),
        landmarks: b.landmarks,
      }))
    );
    const boxSignature = buildAcceptedBoxesSignature(acceptedBoxes);

    const label: any = {
      imageFilename: imgName,
      speciesId: args.speciesId,
      boxes: boxesPayload,
      rejectedDetections: mergedRejectedDetections,
      finalizedDetection: {
        isFinalized: true,
        finalizedAt: new Date().toISOString(),
        acceptedBoxes,
        boxSignature,
      },
    };
    fs.writeFileSync(lblDest, JSON.stringify(label, null, 2));

    try {
      const existing = readFinalizedList(sessionDir);
      const lower = path.basename(imgName).toLowerCase();
      const alreadyFinalized = existing.some(
        (name) => path.basename(String(name || "")).toLowerCase() === lower
      );
      if (!alreadyFinalized) {
        existing.push(imgName);
        writeFinalizedList(sessionDir, existing);
      }
    } catch {
      // non-fatal
    }

    const sessionJsonPath = path.join(sessionDir, "session.json");
    if (fs.existsSync(sessionJsonPath)) {
      try {
        const meta = JSON.parse(fs.readFileSync(sessionJsonPath, "utf-8"));
        meta.imageCount = fs.readdirSync(imagesDir).filter((f) => IMAGE_EXTS.test(f)).length;
        meta.lastModified = new Date().toISOString();
        fs.writeFileSync(sessionJsonPath, JSON.stringify(meta, null, 2));
      } catch {
        // non-fatal
      }
    }

    return { ok: true, savedPath: lblDest, imageName: imgName };
  } catch (e: any) {
    return { ok: false, error: String(e?.message || e) };
  }
}

function persistDetectionCorrectionToSession(args: {
  speciesId: string;
  imagePath: string;
  boxes: {
    left: number;
    top: number;
    width: number;
    height: number;
    orientation_override?: "left" | "right" | "uncertain";
    obbCorners?: [number, number][];
    angle?: number;
    class_id?: number;
  }[];
  imageWidth: number;
  imageHeight: number;
  filename?: string;
}): { ok: boolean; savedPath?: string; error?: string; imageName?: string } {
  try {
    const sessionDir = getSessionDir(args.speciesId);
    const imagesDir = path.join(sessionDir, "images");
    const detLblDir = path.join(sessionDir, "detection_labels");
    fs.mkdirSync(imagesDir, { recursive: true });
    fs.mkdirSync(detLblDir, { recursive: true });

    const imgName = args.filename ?? path.basename(args.imagePath);
    const imgDest = path.join(imagesDir, imgName);

    if (!fs.existsSync(imgDest) && fs.existsSync(args.imagePath)) {
      fs.copyFileSync(args.imagePath, imgDest);
    }

    const iw = Math.max(args.imageWidth, 1);
    const ih = Math.max(args.imageHeight, 1);
    const lines = (args.boxes || [])
      .filter((b) => b.width > 0 && b.height > 0)
      .map((b) => {
        const cx = ((b.left + b.width / 2) / iw).toFixed(6);
        const cy = ((b.top + b.height / 2) / ih).toFixed(6);
        const w = (b.width / iw).toFixed(6);
        const h = (b.height / ih).toFixed(6);
        return `0 ${cx} ${cy} ${w} ${h}`;
      });

    const lblDest = path.join(detLblDir, imgName.replace(/\.\w+$/, ".txt"));
    fs.writeFileSync(lblDest, lines.join("\n") + (lines.length > 0 ? "\n" : ""));
    return { ok: true, savedPath: lblDest, imageName: imgName };
  } catch (e: any) {
    return { ok: false, error: String(e?.message || e) };
  }
}

// Legacy IPC handler kept for backward compatibility with older renderer builds.
// Current renderer persists review edits through session:save-inference-review-draft.
ipcMain.handle(
  "session:save-inference-correction",
  async (
    _event,
    args: {
      speciesId: string;
      imagePath: string;
      box?: { left: number; top: number; width: number; height: number };
      landmarks?: { id: number; x: number; y: number }[];
      specimens?: {
        box: {
          left: number;
          top: number;
          width: number;
          height: number;
          orientation_override?: "left" | "right" | "uncertain";
        };
        landmarks: { id: number; x: number; y: number }[];
      }[];
      rejectedDetections?: {
        left: number;
        top: number;
        width: number;
        height: number;
        confidence?: number;
        className?: string;
        detectionMethod?: string;
      }[];
      allowEmpty?: boolean;
      filename?: string;
    }
  ) => {
    try {
      const result = persistInferenceCorrectionToSession(args);
      if (!result.ok) {
        return { ok: false, error: result.error || "Failed to save inference correction." };
      }
      return { ok: true, savedPath: result.savedPath };
    } catch (e: any) {
      console.error("session:save-inference-correction failed:", e);
      return { ok: false, error: e.message };
    }
  }
);

// Legacy IPC handler kept for backward compatibility with older renderer builds.
// Current renderer workflow uses inference review drafts + commit-inference-review.
ipcMain.handle(
  "session:open-inference-session",
  async (
    _event,
    args: {
      speciesId: string;
      landmarkModelKey: string;
      landmarkModelName?: string;
      landmarkPredictorType?: "dlib" | "cnn" | "yolo_pose";
      detectionModelKey?: string;
      detectionModelName?: string;
    }
  ) => {
    try {
      if (!args.speciesId) {
        return { ok: false, error: "speciesId is required." };
      }
      if (!args.landmarkModelKey) {
        return { ok: false, error: "landmarkModelKey is required." };
      }
      const inferenceSessionId = buildInferenceSessionId(
        args.speciesId,
        args.landmarkModelKey,
        args.detectionModelKey
      );
      const manifest = ensureInferenceSessionManifest({
        speciesId: args.speciesId,
        inferenceSessionId,
        landmarkModelKey: args.landmarkModelKey,
        landmarkModelName: args.landmarkModelName,
        landmarkPredictorType: args.landmarkPredictorType,
        detectionModelKey: args.detectionModelKey,
        detectionModelName: args.detectionModelName,
      });
      migrateLegacyInferenceArtifactsToSession(args.speciesId, inferenceSessionId);
      return { ok: true, inferenceSessionId, manifest };
    } catch (e: any) {
      console.error("session:open-inference-session failed:", e);
      return { ok: false, error: e.message };
    }
  }
);

ipcMain.handle("session:list-inference-sessions", async () => {
  try {
    const sessionsRoot = path.join(projectRoot, "sessions");
    if (!fs.existsSync(sessionsRoot)) {
      return { ok: true, sessions: [] };
    }

    const schemas = fs
      .readdirSync(sessionsRoot, { withFileTypes: true })
      .filter((entry) => entry.isDirectory());

    const sessions = schemas
      .map((entry) => {
        const speciesId = entry.name;
        const sessionJsonPath = path.join(sessionsRoot, speciesId, "session.json");
        if (!fs.existsSync(sessionJsonPath)) return null;
        let schemaName = speciesId;
        let schemaImageCount = 0;
        let schemaUpdatedAt = "";
        try {
          const meta = JSON.parse(fs.readFileSync(sessionJsonPath, "utf-8"));
          schemaName = String(meta?.name || speciesId);
          schemaImageCount = Math.max(0, Number(meta?.imageCount || 0));
          schemaUpdatedAt = String(meta?.lastModified || meta?.createdAt || "");
        } catch {
          // keep defaults
        }
        const resolved = resolveCanonicalInferenceSession(speciesId, { createIfMissing: false });
        return {
          speciesId,
          schemaName,
          schemaImageCount,
          schemaUpdatedAt,
          exists: Boolean(resolved.inferenceSessionId && resolved.manifest),
          inferenceSessionId: resolved.inferenceSessionId ?? undefined,
          displayName: resolved.manifest?.displayName,
          createdAt: resolved.manifest?.createdAt,
          updatedAt: resolved.manifest?.updatedAt,
          migratedFrom: resolved.migratedFrom,
        };
      })
      .filter((item): item is NonNullable<typeof item> => item !== null)
      .sort((a, b) => {
        const aTime = Date.parse(a.updatedAt || a.schemaUpdatedAt || "");
        const bTime = Date.parse(b.updatedAt || b.schemaUpdatedAt || "");
        return (Number.isFinite(bTime) ? bTime : 0) - (Number.isFinite(aTime) ? aTime : 0);
      });

    return { ok: true, sessions };
  } catch (e: any) {
    console.error("session:list-inference-sessions failed:", e);
    return { ok: false, error: e.message, sessions: [] };
  }
});

ipcMain.handle(
  "session:create-inference-session",
  async (_event, args: { speciesId: string; displayName?: string }) => {
    try {
      if (!args.speciesId) {
        return { ok: false, error: "speciesId is required." };
      }
      const schemaSessionJson = path.join(getSessionDir(args.speciesId), "session.json");
      if (!fs.existsSync(schemaSessionJson)) {
        return { ok: false, error: `Schema session not found: ${args.speciesId}` };
      }

      const existing = resolveCanonicalInferenceSession(args.speciesId, {
        createIfMissing: false,
      });
      if (existing.inferenceSessionId && existing.manifest) {
        return {
          ok: false,
          error: `Inference session already exists for schema ${args.speciesId}.`,
          inferenceSessionId: existing.inferenceSessionId,
          manifest: existing.manifest,
        };
      }

      const displayName =
        typeof args.displayName === "string" && args.displayName.trim()
          ? args.displayName.trim()
          : "Inference Session";
      const manifest = ensureInferenceSessionManifest({
        speciesId: args.speciesId,
        inferenceSessionId: CANONICAL_INFERENCE_SESSION_ID,
        displayName,
        detectionModelKey: "session_detection_default",
        detectionModelName: "Session Detection Model",
      });
      writeInferenceSessionIndex(args.speciesId, {
        version: 1,
        canonicalSessionId: CANONICAL_INFERENCE_SESSION_ID,
        updatedAt: new Date().toISOString(),
      });
      return {
        ok: true,
        inferenceSessionId: CANONICAL_INFERENCE_SESSION_ID,
        manifest,
      };
    } catch (e: any) {
      console.error("session:create-inference-session failed:", e);
      return { ok: false, error: e.message };
    }
  }
);

ipcMain.handle(
  "session:delete-schema-session",
  async (_event, args: { speciesId: string }) => {
    try {
      if (!args.speciesId) {
        return { ok: false, error: "speciesId is required." };
      }
      const sessionDir = getSessionDir(args.speciesId);
      if (!fs.existsSync(sessionDir)) {
        return { ok: true, deleted: false };
      }
      fs.rmSync(sessionDir, { recursive: true, force: true });
      return { ok: true, deleted: true };
    } catch (e: any) {
      console.error("session:delete-schema-session failed:", e);
      return { ok: false, error: e.message };
    }
  }
);

ipcMain.handle(
  "session:get-inference-session",
  async (_event, args: { speciesId: string }) => {
    try {
      if (!args.speciesId) {
        return { ok: false, error: "speciesId is required." };
      }
      const resolved = resolveCanonicalInferenceSession(args.speciesId, { createIfMissing: false });
      if (!resolved.inferenceSessionId || !resolved.manifest) {
        return { ok: true, exists: false };
      }
      return {
        ok: true,
        exists: true,
        inferenceSessionId: resolved.inferenceSessionId,
        manifest: resolved.manifest,
        migratedFrom: resolved.migratedFrom,
      };
    } catch (e: any) {
      console.error("session:get-inference-session failed:", e);
      return { ok: false, error: e.message };
    }
  }
);

ipcMain.handle(
  "session:update-inference-session-preferences",
  async (
    _event,
    args: {
      speciesId: string;
      inferenceSessionId?: string;
      displayName?: string;
      preferences?: {
        lastUsedLandmarkModelKey?: string;
        lastUsedPredictorType?: "dlib" | "cnn" | "yolo_pose";
        detectionModelKey?: string;
        detectionModelName?: string;
      };
    }
  ) => {
    try {
      if (!args.speciesId) {
        return { ok: false, error: "speciesId is required." };
      }
      const resolved = resolveCanonicalInferenceSession(args.speciesId, { createIfMissing: false });
      if (!resolved.inferenceSessionId || !resolved.manifest) {
        return { ok: false, error: "Inference session does not exist for this schema." };
      }
      const manifest = ensureInferenceSessionManifest({
        speciesId: args.speciesId,
        inferenceSessionId: resolved.inferenceSessionId,
        displayName: args.displayName || resolved.manifest.displayName,
        landmarkModelKey:
          args.preferences?.lastUsedLandmarkModelKey ||
          resolved.manifest.models?.landmark?.key,
        landmarkPredictorType:
          args.preferences?.lastUsedPredictorType ||
          resolved.manifest.models?.landmark?.predictorType,
        detectionModelKey:
          args.preferences?.detectionModelKey ||
          resolved.manifest.models?.detection?.key,
        detectionModelName:
          args.preferences?.detectionModelName ||
          resolved.manifest.models?.detection?.name,
        preferences: {
          lastUsedLandmarkModelKey:
            args.preferences?.lastUsedLandmarkModelKey ??
            resolved.manifest.preferences?.lastUsedLandmarkModelKey,
          lastUsedPredictorType:
            args.preferences?.lastUsedPredictorType ??
            resolved.manifest.preferences?.lastUsedPredictorType,
          detectionModelKey:
            args.preferences?.detectionModelKey ??
            resolved.manifest.preferences?.detectionModelKey,
          detectionModelName:
            args.preferences?.detectionModelName ??
            resolved.manifest.preferences?.detectionModelName,
        },
      });
      return { ok: true, inferenceSessionId: resolved.inferenceSessionId, manifest };
    } catch (e: any) {
      console.error("session:update-inference-session-preferences failed:", e);
      return { ok: false, error: e.message };
    }
  }
);

ipcMain.handle(
  "session:commit-inference-review",
  async (
    _event,
    args: { speciesId: string; inferenceSessionId?: string; onlyReviewComplete?: boolean }
  ) => {
    try {
      if (!args.speciesId) {
        return { ok: false, error: "speciesId is required." };
      }

      const resolved = resolveCanonicalInferenceSession(args.speciesId, { createIfMissing: false });
      if (!resolved.inferenceSessionId || !resolved.manifest) {
        return { ok: false, error: "Inference session does not exist for this schema." };
      }

      const drafts = readInferenceReviewDrafts(args.speciesId, resolved.inferenceSessionId);
      const items = Object.values(drafts.items);
      const onlyReviewComplete = args.onlyReviewComplete !== false;
      let committed = 0;
      let skipped = 0;
      let failed = 0;
      const failures: Array<{ filename: string; error: string }> = [];
      const now = new Date().toISOString();

      for (const item of items) {
        if (!item) continue;
        if (onlyReviewComplete && !item.reviewComplete) {
          skipped += 1;
          continue;
        }
        const updatedAtMs = Date.parse(String(item.updatedAt || ""));
        const committedAtMs = Date.parse(String(item.committedAt || ""));
        if (
          item.committedAt &&
          Number.isFinite(updatedAtMs) &&
          Number.isFinite(committedAtMs) &&
          committedAtMs >= updatedAtMs
        ) {
          skipped += 1;
          continue;
        }

        const filename = path.basename(String(item.filename || "").trim());
        const imagePathCandidate = item.imagePath
          ? path.resolve(item.imagePath)
          : path.join(getSessionDir(args.speciesId), "images", filename);
        const imagePath = fs.existsSync(imagePathCandidate)
          ? imagePathCandidate
          : path.join(getSessionDir(args.speciesId), "images", filename);
        if (!filename || !fs.existsSync(imagePath)) {
          failed += 1;
          failures.push({
            filename: filename || "(unknown)",
            error: "Image source not found for committed draft.",
          });
          continue;
        }

        const normalizedSpecimens = sanitizeDraftSpecimens(item.specimens || []);
        const commitSpecimens = normalizedSpecimens.map((s) => ({
          box: {
            left: s.box.left,
            top: s.box.top,
            width: s.box.width,
            height: s.box.height,
            orientation_override: s.box.orientation_override,
            ...(s.box.obbCorners ? { obbCorners: s.box.obbCorners } : {}),
            ...(s.box.angle != null ? { angle: s.box.angle } : {}),
            ...(s.box.class_id != null ? { class_id: s.box.class_id } : {}),
          },
          landmarks: s.landmarks.map((lm) => ({
            id: lm.id,
            x: lm.x,
            y: lm.y,
          })),
        }));

        const landmarkSave = persistInferenceCorrectionToSession({
          speciesId: args.speciesId,
          imagePath,
          filename,
          specimens: commitSpecimens,
          allowEmpty: true,
        });
        if (!landmarkSave.ok) {
          failed += 1;
          failures.push({
            filename,
            error: landmarkSave.error || "Failed to save landmark corrections.",
          });
          continue;
        }

        const boxes = commitSpecimens.map((s) => ({
          left: s.box.left,
          top: s.box.top,
          width: s.box.width,
          height: s.box.height,
          orientation_override: s.box.orientation_override,
          ...(s.box.obbCorners ? { obbCorners: s.box.obbCorners } : {}),
          ...(s.box.angle != null ? { angle: s.box.angle } : {}),
          ...(s.box.class_id != null ? { class_id: s.box.class_id } : {}),
        }));
        const dims = inferImageDimensionsForDetection(imagePath, boxes);
        const detectionSave = persistDetectionCorrectionToSession({
          speciesId: args.speciesId,
          imagePath,
          filename,
          boxes,
          imageWidth: dims.width,
          imageHeight: dims.height,
        });
        if (!detectionSave.ok) {
          failed += 1;
          failures.push({
            filename,
            error: detectionSave.error || "Failed to save detection labels.",
          });
          continue;
        }

        item.saved = true;
        item.edited = false;
        item.committedAt = now;
        committed += 1;
      }

      drafts.updatedAt = new Date().toISOString();
      writeInferenceReviewDrafts(args.speciesId, drafts, resolved.inferenceSessionId);

      return {
        ok: true,
        inferenceSessionId: resolved.inferenceSessionId,
        committed,
        skipped,
        failed,
        failures,
      };
    } catch (e: any) {
      console.error("session:commit-inference-review failed:", e);
      return { ok: false, error: e.message };
    }
  }
);

ipcMain.handle(
  "session:save-inference-review-draft",
  async (
    _event,
    args: {
      speciesId: string;
      inferenceSessionId?: string;
      imagePath: string;
      filename?: string;
      specimens?: InferenceDraftSpecimen[];
      edited?: boolean;
      saved?: boolean;
      reviewComplete?: boolean;
      committedAt?: string | null;
      landmarkModelKey?: string | null;
      landmarkPredictorType?: "dlib" | "cnn" | "yolo_pose" | null;
      boxSignature?: string | null;
      inferenceSignature?: string | null;
      clear?: boolean;
    }
  ) => {
    try {
      if (!args.speciesId) {
        return { ok: false, error: "speciesId is required." };
      }
      if (!args.imagePath && !args.filename) {
        return { ok: false, error: "imagePath or filename is required." };
      }
      if (args.inferenceSessionId) {
        ensureInferenceSessionManifest({
          speciesId: args.speciesId,
          inferenceSessionId: args.inferenceSessionId,
        });
      }
      const drafts = readInferenceReviewDrafts(args.speciesId, args.inferenceSessionId);
      const filename = args.filename ?? path.basename(args.imagePath);
      const key = buildInferenceReviewDraftKey(args.imagePath || filename, filename);

      if (args.clear) {
        delete drafts.items[key];
      } else {
        drafts.items[key] = {
          key,
          imagePath: args.imagePath || "",
          filename,
          specimens: sanitizeDraftSpecimens(args.specimens),
          edited: Boolean(args.edited),
          saved: Boolean(args.saved),
          reviewComplete:
            typeof args.reviewComplete === "boolean"
              ? args.reviewComplete
              : Boolean(drafts.items[key]?.reviewComplete),
          committedAt:
            args.committedAt === null
              ? null
              : typeof args.committedAt === "string"
              ? args.committedAt
              : drafts.items[key]?.committedAt ?? null,
          landmarkModelKey:
            args.landmarkModelKey === null
              ? undefined
              : typeof args.landmarkModelKey === "string"
              ? args.landmarkModelKey
              : drafts.items[key]?.landmarkModelKey,
          landmarkPredictorType:
            args.landmarkPredictorType === null
              ? undefined
              : args.landmarkPredictorType ??
                drafts.items[key]?.landmarkPredictorType,
          boxSignature:
            args.boxSignature === null
              ? undefined
              : typeof args.boxSignature === "string"
              ? args.boxSignature
              : drafts.items[key]?.boxSignature,
          inferenceSignature:
            args.inferenceSignature === null
              ? undefined
              : typeof args.inferenceSignature === "string"
              ? args.inferenceSignature
              : drafts.items[key]?.inferenceSignature,
          updatedAt: new Date().toISOString(),
        };
      }
      drafts.updatedAt = new Date().toISOString();
      writeInferenceReviewDrafts(args.speciesId, drafts, args.inferenceSessionId);
      return { ok: true };
    } catch (e: any) {
      console.error("session:save-inference-review-draft failed:", e);
      return { ok: false, error: e.message };
    }
  }
);

ipcMain.handle(
  "session:update-orientation-policy",
  async (
    _event,
    args: {
      speciesId: string;
      orientationPolicy: {
        mode?: "directional" | "bilateral" | "axial" | "invariant";
        targetOrientation?: "left" | "right";
        headCategories?: string[];
        tailCategories?: string[];
        bilateralPairs?: [number, number][];
        pcaLevelingMode?: "off" | "on" | "auto";
      };
    }
  ) => {
    try {
      const sessionDir = getSessionDir(args.speciesId);
      const sessionJsonPath = path.join(sessionDir, "session.json");
      if (!fs.existsSync(sessionJsonPath)) {
        return { ok: false, error: `Session not found: ${args.speciesId}` };
      }
      const meta = JSON.parse(fs.readFileSync(sessionJsonPath, "utf-8"));
      meta.orientationPolicy = args.orientationPolicy || undefined;
      meta.orientationPolicyConfigured = true;
      meta.orientationPolicyConfiguredAt = new Date().toISOString();
      meta.lastModified = new Date().toISOString();
      fs.writeFileSync(sessionJsonPath, JSON.stringify(meta, null, 2));
      return { ok: true };
    } catch (e: any) {
      console.error("session:update-orientation-policy failed:", e);
      return { ok: false, error: e.message };
    }
  }
);

ipcMain.handle(
  "session:update-augmentation",
  async (
    _event,
    args: {
      speciesId: string;
      augmentationPolicy: {
        gravity_aligned?: boolean;
        rotation_range?: [number, number];
        scale_range?: [number, number];
        flip_prob?: number;
        vertical_flip_prob?: number;
        rotate_180_prob?: number;
        translate_ratio?: number;
      };
    }
  ) => {
    try {
      const sessionDir = getSessionDir(args.speciesId);
      const sessionJsonPath = path.join(sessionDir, "session.json");
      if (!fs.existsSync(sessionJsonPath)) {
        return { ok: false, error: `Session not found: ${args.speciesId}` };
      }
      const meta = JSON.parse(fs.readFileSync(sessionJsonPath, "utf-8"));
      meta.augmentationPolicy = args.augmentationPolicy;
      meta.lastModified = new Date().toISOString();
      fs.writeFileSync(sessionJsonPath, JSON.stringify(meta, null, 2));
      return { ok: true };
    } catch (e: any) {
      console.error("session:update-augmentation failed:", e);
      return { ok: false, error: e.message };
    }
  }
);

ipcMain.handle(
  "session:load-inference-review-drafts",
  async (_event, args: { speciesId: string; inferenceSessionId?: string }) => {
    try {
      if (!args.speciesId) {
        return { ok: false, error: "speciesId is required.", drafts: [] };
      }
      if (args.inferenceSessionId) {
        ensureInferenceSessionManifest({
          speciesId: args.speciesId,
          inferenceSessionId: args.inferenceSessionId,
        });
        migrateLegacyInferenceArtifactsToSession(args.speciesId, args.inferenceSessionId);
      }
      const drafts = readInferenceReviewDrafts(args.speciesId, args.inferenceSessionId);
      return { ok: true, drafts: Object.values(drafts.items) };
    } catch (e: any) {
      console.error("session:load-inference-review-drafts failed:", e);
      return { ok: false, error: e.message, drafts: [] };
    }
  }
);

ipcMain.handle(
  "session:save-inference-image-paths",
  async (_event, args: { speciesId: string; inferenceSessionId: string; imagePaths: { path: string; name: string }[] }) => {
    try {
      if (!args.speciesId || !args.inferenceSessionId) return { ok: false, error: "speciesId and inferenceSessionId are required." };
      const sessionDir = getInferenceSessionDir(args.speciesId, args.inferenceSessionId);
      fs.mkdirSync(sessionDir, { recursive: true });
      fs.writeFileSync(
        path.join(sessionDir, "image_paths.json"),
        JSON.stringify({ version: 1, imagePaths: args.imagePaths }, null, 2),
        "utf-8"
      );
      return { ok: true };
    } catch (e: any) {
      console.error("session:save-inference-image-paths failed:", e);
      return { ok: false, error: e.message };
    }
  }
);

ipcMain.handle(
  "session:load-inference-image-paths",
  async (_event, args: { speciesId: string; inferenceSessionId: string }) => {
    try {
      if (!args.speciesId || !args.inferenceSessionId) return { ok: true, images: [] };
      const listPath = path.join(
        getInferenceSessionDir(args.speciesId, args.inferenceSessionId),
        "image_paths.json"
      );
      if (!fs.existsSync(listPath)) return { ok: true, images: [] };
      const { imagePaths } = JSON.parse(fs.readFileSync(listPath, "utf-8")) as {
        imagePaths: { path: string; name: string }[];
      };
      const images = imagePaths
        .filter((p) => fs.existsSync(p.path))
        .map((p) => {
          const data = fs.readFileSync(p.path).toString("base64");
          const ext = path.extname(p.name).toLowerCase().slice(1);
          const MIME_MAP: Record<string, string> = {
            png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg",
            webp: "image/webp", bmp: "image/bmp", gif: "image/gif",
            tiff: "image/tiff", tif: "image/tiff",
          };
          const mimeType = MIME_MAP[ext] ?? "image/jpeg";
          return { path: p.path, name: p.name, data, mimeType };
        });
      return { ok: true, images };
    } catch (e: any) {
      console.error("session:load-inference-image-paths failed:", e);
      return { ok: true, images: [] };
    }
  }
);

// Legacy retrain-queue IPC handlers are intentionally retained for compatibility.
// Current renderer workflow commits review-complete items directly to training data.
ipcMain.handle(
  "session:queue-retrain-item",
  async (
    _event,
    args: {
      speciesId: string;
      inferenceSessionId?: string;
      landmarkModelKey?: string;
      landmarkModelName?: string;
      landmarkPredictorType?: "dlib" | "cnn" | "yolo_pose";
      detectionModelKey?: string;
      detectionModelName?: string;
      filename: string;
      imagePath?: string;
      source?: string;
      boxesCount?: number;
      landmarksCount?: number;
    }
  ) => {
    try {
      if (!args.speciesId) {
        return { ok: false, error: "speciesId is required." };
      }
      if (!args.filename) {
        return { ok: false, error: "filename is required." };
      }
      if (args.inferenceSessionId) {
        ensureInferenceSessionManifest({
          speciesId: args.speciesId,
          inferenceSessionId: args.inferenceSessionId,
          landmarkModelKey: args.landmarkModelKey,
          landmarkModelName: args.landmarkModelName,
          landmarkPredictorType: args.landmarkPredictorType,
          detectionModelKey: args.detectionModelKey,
          detectionModelName: args.detectionModelName,
        });
      }
      const queue = readRetrainQueue(args.speciesId, args.inferenceSessionId);
      const now = new Date().toISOString();
      const key = buildRetrainQueueKey(args.filename);
      const existing = queue.items[key];
      queue.items[key] = {
        key,
        speciesId: args.speciesId,
        inferenceSessionId: args.inferenceSessionId,
        landmarkModelKey: args.landmarkModelKey || existing?.landmarkModelKey,
        landmarkModelName: args.landmarkModelName || existing?.landmarkModelName,
        landmarkPredictorType:
          args.landmarkPredictorType || existing?.landmarkPredictorType,
        detectionModelKey: args.detectionModelKey || existing?.detectionModelKey,
        detectionModelName: args.detectionModelName || existing?.detectionModelName,
        filename: args.filename,
        imagePath: args.imagePath || existing?.imagePath || "",
        source: args.source || existing?.source || "inference_review",
        boxesCount: Math.max(0, Math.round(Number(args.boxesCount) || existing?.boxesCount || 0)),
        landmarksCount: Math.max(0, Math.round(Number(args.landmarksCount) || existing?.landmarksCount || 0)),
        queuedAt: existing?.queuedAt || now,
        updatedAt: now,
      };
      queue.updatedAt = now;
      writeRetrainQueue(args.speciesId, queue, args.inferenceSessionId);

      return {
        ok: true,
        item: queue.items[key],
        queuedCount: Object.keys(queue.items).length,
      };
    } catch (e: any) {
      console.error("session:queue-retrain-item failed:", e);
      return { ok: false, error: e.message };
    }
  }
);

ipcMain.handle(
  "session:get-retrain-queue",
  async (_event, args: { speciesId: string; inferenceSessionId?: string }) => {
    try {
      if (!args.speciesId) {
        return { ok: false, error: "speciesId is required.", items: [], count: 0 };
      }
      if (args.inferenceSessionId) {
        ensureInferenceSessionManifest({
          speciesId: args.speciesId,
          inferenceSessionId: args.inferenceSessionId,
        });
        migrateLegacyInferenceArtifactsToSession(args.speciesId, args.inferenceSessionId);
      }
      const queue = readRetrainQueue(args.speciesId, args.inferenceSessionId);
      return { ok: true, items: Object.values(queue.items), count: Object.keys(queue.items).length };
    } catch (e: any) {
      console.error("session:get-retrain-queue failed:", e);
      return { ok: false, error: e.message, items: [], count: 0 };
    }
  }
);

ipcMain.handle(
  "session:clear-retrain-queue",
  async (_event, args: { speciesId: string; inferenceSessionId?: string; filenames?: string[] }) => {
    try {
      if (!args.speciesId) {
        return { ok: false, error: "speciesId is required.", count: 0 };
      }
      if (args.inferenceSessionId) {
        ensureInferenceSessionManifest({
          speciesId: args.speciesId,
          inferenceSessionId: args.inferenceSessionId,
        });
      }
      const queue = readRetrainQueue(args.speciesId, args.inferenceSessionId);
      if (Array.isArray(args.filenames) && args.filenames.length > 0) {
        const keys = new Set(args.filenames.map((name) => buildRetrainQueueKey(name)));
        for (const key of keys) {
          delete queue.items[key];
        }
      } else {
        queue.items = {};
      }
      queue.updatedAt = new Date().toISOString();
      writeRetrainQueue(args.speciesId, queue, args.inferenceSessionId);
      return { ok: true, count: Object.keys(queue.items).length };
    } catch (e: any) {
      console.error("session:clear-retrain-queue failed:", e);
      return { ok: false, error: e.message, count: 0 };
    }
  }
);

// Legacy IPC handler kept for backward compatibility with older renderer builds.
// Current renderer commits detection updates through session:commit-inference-review.
ipcMain.handle(
  "session:save-detection-correction",
  async (
    _event,
    args: {
      speciesId: string;
      imagePath: string;
      boxes: { left: number; top: number; width: number; height: number }[];
      imageWidth: number;
      imageHeight: number;
      filename?: string;
    }
  ) => {
    try {
      const result = persistDetectionCorrectionToSession(args);
      if (!result.ok) {
        return { ok: false, error: result.error || "Failed to save detection correction." };
      }
      return { ok: true, savedPath: result.savedPath };
    } catch (e: any) {
      console.error("session:save-detection-correction failed:", e);
      return { ok: false, error: e.message };
    }
  }
);

ipcMain.handle(
  "session:delete-image",
  async (_event, args: { speciesId: string; filename: string }) => {
    try {
      const sessionDir = getSessionDir(args.speciesId);
      const imagePath = path.join(sessionDir, "images", args.filename);
      const labelPath = path.join(
        sessionDir,
        "labels",
        args.filename.replace(/\.\w+$/, ".json")
      );

      if (fs.existsSync(imagePath)) fs.unlinkSync(imagePath);
      if (fs.existsSync(labelPath)) fs.unlinkSync(labelPath);

      // Remove persisted inference-review drafts for this image (legacy root + canonical session).
      const lowerName = (args.filename || "").toLowerCase();
      cancelQueuedSegmentSave(args.speciesId, args.filename);
      clearFinalizedSegmentCache(args.speciesId, args.filename);
      try {
        const drafts = readInferenceReviewDrafts(args.speciesId);
        for (const [key, item] of Object.entries(drafts.items)) {
          if ((item?.filename || "").toLowerCase() === lowerName) {
            delete drafts.items[key];
          }
        }
        drafts.updatedAt = new Date().toISOString();
        writeInferenceReviewDrafts(args.speciesId, drafts);
      } catch (_) {
        // non-critical cleanup
      }
      try {
        const sessionDrafts = readInferenceReviewDrafts(args.speciesId, CANONICAL_INFERENCE_SESSION_ID);
        for (const [key, item] of Object.entries(sessionDrafts.items)) {
          if ((item?.filename || "").toLowerCase() === lowerName) delete sessionDrafts.items[key];
        }
        sessionDrafts.updatedAt = new Date().toISOString();
        writeInferenceReviewDrafts(args.speciesId, sessionDrafts, CANONICAL_INFERENCE_SESSION_ID);
      } catch (_) {
        // non-critical cleanup
      }
      try {
        const pathsFile = path.join(getInferenceSessionDir(args.speciesId, CANONICAL_INFERENCE_SESSION_ID), "image_paths.json");
        if (fs.existsSync(pathsFile)) {
          const { imagePaths } = JSON.parse(fs.readFileSync(pathsFile, "utf-8"));
          const filtered = (imagePaths as Array<{ path: string }>).filter(
            (p) => path.basename(p.path).toLowerCase() !== lowerName
          );
          fs.writeFileSync(pathsFile, JSON.stringify({ imagePaths: filtered }, null, 2));
        }
      } catch (_) {
        // non-critical cleanup
      }

      // Remove retrain queue entry for this image.
      try {
        const queue = readRetrainQueue(args.speciesId);
        delete queue.items[buildRetrainQueueKey(args.filename)];
        queue.updatedAt = new Date().toISOString();
        writeRetrainQueue(args.speciesId, queue);
      } catch (_) {
        // non-critical cleanup
      }

      // Update session.json imageCount
      const sessionJsonPath = path.join(sessionDir, "session.json");
      if (fs.existsSync(sessionJsonPath)) {
        try {
          const meta = JSON.parse(fs.readFileSync(sessionJsonPath, "utf-8"));
          const imagesDir = path.join(sessionDir, "images");
          const imageFiles = fs.existsSync(imagesDir)
            ? fs.readdirSync(imagesDir).filter((f) => IMAGE_EXTS.test(f))
            : [];
          meta.imageCount = imageFiles.length;
          meta.lastModified = new Date().toISOString();
          fs.writeFileSync(sessionJsonPath, JSON.stringify(meta, null, 2));
        } catch (_) {
          // non-critical
        }
      }

      return { ok: true };
    } catch (e: any) {
      console.error("session:delete-image failed:", e);
      return { ok: false, error: e.message };
    }
  }
);

// Ã¢â€â‚¬Ã¢â€â‚¬ SuperAnnotator persistent process manager Ã¢â€â‚¬Ã¢â€â‚¬

class SuperAnnotatorProcess {
  private process: ChildProcess | null = null;
  private rl: ReadlineInterface | null = null;
  private pending: Map<
    string,
    {
      resolve: (v: any) => void;
      reject: (e: Error) => void;
      cmdName: string;
      startedAt: number;
      timeoutMs: number;
      timeout: ReturnType<typeof setTimeout>;
    }
  > = new Map();
  private idleTimer: ReturnType<typeof setTimeout> | null = null;
  private requestId = 0;
  /** True after a successful `init` command Ã¢â‚¬â€ models are loaded and ready. */
  initCompleted = false;
  private readonly IDLE_TIMEOUT_MS = 10 * 60 * 1000; // 10 minutes
  private stderrTail: string[] = [];
  private readonly MAX_STDERR_TAIL_LINES = 60;

  async start(): Promise<void> {
    if (this.process) return; // already running

    const pyPath = getPythonPath();
    const scriptPath = path.join(__dirname, "../backend/annotation/super_annotator.py");

    this.process = spawn(pyPath, [scriptPath], {
      stdio: ["pipe", "pipe", "pipe"],
      cwd: path.join(__dirname, ".."),
      // Force UTF-8 for stdin/stdout so non-ASCII paths (e.g. macOS narrow
      // no-break space U+202F in screenshot filenames) survive the JSON pipe.
      // Without this, Windows Python defaults to CP1252 and mangles the path.
      env: { ...process.env, PYTHONUTF8: "1", PYTHONIOENCODING: "utf-8" },
    });

    this.process.stderr?.on("data", (d: Buffer) => {
      const text = d.toString();
      this.pushStderr(text);
      console.log("[SuperAnnotator]", text.trim());
    });

    // Absorb EPIPE and other stdin write errors so they don't surface as an
    // uncaught exception and crash the main process.  The 'close' handler
    // already rejects all pending requests when the subprocess dies, so
    // there is nothing else to do here.
    this.process.stdin?.on("error", (err: NodeJS.ErrnoException) => {
      if (err.code === "EPIPE" || err.code === "ERR_STREAM_DESTROYED") {
        console.warn("[SuperAnnotator] stdin write error (process likely died):", err.code);
      } else {
        console.error("[SuperAnnotator] stdin error:", err);
      }
    });

    this.rl = createInterface({ input: this.process.stdout! });
    this.rl.on("line", (line: string) => {
      try {
        const msg = JSON.parse(line);

        // Forward progress events to renderer
        if (msg.status === "progress") {
          if (msg._request_id) {
            this.refreshRequestTimeout(String(msg._request_id));
          }
          this.resetIdleTimer();
          mainWindow?.webContents.send("ml:super-annotate-progress", msg);
          return;
        }

        // Match response to pending request by explicit request id.
        if (msg._request_id) {
          const requestId = String(msg._request_id);
          const entry = this.pending.get(requestId);
          if (!entry) {
            console.warn(`[SuperAnnotator] Unmatched response for request ${requestId}; ignoring stale message.`);
            return;
          }
          clearTimeout(entry.timeout);
          this.pending.delete(requestId);
          entry.resolve(msg);
          this.resetIdleTimer();
          return;
        }

        // Backward-compatible fallback for responses without _request_id.
        if (this.pending.size === 1) {
          const [firstId, handler] = this.pending.entries().next().value!;
          clearTimeout(handler.timeout);
          this.pending.delete(firstId);
          handler.resolve(msg);
          this.resetIdleTimer();
          return;
        }
        console.warn("[SuperAnnotator] Received response without _request_id while multiple requests are pending; ignoring.");
      } catch (e) {
        const trimmed = line.trim();
        if (!trimmed) return;
        // Ultralytics may print plain text (e.g. requirements/autoupdate notices) to stdout.
        // Keep logs visible but do not treat them as protocol errors.
        console.log("[SuperAnnotator stdout]", trimmed);
      }
    });

    this.process.on("close", (code: number | null, signal: NodeJS.Signals | null) => {
      const stderrTail = this.getLastStderrTail();
      const suffix = stderrTail ? `\nRecent stderr:\n${stderrTail}` : "";
      console.log(`[SuperAnnotator] Process exited with code ${code}, signal ${signal}`);
      // Reject all pending
      for (const [, handler] of this.pending) {
        clearTimeout(handler.timeout);
        handler.reject(
          new Error(
            `SuperAnnotator process exited (code=${code}, signal=${signal}) while handling "${handler.cmdName}" after ${Math.round((Date.now() - handler.startedAt) / 1000)}s.${suffix}`
          )
        );
      }
      this.pending.clear();
      this.process = null;
      this.rl = null;
      this.initCompleted = false; // models must be reloaded on next start
    });

    this.process.on("error", (err: Error) => {
      console.error("[SuperAnnotator] Child process error:", err);
    });

    this.resetIdleTimer();
  }

  async send(cmd: Record<string, unknown>): Promise<any> {
    if (!this.process) {
      await this.start();
    }

    this.resetIdleTimer();

    const id = `req_${++this.requestId}`;
    const payload = { ...cmd, _request_id: id };
    const cmdName = String(cmd?.cmd ?? "unknown");
    const timeoutMs = this.getRequestTimeoutMs(cmdName);

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        const entry = this.pending.get(id);
        if (!entry) return;
        this.pending.delete(id);
        entry.reject(
          new Error(
            `SuperAnnotator request timed out after ${Math.round(timeoutMs / 1000)}s (cmd="${cmdName}").`
          )
        );
        this.resetIdleTimer();
      }, timeoutMs);

      this.pending.set(id, {
        resolve,
        reject,
        cmdName,
        startedAt: Date.now(),
        timeoutMs,
        timeout,
      });

      try {
        if (!this.process?.stdin || this.process.stdin.destroyed) {
          throw new Error("stdin is closed or destroyed");
        }
        this.process.stdin.write(JSON.stringify(payload) + "\n");
      } catch (err: any) {
        clearTimeout(timeout);
        this.pending.delete(id);
        reject(new Error(`Failed to write SuperAnnotator command "${cmdName}": ${err?.message || err}`));
      }
    });
  }

  async stop(): Promise<void> {
    if (this.idleTimer) {
      clearTimeout(this.idleTimer);
      this.idleTimer = null;
    }
    if (!this.process) return;
    try {
      if (!this.process.stdin?.destroyed) {
        this.process.stdin!.write(JSON.stringify({ cmd: "shutdown" }) + "\n");
      }
    } catch (_) {
      // process may already be dead
    }
    // Give it 3s to shutdown gracefully, then kill
    await new Promise<void>((resolve) => {
      const timeout = setTimeout(() => {
        this.process?.kill("SIGKILL");
        resolve();
      }, 3000);
      this.process!.on("close", () => {
        clearTimeout(timeout);
        resolve();
      });
    });
    this.process = null;
    this.rl = null;
  }

  private resetIdleTimer(): void {
    if (this.idleTimer) clearTimeout(this.idleTimer);
    this.idleTimer = setTimeout(() => {
      if (this.pending.size > 0) {
        console.log(`[SuperAnnotator] Idle timeout deferred: ${this.pending.size} request(s) still pending`);
        this.resetIdleTimer();
        return;
      }
      console.log("[SuperAnnotator] Idle timeout, shutting down process");
      this.stop();
    }, this.IDLE_TIMEOUT_MS);
  }

  private pushStderr(text: string): void {
    const lines = text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);
    for (const line of lines) {
      this.stderrTail.push(line);
      if (this.stderrTail.length > this.MAX_STDERR_TAIL_LINES) {
        this.stderrTail.shift();
      }
    }
  }

  private getLastStderrTail(maxLines = 25): string {
    if (this.stderrTail.length === 0) return "";
    return this.stderrTail.slice(-maxLines).join("\n");
  }

  private getRequestTimeoutMs(cmdName: string): number {
    if (cmdName === "annotate") return 15 * 60 * 1000;
    if (cmdName === "init" || cmdName === "check") return 2 * 60 * 1000;
    return 5 * 60 * 1000;
  }

  private refreshRequestTimeout(requestId: string): void {
    const entry = this.pending.get(requestId);
    if (!entry) return;
    clearTimeout(entry.timeout);
    entry.timeout = setTimeout(() => {
      const active = this.pending.get(requestId);
      if (!active) return;
      this.pending.delete(requestId);
      active.reject(
        new Error(
          `SuperAnnotator request timed out after ${Math.round(active.timeoutMs / 1000)}s (cmd="${active.cmdName}").`
        )
      );
      this.resetIdleTimer();
    }, entry.timeoutMs);
  }

  get isRunning(): boolean {
    return this.process !== null;
  }
}

const superAnnotator = new SuperAnnotatorProcess();

const SAM2_MINIMUM_REQUIREMENTS_CACHE_TTL_MS = 30_000;
let sam2MinimumRequirementsCache:
  | { ok: boolean; error?: string; checkedAtMs: number }
  | null = null;

async function checkSam2MinimumRequirements(force = false): Promise<{ ok: boolean; error?: string }> {
  const now = Date.now();
  if (
    !force &&
    sam2MinimumRequirementsCache &&
    now - sam2MinimumRequirementsCache.checkedAtMs <= SAM2_MINIMUM_REQUIREMENTS_CACHE_TTL_MS
  ) {
    return { ok: sam2MinimumRequirementsCache.ok, error: sam2MinimumRequirementsCache.error };
  }

  try {
    const hwScript = path.join(__dirname, "../backend/hardware_probe.py");
    const hwOut = await runPython([hwScript]);
    const hw = JSON.parse(hwOut.trim());
    const device = String(hw?.device ?? "cpu");
    const ramGb = Number(hw?.ram_gb ?? 0);
    const ok = device !== "cpu" && ramGb >= 8;
    const error = ok
      ? undefined
      : `SAM2 requires non-CPU acceleration and at least 8 GB RAM. Detected device=${device}, RAM=${Number.isFinite(ramGb) ? ramGb : 0} GB.`;
    sam2MinimumRequirementsCache = { ok, error, checkedAtMs: now };
    return { ok, error };
  } catch (e: any) {
    const error = e?.message || "Failed to verify SAM2 hardware requirements.";
    sam2MinimumRequirementsCache = { ok: false, error, checkedAtMs: now };
    return { ok: false, error };
  }
}

function persistFinalizedDetection(
  sessionDir: string,
  speciesId: string,
  filename: string,
  acceptedBoxes: FinalizedAcceptedBox[],
  signature: string
): void {
  const labelsDir = path.join(sessionDir, "labels");
  fs.mkdirSync(labelsDir, { recursive: true });
  const labelPath = path.join(labelsDir, filename.replace(/\.\w+$/, ".json"));
  let payload: any = {
    imageFilename: filename,
    speciesId,
    boxes: [],
    rejectedDetections: [],
  };
  if (fs.existsSync(labelPath)) {
    try {
      payload = JSON.parse(fs.readFileSync(labelPath, "utf-8"));
    } catch {
      // keep default payload
    }
  }
  payload.imageFilename = payload.imageFilename || filename;
  payload.speciesId = payload.speciesId || speciesId;
  payload.boxes = Array.isArray(payload.boxes) ? payload.boxes : [];
  payload.rejectedDetections = Array.isArray(payload.rejectedDetections) ? payload.rejectedDetections : [];
  payload.finalizedDetection = {
    isFinalized: true,
    finalizedAt: new Date().toISOString(),
    acceptedBoxes,
    boxSignature: signature,
  };
  fs.writeFileSync(labelPath, JSON.stringify(payload, null, 2));

  const existing = readFinalizedList(sessionDir);
  const lower = path.basename(filename).toLowerCase();
  const alreadyFinalized = existing.some(
    (name) => path.basename(String(name || "")).toLowerCase() === lower
  );
  if (!alreadyFinalized) {
    existing.push(filename);
    writeFinalizedList(sessionDir, existing);
  }
}

async function ensureSam2Ready(): Promise<{ ok: boolean; error?: string }> {
  try {
    const minimumRequirements = await checkSam2MinimumRequirements();
    if (!minimumRequirements.ok) {
      return minimumRequirements;
    }
    if (!superAnnotator.isRunning) {
      const initRes = await superAnnotator.send({ cmd: "init" });
      if (initRes?.sam2_loaded) {
        superAnnotator.initCompleted = true;
        kickAllSegmentSaveQueues();
        return { ok: true };
      }
      const checkRes = await superAnnotator.send({ cmd: "check" });
      return { ok: false, error: checkRes?.sam2_error || "SAM2 is not available." };
    }

    const checkRes = await superAnnotator.send({ cmd: "check" });
    if (checkRes?.sam2_ready) {
      return { ok: true };
    }

    const initRes = await superAnnotator.send({ cmd: "init" });
    if (initRes?.sam2_loaded) {
      superAnnotator.initCompleted = true;
      kickAllSegmentSaveQueues();
      return { ok: true };
    }
    const recheck = await superAnnotator.send({ cmd: "check" });
    return { ok: false, error: recheck?.sam2_error || "SAM2 is not available." };
  } catch (e: any) {
    return { ok: false, error: e?.message || "Failed to initialize SAM2." };
  }
}

async function deriveImportedBoxGeometryWithSam2(
  imagePath: string,
  box: ImportedNormalizedBox,
  imageDims: { width: number; height: number } | null,
  geometryConfig?: ImportGeometryConfig
): Promise<Pick<ImportedNormalizedBox, "left" | "top" | "width" | "height" | "maskOutline" | "obbCorners" | "angle"> | null> {
  const resolvedGeometry = normalizeImportGeometryConfig(geometryConfig);
  const initialBox = buildImportSam2PromptBox(box, imageDims, geometryConfig);
  const expandRatio = Math.max(
    resolvedGeometry.paddingProfile.forward,
    resolvedGeometry.paddingProfile.backward,
    resolvedGeometry.paddingProfile.top,
    resolvedGeometry.paddingProfile.bottom,
    0.10
  );

  const result = await superAnnotator.send({
    cmd: "resegment_box",
    image_path: imagePath,
    box_xyxy: initialBox,
    iterative: true,
    expand_ratio: expandRatio,
  });

  if (!result?.ok) {
    return null;
  }

  const normalizePointPairs = (points: unknown): [number, number][] | undefined => {
    if (!Array.isArray(points)) return undefined;
    const normalized = points
      .map((point) => {
        if (!Array.isArray(point) || point.length < 2) return null;
        const x = Number(point[0]);
        const y = Number(point[1]);
        if (!isFiniteNumber(x) || !isFiniteNumber(y)) return null;
        return [Math.round(x), Math.round(y)] as [number, number];
      })
      .filter((point): point is [number, number] => Boolean(point));
    return normalized.length > 0 ? normalized : undefined;
  };

  const maskOutline = normalizePointPairs(result.mask_outline);
  const obbCorners = normalizePointPairs(result.obb_corners);
  const resultBoxXyxy =
    Array.isArray(result.box_xyxy) &&
    result.box_xyxy.length === 4 &&
    result.box_xyxy.every((value: unknown) => isFiniteNumber(Number(value)))
      ? (result.box_xyxy.map((value: unknown) => Math.round(Number(value))) as [number, number, number, number])
      : null;

  let left: number;
  let top: number;
  let width: number;
  let height: number;

  if (resultBoxXyxy) {
    left = resultBoxXyxy[0];
    top = resultBoxXyxy[1];
    width = Math.max(1, resultBoxXyxy[2] - resultBoxXyxy[0]);
    height = Math.max(1, resultBoxXyxy[3] - resultBoxXyxy[1]);
  } else if (obbCorners && obbCorners.length === 4) {
    const aabb = importedCornersToAabb(obbCorners, null);
    left = aabb.left;
    top = aabb.top;
    width = aabb.width;
    height = aabb.height;
  } else {
    return null;
  }

  return {
    left,
    top,
    width,
    height,
    ...(maskOutline ? { maskOutline } : {}),
    ...(obbCorners && obbCorners.length === 4 ? { obbCorners } : {}),
    ...(isFiniteNumber(result.angle) ? { angle: Number(result.angle) } : {}),
  };
}

function normalizeSessionStoredBoxes(boxes: any[]): any[] {
  if (!Array.isArray(boxes)) return [];
  const usedIds = new Set<number>();
  let nextFallbackId = 0;

  return boxes.map((box: any, index: number) => {
    let id = isFiniteNumber(box?.id) ? Math.round(Number(box.id)) : index;
    while (usedIds.has(id)) {
      id = nextFallbackId;
      nextFallbackId += 1;
    }
    usedIds.add(id);
    nextFallbackId = Math.max(nextFallbackId, id + 1);
    return { ...box, id };
  });
}

type LandmarkWorkerProgress = {
  percent: number;
  stage: string;
  current_specimen?: number;
  total_specimens?: number;
};

class LandmarkInferenceWorkerProcess {
  private process: ChildProcess | null = null;
  private rl: ReadlineInterface | null = null;
  private requestId = 0;
  private pending = new Map<
    string,
    {
      resolve: (value: any) => void;
      reject: (error: Error) => void;
      onProgress?: (data: LandmarkWorkerProgress) => void;
      timeout: ReturnType<typeof setTimeout>;
    }
  >();
  private idleTimer: ReturnType<typeof setTimeout> | null = null;
  private readonly IDLE_TIMEOUT_MS = 10 * 60 * 1000;

  async start(): Promise<void> {
    if (this.process) return;
    const pyPath = getPythonPath();
    const scriptPath = path.join(__dirname, "../backend/inference/predict_worker.py");
    this.process = spawn(pyPath, [scriptPath], {
      stdio: ["pipe", "pipe", "pipe"],
      cwd: path.join(__dirname, ".."),
      env: { ...process.env, PYTHONUTF8: "1", PYTHONIOENCODING: "utf-8" },
    });

    this.process.stderr?.on("data", (d: Buffer) => {
      console.log("[LandmarkWorker]", d.toString().trim());
    });

    this.process.stdin?.on("error", () => {
      // close handler rejects pending work
    });

    this.rl = createInterface({ input: this.process.stdout! });
    this.rl.on("line", (line: string) => {
      try {
        const msg = JSON.parse(line);
        const requestId = String(msg?._request_id || "");
        if (msg?.status === "progress") {
          const entry = this.pending.get(requestId);
          entry?.onProgress?.({
            percent: Number(msg.percent) || 0,
            stage: String(msg.stage || "progress"),
            current_specimen: Number.isFinite(Number(msg.current_specimen)) ? Number(msg.current_specimen) : undefined,
            total_specimens: Number.isFinite(Number(msg.total_specimens)) ? Number(msg.total_specimens) : undefined,
          });
          this.resetIdleTimer();
          return;
        }
        const entry = this.pending.get(requestId);
        if (!entry) return;
        clearTimeout(entry.timeout);
        this.pending.delete(requestId);
        if (msg?.status === "error" || msg?.ok === false) {
          entry.reject(new Error(String(msg?.error || "Landmark worker request failed.")));
        } else {
          entry.resolve(msg?.data);
        }
        this.resetIdleTimer();
      } catch (error) {
        console.error("[LandmarkWorker] Failed to parse response:", error);
      }
    });

    this.process.on("close", (code) => {
      const error = new Error(`Landmark worker exited with code ${code ?? -1}`);
      for (const [, entry] of this.pending) {
        clearTimeout(entry.timeout);
        entry.reject(error);
      }
      this.pending.clear();
      this.cleanup();
    });

    this.resetIdleTimer();
  }

  async stop(): Promise<void> {
    if (!this.process) return;
    try {
      await this.sendRequest("shutdown", {}, 2000);
    } catch {
      // ignore best-effort shutdown
    } finally {
      this.process.kill();
      this.cleanup();
    }
  }

  async predict(
    payload: {
      project_root: string;
      tag: string;
      predictor_type: "dlib" | "cnn";
      image_path: string;
      boxes: NonNullable<PredictOptions["boxes"]>;
    },
    onProgress?: (data: LandmarkWorkerProgress) => void
  ): Promise<any> {
    return this.sendRequest("predict", payload, 5 * 60 * 1000, onProgress);
  }

  private async sendRequest(
    cmd: string,
    payload: Record<string, unknown>,
    timeoutMs: number,
    onProgress?: (data: LandmarkWorkerProgress) => void
  ): Promise<any> {
    await this.start();
    if (!this.process?.stdin?.writable) {
      throw new Error("Landmark worker is not writable.");
    }
    const requestId = `lw_${Date.now()}_${++this.requestId}`;
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pending.delete(requestId);
        reject(new Error(`Landmark worker request timed out for ${cmd}.`));
      }, timeoutMs);
      this.pending.set(requestId, { resolve, reject, onProgress, timeout });
      this.process!.stdin!.write(
        JSON.stringify({ cmd, _request_id: requestId, ...payload }) + "\n",
        "utf8"
      );
      this.resetIdleTimer();
    });
  }

  private resetIdleTimer(): void {
    if (this.idleTimer) clearTimeout(this.idleTimer);
    this.idleTimer = setTimeout(() => {
      void this.stop();
    }, this.IDLE_TIMEOUT_MS);
  }

  private cleanup(): void {
    if (this.rl) {
      this.rl.removeAllListeners();
      this.rl.close();
    }
    this.rl = null;
    this.process = null;
    if (this.idleTimer) clearTimeout(this.idleTimer);
    this.idleTimer = null;
  }
}

const landmarkInferenceWorker = new LandmarkInferenceWorkerProcess();

// Ã¢â€â‚¬Ã¢â€â‚¬ SuperAnnotator IPC handlers Ã¢â€â‚¬Ã¢â€â‚¬

/**
 * Fast, synchronous capability estimate using Node.js OS APIs.
 * Used when the Python process isn't running yet to avoid blocking the UI
 * on a cold Python startup just to show the capability badges.
 * GPU presence is unknown without Python; mode may be upgraded to
 * auto_high_performance once the real check runs after init.
 */
function getLocalCapabilityEstimate() {
  const freeRamGb = os.freemem() / (1024 ** 3);
  const mode = freeRamGb > 1.0 ? "auto_lite" : "classic_fallback";
  return {
    available: true,
    mode,
    gpu: false,
    yolo_ready: false,
    sam2_ready: false,
    yolo_failed: false,
    sam2_failed: false,
  };
}

ipcMain.handle("ml:check-super-annotator", async () => {
  // If the process isn't running yet, return an instant local estimate so
  // the UI renders immediately without waiting for a cold Python startup.
  if (!superAnnotator.isRunning) {
    return getLocalCapabilityEstimate();
  }
  try {
    const result = await superAnnotator.send({ cmd: "check" });
    return result;
  } catch (e: any) {
    return {
      available: true,
      mode: "classic_fallback",
      gpu: false,
      yolo_ready: false,
      sam2_ready: false,
      yolo_failed: false,
      sam2_failed: false,
      error: e.message,
    };
  }
});

ipcMain.handle("ml:init-super-annotator", async () => {
  try {
    const result = await superAnnotator.send({ cmd: "init" });
    if (result?.sam2_loaded) {
      superAnnotator.initCompleted = true;
      kickAllSegmentSaveQueues();
    }
    return { ok: true, ...result };
  } catch (e: any) {
    return { ok: false, error: e.message };
  }
});

ipcMain.handle("ml:train-obb-detector", async (_event, speciesId: string, options?: { epochs?: number; modelTier?: "nano" | "small"; iou?: number; cls?: number; box?: number }) => {
  try {
    if (!superAnnotator.isRunning) {
      const initResult = await superAnnotator.send({ cmd: "init" });
      if (initResult?.sam2_loaded) {
        superAnnotator.initCompleted = true;
        kickAllSegmentSaveQueues();
      }
    }
    const sessionDir = getSessionDir(speciesId);
    if (!fs.existsSync(sessionDir)) {
      return { ok: false, error: `Session directory not found: ${sessionDir}` };
    }

    // Get hardware tier; user-provided modelTier takes precedence
    const caps = await superAnnotator.send({ cmd: "check" });
    const modelTier = options?.modelTier ?? caps?.obb_model_tier ?? "nano";

    // Probe hardware to determine device + SAM2 availability for routing
    let hwDevice = "cpu";
    let hwSam2Enabled = false;
    try {
      const hwScript = path.join(__dirname, "../backend/hardware_probe.py");
      const hwOut = await runPython([hwScript]);
      const hw = JSON.parse(hwOut.trim());
      hwDevice = hw.device ?? "cpu";
      hwSam2Enabled = hwDevice !== "cpu" && (hw.ram_gb ?? 0) >= 8;
    } catch (_hwErr) {
      console.warn("Hardware probe failed during OBB training Ã¢â‚¬â€ defaulting to cpu");
    }

    // Read orientation schema from session.json
    let orientationSchema = "invariant";
    const sessionJsonPath = path.join(sessionDir, "session.json");
    if (fs.existsSync(sessionJsonPath)) {
      try {
        const session = safeReadJson(sessionJsonPath) ?? {};
        orientationSchema = (session as any).orientationPolicy?.mode ?? "invariant";
      } catch (_) { /* non-fatal */ }
    }

    const result = await superAnnotator.send({
      cmd: "train_yolo_obb",
      session_dir: sessionDir,
      epochs: options?.epochs,   // undefined Ã¢â€ â€™ Python selects hardware-appropriate default
      model_tier: modelTier,
      device: hwDevice,
      sam2_enabled: hwSam2Enabled,
      iou_loss: options?.iou ?? 0.3,
      cls_loss: options?.cls ?? 1.5,
      box_loss: options?.box ?? 5.0,
      orientation_schema: orientationSchema,
    });

    if (result?.status === "error") {
      return { ok: false, error: result.error ?? "OBB detector training failed" };
    }

    // Mark session as obb_detector_ready
    if (fs.existsSync(sessionJsonPath)) {
      try {
        const session = safeReadJson(sessionJsonPath) ?? {};
        (session as any).obbDetectorReady = true;
        fs.writeFileSync(sessionJsonPath, JSON.stringify(session, null, 2), "utf-8");
      } catch (_e) {
        // non-fatal
      }
    }

    return {
      ok: true,
      modelPath: result?.model_path,
      map50: result?.map50 ?? null,
      warnings: Array.isArray(result?.warnings) ? result.warnings : [],
    };
  } catch (e: any) {
    console.error("OBB detector training failed:", e);
    return { ok: false, error: e.message };
  }
});

ipcMain.handle(
  "ml:super-annotate",
  async (
    _event,
    args: {
      imagePath: string;
      className: string;
      modelTag?: string;
      speciesId?: string;
      options?: {
        samEnabled?: boolean;
        maxObjects?: number;
        detectionMode?: string;
        detectionPreset?: string;
      };
    }
  ) => {
    try {
      // Ensure models are initialized. Use initCompleted (not isRunning) so that
      // if the process was started by a prior command (e.g. check-super-annotator)
      // without loading models, we still call init before running annotation.
      if (!superAnnotator.initCompleted) {
        await superAnnotator.send({ cmd: "init" });
        superAnnotator.initCompleted = true;
        kickAllSegmentSaveQueues();
      }

      // Resolve session root
      const effectiveRoot = args.speciesId
        ? getSessionDir(args.speciesId)
        : projectRoot;

      // Read session orientationPolicy for schema-aware canonicalization
      let sessionOrientationPolicy: Record<string, unknown> | null = null;
      const sessionJsonPath = path.join(effectiveRoot, "session.json");
      if (fs.existsSync(sessionJsonPath)) {
        try {
          const sess = safeReadJson(sessionJsonPath) ?? {};
          sessionOrientationPolicy = (sess as any).orientationPolicy ?? null;
        } catch (_) { /* non-fatal */ }
      }

      // Resolve dlib model path if tag provided
      let dlibModel: string | undefined;
      let idMappingPath: string | undefined;
      if (args.modelTag) {
        const modelPath = path.join(effectiveRoot, "models", `predictor_${args.modelTag}.dat`);
        if (fs.existsSync(modelPath)) {
          dlibModel = modelPath;
          const mappingPath = path.join(effectiveRoot, "debug", `id_mapping_${args.modelTag}.json`);
          if (fs.existsSync(mappingPath)) {
            idMappingPath = mappingPath;
          }
        }
      }

      const sessionObbPath = path.join(effectiveRoot, "models", "session_obb_detector.pt");
      const finetunedModel = fs.existsSync(sessionObbPath) ? sessionObbPath : undefined;

      const samEnabled = args.options?.samEnabled ?? true;
      const result = await superAnnotator.send({
        cmd: "annotate",
        image_path: args.imagePath,
        class_name: args.className,
        dlib_model: dlibModel,
        id_mapping_path: idMappingPath,
        options: {
          conf_threshold: 0.3,
          sam_enabled: samEnabled,
          max_objects: args.options?.maxObjects ?? 20,
          detection_mode: args.options?.detectionMode ?? "auto",
          detection_preset: args.options?.detectionPreset ?? "balanced",
          finetuned_model: finetunedModel,
          // Pass session_dir so SAM2 segments are auto-saved for synthetic augmentation
          session_dir: samEnabled ? effectiveRoot : undefined,
          orientation_policy: sessionOrientationPolicy,
        },
      });

      if (result?.status === "error") {
        return { ok: false, error: result.error ?? "SuperAnnotator annotate failed", objects: [] };
      }

      return { ok: true, ...result, objects: Array.isArray(result?.objects) ? result.objects : [] };
    } catch (e: any) {
      console.error("Super-annotate failed:", e);
      return { ok: false, error: e.message, objects: [] };
    }
  }
);

ipcMain.handle(
  "ml:refine-sam",
  async (
    _event,
    args: {
      imagePath: string;
      objectIndex: number;
      clickPoint: [number, number];
      clickLabel: number;
    }
  ) => {
    try {
      const ready = await ensureSam2Ready();
      if (!ready.ok) {
        return { ok: false, error: ready.error || "SAM2 is not ready." };
      }

      const result = await superAnnotator.send({
        cmd: "refine_sam",
        image_path: args.imagePath,
        object_index: args.objectIndex,
        click_point: args.clickPoint,
        click_label: args.clickLabel,
      });

      return { ok: true, ...result };
    } catch (e: any) {
      console.error("SAM refinement failed:", e);
      return { ok: false, error: e.message };
    }
  }
);

ipcMain.handle(
  "ml:resegment-box",
  async (
    _event,
    args: {
      imagePath: string;
      boxXyxy: [number, number, number, number];
      iterative?: boolean;
    }
  ) => {
    try {
      const ready = await ensureSam2Ready();
      if (!ready.ok) {
        return { ok: false, error: ready.error || "SAM2 is not ready." };
      }

      const result = await superAnnotator.send({
        cmd: "resegment_box",
        image_path: args.imagePath,
        box_xyxy: args.boxXyxy,
        iterative: Boolean(args.iterative),
        expand_ratio: 0.10,
      });
      if (result.ok) {
        return {
          ok: true,
          maskOutline: result.mask_outline as [number, number][],
          obbCorners: result.obb_corners as [number, number][] | undefined,
          angle: Number.isFinite(result.angle) ? Number(result.angle) : undefined,
          boxXyxy: result.box_xyxy as [number, number, number, number] | undefined,
          score: result.score,
        };
      }
      return { ok: false, error: result.error };
    } catch (e: any) {
      console.error("SAM re-segmentation failed:", e);
      return { ok: false, error: e.message };
    }
  }
);

// Ã¢â€â‚¬Ã¢â€â‚¬ App lifecycle Ã¢â€â‚¬Ã¢â€â‚¬

app.on("before-quit", async () => {
  await superAnnotator.stop();
});

app.on("window-all-closed", () => {
  superAnnotator.stop();
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

