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
      contextIsolation: true, // ← REQUIRED
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
// Hardware capability probe — called once at app startup from React
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
    // Safe fallback — treat as CPU-only
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

function getEffectiveRoot(speciesId?: string): string {
  return speciesId ? path.join(projectRoot, "sessions", speciesId) : projectRoot;
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
  canonicalMaskSource: "none" | "segments" | "rough_otsu" | "mixed" | "unknown";
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
    trainedMaskSource?: "none" | "segments" | "rough_otsu" | "mixed" | "unknown";
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
  source: "none" | "segments" | "rough_otsu" | "mixed" | "unknown";
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
  let unknown = 0;

  raw.forEach((entry) => {
    if (!entry || typeof entry !== "object") return;
    const canonicalEnabled = Boolean((entry as any).canonical_training_enabled);
    const canonicalMeta = (entry as any).canonicalization;
    if (!canonicalEnabled && (!canonicalMeta || typeof canonicalMeta !== "object")) {
      return;
    }
    total += 1;
    const source = String((entry as any).canonical_mask_source || "")
      .trim()
      .toLowerCase();
    if (source === "segments") {
      segments += 1;
    } else if (source === "rough_otsu") {
      roughOtsu += 1;
    } else {
      unknown += 1;
    }
  });

  let resolved: "none" | "segments" | "rough_otsu" | "mixed" | "unknown" = "unknown";
  if (total === 0) {
    resolved = "none";
  } else if (segments > 0 && roughOtsu > 0) {
    resolved = "mixed";
  } else if (segments > 0) {
    resolved = "segments";
  } else if (roughOtsu > 0) {
    resolved = "rough_otsu";
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
      if (profile.canonicalMaskSource === "unknown" || profile.canonicalMaskSource === "none") {
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
          code: "sam2_unavailable_using_rough_mask_fallback",
          severity: "warning",
          message:
            "SAM2 is unavailable; inference will use rough-mask PCA fallback instead of SAM2 masks.",
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
  landmarks?: FinalizedAcceptedLandmark[];
};

function normalizeFinalizedAcceptedBoxes(
  rawBoxes: Array<{
    left: number;
    top: number;
    width: number;
    height: number;
    orientation_override?: "left" | "right" | "uncertain";
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

// ── Annotation file parsers ──

type AnnotationEntry = {
  box: { left: number; top: number; width: number; height: number };
  landmarks: Array<{ id: number; x: number; y: number }>;
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
  // Handle quoted values (single or double quotes) — paths may contain slashes
  const re = /(\w+)=(?:'([^']*)'|"([^"]*)")/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(str)) !== null) {
    attrs[m[1]] = m[2] ?? m[3] ?? "";
  }
  return attrs;
}

/**
 * Parse an imglab-format dlib XML annotation file.
 * Returns a Map keyed by image basename → { box, landmarks[] }.
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

    // Parse <box ...>
    const boxTagMatch = body.match(/<box\s([^>]+)>/);
    if (!boxTagMatch) continue;
    const boxAttrs = parseXmlAttrs(boxTagMatch[1]);
    const box = {
      left: parseFloat(boxAttrs["left"] ?? "0"),
      top: parseFloat(boxAttrs["top"] ?? "0"),
      width: parseFloat(boxAttrs["width"] ?? "0"),
      height: parseFloat(boxAttrs["height"] ?? "0"),
    };

    // Parse all <part name=N x=X y=Y/>
    const landmarks: Array<{ id: number; x: number; y: number }> = [];
    const partRe = /<part\s([^>]+?)\/>/g;
    let partMatch: RegExpExecArray | null;
    while ((partMatch = partRe.exec(body)) !== null) {
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
    const entry = { box, landmarks };
    for (const key of keys) {
      result.set(key, entry);
    }
  }

  return result;
}

/**
 * Parse a BioVision JSON annotation file.
 * Accepts a single object { imageFilename, boxes[] } or an array of such objects.
 * Returns a Map keyed by image filename → { box, landmarks[] }.
 */
function parseBioVisionJson(filePath: string): Map<string, AnnotationEntry> {
  const raw = JSON.parse(fs.readFileSync(filePath, "utf-8"));
  const records: any[] = Array.isArray(raw) ? raw : [raw];
  const result = new Map<string, AnnotationEntry>();

  for (const record of records) {
    const imageFilename: string = record?.imageFilename;
    if (!imageFilename) continue;

    const boxes: any[] = record?.boxes ?? [];
    if (!boxes.length) continue;

    // Use the first box
    const firstBox = boxes[0];
    const box = {
      left: Number(firstBox.left ?? 0),
      top: Number(firstBox.top ?? 0),
      width: Number(firstBox.width ?? 0),
      height: Number(firstBox.height ?? 0),
    };

    const rawLandmarks: any[] = firstBox?.landmarks ?? [];
    const landmarks = rawLandmarks
      .filter((lm: any) => !lm?.isSkipped)
      .map((lm: any, i: number) => ({
        id: Number(lm?.id ?? i),
        x: Number(lm?.x ?? 0),
        y: Number(lm?.y ?? 0),
      }));

    const entry = { box, landmarks };
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
    boxes: Array<{
      left: number;
      top: number;
      width: number;
      height: number;
      landmarks: Array<{
        x: number;
        y: number;
        id: number;
        isSkipped?: boolean;
      }>;
    }>;
  };
}

function collectPreAnnotatedRecords(datasetDir: string): { records: PreAnnotatedRecord[]; warnings: string[] } {
  const warnings: string[] = [];
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
      warnings.push(`${path.basename(labelPath)} references a non-standard image extension: ${imageFilename}`);
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
      warnings.push(`${path.basename(labelPath)}: could not read image dimensions; derived boxes won't be clamped.`);
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

    const normalizedBoxes: PreAnnotatedRecord["normalizedLabel"]["boxes"] = [];
    const landmarkIds = new Set<number>();
    let validLandmarkCount = 0;
    if (hasBoxes) {
      boxes.forEach((box: any, boxIndex: number) => {
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

        let left = Number(box?.left);
        let top = Number(box?.top);
        let width = Number(box?.width);
        let height = Number(box?.height);
        const hasValidBox = [left, top, width, height].every(isFiniteNumber) && width > 0 && height > 0;

        if (!hasValidBox) {
          const derived = deriveBoxFromLandmarks(normalizedLandmarks, imageDims);
          left = derived.left;
          top = derived.top;
          width = derived.width;
          height = derived.height;
          warnings.push(
            `${path.basename(labelPath)} box ${boxIndex}: missing/invalid box dimensions; derived from landmarks.`
          );
        }

        normalizedBoxes.push({
          left,
          top,
          width,
          height,
          landmarks: normalizedLandmarks,
        });
      });
    } else {
      const normalizedLandmarks = normalizeLandmarks(
        topLevelLandmarks,
        `${path.basename(labelPath)}`
      );
      normalizedLandmarks.forEach((lm) => {
        if (!lm.isSkipped) {
          landmarkIds.add(lm.id);
          validLandmarkCount += 1;
        }
      });

      const derived = deriveBoxFromLandmarks(normalizedLandmarks, imageDims);
      normalizedBoxes.push({
        left: derived.left,
        top: derived.top,
        width: derived.width,
        height: derived.height,
        landmarks: normalizedLandmarks,
      });
      warnings.push(`${path.basename(labelPath)}: no boxes found; derived 1 bounding box from landmarks.`);
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
      warnings.push("No common landmark IDs across imported samples. prepare_dataset may fail.");
    } else {
      const uniqueAll = new Set<number>();
      idSets.forEach((ids) => ids.forEach((id) => uniqueAll.add(id)));
      if (common.size < uniqueAll.size) {
        warnings.push(
          `Landmark IDs are inconsistent across files. Common IDs retained for training: ${[...common]
            .sort((a, b) => a - b)
            .join(", ")}`
        );
      }
    }
  }

  return { records, warnings };
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
      const skipParity = Boolean((options as any)?.customOptions?.skip_parity);
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
      if (skipParity) {
        cnnArgs.push("--skip-parity");
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

    // Parse output for train/test errors
    const trainErrorMatch = out.match(/TRAIN_ERROR\s+([\d.]+)/);
    const testErrorMatch = out.match(/TEST_ERROR\s+([\d.]+)/);
    const modelPathMatch = out.match(/MODEL_PATH\s+(.+)/);

    emitTrainProgress(100, "done", "Training complete.");

    return {
      ok: true,
      output: out,
      trainError: trainErrorMatch ? parseFloat(trainErrorMatch[1]) : null,
      testError: testErrorMatch ? parseFloat(testErrorMatch[1]) : null,
      modelPath: modelPathMatch ? modelPathMatch[1].trim() : null,
      auditReport: auditReport ?? undefined,
    };
  } catch (e: any) {
    emitTrainProgress(100, "error", `Training failed: ${e.message}`);
    console.error("Training failed:", e);
    return { ok: false, error: e.message };
  }
});

ipcMain.handle("ml:import-preannotated-dataset", async (_event, args?: { speciesId?: string }) => {
  try {
    const picker = await dialog.showOpenDialog({
      properties: ["openDirectory"],
      title: "Select pre-annotated dataset folder",
    });
    if (picker.canceled || picker.filePaths.length === 0) {
      return { ok: false, canceled: true, warnings: [] };
    }

    const sourceDir = picker.filePaths[0];
    const { records, warnings } = collectPreAnnotatedRecords(sourceDir);

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
      fs.writeFileSync(labelDest, JSON.stringify(record.normalizedLabel, null, 2));
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
    orientation_hint?: {
      orientation?: "left" | "right";
      confidence?: number;
      source?: string;
      head_point?: [number, number];
      tail_point?: [number, number];
    };
  }>;
}

ipcMain.handle("ml:predict", async (_event, imagePath: string, tag: string, speciesId?: string, options?: PredictOptions) => {
  let tempFile: string | null = null;
  let tempBoxesFile: string | null = null;
  let obbJsonFile: string | null = null;
  try {
    const modelRoot = getEffectiveRoot(speciesId);

    // Paths containing non-ASCII characters (e.g. macOS narrow no-break space U+202F
    // in screenshot names) can get corrupted when passed as spawn() argv on Windows.
    // Copy to a temp file with an ASCII-safe name to avoid the encoding issue.
    let effectivePath = imagePath;
    if (/[^\x00-\x7F]/.test(imagePath)) {
      const ext = path.extname(imagePath);
      tempFile = path.join(app.getPath("temp"), `bv_infer_${Date.now()}${ext}`);
      fs.copyFileSync(imagePath, tempFile);
      effectivePath = tempFile;
    }

    // Wire in session YOLO detection model so inference uses the same detector
    // that was trained for this session (falls back to OpenCV if none found).
    const yoloModel = getSessionYoloModel(speciesId);

    // Respect explicit predictor choice, otherwise auto-detect.
    const cnnModelPath = path.join(modelRoot, "models", `cnn_${tag}.pth`);
    const dlibModelPath = path.join(modelRoot, "models", `predictor_${tag}.dat`);
    const requestedPredictor = options?.predictorType;
    let predictorType: "dlib" | "cnn";
    if (requestedPredictor === "cnn") {
      if (!fs.existsSync(cnnModelPath)) {
        throw new Error(`CNN model not found for "${tag}".`);
      }
      predictorType = "cnn";
    } else if (requestedPredictor === "dlib") {
      if (!fs.existsSync(dlibModelPath)) {
        throw new Error(`dlib model not found for "${tag}".`);
      }
      predictorType = "dlib";
    } else {
      predictorType = fs.existsSync(cnnModelPath) ? "cnn" : "dlib";
    }

    const compatibility = await evaluateModelCompatibility({
      speciesId,
      modelName: tag,
      predictorType,
      includeRuntime: true,
    });
    if (!compatibility.ok) {
      throw new Error(
        compatibility.error ||
          "Model/session compatibility check failed."
      );
    }
    if (compatibility.blocking && !options?.allowIncompatible) {
      throw new Error(
        `Inference blocked by compatibility checks. ${formatCompatibilityErrorSummary(
          compatibility.issues
        )} Use override to continue.`
      );
    }

    // OBB detect-first: if session_obb_detector.pt exists and no boxes were
    // already provided by the caller, run the OBB detector on the image first
    // and pass the best detection as --obb-json so predict.py uses the
    // geometry engine instead of the PCA fallback.
    const obbDetectorPath = compatibility.obbDetectorPath;
    if (obbDetectorPath && superAnnotator && !options?.boxes?.length) {
      try {
        const obbResult = await superAnnotator.send({
          cmd: "detect_obb",
          image_path: effectivePath,
          model_path: obbDetectorPath,
          conf: 0.35,
        });
        const detections: Array<{ corners: [number,number][]; angle: number; class_id: number; confidence: number }> =
          Array.isArray(obbResult) ? obbResult : (obbResult?.detections ?? []);
        if (detections.length > 0) {
          // Pick the highest-confidence detection
          const best = detections.reduce((a, b) => (a.confidence >= b.confidence ? a : b));
          obbJsonFile = path.join(
            app.getPath("temp"),
            `bv_obb_${Date.now()}_${Math.random().toString(16).slice(2)}.json`
          );
          fs.writeFileSync(obbJsonFile, JSON.stringify({
            obbCorners: best.corners,
            angle: best.angle,
            class_id: best.class_id,
            confidence: best.confidence,
          }));
        }
      } catch (obbErr) {
        console.warn("OBB detection failed (non-fatal), falling back to standard crop:", obbErr);
      }
    }

    const pythonArgs = [
      path.join(__dirname, "../backend/inference/predict.py"),
      modelRoot,
      tag,
      effectivePath,
      "--predictor-type", predictorType,
    ];
    if (obbJsonFile) {
      pythonArgs.push("--obb-json", obbJsonFile);
    }
    if (options?.multiSpecimen) {
      pythonArgs.push("--multi");
    }
    if (yoloModel) {
      pythonArgs.push("--yolo-model", yoloModel);
    }
    let boxesForInference: PredictOptions["boxes"] =
      Array.isArray(options?.boxes) && options!.boxes.length > 0 ? options!.boxes : undefined;

    if (Array.isArray(boxesForInference) && boxesForInference.length > 0) {
      tempBoxesFile = path.join(app.getPath("temp"), `bv_boxes_${Date.now()}_${Math.random().toString(16).slice(2)}.json`);
      fs.writeFileSync(tempBoxesFile, JSON.stringify(boxesForInference));
      pythonArgs.push("--boxes-json", tempBoxesFile);
    }

    const out = await runPythonWithProgress(pythonArgs, (percent, stage) => {
      mainWindow?.webContents.send("ml:predict-progress", { percent, stage });
    });

    const data = JSON.parse(out);
    return { ok: true, data };
  } catch (e: any) {
    console.error("Prediction failed:", e);
    return { ok: false, error: e.message };
  } finally {
    if (tempFile) {
      try { fs.unlinkSync(tempFile); } catch {}
    }
    if (tempBoxesFile) {
      try { fs.unlinkSync(tempBoxesFile); } catch {}
    }
    if (obbJsonFile) {
      try { fs.unlinkSync(obbJsonFile); } catch {}
    }
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
    args: { imageFolderPath: string; annotationFilePath: string; speciesId: string }
  ) => {
    try {
      const { imageFolderPath, annotationFilePath, speciesId } = args;

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

      for (const filename of imageFiles) {
        const srcPath = path.join(imageFolderPath, filename);
        const diskPath = path.join(imagesDir, filename);
        const imgExt = path.extname(filename).toLowerCase();
        const mimeType = MIME_TYPES[imgExt] || "image/jpeg";

        // Copy image into session (no base64 read — renderer uses localfile:// URLs)
        fs.copyFileSync(srcPath, diskPath);

        const annotationKeys = buildMatchableKeys(filename);
        const annotation = annotationKeys
          .map((key) => annotationMap.get(key))
          .find((entry) => Boolean(entry));

        let boxes: any[] = [];
        if (annotation) {
          const box = {
            id: 0,
            left: annotation.box.left,
            top: annotation.box.top,
            width: annotation.box.width,
            height: annotation.box.height,
            landmarks: annotation.landmarks,
            source: "manual" as const,
          };
          boxes = [box];

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

      const warnings: string[] =
        unmatched.length > 0
          ? [`${unmatched.length} image(s) had no matching annotation.`]
          : [];

      return { ok: true, images, unmatched, warnings };
    } catch (e: any) {
      console.error("ml:load-annotated-folder failed:", e);
      return { ok: false, error: e.message };
    }
  }
);

// List trained models in the session (or global) models directory
ipcMain.handle("ml:list-models", async (_event, speciesId?: string) => {
  try {
    const modelsDir = path.join(getEffectiveRoot(speciesId), "models");

    if (!fs.existsSync(modelsDir)) {
      fs.mkdirSync(modelsDir, { recursive: true });
      return { ok: true, models: [] };
    }

    const files = fs.readdirSync(modelsDir);

    const dlibModels = files
      .filter((f) => f.endsWith(".dat") && f.startsWith("predictor_"))
      .map((file) => {
        const filePath = path.join(modelsDir, file);
        const stats = fs.statSync(filePath);
        const tag = file.replace(/^predictor_/, "").replace(/\.dat$/, "");
        return {
          name: tag,
          path: filePath,
          size: stats.size,
          createdAt: stats.birthtime,
          predictorType: "dlib" as const,
        };
      });

    const cnnModels = files
      .filter((f) => f.endsWith(".pth") && f.startsWith("cnn_"))
      .map((file) => {
        const filePath = path.join(modelsDir, file);
        const stats = fs.statSync(filePath);
        const tag = file.replace(/^cnn_/, "").replace(/\.pth$/, "");
        return {
          name: tag,
          path: filePath,
          size: stats.size,
          createdAt: stats.birthtime,
          predictorType: "cnn" as const,
        };
      });

    const models = [...dlibModels, ...cnnModels]
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
      return { ok: true };
    }
    if (predictorType === "cnn") {
      if (!cnnExists) return { ok: false, error: "CNN model not found" };
      fs.unlinkSync(cnnPath);
      if (fs.existsSync(cnnConfigPath)) fs.unlinkSync(cnnConfigPath);
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
      return { ok: true };
    }
    if (predictorType === "cnn") {
      if (!isCnn) return { ok: false, error: "CNN model not found" };
      if (fs.existsSync(newCnn)) return { ok: false, error: "A CNN model with that name already exists" };
      fs.renameSync(oldCnn, newCnn);
      if (fs.existsSync(oldCnnCfg)) fs.renameSync(oldCnnCfg, newCnnCfg);
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
    const yoloModel = getSessionYoloModel(speciesId);

    const pythonArgs = [
      path.join(__dirname, "../backend/detection/detect_specimen.py"),
      imagePath,
      "--multi",
    ];
    if (yoloModel) {
      pythonArgs.push("--yolo-model", yoloModel);
    }

    const out = await runPython(pythonArgs);

    const data = JSON.parse(out.trim());
    if (!data) {
      return { ok: false, error: "No detection result", boxes: [] };
    }

    if (data.ok === true && Array.isArray(data.boxes)) {
      return data;
    }

    if (typeof data.left === "number") {
      return {
        ok: true,
        boxes: [{
          left: data.left,
          top: data.top,
          right: data.right,
          bottom: data.bottom,
          width: data.width,
          height: data.height,
          confidence: 1.0,
          class_id: 0,
          class_name: "specimen",
        }],
        image_width: undefined,
        image_height: undefined,
        num_detections: 1,
        detection_method: "opencv_single",
      };
    }

    return { ok: false, error: "Unexpected detection format", boxes: [] };
  } catch (e: any) {
    console.error("Specimen detection failed:", e);
    return { ok: false, error: e.message, boxes: [] };
  }
});

// Check detection availability (OpenCV always available)
ipcMain.handle("ml:check-yolo", async () => {
  try {
    const out = await runPython([
      path.join(__dirname, "../backend/detection/detect_specimen.py"),
      "--check",
    ]);

    return JSON.parse(out.trim());
  } catch (e: any) {
    return { available: true, primary_method: "opencv" };
  }
});

// ── Session management IPC handlers ──

function getSessionDir(speciesId: string): string {
  const safeSpeciesId = String(speciesId || "")
    .trim()
    .replace(/[^a-zA-Z0-9_-]/g, "_");
  return path.join(projectRoot, "sessions", safeSpeciesId || "default");
}

const INFERENCE_REVIEW_DRAFTS_FILE = "inference_review_drafts.json";
const RETRAIN_QUEUE_FILE = "retrain_queue.json";
const INFERENCE_SESSIONS_DIR = "inference_sessions";
const INFERENCE_SESSION_MANIFEST_FILE = "manifest.json";

type InferenceDraftSpecimen = {
  box: {
    left: number;
    top: number;
    width: number;
    height: number;
    confidence?: number;
    class_id?: number;
    class_name?: string;
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
  createdAt: string;
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

function getInferenceSessionManifestPath(speciesId: string, inferenceSessionId: string): string {
  return path.join(
    getInferenceSessionDir(speciesId, inferenceSessionId),
    INFERENCE_SESSION_MANIFEST_FILE
  );
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
  landmarkModelKey?: string;
  landmarkModelName?: string;
  landmarkPredictorType?: "dlib" | "cnn" | "yolo_pose";
  detectionModelKey?: string;
  detectionModelName?: string;
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
    createdAt: existing?.createdAt || now,
    updatedAt: now,
  };
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  return manifest;
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

function safeClassName(className: string): string {
  return (className || "object").toLowerCase().trim().replace(/\s+/g, "_");
}

function getSessionYoloAliasPath(speciesId: string, className: string): string {
  return path.join(getSessionDir(speciesId), "models", `yolo_${safeClassName(className)}.pt`);
}

/**
 * Find the best YOLO detection model for a session.
 * Selection order:
 * 1) class-specific promoted alias (if className provided)
 * 2) active_model.path from any YOLO registry in models/
 * 3) single promoted alias yolo_<class>.pt (non-versioned) if exactly one exists
 * 4) fallback: most recently modified yolo_*.pt
 */
function getSessionYoloModel(speciesId?: string, className?: string): string | undefined {
  const modelsDir = path.join(getEffectiveRoot(speciesId), "models");
  if (!fs.existsSync(modelsDir)) return undefined;

  if (speciesId && className) {
    const aliasPath = getSessionYoloAliasPath(speciesId, className);
    if (fs.existsSync(aliasPath)) return aliasPath;
  }

  const registryCandidates: { full: string; updatedMs: number }[] = [];
  const registryFiles = fs
    .readdirSync(modelsDir)
    .filter((f) => /^yolo_.+_registry\.json$/i.test(f));
  for (const registryFile of registryFiles) {
    try {
      const registryPath = path.join(modelsDir, registryFile);
      const raw = JSON.parse(fs.readFileSync(registryPath, "utf-8"));
      const active = raw?.active_model;
      if (!active || typeof active.path !== "string") continue;
      if (Boolean(active?.use_pose)) continue;
      const activePath = path.isAbsolute(active.path)
        ? active.path
        : path.join(modelsDir, active.path);
      if (!fs.existsSync(activePath)) continue;
      let updatedMs = 0;
      if (typeof active.updated_at === "string") {
        const parsed = Date.parse(active.updated_at);
        if (Number.isFinite(parsed)) updatedMs = parsed;
      }
      if (!updatedMs) {
        updatedMs = fs.statSync(activePath).mtimeMs;
      }
      registryCandidates.push({ full: activePath, updatedMs });
    } catch (_) {
      // Ignore malformed registry and continue fallback resolution.
    }
  }
  if (registryCandidates.length > 0) {
    registryCandidates.sort((a, b) => b.updatedMs - a.updatedMs);
    return registryCandidates[0].full;
  }

  const promotedAliases = fs
    .readdirSync(modelsDir)
    .filter((f) => /^yolo_.+\.pt$/i.test(f))
    .filter((f) => !/_v\d+\.pt$/i.test(f))
    .filter((f) => {
      const clsTag = f.replace(/^yolo_/, "").replace(/\.pt$/i, "");
      const registryPath = path.join(modelsDir, `yolo_${clsTag}_registry.json`);
      if (!fs.existsSync(registryPath)) return true;
      try {
        const raw = JSON.parse(fs.readFileSync(registryPath, "utf-8"));
        const active = raw?.active_model;
        if (!active) return true;
        return !Boolean(active.use_pose);
      } catch {
        return true;
      }
    });
  if (promotedAliases.length === 1) {
    return path.join(modelsDir, promotedAliases[0]);
  }

  const candidates = fs
    .readdirSync(modelsDir)
    .filter((f) => f.startsWith("yolo_") && f.endsWith(".pt"))
    .filter((f) => {
      const clsTag = f
        .replace(/^yolo_/, "")
        .replace(/_v\d+\.pt$/i, "")
        .replace(/\.pt$/i, "");
      const registryPath = path.join(modelsDir, `yolo_${clsTag}_registry.json`);
      if (!fs.existsSync(registryPath)) return true;
      try {
        const raw = JSON.parse(fs.readFileSync(registryPath, "utf-8"));
        const runs = Array.isArray(raw?.training_runs) ? raw.training_runs : [];
        const fullPath = path.join(modelsDir, f);
        const matched = runs.find((r: any) => {
          const p = typeof r?.path === "string" ? r.path : "";
          if (!p) return false;
          const resolved = path.isAbsolute(p) ? p : path.join(modelsDir, p);
          return path.resolve(resolved) === path.resolve(fullPath);
        });
        if (!matched) return true;
        return !Boolean(matched.use_pose);
      } catch {
        return true;
      }
    })
    .map((f) => {
      const full = path.join(modelsDir, f);
      return { full, mtime: fs.statSync(full).mtimeMs };
    });

  if (candidates.length === 0) return undefined;
  candidates.sort((a, b) => b.mtime - a.mtime);
  return candidates[0].full;
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

      const acceptedBoxes = normalizeFinalizedAcceptedBoxes(args.boxes || []);

      const xyxyBoxes = acceptedBoxes.map((b) => [
        b.left,
        b.top,
        b.left + b.width,
        b.top + b.height,
      ]);

      const cacheKey = `${args.speciesId}::${args.filename}`;
      const signature = buildAcceptedBoxesSignature(acceptedBoxes);
      const cacheHit = finalizedSegmentSignatureCache.get(cacheKey) === signature;
      let segmentSaveAttempted = false;
      let segmentSaveSkipped = false;

      // Only save segments if SAM2 is already warm — avoids blocking model load on finalize
      if (superAnnotator.isRunning && resolvedImagePath && xyxyBoxes.length > 0 && !cacheHit) {
        segmentSaveAttempted = true;
        try {
          await superAnnotator.send({
            cmd: "save_segments_for_boxes",
            image_path: resolvedImagePath,
            boxes: xyxyBoxes,
            session_dir: sessionDir,
          });
        } catch (_) {
          // non-fatal
        }
      } else {
        segmentSaveSkipped = true;
      }
      // Always mark as finalized (UI lock) regardless of whether SAM2 was available
      finalizedSegmentSignatureCache.set(cacheKey, signature);

      // Persist finalized box snapshot into label JSON.
      const labelsDir = path.join(sessionDir, "labels");
      fs.mkdirSync(labelsDir, { recursive: true });
      const labelPath = path.join(labelsDir, args.filename.replace(/\.\w+$/, ".json"));
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
          // keep default payload
        }
      }
      payload.imageFilename = payload.imageFilename || args.filename;
      payload.speciesId = payload.speciesId || args.speciesId;
      payload.boxes = Array.isArray(payload.boxes) ? payload.boxes : [];
      payload.rejectedDetections = Array.isArray(payload.rejectedDetections) ? payload.rejectedDetections : [];
      payload.finalizedDetection = {
        isFinalized: true,
        finalizedAt: new Date().toISOString(),
        acceptedBoxes,
        boxSignature: signature,
      };
      fs.writeFileSync(labelPath, JSON.stringify(payload, null, 2));

      // Persist finalized filename to disk so it survives app restart
      const finalizedListPath = path.join(sessionDir, "finalized_images.json");
      try {
        const existing: string[] = fs.existsSync(finalizedListPath)
          ? JSON.parse(fs.readFileSync(finalizedListPath, "utf-8"))
          : [];
        if (!existing.includes(args.filename)) {
          existing.push(args.filename);
          fs.writeFileSync(finalizedListPath, JSON.stringify(existing));
        }
      } catch (_) { /* non-fatal */ }

      return {
        ok: true,
        finalized: true,
        acceptedCount: acceptedBoxes.length,
        signature,
        skipped: cacheHit && segmentSaveSkipped,
        segmentSaveAttempted,
      };
    } catch (e: any) {
      console.error("session:finalize-accepted-boxes failed:", e);
      return { ok: false, error: e.message };
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

      if (!fs.existsSync(imagesDir)) {
        return { ok: true, images: [], meta };
      }

      // Load finalized filenames
      const finalizedListPath = path.join(sessionDir, "finalized_images.json");
      let finalizedSet = new Set<string>();
      if (fs.existsSync(finalizedListPath)) {
        try {
          const list = JSON.parse(fs.readFileSync(finalizedListPath, "utf-8")) as string[];
          finalizedSet = new Set(list);
        } catch (_) { /* ignore bad file */ }
      }

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
              boxes = labelData.boxes;
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
          finalized: finalizedSet.has(filename) || isFinalizedFromLabel,
        };
      });

      return { ok: true, images, meta };
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
      const safeFilename = path.basename(String(args.filename || "").trim());
      if (!safeFilename) {
        return { ok: false, error: "Invalid filename", boxes: [] };
      }
      const labelPath = path.join(labelsDir, safeFilename.replace(/\.\w+$/, ".json"));

      if (!fs.existsSync(labelPath)) {
        return { ok: true, boxes: [], finalized: false };
      }

      const labelData = JSON.parse(fs.readFileSync(labelPath, "utf-8"));
      return {
        ok: true,
        boxes: Array.isArray(labelData?.boxes) ? labelData.boxes : [],
        finalized: Boolean(labelData?.finalizedDetection?.isFinalized),
      };
    } catch (e: any) {
      console.error("session:load-annotation failed:", e);
      return { ok: false, error: e.message, boxes: [] };
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
      // Delete persisted inference-review drafts
      try {
        const draftPath = getInferenceReviewDraftsPath(args.speciesId);
        if (fs.existsSync(draftPath)) {
          fs.unlinkSync(draftPath);
        }
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
      const sessionDir = getSessionDir(args.speciesId);
      const imagesDir = path.join(sessionDir, "images");
      const labelsDir = path.join(sessionDir, "labels");
      fs.mkdirSync(imagesDir, { recursive: true });
      fs.mkdirSync(labelsDir, { recursive: true });

      const imgName = args.filename ?? path.basename(args.imagePath);
      const imgDest = path.join(imagesDir, imgName);
      const lblDest = path.join(labelsDir, imgName.replace(/\.\w+$/, ".json"));

      // Copy image into session if not already there
      if (!fs.existsSync(imgDest)) {
        fs.copyFileSync(args.imagePath, imgDest);
      }

      let boxesPayload: any[] = [];
      if (Array.isArray(args.specimens) && args.specimens.length > 0) {
        boxesPayload = args.specimens
          .filter((s) => s?.box && s.box.width > 0 && s.box.height > 0)
          .map((s) => ({
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
          }));
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
        } catch (_) {
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
          mergedRejectedMap.set(key, {
            ...d,
            left,
            top,
            width,
            height,
          });
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
          landmarks: b.landmarks,
        }))
      );
      const boxSignature = buildAcceptedBoxesSignature(acceptedBoxes);

      // Write label JSON in BioVision format (+ finalized snapshot for YOLO export)
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

      // Persist finalized filename for finalized-only YOLO export.
      const finalizedListPath = path.join(sessionDir, "finalized_images.json");
      try {
        const existing: string[] = fs.existsSync(finalizedListPath)
          ? JSON.parse(fs.readFileSync(finalizedListPath, "utf-8"))
          : [];
        if (!existing.includes(imgName)) {
          existing.push(imgName);
          fs.writeFileSync(finalizedListPath, JSON.stringify(existing));
        }
      } catch (_) {
        // non-fatal
      }

      // Update session.json imageCount
      const sessionJsonPath = path.join(sessionDir, "session.json");
      if (fs.existsSync(sessionJsonPath)) {
        try {
          const meta = JSON.parse(fs.readFileSync(sessionJsonPath, "utf-8"));
          meta.imageCount = fs.readdirSync(imagesDir).filter((f) => IMAGE_EXTS.test(f)).length;
          meta.lastModified = new Date().toISOString();
          fs.writeFileSync(sessionJsonPath, JSON.stringify(meta, null, 2));
        } catch (_) {}
      }

      return { ok: true, savedPath: lblDest };
    } catch (e: any) {
      console.error("session:save-inference-correction failed:", e);
      return { ok: false, error: e.message };
    }
  }
);

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
      const sessionDir = getSessionDir(args.speciesId);
      const imagesDir = path.join(sessionDir, "images");
      const detLblDir = path.join(sessionDir, "detection_labels");
      fs.mkdirSync(imagesDir, { recursive: true });
      fs.mkdirSync(detLblDir, { recursive: true });

      const imgName = args.filename ?? path.basename(args.imagePath);
      const imgDest = path.join(imagesDir, imgName);

      // Copy image into session if not already there
      if (!fs.existsSync(imgDest)) {
        fs.copyFileSync(args.imagePath, imgDest);
      }

      // Write YOLO-format detection label (normalized coords)
      const iw = Math.max(args.imageWidth, 1);
      const ih = Math.max(args.imageHeight, 1);
      const lines = args.boxes
        .filter((b) => b.width > 0 && b.height > 0)
        .map((b) => {
          const cx = ((b.left + b.width / 2) / iw).toFixed(6);
          const cy = ((b.top + b.height / 2) / ih).toFixed(6);
          const w  = (b.width  / iw).toFixed(6);
          const h  = (b.height / ih).toFixed(6);
          return `0 ${cx} ${cy} ${w} ${h}`;
        });

      const lblDest = path.join(detLblDir, imgName.replace(/\.\w+$/, ".txt"));
      fs.writeFileSync(lblDest, lines.join("\n") + (lines.length > 0 ? "\n" : ""));

      return { ok: true, savedPath: lblDest };
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

      // Remove persisted inference-review drafts for this image.
      try {
        const drafts = readInferenceReviewDrafts(args.speciesId);
        const lowerName = (args.filename || "").toLowerCase();
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

// ── SuperAnnotator persistent process manager ──

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
  /** True after a successful `init` command — models are loaded and ready. */
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
    if (cmdName === "train_yolo") return 4 * 60 * 60 * 1000;
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

async function ensureSam2Ready(): Promise<{ ok: boolean; error?: string }> {
  try {
    if (!superAnnotator.isRunning) {
      const initRes = await superAnnotator.send({ cmd: "init" });
      if (initRes?.sam2_loaded) {
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
      return { ok: true };
    }
    const recheck = await superAnnotator.send({ cmd: "check" });
    return { ok: false, error: recheck?.sam2_error || "SAM2 is not available." };
  } catch (e: any) {
    return { ok: false, error: e?.message || "Failed to initialize SAM2." };
  }
}

// ── SuperAnnotator IPC handlers ──

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
    return { ok: true, ...result };
  } catch (e: any) {
    return { ok: false, error: e.message };
  }
});

ipcMain.handle("ml:train-obb-detector", async (_event, speciesId: string, options?: { epochs?: number; modelTier?: "nano" | "small" }) => {
  try {
    if (!superAnnotator.isRunning) {
      await superAnnotator.send({ cmd: "init" });
    }
    const sessionDir = path.join(projectRoot, "sessions", speciesId);
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
      console.warn("Hardware probe failed during OBB training — defaulting to cpu");
    }

    const result = await superAnnotator.send({
      cmd: "train_yolo_obb",
      session_dir: sessionDir,
      epochs: options?.epochs,   // undefined → Python selects hardware-appropriate default
      model_tier: modelTier,
      device: hwDevice,
      sam2_enabled: hwSam2Enabled,
    });

    if (result?.status === "error") {
      return { ok: false, error: result.error ?? "OBB detector training failed" };
    }

    // Mark session as obb_detector_ready
    const sessionJsonPath = path.join(sessionDir, "session.json");
    if (fs.existsSync(sessionJsonPath)) {
      try {
        const session = safeReadJson(sessionJsonPath) ?? {};
        (session as any).obbDetectorReady = true;
        fs.writeFileSync(sessionJsonPath, JSON.stringify(session, null, 2), "utf-8");
      } catch (_e) {
        // non-fatal
      }
    }

    return { ok: true, modelPath: result?.model_path, map50: result?.map50 ?? null };
  } catch (e: any) {
    console.error("OBB detector training failed:", e);
    return { ok: false, error: e.message };
  }
});

ipcMain.handle("ml:tag-class-ids", async (_event, speciesId: string, boxes: any[]) => {
  try {
    if (!superAnnotator.isRunning) {
      await superAnnotator.send({ cmd: "init" });
    }
    const sessionDir = path.join(projectRoot, "sessions", speciesId);
    if (!fs.existsSync(sessionDir)) {
      return { ok: false, error: `Session not found: ${sessionDir}` };
    }

    // Load orientation policy from session.json
    let orientationPolicy: any = null;
    const sessionJsonPath = path.join(sessionDir, "session.json");
    if (fs.existsSync(sessionJsonPath)) {
      try {
        const session = safeReadJson(sessionJsonPath) ?? {};
        orientationPolicy = (session as any).orientationPolicy ?? null;
      } catch (_) { /* non-fatal */ }
    }

    const result = await superAnnotator.send({
      cmd: "tag_class_ids",
      session_dir: sessionDir,
      boxes: boxes || [],
      orientation_policy: orientationPolicy,
    });

    if (result?.status === "error") {
      return { ok: false, error: result.error ?? "tag_class_ids failed" };
    }
    return { ok: true, taggedBoxes: result?.tagged_boxes ?? [] };
  } catch (e: any) {
    console.error("ml:tag-class-ids failed:", e);
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
        confThreshold?: number;
        samEnabled?: boolean;
        maxObjects?: number;
        detectionMode?: string;
        detectionPreset?: string;
        pcaMode?: "off" | "on" | "auto";
        useOrientationHint?: boolean;
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
      }

      // Resolve session root
      const effectiveRoot = args.speciesId
        ? path.join(projectRoot, "sessions", args.speciesId)
        : projectRoot;

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

      // Check for fine-tuned YOLO detection model
      const ftModelPath = args.speciesId
        ? getSessionYoloAliasPath(args.speciesId, args.className)
        : path.join(effectiveRoot, "models", `yolo_${safeClassName(args.className)}.pt`);
      const finetunedModel = fs.existsSync(ftModelPath) ? ftModelPath : undefined;

      // If a session OBB detector exists, use it for annotation auto-detect
      // instead of YOLO-World, since it is trained specifically for this session's data.
      const sessionObbPath = path.join(effectiveRoot, "models", "session_obb_detector.pt");
      if (fs.existsSync(sessionObbPath) && superAnnotator) {
        try {
          const obbResult = await superAnnotator.send({
            cmd: "detect_obb",
            image_path: args.imagePath,
            model_path: sessionObbPath,
            conf: args.options?.confThreshold ?? 0.35,
          });
          const rawDetections: any[] = Array.isArray(obbResult?.detections) ? obbResult.detections : [];
          const maxObj = args.options?.maxObjects ?? 20;
          const limited = rawDetections.slice(0, maxObj);

          if (limited.length > 0) {
            const objects: any[] = [];
            for (const det of limited) {
              const corners: [number, number][] = det.corners ?? [];
              const xs = corners.map((p) => p[0]);
              const ys = corners.map((p) => p[1]);
              const left   = Math.round(Math.min(...xs));
              const top    = Math.round(Math.min(...ys));
              const right  = Math.round(Math.max(...xs));
              const bottom = Math.round(Math.max(...ys));
              const obj: any = {
                box: { left, top, right, bottom, width: right - left, height: bottom - top },
                obb: {
                  corners: corners,
                  angle: det.angle ?? 0,
                  center: [(left + right) / 2, (top + bottom) / 2],
                  size: [right - left, bottom - top],
                },
                confidence: det.confidence ?? 0,
                class_name: args.className ?? "specimen",
                detection_method: "yolo_obb",
                mask_outline: [],
                landmarks: [],
              };
              // Optionally refine with SAM2 segmentation
              if (args.options?.samEnabled !== false) {
                try {
                  const seg = await superAnnotator.send({
                    cmd: "resegment_box",
                    image_path: args.imagePath,
                    box_xyxy: [left, top, right, bottom],
                  });
                  if (seg?.ok && Array.isArray(seg.mask_outline) && seg.mask_outline.length > 0) {
                    obj.mask_outline = seg.mask_outline;
                  }
                } catch (_) { /* non-fatal */ }
              }
              objects.push(obj);
            }
            return { ok: true, objects, detection_method: "yolo_obb" };
          }
        } catch (_) {
          // Fall through to normal annotate pipeline on any error
        }
      }

      const samEnabled = args.options?.samEnabled ?? true;
      const result = await superAnnotator.send({
        cmd: "annotate",
        image_path: args.imagePath,
        class_name: args.className,
        dlib_model: dlibModel,
        id_mapping_path: idMappingPath,
        options: {
          conf_threshold: args.options?.confThreshold ?? 0.3,
          sam_enabled: samEnabled,
          max_objects: args.options?.maxObjects ?? 20,
          detection_mode: args.options?.detectionMode ?? "auto",
          detection_preset: args.options?.detectionPreset ?? "balanced",
          pca_mode: args.options?.pcaMode ?? "auto",
          use_orientation_hint: args.options?.useOrientationHint ?? true,
          finetuned_model: finetunedModel,
          // Pass session_dir so SAM2 segments are auto-saved for synthetic augmentation
          session_dir: samEnabled ? effectiveRoot : undefined,
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
      });
      if (result.ok) {
        return { ok: true, maskOutline: result.mask_outline as [number, number][], score: result.score };
      }
      return { ok: false, error: result.error };
    } catch (e: any) {
      console.error("SAM re-segmentation failed:", e);
      return { ok: false, error: e.message };
    }
  }
);

ipcMain.handle(
  "ml:get-yolo-train-plan",
  async (
    _event,
    args: {
      speciesId: string;
      className: string;
      epochs?: number;
      detectionPreset?: string;
      datasetSize?: number;
      autoTune?: boolean;
    }
  ) => {
    try {
      const sessionDir = path.join(projectRoot, "sessions", args.speciesId);
      if (!fs.existsSync(sessionDir)) {
        return { ok: false, error: `Session directory not found: ${sessionDir}` };
      }

      if (!superAnnotator.initCompleted) {
        await superAnnotator.send({ cmd: "init" });
        superAnnotator.initCompleted = true;
      }

      const result = await superAnnotator.send({
        cmd: "preview_yolo_train_plan",
        session_dir: sessionDir,
        class_name: args.className,
        epochs: args.epochs,
        detection_preset: args.detectionPreset ?? "balanced",
        dataset_size: args.datasetSize,
        auto_tune: args.autoTune ?? true,
      });

      if (result?.status === "error") {
        return { ok: false, error: result.error ?? "YOLO training plan preview failed" };
      }

      return {
        ok: true,
        dataset: result?.dataset,
        usePose: result?.use_pose,
        detectionPreset: result?.detection_preset,
        autoTune: result?.auto_tune,
        datasetSizeEffective: result?.dataset_size_effective,
        datasetSizeSource: result?.dataset_size_source,
        resolvedTrainParams: result?.resolved_train_params,
        preflightWarnings: result?.preflight_warnings,
      };
    } catch (e: any) {
      console.error("YOLO train plan preview failed:", e);
      return { ok: false, error: e.message };
    }
  }
);

ipcMain.handle(
  "ml:train-yolo",
  async (
    _event,
    args: {
      speciesId: string;
      className: string;
      epochs?: number;
      detectionPreset?: string;
      datasetSize?: number;
      autoTune?: boolean;
    }
  ) => {
    try {
      const sessionDir = path.join(projectRoot, "sessions", args.speciesId);
      if (!fs.existsSync(sessionDir)) {
        return { ok: false, error: `Session directory not found: ${sessionDir}` };
      }

      if (!superAnnotator.initCompleted) {
        await superAnnotator.send({ cmd: "init" });
        superAnnotator.initCompleted = true;
      }

      const result = await superAnnotator.send({
        cmd: "train_yolo",
        session_dir: sessionDir,
        class_name: args.className,
        epochs: args.epochs,
        detection_preset: args.detectionPreset ?? "balanced",
        dataset_size: args.datasetSize,
        auto_tune: args.autoTune ?? true,
      });

      if (result?.status === "error") {
        return { ok: false, error: result.error ?? "YOLO training failed" };
      }

      return {
        ok: true,
        modelPath: result?.active_model_path || result?.model_path,
        candidateModelPath: result?.candidate_model_path,
        version: result?.version,
        promoted: result?.promoted,
        evaluationMetricType: result?.evaluation_metric_type,
        candidateMap50: result?.candidate_map50,
        candidateMap50_95: result?.candidate_map50_95,
        incumbentMap50: result?.incumbent_map50,
        incumbentMap50_95: result?.incumbent_map50_95,
        candidatePoseMap50: result?.candidate_pose_map50,
        candidatePoseMap50_95: result?.candidate_pose_map50_95,
        candidateBoxMap50: result?.candidate_box_map50,
        candidateBoxMap50_95: result?.candidate_box_map50_95,
        incumbentPoseMap50: result?.incumbent_pose_map50,
        incumbentPoseMap50_95: result?.incumbent_pose_map50_95,
        incumbentBoxMap50: result?.incumbent_box_map50,
        incumbentBoxMap50_95: result?.incumbent_box_map50_95,
        candidateMetrics: result?.candidate_metrics,
        incumbentMetrics: result?.incumbent_metrics,
        dataset: result?.dataset,
        registryPath: result?.registry_path,
        detectionPreset: result?.detection_preset,
        autoTune: result?.auto_tune,
        datasetSizeEffective: result?.dataset_size_effective,
        datasetSizeSource: result?.dataset_size_source,
        resolvedTrainParams: result?.resolved_train_params,
        preflightWarnings: result?.preflight_warnings,
      };
    } catch (e: any) {
      console.error("YOLO training failed:", e);
      return { ok: false, error: e.message };
    }
  }
);

// ── App lifecycle ──

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
