import { app, BrowserWindow, ipcMain, dialog, nativeImage, protocol, shell } from "electron";
import fs from "fs";
import * as path from "path";
import { spawn, ChildProcess } from "child_process";
import { createInterface, Interface as ReadlineInterface } from "readline";
import { createHash, randomUUID } from "crypto";
import {
  computeSchemaSemanticFingerprint,
  SCHEMA_SEMANTIC_VERSION,
} from "../src/lib/schemaFingerprint";
import {
  atomicCopyFileSync,
  atomicWriteFileSync,
  atomicWriteJsonSync,
  calculateHitlReviewPriority,
  commitHitlFileTransaction,
  resolveContentAddressedHitlImage,
  resolveHitlNewTrainingSample,
  resolveReviewStatus,
  resolveReviewWasEdited,
  sha256FileSync,
  stageInferenceImagePaths,
  summarizeRetrainingReviewEvents,
  type HitlPrioritySignals,
  type HitlReviewPriority,
} from "./hitlPersistence";
import { compareModelSchemaContract } from "./modelCompatibility";
import {
  isVerifiedTrainedObbDetector,
  resolveTrainedObbDetector,
  resolveZeroShotDetector,
  validateObbPromotionCandidate,
  type DetectionModelProvenance,
  type ObbDetectorResolution,
} from "./detectorProvenance";
import { parseLandmarkTrainingPublication } from "./trainingProtocol";
import { validateLandmarkPromotionArtifact } from "./modelArtifactIntegrity";
import { resolveObbInferenceThresholdPlan } from "./obbInferenceOptions";

const contextMenu = require("electron-context-menu");

let mainWindow: BrowserWindow | null;
const userDataDir = app.getPath("userData");
const defaultProjectRoot = path.join(userDataDir, "training-model");
const configPath = path.join(userDataDir, "biovision-config.json");
const customSchemaLibraryPath = path.join(userDataDir, "biovision-schema-library.json");

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

function restoreFileSnapshot(filePath: string, snapshot: Buffer | null): void {
  if (snapshot) atomicWriteFileSync(filePath, snapshot);
  else if (fs.existsSync(filePath)) fs.unlinkSync(filePath);
}
const finalizedSegmentSignatureCache = new Map<string, string>();
type SegmentQueueState =
  | "idle"
  | "queued"
  | "running"
  | "saved"
  | "already_finalized"
  | "finalized_without_segments"
  | "skipped"
  | "failed";
type SegmentSaveDetail = {
  index: number;
  status: "saved" | "failed";
  maskSource?: string;
  reason?: string;
};
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
  details?: SegmentSaveDetail[];
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
    mainWindow.loadFile(path.join(__dirname, "../dist/index.html"));
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

function getPythonResolution(): { pythonPath: string; usingRepoVenv: boolean } {
  // Packaged: the PyInstaller dispatcher ships its own torch/psutil, so treat
  // it as a trusted runtime (usingRepoVenv=true silences the fallback-interpreter
  // warning in the UI). If the bundle is missing, surface that via usingRepoVenv=false.
  if (app.isPackaged) {
    const ext = process.platform === "win32" ? ".exe" : "";
    const bundled = path.join(process.resourcesPath, "python", `biovision_backend${ext}`);
    if (fs.existsSync(bundled)) {
      return { pythonPath: bundled, usingRepoVenv: true };
    }
    return { pythonPath: bundled, usingRepoVenv: false };
  }

  // Windows: venv\Scripts\python.exe
  const venvWin = path.join(__dirname, "..", "venv", "Scripts", "python.exe");
  if (fs.existsSync(venvWin)) return { pythonPath: venvWin, usingRepoVenv: true };

  // Unix/macOS: venv/bin/python
  const venvUnix = path.join(__dirname, "..", "venv", "bin", "python");
  if (fs.existsSync(venvUnix)) return { pythonPath: venvUnix, usingRepoVenv: true };

  // Fall back to system Python
  return { pythonPath: "python", usingRepoVenv: false };
}

function getPythonPath(): string {
  return getPythonResolution().pythonPath;
}

/**
 * Resolve a backend script to either a bundled executable (production) or
 * a [pythonPath, scriptPath] pair (development).
 *
 * @param scriptName - base name without extension, e.g. "predict", "super_annotator"
 * @returns { cmd, args } where cmd is the executable and args are the remaining arguments
 */
function resolveBundledScript(scriptName: string): { cmd: string; args: string[] } {
  if (app.isPackaged) {
    const ext = process.platform === "win32" ? ".exe" : "";
    const bundledPath = path.join(process.resourcesPath, "python", `biovision_backend${ext}`);
    if (fs.existsSync(bundledPath)) {
      return { cmd: bundledPath, args: [scriptName] };
    }
    // Do NOT silently fall through to the dev-mode branch in a packaged app:
    // backend/*.py isn't shipped, so spawning a system `python` against a
    // non-existent script would just produce an opaque failure. Fail loudly
    // so the caller's error surfaces in the UI.
    throw new Error(
      `Bundled Python backend not found at ${bundledPath}. The installer is ` +
        `missing backend/dist/biovision_backend${ext} — rebuild with "npm run backend:build" ` +
        `before packaging, or reinstall the app.`
    );
  }
  // Dev mode: map script name to source path and run via Python interpreter
  const scriptMap: Record<string, string> = {
    prepare_dataset: "data/prepare_dataset.py",
    prepare_imported_dlib_dataset: "data/prepare_imported_dlib_dataset.py",
    train_shape_model: "training/train_shape_model.py",
    train_cnn_model: "training/train_cnn_model.py",
    predict: "inference/predict.py",
    predict_worker: "inference/predict_worker.py",
    shape_tester: "inference/shape_tester.py",
    list_cnn_variants: "inference/list_cnn_variants.py",
    detect_specimen: "detection/detect_specimen.py",
    super_annotator: "annotation/super_annotator.py",
    validate_dlib_xml: "data/validate_dlib_xml.py",
    audit_dataset: "data/audit_dataset.py",
    export_yolo_dataset: "data/export_yolo_dataset.py",
    hardware_probe: "hardware_probe.py",
  };
  const relPath = scriptMap[scriptName];
  const scriptPath = relPath
    ? path.join(__dirname, "../backend", relPath)
    : path.join(__dirname, "../backend", `${scriptName}.py`);
  return { cmd: getPythonPath(), args: [scriptPath] };
}

let warnedAboutSystemPythonFallback = false;

function warnIfUsingSystemPython(pythonResolution = getPythonResolution()): void {
  if (pythonResolution.usingRepoVenv || warnedAboutSystemPythonFallback) return;
  warnedAboutSystemPythonFallback = true;
  console.warn(
    `[Python] Repo venv not found. Falling back to "${pythonResolution.pythonPath}". ` +
      "Hardware probes may report CPU-only if this interpreter is missing torch or psutil."
  );
}

/**
 * Run a bundled backend script by name. In production uses the PyInstaller
 * executable; in dev falls back to the Python interpreter + .py source.
 */
function runBundledScript(scriptName: string, extraArgs: string[] = []): Promise<string> {
  return new Promise((resolve, reject) => {
    const resolved = resolveBundledScript(scriptName);
    const proc = spawn(resolved.cmd, [...resolved.args, ...extraArgs]);

    let out = "";
    let err = "";

    proc.stdout.on("data", (d) => (out += d.toString()));
    proc.stderr.on("data", (d) => (err += d.toString()));

    proc.on("close", (code) => {
      if (code !== 0) {
        return reject(new Error(err || `${scriptName} exited with code ${code}`));
      }
      resolve(out.trim());
    });
  });
}

function runBundledScriptWithProgress(
  scriptName: string,
  extraArgs: string[] = [],
  onProgress?: (percent: number, stage: string, details?: Record<string, unknown>) => void
): Promise<string> {
  return new Promise((resolve, reject) => {
    const resolved = resolveBundledScript(scriptName);
    const proc = spawn(resolved.cmd, [...resolved.args, ...extraArgs]);
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
        return reject(new Error(err || `${scriptName} exited with code ${code}`));
      }
      resolve(out.trim());
    });
  });
}


// ---------------------------------------------------------------------------
// Hardware capability probe Ã¢â‚¬â€ called once at app startup from React
// ---------------------------------------------------------------------------
ipcMain.handle("system:probe-hardware", async () => {
  const pythonResolution = getPythonResolution();
  warnIfUsingSystemPython(pythonResolution);
  try {
    const out = await runBundledScript("hardware_probe");
    const parsed = JSON.parse(out.trim());
    return {
      device: parsed.device ?? "cpu",
      gpuName: parsed.gpu_name ?? null,
      gpuMemoryGb: parsed.gpu_memory_gb ?? null,
      ramGb: parsed.ram_gb ?? null,
      runtimeState: getSuperAnnotatorRuntimeState(),
      statusSource: "python_probe" as CapabilityStatusSource,
      pythonPath: pythonResolution.pythonPath,
      usingRepoVenv: pythonResolution.usingRepoVenv,
    };
  } catch (err) {
    console.warn("Hardware probe failed:", err);
    // Safe fallback Ã¢â‚¬â€ treat as CPU-only
    return {
      device: "cpu",
      gpuName: null,
      gpuMemoryGb: null,
      ramGb: null,
      runtimeState: "failed" as SuperAnnotatorRuntimeState,
      statusSource: "local_estimate" as CapabilityStatusSource,
      pythonPath: pythonResolution.pythonPath,
      usingRepoVenv: pythonResolution.usingRepoVenv,
    };
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

type SessionSchemaKind = "default" | "custom";
type ReusableSchemaTemplateRecord = {
  id: string;
  kind: "custom";
  name: string;
  description: string;
  landmarks: Array<{
    index: number;
    name: string;
    description?: string;
    category?: string;
    required?: boolean;
  }>;
  orientationPolicy?: NormalizedOrientationPolicy;
  sourcePresetId?: string;
  createdAt: string;
  updatedAt: string;
};

type ReusableSchemaLibrary = {
  version: 1;
  templates: ReusableSchemaTemplateRecord[];
};

function normalizeSchemaComponent(value: unknown): string {
  return String(value || "").trim().toLowerCase();
}

function normalizeSchemaSlug(value: string): string {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

function normalizeLandmarkTemplate(
  landmarkTemplate: unknown
): Array<{
  index: number;
  name: string;
  description?: string;
  category?: string;
  required: boolean;
}> {
  if (!Array.isArray(landmarkTemplate)) return [];
  return landmarkTemplate.map((entry: any, position: number) => {
    const normalizedIndex = Number.isFinite(Number(entry?.index))
      ? Math.max(1, Math.round(Number(entry.index)))
      : position + 1;
    return {
      index: normalizedIndex,
      name: String(entry?.name || `Landmark ${normalizedIndex}`).trim() || `Landmark ${normalizedIndex}`,
      ...(String(entry?.description || "").trim()
        ? { description: String(entry.description).trim() }
        : {}),
      ...(String(entry?.category || "").trim()
        ? { category: String(entry.category).trim() }
        : {}),
      required: entry?.required !== false,
    };
  });
}

function normalizeSchemaFingerprintTemplate(
  landmarkTemplate: unknown
): Array<{ index: number; name: string; category: string }> {
  return normalizeLandmarkTemplate(landmarkTemplate).map((entry, position) => ({
    index: Number.isFinite(Number(entry?.index)) ? Number(entry.index) : position + 1,
    name: normalizeSchemaComponent(entry?.name),
    category: normalizeSchemaComponent(entry?.category),
  }));
}

function computeSchemaFingerprint(landmarkTemplate: unknown): string {
  const normalized = normalizeSchemaFingerprintTemplate(landmarkTemplate);
  const input = JSON.stringify(normalized);
  let hash = 2166136261;
  for (let index = 0; index < input.length; index += 1) {
    hash ^= input.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

function computeSessionSchemaSemanticFingerprint(
  landmarkTemplate: unknown,
  rawOrientationPolicy: unknown
): string | undefined {
  if (!rawOrientationPolicy || typeof rawOrientationPolicy !== "object") return undefined;
  const rawMode = String((rawOrientationPolicy as any).mode || "").trim().toLowerCase();
  if (!new Set(["directional", "bilateral", "axial", "invariant"]).has(rawMode)) {
    return undefined;
  }
  const normalizedLandmarks = normalizeLandmarkTemplate(landmarkTemplate);
  const normalizedPolicy = normalizeOrientationPolicy(rawOrientationPolicy, normalizedLandmarks);
  return computeSchemaSemanticFingerprint(normalizedLandmarks as any, normalizedPolicy as any);
}

function normalizeSessionSchemaKind(value: unknown): SessionSchemaKind | undefined {
  return value === "default" || value === "custom" ? value : undefined;
}

function normalizeSessionSchemaSourceId(value: unknown): string | undefined {
  const normalized = String(value || "").trim();
  return normalized.length > 0 ? normalized : undefined;
}

function readReusableSchemaLibrary(): ReusableSchemaLibrary {
  if (!fs.existsSync(customSchemaLibraryPath)) {
    return { version: 1, templates: [] };
  }
  try {
    const parsed = JSON.parse(fs.readFileSync(customSchemaLibraryPath, "utf-8"));
    const templates = Array.isArray(parsed?.templates)
      ? parsed.templates
          .map((entry: any) => normalizeReusableSchemaTemplate(entry))
          .filter(Boolean) as ReusableSchemaTemplateRecord[]
      : [];
    return { version: 1, templates };
  } catch (error) {
    console.warn("Failed to read schema library:", error);
    return { version: 1, templates: [] };
  }
}

function writeReusableSchemaLibrary(library: ReusableSchemaLibrary): void {
  fs.mkdirSync(path.dirname(customSchemaLibraryPath), { recursive: true });
  fs.writeFileSync(customSchemaLibraryPath, JSON.stringify(library, null, 2), "utf-8");
}

function normalizeReusableSchemaTemplate(entry: any): ReusableSchemaTemplateRecord | null {
  const id = String(entry?.id || "").trim();
  const name = String(entry?.name || "").trim();
  if (!id || !name) return null;
  const normalizedLandmarks = normalizeLandmarkTemplate(entry?.landmarks);
  if (normalizedLandmarks.length === 0) return null;
  const createdAt = String(entry?.createdAt || "").trim() || new Date().toISOString();
  const updatedAt = String(entry?.updatedAt || "").trim() || createdAt;
  return {
    id,
    kind: "custom",
    name,
    description: String(entry?.description || "").trim() || `Custom schema with ${normalizedLandmarks.length} landmarks`,
    landmarks: normalizedLandmarks,
    ...(entry?.orientationPolicy && typeof entry.orientationPolicy === "object"
      ? { orientationPolicy: normalizeOrientationPolicy(entry.orientationPolicy, normalizedLandmarks) }
      : {}),
    ...(String(entry?.sourcePresetId || "").trim()
      ? { sourcePresetId: String(entry.sourcePresetId).trim() }
      : {}),
    createdAt,
    updatedAt,
  };
}

function buildLegacyCustomTemplateId(sessionMeta: any): string {
  const sourceId = normalizeSessionSchemaSourceId(sessionMeta?.schemaSourceId);
  if (sourceId) return sourceId;
  const slug = normalizeSchemaSlug(String(sessionMeta?.name || ""));
  const fingerprint = computeSchemaFingerprint(sessionMeta?.landmarkTemplate || []);
  return `legacy-custom-${slug || "untitled"}-${String(fingerprint).slice(0, 8)}`;
}

function bootstrapReusableSchemasFromSessions(
  library: ReusableSchemaLibrary
): ReusableSchemaLibrary {
  const sessionsDir = path.join(projectRoot, "sessions");
  if (!fs.existsSync(sessionsDir)) return library;

  const existingIds = new Set(library.templates.map((template) => template.id));
  let changed = false;
  const nextTemplates = [...library.templates];

  for (const entry of fs.readdirSync(sessionsDir, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const sessionJsonPath = path.join(sessionsDir, entry.name, "session.json");
    if (!fs.existsSync(sessionJsonPath)) continue;
    try {
      const meta = JSON.parse(fs.readFileSync(sessionJsonPath, "utf-8"));
      const schemaKind = normalizeSessionSchemaKind(meta?.schemaKind);
      const normalizedLandmarks = normalizeLandmarkTemplate(meta?.landmarkTemplate);
      if (schemaKind !== "custom" || normalizedLandmarks.length === 0) continue;

      const templateId = buildLegacyCustomTemplateId(meta);
      if (existingIds.has(templateId)) continue;

      nextTemplates.push({
        id: templateId,
        kind: "custom",
        name: String(meta?.name || entry.name).trim() || entry.name,
        description:
          String(meta?.description || "").trim() ||
          `Imported from custom session ${String(meta?.name || entry.name).trim() || entry.name}`,
        landmarks: normalizedLandmarks,
        ...(meta?.orientationPolicy && typeof meta.orientationPolicy === "object"
          ? { orientationPolicy: normalizeOrientationPolicy(meta.orientationPolicy, normalizedLandmarks) }
          : {}),
        createdAt: String(meta?.createdAt || "").trim() || new Date().toISOString(),
        updatedAt:
          String(meta?.lastModified || meta?.createdAt || "").trim() || new Date().toISOString(),
      });
      existingIds.add(templateId);
      changed = true;
    } catch {
      // ignore malformed legacy sessions
    }
  }

  if (!changed) return library;
  const nextLibrary = { version: 1 as const, templates: nextTemplates };
  writeReusableSchemaLibrary(nextLibrary);
  return nextLibrary;
}

function listReusableSchemaTemplates(): ReusableSchemaTemplateRecord[] {
  const library = bootstrapReusableSchemasFromSessions(readReusableSchemaLibrary());
  return [...library.templates].sort((a, b) => {
    return new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime();
  });
}

function resolveSessionSchemaMetadata(meta: any): {
  schemaFingerprint: string;
  schemaSemanticFingerprint?: string;
  schemaSemanticVersion?: number;
  schemaKind?: SessionSchemaKind;
  schemaSourceId?: string;
} {
  const landmarkTemplate = Array.isArray(meta?.landmarkTemplate) ? meta.landmarkTemplate : [];
  const schemaSemanticFingerprint = computeSessionSchemaSemanticFingerprint(
    landmarkTemplate,
    meta?.orientationPolicy
  );
  return {
    schemaFingerprint: computeSchemaFingerprint(landmarkTemplate),
    ...(schemaSemanticFingerprint
      ? {
          schemaSemanticFingerprint,
          schemaSemanticVersion: SCHEMA_SEMANTIC_VERSION,
        }
      : {}),
    ...(normalizeSessionSchemaKind(meta?.schemaKind)
      ? { schemaKind: normalizeSessionSchemaKind(meta?.schemaKind) }
      : {}),
    ...(normalizeSessionSchemaSourceId(meta?.schemaSourceId)
      ? { schemaSourceId: normalizeSessionSchemaSourceId(meta?.schemaSourceId) }
      : {}),
  };
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
type BilateralClassAxis = "vertical_obb";
type ObbModelTier = "nano" | "small" | "medium" | "large";
type ObbImageSize = 640 | 960 | 1280;
type ObbDetectionPreset =
  | "balanced"
  | "precision"
  | "recall"
  | "single_object"
  | "custom";

type NormalizedOrientationPolicy = {
  mode: OrientationMode;
  targetOrientation?: "left" | "right";
  headCategories: string[];
  tailCategories: string[];
  anteriorAnchorIds: number[];
  posteriorAnchorIds: number[];
  bilateralPairs: [number, number][];
  bilateralClassAxis: BilateralClassAxis;
  obbLevelingMode: "on" | "off";
};

type NormalizedObbTrainingSettings = {
  modelTier?: ObbModelTier;
  imgsz?: ObbImageSize;
  epochs?: number;
  batch?: number;
  iou: number;
  cls: number;
  box: number;
};

type NormalizedObbDetectionSettings = {
  detectionPreset: ObbDetectionPreset;
  conf: number;
  nmsIou: number;
  maxObjects: number;
  imgsz: ObbImageSize;
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
  targetOrientation?: "left" | "right";
  headCategories: string[];
  tailCategories: string[];
  anteriorAnchorIds: number[];
  posteriorAnchorIds: number[];
  bilateralPairs: [number, number][];
  bilateralClassAxis: BilateralClassAxis;
  obbLevelingMode: "on" | "off";
  schemaSemanticFingerprint?: string;
  schemaSemanticVersion?: number;
  landmarkContract: Array<{
    index: number;
    name?: string;
    required?: boolean;
  }>;
  landmarkIds: number[];
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
  obbDetector?: ObbDetectorResolution;
};

const COMPAT_ORIENTATION_MODES = new Set<OrientationMode>([
  "directional",
  "bilateral",
  "axial",
  "invariant",
]);
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

function normalizeAnchorIdList(value: unknown): number[] {
  if (!Array.isArray(value)) return [];
  const seen = new Set<number>();
  const out: number[] = [];
  value.forEach((item) => {
    const id = Math.round(Number(item));
    if (!Number.isFinite(id) || id < 1 || seen.has(id)) return;
    seen.add(id);
    out.push(id);
  });
  out.sort((a, b) => a - b);
  return out;
}

function clampNumber(value: unknown, fallback: number, min: number, max: number): number {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return Math.min(max, Math.max(min, numeric));
}

function normalizeOptionalPositiveInt(
  value: unknown,
  minimum: number,
  maximum: number
): number | undefined {
  if (value === undefined || value === null || value === "") return undefined;
  const numeric = Math.round(Number(value));
  if (!Number.isFinite(numeric)) return undefined;
  return Math.min(maximum, Math.max(minimum, numeric));
}

function normalizeObbModelTier(value: unknown): ObbModelTier | undefined {
  return value === "nano" || value === "small" || value === "medium" || value === "large"
    ? value
    : undefined;
}

function normalizeObbImageSize(value: unknown, fallback: ObbImageSize): ObbImageSize {
  const numeric = Math.round(Number(value));
  if (numeric === 960) return 960;
  if (numeric === 1280) return 1280;
  if (numeric === 640) return 640;
  return fallback;
}

function normalizeObbDetectionPreset(value: unknown): ObbDetectionPreset {
  if (
    value === "balanced" ||
    value === "precision" ||
    value === "recall" ||
    value === "single_object" ||
    value === "custom"
  ) {
    return value;
  }
  return "balanced";
}

function normalizeObbTrainingSettings(rawSettings: unknown): NormalizedObbTrainingSettings {
  const raw = rawSettings && typeof rawSettings === "object"
    ? (rawSettings as Record<string, unknown>)
    : {};
  return {
    ...(normalizeObbModelTier(raw.modelTier) ? { modelTier: normalizeObbModelTier(raw.modelTier) } : {}),
    ...(raw.imgsz !== undefined ? { imgsz: normalizeObbImageSize(raw.imgsz, 640) } : {}),
    ...(normalizeOptionalPositiveInt(raw.epochs, 1, 500) !== undefined
      ? { epochs: normalizeOptionalPositiveInt(raw.epochs, 1, 500) }
      : {}),
    ...(normalizeOptionalPositiveInt(raw.batch, 1, 128) !== undefined
      ? { batch: normalizeOptionalPositiveInt(raw.batch, 1, 128) }
      : {}),
    iou: clampNumber(raw.iou, 0.3, 0.05, 0.95),
    cls: clampNumber(raw.cls, 1.5, 0.1, 10.0),
    box: clampNumber(raw.box, 5.0, 0.1, 20.0),
  };
}

function buildInferenceSchemaGroupKey(speciesId: string, meta: any): string {
  const schemaKind = normalizeSessionSchemaKind(meta?.schemaKind);
  const schemaSourceId = normalizeSessionSchemaSourceId(meta?.schemaSourceId);
  if (schemaKind && schemaSourceId) {
    return `${schemaKind}:${normalizeSchemaSlug(schemaSourceId) || schemaSourceId}`;
  }

  const schemaName = String(meta?.name || "").trim();
  const normalizedName = normalizeSchemaSlug(schemaName);
  if (normalizedName) {
    return `name:${normalizedName}`;
  }

  return `species:${speciesId}`;
}

type InferenceSessionListEntry = {
  speciesId: string;
  schemaName: string;
  schemaImageCount: number;
  schemaUpdatedAt: string;
  schemaGroupKey: string;
  canonicalSpeciesId: string;
  hiddenSessionCount?: number;
  exists: boolean;
  inferenceSessionId?: string;
  displayName?: string;
  createdAt?: string;
  updatedAt?: string;
  migratedFrom?: string;
};

function parseSortableTimestamp(...values: Array<string | undefined>): number {
  for (const value of values) {
    if (!value) continue;
    const parsed = Date.parse(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return 0;
}

function normalizeObbDetectionSettings(rawSettings: unknown): NormalizedObbDetectionSettings {
  const raw = rawSettings && typeof rawSettings === "object"
    ? (rawSettings as Record<string, unknown>)
    : {};
  return {
    detectionPreset: normalizeObbDetectionPreset(raw.detectionPreset),
    conf: clampNumber(raw.conf, 0.3, 0.01, 0.99),
    nmsIou: clampNumber(raw.nmsIou, 0.3, 0.05, 0.95),
    maxObjects: Math.round(clampNumber(raw.maxObjects, 20, 1, 250)),
    imgsz: normalizeObbImageSize(raw.imgsz, 640),
  };
}

function inferOrientationPolicyFromTemplate(landmarkTemplate: unknown): NormalizedOrientationPolicy {
  const normalizedLandmarks = normalizeLandmarkTemplate(landmarkTemplate);
  const categories = new Set<string>();
  normalizedLandmarks.forEach((lm: any) => {
      const cat = String(lm?.category || "").trim().toLowerCase();
      if (cat) categories.add(cat);
  });
  const hasHead = categories.has("head");
  const inferredTail = ["tail", "caudal-fin"].filter((cat) => categories.has(cat));
  const tailCategories = inferredTail.length > 0 ? inferredTail : ["tail", "caudal-fin"];
  const findByName = (name: string) =>
    normalizedLandmarks.find((lm) => String(lm?.name || "").trim().toLowerCase() === name)?.index;
  if (hasHead || inferredTail.length > 0) {
    const snoutTip = findByName("snout tip");
    const upperCaudal = findByName("upper caudal peduncle");
    const lowerCaudal = findByName("lower caudal peduncle");
    return {
      mode: "directional",
      targetOrientation: "left",
      headCategories: hasHead ? ["head"] : [],
      tailCategories,
      anteriorAnchorIds: Number.isFinite(Number(snoutTip)) ? [Number(snoutTip)] : [],
      posteriorAnchorIds:
        Number.isFinite(Number(upperCaudal)) && Number.isFinite(Number(lowerCaudal))
          ? [Number(upperCaudal), Number(lowerCaudal)]
          : [],
      bilateralPairs: [],
      bilateralClassAxis: "vertical_obb",
      obbLevelingMode: "on",
    };
  }
  return {
    mode: "invariant",
    headCategories: [],
    tailCategories: [],
    anteriorAnchorIds: [],
    posteriorAnchorIds: [],
    bilateralPairs: [],
    bilateralClassAxis: "vertical_obb",
    obbLevelingMode: "on",
  };
}

function normalizeBilateralClassAxis(
  value: unknown,
  mode: OrientationMode
): BilateralClassAxis {
  if (
    mode === "bilateral" &&
    value != null &&
    String(value).trim() !== "" &&
    String(value).trim().toLowerCase() !== "vertical_obb"
  ) {
    console.warn(
      `[Session] Unsupported bilateralClassAxis "${String(value)}" encountered; normalizing to vertical_obb.`
    );
  }
  return "vertical_obb";
}

function normalizeOrientationPolicy(
  rawPolicy: unknown,
  landmarkTemplate: unknown
): NormalizedOrientationPolicy {
  const inferred = inferOrientationPolicyFromTemplate(landmarkTemplate);
  const hasExplicitPolicy = Boolean(rawPolicy && typeof rawPolicy === "object");
  const raw = hasExplicitPolicy ? (rawPolicy as Record<string, unknown>) : {};

  const mode = normalizeOrientationMode(raw.mode ?? inferred.mode);
  const rawTarget = String(raw.targetOrientation || inferred.targetOrientation || "left").trim().toLowerCase();
  const targetOrientation = rawTarget === "right" ? "right" : "left";

  const headCategories = normalizeCategoryList(
    hasExplicitPolicy ? raw.headCategories : inferred.headCategories
  );
  const tailCategories = normalizeCategoryList(
    hasExplicitPolicy ? raw.tailCategories : inferred.tailCategories
  );

  const anteriorAnchorIds = normalizeAnchorIdList(
    hasExplicitPolicy ? raw.anteriorAnchorIds : inferred.anteriorAnchorIds
  );
  const posteriorAnchorIds = normalizeAnchorIdList(
    hasExplicitPolicy ? raw.posteriorAnchorIds : inferred.posteriorAnchorIds
  );

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

  const rawObbLevelingMode = String(raw.obbLevelingMode || inferred.obbLevelingMode || "on").trim().toLowerCase();
  const policy: NormalizedOrientationPolicy = {
    mode,
    headCategories: mode === "directional" ? headCategories : [],
    tailCategories: mode === "directional" ? tailCategories : [],
    anteriorAnchorIds,
    posteriorAnchorIds,
    bilateralPairs: mode === "bilateral" ? normalizedPairs : [],
    bilateralClassAxis: normalizeBilateralClassAxis(raw.bilateralClassAxis, mode),
    obbLevelingMode: rawObbLevelingMode === "off" ? "off" : "on",
  };
  if (mode === "directional") {
    policy.targetOrientation = targetOrientation;
  }
  return policy;
}

function readNormalizedSessionObbTrainingSettings(sessionMeta: unknown): NormalizedObbTrainingSettings {
  const raw = sessionMeta && typeof sessionMeta === "object"
    ? (sessionMeta as Record<string, unknown>).obbTrainingSettings
    : undefined;
  return normalizeObbTrainingSettings(raw);
}

function readNormalizedSessionObbDetectionSettings(sessionMeta: unknown): NormalizedObbDetectionSettings {
  const raw = sessionMeta && typeof sessionMeta === "object"
    ? (sessionMeta as Record<string, unknown>).obbDetectionSettings
    : undefined;
  return normalizeObbDetectionSettings(raw);
}

function readSessionObbTrainingSettingsCustomized(sessionMeta: unknown): boolean {
  const raw = sessionMeta && typeof sessionMeta === "object"
    ? (sessionMeta as Record<string, unknown>)
    : {};
  if (typeof raw.obbTrainingSettingsCustomized === "boolean") {
    return raw.obbTrainingSettingsCustomized;
  }
  const normalized = normalizeObbTrainingSettings(raw.obbTrainingSettings);
  const defaults = normalizeObbTrainingSettings(undefined);
  return JSON.stringify(normalized) !== JSON.stringify(defaults);
}

function readSessionObbDetectionSettingsCustomized(sessionMeta: unknown): boolean {
  const raw = sessionMeta && typeof sessionMeta === "object"
    ? (sessionMeta as Record<string, unknown>)
    : {};
  if (typeof raw.obbDetectionSettingsCustomized === "boolean") {
    return raw.obbDetectionSettingsCustomized;
  }
  const normalized = normalizeObbDetectionSettings(raw.obbDetectionSettings);
  const defaults = normalizeObbDetectionSettings(undefined);
  return JSON.stringify(normalized) !== JSON.stringify(defaults);
}

function loadSessionOrientationPolicyForCompatibility(
  speciesId: string
): {
  policy: NormalizedOrientationPolicy;
  landmarkTemplate: any[];
  schemaSemanticFingerprint?: string;
  schemaSemanticVersion?: number;
} {
  const sessionPath = path.join(getEffectiveRoot(speciesId), "session.json");
  const sessionRaw = safeReadJson(sessionPath) || {};
  const template = Array.isArray(sessionRaw.landmarkTemplate) ? sessionRaw.landmarkTemplate : [];
  return {
    policy: normalizeOrientationPolicy(sessionRaw.orientationPolicy, template),
    landmarkTemplate: template,
    ...(computeSessionSchemaSemanticFingerprint(template, sessionRaw.orientationPolicy)
      ? {
          schemaSemanticFingerprint: computeSessionSchemaSemanticFingerprint(
            template,
            sessionRaw.orientationPolicy
          ),
          schemaSemanticVersion: SCHEMA_SEMANTIC_VERSION,
        }
      : {}),
  };
}

function resolveSessionObbDetectorForCompatibility(
  speciesId: string
): ObbDetectorResolution | undefined {
  const effectiveRoot = getEffectiveRoot(speciesId);
  return resolveObbDetectorForEffectiveRoot(effectiveRoot);
}

function resolveObbDetectorForEffectiveRoot(
  effectiveRoot: string
): ObbDetectorResolution | undefined {
  const aliasPath = path.join(effectiveRoot, "models", "session_obb_detector.pt");
  if (!fs.existsSync(aliasPath)) return undefined;
  const rawSession = safeReadJson(path.join(effectiveRoot, "session.json")) || {};
  const template = Array.isArray(rawSession.landmarkTemplate)
    ? rawSession.landmarkTemplate
    : [];
  const schemaSemanticFingerprint = computeSessionSchemaSemanticFingerprint(
    template,
    rawSession.orientationPolicy
  );
  return resolveTrainedObbDetector({
    aliasPath,
    registryPath: path.join(effectiveRoot, "models", "obb_registry.json"),
    sessionSemanticFingerprint: schemaSemanticFingerprint,
    sessionSemanticVersion: schemaSemanticFingerprint
      ? SCHEMA_SEMANTIC_VERSION
      : undefined,
    sessionOrientationContract: rawSession.orientationPolicy,
  });
}

function landmarkTrainingObbGateError(
  resolution: ObbDetectorResolution | undefined
): string | null {
  if (isVerifiedTrainedObbDetector(resolution)) return null;
  const detail = resolution?.issues.find((issue) => issue.severity === "error")?.message;
  return (
    "Landmark predictor training is locked until this session has a verified, active OBB detector. " +
    (detail || "Train and promote the OBB detector in step 1, then retry.")
  );
}

function loadModelTrainingProfileForCompatibility(
  speciesId: string,
  modelName: string,
  predictorType: "dlib" | "cnn"
): ModelTrainingProfile {
  const effectiveRoot = getEffectiveRoot(speciesId);
  const metadataSources: string[] = [];
  const fallbackSession = loadSessionOrientationPolicyForCompatibility(speciesId);
  const registryEntry = findLandmarkRegistryEntry(
    readLandmarkModelRegistry(effectiveRoot),
    modelName,
    predictorType
  );
  const manifestCandidates = [
    registryEntry?.runManifestPath,
    registryEntry?.path ? path.join(path.dirname(registryEntry.path), "manifest.json") : undefined,
  ].filter((candidate): candidate is string => Boolean(candidate));
  let runManifest: any = null;
  for (const candidate of manifestCandidates) {
    const parsed = safeReadJson(candidate);
    if (parsed && typeof parsed === "object") {
      runManifest = parsed;
      metadataSources.push(candidate);
      break;
    }
  }
  const schemaSnapshot = runManifest?.lineage?.schema;
  const landmarkContract = Array.isArray(schemaSnapshot?.semantics?.landmarks)
    ? schemaSnapshot.semantics.landmarks
    : [];

  const immutableIdMappingPath = String(
    registryEntry?.sidecars?.idMapping?.path || ""
  ).trim();
  const idMappingPath = immutableIdMappingPath ||
    path.join(effectiveRoot, "debug", `id_mapping_${modelName}.json`);
  const idMappingRaw = safeReadJson(idMappingPath);
  let rawTrainingConfig: Record<string, any> =
    idMappingRaw && typeof idMappingRaw === "object"
      ? ((idMappingRaw.training_config || {}) as Record<string, any>)
      : {};
  if (idMappingRaw && typeof idMappingRaw === "object") {
    metadataSources.push(idMappingPath);
  }
  let cnnConfigRaw: any = null;
  if (predictorType === "cnn") {
    const cnnTrainParamsPath = path.join(effectiveRoot, "debug", `training_params_${modelName}_cnn.json`);
    const cnnTrainParamsRaw = safeReadJson(cnnTrainParamsPath);
    if (cnnTrainParamsRaw && typeof cnnTrainParamsRaw === "object") {
      rawTrainingConfig = {
        ...rawTrainingConfig,
        orientation_mode: cnnTrainParamsRaw.orientation_mode,
        orientation_policy: cnnTrainParamsRaw.orientation_policy,
        canonical_training_enabled: cnnTrainParamsRaw.canonical_training_enabled,
        target_orientation: cnnTrainParamsRaw.target_orientation,
      };
      metadataSources.push(cnnTrainParamsPath);
    }

    const cnnConfigPath = path.join(effectiveRoot, "models", `cnn_${modelName}_config.json`);
    cnnConfigRaw = safeReadJson(registryEntry?.configPath || cnnConfigPath);
    if (cnnConfigRaw && typeof cnnConfigRaw === "object") {
      metadataSources.push(registryEntry?.configPath || cnnConfigPath);
    }
  }

  const rawLandmarkIds: unknown[] = landmarkContract.length > 0
    ? landmarkContract.map((entry: any) => entry?.index)
    : Array.isArray(idMappingRaw?.original_ids)
      ? idMappingRaw.original_ids
      : Array.isArray(cnnConfigRaw?.landmark_ids)
        ? cnnConfigRaw.landmark_ids
        : [];
  const landmarkIds = [...new Set<number>(
    rawLandmarkIds
      .map((value) => Number(value))
      .filter((value) => Number.isFinite(value))
      .map((value: number) => Math.max(1, Math.round(value)))
  )].sort((left, right) => left - right);

  const trainingPolicy = normalizeOrientationPolicy(
    rawTrainingConfig.orientation_policy,
    fallbackSession.landmarkTemplate
  );
  const orientationMode = normalizeOrientationMode(
    rawTrainingConfig.orientation_mode ?? trainingPolicy.mode
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
      : orientationMode !== "invariant";
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
    targetOrientation: orientationMode === "directional" ? targetOrientation : undefined,
    headCategories:
      orientationMode === "directional"
        ? normalizeCategoryList(trainingPolicy.headCategories)
        : [],
    tailCategories:
      orientationMode === "directional"
        ? normalizeCategoryList(trainingPolicy.tailCategories)
        : [],
    anteriorAnchorIds: normalizeAnchorIdList(trainingPolicy.anteriorAnchorIds),
    posteriorAnchorIds: normalizeAnchorIdList(trainingPolicy.posteriorAnchorIds),
    bilateralPairs:
      orientationMode === "bilateral"
        ? (trainingPolicy.bilateralPairs || []).map((pair) => [pair[0], pair[1]] as [number, number])
        : [],
    bilateralClassAxis: trainingPolicy.bilateralClassAxis,
    obbLevelingMode: trainingPolicy.obbLevelingMode,
    ...(typeof schemaSnapshot?.semanticFingerprint === "string" && schemaSnapshot.semanticFingerprint.trim()
      ? { schemaSemanticFingerprint: schemaSnapshot.semanticFingerprint.trim() }
      : {}),
    ...(Number.isFinite(Number(schemaSnapshot?.semanticVersion))
      ? { schemaSemanticVersion: Number(schemaSnapshot.semanticVersion) }
      : {}),
    landmarkContract,
    landmarkIds,
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
    const issues: ModelCompatibilityIssue[] = compareModelSchemaContract({
      sessionSemanticFingerprint: session.schemaSemanticFingerprint,
      sessionLandmarks: session.landmarkTemplate,
      modelSemanticFingerprint: profile.schemaSemanticFingerprint,
      modelSemanticVersion: profile.schemaSemanticVersion,
      modelLandmarks: profile.landmarkContract,
      modelLandmarkIds: profile.landmarkIds,
    });

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
      if (profile.bilateralClassAxis !== session.policy.bilateralClassAxis) {
        issues.push({
          code: "bilateral_class_axis_mismatch",
          severity: "error",
          message: `Model bilateral class axis is "${profile.bilateralClassAxis}" but session uses "${session.policy.bilateralClassAxis}".`,
        });
      }
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
      if (profile.canonicalMaskSource === "unknown") {
        issues.push({
          code: "canonical_mask_source_unknown",
          severity: "warning",
          message:
            "Training mask source metadata is missing/unknown; SAM2 parity requirements cannot be fully verified.",
        });
      }
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

    // The landmark model is only safe in the same pipeline when the exact active
    // OBB alias also satisfies the immutable session schema contract.
    const effectiveRoot = getEffectiveRoot(args.speciesId);
    const obbDetectorPath = path.join(effectiveRoot, "models", "session_obb_detector.pt");
    const obbDetector = resolveSessionObbDetectorForCompatibility(args.speciesId);
    if (obbDetector) {
      issues.push(...obbDetector.issues);
    }

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
      obbDetectorReady: isVerifiedTrainedObbDetector(obbDetector),
      obbDetectorPath: isVerifiedTrainedObbDetector(obbDetector)
        ? obbDetectorPath
        : undefined,
      ...(obbDetector ? { obbDetector } : {}),
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
  orientation_override?: "left" | "right" | "up" | "down" | "uncertain";
  orientation_hint?: {
    orientation?: "left" | "right" | "up" | "down";
    confidence?: number;
    source?: string;
  };
  obbCorners?: [number, number][];
  angle?: number;
  class_id?: number;
  landmarks?: FinalizedAcceptedLandmark[];
  trainingTargets?: Array<"landmark" | "obb">;
};

function normalizeFinalizedAcceptedBoxes(
  rawBoxes: Array<{
    left: number;
    top: number;
    width: number;
    height: number;
    orientation_override?: "left" | "right" | "up" | "down" | "uncertain";
    orientation_hint?: {
      orientation?: "left" | "right" | "up" | "down";
      confidence?: number;
      source?: string;
    };
    obbCorners?: [number, number][];
    angle?: number;
    class_id?: number;
    landmarks?: Array<{ id: number; x: number; y: number; isSkipped?: boolean }>;
    trainingTargets?: Array<"landmark" | "obb">;
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
      raw?.orientation_override === "up" ||
      raw?.orientation_override === "down" ||
      raw?.orientation_override === "uncertain"
    ) {
      box.orientation_override = raw.orientation_override;
    }
    if (
      raw?.orientation_hint?.orientation === "left" ||
      raw?.orientation_hint?.orientation === "right" ||
      raw?.orientation_hint?.orientation === "up" ||
      raw?.orientation_hint?.orientation === "down"
    ) {
      box.orientation_hint = {
        orientation: raw.orientation_hint.orientation,
        ...(isFiniteNumber(raw?.orientation_hint?.confidence)
          ? { confidence: Number(raw.orientation_hint.confidence) }
          : {}),
        ...(typeof raw?.orientation_hint?.source === "string" && raw.orientation_hint.source.trim()
          ? { source: raw.orientation_hint.source.trim() }
          : {}),
      };
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
    if (Array.isArray(raw?.trainingTargets)) {
      const targets = Array.from(
        new Set(
          raw.trainingTargets.filter(
            (target): target is "landmark" | "obb" =>
              target === "landmark" || target === "obb"
          )
        )
      );
      if (targets.length > 0) {
        box.trainingTargets = targets;
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

function buildAcceptedBoxesSignature(boxes: Array<{
  left: number;
  top: number;
  width: number;
  height: number;
  obbCorners?: [number, number][];
  angle?: number;
  class_id?: number;
  orientation_override?: "left" | "right" | "up" | "down" | "uncertain";
}>): string {
  const reduced = (boxes || [])
    .map((b) => ({
      left: Math.round(Number(b.left) || 0),
      top: Math.round(Number(b.top) || 0),
      width: Math.round(Number(b.width) || 0),
      height: Math.round(Number(b.height) || 0),
      ...(Array.isArray(b.obbCorners) && b.obbCorners.length === 4
        ? {
            obbCorners: b.obbCorners.map((point) => [
              Number(Number(point?.[0] || 0).toFixed(3)),
              Number(Number(point?.[1] || 0).toFixed(3)),
            ]),
          }
        : {}),
      ...(isFiniteNumber(b.angle) ? { angle: Number(Number(b.angle).toFixed(3)) } : {}),
      ...(isFiniteNumber(b.class_id) ? { class_id: Math.round(Number(b.class_id)) } : {}),
      ...(b.orientation_override ? { orientation_override: b.orientation_override } : {}),
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
  counts?: { expectedCount?: number; savedCount?: number },
  details?: SegmentSaveDetail[]
): void {
  const key = buildSegmentQueueImageKey(speciesId, filename);
  const entry: SegmentQueueStatusEntry = {
    state,
    updatedAt: new Date().toISOString(),
    ...(signature ? { signature } : {}),
    ...(reason ? { reason } : {}),
    ...(counts?.expectedCount != null ? { expectedCount: counts.expectedCount } : {}),
    ...(counts?.savedCount != null ? { savedCount: counts.savedCount } : {}),
    ...(details?.length ? { details } : {}),
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

function summarizeSegmentSaveDetails(details?: SegmentSaveDetail[]): string | undefined {
  if (!details || details.length === 0) return undefined;
  const firstFailure = details.find((detail) => detail.status === "failed" && detail.reason);
  if (!firstFailure?.reason) return undefined;
  return String(firstFailure.reason).trim();
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

function summarizeRepresentativeImageDimensions(
  imagePaths: string[],
  maxSamples = 12
): { width: number; height: number; sampleCount: number; megapixels: number } | undefined {
  const widths: number[] = [];
  const heights: number[] = [];

  for (const imagePath of imagePaths.slice(0, maxSamples)) {
    const dims = getImageDimensions(imagePath);
    if (!dims || dims.width <= 0 || dims.height <= 0) continue;
    widths.push(Math.round(dims.width));
    heights.push(Math.round(dims.height));
  }

  if (widths.length === 0 || heights.length === 0) return undefined;

  const medianValue = (values: number[]) => {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    if (sorted.length % 2 === 1) return sorted[mid];
    return Math.round((sorted[mid - 1] + sorted[mid]) / 2);
  };

  const width = medianValue(widths);
  const height = medianValue(heights);
  return {
    width,
    height,
    sampleCount: widths.length,
    megapixels: Number(((width * height) / 1_000_000).toFixed(2)),
  };
}

function normalizeLandmarks(
  rawLandmarks: any[],
  context: string,
  options?: NormalizeLandmarkOptions
): Array<{ id: number; x: number; y: number; isSkipped?: boolean }> {
  const normalized: Array<{ id: number; x: number; y: number; isSkipped?: boolean }> = [];
  rawLandmarks.forEach((landmark: any, landmarkIndex: number) => {
    const skipped = Boolean(landmark?.isSkipped);
    const rawId = landmark?.id;
    const fallbackTemplateIndex = Number(options?.fallbackLandmarkTemplate?.[landmarkIndex]?.index);
    const id = isFiniteNumber(rawId)
      ? Number(rawId)
      : isFiniteNumber(fallbackTemplateIndex)
        ? fallbackTemplateIndex
        : landmarkIndex + 1;
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
    if (!isFiniteNumber(rawId) && options?.metadata) {
      options.metadata.missingLandmarkIdsMappedFromSchemaOrder += 1;
    }
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
  orientation_hint?: {
    orientation?: "left" | "right" | "up" | "down";
    confidence?: number;
    source?: string;
  };
  geometryOrigin?: "source" | "derived";
};

type ImportGeometryConfig = {
  axisMode: "auto" | "manual_anchors";
  anchorLandmarkIds?: {
    anteriorIds: number[];
    posteriorIds: number[];
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
    anteriorIds: number[];
    posteriorIds: number[];
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
  sourceObbReorientedToAnchors: number;
  translatedToFitImage: number;
  manualAnchorDerived: number;
  autoDerived: number;
  fallbackBoxes: number;
  usedAsymmetricPadding: boolean;
  xmlLandmarkIdsPreservedAsSchemaIndexed: number;
  xmlLandmarkIdsShiftedToOneBased: number;
  xmlLandmarkIdsAmbiguous: number;
  missingLandmarkIdsMappedFromSchemaOrder: number;
};

type NormalizeLandmarkOptions = {
  fallbackLandmarkTemplate?: Array<{ index?: number }>;
  metadata?: {
    missingLandmarkIdsMappedFromSchemaOrder: number;
  };
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

function getImportedObbBounds(
  corners: ImportedObbCorners
): { minX: number; maxX: number; minY: number; maxY: number } {
  const xs = corners.map(([x]) => Number(x));
  const ys = corners.map(([, y]) => Number(y));
  return {
    minX: Math.min(...xs),
    maxX: Math.max(...xs),
    minY: Math.min(...ys),
    maxY: Math.max(...ys),
  };
}

function isImportedPointInsideObb(
  point: { x: number; y: number },
  corners: ImportedObbCorners,
  tolerance = 1e-6
): boolean {
  if (!corners || corners.length !== 4) return false;
  const cp0 = corners[0];
  const hAxisX = Number(corners[1][0]) - Number(cp0[0]);
  const hAxisY = Number(corners[1][1]) - Number(cp0[1]);
  const vAxisX = Number(corners[3][0]) - Number(cp0[0]);
  const vAxisY = Number(corners[3][1]) - Number(cp0[1]);
  const widthSq = hAxisX * hAxisX + hAxisY * hAxisY;
  const heightSq = vAxisX * vAxisX + vAxisY * vAxisY;
  if (widthSq <= tolerance || heightSq <= tolerance) return false;
  const relX = Number(point.x) - Number(cp0[0]);
  const relY = Number(point.y) - Number(cp0[1]);
  const hProj = (relX * hAxisX + relY * hAxisY) / widthSq;
  const vProj = (relX * vAxisX + relY * vAxisY) / heightSq;
  return hProj >= -tolerance && hProj <= 1 + tolerance && vProj >= -tolerance && vProj <= 1 + tolerance;
}

function importedObbContainsAllLandmarks(
  corners: ImportedObbCorners,
  landmarks: ImportedLandmark[]
): boolean {
  const valid = getValidImportedLandmarks(landmarks);
  return valid.every((landmark) => isImportedPointInsideObb(landmark, corners));
}

function fitImportedObbCornersRigidly(
  corners: ImportedObbCorners,
  imageDims: { width: number; height: number } | null,
  landmarks: ImportedLandmark[] = []
): { corners: ImportedObbCorners; translated: boolean } {
  if (!imageDims) {
    return { corners, translated: false };
  }
  const maxXBound = Math.max(0, Number(imageDims.width) - 1);
  const maxYBound = Math.max(0, Number(imageDims.height) - 1);
  let resolvedCorners = corners.map(([x, y]) => [Number(x), Number(y)] as [number, number]) as ImportedObbCorners;
  let translated = false;

  const fittedBounds = getImportedObbBounds(resolvedCorners);
  let dx = 0;
  let dy = 0;
  if (fittedBounds.minX < 0) dx = -fittedBounds.minX;
  if (fittedBounds.maxX + dx > maxXBound) {
    dx += maxXBound - (fittedBounds.maxX + dx);
  }
  if (fittedBounds.minY < 0) dy = -fittedBounds.minY;
  if (fittedBounds.maxY + dy > maxYBound) {
    dy += maxYBound - (fittedBounds.maxY + dy);
  }
  if (Math.abs(dx) > 1e-6 || Math.abs(dy) > 1e-6) {
    const translatedCorners = resolvedCorners.map(([x, y]) => ([Number(x) + dx, Number(y) + dy])) as ImportedObbCorners;
    if (landmarks.length === 0 || importedObbContainsAllLandmarks(translatedCorners, landmarks)) {
      resolvedCorners = translatedCorners;
      translated = true;
    }
  }

  return { corners: resolvedCorners, translated };
}

function normalizeImportedObbCorners(
  rawCorners: unknown,
  _imageDims: { width: number; height: number } | null
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
  return corners as ImportedObbCorners;
}

function importedCornersToAabb(
  corners: ImportedObbCorners,
  _imageDims: { width: number; height: number } | null
): { left: number; top: number; width: number; height: number } {
  const xs = corners.map((c) => Number(c[0]));
  const ys = corners.map((c) => Number(c[1]));
  const left = Math.min(...xs);
  const top = Math.min(...ys);
  const right = Math.max(...xs);
  const bottom = Math.max(...ys);
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
    const expandedCorners = fitImportedObbCornersRigidly(
      buildImportedObbCorners(
        expandedCenterX,
        expandedCenterY,
        width + padForward + padBackward,
        height + padTop + padBottom,
        angle
      ),
      imageDims
    ).corners;
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

function resolveImportedAnchorCentroid(
  landmarks: ImportedLandmark[],
  ids: number[] | undefined
): { id: number; x: number; y: number } | null {
  const normalizedIds = normalizeAnchorIdList(ids);
  if (normalizedIds.length === 0) return null;
  const points = normalizedIds
    .map((id) => findImportedLandmarkById(landmarks, id))
    .filter(Boolean) as ImportedLandmark[];
  if (points.length === 0) return null;
  return {
    id: normalizedIds[0],
    x: points.reduce((sum, point) => sum + Number(point.x), 0) / points.length,
    y: points.reduce((sum, point) => sum + Number(point.y), 0) / points.length,
  };
}

function resolveImportedSemanticAnchors(
  landmarks: ImportedLandmark[],
  orientationPolicy: NormalizedOrientationPolicy,
  landmarkTemplate: any[],
  geometryConfig?: ImportGeometryConfig
): { head: { id: number; x: number; y: number } | null; tail: { id: number; x: number; y: number } | null; source: "manual_anchors" | "schema_anchors" | "head_tail" | "none" } {
  const resolvedGeometry = normalizeImportGeometryConfig(geometryConfig);
  if (resolvedGeometry.axisMode === "manual_anchors") {
    const head = resolveImportedAnchorCentroid(landmarks, resolvedGeometry.anchorLandmarkIds?.anteriorIds);
    const tail = resolveImportedAnchorCentroid(landmarks, resolvedGeometry.anchorLandmarkIds?.posteriorIds);
    if (head && tail) return { head, tail, source: "manual_anchors" };
  }

  const schemaHead = resolveImportedAnchorCentroid(landmarks, orientationPolicy.anteriorAnchorIds);
  const schemaTail = resolveImportedAnchorCentroid(landmarks, orientationPolicy.posteriorAnchorIds);
  if (schemaHead && schemaTail) {
    return { head: schemaHead, tail: schemaTail, source: "schema_anchors" };
  }

  const { headId, tailId } = resolveHeadTailLandmarkIdsForImport(orientationPolicy, landmarkTemplate);
  const head = findImportedLandmarkById(landmarks, headId);
  const tail = findImportedLandmarkById(landmarks, tailId);
  if (head && tail) return { head, tail, source: "head_tail" };
  return { head: null, tail: null, source: "none" };
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
  const anteriorIds = normalizeAnchorIdList(raw?.anchorLandmarkIds?.anteriorIds);
  const posteriorIds = normalizeAnchorIdList(raw?.anchorLandmarkIds?.posteriorIds);
  const anchorLandmarkIds =
    axisMode === "manual_anchors" &&
    anteriorIds.length > 0 &&
    posteriorIds.length > 0 &&
    anteriorIds.every((id) => !posteriorIds.includes(id))
      ? {
          anteriorIds,
          posteriorIds,
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

function normalizeImportedVector(x: number, y: number): { x: number; y: number } | null {
  const mag = Math.hypot(Number(x), Number(y));
  if (mag <= 1e-6) return null;
  return { x: Number(x) / mag, y: Number(y) / mag };
}

function usesImportedVerticalSemanticAxis(
  orientationPolicy: NormalizedOrientationPolicy
): boolean {
  return (
    orientationPolicy.mode === "axial" ||
    (
      orientationPolicy.mode === "bilateral" &&
      orientationPolicy.bilateralClassAxis === "vertical_obb"
    )
  );
}

function resolveImportedObbAxes(
  semanticAxisX: number,
  semanticAxisY: number,
  orientationPolicy: NormalizedOrientationPolicy
): { hx: number; hy: number; vx: number; vy: number } {
  const axis = normalizeImportedVector(semanticAxisX, semanticAxisY) ?? { x: 1, y: 0 };
  if (usesImportedVerticalSemanticAxis(orientationPolicy)) {
    // Vertical schemas treat the semantic axis as biological top -> bottom.
    return {
      hx: axis.y,
      hy: -axis.x,
      vx: axis.x,
      vy: axis.y,
    };
  }
  // Horizontal schemas treat the semantic axis as biological left -> right.
  return {
    hx: axis.x,
    hy: axis.y,
    vx: -axis.y,
    vy: axis.x,
  };
}

function buildImportedObbCornersFromAxes(
  cx: number,
  cy: number,
  width: number,
  height: number,
  hx: number,
  hy: number,
  vx: number,
  vy: number
): ImportedObbCorners {
  const hw = width / 2;
  const hh = height / 2;
  return [
    [cx - hw * hx - hh * vx, cy - hw * hy - hh * vy],
    [cx + hw * hx - hh * vx, cy + hw * hy - hh * vy],
    [cx + hw * hx + hh * vx, cy + hw * hy + hh * vy],
    [cx - hw * hx + hh * vx, cy - hw * hy + hh * vy],
  ] as ImportedObbCorners;
}

function buildImportedObbFromSemanticAxis(
  centerX: number,
  centerY: number,
  width: number,
  height: number,
  semanticAxisX: number,
  semanticAxisY: number,
  orientationPolicy: NormalizedOrientationPolicy
): { obbCorners: ImportedObbCorners; angle: number } {
  const { hx, hy } = resolveImportedObbAxes(
    semanticAxisX,
    semanticAxisY,
    orientationPolicy
  );
  const snappedAngle = snapDerivedImportedAngle(
    Math.atan2(hy, hx) * 180 / Math.PI,
    orientationPolicy.mode
  );
  const angleRad = snappedAngle * (Math.PI / 180);
  const cos = Math.cos(angleRad);
  const sin = Math.sin(angleRad);
  const snappedHx = cos;
  const snappedHy = sin;
  const snappedVx = -sin;
  const snappedVy = cos;
  return {
    obbCorners: buildImportedObbCornersFromAxes(
      centerX,
      centerY,
      width,
      height,
      snappedHx,
      snappedHy,
      snappedVx,
      snappedVy
    ),
    angle: normalizeImportedAngleSigned(snappedAngle),
  };
}

function validateImportedVerticalAnchorAlignment(
  head: ImportedLandmark | null,
  tail: ImportedLandmark | null,
  obbCorners?: ImportedObbCorners | null
): boolean {
  if (!head || !tail || !obbCorners || obbCorners.length !== 4) return true;
  const [cp0, cp1, cp2, cp3] = obbCorners;
  const topMidX = (cp0[0] + cp1[0]) / 2;
  const topMidY = (cp0[1] + cp1[1]) / 2;
  const bottomMidX = (cp2[0] + cp3[0]) / 2;
  const bottomMidY = (cp2[1] + cp3[1]) / 2;
  const headToTop = Math.hypot(Number(head.x) - topMidX, Number(head.y) - topMidY);
  const headToBottom = Math.hypot(Number(head.x) - bottomMidX, Number(head.y) - bottomMidY);
  const tailToBottom = Math.hypot(Number(tail.x) - bottomMidX, Number(tail.y) - bottomMidY);
  const tailToTop = Math.hypot(Number(tail.x) - topMidX, Number(tail.y) - topMidY);
  return headToTop <= headToBottom && tailToBottom <= tailToTop;
}

function maybeFlipImportedObbForVerticalCanonical(
  head: ImportedLandmark | null,
  tail: ImportedLandmark | null,
  centerX: number,
  centerY: number,
  width: number,
  height: number,
  semanticAxisX: number,
  semanticAxisY: number,
  orientationPolicy: NormalizedOrientationPolicy
): { obbCorners: ImportedObbCorners; angle: number } {
  const built = buildImportedObbFromSemanticAxis(
    centerX,
    centerY,
    width,
    height,
    semanticAxisX,
    semanticAxisY,
    orientationPolicy
  );
  if (
    usesImportedVerticalSemanticAxis(orientationPolicy) &&
    !validateImportedVerticalAnchorAlignment(head, tail, built.obbCorners)
  ) {
    const flipped = buildImportedObbFromSemanticAxis(
      centerX,
      centerY,
      width,
      height,
      -semanticAxisX,
      -semanticAxisY,
      orientationPolicy
    );
    return { obbCorners: flipped.obbCorners, angle: flipped.angle };
  }
  return { obbCorners: built.obbCorners, angle: built.angle };
}

function deriveImportedBilateralVerticalClassId(
  head: ImportedLandmark,
  tail: ImportedLandmark,
  obbCorners?: ImportedObbCorners | null
): number {
  if (!obbCorners || obbCorners.length !== 4) {
    return Number(head.y) < Number(tail.y) ? 0 : 1;
  }
  const [cp0, cp1, cp2, cp3] = obbCorners;
  const centerX = (cp0[0] + cp1[0] + cp2[0] + cp3[0]) / 4;
  const centerY = (cp0[1] + cp1[1] + cp2[1] + cp3[1]) / 4;
  const topMidX = (cp0[0] + cp1[0]) / 2;
  const topMidY = (cp0[1] + cp1[1]) / 2;
  const upX = topMidX - centerX;
  const upY = topMidY - centerY;
  const magnitude = Math.hypot(upX, upY);
  if (magnitude <= 1e-6) {
    return Number(head.y) < Number(tail.y) ? 0 : 1;
  }
  const normX = upX / magnitude;
  const normY = upY / magnitude;
  const headProjection = (Number(head.x) - centerX) * normX + (Number(head.y) - centerY) * normY;
  const tailProjection = (Number(tail.x) - centerX) * normX + (Number(tail.y) - centerY) * normY;
  if (Math.abs(headProjection - tailProjection) <= 1e-6) {
    return Number(head.y) < Number(tail.y) ? 0 : 1;
  }
  return headProjection > tailProjection ? 0 : 1;
}

function deriveImportedDirectionalClassId(
  head: ImportedLandmark,
  tail: ImportedLandmark,
  obbCorners?: ImportedObbCorners | null
): number {
  if (!obbCorners || obbCorners.length !== 4) {
    return Number(head.x) < Number(tail.x) ? 0 : 1;
  }
  const [cp0, cp1, cp2, cp3] = obbCorners;
  const centerX = (cp0[0] + cp1[0] + cp2[0] + cp3[0]) / 4;
  const centerY = (cp0[1] + cp1[1] + cp2[1] + cp3[1]) / 4;
  const rightMidX = (cp1[0] + cp2[0]) / 2;
  const rightMidY = (cp1[1] + cp2[1]) / 2;
  const rightX = rightMidX - centerX;
  const rightY = rightMidY - centerY;
  const magnitude = Math.hypot(rightX, rightY);
  if (magnitude <= 1e-6) {
    return Number(head.x) < Number(tail.x) ? 0 : 1;
  }
  const normX = rightX / magnitude;
  const normY = rightY / magnitude;
  const headProjection = (Number(head.x) - centerX) * normX + (Number(head.y) - centerY) * normY;
  const tailProjection = (Number(tail.x) - centerX) * normX + (Number(tail.y) - centerY) * normY;
  if (Math.abs(headProjection - tailProjection) <= 1e-6) {
    return Number(head.x) < Number(tail.x) ? 0 : 1;
  }
  return headProjection < tailProjection ? 0 : 1;
}

function getImportedOrientationLabelForClassId(
  orientationPolicy: NormalizedOrientationPolicy,
  classId: number | undefined | null
): "left" | "right" | "up" | "down" | undefined {
  if (classId !== 0 && classId !== 1) return undefined;
  // Axial and invariant schemas are one-class contracts.  They can render an
  // undirected centerline, but must not persist a directional arrow/hint.
  if (orientationPolicy.mode === "invariant" || orientationPolicy.mode === "axial") {
    return undefined;
  }
  if (
    orientationPolicy.mode === "bilateral" &&
    orientationPolicy.bilateralClassAxis === "vertical_obb"
  ) {
    return classId === 0 ? "up" : "down";
  }
  return classId === 0 ? "left" : "right";
}

function buildImportedOrientationHint(
  orientationPolicy: NormalizedOrientationPolicy,
  classId: number | undefined | null,
  source: "manual_anchors" | "schema_anchors" | "canonical_default" | "imported_class_id"
): ImportedNormalizedBox["orientation_hint"] | undefined {
  const orientation = getImportedOrientationLabelForClassId(orientationPolicy, classId);
  if (!orientation) return undefined;
  return {
    orientation,
    confidence: 1,
    source,
  };
}

function createImportGeometrySummary(): ImportGeometrySummary {
  return {
    sourceObbPreserved: 0,
    sourceObbReorientedToAnchors: 0,
    translatedToFitImage: 0,
    manualAnchorDerived: 0,
    autoDerived: 0,
    fallbackBoxes: 0,
    usedAsymmetricPadding: false,
    xmlLandmarkIdsPreservedAsSchemaIndexed: 0,
    xmlLandmarkIdsShiftedToOneBased: 0,
    xmlLandmarkIdsAmbiguous: 0,
    missingLandmarkIdsMappedFromSchemaOrder: 0,
  };
}

function summarizeImportGeometry(summary: ImportGeometrySummary, unmatchedCount = 0): string[] {
  const warnings: string[] = [];
  if (summary.manualAnchorDerived > 0) {
    warnings.push(`Derived OBBs and orientation from configured anchor centroids for ${summary.manualAnchorDerived} box${summary.manualAnchorDerived === 1 ? "" : "es"}.`);
  } else if (summary.autoDerived > 0) {
    warnings.push(`Derived OBBs automatically for ${summary.autoDerived} box${summary.autoDerived === 1 ? "" : "es"}.`);
    warnings.push(`Used category or bi-centroid fallback for ${summary.autoDerived} box${summary.autoDerived === 1 ? "" : "es"} because configured anchors were unavailable.`);
  }
  if (summary.sourceObbReorientedToAnchors > 0) {
    warnings.push(`Reoriented ${summary.sourceObbReorientedToAnchors} imported source OBB${summary.sourceObbReorientedToAnchors === 1 ? "" : "s"} to match the selected anchor axis.`);
  }
  if (summary.translatedToFitImage > 0) {
    warnings.push(`Translated ${summary.translatedToFitImage} imported OBB${summary.translatedToFitImage === 1 ? "" : "s"} inward while preserving landmark coverage.`);
  }
  if (summary.xmlLandmarkIdsPreservedAsSchemaIndexed > 0) {
    warnings.push(`Preserved ${summary.xmlLandmarkIdsPreservedAsSchemaIndexed} XML landmark id${summary.xmlLandmarkIdsPreservedAsSchemaIndexed === 1 ? "" : "s"} because they already matched the schema's 1-based indexing.`);
  }
  if (summary.xmlLandmarkIdsShiftedToOneBased > 0) {
    warnings.push(`Shifted ${summary.xmlLandmarkIdsShiftedToOneBased} XML landmark id${summary.xmlLandmarkIdsShiftedToOneBased === 1 ? "" : "s"} from 0-based to schema 1-based indexing.`);
  }
  if (summary.xmlLandmarkIdsAmbiguous > 0) {
    warnings.push(`Left ${summary.xmlLandmarkIdsAmbiguous} XML landmark id${summary.xmlLandmarkIdsAmbiguous === 1 ? "" : "s"} unchanged because the source indexing did not safely match either schema 1-based ids or a clean 0-based offset.`);
  }
  if (summary.missingLandmarkIdsMappedFromSchemaOrder > 0) {
    warnings.push(`Mapped ${summary.missingLandmarkIdsMappedFromSchemaOrder} imported landmark id${summary.missingLandmarkIdsMappedFromSchemaOrder === 1 ? "" : "s"} from schema order because the source annotation omitted explicit ids.`);
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
  landmarkTemplate: any[],
  geometryConfig?: ImportGeometryConfig,
  obbCorners?: ImportedObbCorners | null
): number | undefined {
  if (orientationPolicy.mode === "invariant" || orientationPolicy.mode === "axial") return 0;
  const { head, tail } = resolveImportedSemanticAnchors(
    landmarks,
    orientationPolicy,
    landmarkTemplate,
    geometryConfig
  );
  if (!head || !tail) return 0;
  if (
    orientationPolicy.mode === "bilateral" &&
    orientationPolicy.bilateralClassAxis === "vertical_obb"
  ) {
    return deriveImportedBilateralVerticalClassId(head, tail, obbCorners);
  }
  return deriveImportedDirectionalClassId(head, tail, obbCorners);
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
    const obbCorners = fitImportedObbCornersRigidly(
      buildImportedObbCorners(cx, cy, aabb.width, aabb.height, 0),
      imageDims,
      landmarks
    ).corners;
    return {
      ...importedCornersToAabb(obbCorners, imageDims),
      obbCorners,
      angle: 0,
      class_id: deriveImportedClassId(landmarks, orientationPolicy, landmarkTemplate, geometryConfig, obbCorners),
      derivation: "fallback",
    };
  }

  const { head, tail } = resolveImportedSemanticAnchors(
    landmarks,
    orientationPolicy,
    landmarkTemplate,
    geometryConfig
  );
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
    const obbCorners = fitImportedObbCornersRigidly(
      buildImportedObbCorners(cx, cy, aabb.width, aabb.height, 0),
      imageDims,
      landmarks
    ).corners;
    return {
      ...importedCornersToAabb(obbCorners, imageDims),
      obbCorners,
      angle: 0,
      class_id: deriveImportedClassId(landmarks, orientationPolicy, landmarkTemplate, geometryConfig, obbCorners),
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
    const obbCorners = fitImportedObbCornersRigidly(
      buildImportedObbCorners(cx, cy, aabb.width, aabb.height, 0),
      imageDims,
      landmarks
    ).corners;
    return {
      ...importedCornersToAabb(obbCorners, imageDims),
      obbCorners,
      angle: 0,
      class_id: deriveImportedClassId(landmarks, orientationPolicy, landmarkTemplate, geometryConfig, obbCorners),
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

  const { hx, hy, vx, vy } = resolveImportedObbAxes(ux, uy, orientationPolicy);
  const hValues = valid.map((lm) => lm.x * hx + lm.y * hy);
  const vValues = valid.map((lm) => lm.x * vx + lm.y * vy);
  const minH = Math.min(...hValues);
  const maxH = Math.max(...hValues);
  const minV = Math.min(...vValues);
  const maxV = Math.max(...vValues);
  const rawWidth = Math.max(2, maxH - minH);
  const rawHeight = Math.max(2, maxV - minV);
  const padForward = Math.max(4, rawWidth * resolvedGeometry.paddingProfile.forward);
  const padBackward = Math.max(4, rawWidth * resolvedGeometry.paddingProfile.backward);
  const padTop = Math.max(4, rawHeight * resolvedGeometry.paddingProfile.top);
  const padBottom = Math.max(4, rawHeight * resolvedGeometry.paddingProfile.bottom);
  const minHWithPad = minH - padForward;
  const maxHWithPad = maxH + padBackward;
  const minVWithPad = minV - padTop;
  const maxVWithPad = maxV + padBottom;
  const centerH = (minHWithPad + maxHWithPad) / 2;
  const centerV = (minVWithPad + maxVWithPad) / 2;
  const centerX = centerH * hx + centerV * vx;
  const centerY = centerH * hy + centerV * vy;
  const width = maxHWithPad - minHWithPad;
  const height = maxVWithPad - minVWithPad;
  const canonicalObb = maybeFlipImportedObbForVerticalCanonical(
    head,
    tail,
    centerX,
    centerY,
    width,
    height,
    ux,
    uy,
    orientationPolicy
  );
  const obbCorners = fitImportedObbCornersRigidly(canonicalObb.obbCorners, imageDims, landmarks).corners;
  return {
    ...importedCornersToAabb(obbCorners, imageDims),
    obbCorners,
    angle: canonicalObb.angle,
    class_id: deriveImportedClassId(landmarks, orientationPolicy, landmarkTemplate, geometryConfig, obbCorners),
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
  const resolvedGeometry = normalizeImportGeometryConfig(geometryConfig);
  const semanticAnchors = resolveImportedSemanticAnchors(
    normalizedLandmarks,
    orientationPolicy,
    landmarkTemplate,
    geometryConfig
  );
  const sourceObbCorners = normalizeImportedObbCorners(rawBox?.obbCorners, imageDims);
  if (sourceObbCorners) {
    if (summary) {
      summary.sourceObbPreserved += 1;
    }
    let resolvedObbCorners = sourceObbCorners;
    let angle = isFiniteNumber(rawBox?.angle)
      ? Number(rawBox?.angle)
      : Math.atan2(
          Number(sourceObbCorners[1][1]) - Number(sourceObbCorners[0][1]),
          Number(sourceObbCorners[1][0]) - Number(sourceObbCorners[0][0])
        ) * 180 / Math.PI;
    if (semanticAnchors.head && semanticAnchors.tail) {
      const head = semanticAnchors.head;
      const tail = semanticAnchors.tail;
      if (semanticAnchors.source !== "head_tail" || resolvedGeometry.axisMode === "manual_anchors") {
        if (head && tail) {
          const semanticAxisX = Number(tail.x) - Number(head.x);
          const semanticAxisY = Number(tail.y) - Number(head.y);
          // Center from corner average — corner-order independent
          const srcCX = (Number(sourceObbCorners[0][0]) + Number(sourceObbCorners[1][0]) + Number(sourceObbCorners[2][0]) + Number(sourceObbCorners[3][0])) / 4;
          const srcCY = (Number(sourceObbCorners[0][1]) + Number(sourceObbCorners[1][1]) + Number(sourceObbCorners[2][1]) + Number(sourceObbCorners[3][1])) / 4;
          // Compute edge vectors and lengths (consecutive corners always form perpendicular edges)
          const e01x = Number(sourceObbCorners[1][0]) - Number(sourceObbCorners[0][0]);
          const e01y = Number(sourceObbCorners[1][1]) - Number(sourceObbCorners[0][1]);
          const e12x = Number(sourceObbCorners[2][0]) - Number(sourceObbCorners[1][0]);
          const e12y = Number(sourceObbCorners[2][1]) - Number(sourceObbCorners[1][1]);
          const edge01Len = Math.max(2, Math.hypot(e01x, e01y));
          const edge12Len = Math.max(2, Math.hypot(e12x, e12y));
          // Determine which edge is more aligned with the biological (semantic) axis via dot product
          const semanticLen = Math.hypot(semanticAxisX, semanticAxisY);
          const ux = semanticLen > 0 ? semanticAxisX / semanticLen : 0;
          const uy = semanticLen > 0 ? semanticAxisY / semanticLen : 1;
          const dot01 = Math.abs(e01x * ux + e01y * uy) / edge01Len;
          // The edge aligned with the declared semantic anchor axis is height; the other is width.
          const axisHeight = dot01 > 0.5 ? edge01Len : edge12Len;
          const axisWidth  = dot01 > 0.5 ? edge12Len : edge01Len;
          const rebuilt = maybeFlipImportedObbForVerticalCanonical(
            head,
            tail,
            srcCX,
            srcCY,
            axisWidth,
            axisHeight,
            semanticAxisX,
            semanticAxisY,
            orientationPolicy
          );
          resolvedObbCorners = fitImportedObbCornersRigidly(rebuilt.obbCorners, imageDims, normalizedLandmarks).corners;
          angle = rebuilt.angle;
          if (summary) {
            summary.sourceObbReorientedToAnchors += 1;
          }
          warnings.push(`${context}: reoriented source OBB to match ${semanticAnchors.source === "manual_anchors" ? "manual" : "schema"} anchors.`);
        }
      }
    }
    const fittedSourceObb = fitImportedObbCornersRigidly(resolvedObbCorners, imageDims, normalizedLandmarks);
    resolvedObbCorners = fittedSourceObb.corners;
    if (summary) {
      if (fittedSourceObb.translated) summary.translatedToFitImage += 1;
    }
    const aabb = importedCornersToAabb(resolvedObbCorners, imageDims);
    const derivedClassId = deriveImportedClassId(
      normalizedLandmarks,
      orientationPolicy,
      landmarkTemplate,
      geometryConfig,
      resolvedObbCorners
    );
    const classId = derivedClassId;
    const orientationHint = buildImportedOrientationHint(
      orientationPolicy,
      classId,
      semanticAnchors.source === "manual_anchors"
        ? "manual_anchors"
        : semanticAnchors.source === "schema_anchors"
        ? "schema_anchors"
        : "canonical_default"
    );
    return {
      ...aabb,
      landmarks: normalizedLandmarks,
      obbCorners: resolvedObbCorners,
      angle,
      geometryOrigin: "source",
      ...(classId != null ? { class_id: classId } : {}),
      ...(orientationHint ? { orientation_hint: orientationHint } : {}),
    };
  }

  const derived = deriveImportedObbFromLandmarks(
    normalizedLandmarks,
    imageDims,
    orientationPolicy,
    landmarkTemplate,
    geometryConfig
  );
  if (summary) {
    if (resolvedGeometry.paddingMode === "asymmetric") {
      summary.usedAsymmetricPadding = true;
    }
    const fittedDerivedObb = fitImportedObbCornersRigidly(derived.obbCorners, imageDims, normalizedLandmarks);
    derived.obbCorners = fittedDerivedObb.corners;
    if (fittedDerivedObb.translated) summary.translatedToFitImage += 1;
    if (derived.derivation === "fallback") {
      summary.fallbackBoxes += 1;
    } else if (semanticAnchors.source === "manual_anchors" || semanticAnchors.source === "schema_anchors") {
      summary.manualAnchorDerived += 1;
    } else {
      summary.autoDerived += 1;
    }
    const fittedAabb = importedCornersToAabb(derived.obbCorners, imageDims);
    derived.left = fittedAabb.left;
    derived.top = fittedAabb.top;
    derived.width = fittedAabb.width;
    derived.height = fittedAabb.height;
  } else {
    const fittedDerivedObb = fitImportedObbCornersRigidly(derived.obbCorners, imageDims, normalizedLandmarks);
    derived.obbCorners = fittedDerivedObb.corners;
    const fittedAabb = importedCornersToAabb(derived.obbCorners, imageDims);
    derived.left = fittedAabb.left;
    derived.top = fittedAabb.top;
    derived.width = fittedAabb.width;
    derived.height = fittedAabb.height;
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
    ...(buildImportedOrientationHint(
      orientationPolicy,
      derived.class_id,
      resolvedGeometry.axisMode === "manual_anchors" ? "manual_anchors" : "canonical_default"
    )
      ? {
          orientation_hint: buildImportedOrientationHint(
            orientationPolicy,
            derived.class_id,
            semanticAnchors.source === "manual_anchors"
              ? "manual_anchors"
              : semanticAnchors.source === "schema_anchors"
              ? "schema_anchors"
              : "canonical_default"
          ),
        }
      : {}),
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

function classifyXmlLandmarkIdStrategy(
  observedIds: number[],
  landmarkTemplate?: Array<{ index?: number }>
): "preserve" | "shift_to_one_based" | "ambiguous" {
  const templateIds = new Set(
    (Array.isArray(landmarkTemplate) ? landmarkTemplate : [])
      .map((landmark) => Number(landmark?.index))
      .filter((value) => Number.isFinite(value))
  );
  if (observedIds.length === 0 || templateIds.size === 0) {
    return "preserve";
  }
  const uniqueObserved = [...new Set(observedIds.filter((value) => Number.isFinite(value)).map((value) => Math.round(value)))];
  if (uniqueObserved.length === 0) {
    return "preserve";
  }
  const rawMatchesSchema = uniqueObserved.every((value) => templateIds.has(value));
  const shiftedMatchesSchema = uniqueObserved.every((value) => templateIds.has(value + 1));
  if (uniqueObserved.some((value) => value === 0) && shiftedMatchesSchema) {
    return "shift_to_one_based";
  }
  if (rawMatchesSchema) {
    return "preserve";
  }
  if (shiftedMatchesSchema) {
    return "shift_to_one_based";
  }
  return "ambiguous";
}

/**
 * Parse an imglab-format dlib XML annotation file.
 * Returns a Map keyed by image basename Ã¢â€ â€™ { box, landmarks[] }.
 */
function parseImglabXml(
  filePath: string,
  landmarkTemplate?: Array<{ index?: number }>,
  summary?: Pick<
    ImportGeometrySummary,
    "xmlLandmarkIdsPreservedAsSchemaIndexed" | "xmlLandmarkIdsShiftedToOneBased" | "xmlLandmarkIdsAmbiguous"
  >
): Map<string, AnnotationEntry> {
  const content = fs.readFileSync(filePath, "utf-8");
  const result = new Map<string, AnnotationEntry>();
  const observedXmlIds: number[] = [];

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
          const parsedId = parseInt(pa["name"], 10);
          if (Number.isFinite(parsedId)) {
            observedXmlIds.push(parsedId);
          }
          landmarks.push({
            id: parsedId,
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

  const xmlIdStrategy = classifyXmlLandmarkIdStrategy(observedXmlIds, landmarkTemplate);
  if (xmlIdStrategy !== "preserve" || summary) {
    result.forEach((entry) => {
      entry.boxes.forEach((box) => {
        if (!Array.isArray(box.landmarks)) return;
        box.landmarks = box.landmarks.map((landmark) => {
          if (!Number.isFinite(Number(landmark?.id))) return landmark;
          const id = Number(landmark.id);
          const normalizedId = xmlIdStrategy === "shift_to_one_based" ? id + 1 : id;
          if (summary) {
            if (xmlIdStrategy === "shift_to_one_based") {
              summary.xmlLandmarkIdsShiftedToOneBased += 1;
            } else if (xmlIdStrategy === "ambiguous") {
              summary.xmlLandmarkIdsAmbiguous += 1;
            } else {
              summary.xmlLandmarkIdsPreservedAsSchemaIndexed += 1;
            }
          }
          return { ...landmark, id: normalizedId };
        });
      });
    });
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
  blockingErrors: string[];
} {
  const warnings: string[] = [];
  const blockingErrors: string[] = [];
  const labelsDir = path.join(effectiveRoot, "labels");
  if (!fs.existsSync(labelsDir)) {
    return {
      labelFiles: 0,
      trainableImages: 0,
      landmarkStatus: "warning",
      landmarkMessage: "No labels found",
      warnings: ["labels directory does not exist"],
      blockingErrors: ["labels directory does not exist"],
    };
  }

  const jsonPaths = fs
    .readdirSync(labelsDir)
    .filter((name) => name.toLowerCase().endsWith(".json"))
    .map((name) => path.join(labelsDir, name));

  const sessionMeta = safeReadJson(path.join(effectiveRoot, "session.json")) || {};
  const requiredIds = new Set(
    normalizeLandmarkTemplate(sessionMeta.landmarkTemplate)
      .filter((entry: any) => entry?.required !== false && entry?.optional !== true)
      .map((entry) => Number(entry.index))
      .filter(Number.isFinite)
  );
  const idSets: Array<Set<number>> = [];
  let trainableImages = 0;

  for (const labelPath of jsonPaths) {
    try {
      const parsed = JSON.parse(fs.readFileSync(labelPath, "utf-8"));
      const boxes = Array.isArray(parsed?.boxes) ? parsed.boxes : [];
      let imageHasTrainableSpecimen = false;

      for (let boxIndex = 0; boxIndex < boxes.length; boxIndex += 1) {
        const box = boxes[boxIndex];
        const ids = new Set<number>();
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
        if (ids.size > 0) {
          idSets.push(ids);
          imageHasTrainableSpecimen = true;
        }
        const missing = [...requiredIds].filter((id) => !ids.has(id)).sort((a, b) => a - b);
        if (landmarks.length > 0 && missing.length > 0) {
          blockingErrors.push(
            `${path.basename(labelPath)} box ${boxIndex} is missing required landmark IDs: ${missing.join(", ")}`
          );
        }
      }
      if (imageHasTrainableSpecimen) trainableImages += 1;
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
      blockingErrors: blockingErrors.length > 0 ? blockingErrors : ["No valid landmark sets"],
    };
  }

  let landmarkStatus: "ok" | "warning" = "ok";
  let landmarkMessage = "consistent";
  if (blockingErrors.length > 0) {
    landmarkStatus = "warning";
    landmarkMessage = `${blockingErrors.length} specimen(s) are missing required schema landmarks`;
  } else if (idSets.length > 1) {
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
    blockingErrors,
  };
}

function readExplicitSessionOrientationIssue(effectiveRoot: string): string | null {
  const sessionPath = path.join(effectiveRoot, "session.json");
  if (!fs.existsSync(sessionPath)) {
    return "Session metadata is missing. Create or select a session before training.";
  }
  const meta = safeReadJson(sessionPath);
  if (!hasConfiguredOrientationPolicy(meta)) {
    return "Choose and save an explicit session orientation policy before training.";
  }
  return null;
}

function persistExplicitSessionTrainingContract(effectiveRoot: string): {
  landmarkTemplate: ReturnType<typeof normalizeLandmarkTemplate>;
  orientationPolicy: Record<string, unknown>;
  orientationMode: OrientationMode;
  schemaSemanticFingerprint: string;
} {
  const issue = readExplicitSessionOrientationIssue(effectiveRoot);
  if (issue) throw new Error(issue);
  const sessionPath = path.join(effectiveRoot, "session.json");
  const meta = safeReadJson(sessionPath);
  if (!meta || typeof meta !== "object") {
    throw new Error("Session metadata is unreadable. Repair session.json before training.");
  }
  const landmarkTemplate = normalizeLandmarkTemplate((meta as any).landmarkTemplate);
  if (landmarkTemplate.length === 0) {
    throw new Error("Choose a non-empty landmark schema before training.");
  }
  const rawPolicy = (meta as any).orientationPolicy;
  const schemaSemanticFingerprint = computeSessionSchemaSemanticFingerprint(
    landmarkTemplate,
    rawPolicy
  );
  if (!schemaSemanticFingerprint) {
    throw new Error(
      "The saved orientation policy cannot be fingerprinted. Re-save an explicit session policy before training."
    );
  }
  const orientationMode = normalizeOrientationMode(rawPolicy?.mode);
  (meta as any).landmarkTemplate = landmarkTemplate;
  // Preserve the user's explicit raw policy. Inferred compatibility defaults
  // must never be persisted as though the user selected them.
  (meta as any).schemaSemanticFingerprint = schemaSemanticFingerprint;
  (meta as any).schemaSemanticVersion = SCHEMA_SEMANTIC_VERSION;
  atomicWriteJsonSync(sessionPath, meta);
  return {
    landmarkTemplate,
    orientationPolicy: rawPolicy as Record<string, unknown>,
    orientationMode,
    schemaSemanticFingerprint,
  };
}

type ImportedDlibPreparationResult = {
  ok: boolean;
  error?: string;
  code?: string;
  requiresMappingConfirmation?: boolean;
  mappingProposal?: Array<{
    dlibSlot: number;
    schemaId: number;
    schemaName: string;
    required: boolean;
  }>;
  mappingMode?: string;
  mappingPath?: string;
  splitInfoPath?: string;
  cohortManifestPath?: string;
  validationCohortRevision?: string;
  testCohortRevision?: string | null;
  trainImages?: number;
  validationImages?: number;
  testImages?: number;
  landmarkIds?: number[];
};

async function runImportedDlibPreparation(args: {
  effectiveRoot: string;
  modelName: string;
  mode: "prepare" | "verify";
  validationMode?: "derive" | "explicit";
  testMode?: "none" | "derive" | "explicit";
  confirmTemplateOrder?: boolean;
  validationFraction?: number;
  seed?: number;
}): Promise<ImportedDlibPreparationResult> {
  persistExplicitSessionTrainingContract(args.effectiveRoot);
  const scriptArgs = [
    args.effectiveRoot,
    args.modelName,
    "--mode",
    args.mode,
    "--validation-mode",
    args.validationMode ?? "derive",
    "--test-mode",
    args.testMode ?? "derive",
    "--validation-fraction",
    String(args.validationFraction ?? 0.2),
    "--seed",
    String(args.seed ?? 42),
  ];
  if (args.confirmTemplateOrder) scriptArgs.push("--confirm-template-order");
  const output = await runBundledScript("prepare_imported_dlib_dataset", scriptArgs);
  const lines = String(output || "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  const parsed = JSON.parse(lines[lines.length - 1] || "{}") as ImportedDlibPreparationResult;
  if (!parsed || typeof parsed !== "object" || typeof parsed.ok !== "boolean") {
    throw new Error("Imported dlib preparation returned an invalid response.");
  }
  return parsed;
}

interface ImportedXmlGateResult {
  ok: boolean;
  error?: string;
  contract?: ImportedDlibPreparationResult;
  trainValidation?: any;
  validationValidation?: any;
  testValidation?: any;
}

/**
 * The single gate every imported-XML training entry point must pass.
 *
 * Preflight and train both need identical checks; keeping two copies let them
 * drift (train never verified that validation_<tag>.xml existed, so a missing
 * frozen promotion cohort surfaced as a Python traceback instead of an
 * actionable message).
 */
async function verifyImportedXmlContract(args: {
  effectiveRoot: string;
  modelName: string;
  validationFraction?: number;
  seed?: number;
}): Promise<ImportedXmlGateResult> {
  const { effectiveRoot, modelName } = args;
  const xmlPath = (cohort: string) =>
    path.join(effectiveRoot, "xml", `${cohort}_${modelName}.xml`);

  const trainXml = xmlPath("train");
  if (!fs.existsSync(trainXml)) {
    return { ok: false, error: `train_${modelName}.xml not found.` };
  }
  const trainValidation = JSON.parse(await runBundledScript("validate_dlib_xml", [trainXml]));
  if (!trainValidation.ok) {
    return {
      ok: false,
      error: `Train XML validation failed: ${summarizeValidationErrors(trainValidation)}`,
    };
  }

  const testXml = xmlPath("test");
  let testValidation: any = null;
  if (fs.existsSync(testXml)) {
    testValidation = JSON.parse(await runBundledScript("validate_dlib_xml", [testXml]));
    if (!testValidation.ok) {
      return {
        ok: false,
        error: `Test XML validation failed: ${summarizeValidationErrors(testValidation)}`,
      };
    }
  }

  // The frozen validation cohort is what makes promotion decisions comparable,
  // so it is required rather than optional.
  const validationXml = xmlPath("validation");
  if (!fs.existsSync(validationXml)) {
    return {
      ok: false,
      error:
        `validation_${modelName}.xml not found. Re-import the dlib XML so BioVision can ` +
        "create or validate a frozen promotion cohort.",
    };
  }
  const validationValidation = JSON.parse(await runBundledScript("validate_dlib_xml", [validationXml]));
  if (!validationValidation.ok) {
    return {
      ok: false,
      error: `Validation XML validation failed: ${summarizeValidationErrors(validationValidation)}`,
    };
  }

  const contract = await runImportedDlibPreparation({
    effectiveRoot,
    modelName,
    mode: "verify",
    validationFraction: args.validationFraction,
    seed: args.seed,
  });
  if (!contract.ok) {
    return {
      ok: false,
      error:
        contract.error ||
        "Imported XML mapping or frozen validation contract is invalid. Re-import the XML.",
    };
  }

  return { ok: true, contract, trainValidation, validationValidation, testValidation };
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
    const out = await runBundledScript("list_cnn_variants");
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
      const orientationIssue = readExplicitSessionOrientationIssue(effectiveRoot);
      if (orientationIssue) {
        return { ok: false, error: orientationIssue };
      }
      const obbGateError = landmarkTrainingObbGateError(
        resolveObbDetectorForEffectiveRoot(effectiveRoot)
      );
      if (obbGateError) {
        return { ok: false, error: obbGateError, obbDetectorReady: false };
      }

      if (useImportedXml) {
        const gate = await verifyImportedXmlContract({ effectiveRoot, modelName });
        if (!gate.ok) {
          return { ok: false, error: gate.error };
        }
        const importedContract = gate.contract!;

        return {
          ok: true,
          useImportedXml: true,
          trainXmlImages: importedContract.trainImages ?? gate.trainValidation.num_images,
          validationXmlImages:
            importedContract.validationImages ?? gate.validationValidation.num_images,
          testXmlImages:
            importedContract.testImages ??
            (gate.testValidation ? gate.testValidation.num_images : 0),
          landmarkStatus: "ok",
          landmarkMessage: "validated with explicit schema mapping and frozen validation cohort",
          warnings: [],
        };
      }

      const summary = summarizeLabelDataset(effectiveRoot);
      if (summary.blockingErrors.length > 0) {
        return {
          ok: false,
          error:
            `Training data does not satisfy the fixed landmark schema. ${summary.blockingErrors[0]}` +
            (summary.blockingErrors.length > 1
              ? ` (+${summary.blockingErrors.length - 1} more)`
              : ""),
          blockingErrors: summary.blockingErrors,
        };
      }
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
    const orientationIssue = readExplicitSessionOrientationIssue(effectiveRoot);
    if (orientationIssue) throw new Error(orientationIssue);
    const obbGateError = landmarkTrainingObbGateError(
      resolveObbDetectorForEffectiveRoot(effectiveRoot)
    );
    if (obbGateError) throw new Error(obbGateError);
    emitTrainProgress(3, "preflight", "Validating training inputs...");

    if (options?.useImportedXml) {
      const gate = await verifyImportedXmlContract({
        effectiveRoot,
        modelName,
        validationFraction: testSplit,
        seed,
      });
      if (!gate.ok) {
        throw new Error(gate.error);
      }
      const importedContract = gate.contract!;
      emitTrainProgress(
        20,
        "preflight",
        `Imported XML contract verified (${importedContract.trainImages ?? 0} train, ` +
        `${importedContract.validationImages ?? 0} validation).`
      );
    } else {
      // Prepare dataset with train/test split
      emitTrainProgress(12, "prepare_dataset", "Preparing dataset...");
      await runBundledScriptWithProgress(
        "prepare_dataset",
        [effectiveRoot, modelName, testSplit.toString(), seed.toString()],
        (pct, stage, details) => {
          const scaled = 12 + Math.round((Math.max(0, Math.min(100, pct)) / 100) * 18);
          const uiStage = resolveProgressStage("prepare_dataset", details);
          const uiMessage = resolveProgressMessage(stage, details);
          emitTrainProgress(scaled, uiStage, uiMessage, details);
        }
      );
      emitTrainProgress(30, "prepare_dataset", "Dataset preparation complete.");
    }

    // Run dataset audit (non-blocking: surface warnings but don't abort)
    let auditReport: Record<string, unknown> | null = null;
    emitTrainProgress(35, "evaluation", "Auditing dataset...");
    try {
      const auditOut = await runBundledScript("audit_dataset", [
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
        effectiveRoot,
        modelName,
        "--model-variant", cnnVariant,
        "--seed", String(seed),
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
      out = await runBundledScriptWithProgress("train_cnn_model", cnnArgs, (pct, stage, details) => {
        const scaled = 42 + Math.round((Math.max(0, Math.min(100, pct)) / 100) * 50);
        const uiStage = resolveProgressStage("training", details);
        const uiMessage = resolveProgressMessage(stage, details);
        emitTrainProgress(scaled, uiStage, uiMessage, details);
      });
    } else {
      const trainArgs = [
        effectiveRoot,
        modelName,
      ];
      const requestedDlibOptions = options?.customOptions || {};
      trainArgs.push(JSON.stringify({
        ...requestedDlibOptions,
        random_seed: requestedDlibOptions.random_seed ?? String(seed),
        augmentation_seed: requestedDlibOptions.augmentation_seed ?? seed,
      }));
      emitTrainProgress(42, "training", "Training dlib shape predictor...");
      out = await runBundledScriptWithProgress("train_shape_model", trainArgs, (pct, stage, details) => {
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
    const publication = parseLandmarkTrainingPublication(out);

    syncLandmarkModelRegistry(
      effectiveRoot,
      publication.modelId
        ? undefined
        : {
            setActive: {
              name: modelName,
              predictorType,
              ...(publication.artifactTag
                ? { artifactTag: publication.artifactTag }
                : {}),
            },
          }
    );

    const candidateRetained =
      publication.modelStatus === "candidate" ||
      publication.promotion?.promoted === false;
    emitTrainProgress(
      100,
      candidateRetained ? "candidate" : "done",
      candidateRetained
        ? "Candidate retained; active landmark model unchanged."
        : "Training complete."
    );

    return {
      ok: true,
      output: out,
      trainError:       trainErrorMatch       ? parseFloat(trainErrorMatch[1])       : null,
      testError:        testErrorMatch        ? parseFloat(testErrorMatch[1])        : null,
      trainMedianError: trainMedianErrorMatch ? parseFloat(trainMedianErrorMatch[1]) : null,
      testMedianError:  testMedianErrorMatch  ? parseFloat(testMedianErrorMatch[1])  : null,
      modelPath:        modelPathMatch        ? modelPathMatch[1].trim()             : null,
      modelId:          publication.modelId ?? null,
      artifactTag:      publication.artifactTag ?? null,
      modelStatus:      publication.modelStatus ?? null,
      promotion:        publication.promotion ?? null,
      auditReport:      auditReport ?? undefined,
    };
  } catch (e: any) {
    emitTrainProgress(100, "error", `Training failed: ${e.message}`);
    console.error("Training failed:", e);
    return { ok: false, error: e.message };
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

    const effectiveRoot = getEffectiveRoot(args.speciesId);
    ensureTrainingLayout(effectiveRoot);
    persistExplicitSessionTrainingContract(effectiveRoot);

    const picker = await dialog.showOpenDialog({
      properties: ["openFile", "multiSelections"],
      filters: [{ name: "XML", extensions: ["xml"] }],
      title: "Select dlib XML files (train, optional validation, optional test)",
    });
    if (picker.canceled || picker.filePaths.length === 0) {
      return { ok: false, canceled: true, warnings: [] };
    }

    const xmlFiles = picker.filePaths.filter((file) => file.toLowerCase().endsWith(".xml"));
    if (xmlFiles.length === 0) {
      return { ok: false, canceled: false, warnings: [], error: "No XML files selected." };
    }

    const warnings: string[] = [];
    const basename = (file: string) => path.basename(file).toLowerCase();
    const namedTrain = xmlFiles.filter((file) => /train/.test(basename(file)));
    const namedValidation = xmlFiles.filter((file) =>
      /validation|(^|[_.-])val([_.-]|$)/.test(basename(file))
    );
    const namedTest = xmlFiles.filter((file) => /test/.test(basename(file)));
    if (namedTrain.length > 1 || namedValidation.length > 1 || namedTest.length > 1) {
      return {
        ok: false,
        canceled: false,
        warnings,
        error:
          "Selected XML filenames assign more than one file to the same cohort. " +
          "Use one filename containing train, one containing validation/val, and one containing test.",
      };
    }
    const roleFiles = new Set([...namedTrain, ...namedValidation, ...namedTest]);
    const unclassified = xmlFiles.filter((file) => !roleFiles.has(file));
    const trainXml = namedTrain[0] ?? unclassified.shift();
    if (!trainXml) {
      return {
        ok: false,
        canceled: false,
        warnings,
        error: "No train XML could be identified. Include 'train' in its filename.",
      };
    }
    if (!namedTrain[0]) {
      warnings.push(`Train XML inferred from first unclassified file: ${path.basename(trainXml)}`);
    }
    const validationXml: string | undefined = namedValidation[0];
    let testXml: string | undefined = namedTest[0];
    if (unclassified.length === 1 && !testXml) {
      testXml = unclassified.shift();
      warnings.push(`Test XML inferred from unclassified file: ${path.basename(testXml!)}`);
    }
    if (unclassified.length > 0) {
      return {
        ok: false,
        canceled: false,
        warnings,
        error:
          "Some selected XML files have ambiguous cohort roles. Rename them to contain train, " +
          "validation (or val), or test and import again.",
      };
    }

    const xmlDir = path.join(effectiveRoot, "xml");
    const trainDest = path.join(xmlDir, `train_${modelName}.xml`);
    const validationDest = path.join(xmlDir, `validation_${modelName}.xml`);
    const testDest = path.join(xmlDir, `test_${modelName}.xml`);
    const mappingDest = path.join(effectiveRoot, "debug", `id_mapping_${modelName}.json`);
    const splitDest = path.join(effectiveRoot, "debug", `split_info_${modelName}.json`);
    const cohortDest = path.join(
      effectiveRoot,
      "debug",
      "cohorts",
      `imported_dlib_${modelName}.json`
    );
    const cropDest = path.join(
      effectiveRoot,
      "corrected_images",
      `imported_dlib_${modelName}`
    );
    const mutableTargets = [
      trainDest,
      validationDest,
      testDest,
      mappingDest,
      splitDest,
      cohortDest,
    ];
    const snapshots = new Map<string, Buffer | null>(
      mutableTargets.map((target) => [
        target,
        fs.existsSync(target) ? fs.readFileSync(target) : null,
      ])
    );
    // Canonicalized crops are content-addressed, so restoring the six metadata
    // files is not enough: crops written by a prepare that later fails would be
    // left behind referencing nothing.  Record what was already there and drop
    // only the additions.
    const cropsBefore = new Set<string>(
      fs.existsSync(cropDest) ? fs.readdirSync(cropDest) : []
    );
    const rollback = () => {
      snapshots.forEach((previous, target) => {
        if (previous) {
          fs.mkdirSync(path.dirname(target), { recursive: true });
          fs.writeFileSync(target, previous);
        } else if (fs.existsSync(target)) {
          fs.unlinkSync(target);
        }
      });
      if (!fs.existsSync(cropDest)) return;
      for (const name of fs.readdirSync(cropDest)) {
        if (cropsBefore.has(name)) continue;
        try {
          fs.rmSync(path.join(cropDest, name), { recursive: true, force: true });
        } catch (cleanupError) {
          console.warn("Could not remove orphaned imported crop:", name, cleanupError);
        }
      }
    };

    try {
      // Validate every source before replacing any session artifact.
      const sourceValidations = await Promise.all(
        [
          { cohort: "Train", source: trainXml },
          ...(validationXml ? [{ cohort: "Validation", source: validationXml }] : []),
          ...(testXml ? [{ cohort: "Test", source: testXml }] : []),
        ].map(async ({ cohort, source }) => ({
          cohort,
          source,
          result: JSON.parse(await runBundledScript("validate_dlib_xml", [source])),
        }))
      );
      const invalidSource = sourceValidations.find(({ result }) => !result.ok);
      if (invalidSource) {
        return {
          ok: false,
          canceled: false,
          warnings,
          error:
            `${invalidSource.cohort} XML validation failed: ` +
            summarizeValidationErrors(invalidSource.result),
        };
      }

      const trainValidation = JSON.parse(
        await runBundledScript("validate_dlib_xml", [trainXml, trainDest])
      );
      if (!trainValidation.ok) {
        throw new Error(`Train XML validation failed: ${summarizeValidationErrors(trainValidation)}`);
      }
      if (validationXml) {
        const result = JSON.parse(
          await runBundledScript("validate_dlib_xml", [validationXml, validationDest])
        );
        if (!result.ok) {
          throw new Error(`Validation XML validation failed: ${summarizeValidationErrors(result)}`);
        }
      }
      if (testXml) {
        const result = JSON.parse(await runBundledScript("validate_dlib_xml", [testXml, testDest]));
        if (!result.ok) {
          throw new Error(`Test XML validation failed: ${summarizeValidationErrors(result)}`);
        }
      }

      let prepared = await runImportedDlibPreparation({
        effectiveRoot,
        modelName,
        mode: "prepare",
        validationMode: validationXml ? "explicit" : "derive",
        testMode: testXml ? "explicit" : "derive",
      });
      if (prepared.requiresMappingConfirmation && prepared.mappingProposal?.length) {
        const mappingLines = prepared.mappingProposal
          .slice(0, 20)
          .map(
            (entry) =>
              `slot ${entry.dlibSlot} → schema ${entry.schemaId}: ${entry.schemaName}` +
              (entry.required ? " (required)" : " (optional)")
          );
        if (prepared.mappingProposal.length > mappingLines.length) {
          mappingLines.push(`…and ${prepared.mappingProposal.length - mappingLines.length} more`);
        }
        const confirmation = await dialog.showMessageBox({
          type: "warning",
          title: "Confirm imported landmark mapping",
          message: "Confirm how dlib part slots map to this session's landmark schema.",
          detail:
            `${prepared.error || "The XML does not encode BioVision schema IDs."}\n\n` +
            `${mappingLines.join("\n")}\n\n` +
            "Confirm only if the XML author used exactly this landmark order.",
          buttons: ["Confirm mapping", "Cancel import"],
          defaultId: 1,
          cancelId: 1,
          noLink: true,
        });
        if (confirmation.response !== 0) {
          rollback();
          return { ok: false, canceled: true, warnings };
        }
        prepared = await runImportedDlibPreparation({
          effectiveRoot,
          modelName,
          mode: "prepare",
          validationMode: validationXml ? "explicit" : "derive",
          testMode: testXml ? "explicit" : "derive",
          confirmTemplateOrder: true,
        });
      }
      if (!prepared.ok) {
        throw new Error(
          prepared.error ||
          "Imported XML could not be tied to the active landmark/orientation schema."
        );
      }
      return {
        ok: true,
        canceled: false,
        warnings,
        trainXmlPath: trainDest,
        validationXmlPath: validationDest,
        testXmlPath: (prepared.testImages ?? 0) > 0 ? testDest : undefined,
        mappingPath: prepared.mappingPath,
        splitInfoPath: prepared.splitInfoPath,
        validationCohortRevision: prepared.validationCohortRevision,
        mappingMode: prepared.mappingMode,
        trainStats: {
          num_images: prepared.trainImages ?? 0,
          num_boxes: prepared.trainImages ?? 0,
          num_parts: (prepared.trainImages ?? 0) * (prepared.landmarkIds?.length ?? 0),
        },
        validationStats: {
          num_images: prepared.validationImages ?? 0,
          num_boxes: prepared.validationImages ?? 0,
          num_parts:
            (prepared.validationImages ?? 0) * (prepared.landmarkIds?.length ?? 0),
        },
        testStats: (prepared.testImages ?? 0) > 0
          ? {
              num_images: prepared.testImages ?? 0,
              num_boxes: prepared.testImages ?? 0,
              num_parts: (prepared.testImages ?? 0) * (prepared.landmarkIds?.length ?? 0),
            }
          : null,
      };
    } catch (error) {
      rollback();
      throw error;
    }
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

    const out = await runBundledScript("shape_tester", [
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
      orientation?: "left" | "right" | "up" | "down";
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

    const predictorType = resolveLandmarkPredictorForInference(
      modelRoot,
      args.tag,
      args.options?.predictorType
    );

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

    const predictArgs = [
      modelRoot,
      args.tag,
      effectivePath,
      "--predictor-type",
      predictorType,
    ];
    if (args.options?.multiSpecimen) {
      predictArgs.push("--multi");
    }
    if (obbDetectorPath) {
      predictArgs.push("--yolo-model", obbDetectorPath);
    }
    const boxesForInference =
      Array.isArray(args.options?.boxes) && args.options!.boxes.length > 0 ? args.options!.boxes : undefined;

    if (Array.isArray(boxesForInference) && boxesForInference.length > 0) {
      tempBoxesFile = path.join(
        app.getPath("temp"),
        `bv_boxes_${Date.now()}_${Math.random().toString(16).slice(2)}.json`
      );
      fs.writeFileSync(tempBoxesFile, JSON.stringify(boxesForInference));
      predictArgs.push("--boxes-json", tempBoxesFile);
    }

    const out = await runBundledScriptWithProgress("predict", predictArgs, args.onProgress);
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
    resolveLandmarkPredictorForInference(
      modelRoot,
      args.modelName,
      requestedPredictor
    );
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
      const summary = createImportGeometrySummary();
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
        annotationMap = parseImglabXml(annotationFilePath, importContext.landmarkTemplate, summary);
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
              const normalizedLandmarkMetadata = {
                missingLandmarkIdsMappedFromSchemaOrder: 0,
              };
              const normalizedLandmarks = normalizeLandmarks(rawBox.landmarks ?? [], context, {
                fallbackLandmarkTemplate: importContext.landmarkTemplate,
                metadata: normalizedLandmarkMetadata,
              });
              summary.missingLandmarkIdsMappedFromSchemaOrder +=
                normalizedLandmarkMetadata.missingLandmarkIdsMappedFromSchemaOrder;
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
type LandmarkModelStatus = "active" | "candidate" | "deprecated";

type LandmarkModelRegistryEntry = {
  modelId?: string;
  key: string;
  name: string;
  displayName?: string;
  artifactTag?: string;
  predictorType: "dlib" | "cnn";
  path: string;
  artifact?: { path?: string; sha256?: string };
  immutableArtifact?: boolean;
  legacyPath?: string;
  currentAliasPath?: string;
  configPath?: string;
  config?: { path?: string; sha256?: string };
  sidecars?: Record<string, { path?: string; sha256?: string }>;
  runId?: string;
  runManifestPath?: string;
  metrics?: Record<string, unknown>;
  promotion?: Record<string, unknown>;
  createdAt: string;
  status: LandmarkModelStatus;
};

type LandmarkModelRegistryPayload = {
  version: 1 | 2;
  updatedAt: string;
  models: LandmarkModelRegistryEntry[];
};

function getCnnConfigPath(modelsDir: string, name: string): string {
  return path.join(modelsDir, `cnn_${name}_config.json`);
}

function isImmutableLandmarkRegistryEntry(
  entry: LandmarkModelRegistryEntry | undefined
): boolean {
  if (!entry) return false;
  if (entry.immutableArtifact === true) return true;
  return Boolean(
    entry.modelId &&
    !entry.modelId.startsWith("legacy-") &&
    entry.runManifestPath &&
    entry.runId
  );
}

function getCnnModelCompatibility(
  effectiveRoot: string,
  name: string,
  registryEntry?: LandmarkModelRegistryEntry
): { compatible: boolean; reason?: string } {
  const modelsDir = path.join(effectiveRoot, "models");
  const configPath = registryEntry && isImmutableLandmarkRegistryEntry(registryEntry)
    ? String(registryEntry.configPath || registryEntry.config?.path || "").trim()
    : String(registryEntry?.configPath || getCnnConfigPath(modelsDir, name)).trim();
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

function getLandmarkModelCompatibility(
  effectiveRoot: string,
  name: string,
  predictorType: "dlib" | "cnn",
  registryEntry?: LandmarkModelRegistryEntry
): { compatible: boolean; reason?: string } {
  if (registryEntry && isImmutableLandmarkRegistryEntry(registryEntry)) {
    const integrityIssues = validateLandmarkPromotionArtifact(registryEntry);
    if (integrityIssues.length > 0) {
      return {
        compatible: false,
        reason: `Immutable artifact verification failed: ${integrityIssues
          .map((issue) => issue.message)
          .join(" ")}`,
      };
    }
  }
  return predictorType === "cnn"
    ? getCnnModelCompatibility(effectiveRoot, name, registryEntry)
    : { compatible: true };
}

function getModelRegistryPath(effectiveRoot: string): string {
  return path.join(effectiveRoot, "models", "model_registry.json");
}

function buildLandmarkModelKey(name: string, predictorType: "dlib" | "cnn"): string {
  return `${name}::${predictorType}`;
}

function scanLandmarkModels(effectiveRoot: string): Array<{
  key: string;
  modelId?: string;
  name: string;
  artifactTag?: string;
  path: string;
  predictorType: "dlib" | "cnn";
  createdAt: string;
  size: number;
  status?: LandmarkModelStatus;
  metrics?: Record<string, unknown>;
  promotion?: Record<string, unknown>;
}> {
  const modelsDir = path.join(effectiveRoot, "models");
  if (!fs.existsSync(modelsDir)) {
    fs.mkdirSync(modelsDir, { recursive: true });
    return [];
  }

  const registry = readLandmarkModelRegistry(effectiveRoot);
  const registered: ReturnType<typeof scanLandmarkModels> = [];
  const knownAliasPaths = new Set<string>();
  const knownArtifactTags = new Set<string>();
  const normalizePathKey = (value: string) => path.resolve(value).replace(/\\/g, "/").toLowerCase();

  for (const entry of registry.models) {
    if (!entry || (entry.predictorType !== "dlib" && entry.predictorType !== "cnn")) continue;
    for (const candidate of [entry.legacyPath, entry.currentAliasPath]) {
      if (candidate) knownAliasPaths.add(normalizePathKey(candidate));
    }
    if (entry.artifactTag) knownArtifactTags.add(entry.artifactTag);
    const pathCandidates = isImmutableLandmarkRegistryEntry(entry)
      ? [entry.path]
      : [entry.path, entry.legacyPath, entry.currentAliasPath];
    const usablePath = pathCandidates
      .find((candidate) => candidate && fs.existsSync(candidate));
    if (!usablePath) continue;
    const stats = fs.statSync(usablePath);
    registered.push({
      key: entry.modelId || entry.key,
      ...(entry.modelId ? { modelId: entry.modelId } : {}),
      name: entry.displayName || entry.name,
      ...(entry.artifactTag ? { artifactTag: entry.artifactTag } : {}),
      path: entry.path && fs.existsSync(entry.path) ? entry.path : usablePath,
      predictorType: entry.predictorType,
      createdAt: entry.createdAt || stats.birthtime.toISOString(),
      size: stats.size,
      status: entry.status,
      metrics: entry.metrics,
      promotion: entry.promotion,
    });
  }

  const legacyModels: ReturnType<typeof scanLandmarkModels> = [];
  for (const file of fs.readdirSync(modelsDir)) {
    const predictorType = file.startsWith("predictor_") && file.endsWith(".dat")
      ? "dlib" as const
      : file.startsWith("cnn_") && file.endsWith(".pth")
      ? "cnn" as const
      : null;
    if (!predictorType) continue;
    const filePath = path.join(modelsDir, file);
    if (knownAliasPaths.has(normalizePathKey(filePath))) continue;
    const name = predictorType === "dlib"
      ? file.replace(/^predictor_/, "").replace(/\.dat$/, "")
      : file.replace(/^cnn_/, "").replace(/\.pth$/, "");
    if (knownArtifactTags.has(name)) continue;
    const stats = fs.statSync(filePath);
    legacyModels.push({
      key: buildLandmarkModelKey(name, predictorType),
      name,
      artifactTag: name,
      path: filePath,
      predictorType,
      createdAt: stats.birthtime.toISOString(),
      size: stats.size,
    });
  }

  return [...registered, ...legacyModels];
}

function readSchemaDisplayName(effectiveRoot: string, fallbackSpeciesId?: string): string {
  const sessionJsonPath = path.join(effectiveRoot, "session.json");
  try {
    if (fs.existsSync(sessionJsonPath)) {
      const parsed = JSON.parse(fs.readFileSync(sessionJsonPath, "utf-8"));
      const name = String(parsed?.name || "").trim();
      if (name) return name;
    }
  } catch {
    // ignore and fall back
  }
  return fallbackSpeciesId || path.basename(effectiveRoot);
}

function listSupportedModelsForRoot(
  effectiveRoot: string,
  speciesId?: string
): Array<{
  name: string;
  path: string;
  size: number;
  createdAt: Date;
  predictorType?: "dlib" | "cnn";
  modelKind: "landmark" | "obb_detector";
  speciesId?: string;
  schemaName?: string;
  status?: LandmarkModelStatus;
  metrics?: Record<string, unknown>;
  promotion?: Record<string, unknown>;
  compatible?: boolean;
  reason?: string;
}> {
  const schemaName = readSchemaDisplayName(effectiveRoot, speciesId);
  const registry = syncLandmarkModelRegistry(effectiveRoot);
  const landmarkModels = scanLandmarkModels(effectiveRoot).map((model) => {
    const identifier = model.modelId || model.artifactTag || model.name;
    const registryEntry = findLandmarkRegistryEntry(
      registry,
      identifier,
      model.predictorType
    );
    return {
      ...model,
      createdAt: new Date(model.createdAt),
      modelKind: "landmark" as const,
      speciesId,
      schemaName,
      status: model.status ?? resolveModelStatus(
        registry,
        identifier,
        model.predictorType
      ),
      ...getLandmarkModelCompatibility(
        effectiveRoot,
        model.artifactTag || model.name,
        model.predictorType,
        registryEntry
      ),
    };
  });

  const obbDetectorPath = path.join(effectiveRoot, "models", "session_obb_detector.pt");
  const obbRegistry = safeReadJson(path.join(effectiveRoot, "models", "obb_registry.json"));
  const registeredObbRuns = Array.isArray(obbRegistry?.models)
    ? obbRegistry.models.filter(
        (entry: any) => entry && entry.path && fs.existsSync(String(entry.path))
      )
    : [];
  const obbModels = registeredObbRuns.length > 0
    ? registeredObbRuns.map((entry: any) => ({
        name: String(entry.name || "Session OBB Detector"),
        ...(entry.modelId ? { modelId: String(entry.modelId) } : {}),
        path: String(entry.path),
        size: fs.statSync(String(entry.path)).size,
        createdAt: new Date(entry.createdAt || fs.statSync(String(entry.path)).birthtime),
        modelKind: "obb_detector" as const,
        speciesId,
        schemaName,
        status: (entry.status === "candidate" || entry.status === "deprecated"
          ? entry.status
          : "active") as LandmarkModelStatus,
        metrics: entry.metrics,
        promotion: entry.promotion,
        compatible: true as const,
      }))
    : fs.existsSync(obbDetectorPath)
    ? [{
        name: "Session OBB Detector",
        path: obbDetectorPath,
        size: fs.statSync(obbDetectorPath).size,
        createdAt: fs.statSync(obbDetectorPath).birthtime,
        modelKind: "obb_detector" as const,
        speciesId,
        schemaName,
        status: "active" as LandmarkModelStatus,
        compatible: true as const,
      }]
    : [];

  return [...landmarkModels, ...obbModels];
}

function listGlobalSupportedModels(): Array<{
  name: string;
  path: string;
  size: number;
  createdAt: Date;
  predictorType?: "dlib" | "cnn";
  modelKind: "landmark" | "obb_detector";
  speciesId?: string;
  schemaName?: string;
  status?: LandmarkModelStatus;
  compatible?: boolean;
  reason?: string;
}> {
  const models: Array<{
    name: string;
    path: string;
    size: number;
    createdAt: Date;
    predictorType?: "dlib" | "cnn";
    modelKind: "landmark" | "obb_detector";
    speciesId?: string;
    schemaName?: string;
    status?: LandmarkModelStatus;
    compatible?: boolean;
    reason?: string;
  }> = [];
  const sessionsRoot = getSessionsRoot();
  if (!fs.existsSync(sessionsRoot)) return models;

  for (const entry of fs.readdirSync(sessionsRoot, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const speciesId = entry.name;
    const effectiveRoot = path.join(sessionsRoot, speciesId);
    models.push(...listSupportedModelsForRoot(effectiveRoot, speciesId));
  }
  return models;
}

function readLandmarkModelRegistry(effectiveRoot: string): LandmarkModelRegistryPayload {
  const registryPath = getModelRegistryPath(effectiveRoot);
  if (!fs.existsSync(registryPath)) {
    return { version: 1, updatedAt: new Date().toISOString(), models: [] };
  }
  try {
    const parsed = JSON.parse(fs.readFileSync(registryPath, "utf-8"));
    return {
      version: Number(parsed?.version) === 2 ? 2 : 1,
      updatedAt: String(parsed?.updatedAt || new Date().toISOString()),
      models: Array.isArray(parsed?.models) ? parsed.models : [],
    };
  } catch {
    return { version: 1, updatedAt: new Date().toISOString(), models: [] };
  }
}

function writeLandmarkModelRegistry(effectiveRoot: string, payload: LandmarkModelRegistryPayload): void {
  const registryPath = getModelRegistryPath(effectiveRoot);
  atomicWriteJsonSync(registryPath, payload);
}

function syncLandmarkModelRegistry(
  effectiveRoot: string,
  options?: {
    setActive?: {
      name?: string;
      modelId?: string;
      artifactTag?: string;
      predictorType: "dlib" | "cnn";
    } | null;
  }
): LandmarkModelRegistryPayload {
  const scanned = scanLandmarkModels(effectiveRoot);
  const existing = readLandmarkModelRegistry(effectiveRoot);
  const existingByKey = new Map(existing.models.map((entry) => [entry.key, entry]));

  const nextModels: LandmarkModelRegistryEntry[] = scanned.map((model) => {
    const existingEntry =
      existingByKey.get(model.key) ||
      existing.models.find(
        (entry) =>
          entry.predictorType === model.predictorType &&
          ((model.modelId && entry.modelId === model.modelId) ||
            (model.artifactTag && entry.artifactTag === model.artifactTag))
      );
    if (existingEntry) {
      return {
        ...existingEntry,
        key: existingEntry.modelId || existingEntry.key,
        name: existingEntry.displayName || existingEntry.name,
        path: existingEntry.path || model.path,
        createdAt: existingEntry.createdAt || model.createdAt,
        status: existingEntry.status ?? "deprecated",
      };
    }
    return {
      key: model.key,
      name: model.name,
      displayName: model.name,
      artifactTag: model.artifactTag || model.name,
      predictorType: model.predictorType,
      path: model.path,
      createdAt: model.createdAt,
      status: "deprecated",
    };
  });

  (["dlib", "cnn"] as const).forEach((predictorType) => {
    const sameType = nextModels
      .filter((entry) => entry.predictorType === predictorType)
      .sort((a, b) => Date.parse(b.createdAt) - Date.parse(a.createdAt));
    if (sameType.length === 0) return;

    const preferred = options?.setActive?.predictorType === predictorType
      ? sameType.find(
          (entry) =>
            (!!options.setActive?.modelId && entry.modelId === options.setActive.modelId) ||
            (!!options.setActive?.artifactTag && entry.artifactTag === options.setActive.artifactTag) ||
            (!!options.setActive?.name &&
              (entry.name === options.setActive.name || entry.artifactTag === options.setActive.name))
        )
      : undefined;
    let activeKey = preferred?.key;
    if (!activeKey || !sameType.some((entry) => entry.key === activeKey)) {
      const existingActive = sameType.find((entry) => entry.status === "active");
      activeKey = existingActive?.key;
      // Bootstrap legacy registries only. A v2 registry with no active model is
      // an intentional lifecycle state and must require explicit promotion.
      if (!activeKey && existing.version === 1) activeKey = sameType[0].key;
    }

    sameType.forEach((entry) => {
      entry.status = activeKey && entry.key === activeKey
        ? "active"
        : entry.status === "candidate"
        ? "candidate"
        : "deprecated";
    });
  });

  const payload: LandmarkModelRegistryPayload = {
    version: existing.version === 2 || nextModels.some((entry) => !!entry.modelId) ? 2 : 1,
    updatedAt: new Date().toISOString(),
    models: nextModels,
  };
  writeLandmarkModelRegistry(effectiveRoot, payload);
  return payload;
}

function resolveModelStatus(
  registry: LandmarkModelRegistryPayload,
  identifier: string,
  predictorType: "dlib" | "cnn"
): LandmarkModelStatus {
  return (
    registry.models.find(
      (entry) =>
        entry.predictorType === predictorType &&
        (entry.modelId === identifier ||
          entry.key === identifier ||
          entry.artifactTag === identifier ||
          entry.name === identifier ||
          entry.key === buildLandmarkModelKey(identifier, predictorType))
    )?.status ??
    "active"
  );
}

function findLandmarkRegistryEntry(
  registry: LandmarkModelRegistryPayload,
  identifier: string,
  predictorType?: "dlib" | "cnn"
): LandmarkModelRegistryEntry | undefined {
  const matches = registry.models.filter(
    (entry) =>
      (!predictorType || entry.predictorType === predictorType) &&
      (entry.modelId === identifier ||
        entry.key === identifier ||
        entry.name === identifier ||
        entry.displayName === identifier ||
        entry.artifactTag === identifier ||
        entry.key === buildLandmarkModelKey(identifier, entry.predictorType))
  );
  return matches.find((entry) => entry.status === "active") ??
    matches.sort((a, b) => Date.parse(b.createdAt) - Date.parse(a.createdAt))[0];
}

function resolveLandmarkPredictorForInference(
  effectiveRoot: string,
  identifier: string,
  requestedPredictor?: "dlib" | "cnn"
): "dlib" | "cnn" {
  const registry = readLandmarkModelRegistry(effectiveRoot);
  const predictorOrder: Array<"cnn" | "dlib"> = requestedPredictor
    ? [requestedPredictor]
    : ["cnn", "dlib"];
  const registered = predictorOrder
    .map((predictorType) => findLandmarkRegistryEntry(registry, identifier, predictorType))
    .find((entry): entry is LandmarkModelRegistryEntry => Boolean(entry));

  if (registered) {
    if (isImmutableLandmarkRegistryEntry(registered)) {
      const issues = validateLandmarkPromotionArtifact(registered);
      if (issues.length > 0) {
        throw new Error(
          `Immutable ${registered.predictorType} model verification failed for "${identifier}". ` +
          issues.map((issue) => issue.message).join(" ")
        );
      }
      return registered.predictorType;
    }

    const modelPath = String(registered.legacyPath || registered.path || "").trim();
    const configPath = String(registered.configPath || "").trim();
    if (!modelPath || !fs.existsSync(modelPath)) {
      throw new Error(`Legacy ${registered.predictorType} model not found for "${identifier}".`);
    }
    if (
      registered.predictorType === "cnn" &&
      (!configPath || !fs.existsSync(configPath))
    ) {
      throw new Error(`Legacy CNN config not found for "${identifier}".`);
    }
    return registered.predictorType;
  }

  const modelsDir = path.join(effectiveRoot, "models");
  for (const predictorType of predictorOrder) {
    const modelPath = predictorType === "cnn"
      ? path.join(modelsDir, `cnn_${identifier}.pth`)
      : path.join(modelsDir, `predictor_${identifier}.dat`);
    if (!fs.existsSync(modelPath)) continue;
    if (
      predictorType === "cnn" &&
      !fs.existsSync(getCnnConfigPath(modelsDir, identifier))
    ) {
      throw new Error(`Legacy CNN config not found for "${identifier}".`);
    }
    return predictorType;
  }

  const expectedType = requestedPredictor ? `${requestedPredictor} ` : "";
  throw new Error(`No registered or legacy ${expectedType}model found for "${identifier}".`);
}

function ensureImmutableRegistryIdentity(
  effectiveRoot: string,
  identifier: string,
  predictorType: "dlib" | "cnn"
): { registry: LandmarkModelRegistryPayload; entry?: LandmarkModelRegistryEntry } {
  const registry = syncLandmarkModelRegistry(effectiveRoot);
  const entry = findLandmarkRegistryEntry(registry, identifier, predictorType);
  if (!entry || entry.modelId) return { registry, entry };

  const identitySeed = `${predictorType}:${path.resolve(entry.path)}:${entry.createdAt}`;
  const modelId = `legacy-${predictorType}:${createHash("sha256")
    .update(identitySeed)
    .digest("hex")
    .slice(0, 20)}`;
  entry.modelId = modelId;
  entry.key = modelId;
  entry.displayName = entry.displayName || entry.name;
  entry.artifactTag = entry.artifactTag || entry.name;
  entry.legacyPath = entry.legacyPath || entry.path;
  registry.version = 2;
  registry.updatedAt = new Date().toISOString();
  writeLandmarkModelRegistry(effectiveRoot, registry);
  return { registry, entry };
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
    const scannedModels = speciesId
      ? listSupportedModelsForRoot(getEffectiveRoot(speciesId), speciesId)
      : listGlobalSupportedModels();

    const models = scannedModels.filter((model) => {
      if (model.modelKind === "obb_detector") {
        if (!includeDeprecated && model.status === "deprecated") return false;
        if (activeOnly && model.status !== "active") return false;
        return true;
      }
      if (activeOnly && model.compatible === false) return false;
      if (!includeDeprecated && model.status === "deprecated") return false;
      if (activeOnly && model.status !== "active") return false;
      return true;
    })
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
    predictorType?: "dlib" | "cnn",
    modelKind?: "landmark" | "obb_detector"
  ) => {
  try {
    const effectiveRoot = getEffectiveRoot(speciesId);
    const modelsDir = path.join(effectiveRoot, "models");
    const dlibPath = path.join(modelsDir, `predictor_${modelName}.dat`);
    const cnnPath = path.join(modelsDir, `cnn_${modelName}.pth`);
    const cnnConfigPath = path.join(modelsDir, `cnn_${modelName}_config.json`);
    const sessionObbPath = path.join(modelsDir, "session_obb_detector.pt");

    const dlibExists = fs.existsSync(dlibPath);
    const cnnExists = fs.existsSync(cnnPath);
    const obbExists = fs.existsSync(sessionObbPath);

    if (predictorType === "dlib" || predictorType === "cnn") {
      const { registry, entry } = ensureImmutableRegistryIdentity(
        effectiveRoot,
        modelName,
        predictorType
      );
      if (!entry) return { ok: false, error: `${predictorType} model not found` };
      const remaining = registry.models.filter((candidate) => candidate.key !== entry.key);
      const unlinkIfFile = (candidate?: string) => {
        if (candidate && fs.existsSync(candidate) && fs.statSync(candidate).isFile()) {
          fs.unlinkSync(candidate);
        }
      };
      // Immutable run artifacts remain archived for audit/rollback. Runtime
      // aliases and registry visibility are removed by an explicit user delete.
      if (entry.legacyPath && path.resolve(entry.legacyPath) !== path.resolve(entry.path)) {
        unlinkIfFile(entry.legacyPath);
      } else if (!entry.runId) {
        unlinkIfFile(entry.path);
      }
      if (entry.artifactTag && entry.predictorType === "cnn") {
        unlinkIfFile(path.join(modelsDir, `cnn_${entry.artifactTag}_config.json`));
      }
      const sameType = remaining
        .filter((candidate) => candidate.predictorType === predictorType)
        .sort((a, b) => Date.parse(b.createdAt) - Date.parse(a.createdAt));
      const replacement = entry.status === "active"
        ? sameType.find(
            (candidate) =>
              candidate.status === "deprecated" &&
              candidate.promotion?.promoted === true &&
              fs.existsSync(candidate.path) &&
              validateLandmarkPromotionArtifact(candidate).length === 0
          )
        : undefined;
      if (replacement) {
        replacement.status = "active";
        replacement.promotion = {
          ...(replacement.promotion || {}),
          promoted: true,
          restoredAfterActiveDeletionAt: new Date().toISOString(),
        };
      }
      if (entry.currentAliasPath && entry.status === "active") {
        const replacement = sameType.find(
          (candidate) =>
            candidate.status === "active" &&
            candidate.currentAliasPath === entry.currentAliasPath &&
            fs.existsSync(candidate.path)
        );
        if (replacement) {
          atomicCopyFileSync(replacement.path, entry.currentAliasPath);
          if (predictorType === "cnn" && replacement.configPath && fs.existsSync(replacement.configPath)) {
            const configAlias = entry.currentAliasPath.replace(/\.pth$/i, "_config.json");
            atomicCopyFileSync(replacement.configPath, configAlias);
          } else if (predictorType === "cnn") {
            unlinkIfFile(entry.currentAliasPath.replace(/\.pth$/i, "_config.json"));
          }
        } else {
          unlinkIfFile(entry.currentAliasPath);
          if (predictorType === "cnn") {
            unlinkIfFile(entry.currentAliasPath.replace(/\.pth$/i, "_config.json"));
          }
        }
      }
      registry.models = remaining;
      registry.version = 2;
      registry.updatedAt = new Date().toISOString();
      writeLandmarkModelRegistry(effectiveRoot, registry);
      return { ok: true, archivedArtifactPath: entry.runId ? entry.path : undefined };
    }
    if (modelKind === "obb_detector") {
      const registryPath = path.join(modelsDir, "obb_registry.json");
      const registry = safeReadJson(registryPath);
      const registered = Array.isArray(registry?.models) ? registry.models : [];
      const entry = registered.find(
        (candidate: any) =>
          candidate?.modelId === modelName || candidate?.path === modelName
      );
      if (!entry) {
        if (!obbExists) return { ok: false, error: "OBB detector not found" };
        fs.unlinkSync(sessionObbPath);
        return { ok: true };
      }
      const remaining = registered.filter((candidate: any) => candidate !== entry);
      const wasActive = entry.status === "active";
      const sessionRaw = safeReadJson(path.join(effectiveRoot, "session.json")) || {};
      const replacement = wasActive
        ? [...remaining]
            .filter(
              (candidate: any) =>
                candidate?.status === "deprecated" &&
                candidate?.promotion?.promoted === true &&
                candidate?.path &&
                fs.existsSync(String(candidate.path)) &&
                !validateObbPromotionCandidate({
                  registryPath,
                  modelIdentifier: String(candidate.modelId || candidate.path),
                  sessionSemanticFingerprint: computeSessionSchemaSemanticFingerprint(
                    sessionRaw.landmarkTemplate,
                    sessionRaw.orientationPolicy
                  ),
                  sessionSemanticVersion: SCHEMA_SEMANTIC_VERSION,
                  sessionOrientationContract: sessionRaw.orientationPolicy,
                }).blocking
            )
            .sort(
              (left: any, right: any) =>
                Date.parse(String(right.createdAt || "")) - Date.parse(String(left.createdAt || ""))
            )[0]
        : undefined;
      if (replacement) {
        replacement.status = "active";
        replacement.promotion = {
          ...(replacement.promotion || {}),
          promoted: true,
          restoredAfterActiveDeletionAt: new Date().toISOString(),
        };
      }
      const configAlias = path.join(modelsDir, "session_obb_detector_config.json");
      const detectorSnapshot = fs.existsSync(sessionObbPath) ? fs.readFileSync(sessionObbPath) : null;
      const configSnapshot = fs.existsSync(configAlias) ? fs.readFileSync(configAlias) : null;
      try {
        if (wasActive && replacement) {
          atomicCopyFileSync(String(replacement.path), sessionObbPath);
          if (replacement.configPath && fs.existsSync(String(replacement.configPath))) {
            atomicCopyFileSync(String(replacement.configPath), configAlias);
          } else {
            restoreFileSnapshot(configAlias, null);
          }
        } else if (wasActive) {
          restoreFileSnapshot(sessionObbPath, null);
          restoreFileSnapshot(configAlias, null);
        }
        atomicWriteJsonSync(registryPath, {
          version: 2,
          updatedAt: new Date().toISOString(),
          models: remaining,
        });
      } catch (error) {
        restoreFileSnapshot(sessionObbPath, detectorSnapshot);
        restoreFileSnapshot(configAlias, configSnapshot);
        throw error;
      }
      return { ok: true, archivedArtifactPath: entry.path };
    }

    // Backward-compatible behavior: if model kind is not specified,
    // delete any supported model matching this session root.
    if (!dlibExists && !cnnExists && !obbExists) return { ok: false, error: "Model not found" };
    if (dlibExists) fs.unlinkSync(dlibPath);
    if (cnnExists) {
      fs.unlinkSync(cnnPath);
      if (fs.existsSync(cnnConfigPath)) fs.unlinkSync(cnnConfigPath);
    }
    if (obbExists) fs.unlinkSync(sessionObbPath);
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
    predictorType?: "dlib" | "cnn",
    modelKind?: "landmark" | "obb_detector"
  ) => {
  try {
    const effectiveRoot = getEffectiveRoot(speciesId);
    if (!MODEL_NAME_RE.test(newName)) {
      return { ok: false, error: "Invalid model name. Use letters, numbers, dot, underscore, or hyphen." };
    }
    if (modelKind === "obb_detector") {
      return { ok: false, error: "OBB detector uses a fixed session model name and cannot be renamed." };
    }
    let resolvedPredictorType = predictorType;
    if (!resolvedPredictorType) {
      const candidates = syncLandmarkModelRegistry(effectiveRoot).models.filter(
        (entry) =>
          entry.modelId === oldName ||
          entry.key === oldName ||
          entry.name === oldName ||
          entry.displayName === oldName ||
          entry.artifactTag === oldName
      );
      resolvedPredictorType =
        candidates.find((entry) => entry.status === "active")?.predictorType ??
        candidates[0]?.predictorType;
    }
    if (resolvedPredictorType === "dlib" || resolvedPredictorType === "cnn") {
      const { registry, entry } = ensureImmutableRegistryIdentity(
        effectiveRoot,
        oldName,
        resolvedPredictorType
      );
      if (!entry) return { ok: false, error: `${resolvedPredictorType} model not found` };
      const collision = registry.models.some(
        (candidate) =>
          candidate.key !== entry.key &&
          candidate.predictorType === resolvedPredictorType &&
          (candidate.displayName || candidate.name).toLowerCase() === newName.toLowerCase()
      );
      if (collision) {
        return { ok: false, error: `A ${resolvedPredictorType} model with that display name already exists` };
      }
      entry.name = newName;
      entry.displayName = newName;
      registry.version = 2;
      registry.updatedAt = new Date().toISOString();
      writeLandmarkModelRegistry(effectiveRoot, registry);
      return { ok: true, modelId: entry.modelId, artifactTag: entry.artifactTag };
    }
    return { ok: false, error: "Model not found" };
  } catch (e: any) {
    console.error("Failed to rename model:", e);
    return { ok: false, error: e.message };
  }
});

ipcMain.handle(
  "ml:promote-model",
  async (
    _event,
    modelIdentifier: string,
    speciesId?: string,
    predictorType?: "dlib" | "cnn",
    modelKind?: "landmark" | "obb_detector"
  ) => {
    try {
      if (modelKind === "obb_detector" || String(modelIdentifier).startsWith("obb:")) {
        const effectiveRoot = getEffectiveRoot(speciesId);
        const modelsDir = path.join(effectiveRoot, "models");
        const registryPath = path.join(modelsDir, "obb_registry.json");
        const registry = safeReadJson(registryPath);
        const models = Array.isArray(registry?.models) ? registry.models : [];
        const entry = models.find(
          (candidate: any) =>
            candidate?.modelId === modelIdentifier || candidate?.path === modelIdentifier
        );
        if (!entry || !entry.path || !fs.existsSync(String(entry.path))) {
          return { ok: false, error: "OBB model not found." };
        }
        const sessionRaw = safeReadJson(path.join(effectiveRoot, "session.json")) || {};
        const candidateValidation = validateObbPromotionCandidate({
          registryPath,
          modelIdentifier,
          sessionSemanticFingerprint: computeSessionSchemaSemanticFingerprint(
            sessionRaw.landmarkTemplate,
            sessionRaw.orientationPolicy
          ),
          sessionSemanticVersion: SCHEMA_SEMANTIC_VERSION,
          sessionOrientationContract: sessionRaw.orientationPolicy,
        });
        if (candidateValidation.blocking) {
          return {
            ok: false,
            error:
              "OBB promotion blocked by immutable artifact/schema verification. " +
              candidateValidation.issues.map((issue) => issue.message).join(" "),
            issues: candidateValidation.issues,
          };
        }
        const priorActiveModelId = String(
          models.find((candidate: any) => candidate !== entry && candidate?.status === "active")
            ?.modelId || ""
        ).trim() || null;
        models.forEach((candidate: any) => {
          candidate.status = candidate === entry ? "active" : "deprecated";
        });
        entry.promotion = {
          ...(entry.promotion || {}),
          promoted: true,
          reason: "manual_override",
          manuallyPromotedAt: new Date().toISOString(),
          priorActiveModelId,
        };
        const detectorAlias = path.join(modelsDir, "session_obb_detector.pt");
        const configAlias = path.join(modelsDir, "session_obb_detector_config.json");
        const detectorSnapshot = fs.existsSync(detectorAlias) ? fs.readFileSync(detectorAlias) : null;
        const configSnapshot = fs.existsSync(configAlias) ? fs.readFileSync(configAlias) : null;
        try {
          atomicCopyFileSync(String(entry.path), detectorAlias);
          if (candidateValidation.configPath) {
            atomicCopyFileSync(candidateValidation.configPath, configAlias);
          } else {
            restoreFileSnapshot(configAlias, null);
          }
          atomicWriteJsonSync(registryPath, {
            version: 2,
            updatedAt: new Date().toISOString(),
            models,
          });
        } catch (error) {
          restoreFileSnapshot(detectorAlias, detectorSnapshot);
          restoreFileSnapshot(configAlias, configSnapshot);
          throw error;
        }
        return { ok: true, modelId: entry.modelId, status: "active" };
      }
      if (predictorType !== "dlib" && predictorType !== "cnn") {
        return { ok: false, error: "A landmark predictor type is required." };
      }
      const effectiveRoot = getEffectiveRoot(speciesId);
      const { registry, entry } = ensureImmutableRegistryIdentity(
        effectiveRoot,
        modelIdentifier,
        predictorType
      );
      if (!entry) return { ok: false, error: "Model not found." };
      const integrityIssues = validateLandmarkPromotionArtifact(entry);
      if (integrityIssues.length > 0) {
        return {
          ok: false,
          error:
            "Landmark promotion blocked by immutable artifact verification. " +
            integrityIssues.map((issue) => issue.message).join(" "),
          issues: integrityIssues,
        };
      }
      const compatibility = await evaluateModelCompatibility({
        speciesId,
        modelName: entry.artifactTag || entry.modelId || entry.name,
        predictorType,
        includeRuntime: false,
      });
      if (compatibility.blocking) {
        return {
          ok: false,
          error:
            "Landmark promotion blocked by schema compatibility verification. " +
            formatCompatibilityErrorSummary(compatibility.issues),
          issues: compatibility.issues,
        };
      }
      const priorActiveModelId =
        registry.models.find(
          (candidate) =>
            candidate.predictorType === predictorType &&
            candidate.key !== entry.key &&
            candidate.status === "active"
        )?.modelId ?? null;
      registry.models.forEach((candidate) => {
        if (candidate.predictorType === predictorType) {
          candidate.status = candidate.key === entry.key ? "active" : "deprecated";
        }
      });
      entry.promotion = {
        ...(entry.promotion || {}),
        promoted: true,
        reason: "manual_override",
        manuallyPromotedAt: new Date().toISOString(),
        priorActiveModelId,
      };
      registry.version = 2;
      registry.updatedAt = new Date().toISOString();
      const currentConfigAlias =
        predictorType === "cnn"
          ? entry.currentAliasPath?.replace(/\.pth$/i, "_config.json")
          : undefined;
      const aliasSnapshot =
        entry.currentAliasPath && fs.existsSync(entry.currentAliasPath)
          ? fs.readFileSync(entry.currentAliasPath)
          : null;
      const configSnapshot =
        currentConfigAlias && fs.existsSync(currentConfigAlias)
          ? fs.readFileSync(currentConfigAlias)
          : null;
      try {
        if (
          entry.currentAliasPath &&
          fs.existsSync(entry.path) &&
          path.resolve(entry.currentAliasPath) !== path.resolve(entry.path)
        ) {
          atomicCopyFileSync(entry.path, entry.currentAliasPath);
        }
        if (
          predictorType === "cnn" &&
          entry.configPath &&
          fs.existsSync(entry.configPath) &&
          currentConfigAlias
        ) {
          atomicCopyFileSync(entry.configPath, currentConfigAlias);
        } else if (predictorType === "cnn" && currentConfigAlias) {
          restoreFileSnapshot(currentConfigAlias, null);
        }
        writeLandmarkModelRegistry(effectiveRoot, registry);
      } catch (error) {
        if (entry.currentAliasPath) restoreFileSnapshot(entry.currentAliasPath, aliasSnapshot);
        if (currentConfigAlias) restoreFileSnapshot(currentConfigAlias, configSnapshot);
        throw error;
      }
      return { ok: true, modelId: entry.modelId, status: "active" };
    } catch (e: any) {
      console.error("Failed to promote model:", e);
      return { ok: false, error: e.message };
    }
  }
);

// Get info about a specific model
ipcMain.handle("ml:get-model-info", async (_event, modelName: string, speciesId?: string) => {
  try {
    const effectiveRoot = getEffectiveRoot(speciesId);
    const modelsDir = path.join(effectiveRoot, "models");
    const dlibPath = path.join(modelsDir, `predictor_${modelName}.dat`);
    const cnnPath = path.join(modelsDir, `cnn_${modelName}.pth`);
    const sessionObbPath = path.join(modelsDir, "session_obb_detector.pt");
    const schemaName = readSchemaDisplayName(effectiveRoot, speciesId);

    if (fs.existsSync(dlibPath)) {
      const stats = fs.statSync(dlibPath);
      return {
        ok: true,
        model: { name: modelName, path: dlibPath, size: stats.size, createdAt: stats.birthtime, predictorType: "dlib", modelKind: "landmark", speciesId, schemaName },
      };
    }
    if (fs.existsSync(cnnPath)) {
      const stats = fs.statSync(cnnPath);
      return {
        ok: true,
        model: { name: modelName, path: cnnPath, size: stats.size, createdAt: stats.birthtime, predictorType: "cnn", modelKind: "landmark", speciesId, schemaName },
      };
    }
    if (modelName === "Session OBB Detector" && fs.existsSync(sessionObbPath)) {
      const stats = fs.statSync(sessionObbPath);
      return {
        ok: true,
        model: { name: modelName, path: sessionObbPath, size: stats.size, createdAt: stats.birthtime, modelKind: "obb_detector", speciesId, schemaName },
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
  conf?: number;
  nmsIou?: number;
  maxObjects?: number;
  imgsz?: ObbImageSize;
  detectionPreset?: ObbDetectionPreset;
  allowIncompatible?: boolean;
}

ipcMain.handle("ml:detect-specimens", async (_event, imagePath: string, options?: DetectionOptions) => {
  try {
    const speciesId = options?.speciesId;
    const effectiveRoot = getEffectiveRoot(speciesId);
    const sessionJsonPath = path.join(effectiveRoot, "session.json");
    const sessionMeta = safeReadJson(sessionJsonPath) || {};
    const persistedDetectionSettings = readNormalizedSessionObbDetectionSettings(sessionMeta);
    const resolvedDetectionSettings = normalizeObbDetectionSettings({
      ...persistedDetectionSettings,
      ...options,
    });

    const sessionObbPath = path.join(effectiveRoot, "models", "session_obb_detector.pt");
    if (fs.existsSync(sessionObbPath)) {
      const sessionContract = loadSessionOrientationPolicyForCompatibility(String(speciesId || ""));
      const detector = resolveTrainedObbDetector({
        aliasPath: sessionObbPath,
        registryPath: path.join(effectiveRoot, "models", "obb_registry.json"),
        sessionSemanticFingerprint: sessionContract.schemaSemanticFingerprint,
        sessionSemanticVersion: sessionContract.schemaSemanticVersion,
        sessionOrientationContract: sessionMeta?.orientationPolicy,
      });
      const compatibility = {
        compatible: detector.compatible,
        blocking: detector.blocking,
        requiresOverride: detector.blocking,
        issues: detector.issues,
      };
      if (detector.blocking && options?.allowIncompatible !== true) {
        return {
          ok: false,
          boxes: [],
          requiresOverride: true,
          compatibility,
          detectorProvenance: detector.provenance,
          error:
            `OBB detector compatibility check blocked inference. ${formatCompatibilityErrorSummary(
              detector.issues
            )}`,
        };
      }
      const thresholdPlan = resolveObbInferenceThresholdPlan({
        hasTrainedArtifact: true,
        sessionSettingsCustomized: readSessionObbDetectionSettingsCustomized(sessionMeta),
        detectionPreset: resolvedDetectionSettings.detectionPreset,
        conf: resolvedDetectionSettings.conf,
        nmsIou: resolvedDetectionSettings.nmsIou,
      });
      // Route all trained-session detection through the shared OBB detector script.
      const detectArgs = [
        imagePath,
        "--multi",
        "--yolo-model",
        sessionObbPath,
        "--max-specimens",
        String(resolvedDetectionSettings.maxObjects),
        "--detection-preset",
        thresholdPlan.detectionPreset,
        "--imgsz",
        String(resolvedDetectionSettings.imgsz),
      ];
      if (thresholdPlan.conf !== undefined) {
        detectArgs.push("--conf", String(thresholdPlan.conf));
      }
      if (thresholdPlan.nmsIou !== undefined) {
        detectArgs.push("--nms-iou", String(thresholdPlan.nmsIou));
      }

      const out = await runBundledScript("detect_specimen", detectArgs);
      const data = JSON.parse(out.trim());
      if (!data) {
        return { ok: false, error: "No detection result", boxes: [] };
      }
      if (data.ok === true && Array.isArray(data.boxes)) {
        return {
          ...data,
          detectorProvenance: detector.provenance,
          compatibility,
        };
      }
      return { ok: false, error: "Unexpected detection format", boxes: [] };
    }

    if (!superAnnotator.isRunning) {
      await superAnnotator.send({ cmd: "init" });
    }

    const classPrompt =
      String(sessionMeta?.name || "").trim() ||
      String(speciesId || "").trim().replace(/[-_]+/g, " ") ||
      "specimen";

    const result = await superAnnotator.send({
      cmd: "annotate",
      image_path: imagePath,
      class_name: classPrompt,
      options: {
        conf_threshold: resolvedDetectionSettings.conf,
        nms_iou: resolvedDetectionSettings.nmsIou,
        sam_enabled: false,
        max_objects: resolvedDetectionSettings.maxObjects,
        imgsz: resolvedDetectionSettings.imgsz,
        detection_mode: "auto",
        detection_preset: resolvedDetectionSettings.detectionPreset,
        orientation_policy: sessionMeta?.orientationPolicy ?? undefined,
      },
    });

    if (result?.status === "result" && Array.isArray(result.objects)) {
      const boxes = result.objects.map((obj: any) => ({
        ...obj.box,
        ...(obj.obb?.corners ? { obbCorners: obj.obb.corners } : {}),
        ...(typeof obj.obb?.angle === "number" ? { angle: obj.obb.angle } : {}),
      }));
      const detectionMethod = result.detection_method ?? "yolo_world";
      return {
        ok: true,
        boxes,
        num_detections: boxes.length,
        detection_method: detectionMethod,
        fallback: result.detection_method !== "yolo_obb",
        detectorProvenance: resolveZeroShotDetector(
          path.join(__dirname, "..", "yolov8s-worldv2.pt"),
          detectionMethod
        ),
      };
    }

    return { ok: false, error: result?.error || "Zero-shot detection failed", boxes: [] };
  } catch (e: any) {
    console.error("Specimen detection failed:", e);
    return { ok: false, error: e.message, boxes: [] };
  }
});

// Check OBB detector runtime availability
/**
 * Reveal a model artifact in the OS file manager.
 *
 * Only paths inside the active project root are accepted: the renderer must
 * never be able to make the main process open an arbitrary location.
 */
ipcMain.handle("shell:show-item-in-folder", async (_event, targetPath: string) => {
  try {
    if (typeof targetPath !== "string" || !targetPath.trim()) {
      return { ok: false, error: "No path provided." };
    }
    const resolved = path.resolve(targetPath);
    const root = path.resolve(projectRoot);
    const relative = path.relative(root, resolved);
    if (relative.startsWith("..") || path.isAbsolute(relative)) {
      return { ok: false, error: "Path is outside the current project root." };
    }
    if (!fs.existsSync(resolved)) {
      return { ok: false, error: "That file no longer exists." };
    }
    shell.showItemInFolder(resolved);
    return { ok: true };
  } catch (e: any) {
    console.error("shell:show-item-in-folder failed:", e);
    return { ok: false, error: e.message };
  }
});

ipcMain.handle("ml:check-yolo", async () => {
  try {
    const out = await runBundledScript("detect_specimen", ["--check"]);

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
  atomicWriteJsonSync(finalizedListPath, deduped);
}

function finalizedNameListsMatch(left: string[], right: string[]): boolean {
  if (left.length !== right.length) return false;
  return left.every((name, index) => name === right[index]);
}

function resolveFinalizedState(
  sessionDir: string,
  options?: { reconcile?: boolean }
): { names: string[]; nameSet: Set<string>; changed: boolean } {
  const imagesDir = path.join(sessionDir, "images");
  const labelsDir = path.join(sessionDir, "labels");
  const imageFiles = fs.existsSync(imagesDir)
    ? fs.readdirSync(imagesDir).filter((filename) => IMAGE_EXTS.test(filename))
    : [];
  const imageNameByLower = new Map<string, string>();
  imageFiles.forEach((filename) => {
    imageNameByLower.set(filename.toLowerCase(), filename);
  });

  const resolvedLower = new Set<string>();
  const listNames = readFinalizedList(sessionDir);
  listNames.forEach((name) => {
    const lower = path.basename(String(name || "")).toLowerCase();
    if (imageNameByLower.has(lower)) {
      resolvedLower.add(lower);
    }
  });

  imageFiles.forEach((filename) => {
    const labelPath = path.join(labelsDir, filename.replace(/\.\w+$/, ".json"));
    if (!fs.existsSync(labelPath)) return;
    try {
      const payload = JSON.parse(fs.readFileSync(labelPath, "utf-8"));
      if (payload?.finalizedDetection?.isFinalized) {
        resolvedLower.add(filename.toLowerCase());
      }
    } catch {
      // Ignore malformed labels during reconciliation.
    }
  });

  const resolvedNames = imageFiles.filter((filename) => resolvedLower.has(filename.toLowerCase()));
  const existingCanonical = listNames
    .map((name) => imageNameByLower.get(path.basename(String(name || "")).toLowerCase()) || "")
    .filter(Boolean);
  const changed = !finalizedNameListsMatch(existingCanonical, resolvedNames);

  if (options?.reconcile && changed) {
    writeFinalizedList(sessionDir, resolvedNames);
  }

  return {
    names: resolvedNames,
    nameSet: new Set(resolvedNames.map((name) => name.toLowerCase())),
    changed,
  };
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

function readFinalizedDetectionSnapshot(
  sessionDir: string,
  filename: string
): { isFinalized: boolean; boxSignature?: string } {
  const safeFilename = path.basename(String(filename || "").trim());
  if (!safeFilename) return { isFinalized: false };
  const labelPath = path.join(
    sessionDir,
    "labels",
    safeFilename.replace(/\.\w+$/, ".json")
  );
  if (!fs.existsSync(labelPath)) {
    return { isFinalized: false };
  }
  try {
    const payload = JSON.parse(fs.readFileSync(labelPath, "utf-8"));
    return {
      isFinalized: Boolean(payload?.finalizedDetection?.isFinalized),
      ...(typeof payload?.finalizedDetection?.boxSignature === "string"
        ? { boxSignature: String(payload.finalizedDetection.boxSignature) }
        : {}),
    };
  } catch {
    return { isFinalized: false };
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

function clearFinalizedStateForImage(
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
  const resolvedState = resolveFinalizedState(sessionDir, { reconcile: true });
  const retainedNames = resolvedState.names.filter((entry) => entry.toLowerCase() !== safeLower);
  const removedFromList = retainedNames.length !== resolvedState.names.length;
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

function clearFinalizedStateForSession(
  speciesId: string
): { removedSegmentsTotal: number; clearedLabels: number } {
  const sessionDir = getSessionDir(speciesId);
  const labelsDir = path.join(sessionDir, "labels");
  const segmentsDir = path.join(sessionDir, "segments");
  let clearedLabels = 0;
  let removedSegmentsTotal = 0;

  if (fs.existsSync(labelsDir)) {
    for (const filename of fs.readdirSync(labelsDir)) {
      if (!filename.endsWith(".json")) continue;
      const result = removeFinalizedFromLabel(path.join(labelsDir, filename));
      if (result.hadFinalizedDetection) {
        clearedLabels += 1;
      }
    }
  }

  if (fs.existsSync(segmentsDir)) {
    removedSegmentsTotal = fs.readdirSync(segmentsDir).length;
    try {
      fs.rmSync(segmentsDir, { recursive: true, force: true });
    } catch {
      // best effort cleanup
    }
  }
  fs.mkdirSync(segmentsDir, { recursive: true });
  writeFinalizedList(sessionDir, []);
  cancelAllQueuedSegmentSavesForSpecies(speciesId);

  return { removedSegmentsTotal, clearedLabels };
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
      let iterativeSam2Eligible = false;
      try {
        iterativeSam2Eligible = (await checkSam2MinimumRequirements()).ok;
      } catch {
        iterativeSam2Eligible = false;
      }
      if (iterativeSam2Eligible && !superAnnotator.initCompleted) {
        try {
          const initRes = await superAnnotator.send({ cmd: "init" });
          if (initRes?.status !== "error") {
            superAnnotator.initCompleted = true;
          } else {
            iterativeSam2Eligible = false;
          }
        } catch {
          iterativeSam2Eligible = false;
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
          iterative: iterativeSam2Eligible,
          expand_ratio: 0.10,
          allow_rectangle_fallback: false,
        });

        const currentSignature = finalizedSegmentSignatureCache.get(job.queueKey);
        const expectedCount = job.acceptedBoxes.length;
        const savedCount = Number(saveResult?.saved ?? 0);
        const details = Array.isArray(saveResult?.details)
          ? (saveResult.details as SegmentSaveDetail[])
          : undefined;
        if (currentSignature !== job.signature) {
          setSegmentQueueStatus(job.speciesId, job.filename, "idle");
        } else if (saveResult?.status === "error") {
          deleteSegmentsForImage(job.sessionDir, job.filename, job.imagePath);
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
            "finalized_without_segments",
            job.signature,
            saveResult.error || "segment_save_failed",
            { expectedCount, savedCount: 0 },
            details
          );
        } else if (savedCount < expectedCount) {
          deleteSegmentsForImage(job.sessionDir, job.filename, job.imagePath);
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
            "finalized_without_segments",
            job.signature,
            summarizeSegmentSaveDetails(details) || `partial_segment_save:${savedCount}/${expectedCount}`,
            { expectedCount, savedCount },
            details
          );
        } else {
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
            { expectedCount, savedCount },
            details
          );
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
  return clearFinalizedStateForImage(speciesId, filename, imagePath);
}

const INFERENCE_REVIEW_DRAFTS_FILE = "inference_review_drafts.json";
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
    orientation_override?: "left" | "right" | "up" | "down" | "uncertain";
    orientation_hint?: {
      orientation?: "left" | "right" | "up" | "down";
      confidence?: number;
      source?: string;
      head_point?: [number, number];
      tail_point?: [number, number];
    };
  };
  landmarks: {
    id: number;
    x: number;
    y: number;
    confidence?: number;
    heatmap_entropy?: number;
    isPredicted?: boolean;
    isCorrected?: boolean;
    predictionVersion?: string;
  }[];
  inference_metadata?: {
    mask_source?: string;
    canonical_flip_applied?: boolean;
    direction_source?: string;
    inferred_direction?: "left" | "right" | null;
    inferred_direction_confidence?: number;
    direction_confidence?: number;
    used_flipped_crop?: boolean;
    was_flipped?: boolean;
    selection_reason?: string;
    detector_hint_orientation?: string | null;
    detector_hint_source?: string | null;
    orientation_warning?: { code?: string; message?: string } | null;
    clamped_landmark_count?: number;
    landmark_count?: number;
    landmark_heatmap_entropy?: number;
    model_disagreement?: number;
    ood_score?: number;
    ood_score_source?: string;
    inferenceSignature?: string;
  };
};

type InferenceReviewDraftItem = {
  key: string;
  imagePath: string;
  filename: string;
  specimens: InferenceDraftSpecimen[];
  edited: boolean;
  wasEdited?: boolean;
  saved: boolean;
  reviewComplete?: boolean;
  committedAt?: string | null;
  landmarkModelKey?: string;
  landmarkPredictorType?: "dlib" | "cnn";
  boxSignature?: string;
  inferenceSignature?: string;
  detectorProvenance?: DetectionModelProvenance;
  prioritySignals?: HitlPrioritySignals;
  reviewPriority?: HitlReviewPriority;
  commitFailureCount?: number;
  updatedAt: string;
};

type InferenceReviewDraftsPayload = {
  version: 1;
  updatedAt: string;
  items: Record<string, InferenceReviewDraftItem>;
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
      predictorType?: "dlib" | "cnn";
    };
    detection: {
      key: string;
      name?: string;
    };
  };
  preferences?: {
    lastUsedLandmarkModelKey?: string;
    lastUsedPredictorType?: "dlib" | "cnn";
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

function ensureInferenceSessionManifest(args: {
  speciesId: string;
  inferenceSessionId: string;
  displayName?: string;
  landmarkModelKey?: string;
  landmarkModelName?: string;
  landmarkPredictorType?: "dlib" | "cnn";
  detectionModelKey?: string;
  detectionModelName?: string;
  preferences?: {
    lastUsedLandmarkModelKey?: string;
    lastUsedPredictorType?: "dlib" | "cnn";
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
  const sessionDraftPath = getInferenceReviewDraftsPath(speciesId, inferenceSessionId);
  const sessionDir = getInferenceSessionDir(speciesId, inferenceSessionId);
  fs.mkdirSync(sessionDir, { recursive: true });

  if (!fs.existsSync(sessionDraftPath) && fs.existsSync(legacyDraftPath)) {
    try {
      fs.copyFileSync(legacyDraftPath, sessionDraftPath);
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
  atomicWriteJsonSync(draftPath, payload);
}

function sanitizeDraftInferenceMetadata(
  value: unknown
): InferenceDraftSpecimen["inference_metadata"] | undefined {
  if (!value || typeof value !== "object") return undefined;
  const raw = value as Record<string, unknown>;
  const finiteNumber = (candidate: unknown): number | undefined => {
    const numeric = Number(candidate);
    return Number.isFinite(numeric) ? numeric : undefined;
  };
  const nonEmptyString = (candidate: unknown): string | undefined => {
    if (typeof candidate !== "string") return undefined;
    const normalized = candidate.trim();
    return normalized || undefined;
  };
  const orientationWarning = raw.orientation_warning === null
    ? null
    : raw.orientation_warning && typeof raw.orientation_warning === "object"
      ? {
          ...(nonEmptyString((raw.orientation_warning as any).code)
            ? { code: nonEmptyString((raw.orientation_warning as any).code) }
            : {}),
          ...(nonEmptyString((raw.orientation_warning as any).message)
            ? { message: nonEmptyString((raw.orientation_warning as any).message) }
            : {}),
        }
      : undefined;
  const metadata: NonNullable<InferenceDraftSpecimen["inference_metadata"]> = {
    ...(nonEmptyString(raw.mask_source) ? { mask_source: nonEmptyString(raw.mask_source) } : {}),
    ...(typeof raw.canonical_flip_applied === "boolean"
      ? { canonical_flip_applied: raw.canonical_flip_applied }
      : {}),
    ...(nonEmptyString(raw.direction_source)
      ? { direction_source: nonEmptyString(raw.direction_source) }
      : {}),
    ...(raw.inferred_direction === null || raw.inferred_direction === "left" || raw.inferred_direction === "right"
      ? { inferred_direction: raw.inferred_direction }
      : {}),
    ...(finiteNumber(raw.inferred_direction_confidence) !== undefined
      ? { inferred_direction_confidence: finiteNumber(raw.inferred_direction_confidence) }
      : {}),
    ...(finiteNumber(raw.direction_confidence) !== undefined
      ? { direction_confidence: finiteNumber(raw.direction_confidence) }
      : {}),
    ...(typeof raw.used_flipped_crop === "boolean" ? { used_flipped_crop: raw.used_flipped_crop } : {}),
    ...(typeof raw.was_flipped === "boolean" ? { was_flipped: raw.was_flipped } : {}),
    ...(nonEmptyString(raw.selection_reason)
      ? { selection_reason: nonEmptyString(raw.selection_reason) }
      : {}),
    ...(raw.detector_hint_orientation === null || nonEmptyString(raw.detector_hint_orientation)
      ? { detector_hint_orientation: raw.detector_hint_orientation === null
          ? null
          : nonEmptyString(raw.detector_hint_orientation) }
      : {}),
    ...(raw.detector_hint_source === null || nonEmptyString(raw.detector_hint_source)
      ? { detector_hint_source: raw.detector_hint_source === null
          ? null
          : nonEmptyString(raw.detector_hint_source) }
      : {}),
    ...(orientationWarning !== undefined ? { orientation_warning: orientationWarning } : {}),
    ...(finiteNumber(raw.clamped_landmark_count) !== undefined
      ? { clamped_landmark_count: finiteNumber(raw.clamped_landmark_count) }
      : {}),
    ...(finiteNumber(raw.landmark_count) !== undefined
      ? { landmark_count: finiteNumber(raw.landmark_count) }
      : {}),
    ...(finiteNumber(raw.landmark_heatmap_entropy) !== undefined
      ? { landmark_heatmap_entropy: finiteNumber(raw.landmark_heatmap_entropy) }
      : {}),
    ...(finiteNumber(raw.model_disagreement) !== undefined
      ? { model_disagreement: finiteNumber(raw.model_disagreement) }
      : {}),
    ...(finiteNumber(raw.ood_score) !== undefined ? { ood_score: finiteNumber(raw.ood_score) } : {}),
    ...(nonEmptyString(raw.ood_score_source)
      ? { ood_score_source: nonEmptyString(raw.ood_score_source) }
      : {}),
    ...(nonEmptyString(raw.inferenceSignature)
      ? { inferenceSignature: nonEmptyString(raw.inferenceSignature) }
      : {}),
  };
  return Object.keys(metadata).length > 0 ? metadata : undefined;
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
          s.box.orientation_override === "up" ||
          s.box.orientation_override === "down" ||
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
                s.box.orientation_hint.orientation === "left" ||
                s.box.orientation_hint.orientation === "right" ||
                s.box.orientation_hint.orientation === "up" ||
                s.box.orientation_hint.orientation === "down"
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
            ...(Number.isFinite(Number(lm?.confidence))
              ? { confidence: Number(lm.confidence) }
              : {}),
            ...(Number.isFinite(Number(lm?.heatmap_entropy))
              ? { heatmap_entropy: Number(lm.heatmap_entropy) }
              : {}),
            ...(typeof lm?.isPredicted === "boolean" ? { isPredicted: lm.isPredicted } : {}),
            ...(typeof lm?.isCorrected === "boolean" ? { isCorrected: lm.isCorrected } : {}),
            ...(typeof lm?.predictionVersion === "string" && lm.predictionVersion.trim()
              ? { predictionVersion: lm.predictionVersion.trim() }
              : {}),
          }))
        : [],
      ...(sanitizeDraftInferenceMetadata(s?.inference_metadata)
        ? { inference_metadata: sanitizeDraftInferenceMetadata(s.inference_metadata) }
        : {}),
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
      schemaKind?: SessionSchemaKind;
      schemaSourceId?: string;
      schemaFingerprint?: string;
      schemaSemanticFingerprint?: string;
      schemaSemanticVersion?: number;
      orientationPolicy?: {
        mode?: "directional" | "bilateral" | "axial" | "invariant";
        targetOrientation?: "left" | "right";
        headCategories?: string[];
        tailCategories?: string[];
        anteriorAnchorIds?: number[];
        posteriorAnchorIds?: number[];
        bilateralPairs?: [number, number][];
        bilateralClassAxis?: "vertical_obb";
        obbLevelingMode?: "on" | "off";
      };
    }
  ) => {
    try {
      const sessionDir = getSessionDir(args.speciesId);
      const normalizedLandmarkTemplate = normalizeLandmarkTemplate(args.landmarkTemplate);
      if (normalizedLandmarkTemplate.length === 0) {
        return { ok: false, error: "A session requires at least one landmark." };
      }
      const orientationMode = String(args.orientationPolicy?.mode || "").trim().toLowerCase();
      if (!ORIENTATION_MODES.has(orientationMode)) {
        return {
          ok: false,
          error: "Choose an explicit orientation policy before creating the session.",
        };
      }
      const normalizedOrientationPolicy = normalizeOrientationPolicy(
        args.orientationPolicy,
        normalizedLandmarkTemplate
      );
      const schemaFingerprint =
        String(args.schemaFingerprint || "").trim() ||
        computeSchemaFingerprint(normalizedLandmarkTemplate);
      const schemaSemanticFingerprint = computeSessionSchemaSemanticFingerprint(
        normalizedLandmarkTemplate,
        normalizedOrientationPolicy
      );
      for (const sub of ["images", "labels", "models", "xml", "corrected_images", "debug"]) {
        fs.mkdirSync(path.join(sessionDir, sub), { recursive: true });
      }
      fs.writeFileSync(
        path.join(sessionDir, "session.json"),
        JSON.stringify(
          {
            speciesId: args.speciesId,
            name: args.name,
            landmarkTemplate: normalizedLandmarkTemplate,
            schemaFingerprint,
            schemaSemanticFingerprint,
            schemaSemanticVersion: SCHEMA_SEMANTIC_VERSION,
            schemaKind: normalizeSessionSchemaKind(args.schemaKind),
            schemaSourceId: normalizeSessionSchemaSourceId(args.schemaSourceId),
            orientationPolicy: normalizedOrientationPolicy,
            orientationPolicyConfigured: true,
            orientationPolicyConfiguredAt: new Date().toISOString(),
            augmentationPolicy: { gravity_aligned: true },
            obbTrainingSettingsCustomized: false,
            obbDetectionSettingsCustomized: false,
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
        orientation_override?: "left" | "right" | "up" | "down" | "uncertain";
        obbCorners?: [number, number][];
        angle?: number;
        class_id?: number;
        orientation_hint?: {
          orientation?: "left" | "right" | "up" | "down";
          confidence?: number;
          source?: string;
        };
        landmarks?: { id: number; x: number; y: number; isSkipped?: boolean }[];
      }[];
      imagePath?: string;
      generateSegments?: boolean;
    }
  ) => {
    try {
      const sessionDir = getSessionDir(args.speciesId);
      const sessionImagePath = path.join(sessionDir, "images", args.filename);
      const resolvedImagePath = fs.existsSync(sessionImagePath) ? sessionImagePath : args.imagePath;
      const generateSegments = Boolean(args.generateSegments);
      const acceptedBoxes = normalizeFinalizedAcceptedBoxes(args.boxes || []);

      const cacheKey = buildSegmentQueueImageKey(args.speciesId, args.filename);
      const signature = buildAcceptedBoxesSignature(acceptedBoxes);
      const currentStatus = getSegmentQueueStatus(args.speciesId, args.filename);
      const persistedFinalized = readFinalizedDetectionSnapshot(sessionDir, args.filename);
      const inFlightForSameSignature =
        (currentStatus.state === "queued" || currentStatus.state === "running") &&
        currentStatus.signature === signature;
      const alreadyFinalized =
        persistedFinalized.isFinalized &&
        persistedFinalized.boxSignature === signature &&
        !inFlightForSameSignature;

      if (alreadyFinalized) {
        setSegmentQueueStatus(
          args.speciesId,
          args.filename,
          "already_finalized",
          signature,
          undefined,
          { expectedCount: acceptedBoxes.length, savedCount: acceptedBoxes.length }
        );
        finalizedSegmentSignatureCache.set(cacheKey, signature);
        return {
          ok: true,
          finalized: true,
          queued: false,
          acceptedCount: acceptedBoxes.length,
          signature,
          skipped: false,
          segmentSaveQueued: false,
          segmentQueueState: "already_finalized" as const,
          expectedCount: acceptedBoxes.length,
          savedCount: acceptedBoxes.length,
        };
      }

      if (inFlightForSameSignature) {
        finalizedSegmentSignatureCache.set(cacheKey, signature);
        return {
          ok: true,
          finalized: false,
          queued: currentStatus.state === "queued",
          acceptedCount: acceptedBoxes.length,
          signature,
          skipped: false,
          segmentSaveQueued: currentStatus.state === "queued",
          segmentQueueState: currentStatus.state,
          reason: currentStatus.reason,
          expectedCount: currentStatus.expectedCount,
          savedCount: currentStatus.savedCount,
          details: currentStatus.details,
        };
      }

      if (!generateSegments) {
        deleteSegmentsForImage(sessionDir, args.filename, resolvedImagePath);
        persistFinalizedDetection(
          sessionDir,
          args.speciesId,
          args.filename,
          acceptedBoxes,
          signature
        );
        setSegmentQueueStatus(
          args.speciesId,
          args.filename,
          "finalized_without_segments",
          signature,
          "segments_disabled_sam2_toggle_off",
          { expectedCount: acceptedBoxes.length, savedCount: 0 }
        );
        finalizedSegmentSignatureCache.set(cacheKey, signature);
        return {
          ok: true,
          finalized: true,
          queued: false,
          acceptedCount: acceptedBoxes.length,
          signature,
          skipped: false,
          segmentSaveQueued: false,
          segmentQueueState: "finalized_without_segments" as const,
          reason: "segments_disabled_sam2_toggle_off",
          expectedCount: acceptedBoxes.length,
          savedCount: 0,
        };
      }

      finalizedSegmentSignatureCache.set(cacheKey, signature);

      let segmentQueueState: "queued" | "skipped" = "skipped";
      let segmentSaveQueued = false;
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

      return {
        ok: true,
        finalized: false,
        queued: segmentSaveQueued,
        acceptedCount: acceptedBoxes.length,
        signature,
        skipped: segmentQueueState === "skipped",
        segmentSaveQueued,
        segmentQueueState,
        ...(segmentQueueState === "skipped" ? { reason: "no_boxes_or_image" } : {}),
        ...(acceptedBoxes.length > 0 ? { expectedCount: acceptedBoxes.length } : {}),
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
      const representativeImageDimensions = summarizeRepresentativeImageDimensions(
        fs.existsSync(imagesDir)
          ? fs.readdirSync(imagesDir)
            .filter((f) => IMAGE_EXTS.test(f))
            .map((filename) => path.join(imagesDir, filename))
          : []
      );
      if (fs.existsSync(sessionJsonPath)) {
        try {
          meta = JSON.parse(fs.readFileSync(sessionJsonPath, "utf-8"));
          if (meta && typeof meta === "object") {
            meta.orientationPolicyConfigured = hasConfiguredOrientationPolicy(meta);
            meta.orientationPolicy = normalizeOrientationPolicy(
              meta.orientationPolicy,
              meta.landmarkTemplate
            );
            meta.obbTrainingSettings = readNormalizedSessionObbTrainingSettings(meta);
            meta.obbDetectionSettings = readNormalizedSessionObbDetectionSettings(meta);
            meta.obbTrainingSettingsCustomized = readSessionObbTrainingSettingsCustomized(meta);
            meta.obbDetectionSettingsCustomized = readSessionObbDetectionSettingsCustomized(meta);
            meta.representativeImageDimensions = representativeImageDimensions;
            const schemaMetadata = resolveSessionSchemaMetadata(meta);
            Object.assign(meta, schemaMetadata);
          }
        } catch (_) {
          // skip bad session.json
        }
      }
      const effectiveRoot = getEffectiveRoot(args.speciesId);
      meta = meta && typeof meta === "object" ? meta : {};
      meta.obbDetectorReady = isVerifiedTrainedObbDetector(
        resolveObbDetectorForEffectiveRoot(effectiveRoot)
      );
      meta.obbTrainingSettings = readNormalizedSessionObbTrainingSettings(meta);
      meta.obbDetectionSettings = readNormalizedSessionObbDetectionSettings(meta);
      meta.obbTrainingSettingsCustomized = readSessionObbTrainingSettingsCustomized(meta);
      meta.obbDetectionSettingsCustomized = readSessionObbDetectionSettingsCustomized(meta);
      meta.representativeImageDimensions = representativeImageDimensions;

      if (!fs.existsSync(imagesDir)) {
        resolveFinalizedState(sessionDir, { reconcile: true });
        return { ok: true, images: [], meta, warnings };
      }

      const finalizedState = resolveFinalizedState(sessionDir, { reconcile: true });
      const finalizedSet = finalizedState.nameSet;

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
            const finalizedAccepted = Array.isArray(labelData?.finalizedDetection?.acceptedBoxes)
              ? labelData.finalizedDetection.acceptedBoxes
              : [];
            if (isFinalizedFromLabel && finalizedAcceptedBoxesHaveObb(finalizedAccepted)) {
              boxes = normalizeFinalizedAcceptedBoxesForSession(finalizedAccepted);
            } else if (Array.isArray(labelData?.boxes)) {
              boxes = normalizeSessionStoredBoxes(labelData.boxes);
              hasBoxes = boxes.some((b: any) => Number(b?.width) > 0 && Number(b?.height) > 0);
            }
            hasBoxes = boxes.some((b: any) => Number(b?.width) > 0 && Number(b?.height) > 0);
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
      const finalizedState = resolveFinalizedState(sessionDir, { reconcile: true });

      if (!fs.existsSync(labelPath)) {
        return {
          ok: true,
          boxes: [],
          finalized: finalizedState.nameSet.has(safeFilename.toLowerCase()),
          warnings,
        };
      }

      const labelData = JSON.parse(fs.readFileSync(labelPath, "utf-8"));
      const finalizedAccepted = Array.isArray(labelData?.finalizedDetection?.acceptedBoxes)
        ? labelData.finalizedDetection.acceptedBoxes
        : [];
      return {
        ok: true,
        boxes:
          Boolean(labelData?.finalizedDetection?.isFinalized) &&
          finalizedAcceptedBoxesHaveObb(finalizedAccepted)
            ? normalizeFinalizedAcceptedBoxesForSession(finalizedAccepted)
            : normalizeSessionStoredBoxes(labelData?.boxes),
        finalized:
          finalizedState.nameSet.has(safeFilename.toLowerCase()) ||
          Boolean(labelData?.finalizedDetection?.isFinalized),
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

ipcMain.handle("schema:list-templates", async () => {
  try {
    return { ok: true, templates: listReusableSchemaTemplates() };
  } catch (e: any) {
    console.error("schema:list-templates failed:", e);
    return { ok: false, templates: [], error: e.message };
  }
});

ipcMain.handle(
  "schema:save-custom-template",
  async (
    _event,
    args: {
      name: string;
      description?: string;
      landmarks: any[];
      orientationPolicy?: any;
      sourcePresetId?: string;
    }
  ) => {
    try {
      const normalizedLandmarks = normalizeLandmarkTemplate(args.landmarks);
      if (normalizedLandmarks.length === 0) {
        return { ok: false, error: "Custom schema requires at least one landmark." };
      }
      const now = new Date().toISOString();
      const normalizedName = String(args.name || "").trim();
      if (!normalizedName) {
        return { ok: false, error: "Schema name is required." };
      }
      const library = bootstrapReusableSchemasFromSessions(readReusableSchemaLibrary());
      const template = normalizeReusableSchemaTemplate({
        id: `custom-template-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        name: normalizedName,
        description: String(args.description || "").trim(),
        landmarks: normalizedLandmarks,
        orientationPolicy: args.orientationPolicy,
        sourcePresetId: String(args.sourcePresetId || "").trim() || undefined,
        createdAt: now,
        updatedAt: now,
      });
      if (!template) {
        return { ok: false, error: "Failed to normalize custom schema template." };
      }
      const nextLibrary = {
        version: 1 as const,
        templates: [...library.templates, template],
      };
      writeReusableSchemaLibrary(nextLibrary);
      return { ok: true, template };
    } catch (e: any) {
      console.error("schema:save-custom-template failed:", e);
      return { ok: false, error: e.message };
    }
  }
);

ipcMain.handle(
  "schema:update-custom-template",
  async (
    _event,
    args: {
      templateId: string;
      updates: {
        name: string;
        description?: string;
        landmarks: any[];
        orientationPolicy?: any;
        sourcePresetId?: string;
      };
    }
  ) => {
    try {
      const templateId = String(args.templateId || "").trim();
      if (!templateId) {
        return { ok: false, error: "Template id is required." };
      }
      const normalizedLandmarks = normalizeLandmarkTemplate(args.updates?.landmarks);
      if (normalizedLandmarks.length === 0) {
        return { ok: false, error: "Custom schema requires at least one landmark." };
      }
      const normalizedName = String(args.updates?.name || "").trim();
      if (!normalizedName) {
        return { ok: false, error: "Schema name is required." };
      }
      const library = bootstrapReusableSchemasFromSessions(readReusableSchemaLibrary());
      const templateIndex = library.templates.findIndex((template) => template.id === templateId);
      if (templateIndex < 0) {
        return { ok: false, error: `Custom schema template not found: ${templateId}` };
      }
      const existing = library.templates[templateIndex];
      const updated = normalizeReusableSchemaTemplate({
        ...existing,
        name: normalizedName,
        description: String(args.updates?.description || "").trim(),
        landmarks: normalizedLandmarks,
        orientationPolicy:
          args.updates?.orientationPolicy === undefined
            ? existing.orientationPolicy
            : args.updates.orientationPolicy,
        sourcePresetId: String(args.updates?.sourcePresetId || "").trim() || undefined,
        updatedAt: new Date().toISOString(),
      });
      if (!updated) {
        return { ok: false, error: "Failed to normalize custom schema template." };
      }
      const nextTemplates = [...library.templates];
      nextTemplates[templateIndex] = updated;
      writeReusableSchemaLibrary({ version: 1, templates: nextTemplates });
      return { ok: true, template: updated };
    } catch (e: any) {
      console.error("schema:update-custom-template failed:", e);
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
        const sessionImagesDir = path.join(sessionsDir, entry.name, "images");
        const schemaMetadata = resolveSessionSchemaMetadata(meta);
        sessions.push({
          speciesId: meta.speciesId || entry.name,
          name: meta.name || entry.name,
          imageCount: meta.imageCount || 0,
          lastModified: meta.lastModified || meta.createdAt || "",
          landmarkCount: (meta.landmarkTemplate || []).length,
          schemaFingerprint: schemaMetadata.schemaFingerprint,
          schemaSemanticFingerprint: schemaMetadata.schemaSemanticFingerprint,
          schemaSemanticVersion: schemaMetadata.schemaSemanticVersion,
          schemaKind: schemaMetadata.schemaKind,
          schemaSourceId: schemaMetadata.schemaSourceId,
          orientationPolicy: normalizeOrientationPolicy(meta.orientationPolicy, meta.landmarkTemplate),
          orientationPolicyConfigured: hasConfiguredOrientationPolicy(meta),
          obbTrainingSettings: readNormalizedSessionObbTrainingSettings(meta),
          obbDetectionSettings: readNormalizedSessionObbDetectionSettings(meta),
          obbTrainingSettingsCustomized: readSessionObbTrainingSettingsCustomized(meta),
          obbDetectionSettingsCustomized: readSessionObbDetectionSettingsCustomized(meta),
          representativeImageDimensions: summarizeRepresentativeImageDimensions(
            fs.existsSync(sessionImagesDir)
              ? fs.readdirSync(sessionImagesDir)
                .filter((f) => IMAGE_EXTS.test(f))
                .map((filename) => path.join(sessionImagesDir, filename))
              : []
          ),
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
      clearFinalizedStateForSession(args.speciesId);

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
      confidence?: number;
      class_name?: string;
      orientation_override?: "left" | "right" | "up" | "down" | "uncertain";
      obbCorners?: [number, number][];
      angle?: number;
      class_id?: number;
      orientation_hint?: InferenceDraftSpecimen["box"]["orientation_hint"];
    };
    landmarks: InferenceDraftSpecimen["landmarks"];
    inference_metadata?: InferenceDraftSpecimen["inference_metadata"];
    trainingTargets?: Array<"landmark" | "obb">;
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
  provenance?: {
    commitId?: string;
    inferenceSessionId?: string;
    landmarkModelKey?: string;
    landmarkPredictorType?: "dlib" | "cnn";
    detectionModelKey?: string;
    detectorProvenance?: DetectionModelProvenance;
    originalPredictionHash?: string;
    reviewedPredictionHash?: string;
    reviewOutcome?: "accepted_unchanged" | "corrected" | "rejected_all";
    reviewer?: string;
    reviewedAt?: string;
    repeatedFailureCount?: number;
    prioritySignals?: HitlPrioritySignals;
  };
}): { ok: boolean; savedPath?: string; error?: string; imageName?: string } {
  try {
    const sessionDir = getSessionDir(args.speciesId);
    const imagesDir = path.join(sessionDir, "images");
    const labelsDir = path.join(sessionDir, "labels");
    fs.mkdirSync(imagesDir, { recursive: true });
    fs.mkdirSync(labelsDir, { recursive: true });

    const resolvedImage = resolveContentAddressedHitlImage({
      imagesDir,
      labelsDir,
      sourcePath: args.imagePath,
      requestedFilename: args.filename,
    });
    const imgName = resolvedImage.imageName;
    const sourceSha256 = resolvedImage.sourceSha256;

    // Content addressing resolves same-name collisions before the transaction.
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
            ...(Number.isFinite(Number(s.box.confidence))
              ? { confidence: Number(s.box.confidence) }
              : {}),
            ...(s.box.class_name ? { class_name: String(s.box.class_name) } : {}),
            ...(s.box.orientation_override === "left" ||
            s.box.orientation_override === "right" ||
            s.box.orientation_override === "up" ||
            s.box.orientation_override === "down" ||
            s.box.orientation_override === "uncertain"
              ? { orientation_override: s.box.orientation_override }
              : {}),
            landmarks: (s.landmarks || []).map((lm) => ({
              id: Number(lm.id),
              x: Number(lm.x),
              y: Number(lm.y),
              isSkipped: false,
              ...(Number.isFinite(Number(lm.confidence))
                ? { confidence: Number(lm.confidence) }
                : {}),
              ...(Number.isFinite(Number(lm.heatmap_entropy))
                ? { heatmap_entropy: Number(lm.heatmap_entropy) }
                : {}),
              ...(typeof lm.isPredicted === "boolean" ? { isPredicted: lm.isPredicted } : {}),
              ...(typeof lm.isCorrected === "boolean" ? { isCorrected: lm.isCorrected } : {}),
              ...(lm.predictionVersion ? { predictionVersion: lm.predictionVersion } : {}),
            })),
            trainingTargets: Array.isArray(s.trainingTargets)
              ? s.trainingTargets.filter((target) => target === "landmark" || target === "obb")
              : ((s.landmarks || []).length > 0 ? ["landmark", "obb"] : ["obb"]),
            ...(s.inference_metadata ? { inference_metadata: s.inference_metadata } : {}),
          };
          if (Array.isArray(s.box.obbCorners) && s.box.obbCorners.length === 4) entry.obbCorners = s.box.obbCorners;
          if (s.box.angle != null) entry.angle = s.box.angle;
          if (s.box.class_id != null) entry.class_id = s.box.class_id;
          if (s.box.orientation_hint) entry.orientation_hint = s.box.orientation_hint;
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

    const labelExistedBeforeCommit = fs.existsSync(lblDest);
    let existingLabel: any = {};
    let existingRejectedDetections: any[] = [];
    if (fs.existsSync(lblDest)) {
      try {
        existingLabel = JSON.parse(fs.readFileSync(lblDest, "utf-8"));
        if (Array.isArray(existingLabel?.rejectedDetections)) {
          existingRejectedDetections = existingLabel.rejectedDetections;
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
          b.orientation_override === "up" ||
          b.orientation_override === "down" ||
          b.orientation_override === "uncertain"
            ? b.orientation_override
            : undefined,
        ...(Array.isArray(b.obbCorners) && b.obbCorners.length === 4
          ? { obbCorners: b.obbCorners }
          : {}),
        ...(b.angle != null ? { angle: b.angle } : {}),
        ...(b.class_id != null ? { class_id: b.class_id } : {}),
        landmarks: b.landmarks,
        trainingTargets: b.trainingTargets,
      }))
    );
    const boxSignature = buildAcceptedBoxesSignature(acceptedBoxes);
    const reviewedAt = args.provenance?.reviewedAt || new Date().toISOString();
    const commitId = args.provenance?.commitId || randomUUID();
    const isNewTrainingSample = resolveHitlNewTrainingSample({
      labelExistedBeforeCommit,
      commitId,
      existingLabel,
    });
    const confidenceValues = boxesPayload
      .map((box) => Number(box?.confidence))
      .filter((value) => Number.isFinite(value));
    const reviewEvent = {
      eventId: commitId,
      commitId,
      source: "hitl_review",
      speciesId: args.speciesId,
      inferenceSessionId: args.provenance?.inferenceSessionId,
      imageFilename: imgName,
      sourceImageSha256: sourceSha256,
      landmarkModelKey: args.provenance?.landmarkModelKey,
      landmarkPredictorType: args.provenance?.landmarkPredictorType,
      detectionModelKey: args.provenance?.detectionModelKey,
      detectionModelId: args.provenance?.detectorProvenance?.modelId,
      detectionArtifactSha256:
        args.provenance?.detectorProvenance?.artifactSha256,
      detectionModelName: args.provenance?.detectorProvenance?.displayName,
      detectionModelKind: args.provenance?.detectorProvenance?.kind,
      detectorProvenance: args.provenance?.detectorProvenance,
      originalPredictionHash: args.provenance?.originalPredictionHash,
      reviewedPredictionHash: args.provenance?.reviewedPredictionHash || boxSignature,
      reviewOutcome:
        args.provenance?.reviewOutcome ||
        (boxesPayload.length === 0 ? "rejected_all" : "corrected"),
      isNewTrainingSample,
      reviewer: args.provenance?.reviewer || "local_user",
      reviewedAt,
      acceptedSpecimens: boxesPayload.length,
      rejectedDetections: mergedRejectedDetections.length,
      detectionConfidence: confidenceValues.length > 0
        ? {
            min: Math.min(...confidenceValues),
            max: Math.max(...confidenceValues),
            mean: confidenceValues.reduce((sum, value) => sum + value, 0) / confidenceValues.length,
          }
        : null,
      prioritySignals: {
        ...(args.provenance?.prioritySignals || {}),
        repeatedFailureCount: Number(args.provenance?.repeatedFailureCount) || 0,
      },
      reviewPriority: calculateHitlReviewPriority(boxesPayload, {
        ...(args.provenance?.prioritySignals || {}),
        repeatedFailureCount: Number(args.provenance?.repeatedFailureCount) || 0,
      }),
    };
    const existingReviewHistory = Array.isArray(existingLabel?.reviewHistory)
      ? existingLabel.reviewHistory
      : [];
    const reviewHistory = [
      ...existingReviewHistory.filter((event: any) => event?.eventId !== commitId),
      reviewEvent,
    ];

    const label: any = {
      imageFilename: imgName,
      speciesId: args.speciesId,
      boxes: boxesPayload,
      rejectedDetections: mergedRejectedDetections,
      provenance: reviewEvent,
      reviewHistory,
      finalizedDetection: {
        isFinalized: true,
        finalizedAt: reviewedAt,
        acceptedBoxes,
        boxSignature,
      },
    };
    const reviewEventsPath = path.join(sessionDir, "review_events.json");
    const existingEventsRaw = fs.existsSync(reviewEventsPath)
      ? JSON.parse(fs.readFileSync(reviewEventsPath, "utf-8"))
      : [];
    const existingEvents = Array.isArray(existingEventsRaw) ? existingEventsRaw : [];
    const existingFinalized = readFinalizedList(sessionDir);
    const finalizedByName = new Map<string, string>();
    [...existingFinalized, imgName].forEach((name) => {
      const safeName = path.basename(String(name || "").trim());
      if (safeName) finalizedByName.set(safeName.toLowerCase(), safeName);
    });
    const additionalJsonWrites: Array<{ path: string; payload: unknown }> = [
      {
        path: getFinalizedImagesPath(sessionDir),
        payload: Array.from(finalizedByName.values()),
      },
    ];
    const sessionJsonPath = path.join(sessionDir, "session.json");
    if (fs.existsSync(sessionJsonPath)) {
      const meta = JSON.parse(fs.readFileSync(sessionJsonPath, "utf-8"));
      const existingImageCount = fs.readdirSync(imagesDir).filter((file) => IMAGE_EXTS.test(file)).length;
      const imageWillBeCreated =
        resolvedImage.sourceExists && !fs.existsSync(resolvedImage.imageDestination);
      meta.imageCount = existingImageCount + (imageWillBeCreated ? 1 : 0);
      meta.lastModified = reviewedAt;
      additionalJsonWrites.push({ path: sessionJsonPath, payload: meta });
    }
    commitHitlFileTransaction({
      resolvedImage,
      sourcePath: args.imagePath,
      labelPath: lblDest,
      labelPayload: label,
      reviewEventsPath,
      reviewEventsPayload: [
        ...existingEvents.filter((event: any) => event?.eventId !== commitId),
        reviewEvent,
      ],
      additionalJsonWrites,
    });

    return { ok: true, savedPath: lblDest, imageName: imgName };
  } catch (e: any) {
    return { ok: false, error: String(e?.message || e) };
  }
}

// Open an inference session using the current draft/transaction workflow.
ipcMain.handle("session:list-inference-sessions", async () => {
  try {
    const sessionsRoot = path.join(projectRoot, "sessions");
    if (!fs.existsSync(sessionsRoot)) {
      return { ok: true, sessions: [] };
    }

    const schemas = fs
      .readdirSync(sessionsRoot, { withFileTypes: true })
      .filter((entry) => entry.isDirectory());

    const candidates = schemas
      .map((entry) => {
        const speciesId = entry.name;
        const sessionJsonPath = path.join(sessionsRoot, speciesId, "session.json");
        if (!fs.existsSync(sessionJsonPath)) return null;
        let schemaName = speciesId;
        let schemaImageCount = 0;
        let schemaUpdatedAt = "";
        let schemaGroupKey = `species:${speciesId}`;
        try {
          const meta = JSON.parse(fs.readFileSync(sessionJsonPath, "utf-8"));
          schemaName = String(meta?.name || speciesId);
          schemaImageCount = Math.max(0, Number(meta?.imageCount || 0));
          schemaUpdatedAt = String(meta?.lastModified || meta?.createdAt || "");
          schemaGroupKey = buildInferenceSchemaGroupKey(speciesId, meta);
        } catch {
          // keep defaults
        }
        const resolved = resolveCanonicalInferenceSession(speciesId, { createIfMissing: false });
        return {
          speciesId,
          schemaName,
          schemaImageCount,
          schemaUpdatedAt,
          schemaGroupKey,
          exists: Boolean(resolved.inferenceSessionId && resolved.manifest),
          inferenceSessionId: resolved.inferenceSessionId ?? undefined,
          displayName: resolved.manifest?.displayName,
          createdAt: resolved.manifest?.createdAt,
          updatedAt: resolved.manifest?.updatedAt,
          migratedFrom: resolved.migratedFrom,
        };
      })
      .filter((item): item is NonNullable<typeof item> => item !== null);

    const grouped = new Map<string, Array<(typeof candidates)[number]>>();
    candidates.forEach((candidate) => {
      const existing = grouped.get(candidate.schemaGroupKey);
      if (existing) {
        existing.push(candidate);
      } else {
        grouped.set(candidate.schemaGroupKey, [candidate]);
      }
    });

    const sessions = Array.from(grouped.entries())
      .map(([schemaGroupKey, entries]): InferenceSessionListEntry => {
        const representative = [...entries].sort((a, b) => {
          if (a.exists !== b.exists) return a.exists ? -1 : 1;
          const aTime = parseSortableTimestamp(a.updatedAt, a.schemaUpdatedAt, a.createdAt);
          const bTime = parseSortableTimestamp(b.updatedAt, b.schemaUpdatedAt, b.createdAt);
          if (aTime !== bTime) return bTime - aTime;
          return a.schemaName.localeCompare(b.schemaName);
        })[0];

        return {
          speciesId: representative.speciesId,
          schemaName: representative.schemaName,
          schemaImageCount: Math.max(...entries.map((entry) => entry.schemaImageCount)),
          schemaUpdatedAt: entries
            .slice()
            .sort(
              (a, b) =>
                parseSortableTimestamp(b.schemaUpdatedAt, b.updatedAt, b.createdAt) -
                parseSortableTimestamp(a.schemaUpdatedAt, a.updatedAt, a.createdAt)
            )[0]?.schemaUpdatedAt || representative.schemaUpdatedAt,
          schemaGroupKey,
          canonicalSpeciesId: representative.speciesId,
          ...(entries.length > 1 ? { hiddenSessionCount: entries.length - 1 } : {}),
          exists: representative.exists,
          inferenceSessionId: representative.inferenceSessionId,
          displayName: representative.displayName,
          createdAt: representative.createdAt,
          updatedAt: representative.updatedAt,
          migratedFrom: representative.migratedFrom,
        };
      })
      .sort((a, b) => a.schemaName.localeCompare(b.schemaName));

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
        lastUsedPredictorType?: "dlib" | "cnn";
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
      const imageSourcesRaw = safeReadJson(
        path.join(
          getInferenceSessionDir(args.speciesId, resolved.inferenceSessionId),
          "image_paths.json"
        )
      );
      const persistedImageSources = Array.isArray(imageSourcesRaw?.imagePaths)
        ? imageSourcesRaw.imagePaths.filter(
            (entry: any) =>
              entry &&
              typeof entry.path === "string" &&
              typeof entry.name === "string" &&
              typeof entry.sourceSha256 === "string"
          )
        : [];
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

        let filename = path.basename(String(item.filename || "").trim());
        const imagePathCandidate = item.imagePath
          ? path.resolve(item.imagePath)
          : path.join(getSessionDir(args.speciesId), "images", filename);
        let imagePath: string | null = null;
        const normalizedCandidate = imagePathCandidate.toLowerCase();
        const exactSource = persistedImageSources.find(
          (entry: any) =>
            path.resolve(String(entry.sourcePath || entry.path)).toLowerCase() === normalizedCandidate ||
            path.resolve(String(entry.path)).toLowerCase() === normalizedCandidate
        );
        const filenameSources = !exactSource && filename
          ? persistedImageSources.filter(
              (entry: any) =>
                String(entry.sourceName || "").toLowerCase() === filename.toLowerCase() ||
                String(entry.name || "").toLowerCase() === filename.toLowerCase()
            )
          : [];
        const persistedSource = exactSource || (filenameSources.length === 1 ? filenameSources[0] : null);
        if (persistedSource) {
          const persistedPath = path.resolve(String(persistedSource.path));
          const expectedSha256 = String(persistedSource.sourceSha256).toLowerCase();
          if (
            fs.existsSync(persistedPath) &&
            sha256FileSync(persistedPath).toLowerCase() === expectedSha256
          ) {
            imagePath = persistedPath;
            filename = path.basename(String(persistedSource.name));
          }
        } else if (fs.existsSync(imagePathCandidate)) {
          // Legacy v1 sessions have no stored source digest. They may use the
          // exact original path, but never an unverified same-stem fallback.
          imagePath = imagePathCandidate;
        }
        if (!filename || !imagePath) {
          failed += 1;
          failures.push({
            filename: filename || "(unknown)",
            error: persistedSource
              ? "The staged inference image is missing or its content hash changed; re-add the source image before committing."
              : `Image not found at its original path (${imagePathCandidate}); re-add it so BioVision can verify the source bytes.`,
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
            confidence: s.box.confidence,
            class_name: s.box.class_name,
            orientation_override: s.box.orientation_override,
            ...(s.box.obbCorners ? { obbCorners: s.box.obbCorners } : {}),
            ...(s.box.angle != null ? { angle: s.box.angle } : {}),
            ...(s.box.class_id != null ? { class_id: s.box.class_id } : {}),
            ...(s.box.orientation_hint ? { orientation_hint: s.box.orientation_hint } : {}),
          },
          landmarks: s.landmarks.map((lm) => ({
            ...lm,
          })),
          trainingTargets: s.landmarks.length > 0
            ? ["landmark", "obb"] as Array<"landmark" | "obb">
            : ["obb"] as Array<"landmark" | "obb">,
          ...(s.inference_metadata ? { inference_metadata: s.inference_metadata } : {}),
        }));
        const hasLandmarkAnnotations = commitSpecimens.some(
          (specimen) => specimen.landmarks.length > 0
        );

        const reviewedPredictionHash = createHash("sha256")
          .update(JSON.stringify(commitSpecimens))
          .digest("hex");
        const commitId = `hitl-${createHash("sha256")
          .update(
            `${args.speciesId}:${resolved.inferenceSessionId}:${item.key}:${item.updatedAt}:${reviewedPredictionHash}`
          )
          .digest("hex")
          .slice(0, 24)}`;

        const landmarkSave = persistInferenceCorrectionToSession({
          speciesId: args.speciesId,
          imagePath,
          filename,
          specimens: commitSpecimens,
          allowEmpty: true,
          provenance: {
            commitId,
            inferenceSessionId: resolved.inferenceSessionId,
            landmarkModelKey: hasLandmarkAnnotations
              ? item.landmarkModelKey || resolved.manifest.models.landmark.key
              : undefined,
            landmarkPredictorType:
              hasLandmarkAnnotations
                ? item.landmarkPredictorType || resolved.manifest.models.landmark.predictorType
                : undefined,
            detectionModelKey:
              item.detectorProvenance?.modelId ||
              resolved.manifest.models.detection.key,
            detectorProvenance: item.detectorProvenance,
            originalPredictionHash: item.inferenceSignature || item.boxSignature,
            reviewedPredictionHash,
            reviewOutcome:
              commitSpecimens.length === 0
                ? "rejected_all"
                : item.wasEdited || item.edited
                ? "corrected"
                : "accepted_unchanged",
            reviewedAt: now,
            repeatedFailureCount: item.commitFailureCount,
            prioritySignals: item.prioritySignals,
          },
        });
        if (!landmarkSave.ok) {
          item.commitFailureCount = Math.max(0, Number(item.commitFailureCount) || 0) + 1;
          item.prioritySignals = {
            ...(item.prioritySignals || {}),
            repeatedFailureCount: item.commitFailureCount,
          };
          item.reviewPriority = calculateHitlReviewPriority(
            item.specimens,
            item.prioritySignals
          );
          item.updatedAt = new Date().toISOString();
          failed += 1;
          failures.push({
            filename,
            error: landmarkSave.error || "Failed to save landmark corrections.",
          });
          continue;
        }

        item.saved = true;
        item.edited = false;
        if (landmarkSave.imageName) {
          item.filename = landmarkSave.imageName;
          item.imagePath = path.join(
            getSessionDir(args.speciesId),
            "images",
            landmarkSave.imageName
          );
        }
        item.committedAt = now;
        item.reviewPriority = calculateHitlReviewPriority(
          item.specimens,
          {
            ...(item.prioritySignals || {}),
            repeatedFailureCount: item.commitFailureCount,
          }
        );
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
      wasEdited?: boolean;
      saved?: boolean;
      reviewComplete?: boolean;
      committedAt?: string | null;
      landmarkModelKey?: string | null;
      landmarkPredictorType?: "dlib" | "cnn" | null;
      boxSignature?: string | null;
      inferenceSignature?: string | null;
      detectorProvenance?: DetectionModelProvenance | null;
      prioritySignals?: HitlPrioritySignals;
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
        const previous = drafts.items[key];
        const specimens = sanitizeDraftSpecimens(args.specimens);
        const commitFailureCount = Math.max(0, Number(previous?.commitFailureCount) || 0);
        const reviewStatus = resolveReviewStatus({
          previousReviewComplete: previous?.reviewComplete,
          previousCommittedAt: previous?.committedAt,
          edited: args.edited,
          requestedReviewComplete: args.reviewComplete,
          requestedCommittedAt: args.committedAt,
        });
        const prioritySignals: HitlPrioritySignals = {
          ...(previous?.prioritySignals || {}),
          ...(args.prioritySignals || {}),
          repeatedFailureCount:
            args.prioritySignals?.repeatedFailureCount ?? commitFailureCount,
        };
        drafts.items[key] = {
          key,
          imagePath: args.imagePath || "",
          filename,
          specimens,
          edited: Boolean(args.edited),
          wasEdited: resolveReviewWasEdited(
            previous?.wasEdited,
            args.edited,
            args.wasEdited
          ),
          saved: Boolean(args.saved),
          reviewComplete: reviewStatus.reviewComplete,
          committedAt: reviewStatus.committedAt,
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
          detectorProvenance:
            args.detectorProvenance === null
              ? undefined
              : args.detectorProvenance ?? drafts.items[key]?.detectorProvenance,
          prioritySignals,
          reviewPriority: calculateHitlReviewPriority(specimens, prioritySignals),
          commitFailureCount,
          updatedAt: new Date().toISOString(),
        };
      }
      drafts.updatedAt = new Date().toISOString();
      writeInferenceReviewDrafts(args.speciesId, drafts, args.inferenceSessionId);
      return {
        ok: true,
        reviewPriority: args.clear ? undefined : drafts.items[key]?.reviewPriority,
      };
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
        anteriorAnchorIds?: number[];
        posteriorAnchorIds?: number[];
        bilateralPairs?: [number, number][];
        bilateralClassAxis?: "vertical_obb";
        obbLevelingMode?: "on" | "off";
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
      const mode = String(args.orientationPolicy?.mode || "").trim().toLowerCase();
      if (!ORIENTATION_MODES.has(mode)) {
        return { ok: false, error: "A valid explicit orientation mode is required." };
      }
      meta.orientationPolicy = normalizeOrientationPolicy(
        args.orientationPolicy,
        meta.landmarkTemplate
      );
      meta.orientationPolicyConfigured = true;
      meta.orientationPolicyConfiguredAt = new Date().toISOString();
      meta.schemaSemanticFingerprint = computeSessionSchemaSemanticFingerprint(
        meta.landmarkTemplate,
        meta.orientationPolicy
      );
      meta.schemaSemanticVersion = SCHEMA_SEMANTIC_VERSION;
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
  "session:update-obb-detector-settings",
  async (
    _event,
    args: {
      speciesId: string;
      obbTrainingSettings?: Record<string, unknown>;
      obbDetectionSettings?: Record<string, unknown>;
      obbTrainingSettingsCustomized?: boolean;
      obbDetectionSettingsCustomized?: boolean;
    }
  ) => {
    try {
      const sessionDir = getSessionDir(args.speciesId);
      const sessionJsonPath = path.join(sessionDir, "session.json");
      if (!fs.existsSync(sessionJsonPath)) {
        return { ok: false, error: `Session not found: ${args.speciesId}` };
      }
      const meta = JSON.parse(fs.readFileSync(sessionJsonPath, "utf-8"));
      if (args.obbTrainingSettings !== undefined) {
        meta.obbTrainingSettings = normalizeObbTrainingSettings(args.obbTrainingSettings);
        meta.obbTrainingSettingsCustomized =
          typeof args.obbTrainingSettingsCustomized === "boolean"
            ? args.obbTrainingSettingsCustomized
            : true;
      } else if (typeof args.obbTrainingSettingsCustomized === "boolean") {
        meta.obbTrainingSettingsCustomized = args.obbTrainingSettingsCustomized;
      }
      if (args.obbDetectionSettings !== undefined) {
        meta.obbDetectionSettings = normalizeObbDetectionSettings(args.obbDetectionSettings);
        meta.obbDetectionSettingsCustomized =
          typeof args.obbDetectionSettingsCustomized === "boolean"
            ? args.obbDetectionSettingsCustomized
            : true;
      } else if (typeof args.obbDetectionSettingsCustomized === "boolean") {
        meta.obbDetectionSettingsCustomized = args.obbDetectionSettingsCustomized;
      }
      meta.lastModified = new Date().toISOString();
      fs.writeFileSync(sessionJsonPath, JSON.stringify(meta, null, 2));
      return { ok: true };
    } catch (e: any) {
      console.error("session:update-obb-detector-settings failed:", e);
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
      const sortedDrafts = Object.values(drafts.items)
        .map((item) => ({
          ...item,
          reviewPriority:
            item.reviewPriority ||
            calculateHitlReviewPriority(item.specimens, item.prioritySignals),
        }))
        .sort(
          (left, right) =>
            Number(right.reviewPriority?.score || 0) -
              Number(left.reviewPriority?.score || 0) ||
            String(left.filename || "").localeCompare(String(right.filename || ""))
        );
      return { ok: true, drafts: sortedDrafts };
    } catch (e: any) {
      console.error("session:load-inference-review-drafts failed:", e);
      return { ok: false, error: e.message, drafts: [] };
    }
  }
);

ipcMain.handle(
  "session:get-retraining-batch",
  async (
    _event,
    args: { speciesId: string; inferenceSessionId?: string; since?: string }
  ) => {
    try {
      if (!args.speciesId) return { ok: false, error: "speciesId is required." };
      const sessionDir = getSessionDir(args.speciesId);
      const registry = readLandmarkModelRegistry(sessionDir);
      const activeModel = [...registry.models]
        .filter((model) => model.status === "active")
        .sort(
          (left, right) =>
            Date.parse(String(right.createdAt || "")) -
            Date.parse(String(left.createdAt || ""))
        )[0];
      const since = String(args.since || activeModel?.createdAt || "").trim() || undefined;
      const sinceMs = since ? Date.parse(since) : Number.NaN;
      const eventsPath = path.join(sessionDir, "review_events.json");
      const rawEvents = fs.existsSync(eventsPath)
        ? JSON.parse(fs.readFileSync(eventsPath, "utf-8"))
        : [];
      const events = (Array.isArray(rawEvents) ? rawEvents : []).filter((event: any) => {
        if (!Number.isFinite(sinceMs)) return true;
        const reviewedAtMs = Date.parse(String(event?.reviewedAt || ""));
        return Number.isFinite(reviewedAtMs) && reviewedAtMs > sinceMs;
      });
      const resolvedSession = args.inferenceSessionId
        ? { inferenceSessionId: args.inferenceSessionId }
        : resolveCanonicalInferenceSession(args.speciesId, { createIfMissing: false });
      const drafts = readInferenceReviewDrafts(
        args.speciesId,
        resolvedSession.inferenceSessionId || undefined
      );
      const pendingItems = Object.values(drafts.items)
        .filter((item) => {
          const updatedAtMs = Date.parse(String(item.updatedAt || ""));
          const committedAtMs = Date.parse(String(item.committedAt || ""));
          return !item.committedAt || !Number.isFinite(committedAtMs) || committedAtMs < updatedAtMs;
        })
        .map((item) => ({
          key: item.key,
          filename: item.filename,
          reviewComplete: Boolean(item.reviewComplete),
          edited: Boolean(item.edited),
          priority:
            item.reviewPriority ||
            calculateHitlReviewPriority(item.specimens, item.prioritySignals),
          updatedAt: item.updatedAt,
        }))
        .sort(
          (left, right) =>
            Number(right.priority.score || 0) - Number(left.priority.score || 0)
        );
      return {
        ok: true,
        since: since || null,
        baselineModelId: activeModel?.modelId || null,
        summary: summarizeRetrainingReviewEvents(events),
        pendingReview: pendingItems.filter((item) => !item.reviewComplete).length,
        pendingCommit: pendingItems.filter((item) => item.reviewComplete).length,
        pendingItems,
        events,
      };
    } catch (e: any) {
      console.error("session:get-retraining-batch failed:", e);
      return { ok: false, error: e.message };
    }
  }
);

ipcMain.handle(
  "session:save-inference-image-paths",
  async (_event, args: {
    speciesId: string;
    inferenceSessionId: string;
    imagePaths: Array<{
      path: string;
      name: string;
      sourcePath?: string;
      sourceName?: string;
      sourceSha256?: string;
      contentId?: string;
    }>;
  }) => {
    try {
      if (!args.speciesId || !args.inferenceSessionId) return { ok: false, error: "speciesId and inferenceSessionId are required." };
      const sessionDir = getInferenceSessionDir(args.speciesId, args.inferenceSessionId);
      fs.mkdirSync(sessionDir, { recursive: true });
      const staged = stageInferenceImagePaths({
        sourceImagesDir: path.join(sessionDir, "source_images"),
        imagePaths: args.imagePaths || [],
      });
      const stagedImagePaths = staged.map((entry, index) => {
        const requested = args.imagePaths[index] || {};
        const requestedSha = String(requested.sourceSha256 || requested.contentId || "")
          .trim()
          .toLowerCase();
        if (requestedSha && requestedSha !== entry.sourceSha256.toLowerCase()) {
          throw new Error(
            `Inference source content changed while staging: ${requested.sourceName || requested.name || entry.name}`
          );
        }
        return {
          ...entry,
          // Preserve the original stable source identity when an already-staged
          // image list is saved again; the mutable staged path is not provenance.
          sourcePath: String(requested.sourcePath || entry.sourcePath),
          sourceName: String(requested.sourceName || entry.sourceName),
          sourceSha256: entry.sourceSha256,
          contentId: entry.sourceSha256,
        };
      });
      atomicWriteJsonSync(
        path.join(sessionDir, "image_paths.json"),
        { version: 2, imagePaths: stagedImagePaths }
      );
      return {
        ok: true,
        imagePaths: stagedImagePaths,
      };
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
        imagePaths: {
          path: string;
          name: string;
          sourcePath?: string;
          sourceName?: string;
          sourceSha256?: string;
        }[];
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
          return {
            ...p,
            contentId: p.sourceSha256,
            data,
            mimeType,
          };
        });
      return { ok: true, images };
    } catch (e: any) {
      console.error("session:load-inference-image-paths failed:", e);
      return { ok: true, images: [] };
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
      clearFinalizedStateForImage(args.speciesId, args.filename, imagePath);

      if (fs.existsSync(imagePath)) fs.unlinkSync(imagePath);
      if (fs.existsSync(labelPath)) fs.unlinkSync(labelPath);

      // Remove persisted inference-review drafts for this image (legacy root + canonical session).
      const lowerName = (args.filename || "").toLowerCase();
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
  private generation = 0;
  private activeGeneration = 0;
  private restartingAfterTimeout = false;
  private pending: Map<
    string,
    {
      resolve: (v: any) => void;
      reject: (e: Error) => void;
      cmdName: string;
      startedAt: number;
      timeoutMs: number;
      timeout: ReturnType<typeof setTimeout>;
      generation: number;
    }
  > = new Map();
  private idleTimer: ReturnType<typeof setTimeout> | null = null;
  private requestId = 0;
  /** True after a successful `init` command Ã¢â‚¬â€ models are loaded and ready. */
  initCompleted = false;
  private readonly IDLE_TIMEOUT_MS = 10 * 60 * 1000; // 10 minutes
  private stderrTail: string[] = [];
  private readonly MAX_STDERR_TAIL_LINES = 60;
  private activeInitRequestIds: Set<string> = new Set();
  private backendSignature: string | null = null;

  private getBackendSignature(): string {
    const files = app.isPackaged
      ? [
          path.join(process.resourcesPath, "python", `biovision_backend${process.platform === "win32" ? ".exe" : ""}`),
        ]
      : [
          path.join(__dirname, "../backend/annotation/super_annotator.py"),
          path.join(__dirname, "../backend/data/export_yolo_dataset.py"),
        ];
    return files
      .map((filePath) => {
        try {
          const stat = fs.statSync(filePath);
          return `${filePath}:${stat.mtimeMs}:${stat.size}`;
        } catch {
          return `${filePath}:missing`;
        }
      })
      .join("|");
  }

  async start(): Promise<void> {
    if (this.process) return; // already running

    const resolved = resolveBundledScript("super_annotator");
    this.backendSignature = this.getBackendSignature();
    const generation = ++this.generation;
    this.activeGeneration = generation;

    this.process = spawn(resolved.cmd, resolved.args, {
      stdio: ["pipe", "pipe", "pipe"],
      cwd: app.isPackaged ? process.resourcesPath : path.join(__dirname, ".."),
      env: { ...process.env, PYTHONUTF8: "1", PYTHONIOENCODING: "utf-8" },
    });

    this.process.stderr?.on("data", (d: Buffer) => {
      if (generation !== this.activeGeneration) return;
      const text = d.toString();
      this.pushStderr(text);
      const progressPrefix = "__BV_OBB_PROGRESS__";
      const lines = text.split(/\r?\n/);
      const passthroughLines: string[] = [];
      for (const rawLine of lines) {
        const line = rawLine.trim();
        if (!line) continue;
        if (line.startsWith(progressPrefix)) {
          const payload = line.slice(progressPrefix.length);
          try {
            const msg = JSON.parse(payload);
            const requestId = msg?._request_id ? String(msg._request_id) : null;
            if (requestId) {
              this.refreshRequestTimeout(requestId);
            }
            this.resetIdleTimer();
            mainWindow?.webContents.send("ml:obb-train-progress", msg);
          } catch (err) {
            passthroughLines.push(line);
          }
        } else {
          passthroughLines.push(line);
        }
      }
      if (passthroughLines.length > 0) {
        console.log("[SuperAnnotator]", passthroughLines.join("\n"));
      }
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
      if (generation !== this.activeGeneration) return;
      try {
        const msg = JSON.parse(line);

        // Forward progress events to renderer
        if (msg.status === "progress") {
          const requestId = msg._request_id ? String(msg._request_id) : null;
          const pendingEntry = requestId ? this.pending.get(requestId) : null;
          if (requestId) {
            this.refreshRequestTimeout(requestId);
          }
          this.resetIdleTimer();
          if (pendingEntry?.cmdName === "train_yolo_obb") {
            mainWindow?.webContents.send("ml:obb-train-progress", msg);
          } else {
            mainWindow?.webContents.send("ml:super-annotate-progress", msg);
          }
          return;
        }

        // Match response to pending request by explicit request id.
        if (msg._request_id) {
          const requestId = String(msg._request_id);
          const entry = this.pending.get(requestId);
          if (!entry) {
            console.debug(`[SuperAnnotator] Ignored response for stale or cleared request ${requestId}.`);
            return;
          }
          if (entry.generation !== generation) {
            clearTimeout(entry.timeout);
            this.pending.delete(requestId);
            this.activeInitRequestIds.delete(requestId);
            return;
          }
          clearTimeout(entry.timeout);
          this.pending.delete(requestId);
          this.activeInitRequestIds.delete(requestId);
          entry.resolve(msg);
          this.resetIdleTimer();
          return;
        }

        // Backward-compatible fallback for responses without _request_id.
        if (this.pending.size === 1) {
          const [firstId, handler] = this.pending.entries().next().value!;
          clearTimeout(handler.timeout);
          this.pending.delete(firstId);
          this.activeInitRequestIds.delete(firstId);
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
      if (generation !== this.activeGeneration) return;
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
      this.activeInitRequestIds.clear();
      this.process = null;
      this.rl = null;
      this.initCompleted = false; // models must be reloaded on next start
      this.restartingAfterTimeout = false;
    });

    this.process.on("error", (err: Error) => {
      console.error("[SuperAnnotator] Child process error:", err);
    });

    this.resetIdleTimer();
  }

  async send(cmd: Record<string, unknown>): Promise<any> {
    const currentSignature = this.getBackendSignature();
    if (this.process && this.backendSignature && currentSignature !== this.backendSignature) {
      console.log("[SuperAnnotator] Backend scripts changed on disk, restarting process");
      await this.stop();
    }
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
        this.activeInitRequestIds.delete(id);
        const elapsedSec = Math.round((Date.now() - entry.startedAt) / 1000);
        entry.reject(
          new Error(
            `SuperAnnotator request timed out after ${Math.round(timeoutMs / 1000)}s (cmd="${cmdName}").`
          )
        );

        // Don't restart if a long-running command is still active
        const hasLongRunning = [...this.pending.values()].some(
          (p) => this.LONG_RUNNING_CMDS.has(p.cmdName)
        );
        if (hasLongRunning && !this.LONG_RUNNING_CMDS.has(cmdName)) {
          console.log(
            `[SuperAnnotator] Ignoring timeout restart for "${cmdName}" — long-running command still active.`
          );
          this.resetIdleTimer();
          return;
        }

        void this.restartAfterTimeout(id, cmdName, elapsedSec);
        this.resetIdleTimer();
      }, timeoutMs);

      this.pending.set(id, {
        resolve,
        reject,
        cmdName,
        startedAt: Date.now(),
        timeoutMs,
        timeout,
        generation: this.activeGeneration,
      });
      if (cmdName === "init") {
        this.activeInitRequestIds.add(id);
      }

      try {
        if (!this.process?.stdin || this.process.stdin.destroyed) {
          throw new Error("stdin is closed or destroyed");
        }
        this.process.stdin.write(JSON.stringify(payload) + "\n");
      } catch (err: any) {
        clearTimeout(timeout);
        this.pending.delete(id);
        this.activeInitRequestIds.delete(id);
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

  private async restartAfterTimeout(requestId: string, cmdName: string, elapsedSec: number): Promise<void> {
    if (this.restartingAfterTimeout) return;
    this.restartingAfterTimeout = true;
    const pendingEntries = [...this.pending.entries()];
    console.warn(
      `[SuperAnnotator] Request ${requestId} timed out after ${elapsedSec}s (cmd="${cmdName}"); restarting backend and clearing ${pendingEntries.length} pending request(s).`
    );
    for (const [pendingId, handler] of pendingEntries) {
      clearTimeout(handler.timeout);
      handler.reject(
        new Error(
          `SuperAnnotator backend restarted after timeout while handling "${handler.cmdName}".`
        )
      );
      this.pending.delete(pendingId);
      this.activeInitRequestIds.delete(pendingId);
    }
    this.activeGeneration += 1;
    try {
      await this.stop();
    } finally {
      this.restartingAfterTimeout = false;
    }
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
    if (cmdName === "train_yolo_obb") return 30 * 60 * 1000;
    if (cmdName === "init" || cmdName === "check") return 2 * 60 * 1000;
    return 5 * 60 * 1000;
  }

  private readonly LONG_RUNNING_CMDS = new Set(["train_yolo_obb", "annotate"]);

  private refreshRequestTimeout(requestId: string): void {
    const entry = this.pending.get(requestId);
    if (!entry) return;
    clearTimeout(entry.timeout);
    entry.timeout = setTimeout(() => {
      const active = this.pending.get(requestId);
      if (!active) return;
      this.pending.delete(requestId);
      this.activeInitRequestIds.delete(requestId);
      const elapsedSec = Math.round((Date.now() - active.startedAt) / 1000);
      active.reject(
        new Error(
          `SuperAnnotator request timed out after ${Math.round(active.timeoutMs / 1000)}s (cmd="${active.cmdName}").`
        )
      );

      // If a long-running command is still active, don't restart the backend
      // just because a short-lived command (like "check") timed out.
      const hasLongRunning = [...this.pending.values()].some(
        (p) => this.LONG_RUNNING_CMDS.has(p.cmdName)
      );
      if (hasLongRunning && !this.LONG_RUNNING_CMDS.has(active.cmdName)) {
        console.log(
          `[SuperAnnotator] Ignoring timeout restart for "${active.cmdName}" — long-running command still active.`
        );
        this.resetIdleTimer();
        return;
      }

      void this.restartAfterTimeout(requestId, active.cmdName, elapsedSec);
      this.resetIdleTimer();
    }, entry.timeoutMs);
  }

  get isRunning(): boolean {
    return this.process !== null;
  }

  get isInitializing(): boolean {
    return this.activeInitRequestIds.size > 0;
  }
}

const superAnnotator = new SuperAnnotatorProcess();

type SuperAnnotatorRuntimeState =
  | "not_started"
  | "checking"
  | "not_initialized"
  | "initializing"
  | "ready"
  | "failed";

type CapabilityStatusSource = "local_estimate" | "python_probe" | "python_check";

function getSuperAnnotatorRuntimeState(result?: {
  yolo_ready?: boolean;
  sam2_ready?: boolean;
  yolo_failed?: boolean;
  sam2_failed?: boolean;
}): SuperAnnotatorRuntimeState {
  if (result?.yolo_failed || result?.sam2_failed) {
    return "failed";
  }
  if (superAnnotator.isInitializing) {
    return "initializing";
  }
  if (superAnnotator.initCompleted || result?.yolo_ready || result?.sam2_ready) {
    return "ready";
  }
  if (superAnnotator.isRunning) {
    return "not_initialized";
  }
  return "not_started";
}

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
    const hwOut = await runBundledScript("hardware_probe");
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
  resolveFinalizedState(sessionDir, { reconcile: true });
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

function finalizedAcceptedBoxesHaveObb(boxes: any[]): boolean {
  return Array.isArray(boxes) && boxes.some((box: any) =>
    Array.isArray(box?.obbCorners) &&
    box.obbCorners.length === 4 &&
    Number.isFinite(Number(box?.angle)) &&
    Number.isFinite(Number(box?.class_id))
  );
}

function normalizeFinalizedAcceptedBoxesForSession(boxes: any[]): any[] {
  if (!Array.isArray(boxes)) return [];
  const normalized = normalizeFinalizedAcceptedBoxes(boxes as any);
  return normalizeSessionStoredBoxes(
    normalized.map((box, index) => ({
      id: index,
      left: box.left,
      top: box.top,
      width: box.width,
      height: box.height,
      ...(box.orientation_override ? { orientation_override: box.orientation_override } : {}),
      ...(box.orientation_hint ? { orientation_hint: box.orientation_hint } : {}),
      ...(Array.isArray(box.obbCorners) && box.obbCorners.length === 4 ? { obbCorners: box.obbCorners } : {}),
      ...(box.angle != null ? { angle: box.angle } : {}),
      ...(box.class_id != null ? { class_id: box.class_id } : {}),
      ...(Array.isArray(box.landmarks) ? { landmarks: box.landmarks } : {}),
    }))
  );
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
    const resolved = resolveBundledScript("predict_worker");
    this.process = spawn(resolved.cmd, resolved.args, {
      stdio: ["pipe", "pipe", "pipe"],
      cwd: app.isPackaged ? process.resourcesPath : path.join(__dirname, ".."),
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

ipcMain.handle("ml:check-super-annotator", async () => {
  const pythonResolution = getPythonResolution();
  warnIfUsingSystemPython(pythonResolution);
  try {
    const result = await superAnnotator.send({ cmd: "check" });
    return {
      ...result,
      mode: result?.mode ?? "unknown",
      runtimeState: getSuperAnnotatorRuntimeState(result),
      statusSource: "python_check" as CapabilityStatusSource,
      pythonPath: pythonResolution.pythonPath,
      usingRepoVenv: pythonResolution.usingRepoVenv,
    };
  } catch (e: any) {
    return {
      available: true,
      mode: "unknown",
      gpu: false,
      yolo_ready: false,
      sam2_ready: false,
      yolo_failed: false,
      sam2_failed: false,
      runtimeState: "failed" as SuperAnnotatorRuntimeState,
      statusSource: "local_estimate" as CapabilityStatusSource,
      pythonPath: pythonResolution.pythonPath,
      usingRepoVenv: pythonResolution.usingRepoVenv,
      error: e.message,
    };
  }
});

ipcMain.handle("ml:train-obb-detector", async (_event, speciesId: string, options?: {
  epochs?: number;
  batch?: number;
  modelTier?: ObbModelTier;
  imgsz?: ObbImageSize;
  iou?: number;
  cls?: number;
  box?: number;
  samEnabled?: boolean;
}) => {
  try {
    const sessionDir = getSessionDir(speciesId);
    if (!fs.existsSync(sessionDir)) {
      return { ok: false, error: `Session directory not found: ${sessionDir}` };
    }
    const sessionJsonPath = path.join(sessionDir, "session.json");
    const sessionMeta = safeReadJson(sessionJsonPath) ?? {};
    const persistedTrainingSettings = readNormalizedSessionObbTrainingSettings(sessionMeta);
    let explicitTrainingContract: ReturnType<
      typeof persistExplicitSessionTrainingContract
    >;
    try {
      explicitTrainingContract = persistExplicitSessionTrainingContract(sessionDir);
    } catch (error: any) {
      return {
        ok: false,
        error:
          error?.message ||
          "Choose and save an explicit session orientation policy before OBB training.",
      };
    }

    // Get hardware tier; user-provided modelTier takes precedence
    const caps = await superAnnotator.send({ cmd: "check" });
    const modelTier =
      options?.modelTier ??
      persistedTrainingSettings.modelTier ??
      caps?.obb_model_tier ??
      "nano";

    // Probe hardware to determine device for routing.
    let hwDevice = "cpu";
    try {
      const hwOut = await runBundledScript("hardware_probe");
      const hw = JSON.parse(hwOut.trim());
      hwDevice = hw.device ?? "cpu";
    } catch (_hwErr) {
      console.warn("Hardware probe failed during OBB training Ã¢â‚¬â€ defaulting to cpu");
    }

    // The schema passed to Python comes only from a user-confirmed session
    // policy. Never normalize a missing policy and mark it as configured.
    const orientationSchema = explicitTrainingContract.orientationMode;

    const samEnabled = options?.samEnabled === true;
    const resolvedTrainingSettings = normalizeObbTrainingSettings({
      ...persistedTrainingSettings,
      ...options,
      modelTier,
    });
    const result = await superAnnotator.send({
      cmd: "train_yolo_obb",
      session_dir: sessionDir,
      epochs: resolvedTrainingSettings.epochs,
      batch: resolvedTrainingSettings.batch,
      imgsz: resolvedTrainingSettings.imgsz,
      model_tier: modelTier,
      device: hwDevice,
      sam2_enabled: samEnabled,
      iou_loss: resolvedTrainingSettings.iou,
      cls_loss: resolvedTrainingSettings.cls,
      box_loss: resolvedTrainingSettings.box,
      orientation_schema: orientationSchema,
      seed: 42,
    });

    if (result?.status === "error") {
      return { ok: false, error: result.error ?? "OBB detector training failed" };
    }

    const trainedObbResolution = resolveObbDetectorForEffectiveRoot(sessionDir);
    const verifiedObbReady = isVerifiedTrainedObbDetector(trainedObbResolution);
    if (result?.model_status === "active" && !verifiedObbReady) {
      return {
        ok: false,
        error:
          "OBB training produced an active artifact, but immutable artifact/schema verification failed. " +
          (trainedObbResolution?.issues[0]?.message || "Landmark training remains locked."),
      };
    }

    // Persist only verified readiness; alias existence alone must never unlock
    // landmark predictor training.
    if (fs.existsSync(sessionJsonPath)) {
      try {
        const session = safeReadJson(sessionJsonPath) ?? {};
        (session as any).obbDetectorReady = verifiedObbReady;
        (session as any).obbTrainingSettings = resolvedTrainingSettings;
        atomicWriteJsonSync(sessionJsonPath, session);
      } catch (_e) {
        // non-fatal
      }
    }

    return {
      ok: true,
      modelPath: result?.model_path,
      map50: result?.map50 ?? null,
      map50_95: result?.map50_95 ?? null,
      precision: result?.precision ?? null,
      recall: result?.recall ?? null,
      perClass: Array.isArray(result?.per_class) ? result.per_class : [],
      modelId: result?.model_id ?? null,
      modelStatus: result?.model_status ?? null,
      obbDetectorReady: verifiedObbReady,
      promotion: result?.promotion ?? null,
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
        detectionPreset?: ObbDetectionPreset;
        conf?: number;
        nmsIou?: number;
        imgsz?: ObbImageSize;
        allowIncompatible?: boolean;
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
      const sessionMeta = safeReadJson(sessionJsonPath) ?? {};
      const persistedDetectionSettings = readNormalizedSessionObbDetectionSettings(sessionMeta);
      const resolvedDetectionSettings = normalizeObbDetectionSettings({
        ...persistedDetectionSettings,
        detectionPreset: args.options?.detectionPreset,
        conf: args.options?.conf,
        nmsIou: args.options?.nmsIou,
        maxObjects: args.options?.maxObjects,
        imgsz: args.options?.imgsz,
      });

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
      const detector = finetunedModel
        ? resolveTrainedObbDetector({
            aliasPath: finetunedModel,
            registryPath: path.join(effectiveRoot, "models", "obb_registry.json"),
            sessionSemanticFingerprint:
              computeSessionSchemaSemanticFingerprint(
                sessionMeta?.landmarkTemplate,
                sessionMeta?.orientationPolicy
              ),
            sessionSemanticVersion: SCHEMA_SEMANTIC_VERSION,
            sessionOrientationContract: sessionMeta?.orientationPolicy,
          })
        : undefined;
      const compatibility = detector
        ? {
            compatible: detector.compatible,
            blocking: detector.blocking,
            requiresOverride: detector.blocking,
            issues: detector.issues,
          }
        : undefined;
      if (detector?.blocking && args.options?.allowIncompatible !== true) {
        return {
          ok: false,
          objects: [],
          requiresOverride: true,
          compatibility,
          detectorProvenance: detector.provenance,
          error:
            `OBB detector compatibility check blocked auto-detection. ${formatCompatibilityErrorSummary(
              detector.issues
            )}`,
        };
      }

      const samEnabled = args.options?.samEnabled ?? true;
      const thresholdPlan = resolveObbInferenceThresholdPlan({
        hasTrainedArtifact: Boolean(finetunedModel),
        sessionSettingsCustomized: readSessionObbDetectionSettingsCustomized(sessionMeta),
        detectionPreset: resolvedDetectionSettings.detectionPreset,
        conf: resolvedDetectionSettings.conf,
        nmsIou: resolvedDetectionSettings.nmsIou,
      });
      const result = await superAnnotator.send({
        cmd: "annotate",
        image_path: args.imagePath,
        class_name: args.className,
        dlib_model: dlibModel,
        id_mapping_path: idMappingPath,
        options: {
          ...(thresholdPlan.conf !== undefined
            ? { conf_threshold: thresholdPlan.conf }
            : {}),
          ...(thresholdPlan.nmsIou !== undefined
            ? { nms_iou: thresholdPlan.nmsIou }
            : {}),
          sam_enabled: samEnabled,
          max_objects: resolvedDetectionSettings.maxObjects,
          imgsz: resolvedDetectionSettings.imgsz,
          detection_mode: args.options?.detectionMode ?? "auto",
          detection_preset: thresholdPlan.detectionPreset,
          finetuned_model: finetunedModel,
          // Pass session_dir so SAM2 segments are auto-saved for synthetic augmentation
          session_dir: samEnabled ? effectiveRoot : undefined,
          orientation_policy: sessionOrientationPolicy,
        },
      });

      if (result?.status === "error") {
        return { ok: false, error: result.error ?? "SuperAnnotator annotate failed", objects: [] };
      }

      const detectionMethod = result?.detection_method ?? (detector ? "yolo_obb" : "yolo_world");
      return {
        ok: true,
        ...result,
        objects: Array.isArray(result?.objects) ? result.objects : [],
        detectorProvenance:
          detector?.provenance ??
          resolveZeroShotDetector(
            path.join(__dirname, "..", "yolov8s-worldv2.pt"),
            detectionMethod
          ),
        ...(compatibility ? { compatibility } : {}),
      };
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
