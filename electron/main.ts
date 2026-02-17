import { app, BrowserWindow, ipcMain, dialog, nativeImage } from "electron";
import fs from "fs";
import * as path from "path";
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

app.on("ready", createWindow);

function getPythonPath(): string {
  // Try to use the venv Python if it exists
  const venvPython = path.join(__dirname, "..", "venv", "bin", "python");
  if (fs.existsSync(venvPython)) {
    return venvPython;
  }
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
  customOptions?: Record<string, number>;  // Custom training parameters
  speciesId?: string;  // Session-scoped training
  useImportedXml?: boolean; // Train directly from existing xml/train_{tag}.xml
}

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
        const xmlValidator = path.join(__dirname, "../backend/validate_dlib_xml.py");
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
  try {
    if (!MODEL_NAME_RE.test(modelName)) {
      throw new Error("Invalid model name. Use letters, numbers, dot, underscore, or hyphen.");
    }

    const testSplit = options?.testSplit ?? 0.2;
    const seed = options?.seed ?? 42;
    const effectiveRoot = getEffectiveRoot(options?.speciesId);
    ensureTrainingLayout(effectiveRoot);

    if (options?.useImportedXml) {
      const xmlValidator = path.join(__dirname, "../backend/validate_dlib_xml.py");
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
    } else {
      // Prepare dataset with train/test split
      await runPython([
        path.join(__dirname, "../backend/prepare_dataset.py"),
        effectiveRoot,
        modelName,
        testSplit.toString(),
        seed.toString(),
      ]);
    }

    // Train with optional custom parameters
    const trainArgs = [
      path.join(__dirname, "../backend/train_shape_model.py"),
      effectiveRoot,
      modelName,
    ];
    if (options?.customOptions) {
      trainArgs.push(JSON.stringify(options.customOptions));
    }

    const out = await runPython(trainArgs);

    // Parse output for train/test errors
    const trainErrorMatch = out.match(/TRAIN_ERROR\s+([\d.]+)/);
    const testErrorMatch = out.match(/TEST_ERROR\s+([\d.]+)/);
    const modelPathMatch = out.match(/MODEL_PATH\s+(.+)/);

    return {
      ok: true,
      output: out,
      trainError: trainErrorMatch ? parseFloat(trainErrorMatch[1]) : null,
      testError: testErrorMatch ? parseFloat(testErrorMatch[1]) : null,
      modelPath: modelPathMatch ? modelPathMatch[1].trim() : null,
    };
  } catch (e: any) {
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
    const xmlValidator = path.join(__dirname, "../backend/validate_dlib_xml.py");

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
ipcMain.handle("ml:test-model", async (_event, modelName: string) => {
  try {
    const out = await runPython([
      path.join(__dirname, "../backend/shape_tester.py"),
      projectRoot,
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

ipcMain.handle("ml:predict", async (_event, imagePath: string, tag: string) => {
  try {
    const out = await runPython([
      path.join(__dirname, "../backend/predict.py"),
      projectRoot,
      tag,
      imagePath,
    ]);

    const data = JSON.parse(out);
    return { ok: true, data };
  } catch (e: any) {
    console.error("Prediction failed:", e);
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

// List trained models in the models directory
ipcMain.handle("ml:list-models", async () => {
  try {
    const modelsDir = path.join(projectRoot, "models");

    if (!fs.existsSync(modelsDir)) {
      fs.mkdirSync(modelsDir, { recursive: true });
      return { ok: true, models: [] };
    }

    const files = fs.readdirSync(modelsDir);
    const models = files
      .filter((f) => f.endsWith(".dat") && f.startsWith("predictor_"))
      .map((file) => {
        const filePath = path.join(modelsDir, file);
        const stats = fs.statSync(filePath);
        // Strip "predictor_" prefix and ".dat" suffix to get just the tag name
        const tag = file.replace(/^predictor_/, "").replace(/\.dat$/, "");
        return {
          name: tag,
          path: filePath,
          size: stats.size,
          createdAt: stats.birthtime,
        };
      })
      .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());

    return { ok: true, models };
  } catch (e: any) {
    console.error("Failed to list models:", e);
    return { ok: false, error: e.message };
  }
});

// Delete a trained model
ipcMain.handle("ml:delete-model", async (_event, modelName: string) => {
  try {
    // modelName is the tag, file is stored as predictor_{tag}.dat
    const modelPath = path.join(projectRoot, "models", `predictor_${modelName}.dat`);

    if (!fs.existsSync(modelPath)) {
      return { ok: false, error: "Model not found" };
    }

    fs.unlinkSync(modelPath);
    return { ok: true };
  } catch (e: any) {
    console.error("Failed to delete model:", e);
    return { ok: false, error: e.message };
  }
});

// Rename a trained model
ipcMain.handle("ml:rename-model", async (_event, oldName: string, newName: string) => {
  try {
    // Names are tags, files are stored as predictor_{tag}.dat
    const oldPath = path.join(projectRoot, "models", `predictor_${oldName}.dat`);
    const newPath = path.join(projectRoot, "models", `predictor_${newName}.dat`);

    if (!fs.existsSync(oldPath)) {
      return { ok: false, error: "Model not found" };
    }

    if (fs.existsSync(newPath)) {
      return { ok: false, error: "A model with that name already exists" };
    }

    fs.renameSync(oldPath, newPath);
    return { ok: true };
  } catch (e: any) {
    console.error("Failed to rename model:", e);
    return { ok: false, error: e.message };
  }
});

// Get info about a specific model
ipcMain.handle("ml:get-model-info", async (_event, modelName: string) => {
  try {
    // modelName is the tag, file is stored as predictor_{tag}.dat
    const modelPath = path.join(projectRoot, "models", `predictor_${modelName}.dat`);

    if (!fs.existsSync(modelPath)) {
      return { ok: false, error: "Model not found" };
    }

    const stats = fs.statSync(modelPath);
    return {
      ok: true,
      model: {
        name: modelName,
        path: modelPath,
        size: stats.size,
        createdAt: stats.birthtime,
      },
    };
  } catch (e: any) {
    console.error("Failed to get model info:", e);
    return { ok: false, error: e.message };
  }
});

interface DetectionOptions {}

ipcMain.handle("ml:detect-specimens", async (_event, imagePath: string, _options?: DetectionOptions) => {
  try {
    const out = await runPython([
      path.join(__dirname, "../backend/detect_specimen.py"),
      imagePath,
    ]);

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
      path.join(__dirname, "../backend/detect_specimen.py"),
      "--check",
    ]);

    return JSON.parse(out.trim());
  } catch (e: any) {
    return { available: true, primary_method: "opencv" };
  }
});

// ── Session management IPC handlers ──

function getSessionDir(speciesId: string): string {
  return path.join(projectRoot, "sessions", speciesId);
}

function safeClassName(className: string): string {
  return (className || "object").toLowerCase().trim().replace(/\s+/g, "_");
}

function getSessionYoloAliasPath(speciesId: string, className: string): string {
  return path.join(getSessionDir(speciesId), "models", `yolo_${safeClassName(className)}.pt`);
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

ipcMain.handle(
  "session:create",
  async (
    _event,
    args: {
      speciesId: string;
      name: string;
      landmarkTemplate: any[];
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
      const labelsDir = path.join(sessionDir, "labels");
      fs.mkdirSync(labelsDir, { recursive: true });

      const basename = args.filename.replace(/\.\w+$/, ".json");
      const labelPath = path.join(labelsDir, basename);
      const boxes = (args.boxes || []).map((box: any) => ({
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
      }));

      let rejectedDetections: any[] = [];
      if (fs.existsSync(labelPath)) {
        try {
          const previous = JSON.parse(fs.readFileSync(labelPath, "utf-8"));
          if (Array.isArray(previous?.rejectedDetections)) {
            rejectedDetections = previous.rejectedDetections;
          }
        } catch (_) {
          // ignore malformed previous JSON
        }
      }

      fs.writeFileSync(
        labelPath,
        JSON.stringify(
          {
            imageFilename: args.filename,
            speciesId: args.speciesId,
            boxes,
            rejectedDetections,
          },
          null,
          2
        )
      );

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
        } catch (_) {
          // skip bad session.json
        }
      }

      if (!fs.existsSync(imagesDir)) {
        return { ok: true, images: [], meta };
      }

      const imageFiles = fs
        .readdirSync(imagesDir)
        .filter((f) => IMAGE_EXTS.test(f));

      const images = imageFiles.map((filename) => {
        const filePath = path.join(imagesDir, filename);
        const data = fs.readFileSync(filePath);
        const ext = path.extname(filename).toLowerCase();

        // Load annotations if they exist
        let boxes: any[] = [];
        const labelPath = path.join(
          labelsDir,
          filename.replace(/\.\w+$/, ".json")
        );
        if (fs.existsSync(labelPath)) {
          try {
            const labelData = JSON.parse(fs.readFileSync(labelPath, "utf-8"));
            boxes = labelData.boxes || [];
          } catch (_) {
            // skip bad label files
          }
        }

        return {
          filename,
          diskPath: filePath,
          data: data.toString("base64"),
          mimeType: MIME_TYPES[ext] || "image/jpeg",
          boxes,
        };
      });

      return { ok: true, images, meta };
    } catch (e: any) {
      console.error("session:load failed:", e);
      return { ok: false, error: e.message, images: [] };
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
  private pending: Map<string, { resolve: (v: any) => void; reject: (e: Error) => void }> = new Map();
  private idleTimer: ReturnType<typeof setTimeout> | null = null;
  private requestId = 0;
  private readonly IDLE_TIMEOUT_MS = 10 * 60 * 1000; // 10 minutes

  async start(): Promise<void> {
    if (this.process) return; // already running

    const pyPath = getPythonPath();
    const scriptPath = path.join(__dirname, "../backend/super_annotator.py");

    this.process = spawn(pyPath, [scriptPath], {
      stdio: ["pipe", "pipe", "pipe"],
      cwd: path.join(__dirname, ".."),
    });

    this.process.stderr?.on("data", (d: Buffer) => {
      console.log("[SuperAnnotator]", d.toString().trim());
    });

    this.rl = createInterface({ input: this.process.stdout! });
    this.rl.on("line", (line: string) => {
      try {
        const msg = JSON.parse(line);

        // Forward progress events to renderer
        if (msg.status === "progress") {
          mainWindow?.webContents.send("ml:super-annotate-progress", msg);
          return;
        }

        // Match response to pending request
        if (msg._request_id && this.pending.has(msg._request_id)) {
          const { resolve } = this.pending.get(msg._request_id)!;
          this.pending.delete(msg._request_id);
          resolve(msg);
          return;
        }

        // Unmatched response — resolve oldest pending request
        if (this.pending.size > 0) {
          const [firstId, handler] = this.pending.entries().next().value!;
          this.pending.delete(firstId);
          handler.resolve(msg);
        }
      } catch (e) {
        const trimmed = line.trim();
        if (!trimmed) return;
        // Ultralytics may print plain text (e.g. requirements/autoupdate notices) to stdout.
        // Keep logs visible but do not treat them as protocol errors.
        console.log("[SuperAnnotator stdout]", trimmed);
      }
    });

    this.process.on("close", (code: number | null) => {
      console.log(`[SuperAnnotator] Process exited with code ${code}`);
      // Reject all pending
      for (const [, handler] of this.pending) {
        handler.reject(new Error(`SuperAnnotator process exited with code ${code}`));
      }
      this.pending.clear();
      this.process = null;
      this.rl = null;
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

    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.process!.stdin!.write(JSON.stringify(payload) + "\n");
    });
  }

  async stop(): Promise<void> {
    if (this.idleTimer) {
      clearTimeout(this.idleTimer);
      this.idleTimer = null;
    }
    if (!this.process) return;
    try {
      this.process.stdin!.write(JSON.stringify({ cmd: "shutdown" }) + "\n");
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
      console.log("[SuperAnnotator] Idle timeout, shutting down process");
      this.stop();
    }, this.IDLE_TIMEOUT_MS);
  }

  get isRunning(): boolean {
    return this.process !== null;
  }
}

const superAnnotator = new SuperAnnotatorProcess();

// ── SuperAnnotator IPC handlers ──

ipcMain.handle("ml:check-super-annotator", async () => {
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
      };
    }
  ) => {
    try {
      // Ensure process is running and initialized
      if (!superAnnotator.isRunning) {
        await superAnnotator.send({ cmd: "init" });
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

      const result = await superAnnotator.send({
        cmd: "annotate",
        image_path: args.imagePath,
        class_name: args.className,
        dlib_model: dlibModel,
        id_mapping_path: idMappingPath,
        options: {
          conf_threshold: args.options?.confThreshold ?? 0.3,
          sam_enabled: args.options?.samEnabled ?? true,
          max_objects: args.options?.maxObjects ?? 20,
          detection_mode: args.options?.detectionMode ?? "auto",
          detection_preset: args.options?.detectionPreset ?? "balanced",
          finetuned_model: finetunedModel,
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
  "ml:train-yolo",
  async (
    _event,
    args: {
      speciesId: string;
      className: string;
      epochs?: number;
      detectionPreset?: string;
    }
  ) => {
    try {
      const sessionDir = path.join(projectRoot, "sessions", args.speciesId);
      if (!fs.existsSync(sessionDir)) {
        return { ok: false, error: `Session directory not found: ${sessionDir}` };
      }

      // Ensure process is running and initialized
      if (!superAnnotator.isRunning) {
        await superAnnotator.send({ cmd: "init" });
      }

      const result = await superAnnotator.send({
        cmd: "train_yolo",
        session_dir: sessionDir,
        class_name: args.className,
        epochs: args.epochs ?? 25,
        detection_preset: args.detectionPreset ?? "balanced",
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
        candidateMap50: result?.candidate_map50,
        incumbentMap50: result?.incumbent_map50,
        dataset: result?.dataset,
        registryPath: result?.registry_path,
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
