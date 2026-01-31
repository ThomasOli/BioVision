import { app, BrowserWindow, ipcMain, dialog } from "electron";
import fs from "fs";
import * as path from "path";
import { spawn } from "child_process";

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

function runPython(args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    const pyPath = "python";
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

ipcMain.handle("ml:train", async (_event, modelName) => {
  try {
    await runPython([
      path.join(__dirname, "../backend/prepare_dataset.py"),
      projectRoot,
      modelName,
    ]);

    const out = await runPython([
      path.join(__dirname, "../backend/train_shape_model.py"),
      projectRoot,
      modelName,
    ]);

    return { ok: true, output: out };
  } catch (e: any) {
    console.error("Training failed:", e);
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

  const images = fs
    .readdirSync(folderPath)
    .filter((f) => /\.(jpg|jpeg|png)$/i.test(f))
    .map((file) => ({
      filename: file,
      path: path.join(folderPath, file),
    }));

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

  const files = result.filePaths.map((filePath) => ({
    path: filePath,
    name: path.basename(filePath),
  }));

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
      .filter((f) => f.endsWith(".dat"))
      .map((file) => {
        const filePath = path.join(modelsDir, file);
        const stats = fs.statSync(filePath);
        return {
          name: file.replace(/\.dat$/, ""),
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
    const modelPath = path.join(projectRoot, "models", `${modelName}.dat`);

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
    const oldPath = path.join(projectRoot, "models", `${oldName}.dat`);
    const newPath = path.join(projectRoot, "models", `${newName}.dat`);

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
    const modelPath = path.join(projectRoot, "models", `${modelName}.dat`);

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

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});
