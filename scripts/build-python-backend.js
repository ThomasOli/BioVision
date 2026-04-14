const { spawnSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const projectRoot = path.resolve(__dirname, "..");
const backendDir = path.join(projectRoot, "backend");
const distDir = path.join(backendDir, "dist");
const buildDir = path.join(backendDir, "build");
const specDir = path.join(buildDir, "spec");

function run(cmd, args) {
  const result = spawnSync(cmd, args, {
    cwd: projectRoot,
    stdio: "inherit",
    shell: process.platform === "win32",
  });

  if (result.status !== 0) {
    process.exit(result.status || 1);
  }
}

function assertPyInstaller() {
  const result = spawnSync("pyinstaller", ["--version"], {
    cwd: projectRoot,
    stdio: "ignore",
    shell: process.platform === "win32",
  });

  if (result.status !== 0) {
    console.error("pyinstaller is required to bundle the Python backend.");
    console.error("Install it with: pip install pyinstaller");
    process.exit(1);
  }
}

function main() {
  assertPyInstaller();

  fs.rmSync(distDir, { recursive: true, force: true });
  fs.rmSync(buildDir, { recursive: true, force: true });
  fs.mkdirSync(distDir, { recursive: true });
  fs.mkdirSync(specDir, { recursive: true });

  // Bundle the single CLI dispatcher. PyInstaller automatically discovers
  // all imported modules, so torch/ultralytics/dlib/opencv are included once.
  const cliScript = path.join(backendDir, "cli.py");
  run("pyinstaller", [
    "--noconfirm",
    "--clean",
    "--onefile",
    "--name", "biovision_backend",
    "--distpath", distDir,
    "--workpath", path.join(buildDir, "biovision_backend"),
    "--specpath", specDir,
    // Add all backend subpackages as hidden imports so PyInstaller
    // includes them even though they're loaded dynamically via runpy
    "--hidden-import", "data.prepare_dataset",
    "--hidden-import", "data.validate_dlib_xml",
    "--hidden-import", "data.audit_dataset",
    "--hidden-import", "data.export_yolo_dataset",
    "--hidden-import", "training.train_shape_model",
    "--hidden-import", "training.train_cnn_model",
    "--hidden-import", "inference.predict",
    "--hidden-import", "inference.predict_worker",
    "--hidden-import", "inference.shape_tester",
    "--hidden-import", "inference.list_cnn_variants",
    "--hidden-import", "detection.detect_specimen",
    "--hidden-import", "annotation.super_annotator",
    "--hidden-import", "hardware_probe",
    // Add the backend directory to the search path
    "--paths", backendDir,
    cliScript,
  ]);

  console.log(`Bundled biovision_backend in ${distDir}`);
}

main();
