const { spawnSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const projectRoot = path.resolve(__dirname, "..");
const backendDir = path.join(projectRoot, "backend");
const distDir = path.join(backendDir, "dist");
const buildDir = path.join(backendDir, "build");
const specDir = path.join(buildDir, "spec");
const scripts = ["prepare_dataset", "train_shape_model", "predict"];

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
    console.error("Install it with: python -m pip install -r backend/requirements.txt pyinstaller");
    process.exit(1);
  }
}

function main() {
  assertPyInstaller();

  fs.rmSync(distDir, { recursive: true, force: true });
  fs.rmSync(buildDir, { recursive: true, force: true });
  fs.mkdirSync(distDir, { recursive: true });
  fs.mkdirSync(specDir, { recursive: true });

  for (const script of scripts) {
    const scriptPath = path.join(backendDir, `${script}.py`);
    run("pyinstaller", [
      "--noconfirm",
      "--clean",
      "--onefile",
      "--name",
      script,
      "--distpath",
      distDir,
      "--workpath",
      path.join(buildDir, script),
      "--specpath",
      specDir,
      scriptPath,
    ]);
  }

  console.log(`Bundled backend executables in ${distDir}`);
}

main();
