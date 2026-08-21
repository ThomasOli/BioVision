const fs = require("node:fs");
const path = require("node:path");
const { spawnSync } = require("node:child_process");

const projectRoot = path.resolve(__dirname, "..");
const outputDir = path.join(projectRoot, ".electron-tests-build");
const tscPath = path.join(projectRoot, "node_modules", "typescript", "bin", "tsc");
try {
  fs.rmSync(outputDir, { recursive: true, force: true });
  const compile = spawnSync(
    process.execPath,
    [tscPath, "--project", path.join(projectRoot, "scripts", "tsconfig.electron-tests.json")],
    { cwd: projectRoot, stdio: "inherit" }
  );
  if (compile.status !== 0) {
    process.exitCode = compile.status == null ? 1 : compile.status;
  } else {
    const testFiles = [
      path.join(outputDir, "electron", "detectorProvenance.test.js"),
      path.join(outputDir, "electron", "hitlPersistence.test.js"),
      path.join(outputDir, "electron", "fourModePipeline.integration.test.js"),
      path.join(outputDir, "electron", "modelArtifactIntegrity.test.js"),
      path.join(outputDir, "electron", "modelCompatibility.test.js"),
      path.join(outputDir, "electron", "obbInferenceOptions.test.js"),
      path.join(outputDir, "electron", "trainingProtocol.test.js"),
    ];
    const result = spawnSync(process.execPath, ["--test", ...testFiles], {
      cwd: projectRoot,
      stdio: "inherit",
    });
    process.exitCode = result.status == null ? 1 : result.status;
  }
} finally {
  fs.rmSync(outputDir, { recursive: true, force: true });
}
