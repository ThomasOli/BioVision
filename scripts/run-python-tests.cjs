const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const projectRoot = path.resolve(__dirname, "..");
const candidates = process.platform === "win32"
  ? [
      path.join(projectRoot, "venv", "Scripts", "python.exe"),
      path.join(projectRoot, ".venv", "Scripts", "python.exe"),
    ]
  : [
      path.join(projectRoot, "venv", "bin", "python"),
      path.join(projectRoot, ".venv", "bin", "python"),
    ];
const python = process.env.BIOVISION_TEST_PYTHON
  || candidates.find((candidate) => fs.existsSync(candidate))
  || (process.platform === "win32" ? "python" : "python3");

const result = spawnSync(
  python,
  ["-B", "-m", "pytest", ...process.argv.slice(2)],
  {
    cwd: projectRoot,
    env: { ...process.env, PYTHONUTF8: "1" },
    stdio: "inherit",
  }
);

if (result.error) {
  console.error(`Could not start Python test runner '${python}': ${result.error.message}`);
  process.exit(1);
}
process.exit(result.status ?? 1);
