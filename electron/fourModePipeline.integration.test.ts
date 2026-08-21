import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import * as path from "node:path";
import test from "node:test";

import {
  commitHitlFileTransaction,
  resolveContentAddressedHitlImage,
  sha256FileSync,
} from "./hitlPersistence";

type HarnessEntry = {
  mode: string;
  sessionDir: string;
  reviewSourcePath: string;
  reviewRequestedFilename: string;
  reviewLabelPayload: Record<string, unknown>;
  reviewEvent: Record<string, unknown>;
  committedImageName?: string;
  committedSourceSha256?: string;
};

type HarnessState = {
  version: number;
  modes: HarnessEntry[];
};

function resolvePython(projectRoot: string): string {
  if (process.env.BIOVISION_TEST_PYTHON) return process.env.BIOVISION_TEST_PYTHON;
  const candidates = process.platform === "win32"
    ? [
        path.join(projectRoot, "venv", "Scripts", "python.exe"),
        path.join(projectRoot, ".venv", "Scripts", "python.exe"),
      ]
    : [
        path.join(projectRoot, "venv", "bin", "python"),
        path.join(projectRoot, ".venv", "bin", "python"),
      ];
  return candidates.find((candidate) => fs.existsSync(candidate)) ||
    (process.platform === "win32" ? "python" : "python3");
}

function runHarness(
  python: string,
  harnessPath: string,
  args: string[],
  projectRoot: string
): string {
  const result = spawnSync(python, ["-B", harnessPath, ...args], {
    cwd: projectRoot,
    encoding: "utf-8",
    env: { ...process.env, PYTHONUTF8: "1" },
    maxBuffer: 16 * 1024 * 1024,
  });
  if (result.error) throw result.error;
  assert.equal(
    result.status,
    0,
    [
      `four-mode Python harness exited with ${result.status}`,
      result.stdout,
      result.stderr,
    ].filter(Boolean).join("\n")
  );
  return result.stdout.trim();
}

test(
  "all four modes survive a real transactional HITL commit and mocked live retraining loop",
  { timeout: 240_000 },
  () => {
    const projectRoot = process.cwd();
    const harnessPath = path.join(projectRoot, "scripts", "four_mode_pipeline_harness.py");
    const python = resolvePython(projectRoot);
    const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "biovision-four-mode-live-loop-"));
    const workspace = path.join(tempRoot, "workspace");
    const statePath = path.join(tempRoot, "harness-state.json");

    try {
      runHarness(
        python,
        harnessPath,
        ["initial", "--workspace", workspace, "--state", statePath],
        projectRoot
      );
      const state = JSON.parse(fs.readFileSync(statePath, "utf-8")) as HarnessState;
      assert.equal(state.version, 1);
      assert.deepEqual(
        state.modes.map((entry) => entry.mode),
        ["directional", "bilateral", "axial", "invariant"]
      );

      for (const entry of state.modes) {
        const imagesDir = path.join(entry.sessionDir, "images");
        const labelsDir = path.join(entry.sessionDir, "labels");
        const eventsPath = path.join(entry.sessionDir, "review_events.json");
        const finalizedPath = path.join(entry.sessionDir, "finalized_images.json");
        const sessionPath = path.join(entry.sessionDir, "session.json");
        const resolved = resolveContentAddressedHitlImage({
          imagesDir,
          labelsDir,
          sourcePath: entry.reviewSourcePath,
          requestedFilename: entry.reviewRequestedFilename,
        });
        const reviewEvent: Record<string, unknown> = {
          ...entry.reviewEvent,
          sourceImageSha256: resolved.sourceSha256,
          imageFilename: resolved.imageName,
        };
        const labelPayload = {
          ...entry.reviewLabelPayload,
          imageFilename: resolved.imageName,
          provenance: reviewEvent,
          reviewHistory: [reviewEvent],
        };
        const priorEvents = fs.existsSync(eventsPath)
          ? JSON.parse(fs.readFileSync(eventsPath, "utf-8"))
          : [];
        const finalized = fs.existsSync(finalizedPath)
          ? JSON.parse(fs.readFileSync(finalizedPath, "utf-8"))
          : [];
        const session = JSON.parse(fs.readFileSync(sessionPath, "utf-8"));
        const labelPath = path.join(labelsDir, `${path.parse(resolved.imageName).name}.json`);

        commitHitlFileTransaction({
          resolvedImage: resolved,
          sourcePath: entry.reviewSourcePath,
          labelPath,
          labelPayload,
          reviewEventsPath: eventsPath,
          reviewEventsPayload: [...priorEvents, reviewEvent],
          additionalJsonWrites: [
            { path: finalizedPath, payload: [...finalized, resolved.imageName] },
            { path: sessionPath, payload: { ...session, imageCount: Number(session.imageCount || 0) + 1 } },
          ],
        });

        assert.equal(sha256FileSync(resolved.imageDestination), resolved.sourceSha256);
        assert.equal(JSON.parse(fs.readFileSync(labelPath, "utf-8")).provenance.source, "hitl_review");
        assert.equal(
          JSON.parse(fs.readFileSync(eventsPath, "utf-8")).at(-1).eventId,
          reviewEvent["eventId"]
        );
        assert.equal(fs.existsSync(path.join(entry.sessionDir, ".hitl_commit_journal.json")), false);
        entry.committedImageName = resolved.imageName;
        entry.committedSourceSha256 = resolved.sourceSha256;
      }
      fs.writeFileSync(statePath, JSON.stringify(state, null, 2));

      const stdout = runHarness(
        python,
        harnessPath,
        ["resume", "--state", statePath],
        projectRoot
      );
      const summary = JSON.parse(stdout.split(/\r?\n/).at(-1) || "{}") as {
        ok?: boolean;
        modes?: Array<{
          mode: string;
          semantics: { nc: number; classes: number[] };
          improvements: {
            landmarkValidationErrorReductionPercent: number;
            obbValidationMap50_95Gain: number;
            obbTestReportRole: string;
          };
        }>;
      };
      assert.equal(summary.ok, true);
      assert.equal(summary.modes?.length, 4);
      for (const mode of summary.modes || []) {
        assert.ok(mode.improvements.landmarkValidationErrorReductionPercent > 0);
        assert.ok(mode.improvements.obbValidationMap50_95Gain > 0);
        assert.equal(mode.improvements.obbTestReportRole, "report_only");
        assert.equal(
          mode.semantics.nc,
          mode.mode === "directional" || mode.mode === "bilateral" ? 2 : 1
        );
      }
    } finally {
      fs.rmSync(tempRoot, { recursive: true, force: true });
    }
  }
);
