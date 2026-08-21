import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import * as path from "node:path";
import test from "node:test";

import {
  calculateHitlReviewPriority,
  commitHitlFileTransaction,
  recoverHitlFileTransaction,
  resolveContentAddressedHitlImage,
  resolveHitlNewTrainingSample,
  resolveReviewStatus,
  resolveReviewWasEdited,
  sha256FileSync,
  stageInferenceImagePaths,
  summarizeRetrainingReviewEvents,
} from "./hitlPersistence";

test("saved review drafts retain correction provenance until a fresh inference resets it", () => {
  assert.equal(resolveReviewWasEdited(false, true), true);
  assert.equal(resolveReviewWasEdited(true, false), true);
  assert.equal(resolveReviewWasEdited(true, false, false), false);
});

test("editing an item always invalidates review completion and commit state", () => {
  assert.deepEqual(
    resolveReviewStatus({
      previousReviewComplete: true,
      previousCommittedAt: "2026-01-01T00:00:00.000Z",
      edited: true,
      requestedReviewComplete: true,
      requestedCommittedAt: "2026-01-02T00:00:00.000Z",
    }),
    { reviewComplete: false, committedAt: null }
  );
});

test("an idempotent commit retry preserves the original new-sample receipt", () => {
  assert.equal(resolveHitlNewTrainingSample({
    labelExistedBeforeCommit: false,
    commitId: "commit-1",
  }), true);
  assert.equal(resolveHitlNewTrainingSample({
    labelExistedBeforeCommit: true,
    commitId: "commit-1",
    existingLabel: {
      reviewHistory: [{
        eventId: "commit-1",
        commitId: "commit-1",
        isNewTrainingSample: true,
      }],
    },
  }), true);
  assert.equal(resolveHitlNewTrainingSample({
    labelExistedBeforeCommit: true,
    commitId: "commit-2",
    existingLabel: {
      provenance: { eventId: "older", isNewTrainingSample: true },
    },
  }), false);
});

function withTempDir(run: (root: string) => void): void {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "biovision-hitl-"));
  try {
    run(root);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
}

test("same-name different-content images use a deterministic content suffix", () => {
  withTempDir((root) => {
    const imagesDir = path.join(root, "images");
    fs.mkdirSync(imagesDir);
    fs.writeFileSync(path.join(imagesDir, "specimen.png"), Buffer.from("existing"));
    const incoming = path.join(root, "incoming.png");
    fs.writeFileSync(incoming, Buffer.from("incoming"));

    const resolved = resolveContentAddressedHitlImage({
      imagesDir,
      sourcePath: incoming,
      requestedFilename: "specimen.png",
    });
    assert.equal(
      resolved.imageName,
      `specimen_png_${sha256FileSync(incoming).slice(0, 12)}.png`
    );

    const labelPath = path.join(root, "labels", resolved.imageName.replace(/\.png$/, ".json"));
    const eventsPath = path.join(root, "review_events.json");
    commitHitlFileTransaction({
      resolvedImage: resolved,
      sourcePath: incoming,
      labelPath,
      labelPayload: { imageFilename: resolved.imageName, provenance: { source: "hitl_review" } },
      reviewEventsPath: eventsPath,
      reviewEventsPayload: [{ eventId: "event-1" }],
    });
    assert.equal(fs.readFileSync(resolved.imageDestination, "utf-8"), "incoming");
    assert.equal(JSON.parse(fs.readFileSync(labelPath, "utf-8")).imageFilename, resolved.imageName);
  });
});

test("different image extensions cannot overwrite the same stem-derived label", () => {
  withTempDir((root) => {
    const imagesDir = path.join(root, "images");
    const labelsDir = path.join(root, "labels");
    fs.mkdirSync(imagesDir);
    fs.mkdirSync(labelsDir);
    fs.writeFileSync(path.join(imagesDir, "specimen.jpg"), Buffer.from("jpeg-image"));
    fs.writeFileSync(
      path.join(labelsDir, "specimen.json"),
      JSON.stringify({ imageFilename: "specimen.jpg", marker: "original" })
    );
    const incoming = path.join(root, "specimen.png");
    fs.writeFileSync(incoming, Buffer.from("png-image"));
    fs.copyFileSync(incoming, path.join(imagesDir, "specimen.png"));

    const resolved = resolveContentAddressedHitlImage({
      imagesDir,
      labelsDir,
      sourcePath: incoming,
      requestedFilename: "specimen.png",
    });
    assert.equal(
      resolved.imageName,
      `specimen_png_${sha256FileSync(incoming).slice(0, 12)}.png`
    );
    assert.notEqual(path.parse(resolved.imageName).name, "specimen");

    const labelPath = path.join(labelsDir, `${path.parse(resolved.imageName).name}.json`);
    commitHitlFileTransaction({
      resolvedImage: resolved,
      sourcePath: incoming,
      labelPath,
      labelPayload: { imageFilename: resolved.imageName },
      reviewEventsPath: path.join(root, "review_events.json"),
      reviewEventsPayload: [{ eventId: "event-cross-extension" }],
    });
    assert.deepEqual(
      JSON.parse(fs.readFileSync(path.join(labelsDir, "specimen.json"), "utf-8")),
      { imageFilename: "specimen.jpg", marker: "original" }
    );
  });
});

test("inference sources are staged by verified content instead of filename", () => {
  withTempDir((root) => {
    const firstDir = path.join(root, "first");
    const secondDir = path.join(root, "second");
    const stagedDir = path.join(root, "staged");
    fs.mkdirSync(firstDir);
    fs.mkdirSync(secondDir);
    const first = path.join(firstDir, "specimen.png");
    const second = path.join(secondDir, "specimen.png");
    fs.writeFileSync(first, Buffer.from("first-pixels"));
    fs.writeFileSync(second, Buffer.from("second-pixels"));

    const staged = stageInferenceImagePaths({
      sourceImagesDir: stagedDir,
      imagePaths: [
        { path: first, name: "specimen.png" },
        { path: second, name: "specimen.png" },
      ],
    });
    assert.equal(staged.length, 2);
    assert.notEqual(staged[0].name, staged[1].name);
    assert.equal(sha256FileSync(staged[0].path), staged[0].sourceSha256);
    assert.equal(sha256FileSync(staged[1].path), staged[1].sourceSha256);
  });
});

test("mocked HITL run publishes only the reviewed staged image as one transaction", () => {
  withTempDir((root) => {
    const incoming = path.join(root, "incoming", "specimen.png");
    const sourceImagesDir = path.join(root, "inference", "source_images");
    const imagesDir = path.join(root, "schema", "images");
    const labelsDir = path.join(root, "schema", "labels");
    const eventsPath = path.join(root, "schema", "review_events.json");
    const finalizedPath = path.join(root, "schema", "finalized_images.json");
    const sessionPath = path.join(root, "schema", "session.json");
    fs.mkdirSync(path.dirname(incoming), { recursive: true });
    fs.mkdirSync(path.dirname(sessionPath), { recursive: true });
    fs.writeFileSync(incoming, "reviewed-pixels");
    fs.writeFileSync(sessionPath, JSON.stringify({ imageCount: 0 }));

    const [staged] = stageInferenceImagePaths({
      sourceImagesDir,
      imagePaths: [{ path: incoming, name: "specimen.png" }],
    });
    assert.equal(fs.existsSync(path.join(imagesDir, staged.name)), false);

    const resolved = resolveContentAddressedHitlImage({
      imagesDir,
      labelsDir,
      sourcePath: staged.path,
      requestedFilename: staged.name,
    });
    const reviewEvent = {
      eventId: "mock-review-1",
      source: "hitl_review",
      reviewOutcome: "corrected",
      landmarkModelKey: "dlib-run-1",
      detectionModelKey: "obb-run-1",
      sourceImageSha256: staged.sourceSha256,
    };
    const labelPath = path.join(labelsDir, `${path.parse(resolved.imageName).name}.json`);
    commitHitlFileTransaction({
      resolvedImage: resolved,
      sourcePath: staged.path,
      labelPath,
      labelPayload: {
        imageFilename: resolved.imageName,
        boxes: [{ landmarks: [{ id: 1, x: 10, y: 20 }], trainingTargets: ["landmark", "obb"] }],
        provenance: reviewEvent,
        reviewHistory: [reviewEvent],
      },
      reviewEventsPath: eventsPath,
      reviewEventsPayload: [reviewEvent],
      additionalJsonWrites: [
        { path: finalizedPath, payload: [resolved.imageName] },
        { path: sessionPath, payload: { imageCount: 1 } },
      ],
    });

    assert.equal(sha256FileSync(resolved.imageDestination), staged.sourceSha256);
    assert.equal(JSON.parse(fs.readFileSync(sessionPath, "utf-8")).imageCount, 1);
    assert.deepEqual(JSON.parse(fs.readFileSync(finalizedPath, "utf-8")), [resolved.imageName]);
    assert.equal(JSON.parse(fs.readFileSync(labelPath, "utf-8")).provenance.reviewOutcome, "corrected");
    assert.equal(JSON.parse(fs.readFileSync(eventsPath, "utf-8"))[0].eventId, "mock-review-1");
  });
});

test("failure after label write restores all prior state", () => {
  withTempDir((root) => {
    const imagesDir = path.join(root, "images");
    fs.mkdirSync(imagesDir);
    const incoming = path.join(root, "source.png");
    fs.writeFileSync(incoming, Buffer.from("new-image"));
    const labelPath = path.join(root, "labels", "source.json");
    const eventsPath = path.join(root, "review_events.json");
    fs.mkdirSync(path.dirname(labelPath));
    fs.writeFileSync(labelPath, JSON.stringify({ previous: true }));
    fs.writeFileSync(eventsPath, JSON.stringify([{ eventId: "old" }]));
    const resolved = resolveContentAddressedHitlImage({ imagesDir, sourcePath: incoming });

    assert.throws(
      () => commitHitlFileTransaction({
        resolvedImage: resolved,
        sourcePath: incoming,
        labelPath,
        labelPayload: { previous: false },
        reviewEventsPath: eventsPath,
        reviewEventsPayload: [{ eventId: "new" }],
        testFailAfterStep: "label",
      }),
      /Injected failure/
    );
    assert.equal(fs.existsSync(resolved.imageDestination), false);
    assert.deepEqual(JSON.parse(fs.readFileSync(labelPath, "utf-8")), { previous: true });
    assert.deepEqual(JSON.parse(fs.readFileSync(eventsPath, "utf-8")), [{ eventId: "old" }]);
  });
});

test("failure after finalized-state writes rolls back the complete HITL commit", () => {
  withTempDir((root) => {
    const imagesDir = path.join(root, "images");
    fs.mkdirSync(imagesDir);
    const incoming = path.join(root, "review.png");
    fs.writeFileSync(incoming, Buffer.from("review-image"));
    const resolved = resolveContentAddressedHitlImage({ imagesDir, sourcePath: incoming });
    const labelPath = path.join(root, "labels", "review.json");
    const eventsPath = path.join(root, "review_events.json");
    const finalizedPath = path.join(root, "finalized_images.json");
    fs.writeFileSync(finalizedPath, JSON.stringify(["previous.png"]));

    assert.throws(
      () => commitHitlFileTransaction({
        resolvedImage: resolved,
        sourcePath: incoming,
        labelPath,
        labelPayload: { imageFilename: "review.png", provenance: { source: "hitl_review" } },
        reviewEventsPath: eventsPath,
        reviewEventsPayload: [{ eventId: "review-1" }],
        additionalJsonWrites: [{ path: finalizedPath, payload: ["previous.png", "review.png"] }],
        testFailAfterStep: "additional",
      }),
      /Injected failure/
    );
    assert.equal(fs.existsSync(resolved.imageDestination), false);
    assert.equal(fs.existsSync(labelPath), false);
    assert.equal(fs.existsSync(eventsPath), false);
    assert.deepEqual(JSON.parse(fs.readFileSync(finalizedPath, "utf-8")), ["previous.png"]);
  });
});

test("a prepared crash journal restores every file before the next commit", () => {
  withTempDir((root) => {
    const imagePath = path.join(root, "images", "crashed.png");
    const labelPath = path.join(root, "labels", "crashed.json");
    const eventsPath = path.join(root, "review_events.json");
    const sessionPath = path.join(root, "session.json");
    fs.mkdirSync(path.dirname(imagePath), { recursive: true });
    fs.mkdirSync(path.dirname(labelPath), { recursive: true });
    fs.writeFileSync(imagePath, "partially-published-image");
    fs.writeFileSync(labelPath, JSON.stringify({ partial: true }));
    fs.writeFileSync(eventsPath, JSON.stringify([{ eventId: "partial" }]));
    fs.writeFileSync(sessionPath, JSON.stringify({ imageCount: 9 }));
    fs.writeFileSync(
      path.join(root, ".hitl_commit_journal.json"),
      JSON.stringify({
        version: 1,
        transactionId: "simulated-crash",
        state: "prepared",
        createdAt: "2026-01-01T00:00:00.000Z",
        imageDestination: imagePath,
        removeImageOnRollback: true,
        files: [
          { path: labelPath, previousBase64: null },
          {
            path: eventsPath,
            previousBase64: Buffer.from(JSON.stringify([{ eventId: "before" }])).toString("base64"),
          },
          {
            path: sessionPath,
            previousBase64: Buffer.from(JSON.stringify({ imageCount: 1 })).toString("base64"),
          },
        ],
      })
    );

    assert.deepEqual(recoverHitlFileTransaction(eventsPath), {
      recovered: true,
      action: "rolled_back",
    });
    assert.equal(fs.existsSync(imagePath), false);
    assert.equal(fs.existsSync(labelPath), false);
    assert.deepEqual(JSON.parse(fs.readFileSync(eventsPath, "utf-8")), [{ eventId: "before" }]);
    assert.deepEqual(JSON.parse(fs.readFileSync(sessionPath, "utf-8")), { imageCount: 1 });
    assert.equal(fs.existsSync(path.join(root, ".hitl_commit_journal.json")), false);
  });
});

test("a committed crash journal preserves published files and only finalizes cleanup", () => {
  withTempDir((root) => {
    const imagePath = path.join(root, "images", "committed.png");
    const labelPath = path.join(root, "labels", "committed.json");
    const eventsPath = path.join(root, "review_events.json");
    fs.mkdirSync(path.dirname(imagePath), { recursive: true });
    fs.mkdirSync(path.dirname(labelPath), { recursive: true });
    fs.writeFileSync(imagePath, "committed-image");
    fs.writeFileSync(labelPath, JSON.stringify({ committed: true }));
    fs.writeFileSync(eventsPath, JSON.stringify([{ eventId: "committed" }]));
    fs.writeFileSync(
      path.join(root, ".hitl_commit_journal.json"),
      JSON.stringify({
        version: 1,
        transactionId: "simulated-committed-crash",
        state: "committed",
        createdAt: "2026-01-01T00:00:00.000Z",
        imageDestination: imagePath,
        removeImageOnRollback: true,
        files: [
          { path: labelPath, previousBase64: null },
          { path: eventsPath, previousBase64: null },
        ],
      })
    );

    assert.deepEqual(recoverHitlFileTransaction(eventsPath), {
      recovered: true,
      action: "finalized",
    });
    assert.equal(fs.readFileSync(imagePath, "utf-8"), "committed-image");
    assert.deepEqual(JSON.parse(fs.readFileSync(labelPath, "utf-8")), { committed: true });
    assert.deepEqual(JSON.parse(fs.readFileSync(eventsPath, "utf-8")), [{ eventId: "committed" }]);
    assert.equal(fs.existsSync(path.join(root, ".hitl_commit_journal.json")), false);
  });
});

test("review priority combines available uncertainty without inventing missing signals", () => {
  const high = calculateHitlReviewPriority(
    [{ box: { confidence: 0.2 }, landmarks: [{ confidence: 0.3 }, { confidence: 0.4 }] }],
    { modelDisagreement: 0.9, oodScore: 0.8, repeatedFailureCount: 4 }
  );
  const low = calculateHitlReviewPriority(
    [{ box: { confidence: 0.95 }, landmarks: [{ confidence: 0.9 }] }]
  );
  assert.equal(high.band, "high");
  assert.equal(low.band, "low");
  assert.ok(high.score > low.score);
  assert.ok(high.reasons.includes("model disagreement"));
  assert.equal(low.factors.oodScore, undefined);
});

test("retraining batch summary keeps review outcomes mutually exclusive", () => {
  assert.deepEqual(
    summarizeRetrainingReviewEvents([
      { sourceImageSha256: "a", reviewOutcome: "corrected", isNewTrainingSample: true },
      { sourceImageSha256: "b", reviewOutcome: "corrected" },
      { sourceImageSha256: "c", reviewOutcome: "accepted_unchanged" },
      { sourceImageSha256: "d", reviewOutcome: "rejected_all" },
    ]),
    { newSamples: 1, corrected: 2, unchanged: 1, rejected: 1, totalCommitted: 4 }
  );
});

test("retraining summary keeps only the latest outcome per sample while retaining newness", () => {
  assert.deepEqual(
    summarizeRetrainingReviewEvents([
      {
        sourceImageSha256: "same-content",
        reviewedAt: "2026-01-01T00:00:00.000Z",
        reviewOutcome: "corrected",
        isNewTrainingSample: true,
      },
      {
        sourceImageSha256: "same-content",
        reviewedAt: "2026-01-02T00:00:00.000Z",
        reviewOutcome: "accepted_unchanged",
      },
    ]),
    { newSamples: 1, corrected: 0, unchanged: 1, rejected: 0, totalCommitted: 1 }
  );
});
