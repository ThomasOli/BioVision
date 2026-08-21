import fs from "fs";
import * as path from "path";
import { createHash, randomUUID } from "crypto";

export function atomicWriteFileSync(targetPath: string, data: string | Buffer): void {
  fs.mkdirSync(path.dirname(targetPath), { recursive: true });
  const tempPath = path.join(
    path.dirname(targetPath),
    `.${path.basename(targetPath)}.${randomUUID()}.tmp`
  );
  try {
    fs.writeFileSync(tempPath, data);
    const descriptor = fs.openSync(tempPath, "r");
    try {
      try {
        fs.fsyncSync(descriptor);
      } catch (error: any) {
        // Some Windows/sandbox filesystems reject fsync on ordinary files even
        // though same-directory rename remains atomic.
        if (error?.code !== "EPERM" && error?.code !== "EINVAL") throw error;
      }
    } finally {
      fs.closeSync(descriptor);
    }
    fs.renameSync(tempPath, targetPath);
  } finally {
    try {
      if (fs.existsSync(tempPath)) fs.unlinkSync(tempPath);
    } catch {}
  }
}

export function atomicWriteJsonSync(targetPath: string, payload: unknown): void {
  atomicWriteFileSync(targetPath, JSON.stringify(payload, null, 2));
}

export function atomicCopyFileSync(sourcePath: string, targetPath: string): void {
  atomicWriteFileSync(targetPath, fs.readFileSync(sourcePath));
}

export function sha256FileSync(filePath: string): string {
  return createHash("sha256").update(fs.readFileSync(filePath)).digest("hex");
}

export type HitlPrioritySignals = {
  detectorConfidence?: number;
  landmarkConfidence?: number;
  landmarkHeatmapEntropy?: number;
  modelDisagreement?: number;
  oodScore?: number;
  repeatedFailureCount?: number;
};

export type HitlReviewPriority = {
  score: number;
  band: "high" | "medium" | "low";
  factors: {
    detectorUncertainty?: number;
    landmarkUncertainty?: number;
    modelDisagreement?: number;
    oodScore?: number;
    repeatedFailure?: number;
  };
  reasons: string[];
};

/**
 * Keep correction provenance after a draft is saved (and therefore no longer
 * dirty), while allowing a fresh inference pass to explicitly reset it.
 */
export function resolveReviewWasEdited(
  previousWasEdited: unknown,
  edited: unknown,
  explicitWasEdited?: boolean
): boolean {
  if (typeof explicitWasEdited === "boolean") return explicitWasEdited;
  return Boolean(previousWasEdited || edited);
}

export function resolveReviewStatus(args: {
  previousReviewComplete?: unknown;
  previousCommittedAt?: unknown;
  edited?: unknown;
  requestedReviewComplete?: boolean;
  requestedCommittedAt?: string | null;
}): { reviewComplete: boolean; committedAt: string | null } {
  if (Boolean(args.edited)) {
    return { reviewComplete: false, committedAt: null };
  }
  const reviewComplete = typeof args.requestedReviewComplete === "boolean"
    ? args.requestedReviewComplete
    : Boolean(args.previousReviewComplete);
  if (!reviewComplete) {
    return { reviewComplete: false, committedAt: null };
  }
  const committedAt = args.requestedCommittedAt === null
    ? null
    : typeof args.requestedCommittedAt === "string"
      ? args.requestedCommittedAt
      : typeof args.previousCommittedAt === "string"
        ? args.previousCommittedAt
        : null;
  return { reviewComplete: true, committedAt };
}

function clamp01(value: unknown): number | undefined {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return undefined;
  return Math.max(0, Math.min(1, numeric));
}

/**
 * Rank review work from uncertainty signals without pretending unavailable
 * measures were observed. Detector/landmark confidences are useful fallbacks;
 * callers can add calibrated entropy, disagreement and OOD scores later.
 */
export function calculateHitlReviewPriority(
  specimens: Array<{
    box?: { confidence?: number };
    landmarks?: Array<{ confidence?: number }>;
  }> = [],
  signals: HitlPrioritySignals = {}
): HitlReviewPriority {
  const detectorValues = specimens
    .map((specimen) => clamp01(specimen?.box?.confidence))
    .filter((value): value is number => value !== undefined);
  const landmarkValues = specimens
    .flatMap((specimen) => specimen?.landmarks || [])
    .map((landmark) => clamp01(landmark?.confidence))
    .filter((value): value is number => value !== undefined);

  const explicitDetectorConfidence = clamp01(signals.detectorConfidence);
  const detectorConfidence = explicitDetectorConfidence ?? (
    detectorValues.length
      ? detectorValues.reduce((sum, value) => sum + value, 0) / detectorValues.length
      : undefined
  );
  const heatmapEntropy = clamp01(signals.landmarkHeatmapEntropy);
  const explicitLandmarkConfidence = clamp01(signals.landmarkConfidence);
  const landmarkConfidence = explicitLandmarkConfidence ?? (
    landmarkValues.length
      ? landmarkValues.reduce((sum, value) => sum + value, 0) / landmarkValues.length
      : undefined
  );
  const repeatedFailureCount = Math.max(0, Number(signals.repeatedFailureCount) || 0);

  const factors: HitlReviewPriority["factors"] = {};
  if (detectorConfidence !== undefined) factors.detectorUncertainty = 1 - detectorConfidence;
  else if (specimens.length === 0) factors.detectorUncertainty = 1;
  if (heatmapEntropy !== undefined) factors.landmarkUncertainty = heatmapEntropy;
  else if (landmarkConfidence !== undefined) factors.landmarkUncertainty = 1 - landmarkConfidence;
  const disagreement = clamp01(signals.modelDisagreement);
  if (disagreement !== undefined) factors.modelDisagreement = disagreement;
  const oodScore = clamp01(signals.oodScore);
  if (oodScore !== undefined) factors.oodScore = oodScore;
  if (repeatedFailureCount > 0) {
    factors.repeatedFailure = 1 - Math.exp(-repeatedFailureCount / 3);
  }

  const weightedFactors: Array<[keyof HitlReviewPriority["factors"], number]> = [
    ["detectorUncertainty", 0.30],
    ["landmarkUncertainty", 0.25],
    ["modelDisagreement", 0.20],
    ["oodScore", 0.15],
    ["repeatedFailure", 0.10],
  ];
  let weightedScore = 0;
  let observedWeight = 0;
  for (const [factor, weight] of weightedFactors) {
    const value = factors[factor];
    if (value === undefined) continue;
    weightedScore += value * weight;
    observedWeight += weight;
  }
  const score = Number((observedWeight > 0 ? weightedScore / observedWeight : 0.5).toFixed(6));
  const band: HitlReviewPriority["band"] = score >= 0.67 ? "high" : score >= 0.34 ? "medium" : "low";
  const reasonLabels: Record<keyof HitlReviewPriority["factors"], string> = {
    detectorUncertainty: "low detector confidence",
    landmarkUncertainty: "uncertain landmarks",
    modelDisagreement: "model disagreement",
    oodScore: "out-of-distribution signal",
    repeatedFailure: "repeated failure history",
  };
  const reasons = weightedFactors
    .filter(([factor]) => Number(factors[factor] ?? 0) >= 0.5)
    .sort(([left], [right]) => Number(factors[right] ?? 0) - Number(factors[left] ?? 0))
    .map(([factor]) => reasonLabels[factor]);
  return { score, band, factors, reasons };
}

export type RetrainingBatchSummary = {
  newSamples: number;
  corrected: number;
  unchanged: number;
  rejected: number;
  totalCommitted: number;
};

/**
 * Keep the original "new sample" receipt when an idempotent commit is retried.
 * The training-data transaction may finish just before the renderer draft is
 * marked committed; replaying the same commit ID must not turn that sample into
 * an update merely because its label now exists.
 */
export function resolveHitlNewTrainingSample(args: {
  labelExistedBeforeCommit: boolean;
  commitId: string;
  existingLabel?: {
    provenance?: { eventId?: string; commitId?: string; isNewTrainingSample?: boolean };
    reviewHistory?: Array<{
      eventId?: string;
      commitId?: string;
      isNewTrainingSample?: boolean;
    }>;
  } | null;
}): boolean {
  if (!args.labelExistedBeforeCommit) return true;
  const priorEvents = [
    ...(Array.isArray(args.existingLabel?.reviewHistory)
      ? args.existingLabel!.reviewHistory!
      : []),
    ...(args.existingLabel?.provenance ? [args.existingLabel.provenance] : []),
  ];
  return priorEvents.some((event) =>
    (event?.eventId === args.commitId || event?.commitId === args.commitId) &&
    event?.isNewTrainingSample === true
  );
}

export function summarizeRetrainingReviewEvents(
  events: Array<{
    eventId?: string;
    sourceImageSha256?: string;
    imageFilename?: string;
    reviewedAt?: string;
    reviewOutcome?: string;
    isNewTrainingSample?: boolean;
  }> = []
): RetrainingBatchSummary {
  const summary: RetrainingBatchSummary = {
    newSamples: 0,
    corrected: 0,
    unchanged: 0,
    rejected: 0,
    totalCommitted: 0,
  };
  const latestBySample = new Map<string, {
    event: (typeof events)[number];
    order: number;
    newInWindow: boolean;
  }>();
  events.forEach((event, order) => {
    const identity = String(
      event?.sourceImageSha256 || event?.imageFilename || event?.eventId || `event:${order}`
    ).trim().toLowerCase();
    const previous = latestBySample.get(identity);
    const eventTime = Date.parse(String(event?.reviewedAt || ""));
    const previousTime = Date.parse(String(previous?.event?.reviewedAt || ""));
    const shouldReplace = !previous ||
      (Number.isFinite(eventTime) && (!Number.isFinite(previousTime) || eventTime >= previousTime)) ||
      (!Number.isFinite(eventTime) && !Number.isFinite(previousTime) && order > previous.order);
    const newInWindow = Boolean(
      previous?.newInWindow || event?.isNewTrainingSample || event?.reviewOutcome === "new"
    );
    if (shouldReplace) latestBySample.set(identity, { event, order, newInWindow });
    else if (previous) previous.newInWindow = newInWindow;
  });

  for (const { event, newInWindow } of latestBySample.values()) {
    const outcome = String(event?.reviewOutcome || "").trim().toLowerCase();
    summary.totalCommitted += 1;
    if (newInWindow) summary.newSamples += 1;
    if (outcome === "corrected") summary.corrected += 1;
    else if (outcome === "accepted_unchanged") summary.unchanged += 1;
    else if (outcome === "rejected_all" || outcome === "rejected") summary.rejected += 1;
  }
  return summary;
}

export type ResolvedHitlImage = {
  imageName: string;
  imageDestination: string;
  sourceSha256: string;
  sourceExists: boolean;
};

export type StagedInferenceImage = {
  path: string;
  name: string;
  sourcePath: string;
  sourceName: string;
  sourceSha256: string;
};

export function resolveContentAddressedHitlImage(args: {
  imagesDir: string;
  labelsDir?: string;
  sourcePath: string;
  requestedFilename?: string;
}): ResolvedHitlImage {
  let imageName = path.basename(args.requestedFilename || path.basename(args.sourcePath));
  if (!imageName) throw new Error("A valid image filename is required.");
  let imageDestination = path.join(args.imagesDir, imageName);
  const sourceExists = fs.existsSync(args.sourcePath);
  if (!sourceExists && !fs.existsSync(imageDestination)) {
    throw new Error(`Image not found: ${args.sourcePath}`);
  }
  const sourceSha256 = sourceExists
    ? sha256FileSync(args.sourcePath)
    : sha256FileSync(imageDestination);

  const sameStoredContent = (candidatePath: string): boolean =>
    fs.existsSync(candidatePath) && sha256FileSync(candidatePath) === sourceSha256;
  const stemIsClaimed = (candidateName: string): boolean => {
    const parsed = path.parse(candidateName);
    const stem = parsed.name.toLowerCase();
    const candidatePath = path.resolve(args.imagesDir, candidateName).toLowerCase();
    if (fs.existsSync(args.imagesDir)) {
      const claimedByAnotherImage = fs.readdirSync(args.imagesDir).some((entry) =>
        path.parse(entry).name.toLowerCase() === stem &&
        path.resolve(args.imagesDir, entry).toLowerCase() !== candidatePath
      );
      if (claimedByAnotherImage) return true;
    }
    if (!args.labelsDir) return false;
    const labelPath = path.join(args.labelsDir, `${parsed.name}.json`);
    if (!fs.existsSync(labelPath)) return false;
    try {
      const label = JSON.parse(fs.readFileSync(labelPath, "utf-8"));
      return String(label?.imageFilename || "").toLowerCase() !== candidateName.toLowerCase();
    } catch {
      return true;
    }
  };

  if (sameStoredContent(imageDestination) && !stemIsClaimed(imageName)) {
    return { imageName, imageDestination, sourceSha256, sourceExists };
  }

  const exactNameCollision = fs.existsSync(imageDestination);
  if (sourceExists && (exactNameCollision || stemIsClaimed(imageName))) {
    const extension = path.extname(imageName);
    const extensionToken = extension.replace(/^\./, "").toLowerCase() || "image";
    const base = path.basename(imageName, extension);
    const candidates = [sourceSha256.slice(0, 12), sourceSha256];
    let resolved = false;
    for (const digest of candidates) {
      const candidateName = `${base}_${extensionToken}_${digest}${extension}`;
      const candidateDestination = path.join(args.imagesDir, candidateName);
      if (sameStoredContent(candidateDestination) && !stemIsClaimed(candidateName)) {
        imageName = candidateName;
        imageDestination = candidateDestination;
        resolved = true;
        break;
      }
      if (!fs.existsSync(candidateDestination) && !stemIsClaimed(candidateName)) {
        imageName = candidateName;
        imageDestination = candidateDestination;
        resolved = true;
        break;
      }
    }
    if (!resolved) {
      throw new Error(`Content-addressed image collision for ${imageName}.`);
    }
  }
  return { imageName, imageDestination, sourceSha256, sourceExists };
}

export function stageInferenceImagePaths(args: {
  sourceImagesDir: string;
  imagePaths: Array<{ path: string; name: string }>;
}): StagedInferenceImage[] {
  fs.mkdirSync(args.sourceImagesDir, { recursive: true });
  return (args.imagePaths || []).map((entry) => {
    if (!entry?.path || !entry?.name || !fs.existsSync(entry.path)) {
      throw new Error(`Inference source image is unavailable: ${entry?.path || entry?.name || "(unknown)"}`);
    }
    const resolved = resolveContentAddressedHitlImage({
      imagesDir: args.sourceImagesDir,
      sourcePath: entry.path,
      requestedFilename: entry.name,
    });
    if (!fs.existsSync(resolved.imageDestination)) {
      atomicCopyFileSync(entry.path, resolved.imageDestination);
    }
    return {
      path: resolved.imageDestination,
      name: resolved.imageName,
      sourcePath: path.resolve(entry.path),
      sourceName: path.basename(entry.name),
      sourceSha256: resolved.sourceSha256,
    };
  });
}

function restoreFile(targetPath: string, previous: Buffer | null): void {
  if (previous) atomicWriteFileSync(targetPath, previous);
  else if (fs.existsSync(targetPath)) fs.unlinkSync(targetPath);
}

type HitlTransactionJournal = {
  version: 1;
  transactionId: string;
  state: "prepared" | "committed";
  createdAt: string;
  imageDestination: string;
  removeImageOnRollback: boolean;
  files: Array<{ path: string; previousBase64: string | null }>;
};

function getHitlTransactionJournalPath(reviewEventsPath: string): string {
  return path.join(path.dirname(reviewEventsPath), ".hitl_commit_journal.json");
}

/**
 * Recover the only durable states a synchronous HITL commit may leave behind.
 * A prepared journal is rolled back; a committed journal only needs cleanup.
 * The journal intentionally lives beside review_events.json so recovery does
 * not depend on a process-local temporary directory.
 */
export function recoverHitlFileTransaction(reviewEventsPath: string): {
  recovered: boolean;
  action?: "rolled_back" | "finalized";
} {
  const journalPath = getHitlTransactionJournalPath(reviewEventsPath);
  if (!fs.existsSync(journalPath)) return { recovered: false };

  let journal: HitlTransactionJournal;
  try {
    journal = JSON.parse(fs.readFileSync(journalPath, "utf-8")) as HitlTransactionJournal;
  } catch (error: any) {
    throw new Error(
      `Cannot recover interrupted HITL commit because its journal is unreadable: ${String(error?.message || error)}`
    );
  }
  if (journal?.version !== 1 || !Array.isArray(journal.files)) {
    throw new Error("Cannot recover interrupted HITL commit because its journal format is invalid.");
  }

  if (journal.state === "prepared") {
    const recoveryErrors: string[] = [];
    for (const entry of journal.files) {
      try {
        restoreFile(
          String(entry.path),
          entry.previousBase64 === null ? null : Buffer.from(entry.previousBase64, "base64")
        );
      } catch (error: any) {
        recoveryErrors.push(`${entry.path}: ${String(error?.message || error)}`);
      }
    }
    if (journal.removeImageOnRollback && journal.imageDestination) {
      try {
        if (fs.existsSync(journal.imageDestination)) fs.unlinkSync(journal.imageDestination);
      } catch (error: any) {
        recoveryErrors.push(
          `${journal.imageDestination}: ${String(error?.message || error)}`
        );
      }
    }
    if (recoveryErrors.length > 0) {
      throw new Error(
        `Interrupted HITL commit rollback was incomplete: ${recoveryErrors.join("; ")}`
      );
    }
  } else if (journal.state !== "committed") {
    throw new Error(`Cannot recover HITL commit with unknown journal state: ${String(journal.state)}`);
  }

  try {
    fs.unlinkSync(journalPath);
  } catch (error: any) {
    throw new Error(`Recovered HITL commit but could not remove its journal: ${String(error?.message || error)}`);
  }
  return {
    recovered: true,
    action: journal.state === "prepared" ? "rolled_back" : "finalized",
  };
}

export function commitHitlFileTransaction(args: {
  resolvedImage: ResolvedHitlImage;
  sourcePath: string;
  labelPath: string;
  labelPayload: unknown;
  reviewEventsPath: string;
  reviewEventsPayload: unknown;
  additionalJsonWrites?: Array<{ path: string; payload: unknown }>;
  /** Deterministic failure injection used by the filesystem regression test. */
  testFailAfterStep?: "image" | "label" | "events" | "additional";
}): { imageCreated: boolean } {
  recoverHitlFileTransaction(args.reviewEventsPath);
  const previousLabel = fs.existsSync(args.labelPath) ? fs.readFileSync(args.labelPath) : null;
  const previousEvents = fs.existsSync(args.reviewEventsPath)
    ? fs.readFileSync(args.reviewEventsPath)
    : null;
  const additionalWrites = args.additionalJsonWrites || [];
  const previousAdditional = additionalWrites.map((write) => ({
    path: write.path,
    previous: fs.existsSync(write.path) ? fs.readFileSync(write.path) : null,
  }));
  const imageExisted = fs.existsSync(args.resolvedImage.imageDestination);
  const journalPath = getHitlTransactionJournalPath(args.reviewEventsPath);
  const journal: HitlTransactionJournal = {
    version: 1,
    transactionId: randomUUID(),
    state: "prepared",
    createdAt: new Date().toISOString(),
    imageDestination: args.resolvedImage.imageDestination,
    removeImageOnRollback: !imageExisted,
    files: [
      { path: args.labelPath, previousBase64: previousLabel?.toString("base64") ?? null },
      { path: args.reviewEventsPath, previousBase64: previousEvents?.toString("base64") ?? null },
      ...previousAdditional.map((entry) => ({
        path: entry.path,
        previousBase64: entry.previous?.toString("base64") ?? null,
      })),
    ],
  };
  let imageCreated = false;
  try {
    atomicWriteJsonSync(journalPath, journal);
    if (args.resolvedImage.sourceExists && !imageExisted) {
      atomicCopyFileSync(args.sourcePath, args.resolvedImage.imageDestination);
      imageCreated = true;
    }
    if (args.testFailAfterStep === "image") throw new Error("Injected failure after image");
    atomicWriteJsonSync(args.labelPath, args.labelPayload);
    if (args.testFailAfterStep === "label") throw new Error("Injected failure after label");
    atomicWriteJsonSync(args.reviewEventsPath, args.reviewEventsPayload);
    if (args.testFailAfterStep === "events") throw new Error("Injected failure after events");
    for (const write of additionalWrites) {
      atomicWriteJsonSync(write.path, write.payload);
    }
    if (args.testFailAfterStep === "additional") throw new Error("Injected failure after additional writes");
    atomicWriteJsonSync(journalPath, { ...journal, state: "committed" });
    try {
      fs.unlinkSync(journalPath);
    } catch {
      // Publication is already durable and explicitly marked committed. Do not
      // report a failed review after all user data was saved; the next recovery
      // pass will safely remove this cleanup-only journal.
    }
    return { imageCreated };
  } catch (error) {
    try {
      recoverHitlFileTransaction(args.reviewEventsPath);
    } catch (rollbackError: any) {
      const originalMessage = error instanceof Error ? error.message : String(error);
      throw new Error(
        `${originalMessage}; HITL rollback also failed: ${String(rollbackError?.message || rollbackError)}`
      );
    }
    throw error;
  }
}
