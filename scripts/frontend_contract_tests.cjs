const assert = require("node:assert/strict");
const fs = require("node:fs");
const Module = require("node:module");
const path = require("node:path");
const ts = require("typescript");

const projectRoot = path.resolve(__dirname, "..");

function loadTypeScriptModule(relativePath) {
  const filename = path.join(projectRoot, relativePath);
  const source = fs.readFileSync(filename, "utf8");
  const output = ts.transpileModule(source, {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2020,
      esModuleInterop: true,
    },
    fileName: filename,
  }).outputText;
  const loaded = new Module(filename, module);
  loaded.filename = filename;
  loaded.paths = Module._nodeModulePaths(path.dirname(filename));
  loaded._compile(output, `${filename}.js`);
  return loaded.exports;
}

const obb = loadTypeScriptModule("src/lib/obbDetectorSettings.ts");
const schema = loadTypeScriptModule("src/lib/schemaFingerprint.ts");
const modelIdentity = loadTypeScriptModule("src/lib/modelIdentity.ts");
const trainingPipeline = loadTypeScriptModule("src/lib/trainingPipelineGate.ts");
const hitlReviewUi = loadTypeScriptModule("src/lib/hitlReviewUi.ts");
const orientationDisplay = loadTypeScriptModule("src/lib/orientationDisplay.ts");
const { DEFAULT_SCHEMAS } = loadTypeScriptModule("src/data/defaultSchemas.ts");

assert.equal(orientationDisplay.getOrientationLabelForClassId("directional", 0), "left");
assert.equal(orientationDisplay.getOrientationLabelForClassId("directional", 1), "right");
assert.equal(
  orientationDisplay.getOrientationLabelForClassId("bilateral", 1, "vertical_obb"),
  "down"
);
assert.equal(orientationDisplay.getOrientationLabelForClassId("axial", 0), "up");
assert.equal(orientationDisplay.getOrientationLabelForClassId("axial", 1), null);
assert.equal(orientationDisplay.getOrientationLabelForClassId("invariant", 1), null);
assert.equal(orientationDisplay.getClassIdForOrientationLabel("axial", "down"), 0);
assert.equal(orientationDisplay.getClassIdForOrientationLabel("invariant", "right"), 0);

assert.deepEqual(
  trainingPipeline.resolveTrainingPipelineGate({
    hasActiveSession: true,
    hasFinalizedBoxes: true,
    obbDetectorVerified: false,
  }),
  { showObbStep: true, canTrainObb: true, showLandmarkStep: false }
);

assert.equal(
  hitlReviewUi.resolveHitlReviewUiState({
    hasResults: true,
    hydrated: true,
    edited: true,
    saved: false,
    approved: false,
    committed: false,
  }).key,
  "changes_pending"
);
assert.equal(
  hitlReviewUi.resolveHitlReviewUiState({
    hasResults: true,
    hydrated: true,
    edited: false,
    saved: true,
    approved: true,
    committed: false,
  }).key,
  "approved"
);
assert.equal(
  hitlReviewUi.resolveHitlReviewUiState({
    hasResults: true,
    hydrated: true,
    edited: false,
    saved: true,
    approved: true,
    committed: true,
  }).key,
  "added_to_training"
);
assert.deepEqual(
  trainingPipeline.resolveTrainingPipelineGate({
    hasActiveSession: true,
    hasFinalizedBoxes: false,
    obbDetectorVerified: false,
  }),
  { showObbStep: true, canTrainObb: false, showLandmarkStep: false }
);
assert.deepEqual(
  trainingPipeline.resolveTrainingPipelineGate({
    hasActiveSession: true,
    hasFinalizedBoxes: true,
    obbDetectorVerified: true,
  }),
  { showObbStep: true, canTrainObb: true, showLandmarkStep: true }
);

const makeImage = (id, boxCount) => ({
  id,
  path: `${id}.png`,
  url: `${id}.png`,
  filename: `${id}.png`,
  selectedBoxId: null,
  history: [],
  future: [],
  isFinalized: true,
  boxes: Array.from({ length: boxCount }, (_, index) => ({
    id: index,
    left: 10 + index * 20,
    top: 10,
    width: 12,
    height: 20,
    landmarks: [],
    class_id: index === 0 ? 1 : 0,
    angle: index % 2 === 0 ? 15 : 0,
  })),
});

const imageProfile = { width: 1000, height: 1000 };
const datasetProfile = obb.summarizeObbDatasetProfile(
  [makeImage("a", 2), makeImage("b", 10), makeImage("c", 0)],
  imageProfile
);
assert.equal(datasetProfile.objectCount, 12);
assert.equal(datasetProfile.medianObjectsPerImage, 2);
assert.equal(datasetProfile.p90ObjectsPerImage, 10);
assert.ok(datasetProfile.classImbalanceRatio > 4);

const overlappingImage = makeImage("overlap", 2);
overlappingImage.boxes[1] = { ...overlappingImage.boxes[1], left: 12, top: 12 };
const overlapProfile = obb.summarizeObbDatasetProfile([overlappingImage], imageProfile);
assert.equal(overlapProfile.overlapProxyFraction, 1);

const sparseLargeDetection = obb.getRecommendedObbDetectionSettings(
  1000,
  { width: 640, height: 480 },
  { medianObjectsPerImage: 1, p90ObjectsPerImage: 1 }
);
assert.equal(sparseLargeDetection.settings.detectionPreset, "balanced");
assert.doesNotMatch(sparseLargeDetection.summary, /dense/i);

const crowdedDetection = obb.getRecommendedObbDetectionSettings(
  20,
  imageProfile,
  { medianObjectsPerImage: 8, p90ObjectsPerImage: 20 }
);
assert.equal(crowdedDetection.settings.detectionPreset, "recall");
assert.equal(crowdedDetection.settings.maxObjects, 30);

const strongGpuTraining = obb.getRecommendedObbTrainingSettings(
  400,
  { device: "cuda", ramGb: 4, gpuMemoryGb: 12 },
  { width: 1920, height: 1280 },
  { p10ObjectShortSidePx: 20, medianObjectAreaFraction: 0.005 }
);
assert.equal(strongGpuTraining.settings.modelTier, "medium");

const lowVramTraining = obb.getRecommendedObbTrainingSettings(
  400,
  { device: "cuda", ramGb: 64, gpuMemoryGb: 4 },
  { width: 1920, height: 1280 },
  { p10ObjectShortSidePx: 20, medianObjectAreaFraction: 0.005 }
);
assert.equal(lowVramTraining.settings.modelTier, "small");

const landmarks = [
  { index: 2, name: "Tail", category: "tail" },
  { index: 1, name: "Head", category: "head" },
];
const directionalA = {
  mode: "directional",
  targetOrientation: "left",
  headCategories: ["HEAD"],
  anteriorAnchorIds: [1],
  posteriorAnchorIds: [2],
};
const directionalB = {
  ...directionalA,
  headCategories: ["head", "head"],
  anteriorAnchorIds: [1, 1],
};
assert.equal(
  schema.computeSchemaSemanticFingerprint(landmarks, directionalA),
  schema.computeSchemaSemanticFingerprint([...landmarks].reverse(), directionalB)
);
assert.notEqual(
  schema.computeSchemaSemanticFingerprint(landmarks, directionalA),
  schema.computeSchemaSemanticFingerprint(landmarks, { mode: "invariant" })
);
assert.notEqual(
  schema.computeSchemaSemanticFingerprint(landmarks, directionalA),
  schema.computeSchemaSemanticFingerprint(
    [{ ...landmarks[0], required: false }, landmarks[1]],
    directionalA
  )
);
assert.equal(
  JSON.parse(schema.canonicalizeSchemaSemantics(landmarks, directionalA)).landmarks[0].required,
  true
);

const landingSource = fs.readFileSync(
  path.join(projectRoot, "src", "Components", "LandingPage.tsx"),
  "utf8"
);
assert.doesNotMatch(landingSource, /Number\(lm\.index\) === (?:3|12)/);
assert.match(landingSource, /built-in schema's orientation policy is part of its versioned/);
assert.match(
  landingSource,
  /built-in schema's orientation policy is part of its versioned[\s\S]{0,900}await createNewSession\(/
);

const inferenceSource = fs.readFileSync(
  path.join(projectRoot, "src", "Components", "InferencePage.tsx"),
  "utf8"
);
assert.match(
  inferenceSource,
  /m\.modelKind === "landmark"\s*&&\s*\(m\.predictorType === "dlib" \|\| m\.predictorType === "cnn"\)/
);
assert.equal(
  (inferenceSource.match(/wasEdited:\s*false,\s*reviewComplete:\s*false,/g) || []).length,
  2
);
assert.doesNotMatch(inferenceSource, /setDetectionRerunModelKey\(newKey\)/);
assert.match(inferenceSource, /staged\.imagePaths!\[index\]\.contentId/);
assert.match(inferenceSource, /hasCurrentLandmarkInferenceForImageIndex/);
assert.match(inferenceSource, /quarantineFailedLandmarks/);
assert.match(inferenceSource, /detectorProvenance:\s*stampedResult\.detectorProvenance/);
assert.match(inferenceSource, /function stableInferenceImageIdentityKeys\(/);
assert.match(inferenceSource, /sourcePath\s*=\s*String\(image\.sourcePath/);
assert.match(inferenceSource, /contentId\s*=\s*String\(image\.contentId/);
assert.match(inferenceSource, /sourceSha256\s*=\s*String\(image\.sourceSha256/);
assert.match(inferenceSource, /dedupeInferenceImageSelection\(images, newImages\)/);
assert.doesNotMatch(inferenceSource, /new Set\(images\.map\(\(img\) => img\.path\)\)/);
assert.doesNotMatch(inferenceSource, /if \(false &&/);
assert.doesNotMatch(inferenceSource, /previewObbPts/);
assert.match(inferenceSource, /Correct, approve, then add to training\./);
assert.match(inferenceSource, /Approve image/);
assert.match(inferenceSource, /Add approved reviews/);
assert.match(inferenceSource, /No JSON or XML export is required\./);
assert.match(inferenceSource, /Downloads are not used by the HITL training workflow\./);
assert.doesNotMatch(inferenceSource, /Mark Review Complete/);
assert.doesNotMatch(inferenceSource, /Commit to Training Data/);
assert.match(
  inferenceSource,
  /Approval is also the explicit save boundary[\s\S]{0,500}edited:\s*false,[\s\S]{0,250}saved:\s*reviewComplete \? true/
);

const electronMainSource = fs.readFileSync(path.join(projectRoot, "electron", "main.ts"), "utf8");
const preloadSource = fs.readFileSync(path.join(projectRoot, "electron", "preload.ts"), "utf8");
assert.doesNotMatch(electronMainSource, /Try alternate extensions/);
assert.match(electronMainSource, /sha256FileSync\(persistedPath\)/);
assert.doesNotMatch(electronMainSource, /session:save-(?:inference|detection)-correction/);

const supportedContractSources = [
  electronMainSource,
  fs.readFileSync(path.join(projectRoot, "electron", "preload.ts"), "utf8"),
  fs.readFileSync(path.join(projectRoot, "src", "types", "global.d.ts"), "utf8"),
  inferenceSource,
];
for (const source of supportedContractSources) {
  assert.doesNotMatch(source, /yolo_pose/);
}
assert.match(electronMainSource, /resolveTrainedObbDetector\(/);
assert.match(electronMainSource, /detectionArtifactSha256/);
assert.match(electronMainSource, /item\.detectorProvenance\?\.modelId/);
assert.match(electronMainSource, /existing\.version === 1/);
assert.match(electronMainSource, /candidate\.status === "deprecated"\s*&&\s*candidate\.promotion\?\.promoted === true/);
assert.ok((electronMainSource.match(/reason:\s*"manual_override"/g) || []).length >= 2);
assert.ok((electronMainSource.match(/priorActiveModelId/g) || []).length >= 4);
assert.match(electronMainSource, /validateLandmarkPromotionArtifact\(entry\)/);
assert.match(electronMainSource, /validateObbPromotionCandidate\(/);
assert.match(electronMainSource, /prepare_imported_dlib_dataset\.py/);
assert.match(electronMainSource, /ipcMain\.handle\("ml:import-dlib-xml"/);
assert.match(electronMainSource, /requiresMappingConfirmation/);
assert.match(electronMainSource, /Confirm imported landmark mapping/);
assert.match(electronMainSource, /validationMode:\s*validationXml \? "explicit" : "derive"/);
assert.match(electronMainSource, /testMode:\s*testXml \? "explicit" : "derive"/);
// A rolled-back import must not leave orphaned canonicalized crops behind.
assert.match(electronMainSource, /const cropsBefore = new Set<string>\(/);
assert.match(electronMainSource, /if \(cropsBefore\.has\(name\)\) continue;/);

// Revealing a model in the file manager must stay confined to the project root.
assert.match(electronMainSource, /ipcMain\.handle\("shell:show-item-in-folder"/);
assert.match(electronMainSource, /Path is outside the current project root/);
assert.match(preloadSource, /openPath:[\s\S]{0,120}shell:show-item-in-folder/);

// Every IPC handler must be reachable from the renderer: an unexposed handler
// is dead backend surface, and an unhandled channel is a broken bridge.
{
  const handlerChannels = [
    ...electronMainSource.matchAll(/ipcMain\.handle\(\s*"([^"]+)"/g),
  ].map((match) => match[1]);
  const unexposed = handlerChannels.filter(
    (channel) => !preloadSource.includes(`"${channel}"`)
  );
  assert.deepEqual(unexposed, [], `IPC handlers with no preload bridge: ${unexposed}`);
}
// Preflight and train share one gate so the two paths cannot drift; the gate
// itself must require the frozen validation cohort and run the Python verify.
assert.match(electronMainSource, /async function verifyImportedXmlContract\(/);
assert.ok(
  (electronMainSource.match(/verifyImportedXmlContract\(\{/g) || []).length >= 2,
  "imported XML must be verified at preflight and immediately before training"
);
const importedGateStart = electronMainSource.indexOf(
  "async function verifyImportedXmlContract("
);
const importedGate = electronMainSource.slice(
  importedGateStart,
  importedGateStart + 4000
);
assert.match(importedGate, /mode:\s*"verify"/);
assert.match(importedGate, /validation_\$\{modelName\}\.xml not found/);
const obbTrainingStart = electronMainSource.indexOf('ipcMain.handle("ml:train-obb-detector"');
const obbTrainingEnd = electronMainSource.indexOf('"ml:super-annotate"', obbTrainingStart);
assert.ok(obbTrainingStart >= 0 && obbTrainingEnd > obbTrainingStart);
const obbTrainingHandler = electronMainSource.slice(obbTrainingStart, obbTrainingEnd);
assert.match(obbTrainingHandler, /persistExplicitSessionTrainingContract\(sessionDir\)/);
assert.doesNotMatch(obbTrainingHandler, /orientationPolicyConfigured\s*=\s*true/);
assert.match(obbTrainingHandler, /orientation_schema:\s*orientationSchema/);

const importedPreparationSource = fs.readFileSync(
  path.join(projectRoot, "backend", "data", "prepare_imported_dlib_dataset.py"),
  "utf8"
);
assert.match(importedPreparationSource, /class _ImportFileTransaction/);
assert.match(importedPreparationSource, /transaction\.rollback\(\)/);
assert.match(importedPreparationSource, /"explicit_schema_ids"/);
assert.match(importedPreparationSource, /"confirmed_template_order"/);
assert.match(importedPreparationSource, /validationCohortRevision/);
assert.match(importedPreparationSource, /testCohortRevision/);
assert.match(importedPreparationSource, /_canonicalize_cohort\(/);

const globalTypesSource = fs.readFileSync(
  path.join(projectRoot, "src", "types", "global.d.ts"),
  "utf8"
);
assert.match(preloadSource, /importDlibXml:[\s\S]*ml:import-dlib-xml/);
assert.match(globalTypesSource, /importDlibXml:[\s\S]*ImportDlibXmlResult/);

const superAnnotatorSource = fs.readFileSync(
  path.join(projectRoot, "backend", "annotation", "super_annotator.py"),
  "utf8"
);
assert.match(superAnnotatorSource, /"schemaSemanticFingerprint": schema_contract\.get\("semanticFingerprint"\)/);
assert.match(superAnnotatorSource, /"orientationContract": orientation_contract/);

const menuSource = fs.readFileSync(
  path.join(projectRoot, "src", "Components", "Menu.tsx"),
  "utf8"
);
assert.match(menuSource, /candidate retained; active landmark model unchanged/);
assert.match(menuSource, /resolveTrainingPipelineGate\(\{/);
assert.match(menuSource, /canTrainObb=\{trainingPipelineGate\.canTrainObb\}/);

const trainDialogSource = fs.readFileSync(
  path.join(projectRoot, "src", "Components", "PopUp.tsx"),
  "utf8"
);
assert.match(trainDialogSource, /const showLandmarkStep = showObbStep && obbDetectorReady/);
assert.match(trainDialogSource, /\{showLandmarkStep && \(\s*<div className="space-y-4 py-4">/);
assert.doesNotMatch(trainDialogSource, /obbDetectorReady \|\| useImportedXml/);
assert.doesNotMatch(trainDialogSource, /showObbStep \|\| obbDetectorReady/);
assert.match(electronMainSource, /function landmarkTrainingObbGateError\(/);
assert.ok(
  (electronMainSource.match(/landmarkTrainingObbGateError\(/g) || []).length >= 3,
  "verified OBB gating must cover the helper, preflight, and direct landmark training"
);

const registeredModel = {
  modelId: "model-123",
  artifactTag: "artifact-456",
  name: "Friendly name",
  predictorType: "cnn",
};
assert.equal(modelIdentity.getModelKey(registeredModel), "model-123");
assert.equal(modelIdentity.getModelArtifactTag(registeredModel), "artifact-456");
assert.ok(modelIdentity.modelMatchesKey(registeredModel, "Friendly name::cnn"));

assert.equal(
  DEFAULT_SCHEMAS.find((entry) => entry.id === "fly-wing").orientationPolicy.mode,
  "directional"
);
for (const preset of DEFAULT_SCHEMAS.filter((entry) => entry.orientationPolicy?.mode === "directional")) {
  assert.equal(preset.orientationPolicy.targetOrientation, "left");
  assert.equal(preset.orientationPolicy.anteriorAnchorIds, undefined);
  assert.equal(preset.orientationPolicy.posteriorAnchorIds, undefined);
}
const fishPolicy = DEFAULT_SCHEMAS.find((entry) => entry.id === "fish-morphometrics").orientationPolicy;
assert.equal(fishPolicy.mode, "directional");

const customSchemaEditorSource = fs.readFileSync(
  path.join(projectRoot, "src", "Components", "CustomSchemaEditor.tsx"),
  "utf8"
);
assert.doesNotMatch(customSchemaEditorSource, /Select at least one anterior and one posterior anchor/);
assert.doesNotMatch(customSchemaEditorSource, /Anterior Anchors|Posterior Anchors/);

const orientationDisplaySource = fs.readFileSync(
  path.join(projectRoot, "src", "lib", "orientationDisplay.ts"),
  "utf8"
);
assert.doesNotMatch(orientationDisplaySource, /Head \u2192|\u2190 Head|Head \u2193|\u2191 Head/);
assert.match(orientationDisplaySource, /Direction/);

assert.match(
  electronMainSource,
  /ipcMain\.handle\("ml:init-super-annotator"/,
  "the preload initialization request must have a main-process handler"
);
assert.match(electronMainSource, /function isYoloRuntimeReady\(/);
assert.doesNotMatch(
  electronMainSource,
  /await superAnnotator\.send\(\{ cmd: "init" \}\);\s*superAnnotator\.initCompleted = true;/,
  "a failed YOLO-World initialization must not be marked ready"
);

console.log("Frontend contract tests passed.");
