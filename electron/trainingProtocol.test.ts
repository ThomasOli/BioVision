import assert from "node:assert/strict";
import test from "node:test";

import { parseLandmarkTrainingPublication } from "./trainingProtocol";

test("parses landmark candidate promotion details from the final protocol markers", () => {
  const parsed = parseLandmarkTrainingPublication([
    "MODEL_ID landmark:old",
    "MODEL_ID landmark:run-2",
    "MODEL_TAG fish__run-2",
    "MODEL_STATUS candidate",
    'PROMOTION_JSON {"promoted":false,"reason":"metric_not_improved","candidateScore":0.12,"baselineScore":0.1}',
  ].join("\n"));

  assert.equal(parsed.modelId, "landmark:run-2");
  assert.equal(parsed.artifactTag, "fish__run-2");
  assert.equal(parsed.modelStatus, "candidate");
  assert.equal(parsed.promotion?.promoted, false);
  assert.equal(parsed.promotion?.reason, "metric_not_improved");
});

test("malformed optional promotion JSON does not discard stable identity markers", () => {
  const parsed = parseLandmarkTrainingPublication([
    "MODEL_ID landmark:run-3",
    "MODEL_TAG fish__run-3",
    "MODEL_STATUS active",
    "PROMOTION_JSON {not-json}",
  ].join("\n"));

  assert.equal(parsed.modelId, "landmark:run-3");
  assert.equal(parsed.modelStatus, "active");
  assert.equal(parsed.promotion, undefined);
});
