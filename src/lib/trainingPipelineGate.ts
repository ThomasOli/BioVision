export interface TrainingPipelineGateInput {
  hasActiveSession: boolean;
  hasFinalizedBoxes: boolean;
  obbDetectorVerified: boolean;
}

export interface TrainingPipelineGate {
  showObbStep: boolean;
  canTrainObb: boolean;
  showLandmarkStep: boolean;
}

/**
 * Training is deliberately sequential: annotations unlock OBB training, and
 * only a verified active OBB artifact unlocks landmark predictor training.
 */
export function resolveTrainingPipelineGate(
  input: TrainingPipelineGateInput
): TrainingPipelineGate {
  return {
    showObbStep: input.hasActiveSession,
    canTrainObb: input.hasActiveSession && input.hasFinalizedBoxes,
    showLandmarkStep: input.hasActiveSession && input.obbDetectorVerified,
  };
}
