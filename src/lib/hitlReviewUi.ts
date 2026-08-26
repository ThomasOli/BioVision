export type HitlReviewStatusKey =
  | "awaiting_inference"
  | "restoring"
  | "needs_review"
  | "changes_pending"
  | "draft_saved"
  | "approved"
  | "commit_failed"
  | "added_to_training";

export interface HitlReviewUiState {
  key: HitlReviewStatusKey;
  label: string;
  description: string;
  workflowStep: 0 | 1 | 2 | 3;
}

export interface HitlReviewUiInput {
  hasResults: boolean;
  hydrated: boolean;
  edited: boolean;
  saved: boolean;
  approved: boolean;
  committed: boolean;
  commitFailed?: boolean;
}

/**
 * Resolve persisted HITL flags into one user-facing state. The persistence
 * flags are intentionally kept out of UI copy: users need to know what action
 * comes next, rather than whether an internal draft boolean is set.
 */
export function resolveHitlReviewUiState(
  input: HitlReviewUiInput
): HitlReviewUiState {
  if (!input.hasResults) {
    return {
      key: "awaiting_inference",
      label: "Awaiting inference",
      description: "Run detection and landmark inference before reviewing this image.",
      workflowStep: 0,
    };
  }
  if (!input.hydrated) {
    return {
      key: "restoring",
      label: "Restoring review",
      description: "BioVision is restoring the saved review state for this image.",
      workflowStep: 0,
    };
  }
  if (input.commitFailed) {
    return {
      key: "commit_failed",
      label: "Add failed - retry",
      description: "The approved review was not added to training data. Resolve the error and retry.",
      workflowStep: 2,
    };
  }
  if (input.committed) {
    return {
      key: "added_to_training",
      label: "Added to training",
      description: "This reviewed annotation is part of the schema training set.",
      workflowStep: 3,
    };
  }
  if (input.edited) {
    return {
      key: "changes_pending",
      label: "Edited - needs approval",
      description: "Corrections are saved as a draft. Approve this image when the OBB, direction, and landmarks are correct.",
      workflowStep: 1,
    };
  }
  if (input.approved) {
    return {
      key: "approved",
      label: "Approved - ready to add",
      description: "This image is approved and ready to be added to the schema training set.",
      workflowStep: 2,
    };
  }
  if (input.saved) {
    return {
      key: "draft_saved",
      label: "Draft saved - needs approval",
      description: "The draft is safely stored. Review it, then approve the image.",
      workflowStep: 1,
    };
  }
  return {
    key: "needs_review",
    label: "Needs review",
    description: "Inspect the OBB geometry, direction arrow, and landmarks before approval.",
    workflowStep: 1,
  };
}
