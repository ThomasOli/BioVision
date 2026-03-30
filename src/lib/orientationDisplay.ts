import type {
  BilateralClassAxis,
  BoundingBox,
  OrientationLabel,
  OrientationMode,
  StoredOrientationLabel,
} from "@/types/Image";

export type InternalOrientationLabel = OrientationLabel;
export type DisplayOrientationLabel = OrientationLabel;
export type OrientationRenderMode = "arrow" | "centerline" | "none";
export type ResolvedOrientationArrow = {
  points: [number, number, number, number];
  length: number;
  renderMode: Exclude<OrientationRenderMode, "none">;
};

type OrientationCarrier = {
  class_id?: number | null;
  class_name?: string | null;
  className?: string | null;
  orientation_override?: string | null;
  orientation_hint?: {
    orientation?: string | null;
    confidence?: number | null;
  } | null;
  obbCorners?: [number, number][] | null;
  left: number;
  top: number;
  width: number;
  height: number;
};

function normalizeOrientationMode(mode?: string | null): OrientationMode | undefined {
  if (
    mode === "directional" ||
    mode === "bilateral" ||
    mode === "axial" ||
    mode === "invariant"
  ) {
    return mode;
  }
  return undefined;
}

export function normalizeBilateralClassAxis(
  _axis?: string | null
): BilateralClassAxis {
  return "vertical_obb";
}

function isStoredOrientationLabel(value: string | undefined | null): value is StoredOrientationLabel {
  return value === "left" || value === "right" || value === "up" || value === "down";
}

function getSessionPrimaryOrientation(
  orientationMode: string | undefined | null,
  bilateralClassAxis?: string | null
): StoredOrientationLabel {
  if (normalizeOrientationMode(orientationMode) === "axial") return "up";
  return usesVerticalObbBilateralSemantics(orientationMode, bilateralClassAxis) ? "up" : "left";
}

function getSessionSecondaryOrientation(
  orientationMode: string | undefined | null,
  bilateralClassAxis?: string | null
): StoredOrientationLabel {
  if (normalizeOrientationMode(orientationMode) === "axial") return "down";
  return usesVerticalObbBilateralSemantics(orientationMode, bilateralClassAxis) ? "down" : "right";
}

export function usesVerticalObbBilateralSemantics(
  orientationMode: string | undefined | null,
  bilateralClassAxis?: string | null
): boolean {
  return (
    normalizeOrientationMode(orientationMode) === "bilateral" &&
    normalizeBilateralClassAxis(bilateralClassAxis) === "vertical_obb"
  );
}

export function normalizeOrientationLabelForSession(
  orientationMode: string | undefined | null,
  orientation: string | undefined | null,
  bilateralClassAxis?: string | null
): OrientationLabel {
  const normalizedMode = normalizeOrientationMode(orientationMode);
  if (orientation === "uncertain") return "uncertain";
  if (!isStoredOrientationLabel(orientation)) return "uncertain";
  if (normalizedMode === "axial") {
    if (orientation === "left") return "up";
    if (orientation === "right") return "down";
    return orientation;
  }
  if (usesVerticalObbBilateralSemantics(orientationMode, bilateralClassAxis)) {
    if (orientation === "left") return "up";
    if (orientation === "right") return "down";
    return orientation;
  }
  if (orientation === "up") return "left";
  if (orientation === "down") return "right";
  return orientation;
}

export function getOrientationRenderMode(
  orientationMode: string | undefined | null
): OrientationRenderMode {
  const normalizedMode = normalizeOrientationMode(orientationMode);
  if (!normalizedMode) return "arrow";
  if (normalizedMode === "invariant") return "none";
  if (normalizedMode === "axial") return "centerline";
  return "arrow";
}

export function getOrientationLabelForClassId(
  orientationMode: string | undefined | null,
  classId: number | undefined | null,
  bilateralClassAxis?: string | null
): StoredOrientationLabel | null {
  if (classId !== 0 && classId !== 1) return null;
  return classId === 0
    ? getSessionPrimaryOrientation(orientationMode, bilateralClassAxis)
    : getSessionSecondaryOrientation(orientationMode, bilateralClassAxis);
}

export function getClassIdForOrientationLabel(
  orientationMode: string | undefined | null,
  orientation: string | undefined | null,
  bilateralClassAxis?: string | null
): 0 | 1 | null {
  const normalized = normalizeOrientationLabelForSession(
    orientationMode,
    orientation,
    bilateralClassAxis
  );
  if (normalized === getSessionPrimaryOrientation(orientationMode, bilateralClassAxis)) return 0;
  if (normalized === getSessionSecondaryOrientation(orientationMode, bilateralClassAxis)) return 1;
  return null;
}

export function getOrientationHintForClassId(
  orientationMode: string | undefined | null,
  classId: number | undefined | null,
  bilateralClassAxis?: string | null
): StoredOrientationLabel | null {
  return getOrientationLabelForClassId(orientationMode, classId, bilateralClassAxis);
}

export function getOppositeOrientationLabel(
  orientationMode: string | undefined | null,
  orientation: string | undefined | null,
  bilateralClassAxis?: string | null
): StoredOrientationLabel {
  const normalized = normalizeOrientationLabelForSession(
    orientationMode,
    orientation,
    bilateralClassAxis
  );
  return normalized === getSessionSecondaryOrientation(orientationMode, bilateralClassAxis)
    ? getSessionPrimaryOrientation(orientationMode, bilateralClassAxis)
    : getSessionSecondaryOrientation(orientationMode, bilateralClassAxis);
}

function getOrientationTokenLabel(
  orientationMode: string | undefined | null,
  classToken: string,
  bilateralClassAxis?: string | null
): OrientationLabel {
  if (
    classToken.endsWith("_left") ||
    classToken === "left" ||
    classToken.includes("_left_")
  ) {
    return normalizeOrientationLabelForSession(orientationMode, "left", bilateralClassAxis);
  }
  if (
    classToken.endsWith("_right") ||
    classToken === "right" ||
    classToken.includes("_right_")
  ) {
    return normalizeOrientationLabelForSession(orientationMode, "right", bilateralClassAxis);
  }
  if (
    classToken.endsWith("_up") ||
    classToken === "up" ||
    classToken.includes("_up_")
  ) {
    return normalizeOrientationLabelForSession(orientationMode, "up", bilateralClassAxis);
  }
  if (
    classToken.endsWith("_down") ||
    classToken === "down" ||
    classToken.includes("_down_")
  ) {
    return normalizeOrientationLabelForSession(orientationMode, "down", bilateralClassAxis);
  }
  return "uncertain";
}

export function getOrientationLabelFromBox(
  orientationMode: string | undefined | null,
  box: OrientationCarrier | BoundingBox | undefined | null,
  bilateralClassAxis?: string | null,
  minimumConfidence = 0.35
): OrientationLabel {
  if (!box) return "uncertain";

  const override = normalizeOrientationLabelForSession(
    orientationMode,
    "orientation_override" in box ? box.orientation_override : undefined,
    bilateralClassAxis
  );
  if (override !== "uncertain") return override;
  if ("orientation_override" in box && box.orientation_override === "uncertain") {
    return "uncertain";
  }

  const hintOrientation = normalizeOrientationLabelForSession(
    orientationMode,
    box.orientation_hint?.orientation,
    bilateralClassAxis
  );
  const hintConfidence = Number(box.orientation_hint?.confidence);
  if (
    hintOrientation !== "uncertain" &&
    (!Number.isFinite(hintConfidence) || hintConfidence >= minimumConfidence)
  ) {
    return hintOrientation;
  }

  const classOrientation = getOrientationLabelForClassId(
    orientationMode,
    Number.isFinite(Number(box.class_id)) ? Number(box.class_id) : null,
    bilateralClassAxis
  );
  if (classOrientation) return classOrientation;

  const classToken = String(
    ("class_name" in box ? box.class_name : undefined) || box.className || ""
  )
    .trim()
    .toLowerCase()
    .replace(/[-\s]+/g, "_");
  return getOrientationTokenLabel(orientationMode, classToken, bilateralClassAxis);
}

export function getDisplayOrientationLabel(
  orientationMode: string | undefined | null,
  orientation: InternalOrientationLabel,
  bilateralClassAxis?: string | null
): DisplayOrientationLabel {
  return normalizeOrientationLabelForSession(
    orientationMode,
    orientation,
    bilateralClassAxis
  );
}

export function getOrientationToggleLabel(
  orientationMode: string | undefined | null,
  orientation: StoredOrientationLabel,
  bilateralClassAxis?: string | null
): string {
  const display = getDisplayOrientationLabel(
    orientationMode,
    orientation,
    bilateralClassAxis
  );
  if (display === "up") return "\u2191 Head";
  if (display === "down") return "Head \u2193";
  return display === "left" ? "\u2190 Head" : "Head \u2192";
}

export function getOrientationOptionLabel(
  orientationMode: string | undefined | null,
  orientation: StoredOrientationLabel,
  bilateralClassAxis?: string | null
): string {
  const display = getDisplayOrientationLabel(
    orientationMode,
    orientation,
    bilateralClassAxis
  );
  return display === "up"
    ? "Up"
    : display === "down"
      ? "Down"
      : display === "left"
        ? "Left"
        : "Right";
}

function getBoxCorners(box: OrientationCarrier | BoundingBox): [number, number][] {
  if (Array.isArray(box.obbCorners) && box.obbCorners.length === 4) {
    return box.obbCorners as [number, number][];
  }
  return [
    [box.left, box.top],
    [box.left + box.width, box.top],
    [box.left + box.width, box.top + box.height],
    [box.left, box.top + box.height],
  ];
}

function resolveArrowFromEdge(
  corners: [number, number][],
  edge: "left" | "right" | "top" | "bottom",
  renderMode: Exclude<OrientationRenderMode, "none">
): ResolvedOrientationArrow | null {
  const [cp0, cp1, cp2, cp3] = corners;
  let edgeMidX = 0;
  let edgeMidY = 0;
  if (edge === "left") {
    edgeMidX = (cp3[0] + cp0[0]) / 2;
    edgeMidY = (cp3[1] + cp0[1]) / 2;
  } else if (edge === "right") {
    edgeMidX = (cp1[0] + cp2[0]) / 2;
    edgeMidY = (cp1[1] + cp2[1]) / 2;
  } else if (edge === "top") {
    edgeMidX = (cp0[0] + cp1[0]) / 2;
    edgeMidY = (cp0[1] + cp1[1]) / 2;
  } else {
    edgeMidX = (cp2[0] + cp3[0]) / 2;
    edgeMidY = (cp2[1] + cp3[1]) / 2;
  }

  const centerX = (cp0[0] + cp1[0] + cp2[0] + cp3[0]) / 4;
  const centerY = (cp0[1] + cp1[1] + cp2[1] + cp3[1]) / 4;
  const outLen = Math.hypot(edgeMidX - centerX, edgeMidY - centerY) || 1;
  const nx = (edgeMidX - centerX) / outLen;
  const ny = (edgeMidY - centerY) / outLen;
  const arrowLen = Math.min(Math.max(outLen * 0.6, 14), 40);
  const tipX = edgeMidX + nx * 4;
  const tipY = edgeMidY + ny * 4;
  const tailX = edgeMidX - nx * arrowLen;
  const tailY = edgeMidY - ny * arrowLen;
  const fullLen = Math.hypot(tipX - tailX, tipY - tailY);
  if (fullLen < 24) return null;
  return {
    points: [tailX, tailY, tipX, tipY],
    length: fullLen,
    renderMode,
  };
}

export function getPreviewOrientationArrow(
  orientationMode: string | undefined | null,
  orientation: string | undefined | null,
  box: { left: number; top: number; width: number; height: number },
  bilateralClassAxis?: string | null
): ResolvedOrientationArrow | null {
  const renderMode = getOrientationRenderMode(orientationMode);
  if (renderMode === "none") return null;
  const normalized = normalizeOrientationLabelForSession(
    orientationMode,
    orientation,
    bilateralClassAxis
  );
  if (normalized === "uncertain") return null;
  const corners = getBoxCorners(box as BoundingBox);
  if (normalized === "up") return resolveArrowFromEdge(corners, "top", renderMode);
  if (normalized === "down") return resolveArrowFromEdge(corners, "bottom", renderMode);
  if (normalized === "left") return resolveArrowFromEdge(corners, "left", renderMode);
  return resolveArrowFromEdge(corners, "right", renderMode);
}

export function getBoxOrientationArrow(
  orientationMode: string | undefined | null,
  box: OrientationCarrier | BoundingBox | undefined | null,
  bilateralClassAxis?: string | null,
  minimumConfidence = 0.35
): ResolvedOrientationArrow | null {
  if (!box) return null;
  const renderMode = getOrientationRenderMode(orientationMode);
  if (renderMode === "none") return null;
  const orientation = getOrientationLabelFromBox(
    orientationMode,
    box,
    bilateralClassAxis,
    minimumConfidence
  );
  if (orientation === "uncertain") return null;
  const corners = getBoxCorners(box);
  if (orientation === "up") return resolveArrowFromEdge(corners, "top", renderMode);
  if (orientation === "down") return resolveArrowFromEdge(corners, "bottom", renderMode);
  if (orientation === "left") return resolveArrowFromEdge(corners, "left", renderMode);
  return resolveArrowFromEdge(corners, "right", renderMode);
}
