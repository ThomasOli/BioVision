// src/state/filesState/fileSlice.ts
import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import {
  AnnotatedImage,
  BoundingBox,
  FinalizePhaseMetadata,
} from "../../types/Image";

interface FilesState {
  fileArray: AnnotatedImage[];
}

const initialState: FilesState = {
  fileArray: [],
};

function normalizeFinalizeFilename(value?: string): string {
  const raw = String(value || "").trim();
  if (!raw) return "";
  const pieces = raw.split(/[\\/]/);
  return (pieces[pieces.length - 1] || raw).toLowerCase();
}

function matchesFinalizeTarget(
  image: AnnotatedImage,
  speciesId?: string,
  filename?: string
): boolean {
  return (
    String(image.speciesId || "").trim() === String(speciesId || "").trim() &&
    normalizeFinalizeFilename(image.filename) === normalizeFinalizeFilename(filename)
  );
}

function isFinalizeInFlight(image: AnnotatedImage): boolean {
  return image.finalizePhase?.state === "queued" || image.finalizePhase?.state === "running";
}

function isTerminalFinalizeState(image: AnnotatedImage): boolean {
  const state = image.finalizePhase?.state;
  return (
    state === "saved" ||
    state === "already_finalized" ||
    state === "finalized_without_segments"
  );
}

const fileSlice = createSlice({
  name: "files",
  initialState,
  reducers: {
    removeFile: (state, action: PayloadAction<number>) => {
      const imageToRemove = state.fileArray.find(
        (img) => img.id === action.payload
      );
      if (imageToRemove) {
        // Defer revocation to allow animations and state sync to complete
        const urlToRevoke = imageToRemove.url;
        setTimeout(() => URL.revokeObjectURL(urlToRevoke), 1000);
      }
      state.fileArray = state.fileArray.filter(
        (image) => image.id !== action.payload
      );
    },
    updateBoxes: (
      state,
      action: PayloadAction<{ id: number; boxes: BoundingBox[] }>
    ) => {
      const { id, boxes } = action.payload;
      const image = state.fileArray.find((img) => img.id === id);
      if (image) {
        image.boxes = boxes;
      }
    },
    clearFiles: (state) => {
      state.fileArray.forEach((image) => {
        URL.revokeObjectURL(image.url);
      });
      state.fileArray = [];
    },
    // Replace entire fileArray when loading a session from disk
    setSessionImages: (state, action: PayloadAction<AnnotatedImage[]>) => {
      state.fileArray.forEach((image) => {
        URL.revokeObjectURL(image.url);
      });
      state.fileArray = action.payload;
    },
    // Add files tagged with a speciesId
    addFilesWithSpecies: (
      state,
      action: PayloadAction<{ files: AnnotatedImage[]; speciesId: string }>
    ) => {
      const tagged = action.payload.files.map((f) => ({
        ...f,
        speciesId: action.payload.speciesId,
      }));
      state.fileArray = [...state.fileArray, ...tagged];
    },
    // Mark a single image as finalized (detection locked, landmark-only)
    setImageFinalized: (state, action: PayloadAction<{ id: number }>) => {
      const image = state.fileArray.find((img) => img.id === action.payload.id);
      if (image) {
        image.isFinalized = true;
      }
    },
    setImageUnfinalized: (state, action: PayloadAction<{ id: number }>) => {
      const image = state.fileArray.find((img) => img.id === action.payload.id);
      if (image) {
        image.isFinalized = false;
      }
    },
    setImagesUnfinalized: (state, action: PayloadAction<{ ids: number[] }>) => {
      const idSet = new Set(action.payload.ids || []);
      if (idSet.size === 0) return;
      state.fileArray.forEach((image) => {
        if (idSet.has(image.id)) {
          image.isFinalized = false;
        }
      });
    },
    setFinalizePhaseForImage: (
      state,
      action: PayloadAction<{
        speciesId: string;
        filename: string;
        finalizePhase: FinalizePhaseMetadata;
        isFinalized?: boolean;
      }>
    ) => {
      const { speciesId, filename, finalizePhase, isFinalized } = action.payload;
      state.fileArray.forEach((image) => {
        if (!matchesFinalizeTarget(image, speciesId, filename)) return;
        image.finalizePhase = finalizePhase;
        if (typeof isFinalized === "boolean") {
          image.isFinalized = isFinalized;
        }
      });
    },
    clearFinalizePhaseForImages: (
      state,
      action: PayloadAction<{ ids: number[] }>
    ) => {
      const idSet = new Set(action.payload.ids || []);
      if (idSet.size === 0) return;
      state.fileArray.forEach((image) => {
        if (idSet.has(image.id)) {
          delete image.finalizePhase;
        }
      });
    },
    // Flip class_id (0/1) and orientation_hint on every box in the entire dataset
    flipAllOrientations: (state) => {
      state.fileArray = state.fileArray.map((img) => ({
        ...img,
        boxes: (img.boxes ?? []).map((box) => {
          const currentId = box.class_id !== undefined ? box.class_id : 0;
          const nextId = currentId === 0 ? 1 : 0;
          return {
            ...box,
            class_id: nextId,
            orientation_hint: box.orientation_hint
              ? {
                  ...box.orientation_hint,
                  orientation: (nextId === 0 ? "left" : "right") as "left" | "right",
                }
              : undefined,
          };
        }),
      }));
    },
  },
});

export const {
  removeFile,
  updateBoxes,
  clearFiles,
  setSessionImages,
  addFilesWithSpecies,
  setImageFinalized,
  setImageUnfinalized,
  setImagesUnfinalized,
  setFinalizePhaseForImage,
  clearFinalizePhaseForImages,
  flipAllOrientations,
} = fileSlice.actions;

export {
  normalizeFinalizeFilename,
  isFinalizeInFlight,
  isTerminalFinalizeState,
};

export const selectTerminalFinalizedImages = (state: { files: FilesState }): AnnotatedImage[] =>
  state.files.fileArray.filter(
    (image) =>
      !isFinalizeInFlight(image) &&
      (isTerminalFinalizeState(image) || Boolean(image.isFinalized))
  );

export const selectTerminalFinalizedImageCount = (
  state: { files: FilesState }
): number => selectTerminalFinalizedImages(state).length;

export const selectHasTerminalFinalizedBoxes = (
  state: { files: FilesState }
): boolean => selectTerminalFinalizedImageCount(state) > 0;

export default fileSlice.reducer;
