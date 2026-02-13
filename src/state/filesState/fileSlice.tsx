// src/state/filesState/fileSlice.ts
import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import { BoundingBox, AnnotatedImage } from "../../types/Image";

interface FilesState {
  fileArray: AnnotatedImage[];
}

export const initialState: FilesState = {
  fileArray: [],
};

const fileSlice = createSlice({
  name: "files",
  initialState,
  reducers: {
    addFiles: (state, action: PayloadAction<File[]>) => {
      const newImages = action.payload.map((file) => ({
        id: Date.now() + Math.random(),
        path: file.path,
        url: URL.createObjectURL(file),
        filename: file.name,
        boxes: [] as BoundingBox[],
        selectedBoxId: null,
        history: [] as BoundingBox[][],
        future: [] as BoundingBox[][],
      }));
      state.fileArray = [...state.fileArray, ...newImages];
    },
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
  },
});

export const { addFiles, removeFile, updateBoxes, clearFiles, setSessionImages, addFilesWithSpecies } =
  fileSlice.actions;

export default fileSlice.reducer;
