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
      console.log(state.fileArray);
    },
    removeFile: (state, action: PayloadAction<number>) => {
      const imageToRemove = state.fileArray.find(
        (img) => img.id === action.payload
      );
      if (imageToRemove) {
        URL.revokeObjectURL(imageToRemove.url);
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
  },
});

export const { addFiles, removeFile, updateBoxes, clearFiles } =
  fileSlice.actions;

export default fileSlice.reducer;
