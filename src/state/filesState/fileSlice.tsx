// src/state/filesState/fileSlice.ts
import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface Point {
  x: number;
  y: number;
  id: number;
}

interface ImageData {
  id: number;
  url: string;
  labels: Point[];
  labelHistory: Point[]
}

interface FilesState {
  fileArray: ImageData[];
}

const initialState: FilesState = {
  fileArray: [],
};

const fileSlice = createSlice({
  name: 'files',
  initialState,
  reducers: {
    addFiles: (state, action: PayloadAction<File[]>) => {
      const newImages = action.payload.map((file) => ({
        id: Date.now() + Math.random(), // Generate a unique ID
        url: URL.createObjectURL(file),
        labels: [],
      }));
      state.fileArray = [...state.fileArray, ...newImages];
    },
    removeFile: (state, action: PayloadAction<number>) => {
      const imageToRemove = state.fileArray.find((img) => img.id === action.payload);
      if (imageToRemove) {
        URL.revokeObjectURL(imageToRemove.url); // Revoke URL to free memory
      }
      state.fileArray = state.fileArray.filter((image) => image.id !== action.payload);
    },
    updateLabels: (state, action: PayloadAction<{ id: number; labels: Point[] }>) => {
      const { id, labels } = action.payload;
      const image = state.fileArray.find((img) => img.id === id);
      if (image) {
        image.labels = labels;
      }
    },
    clearFiles: (state) => {
      state.fileArray.forEach((image) => {
        URL.revokeObjectURL(image.url); // Revoke all URLs
      });
      state.fileArray = [];
    },
  },
});

export const { addFiles, removeFile, updateLabels, clearFiles } = fileSlice.actions;

export default fileSlice.reducer;
