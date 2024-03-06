import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface CanvasState {
  [imageId: string]: any; // Change 'any' to the appropriate canvas state type
}

const initialState: CanvasState = {};

const canvasSlice = createSlice({
  name: 'canvas',
  initialState,
  reducers: {
    setCanvasState: (state, action: PayloadAction<{ imageId: string; canvasState: any }>) => {
      const { imageId, canvasState } = action.payload;
      state[imageId] = canvasState;
    },
  },
});

export const { setCanvasState } = canvasSlice.actions;
export default canvasSlice.reducer;