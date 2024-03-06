import { configureStore } from "@reduxjs/toolkit";
import fileReducer from "./filesState/fileSlice";
import canvasReducer from './canvasState/canvasSlice';


export const store = configureStore({
    reducer: {
        files: fileReducer,
        canvas: canvasReducer, 
    },
})

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch  = typeof store.dispatch;
