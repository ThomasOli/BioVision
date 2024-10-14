// src/state/store.ts
import { configureStore } from '@reduxjs/toolkit';
import fileReducer from './filesState/fileSlice';

const store = configureStore({
  reducer: {
    files: fileReducer,
    // ... other reducers
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

export default store;
