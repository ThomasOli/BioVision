// src/state/store.ts
import { configureStore, combineReducers } from '@reduxjs/toolkit';
import { persistStore, persistReducer, FLUSH, REHYDRATE, PAUSE, PERSIST, PURGE, REGISTER } from 'redux-persist';
import storage from 'redux-persist/lib/storage';
import fileReducer from './filesState/fileSlice';
import speciesReducer from './speciesState/speciesSlice';

const persistConfig = {
  key: 'biovision-root',
  version: 2, // Bump version to invalidate old persisted state
  storage,
  whitelist: ['species'], // Only persist species - blob URLs can't survive reload
};

const rootReducer = combineReducers({
  files: fileReducer,
  species: speciesReducer,
});

const persistedReducer = persistReducer(persistConfig, rootReducer);

const store = configureStore({
  reducer: persistedReducer,
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: [FLUSH, REHYDRATE, PAUSE, PERSIST, PURGE, REGISTER],
      },
    }),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

export const persistor = persistStore(store);
export default store;
