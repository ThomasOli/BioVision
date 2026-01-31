# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BioVision is an Electron desktop application for training and using dlib shape predictor models on images. Users annotate images with landmark points, train machine learning models, and run inference on new images.

## Development Commands

```bash
npm run dev      # Start Vite dev server with Electron
npm run build    # TypeScript check + Vite build + electron-builder packaging
npm run lint     # ESLint check on src/ directory
npm run preview  # Preview production build
```

## Architecture

### Three-Layer Stack

1. **React Frontend** (`src/`) - UI with Konva.js canvas for image annotation
2. **Electron Main Process** (`electron/main.ts`) - IPC handlers, file operations, Python process spawning
3. **Python Backend** (`backend/`) - dlib model training and inference

### Key Data Flow

- User annotates images → Redux stores landmarks → IPC saves to disk → Python trains model
- Inference: IPC calls Python predict script → returns landmarks → rendered on canvas

### IPC API (window.api)

Defined in `src/types/global.d.ts`, exposed via `electron/preload.ts`:
- `saveLabels(images)` - Save annotated images and JSON labels to project directory
- `trainModel(modelName)` - Run prepare_dataset.py then train_shape_model.py
- `predictImage(imagePath, tag)` - Run inference with trained model
- `selectImageFolder()` - Open folder dialog for image import
- `getProjectRoot()` / `selectProjectRoot()` - Manage project storage directory

### State Management

- Redux Toolkit with single slice (`src/state/filesState/fileSlice.tsx`)
- `AnnotatedImage` type: id, path, url, filename, labels (Point[]), history/future for undo/redo
- redux-persist for state persistence

### Python Backend

Requires Python with dlib and OpenCV. Scripts in `backend/`:
- `prepare_dataset.py` - Converts JSON labels to dlib XML format
- `train_shape_model.py` - Trains dlib shape predictor
- `predict.py` - Runs inference using trained .dat model

Project data stored in user-configurable directory (default: `userData/training-model/`):
- `images/` - Copied source images
- `labels/` - JSON landmark files
- `xml/` - dlib training XML
- `models/` - Trained .dat predictor files

### UI Components

- `ImageLabeler.tsx` - Main canvas for placing/editing landmarks
- `MagnifiedZoomLabeler.tsx` - Precision zoomed view
- `ImageLablerCarousel.tsx` - Image navigation
- `PopUp.tsx` - Model training dialog
- `Menu.tsx` - Left sidebar
- Uses Radix UI primitives in `src/Components/ui/`

## Tech Stack

- React 18 + TypeScript + Vite
- Electron 26 with context isolation
- Tailwind CSS 3.4 + Radix UI
- Konva.js / react-konva for canvas
- Redux Toolkit + redux-persist
