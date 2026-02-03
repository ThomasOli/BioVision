# BioVision

BioVision is a cross-platform desktop application for training and using dlib shape predictor models on images. It provides an intuitive interface for annotating images with landmark points, training machine learning models, and running inference on new images.

## Features

- **Image Annotation**: Place and edit landmark points on images using an interactive canvas
- **Magnified Zoom View**: Precision editing with a zoomed view for accurate landmark placement
- **Model Training**: Train dlib shape predictor models directly from annotated data
- **Inference**: Run trained models on new images to predict landmarks automatically
- **Project Management**: Organize images, labels, and models in configurable project directories
- **Undo/Redo Support**: Full history management for annotation changes
- **Image Carousel**: Easy navigation between multiple images in a project

## Prerequisites

### Node.js
- Node.js 18.x or higher
- npm 9.x or higher

### Python
- Python 3.8 or higher
- Required packages:
  - `dlib` - Machine learning toolkit for shape prediction
  - `opencv-python` - Image processing library

Install Python dependencies:
```bash
pip install dlib opencv-python
```

> **Note**: Installing dlib may require CMake and a C++ compiler. On Windows, you may need Visual Studio Build Tools. On macOS, install Xcode Command Line Tools. On Linux, install `build-essential` and `cmake`.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ThomasOli/senior-project-.git
   cd senior-project-
   ```

2. **Install Node.js dependencies**
   ```bash
   npm install
   ```

3. **Verify Python setup**
   ```bash
   python -c "import dlib; import cv2; print('Dependencies OK')"
   ```

## Usage

### Development Mode
Start the application in development mode with hot reload:
```bash
npm run dev
```

### Production Build
Build the application for distribution:
```bash
npm run build
```

### Preview Production Build
```bash
npm run preview
```

### Linting
Run ESLint on the source code:
```bash
npm run lint
```

## Workflow

### 1. Import Images
- Click the folder icon to select a directory containing images
- Images are copied to your project's `images/` directory

### 2. Annotate Landmarks
- Click on the image canvas to place landmark points
- Use the magnified zoom view for precise placement
- Navigate between images using the carousel
- Use undo/redo to correct mistakes

### 3. Save Labels
- Labels are automatically saved as JSON files in the `labels/` directory
- Each image has a corresponding JSON file with landmark coordinates

### 4. Train Model
- Open the training dialog from the menu
- Enter a model name/tag
- Click train to start the training process
- The trained model is saved as `predictor_{tag}.dat` in the `models/` directory

### 5. Run Inference
- Load a trained model
- Select new images for prediction
- The model will automatically place landmarks based on training

## Project Structure

```
biovision/
├── backend/                    # Python ML scripts
│   ├── prepare_dataset.py      # Converts JSON labels to dlib XML format
│   ├── train_shape_model.py    # Trains dlib shape predictor
│   └── predict.py              # Runs inference using trained model
├── electron/                   # Electron main process
│   ├── main.ts                 # IPC handlers, file operations, Python spawning
│   └── preload.ts              # Context bridge for renderer process
├── src/                        # React frontend
│   ├── Components/             # UI components
│   │   ├── ImageLabeler.tsx    # Main annotation canvas
│   │   ├── MagnifiedZoomLabeler.tsx  # Precision zoom view
│   │   ├── ImageLablerCarousel.tsx   # Image navigation
│   │   ├── PopUp.tsx           # Training dialog
│   │   ├── Menu.tsx            # Left sidebar
│   │   └── ui/                 # Radix UI primitives
│   ├── state/                  # Redux state management
│   │   └── filesState/
│   │       └── fileSlice.tsx   # Main application state
│   ├── types/                  # TypeScript type definitions
│   │   └── global.d.ts         # IPC API types
│   ├── hooks/                  # Custom React hooks
│   ├── lib/                    # Utility functions
│   ├── App.tsx                 # Main application component
│   └── main.tsx                # React entry point
├── dist-electron/              # Compiled Electron files (generated)
├── dist/                       # Compiled frontend (generated)
└── package.json
```

### Project Data Directory

User data is stored in a configurable directory (default: `userData/training-model/`):

```
project-root/
├── images/     # Source images (copied on import)
├── labels/     # JSON landmark annotations
├── xml/        # dlib training XML files (generated)
└── models/     # Trained .dat predictor files
```

## Architecture

### Three-Layer Stack

1. **React Frontend** - User interface built with React, TypeScript, and Konva.js for canvas rendering
2. **Electron Main Process** - Handles IPC communication, file system operations, and Python process management
3. **Python Backend** - dlib-based machine learning for model training and inference

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        React Frontend                           │
│  User annotates images → Redux stores landmarks → IPC calls     │
└─────────────────────────┬───────────────────────────────────────┘
                          │ IPC (window.api)
┌─────────────────────────▼───────────────────────────────────────┐
│                    Electron Main Process                        │
│  Saves labels to disk → Spawns Python processes                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │ Child Process
┌─────────────────────────▼───────────────────────────────────────┐
│                      Python Backend                             │
│  prepare_dataset.py → train_shape_model.py → predict.py         │
└─────────────────────────────────────────────────────────────────┘
```

### IPC API

The frontend communicates with the backend through these exposed methods (defined in `window.api`):

| Method | Description |
|--------|-------------|
| `saveLabels(images)` | Save annotated images and JSON labels to project directory |
| `trainModel(modelName)` | Run dataset preparation and model training |
| `predictImage(imagePath, tag)` | Run inference with a trained model |
| `selectImageFolder()` | Open folder dialog for image import |
| `getProjectRoot()` | Get current project storage directory |
| `selectProjectRoot()` | Change project storage directory |

### State Management

- **Redux Toolkit** with a single slice for application state
- **redux-persist** for automatic state persistence
- **AnnotatedImage** type containing:
  - `id` - Unique identifier
  - `path` - File system path
  - `url` - Display URL
  - `filename` - Original filename
  - `labels` - Array of landmark points
  - `history/future` - Undo/redo stacks

## Tech Stack

| Category | Technology |
|----------|------------|
| Frontend | React 18, TypeScript, Vite |
| Desktop | Electron 26 |
| Styling | Tailwind CSS 3.4, Radix UI |
| Canvas | Konva.js, react-konva |
| State | Redux Toolkit, redux-persist |
| ML Backend | Python, dlib, OpenCV |

## Training Parameters

The shape predictor uses these default training options (configurable in `train_shape_model.py`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `tree_depth` | 4 | Depth of regression trees |
| `nu` | 0.1 | Regularization parameter |
| `cascade_depth` | 10 | Number of cascaded regressors |
| `feature_pool_size` | 400 | Number of pixels sampled for features |
| `num_test_splits` | 50 | Splits tested at each tree node |
| `oversampling_amount` | 20 | Data augmentation factor |
| `num_threads` | 4 | Parallel training threads |

## Troubleshooting

### dlib Installation Issues
- **Windows**: Install Visual Studio Build Tools with C++ workload, then `pip install dlib`
- **macOS**: Run `xcode-select --install`, then `pip install dlib`
- **Linux**: Run `sudo apt install build-essential cmake`, then `pip install dlib`

### Model Training Fails
- Ensure you have at least 2-3 annotated images
- Check that all images have the same number of landmarks
- Verify the project directory has write permissions

### Inference Returns Incorrect Landmarks
- Use bounding boxes similar to those used during training
- Ensure the model was trained on similar image types
- Try increasing training data or adjusting training parameters

## License

This project is part of a senior project at the University of Florida.

## Contributors

- Thomas Oliwa
