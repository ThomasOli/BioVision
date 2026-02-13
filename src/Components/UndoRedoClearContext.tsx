// src/Components/UndoRedoClearContext.tsx
import React, { createContext, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSelector } from "react-redux";
import { RootState } from "../state/store";
import { Point, BoundingBox, AnnotatedImage } from "../types/Image";

// Deep clone helper for boxes
const cloneBoxes = (boxes: BoundingBox[]): BoundingBox[] =>
  boxes.map((box) => ({
    ...box,
    landmarks: box.landmarks.map((lm) => ({ ...lm })),
  }));

// Detected box from multi-specimen detection
interface DetectedBoxData {
  left: number;
  top: number;
  width: number;
  height: number;
  confidence?: number;
  class_name?: string;
}

interface UndoRedoClearContextProps {
  images: AnnotatedImage[];
  setImages: React.Dispatch<React.SetStateAction<AnnotatedImage[]>>;
  undo: () => void;
  redo: () => void;
  clear: () => void;
  // Box operations
  boxes: BoundingBox[];
  selectedBoxId: number | null;
  addBox: (box: Omit<BoundingBox, "id" | "landmarks">) => void;
  deleteBox: (boxId: number) => void;
  selectBox: (boxId: number | null) => void;
  updateBox: (boxId: number, updates: Partial<Omit<BoundingBox, "id" | "landmarks">>) => void;
  setBoxesFromDetection: (detectedBoxes: DetectedBoxData[]) => void; // Bulk set from detection
  // Landmark operations (now takes boxId)
  addLandmark: (boxId: number, point: Omit<Point, "id">) => void;
  deleteLandmark: (boxId: number, pointId: number) => void;
  skipLandmark: (boxId: number) => void; // Skip the next landmark in sequence
  setSelectedImage: React.Dispatch<React.SetStateAction<number>>;
  // Legacy points (all landmarks from all boxes, for backward compat)
  points: Point[];
}

export const UndoRedoClearContext = createContext<UndoRedoClearContextProps>({} as UndoRedoClearContextProps);

export const UndoRedoClearContextProvider = ({ children }: React.PropsWithChildren<object>) => {
  const fileArray = useSelector((state: RootState) => state.files.fileArray);
  const activeSpeciesId = useSelector((state: RootState) => state.species.activeSpeciesId);

  const [images, setImages] = useState<AnnotatedImage[]>([]);
  const [selectedImage, setSelectedImage] = useState<number>(0);
  const dirtyImageIds = useRef<Set<number>>(new Set());
  const autoSaveTimeoutRef = useRef<ReturnType<typeof setTimeout>>();

  // Keep images in sync with Redux fileArray, filtered by activeSpeciesId
  useEffect(() => {
    setImages((prevImages) => {
      const relevantFiles = activeSpeciesId
        ? fileArray.filter((f) => f.speciesId === activeSpeciesId)
        : fileArray;

      const existingIds = new Set(prevImages.map((img) => img.id));

      const newImages: AnnotatedImage[] = relevantFiles
        .filter((file) => !existingIds.has(file.id))
        .map((file) => ({
          ...file,
          boxes: file.boxes?.length ? file.boxes : [],
          selectedBoxId: null,
          history: [],
          future: [],
        }));

      const relevantIds = new Set(relevantFiles.map((f) => f.id));
      const updatedImages = prevImages.filter((img) => relevantIds.has(img.id));

      return [...updatedImages, ...newImages];
    });
  }, [fileArray, activeSpeciesId]);

  // Clamp selectedImage whenever images length changes
  useEffect(() => {
    setSelectedImage((prev) => {
      if (images.length === 0) return 0;
      return Math.min(Math.max(prev, 0), images.length - 1);
    });
  }, [images.length]);

  // Safe derived index
  const safeSelectedIndex = useMemo(() => {
    if (images.length === 0) return 0;
    return Math.min(Math.max(selectedImage, 0), images.length - 1);
  }, [selectedImage, images.length]);

  // Mark the current image as dirty (needs auto-save)
  const markDirty = useCallback((imageId: number) => {
    dirtyImageIds.current.add(imageId);
  }, []);

  // Debounced auto-save: persist dirty images' annotations to disk
  useEffect(() => {
    if (!activeSpeciesId || images.length === 0 || dirtyImageIds.current.size === 0) return;

    clearTimeout(autoSaveTimeoutRef.current);
    autoSaveTimeoutRef.current = setTimeout(async () => {
      const idsToSave = new Set(dirtyImageIds.current);
      dirtyImageIds.current.clear();

      for (const img of images) {
        if (!idsToSave.has(img.id)) continue;
        if (!img.speciesId) continue;
        try {
          await window.api.sessionSaveAnnotations(
            img.speciesId,
            img.filename,
            img.boxes
          );
        } catch (err) {
          console.error(`Auto-save failed for ${img.filename}:`, err);
        }
      }
    }, 1000);

    return () => clearTimeout(autoSaveTimeoutRef.current);
  }, [images, activeSpeciesId]);

  // Current boxes for the active image
  const boxes = useMemo<BoundingBox[]>(() => {
    if (images.length === 0) return [];
    const img = images[safeSelectedIndex];
    return img?.boxes ?? [];
  }, [images, safeSelectedIndex]);

  // Current selected box ID
  const selectedBoxId = useMemo<number | null>(() => {
    if (images.length === 0) return null;
    const img = images[safeSelectedIndex];
    return img?.selectedBoxId ?? null;
  }, [images, safeSelectedIndex]);

  // Legacy: all points from all boxes (for backward compat)
  const points = useMemo<Point[]>(() => {
    return boxes.flatMap((box) => box.landmarks);
  }, [boxes]);

  // Helper to save snapshot before change (also marks image dirty for auto-save)
  const saveSnapshot = useCallback(
    (activeImage: AnnotatedImage): AnnotatedImage => {
      markDirty(activeImage.id);
      return {
        ...activeImage,
        history: [...(activeImage.history ?? []), cloneBoxes(activeImage.boxes)],
        future: [],
      };
    },
    [markDirty]
  );

  const undo = useCallback(() => {
    setImages((prevImages) => {
      if (prevImages.length === 0) return prevImages;

      const idx = Math.min(Math.max(selectedImage, 0), prevImages.length - 1);
      const currentImg = prevImages[idx];
      if (!currentImg) return prevImages;

      if (!currentImg.history || currentImg.history.length === 0) return prevImages;

      markDirty(currentImg.id);
      const newImages = [...prevImages];
      const activeImage = { ...currentImg };

      // Last snapshot
      const previousSnapshot = activeImage.history[activeImage.history.length - 1];

      // Move current boxes to future
      activeImage.future = [...(activeImage.future ?? []), cloneBoxes(activeImage.boxes)];

      // Restore snapshot
      activeImage.boxes = cloneBoxes(previousSnapshot);

      // Pop history
      const newHistory = [...activeImage.history];
      newHistory.pop();
      activeImage.history = newHistory;

      // Keep selectedBoxId valid
      if (activeImage.selectedBoxId !== null) {
        const stillExists = activeImage.boxes.some((b) => b.id === activeImage.selectedBoxId);
        if (!stillExists) {
          activeImage.selectedBoxId = activeImage.boxes.length > 0 ? activeImage.boxes[0].id : null;
        }
      }

      newImages[idx] = activeImage;
      return newImages;
    });
  }, [selectedImage, markDirty]);

  const redo = useCallback(() => {
    setImages((prevImages) => {
      if (prevImages.length === 0) return prevImages;

      const idx = Math.min(Math.max(selectedImage, 0), prevImages.length - 1);
      const currentImg = prevImages[idx];
      if (!currentImg) return prevImages;

      if (!currentImg.future || currentImg.future.length === 0) return prevImages;

      markDirty(currentImg.id);
      const newImages = [...prevImages];
      const activeImage = { ...currentImg };

      const restoredSnapshot = activeImage.future[activeImage.future.length - 1];

      // Pop future
      const newFuture = [...activeImage.future];
      newFuture.pop();
      activeImage.future = newFuture;

      // Push current boxes to history
      activeImage.history = [...(activeImage.history ?? []), cloneBoxes(activeImage.boxes)];

      // Restore snapshot
      activeImage.boxes = cloneBoxes(restoredSnapshot);

      // Keep selectedBoxId valid
      if (activeImage.selectedBoxId !== null) {
        const stillExists = activeImage.boxes.some((b) => b.id === activeImage.selectedBoxId);
        if (!stillExists) {
          activeImage.selectedBoxId = activeImage.boxes.length > 0 ? activeImage.boxes[0].id : null;
        }
      }

      newImages[idx] = activeImage;
      return newImages;
    });
  }, [selectedImage, markDirty]);

  const clear = useCallback(() => {
    setImages((prevImages) => {
      if (prevImages.length === 0) return prevImages;

      const idx = Math.min(Math.max(selectedImage, 0), prevImages.length - 1);
      const currentImg = prevImages[idx];
      if (!currentImg) return prevImages;

      markDirty(currentImg.id);
      const newImages = [...prevImages];
      const activeImage = { ...currentImg };

      if (activeImage.boxes.length > 0) {
        // Save snapshot before clearing
        activeImage.history = [...(activeImage.history ?? []), cloneBoxes(activeImage.boxes)];
      }

      activeImage.boxes = [];
      activeImage.selectedBoxId = null;
      activeImage.future = [];

      newImages[idx] = activeImage;
      return newImages;
    });
  }, [selectedImage, markDirty]);

  const addBox = useCallback(
    (boxData: Omit<BoundingBox, "id" | "landmarks">) => {
      setImages((prevImages) => {
        if (prevImages.length === 0) return prevImages;

        const idx = Math.min(Math.max(selectedImage, 0), prevImages.length - 1);
        const currentImg = prevImages[idx];
        if (!currentImg) return prevImages;

        const newImages = [...prevImages];
        const activeImage = saveSnapshot(currentImg);

        const newBox: BoundingBox = {
          ...boxData,
          id: Date.now(),
          landmarks: [],
        };

        activeImage.boxes = [...activeImage.boxes, newBox];
        activeImage.selectedBoxId = newBox.id;

        newImages[idx] = activeImage;
        return newImages;
      });
    },
    [selectedImage, saveSnapshot]
  );

  const deleteBox = useCallback(
    (boxId: number) => {
      setImages((prevImages) => {
        if (prevImages.length === 0) return prevImages;

        const idx = Math.min(Math.max(selectedImage, 0), prevImages.length - 1);
        const currentImg = prevImages[idx];
        if (!currentImg) return prevImages;

        const newImages = [...prevImages];
        const activeImage = saveSnapshot(currentImg);

        activeImage.boxes = activeImage.boxes.filter((b) => b.id !== boxId);

        // Update selectedBoxId if deleted
        if (activeImage.selectedBoxId === boxId) {
          activeImage.selectedBoxId = activeImage.boxes.length > 0 ? activeImage.boxes[0].id : null;
        }

        newImages[idx] = activeImage;
        return newImages;
      });
    },
    [selectedImage, saveSnapshot]
  );

  const selectBox = useCallback(
    (boxId: number | null) => {
      setImages((prevImages) => {
        if (prevImages.length === 0) return prevImages;

        const idx = Math.min(Math.max(selectedImage, 0), prevImages.length - 1);
        const currentImg = prevImages[idx];
        if (!currentImg) return prevImages;

        // Don't save snapshot for selection changes
        const newImages = [...prevImages];
        const activeImage = { ...currentImg };
        activeImage.selectedBoxId = boxId;

        newImages[idx] = activeImage;
        return newImages;
      });
    },
    [selectedImage]
  );

  const updateBox = useCallback(
    (boxId: number, updates: Partial<Omit<BoundingBox, "id" | "landmarks">>) => {
      setImages((prevImages) => {
        if (prevImages.length === 0) return prevImages;

        const idx = Math.min(Math.max(selectedImage, 0), prevImages.length - 1);
        const currentImg = prevImages[idx];
        if (!currentImg) return prevImages;

        const newImages = [...prevImages];
        const activeImage = saveSnapshot(currentImg);

        activeImage.boxes = activeImage.boxes.map((box) => {
          if (box.id === boxId) {
            return { ...box, ...updates };
          }
          return box;
        });

        newImages[idx] = activeImage;
        return newImages;
      });
    },
    [selectedImage, saveSnapshot]
  );

  const addLandmark = useCallback(
    (boxId: number, pointData: Omit<Point, "id">) => {
      setImages((prevImages) => {
        if (prevImages.length === 0) return prevImages;

        const idx = Math.min(Math.max(selectedImage, 0), prevImages.length - 1);
        const currentImg = prevImages[idx];
        if (!currentImg) return prevImages;

        const newImages = [...prevImages];
        const activeImage = saveSnapshot(currentImg);

        // Find the box we're adding to
        const box = activeImage.boxes.find((b) => b.id === boxId);
        if (!box) return prevImages;

        // Find next sequential index (0, 1, 2...) based on what's already placed
        const existingIndices = new Set(box.landmarks.map((lm) => lm.id));
        let nextIndex = 0;
        while (existingIndices.has(nextIndex)) {
          nextIndex++;
        }

        const newPoint: Point = {
          ...pointData,
          id: nextIndex, // ✅ Sequential index, not timestamp
        };

        activeImage.boxes = activeImage.boxes.map((b) => {
          if (b.id === boxId) {
            return {
              ...b,
              landmarks: [...b.landmarks, newPoint],
            };
          }
          return b;
        });

        // Auto-select the box where landmark was added
        activeImage.selectedBoxId = boxId;

        newImages[idx] = activeImage;
        return newImages;
      });
    },
    [selectedImage, saveSnapshot]
  );

  const deleteLandmark = useCallback(
    (boxId: number, pointId: number) => {
      setImages((prevImages) => {
        if (prevImages.length === 0) return prevImages;

        const idx = Math.min(Math.max(selectedImage, 0), prevImages.length - 1);
        const currentImg = prevImages[idx];
        if (!currentImg) return prevImages;

        const newImages = [...prevImages];
        const activeImage = saveSnapshot(currentImg);

        activeImage.boxes = activeImage.boxes.map((box) => {
          if (box.id === boxId) {
            return {
              ...box,
              landmarks: box.landmarks.filter((lm) => lm.id !== pointId),
            };
          }
          return box;
        });

        newImages[idx] = activeImage;
        return newImages;
      });
    },
    [selectedImage, saveSnapshot]
  );

  // Skip a landmark (marks it as skipped so it doesn't need to be placed)
  const skipLandmark = useCallback(
    (boxId: number) => {
      setImages((prevImages) => {
        if (prevImages.length === 0) return prevImages;

        const idx = Math.min(Math.max(selectedImage, 0), prevImages.length - 1);
        const currentImg = prevImages[idx];
        if (!currentImg) return prevImages;

        const newImages = [...prevImages];
        const activeImage = saveSnapshot(currentImg);

        // Find the box we're adding to
        const box = activeImage.boxes.find((b) => b.id === boxId);
        if (!box) return prevImages;

        // Find next sequential index (0, 1, 2...) based on what's already placed
        const existingIndices = new Set(box.landmarks.map((lm) => lm.id));
        let nextIndex = 0;
        while (existingIndices.has(nextIndex)) {
          nextIndex++;
        }

        // Create a skipped landmark placeholder (coordinates don't matter)
        const skippedPoint: Point = {
          x: -1,
          y: -1,
          id: nextIndex,
          isSkipped: true,
        };

        activeImage.boxes = activeImage.boxes.map((b) => {
          if (b.id === boxId) {
            return {
              ...b,
              landmarks: [...b.landmarks, skippedPoint],
            };
          }
          return b;
        });

        // Auto-select the box
        activeImage.selectedBoxId = boxId;

        newImages[idx] = activeImage;
        return newImages;
      });
    },
    [selectedImage, saveSnapshot]
  );

  // Bulk set boxes from detection
  const setBoxesFromDetection = useCallback(
    (detectedBoxes: DetectedBoxData[]) => {
      setImages((prevImages) => {
        if (prevImages.length === 0) return prevImages;

        const idx = Math.min(Math.max(selectedImage, 0), prevImages.length - 1);
        const currentImg = prevImages[idx];
        if (!currentImg) return prevImages;

        const newImages = [...prevImages];
        const activeImage = saveSnapshot(currentImg);

        // Convert detected boxes to BoundingBox format
        const newBoxes: BoundingBox[] = detectedBoxes.map((det, i) => ({
          id: Date.now() + i,
          left: det.left,
          top: det.top,
          width: det.width,
          height: det.height,
          landmarks: [],
          confidence: det.confidence,
          source: "predicted" as const,
        }));

        activeImage.boxes = newBoxes;
        activeImage.selectedBoxId = newBoxes.length > 0 ? newBoxes[0].id : null;

        newImages[idx] = activeImage;
        return newImages;
      });
    },
    [selectedImage, saveSnapshot]
  );

  return (
    <UndoRedoClearContext.Provider
      value={{
        images,
        setImages,
        undo,
        redo,
        clear,
        boxes,
        selectedBoxId,
        addBox,
        deleteBox,
        selectBox,
        updateBox,
        setBoxesFromDetection,
        addLandmark,
        deleteLandmark,
        skipLandmark,
        setSelectedImage,
        points,
      }}
    >
      {children}
    </UndoRedoClearContext.Provider>
  );
};
