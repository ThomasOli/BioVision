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

// Preserve any user-owned boxes when refreshing model detections.
// Only raw model predictions are replaceable.
const isReplaceablePredictedBox = (box: BoundingBox): boolean => box.source === "predicted";

// Detected box from multi-specimen detection
interface DetectedBoxData {
  left: number;
  top: number;
  width: number;
  height: number;
  confidence?: number;
  class_name?: string;
}

// SuperAnnotator result object (from pipeline)
interface SuperAnnotateObjectData {
  box: {
    left: number;
    top: number;
    right: number;
    bottom: number;
    width: number;
    height: number;
    confidence: number;
    class_name: string;
  };
  mask_outline: [number, number][];
  landmarks: { id: number; x: number; y: number }[];
  confidence: number;
  class_name: string;
  instance_metadata: {
    center: [number, number];
    crop_origin: [number, number];
    crop_size: [number, number];
    rotation: number;
    scale: number;
  };
  detection_method: string;
  class_id?: number;
  orientation_hint?: {
    orientation?: "left" | "right";
    confidence?: number;
    source?: string;
  };
  obb?: {
    angle: number;
    corners: [number, number][];
    center: [number, number];
    size: [number, number];
  } | null;
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
  setBoxesFromDetection: (detectedBoxes: DetectedBoxData[]) => void;
  setBoxesFromSuperAnnotation: (objects: SuperAnnotateObjectData[]) => void;
  flipAllBoxOrientations: () => void;
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
  const speciesList = useSelector((state: RootState) => state.species.species);

  const [images, setImages] = useState<AnnotatedImage[]>([]);
  const [selectedImage, setSelectedImageState] = useState<number>(0);
  const dirtyImageIds = useRef<Set<number>>(new Set());
  const autoSaveTimeoutRef = useRef<ReturnType<typeof setTimeout>>();

  // Minimum landmark index for the active schema (0 for 0-based schemas, 1 for 1-based)
  const schemaMinIndex = useMemo(() => {
    if (!activeSpeciesId) return 0;
    const activeSpecies = speciesList.find((s) => s.id === activeSpeciesId);
    const template = activeSpecies?.landmarkTemplate;
    if (!template?.length) return 0;
    return Math.min(...template.map((lm) => lm.index));
  }, [activeSpeciesId, speciesList]);

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
      // Propagate field updates (e.g. isFinalized) from Redux into existing context images
      const updatedImages = prevImages
        .filter((img) => relevantIds.has(img.id))
        .map((img) => {
          const fromRedux = relevantFiles.find((f) => f.id === img.id);
          if (!fromRedux) return img;

          const next = { ...img };
          let changed = false;

          const reduxFinalized = Boolean(fromRedux.isFinalized);
          if (Boolean(img.isFinalized) !== reduxFinalized) {
            next.isFinalized = reduxFinalized;
            changed = true;
          }
          if (fromRedux.hasBoxes && !img.hasBoxes) {
            next.hasBoxes = true;
            changed = true;
          }

          // Sync lazy-loaded annotations from Redux into context images.
          // Limit this to empty-context -> populated-redux to avoid clobbering
          // in-context edits that have not been persisted yet.
          const reduxBoxes = fromRedux.boxes || [];
          const contextBoxes = img.boxes || [];
          if (contextBoxes.length === 0 && reduxBoxes.length > 0) {
            next.boxes = reduxBoxes;
            changed = true;
          }

          return changed ? next : img;
        });

      return [...updatedImages, ...newImages];
    });
  }, [fileArray, activeSpeciesId]);

  // Clamp selectedImage whenever images length changes
  useEffect(() => {
    setSelectedImageState((prev) => {
      if (images.length === 0) return 0;
      return Math.min(Math.max(prev, 0), images.length - 1);
    });
  }, [images.length]);

  // Safe derived index
  const safeSelectedIndex = useMemo(() => {
    if (images.length === 0) return 0;
    return Math.min(Math.max(selectedImage, 0), images.length - 1);
  }, [selectedImage, images.length]);

  // Keep a ref to images so the auto-save callback can read current state
  // without needing images in the auto-save effect's dependency array.
  const imagesRef = useRef(images);
  useEffect(() => { imagesRef.current = images; }, [images]);

  const activeSpeciesIdRef = useRef(activeSpeciesId);
  useEffect(() => { activeSpeciesIdRef.current = activeSpeciesId; }, [activeSpeciesId]);

  // Mark the current image as dirty and schedule a debounced auto-save.
  // Scheduling here (not in a useEffect) means the timer only resets when
  // an actual edit happens — not on every render caused by state changes.
  const markDirty = useCallback((imageId: number) => {
    dirtyImageIds.current.add(imageId);
    clearTimeout(autoSaveTimeoutRef.current);
    autoSaveTimeoutRef.current = setTimeout(async () => {
      if (!activeSpeciesIdRef.current) return;
      const idsToSave = new Set(dirtyImageIds.current);
      dirtyImageIds.current.clear();
      for (const img of imagesRef.current) {
        if (!idsToSave.has(img.id) || !img.speciesId) continue;
        try {
          await window.api.sessionSaveAnnotations(img.speciesId, img.filename, img.boxes);
        } catch (err) {
          console.error(`Auto-save failed for ${img.filename}:`, err);
        }
      }
    }, 2000);
  }, []);

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

  const setSelectedImage = useCallback<React.Dispatch<React.SetStateAction<number>>>((value) => {
    setImages((prevImages) => {
      let changed = false;
      const nextImages = prevImages.map((img) => {
        if (img.selectedBoxId === null) return img;
        changed = true;
        return { ...img, selectedBoxId: null };
      });
      return changed ? nextImages : prevImages;
    });
    setSelectedImageState(value);
  }, []);

  // Helper to save snapshot before change (also marks image dirty for auto-save)
  const saveSnapshot = useCallback(
    (activeImage: AnnotatedImage): AnnotatedImage => {
      markDirty(activeImage.id);
      return {
        ...activeImage,
        history: [...(activeImage.history ?? []), cloneBoxes(activeImage.boxes)].slice(-30),
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

  const flipAllBoxOrientations = useCallback(() => {
    setImages((prevImages) => {
      return prevImages.map((img) => {
        if (!img.boxes?.length) return img;
        const flippedBoxes = img.boxes.map((box) => {
          const currentId = box.class_id !== undefined ? box.class_id : 0;
          const nextId = currentId === 0 ? 1 : 0;
          return {
            ...box,
            class_id: nextId,
            orientation_hint: box.orientation_hint
              ? { ...box.orientation_hint, orientation: nextId === 0 ? "left" as const : "right" as const }
              : undefined,
          };
        });
        markDirty(img.id);
        return { ...img, boxes: flippedBoxes };
      });
    });
  }, [markDirty]);

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

        // Find next sequential index based on what's already placed,
        // starting from the schema's minimum index (0 for 0-based, 1 for 1-based schemas)
        const existingIndices = new Set(box.landmarks.map((lm) => lm.id));
        let nextIndex = schemaMinIndex;
        while (existingIndices.has(nextIndex)) {
          nextIndex++;
        }

        const newPoint: Point = {
          ...pointData,
          id: nextIndex, // ✅ Sequential index aligned to schema
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
    [selectedImage, saveSnapshot, schemaMinIndex]
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

        // Find next sequential index aligned to the schema's min index
        const existingIndices = new Set(box.landmarks.map((lm) => lm.id));
        let nextIndex = schemaMinIndex;
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
    [selectedImage, saveSnapshot, schemaMinIndex]
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

        // Keep user/manual/corrected boxes; refresh only replaceable predicted boxes.
        const preservedBoxes = cloneBoxes(
          activeImage.boxes.filter((b) => !isReplaceablePredictedBox(b))
        );
        const usedIds = new Set<number>(preservedBoxes.map((b) => b.id));
        let idCursor = Date.now();

        // Convert fresh detections to BoundingBox format with collision-safe IDs.
        const newBoxes: BoundingBox[] = detectedBoxes.map((det) => {
          while (usedIds.has(idCursor)) idCursor += 1;
          const id = idCursor;
          usedIds.add(id);
          idCursor += 1;
          return {
            id,
            left: det.left,
            top: det.top,
            width: det.width,
            height: det.height,
            landmarks: [],
            confidence: det.confidence,
            source: "predicted" as const,
          };
        });

        activeImage.boxes = [...preservedBoxes, ...newBoxes];
        const preferredSelectedId =
          newBoxes[0]?.id ??
          (activeImage.selectedBoxId !== null &&
          activeImage.boxes.some((b) => b.id === activeImage.selectedBoxId)
            ? activeImage.selectedBoxId
            : null) ??
          activeImage.boxes[0]?.id ??
          null;
        activeImage.selectedBoxId = preferredSelectedId;

        newImages[idx] = activeImage;
        return newImages;
      });
    },
    [selectedImage, saveSnapshot]
  );

  // Bulk set boxes from SuperAnnotator pipeline (with landmarks, masks, metadata)
  const setBoxesFromSuperAnnotation = useCallback(
    (objects: SuperAnnotateObjectData[]) => {
      setImages((prevImages) => {
        if (prevImages.length === 0) return prevImages;

        const idx = Math.min(Math.max(selectedImage, 0), prevImages.length - 1);
        const currentImg = prevImages[idx];
        if (!currentImg) return prevImages;

        const newImages = [...prevImages];
        const activeImage = saveSnapshot(currentImg);

        // Keep user/manual/corrected boxes; refresh only replaceable predicted boxes.
        const preservedBoxes = cloneBoxes(
          activeImage.boxes.filter((b) => !isReplaceablePredictedBox(b))
        );
        const usedIds = new Set<number>(preservedBoxes.map((b) => b.id));
        let idCursor = Date.now();

        const newBoxes: BoundingBox[] = objects.map((obj) => {
          while (usedIds.has(idCursor)) idCursor += 1;
          const id = idCursor;
          usedIds.add(id);
          idCursor += 1;
          const classIdFromObject = Number.isFinite(Number(obj.class_id))
            ? Number(obj.class_id)
            : Number.isFinite(Number((obj as unknown as { box?: { class_id?: number } }).box?.class_id))
              ? Number((obj as unknown as { box?: { class_id?: number } }).box?.class_id)
              : undefined;
          return {
            id,
            left: obj.box.left,
            top: obj.box.top,
            width: obj.box.width,
            height: obj.box.height,
            landmarks: obj.landmarks.map((lm) => ({
              x: lm.x,
              y: lm.y,
              id: lm.id,
              isPredicted: true,
            })),
            confidence: obj.confidence,
            source: "predicted" as const,
            maskOutline: obj.mask_outline,
            className: obj.class_name,
            instanceMetadata: obj.instance_metadata,
            detectionMethod: obj.detection_method,
            class_id: classIdFromObject,
            orientation_hint: obj.orientation_hint,
            ...(obj.obb ? {
              angle: obj.obb.angle,
              obbCorners: obj.obb.corners as [number, number][],
            } : {}),
          };
        });

        activeImage.boxes = [...preservedBoxes, ...newBoxes];
        const preferredSelectedId =
          newBoxes[0]?.id ??
          (activeImage.selectedBoxId !== null &&
          activeImage.boxes.some((b) => b.id === activeImage.selectedBoxId)
            ? activeImage.selectedBoxId
            : null) ??
          activeImage.boxes[0]?.id ??
          null;
        activeImage.selectedBoxId = preferredSelectedId;

        newImages[idx] = activeImage;
        return newImages;
      });
    },
    [selectedImage, saveSnapshot]
  );

  const contextValue = useMemo(() => ({
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
    flipAllBoxOrientations,
    setBoxesFromDetection,
    setBoxesFromSuperAnnotation,
    addLandmark,
    deleteLandmark,
    skipLandmark,
    setSelectedImage,
    points,
  }), [images, setImages, undo, redo, clear, boxes, selectedBoxId,
      addBox, deleteBox, selectBox, updateBox, flipAllBoxOrientations,
      setBoxesFromDetection, setBoxesFromSuperAnnotation,
      addLandmark, deleteLandmark, skipLandmark, setSelectedImage, points]);

  return (
    <UndoRedoClearContext.Provider value={contextValue}>
      {children}
    </UndoRedoClearContext.Provider>
  );
};
