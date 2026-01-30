// src/Components/UndoRedoClearContext.tsx
import React, { createContext, useCallback, useEffect, useMemo, useState } from "react";
import { useSelector } from "react-redux";
import { RootState } from "../state/store";
import { Point, AnnotatedImage } from "../types/Image";

interface UndoRedoClearContextProps {
  images: AnnotatedImage[];
  setImages: React.Dispatch<React.SetStateAction<AnnotatedImage[]>>;
  undo: () => void;
  redo: () => void;
  clear: () => void;
  addPoint: (newPoint: Point) => void;
  setSelectedImage: React.Dispatch<React.SetStateAction<number>>;
  points: Point[];
}

export const UndoRedoClearContext = createContext<UndoRedoClearContextProps>({} as UndoRedoClearContextProps);

export const UndoRedoClearContextProvider = ({ children }: React.PropsWithChildren<{}>) => {
  const fileArray = useSelector((state: RootState) => state.files.fileArray);

  const [images, setImages] = useState<AnnotatedImage[]>([]);
  const [selectedImage, setSelectedImage] = useState<number>(0);

  // Keep images in sync with Redux fileArray
  useEffect(() => {
    setImages((prevImages) => {
      const existingIds = new Set(prevImages.map((img) => img.id));

      const newImages: AnnotatedImage[] = fileArray
        .filter((file) => !existingIds.has(file.id))
        .map((file) => ({
          ...file,
          labels: [],
          history: [], // expected Point[][]
          future: [], // expected Point[][]
        }));

      const updatedImages = prevImages.filter((img) => fileArray.some((file) => file.id === img.id));

      return [...updatedImages, ...newImages];
    });
  }, [fileArray]);

  // Clamp selectedImage whenever images length changes
  useEffect(() => {
    setSelectedImage((prev) => {
      if (images.length === 0) return 0;
      return Math.min(Math.max(prev, 0), images.length - 1);
    });
  }, [images.length]);

  // Safe derived index (still useful for callers during transitions)
  const safeSelectedIndex = useMemo(() => {
    if (images.length === 0) return 0;
    return Math.min(Math.max(selectedImage, 0), images.length - 1);
  }, [selectedImage, images.length]);

  // Current points for the active image
  const points = useMemo<Point[]>(() => {
    if (images.length === 0) return [];
    const img = images[safeSelectedIndex];
    return img?.labels ?? [];
  }, [images, safeSelectedIndex]);

  const undo = useCallback(() => {
    setImages((prevImages) => {
      if (prevImages.length === 0) return prevImages;

      const idx = Math.min(Math.max(selectedImage, 0), prevImages.length - 1);
      const currentImg = prevImages[idx];
      if (!currentImg) return prevImages;

      if (!currentImg.history || currentImg.history.length === 0) return prevImages;

      const newImages = [...prevImages];
      const activeImage = { ...currentImg };

      // Last snapshot
      const previousSnapshot = activeImage.history[activeImage.history.length - 1];

      // Move current labels to future (store snapshot copy)
      activeImage.future = [...(activeImage.future ?? []), [...activeImage.labels]];

      // Restore snapshot (copy)
      activeImage.labels = [...previousSnapshot];

      // Pop history
      const newHistory = [...activeImage.history];
      newHistory.pop();
      activeImage.history = newHistory;

      newImages[idx] = activeImage;
      return newImages;
    });
  }, [selectedImage]);

  const redo = useCallback(() => {
    setImages((prevImages) => {
      if (prevImages.length === 0) return prevImages;

      const idx = Math.min(Math.max(selectedImage, 0), prevImages.length - 1);
      const currentImg = prevImages[idx];
      if (!currentImg) return prevImages;

      if (!currentImg.future || currentImg.future.length === 0) return prevImages;

      const newImages = [...prevImages];
      const activeImage = { ...currentImg };

      const restoredSnapshot = activeImage.future[activeImage.future.length - 1];

      // Pop future
      const newFuture = [...activeImage.future];
      newFuture.pop();
      activeImage.future = newFuture;

      // Push current labels to history (snapshot copy)
      activeImage.history = [...(activeImage.history ?? []), [...activeImage.labels]];

      // Restore snapshot (copy)
      activeImage.labels = [...restoredSnapshot];

      newImages[idx] = activeImage;
      return newImages;
    });
  }, [selectedImage]);

  const clear = useCallback(() => {
    setImages((prevImages) => {
      if (prevImages.length === 0) return prevImages;

      const idx = Math.min(Math.max(selectedImage, 0), prevImages.length - 1);
      const currentImg = prevImages[idx];
      if (!currentImg) return prevImages;

      const newImages = [...prevImages];
      const activeImage = { ...currentImg };

      if (activeImage.labels.length > 0) {
        // Save snapshot copy
        activeImage.history = [...(activeImage.history ?? []), [...activeImage.labels]];
      }

      activeImage.labels = [];
      activeImage.future = [];

      newImages[idx] = activeImage;
      return newImages;
    });
  }, [selectedImage]);

  const addPoint = useCallback(
    (newPoint: Point) => {
      setImages((prevImages) => {
        if (prevImages.length === 0) return prevImages;

        const idx = Math.min(Math.max(selectedImage, 0), prevImages.length - 1);
        const currentImg = prevImages[idx];
        if (!currentImg) return prevImages;

        const newImages = [...prevImages];
        const activeImage = { ...currentImg };

        // Save snapshot copy before change
        activeImage.history = [...(activeImage.history ?? []), [...activeImage.labels]];

        // Add point
        activeImage.labels = [...activeImage.labels, newPoint];

        // Clear redo stack
        activeImage.future = [];

        newImages[idx] = activeImage;
        return newImages;
      });
    },
    [selectedImage]
  );

  return (
    <UndoRedoClearContext.Provider
      value={{
        images,
        setImages,
        undo,
        redo,
        clear,
        addPoint,
        setSelectedImage,
        points,
      }}
    >
      {children}
    </UndoRedoClearContext.Provider>
  );
};
