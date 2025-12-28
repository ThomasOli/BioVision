import { createContext, useCallback, useState, useEffect } from "react";
import { useSelector } from "react-redux"; // Import useSelector
import { RootState } from "../state/store"; // Adjust the import based on your store setup
import { Point, AnnotatedImage } from "../types/Image";
export const UndoRedoClearContext = createContext<UndoRedoClearContextProps>(
  {} as UndoRedoClearContextProps
);

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

export const UndoRedoClearContextProvider = ({
  children,
}: React.PropsWithChildren<{}>) => {
  const fileArray = useSelector((state: RootState) => state.files.fileArray); // Access fileArray from the Redux store

  let [images, setImages] = useState<AnnotatedImage[]>([]);
  const [selectedImage, setSelectedImage] = useState(0);
  let points = [] as Point[];

  // Add useEffect to update images when fileArray changes

  useEffect(() => {
    setImages((prevImages) => {
      const existingIds = new Set(prevImages.map((image) => image.id));

      // Initialize new files with required properties
      const newImages = fileArray
        .filter((file) => !existingIds.has(file.id))
        .map((file) => ({
          ...file,
          labels: [],
          history: [],
          future: [],
        }));

      // Keep only images that still exist in fileArray
      const updatedImages = prevImages.filter((image) =>
        fileArray.some((file) => file.id === image.id)
      );

      return [...updatedImages, ...newImages];
    });
  }, [fileArray]);

  useEffect(() => {
    // Logic: If selection is out of bounds (e.g. we deleted the selected image), reset it.
    if (selectedImage >= images.length) {
      setSelectedImage(Math.max(0, images.length - 1));
    }
  }, [images.length, selectedImage]);

  if (images.length > 0) points = images[selectedImage].labels;

  const undo = useCallback(() => {
    if (images.length > 0) {
      setImages((prevImages) => {
        const newImages = [...prevImages];

        const activeImage = { ...newImages[selectedImage] };

        if (activeImage.history.length === 0) return prevImages;

        const previousSnapshot =
          activeImage.history[activeImage.history.length - 1];

        activeImage.future = [...activeImage.future, activeImage.labels];

        activeImage.labels = previousSnapshot;

        const newHistory = [...activeImage.history];
        newHistory.pop();
        activeImage.history = newHistory;

        newImages[selectedImage] = activeImage;

        return newImages;
      });
    }
  }, [selectedImage]);

  function addPoint(newPoint: Point) {
    setImages((prevImages) => {
      const updatedImages = [...prevImages];
      const activeImage = { ...updatedImages[selectedImage] };

      activeImage.history = [...activeImage.history, activeImage.labels];

      activeImage.labels = [...activeImage.labels, newPoint];

      activeImage.future = [];

      updatedImages[selectedImage] = activeImage;
      return updatedImages;
    });
  }

  const redo = useCallback(() => {
    if (images.length > 0) {
      setImages((prevImages) => {
        const newImages = [...prevImages];
        const activeImage = { ...newImages[selectedImage] };

        if (activeImage.future.length === 0) return prevImages;
        const itemRestored = activeImage.future[activeImage.future.length - 1];

        const newFuture = [...activeImage.future];
        newFuture.pop();
        activeImage.future = newFuture;

        activeImage.history = [...activeImage.history, activeImage.labels];

        if (Array.isArray(itemRestored)) {
          activeImage.labels = itemRestored;
        } else {
          activeImage.labels = [...activeImage.labels, itemRestored];
        }
        newImages[selectedImage] = activeImage;
        return newImages;
      });
    }
  }, [selectedImage]);

  const clear = useCallback(() => {
    if (images.length > 0) {
      setImages((prevImages) => {
        const newImages = [...prevImages];
        const activeImage = { ...newImages[selectedImage] };

        if (activeImage.labels.length > 0) {
          activeImage.history = [...activeImage.history, activeImage.labels];
        }
        activeImage.labels = [];

        activeImage.future = [];

        newImages[selectedImage] = activeImage;

        return newImages;
      });
    }
  }, [selectedImage]);

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
