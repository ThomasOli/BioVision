import { createContext, useState, useCallback } from "react";

interface Point {
  x: number;
  y: number;
  id: number;
}

interface ImageData {
  id: number;
  url: string;
  labels: Point[];
  history: Point[];
  future: Point[];
}

export const UndoRedoClearContext = createContext<UndoRedoClearContextProps>(
  {} as UndoRedoClearContextProps
);

interface UndoRedoClearContextProps {
  images: ImageData[];
  setImages: React.Dispatch<React.SetStateAction<ImageData[]>>;
  undo: () => void;
  redo: () => void;
  clear: () => void;
  usedClear: boolean;
  setUsedClear: React.Dispatch<React.SetStateAction<boolean>>;
  addPoint: (newPoint: Point) => void;
  setSelectedImage: React.Dispatch<React.SetStateAction<number>>;
  // pushToHistory: (newState: Point) => void;
  points: Point[];
}

export const UndoRedoClearContextProvider = ({
  children,
}: React.PropsWithChildren<{}>) => {
  // CHANGE THIS
  // need to get files from upload / system
  const [images, setImages] = useState<ImageData[]>([
    {
      id: 1,
      url: "https://via.placeholder.com/800x600.png?text=Image+1",
      labels: [],
      history: [],
      future: [],
    },
    {
      id: 2,
      url: "https://via.placeholder.com/800x600.png?text=Image+2",
      labels: [],
      history: [],
      future: [],
    },
  ]);

  const [selectedImage, setSelectedImage] = useState(0);
  const [usedClear, setUsedClear] = useState(false);

  let points = images[selectedImage].labels;

  // const pushToHistory = useCallback((newPoint: Point) => {
  //   const newImages = [...images];

  //   // add point to history
  //   images[selectedImage].history = [
  //     ...newImages[selectedImage].history,
  //     newPoint,
  //   ];

  //   // clear future array
  //   images[selectedImage].future = [];

  //   setImages(newImages);
  // }, []);

  const undo = () => {
    const newImages = [...images];
    let image = newImages[selectedImage];

    if (usedClear) {
      image.future = [];
      image.labels = image.history;

      setUsedClear(false);
      setImages(newImages);
    } else {
      const newRedoPoint = image.history[image.history.length - 1];
      console.log("history is: ", images[selectedImage].history);
      console.log("newRedoPoint is: ", newRedoPoint);

      const newHistory = [...image.history];
      newHistory.splice(-1, 1);
      image.history = newHistory;

      const newFuture = [...image.future, newRedoPoint];
      image.future = newFuture;

      const newPoints = [...image.labels]; // Create a copy of the array
      newPoints.splice(-1, 1); // Remove the last element using splice
      image.labels = newPoints;

      // console.log("the new points are", newPoints);
      // console.log("the new history is", newHistory);
      // console.log("the new future is", newFuture);
      setImages(newImages);
    }
  };

  const addPoint = (newPoint: Point) => {
    const updatedImages = [...images];
    let image = updatedImages[selectedImage];

    image.labels.push(newPoint);
    image.history.push(newPoint);

    setImages(updatedImages);
  };

  const redo = () => {
    console.log("redo from MyContext");
    console.log(images[selectedImage].labels);
    const newImages = [...images];
    let image = newImages[selectedImage];

    if (image.future.length === 0) return; // Ensure there is something to redo

    const newUndoPoint = image.future[image.future.length - 1]; // Get the last element from future
    console.log("newUndoPoint is", newUndoPoint);

    const newFuture = [...image.future]; // Copy the future array
    newFuture.splice(-1, 1); // Remove the last element from future
    image.future = newFuture; // Update future state

    const newHistory = [...image.history, newUndoPoint]; // Add the redo point back to history
    image.history = newHistory; // Update history state

    const newPoints = [...image.labels, newUndoPoint]; // Add the redo point back to points
    image.labels = newPoints; // Update points state

    // console.log("the new points are", newPoints);
    // console.log("the new history is", newHistory);
    // console.log("the new future is", newFuture);
    setImages(newImages);
  };

  const clear = () => {
    console.log("clear from MyContext");
    console.log(images[selectedImage].labels);
    const newImages = [...images];
    let image = newImages[selectedImage];

    // Save the current points to history so that undo can bring them back
    const newHistory = [...image.history]; // Add the current points to history
    newHistory.concat(image.labels);
    image.history = newHistory;
    // console.log("history is ", history)

    // Clear the points by setting it to an empty array
    image.labels = [];

    // Optionally reset future, since clearing might represent a new action that prevents redo
    image.future = [];

    // console.log("Points cleared.");
    // console.log("History updated:", newHistory);
    setUsedClear(true);

    setImages(newImages);
  };

  return (
    <UndoRedoClearContext.Provider
      value={{
        images,
        setImages,
        undo,
        redo,
        clear,
        usedClear,
        setUsedClear,
        addPoint,
        setSelectedImage,
        // pushToHistory,
        points,
      }}
    >
      {children}
    </UndoRedoClearContext.Provider>
  );
};
