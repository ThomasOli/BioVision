import { selectClasses } from "@mui/material";
import { createContext, useState, useReducer, useCallback } from "react";

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
  future: Point[]
}

export const MyContext = createContext<MyContextProps>({} as MyContextProps);

interface MyContextProps {
  images: ImageData[];
  setImages: React.Dispatch<React.SetStateAction<ImageData[]>>;
  undo: () => void;
  redo: () => void;
  clear: () => void;
  usedClear: boolean
  setUsedClear: React.Dispatch<React.SetStateAction<boolean>>
  setPoints2: (selectedImage: number, newPoint: Point) => void
  setSelectedImage: React.Dispatch<React.SetStateAction<number>>
}

export const MyContextProvider = ({
  children,
}: React.PropsWithChildren<{}>) => {
  const [images, setImages] = useState<ImageData[]>([
    {
      id: 1,
      url: "https://via.placeholder.com/800x600.png?text=Image+1",
      labels: [],
      history: [],
      future: []
    },
    {
      id: 2,
      url: "https://via.placeholder.com/800x600.png?text=Image+2",
      labels: [],
      history: [],
      future: []
    },
  ]);

  const [selectedImage, setSelectedImage] = useState(0); // Store past states (e.g., canvas data URL)
  const [usedClear, setUsedClear] = useState(false); // Store past states (e.g., canvas data URL)

  const pushToHistory = useCallback((newState: Point) => {
    const newImages = [...images]

    let history = images[selectedImage].history
    let future = images[selectedImage].future

    history = [...newImages[selectedImage].history, newState]
    future = []

    setImages(newImages)
  }, []);

  const undo = () => {
    console.log("undo from MyContext")
    
    const newImages = [...images]
    let image =  newImages[selectedImage]

    if (usedClear) {
      image.future = []
      image.labels = image.history
      

      setUsedClear(false)
      setImages(newImages)
    } else {
      const newRedoPoint = image.history[image.history.length - 1]
      console.log("history is: ", images[selectedImage].history)
      console.log("newRedoPoint is: ", newRedoPoint)

      const newHistory = [...image.history]
      newHistory.splice(-1, 1)
      image.history = newHistory
      
      const newFuture = [...image.future, newRedoPoint]
      image.future = newFuture
  
      const newPoints = [...image.labels];  // Create a copy of the array
      newPoints.splice(-1, 1);       // Remove the last element using splice
      image.labels = newPoints
  
      // console.log("the new points are", newPoints);
      // console.log("the new history is", newHistory);
      // console.log("the new future is", newFuture);
      setImages(newImages)
    }
  };

  const setPoints2 = ((selectedImage: number, newPoint: Point) => {
    const updatedImages = [...images]
    let image = updatedImages[selectedImage]

    image.labels.push(newPoint)
    image.history.push(newPoint)

    setImages(updatedImages)
  })

  const redo =() => {
    console.log("redo from MyContext")
    console.log(images[selectedImage].labels)
    const newImages = [...images]
    let image =  newImages[selectedImage]

    if (image.future.length === 0) return; // Ensure there is something to redo

    const newUndoPoint = image.future[image.future.length - 1]; // Get the last element from future
    console.log("newUndoPoint is", newUndoPoint);

    const newFuture = [...image.future];  // Copy the future array
    newFuture.splice(-1, 1);        // Remove the last element from future
    image.future = newFuture         // Update future state

    const newHistory = [...image.history, newUndoPoint]; // Add the redo point back to history
    image.history = newHistory                      // Update history state

    const newPoints = [...image.labels, newUndoPoint];   // Add the redo point back to points
    image.labels = newPoints                        // Update points state

    // console.log("the new points are", newPoints);
    // console.log("the new history is", newHistory);
    // console.log("the new future is", newFuture);
    setImages(newImages)
  }

  const clear = () => {
    console.log("clear from MyContext")
    console.log(images[selectedImage].labels)
    const newImages = [...images]
    let image =  newImages[selectedImage]

    // Save the current points to history so that undo can bring them back
    const newHistory = [...image.history];  // Add the current points to history
    newHistory.concat(image.labels)
    image.history = newHistory
    // console.log("history is ", history)

    // Clear the points by setting it to an empty array
    image.labels = []

    // Optionally reset future, since clearing might represent a new action that prevents redo
    image.future = []

    // console.log("Points cleared.");
    // console.log("History updated:", newHistory);
    setUsedClear(true)

    setImages(newImages)
  }

  return (
    <MyContext.Provider value={{ images, setImages, undo, redo, clear, usedClear, setUsedClear, setPoints2, setSelectedImage}}>
      {children}
    </MyContext.Provider>
  );
};
