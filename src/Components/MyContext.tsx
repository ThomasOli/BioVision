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
}

interface Action {
  type: "save" | "undo" | "redo" | "clear";
}

export const canvasStateReducer = (
  state: ImageData,
  action: Action
): ImageData => {
  switch (action.type) {
    case "undo":
      console.log(state);
      return state;
    default:
      throw console.error();
  }
};
// Create a context
export const MyContext = createContext<MyContextProps>({} as MyContextProps);

interface MyContextProps {
  images: ImageData[];
  setImages: React.Dispatch<React.SetStateAction<ImageData[]>>;
  undo: () => void;
  setPoints: React.Dispatch<React.SetStateAction<Point[]>>;
}

export const MyContextProvider = ({
  children,
}: React.PropsWithChildren<{}>) => {
  const [images, setImages] = useState<ImageData[]>([
    {
      id: 1,
      url: "https://via.placeholder.com/800x600.png?text=Image+1",
      labels: [],
    },
    {
      id: 2,
      url: "https://via.placeholder.com/800x600.png?text=Image+2",
      labels: [],
    },
  ]);

  const [canvasState, dispatch] = useReducer(canvasStateReducer, images[0]);
  const [points, setPoints] = useState<Point[]>([]);

  const undo = useCallback(() => dispatch({ type: "undo" }), [dispatch]);

  return (
    <MyContext.Provider value={{ images, setImages, undo, setPoints }}>
      {children}
    </MyContext.Provider>
  );
};
