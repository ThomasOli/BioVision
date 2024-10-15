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
  labelHistory: Point[]
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
    case "clear":
      console.log("clear")
      state.labels = []
      return state
    default:
      throw console.error();
  }
};

export const MyContext = createContext<MyContextProps>({} as MyContextProps);

interface MyContextProps {
  images: ImageData[];
  setImages: React.Dispatch<React.SetStateAction<ImageData[]>>;
  clear: () => void;
  undo: () => void;
}

export const MyContextProvider = ({
  children,
}: React.PropsWithChildren<{}>) => {
  const [images, setImages] = useState<ImageData[]>([
    {
      id: 1,
      url: "https://via.placeholder.com/800x600.png?text=Image+1",
      labels: [],
      labelHistory: []
    },
    {
      id: 2,
      url: "https://via.placeholder.com/800x600.png?text=Image+2",
      labels: [],
      labelHistory: []
    },
  ]);

  const [canvasState, dispatch] = useReducer(canvasStateReducer, images[0]);

  const undo = useCallback(() => dispatch({ type: "undo" }), [dispatch]);
  const clear = useCallback(() => dispatch({ type: "clear" }), [dispatch]);

  return (
    <MyContext.Provider value={{ images, setImages, clear,  undo, }}>
      {children}
    </MyContext.Provider>
  );
};
