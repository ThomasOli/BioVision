// src/App.tsx
import React, { useState } from "react";
import Menu from "./Components/Menu";
import { Grid } from "@mui/material";
import ImageLabelerCarousel from "./Components/ImageLablerCarousel";

import { MyContextProvider } from "./Components/MyContext";

const App: React.FC = () => {
  const [color, setColor] = useState<string>("red");
  const handleColorChange = (selectedColor: string) => {
    setColor(selectedColor);
  };

  const [opacity, setOpacity] = useState<number>(100);
  const handleOpacityChange = (selectedOpacity: number) => {
    setOpacity(selectedOpacity);
  };

  return (
    <MyContextProvider>
      <Grid container sx={{ maxWidth: "100vw", height: "100vh" }}>
        <Grid item xs={4}>
          <Menu
            onOpacityChange={handleOpacityChange}
            onColorChange={handleColorChange}
          />
        </Grid>
        <Grid item xs={8}>
          <ImageLabelerCarousel color={color} opacity={opacity} />
        </Grid>
      </Grid>
    </MyContextProvider>
  );
};

export default App;
