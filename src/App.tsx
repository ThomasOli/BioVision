import React, { useState } from 'react';
import Menu from './Components/Menu';
import {Grid, Stack} from '@mui/material'
import ImageCarousel from './Components/ImageCarousel';


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
    <Grid container sx={{maxWidth: "100vw"}}>
      <Grid item xs={4}>
        <Menu  onOpacityChange = {handleOpacityChange} onColorChange={handleColorChange}/>
      </Grid>
      <Grid item xs={8}>
        <ImageCarousel color ={color} opacity={opacity}/>
      </Grid>
      
    </Grid>
  );
};

export default App;