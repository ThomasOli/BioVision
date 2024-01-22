import React, { useState } from 'react';
import Menu from './Components/Menu';
import ImageCarousel from './Components/ImageCarousel';


const App: React.FC = () => {
  const [color, setColor] = useState<string>("red");
  const handleColorChange = (selectedColor: string) => {
    setColor(selectedColor);
  };
  return (
    <div style={{display: 'flex', height: '100vh'}}>
      <Menu  onColorChange={handleColorChange}/>
      <div style={{
       paddingLeft: "20vw",
       paddingRight:"20vw",
       
      }}>
        <ImageCarousel color ={color}/>
      </div>
      
    </div>
  );
};

export default App;