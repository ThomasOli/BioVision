import React from 'react';
import Menu from './Components/Menu';
import { useState } from 'react';
import MainMenu from './Components/MainMenu';

const App: React.FC = () => {
  return (
    <div style={{display: 'flex', height: '100vh'}}>
      <Menu/>
      <div style={{
       paddingLeft: "20vw",
       paddingRight:"20vw",
      }}><MainMenu/>
      </div>
      
    </div>
  );
};

export default App;