import React from 'react';
import Menu from './Components/Menu';

import MainMenu from './Components/MainMenu'; // Import MainMenu component
=======
import { useState } from 'react';



const App: React.FC = () => {
  return (
    <div style={{ display: 'flex', height: '100vh' }}>
      <Menu>
        {/* Your content for the left side goes here */}
      </Menu>
      {/* The rest of your application content goes here */}
      <MainMenu />
    </div>
  );
};

export default App;
