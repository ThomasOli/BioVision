import React from 'react';
import Sidebar from './Components/Menu';

const App: React.FC = () => {
  return (
    <div style={{ display: 'flex', height: '100vh' }}>
      <Sidebar>
        {/* Your content for the left side goes here */}
      </Sidebar>
      {/* The rest of your application content goes here */}
    </div>
  );
};

export default App;