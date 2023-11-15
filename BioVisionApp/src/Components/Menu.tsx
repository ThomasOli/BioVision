import React from 'react';
import Paper from '@mui/material/Paper';
import Button from '@mui/material/Button';
import Box from '@mui/material/Box'
import UploadImages from './UploadImages';
interface MenuProps {
  children: React.ReactNode;
}

const Menu: React.FC<MenuProps> = ({ children }) => {
  return (
    <Paper style={{ flex: '0 0 100%', height: '100vh', padding: '16px' }}>
      {children}
      <div style={{ marginTop: '16px' }}>
        {/* Add your sidebar buttons here */}
        <UploadImages></UploadImages>
        <Box>
        <Button variant="contained" color="primary">
          Button 1
        </Button>
        <Button variant="contained" color="primary" style={{ marginTop: '8px' }}>
          Button 2
        </Button>
        </Box>
        {/* Add more buttons as needed */}
      </div>
    </Paper>
  );
};

export default Menu;