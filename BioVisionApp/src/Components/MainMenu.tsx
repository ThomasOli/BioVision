import React, { useState } from 'react';
import Button from '@mui/material/Button';
import TextField from '@mui/material/TextField';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import Paper from '@mui/material/Paper';

const MainMenu: React.FC = () => {
  const [inputLabel, setInputLabel] = useState('Define Image Step');

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setInputLabel(event.target.value);
  };

  const handleShowHideClick = () => {
    alert('Show/Hide button clicked');
    // Add your logic for Show/Hide button click
  };

  const handleDeleteImageClick = () => {
    alert('Delete Image button clicked');
    // Add your logic for Delete Image button click
  };

  const handleResetClick = () => {
    setInputLabel('Define Image Step');
  };

  return (
    <div>
    <Paper
    elevation={3}
    style={{
      padding: '20px',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center', // Center horizontally
      gap: '10px',
    }}
  >
      <div style={{ display: 'flex', flexDirection: 'row', gap: '10px' }}>
        <Button
          variant="contained"
          color="primary"
          style={{ backgroundColor: 'red' }}
          onClick={handleShowHideClick}
        >
          Show/Hide
        </Button>
        <Button variant="contained" style={{ backgroundColor: 'blue' }}>
          <ArrowBackIcon />
        </Button>
        <TextField
          id="defineImageStep"
          label="Define Image Step"
          variant="outlined"
          value={inputLabel}
          onChange={handleInputChange}
        />
        <Button variant="contained" style={{ backgroundColor: 'blue' }}>
          <ArrowForwardIcon />
        </Button>
        <Button
          variant="contained"
          color="secondary"
          style={{ backgroundColor: 'red' }}
          onClick={handleDeleteImageClick}
        >
          Delete Image
        </Button>
      </div>
      <div style={{ marginTop: '10px', marginRight: '278px'}}>
        <Button variant="contained" onClick={handleResetClick}>
          Reset Step
        </Button>
      </div>
    </Paper>
    </div>
  );
};

export default MainMenu;