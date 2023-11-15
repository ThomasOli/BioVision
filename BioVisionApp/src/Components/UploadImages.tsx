import React, { ChangeEvent, useState, DragEvent } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import LinearProgress from '@mui/material/LinearProgress';

interface UploadImagesProps {
}

const UploadImages: React.FC<UploadImagesProps> = (props) => {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('');
  const [isError, setIsError] = useState(false);
  const [showClearAll, setShowClearAll] = useState(false);

  const handleSelectFiles = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      const filesArray = Array.from(event.target.files);

      const filePreviews = filesArray.map((file) => URL.createObjectURL(file));

      setSelectedFiles(filesArray);
      setPreviews(filePreviews);
      setProgress(0);
      setMessage('');
      setShowClearAll(true);
    }
  };

  const handleUpload = () => {
    // Implement your upload logic here
    // Example: Simulate upload progress
    let currentProgress = 0;
    const interval = setInterval(() => {
      currentProgress += 10;
      setProgress(currentProgress);
      if (currentProgress >= 100) {
        clearInterval(interval);
        setSelectedFiles([]);
        setPreviews([]);
        setMessage('Upload completed!');
        setShowClearAll(false);
      }
    }, 500);
  };

  const handleRemoveImage = (index: number) => {
    const updatedFiles = [...selectedFiles];
    const updatedPreviews = [...previews];

    updatedFiles.splice(index, 1);
    updatedPreviews.splice(index, 1);

    setSelectedFiles(updatedFiles);
    setPreviews(updatedPreviews);

    if (updatedFiles.length === 0) {
      setShowClearAll(false);
    }
  };

  const handleClearAll = () => {
    setSelectedFiles([]);
    setPreviews([]);
    setShowClearAll(false);
  };

  const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const files = event.dataTransfer.files;

    if (files && files.length > 0) {
      const filesArray = Array.from(files);

      const filePreviews = filesArray.map((file) => URL.createObjectURL(file));

      setSelectedFiles(filesArray);
      setPreviews(filePreviews);
      setProgress(0);
      setMessage('');
      setShowClearAll(true);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <Box
        border={2}
        borderRadius={5}
        borderColor="grey.300"
        width="250px"
        height="100px"
        display="flex"
        justifyContent="center"
        alignItems="center"
        mb={2}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        <Typography variant="body1" color="textSecondary">
          Drag & Drop Images Here
        </Typography>
      </Box>
      {previews.length > 0 && (
        <div style={{ height: '200px', overflowY: 'scroll', width: '200px', marginBottom: '20px' }}>
          {previews.map((preview, index) => (
            <div key={index} style={{ display: 'flex', alignItems: 'center', margin: '10px 0' }}>
              <img src={preview} alt={`Preview ${index}`} style={{ width: '100px', height: '100px' }} />
              <Button
                onClick={() => handleRemoveImage(index)}
                variant="outlined"
                sx={{
                  marginLeft: '20px',
                  borderRadius: '5px',
                  backgroundColor: 'transparent',
                  transition: 'background-color 0.2s ease',
                  '&:hover': {
                    backgroundColor: '#ADD8E6',
                  },
                }}
              >
                Remove
              </Button>
            </div>
          ))}
        </div>
      )}
      <label htmlFor="btn-upload">
        <input
          id="btn-upload"
          name="btn-upload"
          style={{ display: 'none' }}
          type="file"
          accept="image/*"
          multiple
          onChange={handleSelectFiles}
        />
        <Button className="btn-choose" variant="outlined" component="span">
          Choose Images
        </Button>
      </label>
      {showClearAll && progress < 100 && (
        <Button
          className="btn-clear"
          color="secondary"
          variant="outlined"
          component="span"
          onClick={handleClearAll}
          style={{ marginTop: '10px', marginBottom: '10px' }}
        >
          Clear All
        </Button>
      )}
      {progress > 0 && progress < 100 && (
        <Box className="my20" width="100%">
          <LinearProgress variant="determinate" value={progress} />
          <Typography variant="body2" color="textSecondary" align="center">{`${progress}%`}</Typography>
        </Box>
      )}
      <Button
        className="btn-upload"
        color="primary"
        variant="contained"
        component="span"
        disabled={selectedFiles.length === 0}
        onClick={handleUpload}
      >
        Upload
      </Button>
      {message && (
        <Typography variant="subtitle2" className={`upload-message ${isError ? 'error' : ''}`}>
          {message}
        </Typography>
      )}
    </div>
  );
};

export default UploadImages;
