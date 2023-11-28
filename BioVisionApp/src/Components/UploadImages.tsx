import React, { ChangeEvent, useState, DragEvent, useEffect } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import LinearProgress from '@mui/material/LinearProgress';
import { RootState } from '../state/store';
import { useDispatch, useSelector } from 'react-redux';
import { clearFiles, addFile, removeFile } from '../state/filesState/fileSlice';

interface UploadImagesProps {
}

const UploadImages: React.FC<UploadImagesProps> = (props) => {

  const files = useSelector((state: RootState) => state.files.fileArray);
  const dispatch = useDispatch();

  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('');
  const [isError, setIsError] = useState(false);
  const [disableClear, setDisableClear] = useState(true);

  const handleSelectFiles = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      const filesArray = Array.from(event.target.files);

      //Redux Addition
      dispatch(addFile(filesArray));

      setProgress(0);
      setMessage('');
      setDisableClear(false);
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
        setMessage('Upload completed!');
        setDisableClear(true);
      }
    }, 500);
  };

  const handleRemoveImage = (file: File) => {

    dispatch(removeFile(file.name))
    if (files.length === 0) {
      setDisableClear(true);
    }

  };

  const handleClearAll = () => {

    setDisableClear(true);

    //Redux Addition
    dispatch(clearFiles());
  };

  const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const droppedFiles = event.dataTransfer.files;

    if (droppedFiles && droppedFiles.length > 0) {
      const filesArray = Array.from(droppedFiles);

      //Redux Addition
      dispatch(addFile(filesArray));

      setProgress(0);
      setMessage('');
      setDisableClear(false);
    }
  };

  useEffect(() => {
    setDisableClear(files.length === 0);
  }, [files]);

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
      {files.length > 0 && (
        <div style={{ height: '200px', overflowY: 'scroll', width: '200px', marginBottom: '20px' }}>
          {files.map((file, index) => (
            <div key={index} style={{ display: 'flex', alignItems: 'center', margin: '10px 0' }}>
              <img src={URL.createObjectURL(file)} alt={`Preview ${index}`} style={{ width: '100px', height: '100px' }} />
              <Button
                onClick={() => handleRemoveImage(file)}
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
        <Button
          className="btn-clear"
          color="secondary"
          variant="outlined"
          component="span"
          onClick={handleClearAll}
          disabled={disableClear}
          style={{ marginTop: '10px', marginBottom: '10px' }}
        >
          Clear All
        </Button>
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
        disabled={files.length === 0}
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
