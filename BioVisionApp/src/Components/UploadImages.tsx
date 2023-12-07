import React, { ChangeEvent, useState, DragEvent } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import LinearProgress from '@mui/material/LinearProgress';
import { ipcRenderer } from 'electron';

interface UploadImagesProps {
}

const UploadImages: React.FC<UploadImagesProps> = (props) => {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('');
  const [isError, setIsError] = useState(false);
  const [showClearAll, setShowClearAll] = useState(false);

  const handleSelectFolder = () => {
    ipcRenderer.invoke('open-folder-dialog').then((result) => {
      if (!result.canceled && result.filePaths.length > 0) {
        const folderPath = result.filePaths[0];

        const fs = (window as any).require('fs');

        fs.readdir(folderPath, async (err: NodeJS.ErrnoException | null, files: string[] | Buffer[]) => {
          if (err) {
            console.error(err);
            return;
          }

          const imageFiles = files.filter((file) => /\.(jpg|jpeg|png|gif)$/i.test(file.toString()));

          const filePreviews: string[] = [];
          for (const fileName of imageFiles) {
            const fileData = await fs.promises.readFile(`${folderPath}/${fileName}`);
            const objectUrl = URL.createObjectURL(new Blob([fileData], { type: 'image/jpeg' }));
            filePreviews.push(objectUrl);
          }

          const fileObjects: File[] = await Promise.all(imageFiles.map(async (file) => {
            const fileName = file.toString();
            const fileData = await fs.promises.readFile(`${folderPath}/${fileName}`);
            return new File([fileData], fileName);
          }));

          setSelectedFiles(fileObjects);
          setPreviews(filePreviews);
          setProgress(0);
          setMessage('');
          setShowClearAll(true);
        });
      }
    }).catch((err) => {
      console.error(err);
    });
  };


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
    // Not done yet, needs actual upload to backend system
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

  const handleFolderDrop = async (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const items = event.dataTransfer.items;

    if (items) {
      const newFilesArray: File[] = [];
      let isFolder = false;

      for (let i = 0; i < items.length; i++) {
        const entry = items[i].webkitGetAsEntry();

        if (entry) {
          if (entry.isFile) {
            const file = items[i].getAsFile();
            if (file) {
              newFilesArray.push(file);
            }
          } else if (entry.isDirectory) {
            isFolder = true;
            await processEntry(entry, newFilesArray);
          }
        }
      }

      if (isFolder || newFilesArray.length > 0) {
        const filteredNewFiles = newFilesArray.filter(
          (file) => !selectedFiles.some((existingFile) => existingFile.name === file.name)
        );

        const mergedFiles = [...selectedFiles, ...filteredNewFiles];
        const filePreviews = mergedFiles.map((file) => URL.createObjectURL(file));

        setSelectedFiles(mergedFiles);
        setPreviews(filePreviews);
        setProgress(0);
        setMessage('');
        setShowClearAll(true);
      }
    }
  };

  const processEntry = async (entry: any, filesArray: File[]) => {
    return new Promise<void>((resolve, reject) => {
      if (entry.isFile) {
        entry.file((file: File) => {
          filesArray.push(file);
          resolve();
        });
      } else if (entry.isDirectory) {
        const directoryReader = entry.createReader();
        const readEntries = () => {
          directoryReader.readEntries(async (entries: any[]) => {
            if (entries.length === 0) {
              resolve();
            } else {
              for (let i = 0; i < entries.length; i++) {
                await processEntry(entries[i], filesArray);
              }
              readEntries();
            }
          }, reject);
        };

        readEntries();
      }
    });
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
        onDrop={handleFolderDrop}
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
        className="btn-choose-folder"
        variant="outlined"
        component="span"
        onClick={handleSelectFolder}
      >
        Choose Folder
      </Button>
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
