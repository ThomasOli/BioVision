import React, { ChangeEvent, Component } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import ListItem from '@mui/material/ListItem';
import BorderLinearProgress from './BorderLinearProgress';

interface UploadImagesProps {}

interface UploadImagesState {
  currentFile?: File;
  previewImage?: string;
  progress: number;
  message: string;
  isError: boolean;
  imageInfos: ImageInfo[];
}

interface ImageInfo {
  url: string;
  name: string;
}

export default class UploadImages extends Component<UploadImagesProps, UploadImagesState> {
  constructor(props: UploadImagesProps) {
    super(props);

    this.state = {
      currentFile: undefined,
      previewImage: undefined,
      progress: 0,
      message: '',
      isError: false,
      imageInfos: [],
    };
  }

  selectFile = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      const selectedFile = event.target.files[0];

      this.setState({
        currentFile: selectedFile,
        previewImage: URL.createObjectURL(selectedFile),
        progress: 0,
        message: '',
      });
    }
  };

  upload = () => {
    // Implement your upload logic here
  };

  removeImage = () => {
    this.setState({
      currentFile: undefined,
      previewImage: undefined,
      progress: 0,
      message: '',
    });
  };

  render() {
    const { currentFile, previewImage, progress, message, imageInfos, isError } = this.state;

    return (
      <div className="mg20">
        <Box style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <label htmlFor="btn-upload">
            <input
              id="btn-upload"
              name="btn-upload"
              style={{ display: 'none' }}
              type="file"
              accept="image/*"
              onChange={this.selectFile}
            />
            <Button className="btn-choose" variant="outlined" component="span">
              Choose Image
            </Button>
          </label>
         
          <Button
            className="btn-upload"
            color="primary"
            variant="contained"
            component="span"
            disabled={!currentFile}
            onClick={this.upload}
          >
            Upload
          </Button>
          </Box>
          {currentFile && (
            <Box className="my20" display="flex" alignItems="center">
              <Box width="100%" mr={1}>
                <BorderLinearProgress />
              </Box>
              <Box minWidth={35}>
                <Typography variant="body2" color="textSecondary">{`${progress}%`}</Typography>
              </Box>
             
            </Box>
            
          )}
       
       <div className="file-name">{currentFile ? currentFile.name : null}</div>
        {previewImage && (
          <div>
            <img className="preview my20" src={previewImage} alt="" style={{ maxWidth: '100%', maxHeight: '100px' }} />
          </div>
        )}

        {message && (
          <Typography variant="subtitle2" className={`upload-message ${isError ? 'error' : ''}`}>
            {message}
          </Typography>
        )}

        <Typography variant="h6" className="list-header">
          List of Images
        </Typography>
        <ul className="list-group">
          {imageInfos &&
            imageInfos.map((image, index) => (
              <ListItem divider key={index}>
                <img src={image.url} alt={image.name} height="80px" className="mr20" />
                <a href={image.url}>{image.name}</a>
              </ListItem>
            ))}
            
        </ul>
         <Box style={{ display: "flex", justifyContent: "space-between" }}>
          <Button
            className="btn-remove"
            color="secondary"
            variant="contained"
            component="span"
            disabled={!currentFile}
            onClick={this.removeImage}
          >
            Remove
          </Button>
          <Button variant="contained" color="primary">
            Clear All
          </Button>
        </Box>
      </div>
    );
  }
}