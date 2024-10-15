// src/Components/ImageLabelerCarousel.tsx
import React, { useCallback, useContext, useEffect, useState } from 'react';
import ImageLabeler from './ImageLabeler';
import { useSelector, useDispatch } from 'react-redux';
import { RootState, AppDispatch } from '../state/store';
import { removeFile, updateLabels } from '../state/filesState/fileSlice';
import { Button, IconButton, Box, Typography, Tooltip } from '@mui/material';
import ArrowBackIosIcon from '@mui/icons-material/ArrowBackIos';
import ArrowForwardIosIcon from '@mui/icons-material/ArrowForwardIos';
import DeleteIcon from '@mui/icons-material/Delete';
import ZoomInIcon from '@mui/icons-material/ZoomIn';
import MagnifiedImageLabeler from './MagnifiedZoomLabeler';
interface ImageLabelerCarouselProps {
  color: string;
  opacity: number;
}

interface Point {
  x: number;
  y: number;
  id: number;
}

interface ImageData {
  id: number;
  url: string;
  labels: Point[];
  labelHistory: Point[]
}

const ImageLabelerCarousel: React.FC<ImageLabelerCarouselProps> = ({ color, opacity }) => {
  // CHANGE THIS
  // const images = useSelector((state: RootState) => state.files.fileArray);
  const images = [
    {
      id: 1,
      url: "https://via.placeholder.com/800x600.png?text=Image+1",
      labels: [],
      labelHistory: []
    },
    {
      id: 2,
      url: "https://via.placeholder.com/800x600.png?text=Image+2",
      labels: [],
      labelHistory: []
    },
  ];


  const dispatch = useDispatch<AppDispatch>();

  const [currentIndex, setCurrentIndex] = useState<number>(0);

  const [isMagnified, setIsMagnified] = useState<boolean>(false);

  const totalImages = images.length;

  const handleUpdateLabels = useCallback(
    (id: number, labels: { x: number; y: number; id: number }[]) => {
      dispatch(updateLabels({ id, labels }));
    },
    [dispatch]
  );

  const handleDeleteImage = useCallback(
    (id: number) => {
      dispatch(removeFile(id));
      setCurrentIndex((prevIndex) => {
        const newTotal = totalImages - 1;
        if (newTotal === 0) return 0;
        return prevIndex >= newTotal ? 0 : prevIndex;
      });
    },
    [dispatch, totalImages]
  );

  const handleNext = useCallback(() => {
    setCurrentIndex((prevIndex) => (prevIndex + 1) % totalImages);
  }, [totalImages]);

  const handlePrev = useCallback(() => {
    setCurrentIndex((prevIndex) => (prevIndex - 1 + totalImages) % totalImages);
  }, [totalImages]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'ArrowRight') {
        handleNext();
      } else if (e.key === 'ArrowLeft') {
        handlePrev();
      }
    },
    [handleNext, handlePrev]
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);

    // Clean up the event listener on component unmount
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [handleKeyDown]);

  const toggleMagnifiedView = () => {
    setIsMagnified((prev) => !prev);
  };

  if (totalImages === 0) {
    return (
      <Box sx={{ padding: '20px', textAlign: 'center' }}>
        <Typography variant="h5" sx={{ marginBottom: '10px' }}>
          No images available.
        </Typography>
        <Typography variant="body1" sx={{ maxWidth: '400px', margin: '0 auto' }}>
          Press <strong>Ctrl+N</strong> to select images to upload, or manually select an image using the left sidebar to start labeling.
        </Typography>
      </Box>
    );
  }

  return (
    <Box
      sx={{
        padding: '20px',
        maxWidth: '1000px',
        margin: '0 auto',
        position: 'relative',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
      }}
    >
      {/* Top Bar with Delete and Zoom Buttons */}
      <Box sx={{ width: '100%', display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
        <Tooltip title="Delete Image">
          <IconButton
            onClick={() => handleDeleteImage(images[currentIndex].id)}
            aria-label="Delete Image"
            sx={{
              backgroundColor: 'rgba(255,255,255,0.7)',
              '&:hover': {
                backgroundColor: 'rgba(255,255,255,1)',
              },
            }}
          >
            <DeleteIcon />
          </IconButton>
        </Tooltip>
        <Tooltip title="Magnify Image">
          <IconButton
            onClick={toggleMagnifiedView}
            aria-label="Magnify Image"
            sx={{
              backgroundColor: 'rgba(255,255,255,0.7)',
              '&:hover': {
                backgroundColor: 'rgba(255,255,255,1)',
              },
            }}
          >
            <ZoomInIcon />
          </IconButton>
        </Tooltip>
      </Box>

      {/* Carousel Container */}
      <Box
        sx={{
          position: 'relative',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          marginTop: '10px',
        }}
      >
        {/* Left Navigation Arrow */}
        <Tooltip title="Previous Image">
          <IconButton
            onClick={handlePrev}
            disabled={totalImages === 1}
            aria-label="Previous Image"
            sx={{
              position: 'absolute',
              left: '-40px', // Adjusted position to avoid overlap
              top: '50%',
              transform: 'translateY(-50%)',
              backgroundColor: 'rgba(255,255,255,0.7)',
              '&:hover': {
                backgroundColor: 'rgba(255,255,255,1)',
              },
              zIndex: 2,
            }}
          >
            <ArrowBackIosIcon />
          </IconButton>
        </Tooltip>

        {/* Image Labeler */}
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
          }}
        >
          <ImageLabeler
            key={images[currentIndex].id}
            imageURL={images[currentIndex].url}
            initialPoints={images[currentIndex].labels}
            initialHistory={images[currentIndex].labelHistory}
            onPointsChange={(newPoints) => handleUpdateLabels(images[currentIndex].id, newPoints)}
            color={color}
            opacity={opacity}
          />
        </Box>

        {/* Right Navigation Arrow */}
        <Tooltip title="Next Image">
          <IconButton
            onClick={handleNext}
            disabled={totalImages === 1}
            aria-label="Next Image"
            sx={{
              position: 'absolute',
              right: '-40px', // Adjusted position to avoid overlap
              top: '50%',
              transform: 'translateY(-50%)',
              backgroundColor: 'rgba(255,255,255,0.7)',
              '&:hover': {
                backgroundColor: 'rgba(255,255,255,1)',
              },
              zIndex: 2,
            }}
          >
            <ArrowForwardIosIcon />
          </IconButton>
        </Tooltip>
      </Box>

      {/* Carousel Information and Export Button */}
      <Box sx={{ textAlign: 'center', marginTop: '20px' }}>
        <Typography variant="body1" sx={{ marginBottom: '10px' }}>
          Image {currentIndex + 1} of {totalImages}
        </Typography>
        <Button
          variant="contained"
          color="primary"
          onClick={() => {
            const data = images.map(({ id, url, labels }) => ({
              id,
              url,
              labels,
            }));
            const jsonData = JSON.stringify(data, null, 2);
            const blob = new Blob([jsonData], { type: 'application/json' });
            const urlBlob = URL.createObjectURL(blob);

            const a = document.createElement('a');
            a.href = urlBlob;
            a.download = `all_labeled_data_${Date.now()}.json`;
            a.click();
            URL.revokeObjectURL(urlBlob);
          }}
          sx={{ marginTop: '10px' }}
        >
          Export All Labeled Data
        </Button>
      </Box>

      {/* Magnified Image Labeler Modal */}
      <MagnifiedImageLabeler
        imageURL={images[currentIndex].url}
        initialPoints={images[currentIndex].labels}
        onPointsChange={(newPoints: { x: number; y: number; id: number; }[]) => handleUpdateLabels(images[currentIndex].id, newPoints)}
        color={color}
        opacity={opacity}
        open={isMagnified}
        onClose={toggleMagnifiedView}
      />
    </Box>
  );
};

export default ImageLabelerCarousel;
