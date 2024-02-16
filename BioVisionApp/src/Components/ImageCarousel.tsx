import React, { useState, useEffect } from "react";
import { useSelector, useDispatch } from "react-redux";
import { RootState } from "../state/store";
import { clearFiles, addFile, removeFile } from "../state/filesState/fileSlice";
import {
  Paper,
  IconButton,
  Card,
  Stack,
  List,
  ListItem,
  ListItemText,
  Slider,
} from "@mui/material";
import ArrowBackIosNewIcon from "@mui/icons-material/ArrowBackIosNew";
import ArrowForwardIosIcon from "@mui/icons-material/ArrowForwardIos";
import CloseIcon from "@mui/icons-material/Close";
import ZoomInIcon from "@mui/icons-material/ZoomIn";
import ZoomOutIcon from "@mui/icons-material/ZoomOut";

interface Dot {
  x: number;
  y: number;
  size: number;
}

interface ImageCarouselProps {
  color: string;
  opacity: number;
}

const ImageCarousel: React.FC<ImageCarouselProps> = ({ color, opacity }) => {
  const containerWidth = 800; // Set your desired fixed width
  const containerHeight = 600; // Set your desired fixed height

  const [dots, setDots] = useState<{ [key: string]: Dot[] }>({});
  const [imageDimensions, setImageDimensions] = useState<{
    width: number;
    height: number;
  }>({
    width: containerWidth,
    height: containerHeight,
  });

  const [dotSize] = useState<number>(10);
  const [zoomLevel, setZoomLevel] = useState<number>(1);

  const handleImageClick = (e: React.MouseEvent<HTMLImageElement>) => {
    const currentImageKey = getCurrentImageKey();
    const imageContainer = document.getElementById(currentImageKey);
    if (!imageContainer) return;

    const rect = imageContainer.getBoundingClientRect();
    const x = (e.clientX - rect.left) / zoomLevel;
    const y = (e.clientY - rect.top) / zoomLevel;

    setDots((prevDots) => ({
      ...prevDots,
      [currentImageKey]: [...(prevDots[currentImageKey] || []), { x, y, size: dotSize }],
    }));
  };

  const handleResize = () => {
    updateImageDimensions();
  };

  const updateImageDimensions = () => {
    const currentImageKey = getCurrentImageKey();
    const imageContainer = document.getElementById(currentImageKey);

    if (imageContainer) {
      const { clientWidth, clientHeight } = imageContainer;
      setImageDimensions({ width: clientWidth, height: clientHeight });
    }
  };

  useEffect(() => {
    updateImageDimensions();
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, []);

  const files = useSelector((state: RootState) => state.files.fileArray);
  const dispatch = useDispatch();
  const [currentIndex, setCurrentIndex] = useState(0);
  const [showNav, setShowNav] = useState(false);

  const handleNext = () => {
    saveAnnotations();
    setCurrentIndex((prevIndex) => (prevIndex + 1) % files.length);
  };

  const handleBack = () => {
    saveAnnotations();
    setCurrentIndex(
      (prevIndex) => (prevIndex - 1 + files.length) % files.length
    );
  };

  const saveAnnotations = () => {
    const currentImageKey = getCurrentImageKey();
    setDots((prevDots) => ({
      ...prevDots,
      [currentImageKey]: dots[currentImageKey] || [],
    }));
  };

  const getCurrentImageKey = () => {
    return `image-container-${currentIndex}`;
  };

  const currentFile = files[currentIndex];
  const prevFile = files[(currentIndex - 1 + files.length) % files.length];
  const nextFile = files[(currentIndex + 1) % files.length];

  useEffect(() => {
    if (files.length > 0) {
      setShowNav(true);
    }
  }, [files]);

  const Navigation = () => {
    return (
      <>
        <IconButton onClick={handleBack}>
          <ArrowBackIosNewIcon />
        </IconButton>
        <IconButton onClick={handleNext}>
          <ArrowForwardIosIcon />
        </IconButton>
        <CloseIcon />
      </>
    );
  };

  const handleDotRemove = (index: number) => {
    const currentImageKey = getCurrentImageKey();
    setDots((prevDots) => ({
      ...prevDots,
      [currentImageKey]: prevDots[currentImageKey].filter((_, i) => i !== index),
    }));
  };

  const handleZoom = (zoomIn: boolean) => {
    if (zoomIn) {
      setZoomLevel((prevZoom) => prevZoom * 1.1); // Increase zoom level by 10%
    } else {
      setZoomLevel((prevZoom) => prevZoom / 1.1); // Decrease zoom level by 10%
    }
  };

  return (
    <>
      {showNav && (
        <Stack direction="column" alignItems="center">
          <Stack direction="row">
            <Paper
              sx={{
                width: "300px",
                maxHeight: "100%",
                overflow: "auto",
                backgroundColor: "#fff",
                height: `${containerHeight}px`, // Set the height to match the image container
              }}
            >
              <List>
                {(dots[getCurrentImageKey()] || []).map((dot, index) => (
                  <ListItem key={index}>
                    <ListItemText
                      primary={`Dot ${index + 1}`}
                      secondary={`(${dot.x}, ${dot.y})`}
                    />
                    <Stack direction="row" alignItems="center">
                      <Slider
                        value={dot.size}
                        onChange={(_e, newValue) => {
                          const newSize = newValue as number;
                          setDots((prevDots) => {
                            const currentImageKey = getCurrentImageKey();
                            const updatedDots = [...prevDots[currentImageKey]];
                            updatedDots[index].size = newSize;
                            return {
                              ...prevDots,
                              [currentImageKey]: updatedDots,
                            };
                          });
                        }}
                        min={2}
                        max={10}
                        aria-label="Dot Size"
                        sx={{ width: 100 }}
                      />
                      <IconButton onClick={() => handleDotRemove(index)}>
                        <CloseIcon />
                      </IconButton>
                    </Stack>
                  </ListItem>
                ))}
              </List>
            </Paper>
            <Card
              id={getCurrentImageKey()}
              sx={{
                position: "relative",
                display: "flex",
                height: `${containerHeight}px`,
                width: `${containerWidth}px`,
                alignItems: "center",
                justifyContent: "center",
                mb: "2rem",
                overflow: "hidden",
              }}
            >
              {currentFile && (
                <img
                  src={URL.createObjectURL(currentFile)}
                  alt="current"
                  style={{
                    width: `${imageDimensions.width}px`,
                    height: `${imageDimensions.height}px`,
                    objectFit: "contain",
                    transformOrigin: "top left",
                    transform: `scale(${zoomLevel})`,
                  }}
                  onClick={handleImageClick}
                />
              )}
              {(dots[getCurrentImageKey()] || []).map((dot, index) => (
                <div
                  key={index}
                  style={{
                    position: "absolute",
                    top: `${dot.y}px`,
                    left: `${dot.x}px`,
                    width: `${dot.size}px`,
                    height: `${dot.size}px`,
                    backgroundColor: `${color}`,
                    borderRadius: "50%",
                    opacity: `${opacity / 100}`,
                    transform: "translate(-50%, -50%)",
                  }}
                />
              ))}
              <div
                style={{
                  position: "absolute",
                  top: "0",
                  right: "0",
                  zIndex: 1000,
                }}
              >
                <IconButton onClick={() => handleZoom(true)}>
                  <ZoomInIcon sx={{ bgcolor: "transparent" }} />
                </IconButton>
                <IconButton onClick={() => handleZoom(false)}>
                  <ZoomOutIcon sx={{ bgcolor: "transparent" }} />
                </IconButton>
              </div>
            </Card>
          </Stack>
          <Stack direction="row" sx={{ backgroundColor: "#fff", padding: "10px" }}>
            {prevFile && (
              <img
                src={URL.createObjectURL(prevFile)}
                alt="previous"
                style={{ width: "100px", height: "auto", marginRight: "10px" }}
              />
            )}
            {nextFile && (
              <img
                src={URL.createObjectURL(nextFile)}
                alt="next"
                style={{ width: "100px", height: "auto", marginRight: "10px" }}
              />
            )}
            <Navigation />
          </Stack>
        </Stack>
      )}
    </>
  );
};

export default ImageCarousel;
