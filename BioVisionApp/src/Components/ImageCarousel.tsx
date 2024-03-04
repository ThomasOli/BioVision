import { useSelector, useDispatch } from "react-redux";
import { useState, useEffect, useRef} from "react";

import * as fabric from "fabric";
import { clearFiles, addFile, removeFile } from "../state/filesState/fileSlice";
import { RootState } from "../state/store";
import { Paper, IconButton, Card, Stack, Grid, ImageList, ImageListItem } from "@mui/material";
import ArrowBackIosNewIcon from "@mui/icons-material/ArrowBackIosNew";
import ArrowForwardIosIcon from "@mui/icons-material/ArrowForwardIos";
import { Delete, Draw } from "@mui/icons-material";
import DrawableCanvas from "./DrawableCanvas";
import { current } from "@reduxjs/toolkit";
const ImageCarousel: React.FC = () => {
  const imageListContainerRef = useRef<HTMLDivElement>(null);
  const imageContainer = document.getElementById("image-container");
  const [imageDimensions, setImageDimensions] = useState<{
    width: number;
    height: number;
  }>({
    width: 50 ,
    height: 50,
  });
  const [pastImageDimensions, setPastImageDimensions] = useState<{
    width: number;
    height: number;
  }>({
    width: 0,
    height: 0,
  });

  const handleImageClick = (e: React.MouseEvent<HTMLImageElement>) => {

    if (imageContainer) {
      const { width, height } = imageContainer.getBoundingClientRect();
      setPastImageDimensions({ width, height });
    }
  };

  const files = useSelector((state: RootState) => state.files.fileArray);
  const dispatch = useDispatch();
  const [currentIndex, setCurrentIndex] = useState(0);

  const [showNav, setshowNav] = useState(false);
  const handleNext = () => {
    setCurrentIndex((prevIndex) => (prevIndex + 1) % files.length);
  };

  const handleBack = () => {
    setCurrentIndex(
      (prevIndex) => (prevIndex - 1 + files.length) % files.length
    );
  };
  const currentFile = files[currentIndex];
  const prevFile = files[(currentIndex - 1 + files.length) % files.length];
  const nextFile = files[(currentIndex + 1) % files.length];
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const scrollToCurrentImage = () => {
    if (imageListContainerRef.current) {
      const imageListItem = imageListContainerRef.current.querySelector(
        `[data-index="${currentIndex}"]`
      ) as HTMLDivElement;

      if (imageListItem) {
        imageListItem.scrollIntoView({
          behavior: "smooth",
          inline: "center",
        });
      }
    }
  };

  const handleImageListItemClick = (index: number) => {
    setCurrentIndex(index);
  }
  useEffect(() => {
    if (currentFile) {
      const img = new Image();
      img.onload = () => {
        // Set state with image dimensions
        setImageSize({ width: img.naturalWidth, height: img.naturalHeight });
      };
      img.src = URL.createObjectURL(currentFile);
  
      // Clean up the object URL to avoid memory leaks
      return () => URL.revokeObjectURL(img.src);
    }
  }, [currentFile]);

  useEffect(() => {
    if (files.length > 0) {
      setshowNav(true);
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
        <Delete />
      </>
    );
  };

  return (
    <>
      {showNav &&(
        <Stack sx={{display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
            
          <Card
           id="image-container"
           elevation={0}
           sx={{
             display: "flex",
             height: "70vh",
             width: "100vh",
             alignItems: "center",
             justifyContent: "center",
             mb: "2rem",
             mt: "2rem",
             backgroundColor: '#242424' // matching the background of the default color,
           }}
          >
            {currentFile && (
              <img
                src={URL.createObjectURL(currentFile)}
                alt="current"
                style={{ width: "100%", 
                  height: "100%", 
                  objectFit: "contain" // Added to remove cropping of main image
                }}
                onClick={handleImageClick}
              />
            )}
            <DrawableCanvas
              fillColor={"rgba(255, 165, 0, 0.3)"}
              strokeWidth={2}
              strokeColor={"#000000"}
              backgroundColor={"#ffffff"}
              backgroundImageURL={URL.createObjectURL(currentFile)}
              drawingMode={"point"}
              initialDrawing={{}}
              displayToolbar={true}
              height = {imageSize.height}
              width = {imageSize.width}
            />
            
          </Card>
          <div
            ref={imageListContainerRef}
            style={{
              overflowX: "auto",
              maxWidth: "70%",
            }}
          >
            <ImageList
              sx={{
                gridAutoFlow: "column",
                justifyContent: "space-between",
              }}
            >
              {files.map((file, index) => (
                <ImageListItem
                  key={index}
                  data-index={index}
                  sx={{ width: "100px", height: "100px" }}
                  onClick={() => handleImageListItemClick(index)}
                >
                  <img
                    src={URL.createObjectURL(file)}
                    alt={`image-${index}`}
                    style={{
                      width: "100%",
                      height: "100%",
                      objectFit: "cover",
                      border: currentIndex === index ? "3px solid lime" : "none",
                    }}
                  />
                </ImageListItem>
              ))}
            </ImageList>
          </div>

          <Card
            sx={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <Navigation />
          </Card>
        </Stack>
      )}
    </>
  );
};

export default ImageCarousel;