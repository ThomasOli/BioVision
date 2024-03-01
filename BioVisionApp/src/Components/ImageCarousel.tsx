import { useSelector, useDispatch } from "react-redux";
import { useState, useEffect } from "react";

import * as fabric from "fabric";
import { clearFiles, addFile, removeFile } from "../state/filesState/fileSlice";
import { RootState } from "../state/store";
import { IconButton, Card, Stack } from "@mui/material";
import ArrowBackIosNewIcon from "@mui/icons-material/ArrowBackIosNew";
import ArrowForwardIosIcon from "@mui/icons-material/ArrowForwardIos";
import { Delete, Draw } from "@mui/icons-material";
import DrawableCanvas from "./DrawableCanvas";
import { current } from "@reduxjs/toolkit";
const ImageCarousel: React.FC = () => {
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
      {showNav && (
        <Stack>
            
          <Card
           id="image-container"
           elevation={0}
           sx={{
             display: "flex",
             height: "70vh",
             width: "100vh",
             

             mb: "2rem",
             mt: "2rem",
             backgroundColor: '#242424' // matching the background of the default color,
           }}
          >
            
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
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            {prevFile && (
              <img
                src={URL.createObjectURL(prevFile)}
                alt="previous"
                style={{ width: "100px", height: "auto" }}
              />
            )}
            {nextFile && (
              <img
                src={URL.createObjectURL(nextFile)}
                alt="next"
                style={{ width: "100px", height: "auto" }}
              />
            )}
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
