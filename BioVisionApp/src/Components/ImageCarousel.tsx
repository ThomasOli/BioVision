import { useSelector, useDispatch } from "react-redux";
import { useState, useEffect, useRef} from "react";

import { clearFiles, addFile, removeFile } from "../state/filesState/fileSlice";
import { RootState } from "../state/store";
import { Paper, IconButton, Card, Stack, Grid, ImageList, ImageListItem } from "@mui/material";
import ArrowBackIosNewIcon from "@mui/icons-material/ArrowBackIosNew";
import ArrowForwardIosIcon from "@mui/icons-material/ArrowForwardIos";
import { Delete } from "@mui/icons-material";

interface Dot {
  x: number;
  y: number;
}
interface ImageCarouselProps {
  color: string;
  opacity: number;
}

const ImageCarousel: React.FC<ImageCarouselProps> = ({ color, opacity }) => {

  const imageListContainerRef = useRef<HTMLDivElement>(null);

  const imageContainer = document.getElementById("image-container");

  const [dots, setDots] = useState<Dot[]>([]);
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
    const x = e.clientX;
    const y = e.clientY;

    setDots((prevDots) => [...prevDots, { x, y }]);
  };

  const handleResize = () => {
    const imageContainer = document.getElementById("image-container");

    if (imageContainer) {
      setPastImageDimensions({
        width: imageDimensions.width,
        height: imageDimensions.height,
      });
      updateDotPositions();
    }
  };

  const updateDotPositions = () => {
   
    const newDots = dots.map((dot) => ({
      x: dot.x * (imageDimensions.width / pastImageDimensions.width),
      y: dot.y * (imageDimensions.height / pastImageDimensions.height),
    }));
        
      setDots(newDots);
  };

  useEffect(() => {
    if (imageContainer) {
      const { left, top } = imageContainer.getBoundingClientRect();

      setImageDimensions({ width: left, height: top });
    }

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, [dots]);

  const files = useSelector((state: RootState) => state.files.fileArray);
  const dispatch = useDispatch();
  const [currentIndex, setCurrentIndex] = useState(0);

  const [showNav, setshowNav] = useState(false);

  const handleNext = () => {
    setCurrentIndex((prevIndex) => (prevIndex + 1) % files.length);
    scrollToCurrentImage();
  };

  const handleBack = () => {
    setCurrentIndex(
      (prevIndex) => (prevIndex - 1 + files.length) % files.length
    );
    scrollToCurrentImage();
  };


  const currentFile = files[currentIndex];
  const prevFile = files[(currentIndex - 1 + files.length) % files.length];
  const nextFile = files[(currentIndex + 1) % files.length];

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
        <Stack sx={{display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
          <Card
            id="image-container"
            sx={{
              display: "flex",
              height: "70vh",
              width: "100vh",
              alignItems: "center",
              justifyContent: "center",
              mb: "2rem",
            }}
          >
            {currentFile && (
              <img
                src={URL.createObjectURL(currentFile)}
                alt="current"
                style={{ width: "1000px", height: "auto" }}
                onClick={handleImageClick}
              />
            )}
            {dots.map((dot, index) => (
              <div
                key={index}
                style={{
                  position: "absolute",
                  top: dot.y,
                  left: dot.x,
                  width: `${imageDimensions.width / 100}px`,
                  height: `${imageDimensions.width / 100}px`,
                  backgroundColor: `${color}`,
                  borderRadius: "10px",
                  opacity: `${opacity/100}`
                }}
              />
            ))}
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
