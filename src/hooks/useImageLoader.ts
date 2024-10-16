// src/hooks/useImageLoader.ts
import { useState, useEffect } from "react";

const useImageLoader = (
  url: string
): [
  HTMLImageElement | null,
  { width: number; height: number } | null,
  boolean
] => {
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [dimensions, setDimensions] = useState<{
    width: number;
    height: number;
  } | null>(null);
  const [error, setError] = useState<boolean>(false);
  // src/constants.ts
  const MAX_IMAGE_WIDTH = 800;
  const MAX_IMAGE_HEIGHT = 600;

  useEffect(() => {
    const img = new Image();
    img.src = url;
    img.onload = () => {
      let { width, height } = img;
      let scalingFactor = 1;

      if (width > MAX_IMAGE_WIDTH || height > MAX_IMAGE_HEIGHT) {
        const widthRatio = MAX_IMAGE_WIDTH / width;
        const heightRatio = MAX_IMAGE_HEIGHT / height;
        scalingFactor = Math.min(widthRatio, heightRatio);
        width = Math.round(width * scalingFactor);
        height = Math.round(height * scalingFactor);
      }

      setDimensions({ width, height });
      setImage(img);
    };
    img.onerror = () => {
      setError(true);
    };
    // Cleanup
    return () => {
      img.onload = null;
      img.onerror = null;
    };
  }, [url]);

  return [image, dimensions, error];
};

export default useImageLoader;
