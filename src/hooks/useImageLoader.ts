// src/hooks/useImageLoader.ts
import { useState, useEffect } from "react";

/**
 * Hook to load an image and get its ORIGINAL dimensions.
 *
 * IMPORTANT: This returns the actual/natural image dimensions, NOT scaled dimensions.
 * The ImageLabeler component handles display scaling separately.
 * Landmark coordinates must be stored in original image coordinates so they
 * match what the Python backend sees when reading the same image.
 */
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

  useEffect(() => {
    // Only reset error state - keep previous image visible until new one loads
    setError(false);

    if (!url) {
      setImage(null);
      setDimensions(null);
      return;
    }

    const img = new Image();
    img.src = url;
    img.onload = () => {
      // Use naturalWidth/naturalHeight to get ORIGINAL image dimensions
      // This is critical for coordinate consistency with the Python backend
      const width = img.naturalWidth || img.width;
      const height = img.naturalHeight || img.height;

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
