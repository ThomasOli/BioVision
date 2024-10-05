// hooks/useImageLoader.ts
import { useState, useEffect } from 'react';

interface ImageDimensions {
  width: number;
  height: number;
}

const useImageLoader = (imageURL: string | null): [HTMLImageElement | null, ImageDimensions | null] => {
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [dimensions, setDimensions] = useState<ImageDimensions | null>(null);

  useEffect(() => {
    if (!imageURL) {
      setImage(null);
      setDimensions(null);
      return;
    }

    const img = new window.Image();
    img.src = imageURL;
    img.onload = () => {
      setImage(img);
      setDimensions({ width: img.width, height: img.height });
    };

    return () => {
      URL.revokeObjectURL(imageURL);
    };
  }, [imageURL]);

  return [image, dimensions];
};

export default useImageLoader;
