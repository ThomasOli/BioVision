// src/Components/ImageLabelerCarousel.tsx
import React, { useState, useCallback } from 'react';
import Slider from 'react-slick';
import "slick-carousel/slick/slick.css";
import "slick-carousel/slick/slick-theme.css";
import ImageLabeler from './ImageLabeler';

interface Point {
  x: number;
  y: number;
  id: number;
}

interface ImageData {
  id: number;
  url: string;
  labels: Point[];
}

interface ImageLabelerCarouselProps {
  color: string;
  opacity: number;
}

const ImageLabelerCarousel: React.FC<ImageLabelerCarouselProps> = ({ color, opacity }) => {
  const [images, setImages] = useState<ImageData[]>([
    // Initialize with sample images or leave empty
    {
      id: 1,
      url: 'https://via.placeholder.com/800x600.png?text=Image+1',
      labels: [],
    },
    {
      id: 2,
      url: 'https://via.placeholder.com/800x600.png?text=Image+2',
      labels: [],
    },
  ]);

  // Handle updating labels for a specific image
  const handleUpdateLabels = useCallback((imageId: number, labels: Point[]) => {
    setImages((prevImages) =>
      prevImages.map((img) =>
        img.id === imageId ? { ...img, labels } : img
      )
    );
  }, []);

  // Handle adding new images via file input
  const handleAddImages = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    const newImages: ImageData[] = Array.from(files).map((file, index) => ({
      id: Date.now() + index,
      url: URL.createObjectURL(file),
      labels: [],
    }));

    setImages((prevImages) => [...prevImages, ...newImages]);

    // Reset the file input
    e.target.value = '';
  };

  // Handle deleting an image
  const handleDeleteImage = (imageId: number) => {
    setImages((prevImages) => prevImages.filter((img) => img.id !== imageId));
  };

  // Handle exporting all labeled data
  const handleExportAll = () => {
    const data = images.map(({ id, url, labels }) => ({
      id,
      url,
      labels,
    }));
    const jsonData = JSON.stringify(data, null, 2);
    const blob = new Blob([jsonData], { type: 'application/json' });
    const urlBlob = URL.createObjectURL(blob);

    // Create a link to trigger download
    const a = document.createElement('a');
    a.href = urlBlob;
    a.download = `all_labeled_data_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(urlBlob);
  };

  const settings = {
    dots: true,
    infinite: false,
    speed: 500,
    slidesToShow: 1,
    slidesToScroll: 1,
    adaptiveHeight: true,
  };

  return (
    <div style={{ padding: '20px' }}>
      <input
        type="file"
        accept="image/*"
        multiple
        onChange={handleAddImages}
        style={{ marginBottom: '20px' }}
      />
      <Slider {...settings}>
        {images.map((image) => (
          <div key={image.id} style={{ position: 'relative' }}>
            <ImageLabeler
              imageURL={image.url}
              initialPoints={image.labels}
              onPointsChange={(newPoints) => handleUpdateLabels(image.id, newPoints)}
              color={color}
              opacity={opacity}
            />
            {/* Delete Button */}
            <button
              onClick={() => handleDeleteImage(image.id)}
              style={{
                position: 'absolute',
                top: '10px',
                right: '10px',
                padding: '5px 10px',
                backgroundColor: 'red',
                color: 'white',
                border: 'none',
                borderRadius: '3px',
                cursor: 'pointer',
              }}
            >
              Delete Image
            </button>
          </div>
        ))}
      </Slider>
      {images.length > 0 && (
        <button onClick={handleExportAll} style={{ marginTop: '20px', padding: '10px 20px' }}>
          Export All Labeled Data as JSON
        </button>
      )}
    </div>
  );
};

export default ImageLabelerCarousel;
