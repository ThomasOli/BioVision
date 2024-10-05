// components/ImageLabelerCarousel.tsx
import React, { useState } from 'react';
import ImageCarousel from './ImageCarousel';

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

const ImageLabelerCarousel: React.FC = () => {
  const [images, setImages] = useState<ImageData[]>([
    // Initialize with some images or leave empty and add via file input
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
    // Add more images as needed
  ]);

  // Handle updating labels for a specific image
  const handleUpdateLabels = (imageId: number, labels: Point[]) => {
    setImages((prevImages) =>
      prevImages.map((img) =>
        img.id === imageId ? { ...img, labels } : img
      )
    );
  };

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

  // Handle deleting an image
  const handleDeleteImage = (imageId: number) => {
    setImages((prevImages) => prevImages.filter((img) => img.id !== imageId));
  };

  return (
    <div>
      <h2>Image Labeler Carousel</h2>
      <input type="file" accept="image/*" multiple onChange={handleAddImages} />
      <ImageCarousel
        images={images}
        onUpdateLabels={handleUpdateLabels}
        onDeleteImage={handleDeleteImage}
      />
      {images.length > 0 && (
        <button onClick={handleExportAll} style={{ marginTop: '20px' }}>
          Export All Labeled Data as JSON
        </button>
      )}
    </div>
  );
};

export default ImageLabelerCarousel;
