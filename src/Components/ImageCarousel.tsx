// components/ImageCarousel.tsx
import React from 'react';
import Slider from 'react-slick';
import 'slick-carousel/slick/slick.css'; 
import 'slick-carousel/slick/slick-theme.css';
import ImageLabeler from './ImageLabeler';

interface ImageData {
  id: number;
  url: string;
  labels: Point[];
}

interface Point {
  x: number;
  y: number;
  id: number;
}

interface ImageCarouselProps {
  images: ImageData[];
  onUpdateLabels: (imageId: number, labels: Point[]) => void;
  onDeleteImage: (imageId: number) => void;
}

const ImageCarousel: React.FC<ImageCarouselProps> = ({ images, onUpdateLabels, onDeleteImage }) => {
  const settings = {
    dots: true,
    infinite: false,
    speed: 500,
    slidesToShow: 1,
    slidesToScroll: 1,
    adaptiveHeight: true,
  };

  return (
    <Slider {...settings}>
      {images.map((image) => (
        <div key={image.id} style={{ position: 'relative' }}>
          <ImageLabeler
            color ={"red"}
            opacity = {100}
            imageURL={image.url}
            initialPoints={image.labels}
            onPointsChange={(newPoints) => onUpdateLabels(image.id, newPoints)}
          />
          {/* Delete Button */}
          <button
            onClick={() => onDeleteImage(image.id)}
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
  );
};

export default ImageCarousel;
