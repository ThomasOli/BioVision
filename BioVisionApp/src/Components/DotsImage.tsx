// DotsImage.tsx
import React, { useState } from 'react';

interface Dot {
  x: number;
  y: number;
}

const DotsImage: React.FC = () => {
  const [dots, setDots] = useState<Dot[]>([]);

  const handleImageClick = (e: React.MouseEvent<HTMLImageElement>) => {
    const x = e.nativeEvent.offsetX;
    const y = e.nativeEvent.offsetY;

    setDots([...dots, { x, y }]);
  };

  return (
    <div>
      <img
        src="/images/your_image.jpg"  // Update with your image path
        alt="Dots Image"
        onClick={handleImageClick}
      />
      {dots.map((dot, index) => (
        <div
          key={index}
          style={{
            position: 'absolute',
            top: dot.y,
            left: dot.x,
            width: '10px',
            height: '10px',
            backgroundColor: 'red',
          }}
        />
      ))}
    </div>
  );
};

export default DotsImage;
