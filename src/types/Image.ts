export interface Point {
  x: number;
  y: number;
  id: number;
}

export interface BoundingBox {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface AnnotatedImage {
  id: number;
  path: string;
  url: string;
  filename: string;
  labels: Point[];
  history: Point[][];
  future: Point[][];
  boundingBox?: BoundingBox;
}