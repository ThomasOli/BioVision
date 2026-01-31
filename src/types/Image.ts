export interface Point {
  x: number;
  y: number;
  id: number;
}

export interface BoundingBox {
  id: number;
  left: number;
  top: number;
  width: number;
  height: number;
  landmarks: Point[];
}

export interface AnnotatedImage {
  id: number;
  path: string;
  url: string;
  filename: string;
  boxes: BoundingBox[];
  selectedBoxId: number | null;
  history: BoundingBox[][];
  future: BoundingBox[][];
}

// Tool modes for the annotation workflow
export type ToolMode = 'box' | 'landmark' | 'select';

// App navigation views
export type AppView = 'landing' | 'workspace' | 'models' | 'inference';

// Trained model metadata
export interface TrainedModel {
  name: string;
  path: string;
  size: number;
  createdAt: Date;
}
