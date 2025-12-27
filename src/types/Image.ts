export interface Point {
  x: number;
  y: number;
  id: number;
}


export interface AnnotatedImage {
  id: number;
  url: string;
  filename: string;
  labels: Point[];
  history: Point[][];
  future: Point[][];
}