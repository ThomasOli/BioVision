export interface Point {
  x: number;
  y: number;
  id: number;
}


export interface ImageData {
  id: number;
  url: string;
  labels: Point[];
  history: Point[][];
  future: Point[][];
}