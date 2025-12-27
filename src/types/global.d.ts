import { AnnotatedImage } from "./Image";
export {};

declare global {
  interface Window {
    api: {
      saveLabels: (fileArray: AnnotatedImage []) => Promise<any>;
      trainModel: () => Promise<any>;
      predictImage: (imagePath: string) => Promise<any>;
      selectImageFolder: () => Promise<any>;
    };
  }
}
