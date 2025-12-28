import { AnnotatedImage } from "./Image";
export {};

declare global {
  interface Window {
    api: {
      saveLabels: (fileArray: AnnotatedImage []) => Promise<any>;
      trainModel: (modelName: string) => Promise<any>;
      predictImage: (imagePath: string) => Promise<any>;
      selectImageFolder: () => Promise<any>;
    };
  }
}
