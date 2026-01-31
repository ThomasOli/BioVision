import { AnnotatedImage } from "./Image";
export {};

declare global {
  interface Window {
    api: {
      saveLabels: (fileArray: AnnotatedImage []) => Promise<any>;
      trainModel: (modelName: string) => Promise<any>;
      predictImage: (imagePath: string, tag: string) => Promise<any>;
      selectImageFolder: () => Promise<any>;
      getProjectRoot: () => Promise<{ projectRoot: string }>;
      selectProjectRoot: () => Promise<{ canceled?: boolean; projectRoot?: string }>;
    };
  }
}
