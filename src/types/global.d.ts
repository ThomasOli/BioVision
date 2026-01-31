import { AnnotatedImage, TrainedModel, BoundingBox } from "./Image";
export {};

interface SaveLabelsResult {
  ok: boolean;
}

interface TrainModelResult {
  ok: boolean;
  output?: string;
  error?: string;
}

interface PredictImageResult {
  ok: boolean;
  data?: { boxes: BoundingBox[] };
  error?: string;
}

interface SelectImageFolderResult {
  canceled: boolean;
  images?: { filename: string; path: string }[];
  image?: File[];  // Legacy field for backwards compatibility
}

declare global {
  interface Window {
    api: {
      saveLabels: (fileArray: AnnotatedImage []) => Promise<SaveLabelsResult>;
      trainModel: (modelName: string) => Promise<TrainModelResult>;
      predictImage: (imagePath: string, tag: string) => Promise<PredictImageResult>;
      selectImageFolder: () => Promise<SelectImageFolderResult>;
      getProjectRoot: () => Promise<{ projectRoot: string }>;
      selectProjectRoot: () => Promise<{ canceled?: boolean; projectRoot?: string }>;
      listModels: () => Promise<{ ok: boolean; models?: TrainedModel[]; error?: string }>;
      deleteModel: (modelName: string) => Promise<{ ok: boolean; error?: string }>;
      renameModel: (oldName: string, newName: string) => Promise<{ ok: boolean; error?: string }>;
      getModelInfo: (modelName: string) => Promise<{ ok: boolean; model?: TrainedModel; error?: string }>;
      selectImages: () => Promise<{ canceled: boolean; files?: { path: string; name: string }[] }>;
    };
  }
}
