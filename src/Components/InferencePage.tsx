import React, { useEffect, useState, useRef } from "react";
import { motion } from "framer-motion";
import {
  ArrowLeft,
  Microscope,
  Upload,
  Play,
  Download,
  ChevronLeft,
  ChevronRight,
  Loader2,
  ImageIcon,
  X,
  FileJson,
  FileSpreadsheet,
} from "lucide-react";
import { toast } from "sonner";

import { cn } from "@/lib/utils";
import { Button } from "@/Components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/Components/ui/card";
import { Label } from "@/Components/ui/label";
import { ScrollArea } from "@/Components/ui/scroll-area";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/Components/ui/popover";
import { staggerContainer, staggerItem, buttonHover, buttonTap, cardHover } from "@/lib/animations";
import { TrainedModel, AppView, BoundingBox } from "@/types/Image";

interface InferencePageProps {
  onNavigate: (view: AppView) => void;
  initialModel?: string;
}

interface InferenceImage {
  path: string;
  name: string;
  url: string;
  results?: {
    boxes: BoundingBox[];
  };
  error?: string;
}

export const InferencePage: React.FC<InferencePageProps> = ({
  onNavigate,
  initialModel,
}) => {
  const [models, setModels] = useState<TrainedModel[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>(initialModel || "");
  const [images, setImages] = useState<InferenceImage[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [loadingModels, setLoadingModels] = useState(true);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);

  // Load models on mount
  useEffect(() => {
    const loadModels = async () => {
      try {
        const result = await window.api.listModels();
        if (result.ok && result.models) {
          setModels(result.models);
          // Only set default if no model was provided via initialModel
          if (!initialModel && result.models.length > 0) {
            setSelectedModel(result.models[0].name);
          }
        }
      } catch (err) {
        console.error("Failed to load models:", err);
        toast.error("Failed to load models");
      } finally {
        setLoadingModels(false);
      }
    };
    loadModels();
  }, [initialModel]);

  // Set initial model when provided
  useEffect(() => {
    if (initialModel) {
      setSelectedModel(initialModel);
    }
  }, [initialModel]);

  const handleSelectImages = async () => {
    try {
      const result = await window.api.selectImages();
      if (!result.canceled && result.files) {
        const newImages: InferenceImage[] = result.files.map((file) => ({
          path: file.path,
          name: file.name,
          url: `file://${file.path}`,
        }));
        setImages((prev) => [...prev, ...newImages]);
        toast.success(`Added ${result.files.length} image(s)`);
      }
    } catch (err) {
      console.error("Failed to select images:", err);
      toast.error("Failed to select images");
    }
  };

  const handleRemoveImage = (index: number) => {
    setImages((prev) => {
      const next = prev.filter((_, i) => i !== index);
      if (currentIndex >= next.length && next.length > 0) {
        setCurrentIndex(next.length - 1);
      }
      return next;
    });
  };

  const handleRunInference = async () => {
    if (!selectedModel || images.length === 0) {
      toast.error("Please select a model and add images");
      return;
    }

    setIsRunning(true);
    let successCount = 0;
    let errorCount = 0;

    for (let i = 0; i < images.length; i++) {
      const img = images[i];
      try {
        const result = await window.api.predictImage(img.path, selectedModel);
        if (result.ok && result.data) {
          setImages((prev) =>
            prev.map((item, idx) =>
              idx === i ? { ...item, results: result.data, error: undefined } : item
            )
          );
          successCount++;
        } else {
          setImages((prev) =>
            prev.map((item, idx) =>
              idx === i ? { ...item, error: result.error || "Prediction failed" } : item
            )
          );
          errorCount++;
        }
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : "Prediction failed";
        setImages((prev) =>
          prev.map((item, idx) =>
            idx === i ? { ...item, error: errorMessage } : item
          )
        );
        errorCount++;
      }
    }

    setIsRunning(false);
    if (successCount > 0) {
      toast.success(`Inference complete: ${successCount} succeeded, ${errorCount} failed`);
    } else {
      toast.error("All inferences failed");
    }
  };

  const handleExport = (format: "json" | "csv") => {
    const imagesWithResults = images.filter((img) => img.results);
    if (imagesWithResults.length === 0) {
      toast.error("No results to export");
      return;
    }

    let content: string;
    let filename: string;
    let mimeType: string;

    if (format === "json") {
      const data = imagesWithResults.map((img) => ({
        filename: img.name,
        path: img.path,
        boxes: img.results?.boxes || [],
      }));
      content = JSON.stringify(data, null, 2);
      filename = `inference_results_${Date.now()}.json`;
      mimeType = "application/json";
    } else {
      // CSV format: filename, box_id, left, top, width, height, landmark_id, landmark_x, landmark_y
      const rows: string[] = [
        "filename,box_id,box_left,box_top,box_width,box_height,landmark_id,landmark_x,landmark_y",
      ];
      imagesWithResults.forEach((img) => {
        img.results?.boxes.forEach((box, boxIdx) => {
          if (box.landmarks && box.landmarks.length > 0) {
            box.landmarks.forEach((lm) => {
              rows.push(
                `"${img.name}",${boxIdx},${box.left},${box.top},${box.width},${box.height},${lm.id},${lm.x},${lm.y}`
              );
            });
          } else {
            rows.push(
              `"${img.name}",${boxIdx},${box.left},${box.top},${box.width},${box.height},,,`
            );
          }
        });
      });
      content = rows.join("\n");
      filename = `inference_results_${Date.now()}.csv`;
      mimeType = "text/csv";
    }

    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
    toast.success(`Exported to ${filename}`);
  };

  // Draw landmarks on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const currentImage = images[currentIndex];
    if (!currentImage) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }

    const img = new Image();
    img.src = currentImage.url;
    imageRef.current = img;

    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      // Draw results if available
      if (currentImage.results?.boxes) {
        currentImage.results.boxes.forEach((box) => {
          // Draw bounding box
          ctx.strokeStyle = "rgba(0, 255, 0, 0.8)";
          ctx.lineWidth = 2;
          ctx.strokeRect(box.left, box.top, box.width, box.height);

          // Draw landmarks
          if (box.landmarks) {
            box.landmarks.forEach((lm, idx) => {
              // Draw point
              ctx.fillStyle = "rgba(255, 0, 0, 0.9)";
              ctx.beginPath();
              ctx.arc(lm.x, lm.y, 4, 0, Math.PI * 2);
              ctx.fill();

              // Draw label
              ctx.fillStyle = "white";
              ctx.font = "12px sans-serif";
              ctx.fillText(String(idx + 1), lm.x + 6, lm.y + 4);
            });
          }
        });
      }
    };
  }, [currentIndex, images]);

  const currentImage = images[currentIndex];

  return (
    <div className="flex h-screen w-screen flex-col bg-background">
      {/* Header */}
      <div className="flex items-center justify-between border-b p-4">
        <div className="flex items-center gap-4">
          <motion.div {...buttonHover} {...buttonTap}>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => onNavigate("landing")}
            >
              <ArrowLeft className="h-5 w-5" />
            </Button>
          </motion.div>
          <div className="flex items-center gap-2">
            <Microscope className="h-5 w-5 text-primary" />
            <h1 className="text-lg font-bold">Run Inference</h1>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {images.some((img) => img.results) && (
            <Popover>
              <PopoverTrigger asChild>
                <Button variant="outline" size="sm">
                  <Download className="mr-2 h-4 w-4" />
                  Export
                </Button>
              </PopoverTrigger>
              <PopoverContent align="end" className="w-40 p-1">
                <Button
                  variant="ghost"
                  size="sm"
                  className="w-full justify-start"
                  onClick={() => handleExport("json")}
                >
                  <FileJson className="mr-2 h-4 w-4" />
                  JSON
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="w-full justify-start"
                  onClick={() => handleExport("csv")}
                >
                  <FileSpreadsheet className="mr-2 h-4 w-4" />
                  CSV
                </Button>
              </PopoverContent>
            </Popover>
          )}
          <motion.div {...buttonHover} {...buttonTap}>
            <Button
              onClick={handleRunInference}
              disabled={isRunning || !selectedModel || images.length === 0}
            >
              {isRunning ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <Play className="mr-2 h-4 w-4" />
                  Run Inference
                </>
              )}
            </Button>
          </motion.div>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <div className="w-72 shrink-0 border-r bg-card">
          <ScrollArea className="h-full">
            <div className="space-y-4 p-4">
              {/* Model Selection */}
              <motion.div
                variants={cardHover}
                initial="initial"
                whileHover="hover"
              >
                <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-xs font-bold uppercase tracking-wide text-muted-foreground">
                      Model
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    {loadingModels ? (
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Loading...
                      </div>
                    ) : models.length === 0 ? (
                      <p className="text-xs text-muted-foreground">
                        No trained models found. Train a model first.
                      </p>
                    ) : (
                      <div className="space-y-2">
                        <Label className="text-sm">Select Model</Label>
                        <select
                          value={selectedModel}
                          onChange={(e) => setSelectedModel(e.target.value)}
                          className="w-full rounded-md border bg-background px-3 py-2 text-sm"
                        >
                          {models.map((model) => (
                            <option key={model.name} value={model.name}>
                              {model.name}
                            </option>
                          ))}
                        </select>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </motion.div>

              {/* Images */}
              <motion.div
                variants={cardHover}
                initial="initial"
                whileHover="hover"
              >
                <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-xs font-bold uppercase tracking-wide text-muted-foreground">
                        Images ({images.length})
                      </CardTitle>
                      <motion.div {...buttonHover} {...buttonTap}>
                        <Button size="sm" variant="outline" onClick={handleSelectImages}>
                          <Upload className="mr-1 h-3 w-3" />
                          Add
                        </Button>
                      </motion.div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    {images.length === 0 ? (
                      <div className="flex flex-col items-center rounded-md border border-dashed p-6 text-center">
                        <ImageIcon className="mb-2 h-8 w-8 text-muted-foreground/50" />
                        <p className="text-xs text-muted-foreground">
                          No images added yet
                        </p>
                        <motion.div {...buttonHover} {...buttonTap} className="mt-2">
                          <Button size="sm" variant="outline" onClick={handleSelectImages}>
                            Select Images
                          </Button>
                        </motion.div>
                      </div>
                    ) : (
                      <motion.div
                        variants={staggerContainer}
                        initial="initial"
                        animate="animate"
                        className="space-y-2"
                      >
                        {images.map((img, idx) => (
                          <motion.div
                            key={img.path}
                            variants={staggerItem}
                            className={cn(
                              "flex items-center gap-2 rounded-md border p-2 text-xs",
                              idx === currentIndex
                                ? "border-primary bg-primary/5"
                                : "border-border/50",
                              img.results && "border-green-500/50",
                              img.error && "border-destructive/50"
                            )}
                            onClick={() => setCurrentIndex(idx)}
                          >
                            <div className="min-w-0 flex-1 truncate">
                              {img.name}
                            </div>
                            {img.results && (
                              <span className="shrink-0 text-green-500">
                                {img.results.boxes.length} box(es)
                              </span>
                            )}
                            {img.error && (
                              <span className="shrink-0 text-destructive">Error</span>
                            )}
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-6 w-6 shrink-0"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleRemoveImage(idx);
                              }}
                            >
                              <X className="h-3 w-3" />
                            </Button>
                          </motion.div>
                        ))}
                      </motion.div>
                    )}
                  </CardContent>
                </Card>
              </motion.div>
            </div>
          </ScrollArea>
        </div>

        {/* Main content area */}
        <div className="relative flex flex-1 flex-col items-center justify-center overflow-hidden bg-muted/30 p-4">
          {images.length === 0 ? (
            <div className="text-center">
              <Microscope className="mx-auto mb-4 h-16 w-16 text-muted-foreground/50" />
              <h2 className="text-lg font-semibold">No images to display</h2>
              <p className="mt-2 max-w-sm text-sm text-muted-foreground">
                Add images and select a model to run inference.
              </p>
            </div>
          ) : (
            <>
              {/* Canvas */}
              <div className="relative flex-1 overflow-hidden">
                <canvas
                  ref={canvasRef}
                  className="max-h-full max-w-full object-contain"
                  style={{ display: "block", margin: "auto" }}
                />
              </div>

              {/* Navigation */}
              <div className="mt-4 flex items-center gap-4">
                <motion.div {...buttonHover} {...buttonTap}>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => setCurrentIndex(Math.max(0, currentIndex - 1))}
                    disabled={currentIndex === 0}
                  >
                    <ChevronLeft className="h-5 w-5" />
                  </Button>
                </motion.div>
                <span className="text-sm text-muted-foreground">
                  {currentIndex + 1} / {images.length}
                </span>
                <motion.div {...buttonHover} {...buttonTap}>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() =>
                      setCurrentIndex(Math.min(images.length - 1, currentIndex + 1))
                    }
                    disabled={currentIndex === images.length - 1}
                  >
                    <ChevronRight className="h-5 w-5" />
                  </Button>
                </motion.div>
              </div>

              {/* Current image info */}
              {currentImage && (
                <div className="mt-2 text-center text-xs text-muted-foreground">
                  <p className="font-medium">{currentImage.name}</p>
                  {currentImage.error && (
                    <p className="text-destructive">{currentImage.error}</p>
                  )}
                  {currentImage.results && (
                    <p className="text-green-500">
                      Found {currentImage.results.boxes.length} region(s) with{" "}
                      {currentImage.results.boxes.reduce(
                        (sum, box) => sum + (box.landmarks?.length || 0),
                        0
                      )}{" "}
                      total landmarks
                    </p>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default InferencePage;
