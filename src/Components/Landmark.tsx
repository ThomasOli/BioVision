import React, { useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useSelector } from "react-redux";
import { HexColorPicker } from "react-colorful";
import { Trash2, Undo2, Redo2, X, ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/Components/ui/button";
import { Switch } from "@/Components/ui/switch";
import { Slider } from "@/Components/ui/slider";
import { Input } from "@/Components/ui/input";
import { Label } from "@/Components/ui/label";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/Components/ui/popover";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipProvider,
} from "@/Components/ui/tooltip";
import { Separator } from "@/Components/ui/separator";
import { Card, CardContent, CardHeader, CardTitle } from "@/Components/ui/card";
import { UndoRedoClearContext } from "./UndoRedoClearContext";
import { LandmarkPlacementGuide } from "./LandmarkPlacementGuide";
import { buttonHover, buttonTap, cardHover } from "@/lib/animations";
import type { RootState } from "../state/store";

interface LandmarkProps {
  onColorChange: (selectedColor: string) => void;
  onOpacityChange: (selectedOpacity: number) => void;
  onSwitchChange: () => void;
}

const isValidHex = (v: string) => /^#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})$/.test(v);

const normalizeHex = (raw: string) => {
  let v = raw.trim();
  if (!v.startsWith("#")) v = `#${v}`;
  return v;
};

const clamp = (n: number, min: number, max: number) =>
  Math.min(max, Math.max(min, n));

const hexToRgba = (hex: string, alpha01: number) => {
  const h = hex.replace("#", "");
  const full = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
  const r = parseInt(full.slice(0, 2), 16);
  const g = parseInt(full.slice(2, 4), 16);
  const b = parseInt(full.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha01})`;
};

const Landmark: React.FC<LandmarkProps> = ({
  onColorChange,
  onOpacityChange,
  onSwitchChange,
}) => {
  const [previewOnly, setPreviewOnly] = useState(false);
  const [color, setColor] = useState("#ff0000");
  const [hexInput, setHexInput] = useState("#ff0000");
  const [opacity, setOpacity] = useState<number>(100);
  const [colorOpen, setColorOpen] = useState(false);

  const { clear, undo, redo, boxes, skipLandmark } = React.useContext(UndoRedoClearContext);

  // Get active species schema from Redux
  const activeSpecies = useSelector((state: RootState) =>
    state.species.species.find((s) => s.id === state.species.activeSpeciesId)
  );

  // Get landmarks from the first box (auto-created)
  const placedLandmarks = boxes.length > 0 ? boxes[0].landmarks : [];
  const firstBoxId = boxes.length > 0 ? boxes[0].id : null;

  // Total landmarks count
  const totalLandmarks = boxes.reduce((sum, box) => sum + box.landmarks.length, 0);

  const swatches = useMemo(
    () => [
      "#ff3b30",
      "#ff9500",
      "#ffcc00",
      "#34c759",
      "#00c7be",
      "#007aff",
      "#5856d6",
      "#af52de",
      "#ff2d55",
      "#111827",
      "#ffffff",
    ],
    []
  );

  const handlePreviewOnlyChange = () => {
    setPreviewOnly((prev) => !prev);
    onSwitchChange();
  };

  const handleColorChange = (newColor: string) => {
    setColor(newColor);
    setHexInput(newColor);
    onColorChange(newColor);
  };

  const handleHexInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const next = normalizeHex(e.target.value);
    setHexInput(next);
    if (isValidHex(next)) {
      handleColorChange(next.toLowerCase());
    }
  };

  const handleOpacityChange = (value: number[]) => {
    const v = value[0];
    setOpacity(v);
    onOpacityChange(v);
  };

  const handleOpacityInputChange = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const value = event.target.value === "" ? 0 : Number(event.target.value);
    const clampedValue = clamp(value, 0, 100);
    setOpacity(clampedValue);
    onOpacityChange(clampedValue);
  };

  const opacity01 = opacity / 100;

  return (
    <TooltipProvider>
      <motion.div variants={cardHover} initial="initial" whileHover="hover">
        <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-semibold tracking-wide uppercase">
                Annotation Tools
              </CardTitle>
              <span className="text-xs text-muted-foreground">
                {totalLandmarks} landmark{totalLandmarks !== 1 ? "s" : ""}
              </span>
            </div>
            <div className="flex items-center justify-between pt-1">
              <Label htmlFor="preview-mode" className="text-xs text-muted-foreground">
                View Only
              </Label>
              <Switch
                id="preview-mode"
                checked={previewOnly}
                onCheckedChange={handlePreviewOnlyChange}
              />
            </div>
          </CardHeader>

          <CardContent className="space-y-4">
            <AnimatePresence>
              {previewOnly && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="rounded-lg border bg-muted/50 px-3 py-2"
                >
                  <p className="text-xs text-muted-foreground">
                    Editing tools hidden and changes disabled.
                  </p>
                </motion.div>
              )}
            </AnimatePresence>

            <motion.div
              animate={{
                maxHeight: previewOnly ? 0 : 1200,
                opacity: previewOnly ? 0 : 1,
              }}
              transition={{ duration: 0.3, ease: "easeInOut" }}
              className={cn("space-y-4 overflow-hidden", previewOnly && "pointer-events-none")}
            >
              {/* Schema-Guided Placement */}
              {activeSpecies && (
                <LandmarkPlacementGuide
                  schema={{
                    id: activeSpecies.id,
                    name: activeSpecies.name,
                    description: activeSpecies.description || "",
                    landmarks: activeSpecies.landmarkTemplate,
                  }}
                  placedLandmarks={placedLandmarks}
                  onSkip={firstBoxId ? () => skipLandmark(firstBoxId) : undefined}
                />
              )}

              {activeSpecies && <Separator />}

              {/* Action buttons */}
              <div className="flex justify-center">
                <div className="flex items-center gap-6 rounded-full border bg-muted/50 px-3 py-1.5">
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <motion.div {...buttonHover} {...buttonTap}>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8 rounded-full"
                          onClick={() => clear()}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </motion.div>
                    </TooltipTrigger>
                    <TooltipContent>Clear all landmarks</TooltipContent>
                  </Tooltip>

                  <Tooltip>
                    <TooltipTrigger asChild>
                      <motion.div {...buttonHover} {...buttonTap}>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8 rounded-full"
                          onClick={() => undo()}
                        >
                          <Undo2 className="h-4 w-4" />
                        </Button>
                      </motion.div>
                    </TooltipTrigger>
                    <TooltipContent>Undo last action</TooltipContent>
                  </Tooltip>

                  <Tooltip>
                    <TooltipTrigger asChild>
                      <motion.div {...buttonHover} {...buttonTap}>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8 rounded-full"
                          onClick={() => redo()}
                        >
                          <Redo2 className="h-4 w-4" />
                        </Button>
                      </motion.div>
                    </TooltipTrigger>
                    <TooltipContent>Redo last action</TooltipContent>
                  </Tooltip>
                </div>
              </div>

              {/* Color picker */}
              <div className="flex items-center justify-between">
                <Label className="text-xs font-semibold text-muted-foreground">
                  Landmark color
                </Label>

                <Popover open={colorOpen} onOpenChange={setColorOpen}>
                  <PopoverTrigger asChild>
                    <Button
                      variant="outline"
                      className="h-9 gap-2 px-3"
                      disabled={previewOnly}
                    >
                      <div
                        className="h-4 w-4 rounded border"
                        style={{
                          backgroundColor: hexToRgba(color, opacity01),
                        }}
                      />
                      <span className="text-xs font-semibold">
                        {color.toUpperCase()}
                      </span>
                      <ChevronDown className="h-3 w-3 text-muted-foreground" />
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-72" align="end">
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-semibold">Color</span>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-6 w-6"
                          onClick={() => setColorOpen(false)}
                        >
                          <X className="h-3 w-3" />
                        </Button>
                      </div>

                      <div className="overflow-hidden rounded-lg border">
                        <HexColorPicker
                          color={color}
                          onChange={handleColorChange}
                          style={{ width: "100%", height: 170 }}
                        />
                      </div>

                      <div className="flex flex-wrap gap-1.5">
                        {swatches.map((s) => (
                          <button
                            key={s}
                            onClick={() => handleColorChange(s)}
                            className={cn(
                              "h-6 w-6 rounded-md border transition-all hover:scale-110",
                              s.toLowerCase() === color.toLowerCase() &&
                                "ring-2 ring-primary ring-offset-2"
                            )}
                            style={{ backgroundColor: s }}
                          />
                        ))}
                      </div>

                      <Separator />

                      <div className="space-y-1.5">
                        <Label htmlFor="hex-input" className="text-xs">
                          HEX
                        </Label>
                        <Input
                          id="hex-input"
                          value={hexInput}
                          onChange={handleHexInputChange}
                          className={cn(
                            "h-9",
                            hexInput.length > 1 &&
                              !isValidHex(hexInput) &&
                              "border-destructive focus-visible:ring-destructive"
                          )}
                        />
                        {hexInput.length > 1 && !isValidHex(hexInput) && (
                          <p className="text-xs text-destructive">
                            Use #RGB or #RRGGBB
                          </p>
                        )}
                      </div>
                    </div>
                  </PopoverContent>
                </Popover>
              </div>

              {/* Transparency slider */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label className="text-xs font-semibold text-muted-foreground">
                    Transparency
                  </Label>
                  <div className="flex items-center gap-1">
                    <Input
                      type="number"
                      min={0}
                      max={100}
                      value={opacity}
                      onChange={handleOpacityInputChange}
                      className="h-8 w-16 text-right text-sm"
                    />
                    <span className="text-xs text-muted-foreground">%</span>
                  </div>
                </div>
                <Slider
                  value={[opacity]}
                  onValueChange={handleOpacityChange}
                  max={100}
                  step={1}
                  className="w-full"
                />
              </div>
            </motion.div>
          </CardContent>
        </Card>
      </motion.div>
    </TooltipProvider>
  );
};

export default Landmark;
