import React, { useEffect, useMemo, useState } from "react";
import { Plus, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { motion } from "framer-motion";

import { LandmarkDefinition, OrientationPolicy } from "@/types/Image";
import { Button } from "@/Components/ui/button";
import { Card } from "@/Components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/Components/ui/dialog";
import { Input } from "@/Components/ui/input";
import { Label } from "@/Components/ui/label";
import { ScrollArea } from "@/Components/ui/scroll-area";
import { buttonHover, buttonTap } from "@/lib/animations";

type EditableSchemaDraft = {
  id?: string;
  name: string;
  description: string;
  landmarks: LandmarkDefinition[];
  sourcePresetId?: string;
  orientationPolicy?: OrientationPolicy;
};

interface CustomSchemaEditorProps {
  open: boolean;
  mode: "create" | "edit-default" | "edit-custom";
  initialSchema?: EditableSchemaDraft | null;
  onSave: (schema: EditableSchemaDraft) => void;
  onCancel: () => void;
}

function normalizeLandmarks(landmarks: LandmarkDefinition[]): LandmarkDefinition[] {
  return (landmarks || []).map((landmark, index) => ({
    ...landmark,
    index: index + 1,
  }));
}

function inferEditorOrientationPolicy(landmarks: LandmarkDefinition[]): OrientationPolicy {
  const categories = new Set(
    (landmarks || []).map((landmark) => String(landmark.category || "").trim().toLowerCase()).filter(Boolean)
  );
  if (categories.has("head") || categories.has("tail") || categories.has("caudal-fin")) {
    return { mode: "directional", targetOrientation: "left", headCategories: ["head"], tailCategories: ["tail", "caudal-fin"] };
  }
  return { mode: "invariant" };
}

export const CustomSchemaEditor: React.FC<CustomSchemaEditorProps> = ({
  open,
  mode,
  initialSchema,
  onSave,
  onCancel,
}) => {
  const [schemaName, setSchemaName] = useState("");
  const [schemaDescription, setSchemaDescription] = useState("");
  const [landmarks, setLandmarks] = useState<LandmarkDefinition[]>([]);
  const [otherLabels, setOtherLabels] = useState<Record<number, string>>({});
  const [orientationMode, setOrientationMode] = useState<OrientationPolicy["mode"]>("invariant");
  const [anteriorAnchorIds, setAnteriorAnchorIds] = useState<number[]>([]);
  const [posteriorAnchorIds, setPosteriorAnchorIds] = useState<number[]>([]);

  useEffect(() => {
    if (!open) return;
    const normalizedLandmarks = normalizeLandmarks(initialSchema?.landmarks || []);
    const initialPolicy = initialSchema?.orientationPolicy || inferEditorOrientationPolicy(normalizedLandmarks);
    setSchemaName(initialSchema?.name || "");
    setSchemaDescription(initialSchema?.description || "");
    setLandmarks(normalizedLandmarks);
    setOrientationMode(initialPolicy.mode);
    setAnteriorAnchorIds(
      Array.isArray(initialPolicy.anteriorAnchorIds)
        ? initialPolicy.anteriorAnchorIds.map((id) => Number(id)).filter((id) => Number.isFinite(id))
        : []
    );
    setPosteriorAnchorIds(
      Array.isArray(initialPolicy.posteriorAnchorIds)
        ? initialPolicy.posteriorAnchorIds.map((id) => Number(id)).filter((id) => Number.isFinite(id))
        : []
    );
    const nextOtherLabels: Record<number, string> = {};
    normalizedLandmarks.forEach((landmark, index) => {
      const category = String(landmark.category || "").trim();
      if (
        category &&
        !["head", "body", "fins", "wings", "legs", "antennae", "veins", "forewing", "hindwing", "other"].includes(category)
      ) {
        nextOtherLabels[index] = category;
        normalizedLandmarks[index] = { ...normalizedLandmarks[index], category: "other" };
      }
    });
    setLandmarks([...normalizedLandmarks]);
    setOtherLabels(nextOtherLabels);
  }, [initialSchema, open]);

  const addLandmark = () => {
    const newIndex = landmarks.length + 1;
    setLandmarks([
      ...landmarks,
      { index: newIndex, name: `Landmark ${newIndex}`, description: "", category: "body" },
    ]);
  };

  const toggleAnchorId = (
    currentIds: number[],
    setter: React.Dispatch<React.SetStateAction<number[]>>,
    id: number
  ) => {
    setter(
      currentIds.includes(id)
        ? currentIds.filter((value) => value !== id)
        : [...currentIds, id].sort((a, b) => a - b)
    );
  };

  const updateLandmark = (index: number, updates: Partial<LandmarkDefinition>) => {
    setLandmarks(landmarks.map((landmark, position) => (
      position === index ? { ...landmark, ...updates } : landmark
    )));
  };

  const removeLandmark = (index: number) => {
    const removedId = Number(landmarks[index]?.index ?? index + 1);
    const filtered = landmarks.filter((_, position) => position !== index);
    const reindexed = filtered.map((landmark, position) => ({
      ...landmark,
      index: position + 1,
    }));
    setLandmarks(reindexed);
    const nextOther: Record<number, string> = {};
    reindexed.forEach((_, position) => {
      const oldIndex = position >= index ? position + 1 : position;
      if (otherLabels[oldIndex] !== undefined) nextOther[position] = otherLabels[oldIndex];
    });
    setOtherLabels(nextOther);
    const remapAnchorIds = (ids: number[]) =>
      ids
        .filter((id) => id !== removedId)
        .map((id) => (id > removedId ? id - 1 : id));
    setAnteriorAnchorIds((current) => remapAnchorIds(current));
    setPosteriorAnchorIds((current) => remapAnchorIds(current));
  };

  const dialogTitle = useMemo(() => {
    if (mode === "edit-custom") return "Edit Custom Landmark Schema";
    if (mode === "edit-default") return "Customize Preset Schema";
    return "Create Custom Landmark Schema";
  }, [mode]);

  const saveLabel = mode === "edit-custom" ? "Update Schema" : "Save Schema";

  const handleSave = () => {
    if (!schemaName.trim()) {
      toast.error("Schema name is required");
      return;
    }
    if (landmarks.length === 0) {
      toast.error("At least one landmark is required");
      return;
    }
    if (orientationMode !== "invariant" && (anteriorAnchorIds.length === 0 || posteriorAnchorIds.length === 0)) {
      toast.error("Select at least one anterior and one posterior anchor landmark.");
      return;
    }

    const resolvedLandmarks = landmarks.map((landmark, index) => {
      if (landmark.category === "other") {
        const customCategory = (otherLabels[index] || "").trim();
        return { ...landmark, category: customCategory || "other", index: index + 1 };
      }
      return { ...landmark, index: index + 1 };
    });

    onSave({
      ...(initialSchema?.id ? { id: initialSchema.id } : {}),
      ...(initialSchema?.sourcePresetId ? { sourcePresetId: initialSchema.sourcePresetId } : {}),
      name: schemaName.trim(),
      description: schemaDescription.trim() || `Custom schema with ${resolvedLandmarks.length} landmarks`,
      landmarks: resolvedLandmarks,
      orientationPolicy: {
        mode: orientationMode,
        ...(orientationMode === "directional" ? { targetOrientation: "left" as const } : {}),
        ...(orientationMode === "bilateral" ? { bilateralClassAxis: "vertical_obb" as const } : {}),
        ...(anteriorAnchorIds.length > 0 ? { anteriorAnchorIds } : {}),
        ...(posteriorAnchorIds.length > 0 ? { posteriorAnchorIds } : {}),
      },
    });
  };

  return (
    <Dialog open={open} onOpenChange={(isOpen) => !isOpen && onCancel()}>
      <DialogContent className="max-w-3xl flex flex-col">
        <DialogHeader>
          <DialogTitle>{dialogTitle}</DialogTitle>
        </DialogHeader>

        <div className="flex flex-col space-y-4">
          <div className="space-y-3">
            <div>
              <Label>Schema Name</Label>
              <Input
                value={schemaName}
                onChange={(event) => setSchemaName(event.target.value)}
                placeholder="E.g., Beetle Wing Morphometrics"
                className="mt-1"
              />
            </div>
            <div>
              <Label>Description</Label>
              <Input
                value={schemaDescription}
                onChange={(event) => setSchemaDescription(event.target.value)}
                placeholder="Short description shown in the schema picker"
                className="mt-1"
              />
            </div>
            <div>
              <Label>Orientation Schema</Label>
              <select
                value={orientationMode}
                onChange={(event) => setOrientationMode(event.target.value as OrientationPolicy["mode"])}
                className="mt-1 w-full rounded-md border bg-background px-3 py-2 text-sm"
              >
                <option value="directional">Directional</option>
                <option value="bilateral">Bilateral</option>
                <option value="axial">Axial</option>
                <option value="invariant">Invariant</option>
              </select>
            </div>
          </div>

          <div className="flex flex-col">
            <div className="flex items-center justify-between mb-2">
              <Label>Landmarks ({landmarks.length})</Label>
              <motion.div {...buttonHover} {...buttonTap}>
                <Button size="sm" onClick={addLandmark}>
                  <Plus className="h-4 w-4 mr-1" />
                  Add Landmark
                </Button>
              </motion.div>
            </div>

            <ScrollArea className="h-[50vh] border rounded-md p-2">
              <div className="space-y-2">
                {landmarks.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-8 text-center">
                    <p className="text-sm text-muted-foreground mb-2">No landmarks yet</p>
                    <motion.div {...buttonHover} {...buttonTap}>
                      <Button size="sm" variant="outline" onClick={addLandmark}>
                        <Plus className="h-4 w-4 mr-1" />
                        Add First Landmark
                      </Button>
                    </motion.div>
                  </div>
                ) : (
                  landmarks.map((landmark, index) => (
                    <Card key={`${landmark.index}-${index}`} className="p-3">
                      <div className="flex items-start gap-2">
                        <span className="shrink-0 font-mono text-xs font-semibold text-muted-foreground mt-2">
                          {index + 1}
                        </span>
                        <div className="flex-1 space-y-2">
                          <Input
                            value={landmark.name}
                            onChange={(event) => updateLandmark(index, { name: event.target.value })}
                            placeholder="Landmark name"
                            className="text-sm"
                          />
                          <Input
                            value={landmark.description}
                            onChange={(event) => updateLandmark(index, { description: event.target.value })}
                            placeholder="Description (optional)"
                            className="text-xs"
                          />
                          <select
                            value={landmark.category}
                            onChange={(event) => {
                              updateLandmark(index, { category: event.target.value });
                              if (event.target.value !== "other") {
                                setOtherLabels((prev) => {
                                  const next = { ...prev };
                                  delete next[index];
                                  return next;
                                });
                              }
                            }}
                            className="w-full rounded-md border bg-background px-2 py-1 text-xs"
                          >
                            <option value="head">Head</option>
                            <option value="body">Body</option>
                            <option value="fins">Fins</option>
                            <option value="wings">Wings</option>
                            <option value="legs">Legs</option>
                            <option value="antennae">Antennae</option>
                            <option value="veins">Veins</option>
                            <option value="forewing">Forewing</option>
                            <option value="hindwing">Hindwing</option>
                            <option value="other">Other</option>
                          </select>
                          {landmark.category === "other" && (
                            <Input
                              value={otherLabels[index] ?? ""}
                              onChange={(event) =>
                                setOtherLabels((prev) => ({ ...prev, [index]: event.target.value }))
                              }
                              placeholder="Specify category (e.g. scales, gills)"
                              className="text-xs"
                            />
                          )}
                        </div>
                        <motion.div {...buttonHover} {...buttonTap}>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="shrink-0 h-8 w-8 text-destructive hover:text-destructive"
                            onClick={() => removeLandmark(index)}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </motion.div>
                      </div>
                    </Card>
                  ))
                )}
              </div>
            </ScrollArea>
          </div>

          {orientationMode !== "invariant" && landmarks.length > 0 && (
            <div className="space-y-3">
              <div>
                <Label>Anterior Anchors</Label>
                <p className="text-xs text-muted-foreground">
                  Selected landmarks are averaged into an anterior anchor centroid during preannotated import.
                </p>
              </div>
              <ScrollArea className="h-36 rounded-md border bg-muted/20 px-3 py-2">
                <div className="grid grid-cols-2 gap-2 pr-3">
                  {landmarks.map((landmark) => (
                    <label key={`anterior-${landmark.index}`} className="flex items-center gap-2 rounded-md px-1 py-1 text-xs hover:bg-muted/40">
                      <input
                        type="checkbox"
                        checked={anteriorAnchorIds.includes(Number(landmark.index))}
                        onChange={() => toggleAnchorId(anteriorAnchorIds, setAnteriorAnchorIds, Number(landmark.index))}
                      />
                      <span>{landmark.name} (#{landmark.index})</span>
                    </label>
                  ))}
                </div>
              </ScrollArea>
              <div>
                <Label>Posterior Anchors</Label>
                <p className="text-xs text-muted-foreground">
                  Selected landmarks are averaged into a posterior anchor centroid during preannotated import.
                </p>
              </div>
              <ScrollArea className="h-36 rounded-md border bg-muted/20 px-3 py-2">
                <div className="grid grid-cols-2 gap-2 pr-3">
                  {landmarks.map((landmark) => (
                    <label key={`posterior-${landmark.index}`} className="flex items-center gap-2 rounded-md px-1 py-1 text-xs hover:bg-muted/40">
                      <input
                        type="checkbox"
                        checked={posteriorAnchorIds.includes(Number(landmark.index))}
                        onChange={() => toggleAnchorId(posteriorAnchorIds, setPosteriorAnchorIds, Number(landmark.index))}
                      />
                      <span>{landmark.name} (#{landmark.index})</span>
                    </label>
                  ))}
                </div>
              </ScrollArea>
            </div>
          )}
        </div>

        <DialogFooter>
          <motion.div {...buttonHover} {...buttonTap}>
            <Button variant="outline" onClick={onCancel}>
              Cancel
            </Button>
          </motion.div>
          <motion.div {...buttonHover} {...buttonTap}>
            <Button onClick={handleSave}>
              {saveLabel}
            </Button>
          </motion.div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default CustomSchemaEditor;
