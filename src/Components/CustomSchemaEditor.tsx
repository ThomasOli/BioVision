import React, { useState } from "react";
import { Plus, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { motion } from "framer-motion";

import { LandmarkDefinition, LandmarkSchema } from "@/types/Image";
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

interface CustomSchemaEditorProps {
  open: boolean;
  onSave: (schema: LandmarkSchema) => void;
  onCancel: () => void;
}

export const CustomSchemaEditor: React.FC<CustomSchemaEditorProps> = ({ open, onSave, onCancel }) => {
  const [schemaName, setSchemaName] = useState("");
  const [landmarks, setLandmarks] = useState<LandmarkDefinition[]>([]);

  const addLandmark = () => {
    const newIndex = landmarks.length;
    setLandmarks([
      ...landmarks,
      { index: newIndex, name: `Landmark ${newIndex}`, description: "", category: "body" },
    ]);
  };

  const updateLandmark = (index: number, updates: Partial<LandmarkDefinition>) => {
    setLandmarks(landmarks.map((lm, i) => (i === index ? { ...lm, ...updates } : lm)));
  };

  const removeLandmark = (index: number) => {
    const filtered = landmarks.filter((_, i) => i !== index);
    // Re-index remaining landmarks
    const reindexed = filtered.map((lm, i) => ({ ...lm, index: i }));
    setLandmarks(reindexed);
  };

  const handleSave = () => {
    if (!schemaName.trim()) {
      toast.error("Schema name is required");
      return;
    }

    if (landmarks.length === 0) {
      toast.error("At least one landmark is required");
      return;
    }

    onSave({
      id: `custom-${Date.now()}`,
      name: schemaName,
      description: `Custom schema with ${landmarks.length} landmarks`,
      landmarks,
    });

    // Reset form
    setSchemaName("");
    setLandmarks([]);
  };

  return (
    <Dialog open={open} onOpenChange={(isOpen) => !isOpen && onCancel()}>
      <DialogContent className="max-w-3xl max-h-[80vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle>Create Custom Landmark Schema</DialogTitle>
        </DialogHeader>

        <div className="flex-1 overflow-hidden flex flex-col space-y-4">
          {/* Schema Name */}
          <div>
            <Label>Schema Name</Label>
            <Input
              value={schemaName}
              onChange={(e) => setSchemaName(e.target.value)}
              placeholder="E.g., Beetle Wing Morphometrics"
              className="mt-1"
            />
          </div>

          {/* Landmarks List */}
          <div className="flex-1 overflow-hidden flex flex-col">
            <div className="flex items-center justify-between mb-2">
              <Label>Landmarks ({landmarks.length})</Label>
              <motion.div {...buttonHover} {...buttonTap}>
                <Button size="sm" onClick={addLandmark}>
                  <Plus className="h-4 w-4 mr-1" />
                  Add Landmark
                </Button>
              </motion.div>
            </div>

            <ScrollArea className="flex-1 border rounded-md p-2">
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
                  landmarks.map((lm, idx) => (
                    <Card key={idx} className="p-3">
                      <div className="flex items-start gap-2">
                        <span className="shrink-0 font-mono text-xs font-semibold text-muted-foreground mt-2">
                          {idx}
                        </span>
                        <div className="flex-1 space-y-2">
                          <Input
                            value={lm.name}
                            onChange={(e) => updateLandmark(idx, { name: e.target.value })}
                            placeholder="Landmark name"
                            className="text-sm"
                          />
                          <Input
                            value={lm.description}
                            onChange={(e) => updateLandmark(idx, { description: e.target.value })}
                            placeholder="Description (optional)"
                            className="text-xs"
                          />
                          <select
                            value={lm.category}
                            onChange={(e) => updateLandmark(idx, { category: e.target.value })}
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
                        </div>
                        <motion.div {...buttonHover} {...buttonTap}>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="shrink-0 h-8 w-8 text-destructive hover:text-destructive"
                            onClick={() => removeLandmark(idx)}
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
        </div>

        <DialogFooter>
          <motion.div {...buttonHover} {...buttonTap}>
            <Button variant="outline" onClick={onCancel}>
              Cancel
            </Button>
          </motion.div>
          <motion.div {...buttonHover} {...buttonTap}>
            <Button onClick={handleSave}>
              Save Schema
            </Button>
          </motion.div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default CustomSchemaEditor;
