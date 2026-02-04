import React from "react";
import { Plus } from "lucide-react";
import { motion } from "framer-motion";

import { DEFAULT_SCHEMAS } from "@/data/defaultSchemas";
import { LandmarkSchema } from "@/types/Image";
import { Card, CardContent, CardHeader, CardTitle } from "@/Components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/Components/ui/dialog";
import { cardHover, buttonHover, buttonTap } from "@/lib/animations";

interface SchemaSelectorProps {
  open: boolean;
  onSelect: (schema: LandmarkSchema | "custom") => void;
  onCancel: () => void;
}

export const SchemaSelector: React.FC<SchemaSelectorProps> = ({ open, onSelect, onCancel }) => {
  return (
    <Dialog open={open} onOpenChange={(isOpen) => !isOpen && onCancel()}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Choose Landmark Schema</DialogTitle>
          <DialogDescription>
            Select a predefined schema or create your own custom landmarks
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-4">
          {/* Default Schemas */}
          {DEFAULT_SCHEMAS.map((schema) => {
            const categories = Array.from(new Set(schema.landmarks.map((l) => l.category)));

            return (
              <motion.div
                key={schema.id}
                variants={cardHover}
                initial="initial"
                whileHover="hover"
              >
                <Card
                  className="cursor-pointer border-border/50 hover:border-primary transition-colors"
                  onClick={() => onSelect(schema)}
                >
                  <CardHeader>
                    <CardTitle className="text-sm">{schema.name}</CardTitle>
                    <p className="text-xs text-muted-foreground">{schema.description}</p>
                  </CardHeader>
                  <CardContent>
                    <p className="text-xs text-muted-foreground mb-2">
                      {schema.landmarks.length} landmarks
                    </p>
                    <div className="flex flex-wrap gap-1">
                      {categories.map((cat) => (
                        <span
                          key={cat}
                          className="rounded-full bg-primary/10 px-2 py-0.5 text-xs text-primary"
                        >
                          {cat}
                        </span>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            );
          })}

          {/* Custom Schema Option */}
          <motion.div
            variants={cardHover}
            initial="initial"
            whileHover="hover"
          >
            <Card
              className="cursor-pointer border-dashed border-border/50 hover:border-primary transition-colors"
              onClick={() => onSelect("custom")}
            >
              <CardContent className="flex items-center justify-center p-6">
                <div className="text-center">
                  <motion.div {...buttonHover} {...buttonTap}>
                    <Plus className="mx-auto h-8 w-8 text-muted-foreground mb-2" />
                  </motion.div>
                  <p className="text-sm font-semibold">Create Custom Schema</p>
                  <p className="text-xs text-muted-foreground mt-1">
                    Define your own landmark template
                  </p>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default SchemaSelector;
