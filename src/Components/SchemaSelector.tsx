import { Pencil, Plus } from "lucide-react";
import { motion } from "framer-motion";

import { LandmarkSchema, ReusableSchemaTemplate } from "@/types/Image";
import { DEFAULT_SCHEMAS } from "@/data/defaultSchemas";
import { Button } from "@/Components/ui/button";
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
  customSchemas: ReusableSchemaTemplate[];
  onSelect: (schema: LandmarkSchema | ReusableSchemaTemplate | "custom") => void;
  onEditDefault: (schema: LandmarkSchema) => void;
  onEditCustom: (schema: ReusableSchemaTemplate) => void;
  onCancel: () => void;
}

function SchemaCard({
  name,
  description,
  landmarkCount,
  categories,
  badgeLabel,
  onClick,
  onEdit,
}: {
  name: string;
  description: string;
  landmarkCount: number;
  categories: string[];
  badgeLabel: string;
  onClick: () => void;
  onEdit: () => void;
}) {
  return (
    <motion.div variants={cardHover} initial="initial" whileHover="hover">
      <Card
        className="cursor-pointer border-border/50 hover:border-primary transition-colors"
        onClick={onClick}
      >
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <CardTitle className="text-sm">{name}</CardTitle>
              <p className="text-xs text-muted-foreground mt-1">{description}</p>
            </div>
            <motion.div {...buttonHover} {...buttonTap}>
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8 shrink-0"
                onClick={(event) => {
                  event.stopPropagation();
                  onEdit();
                }}
                title="Edit landmark template"
              >
                <Pencil className="h-4 w-4" />
              </Button>
            </motion.div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-2 mb-2">
            <span className="rounded-full bg-muted px-2 py-0.5 text-[10px] font-semibold text-muted-foreground">
              {badgeLabel}
            </span>
            <p className="text-xs text-muted-foreground">
              {landmarkCount} landmarks
            </p>
          </div>
          <div className="flex flex-wrap gap-1">
            {categories.map((category) => (
              <span
                key={category}
                className="rounded-full bg-primary/10 px-2 py-0.5 text-xs text-primary"
              >
                {category}
              </span>
            ))}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}

export const SchemaSelector: React.FC<SchemaSelectorProps> = ({
  open,
  customSchemas,
  onSelect,
  onEditDefault,
  onEditCustom,
  onCancel,
}) => {
  return (
    <Dialog open={open} onOpenChange={(isOpen) => !isOpen && onCancel()}>
      <DialogContent className="max-w-3xl">
        <DialogHeader>
          <DialogTitle>Choose Landmark Schema</DialogTitle>
          <DialogDescription>
            Select a preset, reuse a saved custom template, or create a new landmark schema.
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-5">
          <div className="grid gap-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold">Preset Schemas</h3>
              <p className="text-xs text-muted-foreground">Built-ins stay immutable; edits save as custom templates.</p>
            </div>
            {DEFAULT_SCHEMAS.map((schema) => {
              const categories = Array.from(new Set(schema.landmarks.map((landmark) => landmark.category || "uncategorized")));
              return (
                <SchemaCard
                  key={schema.id}
                  name={schema.name}
                  description={schema.description}
                  landmarkCount={schema.landmarks.length}
                  categories={categories}
                  badgeLabel="Preset"
                  onClick={() => onSelect(schema)}
                  onEdit={() => onEditDefault(schema)}
                />
              );
            })}
          </div>

          <div className="grid gap-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold">Custom Templates</h3>
              <motion.div {...buttonHover} {...buttonTap}>
                <Button size="sm" onClick={() => onSelect("custom")}>
                  <Plus className="mr-1 h-4 w-4" />
                  Create Custom Schema
                </Button>
              </motion.div>
            </div>

            {customSchemas.length === 0 ? (
              <Card className="border-dashed border-border/50">
                <CardContent className="flex flex-col items-center justify-center gap-2 p-6 text-center">
                  <Plus className="h-8 w-8 text-muted-foreground" />
                  <div>
                    <p className="text-sm font-semibold">No saved custom schemas yet</p>
                    <p className="text-xs text-muted-foreground mt-1">
                      Create one or edit a preset to save a reusable custom template.
                    </p>
                  </div>
                </CardContent>
              </Card>
            ) : (
              customSchemas.map((schema) => {
                const categories = Array.from(new Set(schema.landmarks.map((landmark) => landmark.category || "uncategorized")));
                return (
                  <SchemaCard
                    key={schema.id}
                    name={schema.name}
                    description={schema.description}
                    landmarkCount={schema.landmarks.length}
                    categories={categories}
                    badgeLabel="Custom"
                    onClick={() => onSelect(schema)}
                    onEdit={() => onEditCustom(schema)}
                  />
                );
              })
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default SchemaSelector;
