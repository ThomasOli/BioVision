import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  X,
  ChevronDown,
  Pencil,
  Target,
  Microscope,
  Keyboard,
  Lightbulb,
  Rocket,
} from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/Components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/Components/ui/card";
import { ScrollArea } from "@/Components/ui/scroll-area";
import { Separator } from "@/Components/ui/separator";
import { buttonHover, buttonTap } from "@/lib/animations";

interface HelpPanelProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

interface AccordionItemProps {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
  defaultOpen?: boolean;
}

const AccordionItem: React.FC<AccordionItemProps> = ({
  title,
  icon,
  children,
  defaultOpen = false,
}) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
      <CardHeader
        className="cursor-pointer pb-3"
        onClick={() => setIsOpen(!isOpen)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="text-primary">{icon}</div>
            <CardTitle className="text-sm font-semibold">{title}</CardTitle>
          </div>
          <ChevronDown
            className={cn(
              "h-4 w-4 text-muted-foreground transition-transform",
              isOpen && "rotate-180"
            )}
          />
        </div>
      </CardHeader>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            <CardContent className="pt-0">{children}</CardContent>
          </motion.div>
        )}
      </AnimatePresence>
    </Card>
  );
};

interface KeyboardShortcutProps {
  keys: string[];
  description: string;
}

const KeyboardShortcut: React.FC<KeyboardShortcutProps> = ({ keys, description }) => (
  <div className="flex items-center justify-between py-1">
    <span className="text-xs text-muted-foreground">{description}</span>
    <div className="flex gap-1">
      {keys.map((key, idx) => (
        <kbd
          key={idx}
          className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs"
        >
          {key}
        </kbd>
      ))}
    </div>
  </div>
);

export const HelpPanel: React.FC<HelpPanelProps> = ({ open, onOpenChange }) => {
  return (
    <AnimatePresence>
      {open && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm"
            onClick={() => onOpenChange(false)}
          />

          {/* Panel */}
          <motion.div
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", damping: 30, stiffness: 300 }}
            className="fixed right-0 top-0 z-50 h-full w-full max-w-md border-l bg-background shadow-xl"
          >
            {/* Header */}
            <div className="flex items-center justify-between border-b p-4">
              <h2 className="text-lg font-bold">Help & Documentation</h2>
              <motion.div {...buttonHover} {...buttonTap}>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => onOpenChange(false)}
                >
                  <X className="h-5 w-5" />
                </Button>
              </motion.div>
            </div>

            {/* Content */}
            <ScrollArea className="h-[calc(100vh-65px)]">
              <div className="space-y-4 p-4">
                {/* Getting Started */}
                <AccordionItem
                  title="Getting Started"
                  icon={<Rocket className="h-4 w-4" />}
                  defaultOpen={true}
                >
                  <div className="space-y-3 text-sm text-muted-foreground">
                    <p>
                      BioVision helps you train machine learning models to automatically
                      detect landmarks on biological images.
                    </p>
                    <div className="space-y-2">
                      <p className="font-medium text-foreground">Quick Start:</p>
                      <ol className="ml-4 list-decimal space-y-1">
                        <li>Import your images using "Annotate Images"</li>
                        <li>Draw bounding boxes around regions of interest</li>
                        <li>Add landmark points within each box</li>
                        <li>Train a model when you have enough annotations</li>
                        <li>Use your model to predict landmarks on new images</li>
                      </ol>
                    </div>
                  </div>
                </AccordionItem>

                {/* Annotation Workflow */}
                <AccordionItem
                  title="Annotation Workflow"
                  icon={<Pencil className="h-4 w-4" />}
                >
                  <div className="space-y-3 text-sm text-muted-foreground">
                    <div className="space-y-2">
                      <p className="font-medium text-foreground">Bounding Boxes</p>
                      <p>
                        Use the Box tool to draw rectangles around regions you want to
                        annotate. Each box represents one instance of your subject.
                      </p>
                    </div>
                    <Separator />
                    <div className="space-y-2">
                      <p className="font-medium text-foreground">Landmarks</p>
                      <p>
                        Switch to the Landmark tool to place points within a selected box.
                        Consistent landmark ordering across all images is important for
                        training quality.
                      </p>
                    </div>
                    <Separator />
                    <div className="space-y-2">
                      <p className="font-medium text-foreground">Selection & Editing</p>
                      <p>
                        Use the Select tool to move boxes or landmarks. Click on a box to
                        select it, then drag to reposition.
                      </p>
                    </div>
                  </div>
                </AccordionItem>

                {/* Training */}
                <AccordionItem
                  title="Training Models"
                  icon={<Target className="h-4 w-4" />}
                >
                  <div className="space-y-3 text-sm text-muted-foreground">
                    <p>
                      BioVision uses dlib's shape predictor to train landmark detection
                      models.
                    </p>
                    <div className="space-y-2">
                      <p className="font-medium text-foreground">Best Practices:</p>
                      <ul className="ml-4 list-disc space-y-1">
                        <li>Annotate at least 20-30 images for basic training</li>
                        <li>Include diverse examples (angles, lighting, sizes)</li>
                        <li>Be consistent with landmark placement order</li>
                        <li>Use clear, versioned model names (e.g., "model_v1")</li>
                      </ul>
                    </div>
                    <div className="rounded-md bg-muted/50 p-3">
                      <p className="text-xs">
                        Training time varies based on the number of images and landmarks.
                        Larger datasets produce more accurate models.
                      </p>
                    </div>
                  </div>
                </AccordionItem>

                {/* Inference */}
                <AccordionItem
                  title="Running Inference"
                  icon={<Microscope className="h-4 w-4" />}
                >
                  <div className="space-y-3 text-sm text-muted-foreground">
                    <p>
                      Once you have a trained model, use it to automatically detect
                      landmarks on new images.
                    </p>
                    <div className="space-y-2">
                      <p className="font-medium text-foreground">Steps:</p>
                      <ol className="ml-4 list-decimal space-y-1">
                        <li>Go to "Run Inference" from the landing page</li>
                        <li>Select your trained model</li>
                        <li>Upload images to analyze</li>
                        <li>Review the predicted landmarks</li>
                        <li>Export results as CSV or JSON</li>
                      </ol>
                    </div>
                  </div>
                </AccordionItem>

                {/* Keyboard Shortcuts */}
                <AccordionItem
                  title="Keyboard Shortcuts"
                  icon={<Keyboard className="h-4 w-4" />}
                >
                  <div className="space-y-1">
                    <KeyboardShortcut keys={["Ctrl", "N"]} description="Upload images" />
                    <KeyboardShortcut keys={["Ctrl", "Z"]} description="Undo" />
                    <KeyboardShortcut keys={["Ctrl", "Y"]} description="Redo" />
                    <KeyboardShortcut keys={["B"]} description="Box tool" />
                    <KeyboardShortcut keys={["L"]} description="Landmark tool" />
                    <KeyboardShortcut keys={["V"]} description="Select tool" />
                    <KeyboardShortcut keys={["Del"]} description="Delete selected" />
                    <KeyboardShortcut keys={["Left"]} description="Previous image" />
                    <KeyboardShortcut keys={["Right"]} description="Next image" />
                    <KeyboardShortcut keys={["?"]} description="Show this help" />
                  </div>
                </AccordionItem>

                {/* Tips for Biologists */}
                <AccordionItem
                  title="Tips for Biologists"
                  icon={<Lightbulb className="h-4 w-4" />}
                >
                  <div className="space-y-3 text-sm text-muted-foreground">
                    <div className="space-y-2">
                      <p className="font-medium text-foreground">Image Preparation</p>
                      <ul className="ml-4 list-disc space-y-1">
                        <li>Use consistent imaging conditions when possible</li>
                        <li>Include scale markers in training images</li>
                        <li>Crop to standard orientations for best results</li>
                      </ul>
                    </div>
                    <Separator />
                    <div className="space-y-2">
                      <p className="font-medium text-foreground">Landmark Selection</p>
                      <ul className="ml-4 list-disc space-y-1">
                        <li>Choose biologically meaningful points</li>
                        <li>Use homologous landmarks across specimens</li>
                        <li>Start with fewer, reliable landmarks</li>
                        <li>Add more landmarks as model accuracy improves</li>
                      </ul>
                    </div>
                    <Separator />
                    <div className="space-y-2">
                      <p className="font-medium text-foreground">Data Management</p>
                      <ul className="ml-4 list-disc space-y-1">
                        <li>Back up your project directory regularly</li>
                        <li>Version your trained models (e.g., species_v1, species_v2)</li>
                        <li>Export landmark data for statistical analysis</li>
                      </ul>
                    </div>
                  </div>
                </AccordionItem>
              </div>
            </ScrollArea>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};

export default HelpPanel;
