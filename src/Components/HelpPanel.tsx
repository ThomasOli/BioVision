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
  ListChecks,
  Database,
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
                      BioVision supports a full annotation loop:
                      detection (YOLO/CV), user correction, landmarking, and model training.
                    </p>
                    <div className="space-y-2">
                      <p className="font-medium text-foreground">Quick Start:</p>
                      <ol className="ml-4 list-decimal space-y-1">
                        <li>Create or resume a session from Annotate Images</li>
                        <li>Upload images and optionally import pre-annotated labels/XML</li>
                        <li>Choose Manual or Auto detection and annotate landmarks</li>
                        <li>Train landmark model (dlib) and optionally fine-tune session detector (YOLO)</li>
                        <li>Run inference on unseen images and export CSV/JSON results</li>
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
                        Manual mode: no auto boxes are shown; draw boxes yourself.
                        Auto mode: click Auto-detect, then accept/correct/delete detected boxes.
                      </p>
                    </div>
                    <Separator />
                    <div className="space-y-2">
                      <p className="font-medium text-foreground">Landmarks</p>
                      <p>
                        Place landmarks consistently inside each accepted box.
                        Landmark order/ID consistency is critical for reliable dlib training.
                      </p>
                    </div>
                    <Separator />
                    <div className="space-y-2">
                      <p className="font-medium text-foreground">Correction Loop</p>
                      <p>
                        In Auto mode, correction mode lets you redraw/resize selected boxes.
                        Deleting bad auto boxes records hard negatives for future YOLO fine-tuning.
                      </p>
                    </div>
                  </div>
                </AccordionItem>

                <AccordionItem
                  title="Pre-Annotated Import"
                  icon={<Database className="h-4 w-4" />}
                >
                  <div className="space-y-3 text-sm text-muted-foreground">
                    <p>
                      You can import pre-annotated datasets into a session before continuing annotation.
                    </p>
                    <ul className="ml-4 list-disc space-y-1">
                      <li>Supported: JSON with boxes + landmarks</li>
                      <li>Supported: landmarks-only JSON (box is auto-derived and validated)</li>
                      <li>Advanced: import dlib XML for direct XML-based training</li>
                    </ul>
                    <div className="rounded-md bg-muted/50 p-3">
                      <p className="text-xs">
                        Imported labels and in-app edits are merged in the same session labels.
                        For dlib, training consumes XML; for YOLO detection, training consumes session label JSON.
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
                      Landmark training uses dlib shape predictor. Detection fine-tuning
                      uses session YOLO checkpoints with versioning and promotion.
                    </p>
                    <div className="space-y-2">
                      <p className="font-medium text-foreground">Best Practices:</p>
                      <ul className="ml-4 list-disc space-y-1">
                        <li>Annotate at least 20-30 images for basic training</li>
                        <li>Include diverse examples (angles, lighting, sizes)</li>
                        <li>Be consistent with landmark placement order</li>
                        <li>Use detection presets based on goal: Precision/Recall/Single Object</li>
                      </ul>
                    </div>
                    <div className="rounded-md bg-muted/50 p-3">
                      <p className="text-xs">
                        YOLO training keeps versioned candidates and promotes only if validation
                        quality is not worse than the active detector.
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
                      Run inference with your trained dlib model to predict landmarks on new images.
                    </p>
                    <div className="space-y-2">
                      <p className="font-medium text-foreground">Steps:</p>
                      <ol className="ml-4 list-decimal space-y-1">
                        <li>Go to "Run Inference" from the landing page</li>
                        <li>Select your trained model</li>
                        <li>Upload images to analyze</li>
                        <li>Review predicted landmarks and confidence</li>
                        <li>Export results as CSV or JSON</li>
                      </ol>
                    </div>
                  </div>
                </AccordionItem>

                <AccordionItem
                  title="End-to-End Test Pipeline"
                  icon={<ListChecks className="h-4 w-4" />}
                >
                  <div className="space-y-3 text-sm text-muted-foreground">
                    <p className="font-medium text-foreground">Session QA Checklist</p>
                    <ol className="ml-4 list-decimal space-y-1">
                      <li>Create a fresh session and upload 10-20 images</li>
                      <li>Import pre-annotated JSON (include one landmarks-only sample)</li>
                      <li>Verify imported images/boxes/landmarks appear in workspace</li>
                      <li>Auto-detect on 3-5 images using each preset (Balanced/Precision/Recall/Single)</li>
                      <li>Correct and delete some auto boxes, then annotate landmarks</li>
                      <li>Train YOLO detection once, then annotate more images and train again</li>
                      <li>Confirm second YOLO run creates a new version and reports promoted/not promoted</li>
                      <li>Run dlib train preflight, then train dlib model</li>
                      <li>Open Inference page, run model on unseen images, export CSV and JSON</li>
                      <li>Spot-check outputs: landmark IDs, coordinates, and image filenames</li>
                    </ol>
                    <div className="rounded-md bg-muted/50 p-3">
                      <p className="text-xs">
                        Minimum acceptance: no import validation errors, no training crash,
                        inference exports complete, and qualitative landmark placement is stable.
                      </p>
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
