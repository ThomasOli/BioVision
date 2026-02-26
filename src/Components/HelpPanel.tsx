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
                      BioVision supports a full loop: detection, correction, landmarking,
                      training, inference review, and retraining from accepted edits.
                    </p>
                    <div className="space-y-2">
                      <p className="font-medium text-foreground">Install:</p>
                      <ul className="ml-4 list-disc space-y-1">
                        <li>Windows: run <code>setup.bat</code></li>
                        <li>macOS/Linux: run <code>bash setup.sh</code></li>
                      </ul>
                    </div>
                    <div className="space-y-2">
                      <p className="font-medium text-foreground">Quick Start:</p>
                      <ol className="ml-4 list-decimal space-y-1">
                        <li>Create or resume a schema session</li>
                        <li>Set orientation policy (Directional/Bilateral/Axial/Invariant)</li>
                        <li>Upload images and optionally import pre-annotated labels/XML</li>
                        <li>Detect boxes (manual/auto), correct, then annotate landmarks</li>
                        <li>Train landmark model (dlib or CNN) and optionally fine-tune detector</li>
                        <li>Run inference, review edits, then save/queue retrain</li>
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
                        Landmark order/ID consistency is critical for reliable landmark model training.
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
                    <Separator />
                    <div className="space-y-2">
                      <p className="font-medium text-foreground">Orientation Schemas</p>
                      <p>
                        Directional is for strict head/tail objects; Mirrored/Bilateral is for paired left-right structures;
                        Axial is for elongated rotating specimens; Invariant is for radial/no-stable-direction objects.
                      </p>
                      <p>
                        The OBB detector levels specimen crops to a canonical orientation before landmark
                        prediction. class_id (0 = canonical, 1 = mirrored) is tagged at annotation time and
                        corrects orientation automatically during both training and inference.
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
                      Landmark training supports <span className="font-medium text-foreground">dlib</span> and{" "}
                      <span className="font-medium text-foreground">CNN</span> (SimpleBaseline default, with
                      system-gated variants). Detection fine-tuning uses session-scoped YOLO checkpoints.
                    </p>
                    <div className="space-y-2">
                      <p className="font-medium text-foreground">Best Practices:</p>
                      <ul className="ml-4 list-disc space-y-1">
                        <li>Annotate at least 30+ images for stable first models</li>
                        <li>Include diverse examples (angles, lighting, sizes)</li>
                        <li>Be consistent with landmark placement order</li>
                        <li>Use SAM2 consistently between training and inference when possible</li>
                        <li>Review preflight warnings before starting long training runs</li>
                      </ul>
                    </div>
                    <div className="rounded-md bg-muted/50 p-3">
                      <p className="text-xs">
                        Training progress is staged (dataset prep, fit, parity evaluation, finalize).
                        Session models remain scoped to the current schema session.
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
                      Inference is a review workflow: detect, correct, segment (optional), landmark, then persist.
                    </p>
                    <div className="space-y-2">
                      <p className="font-medium text-foreground">Steps:</p>
                      <ol className="ml-4 list-decimal space-y-1">
                        <li>Select a trained model tied to the current schema session</li>
                        <li>Upload images and run detection</li>
                        <li>Edit/add/delete boxes and set orientation overrides when needed</li>
                        <li>Optionally run SAM2 segmentation and mask overlay review</li>
                        <li>Run landmark inference on accepted boxes</li>
                        <li>Use <span className="font-medium text-foreground">Save all changes</span> to persist edits</li>
                        <li>Use <span className="font-medium text-foreground">Queue retrain</span> to stage finalized images for next training run</li>
                      </ol>
                    </div>
                    <div className="rounded-md bg-muted/50 p-3">
                      <p className="text-xs">
                        Compatibility gate: if a model was trained with OBB canonicalization and no OBB detector is available
                        at inference time, BioVision warns and requires explicit override.
                      </p>
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
                      <li>Run training preflight, then train dlib or CNN model</li>
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
                        <li>Choose an orientation schema that matches specimen geometry before annotating</li>
                        <li>When OBB annotations are present, the geometry engine levels specimens to canonical orientation more reliably than PCA-based approaches</li>
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
