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
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm"
            onClick={() => onOpenChange(false)}
          />

          <div className="fixed inset-0 z-50 flex items-center justify-center pointer-events-none">
          <motion.div
            initial={{ opacity: 0, scale: 0.96, y: 12 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: 12 }}
            transition={{ type: "spring", damping: 30, stiffness: 320 }}
            className="w-full max-w-2xl rounded-lg border bg-background shadow-xl pointer-events-auto"
          >
            <div className="flex items-center justify-between border-b p-4">
              <h2 className="font-display text-base font-semibold">Help & Documentation</h2>
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

            <ScrollArea className="h-[75vh]">
              <div className="space-y-4 p-4">
                <AccordionItem
                  title="Getting Started"
                  icon={<Rocket className="h-4 w-4" />}
                  defaultOpen={true}
                >
                  <div className="space-y-3 text-sm text-muted-foreground">
                    <p>
                      BioVision uses a commit-now, train-later loop: annotate and train inside a schema
                      session, review predictions in inference, then commit reviewed images back to schema
                      training data.
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
                        <li>Choose orientation policy (Directional/Bilateral/Axial/Invariant)</li>
                        <li>Upload images and annotate using Manual OBB or Auto detection</li>
                        <li>Finalize accepted boxes and landmarks</li>
                        <li>Train OBB detector (when applicable), then train dlib or CNN</li>
                        <li>Open Inference hub and create the schema-bound inference session</li>
                        <li>Run detection and landmark inference, review, then mark images complete</li>
                        <li>Commit review-complete images to training data, then retrain later from Annotate/Train</li>
                      </ol>
                    </div>
                  </div>
                </AccordionItem>

                <AccordionItem
                  title="Annotation Workflow"
                  icon={<Pencil className="h-4 w-4" />}
                >
                  <div className="space-y-3 text-sm text-muted-foreground">
                    <div className="space-y-2">
                      <p className="font-medium text-foreground">Auto Detection</p>
                      <p>
                        Auto mode uses the session OBB detector, then optionally refines masks with SAM2.
                        Use Auto-Detect, then correct or delete bad boxes before finalizing.
                      </p>
                    </div>
                    <Separator />
                    <div className="space-y-2">
                      <p className="font-medium text-foreground">Manual OBB Drawing</p>
                      <p>
                        Manual mode draws boxes directly. Select a box to move, resize, and rotate it
                        into an oriented bounding box (OBB). In directional schemas, use the head-direction
                        toggle so new boxes start with the intended default orientation.
                      </p>
                    </div>
                    <Separator />
                    <div className="space-y-2">
                      <p className="font-medium text-foreground">SAM2 Segmentation (Optional)</p>
                      <p>
                        SAM2 refinement requires GPU acceleration (CUDA or Apple MPS) and at least 8 GB RAM,
                        and is only available in high-performance mode. Lite mode still supports detection
                        without SAM2.
                      </p>
                    </div>
                    <Separator />
                    <div className="space-y-2">
                      <p className="font-medium text-foreground">Landmark Placement</p>
                      <p>
                        Place landmarks consistently inside each accepted box. Landmark order and landmark ID
                        consistency are critical for stable training.
                      </p>
                    </div>
                    <Separator />
                    <div className="space-y-2">
                      <p className="font-medium text-foreground">Correction and Finalization</p>
                      <p>
                        In Auto mode, correction mode lets you redraw and resize selected detections.
                        Deleting bad auto detections records hard negatives for future detector fine-tuning.
                        Finalize only when boxes and landmarks are accepted.
                      </p>
                    </div>
                    <Separator />
                    <div className="space-y-2">
                      <p className="font-medium text-foreground">Orientation Schemas</p>
                      <p>
                        Directional is for strict head/tail objects. Bilateral is for up/down symmetry
                        along a primary biological axis. Axial is for elongated specimens with an axis but
                        no true polarity. Invariant is for objects with no stable directional axis.
                      </p>
                      <p>
                        The OBB detector levels specimen crops to a canonical orientation before landmark
                        prediction. For directional schemas, class_id encodes left/right; for bilateral it
                        encodes up/down; axial and invariant detector export stay one-class.
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
                      You can import pre-annotated data into a schema session before continuing manual
                      annotation and review.
                    </p>
                    <ul className="ml-4 list-disc space-y-1">
                      <li>Supported: JSON with boxes and landmarks</li>
                      <li>Supported: landmarks-only JSON (box is auto-derived and validated)</li>
                      <li>Supported: annotated image folder import with matched label files</li>
                    </ul>
                    <div className="rounded-md bg-muted/50 p-3">
                      <p className="text-xs">
                        Imported labels and in-app edits are merged into one session label store,
                        so training uses one consistent schema-scoped dataset.
                      </p>
                    </div>
                  </div>
                </AccordionItem>

                <AccordionItem
                  title="Training Models"
                  icon={<Target className="h-4 w-4" />}
                >
                  <div className="space-y-3 text-sm text-muted-foreground">
                    <p>
                      Landmark training supports <span className="font-medium text-foreground">dlib</span> and{" "}
                      <span className="font-medium text-foreground">CNN</span> (SimpleBaseline default with
                      system-gated variants). Detection fine-tuning uses schema-scoped YOLO OBB checkpoints.
                    </p>
                    <div className="space-y-2">
                      <p className="font-medium text-foreground">Training Order</p>
                      <ol className="ml-4 list-decimal space-y-1">
                        <li>Train OBB detector first when OBB annotations are present</li>
                        <li>Train landmark predictor second (dlib or CNN)</li>
                        <li>Use inference commits to expand data, then retrain later</li>
                      </ol>
                    </div>
                    <Separator />
                    <div className="space-y-2">
                      <p className="font-medium text-foreground">Predictor Guidance</p>
                      <ul className="ml-4 list-disc space-y-1">
                        <li>dlib: faster training, strongest on standardized imaging</li>
                        <li>CNN: slower training, stronger under higher image variation</li>
                        <li>CNN variant and augmentation controls are in the training dialog</li>
                        <li>Review preflight settings before long runs</li>
                      </ul>
                    </div>
                    <div className="rounded-md bg-muted/50 p-3">
                      <p className="text-xs">
                        Training is staged (dataset prep, fit, finalize). Models and
                        detector artifacts remain scoped to the current schema session.
                      </p>
                    </div>
                  </div>
                </AccordionItem>

                <AccordionItem
                  title="Running Inference"
                  icon={<Microscope className="h-4 w-4" />}
                >
                  <div className="space-y-3 text-sm text-muted-foreground">
                    <p>
                      Inference is a schema-bound review workflow. Open a session from the card hub,
                      run detection and landmark prediction, then commit reviewed results back to
                      schema training data.
                    </p>
                    <div className="space-y-2">
                      <p className="font-medium text-foreground">Steps</p>
                      <ol className="ml-4 list-decimal space-y-1">
                        <li>Open Inference hub and enter the schema session card</li>
                        <li>Select any available landmark model for that schema (dlib or CNN)</li>
                        <li>Add images (saved image paths auto-reload when files still exist)</li>
                        <li>Run detection, then correct boxes and orientation when needed</li>
                        <li>Run landmark inference on accepted boxes</li>
                        <li>Use <span className="font-medium text-foreground">Save All Changes</span> to persist drafts</li>
                        <li>Use <span className="font-medium text-foreground">Mark Review Complete</span> or <span className="font-medium text-foreground">Mark In Progress</span></li>
                        <li>Use <span className="font-medium text-foreground">Commit to Training Data</span> to commit only review-complete images</li>
                      </ol>
                    </div>
                    <Separator />
                    <div className="space-y-2">
                      <p className="font-medium text-foreground">Session Rules</p>
                      <ul className="ml-4 list-disc space-y-1">
                        <li>One inference session exists per schema</li>
                        <li>Inference hub is available once a session with a trained model is selected</li>
                        <li>Draft edits and landmark corrections are autosaved in-session</li>
                        <li>Retraining is manual from Annotate/Train after commits</li>
                      </ul>
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
                      <li>Create a fresh schema session and upload 10-20 images</li>
                      <li>Set orientation policy and run Auto detection on 3-5 images</li>
                      <li>Correct and rotate OBB boxes, then finalize accepted boxes</li>
                      <li>Annotate landmarks and verify landmark ID consistency</li>
                      <li>Train OBB detector (if OBB labels exist), then train dlib or CNN</li>
                      <li>Open inference hub and create or open the schema inference session</li>
                      <li>Run detection plus landmark inference on unseen images</li>
                      <li>Mark some images Review Complete and keep some In Progress</li>
                      <li>Commit review-complete images to training data</li>
                      <li>Return to Annotate/Train and confirm retraining uses committed additions</li>
                    </ol>
                    <div className="rounded-md bg-muted/50 p-3">
                      <p className="text-xs">
                        Minimum acceptance: no import or validation errors, no training crash, inference review
                        persists across reopen, and only review-complete images are committed.
                      </p>
                    </div>
                  </div>
                </AccordionItem>

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
                        <li>Version your trained models (for example, species_v1, species_v2)</li>
                        <li>Export landmark data for statistical analysis</li>
                      </ul>
                    </div>
                  </div>
                </AccordionItem>
              </div>
            </ScrollArea>
          </motion.div>
          </div>
        </>
      )}
    </AnimatePresence>
  );
};

export default HelpPanel;
