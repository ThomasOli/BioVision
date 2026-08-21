import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Microscope,
  Pencil,
  Target,
  Database,
  Rocket,
  ChevronRight,
  ChevronLeft,
} from "lucide-react";

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/Components/ui/dialog";
import { Button } from "@/Components/ui/button";
import { buttonHover, buttonTap } from "@/lib/animations";

interface OnboardingGuideProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const steps = [
  {
    icon: <Microscope className="h-6 w-6" />,
    title: "Welcome to BioVision",
    content: (
      <div className="space-y-3 text-sm text-muted-foreground">
        <p>
          BioVision helps you train machine learning models for biological image
          landmarking. Whether you're studying morphology, tracking specimens, or
          measuring anatomical features — this tool streamlines the full workflow.
        </p>
        <p>
          This guide walks you through the four core steps: creating a schema,
          annotating images, training a model, and running inference.
        </p>
      </div>
    ),
  },
  {
    icon: <Pencil className="h-6 w-6" />,
    title: "Create a Schema",
    content: (
      <div className="space-y-3 text-sm text-muted-foreground">
        <p>
          A <span className="font-medium text-foreground">schema</span> defines
          the set of landmark points you want to track on your specimens.
        </p>
        <ul className="ml-4 list-disc space-y-1">
          <li>Choose from preset schemas for common organisms</li>
          <li>Or create a custom schema with your own landmark definitions</li>
          <li>
            Select an orientation policy (Directional, Bilateral, Axial, or
            Invariant) that matches your specimen's geometry
          </li>
        </ul>
        <p>
          Click{" "}
          <span className="font-medium text-foreground">Annotate Images</span>{" "}
          on the home screen to get started.
        </p>
      </div>
    ),
  },
  {
    icon: <Target className="h-6 w-6" />,
    title: "Annotate Your Images",
    content: (
      <div className="space-y-3 text-sm text-muted-foreground">
        <p>
          Upload your images and annotate them with bounding boxes and landmark
          points.
        </p>
        <ul className="ml-4 list-disc space-y-1">
          <li>
            Use <span className="font-medium text-foreground">Auto Detection</span>{" "}
            to find specimens automatically, or draw boxes manually
          </li>
          <li>Place landmarks consistently inside each accepted box</li>
          <li>Use the magnified zoom view for precision placement</li>
          <li>Finalize your annotations when boxes and landmarks look correct</li>
        </ul>
        <p>
          Landmark order and consistency are critical for stable model training.
        </p>
      </div>
    ),
  },
  {
    icon: <Database className="h-6 w-6" />,
    title: "Train a Model",
    content: (
      <div className="space-y-3 text-sm text-muted-foreground">
        <p>
          Once you have enough annotated images, train a model to predict
          landmarks automatically.
        </p>
        <ul className="ml-4 list-disc space-y-1">
          <li>Train the OBB detector first (if you have bounding box annotations)</li>
          <li>
            Then train a landmark predictor:{" "}
            <span className="font-medium text-foreground">dlib</span> (faster,
            great for standardized imaging) or{" "}
            <span className="font-medium text-foreground">CNN</span> (slower,
            better for varied images)
          </li>
          <li>Review preflight settings before starting long training runs</li>
        </ul>
      </div>
    ),
  },
  {
    icon: <Rocket className="h-6 w-6" />,
    title: "Run Inference",
    content: (
      <div className="space-y-3 text-sm text-muted-foreground">
        <p>
          Apply your trained model to new images and review the predictions.
        </p>
        <ul className="ml-4 list-disc space-y-1">
          <li>Open the Inference hub and select your trained model</li>
          <li>Run detection and landmark prediction on new images</li>
          <li>Review and correct predictions as needed</li>
          <li>
            Mark images as{" "}
            <span className="font-medium text-foreground">Review Complete</span>,
            then commit them back to your training data
          </li>
        </ul>
        <p>
          This review-and-commit loop continuously improves your model over time.
        </p>
      </div>
    ),
  },
];

export const OnboardingGuide: React.FC<OnboardingGuideProps> = ({
  open,
  onOpenChange,
}) => {
  const [step, setStep] = useState(0);
  const [dontShowAgain, setDontShowAgain] = useState(false);
  const isLast = step === steps.length - 1;
  const current = steps[step];

  const handleDismiss = () => {
    if (dontShowAgain) {
      localStorage.setItem("biovision-onboarding-dismissed", "true");
    }
    setStep(0);
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={(val) => { if (!val) handleDismiss(); }}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <div className="flex items-center gap-2">
            <div className="text-primary">{current.icon}</div>
            <DialogTitle>{current.title}</DialogTitle>
          </div>
          <DialogDescription>
            Step {step + 1} of {steps.length}
          </DialogDescription>
        </DialogHeader>

        <AnimatePresence mode="wait">
          <motion.div
            key={step}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.2 }}
            className="min-h-[140px]"
          >
            {current.content}
          </motion.div>
        </AnimatePresence>

        <DialogFooter className="sm:justify-between">
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="sm" onClick={handleDismiss}>
              Skip
            </Button>
            <label className="flex items-center gap-1.5 text-xs text-muted-foreground cursor-pointer select-none">
              <input
                type="checkbox"
                checked={dontShowAgain}
                onChange={(e) => setDontShowAgain(e.target.checked)}
                className="rounded border-muted-foreground/30"
              />
              Don't show again
            </label>
          </div>
          <div className="flex items-center gap-3">
            {/* Dot indicators */}
            <div className="flex gap-1.5">
              {steps.map((_, i) => (
                <div
                  key={i}
                  className={`h-1.5 w-1.5 rounded-full transition-colors ${
                    i === step ? "bg-primary" : "bg-muted-foreground/30"
                  }`}
                />
              ))}
            </div>
            <div className="flex gap-2">
              <motion.div {...buttonHover} {...buttonTap}>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setStep(step - 1)}
                  disabled={step === 0}
                >
                  <ChevronLeft className="h-4 w-4" />
                  Back
                </Button>
              </motion.div>
              <motion.div {...buttonHover} {...buttonTap}>
                <Button
                  size="sm"
                  onClick={() => {
                    if (isLast) {
                      handleDismiss();
                    } else {
                      setStep(step + 1);
                    }
                  }}
                >
                  {isLast ? "Get Started" : "Next"}
                  {!isLast && <ChevronRight className="h-4 w-4" />}
                </Button>
              </motion.div>
            </div>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default OnboardingGuide;
