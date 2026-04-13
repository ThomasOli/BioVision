import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Microscope, Pencil, Target, FlaskConical, ArrowRight, Sparkles } from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/Components/ui/button";
import { useTutorial } from "./TutorialContext";

const WORKFLOW_STEPS = [
  {
    icon: <Pencil className="h-5 w-5" />,
    title: "Annotate",
    description: "Define schemas, draw boxes, place landmarks",
  },
  {
    icon: <Target className="h-5 w-5" />,
    title: "Train",
    description: "Build dlib or CNN shape predictors",
  },
  {
    icon: <FlaskConical className="h-5 w-5" />,
    title: "Infer & Review",
    description: "Predict, correct, and commit results",
  },
];

export const WelcomeModal: React.FC = () => {
  const { showWelcome, dismissWelcome, startTour } = useTutorial();

  const handleStartTour = () => {
    dismissWelcome();
    startTour("welcome");
  };

  const handleSkip = () => {
    dismissWelcome();
  };

  return (
    <AnimatePresence>
      {showWelcome && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[10000] bg-background/90 backdrop-blur-md"
          />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.94, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.94, y: 20 }}
            transition={{ type: "spring", stiffness: 350, damping: 30 }}
            className="fixed inset-0 z-[10001] flex items-center justify-center p-4"
          >
            <div
              className={cn(
                "relative w-full max-w-lg rounded-2xl border border-border/50",
                "bg-card shadow-2xl overflow-hidden"
              )}
            >
              {/* Decorative gradient header */}
              <div className="relative h-32 bg-gradient-to-br from-primary/20 via-primary/10 to-transparent overflow-hidden">
                <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,hsl(var(--primary)/0.15),transparent_70%)]" />
                <div className="absolute inset-0 flex items-center justify-center">
                  <motion.div
                    initial={{ scale: 0, rotate: -10 }}
                    animate={{ scale: 1, rotate: 0 }}
                    transition={{ type: "spring", stiffness: 300, damping: 20, delay: 0.15 }}
                    className="rounded-2xl bg-primary/15 p-4 backdrop-blur-sm border border-primary/20"
                  >
                    <Microscope className="h-10 w-10 text-primary" />
                  </motion.div>
                </div>
                {/* Floating particles */}
                <motion.div
                  animate={{ y: [0, -8, 0], opacity: [0.3, 0.6, 0.3] }}
                  transition={{ duration: 3, repeat: Infinity }}
                  className="absolute top-6 left-12"
                >
                  <Sparkles className="h-4 w-4 text-primary/40" />
                </motion.div>
                <motion.div
                  animate={{ y: [0, -6, 0], opacity: [0.2, 0.5, 0.2] }}
                  transition={{ duration: 2.5, repeat: Infinity, delay: 0.5 }}
                  className="absolute top-10 right-16"
                >
                  <Sparkles className="h-3 w-3 text-primary/30" />
                </motion.div>
              </div>

              {/* Content */}
              <div className="px-8 pb-8 pt-6 space-y-6">
                <div className="text-center space-y-2">
                  <h1 className="text-2xl font-bold text-foreground">
                    Welcome to BioVision
                  </h1>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    A professional-grade tool for training ML models on biological
                    images. Used by researchers worldwide for morphometric
                    landmarking and shape analysis.
                  </p>
                </div>

                {/* Workflow steps */}
                <div className="flex items-start justify-between gap-2">
                  {WORKFLOW_STEPS.map((step, i) => (
                    <React.Fragment key={step.title}>
                      <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.2 + i * 0.1 }}
                        className="flex-1 text-center space-y-2"
                      >
                        <div className="mx-auto flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
                          {step.icon}
                        </div>
                        <p className="text-xs font-bold text-foreground">
                          {step.title}
                        </p>
                        <p className="text-[11px] text-muted-foreground leading-snug">
                          {step.description}
                        </p>
                      </motion.div>
                      {i < WORKFLOW_STEPS.length - 1 && (
                        <div className="flex items-center pt-4">
                          <ArrowRight className="h-3.5 w-3.5 text-muted-foreground/40" />
                        </div>
                      )}
                    </React.Fragment>
                  ))}
                </div>

                {/* Actions */}
                <div className="flex flex-col gap-3 pt-2">
                  <Button
                    size="lg"
                    onClick={handleStartTour}
                    className="w-full font-semibold gap-2"
                  >
                    <Sparkles className="h-4 w-4" />
                    Take the Guided Tour
                  </Button>
                  <button
                    onClick={handleSkip}
                    className="text-xs text-muted-foreground hover:text-foreground transition-colors py-1"
                  >
                    I'm familiar with BioVision — skip the tour
                  </button>
                </div>

                {/* Footer note */}
                <p className="text-center text-[11px] text-muted-foreground/60">
                  You can replay tours anytime from the{" "}
                  <span className="font-medium text-muted-foreground">
                    Help & Docs
                  </span>{" "}
                  menu
                </p>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};
