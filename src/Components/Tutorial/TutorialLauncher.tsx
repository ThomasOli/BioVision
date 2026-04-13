import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  X,
  Rocket,
  Pencil,
  Target,
  Microscope,
  CheckCircle2,
  PlayCircle,
  GraduationCap,
} from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/Components/ui/button";
import { Card, CardContent } from "@/Components/ui/card";
import { ScrollArea } from "@/Components/ui/scroll-area";
import { Separator } from "@/Components/ui/separator";
import { buttonHover, buttonTap } from "@/lib/animations";
import { useTutorial } from "./TutorialContext";

const TOUR_ICONS: Record<string, React.ReactNode> = {
  Rocket: <Rocket className="h-5 w-5" />,
  Pencil: <Pencil className="h-5 w-5" />,
  Target: <Target className="h-5 w-5" />,
  Microscope: <Microscope className="h-5 w-5" />,
};

export const TutorialLauncher: React.FC = () => {
  const {
    launcherOpen,
    setLauncherOpen,
    tours,
    startTour,
    isTourCompleted,
  } = useTutorial();

  const completedCount = tours.filter((t) => isTourCompleted(t.id)).length;

  return (
    <AnimatePresence>
      {launcherOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm"
            onClick={() => setLauncherOpen(false)}
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
              <div className="flex items-center gap-2.5">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10 text-primary">
                  <GraduationCap className="h-4.5 w-4.5" />
                </div>
                <div>
                  <h2 className="text-base font-bold">Interactive Tutorials</h2>
                  <p className="text-[11px] text-muted-foreground">
                    {completedCount} of {tours.length} completed
                  </p>
                </div>
              </div>
              <motion.div {...buttonHover} {...buttonTap}>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setLauncherOpen(false)}
                >
                  <X className="h-5 w-5" />
                </Button>
              </motion.div>
            </div>

            {/* Overall progress */}
            <div className="px-4 pt-4 pb-2">
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                <motion.div
                  className="h-full rounded-full bg-primary"
                  initial={{ width: 0 }}
                  animate={{ width: `${(completedCount / Math.max(tours.length, 1)) * 100}%` }}
                  transition={{ duration: 0.5, ease: "easeOut" }}
                />
              </div>
            </div>

            <ScrollArea className="h-[calc(100vh-130px)]">
              <div className="space-y-3 p-4">
                {/* Recommended order banner */}
                <div className="rounded-lg border border-primary/20 bg-primary/5 p-3">
                  <p className="text-xs text-foreground leading-relaxed">
                    <span className="font-semibold">Recommended:</span>{" "}
                    Start with "Getting Started" for an overview, then explore
                    each workflow tour as you begin using that feature.
                  </p>
                </div>

                <Separator />

                {/* Tour cards */}
                {tours.map((tour, index) => {
                  const completed = isTourCompleted(tour.id);
                  const icon = TOUR_ICONS[tour.icon] ?? <Rocket className="h-5 w-5" />;

                  return (
                    <motion.div
                      key={tour.id}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.05 }}
                    >
                      <Card
                        className={cn(
                          "border-border/50 bg-card/50 backdrop-blur-sm transition-all cursor-pointer",
                          "hover:border-primary/30 hover:bg-card/80 hover:shadow-md",
                          completed && "border-primary/20 bg-primary/5"
                        )}
                        onClick={() => startTour(tour.id)}
                      >
                        <CardContent className="p-4">
                          <div className="flex items-start gap-3">
                            {/* Icon */}
                            <div
                              className={cn(
                                "flex h-10 w-10 shrink-0 items-center justify-center rounded-lg",
                                completed
                                  ? "bg-primary/15 text-primary"
                                  : "bg-muted text-muted-foreground"
                              )}
                            >
                              {icon}
                            </div>

                            {/* Content */}
                            <div className="flex-1 min-w-0 space-y-1">
                              <div className="flex items-center justify-between gap-2">
                                <h3 className="text-sm font-bold text-foreground">
                                  {tour.name}
                                </h3>
                                {completed ? (
                                  <CheckCircle2 className="h-4 w-4 shrink-0 text-primary" />
                                ) : (
                                  <span className="shrink-0 rounded-full bg-muted px-2 py-0.5 text-[10px] font-medium text-muted-foreground">
                                    {tour.steps.length} steps
                                  </span>
                                )}
                              </div>
                              <p className="text-xs text-muted-foreground leading-relaxed">
                                {tour.description}
                              </p>
                            </div>
                          </div>

                          {/* Action */}
                          <div className="mt-3 flex justify-end">
                            <Button
                              variant={completed ? "outline" : "default"}
                              size="sm"
                              className="h-7 gap-1.5 text-xs font-semibold"
                              onClick={(e) => {
                                e.stopPropagation();
                                startTour(tour.id);
                              }}
                            >
                              <PlayCircle className="h-3.5 w-3.5" />
                              {completed ? "Replay" : "Start Tour"}
                            </Button>
                          </div>
                        </CardContent>
                      </Card>
                    </motion.div>
                  );
                })}

                {/* Footer tip */}
                <div className="rounded-lg bg-muted/50 p-3 mt-2">
                  <p className="text-[11px] text-muted-foreground leading-relaxed">
                    <span className="font-medium">Tip:</span> During any tour, use{" "}
                    <kbd className="rounded bg-muted px-1 py-0.5 font-mono text-[10px]">
                      Arrow keys
                    </kbd>{" "}
                    to navigate and{" "}
                    <kbd className="rounded bg-muted px-1 py-0.5 font-mono text-[10px]">
                      Esc
                    </kbd>{" "}
                    to exit. Tours remember your progress.
                  </p>
                </div>
              </div>
            </ScrollArea>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};
