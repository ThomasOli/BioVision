import React from "react";
import { Check, SkipForward } from "lucide-react";

import { LandmarkSchema, Point } from "@/types/Image";
import { Card, CardContent } from "@/Components/ui/card";
import { Button } from "@/Components/ui/button";
import { Progress } from "@/Components/ui/progress";

interface LandmarkPlacementGuideProps {
  schema: LandmarkSchema;
  placedLandmarks: Point[];
  onSkip?: () => void; // Callback to skip current landmark
}

export const LandmarkPlacementGuide: React.FC<LandmarkPlacementGuideProps> = ({
  schema,
  placedLandmarks,
  onSkip,
}) => {
  // Find next landmark that hasn't been placed yet
  const nextLandmark = schema.landmarks.find(
    (lm) => !placedLandmarks.some((placed) => placed.id === lm.index)
  );

  // Count actual placed vs skipped landmarks
  const actualPlaced = placedLandmarks.filter((lm) => !lm.isSkipped).length;
  const skippedCount = placedLandmarks.filter((lm) => lm.isSkipped).length;

  // Show completion message if all landmarks are placed or skipped
  if (!nextLandmark) {
    return (
      <div className="rounded-md border border-primary/30 bg-primary/8 px-3 py-2.5 flex items-center gap-2.5">
        <div className="shrink-0 rounded-full bg-primary/15 p-1">
          <Check className="h-3.5 w-3.5 text-primary" />
        </div>
        <div>
          <p className="text-xs font-semibold text-primary">All landmarks placed</p>
          <p className="text-[10px] text-muted-foreground mt-0.5">
            {actualPlaced} placed{skippedCount > 0 && `, ${skippedCount} skipped`}
          </p>
        </div>
      </div>
    );
  }

  const progress = (placedLandmarks.length / schema.landmarks.length) * 100;

  return (
    <div className="rounded-md border border-border/50 bg-muted/30 p-3 space-y-2.5">
      {/* Current landmark info */}
      <div className="flex items-start gap-2.5">
        <div className="shrink-0 rounded-md bg-primary/15 px-2 py-0.5 font-mono text-xs font-bold text-primary min-w-[28px] text-center">
          {schema.landmarks.indexOf(nextLandmark) + 1}
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-xs font-semibold text-foreground leading-tight">{nextLandmark.name}</p>
          {nextLandmark.description && (
            <p className="text-[10px] text-muted-foreground mt-0.5 leading-relaxed">{nextLandmark.description}</p>
          )}
          {nextLandmark.category && (
            <span className="inline-block mt-1.5 rounded bg-primary/10 px-1.5 py-0.5 text-[10px] font-medium text-primary">
              {nextLandmark.category}
            </span>
          )}
        </div>
      </div>

      {/* Progress */}
      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <span className="font-mono text-[10px] text-muted-foreground/70">
            {placedLandmarks.length} / {schema.landmarks.length}
            {skippedCount > 0 && ` · ${skippedCount} skipped`}
          </span>
          <span className="font-mono text-[10px] text-primary">
            {Math.round(progress)}%
          </span>
        </div>
        <Progress value={progress} className="h-1" />
      </div>

      {/* Skip button */}
      {onSkip && (
        <Button
          variant="ghost"
          size="sm"
          onClick={onSkip}
          className="w-full h-7 gap-1.5 text-[11px] text-muted-foreground hover:text-foreground"
        >
          <SkipForward className="h-3 w-3" />
          Skip
        </Button>
      )}
    </div>
  );
};

export default LandmarkPlacementGuide;
