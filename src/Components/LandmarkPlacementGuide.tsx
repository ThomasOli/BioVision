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
      <Card className="border-green-500/50 bg-green-500/10">
        <CardContent className="p-4 text-center">
          <Check className="h-6 w-6 mx-auto mb-2 text-green-500" />
          <p className="text-sm font-semibold text-green-500">
            All landmarks complete!
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            {actualPlaced} placed{skippedCount > 0 && `, ${skippedCount} skipped`}
          </p>
        </CardContent>
      </Card>
    );
  }

  const progress = (placedLandmarks.length / schema.landmarks.length) * 100;

  return (
    <Card className="border-primary/50 bg-primary/5">
      <CardContent className="p-4 space-y-3">
        {/* Current landmark info */}
        <div className="flex items-start gap-3">
          <div className="shrink-0 rounded-full bg-primary/20 px-3 py-1 font-mono text-sm font-bold text-primary">
            {nextLandmark.index + 1}
          </div>
          <div className="flex-1">
            <p className="text-sm font-semibold text-foreground">{nextLandmark.name}</p>
            {nextLandmark.description && (
              <p className="text-xs text-muted-foreground mt-1">{nextLandmark.description}</p>
            )}
            {nextLandmark.category && (
              <span className="inline-block mt-2 rounded-full bg-primary/10 px-2 py-0.5 text-xs text-primary">
                {nextLandmark.category}
              </span>
            )}
          </div>
        </div>

        {/* Skip button */}
        {onSkip && (
          <Button
            variant="outline"
            size="sm"
            onClick={onSkip}
            className="w-full gap-2 text-muted-foreground hover:text-foreground"
          >
            <SkipForward className="h-4 w-4" />
            Skip this landmark
          </Button>
        )}

        {/* Progress */}
        <div>
          <div className="flex items-center justify-between text-xs text-muted-foreground mb-1">
            <span>Progress</span>
            <span>
              {placedLandmarks.length} / {schema.landmarks.length}
              {skippedCount > 0 && ` (${skippedCount} skipped)`}
            </span>
          </div>
          <Progress value={progress} className="h-2" />
        </div>
      </CardContent>
    </Card>
  );
};

export default LandmarkPlacementGuide;
