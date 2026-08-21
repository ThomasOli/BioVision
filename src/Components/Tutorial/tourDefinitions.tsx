import type { Tour } from "./TutorialContext";

/**
 * All guided tours for BioVision.
 *
 * Each step targets a DOM element via `[data-tutorial="<target>"]` or a CSS
 * selector.  Steps with `isFullscreen: true` render a centred card with no
 * spotlight cutout — ideal for intro/outro slides.
 */

export const TOURS: Tour[] = [
  // ── Welcome / Getting Started ───────────────────────────────────────────
  {
    id: "welcome",
    name: "Getting Started",
    description: "Learn the basics of BioVision — navigate the interface and understand the core workflow.",
    icon: "Rocket",
    steps: [
      {
        target: "__fullscreen__",
        title: "Welcome to BioVision",
        description: (
          <div className="space-y-2">
            <p>
              BioVision is a professional tool for training machine learning models
              on biological images. Researchers worldwide use it to annotate
              specimens with landmark points, train shape predictors, and run
              inference on new data.
            </p>
            <p className="text-muted-foreground text-xs">
              This tour will walk you through the main interface. You can replay
              it anytime from the Help menu.
            </p>
          </div>
        ),
        placement: "center",
        isFullscreen: true,
        actionLabel: "Let's get started",
      },
      {
        target: '[data-tutorial="menu-annotate"]',
        title: "Annotate Images",
        description: (
          <p>
            Start here. Create a <strong>schema session</strong> that defines the
            landmark points for your organism, then upload images and annotate
            them with bounding boxes and landmarks.
          </p>
        ),
        placement: "right",
      },
      {
        target: '[data-tutorial="menu-inference"]',
        title: "Run Inference",
        description: (
          <p>
            After training a model, use it to automatically predict landmarks on
            new images. Review and correct predictions, then commit approved
            results back to your training data.
          </p>
        ),
        placement: "right",
      },
      {
        target: '[data-tutorial="menu-models"]',
        title: "My Models",
        description: (
          <p>
            View and manage all your trained models. Each model is scoped to the
            schema session it was trained from.
          </p>
        ),
        placement: "left",
      },
      {
        target: '[data-tutorial="menu-help"]',
        title: "Help & Documentation",
        description: (
          <p>
            Detailed reference documentation, keyboard shortcuts, and tips for
            biologists are always available here.
          </p>
        ),
        placement: "left",
      },
      {
        target: '[data-tutorial="recent-sessions"]',
        title: "Recent Sessions",
        description: (
          <p>
            Your most recent annotation sessions appear here for quick access.
            Click any session card to resume exactly where you left off.
          </p>
        ),
        placement: "top",
      },
      {
        target: "__fullscreen__",
        title: "The BioVision Workflow",
        description: (
          <div className="space-y-3">
            <p className="font-medium">The core loop is simple:</p>
            <ol className="ml-4 list-decimal space-y-1.5 text-sm">
              <li>
                <strong>Create a schema</strong> — define your landmark points and
                orientation policy
              </li>
              <li>
                <strong>Annotate</strong> — upload images, draw bounding boxes,
                and place landmarks
              </li>
              <li>
                <strong>Train</strong> — train an OBB detector and landmark
                predictor (dlib or CNN)
              </li>
              <li>
                <strong>Infer & Review</strong> — predict on new images, correct
                errors, and commit
              </li>
              <li>
                <strong>Retrain</strong> — expand your dataset and improve
                accuracy iteratively
              </li>
            </ol>
          </div>
        ),
        placement: "center",
        isFullscreen: true,
        actionLabel: "Got it!",
      },
    ],
  },

  // ── Annotation Workflow ─────────────────────────────────────────────────
  {
    id: "annotation",
    name: "Annotation Workflow",
    description: "Learn how to annotate images with bounding boxes and landmark points.",
    icon: "Pencil",
    steps: [
      {
        target: "__fullscreen__",
        title: "Annotation Workspace",
        description: (
          <div className="space-y-2">
            <p>
              The workspace is where you spend most of your time. The left panel
              contains tools and settings; the right side is your annotation
              canvas.
            </p>
            <p className="text-xs text-muted-foreground">
              This tour covers the annotation workflow. Open a schema session
              first for the best experience.
            </p>
          </div>
        ),
        placement: "center",
        isFullscreen: true,
        actionLabel: "Show me the tools",
      },
      {
        target: '[data-tutorial="upload-images"]',
        title: "Upload Images",
        description: (
          <p>
            Drag and drop images or click to browse. BioVision copies images
            into the session directory so your originals remain untouched.
            Supports JPG, PNG, TIFF, and BMP formats.
          </p>
        ),
        placement: "right",
      },
      {
        target: '[data-tutorial="detection-mode"]',
        title: "Detection Mode",
        description: (
          <div className="space-y-2">
            <p>
              <strong>Manual:</strong> Draw bounding boxes by clicking and
              dragging directly on the canvas. Rotate and resize to create
              oriented bounding boxes (OBBs).
            </p>
            <p>
              <strong>Auto:</strong> Uses YOLO-World for detection with optional
              SAM2 segmentation. Review and correct auto-detected boxes before
              finalizing.
            </p>
          </div>
        ),
        placement: "right",
      },
      {
        target: '[data-tutorial="landmark-controls"]',
        title: "Landmark Controls",
        description: (
          <p>
            Customize landmark appearance — change colors for better visibility
            on different backgrounds, and adjust opacity. Use the landmark toggle
            to switch between placing and selecting landmarks.
          </p>
        ),
        placement: "right",
      },
      {
        target: '[data-tutorial="canvas-area"]',
        title: "Annotation Canvas",
        description: (
          <div className="space-y-2">
            <p>
              This is your main workspace. Draw bounding boxes around specimens,
              then click inside each box to place landmarks in the order defined
              by your schema.
            </p>
            <p className="text-xs text-muted-foreground">
              Use the magnifier tool for precise landmark placement. Press
              <kbd className="mx-1 rounded bg-muted px-1 py-0.5 font-mono text-xs">Space</kbd>
              + drag to pan when zoomed in.
            </p>
          </div>
        ),
        placement: "left",
      },
      {
        target: '[data-tutorial="image-nav"]',
        title: "Image Navigation",
        description: (
          <p>
            Navigate between images with the arrow buttons or keyboard shortcuts
            (<kbd className="mx-0.5 rounded bg-muted px-1 py-0.5 font-mono text-xs">Left</kbd> /
            <kbd className="mx-0.5 rounded bg-muted px-1 py-0.5 font-mono text-xs">Right</kbd>).
            The counter shows your progress through the dataset.
          </p>
        ),
        placement: "top",
      },
      {
        target: '[data-tutorial="train-button"]',
        title: "Train Model",
        description: (
          <p>
            Once you've annotated and finalized enough images, click here to open
            the training dialog. BioVision supports both dlib and CNN predictors.
          </p>
        ),
        placement: "top",
      },
      {
        target: "__fullscreen__",
        title: "Annotation Tips",
        description: (
          <div className="space-y-3">
            <p className="font-medium">Pro tips for accurate annotations:</p>
            <ul className="ml-4 list-disc space-y-1.5 text-sm">
              <li>
                Be consistent — place landmarks in the same order on every specimen
              </li>
              <li>
                Use the magnifier for fine-grained placement near edges
              </li>
              <li>
                Finalize boxes before placing landmarks to lock the bounding region
              </li>
              <li>
                Start with 15–20 well-annotated images before your first training run
              </li>
              <li>
                Use <kbd className="rounded bg-muted px-1 py-0.5 font-mono text-xs">Ctrl+Z</kbd> to
                undo mistakes immediately
              </li>
            </ul>
          </div>
        ),
        placement: "center",
        isFullscreen: true,
        actionLabel: "Finish tour",
      },
    ],
  },

  // ── Model Training ──────────────────────────────────────────────────────
  {
    id: "training",
    name: "Model Training",
    description: "Understand the training pipeline — from dataset preparation to trained model.",
    icon: "Target",
    steps: [
      {
        target: "__fullscreen__",
        title: "Training Overview",
        description: (
          <div className="space-y-2">
            <p>
              BioVision's training pipeline converts your annotations into a
              machine learning model that can predict landmarks automatically.
            </p>
            <p>
              The pipeline has two stages: an optional <strong>OBB detector</strong>{" "}
              for specimen detection, and a <strong>landmark predictor</strong>{" "}
              (dlib or CNN) for point placement.
            </p>
          </div>
        ),
        placement: "center",
        isFullscreen: true,
        actionLabel: "Learn more",
      },
      {
        target: "__fullscreen__",
        title: "Step 1: OBB Detector (Optional)",
        description: (
          <div className="space-y-2">
            <p>
              If your annotations include oriented bounding boxes, you can train
              a YOLO-based OBB detector first. This detector will automatically
              find and orient specimens during inference.
            </p>
            <p className="text-xs text-muted-foreground">
              The OBB detector normalizes specimen orientation before landmark
              prediction, significantly improving accuracy for directional organisms.
            </p>
          </div>
        ),
        placement: "center",
        isFullscreen: true,
      },
      {
        target: "__fullscreen__",
        title: "Step 2: Landmark Predictor",
        description: (
          <div className="space-y-3">
            <p>Choose the predictor that best fits your data:</p>
            <div className="grid gap-2">
              <div className="rounded-md border border-border/50 bg-muted/30 p-2">
                <p className="text-sm font-medium">dlib Shape Predictor</p>
                <p className="text-xs text-muted-foreground">
                  Fast training, lightweight. Best for standardized imaging
                  conditions with consistent backgrounds.
                </p>
              </div>
              <div className="rounded-md border border-border/50 bg-muted/30 p-2">
                <p className="text-sm font-medium">CNN Predictor</p>
                <p className="text-xs text-muted-foreground">
                  Slower training, GPU recommended. Better for varied imaging
                  conditions, complex backgrounds, and larger datasets.
                </p>
              </div>
            </div>
          </div>
        ),
        placement: "center",
        isFullscreen: true,
      },
      {
        target: "__fullscreen__",
        title: "Augmentation & Orientation",
        description: (
          <div className="space-y-2">
            <p>
              The training dialog includes an <strong>augmentation studio</strong> that
              automatically configures data augmentation based on your orientation policy.
            </p>
            <ul className="ml-4 list-disc space-y-1 text-sm">
              <li>Directional schemas: moderate rotation, horizontal flip</li>
              <li>Bilateral schemas: symmetric augmentation for paired structures</li>
              <li>Invariant schemas: full 360° rotation coverage</li>
            </ul>
            <p className="text-xs text-muted-foreground">
              You can fine-tune augmentation parameters in the advanced section
              of the training dialog.
            </p>
          </div>
        ),
        placement: "center",
        isFullscreen: true,
      },
      {
        target: "__fullscreen__",
        title: "Training Progress",
        description: (
          <div className="space-y-2">
            <p>
              During training, BioVision shows real-time progress including:
            </p>
            <ul className="ml-4 list-disc space-y-1 text-sm">
              <li>Current epoch and total epochs</li>
              <li>Training loss and learning rate</li>
              <li>Estimated time remaining</li>
              <li>Parity evaluation (optional quality check)</li>
            </ul>
            <p className="text-xs text-muted-foreground">
              Models are saved to your project directory and remain scoped to
              the schema session they were trained from.
            </p>
          </div>
        ),
        placement: "center",
        isFullscreen: true,
        actionLabel: "Finish tour",
      },
    ],
  },

  // ── Inference ───────────────────────────────────────────────────────────
  {
    id: "inference",
    name: "Inference & Review",
    description: "Apply trained models to new images and review predictions.",
    icon: "Microscope",
    steps: [
      {
        target: "__fullscreen__",
        title: "Inference Workflow",
        description: (
          <div className="space-y-2">
            <p>
              The inference hub lets you apply trained models to new images.
              Predictions go through a review workflow before being committed
              back to your training dataset.
            </p>
            <p className="text-xs text-muted-foreground">
              This creates a virtuous cycle: train → predict → review → retrain
              with more data.
            </p>
          </div>
        ),
        placement: "center",
        isFullscreen: true,
        actionLabel: "Show me the steps",
      },
      {
        target: "__fullscreen__",
        title: "Step 1: Select a Model",
        description: (
          <p>
            Choose a trained landmark model from the available models for your
            schema. Both dlib and CNN models are supported. The inference session
            is bound to the schema it was trained from.
          </p>
        ),
        placement: "center",
        isFullscreen: true,
      },
      {
        target: "__fullscreen__",
        title: "Step 2: Add & Detect",
        description: (
          <div className="space-y-2">
            <p>
              Add new images to the inference session, then run detection to find
              specimens. If you trained an OBB detector, it will automatically
              orient each specimen before landmark prediction.
            </p>
            <p>
              Correct any mis-detected or misoriented boxes before running
              landmark inference.
            </p>
          </div>
        ),
        placement: "center",
        isFullscreen: true,
      },
      {
        target: "__fullscreen__",
        title: "Step 3: Review & Correct",
        description: (
          <div className="space-y-2">
            <p>
              After landmark prediction, review each image carefully. You can
              drag landmarks to correct positions, and use <strong>Save All
              Changes</strong> to persist your corrections.
            </p>
            <p>
              Mark images as <strong>Review Complete</strong> when satisfied, or
              keep them <strong>In Progress</strong> for later review.
            </p>
          </div>
        ),
        placement: "center",
        isFullscreen: true,
      },
      {
        target: "__fullscreen__",
        title: "Step 4: Commit to Training Data",
        description: (
          <div className="space-y-2">
            <p>
              When you're satisfied with your reviewed predictions, commit them
              back to the schema's training dataset. Only images marked as
              <strong> Review Complete</strong> will be committed.
            </p>
            <p className="font-medium text-sm">
              Then retrain your model with the expanded dataset for even better
              accuracy!
            </p>
          </div>
        ),
        placement: "center",
        isFullscreen: true,
        actionLabel: "Finish tour",
      },
    ],
  },
];
