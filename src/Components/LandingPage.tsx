import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { useDispatch, useSelector } from "react-redux";
import {
  Pencil,
  Microscope,
  Database,
  BookOpen,
  Settings,
  Clock,
  ImageIcon,
} from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/Components/ui/button";
import { Card, CardContent } from "@/Components/ui/card";
import { Separator } from "@/Components/ui/separator";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/Components/ui/dialog";
import { staggerContainer, staggerItem, buttonHover, buttonTap, cardHover } from "@/lib/animations";
import {
  AppView,
  Species,
  LandmarkSchema,
  BoundingBox,
  LandmarkDefinition,
  OrientationPolicy,
} from "@/types/Image";
import type { RootState } from "@/state/store";
import { SettingsModal } from "./SettingsModal";
import { HelpPanel } from "./HelpPanel";
import { SchemaSelector } from "./SchemaSelector";
import { CustomSchemaEditor } from "./CustomSchemaEditor";
import { addSpecies, setActiveSpecies, updateSpecies } from "@/state/speciesState/speciesSlice";
import { clearFiles, setSessionImages } from "@/state/filesState/fileSlice";
import { toast } from "sonner";

interface LandingPageProps {
  onNavigate: (view: AppView) => void;
  openSchemaDialogOnMount?: boolean;
  onSchemaDialogOpened?: () => void;
}

interface MenuButtonProps {
  icon: React.ReactNode;
  title: string;
  description: string;
  onClick: () => void;
}

const MenuButton: React.FC<MenuButtonProps> = ({ icon, title, description, onClick }) => {
  return (
    <motion.div variants={staggerItem}>
      <motion.div
        variants={cardHover}
        initial="initial"
        whileHover="hover"
        className="h-full"
      >
        <motion.div {...buttonHover} {...buttonTap} className="h-full">
          <Card
            className={cn(
              "h-full cursor-pointer border-border/50 bg-card/50 backdrop-blur-sm",
              "transition-colors hover:border-primary/30 hover:bg-card/80"
            )}
            onClick={onClick}
          >
            <CardContent className="flex flex-col items-center justify-center p-6 text-center">
              <div className="mb-3 rounded-xl bg-primary/10 p-3 text-primary">
                {icon}
              </div>
              <h3 className="text-sm font-bold text-foreground">{title}</h3>
              <p className="mt-1 text-xs text-muted-foreground">{description}</p>
            </CardContent>
          </Card>
        </motion.div>
      </motion.div>
    </motion.div>
  );
};

function formatDate(dateStr: string): string {
  if (!dateStr) return "";
  try {
    const date = new Date(dateStr);
    return date.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });
  } catch {
    return dateStr;
  }
}

/** Deterministic session ID for a default schema so the same schema always maps to the same session */
function schemaToSessionId(schema: LandmarkSchema): string {
  return `schema-${schema.id}`;
}

/** Deterministic session ID for a custom schema based on its name */
function customSchemaToSessionId(name: string): string {
  const normalized = name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  return `schema-custom-${normalized || Date.now()}`;
}

function inferDefaultOrientationPolicy(
  landmarkTemplate: LandmarkDefinition[]
): OrientationPolicy {
  const categories = new Set(
    (landmarkTemplate || [])
      .map((lm) => (lm.category || "").trim().toLowerCase())
      .filter(Boolean)
  );
  const hasHead = categories.has("head");
  const hasTail = categories.has("tail");
  const hasCaudalTail = categories.has("caudal-fin");
  const tailCategories = ["tail", "caudal-fin"].filter((cat) => categories.has(cat));
  if (hasHead || hasTail || hasCaudalTail) {
    return {
      mode: "directional",
      targetOrientation: "left",
      headCategories: ["head"],
      tailCategories: tailCategories.length > 0 ? tailCategories : ["tail", "caudal-fin"],
      pcaLevelingMode: "auto",
    };
  }
  return {
    mode: "invariant",
    pcaLevelingMode: "off",
  };
}

type PendingSessionLaunch =
  | {
      type: "create";
      speciesId: string;
      name: string;
      landmarkTemplate: LandmarkDefinition[];
    }
  | {
      type: "resume";
      speciesId: string;
      name: string;
      landmarkTemplate: LandmarkDefinition[];
    };

const ORIENTATION_MODE_LABELS: Record<OrientationPolicy["mode"], { title: string; description: string }> = {
  directional: {
    title: "Directional (Head/Tail)",
    description: "Use for strict head/tail objects (for example side-view fish or isolated single wings).",
  },
  bilateral: {
    title: "Mirrored / Bilateral",
    description: "Use when left/right pairs exist on the same specimen (for example full fly with both wings).",
  },
  axial: {
    title: "Axial",
    description: "Use for elongated specimens that rotate around a dominant axis (for example worms/diatoms).",
  },
  invariant: {
    title: "Invariant",
    description: "Use for radial/non-directional objects where no stable head/tail axis exists.",
  },
};

const ORIENTATION_MODE_DETAILS: Record<OrientationPolicy["mode"], string> = {
  directional:
    "Backend goal: align the major axis, then force a canonical facing direction (for example head-left). Best for side-view organisms.",
  bilateral:
    "Backend goal: align symmetry axis and allow mirrored augmentation for paired structures (left/right sides).",
  axial:
    "Backend goal: align long axis while keeping stronger rotational augmentation for elongated specimens with variable pose.",
  invariant:
    "Backend goal: avoid forced PCA direction when no stable axis exists; rely on broad rotational augmentation.",
};

function buildOrientationPolicy(
  mode: OrientationPolicy["mode"],
  landmarkTemplate: LandmarkDefinition[]
): OrientationPolicy {
  const categories = new Set(
    (landmarkTemplate || [])
      .map((lm) => (lm.category || "").trim().toLowerCase())
      .filter(Boolean)
  );
  const hasHead = categories.has("head");
  const hasTail = categories.has("tail");
  const hasCaudalTail = categories.has("caudal-fin");
  const tailCategories = ["tail", "caudal-fin"].filter((cat) => categories.has(cat));

  if (mode === "directional") {
    return {
      mode,
      targetOrientation: "left",
      headCategories: hasHead ? ["head"] : [],
      tailCategories:
        hasTail || hasCaudalTail ? (tailCategories.length > 0 ? tailCategories : ["tail", "caudal-fin"]) : [],
      pcaLevelingMode: "auto",
    };
  }
  if (mode === "bilateral") {
    return { mode, pcaLevelingMode: "auto" };
  }
  if (mode === "axial") {
    return { mode, pcaLevelingMode: "auto" };
  }
  return { mode: "invariant", pcaLevelingMode: "off" };
}

export const LandingPage: React.FC<LandingPageProps> = ({
  onNavigate,
  openSchemaDialogOnMount,
  onSchemaDialogOpened,
}) => {
  const dispatch = useDispatch();
  const speciesList = useSelector((state: RootState) => state.species.species);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const [schemaDialogOpen, setSchemaDialogOpen] = useState(false);
  const [customSchemaDialogOpen, setCustomSchemaDialogOpen] = useState(false);
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [loadingSessions, setLoadingSessions] = useState(true);
  const [resumingSessionId, setResumingSessionId] = useState<string | null>(null);
  const [orientationDialogOpen, setOrientationDialogOpen] = useState(false);
  const [pendingSessionLaunch, setPendingSessionLaunch] = useState<PendingSessionLaunch | null>(null);
  const [selectedOrientationMode, setSelectedOrientationMode] = useState<OrientationPolicy["mode"]>("invariant");

  // Load existing sessions on mount
  useEffect(() => {
    const loadSessions = async () => {
      try {
        const result = await window.api.sessionList();
        if (result.ok) {
          setSessions(result.sessions);
        }
      } catch (err) {
        console.error("Failed to load sessions:", err);
      } finally {
        setLoadingSessions(false);
      }
    };
    loadSessions();
  }, []);

  // Auto-open schema dialog when navigated here from models page
  useEffect(() => {
    if (openSchemaDialogOnMount) {
      setSchemaDialogOpen(true);
      onSchemaDialogOpened?.();
    }
  }, [openSchemaDialogOnMount]);

  const openOrientationDialogForLaunch = (launch: PendingSessionLaunch) => {
    const suggested = inferDefaultOrientationPolicy(launch.landmarkTemplate).mode;
    setPendingSessionLaunch(launch);
    setSelectedOrientationMode(suggested);
    setOrientationDialogOpen(true);
  };

  const beginResumeWithOrientationCheck = async (
    speciesId: string,
    sessionName: string,
    fallbackTemplate: LandmarkDefinition[] = []
  ) => {
    const existingSession = sessions.find((s) => s.speciesId === speciesId);
    const hasConfiguredFromList = Boolean(
      existingSession?.orientationPolicyConfigured &&
      existingSession?.orientationPolicy?.mode
    );
    if (hasConfiguredFromList) {
      await resumeSession(speciesId, sessionName);
      return;
    }

    let template = fallbackTemplate;
    let hasConfiguredFromLoad = false;
    try {
      const loaded = await window.api.sessionLoad(speciesId);
      if (loaded.ok && loaded.meta?.landmarkTemplate?.length) {
        template = loaded.meta.landmarkTemplate;
      }
      hasConfiguredFromLoad = Boolean(
        loaded.ok &&
        loaded.meta?.orientationPolicyConfigured &&
        loaded.meta?.orientationPolicy?.mode
      );
    } catch {
      // Fallback template is good enough for policy prompt.
    }

    if (hasConfiguredFromLoad) {
      await resumeSession(speciesId, sessionName);
      return;
    }

    openOrientationDialogForLaunch({
      type: "resume",
      speciesId,
      name: sessionName,
      landmarkTemplate: template,
    });
  };

  const confirmOrientationSelection = async () => {
    if (!pendingSessionLaunch) return;
    const launch = pendingSessionLaunch;
    const policy = buildOrientationPolicy(
      selectedOrientationMode,
      launch.landmarkTemplate
    );
    setOrientationDialogOpen(false);
    setPendingSessionLaunch(null);

    if (launch.type === "create") {
      await createNewSession(launch.speciesId, launch.name, launch.landmarkTemplate, policy);
      return;
    }

    try {
      await window.api.sessionUpdateOrientationPolicy(launch.speciesId, policy);
    } catch (err) {
      console.warn("Failed to persist selected orientation policy:", err);
    }
    await resumeSession(launch.speciesId, launch.name);
  };

  /** Resume an existing session by loading its images from disk */
  const resumeSession = async (speciesId: string, sessionName: string) => {
    setResumingSessionId(speciesId);
    try {
      dispatch(clearFiles());

      const result = await window.api.sessionLoad(speciesId);

      // Ensure the Species object exists in Redux (may be missing if persist was cleared)
      const existingSpecies = speciesList.find((s) => s.id === speciesId);
      if (!existingSpecies) {
        const landmarkTemplate: LandmarkDefinition[] = result.meta?.landmarkTemplate || [];
        const orientationPolicy =
          result.meta?.orientationPolicy ||
          inferDefaultOrientationPolicy(landmarkTemplate);
        const reconstructed: Species = {
          id: speciesId,
          name: result.meta?.name || sessionName,
          landmarkTemplate,
          orientationPolicy,
          models: [],
          imageCount: 0,
          annotationCount: 0,
          createdAt: new Date(),
        };
        dispatch(addSpecies(reconstructed));
      } else {
        dispatch(setActiveSpecies(speciesId));
        // Always sync the landmark template from disk — session.json may have been
        // updated (e.g. index renumbering) since the species was last persisted in Redux.
        const syncedTemplate =
          result.meta?.landmarkTemplate?.length
            ? result.meta.landmarkTemplate
            : existingSpecies.landmarkTemplate;
        const syncedPolicy =
          result.meta?.orientationPolicy ||
          existingSpecies.orientationPolicy ||
          inferDefaultOrientationPolicy(syncedTemplate || []);
        dispatch(
          updateSpecies({
            id: speciesId,
            updates: {
              landmarkTemplate: syncedTemplate,
              orientationPolicy: syncedPolicy,
            },
          })
        );
      }

      if (!result.ok || !result.images?.length) {
        onNavigate("workspace");
        return;
      }

      const loadedImages = result.images.map((img) => {
        const safePath = img.diskPath.replace(/\\/g, "/");
        const url = `localfile:///${safePath.replace(/^\//, "")}`;

        return {
          id: Date.now() + Math.random(),
          path: img.diskPath,
          diskPath: img.diskPath,
          url,
          filename: img.filename,
          boxes: img.boxes as BoundingBox[],
          selectedBoxId: null,
          history: [] as BoundingBox[][],
          future: [] as BoundingBox[][],
          speciesId,
          hasBoxes: img.hasBoxes ?? false,
        };
      });

      dispatch(setSessionImages(loadedImages));
      onNavigate("workspace");
    } catch (err) {
      console.error("Failed to resume session:", err);
      toast.error("Failed to load session.");
    } finally {
      setResumingSessionId(null);
    }
  };

  /** Create a brand-new session on disk + Redux, then navigate */
  const createNewSession = async (
    speciesId: string,
    name: string,
    landmarkTemplate: LandmarkDefinition[],
    orientationPolicyOverride?: OrientationPolicy
  ) => {
    const orientationPolicy =
      orientationPolicyOverride || inferDefaultOrientationPolicy(landmarkTemplate);
    const newSpecies: Species = {
      id: speciesId,
      name,
      landmarkTemplate,
      orientationPolicy,
      models: [],
      imageCount: 0,
      annotationCount: 0,
      createdAt: new Date(),
    };

    dispatch(addSpecies(newSpecies));
    dispatch(clearFiles());

    try {
      await window.api.sessionCreate(
        speciesId,
        name,
        landmarkTemplate,
        orientationPolicy
      );
    } catch (err) {
      console.error("Failed to create session on disk:", err);
      toast.error("Failed to create session directory.");
    }

    onNavigate("workspace");
  };

  /** Schema selected: resume the existing session for this schema, or create a new one */
  const handleSchemaSelect = async (schema: LandmarkSchema | "custom") => {
    if (schema === "custom") {
      setSchemaDialogOpen(false);
      setCustomSchemaDialogOpen(true);
      return;
    }

    setSchemaDialogOpen(false);

    const sessionId = schemaToSessionId(schema);
    const existingSession = sessions.find((s) => s.speciesId === sessionId);

    if (existingSession) {
      await beginResumeWithOrientationCheck(sessionId, schema.name, schema.landmarks);
    } else {
      openOrientationDialogForLaunch({
        type: "create",
        speciesId: sessionId,
        name: schema.name,
        landmarkTemplate: schema.landmarks,
      });
    }
  };

  /** Custom schema submitted: resume if a session with this name exists, or create new */
  const handleCustomSchemaSubmit = async (schema: LandmarkSchema) => {
    setCustomSchemaDialogOpen(false);

    const sessionId = customSchemaToSessionId(schema.name);
    const existingSession = sessions.find((s) => s.speciesId === sessionId);

    if (existingSession) {
      await beginResumeWithOrientationCheck(sessionId, schema.name, schema.landmarks);
    } else {
      openOrientationDialogForLaunch({
        type: "create",
        speciesId: sessionId,
        name: schema.name,
        landmarkTemplate: schema.landmarks,
      });
    }
  };

  const menuItems = [
    {
      icon: <Pencil className="h-6 w-6" />,
      title: "Annotate Images",
      description: "Add landmark points to your images",
      onClick: () => setSchemaDialogOpen(true),
    },
    {
      icon: <Microscope className="h-6 w-6" />,
      title: "Run Inference",
      description: "Apply trained models to new images",
      onClick: () => onNavigate("inference"),
    },
    {
      icon: <Database className="h-6 w-6" />,
      title: "My Models",
      description: "Manage your trained models",
      onClick: () => onNavigate("models"),
    },
    {
      icon: <BookOpen className="h-6 w-6" />,
      title: "Help & Docs",
      description: "Learn how to use BioVision",
      onClick: () => setHelpOpen(true),
    },
  ];

  // Show at most 4 recently-used sessions, sorted by lastModified (already sorted from backend)
  const recentSessions = sessions.slice(0, 4);

  return (
    <>
      <div className="flex h-screen w-screen flex-col items-center bg-background p-8 overflow-y-auto">
        {/* Settings button */}
        <div className="absolute right-4 top-4">
          <motion.div {...buttonHover} {...buttonTap}>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setSettingsOpen(true)}
              className="text-muted-foreground hover:text-foreground"
            >
              <Settings className="h-5 w-5" />
            </Button>
          </motion.div>
        </div>

        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="mb-12 text-center mt-8"
        >
          <div className="mb-4 flex items-center justify-center gap-3">
            <div className="rounded-xl bg-primary/10 p-3">
              <Microscope className="h-10 w-10 text-primary" />
            </div>
          </div>
          <h1 className="text-3xl font-bold text-foreground">BioVision</h1>
          <p className="mt-2 text-sm text-muted-foreground">
            Train ML models for biological image landmarking
          </p>
        </motion.div>

        {/* Menu Grid — 2x2 */}
        <motion.div
          variants={staggerContainer}
          initial="initial"
          animate="animate"
          className="grid w-full max-w-xl grid-cols-2 gap-4"
        >
          {menuItems.map((item, index) => (
            <MenuButton
              key={index}
              icon={item.icon}
              title={item.title}
              description={item.description}
              onClick={item.onClick}
            />
          ))}
        </motion.div>

        {/* Recently Used Sessions — up to 4 in a row */}
        {!loadingSessions && recentSessions.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.4 }}
            className="mt-10 w-full max-w-xl"
          >
            <Separator className="mb-6" />
            <div className="mb-4 flex items-center gap-2">
              <Clock className="h-4 w-4 text-muted-foreground" />
              <h2 className="text-sm font-bold uppercase tracking-wide text-muted-foreground">
                Recently Used
              </h2>
            </div>
            <div className="flex gap-3">
              {recentSessions.map((session) => (
                <motion.div
                  key={session.speciesId}
                  variants={cardHover}
                  initial="initial"
                  whileHover="hover"
                  className="flex-1 min-w-0"
                >
                  <Card
                    className={cn(
                      "cursor-pointer border-border/50 bg-card/50 backdrop-blur-sm h-full",
                      "transition-colors hover:border-primary/30 hover:bg-card/80",
                      resumingSessionId === session.speciesId && "opacity-60 pointer-events-none"
                    )}
                    onClick={() =>
                      beginResumeWithOrientationCheck(session.speciesId, session.name)
                    }
                  >
                    <CardContent className="p-3">
                      <h3 className="text-xs font-bold text-foreground truncate">
                        {session.name}
                      </h3>
                      <div className="mt-1.5 flex items-center gap-1.5 text-[11px] text-muted-foreground">
                        <ImageIcon className="h-3 w-3 shrink-0" />
                        <span>{session.imageCount} img{session.imageCount !== 1 ? "s" : ""}</span>
                      </div>
                      {session.lastModified && (
                        <p className="mt-0.5 text-[10px] text-muted-foreground/60 truncate">
                          {formatDate(session.lastModified)}
                        </p>
                      )}
                      {resumingSessionId === session.speciesId && (
                        <p className="mt-1 text-[10px] font-semibold text-primary">Loading...</p>
                      )}
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}

        {/* Footer */}
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
          className="mt-12 mb-8 text-xs text-muted-foreground"
        >
          Press <kbd className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs">?</kbd> for keyboard shortcuts
        </motion.p>
      </div>

      <Dialog
        open={orientationDialogOpen}
        onOpenChange={(open) => {
          if (!open) {
            setOrientationDialogOpen(false);
            setPendingSessionLaunch(null);
          }
        }}
      >
        <DialogContent className="max-w-xl">
          <DialogHeader>
            <DialogTitle>Choose Orientation Schema</DialogTitle>
            <DialogDescription>
              Select how this session should handle orientation normalization during training and inference.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-2">
            {(Object.keys(ORIENTATION_MODE_LABELS) as OrientationPolicy["mode"][]).map((mode) => (
              <button
                key={mode}
                type="button"
                onClick={() => setSelectedOrientationMode(mode)}
                className={cn(
                  "rounded-md border px-3 py-2 text-left transition-colors",
                  selectedOrientationMode === mode
                    ? "border-primary bg-primary/10"
                    : "border-border hover:border-primary/40"
                )}
              >
                <p className="text-sm font-semibold">{ORIENTATION_MODE_LABELS[mode].title}</p>
                <p className="text-xs text-muted-foreground">
                  {ORIENTATION_MODE_LABELS[mode].description}
                </p>
              </button>
            ))}
          </div>
          <div className="rounded-md border border-primary/30 bg-primary/5 px-3 py-2 text-xs text-foreground">
            {ORIENTATION_MODE_DETAILS[selectedOrientationMode]}
          </div>
          <div className="rounded-md border border-border/70 bg-muted/30 px-3 py-2 text-xs text-muted-foreground">
            <p>
              SAM2 + PCA (when available) will level specimens before landmark prediction.
              Directional schemas then enforce a canonical facing direction; bilateral and axial
              schemas keep symmetry-aware augmentation; invariant schemas avoid unstable forced leveling.
            </p>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setOrientationDialogOpen(false);
                setPendingSessionLaunch(null);
              }}
            >
              Cancel
            </Button>
            <Button onClick={confirmOrientationSelection}>
              Continue
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <SettingsModal open={settingsOpen} onOpenChange={setSettingsOpen} />
      <HelpPanel open={helpOpen} onOpenChange={setHelpOpen} />
      <SchemaSelector
        open={schemaDialogOpen}
        onSelect={handleSchemaSelect}
        onCancel={() => setSchemaDialogOpen(false)}
      />
      <CustomSchemaEditor
        open={customSchemaDialogOpen}
        onSave={handleCustomSchemaSubmit}
        onCancel={() => setCustomSchemaDialogOpen(false)}
      />
    </>
  );
};

export default LandingPage;
