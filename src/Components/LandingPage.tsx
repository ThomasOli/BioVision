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
  ReusableSchemaTemplate,
} from "@/types/Image";
import type { RootState } from "@/state/store";
import { SettingsModal } from "./SettingsModal";
import { HelpPanel } from "./HelpPanel";
import { OnboardingGuide } from "./OnboardingGuide";
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

type SchemaKind = "default" | "custom";
type EditableSchemaDraft = {
  id?: string;
  name: string;
  description: string;
  landmarks: LandmarkDefinition[];
  sourcePresetId?: string;
  orientationPolicy?: OrientationPolicy;
};

function normalizeSchemaComponent(value: string | undefined): string {
  return String(value || "").trim().toLowerCase();
}

function normalizeSchemaSlug(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

function computeSchemaFingerprint(landmarkTemplate: LandmarkDefinition[]): string {
  const normalized = (landmarkTemplate || []).map((landmark, position) => ({
    index: Number.isFinite(Number(landmark?.index)) ? Math.max(1, Number(landmark.index)) : position + 1,
    name: normalizeSchemaComponent(landmark?.name),
    category: normalizeSchemaComponent(landmark?.category),
  }));

  let hash = 2166136261;
  const input = JSON.stringify(normalized);
  for (let index = 0; index < input.length; index += 1) {
    hash ^= input.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

function shortSchemaFingerprint(fingerprint: string): string {
  return String(fingerprint || "").slice(0, 8);
}

function schemaToSessionId(schema: LandmarkSchema): string {
  return `schema-${schema.id}`;
}

function customSchemaBaseSessionId(name: string): string {
  const normalized = normalizeSchemaSlug(name);
  return `schema-custom-${normalized || "untitled"}`;
}

function buildForkedSessionId(baseSessionId: string, schemaFingerprint: string): string {
  return `${baseSessionId}-${shortSchemaFingerprint(schemaFingerprint)}`;
}

function inferDefaultOrientationPolicy(
  landmarkTemplate: LandmarkDefinition[]
): OrientationPolicy {
  const byName = (name: string) =>
    landmarkTemplate.find((lm) => String(lm.name || "").trim().toLowerCase() === name)?.index;
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
    const snoutTip = byName("snout tip");
    const upperCaudal = byName("upper caudal peduncle");
    const lowerCaudal = byName("lower caudal peduncle");
    return {
      mode: "directional",
      targetOrientation: "left",
      headCategories: ["head"],
      tailCategories: tailCategories.length > 0 ? tailCategories : ["tail", "caudal-fin"],
      ...(Number.isFinite(Number(snoutTip)) ? { anteriorAnchorIds: [Number(snoutTip)] } : {}),
      ...(Number.isFinite(Number(upperCaudal)) && Number.isFinite(Number(lowerCaudal))
        ? { posteriorAnchorIds: [Number(upperCaudal), Number(lowerCaudal)] }
        : {}),
    };
  }
  return {
    mode: "invariant",
  };
}

type PendingSessionLaunch =
  | {
      type: "create";
      speciesId: string;
      name: string;
      landmarkTemplate: LandmarkDefinition[];
      orientationPolicy?: OrientationPolicy;
      schemaKind: SchemaKind;
      schemaSourceId: string;
      schemaFingerprint: string;
    }
  | {
      type: "resume";
      speciesId: string;
      name: string;
      landmarkTemplate: LandmarkDefinition[];
      orientationPolicy?: OrientationPolicy;
      schemaKind: SchemaKind;
      schemaSourceId: string;
      schemaFingerprint: string;
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
    "Backend goal: the OBB detector levels the crop to the major axis, then class_id enforces a canonical facing direction (head-left). Best for side-view organisms.",
  bilateral:
    "Backend goal: the OBB detector levels the crop along the symmetry axis; class_id encodes canonical up/down orientation for vertically symmetric specimens.",
  axial:
    "Backend goal: the OBB detector levels the long axis; class_id (0=up, 1=down) triggers a 180° rotation to a canonical up-facing orientation.",
  invariant:
    "Backend goal: OBB crops the specimen without orientation enforcement; wide rotational augmentation (±6° dlib, full 360° CNN) exploits complete angular coverage.",
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
    const snoutTip = landmarkTemplate.find((lm) => String(lm.name || "").trim().toLowerCase() === "snout tip")?.index;
    const upperCaudal = landmarkTemplate.find((lm) => String(lm.name || "").trim().toLowerCase() === "upper caudal peduncle")?.index;
    const lowerCaudal = landmarkTemplate.find((lm) => String(lm.name || "").trim().toLowerCase() === "lower caudal peduncle")?.index;
    return {
      mode,
      targetOrientation: "left",
      headCategories: hasHead ? ["head"] : [],
      tailCategories:
        hasTail || hasCaudalTail ? (tailCategories.length > 0 ? tailCategories : ["tail", "caudal-fin"]) : [],
      ...(Number.isFinite(Number(snoutTip)) ? { anteriorAnchorIds: [Number(snoutTip)] } : {}),
      ...(Number.isFinite(Number(upperCaudal)) && Number.isFinite(Number(lowerCaudal))
        ? { posteriorAnchorIds: [Number(upperCaudal), Number(lowerCaudal)] }
        : {}),
    };
  }
  if (mode === "bilateral") {
    const distal = landmarkTemplate.find((lm) => Number(lm.index) === 3)?.index;
    const basal = landmarkTemplate.find((lm) => Number(lm.index) === 12)?.index;
    return {
      mode,
      bilateralClassAxis: "vertical_obb",
      ...(Number.isFinite(Number(distal)) ? { anteriorAnchorIds: [Number(distal)] } : {}),
      ...(Number.isFinite(Number(basal)) ? { posteriorAnchorIds: [Number(basal)] } : {}),
    };
  }
  if (mode === "axial") {
    return { mode };
  }
  return { mode: "invariant" };
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
  const [customSchemaTemplates, setCustomSchemaTemplates] = useState<ReusableSchemaTemplate[]>([]);
  const [schemaEditorMode, setSchemaEditorMode] = useState<"create" | "edit-default" | "edit-custom">("create");
  const [schemaEditorInitial, setSchemaEditorInitial] = useState<EditableSchemaDraft | null>(null);
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [loadingSessions, setLoadingSessions] = useState(true);
  const [resumingSessionId, setResumingSessionId] = useState<string | null>(null);
  const [orientationDialogOpen, setOrientationDialogOpen] = useState(false);
  const [pendingSessionLaunch, setPendingSessionLaunch] = useState<PendingSessionLaunch | null>(null);
  const [selectedOrientationMode, setSelectedOrientationMode] = useState<OrientationPolicy["mode"]>("invariant");
  const [onboardingOpen, setOnboardingOpen] = useState(false);

  // Load existing sessions on mount
  useEffect(() => {
    const loadSessions = async () => {
      try {
        const [sessionResult, templateResult] = await Promise.all([
          window.api.sessionList(),
          window.api.schemaListTemplates(),
        ]);
        if (sessionResult.ok) {
          setSessions(sessionResult.sessions);
        }
        if (templateResult.ok) {
          setCustomSchemaTemplates(templateResult.templates);
        }
      } catch (err) {
        console.error("Failed to load sessions:", err);
      } finally {
        setLoadingSessions(false);
      }
    };
    loadSessions();
  }, []);

  const upsertCustomTemplate = (template: ReusableSchemaTemplate) => {
    setCustomSchemaTemplates((prev) => {
      const next = [template, ...prev.filter((existing) => existing.id !== template.id)];
      next.sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime());
      return next;
    });
  };

  // Auto-open onboarding for first-time users
  useEffect(() => {
    if (
      !loadingSessions &&
      sessions.length === 0 &&
      !localStorage.getItem("biovision-onboarding-dismissed")
    ) {
      setOnboardingOpen(true);
    }
  }, [loadingSessions, sessions.length]);

  // Auto-open schema dialog when navigated here from models page
  useEffect(() => {
    if (openSchemaDialogOnMount) {
      setSchemaDialogOpen(true);
      onSchemaDialogOpened?.();
    }
  }, [openSchemaDialogOnMount, onSchemaDialogOpened]);

  const openOrientationDialogForLaunch = (launch: PendingSessionLaunch) => {
    const suggested = launch.orientationPolicy?.mode || inferDefaultOrientationPolicy(launch.landmarkTemplate).mode;
    setPendingSessionLaunch(launch);
    setSelectedOrientationMode(suggested);
    setOrientationDialogOpen(true);
  };

  const resolveSchemaSession = async (args: {
    schemaKind: SchemaKind;
    schemaSourceId: string;
    schemaFingerprint: string;
    baseSessionId: string;
    preferredSessionId: string;
  }): Promise<{ speciesId: string; exists: boolean }> => {
    const exact = sessions.find(
      (session) =>
        session.schemaKind === args.schemaKind &&
        session.schemaSourceId === args.schemaSourceId &&
        session.schemaFingerprint === args.schemaFingerprint
    );
    if (exact) {
      return { speciesId: exact.speciesId, exists: true };
    }

    const legacyBase = sessions.find((session) => session.speciesId === args.baseSessionId);
    if (legacyBase) {
      try {
        const loaded = await window.api.sessionLoad(args.baseSessionId);
        if (loaded.ok) {
          const loadedTemplate = Array.isArray(loaded.meta?.landmarkTemplate)
            ? loaded.meta.landmarkTemplate
            : [];
          const loadedFingerprint =
            String(loaded.meta?.schemaFingerprint || "") ||
            computeSchemaFingerprint(loadedTemplate);
          if (loadedFingerprint === args.schemaFingerprint) {
            return { speciesId: args.baseSessionId, exists: true };
          }
        }
      } catch {
        // Fall through to fork/create resolution.
      }
    }

    const speciesId =
      args.schemaKind === "default" && !legacyBase
        ? args.baseSessionId
        : args.preferredSessionId;
    return {
      speciesId,
      exists: sessions.some((session) => session.speciesId === speciesId),
    };
  };

  const beginResumeWithOrientationCheck = async (
    speciesId: string,
    sessionName: string,
    fallbackTemplate: LandmarkDefinition[] = [],
    schemaMetadata?: {
      schemaKind: SchemaKind;
      schemaSourceId: string;
      schemaFingerprint: string;
    }
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
      orientationPolicy: existingSession?.orientationPolicy,
      schemaKind: schemaMetadata?.schemaKind || "default",
      schemaSourceId: schemaMetadata?.schemaSourceId || speciesId,
      schemaFingerprint:
        schemaMetadata?.schemaFingerprint || computeSchemaFingerprint(template),
    });
  };

  const confirmOrientationSelection = async () => {
    if (!pendingSessionLaunch) return;
    const launch = pendingSessionLaunch;
    const builtPolicy = buildOrientationPolicy(
      selectedOrientationMode,
      launch.landmarkTemplate
    );
    const policy: OrientationPolicy = {
      ...builtPolicy,
      ...(Array.isArray(launch.orientationPolicy?.anteriorAnchorIds) && launch.orientationPolicy.anteriorAnchorIds.length > 0
        ? { anteriorAnchorIds: launch.orientationPolicy.anteriorAnchorIds }
        : {}),
      ...(Array.isArray(launch.orientationPolicy?.posteriorAnchorIds) && launch.orientationPolicy.posteriorAnchorIds.length > 0
        ? { posteriorAnchorIds: launch.orientationPolicy.posteriorAnchorIds }
        : {}),
      ...(Array.isArray(launch.orientationPolicy?.headCategories) && launch.orientationPolicy.headCategories.length > 0
        ? { headCategories: launch.orientationPolicy.headCategories }
        : {}),
      ...(Array.isArray(launch.orientationPolicy?.tailCategories) && launch.orientationPolicy.tailCategories.length > 0
        ? { tailCategories: launch.orientationPolicy.tailCategories }
        : {}),
      ...(launch.orientationPolicy?.obbLevelingMode ? { obbLevelingMode: launch.orientationPolicy.obbLevelingMode } : {}),
    };
    setOrientationDialogOpen(false);
    setPendingSessionLaunch(null);

    if (launch.type === "create") {
      await createNewSession(
        launch.speciesId,
        launch.name,
        launch.landmarkTemplate,
        policy,
        {
          schemaKind: launch.schemaKind,
          schemaSourceId: launch.schemaSourceId,
          schemaFingerprint: launch.schemaFingerprint,
        }
      );
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
          createdAt: new Date().toISOString(),
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
          isFinalized: img.finalized ?? false,
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
    orientationPolicyOverride?: OrientationPolicy,
    schemaMetadata?: {
      schemaKind: SchemaKind;
      schemaSourceId: string;
      schemaFingerprint: string;
    }
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
      createdAt: new Date().toISOString(),
    };

    dispatch(addSpecies(newSpecies));
    dispatch(clearFiles());

    try {
      await window.api.sessionCreate(
        speciesId,
        name,
        landmarkTemplate,
        orientationPolicy,
        schemaMetadata
      );
    } catch (err) {
      console.error("Failed to create session on disk:", err);
      toast.error("Failed to create session directory.");
    }

    onNavigate("workspace");
  };

  const handleLaunchCustomTemplate = async (schema: ReusableSchemaTemplate) => {
    const schemaFingerprint = computeSchemaFingerprint(schema.landmarks);
    const schemaSourceId = schema.id;
    const baseSessionId = customSchemaBaseSessionId(schema.id);
    const preferredSessionId = buildForkedSessionId(baseSessionId, schemaFingerprint);
    const resolved = await resolveSchemaSession({
      schemaKind: "custom",
      schemaSourceId,
      schemaFingerprint,
      baseSessionId,
      preferredSessionId,
    });

    if (resolved.exists) {
      await beginResumeWithOrientationCheck(resolved.speciesId, schema.name, schema.landmarks, {
        schemaKind: "custom",
        schemaSourceId,
        schemaFingerprint,
      });
    } else {
      openOrientationDialogForLaunch({
        type: "create",
        speciesId: resolved.speciesId,
        name: schema.name,
        landmarkTemplate: schema.landmarks,
        orientationPolicy: schema.orientationPolicy,
        schemaKind: "custom",
        schemaSourceId,
        schemaFingerprint,
      });
    }
  };

  /** Schema selected: resume the existing session for this schema, or create a new one */
  const handleSchemaSelect = async (schema: LandmarkSchema | ReusableSchemaTemplate | "custom") => {
    if (schema === "custom") {
      setSchemaDialogOpen(false);
      setSchemaEditorMode("create");
      setSchemaEditorInitial(null);
      setCustomSchemaDialogOpen(true);
      return;
    }

    if ("kind" in schema && schema.kind === "custom") {
      setSchemaDialogOpen(false);
      await handleLaunchCustomTemplate(schema);
      return;
    }

    setSchemaDialogOpen(false);

    const schemaFingerprint = computeSchemaFingerprint(schema.landmarks);
    const baseSessionId = schemaToSessionId(schema);
    const preferredSessionId = buildForkedSessionId(baseSessionId, schemaFingerprint);
    const resolved = await resolveSchemaSession({
      schemaKind: "default",
      schemaSourceId: schema.id,
      schemaFingerprint,
      baseSessionId,
      preferredSessionId,
    });

    if (resolved.exists) {
      await beginResumeWithOrientationCheck(resolved.speciesId, schema.name, schema.landmarks, {
        schemaKind: "default",
        schemaSourceId: schema.id,
        schemaFingerprint,
      });
    } else {
      openOrientationDialogForLaunch({
        type: "create",
        speciesId: resolved.speciesId,
        name: schema.name,
        landmarkTemplate: schema.landmarks,
        schemaKind: "default",
        schemaSourceId: schema.id,
        schemaFingerprint,
      });
    }
  };

  /** Custom schema submitted: save/update reusable template, then resume or create a session */
  const handleCustomSchemaSubmit = async (schema: EditableSchemaDraft) => {
    setCustomSchemaDialogOpen(false);
    try {
      const result =
        schemaEditorMode === "edit-custom" && schema.id
          ? await window.api.schemaUpdateCustomTemplate(schema.id, {
              name: schema.name,
              description: schema.description,
              landmarks: schema.landmarks,
              orientationPolicy: schema.orientationPolicy,
              sourcePresetId: schema.sourcePresetId,
            })
          : await window.api.schemaSaveCustomTemplate({
              name: schema.name,
              description: schema.description,
              landmarks: schema.landmarks,
              orientationPolicy: schema.orientationPolicy,
              sourcePresetId: schema.sourcePresetId,
            });

      if (!result.ok || !result.template) {
        toast.error(result.error || "Failed to save custom schema.");
        return;
      }

      upsertCustomTemplate(result.template);
      await handleLaunchCustomTemplate(result.template);
    } catch (error) {
      console.error("Failed to save custom schema:", error);
      toast.error("Failed to save custom schema.");
    }
  };

  const handleEditDefaultSchema = (schema: LandmarkSchema) => {
    setSchemaDialogOpen(false);
    setSchemaEditorMode("edit-default");
    setSchemaEditorInitial({
      name: schema.name,
      description: schema.description,
      landmarks: schema.landmarks,
      sourcePresetId: schema.id,
      orientationPolicy: inferDefaultOrientationPolicy(schema.landmarks),
    });
    setCustomSchemaDialogOpen(true);
  };

  const handleEditCustomSchema = (schema: ReusableSchemaTemplate) => {
    setSchemaDialogOpen(false);
    setSchemaEditorMode("edit-custom");
    setSchemaEditorInitial({
      id: schema.id,
      name: schema.name,
      description: schema.description,
      landmarks: schema.landmarks,
      sourcePresetId: schema.sourcePresetId,
      orientationPolicy: schema.orientationPolicy,
    });
    setCustomSchemaDialogOpen(true);
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
      <div className="flex h-screen w-screen flex-col items-center bg-background p-8 overflow-y-auto scrollbar-app">
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
              The OBB detector levels each specimen crop to a canonical orientation before
              landmark prediction. For directional schemas, class_id tracks left/right; for
              bilateral it tracks up/down; axial and invariant detector export stay one-class.
              Invariant schemas skip orientation enforcement and rely on broad augmentation instead.
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
      <HelpPanel
        open={helpOpen}
        onOpenChange={setHelpOpen}
        onShowOnboarding={() => {
          setHelpOpen(false);
          setOnboardingOpen(true);
        }}
      />
      <OnboardingGuide
        open={onboardingOpen}
        onOpenChange={setOnboardingOpen}
      />
      <SchemaSelector
        open={schemaDialogOpen}
        customSchemas={customSchemaTemplates}
        onSelect={handleSchemaSelect}
        onEditDefault={handleEditDefaultSchema}
        onEditCustom={handleEditCustomSchema}
        onCancel={() => setSchemaDialogOpen(false)}
      />
      <CustomSchemaEditor
        open={customSchemaDialogOpen}
        mode={schemaEditorMode}
        initialSchema={schemaEditorInitial}
        onSave={handleCustomSchemaSubmit}
        onCancel={() => setCustomSchemaDialogOpen(false)}
      />
    </>
  );
};

export default LandingPage;
