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
import { staggerContainer, staggerItem, buttonHover, buttonTap, cardHover } from "@/lib/animations";
import { AppView, Species, LandmarkSchema, BoundingBox, LandmarkDefinition } from "@/types/Image";
import type { RootState } from "@/state/store";
import { SettingsModal } from "./SettingsModal";
import { HelpPanel } from "./HelpPanel";
import { SchemaSelector } from "./SchemaSelector";
import { CustomSchemaEditor } from "./CustomSchemaEditor";
import { addSpecies, setActiveSpecies } from "@/state/speciesState/speciesSlice";
import { clearFiles, setSessionImages } from "@/state/filesState/fileSlice";
import { toast } from "sonner";

interface LandingPageProps {
  onNavigate: (view: AppView) => void;
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

export const LandingPage: React.FC<LandingPageProps> = ({
  onNavigate,
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
        const reconstructed: Species = {
          id: speciesId,
          name: result.meta?.name || sessionName,
          landmarkTemplate,
          models: [],
          imageCount: 0,
          annotationCount: 0,
          createdAt: new Date(),
        };
        dispatch(addSpecies(reconstructed));
      } else {
        dispatch(setActiveSpecies(speciesId));
      }

      if (!result.ok || !result.images?.length) {
        onNavigate("workspace");
        return;
      }

      const loadedImages = result.images.map((img) => {
        const bytes = Uint8Array.from(atob(img.data), (c) => c.charCodeAt(0));
        const blob = new Blob([bytes], { type: img.mimeType });
        const url = URL.createObjectURL(blob);

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
  const createNewSession = async (speciesId: string, name: string, landmarkTemplate: LandmarkDefinition[]) => {
    const newSpecies: Species = {
      id: speciesId,
      name,
      landmarkTemplate,
      models: [],
      imageCount: 0,
      annotationCount: 0,
      createdAt: new Date(),
    };

    dispatch(addSpecies(newSpecies));
    dispatch(clearFiles());

    try {
      await window.api.sessionCreate(speciesId, name, landmarkTemplate);
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
      await resumeSession(sessionId, schema.name);
    } else {
      await createNewSession(sessionId, schema.name, schema.landmarks);
    }
  };

  /** Custom schema submitted: resume if a session with this name exists, or create new */
  const handleCustomSchemaSubmit = async (schema: LandmarkSchema) => {
    setCustomSchemaDialogOpen(false);

    const sessionId = customSchemaToSessionId(schema.name);
    const existingSession = sessions.find((s) => s.speciesId === sessionId);

    if (existingSession) {
      await resumeSession(sessionId, schema.name);
    } else {
      await createNewSession(sessionId, schema.name, schema.landmarks);
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
                    onClick={() => resumeSession(session.speciesId, session.name)}
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
