import React, { useState } from "react";
import { motion } from "framer-motion";
import { useDispatch } from "react-redux";
import {
  Pencil,
  Microscope,
  Database,
  BookOpen,
  Settings,
} from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/Components/ui/button";
import { Card, CardContent } from "@/Components/ui/card";
import { staggerContainer, staggerItem, buttonHover, buttonTap, cardHover } from "@/lib/animations";
import { AppView, Species, LandmarkSchema } from "@/types/Image";
import { SettingsModal } from "./SettingsModal";
import { HelpPanel } from "./HelpPanel";
import { SchemaSelector } from "./SchemaSelector";
import { CustomSchemaEditor } from "./CustomSchemaEditor";
import { addSpecies } from "@/state/speciesState/speciesSlice";

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

export const LandingPage: React.FC<LandingPageProps> = ({
  onNavigate,
}) => {
  const dispatch = useDispatch();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const [schemaDialogOpen, setSchemaDialogOpen] = useState(false);
  const [customSchemaDialogOpen, setCustomSchemaDialogOpen] = useState(false);

  const handleSchemaSelect = (schema: LandmarkSchema | "custom") => {
    if (schema === "custom") {
      setSchemaDialogOpen(false);
      setCustomSchemaDialogOpen(true);
    } else {
      // Create species with selected schema
      const newSpecies: Species = {
        id: `species-${Date.now()}`,
        name: schema.name,
        landmarkTemplate: schema.landmarks,
        models: [],
        imageCount: 0,
        annotationCount: 0,
        createdAt: new Date(),
      };

      dispatch(addSpecies(newSpecies));
      setSchemaDialogOpen(false);
      onNavigate("workspace");
    }
  };

  const handleCustomSchemaSubmit = (schema: LandmarkSchema) => {
    // Create species with custom schema
    const newSpecies: Species = {
      id: `species-${Date.now()}`,
      name: schema.name,
      landmarkTemplate: schema.landmarks,
      models: [],
      imageCount: 0,
      annotationCount: 0,
      createdAt: new Date(),
    };

    dispatch(addSpecies(newSpecies));
    setCustomSchemaDialogOpen(false);
    onNavigate("workspace");
  };

  const menuItems = [
    {
      icon: <Pencil className="h-6 w-6" />,
      title: "Annotate Images",
      description: "Add landmark points to your images",
      onClick: () => setSchemaDialogOpen(true), // Show schema selector first
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

  return (
    <>
      <div className="flex h-screen w-screen flex-col items-center justify-center bg-background p-8">
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
          className="mb-12 text-center"
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

        {/* Menu Grid */}
        <motion.div
          variants={staggerContainer}
          initial="initial"
          animate="animate"
          className="grid max-w-3xl grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3"
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

        {/* Footer */}
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
          className="mt-12 text-xs text-muted-foreground"
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
