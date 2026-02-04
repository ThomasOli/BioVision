import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { FolderOpen, Monitor, Moon, Sun, Copy, ChevronDown } from "lucide-react";
import { toast } from "sonner";

import { cn } from "@/lib/utils";
import { Button } from "@/Components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/Components/ui/card";
import { Label } from "@/Components/ui/label";
import { Separator } from "@/Components/ui/separator";
import { Slider } from "@/Components/ui/slider";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/Components/ui/dialog";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/Components/ui/tooltip";
import { buttonHover, buttonTap } from "@/lib/animations";
import { useTheme } from "./theme-provider";

interface SettingsModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

type Theme = "dark" | "light" | "system";

const LANDMARK_COLORS = [
  { name: "Red", value: "red" },
  { name: "Green", value: "green" },
  { name: "Blue", value: "blue" },
  { name: "Yellow", value: "yellow" },
  { name: "Cyan", value: "cyan" },
  { name: "Magenta", value: "magenta" },
  { name: "Orange", value: "orange" },
  { name: "White", value: "white" },
];

interface SettingsSectionProps {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}

const SettingsSection: React.FC<SettingsSectionProps> = ({
  title,
  children,
  defaultOpen = true,
}) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
      <CardHeader
        className="cursor-pointer pb-3"
        onClick={() => setIsOpen(!isOpen)}
      >
        <div className="flex items-center justify-between">
          <CardTitle className="text-xs font-bold uppercase tracking-wide text-muted-foreground">
            {title}
          </CardTitle>
          <ChevronDown
            className={cn(
              "h-4 w-4 text-muted-foreground transition-transform",
              isOpen && "rotate-180"
            )}
          />
        </div>
      </CardHeader>
      {isOpen && <CardContent className="space-y-4">{children}</CardContent>}
    </Card>
  );
};

export const SettingsModal: React.FC<SettingsModalProps> = ({
  open,
  onOpenChange,
}) => {
  const { theme, setTheme } = useTheme();
  const [projectPath, setProjectPath] = useState("");
  const [defaultColor, setDefaultColor] = useState(() =>
    localStorage.getItem("biovision-default-color") || "red"
  );
  const [defaultOpacity, setDefaultOpacity] = useState(() => {
    const saved = localStorage.getItem("biovision-default-opacity");
    return saved ? parseInt(saved, 10) : 100;
  });

  useEffect(() => {
    const fetchProjectRoot = async () => {
      try {
        const result = await window.api.getProjectRoot();
        if (result?.projectRoot) setProjectPath(result.projectRoot);
      } catch (err) {
        console.error("Failed to load project path", err);
      }
    };
    if (open) fetchProjectRoot();
  }, [open]);

  const handleSelectProjectPath = async () => {
    try {
      const result = await window.api.selectProjectRoot();
      if (!result?.canceled && result?.projectRoot) {
        setProjectPath(result.projectRoot);
        toast.success("Project location updated.");
      }
    } catch (err) {
      console.error("Failed to select project path", err);
      toast.error("Failed to select project location.");
    }
  };

  const handleCopyPath = async () => {
    if (!projectPath) return;
    try {
      await navigator.clipboard.writeText(projectPath);
      toast.success("Path copied to clipboard.");
    } catch (err) {
      console.error("Clipboard copy failed", err);
      toast.error("Could not copy path.");
    }
  };

  const handleColorChange = (color: string) => {
    setDefaultColor(color);
    localStorage.setItem("biovision-default-color", color);
  };

  const handleOpacityChange = (value: number[]) => {
    const opacity = value[0];
    setDefaultOpacity(opacity);
    localStorage.setItem("biovision-default-opacity", opacity.toString());
  };

  const themeOptions: { label: string; value: Theme; icon: React.ReactNode }[] = [
    { label: "Light", value: "light", icon: <Sun className="h-4 w-4" /> },
    { label: "Dark", value: "dark", icon: <Moon className="h-4 w-4" /> },
    { label: "System", value: "system", icon: <Monitor className="h-4 w-4" /> },
  ];

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[85vh] overflow-y-auto sm:max-w-lg">
        <DialogHeader>
          <DialogTitle className="text-sm font-bold">Settings</DialogTitle>
        </DialogHeader>

          <div className="space-y-4 py-4">
            {/* Project Settings */}
            <SettingsSection title="Project">
              <div className="space-y-3">
                <div>
                  <Label className="text-sm font-medium">Project Directory</Label>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Where images, labels, and models are stored
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <div className="min-w-0 flex-1 truncate rounded-md border bg-muted/50 px-3 py-2 font-mono text-xs">
                        {projectPath || "Loading..."}
                      </div>
                    </TooltipTrigger>
                    <TooltipContent side="bottom" align="start">
                      {projectPath || "Loading..."}
                    </TooltipContent>
                  </Tooltip>
                  <motion.div {...buttonHover} {...buttonTap}>
                    <Button
                      variant="outline"
                      size="icon"
                      onClick={handleCopyPath}
                      disabled={!projectPath}
                    >
                      <Copy className="h-4 w-4" />
                    </Button>
                  </motion.div>
                </div>
                <motion.div {...buttonHover} {...buttonTap}>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleSelectProjectPath}
                    className="w-full"
                  >
                    <FolderOpen className="mr-2 h-4 w-4" />
                    Change Directory
                  </Button>
                </motion.div>
              </div>
            </SettingsSection>

            {/* Display Settings */}
            <SettingsSection title="Display">
              <div className="space-y-4">
                {/* Theme */}
                <div className="space-y-2">
                  <Label className="text-sm font-medium">Theme</Label>
                  <div className="flex gap-2">
                    {themeOptions.map((option) => (
                      <motion.div key={option.value} {...buttonHover} {...buttonTap}>
                        <Button
                          variant={theme === option.value ? "default" : "outline"}
                          size="sm"
                          onClick={() => setTheme(option.value)}
                          className="flex-1"
                        >
                          {option.icon}
                          <span className="ml-2">{option.label}</span>
                        </Button>
                      </motion.div>
                    ))}
                  </div>
                </div>

                <Separator />

                {/* Default Landmark Color */}
                <div className="space-y-2">
                  <Label className="text-sm font-medium">Default Landmark Color</Label>
                  <div className="grid grid-cols-4 gap-2">
                    {LANDMARK_COLORS.map((color) => (
                      <motion.div key={color.value} {...buttonHover} {...buttonTap}>
                        <button
                          onClick={() => handleColorChange(color.value)}
                          className={cn(
                            "flex h-10 w-full items-center justify-center rounded-md border transition-all",
                            defaultColor === color.value
                              ? "border-primary ring-2 ring-primary ring-offset-2 ring-offset-background"
                              : "border-border hover:border-primary/50"
                          )}
                        >
                          <div
                            className="h-5 w-5 rounded-full"
                            style={{ backgroundColor: color.value }}
                          />
                        </button>
                      </motion.div>
                    ))}
                  </div>
                </div>

                <Separator />

                {/* Default Opacity */}
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <Label className="text-sm font-medium">Default Opacity</Label>
                    <span className="text-xs text-muted-foreground">
                      {defaultOpacity}%
                    </span>
                  </div>
                  <Slider
                    value={[defaultOpacity]}
                    onValueChange={handleOpacityChange}
                    min={10}
                    max={100}
                    step={5}
                    className="w-full"
                  />
                </div>
              </div>
            </SettingsSection>

            {/* Training Settings (future) */}
            <SettingsSection title="Training (Advanced)" defaultOpen={false}>
              <p className="text-xs text-muted-foreground">
                Advanced training parameters will be available in a future update.
              </p>
            </SettingsSection>
          </div>
      </DialogContent>
    </Dialog>
  );
};

export default SettingsModal;
