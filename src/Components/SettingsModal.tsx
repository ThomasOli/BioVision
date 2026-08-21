import React, { useEffect, useState } from "react";
import { FolderOpen, Monitor, Moon, Sun, Copy } from "lucide-react";
import { toast } from "sonner";

import { cn } from "@/lib/utils";
import { Button } from "@/Components/ui/button";
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

const SectionLabel: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <p className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground/70 mb-3">
    {children}
  </p>
);

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
      <DialogContent className="sm:max-w-[500px] max-h-[90vh] overflow-x-hidden overflow-y-auto scrollbar-app">
        <DialogHeader className="pb-1">
          <DialogTitle className="font-display text-base font-semibold">Settings</DialogTitle>
        </DialogHeader>

        <div className="space-y-6 py-2">

          {/* ── Project ─────────────────────────────────────── */}
          <div>
            <SectionLabel>Project</SectionLabel>
            <div className="space-y-2">
              <Label className="text-sm font-medium">Project Directory</Label>
              <p className="text-xs text-muted-foreground">
                Where images, labels, and models are stored
              </p>
              <div className="flex items-center gap-2 rounded-md border border-border/60 bg-muted/40 px-3 py-2">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span className="flex-1 truncate font-mono text-xs text-muted-foreground cursor-default">
                      {projectPath || "Loading..."}
                    </span>
                  </TooltipTrigger>
                  <TooltipContent side="bottom" align="start">
                    {projectPath || "Loading..."}
                  </TooltipContent>
                </Tooltip>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button
                      onClick={handleCopyPath}
                      disabled={!projectPath}
                      className="shrink-0 text-muted-foreground/50 transition-colors hover:text-foreground disabled:opacity-30"
                    >
                      <Copy className="h-3.5 w-3.5" />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent side="bottom">Copy path</TooltipContent>
                </Tooltip>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={handleSelectProjectPath}
                className="w-full text-xs"
              >
                <FolderOpen className="mr-2 h-3.5 w-3.5" />
                Browse...
              </Button>
            </div>
          </div>

          <Separator />

          {/* ── Display ─────────────────────────────────────── */}
          <div className="space-y-4">
            <SectionLabel>Display</SectionLabel>

            {/* Theme — segmented control */}
            <div className="space-y-1.5">
              <Label className="text-sm font-medium">Theme</Label>
              <div className="grid grid-cols-3 gap-1 rounded-lg border border-border/50 bg-muted/30 p-1">
                {themeOptions.map((option) => (
                  <button
                    key={option.value}
                    onClick={() => setTheme(option.value)}
                    className={cn(
                      "flex items-center justify-center gap-1.5 rounded-md py-1.5 text-xs font-medium transition-all duration-150",
                      theme === option.value
                        ? "bg-background text-foreground shadow-sm ring-1 ring-border/40"
                        : "text-muted-foreground hover:text-foreground"
                    )}
                  >
                    {option.icon}
                    {option.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Landmark color — circles */}
            <div className="space-y-2">
              <Label className="text-sm font-medium">Default Landmark Color</Label>
              <div className="flex flex-wrap gap-2.5">
                {LANDMARK_COLORS.map((color) => (
                  <Tooltip key={color.value}>
                    <TooltipTrigger asChild>
                      <button
                        onClick={() => handleColorChange(color.value)}
                        className={cn(
                          "h-7 w-7 rounded-full border-2 transition-all duration-150",
                          defaultColor === color.value
                            ? "border-foreground shadow-md scale-110"
                            : "border-transparent opacity-70 hover:opacity-100 hover:scale-105"
                        )}
                        style={{ backgroundColor: color.value }}
                      />
                    </TooltipTrigger>
                    <TooltipContent side="bottom">{color.name}</TooltipContent>
                  </Tooltip>
                ))}
              </div>
            </div>

            {/* Opacity */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-sm font-medium">Default Opacity</Label>
                <span className="font-mono text-xs text-muted-foreground">{defaultOpacity}%</span>
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

        </div>
      </DialogContent>
    </Dialog>
  );
};

export default SettingsModal;
