import React, { useMemo, useState } from "react";
import {
  Box,
  IconButton,
  Slider,
  Typography,
  Switch,
  Tooltip,
  TextField,
  Popover,
  Divider,
  Stack,
  ClickAwayListener,
} from "@mui/material";
import {
  Clear as ClearIcon,
  Undo as UndoIcon,
  Redo as RedoIcon,
  Close as CloseIcon,
  KeyboardArrowDown as ChevronDownIcon,
} from "@mui/icons-material";
import { HexColorPicker } from "react-colorful";
import { UndoRedoClearContext } from "./UndoRedoClearContext";

interface LandmarkProps {
  onColorChange: (selectedColor: string) => void;
  onOpacityChange: (selectedOpacity: number) => void;
  onSwitchChange: () => void; // compatibility
}

const isValidHex = (v: string) => /^#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})$/.test(v);

const normalizeHex = (raw: string) => {
  let v = raw.trim();
  if (!v.startsWith("#")) v = `#${v}`;
  return v;
};

const clamp = (n: number, min: number, max: number) => Math.min(max, Math.max(min, n));

const hexToRgba = (hex: string, alpha01: number) => {
  const h = hex.replace("#", "");
  const full = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
  const r = parseInt(full.slice(0, 2), 16);
  const g = parseInt(full.slice(2, 4), 16);
  const b = parseInt(full.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha01})`;
};

const fontFamily = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";

const Landmark: React.FC<LandmarkProps> = ({ onColorChange, onOpacityChange, onSwitchChange }) => {
  // Preview-only toggle (ON = preview only / tools hidden / edits disabled)
  const [previewOnly, setPreviewOnly] = useState(false);

  const [color, setColor] = useState("#ff0000");
  const [hexInput, setHexInput] = useState("#ff0000");
  const [opacity, setOpacity] = useState<number>(100);

  // Popover state
  const [colorAnchor, setColorAnchor] = useState<HTMLElement | null>(null);
  const colorOpen = Boolean(colorAnchor);

  const { clear, undo, redo } = React.useContext(UndoRedoClearContext);

  const swatches = useMemo(
    () => [
      "#ff3b30", // red
      "#ff9500", // orange
      "#ffcc00", // yellow
      "#34c759", // green
      "#00c7be", // teal
      "#007aff", // blue
      "#5856d6", // indigo
      "#af52de", // purple
      "#ff2d55", // pink
      "#111827", // near-black
      "#ffffff", // white
    ],
    []
  );

  const handlePreviewOnlyChange = () => {
    setPreviewOnly((prev) => !prev);
    onSwitchChange();
  };

  const handleColorChange = (newColor: string) => {
    setColor(newColor);
    setHexInput(newColor);
    onColorChange(newColor);
  };

  const handleOpenColor = (e: React.MouseEvent<HTMLElement>) => {
    if (!previewOnly) setColorAnchor(e.currentTarget);
  };

  const handleCloseColor = () => setColorAnchor(null);

  const handleHexInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const next = normalizeHex(e.target.value);
    setHexInput(next);
    if (isValidHex(next)) {
      handleColorChange(next.toLowerCase());
    }
  };

  const handleOpacityChange = (_event: Event, newValue: number | number[]) => {
    const v = newValue as number;
    setOpacity(v);
    onOpacityChange(v);
  };

  const handleOpacityInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const value = event.target.value === "" ? 0 : Number(event.target.value);
    const clampedValue = clamp(value, 0, 100);
    setOpacity(clampedValue);
    onOpacityChange(clampedValue);
  };

  const opacity01 = opacity / 100;

  return (
    <Box
      sx={{
        p: 2,
        borderRadius: "12px",
        backgroundColor: "#ffffff",
        border: "1px solid #e5e7eb",
        boxShadow: "0 1px 0 rgba(15, 23, 42, 0.03)",
        fontFamily,
      }}
    >
      {/* Header */}
      <Box sx={{ mb: 1.5 }}>
        <Typography
          sx={{
            fontFamily,
            fontWeight: 700,
            fontSize: "13px",
            color: "#0f172a",
            lineHeight: 1.2,
          }}
        >
          LANDMARK TOOLS
        </Typography>

        {/* Toggle under title */}
        <Box sx={{ display: "flex", alignItems: "center", gap: 1, mt: 0.75 }}>
          <Typography sx={{ fontFamily, fontSize: "12px", color: "#475569" }}>View Only</Typography>
          <Switch
            checked={previewOnly}
            onChange={handlePreviewOnlyChange}
            sx={{
              ml: "auto",
              mt: "-4px",
              "& .MuiSwitch-switchBase.Mui-checked": { color: "#3b82f6" },
              "& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track": { backgroundColor: "#3b82f6" },
            }}
          />
        </Box>
      </Box>

      {/* Preview-only hint */}
      {previewOnly && (
        <Box
          sx={{
            mb: 1.5,
            px: 1.25,
            py: 1,
            borderRadius: "10px",
            border: "1px solid #e5e7eb",
            backgroundColor: "#f8fafc",
          }}
        >
          <Typography sx={{ fontFamily, fontSize: "11px", color: "#475569" }}>
            Editing tools hidden and changes disabled.
          </Typography>
        </Box>
      )}

      {/* Collapsible body */}
      <Box
        sx={{
          maxHeight: previewOnly ? 0 : "900px",
          overflow: "hidden",
          opacity: previewOnly ? 0 : 1,
          transition: "all 0.32s ease-in-out",
          pointerEvents: previewOnly ? "none" : "auto",
        }}
      >
        <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
          {/* Action icons (more spaced out) */}
          <Box sx={{ display: "flex", justifyContent: "center" }}>
            <Box
              sx={{
                display: "flex",
                gap: 8, // increased spacing
                p: 0.75,
                borderRadius: "999px",
                border: "1px solid #e5e7eb",
                backgroundColor: "#f8fafc",
              }}
            >
              <Tooltip title="Clear all landmarks" arrow>
                <span>
                  <IconButton
                    onClick={() => clear()}
                    size="small"
                    aria-label="Clear landmarks"
                    sx={{
                      borderRadius: "999px",
                      color: "#334155",
                      "&:hover": { backgroundColor: "#eef2ff" },
                    }}
                  >
                    <ClearIcon fontSize="small" />
                  </IconButton>
                </span>
              </Tooltip>

              <Tooltip title="Undo last action" arrow>
                <span>
                  <IconButton
                    onClick={() => undo()}
                    size="small"
                    aria-label="Undo"
                    sx={{
                      borderRadius: "999px",
                      color: "#334155",
                      "&:hover": { backgroundColor: "#eef2ff" },
                    }}
                  >
                    <UndoIcon fontSize="small" />
                  </IconButton>
                </span>
              </Tooltip>

              <Tooltip title="Redo last action" arrow>
                <span>
                  <IconButton
                    onClick={() => redo()}
                    size="small"
                    aria-label="Redo"
                    sx={{
                      borderRadius: "999px",
                      color: "#334155",
                      "&:hover": { backgroundColor: "#eef2ff" },
                    }}
                  >
                    <RedoIcon fontSize="small" />
                  </IconButton>
                </span>
              </Tooltip>
            </Box>
          </Box>

          {/* Color row */}
          <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <Box>
              <Typography sx={{ fontFamily, fontWeight: 700, fontSize: "12px", color: "#475569" }}>
                Landmark color
              </Typography>
            </Box>

            <Box
              role="button"
              onClick={handleOpenColor}
              sx={{
                display: "flex",
                alignItems: "center",
                gap: 1,
                px: 1.25,
                py: 0.75,
                borderRadius: "10px",
                border: "1px solid #e5e7eb",
                backgroundColor: "#ffffff",
                cursor: "pointer",
                userSelect: "none",
                "&:hover": { backgroundColor: "#f8fafc", borderColor: "#d1d5db" },
              }}
            >
              {/* color chip with transparency applied */}
              <Box
                sx={{
                  width: 18,
                  height: 18,
                  borderRadius: "6px",
                  border: "1px solid #e5e7eb",
                  backgroundImage:
                    "linear-gradient(45deg, #e5e7eb 25%, transparent 25%), linear-gradient(-45deg, #e5e7eb 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #e5e7eb 75%), linear-gradient(-45deg, transparent 75%, #e5e7eb 75%)",
                  backgroundSize: "8px 8px",
                  backgroundPosition: "0 0, 0 4px, 4px -4px, -4px 0px",
                  overflow: "hidden",
                }}
              >
                <Box sx={{ width: "100%", height: "100%", backgroundColor: hexToRgba(color, opacity01) }} />
              </Box>

              <Typography sx={{ fontFamily, fontSize: "12.5px", fontWeight: 700, color: "#0f172a" }}>
                {color.toUpperCase()}
              </Typography>

              <ChevronDownIcon sx={{ color: "#64748b", fontSize: 18 }} />
            </Box>
          </Box>

          {/* Color popover */}
          <Popover
            open={colorOpen}
            anchorEl={colorAnchor}
            onClose={handleCloseColor}
            anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
            transformOrigin={{ vertical: "top", horizontal: "right" }}
            PaperProps={{
              sx: {
                width: 290,
                borderRadius: "14px",
                border: "1px solid #e5e7eb",
                boxShadow: "0 18px 55px rgba(0,0,0,0.14)",
                p: 1.5,
                overflow: "hidden",
              },
            }}
          >
            <ClickAwayListener onClickAway={handleCloseColor}>
              <Box>
                <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 1 }}>
                  <Typography sx={{ fontFamily, fontWeight: 800, fontSize: "12px", color: "#0f172a" }}>
                    Color
                  </Typography>
                  <IconButton size="small" onClick={handleCloseColor} aria-label="Close color picker">
                    <CloseIcon fontSize="small" />
                  </IconButton>
                </Box>

                <Box sx={{ borderRadius: "12px", overflow: "hidden", border: "1px solid #e5e7eb" }}>
                  <HexColorPicker color={color} onChange={handleColorChange} style={{ width: "100%", height: 170 }} />
                </Box>

                <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap" sx={{ mt: 1.25 }}>
                  {swatches.map((s) => (
                    <Box
                      key={s}
                      onClick={() => handleColorChange(s)}
                      sx={{
                        width: 22,
                        height: 22,
                        borderRadius: "7px",
                        backgroundColor: s,
                        border: s.toLowerCase() === color.toLowerCase() ? "2px solid #0f172a" : "1px solid #e5e7eb",
                        cursor: "pointer",
                      }}
                      title={s.toUpperCase()}
                    />
                  ))}
                </Stack>

                <Divider sx={{ my: 1.25 }} />

                <TextField
                  label="HEX"
                  value={hexInput}
                  onChange={handleHexInputChange}
                  size="small"
                  fullWidth
                  error={hexInput.length > 1 && !isValidHex(hexInput)}
                  helperText={hexInput.length > 1 && !isValidHex(hexInput) ? "Use #RGB or #RRGGBB" : " "}
                  sx={{
                    "& .MuiOutlinedInput-root": {
                      borderRadius: "12px",
                      "& fieldset": { borderColor: "#d1d5db" },
                      "&:hover fieldset": { borderColor: "#9ca3af" },
                      "&.Mui-focused fieldset": { borderColor: "#3b82f6" },
                    },
                  }}
                />
              </Box>
            </ClickAwayListener>
          </Popover>

          {/* Transparency */}
          <Box>
            <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", mb: 1 }}>
              <Typography sx={{ fontFamily, fontWeight: 700, fontSize: "12px", color: "#475569" }}>
                Transparency
              </Typography>

              <TextField
                value={opacity.toString()}
                onChange={handleOpacityInputChange}
                size="small"
                type="number"
                inputProps={{
                  min: 0,
                  max: 100,
                  style: { fontFamily, textAlign: "right", width: "38px", paddingRight: "4px" },
                }}
                InputProps={{
                  endAdornment: (
                    <Typography sx={{ fontFamily, fontSize: "12px", color: "#64748b", ml: 0.5 }}>%</Typography>
                  ),
                }}
                sx={{
                  width: "78px",
                  "& .MuiOutlinedInput-root": {
                    height: "34px",
                    borderRadius: "12px",
                    "& fieldset": { borderColor: "#d1d5db" },
                    "&:hover fieldset": { borderColor: "#9ca3af" },
                    "&.Mui-focused fieldset": { borderColor: "#3b82f6" },
                  },
                  "& input[type=number]::-webkit-inner-spin-button, & input[type=number]::-webkit-outer-spin-button": {
                    display: "none",
                  },
                  "& input[type=number]": { MozAppearance: "textfield" },
                }}
              />
            </Box>

            <Box sx={{ px: 1 }}>
              <Slider
                aria-label="Transparency"
                value={opacity}
                onChange={handleOpacityChange}
                valueLabelDisplay="auto"
                sx={{
                  color: "#3b82f6",
                  "& .MuiSlider-thumb": {
                    width: 10,
                    height: 10,
                    "&:hover, &.Mui-focusVisible": { boxShadow: "0 0 0 8px rgba(59, 130, 246, 0.16)" },
                  },
                  "& .MuiSlider-track": { height: 2 },
                  "& .MuiSlider-rail": { height: 2, backgroundColor: "#e5e7eb" },
                }}
              />
            </Box>
          </Box>
        </Box>
      </Box>
    </Box>
  );
};

export default Landmark;
