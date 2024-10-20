import React, { useState } from "react";
import {
  Box,
  Button,
  ButtonGroup,
  Slider,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
  Switch,
} from "@mui/material";
import {
  FormatBold as FormatBoldIcon,
  FormatItalic as FormatItalicIcon,
  FormatUnderlined as FormatUnderlinedIcon,
} from "@mui/icons-material";
import { HexColorPicker } from "react-colorful";

interface LandmarkProps {
  onColorChange: (selectedColor: string) => void;
  onOpacityChange: (selectedOpacity: number) => void;
}

function valuetext(value: number) {
  return `${value}°C`;
}

const Landmark: React.FC<LandmarkProps> = ({ onColorChange, onOpacityChange }) => {
  const [isSwitchOn, setIsSwitchOn] = useState(false);
  const [color, setColor] = useState('#ff0000'); // Default to red color
  const [opacity, setOpacity] = useState<number>(100);
  const [formats, setFormats] = useState(() => ["bold", "italic"]);

  const handleSwitchChange = () => {
    setIsSwitchOn((prev) => !prev);
  };

  const handleColorChange = (newColor: string) => {
    setColor(newColor);
    onColorChange(newColor); // Call the parent's callback with new color
  };

  const handleOpacityChange = (event: Event, newValue: number | number[]) => {
    setOpacity(newValue as number);
    onOpacityChange(newValue as number);
  };

  const handleFormat = (
    event: React.MouseEvent<HTMLElement>,
    newFormats: string[]
  ) => {
    setFormats(newFormats);
  };

  return (
    <div style={{ padding: "20px", border: "1px solid #ccc" }}>
      <Box
        sx={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "20px",
        }}
      >
        <Typography>View Mode</Typography>
        <Switch
          checked={isSwitchOn}
          onChange={handleSwitchChange}
          color="primary"
        />
      </Box>

      <div
        style={{
          filter: isSwitchOn ? "grayscale(100%)" : "none",
          pointerEvents: isSwitchOn ? "none" : "auto",
        }}
      >
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            flexDirection: "column",
          }}
        >
          <ButtonGroup
            variant="contained"
            aria-label="outlined primary button group"
            disabled={isSwitchOn}
          >
            <Button>Clear</Button>
            <Button>Undo</Button>
            <Button>Redo</Button>
          </ButtonGroup>

          <br />

          <ToggleButtonGroup
            value={formats}
            onChange={handleFormat}
            aria-label="text formatting"
            disabled={isSwitchOn}
          >
            <ToggleButton value="bold" aria-label="bold">
              <FormatBoldIcon />
            </ToggleButton>
            <ToggleButton value="italic" aria-label="italic">
              <FormatItalicIcon />
            </ToggleButton>
            <ToggleButton value="underlined" aria-label="underlined">
              <FormatUnderlinedIcon />
            </ToggleButton>
          </ToggleButtonGroup>

          <br />

          <Typography variant="body1" color="textSecondary">
            Select Color
          </Typography>
          <Box sx={{ width: "150px", height: "150px", marginBottom: "20px" }}>
            <HexColorPicker
              color={color}
              onChange={handleColorChange}
              style={{ width: "100%", height: "100%" }}
            />
          </Box>
          <Typography variant="body1" color="textSecondary">
            Transparency
          </Typography>
          <Slider
            aria-label="Transparency"
            defaultValue={opacity}
            getAriaValueText={valuetext}
            color="primary"
            disabled={isSwitchOn}
            valueLabelDisplay="auto"
            value={opacity}
            onChange={handleOpacityChange}
          />
        </Box>
      </div>
    </div>
  );
};

export default Landmark;

