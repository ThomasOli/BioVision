import React, { useState } from "react";
import Select, { SelectChangeEvent } from "@mui/material/Select";

import {
  Box,
  Button,
  ButtonGroup,
  Slider,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
  Switch,
  MenuItem,
  InputLabel,
  FormControl,
} from "@mui/material";
import {
  ArrowDropDown as ArrowDropDownIcon,
  FormatBold as FormatBoldIcon,
  FormatItalic as FormatItalicIcon,
  FormatUnderlined as FormatUnderlinedIcon,
  FormatColorFill as FormatColorFillIcon,
} from "@mui/icons-material";

interface LandmarkProps {}

function valuetext(value: number) {
  return `${value}°C`;
}

const Landmark: React.FC<LandmarkProps> = () => {
  const [isSwitchOn, setIsSwitchOn] = useState(false);

  const handleSwitchChange = () => {
    setIsSwitchOn((prev) => !prev);
  };
  const [color, setColor] = React.useState("");

  const handleChange = (event: SelectChangeEvent) => {
    setColor(event.target.value as string);
  };
  const [formats, setFormats] = React.useState(() => ["bold", "italic"]);

  const handleFormat = (
    event: React.MouseEvent<HTMLElement>,
    newFormats: string[]
  ) => {
    setFormats(newFormats);
  };

  return (
    <div style={{ padding: "20px", border: "1px solid #ccc" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <label>View Mode</label>
        <Switch
          checked={isSwitchOn}
          onChange={handleSwitchChange}
          color="primary"
        />
      </div>
      <div
        style={{
          filter: isSwitchOn ? "grayscale(100%)" : "none",
          pointerEvents: isSwitchOn ? "none" : "auto",
        }}
      >
        {/* Elements inside the box */}
        <Box
          style={{
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
          <br></br>
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
            <Box sx={{ minWidth: 120 }}>
            <FormControl fullWidth>
              <InputLabel id="demo-simple-select-label">Color</InputLabel>
              <Select
                labelId="demo-simple-select-label"
                id="demo-simple-select"
                value={color}
                label="color"
                onChange={handleChange}
                disabled={isSwitchOn}
              >
                <MenuItem value={10}>Red</MenuItem>
                <MenuItem value={20}>Orange</MenuItem>
                <MenuItem value={30}>Yellow</MenuItem>
                <MenuItem value={40}>Green</MenuItem>
                <MenuItem value={50}>Blue</MenuItem>
                <MenuItem value={60}>Purple</MenuItem>
                <MenuItem value={70}>White</MenuItem>
                <MenuItem value={80}>Black</MenuItem>
              </Select>
            </FormControl>
            </Box>
          </ToggleButtonGroup>

          <br></br>
          <Typography variant="body1" color="textSecondary">
            Transparency
          </Typography>
          <Slider
            aria-label="Transparency"
            defaultValue={30}
            getAriaValueText={valuetext}
            color="primary"
            disabled={isSwitchOn}
          />
        </Box>
        {/* Add more elements as needed */}
      </div>
    </div>
  );
};

export default Landmark;
