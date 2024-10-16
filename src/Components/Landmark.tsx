import React, { useState, useContext } from "react";
import Select, { SelectChangeEvent } from "@mui/material/Select";

import { UndoRedoClearContext } from "./UndoRedoClearContext";

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
  // ArrowDropDown as ArrowDropDownIcon,
  FormatBold as FormatBoldIcon,
  FormatItalic as FormatItalicIcon,
  FormatUnderlined as FormatUnderlinedIcon,
  // FormatColorFill as FormatColorFillIcon,
} from "@mui/icons-material";

interface LandmarkProps {
  onColorChange: (selectedColor: string) => void;
  onOpacityChange: (selectedOpacity: number) => void;
}

function valuetext(value: number) {
  return `${value}°C`;
}

const Landmark: React.FC<LandmarkProps> = ({
  onColorChange,
  onOpacityChange,
}) => {
  const [isSwitchOn, setIsSwitchOn] = useState(false);

  const { undo, redo, clear } = useContext(UndoRedoClearContext); 

  const handleSwitchChange = () => {
    setIsSwitchOn((prev) => !prev);
  };

  const [color, setColor] = React.useState("red");
  const [opacity, setOpacity] = React.useState<number>(100);

  const handleColorChange = (event: SelectChangeEvent) => {
    const selectedColor = event.target.value as string;
    setColor(selectedColor);
    // Call the onColorChange callback with the selected color
    onColorChange(selectedColor);
  };

  const handleOpacityChange = (event: Event, newValue: number | number[]) => {
    setOpacity(newValue as number);
    onOpacityChange(newValue as number);
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
            <Button onClick={() => clear()}>Clear</Button>
            <Button onClick={() => undo()}>Undo</Button>
            <Button onClick={() => redo()}>Redo</Button>
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
                  onChange={handleColorChange}
                  disabled={isSwitchOn}
                >
                  <MenuItem value={"red"}>Red</MenuItem>
                  <MenuItem value={"orange"}>Orange</MenuItem>
                  <MenuItem value={"yellow"}>Yellow</MenuItem>
                  <MenuItem value={"green"}>Green</MenuItem>
                  <MenuItem value={"blue"}>Blue</MenuItem>
                  <MenuItem value={"purple"}>Purple</MenuItem>
                  <MenuItem value={"white"}>White</MenuItem>
                  <MenuItem value={"black"}>Black</MenuItem>
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
            defaultValue={opacity}
            getAriaValueText={valuetext}
            color="primary"
            disabled={isSwitchOn}
            valueLabelDisplay="auto"
            value={opacity}
            onChange={handleOpacityChange}
          />
        </Box>
        {/* Add more elements as needed */}
      </div>
    </div>
  );
};

export default Landmark;
