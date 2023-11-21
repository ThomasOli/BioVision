import React from "react";
import Paper from "@mui/material/Paper";
import Button from "@mui/material/Button";
import Box from "@mui/material/Box";
import UploadImages from "./UploadImages";
import IconButton from "@mui/material/IconButton";
import Slider from '@mui/material/Slider';
import FileOpenIcon from "@mui/icons-material/FileOpen";

import FolderOpenIcon from "@mui/icons-material/FolderOpen";
interface MenuProps {
  children: React.ReactNode;
}
function valuetext(value: number) {
  return `${value}°C`;
}
const Menu: React.FC<MenuProps> = ({ children }) => {
  return (
    <Paper style={{ flex: "0 0 100%", height: "100vh", padding: "16px", display: "flex", flexDirection: "column" }}>
      {children}
      <div style={{ marginTop: "1px", display: "flex", flexDirection: "column", justifyContent: "space-between", height: "100%" }}>
      <div>
        <Box style={{ display: "flex", alignItems: "center" }}>
          <IconButton
            style={{
              backgroundColor: "transparent",
              color: "blue",
            }}
            size="small"
          >
            <FolderOpenIcon />
          </IconButton>
          <p style={{ marginLeft: "8px" }}>Open Folder</p>
        </Box>

        <UploadImages />
        </div>
        <Slider
         aria-label="Temperature"
  defaultValue={30}
  getAriaValueText={valuetext}
  color="secondary"
/>
        <div style={{ display: "flex", justifyContent: "center", marginBottom: "100px" }}>
          {/* Add more buttons as needed */}
          <Button variant="contained">Auto Landmark</Button>
          {/* Add more buttons as needed */}
        </div>
      </div>
    </Paper>
  );
};

export default Menu;
