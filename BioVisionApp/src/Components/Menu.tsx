import React from "react";
import Paper from "@mui/material/Paper";
import Button from "@mui/material/Button";
import Box from "@mui/material/Box";
import UploadImages from "./UploadImages";
import IconButton from "@mui/material/IconButton";
import FileOpenIcon from "@mui/icons-material/FileOpen";

import FolderOpenIcon from "@mui/icons-material/FolderOpen";
interface MenuProps {
  children: React.ReactNode;
}

const Menu: React.FC<MenuProps> = ({ children }) => {
  return (
    <Paper style={{ flex: "0 0 100%", height: "100vh", padding: "16px" }}>
      {children}
      <div style={{ marginTop: "1px" }}>
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
          <p>Open Folder</p>
        </Box>

        <UploadImages></UploadImages>
        
        {/* Add more buttons as needed */}
      </div>
      
    </Paper>
    
  );
};

export default Menu;
