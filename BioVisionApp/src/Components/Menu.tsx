import React from "react";
import Paper from "@mui/material/Paper";
import Button from "@mui/material/Button";
import Box from "@mui/material/Box";
import UploadImages from "./UploadImages";
import IconButton from "@mui/material/IconButton";
import FolderOpenIcon from "@mui/icons-material/FolderOpen";
import Landmark from "./Landmark";
interface MenuProps {
  children: React.ReactNode;
}

const Menu: React.FC<MenuProps> = ({ children }) => {
  return (
    <Paper
      style={{
        flex: "0 0 100%",
        height: "96.5vh",
        padding: "16px",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {children}
      <div
        style={{
          marginTop: "1px",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          height: "100%",
        }}
      >
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
       
        <Landmark></Landmark>
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            marginBottom: "90px",
          }}
        >
          {/* Add more buttons as needed */}
          <Button variant="contained">Auto Landmark</Button>
          {/* Add more buttons as needed */}
        </div>
      </div>
    </Paper>
  );
};

export default Menu;
