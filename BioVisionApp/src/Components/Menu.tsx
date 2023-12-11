import React from "react";
import Paper from "@mui/material/Paper";
import Button from "@mui/material/Button";
import Box from "@mui/material/Box";
import UploadImages from "./UploadImages";
import IconButton from "@mui/material/IconButton";
import FolderOpenIcon from "@mui/icons-material/FolderOpen";
import Landmark from "./Landmark";
import { Typography } from "@mui/material";
interface MenuProps {
  children: React.ReactNode;
}

const Menu: React.FC<MenuProps> = ({ children }) => {
  return (
    <Paper
      style={{
       
        height: "100vh",
        paddingLeft: "5px",
        paddingRight: "5px",
        display: "flex",
        flexDirection: "column",
        width: '325px',
        boxSizing: 'border-box',
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
        <div 
        style = {{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          whiteSpace: 'nowrap' 
          
        }}
        >
          <h3>Auto Landmarking Selection Menu</h3>
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
export default Menu 