import React, { useEffect } from "react";
import Paper from "@mui/material/Paper";
import Button from "@mui/material/Button";

import UploadImages from "./UploadImages";

import Landmark from "./Landmark";
interface MenuProps {
  onOpacityChange: (selectedOpacity: number) => void;
  onColorChange: (selectedColor: string) => void;
  onSwitchChange: () => void;
}

const Menu: React.FC<MenuProps> = ({ onColorChange, onOpacityChange, onSwitchChange }) => {

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.key === "n") {
        e.preventDefault();
        const event = new CustomEvent("open-upload-dialog");
        window.dispatchEvent(event);
      }
    };
  
    window.addEventListener("keydown", handleKeyDown);
  
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

  useEffect(() => {
    const openUploadDialog = () => {
      document.getElementById("btn-upload")?.click();
    };
  
    window.addEventListener("open-upload-dialog", openUploadDialog);
  
    return () => {
      window.removeEventListener("open-upload-dialog", openUploadDialog);
    };
  }, []);
  return (
    <Paper
      elevation={9}
      style={{
        height: "100vh",
        paddingLeft: "5px",
        paddingRight: "5px",
        display: "flex",
        flexDirection: "column",
        width: "325px",
        boxSizing: "border-box",
      }}
    >
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
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            whiteSpace: "nowrap",
          }}
        >
          <h3>Auto Landmarking Selection Menu</h3>
          <UploadImages />
        </div>

        <Landmark
          onOpacityChange={onOpacityChange}
          onColorChange={onColorChange}
          onSwitchChange={onSwitchChange}
        />

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
