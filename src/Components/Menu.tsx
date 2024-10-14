import React from "react";
import Paper from "@mui/material/Paper";
import Button from "@mui/material/Button";

import UploadImages from "./UploadImages";

import Landmark from "./Landmark";
interface MenuProps {
  onOpacityChange: (selectedOpacity: number) => void;
  onColorChange: (selectedColor: string) => void;
}

const Menu: React.FC<MenuProps> = ({ onColorChange, onOpacityChange }) => {
  return (
    <Paper
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
