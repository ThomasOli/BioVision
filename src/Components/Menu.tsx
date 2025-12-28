import React, { useEffect, useState } from "react";
import Paper from "@mui/material/Paper";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";
import Box from "@mui/material/Box";
import Tooltip from "@mui/material/Tooltip";
import { useSelector } from "react-redux";
import UploadImages from "./UploadImages";
import type { RootState } from "../state/store";
import Landmark from "./Landmark";
import { AnnotatedImage } from "../types/Image";
import { TrainModelDialog } from "./PopUp";
interface MenuProps {
  onOpacityChange: (selectedOpacity: number) => void;
  onColorChange: (selectedColor: string) => void;
  onSwitchChange: () => void;
}

async function saveLabels(fileArray: AnnotatedImage[]) {
  console.log(fileArray);
  await window.api.saveLabels(fileArray);
}

const Menu: React.FC<MenuProps> = ({
  onColorChange,
  onOpacityChange,
  onSwitchChange,
}) => {
  const [openTrainDialog, setOpenTrainDialog] = useState(false);
  const [modelName, setModelName] = useState("");
  const [isTraining, setIsTraining] = useState(false);
  const [modelPath, setModelPath] = useState("");

  const fileArray = useSelector((state: RootState) => state.files.fileArray);

  useEffect(() => {
    const fetchProjectRoot = async () => {
      try {
        const result = await window.api.getProjectRoot();
        if (result?.projectRoot) {
          setModelPath(result.projectRoot);
        }
      } catch (err) {
        console.error("Failed to load model path", err);
      }
    };
    fetchProjectRoot();
  }, []);

  const handleSelectModelPath = async () => {
    try {
      const result = await window.api.selectProjectRoot();
      if (!result?.canceled && result?.projectRoot) {
        setModelPath(result.projectRoot);
      }
    } catch (err) {
      console.error("Failed to select model path", err);
    }
  };

  const handleTrainConfirm = async () => {
    const name = modelName.trim();
    if (!name) return;

    try {
      setIsTraining(true);
      await saveLabels(fileArray);

      const result = await window.api.trainModel(name);

      if (!result.ok) {
        throw new Error(result.error);
      }

      console.log("Training output:", result.output);

      setOpenTrainDialog(false);
      setModelName("");
    } catch (err) {
      console.error(err);
      alert(`Training failed. ${err}`);
    } finally {
      setIsTraining(false);
    }
  };

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
      <TrainModelDialog handleTrainConfirm={handleTrainConfirm} open={openTrainDialog} setOpen={setOpenTrainDialog} modelName={modelName} isTraining={isTraining} setModelName={setModelName}/> 
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
          <Box
            sx={{
              width: "100%",
              p: "10px 12px",
              boxSizing: "border-box",
              backgroundColor: "#f8fafc",
              border: "1px solid #e5e7eb",
              borderRadius: "8px",
              boxShadow: "inset 0 1px 0 rgba(255,255,255,0.6)",
              display: "flex",
              flexDirection: "column",
              gap: "6px",
            }}
          >
            <Typography variant="body2" color="textSecondary" sx={{ fontWeight: 600 }}>
              Model folder
            </Typography>
            <Tooltip title={modelPath || "Loading..."} placement="bottom-start">
              <Typography
                variant="body2"
                sx={{
                  fontFamily: '"Fira Code", "SFMono-Regular", Consolas, monospace',
                  color: "#374151",
                  maxWidth: "100%",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {modelPath || "Loading..."}
              </Typography>
            </Tooltip>
            <Button
              size="small"
              variant="outlined"
              onClick={handleSelectModelPath}
              sx={{
                alignSelf: "flex-start",
                textTransform: "none",
                borderRadius: "999px",
                paddingInline: "12px",
              }}
            >
              Change model folder
            </Button>
          </Box>
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
          <Button
            variant="contained"
            color="secondary"
            onClick={() => setOpenTrainDialog(true)}
          >
            Train Model
          </Button>

          {/* Add more buttons as needed */}
        </div>
      </div>
    </Paper>
  );
};
export default Menu;
