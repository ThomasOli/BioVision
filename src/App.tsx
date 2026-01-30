// src/App.tsx
import React, { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Box, useMediaQuery } from "@mui/material";
import Menu from "./Components/Menu";
import ImageLabelerCarousel from "./Components/ImageLablerCarousel";
import { UndoRedoClearContextProvider } from "./Components/UndoRedoClearContext";

const clamp = (n: number, min: number, max: number) => Math.min(max, Math.max(min, n));

const App: React.FC = () => {
  const [color, setColor] = useState<string>("red");
  const [isSwitchOn, setIsSwitchOn] = useState(false);
  const [opacity, setOpacity] = useState<number>(100);

  const handleColorChange = (selectedColor: string) => setColor(selectedColor);
  const handleSwitchChange = () => setIsSwitchOn((prev) => !prev);
  const handleOpacityChange = (selectedOpacity: number) => setOpacity(selectedOpacity);

  // ---- responsive bounds for menu ----
  const isXs = useMediaQuery("(max-width:600px)");
  const MIN_MENU = isXs ? 200 : 305;
  const MAX_MENU = isXs ? 360 : 680;

  // ---- menu width that tracks real menu size until user drags ----
  const pageRef = useRef<HTMLDivElement | null>(null);
  const menuWrapRef = useRef<HTMLDivElement | null>(null);

  const draggingRef = useRef(false);
  const userResizedRef = useRef(false);

  const [menuWidth, setMenuWidth] = useState<number>(() => {
    const saved = Number(localStorage.getItem("menuWidth"));
    return Number.isFinite(saved) && saved > 0 ? saved : isXs ? 300 : 380;
  });

  // Persist only after user has resized (so auto-sizing doesn't overwrite saved width constantly)
  useEffect(() => {
    if (userResizedRef.current) localStorage.setItem("menuWidth", String(menuWidth));
  }, [menuWidth]);

  // Measure initial natural width once mounted
  useLayoutEffect(() => {
    if (!menuWrapRef.current) return;

    const el = menuWrapRef.current;
    const natural = el.scrollWidth || el.getBoundingClientRect().width;

    setMenuWidth(() => {
      const saved = Number(localStorage.getItem("menuWidth"));
      const hasSaved = Number.isFinite(saved) && saved > 0;
      return hasSaved ? clamp(saved, MIN_MENU, MAX_MENU) : clamp(Math.round(natural), MIN_MENU, MAX_MENU);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Track menu content width changes automatically (until user drags)
  useEffect(() => {
    if (!menuWrapRef.current) return;

    const el = menuWrapRef.current;

    const ro = new ResizeObserver(() => {
      if (userResizedRef.current) return;
      const natural = el.scrollWidth || el.getBoundingClientRect().width;
      setMenuWidth(clamp(Math.round(natural), MIN_MENU, MAX_MENU));
    });

    ro.observe(el);
    return () => ro.disconnect();
  }, [MIN_MENU, MAX_MENU]);

  // Drag handlers
  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      if (!draggingRef.current) return;
      if (!pageRef.current) return;

      const rect = pageRef.current.getBoundingClientRect();
      const next = e.clientX - rect.left;
      setMenuWidth(clamp(next, MIN_MENU, MAX_MENU));
    };

    const onMouseUp = () => {
      if (!draggingRef.current) return;
      draggingRef.current = false;
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };

    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };
  }, [MIN_MENU, MAX_MENU]);

  const startDrag = () => {
    userResizedRef.current = true;
    draggingRef.current = true;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  };

  return (
    <UndoRedoClearContextProvider>
      <Box
        ref={pageRef}
        sx={{
          width: "100vw",
          height: "100vh",
          minWidth: 800,
          minHeight: 500,
          bgcolor: "#fff",
          display: "flex",
          overflow: "auto",
        }}
      >
        {/* Left: menu container */}
        <Box
          sx={{
            width: menuWidth,
            flexShrink: 0,
            height: "100%",
            overflow: "hidden",
            borderRight: "1px solid #e5e7eb",
            bgcolor: "#fff",
          }}
        >
          <Box
            ref={menuWrapRef}
            sx={{
              height: "100%",
              overflowY: "auto",
              overflowX: "hidden",
              bgcolor: "#fff",
            }}
          >
            <Menu
              onOpacityChange={handleOpacityChange}
              onColorChange={handleColorChange}
              onSwitchChange={handleSwitchChange}
            />
          </Box>
        </Box>

        {/* Drag handle */}
        <Box
          onMouseDown={startDrag}
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize menu"
          sx={{
            width: 10,
            flexShrink: 0,
            cursor: "col-resize",
            position: "relative",
            bgcolor: "transparent",
            "&::after": {
              content: '""',
              position: "absolute",
              top: 0,
              bottom: 0,
              left: "50%",
              transform: "translateX(-50%)",
              width: "2px",
              backgroundColor: "#e5e7eb",
              opacity: 0.95,
            },
            "&:hover": { backgroundColor: "rgba(59,130,246,0.06)" },
          }}
        />

        {/* Right: fills ALL remaining space */}
        <Box
          sx={{
            flex: 1,
            height: "100%",
            bgcolor: "#fbfbfb",
            overflow: "hidden", // keep the pane clean; carousel can manage its own scroll if needed
            display: "flex",
            minWidth: 0,
          }}
        >
          {/* This is the container that is exactly "everything not menu" */}
          <Box
            sx={{
              width: "100%",
              height: "100%",
              minWidth: 0,
              minHeight: 0,
              p: { xs: 1, md: 1 },     // padding around the card
              boxSizing: "border-box",
              display: "flex",
            }}
          >
            {/* Let the carousel be full size */}
            <Box sx={{ width: "100%", height: "100%", minWidth: 0, minHeight: 0, display: "flex" }}>
              <ImageLabelerCarousel color={color} opacity={opacity} isSwitchOn={isSwitchOn} />
            </Box>
          </Box>
        </Box>
      </Box>
    </UndoRedoClearContextProvider>
  );
};

export default App;
