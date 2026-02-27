import { createSlice, PayloadAction } from "@reduxjs/toolkit";

export type Device = "cpu" | "mps" | "cuda" | null;
export type CnnTier = "fast" | "slow";

export interface HardwareState {
  probed: boolean;
  device: Device;
  ramGb: number | null;
  gpuName: string | null;
  /** True only when device is cuda/mps AND system RAM >= 8 GB */
  sam2Enabled: boolean;
  /** Always true — YOLO-World runs on CPU */
  yoloWorldEnabled: boolean;
  /** "fast" for cuda/mps; "slow" for cpu-only */
  cnnTier: CnnTier;
}

const initialState: HardwareState = {
  probed: false,
  device: null,
  ramGb: null,
  gpuName: null,
  sam2Enabled: false,
  yoloWorldEnabled: true,
  cnnTier: "slow",
};

const hardwareSlice = createSlice({
  name: "hardware",
  initialState,
  reducers: {
    setHardwareCapabilities: (state, action: PayloadAction<HardwareState>) => {
      Object.assign(state, action.payload);
    },
  },
});

export const { setHardwareCapabilities } = hardwareSlice.actions;

// Selectors — typed with `any` to avoid circular dep with store.tsx
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const selectHardware = (state: any): HardwareState => state.hardware;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const selectSam2Enabled = (state: any): boolean => state.hardware.sam2Enabled;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const selectCnnTier = (state: any): CnnTier => state.hardware.cnnTier;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const selectDevice = (state: any): Device => state.hardware.device;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const selectHardwareProbed = (state: any): boolean => state.hardware.probed;

export default hardwareSlice.reducer;
