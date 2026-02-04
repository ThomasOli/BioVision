import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import { Species } from "../../types/Image";

interface SpeciesState {
  species: Species[];
  activeSpeciesId: string | null;
}

const initialState: SpeciesState = {
  species: [],
  activeSpeciesId: null,
};

const speciesSlice = createSlice({
  name: "species",
  initialState,
  reducers: {
    addSpecies: (state, action: PayloadAction<Species>) => {
      state.species.push(action.payload);
      state.activeSpeciesId = action.payload.id;
    },
    setActiveSpecies: (state, action: PayloadAction<string>) => {
      state.activeSpeciesId = action.payload;
    },
    updateSpecies: (state, action: PayloadAction<{ id: string; updates: Partial<Species> }>) => {
      const species = state.species.find((s) => s.id === action.payload.id);
      if (species) {
        Object.assign(species, action.payload.updates);
      }
    },
    deleteSpecies: (state, action: PayloadAction<string>) => {
      state.species = state.species.filter((s) => s.id !== action.payload);
      if (state.activeSpeciesId === action.payload) {
        state.activeSpeciesId = null;
      }
    },
  },
});

export const { addSpecies, setActiveSpecies, updateSpecies, deleteSpecies } = speciesSlice.actions;
export default speciesSlice.reducer;
