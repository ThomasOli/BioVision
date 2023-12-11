import { PayloadAction, createSlice } from "@reduxjs/toolkit";


interface FileState{
    fileArray: File[];
}

const initialState: FileState = {
    fileArray: [],
}

const fileSlice = createSlice({
    name: "filearray",
    initialState,
    reducers: {
        addFile: (state, action: PayloadAction<File[]>) => {

            state.fileArray.push(...action.payload);
        },
        removeFile: (state, action: PayloadAction<string>) => {
            state.fileArray = state.fileArray.filter(file => file.name !== action.payload);
        },
        clearFiles: (state) => {
            state.fileArray = [];
        }
    },
});

export const {addFile, removeFile, clearFiles} = fileSlice.actions;

export default fileSlice.reducer;
