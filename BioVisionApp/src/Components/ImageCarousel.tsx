import { useSelector, useDispatch } from "react-redux";
import { useState, useEffect } from 'react';

  import * as fabric from 'fabric'; 
  import { clearFiles, addFile, removeFile } from '../state/filesState/fileSlice';
import { RootState } from '../state/store';
import { IconButton, Card, Stack } from '@mui/material';
import ArrowBackIosNewIcon from '@mui/icons-material/ArrowBackIosNew';
import ArrowForwardIosIcon from '@mui/icons-material/ArrowForwardIos';
import { Delete, Draw } from "@mui/icons-material";
import DrawableCanvas from "./DrawableCanvas";
const ImageCarousel: React.FC = () => {
    const files = useSelector((state: RootState) => state.files.fileArray);
    const dispatch = useDispatch();
    const [currentIndex, setCurrentIndex] = useState(0);

    const [showNav, setshowNav] = useState(false);

    const handleNext = () => {
        setCurrentIndex((prevIndex) => (prevIndex + 1) % files.length);
    };

    const handleBack = () => {
        setCurrentIndex((prevIndex) => (prevIndex - 1 + files.length) % files.length);
    };

    const currentFile = files[currentIndex];
    const prevFile = files[(currentIndex - 1 + files.length) % files.length];
    const nextFile = files[(currentIndex + 1) % files.length];

    useEffect(() => {
        if (files.length > 0){
            setshowNav(true);
        }
    },[files])

    const Navigation = () => {
        return (
            <>
                <IconButton onClick={handleBack}>
                    <ArrowBackIosNewIcon />
                </IconButton>
                <IconButton onClick={handleNext}>
                    <ArrowForwardIosIcon />
                </IconButton>
                <Delete/>
            </>
        );
    };

    return (

    <>
        {showNav && (
        <Stack>
            
            <Card sx={{ display: 'flex', height: '70vh', width:'100vh', alignItems: 'center', justifyContent: 'center', mb:'2rem' }}>
                <DrawableCanvas fillColor={""} strokeWidth={0} strokeColor={""} backgroundColor={""} backgroundImageURL={URL.createObjectURL(currentFile)} realtimeUpdateStreamlit={false} canvasWidth={0} canvasHeight={0} drawingMode={""} initialDrawing={undefined} displayToolbar={false} displayRadius={0}/>
            </Card>
            <div style = {{display:'flex', justifyContent:'space-between'}}>
            {prevFile && <img src={URL.createObjectURL(prevFile)} alt="previous" style={{ width: '100px', height: 'auto' }} />}
            {nextFile && <img src={URL.createObjectURL(nextFile)} alt="next" style={{ width: '100px', height: 'auto' }} />}
            </div>
            <Card sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
                <Navigation/>
            </Card>
        </Stack>

        )}
    </>
    );
};

export default ImageCarousel;