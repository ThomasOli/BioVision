import LinearProgress from '@mui/material/LinearProgress';

const BorderLinearProgress: React.FC = () => (
  <LinearProgress
    sx={{
      height: 15,
      borderRadius: 5,
      '&.MuiLinearProgress-colorPrimary': {
        backgroundColor: '#EEEEEE',
      },
      '& .MuiLinearProgress-bar': {
        borderRadius: 5,
        backgroundColor: '#1a90ff',
      },
    }}
  />
);

export default BorderLinearProgress;