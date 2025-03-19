import React, { useState } from 'react';
import { Container, Box, Typography, Paper, CircularProgress } from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import HurricaneMap from './components/HurricaneMap';
import ImageUpload from './components/ImageUpload';
import PredictionResults from './components/PredictionResults';
import './App.css';

// Create a dark theme
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#90caf9',
    },
    secondary: {
      main: '#f48fb1',
    },
  },
});

function App() {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handlePrediction = async (imageFile) => {
    setLoading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('file', imageFile);
      formData.append('predict_track', true);

      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Prediction failed');
      }

      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Container maxWidth="lg">
        <Box sx={{ my: 4 }}>
          <Typography variant="h3" component="h1" gutterBottom align="center">
            Hurricane Prediction System
          </Typography>
          
          <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
            <ImageUpload onImageUpload={handlePrediction} />
          </Paper>

          {loading && (
            <Box display="flex" justifyContent="center" my={4}>
              <CircularProgress />
            </Box>
          )}

          {error && (
            <Paper elevation={3} sx={{ p: 2, mb: 3, bgcolor: 'error.dark' }}>
              <Typography color="error">{error}</Typography>
            </Paper>
          )}

          {prediction && (
            <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 3 }}>
              <Paper elevation={3} sx={{ p: 3 }}>
                <PredictionResults prediction={prediction} />
              </Paper>
              
              <Paper elevation={3} sx={{ p: 3 }}>
                <HurricaneMap 
                  trackCoordinates={prediction.track_coordinates}
                  category={prediction.category}
                />
              </Paper>
            </Box>
          )}
        </Box>
      </Container>
    </ThemeProvider>
  );
}

export default App; 