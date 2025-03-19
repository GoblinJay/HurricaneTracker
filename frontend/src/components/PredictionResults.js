import React from 'react';
import {
  Box,
  Typography,
  LinearProgress,
  Grid,
  Paper,
  List,
  ListItem,
  ListItemText,
} from '@mui/material';

const getCategoryColor = (category) => {
  const colors = {
    1: '#00ff00', // Tropical Depression
    2: '#ffff00', // Tropical Storm
    3: '#ffa500', // Category 1
    4: '#ff4500', // Category 2
    5: '#ff0000', // Category 3+
  };
  return colors[category] || '#808080';
};

const getCategoryName = (category) => {
  const names = {
    1: 'Tropical Depression',
    2: 'Tropical Storm',
    3: 'Category 1 Hurricane',
    4: 'Category 2 Hurricane',
    5: 'Category 3+ Hurricane',
  };
  return names[category] || 'Unknown';
};

const PredictionResults = ({ prediction }) => {
  const {
    category,
    category_probability,
    confidence_score,
    track_coordinates,
  } = prediction;

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Prediction Results
      </Typography>

      {/* Category Prediction */}
      <Paper elevation={2} sx={{ p: 2, mb: 2 }}>
        <Typography variant="subtitle1" gutterBottom>
          Predicted Category
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
          <Typography
            variant="h4"
            sx={{
              color: getCategoryColor(category),
              mr: 2,
            }}
          >
            {getCategoryName(category)}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            ({category_probability.toFixed(2)}% confidence)
          </Typography>
        </Box>
        <LinearProgress
          variant="determinate"
          value={category_probability * 100}
          sx={{
            height: 10,
            borderRadius: 5,
            backgroundColor: 'grey.700',
            '& .MuiLinearProgress-bar': {
              backgroundColor: getCategoryColor(category),
            },
          }}
        />
      </Paper>

      {/* Track Prediction */}
      {track_coordinates && (
        <Paper elevation={2} sx={{ p: 2 }}>
          <Typography variant="subtitle1" gutterBottom>
            Predicted Track
          </Typography>
          <List>
            {track_coordinates.map((coord, index) => (
              <ListItem key={index}>
                <ListItemText
                  primary={`Point ${index + 1}`}
                  secondary={`Lat: ${coord.latitude.toFixed(4)}°, Lon: ${coord.longitude.toFixed(4)}°`}
                />
              </ListItem>
            ))}
          </List>
        </Paper>
      )}

      {/* Confidence Score */}
      <Paper elevation={2} sx={{ p: 2, mt: 2 }}>
        <Typography variant="subtitle1" gutterBottom>
          Overall Confidence
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <Box sx={{ flexGrow: 1, mr: 2 }}>
            <LinearProgress
              variant="determinate"
              value={confidence_score * 100}
              sx={{
                height: 10,
                borderRadius: 5,
                backgroundColor: 'grey.700',
              }}
            />
          </Box>
          <Typography variant="body2" color="text.secondary">
            {(confidence_score * 100).toFixed(1)}%
          </Typography>
        </Box>
      </Paper>
    </Box>
  );
};

export default PredictionResults; 