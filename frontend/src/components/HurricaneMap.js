import React, { useEffect, useRef } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import { Box, Typography } from '@mui/material';

// Replace with your Mapbox access token
mapboxgl.accessToken = 'your_mapbox_token_here';

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

const HurricaneMap = ({ trackCoordinates, category }) => {
  const mapContainer = useRef(null);
  const map = useRef(null);
  const marker = useRef(null);

  useEffect(() => {
    if (!mapContainer.current) return;

    // Initialize map
    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/dark-v10',
      center: [-85.0, 25.0], // Center on Gulf of Mexico
      zoom: 4
    });

    // Add navigation controls
    map.current.addControl(new mapboxgl.NavigationControl(), 'top-right');

    // Cleanup
    return () => {
      if (map.current) {
        map.current.remove();
      }
    };
  }, []);

  useEffect(() => {
    if (!map.current || !trackCoordinates || trackCoordinates.length === 0) return;

    // Remove existing layers and markers
    if (map.current.getLayer('route')) {
      map.current.removeLayer('route');
    }
    if (map.current.getSource('route')) {
      map.current.removeSource('route');
    }
    if (marker.current) {
      marker.current.remove();
    }

    // Create route line
    const coordinates = trackCoordinates.map(coord => [coord.longitude, coord.latitude]);
    
    map.current.addSource('route', {
      type: 'geojson',
      data: {
        type: 'Feature',
        properties: {},
        geometry: {
          type: 'LineString',
          coordinates: coordinates
        }
      }
    });

    map.current.addLayer({
      id: 'route',
      type: 'line',
      source: 'route',
      layout: {
        'line-join': 'round',
        'line-cap': 'round'
      },
      paint: {
        'line-color': getCategoryColor(category),
        'line-width': 3
      }
    });

    // Add markers for each point
    coordinates.forEach((coord, index) => {
      const el = document.createElement('div');
      el.className = 'marker';
      el.style.backgroundColor = getCategoryColor(category);
      el.style.width = '10px';
      el.style.height = '10px';
      el.style.borderRadius = '50%';
      el.style.border = '2px solid white';

      new mapboxgl.Marker(el)
        .setLngLat(coord)
        .setPopup(new mapboxgl.Popup({ offset: 25 })
          .setHTML(`<h3>Point ${index + 1}</h3>
                    <p>Lat: ${coord[1].toFixed(4)}°</p>
                    <p>Lon: ${coord[0].toFixed(4)}°</p>`))
        .addTo(map.current);
    });

    // Fit bounds to show entire track
    const bounds = coordinates.reduce((bounds, coord) => {
      return bounds.extend(coord);
    }, new mapboxgl.LngLatBounds(coordinates[0], coordinates[0]));

    map.current.fitBounds(bounds, {
      padding: 50
    });
  }, [trackCoordinates, category]);

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Predicted Track
      </Typography>
      <Box
        ref={mapContainer}
        sx={{
          height: 400,
          width: '100%',
          borderRadius: 1,
          overflow: 'hidden'
        }}
      />
    </Box>
  );
};

export default HurricaneMap; 