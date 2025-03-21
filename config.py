"""
Configuration settings for the Hurricane Tracker application.
"""

# Time range settings
DEFAULT_START_YEAR = 2004
DEFAULT_END_YEAR = 2023  # Will be updated dynamically based on data

# Model parameters
MODEL_PARAMS = {
    'input_size': 15,  # Set to match the checkpoint's input size
    'hidden_size': 256,  # Set to match the checkpoint's hidden size
    'num_layers': 4,  # Set to match the checkpoint's number of layers
    'dropout': 0.3,  # Keep the same dropout rate
    'learning_rate': 0.0005,  # Keep the same learning rate
    'batch_size': 32,
    'sequence_length': 10,  # Keep the same sequence length
    'num_epochs': 100,  # Keep the same number of epochs
}

# Feature configuration
FEATURES = [
    'latitude',
    'longitude',
    'max_wind',
    'min_pressure',
    'category'
]

# Visualization settings
MAP_SETTINGS = {
    'default_zoom': 5,
    'min_zoom': 3,
    'max_zoom': 10,
    'tile_layer': 'OpenStreetMap',
    'track_color': 'blue',  # Actual track color
    'prediction_color': 'red',  # More distinct prediction color
    'marker_colors': {
        'start': 'green',  # Using valid folium colors
        'end': 'red',
        'prediction': 'orange',  # Changed to orange for better contrast
        'actual': 'blue'
    }
}

# Testing framework settings
TEST_SETTINGS = {
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
    'min_sequence_length': 10,
    'max_sequence_length': 48,
    'prediction_horizon': 24
}

# Performance metrics
METRICS = [
    'mean_squared_error',
    'mean_absolute_error',
    'haversine_distance',
    'direction_accuracy'
] 