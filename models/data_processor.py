"""
Data preprocessing and feature engineering for hurricane prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict
import logging
from config import FEATURES, MODEL_PARAMS

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, sequence_length=None):
        self.sequence_length = sequence_length if sequence_length is not None else MODEL_PARAMS['sequence_length']
        self.scaler = StandardScaler()
        # Define core base features that match what the model expects
        self.base_features = [
            'latitude', 'longitude', 'max_wind', 'min_pressure', 'category'
        ]
        # Optional extended features
        self.extended_features = [
            'distance_from_land', 'ocean_temperature', 'wind_pressure_ratio',
            'movement_speed', 'pressure_trend', 'wind_trend'
        ]
        # Extended features from config (might include engineered features)
        self.feature_columns = FEATURES if FEATURES else self.base_features.copy()
        
    def calculate_distance_from_land(self, lat: float, lon: float) -> float:
        """Calculate approximate distance from nearest land mass."""
        # Simplified calculation - can be enhanced with actual coastline data
        # This is a placeholder that should be replaced with actual coastline data
        return 0.0  # TODO: Implement actual coastline distance calculation
        
    def get_ocean_temperature(self, lat: float, lon: float, date: pd.Timestamp) -> float:
        """Get ocean temperature at given coordinates and date."""
        # Placeholder - should be replaced with actual ocean temperature data
        # Could use NOAA's OISST dataset or similar
        return 0.0  # TODO: Implement actual ocean temperature lookup
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features from base data."""
        try:
            # Create a copy to avoid modifying the original
            df = df.copy()
            
            # Extract time-based features
            df['hour'] = df['date'].dt.hour
            df['day'] = df['date'].dt.day
            df['month'] = df['date'].dt.month
            df['year'] = df['date'].dt.year
            
            # Calculate movement features
            df['movement_speed'] = df.groupby('storm_id').apply(
                lambda x: np.sqrt(x['latitude'].diff()**2 + x['longitude'].diff()**2)
            ).reset_index(level=0, drop=True)
            
            # Calculate trends
            df['pressure_trend'] = df.groupby('storm_id')['min_pressure'].diff()
            df['wind_trend'] = df.groupby('storm_id')['max_wind'].diff()
            
            # Calculate wind-pressure ratio
            df['wind_pressure_ratio'] = df['max_wind'] / df['min_pressure']
            
            # Calculate distance from land (simplified)
            df['distance_from_land'] = np.sqrt(
                (df['latitude'] - 25)**2 +  # Approximate US coastline
                (df['longitude'] + 80)**2
            )
            
            # Estimate ocean temperature based on latitude and month
            df['ocean_temperature'] = 30 - (df['latitude'] - 25)**2 / 100
            
            # Fill NaN values with forward fill then backward fill
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            logger.info(f"Engineered features: {df.columns.tolist()}")
            return df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise
            
    def prepare_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training."""
        try:
            # Group by storm_id to maintain storm paths
            sequences = []
            targets = []
            
            for storm_id, storm_data in df.groupby('storm_id'):
                # Sort by date to ensure proper sequence order
                storm_data = storm_data.sort_values('date')
                
                # Get features excluding date and storm_id
                feature_cols = [col for col in storm_data.columns 
                              if col not in ['date', 'storm_id']]
                
                # Convert to numpy array for easier manipulation
                data = storm_data[feature_cols].values
                
                # Create sequences
                for i in range(len(data) - self.sequence_length):
                    # Input sequence
                    seq = data[i:(i + self.sequence_length)]
                    # Target (next position and intensity)
                    target = data[i + self.sequence_length, :4]  # lat, lon, wind, pressure
                    
                    sequences.append(seq)
                    targets.append(target)
            
            # Convert to numpy arrays
            sequences = np.array(sequences)
            targets = np.array(targets)
            
            logger.info(f"Prepared sequences shape: {sequences.shape}")
            logger.info(f"Targets shape: {targets.shape}")
            
            return sequences, targets
            
        except Exception as e:
            logger.error(f"Error in sequence preparation: {str(e)}")
            raise
            
    def get_feature_subset(self, df: pd.DataFrame, features: List[str] = None) -> pd.DataFrame:
        """Get a subset of features from dataframe, adding zeros for missing ones."""
        if features is None:
            features = self.base_features
            
        # Create a copy of the dataframe with only the needed features
        result = pd.DataFrame()
        
        # Add each requested feature, using zeros for missing ones
        for feature in features:
            if feature in df.columns:
                result[feature] = df[feature]
            else:
                result[feature] = 0.0
                
        return result
        
    def split_data(self, features: np.ndarray, targets: np.ndarray, 
                   train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Split data into train, validation, and test sets."""
        try:
            # Handle empty datasets
            if len(features) == 0:
                logger.warning("No features to split, returning empty datasets")
                empty_features = np.array([])
                empty_targets = np.array([])
                return {
                    'train': (empty_features, empty_targets),
                    'val': (empty_features, empty_targets),
                    'test': (empty_features, empty_targets)
                }
                
            # Calculate split indices
            n_samples = len(features)
            train_size = max(1, int(n_samples * train_ratio))
            val_size = max(1, int(n_samples * val_ratio))
            
            # Ensure we don't exceed the dataset size
            if train_size + val_size > n_samples:
                val_size = max(1, n_samples - train_size)
            
            # Shuffle indices
            indices = np.random.permutation(n_samples)
            
            # Split data
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]
            
            return {
                'train': (features[train_indices], targets[train_indices]),
                'val': (features[val_indices], targets[val_indices]),
                'test': (features[test_indices], targets[test_indices])
            }
            
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise
            
    def fit_scaler(self, data: np.ndarray):
        """Fit the scaler on training data."""
        try:
            # Check if data is empty
            if data.size == 0:
                logger.warning("Empty dataset provided to fit_scaler, skipping")
                return
                
            # Reshape data to 2D if needed
            if len(data.shape) == 3:
                n_samples, seq_len, n_features = data.shape
                data_2d = data.reshape(-1, n_features)
            else:
                data_2d = data
                
            self.scaler.fit(data_2d)
            
        except Exception as e:
            logger.error(f"Error fitting scaler: {str(e)}")
            raise
            
    def transform_data(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler."""
        try:
            # Check if data is empty
            if data.size == 0:
                logger.warning("Empty dataset provided to transform_data, returning empty array")
                return data
                
            # Reshape data to 2D if needed
            if len(data.shape) == 3:
                n_samples, seq_len, n_features = data.shape
                data_2d = data.reshape(-1, n_features)
                transformed = self.scaler.transform(data_2d)
                return transformed.reshape(n_samples, seq_len, n_features)
            else:
                return self.scaler.transform(data)
                
        except Exception as e:
            logger.error(f"Error transforming data: {str(e)}")
            raise
            
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data."""
        try:
            return self.scaler.inverse_transform(data)
        except Exception as e:
            logger.error(f"Error in inverse transform: {str(e)}")
            raise 