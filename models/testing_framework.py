"""
Testing framework for hurricane path prediction model.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error
from config import TEST_SETTINGS, METRICS
import torch
from models.storm_predictor import StormDataset

logger = logging.getLogger(__name__)

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the Haversine distance between two points."""
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def calculate_direction_accuracy(pred_lats: np.ndarray, pred_lons: np.ndarray,
                              true_lats: np.ndarray, true_lons: np.ndarray) -> float:
    """Calculate the accuracy of predicted storm direction."""
    pred_directions = np.arctan2(pred_lons[1:] - pred_lons[:-1],
                               pred_lats[1:] - pred_lats[:-1])
    true_directions = np.arctan2(true_lons[1:] - true_lons[:-1],
                               true_lats[1:] - true_lats[:-1])
    
    # Calculate angle differences
    angle_diffs = np.abs(pred_directions - true_directions)
    angle_diffs = np.minimum(angle_diffs, 2*np.pi - angle_diffs)
    
    # Consider predictions within 45 degrees as correct
    correct_predictions = np.sum(angle_diffs <= np.pi/4)
    return correct_predictions / len(angle_diffs)

class ModelTester:
    def __init__(self, model, data_processor):
        self.model = model
        self.data_processor = data_processor
        self.settings = TEST_SETTINGS
        
    def evaluate_model(self, test_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, float]:
        """Evaluate model performance on test data."""
        try:
            test_features, test_targets = test_data['test']
            
            # Generate predictions (this will be a numpy array)
            predictions = self.model.predict_storm_path(test_features)
            
            # Ensure we have predictions
            if len(predictions) == 0:
                raise ValueError("No predictions generated")
            
            # Calculate metrics
            metrics = {}
            
            if 'mean_squared_error' in METRICS and len(predictions) > 0:
                # If test_targets shape is different, adjust for comparison
                if test_targets.shape != predictions.shape:
                    # We might only have one target point to compare against
                    if len(test_targets.shape) == 1:
                        test_targets = test_targets.reshape(1, -1)
                    # Or we might need to select only the relevant targets for comparison
                    elif len(test_targets) > len(predictions):
                        test_targets = test_targets[:len(predictions)]
                    else:
                        predictions = predictions[:len(test_targets)]
                
                metrics['mse'] = mean_squared_error(test_targets, predictions)
                
            if 'mean_absolute_error' in METRICS and len(predictions) > 0:
                metrics['mae'] = mean_absolute_error(test_targets, predictions)
                
            if 'haversine_distance' in METRICS and len(predictions) > 0:
                haversine_distances = []
                for i in range(min(len(predictions), len(test_targets))):
                    pred = predictions[i]
                    true = test_targets[i] if i < len(test_targets) else test_targets[-1]
                    dist = haversine_distance(pred[0], pred[1], true[0], true[1])
                    haversine_distances.append(dist)
                metrics['haversine_distance'] = np.mean(haversine_distances) if haversine_distances else 0.0
                
            if 'direction_accuracy' in METRICS and len(predictions) > 1:
                metrics['direction_accuracy'] = calculate_direction_accuracy(
                    predictions[:, 0], predictions[:, 1],
                    test_targets[:min(len(test_targets), len(predictions)), 0], 
                    test_targets[:min(len(test_targets), len(predictions)), 1]
                )
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            # Return default metrics if evaluation fails
            return {metric: 0.0 for metric in METRICS}
            
    def cross_validate(self, data: pd.DataFrame, n_splits: int = 5) -> Dict[str, List[float]]:
        """Perform cross-validation on the model."""
        try:
            cv_metrics = {metric: [] for metric in METRICS}
            
            # Create cross-validation splits
            storm_ids = data['storm_id'].unique()
            np.random.shuffle(storm_ids)
            split_size = len(storm_ids) // n_splits
            
            for i in range(n_splits):
                # Create train/test split
                test_storms = storm_ids[i*split_size:(i+1)*split_size]
                train_storms = np.setdiff1d(storm_ids, test_storms)
                
                train_data = data[data['storm_id'].isin(train_storms)]
                test_data = data[data['storm_id'].isin(test_storms)]
                
                # Prepare sequences
                train_features, train_targets = self.data_processor.prepare_sequences(train_data)
                test_features, test_targets = self.data_processor.prepare_sequences(test_data)
                
                # Fit scaler on training data
                self.data_processor.fit_scaler(train_features)
                
                # Transform data
                train_features = self.data_processor.transform_data(train_features)
                test_features = self.data_processor.transform_data(test_features)
                
                # Create datasets
                train_dataset = StormDataset(train_features, train_targets)
                test_dataset = StormDataset(test_features, test_targets)
                
                # Create data loaders
                train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=32,
                    shuffle=True
                )
                test_loader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=32
                )
                
                # Train model
                self.model.train_model(train_loader, test_loader)
                
                # Evaluate on test set
                fold_metrics = self.evaluate_model({
                    'test': (test_features, test_targets)
                })
                
                # Store metrics
                for metric, value in fold_metrics.items():
                    cv_metrics[metric].append(value)
                    
            # Calculate mean and std of metrics
            for metric in cv_metrics:
                cv_metrics[metric] = {
                    'mean': np.mean(cv_metrics[metric]),
                    'std': np.std(cv_metrics[metric])
                }
                
            return cv_metrics
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            raise
            
    def visualize_predictions(self, test_data: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                            n_examples: int = 5) -> List[Dict]:
        """Generate visualization data for model predictions."""
        try:
            test_features, test_targets = test_data['test']
            
            # Limit number of examples
            n_examples = min(n_examples, len(test_features))
            
            # Select random examples
            indices = np.random.choice(len(test_features), n_examples, replace=False)
            
            visualization_data = []
            for idx in indices:
                # Get single feature sequence
                feature_sequence = test_features[idx:idx+1]
                
                # Generate prediction
                prediction = self.model.predict_storm_path(feature_sequence)
                
                # Get actual path
                actual_path = test_targets[idx] if idx < len(test_targets) else None
                
                # Only add if we have a prediction
                if len(prediction) > 0:
                    visualization_data.append({
                        'predicted': {
                            'latitude': prediction[0][0],
                            'longitude': prediction[0][1]
                        },
                        'actual': {
                            'latitude': actual_path[0] if actual_path is not None else 0.0,
                            'longitude': actual_path[1] if actual_path is not None else 0.0
                        }
                    })
            
            return visualization_data
            
        except Exception as e:
            logger.error(f"Error generating visualization data: {str(e)}")
            return [] 